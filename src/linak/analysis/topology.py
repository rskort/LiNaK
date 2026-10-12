"""Exact O/H membership, shared by frames and optionally persisted in trajectories.

No geometry-derived centers or distances are cached. Providers are owned by the
loaded frames/sequence; the registry holds only weak references and no file handles.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import TYPE_CHECKING
from types import MappingProxyType
from weakref import ReferenceType, WeakValueDictionary, ref

from ase import Atoms
import h5py
import numpy as np

from ..progress import ProgressBar, progress_stage
from ..storage.policies import dataset_create_options_for_shape
from ..temporary_resources import current_owner, process_alive

if TYPE_CHECKING:
    from .water import OHTopology

LOGGER = logging.getLogger(__name__)
VERSION = 1
CLASSIFIER_VERSION = "oh-1"
FRAME_KEY = "_linak_oh_topology"
WIDTHS = {"mol:H": 1, "mol:O": 1, "mol:OH": 2, "mol:H2O": 3, "mol:H3O": 4}
DIAGNOSTICS = (
    "ambiguous_hydrogen_indices",
    "ambiguous_oxygen_indices",
    "overcoordinated_oxygen_indices",
)
_PROVIDERS: WeakValueDictionary[tuple[str, tuple[int, int]], TopologyProvider] = (
    WeakValueDictionary()
)


def fingerprint(frame: Atoms) -> bytes:
    """Fingerprint actual geometry, including changes smaller than float32 precision."""
    digest = hashlib.sha256()
    for values, dtype in (
        (frame.numbers, "<i8"),
        (frame.positions, "<f8"),
        (frame.cell.array, "<f8"),
        (frame.pbc, "u1"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.hexdigest().encode("ascii")


def profile_key(cutoff: float, backend: str) -> str:
    return hashlib.sha256(
        f"{CLASSIFIER_VERSION}|{float(cutoff).hex()}|{backend}".encode()
    ).hexdigest()[:24]


def _immutable(topology: OHTopology) -> OHTopology:
    from .water import OHTopology

    for values in (*topology.groups.values(), *(getattr(topology, name) for name in DIAGNOSTICS)):
        values.flags.writeable = False
    return OHTopology(
        MappingProxyType(dict(topology.groups)),
        topology.ambiguous_hydrogen_indices,
        topology.ambiguous_oxygen_indices,
        topology.overcoordinated_oxygen_indices,
    )


def _arrays(topology: OHTopology) -> dict[str, np.ndarray]:
    return {**topology.groups, **{name: getattr(topology, name) for name in DIAGNOSTICS}}


class TopologyProfile:
    """Deduplicated immutable states with one state and fingerprint per source frame."""

    def __init__(self, cutoff: float, backend: str) -> None:
        self.cutoff = cutoff
        self.backend = backend
        self.states: list[OHTopology] = []
        self.frame_state_index: list[int] = []
        self.fingerprints: list[bytes] = []
        self._state_keys: dict[tuple[bytes, ...], int] = {}

    def append(self, frame: Atoms, topology: OHTopology) -> None:
        arrays = _arrays(topology)
        key = tuple(np.asarray(values, dtype="<i8").tobytes() for values in arrays.values())
        state = self._state_keys.get(key)
        if state is None:
            state = len(self.states)
            self.states.append(_immutable(topology))
            self._state_keys[key] = state
        self.frame_state_index.append(state)
        self.fingerprints.append(fingerprint(frame))

    def select(self, frame: Atoms, index: int, geometry: bytes) -> OHTopology | None:
        if not 0 <= index < len(self.fingerprints) or self.fingerprints[index] != geometry:
            return None
        topology = self.states[self.frame_state_index[index]]
        if any(
            values.size and (values.min() < 0 or values.max() >= len(frame))
            for values in _arrays(topology).values()
        ):
            return None
        return topology


class FrameTopology:
    def __init__(
        self,
        provider: TopologyProvider | None = None,
        index: int = 0,
        source_geometry: bytes | None = None,
    ) -> None:
        self.provider = provider
        self.index = index
        self.source_geometry = source_geometry
        # One current geometry per profile; copied Atoms share this record safely.
        self.local: dict[str, tuple[bytes, OHTopology]] = {}

    def __deepcopy__(self, memo: dict) -> FrameTopology:
        return self


def resolve_topology(
    frame: Atoms, *, cutoff: float, backend: str, build_source: bool = True
) -> OHTopology:
    from .water import _normalize_topology_backend, _oh_molecule_topology_with_backend

    backend = _normalize_topology_backend(backend)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("O/H cutoff must be a finite positive value.")
    key = profile_key(cutoff, backend)
    record = frame.info.get(FRAME_KEY)
    if not isinstance(record, FrameTopology):
        record = FrameTopology()
        frame.info[FRAME_KEY] = record
    geometry = fingerprint(frame)
    cached = record.local.get(key)
    if cached is not None and cached[0] == geometry:
        return cached[1]
    topology = None
    if record.provider is not None:
        runtime = record.provider.runtime.get((record.index, key))
        if runtime is not None and runtime[0] == geometry:
            topology = runtime[1]
    if topology is None and record.provider is not None and geometry == record.source_geometry:
        topology = record.provider.select(
            frame, record.index, geometry, cutoff, backend, build_missing=build_source
        )
    if topology is None:
        topology, _ = _oh_molecule_topology_with_backend(frame, oh_cutoff=cutoff, backend=backend)
        topology = _immutable(topology)
        if record.provider is not None:
            record.provider.runtime[(record.index, key)] = (geometry, topology)
    record.local[key] = (geometry, topology)
    return topology


def build_profile(
    frames: Iterable[Atoms],
    *,
    cutoff: float = 1.27,
    backend: str = "auto",
    count: int | None = None,
    storage_precision: bool = False,
) -> TopologyProfile:
    profile = TopologyProfile(cutoff, backend)
    with progress_stage(LOGGER, "Build molecular topology"):
        with ProgressBar(desc="Molecular topology", total=count, unit="frame") as progress:
            for frame in frames:
                working = frame
                if storage_precision:
                    working = frame.copy()
                    working.positions[:] = np.asarray(frame.positions, dtype=np.float32)
                topology = resolve_topology(
                    working, cutoff=cutoff, backend=backend, build_source=False
                )
                profile.append(working, topology)
                progress.update()
    return profile


def prepare_topology(frames: Sequence[Atoms]) -> TopologyProfile:
    """Build at HDF5 storage precision and share memberships with surface analysis."""
    profile = build_profile(frames, count=len(frames), storage_precision=True)
    for index, frame in enumerate(frames):
        # Quantization must precede surface populations too, to use exactly what is stored.
        frame.positions[:] = np.asarray(frame.positions, dtype=np.float32)
        record = frame.info.get(FRAME_KEY)
        if not isinstance(record, FrameTopology):
            record = FrameTopology()
            frame.info[FRAME_KEY] = record
        record.local[profile_key(1.27, "auto")] = (
            profile.fingerprints[index],
            profile.states[profile.frame_state_index[index]],
        )
    return profile


def write_profile(handle: h5py.File, profile: TopologyProfile, compression: str = "fast") -> None:
    root = handle.require_group("molecular_topology").require_group("oh")
    key = profile_key(profile.cutoff, profile.backend)
    if key in root:
        del root[key]
    group = root.create_group(key)
    group.attrs.update(
        {
            "version": VERSION,
            "classifier_version": CLASSIFIER_VERSION,
            "cutoff_angstrom": profile.cutoff,
            "requested_backend": profile.backend,
            "policy": "exact-per-frame",
            "frame_count": len(profile.fingerprints),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
        }
    )

    def dataset(name: str, shape: tuple[int, ...], dtype: object) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            **dataset_create_options_for_shape(shape, dtype=dtype, compression_profile=compression),
        )

    count = len(profile.fingerprints)
    states = dataset("frame_state_index", (count,), np.int64)
    fingerprints = dataset("geometry_fingerprint", (count,), "S64")
    for start in range(0, count, 1024):
        stop = min(start + 1024, count)
        states[start:stop] = profile.frame_state_index[start:stop]
        fingerprints[start:stop] = profile.fingerprints[start:stop]
    for name in (*WIDTHS, *DIAGNOSTICS):
        arrays = [_arrays(topology)[name] for topology in profile.states]
        offsets = np.concatenate(([0], np.cumsum([len(values) for values in arrays])))
        group.create_dataset(name + "_offsets", data=offsets.astype(np.int64))
        shape = (int(offsets[-1]), WIDTHS[name]) if name in WIDTHS else (int(offsets[-1]),)
        values_dataset = dataset(name, shape, np.int64)
        for state, values in enumerate(arrays):
            values_dataset[int(offsets[state]) : int(offsets[state + 1])] = values
    group.attrs["complete"] = True


def read_profiles(handle: h5py.File, frame_count: int) -> dict[str, TopologyProfile]:
    from .water import OHTopology

    profiles: dict[str, TopologyProfile] = {}
    root = handle.get("molecular_topology/oh")
    if not isinstance(root, h5py.Group):
        return profiles
    for key, group in root.items():
        try:
            if (
                not group.attrs.get("complete", False)
                or group.attrs["version"] != VERSION
                or group.attrs["classifier_version"] != CLASSIFIER_VERSION
                or group.attrs["policy"] != "exact-per-frame"
                or group.attrs["frame_count"] != frame_count
            ):
                continue
            cutoff = float(group.attrs["cutoff_angstrom"])
            backend = str(group.attrs["requested_backend"])
            if backend not in ("auto", "ase", "ckdtree") or not np.isfinite(cutoff) or cutoff <= 0:
                continue
            if key != profile_key(cutoff, backend):
                continue
            profile = TopologyProfile(cutoff, backend)
            if group["frame_state_index"].dtype.kind not in "iu":
                raise ValueError("Invalid topology state dtype")
            indices = np.asarray(group["frame_state_index"], dtype=np.int64)
            prints = np.asarray(group["geometry_fingerprint"])
            if indices.shape != (frame_count,) or prints.shape != (frame_count,):
                raise ValueError("Invalid frame map")
            offsets = {
                name: np.asarray(group[name + "_offsets"], dtype=np.int64)
                for name in (*WIDTHS, *DIAGNOSTICS)
            }
            state_count = len(offsets["mol:H"]) - 1
            if state_count <= 0 or np.any(indices < 0) or np.any(indices >= state_count):
                raise ValueError("Invalid topology state index")
            for name, offset in offsets.items():
                dataset = group[name]
                shape = (dataset.shape[0], WIDTHS[name]) if name in WIDTHS else (dataset.shape[0],)
                if (
                    dataset.dtype.kind not in "iu"
                    or group[name + "_offsets"].dtype.kind not in "iu"
                    or dataset.shape != shape
                    or len(offset) != state_count + 1
                    or offset[0] != 0
                    or np.any(np.diff(offset) < 0)
                    or offset[-1] != dataset.shape[0]
                ):
                    raise ValueError("Invalid topology state offsets")
            for state in range(state_count):
                arrays = {
                    name: np.asarray(group[name][offset[state] : offset[state + 1]], dtype=int)
                    for name, offset in offsets.items()
                }
                topology = OHTopology(
                    groups={name: arrays[name] for name in WIDTHS},
                    **{name: arrays[name] for name in DIAGNOSTICS},
                )
                profile.states.append(_immutable(topology))
            profile.frame_state_index = indices.tolist()
            profile.fingerprints = prints.tolist()
            profiles[key] = profile
        except (KeyError, ValueError, TypeError, IndexError, OSError):
            LOGGER.warning("Ignoring invalid molecular topology profile '%s'.", key)
    return profiles


def _source_stamp(path: Path) -> tuple[int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns


class TopologyProvider:
    """One source's profiles; first use upgrades old files after readers close."""

    def __init__(self, path: Path, handle: h5py.File, count: int) -> None:
        self.path = path
        self.stamp = _source_stamp(path)
        self.count = count
        self.profiles = read_profiles(handle, count)
        self.frames: list[ReferenceType[Atoms]] | None = None
        self.active_readers = 0
        self.pending: dict[str, TopologyProfile] = {}
        self.warned = False
        self._building = False
        self._logged: set[str] = set()
        # Transformed streaming frames reuse assignments without changing source profiles.
        self.runtime: dict[tuple[int, str], tuple[bytes, OHTopology]] = {}

    def remember_frames(self, frames: Sequence[Atoms]) -> None:
        self.frames = [ref(frame) for frame in frames]

    def select(
        self,
        frame: Atoms,
        index: int,
        geometry: bytes,
        cutoff: float,
        backend: str,
        *,
        build_missing: bool = True,
    ) -> OHTopology | None:
        key = profile_key(cutoff, backend)
        profile = self.profiles.get(key)
        if profile is not None:
            topology = profile.select(frame, index, geometry)
            if topology is not None:
                if key not in self._logged:
                    LOGGER.info("Reusing exact molecular topology for %d frames.", self.count)
                    self._logged.add(key)
                return topology
        # Changed/cropped/overridden frames must never upgrade the source from their geometry.
        if not build_missing or self._building:
            return None
        try:
            if _source_stamp(self.path) != self.stamp:
                return None
        except OSError:
            return None
        self._building = True
        try:
            originals = None if self.frames is None else [reference() for reference in self.frames]
            if originals is not None and all(
                original is not None
                and isinstance(original.info.get(FRAME_KEY), FrameTopology)
                and fingerprint(original) == original.info[FRAME_KEY].source_geometry
                for original in originals
            ):
                frames: Generator[Atoms, None, None] = (
                    original for original in originals if original is not None
                )
            else:
                from ..trajectory.io import _read_trajectory_hdf5_chunks

                frames = (
                    item
                    for chunk in _read_trajectory_hdf5_chunks(
                        self.path, chunk_size=256, _attach_topology=False
                    )
                    for item in chunk
                )
            # The low-level detector avoids recursively resolving through this provider.
            from .water import _oh_molecule_topology_with_backend

            profile = TopologyProfile(cutoff, backend)
            with closing(frames), progress_stage(LOGGER, "Build molecular topology"):
                with ProgressBar(desc="Molecular topology", total=self.count, unit="frame") as bar:
                    for original in frames:
                        topology, _ = _oh_molecule_topology_with_backend(
                            original, oh_cutoff=cutoff, backend=backend
                        )
                        profile.append(original, topology)
                        bar.update()
            self.profiles[key] = profile
            self.pending[key] = profile
        finally:
            self._building = False
        self.flush()
        return profile.select(frame, index, geometry)

    def _warning(self, error: object) -> None:
        if not self.warned:
            LOGGER.warning(
                "Cannot save molecular topology in '%s': %s. Reusing it in memory.",
                self.path,
                error,
            )
            self.warned = True

    def flush(self) -> None:
        if not self.pending or self.active_readers:
            return
        lock = self.path.with_name(self.path.name + ".topology.lock")
        temporary: Path | None = None
        owned = False
        try:
            if not self.path.stat().st_mode & stat.S_IWRITE:
                raise PermissionError("trajectory is read-only")
            if lock.exists():
                try:
                    marker = json.loads(lock.read_text(encoding="utf-8"))
                    if marker.get("owner") == current_owner() and not process_alive(
                        int(marker["pid"])
                    ):
                        lock.unlink()
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            with lock.open("x", encoding="utf-8") as stream:
                owned = True
                json.dump({"owner": current_owner(), "pid": os.getpid()}, stream)
            if _source_stamp(self.path) != self.stamp:
                raise OSError("source changed during topology computation")
            fd, name = tempfile.mkstemp(
                prefix=".linak-topology-", suffix=".tmp", dir=self.path.parent
            )
            os.close(fd)
            temporary = Path(name)
            shutil.copyfile(self.path, temporary)
            with h5py.File(temporary, "r+") as handle:
                compression = str(handle.attrs.get("compression_profile", "fast"))
                for profile in self.pending.values():
                    write_profile(handle, profile, compression)
            if _source_stamp(self.path) != self.stamp:
                raise OSError("source changed during topology publication")
            os.replace(temporary, self.path)
            temporary = None
            self.stamp = _source_stamp(self.path)
            _PROVIDERS[(str(self.path), self.stamp)] = self
            LOGGER.info("Saved exact molecular topology in '%s'.", self.path)
        except (OSError, ValueError) as error:
            self._warning(error)
        finally:
            self.pending.clear()
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    self._warning(error)
            if owned:
                try:
                    lock.unlink(missing_ok=True)
                except OSError as error:
                    self._warning(error)


def source_provider(path: Path, handle: h5py.File, count: int) -> TopologyProvider:
    path = path.resolve()
    key = (str(path), _source_stamp(path))
    provider = _PROVIDERS.get(key)
    if provider is None:
        provider = TopologyProvider(path, handle, count)
        _PROVIDERS[key] = provider
    return provider


def attach_frames(frames: Sequence[Atoms], provider: TopologyProvider, start: int = 0) -> None:
    for index, frame in enumerate(frames, start):
        frame.info[FRAME_KEY] = FrameTopology(provider, index, fingerprint(frame))
