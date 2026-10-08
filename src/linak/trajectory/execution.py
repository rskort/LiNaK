"""Bounded-memory CLI trajectory execution and temporary spooling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import atexit
from bisect import bisect_right
from itertools import accumulate
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Literal, overload

from ase import Atoms
import h5py
import numpy as np

from ..temporary_resources import cleanup_stale_directories, current_owner, remove_owned_directory

from . import io as trajectory_io

TrajectoryExecution = Literal["auto", "eager", "stream"]
MIB = 1024 * 1024
_SPOOL_PREFIX = "linak-spool-"
_MARKER_NAME = ".linak-spool.json"
_STALE_AGE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class TrajectoryExecutionOptions:
    # Existing analyses traverse frames repeatedly; streaming must be opt-in
    # until their metadata/normalization passes no longer reread the payload.
    mode: TrajectoryExecution = "eager"
    stream_threshold_mib: float = 256.0
    chunk_target_mib: float = 64.0
    temporary_directory: str | Path | None = None


@dataclass(frozen=True)
class TrajectoryExecutionDiagnostics:
    resolved_mode: str
    estimated_materialized_bytes: int
    chunk_frames: int
    spool_directory: str | None = None


class ChunkedTrajectorySequence(Sequence[Atoms]):
    """Seekable multi-pass trajectory sequence that retains at most one chunk."""

    def __init__(self, paths: Sequence[Path], counts: Sequence[int], *, chunk_frames: int) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.counts = tuple(int(value) for value in counts)
        if len(self.paths) != len(self.counts) or any(count < 0 for count in self.counts):
            raise ValueError("Trajectory paths and nonnegative frame counts must align.")
        self._ends = tuple(accumulate(self.counts))
        self._cached_key: tuple[int, int] | None = None
        self._cached_frames: list[Atoms] = []
        self.chunk_frames = max(1, int(chunk_frames))
        self._length = sum(self.counts)
        self._cell_override: np.ndarray | None = None

    def set_cell_override(self, cell: Any) -> None:
        matrix = np.asarray(cell, dtype=float)
        if matrix.shape == (3,):
            matrix = np.diag(matrix)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Trajectory cell override must be a finite 3-vector or 3x3 matrix.")
        self._cell_override = matrix.copy()

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Atoms]:
        for path in self.paths:
            for chunk in trajectory_io.read_trajectory_chunks(path, chunk_size=self.chunk_frames):
                if self._cell_override is not None:
                    for frame in chunk:
                        frame.set_cell(self._cell_override)
                        frame.set_pbc((True, True, True))
                yield from chunk

    @overload
    def __getitem__(self, index: int) -> Atoms: ...

    @overload
    def __getitem__(self, index: slice) -> list[Atoms]: ...

    def __getitem__(self, index: int | slice) -> Atoms | list[Atoms]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self[frame_index] for frame_index in range(start, stop, step)]
        resolved = int(index)
        if resolved < 0:
            resolved += self._length
        if resolved < 0 or resolved >= self._length:
            raise IndexError("trajectory frame index out of range")
        file_index = bisect_right(self._ends, resolved)
        local_index = resolved - (self._ends[file_index - 1] if file_index else 0)
        start = (local_index // self.chunk_frames) * self.chunk_frames
        key = (file_index, start)
        if self._cached_key != key:
            from ..out_h5 import is_linak_out_hdf5, read_out_h5_trajectory_chunks

            path = self.paths[file_index]
            reader = (
                read_out_h5_trajectory_chunks
                if is_linak_out_hdf5(path)
                else trajectory_io._read_trajectory_hdf5_chunks
            )
            chunks = reader(
                path,
                chunk_size=self.chunk_frames,
                start_frame=start,
                stop_frame=min(start + self.chunk_frames, self.counts[file_index]),
            )
            try:
                self._cached_frames = next(chunks)
            finally:
                chunks.close()
            self._cached_key = key
        frame = self._cached_frames[local_index - start].copy()
        if self._cell_override is not None:
            frame.set_cell(self._cell_override)
            frame.set_pbc((True, True, True))
        return frame


class ManagedTrajectorySequence(Sequence[Atoms]):
    """Sequence wrapper that releases an optional spool when it leaves scope."""

    def __init__(self, inner: Sequence[Atoms], closer: Any) -> None:
        self.inner = inner
        self._closer = closer

    def __len__(self) -> int:
        return len(self.inner)

    def __iter__(self) -> Iterator[Atoms]:
        return iter(self.inner)

    def __getitem__(self, index: int | slice) -> Atoms | list[Atoms]:
        return self.inner[index]

    def close(self) -> None:
        closer, self._closer = self._closer, None
        if closer is not None:
            closer()

    def set_cell_override(self, cell: Any) -> None:
        setter = getattr(self.inner, "set_cell_override", None)
        if callable(setter):
            setter(cell)
            return
        for frame in self.inner:
            frame.set_cell(cell)
            frame.set_pbc((True, True, True))

    def __del__(self) -> None:
        self.close()


def _hdf5_frame_shape(path: Path) -> tuple[int, int] | None:
    try:
        with h5py.File(path, "r") as handle:
            positions = handle.get("positions")
            if isinstance(positions, h5py.Dataset) and positions.ndim == 3:
                return int(positions.shape[0]), int(positions.shape[1])
            trajectory = handle.get("trajectory")
            if isinstance(trajectory, h5py.Group):
                positions = trajectory.get("positions")
                if isinstance(positions, h5py.Dataset) and positions.ndim == 3:
                    return int(positions.shape[0]), int(positions.shape[1])
    except OSError:
        return None
    return None


def estimate_materialized_trajectory_bytes(path: str | Path) -> int:
    """Estimate eager Atoms/array materialization before loading trajectory payloads."""

    source = Path(path).expanduser().resolve()
    shape = _hdf5_frame_shape(source)
    if shape is not None:
        frames, atoms = shape
        return max(1, frames * (atoms * 64 + 512))
    try:
        return max(1, int(source.stat().st_size * 4.0))
    except OSError:
        return 1


def resolve_chunk_frames(*, bytes_per_frame: int, chunk_target_mib: float) -> int:
    target = max(1, int(float(chunk_target_mib) * MIB))
    return max(16, min(1024, target // max(1, int(bytes_per_frame))))


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def cleanup_stale_spools(parent: Path) -> int:
    """Remove only validated stale directories whose owner process is dead."""
    return cleanup_stale_directories(
        parent,
        prefix=_SPOOL_PREFIX,
        marker_name=_MARKER_NAME,
        timestamp_key="start_time",
        stale_seconds=_STALE_AGE_SECONDS,
    )


def _temporary_parent(value: str | Path | None) -> Path:
    configured = value or os.getenv("LINAK_TEMP_DIR")
    parent = Path(configured).expanduser().resolve() if configured else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _preflight_spool_space(parent: Path, estimated_bytes: int) -> None:
    free = int(shutil.disk_usage(parent).free)
    required = int(max(1, estimated_bytes) * 1.2)
    if free < required:
        raise ValueError(
            f"Insufficient temporary disk space in '{parent}': need about "
            f"{required / MIB:.1f} MiB including reserve, available {free / MIB:.1f} MiB."
        )


def _spool_text_trajectory(
    source: Path,
    spool_dir: Path,
    *,
    chunk_frames: int,
    atom_aliases: Any,
) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    counts: list[int] = []
    for chunk_index, chunk in enumerate(
        trajectory_io.read_trajectory_chunks(
            source, chunk_size=chunk_frames, atom_aliases=atom_aliases
        )
    ):
        chunk_path = spool_dir / f"chunk-{chunk_index:06d}.traj.h5"
        trajectory_io.write_trajectory(
            chunk,
            chunk_path,
            source_path=source,
            source_format=source.suffix.lower().lstrip("."),
            compression_profile="fast",
        )
        paths.append(chunk_path)
        counts.append(len(chunk))
    if not paths:
        raise ValueError(f"No frames were read from: {source}")
    return paths, counts


def _read_eager(source: Path, atom_aliases: Any) -> list[Atoms]:
    if atom_aliases:
        return trajectory_io.read_trajectory(source, atom_aliases=atom_aliases)
    return trajectory_io.read_trajectory(source)


@contextmanager
def prepare_cli_trajectory(
    source: str | Path,
    *,
    options: TrajectoryExecutionOptions,
    atom_aliases: Any = None,
) -> Iterator[tuple[Sequence[Atoms], TrajectoryExecutionDiagnostics]]:
    """Yield an eager list or seekable bounded-memory sequence for CLI analyses."""

    source_path = Path(source).expanduser().resolve()
    mode = str(options.mode).strip().lower()
    if mode not in {"auto", "eager", "stream"}:
        raise ValueError("trajectory_execution must be one of: auto, eager, stream")
    estimated = estimate_materialized_trajectory_bytes(source_path)
    resolved = (
        "stream"
        if mode == "stream"
        or (mode == "auto" and estimated >= float(options.stream_threshold_mib) * MIB)
        else "eager"
    )
    if resolved == "eager":
        frames = _read_eager(source_path, atom_aliases)
        per_frame = estimated // max(1, len(frames))
        chunk_frames = resolve_chunk_frames(
            bytes_per_frame=per_frame,
            chunk_target_mib=options.chunk_target_mib,
        )
        yield frames, TrajectoryExecutionDiagnostics("eager", estimated, chunk_frames)
        return

    shape = _hdf5_frame_shape(source_path)
    if shape is not None:
        frame_count, atom_count = shape
        chunk_frames = resolve_chunk_frames(
            bytes_per_frame=max(1, atom_count * 64 + 512),
            chunk_target_mib=options.chunk_target_mib,
        )
        yield (
            ChunkedTrajectorySequence([source_path], [frame_count], chunk_frames=chunk_frames),
            TrajectoryExecutionDiagnostics("stream", estimated, chunk_frames),
        )
        return

    parent = _temporary_parent(options.temporary_directory)
    cleanup_stale_spools(parent)
    _preflight_spool_space(parent, estimated)
    spool_dir = Path(tempfile.mkdtemp(prefix=_SPOOL_PREFIX, dir=parent))
    marker = {
        "pid": os.getpid(),
        "uid": getattr(os, "getuid", lambda: None)(),
        "owner": current_owner(),
        "start_time": time.time(),
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": _source_fingerprint(source_path),
    }
    (spool_dir / _MARKER_NAME).write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    cleanup_registered = True

    def _cleanup() -> None:
        nonlocal cleanup_registered
        if cleanup_registered and spool_dir.exists() and (spool_dir / _MARKER_NAME).exists():
            remove_owned_directory(
                spool_dir, parent=parent, prefix=_SPOOL_PREFIX, marker_name=_MARKER_NAME
            )
        cleanup_registered = False

    atexit.register(_cleanup)
    try:
        bytes_per_frame = max(1, estimated // 1024)
        chunk_frames = resolve_chunk_frames(
            bytes_per_frame=bytes_per_frame,
            chunk_target_mib=options.chunk_target_mib,
        )
        paths, counts = _spool_text_trajectory(
            source_path,
            spool_dir,
            chunk_frames=chunk_frames,
            atom_aliases=atom_aliases,
        )
        sequence = ChunkedTrajectorySequence(paths, counts, chunk_frames=chunk_frames)
        yield (
            sequence,
            TrajectoryExecutionDiagnostics("stream", estimated, chunk_frames, str(spool_dir)),
        )
    finally:
        _cleanup()


def open_cli_trajectory(
    source: str | Path,
    *,
    options: TrajectoryExecutionOptions,
    atom_aliases: Any = None,
) -> tuple[ManagedTrajectorySequence, TrajectoryExecutionDiagnostics]:
    """Open a managed sequence for command handlers that cannot use a ``with`` block."""

    manager = prepare_cli_trajectory(source, options=options, atom_aliases=atom_aliases)
    sequence, diagnostics = manager.__enter__()
    managed = ManagedTrajectorySequence(
        sequence,
        lambda: manager.__exit__(None, None, None),
    )
    return managed, diagnostics


def spool_trajectory_chunks(
    chunks: Iterable[list[Atoms]],
    *,
    temporary_directory: str | Path | None,
    chunk_frames: int,
    source_fingerprint: str,
    estimated_bytes: int | None = None,
) -> ManagedTrajectorySequence:
    """Persist transformed chunks into one managed, seekable temporary sequence."""

    parent = _temporary_parent(temporary_directory)
    cleanup_stale_spools(parent)
    if estimated_bytes is not None:
        _preflight_spool_space(parent, int(estimated_bytes))
    spool_dir = Path(tempfile.mkdtemp(prefix=_SPOOL_PREFIX, dir=parent))
    marker = {
        "pid": os.getpid(),
        "uid": getattr(os, "getuid", lambda: None)(),
        "owner": current_owner(),
        "start_time": time.time(),
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": str(source_fingerprint),
    }
    (spool_dir / _MARKER_NAME).write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    paths: list[Path] = []
    counts: list[int] = []
    cleanup_registered = True

    def _cleanup() -> None:
        nonlocal cleanup_registered
        if cleanup_registered and spool_dir.exists() and (spool_dir / _MARKER_NAME).exists():
            remove_owned_directory(
                spool_dir, parent=parent, prefix=_SPOOL_PREFIX, marker_name=_MARKER_NAME
            )
        cleanup_registered = False

    atexit.register(_cleanup)
    try:
        for chunk_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            chunk_path = spool_dir / f"chunk-{chunk_index:06d}.traj.h5"
            trajectory_io.write_trajectory(
                chunk,
                chunk_path,
                source_format="spatial-filter",
                compression_profile="fast",
            )
            paths.append(chunk_path)
            counts.append(len(chunk))
        if not paths:
            raise ValueError("Spatial filtering produced no trajectory frames.")
    except Exception:
        _cleanup()
        raise
    return ManagedTrajectorySequence(
        ChunkedTrajectorySequence(paths, counts, chunk_frames=max(1, int(chunk_frames))),
        _cleanup,
    )


__all__ = (
    "ChunkedTrajectorySequence",
    "ManagedTrajectorySequence",
    "TrajectoryExecutionDiagnostics",
    "TrajectoryExecutionOptions",
    "cleanup_stale_spools",
    "estimate_materialized_trajectory_bytes",
    "open_cli_trajectory",
    "prepare_cli_trajectory",
    "resolve_chunk_frames",
    "spool_trajectory_chunks",
)
