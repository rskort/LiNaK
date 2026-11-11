"""Electrode-potential analysis from CP2K Hartree-potential cube files."""

from __future__ import annotations

from collections.abc import Sequence
from ..parallel import bounded_map, current_execution, stage_workers, timed_stage
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from types import TracebackType
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from ase import Atoms

from .. import __version__
from ..cube_io import CubeDataset, read_cube_sources, validate_cube_source
from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.potential_mapping import (
    resolve_potential_plot_mapping,
)
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
)
from ..storage.hdf5_utils import (
    _read_required_linak_header,
    LINAK_HDF5_FORMAT,
    LINAK_HDF5_VERSION,
    hdf5_string_dtype,
    require_h5py,
    resolve_hdf5_output_path,
)
from ..storage.policies import STORAGE_PRECISION_POLICY, normalize_compression_profile
from ..progress import ProgressBar
from ..runtime_resources import MIB
from .potential_reference import (
    PotentialConfig, build_reference_plan, evaluate_reference, reference_geometry,
    sensitivity_windows, validate_config, serialize_config,
)

if TYPE_CHECKING:
    import h5py

LOGGER = logging.getLogger(__name__)
_POTENTIAL_SCHEMA_VERSION = 3

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_EV = 27.211386245988


FERMI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"Fermi\s+energy\s*\[(?:a\.u\.|hartree)\]\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", re.I),
        "au",
    ),
    (
        re.compile(r"Fermi\s+energy\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"),
        "au",
    ),
    (
        re.compile(r"Fermi\s+energy\s*\[eV\]\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"),
        "ev",
    ),
    (
        re.compile(
            r"E\(Fermi\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*(?:a\.u\.|au|A\.U\.)"
        ),
        "au",
    ),
]

_NUMERIC_COLUMNS = {
    "efermi_ev", "water_bulk_potential_ev", "electrode_cshe_ev", "requested_padding_ang",
    "window_min_A", "window_max_A", "window_width_A", "window_mean_ev", "window_std_ev",
    "window_range_ev", "window_gradient_ev_A", "water_density_A3", "cshe_offset_ev", "time_fs",
}
_INTEGER_COLUMNS = {"id", "reference_valid", "window_grid_points"}
POTENTIAL_COLUMNS = [
    "id", "source", "source_dir", "output_out", "efermi_ev", "water_bulk_potential_ev",
    "electrode_cshe_ev", "requested_padding_ang", "window_min_A", "window_max_A",
    "window_width_A", "window_grid_points", "window_mean_ev", "window_std_ev", "window_range_ev",
    "window_gradient_ev_A", "water_density_A3", "surface_layout_requested", "surface_layout_resolved",
    "reference_valid", "reference_rejection_reason", "reference_plan_json", "reference_diagnostics_json",
    "sensitivity_json", "config_json", "fermi_provenance", "gauge_assumption", "cshe_offset_ev",
    "time_fs", "status", "error",
]
_COMBINED_POTENTIAL_SOURCES_GROUP = "combined_sources"
_POTENTIAL_PLOT_SERIES_SPECS = (
    ("water_bulk_potential_ev", "Water bulk"),
    ("efermi_ev", "Fermi"),
    ("electrode_cshe_ev", "cSHE"),
)


@dataclass(frozen=True)
class CubeHeader:
    """Header metadata extracted from a CP2K cube file."""

    natoms: int
    origin_bohr: np.ndarray
    nx: int
    ny: int
    nz: int
    vx_bohr: np.ndarray
    vy_bohr: np.ndarray
    vz_bohr: np.ndarray
    atom_numbers: np.ndarray
    atom_z_bohr: np.ndarray
    atom_positions_bohr: np.ndarray | None = None


@dataclass(frozen=True)
class PotentialRecord:
    """One snapshot, including rejected candidate statistics and provenance."""

    id: int | None
    source: str
    source_dir: str
    output_out: str | None
    efermi_ev: float | None
    water_bulk_potential_ev: float | None
    electrode_cshe_ev: float | None
    status: str
    error: str | None
    requested_padding_ang: float | None = None
    window_min_A: float | None = None
    window_max_A: float | None = None
    window_width_A: float | None = None
    window_grid_points: int = 0
    window_mean_ev: float | None = None
    window_std_ev: float | None = None
    window_range_ev: float | None = None
    window_gradient_ev_A: float | None = None
    water_density_A3: float | None = None
    surface_layout_requested: str | None = None
    surface_layout_resolved: str | None = None
    reference_valid: bool = False
    reference_rejection_reason: str | None = None
    reference_plan_json: str | None = None
    reference_diagnostics_json: str | None = None
    sensitivity_json: str | None = None
    config_json: str | None = None
    fermi_provenance: str | None = None
    gauge_assumption: str | None = None
    cshe_offset_ev: float | None = None
    time_fs: float | None = None
    profile: dict[str, np.ndarray] | None = None

    def as_row_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in POTENTIAL_COLUMNS}

    def is_complete(self) -> bool:
        return self.reference_valid and self.efermi_ev is not None and self.electrode_cshe_ev is not None


@dataclass(frozen=True)
class PotentialPlotSeries:
    """One rendered potential summary series."""

    series_id: str
    default_label: str
    x_values: np.ndarray
    y_values: np.ndarray
    source_path: str
    total_rows: int
    complete_rows: int
    incomplete_rows: int


@dataclass(frozen=True)
class PotentialComputationFailure:
    """Represents one failed source during batch computation."""

    source: str
    error: str


PotentialRecordCallback = Callable[[PotentialRecord], None]
PotentialFailureCallback = Callable[[PotentialComputationFailure], None]


@dataclass(frozen=True)
class PotentialHdf5WriteResult:
    """Result metadata from an HDF5 write operation."""

    path: Path
    rows_written: int
    mode: str


class PotentialHdf5Appender:
    """Incremental HDF5 writer for robust long-running potential workflows."""

    def __init__(
        self,
        *,
        output: str | Path,
        append: bool = True,
        overwrite: bool = False,
        sync_on_write: bool = True,
        compression_profile: str = "fast",
    ) -> None:
        self.requested_output = resolve_hdf5_output_path(output)
        self.plan = plan_potential_hdf5_output(
            self.requested_output,
            append=append,
            overwrite=overwrite,
        )
        self.path = self.plan.target_path
        self.mode = self.plan.mode
        self.fieldnames = list(self.plan.fieldnames)
        self.existing_source_keys = set(self.plan.existing_source_keys)
        self.sync_on_write = sync_on_write
        self.compression_profile = normalize_compression_profile(compression_profile)

        self._handle: Any | None = None
        self._datasets: dict[str, Any] = {}
        self.rows_written = 0
        self._next_id = 1

    def __enter__(self) -> PotentialHdf5Appender:
        self._open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _dataset_dtype_for(field: str) -> Any:
        if field in _INTEGER_COLUMNS:
            return np.int64
        return np.float64 if field in _NUMERIC_COLUMNS else hdf5_string_dtype()

    @staticmethod
    def _storage_value(field: str, value: Any) -> Any:
        if field in _INTEGER_COLUMNS:
            return -1 if value is None else int(value)
        if field in _NUMERIC_COLUMNS:
            return np.nan if value is None else float(value)
        return "" if value is None else str(value)

    def _open_hdf5_for_write(self, path: Path, *, mode: str) -> Any:
        require_h5py()
        import h5py

        handle = h5py.File(path, mode)
        records_group = handle.require_group("records")
        existing_rows = int(records_group["id"].shape[0]) if "id" in records_group else 0
        for field in POTENTIAL_COLUMNS:
            if field in records_group:
                dataset = records_group[field]
                if dataset.ndim != 1:
                    raise ValueError(f"Potential dataset '{field}' must be 1D.")
                self._datasets[field] = dataset
                continue
            dataset_options: dict[str, Any] = {"chunks": (4096,)}
            if self.compression_profile == "fast":
                dataset_options["compression"] = "lzf"
            elif self.compression_profile == "balanced":
                dataset_options["compression"] = "gzip"
                dataset_options["compression_opts"] = 4
            if self.compression_profile != "none" and field == "id":
                dataset_options["shuffle"] = True
            self._datasets[field] = records_group.create_dataset(
                field,
                shape=(existing_rows,),
                maxshape=(None,),
                dtype=self._dataset_dtype_for(field),
                **dataset_options,
            )
            if existing_rows:
                self._datasets[field][...] = self._storage_value(field, None)

        handle.attrs["linak_format"] = LINAK_HDF5_FORMAT
        handle.attrs["linak_format_version"] = LINAK_HDF5_VERSION
        handle.attrs["analysis"] = "potential"
        handle.attrs["analysis_schema_version"] = _POTENTIAL_SCHEMA_VERSION
        if "created_utc" not in handle.attrs:
            handle.attrs["created_utc"] = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
        handle.attrs["linak_version"] = __version__
        handle.attrs["compression_profile"] = self.compression_profile
        handle.attrs["storage_precision_policy"] = STORAGE_PRECISION_POLICY
        handle.attrs["columns_json"] = json.dumps(POTENTIAL_COLUMNS)
        if "metadata_json" not in handle.attrs:
            handle.attrs["metadata_json"] = json.dumps(
                {
                    "columns": POTENTIAL_COLUMNS,
                    "analysis": "potential",
                    "analysis_schema_version": _POTENTIAL_SCHEMA_VERSION,
                    "potential_convention": "CP2K electron-energy convention; no sign inversion",
                    "electrode_potential_definition": "V_water - E_F - cshe_offset_ev",
                    "units_map": {
                        **{name: "eV" for name in _NUMERIC_COLUMNS if name.endswith("_ev")},
                        **{name: "Angstrom" for name in ("requested_padding_ang", "window_min_A", "window_max_A", "window_width_A")},
                        "window_gradient_ev_A": "eV/Angstrom", "water_density_A3": "molecules/Angstrom^3",
                        "window_grid_points": "count", "time_fs": "fs",
                    },
                    "compression_profile": self.compression_profile,
                    "storage_precision_policy": STORAGE_PRECISION_POLICY,
                },
                sort_keys=True,
            )
        return handle

    @staticmethod
    def _try_fsync_hdf5(handle: Any) -> None:
        try:
            raw_handle = handle.id.get_vfd_handle()
            fd = raw_handle[0] if isinstance(raw_handle, tuple) else raw_handle
            if isinstance(fd, int):
                os.fsync(fd)
        except Exception:
            LOGGER.debug("fsync failed while writing HDF5 potential output; continuing.")

    def _open(self) -> None:
        target_path = self.path
        existing_path_has_rows = target_path.exists() and bool(_read_table_columns(target_path))
        if self.mode == "a" and existing_path_has_rows:
            self._next_id = _read_max_existing_id(target_path) + 1

        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._open_hdf5_for_write(target_path, mode=self.mode)

        assert self._handle is not None
        self._handle.flush()
        if self.sync_on_write:
            self._try_fsync_hdf5(self._handle)

    @timed_stage("writing")
    def append_record(self, record: PotentialRecord) -> None:
        if self._handle is None or not self._datasets:
            raise RuntimeError("HDF5 appender is not open.")

        if record.id is None:
            record = replace(record, id=self._next_id)
            self._next_id += 1

        if record.reference_plan_json:
            previous = self._handle.attrs.get("reference_plan_json")
            if previous is not None and str(previous) != record.reference_plan_json:
                raise ValueError("Cannot append snapshots selected with a different reference plan. Use --overwrite.")
        if record.config_json:
            previous_config = self._handle.attrs.get("reference_config_json")
            if previous_config is not None and str(previous_config) != record.config_json:
                raise ValueError("Cannot append snapshots with different validation or referencing settings. Use a separate output.")
        row = record.as_row_dict()
        next_index = int(self._datasets["id"].shape[0])
        for field in POTENTIAL_COLUMNS:
            dataset = self._datasets[field]
            dataset.resize((next_index + 1,))
            dataset[next_index] = self._storage_value(field, row.get(field))

        if record.reference_plan_json:
            self._handle.attrs["reference_plan_json"] = record.reference_plan_json
        if record.config_json:
            self._handle.attrs["reference_config_json"] = record.config_json
        if record.profile is not None:
            group = self._handle.require_group("profiles").create_group(str(record.id))
            for name, values in record.profile.items():
                group.create_dataset(name, data=values, compression="lzf")
        self.rows_written += 1
        self._handle.flush()
        if self.sync_on_write:
            self._try_fsync_hdf5(self._handle)

    def append_records(self, records: list[PotentialRecord]) -> None:
        for record in records:
            self.append_record(record)

    @timed_stage("writing")
    def close(self) -> None:
        if self._handle is not None:
            if self.sync_on_write:
                self._try_fsync_hdf5(self._handle)
            self._handle.close()
            self._handle = None
            self._datasets = {}


@dataclass(frozen=True)
class PotentialHdf5Plan:
    """Pre-write planning result for potential HDF5 output handling."""

    target_path: Path
    mode: str
    fieldnames: list[str]
    existing_source_keys: set[str]


def _compact_path(path: str | Path, *, max_chars: int = 36) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text

    path_obj = Path(text)
    parts = path_obj.parts
    if not parts:
        return text[-max_chars:]

    suffix_parts: list[str] = []
    current_length = 1  # for leading ellipsis
    for part in reversed(parts):
        extra = len(part) + (1 if suffix_parts else 0)
        if current_length + extra > max_chars:
            break
        suffix_parts.append(part)
        current_length += extra

    if not suffix_parts:
        return text[: max_chars - 1] + "…"
    return "…" + str(Path(*reversed(suffix_parts)))


def _normalize_source_key(value: str | Path) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        return raw


def validate_hartree_cube_source(path: str | Path) -> Path:
    """Validate and resolve one Hartree cube input path."""
    return validate_cube_source(path)


def _parse_fermi_in_text(text: str) -> tuple[float, str] | None:
    best_match: tuple[float, str, int] | None = None
    for pattern, unit in FERMI_PATTERNS:
        for match in pattern.finditer(text):
            value = float(match.group(1))
            offset = int(match.start())
            if best_match is None or offset >= best_match[2]:
                best_match = (value, unit, offset)

    if best_match is None:
        return None
    return best_match[0], best_match[1]


def parse_fermi_ev(path: str | Path) -> float | None:
    """Parse the last CP2K Fermi energy entry from an output file and return eV."""
    output_path = Path(path).expanduser().resolve()
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    tail_bytes = 1_000_000
    size = 0
    read_n = 0
    try:
        with output_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            read_n = min(size, tail_bytes)
            handle.seek(max(0, size - read_n), os.SEEK_SET)
            tail_text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        tail_text = output_path.read_text(encoding="utf-8", errors="replace")
        size = len(tail_text)
        read_n = size

    parsed = _parse_fermi_in_text(tail_text)
    if parsed is None and read_n < size:
        parsed = _parse_fermi_in_text(output_path.read_text(encoding="utf-8", errors="replace"))
    if parsed is None:
        return None

    value, unit = parsed
    value = float(value * HARTREE_TO_EV) if unit == "au" else float(value)
    return value if np.isfinite(value) else None


def find_cp2k_output_file(search_dir: str | Path) -> Path | None:
    root = Path(search_dir)
    preferred = root / "output.out"
    if preferred.is_file():
        return preferred
    candidates = [p for p in root.glob("*.out") if not p.name.startswith("slurm-")]
    if len(candidates) > 1:
        raise ValueError("Ambiguous CP2K output files; provide a source-specific --fermi-map.")
    return candidates[0] if candidates else None


def _display_source_dir(source_path: Path) -> str:
    parent = source_path.parent
    if parent.name:
        return parent.name
    return str(parent)


def _display_output_out(source_dir: Path, output_out: Path | None) -> str | None:
    if output_out is None:
        return None
    if output_out.parent == source_dir:
        return output_out.name
    return str(output_out)


def _cube_header_from_dataset(dataset: CubeDataset) -> CubeHeader:
    grid_counts = np.asarray(dataset.grid_counts_signed, dtype=int)
    if grid_counts.shape != (3,):
        raise ValueError("Cube dataset has invalid grid_counts_signed shape.")
    if np.any(grid_counts == 0) or not (np.all(grid_counts > 0) or np.all(grid_counts < 0)):
        raise ValueError("Cube grid counts must have a consistent nonzero coordinate-unit convention.")
    # Negative cube grid counts denote Angstrom coordinates. Scalar values
    # still carry CP2K Hartree units; do not rescale the potential here.
    coordinate_factor = 1 / BOHR_TO_ANG if np.all(grid_counts < 0) else 1.0
    atom_positions_bohr = np.asarray(dataset.atom_positions_bohr, dtype=float)
    if atom_positions_bohr.ndim != 2 or atom_positions_bohr.shape[1] != 3:
        raise ValueError("Cube dataset has invalid atom_positions_bohr shape.")
    return CubeHeader(
        natoms=abs(int(dataset.natoms_signed)),
        origin_bohr=np.asarray(dataset.origin_bohr, dtype=float) * coordinate_factor,
        nx=int(abs(grid_counts[0])),
        ny=int(abs(grid_counts[1])),
        nz=int(abs(grid_counts[2])),
        vx_bohr=np.asarray(dataset.grid_vectors_bohr[0], dtype=float) * coordinate_factor,
        vy_bohr=np.asarray(dataset.grid_vectors_bohr[1], dtype=float) * coordinate_factor,
        vz_bohr=np.asarray(dataset.grid_vectors_bohr[2], dtype=float) * coordinate_factor,
        atom_numbers=np.asarray(dataset.atom_numbers, dtype=int),
        atom_z_bohr=np.asarray(atom_positions_bohr[:, 2], dtype=float) * coordinate_factor,
        atom_positions_bohr=atom_positions_bohr * coordinate_factor,
    )


def cube_xyavg_vs_dataset(dataset: CubeDataset) -> tuple[np.ndarray, np.ndarray, CubeHeader]:
    """Cube ordering is X outermost, Z innermost; never select by smoothness."""
    header = _cube_header_from_dataset(dataset)
    values = np.asarray(dataset.values, dtype=float)
    if values.shape != (header.nx, header.ny, header.nz):
        raise ValueError("Cube field shape does not match its grid counts.")
    if (not np.allclose(header.vx_bohr[2], 0, atol=1e-10)
            or not np.allclose(header.vy_bohr[2], 0, atol=1e-10)
            or not np.allclose(header.vz_bohr[:2], 0, atol=1e-10)
            or header.vz_bohr[2] <= 0):
        raise ValueError("Hartree profile requires an XY grid with a decoupled positive Cartesian Z axis.")
    z = (header.origin_bohr[2] + np.arange(header.nz) * header.vz_bohr[2]) * BOHR_TO_ANG
    return z, values.mean(axis=(0, 1)), header


def cube_xyavg_vs_z(path: str | Path) -> tuple[np.ndarray, np.ndarray, CubeHeader]:
    datasets = read_cube_sources(path)
    if len(datasets) != 1:
        raise ValueError("Expected exactly one cube field.")
    return cube_xyavg_vs_dataset(datasets[0])


def _cube_atoms(header: CubeHeader) -> Atoms | None:
    positions_bohr = header.atom_positions_bohr
    if positions_bohr is None:
        return None
    positions = np.asarray(positions_bohr, dtype=float) * BOHR_TO_ANG
    cell = (
        np.asarray(
            [
                float(header.nx) * np.asarray(header.vx_bohr, dtype=float),
                float(header.ny) * np.asarray(header.vy_bohr, dtype=float),
                float(header.nz) * np.asarray(header.vz_bohr, dtype=float),
            ],
            dtype=float,
        )
        * BOHR_TO_ANG
    )
    atoms = Atoms(
        numbers=np.asarray(header.atom_numbers, dtype=int),
        positions=positions,
        cell=cell,
        pbc=True,
    )
    atoms.set_celldisp(np.asarray(header.origin_bohr, dtype=float) * BOHR_TO_ANG)
    return atoms


def _resolve_cube_dataset_provenance(dataset: CubeDataset) -> Path:
    if dataset.source_path:
        return Path(dataset.source_path).expanduser().resolve()
    fallback_name = dataset.source_name or "cube_dataset.cube"
    return Path(fallback_name).expanduser().resolve()


def expand_hartree_cube_sources(source: str | Path) -> list[CubeDataset]:
    """Expand one raw cube or `.cube.h5` container into logical cube datasets."""

    return read_cube_sources(source)


@dataclass
class _PreparedPotential:
    source: str
    path: Path
    z: np.ndarray
    values: np.ndarray
    geometry: Any
    geometry_error: str | None
    input_metadata: dict


def _prepare_potential(source, config: PotentialConfig) -> _PreparedPotential:
    dataset = source if isinstance(source, CubeDataset) else timed_stage("loading")(read_cube_sources)(validate_hartree_cube_source(source))[0]
    path = _resolve_cube_dataset_provenance(dataset)
    key = str(path)
    if dataset.source_profile_index:
        key += f"::profile:{dataset.source_profile_index}"
    description = f"{dataset.comment_1} {dataset.comment_2}".lower()
    if "density" in description and "hartree" not in description:
        raise ValueError("A density cube is not a CP2K V_HARTREE reference.")
    z, values, header = cube_xyavg_vs_dataset(dataset)
    atoms = _cube_atoms(header)
    geometry, reason = None, None
    try:
        if atoms is None:
            raise ValueError("Cube lacks complete atomic coordinates.")
        geometry = reference_geometry(atoms, config)
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
    return _PreparedPotential(key, path, z, values * HARTREE_TO_EV, geometry, reason, {
        "input_coordinate_unit": "Angstrom" if np.all(dataset.grid_counts_signed < 0) else "Bohr",
        "input_scalar_unit": "Hartree", "cube_grid_order": "X outermost; Z innermost",
        "cube_shape": list(dataset.values.shape), "stored_profile_unit": "eV",
    })


def _snapshot_fermi(item: _PreparedPotential, config: PotentialConfig, shared_output: bool):
    if config.fermi_map is not None:
        value = config.fermi_map.get(item.source)
        if value is None or not np.isfinite(value):
            return None, None, "Missing finite source-specific Fermi energy in --fermi-map."
        return float(value), None, "explicit source-specific Fermi map (user asserts matching CP2K gauge)"
    try:
        output = find_cp2k_output_file(item.path.parent)
    except ValueError as exc:
        return None, None, str(exc)
    if output is None:
        return None, None, "No matching CP2K output found."
    if shared_output:
        return None, output, "Multiple snapshots share one output; provide --fermi-map instead of reusing its last Fermi energy."
    text = output.read_text(encoding="utf-8", errors="replace")
    steps = re.findall(r"MD\|\s*Step number\s+(\d+)", text)
    if len(set(steps)) > 1:
        return None, output, "Multi-step CP2K output has ambiguous snapshot association; provide --fermi-map."
    convergence = re.findall(r"SCF run[^\n]*", text, re.I)
    if convergence and re.search(r"NOT\s+converged", convergence[-1], re.I):
        return None, output, "The final CP2K SCF run did not converge; its Fermi energy is not a valid reference."
    value = parse_fermi_ev(output)
    note = "last Fermi entry in source-specific CP2K output; " + (
        "SCF convergence declared" if convergence else "SCF convergence marker not available"
    )
    return value, output, note if value is not None else "No finite Fermi energy parsed."


def _record_from_prepared(item, config, plan, *, shared_output=False, index=0):
    efermi, output, fermi_note = _snapshot_fermi(item, config, shared_output)
    diagnostics = {"valid": False, "reasons": [item.geometry_error], "grid_points": 0}
    selected = np.zeros(len(item.z), dtype=bool)
    sensitivity = {}
    profile = {"z_A": item.z, "hartree_ev": item.values, "selected": selected.astype(np.int8)}
    if item.geometry is not None:
        diagnostics, weights = evaluate_reference(item.z, item.values, item.geometry, plan, config)
        selected = weights > 1e-14
        sensitivity = sensitivity_windows(item.z, item.values, item.geometry, plan, config)
        g = item.geometry
        profile.update(solvent_s_A=g.coordinate(item.z), gap_A=np.asarray([g.length]),
                       water_s_A=g.water_s, area_A2=np.asarray([g.area]), selected=selected.astype(np.int8),
                       averaging_weights=weights)
        diagnostics.update(slab_bottom_z_A=g.slab_bottom, slab_top_z_A=g.slab_top,
                           solvent_start_z_A=g.start, solvent_gap_A=g.length,
                           cell_origin_z_A=g.origin, cell_period_A=g.period)
    diagnostics.update(item.input_metadata)
    valid = bool(diagnostics["valid"])
    water = diagnostics.get("mean_ev") if valid else None
    cshe = water - efermi - config.cshe_offset_ev if water is not None and efermi is not None else None
    reasons = [str(reason) for reason in diagnostics["reasons"] if reason]
    status = "ok" if cshe is not None else "incomplete" if valid else "invalid"
    if status != "ok":
        LOGGER.warning("Potential reference %s for '%s': %s", status, _compact_path(item.source),
                       "; ".join(reasons) if reasons else fermi_note)
    return PotentialRecord(
        id=None, source=item.source, source_dir=_display_source_dir(item.path),
        output_out=None if output is None else _display_output_out(item.path.parent, output),
        efermi_ev=efermi, water_bulk_potential_ev=water, electrode_cshe_ev=cshe,
        status=status, error=None if cshe is not None else "; ".join([*reasons, *([] if efermi is not None else [fermi_note])]),
        requested_padding_ang=config.water_padding_ang,
        window_min_A=diagnostics.get("min_A"), window_max_A=diagnostics.get("max_A"),
        window_width_A=diagnostics.get("width_A"), window_grid_points=diagnostics["grid_points"],
        window_mean_ev=diagnostics.get("mean_ev"), window_std_ev=diagnostics.get("std_ev"),
        window_range_ev=diagnostics.get("range_ev"), window_gradient_ev_A=diagnostics.get("gradient_ev_A"),
        water_density_A3=diagnostics.get("water_density_A3"),
        surface_layout_requested=config.surface_layout,
        surface_layout_resolved=None if item.geometry is None else item.geometry.layout,
        reference_valid=valid, reference_rejection_reason="; ".join(reasons) or None,
        reference_plan_json=json.dumps(plan, sort_keys=True),
        reference_diagnostics_json=json.dumps(diagnostics, sort_keys=True),
        sensitivity_json=json.dumps(sensitivity, sort_keys=True), config_json=serialize_config(config),
        fermi_provenance=fermi_note, cshe_offset_ev=config.cshe_offset_ev,
        gauge_assumption="CP2K electron-energy convention; V_HARTREE and E_F must be from the same SCF gauge. No per-frame recentering applied.",
        time_fs=None if config.snapshot_interval_fs is None else index * config.snapshot_interval_fs,
        profile=profile,
    )


def compute_potential_record(source, *, config: PotentialConfig) -> PotentialRecord:
    validate_config(config)
    item = _prepare_potential(source, config)
    plan = build_reference_plan([] if item.geometry is None else [item.geometry], config)
    return _record_from_prepared(item, config, plan)


def error_record_for_source(source: str | Path, error: str, config: PotentialConfig | None = None) -> PotentialRecord:
    """Create a failure record suitable for HDF5 persistence."""
    source_path = Path(source).expanduser()
    try:
        source_path = source_path.resolve()
    except OSError:
        source_path = source_path.absolute()
    return PotentialRecord(
        id=None,
        source=str(source_path),
        source_dir=_display_source_dir(source_path),
        output_out=None,
        efermi_ev=None,
        water_bulk_potential_ev=None,
        electrode_cshe_ev=None,
        status="error",
        error=error,
        reference_rejection_reason=error,
        requested_padding_ang=None if config is None else config.water_padding_ang,
        cshe_offset_ev=None if config is None else config.cshe_offset_ev,
    )


def _estimate_cube_worker_bytes(source: str | Path | CubeDataset) -> int:
    """Estimate cube payload and scratch bytes without loading raw cube values."""

    if isinstance(source, CubeDataset):
        return max(64 * MIB, int(np.asarray(source.values).nbytes) * 4)
    path = Path(source).expanduser().resolve()
    if path.suffix.lower() == ".cube":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.readline()
                handle.readline()
                origin_fields = handle.readline().split()
                atom_count = abs(int(origin_fields[0]))
                counts = [abs(int(handle.readline().split()[0])) for _ in range(3)]
            value_bytes = int(np.prod(counts, dtype=np.int64)) * 8
            atom_bytes = atom_count * 5 * 8
            return max(64 * MIB, 4 * (value_bytes + atom_bytes))
        except (OSError, ValueError, IndexError):
            pass
    try:
        return max(64 * MIB, int(path.stat().st_size) * 4)
    except OSError:
        return 64 * MIB


def _resolve_worker_count(
    threads: int | None,
    n_sources: int,
    *,
    estimated_bytes_per_worker: int = 64 * MIB,
    memory_budget_mib: float | None = None,
) -> int:
    if n_sources <= 0:
        return 1

    if threads is not None and threads < 1:
        raise ValueError("Potential threads must be >= 1.")
    return stage_workers('potential.files', n_sources,
        bytes_per_task=max(MIB, estimated_bytes_per_worker // 4), memory_budget_mib=memory_budget_mib,
        requested_threads=threads, use_requested=threads is not None or current_execution() is None,
    )


@timed_stage("calculation")
def compute_potential_records(
    sources: list[str | Path | CubeDataset], *, config: PotentialConfig,
    threads: int | None = None, memory_budget_mib: float | None = None,
    on_record: PotentialRecordCallback | None = None,
    on_failure: PotentialFailureCallback | None = None,
    start_index: int = 0,
    previous_sources: Sequence[str] = (),
) -> tuple[list[PotentialRecord], list[PotentialComputationFailure]]:
    """Plan from all structures, then evaluate a fixed plan in source order."""
    validate_config(config)
    if not sources:
        raise ValueError("At least one source is required.")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a nonnegative snapshot index.")
    workers = _resolve_worker_count(threads, len(sources), memory_budget_mib=memory_budget_mib,
        estimated_bytes_per_worker=max(_estimate_cube_worker_bytes(source) for source in sources))
    prepared = {}
    failures = {}
    def prepare(index, source):
        try:
            return index, _prepare_potential(source, config), None
        except Exception as exc:
            name = str(_resolve_cube_dataset_provenance(source)) if isinstance(source, CubeDataset) else str(source)
            return index, None, PotentialComputationFailure(name, str(exc))
    with ProgressBar(desc="Planning potential references", total=len(sources), unit="file") as progress:
        for index, item, failure in bounded_map(
            lambda pair: prepare(*pair), enumerate(sources), workers=workers, stage='potential.files',
        ):
            if item is not None:
                prepared[index] = item
            if failure is not None:
                failures[index] = failure
            progress.update()
    plan = build_reference_plan([item.geometry for item in prepared.values() if item.geometry is not None], config)
    from collections import Counter
    directories = Counter(str(item.path.parent) for item in prepared.values())
    directories.update(str(Path(source.split("::profile:")[0]).parent) for source in previous_sources)
    records = []
    with ProgressBar(desc="Computing potentials", total=len(sources), unit="file") as progress:
        for index in range(len(sources)):
            if index in prepared:
                item = prepared.pop(index)
                record = _record_from_prepared(item, config, plan,
                    shared_output=directories[str(item.path.parent)] > 1, index=start_index + index)
                records.append(record)
                if on_record is not None:
                    on_record(record)
            elif on_failure is not None:
                on_failure(failures[index])
            progress.update()
    return records, [failures[index] for index in sorted(failures)]


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return None, None, 0

    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    return mean, std, int(valid.size)


def summarize_potential_statistics(
    records: list[PotentialRecord],
) -> dict[str, tuple[float | None, float | None, int]]:
    """Return mean, std, and valid-count for key potential metrics."""
    return {
        "efermi_ev": _mean_std([record.efermi_ev for record in records]),
        "water_bulk_potential_ev": _mean_std(
            [record.water_bulk_potential_ev for record in records]
        ),
        "electrode_cshe_ev": _mean_std([record.electrode_cshe_ev for record in records]),
    }


def _validate_potential_table(handle: h5py.File, *, path: Path) -> None:
    """Check current table metadata and column shapes without reading payloads."""
    import h5py

    _read_required_linak_header(handle, path=path, expected_analysis="potential")
    if handle.attrs.get("analysis_schema_version") != _POTENTIAL_SCHEMA_VERSION:
        raise ValueError(
            f"Potential HDF5 '{path}' requires analysis schema version "
            f"{_POTENTIAL_SCHEMA_VERSION}. Recompute this analysis with the current LiNaK."
        )
    records = handle.get("records")
    if not isinstance(records, h5py.Group):
        raise ValueError(f"Potential HDF5 '{path}' is missing '/records'. Recompute this analysis.")
    missing = [
        column
        for column in POTENTIAL_COLUMNS
        if not isinstance(records.get(column), h5py.Dataset)
    ]
    if missing:
        raise ValueError(
            f"Potential HDF5 '{path}' is missing required column(s): {', '.join(missing)}. "
            "Recompute this analysis."
        )
    row_shape = records["id"].shape
    if len(row_shape) != 1 or any(
        records[column].shape != row_shape for column in POTENTIAL_COLUMNS
    ):
        raise ValueError(
            f"Potential HDF5 '{path}' requires aligned one-dimensional record columns. "
            "Recompute this analysis."
        )


def load_potential_records(source: str | Path, *, include_profiles: bool = False) -> list[PotentialRecord]:
    require_h5py()
    import h5py
    path = Path(source).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        _validate_potential_table(handle, path=path)
        group = handle["records"]
        columns = {
            name: (group[name][...] if name in _NUMERIC_COLUMNS | _INTEGER_COLUMNS else group[name].asstr()[...])
            for name in POTENTIAL_COLUMNS
        }
        rows = []
        for index in range(len(columns["id"])):
            row = {}
            for name, values in columns.items():
                value = values[index]
                if name in _NUMERIC_COLUMNS:
                    row[name] = float(value) if np.isfinite(value) else None
                elif name in _INTEGER_COLUMNS:
                    row[name] = bool(value) if name == "reference_valid" else None if value < 0 else int(value)
                else:
                    row[name] = str(value) or None
            if include_profiles and "profiles" in handle and str(row["id"]) in handle["profiles"]:
                row["profile"] = {name: data[...] for name, data in handle["profiles"][str(row["id"])].items()}
            rows.append(PotentialRecord(**row))
        return rows


def combine_potential_hdf5_sources(
    sources: Sequence[str | Path],
    *,
    output: str | Path,
) -> Path:
    """Write a combined potential HDF5 that contains rows from every source file."""
    from ..plot.gui_session import report_progress, own_preparation_output
    from ..cancellation import check_cancelled

    report_progress("Preparing combined results")
    existed = Path(output).expanduser().resolve().exists()
    records: list[PotentialRecord] = []
    source_segments: list[tuple[Path, int, int]] = []
    for index, source in enumerate(sources, 1):
        report_progress("Preparing combined results", Path(source).name, index, len(sources))
        source_path = Path(source).expanduser().resolve()
        source_records = load_potential_records(source_path, include_profiles=True)
        row_start = len(records)
        records.extend(source_records)
        if source_records:
            source_segments.append((source_path, row_start, len(source_records)))
    if not records:
        raise ValueError("No potential rows were available to combine.")
    from ..storage.hdf5_utils import _atomic_hdf5_target

    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_hdf5_target(target) as temporary:
        result = write_potential_records(records, output=temporary, append=False, overwrite=True)
        _write_combined_potential_source_metadata(result.path, source_segments)
        check_cancelled()
    if not existed:
        own_preparation_output(target)
    check_cancelled()
    return target


def _write_combined_potential_source_metadata(
    path: str | Path,
    source_segments: Sequence[tuple[Path, int, int]],
) -> None:
    """Persist row spans for the original HDF5 files in a combined potential output."""
    require_h5py()
    import h5py

    target_path = Path(path).expanduser().resolve()
    with h5py.File(target_path, "a") as handle:
        if _COMBINED_POTENTIAL_SOURCES_GROUP in handle:
            del handle[_COMBINED_POTENTIAL_SOURCES_GROUP]
        group = handle.create_group(_COMBINED_POTENTIAL_SOURCES_GROUP)
        group.attrs["schema_version"] = 1
        group.create_dataset(
            "source_path",
            data=np.asarray(
                [str(source_path) for source_path, _start, _count in source_segments],
                dtype=object,
            ),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        group.create_dataset(
            "row_start",
            data=np.asarray(
                [row_start for _path, row_start, _count in source_segments], dtype=np.int64
            ),
        )
        group.create_dataset(
            "row_count",
            data=np.asarray(
                [row_count for _path, _start, row_count in source_segments], dtype=np.int64
            ),
        )


def _read_combined_potential_source_metadata(
    handle: Any,
    *,
    total_rows: int,
) -> list[tuple[Path, int, int]]:
    """Return valid combined-source row spans, or an empty list for ordinary files."""
    if _COMBINED_POTENTIAL_SOURCES_GROUP not in handle:
        return []
    group = handle[_COMBINED_POTENTIAL_SOURCES_GROUP]
    required = ("source_path", "row_start", "row_count")
    if any(name not in group for name in required):
        return []
    source_paths = np.asarray(group["source_path"].asstr()[...], dtype=object)
    row_starts = np.asarray(group["row_start"], dtype=np.int64)
    row_counts = np.asarray(group["row_count"], dtype=np.int64)
    if not (len(source_paths) == len(row_starts) == len(row_counts)):
        return []

    segments: list[tuple[Path, int, int]] = []
    for source_path_raw, row_start_raw, row_count_raw in zip(
        source_paths,
        row_starts,
        row_counts,
    ):
        source_path = Path(str(source_path_raw).strip()).expanduser()
        row_start = int(row_start_raw)
        row_count = int(row_count_raw)
        if not str(source_path).strip() or row_count <= 0:
            continue
        if row_start < 0 or row_start + row_count > total_rows:
            continue
        segments.append((source_path.resolve(), row_start, row_count))
    return segments


def _potential_plot_order_for_rows(
    raw_ids: np.ndarray,
    *,
    row_start: int,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_indices = np.arange(row_start, row_start + row_count, dtype=np.int64)
    segment_ids = np.asarray(raw_ids[row_indices], dtype=np.int64)
    finite_positive_ids = segment_ids.size > 0 and bool(np.all(segment_ids > 0))
    unique_positive_ids = finite_positive_ids and len(np.unique(segment_ids)) == len(segment_ids)
    if finite_positive_ids and unique_positive_ids:
        local_order = np.argsort(segment_ids, kind="mergesort")
        return row_indices[local_order], np.asarray(segment_ids[local_order], dtype=float)
    return row_indices, np.arange(1, row_count + 1, dtype=float)


def load_potential_plot_profiles(
    source: str | Path,
    *,
    _metadata_only: bool = False,
) -> tuple[list[PotentialPlotSeries], dict[str, Any]]:
    """Load a potential HDF5 file into fixed plotting series."""
    require_h5py()
    import h5py

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")

    with h5py.File(source_path, "r") as handle:
        _validate_potential_table(handle, path=source_path)
        records = handle["records"]

        total_rows = int(records["id"].shape[0])
        if _metadata_only:
            raw_ids = np.empty(0, dtype=np.int64)
            values_by_column = {}
            status_values = np.empty(0, dtype=object)
        else:
            raw_ids = np.asarray(records["id"], dtype=np.int64)
            values_by_column = {
                column: np.asarray(records[column], dtype=float)
                for column, _label in _POTENTIAL_PLOT_SERIES_SPECS
            }
            status_values = np.asarray(records["status"].asstr()[...], dtype=object)
        combined_segments = _read_combined_potential_source_metadata(
            handle,
            total_rows=total_rows,
        )

    if not combined_segments:
        combined_segments = [(source_path, 0, total_rows)]

    profiles: list[PotentialPlotSeries] = []
    total_complete_rows = 0
    for segment_source_path, row_start, row_count in combined_segments:
        if _metadata_only:
            order = np.empty(0, dtype=np.int64)
            x_values = np.empty(0)
            complete_rows = incomplete_rows = 0
        else:
            order, x_values = _potential_plot_order_for_rows(
                raw_ids,
                row_start=row_start,
                row_count=row_count,
            )
            ordered_status = [str(status_values[index]).strip().lower() for index in order]
            complete_rows = sum(1 for value in ordered_status if value == "ok")
            incomplete_rows = row_count - complete_rows
        total_complete_rows += complete_rows
        is_combined_segment = segment_source_path != source_path
        label_prefix = (
            segment_source_path.stem or segment_source_path.name or str(segment_source_path)
        )
        for series_id, default_label in _POTENTIAL_PLOT_SERIES_SPECS:
            rendered_series_id = (
                f"{segment_source_path}::{series_id}" if is_combined_segment else series_id
            )
            rendered_label = (
                f"{label_prefix}: {default_label}" if is_combined_segment else default_label
            )
            profiles.append(
                PotentialPlotSeries(
                    series_id=rendered_series_id,
                    default_label=rendered_label,
                    x_values=np.asarray(x_values, dtype=float),
                    y_values=(
                        np.empty(0)
                        if _metadata_only
                        else np.asarray(values_by_column[series_id][order], dtype=float)
                    ),
                    source_path=str(segment_source_path),
                    total_rows=row_count,
                    complete_rows=complete_rows,
                    incomplete_rows=incomplete_rows,
                )
            )

    return profiles, {
        "x_axis_label": "Record ID",
        "total_rows": total_rows,
        "complete_rows": total_complete_rows,
        "incomplete_rows": total_rows - total_complete_rows,
    }


def plot_potential_profiles(
    profiles: list[PotentialPlotSeries],
    *,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    series_ids: list[str] | None = None,
    title: str = "Hartree potential summary",
    x_label: str | None = None,
    y_label: str | None = None,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    line_colors: list[str] | None = None,
    series_labels: list[str] | None = None,
    series_error_configs: list[dict[str, Any] | None] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_cumulative_configs: list[dict[str, Any] | None] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_font_size: int | None = None,
    y_label_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_pad: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    capture_state: dict[str, Any] | None = None,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
) -> Path | None:
    """Plot potential summary series with optional fitted child overlays."""
    if not profiles:
        raise ValueError("At least one potential series is required for plotting.")
    resolved_mapping = resolve_potential_plot_mapping(
        contract=data_contract,
        profiles=profiles,
        mapping=view_mapping,
    )
    runtime_y_quantity = resolved_mapping.y_quantity
    runtime_standard_plot = resolved_mapping.standard_plot
    selected_profiles = profiles
    if runtime_standard_plot != "summary":
        series_id_by_quantity = {
            "water_bulk_potential": "water_bulk_potential_ev",
            "efermi": "efermi_ev",
            "electrode_cshe": "electrode_cshe_ev",
        }
        target_series_id = series_id_by_quantity[runtime_y_quantity]
        selected_profiles = [
            profile for profile in profiles if str(profile.series_id).strip() == target_series_id
        ]
        if not selected_profiles:
            raise ValueError(
                f"Potential series '{target_series_id}' is unavailable for the requested mapping."
            )

    labels = resolve_series_labels(
        [profile.default_label for profile in selected_profiles],
        series_labels,
        series_kind="potential",
    )
    x_series = [np.asarray(profile.x_values, dtype=float) for profile in selected_profiles]
    y_series = [np.asarray(profile.y_values, dtype=float) for profile in selected_profiles]
    output_path = plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or "Hartree potential summary",
        x_label=resolve_explicit_plot_text(x_label, "Record ID"),
        y_label=resolve_explicit_plot_text(y_label, "Potential (eV)"),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_ids=series_ids or [profile.series_id for profile in selected_profiles],
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_cumulative_configs=series_cumulative_configs,
        series_error_configs=series_error_configs,
        series_raw_statistics=[True] * len(selected_profiles),
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="potential",
        annotations=annotations,
        integration_config=integration_config,
        line_kwargs=line_kwargs,
        series_line_kwargs=series_line_kwargs,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        x_label_font_size=x_label_font_size,
        y_label_font_size=y_label_font_size,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
        title_pad=title_pad,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )
    if capture_state is not None:
        capture_state["potential_summary"] = {
            "x_axis_label": "Record ID",
            "total_rows": int(selected_profiles[0].total_rows),
            "complete_rows": int(selected_profiles[0].complete_rows),
            "incomplete_rows": int(selected_profiles[0].incomplete_rows),
        }
    return output_path


def _read_table_columns(path: Path) -> list[str]:
    require_h5py()
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            _validate_potential_table(handle, path=path)
            return list(POTENTIAL_COLUMNS)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _read_existing_source_keys(path: Path, *, header: list[str] | None = None) -> set[str]:
    columns = header if header is not None else _read_table_columns(path)
    if "source" not in columns:
        return set()

    require_h5py()
    import h5py

    source_keys: set[str] = set()
    with h5py.File(path, "r") as handle:
        if "records" not in handle:
            return set()
        records = handle["records"]
        if "source" not in records:
            return set()
        for raw in records["source"].asstr()[...]:
            key = _normalize_source_key(str(raw).strip())
            if key:
                source_keys.add(key)
    return source_keys


def _read_max_existing_id(path: Path) -> int:
    columns = _read_table_columns(path)
    if "id" not in columns:
        return 0

    require_h5py()
    import h5py

    with h5py.File(path, "r") as handle:
        if "records" not in handle:
            return 0
        records = handle["records"]
        if "id" not in records:
            return 0
        values = np.asarray(records["id"], dtype=np.int64)
    positive = values[values > 0]
    if positive.size == 0:
        return 0
    return int(np.max(positive))


def plan_potential_hdf5_output(output: str | Path, *, append: bool = True, overwrite: bool = False) -> PotentialHdf5Plan:
    path = resolve_hdf5_output_path(output)
    if path.exists() and not overwrite:
        if not append:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite or --append.")
        import h5py
        with h5py.File(path, "r") as handle:
            _validate_potential_table(handle, path=path)
        return PotentialHdf5Plan(path, "a", list(POTENTIAL_COLUMNS), _read_existing_source_keys(path))
    return PotentialHdf5Plan(path, "w", list(POTENTIAL_COLUMNS), set())


@timed_stage("writing")
def write_potential_records(
    records: list[PotentialRecord],
    output: str | Path,
    *,
    append: bool = True,
    overwrite: bool = False,
) -> PotentialHdf5WriteResult:
    """Write potential records to HDF5 with explicit schema validation and append behavior."""
    if not records:
        raise ValueError("At least one potential record is required for HDF5 export.")

    with PotentialHdf5Appender(
        output=output,
        append=append,
        overwrite=overwrite,
        sync_on_write=True,
    ) as appender:
        appender.append_records(records)
        target_path = appender.path
        mode = appender.mode
        rows_written = appender.rows_written

    LOGGER.info("Saved %d potential row(s) to '%s'.", rows_written, _compact_path(target_path))
    LOGGER.debug("Potential HDF5 path: %s", target_path)
    return PotentialHdf5WriteResult(
        path=target_path,
        rows_written=rows_written,
        mode=mode,
    )
