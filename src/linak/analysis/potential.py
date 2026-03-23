"""Electrode-potential analysis from CP2K Hartree-potential cube files."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from types import TracebackType
from typing import Any, Callable

import numpy as np

from .. import __version__
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_multi_line_series,
    resolve_series_labels,
)
from ..storage.hdf5_utils import (
    LINAK_HDF5_FORMAT,
    LINAK_HDF5_VERSION,
    hdf5_string_dtype,
    require_h5py,
    resolve_hdf5_output_path,
)
from ..progress import ProgressBar

LOGGER = logging.getLogger(__name__)

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_EV = 27.211386245988
DEFAULT_CSHE_OFFSET_EV = 0.81
_DEFAULT_POTENTIAL_THREAD_CAP = 4
_WATER_ATOMIC_NUMBERS = {1, 8}

FERMI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
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

POTENTIAL_CSV_COLUMNS = [
    "id",
    "source",
    "source_dir",
    "output_out",
    "efermi_ev",
    "water_bulk_potential_ev",
    "electrode_cshe_ev",
    "status",
    "error",
]


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


@dataclass(frozen=True)
class PotentialConfig:
    """Configurable controls for potential analysis."""

    water_padding_ang: float = 5.0
    cshe_offset_ev: float = DEFAULT_CSHE_OFFSET_EV


@dataclass(frozen=True)
class PotentialRecord:
    """One computed row for potential HDF5 export."""

    id: int | None
    source: str
    source_dir: str
    output_out: str | None
    efermi_ev: float | None
    water_bulk_potential_ev: float | None
    electrode_cshe_ev: float | None
    status: str
    error: str | None

    def as_row_dict(self) -> dict[str, object]:
        row: dict[str, object] = {}
        for field in POTENTIAL_CSV_COLUMNS:
            row[field] = getattr(self, field)
        return row

    def as_csv_row(self) -> dict[str, object]:
        """Backward-compatible alias used by existing call sites."""
        return self.as_row_dict()

    def is_complete(self) -> bool:
        return (
            self.efermi_ev is not None
            and self.water_bulk_potential_ev is not None
            and self.electrode_cshe_ev is not None
        )


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
class PotentialCsvWriteResult:
    """Result metadata from an HDF5 write operation."""

    path: Path
    rows_written: int
    mode: str
    used_fallback_path: bool


class PotentialCsvAppender:
    """Incremental HDF5 writer for robust long-running potential workflows."""

    def __init__(
        self,
        *,
        output: str | Path,
        append: bool = True,
        overwrite: bool = False,
        sync_on_write: bool = True,
    ) -> None:
        self.requested_output = resolve_hdf5_output_path(output)
        self.plan = plan_potential_csv_output(
            self.requested_output,
            append=append,
            overwrite=overwrite,
        )
        self.path = self.plan.target_path
        self.mode = self.plan.mode
        self.fieldnames = list(self.plan.fieldnames)
        self.used_fallback_path = self.plan.used_fallback_path
        self.existing_source_keys = set(self.plan.existing_source_keys)
        self.sync_on_write = sync_on_write

        self._handle: Any | None = None
        self._datasets: dict[str, Any] = {}
        self.rows_written = 0
        self._next_id = 1

    def __enter__(self) -> PotentialCsvAppender:
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
        if field == "id":
            return np.int64
        if field in {"efermi_ev", "water_bulk_potential_ev", "electrode_cshe_ev"}:
            return np.float64
        return hdf5_string_dtype()

    @staticmethod
    def _storage_value(field: str, value: Any) -> Any:
        if field == "id":
            return int(value) if value is not None else -1
        if field in {"efermi_ev", "water_bulk_potential_ev", "electrode_cshe_ev"}:
            return np.nan if value is None else float(value)
        return "" if value is None else str(value)

    def _open_hdf5_for_write(self, path: Path, *, mode: str) -> Any:
        require_h5py()
        import h5py

        handle = h5py.File(path, mode)
        records_group = handle.require_group("records")
        for field in POTENTIAL_CSV_COLUMNS:
            if field in records_group:
                dataset = records_group[field]
                if dataset.ndim != 1:
                    raise ValueError(f"Potential dataset '{field}' must be 1D.")
                self._datasets[field] = dataset
                continue
            self._datasets[field] = records_group.create_dataset(
                field,
                shape=(0,),
                maxshape=(None,),
                dtype=self._dataset_dtype_for(field),
            )

        handle.attrs["linak_format"] = LINAK_HDF5_FORMAT
        handle.attrs["linak_format_version"] = LINAK_HDF5_VERSION
        handle.attrs["analysis"] = "potential"
        if "created_utc" not in handle.attrs:
            handle.attrs["created_utc"] = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
        handle.attrs["linak_version"] = __version__
        handle.attrs["columns_json"] = json.dumps(POTENTIAL_CSV_COLUMNS)
        if "metadata_json" not in handle.attrs:
            handle.attrs["metadata_json"] = json.dumps(
                {"columns": POTENTIAL_CSV_COLUMNS, "analysis": "potential"},
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
        existing_path_has_rows = target_path.exists() and bool(_read_csv_header(target_path))
        if self.mode == "a" and existing_path_has_rows:
            self._next_id = _read_max_existing_id(target_path) + 1

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            open_mode = "a" if self.mode == "a" else "w"
            self._handle = self._open_hdf5_for_write(target_path, mode=open_mode)
        except (OSError, ValueError) as exc:
            fallback_path = _fallback_csv_path(Path.cwd() / target_path.name)
            if fallback_path == target_path:
                raise
            LOGGER.warning(
                "Could not open HDF5 '%s' (%s); using fallback '%s'.",
                _compact_path(target_path),
                exc,
                _compact_path(fallback_path),
            )
            LOGGER.debug("HDF5 open fallback: from %s to %s", target_path, fallback_path)
            self.path = fallback_path
            self.mode = "w"
            self.fieldnames = list(POTENTIAL_CSV_COLUMNS)
            self.used_fallback_path = True
            self._next_id = 1
            self._datasets = {}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._open_hdf5_for_write(self.path, mode="w")

        assert self._handle is not None
        self._handle.flush()
        if self.sync_on_write:
            self._try_fsync_hdf5(self._handle)

    def append_record(self, record: PotentialRecord) -> None:
        if self._handle is None or not self._datasets:
            raise RuntimeError("HDF5 appender is not open.")

        if record.id is None:
            record = replace(record, id=self._next_id)
            self._next_id += 1

        row = record.as_row_dict()
        next_index = int(self._datasets["id"].shape[0])
        for field in POTENTIAL_CSV_COLUMNS:
            dataset = self._datasets[field]
            dataset.resize((next_index + 1,))
            dataset[next_index] = self._storage_value(field, row.get(field))

        self.rows_written += 1
        self._handle.flush()
        if self.sync_on_write:
            self._try_fsync_hdf5(self._handle)

    def append_records(self, records: list[PotentialRecord]) -> None:
        for record in records:
            self.append_record(record)

    def close(self) -> None:
        if self._handle is not None:
            if self.sync_on_write:
                self._try_fsync_hdf5(self._handle)
            self._handle.close()
            self._handle = None
            self._datasets = {}


@dataclass(frozen=True)
class PotentialCsvPlan:
    """Pre-write planning result for potential HDF5 output handling."""

    target_path: Path
    mode: str
    fieldnames: list[str]
    used_fallback_path: bool
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


def validate_potential_config(config: PotentialConfig) -> None:
    """Validate potential-analysis configuration values."""
    if config.water_padding_ang < 0.0:
        raise ValueError(f"water_padding_ang must be >= 0 (got {config.water_padding_ang}).")


def validate_hartree_cube_source(path: str | Path) -> Path:
    """Validate and resolve one Hartree cube input path."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Hartree cube file not found: {source_path}")
    if source_path.is_dir():
        raise ValueError(f"Hartree cube source is a directory, not a file: {source_path}")
    if source_path.suffix.lower() != ".cube":
        raise ValueError(f"Hartree cube source must be a .cube file: {source_path}")
    return source_path


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
    if unit == "au":
        return float(value * HARTREE_TO_EV)
    return float(value)


def find_cp2k_output_file(search_dir: str | Path) -> Path | None:
    """Find the most likely CP2K output file in the same directory as the cube file."""
    root = Path(search_dir).expanduser().resolve()
    output_out = root / "output.out"
    if output_out.exists() and output_out.is_file():
        return output_out

    out_files = sorted(path for path in root.glob("*.out") if path.is_file())
    if not out_files:
        return None

    preferred = [
        path
        for path in out_files
        if path.name.lower() != "output.out" and not path.name.lower().startswith("slurm-")
    ]
    ordered = preferred + [path for path in out_files if path not in preferred]

    for candidate in ordered:
        try:
            if parse_fermi_ev(candidate) is not None:
                LOGGER.warning(
                    "Using fallback CP2K output '%s' (output.out not found).",
                    _compact_path(candidate),
                )
                LOGGER.debug("Selected fallback CP2K output file: %s", candidate)
                return candidate
        except OSError:
            continue
    return ordered[0]


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


def _read_cube_header(path: Path) -> tuple[CubeHeader, int]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(2):
            if not handle.readline():
                raise ValueError(f"Cube file ended unexpectedly while reading comments: {path}")

        third = handle.readline()
        if not third:
            raise ValueError(f"Cube file ended unexpectedly: {path}")
        parts = third.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid cube header line (natoms/origin) in '{path}'.")

        natoms = abs(int(parts[0]))
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)

        def read_grid_line() -> tuple[int, np.ndarray]:
            line = handle.readline()
            if not line:
                raise ValueError(f"Cube file ended unexpectedly while reading grid: {path}")
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError(f"Invalid cube grid line in '{path}'.")
            count = abs(int(tokens[0]))
            vector = np.array(
                [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                dtype=float,
            )
            return count, vector

        nx, vx = read_grid_line()
        ny, vy = read_grid_line()
        nz, vz = read_grid_line()

        atom_numbers = np.empty((natoms,), dtype=int)
        atom_z_bohr = np.empty((natoms,), dtype=float)
        for atom_index in range(natoms):
            line = handle.readline()
            if not line:
                raise ValueError(f"Cube file ended unexpectedly while reading atom list: {path}")
            tokens = line.split()
            if len(tokens) < 5:
                raise ValueError(f"Invalid atom line in cube file '{path}'.")
            atom_numbers[atom_index] = int(round(float(tokens[0])))
            atom_z_bohr[atom_index] = float(tokens[4])

    data_start_line = 2 + 1 + 3 + natoms
    header = CubeHeader(
        natoms=natoms,
        origin_bohr=origin,
        nx=nx,
        ny=ny,
        nz=nz,
        vx_bohr=vx,
        vy_bohr=vy,
        vz_bohr=vz,
        atom_numbers=atom_numbers,
        atom_z_bohr=atom_z_bohr,
    )
    return header, data_start_line


def _cube_xyavg_stream_both(
    path: Path,
    *,
    data_start_line: int,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_expected = nx * ny * nz

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(data_start_line):
            handle.readline()
        payload = handle.read()

    if "D" in payload or "d" in payload:
        payload = payload.replace("D", "E").replace("d", "E")

    values = np.fromstring(payload, sep=" ", dtype=float, count=n_expected)
    if values.size != n_expected:
        raise ValueError(
            f"Cube data length mismatch in '{path}'. Expected {n_expected}, got {values.size}."
        )

    v_xfast = values.reshape((nz, ny, nx), order="C").mean(axis=(1, 2))
    v_zfast = values.reshape((nx, ny, nz), order="C").mean(axis=(0, 1))
    return v_xfast, v_zfast


def _profile_roughness(values: np.ndarray) -> float:
    dv = np.diff(values)
    if dv.size == 0:
        return 0.0
    second = np.diff(dv)
    second_term = 0.25 * float(np.mean(np.abs(second))) if second.size else 0.0
    return float(np.mean(np.abs(dv)) + second_term)


def cube_xyavg_vs_z(path: str | Path) -> tuple[np.ndarray, np.ndarray, CubeHeader]:
    """Read one CP2K cube file and return z-grid and xy-averaged values."""
    cube_path = Path(path).expanduser().resolve()
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube file not found: {cube_path}")

    header, data_start_line = _read_cube_header(cube_path)
    v_xfast, v_zfast = _cube_xyavg_stream_both(
        cube_path,
        data_start_line=data_start_line,
        nx=header.nx,
        ny=header.ny,
        nz=header.nz,
    )

    profile = v_xfast
    if _profile_roughness(v_zfast) < _profile_roughness(v_xfast):
        profile = v_zfast

    z_step_bohr = float(header.vz_bohr[2])
    if np.isclose(z_step_bohr, 0.0):
        z_step_bohr = float(np.linalg.norm(header.vz_bohr))
        LOGGER.warning(
            "Detected near-zero z component in cube grid vector for '%s'; using |vz|=%.6g bohr.",
            cube_path,
            z_step_bohr,
        )
    z_bohr = header.origin_bohr[2] + np.arange(header.nz, dtype=float) * z_step_bohr
    z_ang = z_bohr * BOHR_TO_ANG
    return z_ang, profile, header


def water_z_bounds_ang(header: CubeHeader) -> tuple[float, float] | None:
    """Return water-like (O/H) z bounds in Angstrom from cube atom list."""
    mask = np.isin(header.atom_numbers, tuple(_WATER_ATOMIC_NUMBERS))
    if not np.any(mask):
        return None

    z_oh = header.atom_z_bohr[mask] * BOHR_TO_ANG
    return float(np.min(z_oh)), float(np.max(z_oh))


def _water_bulk_potential_ev(
    *,
    z_ang: np.ndarray,
    v_xyavg_ev: np.ndarray,
    water_z_min_ang: float,
    water_z_max_ang: float,
    padding_ang: float,
) -> tuple[float | None, float | None, float | None]:
    if water_z_min_ang > water_z_max_ang:
        water_z_min_ang, water_z_max_ang = water_z_max_ang, water_z_min_ang

    paddings = [float(padding_ang)]
    if padding_ang > 0.0:
        paddings.extend([float(padding_ang * 0.5), float(padding_ang * 0.25), 0.0])

    seen: set[float] = set()
    for pad in paddings:
        key = round(pad, 12)
        if key in seen:
            continue
        seen.add(key)

        z_min = water_z_min_ang + pad
        z_max = water_z_max_ang - pad
        if z_min >= z_max:
            continue

        mask = (z_ang >= z_min) & (z_ang <= z_max)
        if np.any(mask):
            return float(np.mean(v_xyavg_ev[mask])), float(z_min), float(z_max)

    if z_ang.size == 0:
        return None, None, None

    midpoint = 0.5 * (water_z_min_ang + water_z_max_ang)
    nearest_idx = int(np.argmin(np.abs(z_ang - midpoint)))
    LOGGER.warning(
        "Could not resolve a finite water-bulk averaging window; using nearest z-point fallback."
    )
    z_value = float(z_ang[nearest_idx])
    return float(v_xyavg_ev[nearest_idx]), z_value, z_value


def compute_potential_record(source: str | Path, *, config: PotentialConfig) -> PotentialRecord:
    """Compute cSHE-related quantities for one Hartree cube file."""
    validate_potential_config(config)

    source_path = validate_hartree_cube_source(source)

    source_dir = source_path.parent
    z_ang, v_xyavg_hartree, header = cube_xyavg_vs_z(source_path)
    if z_ang.size == 0:
        raise ValueError("Potential profile is empty.")

    v_xyavg_ev = np.asarray(v_xyavg_hartree, dtype=float) * HARTREE_TO_EV
    water_bounds = water_z_bounds_ang(header)

    water_bulk_potential_ev: float | None = None

    if water_bounds is None:
        LOGGER.warning(
            "No O/H atoms found in cube header for '%s'; water-bulk potential unavailable.",
            _compact_path(source_path),
        )
        LOGGER.debug("No O/H atoms found for source: %s", source_path)
    else:
        water_z_min_ang, water_z_max_ang = water_bounds
        (
            water_bulk_potential_ev,
            _water_bulk_z_min_ang,
            _water_bulk_z_max_ang,
        ) = _water_bulk_potential_ev(
            z_ang=z_ang,
            v_xyavg_ev=v_xyavg_ev,
            water_z_min_ang=water_z_min_ang,
            water_z_max_ang=water_z_max_ang,
            padding_ang=float(config.water_padding_ang),
        )

    output_out = find_cp2k_output_file(source_dir)
    efermi_ev = parse_fermi_ev(output_out) if output_out is not None else None

    electrode_cshe_ev = None
    if efermi_ev is not None and water_bulk_potential_ev is not None:
        electrode_cshe_ev = water_bulk_potential_ev - efermi_ev - float(config.cshe_offset_ev)

    status = "ok" if electrode_cshe_ev is not None else "incomplete"
    if efermi_ev is None:
        LOGGER.warning(
            "Could not parse Fermi energy for '%s'; electrode cSHE remains unavailable.",
            _compact_path(source_path),
        )
        LOGGER.debug("Fermi parsing failed for source: %s", source_path)
    if water_bulk_potential_ev is None:
        LOGGER.warning(
            "Could not resolve water-bulk potential for '%s'; electrode cSHE remains unavailable.",
            _compact_path(source_path),
        )
        LOGGER.debug("Water-bulk resolution failed for source: %s", source_path)

    return PotentialRecord(
        id=None,
        source=str(source_path),
        source_dir=_display_source_dir(source_path),
        output_out=_display_output_out(source_dir, output_out),
        efermi_ev=efermi_ev,
        water_bulk_potential_ev=water_bulk_potential_ev,
        electrode_cshe_ev=electrode_cshe_ev,
        status=status,
        error=None,
    )


def error_record_for_source(source: str | Path, error: str) -> PotentialRecord:
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
    )


def _resolve_worker_count(threads: int | None, n_sources: int) -> int:
    if n_sources <= 0:
        return 1

    if threads is None:
        threads = min(_DEFAULT_POTENTIAL_THREAD_CAP, os.cpu_count() or 1)
    if threads < 1:
        raise ValueError("Potential threads must be >= 1.")
    return min(threads, n_sources)


def compute_potential_records(
    sources: Sequence[str | Path],
    *,
    config: PotentialConfig,
    threads: int | None = None,
    on_record: PotentialRecordCallback | None = None,
    on_failure: PotentialFailureCallback | None = None,
) -> tuple[list[PotentialRecord], list[PotentialComputationFailure]]:
    """Compute potential records for one or many Hartree cube files.

    Parameters
    ----------
    sources
        Input Hartree cube files.
    config
        Potential-analysis configuration.
    threads
        Optional worker count. If ``None``, LiNaK derives a bounded default.
    on_record
        Optional callback invoked for every successful record.
    on_failure
        Optional callback invoked for every failed source.

    Returns
    -------
    tuple[list[PotentialRecord], list[PotentialComputationFailure]]
        Ordered successful records and ordered failures.
    """
    if not sources:
        raise ValueError("At least one source path is required.")

    worker_count = _resolve_worker_count(threads, len(sources))
    records_by_index: dict[int, PotentialRecord] = {}
    failures_by_index: dict[int, PotentialComputationFailure] = {}

    def _compute_one(
        index: int,
        source_item: str | Path,
    ) -> tuple[int, PotentialRecord | None, PotentialComputationFailure | None]:
        try:
            record = compute_potential_record(source_item, config=config)
            return index, record, None
        except Exception as exc:
            failure = PotentialComputationFailure(source=str(source_item), error=str(exc))
            return index, None, failure

    if worker_count > 1:
        LOGGER.info(
            "Using %d thread(s) for potential computation across %d file(s).",
            worker_count,
            len(sources),
        )

    with ProgressBar(desc="Computing potentials", total=len(sources), unit="file") as progress:
        if worker_count == 1:
            for index, source in enumerate(sources):
                idx, record, failure = _compute_one(index, source)
                if record is not None:
                    records_by_index[idx] = record
                    if on_record is not None:
                        on_record(record)
                if failure is not None:
                    failures_by_index[idx] = failure
                    if on_failure is not None:
                        on_failure(failure)
                progress.update()
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_compute_one, index, source): index
                    for index, source in enumerate(sources)
                }
                for future in as_completed(futures):
                    idx, record, failure = future.result()
                    if record is not None:
                        records_by_index[idx] = record
                        if on_record is not None:
                            on_record(record)
                    if failure is not None:
                        failures_by_index[idx] = failure
                        if on_failure is not None:
                            on_failure(failure)
                    progress.update()

    ordered_records = [records_by_index[index] for index in sorted(records_by_index)]
    ordered_failures = [failures_by_index[index] for index in sorted(failures_by_index)]
    return ordered_records, ordered_failures


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


def load_potential_plot_profiles(
    source: str | Path,
) -> tuple[list[PotentialPlotSeries], dict[str, Any]]:
    """Load a potential HDF5 file into fixed plotting series."""
    require_h5py()
    import h5py

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")

    required_columns = (
        "id",
        "efermi_ev",
        "water_bulk_potential_ev",
        "electrode_cshe_ev",
        "status",
        "source",
        "source_dir",
    )
    ordered_series_values: dict[str, np.ndarray] = {}
    with h5py.File(source_path, "r") as handle:
        if str(handle.attrs.get("analysis", "")) != "potential":
            raise ValueError(f"HDF5 analysis mismatch for '{source_path}': expected 'potential'.")
        if "records" not in handle:
            raise ValueError(f"Potential HDF5 '{source_path}' is missing '/records'.")
        records = handle["records"]
        missing = [column for column in required_columns if column not in records]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Potential HDF5 '{source_path}' is missing required column(s): {joined}."
            )

        total_rows = int(records["id"].shape[0])
        raw_ids = np.asarray(records["id"], dtype=np.int64)
        finite_positive_ids = raw_ids.size > 0 and bool(np.all(raw_ids > 0))
        unique_positive_ids = finite_positive_ids and len(np.unique(raw_ids)) == len(raw_ids)
        if finite_positive_ids and unique_positive_ids:
            order = np.argsort(raw_ids, kind="mergesort")
            x_values = np.asarray(raw_ids[order], dtype=float)
        else:
            order = np.arange(total_rows, dtype=np.int64)
            x_values = np.arange(1, total_rows + 1, dtype=float)

        for column in (
            "water_bulk_potential_ev",
            "efermi_ev",
            "electrode_cshe_ev",
        ):
            values = np.asarray(records[column], dtype=float)
            ordered_series_values[column] = np.asarray(values[order], dtype=float)

        status_values = np.asarray(records["status"].asstr()[...], dtype=object)
        ordered_status = [str(status_values[index]).strip().lower() for index in order]
        complete_rows = sum(1 for value in ordered_status if value == "ok")
        incomplete_rows = total_rows - complete_rows

    series_specs = (
        ("water_bulk_potential_ev", "Water bulk"),
        ("efermi_ev", "Fermi"),
        ("electrode_cshe_ev", "cSHE"),
    )
    profiles = [
        PotentialPlotSeries(
            series_id=series_id,
            default_label=default_label,
            x_values=np.asarray(x_values, dtype=float),
            y_values=ordered_series_values[series_id],
            source_path=str(source_path),
            total_rows=total_rows,
            complete_rows=complete_rows,
            incomplete_rows=incomplete_rows,
        )
        for series_id, default_label in series_specs
    ]
    return profiles, {
        "x_axis_label": "Record ID",
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "incomplete_rows": incomplete_rows,
    }


def plot_potential_profiles(
    profiles: list[PotentialPlotSeries],
    *,
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
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_fit_enabled: list[bool] | None = None,
    series_fit_labels: list[str | None] | None = None,
    series_fit_show_in_legend: list[bool] | None = None,
    series_normalization_modes: list[str] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
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

    labels = resolve_series_labels(
        [profile.default_label for profile in profiles],
        series_labels,
        series_kind="potential",
    )
    x_series = [np.asarray(profile.x_values, dtype=float) for profile in profiles]
    y_series = [np.asarray(profile.y_values, dtype=float) for profile in profiles]
    output_path = plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or "Hartree potential summary",
        x_label=x_label or "Record ID",
        y_label=y_label or "Potential (eV)",
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_ids=series_ids or [profile.series_id for profile in profiles],
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_fit_enabled=series_fit_enabled,
        series_fit_labels=series_fit_labels,
        series_fit_show_in_legend=series_fit_show_in_legend,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        line_kwargs=line_kwargs,
        series_line_kwargs=series_line_kwargs,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
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
            "total_rows": int(profiles[0].total_rows),
            "complete_rows": int(profiles[0].complete_rows),
            "incomplete_rows": int(profiles[0].incomplete_rows),
        }
    return output_path


def _read_csv_header(path: Path) -> list[str]:
    require_h5py()
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("linak_format", "")) != LINAK_HDF5_FORMAT:
                return []
            if str(handle.attrs.get("analysis", "")) != "potential":
                return []

            columns_raw = handle.attrs.get("columns_json")
            columns: list[str] = []
            if columns_raw is not None:
                if isinstance(columns_raw, bytes):
                    columns_raw = columns_raw.decode("utf-8", errors="replace")
                decoded = json.loads(str(columns_raw))
                if isinstance(decoded, list):
                    columns = [str(item) for item in decoded]

            if not columns:
                if "records" not in handle:
                    return []
                columns = [str(name) for name in handle["records"].keys()]

            return [column for column in columns if column in POTENTIAL_CSV_COLUMNS]
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _can_append_to_header(header: list[str]) -> bool:
    if not header:
        return True
    return all(column in header for column in POTENTIAL_CSV_COLUMNS)


def _is_compatible_potential_hdf5(path: Path) -> bool:
    header = _read_csv_header(path)
    if not header:
        return False
    return _can_append_to_header(header)


def _read_existing_source_keys(path: Path, *, header: list[str] | None = None) -> set[str]:
    columns = header if header is not None else _read_csv_header(path)
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
    columns = _read_csv_header(path)
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


def _fallback_csv_path(path: Path) -> Path:
    suffix = path.suffix if path.suffix.lower() in {".h5", ".hdf5"} else ".h5"
    stem = path.stem if path.suffix else path.name
    candidate = path.with_name(f"{stem}_linak_potential{suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{stem}_linak_potential_{counter}{suffix}")
        counter += 1
    return candidate


def plan_potential_csv_output(
    output: str | Path,
    *,
    append: bool = True,
    overwrite: bool = False,
) -> PotentialCsvPlan:
    """Plan HDF5 output strategy and collect existing source keys when appendable."""
    output_path = resolve_hdf5_output_path(output)
    target_path = output_path
    mode = "w"
    fieldnames = POTENTIAL_CSV_COLUMNS
    used_fallback = False
    existing_sources: set[str] = set()

    if overwrite:
        return PotentialCsvPlan(
            target_path=target_path,
            mode=mode,
            fieldnames=fieldnames,
            used_fallback_path=False,
            existing_source_keys=set(),
        )

    if not output_path.exists():
        return PotentialCsvPlan(
            target_path=target_path,
            mode=mode,
            fieldnames=fieldnames,
            used_fallback_path=False,
            existing_source_keys=set(),
        )

    if append:
        if _is_compatible_potential_hdf5(output_path):
            header = _read_csv_header(output_path)
            mode = "a"
            fieldnames = header if header else POTENTIAL_CSV_COLUMNS
            existing_sources = _read_existing_source_keys(output_path, header=header)
            return PotentialCsvPlan(
                target_path=target_path,
                mode=mode,
                fieldnames=fieldnames,
                used_fallback_path=False,
                existing_source_keys=existing_sources,
            )

    target_path = _fallback_csv_path(output_path)
    used_fallback = True
    return PotentialCsvPlan(
        target_path=target_path,
        mode="w",
        fieldnames=POTENTIAL_CSV_COLUMNS,
        used_fallback_path=used_fallback,
        existing_source_keys=set(),
    )


def write_potential_records_csv(
    records: list[PotentialRecord],
    output: str | Path,
    *,
    append: bool = True,
    overwrite: bool = False,
) -> PotentialCsvWriteResult:
    """Write potential records to HDF5 with schema-aware append/fallback behavior."""
    if not records:
        raise ValueError("At least one potential record is required for HDF5 export.")

    with PotentialCsvAppender(
        output=output,
        append=append,
        overwrite=overwrite,
        sync_on_write=True,
    ) as appender:
        appender.append_records(records)
        target_path = appender.path
        mode = appender.mode
        used_fallback = appender.used_fallback_path
        rows_written = appender.rows_written

    LOGGER.info("Saved %d potential row(s) to '%s'.", rows_written, _compact_path(target_path))
    LOGGER.debug("Potential HDF5 path: %s", target_path)
    return PotentialCsvWriteResult(
        path=target_path,
        rows_written=rows_written,
        mode=mode,
        used_fallback_path=used_fallback,
    )
