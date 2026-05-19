"""Unified LiNaK simulation-output HDF5 container support."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
import csv
import fnmatch
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Literal, Optional

import numpy as np
from ase import Atoms

from . import __version__
from .cube_io import CubeDataset, is_linak_cube_hdf5, read_cube_sources, save_cube_datasets
from .storage.compress import (
    FORCES_FIELDS,
    HIRSHFELD_FIELDS,
    MD_STEPS_FIELDS,
    MULLIKEN_FIELDS,
    SCF_ITERATION_FIELDS,
    CP2KOutputParser,
    ParserOptions,
    build_parser_options_from_drop_sections,
)
from .storage.hdf5_utils import require_h5py

LOGGER = logging.getLogger(__name__)

LINAK_OUT_HDF5_FORMAT = "linak-out-hdf5"
LINAK_OUT_HDF5_SCHEMA_VERSION = 1

OutH5Component = Literal["trajectory", "cube"]
ProgressCallback = Callable[[str, int, Optional[int]], None]
LogCallback = Callable[[str, str], None]
CancelCallback = Callable[[], bool]

_RAW_TRAJECTORY_SUFFIXES = (".xyz", ".extxyz", ".dump", ".lmp")
_CUBE_SUFFIXES = (".cube", ".cube.h5", ".cube.hdf5")
_CP2K_OUTPUT_SUFFIXES = (".out",)


@dataclass(frozen=True)
class OutH5PackOptions:
    """Options for directory-to-`.out.h5` packing."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    overwrite: bool = False
    drop_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutH5Discovery:
    """Lightweight directory discovery result."""

    source_dir: Path
    trajectories: tuple[Path, ...] = ()
    cubes: tuple[Path, ...] = ()
    cp2k_outputs: tuple[Path, ...] = ()
    skipped: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class OutH5Summary:
    """Metadata-only summary of one `.out.h5` file."""

    path: Path
    schema_version: int
    source_directory: str
    trajectory_present: bool
    frame_count: int | None = None
    atom_count: int | None = None
    cube_count: int = 0
    cp2k_output_count: int = 0
    singlepoint_sections: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    cell_angstrom: tuple[float, float, float] | None = None
    timestep_fs: float | None = None
    trajectory_source_path: str = ""
    trajectory_source_format: str = ""
    cell_matrix_angstrom: tuple[tuple[float, float, float], ...] = ()
    pbc: tuple[bool, bool, bool] | None = None
    timestep_candidates_fs: tuple[float, ...] = ()
    frame_range: tuple[int, int] | None = None
    cube_kinds: tuple[str, ...] = ()
    cube_source_names: tuple[str, ...] = ()
    cp2k_table_counts: dict[str, int] = dataclass_field(default_factory=dict)
    provenance_messages: tuple[str, ...] = ()
    discovery_summary: dict[str, Any] = dataclass_field(default_factory=dict)
    parser_coverage: dict[str, Any] = dataclass_field(default_factory=dict)


class OutH5PackCanceled(RuntimeError):
    """Raised when a pack operation is cooperatively canceled."""


def _check_canceled(cancel_requested: CancelCallback | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise OutH5PackCanceled("Operation canceled by user.")


@dataclass(frozen=True)
class OutH5PackResult:
    """Result returned after packing a simulation directory."""

    output_path: Path
    discovery: OutH5Discovery
    summary: OutH5Summary
    messages: tuple[str, ...] = ()


def _h5py() -> Any:
    require_h5py()
    import h5py

    return h5py


def _string_dtype() -> Any:
    h5py = _h5py()
    return h5py.string_dtype(encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _json_attr(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _loads_json_attr(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(_decode_attr(value)))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _safe_id(path: Path, index: int) -> str:
    stem = "".join(ch if ch.isalnum() else "_" for ch in path.stem).strip("_") or "item"
    return f"{index:04d}_{stem[:48]}"


def _infer_cube_kind(name: str) -> str:
    lower = name.lower()
    for token in ("hartree", "potential", "density", "charge", "spin"):
        if token in lower:
            return token
    return "cube"


def _path_matches(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    rel = path.relative_to(root).as_posix()
    return any(
        fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns
    )


def _has_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in suffixes)


def _is_existing_trajectory_hdf5(path: Path) -> bool:
    try:
        from .trajectory.io import is_linak_trajectory_hdf5

        return bool(is_linak_trajectory_hdf5(path))
    except Exception:
        return False


def default_out_h5_output_path(source_dir: str | Path) -> Path:
    """Return `<source_dir_name>.out.h5` next to the source directory."""

    source_path = Path(source_dir).expanduser().resolve()
    return source_path.with_name(f"{source_path.name}.out.h5")


def unique_out_h5_output_path(path: str | Path) -> Path:
    """Return a non-existing `.out.h5` path by appending `_N` before the suffix."""

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return target
    suffix = ".out.h5" if target.name.lower().endswith(".out.h5") else target.suffix
    base = target.with_name(target.name[: -len(suffix)] if suffix else target.stem)
    counter = 1
    while True:
        candidate = Path(f"{base}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def discover_simulation_directory(
    source_dir: str | Path,
    *,
    options: OutH5PackOptions | None = None,
) -> OutH5Discovery:
    """Recursively discover LiNaK-packable files under a simulation directory."""

    resolved = Path(source_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Simulation directory not found: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Simulation source must be a directory: {resolved}")

    resolved_options = options or OutH5PackOptions()
    trajectories: list[Path] = []
    cubes: list[Path] = []
    cp2k_outputs: list[Path] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(p for p in resolved.rglob("*") if p.is_file()):
        if resolved_options.include and not _path_matches(path, resolved, resolved_options.include):
            continue
        if _path_matches(path, resolved, resolved_options.exclude):
            skipped.append({"path": str(path), "reason": "excluded by pattern"})
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(".out.h5"):
            skipped.append({"path": str(path), "reason": "existing out_hdf5 container"})
        elif _is_existing_trajectory_hdf5(path) or _has_suffix(path, _RAW_TRAJECTORY_SUFFIXES):
            trajectories.append(path)
        elif _has_suffix(path, _CUBE_SUFFIXES) or is_linak_cube_hdf5(path):
            cubes.append(path)
        elif _has_suffix(path, _CP2K_OUTPUT_SUFFIXES):
            cp2k_outputs.append(path)

    return OutH5Discovery(
        source_dir=resolved,
        trajectories=tuple(trajectories),
        cubes=tuple(cubes),
        cp2k_outputs=tuple(cp2k_outputs),
        skipped=tuple(skipped),
    )


def _write_string_dataset(group: Any, name: str, values: Iterable[Any]) -> None:
    data = np.asarray([str(value) for value in values], dtype=object)
    group.create_dataset(name, data=data, dtype=_string_dtype())


def _write_table(
    group: Any, name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    if not rows:
        return
    table = group.create_group(name)
    for field in fields:
        values = [row.get(field) for row in rows]
        numeric_values: list[float] = []
        numeric = True
        integer = True
        for value in values:
            if value is None or value == "":
                numeric_values.append(np.nan)
                integer = False
                continue
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                numeric = False
                break
            numeric_values.append(as_float)
            if not float(as_float).is_integer():
                integer = False
        if numeric:
            if integer and all(np.isfinite(v) for v in numeric_values):
                table.create_dataset(
                    field, data=np.asarray(numeric_values, dtype=np.int64), compression="lzf"
                )
            else:
                table.create_dataset(
                    field, data=np.asarray(numeric_values, dtype=np.float64), compression="lzf"
                )
        else:
            _write_string_dataset(
                table, field, ["" if value is None else value for value in values]
            )
    table.attrs["row_count"] = len(rows)
    table.attrs["columns_json"] = _json_attr(list(fields))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _copy_dataset(source_group: Any, target_group: Any, name: str) -> bool:
    if name not in source_group:
        return False
    dataset = source_group[name]
    kwargs: dict[str, Any] = {}
    if getattr(dataset, "ndim", 0) > 0 and getattr(dataset, "size", 0) > 0:
        kwargs["compression"] = "lzf"
        if dataset.dtype.kind in {"i", "u", "f", "b"}:
            kwargs["shuffle"] = True
    target_group.create_dataset(name, data=dataset[...], **kwargs)
    return True


def _write_existing_trajectory_hdf5(source: Path, group: Any) -> None:
    h5py = _h5py()
    with h5py.File(source, "r") as handle:
        for name in ("positions", "cell", "pbc", "atomic_numbers", "atom_counts"):
            _copy_dataset(handle, group, name)
        if "frame_info" in handle:
            info_target = group.create_group("frame_info")
            for name in handle["frame_info"].keys():
                _copy_dataset(handle["frame_info"], info_target, name)
        for key, value in handle.attrs.items():
            if key in {"frame_count", "atom_count", "topology_mode"}:
                group.attrs[key] = value
    group.attrs["present"] = True
    group.attrs["source_path"] = str(source)
    group.attrs["source_format"] = "trajectory_hdf5"


def _topology_is_fixed(frames: Sequence[Atoms]) -> bool:
    if not frames:
        return False
    atomic_numbers = frames[0].get_atomic_numbers().tolist()
    return all(frame.get_atomic_numbers().tolist() == atomic_numbers for frame in frames)


def _write_frames_trajectory(source: Path, group: Any) -> None:
    from .trajectory.io import read_trajectory

    frames = read_trajectory(source)
    if not frames:
        raise ValueError(f"No trajectory frames were read from {source}.")
    frame_count = len(frames)
    atom_counts = np.asarray([len(frame) for frame in frames], dtype=np.int64)
    max_atoms = int(np.max(atom_counts))
    fixed = _topology_is_fixed(frames)

    positions = np.zeros((frame_count, max_atoms, 3), dtype=np.float64)
    cells = np.empty((frame_count, 3, 3), dtype=np.float64)
    pbc = np.empty((frame_count, 3), dtype=bool)
    atomic_numbers = (
        np.asarray(frames[0].get_atomic_numbers(), dtype=np.int64)
        if fixed
        else np.zeros((frame_count, max_atoms), dtype=np.int64)
    )
    frame_info: dict[str, list[float | int]] = {}
    info_keys = (
        "timestep",
        "frame_timestep_fs",
        "md_timestep_fs",
        "trajectory_stride_md",
        "time_fs",
    )
    for index, frame in enumerate(frames):
        count = len(frame)
        positions[index, :count] = np.asarray(frame.get_positions(), dtype=np.float64)
        cells[index] = np.asarray(frame.cell.array, dtype=np.float64)
        pbc[index] = np.asarray(frame.get_pbc(), dtype=bool)
        if not fixed:
            atomic_numbers[index, :count] = np.asarray(frame.get_atomic_numbers(), dtype=np.int64)
        for key in info_keys:
            value = frame.info.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                frame_info.setdefault(key, [np.nan] * frame_count)
                frame_info[key][index] = float(value)

    chunk_frames = max(1, min(frame_count, 64))
    group.create_dataset(
        "positions",
        data=positions,
        chunks=(chunk_frames, max_atoms, 3),
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset(
        "cell", data=cells, chunks=(chunk_frames, 3, 3), compression="lzf", shuffle=True
    )
    group.create_dataset("pbc", data=pbc, chunks=(chunk_frames, 3), compression="lzf")
    group.create_dataset(
        "atomic_numbers",
        data=atomic_numbers,
        compression="lzf" if not fixed else None,
        shuffle=not fixed,
    )
    if not fixed:
        group.create_dataset(
            "atom_counts", data=atom_counts, chunks=(chunk_frames,), compression="lzf", shuffle=True
        )
    if frame_info:
        info_group = group.create_group("frame_info")
        for key, values in frame_info.items():
            info_group.create_dataset(
                key, data=np.asarray(values, dtype=np.float64), compression="lzf", shuffle=True
            )
    group.attrs["present"] = True
    group.attrs["source_path"] = str(source)
    group.attrs["source_format"] = source.suffix.lower().lstrip(".") or "trajectory"
    group.attrs["frame_count"] = frame_count
    group.attrs["atom_count"] = max_atoms
    group.attrs["topology_mode"] = "fixed" if fixed else "variable"


def _write_trajectory(source: Path | None, group: Any, messages: list[str]) -> None:
    if source is None:
        group.attrs["present"] = False
        messages.append("No trajectory source was detected.")
        return
    try:
        if _is_existing_trajectory_hdf5(source):
            _write_existing_trajectory_hdf5(source, group)
        else:
            _write_frames_trajectory(source, group)
        messages.append(f"Packed trajectory: {source}")
    except Exception as exc:
        group.attrs["present"] = False
        group.attrs["error"] = str(exc)
        messages.append(f"Skipped trajectory {source}: {exc}")


def _write_cube_dataset(group: Any, dataset: CubeDataset, source: Path, index: int) -> None:
    cube_group = group.create_group(_safe_id(source, index))
    cube_group.create_dataset("origin_bohr", data=np.asarray(dataset.origin_bohr, dtype=float))
    cube_group.create_dataset(
        "grid_counts_signed", data=np.asarray(dataset.grid_counts_signed, dtype=int)
    )
    cube_group.create_dataset(
        "grid_vectors_bohr", data=np.asarray(dataset.grid_vectors_bohr, dtype=float)
    )
    cube_group.create_dataset("atom_numbers", data=np.asarray(dataset.atom_numbers, dtype=int))
    cube_group.create_dataset("atom_charges", data=np.asarray(dataset.atom_charges, dtype=float))
    cube_group.create_dataset(
        "atom_positions_bohr", data=np.asarray(dataset.atom_positions_bohr, dtype=float)
    )
    cube_group.create_dataset(
        "values", data=np.asarray(dataset.values, dtype=float), compression="lzf", shuffle=True
    )
    cube_group.attrs["comment_1"] = dataset.comment_1
    cube_group.attrs["comment_2"] = dataset.comment_2
    cube_group.attrs["natoms_signed"] = int(dataset.natoms_signed)
    cube_group.attrs["source_path"] = dataset.source_path or str(source)
    cube_group.attrs["source_name"] = dataset.source_name or source.name
    cube_group.attrs["source_file_type"] = dataset.source_file_type or (
        "cube_hdf5" if is_linak_cube_hdf5(source) else "cube_file"
    )
    cube_group.attrs["source_profile_index"] = (
        -1 if dataset.source_profile_index is None else int(dataset.source_profile_index)
    )
    cube_group.attrs["parse_status"] = "ok"


def _write_cubes(sources: Sequence[Path], group: Any, messages: list[str]) -> int:
    count = 0
    for source in sources:
        try:
            datasets = read_cube_sources(source)
            for dataset in datasets:
                _write_cube_dataset(group, dataset, source, count)
                count += 1
            messages.append(f"Packed {len(datasets)} cube dataset(s): {source}")
        except Exception as exc:
            skipped = group.require_group("_skipped")
            skipped.attrs[str(source)] = str(exc)
            messages.append(f"Skipped cube {source}: {exc}")
    group.attrs["count"] = count
    return count


def _parser_options(drop_sections: Sequence[str]) -> ParserOptions:
    return build_parser_options_from_drop_sections(sorted(set(drop_sections)))


def _write_cp2k_output_tables(
    source: Path,
    parent: Any,
    *,
    options: ParserOptions,
    index: int,
) -> tuple[int, tuple[str, ...]]:
    output_group = parent.create_group(_safe_id(source, index))
    output_group.attrs["source_path"] = str(source)
    output_group.attrs["source_name"] = source.name
    sections: list[str] = []
    with tempfile.TemporaryDirectory(prefix="linak_out_h5_cp2k_") as tmp:
        tmp_path = Path(tmp)
        parser = CP2KOutputParser(source, None, tmp_path, options)
        result = parser.parse()
        tables = (
            (
                "scf_iterations",
                tmp_path / "scf_iterations.csv",
                result.scf_iterations,
                SCF_ITERATION_FIELDS,
            ),
            ("mulliken", tmp_path / "mulliken.csv", result.mulliken_rows, MULLIKEN_FIELDS),
            ("hirshfeld", tmp_path / "hirshfeld.csv", result.hirshfeld_rows, HIRSHFELD_FIELDS),
            ("forces", tmp_path / "forces.csv", result.forces_rows, FORCES_FIELDS),
            ("md_steps", tmp_path / "md_steps.csv", result.md_steps, MD_STEPS_FIELDS),
        )
        for name, csv_path, memory_rows, fields in tables:
            rows = _read_csv_rows(csv_path) or list(memory_rows)
            if rows:
                _write_table(output_group, name, rows, fields)
                sections.append(name)
        _write_table(
            output_group,
            "cell",
            result.cell_rows,
            (
                "prefix",
                "a_x",
                "a_y",
                "a_z",
                "a_len",
                "b_x",
                "b_y",
                "b_z",
                "b_len",
                "c_x",
                "c_y",
                "c_z",
                "c_len",
            ),
        )
        if result.atomic_kinds:
            _write_table(
                output_group,
                "atomic_kinds",
                result.atomic_kinds,
                ("kind_index", "kind", "atom_count"),
            )
        if result.warnings_counter:
            warning_group = output_group.create_group("warnings")
            _write_string_dataset(warning_group, "message", result.warnings_counter.keys())
            warning_group.create_dataset(
                "count", data=np.asarray(list(result.warnings_counter.values()), dtype=np.int64)
            )
        output_group.attrs["sections_json"] = _json_attr(sections)
        output_group.attrs["parse_status"] = "ok"
        return len(sections), tuple(sections)


def _write_cp2k_outputs(
    sources: Sequence[Path],
    group: Any,
    *,
    options: OutH5PackOptions,
    messages: list[str],
) -> tuple[int, tuple[str, ...]]:
    cp2k_group = group.require_group("cp2k")
    parser_options = _parser_options(options.drop_sections)
    all_sections: set[str] = set()
    parsed = 0
    for index, source in enumerate(sources):
        try:
            section_count, sections = _write_cp2k_output_tables(
                source, cp2k_group, options=parser_options, index=index
            )
            parsed += 1
            all_sections.update(sections)
            messages.append(f"Packed CP2K output: {source} ({section_count} section(s))")
        except Exception as exc:
            skipped = cp2k_group.require_group("_skipped")
            skipped.attrs[str(source)] = str(exc)
            messages.append(f"Skipped CP2K output {source}: {exc}")
    cp2k_group.attrs["output_count"] = parsed
    cp2k_group.attrs["sections_json"] = _json_attr(sorted(all_sections))
    return parsed, tuple(sorted(all_sections))


def _species_from_atomic_numbers(numbers: np.ndarray) -> tuple[str, ...]:
    try:
        from ase.data import chemical_symbols

        unique = sorted({int(value) for value in np.asarray(numbers).reshape(-1) if int(value) > 0})
        return tuple(
            chemical_symbols[number] for number in unique if number < len(chemical_symbols)
        )
    except Exception:
        return ()


def _write_system_group(handle: Any) -> None:
    group = handle.require_group("system")
    trajectory = handle.get("trajectory")
    if trajectory is not None and bool(trajectory.attrs.get("present", False)):
        numbers = np.asarray(trajectory["atomic_numbers"])
        species = _species_from_atomic_numbers(numbers)
        if species:
            _write_string_dataset(group, "species", species)
        group.attrs["frame_count"] = int(trajectory.attrs.get("frame_count", 0))
        group.attrs["atom_count"] = int(trajectory.attrs.get("atom_count", 0))
        if "cell" in trajectory:
            cell = np.asarray(trajectory["cell"][0], dtype=float)
            group.create_dataset("cell_angstrom", data=cell)
    group.attrs["units_json"] = _json_attr(
        {
            "trajectory_positions": "angstrom",
            "trajectory_cell": "angstrom",
            "cube_origin": "bohr",
            "cube_grid_vectors": "bohr",
            "cube_values": "source",
            "cp2k_energy": "hartree",
            "cp2k_force": "hartree_per_bohr",
        }
    )


def _write_provenance(
    handle: Any,
    *,
    discovery: OutH5Discovery,
    options: OutH5PackOptions,
    messages: Sequence[str],
) -> None:
    group = handle.require_group("provenance")
    group.attrs["source_directory"] = str(discovery.source_dir)
    group.attrs["schema_version"] = LINAK_OUT_HDF5_SCHEMA_VERSION
    group.attrs["conversion_settings_json"] = _json_attr(
        {
            "include": list(options.include),
            "exclude": list(options.exclude),
            "drop_sections": list(options.drop_sections),
        }
    )
    group.attrs["discovery_summary_json"] = _json_attr(
        {
            "trajectories": [str(path) for path in discovery.trajectories],
            "cubes": [str(path) for path in discovery.cubes],
            "cp2k_outputs": [str(path) for path in discovery.cp2k_outputs],
            "skipped": list(discovery.skipped),
        }
    )
    _write_string_dataset(group, "messages", messages)


def pack_simulation_directory(
    source_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    options: OutH5PackOptions | None = None,
    progress: ProgressCallback | None = None,
    logger: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
) -> OutH5PackResult:
    """Pack one simulation output directory into a schema-versioned `.out.h5` file."""

    resolved_options = options or OutH5PackOptions()
    if progress:
        progress("Discovering files", 0, 6)
    _check_canceled(cancel_requested)
    discovery = discover_simulation_directory(source_dir, options=resolved_options)
    _check_canceled(cancel_requested)
    if progress:
        progress("Discovery complete", 1, 6)
    target = (
        default_out_h5_output_path(discovery.source_dir)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if not target.name.lower().endswith(".out.h5"):
        target = target.with_suffix("").with_suffix(".out.h5")
    if target.exists() and not resolved_options.overwrite:
        target = unique_out_h5_output_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    log = logger or (lambda _level, message: LOGGER.info("%s", message))
    log("INFO", f"Packing simulation directory: {discovery.source_dir}")
    log("INFO", f"Output container: {target}")

    h5py = _h5py()
    with h5py.File(target, "w") as handle:
        handle.attrs["linak_format"] = LINAK_OUT_HDF5_FORMAT
        handle.attrs["linak_out_schema_version"] = LINAK_OUT_HDF5_SCHEMA_VERSION
        handle.attrs["linak_version"] = __version__
        handle.attrs["created_utc"] = _now_utc()
        handle.attrs["source_directory"] = str(discovery.source_dir)
        handle.attrs["schema_capabilities_json"] = _json_attr(
            {
                "trajectory": True,
                "cubes": True,
                "singlepoint": ["cp2k"],
                "hybrid_cache_compatible": True,
            }
        )

        if progress:
            progress("Packing trajectory", 1, 6)
        _check_canceled(cancel_requested)
        trajectory_group = handle.create_group("trajectory")
        primary_trajectory = discovery.trajectories[0] if discovery.trajectories else None
        _write_trajectory(primary_trajectory, trajectory_group, messages)
        for extra in discovery.trajectories[1:]:
            messages.append(f"Additional trajectory source not packed in v1: {extra}")
        if progress:
            progress("Trajectory packed", 2, 6)

        if progress:
            progress("Packing cubes", 2, 6)
        _check_canceled(cancel_requested)
        cubes_group = handle.create_group("cubes")
        cube_count = _write_cubes(discovery.cubes, cubes_group, messages)
        log("INFO", f"Packed cube dataset count: {cube_count}")
        if progress:
            progress("Cubes packed", 3, 6)

        if progress:
            progress("Packing CP2K outputs", 3, 6)
        _check_canceled(cancel_requested)
        singlepoint_group = handle.create_group("singlepoint")
        output_count, sections = _write_cp2k_outputs(
            discovery.cp2k_outputs,
            singlepoint_group,
            options=resolved_options,
            messages=messages,
        )
        log("INFO", f"Packed CP2K output count: {output_count}")
        if sections:
            log("INFO", "Singlepoint sections: " + ", ".join(sections))
        if progress:
            progress("Writing metadata", 4, 6)
        _check_canceled(cancel_requested)

        _write_system_group(handle)
        _write_provenance(
            handle,
            discovery=discovery,
            options=resolved_options,
            messages=messages,
        )
        if progress:
            progress("Finalizing container", 5, 6)

    _check_canceled(cancel_requested)
    summary = inspect_out_h5(target)
    if progress:
        progress("Finished", 6, 6)
    return OutH5PackResult(
        output_path=target,
        discovery=discovery,
        summary=summary,
        messages=tuple(messages),
    )


def is_linak_out_hdf5(path: str | Path) -> bool:
    """Return whether *path* is a LiNaK `.out.h5` container."""

    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return False
    if not candidate.name.lower().endswith((".out.h5", ".out.hdf5", ".h5", ".hdf5")):
        return False
    try:
        h5py = _h5py()
        with h5py.File(candidate, "r") as handle:
            return (
                str(_decode_attr(handle.attrs.get("linak_format", ""))).strip()
                == LINAK_OUT_HDF5_FORMAT
            )
    except Exception:
        return False


def inspect_out_h5(path: str | Path) -> OutH5Summary:
    """Read lightweight `.out.h5` metadata without loading large arrays."""

    source_path = Path(path).expanduser().resolve()
    h5py = _h5py()
    with h5py.File(source_path, "r") as handle:
        fmt = str(_decode_attr(handle.attrs.get("linak_format", ""))).strip()
        if fmt != LINAK_OUT_HDF5_FORMAT:
            raise ValueError(f"Unsupported LiNaK output container: {source_path}")
        trajectory = handle.get("trajectory")
        trajectory_present = bool(trajectory is not None and trajectory.attrs.get("present", False))
        frame_count = None
        atom_count = None
        trajectory_source_path = ""
        trajectory_source_format = ""
        frame_range = None
        pbc: tuple[bool, bool, bool] | None = None
        if trajectory_present and trajectory is not None:
            frame_count = int(
                trajectory.attrs.get(
                    "frame_count",
                    trajectory["positions"].shape[0] if "positions" in trajectory else 0,
                )
            )
            atom_count = int(
                trajectory.attrs.get(
                    "atom_count",
                    trajectory["positions"].shape[1] if "positions" in trajectory else 0,
                )
            )
            trajectory_source_path = str(_decode_attr(trajectory.attrs.get("source_path", "")))
            trajectory_source_format = str(_decode_attr(trajectory.attrs.get("source_format", "")))
            if "pbc" in trajectory and getattr(trajectory["pbc"], "shape", ()):
                first_pbc = np.asarray(trajectory["pbc"][0], dtype=bool).reshape(-1)
                if first_pbc.size >= 3:
                    pbc = tuple(bool(value) for value in first_pbc[:3])
            if frame_count and frame_count > 0:
                frame_range = (0, int(frame_count) - 1)
        cubes = handle.get("cubes")
        cube_count = int(cubes.attrs.get("count", 0)) if cubes is not None else 0
        cube_kinds: list[str] = []
        cube_source_names: list[str] = []
        if cubes is not None:
            for name, group in cubes.items():
                if str(name).startswith("_") or not hasattr(group, "attrs"):
                    continue
                source_name = str(_decode_attr(group.attrs.get("source_name", name)))
                cube_source_names.append(source_name)
                explicit_kind = str(_decode_attr(group.attrs.get("cube_kind", ""))).strip()
                cube_kinds.append(explicit_kind or _infer_cube_kind(source_name))
        cp2k = handle.get("singlepoint/cp2k")
        cp2k_output_count = int(cp2k.attrs.get("output_count", 0)) if cp2k is not None else 0
        sections = ()
        cp2k_table_counts: dict[str, int] = {}
        if cp2k is not None:
            sections = tuple(_loads_json_attr(cp2k.attrs.get("sections_json", "[]"), ()))
            for output_name, output_group in cp2k.items():
                if str(output_name).startswith("_") or not hasattr(output_group, "items"):
                    continue
                for table_name, table_group in output_group.items():
                    if str(table_name).startswith("_") or not hasattr(table_group, "items"):
                        continue
                    first_dataset = next(iter(table_group.values()), None)
                    if first_dataset is not None and hasattr(first_dataset, "shape"):
                        cp2k_table_counts[str(table_name)] = cp2k_table_counts.get(
                            str(table_name), 0
                        ) + int(first_dataset.shape[0])
        species: tuple[str, ...] = ()
        if "system/species" in handle:
            species = tuple(str(_decode_attr(value)) for value in handle["system/species"][...])
        cell_angstrom: tuple[float, float, float] | None = None
        cell_matrix_angstrom: tuple[tuple[float, float, float], ...] = ()
        if "system/cell_angstrom" in handle:
            raw_cell = np.asarray(handle["system/cell_angstrom"][...], dtype=float)
            if raw_cell.shape == (3, 3):
                cell_matrix_angstrom = tuple(
                    tuple(float(value) for value in row) for row in raw_cell
                )
                cell_angstrom = tuple(float(np.linalg.norm(row)) for row in raw_cell)
            elif raw_cell.size >= 3:
                cell_angstrom = tuple(float(value) for value in raw_cell.reshape(-1)[:3])
        timestep_fs: float | None = None
        timestep_candidates: list[float] = []
        if trajectory is not None and "frame_info" in trajectory:
            frame_info = trajectory["frame_info"]
            for key in ("frame_timestep_fs", "md_timestep_fs", "timestep", "time_fs"):
                if key not in frame_info:
                    continue
                dataset = frame_info[key]
                sample_count = min(int(dataset.shape[0]) if dataset.shape else 1, 16)
                values = np.asarray(dataset[:sample_count], dtype=float).reshape(-1)
                finite = values[np.isfinite(values)]
                if finite.size:
                    candidate = float(finite[0])
                    timestep_candidates.append(candidate)
                    if timestep_fs is None:
                        timestep_fs = candidate
        provenance_messages: tuple[str, ...] = ()
        if "provenance/messages" in handle:
            provenance_messages = tuple(
                str(_decode_attr(value)) for value in handle["provenance/messages"][...]
            )
        warnings = tuple(
            message
            for message in provenance_messages
            if "skipped" in message.lower() or "warning" in message.lower()
        )
        provenance = handle.get("provenance")
        discovery_summary: dict[str, Any] = {}
        if provenance is not None:
            raw_discovery = provenance.attrs.get("discovery_summary_json")
            loaded_discovery = _loads_json_attr(raw_discovery, {})
            if isinstance(loaded_discovery, dict):
                discovery_summary = loaded_discovery
        parser_coverage = {
            "trajectory": trajectory_present,
            "cubes": cube_count,
            "cp2k_outputs": cp2k_output_count,
            "singlepoint_sections": list(sections),
            "warnings": len(warnings),
        }
        return OutH5Summary(
            path=source_path,
            schema_version=int(handle.attrs.get("linak_out_schema_version", 0)),
            source_directory=str(_decode_attr(handle.attrs.get("source_directory", ""))),
            trajectory_present=trajectory_present,
            frame_count=frame_count,
            atom_count=atom_count,
            cube_count=cube_count,
            cp2k_output_count=cp2k_output_count,
            singlepoint_sections=sections,
            species=species,
            warnings=warnings,
            cell_angstrom=cell_angstrom,
            timestep_fs=timestep_fs,
            trajectory_source_path=trajectory_source_path,
            trajectory_source_format=trajectory_source_format,
            cell_matrix_angstrom=cell_matrix_angstrom,
            pbc=pbc,
            timestep_candidates_fs=tuple(dict.fromkeys(timestep_candidates)),
            frame_range=frame_range,
            cube_kinds=tuple(dict.fromkeys(cube_kinds)),
            cube_source_names=tuple(cube_source_names),
            cp2k_table_counts=cp2k_table_counts,
            provenance_messages=provenance_messages,
            discovery_summary=discovery_summary,
            parser_coverage=parser_coverage,
        )


def _build_atoms_from_group(
    *,
    positions_chunk: np.ndarray,
    cell_chunk: np.ndarray,
    pbc_chunk: np.ndarray,
    atomic_numbers: np.ndarray,
    atom_counts_chunk: np.ndarray | None,
    frame_info_chunks: Mapping[str, np.ndarray],
) -> list[Atoms]:
    frames: list[Atoms] = []
    for offset in range(positions_chunk.shape[0]):
        if atom_counts_chunk is None:
            numbers = np.asarray(atomic_numbers, dtype=np.int64)
            positions = positions_chunk[offset]
        else:
            atom_count = int(atom_counts_chunk[offset])
            numbers = np.asarray(atomic_numbers[offset, :atom_count], dtype=np.int64)
            positions = positions_chunk[offset, :atom_count]
        frame = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell_chunk[offset],
            pbc=tuple(bool(value) for value in pbc_chunk[offset]),
        )
        for key, values in frame_info_chunks.items():
            value = values[offset]
            if np.issubdtype(values.dtype, np.integer):
                frame.info[key] = int(value)
            elif np.issubdtype(values.dtype, np.floating) and np.isfinite(value):
                frame.info[key] = float(value)
        frames.append(frame)
    return frames


def read_out_h5_trajectory_chunks(path: str | Path, *, chunk_size: int) -> Iterator[list[Atoms]]:
    """Yield trajectory frames stored in `.out.h5`."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    source_path = Path(path).expanduser().resolve()
    h5py = _h5py()
    with h5py.File(source_path, "r") as handle:
        if str(_decode_attr(handle.attrs.get("linak_format", ""))).strip() != LINAK_OUT_HDF5_FORMAT:
            raise ValueError(f"Unsupported LiNaK output container: {source_path}")
        trajectory = handle.get("trajectory")
        if trajectory is None or not bool(trajectory.attrs.get("present", False)):
            raise ValueError(f"Output container does not contain trajectory data: {source_path}")
        positions = trajectory["positions"]
        cells = trajectory["cell"]
        pbc = trajectory["pbc"]
        atomic_numbers = np.asarray(trajectory["atomic_numbers"], dtype=np.int64)
        atom_counts_dataset = trajectory.get("atom_counts")
        frame_count = int(trajectory.attrs.get("frame_count", positions.shape[0]))
        info_group = trajectory.get("frame_info")
        info_names = list(info_group.keys()) if info_group is not None else []
        for start in range(0, frame_count, chunk_size):
            stop = min(start + chunk_size, frame_count)
            atom_counts_chunk = (
                None
                if atom_counts_dataset is None
                else np.asarray(atom_counts_dataset[start:stop], dtype=np.int64)
            )
            frame_info_chunks = (
                {key: np.asarray(info_group[key][start:stop]) for key in info_names}
                if info_group is not None
                else {}
            )
            yield _build_atoms_from_group(
                positions_chunk=np.asarray(positions[start:stop], dtype=np.float64),
                cell_chunk=np.asarray(cells[start:stop], dtype=np.float64),
                pbc_chunk=np.asarray(pbc[start:stop], dtype=bool),
                atomic_numbers=(
                    atomic_numbers
                    if atom_counts_chunk is None
                    else np.asarray(atomic_numbers[start:stop], dtype=np.int64)
                ),
                atom_counts_chunk=atom_counts_chunk,
                frame_info_chunks=frame_info_chunks,
            )


def read_out_h5_trajectory(path: str | Path) -> list[Atoms]:
    """Read all trajectory frames from `.out.h5`."""

    frames: list[Atoms] = []
    for chunk in read_out_h5_trajectory_chunks(path, chunk_size=256):
        frames.extend(chunk)
    return frames


def read_out_h5_cube_datasets(
    path: str | Path,
    *,
    selector: str | None = None,
) -> list[CubeDataset]:
    """Read cube datasets stored in `.out.h5`."""

    source_path = Path(path).expanduser().resolve()
    h5py = _h5py()
    datasets: list[CubeDataset] = []
    with h5py.File(source_path, "r") as handle:
        if str(_decode_attr(handle.attrs.get("linak_format", ""))).strip() != LINAK_OUT_HDF5_FORMAT:
            raise ValueError(f"Unsupported LiNaK output container: {source_path}")
        cubes = handle.get("cubes")
        if cubes is None:
            return []
        for name in sorted(cubes.keys()):
            if name.startswith("_"):
                continue
            if selector and selector not in name:
                continue
            group = cubes[name]
            datasets.append(
                CubeDataset(
                    comment_1=str(_decode_attr(group.attrs.get("comment_1", "LiNaK out.h5 cube"))),
                    comment_2=str(
                        _decode_attr(group.attrs.get("comment_2", "Generated by LiNaK pack"))
                    ),
                    natoms_signed=int(
                        group.attrs.get("natoms_signed", np.asarray(group["atom_numbers"]).shape[0])
                    ),
                    origin_bohr=np.asarray(group["origin_bohr"], dtype=float),
                    grid_counts_signed=np.asarray(group["grid_counts_signed"], dtype=int),
                    grid_vectors_bohr=np.asarray(group["grid_vectors_bohr"], dtype=float),
                    atom_numbers=np.asarray(group["atom_numbers"], dtype=int),
                    atom_charges=np.asarray(group["atom_charges"], dtype=float),
                    atom_positions_bohr=np.asarray(group["atom_positions_bohr"], dtype=float),
                    values=np.asarray(group["values"], dtype=float),
                    source_path=str(_decode_attr(group.attrs.get("source_path", source_path))),
                    source_name=str(_decode_attr(group.attrs.get("source_name", name))),
                    source_file_type="out_hdf5",
                    source_profile_index=int(group.attrs.get("source_profile_index", -1)),
                )
            )
    return datasets


def export_out_h5_component(
    path: str | Path,
    component: OutH5Component,
    output_path: str | Path,
) -> Path:
    """Export one user-visible component from `.out.h5`."""

    if component == "trajectory":
        from .trajectory.io import write_trajectory

        frames = read_out_h5_trajectory(path)
        return write_trajectory(
            frames,
            output_path,
            source_path=path,
            source_format="out_hdf5",
        )
    if component == "cube":
        datasets = read_out_h5_cube_datasets(path)
        if not datasets:
            raise ValueError(
                f"Output container has no cube data: {Path(path).expanduser().resolve()}"
            )
        return save_cube_datasets(
            datasets,
            output_path,
            additional_metadata={"source_path": str(Path(path).expanduser().resolve())},
        )
    raise ValueError(f"Unsupported `.out.h5` export component: {component}")
