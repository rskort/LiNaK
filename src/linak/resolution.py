"""Cell and timestep resolution helpers for trajectory analyses."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from ase import Atoms
import numpy as np

from .pbc import (
    extract_cell_from_simulation_input,
    extract_frame_timestep_fs_from_simulation_input,
    find_unique_simulation_input,
)
from .trajectory.io import read_trajectory_hdf5_metadata
from .utils import ensure_positive

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellResolution:
    """Resolved periodic cell with provenance."""

    cell_angstrom: tuple[float, float, float]
    source: str
    input_path: Path | None = None


@dataclass(frozen=True)
class TimestepResolution:
    """Resolved per-frame timestep with provenance."""

    frame_timestep_fs: float
    source: str
    input_path: Path | None = None
    md_timestep_fs: float | None = None
    trajectory_stride_md: int | None = None


def _normalize_cell(cell: tuple[float, float, float]) -> tuple[float, float, float]:
    normalized = (
        float(cell[0]),
        float(cell[1]),
        float(cell[2]),
    )
    ensure_positive("cell_a", normalized[0])
    ensure_positive("cell_b", normalized[1])
    ensure_positive("cell_c", normalized[2])
    return normalized


def _normalize_timestep_fs(timestep_fs: float) -> float:
    normalized = float(timestep_fs)
    ensure_positive("timestep_fs", normalized)
    return normalized


def _auto_detect_cell(trajectory_path: str | Path) -> tuple[tuple[float, float, float], Path]:
    trajectory = Path(trajectory_path).expanduser().resolve()
    input_path = find_unique_simulation_input(trajectory.parent)
    cell = extract_cell_from_simulation_input(input_path)
    return cell, input_path


def _auto_detect_frame_timestep_fs(trajectory_path: str | Path) -> tuple[float, float, int, Path]:
    trajectory = Path(trajectory_path).expanduser().resolve()
    input_path = find_unique_simulation_input(trajectory.parent)
    frame_timestep_fs, md_timestep_fs, stride_md = extract_frame_timestep_fs_from_simulation_input(
        input_path
    )
    return frame_timestep_fs, md_timestep_fs, stride_md, input_path


def _trajectory_hdf5_cell_metadata(
    trajectory_path: str | Path,
) -> tuple[tuple[float, float, float] | None, Path | None]:
    metadata = read_trajectory_hdf5_metadata(trajectory_path)
    if metadata is None or metadata.cell_angstrom is None:
        return None, None
    return metadata.cell_angstrom, metadata.input_path


def _trajectory_hdf5_timestep_metadata(
    trajectory_path: str | Path,
) -> tuple[float | None, float | None, int | None, Path | None]:
    metadata = read_trajectory_hdf5_metadata(trajectory_path)
    if metadata is None or metadata.frame_timestep_fs is None:
        return None, None, None, None
    return (
        metadata.frame_timestep_fs,
        metadata.md_timestep_fs,
        metadata.trajectory_stride_md,
        metadata.input_path,
    )


def _normalize_info_key(key: object) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _extract_constant_positive_step(values: list[float]) -> float | None:
    if len(values) < 2:
        return None

    diffs = np.diff(np.asarray(values, dtype=float))
    if diffs.size == 0:
        return None
    if np.any(diffs <= 0.0):
        return None

    step = float(np.median(diffs))
    tolerance = max(1e-12, abs(step) * 1e-6)
    if not np.allclose(diffs, step, rtol=0.0, atol=tolerance):
        return None
    return step


def _extract_metadata_timestep_details(
    frames: list[Atoms] | None,
) -> tuple[float | None, float | None, int | None]:
    """Infer timestep details from in-memory frame metadata."""
    if frames is None or len(frames) < 2:
        return None, None, None

    info_first = getattr(frames[0], "info", None)
    if not isinstance(info_first, dict) or not info_first:
        return None, None, None

    direct_key_units = {
        "timestepfs": 1.0,
        "dtfs": 1.0,
        "deltatfs": 1.0,
        "frametimestepfs": 1.0,
        "timestepps": 1000.0,
        "dtps": 1000.0,
        "deltatps": 1000.0,
        "frametimestepps": 1000.0,
    }
    absolute_time_key_units = {
        "timefs": 1.0,
        "tfs": 1.0,
        "timeps": 1000.0,
        "tps": 1000.0,
    }

    for key, value in info_first.items():
        factor = direct_key_units.get(_normalize_info_key(key))
        if factor is None:
            continue
        parsed = _coerce_float(value)
        if parsed is None or parsed <= 0.0:
            continue
        inferred = _normalize_timestep_fs(parsed * factor)
        md_timestep_fs = _coerce_float(info_first.get("md_timestep_fs"))
        stride_raw = info_first.get("trajectory_stride_md")
        stride_md = int(stride_raw) if isinstance(stride_raw, (int, np.integer)) else None
        LOGGER.debug(
            "Inferred frame timestep from trajectory metadata key '%s': %.6g fs.",
            key,
            inferred,
        )
        return inferred, md_timestep_fs, stride_md

    for key, factor in absolute_time_key_units.items():
        values: list[float] = []
        display_key: str | None = None
        for frame in frames:
            info = getattr(frame, "info", None)
            if not isinstance(info, dict):
                values = []
                break

            matched = False
            for candidate_key, candidate_value in info.items():
                if _normalize_info_key(candidate_key) != key:
                    continue
                parsed = _coerce_float(candidate_value)
                if parsed is None:
                    values = []
                    matched = True
                    break
                values.append(parsed)
                if display_key is None:
                    display_key = str(candidate_key)
                matched = True
                break
            if not matched:
                values = []
                break
            if not values:
                break

        if len(values) < 2:
            continue

        step = _extract_constant_positive_step(values)
        if step is None:
            continue

        inferred = _normalize_timestep_fs(step * factor)
        md_timestep_fs = _coerce_float(info_first.get("md_timestep_fs"))
        stride_raw = info_first.get("trajectory_stride_md")
        stride_md = int(stride_raw) if isinstance(stride_raw, (int, np.integer)) else None
        LOGGER.debug(
            "Inferred frame timestep from trajectory metadata key '%s': %.6g fs.",
            display_key or key,
            inferred,
        )
        return inferred, md_timestep_fs, stride_md

    return None, None, None


def _warn_on_mismatched_cells(
    trajectory: Path,
    candidates: list[tuple[str, tuple[float, float, float]]],
) -> None:
    for i, (name_a, cell_a) in enumerate(candidates):
        for name_b, cell_b in candidates[i + 1 :]:
            if np.allclose(cell_a, cell_b, rtol=0.0, atol=1e-6):
                continue
            LOGGER.warning(
                "Cell sources disagree for '%s': %s=%s, %s=%s.",
                trajectory,
                name_a,
                cell_a,
                name_b,
                cell_b,
            )


def _warn_on_mismatched_timestep_sources(
    trajectory: Path,
    candidates: list[tuple[str, float]],
) -> None:
    for i, (name_a, timestep_a) in enumerate(candidates):
        for name_b, timestep_b in candidates[i + 1 :]:
            if np.isclose(timestep_a, timestep_b, rtol=0.0, atol=1e-9):
                continue
            LOGGER.warning(
                "Timestep sources disagree for '%s': %s=%.6g fs, %s=%.6g fs.",
                trajectory,
                name_a,
                timestep_a,
                name_b,
                timestep_b,
            )


def resolve_analysis_cell(
    trajectory_path: str | Path,
    *,
    cell: tuple[float, float, float] | None = None,
    input_path: str | Path | None = None,
) -> CellResolution:
    """Resolve cell dimensions from explicit args or simulation input files."""
    trajectory = Path(trajectory_path).expanduser().resolve()

    explicit_cell = _normalize_cell(cell) if cell is not None else None
    explicit_input_cell = (
        extract_cell_from_simulation_input(input_path) if input_path is not None else None
    )
    explicit_input_resolved = (
        Path(input_path).expanduser().resolve() if input_path is not None else None
    )
    trajectory_hdf5_cell, trajectory_hdf5_input = _trajectory_hdf5_cell_metadata(trajectory)

    if explicit_cell is not None:
        candidates: list[tuple[str, tuple[float, float, float]]] = [
            ("explicit --cell", explicit_cell)
        ]
        if trajectory_hdf5_cell is not None:
            candidates.append(("trajectory HDF5 metadata", trajectory_hdf5_cell))
        if explicit_input_cell is not None and explicit_input_resolved is not None:
            candidates.append(
                (f"explicit --input ({explicit_input_resolved})", explicit_input_cell)
            )
        _warn_on_mismatched_cells(trajectory, candidates)
        return CellResolution(cell_angstrom=explicit_cell, source="explicit --cell")

    if trajectory_hdf5_cell is not None:
        candidates = [("trajectory HDF5 metadata", trajectory_hdf5_cell)]
        if explicit_input_cell is not None and explicit_input_resolved is not None:
            candidates.append(
                (f"explicit --input ({explicit_input_resolved})", explicit_input_cell)
            )
        _warn_on_mismatched_cells(trajectory, candidates)
        return CellResolution(
            cell_angstrom=trajectory_hdf5_cell,
            source="trajectory HDF5 metadata",
            input_path=trajectory_hdf5_input,
        )

    if explicit_input_cell is not None and explicit_input_resolved is not None:
        return CellResolution(
            cell_angstrom=explicit_input_cell,
            source=f"explicit --input ({explicit_input_resolved})",
            input_path=explicit_input_resolved,
        )

    auto_cell: tuple[float, float, float] | None = None
    auto_input: Path | None = None
    try:
        auto_cell, auto_input = _auto_detect_cell(trajectory)
    except Exception as exc:
        LOGGER.debug("Automatic cell detection failed for '%s': %s", trajectory, exc)
    if auto_cell is not None and auto_input is not None:
        return CellResolution(
            cell_angstrom=auto_cell,
            source=f"auto-detected ({auto_input})",
            input_path=auto_input,
        )

    raise ValueError(
        f"Could not resolve cell dimensions for trajectory '{trajectory}'. "
        f"Checked automatic .inp/.lmp discovery in '{trajectory.parent}'. "
        "Provide --cell A B C or --input /path/to/input.inp (or input.lmp)."
    )


def resolve_analysis_timestep_fs(
    trajectory_path: str | Path,
    *,
    timestep_fs: float | None = None,
    input_path: str | Path | None = None,
    frames: list[Atoms] | None = None,
) -> TimestepResolution:
    """Resolve per-frame timestep from explicit args, metadata, or simulation inputs."""
    trajectory = Path(trajectory_path).expanduser().resolve()

    explicit_timestep = _normalize_timestep_fs(timestep_fs) if timestep_fs is not None else None

    metadata_timestep, metadata_md_timestep, metadata_stride = _extract_metadata_timestep_details(
        frames
    )
    (
        trajectory_hdf5_timestep,
        trajectory_hdf5_md_timestep,
        trajectory_hdf5_stride,
        trajectory_hdf5_input,
    ) = _trajectory_hdf5_timestep_metadata(trajectory)

    explicit_input_timestep: float | None = None
    explicit_input_md_timestep: float | None = None
    explicit_input_stride: int | None = None
    explicit_input_resolved: Path | None = None
    if input_path is not None:
        explicit_input_resolved = Path(input_path).expanduser().resolve()
        (
            explicit_input_timestep,
            explicit_input_md_timestep,
            explicit_input_stride,
        ) = extract_frame_timestep_fs_from_simulation_input(explicit_input_resolved)

    if explicit_timestep is not None:
        candidates: list[tuple[str, float]] = [("explicit --timestep-fs", explicit_timestep)]
        if trajectory_hdf5_timestep is not None:
            candidates.append(("trajectory HDF5 metadata", trajectory_hdf5_timestep))
        if metadata_timestep is not None:
            candidates.append(("trajectory metadata", metadata_timestep))
        if explicit_input_timestep is not None and explicit_input_resolved is not None:
            candidates.append(
                (f"explicit --input ({explicit_input_resolved})", explicit_input_timestep)
            )
        _warn_on_mismatched_timestep_sources(trajectory, candidates)
        return TimestepResolution(
            frame_timestep_fs=explicit_timestep, source="explicit --timestep-fs"
        )

    if trajectory_hdf5_timestep is not None:
        candidates = [("trajectory HDF5 metadata", trajectory_hdf5_timestep)]
        if metadata_timestep is not None:
            candidates.append(("trajectory metadata", metadata_timestep))
        if explicit_input_timestep is not None and explicit_input_resolved is not None:
            candidates.append(
                (f"explicit --input ({explicit_input_resolved})", explicit_input_timestep)
            )
        _warn_on_mismatched_timestep_sources(trajectory, candidates)
        return TimestepResolution(
            frame_timestep_fs=trajectory_hdf5_timestep,
            source="trajectory HDF5 metadata",
            input_path=trajectory_hdf5_input,
            md_timestep_fs=trajectory_hdf5_md_timestep,
            trajectory_stride_md=trajectory_hdf5_stride,
        )

    if metadata_timestep is not None:
        candidates = [("trajectory metadata", metadata_timestep)]
        if explicit_input_timestep is not None and explicit_input_resolved is not None:
            candidates.append(
                (f"explicit --input ({explicit_input_resolved})", explicit_input_timestep)
            )
        _warn_on_mismatched_timestep_sources(trajectory, candidates)
        return TimestepResolution(
            frame_timestep_fs=metadata_timestep,
            source="trajectory metadata",
            md_timestep_fs=metadata_md_timestep,
            trajectory_stride_md=metadata_stride,
        )

    if (
        explicit_input_timestep is not None
        and explicit_input_resolved is not None
        and explicit_input_md_timestep is not None
        and explicit_input_stride is not None
    ):
        return TimestepResolution(
            frame_timestep_fs=explicit_input_timestep,
            source=f"explicit --input ({explicit_input_resolved})",
            input_path=explicit_input_resolved,
            md_timestep_fs=explicit_input_md_timestep,
            trajectory_stride_md=explicit_input_stride,
        )

    auto_timestep: float | None = None
    auto_md_timestep: float | None = None
    auto_stride: int | None = None
    auto_input: Path | None = None
    try:
        auto_timestep, auto_md_timestep, auto_stride, auto_input = _auto_detect_frame_timestep_fs(
            trajectory
        )
    except Exception as exc:
        LOGGER.debug("Automatic timestep detection failed for '%s': %s", trajectory, exc)
    if (
        auto_timestep is not None
        and auto_input is not None
        and auto_md_timestep is not None
        and auto_stride is not None
    ):
        return TimestepResolution(
            frame_timestep_fs=auto_timestep,
            source=f"auto-detected ({auto_input})",
            input_path=auto_input,
            md_timestep_fs=auto_md_timestep,
            trajectory_stride_md=auto_stride,
        )

    raise ValueError(
        f"Could not resolve timestep for trajectory '{trajectory}'. "
        f"Checked trajectory metadata and automatic .inp/.lmp discovery in '{trajectory.parent}'. "
        "Provide --timestep-fs explicitly or --input /path/to/input.inp (or input.lmp)."
    )
