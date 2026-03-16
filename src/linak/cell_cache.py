"""Cell resolution and caching helpers for trajectory analyses."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ase import Atoms
import numpy as np

from .pbc import (
    extract_cell_from_simulation_input,
    extract_frame_timestep_fs_from_simulation_input,
    find_unique_simulation_input,
)
from .utils import ensure_positive

LOGGER = logging.getLogger(__name__)

CACHE_DIRNAME = "linak"
CACHE_FILENAME = "cells.json"
CACHE_VERSION = 1


def _global_cache_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        base = Path(cache_home).expanduser()
    else:
        base = Path.home() / ".cache"
    return base / CACHE_DIRNAME / CACHE_FILENAME


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


def _read_cache_payload(cache_path: Path) -> dict[str, object]:
    if not cache_path.exists():
        return {"version": CACHE_VERSION, "trajectories": {}}

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not parse LiNaK cache '%s': %s", cache_path, exc)
        return {"version": CACHE_VERSION, "trajectories": {}}

    if not isinstance(payload, dict):
        LOGGER.warning(
            "LiNaK cache '%s' has invalid top-level payload; resetting cache.", cache_path
        )
        return {"version": CACHE_VERSION, "trajectories": {}}

    entries = payload.get("trajectories")
    if not isinstance(entries, dict):
        LOGGER.warning(
            "LiNaK cache '%s' has invalid 'trajectories' payload; resetting cache.", cache_path
        )
        return {"version": CACHE_VERSION, "trajectories": {}}

    return payload


def _get_cache_entry(
    payload: dict[str, object],
    trajectory: Path,
) -> dict[str, object]:
    entries = payload.get("trajectories")
    if not isinstance(entries, dict):
        entries = {}
        payload["trajectories"] = entries

    key = str(trajectory)
    entry = entries.get(key)
    if not isinstance(entry, dict):
        entry = {}

    entries[key] = entry
    return entry


def _write_cache_payload(cache_path: Path, payload: dict[str, object]) -> None:
    payload["version"] = CACHE_VERSION
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cached_cell(trajectory_path: str | Path) -> tuple[float, float, float] | None:
    """Load cached cell dimensions for a trajectory from the global cache."""
    trajectory = Path(trajectory_path).expanduser().resolve()
    cache_path = _global_cache_path()
    payload = _read_cache_payload(cache_path)
    entries = payload.get("trajectories")
    if not isinstance(entries, dict):
        return None

    raw_entry = entries.get(str(trajectory))
    if not isinstance(raw_entry, dict):
        return None

    raw_cell = raw_entry.get("cell_angstrom")
    if raw_cell is None:
        return None
    if not isinstance(raw_cell, list) or len(raw_cell) != 3:
        LOGGER.warning(
            "LiNaK cache '%s' has invalid cell payload for trajectory '%s'; ignoring cache entry.",
            cache_path,
            trajectory,
        )
        return None

    try:
        return _normalize_cell((raw_cell[0], raw_cell[1], raw_cell[2]))
    except Exception as exc:
        LOGGER.warning(
            "LiNaK cache '%s' has invalid cell values for trajectory '%s': %s",
            cache_path,
            trajectory,
            exc,
        )
        return None


def store_cached_cell(
    trajectory_path: str | Path,
    cell: tuple[float, float, float],
    *,
    source: str,
    input_path: str | Path | None = None,
) -> Path:
    """Store cell dimensions in the global cache for one trajectory path."""
    trajectory = Path(trajectory_path).expanduser().resolve()
    cache_path = _global_cache_path()
    normalized_cell = _normalize_cell(cell)

    payload = _read_cache_payload(cache_path)
    entry = _get_cache_entry(payload, trajectory)
    entry.update(
        {
            "cell_angstrom": [normalized_cell[0], normalized_cell[1], normalized_cell[2]],
            "source": source,
        }
    )
    if input_path is not None:
        entry["input_path"] = str(Path(input_path).expanduser().resolve())

    _write_cache_payload(cache_path, payload)
    LOGGER.info(
        "Saved LiNaK global cell cache to '%s' for trajectory '%s'.", cache_path, trajectory
    )
    return cache_path


def load_cached_timestep_fs(trajectory_path: str | Path) -> float | None:
    """Load cached per-frame timestep in fs for a trajectory from the global cache."""
    trajectory = Path(trajectory_path).expanduser().resolve()
    cache_path = _global_cache_path()
    payload = _read_cache_payload(cache_path)
    entries = payload.get("trajectories")
    if not isinstance(entries, dict):
        return None

    raw_entry = entries.get(str(trajectory))
    if not isinstance(raw_entry, dict):
        return None

    raw_timestep = raw_entry.get("frame_timestep_fs")
    # Backward compatibility with older cache entries.
    if raw_timestep is None:
        raw_timestep = raw_entry.get("timestep_fs")
    if raw_timestep is None:
        return None

    try:
        return _normalize_timestep_fs(float(raw_timestep))
    except Exception as exc:
        LOGGER.warning(
            "LiNaK cache '%s' has invalid timestep for trajectory '%s': %s",
            cache_path,
            trajectory,
            exc,
        )
        return None


def store_cached_timestep_fs(
    trajectory_path: str | Path,
    timestep_fs: float,
    *,
    source: str,
    input_path: str | Path | None = None,
    md_timestep_fs: float | None = None,
    trajectory_stride_md: int | None = None,
) -> Path:
    """Store per-frame timestep in fs in the global cache for one trajectory path."""
    trajectory = Path(trajectory_path).expanduser().resolve()
    cache_path = _global_cache_path()
    normalized_timestep = _normalize_timestep_fs(timestep_fs)

    payload = _read_cache_payload(cache_path)
    entry = _get_cache_entry(payload, trajectory)

    entry.update(
        {
            "frame_timestep_fs": normalized_timestep,
            # Keep legacy key for compatibility with older code.
            "timestep_fs": normalized_timestep,
            "timestep_source": source,
        }
    )
    if input_path is not None:
        entry["timestep_input_path"] = str(Path(input_path).expanduser().resolve())
    if md_timestep_fs is not None:
        entry["md_timestep_fs"] = _normalize_timestep_fs(md_timestep_fs)
    if trajectory_stride_md is not None:
        stride_md = int(trajectory_stride_md)
        if stride_md <= 0:
            raise ValueError(f"trajectory_stride_md must be > 0, got {trajectory_stride_md}.")
        entry["trajectory_stride_md"] = stride_md

    _write_cache_payload(cache_path, payload)
    LOGGER.info(
        "Saved LiNaK global timestep cache to '%s' for trajectory '%s'.", cache_path, trajectory
    )
    return cache_path


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


def _infer_frame_timestep_fs_from_frames(frames: list[Atoms] | None) -> float | None:
    """Infer per-frame timestep from trajectory metadata when possible."""
    if frames is None or len(frames) < 2:
        return None

    info_first = getattr(frames[0], "info", None)
    if not isinstance(info_first, dict) or not info_first:
        return None

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
        LOGGER.info(
            "Inferred frame timestep from trajectory metadata key '%s': %.6g fs.",
            key,
            inferred,
        )
        return inferred

    for key, factor in absolute_time_key_units.items():
        values: list[float] = []
        display_key: str | None = None
        for frame in frames:
            info = getattr(frame, "info", None)
            if not isinstance(info, dict):
                values = []
                break

            matched_key = None
            for candidate_key, candidate_value in info.items():
                if _normalize_info_key(candidate_key) == key:
                    matched_key = candidate_key
                    parsed = _coerce_float(candidate_value)
                    if parsed is None:
                        values = []
                    else:
                        values.append(parsed)
                    break
            if matched_key is None:
                values = []
                break
            if display_key is None:
                display_key = str(matched_key)
            if not values:
                break

        if len(values) < 2:
            continue

        step = _extract_constant_positive_step(values)
        if step is None:
            continue

        inferred = _normalize_timestep_fs(step * factor)
        LOGGER.info(
            "Inferred frame timestep from trajectory metadata key '%s': %.6g fs.",
            display_key or key,
            inferred,
        )
        return inferred

    return None


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
) -> tuple[float, float, float]:
    """Resolve cell dimensions for analysis with priority:

    1) explicit args (`--cell`, then `--input`), 2) auto `.inp`/`.lmp`, 3) global cache.

    Any available but disagreeing sources are reported as warnings.
    """
    trajectory = Path(trajectory_path).expanduser().resolve()
    cache_path = _global_cache_path()

    explicit_cell = _normalize_cell(cell) if cell is not None else None
    explicit_input_cell = (
        extract_cell_from_simulation_input(input_path) if input_path is not None else None
    )
    explicit_input_resolved = (
        Path(input_path).expanduser().resolve() if input_path is not None else None
    )

    auto_cell: tuple[float, float, float] | None = None
    auto_input: Path | None = None
    try:
        auto_cell, auto_input = _auto_detect_cell(trajectory)
    except Exception as exc:
        LOGGER.debug("Automatic cell detection failed for '%s': %s", trajectory, exc)

    cached_cell = load_cached_cell(trajectory)

    candidates: list[tuple[str, tuple[float, float, float]]] = []
    if explicit_cell is not None:
        candidates.append(("explicit --cell", explicit_cell))
    if explicit_input_cell is not None and explicit_input_resolved is not None:
        candidates.append((f"explicit --input ({explicit_input_resolved})", explicit_input_cell))
    if auto_cell is not None and auto_input is not None:
        candidates.append((f"auto-detected ({auto_input})", auto_cell))
    if cached_cell is not None:
        candidates.append((f"global cache ({cache_path})", cached_cell))
    _warn_on_mismatched_cells(trajectory, candidates)

    if explicit_cell is not None:
        store_cached_cell(trajectory, explicit_cell, source="arg_cell")
        return explicit_cell

    if explicit_input_cell is not None and explicit_input_resolved is not None:
        store_cached_cell(
            trajectory,
            explicit_input_cell,
            source="arg_input",
            input_path=explicit_input_resolved,
        )
        return explicit_input_cell

    if auto_cell is not None and auto_input is not None:
        store_cached_cell(
            trajectory,
            auto_cell,
            source="auto_inp",
            input_path=auto_input,
        )
        return auto_cell

    if cached_cell is not None:
        LOGGER.info(
            "Using cached cell dimensions from global cache '%s' for trajectory '%s'.",
            cache_path,
            trajectory,
        )
        return cached_cell

    raise ValueError(
        f"Could not resolve cell dimensions for trajectory '{trajectory}'. "
        f"Checked automatic .inp/.lmp discovery in '{trajectory.parent}' and global cache '{cache_path}'. "
        "Provide --cell A B C or --input /path/to/input.inp (or input.lmp)."
    )


def resolve_analysis_timestep_fs(
    trajectory_path: str | Path,
    *,
    timestep_fs: float | None = None,
    input_path: str | Path | None = None,
    frames: list[Atoms] | None = None,
) -> float:
    """Resolve per-frame timestep (fs) for analysis.

    Priority:
    1) explicit arg (`--timestep-fs`)
    2) trajectory metadata
    3) explicit simulation `--input` (CP2K or LAMMPS)
    4) auto-detected `.inp`/`.lmp` in trajectory directory
    5) global cache
    """
    trajectory = Path(trajectory_path).expanduser().resolve()
    cache_path = _global_cache_path()

    explicit_timestep = _normalize_timestep_fs(timestep_fs) if timestep_fs is not None else None

    metadata_timestep = _infer_frame_timestep_fs_from_frames(frames)

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

    cached_timestep = load_cached_timestep_fs(trajectory)

    candidates: list[tuple[str, float]] = []
    if explicit_timestep is not None:
        candidates.append(("explicit --timestep-fs", explicit_timestep))
    if metadata_timestep is not None:
        candidates.append(("trajectory metadata", metadata_timestep))
    if explicit_input_timestep is not None and explicit_input_resolved is not None:
        candidates.append(
            (f"explicit --input ({explicit_input_resolved})", explicit_input_timestep)
        )
    if auto_timestep is not None and auto_input is not None:
        candidates.append((f"auto-detected ({auto_input})", auto_timestep))
    if cached_timestep is not None:
        candidates.append((f"global cache ({cache_path})", cached_timestep))
    _warn_on_mismatched_timestep_sources(trajectory, candidates)

    if explicit_timestep is not None:
        store_cached_timestep_fs(trajectory, explicit_timestep, source="arg_timestep")
        return explicit_timestep

    if metadata_timestep is not None:
        store_cached_timestep_fs(trajectory, metadata_timestep, source="trajectory_metadata")
        return metadata_timestep

    if (
        explicit_input_timestep is not None
        and explicit_input_resolved is not None
        and explicit_input_md_timestep is not None
        and explicit_input_stride is not None
    ):
        store_cached_timestep_fs(
            trajectory,
            explicit_input_timestep,
            source="arg_input",
            input_path=explicit_input_resolved,
            md_timestep_fs=explicit_input_md_timestep,
            trajectory_stride_md=explicit_input_stride,
        )
        return explicit_input_timestep

    if (
        auto_timestep is not None
        and auto_input is not None
        and auto_md_timestep is not None
        and auto_stride is not None
    ):
        store_cached_timestep_fs(
            trajectory,
            auto_timestep,
            source="auto_inp",
            input_path=auto_input,
            md_timestep_fs=auto_md_timestep,
            trajectory_stride_md=auto_stride,
        )
        return auto_timestep

    if cached_timestep is not None:
        LOGGER.info(
            "Using cached timestep from global cache '%s' for trajectory '%s': %.6g fs.",
            cache_path,
            trajectory,
            cached_timestep,
        )
        return cached_timestep

    raise ValueError(
        f"Could not resolve timestep for trajectory '{trajectory}'. "
        f"Checked trajectory metadata, automatic .inp/.lmp discovery in '{trajectory.parent}', "
        f"and global cache '{cache_path}'. "
        "Provide --timestep-fs explicitly or --input /path/to/input.inp (or input.lmp)."
    )
