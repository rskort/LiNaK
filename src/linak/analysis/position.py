"""Atom-resolved position analysis routines."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..progress import ProgressBar
from ..storage.hdf5_utils import write_linak_hdf5
from .common import (
    MOLECULE_SPECIES_LABELS,
    available_distinct_raw_species,
    available_element_species,
    frame_has_usable_cell as _common_frame_has_usable_cell,
    is_molecule_species_label,
    molecule_display_label,
    normalize_species_label as _normalize_species,
    normalize_species_query as _normalize_species_query,
    optional_cell_lengths as _optional_cell_lengths,
    optional_finite_float as _optional_finite_float,
    raw_species_labels,
    read_profile_payloads,
    read_profile_payloads_by_index,
    species_selector_raw_label,
    use_multi_series_plot,
    validate_stable_atom_layout as _validate_stable_atom_layout,
    write_profile_collection,
)
from .water import (
    H2O_OH_CUTOFF_A,
    H2O_VALIDATION_STRIDE,
    OHTopologyCache,
    molecule_positions_with_masses as _molecule_positions_with_masses,
)
from .surface import (
    _log_framewise_surface_alignment,
    _select_surface_estimate,
    _surface_estimate_datasets,
    _surface_estimate_from_payload,
    _surface_estimate_supports_distance_mode,
    _surface_metadata_payload,
    _surface_metadata_view,
    SurfaceEstimate,
    SurfaceEstimatorOptions,
)
from .schema import build_profile_metadata, default_plot_labels
from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.position_mapping import resolve_position_plot_mapping
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    _extract_tick_controls,
    _resolve_tick_visibility,
    _render_plot_annotations,
    _sanitize_line_collection_kwargs,
    configure_matplotlib_backend,
    default_series_colors,
    format_axis_label_units,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
    resolve_single_series_options,
)
from ..utils import axis_to_index, ensure_positive

LOGGER = logging.getLogger(__name__)
DEFAULT_MIN_MOLECULE_FRAMES = 5
_POSITION_PROJECTION_COMPONENT = "2d-projection"
_POSITION_PROJECTION_QUANTITIES = (
    "x",
    "y",
    "z",
    "distance",
    "ps",
    "fs",
    "step",
    "frame",
)
_POSITION_PROJECTION_RENDER_MODES = ("color-scale", "line-colors")


@dataclass(frozen=True)
class PositionProfile:
    """Container for atom-resolved positions."""

    species: str
    axis: str
    atom_indices: np.ndarray
    frame_index: np.ndarray
    step: np.ndarray
    time_fs: np.ndarray
    time_ps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    distance_to_surface: np.ndarray
    n_frames: int
    n_atoms: int
    coordinate_mode: str = "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_position_per_frame: np.ndarray | None = None
    surface_estimate: SurfaceEstimate | None = None
    cell_lengths_angstrom: tuple[float, float, float] | None = None
    selection_kind: str = "element"
    entity_kind: str = "atom"
    entity_counts_per_frame: np.ndarray | None = None
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None
    oh_topology_stride: int | None = None


def _frame_has_usable_cell(frame: Atoms) -> bool:
    """Preserve position-profile periodic-cell validation semantics."""
    return _common_frame_has_usable_cell(frame, require_all_pbc=True)


def _resolve_step_values(frames: list[Atoms]) -> np.ndarray:
    values = np.zeros(len(frames), dtype=float)
    all_have_steps = True
    for index, frame in enumerate(frames):
        info = getattr(frame, "info", None)
        if not isinstance(info, dict) or "timestep" not in info:
            all_have_steps = False
            break
        raw = info.get("timestep")
        parsed: float | None = None
        if isinstance(raw, (int, float, np.integer, np.floating)):
            parsed = float(raw)
        elif isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                try:
                    parsed = float(stripped)
                except ValueError:
                    parsed = None
        if parsed is None or not np.isfinite(parsed):
            all_have_steps = False
            break
        values[index] = parsed
    if all_have_steps:
        return values
    return np.arange(len(frames), dtype=float)


def _resolve_cell_lengths_from_frames(
    frames: list[Atoms],
) -> tuple[float, float, float] | None:
    if not frames:
        return None
    if not all(_frame_has_usable_cell(frame) for frame in frames):
        return None
    lengths = np.asarray(frames[0].cell.lengths(), dtype=float)
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]))


def _resolve_surface_distance_values(
    *,
    frames: list[Atoms],
    axis: str,
    axis_values_all: np.ndarray,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
    precomputed_surface_estimate: SurfaceEstimate | None,
) -> tuple[np.ndarray, str, float | None, float | None, np.ndarray | None, SurfaceEstimate | None]:
    if str(surface_mode).strip().lower() == "none":
        return np.array(axis_values_all, copy=True), "axis", None, None, None, None

    surface_estimate: SurfaceEstimate | None
    if precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (axis_values_all.shape[0],):
            raise ValueError(
                "precomputed_surface_estimate frame count does not match the trajectory."
            )
        surface_estimate = precomputed_surface_estimate
    else:
        surface_estimate, _surface_method = _select_surface_estimate(
            frames,
            axis,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
    if surface_estimate is None:
        LOGGER.warning(
            "Could not estimate a surface position along %s; storing raw %s coordinates "
            "for distance-to-surface values.",
            axis.lower(),
            axis.lower(),
        )
        return np.array(axis_values_all, copy=True), "axis", None, None, None, None

    surface_per_frame = np.asarray(surface_estimate.per_frame, dtype=float)
    if _surface_estimate_supports_distance_mode(
        surface_estimate, frame_count=axis_values_all.shape[0]
    ):
        assert surface_estimate.position is not None
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_estimate.position,
            surface_position_std=surface_estimate.std,
        )
        return (
            axis_values_all - surface_per_frame[:, np.newaxis],
            "distance",
            float(surface_estimate.position),
            None if surface_estimate.std is None else float(surface_estimate.std),
            surface_per_frame,
            surface_estimate,
        )

    LOGGER.warning(
        "Surface position was estimated for %s, but frame-wise alignment was unavailable; "
        "storing raw %s coordinates for distance-to-surface values.",
        axis.lower(),
        axis.lower(),
    )
    return np.array(axis_values_all, copy=True), "axis", None, None, None, surface_estimate


def _position_species_display_label(species_label: str) -> str:
    if is_molecule_species_label(species_label):
        return molecule_display_label(species_label)
    if str(species_label).lower().startswith("species:"):
        return f"{species_selector_raw_label(species_label)} (species)"
    return str(species_label)


def _position_selection_kind_for_label(species_label: str) -> str:
    if is_molecule_species_label(species_label):
        return "molecule"
    if str(species_label).lower().startswith("species:"):
        return "species"
    return "element"


def _position_selector_mask(
    frame: Atoms,
    species_label: str,
    *,
    symbols: np.ndarray | None = None,
) -> np.ndarray:
    if species_label == "ALL":
        return np.ones(len(frame), dtype=bool)
    selection_mode, selection_label = _normalize_species_query(species_label)
    if selection_mode == "species":
        labels = raw_species_labels(frame)
        return labels == species_selector_raw_label(selection_label)
    frame_symbols = (
        np.asarray(frame.get_chemical_symbols(), dtype=object)
        if symbols is None
        else np.asarray(symbols, dtype=object)
    )
    if selection_mode == "element":
        return frame_symbols == selection_label
    return frame_symbols == species_label


def _position_selection_summary(
    *,
    element_labels: list[str],
    raw_species_labels: list[str] | None = None,
    molecule_labels: list[str],
) -> str:
    parts: list[str] = []
    if element_labels:
        parts.append("elements=" + ",".join(element_labels))
    if raw_species_labels:
        parts.append(
            "species="
            + ",".join(_position_species_display_label(label) for label in raw_species_labels)
        )
    if molecule_labels:
        parts.append(
            "molecules="
            + ",".join(_position_species_display_label(label) for label in molecule_labels)
        )
    return "; ".join(parts) if parts else "none"


def _molecule_position_label_is_active(
    *,
    species_label: str,
    active_frame_count: int,
    consecutive_frame_count: int | None = None,
    apply_min_molecule_frames: bool,
    min_molecule_frames: int,
) -> bool:
    if active_frame_count <= 0:
        return False
    if not apply_min_molecule_frames:
        return True
    if species_label == "mol:H2O":
        return True
    threshold_count = active_frame_count if consecutive_frame_count is None else consecutive_frame_count
    return int(threshold_count) >= int(min_molecule_frames)


def _max_consecutive_truthy(values: Any) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _position_surface_context(
    *,
    frames: list[Atoms],
    axis: str,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
    precomputed_surface_estimate: SurfaceEstimate | None,
) -> tuple[str, float | None, float | None, np.ndarray | None, SurfaceEstimate | None]:
    if str(surface_mode).strip().lower() == "none":
        return "axis", None, None, None, None

    if precomputed_surface_estimate is not None:
        surface_estimate = precomputed_surface_estimate
    else:
        surface_estimate, _surface_method = _select_surface_estimate(
            frames,
            axis,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
    if surface_estimate is None:
        LOGGER.warning(
            "Could not estimate a surface position along %s; storing raw %s coordinates "
            "for distance-to-surface values.",
            axis.lower(),
            axis.lower(),
        )
        return "axis", None, None, None, None

    if _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames)):
        assert surface_estimate.position is not None
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_estimate.position,
            surface_position_std=surface_estimate.std,
        )
        return (
            "distance",
            float(surface_estimate.position),
            None if surface_estimate.std is None else float(surface_estimate.std),
            np.asarray(surface_estimate.per_frame, dtype=float),
            surface_estimate,
        )

    LOGGER.warning(
        "Surface position was estimated for %s, but frame-wise alignment was unavailable; "
        "storing raw %s coordinates for distance-to-surface values.",
        axis.lower(),
        axis.lower(),
    )
    return "axis", None, None, None, surface_estimate


def _distance_values_for_axis_matrix(
    axis_values: np.ndarray,
    *,
    surface_position_per_frame: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(axis_values, dtype=float)
    if surface_position_per_frame is None:
        return np.array(values, copy=True)
    return values - np.asarray(surface_position_per_frame, dtype=float)[:, np.newaxis]


def _position_time_arrays(
    *,
    frames: list[Atoms],
    timestep_fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_index = np.arange(len(frames), dtype=int)
    step = _resolve_step_values(frames)
    time_fs = frame_index.astype(float) * float(timestep_fs)
    time_ps = time_fs / 1000.0
    return frame_index, step, time_fs, time_ps


def _compute_position_profiles_for_labels(
    *,
    frames: list[Atoms],
    species_labels: list[str],
    axis: str,
    timestep_fs: float,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
    precomputed_surface_estimate: SurfaceEstimate | None,
) -> list[PositionProfile]:
    ensure_positive("timestep_fs", timestep_fs)
    if not frames:
        raise ValueError("At least one trajectory frame is required.")
    stable_layout = True
    try:
        symbols = _validate_stable_atom_layout(
            frames,
            description="Atom-resolved position tracking",
        )
    except ValueError as exc:
        stable_layout = False
        LOGGER.warning(
            "%s Falling back to frame-wise position selection with NaN padding.",
            exc,
        )
        symbols = np.asarray(frames[0].get_chemical_symbols(), dtype=object)
    if stable_layout and any(
        str(label).lower().startswith("species:") for label in species_labels
    ):
        reference_raw_labels = raw_species_labels(frames[0])
        for frame_index, frame in enumerate(frames[1:], start=1):
            current_raw_labels = raw_species_labels(frame)
            if current_raw_labels.shape != reference_raw_labels.shape or not np.array_equal(
                current_raw_labels,
                reference_raw_labels,
            ):
                stable_layout = False
                LOGGER.warning(
                    "Raw atom species/order changed at frame %d. Falling back to frame-wise "
                    "position selection with NaN padding.",
                    frame_index,
                )
                break

    cell_lengths_angstrom = _resolve_cell_lengths_from_frames(frames)
    axis_index = axis_to_index(axis)
    (
        coordinate_mode,
        surface_position,
        surface_position_std,
        surface_position_per_frame,
        surface_estimate,
    ) = _position_surface_context(
        frames=frames,
        axis=axis,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
    )
    frame_index, step, time_fs, time_ps = _position_time_arrays(
        frames=frames,
        timestep_fs=timestep_fs,
    )

    profiles: list[PositionProfile] = []
    if stable_layout:
        selected_indices_by_label: dict[str, np.ndarray] = {}
        for species_label in species_labels:
            if species_label == "ALL":
                atom_indices = np.arange(symbols.size, dtype=int)
            else:
                atom_indices = np.where(
                    _position_selector_mask(frames[0], species_label, symbols=symbols)
                )[0].astype(int, copy=False)
            if atom_indices.size == 0:
                raise ValueError(f"No atoms found for species '{species_label}' in frame 0.")
            selected_indices_by_label[species_label] = np.asarray(atom_indices, dtype=int)

        matrices_by_label: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for species_label, atom_indices in selected_indices_by_label.items():
            shape = (len(frames), int(atom_indices.size))
            matrices_by_label[species_label] = (
                np.empty(shape, dtype=float),
                np.empty(shape, dtype=float),
                np.empty(shape, dtype=float),
            )

        with ProgressBar(desc="Selecting atom positions", total=len(frames), unit="frame") as progress:
            for frame_i, frame in enumerate(frames):
                positions = np.asarray(frame.positions, dtype=float)
                for species_label, atom_indices in selected_indices_by_label.items():
                    x_values, y_values, z_values = matrices_by_label[species_label]
                    selected = positions[atom_indices]
                    x_values[frame_i, :] = selected[:, 0]
                    y_values[frame_i, :] = selected[:, 1]
                    z_values[frame_i, :] = selected[:, 2]
                progress.update()

        for species_label, atom_indices in selected_indices_by_label.items():
            x_values, y_values, z_values = matrices_by_label[species_label]
            axis_values_all = (x_values, y_values, z_values)[axis_index]
            distance_to_surface_all = _distance_values_for_axis_matrix(
                axis_values_all,
                surface_position_per_frame=surface_position_per_frame,
            )
            profiles.append(
                PositionProfile(
                    species=species_label,
                    axis=axis.lower(),
                    atom_indices=np.asarray(atom_indices, dtype=int),
                    frame_index=np.asarray(frame_index, dtype=int),
                    step=np.asarray(step, dtype=float),
                    time_fs=np.asarray(time_fs, dtype=float),
                    time_ps=np.asarray(time_ps, dtype=float),
                    x=np.asarray(x_values, dtype=float),
                    y=np.asarray(y_values, dtype=float),
                    z=np.asarray(z_values, dtype=float),
                    distance_to_surface=np.asarray(distance_to_surface_all, dtype=float),
                    n_frames=len(frames),
                    n_atoms=int(atom_indices.size),
                    coordinate_mode=coordinate_mode,
                    selection_kind=_position_selection_kind_for_label(species_label),
                    surface_position=surface_position,
                    surface_position_std=surface_position_std,
                    surface_position_per_frame=(
                        None
                        if surface_position_per_frame is None
                        else np.asarray(surface_position_per_frame, dtype=float)
                    ),
                    surface_estimate=surface_estimate,
                    cell_lengths_angstrom=cell_lengths_angstrom,
                )
            )
        return profiles

    counts_by_label: dict[str, np.ndarray] = {
        label: np.zeros(len(frames), dtype=int) for label in species_labels
    }
    max_counts_by_label: dict[str, int] = {label: 0 for label in species_labels}
    for frame_i, frame in enumerate(frames):
        frame_symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
        for species_label in species_labels:
            if species_label == "ALL":
                count = int(frame_symbols.size)
            else:
                count = int(
                    np.count_nonzero(
                        _position_selector_mask(frame, species_label, symbols=frame_symbols)
                    )
                )
            counts_by_label[species_label][frame_i] = count
            max_counts_by_label[species_label] = max(max_counts_by_label[species_label], count)

    matrices_by_label = {}
    for species_label in species_labels:
        max_count = max_counts_by_label[species_label]
        if max_count <= 0:
            raise ValueError(f"No atoms found for species '{species_label}' in trajectory.")
        shape = (len(frames), max_count)
        matrices_by_label[species_label] = (
            np.full(shape, np.nan, dtype=float),
            np.full(shape, np.nan, dtype=float),
            np.full(shape, np.nan, dtype=float),
        )

    with ProgressBar(desc="Selecting atom positions", total=len(frames), unit="frame") as progress:
        for frame_i, frame in enumerate(frames):
            frame_symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
            positions = np.asarray(frame.positions, dtype=float)
            for species_label in species_labels:
                if species_label == "ALL":
                    atom_indices = np.arange(frame_symbols.size, dtype=int)
                else:
                    atom_indices = np.where(
                        _position_selector_mask(frame, species_label, symbols=frame_symbols)
                    )[0].astype(int, copy=False)
                if atom_indices.size == 0:
                    continue
                x_values, y_values, z_values = matrices_by_label[species_label]
                selected = positions[atom_indices]
                count = int(atom_indices.size)
                x_values[frame_i, :count] = selected[:, 0]
                y_values[frame_i, :count] = selected[:, 1]
                z_values[frame_i, :count] = selected[:, 2]
            progress.update()

    for species_label in species_labels:
        x_values, y_values, z_values = matrices_by_label[species_label]
        max_count = int(max_counts_by_label[species_label])
        axis_values_all = (x_values, y_values, z_values)[axis_index]
        distance_to_surface_all = _distance_values_for_axis_matrix(
            axis_values_all,
            surface_position_per_frame=surface_position_per_frame,
        )
        profiles.append(
            PositionProfile(
                species=species_label,
                axis=axis.lower(),
                atom_indices=np.arange(max_count, dtype=int),
                frame_index=np.asarray(frame_index, dtype=int),
                step=np.asarray(step, dtype=float),
                time_fs=np.asarray(time_fs, dtype=float),
                time_ps=np.asarray(time_ps, dtype=float),
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(y_values, dtype=float),
                z=np.asarray(z_values, dtype=float),
                distance_to_surface=np.asarray(distance_to_surface_all, dtype=float),
                n_frames=len(frames),
                n_atoms=max_count,
                coordinate_mode=coordinate_mode,
                selection_kind=_position_selection_kind_for_label(species_label),
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_position_per_frame=(
                    None
                    if surface_position_per_frame is None
                    else np.asarray(surface_position_per_frame, dtype=float)
                ),
                surface_estimate=surface_estimate,
                cell_lengths_angstrom=cell_lengths_angstrom,
                entity_counts_per_frame=np.asarray(counts_by_label[species_label], dtype=int),
            )
        )
    return profiles


def _pad_molecule_position_frames(
    positions_per_frame: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray([np.asarray(values).shape[0] for values in positions_per_frame], dtype=int)
    max_count = int(np.max(counts)) if counts.size else 0
    shape = (len(positions_per_frame), max_count)
    x = np.full(shape, np.nan, dtype=float)
    y = np.full(shape, np.nan, dtype=float)
    z = np.full(shape, np.nan, dtype=float)
    for frame_index, positions in enumerate(positions_per_frame):
        values = np.asarray(positions, dtype=float)
        if values.size == 0:
            continue
        count = values.shape[0]
        x[frame_index, :count] = values[:, 0]
        y[frame_index, :count] = values[:, 1]
        z[frame_index, :count] = values[:, 2]
    return x, y, z, counts


def _compute_molecule_position_profiles_for_labels(
    *,
    frames: list[Atoms],
    molecule_labels: list[str],
    axis: str,
    timestep_fs: float,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
    precomputed_surface_estimate: SurfaceEstimate | None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
    oh_topology_stride: int = H2O_VALIDATION_STRIDE,
    apply_min_molecule_frames: bool = False,
) -> list[PositionProfile]:
    ensure_positive("timestep_fs", timestep_fs)
    ensure_positive("oh_cutoff", oh_cutoff)
    if int(min_molecule_frames) < 1:
        raise ValueError("min_molecule_frames must be >= 1.")
    if int(oh_topology_stride) < 1:
        raise ValueError("oh_topology_stride must be >= 1.")
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    axis_index = axis_to_index(axis)
    cell_lengths_angstrom = _resolve_cell_lengths_from_frames(frames)
    frame_index, step, time_fs, time_ps = _position_time_arrays(
        frames=frames,
        timestep_fs=timestep_fs,
    )
    (
        coordinate_mode,
        surface_position,
        surface_position_std,
        surface_position_per_frame,
        surface_estimate,
    ) = _position_surface_context(
        frames=frames,
        axis=axis,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
    )

    requested_labels = list(dict.fromkeys(molecule_labels))
    topology_indices_by_frame: list[dict[str, np.ndarray]] = []
    counts_by_label: dict[str, np.ndarray] = {
        label: np.zeros(len(frames), dtype=int) for label in requested_labels
    }
    max_counts_by_label: dict[str, int] = {label: 0 for label in requested_labels}
    topology_cache = OHTopologyCache(
        oh_cutoff=oh_cutoff,
        validation_stride=oh_topology_stride,
        logger=LOGGER,
        context="position O/H molecule topology",
    )
    LOGGER.info("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
    with ProgressBar(desc="Selecting O/H molecules for position", total=len(frames), unit="frame") as progress:
        for frame_i, frame in enumerate(frames):
            topology = topology_cache.select(frame, frame_index=frame_i)
            frame_indices: dict[str, np.ndarray] = {}
            for label in requested_labels:
                indices = np.asarray(topology.indices_for(label), dtype=int)
                frame_indices[label] = indices
                count = int(indices.shape[0])
                counts_by_label[label][frame_i] = count
                max_counts_by_label[label] = max(max_counts_by_label[label], count)
            topology_indices_by_frame.append(frame_indices)
            progress.update()

    active_labels = [
        label
        for label in requested_labels
        if _molecule_position_label_is_active(
            species_label=label,
            active_frame_count=int(np.count_nonzero(counts_by_label[label] > 0)),
            consecutive_frame_count=_max_consecutive_truthy(counts_by_label[label] > 0),
            apply_min_molecule_frames=apply_min_molecule_frames,
            min_molecule_frames=int(min_molecule_frames),
        )
    ]
    detected_non_water = [
        _position_species_display_label(label)
        for label in active_labels
        if label != "mol:H2O" and int(np.count_nonzero(counts_by_label[label] > 0)) > 0
    ]
    if detected_non_water:
        LOGGER.info(
            "Detected O/H molecule types: %s.",
            ",".join(detected_non_water),
        )

    profiles: list[PositionProfile] = []
    for label in active_labels:
        max_entities = int(max_counts_by_label[label])
        if max_entities == 0:
            continue
        shape = (len(frames), max_entities)
        x_values = np.full(shape, np.nan, dtype=float)
        y_values = np.full(shape, np.nan, dtype=float)
        z_values = np.full(shape, np.nan, dtype=float)
        for frame_i, frame in enumerate(frames):
            molecule_indices = topology_indices_by_frame[frame_i][label]
            if molecule_indices.size == 0:
                continue
            positions, _masses = _molecule_positions_with_masses(
                frame,
                molecule_indices,
            )
            positions = np.asarray(positions, dtype=float)
            count = int(positions.shape[0])
            x_values[frame_i, :count] = positions[:, 0]
            y_values[frame_i, :count] = positions[:, 1]
            z_values[frame_i, :count] = positions[:, 2]

        axis_values_all = (x_values, y_values, z_values)[axis_index]
        distance_to_surface_all = _distance_values_for_axis_matrix(
            axis_values_all,
            surface_position_per_frame=surface_position_per_frame,
        )
        profiles.append(
            PositionProfile(
                species=label,
                axis=axis.lower(),
                atom_indices=np.arange(max_entities, dtype=int),
                frame_index=np.asarray(frame_index, dtype=int),
                step=np.asarray(step, dtype=float),
                time_fs=np.asarray(time_fs, dtype=float),
                time_ps=np.asarray(time_ps, dtype=float),
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(y_values, dtype=float),
                z=np.asarray(z_values, dtype=float),
                distance_to_surface=np.asarray(distance_to_surface_all, dtype=float),
                n_frames=len(frames),
                n_atoms=max_entities,
                coordinate_mode=coordinate_mode,
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_position_per_frame=(
                    None
                    if surface_position_per_frame is None
                    else np.asarray(surface_position_per_frame, dtype=float)
                ),
                surface_estimate=surface_estimate,
                cell_lengths_angstrom=cell_lengths_angstrom,
                selection_kind="molecule",
                entity_kind="molecule",
                entity_counts_per_frame=np.asarray(counts_by_label[label], dtype=int),
                oh_cutoff_A=float(oh_cutoff),
                min_molecule_frames=(
                    int(min_molecule_frames) if apply_min_molecule_frames else None
                ),
                oh_topology_stride=int(oh_topology_stride),
            )
        )
    return profiles


def compute_position_profile(
    frames: list[Atoms],
    species: str | None = "all",
    *,
    axis: str = "z",
    timestep_fs: float = 1.0,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
    oh_topology_stride: int = H2O_VALIDATION_STRIDE,
) -> PositionProfile:
    """Compute one atom-resolved position profile."""
    selection_mode, species_label = _normalize_species_query(
        species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if selection_mode == "molecule":
        profiles = _compute_molecule_position_profiles_for_labels(
            frames=frames,
            molecule_labels=[species_label],
            axis=axis,
            timestep_fs=timestep_fs,
            surface_mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            precomputed_surface_estimate=precomputed_surface_estimate,
            oh_cutoff=oh_cutoff,
            min_molecule_frames=min_molecule_frames,
            oh_topology_stride=oh_topology_stride,
            apply_min_molecule_frames=False,
        )
        if not profiles:
            raise ValueError(f"No molecules found for species '{species_label}' in trajectory.")
        return profiles[0]
    if selection_mode in {"elements", "molecules"}:
        raise ValueError(
            "compute_position_profile accepts one element or molecule selector; "
            "use compute_position_profiles for group selectors."
        )
    profiles = _compute_position_profiles_for_labels(
        frames=frames,
        species_labels=[species_label],
        axis=axis,
        timestep_fs=timestep_fs,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
    )
    return profiles[0]


def compute_position_profiles(
    frames: list[Atoms],
    species: str | None = "all",
    *,
    axis: str = "z",
    timestep_fs: float = 1.0,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
    oh_topology_stride: int = H2O_VALIDATION_STRIDE,
) -> list[PositionProfile]:
    """Compute one or more atom-resolved position profiles."""
    ensure_positive("oh_cutoff", oh_cutoff)
    if int(min_molecule_frames) < 1:
        raise ValueError("min_molecule_frames must be >= 1.")
    if int(oh_topology_stride) < 1:
        raise ValueError("oh_topology_stride must be >= 1.")
    selection_mode, species_label = _normalize_species_query(
        species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if selection_mode == "molecule":
        profiles = _compute_molecule_position_profiles_for_labels(
            frames=frames,
            molecule_labels=[species_label],
            axis=axis,
            timestep_fs=timestep_fs,
            surface_mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            precomputed_surface_estimate=precomputed_surface_estimate,
            oh_cutoff=oh_cutoff,
            min_molecule_frames=min_molecule_frames,
            oh_topology_stride=oh_topology_stride,
            apply_min_molecule_frames=False,
        )
        if not profiles:
            raise ValueError(f"No molecules found for species '{species_label}' in trajectory.")
        return profiles
    if selection_mode in {"element", "species"}:
        return [
            compute_position_profile(
                frames=frames,
                species=species_label,
                axis=axis,
                timestep_fs=timestep_fs,
                surface_mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                surface_options=surface_options,
                precomputed_surface_estimate=precomputed_surface_estimate,
                oh_cutoff=oh_cutoff,
                min_molecule_frames=min_molecule_frames,
                oh_topology_stride=oh_topology_stride,
            )
        ]

    element_species = (
        available_element_species(frames)
        if selection_mode in {"all", "elements"}
        else []
    )
    if selection_mode in {"all", "elements"} and not element_species:
        raise ValueError("No elements found in trajectory.")
    raw_species_labels_for_output = (
        [f"species:{label}" for label in available_distinct_raw_species(frames)]
        if selection_mode == "all"
        else []
    )
    molecule_labels = list(MOLECULE_SPECIES_LABELS) if selection_mode in {"all", "molecules"} else []
    profiles: list[PositionProfile] = []
    if element_species:
        profiles.extend(
            _compute_position_profiles_for_labels(
                frames=frames,
                species_labels=element_species,
                axis=axis,
                timestep_fs=timestep_fs,
                surface_mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                surface_options=surface_options,
                precomputed_surface_estimate=precomputed_surface_estimate,
            )
        )
    if raw_species_labels_for_output:
        profiles.extend(
            _compute_position_profiles_for_labels(
                frames=frames,
                species_labels=raw_species_labels_for_output,
                axis=axis,
                timestep_fs=timestep_fs,
                surface_mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                surface_options=surface_options,
                precomputed_surface_estimate=precomputed_surface_estimate,
            )
        )
    if molecule_labels:
        molecule_profiles = _compute_molecule_position_profiles_for_labels(
            frames=frames,
            molecule_labels=molecule_labels,
            axis=axis,
            timestep_fs=timestep_fs,
            surface_mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            precomputed_surface_estimate=precomputed_surface_estimate,
            oh_cutoff=oh_cutoff,
            min_molecule_frames=min_molecule_frames,
            oh_topology_stride=oh_topology_stride,
            apply_min_molecule_frames=True,
        )
        profiles.extend(molecule_profiles)
    LOGGER.info(
        "Position selections: %s.",
        _position_selection_summary(
            element_labels=element_species,
            raw_species_labels=[
                profile.species for profile in profiles if profile.selection_kind == "species"
            ],
            molecule_labels=[
                profile.species for profile in profiles if profile.selection_kind == "molecule"
            ],
        ),
    )
    return profiles


def _position_profile_hdf5_payload(profile: PositionProfile) -> dict[str, Any]:
    metadata = build_profile_metadata(
        analysis="position",
        metadata={
            "species": profile.species,
            "axis": profile.axis,
            "n_frames": int(profile.n_frames),
            "n_atoms": int(profile.n_atoms),
            "n_entities": int(profile.n_atoms),
            "coordinate_mode": profile.coordinate_mode,
            "selection_kind": profile.selection_kind,
            "entity_kind": profile.entity_kind,
            "entity_counts_available": profile.entity_counts_per_frame is not None,
            "oh_cutoff_A": profile.oh_cutoff_A,
            "min_molecule_frames": profile.min_molecule_frames,
            "oh_topology_stride": profile.oh_topology_stride,
            "cell_lengths_angstrom": (
                None
                if profile.cell_lengths_angstrom is None
                else [float(value) for value in profile.cell_lengths_angstrom]
            ),
            **_surface_metadata_payload(
                surface_position=profile.surface_position,
                surface_position_std=profile.surface_position_std,
                estimate=profile.surface_estimate,
            ),
        },
    )
    return {
        "datasets": {
            "frame_index": profile.frame_index,
            "step": profile.step,
            "time_fs": profile.time_fs,
            "time_ps": profile.time_ps,
            "atom_indices": profile.atom_indices,
            "x_A": profile.x,
            "y_A": profile.y,
            "z_A": profile.z,
            "distance_to_surface_A": profile.distance_to_surface,
            "entity_counts": profile.entity_counts_per_frame,
            "surface_position_per_frame_A": profile.surface_position_per_frame,
            **_surface_estimate_datasets(profile.surface_estimate),
        },
        "metadata": metadata,
    }


def save_position_profile(
    profile: PositionProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one position profile to LiNaK HDF5 and return the written path."""
    payload = _position_profile_hdf5_payload(profile)
    metadata = dict(payload["metadata"])
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="position",
        datasets=payload["datasets"],
        metadata=metadata,
    )
    LOGGER.info("Saved position data to '%s'.", output_path)
    return output_path


def save_position_profiles(
    profiles: list[PositionProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one or more position profiles to LiNaK HDF5 and return the written path."""

    if not profiles:
        raise ValueError("At least one position profile is required.")
    if len(profiles) == 1:
        return save_position_profile(
            profiles[0],
            output,
            additional_metadata=additional_metadata,
        )
    output_path = write_profile_collection(
        output,
        analysis="position",
        profiles=[_position_profile_hdf5_payload(profile) for profile in profiles],
        metadata=dict(additional_metadata or {}),
    )
    LOGGER.info("Saved %d position profiles to '%s'.", len(profiles), output_path)
    return output_path


def load_position_profile(
    path: str | Path,
    *,
    species: str | None = None,
    axis: str | None = None,
) -> PositionProfile:
    """Load one position profile from LiNaK HDF5."""
    profiles = load_position_profiles(path, species=species, axis=axis)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(
            f"Position HDF5 '{source_path}' does not contain matching position profiles."
        )
    return profiles[0]


def load_position_profiles(
    path: str | Path,
    *,
    species: str | None = None,
    axis: str | None = None,
) -> list[PositionProfile]:
    """Load one or more position profiles from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads(
        path,
        analysis="position",
        label="Position",
    )
    return _load_position_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
        axis=axis,
    )


def _optional_int_metadata(metadata: Mapping[str, Any], name: str) -> int | None:
    raw_value = metadata.get(name)
    if raw_value is None:
        return None
    return int(raw_value)


def _position_payload_matches_selection(
    metadata: Mapping[str, Any],
    *,
    species: str | None = None,
    axis: str | None = None,
) -> bool:
    wanted_axis = None if axis is None or not axis.strip() else axis.strip().lower()
    if wanted_axis is not None:
        metadata_axis = str(metadata.get("axis", "z")).strip().lower() or "z"
        if metadata_axis != wanted_axis:
            return False

    requested_species = None if species is None or not str(species).strip() else str(species)
    if requested_species is None:
        return True

    selection_mode, requested_label = _normalize_species_query(
        requested_species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if selection_mode == "all":
        return True

    metadata_species = str(metadata.get("species", "")).strip()
    if not metadata_species:
        return False
    metadata_mode, metadata_label = _normalize_species_query(
        metadata_species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if selection_mode == "elements":
        return metadata_mode == "element"
    if selection_mode == "molecules":
        return metadata_mode == "molecule"
    return metadata_mode == selection_mode and metadata_label == requested_label


def _load_position_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species: str | None = None,
    axis: str | None = None,
) -> list[PositionProfile]:
    profiles: list[PositionProfile] = []
    for datasets, metadata in payloads:
        required = (
            "frame_index",
            "step",
            "time_fs",
            "time_ps",
            "atom_indices",
            "x_A",
            "y_A",
            "z_A",
            "distance_to_surface_A",
        )
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Position HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )

        resolved_species = str(metadata.get("species", "")).strip() or "UNKNOWN"
        resolved_axis = str(metadata.get("axis", "z")).strip().lower()
        if resolved_axis not in {"x", "y", "z"}:
            resolved_axis = "z"
        if not _position_payload_matches_selection(metadata, species=species, axis=axis):
            continue
        resolved_mode, resolved_label = _normalize_species_query(
            resolved_species,
            allow_h2o=True,
            allow_molecules=True,
        )
        if resolved_mode == "molecule":
            resolved_species = resolved_label

        frame_index = np.asarray(datasets["frame_index"], dtype=int)
        step = np.asarray(datasets["step"], dtype=float)
        time_fs = np.asarray(datasets["time_fs"], dtype=float)
        time_ps = np.asarray(datasets["time_ps"], dtype=float)
        atom_indices = np.asarray(datasets["atom_indices"], dtype=int)
        x_values = np.asarray(datasets["x_A"], dtype=float)
        y_values = np.asarray(datasets["y_A"], dtype=float)
        z_values = np.asarray(datasets["z_A"], dtype=float)
        distance_values = np.asarray(datasets["distance_to_surface_A"], dtype=float)

        if x_values.ndim != 2:
            raise ValueError(f"Position HDF5 '{source_path}' dataset 'x_A' must be 2D.")
        expected_shape = x_values.shape
        for dataset_name, values in (
            ("y_A", y_values),
            ("z_A", z_values),
            ("distance_to_surface_A", distance_values),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"Position HDF5 '{source_path}' dataset '{dataset_name}' shape mismatch: "
                    f"expected {expected_shape}, got {values.shape}."
                )
        if atom_indices.size != expected_shape[1]:
            raise ValueError(
                f"Position HDF5 '{source_path}' has inconsistent atom index count "
                f"({atom_indices.size}) for matrix width {expected_shape[1]}."
            )
        if frame_index.size != expected_shape[0]:
            raise ValueError(
                f"Position HDF5 '{source_path}' has inconsistent frame index count "
                f"({frame_index.size}) for matrix height {expected_shape[0]}."
            )

        coordinate_mode = str(metadata.get("coordinate_mode", "axis")).strip().lower()
        if coordinate_mode not in {"axis", "distance"}:
            coordinate_mode = "axis"

        surface_per_frame = None
        if "surface_position_per_frame_A" in datasets:
            candidate = np.asarray(datasets["surface_position_per_frame_A"], dtype=float)
            if candidate.shape == (expected_shape[0],):
                surface_per_frame = candidate
        entity_counts_per_frame = None
        if "entity_counts" in datasets:
            candidate_counts = np.asarray(datasets["entity_counts"], dtype=int)
            if candidate_counts.shape == (expected_shape[0],):
                entity_counts_per_frame = candidate_counts
        surface_estimate = _surface_estimate_from_payload(
            datasets=datasets,
            metadata=metadata,
        )

        n_frames = int(metadata.get("n_frames", expected_shape[0]))
        n_atoms = int(metadata.get("n_atoms", expected_shape[1]))
        selection_kind = str(metadata.get("selection_kind", "element")).strip().lower() or "element"
        entity_kind = str(metadata.get("entity_kind", "atom")).strip().lower() or "atom"
        cell_lengths_angstrom = (
            _optional_cell_lengths(metadata.get("cell_lengths_angstrom"))
            or _optional_cell_lengths(metadata.get("pbc_cell_angstrom"))
            or _optional_cell_lengths(metadata.get("resolved_cell_angstrom"))
        )
        surface_metadata = _surface_metadata_view(metadata)
        profiles.append(
            PositionProfile(
                species=resolved_species,
                axis=resolved_axis,
                atom_indices=atom_indices,
                frame_index=frame_index,
                step=step,
                time_fs=time_fs,
                time_ps=time_ps,
                x=x_values,
                y=y_values,
                z=z_values,
                distance_to_surface=distance_values,
                n_frames=n_frames,
                n_atoms=n_atoms,
                coordinate_mode=coordinate_mode,
                surface_position=_optional_finite_float(
                    surface_metadata.get("position", metadata.get("surface_position"))
                ),
                surface_position_std=_optional_finite_float(
                    surface_metadata.get("position_std", metadata.get("surface_position_std"))
                ),
                surface_position_per_frame=surface_per_frame,
                surface_estimate=surface_estimate,
                cell_lengths_angstrom=cell_lengths_angstrom,
                selection_kind=selection_kind,
                entity_kind=entity_kind,
                entity_counts_per_frame=entity_counts_per_frame,
                oh_cutoff_A=_optional_finite_float(metadata.get("oh_cutoff_A")),
                min_molecule_frames=_optional_int_metadata(metadata, "min_molecule_frames"),
                oh_topology_stride=_optional_int_metadata(metadata, "oh_topology_stride"),
            )
        )
    return profiles


def load_position_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species: str | None = None,
    axis: str | None = None,
) -> list[PositionProfile]:
    """Load selected position profiles by profile index from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="position",
        label="Position",
    )
    return _load_position_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
        axis=axis,
    )


def _position_time_data(
    profile: PositionProfile,
    *,
    time_axis: str,
) -> tuple[np.ndarray, str]:
    normalized = time_axis.strip().lower()
    if normalized == "ps":
        return profile.time_ps, "Time (ps)"
    if normalized == "fs":
        return profile.time_fs, "Time (fs)"
    if normalized == "step":
        return profile.step, "Timestep"
    if normalized == "frame":
        return profile.frame_index.astype(float), "Frame index"
    raise ValueError(
        f"Unsupported position time_axis '{time_axis}'. Choose 'ps', 'fs', 'step', or 'frame'."
    )


def _normalize_component_token(component: str) -> str:
    token = component.strip().lower().replace("_", "-").replace(" ", "-")
    if token in {"distance", "x", "y", "z"}:
        return token
    if token in {
        "xy-z",
        "xy-z-color",
        "xy-z-colormap",
        "trajectory",
        "xyz",
        "2d-projection",
        "2dprojection",
        "projection-2d",
        "projection2d",
        "projection",
    }:
        return _POSITION_PROJECTION_COMPONENT
    raise ValueError(
        f"Unsupported position component '{component}'. "
        "Choose 'distance', 'x', 'y', 'z', or '2d-projection' (alias 'xy-z')."
    )


def _normalize_map_color_token(map_color: str) -> str:
    token = map_color.strip().lower().replace("_", "-")
    if token in {"distance", "z"}:
        return token
    if token in {"surface-distance", "dist"}:
        return "distance"
    raise ValueError(f"Unsupported position map_color '{map_color}'. Choose 'distance' or 'z'.")


def _normalize_projection_quantity_token(quantity: str) -> str:
    token = quantity.strip().lower().replace("_", "-").replace(" ", "-")
    if token in _POSITION_PROJECTION_QUANTITIES:
        return token
    if token in {"surface-distance", "dist"}:
        return "distance"
    raise ValueError(
        f"Unsupported position projection quantity '{quantity}'. Choose one of "
        + ", ".join(f"'{value}'" for value in _POSITION_PROJECTION_QUANTITIES)
        + "."
    )


def _normalize_projection_render_mode_token(render_mode: str) -> str:
    token = render_mode.strip().lower().replace("_", "-").replace(" ", "-")
    if token in _POSITION_PROJECTION_RENDER_MODES:
        return token
    if token in {"color", "colormap", "colorscale"}:
        return "color-scale"
    if token in {"lines", "line-colour", "line-colours", "line-colors"}:
        return "line-colors"
    raise ValueError(
        f"Unsupported position projection render mode '{render_mode}'. "
        "Choose 'color-scale' or 'line-colors'."
    )


def _position_component_data(
    profile: PositionProfile,
    *,
    component: str,
) -> tuple[np.ndarray, str]:
    normalized = _normalize_component_token(component)
    if normalized == "distance":
        if profile.coordinate_mode != "distance":
            LOGGER.warning(
                "Position profile '%s' has no valid surface-distance reference; using %s-axis values.",
                profile.species,
                profile.axis.upper(),
            )
            return profile.distance_to_surface, f"{profile.axis.upper()} (A)"
        return profile.distance_to_surface, "Distance to the surface ($\\mathrm{\\AA}$)"
    if normalized == "x":
        return profile.x, "X (A)"
    if normalized == "y":
        return profile.y, "Y (A)"
    if normalized == "z":
        return profile.z, "Z (A)"
    raise ValueError(
        "2-D projection components must be rendered via trajectory projection plotting."
    )


def _position_map_color_data(
    profile: PositionProfile,
    *,
    map_color: str,
) -> tuple[np.ndarray, str]:
    normalized = _normalize_map_color_token(map_color)
    if normalized == "distance":
        return _position_component_data(profile, component="distance")
    return np.asarray(profile.z, dtype=float), "Z (A)"


def _position_projection_quantity_data(
    profile: PositionProfile,
    *,
    quantity: str,
) -> tuple[np.ndarray, str]:
    normalized = _normalize_projection_quantity_token(quantity)
    if normalized == "distance":
        return (
            np.asarray(profile.distance_to_surface, dtype=float),
            "Distance to the surface ($\\mathrm{\\AA}$)",
        )
    if normalized in {"x", "y", "z"}:
        return _position_component_data(profile, component=normalized)
    time_values, label = _position_time_data(profile, time_axis=normalized)
    matrix = np.repeat(np.asarray(time_values, dtype=float)[:, np.newaxis], profile.n_atoms, axis=1)
    return matrix, label


def _projection_default_title(
    *,
    x_quantity: str,
    y_quantity: str,
    value_quantity: str,
    render_mode: str,
) -> str:
    x_token = _normalize_projection_quantity_token(x_quantity)
    y_token = _normalize_projection_quantity_token(y_quantity)
    value_token = _normalize_projection_quantity_token(value_quantity)
    if render_mode == "color-scale":
        return f"{x_token.upper()}-{y_token.upper()} trajectories colored by {value_token}"
    return f"{x_token.upper()}-{y_token.upper()} trajectories"


def _build_xy_segments(
    x_values: np.ndarray,
    y_values: np.ndarray,
    color_values: np.ndarray,
    *,
    cell_lengths_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size != y_values.size or x_values.size != color_values.size:
        raise ValueError(
            "x, y, and color arrays must have matching length for XY segment building."
        )
    if x_values.size < 2:
        return (
            np.empty((0, 2, 2), dtype=float),
            np.empty((0,), dtype=float),
        )

    points = np.column_stack((x_values, y_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_colors = 0.5 * (color_values[:-1] + color_values[1:])

    if cell_lengths_xy is None:
        return np.asarray(segments, dtype=float), np.asarray(segment_colors, dtype=float)

    x_length, y_length = cell_lengths_xy
    if x_length <= 0.0 or y_length <= 0.0:
        return np.asarray(segments, dtype=float), np.asarray(segment_colors, dtype=float)

    dx = np.abs(np.diff(x_values))
    dy = np.abs(np.diff(y_values))
    # Break PBC-jump connectors so trajectories do not draw artificial lines across the box.
    keep = (dx <= (0.5 * x_length + 1e-12)) & (dy <= (0.5 * y_length + 1e-12))
    return np.asarray(segments[keep], dtype=float), np.asarray(segment_colors[keep], dtype=float)


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(np.asarray(mask, dtype=bool)):
        if flag:
            if start is None:
                start = index
            continue
        if start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _build_projection_paths(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    cell_lengths_xy: tuple[float, float] | None,
) -> list[np.ndarray]:
    if x_values.size != y_values.size:
        raise ValueError("x and y arrays must have matching length for projection path building.")
    if x_values.size == 0:
        return []

    points = np.column_stack((x_values, y_values))
    if x_values.size == 1 or cell_lengths_xy is None:
        return [np.asarray(points, dtype=float)]

    x_length, y_length = cell_lengths_xy
    if x_length <= 0.0 or y_length <= 0.0:
        return [np.asarray(points, dtype=float)]

    dx = np.abs(np.diff(x_values))
    dy = np.abs(np.diff(y_values))
    keep = (dx <= (0.5 * x_length + 1e-12)) & (dy <= (0.5 * y_length + 1e-12))
    paths: list[np.ndarray] = []
    start = 0
    for index, segment_kept in enumerate(keep):
        if segment_kept:
            continue
        if index + 1 > start:
            paths.append(np.asarray(points[start : index + 1], dtype=float))
        start = index + 1
    if points.shape[0] > start:
        paths.append(np.asarray(points[start:], dtype=float))
    return [path for path in paths if path.size > 0]


def _coerce_projection_filter_bound(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _resolve_projection_settings(
    *,
    component: str,
    map_color: str,
    projection_x: str | None,
    projection_y: str | None,
    projection_value: str | None,
    projection_render_mode: str | None,
    projection_filter_min: float | None,
    projection_filter_max: float | None,
    xy_z_distance_max: float | None,
) -> tuple[str, str, str, str, float | None, float | None]:
    normalized_component = _normalize_component_token(component)
    if normalized_component != _POSITION_PROJECTION_COMPONENT:
        raise ValueError("Projection settings can only be resolved for 2-D position projections.")

    resolved_x = _normalize_projection_quantity_token(projection_x or "x")
    resolved_y = _normalize_projection_quantity_token(projection_y or "y")
    resolved_value = _normalize_projection_quantity_token(
        projection_value if projection_value is not None else _normalize_map_color_token(map_color)
    )
    resolved_render_mode = _normalize_projection_render_mode_token(
        projection_render_mode or "color-scale"
    )
    resolved_filter_min = _coerce_projection_filter_bound(
        projection_filter_min,
        field_name="projection filter minimum",
    )
    resolved_filter_max = _coerce_projection_filter_bound(
        projection_filter_max,
        field_name="projection filter maximum",
    )
    if (
        resolved_filter_max is None
        and xy_z_distance_max is not None
        and resolved_value == "distance"
    ):
        resolved_filter_max = _coerce_projection_filter_bound(
            xy_z_distance_max,
            field_name="xy-z distance max",
        )
    if (
        resolved_filter_min is not None
        and resolved_filter_max is not None
        and resolved_filter_min > resolved_filter_max
    ):
        raise ValueError("Projection filter minimum must not exceed the projection filter maximum.")
    return (
        resolved_x,
        resolved_y,
        resolved_value,
        resolved_render_mode,
        resolved_filter_min,
        resolved_filter_max,
    )


def _default_position_series_labels(profile: PositionProfile) -> list[str]:
    species_label = _position_species_display_label(profile.species)
    return [f"{species_label}[{int(atom_index)}]" for atom_index in profile.atom_indices.tolist()]


def _first_non_none(values: list[float | None] | None) -> float | None:
    if not values:
        return None
    for value in values:
        if value is not None:
            return float(value)
    return None


def _plot_position_xy_z_projection(
    profiles: list[PositionProfile],
    *,
    map_color: str,
    output: str | Path | None,
    show: bool,
    show_blocking: bool,
    preferred_backend: str | None,
    style: PlotStyle,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    x_scale: str,
    y_scale: str,
    x_lim: tuple[float | None, float | None] | list[float | None] | None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None,
    x_ticks: list[float] | tuple[float, ...] | None,
    y_ticks: list[float] | tuple[float, ...] | None,
    x_tick_rotation: float | None,
    y_tick_rotation: float | None,
    x_label_font_size: int | None,
    y_label_font_size: int | None,
    x_label_pad: float | None,
    y_label_pad: float | None,
    title_pad: float | None,
    title_visible: bool | None,
    ticks_visible: bool | None,
    line_colors: list[str] | None,
    series_enabled: list[bool] | None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None,
    series_markers: list[str | None] | None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    series_normalization_modes: list[str | None] | None,
    series_normalization_values: list[float | None] | None,
    series_normalization_x_refs: list[float | None] | None,
    x_bin_width: float | None,
    x_bin_reducer: str | None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    legend_kwargs: dict[str, Any] | None = None,
    annotations: list[dict[str, Any]] | None,
    capture_state: dict[str, Any] | None,
    suppress_output_log: bool,
    matplotlib_rc: dict[str, Any] | None,
    figure_kwargs: dict[str, Any] | None,
    axes_kwargs: dict[str, Any] | None,
    line_kwargs: dict[str, Any] | None,
    grid_kwargs: dict[str, Any] | None,
    tick_params_kwargs: dict[str, Any] | None,
    tight_layout_kwargs: dict[str, Any] | None,
    savefig_kwargs: dict[str, Any] | None,
    component: str = "xy-z",
    projection_x: str | None = None,
    projection_y: str | None = None,
    projection_value: str | None = None,
    projection_render_mode: str | None = None,
    projection_filter_min: float | None = None,
    projection_filter_max: float | None = None,
    xy_z_distance_max: float | None = None,
) -> Path | None:
    if not profiles:
        raise ValueError("At least one position profile is required.")
    return _plot_position_projection(
        profiles,
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        title=title,
        x_label=x_label,
        y_label=y_label,
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
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        series_line_kwargs=series_line_kwargs,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        legend_kwargs=legend_kwargs,
        annotations=annotations,
        capture_state=capture_state,
        suppress_output_log=suppress_output_log,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        grid_kwargs=grid_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        component=component,
        map_color=map_color,
        projection_x=projection_x,
        projection_y=projection_y,
        projection_value=projection_value,
        projection_render_mode=projection_render_mode,
        projection_filter_min=projection_filter_min,
        projection_filter_max=projection_filter_max,
        xy_z_distance_max=xy_z_distance_max,
    )

    series_total = sum(max(0, int(profile.n_atoms)) for profile in profiles)
    if series_enabled is not None and len(series_enabled) != series_total:
        raise ValueError(
            "series_enabled count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if line_colors is not None and len(line_colors) != series_total:
        raise ValueError(
            "line_colors count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_line_widths is not None and len(series_line_widths) != series_total:
        raise ValueError(
            "series_line_widths count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_markers is not None and len(series_markers) != series_total:
        raise ValueError(
            "series_markers count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_normalization_modes is not None and len(series_normalization_modes) != series_total:
        raise ValueError(
            "series_normalization_modes count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_normalization_values is not None and len(series_normalization_values) != series_total:
        raise ValueError(
            "series_normalization_values count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_normalization_x_refs is not None and len(series_normalization_x_refs) != series_total:
        raise ValueError(
            "series_normalization_x_refs count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if xy_z_distance_max is not None and float(xy_z_distance_max) <= 0.0:
        raise ValueError("xy-z distance max must be positive.")

    if x_bin_width is not None:
        LOGGER.warning(
            "Position component 'xy-z' ignores time-section/x-bin settings (received %.6g; reducer=%s).",
            x_bin_width,
            x_bin_reducer or "mean",
        )
    if series_normalization_modes is not None:
        LOGGER.warning("Position component 'xy-z' ignores per-series y-normalization settings.")
    if line_colors is not None and any(str(color).strip() for color in line_colors):
        LOGGER.debug(
            "Position component 'xy-z' ignores per-series fixed line colors and uses %s colormap values.",
            _normalize_map_color_token(map_color),
        )

    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    segment_blocks: list[np.ndarray] = []
    segment_color_blocks: list[np.ndarray] = []
    point_x_values: list[float] = []
    point_y_values: list[float] = []
    point_color_values: list[float] = []
    color_label_reference: str | None = None

    series_index = 0
    for profile in profiles:
        x_matrix = np.asarray(profile.x, dtype=float)
        y_matrix = np.asarray(profile.y, dtype=float)
        color_matrix, color_label = _position_map_color_data(profile, map_color=map_color)
        cell_lengths_xy = None
        if profile.cell_lengths_angstrom is not None:
            cell_lengths_xy = (
                float(profile.cell_lengths_angstrom[0]),
                float(profile.cell_lengths_angstrom[1]),
            )
        if color_label_reference is None:
            color_label_reference = color_label
        elif color_label != color_label_reference:
            color_label_reference = "Color value (A)"
        if not (x_matrix.shape == y_matrix.shape == color_matrix.shape):
            raise ValueError(
                f"Position profile '{profile.species}' has inconsistent x/y/color matrix shapes."
            )
        for atom_column in range(x_matrix.shape[1]):
            is_enabled = True if series_enabled is None else bool(series_enabled[series_index])
            series_index += 1
            if not is_enabled:
                continue

            x_values = x_matrix[:, atom_column]
            y_values = y_matrix[:, atom_column]
            color_values = color_matrix[:, atom_column]
            distance_values = np.asarray(profile.distance_to_surface[:, atom_column], dtype=float)
            visible_mask = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & np.isfinite(color_values)
                & np.isfinite(distance_values)
            )
            if xy_z_distance_max is not None:
                visible_mask &= distance_values <= float(xy_z_distance_max)
            if not np.any(visible_mask):
                continue
            visible_x = np.asarray(x_values[visible_mask], dtype=float)
            visible_y = np.asarray(y_values[visible_mask], dtype=float)
            visible_colors = np.asarray(color_values[visible_mask], dtype=float)
            point_x_values.extend(float(value) for value in visible_x)
            point_y_values.extend(float(value) for value in visible_y)
            point_color_values.extend(float(value) for value in visible_colors)

            for start, stop in _contiguous_true_runs(visible_mask):
                run_x = np.asarray(x_values[start:stop], dtype=float)
                run_y = np.asarray(y_values[start:stop], dtype=float)
                run_colors = np.asarray(color_values[start:stop], dtype=float)
                segments, segment_colors = _build_xy_segments(
                    run_x,
                    run_y,
                    run_colors,
                    cell_lengths_xy=cell_lengths_xy,
                )
                if segments.size == 0:
                    continue
                segment_blocks.append(segments)
                segment_color_blocks.append(segment_colors)

    if not segment_blocks and not point_x_values:
        if xy_z_distance_max is not None:
            raise ValueError("No atom trajectories remain after applying the xy-z distance cutoff.")
        raise ValueError("No enabled atom trajectories available for 'xy-z' position plotting.")

    color_samples: list[np.ndarray] = []
    if segment_color_blocks:
        color_samples.extend(segment_color_blocks)
    if point_color_values:
        color_samples.append(np.asarray(point_color_values, dtype=float))
    color_all = np.concatenate(color_samples)
    color_min = float(np.nanmin(color_all))
    color_max = float(np.nanmax(color_all))
    if not np.isfinite(color_min) or not np.isfinite(color_max):
        raise ValueError("Cannot render 'xy-z' projection because color values are non-finite.")
    if color_min == color_max:
        color_min -= 0.5
        color_max += 0.5
    norm = mcolors.Normalize(vmin=color_min, vmax=color_max)

    line_collection_kwargs = _sanitize_line_collection_kwargs(line_kwargs)
    explicit_line_width = _first_non_none(series_line_widths)
    line_collection_kwargs.setdefault(
        "linewidths",
        style.line_width if explicit_line_width is None else explicit_line_width,
    )

    marker_size = max(9.0, (style.line_width * 7.0) ** 2)
    active_backend = configure_matplotlib_backend(
        interactive=show,
        preferred_backend=preferred_backend,
    )
    rc_context_args: dict[str, Any] = {"font.family": style.font_family, "text.parse_math": True}
    if matplotlib_rc is not None:
        rc_context_args.update(dict(matplotlib_rc))

    with plt.rc_context(rc_context_args):
        fig, ax = plt.subplots(figsize=style.figure_size)
        if figure_kwargs is not None:
            fig.set(**dict(figure_kwargs))

        mappable = None
        if segment_blocks:
            segments_all = np.concatenate(segment_blocks, axis=0)
            segment_color_all = np.concatenate(segment_color_blocks, axis=0)
            collection = LineCollection(
                segments_all,
                cmap="turbo",
                norm=norm,
                **line_collection_kwargs,
            )
            collection.set_array(segment_color_all)
            ax.add_collection(collection)
            mappable = collection
        if point_x_values:
            scatter = ax.scatter(
                np.asarray(point_x_values, dtype=float),
                np.asarray(point_y_values, dtype=float),
                c=np.asarray(point_color_values, dtype=float),
                cmap="turbo",
                norm=norm,
                s=marker_size,
                edgecolors="none",
            )
            if mappable is None:
                mappable = scatter

        ax.autoscale()
        colorbar = fig.colorbar(mappable, ax=ax) if mappable is not None else None
        if colorbar is not None:
            colorbar.set_label(
                color_label_reference or "Color value (A)",
                fontsize=style.label_font_size,
            )
            colorbar.ax.tick_params(labelsize=style.tick_font_size)

        xlabel_kwargs: dict[str, Any] = {"fontsize": x_label_font_size or style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": y_label_font_size or style.label_font_size}
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(
            format_axis_label_units(resolve_explicit_plot_text(x_label, "X (A)")),
            **xlabel_kwargs,
        )
        ax.set_ylabel(
            format_axis_label_units(resolve_explicit_plot_text(y_label, "Y (A)")),
            **ylabel_kwargs,
        )
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size, pad=style.title_pad)
        else:
            ax.set_title(
                title
                or (
                    "XY trajectories colored by distance to surface"
                    if _normalize_map_color_token(map_color) == "distance"
                    else "XY trajectories colored by Z"
                ),
                fontsize=style.title_font_size,
                pad=style.title_pad,
            )

        ax.tick_params(axis="both", labelsize=style.tick_font_size)
        resolved_tick_params, tick_axis_hint, minor_ticks_mode = _extract_tick_controls(
            tick_params_kwargs
        )
        if minor_ticks_mode == "on":
            ax.minorticks_on()
        elif minor_ticks_mode == "off":
            ax.minorticks_off()
        if resolved_tick_params:
            ax.tick_params(**resolved_tick_params)
        x_ticks_visible, y_ticks_visible = _resolve_tick_visibility(
            tick_params_kwargs,
            ticks_visible,
            tick_axis_hint,
        )
        if not x_ticks_visible:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
        if not y_ticks_visible:
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
            )
        if x_tick_rotation is not None:
            ax.tick_params(axis="x", rotation=float(x_tick_rotation))
        if y_tick_rotation is not None:
            ax.tick_params(axis="y", rotation=float(y_tick_rotation))

        if style.grid:
            resolved_grid_kwargs: dict[str, Any] = {
                "linestyle": style.grid_linestyle,
                "linewidth": style.grid_linewidth,
                "alpha": style.grid_alpha,
            }
            if grid_kwargs is not None:
                resolved_grid_kwargs.update(dict(grid_kwargs))
            ax.grid(True, **resolved_grid_kwargs)
        elif grid_kwargs is not None:
            ax.grid(**dict(grid_kwargs))

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        if x_ticks is not None:
            ax.set_xticks([float(value) for value in x_ticks])
        if y_ticks is not None:
            ax.set_yticks([float(value) for value in y_ticks])
        default_cell_x: float | None = None
        default_cell_y: float | None = None
        profile_lengths = [
            profile.cell_lengths_angstrom
            for profile in profiles
            if profile.cell_lengths_angstrom is not None
        ]
        if profile_lengths:
            default_cell_x = max(float(lengths[0]) for lengths in profile_lengths)
            default_cell_y = max(float(lengths[1]) for lengths in profile_lengths)
        effective_x_lim = x_lim
        effective_y_lim = y_lim
        if effective_x_lim is None and default_cell_x is not None:
            effective_x_lim = (0.0, default_cell_x)
        if effective_y_lim is None and default_cell_y is not None:
            effective_y_lim = (0.0, default_cell_y)

        if effective_x_lim is not None:
            left = None if effective_x_lim[0] is None else float(effective_x_lim[0])
            right = None if effective_x_lim[1] is None else float(effective_x_lim[1])
            ax.set_xlim(left=left, right=right)
        if effective_y_lim is not None:
            bottom = None if effective_y_lim[0] is None else float(effective_y_lim[0])
            top = None if effective_y_lim[1] is None else float(effective_y_lim[1])
            ax.set_ylim(bottom=bottom, top=top)
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))
        annotation_summaries = _render_plot_annotations(ax, annotations)

        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()

        if capture_state is not None:
            capture_state.clear()
            capture_state.update(
                {
                    "figure": fig,
                    "axes": ax,
                    "title": str(ax.get_title()),
                    "title_visible": bool(
                        ax.title.get_visible() and bool(str(ax.get_title()).strip())
                    ),
                    "x_label": str(ax.get_xlabel()),
                    "y_label": str(ax.get_ylabel()),
                    "x_scale": str(ax.get_xscale()),
                    "y_scale": str(ax.get_yscale()),
                    "x_lim": [float(value) for value in ax.get_xlim()],
                    "y_lim": [float(value) for value in ax.get_ylim()],
                    "x_ticks": [float(value) for value in ax.get_xticks()],
                    "y_ticks": [float(value) for value in ax.get_yticks()],
                    "legend": False,
                    "legend_title": None,
                    "legend_loc": "best",
                    "series_labels": None,
                    "line_colors": None,
                    "markers": bool(point_x_values),
                    "annotations_summary": annotation_summaries,
                }
            )

        output_path = None
        if output is not None:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs: dict[str, Any] = {}
            if savefig_kwargs is not None:
                save_kwargs.update(dict(savefig_kwargs))
            save_kwargs.setdefault("dpi", style.dpi)
            fig.savefig(output_path, **save_kwargs)
            if not suppress_output_log:
                LOGGER.info("Saved plot to '%s'.", output_path)

        if show:
            if show_blocking:
                LOGGER.info(
                    "Showing interactive plot window using backend '%s'. Close the window to continue.",
                    active_backend,
                )
            else:
                LOGGER.info(
                    "Showing interactive plot window using backend '%s'.",
                    active_backend,
                )
            plt.show(block=show_blocking)
            if not show_blocking:
                plt.pause(0.001)

        if not (show and not show_blocking):
            plt.close(fig)
        return output_path


def _plot_position_projection(
    profiles: list[PositionProfile],
    *,
    output: str | Path | None,
    show: bool,
    show_blocking: bool,
    preferred_backend: str | None,
    style: PlotStyle,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    x_scale: str,
    y_scale: str,
    x_lim: tuple[float | None, float | None] | list[float | None] | None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None,
    x_ticks: list[float] | tuple[float, ...] | None,
    y_ticks: list[float] | tuple[float, ...] | None,
    x_tick_rotation: float | None,
    y_tick_rotation: float | None,
    x_label_font_size: int | None,
    y_label_font_size: int | None,
    x_label_pad: float | None,
    y_label_pad: float | None,
    title_pad: float | None,
    title_visible: bool | None,
    ticks_visible: bool | None,
    line_colors: list[str] | None,
    series_enabled: list[bool] | None,
    series_show_in_legend: list[bool] | None,
    series_line_widths: list[float | None] | None,
    series_markers: list[str | None] | None,
    render_series_descriptors: list[dict[str, Any]] | None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None,
    series_line_kwargs: list[dict[str, Any] | None] | None,
    series_normalization_modes: list[str | None] | None,
    series_normalization_values: list[float | None] | None,
    series_normalization_x_refs: list[float | None] | None,
    x_bin_width: float | None,
    x_bin_reducer: str | None,
    legend: bool | None,
    legend_title: str | None,
    legend_loc: str,
    legend_kwargs: dict[str, Any] | None,
    annotations: list[dict[str, Any]] | None,
    capture_state: dict[str, Any] | None,
    suppress_output_log: bool,
    matplotlib_rc: dict[str, Any] | None,
    figure_kwargs: dict[str, Any] | None,
    axes_kwargs: dict[str, Any] | None,
    line_kwargs: dict[str, Any] | None,
    grid_kwargs: dict[str, Any] | None,
    tick_params_kwargs: dict[str, Any] | None,
    tight_layout_kwargs: dict[str, Any] | None,
    savefig_kwargs: dict[str, Any] | None,
    component: str,
    map_color: str,
    projection_x: str | None,
    projection_y: str | None,
    projection_value: str | None,
    projection_render_mode: str | None,
    projection_filter_min: float | None,
    projection_filter_max: float | None,
    xy_z_distance_max: float | None,
) -> Path | None:
    if not profiles:
        raise ValueError("At least one position profile is required.")
    component_token = str(component).strip().lower().replace("_", "-").replace(" ", "-")
    legacy_xy_z_alias = component_token in {
        "xy-z",
        "xy-z-color",
        "xy-z-colormap",
        "trajectory",
        "xyz",
    }

    (
        resolved_projection_x,
        resolved_projection_y,
        resolved_projection_value,
        resolved_render_mode,
        resolved_filter_min,
        resolved_filter_max,
    ) = _resolve_projection_settings(
        component=component,
        map_color=map_color,
        projection_x=projection_x,
        projection_y=projection_y,
        projection_value=projection_value,
        projection_render_mode=projection_render_mode,
        projection_filter_min=projection_filter_min,
        projection_filter_max=projection_filter_max,
        xy_z_distance_max=xy_z_distance_max,
    )
    raw_line_colors = line_colors
    if resolved_render_mode == "color-scale":
        line_colors = None
    series_total = sum(max(0, int(profile.n_atoms)) for profile in profiles)
    for values, field_name in (
        (series_enabled, "series_enabled"),
        (line_colors, "line_colors"),
        (series_show_in_legend, "series_show_in_legend"),
        (series_line_widths, "series_line_widths"),
        (series_markers, "series_markers"),
        (series_line_kwargs, "series_line_kwargs"),
        (series_normalization_modes, "series_normalization_modes"),
        (series_normalization_values, "series_normalization_values"),
        (series_normalization_x_refs, "series_normalization_x_refs"),
    ):
        if values is not None and len(values) != series_total:
            raise ValueError(
                f"{field_name} count must match the number of plotted position atom series "
                f"({series_total})."
            )

    if x_bin_width is not None:
        LOGGER.warning(
            "Position 2-D projection ignores time-section/x-bin settings (received %.6g; reducer=%s).",
            x_bin_width,
            x_bin_reducer or "mean",
        )
    if series_normalization_modes is not None:
        LOGGER.warning("Position 2-D projection ignores per-series y-normalization settings.")
    if (
        resolved_render_mode == "color-scale"
        and raw_line_colors is not None
        and any(str(color).strip() for color in raw_line_colors)
    ):
        LOGGER.debug(
            "Position 2-D projection ignores per-series fixed line colors in color-scale mode and uses %s values.",
            resolved_projection_value,
        )

    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    def _build_profile_line_shades(base_color: str, count: int) -> list[str]:
        if count <= 0:
            return []
        if count == 1:
            return [str(base_color)]
        base_rgb = mcolors.to_rgb(base_color)
        hue, lightness, saturation = colorsys.rgb_to_hls(*base_rgb)
        dark_lightness = max(0.18, min(0.42, lightness * 0.72))
        light_lightness = min(0.82, max(0.58, lightness + (1.0 - lightness) * 0.28))
        if dark_lightness >= light_lightness:
            dark_lightness = max(0.12, light_lightness - 0.2)
        shades: list[str] = []
        for fraction in np.linspace(0.0, 1.0, int(count), dtype=float):
            shades.append(
                mcolors.to_hex(
                    colorsys.hls_to_rgb(
                        hue,
                        dark_lightness + (light_lightness - dark_lightness) * float(fraction),
                        saturation,
                    )
                )
            )
        return shades

    color_scale_segment_blocks: list[np.ndarray] = []
    color_scale_segment_color_blocks: list[np.ndarray] = []
    color_scale_point_x: list[float] = []
    color_scale_point_y: list[float] = []
    color_scale_point_values: list[float] = []
    visible_x_values: list[float] = []
    visible_y_values: list[float] = []
    line_series_payloads: list[dict[str, Any]] = []
    line_series_labels: list[str] = []
    value_label_reference: str | None = None
    default_x_label = "X (A)"
    default_y_label = "Y (A)"
    default_line_colors = default_series_colors(series_total)
    default_projection_x_lim: tuple[float | None, float | None] | None = None
    default_projection_y_lim: tuple[float | None, float | None] | None = None
    ordered_descriptors = list(render_series_descriptors or [])
    overrides = dict(series_overrides_by_id) if isinstance(series_overrides_by_id, Mapping) else {}
    profile_descriptor_mode = resolved_render_mode == "color-scale" and len(
        ordered_descriptors
    ) == len(profiles)
    atom_descriptor_mode = (
        resolved_render_mode == "line-colors" and len(ordered_descriptors) == series_total
    )

    def _override_entry(series_id: str | None) -> dict[str, Any]:
        token = str(series_id or "").strip()
        value = overrides.get(token)
        return dict(value) if isinstance(value, Mapping) else {}

    if legacy_xy_z_alias:
        cell_x_lengths = [
            float(profile.cell_lengths_angstrom[0])
            for profile in profiles
            if profile.cell_lengths_angstrom is not None
            and np.isfinite(float(profile.cell_lengths_angstrom[0]))
        ]
        cell_y_lengths = [
            float(profile.cell_lengths_angstrom[1])
            for profile in profiles
            if profile.cell_lengths_angstrom is not None
            and np.isfinite(float(profile.cell_lengths_angstrom[1]))
        ]
        if resolved_projection_x == "x" and cell_x_lengths:
            default_projection_x_lim = (0.0, max(cell_x_lengths))
        if resolved_projection_y == "y" and cell_y_lengths:
            default_projection_y_lim = (0.0, max(cell_y_lengths))

    series_index = 0
    for profile_index, profile in enumerate(profiles):
        profile_override: dict[str, Any] = {}
        profile_enabled = True
        if profile_descriptor_mode:
            profile_descriptor = ordered_descriptors[profile_index]
            profile_override = _override_entry(profile_descriptor.get("series_id"))
            profile_enabled = bool(profile_override.get("enabled", True)) and bool(
                profile_override.get("show_raw_line", True)
            )
        x_matrix, default_x_label = _position_projection_quantity_data(
            profile, quantity=resolved_projection_x
        )
        y_matrix, default_y_label = _position_projection_quantity_data(
            profile, quantity=resolved_projection_y
        )
        value_matrix, value_label = _position_projection_quantity_data(
            profile, quantity=resolved_projection_value
        )
        cell_lengths_xy = None
        if profile.cell_lengths_angstrom is not None:
            cell_lengths_xy = (
                float(profile.cell_lengths_angstrom[0]),
                float(profile.cell_lengths_angstrom[1]),
            )
        value_label_reference = (
            value_label if value_label_reference is None else value_label_reference
        )
        if not (x_matrix.shape == y_matrix.shape == value_matrix.shape):
            raise ValueError(
                f"Position profile '{profile.species}' has inconsistent projection matrix shapes."
            )
        profile_base_color = default_series_colors(len(profiles))[profile_index]
        if line_colors is not None:
            profile_series_end = series_index + int(profile.n_atoms)
            for raw_color in line_colors[series_index:profile_series_end]:
                if str(raw_color or "").strip():
                    profile_base_color = str(raw_color).strip()
                    break
        if atom_descriptor_mode:
            profile_series_end = series_index + int(profile.n_atoms)
            for descriptor_index in range(series_index, profile_series_end):
                descriptor = ordered_descriptors[descriptor_index]
                descriptor_override = _override_entry(descriptor.get("series_id"))
                if descriptor_override.get("color") not in {None, ""}:
                    profile_base_color = str(descriptor_override.get("color"))
                    break
        profile_line_payloads: list[dict[str, Any]] = []
        for atom_column, atom_default_label in enumerate(_default_position_series_labels(profile)):
            descriptor_label = atom_default_label
            current_override: dict[str, Any] = {}
            if atom_descriptor_mode:
                descriptor = ordered_descriptors[series_index]
                current_override = _override_entry(descriptor.get("series_id"))
                descriptor_label = str(current_override.get("label_override") or "").strip() or str(
                    descriptor.get("default_label") or atom_default_label
                )
            is_enabled = True if series_enabled is None else bool(series_enabled[series_index])
            if atom_descriptor_mode:
                is_enabled = bool(current_override.get("enabled", True)) and bool(
                    current_override.get("show_raw_line", True)
                )
            elif profile_descriptor_mode:
                is_enabled = profile_enabled
            show_in_legend = (
                True if series_show_in_legend is None else bool(series_show_in_legend[series_index])
            )
            if atom_descriptor_mode:
                show_in_legend = bool(current_override.get("show_in_legend", show_in_legend))
            line_width_value = (
                None if series_line_widths is None else series_line_widths[series_index]
            )
            line_width = style.line_width if line_width_value is None else float(line_width_value)
            if atom_descriptor_mode and current_override.get("line_width") not in {None, ""}:
                line_width = float(current_override["line_width"])
            line_color = (
                default_line_colors[series_index]
                if line_colors is None or not str(line_colors[series_index]).strip()
                else str(line_colors[series_index]).strip()
            )
            if atom_descriptor_mode and current_override.get("color") not in {None, ""}:
                line_color = str(current_override.get("color"))
            extra_line_kwargs = (
                {}
                if series_line_kwargs is None or series_line_kwargs[series_index] is None
                else dict(series_line_kwargs[series_index] or {})
            )
            if atom_descriptor_mode and isinstance(current_override.get("line_kwargs"), Mapping):
                extra_line_kwargs.update(dict(current_override.get("line_kwargs") or {}))
            series_index += 1
            if not is_enabled:
                continue

            x_values = np.asarray(x_matrix[:, atom_column], dtype=float)
            y_values = np.asarray(y_matrix[:, atom_column], dtype=float)
            value_values = np.asarray(value_matrix[:, atom_column], dtype=float)
            visible_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(value_values)
            if resolved_filter_min is not None:
                visible_mask &= value_values >= resolved_filter_min
            if resolved_filter_max is not None:
                visible_mask &= value_values <= resolved_filter_max
            if not np.any(visible_mask):
                continue

            visible_x = np.asarray(x_values[visible_mask], dtype=float)
            visible_y = np.asarray(y_values[visible_mask], dtype=float)
            visible_values = np.asarray(value_values[visible_mask], dtype=float)
            visible_x_values.extend(float(value) for value in visible_x)
            visible_y_values.extend(float(value) for value in visible_y)
            if resolved_render_mode == "color-scale":
                color_scale_point_x.extend(float(value) for value in visible_x)
                color_scale_point_y.extend(float(value) for value in visible_y)
                color_scale_point_values.extend(float(value) for value in visible_values)
                for start, stop in _contiguous_true_runs(visible_mask):
                    segments, segment_values = _build_xy_segments(
                        np.asarray(x_values[start:stop], dtype=float),
                        np.asarray(y_values[start:stop], dtype=float),
                        np.asarray(value_values[start:stop], dtype=float),
                        cell_lengths_xy=cell_lengths_xy,
                    )
                    if segments.size == 0:
                        continue
                    color_scale_segment_blocks.append(segments)
                    color_scale_segment_color_blocks.append(segment_values)
            else:
                paths: list[np.ndarray] = []
                for start, stop in _contiguous_true_runs(visible_mask):
                    paths.extend(
                        _build_projection_paths(
                            np.asarray(x_values[start:stop], dtype=float),
                            np.asarray(y_values[start:stop], dtype=float),
                            cell_lengths_xy=cell_lengths_xy,
                        )
                    )
                payload = {
                    "label": descriptor_label,
                    "show_in_legend": show_in_legend,
                    "base_color": line_color,
                    "line_width": line_width,
                    "line_kwargs": extra_line_kwargs,
                    "paths": [path for path in paths if len(path) >= 2],
                }
                if payload["paths"]:
                    line_series_payloads.append(payload)
                    profile_line_payloads.append(payload)
                    line_series_labels.append(descriptor_label)

        if profile_line_payloads:
            shaded_colors = _build_profile_line_shades(profile_base_color, len(profile_line_payloads))
            for payload, shaded_color in zip(profile_line_payloads, shaded_colors):
                payload["color"] = shaded_color

    if not visible_x_values:
        if (
            legacy_xy_z_alias
            and xy_z_distance_max is not None
            and projection_filter_min is None
            and projection_filter_max is None
            and resolved_projection_value == "distance"
        ):
            raise ValueError("No atom trajectories remain after applying the xy-z distance cutoff.")
        if resolved_filter_min is not None or resolved_filter_max is not None:
            raise ValueError(
                "No atom trajectories remain after applying the projection value filter."
            )
        raise ValueError("No enabled atom trajectories available for 2-D position plotting.")

    configure_matplotlib_backend(
        interactive=show,
        preferred_backend=preferred_backend,
    )
    rc_context_args: dict[str, Any] = {"font.family": style.font_family, "text.parse_math": True}
    if matplotlib_rc is not None:
        rc_context_args.update(dict(matplotlib_rc))

    labels = resolve_series_labels(line_series_labels, None, series_kind="position")
    effective_legend = (len(labels) <= 12) if legend is None else bool(legend)
    marker_size = max(9.0, (style.line_width * 7.0) ** 2)

    with plt.rc_context(rc_context_args):
        fig, ax = plt.subplots(figsize=style.figure_size)
        if figure_kwargs is not None:
            fig.set(**dict(figure_kwargs))

        if resolved_render_mode == "color-scale":
            color_samples = []
            if color_scale_segment_color_blocks:
                color_samples.extend(color_scale_segment_color_blocks)
            if color_scale_point_values:
                color_samples.append(np.asarray(color_scale_point_values, dtype=float))
            color_all = np.concatenate(color_samples)
            color_min = float(np.nanmin(color_all))
            color_max = float(np.nanmax(color_all))
            if color_min == color_max:
                color_min -= 0.5
                color_max += 0.5
            norm = mcolors.Normalize(vmin=color_min, vmax=color_max)
            projection_line_kwargs = _sanitize_line_collection_kwargs(line_kwargs)
            explicit_line_width = _first_non_none(series_line_widths)
            projection_line_kwargs.setdefault(
                "linewidths",
                style.line_width if explicit_line_width is None else explicit_line_width,
            )
            mappable = None
            if color_scale_segment_blocks:
                collection = LineCollection(
                    np.concatenate(color_scale_segment_blocks, axis=0),
                    cmap="turbo",
                    norm=norm,
                    **projection_line_kwargs,
                )
                collection.set_array(np.concatenate(color_scale_segment_color_blocks, axis=0))
                ax.add_collection(collection)
                mappable = collection
            if color_scale_point_x:
                scatter = ax.scatter(
                    np.asarray(color_scale_point_x, dtype=float),
                    np.asarray(color_scale_point_y, dtype=float),
                    c=np.asarray(color_scale_point_values, dtype=float),
                    cmap="turbo",
                    norm=norm,
                    s=marker_size,
                    edgecolors="none",
                )
                if mappable is None:
                    mappable = scatter
            if mappable is not None:
                colorbar = fig.colorbar(mappable, ax=ax)
                colorbar.set_label(
                    value_label_reference or "Projection value",
                    fontsize=style.label_font_size,
                )
                colorbar.ax.tick_params(labelsize=style.tick_font_size)
        else:
            for payload, resolved_label in zip(line_series_payloads, labels):
                label_used = False
                plot_kwargs = dict(line_kwargs or {})
                plot_kwargs.update(payload["line_kwargs"])
                plot_kwargs.setdefault("linewidth", payload["line_width"])
                plot_kwargs.setdefault("color", payload["color"])
                for marker_key in (
                    "marker",
                    "markersize",
                    "markeredgecolor",
                    "markeredgewidth",
                    "markerfacecolor",
                    "markerfacecoloralt",
                ):
                    plot_kwargs.pop(marker_key, None)
                for path in payload["paths"]:
                    ax.plot(
                        path[:, 0],
                        path[:, 1],
                        marker="",
                        label=(
                            resolved_label
                            if payload["show_in_legend"] and effective_legend and not label_used
                            else None
                        ),
                        **plot_kwargs,
                    )
                    label_used = True
            if effective_legend:
                resolved_legend_kwargs = dict(legend_kwargs or {})
                if legend_title is not None:
                    resolved_legend_kwargs.setdefault("title", legend_title)
                handles, legend_labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc=legend_loc, **resolved_legend_kwargs)

        ax.autoscale()
        xlabel_kwargs: dict[str, Any] = {"fontsize": x_label_font_size or style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": y_label_font_size or style.label_font_size}
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(
            format_axis_label_units(resolve_explicit_plot_text(x_label, default_x_label)),
            **xlabel_kwargs,
        )
        ax.set_ylabel(
            format_axis_label_units(resolve_explicit_plot_text(y_label, default_y_label)),
            **ylabel_kwargs,
        )
        ax.set_title(
            ""
            if title_visible is False
            else title
            or _projection_default_title(
                x_quantity=resolved_projection_x,
                y_quantity=resolved_projection_y,
                value_quantity=resolved_projection_value,
                render_mode=resolved_render_mode,
            ),
            fontsize=style.title_font_size,
            pad=style.title_pad if title_pad is None else float(title_pad),
        )
        ax.tick_params(axis="both", labelsize=style.tick_font_size)
        resolved_tick_params, tick_axis_hint, minor_ticks_mode = _extract_tick_controls(
            tick_params_kwargs
        )
        if minor_ticks_mode == "on":
            ax.minorticks_on()
        elif minor_ticks_mode == "off":
            ax.minorticks_off()
        if resolved_tick_params:
            ax.tick_params(**resolved_tick_params)
        x_ticks_visible, y_ticks_visible = _resolve_tick_visibility(
            tick_params_kwargs,
            ticks_visible,
            tick_axis_hint,
        )
        if not x_ticks_visible:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        if not y_ticks_visible:
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        if x_tick_rotation is not None:
            ax.tick_params(axis="x", rotation=float(x_tick_rotation))
        if y_tick_rotation is not None:
            ax.tick_params(axis="y", rotation=float(y_tick_rotation))
        if style.grid:
            resolved_grid_kwargs: dict[str, Any] = {
                "linestyle": style.grid_linestyle,
                "linewidth": style.grid_linewidth,
                "alpha": style.grid_alpha,
            }
            if grid_kwargs is not None:
                resolved_grid_kwargs.update(dict(grid_kwargs))
            ax.grid(True, **resolved_grid_kwargs)
        elif grid_kwargs is not None:
            ax.grid(**dict(grid_kwargs))
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        if x_ticks is not None:
            ax.set_xticks([float(value) for value in x_ticks])
        if y_ticks is not None:
            ax.set_yticks([float(value) for value in y_ticks])
        effective_x_lim = default_projection_x_lim if x_lim is None else x_lim
        effective_y_lim = default_projection_y_lim if y_lim is None else y_lim
        if effective_x_lim is not None:
            ax.set_xlim(
                left=None if effective_x_lim[0] is None else float(effective_x_lim[0]),
                right=None if effective_x_lim[1] is None else float(effective_x_lim[1]),
            )
        if effective_y_lim is not None:
            ax.set_ylim(
                bottom=None if effective_y_lim[0] is None else float(effective_y_lim[0]),
                top=None if effective_y_lim[1] is None else float(effective_y_lim[1]),
            )
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))
        annotation_summaries = _render_plot_annotations(ax, annotations)
        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()

        if capture_state is not None:
            capture_state.clear()
            capture_state.update(
                {
                    "figure": fig,
                    "axes": ax,
                    "title": str(ax.get_title()),
                    "title_visible": bool(
                        ax.title.get_visible() and bool(str(ax.get_title()).strip())
                    ),
                    "x_label": str(ax.get_xlabel()),
                    "y_label": str(ax.get_ylabel()),
                    "x_scale": str(ax.get_xscale()),
                    "y_scale": str(ax.get_yscale()),
                    "x_lim": [float(value) for value in ax.get_xlim()],
                    "y_lim": [float(value) for value in ax.get_ylim()],
                    "x_ticks": [float(value) for value in ax.get_xticks()],
                    "y_ticks": [float(value) for value in ax.get_yticks()],
                    "legend": bool(ax.get_legend() is not None),
                    "legend_title": (
                        None
                        if ax.get_legend() is None or ax.get_legend().get_title() is None
                        else str(ax.get_legend().get_title().get_text())
                    ),
                    "legend_loc": legend_loc,
                    "annotations_summary": annotation_summaries,
                    "projection_x": resolved_projection_x,
                    "projection_y": resolved_projection_y,
                    "projection_value": resolved_projection_value,
                    "projection_render_mode": resolved_render_mode,
                }
            )

        output_path = None
        if output is not None:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs: dict[str, Any] = {}
            if savefig_kwargs is not None:
                save_kwargs.update(dict(savefig_kwargs))
            save_kwargs.setdefault("dpi", style.dpi)
            fig.savefig(output_path, **save_kwargs)
            if not suppress_output_log:
                LOGGER.info("Saved plot to '%s'.", output_path)
        if show:
            plt.show(block=show_blocking)
            if not show_blocking:
                plt.pause(0.001)
        if not (show and not show_blocking):
            plt.close(fig)
        return output_path


def plot_position_profile(
    profile: PositionProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    component: str = "distance",
    map_color: str = "distance",
    time_axis: str = "ps",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
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
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    line_label: str | None = None,
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    error_config: dict[str, Any] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_cumulative_configs: list[dict[str, Any] | None] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    cumulative_config: dict[str, Any] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    projection_x: str | None = None,
    projection_y: str | None = None,
    projection_value: str | None = None,
    projection_render_mode: str | None = None,
    projection_filter_min: float | None = None,
    projection_filter_max: float | None = None,
    xy_z_distance_max: float | None = None,
) -> Path | None:
    """Plot one atom-resolved position profile."""
    resolved_mapping = resolve_position_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
        component=component,
        time_axis=time_axis,
        map_color=map_color,
        projection_x=projection_x,
        projection_y=projection_y,
        projection_value=projection_value,
        projection_render_mode=projection_render_mode,
        projection_filter_min=projection_filter_min,
        projection_filter_max=projection_filter_max,
        xy_z_distance_max=xy_z_distance_max,
    )
    runtime_options = resolved_mapping.renderer_options
    runtime_component = str(runtime_options.get("component") or "distance")
    runtime_time_axis = str(runtime_options.get("time_axis") or "ps")
    runtime_map_color = str(runtime_options.get("map_color") or "distance")
    runtime_projection_x = (
        None
        if runtime_options.get("projection_x") is None
        else str(runtime_options.get("projection_x"))
    )
    runtime_projection_y = (
        None
        if runtime_options.get("projection_y") is None
        else str(runtime_options.get("projection_y"))
    )
    runtime_projection_value = (
        None
        if runtime_options.get("projection_value") is None
        else str(runtime_options.get("projection_value"))
    )
    runtime_projection_render_mode = (
        None
        if runtime_options.get("projection_render_mode") is None
        else str(runtime_options.get("projection_render_mode"))
    )
    runtime_projection_filter_min = runtime_options.get("projection_filter_min")
    runtime_projection_filter_max = runtime_options.get("projection_filter_max")
    runtime_xy_z_distance_max = runtime_options.get("xy_z_distance_max")

    if _normalize_component_token(runtime_component) == _POSITION_PROJECTION_COMPONENT:
        return _plot_position_xy_z_projection(
            [profile],
            map_color=runtime_map_color,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            title=title,
            x_label=x_label,
            y_label=y_label,
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
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            render_series_descriptors=render_series_descriptors,
            series_overrides_by_id=series_overrides_by_id,
            series_line_kwargs=series_line_kwargs,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            legend_kwargs=legend_kwargs,
            annotations=annotations,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            component=runtime_component,
            projection_x=runtime_projection_x,
            projection_y=runtime_projection_y,
            projection_value=runtime_projection_value,
            projection_render_mode=runtime_projection_render_mode,
            projection_filter_min=runtime_projection_filter_min,
            projection_filter_max=runtime_projection_filter_max,
            xy_z_distance_max=runtime_xy_z_distance_max,
        )

    x_values, default_x_label = _position_time_data(profile, time_axis=runtime_time_axis)
    matrix, default_y_label = _position_component_data(profile, component=runtime_component)
    default_labels = _default_position_series_labels(profile)
    effective_legend = (profile.n_atoms <= 12) if legend is None else legend
    schema_labels = default_plot_labels("position")
    entity_descriptor = (
        "molecule-resolved" if profile.entity_kind == "molecule" else "atom-resolved"
    )
    default_title = (
        f"{_position_species_display_label(profile.species)} {entity_descriptor} positions"
        if schema_labels is not None
        else f"{entity_descriptor.capitalize()} positions"
    )
    labels = resolve_series_labels(default_labels, series_labels, series_kind="position")

    if matrix.shape[1] == 1:
        resolved_label = line_label
        if resolved_label is None and effective_legend:
            resolved_label = labels[0]
        single_series = resolve_single_series_options(
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
        )
        return plot_line_series(
            np.asarray(x_values, dtype=float),
            np.asarray(matrix[:, 0], dtype=float),
            title=title or default_title,
            x_label=resolve_explicit_plot_text(x_label, default_x_label),
            y_label=resolve_explicit_plot_text(y_label, default_y_label),
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            series_id=series_id,
            line_label=resolved_label,
            line_color=single_series.line_color,
            line_width_override=single_series.line_width_override,
            line_marker=single_series.line_marker,
            line_visible=single_series.line_visible,
            show_in_legend=True if not series_show_in_legend else bool(series_show_in_legend[0]),
            fit_config=None if not series_fit_configs else series_fit_configs[0],
            cumulative_config=cumulative_config,
            raw_point_statistics=True,
            error_config=error_config,
            normalization_mode=single_series.normalization_mode,
            normalization_value=single_series.normalization_value,
            normalization_x_ref=single_series.normalization_x_ref,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            min_bin_points=min_bin_points,
            analysis_name="position",
            annotations=annotations,
            integration_config=integration_config,
            style=style,
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
            legend=effective_legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            capture_state=capture_state,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            suppress_output_log=suppress_output_log,
        )

    return plot_multi_line_series(
        [np.asarray(x_values, dtype=float) for _ in range(matrix.shape[1])],
        [np.asarray(matrix[:, col], dtype=float) for col in range(matrix.shape[1])],
        labels,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, default_y_label),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_cumulative_configs=series_cumulative_configs,
        series_error_configs=(
            [error_config] * matrix.shape[1] if error_config is not None else None
        ),
        series_raw_statistics=[True] * matrix.shape[1],
        series_line_kwargs=series_line_kwargs,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="position",
        annotations=annotations,
        integration_config=integration_config,
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
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=effective_legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )


def plot_position_profiles(
    profiles: list[PositionProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    component: str = "distance",
    map_color: str = "distance",
    time_axis: str = "ps",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
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
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    series_ids: list[str] | None = None,
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
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
    capture_state: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    projection_x: str | None = None,
    projection_y: str | None = None,
    projection_value: str | None = None,
    projection_render_mode: str | None = None,
    projection_filter_min: float | None = None,
    projection_filter_max: float | None = None,
    xy_z_distance_max: float | None = None,
) -> Path | None:
    """Plot one or more atom-resolved position profiles."""
    if not profiles:
        raise ValueError("At least one position profile is required.")
    first_profile = profiles[0]
    resolved_mapping = resolve_position_plot_mapping(
        contract=data_contract,
        profile=first_profile,
        mapping=view_mapping,
        component=component,
        time_axis=time_axis,
        map_color=map_color,
        projection_x=projection_x,
        projection_y=projection_y,
        projection_value=projection_value,
        projection_render_mode=projection_render_mode,
        projection_filter_min=projection_filter_min,
        projection_filter_max=projection_filter_max,
        xy_z_distance_max=xy_z_distance_max,
    )
    runtime_options = resolved_mapping.renderer_options
    runtime_component = str(runtime_options.get("component") or "distance")
    runtime_time_axis = str(runtime_options.get("time_axis") or "ps")
    runtime_map_color = str(runtime_options.get("map_color") or "distance")
    runtime_projection_x = (
        None
        if runtime_options.get("projection_x") is None
        else str(runtime_options.get("projection_x"))
    )
    runtime_projection_y = (
        None
        if runtime_options.get("projection_y") is None
        else str(runtime_options.get("projection_y"))
    )
    runtime_projection_value = (
        None
        if runtime_options.get("projection_value") is None
        else str(runtime_options.get("projection_value"))
    )
    runtime_projection_render_mode = (
        None
        if runtime_options.get("projection_render_mode") is None
        else str(runtime_options.get("projection_render_mode"))
    )
    runtime_projection_filter_min = runtime_options.get("projection_filter_min")
    runtime_projection_filter_max = runtime_options.get("projection_filter_max")
    runtime_xy_z_distance_max = runtime_options.get("xy_z_distance_max")

    if _normalize_component_token(runtime_component) == _POSITION_PROJECTION_COMPONENT:
        return _plot_position_xy_z_projection(
            profiles,
            map_color=runtime_map_color,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            title=title,
            x_label=x_label,
            y_label=y_label,
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
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            render_series_descriptors=render_series_descriptors,
            series_overrides_by_id=series_overrides_by_id,
            series_line_kwargs=series_line_kwargs,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            legend_kwargs=legend_kwargs,
            annotations=annotations,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            component=runtime_component,
            projection_x=runtime_projection_x,
            projection_y=runtime_projection_y,
            projection_value=runtime_projection_value,
            projection_render_mode=runtime_projection_render_mode,
            projection_filter_min=runtime_projection_filter_min,
            projection_filter_max=runtime_projection_filter_max,
            xy_z_distance_max=runtime_xy_z_distance_max,
        )

    if not use_multi_series_plot(
        profile_count=len(profiles),
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
    ):
        return plot_position_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            data_contract=resolved_mapping.contract,
            view_mapping=resolved_mapping.mapping,
            component=runtime_component,
            map_color=runtime_map_color,
            projection_x=runtime_projection_x,
            projection_y=runtime_projection_y,
            projection_value=runtime_projection_value,
            projection_render_mode=runtime_projection_render_mode,
            projection_filter_min=runtime_projection_filter_min,
            projection_filter_max=runtime_projection_filter_max,
            xy_z_distance_max=runtime_xy_z_distance_max,
            time_axis=runtime_time_axis,
            title=title,
            x_label=x_label,
            y_label=y_label,
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
            x_axis_scale=x_axis_scale,
            x_axis_offset=x_axis_offset,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            series_id=None if not series_ids else str(series_ids[0]),
            series_labels=series_labels,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            series_cumulative_configs=series_cumulative_configs,
            render_series_descriptors=render_series_descriptors,
            series_overrides_by_id=series_overrides_by_id,
            cumulative_config=None
            if not series_cumulative_configs
            else series_cumulative_configs[0],
            error_config=None if not series_error_configs else series_error_configs[0],
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            series_line_kwargs=series_line_kwargs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            min_bin_points=min_bin_points,
            annotations=annotations,
            integration_config=integration_config,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    _x_template, default_x_label = _position_time_data(first_profile, time_axis=runtime_time_axis)
    _matrix, default_y_label = _position_component_data(
        first_profile,
        component=runtime_component,
    )
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []
    default_labels: list[str] = []
    for profile in profiles:
        x_values, _x_label = _position_time_data(profile, time_axis=runtime_time_axis)
        matrix, _y_label = _position_component_data(profile, component=runtime_component)
        for column, atom_index in enumerate(profile.atom_indices.tolist()):
            x_series.append(np.asarray(x_values, dtype=float))
            y_series.append(np.asarray(matrix[:, column], dtype=float))
            default_labels.append(
                f"{_position_species_display_label(profile.species)}[{int(atom_index)}]"
            )

    labels = resolve_series_labels(default_labels, series_labels, series_kind="position")
    effective_legend = (len(labels) <= 12) if legend is None else legend
    schema_labels = default_plot_labels("position")
    default_title = "Position profiles" if schema_labels is not None else "Position profile"

    return plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, default_y_label),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_ids=series_ids,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_cumulative_configs=series_cumulative_configs,
        series_error_configs=series_error_configs,
        series_raw_statistics=[True] * len(labels),
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="position",
        annotations=annotations,
        integration_config=integration_config,
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
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=effective_legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        series_line_kwargs=series_line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )
