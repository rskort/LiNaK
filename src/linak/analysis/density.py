"""Density analysis routines."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from ..storage.hdf5_utils import (
    write_linak_hdf5,
)
from .binning import (
    reconstruct_uniform_bin_edges_from_centers,
    resolve_uniform_bin_width_for_load,
    uniform_bin_width_from_edges,
)
from .schema import (
    build_profile_metadata,
    canonicalize_density_units,
    resolve_units_map,
)
from .statistics import (
    SeriesStatistics,
    block_mean_matrix,
    build_series_statistics,
    build_statistics_metadata,
    statistics_payload_from_series_map,
    statistics_series_map_from_datasets,
    resolve_block_slices,
)
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    _coerce_x_axis_linear_transform,
    _display_x_values,
    _prepare_plot_series_data,
    normalize_backend_name as normalize_backend_name,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
    resolve_single_series_options,
)
from ..progress import ProgressBar
from ..utils import axis_to_index, ensure_positive
from .common import (
    available_element_species,
    frame_has_usable_cell as _common_frame_has_usable_cell,
    normalize_species_query as _normalize_species_query,
    read_profile_payloads,
    read_profile_payloads_by_index,
    use_multi_series_plot,
    write_profile_collection,
)
from .surface import (
    SurfaceEstimate,
    SurfaceEstimatorOptions,
    _cell_histogram_bounds,
    _log_framewise_surface_alignment,
    _select_surface_estimate,
    _shift_axis_values_by_surface_per_frame,
    _surface_estimate_datasets,
    _surface_estimate_from_payload,
    _surface_estimate_supports_distance_mode,
    _surface_metadata_payload,
    _surface_metadata_view,
    estimate_surface_position as estimate_surface_position,
    estimate_surface_reference as estimate_surface_reference,
)
from .water import (
    water_molecule_triplets as _water_molecule_triplets,
    water_triplet_axis_values_with_masses as _water_triplet_axis_values_with_masses,
    water_triplet_geometry as _water_triplet_geometry,
    water_axis_values_per_frame as _water_axis_values_per_frame_impl,
    water_oxygen_indices as _water_oxygen_indices_impl,
)

LOGGER = logging.getLogger(__name__)
H2O_VALIDATION_STRIDE = 100
H2O_OH_CUTOFF_A = 1.25
AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24
ANGSTROM3_TO_NM3 = 1.0e-3
PLOT_AUTO_LIMIT_MARGIN_FRACTION = 0.05
H2O_MASS_G = float(
    (atomic_masses[atomic_numbers["H"]] * 2.0 + atomic_masses[atomic_numbers["O"]]) * AMU_TO_G
)


@dataclass(frozen=True)
class DensityProfile:
    """Container for a 1D density profile."""

    axis: str
    species: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts_per_frame: np.ndarray
    density: np.ndarray
    units: str
    n_frames: int
    entities_per_frame: np.ndarray | None = None
    number_density: np.ndarray | None = None
    number_density_units: str | None = None
    coordinate_mode: str = "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_estimate: SurfaceEstimate | None = None
    series_statistics: dict[str, SeriesStatistics] | None = None


def _frame_has_usable_cell(frame: Atoms, axis_index: int) -> bool:
    """Backward-compatible density cell validation wrapper."""
    return _common_frame_has_usable_cell(frame, axis_index=axis_index)


def _select_axis_values_with_masses(
    frame: Atoms, species: str, axis_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and masses for atoms matching ``species`` in one frame."""
    symbols = np.asarray(frame.get_chemical_symbols())
    mask = symbols == species
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)
    axis_values = np.asarray(frame.positions[mask, axis_index], dtype=float)
    masses = np.asarray(frame.get_masses()[mask], dtype=float) * AMU_TO_G
    return axis_values, masses


def _select_water_axis_values_with_masses(
    frame: Atoms, axis_index: int, oh_cutoff: float = H2O_OH_CUTOFF_A
) -> tuple[np.ndarray, np.ndarray]:
    """Return COM axis positions and molecular masses for detected water molecules."""
    triplets = _water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
    if triplets.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return _water_triplet_axis_values_with_masses(frame, triplets, axis_index)


def _select_water_triplet_axis_values_with_masses(
    frame: Atoms, water_triplets: np.ndarray, axis_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return PBC-aware water COM axis positions and molecular masses."""
    return _water_triplet_axis_values_with_masses(frame, water_triplets, axis_index)


def _select_water_axis_values_per_frame(
    frames: list[Atoms], axis_index: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select water-molecule COM axis values with periodic cached-topology validation."""
    return _water_axis_values_per_frame_impl(
        frames,
        axis_index,
        progress_desc="Selecting H2O for density",
    )


def _water_oxygen_indices(frame: Atoms, oh_cutoff: float = H2O_OH_CUTOFF_A) -> np.ndarray:
    """Return oxygen indices classified as water oxygens (exactly two unique H neighbors)."""
    return _water_oxygen_indices_impl(frame, oh_cutoff=oh_cutoff)


def _entity_density_units_for_species(
    species_label: str,
    *,
    volumetric: bool,
) -> str:
    entity_label = "molecule" if str(species_label).strip().upper() == "H2O" else "atom"
    if volumetric:
        return f"{entity_label}/nm^3"
    return f"{entity_label}/Angstrom"


def _density_normalization_mode(
    *,
    use_volumetric_density: bool,
    variable_slice_volume: bool,
) -> str:
    if use_volumetric_density and variable_slice_volume:
        return "framewise_volume_normalized"
    if use_volumetric_density:
        return "post_accumulate_volume_normalized"
    return "linear_per_length"


def _validate_selected_density_inputs(
    *,
    selected_per_frame: list[np.ndarray],
    selected_masses_per_frame: list[np.ndarray],
) -> None:
    if len(selected_per_frame) != len(selected_masses_per_frame):
        raise ValueError("Selected coordinate and mass arrays must have matching frame counts.")
    for frame_index, (axis_values, masses) in enumerate(
        zip(selected_per_frame, selected_masses_per_frame)
    ):
        if np.asarray(axis_values).shape != np.asarray(masses).shape:
            raise ValueError(
                "Selected coordinate and mass arrays must have matching entity counts "
                f"within each frame (frame {frame_index})."
            )


def _validate_binned_frame_conservation(
    *,
    frame_index: int,
    mass_histogram: np.ndarray,
    entity_histogram: np.ndarray,
    masses: np.ndarray,
    axis_values: np.ndarray,
) -> None:
    expected_mass = float(np.sum(masses))
    observed_mass = float(np.sum(mass_histogram))
    if not np.isclose(observed_mass, expected_mass, rtol=1.0e-12, atol=1.0e-30):
        raise ValueError(
            "Mass histogram does not conserve the selected mass in frame "
            f"{frame_index}: histogram={observed_mass:.16g}, selected={expected_mass:.16g}."
        )

    expected_entities = int(np.asarray(axis_values, dtype=float).size)
    observed_entities = int(np.rint(np.sum(entity_histogram)))
    if observed_entities != expected_entities:
        raise ValueError(
            "Entity histogram does not conserve the selected entity count in frame "
            f"{frame_index}: histogram={observed_entities}, selected={expected_entities}."
        )


def _compute_density_profile_from_selected(
    *,
    frames: list[Atoms],
    selected_per_frame: list[np.ndarray],
    selected_masses_per_frame: list[np.ndarray],
    axis: str,
    axis_index: int,
    species_label: str,
    count_label: str,
    bin_width: float,
    coordinate_mode: str = "axis",
    surface_position: float | None = None,
    surface_position_std: float | None = None,
    surface_estimate: SurfaceEstimate | None = None,
    histogram_bounds: tuple[float, float] | None = None,
    show_binning_progress: bool = True,
    aggregate_binning_progress: ProgressBar | None = None,
) -> DensityProfile:
    """Build a :class:`DensityProfile` from already-selected axis values."""
    _validate_selected_density_inputs(
        selected_per_frame=selected_per_frame,
        selected_masses_per_frame=selected_masses_per_frame,
    )
    n_selected_total = sum(values.size for values in selected_per_frame)
    if n_selected_total == 0:
        raise ValueError(f"No entities found for selection '{species_label}' in trajectory.")
    LOGGER.debug(
        "Selected %d total %s across %d frame(s).",
        n_selected_total,
        count_label,
        len(frames),
    )

    non_empty_selected = [values for values in selected_per_frame if values.size > 0]
    data_min = min(float(np.min(values)) for values in non_empty_selected)
    data_max = max(float(np.max(values)) for values in non_empty_selected)
    if histogram_bounds is not None:
        bounds_min = float(histogram_bounds[0])
        bounds_max = float(histogram_bounds[1])
        if np.isfinite(bounds_min) and np.isfinite(bounds_max) and bounds_max > bounds_min:
            data_min = min(data_min, bounds_min)
            data_max = max(data_max, bounds_max)
        else:
            LOGGER.warning(
                "Ignoring invalid histogram bounds for '%s': %s",
                species_label,
                histogram_bounds,
            )

    if np.isclose(data_min, data_max):
        data_max = data_min + bin_width

    span = data_max - data_min
    n_bins = max(1, int(np.ceil(span / bin_width)))
    bin_edges = data_min + np.arange(n_bins + 1, dtype=float) * bin_width
    if bin_edges[-1] <= data_max:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    LOGGER.debug(
        "Histogram bounds: min=%.6g, max=%.6g, bins=%d.",
        data_min,
        data_max,
        bin_edges.size - 1,
    )

    n_frames = len(frames)
    mass_histogram_sum = np.zeros(bin_edges.size - 1, dtype=float)
    entity_histogram_sum = np.zeros(bin_edges.size - 1, dtype=float)
    use_volumetric_density = all(_frame_has_usable_cell(frame, axis_index) for frame in frames)
    LOGGER.debug("Density mode: %s.", "volumetric" if use_volumetric_density else "linear")

    slice_volumes: np.ndarray | None = None
    variable_slice_volume = False
    if use_volumetric_density:
        slice_volumes = np.empty(n_frames, dtype=float)
        for i, frame in enumerate(frames):
            cell = np.asarray(frame.cell.array, dtype=float)
            axis_length = np.linalg.norm(cell[axis_index])
            volume = abs(float(np.linalg.det(cell)))
            cross_section = volume / axis_length
            slice_volumes[i] = cross_section * bin_width

        variable_slice_volume = not np.allclose(
            slice_volumes, slice_volumes[0], rtol=0.0, atol=1e-12
        )

    framewise_normalization = bool(use_volumetric_density and variable_slice_volume)
    normalization_mode = _density_normalization_mode(
        use_volumetric_density=use_volumetric_density,
        variable_slice_volume=variable_slice_volume,
    )
    LOGGER.debug(
        "Density normalization path for '%s': %s.",
        species_label,
        normalization_mode,
    )
    LOGGER.debug(
        "Binning '%s' on a fixed %s grid over [%.6g, %.6g] Angstrom.",
        species_label,
        coordinate_mode,
        data_min,
        bin_edges[-1],
    )

    framewise_mass_density_sum = (
        np.zeros_like(mass_histogram_sum) if framewise_normalization else None
    )
    framewise_entity_density_sum = (
        np.zeros_like(entity_histogram_sum) if framewise_normalization else None
    )
    per_frame_density_values: list[np.ndarray] = []
    per_frame_number_density_values: list[np.ndarray] = []
    progress_cm = (
        ProgressBar(
            desc=f"Binning {species_label} density",
            total=n_frames,
            unit="frame",
        )
        if show_binning_progress
        else nullcontext(None)
    )
    with progress_cm as progress:
        for frame_index, (axis_values, masses) in enumerate(
            zip(selected_per_frame, selected_masses_per_frame)
        ):
            # NumPy histograms use a fixed global bin grid per profile:
            # bins are left-inclusive, right-exclusive, except the final bin
            # which includes its right edge.
            per_frame_mass_histogram, _ = np.histogram(
                axis_values,
                bins=bin_edges,
                weights=masses,
            )
            per_frame_mass_histogram = per_frame_mass_histogram.astype(float)
            per_frame_entity_histogram, _ = np.histogram(axis_values, bins=bin_edges)
            per_frame_entity_histogram = per_frame_entity_histogram.astype(float)
            _validate_binned_frame_conservation(
                frame_index=frame_index,
                mass_histogram=per_frame_mass_histogram,
                entity_histogram=per_frame_entity_histogram,
                masses=np.asarray(masses, dtype=float),
                axis_values=np.asarray(axis_values, dtype=float),
            )
            mass_histogram_sum += per_frame_mass_histogram
            entity_histogram_sum += per_frame_entity_histogram
            if use_volumetric_density and slice_volumes is not None:
                frame_volume = float(slice_volumes[frame_index])
                frame_density = (per_frame_mass_histogram / frame_volume) / ANGSTROM3_TO_CM3
                frame_number_density = (
                    per_frame_entity_histogram / frame_volume
                ) / ANGSTROM3_TO_NM3
            else:
                frame_density = per_frame_mass_histogram / bin_width
                frame_number_density = per_frame_entity_histogram / bin_width
            per_frame_density_values.append(np.asarray(frame_density, dtype=float))
            per_frame_number_density_values.append(np.asarray(frame_number_density, dtype=float))
            if (
                framewise_mass_density_sum is not None
                and framewise_entity_density_sum is not None
                and slice_volumes is not None
            ):
                frame_volume = float(slice_volumes[frame_index])
                framewise_mass_density_sum += per_frame_mass_histogram / frame_volume
                framewise_entity_density_sum += per_frame_entity_histogram / frame_volume
            if progress is not None:
                progress.update()
    if aggregate_binning_progress is not None:
        aggregate_binning_progress.update()

    if use_volumetric_density and slice_volumes is not None:
        # With a variable cell, density is the time average of per-frame
        # volume-normalized histograms. With a constant cell, raw histograms
        # can be accumulated first and normalized once.
        if framewise_mass_density_sum is None or framewise_entity_density_sum is None:
            reference_slice_volume = float(slice_volumes[0])
            density = (mass_histogram_sum / reference_slice_volume) / n_frames
            number_density = (entity_histogram_sum / reference_slice_volume) / n_frames
        else:
            density = framewise_mass_density_sum / n_frames
            number_density = framewise_entity_density_sum / n_frames
        density = density / ANGSTROM3_TO_CM3
        units = "g/cm^3"
        number_density = number_density / ANGSTROM3_TO_NM3
    else:
        # Linear fallback is intentionally one-dimensional: mass or entity count
        # per length. It is not a volumetric concentration.
        density = (mass_histogram_sum / n_frames) / bin_width
        units = "g/Angstrom"
        number_density = (entity_histogram_sum / n_frames) / bin_width

    avg_binned_mass_per_bin = mass_histogram_sum / n_frames
    avg_entities_per_bin = entity_histogram_sum / n_frames
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    number_density_units = _entity_density_units_for_species(
        species_label,
        volumetric=use_volumetric_density,
    )
    sample_density_matrix = np.vstack(per_frame_density_values)
    sample_number_density_matrix = np.vstack(per_frame_number_density_values)
    block_resolution = resolve_block_slices(n_frames)
    block_slices = None if block_resolution is None else block_resolution[0]
    density_statistics = build_series_statistics(
        point_count=np.rint(entity_histogram_sum).astype(int),
        sample_values=sample_density_matrix,
        block_values=block_mean_matrix(sample_density_matrix, block_slices=block_slices),
    )
    number_density_statistics = build_series_statistics(
        point_count=np.rint(entity_histogram_sum).astype(int),
        sample_values=sample_number_density_matrix,
        block_values=block_mean_matrix(sample_number_density_matrix, block_slices=block_slices),
    )

    return DensityProfile(
        axis=axis.lower(),
        species=species_label,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_per_frame=avg_binned_mass_per_bin,
        density=density,
        units=units,
        n_frames=n_frames,
        entities_per_frame=avg_entities_per_bin,
        number_density=number_density,
        number_density_units=number_density_units,
        coordinate_mode=coordinate_mode,
        surface_position=surface_position,
        surface_position_std=surface_position_std,
        surface_estimate=surface_estimate,
        series_statistics={
            "density": density_statistics,
            "number_density": number_density_statistics,
        },
    )


def _summarize_density_run(
    *,
    frames: list[Atoms],
    axis_index: int,
    coordinate_mode: str,
    labels: list[str],
) -> tuple[str, str]:
    use_volumetric_density = all(_frame_has_usable_cell(frame, axis_index) for frame in frames)
    density_mode = "volumetric" if use_volumetric_density else "linear"
    normalization_mode = _density_normalization_mode(
        use_volumetric_density=use_volumetric_density,
        variable_slice_volume=bool(
            use_volumetric_density and _density_uses_variable_slice_volume(frames, axis_index)
        ),
    )
    summary = ", ".join(labels)
    return (
        density_mode,
        f"{len(labels)} profile(s): {summary}; {coordinate_mode} coordinates; {density_mode} density; {normalization_mode} normalization",
    )


def _density_uses_variable_slice_volume(frames: list[Atoms], axis_index: int) -> bool:
    if not frames or not all(_frame_has_usable_cell(frame, axis_index) for frame in frames):
        return False
    slice_volumes = np.empty(len(frames), dtype=float)
    for i, frame in enumerate(frames):
        cell = np.asarray(frame.cell.array, dtype=float)
        axis_length = np.linalg.norm(cell[axis_index])
        volume = abs(float(np.linalg.det(cell)))
        cross_section = volume / axis_length
        slice_volumes[i] = cross_section
    return not np.allclose(slice_volumes, slice_volumes[0], rtol=0.0, atol=1e-12)


def compute_density_profile(
    frames: list[Atoms],
    species: str | None = "all",
    axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "observed",
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
) -> DensityProfile:
    """Compute a 1D species mass-density profile along a Cartesian axis.

    Parameters
    ----------
    frames
        Trajectory frames.
    species
        Selection string: ``all`` (default), element symbol (for example: ``O``),
        or ``H2O`` for water-molecule density.
    axis
        Axis along which to compute density (`x`, `y`, or `z`).
    bin_width
        Histogram bin width in Angstrom.

    Returns
    -------
    DensityProfile
        Binned per-frame mass and mass-density values.
    """
    LOGGER.debug(
        "Computing density profile (species=%s, axis=%s, bin_width=%.6g).",
        species,
        axis,
        bin_width,
    )
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("bin_width", bin_width)
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    axis_index = axis_to_index(axis)
    surface_estimate: SurfaceEstimate | None
    if str(surface_mode).strip().lower() == "none":
        surface_estimate = None
    elif precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (len(frames),):
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
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    _surface_skipped = str(surface_mode).strip().lower() == "none"
    if surface_position is None and not _surface_skipped:
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            axis.lower(),
            axis.lower(),
        )
    elif surface_position is not None:
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_position,
            surface_position_std=surface_position_std,
        )
    selection_mode, species_label = _normalize_species_query(species, allow_h2o=True)
    count_label = "molecules" if selection_mode == "h2o" else "atoms"
    LOGGER.debug("Selection mode: %s (label=%s).", selection_mode, species_label)

    selected_per_frame: list[np.ndarray] = []
    selected_masses_per_frame: list[np.ndarray] = []
    if selection_mode == "h2o":
        selected_per_frame, selected_masses_per_frame = _select_water_axis_values_per_frame(
            frames, axis_index
        )
    else:
        with ProgressBar(
            desc=f"Selecting {species_label} for density", total=len(frames), unit="frame"
        ) as progress:
            for frame in frames:
                if selection_mode == "all":
                    axis_values = np.asarray(frame.positions[:, axis_index], dtype=float)
                    masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G
                else:
                    axis_values, masses = _select_axis_values_with_masses(
                        frame, species_label, axis_index
                    )
                selected_per_frame.append(axis_values)
                selected_masses_per_frame.append(masses)
                progress.update()
    trusted_surface_estimate = (
        surface_estimate
        if _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames))
        else None
    )
    selected_for_binning, coordinate_mode = _shift_axis_values_by_surface_per_frame(
        selected_per_frame,
        None if trusted_surface_estimate is None else trusted_surface_estimate.per_frame,
    )
    _density_mode, run_summary = _summarize_density_run(
        frames=frames,
        axis_index=axis_index,
        coordinate_mode=coordinate_mode,
        labels=[species_label],
    )
    LOGGER.info("Density compute summary: %s.", run_summary)
    histogram_bounds = None
    if normalized_binning == "cell":
        surface_per_frame = (
            None
            if coordinate_mode != "distance" or trusted_surface_estimate is None
            else trusted_surface_estimate.per_frame
        )
        histogram_bounds = _cell_histogram_bounds(
            frames=frames,
            axis_index=axis_index,
            coordinate_mode=coordinate_mode,
            surface_per_frame=surface_per_frame,
        )
        if histogram_bounds is None:
            LOGGER.warning(
                "Cell binning requested for '%s' along %s, but a usable cell was unavailable. "
                "Falling back to observed-data binning.",
                species_label,
                axis.lower(),
            )
    profile_surface_position = surface_position
    profile_surface_position_std = surface_position_std
    if surface_position is not None and coordinate_mode != "distance":
        LOGGER.warning(
            "Surface was estimated, but frame-wise surface alignment was unavailable; "
            "density bins remain on raw %s coordinates.",
            axis.lower(),
        )
    return _compute_density_profile_from_selected(
        frames=frames,
        selected_per_frame=selected_for_binning,
        selected_masses_per_frame=selected_masses_per_frame,
        axis=axis,
        axis_index=axis_index,
        species_label=species_label,
        count_label=count_label,
        bin_width=bin_width,
        coordinate_mode=coordinate_mode,
        surface_position=profile_surface_position,
        surface_position_std=profile_surface_position_std,
        surface_estimate=surface_estimate,
        histogram_bounds=histogram_bounds,
    )


def compute_density_profiles(
    frames: list[Atoms],
    species: str | None = "all",
    axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "observed",
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
) -> list[DensityProfile]:
    """Compute one or more density profiles based on the species selection policy."""
    ensure_positive("bin_width", bin_width)
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    selection_mode, _ = _normalize_species_query(species, allow_h2o=True)
    if selection_mode != "all":
        return [
            compute_density_profile(
                frames=frames,
                species=species,
                axis=axis,
                bin_width=bin_width,
                surface_mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                binning=normalized_binning,
                surface_options=surface_options,
                precomputed_surface_estimate=precomputed_surface_estimate,
            )
        ]

    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    surface_estimate: SurfaceEstimate | None
    if str(surface_mode).strip().lower() == "none":
        surface_estimate = None
    elif precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (len(frames),):
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
        )
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    _surface_skipped = str(surface_mode).strip().lower() == "none"
    if surface_position is None and not _surface_skipped:
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            axis.lower(),
            axis.lower(),
        )
    elif surface_position is not None:
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_position,
            surface_position_std=surface_position_std,
        )
    element_species = available_element_species(frames)
    if not element_species:
        raise ValueError("No elements found in trajectory.")

    axis_index = axis_to_index(axis)
    selected_by_species: dict[str, list[np.ndarray]] = {element: [] for element in element_species}
    selected_masses_by_species: dict[str, list[np.ndarray]] = {
        element: [] for element in element_species
    }
    empty = np.array([], dtype=float)
    with ProgressBar(
        desc="Selecting element data for density", total=len(frames), unit="frame"
    ) as progress:
        for frame in frames:
            symbols = np.asarray(frame.get_chemical_symbols())
            axis_values = np.asarray(frame.positions[:, axis_index], dtype=float)
            masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G

            frame_selected: dict[str, np.ndarray] = {}
            frame_selected_masses: dict[str, np.ndarray] = {}
            for symbol in np.unique(symbols):
                mask = symbols == symbol
                frame_selected[str(symbol)] = axis_values[mask]
                frame_selected_masses[str(symbol)] = masses[mask]

            for element in element_species:
                selected_by_species[element].append(frame_selected.get(element, empty))
                selected_masses_by_species[element].append(
                    frame_selected_masses.get(element, empty)
                )
            progress.update()

    profiles: list[DensityProfile] = []
    trusted_surface_estimate = (
        surface_estimate
        if _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames))
        else None
    )
    per_frame_surface = (
        None if trusted_surface_estimate is None else trusted_surface_estimate.per_frame
    )
    coordinate_mode_global = "distance" if trusted_surface_estimate is not None else "axis"
    _density_mode, run_summary = _summarize_density_run(
        frames=frames,
        axis_index=axis_index,
        coordinate_mode=coordinate_mode_global,
        labels=[*element_species, "H2O"],
    )
    LOGGER.info("Density compute summary: %s.", run_summary)
    histogram_bounds = None
    if normalized_binning == "cell":
        histogram_bounds = _cell_histogram_bounds(
            frames=frames,
            axis_index=axis_index,
            coordinate_mode=coordinate_mode_global,
            surface_per_frame=per_frame_surface,
        )
        if histogram_bounds is None:
            LOGGER.warning(
                "Cell binning requested for element-resolved density along %s, but a usable "
                "cell was unavailable. Falling back to observed-data binning.",
                axis.lower(),
            )
    water_selected_per_frame: list[np.ndarray] = []
    water_masses_per_frame: list[np.ndarray] = []
    water_selected_per_frame, water_masses_per_frame = _select_water_axis_values_per_frame(
        frames, axis_index
    )
    total_binning_jobs = len(element_species)
    if any(values.size > 0 for values in water_selected_per_frame):
        total_binning_jobs += 1
    LOGGER.info("Binning %d density profiles.", total_binning_jobs)
    with ProgressBar(
        desc="Binning density profiles", total=total_binning_jobs, unit="profile"
    ) as progress:
        for element in element_species:
            selected_for_binning, coordinate_mode = _shift_axis_values_by_surface_per_frame(
                selected_by_species[element],
                per_frame_surface,
            )
            profile_surface_position = surface_position
            profile_surface_position_std = surface_position_std
            profiles.append(
                _compute_density_profile_from_selected(
                    frames=frames,
                    selected_per_frame=selected_for_binning,
                    selected_masses_per_frame=selected_masses_by_species[element],
                    axis=axis,
                    axis_index=axis_index,
                    species_label=element,
                    count_label="atoms",
                    bin_width=bin_width,
                    coordinate_mode=coordinate_mode,
                    surface_position=profile_surface_position,
                    surface_position_std=profile_surface_position_std,
                    surface_estimate=surface_estimate,
                    histogram_bounds=histogram_bounds,
                    show_binning_progress=False,
                    aggregate_binning_progress=progress,
                )
            )

        if any(values.size > 0 for values in water_selected_per_frame):
            selected_for_binning, coordinate_mode = _shift_axis_values_by_surface_per_frame(
                water_selected_per_frame,
                per_frame_surface,
            )
            profile_surface_position = surface_position
            profile_surface_position_std = surface_position_std
            profiles.append(
                _compute_density_profile_from_selected(
                    frames=frames,
                    selected_per_frame=selected_for_binning,
                    selected_masses_per_frame=water_masses_per_frame,
                    axis=axis,
                    axis_index=axis_index,
                    species_label="H2O",
                    count_label="molecules",
                    bin_width=bin_width,
                    coordinate_mode=coordinate_mode,
                    surface_position=profile_surface_position,
                    surface_position_std=profile_surface_position_std,
                    surface_estimate=surface_estimate,
                    histogram_bounds=histogram_bounds,
                    show_binning_progress=False,
                    aggregate_binning_progress=progress,
                )
            )
    return profiles


def compute_all_density_profiles(
    frames: list[Atoms],
    species: str | None = "all",
    surface_axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "cell",
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
) -> list[DensityProfile]:
    """Compute raw X/Y/Z and distance density profiles in a single pass.

    This is the efficient replacement for calling ``compute_density_profiles``
    four times (once per raw axis and once for surface-distance).  All frame
    iteration, species selection, and water detection happen exactly once.
    """
    ensure_positive("bin_width", bin_width)
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    selection_mode, species_label = _normalize_species_query(species, allow_h2o=True)
    surface_axis_index = axis_to_index(surface_axis)
    raw_axes = ("x", "y", "z")
    axis_indices = {ax: axis_to_index(ax) for ax in raw_axes}

    # ---- Surface estimation (once, for distance mode only) ----
    surface_estimate: SurfaceEstimate | None
    if str(surface_mode).strip().lower() == "none":
        surface_estimate = None
    elif precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (len(frames),):
            raise ValueError(
                "precomputed_surface_estimate frame count does not match the trajectory."
            )
        surface_estimate = precomputed_surface_estimate
    else:
        surface_estimate, _surface_method = _select_surface_estimate(
            frames,
            surface_axis,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    if surface_position is None and str(surface_mode).strip().lower() != "none":
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            surface_axis.lower(),
            surface_axis.lower(),
        )
    elif surface_position is not None:
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=surface_axis,
            surface_position=surface_position,
            surface_position_std=surface_position_std,
        )

    trusted_surface_estimate = (
        surface_estimate
        if _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames))
        else None
    )
    per_frame_surface = (
        None if trusted_surface_estimate is None else trusted_surface_estimate.per_frame
    )

    # ---- Determine species to iterate ----
    if selection_mode == "all":
        element_species = available_element_species(frames)
        if not element_species:
            raise ValueError("No elements found in trajectory.")
    elif selection_mode == "h2o":
        element_species = []
    else:
        element_species = [species_label]

    include_water = selection_mode in ("all", "h2o")

    # ---- Single pass: extract per-frame data for all axes at once ----
    # selected[axis][element] = list of per-frame arrays
    selected: dict[str, dict[str, list[np.ndarray]]] = {
        ax: {el: [] for el in element_species} for ax in raw_axes
    }
    masses_by_species: dict[str, list[np.ndarray]] = {el: [] for el in element_species}
    water_selected: dict[str, list[np.ndarray]] = {ax: [] for ax in raw_axes}
    water_masses: list[np.ndarray] = []
    empty = np.array([], dtype=float)
    cached_water_triplets: np.ndarray | None = None

    with ProgressBar(
        desc="Selecting species for density (all axes)", total=len(frames), unit="frame"
    ) as progress:
        for frame_index, frame in enumerate(frames):
            # -- Element selection (all axes extracted at once) --
            if element_species:
                if selection_mode == "all":
                    symbols = np.asarray(frame.get_chemical_symbols())
                    positions = np.asarray(frame.positions, dtype=float)
                    masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G
                    frame_selected: dict[str, np.ndarray] = {}
                    frame_selected_masses: dict[str, np.ndarray] = {}
                    for symbol in np.unique(symbols):
                        mask = symbols == symbol
                        frame_selected[str(symbol)] = positions[mask]
                        frame_selected_masses[str(symbol)] = masses[mask]
                    for element in element_species:
                        pos_block = frame_selected.get(element)
                        mass_block = frame_selected_masses.get(element, empty)
                        if pos_block is not None:
                            for ax in raw_axes:
                                selected[ax][element].append(pos_block[:, axis_indices[ax]])
                            masses_by_species[element].append(mass_block)
                        else:
                            for ax in raw_axes:
                                selected[ax][element].append(empty)
                            masses_by_species[element].append(empty)
                else:
                    # Single element selection
                    el = element_species[0]
                    ax_vals, masses = _select_axis_values_with_masses(
                        frame, el, 0
                    )
                    if ax_vals.size > 0:
                        positions = np.asarray(frame.positions, dtype=float)
                        symbols = np.asarray(frame.get_chemical_symbols())
                        mask = symbols == el
                        full_pos = positions[mask]
                        for ax in raw_axes:
                            selected[ax][el].append(full_pos[:, axis_indices[ax]])
                        masses_by_species[el].append(masses)
                    else:
                        for ax in raw_axes:
                            selected[ax][el].append(empty)
                        masses_by_species[el].append(empty)

            # -- Water detection (once per frame, all axes extracted) --
            if include_water:
                if cached_water_triplets is None:
                    cached_water_triplets = _water_molecule_triplets(frame)
                elif frame_index % H2O_VALIDATION_STRIDE == 0:
                    validated = _water_molecule_triplets(frame)
                    if not np.array_equal(validated, cached_water_triplets):
                        LOGGER.warning(
                            "Detected H2O topology change at frame %d; "
                            "refreshing cached water triplets.",
                            frame_index,
                        )
                        cached_water_triplets = validated

                geom = _water_triplet_geometry(frame, cached_water_triplets)
                if geom.com_positions.size > 0:
                    for ax in raw_axes:
                        water_selected[ax].append(
                            np.asarray(geom.com_positions[:, axis_indices[ax]], dtype=float)
                        )
                    water_masses.append(
                        np.asarray(geom.molecular_masses_amu * AMU_TO_G, dtype=float)
                    )
                else:
                    for ax in raw_axes:
                        water_selected[ax].append(empty)
                    water_masses.append(empty)

            progress.update()

    LOGGER.info(
        "Single-pass selection complete: %d element species + %s water, %d frames.",
        len(element_species),
        "with" if include_water else "no",
        len(frames),
    )

    # ---- Build profiles for each (species, axis/mode) combination ----
    profiles: list[DensityProfile] = []
    water_present = include_water and any(v.size > 0 for v in water_selected.get("x", []))
    total_binning_jobs = len(element_species) * 4 + (4 if water_present else 0)
    LOGGER.info("Binning %d density profiles.", total_binning_jobs)

    # Helper to compute histogram bounds and build profiles for a species
    def _build_profiles_for_species(
        species_lbl: str,
        count_lbl: str,
        raw_data: dict[str, list[np.ndarray]],
        mass_data: list[np.ndarray],
        *,
        progress: ProgressBar,
    ) -> None:
        # Raw axis profiles (surface_mode="none")
        for ax in raw_axes:
            axis_idx = axis_indices[ax]
            hist_bounds = None
            if normalized_binning == "cell":
                hist_bounds = _cell_histogram_bounds(
                    frames=frames,
                    axis_index=axis_idx,
                    coordinate_mode="axis",
                    surface_per_frame=None,
                )
                if hist_bounds is None:
                    LOGGER.warning(
                        "Cell binning requested for '%s' along %s, but a usable cell was "
                        "unavailable. Falling back to observed-data binning.",
                        species_lbl,
                        ax,
                    )
            profiles.append(
                _compute_density_profile_from_selected(
                    frames=frames,
                    selected_per_frame=raw_data[ax],
                    selected_masses_per_frame=mass_data,
                    axis=ax,
                    axis_index=axis_idx,
                    species_label=species_lbl,
                    count_label=count_lbl,
                    bin_width=bin_width,
                    coordinate_mode="axis",
                    surface_position=None,
                    surface_position_std=None,
                    surface_estimate=None,
                    histogram_bounds=hist_bounds,
                    show_binning_progress=False,
                    aggregate_binning_progress=progress,
                )
            )

        # Distance profile (surface-shifted along surface_axis)
        shifted_data, coord_mode = _shift_axis_values_by_surface_per_frame(
            raw_data[surface_axis],
            per_frame_surface,
        )
        dist_hist_bounds = None
        if normalized_binning == "cell":
            dist_hist_bounds = _cell_histogram_bounds(
                frames=frames,
                axis_index=surface_axis_index,
                coordinate_mode=coord_mode,
                surface_per_frame=per_frame_surface if coord_mode == "distance" else None,
            )
            if dist_hist_bounds is None:
                LOGGER.warning(
                    "Cell binning requested for '%s' distance along %s, but a usable cell was "
                    "unavailable. Falling back to observed-data binning.",
                    species_lbl,
                    surface_axis,
                )
        profiles.append(
            _compute_density_profile_from_selected(
                frames=frames,
                selected_per_frame=shifted_data,
                selected_masses_per_frame=mass_data,
                axis=surface_axis,
                axis_index=surface_axis_index,
                species_label=species_lbl,
                count_label=count_lbl,
                bin_width=bin_width,
                coordinate_mode=coord_mode,
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_estimate=surface_estimate,
                histogram_bounds=dist_hist_bounds,
                show_binning_progress=False,
                aggregate_binning_progress=progress,
            )
        )

    # Element species
    with ProgressBar(
        desc="Binning density profiles", total=total_binning_jobs, unit="profile"
    ) as progress:
        for element in element_species:
            _build_profiles_for_species(
                element,
                "atoms",
                {ax: selected[ax][element] for ax in raw_axes},
                masses_by_species[element],
                progress=progress,
            )

        # Water
        if water_present:
            _build_profiles_for_species(
                "H2O",
                "molecules",
                water_selected,
                water_masses,
                progress=progress,
            )

    return profiles


def _density_profile_hdf5_payload(profile: DensityProfile) -> dict[str, Any]:
    """Return validated HDF5 datasets/metadata payload for one density profile."""
    bin_edges = np.asarray(profile.bin_edges, dtype=float)
    bin_centers = np.asarray(profile.bin_centers, dtype=float)
    if bin_edges.size != bin_centers.size + 1:
        raise ValueError(
            "Density profile bin_edges and bin_centers sizes are inconsistent "
            f"(edges={bin_edges.size}, centers={bin_centers.size})."
        )
    expected_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if not np.allclose(expected_centers, bin_centers, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("Density profile bin_centers are inconsistent with bin_edges.")
    bin_width = uniform_bin_width_from_edges(
        bin_edges,
        source_label=f"Density profile '{profile.species}'",
    )

    canonical_density, canonical_density_units, canonical_number_density, canonical_number_units = (
        canonicalize_density_units(
            density=profile.density,
            density_units=profile.units,
            number_density=profile.number_density,
            number_density_units=profile.number_density_units,
        )
    )

    units_map = {
        "density": canonical_density_units,
    }
    if canonical_number_units is not None:
        units_map["number_density"] = canonical_number_units

    metadata_payload = {
        "axis": profile.axis,
        "species": profile.species,
        "units": canonical_density_units,
        "n_frames": profile.n_frames,
        "number_density_units": canonical_number_units,
        "coordinate_mode": profile.coordinate_mode,
        "bin_width_A": bin_width,
        "counts_per_frame_available": False,
        **_surface_metadata_payload(
            surface_position=profile.surface_position,
            surface_position_std=profile.surface_position_std,
            estimate=profile.surface_estimate,
        ),
    }
    if profile.series_statistics:
        block_resolution = resolve_block_slices(int(profile.n_frames))
        block_lengths = None if block_resolution is None else block_resolution[1]
        metadata_payload["statistics"] = build_statistics_metadata(
            statistics_by_series=profile.series_statistics,
            block_lengths=block_lengths,
        )
    metadata = build_profile_metadata(
        analysis="density",
        metadata=metadata_payload,
        units_map=units_map,
    )
    datasets = {
        "bin_centers_A": profile.bin_centers,
        "density": canonical_density,
        "number_density": canonical_number_density,
        **statistics_payload_from_series_map(profile.series_statistics),
        **_surface_estimate_datasets(profile.surface_estimate),
    }
    return {
        "datasets": datasets,
        "metadata": metadata,
    }


def save_density_profile(
    profile: DensityProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a density profile to LiNaK HDF5 and return the written path."""
    payload = _density_profile_hdf5_payload(profile)
    metadata = dict(payload["metadata"])
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="density",
        datasets=payload["datasets"],
        metadata=metadata,
    )
    LOGGER.info("Saved density data to '%s'.", output_path)
    return output_path


def save_density_profiles(
    profiles: list[DensityProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one or more density profiles to LiNaK HDF5 and return the written path."""
    if not profiles:
        raise ValueError("At least one density profile is required.")
    if len(profiles) == 1:
        return save_density_profile(
            profiles[0],
            output,
            additional_metadata=additional_metadata,
        )

    output_path = write_profile_collection(
        output,
        analysis="density",
        profiles=[_density_profile_hdf5_payload(profile) for profile in profiles],
        metadata=dict(additional_metadata or {}),
    )
    LOGGER.info("Saved %d density profiles to '%s'.", len(profiles), output_path)
    return output_path


def load_density_profile(
    path: str | Path,
    *,
    axis: str | None = None,
    species: str | None = None,
) -> DensityProfile:
    """Load one density profile from LiNaK HDF5.

    For profile-collection files, this returns the first profile.
    """
    profiles = load_density_profiles(path, axis=axis, species=species)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(f"Density HDF5 '{source_path}' does not contain any density profiles.")
    return profiles[0]


def _density_payload_matches_selection(
    metadata: Mapping[str, Any],
    *,
    axis: str | None = None,
    species: str | None = None,
) -> bool:
    requested_axis = None if axis is None or not str(axis).strip() else str(axis).strip().lower()
    if requested_axis is not None:
        metadata_axis = str(metadata.get("axis", "z")).strip().lower() or "z"
        if metadata_axis != requested_axis:
            return False

    requested_species = None if species is None or not str(species).strip() else str(species)
    if requested_species is not None:
        selection_mode, requested_label = _normalize_species_query(
            requested_species,
            allow_h2o=True,
        )
        if selection_mode != "all":
            metadata_species = str(metadata.get("species", "")).strip()
            if not metadata_species:
                return False
            metadata_mode, metadata_label = _normalize_species_query(
                metadata_species,
                allow_h2o=True,
            )
            if metadata_mode != selection_mode or metadata_label != requested_label:
                return False

    return True


def _load_density_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    axis: str | None = None,
    species: str | None = None,
) -> list[DensityProfile]:
    profiles: list[DensityProfile] = []
    for datasets, metadata in payloads:
        if not _density_payload_matches_selection(metadata, axis=axis, species=species):
            continue
        required = ("bin_centers_A", "density")
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Density HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )

        axis_meta = str(metadata.get("axis", "")).strip()
        if axis is not None and axis.strip():
            axis_label = "xyz"[axis_to_index(axis)]
        elif axis_meta:
            axis_label = "xyz"[axis_to_index(axis_meta)]
        else:
            axis_label = "z"

        species_meta = str(metadata.get("species", "")).strip()
        if species is not None and species.strip():
            _selection_mode, species_label = _normalize_species_query(species, allow_h2o=True)
        elif species_meta:
            species_label = species_meta
        else:
            species_label = "UNKNOWN"

        resolved_units = resolve_units_map(analysis="density", metadata=metadata)
        units = str(metadata.get("units") or resolved_units.get("density") or "g/cm^3")
        number_density_units_raw = metadata.get("number_density_units")
        if number_density_units_raw is None:
            number_density_units_raw = resolved_units.get("number_density")
        number_density_units = (
            str(number_density_units_raw).strip() if number_density_units_raw is not None else None
        )
        if number_density_units == "":
            number_density_units = None
        coordinate_mode = str(metadata.get("coordinate_mode", "axis")).strip().lower()
        if coordinate_mode not in {"axis", "distance"}:
            coordinate_mode = "axis"
        series_statistics = statistics_series_map_from_datasets(
            datasets,
            dataset_names=("density", "number_density"),
        )

        surface_metadata = _surface_metadata_view(metadata)
        surface_position_raw = surface_metadata.get("position", metadata.get("surface_position"))
        surface_position = None
        if surface_position_raw is not None:
            value = float(surface_position_raw)
            if np.isfinite(value):
                surface_position = value

        surface_std_raw = surface_metadata.get("position_std", metadata.get("surface_position_std"))
        surface_position_std = None
        if surface_std_raw is not None:
            value = float(surface_std_raw)
            if np.isfinite(value):
                surface_position_std = value

        surface_estimate = _surface_estimate_from_payload(
            datasets=datasets,
            metadata=metadata,
        )

        number_density = None
        if "number_density" in datasets:
            number_density = np.asarray(datasets["number_density"], dtype=float)

        canonical_density, canonical_units, canonical_number_density, canonical_number_units = (
            canonicalize_density_units(
                density=np.asarray(datasets["density"], dtype=float),
                density_units=units,
                number_density=number_density,
                number_density_units=number_density_units,
            )
        )

        entities_per_frame = None
        if "entities_per_frame" in datasets:
            entities_per_frame = np.asarray(datasets["entities_per_frame"], dtype=float)

        bin_centers = np.asarray(datasets["bin_centers_A"], dtype=float)
        bin_width = resolve_uniform_bin_width_for_load(
            metadata=metadata,
            bin_centers=bin_centers,
            source_path=source_path,
            analysis_name="Density",
        )
        bin_edges = reconstruct_uniform_bin_edges_from_centers(
            bin_centers,
            bin_width=bin_width,
        )
        if bin_edges.size != bin_centers.size + 1:
            raise ValueError(f"Density HDF5 '{source_path}' has incompatible bin geometry.")

        if "counts_per_frame" in datasets:
            counts_per_frame = np.asarray(datasets["counts_per_frame"], dtype=float)
        else:
            counts_per_frame = np.full(bin_centers.shape, np.nan, dtype=float)

        profiles.append(
            DensityProfile(
                axis=axis_label,
                species=species_label,
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                counts_per_frame=counts_per_frame,
                density=canonical_density,
                units=canonical_units,
                n_frames=int(metadata.get("n_frames", 0)),
                entities_per_frame=entities_per_frame,
                number_density=canonical_number_density,
                number_density_units=canonical_number_units,
                coordinate_mode=coordinate_mode,
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_estimate=surface_estimate,
                series_statistics=series_statistics,
            )
        )
    return profiles


def load_density_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    axis: str | None = None,
    species: str | None = None,
) -> list[DensityProfile]:
    """Load selected density profiles by profile index from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="density",
        label="Density",
    )
    return _load_density_profiles_from_payloads(
        source_path,
        payloads,
        axis=axis,
        species=species,
    )


def load_density_profiles(
    path: str | Path,
    *,
    axis: str | None = None,
    species: str | None = None,
) -> list[DensityProfile]:
    """Load one or more density profiles from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads(
        path,
        analysis="density",
        label="Density",
    )
    return _load_density_profiles_from_payloads(
        source_path,
        payloads,
        axis=axis,
        species=species,
    )


def _format_plot_density_units(units: str) -> str:
    return units.replace("Angstrom", "A")


_ENTITY_NUMBER_DENSITY_PATTERN = re.compile(r"^(atom|molecule|atoms|molecules)(/.*)")


def _unify_number_density_units(all_units: list[str]) -> str | None:
    """Return a unified unit string when units differ only in the entity label (atom vs molecule).

    Returns ``None`` when the units are incompatible beyond the entity label.
    """
    if not all_units:
        return None
    suffixes: set[str] = set()
    for unit in all_units:
        match = _ENTITY_NUMBER_DENSITY_PATTERN.match(unit)
        if match is None:
            return None
        suffixes.add(match.group(2))
    if len(suffixes) != 1:
        return None
    return f"entities{suffixes.pop()}"


def _profile_has_surface_reference(profile: DensityProfile) -> bool:
    if profile.coordinate_mode == "distance":
        return True
    return profile.surface_position is not None and np.isfinite(profile.surface_position)


def _density_x_data(
    profile: DensityProfile,
    *,
    x_mode: str,
) -> tuple[np.ndarray, str]:
    normalized_x_mode = str(x_mode).strip().lower() or "distance"
    if normalized_x_mode in {"x", "y", "z"} and profile.axis != normalized_x_mode:
        raise ValueError(
            f"Density profile '{profile.species}' is stored along the {profile.axis.upper()} axis, "
            f"so it cannot be plotted against {normalized_x_mode.upper()} coordinates."
        )

    if normalized_x_mode in {"axis", "x", "y", "z"}:
        if profile.coordinate_mode == "distance":
            if profile.surface_position is not None and np.isfinite(profile.surface_position):
                return profile.bin_centers + float(
                    profile.surface_position
                ), f"{profile.axis.upper()} (A)"
            LOGGER.warning(
                "Density profile '%s' stores distance-aligned bins with no absolute surface "
                "offset; using distance coordinates.",
                profile.species,
            )
            return profile.bin_centers, "Distance to the surface ($\\mathrm{\\AA}$)"
        return profile.bin_centers, f"{profile.axis.upper()} (A)"

    if normalized_x_mode == "distance":
        if profile.coordinate_mode == "distance":
            return profile.bin_centers, "Distance to the surface ($\\mathrm{\\AA}$)"
        if _profile_has_surface_reference(profile):
            assert profile.surface_position is not None
            return (
                profile.bin_centers - float(profile.surface_position),
                "Distance to the surface ($\\mathrm{\\AA}$)",
            )
        LOGGER.warning(
            "Density profile '%s' has no surface reference; falling back to axis coordinates.",
            profile.species,
        )
        return profile.bin_centers, f"{profile.axis.upper()} (A)"

    raise ValueError(
        f"Unsupported density x_mode '{x_mode}'. Choose 'distance', 'x', 'y', 'z', or legacy 'axis'."
    )


def _density_y_data(
    profile: DensityProfile,
    *,
    quantity: str,
) -> tuple[np.ndarray, str, str]:
    if quantity == "number":
        if profile.number_density is None or profile.number_density_units is None:
            raise ValueError(
                f"Number-density data unavailable for profile '{profile.species}'. "
                "Use quantity='mass' or recompute volumetric density."
            )
        _mass, _mass_units, canonical_number, canonical_number_units = canonicalize_density_units(
            density=profile.density,
            density_units=profile.units,
            number_density=profile.number_density,
            number_density_units=profile.number_density_units,
        )
        assert canonical_number is not None
        assert canonical_number_units is not None
        return canonical_number, canonical_number_units, "Entity density"

    if quantity == "mass":
        density_values, units, _number, _number_units = canonicalize_density_units(
            density=profile.density,
            density_units=profile.units,
            number_density=profile.number_density,
            number_density_units=profile.number_density_units,
        )
        return density_values, units, "Density"

    raise ValueError(f"Unsupported density quantity '{quantity}'. Choose 'mass' or 'number'.")


def _density_profile_visible_for_x_mode(profile: DensityProfile, *, x_mode: str) -> bool:
    normalized_x_mode = str(x_mode).strip().lower() or "distance"
    if normalized_x_mode in {"x", "y", "z"}:
        return (
            profile.coordinate_mode != "distance"
            and str(profile.axis).strip().lower() == normalized_x_mode
        )
    return True


def _expand_linear_limits(lower: float, upper: float) -> list[float]:
    if np.isclose(lower, upper):
        margin = max(abs(lower), 1.0) * PLOT_AUTO_LIMIT_MARGIN_FRACTION
    else:
        margin = abs(upper - lower) * PLOT_AUTO_LIMIT_MARGIN_FRACTION
    if margin <= 0.0:
        margin = PLOT_AUTO_LIMIT_MARGIN_FRACTION
    return [float(lower - margin), float(upper + margin)]


def _resolve_auto_axis_limits(
    values: np.ndarray,
    *,
    scale: str,
    clamp_nonnegative_to_zero: bool = False,
) -> list[float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    normalized_scale = scale.strip().lower()
    if normalized_scale == "linear":
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        if clamp_nonnegative_to_zero and lower >= 0.0:
            if upper <= 0.0:
                return [0.0, 1.0]
            margin = max(upper * PLOT_AUTO_LIMIT_MARGIN_FRACTION, 1.0e-12)
            return [0.0, float(upper + margin)]
        return _expand_linear_limits(lower, upper)

    if normalized_scale == "log":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return None
        lower = float(np.min(positive))
        upper = float(np.max(positive))
        if np.isclose(lower, upper):
            return [lower / 1.2, upper * 1.2]
        factor = float(np.exp(np.log(upper / lower) * PLOT_AUTO_LIMIT_MARGIN_FRACTION))
        return [lower / factor, upper * factor]

    return None


def _merge_plot_limits(
    requested: tuple[float | None, float | None] | list[float | None] | None,
    auto: list[float] | None,
) -> list[float | None] | None:
    if requested is None:
        return None if auto is None else [float(auto[0]), float(auto[1])]

    resolved: list[float | None] = [
        None if requested[0] is None else float(requested[0]),
        None if requested[1] is None else float(requested[1]),
    ]
    if auto is None:
        return resolved
    if resolved[0] is None:
        resolved[0] = float(auto[0])
    if resolved[1] is None:
        resolved[1] = float(auto[1])
    return resolved


def _density_auto_plot_limits(
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    *,
    x_scale: str,
    y_scale: str,
) -> tuple[list[float] | None, list[float] | None]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    nonzero_x: list[np.ndarray] = []
    nonzero_y: list[np.ndarray] = []

    for x_values, y_values in zip(x_series, y_series):
        x_data = np.asarray(x_values, dtype=float)
        y_data = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        if not np.any(finite_mask):
            continue
        x_finite = x_data[finite_mask]
        y_finite = y_data[finite_mask]
        all_x.append(x_finite)
        all_y.append(y_finite)

        nonzero_mask = y_finite != 0.0
        if np.any(nonzero_mask):
            nonzero_x.append(x_finite[nonzero_mask])
            nonzero_y.append(y_finite[nonzero_mask])

    if not all_x:
        return None, None

    x_focus = np.concatenate(nonzero_x) if nonzero_x else np.concatenate(all_x)
    y_focus = np.concatenate(nonzero_y) if nonzero_y else np.concatenate(all_y)
    auto_x = _resolve_auto_axis_limits(
        x_focus,
        scale=x_scale,
        clamp_nonnegative_to_zero=False,
    )
    auto_y = _resolve_auto_axis_limits(
        y_focus,
        scale=y_scale,
        clamp_nonnegative_to_zero=True,
    )
    return auto_x, auto_y


def _prepared_density_auto_limit_series(
    *,
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    labels: list[str],
    series_enabled: list[bool] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    visible_indices = list(range(len(labels)))
    if series_enabled is not None and len(series_enabled) == len(labels):
        visible_indices = [index for index, enabled in enumerate(series_enabled) if bool(enabled)]
    if not visible_indices:
        return [], []

    def _select_visible(values: list[Any] | None) -> list[Any] | None:
        if values is None:
            return None
        if len(values) != len(labels):
            return values
        return [values[index] for index in visible_indices]

    prepared_x, prepared_y, _normalized_count = _prepare_plot_series_data(
        x_series=[x_series[index] for index in visible_indices],
        y_series=[y_series[index] for index in visible_indices],
        labels=[labels[index] for index in visible_indices],
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        series_normalization_modes=_select_visible(series_normalization_modes),
        series_normalization_values=_select_visible(series_normalization_values),
        series_normalization_x_refs=_select_visible(series_normalization_x_refs),
    )
    return prepared_x, prepared_y


def _has_active_normalization(
    modes: list[str | None] | None,
    *,
    enabled: list[bool] | None = None,
) -> bool:
    if not modes:
        return False
    for index, raw_mode in enumerate(modes):
        if enabled is not None and index < len(enabled) and not bool(enabled[index]):
            continue
        if str(raw_mode or "").strip().lower() not in {"", "none"}:
            return True
    return False


def plot_density_profile(
    profile: DensityProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_mode: str = "distance",
    quantity: str = "mass",
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
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    line_label: str | None = None,
    line_colors: list[str] | None = None,
    error_config: dict[str, Any] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    cumulative_config: dict[str, Any] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
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
) -> Path | None:
    """Plot and optionally save a density profile."""
    x_values, default_x_label = _density_x_data(profile, x_mode=x_mode)
    density_values, units, y_label_prefix = _density_y_data(profile, quantity=quantity)
    units = _format_plot_density_units(units)
    resolved_line_label = line_label
    if resolved_line_label is None and legend:
        resolved_line_label = profile.species
    single_series = resolve_single_series_options(
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
    )
    resolved_x_lim = x_lim
    resolved_y_lim = y_lim
    if not _has_active_normalization([single_series.normalization_mode], enabled=[True]):
        auto_x_series, auto_y_series = _prepared_density_auto_limit_series(
            x_series=[x_values],
            y_series=[density_values],
            labels=[resolved_line_label or profile.species or "Series"],
            series_enabled=[single_series.line_visible],
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            series_normalization_modes=[single_series.normalization_mode or "none"],
            series_normalization_values=[single_series.normalization_value],
            series_normalization_x_refs=[single_series.normalization_x_ref],
        )
        resolved_x_axis_scale, resolved_x_axis_offset = _coerce_x_axis_linear_transform(
            x_axis_scale,
            x_axis_offset,
        )
        auto_x_series = [
            _display_x_values(
                values,
                auto_y_series[index],
                scale=resolved_x_axis_scale,
                offset=resolved_x_axis_offset,
            )
            for index, values in enumerate(auto_x_series)
        ]
        auto_x_lim, auto_y_lim = _density_auto_plot_limits(
            auto_x_series,
            auto_y_series,
            x_scale=x_scale,
            y_scale=y_scale,
        )
        resolved_x_lim = _merge_plot_limits(x_lim, auto_x_lim)
        resolved_y_lim = _merge_plot_limits(y_lim, auto_y_lim)
    stats_key = "number_density" if quantity.strip().lower() == "number" else "density"

    return plot_line_series(
        x_values,
        density_values,
        title=title or f"{profile.species} density profile",
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, f"{y_label_prefix} ({units})"),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_id=series_id,
        line_label=resolved_line_label,
        line_color=single_series.line_color,
        line_width_override=single_series.line_width_override,
        line_marker=single_series.line_marker,
        line_visible=single_series.line_visible,
        show_in_legend=True if not series_show_in_legend else bool(series_show_in_legend[0]),
        fit_config=None if not series_fit_configs else series_fit_configs[0],
        cumulative_config=cumulative_config,
        series_statistics=None
        if profile.series_statistics is None
        else profile.series_statistics.get(stats_key),
        error_config=error_config,
        normalization_mode=single_series.normalization_mode,
        normalization_value=single_series.normalization_value,
        normalization_x_ref=single_series.normalization_x_ref,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        style=style,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=resolved_x_lim,
        y_lim=resolved_y_lim,
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
        analysis_name="density",
        annotations=annotations,
        integration_config=integration_config,
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


def plot_density_profiles(
    profiles: list[DensityProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_mode: str = "distance",
    quantity: str = "mass",
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
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
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
) -> Path | None:
    """Plot one or more density profiles."""
    if not profiles:
        raise ValueError("At least one density profile is required.")
    default_labels = [profile.species for profile in profiles]
    labels = resolve_series_labels(
        default_labels,
        series_labels,
        series_kind="density",
    )

    if not use_multi_series_plot(
        profile_count=len(profiles),
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
    ):
        return plot_density_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            x_mode=x_mode,
            quantity=quantity,
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
            annotations=annotations,
            integration_config=integration_config,
            line_label=labels[0] if labels else None,
            line_colors=line_colors,
            error_config=(None if not series_error_configs else series_error_configs[0]),
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            cumulative_config=None
            if not series_cumulative_configs
            else series_cumulative_configs[0],
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
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

    if x_mode in {"x", "y", "z"}:
        compat = [
            i for i, p in enumerate(profiles)
            if _density_profile_visible_for_x_mode(p, x_mode=x_mode)
        ]
        if not compat:
            raise ValueError(
                f"No density profiles match the selected axis '{x_mode.upper()}'."
            )

        def _pick(lst: list[Any] | None) -> list[Any] | None:
            return None if lst is None else [lst[i] for i in compat]

        profiles = _pick(profiles)  # type: ignore[assignment]
        labels = _pick(labels)  # type: ignore[assignment]
        series_ids = _pick(series_ids)
        line_colors = _pick(line_colors)
        series_error_configs = _pick(series_error_configs)
        series_enabled = _pick(series_enabled)
        series_show_in_legend = _pick(series_show_in_legend)
        series_line_widths = _pick(series_line_widths)
        series_markers = _pick(series_markers)
        series_fit_configs = _pick(series_fit_configs)
        series_cumulative_configs = _pick(series_cumulative_configs)
        series_normalization_modes = _pick(series_normalization_modes)
        series_normalization_values = _pick(series_normalization_values)
        series_normalization_x_refs = _pick(series_normalization_x_refs)
        series_line_kwargs = _pick(series_line_kwargs)
        if render_series_descriptors is not None:
            render_series_descriptors = _pick(render_series_descriptors)  # type: ignore[assignment]

    first = profiles[0]
    if x_mode == "distance" and any(
        not _profile_has_surface_reference(profile) for profile in profiles
    ):
        LOGGER.warning(
            "At least one profile has no surface reference; combined plot falls back to axis coordinates."
        )
        x_mode = "axis"

    y_resolved = [_density_y_data(profile, quantity=quantity) for profile in profiles]
    y_units = y_resolved[0][1]
    y_label_prefix = y_resolved[0][2]
    if any(units != y_units for _, units, _ in y_resolved[1:]):
        unified = _unify_number_density_units([units for _, units, _ in y_resolved])
        if unified is not None:
            y_units = unified
        else:
            raise ValueError("All density profiles must use the same units for combined plotting.")
    y_series = [values for values, _, _ in y_resolved]
    x_series = [_density_x_data(profile, x_mode=x_mode)[0] for profile in profiles]
    default_x_label = _density_x_data(first, x_mode=x_mode)[1]
    display_units = _format_plot_density_units(y_units)
    resolved_x_lim = x_lim
    resolved_y_lim = y_lim
    gui_render_model_active = bool(render_series_descriptors) or bool(series_overrides_by_id)
    if not gui_render_model_active and not _has_active_normalization(
        series_normalization_modes,
        enabled=series_enabled,
    ):
        auto_x_series, auto_y_series = _prepared_density_auto_limit_series(
            x_series=x_series,
            y_series=y_series,
            labels=labels,
            series_enabled=series_enabled,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
        )
        resolved_x_axis_scale, resolved_x_axis_offset = _coerce_x_axis_linear_transform(
            x_axis_scale,
            x_axis_offset,
        )
        auto_x_series = [
            _display_x_values(
                values,
                auto_y_series[index],
                scale=resolved_x_axis_scale,
                offset=resolved_x_axis_offset,
            )
            for index, values in enumerate(auto_x_series)
        ]
        auto_x_lim, auto_y_lim = _density_auto_plot_limits(
            auto_x_series,
            auto_y_series,
            x_scale=x_scale,
            y_scale=y_scale,
        )
        resolved_x_lim = _merge_plot_limits(x_lim, auto_x_lim)
        resolved_y_lim = _merge_plot_limits(y_lim, auto_y_lim)
    stats_key = "number_density" if quantity.strip().lower() == "number" else "density"

    return plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or "Element-resolved density profile",
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, f"{y_label_prefix} ({display_units})"),
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
        series_statistics_data=[
            None if profile.series_statistics is None else profile.series_statistics.get(stats_key)
            for profile in profiles
        ],
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=resolved_x_lim,
        y_lim=resolved_y_lim,
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
        analysis_name="density",
        annotations=annotations,
        integration_config=integration_config,
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
