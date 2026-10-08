"""Density analysis routines."""

from __future__ import annotations

from collections.abc import Sequence

from ..cell_geometry import (
    bin_volumes,
    cartesian_bounds,
    cell_matrix,
    frame_origin,
    intersection_volume,
    is_axis_prismatic,
    needs_general_geometry,
    wrapped_frame_positions,
    wrap_trajectory_frames,
)

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
import logging
from pathlib import Path
import re
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.density_mapping import resolve_density_plot_mapping
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
    build_series_statistics_from_moments,
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
    plot_heatmap_series,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
    resolve_single_series_options,
)
from ..progress import ProgressBar
from ..utils import axis_to_index, ensure_positive
from .common import (
    MOLECULE_SPECIES_LABELS,
    available_element_species,
    available_raw_species,
    frame_has_usable_cell as _common_frame_has_usable_cell,
    grouped_raw_species_for_split_elements,
    is_molecule_species_label,
    molecule_display_label,
    normalize_species_query as _normalize_species_query,
    raw_species_labels,
    species_selector_raw_label,
    read_profile_payloads,
    read_profile_payloads_by_index,
    use_multi_series_plot,
    write_profile_collection,
)
from .surface import (
    SurfaceEstimate,
    SurfaceEstimatorOptions,
    SurfaceReference,
    _cell_histogram_bounds,
    _log_framewise_surface_alignment,
    _select_surface_estimate,
    _shift_axis_values_by_surface_per_frame,
    _surface_estimate_datasets,
    _surface_estimate_from_payload,
    _surface_estimate_supports_distance_mode,
    _single_interface_wrap_threshold,
    _surface_metadata_payload,
    _surface_metadata_view,
    _surface_reference_datasets,
    _surface_reference_from_payload,
    _surface_reference_metadata,
    estimate_surface_layout_reference,
    surface_distance_components,
    wrapped_surface_positions,
    estimate_surface_position as estimate_surface_position,
    estimate_surface_reference as estimate_surface_reference,
)
from .water import (
    OHTopologyCache,
    molecule_positions_with_masses as _molecule_positions_with_masses,
    water_molecule_triplets as _water_molecule_triplets,
    water_triplet_axis_values_with_masses as _water_triplet_axis_values_with_masses,
    water_axis_values_per_frame as _water_axis_values_per_frame_impl,
)

LOGGER = logging.getLogger(__name__)
H2O_VALIDATION_STRIDE = 100
H2O_OH_CUTOFF_A = 1.27
DEFAULT_MIN_MOLECULE_FRAMES = 5
AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24
ANGSTROM3_TO_NM3 = 1.0e-3
PLOT_AUTO_LIMIT_MARGIN_FRACTION = 0.05
H2O_MASS_G = float(
    (atomic_masses[atomic_numbers["H"]] * 2.0 + atomic_masses[atomic_numbers["O"]]) * AMU_TO_G
)
_DENSITY_HEATMAP_MAX_CELLS = 2_000_000
_DENSITY_HEATMAP_ARRAY_COUNT_ESTIMATE = 4
_DENSITY_STRICT_VALIDATION = False
DEFAULT_DENSITY_GRID_MAX_NONZERO_BINS = 20_000_000
_DENSITY_GRID_AXES = ("x", "y", "z", "distance")
_DENSITY_GRID_STORAGE_DTYPE = np.float32


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
    surface_reference: SurfaceReference | None = None
    interface_view: str = "single"
    top_density: np.ndarray | None = None
    bottom_density: np.ndarray | None = None
    top_number_density: np.ndarray | None = None
    bottom_number_density: np.ndarray | None = None
    top_counts_per_frame: np.ndarray | None = None
    bottom_counts_per_frame: np.ndarray | None = None
    series_statistics: dict[str, SeriesStatistics] | None = None
    visible_x_min_A: float | None = None
    visible_x_max_A: float | None = None
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None


@dataclass(frozen=True)
class DensityHeatmapProfile:
    """Container for a 2D planar density field."""

    plane: str
    plane_axes: tuple[str, str]
    species: str
    x_bin_edges: np.ndarray
    y_bin_edges: np.ndarray
    x_bin_centers: np.ndarray
    y_bin_centers: np.ndarray
    density: np.ndarray
    units: str
    n_frames: int
    number_density: np.ndarray | None = None
    number_density_units: str | None = None
    interface_view: str = "single"
    top_density: np.ndarray | None = None
    bottom_density: np.ndarray | None = None
    top_number_density: np.ndarray | None = None
    bottom_number_density: np.ndarray | None = None
    visible_x_min_A: float | None = None
    visible_x_max_A: float | None = None
    visible_y_min_A: float | None = None
    visible_y_max_A: float | None = None
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None


@dataclass(frozen=True)
class DensityGridProfile:
    """Sparse X/Y/Z/distance density grid used for display-time slicing."""

    species: str
    x_bin_edges: np.ndarray
    y_bin_edges: np.ndarray
    z_bin_edges: np.ndarray
    distance_bin_edges: np.ndarray
    flat_indices: np.ndarray
    mass_sum_g: np.ndarray
    entity_sum: np.ndarray
    n_frames: int
    units: str = "g"
    number_density_units: str | None = None
    grid_bin_width_A: float | None = None
    grid_max_nonzero_bins: int | None = None
    surface_axis: str = "z"
    distance_coordinate_mode: str = "distance"
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_estimate: SurfaceEstimate | None = None
    surface_reference: SurfaceReference | None = None
    top_mass_sum_g: np.ndarray | None = None
    bottom_mass_sum_g: np.ndarray | None = None
    top_entity_sum: np.ndarray | None = None
    bottom_entity_sum: np.ndarray | None = None
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None

    geometry_groups: list[dict[str, Any]] | None = None

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        return (
            int(np.asarray(self.x_bin_edges).size - 1),
            int(np.asarray(self.y_bin_edges).size - 1),
            int(np.asarray(self.z_bin_edges).size - 1),
            int(np.asarray(self.distance_bin_edges).size - 1),
        )


@dataclass
class _DensityStatisticsMoments:
    """Compact per-bin density statistics accumulated during the main binning pass."""

    point_count: np.ndarray
    sample_n: np.ndarray
    sample_sum: np.ndarray
    sample_sumsq: np.ndarray
    block_sum: np.ndarray | None = None
    block_n: np.ndarray | None = None


@dataclass(frozen=True)
class _DensityLineNormalizationCache:
    """Shared 1D normalization data reused across many density outputs."""

    use_volumetric_density: bool
    slice_volumes: np.ndarray | None
    framewise_normalization: bool
    frames: Sequence[Atoms] | None = None
    axis_index: int = 2


@dataclass(frozen=True)
class _DensityHeatmapNormalizationCache:
    """Shared 2D normalization data reused across many density heatmaps."""

    use_volumetric_density: bool
    slice_volumes: np.ndarray | None
    framewise_normalization: bool
    frames: Sequence[Atoms] | None = None
    plane_axis_indices: tuple[int, int] | None = None


def _cartesian_column_volumes(
    frame: Atoms,
    x_bin_edges: np.ndarray,
    y_bin_edges: np.ndarray,
    plane_axis_indices: tuple[int, int],
) -> np.ndarray:
    """Return physical cell volume in each Cartesian column."""
    result = np.empty((x_bin_edges.size - 1, y_bin_edges.size - 1), dtype=float)
    bounds = np.array([[-np.inf, np.inf]] * 3, dtype=float)
    x_axis, y_axis = plane_axis_indices
    for ix, (x_lo, x_hi) in enumerate(zip(x_bin_edges[:-1], x_bin_edges[1:])):
        bounds[x_axis] = (x_lo, x_hi)
        for iy, (y_lo, y_hi) in enumerate(zip(y_bin_edges[:-1], y_bin_edges[1:])):
            bounds[y_axis] = (y_lo, y_hi)
            result[ix, iy] = intersection_volume(
                frame.cell,
                bounds,
                origin=frame_origin(frame),
            )
    return result


@dataclass(frozen=True)
class _DensityValidationPlan:
    """Representative validation sampling for the density hot path."""

    strict: bool
    frame_indices: frozenset[int]


@dataclass(frozen=True)
class _DensityTargetSpec:
    """One logical density selection target processed by the shared engine."""

    species_label: str
    count_label: str
    selection_kind: str
    apply_min_molecule_frames: bool = False


@dataclass
class _DensityObservedBounds:
    """Observed coordinate extrema gathered during the density prepass."""

    axis_lower: dict[str, float]
    axis_upper: dict[str, float]
    selected_count: int = 0
    active_frame_count: int = 0
    max_consecutive_active_frames: int = 0
    _current_consecutive_active_frames: int = 0
    _last_active_frame_index: int | None = None

    def mark_frame_active(self, frame_index: int | None = None) -> None:
        self.active_frame_count += 1
        if frame_index is None:
            self._current_consecutive_active_frames += 1
        elif (
            self._last_active_frame_index is None
            or frame_index == self._last_active_frame_index + 1
        ):
            self._current_consecutive_active_frames += 1
        else:
            self._current_consecutive_active_frames = 1
        self._last_active_frame_index = frame_index
        self.max_consecutive_active_frames = max(
            self.max_consecutive_active_frames,
            self._current_consecutive_active_frames,
        )

    def update_axis(self, axis_id: str, values: np.ndarray) -> None:
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        current_lower = self.axis_lower.get(axis_id)
        current_upper = self.axis_upper.get(axis_id)
        self.axis_lower[axis_id] = lower if current_lower is None else min(current_lower, lower)
        self.axis_upper[axis_id] = upper if current_upper is None else max(current_upper, upper)
        self.selected_count += int(finite.size)

    def has_axis_data(self, axis_id: str) -> bool:
        return axis_id in self.axis_lower and axis_id in self.axis_upper


def _density_species_display_label(species_label: str) -> str:
    if is_molecule_species_label(species_label):
        return molecule_display_label(species_label)
    if str(species_label).lower().startswith("species:"):
        return f"{species_selector_raw_label(species_label)} (species)"
    return str(species_label)


def _density_selection_kind_for_label(species_label: str) -> str:
    if is_molecule_species_label(species_label):
        return "molecule"
    if str(species_label).lower().startswith("species:"):
        return "species"
    return "element"


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


def _density_selection_summary(targets: list[_DensityTargetSpec]) -> str:
    elements = [target.species_label for target in targets if target.selection_kind == "element"]
    raw_species = [
        _density_species_display_label(target.species_label)
        for target in targets
        if target.selection_kind == "species"
    ]
    molecules = [
        _density_species_display_label(target.species_label)
        for target in targets
        if target.selection_kind == "molecule"
    ]
    parts: list[str] = []
    if elements:
        parts.append("elements=" + ",".join(elements))
    if raw_species:
        parts.append("species=" + ",".join(raw_species))
    if molecules:
        parts.append("molecules=" + ",".join(molecules))
    return "; ".join(parts) if parts else "none"


def _molecule_target_is_active(
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
    threshold_count = (
        active_frame_count if consecutive_frame_count is None else consecutive_frame_count
    )
    return int(threshold_count) >= int(min_molecule_frames)


def _line_visible_range_from_density(
    *,
    bin_edges: np.ndarray,
    values: np.ndarray,
) -> tuple[float | None, float | None]:
    density = np.asarray(values, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    if density.size == 0 or edges.size != density.size + 1:
        return None, None
    active = np.isfinite(density) & (density != 0.0)
    if not np.any(active):
        return None, None
    indices = np.flatnonzero(active)
    lower = float(edges[int(indices[0])])
    upper = float(edges[int(indices[-1]) + 1])
    return lower, upper


def _heatmap_visible_ranges_from_density(
    *,
    x_bin_edges: np.ndarray,
    y_bin_edges: np.ndarray,
    values: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    density = np.asarray(values, dtype=float)
    x_edges = np.asarray(x_bin_edges, dtype=float)
    y_edges = np.asarray(y_bin_edges, dtype=float)
    if (
        density.ndim != 2
        or x_edges.size != density.shape[0] + 1
        or y_edges.size != density.shape[1] + 1
    ):
        return None, None, None, None
    active = np.isfinite(density) & (density != 0.0)
    if not np.any(active):
        return None, None, None, None
    x_indices, y_indices = np.nonzero(active)
    return (
        float(x_edges[int(np.min(x_indices))]),
        float(x_edges[int(np.max(x_indices)) + 1]),
        float(y_edges[int(np.min(y_indices))]),
        float(y_edges[int(np.max(y_indices)) + 1]),
    )


@dataclass
class _DensityLineAccumulator:
    """Incrementally accumulate one 1D density output over shared frame data."""

    axis: str
    species_label: str
    count_label: str
    bin_edges: np.ndarray
    bin_width: float
    coordinate_mode: str
    normalization_cache: _DensityLineNormalizationCache
    density_moments: _DensityStatisticsMoments
    number_density_moments: _DensityStatisticsMoments
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_estimate: SurfaceEstimate | None = None
    surface_reference: SurfaceReference | None = None
    interface_view: str = "single"
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None
    mass_histogram_sum: np.ndarray | None = None
    entity_histogram_sum: np.ndarray | None = None
    framewise_mass_density_sum: np.ndarray | None = None
    framewise_entity_density_sum: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_bins = int(np.asarray(self.bin_edges, dtype=float).size - 1)
        if self.normalization_cache.frames is not None and self.surface_reference is None:
            frames = self.normalization_cache.frames
            volumes = np.asarray(
                [
                    bin_volumes(
                        frame.cell,
                        self.bin_edges,
                        self.normalization_cache.axis_index,
                        origin=frame_origin(frame),
                        shift=(
                            float(self.surface_estimate.frame_values[i])
                            if self.coordinate_mode == "distance"
                            and self.surface_estimate is not None
                            else 0.0
                        ),
                    )
                    for i, frame in enumerate(frames)
                ]
            )
            self.normalization_cache = replace(
                self.normalization_cache, slice_volumes=volumes, framewise_normalization=True
            )

        if self.mass_histogram_sum is None:
            self.mass_histogram_sum = np.zeros(n_bins, dtype=float)
        if self.entity_histogram_sum is None:
            self.entity_histogram_sum = np.zeros(n_bins, dtype=float)
        if (
            self.normalization_cache.framewise_normalization
            and self.framewise_mass_density_sum is None
        ):
            self.framewise_mass_density_sum = np.zeros(n_bins, dtype=float)
        if (
            self.normalization_cache.framewise_normalization
            and self.framewise_entity_density_sum is None
        ):
            self.framewise_entity_density_sum = np.zeros(n_bins, dtype=float)

    @property
    def n_bins(self) -> int:
        return int(self.bin_edges.size - 1)

    @property
    def bin_start(self) -> float:
        return float(self.bin_edges[0])

    @property
    def bin_end(self) -> float:
        return float(self.bin_edges[-1])

    def update(
        self,
        *,
        axis_values: np.ndarray,
        masses: np.ndarray,
        frame_index: int,
        validation_plan: _DensityValidationPlan,
        block_index_by_frame: np.ndarray | None,
        entity_weights: np.ndarray | None = None,
    ) -> None:
        per_frame_mass_histogram, per_frame_entity_histogram = _bincount_1d_histograms(
            values=axis_values,
            masses=masses,
            bin_start=self.bin_start,
            bin_width=self.bin_width,
            n_bins=self.n_bins,
            bin_end=self.bin_end,
            entity_weights=entity_weights,
        )
        if validation_plan.strict or frame_index in validation_plan.frame_indices:
            _validate_binned_frame_conservation(
                frame_index=frame_index,
                mass_histogram=per_frame_mass_histogram,
                entity_histogram=per_frame_entity_histogram,
                masses=masses,
                axis_values=axis_values,
                entity_weights=entity_weights,
            )
        assert self.mass_histogram_sum is not None
        assert self.entity_histogram_sum is not None
        self.mass_histogram_sum += per_frame_mass_histogram
        self.entity_histogram_sum += per_frame_entity_histogram
        if (
            self.normalization_cache.use_volumetric_density
            and self.normalization_cache.slice_volumes is not None
        ):
            frame_volume = np.where(
                self.normalization_cache.slice_volumes[frame_index] > 0,
                self.normalization_cache.slice_volumes[frame_index],
                np.inf,
            )
            frame_density = (per_frame_mass_histogram / frame_volume) / ANGSTROM3_TO_CM3
            frame_number_density = (per_frame_entity_histogram / frame_volume) / ANGSTROM3_TO_NM3
            if (
                self.framewise_mass_density_sum is not None
                and self.framewise_entity_density_sum is not None
            ):
                self.framewise_mass_density_sum += per_frame_mass_histogram / frame_volume
                self.framewise_entity_density_sum += per_frame_entity_histogram / frame_volume
        else:
            frame_density = per_frame_mass_histogram / self.bin_width
            frame_number_density = per_frame_entity_histogram / self.bin_width
        _update_density_statistics_moments(
            self.density_moments,
            histogram=per_frame_entity_histogram,
            density_values=frame_density,
            frame_index=frame_index,
            block_index_by_frame=block_index_by_frame,
        )
        _update_density_statistics_moments(
            self.number_density_moments,
            histogram=per_frame_entity_histogram,
            density_values=frame_number_density,
            frame_index=frame_index,
            block_index_by_frame=block_index_by_frame,
        )

    def finalize(self, *, n_frames: int) -> DensityProfile:
        assert self.mass_histogram_sum is not None
        assert self.entity_histogram_sum is not None
        if (
            self.normalization_cache.use_volumetric_density
            and self.normalization_cache.slice_volumes is not None
        ):
            if self.framewise_mass_density_sum is None or self.framewise_entity_density_sum is None:
                reference_slice_volume = self.normalization_cache.slice_volumes[0]
                density = (self.mass_histogram_sum / reference_slice_volume) / n_frames
                number_density = (self.entity_histogram_sum / reference_slice_volume) / n_frames
            else:
                coverage = (
                    np.sum(self.normalization_cache.slice_volumes > 0, axis=0)
                    if self.normalization_cache.frames is not None
                    else n_frames
                )
                denominator = np.where(np.asarray(coverage) > 0, coverage, np.nan)
                density = self.framewise_mass_density_sum / denominator
                number_density = self.framewise_entity_density_sum / denominator
            density = density / ANGSTROM3_TO_CM3
            number_density = number_density / ANGSTROM3_TO_NM3
            units = "g/cm^3"
        else:
            density = (self.mass_histogram_sum / n_frames) / self.bin_width
            number_density = (self.entity_histogram_sum / n_frames) / self.bin_width
            units = "g/Angstrom"
        avg_binned_mass_per_bin = self.mass_histogram_sum / n_frames
        avg_entities_per_bin = self.entity_histogram_sum / n_frames
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        number_density_units = _entity_density_units_for_species(
            self.species_label,
            volumetric=self.normalization_cache.use_volumetric_density,
        )
        visible_x_min_A, visible_x_max_A = _line_visible_range_from_density(
            bin_edges=self.bin_edges,
            values=density,
        )
        return DensityProfile(
            axis=self.axis.lower(),
            species=self.species_label,
            bin_edges=self.bin_edges,
            bin_centers=bin_centers,
            counts_per_frame=avg_binned_mass_per_bin,
            density=density,
            units=units,
            n_frames=n_frames,
            entities_per_frame=avg_entities_per_bin,
            number_density=number_density,
            number_density_units=number_density_units,
            coordinate_mode=self.coordinate_mode,
            surface_position=self.surface_position,
            surface_position_std=self.surface_position_std,
            surface_estimate=self.surface_estimate,
            surface_reference=self.surface_reference,
            interface_view=self.interface_view,
            series_statistics={
                "density": _finalize_density_statistics_moments(self.density_moments),
                "number_density": _finalize_density_statistics_moments(self.number_density_moments),
            },
            visible_x_min_A=visible_x_min_A,
            visible_x_max_A=visible_x_max_A,
            oh_cutoff_A=self.oh_cutoff_A,
            min_molecule_frames=self.min_molecule_frames,
        )


def _fold_symmetric_density_profiles(
    top: DensityProfile,
    bottom: DensityProfile,
    *,
    surface_reference: SurfaceReference,
    surface_position: float | None,
    surface_position_std: float | None,
) -> DensityProfile:
    """Combine independently normalized interface profiles without doubling density."""

    if not np.array_equal(top.bin_edges, bottom.bin_edges):
        raise ValueError("Top and bottom symmetric density grids do not match.")
    density = 0.5 * (np.asarray(top.density) + np.asarray(bottom.density))
    number_density = (
        None
        if top.number_density is None or bottom.number_density is None
        else 0.5 * (np.asarray(top.number_density) + np.asarray(bottom.number_density))
    )
    entities = (
        None
        if top.entities_per_frame is None or bottom.entities_per_frame is None
        else 0.5 * (np.asarray(top.entities_per_frame) + np.asarray(bottom.entities_per_frame))
    )
    visible_x_min_A, visible_x_max_A = _line_visible_range_from_density(
        bin_edges=top.bin_edges,
        values=density,
    )
    return replace(
        top,
        counts_per_frame=0.5
        * (np.asarray(top.counts_per_frame) + np.asarray(bottom.counts_per_frame)),
        density=density,
        entities_per_frame=entities,
        number_density=number_density,
        surface_position=surface_position,
        surface_position_std=surface_position_std,
        surface_estimate=surface_reference.primary,
        surface_reference=surface_reference,
        interface_view="folded",
        top_density=np.asarray(top.density),
        bottom_density=np.asarray(bottom.density),
        top_number_density=top.number_density,
        bottom_number_density=bottom.number_density,
        top_counts_per_frame=np.asarray(top.counts_per_frame),
        bottom_counts_per_frame=np.asarray(bottom.counts_per_frame),
        series_statistics=None,
        visible_x_min_A=visible_x_min_A,
        visible_x_max_A=visible_x_max_A,
    )


@dataclass
class _DensityHeatmapAccumulator:
    """Incrementally accumulate one 2D density heatmap over shared frame data."""

    plane_axes: tuple[str, str]
    orthogonal_axis: str
    species_label: str
    x_bin_edges: np.ndarray
    y_bin_edges: np.ndarray
    bin_width: float
    normalization_cache: _DensityHeatmapNormalizationCache
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None
    mass_histogram_sum: np.ndarray | None = None
    entity_histogram_sum: np.ndarray | None = None
    framewise_mass_density_sum: np.ndarray | None = None
    framewise_entity_density_sum: np.ndarray | None = None
    volume_coverage_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = (int(self.x_bin_edges.size - 1), int(self.y_bin_edges.size - 1))
        if self.mass_histogram_sum is None:
            self.mass_histogram_sum = np.zeros(shape, dtype=float)
        if self.entity_histogram_sum is None:
            self.entity_histogram_sum = np.zeros(shape, dtype=float)
        if (
            self.normalization_cache.framewise_normalization
            or self.normalization_cache.frames is not None
        ) and self.framewise_mass_density_sum is None:
            self.framewise_mass_density_sum = np.zeros(shape, dtype=float)
        if (
            self.normalization_cache.framewise_normalization
            or self.normalization_cache.frames is not None
        ) and self.framewise_entity_density_sum is None:
            self.framewise_entity_density_sum = np.zeros(shape, dtype=float)
        if self.normalization_cache.frames is not None and self.volume_coverage_count is None:
            self.volume_coverage_count = np.zeros(shape, dtype=np.int64)

    @property
    def x_bin_count(self) -> int:
        return int(self.x_bin_edges.size - 1)

    @property
    def y_bin_count(self) -> int:
        return int(self.y_bin_edges.size - 1)

    def update(
        self,
        *,
        x_values: np.ndarray,
        y_values: np.ndarray,
        masses: np.ndarray,
        frame_index: int,
        validation_plan: _DensityValidationPlan,
        frame: Atoms | None = None,
    ) -> None:
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        mass_array = np.asarray(masses, dtype=float)
        per_frame_mass_histogram, per_frame_entity_histogram = _bincount_2d_histograms(
            x_values=x_array,
            y_values=y_array,
            masses=mass_array,
            x_bin_start=float(self.x_bin_edges[0]),
            y_bin_start=float(self.y_bin_edges[0]),
            bin_width=self.bin_width,
            x_bin_count=self.x_bin_count,
            y_bin_count=self.y_bin_count,
            x_bin_end=float(self.x_bin_edges[-1]),
            y_bin_end=float(self.y_bin_edges[-1]),
        )
        if validation_plan.strict or frame_index in validation_plan.frame_indices:
            _validate_binned_heatmap_frame_conservation(
                frame_index=frame_index,
                mass_histogram=per_frame_mass_histogram,
                entity_histogram=per_frame_entity_histogram,
                masses=mass_array,
                xy_values=(
                    np.column_stack((x_array, y_array))
                    if x_array.size > 0
                    else np.empty((0, 2), dtype=float)
                ),
            )
        assert self.mass_histogram_sum is not None
        assert self.entity_histogram_sum is not None
        self.mass_histogram_sum += per_frame_mass_histogram
        self.entity_histogram_sum += per_frame_entity_histogram
        if (
            self.normalization_cache.frames is not None
            and self.normalization_cache.plane_axis_indices is not None
            and self.framewise_mass_density_sum is not None
            and self.framewise_entity_density_sum is not None
            and self.volume_coverage_count is not None
        ):
            volumes = _cartesian_column_volumes(
                (frame if frame is not None else self.normalization_cache.frames[frame_index]),
                self.x_bin_edges,
                self.y_bin_edges,
                self.normalization_cache.plane_axis_indices,
            )
            covered = volumes > 0
            self.framewise_mass_density_sum[covered] += (
                per_frame_mass_histogram[covered] / volumes[covered]
            )
            self.framewise_entity_density_sum[covered] += (
                per_frame_entity_histogram[covered] / volumes[covered]
            )
            self.volume_coverage_count[covered] += 1
        elif (
            self.normalization_cache.use_volumetric_density
            and self.normalization_cache.slice_volumes is not None
            and self.framewise_mass_density_sum is not None
            and self.framewise_entity_density_sum is not None
        ):
            frame_volume = float(self.normalization_cache.slice_volumes[frame_index])
            self.framewise_mass_density_sum += per_frame_mass_histogram / frame_volume
            self.framewise_entity_density_sum += per_frame_entity_histogram / frame_volume

    def finalize(self, *, n_frames: int) -> DensityHeatmapProfile:
        assert self.mass_histogram_sum is not None
        assert self.entity_histogram_sum is not None
        if (
            self.normalization_cache.use_volumetric_density
            and self.normalization_cache.slice_volumes is not None
        ):
            if (
                self.volume_coverage_count is not None
                and self.framewise_mass_density_sum is not None
                and self.framewise_entity_density_sum is not None
            ):
                density = np.divide(
                    self.framewise_mass_density_sum,
                    self.volume_coverage_count,
                    out=np.full_like(self.framewise_mass_density_sum, np.nan),
                    where=self.volume_coverage_count > 0,
                )
                number_density = np.divide(
                    self.framewise_entity_density_sum,
                    self.volume_coverage_count,
                    out=np.full_like(self.framewise_entity_density_sum, np.nan),
                    where=self.volume_coverage_count > 0,
                )
            elif (
                self.framewise_mass_density_sum is None or self.framewise_entity_density_sum is None
            ):
                reference_slice_volume = float(self.normalization_cache.slice_volumes[0])
                density = (self.mass_histogram_sum / reference_slice_volume) / n_frames
                number_density = (self.entity_histogram_sum / reference_slice_volume) / n_frames
            else:
                density = self.framewise_mass_density_sum / n_frames
                number_density = self.framewise_entity_density_sum / n_frames
            density = density / ANGSTROM3_TO_CM3
            number_density = number_density / ANGSTROM3_TO_NM3
            units = "g/cm^3"
            number_density_units = _entity_density_units_for_species(
                self.species_label, volumetric=True
            )
        else:
            density = (self.mass_histogram_sum / n_frames) / (self.bin_width * self.bin_width)
            number_density = (self.entity_histogram_sum / n_frames) / (
                self.bin_width * self.bin_width
            )
            units = "g/Angstrom^2"
            number_density_units = _entity_areal_density_units_for_species(self.species_label)
        x_bin_centers = 0.5 * (self.x_bin_edges[:-1] + self.x_bin_edges[1:])
        y_bin_centers = 0.5 * (self.y_bin_edges[:-1] + self.y_bin_edges[1:])
        visible_x_min_A, visible_x_max_A, visible_y_min_A, visible_y_max_A = (
            _heatmap_visible_ranges_from_density(
                x_bin_edges=self.x_bin_edges,
                y_bin_edges=self.y_bin_edges,
                values=density,
            )
        )
        return DensityHeatmapProfile(
            plane=f"{self.plane_axes[0]}{self.plane_axes[1]}",
            plane_axes=self.plane_axes,
            species=self.species_label,
            x_bin_edges=self.x_bin_edges,
            y_bin_edges=self.y_bin_edges,
            x_bin_centers=x_bin_centers,
            y_bin_centers=y_bin_centers,
            density=np.asarray(density, dtype=float),
            units=units,
            n_frames=n_frames,
            number_density=np.asarray(number_density, dtype=float),
            number_density_units=number_density_units,
            visible_x_min_A=visible_x_min_A,
            visible_x_max_A=visible_x_max_A,
            visible_y_min_A=visible_y_min_A,
            visible_y_max_A=visible_y_max_A,
            oh_cutoff_A=self.oh_cutoff_A,
            min_molecule_frames=self.min_molecule_frames,
        )


def _grid_visible_axis_range_from_flat_indices(
    *,
    flat_indices: np.ndarray,
    grid_shape: tuple[int, int, int, int],
    axis_index: int,
    bin_edges: np.ndarray,
) -> tuple[float | None, float | None]:
    flat = np.asarray(flat_indices, dtype=np.uint64)
    edges = np.asarray(bin_edges, dtype=float)
    if flat.size == 0 or edges.size < 2:
        return None, None
    nx, ny, nz, nd = (int(value) for value in grid_shape)
    if min(nx, ny, nz, nd) <= 0:
        return None, None
    indices = _density_grid_axis_indices(flat, grid_shape=grid_shape, axis_index=axis_index)
    if indices.size == 0:
        return None, None
    lower_index = int(np.min(indices))
    upper_index = int(np.max(indices))
    if lower_index < 0 or upper_index + 1 >= edges.size:
        return None, None
    return float(edges[lower_index]), float(edges[upper_index + 1])


@dataclass
class _DensitySparseGridAccumulator:
    """Sparse X/Y/Z/distance density-grid accumulator with no dense 4D arrays."""

    species_label: str
    count_label: str
    x_bin_edges: np.ndarray
    y_bin_edges: np.ndarray
    z_bin_edges: np.ndarray
    distance_bin_edges: np.ndarray
    grid_bin_width: float
    max_nonzero_bins: int
    surface_axis: str
    distance_coordinate_mode: str
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_estimate: SurfaceEstimate | None = None
    surface_reference: SurfaceReference | None = None
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None
    _flat_chunks: list[np.ndarray] | None = None
    _mass_chunks: list[np.ndarray] | None = None
    _entity_chunks: list[np.ndarray] | None = None
    _top_counts: dict[int, tuple[float, float]] | None = None
    _bottom_counts: dict[int, tuple[float, float]] | None = None

    geometry_frames: Sequence[Atoms] | None = None
    geometry_groups: dict | None = None
    frame_group_keys: list | None = None

    def __post_init__(self) -> None:
        if self.geometry_frames is not None:
            self.geometry_groups = {}
            self.frame_group_keys = []
            for i, frame in enumerate(self.geometry_frames):
                canonical_reference = (
                    self.surface_reference
                    if self.surface_reference is not None and self.surface_reference.uses_distance
                    else None
                )
                shift = (
                    float(self.surface_estimate.frame_values[i])
                    if self.distance_coordinate_mode == "distance"
                    and self.surface_estimate is not None
                    else 0.0
                )
                top_shift = (
                    float(canonical_reference.top.frame_values[i])
                    if self.distance_coordinate_mode == "distance"
                    and canonical_reference is not None
                    and canonical_reference.top is not None
                    else shift
                )
                bottom_shift = (
                    float(canonical_reference.bottom.frame_values[i])
                    if self.distance_coordinate_mode == "distance"
                    and canonical_reference is not None
                    and canonical_reference.bottom is not None
                    else shift
                )
                canonical_cut = (
                    float(canonical_reference.axis_cut_per_frame_A[i])
                    if canonical_reference is not None
                    else None
                )
                canonical_lower = (
                    float(canonical_reference.axis_lower_per_frame_A[i])
                    if canonical_reference is not None
                    else None
                )
                canonical_period = (
                    float(canonical_reference.axis_period_per_frame_A[i])
                    if canonical_reference is not None
                    else None
                )
                canonical_wrap_threshold = (
                    _single_interface_wrap_threshold(canonical_reference, i)
                    if canonical_reference is not None
                    else None
                )
                key = (
                    *np.asarray(frame.cell).ravel(),
                    *frame_origin(frame),
                    shift,
                    top_shift,
                    bottom_shift,
                    canonical_cut,
                    canonical_lower,
                    canonical_period,
                    canonical_wrap_threshold,
                )
                self.frame_group_keys.append(key)
                group = self.geometry_groups.setdefault(
                    key,
                    dict(
                        cell=np.asarray(frame.cell),
                        origin=frame_origin(frame),
                        shift=shift,
                        top_shift=top_shift,
                        bottom_shift=bottom_shift,
                        canonical_cut=canonical_cut,
                        canonical_lower=canonical_lower,
                        canonical_period=canonical_period,
                        canonical_wrap_threshold=canonical_wrap_threshold,
                        n_frames=0,
                        counts={},
                        top_counts={},
                        bottom_counts={},
                    ),
                )
                group["n_frames"] += 1
        if self._flat_chunks is None:
            self._flat_chunks = []
        if self._mass_chunks is None:
            self._mass_chunks = []
        if self._entity_chunks is None:
            self._entity_chunks = []
        if self._top_counts is None:
            self._top_counts = {}
        if self._bottom_counts is None:
            self._bottom_counts = {}
        dense_cells = self.dense_cell_count
        if dense_cells > np.iinfo(np.uint64).max:
            raise ValueError(
                "Density sparse grid is too large to index safely: "
                f"species='{self.species_label}', dense cells={dense_cells:,}. "
                "Increase --grid-bin-width or request fewer species."
            )

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        return (
            int(np.asarray(self.x_bin_edges).size - 1),
            int(np.asarray(self.y_bin_edges).size - 1),
            int(np.asarray(self.z_bin_edges).size - 1),
            int(np.asarray(self.distance_bin_edges).size - 1),
        )

    @property
    def dense_cell_count(self) -> int:
        dense_cells = 1
        for value in self.grid_shape:
            dense_cells *= int(value)
        return int(dense_cells)

    def update(
        self,
        *,
        positions: np.ndarray,
        distance_values: np.ndarray,
        masses: np.ndarray,
        frame_index: int,
        validation_plan: _DensityValidationPlan,
        interface_side: np.ndarray | None = None,
    ) -> None:
        positions_array = np.asarray(positions, dtype=float)
        mass_array = np.asarray(masses, dtype=float)
        distance_array = np.asarray(distance_values, dtype=float)
        if positions_array.size == 0:
            return
        if positions_array.ndim != 2 or positions_array.shape[1] != 3:
            raise ValueError("Density grid positions must have shape (n_entities, 3).")
        if (
            mass_array.shape[0] != positions_array.shape[0]
            or distance_array.shape[0] != positions_array.shape[0]
        ):
            raise ValueError(
                "Density grid positions, distances, and masses must have matching entity counts."
            )
        flat_indices, mass_sum, entity_sum = _bincount_sparse_density_grid(
            x_values=positions_array[:, 0],
            y_values=positions_array[:, 1],
            z_values=positions_array[:, 2],
            distance_values=distance_array,
            masses=mass_array,
            x_bin_edges=self.x_bin_edges,
            y_bin_edges=self.y_bin_edges,
            z_bin_edges=self.z_bin_edges,
            distance_bin_edges=self.distance_bin_edges,
            grid_shape=self.grid_shape,
        )
        geometry_group = None
        if self.geometry_groups is not None:
            assert self.frame_group_keys is not None
            geometry_group = self.geometry_groups[self.frame_group_keys[frame_index]]
        if interface_side is not None:
            side_array = np.asarray(interface_side, dtype=np.int8)
            if side_array.shape != (positions_array.shape[0],):
                raise ValueError("Density grid interface-side labels must match entity count.")
            for side_name, side_mask, destination in (
                ("top", side_array >= 0, self._top_counts),
                ("bottom", side_array <= 0, self._bottom_counts),
            ):
                if destination is None or not np.any(side_mask):
                    continue
                side_weights = np.where(side_array[side_mask] == 0, 0.5, 1.0)
                side_flat, side_mass, side_entity = _bincount_sparse_density_grid(
                    x_values=positions_array[side_mask, 0],
                    y_values=positions_array[side_mask, 1],
                    z_values=positions_array[side_mask, 2],
                    distance_values=distance_array[side_mask],
                    masses=mass_array[side_mask] * side_weights,
                    x_bin_edges=self.x_bin_edges,
                    y_bin_edges=self.y_bin_edges,
                    z_bin_edges=self.z_bin_edges,
                    distance_bin_edges=self.distance_bin_edges,
                    grid_shape=self.grid_shape,
                    entity_weights=side_weights,
                )
                for index, mass, count in zip(side_flat, side_mass, side_entity):
                    old_mass, old_count = destination.get(int(index), (0.0, 0.0))
                    destination[int(index)] = (
                        old_mass + float(mass),
                        old_count + float(count),
                    )
                if geometry_group is not None:
                    side_counts = geometry_group[f"{side_name}_counts"]
                    for index, mass, count in zip(side_flat, side_mass, side_entity):
                        old_mass, old_count = side_counts.get(int(index), (0.0, 0.0))
                        side_counts[int(index)] = (
                            old_mass + float(mass),
                            old_count + float(count),
                        )
        if validation_plan.strict or frame_index in validation_plan.frame_indices:
            expected_mass = float(np.sum(mass_array))
            observed_mass = float(np.sum(mass_sum))
            if not np.isclose(observed_mass, expected_mass, rtol=1.0e-12, atol=1.0e-30):
                raise ValueError(
                    "Sparse density grid does not conserve selected mass in frame "
                    f"{frame_index}: grid={observed_mass:.16g}, selected={expected_mass:.16g}."
                )
            expected_entities = int(positions_array.shape[0])
            observed_entities = int(np.rint(np.sum(entity_sum)))
            if observed_entities != expected_entities:
                raise ValueError(
                    "Sparse density grid does not conserve selected entity count in frame "
                    f"{frame_index}: grid={observed_entities}, selected={expected_entities}."
                )
        if geometry_group is not None:
            counts = geometry_group["counts"]
            for index, mass, count in zip(flat_indices, mass_sum, entity_sum):
                old_mass, old_count = counts.get(int(index), (0.0, 0.0))
                counts[int(index)] = (old_mass + float(mass), old_count + float(count))
        if flat_indices.size == 0:
            return
        assert self._flat_chunks is not None
        assert self._mass_chunks is not None
        assert self._entity_chunks is not None
        self._flat_chunks.append(np.asarray(flat_indices, dtype=np.uint64))
        self._mass_chunks.append(np.asarray(mass_sum, dtype=float))
        self._entity_chunks.append(np.asarray(entity_sum, dtype=float))
        if len(self._flat_chunks) >= 64:
            self._merge_chunks()

    def _merge_chunks(self) -> None:
        assert self._flat_chunks is not None
        assert self._mass_chunks is not None
        assert self._entity_chunks is not None
        if not self._flat_chunks:
            return
        flat = np.concatenate(self._flat_chunks).astype(np.uint64, copy=False)
        mass = np.concatenate(self._mass_chunks).astype(float, copy=False)
        entity = np.concatenate(self._entity_chunks).astype(float, copy=False)
        if flat.size == 0:
            self._flat_chunks = []
            self._mass_chunks = []
            self._entity_chunks = []
            return
        unique_flat, inverse = np.unique(flat, return_inverse=True)
        merged_mass = np.bincount(inverse, weights=mass, minlength=unique_flat.size).astype(
            float, copy=False
        )
        merged_entity = np.bincount(inverse, weights=entity, minlength=unique_flat.size).astype(
            float, copy=False
        )
        if unique_flat.size > int(self.max_nonzero_bins):
            raise ValueError(
                "Density sparse grid exceeded the configured nonzero-bin limit: "
                f"species='{self.species_label}', nonzero bins={unique_flat.size:,}, "
                f"limit={int(self.max_nonzero_bins):,}. Increase --grid-bin-width or request fewer species."
            )
        self._flat_chunks = [unique_flat]
        self._mass_chunks = [merged_mass]
        self._entity_chunks = [merged_entity]

    def finalize(self, *, n_frames: int) -> DensityGridProfile:
        self._merge_chunks()
        assert self._flat_chunks is not None
        assert self._mass_chunks is not None
        assert self._entity_chunks is not None
        if self._flat_chunks:
            flat_indices = np.asarray(self._flat_chunks[0], dtype=np.uint64)
            mass_sum = np.asarray(self._mass_chunks[0], dtype=float)
            entity_sum = np.asarray(self._entity_chunks[0], dtype=float)
        else:
            flat_indices = np.empty(0, dtype=np.uint64)
            mass_sum = np.empty(0, dtype=float)
            entity_sum = np.empty(0, dtype=float)
        if flat_indices.size > int(self.max_nonzero_bins):
            raise ValueError(
                "Density sparse grid exceeded the configured nonzero-bin limit: "
                f"species='{self.species_label}', nonzero bins={flat_indices.size:,}, "
                f"limit={int(self.max_nonzero_bins):,}. Increase --grid-bin-width or request fewer species."
            )
        footprint_bytes = (
            flat_indices.astype(np.uint64, copy=False).nbytes
            + mass_sum.astype(_DENSITY_GRID_STORAGE_DTYPE, copy=False).nbytes
            + entity_sum.astype(_DENSITY_GRID_STORAGE_DTYPE, copy=False).nbytes
        )
        LOGGER.debug(
            "Density sparse grid for %s: dense=%s cells, sparse=%s nonzero bins, estimated HDF5 payload~%.2f MiB.",
            _density_species_display_label(self.species_label),
            f"{self.dense_cell_count:,}",
            f"{flat_indices.size:,}",
            footprint_bytes / (1024.0 * 1024.0),
        )

        def aligned_side(
            values: dict[int, tuple[float, float]] | None,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            if values is None:
                return None, None
            mass_values = np.zeros(flat_indices.shape, dtype=_DENSITY_GRID_STORAGE_DTYPE)
            entity_values = np.zeros(flat_indices.shape, dtype=_DENSITY_GRID_STORAGE_DTYPE)
            if not values or flat_indices.size == 0:
                return mass_values, entity_values
            lookup = {int(value): index for index, value in enumerate(flat_indices)}
            for flat_index, (mass_value, entity_value) in values.items():
                target = lookup.get(int(flat_index))
                if target is not None:
                    mass_values[target] = mass_value
                    entity_values[target] = entity_value
            return mass_values, entity_values

        symmetric = bool(self.surface_reference is not None and self.surface_reference.is_symmetric)
        top_mass, top_entity = aligned_side(self._top_counts if symmetric else None)
        bottom_mass, bottom_entity = aligned_side(self._bottom_counts if symmetric else None)
        return DensityGridProfile(
            geometry_groups=_finish_geometry_groups(
                self.geometry_groups,
                include_paired=symmetric,
            ),
            species=self.species_label,
            x_bin_edges=np.asarray(self.x_bin_edges, dtype=float),
            y_bin_edges=np.asarray(self.y_bin_edges, dtype=float),
            z_bin_edges=np.asarray(self.z_bin_edges, dtype=float),
            distance_bin_edges=np.asarray(self.distance_bin_edges, dtype=float),
            flat_indices=flat_indices,
            mass_sum_g=np.asarray(mass_sum, dtype=_DENSITY_GRID_STORAGE_DTYPE),
            entity_sum=np.asarray(entity_sum, dtype=_DENSITY_GRID_STORAGE_DTYPE),
            n_frames=int(n_frames),
            units="g",
            number_density_units=_entity_density_units_for_species(
                self.species_label, volumetric=True
            ),
            grid_bin_width_A=float(self.grid_bin_width),
            grid_max_nonzero_bins=int(self.max_nonzero_bins),
            surface_axis=self.surface_axis,
            distance_coordinate_mode=self.distance_coordinate_mode,
            surface_position=self.surface_position,
            surface_position_std=self.surface_position_std,
            surface_estimate=self.surface_estimate,
            surface_reference=self.surface_reference,
            top_mass_sum_g=top_mass,
            bottom_mass_sum_g=bottom_mass,
            top_entity_sum=top_entity,
            bottom_entity_sum=bottom_entity,
            oh_cutoff_A=self.oh_cutoff_A,
            min_molecule_frames=self.min_molecule_frames,
        )


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


def _wrap_positions_to_periodic_cell(frame: Atoms, positions: np.ndarray) -> np.ndarray:
    try:
        return wrapped_frame_positions(frame, positions)
    except ValueError:
        return np.asarray(positions, dtype=float)


def _select_water_axis_values_with_masses(
    frame: Atoms, axis_index: int, oh_cutoff: float = H2O_OH_CUTOFF_A
) -> tuple[np.ndarray, np.ndarray]:
    """Return COM axis positions and molecular masses for detected water molecules."""
    triplets = _water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
    if triplets.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return _water_triplet_axis_values_with_masses(frame, triplets, axis_index)


def _select_water_axis_values_per_frame(
    frames: Sequence[Atoms], axis_index: int, oh_cutoff: float = H2O_OH_CUTOFF_A
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select water-molecule COM axis values with periodic cached-topology validation."""
    return _water_axis_values_per_frame_impl(
        frames,
        axis_index,
        oh_cutoff=oh_cutoff,
        progress_desc="Selecting H2O for density",
    )


def _select_molecule_axis_values_per_frame(
    frames: Sequence[Atoms],
    axis_index: int,
    molecule_label: str,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select O/H molecule COM axis values with adaptive cached-topology validation."""

    selections = _select_molecule_axis_values_for_labels_per_frame(
        frames,
        axis_index,
        (molecule_label,),
        oh_cutoff=oh_cutoff,
        progress_desc=f"Selecting {molecule_label} for density",
    )
    return selections[molecule_label]


def _select_molecule_axis_values_for_labels_per_frame(
    frames: Sequence[Atoms],
    axis_index: int,
    molecule_labels: tuple[str, ...] | list[str],
    *,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    progress_desc: str,
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    """Select O/H molecule COM axis values for multiple labels using one topology cache."""

    labels = tuple(molecule_labels)
    selections: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        label: ([], []) for label in labels
    }
    topology_cache = OHTopologyCache(
        oh_cutoff=oh_cutoff,
        context="density O/H molecule topology",
    )
    with ProgressBar(desc=progress_desc, total=len(frames), unit="frame") as progress:
        for frame_index, frame in enumerate(frames):
            topology = topology_cache.select(frame, frame_index=frame_index)
            for label in labels:
                positions, masses = _molecule_positions_with_masses(
                    frame,
                    topology.indices_for(label),
                )
                if needs_general_geometry([frame]):
                    positions = wrapped_frame_positions(frame, positions)
                selected_per_frame, selected_masses_per_frame = selections[label]
                selected_per_frame.append(np.asarray(positions[:, axis_index], dtype=float))
                selected_masses_per_frame.append(np.asarray(masses, dtype=float))
            progress.update()
    return selections


def _entity_density_units_for_species(
    species_label: str,
    *,
    volumetric: bool,
) -> str:
    entity_label = "molecule" if is_molecule_species_label(species_label) else "atom"
    if volumetric:
        return f"{entity_label}/nm^3"
    return f"{entity_label}/Angstrom"


def _entity_areal_density_units_for_species(species_label: str) -> str:
    """Return fallback areal-density units for one species selection."""

    entity_label = "molecule" if is_molecule_species_label(species_label) else "atom"
    return f"{entity_label}/Angstrom^2"


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
    entity_weights: np.ndarray | None = None,
) -> None:
    expected_mass = float(np.sum(masses))
    observed_mass = float(np.sum(mass_histogram))
    if not np.isclose(observed_mass, expected_mass, rtol=1.0e-12, atol=1.0e-30):
        raise ValueError(
            "Mass histogram does not conserve the selected mass in frame "
            f"{frame_index}: histogram={observed_mass:.16g}, selected={expected_mass:.16g}."
        )

    expected_entities = float(
        np.asarray(axis_values, dtype=float).size
        if entity_weights is None
        else np.sum(np.asarray(entity_weights, dtype=float))
    )
    observed_entities = float(np.sum(entity_histogram))
    if not np.isclose(observed_entities, expected_entities, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(
            "Entity histogram does not conserve the selected entity count in frame "
            f"{frame_index}: histogram={observed_entities}, selected={expected_entities}."
        )


def _validate_binned_heatmap_frame_conservation(
    *,
    frame_index: int,
    mass_histogram: np.ndarray,
    entity_histogram: np.ndarray,
    masses: np.ndarray,
    xy_values: np.ndarray,
) -> None:
    expected_mass = float(np.sum(masses))
    observed_mass = float(np.sum(mass_histogram))
    if not np.isclose(observed_mass, expected_mass, rtol=1.0e-12, atol=1.0e-30):
        raise ValueError(
            "Heatmap mass histogram does not conserve the selected mass in frame "
            f"{frame_index}: histogram={observed_mass:.16g}, selected={expected_mass:.16g}."
        )

    expected_entities = int(np.asarray(xy_values, dtype=float).shape[0])
    observed_entities = int(np.rint(np.sum(entity_histogram)))
    if observed_entities != expected_entities:
        raise ValueError(
            "Heatmap entity histogram does not conserve the selected entity count in frame "
            f"{frame_index}: histogram={observed_entities}, selected={expected_entities}."
        )


def _validate_density_heatmap_geometry(
    *,
    species_label: str,
    plane_axes: tuple[str, str],
    x_bin_count: int,
    y_bin_count: int,
) -> None:
    total_cells = int(x_bin_count) * int(y_bin_count)
    if total_cells <= _DENSITY_HEATMAP_MAX_CELLS:
        return
    estimated_mib = (
        total_cells * np.dtype(float).itemsize * _DENSITY_HEATMAP_ARRAY_COUNT_ESTIMATE
    ) / (1024.0 * 1024.0)
    raise ValueError(
        "Density heatmap grid is too large for a safe compute path: "
        f"species='{species_label}', plane='{plane_axes[0]}{plane_axes[1]}', "
        f"bins={x_bin_count}x{y_bin_count} ({total_cells:,} cells), "
        f"estimated working arrays~{estimated_mib:.2f} MiB. "
        "Increase --bin-width to reduce the heatmap grid size."
    )


def _resolve_frame_block_index_by_frame(frame_count: int) -> np.ndarray | None:
    """Return one frame->block index map for saved block statistics or ``None``."""

    block_resolution = resolve_block_slices(int(frame_count))
    if block_resolution is None:
        return None
    block_slices, _block_lengths = block_resolution
    block_index_by_frame = np.full(int(frame_count), -1, dtype=int)
    for block_index, block_slice in enumerate(block_slices):
        block_index_by_frame[block_slice] = int(block_index)
    return block_index_by_frame


def _empty_density_statistics_moments(
    *,
    n_bins: int,
    n_blocks: int | None,
) -> _DensityStatisticsMoments:
    block_shape = None if n_blocks is None else (int(n_blocks), int(n_bins))
    return _DensityStatisticsMoments(
        point_count=np.zeros(int(n_bins), dtype=int),
        sample_n=np.zeros(int(n_bins), dtype=int),
        sample_sum=np.zeros(int(n_bins), dtype=float),
        sample_sumsq=np.zeros(int(n_bins), dtype=float),
        block_sum=None if block_shape is None else np.zeros(block_shape, dtype=float),
        block_n=None if block_shape is None else np.zeros(block_shape, dtype=int),
    )


def _update_density_statistics_moments(
    moments: _DensityStatisticsMoments,
    *,
    histogram: np.ndarray,
    density_values: np.ndarray,
    frame_index: int,
    block_index_by_frame: np.ndarray | None,
) -> None:
    histogram_array = np.asarray(histogram, dtype=float)
    density_array = np.asarray(density_values, dtype=float)
    finite = np.isfinite(density_array)
    moments.point_count += np.rint(histogram_array).astype(int)
    if not np.any(finite):
        return
    moments.sample_n[finite] += 1
    moments.sample_sum[finite] += density_array[finite]
    moments.sample_sumsq[finite] += density_array[finite] ** 2
    if (
        block_index_by_frame is None
        or moments.block_sum is None
        or moments.block_n is None
        or frame_index < 0
        or frame_index >= block_index_by_frame.size
    ):
        return
    block_index = int(block_index_by_frame[frame_index])
    if block_index < 0:
        return
    moments.block_sum[block_index, finite] += density_array[finite]
    moments.block_n[block_index, finite] += 1


def _block_mean_matrix_from_density_moments(
    moments: _DensityStatisticsMoments,
) -> np.ndarray | None:
    if moments.block_sum is None or moments.block_n is None:
        return None
    block_values = np.full(moments.block_sum.shape, np.nan, dtype=float)
    valid = moments.block_n > 0
    block_values[valid] = moments.block_sum[valid] / moments.block_n[valid].astype(float)
    return block_values


def _finalize_density_statistics_moments(
    moments: _DensityStatisticsMoments,
) -> SeriesStatistics:
    return build_series_statistics_from_moments(
        point_count=moments.point_count,
        sample_n=moments.sample_n,
        sample_sum=moments.sample_sum,
        sample_sumsq=moments.sample_sumsq,
        block_values=_block_mean_matrix_from_density_moments(moments),
    )


def _uniform_bin_indices(
    values: np.ndarray, *, bin_start: float, bin_width: float, n_bins: int
) -> np.ndarray:
    """Map values onto one fixed uniform bin grid using NumPy histogram edge semantics."""

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return np.empty(0, dtype=int)
    indices = np.floor((data - float(bin_start)) / float(bin_width)).astype(int)
    np.clip(indices, 0, int(n_bins) - 1, out=indices)
    return indices


def _bin_edge_tolerance(bin_width: float) -> float:
    return max(abs(float(bin_width)) * 1.0e-9, 1.0e-12)


def _build_density_validation_plan(frame_count: int) -> _DensityValidationPlan:
    """Return representative validation frames for normal mode and full validation in debug."""

    total = int(frame_count)
    strict = _DENSITY_STRICT_VALIDATION or LOGGER.isEnabledFor(logging.DEBUG)
    if total <= 0:
        return _DensityValidationPlan(strict=strict, frame_indices=frozenset())
    if strict or total <= 4:
        return _DensityValidationPlan(
            strict=strict,
            frame_indices=frozenset(range(total)),
        )
    sample_count = min(6, total)
    sample_positions = np.linspace(0, total - 1, num=sample_count, dtype=int)
    sample_indices = {0, total - 1, *(int(index) for index in sample_positions.tolist())}
    return _DensityValidationPlan(strict=False, frame_indices=frozenset(sample_indices))


def _build_line_normalization_cache(
    *,
    frames: Sequence[Atoms],
    axis_index: int,
    bin_width: float,
) -> _DensityLineNormalizationCache:
    use_volumetric_density = all(_frame_has_usable_cell(frame, axis_index) for frame in frames)
    slice_volumes: np.ndarray | None = None
    framewise_normalization = False
    if use_volumetric_density:
        slice_volumes = np.empty(len(frames), dtype=float)
        for index, frame in enumerate(frames):
            cell = np.asarray(frame.cell.array, dtype=float)
            axis_length = np.linalg.norm(cell[axis_index])
            volume = abs(float(np.linalg.det(cell)))
            cross_section = volume / axis_length
            slice_volumes[index] = cross_section * bin_width
        framewise_normalization = not np.allclose(
            slice_volumes,
            slice_volumes[0],
            rtol=0.0,
            atol=1e-12,
        )
    return _DensityLineNormalizationCache(
        use_volumetric_density=use_volumetric_density,
        slice_volumes=slice_volumes,
        framewise_normalization=framewise_normalization,
        frames=frames if use_volumetric_density else None,
        axis_index=axis_index,
    )


def _build_heatmap_normalization_cache(
    *,
    frames: Sequence[Atoms],
    orth_axis_index: int,
    plane_axis_indices: tuple[int, int],
    bin_width: float,
) -> _DensityHeatmapNormalizationCache:
    use_volumetric_density = all(
        _common_frame_has_usable_cell(frame, axis_index=orth_axis_index) for frame in frames
    )
    slice_volumes: np.ndarray | None = None
    framewise_normalization = False
    if use_volumetric_density:
        slice_volumes = np.empty(len(frames), dtype=float)
        for index, frame in enumerate(frames):
            cell = np.asarray(frame.cell.array, dtype=float)
            orth_length = np.linalg.norm(cell[orth_axis_index])
            volume = abs(float(np.linalg.det(cell)))
            slice_volumes[index] = (volume / orth_length) * (bin_width * bin_width)
        framewise_normalization = not np.allclose(
            slice_volumes,
            slice_volumes[0],
            rtol=0.0,
            atol=1e-12,
        )
    return _DensityHeatmapNormalizationCache(
        use_volumetric_density=use_volumetric_density,
        slice_volumes=slice_volumes,
        framewise_normalization=framewise_normalization,
        frames=frames if use_volumetric_density else None,
        plane_axis_indices=plane_axis_indices if use_volumetric_density else None,
    )


def _resolve_density_target_specs(
    frames: Sequence[Atoms],
    *,
    species: str | None,
) -> list[_DensityTargetSpec]:
    selection_mode, species_label = _normalize_species_query(
        species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if selection_mode in {"all", "elements"}:
        element_species = available_element_species(frames)
        if not element_species:
            raise ValueError("No elements found in trajectory.")
        element_targets = [
            _DensityTargetSpec(
                species_label=element,
                count_label="atoms",
                selection_kind="element",
            )
            for element in element_species
        ]
        if selection_mode == "elements":
            return element_targets
        raw_species_targets = [
            _DensityTargetSpec(
                species_label=f"species:{label}",
                count_label="atoms",
                selection_kind="species",
            )
            for label in grouped_raw_species_for_split_elements(frames)
        ]
        return [
            *element_targets,
            *raw_species_targets,
            *(
                _DensityTargetSpec(
                    species_label=label,
                    count_label="molecules",
                    selection_kind="molecule",
                    apply_min_molecule_frames=True,
                )
                for label in MOLECULE_SPECIES_LABELS
            ),
        ]
    if selection_mode == "molecules":
        return [
            _DensityTargetSpec(
                species_label=label,
                count_label="molecules",
                selection_kind="molecule",
                apply_min_molecule_frames=True,
            )
            for label in MOLECULE_SPECIES_LABELS
        ]
    if selection_mode == "molecule":
        return [
            _DensityTargetSpec(
                species_label=species_label,
                count_label="molecules",
                selection_kind="molecule",
            )
        ]
    return [
        _DensityTargetSpec(
            species_label=species_label,
            count_label="atoms",
            selection_kind=selection_mode if selection_mode == "species" else "element",
        )
    ]


def _normalize_density_outputs(outputs: str | None) -> str:
    normalized = str(outputs or "all").strip().lower()
    if normalized == "heatmap":
        raise ValueError(
            "Density heatmap output was replaced by sparse 3D density grids. "
            "Use --outputs 3d for grid-backed 2D slicing, or --outputs all for 1D plus 3D."
        )
    if normalized not in {"1d", "3d", "all"}:
        raise ValueError("density outputs must be one of: 1d, 3d, all")
    return normalized


def _normalize_density_heatmap_planes(
    heatmap_planes: list[str] | tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    if heatmap_planes is None:
        return (("x", "y"), ("x", "z"), ("y", "z"))
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_plane in heatmap_planes:
        token = str(raw_plane or "").strip().lower()
        if token not in {"xy", "xz", "yz"}:
            raise ValueError("density heatmap planes must be selected from: xy, xz, yz")
        plane_axes = (token[0], token[1])
        if plane_axes not in seen:
            normalized.append(plane_axes)
            seen.add(plane_axes)
    if not normalized:
        raise ValueError("At least one density heatmap plane must be selected.")
    return tuple(normalized)


def _density_cell_bounds_cover_line_outputs(
    *,
    raw_axes: tuple[str, ...],
    cell_bounds_by_axis: Mapping[str, tuple[float, float] | None],
    distance_hist_bounds: tuple[float, float] | None,
) -> bool:
    return distance_hist_bounds is not None and all(
        cell_bounds_by_axis.get(axis) is not None for axis in raw_axes
    )


def _surface_reference_histogram_bounds(
    reference: SurfaceReference,
) -> tuple[float, float] | None:
    """Return complete-cell bounds for a canonical surface coordinate."""

    periods = np.asarray(reference.axis_period_per_frame_A, dtype=float)
    if periods.ndim != 1 or periods.size == 0 or np.any(~np.isfinite(periods)):
        return None
    if np.any(periods <= 0.0):
        return None

    def plane_values(estimate: SurfaceEstimate | None) -> np.ndarray | None:
        if estimate is None:
            return None
        values = np.asarray(estimate.frame_values, dtype=float)
        if values.shape != periods.shape or np.any(~np.isfinite(values)):
            return None
        return values

    top = plane_values(reference.top)
    bottom = plane_values(reference.bottom)
    lower: float
    upper: float
    if reference.resolved_layout in {"single-top", "single-bottom"}:
        if top is None or bottom is None:
            return None
        # The slab interior remains signed negative while the entire periodic
        # non-substrate arc is unwrapped outward as positive distance.
        wrap_thresholds = np.asarray(
            [
                _single_interface_wrap_threshold(reference, frame_index)
                for frame_index in range(periods.size)
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(wrap_thresholds)):
            return None
        slab_signed = (
            wrap_thresholds - top
            if reference.resolved_layout == "single-top"
            else bottom - wrap_thresholds
        )
        lower = float(np.min(slab_signed))
        upper = float(np.max(periods + slab_signed))
    elif reference.resolved_layout == "symmetric" and top is not None and bottom is not None:
        # Between the two planes the folded coordinate reaches its minimum at
        # their midplane. The maximum over the complete periodic image lies
        # at one of the canonical cell boundaries.
        lower = float(np.min(0.5 * (bottom - top)))
        upper = float(np.max(np.maximum(periods - top, bottom)))
    else:
        return None
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def _density_needs_observed_bounds_prepass(
    *,
    normalized_binning: str,
    include_line_outputs: bool,
    include_grid_outputs: bool,
    selected_heatmap_planes: tuple[tuple[str, str], ...],
    raw_axes: tuple[str, ...],
    cell_bounds_by_axis: Mapping[str, tuple[float, float] | None],
    distance_hist_bounds: tuple[float, float] | None,
) -> bool:
    heatmap_axes = {axis for plane in selected_heatmap_planes for axis in plane}
    if any(cell_bounds_by_axis.get(axis) is None for axis in heatmap_axes):
        return True
    if include_grid_outputs and (
        any(cell_bounds_by_axis.get(axis) is None for axis in raw_axes)
        or distance_hist_bounds is None
    ):
        return True
    if not include_line_outputs:
        return False
    if normalized_binning != "cell":
        return True
    return not _density_cell_bounds_cover_line_outputs(
        raw_axes=raw_axes,
        cell_bounds_by_axis=cell_bounds_by_axis,
        distance_hist_bounds=distance_hist_bounds,
    )


def _resolve_density_active_targets_without_bounds_prepass(
    frames: Sequence[Atoms],
    target_specs: list[_DensityTargetSpec],
) -> list[_DensityTargetSpec] | None:
    """Return active targets only when that can be decided without selection scanning."""

    element_species = set(available_element_species(frames))
    active_targets: list[_DensityTargetSpec] = []
    for target in target_specs:
        if target.selection_kind == "element":
            if target.species_label in element_species:
                active_targets.append(target)
            continue
        if target.selection_kind == "species":
            raw_species = set(available_raw_species(frames))
            if species_selector_raw_label(target.species_label) in raw_species:
                active_targets.append(target)
            continue
        if target.selection_kind == "molecule":
            # Molecule activity depends on topology detection, so keep the conservative
            # selection scan for O/H molecule runs.
            return None
        return None
    return active_targets


class _DensityElementSelector:
    """Select element positions/masses with a stable-symbol fast path."""

    def __init__(
        self,
        element_labels: list[str],
        *,
        validation_stride: int = H2O_VALIDATION_STRIDE,
    ) -> None:
        self.element_labels = tuple(element_labels)
        self.validation_stride = max(1, int(validation_stride))
        self._reference_symbols: np.ndarray | None = None
        self._masks_by_label: dict[str, np.ndarray] = {}
        self._stable = True

    def select(self, frame: Atoms, *, frame_index: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        positions = np.asarray(frame.positions, dtype=float)
        masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G
        symbols = np.asarray(frame.get_chemical_symbols())
        if self._reference_symbols is None:
            self._reference_symbols = np.asarray(symbols, dtype=object)
            self._masks_by_label = {
                label: np.asarray(symbols == label, dtype=bool) for label in self.element_labels
            }
            return self._select_with_masks(positions, masses)

        if self._stable:
            reference = self._reference_symbols
            needs_validation = int(frame_index) % self.validation_stride == 0
            if symbols.shape != reference.shape:
                self._stable = False
                LOGGER.info(
                    "Density element selection detected atom-count changes; using dynamic element grouping."
                )
            elif needs_validation and not np.array_equal(symbols, reference):
                self._stable = False
                LOGGER.info(
                    "Density element selection detected atom-order changes; using dynamic element grouping."
                )

        if self._stable:
            return self._select_with_masks(positions, masses)
        return self._select_dynamic(positions, masses, symbols)

    def _select_with_masks(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, mask in self._masks_by_label.items():
            if np.any(mask):
                selections[label] = (
                    np.asarray(positions[mask], dtype=float),
                    np.asarray(masses[mask], dtype=float),
                )
        return selections

    def _select_dynamic(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        symbols: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label in self.element_labels:
            mask = symbols == label
            if np.any(mask):
                selections[label] = (
                    np.asarray(positions[mask], dtype=float),
                    np.asarray(masses[mask], dtype=float),
                )
        return selections


def _density_observed_bounds_from_cell_bounds(
    *,
    active_targets: list[_DensityTargetSpec],
    raw_axes: tuple[str, ...],
    cell_bounds_by_axis: Mapping[str, tuple[float, float] | None],
    distance_hist_bounds: tuple[float, float] | None,
) -> dict[str, _DensityObservedBounds]:
    observed_bounds_by_target: dict[str, _DensityObservedBounds] = {}
    for target in active_targets:
        axis_lower: dict[str, float] = {}
        axis_upper: dict[str, float] = {}
        for axis in raw_axes:
            bounds = cell_bounds_by_axis[axis]
            if bounds is None:
                raise ValueError(f"Missing cell bounds for density axis '{axis}'.")
            axis_lower[axis] = float(bounds[0])
            axis_upper[axis] = float(bounds[1])
        if distance_hist_bounds is not None:
            axis_lower["distance"] = float(distance_hist_bounds[0])
            axis_upper["distance"] = float(distance_hist_bounds[1])
        observed_bounds_by_target[target.species_label] = _DensityObservedBounds(
            axis_lower=axis_lower,
            axis_upper=axis_upper,
        )
    return observed_bounds_by_target


def _select_density_targets_in_frame(
    frame: Atoms,
    *,
    frame_index: int,
    target_specs: list[_DensityTargetSpec],
    topology_cache: OHTopologyCache | None,
    element_selector: _DensityElementSelector | None = None,
    wrap_positions_to_cell: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return one frame's selected positions/masses for each active density target."""

    selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    element_labels = [
        target.species_label for target in target_specs if target.selection_kind == "element"
    ]
    if element_labels:
        if element_selector is not None:
            selections.update(element_selector.select(frame, frame_index=frame_index))
        else:
            positions = np.asarray(frame.positions, dtype=float)
            masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G
            symbols = np.asarray(frame.get_chemical_symbols())
            for label in element_labels:
                mask = symbols == label
                if np.any(mask):
                    selections[label] = (
                        np.asarray(positions[mask], dtype=float),
                        np.asarray(masses[mask], dtype=float),
                    )
        if wrap_positions_to_cell:
            selections = {
                label: (_wrap_positions_to_periodic_cell(frame, positions), masses)
                for label, (positions, masses) in selections.items()
            }
    species_labels = [
        target.species_label for target in target_specs if target.selection_kind == "species"
    ]
    if species_labels:
        positions = np.asarray(frame.positions, dtype=float)
        masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G
        labels = raw_species_labels(frame)
        for label in species_labels:
            raw_label = species_selector_raw_label(label)
            mask = labels == raw_label
            if np.any(mask):
                selected_positions = np.asarray(positions[mask], dtype=float)
                selections[label] = (
                    _wrap_positions_to_periodic_cell(frame, selected_positions)
                    if wrap_positions_to_cell
                    else selected_positions,
                    np.asarray(masses[mask], dtype=float),
                )
    molecule_labels = [
        target.species_label for target in target_specs if target.selection_kind == "molecule"
    ]
    if molecule_labels:
        active_cache = topology_cache or OHTopologyCache(context="density O/H molecule topology")
        topology = active_cache.select(frame, frame_index=frame_index)
        for label in molecule_labels:
            positions, masses = _molecule_positions_with_masses(
                frame,
                topology.indices_for(label),
            )
            if positions.size > 0:
                selections[label] = (
                    _wrap_positions_to_periodic_cell(frame, positions)
                    if wrap_positions_to_cell
                    else np.asarray(positions, dtype=float),
                    np.asarray(masses, dtype=float),
                )
    return selections


def _resolve_density_line_bin_edges_from_bounds(
    *,
    observed_bounds: _DensityObservedBounds,
    axis_id: str,
    species_label: str,
    bin_width: float,
    histogram_bounds: tuple[float, float] | None,
) -> np.ndarray:
    if not observed_bounds.has_axis_data(axis_id):
        raise ValueError(f"No entities found for selection '{species_label}' in trajectory.")
    data_min = float(observed_bounds.axis_lower[axis_id])
    data_max = float(observed_bounds.axis_upper[axis_id])
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
    if bin_edges[-1] < data_max and not np.isclose(bin_edges[-1], data_max):
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    elif histogram_bounds is None and n_bins == 1:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    return np.asarray(bin_edges, dtype=float)


def _resolve_density_heatmap_bin_edges_from_bounds(
    *,
    observed_bounds: _DensityObservedBounds,
    plane_axes: tuple[str, str],
    species_label: str,
    bin_width: float,
    histogram_bounds: tuple[tuple[float, float] | None, tuple[float, float] | None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    x_axis, y_axis = plane_axes
    if not observed_bounds.has_axis_data(x_axis) or not observed_bounds.has_axis_data(y_axis):
        raise ValueError(f"No entities found for selection '{species_label}' in trajectory.")
    data_min_x = float(observed_bounds.axis_lower[x_axis])
    data_max_x = float(observed_bounds.axis_upper[x_axis])
    data_min_y = float(observed_bounds.axis_lower[y_axis])
    data_max_y = float(observed_bounds.axis_upper[y_axis])
    if histogram_bounds is not None:
        x_bounds, y_bounds = histogram_bounds
        if x_bounds is not None:
            bounds_min_x = float(x_bounds[0])
            bounds_max_x = float(x_bounds[1])
            if (
                np.isfinite(bounds_min_x)
                and np.isfinite(bounds_max_x)
                and bounds_max_x > bounds_min_x
            ):
                data_min_x = min(data_min_x, bounds_min_x)
                data_max_x = max(data_max_x, bounds_max_x)
        if y_bounds is not None:
            bounds_min_y = float(y_bounds[0])
            bounds_max_y = float(y_bounds[1])
            if (
                np.isfinite(bounds_min_y)
                and np.isfinite(bounds_max_y)
                and bounds_max_y > bounds_min_y
            ):
                data_min_y = min(data_min_y, bounds_min_y)
                data_max_y = max(data_max_y, bounds_max_y)
    if np.isclose(data_min_x, data_max_x):
        data_max_x = data_min_x + bin_width
    if np.isclose(data_min_y, data_max_y):
        data_max_y = data_min_y + bin_width
    x_span = data_max_x - data_min_x
    y_span = data_max_y - data_min_y
    x_bin_count = max(1, int(np.ceil(x_span / bin_width)))
    y_bin_count = max(1, int(np.ceil(y_span / bin_width)))
    _validate_density_heatmap_geometry(
        species_label=species_label,
        plane_axes=plane_axes,
        x_bin_count=x_bin_count,
        y_bin_count=y_bin_count,
    )
    x_bin_edges = data_min_x + np.arange(x_bin_count + 1, dtype=float) * bin_width
    y_bin_edges = data_min_y + np.arange(y_bin_count + 1, dtype=float) * bin_width
    if x_bin_edges[-1] < data_max_x and not np.isclose(x_bin_edges[-1], data_max_x):
        x_bin_edges = np.append(x_bin_edges, x_bin_edges[-1] + bin_width)
    if y_bin_edges[-1] < data_max_y and not np.isclose(y_bin_edges[-1], data_max_y):
        y_bin_edges = np.append(y_bin_edges, y_bin_edges[-1] + bin_width)
    return np.asarray(x_bin_edges, dtype=float), np.asarray(y_bin_edges, dtype=float)


def _resolve_density_grid_bin_edges_from_bounds(
    *,
    observed_bounds: _DensityObservedBounds,
    axis_id: str,
    species_label: str,
    grid_bin_width: float,
    histogram_bounds: tuple[float, float] | None,
) -> np.ndarray:
    return _resolve_density_line_bin_edges_from_bounds(
        observed_bounds=observed_bounds,
        axis_id=axis_id,
        species_label=species_label,
        bin_width=grid_bin_width,
        histogram_bounds=histogram_bounds,
    )


def _bincount_1d_histograms(
    *,
    values: np.ndarray,
    masses: np.ndarray,
    bin_start: float,
    bin_width: float,
    n_bins: int,
    bin_end: float,
    entity_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mass and entity histograms on one fixed uniform grid via ``np.bincount``."""

    axis_values = np.asarray(values, dtype=float)
    mass_values = np.asarray(masses, dtype=float)
    entity_values = (
        np.ones(axis_values.shape, dtype=float)
        if entity_weights is None
        else np.asarray(entity_weights, dtype=float)
    )
    if entity_values.shape != axis_values.shape:
        raise ValueError("Density entity weights must match the selected coordinate shape.")
    edge_tol = _bin_edge_tolerance(bin_width)
    valid = (
        np.isfinite(axis_values)
        & np.isfinite(mass_values)
        & np.isfinite(entity_values)
        & (axis_values >= bin_start - edge_tol)
        & (axis_values <= bin_end + edge_tol)
    )
    if not np.any(valid):
        return np.zeros(int(n_bins), dtype=float), np.zeros(int(n_bins), dtype=float)
    indices = _uniform_bin_indices(
        axis_values[valid],
        bin_start=bin_start,
        bin_width=bin_width,
        n_bins=n_bins,
    )
    return (
        np.bincount(indices, weights=mass_values[valid], minlength=int(n_bins)).astype(
            float, copy=False
        ),
        np.bincount(indices, weights=entity_values[valid], minlength=int(n_bins)).astype(
            float, copy=False
        ),
    )


def _bincount_2d_histograms(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    masses: np.ndarray,
    x_bin_start: float,
    y_bin_start: float,
    bin_width: float,
    x_bin_count: int,
    y_bin_count: int,
    x_bin_end: float,
    y_bin_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mass and entity 2D histograms via flattened ``np.bincount``."""

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    mass_values = np.asarray(masses, dtype=float)
    total_bins = int(x_bin_count) * int(y_bin_count)
    if x_array.size == 0 or y_array.size == 0:
        zero = np.zeros((int(x_bin_count), int(y_bin_count)), dtype=float)
        return zero, zero.copy()
    edge_tol = _bin_edge_tolerance(bin_width)
    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
        & np.isfinite(mass_values)
        & (x_array >= x_bin_start - edge_tol)
        & (x_array <= x_bin_end + edge_tol)
        & (y_array >= y_bin_start - edge_tol)
        & (y_array <= y_bin_end + edge_tol)
    )
    if not np.any(valid):
        zero = np.zeros((int(x_bin_count), int(y_bin_count)), dtype=float)
        return zero, zero.copy()
    x_indices = _uniform_bin_indices(
        x_array[valid],
        bin_start=x_bin_start,
        bin_width=bin_width,
        n_bins=x_bin_count,
    )
    y_indices = _uniform_bin_indices(
        y_array[valid],
        bin_start=y_bin_start,
        bin_width=bin_width,
        n_bins=y_bin_count,
    )
    flat_indices = x_indices * int(y_bin_count) + y_indices
    mass_histogram = (
        np.bincount(
            flat_indices,
            weights=mass_values[valid],
            minlength=total_bins,
        )
        .astype(float, copy=False)
        .reshape((int(x_bin_count), int(y_bin_count)))
    )
    entity_histogram = (
        np.bincount(
            flat_indices,
            minlength=total_bins,
        )
        .astype(float, copy=False)
        .reshape((int(x_bin_count), int(y_bin_count)))
    )
    return mass_histogram, entity_histogram


def _density_grid_axis_indices(
    flat_indices: np.ndarray,
    *,
    grid_shape: tuple[int, int, int, int],
    axis_index: int,
) -> np.ndarray:
    """Decode one axis from flattened X/Y/Z/distance sparse-grid indices."""

    flat = np.asarray(flat_indices, dtype=np.uint64)
    if flat.size == 0:
        return np.empty(0, dtype=np.int64)
    nx, ny, nz, nd = (int(value) for value in grid_shape)
    if axis_index == 0:
        return (flat // np.uint64(ny * nz * nd)).astype(np.int64, copy=False)
    if axis_index == 1:
        return ((flat // np.uint64(nz * nd)) % np.uint64(ny)).astype(np.int64, copy=False)
    if axis_index == 2:
        return ((flat // np.uint64(nd)) % np.uint64(nz)).astype(np.int64, copy=False)
    if axis_index == 3:
        return (flat % np.uint64(nd)).astype(np.int64, copy=False)
    raise ValueError("Density grid axis_index must be 0, 1, 2, or 3.")


def _bincount_sparse_density_grid(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    distance_values: np.ndarray,
    masses: np.ndarray,
    x_bin_edges: np.ndarray,
    y_bin_edges: np.ndarray,
    z_bin_edges: np.ndarray,
    distance_bin_edges: np.ndarray,
    grid_shape: tuple[int, int, int, int],
    entity_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse flat grid hits and mass/entity sums for one frame."""

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    z_array = np.asarray(z_values, dtype=float)
    distance_array = np.asarray(distance_values, dtype=float)
    mass_values = np.asarray(masses, dtype=float)
    entity_values = (
        np.ones(x_array.shape, dtype=float)
        if entity_weights is None
        else np.asarray(entity_weights, dtype=float)
    )
    if entity_values.shape != x_array.shape:
        raise ValueError("Density grid entity weights must match the selected coordinate shape.")
    if x_array.size == 0:
        empty_flat = np.empty(0, dtype=np.uint64)
        empty_values = np.empty(0, dtype=float)
        return empty_flat, empty_values, empty_values.copy()
    nx, ny, nz, nd = (int(value) for value in grid_shape)
    if min(nx, ny, nz, nd) <= 0:
        raise ValueError("Density sparse grid has an invalid zero-length axis.")
    x_edges = np.asarray(x_bin_edges, dtype=float)
    y_edges = np.asarray(y_bin_edges, dtype=float)
    z_edges = np.asarray(z_bin_edges, dtype=float)
    distance_edges = np.asarray(distance_bin_edges, dtype=float)
    x_start, x_end = float(x_edges[0]), float(x_edges[-1])
    y_start, y_end = float(y_edges[0]), float(y_edges[-1])
    z_start, z_end = float(z_edges[0]), float(z_edges[-1])
    distance_start, distance_end = float(distance_edges[0]), float(distance_edges[-1])
    x_width = uniform_bin_width_from_edges(x_edges, source_label="density sparse grid X")
    y_width = uniform_bin_width_from_edges(y_edges, source_label="density sparse grid Y")
    z_width = uniform_bin_width_from_edges(z_edges, source_label="density sparse grid Z")
    distance_width = uniform_bin_width_from_edges(
        distance_edges,
        source_label="density sparse grid distance",
    )
    x_tol = _bin_edge_tolerance(x_width)
    y_tol = _bin_edge_tolerance(y_width)
    z_tol = _bin_edge_tolerance(z_width)
    distance_tol = _bin_edge_tolerance(distance_width)
    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
        & np.isfinite(z_array)
        & np.isfinite(distance_array)
        & np.isfinite(mass_values)
        & np.isfinite(entity_values)
        & (x_array >= x_start - x_tol)
        & (x_array <= x_end + x_tol)
        & (y_array >= y_start - y_tol)
        & (y_array <= y_end + y_tol)
        & (z_array >= z_start - z_tol)
        & (z_array <= z_end + z_tol)
        & (distance_array >= distance_start - distance_tol)
        & (distance_array <= distance_end + distance_tol)
    )
    if not np.any(valid):
        empty_flat = np.empty(0, dtype=np.uint64)
        empty_values = np.empty(0, dtype=float)
        return empty_flat, empty_values, empty_values.copy()
    ix = _uniform_bin_indices(x_array[valid], bin_start=x_start, bin_width=x_width, n_bins=nx)
    iy = _uniform_bin_indices(y_array[valid], bin_start=y_start, bin_width=y_width, n_bins=ny)
    iz = _uniform_bin_indices(z_array[valid], bin_start=z_start, bin_width=z_width, n_bins=nz)
    idistance = _uniform_bin_indices(
        distance_array[valid],
        bin_start=distance_start,
        bin_width=distance_width,
        n_bins=nd,
    )
    flat = (
        (ix.astype(np.uint64) * np.uint64(ny) + iy.astype(np.uint64)) * np.uint64(nz)
        + iz.astype(np.uint64)
    ) * np.uint64(nd) + idistance.astype(np.uint64)
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    mass_sum = np.bincount(
        inverse,
        weights=mass_values[valid],
        minlength=unique_flat.size,
    ).astype(float, copy=False)
    entity_sum = np.bincount(
        inverse,
        weights=entity_values[valid],
        minlength=unique_flat.size,
    ).astype(float, copy=False)
    return unique_flat.astype(np.uint64, copy=False), mass_sum, entity_sum


def _compute_density_heatmap_profile_from_selected(
    *,
    frames: Sequence[Atoms],
    selected_xy_per_frame: list[np.ndarray],
    selected_masses_per_frame: list[np.ndarray],
    plane_axes: tuple[str, str],
    orthogonal_axis: str,
    species_label: str,
    bin_width: float,
    histogram_bounds: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None,
    aggregate_binning_progress: ProgressBar | None = None,
    oh_cutoff_A: float | None = None,
    min_molecule_frames: int | None = None,
) -> DensityHeatmapProfile:
    """Build a :class:`DensityHeatmapProfile` from already-selected planar coordinates."""

    if len(selected_xy_per_frame) != len(selected_masses_per_frame):
        raise ValueError(
            "Selected planar coordinate and mass arrays must have matching frame counts."
        )
    n_selected_total = sum(
        np.asarray(values, dtype=float).shape[0] for values in selected_xy_per_frame
    )
    if n_selected_total == 0:
        raise ValueError(f"No entities found for selection '{species_label}' in trajectory.")

    non_empty = [
        np.asarray(values, dtype=float)
        for values in selected_xy_per_frame
        if np.asarray(values).size > 0
    ]
    x_values_all = np.concatenate([values[:, 0] for values in non_empty])
    y_values_all = np.concatenate([values[:, 1] for values in non_empty])
    data_min_x = float(np.min(x_values_all))
    data_max_x = float(np.max(x_values_all))
    data_min_y = float(np.min(y_values_all))
    data_max_y = float(np.max(y_values_all))
    if histogram_bounds is not None:
        x_bounds, y_bounds = histogram_bounds
        if x_bounds is not None:
            bounds_min_x = float(x_bounds[0])
            bounds_max_x = float(x_bounds[1])
            if (
                np.isfinite(bounds_min_x)
                and np.isfinite(bounds_max_x)
                and bounds_max_x > bounds_min_x
            ):
                data_min_x = min(data_min_x, bounds_min_x)
                data_max_x = max(data_max_x, bounds_max_x)
        if y_bounds is not None:
            bounds_min_y = float(y_bounds[0])
            bounds_max_y = float(y_bounds[1])
            if (
                np.isfinite(bounds_min_y)
                and np.isfinite(bounds_max_y)
                and bounds_max_y > bounds_min_y
            ):
                data_min_y = min(data_min_y, bounds_min_y)
                data_max_y = max(data_max_y, bounds_max_y)

    if np.isclose(data_min_x, data_max_x):
        data_max_x = data_min_x + bin_width
    if np.isclose(data_min_y, data_max_y):
        data_max_y = data_min_y + bin_width

    x_span = data_max_x - data_min_x
    y_span = data_max_y - data_min_y
    x_bin_count = max(1, int(np.ceil(x_span / bin_width)))
    y_bin_count = max(1, int(np.ceil(y_span / bin_width)))
    _validate_density_heatmap_geometry(
        species_label=species_label,
        plane_axes=plane_axes,
        x_bin_count=x_bin_count,
        y_bin_count=y_bin_count,
    )
    x_bin_edges = data_min_x + np.arange(x_bin_count + 1, dtype=float) * bin_width
    y_bin_edges = data_min_y + np.arange(y_bin_count + 1, dtype=float) * bin_width
    if x_bin_edges[-1] < data_max_x and not np.isclose(x_bin_edges[-1], data_max_x):
        x_bin_edges = np.append(x_bin_edges, x_bin_edges[-1] + bin_width)
    if y_bin_edges[-1] < data_max_y and not np.isclose(y_bin_edges[-1], data_max_y):
        y_bin_edges = np.append(y_bin_edges, y_bin_edges[-1] + bin_width)
    x_bin_count = int(x_bin_edges.size - 1)
    y_bin_count = int(y_bin_edges.size - 1)
    x_bin_start = float(x_bin_edges[0])
    y_bin_start = float(y_bin_edges[0])
    x_bin_end = float(x_bin_edges[-1])
    y_bin_end = float(y_bin_edges[-1])

    n_frames = len(frames)
    mass_histogram_sum = np.zeros((x_bin_count, y_bin_count), dtype=float)
    entity_histogram_sum = np.zeros_like(mass_histogram_sum)
    orth_axis_index = axis_to_index(orthogonal_axis)
    use_volumetric_density = all(
        _common_frame_has_usable_cell(frame, axis_index=orth_axis_index) for frame in frames
    )
    slice_volumes: np.ndarray | None = None
    framewise_mass_density_sum: np.ndarray | None = None
    framewise_entity_density_sum: np.ndarray | None = None
    volume_coverage_count: np.ndarray | None = None
    if use_volumetric_density:
        slice_volumes = np.empty(n_frames, dtype=float)
        for index, frame in enumerate(frames):
            cell = np.asarray(frame.cell.array, dtype=float)
            orth_length = np.linalg.norm(cell[orth_axis_index])
            volume = abs(float(np.linalg.det(cell)))
            slice_volume = (volume / orth_length) * (bin_width * bin_width)
            slice_volumes[index] = slice_volume
        framewise_mass_density_sum = np.zeros_like(mass_histogram_sum)
        framewise_entity_density_sum = np.zeros_like(entity_histogram_sum)
        volume_coverage_count = np.zeros_like(entity_histogram_sum, dtype=np.int64)

    for frame_index, (xy_values, masses) in enumerate(
        zip(selected_xy_per_frame, selected_masses_per_frame)
    ):
        points = np.asarray(xy_values, dtype=float)
        per_frame_mass_histogram, per_frame_entity_histogram = _bincount_2d_histograms(
            x_values=points[:, 0] if points.size > 0 else np.empty(0, dtype=float),
            y_values=points[:, 1] if points.size > 0 else np.empty(0, dtype=float),
            masses=np.asarray(masses, dtype=float),
            x_bin_start=x_bin_start,
            y_bin_start=y_bin_start,
            bin_width=bin_width,
            x_bin_count=x_bin_count,
            y_bin_count=y_bin_count,
            x_bin_end=x_bin_end,
            y_bin_end=y_bin_end,
        )
        _validate_binned_heatmap_frame_conservation(
            frame_index=frame_index,
            mass_histogram=per_frame_mass_histogram,
            entity_histogram=per_frame_entity_histogram,
            masses=np.asarray(masses, dtype=float),
            xy_values=points,
        )
        mass_histogram_sum += per_frame_mass_histogram
        entity_histogram_sum += per_frame_entity_histogram
        if (
            use_volumetric_density
            and framewise_mass_density_sum is not None
            and framewise_entity_density_sum is not None
            and volume_coverage_count is not None
        ):
            volumes = _cartesian_column_volumes(
                frames[frame_index],
                x_bin_edges,
                y_bin_edges,
                (axis_to_index(plane_axes[0]), axis_to_index(plane_axes[1])),
            )
            covered = volumes > 0
            framewise_mass_density_sum[covered] += (
                per_frame_mass_histogram[covered] / volumes[covered]
            )
            framewise_entity_density_sum[covered] += (
                per_frame_entity_histogram[covered] / volumes[covered]
            )
            volume_coverage_count[covered] += 1

    if use_volumetric_density and slice_volumes is not None:
        if (
            volume_coverage_count is not None
            and framewise_mass_density_sum is not None
            and framewise_entity_density_sum is not None
        ):
            density = np.divide(
                framewise_mass_density_sum,
                volume_coverage_count,
                out=np.full_like(framewise_mass_density_sum, np.nan),
                where=volume_coverage_count > 0,
            )
            number_density = np.divide(
                framewise_entity_density_sum,
                volume_coverage_count,
                out=np.full_like(framewise_entity_density_sum, np.nan),
                where=volume_coverage_count > 0,
            )
        elif framewise_mass_density_sum is None or framewise_entity_density_sum is None:
            reference_slice_volume = float(slice_volumes[0])
            density = (mass_histogram_sum / reference_slice_volume) / n_frames
            number_density = (entity_histogram_sum / reference_slice_volume) / n_frames
        else:
            density = framewise_mass_density_sum / n_frames
            number_density = framewise_entity_density_sum / n_frames
        density = density / ANGSTROM3_TO_CM3
        number_density = number_density / ANGSTROM3_TO_NM3
        units = "g/cm^3"
        number_density_units = _entity_density_units_for_species(species_label, volumetric=True)
    else:
        density = (mass_histogram_sum / n_frames) / (bin_width * bin_width)
        number_density = (entity_histogram_sum / n_frames) / (bin_width * bin_width)
        units = "g/Angstrom^2"
        number_density_units = _entity_areal_density_units_for_species(species_label)
    if aggregate_binning_progress is not None:
        aggregate_binning_progress.update()

    x_bin_centers = 0.5 * (x_bin_edges[:-1] + x_bin_edges[1:])
    y_bin_centers = 0.5 * (y_bin_edges[:-1] + y_bin_edges[1:])
    visible_x_min_A, visible_x_max_A, visible_y_min_A, visible_y_max_A = (
        _heatmap_visible_ranges_from_density(
            x_bin_edges=x_bin_edges,
            y_bin_edges=y_bin_edges,
            values=density,
        )
    )
    return DensityHeatmapProfile(
        plane=f"{plane_axes[0]}{plane_axes[1]}",
        plane_axes=plane_axes,
        species=species_label,
        x_bin_edges=x_bin_edges,
        y_bin_edges=y_bin_edges,
        x_bin_centers=x_bin_centers,
        y_bin_centers=y_bin_centers,
        density=np.asarray(density, dtype=float),
        units=units,
        n_frames=n_frames,
        number_density=np.asarray(number_density, dtype=float),
        number_density_units=number_density_units,
        visible_x_min_A=visible_x_min_A,
        visible_x_max_A=visible_x_max_A,
        visible_y_min_A=visible_y_min_A,
        visible_y_max_A=visible_y_max_A,
        oh_cutoff_A=oh_cutoff_A,
        min_molecule_frames=min_molecule_frames,
    )


def _compute_density_profile_from_selected(
    *,
    frames: Sequence[Atoms],
    selected_per_frame: list[np.ndarray],
    selected_masses_per_frame: list[np.ndarray],
    selected_entity_weights_per_frame: list[np.ndarray] | None = None,
    axis: str,
    axis_index: int,
    species_label: str,
    count_label: str,
    bin_width: float,
    coordinate_mode: str = "axis",
    surface_position: float | None = None,
    surface_position_std: float | None = None,
    surface_estimate: SurfaceEstimate | None = None,
    surface_reference: SurfaceReference | None = None,
    interface_view: str = "single",
    histogram_bounds: tuple[float, float] | None = None,
    show_binning_progress: bool = True,
    aggregate_binning_progress: ProgressBar | None = None,
    block_index_by_frame: np.ndarray | None = None,
    oh_cutoff_A: float | None = None,
    min_molecule_frames: int | None = None,
) -> DensityProfile:
    """Build a :class:`DensityProfile` from already-selected axis values."""
    _validate_selected_density_inputs(
        selected_per_frame=selected_per_frame,
        selected_masses_per_frame=selected_masses_per_frame,
    )
    if selected_entity_weights_per_frame is not None:
        if len(selected_entity_weights_per_frame) != len(selected_per_frame):
            raise ValueError("Density entity-weight frame count must match selected coordinates.")
        for values, weights in zip(selected_per_frame, selected_entity_weights_per_frame):
            if np.asarray(weights).shape != np.asarray(values).shape:
                raise ValueError(
                    "Density entity weights must match selected coordinates in every frame."
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
    if bin_edges[-1] < data_max and not np.isclose(bin_edges[-1], data_max):
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    elif histogram_bounds is None and n_bins == 1:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    LOGGER.debug(
        "Histogram bounds: min=%.6g, max=%.6g, bins=%d.",
        data_min,
        data_max,
        bin_edges.size - 1,
    )

    n_frames = len(frames)
    n_bins = int(bin_edges.size - 1)
    bin_start = float(bin_edges[0])
    bin_end = float(bin_edges[-1])
    mass_histogram_sum = np.zeros(n_bins, dtype=float)
    entity_histogram_sum = np.zeros(n_bins, dtype=float)
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

    # Folded/side coordinates live in a canonical periodic image. Interior bins
    # have the ordinary cross-section * width volume; clipping them against the
    # original Cartesian image would incorrectly remove the boundary-crossing side.
    general_geometry = use_volumetric_density and surface_reference is None
    if general_geometry:
        slice_volumes = np.asarray(
            [
                bin_volumes(
                    frame.cell,
                    bin_edges,
                    axis_index,
                    origin=frame_origin(frame),
                    shift=(
                        float(surface_estimate.frame_values[i])
                        if coordinate_mode == "distance" and surface_estimate is not None
                        else 0.0
                    ),
                )
                for i, frame in enumerate(frames)
            ]
        )
        variable_slice_volume = True
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
    if block_index_by_frame is None:
        block_index_by_frame = _resolve_frame_block_index_by_frame(n_frames)
    n_blocks = None if block_index_by_frame is None else int(np.max(block_index_by_frame)) + 1
    density_moments = _empty_density_statistics_moments(n_bins=n_bins, n_blocks=n_blocks)
    number_density_moments = _empty_density_statistics_moments(n_bins=n_bins, n_blocks=n_blocks)
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
            entity_weights = (
                None
                if selected_entity_weights_per_frame is None
                else np.asarray(selected_entity_weights_per_frame[frame_index], dtype=float)
            )
            per_frame_mass_histogram, per_frame_entity_histogram = _bincount_1d_histograms(
                values=np.asarray(axis_values, dtype=float),
                masses=np.asarray(masses, dtype=float),
                bin_start=bin_start,
                bin_width=bin_width,
                n_bins=n_bins,
                bin_end=bin_end,
                entity_weights=entity_weights,
            )
            _validate_binned_frame_conservation(
                frame_index=frame_index,
                mass_histogram=per_frame_mass_histogram,
                entity_histogram=per_frame_entity_histogram,
                masses=np.asarray(masses, dtype=float),
                axis_values=np.asarray(axis_values, dtype=float),
                entity_weights=entity_weights,
            )
            mass_histogram_sum += per_frame_mass_histogram
            entity_histogram_sum += per_frame_entity_histogram
            if use_volumetric_density and slice_volumes is not None:
                frame_volume = np.where(
                    slice_volumes[frame_index] > 0, slice_volumes[frame_index], np.inf
                )
                frame_density = (per_frame_mass_histogram / frame_volume) / ANGSTROM3_TO_CM3
                frame_number_density = (
                    per_frame_entity_histogram / frame_volume
                ) / ANGSTROM3_TO_NM3
            else:
                frame_density = per_frame_mass_histogram / bin_width
                frame_number_density = per_frame_entity_histogram / bin_width
            _update_density_statistics_moments(
                density_moments,
                histogram=per_frame_entity_histogram,
                density_values=np.asarray(frame_density, dtype=float),
                frame_index=frame_index,
                block_index_by_frame=block_index_by_frame,
            )
            _update_density_statistics_moments(
                number_density_moments,
                histogram=per_frame_entity_histogram,
                density_values=np.asarray(frame_number_density, dtype=float),
                frame_index=frame_index,
                block_index_by_frame=block_index_by_frame,
            )
            if (
                framewise_mass_density_sum is not None
                and framewise_entity_density_sum is not None
                and slice_volumes is not None
            ):
                frame_volume = np.where(
                    slice_volumes[frame_index] > 0, slice_volumes[frame_index], np.inf
                )
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
            denominator = np.sum(slice_volumes > 0, axis=0) if general_geometry else n_frames
            denominator = np.where(np.asarray(denominator) > 0, denominator, np.nan)
            density = framewise_mass_density_sum / denominator
            number_density = framewise_entity_density_sum / denominator
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
    density_statistics = _finalize_density_statistics_moments(density_moments)
    number_density_statistics = _finalize_density_statistics_moments(number_density_moments)

    visible_x_min_A, visible_x_max_A = _line_visible_range_from_density(
        bin_edges=bin_edges,
        values=density,
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
        surface_reference=surface_reference,
        interface_view=interface_view,
        series_statistics={
            "density": density_statistics,
            "number_density": number_density_statistics,
        },
        visible_x_min_A=visible_x_min_A,
        visible_x_max_A=visible_x_max_A,
        oh_cutoff_A=oh_cutoff_A,
        min_molecule_frames=min_molecule_frames,
    )


def _summarize_density_run(
    *,
    frames: Sequence[Atoms],
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


def _density_uses_variable_slice_volume(frames: Sequence[Atoms], axis_index: int) -> bool:
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
    frames: Sequence[Atoms],
    species: str | None = "all",
    axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_layout: str | None = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "observed",
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | SurfaceReference | None = None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
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
    if frames and needs_general_geometry(frames):
        frames = wrap_trajectory_frames(frames)
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("bin_width", bin_width)
    ensure_positive("oh_cutoff", oh_cutoff)
    if int(min_molecule_frames) < 1:
        raise ValueError("min_molecule_frames must be >= 1.")
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    axis_index = axis_to_index(axis)
    surface_estimate: SurfaceEstimate | None
    surface_reference: SurfaceReference | None = None
    if str(surface_mode).strip().lower() == "none":
        surface_estimate = None
    elif isinstance(precomputed_surface_estimate, SurfaceReference):
        surface_reference = precomputed_surface_estimate
        surface_estimate = surface_reference.primary
    elif precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (len(frames),):
            raise ValueError(
                "precomputed_surface_estimate frame count does not match the trajectory."
            )
        surface_estimate = precomputed_surface_estimate
    elif surface_layout is None:
        surface_estimate, _surface_method = _select_surface_estimate(
            frames,
            axis,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
    else:
        surface_reference = estimate_surface_layout_reference(
            frames,
            axis,
            surface_layout=surface_layout,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
        surface_estimate = surface_reference.primary
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    if (
        surface_reference is not None
        and surface_reference.uses_distance
        and not surface_reference.is_symmetric
        and surface_estimate is not None
    ):
        wrapped_surface = wrapped_surface_positions(surface_reference, surface_estimate)
        surface_position = float(np.median(wrapped_surface))
        surface_position_std = float(np.std(wrapped_surface, ddof=0))
    _surface_skipped = str(surface_mode).strip().lower() == "none"
    if surface_reference is None and surface_position is None and not _surface_skipped:
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            axis.lower(),
            axis.lower(),
        )
    elif surface_reference is None and surface_position is not None:
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_position,
            surface_position_std=surface_position_std,
        )
    selection_mode, species_label = _normalize_species_query(
        species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if selection_mode in {"elements", "molecules"}:
        raise ValueError(
            f"Density selector '{species}' expands to multiple profiles; use compute_density_profiles "
            "or compute_all_density_profiles."
        )
    count_label = "molecules" if selection_mode == "molecule" else "atoms"
    LOGGER.debug("Selection mode: %s (label=%s).", selection_mode, species_label)

    selected_per_frame: list[np.ndarray] = []
    selected_masses_per_frame: list[np.ndarray] = []
    if selection_mode == "molecule":
        selected_per_frame, selected_masses_per_frame = _select_molecule_axis_values_per_frame(
            frames,
            axis_index,
            species_label,
            oh_cutoff=oh_cutoff,
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
        if surface_reference is None
        and _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames))
        else None
    )
    selected_top: list[np.ndarray] | None = None
    masses_top: list[np.ndarray] | None = None
    entity_weights_top: list[np.ndarray] | None = None
    selected_bottom: list[np.ndarray] | None = None
    masses_bottom: list[np.ndarray] | None = None
    entity_weights_bottom: list[np.ndarray] | None = None
    if surface_reference is not None and surface_reference.uses_distance:
        selected_for_binning = []
        selected_top = []
        masses_top = []
        entity_weights_top = []
        selected_bottom = []
        masses_bottom = []
        entity_weights_bottom = []
        for frame_index, (values, masses) in enumerate(
            zip(selected_per_frame, selected_masses_per_frame)
        ):
            folded, top_distance, bottom_distance, side = surface_distance_components(
                surface_reference,
                np.asarray(values, dtype=float),
                frame_index=frame_index,
            )
            selected_for_binning.append(folded)
            top_mask = side >= 0
            bottom_mask = side <= 0
            top_weights = np.where(side[top_mask] == 0, 0.5, 1.0)
            bottom_weights = np.where(side[bottom_mask] == 0, 0.5, 1.0)
            selected_top.append(top_distance[top_mask])
            masses_top.append(np.asarray(masses, dtype=float)[top_mask] * top_weights)
            entity_weights_top.append(top_weights)
            selected_bottom.append(bottom_distance[bottom_mask])
            masses_bottom.append(np.asarray(masses, dtype=float)[bottom_mask] * bottom_weights)
            entity_weights_bottom.append(bottom_weights)
        coordinate_mode = "distance"
    else:
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
    LOGGER.debug(
        "Density selections: %s.",
        _density_selection_summary(
            [
                _DensityTargetSpec(
                    species_label=species_label,
                    count_label=count_label,
                    selection_kind=(
                        "molecule"
                        if selection_mode == "molecule"
                        else ("species" if selection_mode == "species" else "element")
                    ),
                )
            ]
        ),
    )
    if selection_mode == "molecule":
        LOGGER.debug("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
    histogram_bounds = None
    if normalized_binning == "cell":
        if surface_reference is not None and surface_reference.uses_distance:
            histogram_bounds = _surface_reference_histogram_bounds(surface_reference)
        else:
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
    if surface_reference is None and surface_position is not None and coordinate_mode != "distance":
        LOGGER.warning(
            "Surface was estimated, but frame-wise surface alignment was unavailable; "
            "density bins remain on raw %s coordinates.",
            axis.lower(),
        )
    profile = _compute_density_profile_from_selected(
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
        surface_reference=surface_reference,
        interface_view=(
            "folded"
            if surface_reference is not None and surface_reference.is_symmetric
            else "single"
        ),
        histogram_bounds=histogram_bounds,
        oh_cutoff_A=oh_cutoff if selection_mode == "molecule" else None,
        min_molecule_frames=min_molecule_frames if selection_mode == "molecule" else None,
    )
    if (
        surface_reference is None
        or not surface_reference.is_symmetric
        or selected_top is None
        or masses_top is None
        or entity_weights_top is None
        or selected_bottom is None
        or masses_bottom is None
        or entity_weights_bottom is None
    ):
        return profile

    common_bounds = (float(profile.bin_edges[0]), float(profile.bin_edges[-1]))
    top_profile = _compute_density_profile_from_selected(
        frames=frames,
        selected_per_frame=selected_top,
        selected_masses_per_frame=masses_top,
        selected_entity_weights_per_frame=entity_weights_top,
        axis=axis,
        axis_index=axis_index,
        species_label=species_label,
        count_label=count_label,
        bin_width=bin_width,
        coordinate_mode="distance",
        surface_estimate=surface_reference.top,
        surface_reference=surface_reference,
        interface_view="top",
        histogram_bounds=common_bounds,
        show_binning_progress=False,
        oh_cutoff_A=oh_cutoff if selection_mode == "molecule" else None,
        min_molecule_frames=min_molecule_frames if selection_mode == "molecule" else None,
    )
    bottom_profile = _compute_density_profile_from_selected(
        frames=frames,
        selected_per_frame=selected_bottom,
        selected_masses_per_frame=masses_bottom,
        selected_entity_weights_per_frame=entity_weights_bottom,
        axis=axis,
        axis_index=axis_index,
        species_label=species_label,
        count_label=count_label,
        bin_width=bin_width,
        coordinate_mode="distance",
        surface_estimate=surface_reference.bottom,
        surface_reference=surface_reference,
        interface_view="bottom",
        histogram_bounds=common_bounds,
        show_binning_progress=False,
        oh_cutoff_A=oh_cutoff if selection_mode == "molecule" else None,
        min_molecule_frames=min_molecule_frames if selection_mode == "molecule" else None,
    )
    if not np.array_equal(top_profile.bin_edges, bottom_profile.bin_edges):
        raise ValueError("Top and bottom symmetric density grids do not match.")
    return replace(
        profile,
        bin_edges=top_profile.bin_edges,
        bin_centers=top_profile.bin_centers,
        density=0.5 * (top_profile.density + bottom_profile.density),
        number_density=(
            None
            if top_profile.number_density is None or bottom_profile.number_density is None
            else 0.5 * (top_profile.number_density + bottom_profile.number_density)
        ),
        counts_per_frame=0.5 * (top_profile.counts_per_frame + bottom_profile.counts_per_frame),
        entities_per_frame=(
            None
            if top_profile.entities_per_frame is None or bottom_profile.entities_per_frame is None
            else 0.5 * (top_profile.entities_per_frame + bottom_profile.entities_per_frame)
        ),
        top_density=top_profile.density,
        bottom_density=bottom_profile.density,
        top_number_density=top_profile.number_density,
        bottom_number_density=bottom_profile.number_density,
        top_counts_per_frame=top_profile.counts_per_frame,
        bottom_counts_per_frame=bottom_profile.counts_per_frame,
        series_statistics=None,
    )


def compute_density_profiles(
    frames: Sequence[Atoms],
    species: str | None = "all",
    axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_layout: str | None = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "observed",
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | SurfaceReference | None = None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
) -> list[DensityProfile]:
    """Compute one or more density profiles based on the species selection policy."""
    ensure_positive("bin_width", bin_width)
    ensure_positive("oh_cutoff", oh_cutoff)
    if int(min_molecule_frames) < 1:
        raise ValueError("min_molecule_frames must be >= 1.")
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    selection_mode, _ = _normalize_species_query(
        species,
        allow_h2o=True,
        allow_molecules=True,
    )
    if surface_layout is not None:
        if frames and needs_general_geometry(frames):
            frames = wrap_trajectory_frames(frames)
        paired_reference = (
            precomputed_surface_estimate
            if isinstance(precomputed_surface_estimate, SurfaceReference)
            else estimate_surface_layout_reference(
                frames,
                axis,
                surface_layout=surface_layout,
                mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                surface_options=surface_options,
                logger=LOGGER,
            )
        )
        if paired_reference.is_symmetric:
            targets = _resolve_density_target_specs(frames, species=species)
            paired_profiles: list[DensityProfile] = []
            for target in targets:
                try:
                    paired_profiles.append(
                        compute_density_profile(
                            frames=frames,
                            species=target.species_label,
                            axis=axis,
                            bin_width=bin_width,
                            surface_mode=surface_mode,
                            surface_layout=surface_layout,
                            surface_elements=surface_elements,
                            include_fixed_surface_atoms=include_fixed_surface_atoms,
                            binning=normalized_binning,
                            surface_options=surface_options,
                            precomputed_surface_estimate=paired_reference,
                            oh_cutoff=oh_cutoff,
                            min_molecule_frames=min_molecule_frames,
                        )
                    )
                except ValueError as exc:
                    if not str(exc).startswith("No entities found for selection"):
                        raise
            return paired_profiles
        # One-sided and raw-axis layouts can use the optimized shared-topology
        # path, but retain the paired reference so canonical periodic distance
        # handling is not lost.
        precomputed_surface_estimate = paired_reference
        surface_layout = None
    if selection_mode not in {"all", "elements", "molecules"}:
        return [
            compute_density_profile(
                frames=frames,
                species=species,
                axis=axis,
                bin_width=bin_width,
                surface_mode=surface_mode,
                surface_layout=surface_layout,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                binning=normalized_binning,
                surface_options=surface_options,
                precomputed_surface_estimate=precomputed_surface_estimate,
                oh_cutoff=oh_cutoff,
                min_molecule_frames=min_molecule_frames,
            )
        ]

    if frames and needs_general_geometry(frames):
        frames = wrap_trajectory_frames(frames)
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    surface_estimate: SurfaceEstimate | None
    surface_reference: SurfaceReference | None = None
    if str(surface_mode).strip().lower() == "none":
        surface_estimate = None
    elif isinstance(precomputed_surface_estimate, SurfaceReference):
        surface_reference = precomputed_surface_estimate
        surface_estimate = surface_reference.primary
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
    if surface_reference is None and surface_position is None and not _surface_skipped:
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            axis.lower(),
            axis.lower(),
        )
    elif surface_reference is None and surface_position is not None:
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_position,
            surface_position_std=surface_position_std,
        )
    available_elements = available_element_species(frames)
    if selection_mode in {"all", "elements"} and not available_elements:
        raise ValueError("No elements found in trajectory.")
    element_species = available_elements if selection_mode in {"all", "elements"} else []

    axis_index = axis_to_index(axis)
    selected_by_species: dict[str, list[np.ndarray]] = {element: [] for element in element_species}
    selected_masses_by_species: dict[str, list[np.ndarray]] = {
        element: [] for element in element_species
    }
    empty = np.array([], dtype=float)
    if element_species:
        element_selector = _DensityElementSelector(element_species)
        with ProgressBar(
            desc="Selecting element data for density", total=len(frames), unit="frame"
        ) as progress:
            for frame_index, frame in enumerate(frames):
                frame_selected = element_selector.select(frame, frame_index=frame_index)
                for element in element_species:
                    selected_values, selected_masses = frame_selected.get(
                        element,
                        (empty, empty),
                    )
                    axis_selected = (
                        np.asarray(selected_values[:, axis_index], dtype=float)
                        if np.asarray(selected_values).size > 0
                        else empty
                    )
                    selected_by_species[element].append(axis_selected)
                    selected_masses_by_species[element].append(selected_masses)
                progress.update()

    profiles: list[DensityProfile] = []
    trusted_surface_estimate = (
        surface_estimate
        if surface_reference is None
        and _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames))
        else None
    )
    per_frame_surface = (
        None if trusted_surface_estimate is None else trusted_surface_estimate.per_frame
    )
    coordinate_mode_global = (
        "distance"
        if (
            trusted_surface_estimate is not None
            or (surface_reference is not None and surface_reference.uses_distance)
        )
        else "axis"
    )
    histogram_bounds = None
    if normalized_binning == "cell":
        if surface_reference is not None and surface_reference.uses_distance:
            histogram_bounds = _surface_reference_histogram_bounds(surface_reference)
        else:
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
    molecule_selections: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    if selection_mode in {"all", "molecules"}:
        molecule_selection_candidates = _select_molecule_axis_values_for_labels_per_frame(
            frames,
            axis_index,
            MOLECULE_SPECIES_LABELS,
            oh_cutoff=oh_cutoff,
            progress_desc="Selecting O/H molecules for density",
        )
        for molecule_label, (
            selected_values,
            selected_masses,
        ) in molecule_selection_candidates.items():
            active_frame_count = sum(1 for values in selected_values if values.size > 0)
            max_consecutive_frame_count = _max_consecutive_truthy(
                values.size > 0 for values in selected_values
            )
            if _molecule_target_is_active(
                species_label=molecule_label,
                active_frame_count=active_frame_count,
                consecutive_frame_count=max_consecutive_frame_count,
                apply_min_molecule_frames=True,
                min_molecule_frames=min_molecule_frames,
            ):
                molecule_selections[molecule_label] = (selected_values, selected_masses)
    total_binning_jobs = len(element_species)
    total_binning_jobs += len(molecule_selections)
    if total_binning_jobs == 0:
        raise ValueError("No entities found for the requested density selection.")
    active_targets_for_log = [
        *(
            _DensityTargetSpec(
                species_label=element,
                count_label="atoms",
                selection_kind="element",
            )
            for element in element_species
        ),
        *(
            _DensityTargetSpec(
                species_label=molecule_label,
                count_label="molecules",
                selection_kind="molecule",
                apply_min_molecule_frames=True,
            )
            for molecule_label in molecule_selections
        ),
    ]
    LOGGER.debug("Density selections: %s.", _density_selection_summary(active_targets_for_log))
    if molecule_selections:
        LOGGER.debug("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
        non_water_molecules = [
            _density_species_display_label(label)
            for label in molecule_selections
            if label != "mol:H2O"
        ]
        if non_water_molecules:
            LOGGER.debug(
                "Detected O/H molecule species: %s.",
                ",".join(non_water_molecules),
            )
    _density_mode, run_summary = _summarize_density_run(
        frames=frames,
        axis_index=axis_index,
        coordinate_mode=coordinate_mode_global,
        labels=[*element_species, *molecule_selections],
    )
    LOGGER.info("Density compute summary: %s.", run_summary)
    LOGGER.debug("Binning %d density profiles.", total_binning_jobs)
    with ProgressBar(
        desc="Binning density profiles", total=total_binning_jobs, unit="profile"
    ) as progress:

        def aligned_values(values_per_frame: list[np.ndarray]) -> tuple[list[np.ndarray], str]:
            if surface_reference is None or not surface_reference.uses_distance:
                return _shift_axis_values_by_surface_per_frame(
                    values_per_frame,
                    per_frame_surface,
                )
            return (
                [
                    surface_distance_components(
                        surface_reference,
                        np.asarray(values, dtype=float),
                        frame_index=frame_index,
                    )[0]
                    for frame_index, values in enumerate(values_per_frame)
                ],
                "distance",
            )

        for element in element_species:
            selected_for_binning, coordinate_mode = aligned_values(selected_by_species[element])
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
                    surface_reference=surface_reference,
                    interface_view="single",
                    histogram_bounds=histogram_bounds,
                    show_binning_progress=False,
                    aggregate_binning_progress=progress,
                    oh_cutoff_A=None,
                    min_molecule_frames=None,
                )
            )

        for molecule_label, (
            molecule_selected_per_frame,
            molecule_masses_per_frame,
        ) in molecule_selections.items():
            selected_for_binning, coordinate_mode = aligned_values(molecule_selected_per_frame)
            profile_surface_position = surface_position
            profile_surface_position_std = surface_position_std
            profiles.append(
                _compute_density_profile_from_selected(
                    frames=frames,
                    selected_per_frame=selected_for_binning,
                    selected_masses_per_frame=molecule_masses_per_frame,
                    axis=axis,
                    axis_index=axis_index,
                    species_label=molecule_label,
                    count_label="molecules",
                    bin_width=bin_width,
                    coordinate_mode=coordinate_mode,
                    surface_position=profile_surface_position,
                    surface_position_std=profile_surface_position_std,
                    surface_estimate=surface_estimate,
                    surface_reference=surface_reference,
                    interface_view="single",
                    histogram_bounds=histogram_bounds,
                    show_binning_progress=False,
                    aggregate_binning_progress=progress,
                    oh_cutoff_A=oh_cutoff,
                    min_molecule_frames=min_molecule_frames,
                )
            )
    return profiles


def _compute_all_density_profiles_streaming(
    *,
    frames: Sequence[Atoms],
    species: str | None,
    surface_axis: str,
    bin_width: float,
    surface_mode: str,
    surface_layout: str | None,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    binning: str,
    surface_options: SurfaceEstimatorOptions | None,
    precomputed_surface_estimate: SurfaceEstimate | SurfaceReference | None,
    outputs: str | None,
    heatmap_planes: list[str] | tuple[str, ...] | None,
    grid_bin_width: float | None = None,
    grid_max_nonzero_bins: int = DEFAULT_DENSITY_GRID_MAX_NONZERO_BINS,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
    oh_topology_backend: str = "auto",
) -> list[DensityProfile | DensityHeatmapProfile | DensityGridProfile]:
    """Shared streaming density engine for raw-axis, distance, and heatmap outputs."""

    ensure_positive("bin_width", bin_width)
    resolved_grid_bin_width = float(bin_width if grid_bin_width is None else grid_bin_width)
    ensure_positive("grid_bin_width", resolved_grid_bin_width)
    if int(grid_max_nonzero_bins) < 1:
        raise ValueError("grid_max_nonzero_bins must be >= 1.")
    ensure_positive("oh_cutoff", oh_cutoff)
    if int(min_molecule_frames) < 1:
        raise ValueError("min_molecule_frames must be >= 1.")
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    normalized_outputs = _normalize_density_outputs(outputs)
    if heatmap_planes is not None:
        raise ValueError(
            "--heatmap-planes is no longer used for density compute. "
            "Use --outputs 3d to write a sparse density grid for GUI slicing."
        )
    include_line_outputs = normalized_outputs in {"1d", "all"}
    include_grid_outputs = normalized_outputs in {"3d", "all"}
    include_heatmap_outputs = False
    selected_heatmap_planes = (
        _normalize_density_heatmap_planes(heatmap_planes) if include_heatmap_outputs else tuple()
    )
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    raw_axes = ("x", "y", "z")
    axis_indices = {ax: axis_to_index(ax) for ax in raw_axes}
    surface_axis_index = axis_to_index(surface_axis)

    surface_estimate: SurfaceEstimate | None
    surface_reference: SurfaceReference | None = None
    if str(surface_mode).strip().lower() == "none":
        surface_estimate = None
    elif isinstance(precomputed_surface_estimate, SurfaceReference):
        surface_reference = precomputed_surface_estimate
        surface_estimate = surface_reference.primary
    elif precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (len(frames),):
            raise ValueError(
                "precomputed_surface_estimate frame count does not match the trajectory."
            )
        surface_estimate = precomputed_surface_estimate
    elif surface_layout is None:
        surface_estimate, _surface_method = _select_surface_estimate(
            frames,
            surface_axis,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
    else:
        surface_reference = estimate_surface_layout_reference(
            frames,
            surface_axis,
            surface_layout=surface_layout,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            logger=LOGGER,
        )
        surface_estimate = surface_reference.primary
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    if (
        surface_reference is None
        and surface_position is None
        and str(surface_mode).strip().lower() != "none"
    ):
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            surface_axis.lower(),
            surface_axis.lower(),
        )
    elif surface_reference is None and surface_position is not None:
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
    distance_coordinate_mode = (
        "distance"
        if (surface_reference is not None and surface_reference.uses_distance)
        or per_frame_surface is not None
        else "axis"
    )

    # Computed once: scanning all frames per call would make the binning loops quadratic.
    wrap_positions_to_cell = normalized_binning == "cell" or needs_general_geometry(frames)

    target_specs = _resolve_density_target_specs(frames, species=species)

    line_jobs_per_species = 4 if include_line_outputs else 0
    grid_jobs_per_species = 1 if include_grid_outputs else 0
    heatmap_jobs_per_species = len(selected_heatmap_planes)

    block_index_by_frame = _resolve_frame_block_index_by_frame(len(frames))
    validation_plan = _build_density_validation_plan(len(frames))
    n_blocks = None if block_index_by_frame is None else int(np.max(block_index_by_frame)) + 1

    cell_bounds_by_axis: dict[str, tuple[float, float] | None] = {}
    if normalized_binning == "cell" and (
        include_line_outputs or include_heatmap_outputs or include_grid_outputs
    ):
        for ax in raw_axes:
            cell_bounds_by_axis[ax] = _cell_histogram_bounds(
                frames=frames,
                axis_index=axis_indices[ax],
                coordinate_mode="axis",
                surface_per_frame=None,
            )
    if not (include_line_outputs or include_grid_outputs) or normalized_binning != "cell":
        distance_hist_bounds = None
    elif surface_reference is not None and surface_reference.uses_distance:
        distance_hist_bounds = _surface_reference_histogram_bounds(surface_reference)
    else:
        distance_hist_bounds = _cell_histogram_bounds(
            frames=frames,
            axis_index=surface_axis_index,
            coordinate_mode=distance_coordinate_mode,
            surface_per_frame=per_frame_surface if distance_coordinate_mode == "distance" else None,
        )
    line_normalization_by_axis = (
        {
            ax: _build_line_normalization_cache(
                frames=frames,
                axis_index=axis_indices[ax],
                bin_width=bin_width,
            )
            for ax in raw_axes
        }
        if include_line_outputs
        else {}
    )
    heatmap_normalization_by_plane = (
        {
            plane_axes: _build_heatmap_normalization_cache(
                frames=frames,
                orth_axis_index=axis_indices[
                    next(axis for axis in raw_axes if axis not in plane_axes)
                ],
                plane_axis_indices=(axis_indices[plane_axes[0]], axis_indices[plane_axes[1]]),
                bin_width=bin_width,
            )
            for plane_axes in selected_heatmap_planes
        }
        if include_heatmap_outputs
        else {}
    )

    needs_observed_bounds_prepass = _density_needs_observed_bounds_prepass(
        normalized_binning=normalized_binning,
        include_line_outputs=include_line_outputs,
        include_grid_outputs=include_grid_outputs,
        selected_heatmap_planes=selected_heatmap_planes,
        raw_axes=raw_axes,
        cell_bounds_by_axis=cell_bounds_by_axis,
        distance_hist_bounds=distance_hist_bounds,
    )
    active_targets_without_prepass = (
        None
        if needs_observed_bounds_prepass
        else _resolve_density_active_targets_without_bounds_prepass(frames, target_specs)
    )
    if active_targets_without_prepass is not None:
        if include_line_outputs and distance_hist_bounds is None:
            raise ValueError("Missing distance cell bounds for density bin preparation.")
        active_targets = active_targets_without_prepass
        observed_bounds_by_target = _density_observed_bounds_from_cell_bounds(
            active_targets=active_targets,
            raw_axes=raw_axes,
            cell_bounds_by_axis=cell_bounds_by_axis,
            distance_hist_bounds=distance_hist_bounds,
        )
        LOGGER.debug("Density bin preparation uses cell bounds; skipped observed coordinate scan.")
    else:
        observed_bounds_by_target = {
            target.species_label: _DensityObservedBounds(axis_lower={}, axis_upper={})
            for target in target_specs
        }
        run_topology_cache = OHTopologyCache(
            oh_cutoff=oh_cutoff,
            context="density O/H molecule topology",
            backend=oh_topology_backend,
        )
        element_selector = _DensityElementSelector(
            [target.species_label for target in target_specs if target.selection_kind == "element"]
        )
        with ProgressBar(
            desc="Preparing density bins", total=len(frames), unit="frame"
        ) as progress:
            for frame_index, frame in enumerate(frames):
                frame_selections = _select_density_targets_in_frame(
                    frame,
                    frame_index=frame_index,
                    target_specs=target_specs,
                    topology_cache=run_topology_cache,
                    element_selector=element_selector,
                    wrap_positions_to_cell=wrap_positions_to_cell,
                )
                for target in target_specs:
                    selection = frame_selections.get(target.species_label)
                    if selection is None:
                        continue
                    positions, _masses = selection
                    bounds = observed_bounds_by_target[target.species_label]
                    bounds.mark_frame_active(frame_index)
                    for ax in raw_axes:
                        bounds.update_axis(ax, positions[:, axis_indices[ax]])
                    raw_distance_axis = np.asarray(positions[:, surface_axis_index], dtype=float)
                    if surface_reference is not None and surface_reference.uses_distance:
                        distance_values = surface_distance_components(
                            surface_reference,
                            raw_distance_axis,
                            frame_index=frame_index,
                        )[0]
                    else:
                        distance_values = (
                            raw_distance_axis
                            if per_frame_surface is None
                            else raw_distance_axis - float(per_frame_surface[frame_index])
                        )
                    bounds.update_axis("distance", distance_values)
                progress.update()

        active_targets = []
        for target in target_specs:
            bounds = observed_bounds_by_target[target.species_label]
            if not bounds.has_axis_data(surface_axis):
                continue
            if target.selection_kind == "molecule" and not _molecule_target_is_active(
                species_label=target.species_label,
                active_frame_count=bounds.active_frame_count,
                consecutive_frame_count=bounds.max_consecutive_active_frames,
                apply_min_molecule_frames=target.apply_min_molecule_frames,
                min_molecule_frames=min_molecule_frames,
            ):
                continue
            active_targets.append(target)
    if "run_topology_cache" not in locals():
        run_topology_cache = OHTopologyCache(
            oh_cutoff=oh_cutoff,
            context="density O/H molecule topology",
            backend=oh_topology_backend,
        )
    if not active_targets:
        raise ValueError("No entities found for the requested density selection.")

    active_element_count = sum(1 for target in active_targets if target.selection_kind == "element")
    active_raw_species_count = sum(
        1 for target in active_targets if target.selection_kind == "species"
    )
    molecule_active_count = sum(
        1 for target in active_targets if target.selection_kind == "molecule"
    )
    _density_mode, run_summary = _summarize_density_run(
        frames=frames,
        axis_index=surface_axis_index,
        coordinate_mode=distance_coordinate_mode,
        labels=[target.species_label for target in active_targets],
    )
    LOGGER.info("Density compute summary: %s.", run_summary)
    LOGGER.debug("Density selections: %s.", _density_selection_summary(active_targets))
    if molecule_active_count:
        LOGGER.debug("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
        non_water_molecules = [
            _density_species_display_label(target.species_label)
            for target in active_targets
            if target.selection_kind == "molecule" and target.species_label != "mol:H2O"
        ]
        if non_water_molecules:
            LOGGER.debug(
                "Detected O/H molecule species: %s.",
                ",".join(non_water_molecules),
            )
    LOGGER.debug(
        "Single-pass selection complete: %d element selections + %d raw species selections + %d O/H molecule selections, %d frames.",
        active_element_count,
        active_raw_species_count,
        molecule_active_count,
        len(frames),
    )
    LOGGER.debug(
        "Density selection prepass complete: %d active species selection(s), %d frames.",
        len(active_targets),
        len(frames),
    )
    total_binning_jobs = len(active_targets) * (
        line_jobs_per_species + grid_jobs_per_species + heatmap_jobs_per_species
    )
    LOGGER.debug(
        "Binning %d density outputs (%d 1D profiles + %d sparse grid profiles per species selection).",
        total_binning_jobs,
        line_jobs_per_species,
        grid_jobs_per_species,
    )

    line_accumulators_by_target: dict[str, list[tuple[str, _DensityLineAccumulator]]] = {}
    heatmap_accumulators_by_target: dict[
        str, list[tuple[tuple[str, str], _DensityHeatmapAccumulator]]
    ] = {}
    grid_accumulator_by_target: dict[str, _DensitySparseGridAccumulator] = {}
    for target in active_targets:
        species_lbl = target.species_label
        observed_bounds = observed_bounds_by_target[species_lbl]
        target_line_accumulators: list[tuple[str, _DensityLineAccumulator]] = []
        if include_line_outputs:
            for ax in raw_axes:
                hist_bounds = cell_bounds_by_axis.get(ax) if normalized_binning == "cell" else None
                if normalized_binning == "cell" and hist_bounds is None:
                    LOGGER.warning(
                        "Cell binning requested for '%s' along %s, but a usable cell was unavailable. "
                        "Falling back to observed-data binning.",
                        species_lbl,
                        ax,
                    )
                bin_edges = _resolve_density_line_bin_edges_from_bounds(
                    observed_bounds=observed_bounds,
                    axis_id=ax,
                    species_label=species_lbl,
                    bin_width=bin_width,
                    histogram_bounds=hist_bounds,
                )
                target_line_accumulators.append(
                    (
                        ax,
                        _DensityLineAccumulator(
                            axis=ax,
                            species_label=species_lbl,
                            count_label=target.count_label,
                            bin_edges=bin_edges,
                            bin_width=bin_width,
                            coordinate_mode="axis",
                            normalization_cache=line_normalization_by_axis[ax],
                            density_moments=_empty_density_statistics_moments(
                                n_bins=bin_edges.size - 1,
                                n_blocks=n_blocks,
                            ),
                            number_density_moments=_empty_density_statistics_moments(
                                n_bins=bin_edges.size - 1,
                                n_blocks=n_blocks,
                            ),
                            oh_cutoff_A=oh_cutoff if target.selection_kind == "molecule" else None,
                            min_molecule_frames=(
                                min_molecule_frames if target.selection_kind == "molecule" else None
                            ),
                        ),
                    )
                )
            if normalized_binning == "cell" and distance_hist_bounds is None:
                LOGGER.warning(
                    "Cell binning requested for '%s' distance along %s, but a usable cell was unavailable. "
                    "Falling back to observed-data binning.",
                    species_lbl,
                    surface_axis,
                )
            distance_bin_edges = _resolve_density_line_bin_edges_from_bounds(
                observed_bounds=observed_bounds,
                axis_id="distance",
                species_label=species_lbl,
                bin_width=bin_width,
                histogram_bounds=distance_hist_bounds,
            )
            distance_views: tuple[tuple[str, SurfaceEstimate | None, str], ...]
            if (
                surface_reference is not None
                and surface_reference.uses_distance
                and surface_reference.is_symmetric
            ):
                distance_views = (
                    ("distance-top", surface_reference.top, "top"),
                    ("distance-bottom", surface_reference.bottom, "bottom"),
                )
            else:
                distance_views = (("distance", surface_estimate, "single"),)
            for distance_key, distance_surface, interface_view in distance_views:
                target_line_accumulators.append(
                    (
                        distance_key,
                        _DensityLineAccumulator(
                            axis=surface_axis,
                            species_label=species_lbl,
                            count_label=target.count_label,
                            bin_edges=distance_bin_edges,
                            bin_width=bin_width,
                            coordinate_mode=distance_coordinate_mode,
                            normalization_cache=line_normalization_by_axis[surface_axis],
                            density_moments=_empty_density_statistics_moments(
                                n_bins=distance_bin_edges.size - 1,
                                n_blocks=n_blocks,
                            ),
                            number_density_moments=_empty_density_statistics_moments(
                                n_bins=distance_bin_edges.size - 1,
                                n_blocks=n_blocks,
                            ),
                            surface_position=surface_position,
                            surface_position_std=surface_position_std,
                            surface_estimate=distance_surface,
                            surface_reference=surface_reference,
                            interface_view=interface_view,
                            oh_cutoff_A=(
                                oh_cutoff if target.selection_kind == "molecule" else None
                            ),
                            min_molecule_frames=(
                                min_molecule_frames if target.selection_kind == "molecule" else None
                            ),
                        ),
                    )
                )
        line_accumulators_by_target[species_lbl] = target_line_accumulators

        target_heatmap_accumulators: list[tuple[tuple[str, str], _DensityHeatmapAccumulator]] = []
        for plane_axes in selected_heatmap_planes:
            x_bin_edges, y_bin_edges = _resolve_density_heatmap_bin_edges_from_bounds(
                observed_bounds=observed_bounds,
                plane_axes=plane_axes,
                species_label=species_lbl,
                bin_width=bin_width,
                histogram_bounds=(
                    (
                        cell_bounds_by_axis.get(plane_axes[0]),
                        cell_bounds_by_axis.get(plane_axes[1]),
                    )
                    if normalized_binning == "cell"
                    else None
                ),
            )
            orthogonal_axis = next(axis for axis in raw_axes if axis not in plane_axes)
            target_heatmap_accumulators.append(
                (
                    plane_axes,
                    _DensityHeatmapAccumulator(
                        plane_axes=plane_axes,
                        orthogonal_axis=orthogonal_axis,
                        species_label=species_lbl,
                        x_bin_edges=x_bin_edges,
                        y_bin_edges=y_bin_edges,
                        bin_width=bin_width,
                        normalization_cache=heatmap_normalization_by_plane[plane_axes],
                        oh_cutoff_A=oh_cutoff if target.selection_kind == "molecule" else None,
                        min_molecule_frames=(
                            min_molecule_frames if target.selection_kind == "molecule" else None
                        ),
                    ),
                )
            )
        heatmap_accumulators_by_target[species_lbl] = target_heatmap_accumulators

        if include_grid_outputs:
            grid_histogram_bounds = {
                ax: cell_bounds_by_axis.get(ax) if normalized_binning == "cell" else None
                for ax in raw_axes
            }
            distance_grid_bounds = distance_hist_bounds if normalized_binning == "cell" else None
            x_edges = _resolve_density_grid_bin_edges_from_bounds(
                observed_bounds=observed_bounds,
                axis_id="x",
                species_label=species_lbl,
                grid_bin_width=resolved_grid_bin_width,
                histogram_bounds=grid_histogram_bounds["x"],
            )
            y_edges = _resolve_density_grid_bin_edges_from_bounds(
                observed_bounds=observed_bounds,
                axis_id="y",
                species_label=species_lbl,
                grid_bin_width=resolved_grid_bin_width,
                histogram_bounds=grid_histogram_bounds["y"],
            )
            z_edges = _resolve_density_grid_bin_edges_from_bounds(
                observed_bounds=observed_bounds,
                axis_id="z",
                species_label=species_lbl,
                grid_bin_width=resolved_grid_bin_width,
                histogram_bounds=grid_histogram_bounds["z"],
            )
            distance_edges = _resolve_density_grid_bin_edges_from_bounds(
                observed_bounds=observed_bounds,
                axis_id="distance",
                species_label=species_lbl,
                grid_bin_width=resolved_grid_bin_width,
                histogram_bounds=distance_grid_bounds,
            )
            grid_accumulator_by_target[species_lbl] = _DensitySparseGridAccumulator(
                geometry_frames=frames
                if all(_frame_has_usable_cell(f, surface_axis_index) for f in frames)
                else None,
                species_label=species_lbl,
                count_label=target.count_label,
                x_bin_edges=x_edges,
                y_bin_edges=y_edges,
                z_bin_edges=z_edges,
                distance_bin_edges=distance_edges,
                grid_bin_width=resolved_grid_bin_width,
                max_nonzero_bins=int(grid_max_nonzero_bins),
                surface_axis=surface_axis,
                distance_coordinate_mode=distance_coordinate_mode,
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_estimate=surface_estimate,
                surface_reference=surface_reference,
                oh_cutoff_A=oh_cutoff if target.selection_kind == "molecule" else None,
                min_molecule_frames=(
                    min_molecule_frames if target.selection_kind == "molecule" else None
                ),
            )

    profiles: list[DensityProfile | DensityHeatmapProfile | DensityGridProfile] = []
    accumulation_element_selector = _DensityElementSelector(
        [target.species_label for target in active_targets if target.selection_kind == "element"]
    )
    with ProgressBar(desc="Binning density frames", total=len(frames), unit="frame") as progress:
        for frame_index, frame in enumerate(frames):
            frame_selections = _select_density_targets_in_frame(
                frame,
                frame_index=frame_index,
                target_specs=active_targets,
                topology_cache=run_topology_cache,
                element_selector=accumulation_element_selector,
                wrap_positions_to_cell=wrap_positions_to_cell,
            )
            for target in active_targets:
                selection = frame_selections.get(target.species_label)
                if selection is None:
                    continue
                positions, masses = selection
                mass_array = np.asarray(masses, dtype=float)
                raw_surface_axis_values = np.asarray(positions[:, surface_axis_index], dtype=float)
                distance_components = (
                    surface_distance_components(
                        surface_reference,
                        raw_surface_axis_values,
                        frame_index=frame_index,
                    )
                    if surface_reference is not None and surface_reference.uses_distance
                    else None
                )
                for axis_key, accumulator in line_accumulators_by_target[target.species_label]:
                    accumulator_masses = mass_array
                    entity_weights = None
                    if axis_key.startswith("distance"):
                        if distance_components is not None:
                            folded, top_distance, bottom_distance, interface_side = (
                                distance_components
                            )
                            if axis_key == "distance-top":
                                selected = interface_side >= 0
                                entity_weights = np.where(interface_side[selected] == 0, 0.5, 1.0)
                                axis_values = top_distance[selected]
                                accumulator_masses = mass_array[selected] * entity_weights
                            elif axis_key == "distance-bottom":
                                selected = interface_side <= 0
                                entity_weights = np.where(interface_side[selected] == 0, 0.5, 1.0)
                                axis_values = bottom_distance[selected]
                                accumulator_masses = mass_array[selected] * entity_weights
                            else:
                                axis_values = folded
                        else:
                            axis_values = (
                                raw_surface_axis_values
                                if per_frame_surface is None
                                else raw_surface_axis_values - float(per_frame_surface[frame_index])
                            )
                    else:
                        axis_values = np.asarray(positions[:, axis_indices[axis_key]], dtype=float)
                    accumulator.update(
                        axis_values=axis_values,
                        masses=accumulator_masses,
                        frame_index=frame_index,
                        validation_plan=validation_plan,
                        block_index_by_frame=block_index_by_frame,
                        entity_weights=entity_weights,
                    )
                for plane_axes, heatmap_accumulator in heatmap_accumulators_by_target[
                    target.species_label
                ]:
                    heatmap_accumulator.update(
                        x_values=np.asarray(positions[:, axis_indices[plane_axes[0]]], dtype=float),
                        y_values=np.asarray(positions[:, axis_indices[plane_axes[1]]], dtype=float),
                        masses=mass_array,
                        frame_index=frame_index,
                        validation_plan=validation_plan,
                        frame=frame,
                    )
                grid_accumulator = grid_accumulator_by_target.get(target.species_label)
                if grid_accumulator is not None:
                    grid_interface_side = None
                    if distance_components is not None:
                        grid_distance_values = distance_components[0]
                        grid_interface_side = distance_components[3]
                    else:
                        grid_distance_values = (
                            raw_surface_axis_values
                            if per_frame_surface is None
                            else raw_surface_axis_values - float(per_frame_surface[frame_index])
                        )
                    grid_accumulator.update(
                        positions=positions,
                        distance_values=grid_distance_values,
                        masses=mass_array,
                        frame_index=frame_index,
                        validation_plan=validation_plan,
                        interface_side=grid_interface_side,
                    )
            progress.update()

        for target in active_targets:
            finalized_lines = {
                key: accumulator.finalize(n_frames=len(frames))
                for key, accumulator in line_accumulators_by_target[target.species_label]
            }
            if "distance-top" in finalized_lines and "distance-bottom" in finalized_lines:
                assert surface_reference is not None
                finalized_lines["distance"] = _fold_symmetric_density_profiles(
                    finalized_lines.pop("distance-top"),
                    finalized_lines.pop("distance-bottom"),
                    surface_reference=surface_reference,
                    surface_position=surface_position,
                    surface_position_std=surface_position_std,
                )
            target_profiles: list[DensityProfile | DensityHeatmapProfile | DensityGridProfile] = [
                *finalized_lines.values(),
                *(
                    accumulator.finalize(n_frames=len(frames))
                    for _, accumulator in heatmap_accumulators_by_target[target.species_label]
                ),
            ]
            grid_accumulator = grid_accumulator_by_target.get(target.species_label)
            if grid_accumulator is not None:
                target_profiles.append(grid_accumulator.finalize(n_frames=len(frames)))
            profiles.extend(target_profiles)

    LOGGER.info(
        "Density binning summary: outputs=%d, frames=%d, saved_statistics=sample%s.",
        len(profiles),
        len(frames),
        "+block" if block_index_by_frame is not None else "",
    )
    return profiles


def compute_all_density_profiles(
    frames: Sequence[Atoms],
    species: str | None = "all",
    surface_axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_layout: str | None = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "cell",
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | SurfaceReference | None = None,
    outputs: str | None = "all",
    heatmap_planes: list[str] | tuple[str, ...] | None = None,
    grid_bin_width: float | None = None,
    grid_max_nonzero_bins: int = DEFAULT_DENSITY_GRID_MAX_NONZERO_BINS,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
    oh_topology_backend: str = "auto",
) -> list[DensityProfile | DensityHeatmapProfile | DensityGridProfile]:
    return _compute_all_density_profiles_streaming(
        frames=frames,
        species=species,
        surface_axis=surface_axis,
        bin_width=bin_width,
        surface_mode=surface_mode,
        surface_layout=surface_layout,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        binning=binning,
        surface_options=surface_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
        outputs=outputs,
        heatmap_planes=heatmap_planes,
        grid_bin_width=grid_bin_width,
        grid_max_nonzero_bins=grid_max_nonzero_bins,
        oh_cutoff=oh_cutoff,
        min_molecule_frames=min_molecule_frames,
        oh_topology_backend=oh_topology_backend,
    )


def _density_profile_hdf5_payload(profile: DensityProfile) -> dict[str, Any]:
    """Return validated HDF5 datasets/metadata payload for one density profile."""
    _profile_selection_mode, profile_species_label = _normalize_species_query(
        profile.species,
        allow_h2o=True,
        allow_molecules=True,
    )
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
        "profile_kind": "line_1d",
        "axis": profile.axis,
        "species": profile_species_label,
        "selection_kind": _density_selection_kind_for_label(profile_species_label),
        "entity_kind": "molecule" if is_molecule_species_label(profile_species_label) else "atom",
        "units": canonical_density_units,
        "n_frames": profile.n_frames,
        "number_density_units": canonical_number_units,
        "coordinate_mode": profile.coordinate_mode,
        "interface_view": profile.interface_view,
        "bin_width_A": bin_width,
        "counts_per_frame_available": False,
        "visible_x_min_A": profile.visible_x_min_A,
        "visible_x_max_A": profile.visible_x_max_A,
        "oh_cutoff_A": profile.oh_cutoff_A,
        "min_molecule_frames": profile.min_molecule_frames,
        **(
            _surface_reference_metadata(
                profile.surface_reference,
                surface_position=profile.surface_position,
                surface_position_std=profile.surface_position_std,
            )
            if profile.surface_reference is not None
            else _surface_metadata_payload(
                surface_position=profile.surface_position,
                surface_position_std=profile.surface_position_std,
                estimate=profile.surface_estimate,
            )
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
        "density_top": profile.top_density,
        "density_bottom": profile.bottom_density,
        "number_density_top": profile.top_number_density,
        "number_density_bottom": profile.bottom_number_density,
        "counts_per_frame_top": profile.top_counts_per_frame,
        "counts_per_frame_bottom": profile.bottom_counts_per_frame,
        **statistics_payload_from_series_map(profile.series_statistics),
        **(
            {}
            if profile.surface_reference is not None
            else _surface_estimate_datasets(profile.surface_estimate)
        ),
        **_surface_reference_datasets(profile.surface_reference),
    }
    return {
        "datasets": datasets,
        "metadata": metadata,
    }


def _density_heatmap_profile_hdf5_payload(profile: DensityHeatmapProfile) -> dict[str, Any]:
    """Return validated HDF5 datasets/metadata payload for one density heatmap profile."""
    _profile_selection_mode, profile_species_label = _normalize_species_query(
        profile.species,
        allow_h2o=True,
        allow_molecules=True,
    )

    canonical_density, canonical_density_units, canonical_number_density, canonical_number_units = (
        canonicalize_density_units(
            density=profile.density,
            density_units=profile.units,
            number_density=profile.number_density,
            number_density_units=profile.number_density_units,
        )
        if str(profile.units).strip().lower() == "g/cm^3"
        else (
            np.asarray(profile.density, dtype=float),
            str(profile.units),
            None
            if profile.number_density is None
            else np.asarray(profile.number_density, dtype=float),
            profile.number_density_units,
        )
    )
    units_map = {"density": canonical_density_units}
    if canonical_number_units is not None:
        units_map["number_density"] = canonical_number_units
    metadata = build_profile_metadata(
        analysis="density",
        metadata={
            "profile_kind": "heatmap_2d",
            "plane": profile.plane,
            "plane_axes": list(profile.plane_axes),
            "species": profile_species_label,
            "selection_kind": _density_selection_kind_for_label(profile_species_label),
            "entity_kind": "molecule"
            if is_molecule_species_label(profile_species_label)
            else "atom",
            "n_frames": profile.n_frames,
            "units": canonical_density_units,
            "number_density_units": canonical_number_units,
            "visible_x_min_A": profile.visible_x_min_A,
            "visible_x_max_A": profile.visible_x_max_A,
            "visible_y_min_A": profile.visible_y_min_A,
            "visible_y_max_A": profile.visible_y_max_A,
            "oh_cutoff_A": profile.oh_cutoff_A,
            "min_molecule_frames": profile.min_molecule_frames,
            "x_bin_width_A": uniform_bin_width_from_edges(
                np.asarray(profile.x_bin_edges, dtype=float),
                source_label=f"Density heatmap '{profile.species}' x-axis",
            ),
            "y_bin_width_A": uniform_bin_width_from_edges(
                np.asarray(profile.y_bin_edges, dtype=float),
                source_label=f"Density heatmap '{profile.species}' y-axis",
            ),
        },
        units_map=units_map,
    )
    return {
        "datasets": {
            "x_bin_centers_A": np.asarray(profile.x_bin_centers, dtype=float),
            "y_bin_centers_A": np.asarray(profile.y_bin_centers, dtype=float),
            "density": canonical_density,
            "number_density": canonical_number_density,
        },
        "metadata": metadata,
    }


def _density_grid_profile_hdf5_payload(profile: DensityGridProfile) -> dict[str, Any]:
    """Return validated HDF5 datasets/metadata payload for one sparse density grid."""

    _profile_selection_mode, profile_species_label = _normalize_species_query(
        profile.species,
        allow_h2o=True,
        allow_molecules=True,
    )
    x_edges = np.asarray(profile.x_bin_edges, dtype=float)
    y_edges = np.asarray(profile.y_bin_edges, dtype=float)
    z_edges = np.asarray(profile.z_bin_edges, dtype=float)
    distance_edges = np.asarray(profile.distance_bin_edges, dtype=float)
    edge_arrays = {
        "x": x_edges,
        "y": y_edges,
        "z": z_edges,
        "distance": distance_edges,
    }
    for axis_name, edges in edge_arrays.items():
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(f"Density sparse grid has invalid {axis_name} bin edges.")
        uniform_bin_width_from_edges(
            edges,
            source_label=f"Density sparse grid '{profile.species}' {axis_name}",
        )
    grid_shape = (
        int(x_edges.size - 1),
        int(y_edges.size - 1),
        int(z_edges.size - 1),
        int(distance_edges.size - 1),
    )
    flat_indices = np.asarray(profile.flat_indices, dtype=np.uint64)
    mass_sum = np.asarray(profile.mass_sum_g, dtype=_DENSITY_GRID_STORAGE_DTYPE)
    entity_sum = np.asarray(profile.entity_sum, dtype=_DENSITY_GRID_STORAGE_DTYPE)
    if mass_sum.shape != flat_indices.shape or entity_sum.shape != flat_indices.shape:
        raise ValueError(
            "Density sparse grid flat_indices, mass_sum_g, and entity_sum sizes differ."
        )
    if flat_indices.size:
        if np.any(flat_indices[1:] < flat_indices[:-1]):
            sort_order = np.argsort(flat_indices)
            flat_indices = flat_indices[sort_order]
            mass_sum = mass_sum[sort_order]
            entity_sum = entity_sum[sort_order]
        dense_cells = int(np.prod(np.asarray(grid_shape, dtype=object)))
        if int(flat_indices[-1]) >= dense_cells:
            raise ValueError("Density sparse grid contains a flat index outside the grid shape.")
    visible_ranges: dict[str, float | None] = {}
    for axis_index, axis_name in enumerate(_DENSITY_GRID_AXES):
        lower, upper = _grid_visible_axis_range_from_flat_indices(
            flat_indices=flat_indices,
            grid_shape=grid_shape,
            axis_index=axis_index,
            bin_edges=edge_arrays[axis_name],
        )
        visible_ranges[f"visible_{axis_name}_min_A"] = lower
        visible_ranges[f"visible_{axis_name}_max_A"] = upper

    metadata_payload = {
        "profile_kind": "grid_3d_sparse",
        "species": profile_species_label,
        "selection_kind": _density_selection_kind_for_label(profile_species_label),
        "entity_kind": "molecule" if is_molecule_species_label(profile_species_label) else "atom",
        "grid_axes": list(_DENSITY_GRID_AXES),
        "grid_shape": list(grid_shape),
        "grid_bin_width_A": (
            float(profile.grid_bin_width_A)
            if profile.grid_bin_width_A is not None
            else uniform_bin_width_from_edges(x_edges, source_label="Density sparse grid X")
        ),
        "grid_max_nonzero_bins": profile.grid_max_nonzero_bins,
        "grid_nonzero_bins": int(flat_indices.size),
        "grid_storage_dtype": np.dtype(_DENSITY_GRID_STORAGE_DTYPE).name,
        "units": "g",
        "n_frames": int(profile.n_frames),
        "number_density_units": profile.number_density_units,
        "surface_axis": str(profile.surface_axis).strip().lower() or "z",
        "distance_coordinate_mode": str(profile.distance_coordinate_mode).strip().lower() or "axis",
        "oh_cutoff_A": profile.oh_cutoff_A,
        "min_molecule_frames": profile.min_molecule_frames,
        **visible_ranges,
        **(
            _surface_reference_metadata(
                profile.surface_reference,
                surface_position=profile.surface_position,
                surface_position_std=profile.surface_position_std,
            )
            if profile.surface_reference is not None
            else _surface_metadata_payload(
                surface_position=profile.surface_position,
                surface_position_std=profile.surface_position_std,
                estimate=profile.surface_estimate,
            )
        ),
    }
    metadata = build_profile_metadata(
        analysis="density",
        metadata=metadata_payload,
        units_map={
            "mass_sum_g": "g",
            "entity_sum": "entity",
        },
    )
    return {
        "datasets": {
            "x_bin_edges_A": x_edges,
            "y_bin_edges_A": y_edges,
            "z_bin_edges_A": z_edges,
            "distance_bin_edges_A": distance_edges,
            **_geometry_group_datasets(profile.geometry_groups),
            "flat_indices": flat_indices,
            "mass_sum_g": mass_sum,
            "entity_sum": entity_sum,
            "top_mass_sum_g": profile.top_mass_sum_g,
            "bottom_mass_sum_g": profile.bottom_mass_sum_g,
            "top_entity_sum": profile.top_entity_sum,
            "bottom_entity_sum": profile.bottom_entity_sum,
            **(
                {}
                if profile.surface_reference is not None
                else _surface_estimate_datasets(profile.surface_estimate)
            ),
            **_surface_reference_datasets(profile.surface_reference),
        },
        "metadata": metadata,
    }


def save_density_profile(
    profile: DensityProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
    compression_profile: str = "fast",
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
        compression_profile=compression_profile,
    )
    LOGGER.info("Saved density data to '%s'.", output_path)
    return output_path


def save_density_profiles(
    profiles: Sequence[DensityProfile | DensityHeatmapProfile | DensityGridProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
    compression_profile: str = "fast",
) -> Path:
    """Save one or more density profiles to LiNaK HDF5 and return the written path."""
    if not profiles:
        raise ValueError("At least one density profile is required.")
    if len(profiles) == 1:
        single_profile = profiles[0]
        if isinstance(single_profile, DensityProfile):
            return save_density_profile(
                single_profile,
                output,
                additional_metadata=additional_metadata,
                compression_profile=compression_profile,
            )
        payload = (
            _density_grid_profile_hdf5_payload(single_profile)
            if isinstance(single_profile, DensityGridProfile)
            else _density_heatmap_profile_hdf5_payload(single_profile)
        )
        metadata = dict(payload["metadata"])
        if additional_metadata:
            metadata.update(dict(additional_metadata))
        output_path = write_linak_hdf5(
            output,
            analysis="density",
            datasets=payload["datasets"],
            metadata=metadata,
            compression_profile=compression_profile,
        )
        LOGGER.info("Saved density data to '%s'.", output_path)
        return output_path

    output_path = write_profile_collection(
        output,
        analysis="density",
        profiles=[
            _density_profile_hdf5_payload(profile)
            if isinstance(profile, DensityProfile)
            else (
                _density_grid_profile_hdf5_payload(profile)
                if isinstance(profile, DensityGridProfile)
                else _density_heatmap_profile_hdf5_payload(profile)
            )
            for profile in profiles
        ],
        metadata=dict(additional_metadata or {}),
        compression_profile=compression_profile,
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
    profile_kind: str | None = "line_1d",
    plane: str | None = None,
) -> bool:
    requested_profile_kind = None if profile_kind is None else str(profile_kind).strip().lower()
    metadata_profile_kind = (
        str(metadata.get("profile_kind", "line_1d")).strip().lower() or "line_1d"
    )
    if requested_profile_kind is not None and metadata_profile_kind != requested_profile_kind:
        return False

    requested_axis = None if axis is None or not str(axis).strip() else str(axis).strip().lower()
    if requested_axis is not None:
        metadata_axis = str(metadata.get("axis", "z")).strip().lower() or "z"
        if metadata_axis != requested_axis:
            return False

    requested_plane = (
        None if plane is None or not str(plane).strip() else str(plane).strip().lower()
    )
    if requested_plane is not None:
        metadata_plane = str(metadata.get("plane", "")).strip().lower()
        if metadata_plane != requested_plane:
            return False

    requested_species = None if species is None or not str(species).strip() else str(species)
    if requested_species is not None:
        selection_mode, requested_label = _normalize_species_query(
            requested_species,
            allow_h2o=True,
            allow_molecules=True,
        )
        if selection_mode != "all":
            metadata_species = str(metadata.get("species", "")).strip()
            if not metadata_species:
                return False
            metadata_mode, metadata_label = _normalize_species_query(
                metadata_species,
                allow_h2o=True,
                allow_molecules=True,
            )
            if selection_mode == "elements":
                if metadata_mode != "element":
                    return False
                return True
            if selection_mode == "molecules":
                if metadata_mode != "molecule":
                    return False
                return True
            if metadata_mode != selection_mode or metadata_label != requested_label:
                return False

    return True


def _optional_float_metadata(metadata: Mapping[str, Any], name: str) -> float | None:
    raw_value = metadata.get(name)
    if raw_value is None:
        return None
    value = float(raw_value)
    return value if np.isfinite(value) else None


def _optional_int_metadata(metadata: Mapping[str, Any], name: str) -> int | None:
    raw_value = metadata.get(name)
    if raw_value is None:
        return None
    return int(raw_value)


def _load_density_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    axis: str | None = None,
    species: str | None = None,
) -> list[DensityProfile]:
    profiles: list[DensityProfile] = []
    for datasets, metadata in payloads:
        if not _density_payload_matches_selection(
            metadata,
            axis=axis,
            species=species,
            profile_kind="line_1d",
        ):
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
            _selection_mode, species_label = _normalize_species_query(
                species,
                allow_h2o=True,
                allow_molecules=True,
            )
        elif species_meta:
            _metadata_mode, species_label = _normalize_species_query(
                species_meta,
                allow_h2o=True,
                allow_molecules=True,
            )
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
        surface_reference = _surface_reference_from_payload(
            datasets=datasets,
            metadata=metadata,
        )
        if surface_reference is not None:
            surface_estimate = surface_reference.primary

        def optional_series(
            name: str,
            *,
            _datasets: Mapping[str, np.ndarray] = datasets,
        ) -> np.ndarray | None:
            return None if name not in _datasets else np.asarray(_datasets[name], dtype=float)

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

        stored_bin_centers = np.asarray(datasets["bin_centers_A"])
        bin_width = resolve_uniform_bin_width_for_load(
            metadata=metadata,
            bin_centers=stored_bin_centers,
            source_path=source_path,
            analysis_name="Density",
        )
        bin_centers = np.asarray(stored_bin_centers, dtype=float)
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
                surface_reference=surface_reference,
                interface_view=str(metadata.get("interface_view", "single")),
                top_density=optional_series("density_top"),
                bottom_density=optional_series("density_bottom"),
                top_number_density=optional_series("number_density_top"),
                bottom_number_density=optional_series("number_density_bottom"),
                top_counts_per_frame=optional_series("counts_per_frame_top"),
                bottom_counts_per_frame=optional_series("counts_per_frame_bottom"),
                series_statistics=series_statistics,
                visible_x_min_A=_optional_float_metadata(metadata, "visible_x_min_A"),
                visible_x_max_A=_optional_float_metadata(metadata, "visible_x_max_A"),
                oh_cutoff_A=_optional_float_metadata(metadata, "oh_cutoff_A"),
                min_molecule_frames=_optional_int_metadata(metadata, "min_molecule_frames"),
            )
        )
    return profiles


def _load_density_heatmap_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species: str | None = None,
    plane: str | None = None,
) -> list[DensityHeatmapProfile]:
    """Load selected 2D density heatmap profiles from LiNaK HDF5 payloads."""

    profiles: list[DensityHeatmapProfile] = []
    for datasets, metadata in payloads:
        if not _density_payload_matches_selection(
            metadata,
            species=species,
            plane=plane,
            profile_kind="heatmap_2d",
        ):
            continue
        required = ("x_bin_centers_A", "y_bin_centers_A", "density")
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Density HDF5 '{source_path}' is missing required heatmap dataset(s): {', '.join(missing)}."
            )
        plane_token = str(metadata.get("plane", "")).strip().lower() or "xy"
        plane_axes_value = metadata.get("plane_axes")
        if isinstance(plane_axes_value, (list, tuple)) and len(plane_axes_value) >= 2:
            plane_axes = (
                str(plane_axes_value[0]).strip().lower() or plane_token[0],
                str(plane_axes_value[1]).strip().lower() or plane_token[1],
            )
        else:
            plane_axes = (plane_token[0], plane_token[1])
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
        density_values = np.asarray(datasets["density"], dtype=float)
        number_density_values = (
            None
            if "number_density" not in datasets
            else np.asarray(datasets["number_density"], dtype=float)
        )
        if str(units).strip().lower() == "g/cm^3":
            density_values, units, number_density_values, number_density_units = (
                canonicalize_density_units(
                    density=density_values,
                    density_units=units,
                    number_density=number_density_values,
                    number_density_units=number_density_units,
                )
            )
        x_bin_centers = np.asarray(datasets["x_bin_centers_A"], dtype=float)
        y_bin_centers = np.asarray(datasets["y_bin_centers_A"], dtype=float)
        x_bin_width = float(metadata.get("x_bin_width_A", 0.0))
        y_bin_width = float(metadata.get("y_bin_width_A", 0.0))
        if x_bin_width <= 0.0 or y_bin_width <= 0.0:
            raise ValueError(f"Density HDF5 '{source_path}' has incompatible heatmap bin geometry.")
        x_bin_edges = reconstruct_uniform_bin_edges_from_centers(
            x_bin_centers, bin_width=x_bin_width
        )
        y_bin_edges = reconstruct_uniform_bin_edges_from_centers(
            y_bin_centers, bin_width=y_bin_width
        )
        species_meta = str(metadata.get("species", "")).strip() or "UNKNOWN"
        if species_meta != "UNKNOWN":
            _metadata_mode, species_label = _normalize_species_query(
                species_meta,
                allow_h2o=True,
                allow_molecules=True,
            )
        else:
            species_label = species_meta

        profiles.append(
            DensityHeatmapProfile(
                plane=plane_token,
                plane_axes=plane_axes,
                species=species_label,
                x_bin_edges=x_bin_edges,
                y_bin_edges=y_bin_edges,
                x_bin_centers=x_bin_centers,
                y_bin_centers=y_bin_centers,
                density=density_values,
                units=units,
                n_frames=int(metadata.get("n_frames", 0)),
                number_density=number_density_values,
                number_density_units=number_density_units,
                visible_x_min_A=_optional_float_metadata(metadata, "visible_x_min_A"),
                visible_x_max_A=_optional_float_metadata(metadata, "visible_x_max_A"),
                visible_y_min_A=_optional_float_metadata(metadata, "visible_y_min_A"),
                visible_y_max_A=_optional_float_metadata(metadata, "visible_y_max_A"),
                oh_cutoff_A=_optional_float_metadata(metadata, "oh_cutoff_A"),
                min_molecule_frames=_optional_int_metadata(metadata, "min_molecule_frames"),
            )
        )
    return profiles


def _load_density_grid_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species: str | None = None,
) -> list[DensityGridProfile]:
    """Load selected sparse density grid profiles from LiNaK HDF5 payloads."""

    profiles: list[DensityGridProfile] = []
    for datasets, metadata in payloads:
        if not _density_payload_matches_selection(
            metadata,
            species=species,
            profile_kind="grid_3d_sparse",
        ):
            continue
        required = (
            "x_bin_edges_A",
            "y_bin_edges_A",
            "z_bin_edges_A",
            "distance_bin_edges_A",
            "flat_indices",
            "mass_sum_g",
            "entity_sum",
        )
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Density HDF5 '{source_path}' is missing required sparse-grid dataset(s): "
                f"{', '.join(missing)}."
            )

        species_meta = str(metadata.get("species", "")).strip() or "UNKNOWN"
        if species is not None and species.strip():
            _selection_mode, species_label = _normalize_species_query(
                species,
                allow_h2o=True,
                allow_molecules=True,
            )
        elif species_meta != "UNKNOWN":
            _metadata_mode, species_label = _normalize_species_query(
                species_meta,
                allow_h2o=True,
                allow_molecules=True,
            )
        else:
            species_label = species_meta

        resolved_units = resolve_units_map(analysis="density", metadata=metadata)
        number_density_units_raw = metadata.get("number_density_units")
        if number_density_units_raw is None:
            number_density_units_raw = resolved_units.get("number_density")
        number_density_units = (
            str(number_density_units_raw).strip() if number_density_units_raw is not None else None
        )
        if not number_density_units:
            number_density_units = _entity_density_units_for_species(species_label, volumetric=True)

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

        profiles.append(
            DensityGridProfile(
                geometry_groups=_load_geometry_groups(datasets),
                species=species_label,
                x_bin_edges=np.asarray(datasets["x_bin_edges_A"]),
                y_bin_edges=np.asarray(datasets["y_bin_edges_A"]),
                z_bin_edges=np.asarray(datasets["z_bin_edges_A"]),
                distance_bin_edges=np.asarray(datasets["distance_bin_edges_A"]),
                flat_indices=np.asarray(datasets["flat_indices"], dtype=np.uint64),
                mass_sum_g=np.asarray(datasets["mass_sum_g"], dtype=_DENSITY_GRID_STORAGE_DTYPE),
                entity_sum=np.asarray(datasets["entity_sum"], dtype=_DENSITY_GRID_STORAGE_DTYPE),
                n_frames=int(metadata.get("n_frames", 0)),
                units=str(metadata.get("units") or resolved_units.get("mass_sum_g") or "g"),
                number_density_units=number_density_units,
                grid_bin_width_A=_optional_float_metadata(metadata, "grid_bin_width_A"),
                grid_max_nonzero_bins=_optional_int_metadata(metadata, "grid_max_nonzero_bins"),
                surface_axis=str(metadata.get("surface_axis", "z")).strip().lower() or "z",
                distance_coordinate_mode=str(metadata.get("distance_coordinate_mode", "distance"))
                .strip()
                .lower()
                or "distance",
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_estimate=_surface_estimate_from_payload(
                    datasets=datasets,
                    metadata=metadata,
                ),
                surface_reference=_surface_reference_from_payload(
                    datasets=datasets,
                    metadata=metadata,
                ),
                top_mass_sum_g=(
                    None
                    if "top_mass_sum_g" not in datasets
                    else np.asarray(datasets["top_mass_sum_g"], dtype=_DENSITY_GRID_STORAGE_DTYPE)
                ),
                bottom_mass_sum_g=(
                    None
                    if "bottom_mass_sum_g" not in datasets
                    else np.asarray(
                        datasets["bottom_mass_sum_g"], dtype=_DENSITY_GRID_STORAGE_DTYPE
                    )
                ),
                top_entity_sum=(
                    None
                    if "top_entity_sum" not in datasets
                    else np.asarray(datasets["top_entity_sum"], dtype=_DENSITY_GRID_STORAGE_DTYPE)
                ),
                bottom_entity_sum=(
                    None
                    if "bottom_entity_sum" not in datasets
                    else np.asarray(
                        datasets["bottom_entity_sum"], dtype=_DENSITY_GRID_STORAGE_DTYPE
                    )
                ),
                oh_cutoff_A=_optional_float_metadata(metadata, "oh_cutoff_A"),
                min_molecule_frames=_optional_int_metadata(metadata, "min_molecule_frames"),
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


def load_density_heatmap_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species: str | None = None,
    plane: str | None = None,
) -> list[DensityHeatmapProfile]:
    """Load selected density heatmap profiles by profile index from LiNaK HDF5."""

    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="density",
        label="Density",
    )
    return _load_density_heatmap_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
        plane=plane,
    )


def load_density_heatmap_profiles(
    path: str | Path,
    *,
    species: str | None = None,
    plane: str | None = None,
) -> list[DensityHeatmapProfile]:
    """Load one or more 2D density heatmap profiles from LiNaK HDF5."""

    source_path, payloads = read_profile_payloads(
        path,
        analysis="density",
        label="Density",
    )
    return _load_density_heatmap_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
        plane=plane,
    )


def load_density_grid_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species: str | None = None,
) -> list[DensityGridProfile]:
    """Load selected sparse density grid profiles by profile index from LiNaK HDF5."""

    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="density",
        label="Density",
    )
    return _load_density_grid_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
    )


def load_density_grid_profiles(
    path: str | Path,
    *,
    species: str | None = None,
) -> list[DensityGridProfile]:
    """Load one or more sparse density grid profiles from LiNaK HDF5."""

    source_path, payloads = read_profile_payloads(
        path,
        analysis="density",
        label="Density",
    )
    return _load_density_grid_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
    )


DENSITY_GRID_2D_ALLOWED_PAIRS: tuple[tuple[str, str], ...] = (
    ("x", "y"),
    ("y", "x"),
    ("x", "z"),
    ("z", "x"),
    ("x", "distance"),
    ("distance", "x"),
    ("y", "z"),
    ("z", "y"),
    ("y", "distance"),
    ("distance", "y"),
)


def _density_grid_edges(profile: DensityGridProfile, axis_id: str) -> np.ndarray:
    axis_key = str(axis_id).strip().lower()
    if axis_key == "x":
        return np.asarray(profile.x_bin_edges, dtype=float)
    if axis_key == "y":
        return np.asarray(profile.y_bin_edges, dtype=float)
    if axis_key == "z":
        return np.asarray(profile.z_bin_edges, dtype=float)
    if axis_key == "distance":
        return np.asarray(profile.distance_bin_edges, dtype=float)
    raise ValueError("Density grid axis must be one of: x, y, z, distance.")


def _density_grid_axis_position(axis_id: str) -> int:
    axis_key = str(axis_id).strip().lower()
    if axis_key not in _DENSITY_GRID_AXES:
        raise ValueError("Density grid axis must be one of: x, y, z, distance.")
    return int(_DENSITY_GRID_AXES.index(axis_key))


def _density_grid_axis_centers(profile: DensityGridProfile, axis_id: str) -> np.ndarray:
    edges = _density_grid_edges(profile, axis_id)
    return 0.5 * (edges[:-1] + edges[1:])


def _density_grid_axis_width(profile: DensityGridProfile, axis_id: str) -> float:
    edges = _density_grid_edges(profile, axis_id)
    return max(0.0, float(edges[-1] - edges[0]))


def _normalized_density_grid_filters(
    filters: Mapping[str, tuple[float | None, float | None] | None] | None,
) -> dict[str, tuple[float | None, float | None]]:
    normalized: dict[str, tuple[float | None, float | None]] = {}
    if not filters:
        return normalized
    for raw_axis, raw_range in filters.items():
        axis_key = str(raw_axis).strip().lower()
        if axis_key not in _DENSITY_GRID_AXES or raw_range is None:
            continue
        lower_raw, upper_raw = raw_range
        lower = None if lower_raw is None else float(lower_raw)
        upper = None if upper_raw is None else float(upper_raw)
        if lower is not None and upper is not None and upper < lower:
            lower, upper = upper, lower
        normalized[axis_key] = (lower, upper)
    return normalized


def _density_grid_filter_width(
    profile: DensityGridProfile,
    axis_id: str,
    filters: Mapping[str, tuple[float | None, float | None]],
) -> float:
    axis_key = str(axis_id).strip().lower()
    edges = _density_grid_edges(profile, axis_key)
    raw_filter = filters.get(axis_key)
    if raw_filter is None:
        return max(0.0, float(edges[-1] - edges[0]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mask = np.ones(centers.shape, dtype=bool)
    filter_lower, filter_upper = raw_filter
    if filter_lower is not None:
        mask &= centers >= float(filter_lower)
    if filter_upper is not None:
        mask &= centers <= float(filter_upper)
    if not np.any(mask):
        return 0.0
    return max(0.0, float(np.sum(widths[mask])))


def _density_grid_target_bins(
    profile: DensityGridProfile,
    axis_id: str,
    *,
    requested_bin_width: float | None,
) -> tuple[np.ndarray, float]:
    source_edges = _density_grid_edges(profile, axis_id)
    source_width = uniform_bin_width_from_edges(
        source_edges,
        source_label=f"Density sparse grid {axis_id}",
    )
    if requested_bin_width is None or float(requested_bin_width) <= source_width:
        return source_edges, source_width
    target_width = float(requested_bin_width)
    span = float(source_edges[-1] - source_edges[0])
    n_bins = max(1, int(np.ceil(span / target_width)))
    target_edges = float(source_edges[0]) + np.arange(n_bins + 1, dtype=float) * target_width
    if target_edges[-1] < source_edges[-1]:
        target_edges = np.append(target_edges, target_edges[-1] + target_width)
    return np.asarray(target_edges, dtype=float), target_width


def _density_grid_sparse_mask_and_indices(
    profile: DensityGridProfile,
    *,
    filters: Mapping[str, tuple[float | None, float | None]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    flat = np.asarray(profile.flat_indices, dtype=np.uint64)
    mask = np.ones(flat.shape, dtype=bool)
    indices_by_axis: dict[str, np.ndarray] = {}
    for axis_index, axis_id in enumerate(_DENSITY_GRID_AXES):
        indices = _density_grid_axis_indices(
            flat, grid_shape=profile.grid_shape, axis_index=axis_index
        )
        indices_by_axis[axis_id] = indices
        raw_filter = filters.get(axis_id)
        if raw_filter is None or indices.size == 0:
            continue
        centers = _density_grid_axis_centers(profile, axis_id)
        center_values = centers[indices]
        lower, upper = raw_filter
        if lower is not None:
            mask &= center_values >= float(lower)
        if upper is not None:
            mask &= center_values <= float(upper)
    return mask, indices_by_axis


def _density_grid_target_indices(
    profile: DensityGridProfile,
    axis_id: str,
    source_indices: np.ndarray,
    *,
    target_edges: np.ndarray,
    target_bin_width: float,
) -> np.ndarray:
    centers = _density_grid_axis_centers(profile, axis_id)
    center_values = centers[np.asarray(source_indices, dtype=int)]
    return _uniform_bin_indices(
        center_values,
        bin_start=float(target_edges[0]),
        bin_width=float(target_bin_width),
        n_bins=int(target_edges.size - 1),
    )


def _density_grid_volume_A3(
    profile: DensityGridProfile,
    *,
    plotted_axes: tuple[str, ...],
    target_widths: Mapping[str, float],
    filters: Mapping[str, tuple[float | None, float | None]],
) -> float:
    plotted = {str(axis).strip().lower() for axis in plotted_axes}
    widths: dict[str, float] = {}
    for axis_id in ("x", "y", "z"):
        if axis_id in plotted:
            widths[axis_id] = float(target_widths[axis_id])
        else:
            widths[axis_id] = _density_grid_filter_width(profile, axis_id, filters)

    z_is_used = "z" in plotted or "z" in filters
    distance_is_used = "distance" in plotted or "distance" in filters
    if not z_is_used and distance_is_used:
        if "distance" in plotted:
            widths["z"] = float(target_widths["distance"])
        else:
            widths["z"] = _density_grid_filter_width(profile, "distance", filters)
    volume = float(widths["x"]) * float(widths["y"]) * float(widths["z"])
    return volume if np.isfinite(volume) and volume > 0.0 else float("nan")


def _density_grid_has_paired_geometry(profile: DensityGridProfile) -> bool:
    if (
        profile.top_mass_sum_g is None
        or profile.bottom_mass_sum_g is None
        or profile.top_entity_sum is None
        or profile.bottom_entity_sum is None
    ):
        return False
    if profile.geometry_groups is None:
        return True
    return all(
        group.get("top_shift") is not None
        and group.get("bottom_shift") is not None
        and group.get("top_mass_sum_g") is not None
        and group.get("bottom_mass_sum_g") is not None
        and group.get("top_entity_sum") is not None
        and group.get("bottom_entity_sum") is not None
        for group in profile.geometry_groups
    )


def density_grid_to_line_profile(
    profile: DensityGridProfile,
    *,
    x_mode: str = "distance",
    filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
    x_bin_width: float | None = None,
) -> DensityProfile:
    """Derive one display-only 1D profile from a sparse density grid."""

    x_axis = str(x_mode).strip().lower() or "distance"
    if x_axis == "axis":
        x_axis = str(profile.surface_axis).strip().lower() or "z"
    if x_axis not in _DENSITY_GRID_AXES:
        raise ValueError("Density grid 1D x-axis must be distance, x, y, or z.")
    normalized_filters = _normalized_density_grid_filters(filters)
    symmetric = bool(
        profile.surface_reference is not None and profile.surface_reference.is_symmetric
    )
    paired_symmetric = symmetric and _density_grid_has_paired_geometry(profile)
    distance_relevant = x_axis == "distance" or "distance" in normalized_filters
    if symmetric and distance_relevant and not paired_symmetric:
        raise ValueError(
            "This symmetric density grid predates paired per-geometry normalization data. "
            "Recompute the density analysis before plotting folded/top/bottom distance views."
        )
    target_edges, target_width = _density_grid_target_bins(
        profile,
        x_axis,
        requested_bin_width=x_bin_width,
    )
    n_bins = int(target_edges.size - 1)
    mask, indices_by_axis = _density_grid_sparse_mask_and_indices(
        profile,
        filters=normalized_filters,
    )
    x_target_indices = _density_grid_target_indices(
        profile,
        x_axis,
        indices_by_axis[x_axis][mask],
        target_edges=target_edges,
        target_bin_width=target_width,
    )
    mass_sum = np.bincount(
        x_target_indices,
        weights=np.asarray(profile.mass_sum_g, dtype=float)[mask],
        minlength=n_bins,
    ).astype(float, copy=False)
    entity_sum = np.bincount(
        x_target_indices,
        weights=np.asarray(profile.entity_sum, dtype=float)[mask],
        minlength=n_bins,
    ).astype(float, copy=False)
    volume_A3 = _density_grid_volume_A3(
        profile,
        plotted_axes=(x_axis,),
        target_widths={x_axis: target_width},
        filters=normalized_filters,
    )
    density = (mass_sum / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_CM3
    number_density = (entity_sum / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_NM3
    if profile.geometry_groups is not None and not (symmetric and distance_relevant):
        density, number_density, _ = _geometry_project_density(
            profile, (x_axis,), (target_edges,), normalized_filters
        )
    top_density = None
    bottom_density = None
    top_number_density = None
    bottom_number_density = None
    top_mass_sum = None
    bottom_mass_sum = None
    if paired_symmetric:
        side_area_cache: dict[tuple[float, ...], np.ndarray] = {}
        side_volume_cache: dict[tuple[object, ...], np.ndarray] = {}

        def side_projection(side: str):
            side_mass_values = np.asarray(
                profile.top_mass_sum_g if side == "top" else profile.bottom_mass_sum_g,
                dtype=float,
            )
            side_entity_values = np.asarray(
                profile.top_entity_sum if side == "top" else profile.bottom_entity_sum,
                dtype=float,
            )
            projected_mass = np.bincount(
                x_target_indices,
                weights=side_mass_values[mask],
                minlength=n_bins,
            ).astype(float, copy=False)
            projected_entity = np.bincount(
                x_target_indices,
                weights=side_entity_values[mask],
                minlength=n_bins,
            ).astype(float, copy=False)
            projected_density = (
                (projected_mass / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_CM3
            )
            projected_number = (
                (projected_entity / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_NM3
            )
            if profile.geometry_groups is not None:
                projected_density, projected_number, _ = _geometry_project_density(
                    profile,
                    (x_axis,),
                    (target_edges,),
                    normalized_filters,
                    interface_view=side,
                    _area_cache=side_area_cache,
                    _volume_cache=side_volume_cache,
                )
            return projected_mass, projected_entity, projected_density, projected_number

        top_mass_sum, top_entity_sum, top_density, top_number_density = side_projection("top")
        bottom_mass_sum, bottom_entity_sum, bottom_density, bottom_number_density = side_projection(
            "bottom"
        )
        if distance_relevant:
            mass_sum = 0.5 * (top_mass_sum + bottom_mass_sum)
            entity_sum = 0.5 * (top_entity_sum + bottom_entity_sum)
            density = 0.5 * (top_density + bottom_density)
            number_density = 0.5 * (top_number_density + bottom_number_density)
    bin_centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    visible_x_min_A, visible_x_max_A = _line_visible_range_from_density(
        bin_edges=target_edges,
        values=density,
    )
    coordinate_mode = "distance" if x_axis == "distance" else "axis"
    return DensityProfile(
        axis=(str(profile.surface_axis).strip().lower() or "z") if x_axis == "distance" else x_axis,
        species=profile.species,
        bin_edges=target_edges,
        bin_centers=bin_centers,
        counts_per_frame=mass_sum / max(1, int(profile.n_frames)),
        density=density,
        units="g/cm^3",
        n_frames=int(profile.n_frames),
        entities_per_frame=entity_sum / max(1, int(profile.n_frames)),
        number_density=number_density,
        number_density_units=profile.number_density_units
        or _entity_density_units_for_species(profile.species, volumetric=True),
        coordinate_mode=coordinate_mode,
        surface_position=profile.surface_position,
        surface_position_std=profile.surface_position_std,
        surface_estimate=profile.surface_estimate,
        surface_reference=profile.surface_reference,
        interface_view="folded" if symmetric else "single",
        top_density=top_density,
        bottom_density=bottom_density,
        top_number_density=top_number_density,
        bottom_number_density=bottom_number_density,
        top_counts_per_frame=(
            None
            if top_mass_sum is None
            else np.asarray(top_mass_sum) / max(1, int(profile.n_frames))
        ),
        bottom_counts_per_frame=(
            None
            if bottom_mass_sum is None
            else np.asarray(bottom_mass_sum) / max(1, int(profile.n_frames))
        ),
        series_statistics=None,
        visible_x_min_A=visible_x_min_A,
        visible_x_max_A=visible_x_max_A,
        oh_cutoff_A=profile.oh_cutoff_A,
        min_molecule_frames=profile.min_molecule_frames,
    )


def density_grid_to_heatmap_profile(
    profile: DensityGridProfile,
    *,
    x_axis: str = "x",
    y_axis: str = "y",
    filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
    x_bin_width: float | None = None,
    y_bin_width: float | None = None,
) -> DensityHeatmapProfile:
    """Derive one display-only 2D density map from a sparse density grid."""

    x_key = str(x_axis).strip().lower()
    y_key = str(y_axis).strip().lower()
    if (x_key, y_key) not in DENSITY_GRID_2D_ALLOWED_PAIRS:
        raise ValueError(
            "Unsupported density 2D axes. Use X/Y, X/Z, X/distance, Y/Z, "
            "Y/distance, or the reverse of those pairs."
        )
    normalized_filters = _normalized_density_grid_filters(filters)
    symmetric = bool(
        profile.surface_reference is not None and profile.surface_reference.is_symmetric
    )
    paired_symmetric = symmetric and _density_grid_has_paired_geometry(profile)
    distance_relevant = "distance" in {x_key, y_key} or "distance" in normalized_filters
    if symmetric and distance_relevant and not paired_symmetric:
        raise ValueError(
            "This symmetric density grid predates paired per-geometry normalization data. "
            "Recompute the density analysis before plotting folded/top/bottom distance views."
        )
    x_edges, x_width = _density_grid_target_bins(
        profile,
        x_key,
        requested_bin_width=x_bin_width,
    )
    y_edges, y_width = _density_grid_target_bins(
        profile,
        y_key,
        requested_bin_width=y_bin_width,
    )
    x_bin_count = int(x_edges.size - 1)
    y_bin_count = int(y_edges.size - 1)
    mask, indices_by_axis = _density_grid_sparse_mask_and_indices(
        profile,
        filters=normalized_filters,
    )
    x_indices = _density_grid_target_indices(
        profile,
        x_key,
        indices_by_axis[x_key][mask],
        target_edges=x_edges,
        target_bin_width=x_width,
    )
    y_indices = _density_grid_target_indices(
        profile,
        y_key,
        indices_by_axis[y_key][mask],
        target_edges=y_edges,
        target_bin_width=y_width,
    )
    flat_2d = x_indices * y_bin_count + y_indices
    total_bins = x_bin_count * y_bin_count
    mass_sum = (
        np.bincount(
            flat_2d,
            weights=np.asarray(profile.mass_sum_g, dtype=float)[mask],
            minlength=total_bins,
        )
        .astype(float, copy=False)
        .reshape((x_bin_count, y_bin_count))
    )
    entity_sum = (
        np.bincount(
            flat_2d,
            weights=np.asarray(profile.entity_sum, dtype=float)[mask],
            minlength=total_bins,
        )
        .astype(float, copy=False)
        .reshape((x_bin_count, y_bin_count))
    )
    volume_A3 = _density_grid_volume_A3(
        profile,
        plotted_axes=(x_key, y_key),
        target_widths={x_key: x_width, y_key: y_width},
        filters=normalized_filters,
    )
    density = (mass_sum / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_CM3
    number_density = (entity_sum / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_NM3
    if profile.geometry_groups is not None and not (symmetric and distance_relevant):
        density, number_density, _ = _geometry_project_density(
            profile, (x_key, y_key), (x_edges, y_edges), normalized_filters
        )
    top_density = None
    bottom_density = None
    top_number_density = None
    bottom_number_density = None
    if paired_symmetric:
        side_area_cache: dict[tuple[float, ...], np.ndarray] = {}
        side_volume_cache: dict[tuple[object, ...], np.ndarray] = {}

        def side_projection(side: str):
            side_mass_values = np.asarray(
                profile.top_mass_sum_g if side == "top" else profile.bottom_mass_sum_g,
                dtype=float,
            )
            side_entity_values = np.asarray(
                profile.top_entity_sum if side == "top" else profile.bottom_entity_sum,
                dtype=float,
            )
            projected_mass = (
                np.bincount(
                    flat_2d,
                    weights=side_mass_values[mask],
                    minlength=total_bins,
                )
                .astype(float, copy=False)
                .reshape((x_bin_count, y_bin_count))
            )
            projected_entity = (
                np.bincount(
                    flat_2d,
                    weights=side_entity_values[mask],
                    minlength=total_bins,
                )
                .astype(float, copy=False)
                .reshape((x_bin_count, y_bin_count))
            )
            projected_density = (
                (projected_mass / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_CM3
            )
            projected_number = (
                (projected_entity / max(1, int(profile.n_frames))) / volume_A3 / ANGSTROM3_TO_NM3
            )
            if profile.geometry_groups is not None:
                projected_density, projected_number, _ = _geometry_project_density(
                    profile,
                    (x_key, y_key),
                    (x_edges, y_edges),
                    normalized_filters,
                    interface_view=side,
                    _area_cache=side_area_cache,
                    _volume_cache=side_volume_cache,
                )
            return projected_density, projected_number

        top_density, top_number_density = side_projection("top")
        bottom_density, bottom_number_density = side_projection("bottom")
        if distance_relevant:
            density = 0.5 * (top_density + bottom_density)
            number_density = 0.5 * (top_number_density + bottom_number_density)
    visible_x_min_A, visible_x_max_A, visible_y_min_A, visible_y_max_A = (
        _heatmap_visible_ranges_from_density(
            x_bin_edges=x_edges,
            y_bin_edges=y_edges,
            values=density,
        )
    )
    return DensityHeatmapProfile(
        plane=f"{x_key}{y_key}",
        plane_axes=(x_key, y_key),
        species=profile.species,
        x_bin_edges=x_edges,
        y_bin_edges=y_edges,
        x_bin_centers=0.5 * (x_edges[:-1] + x_edges[1:]),
        y_bin_centers=0.5 * (y_edges[:-1] + y_edges[1:]),
        density=density,
        units="g/cm^3",
        n_frames=int(profile.n_frames),
        number_density=number_density,
        number_density_units=profile.number_density_units
        or _entity_density_units_for_species(profile.species, volumetric=True),
        interface_view="folded" if symmetric else "single",
        top_density=top_density,
        bottom_density=bottom_density,
        top_number_density=top_number_density,
        bottom_number_density=bottom_number_density,
        visible_x_min_A=visible_x_min_A,
        visible_x_max_A=visible_x_max_A,
        visible_y_min_A=visible_y_min_A,
        visible_y_max_A=visible_y_max_A,
        oh_cutoff_A=profile.oh_cutoff_A,
        min_molecule_frames=profile.min_molecule_frames,
    )


def _density_grid_common_target_bins(
    profiles: list[DensityGridProfile],
    axis_id: str,
    *,
    requested_bin_width: float | None,
) -> tuple[np.ndarray, float]:
    if not profiles:
        raise ValueError("At least one density sparse grid is required.")
    per_profile = [
        _density_grid_target_bins(
            profile,
            axis_id,
            requested_bin_width=requested_bin_width,
        )
        for profile in profiles
    ]
    widths = [float(width) for _edges, width in per_profile]
    common_width = widths[0]
    for width in widths[1:]:
        if not np.isclose(width, common_width, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError(
                "Cannot average density 2D grids with incompatible bin widths. "
                "Use a shared X/Y bin size or recompute the grids with matching binning."
            )
    lower = min(float(edges[0]) for edges, _width in per_profile)
    upper = max(float(edges[-1]) for edges, _width in per_profile)
    n_bins = max(1, int(np.ceil((upper - lower) / common_width)))
    edges = lower + np.arange(n_bins + 1, dtype=float) * common_width
    if edges[-1] < upper:
        edges = np.append(edges, edges[-1] + common_width)
    return np.asarray(edges, dtype=float), common_width


def density_grids_to_averaged_heatmap_profile(
    profiles: list[DensityGridProfile],
    *,
    x_axis: str = "x",
    y_axis: str = "y",
    filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
    x_bin_width: float | None = None,
    y_bin_width: float | None = None,
    species: str | None = None,
) -> DensityHeatmapProfile:
    """Derive one frame-weighted 2D density map from active sparse grids."""

    if not profiles:
        raise ValueError("No active density 2D source series are enabled.")
    x_key = str(x_axis).strip().lower()
    y_key = str(y_axis).strip().lower()
    if (x_key, y_key) not in DENSITY_GRID_2D_ALLOWED_PAIRS:
        raise ValueError(
            "Unsupported density 2D axes. Use X/Y, X/Z, X/distance, Y/Z, "
            "Y/distance, or the reverse of those pairs."
        )
    normalized_filters = _normalized_density_grid_filters(filters)
    x_edges, x_width = _density_grid_common_target_bins(
        profiles,
        x_key,
        requested_bin_width=x_bin_width,
    )
    y_edges, y_width = _density_grid_common_target_bins(
        profiles,
        y_key,
        requested_bin_width=y_bin_width,
    )
    x_bin_count = int(x_edges.size - 1)
    y_bin_count = int(y_edges.size - 1)
    total_bins = x_bin_count * y_bin_count
    total_mass_sum = np.zeros(total_bins, dtype=float)
    total_entity_sum = np.zeros(total_bins, dtype=float)
    all_symmetric = all(
        profile.surface_reference is not None and profile.surface_reference.is_symmetric
        for profile in profiles
    )
    all_paired_symmetric = all_symmetric and all(
        _density_grid_has_paired_geometry(profile) for profile in profiles
    )
    distance_relevant = "distance" in {x_key, y_key} or "distance" in normalized_filters
    top_mass_sum_total = np.zeros(total_bins, dtype=float) if all_paired_symmetric else None
    bottom_mass_sum_total = np.zeros(total_bins, dtype=float) if all_paired_symmetric else None
    top_entity_sum_total = np.zeros(total_bins, dtype=float) if all_paired_symmetric else None
    bottom_entity_sum_total = np.zeros(total_bins, dtype=float) if all_paired_symmetric else None
    general_geometry = any(p.geometry_groups is not None for p in profiles)
    geometry_mass = np.zeros((x_bin_count, y_bin_count))
    geometry_number = np.zeros_like(geometry_mass)
    geometry_coverage = np.zeros_like(geometry_mass)
    geometry_top_mass = np.zeros_like(geometry_mass) if all_paired_symmetric else None
    geometry_bottom_mass = np.zeros_like(geometry_mass) if all_paired_symmetric else None
    geometry_top_number = np.zeros_like(geometry_mass) if all_paired_symmetric else None
    geometry_bottom_number = np.zeros_like(geometry_mass) if all_paired_symmetric else None
    geometry_top_coverage = np.zeros_like(geometry_mass) if all_paired_symmetric else None
    geometry_bottom_coverage = np.zeros_like(geometry_mass) if all_paired_symmetric else None
    total_frames = 0
    volume_A3: float | None = None
    number_units: str | None = None
    oh_cutoff_values = [
        float(profile.oh_cutoff_A)
        for profile in profiles
        if profile.oh_cutoff_A is not None and np.isfinite(float(profile.oh_cutoff_A))
    ]
    min_molecule_frame_values = [
        int(profile.min_molecule_frames)
        for profile in profiles
        if profile.min_molecule_frames is not None
    ]
    for profile in profiles:
        n_frames = int(profile.n_frames)
        if n_frames <= 0:
            continue
        if general_geometry:
            if profile.geometry_groups is None:
                raise ValueError(
                    "Cannot mix legacy grids lacking cell geometry with geometry-aware grids; recompute the legacy grid."
                )
            gd, gn, gc = _geometry_project_density(
                profile, (x_key, y_key), (x_edges, y_edges), normalized_filters
            )
            geometry_mass += np.nan_to_num(gd) * gc
            geometry_number += np.nan_to_num(gn) * gc
            geometry_coverage += gc
            if all_paired_symmetric:
                assert geometry_top_mass is not None
                assert geometry_bottom_mass is not None
                assert geometry_top_number is not None
                assert geometry_bottom_number is not None
                assert geometry_top_coverage is not None
                assert geometry_bottom_coverage is not None
                td, tn, tc = _geometry_project_density(
                    profile,
                    (x_key, y_key),
                    (x_edges, y_edges),
                    normalized_filters,
                    interface_view="top",
                )
                bd, bn, bc = _geometry_project_density(
                    profile,
                    (x_key, y_key),
                    (x_edges, y_edges),
                    normalized_filters,
                    interface_view="bottom",
                )
                geometry_top_mass += np.nan_to_num(td) * tc
                geometry_top_number += np.nan_to_num(tn) * tc
                geometry_top_coverage += tc
                geometry_bottom_mass += np.nan_to_num(bd) * bc
                geometry_bottom_number += np.nan_to_num(bn) * bc
                geometry_bottom_coverage += bc
        mask, indices_by_axis = _density_grid_sparse_mask_and_indices(
            profile,
            filters=normalized_filters,
        )
        x_indices = _density_grid_target_indices(
            profile,
            x_key,
            indices_by_axis[x_key][mask],
            target_edges=x_edges,
            target_bin_width=x_width,
        )
        y_indices = _density_grid_target_indices(
            profile,
            y_key,
            indices_by_axis[y_key][mask],
            target_edges=y_edges,
            target_bin_width=y_width,
        )
        flat_2d = x_indices * y_bin_count + y_indices
        total_mass_sum += np.bincount(
            flat_2d,
            weights=np.asarray(profile.mass_sum_g, dtype=float)[mask],
            minlength=total_bins,
        ).astype(float, copy=False)
        total_entity_sum += np.bincount(
            flat_2d,
            weights=np.asarray(profile.entity_sum, dtype=float)[mask],
            minlength=total_bins,
        ).astype(float, copy=False)
        if all_paired_symmetric:
            side_sources = (
                profile.top_mass_sum_g,
                profile.bottom_mass_sum_g,
                profile.top_entity_sum,
                profile.bottom_entity_sum,
            )
            if any(value is None for value in side_sources):
                raise ValueError(
                    "A symmetric density grid lacks side-resolved accumulators. "
                    "Recompute the density analysis."
                )
            assert top_mass_sum_total is not None
            assert bottom_mass_sum_total is not None
            assert top_entity_sum_total is not None
            assert bottom_entity_sum_total is not None
            for destination, values in zip(
                (
                    top_mass_sum_total,
                    bottom_mass_sum_total,
                    top_entity_sum_total,
                    bottom_entity_sum_total,
                ),
                side_sources,
            ):
                destination += np.bincount(
                    flat_2d,
                    weights=np.asarray(values, dtype=float)[mask],
                    minlength=total_bins,
                ).astype(float, copy=False)
        profile_volume_A3 = _density_grid_volume_A3(
            profile,
            plotted_axes=(x_key, y_key),
            target_widths={x_key: x_width, y_key: y_width},
            filters=normalized_filters,
        )
        if not np.isfinite(profile_volume_A3) or profile_volume_A3 <= 0.0:
            raise ValueError("Density 2D grid selection has zero or invalid normalization volume.")
        if volume_A3 is None:
            volume_A3 = float(profile_volume_A3)
        elif not general_geometry and not np.isclose(
            profile_volume_A3, volume_A3, rtol=1.0e-9, atol=1.0e-12
        ):
            raise ValueError(
                "Cannot average density 2D grids with incompatible selected volumes. "
                "Use matching cells/ranges or plot the sources separately."
            )
        candidate_units = profile.number_density_units or _entity_density_units_for_species(
            profile.species,
            volumetric=True,
        )
        if number_units is None:
            number_units = candidate_units
        elif candidate_units != number_units:
            unified_units = _unify_number_density_units([number_units, candidate_units])
            if unified_units is None:
                raise ValueError(
                    "Cannot average density 2D grids with incompatible number-density units."
                )
            number_units = unified_units
        total_frames += n_frames
    if total_frames <= 0:
        raise ValueError("No frames are available in the active density 2D source series.")
    if volume_A3 is None:
        raise ValueError("Density 2D grid selection has no normalization volume.")
    mass_sum = total_mass_sum.reshape((x_bin_count, y_bin_count))
    entity_sum = total_entity_sum.reshape((x_bin_count, y_bin_count))
    density = (mass_sum / total_frames) / volume_A3 / ANGSTROM3_TO_CM3
    number_density = (entity_sum / total_frames) / volume_A3 / ANGSTROM3_TO_NM3
    if general_geometry:
        denominator = np.where(geometry_coverage > 0, geometry_coverage, np.nan)
        density = geometry_mass / denominator
        number_density = geometry_number / denominator
    top_density = None
    bottom_density = None
    top_number_density = None
    bottom_number_density = None
    if all_paired_symmetric:
        if general_geometry:
            assert geometry_top_mass is not None
            assert geometry_bottom_mass is not None
            assert geometry_top_number is not None
            assert geometry_bottom_number is not None
            assert geometry_top_coverage is not None
            assert geometry_bottom_coverage is not None
            top_density = geometry_top_mass / np.where(
                geometry_top_coverage > 0, geometry_top_coverage, np.nan
            )
            bottom_density = geometry_bottom_mass / np.where(
                geometry_bottom_coverage > 0, geometry_bottom_coverage, np.nan
            )
            top_number_density = geometry_top_number / np.where(
                geometry_top_coverage > 0, geometry_top_coverage, np.nan
            )
            bottom_number_density = geometry_bottom_number / np.where(
                geometry_bottom_coverage > 0, geometry_bottom_coverage, np.nan
            )
        else:
            assert top_mass_sum_total is not None
            assert bottom_mass_sum_total is not None
            assert top_entity_sum_total is not None
            assert bottom_entity_sum_total is not None
            top_density = (
                (top_mass_sum_total.reshape((x_bin_count, y_bin_count)) / total_frames)
                / volume_A3
                / ANGSTROM3_TO_CM3
            )
            bottom_density = (
                (bottom_mass_sum_total.reshape((x_bin_count, y_bin_count)) / total_frames)
                / volume_A3
                / ANGSTROM3_TO_CM3
            )
            top_number_density = (
                (top_entity_sum_total.reshape((x_bin_count, y_bin_count)) / total_frames)
                / volume_A3
                / ANGSTROM3_TO_NM3
            )
            bottom_number_density = (
                (bottom_entity_sum_total.reshape((x_bin_count, y_bin_count)) / total_frames)
                / volume_A3
                / ANGSTROM3_TO_NM3
            )
        if distance_relevant:
            density = 0.5 * (top_density + bottom_density)
            number_density = 0.5 * (top_number_density + bottom_number_density)
    visible_x_min_A, visible_x_max_A, visible_y_min_A, visible_y_max_A = (
        _heatmap_visible_ranges_from_density(
            x_bin_edges=x_edges,
            y_bin_edges=y_edges,
            values=density,
        )
    )
    display_species = species or profiles[0].species
    return DensityHeatmapProfile(
        plane=f"{x_key}{y_key}",
        plane_axes=(x_key, y_key),
        species=f"{display_species} averaged {x_key.upper()}/{y_key.upper()}",
        x_bin_edges=x_edges,
        y_bin_edges=y_edges,
        x_bin_centers=0.5 * (x_edges[:-1] + x_edges[1:]),
        y_bin_centers=0.5 * (y_edges[:-1] + y_edges[1:]),
        density=density,
        units="g/cm^3",
        n_frames=total_frames,
        number_density=number_density,
        number_density_units=number_units
        or _entity_density_units_for_species(display_species, volumetric=True),
        interface_view="folded" if all_symmetric else "single",
        top_density=top_density,
        bottom_density=bottom_density,
        top_number_density=top_number_density,
        bottom_number_density=bottom_number_density,
        visible_x_min_A=visible_x_min_A,
        visible_x_max_A=visible_x_max_A,
        visible_y_min_A=visible_y_min_A,
        visible_y_max_A=visible_y_max_A,
        oh_cutoff_A=oh_cutoff_values[0] if oh_cutoff_values else None,
        min_molecule_frames=min(min_molecule_frame_values) if min_molecule_frame_values else None,
    )


def _format_plot_density_units(units: str) -> str:
    return units.replace("Angstrom", "A").replace("cm^3", "cm$^3$")


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


def _density_heatmap_z_data(
    profile: DensityHeatmapProfile,
    *,
    quantity: str,
) -> tuple[np.ndarray, str, str]:
    """Return heatmap z-values, units, and a label prefix."""

    if quantity == "number":
        if profile.number_density is None or profile.number_density_units is None:
            raise ValueError(
                f"Number-density data unavailable for heatmap '{profile.species} {profile.plane.upper()}'."
            )
        if str(profile.units).strip().lower() == "g/cm^3":
            _mass, _mass_units, canonical_number, canonical_number_units = (
                canonicalize_density_units(
                    density=profile.density,
                    density_units=profile.units,
                    number_density=profile.number_density,
                    number_density_units=profile.number_density_units,
                )
            )
            assert canonical_number is not None
            assert canonical_number_units is not None
            return canonical_number, canonical_number_units, "Entity density"
        return (
            np.asarray(profile.number_density, dtype=float),
            str(profile.number_density_units),
            "Entity density",
        )

    if quantity == "mass":
        if str(profile.units).strip().lower() == "g/cm^3":
            density_values, units, _number, _number_units = canonicalize_density_units(
                density=profile.density,
                density_units=profile.units,
                number_density=profile.number_density,
                number_density_units=profile.number_density_units,
            )
            return density_values, units, "Density"
        return np.asarray(profile.density, dtype=float), str(profile.units), "Density"

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
    focus_nonzero_x: bool = True,
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

    x_focus = np.concatenate(nonzero_x) if focus_nonzero_x and nonzero_x else np.concatenate(all_x)
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


def _density_profile_visible_x_limits(
    profile: DensityProfile,
    *,
    x_mode: str,
) -> list[float] | None:
    normalized_x_mode = str(x_mode).strip().lower() or "distance"
    if normalized_x_mode != "distance":
        return None
    lower = profile.visible_x_min_A
    upper = profile.visible_x_max_A
    if lower is None or upper is None:
        return None
    if not (np.isfinite(float(lower)) and np.isfinite(float(upper))):
        return None
    lower_value = float(lower)
    upper_value = float(upper)
    if (
        profile.coordinate_mode != "distance"
        and profile.surface_position is not None
        and np.isfinite(float(profile.surface_position))
    ):
        lower_value -= float(profile.surface_position)
        upper_value -= float(profile.surface_position)
    return _expand_linear_limits(lower_value, upper_value)


def _density_profiles_visible_x_limits(
    profiles: list[DensityProfile],
    *,
    x_mode: str,
) -> list[float] | None:
    ranges = [
        value
        for profile in profiles
        if (value := _density_profile_visible_x_limits(profile, x_mode=x_mode)) is not None
    ]
    if not ranges:
        return None
    return _expand_linear_limits(
        min(float(value[0]) for value in ranges),
        max(float(value[1]) for value in ranges),
    )


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


def _has_active_normalization_overrides(
    overrides_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    if not overrides_by_id:
        return False
    for override in overrides_by_id.values():
        if not isinstance(override, Mapping):
            continue
        if str(override.get("normalization_mode") or "").strip().lower() not in {"", "none"}:
            return True
    return False


def plot_density_profile(
    profile: DensityProfile | DensityHeatmapProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    keep_figure_open: bool = False,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    x_mode: str = "distance",
    quantity: str = "mass",
    interface_view: str = "folded",
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
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    heatmap_log_scale: bool = False,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
) -> Path | None:
    """Plot and optionally save a density profile."""
    normalized_interface_view = str(interface_view).strip().lower()
    if normalized_interface_view not in {"full", "folded", "top", "bottom", "all"}:
        raise ValueError("interface_view must be full, folded, top, bottom, or all.")
    if normalized_interface_view == "all":
        raise ValueError(
            "interface_view='all' requires plot_density_profiles so the three views can be overlaid."
        )
    if isinstance(profile, DensityProfile) and normalized_interface_view in {"top", "bottom"}:
        side_density = (
            profile.top_density if normalized_interface_view == "top" else profile.bottom_density
        )
        side_number_density = (
            profile.top_number_density
            if normalized_interface_view == "top"
            else profile.bottom_number_density
        )
        side_counts = (
            profile.top_counts_per_frame
            if normalized_interface_view == "top"
            else profile.bottom_counts_per_frame
        )
        if side_density is None:
            raise ValueError(f"The profile does not contain a {normalized_interface_view} view.")
        profile = replace(
            profile,
            density=np.asarray(side_density, dtype=float),
            number_density=(
                None
                if side_number_density is None
                else np.asarray(side_number_density, dtype=float)
            ),
            counts_per_frame=(
                profile.counts_per_frame
                if side_counts is None
                else np.asarray(side_counts, dtype=float)
            ),
            interface_view=normalized_interface_view,
        )
    elif isinstance(profile, DensityHeatmapProfile) and normalized_interface_view in {
        "top",
        "bottom",
    }:
        side_density = (
            profile.top_density if normalized_interface_view == "top" else profile.bottom_density
        )
        side_number_density = (
            profile.top_number_density
            if normalized_interface_view == "top"
            else profile.bottom_number_density
        )
        if side_density is None:
            raise ValueError(f"The profile does not contain a {normalized_interface_view} view.")
        profile = replace(
            profile,
            density=np.asarray(side_density, dtype=float),
            number_density=(
                None
                if side_number_density is None
                else np.asarray(side_number_density, dtype=float)
            ),
            interface_view=normalized_interface_view,
        )
    mapping_x_mode = (
        "axis"
        if normalized_interface_view == "full" and isinstance(profile, DensityProfile)
        else x_mode
    )
    resolved_mapping = resolve_density_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
        view_type=str(getattr(view_mapping, "view_type_id", "") or "line_1d"),
        x_mode=mapping_x_mode,
        quantity=quantity,
    )
    runtime_x_mode = resolved_mapping.x_mode
    runtime_quantity = resolved_mapping.quantity
    if normalized_interface_view == "full" and isinstance(profile, DensityProfile):
        if profile.coordinate_mode == "distance":
            raise ValueError(
                "The selected density profile contains folded surface-distance bins. "
                "Load the raw surface-axis profile (X, Y, or Z) to plot the full cell."
            )
        runtime_x_mode = "axis"
    if (
        normalized_interface_view == "full"
        and isinstance(profile, DensityHeatmapProfile)
        and "distance" in {str(value).strip().lower() for value in profile.plane_axes}
    ):
        raise ValueError(
            "The selected density heatmap contains a folded distance axis. "
            "Load a raw Cartesian heatmap to plot the full cell."
        )
    if resolved_mapping.is_heatmap:
        if not isinstance(profile, DensityHeatmapProfile):
            raise ValueError("Density heatmap rendering requires a 2D density heatmap profile.")
        z_values, units, z_label_prefix = _density_heatmap_z_data(
            profile, quantity=runtime_quantity
        )
        units = _format_plot_density_units(units)
        display_species = _density_species_display_label(profile.species)
        default_title = f"{display_species} density heatmap ({profile.plane.upper()})"
        default_x_label = f"{profile.plane_axes[0].upper()} (A)"
        default_y_label = f"{profile.plane_axes[1].upper()} (A)"
        return plot_heatmap_series(
            np.asarray(profile.x_bin_edges, dtype=float),
            np.asarray(profile.y_bin_edges, dtype=float),
            np.asarray(z_values, dtype=float),
            title=title or default_title,
            x_label=resolve_explicit_plot_text(x_label, default_x_label),
            y_label=resolve_explicit_plot_text(y_label, default_y_label),
            output=output,
            show=show,
            show_blocking=show_blocking,
            keep_figure_open=keep_figure_open,
            preferred_backend=preferred_backend,
            style=style,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_tick_font_size=None,
            y_tick_font_size=None,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_pad=title_pad,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            capture_state=capture_state,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            grid_kwargs=grid_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            suppress_output_log=suppress_output_log,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_cmap=heatmap_cmap,
            heatmap_log_scale=heatmap_log_scale,
            heatmap_colorbar_enabled=heatmap_colorbar_enabled,
            heatmap_colorbar_label=(
                heatmap_colorbar_label
                if heatmap_colorbar_label is not None
                else f"{z_label_prefix} ({units})"
            ),
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
            annotations=annotations,
            capture_state_extra={
                "heatmap_quantity_label": f"{z_label_prefix} ({units})",
                "heatmap_plane": profile.plane,
            },
        )
    if not isinstance(profile, DensityProfile):
        raise ValueError("Density line rendering requires a 1D density profile.")
    x_values, default_x_label = _density_x_data(profile, x_mode=runtime_x_mode)
    density_values, units, y_label_prefix = _density_y_data(profile, quantity=runtime_quantity)
    units = _format_plot_density_units(units)
    resolved_line_label = line_label
    display_species = _density_species_display_label(profile.species)
    if resolved_line_label is None and legend:
        resolved_line_label = display_species
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
            labels=[resolved_line_label or display_species or "Series"],
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
            focus_nonzero_x=runtime_x_mode == "distance",
        )
        if x_lim is None:
            stored_x_lim = _density_profile_visible_x_limits(
                profile,
                x_mode=runtime_x_mode,
            )
            if stored_x_lim is not None:
                auto_x_lim = stored_x_lim
        resolved_x_lim = _merge_plot_limits(x_lim, auto_x_lim)
        resolved_y_lim = _merge_plot_limits(y_lim, auto_y_lim)
    stats_key = "number_density" if runtime_quantity.strip().lower() == "number" else "density"

    return plot_line_series(
        x_values,
        density_values,
        title=title or f"{display_species} density profile",
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, f"{y_label_prefix} ({units})"),
        output=output,
        show=show,
        show_blocking=show_blocking,
        keep_figure_open=keep_figure_open,
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
    profiles: Sequence[DensityProfile | DensityHeatmapProfile]
    | DensityProfile
    | DensityHeatmapProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    keep_figure_open: bool = False,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    x_mode: str = "distance",
    quantity: str = "mass",
    interface_view: str = "folded",
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
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    heatmap_log_scale: bool = False,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
) -> Path | None:
    """Plot one or more density profiles."""
    if isinstance(profiles, (DensityProfile, DensityHeatmapProfile)):
        profiles = [profiles]
    normalized_interface_view = str(interface_view).strip().lower()
    if normalized_interface_view not in {"full", "folded", "top", "bottom", "all"}:
        raise ValueError("interface_view must be full, folded, top, bottom, or all.")
    profile_view_names: list[str | None] = [None] * len(profiles)
    if normalized_interface_view == "full":
        profile_view_names = ["Full cell (unfolded)"] * len(profiles)
    elif normalized_interface_view in {"top", "bottom"}:
        side = normalized_interface_view
        resolved_profiles: list[DensityProfile | DensityHeatmapProfile] = []
        for item in profiles:
            if isinstance(item, DensityHeatmapProfile):
                side_density = item.top_density if side == "top" else item.bottom_density
                side_number_density = (
                    item.top_number_density if side == "top" else item.bottom_number_density
                )
                if side_density is None:
                    raise ValueError(f"The profile does not contain a {side} view.")
                resolved_profiles.append(
                    replace(
                        item,
                        density=np.asarray(side_density, dtype=float),
                        number_density=(
                            None
                            if side_number_density is None
                            else np.asarray(side_number_density, dtype=float)
                        ),
                        interface_view=side,
                    )
                )
                continue
            if str(item.coordinate_mode).strip().lower() != "distance":
                raise ValueError(
                    f"The requested {side} interface requires a surface-distance profile, "
                    f"but the active profile uses raw {str(item.axis).upper()} coordinates. "
                    "Use the full-cell representation for raw-axis data."
                )
            side_density = item.top_density if side == "top" else item.bottom_density
            if side_density is None:
                raise ValueError(f"The profile does not contain a {side} view.")
            resolved_profiles.append(
                replace(
                    item,
                    density=np.asarray(side_density, dtype=float),
                    number_density=(
                        item.top_number_density if side == "top" else item.bottom_number_density
                    ),
                    counts_per_frame=(
                        (
                            item.top_counts_per_frame
                            if side == "top"
                            else item.bottom_counts_per_frame
                        )
                        if (
                            item.top_counts_per_frame
                            if side == "top"
                            else item.bottom_counts_per_frame
                        )
                        is not None
                        else item.counts_per_frame
                    ),
                    interface_view=side,
                )
            )
        profiles = resolved_profiles
        axis_label = str(getattr(profiles[0], "axis", "z")).upper() if profiles else "Z"
        side_label = (
            f"Positive-axis (+{axis_label})" if side == "top" else f"Negative-axis (-{axis_label})"
        )
        profile_view_names = [side_label] * len(profiles)
    elif normalized_interface_view == "all":
        source_profile_count = len(profiles)
        view_tokens = ("folded", "top", "bottom")
        view_labels = (
            "Combined interfaces (folded mean)",
            "Positive-axis interface",
            "Negative-axis interface",
        )

        def expand_source_values(values: list[Any] | None) -> list[Any] | None:
            if values is None or len(values) != source_profile_count:
                return values
            return [value for value in values for _view in view_tokens]

        original_series_ids = None if series_ids is None else list(series_ids)
        base_ids: list[str] | None = None
        if original_series_ids is not None and len(original_series_ids) == source_profile_count:
            base_ids = [str(value) for value in original_series_ids]
        elif (
            render_series_descriptors is not None
            and len(render_series_descriptors) == source_profile_count
        ):
            base_ids = [
                str(descriptor.get("series_id") or f"density:{index}")
                for index, descriptor in enumerate(render_series_descriptors)
            ]

        expanded_profiles: list[DensityProfile | DensityHeatmapProfile] = []
        expanded_views: list[str | None] = []
        for item in profiles:
            if not isinstance(item, DensityProfile):
                raise ValueError("interface_view='all' is only available for 1D density profiles.")
            if str(item.coordinate_mode).strip().lower() != "distance":
                raise ValueError(
                    "Comparing interfaces requires surface-distance profiles, but the active "
                    f"profile uses raw {str(item.axis).upper()} coordinates."
                )
            if item.top_density is None or item.bottom_density is None:
                raise ValueError("The density profile does not contain both interface views.")
            expanded_profiles.append(replace(item, interface_view="folded"))
            expanded_views.append("Combined interfaces (folded mean)")
            for side in ("top", "bottom"):
                side_density = item.top_density if side == "top" else item.bottom_density
                side_number_density = (
                    item.top_number_density if side == "top" else item.bottom_number_density
                )
                side_counts = (
                    item.top_counts_per_frame if side == "top" else item.bottom_counts_per_frame
                )
                expanded_profiles.append(
                    replace(
                        item,
                        density=np.asarray(side_density, dtype=float),
                        number_density=(
                            None
                            if side_number_density is None
                            else np.asarray(side_number_density, dtype=float)
                        ),
                        counts_per_frame=(
                            item.counts_per_frame
                            if side_counts is None
                            else np.asarray(side_counts, dtype=float)
                        ),
                        interface_view=side,
                    )
                )
                axis_label = str(getattr(item, "axis", "z")).upper()
                expanded_views.append(
                    f"Positive-axis (+{axis_label})"
                    if side == "top"
                    else f"Negative-axis (-{axis_label})"
                )
        profiles = expanded_profiles
        profile_view_names = expanded_views
        if series_labels is not None and len(series_labels) == source_profile_count:
            series_labels = [
                f"{label} ({view_label})" for label in series_labels for view_label in view_labels
            ]
        if base_ids is not None:
            series_ids = [
                f"{series_id}::interface:{view_token}"
                for series_id in base_ids
                for view_token in view_tokens
            ]
        else:
            series_ids = expand_source_values(series_ids)
        line_colors = expand_source_values(line_colors)
        series_error_configs = expand_source_values(series_error_configs)
        series_enabled = expand_source_values(series_enabled)
        series_show_in_legend = expand_source_values(series_show_in_legend)
        series_line_widths = expand_source_values(series_line_widths)
        series_markers = expand_source_values(series_markers)
        series_fit_configs = expand_source_values(series_fit_configs)
        series_cumulative_configs = expand_source_values(series_cumulative_configs)
        series_normalization_modes = expand_source_values(series_normalization_modes)
        series_normalization_values = expand_source_values(series_normalization_values)
        series_normalization_x_refs = expand_source_values(series_normalization_x_refs)
        if series_line_kwargs is None:
            series_line_kwargs = [
                {"linestyle": line_style}
                for _profile in range(source_profile_count)
                for line_style in ("-", "--", ":")
            ]
        elif len(series_line_kwargs) == source_profile_count:
            expanded_line_kwargs: list[dict[str, Any] | None] = []
            for value in series_line_kwargs:
                for line_style in ("-", "--", ":"):
                    resolved_value = dict(value or {})
                    resolved_value.setdefault("linestyle", line_style)
                    expanded_line_kwargs.append(resolved_value)
            series_line_kwargs = expanded_line_kwargs
        if (
            render_series_descriptors is not None
            and len(render_series_descriptors) == source_profile_count
        ):
            expanded_descriptors: list[dict[str, Any]] = []
            for index, descriptor in enumerate(render_series_descriptors):
                source_id = (
                    base_ids[index]
                    if base_ids is not None
                    else str(descriptor.get("series_id") or f"density:{index}")
                )
                for view_token, view_label in zip(view_tokens, view_labels):
                    expanded_descriptor = dict(descriptor)
                    expanded_id = f"{source_id}::interface:{view_token}"
                    expanded_descriptor["series_id"] = expanded_id
                    # Each expanded profile is a real source entry for the
                    # generic multi-line renderer.  Keep the unexpanded id as
                    # lineage metadata instead of pointing source_series_id at
                    # data that no longer exists in the expanded source map.
                    expanded_descriptor["source_series_id"] = expanded_id
                    expanded_descriptor["parent_series_id"] = source_id
                    expanded_descriptor["interface_view"] = view_token
                    default_label = str(descriptor.get("default_label") or f"Series {index + 1}")
                    expanded_descriptor["default_label"] = f"{default_label} ({view_label})"
                    expanded_descriptors.append(expanded_descriptor)
            render_series_descriptors = expanded_descriptors
        if series_overrides_by_id is not None and base_ids is not None:
            expanded_overrides = dict(series_overrides_by_id)
            for source_id in base_ids:
                source_override = series_overrides_by_id.get(source_id)
                if not isinstance(source_override, dict):
                    continue
                for view_token, view_label in zip(view_tokens, view_labels):
                    expanded_override = dict(source_override)
                    label_override = expanded_override.get("label_override")
                    if label_override is not None and str(label_override).strip():
                        expanded_override["label_override"] = (
                            f"{str(label_override).strip()} ({view_label})"
                        )
                    expanded_overrides[f"{source_id}::interface:{view_token}"] = expanded_override
            series_overrides_by_id = expanded_overrides
    if not profiles:
        raise ValueError("At least one density profile is required.")
    first_profile = profiles[0]
    mapping_x_mode = (
        "axis"
        if normalized_interface_view == "full" and isinstance(first_profile, DensityProfile)
        else x_mode
    )
    resolved_mapping = resolve_density_plot_mapping(
        contract=data_contract,
        profile=first_profile,
        mapping=view_mapping,
        view_type=str(getattr(view_mapping, "view_type_id", "") or "line_1d"),
        x_mode=mapping_x_mode,
        quantity=quantity,
    )
    runtime_x_mode = resolved_mapping.x_mode
    runtime_quantity = resolved_mapping.quantity
    if normalized_interface_view == "full":
        for item in profiles:
            if isinstance(item, DensityProfile) and item.coordinate_mode == "distance":
                raise ValueError(
                    "The selected density profile contains folded surface-distance bins. "
                    "Load the raw surface-axis profile (X, Y, or Z) to plot the full cell."
                )
            if isinstance(item, DensityHeatmapProfile) and "distance" in {
                str(value).strip().lower() for value in item.plane_axes
            }:
                raise ValueError(
                    "The selected density heatmap contains a folded distance axis. "
                    "Load a raw Cartesian heatmap to plot the full cell."
                )
        runtime_x_mode = "axis"
    if resolved_mapping.is_heatmap:
        if len(profiles) != 1:
            raise ValueError(
                "Density heatmap rendering currently supports one selected heatmap field at a time."
            )
        return plot_density_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            keep_figure_open=keep_figure_open,
            preferred_backend=preferred_backend,
            style=style,
            data_contract=resolved_mapping.contract,
            view_mapping=resolved_mapping.mapping,
            quantity=runtime_quantity,
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
            annotations=annotations,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            grid_kwargs=grid_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_cmap=heatmap_cmap,
            heatmap_log_scale=heatmap_log_scale,
            heatmap_colorbar_enabled=heatmap_colorbar_enabled,
            heatmap_colorbar_label=heatmap_colorbar_label,
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
        )
    line_profiles = cast(list[DensityProfile], profiles)
    default_labels = [
        (
            _density_species_display_label(profile.species)
            if view_name is None
            else f"{_density_species_display_label(profile.species)} ({view_name})"
        )
        for profile, view_name in zip(line_profiles, profile_view_names)
    ]
    labels = resolve_series_labels(
        default_labels,
        series_labels,
        series_kind="density",
    )

    if not use_multi_series_plot(
        profile_count=len(line_profiles),
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
    ):
        return plot_density_profile(
            line_profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            keep_figure_open=keep_figure_open,
            preferred_backend=preferred_backend,
            style=style,
            data_contract=resolved_mapping.contract,
            view_mapping=resolved_mapping.mapping,
            x_mode=runtime_x_mode,
            quantity=runtime_quantity,
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

    if runtime_x_mode in {"x", "y", "z"}:
        compat = [
            i
            for i, p in enumerate(line_profiles)
            if _density_profile_visible_for_x_mode(p, x_mode=runtime_x_mode)
        ]
        if not compat:
            raise ValueError(
                f"No density profiles match the selected axis '{runtime_x_mode.upper()}'."
            )

        def _pick(lst: list[Any] | None) -> list[Any] | None:
            return None if lst is None else [lst[i] for i in compat]

        line_profiles = [line_profiles[index] for index in compat]
        labels = [labels[index] for index in compat]
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
            render_series_descriptors = _pick(render_series_descriptors)

    first = line_profiles[0]
    if runtime_x_mode == "distance" and any(
        not _profile_has_surface_reference(profile) for profile in line_profiles
    ):
        LOGGER.warning(
            "At least one profile has no surface reference; combined plot falls back to axis coordinates."
        )
        runtime_x_mode = "axis"

    y_resolved = [_density_y_data(profile, quantity=runtime_quantity) for profile in line_profiles]
    y_units = y_resolved[0][1]
    y_label_prefix = y_resolved[0][2]
    if any(units != y_units for _, units, _ in y_resolved[1:]):
        unified = _unify_number_density_units([units for _, units, _ in y_resolved])
        if unified is not None:
            y_units = unified
        else:
            raise ValueError("All density profiles must use the same units for combined plotting.")
    y_series = [values for values, _, _ in y_resolved]
    x_series = [_density_x_data(profile, x_mode=runtime_x_mode)[0] for profile in line_profiles]
    default_x_label = _density_x_data(first, x_mode=runtime_x_mode)[1]
    display_units = _format_plot_density_units(y_units)
    resolved_x_lim = x_lim
    resolved_y_lim = y_lim
    if not _has_active_normalization(
        series_normalization_modes,
        enabled=series_enabled,
    ) and not _has_active_normalization_overrides(series_overrides_by_id):
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
            focus_nonzero_x=runtime_x_mode == "distance",
        )
        if x_lim is None:
            stored_x_lim = _density_profiles_visible_x_limits(
                line_profiles,
                x_mode=runtime_x_mode,
            )
            if stored_x_lim is not None:
                auto_x_lim = stored_x_lim
        resolved_x_lim = _merge_plot_limits(x_lim, auto_x_lim)
        resolved_y_lim = _merge_plot_limits(y_lim, auto_y_lim)
    stats_key = "number_density" if runtime_quantity.strip().lower() == "number" else "density"

    return plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or "Density profile",
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, f"{y_label_prefix} ({display_units})"),
        output=output,
        show=show,
        show_blocking=show_blocking,
        keep_figure_open=keep_figure_open,
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
            for profile in line_profiles
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


def _finish_geometry_groups(groups, *, include_paired: bool = False):
    if groups is None:
        return None
    result = []
    for group in groups.values():
        keys = sorted(group["counts"])
        top_counts = group.get("top_counts", {})
        bottom_counts = group.get("bottom_counts", {})
        payload = dict(
            cell=group["cell"],
            origin=group["origin"],
            shift=group["shift"],
            n_frames=group["n_frames"],
            flat_indices=np.asarray(keys, dtype=np.uint64),
            mass_sum_g=np.asarray([group["counts"][k][0] for k in keys]),
            entity_sum=np.asarray([group["counts"][k][1] for k in keys]),
        )
        if group.get("canonical_period") is not None:
            payload.update(
                top_shift=group["top_shift"],
                bottom_shift=group["bottom_shift"],
                canonical_cut=group["canonical_cut"],
                canonical_lower=group["canonical_lower"],
                canonical_period=group["canonical_period"],
                canonical_wrap_threshold=group.get("canonical_wrap_threshold"),
            )
        if include_paired:
            payload.update(
                top_shift=group.get("top_shift", group["shift"]),
                bottom_shift=group.get("bottom_shift", group["shift"]),
                top_mass_sum_g=np.asarray([top_counts.get(k, (0.0, 0.0))[0] for k in keys]),
                top_entity_sum=np.asarray([top_counts.get(k, (0.0, 0.0))[1] for k in keys]),
                bottom_mass_sum_g=np.asarray([bottom_counts.get(k, (0.0, 0.0))[0] for k in keys]),
                bottom_entity_sum=np.asarray([bottom_counts.get(k, (0.0, 0.0))[1] for k in keys]),
            )
        result.append(payload)
    return result


def _geometry_group_datasets(groups):
    if groups is None:
        return {}
    offsets = np.cumsum([0] + [len(g["flat_indices"]) for g in groups])
    datasets = {
        "geometry_cells_A": np.asarray([g["cell"] for g in groups]),
        "geometry_origins_A": np.asarray([g["origin"] for g in groups]),
        "geometry_surface_offsets_A": np.asarray([g["shift"] for g in groups]),
        "geometry_frame_counts": np.asarray([g["n_frames"] for g in groups], dtype=np.int64),
        "geometry_offsets": offsets,
        "geometry_flat_indices": np.concatenate([g["flat_indices"] for g in groups]),
        "geometry_mass_sum_g": np.concatenate([g["mass_sum_g"] for g in groups]),
        "geometry_entity_sum": np.concatenate([g["entity_sum"] for g in groups]),
    }
    if groups and "top_shift" in groups[0]:
        datasets.update(
            geometry_top_surface_offsets_A=np.asarray([g["top_shift"] for g in groups]),
            geometry_bottom_surface_offsets_A=np.asarray([g["bottom_shift"] for g in groups]),
        )
    if groups and "top_mass_sum_g" in groups[0]:
        datasets.update(
            geometry_top_mass_sum_g=np.concatenate([g["top_mass_sum_g"] for g in groups]),
            geometry_top_entity_sum=np.concatenate([g["top_entity_sum"] for g in groups]),
            geometry_bottom_mass_sum_g=np.concatenate([g["bottom_mass_sum_g"] for g in groups]),
            geometry_bottom_entity_sum=np.concatenate([g["bottom_entity_sum"] for g in groups]),
        )
    if groups and "canonical_period" in groups[0]:
        datasets.update(
            geometry_axis_cuts_A=np.asarray([g["canonical_cut"] for g in groups]),
            geometry_axis_lowers_A=np.asarray([g["canonical_lower"] for g in groups]),
            geometry_axis_periods_A=np.asarray([g["canonical_period"] for g in groups]),
        )
        if all(g.get("canonical_wrap_threshold") is not None for g in groups):
            datasets["geometry_axis_wrap_thresholds_A"] = np.asarray(
                [g["canonical_wrap_threshold"] for g in groups]
            )
    return datasets


def _load_geometry_groups(datasets):
    if "geometry_cells_A" not in datasets:
        return None
    side_offset_names = {
        "geometry_top_surface_offsets_A",
        "geometry_bottom_surface_offsets_A",
    }
    present_side_offset_names = side_offset_names.intersection(datasets)
    if present_side_offset_names and present_side_offset_names != side_offset_names:
        missing = ", ".join(sorted(side_offset_names - present_side_offset_names))
        raise ValueError(
            "Density sparse-grid geometry has incomplete paired surface offsets; "
            f"missing: {missing}. Recompute the density grid."
        )
    has_side_offsets = present_side_offset_names == side_offset_names
    paired_names = {
        "geometry_top_mass_sum_g",
        "geometry_top_entity_sum",
        "geometry_bottom_mass_sum_g",
        "geometry_bottom_entity_sum",
    }
    present_paired_names = paired_names.intersection(datasets)
    if present_paired_names and present_paired_names != paired_names:
        missing = ", ".join(sorted(paired_names - present_paired_names))
        raise ValueError(
            "Density sparse-grid geometry has incomplete paired-interface data; "
            f"missing: {missing}. Recompute the density grid."
        )
    has_paired_geometry = present_paired_names == paired_names
    canonical_names = {
        "geometry_axis_cuts_A",
        "geometry_axis_lowers_A",
        "geometry_axis_periods_A",
    }
    present_canonical_names = canonical_names.intersection(datasets)
    if present_canonical_names and present_canonical_names != canonical_names:
        missing = ", ".join(sorted(canonical_names - present_canonical_names))
        raise ValueError(
            "Density sparse-grid geometry has incomplete canonical-axis data; "
            f"missing: {missing}. Recompute the density grid."
        )
    has_canonical_geometry = present_canonical_names == canonical_names
    groups = []
    for i, cell in enumerate(datasets["geometry_cells_A"]):
        start, stop = datasets["geometry_offsets"][i : i + 2]
        groups.append(
            dict(
                cell=cell,
                origin=datasets["geometry_origins_A"][i],
                shift=float(datasets["geometry_surface_offsets_A"][i]),
                top_shift=(
                    float(datasets["geometry_top_surface_offsets_A"][i])
                    if has_side_offsets
                    else None
                ),
                bottom_shift=(
                    float(datasets["geometry_bottom_surface_offsets_A"][i])
                    if has_side_offsets
                    else None
                ),
                canonical_cut=(
                    float(datasets["geometry_axis_cuts_A"][i]) if has_canonical_geometry else None
                ),
                canonical_lower=(
                    float(datasets["geometry_axis_lowers_A"][i]) if has_canonical_geometry else None
                ),
                canonical_period=(
                    float(datasets["geometry_axis_periods_A"][i])
                    if has_canonical_geometry
                    else None
                ),
                canonical_wrap_threshold=(
                    float(datasets["geometry_axis_wrap_thresholds_A"][i])
                    if "geometry_axis_wrap_thresholds_A" in datasets
                    else None
                ),
                n_frames=int(datasets["geometry_frame_counts"][i]),
                flat_indices=datasets["geometry_flat_indices"][start:stop],
                mass_sum_g=datasets["geometry_mass_sum_g"][start:stop],
                entity_sum=datasets["geometry_entity_sum"][start:stop],
                top_mass_sum_g=(
                    datasets["geometry_top_mass_sum_g"][start:stop] if has_paired_geometry else None
                ),
                top_entity_sum=(
                    datasets["geometry_top_entity_sum"][start:stop] if has_paired_geometry else None
                ),
                bottom_mass_sum_g=(
                    datasets["geometry_bottom_mass_sum_g"][start:stop]
                    if has_paired_geometry
                    else None
                ),
                bottom_entity_sum=(
                    datasets["geometry_bottom_entity_sum"][start:stop]
                    if has_paired_geometry
                    else None
                ),
            )
        )
    return groups


def _merge_bound(
    lower: float,
    upper: float,
    limits: tuple[float | None, float | None],
    *,
    shift: float = 0.0,
) -> tuple[float, float]:
    requested_lower, requested_upper = limits
    if requested_lower is not None:
        lower = max(lower, float(requested_lower) + shift)
    if requested_upper is not None:
        upper = min(upper, float(requested_upper) + shift)
    return lower, upper


def _merge_distance_bound(
    lower: float,
    upper: float,
    limits: tuple[float | None, float | None],
    *,
    shift: float,
    side: str,
) -> tuple[float, float]:
    """Intersect Cartesian normal bounds with an outward-distance interval."""

    requested_lower, requested_upper = limits
    if side == "bottom":
        if requested_upper is not None:
            lower = max(lower, shift - float(requested_upper))
        if requested_lower is not None:
            upper = min(upper, shift - float(requested_lower))
        return lower, upper
    return _merge_bound(lower, upper, limits, shift=shift)


def _clip_polygon_coordinate(
    polygon: np.ndarray,
    *,
    coordinate: int,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    """Clip a convex 2D polygon against one axis-aligned half-plane."""

    if polygon.size == 0 or not np.isfinite(boundary):
        return polygon
    output: list[np.ndarray] = []

    def inside(point: np.ndarray) -> bool:
        value = float(point[coordinate])
        return value >= boundary if keep_greater else value <= boundary

    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = float(current[coordinate] - previous[coordinate])
            if abs(delta) > 1.0e-15:
                fraction = (boundary - float(previous[coordinate])) / delta
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=float).reshape(-1, 2)


def _polygon_rectangle_intersection_area(
    polygon: np.ndarray,
    bounds: np.ndarray,
) -> float:
    clipped = np.asarray(polygon, dtype=float)
    for coordinate in range(2):
        clipped = _clip_polygon_coordinate(
            clipped,
            coordinate=coordinate,
            boundary=float(bounds[coordinate, 0]),
            keep_greater=True,
        )
        clipped = _clip_polygon_coordinate(
            clipped,
            coordinate=coordinate,
            boundary=float(bounds[coordinate, 1]),
            keep_greater=False,
        )
        if clipped.shape[0] < 3:
            return 0.0
    x_values = clipped[:, 0]
    y_values = clipped[:, 1]
    return 0.5 * abs(
        float(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))
    )


def _intersect_1d_intervals(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for left_lower, left_upper in left:
        for right_lower, right_upper in right:
            lower = max(left_lower, right_lower)
            upper = min(left_upper, right_upper)
            if upper > lower:
                result.append((lower, upper))
    return result


def _periodic_interval_segments(
    lower: float,
    upper: float,
    *,
    period: float,
) -> list[tuple[float, float]]:
    """Map one continuous interval onto ``[0, period]`` without approximation."""

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return []
    width = upper - lower
    if width >= period - 1.0e-12:
        return [(0.0, period)]
    start = float(np.mod(lower, period))
    stop = start + width
    if stop <= period + 1.0e-12:
        return [(start, min(stop, period))]
    return [(start, period), (0.0, stop - period)]


def _canonical_raw_axis_intervals(
    group: Mapping[str, Any],
    limits: tuple[float | None, float | None],
) -> list[tuple[float, float]]:
    period = float(group["canonical_period"])
    lower = float(group["canonical_lower"])
    cut = float(group["canonical_cut"])
    requested_lower, requested_upper = limits
    raw_lower = lower if requested_lower is None else max(lower, float(requested_lower))
    raw_upper = (
        lower + period if requested_upper is None else min(lower + period, float(requested_upper))
    )
    return _periodic_interval_segments(
        raw_lower - lower - cut,
        raw_upper - lower - cut,
        period=period,
    )


def _canonical_distance_intervals(
    profile: DensityGridProfile,
    group: Mapping[str, Any],
    limits: tuple[float | None, float | None],
    *,
    distance_side: str,
) -> list[tuple[float, float]]:
    period = float(group["canonical_period"])
    top = float(group.get("top_shift", group.get("shift", 0.0)))
    bottom = float(group.get("bottom_shift", group.get("shift", 0.0)))
    requested_lower, requested_upper = limits
    layout = (
        profile.surface_reference.resolved_layout
        if profile.surface_reference is not None
        else "axis"
    )

    if layout == "single-top":
        threshold_value = group.get("canonical_wrap_threshold")
        wrap_threshold = float(bottom if threshold_value is None else threshold_value)
        distance_lower = wrap_threshold - top
        distance_upper = period + distance_lower
        lower = (
            distance_lower
            if requested_lower is None
            else max(distance_lower, float(requested_lower))
        )
        upper = (
            distance_upper
            if requested_upper is None
            else min(distance_upper, float(requested_upper))
        )
        return _periodic_interval_segments(top + lower, top + upper, period=period)
    if layout == "single-bottom":
        threshold_value = group.get("canonical_wrap_threshold")
        wrap_threshold = float(top if threshold_value is None else threshold_value)
        distance_lower = bottom - wrap_threshold
        distance_upper = period + distance_lower
        lower = (
            distance_lower
            if requested_lower is None
            else max(distance_lower, float(requested_lower))
        )
        upper = (
            distance_upper
            if requested_upper is None
            else min(distance_upper, float(requested_upper))
        )
        return _periodic_interval_segments(bottom - upper, bottom - lower, period=period)

    if layout == "symmetric":
        midplane = 0.5 * (top + bottom)
        if distance_side == "bottom":
            coordinate_lower, coordinate_upper = 0.0, midplane
            if requested_upper is not None:
                coordinate_lower = max(coordinate_lower, bottom - float(requested_upper))
            if requested_lower is not None:
                coordinate_upper = min(coordinate_upper, bottom - float(requested_lower))
        else:
            coordinate_lower, coordinate_upper = midplane, period
            if requested_lower is not None:
                coordinate_lower = max(coordinate_lower, top + float(requested_lower))
            if requested_upper is not None:
                coordinate_upper = min(coordinate_upper, top + float(requested_upper))
        if coordinate_upper <= coordinate_lower:
            return []
        return [(coordinate_lower, coordinate_upper)]
    return []


def _canonical_normal_overlap(
    profile: DensityGridProfile,
    group: Mapping[str, Any],
    *,
    raw_limits: tuple[float | None, float | None],
    distance_limits: tuple[float | None, float | None] | None,
    distance_side: str,
) -> float:
    intervals = _canonical_raw_axis_intervals(group, raw_limits)
    if distance_limits is not None:
        intervals = _intersect_1d_intervals(
            intervals,
            _canonical_distance_intervals(
                profile,
                group,
                distance_limits,
                distance_side=distance_side,
            ),
        )
    return float(sum(upper - lower for lower, upper in intervals))


def _prismatic_geometry_volumes(
    profile: DensityGridProfile,
    group: Mapping[str, Any],
    axes: tuple[str, ...],
    edges: tuple[np.ndarray, ...],
    filters: Mapping[str, tuple[float | None, float | None]],
    *,
    area_cache: dict[tuple[float, ...], np.ndarray],
    distance_side: str = "top",
) -> np.ndarray | None:
    """Return exact projected volumes for a decoupled surface-normal prism."""

    normal_axis = str(profile.surface_axis).strip().lower() or "z"
    normal_index = ("x", "y", "z").index(normal_axis)
    matrix = cell_matrix(group["cell"])
    origin = np.asarray(group["origin"], dtype=float)
    if not is_axis_prismatic(matrix, normal_index):
        return None

    shape = tuple(len(edge) - 1 for edge in edges)
    extent = cartesian_bounds(matrix, origin)
    normal_cell_lower = float(extent[normal_index, 0])
    normal_cell_upper = float(extent[normal_index, 1])
    normal_height = normal_cell_upper - normal_cell_lower
    if not np.isfinite(normal_height) or normal_height <= 0.0:
        return np.zeros(shape, dtype=float)

    normal_dimensions = [
        dimension
        for dimension, axis in enumerate(axes)
        if axis == "distance" or axis == normal_axis
    ]
    if len(normal_dimensions) > 1:
        # Current public mappings do not combine raw surface-axis and distance
        # as two displayed dimensions.  Keep the exact generic fallback for a
        # future mapping that does.
        return None

    area_shape = tuple(
        1 if dimension in normal_dimensions else size for dimension, size in enumerate(shape)
    )
    area_key = tuple(matrix.ravel()) + tuple(origin) + tuple(float(value) for value in area_shape)
    area = area_cache.get(area_key)
    if area is None:
        area = np.zeros(area_shape, dtype=float)
        in_plane_indices = [index for index in range(3) if index != normal_index]
        vector_a = matrix[in_plane_indices[0], in_plane_indices]
        vector_b = matrix[in_plane_indices[1], in_plane_indices]
        origin_2d = origin[in_plane_indices]
        footprint = np.asarray(
            [
                origin_2d,
                origin_2d + vector_a,
                origin_2d + vector_a + vector_b,
                origin_2d + vector_b,
            ],
            dtype=float,
        )
        for index in np.ndindex(area_shape):
            bounds = np.asarray([[-np.inf, np.inf]] * 2, dtype=float)
            for axis, limits in filters.items():
                if axis == "distance" or axis == normal_axis:
                    continue
                coordinate_index = ("x", "y", "z").index(axis)
                plane_index = in_plane_indices.index(coordinate_index)
                bounds[plane_index] = _merge_bound(
                    float(bounds[plane_index, 0]),
                    float(bounds[plane_index, 1]),
                    limits,
                )
            for dimension, (axis, edge) in enumerate(zip(axes, edges)):
                if dimension in normal_dimensions:
                    continue
                coordinate_index = ("x", "y", "z").index(axis)
                plane_index = in_plane_indices.index(coordinate_index)
                bounds[plane_index] = _merge_bound(
                    float(bounds[plane_index, 0]),
                    float(bounds[plane_index, 1]),
                    (float(edge[index[dimension]]), float(edge[index[dimension] + 1])),
                )
            area[index] = _polygon_rectangle_intersection_area(footprint, bounds)
        area_cache[area_key] = area

    canonical_geometry = all(
        group.get(name) is not None
        for name in ("canonical_cut", "canonical_lower", "canonical_period")
    )
    distance_relevant = "distance" in axes or "distance" in filters
    if canonical_geometry and distance_relevant:
        raw_filter = filters.get(normal_axis, (None, None))
        distance_filter = filters.get("distance")
        if not normal_dimensions:
            overlap = _canonical_normal_overlap(
                profile,
                group,
                raw_limits=raw_filter,
                distance_limits=distance_filter,
                distance_side=distance_side,
            )
            return np.broadcast_to(area * overlap, shape).copy()

        dimension = normal_dimensions[0]
        axis = axes[dimension]
        edge = np.asarray(edges[dimension], dtype=float)
        overlap_1d = np.zeros(edge.size - 1, dtype=float)
        for index, (edge_lower, edge_upper) in enumerate(zip(edge[:-1], edge[1:])):
            bin_raw_limits = raw_filter
            bin_distance_limits = distance_filter
            if axis == "distance":
                bin_distance_limits = (float(edge_lower), float(edge_upper))
                if distance_filter is not None:
                    bin_distance_limits = (
                        max(
                            float(edge_lower),
                            -np.inf if distance_filter[0] is None else float(distance_filter[0]),
                        ),
                        min(
                            float(edge_upper),
                            np.inf if distance_filter[1] is None else float(distance_filter[1]),
                        ),
                    )
            else:
                bin_raw_limits = (
                    max(
                        float(edge_lower),
                        -np.inf if raw_filter[0] is None else float(raw_filter[0]),
                    ),
                    min(
                        float(edge_upper),
                        np.inf if raw_filter[1] is None else float(raw_filter[1]),
                    ),
                )
            overlap_1d[index] = _canonical_normal_overlap(
                profile,
                group,
                raw_limits=bin_raw_limits,
                distance_limits=bin_distance_limits,
                distance_side=distance_side,
            )
        overlap_shape = [1] * len(shape)
        overlap_shape[dimension] = overlap_1d.size
        overlap = overlap_1d.reshape(overlap_shape)
        return np.broadcast_to(area * overlap, shape).copy()

    normal_lower = normal_cell_lower
    normal_upper = normal_cell_upper
    legacy_raw_filter = filters.get(normal_axis)
    if legacy_raw_filter is not None:
        normal_lower, normal_upper = _merge_bound(
            normal_lower,
            normal_upper,
            legacy_raw_filter,
        )
    distance_filter = filters.get("distance")
    shift_key = "bottom_shift" if distance_side == "bottom" else "top_shift"
    shift_value = group.get(shift_key, group.get("shift", 0.0))
    shift = float(group.get("shift", 0.0) if shift_value is None else shift_value)
    if distance_filter is not None:
        normal_lower, normal_upper = _merge_distance_bound(
            normal_lower,
            normal_upper,
            distance_filter,
            shift=shift,
            side=distance_side,
        )

    if not normal_dimensions:
        overlap = max(0.0, normal_upper - normal_lower)
        return np.broadcast_to(area * overlap, shape).copy()

    dimension = normal_dimensions[0]
    axis = axes[dimension]
    edge = np.asarray(edges[dimension], dtype=float)
    if axis == "distance" and distance_side == "bottom":
        edge_lower = shift - edge[1:]
        edge_upper = shift - edge[:-1]
    else:
        edge_shift = shift if axis == "distance" else 0.0
        edge_lower = edge[:-1] + edge_shift
        edge_upper = edge[1:] + edge_shift
    lower = np.maximum(edge_lower, normal_lower)
    upper = np.minimum(edge_upper, normal_upper)
    overlap_1d = np.clip(upper - lower, 0.0, None)
    overlap_shape = [1] * len(shape)
    overlap_shape[dimension] = overlap_1d.size
    overlap = overlap_1d.reshape(overlap_shape)
    return np.broadcast_to(area * overlap, shape).copy()


def _generic_geometry_volumes(
    profile: DensityGridProfile,
    group: Mapping[str, Any],
    axes: tuple[str, ...],
    edges: tuple[np.ndarray, ...],
    filters: Mapping[str, tuple[float | None, float | None]],
    *,
    distance_side: str = "top",
) -> np.ndarray:
    shape = tuple(len(edge) - 1 for edge in edges)
    volumes = np.zeros(shape, dtype=float)
    extent = cartesian_bounds(group["cell"], group["origin"])
    for index in np.ndindex(shape):
        bounds = np.asarray([[-np.inf, np.inf]] * 3, dtype=float)
        constraints = list(filters.items()) + [
            (axis, (edge[idx], edge[idx + 1])) for axis, edge, idx in zip(axes, edges, index)
        ]
        for axis, limits in constraints:
            coordinate = profile.surface_axis if axis == "distance" else axis
            axis_index = ("x", "y", "z").index(coordinate)
            if axis == "distance":
                shift_key = "bottom_shift" if distance_side == "bottom" else "top_shift"
                shift_value = group.get(shift_key, group.get("shift", 0.0))
                shift = float(group.get("shift", 0.0) if shift_value is None else shift_value)
                bounds[axis_index] = _merge_distance_bound(
                    float(bounds[axis_index, 0]),
                    float(bounds[axis_index, 1]),
                    limits,
                    shift=shift,
                    side=distance_side,
                )
            else:
                bounds[axis_index] = _merge_bound(
                    float(bounds[axis_index, 0]),
                    float(bounds[axis_index, 1]),
                    limits,
                )
        if np.any(bounds[:, 1] <= extent[:, 0]) or np.any(bounds[:, 0] >= extent[:, 1]):
            continue
        volumes[index] = intersection_volume(group["cell"], bounds, group["origin"])
    return volumes


def _coalesced_density_projection_groups(
    groups: list[Mapping[str, Any]],
    *,
    mass_key: str,
    entity_key: str,
    distance_relevant: bool,
    distance_side: str,
) -> list[dict[str, Any]]:
    """Merge exact duplicate projection geometries before sparse binning."""

    buckets: dict[tuple[object, ...], dict[str, Any]] = {}
    for group in groups:
        key: tuple[object, ...] = tuple(np.asarray(group["cell"], dtype=float).ravel()) + tuple(
            np.asarray(group["origin"], dtype=float)
        )
        if distance_relevant:
            shift_key = "bottom_shift" if distance_side == "bottom" else "top_shift"
            shift_value = group.get(shift_key, group.get("shift", 0.0))
            key += (
                distance_side,
                float(group.get("shift", 0.0) if shift_value is None else shift_value),
                group.get("top_shift"),
                group.get("bottom_shift"),
                group.get("canonical_cut"),
                group.get("canonical_lower"),
                group.get("canonical_period"),
                group.get("canonical_wrap_threshold"),
            )
        bucket = buckets.get(key)
        if bucket is None:
            bucket = dict(group)
            bucket["n_frames"] = 0
            bucket["projection_flat_chunks"] = []
            bucket["projection_mass_chunks"] = []
            bucket["projection_entity_chunks"] = []
            buckets[key] = bucket
        bucket["n_frames"] += int(group["n_frames"])
        bucket["projection_flat_chunks"].append(np.asarray(group["flat_indices"], dtype=np.uint64))
        bucket["projection_mass_chunks"].append(np.asarray(group[mass_key], dtype=float))
        bucket["projection_entity_chunks"].append(np.asarray(group[entity_key], dtype=float))

    result: list[dict[str, Any]] = []
    for bucket in buckets.values():
        flat = np.concatenate(bucket.pop("projection_flat_chunks"))
        mass = np.concatenate(bucket.pop("projection_mass_chunks"))
        entity = np.concatenate(bucket.pop("projection_entity_chunks"))
        if flat.size:
            unique_flat, inverse = np.unique(flat, return_inverse=True)
            mass = np.bincount(inverse, weights=mass, minlength=unique_flat.size)
            entity = np.bincount(inverse, weights=entity, minlength=unique_flat.size)
            flat = unique_flat
        bucket["projection_flat_indices"] = flat
        bucket["projection_mass_sum_g"] = mass
        bucket["projection_entity_sum"] = entity
        result.append(bucket)
    return result


def _geometry_project_density(
    profile,
    axes,
    edges,
    filters,
    *,
    interface_view: str = "folded",
    _area_cache: dict[tuple[float, ...], np.ndarray] | None = None,
    _volume_cache: dict[tuple[object, ...], np.ndarray] | None = None,
):
    shape = tuple(len(e) - 1 for e in edges)
    normalized_axes = tuple(str(axis).strip().lower() for axis in axes)
    normalized_edges = tuple(np.asarray(edge, dtype=float) for edge in edges)
    normalized_view = str(interface_view).strip().lower() or "folded"
    if normalized_view not in {"full", "folded", "top", "bottom", "all"}:
        raise ValueError("interface_view must be full, folded, top, bottom, or all.")
    distance_relevant = "distance" in normalized_axes or "distance" in filters
    symmetric = bool(
        profile.surface_reference is not None and profile.surface_reference.is_symmetric
    )
    area_cache = {} if _area_cache is None else _area_cache
    volume_cache = {} if _volume_cache is None else _volume_cache

    if symmetric and distance_relevant and normalized_view in {"folded", "all"}:
        top_density, top_number, top_coverage = _geometry_project_density(
            profile,
            normalized_axes,
            normalized_edges,
            filters,
            interface_view="top",
            _area_cache=area_cache,
            _volume_cache=volume_cache,
        )
        bottom_density, bottom_number, bottom_coverage = _geometry_project_density(
            profile,
            normalized_axes,
            normalized_edges,
            filters,
            interface_view="bottom",
            _area_cache=area_cache,
            _volume_cache=volume_cache,
        )
        coverage = np.minimum(top_coverage, bottom_coverage)
        valid = coverage > 0
        mass_density = np.full(shape, np.nan, dtype=float)
        number_density = np.full(shape, np.nan, dtype=float)
        mass_density[valid] = 0.5 * (top_density[valid] + bottom_density[valid])
        number_density[valid] = 0.5 * (top_number[valid] + bottom_number[valid])
        return mass_density, number_density, coverage

    if normalized_view in {"top", "bottom"}:
        missing_side_data = [
            group
            for group in profile.geometry_groups
            if group.get(f"{normalized_view}_mass_sum_g") is None
            or group.get(f"{normalized_view}_entity_sum") is None
            or group.get(f"{normalized_view}_shift") is None
        ]
        if missing_side_data:
            raise ValueError(
                "This symmetric density grid predates paired per-geometry normalization data. "
                "Recompute the density analysis before plotting folded/top/bottom distance views."
            )

    mass_density, number_density, coverage = (np.zeros(shape) for _ in range(3))
    if normalized_view == "bottom" or (
        normalized_view not in {"top", "bottom"}
        and profile.surface_reference is not None
        and profile.surface_reference.resolved_layout == "single-bottom"
    ):
        distance_side = "bottom"
    else:
        distance_side = "top"
    mass_key = (
        f"{normalized_view}_mass_sum_g" if normalized_view in {"top", "bottom"} else "mass_sum_g"
    )
    entity_key = (
        f"{normalized_view}_entity_sum" if normalized_view in {"top", "bottom"} else "entity_sum"
    )
    projection_groups = _coalesced_density_projection_groups(
        profile.geometry_groups,
        mass_key=mass_key,
        entity_key=entity_key,
        distance_relevant=distance_relevant,
        distance_side=distance_side,
    )
    prism_group_count = sum(
        1
        for group in projection_groups
        if is_axis_prismatic(
            group["cell"],
            ("x", "y", "z").index(str(profile.surface_axis).strip().lower() or "z"),
        )
    )
    LOGGER.debug(
        "Density geometry projection: view=%s, axes=%s, groups=%d->%d, "
        "exact tilted-prism=%d, exact polyhedral fallback=%d.",
        normalized_view,
        "/".join(normalized_axes),
        len(profile.geometry_groups),
        len(projection_groups),
        prism_group_count,
        len(projection_groups) - prism_group_count,
    )
    for group in projection_groups:
        source = replace(
            profile,
            flat_indices=group["projection_flat_indices"],
            mass_sum_g=group["projection_mass_sum_g"],
            entity_sum=group["projection_entity_sum"],
            n_frames=group["n_frames"],
            geometry_groups=None,
        )
        mask, indices = _density_grid_sparse_mask_and_indices(source, filters=filters)
        target_indices = tuple(
            _density_grid_target_indices(
                source,
                axis,
                indices[axis][mask],
                target_edges=edge,
                target_bin_width=float(edge[1] - edge[0]),
            )
            for axis, edge in zip(axes, edges)
        )
        flat = np.ravel_multi_index(target_indices, shape)
        masses = (
            np.bincount(
                flat, weights=np.asarray(source.mass_sum_g)[mask], minlength=int(np.prod(shape))
            )
            .astype(float, copy=False)
            .reshape(shape)
        )
        numbers = (
            np.bincount(
                flat, weights=np.asarray(source.entity_sum)[mask], minlength=int(np.prod(shape))
            )
            .astype(float, copy=False)
            .reshape(shape)
        )
        geometry_key = tuple(np.asarray(group["cell"], dtype=float).ravel()) + tuple(
            np.asarray(group["origin"], dtype=float)
        )
        if distance_relevant:
            shift_key = "bottom_shift" if distance_side == "bottom" else "top_shift"
            shift_value = group.get(shift_key, group.get("shift", 0.0))
            geometry_key += (
                distance_side,
                float(group.get("shift", 0.0) if shift_value is None else shift_value),
                group.get("top_shift"),
                group.get("bottom_shift"),
                group.get("canonical_cut"),
                group.get("canonical_lower"),
                group.get("canonical_period"),
                group.get("canonical_wrap_threshold"),
            )
        volumes = volume_cache.get(geometry_key)
        if volumes is None:
            volumes = _prismatic_geometry_volumes(
                profile,
                group,
                normalized_axes,
                normalized_edges,
                filters,
                area_cache=area_cache,
                distance_side=distance_side,
            )
            if volumes is None:
                volumes = _generic_geometry_volumes(
                    profile,
                    group,
                    normalized_axes,
                    normalized_edges,
                    filters,
                    distance_side=distance_side,
                )
            volume_cache[geometry_key] = volumes
        valid = volumes > 0.0
        np.divide(
            masses,
            volumes,
            out=masses,
            where=valid,
        )
        np.divide(
            numbers,
            volumes,
            out=numbers,
            where=valid,
        )
        mass_density[valid] += masses[valid]
        number_density[valid] += numbers[valid]
        coverage[valid] += group["n_frames"]
    divisor = np.where(coverage > 0, coverage, np.nan)
    return (
        mass_density / divisor / ANGSTROM3_TO_CM3,
        number_density / divisor / ANGSTROM3_TO_NM3,
        coverage,
    )
