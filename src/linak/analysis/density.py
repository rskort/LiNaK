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
    frame_has_usable_cell as _common_frame_has_usable_cell,
    is_molecule_species_label,
    molecule_display_label,
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
    OHTopologyCache,
    molecule_positions_with_masses as _molecule_positions_with_masses,
    water_molecule_triplets as _water_molecule_triplets,
    water_triplet_axis_values_with_masses as _water_triplet_axis_values_with_masses,
    water_axis_values_per_frame as _water_axis_values_per_frame_impl,
)

LOGGER = logging.getLogger(__name__)
H2O_VALIDATION_STRIDE = 100
H2O_OH_CUTOFF_A = 1.25
DEFAULT_MIN_MOLECULE_FRAMES = 3
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
    visible_x_min_A: float | None = None
    visible_x_max_A: float | None = None
    visible_y_min_A: float | None = None
    visible_y_max_A: float | None = None
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None


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


@dataclass(frozen=True)
class _DensityHeatmapNormalizationCache:
    """Shared 2D normalization data reused across many density heatmaps."""

    use_volumetric_density: bool
    slice_volumes: np.ndarray | None
    framewise_normalization: bool


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

    def mark_frame_active(self) -> None:
        self.active_frame_count += 1

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
    return str(species_label)


def _density_selection_summary(targets: list[_DensityTargetSpec]) -> str:
    elements = [
        target.species_label for target in targets if target.selection_kind == "element"
    ]
    molecules = [
        _density_species_display_label(target.species_label)
        for target in targets
        if target.selection_kind == "molecule"
    ]
    parts: list[str] = []
    if elements:
        parts.append("elements=" + ",".join(elements))
    if molecules:
        parts.append("molecules=" + ",".join(molecules))
    return "; ".join(parts) if parts else "none"


def _molecule_target_is_active(
    *,
    species_label: str,
    active_frame_count: int,
    apply_min_molecule_frames: bool,
    min_molecule_frames: int,
) -> bool:
    if active_frame_count <= 0:
        return False
    if not apply_min_molecule_frames:
        return True
    if species_label == "mol:H2O":
        return True
    return int(active_frame_count) >= int(min_molecule_frames)


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
    if density.ndim != 2 or x_edges.size != density.shape[0] + 1 or y_edges.size != density.shape[1] + 1:
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
    oh_cutoff_A: float | None = None
    min_molecule_frames: int | None = None
    mass_histogram_sum: np.ndarray | None = None
    entity_histogram_sum: np.ndarray | None = None
    framewise_mass_density_sum: np.ndarray | None = None
    framewise_entity_density_sum: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_bins = int(np.asarray(self.bin_edges, dtype=float).size - 1)
        if self.mass_histogram_sum is None:
            self.mass_histogram_sum = np.zeros(n_bins, dtype=float)
        if self.entity_histogram_sum is None:
            self.entity_histogram_sum = np.zeros(n_bins, dtype=float)
        if self.normalization_cache.framewise_normalization and self.framewise_mass_density_sum is None:
            self.framewise_mass_density_sum = np.zeros(n_bins, dtype=float)
        if self.normalization_cache.framewise_normalization and self.framewise_entity_density_sum is None:
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
    ) -> None:
        per_frame_mass_histogram, per_frame_entity_histogram = _bincount_1d_histograms(
            values=axis_values,
            masses=masses,
            bin_start=self.bin_start,
            bin_width=self.bin_width,
            n_bins=self.n_bins,
            bin_end=self.bin_end,
        )
        if validation_plan.strict or frame_index in validation_plan.frame_indices:
            _validate_binned_frame_conservation(
                frame_index=frame_index,
                mass_histogram=per_frame_mass_histogram,
                entity_histogram=per_frame_entity_histogram,
                masses=masses,
                axis_values=axis_values,
            )
        assert self.mass_histogram_sum is not None
        assert self.entity_histogram_sum is not None
        self.mass_histogram_sum += per_frame_mass_histogram
        self.entity_histogram_sum += per_frame_entity_histogram
        if self.normalization_cache.use_volumetric_density and self.normalization_cache.slice_volumes is not None:
            frame_volume = float(self.normalization_cache.slice_volumes[frame_index])
            frame_density = (per_frame_mass_histogram / frame_volume) / ANGSTROM3_TO_CM3
            frame_number_density = (per_frame_entity_histogram / frame_volume) / ANGSTROM3_TO_NM3
            if self.framewise_mass_density_sum is not None and self.framewise_entity_density_sum is not None:
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
        if self.normalization_cache.use_volumetric_density and self.normalization_cache.slice_volumes is not None:
            if self.framewise_mass_density_sum is None or self.framewise_entity_density_sum is None:
                reference_slice_volume = float(self.normalization_cache.slice_volumes[0])
                density = (self.mass_histogram_sum / reference_slice_volume) / n_frames
                number_density = (self.entity_histogram_sum / reference_slice_volume) / n_frames
            else:
                density = self.framewise_mass_density_sum / n_frames
                number_density = self.framewise_entity_density_sum / n_frames
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
            series_statistics={
                "density": _finalize_density_statistics_moments(self.density_moments),
                "number_density": _finalize_density_statistics_moments(self.number_density_moments),
            },
            visible_x_min_A=visible_x_min_A,
            visible_x_max_A=visible_x_max_A,
            oh_cutoff_A=self.oh_cutoff_A,
            min_molecule_frames=self.min_molecule_frames,
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

    def __post_init__(self) -> None:
        shape = (int(self.x_bin_edges.size - 1), int(self.y_bin_edges.size - 1))
        if self.mass_histogram_sum is None:
            self.mass_histogram_sum = np.zeros(shape, dtype=float)
        if self.entity_histogram_sum is None:
            self.entity_histogram_sum = np.zeros(shape, dtype=float)
        if self.normalization_cache.framewise_normalization and self.framewise_mass_density_sum is None:
            self.framewise_mass_density_sum = np.zeros(shape, dtype=float)
        if self.normalization_cache.framewise_normalization and self.framewise_entity_density_sum is None:
            self.framewise_entity_density_sum = np.zeros(shape, dtype=float)

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
        if self.normalization_cache.use_volumetric_density and self.normalization_cache.slice_volumes is not None and self.framewise_mass_density_sum is not None and self.framewise_entity_density_sum is not None:
            frame_volume = float(self.normalization_cache.slice_volumes[frame_index])
            self.framewise_mass_density_sum += per_frame_mass_histogram / frame_volume
            self.framewise_entity_density_sum += per_frame_entity_histogram / frame_volume

    def finalize(self, *, n_frames: int) -> DensityHeatmapProfile:
        assert self.mass_histogram_sum is not None
        assert self.entity_histogram_sum is not None
        if self.normalization_cache.use_volumetric_density and self.normalization_cache.slice_volumes is not None:
            if self.framewise_mass_density_sum is None or self.framewise_entity_density_sum is None:
                reference_slice_volume = float(self.normalization_cache.slice_volumes[0])
                density = (self.mass_histogram_sum / reference_slice_volume) / n_frames
                number_density = (self.entity_histogram_sum / reference_slice_volume) / n_frames
            else:
                density = self.framewise_mass_density_sum / n_frames
                number_density = self.framewise_entity_density_sum / n_frames
            density = density / ANGSTROM3_TO_CM3
            number_density = number_density / ANGSTROM3_TO_NM3
            units = "g/cm^3"
            number_density_units = _entity_density_units_for_species(self.species_label, volumetric=True)
        else:
            density = (self.mass_histogram_sum / n_frames) / (self.bin_width * self.bin_width)
            number_density = (self.entity_histogram_sum / n_frames) / (self.bin_width * self.bin_width)
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
    data = np.asarray(positions, dtype=float)
    if data.size == 0 or data.ndim != 2 or data.shape[1] != 3:
        return data
    pbc = np.asarray(frame.get_pbc(), dtype=bool)
    if not np.any(pbc) or not _common_frame_has_usable_cell(frame):
        return data
    try:
        scaled = np.asarray(frame.cell.scaled_positions(data), dtype=float)
    except Exception:
        return data
    scaled[:, pbc] = np.mod(scaled[:, pbc], 1.0)
    return np.asarray(scaled @ np.asarray(frame.cell.array, dtype=float), dtype=float)


def _select_water_axis_values_with_masses(
    frame: Atoms, axis_index: int, oh_cutoff: float = H2O_OH_CUTOFF_A
) -> tuple[np.ndarray, np.ndarray]:
    """Return COM axis positions and molecular masses for detected water molecules."""
    triplets = _water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
    if triplets.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return _water_triplet_axis_values_with_masses(frame, triplets, axis_index)


def _select_water_axis_values_per_frame(
    frames: list[Atoms], axis_index: int, oh_cutoff: float = H2O_OH_CUTOFF_A
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select water-molecule COM axis values with periodic cached-topology validation."""
    return _water_axis_values_per_frame_impl(
        frames,
        axis_index,
        oh_cutoff=oh_cutoff,
        progress_desc="Selecting H2O for density",
    )


def _select_molecule_axis_values_per_frame(
    frames: list[Atoms],
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
    frames: list[Atoms],
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


def _uniform_bin_indices(values: np.ndarray, *, bin_start: float, bin_width: float, n_bins: int) -> np.ndarray:
    """Map values onto one fixed uniform bin grid using NumPy histogram edge semantics."""

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return np.empty(0, dtype=int)
    indices = np.floor((data - float(bin_start)) / float(bin_width)).astype(int)
    np.clip(indices, 0, int(n_bins) - 1, out=indices)
    return indices


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
    frames: list[Atoms],
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
    )


def _build_heatmap_normalization_cache(
    *,
    frames: list[Atoms],
    orth_axis_index: int,
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
    )


def _resolve_density_target_specs(
    frames: list[Atoms],
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
        return [
            *element_targets,
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
        selection_kind="element",
    )
    ]


def _normalize_density_outputs(outputs: str | None) -> str:
    normalized = str(outputs or "line").strip().lower()
    if normalized not in {"line", "heatmap", "all"}:
        raise ValueError("density outputs must be one of: line, heatmap, all")
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


def _density_needs_observed_bounds_prepass(
    *,
    normalized_binning: str,
    include_line_outputs: bool,
    selected_heatmap_planes: tuple[tuple[str, str], ...],
    raw_axes: tuple[str, ...],
    cell_bounds_by_axis: Mapping[str, tuple[float, float] | None],
    distance_hist_bounds: tuple[float, float] | None,
) -> bool:
    heatmap_axes = {axis for plane in selected_heatmap_planes for axis in plane}
    if any(cell_bounds_by_axis.get(axis) is None for axis in heatmap_axes):
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
    frames: list[Atoms],
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
                label: np.asarray(symbols == label, dtype=bool)
                for label in self.element_labels
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
    if bin_edges[-1] <= data_max:
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
            if np.isfinite(bounds_min_x) and np.isfinite(bounds_max_x) and bounds_max_x > bounds_min_x:
                data_min_x = min(data_min_x, bounds_min_x)
                data_max_x = max(data_max_x, bounds_max_x)
        if y_bounds is not None:
            bounds_min_y = float(y_bounds[0])
            bounds_max_y = float(y_bounds[1])
            if np.isfinite(bounds_min_y) and np.isfinite(bounds_max_y) and bounds_max_y > bounds_min_y:
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
    if x_bin_edges[-1] <= data_max_x:
        x_bin_edges = np.append(x_bin_edges, x_bin_edges[-1] + bin_width)
    if y_bin_edges[-1] <= data_max_y:
        y_bin_edges = np.append(y_bin_edges, y_bin_edges[-1] + bin_width)
    return np.asarray(x_bin_edges, dtype=float), np.asarray(y_bin_edges, dtype=float)


def _bincount_1d_histograms(
    *,
    values: np.ndarray,
    masses: np.ndarray,
    bin_start: float,
    bin_width: float,
    n_bins: int,
    bin_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mass and entity histograms on one fixed uniform grid via ``np.bincount``."""

    axis_values = np.asarray(values, dtype=float)
    mass_values = np.asarray(masses, dtype=float)
    valid = np.isfinite(axis_values) & np.isfinite(mass_values) & (axis_values >= bin_start) & (axis_values <= bin_end)
    if not np.any(valid):
        return np.zeros(int(n_bins), dtype=float), np.zeros(int(n_bins), dtype=float)
    indices = _uniform_bin_indices(
        axis_values[valid],
        bin_start=bin_start,
        bin_width=bin_width,
        n_bins=n_bins,
    )
    return (
        np.bincount(indices, weights=mass_values[valid], minlength=int(n_bins)).astype(float, copy=False),
        np.bincount(indices, minlength=int(n_bins)).astype(float, copy=False),
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
    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
        & np.isfinite(mass_values)
        & (x_array >= x_bin_start)
        & (x_array <= x_bin_end)
        & (y_array >= y_bin_start)
        & (y_array <= y_bin_end)
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
    mass_histogram = np.bincount(
        flat_indices,
        weights=mass_values[valid],
        minlength=total_bins,
    ).astype(float, copy=False).reshape((int(x_bin_count), int(y_bin_count)))
    entity_histogram = np.bincount(
        flat_indices,
        minlength=total_bins,
    ).astype(float, copy=False).reshape((int(x_bin_count), int(y_bin_count)))
    return mass_histogram, entity_histogram


def _compute_density_heatmap_profile_from_selected(
    *,
    frames: list[Atoms],
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
        raise ValueError("Selected planar coordinate and mass arrays must have matching frame counts.")
    n_selected_total = sum(np.asarray(values, dtype=float).shape[0] for values in selected_xy_per_frame)
    if n_selected_total == 0:
        raise ValueError(f"No entities found for selection '{species_label}' in trajectory.")

    non_empty = [np.asarray(values, dtype=float) for values in selected_xy_per_frame if np.asarray(values).size > 0]
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
            if np.isfinite(bounds_min_x) and np.isfinite(bounds_max_x) and bounds_max_x > bounds_min_x:
                data_min_x = min(data_min_x, bounds_min_x)
                data_max_x = max(data_max_x, bounds_max_x)
        if y_bounds is not None:
            bounds_min_y = float(y_bounds[0])
            bounds_max_y = float(y_bounds[1])
            if np.isfinite(bounds_min_y) and np.isfinite(bounds_max_y) and bounds_max_y > bounds_min_y:
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
    if x_bin_edges[-1] <= data_max_x:
        x_bin_edges = np.append(x_bin_edges, x_bin_edges[-1] + bin_width)
    if y_bin_edges[-1] <= data_max_y:
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
    framewise_normalization = False
    framewise_mass_density_sum: np.ndarray | None = None
    framewise_entity_density_sum: np.ndarray | None = None
    if use_volumetric_density:
        slice_volumes = np.empty(n_frames, dtype=float)
        for index, frame in enumerate(frames):
            cell = np.asarray(frame.cell.array, dtype=float)
            orth_length = np.linalg.norm(cell[orth_axis_index])
            volume = abs(float(np.linalg.det(cell)))
            slice_volume = (volume / orth_length) * (bin_width * bin_width)
            slice_volumes[index] = slice_volume
        variable_slice_volume = not np.allclose(slice_volumes, slice_volumes[0], rtol=0.0, atol=1e-12)
        framewise_normalization = bool(variable_slice_volume)
        if framewise_normalization:
            framewise_mass_density_sum = np.zeros_like(mass_histogram_sum)
            framewise_entity_density_sum = np.zeros_like(entity_histogram_sum)

    for frame_index, (xy_values, masses) in enumerate(zip(selected_xy_per_frame, selected_masses_per_frame)):
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
        if use_volumetric_density and slice_volumes is not None and framewise_mass_density_sum is not None and framewise_entity_density_sum is not None:
            frame_volume = float(slice_volumes[frame_index])
            framewise_mass_density_sum += per_frame_mass_histogram / frame_volume
            framewise_entity_density_sum += per_frame_entity_histogram / frame_volume

    if use_volumetric_density and slice_volumes is not None:
        if framewise_mass_density_sum is None or framewise_entity_density_sum is None:
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
    block_index_by_frame: np.ndarray | None = None,
    oh_cutoff_A: float | None = None,
    min_molecule_frames: int | None = None,
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
            per_frame_mass_histogram, per_frame_entity_histogram = _bincount_1d_histograms(
                values=np.asarray(axis_values, dtype=float),
                masses=np.asarray(masses, dtype=float),
                bin_start=bin_start,
                bin_width=bin_width,
                n_bins=n_bins,
                bin_end=bin_end,
            )
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
    LOGGER.info(
        "Density selections: %s.",
        _density_selection_summary(
            [
                _DensityTargetSpec(
                    species_label=species_label,
                    count_label=count_label,
                    selection_kind="molecule" if selection_mode == "molecule" else "element",
                )
            ]
        ),
    )
    if selection_mode == "molecule":
        LOGGER.info("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
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
        oh_cutoff_A=oh_cutoff if selection_mode == "molecule" else None,
        min_molecule_frames=min_molecule_frames if selection_mode == "molecule" else None,
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
    if selection_mode not in {"all", "elements", "molecules"}:
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
                oh_cutoff=oh_cutoff,
                min_molecule_frames=min_molecule_frames,
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
        if _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames))
        else None
    )
    per_frame_surface = (
        None if trusted_surface_estimate is None else trusted_surface_estimate.per_frame
    )
    coordinate_mode_global = "distance" if trusted_surface_estimate is not None else "axis"
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
            if _molecule_target_is_active(
                species_label=molecule_label,
                active_frame_count=active_frame_count,
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
    LOGGER.info("Density selections: %s.", _density_selection_summary(active_targets_for_log))
    if molecule_selections:
        LOGGER.info("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
        non_water_molecules = [
            _density_species_display_label(label)
            for label in molecule_selections
            if label != "mol:H2O"
        ]
        if non_water_molecules:
            LOGGER.info(
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
                    oh_cutoff_A=None,
                    min_molecule_frames=None,
                )
            )

        for molecule_label, (
            molecule_selected_per_frame,
            molecule_masses_per_frame,
        ) in molecule_selections.items():
            selected_for_binning, coordinate_mode = _shift_axis_values_by_surface_per_frame(
                molecule_selected_per_frame,
                per_frame_surface,
            )
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
    frames: list[Atoms],
    species: str | None,
    surface_axis: str,
    bin_width: float,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    binning: str,
    surface_options: SurfaceEstimatorOptions | None,
    precomputed_surface_estimate: SurfaceEstimate | None,
    outputs: str | None,
    heatmap_planes: list[str] | tuple[str, ...] | None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
) -> list[DensityProfile | DensityHeatmapProfile]:
    """Shared streaming density engine for raw-axis, distance, and heatmap outputs."""

    ensure_positive("bin_width", bin_width)
    ensure_positive("oh_cutoff", oh_cutoff)
    if int(min_molecule_frames) < 1:
        raise ValueError("min_molecule_frames must be >= 1.")
    normalized_binning = binning.strip().lower()
    if normalized_binning not in {"observed", "cell"}:
        raise ValueError("binning must be one of: observed, cell")
    normalized_outputs = _normalize_density_outputs(outputs)
    include_line_outputs = normalized_outputs in {"line", "all"}
    include_heatmap_outputs = normalized_outputs in {"heatmap", "all"}
    selected_heatmap_planes = (
        _normalize_density_heatmap_planes(heatmap_planes)
        if include_heatmap_outputs
        else tuple()
    )
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    raw_axes = ("x", "y", "z")
    axis_indices = {ax: axis_to_index(ax) for ax in raw_axes}
    surface_axis_index = axis_to_index(surface_axis)

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
    distance_coordinate_mode = "distance" if per_frame_surface is not None else "axis"

    target_specs = _resolve_density_target_specs(frames, species=species)

    line_jobs_per_species = 4 if include_line_outputs else 0
    heatmap_jobs_per_species = len(selected_heatmap_planes)

    block_index_by_frame = _resolve_frame_block_index_by_frame(len(frames))
    validation_plan = _build_density_validation_plan(len(frames))
    n_blocks = None if block_index_by_frame is None else int(np.max(block_index_by_frame)) + 1

    cell_bounds_by_axis: dict[str, tuple[float, float] | None] = {}
    if normalized_binning == "cell" and (include_line_outputs or include_heatmap_outputs):
        for ax in raw_axes:
            cell_bounds_by_axis[ax] = _cell_histogram_bounds(
                frames=frames,
                axis_index=axis_indices[ax],
                coordinate_mode="axis",
                surface_per_frame=None,
            )
    distance_hist_bounds = (
        None
        if not include_line_outputs or normalized_binning != "cell"
        else _cell_histogram_bounds(
            frames=frames,
            axis_index=surface_axis_index,
            coordinate_mode=distance_coordinate_mode,
            surface_per_frame=per_frame_surface if distance_coordinate_mode == "distance" else None,
        )
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
        LOGGER.info(
            "Density bin preparation uses cell bounds; skipped observed coordinate scan."
        )
    else:
        observed_bounds_by_target = {
            target.species_label: _DensityObservedBounds(axis_lower={}, axis_upper={})
            for target in target_specs
        }
        run_topology_cache = OHTopologyCache(
            oh_cutoff=oh_cutoff,
            context="density O/H molecule topology",
        )
        element_selector = _DensityElementSelector(
            [
                target.species_label
                for target in target_specs
                if target.selection_kind == "element"
            ]
        )
        with ProgressBar(desc="Preparing density bins", total=len(frames), unit="frame") as progress:
            for frame_index, frame in enumerate(frames):
                frame_selections = _select_density_targets_in_frame(
                    frame,
                    frame_index=frame_index,
                    target_specs=target_specs,
                    topology_cache=run_topology_cache,
                    element_selector=element_selector,
                    wrap_positions_to_cell=normalized_binning == "cell",
                )
                for target in target_specs:
                    selection = frame_selections.get(target.species_label)
                    if selection is None:
                        continue
                    positions, _masses = selection
                    bounds = observed_bounds_by_target[target.species_label]
                    bounds.mark_frame_active()
                    for ax in raw_axes:
                        bounds.update_axis(ax, positions[:, axis_indices[ax]])
                    distance_values = (
                        np.asarray(positions[:, surface_axis_index], dtype=float)
                        if per_frame_surface is None
                        else np.asarray(positions[:, surface_axis_index], dtype=float)
                        - float(per_frame_surface[frame_index])
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
                apply_min_molecule_frames=target.apply_min_molecule_frames,
                min_molecule_frames=min_molecule_frames,
            ):
                continue
            active_targets.append(target)
    if "run_topology_cache" not in locals():
        run_topology_cache = OHTopologyCache(
            oh_cutoff=oh_cutoff,
            context="density O/H molecule topology",
        )
    if not active_targets:
        raise ValueError("No entities found for the requested density selection.")

    active_element_count = sum(1 for target in active_targets if target.selection_kind == "element")
    molecule_active_count = sum(1 for target in active_targets if target.selection_kind == "molecule")
    _density_mode, run_summary = _summarize_density_run(
        frames=frames,
        axis_index=surface_axis_index,
        coordinate_mode=distance_coordinate_mode,
        labels=[target.species_label for target in active_targets],
    )
    LOGGER.info("Density compute summary: %s.", run_summary)
    LOGGER.info("Density selections: %s.", _density_selection_summary(active_targets))
    if molecule_active_count:
        LOGGER.info("O/H molecule cutoff: %.6g A.", float(oh_cutoff))
        non_water_molecules = [
            _density_species_display_label(target.species_label)
            for target in active_targets
            if target.selection_kind == "molecule" and target.species_label != "mol:H2O"
        ]
        if non_water_molecules:
            LOGGER.info(
                "Detected O/H molecule species: %s.",
                ",".join(non_water_molecules),
            )
    LOGGER.info(
        "Single-pass selection complete: %d element species + %d O/H molecule selections, %d frames.",
        active_element_count,
        molecule_active_count,
        len(frames),
    )
    LOGGER.info(
        "Density selection prepass complete: %d active species selection(s), %d frames.",
        len(active_targets),
        len(frames),
    )
    total_binning_jobs = len(active_targets) * (line_jobs_per_species + heatmap_jobs_per_species)
    LOGGER.info(
        "Binning %d density outputs (%d line profiles + %d heatmaps per species selection).",
        total_binning_jobs,
        line_jobs_per_species,
        heatmap_jobs_per_species,
    )

    line_accumulators_by_target: dict[str, list[tuple[str, _DensityLineAccumulator]]] = {}
    heatmap_accumulators_by_target: dict[str, list[tuple[tuple[str, str], _DensityHeatmapAccumulator]]] = {}
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
                                min_molecule_frames
                                if target.selection_kind == "molecule"
                                else None
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
            target_line_accumulators.append(
                (
                    "distance",
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
                        surface_estimate=surface_estimate,
                        oh_cutoff_A=oh_cutoff if target.selection_kind == "molecule" else None,
                        min_molecule_frames=(
                            min_molecule_frames
                            if target.selection_kind == "molecule"
                            else None
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
                            min_molecule_frames
                            if target.selection_kind == "molecule"
                            else None
                        ),
                    ),
                )
            )
        heatmap_accumulators_by_target[species_lbl] = target_heatmap_accumulators

    profiles: list[DensityProfile | DensityHeatmapProfile] = []
    accumulation_element_selector = _DensityElementSelector(
        [
            target.species_label
            for target in active_targets
            if target.selection_kind == "element"
        ]
    )
    with ProgressBar(desc="Binning density frames", total=len(frames), unit="frame") as progress:
        for frame_index, frame in enumerate(frames):
            frame_selections = _select_density_targets_in_frame(
                frame,
                frame_index=frame_index,
                target_specs=active_targets,
                topology_cache=run_topology_cache,
                element_selector=accumulation_element_selector,
                wrap_positions_to_cell=normalized_binning == "cell",
            )
            for target in active_targets:
                selection = frame_selections.get(target.species_label)
                if selection is None:
                    continue
                positions, masses = selection
                mass_array = np.asarray(masses, dtype=float)
                for axis_key, accumulator in line_accumulators_by_target[target.species_label]:
                    if axis_key == "distance":
                        axis_values = (
                            np.asarray(positions[:, surface_axis_index], dtype=float)
                            if per_frame_surface is None
                            else np.asarray(positions[:, surface_axis_index], dtype=float)
                            - float(per_frame_surface[frame_index])
                        )
                    else:
                        axis_values = np.asarray(positions[:, axis_indices[axis_key]], dtype=float)
                    accumulator.update(
                        axis_values=axis_values,
                        masses=mass_array,
                        frame_index=frame_index,
                        validation_plan=validation_plan,
                        block_index_by_frame=block_index_by_frame,
                    )
                for plane_axes, accumulator in heatmap_accumulators_by_target[target.species_label]:
                    accumulator.update(
                        x_values=np.asarray(positions[:, axis_indices[plane_axes[0]]], dtype=float),
                        y_values=np.asarray(positions[:, axis_indices[plane_axes[1]]], dtype=float),
                        masses=mass_array,
                        frame_index=frame_index,
                        validation_plan=validation_plan,
                    )
            progress.update()

        for target in active_targets:
            target_profiles: list[DensityProfile | DensityHeatmapProfile] = [
                *(accumulator.finalize(n_frames=len(frames)) for _, accumulator in line_accumulators_by_target[target.species_label]),
                *(accumulator.finalize(n_frames=len(frames)) for _, accumulator in heatmap_accumulators_by_target[target.species_label]),
            ]
            profiles.extend(target_profiles)

    LOGGER.info(
        "Density binning summary: outputs=%d, frames=%d, saved_statistics=sample%s.",
        len(profiles),
        len(frames),
        "+block" if block_index_by_frame is not None else "",
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
    outputs: str | None = "line",
    heatmap_planes: list[str] | tuple[str, ...] | None = None,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    min_molecule_frames: int = DEFAULT_MIN_MOLECULE_FRAMES,
) -> list[DensityProfile | DensityHeatmapProfile]:
    return _compute_all_density_profiles_streaming(
        frames=frames,
        species=species,
        surface_axis=surface_axis,
        bin_width=bin_width,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        binning=binning,
        surface_options=surface_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
        outputs=outputs,
        heatmap_planes=heatmap_planes,
        oh_cutoff=oh_cutoff,
        min_molecule_frames=min_molecule_frames,
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
        "selection_kind": "molecule" if is_molecule_species_label(profile_species_label) else "element",
        "entity_kind": "molecule" if is_molecule_species_label(profile_species_label) else "atom",
        "units": canonical_density_units,
        "n_frames": profile.n_frames,
        "number_density_units": canonical_number_units,
        "coordinate_mode": profile.coordinate_mode,
        "bin_width_A": bin_width,
        "counts_per_frame_available": False,
        "visible_x_min_A": profile.visible_x_min_A,
        "visible_x_max_A": profile.visible_x_max_A,
        "oh_cutoff_A": profile.oh_cutoff_A,
        "min_molecule_frames": profile.min_molecule_frames,
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
            None if profile.number_density is None else np.asarray(profile.number_density, dtype=float),
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
            "selection_kind": "molecule" if is_molecule_species_label(profile_species_label) else "element",
            "entity_kind": "molecule" if is_molecule_species_label(profile_species_label) else "atom",
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
    profiles: list[DensityProfile | DensityHeatmapProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
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
            )
        payload = _density_heatmap_profile_hdf5_payload(single_profile)
        metadata = dict(payload["metadata"])
        if additional_metadata:
            metadata.update(dict(additional_metadata))
        output_path = write_linak_hdf5(
            output,
            analysis="density",
            datasets=payload["datasets"],
            metadata=metadata,
        )
        LOGGER.info("Saved density heatmap data to '%s'.", output_path)
        return output_path

    output_path = write_profile_collection(
        output,
        analysis="density",
        profiles=[
            _density_profile_hdf5_payload(profile)
            if isinstance(profile, DensityProfile)
            else _density_heatmap_profile_hdf5_payload(profile)
            for profile in profiles
        ],
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
    profile_kind: str | None = "line_1d",
    plane: str | None = None,
) -> bool:
    requested_profile_kind = None if profile_kind is None else str(profile_kind).strip().lower()
    metadata_profile_kind = str(metadata.get("profile_kind", "line_1d")).strip().lower() or "line_1d"
    if requested_profile_kind is not None and metadata_profile_kind != requested_profile_kind:
        return False

    requested_axis = None if axis is None or not str(axis).strip() else str(axis).strip().lower()
    if requested_axis is not None:
        metadata_axis = str(metadata.get("axis", "z")).strip().lower() or "z"
        if metadata_axis != requested_axis:
            return False

    requested_plane = None if plane is None or not str(plane).strip() else str(plane).strip().lower()
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
            density_values, units, number_density_values, number_density_units = canonicalize_density_units(
                density=density_values,
                density_units=units,
                number_density=number_density_values,
                number_density_units=number_density_units,
            )
        x_bin_centers = np.asarray(datasets["x_bin_centers_A"], dtype=float)
        y_bin_centers = np.asarray(datasets["y_bin_centers_A"], dtype=float)
        x_bin_width = float(metadata.get("x_bin_width_A", 0.0))
        y_bin_width = float(metadata.get("y_bin_width_A", 0.0))
        if x_bin_width <= 0.0 or y_bin_width <= 0.0:
            raise ValueError(f"Density HDF5 '{source_path}' has incompatible heatmap bin geometry.")
        x_bin_edges = reconstruct_uniform_bin_edges_from_centers(x_bin_centers, bin_width=x_bin_width)
        y_bin_edges = reconstruct_uniform_bin_edges_from_centers(y_bin_centers, bin_width=y_bin_width)
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
            _mass, _mass_units, canonical_number, canonical_number_units = canonicalize_density_units(
                density=profile.density,
                density_units=profile.units,
                number_density=profile.number_density,
                number_density_units=profile.number_density_units,
            )
            assert canonical_number is not None
            assert canonical_number_units is not None
            return canonical_number, canonical_number_units, "Entity density"
        return np.asarray(profile.number_density, dtype=float), str(profile.number_density_units), "Entity density"

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


def _density_profile_visible_x_limits(
    profile: DensityProfile,
    *,
    x_mode: str,
) -> list[float] | None:
    lower = profile.visible_x_min_A
    upper = profile.visible_x_max_A
    if lower is None or upper is None:
        return None
    if not (np.isfinite(float(lower)) and np.isfinite(float(upper))):
        return None
    normalized_x_mode = str(x_mode).strip().lower() or "distance"
    lower_value = float(lower)
    upper_value = float(upper)
    if normalized_x_mode in {"axis", "x", "y", "z"}:
        if (
            profile.coordinate_mode == "distance"
            and profile.surface_position is not None
            and np.isfinite(float(profile.surface_position))
        ):
            lower_value += float(profile.surface_position)
            upper_value += float(profile.surface_position)
    elif normalized_x_mode == "distance":
        if (
            profile.coordinate_mode != "distance"
            and profile.surface_position is not None
            and np.isfinite(float(profile.surface_position))
        ):
            lower_value -= float(profile.surface_position)
            upper_value -= float(profile.surface_position)
    else:
        return None
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


def plot_density_profile(
    profile: DensityProfile | DensityHeatmapProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
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
    resolved_mapping = resolve_density_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
        view_type=str(getattr(view_mapping, "view_type_id", "") or "line_1d"),
        x_mode=x_mode,
        quantity=quantity,
    )
    runtime_x_mode = resolved_mapping.x_mode
    runtime_quantity = resolved_mapping.quantity
    if resolved_mapping.is_heatmap:
        if not isinstance(profile, DensityHeatmapProfile):
            raise ValueError("Density heatmap rendering requires a 2D density heatmap profile.")
        z_values, units, z_label_prefix = _density_heatmap_z_data(profile, quantity=runtime_quantity)
        units = _format_plot_density_units(units)
        display_species = _density_species_display_label(profile.species)
        default_title = f"{display_species} density heatmap ({profile.plane.upper()})"
        default_x_label = f"{profile.plane_axes[0].upper()} (A)"
        default_y_label = f"{profile.plane_axes[1].upper()} (A)"
        resolved_heatmap_x_lim = x_lim
        resolved_heatmap_y_lim = y_lim
        if x_lim is None and profile.visible_x_min_A is not None and profile.visible_x_max_A is not None:
            resolved_heatmap_x_lim = _expand_linear_limits(
                float(profile.visible_x_min_A),
                float(profile.visible_x_max_A),
            )
        if y_lim is None and profile.visible_y_min_A is not None and profile.visible_y_max_A is not None:
            resolved_heatmap_y_lim = _expand_linear_limits(
                float(profile.visible_y_min_A),
                float(profile.visible_y_max_A),
            )
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
            preferred_backend=preferred_backend,
            style=style,
            x_lim=resolved_heatmap_x_lim,
            y_lim=resolved_heatmap_y_lim,
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
    profiles: list[DensityProfile | DensityHeatmapProfile] | DensityProfile | DensityHeatmapProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
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
    if not profiles:
        raise ValueError("At least one density profile is required.")
    first_profile = profiles[0]
    resolved_mapping = resolve_density_plot_mapping(
        contract=data_contract,
        profile=first_profile,
        mapping=view_mapping,
        view_type=str(getattr(view_mapping, "view_type_id", "") or "line_1d"),
        x_mode=x_mode,
        quantity=quantity,
    )
    runtime_x_mode = resolved_mapping.x_mode
    runtime_quantity = resolved_mapping.quantity
    if resolved_mapping.is_heatmap:
        if len(profiles) != 1:
            raise ValueError("Density heatmap rendering currently supports one selected heatmap field at a time.")
        return plot_density_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
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
    default_labels = [_density_species_display_label(profile.species) for profile in profiles]
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
            i for i, p in enumerate(profiles)
            if _density_profile_visible_for_x_mode(p, x_mode=runtime_x_mode)
        ]
        if not compat:
            raise ValueError(
                f"No density profiles match the selected axis '{runtime_x_mode.upper()}'."
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
    if runtime_x_mode == "distance" and any(
        not _profile_has_surface_reference(profile) for profile in profiles
    ):
        LOGGER.warning(
            "At least one profile has no surface reference; combined plot falls back to axis coordinates."
        )
        runtime_x_mode = "axis"

    y_resolved = [_density_y_data(profile, quantity=runtime_quantity) for profile in profiles]
    y_units = y_resolved[0][1]
    y_label_prefix = y_resolved[0][2]
    if any(units != y_units for _, units, _ in y_resolved[1:]):
        unified = _unify_number_density_units([units for _, units, _ in y_resolved])
        if unified is not None:
            y_units = unified
        else:
            raise ValueError("All density profiles must use the same units for combined plotting.")
    y_series = [values for values, _, _ in y_resolved]
    x_series = [_density_x_data(profile, x_mode=runtime_x_mode)[0] for profile in profiles]
    default_x_label = _density_x_data(first, x_mode=runtime_x_mode)[1]
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
        if x_lim is None:
            stored_x_lim = _density_profiles_visible_x_limits(
                [profile for profile in profiles if isinstance(profile, DensityProfile)],
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
