"""Water orientation analysis.

Computes water-orientation observables as a function of either distance to a
trusted frame-wise surface reference or, when no trusted surface reference is
available, along a raw Cartesian axis.

The primary observable is ``cos_polar = b . e_ref``, where ``b`` is the
normalized water-bisector vector and ``e_ref`` is the chosen Cartesian
reference axis. Positive and negative values therefore track the positive and
negative reference-axis directions directly; the sign is not auto-flipped by
surface-side inference.

LiNaK also stores an azimuthal descriptor derived from the water-plane normal.
That quantity is mathematically well defined for a chosen Cartesian in-plane
reference axis, but it is not a unique laboratory-frame observable unless the
system itself defines a preferred in-plane direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.orientation_mapping import resolve_orientation_plot_mapping
from ..storage.hdf5_utils import write_linak_hdf5
from .common import (
    frame_has_usable_cell as _frame_has_usable_cell,
    read_profile_payloads,
    read_profile_payloads_by_index,
)
from .schema import build_profile_metadata
from .statistics import (
    SeriesStatistics,
    block_mean_matrix,
    build_series_statistics,
    build_statistics_metadata,
    statistics_payload_from_series_map,
    statistics_series_map_from_datasets,
    resolve_block_slices,
)
from .surface import (
    _cell_histogram_bounds,
    _surface_estimate_datasets,
    _surface_estimate_from_payload,
    _surface_estimate_supports_distance_mode,
    _surface_metadata_payload,
    _surface_metadata_view,
    SurfaceEstimate,
    SurfaceEstimatorOptions,
)
from .water import (
    H2O_OH_CUTOFF_A,
    OHTopologyCache,
    WaterGeometry,
    water_triplet_geometry,
)
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_heatmap_series,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
)
from ..progress import ProgressBar
from ..utils import axis_to_index

LOGGER = logging.getLogger(__name__)

# Default number of equally-spaced bins over the cos(angle) range [-1, +1].
_DEFAULT_ANGLE_BIN_COUNT: int = 50
_OH_NORM_EPSILON: float = 1.0e-12
_BISECTOR_NORM_EPSILON: float = 1.0e-12
_PLANE_NORMAL_NORM_EPSILON: float = 1.0e-12
_PROJECTED_NORMAL_NORM_EPSILON: float = 1.0e-12
_OH_LENGTH_SANITY_TOLERANCE: float = 1.0e-8
_HEATMAP_BULK_DENSITY_FRACTION: float = 0.8


# Data containers


@dataclass(frozen=True)
class OrientationSparseGrid:
    """Sparse spatial orientation grid used for filtered 1D/2D derivations."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    distance_edges: np.ndarray
    shape: tuple[int, int, int, int]
    flat_indices: np.ndarray
    entity_sum: np.ndarray
    cos_polar_sum: np.ndarray
    count_polar_valid: np.ndarray
    cos_azimuthal_sum: np.ndarray
    count_azimuthal_valid: np.ndarray


@dataclass(frozen=True)
class OrientationProfile:
    """Container for a water-orientation analysis result."""

    axis: str
    """Spatial axis used for distance binning (``"x"``, ``"y"``, ``"z"``)."""

    reference_axis: str
    """Axis treated as the surface normal for angle computation."""

    n_frames: int
    n_molecules_per_frame: int

    # 1-D distance bins.
    bin_edges: np.ndarray
    bin_centers: np.ndarray

    # Mean cos(angle) per distance bin.
    cos_polar_mean: np.ndarray
    cos_azimuthal_mean: np.ndarray
    count_total: np.ndarray
    count_polar_valid: np.ndarray
    count_azimuthal_valid: np.ndarray

    # Density-weighted cos(angle) per distance bin.
    cos_polar_density: np.ndarray
    cos_azimuthal_density: np.ndarray

    # H2O number-density per distance bin.
    density: np.ndarray

    # 2-D heatmaps.
    heatmap_polar: np.ndarray
    heatmap_azimuthal: np.ndarray
    heatmap_angle_bin_edges: np.ndarray
    heatmap_angle_bin_centers: np.ndarray

    # Surface and coordinate metadata.
    coordinate_mode: str  # "distance" or "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_estimate: SurfaceEstimate | None = None
    cell_lengths_angstrom: tuple[float, float, float] | None = None
    series_statistics: dict[str, SeriesStatistics] | None = None
    sparse_grid: OrientationSparseGrid | None = None


@dataclass(frozen=True)
class _OrientationFrameData:
    positions: np.ndarray
    distances: np.ndarray
    cos_polar: np.ndarray
    polar_valid: np.ndarray
    cos_azimuthal: np.ndarray
    azimuthal_valid: np.ndarray


# HDF5 save/load


def _in_plane_reference_vector(ref_index: int) -> np.ndarray:
    vector = np.zeros(3, dtype=float)
    for axis_index in range(3):
        if axis_index == ref_index:
            continue
        vector[axis_index] = 1.0
        return vector
    raise ValueError("reference axis must leave at least one in-plane Cartesian axis")


def _build_orientation_frame_data(
    geom: WaterGeometry,
    *,
    axis_index: int,
    surface_position: float | None,
    ref_vec: np.ndarray,
    in_plane_ref_vec: np.ndarray,
    oh_cutoff: float,
) -> _OrientationFrameData:
    distances = np.asarray(geom.com_positions[:, axis_index], dtype=float)
    if surface_position is not None and np.isfinite(surface_position):
        distances = distances - float(surface_position)

    count = geom.com_positions.shape[0]
    if count == 0:
        empty_float = np.array([], dtype=float)
        empty_bool = np.array([], dtype=bool)
        return _OrientationFrameData(
            positions=np.empty((0, 3), dtype=float),
            distances=empty_float,
            cos_polar=empty_float,
            polar_valid=empty_bool,
            cos_azimuthal=empty_float,
            azimuthal_valid=empty_bool,
        )

    oh1 = np.asarray(geom.hydrogen1_positions - geom.oxygen_positions, dtype=float)
    oh2 = np.asarray(geom.hydrogen2_positions - geom.oxygen_positions, dtype=float)
    oh1_norm = np.linalg.norm(oh1, axis=1)
    oh2_norm = np.linalg.norm(oh2, axis=1)
    bond_valid = (
        np.isfinite(oh1_norm)
        & np.isfinite(oh2_norm)
        & (oh1_norm > _OH_NORM_EPSILON)
        & (oh2_norm > _OH_NORM_EPSILON)
        & (oh1_norm <= float(oh_cutoff) + _OH_LENGTH_SANITY_TOLERANCE)
        & (oh2_norm <= float(oh_cutoff) + _OH_LENGTH_SANITY_TOLERANCE)
    )

    oh1_unit = np.zeros_like(oh1)
    oh2_unit = np.zeros_like(oh2)
    if np.any(bond_valid):
        oh1_unit[bond_valid] = oh1[bond_valid] / oh1_norm[bond_valid, np.newaxis]
        oh2_unit[bond_valid] = oh2[bond_valid] / oh2_norm[bond_valid, np.newaxis]

    bisector_raw = oh1_unit + oh2_unit
    bisector_norm = np.linalg.norm(bisector_raw, axis=1)
    polar_valid = bond_valid & np.isfinite(bisector_norm) & (bisector_norm > _BISECTOR_NORM_EPSILON)
    cos_polar = np.full(count, np.nan, dtype=float)
    if np.any(polar_valid):
        bisector_unit = np.zeros_like(bisector_raw)
        bisector_unit[polar_valid] = (
            bisector_raw[polar_valid] / bisector_norm[polar_valid, np.newaxis]
        )
        cos_polar[polar_valid] = bisector_unit[polar_valid] @ ref_vec

    plane_normal = np.cross(oh1, oh2)
    plane_normal_norm = np.linalg.norm(plane_normal, axis=1)
    plane_normal_valid = (
        bond_valid
        & np.isfinite(plane_normal_norm)
        & (plane_normal_norm > _PLANE_NORMAL_NORM_EPSILON)
    )
    plane_normal_unit = np.zeros_like(plane_normal)
    if np.any(plane_normal_valid):
        plane_normal_unit[plane_normal_valid] = (
            plane_normal[plane_normal_valid] / plane_normal_norm[plane_normal_valid, np.newaxis]
        )

    projection_along_ref = np.sum(plane_normal_unit * ref_vec[np.newaxis, :], axis=1)
    projected_normal = (
        plane_normal_unit - projection_along_ref[:, np.newaxis] * ref_vec[np.newaxis, :]
    )
    projected_norm = np.linalg.norm(projected_normal, axis=1)
    azimuthal_valid = (
        plane_normal_valid
        & np.isfinite(projected_norm)
        & (projected_norm > _PROJECTED_NORMAL_NORM_EPSILON)
    )
    cos_azimuthal = np.full(count, np.nan, dtype=float)
    if np.any(azimuthal_valid):
        projected_unit = np.zeros_like(projected_normal)
        projected_unit[azimuthal_valid] = (
            projected_normal[azimuthal_valid] / projected_norm[azimuthal_valid, np.newaxis]
        )
        cos_azimuthal[azimuthal_valid] = projected_unit[azimuthal_valid] @ in_plane_ref_vec

    return _OrientationFrameData(
        positions=np.asarray(geom.com_positions, dtype=float),
        distances=distances,
        cos_polar=cos_polar,
        polar_valid=np.asarray(polar_valid, dtype=bool),
        cos_azimuthal=cos_azimuthal,
        azimuthal_valid=np.asarray(azimuthal_valid, dtype=bool),
    )


def _build_uniform_bin_edges(lower: float, upper: float, *, bin_width: float) -> np.ndarray:
    start = float(lower)
    stop = float(upper)
    if np.isclose(start, stop):
        stop = start + float(bin_width)
    n_bins = max(1, int(np.ceil((stop - start) / float(bin_width))))
    edges = start + np.arange(n_bins + 1, dtype=float) * float(bin_width)
    if edges[-1] <= stop:
        edges = np.append(edges, edges[-1] + float(bin_width))
    return edges


def _representative_cell_lengths(frames: list[Atoms]) -> tuple[float, float, float] | None:
    if not frames:
        return None
    lengths: list[tuple[float, float, float]] = []
    for frame in frames:
        pbc = np.asarray(frame.get_pbc(), dtype=bool)
        if pbc.size != 3 or not bool(np.all(pbc)):
            return None
        cell = np.asarray(frame.cell.array, dtype=float)
        frame_lengths: tuple[float, float, float] = (
            float(np.linalg.norm(cell[0])),
            float(np.linalg.norm(cell[1])),
            float(np.linalg.norm(cell[2])),
        )
        if any(length <= 0.0 or not np.isfinite(length) for length in frame_lengths):
            return None
        lengths.append(frame_lengths)
    first = lengths[0]
    if all(
        np.allclose(frame_lengths, first, rtol=0.0, atol=1.0e-12) for frame_lengths in lengths[1:]
    ):
        return first
    return None


def _per_frame_slab_volumes(
    frames: list[Atoms],
    *,
    axis_index: int,
    bin_width: float,
) -> np.ndarray | None:
    if not frames:
        return None
    if not all(_frame_has_usable_cell(frame, axis_index=axis_index) for frame in frames):
        return None
    slab_volumes = np.empty(len(frames), dtype=float)
    for frame_index, frame in enumerate(frames):
        cell = np.asarray(frame.cell.array, dtype=float)
        axis_length = float(np.linalg.norm(cell[axis_index]))
        volume = abs(float(np.linalg.det(cell)))
        cross_section = volume / axis_length
        slab_volumes[frame_index] = cross_section * float(bin_width)
    return slab_volumes


def _distance_bin_membership_mask(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(bin_edges, values, side="right") - 1
    return (indices >= 0) & (indices < (bin_edges.size - 1)) & np.isfinite(values)


def _distance_bin_indices(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(bin_edges, values, side="right") - 1


def _angle_bin_indices(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(bin_edges, values, side="right") - 1
    return np.clip(indices, 0, bin_edges.size - 2)


def _mean_with_nan(sum_values: np.ndarray, count_values: np.ndarray) -> np.ndarray:
    mean = np.full(sum_values.shape, np.nan, dtype=float)
    valid = np.asarray(count_values, dtype=int) > 0
    if np.any(valid):
        mean[valid] = np.asarray(sum_values[valid], dtype=float) / np.asarray(
            count_values[valid],
            dtype=float,
        )
    return mean


def _determine_distance_bounds(
    *,
    frames: list[Atoms],
    axis_index: int,
    frame_data: list[_OrientationFrameData],
    binning: str,
    coordinate_mode: str,
    surface_per_frame: np.ndarray | None,
    bin_width: float,
) -> tuple[float, float]:
    """Return ``(min, max)`` of the cached water-COM distance coordinate."""
    normalized_binning = binning.strip().lower()
    if normalized_binning == "cell":
        bounds = _cell_histogram_bounds(
            frames=frames,
            axis_index=axis_index,
            coordinate_mode=coordinate_mode,
            surface_per_frame=surface_per_frame,
        )
        if bounds is not None:
            return bounds
        LOGGER.warning(
            "Cell binning requested for orientation along %s, but a usable cell was unavailable. "
            "Falling back to observed-data bounds.",
            "xyz"[axis_index],
        )

    global_min = float("inf")
    global_max = float("-inf")
    for data in frame_data:
        distances = np.asarray(data.distances, dtype=float)
        finite = distances[np.isfinite(distances)]
        if finite.size > 0:
            global_min = min(global_min, float(np.min(finite)))
            global_max = max(global_max, float(np.max(finite)))
    if not np.isfinite(global_min):
        global_min = 0.0
    if not np.isfinite(global_max):
        global_max = global_min + float(bin_width)
    return global_min, global_max


def _compute_number_density(
    *,
    count_total: np.ndarray,
    n_frames: int,
    bin_width: float,
    slab_volumes: np.ndarray | None,
    framewise_density_sum: np.ndarray | None,
) -> np.ndarray:
    """Convert raw water counts into volumetric or linear molecular density."""
    if slab_volumes is not None:
        if framewise_density_sum is not None:
            return framewise_density_sum / float(n_frames)
        return np.asarray(count_total, dtype=float) / (float(n_frames) * float(slab_volumes[0]))
    return np.asarray(count_total, dtype=float) / (float(n_frames) * float(bin_width))


def _cartesian_grid_edges_for_orientation(
    *,
    frames: list[Atoms],
    frame_data: list[_OrientationFrameData],
    bin_width: float,
    cell_lengths: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_lengths is not None:
        return tuple(
            _build_uniform_bin_edges(0.0, float(length), bin_width=bin_width)
            for length in cell_lengths
        )  # type: ignore[return-value]

    positions = [
        np.asarray(data.positions, dtype=float)
        for data in frame_data
        if np.asarray(data.positions).size > 0
    ]
    if positions:
        all_positions = np.vstack(positions)
        edges: list[np.ndarray] = []
        for axis_index in range(3):
            values = all_positions[:, axis_index]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                lower = 0.0
                upper = float(bin_width)
            else:
                lower = float(np.min(finite))
                upper = float(np.max(finite))
            edges.append(_build_uniform_bin_edges(lower, upper, bin_width=bin_width))
        return edges[0], edges[1], edges[2]

    return (
        _build_uniform_bin_edges(0.0, float(bin_width), bin_width=bin_width),
        _build_uniform_bin_edges(0.0, float(bin_width), bin_width=bin_width),
        _build_uniform_bin_edges(0.0, float(bin_width), bin_width=bin_width),
    )


def _bin_indices_and_mask(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    indices = np.searchsorted(edges, values, side="right") - 1
    if edges.size >= 2:
        upper_match = np.isfinite(values) & np.isclose(values, edges[-1], rtol=0.0, atol=1.0e-12)
        indices[upper_match] = edges.size - 2
    mask = (indices >= 0) & (indices < edges.size - 1) & np.isfinite(values)
    return indices.astype(np.int64, copy=False), mask


def _sparse_sum_by_flat_index(
    flat_indices: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    if flat_indices.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    unique, inverse = np.unique(flat_indices, return_inverse=True)
    if weights is None:
        summed = np.bincount(inverse).astype(float)
    else:
        summed = np.bincount(inverse, weights=np.asarray(weights, dtype=float)).astype(float)
    return unique.astype(np.int64, copy=False), summed


def _align_sparse_values(
    base_indices: np.ndarray,
    value_indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    out = np.zeros(base_indices.shape, dtype=float)
    if value_indices.size == 0:
        return out
    positions = np.searchsorted(base_indices, value_indices)
    valid_bounds = (positions >= 0) & (positions < base_indices.size)
    valid = np.zeros(value_indices.shape, dtype=bool)
    if np.any(valid_bounds):
        valid[valid_bounds] = base_indices[positions[valid_bounds]] == value_indices[valid_bounds]
    out[positions[valid]] = np.asarray(values, dtype=float)[valid]
    return out


def _build_orientation_sparse_grid(
    *,
    frames: list[Atoms],
    frame_data: list[_OrientationFrameData],
    distance_edges: np.ndarray,
    bin_width: float,
    cell_lengths: tuple[float, float, float] | None,
) -> OrientationSparseGrid:
    x_edges, y_edges, z_edges = _cartesian_grid_edges_for_orientation(
        frames=frames,
        frame_data=frame_data,
        bin_width=bin_width,
        cell_lengths=cell_lengths,
    )
    d_edges = np.asarray(distance_edges, dtype=float)
    shape = (
        max(0, x_edges.size - 1),
        max(0, y_edges.size - 1),
        max(0, z_edges.size - 1),
        max(0, d_edges.size - 1),
    )
    entity_blocks: list[np.ndarray] = []
    polar_blocks: list[np.ndarray] = []
    polar_value_blocks: list[np.ndarray] = []
    azimuthal_blocks: list[np.ndarray] = []
    azimuthal_value_blocks: list[np.ndarray] = []

    nx, ny, nz, nd = shape
    if min(shape) <= 0:
        return OrientationSparseGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
            distance_edges=d_edges,
            shape=shape,
            flat_indices=np.array([], dtype=np.int64),
            entity_sum=np.array([], dtype=np.float32),
            cos_polar_sum=np.array([], dtype=np.float32),
            count_polar_valid=np.array([], dtype=np.float32),
            cos_azimuthal_sum=np.array([], dtype=np.float32),
            count_azimuthal_valid=np.array([], dtype=np.float32),
        )

    for data in frame_data:
        positions = np.asarray(data.positions, dtype=float)
        if positions.size == 0:
            continue
        ix, mx = _bin_indices_and_mask(positions[:, 0], x_edges)
        iy, my = _bin_indices_and_mask(positions[:, 1], y_edges)
        iz, mz = _bin_indices_and_mask(positions[:, 2], z_edges)
        idist, md = _bin_indices_and_mask(data.distances, d_edges)
        mask = mx & my & mz & md
        if not np.any(mask):
            continue
        flat = (((ix[mask] * ny + iy[mask]) * nz + iz[mask]) * nd + idist[mask]).astype(
            np.int64,
            copy=False,
        )
        entity_blocks.append(flat)

        polar_mask = mask & np.asarray(data.polar_valid, dtype=bool)
        if np.any(polar_mask):
            polar_flat = (
                ((ix[polar_mask] * ny + iy[polar_mask]) * nz + iz[polar_mask]) * nd
                + idist[polar_mask]
            ).astype(np.int64, copy=False)
            polar_blocks.append(polar_flat)
            polar_value_blocks.append(np.asarray(data.cos_polar[polar_mask], dtype=float))

        azimuthal_mask = mask & np.asarray(data.azimuthal_valid, dtype=bool)
        if np.any(azimuthal_mask):
            azimuthal_flat = (
                ((ix[azimuthal_mask] * ny + iy[azimuthal_mask]) * nz + iz[azimuthal_mask]) * nd
                + idist[azimuthal_mask]
            ).astype(np.int64, copy=False)
            azimuthal_blocks.append(azimuthal_flat)
            azimuthal_value_blocks.append(np.asarray(data.cos_azimuthal[azimuthal_mask], dtype=float))

    if not entity_blocks:
        flat_indices = np.array([], dtype=np.int64)
        entity_sum = np.array([], dtype=float)
    else:
        flat_indices, entity_sum = _sparse_sum_by_flat_index(np.concatenate(entity_blocks))

    polar_flat = np.concatenate(polar_blocks) if polar_blocks else np.array([], dtype=np.int64)
    polar_values = (
        np.concatenate(polar_value_blocks) if polar_value_blocks else np.array([], dtype=float)
    )
    polar_indices, polar_sum = _sparse_sum_by_flat_index(polar_flat, weights=polar_values)
    polar_count_indices, polar_count = _sparse_sum_by_flat_index(polar_flat)

    azimuthal_flat = (
        np.concatenate(azimuthal_blocks) if azimuthal_blocks else np.array([], dtype=np.int64)
    )
    azimuthal_values = (
        np.concatenate(azimuthal_value_blocks)
        if azimuthal_value_blocks
        else np.array([], dtype=float)
    )
    azimuthal_indices, azimuthal_sum = _sparse_sum_by_flat_index(
        azimuthal_flat,
        weights=azimuthal_values,
    )
    azimuthal_count_indices, azimuthal_count = _sparse_sum_by_flat_index(azimuthal_flat)

    return OrientationSparseGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        distance_edges=d_edges,
        shape=shape,
        flat_indices=flat_indices,
        entity_sum=entity_sum.astype(np.float32),
        cos_polar_sum=_align_sparse_values(flat_indices, polar_indices, polar_sum).astype(np.float32),
        count_polar_valid=_align_sparse_values(
            flat_indices,
            polar_count_indices,
            polar_count,
        ).astype(np.float32),
        cos_azimuthal_sum=_align_sparse_values(
            flat_indices,
            azimuthal_indices,
            azimuthal_sum,
        ).astype(np.float32),
        count_azimuthal_valid=_align_sparse_values(
            flat_indices,
            azimuthal_count_indices,
            azimuthal_count,
        ).astype(np.float32),
    )


def compute_orientation_profile(
    frames: list[Atoms],
    *,
    axis: str = "z",
    reference_axis: str = "z",
    bin_width: float = 0.1,
    angle_bin_count: int = _DEFAULT_ANGLE_BIN_COUNT,
    surface_mode: str = "auto",
    surface_elements: list[str] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
    binning: str = "cell",
    oh_cutoff: float = H2O_OH_CUTOFF_A,
) -> OrientationProfile:
    """Compute water-orientation profiles from a trajectory."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    axis_index = axis_to_index(axis)
    ref_index = axis_to_index(reference_axis)
    ref_vec = np.zeros(3, dtype=float)
    ref_vec[ref_index] = 1.0
    in_plane_ref_vec = _in_plane_reference_vector(ref_index)

    surface_estimate: SurfaceEstimate | None
    if precomputed_surface_estimate is not None:
        if precomputed_surface_estimate.frame_values.shape != (len(frames),):
            raise ValueError(
                "precomputed_surface_estimate frame count does not match the trajectory."
            )
        surface_estimate = precomputed_surface_estimate
    else:
        from .density import _select_surface_estimate

        surface_estimate, _method = _select_surface_estimate(
            frames,
            axis,
            mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
        )
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_per_frame: np.ndarray | None = None
    if surface_estimate is not None:
        surface_position = surface_estimate.position
        surface_position_std = surface_estimate.std
        surface_per_frame = surface_estimate.per_frame
        if not _surface_estimate_supports_distance_mode(surface_estimate, frame_count=len(frames)):
            surface_per_frame = None

    coordinate_mode = "distance" if surface_per_frame is not None else "axis"
    topology_cache = OHTopologyCache(
        oh_cutoff=oh_cutoff,
        logger=LOGGER,
        context="H2O topology",
    )
    n_molecules_per_frame = 0
    frame_data: list[_OrientationFrameData] = []

    with ProgressBar(desc="Computing orientation", total=len(frames), unit="frame") as progress:
        for frame_idx, frame in enumerate(frames):
            cached_triplets = topology_cache.select(
                frame,
                frame_index=frame_idx,
            ).indices_for("mol:H2O")
            geom = water_triplet_geometry(frame, cached_triplets)
            n_mol = geom.com_positions.shape[0]
            if frame_idx == 0 or (n_molecules_per_frame == 0 and n_mol > 0):
                n_molecules_per_frame = n_mol
            frame_data.append(
                _build_orientation_frame_data(
                    geom,
                    axis_index=axis_index,
                    surface_position=None
                    if surface_per_frame is None
                    else surface_per_frame[frame_idx],
                    ref_vec=ref_vec,
                    in_plane_ref_vec=in_plane_ref_vec,
                    oh_cutoff=oh_cutoff,
                )
            )
            progress.update()

    LOGGER.info(
        "Orientation frame analysis complete: %d frames, %d H2O/frame. "
        "Aggregating cached frame data into distance and angle bins.",
        len(frame_data),
        n_molecules_per_frame,
    )

    histogram_bounds = _determine_distance_bounds(
        frames=frames,
        axis_index=axis_index,
        frame_data=frame_data,
        binning=binning,
        coordinate_mode=coordinate_mode,
        surface_per_frame=surface_per_frame,
        bin_width=bin_width,
    )
    dist_bin_edges = _build_uniform_bin_edges(
        histogram_bounds[0],
        histogram_bounds[1],
        bin_width=bin_width,
    )
    n_dist_bins = len(dist_bin_edges) - 1
    if n_dist_bins < 1:
        raise ValueError(
            "No distance bins produced "
            f"(range [{histogram_bounds[0]:.3f}, {histogram_bounds[1]:.3f}], "
            f"bin_width={bin_width})."
        )
    dist_bin_centers = 0.5 * (dist_bin_edges[:-1] + dist_bin_edges[1:])

    angle_bin_edges = np.linspace(-1.0, 1.0, angle_bin_count + 1)
    angle_bin_centers = 0.5 * (angle_bin_edges[:-1] + angle_bin_edges[1:])

    cos_polar_sum = np.zeros(n_dist_bins, dtype=float)
    cos_azimuthal_sum = np.zeros(n_dist_bins, dtype=float)
    count_total = np.zeros(n_dist_bins, dtype=int)
    count_polar_valid = np.zeros(n_dist_bins, dtype=int)
    count_azimuthal_valid = np.zeros(n_dist_bins, dtype=int)
    heatmap_polar = np.zeros((n_dist_bins, angle_bin_count), dtype=float)
    heatmap_azimuthal = np.zeros((n_dist_bins, angle_bin_count), dtype=float)

    slab_volumes = _per_frame_slab_volumes(frames, axis_index=axis_index, bin_width=bin_width)
    variable_slab_volume = (
        slab_volumes is not None
        and slab_volumes.size > 1
        and not np.allclose(slab_volumes, slab_volumes[0], rtol=0.0, atol=1.0e-12)
    )
    framewise_density_sum = (
        np.zeros(n_dist_bins, dtype=float)
        if slab_volumes is not None and variable_slab_volume
        else None
    )
    cell_lengths = _representative_cell_lengths(frames)
    sparse_grid = _build_orientation_sparse_grid(
        frames=frames,
        frame_data=frame_data,
        distance_edges=dist_bin_edges,
        bin_width=bin_width,
        cell_lengths=cell_lengths,
    )
    dense_grid_cells = int(np.prod(np.asarray(sparse_grid.shape, dtype=np.int64)))
    sparse_nonzero = int(sparse_grid.flat_indices.size)
    LOGGER.info(
        "Orientation sparse grid: dense=%d cells, sparse=%d nonzero bins.",
        dense_grid_cells,
        sparse_nonzero,
    )
    sample_cos_polar_mean = np.full((len(frames), n_dist_bins), np.nan, dtype=float)
    sample_cos_azimuthal_mean = np.full((len(frames), n_dist_bins), np.nan, dtype=float)
    sample_density = np.full((len(frames), n_dist_bins), np.nan, dtype=float)
    sample_cos_polar_density = np.full((len(frames), n_dist_bins), np.nan, dtype=float)
    sample_cos_azimuthal_density = np.full((len(frames), n_dist_bins), np.nan, dtype=float)

    with ProgressBar(
        desc="Aggregating orientation bins",
        total=len(frame_data),
        unit="frame",
    ) as progress:
        for frame_idx, data in enumerate(frame_data):
            total_mask = _distance_bin_membership_mask(data.distances, dist_bin_edges)
            frame_total_hist = np.zeros(n_dist_bins, dtype=int)
            frame_polar_count = np.zeros(n_dist_bins, dtype=int)
            frame_azimuthal_count = np.zeros(n_dist_bins, dtype=int)
            frame_polar_sum = np.zeros(n_dist_bins, dtype=float)
            frame_azimuthal_sum = np.zeros(n_dist_bins, dtype=float)
            if np.any(total_mask):
                dist_idx_total = _distance_bin_indices(data.distances[total_mask], dist_bin_edges)
                frame_total_hist = np.bincount(dist_idx_total, minlength=n_dist_bins).astype(int)
                count_total += frame_total_hist
                if framewise_density_sum is not None and slab_volumes is not None:
                    framewise_density_sum += frame_total_hist / float(slab_volumes[frame_idx])

            polar_mask = total_mask & data.polar_valid
            if np.any(polar_mask):
                dist_idx_polar = _distance_bin_indices(data.distances[polar_mask], dist_bin_edges)
                polar_values = np.asarray(data.cos_polar[polar_mask], dtype=float)
                frame_polar_count = np.bincount(dist_idx_polar, minlength=n_dist_bins).astype(int)
                count_polar_valid += frame_polar_count
                frame_polar_sum = np.bincount(
                    dist_idx_polar,
                    weights=polar_values,
                    minlength=n_dist_bins,
                )
                cos_polar_sum += frame_polar_sum
                angle_idx_polar = _angle_bin_indices(polar_values, angle_bin_edges)
                np.add.at(heatmap_polar, (dist_idx_polar, angle_idx_polar), 1.0)

            azimuthal_mask = total_mask & data.azimuthal_valid
            if np.any(azimuthal_mask):
                dist_idx_azimuthal = _distance_bin_indices(
                    data.distances[azimuthal_mask],
                    dist_bin_edges,
                )
                azimuthal_values = np.asarray(data.cos_azimuthal[azimuthal_mask], dtype=float)
                frame_azimuthal_count = np.bincount(
                    dist_idx_azimuthal,
                    minlength=n_dist_bins,
                ).astype(int)
                count_azimuthal_valid += frame_azimuthal_count
                frame_azimuthal_sum = np.bincount(
                    dist_idx_azimuthal,
                    weights=azimuthal_values,
                    minlength=n_dist_bins,
                )
                cos_azimuthal_sum += frame_azimuthal_sum
                angle_idx_azimuthal = _angle_bin_indices(azimuthal_values, angle_bin_edges)
                np.add.at(heatmap_azimuthal, (dist_idx_azimuthal, angle_idx_azimuthal), 1.0)

            frame_density = (
                np.asarray(frame_total_hist, dtype=float) / float(slab_volumes[frame_idx])
                if slab_volumes is not None
                else np.asarray(frame_total_hist, dtype=float) / float(bin_width)
            )
            frame_cos_polar_mean = _mean_with_nan(frame_polar_sum, frame_polar_count)
            frame_cos_azimuthal_mean = _mean_with_nan(frame_azimuthal_sum, frame_azimuthal_count)
            sample_density[frame_idx] = frame_density
            sample_cos_polar_mean[frame_idx] = frame_cos_polar_mean
            sample_cos_azimuthal_mean[frame_idx] = frame_cos_azimuthal_mean
            sample_cos_polar_density[frame_idx] = frame_cos_polar_mean * frame_density
            sample_cos_azimuthal_density[frame_idx] = frame_cos_azimuthal_mean * frame_density
            progress.update()

    n_frames = len(frames)
    cos_polar_mean = _mean_with_nan(cos_polar_sum, count_polar_valid)
    cos_azimuthal_mean = _mean_with_nan(cos_azimuthal_sum, count_azimuthal_valid)
    density = _compute_number_density(
        count_total=count_total,
        n_frames=n_frames,
        bin_width=bin_width,
        slab_volumes=slab_volumes,
        framewise_density_sum=framewise_density_sum,
    )
    block_resolution = resolve_block_slices(n_frames)
    block_slices = None if block_resolution is None else block_resolution[0]
    series_statistics = {
        "cos_polar_mean": build_series_statistics(
            point_count=count_polar_valid,
            sample_values=sample_cos_polar_mean,
            block_values=block_mean_matrix(sample_cos_polar_mean, block_slices=block_slices),
        ),
        "cos_azimuthal_mean": build_series_statistics(
            point_count=count_azimuthal_valid,
            sample_values=sample_cos_azimuthal_mean,
            block_values=block_mean_matrix(sample_cos_azimuthal_mean, block_slices=block_slices),
        ),
        "density": build_series_statistics(
            point_count=count_total,
            sample_values=sample_density,
            block_values=block_mean_matrix(sample_density, block_slices=block_slices),
        ),
        "cos_polar_density": build_series_statistics(
            point_count=count_polar_valid,
            sample_values=sample_cos_polar_density,
            block_values=block_mean_matrix(sample_cos_polar_density, block_slices=block_slices),
        ),
        "cos_azimuthal_density": build_series_statistics(
            point_count=count_azimuthal_valid,
            sample_values=sample_cos_azimuthal_density,
            block_values=block_mean_matrix(sample_cos_azimuthal_density, block_slices=block_slices),
        ),
    }

    return OrientationProfile(
        axis=axis,
        reference_axis=reference_axis,
        n_frames=n_frames,
        n_molecules_per_frame=n_molecules_per_frame,
        bin_edges=dist_bin_edges,
        bin_centers=dist_bin_centers,
        cos_polar_mean=cos_polar_mean,
        cos_azimuthal_mean=cos_azimuthal_mean,
        count_total=count_total,
        count_polar_valid=count_polar_valid,
        count_azimuthal_valid=count_azimuthal_valid,
        cos_polar_density=cos_polar_mean * density,
        cos_azimuthal_density=cos_azimuthal_mean * density,
        density=density,
        heatmap_polar=heatmap_polar,
        heatmap_azimuthal=heatmap_azimuthal,
        heatmap_angle_bin_edges=angle_bin_edges,
        heatmap_angle_bin_centers=angle_bin_centers,
        coordinate_mode=coordinate_mode,
        surface_position=surface_position,
        surface_position_std=surface_position_std,
        surface_estimate=surface_estimate,
        cell_lengths_angstrom=cell_lengths,
        series_statistics=series_statistics,
        sparse_grid=sparse_grid,
    )


_ANALYSIS_NAME = "orientation"


def _orientation_profile_hdf5_payload(
    profile: OrientationProfile,
) -> dict[str, Any]:
    metadata_payload = {
        "species": "H2O",
        "axis": profile.axis,
        "reference_axis": profile.reference_axis,
        "n_frames": int(profile.n_frames),
        "n_molecules_per_frame": int(profile.n_molecules_per_frame),
        "coordinate_mode": profile.coordinate_mode,
        "cell_lengths_angstrom": (
            None
            if profile.cell_lengths_angstrom is None
            else [float(v) for v in profile.cell_lengths_angstrom]
        ),
        **_surface_metadata_payload(
            surface_position=profile.surface_position,
            surface_position_std=profile.surface_position_std,
            estimate=profile.surface_estimate,
        ),
    }
    if profile.sparse_grid is not None:
        metadata_payload["orientation_grid_kind"] = "orientation_grid_sparse"
        metadata_payload["orientation_grid_shape"] = [
            int(value) for value in profile.sparse_grid.shape
        ]
    if profile.series_statistics:
        block_resolution = resolve_block_slices(int(profile.n_frames))
        block_lengths = None if block_resolution is None else block_resolution[1]
        metadata_payload["statistics"] = build_statistics_metadata(
            statistics_by_series=profile.series_statistics,
            block_lengths=block_lengths,
        )
    metadata = build_profile_metadata(
        analysis=_ANALYSIS_NAME,
        metadata=metadata_payload,
    )
    datasets: dict[str, np.ndarray | None] = {
        "bin_edges_A": profile.bin_edges,
        "bin_centers_A": profile.bin_centers,
        "cos_polar_mean": profile.cos_polar_mean,
        "cos_azimuthal_mean": profile.cos_azimuthal_mean,
        "count_total": np.asarray(profile.count_total, dtype=int),
        "count_polar_valid": np.asarray(profile.count_polar_valid, dtype=int),
        "count_azimuthal_valid": np.asarray(profile.count_azimuthal_valid, dtype=int),
        "cos_polar_density": profile.cos_polar_density,
        "cos_azimuthal_density": profile.cos_azimuthal_density,
        "density": profile.density,
        "heatmap_polar": profile.heatmap_polar,
        "heatmap_azimuthal": profile.heatmap_azimuthal,
        "heatmap_angle_bin_edges": profile.heatmap_angle_bin_edges,
        "heatmap_angle_bin_centers": profile.heatmap_angle_bin_centers,
        "grid_x_edges_A": None
        if profile.sparse_grid is None
        else profile.sparse_grid.x_edges,
        "grid_y_edges_A": None
        if profile.sparse_grid is None
        else profile.sparse_grid.y_edges,
        "grid_z_edges_A": None
        if profile.sparse_grid is None
        else profile.sparse_grid.z_edges,
        "grid_distance_edges_A": None
        if profile.sparse_grid is None
        else profile.sparse_grid.distance_edges,
        "grid_flat_indices": None
        if profile.sparse_grid is None
        else profile.sparse_grid.flat_indices,
        "grid_entity_sum": None
        if profile.sparse_grid is None
        else profile.sparse_grid.entity_sum,
        "grid_cos_polar_sum": None
        if profile.sparse_grid is None
        else profile.sparse_grid.cos_polar_sum,
        "grid_count_polar_valid": None
        if profile.sparse_grid is None
        else profile.sparse_grid.count_polar_valid,
        "grid_cos_azimuthal_sum": None
        if profile.sparse_grid is None
        else profile.sparse_grid.cos_azimuthal_sum,
        "grid_count_azimuthal_valid": None
        if profile.sparse_grid is None
        else profile.sparse_grid.count_azimuthal_valid,
        **_surface_estimate_datasets(profile.surface_estimate),
        **statistics_payload_from_series_map(profile.series_statistics),
    }
    return {"datasets": datasets, "metadata": metadata}


def save_orientation_profile(
    profile: OrientationProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save an orientation profile to LiNaK HDF5."""
    payload = _orientation_profile_hdf5_payload(profile)
    metadata = dict(payload["metadata"])
    if additional_metadata:
        metadata.update(dict(additional_metadata))
    output_path = write_linak_hdf5(
        output,
        analysis=_ANALYSIS_NAME,
        datasets=payload["datasets"],
        metadata=metadata,
    )
    try:
        _display = os.path.relpath(output_path)
    except ValueError:
        _display = str(output_path)
    LOGGER.info("Saved orientation data to '%s'.", _display)
    return output_path


def _build_orientation_profile_from_hdf5(
    datasets: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> OrientationProfile:
    def _get(name: str) -> np.ndarray:
        arr = datasets.get(name)
        if arr is None:
            raise ValueError(f"Missing dataset '{name}' in orientation HDF5.")
        return np.asarray(arr, dtype=float)

    cell_raw = metadata.get("cell_lengths_angstrom")
    cell_lengths: tuple[float, float, float] | None = None
    if cell_raw is not None:
        try:
            cell_lengths = (float(cell_raw[0]), float(cell_raw[1]), float(cell_raw[2]))
        except (TypeError, IndexError, ValueError):
            cell_lengths = None
    surface_estimate = _surface_estimate_from_payload(datasets=datasets, metadata=metadata)
    surface_metadata = _surface_metadata_view(metadata)
    heatmap_polar = _get("heatmap_polar")
    heatmap_azimuthal = _get("heatmap_azimuthal")
    count_total = np.asarray(_get("count_total"), dtype=int)
    count_polar_valid = np.asarray(_get("count_polar_valid"), dtype=int)
    count_azimuthal_valid = np.asarray(_get("count_azimuthal_valid"), dtype=int)
    series_statistics = statistics_series_map_from_datasets(
        datasets,
        dataset_names=(
            "cos_polar_mean",
            "cos_azimuthal_mean",
            "density",
            "cos_polar_density",
            "cos_azimuthal_density",
        ),
    )
    grid_dataset_names = (
        "grid_x_edges_A",
        "grid_y_edges_A",
        "grid_z_edges_A",
        "grid_distance_edges_A",
        "grid_flat_indices",
        "grid_entity_sum",
        "grid_cos_polar_sum",
        "grid_count_polar_valid",
        "grid_cos_azimuthal_sum",
        "grid_count_azimuthal_valid",
    )
    present_grid_names = [name for name in grid_dataset_names if name in datasets]
    sparse_grid: OrientationSparseGrid | None = None
    if present_grid_names:
        missing_grid_names = [name for name in grid_dataset_names if name not in datasets]
        if missing_grid_names:
            raise ValueError(
                "Incomplete orientation sparse grid in HDF5; missing dataset(s): "
                + ", ".join(missing_grid_names)
            )
        shape_raw = metadata.get("orientation_grid_shape")
        if shape_raw is None:
            shape = (
                int(np.asarray(datasets["grid_x_edges_A"]).size - 1),
                int(np.asarray(datasets["grid_y_edges_A"]).size - 1),
                int(np.asarray(datasets["grid_z_edges_A"]).size - 1),
                int(np.asarray(datasets["grid_distance_edges_A"]).size - 1),
            )
        else:
            shape_values = [int(value) for value in shape_raw]
            if len(shape_values) != 4:
                raise ValueError("orientation_grid_shape must contain four dimensions.")
            shape = tuple(shape_values)  # type: ignore[assignment]
        sparse_grid = OrientationSparseGrid(
            x_edges=np.asarray(datasets["grid_x_edges_A"], dtype=float),
            y_edges=np.asarray(datasets["grid_y_edges_A"], dtype=float),
            z_edges=np.asarray(datasets["grid_z_edges_A"], dtype=float),
            distance_edges=np.asarray(datasets["grid_distance_edges_A"], dtype=float),
            shape=shape,
            flat_indices=np.asarray(datasets["grid_flat_indices"], dtype=np.int64),
            entity_sum=np.asarray(datasets["grid_entity_sum"], dtype=float),
            cos_polar_sum=np.asarray(datasets["grid_cos_polar_sum"], dtype=float),
            count_polar_valid=np.asarray(datasets["grid_count_polar_valid"], dtype=float),
            cos_azimuthal_sum=np.asarray(datasets["grid_cos_azimuthal_sum"], dtype=float),
            count_azimuthal_valid=np.asarray(
                datasets["grid_count_azimuthal_valid"],
                dtype=float,
            ),
        )

    return OrientationProfile(
        axis=str(metadata.get("axis", "z")),
        reference_axis=str(metadata.get("reference_axis", "z")),
        n_frames=int(metadata.get("n_frames", 0)),
        n_molecules_per_frame=int(metadata.get("n_molecules_per_frame", 0)),
        bin_edges=_get("bin_edges_A"),
        bin_centers=_get("bin_centers_A"),
        cos_polar_mean=_get("cos_polar_mean"),
        cos_azimuthal_mean=_get("cos_azimuthal_mean"),
        count_total=count_total,
        count_polar_valid=count_polar_valid,
        count_azimuthal_valid=count_azimuthal_valid,
        cos_polar_density=_get("cos_polar_density"),
        cos_azimuthal_density=_get("cos_azimuthal_density"),
        density=_get("density"),
        heatmap_polar=heatmap_polar,
        heatmap_azimuthal=heatmap_azimuthal,
        heatmap_angle_bin_edges=_get("heatmap_angle_bin_edges"),
        heatmap_angle_bin_centers=_get("heatmap_angle_bin_centers"),
        coordinate_mode=str(metadata.get("coordinate_mode", "axis")),
        surface_position=surface_metadata.get("position", metadata.get("surface_position")),
        surface_position_std=surface_metadata.get(
            "position_std", metadata.get("surface_position_std")
        ),
        surface_estimate=surface_estimate,
        cell_lengths_angstrom=cell_lengths,
        series_statistics=series_statistics,
        sparse_grid=sparse_grid,
    )


def load_orientation_profile(path: str | Path) -> OrientationProfile:
    """Load a single orientation profile from LiNaK HDF5."""
    profiles = load_orientation_profiles(path)
    if not profiles:
        raise ValueError(f"No orientation profiles found in '{path}'.")
    return profiles[0]


def load_orientation_profiles(path: str | Path) -> list[OrientationProfile]:
    """Load all orientation profiles from a LiNaK HDF5 file."""
    _source_path, raw_profiles = read_profile_payloads(
        path,
        analysis=_ANALYSIS_NAME,
        label="Orientation",
    )
    return [
        _build_orientation_profile_from_hdf5(datasets, metadata)
        for datasets, metadata in raw_profiles
    ]


def load_orientation_profiles_by_index(
    path: str | Path,
    indices: list[int],
) -> list[OrientationProfile]:
    """Load selected orientation profiles by index."""
    _source_path, raw = read_profile_payloads_by_index(
        path,
        indices,
        analysis=_ANALYSIS_NAME,
        label="Orientation",
    )
    return [_build_orientation_profile_from_hdf5(datasets, metadata) for datasets, metadata in raw]


# Plotting helpers

_ANGLE_CHOICES = ("polar", "azimuthal")
_COMPONENT_CHOICES = ("average", "density", "density-weighted", "heatmap")
_ORIENTATION_GRID_AXES = ("x", "y", "z", "distance")


def _orientation_grid_edges(grid: OrientationSparseGrid, axis: str) -> np.ndarray:
    normalized = str(axis).strip().lower()
    if normalized == "x":
        return np.asarray(grid.x_edges, dtype=float)
    if normalized == "y":
        return np.asarray(grid.y_edges, dtype=float)
    if normalized == "z":
        return np.asarray(grid.z_edges, dtype=float)
    if normalized == "distance":
        return np.asarray(grid.distance_edges, dtype=float)
    raise ValueError(
        "Orientation grid axis must be one of: " + ", ".join(_ORIENTATION_GRID_AXES) + "."
    )


def _orientation_grid_centers(grid: OrientationSparseGrid, axis: str) -> np.ndarray:
    edges = _orientation_grid_edges(grid, axis)
    return 0.5 * (edges[:-1] + edges[1:])


def _orientation_grid_axis_indices(grid: OrientationSparseGrid) -> dict[str, np.ndarray]:
    flat = np.asarray(grid.flat_indices, dtype=np.int64)
    nx, ny, nz, nd = (int(value) for value in grid.shape)
    if flat.size == 0:
        empty = np.array([], dtype=np.int64)
        return {"x": empty, "y": empty, "z": empty, "distance": empty}
    if min(nx, ny, nz, nd) <= 0:
        raise ValueError("Orientation sparse grid has an invalid zero-length axis.")
    distance_idx = flat % nd
    tmp = flat // nd
    z_idx = tmp % nz
    tmp = tmp // nz
    y_idx = tmp % ny
    x_idx = tmp // ny
    return {
        "x": x_idx.astype(np.int64, copy=False),
        "y": y_idx.astype(np.int64, copy=False),
        "z": z_idx.astype(np.int64, copy=False),
        "distance": distance_idx.astype(np.int64, copy=False),
    }


def _orientation_filter_bounds(
    filters: Mapping[str, tuple[float | None, float | None] | None] | None,
) -> dict[str, tuple[float | None, float | None]]:
    resolved: dict[str, tuple[float | None, float | None]] = {}
    for axis, bounds in (filters or {}).items():
        normalized_axis = str(axis).strip().lower()
        if normalized_axis not in _ORIENTATION_GRID_AXES:
            raise ValueError(
                "Orientation grid filter axis must be one of: "
                + ", ".join(_ORIENTATION_GRID_AXES)
                + "."
            )
        if bounds is None:
            continue
        lower, upper = bounds
        lower_value = None if lower is None else float(lower)
        upper_value = None if upper is None else float(upper)
        if (
            lower_value is not None
            and upper_value is not None
            and lower_value > upper_value
        ):
            raise ValueError(f"{normalized_axis} filter minimum must not exceed maximum.")
        resolved[normalized_axis] = (lower_value, upper_value)
    return resolved


def _orientation_grid_filter_mask(
    grid: OrientationSparseGrid,
    *,
    filters: Mapping[str, tuple[float | None, float | None] | None] | None,
    indices_by_axis: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = _orientation_grid_axis_indices(grid) if indices_by_axis is None else indices_by_axis
    flat_size = np.asarray(grid.flat_indices).size
    mask = np.ones(flat_size, dtype=bool)
    for axis, (lower, upper) in _orientation_filter_bounds(filters).items():
        centers = _orientation_grid_centers(grid, axis)
        axis_indices = indices[axis]
        values = centers[axis_indices]
        if lower is not None:
            mask &= values >= lower
        if upper is not None:
            mask &= values <= upper
    return mask, indices


def _orientation_grid_sum_and_count(
    grid: OrientationSparseGrid,
    angle: str,
) -> tuple[np.ndarray, np.ndarray]:
    norm_angle = _normalize_angle_token(angle)
    if norm_angle == "polar":
        return (
            np.asarray(grid.cos_polar_sum, dtype=float),
            np.asarray(grid.count_polar_valid, dtype=float),
        )
    return (
        np.asarray(grid.cos_azimuthal_sum, dtype=float),
        np.asarray(grid.count_azimuthal_valid, dtype=float),
    )


def orientation_sparse_grid_to_line(
    profile: OrientationProfile,
    *,
    x_axis: str = "distance",
    angle: str = "polar",
    filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive a filtered 1D orientation line from a sparse orientation grid.

    Returns ``(x_edges, x_centers, y_values)``. Data ranges are selection filters, not
    axis-limit controls; excluded bins remain present with ``NaN`` values.
    """
    grid = profile.sparse_grid
    if grid is None:
        raise ValueError(
            "Orientation ranges require an orientation sparse grid. Recompute orientation "
            "with a LiNaK version that writes orientation grid data."
        )
    normalized_x_axis = str(x_axis).strip().lower()
    x_edges = _orientation_grid_edges(grid, normalized_x_axis)
    x_centers = _orientation_grid_centers(grid, normalized_x_axis)
    indices_by_axis = _orientation_grid_axis_indices(grid)
    mask, indices_by_axis = _orientation_grid_filter_mask(
        grid,
        filters=filters,
        indices_by_axis=indices_by_axis,
    )
    value_sum, value_count = _orientation_grid_sum_and_count(grid, angle)
    axis_indices = indices_by_axis[normalized_x_axis]
    n_bins = x_edges.size - 1
    out_sum = np.zeros(n_bins, dtype=float)
    out_count = np.zeros(n_bins, dtype=float)
    if np.any(mask):
        out_sum += np.bincount(
            axis_indices[mask],
            weights=value_sum[mask],
            minlength=n_bins,
        )
        out_count += np.bincount(
            axis_indices[mask],
            weights=value_count[mask],
            minlength=n_bins,
        )
    y_values = np.full(n_bins, np.nan, dtype=float)
    valid = out_count > 0.0
    y_values[valid] = out_sum[valid] / out_count[valid]
    return x_edges, x_centers, y_values


def orientation_sparse_grid_to_heatmap(
    profile: OrientationProfile,
    *,
    x_axis: str = "x",
    y_axis: str = "y",
    angle: str = "polar",
    filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive a filtered 2D orientation heatmap from a sparse orientation grid."""
    x_edges, y_edges, out_sum, out_count = _orientation_sparse_grid_to_heatmap_accumulators(
        profile,
        x_axis=x_axis,
        y_axis=y_axis,
        angle=angle,
        filters=filters,
    )
    heatmap = np.full(out_sum.shape, np.nan, dtype=float)
    valid = out_count > 0.0
    heatmap[valid] = out_sum[valid] / out_count[valid]
    return x_edges, y_edges, heatmap


def _orientation_sparse_grid_to_heatmap_accumulators(
    profile: OrientationProfile,
    *,
    x_axis: str = "x",
    y_axis: str = "y",
    angle: str = "polar",
    filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered sparse-grid sums/counts projected onto two spatial axes."""
    grid = profile.sparse_grid
    if grid is None:
        raise ValueError(
            "Orientation 2D Heatmap ranges require an orientation sparse grid. Recompute "
            "orientation with a LiNaK version that writes orientation grid data."
        )
    normalized_x_axis = str(x_axis).strip().lower()
    normalized_y_axis = str(y_axis).strip().lower()
    if normalized_x_axis == normalized_y_axis:
        raise ValueError("Orientation 2D Heatmap axes must be different.")
    x_edges = _orientation_grid_edges(grid, normalized_x_axis)
    y_edges = _orientation_grid_edges(grid, normalized_y_axis)
    indices_by_axis = _orientation_grid_axis_indices(grid)
    mask, indices_by_axis = _orientation_grid_filter_mask(
        grid,
        filters=filters,
        indices_by_axis=indices_by_axis,
    )
    value_sum, value_count = _orientation_grid_sum_and_count(grid, angle)
    x_indices = indices_by_axis[normalized_x_axis]
    y_indices = indices_by_axis[normalized_y_axis]
    nx = x_edges.size - 1
    ny = y_edges.size - 1
    out_sum = np.zeros((nx, ny), dtype=float)
    out_count = np.zeros((nx, ny), dtype=float)
    if np.any(mask):
        flat_2d = x_indices[mask] * ny + y_indices[mask]
        out_sum_flat = np.bincount(flat_2d, weights=value_sum[mask], minlength=nx * ny)
        out_count_flat = np.bincount(
            flat_2d,
            weights=value_count[mask],
            minlength=nx * ny,
        )
        out_sum = out_sum_flat.reshape(nx, ny)
        out_count = out_count_flat.reshape(nx, ny)
    return x_edges, y_edges, out_sum, out_count


def _orientation_grid_filters_from_plot_kwargs(
    *,
    orientation_grid_filters: Mapping[str, tuple[float | None, float | None] | None] | None,
    orientation_filter_x_min: float | None,
    orientation_filter_x_max: float | None,
    orientation_filter_y_min: float | None,
    orientation_filter_y_max: float | None,
    orientation_filter_z_min: float | None,
    orientation_filter_z_max: float | None,
    orientation_filter_distance_min: float | None,
    orientation_filter_distance_max: float | None,
) -> dict[str, tuple[float | None, float | None]]:
    filters: dict[str, tuple[float | None, float | None]] = {}
    filters.update(_orientation_filter_bounds(orientation_grid_filters))
    scalar_bounds = {
        "x": (orientation_filter_x_min, orientation_filter_x_max),
        "y": (orientation_filter_y_min, orientation_filter_y_max),
        "z": (orientation_filter_z_min, orientation_filter_z_max),
        "distance": (orientation_filter_distance_min, orientation_filter_distance_max),
    }
    for axis, (lower, upper) in scalar_bounds.items():
        if lower is None and upper is None:
            continue
        filters[axis] = _orientation_filter_bounds({axis: (lower, upper)})[axis]
    return filters


def _normalize_angle_token(angle: str | None) -> str:
    token = "polar" if angle is None else str(angle).strip().lower()
    if token not in _ANGLE_CHOICES:
        raise ValueError(f"angle must be one of: {', '.join(_ANGLE_CHOICES)}")
    return token


def _normalize_component_token(component: str | None) -> str:
    token = "average" if component is None else str(component).strip().lower()
    if token not in _COMPONENT_CHOICES:
        raise ValueError(f"component must be one of: {', '.join(_COMPONENT_CHOICES)}")
    return token


def _distance_label(profile: OrientationProfile) -> str:
    if profile.coordinate_mode == "distance":
        return f"Distance to surface along {profile.axis.upper()} (Angstrom)"
    return f"{profile.axis.upper()} (Angstrom)"


def _orientation_grid_axis_label(axis: str) -> str:
    normalized = str(axis).strip().lower()
    if normalized == "distance":
        return "Distance to surface (Angstrom)"
    if normalized in {"x", "y", "z"}:
        return f"{normalized.upper()} (Angstrom)"
    raise ValueError(f"Unknown orientation grid axis: {axis!r}.")


def _y_label_for_component(component: str, angle: str) -> str:
    angle_label = "\u03b8" if angle == "polar" else "\u03c6"
    if component == "average":
        return f"\u27e8cos({angle_label})\u27e9"
    if component == "density":
        return "H2O number density"
    if component == "density-weighted":
        return f"H2O density-weighted \u27e8cos({angle_label})\u27e9"
    return f"cos({angle_label})"


def _select_1d_data(
    profile: OrientationProfile,
    component: str,
    angle: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = profile.bin_centers
    if component == "average":
        y = profile.cos_polar_mean if angle == "polar" else profile.cos_azimuthal_mean
    elif component == "density":
        y = profile.density
    elif component == "density-weighted":
        y = profile.cos_polar_density if angle == "polar" else profile.cos_azimuthal_density
    else:
        raise ValueError(f"Cannot produce 1D data for component '{component}'.")
    return x, y


def _select_heatmap_data(
    profile: OrientationProfile,
    angle: str,
) -> np.ndarray:
    if angle == "polar":
        return profile.heatmap_polar
    return profile.heatmap_azimuthal


# Public plot functions


def plot_orientation_profile(
    profile: OrientationProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    component: str = "average",
    angle: str = "polar",
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
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    y_bin_width: float | None = None,
    y_bin_reducer: str | None = None,
    orientation_line_x_axis: str | None = None,
    orientation_heatmap_x_axis: str | None = None,
    orientation_heatmap_y_axis: str | None = None,
    orientation_grid_filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
    orientation_filter_x_min: float | None = None,
    orientation_filter_x_max: float | None = None,
    orientation_filter_y_min: float | None = None,
    orientation_filter_y_max: float | None = None,
    orientation_filter_z_min: float | None = None,
    orientation_filter_z_max: float | None = None,
    orientation_filter_distance_min: float | None = None,
    orientation_filter_distance_max: float | None = None,
    heatmap_normalize: bool = False,
    heatmap_normalization_mode: str | None = None,
    heatmap_log_scale: bool = False,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
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
) -> Path | None:
    """Plot a single orientation profile."""
    resolved_mapping = resolve_orientation_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
        component=component,
        angle=angle,
        line_x_axis=orientation_line_x_axis,
        heatmap_x_axis=orientation_heatmap_x_axis,
        heatmap_y_axis=orientation_heatmap_y_axis,
    )
    norm_component = _normalize_component_token(resolved_mapping.component)
    norm_angle = _normalize_angle_token(resolved_mapping.angle)
    resolved_line_x_axis = str(
        orientation_line_x_axis
        or resolved_mapping.renderer_options.get("orientation_line_x_axis")
        or "distance"
    )

    if resolved_mapping.is_heatmap:
        heatmap_axes_are_explicit = (
            orientation_heatmap_x_axis is not None
            or orientation_heatmap_y_axis is not None
            or "orientation_heatmap_x_axis" in resolved_mapping.mapping.fixed_values
            or "orientation_heatmap_y_axis" in resolved_mapping.mapping.fixed_values
        )
        resolved_heatmap_x_axis = str(
            orientation_heatmap_x_axis
            or (
                resolved_mapping.renderer_options.get("orientation_heatmap_x_axis")
                if heatmap_axes_are_explicit
                else ""
            )
            or "x"
        )
        resolved_heatmap_y_axis = str(
            orientation_heatmap_y_axis
            or (
                resolved_mapping.renderer_options.get("orientation_heatmap_y_axis")
                if heatmap_axes_are_explicit
                else ""
            )
            or "y"
        )
        return _plot_orientation_heatmap(
            [profile],
            angle=norm_angle,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_lim=x_lim,
            y_lim=y_lim,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            tick_params_kwargs=tick_params_kwargs,
            grid_kwargs=grid_kwargs,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_cmap=heatmap_cmap,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            y_bin_width=y_bin_width,
            y_bin_reducer=y_bin_reducer,
            orientation_heatmap_x_axis=resolved_heatmap_x_axis if heatmap_axes_are_explicit else None,
            orientation_heatmap_y_axis=resolved_heatmap_y_axis if heatmap_axes_are_explicit else None,
            orientation_grid_filters=orientation_grid_filters,
            orientation_filter_x_min=orientation_filter_x_min,
            orientation_filter_x_max=orientation_filter_x_max,
            orientation_filter_y_min=orientation_filter_y_min,
            orientation_filter_y_max=orientation_filter_y_max,
            orientation_filter_z_min=orientation_filter_z_min,
            orientation_filter_z_max=orientation_filter_z_max,
            orientation_filter_distance_min=orientation_filter_distance_min,
            orientation_filter_distance_max=orientation_filter_distance_max,
            heatmap_normalize=heatmap_normalize,
            heatmap_normalization_mode=heatmap_normalization_mode,
            heatmap_log_scale=heatmap_log_scale,
            heatmap_colorbar_enabled=heatmap_colorbar_enabled,
            heatmap_colorbar_label=heatmap_colorbar_label,
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
            annotations=annotations,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_pad=title_pad,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    grid_filters = _orientation_grid_filters_from_plot_kwargs(
        orientation_grid_filters=orientation_grid_filters,
        orientation_filter_x_min=orientation_filter_x_min,
        orientation_filter_x_max=orientation_filter_x_max,
        orientation_filter_y_min=orientation_filter_y_min,
        orientation_filter_y_max=orientation_filter_y_max,
        orientation_filter_z_min=orientation_filter_z_min,
        orientation_filter_z_max=orientation_filter_z_max,
        orientation_filter_distance_min=orientation_filter_distance_min,
        orientation_filter_distance_max=orientation_filter_distance_max,
    )
    grid_line_state: dict[str, Any] | None = None
    x_label_default = _distance_label(profile)
    stats_key = {
        "average": "cos_polar_mean" if norm_angle == "polar" else "cos_azimuthal_mean",
        "density": "density",
        "density-weighted": (
            "cos_polar_density" if norm_angle == "polar" else "cos_azimuthal_density"
        ),
    }[norm_component]
    series_statistics = (
        None if profile.series_statistics is None else profile.series_statistics.get(stats_key)
    )
    grid_x_axis = resolved_line_x_axis.strip().lower() or "distance"
    use_grid_line = bool(grid_filters) or grid_x_axis != "distance"
    if use_grid_line:
        if norm_component != "average":
            raise ValueError(
                "Orientation 1D Line data ranges currently support Mean orientation only. "
                "Choose Mean orientation or clear orientation data ranges."
            )
        _x_edges, x, y = orientation_sparse_grid_to_line(
            profile,
            x_axis=grid_x_axis,
            angle=norm_angle,
            filters=grid_filters,
        )
        x_label_default = _orientation_grid_axis_label(grid_x_axis)
        series_statistics = None
        grid_line_state = {
            "orientation_grid_line": True,
            "orientation_grid_x_axis": grid_x_axis,
            "orientation_grid_filters": dict(grid_filters),
        }
    else:
        x, y = _select_1d_data(profile, norm_component, norm_angle)
    default_title = f"H2O orientation ({norm_angle})"
    default_y = _y_label_for_component(norm_component, norm_angle)
    result = plot_line_series(
        x,
        y,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, x_label_default),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_id=None,
        line_label=line_label or f"cos({norm_angle})",
        line_color=line_colors[0] if line_colors else None,
        line_width_override=series_line_widths[0] if series_line_widths else None,
        line_marker=series_markers[0] if series_markers else None,
        line_visible=True if not series_enabled else bool(series_enabled[0]),
        show_in_legend=True if not series_show_in_legend else bool(series_show_in_legend[0]),
        fit_config=None if not series_fit_configs else series_fit_configs[0],
        cumulative_config=cumulative_config,
        series_statistics=series_statistics,
        error_config=error_config,
        normalization_mode=series_normalization_modes[0] if series_normalization_modes else None,
        normalization_value=series_normalization_values[0] if series_normalization_values else None,
        normalization_x_ref=series_normalization_x_refs[0] if series_normalization_x_refs else None,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="orientation",
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
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
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
        line_kwargs=line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )
    if capture_state is not None and grid_line_state is not None:
        capture_state.update(grid_line_state)
    return result


def plot_orientation_profiles(
    profiles: list[OrientationProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    component: str = "average",
    angle: str = "polar",
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
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    y_bin_width: float | None = None,
    y_bin_reducer: str | None = None,
    orientation_line_x_axis: str | None = None,
    orientation_heatmap_x_axis: str | None = None,
    orientation_heatmap_y_axis: str | None = None,
    orientation_grid_filters: Mapping[str, tuple[float | None, float | None] | None] | None = None,
    orientation_filter_x_min: float | None = None,
    orientation_filter_x_max: float | None = None,
    orientation_filter_y_min: float | None = None,
    orientation_filter_y_max: float | None = None,
    orientation_filter_z_min: float | None = None,
    orientation_filter_z_max: float | None = None,
    orientation_filter_distance_min: float | None = None,
    orientation_filter_distance_max: float | None = None,
    heatmap_normalize: bool = False,
    heatmap_normalization_mode: str | None = None,
    heatmap_log_scale: bool = False,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
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
) -> Path | None:
    """Plot one or more orientation profiles overlaid."""
    if not profiles:
        raise ValueError("At least one orientation profile is required.")
    first_profile = profiles[0]
    resolved_mapping = resolve_orientation_plot_mapping(
        contract=data_contract,
        profile=first_profile,
        mapping=view_mapping,
        component=component,
        angle=angle,
        line_x_axis=orientation_line_x_axis,
        heatmap_x_axis=orientation_heatmap_x_axis,
        heatmap_y_axis=orientation_heatmap_y_axis,
    )
    norm_component = _normalize_component_token(resolved_mapping.component)
    norm_angle = _normalize_angle_token(resolved_mapping.angle)
    resolved_line_x_axis = str(
        orientation_line_x_axis
        or resolved_mapping.renderer_options.get("orientation_line_x_axis")
        or "distance"
    )

    if resolved_mapping.is_heatmap:
        heatmap_axes_are_explicit = (
            orientation_heatmap_x_axis is not None
            or orientation_heatmap_y_axis is not None
            or "orientation_heatmap_x_axis" in resolved_mapping.mapping.fixed_values
            or "orientation_heatmap_y_axis" in resolved_mapping.mapping.fixed_values
        )
        resolved_heatmap_x_axis = str(
            orientation_heatmap_x_axis
            or (
                resolved_mapping.renderer_options.get("orientation_heatmap_x_axis")
                if heatmap_axes_are_explicit
                else ""
            )
            or "x"
        )
        resolved_heatmap_y_axis = str(
            orientation_heatmap_y_axis
            or (
                resolved_mapping.renderer_options.get("orientation_heatmap_y_axis")
                if heatmap_axes_are_explicit
                else ""
            )
            or "y"
        )
        return _plot_orientation_heatmap(
            profiles,
            angle=norm_angle,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_lim=x_lim,
            y_lim=y_lim,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            tick_params_kwargs=tick_params_kwargs,
            grid_kwargs=grid_kwargs,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_cmap=heatmap_cmap,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            y_bin_width=y_bin_width,
            y_bin_reducer=y_bin_reducer,
            orientation_heatmap_x_axis=resolved_heatmap_x_axis if heatmap_axes_are_explicit else None,
            orientation_heatmap_y_axis=resolved_heatmap_y_axis if heatmap_axes_are_explicit else None,
            orientation_grid_filters=orientation_grid_filters,
            orientation_filter_x_min=orientation_filter_x_min,
            orientation_filter_x_max=orientation_filter_x_max,
            orientation_filter_y_min=orientation_filter_y_min,
            orientation_filter_y_max=orientation_filter_y_max,
            orientation_filter_z_min=orientation_filter_z_min,
            orientation_filter_z_max=orientation_filter_z_max,
            orientation_filter_distance_min=orientation_filter_distance_min,
            orientation_filter_distance_max=orientation_filter_distance_max,
            heatmap_normalize=heatmap_normalize,
            heatmap_normalization_mode=heatmap_normalization_mode,
            heatmap_log_scale=heatmap_log_scale,
            heatmap_colorbar_enabled=heatmap_colorbar_enabled,
            heatmap_colorbar_label=heatmap_colorbar_label,
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
            annotations=annotations,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_pad=title_pad,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    # Build series for overlay
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    labels: list[str] = []
    grid_filters = _orientation_grid_filters_from_plot_kwargs(
        orientation_grid_filters=orientation_grid_filters,
        orientation_filter_x_min=orientation_filter_x_min,
        orientation_filter_x_max=orientation_filter_x_max,
        orientation_filter_y_min=orientation_filter_y_min,
        orientation_filter_y_max=orientation_filter_y_max,
        orientation_filter_z_min=orientation_filter_z_min,
        orientation_filter_z_max=orientation_filter_z_max,
        orientation_filter_distance_min=orientation_filter_distance_min,
        orientation_filter_distance_max=orientation_filter_distance_max,
    )
    grid_x_axis = resolved_line_x_axis.strip().lower() or "distance"
    use_grid_line = bool(grid_filters) or grid_x_axis != "distance"
    stats_key = {
        "average": "cos_polar_mean" if norm_angle == "polar" else "cos_azimuthal_mean",
        "density": "density",
        "density-weighted": (
            "cos_polar_density" if norm_angle == "polar" else "cos_azimuthal_density"
        ),
    }[norm_component]
    series_statistics_data = [
        None if profile.series_statistics is None else profile.series_statistics.get(stats_key)
        for profile in profiles
    ]
    x_label_default = _distance_label(first_profile)
    grid_line_state: dict[str, Any] | None = None
    if use_grid_line:
        if norm_component != "average":
            raise ValueError(
                "Orientation 1D Line data ranges currently support Mean orientation only. "
                "Choose Mean orientation or clear orientation data ranges."
            )
        x_label_default = _orientation_grid_axis_label(grid_x_axis)
        series_statistics_data = [None for _profile in profiles]
        grid_line_state = {
            "orientation_grid_line": True,
            "orientation_grid_x_axis": grid_x_axis,
            "orientation_grid_filters": dict(grid_filters),
        }
    for i, profile in enumerate(profiles):
        if use_grid_line:
            _x_edges, x, y = orientation_sparse_grid_to_line(
                profile,
                x_axis=grid_x_axis,
                angle=norm_angle,
                filters=grid_filters,
            )
        else:
            x, y = _select_1d_data(profile, norm_component, norm_angle)
        x_arrays.append(x)
        y_arrays.append(y)
        labels.append(f"cos({norm_angle}) [{i}]" if len(profiles) > 1 else f"cos({norm_angle})")

    if series_labels is not None:
        labels = list(series_labels) + labels[len(series_labels) :]

    default_title = f"H2O orientation ({norm_angle})"
    ref_profile = profiles[0]
    default_y = _y_label_for_component(norm_component, norm_angle)
    result = plot_multi_line_series(
        x_arrays,
        y_arrays,
        labels,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, x_label_default),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        series_ids=series_ids,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_cumulative_configs=series_cumulative_configs,
        series_error_configs=series_error_configs,
        series_statistics_data=series_statistics_data,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="orientation",
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
        legend=legend,
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
    if capture_state is not None and grid_line_state is not None:
        capture_state.update(grid_line_state)
    return result


# Heatmap rebinning

_REDUCERS: dict[str, Any] = {
    "sum": np.sum,
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
}


def _rebin_heatmap_axis(
    data: np.ndarray,
    edges: np.ndarray,
    bin_width: float,
    *,
    axis: int,
    reducer: str = "sum",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge bins along *axis* so each new bin spans approximately *bin_width*.

    Returns the rebinned data array and new edge array.
    """
    n_bins = data.shape[axis]
    old_widths = np.diff(edges)
    reduce_fn = _REDUCERS.get(reducer, np.sum)

    # Determine grouping: greedily merge consecutive bins until width >= bin_width
    groups: list[list[int]] = []
    current: list[int] = []
    current_width = 0.0
    for i in range(n_bins):
        current.append(i)
        current_width += old_widths[i]
        if current_width >= bin_width - 1e-12:
            groups.append(current)
            current = []
            current_width = 0.0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    new_edges = [edges[groups[0][0]]]
    slices: list[np.ndarray] = []
    for group in groups:
        new_edges.append(edges[group[-1] + 1])
        idx = np.array(group)
        chunk = np.take(data, idx, axis=axis)
        reduced = reduce_fn(chunk, axis=axis, keepdims=True)
        slices.append(reduced)

    rebinned = np.concatenate(slices, axis=axis)
    return rebinned, np.array(new_edges)


def _resolve_heatmap_normalization_mode(
    *,
    heatmap_normalization_mode: str | None,
    heatmap_normalize: bool,
) -> str:
    normalization_mode = "global_probability" if heatmap_normalize else "counts"
    if heatmap_normalization_mode is None:
        return normalization_mode

    candidate_mode = str(heatmap_normalization_mode).strip().lower()
    if candidate_mode not in {"counts", "global_probability", "bulk_water_reference"}:
        raise ValueError(
            "heatmap_normalization_mode must be one of: "
            "counts, global_probability, bulk_water_reference."
        )
    return candidate_mode


def _rebin_heatmap_density_for_display(
    density: np.ndarray,
    edges: np.ndarray,
    *,
    x_bin_width: float | None,
) -> np.ndarray:
    density_array = np.asarray(density, dtype=float)
    if x_bin_width is None or x_bin_width <= 0.0:
        return density_array
    rebinned, _new_edges = _rebin_heatmap_axis(
        density_array[:, np.newaxis],
        edges,
        x_bin_width,
        axis=0,
        reducer="mean",
    )
    return np.asarray(rebinned[:, 0], dtype=float)


def _select_bulk_water_reference_rows(
    *,
    bin_centers: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    density_values = np.asarray(density, dtype=float)
    candidate_mask = (
        np.isfinite(centers)
        & np.isfinite(density_values)
        & (centers > 0.0)
        & (density_values > 0.0)
    )
    if not np.any(candidate_mask):
        raise ValueError(
            "Bulk heatmap normalization requires a distance-aligned profile with a resolvable "
            "water-bulk density plateau."
        )

    rho_max = float(np.max(density_values[candidate_mask]))
    if not np.isfinite(rho_max) or rho_max <= 0.0:
        raise ValueError(
            "Bulk heatmap normalization requires a distance-aligned profile with a resolvable "
            "water-bulk density plateau."
        )

    bulk_mask = candidate_mask & (density_values >= (_HEATMAP_BULK_DENSITY_FRACTION * rho_max))
    candidate_indices = np.flatnonzero(bulk_mask)
    if candidate_indices.size == 0:
        raise ValueError(
            "Bulk heatmap normalization requires a distance-aligned profile with a resolvable "
            "water-bulk density plateau."
        )

    best_segment: np.ndarray | None = None
    segment_start = 0
    for index in range(1, candidate_indices.size + 1):
        end_of_segment = index == candidate_indices.size or (
            candidate_indices[index] != candidate_indices[index - 1] + 1
        )
        if not end_of_segment:
            continue
        segment = candidate_indices[segment_start:index]
        segment_start = index
        if best_segment is None or segment.size > best_segment.size:
            best_segment = segment
            continue
        if best_segment is not None and segment.size == best_segment.size:
            current_mean_distance = float(np.mean(centers[segment]))
            best_mean_distance = float(np.mean(centers[best_segment]))
            if current_mean_distance > best_mean_distance:
                best_segment = segment

    assert best_segment is not None
    return np.asarray(best_segment, dtype=int)


# Heatmap renderer


def _plot_orientation_heatmap(
    profiles: list[OrientationProfile],
    *,
    angle: str,
    output: str | Path | None,
    show: bool,
    show_blocking: bool,
    preferred_backend: str | None,
    style: PlotStyle,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    x_lim: tuple[float | None, float | None] | list[float | None] | None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None,
    title_visible: bool | None,
    ticks_visible: bool | None,
    x_tick_rotation: float | None,
    y_tick_rotation: float | None,
    x_label_font_size: int | None,
    y_label_font_size: int | None,
    tick_params_kwargs: dict[str, Any] | None,
    grid_kwargs: dict[str, Any] | None,
    heatmap_vmin: float | None,
    heatmap_vmax: float | None,
    heatmap_cmap: str | None,
    x_bin_width: float | None,
    x_bin_reducer: str | None,
    y_bin_width: float | None,
    y_bin_reducer: str | None,
    orientation_heatmap_x_axis: str | None,
    orientation_heatmap_y_axis: str | None,
    orientation_grid_filters: Mapping[str, tuple[float | None, float | None] | None] | None,
    orientation_filter_x_min: float | None,
    orientation_filter_x_max: float | None,
    orientation_filter_y_min: float | None,
    orientation_filter_y_max: float | None,
    orientation_filter_z_min: float | None,
    orientation_filter_z_max: float | None,
    orientation_filter_distance_min: float | None,
    orientation_filter_distance_max: float | None,
    heatmap_normalize: bool,
    heatmap_normalization_mode: str | None,
    heatmap_log_scale: bool,
    heatmap_colorbar_label: str | None,
    heatmap_colorbar_label_size: int | None,
    heatmap_colorbar_tick_size: int | None,
    heatmap_colorbar_enabled: bool,
    heatmap_colorbar_position: str,
    heatmap_colorbar_pad: float | None,
    heatmap_colorbar_shrink: float | None,
    heatmap_colorbar_aspect: float | None,
    annotations: list[dict[str, Any]] | None,
    x_ticks: list[float] | tuple[float, ...] | None,
    y_ticks: list[float] | tuple[float, ...] | None,
    x_label_pad: float | None,
    y_label_pad: float | None,
    title_pad: float | None,
    capture_state: dict[str, Any] | None,
    suppress_output_log: bool,
    matplotlib_rc: dict[str, Any] | None,
    figure_kwargs: dict[str, Any] | None,
    axes_kwargs: dict[str, Any] | None,
    tight_layout_kwargs: dict[str, Any] | None,
    savefig_kwargs: dict[str, Any] | None,
) -> Path | None:
    """Render a 2-D heatmap of orientation frequency vs distance."""
    normalization_mode = _resolve_heatmap_normalization_mode(
        heatmap_normalization_mode=heatmap_normalization_mode,
        heatmap_normalize=heatmap_normalize,
    )
    grid_filters = _orientation_grid_filters_from_plot_kwargs(
        orientation_grid_filters=orientation_grid_filters,
        orientation_filter_x_min=orientation_filter_x_min,
        orientation_filter_x_max=orientation_filter_x_max,
        orientation_filter_y_min=orientation_filter_y_min,
        orientation_filter_y_max=orientation_filter_y_max,
        orientation_filter_z_min=orientation_filter_z_min,
        orientation_filter_z_max=orientation_filter_z_max,
        orientation_filter_distance_min=orientation_filter_distance_min,
        orientation_filter_distance_max=orientation_filter_distance_max,
    )
    use_sparse_grid = (
        orientation_heatmap_x_axis is not None
        or orientation_heatmap_y_axis is not None
        or bool(grid_filters)
    )

    if use_sparse_grid:
        if normalization_mode != "counts":
            raise ValueError(
                "Orientation sparse-grid heatmaps show mean orientation values and support "
                "only counts normalization."
            )
        x_axis = (orientation_heatmap_x_axis or "x").strip().lower()
        y_axis = (orientation_heatmap_y_axis or "y").strip().lower()
        if x_axis == y_axis:
            raise ValueError("Orientation 2D Heatmap axes must be different.")

        ref = profiles[0]
        x_edges, y_edges, heatmap_sum, heatmap_count = (
            _orientation_sparse_grid_to_heatmap_accumulators(
                ref,
                x_axis=x_axis,
                y_axis=y_axis,
                angle=angle,
                filters=grid_filters,
            )
        )
        for profile in profiles[1:]:
            extra_x_edges, extra_y_edges, extra_sum, extra_count = (
                _orientation_sparse_grid_to_heatmap_accumulators(
                    profile,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    angle=angle,
                    filters=grid_filters,
                )
            )
            if (
                extra_sum.shape != heatmap_sum.shape
                or extra_x_edges.shape != x_edges.shape
                or extra_y_edges.shape != y_edges.shape
                or not np.allclose(extra_x_edges, x_edges)
                or not np.allclose(extra_y_edges, y_edges)
            ):
                raise ValueError(
                    "Orientation sparse-grid heatmaps require compatible grid edges across "
                    "all enabled profiles."
            )
            heatmap_sum += extra_sum
            heatmap_count += extra_count

        original_x_edges = np.asarray(x_edges, dtype=float)
        original_y_edges = np.asarray(y_edges, dtype=float)
        if x_bin_width is not None and x_bin_width > 0:
            heatmap_sum, x_edges = _rebin_heatmap_axis(
                heatmap_sum,
                original_x_edges,
                x_bin_width,
                axis=0,
                reducer="sum",
            )
            heatmap_count, _ = _rebin_heatmap_axis(
                heatmap_count,
                original_x_edges,
                x_bin_width,
                axis=0,
                reducer="sum",
            )
        if y_bin_width is not None and y_bin_width > 0:
            heatmap_sum, y_edges = _rebin_heatmap_axis(
                heatmap_sum,
                original_y_edges,
                y_bin_width,
                axis=1,
                reducer="sum",
            )
            heatmap_count, _ = _rebin_heatmap_axis(
                heatmap_count,
                original_y_edges,
                y_bin_width,
                axis=1,
                reducer="sum",
            )

        heatmap_plot = np.full(heatmap_sum.shape, np.nan, dtype=float)
        valid = heatmap_count > 0.0
        heatmap_plot[valid] = heatmap_sum[valid] / heatmap_count[valid]

        angle_symbol = "\u03b8" if angle == "polar" else "\u03c6"
        default_title = f"H2O orientation 2D Heatmap ({angle})"
        resolved_colorbar_label = (
            heatmap_colorbar_label
            if heatmap_colorbar_label is not None
            else f"Mean cos({angle_symbol})"
        )
        return plot_heatmap_series(
            x_edges,
            y_edges,
            heatmap_plot,
            title=title or default_title,
            x_label=resolve_explicit_plot_text(
                x_label,
                _orientation_grid_axis_label(x_axis),
            ),
            y_label=resolve_explicit_plot_text(
                y_label,
                _orientation_grid_axis_label(y_axis),
            ),
            output=output,
            show=show,
            show_blocking=show_blocking,
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
            heatmap_colorbar_label=resolved_colorbar_label,
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
            annotations=annotations,
            capture_state_extra={
                "orientation_grid_heatmap": True,
                "orientation_grid_x_axis": x_axis,
                "orientation_grid_y_axis": y_axis,
                "orientation_grid_filters": dict(grid_filters),
                "heatmap_normalization_mode": normalization_mode,
                "heatmap_log_scale": bool(heatmap_log_scale),
                "heatmap_colorbar_enabled": bool(heatmap_colorbar_enabled),
                "heatmap_colorbar_label": resolved_colorbar_label,
            },
        )

    # Sum heatmaps if multiple profiles
    ref = profiles[0]
    heatmap = _select_heatmap_data(ref, angle).copy()
    density_accumulator = np.asarray(ref.density, dtype=float).copy()
    density_contributors = 1
    for p in profiles[1:]:
        extra = _select_heatmap_data(p, angle)
        if extra.shape == heatmap.shape:
            heatmap += extra
            if (
                np.asarray(p.density).shape == density_accumulator.shape
                and np.asarray(p.bin_edges).shape == np.asarray(ref.bin_edges).shape
                and np.allclose(
                    np.asarray(p.bin_edges, dtype=float), np.asarray(ref.bin_edges, dtype=float)
                )
            ):
                density_accumulator += np.asarray(p.density, dtype=float)
                density_contributors += 1

    x_edges = ref.bin_edges
    y_edges = ref.heatmap_angle_bin_edges
    x_centers = np.asarray(ref.bin_centers, dtype=float)
    density_reference = density_accumulator / float(density_contributors)

    # Rebin distance axis (x)
    if x_bin_width is not None and x_bin_width > 0:
        heatmap, x_edges = _rebin_heatmap_axis(
            heatmap,
            x_edges,
            x_bin_width,
            axis=0,
            reducer=x_bin_reducer or "sum",
        )
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        density_reference = _rebin_heatmap_density_for_display(
            density_reference,
            np.asarray(ref.bin_edges, dtype=float),
            x_bin_width=x_bin_width,
        )
    # Rebin angle axis (y)
    if y_bin_width is not None and y_bin_width > 0:
        heatmap, y_edges = _rebin_heatmap_axis(
            heatmap,
            y_edges,
            y_bin_width,
            axis=1,
            reducer=y_bin_reducer or "sum",
        )

    if normalization_mode == "global_probability":
        total = float(np.sum(heatmap))
        heatmap_plot = heatmap if total <= 0.0 else heatmap / total
    elif normalization_mode == "bulk_water_reference":
        if ref.coordinate_mode != "distance":
            raise ValueError(
                "Bulk heatmap normalization requires a distance-aligned profile with a "
                "resolvable water-bulk density plateau."
            )
        bulk_rows = _select_bulk_water_reference_rows(
            bin_centers=x_centers,
            density=density_reference,
        )
        bulk_mean = float(np.mean(heatmap[bulk_rows, :], dtype=float))
        if not np.isfinite(bulk_mean) or bulk_mean <= 0.0:
            raise ValueError(
                "Bulk heatmap normalization requires a distance-aligned profile with a "
                "resolvable water-bulk density plateau."
            )
        heatmap_plot = heatmap / bulk_mean
    else:
        heatmap_plot = heatmap

    angle_symbol = "\u03b8" if angle == "polar" else "\u03c6"
    default_title = f"H2O orientation heatmap ({angle})"
    default_x = _distance_label(ref)
    default_y = f"cos({angle_symbol})"
    if normalization_mode == "counts":
        default_cb_label = "Frequency"
    elif normalization_mode == "global_probability":
        default_cb_label = "Global probability"
    else:
        default_cb_label = "Bulk-normalized frequency (bulk mean = 1)"
    resolved_colorbar_label = (
        heatmap_colorbar_label if heatmap_colorbar_label is not None else default_cb_label
    )
    return plot_heatmap_series(
        x_edges,
        y_edges,
        heatmap_plot,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, default_x),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
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
        heatmap_colorbar_label=resolved_colorbar_label,
        heatmap_colorbar_label_size=heatmap_colorbar_label_size,
        heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
        heatmap_colorbar_position=heatmap_colorbar_position,
        heatmap_colorbar_pad=heatmap_colorbar_pad,
        heatmap_colorbar_shrink=heatmap_colorbar_shrink,
        heatmap_colorbar_aspect=heatmap_colorbar_aspect,
        annotations=annotations,
        capture_state_extra={
            "heatmap_normalization_mode": normalization_mode,
            "heatmap_log_scale": bool(heatmap_log_scale),
            "heatmap_colorbar_enabled": bool(heatmap_colorbar_enabled),
            "heatmap_colorbar_label": resolved_colorbar_label,
        },
    )
