"""Geometry-backed spatial orientation maps, sharing density normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
import logging
from time import perf_counter

import numpy as np

from ..plot.gui_session import report_progress
from ..cancellation import check_cancelled
from ..cache import PREVIEW_CACHE

LOGGER = logging.getLogger("linak")

SPATIAL_DENSITY_VERSION = 1
ALIGNMENT_VERSION = 1
ALIGNMENT_RECOMPUTE = "Recompute this orientation file to enable interface alignment."
BOTTOM_MOMENT_CHANNELS = (
    "bottom_cos_polar_sum",
    "bottom_count_polar_valid",
    "bottom_cos_azimuthal_sum",
    "bottom_count_azimuthal_valid",
)
ANGLE_CONVENTION = (
    "Spatial registration only; stored polar and azimuthal angle definitions are preserved."
)
SPATIAL_QUANTITIES = ("average", "density", "density-weighted")
SPATIAL_RECOMPUTE = "Recompute orientation to enable density and density-weighted spatial maps."


def density_capability(profile: Any) -> tuple[bool, str]:
    grid = profile.sparse_grid
    if grid is None or grid.density_geometry_version != SPATIAL_DENSITY_VERSION:
        return False, SPATIAL_RECOMPUTE
    if grid.density_geometry_reason:
        return False, grid.density_geometry_reason
    if not grid.density_geometry_groups:
        return False, SPATIAL_RECOMPUTE
    return True, ""


def density_capability_from_header(header: Mapping[str, Any]) -> tuple[bool, str]:
    """Check capabilities without loading sparse arrays or normalization geometry."""
    if header.get("orientation_spatial_density_version") != SPATIAL_DENSITY_VERSION:
        return False, SPATIAL_RECOMPUTE
    reason = str(header.get("orientation_spatial_density_reason") or "")
    if reason:
        return False, reason
    inventory = header.get("dataset_inventory", {})
    required = {
        "orientation_density_geometry_cells_A",
        "orientation_density_geometry_entity_sum",
        "orientation_density_geometry_frame_counts",
    }
    return (True, "") if required.issubset(inventory) else (False, SPATIAL_RECOMPUTE)


def build_density_accumulator(frames, *, axis, coordinate_mode, estimate, reference, edges):
    """The frames here are geometry-only copies captured in the analysis pass."""
    from .density import _DensitySparseGridAccumulator, _prepare_density_grid_geometry

    if not frames or any(
        not np.all(f.pbc) or not np.all(np.isfinite(f.cell)) or abs(np.linalg.det(f.cell)) <= 1e-12
        for f in frames
    ):
        return None, "Spatial density requires a usable fully periodic cell in every frame."
    if reference is not None and reference.distance_mode != "plane":
        return (
            None,
            "Spatial physical density is unavailable for nearest-atom distance; use surface-plane orientation.",
        )
    geometry_estimate = estimate
    if estimate is not None and not hasattr(estimate, "frame_values"):
        from types import SimpleNamespace

        geometry_estimate = SimpleNamespace(frame_values=np.asarray(estimate.per_frame))
    table = _prepare_density_grid_geometry(frames, coordinate_mode, geometry_estimate, reference)
    accumulator = _DensitySparseGridAccumulator(
        species_label="H2O",
        count_label="molecules",
        x_bin_edges=edges[0],
        y_bin_edges=edges[1],
        z_bin_edges=edges[2],
        distance_bin_edges=edges[3],
        grid_bin_width=float(edges[0][1] - edges[0][0]),
        max_nonzero_bins=np.iinfo(np.int64).max,
        surface_axis=axis,
        distance_coordinate_mode=coordinate_mode,
        surface_estimate=estimate,
        surface_reference=reference,
        geometry_table=table,
    )
    return accumulator, ""


def density_geometry_datasets(grid) -> dict[str, np.ndarray]:
    if grid is None or grid.density_geometry_groups is None:
        return {}
    from .density import _geometry_group_datasets

    result = {
        "orientation_density_" + name: value
        for name, value in _geometry_group_datasets(grid.density_geometry_groups).items()
    }
    for name in BOTTOM_MOMENT_CHANNELS:
        if all(name in group for group in grid.density_geometry_groups):
            result[f"orientation_density_geometry_{name}"] = np.concatenate(
                [group[name] for group in grid.density_geometry_groups]
            )
    return result


def load_density_geometry(datasets: Mapping[str, Any], metadata: Mapping[str, Any]):
    version = metadata.get("orientation_spatial_density_version")
    if version != SPATIAL_DENSITY_VERSION:
        return None
    from .density import _load_geometry_groups

    prefix = "orientation_density_"
    geometry = {
        name[len(prefix) :]: value for name, value in datasets.items() if name.startswith(prefix)
    }
    groups = _load_geometry_groups(geometry)
    names = [f"geometry_{name}" for name in BOTTOM_MOMENT_CHANNELS]
    present = [name for name in names if name in geometry]
    if present and metadata.get("orientation_alignment_version") != ALIGNMENT_VERSION:
        return groups
    if present:
        if len(present) != len(names) or groups is None:
            raise ValueError("Incomplete orientation alignment moments. " + ALIGNMENT_RECOMPUTE)
        offsets = geometry["geometry_offsets"]
        for name in names:
            values = np.asarray(geometry[name], dtype=float)
            if values.shape != np.asarray(geometry["geometry_flat_indices"]).shape:
                raise ValueError(
                    "Invalid orientation alignment moment shape. " + ALIGNMENT_RECOMPUTE
                )
            for index, group in enumerate(groups):
                group[name.removeprefix("geometry_")] = values[offsets[index] : offsets[index + 1]]
    return groups


def _density_adapter(profile):
    from .density import DensityGridProfile

    grid = profile.sparse_grid
    reference = profile.surface_reference
    paired = bool(
        grid.density_geometry_groups and "top_entity_sum" in grid.density_geometry_groups[0]
    )
    if paired and reference is not None:
        reference = replace(reference, resolved_layout="symmetric")
    return DensityGridProfile(
        species="H2O",
        x_bin_edges=grid.x_edges,
        y_bin_edges=grid.y_edges,
        z_bin_edges=grid.z_edges,
        distance_bin_edges=grid.distance_edges,
        flat_indices=grid.flat_indices,
        mass_sum_g=np.zeros_like(grid.entity_sum),
        entity_sum=grid.entity_sum,
        n_frames=profile.n_frames,
        surface_axis=profile.axis,
        distance_coordinate_mode=profile.coordinate_mode,
        surface_estimate=profile.surface_estimate,
        surface_reference=reference,
        geometry_groups=grid.density_geometry_groups,
        interface_reference=profile.interface_reference,
        interface_alignment_metadata=profile.interface_alignment_metadata,
        top_mass_sum_g=None if grid.top_entity_sum is None else np.zeros_like(grid.top_entity_sum),
        bottom_mass_sum_g=None
        if grid.bottom_entity_sum is None
        else np.zeros_like(grid.bottom_entity_sum),
        top_entity_sum=grid.top_entity_sum,
        bottom_entity_sum=grid.bottom_entity_sum,
    )


@dataclass(frozen=True)
class SpatialOrientationMap:
    x_edges: np.ndarray
    y_edges: np.ndarray
    values: np.ndarray
    quantity: str
    units: str
    colorbar_label: str
    normalization: str
    alignments: dict[str, Any] = field(default_factory=dict)


def alignment_capability_from_header(header: Mapping[str, Any]) -> tuple[bool, str]:
    record = header.get("interface_reference")
    if not isinstance(record, Mapping):
        return False, ALIGNMENT_RECOMPUTE
    if record.get("unavailable_reason"):
        return False, str(record["unavailable_reason"])
    reason = header.get("orientation_spatial_density_reason")
    if reason:
        return False, str(reason)
    if header.get("orientation_alignment_version") != ALIGNMENT_VERSION:
        return False, ALIGNMENT_RECOMPUTE
    required = {f"orientation_density_geometry_{name}" for name in BOTTOM_MOMENT_CHANNELS}
    if not required.issubset(header.get("dataset_inventory", {})):
        return False, ALIGNMENT_RECOMPUTE
    return True, ""


def alignment_records(profiles) -> dict[str, Any]:
    return {
        p.interface_reference.reference_id: p.interface_alignment_metadata
        for p in profiles
        if p.interface_reference is not None and p.interface_alignment_metadata is not None
    }


def aligned_orientation_profile(profile, axes, filters=None, interface_alignments=None):
    """Register scalar sample locations before selection, preserving angle conventions."""
    from .interface_alignment import alignment_for_reference, validate_alignment
    from .interface_density import align_density_grid
    from .orientation import _align_sparse_values

    if not interface_alignments or profile.spatial_interface_view in {"full", "top"}:
        return profile
    reference = profile.interface_reference
    if reference is None or reference.reference_id not in interface_alignments:
        return profile
    lateral = {"xyz"[index] for index in reference.lateral_indices}
    if not lateral.intersection((*axes, *(filters or {}))):
        return profile
    if profile.axis in axes:
        raise ValueError(
            "Choose Distance for the interface-normal axis when using interface alignment."
        )
    grid = profile.sparse_grid
    groups = None if grid is None else grid.density_geometry_groups
    if reference.unavailable_reason:
        raise ValueError(reference.unavailable_reason)
    if grid is not None and grid.density_geometry_reason:
        raise ValueError(grid.density_geometry_reason)
    if not groups or any(name not in group for group in groups for name in BOTTOM_MOMENT_CHANNELS):
        raise ValueError(ALIGNMENT_RECOMPUTE)
    if profile.interface_alignment_metadata is not None:
        raise ValueError("Apply interface alignment to the original orientation grid only.")
    alignment = alignment_for_reference(reference, interface_alignments)
    if alignment is None:
        return profile
    if profile.axis != reference.axis:
        raise ValueError("The orientation axis does not match its alignment reference.")
    # Only GUI-owned, read-only source data may reuse an aligned numerical result.
    cache_key = None
    if profile._gui_cache_key is not None:
        arrays = (
            grid.flat_indices,
            grid.entity_sum,
            grid.cos_polar_sum,
            grid.count_polar_valid,
            grid.cos_azimuthal_sum,
            grid.count_azimuthal_valid,
            grid.top_entity_sum,
            grid.bottom_entity_sum,
            grid.top_cos_polar_sum,
            grid.top_count_polar_valid,
            grid.top_cos_azimuthal_sum,
            grid.top_count_azimuthal_valid,
            grid.x_edges,
            grid.y_edges,
            grid.z_edges,
            grid.distance_edges,
        )
        cache_key = (
            profile._gui_cache_key,
            tuple(id(value) for value in arrays),
            id(groups),
            profile.spatial_interface_view,
            profile.coordinate_mode,
            profile.axis,
            alignment.matrix,
            tuple(float(x) for x in np.asarray(alignment.translation) % 1),
        )
        cached = PREVIEW_CACHE.get("orientation.aligned", cache_key)
        if cached is not None:
            check_cancelled()
            LOGGER.debug(
                "Orientation aligned accumulators cache hit: rows=%d", len(cached[1].flat_indices)
            )
            return replace(
                profile,
                sparse_grid=cached[1],
                interface_alignment_metadata={**cached[2], **alignment.to_dict()},
            )
    started = perf_counter()
    validate_alignment(
        replace(
            reference, cells=np.unique(np.asarray([group["cell"] for group in groups]), axis=0)
        ),
        alignment,
    )
    bottom_totals: dict[str, np.ndarray] = {}
    moved = align_density_grid(
        _density_adapter(profile),
        alignment,
        bottom_channels=BOTTOM_MOMENT_CHANNELS,
        _capture_bottom_totals=bottom_totals,
    )
    new_edges = [getattr(moved, f"{axis}_bin_edges") for axis in ("x", "y", "z", "distance")]
    new_shape = tuple(len(edge) - 1 for edge in new_edges)
    old_edges = [
        getattr(grid, name) for name in ("x_edges", "y_edges", "z_edges", "distance_edges")
    ]
    indices = np.asarray(np.unravel_index(grid.flat_indices, grid.shape)).T
    for axis in range(4):
        indices[:, axis] += int(
            round(
                (old_edges[axis][0] - new_edges[axis][0])
                / (new_edges[axis][1] - new_edges[axis][0])
            )
        )
    old_flat = np.ravel_multi_index(indices.T, new_shape)
    new_flat = np.union1d(old_flat, np.asarray(moved.flat_indices, dtype=np.int64))
    fields = {}
    for angle in ("polar", "azimuthal"):
        for suffix in (f"cos_{angle}_sum", f"count_{angle}_valid"):
            top = getattr(grid, f"top_{suffix}")
            if top is None:
                raise ValueError(ALIGNMENT_RECOMPUTE)
            fields[f"top_{suffix}"] = _align_sparse_values(new_flat, old_flat, top)
            fields[f"bottom_{suffix}"] = _align_sparse_values(
                new_flat, moved.flat_indices, bottom_totals[f"bottom_{suffix}"]
            )
            fields[suffix] = (
                fields[f"bottom_{suffix}"]
                if profile.spatial_interface_view == "bottom"
                else fields[f"top_{suffix}"] + fields[f"bottom_{suffix}"]
            )
    for side in ("top", "bottom"):
        fields[f"{side}_entity_sum"] = _align_sparse_values(
            new_flat, moved.flat_indices, getattr(moved, f"{side}_entity_sum")
        )
    fields["entity_sum"] = (
        fields["bottom_entity_sum"]
        if profile.spatial_interface_view == "bottom"
        else fields["top_entity_sum"] + fields["bottom_entity_sum"]
    )
    metadata = {**moved.interface_alignment_metadata, "angle_convention": ANGLE_CONVENTION}
    result = replace(
        profile,
        interface_alignment_metadata=metadata,
        sparse_grid=replace(
            grid,
            **dict(zip(("x_edges", "y_edges", "z_edges", "distance_edges"), new_edges)),
            shape=new_shape,
            flat_indices=new_flat,
            density_geometry_groups=moved.geometry_groups,
            lab_cos_polar_sum=None,
            lab_count_polar_valid=None,
            **fields,
        ),
    )
    check_cancelled()
    if cache_key is not None:
        from .orientation_cache import freeze_arrays

        freeze_arrays(result.sparse_grid)
        check_cancelled()
        PREVIEW_CACHE.put("orientation.aligned", cache_key, (grid, result.sparse_grid, metadata))
    LOGGER.debug(
        "Orientation aligned accumulators built: rows=%d seconds=%.6f",
        len(new_flat),
        perf_counter() - started,
    )
    return result


def enabled_source_mask(
    count,
    *,
    series_enabled=None,
    series_ids=None,
    render_series_descriptors=None,
    series_overrides_by_id=None,
):
    """A source contributes once when any of its render layers is enabled."""
    enabled = [
        series_enabled is None or i >= len(series_enabled) or bool(series_enabled[i])
        for i in range(count)
    ]
    if not series_ids or len(series_ids) != count:
        return enabled
    overrides = series_overrides_by_id or {}
    if not render_series_descriptors:
        return [
            bool(overrides.get(identifier, {}).get("enabled", active))
            for identifier, active in zip(series_ids, enabled)
        ]
    default_by_id = dict(zip(series_ids, enabled))
    active_sources = set()
    for descriptor in render_series_descriptors:
        layer_id = str(descriptor.get("series_id") or "")
        source_id = str(descriptor.get("source_series_id") or layer_id)
        default = default_by_id.get(layer_id, True)
        if overrides.get(layer_id, {}).get("enabled", default):
            active_sources.add(source_id)
    return [identifier in active_sources for identifier in series_ids]


def _common_projection_edges(source_edges: Sequence[np.ndarray]) -> np.ndarray:
    """Coarsen only at boundaries shared by all overlapping source grids.

    A source bin is never split. Disjoint extents may contribute extra boundaries
    where there is no other source bin to split.
    """
    candidates = np.unique(np.concatenate(source_edges))
    tolerance = max(1e-10, float(np.max(np.abs(candidates))) * 1e-12)
    keep = np.ones(len(candidates), dtype=bool)
    for edges in source_edges:
        indices = np.searchsorted(edges, candidates)
        right = edges[np.clip(indices, 0, len(edges) - 1)]
        left = edges[np.clip(indices - 1, 0, len(edges) - 1)]
        keep &= (
            (candidates <= edges[0] + tolerance)
            | (candidates >= edges[-1] - tolerance)
            | (np.minimum(abs(candidates - right), abs(candidates - left)) <= tolerance)
        )
    common = candidates[keep]
    return common[np.r_[True, np.diff(common) > tolerance]]


def spatial_orientation_map(
    profiles: Sequence[Any],
    *,
    x_axis="x",
    y_axis="y",
    angle="polar",
    quantity="average",
    filters=None,
    x_bin_width=None,
    y_bin_width=None,
    interface_alignments=None,
):
    """Aggregate raw moments and normalize geometry before deriving the quantity."""
    from .orientation import (
        _orientation_grid_edges,
        _orientation_grid_axis_indices,
        _orientation_grid_filter_mask,
        _rebin_heatmap_axis,
    )
    from .density import _geometry_project_density

    started = perf_counter()
    if quantity not in SPATIAL_QUANTITIES:
        raise ValueError(
            "Spatial orientation quantity must be average, density, or density-weighted."
        )
    axes = (str(x_axis).strip().lower(), str(y_axis).strip().lower())
    if len(set(axes)) != 2 or any(axis not in {"x", "y", "z", "distance"} for axis in axes):
        raise ValueError(
            "Spatial orientation axes must be two different X/Y/Z/Distance coordinates."
        )
    if not profiles or any(p.sparse_grid is None for p in profiles):
        raise ValueError("Spatial maps require an orientation sparse grid. Recompute orientation.")
    profiles = [
        aligned_orientation_profile(p, axes, filters, interface_alignments) for p in profiles
    ]
    registered_at = perf_counter()
    report_progress("Projecting orientation")
    density_seconds = 0.0
    original = tuple(
        _common_projection_edges(
            [_orientation_grid_edges(profile.sparse_grid, axis) for profile in profiles]
        )
        for axis in axes
    )
    edges = list(original)
    for dimension, width in enumerate((x_bin_width, y_bin_width)):
        if width is not None and width > 0:
            _, edges[dimension] = _rebin_heatmap_axis(
                np.zeros(len(original[dimension]) - 1),
                original[dimension],
                width,
                axis=0,
                reducer="sum",
            )
    shape = tuple(len(edge) - 1 for edge in edges)
    size = int(np.prod(shape))
    # Channels keep real molecule sums/counts until after rebinning and source pooling.
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    density_sum = np.zeros(shape)
    density_coverage = np.zeros(shape)
    folded_flags = set()
    for profile in profiles:
        check_cancelled()
        grid = profile.sparse_grid
        view = profile.spatial_interface_view
        folded = bool(
            view == "folded"
            and profile.coordinate_mode == "distance"
            and profile.surface_reference is not None
            and profile.surface_reference.is_symmetric
        )
        folded_flags.add(folded)
        if "distance" in axes and profile.axis in axes:
            raise ValueError(
                "Distance and the raw surface-axis coordinate cannot form an independent spatial map."
            )
        if view == "full" and "distance" in axes and quantity != "average":
            raise ValueError(
                "Full-cell spatial maps use Cartesian coordinates; select an interface to use Distance."
            )
        indices_by_axis = _orientation_grid_axis_indices(grid)
        mask, _ = _orientation_grid_filter_mask(
            grid, filters=filters, indices_by_axis=indices_by_axis
        )
        target = []
        for dimension, axis in enumerate(axes):
            centers = 0.5 * (
                _orientation_grid_edges(grid, axis)[:-1] + _orientation_grid_edges(grid, axis)[1:]
            )
            target.append(
                np.clip(
                    np.searchsorted(edges[dimension], centers[indices_by_axis[axis]], side="right")
                    - 1,
                    0,
                    shape[dimension] - 1,
                )
            )
        flat = target[0] * shape[1] + target[1]
        for side in (("top", "bottom") if folded else ("single",)) if quantity != "density" else ():
            prefix = "" if side == "single" else side + "_"
            value_sum = getattr(grid, f"{prefix}cos_{angle}_sum")
            value_count = getattr(grid, f"{prefix}count_{angle}_valid")
            if value_sum is None or value_count is None:
                raise ValueError(
                    "Symmetric spatial maps require side-resolved orientation moments. Recompute orientation."
                )
            sums.setdefault(side, np.zeros(size))
            counts.setdefault(side, np.zeros(size))
            sums[side] += np.bincount(
                flat[mask], weights=np.asarray(value_sum)[mask], minlength=size
            )
            counts[side] += np.bincount(
                flat[mask], weights=np.asarray(value_count)[mask], minlength=size
            )
        if quantity != "average":
            available, reason = density_capability(profile)
            if not available:
                raise ValueError(reason)
            if folded and profile.axis in axes:
                raise ValueError(
                    "Use Distance for a folded interface cross-section, or select Full cell for Cartesian planes."
                )
            density_filters = dict(filters or {})
            # Interface XY columns also need their per-interface normal volume.
            if view != "full" and profile.coordinate_mode == "distance":
                density_filters.setdefault("distance", (None, None))
            adapter = _density_adapter(profile)
            normalization_started = perf_counter()
            _, density, coverage = _geometry_project_density(
                adapter,
                axes,
                tuple(edges),
                density_filters,
                interface_view=view if view in {"full", "top", "bottom"} else "folded",
            )
            density_seconds += perf_counter() - normalization_started
            density_sum += (
                np.nan_to_num(density) * coverage / 1000.0
            )  # molecules/nm³ -> molecules/Å³
            density_coverage += coverage
    if len(folded_flags) > 1:
        raise ValueError(
            "Cannot combine folded symmetric and single/full-cell spatial orientation representations."
        )
    mean = np.zeros(size)
    available_sides = np.zeros(size)
    for side in sums:
        valid = counts[side] > 0
        mean[valid] += sums[side][valid] / counts[side][valid]
        available_sides[valid] += 1
    mean = np.divide(
        mean, available_sides, out=np.full(size, np.nan), where=available_sides > 0
    ).reshape(shape)
    density = np.divide(
        density_sum, density_coverage, out=np.full(shape, np.nan), where=density_coverage > 0
    )
    symbol = "θ" if angle == "polar" else "φ"
    values = mean if quantity == "average" else density if quantity == "density" else mean * density
    label = (
        f"Mean cos({symbol})"
        if quantity == "average"
        else "Water density (molecules/Å³)"
        if quantity == "density"
        else f"Density-weighted ⟨cos({symbol})⟩ (molecules/Å³)"
    )
    LOGGER.debug(
        "Orientation spatial map: quantity=%s axes=%s registration_s=%.6f projection_s=%.6f density_s=%.6f",
        quantity,
        axes,
        registered_at - started,
        perf_counter() - registered_at - density_seconds,
        density_seconds,
    )
    return SpatialOrientationMap(
        edges[0],
        edges[1],
        values,
        quantity,
        "1" if quantity == "average" else "molecules/Angstrom^3",
        label,
        "valid-molecule mean; equal interface means when folded"
        if quantity == "average"
        else "per-geometry physical volume and frame coverage; equal interface mean when folded",
        alignment_records(profiles),
    )
