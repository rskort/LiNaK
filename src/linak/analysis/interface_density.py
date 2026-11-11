"""Conservative registration of sparse density accumulators on a periodic slab.

The lattice isometry preserves the periodic cell and its physical coverage.
Only negative-side mass/entity accumulators move. Each clipped source footprint
is distributed by exact overlap area, before any existing density projection.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import logging
from time import perf_counter
from typing import Any

import numpy as np

from ..plot.gui_session import report_progress
from ..cancellation import check_cancelled
from ..cell_geometry import cartesian_bounds
from .interface_alignment import InterfaceAlignment, validate_alignment
from .interface_remap import expansion_budget, interval_compatible, transfer_operator
from .density_sparse import SparseReducer, SparseWorkspace, row_dtype

LOGGER = logging.getLogger("linak")


def _clip(polygon: np.ndarray, axis: int, bound: float, greater: bool) -> np.ndarray:
    if not len(polygon):
        return polygon
    result = []
    previous = polygon[-1]
    previous_in = previous[axis] >= bound if greater else previous[axis] <= bound
    for point in polygon:
        inside = point[axis] >= bound if greater else point[axis] <= bound
        if inside != previous_in:
            fraction = (bound - previous[axis]) / (point[axis] - previous[axis])
            result.append(previous + fraction * (point - previous))
        if inside:
            result.append(point)
        previous, previous_in = point, inside
    return np.asarray(result, dtype=float).reshape((-1, 2))


def _rectangle(polygon: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    for axis in range(2):
        polygon = _clip(polygon, axis, float(lower[axis]), True)
        polygon = _clip(polygon, axis, float(upper[axis]), False)
    return polygon


def _area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    # Translation reduces cancellation when the simulation has a large origin.
    points = polygon - polygon[0]
    return float(
        abs(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
        * 0.5
    )


def footprint_weights(
    source_indices: tuple[int, int],
    source_edges: tuple[np.ndarray, np.ndarray],
    target_edges: tuple[np.ndarray, np.ndarray],
    basis: np.ndarray,
    origin: np.ndarray,
    alignment: InterfaceAlignment,
    *,
    _inverse: np.ndarray | None = None,
    _cell_polygon: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """Return a conservative, periodic overlap stencil for one lateral bin."""
    cell_polygon = (
        _cell_polygon
        if _cell_polygon is not None
        else np.asarray(
            [origin, origin + basis[0], origin + basis[0] + basis[1], origin + basis[1]]
        )
    )
    lower = np.asarray([source_edges[i][source_indices[i]] for i in range(2)])
    upper = np.asarray([source_edges[i][source_indices[i] + 1] for i in range(2)])
    source = _rectangle(cell_polygon, lower, upper)
    source_area = _area(source)
    if source_area <= 1e-18:
        raise ValueError("An occupied density bin has no physical cell coverage.")
    inverse = np.linalg.inv(basis) if _inverse is None else _inverse
    fractional = ((source - origin) @ inverse) @ np.asarray(
        alignment.matrix
    ) + alignment.translation
    ranges = [
        range(int(np.floor(np.min(fractional[:, i]))), int(np.floor(np.max(fractional[:, i]))) + 1)
        for i in range(2)
    ]
    weights: dict[tuple[int, int], float] = {}
    for image in product(*ranges):
        part = _rectangle(fractional - image, np.zeros(2), np.ones(2))
        if _area(part) <= 1e-18:
            continue
        polygon = part @ basis + origin
        lo, hi = np.min(polygon, axis=0), np.max(polygon, axis=0)
        bins = [
            range(
                max(0, int(np.searchsorted(target_edges[i], lo[i], side="right") - 1)),
                min(
                    len(target_edges[i]) - 1,
                    int(np.searchsorted(target_edges[i], hi[i], side="left") + 1),
                ),
            )
            for i in range(2)
        ]
        for ix, iy in product(*bins):
            clipped = _rectangle(
                polygon,
                np.asarray([target_edges[0][ix], target_edges[1][iy]]),
                np.asarray([target_edges[0][ix + 1], target_edges[1][iy + 1]]),
            )
            area = _area(clipped)
            if area > source_area * 1e-13:
                weights[ix, iy] = weights.get((ix, iy), 0.0) + area / source_area
    total = sum(weights.values())
    if not np.isclose(total, 1.0, rtol=1e-8, atol=1e-9):
        raise ValueError("Interface remapping did not preserve the full periodic footprint.")
    # Remove roundoff only; this also gives exact permutations for coincident bins.
    return [(ix, iy, value / total) for (ix, iy), value in sorted(weights.items())]


def _bottom_active(group, extra):
    result = (np.asarray(group["bottom_mass_sum_g"]) != 0) | (
        np.asarray(group["bottom_entity_sum"]) != 0
    )
    for name in extra:
        result |= np.asarray(group[name]) != 0
    return result


def _lateral_ids(flat, shape, axes):
    strides = [int(np.prod(shape[a + 1 :], dtype=object)) for a in axes]
    first = (flat // strides[0]) % shape[axes[0]]
    second = (flat // strides[1]) % shape[axes[1]]
    return first * shape[axes[1]] + second


def _reindex(flat, old_shape, new_shape, offsets):
    if old_shape == new_shape and not np.any(offsets):
        return flat.copy()
    result = np.zeros(len(flat), dtype=np.int64)
    for axis in range(4):
        old_stride = int(np.prod(old_shape[axis + 1 :], dtype=object))
        new_stride = int(np.prod(new_shape[axis + 1 :], dtype=object))
        result += ((flat // old_stride) % old_shape[axis] + offsets[axis]) * new_stride
    return result


def align_density_grid(
    profile: Any,
    alignment: InterfaceAlignment,
    *,
    bottom_channels: tuple[str, ...] = (),
    _capture_bottom_totals: dict[str, np.ndarray] | None = None,
) -> Any:
    started = perf_counter()
    report_progress("Aligning interfaces")
    if profile.interface_reference is None:
        raise ValueError("Recompute this density/position file to enable interface alignment.")
    validate_alignment(profile.interface_reference, alignment)
    if profile.interface_alignment_metadata is not None:
        raise ValueError("Apply alignment to the original density grid only.")
    if not profile.geometry_groups or any(
        group.get(name) is None
        for group in profile.geometry_groups
        for name in ("top_mass_sum_g", "bottom_mass_sum_g", "top_entity_sum", "bottom_entity_sum")
    ):
        raise ValueError(
            "Recompute density with a side-resolved sparse grid to enable interface alignment."
        )
    axes = list(profile.interface_reference.lateral_indices)
    names = ("x", "y", "z", "distance")
    old_edges = [np.asarray(getattr(profile, f"{name}_bin_edges"), dtype=float) for name in names]
    new_edges = [edge.copy() for edge in old_edges]
    offsets = np.zeros(4, dtype=int)
    for axis in axes:
        lower = min(
            cartesian_bounds(g["cell"], g["origin"])[axis, 0] for g in profile.geometry_groups
        )
        upper = max(
            cartesian_bounds(g["cell"], g["origin"])[axis, 1] for g in profile.geometry_groups
        )
        edge = old_edges[axis]
        width = edge[1] - edge[0]
        start = min(0, int(np.floor((lower - edge[0]) / width + 1e-9)))
        stop = max(len(edge) - 1, int(np.ceil((upper - edge[0]) / width - 1e-9)))
        new_edges[axis] = edge[0] + np.arange(start, stop + 1) * width
        offsets[axis] = -start
    old_shape = tuple(len(e) - 1 for e in old_edges)
    new_shape = tuple(len(e) - 1 for e in new_edges)
    base_value_names = (
        "mass_sum_g",
        "entity_sum",
        "top_mass_sum_g",
        "bottom_mass_sum_g",
        "top_entity_sum",
        "bottom_entity_sum",
    )
    if any(not name.startswith("bottom_") for name in bottom_channels):
        raise ValueError("Additional registered channels must belong to the negative interface.")
    value_names = (*base_value_names, *bottom_channels)
    # Compile each lateral geometry once, independently of normal shifts.
    geometries: dict[tuple[bytes, bytes], Any] = {}
    group_keys = []
    for group in profile.geometry_groups:
        check_cancelled()
        basis = np.asarray(group["cell"], dtype=float)[np.ix_(axes, axes)]
        origin = np.asarray(group["origin"], dtype=float)[axes]
        key = basis.tobytes(), origin.tobytes()
        group_keys.append(key)
        geometry = geometries.setdefault(key, {"basis": basis, "origin": origin, "bins": []})
        if not interval_compatible(basis, alignment.matrix):
            active = _bottom_active(group, bottom_channels)
            geometry["bins"].append(
                _lateral_ids(
                    np.asarray(group["flat_indices"], dtype=np.int64)[active], old_shape, axes
                )
            )
    operators = {}
    for key, geometry in geometries.items():
        bins = np.unique(np.concatenate(geometry["bins"])) if geometry["bins"] else None
        operators[key] = transfer_operator(
            tuple(old_edges[a] for a in axes),
            tuple(new_edges[a] for a in axes),
            geometry["basis"],
            geometry["origin"],
            alignment,
            source_bins=bins,
        )
    compiled_at = perf_counter()
    dtype = row_dtype(len(value_names))
    # Reserve space for expanded indices, weights, channel rows and their vector temporaries.
    max_entries = max(1, expansion_budget() // (2 * dtype.itemsize + 128))
    input_batch = min(65536, max_entries)
    new_strides = [int(np.prod(new_shape[a + 1 :], dtype=object)) for a in range(4)]
    with SparseWorkspace() as workspace:
        reducer = SparseReducer(workspace, len(value_names))
        for group_id, (group, key) in enumerate(zip(profile.geometry_groups, group_keys)):
            check_cancelled()
            original_flat = np.asarray(group["flat_indices"], dtype=np.int64)
            active_rows = np.flatnonzero(_bottom_active(group, bottom_channels))
            for start in range(0, len(original_flat), input_batch):
                check_cancelled()
                stop = min(len(original_flat), start + input_batch)
                rows = np.zeros(stop - start, dtype=dtype)
                rows["group"] = group_id
                rows["bin"] = _reindex(original_flat[start:stop], old_shape, new_shape, offsets)
                for column, name in enumerate(value_names):
                    original = np.asarray(group[name])[start:stop]
                    if name.startswith("top_"):
                        rows["values"][:, column] = original
                    elif not name.startswith("bottom_"):
                        rows["values"][:, column] = (
                            original - np.asarray(group[f"bottom_{name}"])[start:stop]
                        )
                reducer.add(rows)
            for start in range(0, len(active_rows), input_batch):
                check_cancelled()
                source = active_rows[start : start + input_batch]
                source_flat = original_flat[source]
                bins = _lateral_ids(source_flat, old_shape, axes)
                base = _reindex(source_flat, old_shape, new_shape, offsets)
                first = bins // old_shape[axes[1]]
                second = bins % old_shape[axes[1]]
                base -= (first + offsets[axes[0]]) * new_strides[axes[0]]
                base -= (second + offsets[axes[1]]) * new_strides[axes[1]]
                for local, target, weights in operators[key].expand(bins, max_entries):
                    rows = np.zeros(len(local), dtype=dtype)
                    rows["group"] = group_id
                    rows["bin"] = (
                        base[local]
                        + (target // new_shape[axes[1]]) * new_strides[axes[0]]
                        + (target % new_shape[axes[1]]) * new_strides[axes[1]]
                    )
                    for column, name in enumerate(value_names):
                        if not name.startswith("top_"):
                            channel = name if name.startswith("bottom_") else f"bottom_{name}"
                            rows["values"][:, column] = (
                                np.asarray(group[channel])[source[local]] * weights
                            )
                    reducer.add(rows)
        run = reducer.finish()
        flat, channel_values, group_offsets = run.unpack(float, len(profile.geometry_groups))
        run.release()
        groups = []
        for group_id, group in enumerate(profile.geometry_groups):
            check_cancelled()
            start, stop = group_offsets[group_id : group_id + 2]
            updated = dict(group, flat_indices=flat[start:stop])
            updated.update(
                {name: values[start:stop] for name, values in zip(value_names, channel_values)}
            )
            groups.append(updated)
    unique, inverse = np.unique(flat, return_inverse=True)
    values = {
        name: np.bincount(inverse, weights=column, minlength=len(unique))
        for name, column in zip(value_names, channel_values)
    }
    if _capture_bottom_totals is not None:
        _capture_bottom_totals.update({name: values[name] for name in bottom_channels})
    check_cancelled()
    LOGGER.debug(
        "Interface remap: geometries=%d rows=%d stencil_s=%.6f apply_s=%.6f",
        len(geometries),
        len(flat),
        compiled_at - started,
        perf_counter() - compiled_at,
    )
    metadata = {
        **alignment.to_dict(),
        "source_grid_resolution_A": [float(e[1] - e[0]) for e in old_edges[:3]],
        "remapping": "conservative periodic footprint overlap",
    }
    return replace(
        profile,
        **{f"{name}_bin_edges": edge for name, edge in zip(names, new_edges)},
        flat_indices=unique,
        geometry_groups=groups,
        interface_alignment_metadata=metadata,
        **{name: values[name] for name in base_value_names},
    )
