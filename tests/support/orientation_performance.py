"""Deterministic, dense lateral occupancy with sparse normal/distance geometry."""

from dataclasses import replace
import numpy as np

from tests.test_orientation_alignment import _frame, _profile


def orientation_case(n=64, groups=4):
    base = _profile([_frame()])
    template = base.sparse_grid.density_geometry_groups[0]
    shape = (n, n, 2, 1)
    flat = np.arange(int(np.prod(shape)), dtype=np.int64)
    top = (np.unravel_index(flat, shape)[2] == 1).astype(float)
    bottom = 1.0 - top
    channels = {
        "entity_sum": np.ones(len(flat)),
        "top_entity_sum": top,
        "bottom_entity_sum": bottom,
        "cos_polar_sum": 0.4 * top - 0.2 * bottom,
        "count_polar_valid": np.ones(len(flat)),
        "cos_azimuthal_sum": 0.5 * top + 0.3 * bottom,
        "count_azimuthal_valid": np.ones(len(flat)),
    }
    for side, weights, polar, azimuthal in (("top", top, 0.4, 0.5), ("bottom", bottom, -0.2, 0.3)):
        channels.update(
            {
                f"{side}_cos_polar_sum": polar * weights,
                f"{side}_count_polar_valid": weights,
                f"{side}_cos_azimuthal_sum": azimuthal * weights,
                f"{side}_count_azimuthal_valid": weights,
            }
        )
    geometry = []
    for i in range(groups):
        group = dict(
            template,
            flat_indices=flat,
            n_frames=1,
            shift=template["shift"] + 0.01 * i,
            top_shift=template["top_shift"] + 0.01 * i,
        )
        for key in ("mass_sum_g", "top_mass_sum_g", "bottom_mass_sum_g"):
            group[key] = np.zeros(len(flat))
        for key in (
            "entity_sum",
            "top_entity_sum",
            "bottom_entity_sum",
            "bottom_cos_polar_sum",
            "bottom_count_polar_valid",
            "bottom_cos_azimuthal_sum",
            "bottom_count_azimuthal_valid",
        ):
            group[key] = channels[key]
        geometry.append(group)
    grid = replace(
        base.sparse_grid,
        x_edges=np.linspace(0.0, 12.0, n + 1),
        y_edges=np.linspace(0.0, 12.0, n + 1),
        z_edges=np.array([0.0, 10.0, 20.0]),
        distance_edges=np.array([3.0, 6.0]),
        shape=shape,
        flat_indices=flat,
        density_geometry_groups=geometry,
        **{key: value * groups for key, value in channels.items()},
    )
    return replace(base, n_frames=groups, sparse_grid=grid)
