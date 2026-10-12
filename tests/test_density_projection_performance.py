from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from linak.analysis import density as density_module


@pytest.mark.parametrize("layout", ["single-top", "single-bottom", "symmetric", "axis"])
@pytest.mark.parametrize("side", ["top", "bottom"])
@pytest.mark.parametrize(
    "filters",
    [{}, {"z": (2.0, 8.0)}, {"distance": (-1.2, 3.4)}, {"z": (4.0, 3.0)}, {"distance": (3.0, 2.0)}],
)
def test_vector_distance_overlap_matches_scalar_periodic_geometry(layout, side, filters):
    profile = SimpleNamespace(
        surface_axis="z", surface_reference=SimpleNamespace(resolved_layout=layout)
    )
    rng = np.random.default_rng(948)
    for cut, threshold in zip(rng.uniform(-10, 10, 5), rng.uniform(1, 3, 5)):
        group = dict(
            canonical_period=10.0,
            canonical_lower=0.0,
            canonical_cut=cut,
            top_shift=6.0,
            bottom_shift=2.0,
            canonical_wrap_threshold=threshold,
        )
        # Fine bins, bins wider than the period, boundary hits and empty bins.
        for edges in (np.linspace(-12, 20, 1001), np.array([-20.0, -10.0, 0.0, 10.0, 30.0])):
            actual = density_module._canonical_distance_bin_overlaps(
                profile, group, edges, filters, distance_side=side
            )
            expected = []
            for lower, upper in zip(edges[:-1], edges[1:]):
                distance_filter = filters.get("distance", (None, None))
                limits = (
                    max(lower, -np.inf if distance_filter[0] is None else distance_filter[0]),
                    min(upper, np.inf if distance_filter[1] is None else distance_filter[1]),
                )
                expected.append(
                    density_module._canonical_normal_overlap(
                        profile,
                        group,
                        raw_limits=filters.get("z", (None, None)),
                        distance_limits=limits,
                        distance_side=side,
                    )
                )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("axes", [("x", "y"), ("x", "distance"), ("distance", "x"), ("x", "z")])
@pytest.mark.parametrize("layout", ["single-top", "single-bottom"])
def test_sparse_prism_projection_matches_dense_normalization(monkeypatch, axes, layout):
    edges = np.linspace(0.0, 10.0, 11)
    flat = np.ravel_multi_index(([1, 1, 3], [2, 2, 4], [3, 4, 5], [4, 4, 6]), (10,) * 4)
    groups = [
        dict(
            cell=np.diag([10.0, 10.0, 10.0]),
            origin=np.zeros(3),
            shift=5.0 + index / 10,
            top_shift=5.0 + index / 10,
            bottom_shift=2.0,
            canonical_cut=index / 5,
            canonical_lower=0.0,
            canonical_period=10.0,
            canonical_wrap_threshold=2.0,
            n_frames=index + 1,
            flat_indices=flat,
            mass_sum_g=np.array([1.0, 2.0, 3.0]) * 1e-23,
            entity_sum=np.array([1.0, 2.0, 3.0]),
        )
        for index in range(12)
    ]
    profile = density_module.DensityGridProfile(
        species="K",
        x_bin_edges=edges,
        y_bin_edges=edges,
        z_bin_edges=edges,
        distance_bin_edges=edges,
        flat_indices=flat,
        mass_sum_g=groups[0]["mass_sum_g"],
        entity_sum=groups[0]["entity_sum"],
        n_frames=78,
        geometry_groups=groups,
    )
    profile = replace(
        profile, surface_reference=SimpleNamespace(resolved_layout=layout, is_symmetric=False)
    )
    cache = {}
    actual = density_module._geometry_project_density(
        profile, axes, (edges, edges), {}, _volume_cache=cache
    )
    assert not cache, "Prisms must not retain dense volume grids per surface position"
    factors_function = density_module._prismatic_geometry_volume_factors

    def dense_oracle(profile, group, axes, edges, filters, **kwargs):
        factors = factors_function(profile, group, axes, edges, filters, area_cache={}, **kwargs)
        return np.broadcast_to(factors[0] * factors[1], (10, 10)).copy()

    monkeypatch.setattr(density_module, "_prismatic_geometry_volume_factors", lambda *a, **kw: None)
    monkeypatch.setattr(density_module, "_prismatic_geometry_volumes", lambda *a, **kw: None)
    monkeypatch.setattr(density_module, "_generic_geometry_volumes", dense_oracle)
    expected = density_module._geometry_project_density(profile, axes, (edges, edges), {})
    for result, reference in zip(actual[:2], expected[:2]):
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[2], expected[2])


def test_fallback_volume_cache_is_bounded(monkeypatch):
    groups = [
        dict(
            cell=np.eye(3) * (1.0 + index / 100),
            origin=np.zeros(3),
            n_frames=1,
            flat_indices=np.array([0]),
            mass_sum_g=np.array([1e-23]),
            entity_sum=np.array([1.0]),
        )
        for index in range(12)
    ]
    edges = np.array([0.0, 1.0])
    profile = density_module.DensityGridProfile(
        species="K",
        x_bin_edges=edges,
        y_bin_edges=edges,
        z_bin_edges=edges,
        distance_bin_edges=edges,
        flat_indices=np.array([0]),
        mass_sum_g=np.array([1e-23]),
        entity_sum=np.array([1.0]),
        n_frames=12,
        geometry_groups=groups,
    )
    monkeypatch.setattr(density_module, "_prismatic_geometry_volume_factors", lambda *a, **kw: None)
    monkeypatch.setattr(density_module, "_prismatic_geometry_volumes", lambda *a, **kw: None)
    monkeypatch.setattr(
        density_module, "_generic_geometry_volumes", lambda *a, **kw: np.ones((512, 512))
    )
    cache = {}
    target_edges = np.linspace(0.0, 1.0, 513)
    density_module._geometry_project_density(
        profile, ("x", "y"), (target_edges, target_edges), {}, _volume_cache=cache
    )
    assert cache
    assert sum(volume.nbytes for volume in cache.values()) <= 16 * 1024 * 1024
