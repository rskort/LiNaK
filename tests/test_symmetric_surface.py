from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

import numpy as np
import pytest
from ase import Atoms
import linak.analysis.density as density_mod
import linak.analysis.water as water_mod

from linak.analysis.density import (
    DensityGridProfile,
    DensityProfile,
    compute_all_density_profiles,
    compute_density_profile,
    compute_density_profiles,
    density_grid_to_heatmap_profile,
    density_grid_to_line_profile,
    load_density_grid_profiles,
    plot_density_profiles,
    save_density_profiles,
)
from linak.analysis.orientation import (
    _orientation_profile_for_interface_view,
    compute_orientation_profile,
    plot_orientation_profile,
)
from linak.analysis.position import (
    _position_profile_for_interface_view,
    compute_position_profile,
)
from linak.analysis.surface import (
    estimate_surface_layout_reference,
    surface_distance_components,
)


class _SurfaceSettings(TypedDict):
    surface_layout: str
    surface_mode: str
    surface_elements: list[str]


class _AxisSurfaceSettings(_SurfaceSettings):
    axis: str


def _symmetric_frame(*, crossing_boundary: bool = False) -> Atoms:
    symbols: list[str] = []
    positions: list[list[float]] = []
    slab_z = (18.0, 19.0, 0.0, 1.0) if crossing_boundary else (8.0, 10.0, 12.0)
    for z_value in slab_z:
        for x_value, y_value in ((2.0, 2.0), (7.0, 2.0), (2.0, 7.0), (7.0, 7.0)):
            symbols.append("Pt")
            positions.append([x_value, y_value, z_value])

    bottom_oxygen_z, top_oxygen_z = (16.0, 3.0) if crossing_boundary else (4.0, 16.0)
    for oxygen_z, outward_sign in ((bottom_oxygen_z, -1.0), (top_oxygen_z, 1.0)):
        symbols.extend(("O", "H", "H"))
        positions.extend(
            (
                [5.0, 5.0, oxygen_z],
                [5.5, 5.0, oxygen_z + outward_sign * 0.65],
                [4.5, 5.0, oxygen_z + outward_sign * 0.65],
            )
        )

    return Atoms(symbols, positions=positions, cell=[10.0, 10.0, 20.0], pbc=True)


def _single_top_boundary_frame() -> Atoms:
    """Slab crosses Z=0 while all fluid lies outward from its top face."""

    symbols: list[str] = []
    positions: list[list[float]] = []
    for z_value in (18.0, 19.0, 0.0, 1.0):
        for x_value, y_value in ((2.0, 2.0), (7.0, 2.0), (2.0, 7.0), (7.0, 7.0)):
            symbols.append("Pt")
            positions.append([x_value, y_value, z_value])
    symbols.extend(("O", "H", "H", "Li"))
    positions.extend(
        (
            [5.0, 5.0, 3.0],
            [5.5, 5.0, 3.65],
            [4.5, 5.0, 3.65],
            [5.0, 5.0, 10.0],
        )
    )
    return Atoms(symbols, positions=positions, cell=[10.0, 10.0, 20.0], pbc=True)


@pytest.mark.parametrize("layout", ["auto", "single-top", "single-bottom", "symmetric"])
def test_current_surface_cache_round_trip_returns_reference(tmp_path, layout):
    from linak.trajectory.io import (
        TrajectoryStoredMetadata,
        read_trajectory_hdf5_surface_cache,
        write_trajectory,
    )

    frames = [_symmetric_frame(), _symmetric_frame()]
    frames[1].positions[:, 2] += 0.1
    reference = estimate_surface_layout_reference(
        frames, surface_layout=layout, surface_elements=["Pt"]
    )
    path = tmp_path / "paired.traj.h5"
    write_trajectory(
        frames,
        path,
        metadata=TrajectoryStoredMetadata(
            surface_cache_status="available",
            surface_cache_axis="z",
            surface_cache_mode="auto",
            surface_cache_layout=layout,
            surface_cache_elements=("Pt",),
            surface_cache_estimate=reference,
        ),
    )
    loaded = read_trajectory_hdf5_surface_cache(
        path,
        axis="z",
        surface_mode="auto",
        surface_layout=layout,
        surface_elements=["Pt"],
        include_fixed_surface_atoms=False,
        rough_surface_envelope_A=None,
        frame_count=len(frames),
    )
    assert loaded is not None
    assert loaded.requested_layout == reference.requested_layout
    assert loaded.resolved_layout == reference.resolved_layout
    np.testing.assert_array_equal(loaded.axis_cut_per_frame_A, reference.axis_cut_per_frame_A)
    for expected, actual in ((reference.top, loaded.top), (reference.bottom, loaded.bottom)):
        assert expected is not None and actual is not None
        np.testing.assert_array_equal(actual.frame_values, expected.frame_values)


def _single_bottom_boundary_frame() -> Atoms:
    frame = _single_top_boundary_frame()
    positions = np.asarray(frame.positions, dtype=float).copy()
    positions[:, 2] = np.mod(20.0 - positions[:, 2], 20.0)
    frame.positions = positions
    return frame


def test_symmetric_reference_assigns_outward_distances_and_auto_detects_both_sides():
    frame = _symmetric_frame()
    reference = estimate_surface_layout_reference(
        [frame],
        axis="z",
        surface_layout="auto",
        mode="layered",
        surface_elements=["Pt"],
    )

    assert reference.resolved_layout == "symmetric"
    assert reference.auto_reason == "auto_two_sided_water"
    folded, top, bottom, side = surface_distance_components(
        reference,
        np.asarray([4.0, 16.0]),
        frame_index=0,
    )
    np.testing.assert_allclose(folded, [4.0, 4.0], atol=1.0e-8)
    np.testing.assert_allclose(top, [-8.0, 4.0], atol=1.0e-8)
    np.testing.assert_allclose(bottom, [4.0, -8.0], atol=1.0e-8)
    np.testing.assert_array_equal(side, [-1, 1])


def test_symmetric_density_position_and_orientation_are_side_resolved():
    frame = _symmetric_frame()
    common: _AxisSurfaceSettings = {
        "axis": "z",
        "surface_layout": "symmetric",
        "surface_mode": "layered",
        "surface_elements": ["Pt"],
    }
    density = compute_density_profile(
        [frame],
        species="O",
        bin_width=1.0,
        **common,
    )
    assert density.coordinate_mode == "distance"
    assert density.top_density is not None
    assert density.bottom_density is not None
    np.testing.assert_allclose(density.top_density, density.bottom_density)
    np.testing.assert_allclose(
        density.density,
        0.5 * (density.top_density + density.bottom_density),
    )

    position = compute_position_profile([frame], species="O", **common)
    assert position.distance_to_top_surface is not None
    assert position.distance_to_bottom_surface is not None
    assert position.interface_side is not None
    np.testing.assert_array_equal(position.interface_side.reshape(-1), [-1, 1])
    np.testing.assert_allclose(position.distance_to_surface.reshape(-1), [4.0, 4.0])

    orientation = compute_orientation_profile(
        [frame],
        reference_axis="z",
        bin_width=1.0,
        **common,
    )
    assert orientation.top_cos_polar_mean is not None
    assert orientation.bottom_cos_polar_mean is not None
    finite_top = orientation.top_cos_polar_mean[np.isfinite(orientation.top_cos_polar_mean)]
    finite_bottom = orientation.bottom_cos_polar_mean[
        np.isfinite(orientation.bottom_cos_polar_mean)
    ]
    assert finite_top.size == finite_bottom.size == 1
    assert finite_top[0] > 0.0
    np.testing.assert_allclose(finite_top, finite_bottom, atol=1.0e-8)


def test_symmetric_reference_canonicalizes_a_slab_crossing_the_periodic_boundary():
    frame = _symmetric_frame(crossing_boundary=True)
    reference = estimate_surface_layout_reference(
        [frame],
        axis="z",
        surface_layout="symmetric",
        mode="layered",
        surface_elements=["Pt"],
    )

    assert reference.resolved_layout == "symmetric"
    folded, _top, _bottom, side = surface_distance_components(
        reference,
        np.asarray([16.0, 3.0]),
        frame_index=0,
    )
    assert np.all(folded > 0.0)
    np.testing.assert_array_equal(side, [-1, 1])


def test_single_top_reference_keeps_boundary_crossing_fluid_at_positive_distance():
    frame = _single_top_boundary_frame()
    reference = estimate_surface_layout_reference(
        [frame],
        axis="z",
        surface_layout="auto",
        mode="layered",
        surface_elements=["Pt"],
    )

    assert reference.resolved_layout == "single-top"
    folded, top, _bottom, side = surface_distance_components(
        reference,
        np.asarray([18.0, 3.0, 10.0]),
        frame_index=0,
    )
    assert folded[0] < 0.0  # Substrate below the exposed top plane.
    assert np.all(folded[1:] > 0.0)  # Fluid remains outward across raw Z=0.
    np.testing.assert_allclose(folded, top)
    np.testing.assert_array_equal(side, [1, 1, 1])


def test_single_bottom_reference_keeps_boundary_crossing_fluid_at_positive_distance():
    frame = _single_bottom_boundary_frame()
    reference = estimate_surface_layout_reference(
        [frame],
        axis="z",
        surface_layout="single-bottom",
        mode="layered",
        surface_elements=["Pt"],
    )

    assert reference.resolved_layout == "single-bottom"
    pt_values = np.asarray(frame.positions[:16, 2], dtype=float)
    fluid_values = np.asarray(frame.positions[16:, 2], dtype=float)
    pt_distance = surface_distance_components(reference, pt_values, frame_index=0)[0]
    fluid_distance = surface_distance_components(reference, fluid_values, frame_index=0)[0]
    assert np.min(pt_distance) < 0.0
    assert np.all(fluid_distance > 0.0)


def test_auto_single_top_multi_species_density_preserves_canonical_surface_reference():
    profiles = compute_density_profiles(
        [_single_top_boundary_frame()],
        species="all",
        axis="z",
        bin_width=1.0,
        surface_layout="auto",
        surface_mode="layered",
        surface_elements=["Pt"],
    )

    for species in ("O", "H", "Li", "mol:H2O"):
        profile = next(item for item in profiles if item.species == species)
        occupied = profile.bin_centers[profile.counts_per_frame > 0.0]
        assert occupied.size
        assert np.all(occupied > 0.0), (species, occupied)


def test_single_top_cell_binning_spans_slab_and_complete_outward_fluid_arc():
    profile = compute_density_profile(
        [_single_top_boundary_frame()],
        species="all",
        axis="z",
        bin_width=1.0,
        binning="cell",
        surface_layout="single-top",
        surface_mode="layered",
        surface_elements=["Pt"],
    )

    assert profile.bin_edges[0] == pytest.approx(-3.0)
    assert profile.bin_edges[-1] == pytest.approx(17.0)
    assert profile.bin_edges[-1] - profile.bin_edges[0] == pytest.approx(20.0)


def test_single_top_sparse_distance_projection_uses_canonical_periodic_volume():
    profiles = compute_all_density_profiles(
        [_single_top_boundary_frame()],
        species="Li",
        surface_axis="z",
        bin_width=1.0,
        grid_bin_width=1.0,
        outputs="all",
        binning="cell",
        surface_layout="single-top",
        surface_mode="layered",
        surface_elements=["Pt"],
    )
    stored = next(
        item
        for item in profiles
        if isinstance(item, DensityProfile) and item.coordinate_mode == "distance"
    )
    grid = next(item for item in profiles if isinstance(item, DensityGridProfile))
    projected = density_grid_to_line_profile(grid, x_mode="distance")

    np.testing.assert_allclose(projected.bin_edges, stored.bin_edges)
    np.testing.assert_allclose(projected.density, stored.density, equal_nan=True)
    assert grid.geometry_groups is not None
    assert grid.geometry_groups[0]["canonical_period"] == pytest.approx(20.0)
    assert grid.geometry_groups[0]["canonical_wrap_threshold"] is not None


def test_single_top_sparse_geometry_round_trip_preserves_wrap_threshold(tmp_path):
    profiles = compute_all_density_profiles(
        [_single_top_boundary_frame()],
        species="Li",
        surface_axis="z",
        bin_width=1.0,
        grid_bin_width=1.0,
        outputs="3d",
        binning="cell",
        surface_layout="single-top",
        surface_mode="layered",
        surface_elements=["Pt"],
    )
    output = tmp_path / "single-top-density.h5"
    save_density_profiles(profiles, output)
    loaded = load_density_grid_profiles(output)[0]

    assert loaded.geometry_groups is not None
    group = loaded.geometry_groups[0]
    assert group["canonical_period"] == pytest.approx(20.0)
    assert group["canonical_wrap_threshold"] is not None
    projected = density_grid_to_line_profile(loaded, x_mode="distance")
    assert np.all(projected.bin_centers[projected.counts_per_frame > 0.0] > 0.0)


def test_symmetric_midplane_entities_are_split_half_weight_between_sides():
    frame = _symmetric_frame()
    frame += Atoms("Li", positions=[[5.0, 5.0, 10.0]])
    common: _SurfaceSettings = {
        "surface_layout": "symmetric",
        "surface_mode": "layered",
        "surface_elements": ["Pt"],
    }

    profile = compute_density_profile([frame], species="Li", axis="z", bin_width=1.0, **common)
    assert profile.entities_per_frame is not None
    assert profile.top_counts_per_frame is not None
    assert profile.bottom_counts_per_frame is not None
    assert np.isclose(np.sum(profile.top_counts_per_frame), np.sum(profile.bottom_counts_per_frame))

    profiles = compute_all_density_profiles(
        [frame], species="Li", surface_axis="z", bin_width=1.0, outputs="3d", **common
    )
    grid = next(item for item in profiles if isinstance(item, DensityGridProfile))
    assert grid.top_entity_sum is not None
    assert grid.bottom_entity_sum is not None
    assert np.isclose(np.sum(grid.top_entity_sum), 0.5)
    assert np.isclose(np.sum(grid.bottom_entity_sum), 0.5)
    assert np.isclose(
        np.sum(profile.top_counts_per_frame) + np.sum(profile.bottom_counts_per_frame),
        np.sum(grid.mass_sum_g),
    )


def test_streaming_symmetric_density_builds_lines_without_finalize_replay(monkeypatch):
    def fail_replay(*_args, **_kwargs):
        raise AssertionError("compute_density_profile finalize replay was used")

    monkeypatch.setattr(density_mod, "compute_density_profile", fail_replay)
    profiles = compute_all_density_profiles(
        [_symmetric_frame()],
        species="O",
        surface_axis="z",
        bin_width=1.0,
        outputs="all",
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )

    distance = next(
        item
        for item in profiles
        if isinstance(item, DensityProfile) and item.coordinate_mode == "distance"
    )
    assert distance.top_density is not None
    assert distance.bottom_density is not None
    np.testing.assert_allclose(
        distance.density,
        0.5 * (distance.top_density + distance.bottom_density),
    )


def test_symmetric_sparse_grid_round_trip_preserves_paired_geometry(tmp_path):
    profiles = compute_all_density_profiles(
        [_symmetric_frame()],
        species="O",
        surface_axis="z",
        bin_width=1.0,
        grid_bin_width=1.0,
        outputs="3d",
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )
    output = tmp_path / "symmetric-density.h5"
    save_density_profiles(profiles, output)
    grid = load_density_grid_profiles(output)[0]

    assert grid.geometry_groups is not None
    group = grid.geometry_groups[0]
    assert group["top_shift"] is not None
    assert group["bottom_shift"] is not None
    assert group["canonical_cut"] is not None
    assert group["canonical_lower"] is not None
    assert group["canonical_period"] is not None
    assert group["top_mass_sum_g"] is not None
    assert group["bottom_mass_sum_g"] is not None
    line = density_grid_to_line_profile(grid, x_mode="distance")
    heatmap = density_grid_to_heatmap_profile(
        grid,
        x_axis="x",
        y_axis="distance",
    )
    np.testing.assert_allclose(
        line.density,
        0.5 * (np.asarray(line.top_density) + np.asarray(line.bottom_density)),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        heatmap.density,
        0.5 * (np.asarray(heatmap.top_density) + np.asarray(heatmap.bottom_density)),
        equal_nan=True,
    )


def test_tilted_prism_acceleration_matches_exact_top_and_bottom_clipping(monkeypatch):
    frame = _symmetric_frame()
    frame.set_cell([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    grid = next(
        item
        for item in compute_all_density_profiles(
            [frame],
            species="O",
            surface_axis="z",
            bin_width=2.0,
            grid_bin_width=2.0,
            outputs="3d",
            surface_layout="symmetric",
            surface_mode="layered",
            surface_elements=["Pt"],
        )
        if isinstance(item, DensityGridProfile)
    )

    accelerated = density_grid_to_heatmap_profile(
        grid,
        x_axis="x",
        y_axis="distance",
    )
    empty_line = density_grid_to_line_profile(
        grid,
        x_mode="distance",
        filters={"x": (100.0, 101.0)},
    )
    assert empty_line.top_density is not None
    assert empty_line.bottom_density is not None
    assert not np.any(np.nan_to_num(empty_line.top_density))
    assert not np.any(np.nan_to_num(empty_line.bottom_density))
    monkeypatch.setattr(density_mod, "is_axis_prismatic", lambda *_args, **_kwargs: False)
    exact = density_grid_to_heatmap_profile(
        grid,
        x_axis="x",
        y_axis="distance",
    )

    np.testing.assert_allclose(accelerated.density, exact.density, rtol=1.0e-10, equal_nan=True)
    np.testing.assert_allclose(
        accelerated.top_density,
        exact.top_density,
        rtol=1.0e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        accelerated.bottom_density,
        exact.bottom_density,
        rtol=1.0e-10,
        equal_nan=True,
    )


def test_legacy_symmetric_grid_keeps_raw_views_but_rejects_distance_views():
    grid = next(
        item
        for item in compute_all_density_profiles(
            [_symmetric_frame()],
            species="O",
            surface_axis="z",
            bin_width=2.0,
            outputs="3d",
            surface_layout="symmetric",
            surface_mode="layered",
            surface_elements=["Pt"],
        )
        if isinstance(item, DensityGridProfile)
    )
    assert grid.geometry_groups is not None
    legacy_groups = []
    for group in grid.geometry_groups:
        legacy = dict(group)
        for name in (
            "top_shift",
            "bottom_shift",
            "top_mass_sum_g",
            "bottom_mass_sum_g",
            "top_entity_sum",
            "bottom_entity_sum",
        ):
            legacy.pop(name, None)
        legacy_groups.append(legacy)
    legacy_grid = replace(grid, geometry_groups=legacy_groups)

    raw = density_grid_to_heatmap_profile(legacy_grid, x_axis="x", y_axis="y")
    assert raw.density.size > 0
    with pytest.raises(ValueError, match="Recompute the density analysis"):
        density_grid_to_heatmap_profile(
            legacy_grid,
            x_axis="x",
            y_axis="distance",
        )


def test_tilted_prism_projection_avoids_per_bin_polyhedral_clipping(monkeypatch):
    frame = _symmetric_frame()
    frame.set_cell([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    grid = next(
        item
        for item in compute_all_density_profiles(
            [frame.copy() for _ in range(6)],
            species="O",
            surface_axis="z",
            bin_width=1.0,
            outputs="3d",
            surface_layout="symmetric",
            surface_mode="layered",
            surface_elements=["Pt"],
        )
        if isinstance(item, DensityGridProfile)
    )
    original = density_mod.intersection_volume
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(density_mod, "intersection_volume", counted)
    density_grid_to_heatmap_profile(
        grid,
        x_axis="x",
        y_axis="distance",
    )

    assert calls == 0


def test_auto_symmetric_occupancy_requirement_counts_empty_frames():
    populated = _symmetric_frame()
    empty = populated[:12]
    frames = [populated.copy() for _ in range(8)] + [empty.copy() for _ in range(2)]

    reference = estimate_surface_layout_reference(
        frames,
        axis="z",
        surface_layout="auto",
        mode="layered",
        surface_elements=["Pt"],
    )

    assert reference.resolved_layout != "symmetric"


def test_symmetric_population_detection_reuses_stable_water_topology(monkeypatch):
    original = water_mod._oh_molecule_topology_with_backend
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(water_mod, "_oh_molecule_topology_with_backend", counted)
    frames = [_symmetric_frame() for _ in range(200)]
    reference = estimate_surface_layout_reference(
        frames,
        axis="z",
        surface_layout="auto",
        mode="layered",
        surface_elements=["Pt"],
    )

    assert reference.resolved_layout == "symmetric"
    assert calls == len(frames)
    estimate_surface_layout_reference(
        frames, axis="z", surface_layout="auto", mode="layered", surface_elements=["Pt"]
    )
    assert calls == len(frames)


def test_all_interface_plot_view_expands_and_labels_density_series(monkeypatch):
    profile = compute_density_profile(
        [_symmetric_frame()],
        species="O",
        axis="z",
        bin_width=1.0,
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    def fake_plot(x_series, y_series, labels, **_kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["labels"] = labels
        return None

    monkeypatch.setattr(density_mod, "plot_multi_line_series", fake_plot)
    plot_density_profiles([profile], show=False, interface_view="all")

    assert captured["labels"] == [
        "O (Combined interfaces (folded mean))",
        "O (Positive-axis (+Z))",
        "O (Negative-axis (-Z))",
    ]
    assert isinstance(captured["y_series"], list)
    assert len(captured["y_series"]) == 3


def test_position_side_view_masks_all_opposing_side_coordinates():
    profile = compute_position_profile(
        [_symmetric_frame()],
        species="O",
        axis="z",
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )

    top = _position_profile_for_interface_view(profile, "top")
    bottom = _position_profile_for_interface_view(profile, "bottom")

    assert np.isnan(top.x[0, 0]) and np.isfinite(top.x[0, 1])
    assert np.isfinite(bottom.x[0, 0]) and np.isnan(bottom.x[0, 1])
    assert np.isnan(top.distance_to_surface[0, 0])
    assert np.isnan(bottom.distance_to_surface[0, 1])


def test_full_cell_position_uses_unfolded_raw_surface_axis():
    profile = compute_position_profile(
        [_symmetric_frame()],
        species="O",
        axis="z",
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )

    full = _position_profile_for_interface_view(profile, "full")

    assert full.coordinate_mode == "axis"
    np.testing.assert_allclose(full.distance_to_surface.reshape(-1), [4.0, 16.0])
    assert np.all(np.isfinite(full.x))


def test_full_cell_orientation_uses_laboratory_positive_axis():
    profile = compute_orientation_profile(
        [_symmetric_frame()],
        axis="z",
        reference_axis="z",
        bin_width=1.0,
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )

    full = _orientation_profile_for_interface_view(profile, "full")
    occupied = np.flatnonzero(np.isfinite(full.cos_polar_mean))

    assert full.coordinate_mode == "axis"
    assert occupied.size == 2
    lower, upper = occupied[np.argsort(full.bin_centers[occupied])]
    assert full.cos_polar_mean[lower] < 0.0
    assert full.cos_polar_mean[upper] > 0.0
    assert full.sparse_grid is not None
    assert full.sparse_grid.lab_cos_polar_sum is not None

    plot_orientation_profile(
        profile,
        show=False,
        component="density",
        interface_view="full",
        orientation_line_x_axis="z",
    )


def test_full_cell_density_requires_raw_axis_profile():
    profiles = compute_all_density_profiles(
        [_symmetric_frame()],
        species="O",
        surface_axis="z",
        bin_width=1.0,
        outputs="1d",
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
    )
    raw = next(
        profile
        for profile in profiles
        if isinstance(profile, DensityProfile)
        and profile.axis == "z"
        and profile.coordinate_mode == "axis"
    )
    folded = next(
        profile
        for profile in profiles
        if isinstance(profile, DensityProfile) and profile.coordinate_mode == "distance"
    )

    plot_density_profiles([raw], show=False, interface_view="full")
    with pytest.raises(ValueError, match="folded surface-distance bins"):
        plot_density_profiles([folded], show=False, interface_view="full")


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("crossing", [False, True])
def test_shared_surface_context_matches_independent_estimates(monkeypatch, fixed, crossing):
    from dataclasses import asdict
    from ase.constraints import FixAtoms
    import linak.analysis.surface as surface

    frames = [_symmetric_frame(crossing_boundary=crossing) for _ in range(3)]
    if fixed:
        for frame in frames:
            frame.set_constraint(FixAtoms(indices=list(range(4))))
    shared = surface.estimate_surface_layout_reference(frames, surface_elements=["Pt"])
    original = surface.estimate_surface_reference

    def independent(*args, **kwargs):
        kwargs.pop("_context", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(surface, "estimate_surface_reference", independent)
    separate = surface.estimate_surface_layout_reference(frames, surface_elements=["Pt"])

    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, np.ndarray):
            np.testing.assert_equal(left, right)
        else:
            assert left == right

    compare(asdict(shared), asdict(separate))
