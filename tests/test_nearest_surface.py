from __future__ import annotations

from itertools import product

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic

from linak.analysis import density, position, orientation
from linak.analysis.surface import (
    NearestSubstrateSearch,
    SurfaceEstimatorOptions,
    estimate_surface_layout_reference,
    surface_distance_components,
)
from linak.cli import main
from linak.trajectory.io import read_trajectory, read_trajectory_hdf5_metadata
from linak.trajectory.spatial_filter import SpatialFilterOptions, apply_spatial_filter


def options(**kwargs):
    return SurfaceEstimatorOptions(distance_mode="nearest-atom", surface_elements=("Pt",), **kwargs)


def slab():
    return Atoms(
        "Pt4KOHH",
        positions=[
            [0, 0, 2],
            [2, 0, 2],
            [0, 0, 4],
            [2, 0, 4],
            [1, 0, 7],
            [1, 0, 8],
            [1.9, 0, 8],
            [1, 0.9, 8],
        ],
        cell=[10, 10, 16],
        pbc=True,
    )


@pytest.mark.parametrize("pbc", list(product((False, True), repeat=3)))
@pytest.mark.parametrize(
    "cell", [np.diag([9.0, 10.0, 11.0]), np.array([[9.0, 0, 0], [8.5, 2.0, 0], [8.0, 1.8, 3.0]])]
)
def test_periodic_search_matches_bruteforce_mic(cell, pbc):
    rng = np.random.default_rng(891)
    frame = Atoms(cell=cell, pbc=pbc)
    candidates = rng.normal(size=(23, 3)) @ cell + [4, -9, 11]
    targets = rng.normal(size=(61, 3)) @ cell + [4, -9, 11]
    actual = NearestSubstrateSearch(frame, candidates).query(targets)
    vectors = targets[:, None, :] - candidates[None, :, :]
    if all(pbc):
        expected = (
            find_mic(vectors.reshape(-1, 3), cell, pbc=pbc)[1].reshape(len(targets), -1).min(axis=1)
        )
    else:
        # Independent exhaustive lattice enumeration avoids ASE's orthorhombic
        # fast-path assumption for a skewed partially periodic cell.
        expected = np.full(len(targets), np.inf)
        shifts = np.asarray(list(product(*[range(-24, 25) if p else (0,) for p in pbc]))) @ cell
        for shift in shifts:
            expected = np.minimum(expected, np.linalg.norm(vectors - shift, axis=2).min(axis=1))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_analytic_distances_no_plane_estimation(monkeypatch):
    import linak.analysis.surface as module

    monkeypatch.setattr(
        module, "estimate_surface_reference", lambda *a, **k: pytest.fail("plane estimator called")
    )
    frame = Atoms(
        "Pt2K2", positions=[[0, 0, 0], [2, 0, 1], [0, 0, 3], [1, 0, 3]], cell=[10, 10, 10], pbc=True
    )
    ref = estimate_surface_layout_reference(
        [frame], surface_layout="single-top", surface_elements=["Pt"], surface_options=options()
    )
    distances = surface_distance_components(
        ref, frame.positions[:, 2], frame_index=0, target_positions=frame.positions
    )[0]
    np.testing.assert_allclose(distances, [0, 0, np.sqrt(8), np.sqrt(5)])
    frame.positions[1] = [1, 0, 2]
    changed = surface_distance_components(
        ref, frame.positions[:, 2], frame_index=0, target_positions=frame.positions
    )[0]
    assert changed[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("layout", ["single-top", "single-bottom", "symmetric", "auto"])
def test_position_density_and_sparse_projections(layout):
    frame = slab()
    kwargs = dict(surface_layout=layout, surface_elements=["Pt"], surface_options=options())
    p = position.compute_position_profile([frame], species="K", **kwargs)
    assert p.distance_to_surface[0, 0] == pytest.approx(np.sqrt(10))
    profiles = density.compute_all_density_profiles(
        [frame], species="K", bin_width=0.5, grid_bin_width=0.5, **kwargs
    )
    dist = [
        p
        for p in profiles
        if isinstance(p, density.DensityProfile) and p.coordinate_mode == "distance"
    ]
    assert all(
        p.units == "amu/Angstrom" and p.number_density_units == "atoms/Angstrom" for p in dist
    )
    grid = next(p for p in profiles if isinstance(p, density.DensityGridProfile))
    line = density.density_grid_to_line_profile(grid)
    heat = density.density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="distance")
    assert line.units == "amu/Angstrom"
    assert heat.units == "amu/Angstrom^2"
    expected_count = 0.5 if layout == "symmetric" else 1.0
    assert np.sum(line.entities_per_frame) == pytest.approx(expected_count)
    assert np.sum(heat.number_density) * 0.25 == pytest.approx(expected_count)


def test_fixed_candidates_and_empty_selection():
    frame = slab()
    frame.set_constraint(FixAtoms(indices=[2, 3]))
    kwargs = dict(surface_layout="single-top", surface_elements=["Pt"])
    excluded = position.compute_position_profile([frame], "K", surface_options=options(), **kwargs)
    included = position.compute_position_profile(
        [frame],
        "K",
        surface_options=options(include_fixed_surface_atoms=True),
        include_fixed_surface_atoms=True,
        **kwargs,
    )
    assert included.distance_to_surface[0, 0] < excluded.distance_to_surface[0, 0]
    frame.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    with pytest.raises(ValueError, match="No eligible"):
        position.compute_position_profile([frame], "K", surface_options=options(), **kwargs)


@pytest.mark.parametrize("binning", ["cell", "observed"])
@pytest.mark.parametrize("layout", ["single-top", "symmetric"])
def test_nearest_density_shared_observed_bins(binning, layout, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="linak.analysis.density")
    kwargs = dict(
        surface_layout=layout,
        surface_elements=["Pt"],
        surface_options=options(),
        binning=binning,
        bin_width=0.5,
        min_molecule_frames=1,
    )
    profiles = density.compute_all_density_profiles(
        [slab()], species="all", grid_bin_width=0.25, **kwargs
    )
    lines = [
        p
        for p in profiles
        if isinstance(p, density.DensityProfile) and p.coordinate_mode == "distance"
    ]
    grids = [p for p in profiles if isinstance(p, density.DensityGridProfile)]
    assert {"Pt", "K", "mol:H2O"} <= {p.species for p in lines}
    for line in lines:
        np.testing.assert_array_equal(line.bin_edges, lines[0].bin_edges)
        np.testing.assert_allclose(np.diff(line.bin_edges), 0.5)
    for grid in grids:
        np.testing.assert_array_equal(grid.distance_bin_edges, grids[0].distance_bin_edges)
        np.testing.assert_allclose(np.diff(grid.distance_bin_edges), 0.25)
    weight = 0.5 if layout == "symmetric" else 1.0
    substrate = next(p for p in lines if p.species == "Pt")
    water = next(p for p in lines if p.species == "mol:H2O")
    assert substrate.bin_edges[0] == 0.0
    assert substrate.entities_per_frame[0] == pytest.approx(4 * weight)
    assert np.count_nonzero(substrate.entities_per_frame) == 1
    assert np.sum(water.entities_per_frame) == pytest.approx(weight)
    assert "shared observed range" in caplog.text
    assert "Falling back to observed-data binning" not in caplog.text

    # The distance-only public API uses the same shared species range.
    caplog.clear()
    distance_profiles = density.compute_density_profiles([slab()], species="all", **kwargs)
    assert [p.species for p in distance_profiles] == [p.species for p in lines]
    for actual, expected in zip(distance_profiles, lines):
        np.testing.assert_array_equal(actual.bin_edges, expected.bin_edges)
        np.testing.assert_allclose(actual.density, expected.density)
        np.testing.assert_allclose(actual.entities_per_frame, expected.entities_per_frame)
    assert "Falling back to observed-data binning" not in caplog.text


@pytest.mark.parametrize("species", ["Pt", "mol:H2O"])
def test_single_nearest_density_cell_binning_is_not_a_warning(species, caplog):
    import logging

    caplog.set_level(logging.INFO)
    profile = density.compute_density_profile(
        [slab()],
        species=species,
        surface_layout="single-top",
        surface_options=options(),
        surface_elements=["Pt"],
        binning="cell",
        bin_width=0.5,
    )
    assert "bins use observed distances" in caplog.text
    assert "Falling back to observed-data binning" not in caplog.text
    expected_count = 4 if species == "Pt" else 1
    assert np.sum(profile.entities_per_frame) == pytest.approx(expected_count)
    if species == "Pt":
        assert profile.entities_per_frame[0] == 4
        assert np.count_nonzero(profile.entities_per_frame) == 1


def test_plane_density_missing_bounds_still_warns(monkeypatch, caplog):
    monkeypatch.setattr(density, "_surface_reference_histogram_bounds", lambda reference: None)
    density.compute_all_density_profiles(
        [slab()],
        species="K",
        surface_layout="single-top",
        surface_elements=["Pt"],
        binning="cell",
        outputs="1d",
    )
    assert "cell-derived distance-bin bounds could not be determined" in caplog.text
    assert "Falling back to observed-data binning" in caplog.text
    assert "usable cell was unavailable" not in caplog.text


def test_molecular_com_orientation_and_filter():
    frame = slab()
    kwargs = dict(surface_layout="single-top", surface_elements=["Pt"], surface_options=options())
    mol = position.compute_position_profile([frame], "mol:H2O", **kwargs)
    target = np.column_stack([mol.x[0], mol.y[0], mol.z[0]])
    expected = NearestSubstrateSearch(frame, frame.positions[:4]).query(target)
    np.testing.assert_allclose(mol.distance_to_surface[0], expected)
    orient = orientation.compute_orientation_profile([frame], bin_width=0.5, **kwargs)
    assert orient.surface_reference.distance_mode == "nearest-atom"
    filtered = apply_spatial_filter(
        [frame],
        options=SpatialFilterOptions(
            distance_range="0:3.3",
            surface_elements=("Pt",),
            surface_layout="single-top",
            surface_distance_mode="nearest-atom",
        ),
    )
    assert filtered.frames[0].get_chemical_symbols() == ["Pt"] * 4 + ["K"]


def test_convert_inherit_override_and_roundtrip(tmp_path, capsys):
    from ase.io import write

    source = tmp_path / "frame.xyz"
    write(source, slab())
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(source),
                "--surface-distance-mode",
                "nearest-atom",
                "--surface-elements",
                "Pt",
                "--surface-layout",
                "single-top",
            ]
        )
        == 0
    )
    target = tmp_path / "LiNaK_outputs/frame.traj.h5"
    stored = read_trajectory_hdf5_metadata(target)
    assert (
        stored is not None and stored.surface_distance_settings["distance_mode"] == "nearest-atom"
    )
    assert main(["--log-level", "ERROR", "compute", "position", str(target), "--species", "K"]) == 0
    result = position.load_position_profiles(tmp_path / "LiNaK_outputs/frame.position.h5")[0]
    assert result.surface_reference.distance_mode == "nearest-atom"
    assert result.distance_to_surface[0, 0] == pytest.approx(np.sqrt(10), rel=1e-6)
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "compute",
                "position",
                str(target),
                "--species",
                "K",
                "--surface-distance-mode",
                "plane",
            ]
        )
        == 0
    )
    plane = position.load_position_profiles(tmp_path / "LiNaK_outputs/frame_1.position.h5")[0]
    assert plane.surface_reference.distance_mode == "plane"
    assert len(read_trajectory(target)) == 1
    with h5py.File(tmp_path / "LiNaK_outputs/frame.position.h5") as f:
        assert "nearest-atom" in str(dict(f.attrs)) or any(
            "nearest-atom" in str(dict(g.attrs)) for g in f.values() if isinstance(g, h5py.Group)
        )
    capsys.readouterr()


def test_plane_default_is_identical():
    frame = slab()
    kwargs = dict(surface_layout="single-top", surface_elements=["Pt"])
    old = position.compute_position_profile([frame], "K", **kwargs)
    explicit = position.compute_position_profile(
        [frame],
        "K",
        surface_options=SurfaceEstimatorOptions(surface_elements=("Pt",), distance_mode="plane"),
        **kwargs,
    )
    np.testing.assert_array_equal(old.distance_to_surface, explicit.distance_to_surface)


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)])
def test_tilted_symmetric_geometry_fails_clearly(pbc):
    frame = slab()
    frame.set_cell([[10, 0, 2], [0, 10, 0], [0, 0, 16]])
    frame.set_pbc(pbc)
    with pytest.raises(ValueError, match="decoupled"):
        estimate_surface_layout_reference(
            [frame], surface_layout="symmetric", surface_elements=["Pt"], surface_options=options()
        )


def test_symmetric_interface_identity_ties_and_zero():
    frame = slab()
    ref = estimate_surface_layout_reference(
        [frame], surface_layout="symmetric", surface_elements=["Pt"], surface_options=options()
    )
    points = np.array([[0, 0, 6], [0, 0, 0], [0, 0, 3], [0, 0, 2]])
    folded, top, bottom, side = surface_distance_components(
        ref, points[:, 2], frame_index=0, target_positions=points
    )
    np.testing.assert_allclose(folded, [2, 2, 1, 0])
    np.testing.assert_array_equal(side, [1, -1, 0, -1])
    np.testing.assert_allclose(top, [2, 4, 1, 2])
    np.testing.assert_allclose(bottom, [4, 2, 1, 0])
    frame.positions[:, 2] += 7
    shifted = points + [0, 0, 7]
    updated = surface_distance_components(
        ref, shifted[:, 2], frame_index=0, target_positions=shifted
    )
    np.testing.assert_allclose(updated[0], folded)
    np.testing.assert_array_equal(updated[3], side)


def test_coordination_numbers_do_not_change():
    from linak.analysis.coordination import (
        CoordinationCutoffResolution,
        compute_coordination_profile,
    )

    kwargs = dict(
        species_a="K",
        species_b="O",
        surface_elements=["Pt"],
        surface_layout="single-top",
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=3.0, smoothing_width_A=0.4, mode="direct"
        ),
    )
    nearest = compute_coordination_profile([slab()], surface_options=options(), **kwargs)
    plane = compute_coordination_profile([slab()], **kwargs)
    np.testing.assert_array_equal(nearest.coordination_number, plane.coordination_number)
    assert nearest.distance_to_surface[0, 0] == pytest.approx(np.sqrt(10))


def test_com_crossing_boundary_changing_cells_and_streamed_filter(tmp_path):
    from linak.trajectory.io import write_trajectory
    from linak.trajectory.spatial_filter import apply_spatial_filter_streaming
    from linak.trajectory.execution import ChunkedTrajectorySequence

    frame = slab()
    frame.positions[5:] = [[9.8, 0, 8], [0.7, 0, 8], [9.8, 0.9, 8]]
    second = frame.copy()
    second.cell[0, 0] = 12
    second.positions[5:] = [[11.8, 0, 8], [0.7, 0, 8], [11.8, 0.9, 8]]
    frames = [frame, second]
    p = position.compute_position_profile(
        frames,
        "mol:H2O",
        surface_options=options(),
        surface_elements=["Pt"],
        surface_layout="single-top",
    )
    for i, f in enumerate(frames):
        com = np.column_stack([p.x[i], p.y[i], p.z[i]])
        expected = NearestSubstrateSearch(f, f.positions[:4]).query(com)
        np.testing.assert_allclose(p.distance_to_surface[i], expected)
    path = write_trajectory(frames, tmp_path / "frames.traj.h5")
    sequence = ChunkedTrajectorySequence([path], [2], chunk_frames=1)
    opts = SpatialFilterOptions(
        distance_range="0:3.3",
        surface_distance_mode="nearest-atom",
        surface_elements=("Pt",),
        surface_layout="single-top",
    )
    expected = apply_spatial_filter(frames, options=opts)
    actual = apply_spatial_filter_streaming(sequence, options=opts)
    try:
        assert [len(f) for f in actual.frames] == [len(f) for f in expected.frames]
    finally:
        close = getattr(actual.frames, "close", None)
        if close is not None:
            close()


def test_nearest_density_persistence_and_gui_preview(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from linak import cli

    profiles = density.compute_all_density_profiles(
        [slab()],
        species="K",
        surface_options=options(),
        surface_elements=["Pt"],
        surface_layout="single-top",
        bin_width=0.5,
        grid_bin_width=0.5,
    )
    path = density.save_density_profiles(profiles, tmp_path / "nearest.density.h5")
    grid = density.load_density_grid_profiles(path)[0]
    assert grid.surface_reference.distance_mode == "nearest-atom"
    assert grid.surface_reference._nearest_provider is None
    projected = density.density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="distance")
    assert projected.units == "amu/Angstrom^2"
    heatmap_path = density.save_density_profiles([projected], tmp_path / "projected.density.h5")
    assert density.load_density_heatmap_profiles(heatmap_path)[0].distance_mode == "nearest-atom"
    heatmap_options = cli._build_density_profile_filter_options(
        [(str(heatmap_path), [density._density_heatmap_profile_hdf5_payload(projected)])],
        axis=None,
        species=None,
    )
    assert heatmap_options["surface_distance_mode"] == "nearest-atom"
    checked = []

    def gui(**kwargs):
        settings = kwargs["initial_settings"]
        assert settings["_profile_filter_options"]["surface_distance_mode"] == "nearest-atom"
        state = kwargs["on_preview_figure"](settings)
        try:
            assert (
                "Angstrom" in state["axes"].get_ylabel() or "\u00c5" in state["axes"].get_ylabel()
            )
            assert "cm" not in state["axes"].get_ylabel()
            checked.append(True)
        finally:
            plt.close(state["figure"])

    monkeypatch.setattr(cli, "_open_plot_settings_gui", gui)
    assert main(["--log-level", "ERROR", "plot", str(path), "--gui"]) == 0
    assert checked


def test_explicit_candidate_indices_and_mask():
    frame = slab()
    for opts in (
        SurfaceEstimatorOptions(distance_mode="nearest-atom", surface_atom_indices=(0, 1)),
        SurfaceEstimatorOptions(
            distance_mode="nearest-atom",
            surface_atom_mask=np.array([True, True, False, False, False, False, False, False]),
        ),
    ):
        profile = position.compute_position_profile(
            [frame], "K", surface_options=opts, surface_layout="single-top"
        )
        assert profile.distance_to_surface[0, 0] == pytest.approx(np.sqrt(26))


def test_shell_volumes_never_used_for_nearest_distance(monkeypatch):
    frame = slab()
    profiles = density.compute_all_density_profiles(
        [frame],
        species="K",
        surface_options=options(),
        surface_elements=["Pt"],
        surface_layout="single-top",
        bin_width=0.5,
        grid_bin_width=0.5,
    )
    grid = next(p for p in profiles if isinstance(p, density.DensityGridProfile))
    monkeypatch.setattr(
        density,
        "_geometry_project_density",
        lambda *a, **k: pytest.fail("planar shell volumes used"),
    )
    heat = density.density_grids_to_averaged_heatmap_profile(
        [grid, grid], x_axis="x", y_axis="distance"
    )
    assert heat.units == "amu/Angstrom^2"
    assert np.sum(heat.number_density) * 0.25 == pytest.approx(1.0)


def test_bounded_image_fallback_matches_replicated_tree(monkeypatch):
    from linak.analysis import surface

    cell = np.array([[9.0, 0, 0], [8.5, 2.0, 0], [8.0, 1.8, 3.0]])
    frame = Atoms(cell=cell, pbc=True)
    rng = np.random.default_rng(712)
    candidates = rng.random((29, 3)) @ cell
    points = rng.random((1041, 3)) @ cell
    expected = NearestSubstrateSearch(frame, candidates).query(points)
    monkeypatch.setattr(surface, "_NEAREST_IMAGE_BUDGET_BYTES", 1)
    search = NearestSubstrateSearch(frame, candidates)
    assert search.query_translations is not None
    assert search.tree.n == len(candidates)
    np.testing.assert_allclose(search.query(points), expected, atol=1e-12)


@pytest.mark.parametrize(
    "command,extra",
    [
        ("density", ["--species", "K"]),
        ("position", ["--species", "K"]),
        ("cn", ["--species", "K", "O", "--cutoff", "3.5"]),
        ("orientation", []),
    ],
)
def test_cli_nearest_direct_xyz(command, extra, tmp_path):
    from ase.io import write

    source = tmp_path / "direct.xyz"
    write(source, slab())
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "compute",
                command,
                str(source),
                "--surface-distance-mode",
                "nearest-atom",
                "--surface-elements",
                "Pt",
                "--surface-layout",
                "single-top",
                *extra,
            ]
        )
        == 0
    )
    assert source.read_text()
    assert not list(source.parent.glob("*.traj.h5"))


def test_rebinding_revalidates_filtered_candidates():
    from linak.analysis.surface import bind_surface_reference

    original = slab()
    reference = estimate_surface_layout_reference(
        [original], surface_options=options(), surface_elements=["Pt"], surface_layout="single-top"
    )
    filtered = original[[0, 1, 4]]
    rebound = bind_surface_reference(reference, [filtered])
    distance = surface_distance_components(
        rebound, filtered.positions[:, 2], frame_index=0, target_positions=filtered.positions
    )[0]
    assert distance[-1] == pytest.approx(np.sqrt(26))
    assert bind_surface_reference(rebound, rebound._nearest_provider.frames) is rebound


def test_explicit_plane_and_nearest_density_cannot_mix():
    nearest = density.compute_density_profile(
        [slab()],
        "K",
        surface_options=options(),
        surface_elements=["Pt"],
        surface_layout="single-top",
    )
    plane = density.compute_density_profile(
        [slab()], "K", surface_elements=["Pt"], surface_layout="single-top"
    )
    with pytest.raises(ValueError, match="Cannot mix"):
        density.plot_density_profiles([nearest, plane], show=False)


def test_studio_shows_nearest_definition_offscreen(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib.pyplot as plt

    del plt
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication([])
    if os.name == "nt" and not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    profiles = density.compute_all_density_profiles(
        [slab()],
        species="K",
        surface_options=options(),
        surface_elements=["Pt"],
        surface_layout="single-top",
        bin_width=0.5,
        grid_bin_width=0.5,
    )
    path = density.save_density_profiles(profiles, tmp_path / "studio.density.h5")
    checked = []

    def inspect(application):
        application.processEvents()
        window = next(
            w
            for w in application.topLevelWidgets()
            if hasattr(w, "_series_enabled_data") and w.isVisible()
        )
        try:
            labels = [label.text() for label in window.findChildren(QLabel)]
            assert any("Nearest surface-atom distance" in text for text in labels)
            assert any("not volumetric densities" in text for text in labels)
            checked.append(True)
        finally:
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect)
    assert main(["--log-level", "ERROR", "plot", str(path), "--gui"]) == 0
    assert checked
    app.processEvents()


def test_nearest_fixed_masks_survive_variable_topology_and_filtered_conversion(tmp_path):
    from linak.trajectory.io import TrajectoryStoredMetadata, write_trajectory

    frame = slab()
    frame.set_constraint(FixAtoms(indices=[2, 3]))
    second = frame[list(range(7))]
    metadata = TrajectoryStoredMetadata(
        spatial_filter_metadata={"used": True},
        surface_distance_settings={
            "distance_mode": "nearest-atom",
            "surface_elements": ["Pt"],
            "surface_layout": "single-top",
            "include_fixed_surface_atoms": False,
        },
    )
    source = write_trajectory([frame, second], tmp_path / "fixed.traj.h5", metadata=metadata)
    restored = read_trajectory(source)
    for current in restored:
        np.testing.assert_array_equal(current.constraints[0].get_indices(), [2, 3])
    target = tmp_path / "filtered.traj.h5"
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(source),
                "--output",
                str(target),
                "--x-range",
                "0:1.5",
            ]
        )
        == 0
    )
    filtered = read_trajectory(target)
    for current in filtered:
        np.testing.assert_array_equal(current.constraints[0].get_indices(), [1])
    profile = position.compute_position_profile(
        filtered,
        "K",
        surface_elements=["Pt"],
        surface_layout="single-top",
        surface_options=options(),
    )
    np.testing.assert_allclose(profile.distance_to_surface, np.sqrt(26), atol=1e-6)
    with h5py.File(target) as handle:
        assert handle["fixed_atom_mask"].compression is not None


def test_conversion_cannot_replace_removed_substrate_with_fluid(tmp_path, capsys):
    from ase.io import write

    source = tmp_path / "original.xyz"
    write(source, slab())
    target = tmp_path / "fluid.traj.h5"
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(source),
                "--output",
                str(target),
                "--surface-distance-mode",
                "nearest-atom",
                "--distance-range",
                "3:5",
            ]
        )
        == 1
    )
    assert not target.exists()
    assert "Pt" in capsys.readouterr().err
