"""Registration moves orientation samples without changing their angle convention."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from linak.analysis.interface_alignment import InterfaceAlignment, apply_interface_alignment
from linak.analysis.orientation import (
    _orientation_profile_for_interface_view,
    load_orientation_profile,
    orientation_sparse_grid_to_line,
    plot_orientation_profile,
    save_orientation_profile,
)
from linak.analysis.orientation_spatial import (
    BOTTOM_MOMENT_CHANNELS,
    aligned_orientation_profile,
    spatial_orientation_map,
)
from tests.test_orientation_symmetric import _profile, _unbalanced_frame


def _frame(length=12.0, origin=0.0):
    frame = _unbalanced_frame()
    frame.set_cell([length, 12.0, 20.0], scale_atoms=True)
    for i, x in enumerate((7.5, 7.5, 1.5, 10.5)):
        start = 12 + 3 * i
        oxygen = np.array([x, 4.5, frame.positions[start, 2]])
        frame.positions[start : start + 3] = [
            oxygen,
            oxygen + [0.0, 0.5, 0.65],
            oxygen + [0.0, -0.5, 0.65],
        ]
    frame.set_celldisp([origin, 0.0, 0.0])
    frame.positions[:, 0] += origin
    return frame


def _setting(profile, matrix=((1, 0), (0, 1)), translation=(0.25, 0.0)):
    alignment = InterfaceAlignment(
        profile.interface_reference.reference_id,
        matrix=matrix,
        translation=translation,
        allow_mismatches=True,
    )
    return {alignment.reference_id: alignment.to_dict()}


def _dense(grid, name):
    result = np.zeros(grid.shape)
    result.ravel()[grid.flat_indices.astype(int)] = getattr(grid, name)
    return result


@pytest.mark.parametrize(
    "matrix,translation",
    [
        (((-1, 0), (0, 1)), (0.0, 0.0)),
        (((0, 1), (-1, 0)), (0.0, 0.0)),
        (((1, 0), (0, 1)), (0.25, 0.5)),
    ],
)
def test_exact_registration_conserves_moments_and_matches_independent_density(matrix, translation):
    from linak.analysis.density import (
        compute_all_density_profiles,
        DensityGridProfile,
        density_grid_to_heatmap_profile,
    )

    frame = _frame()
    profile = _profile([frame])
    settings = _setting(profile, matrix, translation)
    aligned = aligned_orientation_profile(profile, ("x", "y"), interface_alignments=settings)
    original = profile.sparse_grid
    moved = aligned.sparse_grid
    indices = np.asarray(np.unravel_index(original.flat_indices, original.shape)).T
    centers = [
        0.5 * (e[:-1] + e[1:]) for e in (original.x_edges, original.y_edges, original.z_edges)
    ]
    positions = np.column_stack([centers[i][indices[:, i]] for i in range(3)])
    transform = InterfaceAlignment.from_dict(next(iter(settings.values())))
    positions = apply_interface_alignment(
        positions, frame.cell, frame.get_celldisp().ravel(), "z", transform
    )
    target = indices.copy()
    for axis, edges in enumerate((moved.x_edges, moved.y_edges)):
        target[:, axis] = np.searchsorted(edges, positions[:, axis], side="right") - 1
    target_flat = np.ravel_multi_index(target.T, moved.shape)
    for name in (*BOTTOM_MOMENT_CHANNELS, "bottom_entity_sum"):
        expected = np.bincount(
            target_flat, weights=getattr(original, name), minlength=np.prod(moved.shape)
        ).reshape(moved.shape)
        np.testing.assert_allclose(_dense(moved, name), expected, atol=1e-12)
        assert getattr(moved, name).sum() == pytest.approx(getattr(original, name).sum())
    for name in ("top_entity_sum", "top_cos_polar_sum", "top_cos_azimuthal_sum"):
        np.testing.assert_array_equal(_dense(moved, name), _dense(original, name))
    assert profile.interface_alignment_metadata is None

    # Independent density analysis of the transformed water coordinates.
    transformed = frame.copy()
    transformed.positions[12:15] = apply_interface_alignment(
        transformed.positions[12:15], frame.cell, frame.get_celldisp().ravel(), "z", transform
    )
    densities = compute_all_density_profiles(
        [transformed],
        species="H2O",
        surface_axis="z",
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
        bin_width=3.0,
        grid_bin_width=3.0,
        binning="cell",
        outputs="all",
        precomputed_surface_estimate=profile.surface_reference,
    )
    grid = next(p for p in densities if isinstance(p, DensityGridProfile))
    assert grid.interface_reference.reference_id == profile.interface_reference.reference_id
    expected = density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="y")
    density = spatial_orientation_map([profile], quantity="density", interface_alignments=settings)
    np.testing.assert_allclose(
        density.values[:4, :4], expected.number_density / 1000.0, equal_nan=True, rtol=1e-6
    )
    assert np.all(np.isnan(density.values[4:, :]))
    assert np.all(np.isnan(density.values[:, 4:]))
    for angle in ("polar", "azimuthal"):
        mean = spatial_orientation_map([profile], angle=angle, interface_alignments=settings)
        weighted = spatial_orientation_map(
            [profile], angle=angle, quantity="density-weighted", interface_alignments=settings
        )
        np.testing.assert_allclose(weighted.values, mean.values * density.values, equal_nan=True)
    # Azimuthal cosines retain +1 even under a reflection; directions are not rotated.
    azimuthal = spatial_orientation_map([profile], angle="azimuthal", interface_alignments=settings)
    np.testing.assert_allclose(azimuthal.values[np.isfinite(azimuthal.values)], 1.0)


def test_fractional_shift_splits_counts_and_signed_moments_before_filters():
    profile = _profile([_frame()])
    settings = _setting(profile, translation=(0.125, 0.0))  # 1.5 A: half a source bin.
    aligned = aligned_orientation_profile(profile, ("x", "y"), interface_alignments=settings)
    for name in (*BOTTOM_MOMENT_CHANNELS, "bottom_entity_sum"):
        original = _dense(profile.sparse_grid, name)
        expected = 0.5 * original + 0.5 * np.roll(original, 1, axis=0)
        np.testing.assert_allclose(_dense(aligned.sparse_grid, name), expected, atol=1e-12)
    for angle in ("polar", "azimuthal"):
        for quantity in ("average", "density", "density-weighted"):
            options = dict(angle=angle, quantity=quantity, filters={"x": (9.0, 12.0)})
            actual = spatial_orientation_map([profile], **options, interface_alignments=settings)
            expected = spatial_orientation_map([aligned], **options)
            np.testing.assert_allclose(actual.values, expected.values, equal_nan=True)
    _, _, line = orientation_sparse_grid_to_line(
        profile, x_axis="distance", filters={"x": (9.0, 12.0)}, interface_alignments=settings
    )
    _, _, expected_line = orientation_sparse_grid_to_line(
        aligned, x_axis="distance", filters={"x": (9.0, 12.0)}
    )
    np.testing.assert_allclose(line, expected_line, equal_nan=True)


def test_changing_cells_and_origins_use_each_geometry_moments():
    profile = _profile([_frame(), _frame(18.0, 3.0)])
    settings = _setting(profile, translation=(0.5, 0.0))
    transformed = aligned_orientation_profile(profile, ("x", "y"), interface_alignments=settings)
    grid = profile.sparse_grid
    edges = [grid.x_edges, grid.y_edges, grid.z_edges]
    expected = np.zeros(grid.shape)
    alignment = InterfaceAlignment.from_dict(next(iter(settings.values())))
    for group in grid.density_geometry_groups:
        indices = np.asarray(np.unravel_index(group["flat_indices"].astype(int), grid.shape)).T
        points = np.column_stack(
            [0.5 * (edge[indices[:, i]] + edge[indices[:, i] + 1]) for i, edge in enumerate(edges)]
        )
        points = apply_interface_alignment(points, group["cell"], group["origin"], "z", alignment)
        for i in (0, 1):
            indices[:, i] = np.searchsorted(edges[i], points[:, i], side="right") - 1
        np.add.at(expected, tuple(indices.T), group["bottom_cos_polar_sum"])
    np.testing.assert_allclose(_dense(transformed.sparse_grid, "bottom_cos_polar_sum"), expected)
    assert len(profile.sparse_grid.density_geometry_groups) == 2


@pytest.mark.parametrize("view", ["top", "full"])
def test_positive_and_full_cell_views_stay_fixed(view):
    profile = _profile([_frame()])
    selected = _orientation_profile_for_interface_view(profile, view)
    for quantity in ("average", "density", "density-weighted"):
        expected = spatial_orientation_map([selected], quantity=quantity)
        actual = spatial_orientation_map(
            [selected], quantity=quantity, interface_alignments=_setting(profile)
        )
        np.testing.assert_allclose(actual.values, expected.values, equal_nan=True)
        assert not actual.alignments


def test_normal_profiles_and_distributions_are_unaffected():
    profile = _profile([_frame()])
    for options in ({}, {"component": "heatmap", "orientation_heatmap_kind": "distribution"}):
        states = []
        for settings in (None, _setting(profile)):
            state = {}
            plot_orientation_profile(
                profile, show=False, interface_alignments=settings, capture_state=state, **options
            )
            states.append(state)
        if options:
            np.testing.assert_allclose(
                states[0]["plotted_heatmap"]["values"], states[1]["plotted_heatmap"]["values"]
            )
        else:
            assert states[0]["plotted_xy_series"] == states[1]["plotted_xy_series"]


def test_round_trip_lazy_reference_and_legacy_gating(tmp_path):
    from ase.constraints import FixAtoms

    frame = _frame()
    frame.set_constraint(FixAtoms(indices=range(12)))
    from linak.analysis.orientation import compute_orientation_profile

    profile = compute_orientation_profile(
        [frame],
        surface_layout="symmetric",
        surface_mode="layered",
        surface_elements=["Pt"],
        bin_width=3.0,
        include_fixed_surface_atoms=True,
    )
    assert len(profile.interface_reference.snapshots[0].atom_indices) == 12
    path = tmp_path / "new.orientation.h5"
    save_orientation_profile(profile, path)
    loaded = load_orientation_profile(path)
    assert not loaded.interface_reference.snapshots
    assert (
        loaded.interface_reference.materialize().reference_id
        == profile.interface_reference.reference_id
    )
    settings = _setting(profile)
    actual = spatial_orientation_map([loaded], interface_alignments=settings)
    np.testing.assert_allclose(
        actual.values,
        spatial_orientation_map([profile], interface_alignments=settings).values,
        equal_nan=True,
    )
    assert loaded.interface_alignment_metadata is None
    legacy_grid = replace(
        profile.sparse_grid,
        density_geometry_groups=[
            {k: v for k, v in g.items() if k not in BOTTOM_MOMENT_CHANNELS}
            for g in profile.sparse_grid.density_geometry_groups
        ],
    )
    legacy = replace(profile, sparse_grid=legacy_grid)
    assert np.any(np.isfinite(spatial_orientation_map([legacy]).values))
    with pytest.raises(ValueError, match="Recompute this orientation"):
        spatial_orientation_map([legacy], interface_alignments=settings)
    from linak.cli import _interface_reference_options, read_analysis_profile_headers_by_source

    legacy_path = tmp_path / "old.orientation.h5"
    save_orientation_profile(legacy, legacy_path)
    options = _interface_reference_options(
        read_analysis_profile_headers_by_source(sources=[str(legacy_path)], analysis="orientation"),
        "orientation",
    )
    assert not options[0]["can_align"]
    assert "Recompute this orientation" in options[0]["unavailable_reason"]


def test_cli_gui_alignment_persistence_and_export(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from linak.cli import main
    from linak.plot.mappings.orientation_mapping import orientation_plot_options_to_view_mapping
    from linak.plot.profile_persistence import serialize_plot_view_mapping

    profile = _profile([_frame()])
    path = tmp_path / "new.orientation.h5"
    save_orientation_profile(profile, path)
    settings = _setting(profile)
    checked = []

    def gui(**kwargs):
        initial = deepcopy(kwargs["initial_settings"])
        references = initial["_profile_filter_options"]["interface_references"]
        assert references[0]["can_align"]
        initial.update(
            interface_alignments=settings,
            orientation_heatmap_kind="spatial",
            view_mapping=serialize_plot_view_mapping(
                orientation_plot_options_to_view_mapping(orientation_heatmap_kind="spatial")
            ),
            plot_data_include_metadata=True,
        )
        preview = kwargs["on_preview_figure"](initial)
        expected = spatial_orientation_map([profile], interface_alignments=settings)
        np.testing.assert_allclose(
            preview["plotted_heatmap"]["values"], expected.values, equal_nan=True
        )
        assert preview["interface_alignments"][profile.interface_reference.reference_id][
            "angle_convention"
        ]
        plt.close(preview["figure"])
        _, exported = kwargs["on_save_data"](initial, str(tmp_path / "aligned.csv"))
        np.testing.assert_allclose(
            exported["plotted_heatmap"]["values"], expected.values, equal_nan=True
        )
        text = (tmp_path / "aligned.csv").read_text(encoding="utf-8")
        assert "interface_alignments=" in text and "source_grid_resolution_A" in text
        kwargs["on_save"]("Aligned", initial)
        restored = kwargs["on_load_profile"]("Aligned")
        assert restored["interface_alignments"] == settings
        checked.append(True)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", gui)
    assert main(["--log-level", "ERROR", "plot", str(path), "--gui"]) == 0
    assert checked == [True]


def test_preview_defaults_do_not_replace_saved_alignment_or_frequency_mode():
    from linak.cli import _merge_preview_defaults_into_gui_settings

    settings = {
        "interface_alignments": {"reference": {"saved": True}},
        "heatmap_value_mode": "raw_counts",
    }
    preview = {
        "orientation_grid_heatmap": True,
        "interface_alignments": {},
        "heatmap_value_mode": "mean_orientation",
    }
    merged = _merge_preview_defaults_into_gui_settings(settings, preview)
    assert merged["interface_alignments"] == settings["interface_alignments"]
    assert merged["heatmap_value_mode"] == "raw_counts"


@pytest.mark.parametrize("legacy", [False, True])
def test_studio_alignment_apply_cancel_and_view_switches(tmp_path, monkeypatch, legacy):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDialog, QMainWindow
    from PySide6.QtGui import QFontDatabase
    from linak.plot.plot_gui import launch_plot_settings_panel
    from linak.plot.mappings.orientation_mapping import orientation_plot_options_to_view_mapping
    from linak.plot.profile_persistence import serialize_plot_view_mapping
    from linak.cli import _interface_reference_options, read_analysis_profile_headers_by_source

    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        from matplotlib.font_manager import findfont

        QFontDatabase.addApplicationFont(findfont("DejaVu Sans"))
    profile = _profile([_frame()])
    path = tmp_path / "studio.orientation.h5"
    save_orientation_profile(profile, path)
    headers = read_analysis_profile_headers_by_source(sources=[str(path)], analysis="orientation")
    references = [] if legacy else _interface_reference_options(headers, "orientation")
    accepted = []

    class Dialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, items, settings, parent):
            assert not items[0]["reference"].snapshots
            self.result_alignments = _setting(profile)

        def exec(self):
            return self.DialogCode.Accepted if accepted else self.DialogCode.Rejected

        def deleteLater(self):
            pass

    monkeypatch.setattr("linak.plot.interface_alignment_dialog.InterfaceAlignmentDialog", Dialog)
    checked = []

    def inspect(window):
        window._preview_timer.stop()
        monkeypatch.setattr(window, "_schedule_preview_update", lambda *a, **k: None)
        assert window.interface_alignment_button.isEnabled() == (not legacy)
        if legacy:
            assert "Recompute this orientation file" in window.interface_alignment_note.text()
        else:
            window._open_interface_alignment()
            assert not window._collect_settings()["interface_alignments"]
            accepted.append(True)
            window.orientation_heatmap_y_axis.setCurrentText("Z")
            window._open_interface_alignment()
            assert window.interface_alignment_status.text().startswith("Applied")
            assert window.orientation_heatmap_y_axis.currentText() == "Distance"
            assert "angle definitions are preserved" in window.interface_alignment_note.text()
            settings = window._collect_settings()["interface_alignments"]
            window.orientation_heatmap_kind.setCurrentText("Orientation distribution")
            assert window._collect_settings()["interface_alignments"] == settings
            window.orientation_heatmap_kind.setCurrentText("Spatial map")
            for view in ("full", "folded", "bottom"):
                window._set_combo_value(
                    window.interface_view_combo, window._interface_view_labels()[view]
                )
                assert window._collect_settings()["interface_alignments"] == settings
        checked.append(True)
        window._cleanup_preview_artifacts()

    monkeypatch.setattr(QMainWindow, "showMaximized", inspect)
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    launch_plot_settings_panel(
        title=f"Orientation registration {legacy}",
        analysis_name="orientation",
        initial_settings={
            "view_mapping": serialize_plot_view_mapping(
                orientation_plot_options_to_view_mapping(orientation_heatmap_kind="spatial")
            ),
            "_profile_filter_options": {
                "surface_layout_resolved": "symmetric",
                "surface_axis": "z",
                "surface_view_capabilities": ["full", "folded", "top", "bottom", "all"],
                "interface_references": references,
                "orientation_sparse_grid_available": True,
                "orientation_spatial_density_available": True,
                "orientation_axis_ranges": {
                    "x": [0.0, 12.0],
                    "y": [0.0, 12.0],
                    "z": [0.0, 21.0],
                    "distance": [3.0, 6.0],
                },
            },
        },
        on_preview=lambda settings: {},
        on_save=lambda name, settings: "Saved",
    )
    assert checked == [True]
    app.processEvents()


def test_individual_negative_interface_preserves_angles_after_alignment():
    profile = _profile([_frame()])
    bottom = _orientation_profile_for_interface_view(profile, "bottom")
    settings = _setting(profile)
    moved = spatial_orientation_map([bottom], interface_alignments=settings)
    assert moved.alignments
    np.testing.assert_allclose(moved.values[np.isfinite(moved.values)], -1.0)
    azimuthal = spatial_orientation_map([bottom], angle="azimuthal", interface_alignments=settings)
    np.testing.assert_allclose(azimuthal.values[np.isfinite(azimuthal.values)], 1.0)
    xy = spatial_orientation_map([bottom], quantity="density", interface_alignments=settings)
    yx = spatial_orientation_map(
        [bottom], x_axis="y", y_axis="x", quantity="density", interface_alignments=settings
    )
    np.testing.assert_allclose(xy.values, yx.values.T, equal_nan=True)


def test_reference_validation_and_mismatch_acknowledgement_are_enforced():
    profile = _profile([_frame()])
    settings = _setting(profile)
    key = profile.interface_reference.reference_id
    settings[key]["reference_id"] = "a" * 64
    with pytest.raises(ValueError, match="reference|identity"):
        spatial_orientation_map([profile], interface_alignments=settings)
    settings = _setting(profile, translation=(0.123, 0.171))
    settings[key]["allow_mismatches"] = False
    with pytest.raises(ValueError, match="acknowledge"):
        spatial_orientation_map([profile], interface_alignments=settings)


@pytest.mark.parametrize("tilted", [False, True])
def test_stepped_substrate_suggestion_can_register_orientation_with_nonzero_origin(tilted):
    from ase import Atoms
    from tests.test_interface_alignment import stepped_reference
    from linak.analysis.interface_alignment import suggest_interface_alignment

    reference = stepped_reference(
        matrix=((-1, 0), (0, -1)), shift=(0.31, 0.27), tilted=tilted, origin=(3.0, -4.0, 2.0)
    )
    snapshot = reference.snapshots[0]
    positions, symbols = list(snapshot.positions), ["Pt"] * 12
    for uv, z in [((0.45, 0.4), 2.0), ((0.45, 0.4), 18.0), ((0.2, 0.65), 18.0)]:
        oxygen = np.r_[np.asarray(uv) @ snapshot.cell[:2, :2], z] + snapshot.origin
        positions.extend([oxygen, oxygen + [0.0, 0.5, 0.65], oxygen + [0.0, -0.5, 0.65]])
        symbols.extend(["O", "H", "H"])
    frame = Atoms(
        symbols, positions=positions, cell=snapshot.cell, pbc=True, celldisp=snapshot.origin
    )
    profile = _profile([frame])
    before = profile.sparse_grid.bottom_count_polar_valid.copy()
    suggestions = suggest_interface_alignment(profile.interface_reference)
    assert suggestions[0].diagnostics["good_match"]
    assert suggestions[0].matrix == ((-1, 0), (0, -1))
    np.testing.assert_allclose(suggestions[0].translation, [0.31, 0.27], atol=1e-6)
    assert profile.interface_alignment_metadata is None
    np.testing.assert_array_equal(profile.sparse_grid.bottom_count_polar_valid, before)
    settings = {profile.interface_reference.reference_id: suggestions[0].to_dict()}
    aligned = aligned_orientation_profile(profile, ("x", "distance"), interface_alignments=settings)
    assert aligned.interface_alignment_metadata["diagnostics"]["good_match"]
    assert aligned.sparse_grid.bottom_count_polar_valid.sum() == pytest.approx(before.sum())
    result = spatial_orientation_map(
        [profile], x_axis="x", y_axis="distance", interface_alignments=settings
    )
    assert result.alignments
    assert np.any(np.isfinite(result.values))


# Callback tests receive a prepared session; startup itself is tested separately.
pytestmark = pytest.mark.usefixtures("prepared_studio")
