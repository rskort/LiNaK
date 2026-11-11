"""Scientific and Studio contracts for spatial orientation color quantities."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from linak.analysis.orientation import (
    _orientation_profile_for_interface_view,
    load_orientation_profile,
    save_orientation_profile,
    plot_orientation_profile,
)
from linak.analysis.orientation_spatial import density_capability, spatial_orientation_map
from tests.test_orientation_symmetric import _profile, _unbalanced_frame


def _constant_orientation(profile, value, angle="polar"):
    grid = profile.sparse_grid
    fields = {f"cos_{angle}_sum": value * getattr(grid, f"count_{angle}_valid")}
    for side in ("top", "bottom"):
        fields[f"{side}_cos_{angle}_sum"] = value * getattr(grid, f"{side}_count_{angle}_valid")
    return replace(profile, sparse_grid=replace(grid, **fields))


@pytest.mark.parametrize("angle", ["polar", "azimuthal"])
@pytest.mark.parametrize("value", [0.25, -0.5])
def test_weighted_map_is_mean_times_density_after_rebinning(angle, value):
    profile = _constant_orientation(_profile([_unbalanced_frame()]), value, angle)
    settings = dict(x_axis="x", y_axis="y", angle=angle, x_bin_width=6.0, y_bin_width=6.0)
    mean = spatial_orientation_map([profile], quantity="average", **settings)
    density = spatial_orientation_map([profile], quantity="density", **settings)
    weighted = spatial_orientation_map([profile], quantity="density-weighted", **settings)
    np.testing.assert_allclose(mean.values[np.isfinite(mean.values)], value)
    np.testing.assert_allclose(weighted.values, mean.values * density.values, equal_nan=True)
    assert weighted.units == density.units == "molecules/Angstrom^3"
    assert np.any(density.values == 0.0)
    assert np.all(np.isnan(weighted.values[density.values == 0.0]))


@pytest.mark.parametrize(
    "quantity,field",
    [
        ("average", "cos_polar_mean"),
        ("density", "density"),
        ("density-weighted", "cos_polar_density"),
    ],
)
def test_complete_lateral_projection_matches_one_dimensional_result(quantity, field):
    profile = _profile([_unbalanced_frame()])
    result = spatial_orientation_map(
        [profile], x_axis="distance", y_axis="y", y_bin_width=12.0, quantity=quantity
    )
    np.testing.assert_allclose(
        result.values[:, 0], getattr(profile, field), rtol=1e-6, atol=1e-8, equal_nan=True
    )


@pytest.mark.parametrize("tilted", [False, True])
def test_density_matches_independent_water_density_and_counts_empty_frames(tilted):
    from linak.analysis.density import (
        DensityGridProfile,
        compute_all_density_profiles,
        density_grid_to_heatmap_profile,
    )

    populated = _unbalanced_frame(tilted=tilted)
    empty = populated[:12]
    profile = _profile([populated, empty])
    result = spatial_orientation_map([profile], quantity="density")
    populated_only = spatial_orientation_map([_profile([populated])], quantity="density")
    np.testing.assert_allclose(result.values, 0.5 * populated_only.values, equal_nan=True)
    profiles = compute_all_density_profiles(
        [populated, empty],
        species="H2O",
        surface_axis="z",
        surface_layout="symmetric",
        surface_elements=["Pt"],
        surface_mode="layered",
        bin_width=3.0,
        grid_bin_width=3.0,
        binning="cell",
        outputs="all",
        precomputed_surface_estimate=profile.surface_reference,
    )
    grid = next(p for p in profiles if isinstance(p, DensityGridProfile))
    independent = density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="y")
    np.testing.assert_allclose(result.x_edges, independent.x_bin_edges)
    np.testing.assert_allclose(result.y_edges, independent.y_bin_edges)
    np.testing.assert_allclose(
        result.values, independent.number_density / 1000.0, rtol=1e-6, atol=1e-9, equal_nan=True
    )


def test_variable_cells_and_nonzero_origins_use_each_geometry():
    first = _unbalanced_frame()
    second = _unbalanced_frame()
    second.set_cell([20, 10, 20], scale_atoms=True)
    second.set_celldisp([2, -3, 4])
    second.positions += [2, -3, 4]
    profile = _profile([first, second])
    result = spatial_orientation_map(
        [profile], quantity="density", x_bin_width=30.0, y_bin_width=30.0
    )
    assert len(profile.sparse_grid.density_geometry_groups) == 2
    # Four waters in cells of volume 2000 and 4000 Å³; normalize each frame first.
    assert result.values[0, 0] == pytest.approx(0.5 * (4 / 2000 + 4 / 4000), rel=1e-6)


def test_source_pooling_uses_raw_moments_and_frame_weighted_density():
    first = _constant_orientation(_profile([_unbalanced_frame()] * 2), 0.25)
    second = _constant_orientation(_profile([_unbalanced_frame()] * 4), -0.5)
    options = dict(x_axis="x", y_axis="y", x_bin_width=12.0, y_bin_width=12.0)
    mean = spatial_orientation_map([first, second], quantity="average", **options)
    density = spatial_orientation_map([first, second], quantity="density", **options)
    weighted = spatial_orientation_map([first, second], quantity="density-weighted", **options)
    np.testing.assert_allclose(mean.values, -0.25)
    np.testing.assert_allclose(weighted.values, mean.values * density.values)


@pytest.mark.parametrize("view,expected", [("top", 1.0), ("bottom", -1.0), ("full", 1.0)])
def test_interface_and_laboratory_conventions_are_preserved(view, expected):
    profile = _orientation_profile_for_interface_view(_profile([_unbalanced_frame()]), view)
    result = spatial_orientation_map([profile], quantity="density-weighted")
    density = spatial_orientation_map([profile], quantity="density")
    valid = np.isfinite(result.values)
    np.testing.assert_allclose(result.values[valid], expected * density.values[valid], rtol=1e-6)


def test_invalid_orientation_does_not_remove_water_density():
    profile = _profile([_unbalanced_frame()])
    grid = profile.sparse_grid
    invalid = replace(
        grid,
        cos_polar_sum=np.zeros_like(grid.cos_polar_sum),
        count_polar_valid=np.zeros_like(grid.count_polar_valid),
        top_cos_polar_sum=np.zeros_like(grid.top_cos_polar_sum),
        bottom_cos_polar_sum=np.zeros_like(grid.bottom_cos_polar_sum),
        top_count_polar_valid=np.zeros_like(grid.top_count_polar_valid),
        bottom_count_polar_valid=np.zeros_like(grid.bottom_count_polar_valid),
    )
    profile = replace(profile, sparse_grid=invalid)
    density = spatial_orientation_map([profile], quantity="density")
    weighted = spatial_orientation_map([profile], quantity="density-weighted")
    assert np.nanmax(density.values) > 0.0
    assert np.all(np.isnan(weighted.values))


@pytest.mark.parametrize("angle", ["polar", "azimuthal"])
def test_axis_reversal_and_filters_apply_to_all_quantities(angle):
    profile = _profile([_unbalanced_frame()])
    for quantity in ("average", "density", "density-weighted"):
        xy = spatial_orientation_map(
            [profile], quantity=quantity, angle=angle, filters={"x": (0.0, 4.0)}
        )
        yx = spatial_orientation_map(
            [profile],
            x_axis="y",
            y_axis="x",
            quantity=quantity,
            angle=angle,
            filters={"x": (0.0, 4.0)},
        )
        np.testing.assert_allclose(xy.values, yx.values.T, equal_nan=True)


def test_round_trip_and_legacy_mean_capability(tmp_path):
    profile = _profile([_unbalanced_frame()])
    path = tmp_path / "new.orientation.h5"
    save_orientation_profile(profile, path)
    loaded = load_orientation_profile(path)
    assert density_capability(loaded) == (True, "")
    for quantity in ("average", "density", "density-weighted"):
        np.testing.assert_allclose(
            spatial_orientation_map([loaded], quantity=quantity).values,
            spatial_orientation_map([profile], quantity=quantity).values,
            equal_nan=True,
        )
    legacy = replace(
        profile,
        sparse_grid=replace(
            profile.sparse_grid, density_geometry_groups=None, density_geometry_version=None
        ),
    )
    legacy_path = tmp_path / "legacy.orientation.h5"
    save_orientation_profile(legacy, legacy_path)
    legacy = load_orientation_profile(legacy_path)
    assert np.any(np.isfinite(spatial_orientation_map([legacy]).values))
    with pytest.raises(ValueError, match="Recompute orientation"):
        spatial_orientation_map([legacy], quantity="density")


def test_cli_mapping_persistence_and_export(tmp_path):
    from linak.cli import (
        build_plot_parser,
        _resolve_orientation_plotter_kwargs,
        _write_plotted_data_export,
    )
    from linak.plot.profile_persistence import (
        build_plot_profile_payload,
        flatten_plot_profile_payload,
    )
    from linak.plot.mappings.orientation_mapping import orientation_view_mapping_to_plot_options

    args = build_plot_parser(analysis="orientation").parse_args(
        [
            "new.orientation.h5",
            "--heatmap-kind",
            "spatial",
            "--heatmap-quantity",
            "density-weighted",
            "--heatmap-x-axis",
            "x",
            "--heatmap-y-axis",
            "distance",
        ]
    )
    mapping = _resolve_orientation_plotter_kwargs(args)["view_mapping"]
    assert (
        orientation_view_mapping_to_plot_options(mapping)["orientation_heatmap_quantity"]
        == "density-weighted"
    )
    payload = build_plot_profile_payload(
        "plot:orientation",
        {
            "view_mapping": mapping,
            "orientation_heatmap_kind": "spatial",
            "orientation_heatmap_quantity": "density-weighted",
        },
    )
    assert (
        flatten_plot_profile_payload("plot:orientation", payload)["orientation_heatmap_quantity"]
        == "density-weighted"
    )
    profile = _constant_orientation(_profile([_unbalanced_frame()] * 2), -0.5)
    state: dict[str, Any] = {}
    plot_orientation_profile(profile, show=False, view_mapping=mapping, capture_state=state)
    assert state["orientation_heatmap_quantity"] == "density-weighted"
    assert np.nanmin(state["plotted_heatmap"]["values"]) < 0.0
    path = _write_plotted_data_export(state, tmp_path / "spatial.csv", include_metadata=True)
    text = path.read_text(encoding="utf-8")
    assert "orientation_spatial=" in text
    assert "molecules/Angstrom^3" in text
    assert "density-weighted" in text


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("old_mapping", [False, True])
def test_studio_spatial_switching_and_capability_gating(monkeypatch, legacy, old_mapping):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase
    from linak.plot.plot_gui import launch_plot_settings_panel

    app = QApplication.instance() or QApplication([])
    assert app is not None
    if not QFontDatabase.families():
        from matplotlib.font_manager import findfont

        QFontDatabase.addApplicationFont(findfont("DejaVu Sans"))
    inspected = []

    def inspect(application):
        window = next(
            w
            for w in application.topLevelWidgets()
            if w.windowTitle() == f"Spatial orientation integration {legacy} {old_mapping}"
        )
        try:
            window._preview_timer.stop()
            monkeypatch.setattr(window, "_schedule_preview_update", lambda *a, **k: None)
            assert window._orientation_spatial_mode() == old_mapping
            window.orientation_heatmap_kind.setCurrentText("Spatial map")
            assert window._orientation_spatial_mode()
            assert window.orientation_heatmap_x_axis.isEnabled()
            assert window.orientation_heatmap_y_axis.isEnabled()
            assert window.orientation_heatmap_quantity.model().item(1).isEnabled() == (not legacy)
            if not legacy:
                window.orientation_heatmap_quantity.setCurrentText("Density-weighted orientation")
            window.orientation_angle.setCurrentText("azimuthal")
            window.orientation_heatmap_y_axis.setCurrentText("Distance")
            settings = window._collect_settings()
            assert settings["orientation_heatmap_y_axis"] == "distance"
            assert settings["orientation_heatmap_kind"] == "spatial"
            window.orientation_heatmap_kind.setCurrentText("Orientation distribution")
            assert not window._orientation_spatial_mode()
            if not old_mapping:
                assert window.orientation_angle.currentText() == "polar"
            window.orientation_heatmap_kind.setCurrentText("Spatial map")
            assert window.orientation_angle.currentText() == "azimuthal"
            assert window.orientation_heatmap_y_axis.currentText() == "Distance", (
                window._orientation_heatmap_states
            )
            if not legacy:
                assert window._selected_orientation_heatmap_quantity() == "density-weighted"
            inspected.append(True)
        finally:
            window._cleanup_preview_artifacts()
            window.hide()

    monkeypatch.setattr(QApplication, "exec", inspect)
    old_settings = {}
    if old_mapping:
        from linak.plot.mappings.orientation_mapping import orientation_plot_options_to_view_mapping
        from linak.plot.profile_persistence import serialize_plot_view_mapping

        mapping = replace(
            orientation_plot_options_to_view_mapping(component="heatmap"),
            fixed_values={
                "orientation_heatmap_x_axis": "x",
                "orientation_heatmap_y_axis": "distance",
            },
        )
        old_settings = {"view_mapping": serialize_plot_view_mapping(mapping)}
        old_settings["orientation_view_states"] = {
            "plot_2d_heatmap": {**old_settings, "_orientation_view_state_initialized": True}
        }
    launch_plot_settings_panel(
        title=f"Spatial orientation integration {legacy} {old_mapping}",
        analysis_name="orientation",
        initial_settings={
            "component": "heatmap",
            **old_settings,
            "_profile_filter_options": {
                "orientation_sparse_grid_available": True,
                "orientation_spatial_density_available": not legacy,
                "orientation_spatial_density_reason": "Recompute orientation" if legacy else "",
                "orientation_axis_ranges": {
                    "x": [0.0, 10.0],
                    "y": [0.0, 10.0],
                    "z": [0.0, 20.0],
                    "distance": [0.0, 10.0],
                },
            },
        },
        on_preview=lambda settings: {},
        on_save=lambda name, settings: "Saved",
    )
    assert inspected == [True]


def test_python_spatial_maps_only_combine_enabled_sources():
    from linak.analysis.orientation import plot_orientation_profiles

    profile = _constant_orientation(_profile([_unbalanced_frame()]), 0.25)
    disabled = replace(profile, sparse_grid=None)
    state = {}
    plot_orientation_profiles(
        [disabled, profile],
        show=False,
        series_enabled=[False, True],
        orientation_heatmap_kind="spatial",
        orientation_heatmap_quantity="density-weighted",
        capture_state=state,
    )
    np.testing.assert_allclose(
        state["plotted_heatmap"]["values"],
        spatial_orientation_map([profile], quantity="density-weighted").values,
        equal_nan=True,
    )


def test_cli_spatial_loading_and_gui_preview_export_agree(tmp_path, monkeypatch):
    from copy import deepcopy
    import matplotlib.pyplot as plt
    from linak.cli import main
    from linak.plot.mappings.orientation_mapping import orientation_plot_options_to_view_mapping
    from linak.plot.profile_persistence import serialize_plot_view_mapping

    profile = _constant_orientation(_profile([_unbalanced_frame()]), 0.25)
    path = tmp_path / "spatial.orientation.h5"
    save_orientation_profile(profile, path)
    visited = []

    def gui(**kwargs):
        initial = kwargs["initial_settings"]
        capabilities = initial["_profile_filter_options"]
        assert capabilities["orientation_sparse_grid_available"]
        assert capabilities["orientation_spatial_density_available"]
        for quantity in ("average", "density", "density-weighted"):
            settings = deepcopy(initial)
            settings.update(
                orientation_heatmap_kind="spatial",
                orientation_heatmap_quantity=quantity,
                view_mapping=serialize_plot_view_mapping(
                    orientation_plot_options_to_view_mapping(
                        orientation_heatmap_kind="spatial", orientation_heatmap_quantity=quantity
                    )
                ),
            )
            preview = kwargs["on_preview_figure"](settings)
            expected = spatial_orientation_map([profile], quantity=quantity).values
            np.testing.assert_allclose(
                preview["plotted_heatmap"]["values"], expected, equal_nan=True
            )
            plt.close(preview["figure"])
            _, exported = kwargs["on_save_data"](settings, str(tmp_path / f"{quantity}.csv"))
            np.testing.assert_allclose(
                exported["plotted_heatmap"]["values"], expected, equal_nan=True
            )
            visited.append(quantity)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", gui)
    assert main(["--log-level", "ERROR", "plot", str(path), "--gui"]) == 0
    assert visited == ["average", "density", "density-weighted"]
    # Public CLI calls without explicit axes still select the spatial grid.
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "plot",
                str(path),
                "--heatmap-kind",
                "spatial",
                "--heatmap-quantity",
                "density",
                "--output",
                str(tmp_path / "density.png"),
            ]
        )
        == 0
    )


def test_ordinary_loading_keeps_spatial_geometry_lazy(tmp_path, monkeypatch):
    from linak.cli import build_plot_parser, _load_orientation_plot_profiles
    import linak.analysis.orientation as orientation

    path = tmp_path / "lazy.orientation.h5"
    save_orientation_profile(_profile([_unbalanced_frame()]), path)
    original_loader = orientation.load_orientation_profiles_by_index
    selected = []

    def load(*args, **kwargs):
        selected.extend(kwargs["dataset_names"])
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(orientation, "load_orientation_profiles_by_index", load)
    args = build_plot_parser(analysis="orientation").parse_args([str(path)])
    profiles, *_ = _load_orientation_plot_profiles(sources=[str(path)], args=args)
    assert profiles[0].sparse_grid is None
    assert not any(name.startswith("orientation_density_") for name in selected)
    assert not any(name.startswith("grid_") for name in selected)


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_sources_with_different_cells_share_conservative_projection_bins(offset):
    first = _constant_orientation(_profile([_unbalanced_frame()]), 0.25)
    second_frame = _unbalanced_frame()
    second_frame.set_cell([20, 10, 20], scale_atoms=True)
    second_frame.set_celldisp([offset, 0, 0])
    second_frame.positions[:, 0] += offset
    second = _constant_orientation(_profile([second_frame] * 3), -0.5)
    options = dict(x_bin_width=30.0, y_bin_width=30.0)
    mean = spatial_orientation_map([first, second], **options)
    density = spatial_orientation_map([first, second], quantity="density", **options)
    weighted = spatial_orientation_map([first, second], quantity="density-weighted", **options)
    np.testing.assert_allclose(mean.values, (0.25 - 3 * 0.5) / 4)
    np.testing.assert_allclose(density.values, (4 / 2000 + 3 * 4 / 4000) / 4, rtol=1e-6)
    np.testing.assert_allclose(weighted.values, mean.values * density.values)


def test_source_bins_are_not_split_when_combining_different_resolutions():
    from linak.analysis.orientation_spatial import _common_projection_edges
    from linak.analysis.density import _density_grid_target_indices
    from types import SimpleNamespace

    # The two source grids overlap only in [0, 4]. Outside that interval each
    # grid contributes its own boundaries; the resulting target is nonuniform.
    edges = _common_projection_edges(
        [np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])]
    )
    np.testing.assert_allclose(edges, [0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    grid = SimpleNamespace(x_bin_edges=np.array([2.0, 4.0, 6.0, 8.0]))
    np.testing.assert_array_equal(
        _density_grid_target_indices(
            grid, "x", np.arange(3), target_edges=edges, target_bin_width=1.0
        ),
        [2, 3, 4],
    )


def test_nonperiodic_geometry_disables_density_without_losing_mean_maps():
    frame = _unbalanced_frame()
    frame.pbc = [True, True, False]
    profile = _profile([frame])
    assert not density_capability(profile)[0]
    assert "fully periodic" in density_capability(profile)[1]
    assert np.any(np.isfinite(spatial_orientation_map([profile]).values))
    with pytest.raises(ValueError, match="fully periodic"):
        spatial_orientation_map([profile], quantity="density")


def test_spatial_source_overrides_and_copies_do_not_double_count():
    from linak.analysis.orientation import plot_orientation_profiles

    first = _constant_orientation(_profile([_unbalanced_frame()]), 0.25)
    second = _constant_orientation(_profile([_unbalanced_frame()]), -0.5)
    descriptors = [
        {"series_id": "first", "source_series_id": "first"},
        {"series_id": "copy", "source_series_id": "first"},
        {"series_id": "second", "source_series_id": "second"},
    ]
    for overrides in (
        {"second": {"enabled": False}},
        {"first": {"enabled": False}, "second": {"enabled": False}},
    ):
        state = {}
        plot_orientation_profiles(
            [first, second],
            series_ids=["first", "second"],
            render_series_descriptors=descriptors,
            series_overrides_by_id=overrides,
            orientation_heatmap_kind="spatial",
            orientation_heatmap_quantity="density-weighted",
            show=False,
            capture_state=state,
        )
        np.testing.assert_allclose(
            state["plotted_heatmap"]["values"],
            spatial_orientation_map([first], quantity="density-weighted").values,
            equal_nan=True,
        )


# Callback tests receive a prepared session; startup itself is tested separately.
pytestmark = pytest.mark.usefixtures("prepared_studio")
