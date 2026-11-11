"""Symmetric orientation must use equal side weights in every representation."""

from dataclasses import replace
from copy import deepcopy
import logging

import numpy as np
import pytest
from ase import Atoms

from linak.analysis.orientation import (
    _mean_with_nan,
    _orientation_profile_for_interface_view,
    compute_orientation_profile,
    load_orientation_profile,
    orientation_sparse_grid_to_heatmap,
    orientation_sparse_grid_to_line,
    plot_orientation_profiles,
    save_orientation_profile,
)


@pytest.fixture(autouse=True)
def _restore_logging_level():
    logger = logging.getLogger("linak")
    level = logger.level
    yield
    logger.setLevel(level)


def _unbalanced_frame(*, shift=0.0, tilted=False):
    symbols, positions = [], []
    for z in (8, 10, 12):
        for x, y in ((2, 2), (7, 2), (2, 7), (7, 7)):
            symbols.append("Pt")
            positions.append([x, y, z])
    # All bisectors point along laboratory +Z. Three top waters and one
    # bottom water give opposite outward means and unequal side populations.
    for x, z in ((5, 4), (5, 16), (2, 16), (7, 16)):
        symbols.extend(("O", "H", "H"))
        positions.extend(([x, 5, z], [x + 0.5, 5, z + 0.65], [x - 0.5, 5, z + 0.65]))
    frame = Atoms(
        symbols, positions=positions,
        cell=[[10, 0, 0], [-1 if tilted else 0, 10, 0], [0, 0, 20]], pbc=True,
    )
    frame.positions[:, 2] += shift
    frame.wrap()
    return frame


def _profile(frames):
    return compute_orientation_profile(
        frames, surface_layout="symmetric", surface_mode="layered",
        surface_elements=["Pt"], bin_width=3,
    )


@pytest.mark.parametrize("shift", [0, 9])
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("angle", ["polar", "azimuthal"])
def test_symmetric_line_and_heatmap_projections_match_equal_side_mean(shift, tilted, angle):
    profile = _profile([_unbalanced_frame(shift=shift, tilted=tilted)])
    assert profile.surface_reference.resolved_layout == "symmetric"
    direct = getattr(profile, f"cos_{angle}_mean")
    np.testing.assert_allclose(direct, 0, atol=1e-12)
    for filters in (None, {"x": (-100, 100)}, {"x": (-100, 7)}):
        _, _, projected = orientation_sparse_grid_to_line(profile, angle=angle, filters=filters)
        np.testing.assert_allclose(projected, direct, atol=1e-7, equal_nan=True)
        _, _, heatmap = orientation_sparse_grid_to_heatmap(
            profile, angle=angle, x_axis="distance", y_axis="y", filters=filters,
        )
        np.testing.assert_allclose(heatmap[np.isfinite(heatmap)], 0, atol=1e-7)


@pytest.mark.parametrize("view,expected", [("top", 1), ("bottom", -1), ("full", 1)])
def test_projected_side_and_full_views_retain_their_polar_conventions(view, expected):
    profile = _orientation_profile_for_interface_view(_profile([_unbalanced_frame()]), view)
    _, _, projected = orientation_sparse_grid_to_line(profile)
    np.testing.assert_allclose(projected[np.isfinite(projected)], expected, atol=1e-7)
    _, _, heatmap = orientation_sparse_grid_to_heatmap(profile, x_axis="distance", y_axis="y")
    np.testing.assert_allclose(heatmap[np.isfinite(heatmap)], expected, atol=1e-7)


def test_folded_statistics_use_equal_side_means_and_per_interface_density():
    frame = _unbalanced_frame()
    profile = _profile([frame, frame[:-3]])
    np.testing.assert_allclose(profile.cos_polar_mean, 0, atol=1e-12)
    polar_stats = profile.series_statistics["cos_polar_mean"]
    np.testing.assert_allclose(polar_stats.sample_std, 0, atol=1e-12)
    np.testing.assert_allclose(polar_stats.sample_sem, 0, atol=1e-12)
    # Four then three waters, with two interface volumes in each frame.
    expected_density_samples = np.asarray([4, 3]) / (2 * 100 * 3)
    density_stats = profile.series_statistics["density"]
    np.testing.assert_allclose(profile.density, expected_density_samples.mean())
    np.testing.assert_allclose(density_stats.sample_std, expected_density_samples.std(ddof=1))
    np.testing.assert_allclose(density_stats.sample_sem, expected_density_samples.std(ddof=1) / np.sqrt(2))
    # The unfolded reference remains the laboratory frame.
    np.testing.assert_allclose(profile.full_cos_polar_mean[np.isfinite(profile.full_cos_polar_mean)], 1)


def test_symmetric_projection_round_trip_preserves_side_accumulators(tmp_path):
    profile = _profile([_unbalanced_frame()])
    path = tmp_path / "symmetric.orientation.h5"
    save_orientation_profile(profile, path)
    loaded = load_orientation_profile(path)
    _, _, projected = orientation_sparse_grid_to_line(loaded)
    np.testing.assert_allclose(projected, loaded.cos_polar_mean, atol=1e-7)


def test_legacy_symmetric_grid_without_side_accumulators_requires_recompute():
    profile = _profile([_unbalanced_frame()])
    profile = replace(profile, sparse_grid=replace(profile.sparse_grid, top_cos_polar_sum=None))
    with pytest.raises(ValueError, match="top/bottom grid accumulators"):
        orientation_sparse_grid_to_line(profile)


def test_fractional_side_counts_are_valid_orientation_samples():
    np.testing.assert_allclose(_mean_with_nan(np.asarray([0.25]), np.asarray([0.5])), [0.5])


def test_comparison_expands_gui_series_metadata_and_copy_overrides(tmp_path):
    profile = _profile([_unbalanced_frame()])
    state = {}
    descriptors = [
        {"series_id": "source", "source_series_id": "source", "default_label": "Water"},
        {"series_id": "copy", "source_series_id": "source", "source_kind": "source",
         "default_label": "Water copy"},
    ]
    output = tmp_path / "comparison.png"
    plot_orientation_profiles(
        [profile], interface_view="all", output=output, show=False, capture_state=state,
        series_ids=["source"], series_labels=["Water"], line_colors=["blue"],
        series_enabled=[True], series_show_in_legend=[True], series_line_widths=[2],
        series_markers=[""], series_fit_configs=[{"model": "off"}],
        series_error_configs=[{"enabled": False}], series_normalization_modes=["none"],
        series_normalization_values=[1], series_normalization_x_refs=[None],
        render_series_descriptors=descriptors,
        series_overrides_by_id={
            "source": {"line_kwargs": {"linestyle": "-"}},
            "copy": {"label_override": "Replica", "color": "red"},
        },
    )
    assert output.exists()
    artists = state["series_artists"]
    assert len(artists) == 6
    for base in ("source", "copy"):
        for view, line_style in zip(("folded", "top", "bottom"), ("-", "--", ":")):
            artist = artists[f"{base}::interface:{view}"]
            assert artist.get_linestyle() == line_style


def test_cli_gui_preview_switches_all_surface_views_on_distance_axis(tmp_path, monkeypatch):
    from linak.cli import main
    import matplotlib.pyplot as plt

    path = tmp_path / "symmetric.orientation.h5"
    profile = _profile([_unbalanced_frame()])
    save_orientation_profile(profile, path)
    views = []

    def gui(**kwargs):
        initial = kwargs["initial_settings"]
        assert initial.get("interface_view", "folded") == "folded"
        for view in ("folded", "top", "bottom", "all", "full", "top"):
            settings = deepcopy(initial)
            settings["interface_view"] = view
            result = kwargs["on_preview_figure"](settings)
            fig = result["figure"]
            ax = fig.axes[0]
            assert len(ax.lines) == (3 if view == "all" else 1)
            if view != "full":
                assert "Distance" in ax.get_xlabel()
                for line in ax.lines:
                    np.testing.assert_allclose(line.get_xdata(), profile.bin_centers, atol=1e-6)
            views.append(view)
            plt.close(fig)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", gui)
    assert main(["--log-level", "ERROR", "plot", str(path), "--gui"]) == 0
    assert views == ["folded", "top", "bottom", "all", "full", "top"]


def test_gui_defaults_to_distance_and_restores_it_after_full_cell(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from linak.plot.plot_gui import launch_plot_settings_panel

    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)

    for initial_view in ("folded", "full"):
        inspected = []

        def inspect(app):
            for window in app.topLevelWidgets():
                if not window.isVisible() or not hasattr(window, "orientation_line_x_axis"):
                    continue
                assert window._selected_interface_view() == initial_view
                assert window.orientation_line_x_axis.currentText() == (
                    "Z" if initial_view == "full" else "Distance"
                )
                for view in ("top", "bottom", "all", "full", "folded"):
                    window._set_combo_value(
                        window.interface_view_combo, window._interface_view_labels()[view]
                    )
                    assert window.orientation_line_x_axis.currentText() == (
                        "Z" if view == "full" else "Distance"
                    )
                inspected.append(True)
                window.close()
            return 0

        monkeypatch.setattr(QApplication, "exec", inspect)
        launch_plot_settings_panel(
            title="Orientation view regression", analysis_name="orientation",
            initial_settings={
                "interface_view": initial_view,
                "_profile_filter_options": {
                    "surface_layout_resolved": "symmetric", "surface_axis": "z",
                    "surface_view_capabilities": ["full", "top", "bottom", "folded", "all"],
                },
            },
            on_preview=lambda settings: None, on_save=lambda name, settings: "",
        )
        assert inspected


# Callback tests receive a prepared session; startup itself is tested separately.
pytestmark = pytest.mark.usefixtures("prepared_studio")
