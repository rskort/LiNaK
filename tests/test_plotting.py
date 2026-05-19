import numpy as np
import pytest

import linak.plot.plotting as plotting_module
from linak.analysis.orientation import OrientationProfile, plot_orientation_profile
from linak.analysis.rdf import RDFProfile, plot_rdf_profile
from linak.analysis.statistics import SeriesStatistics


def test_format_axis_label_units_wraps_trailing_units_in_mathrm():
    assert plotting_module.format_axis_label_units("Density (g/cm^3)") == "Density (g/cm^3)"
    assert plotting_module.format_axis_label_units("Time (ps)") == "Time (ps)"


def test_format_axis_label_units_preserves_existing_math_and_non_unit_parentheses():
    assert plotting_module.format_axis_label_units(
        "Distance to the surface ($\\mathrm{\\AA}$)"
    ) == ("Distance to the surface ($\\mathrm{\\AA}$)")
    assert plotting_module.format_axis_label_units("g(r)") == "g(r)"


def test_resolve_explicit_plot_text_preserves_blank_override():
    assert plotting_module.resolve_explicit_plot_text(None, "Time (ps)") == "Time (ps)"
    assert plotting_module.resolve_explicit_plot_text("", "Time (ps)") == ""


def test_normalize_series_values_area_uses_numpy_compatibility(monkeypatch):
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 1.0, 1.0], dtype=float)

    monkeypatch.delattr(plotting_module.np, "trapezoid", raising=False)
    trapz_calls = 0

    def fallback_trapz(y_values, x_values):
        nonlocal trapz_calls
        trapz_calls += 1
        return 2.0

    monkeypatch.setattr(plotting_module.np, "trapz", fallback_trapz, raising=False)

    normalized, changed, scale = plotting_module._normalize_series_values(
        x,
        y,
        mode="area",
        target_value=1.0,
        reference_x=None,
        label="series",
    )

    assert changed is True
    assert scale == pytest.approx(0.5)
    assert trapz_calls == 1
    np.testing.assert_allclose(normalized, np.array([0.5, 0.5, 0.5]))


def test_normalize_series_values_none_ignores_stale_target_and_reference():
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    normalized, changed, scale = plotting_module._normalize_series_values(
        x,
        y,
        mode="none",
        target_value=5.0,
        reference_x=1.25,
        label="series",
    )

    assert changed is False
    assert scale == pytest.approx(1.0)
    np.testing.assert_allclose(normalized, y)


def test_sanitize_line_collection_kwargs_removes_marker_only_fields():
    sanitized = plotting_module._sanitize_line_collection_kwargs(
        {
            "label": "Series A",
            "color": "#ff0000",
            "marker": "o",
            "markersize": 8.0,
            "markeredgecolor": "#000000",
            "markevery": 2,
            "lw": 1.5,
            "alpha": 0.7,
        }
    )

    assert sanitized == {
        "linewidths": 1.5,
        "alpha": 0.7,
    }


def test_plot_line_series_hides_all_spines_when_axes_border_disabled(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        title="Line",
        x_label="x",
        y_label="y",
        output=tmp_path / "line.png",
        show=False,
        style=plotting_module.with_style_overrides(axes_border=False),
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert capture_state["axes_border"] is False
    assert all(not spine.get_visible() for spine in ax.spines.values())


def test_plot_multi_line_series_preserves_spines_when_axes_border_enabled(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.array([3.0, 2.0, 1.0], dtype=float),
        ],
        ["a", "b"],
        title="Multi",
        x_label="x",
        y_label="y",
        output=tmp_path / "multi.png",
        show=False,
        style=plotting_module.with_style_overrides(axes_border=True),
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert capture_state["axes_border"] is True
    assert all(spine.get_visible() for spine in ax.spines.values())


def test_plot_heatmap_series_hides_all_spines_when_axes_border_disabled(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_heatmap_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([-1.0, 0.0, 1.0], dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        title="Heat",
        x_label="x",
        y_label="y",
        output=tmp_path / "heat.png",
        show=False,
        style=plotting_module.with_style_overrides(axes_border=False),
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert capture_state["axes_border"] is False
    assert all(not spine.get_visible() for spine in ax.spines.values())


def test_plot_multi_line_series_applies_per_axis_tick_and_font_color_settings(tmp_path):
    capture_state: dict[str, object] = {}
    tick_params = {
        "direction": "out",
        "length": 2.0,
        "width": 0.5,
        "_x_tick_params": {"direction": "in", "length": 7.0, "width": 1.5},
        "_y_tick_params": {"direction": "out", "length": 4.0, "width": 0.75},
        "_x_minor_ticks_mode": "on",
        "_y_minor_ticks_mode": "off",
    }

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["series"],
        title="Styled",
        x_label="x",
        y_label="y",
        output=tmp_path / "styled_ticks.png",
        show=False,
        style=plotting_module.with_style_overrides(font_color="#123456"),
        tick_params_kwargs=tick_params,
        x_label_font_size=15,
        y_label_font_size=16,
        x_tick_font_size=9,
        y_tick_font_size=11,
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert ax.title.get_color() == "#123456"
    assert ax.xaxis.label.get_color() == "#123456"
    assert ax.yaxis.label.get_color() == "#123456"
    assert ax.xaxis.label.get_size() == pytest.approx(15)
    assert ax.yaxis.label.get_size() == pytest.approx(16)
    assert ax.xaxis.get_major_ticks()[0].label1.get_size() == pytest.approx(9)
    assert ax.yaxis.get_major_ticks()[0].label1.get_size() == pytest.approx(11)
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == pytest.approx(7)
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_markersize() == pytest.approx(4)


def test_plot_multi_line_series_applies_axis_specific_tick_visibility(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["series"],
        title="Ticks",
        x_label="x",
        y_label="y",
        output=tmp_path / "axis_ticks_visibility.png",
        show=False,
        ticks_visible=True,
        tick_params_kwargs={
            "_x_ticks_visible": False,
            "_y_ticks_visible": True,
            "_ticks_axis": "both",
            "axis": "both",
        },
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_visible() is False
    assert ax.xaxis.get_major_ticks()[0].label1.get_visible() is False
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_visible() is True
    assert ax.yaxis.get_major_ticks()[0].label1.get_visible() is True


def test_plot_multi_line_series_applies_axis_specific_grid_visibility(tmp_path):
    capture_state_x: dict[str, object] = {}
    capture_state_y: dict[str, object] = {}

    result_x = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["series"],
        title="Grid X",
        x_label="x",
        y_label="y",
        output=tmp_path / "grid_x.png",
        show=False,
        style=plotting_module.with_style_overrides(grid=True),
        grid_kwargs={"axis": "x", "which": "major"},
        capture_state=capture_state_x,
    )
    result_y = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["series"],
        title="Grid Y",
        x_label="x",
        y_label="y",
        output=tmp_path / "grid_y.png",
        show=False,
        style=plotting_module.with_style_overrides(grid=True),
        grid_kwargs={"axis": "y", "which": "major"},
        capture_state=capture_state_y,
    )

    assert result_x is not None
    assert result_y is not None

    ax_x = capture_state_x["axes"]
    ax_y = capture_state_y["axes"]
    assert capture_state_x["grid_kwargs"] == {"axis": "x", "which": "major"}
    assert capture_state_y["grid_kwargs"] == {"axis": "y", "which": "major"}
    assert any(line.get_visible() for line in ax_x.get_xgridlines())
    assert not any(line.get_visible() for line in ax_x.get_ygridlines())
    assert not any(line.get_visible() for line in ax_y.get_xgridlines())
    assert any(line.get_visible() for line in ax_y.get_ygridlines())


def test_plot_multi_line_series_preserves_legend_title_fontsize_in_capture_state(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["series"],
        title="Legend",
        x_label="x",
        y_label="y",
        output=tmp_path / "legend_title_font.png",
        show=False,
        legend=True,
        legend_title="Runs",
        legend_kwargs={"title_fontsize": 17, "ncols": 2, "frameon": False},
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["legend_kwargs"]["title_fontsize"] == 17
    assert capture_state["legend_kwargs"]["ncols"] == 2
    assert capture_state["legend_kwargs"]["frameon"] is False


def test_plot_multi_line_series_integrates_plotted_data_with_boundary_interpolation(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([0.0, 2.0, 4.0], dtype=float)],
        ["line"],
        title="Integrated",
        x_label="x",
        y_label="y",
        output=tmp_path / "integrated.png",
        show=False,
        capture_state=capture_state,
        series_overrides_by_id={
            "series:0": {
                "integration": {
                    "enabled": True,
                    "source": "plotted",
                    "x_min": 0.5,
                    "x_max": 1.5,
                    "baseline": 0.0,
                    "color": "#ff0000",
                    "alpha": 0.4,
                },
            },
        },
    )

    assert result is not None
    summaries = capture_state["integration_summaries"]
    assert summaries[0]["status"] == "ok"
    assert summaries[0]["signed_area"] == pytest.approx(2.0)
    assert summaries[0]["absolute_area"] == pytest.approx(2.0)
    assert summaries[0]["x_min"] == pytest.approx(0.5)
    assert summaries[0]["x_max"] == pytest.approx(1.5)
    assert summaries[0]["color"] == "#ff0000"
    assert summaries[0]["alpha"] == pytest.approx(0.4)


def test_plot_multi_line_series_integrates_raw_profile_data_before_normalization(tmp_path):
    raw_state: dict[str, object] = {}
    plotted_state: dict[str, object] = {}
    x_values = np.array([0.0, 1.0, 2.0], dtype=float)
    y_values = np.array([2.0, 2.0, 2.0], dtype=float)

    plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["line"],
        title="Raw",
        x_label="x",
        y_label="y",
        output=tmp_path / "raw_integrated.png",
        show=False,
        capture_state=raw_state,
        series_normalization_modes=["factor"],
        series_normalization_values=[0.5],
        series_overrides_by_id={
            "series:0": {"integration": {"enabled": True, "source": "raw"}},
        },
    )
    plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["line"],
        title="Plotted",
        x_label="x",
        y_label="y",
        output=tmp_path / "plotted_integrated.png",
        show=False,
        capture_state=plotted_state,
        series_normalization_modes=["factor"],
        series_normalization_values=[0.5],
        series_overrides_by_id={
            "series:0": {"integration": {"enabled": True, "source": "plotted"}},
        },
    )

    assert raw_state["integration_summaries"][0]["signed_area"] == pytest.approx(4.0)
    assert plotted_state["integration_summaries"][0]["signed_area"] == pytest.approx(2.0)


def test_plot_multi_line_series_applies_x_axis_scale_and_offset(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 50.0, 100.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["line"],
        title="Scaled x",
        x_label="x",
        y_label="y",
        output=tmp_path / "scaled_x.png",
        show=False,
        x_axis_scale=0.2,
        x_axis_offset=1.0,
        capture_state=capture_state,
    )

    assert result is not None
    line = capture_state["axes"].lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.array([1.0, 11.0, 21.0]))
    assert capture_state["x_axis_scale"] == pytest.approx(0.2)
    assert capture_state["x_axis_offset"] == pytest.approx(1.0)


def test_plot_multi_line_series_falls_back_to_count_axis_before_mapping(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["line"],
        title="Index x",
        x_label="x",
        y_label="y",
        output=tmp_path / "index_x.png",
        show=False,
        x_axis_scale=0.2,
        x_axis_offset=1.0,
        capture_state=capture_state,
    )

    assert result is not None
    line = capture_state["axes"].lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.array([1.2, 1.4, 1.6]))


def test_x_axis_mapping_affects_plotted_integration_but_not_raw_integration(tmp_path):
    raw_state: dict[str, object] = {}
    plotted_state: dict[str, object] = {}
    x_values = np.array([0.0, 1.0, 2.0], dtype=float)
    y_values = np.array([2.0, 2.0, 2.0], dtype=float)

    plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["line"],
        title="Raw scaled",
        x_label="x",
        y_label="y",
        output=tmp_path / "raw_scaled_integrated.png",
        show=False,
        x_axis_scale=0.5,
        capture_state=raw_state,
        series_overrides_by_id={
            "series:0": {"integration": {"enabled": True, "source": "raw"}},
        },
    )
    plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["line"],
        title="Plotted scaled",
        x_label="x",
        y_label="y",
        output=tmp_path / "plotted_scaled_integrated.png",
        show=False,
        x_axis_scale=0.5,
        capture_state=plotted_state,
        series_overrides_by_id={
            "series:0": {"integration": {"enabled": True, "source": "plotted"}},
        },
    )

    assert raw_state["integration_summaries"][0]["signed_area"] == pytest.approx(4.0)
    assert plotted_state["integration_summaries"][0]["signed_area"] == pytest.approx(2.0)


def test_plot_multi_line_series_rejects_zero_x_axis_scale(tmp_path):
    with pytest.raises(ValueError, match="X-axis scale factor"):
        plotting_module.plot_multi_line_series(
            [np.array([0.0, 1.0], dtype=float)],
            [np.array([0.0, 1.0], dtype=float)],
            ["line"],
            title="Bad x scale",
            x_label="x",
            y_label="y",
            output=tmp_path / "bad_x_scale.png",
            show=False,
            x_axis_scale=0.0,
        )


def test_plot_multi_line_series_rejects_invalid_integration_range(tmp_path):
    with pytest.raises(ValueError, match="Integration x-min"):
        plotting_module.plot_multi_line_series(
            [np.array([0.0, 1.0, 2.0], dtype=float)],
            [np.array([0.0, 2.0, 4.0], dtype=float)],
            ["line"],
            title="Integrated",
            x_label="x",
            y_label="y",
            output=tmp_path / "bad_integrated.png",
            show=False,
            series_overrides_by_id={
                "series:0": {
                    "integration": {"enabled": True, "x_min": 1.0, "x_max": 1.0},
                },
            },
        )


def test_plot_multi_line_series_rejects_all_visible_integration_target(tmp_path):
    with pytest.raises(ValueError, match="Integration target"):
        plotting_module.plot_multi_line_series(
            [np.array([0.0, 1.0], dtype=float)],
            [np.array([0.0, 1.0], dtype=float)],
            ["line"],
            title="Integrated",
            x_label="x",
            y_label="y",
            output=tmp_path / "bad_target.png",
            show=False,
            series_overrides_by_id={
                "series:0": {
                    "integration": {"enabled": True, "target": "all_visible"},
                },
            },
        )


def test_plot_heatmap_series_applies_figure_alpha_and_font_color_settings(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_heatmap_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([-1.0, 0.0, 1.0], dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        title="Heat",
        x_label="x",
        y_label="y",
        output=tmp_path / "styled_heatmap.png",
        show=False,
        style=plotting_module.with_style_overrides(font_color="#654321"),
        figure_kwargs={"facecolor": "#ffffff", "alpha": 0.35},
        heatmap_colorbar_label="Colorbar",
        x_label_font_size=13,
        y_label_font_size=14,
        x_tick_font_size=8,
        y_tick_font_size=10,
        capture_state=capture_state,
    )

    assert result is not None
    fig = capture_state["figure"]
    ax = capture_state["axes"]
    colorbar = capture_state["heatmap_colorbar"]
    assert fig.patch.get_alpha() == pytest.approx(0.35)
    assert ax.patch.get_alpha() == pytest.approx(0.35)
    assert ax.title.get_color() == "#654321"
    assert ax.xaxis.label.get_color() == "#654321"
    assert ax.yaxis.label.get_color() == "#654321"
    assert colorbar.ax.yaxis.label.get_color() == "#654321"
    assert ax.xaxis.label.get_size() == pytest.approx(13)
    assert ax.yaxis.label.get_size() == pytest.approx(14)
    assert ax.xaxis.get_major_ticks()[0].label1.get_size() == pytest.approx(8)
    assert ax.yaxis.get_major_ticks()[0].label1.get_size() == pytest.approx(10)


def test_plot_multi_line_series_applies_figure_alpha_to_axes_face(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0], dtype=float)],
        [np.array([1.0, 2.0], dtype=float)],
        ["line"],
        title="Line",
        x_label="x",
        y_label="y",
        output=tmp_path / "styled_line.png",
        show=False,
        figure_kwargs={"alpha": 0.35},
        capture_state=capture_state,
    )

    assert result is not None
    fig = capture_state["figure"]
    ax = capture_state["axes"]
    assert fig.patch.get_alpha() == pytest.approx(0.35)
    assert ax.patch.get_alpha() == pytest.approx(0.35)


def test_with_style_overrides_updates_axes_border():
    style = plotting_module.with_style_overrides(axes_border=False)

    assert style.axes_border is False
    assert plotting_module.DEFAULT_PLOT_STYLE.axes_border is True


def test_plot_line_series_applies_dict_border_selectively(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        title="Line",
        x_label="x",
        y_label="y",
        output=tmp_path / "line_partial.png",
        show=False,
        style=plotting_module.with_style_overrides(
            axes_border={"left": True, "right": False, "top": False, "bottom": True}
        ),
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert ax.spines["left"].get_visible() is True
    assert ax.spines["right"].get_visible() is False
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["bottom"].get_visible() is True
    # Mixed state → capture emits a dict, not a bool
    assert isinstance(capture_state["axes_border"], dict)


def test_default_plot_font_sizes_follow_base_font_size():
    assert plotting_module.default_plot_font_sizes(12) == {
        "title_font_size": 14,
        "label_font_size": 12,
        "tick_font_size": 10,
        "legend_font_size": 10,
    }


def test_with_style_overrides_updates_inherited_font_sizes_from_base_font_size():
    style = plotting_module.with_style_overrides(font_size=16)

    assert style.base_font_size == 16
    assert style.title_font_size == 18
    assert style.label_font_size == 16
    assert style.tick_font_size == 14
    assert style.legend_font_size == 14


def test_with_style_overrides_preserves_explicit_font_override_when_base_font_size_changes():
    custom = plotting_module.with_style_overrides(label_font_size=15)
    updated = plotting_module.with_style_overrides(base_style=custom, font_size=18)

    assert updated.base_font_size == 18
    assert updated.title_font_size == 20
    assert updated.label_font_size == 15
    assert updated.tick_font_size == 16
    assert updated.legend_font_size == 16


def test_plot_line_series_renders_error_band_and_reports_summary(tmp_path):
    from linak.analysis.statistics import SeriesStatistics

    capture_state: dict[str, object] = {}
    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        title="Line",
        x_label="x",
        y_label="y",
        output=tmp_path / "line_error_band.png",
        show=False,
        series_statistics=SeriesStatistics(
            point_count=np.array([12, 12, 12], dtype=int),
            sample_n=np.array([5, 5, 5], dtype=int),
            sample_std=np.array([0.2, 0.2, 0.2], dtype=float),
            sample_sem=np.array([0.1, 0.1, 0.1], dtype=float),
            block_n=np.array([4, 4, 4], dtype=int),
            block_std=np.array([0.4, 0.4, 0.4], dtype=float),
            block_sem=np.array([0.2, 0.2, 0.2], dtype=float),
        ),
        error_config={
            "enabled": True,
            "stat": "block_sem",
            "style": "band",
            "color": "#cc5500",
        },
        capture_state=capture_state,
    )

    assert result is not None
    error_summary = capture_state["series_error_summaries"]["series"]
    assert error_summary["status"] == "ok"
    assert error_summary["stat"] == "block_sem"
    assert error_summary["style"] == "band"
    assert error_summary["color"] == "#cc5500"
    assert capture_state["series_available_error_stats"]["series"] == [
        "sample_std",
        "sample_sem",
        "block_std",
        "block_sem",
    ]


def test_resolve_series_error_availability_reports_single_stat_reason():
    availability = plotting_module.resolve_series_error_availability(
        supported_for_view=True,
        available_stats=["sample_sem"],
    )

    assert availability.available_stats == ["sample_sem"]
    assert availability.default_stat == "sample_sem"
    assert availability.selector_enabled is False
    assert availability.reason == "Only 'sample_sem' is available for this series."


def test_resolve_series_error_availability_keeps_rebinned_saved_sample_stats():
    availability = plotting_module.resolve_series_error_availability(
        supported_for_view=True,
        available_stats=["sample_std", "sample_sem"],
        error_status="rebinned_saved_profile",
        error_reason="Block uncertainty is unavailable after x rebinning; using sample_sem.",
    )

    assert availability.available_stats == ["sample_std", "sample_sem"]
    assert availability.default_stat == "sample_sem"
    assert availability.selector_enabled is True
    assert (
        availability.reason
        == "Block uncertainty is unavailable after x rebinning; using sample_sem."
    )


def test_resolve_series_error_availability_reports_no_stats_without_preview_placeholder():
    availability = plotting_module.resolve_series_error_availability(
        supported_for_view=True,
        available_stats=[],
    )

    assert availability.available_stats == []
    assert availability.default_stat is None
    assert availability.selector_enabled is False
    assert availability.reason == "No uncertainty statistics are available for this series."


def test_plot_line_series_min_bin_points_masks_sparse_raw_groups(tmp_path):
    capture_state: dict[str, object] = {}
    result = plotting_module.plot_line_series(
        np.array([0.0, 0.0, 1.0, 2.0, 2.0], dtype=float),
        np.array([1.0, 3.0, 7.0, 2.0, 4.0], dtype=float),
        title="Raw",
        x_label="x",
        y_label="y",
        output=tmp_path / "line_raw_masked.png",
        show=False,
        raw_point_statistics=True,
        min_bin_points=2,
        capture_state=capture_state,
    )

    assert result is not None
    stats = capture_state["series_statistics"]["series"]
    assert stats["point_count"] == 2
    assert capture_state["series_masked_bin_counts"]["series"] == 1


def test_plot_line_series_disables_saved_profile_error_overlay_after_rebinning(tmp_path):
    from linak.analysis.statistics import SeriesStatistics

    capture_state: dict[str, object] = {}
    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        title="Rebinned",
        x_label="x",
        y_label="y",
        output=tmp_path / "line_rebinned_error.png",
        show=False,
        x_bin_width=2.0,
        series_statistics=SeriesStatistics(
            point_count=np.array([10, 10, 10, 10], dtype=int),
            sample_n=np.array([5, 5, 5, 5], dtype=int),
            sample_std=np.array([0.2, 0.2, 0.2, 0.2], dtype=float),
            sample_sem=np.array([0.1, 0.1, 0.1, 0.1], dtype=float),
        ),
        error_config={"enabled": True, "stat": "sample_sem", "style": "whiskers"},
        capture_state=capture_state,
    )

    assert result is not None
    error_summary = capture_state["series_error_summaries"]["series"]
    assert error_summary["status"] == "ok"
    assert error_summary["stat"] == "sample_sem"
    assert error_summary["statistics_mode"] == "saved_rebinned_sample"
    assert capture_state["series_available_error_stats"]["series"] == [
        "sample_std",
        "sample_sem",
    ]


def test_plot_line_series_min_bin_points_masks_rebinned_saved_profile(tmp_path):
    from linak.analysis.statistics import SeriesStatistics

    capture_state: dict[str, object] = {}
    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        np.array([1.0, 3.0, 5.0, 7.0], dtype=float),
        title="Rebinned",
        x_label="x",
        y_label="y",
        output=tmp_path / "line_rebinned_saved_masked.png",
        show=False,
        x_bin_width=2.0,
        min_bin_points=10,
        series_statistics=SeriesStatistics(
            point_count=np.array([2, 3, 7, 4], dtype=int),
            sample_n=np.array([2, 3, 7, 4], dtype=int),
            sample_std=np.array([0.5, 0.7, 1.0, 1.2], dtype=float),
            sample_sem=np.array([0.35, 0.4, 0.38, 0.6], dtype=float),
        ),
        error_config={"enabled": True, "stat": "sample_sem", "style": "band"},
        capture_state=capture_state,
    )

    assert result is not None
    error_summary = capture_state["series_error_summaries"]["series"]
    assert error_summary["status"] == "ok"
    assert error_summary["statistics_mode"] == "saved_rebinned_sample"
    assert error_summary["point_count"] == 1
    assert capture_state["series_masked_bin_counts"]["series"] == 1
    stats = capture_state["series_statistics"]["series"]
    assert stats["point_count"] == 1


def test_plot_line_series_rebinned_saved_profile_falls_back_from_block_to_sample(tmp_path):
    from linak.analysis.statistics import SeriesStatistics

    capture_state: dict[str, object] = {}
    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        title="Rebinned",
        x_label="x",
        y_label="y",
        output=tmp_path / "line_rebinned_block_fallback.png",
        show=False,
        x_bin_width=2.0,
        series_statistics=SeriesStatistics(
            point_count=np.array([10, 12, 8, 6], dtype=int),
            sample_n=np.array([5, 6, 4, 3], dtype=int),
            sample_std=np.array([0.2, 0.3, 0.4, 0.5], dtype=float),
            sample_sem=np.array([0.1, 0.12, 0.2, 0.29], dtype=float),
            block_n=np.array([4, 4, 4, 4], dtype=int),
            block_std=np.array([0.4, 0.4, 0.4, 0.4], dtype=float),
            block_sem=np.array([0.2, 0.2, 0.2, 0.2], dtype=float),
        ),
        error_config={"enabled": True, "stat": "block_sem", "style": "band"},
        capture_state=capture_state,
    )

    assert result is not None
    error_summary = capture_state["series_error_summaries"]["series"]
    assert error_summary["status"] == "ok"
    assert error_summary["stat"] == "sample_sem"
    assert error_summary["statistics_mode"] == "saved_rebinned_sample"
    assert "Block uncertainty is unavailable after x rebinning" in error_summary["reason"]


def test_plot_line_series_rejects_section_width_smaller_than_data_bin_width(tmp_path):
    with pytest.raises(ValueError, match="smaller than the data bin width 1"):
        plotting_module.plot_line_series(
            np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            title="Too small",
            x_label="x",
            y_label="y",
            output=tmp_path / "too_small.png",
            show=False,
            x_bin_width=0.5,
        )


def test_plot_line_series_rejects_section_width_larger_than_full_range(tmp_path):
    with pytest.raises(ValueError, match="larger than the available x-range 3"):
        plotting_module.plot_line_series(
            np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            title="Too large",
            x_label="x",
            y_label="y",
            output=tmp_path / "too_large.png",
            show=False,
            x_bin_width=5.0,
        )


def test_plot_line_series_accepts_section_width_equal_to_data_bin_width(tmp_path):
    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        title="Equal width",
        x_label="x",
        y_label="y",
        output=tmp_path / "equal_width.png",
        show=False,
        x_bin_width=1.0,
    )

    assert result is not None
    assert (tmp_path / "equal_width.png").exists()


def test_plot_line_series_renders_annotations_and_reports_summary(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        title="Annotated",
        x_label="x",
        y_label="y",
        output=tmp_path / "annotated_line.png",
        show=False,
        annotations=[
            {
                "id": "text-1",
                "type": "text",
                "name": "Label",
                "coord_system": "axes",
                "x": 0.5,
                "y": 0.9,
                "text": "Peak",
            },
            {
                "id": "line-1",
                "type": "line",
                "name": "Guide",
                "coord_system": "data",
                "x1": 0.5,
                "y1": 1.0,
                "x2": 1.5,
                "y2": 2.5,
            },
            {
                "id": "arrow-1",
                "type": "arrow",
                "name": "Pointer",
                "coord_system": "data",
                "x1": 0.25,
                "y1": 2.8,
                "x2": 1.0,
                "y2": 2.0,
            },
            {
                "id": "text-off",
                "type": "text",
                "name": "Hidden",
                "enabled": False,
                "coord_system": "axes",
                "x": 0.2,
                "y": 0.2,
                "text": "skip",
            },
        ],
        capture_state=capture_state,
    )

    assert result is not None
    summaries = capture_state["annotations_summary"]
    assert [entry["status"] for entry in summaries] == ["ok", "ok", "ok", "disabled"]
    ax = capture_state["axes"]
    assert any(text.get_text() == "Peak" for text in ax.texts)
    assert len(ax.lines) >= 2
    assert any(getattr(text, "arrow_patch", None) is not None for text in ax.texts)


def test_plot_heatmap_series_renders_annotations(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_heatmap_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        title="Heatmap",
        x_label="x",
        y_label="y",
        output=tmp_path / "annotated_heatmap.png",
        show=False,
        annotations=[
            {
                "id": "heat-text",
                "type": "text",
                "name": "Heat label",
                "coord_system": "axes",
                "x": 0.5,
                "y": 1.05,
                "text": "Surface",
            },
            {
                "id": "heat-arrow",
                "type": "arrow",
                "name": "Heat pointer",
                "coord_system": "data",
                "x1": 0.5,
                "y1": 1.8,
                "x2": 1.0,
                "y2": 1.0,
            },
        ],
        capture_state=capture_state,
    )

    assert result is not None
    summaries = capture_state["annotations_summary"]
    assert [entry["type"] for entry in summaries] == ["text", "arrow"]
    assert all(entry["status"] == "ok" for entry in summaries)
    ax = capture_state["axes"]
    assert any(text.get_text() == "Surface" for text in ax.texts)
    assert any(getattr(text, "arrow_patch", None) is not None for text in ax.texts)


def test_plot_rdf_profile_forwards_annotations_and_reports_rdf_provenance(tmp_path):
    capture_state: dict[str, object] = {}
    profile = RDFProfile(
        species_a="Na",
        species_b="Cl",
        bin_edges=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        bin_centers=np.array([0.5, 1.5, 2.5], dtype=float),
        g_r=np.array([0.0, 1.2, 0.8], dtype=float),
        n_frames=25,
        series_statistics={
            "g_r": SeriesStatistics(
                point_count=np.array([10, 12, 9], dtype=int),
                sample_n=np.array([5, 5, 5], dtype=int),
                sample_std=np.array([0.1, 0.2, 0.15], dtype=float),
                sample_sem=np.array([0.04, 0.08, 0.06], dtype=float),
                block_n=np.array([4, 4, 4], dtype=int),
                block_std=np.array([0.12, 0.22, 0.18], dtype=float),
                block_sem=np.array([0.06, 0.11, 0.09], dtype=float),
            )
        },
    )

    result = plot_rdf_profile(
        profile,
        output=tmp_path / "rdf_wrapper_annotations.png",
        show=False,
        annotations=[
            {
                "id": "rdf-text",
                "type": "text",
                "coord_system": "axes",
                "x": 0.5,
                "y": 0.9,
                "text": "RDF peak",
            }
        ],
        error_config={"enabled": True, "stat": "block_sem", "style": "band"},
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["annotations_summary"][0]["status"] == "ok"
    assert any(text.get_text() == "RDF peak" for text in capture_state["axes"].texts)
    summary = capture_state["series_error_summaries"]["series"]
    assert summary["provenance_family"] == "block"
    assert "g(r)" in summary["provenance"]


def test_plot_rdf_profile_writes_output_without_strict_zip_runtime_dependency(tmp_path):
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float),
        bin_centers=np.array([0.05, 0.15, 0.25, 0.35], dtype=float),
        g_r=np.array([0.0, 0.5, 1.2, 0.8], dtype=float),
        n_frames=4,
    )
    output = tmp_path / "rdf_profile.png"

    result = plot_rdf_profile(
        profile,
        output=output,
        show=False,
    )

    assert result == output
    assert output.exists()


def test_plot_orientation_heatmap_profile_forwards_annotations(tmp_path):
    capture_state: dict[str, object] = {}
    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=10,
        n_molecules_per_frame=4,
        bin_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        bin_centers=np.array([0.5, 1.5], dtype=float),
        cos_polar_mean=np.array([0.1, 0.2], dtype=float),
        cos_azimuthal_mean=np.array([0.3, 0.4], dtype=float),
        count_total=np.array([4, 5], dtype=int),
        count_polar_valid=np.array([4, 5], dtype=int),
        count_azimuthal_valid=np.array([4, 5], dtype=int),
        cos_polar_density=np.array([0.01, 0.02], dtype=float),
        cos_azimuthal_density=np.array([0.03, 0.04], dtype=float),
        density=np.array([0.2, 0.3], dtype=float),
        heatmap_polar=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        heatmap_azimuthal=np.array([[4.0, 3.0], [2.0, 1.0]], dtype=float),
        heatmap_angle_bin_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        heatmap_angle_bin_centers=np.array([-0.5, 0.5], dtype=float),
        coordinate_mode="distance",
    )

    result = plot_orientation_profile(
        profile,
        component="heatmap",
        angle="polar",
        output=tmp_path / "orientation_heatmap_annotations.png",
        show=False,
        annotations=[
            {
                "id": "heat-text",
                "type": "text",
                "coord_system": "axes",
                "x": 0.5,
                "y": 1.04,
                "text": "Heatmap annotation",
            }
        ],
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["annotations_summary"][0]["status"] == "ok"
    assert any(text.get_text() == "Heatmap annotation" for text in capture_state["axes"].texts)


def test_plot_multi_line_series_renders_cumulative_when_base_line_is_hidden(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 3.0, 5.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Cumulative",
        x_label="x",
        y_label="y",
        output=tmp_path / "cumulative_hidden_base.png",
        show=False,
        series_enabled=[False],
        series_cumulative_configs=[{"enabled": True, "show_in_legend": True}],
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["series_cumulative_summaries"]["series:a"]["status"] == "ok"
    ax = capture_state["axes"]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "base cumulative average"


def test_plot_multi_line_series_renders_fit_when_base_line_is_hidden(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 3.0, 5.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Fit hidden base",
        x_label="x",
        y_label="y",
        output=tmp_path / "fit_hidden_base.png",
        show=False,
        series_enabled=[False],
        series_fit_configs=[{"fit_enabled": True, "fit_type": "linear"}],
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["series_fit_summaries"]["series:a"]["status"] == "ok"
    ax = capture_state["axes"]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "base fit"


def test_plot_multi_line_series_hides_group_raw_line_when_show_raw_line_is_off(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Group hidden raw",
        x_label="x",
        y_label="y",
        output=tmp_path / "group_hidden_raw.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "base",
                "source_kind": "source",
                "source_series_id": "series:a",
            },
            {
                "series_id": "group:1",
                "default_label": "Group 1",
                "source_kind": "group",
                "member_series_ids": ["series:a"],
                "group_reducer": "mean",
            },
        ],
        series_overrides_by_id={
            "series:a": {"enabled": True, "show_raw_line": False},
            "group:1": {"enabled": True, "show_raw_line": False},
        },
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["series_group_summaries"]["group:1"]["status"] == "ok"
    assert len(capture_state["axes"].lines) == 0


def test_plot_multi_line_series_applies_nested_fit_style_overrides(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 3.0, 5.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Fit style override",
        x_label="x",
        y_label="y",
        output=tmp_path / "fit_style_override.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "base",
                "source_kind": "source",
                "source_series_id": "series:a",
            }
        ],
        series_overrides_by_id={
            "series:a": {
                "fit": {
                    "fit_enabled": True,
                    "fit_type": "linear",
                    "fit_color": "#ff0000",
                    "fit_alpha": 0.4,
                    "fit_line_width": 4.0,
                    "fit_line_style": ":",
                }
            }
        },
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert len(ax.lines) == 2
    fit_line = ax.lines[1]
    assert fit_line.get_color() == "#ff0000"
    assert fit_line.get_alpha() == pytest.approx(0.4)
    assert fit_line.get_linewidth() == pytest.approx(4.0)
    assert fit_line.get_linestyle() == ":"


def test_plot_multi_line_series_autoscales_to_visible_normalized_fit_when_base_line_hidden(
    tmp_path,
):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([100.0, 200.0, 300.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Normalized fit hidden base",
        x_label="x",
        y_label="y",
        output=tmp_path / "normalized_fit_hidden_base.png",
        show=False,
        series_enabled=[False],
        series_fit_configs=[{"fit_enabled": True, "fit_type": "linear"}],
        series_normalization_modes=["factor"],
        series_normalization_values=[0.01],
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    line = ax.lines[0]
    assert np.nanmin(np.asarray(line.get_ydata(), dtype=float)) < 5.0
    assert np.nanmax(np.asarray(line.get_ydata(), dtype=float)) < 5.0
    bottom, top = ax.get_ylim()
    assert bottom < 10.0
    assert top < 10.0


def test_plot_multi_line_series_density_autolimits_ignore_all_zero_tails(tmp_path):
    capture_state: dict[str, object] = {}

    x_values = np.arange(40, dtype=float)
    y_values_a = np.concatenate(
        (np.zeros(10, dtype=float), np.ones(20, dtype=float), np.zeros(10, dtype=float))
    )
    y_values_b = y_values_a * 2.0

    result = plotting_module.plot_multi_line_series(
        [x_values, x_values],
        [y_values_a, y_values_b],
        ["run-a", "run-b"],
        analysis_name="density",
        title="Density zero tails",
        x_label="Distance",
        y_label="Density",
        output=tmp_path / "density_zero_tails.png",
        show=False,
        capture_state=capture_state,
    )

    assert result is not None
    x_lim = capture_state["x_lim"]
    y_lim = capture_state["y_lim"]
    assert isinstance(x_lim, list)
    assert isinstance(y_lim, list)
    assert x_lim[0] > 0.0
    assert x_lim[1] < 39.0
    assert y_lim[0] == pytest.approx(0.0)


def test_plot_multi_line_series_density_autolimits_ignore_zero_tails_with_explicit_ticks(tmp_path):
    capture_state: dict[str, object] = {}

    x_values = np.arange(40, dtype=float)
    y_values = np.concatenate(
        (np.zeros(10, dtype=float), np.ones(20, dtype=float), np.zeros(10, dtype=float))
    )

    result = plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["run-a"],
        analysis_name="density",
        title="Density zero tails with ticks",
        x_label="Distance",
        y_label="Density",
        output=tmp_path / "density_zero_tails_ticks.png",
        show=False,
        x_ticks=[0.0, 10.0, 20.0, 30.0, 40.0],
        capture_state=capture_state,
    )

    assert result is not None
    x_lim = capture_state["x_lim"]
    assert isinstance(x_lim, list)
    assert x_lim[0] > 0.0
    assert x_lim[1] < 39.0


def test_plot_multi_line_series_manual_x_limits_recompute_auto_y_from_visible_data(tmp_path):
    capture_state: dict[str, object] = {}

    x_values = np.array([0.0, 1.0, 2.0, 10.0, 11.0], dtype=float)
    y_values = np.array([1.0, 2.0, 3.0, 100.0, 120.0], dtype=float)

    result = plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["run-a"],
        title="Visible subset y autoscale",
        x_label="x",
        y_label="y",
        output=tmp_path / "visible_subset_y_autoscale.png",
        show=False,
        x_lim=[0.0, 2.0],
        y_lim=[None, None],
        capture_state=capture_state,
    )

    assert result is not None
    y_lim = capture_state["y_lim"]
    assert isinstance(y_lim, list)
    assert y_lim[1] < 10.0


def test_plot_multi_line_series_manual_y_limits_recompute_auto_x_from_visible_data(tmp_path):
    capture_state: dict[str, object] = {}

    x_values = np.array([0.0, 1.0, 2.0, 10.0, 11.0], dtype=float)
    y_values = np.array([1.0, 2.0, 3.0, 100.0, 120.0], dtype=float)

    result = plotting_module.plot_multi_line_series(
        [x_values],
        [y_values],
        ["run-a"],
        title="Visible subset x autoscale",
        x_label="x",
        y_label="y",
        output=tmp_path / "visible_subset_x_autoscale.png",
        show=False,
        x_lim=[None, None],
        y_lim=[0.0, 3.0],
        capture_state=capture_state,
    )

    assert result is not None
    x_lim = capture_state["x_lim"]
    assert isinstance(x_lim, list)
    assert x_lim[1] < 3.0


def test_plot_multi_line_series_autoscales_to_visible_errorbars(tmp_path):
    from linak.analysis.statistics import SeriesStatistics

    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 2.0, 3.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Visible error autoscale",
        x_label="x",
        y_label="y",
        output=tmp_path / "visible_error_autoscale.png",
        show=False,
        series_statistics_data=[
            SeriesStatistics(
                point_count=np.array([10, 10, 10], dtype=int),
                sample_n=np.array([5, 5, 5], dtype=int),
                sample_std=np.array([0.5, 4.0, 8.0], dtype=float),
                sample_sem=np.array([0.25, 2.0, 4.0], dtype=float),
            )
        ],
        series_error_configs=[
            {
                "enabled": True,
                "stat": "sample_std",
                "style": "whiskers",
                "color": "#cc5500",
            }
        ],
        capture_state=capture_state,
    )

    assert result is not None
    bottom, top = capture_state["axes"].get_ylim()
    assert bottom < 1.0
    assert top > 10.0


def test_plot_multi_line_series_renders_error_when_base_line_is_hidden(tmp_path):
    from linak.analysis.statistics import SeriesStatistics

    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 3.0, 5.0], dtype=float)],
        ["base"],
        series_ids=["series:a"],
        title="Error hidden base",
        x_label="x",
        y_label="y",
        output=tmp_path / "error_hidden_base.png",
        show=False,
        series_enabled=[False],
        series_statistics_data=[
            SeriesStatistics(
                point_count=np.array([12, 12, 12], dtype=int),
                sample_n=np.array([5, 5, 5], dtype=int),
                sample_std=np.array([0.2, 0.2, 0.2], dtype=float),
                sample_sem=np.array([0.1, 0.1, 0.1], dtype=float),
            )
        ],
        series_error_configs=[
            {
                "enabled": True,
                "stat": "sample_sem",
                "style": "band",
                "color": "#cc5500",
            }
        ],
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["series_error_summaries"]["series:a"]["status"] == "ok"
    ax = capture_state["axes"]
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


def test_plot_multi_line_series_renders_grouped_mean_series(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 3.0, 5.0], dtype=float),
            np.array([2.0, 4.0, 8.0], dtype=float),
        ],
        ["a", "b"],
        series_ids=["series:a", "series:b"],
        title="Grouped",
        x_label="x",
        y_label="y",
        output=tmp_path / "grouped_mean.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "group:1",
                "default_label": "Grouped mean",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            }
        ],
        capture_state=capture_state,
    )

    assert result is not None
    summary = capture_state["series_group_summaries"]["group:1"]
    assert summary["status"] == "ok"
    ax = capture_state["axes"]
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([1.5, 3.5, 6.5], dtype=float))


def test_plot_multi_line_series_groups_currently_normalized_member_layers(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.array([10.0, 20.0, 30.0], dtype=float),
        ],
        ["a", "b"],
        series_ids=["series:a", "series:b"],
        title="Grouped normalized",
        x_label="x",
        y_label="y",
        output=tmp_path / "grouped_normalized.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "group:1",
                "default_label": "Grouped mean",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            }
        ],
        series_overrides_by_id={
            "series:a": {
                "normalization_mode": "max",
                "normalization_value": 1.0,
            }
        },
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert len(ax.lines) == 1
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        np.array([5.1666666667, 10.3333333333, 15.5], dtype=float),
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_plot_multi_line_series_group_excludes_hidden_member_layers(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 3.0, 5.0], dtype=float),
            np.array([2.0, 4.0, 6.0], dtype=float),
        ],
        ["A", "B"],
        series_ids=["series:a", "series:b"],
        title="Grouped hidden member",
        x_label="x",
        y_label="y",
        output=tmp_path / "group_hidden_member.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "A",
                "source_kind": "source",
                "source_series_id": "series:a",
            },
            {
                "series_id": "series:b",
                "default_label": "B",
                "source_kind": "source",
                "source_series_id": "series:b",
            },
            {
                "series_id": "group:1",
                "default_label": "Grouped",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            },
        ],
        series_overrides_by_id={"series:a": {"enabled": False}},
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert [line.get_label() for line in ax.lines] == ["B", "Grouped"]
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.array([2.0, 4.0, 6.0], dtype=float))


def test_plot_multi_line_series_group_includes_raw_hidden_member_layers(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 3.0, 5.0], dtype=float),
            np.array([2.0, 4.0, 6.0], dtype=float),
        ],
        ["A", "B"],
        series_ids=["series:a", "series:b"],
        title="Grouped raw hidden member",
        x_label="x",
        y_label="y",
        output=tmp_path / "group_raw_hidden_member.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "A",
                "source_kind": "source",
                "source_series_id": "series:a",
            },
            {
                "series_id": "series:b",
                "default_label": "B",
                "source_kind": "source",
                "source_series_id": "series:b",
            },
            {
                "series_id": "group:1",
                "default_label": "Grouped",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            },
        ],
        series_overrides_by_id={"series:a": {"show_raw_line": False}},
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert [line.get_label() for line in ax.lines] == ["B", "Grouped"]
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.array([1.5, 3.5, 5.5], dtype=float))


def test_plot_multi_line_series_renders_generated_copy_from_source_series_id(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [np.array([0.0, 1.0, 2.0], dtype=float)],
        [np.array([1.0, 3.0, 5.0], dtype=float)],
        ["original"],
        series_ids=["series:a"],
        title="Copy",
        x_label="x",
        y_label="y",
        output=tmp_path / "copy_descriptor.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "Original",
                "source_kind": "source",
                "source_series_id": "series:a",
            },
            {
                "series_id": "source:copy",
                "default_label": "Copy",
                "source_kind": "source",
                "source_series_id": "series:a",
                "is_generated": True,
            },
        ],
        series_overrides_by_id={"source:copy": {"label_override": "Copy"}},
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert [line.get_label() for line in ax.lines] == ["Original", "Copy"]
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.array([1.0, 3.0, 5.0], dtype=float))


def test_plot_multi_line_series_hides_disabled_source_render_descriptor(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 3.0, 5.0], dtype=float),
            np.array([2.0, 4.0, 6.0], dtype=float),
        ],
        ["A", "B"],
        series_ids=["series:a", "series:b"],
        title="Hidden source",
        x_label="x",
        y_label="y",
        output=tmp_path / "hidden_source.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "A",
                "source_kind": "source",
                "source_series_id": "series:a",
            },
            {
                "series_id": "series:b",
                "default_label": "B",
                "source_kind": "source",
                "source_series_id": "series:b",
            },
        ],
        series_overrides_by_id={"series:a": {"enabled": False}},
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert [line.get_label() for line in ax.lines] == ["B"]


def test_plot_multi_line_series_keeps_hidden_source_members_out_of_visible_output(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ],
        [
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.array([2.0, 3.0, 4.0], dtype=float),
        ],
        ["A", "B"],
        series_ids=["series:a", "series:b"],
        title="Hidden source member",
        x_label="x",
        y_label="y",
        output=tmp_path / "hidden_member_group.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:a",
                "default_label": "A",
                "source_kind": "source",
                "source_series_id": "series:a",
            },
            {
                "series_id": "series:b",
                "default_label": "B",
                "source_kind": "source",
                "source_series_id": "series:b",
            },
            {
                "series_id": "group:1",
                "default_label": "Grouped",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            },
        ],
        series_overrides_by_id={
            "series:a": {"enabled": False},
            "group:1": {"color": "#cc5500"},
        },
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert [line.get_label() for line in ax.lines] == ["B", "Grouped"]


def test_plot_multi_line_series_rebinned_grouped_path_succeeds_without_strict_zip(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.00, 0.09, 0.21, 0.31], dtype=float),
            np.array([0.02, 0.11, 0.19, 0.29], dtype=float),
        ],
        [
            np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            np.array([1.5, 2.5, 3.5, 4.5], dtype=float),
        ],
        ["a", "b"],
        series_ids=["series:a", "series:b"],
        title="Grouped rebinned",
        x_label="x",
        y_label="y",
        output=tmp_path / "grouped_rebinned.png",
        show=False,
        x_bin_width=0.1,
        x_bin_reducer="mean",
        render_series_descriptors=[
            {
                "series_id": "group:1",
                "default_label": "Grouped mean",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            }
        ],
        capture_state=capture_state,
    )

    assert result is not None
    assert (tmp_path / "grouped_rebinned.png").exists()
    summary = capture_state["series_group_summaries"]["group:1"]
    assert summary["status"] == "ok"


def test_aggregate_grouped_prepared_series_rejects_mismatched_member_lengths():
    prepared = plotting_module.PreparedLineSeries(
        x=np.array([0.0, 1.0, 2.0], dtype=float),
        y=np.array([1.0, 2.0], dtype=float),
        statistics=None,
        available_error_stats=[],
        error_config=plotting_module.SeriesErrorConfig(),
        masked_bin_count=0,
        error_status="unavailable",
        statistics_mode="direct",
    )

    with pytest.raises(ValueError, match="grouped-series x/y point counts do not match"):
        plotting_module._aggregate_grouped_prepared_series(
            [prepared],
            reducer="mean",
            x_bin_width=0.1,
            x_bin_reducer="mean",
        )


def test_plot_multi_line_series_marks_group_unavailable_for_mismatched_x_without_rebin(tmp_path):
    capture_state: dict[str, object] = {}

    result = plotting_module.plot_multi_line_series(
        [
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.5, 1.5, 2.5], dtype=float),
        ],
        [
            np.array([1.0, 3.0, 5.0], dtype=float),
            np.array([2.0, 4.0, 8.0], dtype=float),
        ],
        ["a", "b"],
        series_ids=["series:a", "series:b"],
        title="Grouped mismatch",
        x_label="x",
        y_label="y",
        output=tmp_path / "grouped_mismatch.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "group:1",
                "default_label": "Grouped mean",
                "source_kind": "group",
                "member_series_ids": ["series:a", "series:b"],
                "group_reducer": "mean",
            }
        ],
        capture_state=capture_state,
    )

    assert result is not None
    summary = capture_state["series_group_summaries"]["group:1"]
    assert summary["status"] == "unavailable"
    assert "matching x grid" in summary["reason"]
