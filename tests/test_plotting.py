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


def test_with_style_overrides_updates_axes_border():
    style = plotting_module.with_style_overrides(axes_border=False)

    assert style.axes_border is False
    assert plotting_module.DEFAULT_PLOT_STYLE.axes_border is True


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
    assert "per-frame g(r)" in summary["provenance"]


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
