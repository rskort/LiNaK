import re
from pathlib import Path

from linak.plot.plot_gui import (
    _AUTO_PREVIEW_DEBOUNCE_MS,
    _TOOLTIPS,
    _annotation_display_text_from_entry,
    _annotation_fallback_title,
    _border_setting_to_mode,
    _border_spines_from_setting,
    _current_error_statistics_mode,
    _capture_series_list_view_anchor,
    _coerce_series_error_config,
    _coerce_series_order,
    _coerce_series_descriptors,
    _coerce_series_overrides,
    _default_error_series_label,
    _error_supported_for_view,
    _font_size_placeholder_text,
    _inferred_available_error_stats,
    _derive_synced_field_modes,
    _derive_warning_messages,
    _preview_button_enabled,
    _extract_limit,
    _extract_dict_mode,
    _format_series_display_text,
    _partition_series_ids_by_enabled_state,
    _resolve_error_stat_for_available,
    _resolve_asset_path,
    _resolve_series_id_order,
    _resolve_series_line_colors,
    _restore_series_list_anchor_scroll_value,
    _toggle_to_mode,
    _without_new_profile_series_overrides,
    _without_series_specific_settings,
)


def test_without_series_specific_settings_removes_per_series_keys():
    settings = {
        "title": "Demo",
        "grid": True,
        "line_width": 2.0,
        "series_labels": ["run-a", "run-b"],
        "line_colors": ["#ff0000", "#00ff00"],
        "series_normalization_modes": ["max", "none"],
    }

    filtered = _without_series_specific_settings(settings)

    assert filtered["title"] == "Demo"
    assert filtered["grid"] is True
    assert filtered["line_width"] == 2.0
    assert "series_labels" not in filtered
    assert "line_colors" not in filtered
    assert "series_normalization_modes" not in filtered


def test_auto_preview_debounce_matches_gui_default():
    assert _AUTO_PREVIEW_DEBOUNCE_MS == 1000


def test_preview_button_enabled_only_when_manual_and_not_loading():
    assert _preview_button_enabled(auto_update_enabled=False, preview_loading=False) is True
    assert _preview_button_enabled(auto_update_enabled=True, preview_loading=False) is False
    assert _preview_button_enabled(auto_update_enabled=False, preview_loading=True) is False


def test_all_plot_gui_tooltip_ids_have_registered_messages():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")
    used_tooltip_ids = set(re.findall(r'tooltip_id="([^"]+)"', source))
    assert sorted(used_tooltip_ids - set(_TOOLTIPS)) == []


def test_plot_settings_panel_opens_maximized_by_default():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "window.showMaximized()" in source


def test_plot_settings_panel_uses_item_chooser_for_multi_profile_hdf5_import():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "QInputDialog.getItem(" in source


def test_plot_settings_panel_edits_error_bars_on_base_series_and_keeps_min_bin_points():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '"error::' not in source
    assert '"Color"' in source
    assert "self._series_error_color" in source
    assert '"min_bin_points"' in source


def test_plot_settings_panel_includes_layers_page_with_annotations_and_payload():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '_register_workspace_page("Layers"' in source
    assert "_build_layers_page" in source
    assert 'tabs.addTab(self._tab_annotations, "Annotations")' in source
    assert "def _build_annotations_tab(self) -> None:" in source
    assert '"annotations": annotations_value' in source
    assert '"Add Text"' in source
    assert '"Add Line"' in source
    assert '"Add Arrow"' in source


def test_annotation_fallback_title_prefers_text_preview_and_compact_geometry_labels():
    assert (
        _annotation_fallback_title({"type": "text", "text": "Peak marker"}, index=1)
        == "Peak marker"
    )
    assert (
        _annotation_fallback_title(
            {"type": "line", "x1": "0", "y1": "1", "x2": "2", "y2": "3"},
            index=2,
        )
        == "Line (0, 1 -> 2, 3)"
    )
    assert (
        _annotation_fallback_title(
            {"type": "arrow", "x1": "0.1", "y1": "0.2", "x2": "0.8", "y2": "0.9"},
            index=3,
        )
        == "Arrow (0.1, 0.2 -> 0.8, 0.9)"
    )


def test_annotation_display_text_prefers_name_and_keeps_type_visible():
    assert (
        _annotation_display_text_from_entry(
            {"type": "text", "name": "Peak marker", "enabled": True},
            index=0,
        )
        == "1: Peak marker [Text]"
    )
    assert (
        _annotation_display_text_from_entry(
            {"type": "line", "x1": "0", "y1": "1", "x2": "2", "y2": "3", "enabled": False},
            index=1,
        )
        == "2: Line (0, 1 -> 2, 3) [Line] (off)"
    )


def test_current_error_statistics_mode_tracks_raw_and_rebinned_saved_series():
    assert (
        _current_error_statistics_mode(
            analysis_name="position",
            error_supported=True,
            x_bin_width_active=False,
        )
        == "raw_grouped"
    )
    assert (
        _current_error_statistics_mode(
            analysis_name="rdf",
            error_supported=True,
            x_bin_width_active=True,
        )
        == "saved_rebinned_sample"
    )
    assert (
        _current_error_statistics_mode(
            analysis_name="rdf",
            error_supported=True,
            x_bin_width_active=False,
        )
        == "direct"
    )


def test_inferred_available_error_stats_follow_current_series_mode():
    assert _inferred_available_error_stats(
        analysis_name="position",
        error_supported=True,
        x_bin_width_active=False,
    ) == ["sample_std", "sample_sem"]
    assert _inferred_available_error_stats(
        analysis_name="rdf",
        error_supported=True,
        x_bin_width_active=True,
    ) == ["sample_std", "sample_sem"]
    assert _inferred_available_error_stats(
        analysis_name="rdf",
        error_supported=True,
        x_bin_width_active=False,
    ) == ["sample_std", "sample_sem", "block_std", "block_sem"]


def test_plot_settings_panel_shows_error_provenance_explanation_text():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self._series_error_explanation = QLabel("")' in source
    assert "What it shows:" in source
    assert "What it would show:" in source


def test_plot_settings_panel_includes_cumulative_controls_and_group_actions():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '"Duplicate"' in source
    assert '"Add Group"' in source
    assert 'QGroupBox("Cumulative Average")' in source
    assert "self._series_cumulative_mode" in source
    assert "self._series_group_reducer" in source
    assert "self._series_group_members" in source
    assert '"group_reducer"' in source
    assert '"member_series_ids"' in source


def test_plot_settings_panel_hides_inactive_optional_layer_details():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self._series_error_detail_rows" in source
    assert "self._set_rows_visible(self._series_error_detail_rows, error_active)" in source
    assert "self._series_fit_detail_rows" in source
    assert "self._set_form_row_visible(form, field, visible)" in source
    assert "self._series_cumulative_detail_rows" in source
    assert (
        "self._set_rows_visible(self._series_cumulative_detail_rows, cumulative_active)" in source
    )
    assert "self._normalization_actions_widget" in source
    assert "widget.setVisible(norm_enabled)" in source
    assert 'norm_x_ref_enabled = norm_mode == "value_at_x"' in source


def test_plot_settings_panel_hides_inactive_figure_data_and_annotation_details():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self._title_detail_widgets" in source
    assert "widget.setVisible(title_enabled)" in source
    assert "widget.setVisible(x_label_enabled)" in source
    assert "widget.setVisible(y_label_enabled)" in source
    assert "self._set_rows_visible(self._legend_rows, legend_enabled)" in source
    assert "self._set_rows_visible(self._grid_rows, grid_enabled)" in source
    assert "self._set_rows_visible(self._x_ticks_rows, x_ticks_enabled)" in source
    assert "self._set_rows_visible(self._y_ticks_rows, y_ticks_enabled)" in source
    assert "self._set_rows_visible(self._marker_rows, markers_enabled)" in source
    assert "self._set_rows_visible(self._colorbar_rows, colorbar_enabled)" in source
    assert "self._set_form_row_visible(\n                    self._x_bin_reducer_row[0]" in source
    assert "self._set_form_row_visible(\n                    self._y_bin_reducer_row[0]" in source
    assert "self._annotation_common_detail_rows" in source
    assert 'annotation_enabled and annotation_type == "text"' in source
    assert "self._annotation_summary_group.setVisible(annotation_enabled)" in source


def test_plot_settings_panel_uses_task_first_workspace_pages():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    profiles_index = source.index('_register_workspace_page("Profiles"')
    data_index = source.index('_register_workspace_page("Data"')
    layers_index = source.index('_register_workspace_page("Layers"')
    figure_index = source.index('_register_workspace_page("Figure"')
    advanced_index = source.index('_register_workspace_page("Advanced"')

    assert profiles_index < data_index < layers_index < figure_index < advanced_index
    assert '_register_workspace_page("Export"' not in source
    assert '_register_workspace_page("Overview"' not in source
    assert '_register_workspace_page("Series"' not in source


def test_plot_settings_panel_flattens_figure_controls_into_one_scroll_page():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'QGroupBox("Canvas & Typography")' in source
    assert 'QGroupBox("Lines & Markers")' in source
    assert 'QGroupBox("Axes, Ticks & Grid")' in source
    assert 'QGroupBox("Legend")' in source
    assert 'QGroupBox("Heatmap & Colorbar")' in source
    assert 'QGroupBox("Grid")' in source
    assert 'QGroupBox("Border")' in source
    assert 'tabs.addTab(text_legend_tab, "Text && Legend")' not in source
    assert 'tabs.addTab(axes_limits_tab, "Axes && Limits")' not in source
    assert 'tabs.addTab(ticks_grid_tab, "Ticks && Grid")' not in source
    assert 'tabs.addTab(style_tab, "Style")' not in source


def test_plot_settings_panel_keeps_export_in_preview_toolbar_without_transparent_save():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'transparent_label = QLabel("Transparent save")' not in source
    assert "self.transparent_mode" not in source
    assert 'self.figure_alpha = self._bounded_float_line("0.0 - 1.0"' in source
    assert 'self.save_figure_button = QPushButton("Export Figure")' in source
    assert "header_layout.addWidget(self._save_figure_button)" not in source


def test_plot_settings_panel_supports_detachable_preview_window():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "class _PreviewPane(QFrame)" in source
    assert "class _DetachedPreviewWindow(QMainWindow)" in source
    assert 'self.detach_button = QPushButton("Detach Preview")' in source
    assert 'self.dock_button = QPushButton("Dock Back")' in source
    assert 'self.setWindowTitle("LiNaK Figure Preview")' in source
    assert "self._embedded_preview_pane.setVisible(False)" in source
    assert "detached_window.close_from_dock()" in source
    assert "QTimer.singleShot(0, self._on_dock_requested)" in source


def test_plot_settings_panel_keeps_status_outside_hidden_preview_panel():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "root_layout.addWidget(self._status_label)" in source
    assert "right_layout.addWidget(self._status_label)" not in source
    assert "QFrame#detachedPreviewPanel" in source


def test_plot_settings_panel_exposes_per_axis_ticks_and_color_controls():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self.x_ticks_mode = self._combo(_TOGGLE_MODES)" in source
    assert "self.y_ticks_mode = self._combo(_TOGGLE_MODES)" in source
    assert "self.x_tick_direction = self._combo(_TICK_DIRECTIONS)" in source
    assert "self.y_tick_direction = self._combo(_TICK_DIRECTIONS)" in source
    assert "self.x_label_font = self._positive_int_line()" in source
    assert "self.y_label_font = self._positive_int_line()" in source
    assert 'tooltip_id="figure.canvas.font_color"' in source
    assert 'tooltip_id="figure.lines.marker_color"' in source
    assert '"_x_tick_params"' in source
    assert '"_y_tick_params"' in source


def test_plot_settings_panel_keeps_reset_as_a_profile_action():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '_page_button("Reset Profile to Defaults", self._handle_reset)' in source
    assert 'self._reset_button = QPushButton("Reset to Defaults")' not in source


def test_plot_settings_panel_removes_preview_badge_strip_and_warning_banner():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "status_strip = QFrame(right_panel)" not in source
    assert 'self._warning_summary_label = QLabel("")' not in source


def test_plot_settings_panel_gives_annotation_list_a_dedicated_object_name():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self.annotation_list.setObjectName("annotationList")' in source
    assert "QListWidget#annotationList::item" in source


def test_plot_settings_panel_uses_cumulative_child_ids_in_series_list():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "cumulative::" in source
    # Base rows now show inline badges instead of child rows in the series list
    assert "\u00b7" in source  # middle dot badge separator is present


def test_orientation_gui_offers_density_line_view():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '("average", "density", "density-weighted", "heatmap")' in source


def test_without_series_specific_settings_keeps_non_series_limits():
    settings = {
        "x_lim": [0.0, 10.0],
        "y_lim": [0.0, 1.0],
        "border": False,
        "font_size": 15,
        "font_family": "DejaVu Sans",
        "series_enabled": [True, False],
    }

    filtered = _without_series_specific_settings(settings)

    assert filtered["x_lim"] == [0.0, 10.0]
    assert filtered["y_lim"] == [0.0, 1.0]
    assert filtered["border"] is False
    assert filtered["font_size"] == 15
    assert filtered["font_family"] == "DejaVu Sans"
    assert "series_enabled" not in filtered


def test_font_size_placeholder_text_includes_effective_size_in_brackets():
    assert _font_size_placeholder_text(14) == "Auto (14)"


def test_resolve_error_stat_for_available_prefers_available_semantics():
    assert (
        _resolve_error_stat_for_available("block_sem", ["sample_sem", "sample_std"]) == "sample_sem"
    )
    assert (
        _resolve_error_stat_for_available("sample_std", ["sample_sem", "sample_std"])
        == "sample_std"
    )
    assert _resolve_error_stat_for_available(None, ["block_std"]) == "block_std"


def test_default_error_series_label_uses_effective_stat_name():
    assert _default_error_series_label("H2O density", "sample_sem") == "H2O density sample_sem"


def test_error_supported_for_view_tracks_one_dimensional_modes_only():
    assert _error_supported_for_view("rdf") is True
    assert _error_supported_for_view("orientation", orientation_heatmap=True) is False
    assert _error_supported_for_view("position", position_component="xy-z") is False
    assert (
        _error_supported_for_view("coordination", coordination_component="time-distance") is False
    )


def test_without_new_profile_series_overrides_resets_series_customizations_only():
    settings = {
        "title": "Demo",
        "series_labels": ["Au", "H2O"],
        "series_descriptors": [{"series_id": "a"}, {"series_id": "b"}],
        "series_overrides": {"a": {"enabled": False}},
        "series_enabled": [False, True],
        "line_colors": ["#ff0000", "#00ff00"],
        "series_markers": ["o", ""],
    }

    cleaned = _without_new_profile_series_overrides(settings)

    assert cleaned["title"] == "Demo"
    assert cleaned["series_labels"] == ["Au", "H2O"]
    assert cleaned["series_descriptors"] == [{"series_id": "a"}, {"series_id": "b"}]
    assert "series_overrides" not in cleaned
    assert "series_enabled" not in cleaned
    assert "line_colors" not in cleaned
    assert "series_markers" not in cleaned


def test_resolve_series_line_colors_keeps_none_when_all_blank():
    assert _resolve_series_line_colors(["", "   "]) is None


def test_resolve_series_line_colors_preserves_blank_entries():
    resolved = _resolve_series_line_colors(["", "#ff5500", ""])

    assert resolved == ["", "#ff5500", ""]


def test_format_series_display_text_marks_disabled_entries():
    assert _format_series_display_text(0, "Au", enabled=True) == "1: Au"
    assert _format_series_display_text(1, "H2O", enabled=False) == "2: H2O (off)"


def test_coerce_series_descriptors_preserves_identity_and_metadata():
    descriptors = _coerce_series_descriptors(
        [
            {
                "series_id": "series:0:3",
                "default_label": "O",
                "source_name": "density.h5",
                "source_directory": "runs/run_04",
                "source_path": "/tmp/runs/run_04/density.h5",
            }
        ]
    )

    assert descriptors == [
        {
            "series_id": "series:0:3",
            "default_label": "O",
            "source_kind": "source",
            "source_series_id": None,
            "member_series_ids": [],
            "group_reducer": "mean",
            "source_name": "density.h5",
            "source_directory": "runs/run_04",
            "source_path": "/tmp/runs/run_04/density.h5",
            "source_index": None,
            "series_index": None,
        }
    ]


def test_coerce_series_overrides_keeps_sparse_label_override_mapping():
    overrides = _coerce_series_overrides(
        {
            "series:0:3": {
                "label_override": "H2O",
                "enabled": False,
            }
        }
    )

    assert overrides["series:0:3"]["label_override"] == "H2O"
    assert overrides["series:0:3"]["enabled"] is False


def test_coerce_series_error_config_keeps_color_override():
    config = _coerce_series_error_config(
        {
            "enabled": True,
            "stat": "sample_std",
            "style": "whiskers",
            "color": "#cc5500",
            "label_override": "error",
            "show_in_legend": True,
        }
    )

    assert config["enabled"] is True
    assert config["stat"] == "sample_std"
    assert config["style"] == "whiskers"
    assert config["color"] == "#cc5500"
    assert config["label_override"] == "error"
    assert config["show_in_legend"] is True


def test_coerce_series_order_deduplicates_and_strips_values():
    assert _coerce_series_order([" a ", "", "b", "a", "b", "c"]) == ["a", "b", "c"]


def test_resolve_series_id_order_appends_new_series_after_requested_order():
    assert _resolve_series_id_order(
        ["series-a", "series-b", "series-c"],
        ["series-c", "series-a", "missing"],
    ) == ["series-c", "series-a", "series-b"]


def test_partition_series_ids_by_enabled_state_preserves_current_relative_order():
    resolved = _partition_series_ids_by_enabled_state(
        ["series-c", "series-a", "series-b", "series-d"],
        {
            "series-c": True,
            "series-a": False,
            "series-b": True,
            "series-d": False,
        },
    )

    assert resolved == ["series-c", "series-b", "series-a", "series-d"]


def test_capture_series_list_view_anchor_uses_top_visible_row_and_offset():
    anchor = _capture_series_list_view_anchor(
        [
            ("series-a", -12, 8),
            ("series-b", 8, 28),
            ("series-c", 28, 48),
        ],
        viewport_height=40,
        scroll_value=32,
    )

    assert anchor == {
        "row_id": "series-a",
        "offset": -12,
        "scroll_value": 32,
    }


def test_restore_series_list_anchor_scroll_value_preserves_anchor_offset_after_reorder():
    restored = _restore_series_list_anchor_scroll_value(
        {
            "row_id": "series-a",
            "offset": -12,
            "scroll_value": 32,
        },
        row_tops={
            "series-b": 0,
            "series-c": 20,
            "series-a": 40,
        },
        current_scroll_value=0,
        maximum=200,
    )

    assert restored == 52


def test_restore_series_list_anchor_scroll_value_falls_back_to_previous_scroll_when_missing():
    restored = _restore_series_list_anchor_scroll_value(
        {
            "row_id": "series-missing",
            "offset": -12,
            "scroll_value": 32,
        },
        row_tops={"series-a": 0},
        current_scroll_value=0,
        maximum=200,
    )

    assert restored == 32


def test_derive_synced_field_modes_prefers_explicit_auto_metadata():
    modes = _derive_synced_field_modes(
        {
            "_gui_sync_modes": {"x_label": "manual", "y_label": "auto"},
            "x_label": None,
            "y_label": "Density",
            "x_lim": [0.0, 5.0],
            "x_ticks": [0.0, 1.0, 2.0],
        }
    )

    assert modes["x_label"] == "manual"
    assert modes["y_label"] == "auto"
    assert modes["x_lim"] == "manual"
    assert modes["x_ticks"] == "manual"


def test_derive_synced_field_modes_detects_explicit_limits_ticks_and_label_padding():
    modes = _derive_synced_field_modes(
        {
            "title": "",
            "x_label": None,
            "y_label": "",
            "x_lim": [0.0, 5.0],
            "y_lim": [None, None],
            "x_ticks": [0.0, 1.0, 2.0],
            "y_ticks": None,
            "x_label_pad": 6.0,
        }
    )

    assert modes == {
        "title": "manual",
        "x_label": "auto",
        "y_label": "manual",
        "x_lim": "manual",
        "y_lim": "auto",
        "x_ticks": "manual",
        "y_ticks": "auto",
        "x_label_pad": "manual",
        "y_label_pad": "auto",
    }


def test_derive_synced_field_modes_maps_hidden_title_to_off_and_merges_partial_metadata():
    modes = _derive_synced_field_modes(
        {
            "_gui_sync_modes": {"x_label": "manual"},
            "title_visible": False,
            "title": None,
            "x_label": None,
            "y_label": "Density",
        }
    )

    assert modes["title"] == "off"
    assert modes["x_label"] == "manual"
    assert modes["y_label"] == "manual"
    assert modes["x_lim"] == "auto"


def test_auto_manual_sync_uses_one_mode_map_and_no_locked_field_metadata():
    gui_source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")
    cli_source = Path("src/linak/cli.py").read_text(encoding="utf-8")

    assert "_synced_field_locks" not in gui_source
    assert "_derive_synced_field_locks" not in gui_source
    assert "_gui_locked_fields" not in cli_source
    assert "for key in _SYNCED_FIELD_KEYS" in gui_source
    assert 'if self._synced_field_mode(key) != "manual":' in gui_source
    assert 'if self._synced_field_mode(key) != "auto"' in gui_source
    assert 'if title_mode == "off":' in gui_source


def test_extract_limit_rounds_auto_values_for_gui_display():
    settings = {"x_lim": [0.123456789, 12.0]}

    assert _extract_limit(settings, key="x_lim", index=0) == "0.123457"
    assert _extract_limit(settings, key="x_lim", index=1) == "12"


def test_toggle_to_mode_materializes_auto_to_effective_default():
    assert _toggle_to_mode(None, auto_mode="on") == "on"
    assert _toggle_to_mode(None, auto_mode="off") == "off"


def test_border_setting_to_mode_handles_bool_and_dict():
    assert _border_setting_to_mode(True) == "on"
    assert _border_setting_to_mode(None) == "on"
    assert _border_setting_to_mode(False) == "off"
    assert _border_setting_to_mode({"left": True, "right": False}) == "custom"


def test_border_spines_from_setting_extracts_per_side():
    result = _border_spines_from_setting(
        {"left": False, "right": True, "top": True, "bottom": False}
    )
    assert result == {"left": False, "right": True, "top": True, "bottom": False}

    result_false = _border_spines_from_setting(False)
    assert all(not v for v in result_false.values())
    assert set(result_false.keys()) == {"left", "right", "top", "bottom"}

    result_true = _border_spines_from_setting(True)
    assert all(v for v in result_true.values())

    # Missing keys default to True
    result_partial = _border_spines_from_setting({"left": False})
    assert result_partial["left"] is False
    assert result_partial["right"] is True


def test_extract_dict_mode_materializes_missing_bool_to_effective_default():
    assert _extract_dict_mode({}, key="legend_kwargs", nested_key="frameon", auto_mode="on") == "on"
    assert (
        _extract_dict_mode({}, key="savefig_kwargs", nested_key="transparent", auto_mode="off")
        == "off"
    )


def test_derive_warning_messages_reports_disabled_sections_and_partial_normalization():
    warnings = _derive_warning_messages(
        {
            "legend": False,
            "legend_title": "Runs",
            "grid": False,
            "grid_alpha": 0.3,
            "ticks": False,
            "tick_params_kwargs": {"direction": "out"},
            "series_normalization_modes": ["none", "max"],
        }
    )

    assert all("Legend is off" not in message for message in warnings)
    assert all("Grid is off" not in message for message in warnings)
    assert all("Ticks are off" not in message for message in warnings)
    assert any("Only part of the plotted series is normalized" in message for message in warnings)


def test_derive_warning_messages_ignores_advanced_json_overlap():
    warnings = _derive_warning_messages({"legend": True})

    assert all(
        "Advanced JSON overlaps with standard controls" not in message for message in warnings
    )


def test_resolve_asset_path_prefers_repo_root_assets(tmp_path):
    module_path = tmp_path / "project" / "src" / "linak" / "plot" / "plot_gui.py"
    expected = tmp_path / "project" / "assets" / "linak_gui_banner.svg"
    expected.parent.mkdir(parents=True)
    expected.write_text("<svg />", encoding="utf-8")

    resolved = _resolve_asset_path("linak_gui_banner.svg", module_path=module_path)

    assert resolved == expected.resolve()


def test_resolve_asset_path_finds_installed_share_assets(tmp_path):
    module_path = tmp_path / "venv" / "Lib" / "site-packages" / "linak" / "plot" / "plot_gui.py"
    expected = tmp_path / "venv" / "share" / "linak" / "assets" / "linak_gui_banner.svg"
    expected.parent.mkdir(parents=True)
    expected.write_text("<svg />", encoding="utf-8")

    resolved = _resolve_asset_path("linak_gui_banner.svg", module_path=module_path)

    assert resolved == expected.resolve()
