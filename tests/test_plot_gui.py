import re
from pathlib import Path

from linak.plot.data_contract import PlotViewMapping
from linak.plot.plot_gui import (
    _AUTO_PREVIEW_DEBOUNCE_MS,
    _TOOLTIPS,
    _annotation_defaults_for_gui,
    _annotation_display_text_from_entry,
    _annotation_fallback_title,
    _border_setting_to_mode,
    _border_spines_from_setting,
    _current_error_statistics_mode,
    _capture_series_list_view_anchor,
    _coordination_backend_summary_text,
    _coordination_mapping_preset_label,
    _coerce_series_error_config,
    _coerce_series_order,
    _coerce_series_descriptors,
    _coerce_series_overrides,
    _default_error_series_label,
    _density_backend_summary_text,
    _error_supported_for_view,
    _font_size_placeholder_text,
    _inferred_available_error_stats,
    _derive_synced_field_modes,
    _derive_warning_messages,
    _preview_button_enabled,
    _extract_limit,
    _extract_dict_mode,
    _format_series_display_text,
    _orientation_backend_summary_text,
    _potential_backend_summary_text,
    _partition_series_ids_for_display_order,
    _resolve_error_stat_for_available,
    _resolve_asset_path,
    _resolve_series_id_order,
    _resolve_series_line_colors,
    _restore_series_list_anchor_scroll_value,
    _settings_use_heatmap_rendering,
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


def test_plot_settings_panel_exposes_header_undo_button_and_session_history():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self._undo_button = QPushButton("Undo")' in source
    assert "self._undo_button.clicked.connect(self._handle_undo)" in source
    assert 'self._register_tooltip(self._undo_button, "profiles.undo")' in source
    assert "self._undo_stack: list[dict[str, Any]] = []" in source
    assert "self._redo_stack: list[dict[str, Any]] = []" in source
    assert 'self._redo_button = QPushButton("Redo")' in source
    assert "self._redo_button.clicked.connect(self._handle_redo)" in source
    assert 'self._register_tooltip(self._redo_button, "profiles.redo")' in source
    assert 'self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)' in source
    assert 'self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)' in source
    assert "def _record_history_after_non_text_change(self, *_unused: object) -> None:" in source
    assert "def _handle_undo(self) -> None:" in source
    assert "def _handle_redo(self) -> None:" in source
    assert "def _begin_text_undo_edit(self, widget: QWidget | None) -> None:" in source
    assert "def _finalize_text_undo_edit(self, widget: QWidget | None = None) -> None:" in source
    assert '"_undo_preview_state"' in source
    assert "def _history_snapshot(self, settings: dict[str, Any]) -> dict[str, Any]:" in source


def test_plot_settings_panel_imports_qshortcut_from_qtgui():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    qtgui_block_start = source.index("from PySide6.QtGui")
    qtwidgets_block_start = source.index("from PySide6.QtWidgets")
    qtwidgets_block_end = source.index(")", qtwidgets_block_start)

    assert "QShortcut" in source[qtgui_block_start:qtwidgets_block_start]
    assert "QShortcut" not in source[qtwidgets_block_start:qtwidgets_block_end]


def test_density_view_type_fallback_is_line_only_without_heatmap_sources():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'density_view_type_labels = (_DENSITY_VIEW_TYPE_LABEL_BY_ID["line_1d"],)' in source
    assert "tuple(_DENSITY_VIEW_TYPE_LABEL_BY_ID.values())" not in source


def test_plot_settings_panel_event_filter_uses_qt_modifier_flags_directly():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    event_filter_index = source.rindex("def eventFilter(self, watched: Any, event: Any) -> bool:")
    event_filter_block = source[event_filter_index : event_filter_index + 1800]

    assert "modifiers = event.modifiers()" in event_filter_block
    assert "int(event.modifiers())" not in event_filter_block
    assert "modifiers & Qt.KeyboardModifier.ControlModifier" in event_filter_block
    assert "modifiers & Qt.KeyboardModifier.AltModifier" in event_filter_block


def test_plot_settings_panel_no_longer_discovers_undo_history_during_shell_refresh():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    refresh_shell_state_index = source.index("def _refresh_shell_state(self) -> None:")
    update_header_state_index = source.index(
        "self._update_header_state()", refresh_shell_state_index
    )
    assert "_record_undo_snapshot_if_needed" not in source
    assert (
        "self._record_history_after_non_text_change()"
        not in source[refresh_shell_state_index : update_header_state_index + 200]
    )


def test_plot_settings_panel_populate_resets_all_parallel_series_style_arrays():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self._series_fit_color_data = []" in source
    assert "self._series_fit_alpha_data = []" in source
    assert "self._series_fit_line_width_data = []" in source
    assert "self._series_fit_line_style_data = []" in source
    assert "self._series_cumulative_color_data = []" in source
    assert "self._series_cumulative_alpha_data = []" in source
    assert "self._series_cumulative_line_width_data = []" in source
    assert "self._series_cumulative_line_style_data = []" in source
    assert "self._series_integration_enabled_data = []" in source
    assert "self._series_integration_source_data = []" in source
    assert "self._series_integration_x_min_data = []" in source
    assert "self._series_integration_x_max_data = []" in source
    assert "self._series_integration_baseline_data = []" in source
    assert "self._series_integration_color_mode_data = []" in source
    assert "self._series_integration_color_data = []" in source
    assert "self._series_integration_alpha_data = []" in source


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


def test_plot_settings_panel_duplicates_profiles_without_reload_or_preview():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")
    assert "on_duplicate_profile: Callable[[str, str], str] | None = None" in source
    handler_match = re.search(
        r"def _handle_duplicate_profile\(self\) -> None:(.*?)"
        r"\n        def _handle_rename_profile",
        source,
        re.DOTALL,
    )
    assert handler_match is not None
    handler = handler_match.group(1)

    assert "on_save(current_name, settings)" in handler
    assert "on_duplicate_profile(current_name, name)" in handler
    assert "self._set_profile_names([*self._profile_names, name], active_name=name)" in handler
    assert "self._saved_signature = self._signature(settings)" in handler
    assert "_load_settings_into_editor" not in handler
    assert "_schedule_preview_update" not in handler


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


def test_plot_settings_panel_uses_reusable_collapsible_sections_with_session_state():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "class _CollapsibleSection(QFrame):" in source
    assert "QPropertyAnimation" in source
    assert "QEasingCurve.Type.OutCubic" in source
    assert "self._collapsible_section_state: dict[str, bool] = {}" in source
    assert "self._state_store[self._section_id] = self._expanded" in source
    assert "self.toggle_button.setSizePolicy(" in source
    assert "QSizePolicy.Policy.Expanding" in source
    assert "alignment=Qt.AlignmentFlag.AlignVCenter" in source
    assert (
        "header_layout.addWidget(\n                self.toggle_button,\n                stretch=1,"
        in source
    )
    assert '"_collapsible_section_state"' not in source


def test_plot_settings_panel_themes_message_boxes_and_dialog_buttons():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'f"QMessageBox {{' in source
    assert 'f"QMessageBox QLabel {{' in source
    assert 'f"QMessageBox QTextEdit, QMessageBox QPlainTextEdit {{' in source
    assert 'f"QMessageBox QPushButton {{' in source
    assert 'f"QDialogButtonBox QPushButton {{' in source


def test_plot_settings_panel_blocks_series_and_annotation_sync_during_initial_populate():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "_previous_series_syncing = self._series_syncing" in source
    assert "_previous_annotation_syncing = self._annotation_syncing" in source
    assert "self._series_syncing = True" in source
    assert "self._annotation_syncing = True" in source
    assert "self._populate(initial_settings)" in source


def test_plot_settings_panel_maps_collapsible_section_ids_for_layers_and_figure():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    for token in (
        'section_id="layers.visibility"',
        'section_id="layers.style"',
        'section_id="layers.integration"',
        'section_id="layers.derived"',
        'section_id="layers.derived.uncertainty"',
        'section_id="layers.derived.fit"',
        'section_id="layers.derived.cumulative"',
        'section_id="layers.group_members"',
        'section_id="layers.normalization"',
        'section_id="layers.metadata"',
        'section_id="figure.canvas"',
        'section_id="figure.lines"',
        'section_id="figure.axes"',
        'section_id="figure.axes.title"',
        'section_id="figure.axes.x"',
        'section_id="figure.axes.x_ticks"',
        'section_id="figure.axes.y"',
        'section_id="figure.axes.y_ticks"',
        'section_id="figure.axes.grid"',
        'section_id="figure.axes.border"',
        'section_id="figure.legend"',
        'section_id="figure.heatmap"',
        'section_id="figure.heatmap.rendering"',
        'section_id="figure.heatmap.colorbar"',
    ):
        assert token in source


def test_plot_settings_panel_keeps_group_normalization_live_and_editable():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '"Copied normalization settings to all layers."' in source
    assert "grouped series aggregate already-transformed member series" not in source
    assert "non-group base series only" not in source
    assert "show_normalization=is_line_family," in source
    assert "self._normalization_group.setEnabled(layer_caps.show_normalization)" in source


def test_plot_settings_panel_can_copy_none_normalization_to_all_layers():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'if mode == "none":\n                value = ""\n                x_ref = ""' in source
    assert "widget.setVisible(normalization_actions_visible)" in source
    assert "Turn normalization on first." not in source

    copy_enabled_match = re.search(
        r"normalization_copy_enabled = \((.*?)\)\n\s+self\._normalization_copy_button\.setEnabled",
        source,
        re.DOTALL,
    )
    assert copy_enabled_match is not None
    copy_enabled_expression = copy_enabled_match.group(1)
    assert "norm_enabled" not in copy_enabled_expression
    assert "layer_caps.show_normalization" in copy_enabled_expression
    assert "not is_heatmap" in copy_enabled_expression
    assert "not self._series_active_is_fit_child" in copy_enabled_expression
    assert "not self._series_active_is_cumulative_child" in copy_enabled_expression


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
    assert 'QGroupBox("Preview Summary")' not in source
    assert "_annotation_preview_summary" not in source
    assert "annotations.summary" not in source
    assert "header_widget=" not in source
    assert '"Legend",' in source
    assert '"Show ticks",' in source
    assert '"Show grid",' in source
    assert '"Colorbar",' in source


def test_plot_settings_panel_keeps_x_y_label_font_sizes_independent():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '(self.x_label_font, "x_label_font_size")' in source
    assert '(self.y_label_font, "y_label_font_size")' in source
    assert '"label_font_size": shared_label_font_size_value' in source
    assert '"x_label_font_size": x_label_font_size_value' in source
    assert '"y_label_font_size": y_label_font_size_value' in source
    assert '"tick_font_size": shared_tick_font_size_value' in source


def test_plot_settings_panel_uses_explicit_series_row_theme_tokens():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '"series_row_selected_bg"' in source
    assert '"series_row_selected_border"' in source
    assert '"series_row_selected_text"' in source
    assert '"series_badge_original_bg"' in source
    assert '"series_badge_copy_bg"' in source
    assert '"series_badge_group_bg"' in source
    assert "QToolButton {" in source
    assert '"tooltip_bg"' in source
    assert '"tooltip_text"' in source
    assert '"tooltip_border"' in source
    assert '"placeholder_text"' in source
    assert "QToolTip {" in source
    assert "QMenu {" in source
    assert "QMenu::item:disabled {" in source
    assert "placeholder-text-color:" in source
    assert "QComboBox QAbstractItemView::item {" in source
    assert "show-decoration-selected: 1;" in source
    assert "QAbstractItemView::item:selected {" in source
    assert "QAbstractItemView::indicator:checked {" in source
    assert "QScrollBar:horizontal {" in source
    assert "theme=self._theme_tokens()" in source
    assert "self.palette().color(QPalette.ColorRole.Text)" not in source
    assert "self.palette().color(QPalette.ColorRole.Mid)" not in source
    assert "_win.lightness() < _wtxt.lightness()" not in source


def test_plot_settings_panel_rebuilds_series_list_with_safe_widget_detach():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "def _clear_series_list_widget_items(self) -> None:" in source
    assert "self.series_list.removeItemWidget(item)" in source
    assert "widget.deleteLater()" in source
    assert "old_signal_block = self.series_list.blockSignals(True)" in source
    assert "old_model_block = model.blockSignals(True)" in source
    assert (
        'self._set_active_series_child_kind("base")\n            self._clone_series_at_index'
        in source
    )
    assert "(self._series_fit_enabled_data, False)" in source


def test_plot_settings_panel_centralizes_parallel_series_state_management():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "def _series_state_attr_names(self) -> tuple[str, ...]:" in source
    assert "def _iter_series_state_lists(self) -> list[tuple[str, list[Any]]]:" in source
    assert "def _validate_series_state_lengths(self) -> None:" in source
    assert "def _effective_series_state(self, index: int) -> dict[str, Any]:" in source
    assert "for name, values in self._iter_series_state_lists():" in source
    assert "Internal GUI layer state is inconsistent:" in source
    assert "self._series_show_raw_line_data.append(True)" in source


def test_plot_settings_panel_has_selected_layer_identity_card_and_generated_delete():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self._selected_layer_card.setObjectName("selectedLayerCard")' in source
    assert 'self._selected_layer_title = QLabel("No layer selected")' in source
    assert 'self._series_delete_button = QPushButton("Delete Layer")' in source
    assert "def _series_is_generated(self, index: int) -> bool:" in source
    assert "Original data series cannot be deleted here; turn them off instead." in source
    assert "def _delete_series_at_index(self, index: int) -> None:" in source
    assert "def _default_generated_series_color(self, index: int) -> str:" in source
    assert "if original_source_ids:" in source
    assert "if self._series_is_generated(index):" in source
    assert 'entry["color"] = generated_color' in source


def test_plot_settings_panel_repartitions_layer_order_after_visibility_and_group_changes():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "def _partition_series_ids_for_display_order(" in source
    assert "enabled_non_group_ids = [" in source
    assert "enabled_group_ids = [" in source
    assert "disabled_ids = [" in source
    assert "self._apply_series_id_order(self._enabled_partitioned_series_id_order())" in source


def test_plot_settings_panel_keeps_base_series_checkbox_enabled_when_row_is_off():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self.checkbox.setEnabled(kind == "base")' in source


def test_plot_settings_panel_keeps_base_row_toggle_logic_outside_cumulative_branch():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'elif row_kind == "cumulative":' in source
    assert "else:\n                if checked and self._is_orientation_heatmap_mode():" in source
    assert "self._series_enabled_data[index] = checked" in source


def test_plot_settings_panel_uses_checkable_group_members_under_layer_list():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    selected_card_index = source.index("tab_layout.addWidget(self._selected_layer_card)")
    selector_row_index = source.index("layout.addLayout(selector_row)")
    series_list_index = source.index("layout.addWidget(self.series_list)")
    group_member_index = source.index("layout.addWidget(self._series_group_group)")
    visibility_index = source.index(
        "self._series_visibility_group = self._make_collapsible_section("
    )

    assert (
        selected_card_index
        < selector_row_index
        < series_list_index
        < group_member_index
        < visibility_index
    )
    assert "QAbstractItemView.SelectionMode.NoSelection" in source
    assert "self._series_group_members.itemChanged.connect" in source
    assert "Qt.ItemFlag.ItemIsUserCheckable" in source
    assert "item.checkState() == Qt.CheckState.Checked" in source


def test_plot_settings_panel_busts_preview_pixmap_cache_before_reload():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "QPixmapCache," in source
    assert "QPixmapCache.remove(str(self._preview_image_path))" in source
    assert "QPixmapCache.remove(str(image_path_obj))" in source
    assert "if not pixmap.load(str(image_path_obj)):" in source


def test_plot_settings_panel_composites_preview_transparency_on_theme_matte():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "def _preview_transparency_matte_colors(self) -> tuple[QColor, QColor]:" in source
    assert 'if self._theme_mode == "dark":' in source
    assert 'return QColor("#8793a3"), QColor("#aeb7c3")' in source
    assert 'return QColor("#f4f7fb"), QColor("#dce4ee")' in source
    assert "def _preview_display_pixmap(self, pixmap: QPixmap) -> QPixmap:" in source
    assert "painter.fillRect(composed.rect(), matte_a)" in source
    assert "painter.setBrush(QBrush(matte_b))" in source
    assert "painter.drawPixmap(0, 0, pixmap)" in source
    assert "self._preview_pixmap = self._preview_display_pixmap(pixmap)" in source
    assert "save_result = on_save_figure(settings, str(image_path))" in source
    assert "settings[\"figure_alpha\"]" not in source


def test_plot_settings_panel_runs_preview_in_background_worker():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "class _PreviewWorkerBridge(QObject):" in source
    assert "finished = Signal(int, object)" in source
    assert "failed = Signal(int, object)" in source
    assert "self._preview_worker_bridge.finished.connect" in source
    assert "threading.Thread(" in source
    assert "target=self._run_preview_worker" in source
    assert "daemon=True" in source
    assert "configure_matplotlib_backend(interactive=False)" in source


def test_plot_settings_panel_keeps_latest_preview_request():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self._preview_generation = 0" in source
    assert "self._active_preview_generation: int | None = None" in source
    assert "self._pending_preview_request: tuple[dict[str, Any], bool] | None = None" in source
    assert "Preview updating... changes will render next." in source
    assert "def _preview_worker_result_is_stale(self, generation: int) -> bool:" in source
    assert "or self._pending_preview_request is not None" in source
    assert "self._start_pending_preview_if_available()" in source


def test_plot_settings_panel_preview_loading_only_disables_preview_actions():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    start = source.index("def _set_preview_loading(self, active: bool) -> None:")
    end = source.index("def _new_preview_image_path(self) -> Path:", start)
    body = source[start:end]
    assert "self._preview_button.setEnabled(" in body
    assert "self._save_figure_button.setEnabled(" in body
    assert "self._save_data_button.setEnabled(" in body
    assert "self.setEnabled(False)" not in body


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

    assert 'QGroupBox("Canvas and Typography")' in source
    assert 'QGroupBox("Lines and Markers")' in source
    assert 'QGroupBox("Axes, Ticks and Grid")' in source
    assert 'QGroupBox("Legend")' in source
    assert 'QGroupBox("Heatmap and Colorbar")' in source
    assert 'QGroupBox("Grid")' in source
    assert 'QGroupBox("Border")' in source
    assert 'tabs.addTab(text_legend_tab, "Text && Legend")' not in source
    assert 'tabs.addTab(axes_limits_tab, "Axes && Limits")' not in source
    assert 'tabs.addTab(ticks_grid_tab, "Ticks && Grid")' not in source
    assert 'tabs.addTab(style_tab, "Style")' not in source


def test_plot_settings_panel_includes_integration_controls_and_summary():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self.integration_mode = self._combo(_TOGGLE_MODES)" in source
    assert "self.integration_source = self._combo(_INTEGRATION_SOURCES)" in source
    assert "self.integration_target = self._combo(_INTEGRATION_TARGETS)" not in source
    assert "_INTEGRATION_TARGETS" not in source
    assert 'self.integration_x_min = self._bounded_float_line("min")' in source
    assert 'self.integration_x_max = self._bounded_float_line("max")' in source
    assert 'self.integration_baseline = self._bounded_float_line("0.0")' in source
    assert (
        'self.integration_alpha = self._bounded_float_line("0.0 - 1.0", bottom=0.0, top=1.0)'
        in source
    )
    assert "self._integration_summary_label = QLabel(" in source
    assert 'entry["integration"] = integration_payload' in source
    assert 'integration_group = QGroupBox("Integration")' in source
    assert 'section_id="layers.integration"' in source
    assert '"Integration",' in source
    assert "layout.addWidget(self._tab_integration_content)" not in source
    assert "self._series_integration_group.setVisible(layer_caps.show_integration)" in source
    assert "self._set_rows_visible(self._integration_rows, integration_enabled)" in source


def test_plot_settings_panel_starts_fit_editor_in_off_mode_before_saved_state_load():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    fit_combo_index = source.index("self._series_fit_mode = self._combo(_TOGGLE_MODES)")
    fit_off_index = source.index('self._set_combo_value(self._series_fit_mode, "off")')
    fit_connect_index = source.index(
        "self._series_fit_mode.currentTextChanged.connect(self._on_series_editor_changed)"
    )

    assert fit_combo_index < fit_off_index < fit_connect_index


def test_plot_settings_panel_nests_fit_style_overrides_inside_fit_payload():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'fit_config_payload["fit_color"] = fit_color_out' in source
    assert 'fit_config_payload["fit_alpha"] = fit_alpha_out' in source
    assert 'fit_config_payload["fit_line_width"] = fit_line_width_out' in source
    assert 'fit_config_payload["fit_line_style"] = fit_line_style_out' in source
    assert 'entry["fit_color"] = fit_color_out' not in source
    assert 'entry["fit_alpha"] = fit_alpha_out' not in source
    assert 'entry["fit_line_width"] = fit_line_width_out' not in source
    assert 'entry["fit_line_style"] = fit_line_style_out' not in source


def test_plot_settings_panel_nests_tick_sections_inside_axis_sections_and_keeps_annotation_list_open():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    x_axis_form_index = source.index("x_axis_layout.addLayout(x_axis_form)")
    x_ticks_group_index = source.index("self._x_ticks_group = self._make_collapsible_section(")
    x_axis_section_index = source.index('section_id="figure.axes.x"')
    y_axis_form_index = source.index("y_axis_layout.addLayout(y_axis_form)")
    y_ticks_group_index = source.index("self._y_ticks_group = self._make_collapsible_section(")
    y_axis_section_index = source.index('section_id="figure.axes.y"')
    annotation_splitter_index = source.index(
        "content_splitter = QSplitter(Qt.Orientation.Horizontal)"
    )
    annotation_list_index = source.index(
        'content_splitter.addWidget(\n                self._make_static_section(\n                    title="Annotations"'
    )

    assert x_axis_form_index < x_ticks_group_index < x_axis_section_index
    assert y_axis_form_index < y_ticks_group_index < y_axis_section_index
    assert annotation_splitter_index < annotation_list_index


def test_plot_settings_panel_uses_shared_non_editable_marker_dropdowns():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "_MARKER_TYPES = (" in source
    assert "self.marker_type = self._combo(_MARKER_TYPES)" in source
    assert "self.series_marker = self._combo(_MARKER_TYPES)" in source
    assert 'self.marker_type = self._combo(\n                ("", "o"' not in source
    assert 'self.series_marker = self._combo(\n                ("", "o"' not in source
    assert 'section_id="annotations.list"' not in source
    assert 'QGroupBox("Base Line Style")' not in source
    assert 'QGroupBox("Fit Summary")' not in source


def test_plot_settings_panel_keeps_export_in_preview_toolbar_without_transparent_save():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'transparent_label = QLabel("Transparent save")' not in source
    assert "self.transparent_mode" not in source
    assert 'self.figure_alpha = self._bounded_float_line("0.0 - 1.0"' in source
    assert 'self.save_figure_button = QPushButton("Export Figure")' in source
    assert 'self.save_data_button = QPushButton("Export Data")' in source
    assert "self.save_data_button.clicked.connect(on_save_data_callback)" in source
    assert '"export.data": "Saves the current preview line data to a text data file."' in source
    assert "header_layout.addWidget(self._save_figure_button)" not in source


def test_plot_settings_panel_wires_export_data_callback_and_dialog():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert (
        "on_save_data: Callable[[dict[str, Any], str], str | tuple[str, dict[str, Any]]] | None = None"
        in source
    )
    assert "self._save_data_button.setEnabled(on_save_data is not None)" in source
    assert "def _handle_save_data(self) -> None:" in source
    assert '"Save Data"' in source
    assert "self._data_save_filters, self._data_default_name = _data_filetype_filters()" in source
    assert (
        "CSV data (*.csv);;DAT data (*.dat);;TSV data (*.tsv);;Text data (*.txt);;All files (*)"
        in source
    )


def test_plot_settings_panel_supports_detachable_preview_window():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "class _PreviewPane(QFrame)" in source
    assert "class _DetachedPreviewWindow(QMainWindow)" in source
    assert 'self.detach_button = QPushButton("Detach Preview")' in source
    assert 'self.dock_button = QPushButton("Dock Back")' in source
    assert 'self.setWindowTitle("LiNaK Figure Preview")' in source
    assert "self._embedded_preview_pane.setVisible(False)" in source
    assert "on_save_data_callback=self._handle_save_data" in source
    assert "detached_window.close_from_dock()" in source
    assert "QTimer.singleShot(0, self._on_dock_requested)" in source


def test_plot_settings_panel_allows_resizable_workspace_and_preview_split():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "_WORKSPACE_PANEL_WIDTH = 760" in source
    assert "_WORKSPACE_PANEL_MIN_WIDTH = 520" in source
    assert "left_panel.setMinimumWidth(_WORKSPACE_PANEL_MIN_WIDTH)" in source
    assert (
        "left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)"
        in source
    )
    assert "left_panel.setMaximumWidth(_WORKSPACE_PANEL_WIDTH)" not in source
    assert "splitter.setStretchFactor(0, 1)" in source


def test_plot_settings_panel_exposes_contract_driven_position_mapping_controls():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "def _build_position_mapping_sections(self, layout: QVBoxLayout) -> None:" in source
    assert 'selection_title = "Source" if analysis == "position" else "Profile Selection"' in source
    assert 'title="Mapping"' in source
    assert 'title="Summary"' in source
    assert 'self.position_view_type = self._combo(' in source
    assert 'self.position_mapping_x = self._combo(["ps", "fs", "step", "frame"])' in source
    assert 'self.position_mapping_y = self._combo(["distance", "x", "y", "z"])' in source
    assert 'self.position_mapping_value = self._combo(list(_POSITION_PROJECTION_QUANTITIES))' in source
    assert '"Value / color / filter by"' in source
    assert '"Split by"' in source
    assert "position_plot_options_to_view_mapping(" in source
    assert "position_view_mapping_to_plot_options(mapping)" in source
    assert "generic_view_type_compatibility(" in source
    assert "self.position_component = self._combo(_POSITION_COMPONENT_LABELS)" not in source


def test_plot_settings_panel_uses_rdf_layer_summary_and_coordination_selector_choices():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "Use stored metadata" not in source
    assert "Same as Species A / stored metadata" not in source
    assert 'self._rdf_source_pair_count_label = QLabel("")' in source
    assert 'self._rdf_source_pair_list_label = QLabel("")' in source
    assert "All stored RDF profiles are loaded as layers." in source
    assert '"Available layers"' in source
    assert '"Pairs"' in source
    assert "self.rdf_species_a" not in source
    assert "self.rdf_species_b" not in source
    assert 'self._binning_helper_label = QLabel("")' in source
    assert "_auto_display_note" in source
    assert '"series_source_bin_widths"' in source
    assert "Source bin size:" in source
    assert "Requested display bin size:" in source
    assert "Points per visible bin:" in source


def test_plot_settings_panel_has_manual_light_dark_theme_switch():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self._theme_switch = QCheckBox("Dark mode")' in source
    assert 'self._theme_switch.setObjectName("themeSwitch")' in source
    assert "self._theme_switch.toggled.connect(self._handle_theme_switch_toggled)" in source
    assert "QCheckBox#themeSwitch" in source
    assert 'if self._theme_mode == "dark":' in source
    assert 'if self._theme_mode == "light":' in source
    assert '"_gui_theme_mode"' not in source
    assert "_GUI_THEME_MODES" not in source


def test_plot_settings_panel_keeps_status_outside_hidden_preview_panel():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "root_layout.addWidget(self._status_label)" in source
    assert "right_layout.addWidget(self._status_label)" not in source
    assert "QFrame#detachedPreviewPanel" in source


def test_plot_settings_panel_exposes_per_axis_ticks_and_color_controls():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'self.x_axis_scale = self._bounded_float_line("1.0")' in source
    assert 'self.x_axis_offset = self._bounded_float_line("0.0")' in source
    assert '"X scale factor"' in source
    assert '"X offset"' in source
    assert '"x_axis_scale": x_axis_scale' in source
    assert '"x_axis_offset": x_axis_offset' in source
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
    assert '"_x_ticks_visible"' in source
    assert '"_y_ticks_visible"' in source
    assert "self.title_pad," in source


def test_plot_settings_panel_exposes_legend_title_font_only_under_legend_title():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self.legend_title_font = self._positive_int_line()" in source
    assert '"Legend title font"' in source
    assert 'nested_key="title_fontsize"' in source
    assert 'field_name="legend-title-font-size"' in source


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


def test_plot_settings_panel_uses_inline_annotation_reorder_arrows_instead_of_move_buttons():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "class _AnnotationRowWidget(QWidget):" in source
    assert 'self._annotation_move_up_button = QPushButton("Move Up")' not in source
    assert 'self._annotation_move_down_button = QPushButton("Move Down")' not in source
    assert "self._handle_annotation_row_move(" in source
    assert "QWidget#annotationRowWidget {" in source


def test_plot_settings_panel_uses_static_section_headers_for_annotations_page():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "class _StaticSection(QFrame):" in source
    assert '"staticSectionHeaderLabel"' in source
    assert 'title="Annotations"' in source
    assert 'title="Selected Annotation"' in source
    assert "self._make_static_section(" in source
    assert (
        'self._make_collapsible_section(\n                    title="Selected Annotation"'
        not in source
    )


def test_annotation_defaults_start_visible_in_preview_space():
    entry_text = _annotation_defaults_for_gui("text", index=1)
    entry_line = _annotation_defaults_for_gui("line", index=1)
    entry_arrow = _annotation_defaults_for_gui("arrow", index=1)

    assert entry_text["coord_system"] == "axes"
    assert entry_text["text"] == "Text 1"
    assert entry_line["coord_system"] == "axes"
    assert entry_line["x1"] == "0.15"
    assert entry_line["x2"] == "0.85"
    assert entry_arrow["coord_system"] == "axes"


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
    assert (
        _default_error_series_label("H2O density", "sample_sem") == "H2O density \u00b1Sample SEM"
    )


def test_error_supported_for_view_tracks_one_dimensional_modes_only():
    assert _error_supported_for_view("rdf") is True
    assert _error_supported_for_view("temperature") is True
    assert _error_supported_for_view("orientation", orientation_heatmap=True) is False
    assert _error_supported_for_view("position", position_component="xy-z") is False
    assert _error_supported_for_view("position", position_component="2d-projection") is False
    assert (
        _error_supported_for_view("coordination", coordination_component="time-distance") is False
    )


def test_temperature_is_registered_for_line_fit_controls():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '"density", "msd", "rdf", "temperature"' in source
    assert '"coordination",\n                "temperature",' in source


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
            "source_series_id": "series:0:3",
            "is_generated": False,
            "member_series_ids": [],
            "group_reducer": "mean",
            "source_name": "density.h5",
            "source_directory": "runs/run_04",
            "source_path": "/tmp/runs/run_04/density.h5",
            "source_index": None,
            "series_index": None,
        }
    ]


def test_coerce_series_descriptors_preserves_generated_copy_identity():
    descriptors = _coerce_series_descriptors(
        [
            {
                "series_id": "source:copy",
                "source_kind": "source",
                "source_series_id": "source:original",
                "is_generated": True,
                "default_label": "O Copy",
            },
            {
                "series_id": "group:1",
                "source_kind": "group",
                "default_label": "Group 1",
                "member_series_ids": ["source:original", "source:copy"],
            },
        ]
    )

    assert descriptors[0]["is_generated"] is True
    assert descriptors[0]["source_series_id"] == "source:original"
    assert descriptors[1]["is_generated"] is True
    assert descriptors[1]["source_kind"] == "group"
    assert descriptors[1]["member_series_ids"] == ["source:original", "source:copy"]


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


def test_partition_series_ids_for_display_order_preserves_relative_order_in_partitions():
    resolved = _partition_series_ids_for_display_order(
        ["series-c", "series-a", "series-b", "series-d"],
        enabled_by_id={
            "series-c": True,
            "series-a": False,
            "series-b": True,
            "series-d": True,
        },
        group_by_id={
            "series-c": False,
            "series-a": False,
            "series-b": True,
            "series-d": False,
        },
    )

    assert resolved == ["series-c", "series-d", "series-b", "series-a"]


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


def test_derive_warning_messages_uses_series_overrides_and_skips_heatmap_mode():
    warnings = _derive_warning_messages(
        {
            "series_descriptors": [
                {"series_id": "series:0", "source_kind": "source"},
                {"series_id": "series:1", "source_kind": "source"},
            ],
            "series_overrides": {
                "series:0": {"normalization_mode": "max", "normalization_value": 1.0},
                "series:1": {"enabled": False, "normalization_mode": "factor"},
            },
            "component": "heatmap",
        }
    )

    assert all("normalized" not in message.lower() for message in warnings)


def test_derive_warning_messages_skips_heatmap_mode_when_only_view_mapping_is_present():
    warnings = _derive_warning_messages(
        {
            "series_descriptors": [
                {"series_id": "series:0", "source_kind": "source"},
                {"series_id": "series:1", "source_kind": "source"},
            ],
            "series_overrides": {
                "series:0": {"normalization_mode": "max"},
                "series:1": {"normalization_mode": "none"},
            },
            "view_mapping": PlotViewMapping(
                view_type_id="heatmap_2d",
                x="bin_centers_A",
                y="heatmap_angle_bin_centers",
                role_assignments={"z": "heatmap_polar"},
            ),
        }
    )

    assert all("normalized" not in message.lower() for message in warnings)


def test_backend_summary_helpers_use_view_and_role_wording():
    assert _density_backend_summary_text(
        view_type_id="line_1d",
        x_mode="distance",
        quantity="mass",
    ) == "view type=line_1d, x role=distance, y role=mass_density"
    assert _density_backend_summary_text(
        view_type_id="heatmap_2d",
        x_mode="distance",
        quantity="number",
    ) == "view type=heatmap_2d, source field=number_density_2d"
    assert _coordination_mapping_preset_label("time-distance") == "distance_vs_time"
    assert _coordination_backend_summary_text(
        component="time-distance",
        time_axis="ps",
    ) == "view type=trajectory_2d, view preset=distance_vs_time, x role=time (ps), y role=coordination_number"
    assert _orientation_backend_summary_text(
        component="heatmap",
        angle="azimuthal",
        is_heatmap=True,
    ) == "view type=heatmap_2d, view preset=heatmap, angle role=azimuthal"
    assert _potential_backend_summary_text(
        view_type="table_records",
        y_quantity="water_bulk_potential",
        standard_plot="",
    ) == "view type=table_records, record inspection mode"
    assert _potential_backend_summary_text(
        view_type="line_1d",
        y_quantity="efermi",
        standard_plot="",
    ) == "view type=line_1d, y role=efermi"


def test_settings_use_heatmap_rendering_prefers_view_mapping_over_legacy_component():
    assert (
        _settings_use_heatmap_rendering(
            {
                "component": "heatmap",
                "view_mapping": PlotViewMapping(
                    view_type_id="line_1d",
                    x="distance_A",
                    y="density",
                ),
            }
        )
        is False
    )


def test_settings_use_heatmap_rendering_uses_legacy_component_only_as_fallback():
    assert _settings_use_heatmap_rendering({"component": "heatmap"}) is True


def test_derive_warning_messages_ignores_advanced_json_overlap():
    warnings = _derive_warning_messages({"legend": True})

    assert all(
        "Advanced JSON overlaps with standard controls" not in message for message in warnings
    )


def test_non_position_mapping_sections_use_role_based_labels_and_notes():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert '"X-axis quantity"' in source
    assert '"Z quantity"' in source
    assert '"2D X-axis quantity"' in source
    assert '"Species"' in source
    assert '"Y role"' in source
    assert '"View preset"' in source
    assert '"View type"' in source
    assert '"Time x-role"' in source
    assert '"Angle role"' in source


def test_density_target_filter_and_binning_controls_are_source_level():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "self._density_species_checkboxes = {}" in source
    assert "def _handle_density_species_checkbox_changed" in source
    assert "def _enabled_density_species" in source
    assert '"density_enabled_species": density_enabled_species' in source
    assert "QGridLayout(species_widget)" in source
    assert "def _sync_density_species_selection_for_view_type" in source
    assert "def _handle_density_range_change" in source
    assert "lower.textChanged.connect(self._handle_density_range_change)" in source
    assert "upper.textChanged.connect(self._handle_density_range_change)" in source
    assert "self._density_mapping_1d_rows" in source
    assert "self._density_mapping_2d_rows" in source
    assert '"Density Binning"' in source
    assert '"X-axis bin size"' in source
    assert '"Y-axis bin size"' in source
    assert "two_dimensional_binning = is_heatmap or density_heatmap_mode" in source


def test_non_position_collect_settings_no_longer_emits_active_legacy_mapping_keys():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert 'settings["x_mode"] = density_x_mode' not in source
    assert 'settings["quantity"] = self.density_quantity.currentText().strip() or "mass"' not in source
    assert 'settings["component"] = (' not in source
    assert 'settings["time_axis"] = self.coordination_time_axis.currentText().strip() or "ps"' not in source
    assert 'settings["angle"] = self.orientation_angle.currentText().strip() or "polar"' not in source
    assert 'settings["table_view"] = ' not in source


def test_position_collect_settings_no_longer_reemits_legacy_mapping_options():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "settings.update(position_view_mapping_to_plot_options(mapping))" not in source


def test_current_view_capability_checks_no_longer_depend_on_position_component_tokens():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")

    assert "def _current_position_component_token(self) -> str:" not in source
    assert "position_component=self._current_position_component_token()" not in source


def test_settings_use_heatmap_rendering_checks_mapping_before_legacy_component():
    source = Path("src/linak/plot/plot_gui.py").read_text(encoding="utf-8")
    helper_start = source.index("def _settings_use_heatmap_rendering(settings: dict[str, Any]) -> bool:")
    helper_block = source[helper_start : helper_start + 800]

    assert helper_block.index('mapping = settings.get("view_mapping")') < helper_block.index(
        'if str(settings.get("component") or "").strip().lower() == "heatmap":'
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
