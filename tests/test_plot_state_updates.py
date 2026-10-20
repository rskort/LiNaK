"""Behavioral coverage for Studio view state and rendered tick synchronization."""

import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator

from linak.plot import plot_gui
from linak.plot.data_contract import PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP
from linak.plot.profile_persistence import build_plot_profile_payload, flatten_plot_profile_payload
from linak.plot.plotting import (
    _apply_plot_grid,
    _visible_major_ticks,
    plot_heatmap_series,
    plot_multi_line_series,
    with_style_overrides,
)


def exercise_panel(monkeypatch, analysis, inspect, *, settings_updates=None):
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    if os.name == "nt" and not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    descriptors = [
        {"series_id": "H:0", "default_label": "H 0", "species": "H", "atom_index": 0},
        {"series_id": "water:1", "default_label": "H2O 1", "species": "mol:H2O", "atom_index": 1},
    ]
    settings = {"series_labels": ["H 0", "H2O 1"], "series_descriptors": descriptors}
    if analysis == "density":
        settings["_profile_filter_options"] = {
            "density_view_types": [PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP]
        }
    if settings_updates:
        settings.update(deepcopy(settings_updates))
    failures = []
    checked = []

    def resolve(current):
        updated = deepcopy(descriptors)
        if current.get("view_mapping", {}).get("view_type_id") == PLOT_VIEW_2D_HEATMAP:
            for descriptor in updated:
                descriptor["series_id"] = str(descriptor["series_id"]) + ":2d"
        return {
            "series_descriptors": updated,
            "series_labels": [d["default_label"] for d in updated],
        }

    def run(application):
        window = next(
            w
            for w in application.topLevelWidgets()
            if hasattr(w, "_series_colors_data") and w.isVisible()
        )
        try:
            window._preview_timer.stop()
            window._preview_worker_stop.set()
            window._preview_worker_queue.put_nowait(None)
            if window._preview_worker_thread is not None:
                window._preview_worker_thread.join(timeout=5)
                assert not window._preview_worker_thread.is_alive()
            window._preview_generation += 1
            application.processEvents()
            window._set_preview_loading(False)
            monkeypatch.setattr(window._preview_timer, "start", lambda *args: None)
            inspect(window, application)
            checked.append(True)
        except BaseException as error:
            failures.append(error)
        finally:
            window._cleanup_preview_artifacts()
            window.hide()
            window.deleteLater()
            application.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", run)
    plot_gui.launch_plot_settings_panel(
        title="State regression",
        initial_settings=settings,
        default_profile_settings=settings,
        analysis_name=analysis,
        on_preview=lambda settings: {},
        on_save=lambda *args: "saved",
        on_resolve_series_defaults=resolve,
    )
    if failures:
        raise failures[0]
    assert checked == [True]
    app.processEvents()


def _open_layers(window, app):
    for row in range(window._nav_list.count()):
        if window._nav_list.item(row).text() == "Layers":
            window._nav_list.setCurrentRow(row)
            break
    window._series_visibility_group.set_expanded(True, animate=False)
    window._series_style_group.set_expanded(True, animate=False)
    window._series_derived_group.set_expanded(True, animate=False)
    app.processEvents()


def test_advanced_editors_are_guided_and_preserve_shared_settings(monkeypatch):
    from PySide6.QtWidgets import QLabel

    def inspect(window, app):
        assert not hasattr(window, "_workspace_note")
        assert not hasattr(window, "_page_note_label")
        assert not hasattr(window, "_cache_budget_mib_edit")
        for row in range(window._nav_list.count()):
            window._nav_list.setCurrentRow(row)
            assert window._page_title_label.text() == window._nav_list.item(row).text()
        assert not any(
            "Preview resources" in label.text()
            for label in window._tab_advanced.findChildren(QLabel)
        )
        assert len(window._advanced_option_editors) == 9
        options = window._advanced_option_editors["legend_kwargs"]
        assert options.editor is window.legend_kwargs_json
        options.editor.setPlainText('{"borderpad": 1.5, "frameon": false, "ncols": 2}')
        assert window.legend_frame_mode.currentText() == "off"
        assert window.legend_columns.text() == "2"
        expected = window._collect_settings()["legend_kwargs"]
        options.format_button.click()
        assert window._collect_settings()["legend_kwargs"] == expected
        window._populate(window._collect_settings())
        assert window._collect_settings()["legend_kwargs"] == expected
        assert options.is_valid

        calls = []
        monkeypatch.setattr(window, "_schedule_preview_update", lambda: calls.append(True))
        options.editor.setPlainText('{"borderpad": }')
        assert not options.is_valid
        assert not calls
        assert not window._save_button.isEnabled()
        options.editor.setPlainText('{"borderpad": 1.5}')
        assert options.is_valid
        assert calls
        # Standard-control synchronization blocks editor signals, but its
        # validation message and example/format actions still need refreshing.
        window._set_plaintext_if_different(options.editor, {"borderpad": 2.0, "fontsize": 14})
        assert options.validation_label.text() == "2 options configured"
        assert not options.example_button.isEnabled()
        window._open_session_logs()
        app.processEvents()
        assert window._logs_section.is_expanded()
        assert window._nav_list.currentItem().text() == "Advanced"

    exercise_panel(monkeypatch, "density", inspect)


def test_density_mapping_groups_raw_species_by_element(monkeypatch):
    options = [
        {"value": "H", "label": "H"},
        {"value": "Pt", "label": "Pt"},
        {"value": "species:Pt", "label": "species:Pt"},
        {"value": "species:Pt_top", "label": "species:Pt_top"},
        {"value": "mol:H", "label": "free H"},
        {"value": "mol:H2O", "label": "H2O"},
    ]

    def inspect(window, _app):
        assert set(window._density_species_checkboxes) == {
            "H",
            "Pt",
            "mol:H",
            "mol:H2O",
        }
        assert window._density_species_checkboxes["Pt"].text() == "Pt"
        assert window._density_species_checkboxes["mol:H"].text() == "free H"
        assert window._density_species_checkboxes["mol:H2O"].text() == "H2O"
        assert window._density_species_checkboxes["Pt"].isChecked()
        assert window._density_species_checkboxes["mol:H2O"].isChecked()
        assert not window._density_species_checkboxes["H"].isChecked()
        assert not window._density_species_checkboxes["mol:H"].isChecked()

    exercise_panel(
        monkeypatch,
        "density",
        inspect,
        settings_updates={
            "_profile_filter_options": {
                "density_view_types": [PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP],
                "density_species_options": options,
            },
            "density_enabled_species": ["species:Pt_top", "mol:H2O"],
        },
    )


@pytest.mark.parametrize("analysis", ["position", "density", "orientation"])
def test_layers_page_scrolls_browser_and_preserves_edits(monkeypatch, analysis):
    from time import monotonic

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton

    def inspect(window, app):
        window.resize(1000, 650)
        _open_layers(window, app)
        scroll = window._layers_scroll
        bar = scroll.verticalScrollBar()
        assert bar.maximum() > 0
        browser = window._layer_browser_section
        button = next(b for b in browser.findChildren(QPushButton) if b.text() == "Show all")
        start_y = button.mapTo(scroll.viewport(), QPoint()).y()
        bar.setValue(bar.maximum())
        app.processEvents()
        assert button.mapTo(scroll.viewport(), QPoint()).y() < 0 < start_y
        assert window.series_list.mapTo(scroll.viewport(), QPoint()).y() < 0

        # Edits and figure replacement must not scroll the settings page to the top.
        position = bar.value()
        window.series_label.setText("Custom layer")
        window.series_color.setText("#123456")
        window._sync_series_selection_widgets(window._series_active_index)
        figure = plt.figure()
        figure.add_subplot().plot([0, 1], [0, 1])
        window._install_preview_figure(figure)
        app.processEvents()
        assert bar.value() == position
        assert window.series_label.text() == "Custom layer"
        assert window.series_color.text() == "#123456"
        window.series_list.setCurrentRow(window._display_row_for_selection(1))
        app.processEvents()
        assert bar.value() == position

        # Collapse affects only the browser, retains context and session state.
        bar.setValue(0)
        expanded_height = scroll.widget().height()
        browser.set_expanded(False, animate=False)
        deadline = monotonic() + 2
        while (browser.height() >= 80 or scroll.widget().height() >= expanded_height) and (
            monotonic() < deadline
        ):
            app.processEvents()
            QTest.qWait(10)
        assert browser.height() < 80
        assert scroll.widget().height() < expanded_height
        assert window._collapsible_section_state["layers.browser"] is False
        assert window._selected_layer_card.isVisible()
        assert not window.series_list.isVisible()
        selected = window._active_series_row_id()
        window._refresh_series_list_widgets()
        assert not browser.is_expanded()
        browser.set_expanded(True, animate=False)
        app.processEvents()
        assert window._active_series_row_id() == selected
        assert window.series_list.isVisible()
        scroll.ensureWidgetVisible(browser.toggle_button)
        browser.toggle_button.setFocus()
        QTest.keyClick(browser.toggle_button, Qt.Key.Key_Space)
        assert not browser.is_expanded()
        browser.set_expanded(True, animate=False)
        plt.close(figure)

    exercise_panel(monkeypatch, analysis, inspect)


def test_layer_browser_bounded_rows_filtering_and_scroll_handoff(monkeypatch):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    def wheel(widget, delta, *, pixels=False):
        local = QPointF(widget.rect().center())
        event = QWheelEvent(
            local,
            QPointF(widget.mapToGlobal(local.toPoint())),
            QPoint(0, delta) if pixels else QPoint(),
            QPoint() if pixels else QPoint(0, delta),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(widget, event)

    def inspect(window, app):
        window.resize(950, 650)
        _open_layers(window, app)
        descriptors = [
            {
                "series_id": f"atom:{index}",
                "default_label": f"O [{index}]",
                "position_species": "O",
                "atom_index": index,
            }
            for index in range(1000)
        ]
        window._apply_series_defaults(
            [d["default_label"] for d in descriptors], descriptors=descriptors
        )
        app.processEvents()
        browser = window.series_list
        bar = browser.verticalScrollBar()
        page_bar = window._layers_scroll.verticalScrollBar()
        height = browser.height()
        assert height <= 240
        assert bar.maximum() > 0
        page_bar.setValue(0)
        bar.setValue(0)
        wheel(browser.viewport(), -120)
        assert bar.value() > 0
        assert page_bar.value() == 0
        bar.setValue(bar.maximum())
        wheel(browser.viewport(), -120)
        assert page_bar.value() > 0
        page_bar.setValue(page_bar.maximum())
        bar.setValue(0)
        old_page = page_bar.value()
        wheel(browser.viewport(), 32, pixels=True)
        assert page_bar.value() < old_page

        bar.setValue(100)
        anchor = window._current_series_list_view_anchor()
        selected = window._active_series_row_id()
        window._layer_browser_section.set_expanded(False, animate=False)
        window._refresh_series_list_widgets()
        window._layer_browser_section.set_expanded(True, animate=False)
        app.processEvents()
        assert window._active_series_row_id() == selected
        assert window._current_series_list_view_anchor() == anchor

        window._layer_search.setText("[999]")
        app.processEvents()
        assert browser.height() < height
        assert window._matching_layer_indices() == [999]
        # One heading and one result: no independent list scrolling is necessary.
        assert bar.maximum() == 0
        window._layer_search.clear()
        app.processEvents()
        assert browser.height() == height
        window._layer_search.setText("no matching layer")
        app.processEvents()
        assert browser.height() < height
        assert window._layer_empty_label.isVisible()
        window._layer_search.clear()
        app.processEvents()
        browser.setCurrentRow(0)
        QTest.keyClick(browser, Qt.Key.Key_Left)
        app.processEvents()
        assert browser.height() < height
        QTest.keyClick(browser, Qt.Key.Key_Right)
        app.processEvents()
        assert browser.height() == height

    exercise_panel(monkeypatch, "position", inspect)


def test_layers_narrow_layout_and_wheel_over_controls(monkeypatch):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QPushButton

    def inspect(window, app):
        window.resize(900, 600)
        _open_layers(window, app)
        scroll = window._layers_scroll
        bar = scroll.verticalScrollBar()
        assert scroll.horizontalScrollBar().maximum() == 0
        browser = window._layer_browser_section
        browser_layout = browser.body_frame.layout().itemAt(0).widget().layout()
        assert browser_layout.contentsMargins().left() == 8
        assert browser_layout.contentsMargins().right() == 8
        for button in browser.findChildren(QPushButton):
            assert button.width() >= button.minimumSizeHint().width()
            assert button.mapTo(scroll.widget(), QPoint()).x() >= 0
            assert (
                button.mapTo(scroll.widget(), button.rect().topRight()).x()
                < scroll.viewport().width()
            )
        for target in (
            window._layer_isolate_button,
            browser.toggle_button,
            window.series_color,
            window.series_marker,
            window._series_style_group.body_frame,
        ):
            scroll.ensureWidgetVisible(target)
            app.processEvents()
            previous = bar.value()
            point = target.rect().center()
            position = QPointF(point)
            event = QWheelEvent(
                position,
                QPointF(target.mapToGlobal(point)),
                QPoint(),
                QPoint(0, -120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(target, event)
            app.processEvents()
            assert bar.value() > previous

        # JSON editors retain their own scrolling; the page filter must not
        # consume input intended for their nested viewport.
        editor = window.series_line_kwargs_json
        editor.setPlainText("{\n" + "\n" * 80 + "}")
        scroll.ensureWidgetVisible(editor)
        app.processEvents()
        editor.verticalScrollBar().setValue(0)
        previous = bar.value()
        local = QPointF(editor.viewport().rect().center())
        event = QWheelEvent(
            local,
            QPointF(editor.viewport().mapToGlobal(local.toPoint())),
            QPoint(),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(editor.viewport(), event)
        assert editor.verticalScrollBar().value() > 0
        assert bar.value() == previous

    exercise_panel(monkeypatch, "density", inspect)


@pytest.mark.parametrize("analysis", ["position", "density", "orientation"])
def test_view_round_trip_restores_independent_colors_and_saved_overrides(monkeypatch, analysis):
    def inspect(window, app):
        selector = getattr(window, f"{analysis}_view_type")

        def colors(values):
            for index, color in enumerate(values):
                window.series_list.setCurrentRow(window._display_row_for_selection(index))
                window.series_color.setText(color)

        def switch(index):
            selector.setCurrentIndex(index)
            assert len(window._series_colors_data) == 2

        colors(["#112233", "#445566"])
        window._series_enabled_data[0] = False
        one_d = {
            d["series_id"]: window._series_colors_data[i]
            for i, d in enumerate(window._series_descriptors_data)
        }
        for _ in range(2):
            switch(1)
            colors(["#aabbcc", "#ddeeff"])
            switch(0)
            assert {
                d["series_id"]: window._series_colors_data[i]
                for i, d in enumerate(window._series_descriptors_data)
            } == one_d
            assert not window._series_enabled_data[0]
        switch(1)
        assert window._series_colors_data == ["#aabbcc", "#ddeeff"]
        current = window._collect_settings()
        payload = build_plot_profile_payload(f"plot:{analysis}", current)
        restored = flatten_plot_profile_payload(f"plot:{analysis}", payload)
        window._populate(restored)
        assert window._series_colors_data == ["#aabbcc", "#ddeeff"]
        switch(0)
        assert window._series_colors_data == ["#112233", "#445566"]
        # Descriptor refreshes in another order retain identity-based colours.
        descriptors = list(reversed(window._series_descriptors_data))
        window._apply_series_defaults(
            [d["default_label"] for d in descriptors], descriptors=descriptors
        )
        assert {
            d["series_id"]: window._series_colors_data[i]
            for i, d in enumerate(window._series_descriptors_data)
        } == one_d

    exercise_panel(monkeypatch, analysis, inspect)


@pytest.mark.parametrize("heatmap", [False, True])
def test_tick_controls_follow_draw_and_grid_preserves_locators(monkeypatch, heatmap):
    def inspect(window, app):
        fig, ax = plt.subplots()
        if heatmap:
            ax.pcolormesh(np.arange(5), np.arange(5), np.ones((4, 4)))
        else:
            ax.plot(np.arange(5), np.arange(5))
        window._install_preview_figure(fig)
        window._set_synced_field_mode("x_ticks", "auto")
        window._set_synced_field_mode("y_ticks", "auto")

        def assert_auto():
            window._preview_canvas.draw()
            for field, axis in ((window.x_ticks, ax.xaxis), (window.y_ticks, ax.yaxis)):
                lower, upper = sorted(axis.get_view_interval())
                expected = [v for v in axis.get_majorticklocs() if lower <= v <= upper]
                actual = [float(v) for v in field.text().split(",") if v.strip()]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-10)

        assert_auto()
        ax.set_xlim(0.3, 3.2)
        ax.set_ylim(0.2, 3.6)
        assert_auto()
        fig.set_size_inches(5, 3)
        assert_auto()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        window._set_synced_field_mode("x_ticks", "manual")
        window.x_ticks.setText("0.5, 1.5, 3")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1.5, 3]))
        window._preview_canvas.draw()
        locators = [axis.get_major_locator() for axis in (ax.xaxis, ax.yaxis)]
        minor = [axis.get_minor_locator() for axis in (ax.xaxis, ax.yaxis)]
        positions = ax.get_xticks().copy(), ax.get_yticks().copy()
        limits = ax.get_xlim(), ax.get_ylim()
        for mode, which, axis_name in (
            ("off", "major", "both"),
            ("on", "both", "x"),
            ("on", "minor", "y"),
            ("off", "both", "both"),
        ):
            window.grid_mode.setCurrentText(mode)
            window.grid_which.setCurrentText(which)
            window.grid_axis.setCurrentText(axis_name)
            assert window._apply_axis_style_fields_to_canvas()
            window._preview_canvas.draw()
            assert [axis.get_major_locator() for axis in (ax.xaxis, ax.yaxis)] == locators
            assert [axis.get_minor_locator() for axis in (ax.xaxis, ax.yaxis)] == minor
            np.testing.assert_array_equal(ax.get_xticks(), positions[0])
            np.testing.assert_array_equal(ax.get_yticks(), positions[1])
            assert (ax.get_xlim(), ax.get_ylim()) == limits
            assert window.x_ticks.text() == "0.5, 1.5, 3"
        # A real scale change preserves explicit ticks and lets automatic ticks resolve anew.
        window.x_scale.setCurrentText("log")
        assert window._apply_axis_style_fields_to_canvas()
        window._preview_canvas.draw()
        np.testing.assert_array_equal(ax.get_xticks(), positions[0])
        assert window.x_ticks.text() == "0.5, 1.5, 3"
        window._set_synced_field_mode("x_ticks", "auto")
        window._handle_detach_preview()
        assert_auto()
        window._set_synced_field_mode("x_ticks", "manual")
        window.x_ticks.setText("0.5, 1.5, 3")
        window._preview_canvas.draw()
        assert window.x_ticks.text() == "0.5, 1.5, 3"
        window._handle_dock_preview()
        window._disconnect_preview_axis_callbacks()
        assert window._preview_draw_callback is None
        assert window._preview_axis_callback_ids == []

    exercise_panel(monkeypatch, "position", inspect)


def test_preview_canvas_recenters_after_window_panel_and_detached_resizes(monkeypatch):
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    def inspect(window, app):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot([0, 1, 2], [0, 2, 1])
        window._install_preview_figure(fig)

        def wait_for_layout():
            QTest.qWait(window._preview_layout_timer.interval() + 20)
            app.processEvents()

        def assert_centered():
            viewport = window._preview_canvas_scroll.viewport()
            canvas = window._preview_canvas
            canvas.draw()
            top_left = canvas.mapTo(viewport, QPoint())
            assert canvas.width() <= viewport.width()
            assert canvas.height() <= viewport.height()
            assert abs(2 * top_left.x() + canvas.width() - viewport.width()) <= 1
            assert abs(2 * top_left.y() + canvas.height() - viewport.height()) <= 1
            # Centering the QWidget alone is insufficient: at fractional screen
            # scaling the rendered pixels must fill that widget as well.
            ratio = canvas.device_pixel_ratio
            assert abs(fig.bbox.width / ratio - canvas.width()) <= 1
            assert abs(fig.bbox.height / ratio - canvas.height()) <= 1

        for width, height in ((1750, 950), (1050, 680), (1450, 820)):
            window.resize(width, height)
            wait_for_layout()
            assert_centered()

        # A window-state transition can arrive before nested layouts have their
        # final geometry. The coalesced pass must still use the settled viewport.
        QApplication.sendEvent(window, QEvent(QEvent.Type.WindowStateChange))
        wait_for_layout()
        assert_centered()

        splitter = window._splitter
        splitter.setSizes([800, 350])
        wait_for_layout()
        assert_centered()
        splitter.setSizes([560, 900])
        wait_for_layout()
        assert_centered()

        window._handle_detach_preview()
        wait_for_layout()
        assert window._detached_preview_window is not None
        assert_centered()
        window._detached_preview_window.resize(760, 540)
        wait_for_layout()
        assert_centered()
        window._handle_dock_preview()
        wait_for_layout()
        assert_centered()

        # Moving between monitors changes the backend ratio even when the
        # logical widget size stays unchanged. Exercise that path in ordinary
        # CI runs too, which usually have a device-pixel ratio of one.
        canvas = window._preview_canvas
        for ratio in (1.25, 1.5, 2.0, 1.0):
            canvas._set_device_pixel_ratio(ratio)
            window._resize_preview_canvas_to_figure()
            window._resize_preview_canvas_to_figure()
            assert_centered()
            assert fig.dpi == pytest.approx(fig._original_dpi * ratio)
        plt.close(fig)

    exercise_panel(monkeypatch, "density", inspect)


def test_shared_grid_application_clears_old_grid_without_altering_ticks():
    fig, ax = plt.subplots()
    try:
        ax.set_xticks([0.1, 0.7])
        locator = ax.xaxis.get_major_locator()
        for enabled, axis in ((True, "both"), (True, "x"), (False, "both")):
            _apply_plot_grid(
                ax, style=with_style_overrides(grid=enabled), grid_kwargs={"axis": axis}
            )
            fig.canvas.draw()
            assert ax.xaxis.get_major_locator() is locator
            np.testing.assert_array_equal(ax.get_xticks(), [0.1, 0.7])
            assert any(line.get_visible() for line in ax.get_xgridlines()) == enabled
            assert any(line.get_visible() for line in ax.get_ygridlines()) == (
                enabled and axis == "both"
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("heatmap", [False, True])
def test_saved_image_ticks_match_captured_state_and_minor_ticks_remain_enabled(tmp_path, heatmap):
    state: dict = {}
    options = dict(
        title="Tick regression",
        x_label="X",
        y_label="Y",
        show=False,
        output=tmp_path / "ticks.png",
        capture_state=state,
        tick_params_kwargs={"_x_minor_ticks_mode": "on", "_y_minor_ticks_mode": "on"},
        x_lim=[0.3, 3.2],
        y_lim=[0.2, 3.6],
    )
    if heatmap:
        plot_heatmap_series(np.arange(5), np.arange(5), np.ones((4, 4)), **options)
    else:
        plot_multi_line_series([np.arange(5)], [np.arange(5)], ["H"], **options)
    try:
        ax = state["axes"]
        assert state["x_ticks"] == _visible_major_ticks(ax.xaxis)
        assert state["y_ticks"] == _visible_major_ticks(ax.yaxis)
        assert len(ax.xaxis.get_minorticklocs()) > 0
        assert len(ax.yaxis.get_minorticklocs()) > 0
        assert (tmp_path / "ticks.png").is_file()
    finally:
        plt.close(state["figure"])


def test_new_view_and_edits_reject_inflight_and_debouncing_preview_results(monkeypatch):
    def inspect(window, app):
        window._preview_worker_stop.clear()
        window._active_preview_generation = 101
        window._set_preview_loading(True)
        window.position_view_type.setCurrentIndex(1)
        current = deepcopy(window._series_descriptors_data)
        settings = window._collect_settings()
        window._pending_preview_request = (settings, False)
        monkeypatch.setattr(window, "_start_pending_preview_if_available", lambda: False)
        window._handle_preview_worker_finished(
            101,
            {
                "render_state": {
                    "series_descriptors": [{"series_id": "stale", "default_label": "Old"}],
                    "series_labels": ["Old"],
                    "x_ticks": [999],
                }
            },
        )
        assert window._series_descriptors_data == current
        # Also reject a result arriving during debounce, before a new job has been queued.
        window._pending_preview_request = None
        window._active_preview_generation = 102
        with monkeypatch.context() as patch:
            patch.setattr(window._preview_timer, "isActive", lambda: True)
            assert window._preview_worker_result_is_stale(102)
        assert window._preview_worker_result_is_stale(101)
        window._preview_worker_stop.set()

    exercise_panel(monkeypatch, "position", inspect)


@pytest.fixture
def simulated_system_palette():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    original = app.palette()
    system = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        system.setColor(role, QColor("#101010"))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        system.setColor(role, QColor("#eeeeee"))
    app.setPalette(system)
    try:
        yield app, system
    finally:
        app.setPalette(original)
        app.processEvents()


def assert_light_window(widget):
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QCheckBox
    from linak.gui.styles import light_colors

    colors = light_colors()
    assert widget.palette().color(QPalette.ColorRole.Window).name() == colors["window_bg"]
    assert widget.palette().color(QPalette.ColorRole.Base).name() == colors["input_bg"]
    assert widget.palette().color(QPalette.ColorRole.Text).name() == colors["text"]
    assert not any(box.objectName() == "themeSwitch" for box in widget.findChildren(QCheckBox))
    assert not hasattr(widget, "_theme_mode")


def test_studio_stays_light_for_system_palette_changes_and_detached_preview(
    monkeypatch, simulated_system_palette
):
    import logging
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QLineEdit, QMenu, QMessageBox
    from matplotlib.colors import to_rgba

    app, system = simulated_system_palette

    def inspect(window, application):
        assert_light_window(window)
        app.sendEvent(window, QEvent(QEvent.Type.ApplicationPaletteChange))
        app.processEvents()
        assert_light_window(window)
        for child in (QMenu(window), QMessageBox(window), QLineEdit(window)):
            assert_light_window(child)
            child.deleteLater()
        disabled = QLineEdit(window)
        disabled.setEnabled(False)
        disabled.ensurePolished()
        assert disabled.palette().color(disabled.foregroundRole()).name() == "#8190a3"
        disabled.deleteLater()
        window._session_diagnostics.session.publish("A warning", level=logging.WARNING)
        app.processEvents()
        assert "#92400e" in window._status_label.styleSheet()
        window._session_diagnostics.session.publish("An error", level=logging.ERROR)
        app.processEvents()
        assert "#b91c1c" in window._status_label.styleSheet()
        transparent = QPixmap(40, 40)
        transparent.fill(Qt.GlobalColor.transparent)
        result = window._preview_display_pixmap(transparent).toImage()
        assert result.pixelColor(1, 1).name() == "#dce4ee"
        assert result.pixelColor(20, 1).name() == "#f4f7fb"
        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#123456")
        ax.plot([0, 1], [1, 2], color="#abcdef")
        window._install_preview_figure(fig)
        window._handle_detach_preview()
        assert_light_window(window._detached_preview_window)
        assert fig.get_facecolor() == to_rgba("#123456")
        assert ax.lines[0].get_color() == "#abcdef"
        window._handle_dock_preview()
        window.resize(1000, 700)
        app.processEvents()
        assert_light_window(window)

    exercise_panel(monkeypatch, "position", inspect)


def test_workspace_and_settings_dialog_stay_light_under_system_palette(
    simulated_system_palette, tmp_path
):
    from ase import Atoms
    from ase.io import write
    from linak.gui.actions import ActionRegistry
    from linak.gui.detection import detect_project_item
    from linak.gui.model import ProjectStore
    from linak.gui.workspace import ProjectWorkspaceWindow, SettingsDialog

    app, system = simulated_system_palette
    store = ProjectStore(tmp_path / "project")
    store.initialize()
    workspace = ProjectWorkspaceWindow(store)
    source = tmp_path / "trajectory.xyz"
    write(source, Atoms("O", positions=[[0, 0, 0]]))
    dialog = None
    try:
        workspace.show()
        app.processEvents()
        assert_light_window(workspace.window)
        dialog = SettingsDialog(
            workspace.window,
            action=ActionRegistry().by_id("convert"),
            item=detect_project_item(source, origin="external"),
            project_dir=store.project_dir,
        )
        dialog.dialog.show()
        app.setPalette(system)
        app.processEvents()
        assert_light_window(workspace.window)
        assert_light_window(dialog.dialog)
    finally:
        if dialog is not None:
            dialog.dialog.close()
            dialog.dialog.deleteLater()
        workspace.window.close()
        workspace.window.deleteLater()
        app.processEvents()
