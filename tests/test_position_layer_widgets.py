from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from matplotlib.figure import Figure

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from linak.plot.plot_gui import launch_plot_settings_panel


@pytest.mark.parametrize("dark", [False, True])
def test_position_layer_browser_visibility_style_and_history(monkeypatch, tmp_path, dark):
    app = QApplication.instance() or QApplication([])
    if os.name == "nt" and not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    # Interleaving species in the source catalog must not mix their browser groups.
    species = ["Pt", "mol:H2O", "Pt", "mol:H2O"]
    descriptors = [
        {
            "series_id": f"trajectory:{index}",
            "source_series_id": f"trajectory:{index}",
            "default_label": f"cell1:{label}[{index}]",
            "position_species": label,
            "atom_index": index,
            "source_name": "cell1.position.h5",
        }
        for index, label in enumerate(species)
    ]
    initial = {
        "component": "2d-projection",
        "projection_render_mode": "line-colors",
        "series_descriptors": descriptors,
        "series_enabled": [True, True, False, True],
        "line_colors": ["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
    }
    checked = []

    def inspect(application):
        application.processEvents()
        window = next(
            widget
            for widget in application.topLevelWidgets()
            if widget.windowTitle() == "Position layers integration" and widget.isVisible()
        )
        try:
            window._preview_timer.stop()
            monkeypatch.setattr(
                window, "_schedule_or_apply_series_preview_update", lambda **kw: None
            )
            window._theme_mode = "dark" if dark else "light"
            window._apply_theme_styles()
            window.resize(1100, 720)
            for row in range(window._nav_list.count()):
                if window._nav_list.item(row).text() == "Layers":
                    window._nav_list.setCurrentRow(row)
                    break
            figure = Figure(figsize=(5, 3))
            figure.add_subplot().plot([0, 1, 2], [0, 2, 1])
            window._install_preview_figure(figure)
            application.processEvents()
            assert window._series_species_members == {"Pt": [0, 2], "mol:H2O": [1, 3]}
            assert [row["kind"] for row in window._series_display_rows] == [
                "species",
                "base",
                "base",
                "species",
                "base",
                "base",
            ]
            browser = window.series_list
            # Every layer action stays above the list, including in narrow panels.
            for text in ("Show all", "Hide all", "Duplicate", "Delete", "Isolate"):
                buttons = [
                    b for b in window._tab_series.findChildren(QPushButton) if b.text() == text
                ]
                assert buttons, text
                button = buttons[0]
                assert button.isVisible(), text
                assert (
                    button.mapTo(window, button.rect().bottomLeft()).y()
                    < browser.mapTo(window, browser.rect().topLeft()).y()
                )
            assert not any(b.text() == "Add Group" for b in window.findChildren(QPushButton))
            # A checkbox works immediately without selecting its layer first.
            browser.setCurrentRow(window._display_row_for_selection(0))
            selected_id = window._active_series_row_id()
            target = window._display_row_for_selection(1)
            browser.scrollTo(browser.model().index(target, 0))
            application.processEvents()
            rect = browser.visualItemRect(browser.item(target))
            QTest.mouseClick(
                browser.viewport(),
                Qt.MouseButton.LeftButton,
                pos=browser.itemDelegate()._checkbox_rect(rect).center(),
            )
            assert window._series_enabled_data == [True, False, False, True]
            assert window._active_series_row_id() == selected_id
            QTest.mouseClick(
                browser.viewport(),
                Qt.MouseButton.LeftButton,
                pos=browser.itemDelegate()._checkbox_rect(rect).center(),
            )
            assert window._series_enabled_data == [True, True, False, True]
            browser.scrollToTop()
            header = browser.itemDelegate()._row_state(browser.model().index(0, 0))
            assert header["partial"] and not header["checked"]
            ids = window._current_series_id_order()
            colors = list(window._series_colors_data)
            # Collapse is a browser action and never changes plotted visibility.
            browser.setCurrentRow(0)
            QTest.keyClick(browser, Qt.Key.Key_Left)
            assert browser.isRowHidden(1) and browser.isRowHidden(2)
            assert window._series_enabled_data == [True, True, False, True]
            window._layer_search.setText("Pt [2]")
            assert window._matching_layer_indices() == [2]
            assert not browser.isRowHidden(2)
            window._set_matching_layers_enabled(True)
            assert window._series_enabled_data == [False, False, True, False]
            assert window._current_series_id_order() == ids
            assert window._series_colors_data == colors
            window._layer_search.clear()
            window._set_all_series_enabled(True)
            window._toggle_layer_species("Pt")
            browser.setCurrentRow(0)
            QTest.keyClick(browser, Qt.Key.Key_Space)
            assert window._series_enabled_data == [False, True, False, True]
            assert window._current_series_id_order() == ids
            window._layer_search.setText("Pt")
            assert window._matching_layer_indices() == [0, 2]
            window._layer_search.setText("missing atom")
            assert window._layer_empty_label.isVisible()
            window._reset_undo_history()
            window._set_matching_layers_enabled(True)
            assert not any(window._series_enabled_data)
            window._handle_undo()
            assert window._series_enabled_data == [False, True, False, True]
            window._layer_search.clear()
            window._set_all_series_enabled(True)
            browser.setCurrentRow(window._display_row_for_selection(2))
            window._reset_undo_history()
            window._isolate_active_layer()
            assert window._series_enabled_data == [False, False, True, False]
            window._handle_undo()
            assert window._series_enabled_data == [True] * 4
            assert window._current_series_id_order() == ids
            # A species style action targets only its own trajectories.
            browser.setCurrentRow(window._display_row_for_selection(2))
            window._copy_style_to_species()
            assert window._series_colors_data == ["#0000ff", "#00ff00", "#0000ff", "#ffff00"]
            window._move_series_by_delta(ids[2], -1)
            assert window._current_series_id_order().index(
                ids[2]
            ) < window._current_series_id_order().index(ids[0])
            assert window._layer_species(window._series_active_index) == "Pt"
            assert window._series_colors_data[window._series_active_index] == "#0000ff"
            # Automatic colors are visible styles too, and can be copied to a species.
            window.series_color.clear()
            default_color = window._effective_series_color(window._series_active_index)
            window._copy_style_to_species()
            assert all(
                window._series_colors_data[index] == default_color
                for index in window._series_species_members["Pt"]
            )
            # The header checkbox also supports mouse input; it acts on the entire species.
            browser.scrollToTop()
            application.processEvents()
            rect = browser.visualItemRect(browser.item(0))
            QTest.mouseClick(
                browser.viewport(),
                Qt.MouseButton.LeftButton,
                pos=rect.topLeft()
                + browser.itemDelegate()._checkbox_rect(rect).center()
                - rect.topLeft(),
            )
            assert all(
                not window._series_enabled_data[index]
                for index in window._series_species_members["Pt"]
            )
            # Real Qt layout and painting, including narrower panels in both themes.
            for width in (1100, 900):
                window.resize(width, 720)
                application.processEvents()
                assert browser.isVisible()
                assert window._layers_splitter.count() == 2
                assert window.grab().save(str(tmp_path / f"layers_{dark}_{width}.png"))
            checked.append(True)
        finally:
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect)
    launch_plot_settings_panel(
        title="Position layers integration",
        initial_settings=initial,
        analysis_name="position",
        on_preview=lambda settings: {},
        on_save=lambda name, settings: "Saved",
    )
    app.processEvents()
    assert checked == [True]


def test_density_layers_first_click_filter_copy_and_delete(monkeypatch):
    app = QApplication.instance() or QApplication([])
    checked = []

    def inspect(application):
        application.processEvents()
        window = next(
            w
            for w in application.topLevelWidgets()
            if w.windowTitle() == "Density layer actions" and w.isVisible()
        )
        try:
            window._preview_timer.stop()
            monkeypatch.setattr(
                window, "_schedule_or_apply_series_preview_update", lambda **kw: None
            )
            monkeypatch.setattr(window, "_schedule_preview_update", lambda *args: None)
            for row in range(window._nav_list.count()):
                if window._nav_list.item(row).text() == "Layers":
                    window._nav_list.setCurrentRow(row)
                    break
            application.processEvents()
            browser = window.series_list
            browser.setCurrentRow(0)
            initial_id = window._active_series_row_id()
            rect = browser.visualItemRect(browser.item(1))
            point = browser.itemDelegate()._checkbox_rect(rect).center()
            QTest.mouseClick(browser.viewport(), Qt.MouseButton.LeftButton, pos=point)
            assert window._series_enabled_data == [True, False, True]
            assert window._active_series_row_id() == initial_id
            QTest.mouseClick(browser.viewport(), Qt.MouseButton.LeftButton, pos=point)
            assert window._series_enabled_data == [True, True, True]
            # Right click is not a visibility action.
            QTest.mouseClick(browser.viewport(), Qt.MouseButton.RightButton, pos=point)
            assert window._series_enabled_data == [True, True, True]
            browser.setCurrentRow(0)
            window._reset_undo_history()
            window._layer_search.setText("Na")
            assert window._series_enabled_data == [True, True, True]
            window._layer_match_buttons[0].click()
            assert window._series_enabled_data == [False, True, False]
            window._handle_undo()
            assert window._series_enabled_data == [True, True, True]
            window._layer_match_buttons[1].click()
            assert window._series_enabled_data == [True, False, True]
            window._layer_search.clear()
            browser.setCurrentRow(0)
            QTest.keyClick(browser, Qt.Key.Key_Space)
            assert window._series_enabled_data == [False, False, True]
            window._series_duplicate_button.click()
            assert len(window._series_descriptors_data) == 4
            copy = window._series_descriptors_data[window._series_active_index]
            assert copy["is_generated"] and copy["source_series_id"] == "source:K"
            assert "member_series_ids" not in copy
            assert window._series_delete_button.isEnabled()
            window._series_delete_button.click()
            assert len(window._series_descriptors_data) == 3
            assert not window._series_delete_button.isEnabled()
            checked.append(True)
        finally:
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect)
    launch_plot_settings_panel(
        title="Density layer actions",
        analysis_name="density",
        initial_settings={
            "series_descriptors": [
                {
                    "series_id": f"source:{label}",
                    "source_series_id": f"source:{label}",
                    "default_label": label,
                    "source_path": "/project/LiNaK/density.h5",
                    "source_name": "density.h5",
                }
                for label in ("K", "Na", "Pt")
            ],
            "series_enabled": [True, True, True],
        },
        on_preview=lambda settings: {},
        on_save=lambda name, settings: "Saved",
    )
    app.processEvents()
    assert checked == [True]
