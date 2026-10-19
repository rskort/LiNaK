from __future__ import annotations

import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from ase import Atoms
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from linak import cli
from linak.analysis.density import compute_density_profile, save_density_profiles
from linak.plot.plot_gui import _coerce_series_descriptors


def test_descriptor_normalization_preserves_canonical_species_and_backing_data():
    raw = {
        "series_id": "water",
        "default_label": "H2O",
        "density_species": "mol:H2O",
        "density_profiles": [{"index": 4}],
        "position_species": "mol:H2O",
        "atom_indices": [1, 2, 3],
    }
    result = _coerce_series_descriptors([raw])[0]
    assert result["density_species"] == "mol:H2O"
    assert result["position_species"] == "mol:H2O"
    assert result["atom_indices"] == [1, 2, 3]
    result["density_profiles"][0]["index"] = 9
    assert raw["density_profiles"] == [{"index": 4}]


@pytest.mark.parametrize("axis", ["z", "x"])
def test_density_gui_real_preview_layer_identity_visibility_colors_and_order(
    monkeypatch, tmp_path, axis
):
    frame = Atoms(
        ["Pt", "O", "H", "H", "H", "K"],
        positions=[[8, 8, 1], [2, 2, 3], [2.9, 2, 3], [2, 2.9, 3], [7, 7, 6], [9, 9, 7]],
        cell=[12, 12, 12],
        pbc=True,
    )
    profiles = [
        compute_density_profile(
            [frame], species=species, axis=axis, bin_width=0.5, surface_mode="none"
        )
        for species in ("Pt", "H", "K", "O", "mol:H2O")
    ]
    source = save_density_profiles(profiles, tmp_path / "cell.density.h5")
    app = QApplication.instance() or QApplication([])
    if os.name == "nt" and not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    callbacks = {}
    launcher = cli._open_plot_settings_gui
    checked = []
    failures = []

    def launch(**kwargs):
        callbacks.update(kwargs)
        return launcher(**kwargs)

    def inspect(application):
        application.processEvents()
        window = next(
            w
            for w in application.topLevelWidgets()
            if hasattr(w, "_series_enabled_data") and w.isVisible()
        )
        try:
            window._preview_timer.stop()
            # This integration test renders synchronously below. Stop the normal
            # background renderer first because Matplotlib's global math-text parser
            # is not thread-safe on the Python 3.9-compatible Matplotlib release.
            window._preview_generation += 1
            window._preview_worker_stop.set()
            window._preview_worker_queue.put_nowait(None)
            worker = window._preview_worker_thread
            if worker is not None and worker.is_alive():
                worker.join(timeout=5.0)
            assert worker is None or not worker.is_alive()
            # Exercise real settings and rendering synchronously, without a racing worker.
            monkeypatch.setattr(
                window, "_schedule_or_apply_series_preview_update", lambda **kw: None
            )
            monkeypatch.setattr(window, "_schedule_preview_update", lambda *args: None)
            for row in range(window._nav_list.count()):
                if window._nav_list.item(row).text() == "Layers":
                    window._nav_list.setCurrentRow(row)
                    break
            application.processEvents()
            descriptors = window._series_descriptors_data
            assert [d["default_label"] for d in descriptors] == ["Pt", "H", "K", "O", "H2O"]
            assert descriptors[-1]["density_species"] == "mol:H2O"
            original_colors = {
                d["series_id"]: window._effective_series_color(i) for i, d in enumerate(descriptors)
            }

            def render_and_check():
                settings = window._collect_settings()
                state = callbacks["on_preview_figure"](settings)
                try:
                    expected = [
                        (i, d)
                        for i, d in enumerate(window._series_descriptors_data)
                        if window._series_enabled_data[i]
                    ]
                    lines = state["axes"].lines
                    assert [line.get_label() for line in lines] == [
                        window._effective_series_label(i) for i, _d in expected
                    ]
                    assert [row["series_id"] for row in state["plotted_xy_series"]] == [
                        d["series_id"] for _i, d in expected
                    ]
                    for line, (i, descriptor) in zip(lines, expected):
                        assert to_rgba(line.get_color()) == to_rgba(
                            window._effective_series_color(i)
                        )
                        if descriptor["series_id"] in original_colors:
                            assert to_rgba(line.get_color()) == to_rgba(
                                original_colors[descriptor["series_id"]]
                            )
                    # A preview's metadata refresh must retain visibility, colors and copies.
                    before = deepcopy(settings)
                    window._apply_series_defaults(
                        state["series_labels"], descriptors=state["series_descriptors"]
                    )
                    assert (
                        window._collect_settings()["series_overrides"] == before["series_overrides"]
                    )
                finally:
                    plt.close(state["figure"])
                return settings

            render_and_check()
            window._set_layer_indices_enabled([0], False)
            render_and_check()  # H retains orange when the substrate is hidden.
            window._set_layer_indices_enabled([1, 2], False)
            render_and_check()  # O and water both render.
            window._set_layer_indices_enabled([3], False)
            render_and_check()  # Water renders with elemental O disabled.
            hide = next(
                b for b in window._tab_series.findChildren(QPushButton) if b.text() == "Hide all"
            )
            hide.click()
            browser = window.series_list
            water_row = window._display_row_for_selection(4)
            browser.scrollTo(browser.model().index(water_row, 0))
            application.processEvents()
            rect = browser.visualItemRect(browser.item(water_row))
            QTest.mouseClick(
                browser.viewport(),
                Qt.MouseButton.LeftButton,
                pos=browser.itemDelegate()._checkbox_rect(rect).center(),
            )
            assert window._series_enabled_data == [False, False, False, False, True]
            render_and_check()
            show = next(
                b for b in window._tab_series.findChildren(QPushButton) if b.text() == "Show all"
            )
            show.click()
            window._layer_search.setText("H2O")
            window._layer_match_buttons[0].click()
            render_and_check()
            window._layer_search.clear()
            show.click()
            browser.setCurrentRow(window._display_row_for_selection(4))
            window._isolate_active_layer()
            render_and_check()
            show.click()
            water_id = descriptors[4]["series_id"]
            window._move_series_by_delta(water_id, -4)
            render_and_check()  # Plot and legend use the same order as the list.
            browser.setCurrentRow(window._display_row_for_selection(0))
            window._series_duplicate_button.click()
            copy_id = window._series_descriptors_data[window._series_active_index]["series_id"]
            window._move_series_by_delta(copy_id, -99)
            window._set_all_series_enabled(False)
            window._set_layer_indices_enabled([0], True)
            saved = render_and_check()  # A copy works with its original disabled.
            export_path = tmp_path / "water-copy.png"
            _message, exported = callbacks["on_save_figure"](saved, str(export_path))
            assert export_path.is_file()
            assert len(exported["axes"].lines) == 1
            assert to_rgba(exported["axes"].lines[0].get_color()) == to_rgba(
                window._effective_series_color(0)
            )
            callbacks["on_save"]("Layer regression", saved)
            restored = callbacks["on_load_profile"]("Layer regression")
            assert copy_id in {d["series_id"] for d in restored["series_descriptors"]}
            assert restored["series_overrides"][copy_id]["enabled"]
            window._populate(restored)
            render_and_check()
            checked.append(True)
        except BaseException as error:
            failures.append(error)
            raise
        finally:
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(cli, "_open_plot_settings_gui", launch)
    monkeypatch.setattr(QApplication, "exec", inspect)
    result = cli.main(["--log-level", "ERROR", "plot", str(source), "--gui", "--x-mode", axis])
    if failures:
        raise failures[0]
    assert result == 0
    app.processEvents()
    assert checked == [True]
