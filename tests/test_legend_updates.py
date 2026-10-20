"""Legend entries and presentation stay consistent through live layer edits."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib import pyplot as plt

from linak.plot.plotting import (
    _update_legend,
    plot_multi_line_series,
    with_style_overrides,
)


def legend_signature(ax):
    legend = ax.get_legend()
    if legend is None:
        return None
    return {
        "labels": [t.get_text() for t in legend.get_texts()],
        "widths": [line.get_linewidth() for line in legend.get_lines()],
        "colors": [line.get_color() for line in legend.get_lines()],
        "font": [t.get_fontsize() for t in legend.get_texts()],
        "title": legend.get_title().get_text(),
        "title_size": legend.get_title().get_fontsize(),
        "frame": legend.get_frame_on(),
        "columns": legend._ncols,
        "location": legend._loc,
        "borderpad": legend.borderpad,
        "anchor": tuple(legend.get_bbox_to_anchor()._bbox.bounds),
    }


def test_shared_legend_refresh_handles_empty_entries_and_preserves_options():
    fig, ax = plt.subplots()
    (first,) = ax.plot([0, 1], [1, 2], label="A")
    (second,) = ax.plot([0, 1], [2, 3], label="B")
    options = dict(
        frameon=False, ncols=2, title_fontsize=19, borderpad=2.5, bbox_to_anchor=(0.9, 0.8)
    )
    style = with_style_overrides(legend_font_size=13, font_color="purple")

    def refresh():
        return _update_legend(
            ax,
            style=style,
            title="Runs",
            location="upper left",
            options=options,
            artist_order=sorted(ax.lines, key=lambda line: line.get_zorder()),
        )

    try:
        refresh()
        first.set_linewidth(7)
        first.set_label("Renamed")
        first.set_zorder(4)
        second.set_zorder(2)
        refresh()
        state = legend_signature(ax)
        assert state["labels"] == ["B", "Renamed"]
        assert state["widths"] == [1.5, 7]
        assert state["font"] == [13, 13]
        assert state["title_size"] == 19
        assert not state["frame"] and state["columns"] == 2
        assert all(t.get_color() == "purple" for t in ax.get_legend().get_texts())
        second.set_visible(False)
        first.set_label("_nolegend_")
        assert refresh() is None
        first.set_label("Restored")
        refresh()
        assert legend_signature(ax)["labels"] == ["Restored"]
        assert legend_signature(ax)["borderpad"] == 2.5
        assert _update_legend(ax, style=style, enabled=False) is None
    finally:
        plt.close(fig)


def test_full_renderer_retains_real_labels_when_excluded_from_legend():
    state: dict = {}
    plot_multi_line_series(
        [np.arange(3)],
        [np.arange(3)],
        ["Actual name"],
        title="Test",
        x_label="X",
        y_label="Y",
        show=False,
        series_show_in_legend=[False],
        legend_kwargs={"borderpad": 2, "bbox_to_anchor": [0.9, 0.8]},
        capture_state=state,
    )
    try:
        assert state["series_labels"] == ["Actual name"]
        assert state["axes"].get_legend() is None
        assert len(state["series_artists"]) == 1
        assert state["legend_kwargs"]["bbox_to_anchor"] == [0.9, 0.8]
    finally:
        plt.close(state["figure"])


@pytest.mark.parametrize("included", [False, True])
@pytest.mark.parametrize("error_style", ["band", "whiskers"])
def test_derived_legend_entries_follow_their_own_inclusion_settings(included, error_style):
    from linak.analysis.statistics import SeriesStatistics
    from linak.plot.plotting import plot_line_series

    state: dict = {}
    plot_line_series(
        np.arange(5),
        np.arange(5) + 1,
        title="Derived",
        x_label="X",
        y_label="Y",
        show=False,
        line_label="Raw",
        show_in_legend=included,
        fit_enabled=True,
        fit_label="Fit",
        fit_show_in_legend=included,
        cumulative_config={"enabled": True, "label_override": "Mean", "show_in_legend": included},
        series_statistics=SeriesStatistics(
            point_count=np.full(5, 10),
            sample_n=np.full(5, 10),
            sample_std=np.full(5, 0.3),
            sample_sem=np.full(5, 0.1),
        ),
        error_config={
            "enabled": True,
            "stat": "sample_sem",
            "style": error_style,
            "show_in_legend": included,
            "label_override": "Uncertainty",
        },
        legend_title="Retained",
        legend_kwargs={"frameon": False, "ncols": 2},
        capture_state=state,
    )
    try:
        if included:
            signature = legend_signature(state["axes"])
            assert signature["labels"] == ["Raw", "Uncertainty", "Fit", "Mean"]
            assert signature["title"] == "Retained" and not signature["frame"]
        else:
            assert state["axes"].get_legend() is None
        assert len(state["axes"].lines) >= 3
    finally:
        plt.close(state["figure"])


def test_studio_legend_live_edits_match_render_export_and_profile(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from ase import Atoms
    from linak import cli
    from linak.analysis.density import compute_density_profile, save_density_profiles

    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)
    app = QApplication.instance() or QApplication([])
    frame = Atoms("HK", positions=[[1, 1, 2], [3, 3, 4]], cell=[10] * 3, pbc=True)
    source = save_density_profiles(
        [
            compute_density_profile([frame], species=s, surface_mode="none", bin_width=1)
            for s in ("H", "K")
        ],
        tmp_path / "layers.density.h5",
    )
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
            if hasattr(w, "series_show_in_legend") and w.isVisible()
        )
        try:
            window._preview_timer.stop()
            window._preview_generation += 1
            window._preview_worker_stop.set()
            window._preview_worker_queue.put_nowait(None)
            worker = window._preview_worker_thread
            if worker is not None:
                worker.join(timeout=5)
                assert not worker.is_alive()
            window._set_preview_loading(False)
            window._suspend_preview_events = True
            window.legend_kwargs_json.setPlainText(
                '{"borderpad": 2.5, "bbox_to_anchor": [0.95, 0.85]}'
            )
            window.legend_title.setText("Custom legend")
            window.legend_font.setText("13")
            window.legend_title_font.setText("17")
            window.legend_columns.setText("2")
            window.legend_loc.setCurrentText("upper left")
            window.legend_frame_mode.setCurrentText("off")
            window._suspend_preview_events = False

            def render():
                state = callbacks["on_preview_figure"](window._collect_settings())
                window._install_preview_figure(state["figure"])
                window._apply_preview_state_to_synced_fields(state)
                return state["axes"]

            ax = render()
            # Actual signals must take the immediate path, not require toggling the legend.
            window.line_width.setText("5")
            assert [line.get_linewidth() for line in ax.lines] == [5, 5]
            assert legend_signature(ax)["widths"] == [5, 5]
            window.series_list.setCurrentRow(window._display_row_for_selection(0))
            window.series_line_width.setText("7")
            assert legend_signature(ax)["widths"] == [7, 5]
            window.line_width.setText("3")
            assert legend_signature(ax)["widths"] == [7, 3]  # Layer override survives global edit.
            window.series_color.setText("#881122")
            window.series_alpha.setText("0.4")
            window.line_style.setCurrentText("--")
            window.markers_mode.setCurrentText("on")
            window.marker_type.setCurrentText("s")
            proxy = ax.get_legend().get_lines()[0]
            assert proxy.get_color() == "#881122" and proxy.get_alpha() == 0.4
            assert proxy.get_linestyle() == "--" and proxy.get_marker() == "s"
            window.series_alpha.clear()
            assert ax.get_legend().get_lines()[0].get_alpha() is None
            window.series_show_in_legend.setCurrentText("off")
            assert len(ax.lines) == 2 and all(line.get_visible() for line in ax.lines)
            assert legend_signature(ax)["labels"] == ["K"]
            window.series_label.setText("Hydrogen renamed")
            window.series_show_in_legend.setCurrentText("on")
            assert legend_signature(ax)["labels"] == ["Hydrogen renamed", "K"]
            for enabled in (False, True, False, True):
                window.series_show_in_legend.setCurrentText("on" if enabled else "off")
                assert len(ax.get_legend().get_texts()) == (2 if enabled else 1)
            window.series_list.setCurrentRow(window._display_row_for_selection(1))
            window.series_show_in_legend.setCurrentText("off")
            window.series_list.setCurrentRow(window._display_row_for_selection(0))
            window.series_show_in_legend.setCurrentText("off")
            assert ax.get_legend() is None
            window.series_show_in_legend.setCurrentText("on")
            assert legend_signature(ax)["labels"] == ["Hydrogen renamed"]
            # Reset an override rather than retaining the last width.
            window.series_line_width.clear()
            assert legend_signature(ax)["widths"] == [3]
            before = legend_signature(ax)
            assert before["title"] == "Custom legend" and before["title_size"] == 17
            assert before["font"] == [13] and before["columns"] == 2 and not before["frame"]
            assert before["borderpad"] == 2.5
            fresh = callbacks["on_preview_figure"](window._collect_settings())
            try:
                assert legend_signature(fresh["axes"]) == before
            finally:
                plt.close(fresh["figure"])
            callbacks["on_save"]("Legend regression", window._collect_settings())
            restored = callbacks["on_load_profile"]("Legend regression")
            window._populate(restored)
            ax = render()
            assert legend_signature(ax) == before
            path = tmp_path / "legend.png"
            _, saved = callbacks["on_save_figure"](window._collect_settings(), str(path))
            assert path.is_file()
            # Export figure geometry differs, so compare the normalized presentation separately.
            exported = legend_signature(saved["axes"])
            assert {k: v for k, v in exported.items() if k != "anchor"} == {
                k: v for k, v in before.items() if k != "anchor"
            }
            window.series_list.setCurrentRow(window._display_row_for_selection(1))
            window.series_show_in_legend.setCurrentText("on")
            second_id = window._series_descriptors_data[1]["series_id"]
            window._move_series_by_delta(second_id, -1)
            assert legend_signature(ax)["labels"] == ["K", "Hydrogen renamed"]
            window._series_duplicate_button.click()
            assert not window._apply_series_artist_updates_to_canvas()
            ax = render()
            assert legend_signature(ax)["labels"] == [
                window._effective_series_label(i) for i in range(3)
            ]
            assert legend_signature(ax)["columns"] == 2
            window._handle_detach_preview()
            ax = render()
            window.line_width.setText("4")
            assert legend_signature(ax)["widths"] == [4, 4, 4]
            window._handle_dock_preview()
            ax = render()
            # Busy previews queue the latest legend settings rather than changing an old canvas.
            with monkeypatch.context() as patch:
                queued = []
                patch.setattr(window._preview_timer, "start", lambda *args: queued.append(args))
                window._preview_loading = True
                window.legend_title.setText("Queued title")
                assert queued and legend_signature(ax)["title"] == "Custom legend"
                window._preview_loading = False
                assert window._refresh_preview_legend()
                assert legend_signature(ax)["title"] == "Queued title"
            # Derived artists and absent/new sources deliberately use the full-render path.
            window._series_fit_enabled_data[0] = True
            assert not window._apply_series_artist_updates_to_canvas()
            window._series_fit_enabled_data[0] = False
            window._last_preview_state.pop("series_artists")
            assert not window._apply_series_artist_updates_to_canvas()
            checked.append(True)
        except BaseException as error:
            failures.append(error)
            raise
        finally:
            window._preview_timer.stop()
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(cli, "_open_plot_settings_gui", launch)
    monkeypatch.setattr(QApplication, "exec", inspect)
    result = cli.main(["plot", str(source), "--gui", "--x-mode", "z"])
    if failures:
        raise failures[0]
    assert result == 0
    app.processEvents()
    assert checked == [True]
