from __future__ import annotations

import logging
import os
import threading
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# Load Matplotlib before Qt to avoid the Qt 6.11 signature loader inspecting six.moves.
from matplotlib.figure import Figure  # noqa: F401

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from linak.plot.session_log import SessionLog
from linak.plot.session_widgets import JsonOptionsEditor, PreviewLoadingOverlay, SessionDiagnostics


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    if os.name == "nt" and not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    return application


@pytest.fixture
def diagnostics(app):
    parent = QWidget()
    session = SessionLog()
    panel = SessionDiagnostics(session, parent)
    panel.body.resize(400, 280)
    panel.body.show()
    panel.footer.resize(320, 40)
    panel.footer.show()
    yield panel
    panel.shutdown()
    parent.close()
    parent.deleteLater()
    app.processEvents()


def test_json_options_editor_validation_examples_format_and_undo(app):
    import json

    options = JsonOptionsEditor("Extra legend spacing.", '{"borderpad": 0.8}')
    try:
        assert options.is_valid
        assert options.validation_label.text() == "No extra options"
        assert not options.format_button.isEnabled()
        options.example_button.click()
        assert json.loads(options.editor.toPlainText()) == {"borderpad": 0.8}
        assert not options.example_button.isEnabled()
        options.format_button.click()
        assert options.editor.toPlainText() == '{\n  "borderpad": 0.8\n}'
        options.editor.undo()
        assert options.editor.toPlainText() == '{"borderpad": 0.8}'
        options.clear_button.click()
        assert options.editor.toPlainText() == ""
        options.editor.undo()
        assert json.loads(options.editor.toPlainText()) == {"borderpad": 0.8}
        options.editor.setPlainText('{\n "borderpad": }')
        assert not options.is_valid
        assert "Line 2, column" in options.validation_label.text()
        assert not options.format_button.isEnabled()
        assert not options.example_button.isEnabled()
        options.editor.setPlainText("[]")
        assert not options.is_valid
        assert "JSON object" in options.validation_label.text()
        options.editor.setPlainText('{"borderpad": 1.0, "labelspacing": 0.5}')
        assert options.is_valid
        assert options.validation_label.text() == "2 options configured"
        assert options.validation_label.styleSheet() == ""
        options.resize(320, 250)
        options.show()
        app.processEvents()
        assert options.width() == 320
        for button in (options.format_button, options.example_button, options.clear_button):
            assert button.geometry().right() <= options.width()
    finally:
        options.close()
        options.deleteLater()
        app.processEvents()


def test_worker_order_newest_footer_and_full_history(app, diagnostics):
    panel = diagnostics
    worker = threading.Thread(
        target=lambda: panel.session.publish("worker warning", level=logging.WARNING)
    )
    worker.start()
    worker.join()
    # The older worker notification is still queued when this GUI event arrives.
    panel.footer.setText("Preview updated.")
    app.processEvents()
    assert panel.footer.text() == "Preview updated."
    history = panel.viewer.toPlainText()
    assert history.index("worker warning") < history.index("Preview updated.")
    assert len(panel.session.snapshot()) == 2


def test_single_line_elision_details_accessibility_and_clear(app, diagnostics):
    panel = diagnostics
    panel.session.publish(
        "<b>Error</b>\n" + "long message " * 50, level=logging.ERROR, details="traceback\nlast line"
    )
    app.processEvents()
    from PySide6.QtWidgets import QLabel

    visible_text = QLabel.text(panel.footer)
    assert "\n" not in visible_text
    assert visible_text.endswith("\u2026")
    assert "<b>Error</b>" in panel.footer.toolTip()
    assert "traceback\nlast line" in panel.footer.toolTip()
    assert panel.footer.textFormat() == Qt.TextFormat.PlainText
    assert panel.viewer.height() == 240
    assert panel.body.maximumWidth() == 960
    panel.copy_all()
    assert QApplication.clipboard().text() == panel.viewer.toPlainText()
    panel.session.clear()
    assert panel.viewer.toPlainText() == ""
    assert panel.footer.text() == "Ready."


def test_footer_click_and_keyboard(app, diagnostics):
    calls = []
    diagnostics.footer.activated.connect(lambda: calls.append(True))
    QTest.mouseClick(diagnostics.footer, Qt.MouseButton.LeftButton)
    QTest.keyClick(diagnostics.footer, Qt.Key.Key_Return)
    QTest.keyClick(diagnostics.footer, Qt.Key.Key_Space)
    assert len(calls) == 3


def test_history_preserves_scrolled_reading_position(app, diagnostics):
    for index in range(100):
        diagnostics.session.publish(f"Entry {index}")
    app.processEvents()
    scroll = diagnostics.viewer.verticalScrollBar()
    assert scroll.value() == scroll.maximum()
    scroll.setValue(scroll.maximum() // 2)
    before = scroll.value()
    diagnostics.session.publish("new entry")
    assert scroll.value() == before


def test_overlay_delay_resize_repeated_render_and_cleanup(app):
    target = QWidget()
    target.resize(500, 300)
    QLabel("Retained plot", target)
    target.show()
    overlay = PreviewLoadingOverlay(target)
    overlay.set_active(True)
    assert overlay.delay.isActive()
    assert overlay.delay.interval() == 150
    assert not overlay.isVisible()
    deadline = monotonic() + 2.0
    while not overlay.isVisible() and monotonic() < deadline:
        QTest.qWait(10)
    assert overlay.isVisible()
    assert overlay.animation.isActive()
    overlay.set_active(True)
    assert not overlay.delay.isActive()
    target.resize(300, 200)
    app.processEvents()
    assert overlay.geometry() == target.rect()
    assert not overlay.grab().isNull()
    overlay.shutdown()
    assert not overlay.isVisible()
    assert not overlay.animation.isActive()
    assert not overlay.delay.isActive()
    target.close()


def test_fast_render_never_displays_overlay(app):
    target = QWidget()
    target.show()
    overlay = PreviewLoadingOverlay(target)
    overlay.set_active(True)
    overlay.set_active(False)
    QTest.qWait(180)
    assert not overlay.isVisible()
    assert not overlay.animation.isActive()
    target.close()


@pytest.mark.parametrize("fail_latest", [False, True])
def test_setting_change_supersedes_render_and_stale_callbacks_preserve_latest(
    app, monkeypatch, fail_latest
):
    from matplotlib import pyplot as plt
    from linak.cancellation import ComputationCancelled, check_cancelled, register_cancel_cleanup
    from linak.plot.plot_gui import launch_plot_settings_panel

    started = threading.Event()
    cancelled = threading.Event()
    release_old = threading.Event()
    latest_started = threading.Event()
    release_latest = threading.Event()
    stop = threading.Event()
    requests = []
    obsolete_figures = []

    def wait_until(predicate):
        for _ in range(300):
            if predicate():
                return
            QTest.qWait(10)
        pytest.fail("Preview transition did not complete")

    def render(settings):
        if "request" not in settings:
            return {}
        request = settings["request"]
        requests.append(request)
        if request == 0:
            obsolete_figures.append(plt.figure())
            register_cancel_cleanup(lambda: plt.close(obsolete_figures[0]))
            started.set()
            try:
                while not stop.wait(0.005):
                    check_cancelled()
            except ComputationCancelled:
                cancelled.set()
                release_old.wait(3)
                raise
        else:
            latest_started.set()
            while not release_latest.wait(0.005):
                check_cancelled()
            if fail_latest:
                raise ValueError("Latest preview failed")
        figure = plt.figure()
        figure.add_subplot().set_title(f"Request {request}")
        return {"figure": figure}

    checked = []

    def inspect_window(application):
        application.processEvents()
        window = next(
            widget
            for widget in application.topLevelWidgets()
            if widget.windowTitle() == "Preview replacement" and widget.isVisible()
        )
        try:
            wait_until(lambda: not window._preview_loading)
            window._preview_timer.stop()
            retained = Figure()
            retained.add_subplot().set_title("Retained preview")
            window._install_preview_figure(retained)
            current = {"request": 0}
            monkeypatch.setattr(window, "_collect_settings", lambda: dict(current))
            window._start_preview_worker(dict(current), interactive=False)
            wait_until(started.is_set)
            old_generation = window._active_preview_generation
            unrelated = plt.figure()
            # Text edits normally take the immediate canvas styling path. During
            # rendering they must instead schedule the complete newest settings.
            current["request"] = 1
            window.title_text.setText("Changed during render")
            assert window._preview_timer.isActive()
            assert retained.axes[0].get_title() == "Retained preview"
            window._handle_debounced_preview()
            wait_until(cancelled.is_set)
            for request in (2, 3):
                current["request"] = request
                window._update_embedded_preview(interactive=False)
            newest = window._active_preview_generation
            assert newest != old_generation
            assert window._preview_worker_queue.qsize() == 1
            assert window._pending_preview_request is None
            stale = plt.figure()
            window._handle_preview_worker_finished(old_generation, {"figure": stale})
            window._handle_preview_worker_failed(old_generation, ValueError("Obsolete failure"))
            assert not plt.fignum_exists(stale.number)
            assert window._active_preview_generation == newest
            assert window._preview_loading
            assert window._preview_error is None
            assert window._preview_figure is retained
            release_old.set()
            wait_until(latest_started.is_set)
            assert requests == [0, 3]
            assert not plt.fignum_exists(obsolete_figures[0].number)
            assert plt.fignum_exists(unrelated.number)
            plt.close(unrelated)
            assert window._preview_loading
            release_latest.set()
            wait_until(lambda: not window._preview_loading)
            assert window._active_preview_generation is None
            assert not window._active_preview_pane.loading_overlay.isVisible()
            if fail_latest:
                assert window._preview_error == "Latest preview failed"
                assert window._preview_figure is retained
            else:
                assert window._preview_error is None
                assert window._preview_figure.axes[0].get_title() == "Request 3"
            checked.append(True)
        finally:
            stop.set()
            release_old.set()
            release_latest.set()
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect_window)
    launch_plot_settings_panel(
        title="Preview replacement",
        initial_settings={},
        on_preview=lambda settings: {},
        on_preview_figure=render,
        on_save=lambda name, settings: "Saved",
    )
    assert checked == [True]


def test_plot_window_logs_navigation_queued_render_detach_and_failure(app, monkeypatch, tmp_path):
    from linak.plot.plot_gui import launch_plot_settings_panel
    import warnings

    release = threading.Event()
    original_warning = warnings.showwarning
    original_handlers = tuple(logging.getLogger().handlers)

    def render(settings):
        release.wait(5)
        return {}

    checked = []

    def inspect_window(application):
        application.processEvents()
        window = next(
            widget
            for widget in application.topLevelWidgets()
            if widget.windowTitle() == "Session integration" and widget.isVisible()
        )
        try:
            window.resize(1100, 720)
            window._preview_timer.stop()
            retained = Figure(figsize=(5, 3))
            retained.add_subplot().plot([0, 1, 2], [0, 2, 1])
            window._install_preview_figure(retained)
            logger = logging.getLogger("linak.session.integration")
            monkeypatch.setattr(logger, "level", logging.INFO)
            worker = threading.Thread(target=lambda: logger.warning("Worker warning from renderer"))
            worker.start()
            worker.join()
            window._status_label.setText("Preview updating...")
            application.processEvents()
            assert (
                window._status_label.text() == window._session_diagnostics.session.latest().message
            )
            assert (
                "Worker warning from renderer" in window._session_diagnostics.viewer.toPlainText()
            )
            assert not window._logs_section.is_expanded()
            QTest.mouseClick(window._status_label, Qt.MouseButton.LeftButton)
            application.processEvents()
            assert window._nav_list.currentItem().text() == "Advanced"
            assert window._logs_section.is_expanded()
            assert window._session_diagnostics.viewer.isVisible()
            window._set_preview_loading(True)
            QTest.qWait(180)
            overlay = window._active_preview_pane.loading_overlay
            assert overlay.isVisible()
            assert window.grab().save(str(tmp_path / "loading.png"))
            # A stale finished render hands off to the queued replacement without
            # restarting the overlay's delay or stopping the spinner.
            generation = window._active_preview_generation
            window._pending_preview_request = (window._collect_settings(), False)
            window._handle_preview_worker_finished(generation, {})
            assert overlay.isVisible()
            assert overlay.animation.isActive()
            assert not overlay.delay.isActive()
            window._handle_detach_preview()
            application.processEvents()
            QTest.qWait(180)
            detached = window._active_preview_pane.loading_overlay
            assert detached.isVisible()
            assert not overlay.animation.isActive()
            window._handle_dock_preview()
            application.processEvents()
            assert window._active_preview_pane is window._embedded_preview_pane
            assert overlay.isVisible()
            error = ValueError("Invalid bin geometry\nDetailed explanation")
            window._set_preview_failure(error)
            window._set_preview_loading(False)
            assert not overlay.isVisible()
            assert "Detailed explanation" in window._session_diagnostics.viewer.toPlainText()
            window._status_label.setText("Preview updated.")
            assert window._status_label.text() == "Preview updated."
            assert "ValueError" in window._session_diagnostics.viewer.toPlainText()
            # The light appearance and narrow layout are exercised with real Qt painting.
            assert window.grab().save(str(tmp_path / "session.png"))
            checked.append(True)
        finally:
            release.set()
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        assert not overlay.animation.isActive()
        assert not overlay.delay.isActive()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect_window)
    launch_plot_settings_panel(
        title="Session integration",
        initial_settings={},
        on_preview=render,
        on_save=lambda name, settings: "Saved",
    )
    assert checked == [True]
    assert warnings.showwarning is original_warning
    assert tuple(logging.getLogger().handlers) == original_handlers


def test_launcher_failure_restores_logging_and_warning_hooks(app, monkeypatch):
    from linak.plot.plot_gui import launch_plot_settings_panel
    import warnings
    import linak.plot.session_widgets as widgets

    original_warning = warnings.showwarning
    original_handlers = tuple(logging.getLogger().handlers)

    def fail(*args, **kwargs):
        raise RuntimeError("Construction failed")

    monkeypatch.setattr(widgets, "SessionDiagnostics", fail)
    with pytest.raises(RuntimeError, match="Construction failed"):
        launch_plot_settings_panel(
            title="Failed construction",
            initial_settings={},
            on_preview=lambda settings: {},
            on_save=lambda name, settings: "Saved",
        )
    assert warnings.showwarning is original_warning
    assert tuple(logging.getLogger().handlers) == original_handlers
