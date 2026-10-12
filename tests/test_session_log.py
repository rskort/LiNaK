from __future__ import annotations

import io
import logging
import threading
import warnings

from linak.plot.session_log import SessionLog


def test_logging_levels_worker_messages_exceptions_and_terminal(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("linak.session_test")
    monkeypatch.setattr(logger, "level", logging.WARNING)
    logger.addHandler(handler)
    session = SessionLog()
    root_handlers = tuple(logging.getLogger().handlers)
    session.start()
    session.start()
    try:
        logger.info("hidden")

        def worker():
            logger.warning("worker warning")
            try:
                raise ValueError("bad value")
            except ValueError:
                logger.exception("worker failed")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        session.publish("Preview updated.")
        entries = session.snapshot()
        assert [entry.message for entry in entries] == [
            "worker warning",
            "worker failed",
            "Preview updated.",
        ]
        assert [entry.level for entry in entries] == [logging.WARNING, logging.ERROR, logging.INFO]
        assert "ValueError: bad value" in entries[1].details
        assert stream.getvalue().count("worker warning") == 1
        assert stream.getvalue().count("worker failed") == 1
    finally:
        session.close()
        logger.removeHandler(handler)
    assert tuple(logging.getLogger().handlers) == root_handlers
    logger.warning("after close")
    assert len(session.snapshot()) == 3


def test_warning_tee_multiple_windows_and_cleanup(monkeypatch):
    terminal = []

    def original(*args, **kwargs):
        terminal.append(args)

    monkeypatch.setattr(warnings, "showwarning", original)
    first, second = SessionLog(), SessionLog()
    first.start()
    second.start()
    try:
        warnings.showwarning("a warning", UserWarning, "input.py", 12)
        assert len(terminal) == 1
        assert len(first.snapshot()) == len(second.snapshot()) == 1
        assert first.snapshot()[0].details == "input.py:12"
        first.close()
        warnings.showwarning("another", RuntimeWarning, "render.py", 20)
        assert len(first.snapshot()) == 1
        assert len(second.snapshot()) == 2
    finally:
        first.close()
        second.close()
    assert warnings.showwarning is original


def test_warning_already_captured_by_logging_has_one_entry():
    session = SessionLog()
    logging.captureWarnings(True)
    previous = warnings.showwarning
    session.start()
    try:
        warnings.showwarning("one", UserWarning, "input.py", 1)
        assert len(session.snapshot()) == 1
    finally:
        session.close()
        assert warnings.showwarning is previous
        logging.captureWarnings(False)


def test_clear_and_close_notifications():
    session = SessionLog()
    notifications = []
    session.subscribe(lambda: notifications.append(len(session.snapshot())))
    session.publish("first")
    session.publish("second")
    session.clear()
    session.publish("third")
    session.close()
    session.publish("ignored")
    assert notifications == [1, 2, 0, 1]
    assert [entry.message for entry in session.snapshot()] == ["third"]
