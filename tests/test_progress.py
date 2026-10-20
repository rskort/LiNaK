from __future__ import annotations

import io
import logging
import os
import threading

from linak.progress import ProgressBar


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _terminal_lines(text: str) -> list[str]:
    """Interpret carriage returns so assertions describe the visible terminal."""
    lines = [""]
    column = 0
    for char in text:
        if char == "\r":
            column = 0
        elif char == "\n":
            lines.append("")
            column = 0
        else:
            line = lines[-1]
            lines[-1] = line[:column].ljust(column) + char + line[column + 1 :]
            column += 1
    return [line.rstrip() for line in lines]


def test_log_message_restarts_active_bar_below_message():
    from linak.cli import _ProgressAwareStreamHandler

    stream = _TTYBuffer()
    handler = _ProgressAwareStreamHandler(stream)
    record = logging.LogRecord(
        "linak.topology",
        logging.INFO,
        __file__,
        1,
        "Reusing exact molecular topology for 8010 frames.",
        (),
        None,
    )
    with ProgressBar(desc="Computing", total=8010, stream=stream) as progress:
        worker = threading.Thread(target=lambda: handler.handle(record))
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        lines = _terminal_lines(stream.getvalue())
        assert lines[0] == record.getMessage()
        assert "0/8010" in lines[1]
        assert len(lines) == 2  # No abandoned initial progress row.
        assert progress.count == 0
        handler.handle(record)
        assert _terminal_lines(stream.getvalue())[:2] == [record.getMessage()] * 2
        progress.update(8010)
    lines = _terminal_lines(stream.getvalue())
    assert "8010/8010" in lines[-2]
    assert lines[-1] == ""
    assert id(stream) not in ProgressBar._ACTIVE_BY_STREAM


def test_redirected_logging_has_no_progress_or_terminal_controls():
    from linak.cli import _ProgressAwareStreamHandler

    stream = io.StringIO()
    handler = _ProgressAwareStreamHandler(stream)
    with ProgressBar(desc="Computing", total=1, stream=stream) as progress:
        handler.handle(logging.makeLogRecord({"msg": "Topology ready."}))
        progress.update()
    assert stream.getvalue() == "Topology ready.\n"


def test_prepare_for_external_write_inserts_newline_for_active_progress() -> None:
    stream = _TTYBuffer()
    with ProgressBar(desc="Working", total=2, stream=stream, min_interval=0.0) as progress:
        progress.update()
        before = stream.getvalue()
        assert not before.endswith("\n")

        ProgressBar.prepare_for_external_write(stream)
        assert stream.getvalue().endswith("\n")


def test_prepare_for_external_write_is_noop_without_active_progress() -> None:
    stream = _TTYBuffer()
    ProgressBar.prepare_for_external_write(stream)
    assert stream.getvalue() == ""


def test_progress_bar_known_total_shows_left_and_rate(monkeypatch) -> None:
    stream = _TTYBuffer()
    now = {"value": 0.0}
    monkeypatch.setattr("linak.progress.time.perf_counter", lambda: now["value"])

    with ProgressBar(desc="Working", total=4, stream=stream, min_interval=0.0) as progress:
        now["value"] = 2.0
        progress.update(2)
        text = stream.getvalue()
        assert "[LiNaK] Working:" in text
        assert "2/4" in text
        assert "left" in text
        assert "it/s" in text


def test_progress_bar_unknown_total_omits_left(monkeypatch) -> None:
    stream = _TTYBuffer()
    now = {"value": 0.0}
    monkeypatch.setattr("linak.progress.time.perf_counter", lambda: now["value"])

    with ProgressBar(
        desc="Working",
        total=None,
        unit="frame",
        stream=stream,
        min_interval=0.0,
    ) as progress:
        now["value"] = 1.0
        progress.update(5)
        text = stream.getvalue()
        assert "[LiNaK] Working:" in text
        assert "left" not in text
        assert "frame/s" in text


def test_progress_bar_truncates_to_fit_narrow_terminal(monkeypatch) -> None:
    stream = _TTYBuffer()
    now = {"value": 0.0}
    monkeypatch.setattr("linak.progress.time.perf_counter", lambda: now["value"])
    monkeypatch.setattr(
        "linak.progress.shutil.get_terminal_size",
        lambda fallback=(120, 20): os.terminal_size((80, 20)),
    )

    with ProgressBar(
        desc="Selecting element data for density with a very long description",
        total=100,
        stream=stream,
        min_interval=0.0,
    ) as progress:
        now["value"] = 1.0
        progress.update(10)
        lines = [line for line in stream.getvalue().split("\r") if line]
        assert lines
        assert max(len(line.rstrip("\n")) for line in lines) <= 80
