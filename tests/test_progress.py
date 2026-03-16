import io

from linak.progress import ProgressBar


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


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
