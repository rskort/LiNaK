"""Lightweight terminal progress utilities for LiNaK."""

from __future__ import annotations

import math
import shutil
import sys
import time
from types import TracebackType
from typing import TextIO


class ProgressBar:
    """A minimal tqdm-like progress bar without external dependencies."""

    _SPINNER = ("|", "/", "-", "\\")
    _BRAND = "[LiNaK]"
    _ACTIVE_BY_STREAM: dict[int, int] = {}
    _LINE_OPEN_BY_STREAM: dict[int, bool] = {}

    def __init__(
        self,
        *,
        desc: str,
        total: int | None = None,
        unit: str = "it",
        enabled: bool = True,
        stream: TextIO | None = None,
        width: int = 28,
        min_interval: float = 0.1,
    ) -> None:
        self.desc = desc
        self.total = total
        self.unit = unit
        self.stream = stream or sys.stderr
        self.width = width
        self.min_interval = min_interval
        self.count = 0
        self._spinner_index = 0
        self._start = time.perf_counter()
        self._last_render = 0.0
        self._closed = False
        self._last_line_length = 0
        self._last_terminal_probe = 0.0
        self._terminal_columns = shutil.get_terminal_size(fallback=(120, 20)).columns
        self.enabled = enabled and bool(getattr(self.stream, "isatty", lambda: False)())
        self._stream_id = id(self.stream)

    def __enter__(self) -> ProgressBar:
        if self.enabled:
            self._ACTIVE_BY_STREAM[self._stream_id] = (
                self._ACTIVE_BY_STREAM.get(self._stream_id, 0) + 1
            )
        self._render(force=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        if self._closed:
            return
        self.count += n
        self._render()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._render(force=True, final=True)
        if self.enabled:
            active = self._ACTIVE_BY_STREAM.get(self._stream_id, 0) - 1
            if active > 0:
                self._ACTIVE_BY_STREAM[self._stream_id] = active
            else:
                self._ACTIVE_BY_STREAM.pop(self._stream_id, None)
                self._LINE_OPEN_BY_STREAM.pop(self._stream_id, None)

    @classmethod
    def prepare_for_external_write(cls, stream: TextIO) -> None:
        """Move to a fresh line before external writes when a progress line is active."""
        if not bool(getattr(stream, "isatty", lambda: False)()):
            return
        stream_id = id(stream)
        if cls._ACTIVE_BY_STREAM.get(stream_id, 0) <= 0:
            return
        if not cls._LINE_OPEN_BY_STREAM.get(stream_id, False):
            return
        stream.write("\n")
        stream.flush()
        cls._LINE_OPEN_BY_STREAM[stream_id] = False

    def _render(self, *, force: bool = False, final: bool = False) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        if not force and (now - self._last_render) < self.min_interval:
            return
        self._last_render = now
        elapsed = now - self._start
        self._refresh_terminal_width(now)
        rate = self._update_rate(now, elapsed)
        elapsed_label = self._format_seconds(elapsed)
        rate_label = f"{rate:6.2f} {self.unit}/s" if rate is not None else f"{'--':>6} {self.unit}/s"

        if self.total is not None and self.total > 0:
            ratio = min(1.0, self.count / self.total)
            bar_width = self._resolve_bar_width()
            filled = int(bar_width * ratio)
            bar = "=" * filled + "." * (bar_width - filled)
            percent = ratio * 100.0
            remaining = None
            if ratio > 0.0:
                remaining = max(0.0, elapsed * (1.0 - ratio) / ratio)
            left_label = self._format_seconds(remaining) if remaining is not None else "--"
            line = (
                f"\r{self._BRAND} {self.desc}: [{bar}] {self.count}/{self.total} "
                f"{percent:6.2f}% | elapsed {elapsed_label} | left {left_label} | {rate_label}"
            )
        else:
            spinner = self._SPINNER[self._spinner_index % len(self._SPINNER)]
            self._spinner_index += 1
            line = (
                f"\r{self._BRAND} {self.desc}: {spinner} {self.count} {self.unit} "
                f"| elapsed {elapsed_label} | {rate_label}"
            )

        if len(line) < self._last_line_length:
            line = line + (" " * (self._last_line_length - len(line)))
        self._last_line_length = len(line)

        self.stream.write(line)
        if final:
            self.stream.write("\n")
            self._LINE_OPEN_BY_STREAM[self._stream_id] = False
            self._last_line_length = 0
        else:
            self._LINE_OPEN_BY_STREAM[self._stream_id] = True
        self.stream.flush()

    def _refresh_terminal_width(self, now: float) -> None:
        """Probe terminal width at low frequency to avoid extra overhead."""
        if (now - self._last_terminal_probe) < 1.0:
            return
        self._last_terminal_probe = now
        columns = shutil.get_terminal_size(fallback=(120, 20)).columns
        if columns > 0:
            self._terminal_columns = columns

    def _resolve_bar_width(self) -> int:
        """Compute a bar width that avoids wrapping on narrower terminals."""
        # Keep the bar visible while leaving room for metadata columns.
        max_for_terminal = max(10, self._terminal_columns - 90)
        return min(self.width, max_for_terminal)

    def _update_rate(self, now: float, elapsed: float) -> float | None:
        _ = now
        if self.count <= 0:
            return None
        if elapsed > 0.0:
            average = self.count / elapsed
            if average > 0.0 and math.isfinite(average):
                return average
        return None

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "--"
        if seconds < 60.0:
            return f"{seconds:5.1f}s"
        rounded = int(round(seconds))
        if rounded < 3600:
            minutes, secs = divmod(rounded, 60)
            return f"{minutes:02d}m{secs:02d}s"
        hours, rem = divmod(rounded, 3600)
        minutes = rem // 60
        return f"{hours:02d}h{minutes:02d}m"
