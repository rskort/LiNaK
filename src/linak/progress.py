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

    @staticmethod
    def _stream_is_tty(stream: TextIO) -> bool:
        """Return whether a stream is an open TTY without raising on closed captures."""
        if getattr(stream, "closed", False):
            return False
        try:
            return bool(getattr(stream, "isatty", lambda: False)())
        except ValueError:
            return False

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
        self.enabled = enabled and self._stream_is_tty(self.stream)
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
        if not cls._stream_is_tty(stream):
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
        rate_label = (
            f"{rate:6.2f} {self.unit}/s" if rate is not None else f"{'--':>6} {self.unit}/s"
        )
        desc = self._truncate_desc(self.desc)

        if self.total is not None and self.total > 0:
            ratio = min(1.0, self.count / self.total)
            percent = ratio * 100.0
            remaining = None
            if ratio > 0.0:
                remaining = max(0.0, elapsed * (1.0 - ratio) / ratio)
            left_label = self._format_seconds(remaining) if remaining is not None else "--"
            bar_width = self._resolve_bar_width()
            line = self._build_known_total_line(
                desc=desc,
                bar_width=bar_width,
                ratio=ratio,
                percent=percent,
                elapsed_label=elapsed_label,
                left_label=left_label,
                rate_label=rate_label,
            )
            if len(line) > self._terminal_columns:
                desc = self._truncate_text(
                    desc, max(8, len(desc) - (len(line) - self._terminal_columns))
                )
                line = self._build_known_total_line(
                    desc=desc,
                    bar_width=bar_width,
                    ratio=ratio,
                    percent=percent,
                    elapsed_label=elapsed_label,
                    left_label=left_label,
                    rate_label=rate_label,
                )
            while len(line) > self._terminal_columns and bar_width > 4:
                bar_width -= 1
                line = self._build_known_total_line(
                    desc=desc,
                    bar_width=bar_width,
                    ratio=ratio,
                    percent=percent,
                    elapsed_label=elapsed_label,
                    left_label=left_label,
                    rate_label=rate_label,
                )
        else:
            spinner = self._SPINNER[self._spinner_index % len(self._SPINNER)]
            self._spinner_index += 1
            line = self._build_unknown_total_line(
                desc=desc,
                spinner=spinner,
                elapsed_label=elapsed_label,
                rate_label=rate_label,
            )
            if len(line) > self._terminal_columns:
                desc = self._truncate_text(
                    desc, max(8, len(desc) - (len(line) - self._terminal_columns))
                )
                line = self._build_unknown_total_line(
                    desc=desc,
                    spinner=spinner,
                    elapsed_label=elapsed_label,
                    rate_label=rate_label,
                )

        line = self._clamp_line_to_terminal(line)

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
        # Keep the bar visible while leaving room for metadata columns and description text.
        # The metadata budget intentionally assumes a moderately long description so the line
        # stays on one terminal row instead of wrapping and appearing as multi-line spam.
        max_for_terminal = max(10, self._terminal_columns - 110)
        return min(self.width, max_for_terminal)

    def _truncate_desc(self, desc: str) -> str:
        """Shorten long descriptions so the rendered line stays within terminal width."""
        if self.total is not None and self.total > 0:
            max_desc = max(12, self._terminal_columns - 85 - self._resolve_bar_width())
        else:
            max_desc = max(12, self._terminal_columns - 48)
        if len(desc) <= max_desc:
            return desc
        return self._truncate_text(desc, max_desc)

    def _truncate_text(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        if max_length <= 3:
            return text[:max_length]
        return text[: max_length - 3] + "..."

    def _build_known_total_line(
        self,
        *,
        desc: str,
        bar_width: int,
        ratio: float,
        percent: float,
        elapsed_label: str,
        left_label: str,
        rate_label: str,
    ) -> str:
        filled = int(bar_width * ratio)
        bar = "=" * filled + "." * (bar_width - filled)
        return (
            f"\r{self._BRAND} {desc}: [{bar}] {self.count}/{self.total} "
            f"{percent:6.2f}% | elapsed {elapsed_label} | left {left_label} | {rate_label}"
        )

    def _build_unknown_total_line(
        self,
        *,
        desc: str,
        spinner: str,
        elapsed_label: str,
        rate_label: str,
    ) -> str:
        return (
            f"\r{self._BRAND} {desc}: {spinner} {self.count} {self.unit} "
            f"| elapsed {elapsed_label} | {rate_label}"
        )

    def _clamp_line_to_terminal(self, line: str) -> str:
        if self._terminal_columns <= 0 or len(line) <= self._terminal_columns:
            return line
        if line.startswith("\r") and self._terminal_columns > 1:
            return "\r" + line[1 : self._terminal_columns]
        return line[: self._terminal_columns]

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
