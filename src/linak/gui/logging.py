"""Logging bridge shared by CLI backends and GUI tasks."""

from __future__ import annotations

import logging
from collections.abc import Callable


class TaskLogHandler(logging.Handler):
    """Forward structured log records to one task."""

    def __init__(self, emit_log: Callable[[str, str], None]) -> None:
        super().__init__(level=logging.DEBUG)
        self._emit_log = emit_log
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_log(record.levelname, self.format(record))
        except Exception:
            self.handleError(record)
