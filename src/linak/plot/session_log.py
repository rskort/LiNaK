"""Window-session diagnostics, independent of Qt and terminal formatting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import traceback
import warnings
from collections.abc import Callable
from typing import Any


@dataclass(frozen=True)
class SessionMessage:
    timestamp: datetime
    level: int
    source: str
    message: str
    details: str = ""

    def formatted(self) -> str:
        text = f"{self.timestamp:%H:%M:%S} {logging.getLevelName(self.level)} {self.source}: {self.message}"
        return f"{text}\n{self.details}" if self.details else text


# One tee for Python warnings, shared by all open windows. The original warning
# function still owns terminal output and warning filtering remains unchanged.
_warning_lock = threading.RLock()
_warning_sessions: set[SessionLog] = set()
_previous_showwarning: Callable[..., Any] | None = None


def _showwarning(
    message: Any,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    with _warning_lock:
        sessions = tuple(_warning_sessions)
        previous = _previous_showwarning
    # captureWarnings already routes through logging; do not record it twice.
    through_logging = previous is getattr(logging, "_showwarning", None) and file is None
    if not through_logging:
        for session in sessions:
            session.publish(
                str(message),
                level=logging.WARNING,
                source=category.__name__,
                details=f"{filename}:{lineno}",
            )
    if previous is not None:
        previous(message, category, filename, lineno, file=file, line=line)


class _SessionHandler(logging.Handler):
    def __init__(self, session: SessionLog) -> None:
        super().__init__()
        self.session = session

    def emit(self, record: logging.LogRecord) -> None:
        details = []
        if record.exc_info:
            details.append(logging.Formatter().formatException(record.exc_info))
        if record.stack_info:
            details.append(record.stack_info)
        self.session.publish(
            record.getMessage(),
            level=record.levelno,
            source=record.name,
            details="\n".join(details),
            timestamp=datetime.fromtimestamp(record.created),
        )


class SessionLog:
    """Collect messages once; subscribers are notified without touching widgets."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: list[SessionMessage] = []
        self._listeners: list[Callable[[], None]] = []
        self._closed = False
        self._revision = 0
        self._handler: _SessionHandler | None = None

    def subscribe(self, listener: Callable[[], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def snapshot(self) -> list[SessionMessage]:
        with self._lock:
            return list(self._entries)

    def latest(self) -> SessionMessage | None:
        with self._lock:
            return self._entries[-1] if self._entries else None

    def read_since(self, revision: int, count: int) -> tuple[int, int, list[SessionMessage]]:
        """Return only new entries, resetting the cursor when history is cleared."""
        with self._lock:
            if revision != self._revision:
                count = 0
            return self._revision, len(self._entries), self._entries[count:]

    def publish(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        source: str = "GUI",
        details: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        with self._lock:
            if self._closed or not str(message).strip():
                return
            self._entries.append(
                SessionMessage(timestamp or datetime.now(), level, source, str(message), details)
            )
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener()

    def exception(self, message: str, error: BaseException) -> None:
        self.publish(
            message,
            level=logging.ERROR,
            details="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ).rstrip(),
        )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._revision += 1
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener()

    def start(self) -> None:
        global _previous_showwarning
        with _warning_lock:
            if self._handler is not None or self._closed:
                return
            self._handler = _SessionHandler(self)
            logging.getLogger().addHandler(self._handler)
            if not _warning_sessions:
                _previous_showwarning = warnings.showwarning
                warnings.showwarning = _showwarning
            _warning_sessions.add(self)

    def close(self) -> None:
        global _previous_showwarning
        with _warning_lock:
            with self._lock:
                self._closed = True
                self._listeners.clear()
            if self._handler is not None:
                logging.getLogger().removeHandler(self._handler)
                self._handler.close()
                self._handler = None
            _warning_sessions.discard(self)
            if not _warning_sessions and _previous_showwarning is not None:
                if warnings.showwarning is _showwarning:
                    warnings.showwarning = _previous_showwarning
                _previous_showwarning = None
