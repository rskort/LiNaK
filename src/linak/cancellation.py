"""Cooperative cancellation scoped to a single background operation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


class ComputationCancelled(BaseException):
    """Stop obsolete work without treating it as an analysis failure."""


@dataclass
class _Operation:
    cancelled: Callable[[], bool]
    cleanups: list[Callable[[], None]] = field(default_factory=list)


_OPERATION: ContextVar[_Operation | None] = ContextVar("linak_operation", default=None)


def check_cancelled() -> None:
    operation = _OPERATION.get()
    if operation is not None and operation.cancelled():
        raise ComputationCancelled()


def register_cancel_cleanup(cleanup: Callable[[], None]) -> None:
    """Release only resources owned by this operation if it is cancelled."""
    operation = _OPERATION.get()
    if operation is not None:
        operation.cleanups.append(cleanup)


@contextmanager
def cancellation_scope(cancelled: Callable[[], bool]) -> Iterator[None]:
    operation = _Operation(cancelled)
    token = _OPERATION.set(operation)
    try:
        check_cancelled()
        yield
    except ComputationCancelled:
        for cleanup in reversed(operation.cleanups):
            try:
                cleanup()
            except Exception:
                pass
        raise
    finally:
        _OPERATION.reset(token)
