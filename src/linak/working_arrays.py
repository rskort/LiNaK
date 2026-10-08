"""File-backed working arrays for bounded-memory CLI execution."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from uuid import uuid4

import numpy as np

from .temporary_resources import cleanup_stale_directories, current_owner, remove_owned_directory

_PREFIX = "linak-arrays-"
_MARKER = ".linak-working-arrays.json"
_STALE_SECONDS = 24 * 60 * 60


class _WorkingArrayManager:
    def __init__(self, parent: Path, *, threshold_bytes: int) -> None:
        self.threshold_bytes = max(1, int(threshold_bytes))
        self.directory = Path(tempfile.mkdtemp(prefix=_PREFIX, dir=parent))
        (self.directory / _MARKER).write_text(
            json.dumps({"pid": os.getpid(), "created": time.time(), "owner": current_owner()}),
            encoding="utf-8",
        )
        self._closed = False
        self._memmaps: list[np.memmap] = []

    def allocate(
        self,
        shape: tuple[int, ...],
        *,
        dtype: Any,
        fill: float | int | None,
    ) -> np.ndarray:
        resolved_dtype = np.dtype(dtype)
        size = int(np.prod(shape, dtype=np.int64)) * resolved_dtype.itemsize
        if size < self.threshold_bytes:
            values = np.empty(shape, dtype=resolved_dtype)
        else:
            path = self.directory / f"array-{uuid4().hex}.bin"
            values = np.memmap(path, mode="w+", shape=shape, dtype=resolved_dtype)
            self._memmaps.append(values)
        if fill is not None:
            values.fill(fill)
        return values

    def close(self) -> None:
        if self._closed:
            return
        for values in self._memmaps:
            try:
                values.flush()
                mmap = getattr(values, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            except (OSError, ValueError):
                pass
        self._memmaps.clear()
        remove_owned_directory(
            self.directory, parent=self.directory.parent, prefix=_PREFIX, marker_name=_MARKER
        )
        self._closed = True


_ACTIVE_MANAGER: _WorkingArrayManager | None = None


def cleanup_stale_working_arrays(parent: Path) -> int:
    """Remove only validated stale directories whose owner process is dead."""
    return cleanup_stale_directories(
        parent,
        prefix=_PREFIX,
        marker_name=_MARKER,
        timestamp_key="created",
        stale_seconds=_STALE_SECONDS,
    )


def enable_bounded_working_arrays(
    *,
    temporary_directory: str | Path | None,
    threshold_mib: float,
) -> Path:
    """Enable process-scoped file-backed arrays and return their directory."""

    global _ACTIVE_MANAGER
    if _ACTIVE_MANAGER is not None:
        return _ACTIVE_MANAGER.directory
    configured = temporary_directory or os.getenv("LINAK_TEMP_DIR")
    parent = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(tempfile.gettempdir()).resolve()
    )
    parent.mkdir(parents=True, exist_ok=True)
    cleanup_stale_working_arrays(parent)
    _ACTIVE_MANAGER = _WorkingArrayManager(
        parent,
        threshold_bytes=max(1, int(float(threshold_mib) * 1024 * 1024)),
    )
    atexit.register(disable_bounded_working_arrays)
    return _ACTIVE_MANAGER.directory


def disable_bounded_working_arrays() -> None:
    global _ACTIVE_MANAGER
    manager, _ACTIVE_MANAGER = _ACTIVE_MANAGER, None
    if manager is not None:
        manager.close()


def working_array(
    shape: tuple[int, ...],
    *,
    dtype: Any = float,
    fill: float | int | None = None,
) -> np.ndarray:
    """Allocate an ndarray or active file-backed equivalent."""

    if _ACTIVE_MANAGER is None:
        values = np.empty(shape, dtype=dtype)
        if fill is not None:
            values.fill(fill)
        return values
    return _ACTIVE_MANAGER.allocate(shape, dtype=dtype, fill=fill)


__all__ = (
    "cleanup_stale_working_arrays",
    "disable_bounded_working_arrays",
    "enable_bounded_working_arrays",
    "working_array",
)
