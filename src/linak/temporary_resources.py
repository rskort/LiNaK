"""Conservative lifecycle checks for LiNaK-owned temporary directories.

An inconclusive process query is treated as alive: retaining temporary data is
preferable to deleting the working files of an active analysis.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import shutil
import stat
import time


def current_owner() -> str:
    return str(os.getenv("USERNAME") or os.getenv("USER") or "")


def _windows_process_alive(pid: int) -> bool:
    # PROCESS_QUERY_LIMITED_INFORMATION needs no termination or write rights.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER indicates a PID that does not exist. Access
        # denied and every other error remain unknown, hence protected.
        return ctypes.get_last_error() != 87
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def process_alive(pid: int) -> bool:
    """Return False only for a confirmed absent process; never send a Windows signal."""
    if pid <= 0:
        return True
    try:
        if os.name == "nt":
            return _windows_process_alive(pid)
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, OverflowError, ValueError):
        return True
    return True


def _is_link(path: Path) -> bool:
    """Include Windows junctions on Python 3.9, which lacks Path.is_junction."""
    attributes = path.lstat()
    return stat.S_ISLNK(attributes.st_mode) or bool(
        getattr(attributes, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def owned_directory_payload(
    directory: Path, *, parent: Path, prefix: str, marker_name: str
) -> dict | None:
    """Validate a direct child and its marker without following links."""
    try:
        parent = parent.resolve(strict=True)
        if not directory.name.startswith(prefix) or _is_link(directory):
            return None
        resolved = directory.resolve(strict=True)
        if resolved.parent != parent or not resolved.is_dir():
            return None
        marker = resolved / marker_name
        if _is_link(marker) or not marker.is_file():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        owner = current_owner()
        if not owner or payload.get("owner") != owner:
            return None
        if hasattr(os, "getuid") and resolved.stat().st_uid != os.getuid():
            return None
        return payload
    except (OSError, ValueError, RuntimeError):
        return None


def remove_owned_directory(directory: Path, *, parent: Path, prefix: str, marker_name: str) -> bool:
    """Remove a validated run directory; callers must establish process ownership."""
    if (
        owned_directory_payload(directory, parent=parent, prefix=prefix, marker_name=marker_name)
        is None
    ):
        return False
    try:
        shutil.rmtree(directory)
    except OSError:
        return False
    return True


def cleanup_stale_directories(
    parent: Path, *, prefix: str, marker_name: str, timestamp_key: str, stale_seconds: float
) -> int:
    """Remove only marked current-user directories belonging to confirmed dead processes."""
    removed = 0
    try:
        candidates = list(parent.glob(f"{prefix}*"))
    except OSError:
        return 0
    for directory in candidates:
        payload = owned_directory_payload(
            directory, parent=parent, prefix=prefix, marker_name=marker_name
        )
        if payload is None:
            continue
        pid = payload.get("pid")
        started = payload.get(timestamp_key)
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            continue
        if not isinstance(started, (int, float)) or isinstance(started, bool):
            continue
        if not math.isfinite(started) or started <= 0 or time.time() - started < stale_seconds:
            continue
        if process_alive(pid):
            continue
        removed += remove_owned_directory(
            directory, parent=parent, prefix=prefix, marker_name=marker_name
        )
    return removed
