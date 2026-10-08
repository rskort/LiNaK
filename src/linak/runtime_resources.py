"""Cross-platform runtime memory and worker budgeting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ctypes
import logging
import os
from pathlib import Path
import subprocess

MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_FALLBACK_BUDGET_BYTES = 4 * GIB
MINIMUM_FREE_BYTES = 2 * GIB
AUTOMATIC_AVAILABLE_FRACTION = 0.55


@dataclass(frozen=True)
class ResourcePolicy:
    """Resolved working-memory policy for one LiNaK command."""

    available_memory_bytes: int | None
    memory_budget_bytes: int
    source: str

    @property
    def memory_budget_mib(self) -> float:
        return self.memory_budget_bytes / MIB


def _windows_available_memory() -> int | None:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    return int(status.ullAvailPhys) if success else None


def _linux_available_memory() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
    return None


def _macos_available_memory() -> int | None:
    try:
        page_size = int(
            subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True, timeout=2).strip()
        )
        output = subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    pages = 0
    for line in output.splitlines():
        if line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
            try:
                pages += int(line.split(":", 1)[1].strip().rstrip("."))
            except ValueError:
                continue
    return pages * page_size if pages > 0 else None


def detect_available_memory_bytes() -> int | None:
    """Return currently available physical memory using only the standard library."""

    if os.name == "nt":
        return _windows_available_memory()
    if Path("/proc/meminfo").exists():
        return _linux_available_memory()
    if os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
        return _macos_available_memory()
    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        return None
    try:
        pages = int(sysconf("SC_AVPHYS_PAGES"))
        page_size = int(sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    result = pages * page_size
    return result if result > 0 else None


def _parse_budget_mib(value: object, *, source: str) -> int:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive number of MiB.") from exc
    if not parsed > 0:
        raise ValueError(f"{source} must be a positive number of MiB.")
    return max(1, int(parsed * MIB))


def resolve_resource_policy(
    memory_budget_mib: float | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    available_memory_bytes: int | None = None,
) -> ResourcePolicy:
    """Resolve explicit, environment, or automatic working-memory budget."""

    env = os.environ if environ is None else environ
    if memory_budget_mib is not None:
        budget = _parse_budget_mib(memory_budget_mib, source="memory_budget_mib")
        return ResourcePolicy(available_memory_bytes, budget, "cli")
    env_value = str(env.get("LINAK_MEMORY_BUDGET_MIB", "")).strip()
    if env_value:
        budget = _parse_budget_mib(env_value, source="LINAK_MEMORY_BUDGET_MIB")
        return ResourcePolicy(available_memory_bytes, budget, "environment")

    available = (
        detect_available_memory_bytes()
        if available_memory_bytes is None
        else int(available_memory_bytes)
    )
    if available is None or available <= 0:
        return ResourcePolicy(None, DEFAULT_FALLBACK_BUDGET_BYTES, "fallback")
    fractional = int(AUTOMATIC_AVAILABLE_FRACTION * available)
    reserve_limited = max(1, available - MINIMUM_FREE_BYTES)
    budget = max(1, min(fractional, reserve_limited))
    return ResourcePolicy(available, budget, "automatic")


def resolve_worker_count(
    *,
    task_count: int,
    estimated_bytes_per_worker: int,
    requested_threads: int | None,
    policy: ResourcePolicy,
    cpu_count: int | None = None,
    logger: logging.Logger | None = None,
    context: str = "work",
) -> int:
    """Resolve workers from tasks, CPUs, and memory, honoring explicit threads."""

    tasks = max(1, int(task_count))
    cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    per_worker = max(1, int(estimated_bytes_per_worker))
    if requested_threads is not None:
        workers = min(tasks, max(1, int(requested_threads)))
        estimate = workers * per_worker
        if estimate > policy.memory_budget_bytes and logger is not None:
            logger.warning(
                "%s explicitly requests %d workers (estimated %.1f MiB), exceeding the "
                "%.1f MiB memory budget; continuing because --threads is authoritative.",
                context,
                workers,
                estimate / MIB,
                policy.memory_budget_mib,
            )
        return workers
    memory_workers = max(1, policy.memory_budget_bytes // per_worker)
    return max(1, min(tasks, cpus, memory_workers))


__all__ = (
    "AUTOMATIC_AVAILABLE_FRACTION",
    "DEFAULT_FALLBACK_BUDGET_BYTES",
    "GIB",
    "MIB",
    "MINIMUM_FREE_BYTES",
    "ResourcePolicy",
    "detect_available_memory_bytes",
    "resolve_resource_policy",
    "resolve_worker_count",
)
