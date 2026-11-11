"""Cross-platform runtime memory and worker budgeting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ctypes
import logging
import os
from pathlib import Path
import subprocess
import math

MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_FALLBACK_BUDGET_BYTES = 4 * GIB
MINIMUM_FREE_BYTES = 2 * GIB
AUTOMATIC_AVAILABLE_FRACTION = 0.55


def _positive_int(value: object) -> int | None:
    try:
        number = int(str(value).strip())
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _cgroup_directories() -> list[Path]:
    """Find this process's cgroups, including delegated/container hierarchies."""
    result = [Path("/sys/fs/cgroup")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _hierarchy, controllers, relative = line.split(":", 2)
            root = Path("/sys/fs/cgroup")
            if controllers:
                root = root / controllers
            candidate = root / relative.lstrip("/")
            if ".." not in candidate.parts:
                # Do not use parents slicing (Python 3.10+ only).
                while True:
                    result.append(candidate)
                    if candidate == root:
                        break
                    candidate = candidate.parent
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(result))


def detect_cpu_count(*, environ: Mapping[str, str] | None = None) -> int:
    """Logical CPUs actually available to this task, not the entire Slurm node."""
    env = os.environ if environ is None else environ
    process_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    limits = [max(1, process_count)]
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            limits.append(max(1, len(affinity(0))))
        except OSError:
            pass
    allocated = _positive_int(env.get("SLURM_CPUS_PER_TASK"))
    if allocated is not None:
        limits.append(allocated)
    for directory in _cgroup_directories():
        try:
            quota, period = (directory / "cpu.max").read_text().split()
            if quota != "max" and int(period) > 0:
                limits.append(max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError):
            try:
                old_quota = int((directory / "cpu.cfs_quota_us").read_text())
                old_period = int((directory / "cpu.cfs_period_us").read_text())
                if old_quota > 0 and old_period > 0:
                    limits.append(max(1, old_quota // old_period))
            except (OSError, ValueError):
                pass
    return min(limits)


def _job_available_memory(environ: Mapping[str, str]) -> int | None:
    limits = []
    for directory in _cgroup_directories():
        for maximum, current in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            try:
                limit = int((directory / maximum).read_text())
                usage = int((directory / current).read_text())
                if 0 < limit < (1 << 60):
                    limits.append(max(1, limit - usage))
            except (OSError, ValueError):
                pass
    per_cpu = _positive_int(environ.get("SLURM_MEM_PER_CPU"))
    cpus = _positive_int(environ.get("SLURM_CPUS_PER_TASK"))
    per_node = _positive_int(environ.get("SLURM_MEM_PER_NODE"))
    resident = 0
    try:
        sysconf = getattr(os, "sysconf", None)
        if sysconf is not None:
            resident = int(Path("/proc/self/statm").read_text().split()[1]) * sysconf(
                "SC_PAGE_SIZE"
            )
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    if per_cpu and cpus:
        limits.append(max(1, per_cpu * cpus * MIB - resident))
    # A node allocation is exclusive to this process only in a single-task job.
    if per_node and _positive_int(environ.get("SLURM_NTASKS")) == 1:
        limits.append(max(1, per_node * MIB - resident))
    return min(limits) if limits else None


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
    job_available = _job_available_memory(env)
    if memory_budget_mib is None:
        from .parallel import current_execution

        execution = current_execution()
        if execution is not None:
            memory_budget_mib = execution.memory_budget_mib
    if memory_budget_mib is not None:
        budget = _parse_budget_mib(memory_budget_mib, source="memory_budget_mib")
        if job_available is not None:
            budget = min(budget, job_available)
        return ResourcePolicy(available_memory_bytes, budget, "cli")
    env_value = str(env.get("LINAK_MEMORY_BUDGET_MIB", "")).strip()
    if env_value:
        budget = _parse_budget_mib(env_value, source="LINAK_MEMORY_BUDGET_MIB")
        if job_available is not None:
            budget = min(budget, job_available)
        return ResourcePolicy(available_memory_bytes, budget, "environment")

    available = (
        detect_available_memory_bytes()
        if available_memory_bytes is None
        else int(available_memory_bytes)
    )
    if job_available is not None:
        available = min(available, job_available) if available else job_available
    if available is None or available <= 0:
        return ResourcePolicy(None, DEFAULT_FALLBACK_BUDGET_BYTES, "fallback")
    fractional = int(AUTOMATIC_AVAILABLE_FRACTION * available)
    reserve_limited = max(1, available - min(MINIMUM_FREE_BYTES, available // 4))
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
    cpus = max(1, min(int(cpu_count or detect_cpu_count()), detect_cpu_count()))
    per_worker = max(1, int(estimated_bytes_per_worker))
    if requested_threads is not None:
        workers = min(
            tasks,
            cpus,
            max(1, int(requested_threads)),
            max(1, policy.memory_budget_bytes // per_worker),
        )
        if workers < min(tasks, int(requested_threads)) and logger is not None:
            logger.warning(
                "%s requests %d workers; using %d within the CPU/memory budget (%.1f MiB).",
                context,
                requested_threads,
                workers,
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
    "detect_cpu_count",
    "resolve_resource_policy",
    "resolve_worker_count",
)
