"""Bounded, allocation-aware computation stages; no plotting or I/O dependencies.

Automatic policies are deliberately conservative until representative benchmarks
justify changing them. Explicit --threads exercises the available parallel paths.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError, Future
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import wraps
from itertools import islice
import logging
import multiprocessing
import os
from time import perf_counter
from threading import Lock
from typing import Any, TypeVar

from .cancellation import check_cancelled
from .runtime_resources import MIB, detect_cpu_count, resolve_resource_policy, resolve_worker_count

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")

# (automatic worker ceiling, minimum logical tasks). Updated only from verified
# benchmark reports, per stage rather than globally or by machine CPU count.
AUTO_POLICIES = {
    "surface.frames": (1, 8),
    "density.selection": (1, 8),
    "density.frames": (1, 8),
    "orientation.frames": (1, 8),
    "orientation.bins": (1, 8),
    "coordination.distances": (1, 4),
    "coordination.entities": (1, 8),
    "rdf.distances": (4, 2),
    "rdf.entities": (1, 8),
    "potential.files": (1, 2),
    "position.nearest": (1, 4096),
    "temperature.velocities": (1, 4),
}


@dataclass
class Execution:
    threads: int | None = 1
    memory_budget_mib: float | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    nearest_decisions: dict[int, int] = field(default_factory=dict)
    timing_lock: Any = field(default_factory=Lock, repr=False)


_EXECUTION: ContextVar[Execution | None] = ContextVar("linak_execution", default=None)
_IN_WORKER: ContextVar[bool] = ContextVar("linak_in_worker", default=False)
_TIMING_ACTIVE: ContextVar[frozenset[str]] = ContextVar("linak_timing_active", default=frozenset())


def current_execution() -> Execution | None:
    return _EXECUTION.get()


def in_worker() -> bool:
    return _IN_WORKER.get()


def timed_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if name in _TIMING_ACTIVE.get():
                return function(*args, **kwargs)
            started = perf_counter()
            token = _TIMING_ACTIVE.set(_TIMING_ACTIVE.get() | {name})
            try:
                return function(*args, **kwargs)
            finally:
                execution = current_execution()
                if execution is not None:
                    with execution.timing_lock:
                        execution.timings[name] = (
                            execution.timings.get(name, 0.0) + perf_counter() - started
                        )
                _TIMING_ACTIVE.reset(token)

        return wrapped

    return decorate


@contextmanager
def native_thread_limit():
    """One scope around outer work; never race process-global limits per task."""
    if os.getenv("LINAK_DISABLE_THREAD_CAP", "").strip().lower() in {"1", "true", "yes", "on"}:
        yield
        return
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


@contextmanager
def computation(threads: int | None = None, memory_budget_mib: float | None = None):
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1.")
    execution = Execution(threads, memory_budget_mib)
    token = _EXECUTION.set(execution)
    try:
        with native_thread_limit():
            yield execution
    finally:
        _EXECUTION.reset(token)


def analysis_execution(function):
    """Keep direct Python calls serial by default; inherit the CLI execution scope."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        existing = current_execution()
        explicit = "threads" in kwargs
        threads = kwargs.get("threads", existing.threads if existing else 1)
        memory = kwargs.get("memory_budget_mib")
        if existing is not None and not explicit:
            return timed_stage("calculation")(function)(*args, **kwargs)
        with computation(threads, memory):
            return timed_stage("calculation")(function)(*args, **kwargs)

    return wrapped


def stage_workers(
    stage: str,
    tasks: int,
    *,
    bytes_per_task: int = 8 * MIB,
    memory_budget_mib: float | None = None,
    requested_threads: int | None = None,
    use_requested: bool = False,
    serial_reason: str | None = None,
) -> int:
    execution = current_execution()
    requested = execution.threads if execution else 1
    if use_requested:
        requested = requested_threads
    ceiling, minimum = AUTO_POLICIES[stage]
    if _IN_WORKER.get():
        return 1
    memory = memory_budget_mib
    if memory is None and execution:
        memory = execution.memory_budget_mib
    policy = resolve_resource_policy(memory)
    # At most two pending tasks per worker; reserve half the incremental budget
    # for coordinator accumulation and readers. Existing resident inputs are not
    # duplicated by the threaded backends.
    per_worker = max(MIB, 4 * int(bytes_per_task))
    workers = resolve_worker_count(
        task_count=tasks,
        estimated_bytes_per_worker=per_worker,
        requested_threads=requested,
        policy=policy,
        logger=LOGGER,
        context=stage,
    )
    if requested is None:
        workers = min(workers, ceiling)
        if tasks < minimum:
            workers = 1
    if serial_reason is not None:
        workers = 1
    allocated = detect_cpu_count()
    reason = serial_reason
    if reason is None:
        if requested == 1:
            reason = "serial CPU budget"
        elif requested is None and tasks < minimum:
            reason = "below automatic parallel threshold"
        elif requested is None and workers == ceiling:
            reason = "automatic worker policy; see benchmark table"
        elif workers < min(tasks, allocated, requested or allocated):
            reason = "memory budget"
        elif tasks < (requested or allocated):
            reason = "available tasks"
        elif requested is not None and requested > allocated:
            reason = "CPU allocation"
        else:
            reason = "requested limit" if requested is not None else "CPU allocation"
    details = dict(
        stage=stage,
        workers=workers,
        allocated_cpus=allocated,
        tasks=tasks,
        requested=requested,
        memory_budget_mib=policy.memory_budget_mib,
        reason=reason,
        estimated_bytes_per_worker=per_worker,
    )
    if execution:
        execution.stages.append(details)
    LOGGER.info(
        "%s: %d worker(s), %d CPU(s) available%s.",
        stage,
        workers,
        details["allocated_cpus"],
        "; " + reason,
    )
    return workers


def batches(values: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, max(1, size))):
        check_cancelled()
        yield chunk


def frame_batch_size(bytes_per_frame: int, maximum: int = 16) -> int:
    """Keep each staged batch below one eighth of the incremental budget."""
    execution = current_execution()
    policy = resolve_resource_policy(execution.memory_budget_mib if execution else None)
    return max(1, min(maximum, policy.memory_budget_bytes // max(1, 8 * bytes_per_frame)))


def _initialize_worker() -> None:
    _IN_WORKER.set(True)
    if os.getenv("LINAK_DISABLE_THREAD_CAP", "").strip().lower() not in {"1", "true", "yes", "on"}:
        from threadpoolctl import threadpool_limits

        # Also set thread-local OpenMP limits in each executor thread.
        threadpool_limits(limits=1)


def _thread_call(context, function, argument):
    def run():
        token = _IN_WORKER.set(True)
        try:
            check_cancelled()
            return function(argument)
        finally:
            _IN_WORKER.reset(token)

    return context.run(run)


def _record_stage_time(stage: str | None, started: float) -> None:
    execution = current_execution()
    if stage is not None and execution is not None:
        with execution.timing_lock:
            execution.timings[stage] = execution.timings.get(stage, 0.0) + perf_counter() - started


def bounded_map(
    function: Callable[[T], R],
    values: Iterable[T],
    *,
    workers: int,
    processes: bool = False,
    initializer=None,
    initargs=(),
    stage: str | None = None,
) -> Iterator[R]:
    """Ordered map with backpressure, prefetch and cooperative cancellation.

    Only the coordinator advances the input iterator (including HDF5 readers).
    At most two tasks per worker are submitted; results are consumed in order.
    """
    started = perf_counter()
    if workers <= 1:
        try:
            for value in values:
                check_cancelled()
                yield function(value)
        finally:
            _record_stage_time(stage, started)
        return
    executor_type = ProcessPoolExecutor if processes else ThreadPoolExecutor
    options = dict(
        max_workers=workers, initializer=initializer or _initialize_worker, initargs=initargs
    )
    if processes:
        options["mp_context"] = multiprocessing.get_context("spawn")
    iterator = iter(values)
    pending: deque[Future[R]] = deque()
    limits = native_thread_limit()
    limits.__enter__()
    try:
        pool = executor_type(**options)
    except BaseException:
        limits.__exit__(None, None, None)
        raise
    try:

        def submit(value):
            if processes:
                return pool.submit(function, value)
            return pool.submit(_thread_call, copy_context(), function, value)

        for value in islice(iterator, workers * 2):
            check_cancelled()
            pending.append(submit(value))
        while pending:
            check_cancelled()
            future = pending.popleft()
            while True:
                try:
                    result = future.result(timeout=0.1)
                    break
                except TimeoutError:
                    if future.done():
                        raise
                    check_cancelled()
            yield result
            check_cancelled()
            try:
                value = next(iterator)
            except StopIteration:
                continue
            pending.append(submit(value))
    finally:
        try:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        finally:
            limits.__exit__(None, None, None)
            _record_stage_time(stage, started)
        LOGGER.debug(
            "Parallel map: %.3f s, %d workers, %s.",
            perf_counter() - started,
            workers,
            "processes" if processes else "threads",
        )
