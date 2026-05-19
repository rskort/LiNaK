"""Background task management for the project workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any

from .actions import Action, ActionCanceled, ActionContext, validate_action_settings
from .logging import TaskLogHandler
from .model import ProjectItem, ProjectStore, Task
from .services import descriptor_for_action, execute_action_service
from .settings import GuiActionSettings, build_gui_action_settings


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_requested(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class _QueuedWork:
    task: Task
    action: Action
    item: ProjectItem
    settings: GuiActionSettings
    on_update: Callable[[Task], None] | None
    on_finished: Callable[[Task], None] | None
    token: CancellationToken


class TaskManager:
    """Single-worker priority queue with cooperative cancellation."""

    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._running_task_id: str | None = None
        self._queue: list[_QueuedWork] = []
        self._tokens: dict[str, CancellationToken] = {}
        self._queue_paused = False

    @property
    def running_task_id(self) -> str | None:
        with self._lock:
            return self._running_task_id

    def can_start_task(self) -> bool:
        return True

    def is_queue_paused(self) -> bool:
        with self._lock:
            return self._queue_paused

    def start(
        self,
        *,
        action: Action,
        item: ProjectItem,
        settings: dict[str, Any],
        priority: int = 0,
        on_update: Callable[[Task], None] | None = None,
        on_finished: Callable[[Task], None] | None = None,
    ) -> Task:
        if action.backend is None:
            raise ValueError(f"Action '{action.name}' does not have an execution backend.")
        validate_action_settings(action, item, settings)
        gui_settings = build_gui_action_settings(
            action,
            item,
            settings,
            project_dir=self.store.project_dir,
        )
        descriptor = descriptor_for_action(action)
        token = CancellationToken()
        task = Task(
            action_id=action.action_id,
            action_name=action.name,
            input_item_id=item.item_id,
            status="pending",
            settings_snapshot=gui_settings.to_backend_dict(),
            settings_hash=gui_settings.settings_hash,
            output_metadata={
                "expected_outputs": [str(path) for path in gui_settings.output_paths],
                "collision_policy": gui_settings.collision_policy,
                "auto_fields": dict(gui_settings.auto_fields),
            },
            priority=int(priority),
            cancel_capability=descriptor.cancel_capability,
        )
        work = _QueuedWork(
            task=task,
            action=action,
            item=item,
            settings=gui_settings,
            on_update=on_update,
            on_finished=on_finished,
            token=token,
        )
        with self._lock:
            self.store.add_task(task)
            self._tokens[task.task_id] = token
            if self._running_task_id is None and not self._queue_paused:
                self._running_task_id = task.task_id
                launch_now = True
            else:
                task.status = "queued"
                task.add_log("INFO", "Queued behind the active task.")
                self._enqueue_locked(work)
                launch_now = False
        self._emit_update(on_update, task)
        if launch_now:
            self._launch(work)
        return task

    def pause_queue(self) -> None:
        with self._lock:
            self._queue_paused = True

    def resume_queue(self) -> None:
        next_work: _QueuedWork | None = None
        with self._lock:
            self._queue_paused = False
            if self._running_task_id is None and self._queue:
                next_work = self._queue.pop(0)
                self._running_task_id = next_work.task.task_id
        if next_work is not None:
            self._launch(next_work)

    def cancel_all_queued(self) -> int:
        with self._lock:
            queued = list(self._queue)
            self._queue.clear()
            for work in queued:
                self._tokens.pop(work.task.task_id, None)
                work.task.status = "canceled"
                work.task.finished_utc = _now()
                work.task.add_log("WARNING", "Canceled while queued.")
        for work in queued:
            self._emit_update(work.on_update, work.task)
            if work.on_finished is not None:
                work.on_finished(work.task)
        self._safe_save()
        return len(queued)

    def reorder_queued(self, task_id: str, direction: int) -> bool:
        with self._lock:
            index = next(
                (idx for idx, work in enumerate(self._queue) if work.task.task_id == task_id),
                None,
            )
            if index is None:
                return False
            target = max(0, min(len(self._queue) - 1, index + int(direction)))
            if target == index:
                return False
            self._queue[index], self._queue[target] = self._queue[target], self._queue[index]
            return True

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            for index, work in enumerate(self._queue):
                if work.task.task_id == task_id:
                    self._queue.pop(index)
                    self._tokens.pop(task_id, None)
                    work.task.status = "canceled"
                    work.task.finished_utc = _now()
                    work.task.add_log("WARNING", "Canceled while queued.")
                    callbacks = (work.on_update, work.on_finished, work.task)
                    break
            else:
                token = self._tokens.get(task_id)
                task = self._task_by_id(task_id)
                if token is None or task is None:
                    return False
                token.cancel()
                if task.status == "running":
                    task.status = "canceling"
                    task.add_log("WARNING", "Cancel requested. Waiting for backend checkpoint.")
                elif task.status == "pending":
                    task.status = "canceled"
                    task.finished_utc = _now()
                    task.add_log("WARNING", "Canceled before start.")
                callbacks = (None, None, task)
        self._safe_save()
        self._emit_update(callbacks[0], callbacks[2])
        if callbacks[1] is not None:
            callbacks[1](callbacks[2])
        return True

    def _enqueue_locked(self, work: _QueuedWork) -> None:
        self._queue.append(work)
        self._queue.sort(key=lambda queued: (-queued.task.priority, queued.task.created_utc))

    def _task_by_id(self, task_id: str) -> Task | None:
        for task in self.store.tasks:
            if task.task_id == task_id:
                return task
        return None

    def _launch(self, work: _QueuedWork) -> None:
        thread = threading.Thread(target=self._run_task, kwargs={"work": work}, daemon=True)
        thread.start()

    def _emit_update(self, callback: Callable[[Task], None] | None, task: Task) -> None:
        if callback is not None:
            callback(task)

    def _run_task(self, *, work: _QueuedWork) -> None:
        task = work.task
        action = work.action
        item = work.item
        settings = work.settings
        token = work.token

        def _log(level: str, message: str) -> None:
            task.add_log(level, message)
            self._emit_update(work.on_update, task)

        last_progress_emit = 0.0
        last_progress_log_key: tuple[str, int | None] | None = None

        def _progress(label: str, current: int | None, total: int | None) -> None:
            nonlocal last_progress_emit, last_progress_log_key
            changed = task.set_progress(label, current, total)
            if not changed:
                return
            fraction = task.progress_fraction
            bucket = None if fraction is None else int(fraction * 10.0)
            log_key = (task.progress_label or "", bucket)
            if log_key != last_progress_log_key:
                last_progress_log_key = log_key
                if fraction is None:
                    task.add_log("INFO", f"Progress: {task.progress_label}")
                else:
                    task.add_log(
                        "INFO",
                        f"Progress: {round(fraction * 100.0):.0f}% - {task.progress_label}",
                    )
            now = time.monotonic()
            if now - last_progress_emit >= 0.1 or fraction in {0.0, 1.0}:
                last_progress_emit = now
                self._emit_update(work.on_update, task)

        if token.is_requested():
            task.status = "canceled"
            task.finished_utc = _now()
            task.add_log("WARNING", "Canceled before start.")
            self._finish_work(work)
            return

        task.status = "running"
        task.started_utc = _now()
        task.set_progress("Starting", 0, 1)
        _log("INFO", f"Started {action.name} for {item.display_name}.")

        logger = logging.getLogger("linak")
        previous_level = logger.level
        handler = TaskLogHandler(_log)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            result = execute_action_service(
                action,
                ActionContext(
                    project_dir=self.store.project_dir,
                    item=item,
                    settings=settings.to_backend_dict(),
                    log=_log,
                    progress=_progress,
                    cancel_requested=token.is_requested,
                ),
                settings,
            )
            if token.is_requested():
                raise ActionCanceled("Task canceled after backend checkpoint.")
            for message in result.messages:
                _log("INFO", message)
            task.output_paths = [Path(path).expanduser().resolve() for path in result.output_paths]
            task.set_progress("Finished", 1, 1)
            task.status = "finished"
            if task.output_paths:
                first = self.store.find_item_by_path(task.output_paths[0])
                if first is not None:
                    task.primary_output_item_id = first.item_id
                _log("INFO", "Generated: " + ", ".join(path.name for path in task.output_paths))
            else:
                _log("INFO", "Task finished without declaring a generated output.")
        except ActionCanceled as exc:
            task.status = "canceled"
            task.output_paths = []
            task.error = None
            _log("WARNING", str(exc) or "Task canceled.")
        except Exception as exc:
            if token.is_requested():
                task.status = "canceled"
                task.output_paths = []
                task.error = None
                _log("WARNING", str(exc) or "Task canceled.")
            else:
                task.status = "failed"
                task.error = str(exc)
                _log("ERROR", str(exc))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            task.finished_utc = _now()
            self._finish_work(work)

    def _finish_work(self, work: _QueuedWork) -> None:
        task = work.task
        next_work: _QueuedWork | None = None
        with self._lock:
            self._tokens.pop(task.task_id, None)
            if self._running_task_id == task.task_id:
                self._running_task_id = None
            if self._running_task_id is None and self._queue and not self._queue_paused:
                next_work = self._queue.pop(0)
                self._running_task_id = next_work.task.task_id
        self._safe_save()
        self._emit_update(work.on_update, task)
        if work.on_finished is not None:
            work.on_finished(task)
        if next_work is not None:
            self._launch(next_work)

    def _safe_save(self) -> None:
        try:
            self.store.save()
        except OSError:
            pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
