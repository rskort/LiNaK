"""Qt-independent component view models for the workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .actions import Action
from .defaults import default_settings_for_action, readiness_for_action
from .model import ProjectItem, ProjectStore, Task
from .settings import build_gui_action_settings
from .viewmodels import Badge, display_for_item, display_for_task, item_type_label

ItemGroupingMode = Literal["Workflow", "Type", "Source run", "Flat"]


@dataclass(frozen=True)
class GroupedItemRow:
    item: ProjectItem
    group_label: str
    title: str
    subtitle: str
    badges: tuple[Badge, ...]
    de_emphasized: bool = False


@dataclass(frozen=True)
class ActionRowDisplay:
    action_id: str
    title: str
    description: str
    status: Badge
    readiness_reason: str
    can_run_defaults: bool
    output_preview: tuple[str, ...]
    output_exists: bool
    stale: bool
    cancel_capability: str


@dataclass(frozen=True)
class TaskDetailDisplay:
    title: str
    status: Badge
    settings_hash: str
    settings_lines: tuple[str, ...]
    output_lines: tuple[str, ...]
    failure_summary: str | None
    cancel_capability: str


def grouped_item_rows(
    store: ProjectStore,
    *,
    mode: ItemGroupingMode,
) -> tuple[GroupedItemRow, ...]:
    out_sources = {
        str(item.metadata.get("source_directory") or "")
        for item in store.items
        if item.item_type == "out_hdf5"
    }
    rows: list[GroupedItemRow] = []
    for item in store.items:
        display = display_for_item(item)
        source_dir = str(item.metadata.get("source_directory") or "")
        if mode == "Workflow":
            group = _workflow_group(item)
        elif mode == "Type":
            group = item_type_label(item.item_type)
        elif mode == "Source run":
            group = Path(source_dir).name if source_dir else "Unassigned"
        else:
            group = "Project"
        raw_source = str(item.path.parent) if item.item_type in {"raw_trajectory", "cube_file"} else ""
        rows.append(
            GroupedItemRow(
                item=item,
                group_label=group,
                title=display.title,
                subtitle=display.subtitle,
                badges=display.badges,
                de_emphasized=bool(raw_source and raw_source in out_sources),
            )
        )
    return tuple(
        sorted(rows, key=lambda row: (row.group_label, row.de_emphasized, row.title.lower()))
    )


def action_row_display(
    *,
    action: Action,
    item: ProjectItem,
    store: ProjectStore,
    latest_task: Task | None,
    cancel_capability: str,
) -> ActionRowDisplay:
    readiness = readiness_for_action(action, item)
    can_run_defaults = False
    output_paths: tuple[Path, ...] = ()
    if readiness.available:
        try:
            gui_settings = build_gui_action_settings(
                action,
                item,
                default_settings_for_action(action, item, store.project_dir),
                project_dir=store.project_dir,
            )
            output_paths = gui_settings.output_paths
            can_run_defaults = True
        except Exception:
            can_run_defaults = False
    status = Badge("ready", "neutral")
    if latest_task is not None:
        status = display_for_task(latest_task).status_badge
    output_exists = any(path.exists() for path in output_paths)
    stale = output_exists and _is_stale(item, output_paths)
    return ActionRowDisplay(
        action_id=action.action_id,
        title=action.name,
        description=action.description,
        status=status,
        readiness_reason=readiness.reason,
        can_run_defaults=can_run_defaults,
        output_preview=tuple(path.name for path in output_paths),
        output_exists=output_exists,
        stale=stale,
        cancel_capability=cancel_capability,
    )


def task_detail_display(task: Task) -> TaskDetailDisplay:
    settings = task.settings_snapshot or {}
    settings_lines = tuple(f"{key}: {value}" for key, value in sorted(settings.items()))
    return TaskDetailDisplay(
        title=task.action_name,
        status=display_for_task(task).status_badge,
        settings_hash=task.settings_hash or "",
        settings_lines=settings_lines,
        output_lines=tuple(str(path) for path in task.output_paths),
        failure_summary=task.error,
        cancel_capability=task.cancel_capability,
    )


def _workflow_group(item: ProjectItem) -> str:
    if item.item_type == "simulation_directory":
        return "1. Simulation directories"
    if item.item_type == "out_hdf5":
        return "2. Output containers"
    if item.item_type in {"analysis_hdf5", "cube_hdf5", "trajectory_hdf5"}:
        return "3. Derived outputs"
    return "4. External raw inputs"


def _is_stale(item: ProjectItem, output_paths: tuple[Path, ...]) -> bool:
    try:
        source_mtime = item.path.expanduser().resolve().stat().st_mtime
    except OSError:
        return True
    for path in output_paths:
        try:
            if path.expanduser().resolve().stat().st_mtime < source_mtime:
                return True
        except OSError:
            continue
    return False
