"""Display view-model helpers for the LiNaK workspace GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .actions import Action, ActionRegistry
from .model import ProjectItem, ProjectStore, Task


@dataclass(frozen=True)
class Badge:
    text: str
    tone: str


@dataclass(frozen=True)
class ItemDisplay:
    item_id: str
    icon: str
    title: str
    subtitle: str
    badges: tuple[Badge, ...]
    metadata_lines: tuple[str, ...]
    tooltip: str


@dataclass(frozen=True)
class TaskDisplay:
    task_id: str
    title: str
    subtitle: str
    status_badge: Badge
    log_counts: dict[str, int]
    output_labels: tuple[str, ...]
    progress_label: str | None = None
    progress_fraction: float | None = None


def type_icon(item_type: str) -> str:
    return {
        "raw_trajectory": "[T]",
        "simulation_directory": "[DIR]",
        "out_hdf5": "[OUT]",
        "trajectory_hdf5": "[H5]",
        "temperature_file": "[T]",
        "analysis_hdf5": "[A]",
        "cube_file": "[C]",
        "cube_hdf5": "[C5]",
        "table_hdf5": "[H5]",
    }.get(item_type, "[?]")


def item_type_label(item_type: str) -> str:
    return {
        "raw_trajectory": "Raw trajectory",
        "simulation_directory": "Simulation directory",
        "out_hdf5": "Output container",
        "trajectory_hdf5": "Trajectory HDF5",
        "temperature_file": "Temperature source",
        "analysis_hdf5": "Analysis HDF5",
        "cube_file": "Cube file",
        "cube_hdf5": "Cube HDF5",
        "table_hdf5": "Table HDF5",
        "unsupported": "Unsupported",
        "missing": "Missing",
    }.get(item_type, item_type.replace("_", " ").title())


def validation_badge(item: ProjectItem) -> Badge:
    return {
        "valid": Badge("valid", "success"),
        "warning": Badge("warning", "warning"),
        "invalid": Badge("invalid", "danger"),
    }.get(item.validation.state, Badge(str(item.validation.state), "neutral"))


def origin_badge(item: ProjectItem) -> Badge:
    return Badge("external", "external") if item.origin == "external" else Badge("generated", "generated")


def metadata_lines_for_item(item: ProjectItem, *, max_lines: int = 5) -> tuple[str, ...]:
    metadata = item.metadata
    lines: list[str] = []
    analysis = metadata.get("analysis")
    if analysis:
        lines.append(f"Analysis: {analysis}")
    profile_count = metadata.get("profile_count")
    if profile_count is not None:
        lines.append(f"Profiles: {profile_count}")
    frame_count = metadata.get("frame_count")
    if frame_count is not None:
        lines.append(f"Frames: {frame_count}")
    cube_count = metadata.get("cube_count")
    if cube_count is not None:
        lines.append(f"Cubes: {cube_count}")
    cp2k_output_count = metadata.get("cp2k_output_count")
    if cp2k_output_count is not None:
        lines.append(f"CP2K outputs: {cp2k_output_count}")
    sections = metadata.get("singlepoint_sections")
    if sections:
        lines.append("Singlepoint: " + ", ".join(str(value) for value in sections[:3]))
    size_label = metadata.get("size_label")
    if size_label:
        lines.append(f"Size: {size_label}")
    outputs = item.relationships.get("outputs") or []
    inputs = item.relationships.get("inputs") or []
    if inputs:
        lines.append(f"Inputs: {len(inputs)}")
    if outputs:
        lines.append(f"Outputs: {len(outputs)}")
    action_id = metadata.get("action_id")
    if action_id:
        lines.append(f"Action: {action_id}")
    settings_hash = metadata.get("settings_hash")
    if settings_hash:
        lines.append(f"Settings: {settings_hash}")
    return tuple(lines[:max_lines])


def display_for_item(item: ProjectItem) -> ItemDisplay:
    subtitle_parts = [item_type_label(item.item_type), item.origin]
    if item.metadata.get("analysis"):
        subtitle_parts.insert(0, str(item.metadata["analysis"]).upper())
    return ItemDisplay(
        item_id=item.item_id,
        icon=type_icon(item.item_type),
        title=item.display_name,
        subtitle=" - ".join(subtitle_parts),
        badges=(origin_badge(item), validation_badge(item)),
        metadata_lines=metadata_lines_for_item(item),
        tooltip=f"{item.path}\n{item.validation.state}: {item.validation.message}",
    )


def task_log_counts(task: Task) -> dict[str, int]:
    counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
    for entry in task.logs:
        level = str(entry.get("level", "INFO")).upper()
        if level not in counts:
            counts[level] = 0
        counts[level] += 1
    return counts


def task_status_badge(task: Task) -> Badge:
    return {
        "queued": Badge("queued", "queued"),
        "pending": Badge("pending", "neutral"),
        "running": Badge("running", "info"),
        "canceling": Badge("canceling", "canceling"),
        "canceled": Badge("canceled", "canceled"),
        "finished": Badge("finished", "success"),
        "failed": Badge("failed", "danger"),
    }.get(task.status, Badge(str(task.status), "neutral"))


def task_progress_label(task: Task) -> str | None:
    label = task.progress_label
    fraction = task.progress_fraction
    if label and fraction is not None:
        return f"{int(round(fraction * 100.0))}% - {label}"
    if label:
        return label
    if fraction is not None:
        return f"{int(round(fraction * 100.0))}%"
    return None


def _elapsed_label(task: Task) -> str | None:
    elapsed = task.elapsed_seconds
    if elapsed is None:
        return None
    if elapsed < 60:
        return f"{elapsed:.1f} s"
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes} min {seconds:02d} s"


def display_for_task(task: Task) -> TaskDisplay:
    subtitle_parts: list[str] = []
    if task.started_utc:
        subtitle_parts.append(f"started {task.started_utc}")
    elapsed = _elapsed_label(task)
    if elapsed:
        subtitle_parts.append(f"elapsed {elapsed}")
    if task.error:
        subtitle_parts.append(task.error)
    if task.settings_hash:
        subtitle_parts.append(f"settings {task.settings_hash}")
    if task.cancel_capability == "limited" and task.status in {"running", "canceling"}:
        subtitle_parts.append("limited cancel")
    progress = task_progress_label(task)
    if task.status in {"running", "canceling"} and progress:
        subtitle_parts.append(f"{task.status} {progress}")
    elif task.status == "queued":
        subtitle_parts.append("waiting for the active task")
    return TaskDisplay(
        task_id=task.task_id,
        title=task.action_name,
        subtitle=" - ".join(subtitle_parts),
        status_badge=task_status_badge(task),
        log_counts=task_log_counts(task),
        output_labels=tuple(Path(path).name for path in task.output_paths),
        progress_label=progress,
        progress_fraction=task.progress_fraction,
    )


def suggested_actions_for_item(
    item: ProjectItem,
    registry: ActionRegistry,
    *,
    limit: int = 3,
) -> tuple[Action, ...]:
    actions = registry.available_for(item)
    priority_by_type = {
        "raw_trajectory": ("convert", "density", "msd"),
        "temperature_file": ("temperature",),
        "simulation_directory": ("pack_out_h5",),
        "out_hdf5": ("density", "msd", "rdf"),
        "trajectory_hdf5": ("density", "msd", "rdf"),
        "cube_file": ("convert", "potential"),
        "cube_hdf5": ("potential", "open_plot"),
        "analysis_hdf5": ("open_plot",),
    }
    priority = priority_by_type.get(item.item_type, ())
    ordered = sorted(
        actions,
        key=lambda action: (
            priority.index(action.action_id) if action.action_id in priority else 99,
            action.category,
            action.name,
        ),
    )
    return tuple(ordered[:limit])


def filter_items(
    items: list[ProjectItem],
    *,
    query: str = "",
    origin: str = "all",
    item_type: str = "all",
) -> list[ProjectItem]:
    normalized_query = query.strip().lower()
    filtered: list[ProjectItem] = []
    for item in items:
        if origin != "all" and item.origin != origin:
            continue
        if item_type != "all" and item.item_type != item_type:
            continue
        haystack = " ".join(
            [
                item.display_name,
                str(item.path),
                item.item_type,
                str(item.metadata.get("analysis") or ""),
            ]
        ).lower()
        if normalized_query and normalized_query not in haystack:
            continue
        filtered.append(item)
    return sorted(filtered, key=lambda item: (0 if item.origin == "external" else 1, item.display_name.lower()))


def relationship_names(store: ProjectStore, item: ProjectItem, relation: str) -> tuple[str, ...]:
    names: list[str] = []
    for item_id in item.relationships.get(relation, []):
        related = store.item_by_id(item_id)
        if related is not None:
            names.append(related.display_name)
    return tuple(names)
