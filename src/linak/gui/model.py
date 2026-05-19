"""Central project workspace data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

ProjectItemOrigin = Literal["external", "generated"]
ValidationState = Literal["valid", "warning", "invalid"]
TaskStatus = Literal[
    "queued",
    "pending",
    "running",
    "canceling",
    "canceled",
    "finished",
    "failed",
]

MANIFEST_NAME = ".linak_project.json"


@dataclass
class ValidationResult:
    """Lightweight validation result for one project item."""

    state: ValidationState
    message: str = ""


@dataclass
class ProjectItem:
    """A referenced external input or generated project output."""

    path: Path
    item_type: str
    origin: ProjectItemOrigin
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult("warning", "Not validated")
    )
    relationships: dict[str, list[str]] = field(default_factory=dict)
    item_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def display_name(self) -> str:
        return self.path.name or str(self.path)

    @property
    def type_label(self) -> str:
        return self.item_type.replace("_", " ")

    def to_manifest_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload

    @classmethod
    def from_manifest_record(cls, payload: dict[str, Any]) -> ProjectItem:
        validation_payload = payload.get("validation") or {}
        return cls(
            path=Path(str(payload.get("path", ""))).expanduser(),
            item_type=str(payload.get("item_type", "unsupported")),
            origin=str(payload.get("origin", "external")),  # type: ignore[arg-type]
            metadata=dict(payload.get("metadata") or {}),
            validation=ValidationResult(
                state=str(validation_payload.get("state", "warning")),  # type: ignore[arg-type]
                message=str(validation_payload.get("message", "")),
            ),
            relationships={
                str(key): [str(value) for value in values]
                for key, values in dict(payload.get("relationships") or {}).items()
            },
            item_id=str(payload.get("item_id") or uuid4().hex),
        )


@dataclass
class Task:
    """A running or completed workspace action."""

    action_id: str
    action_name: str
    input_item_id: str
    status: TaskStatus = "pending"
    logs: list[dict[str, str]] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)
    primary_output_item_id: str | None = None
    error: str | None = None
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    settings_hash: str | None = None
    output_metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    cancel_capability: str = "limited"
    progress_label: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    task_id: str = field(default_factory=lambda: uuid4().hex)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    started_utc: str | None = None
    finished_utc: str | None = None

    @property
    def elapsed_seconds(self) -> float | None:
        if not self.started_utc or not self.finished_utc:
            return None
        try:
            start = datetime.fromisoformat(self.started_utc)
            finish = datetime.fromisoformat(self.finished_utc)
        except ValueError:
            return None
        return max(0.0, (finish - start).total_seconds())

    @property
    def progress_fraction(self) -> float | None:
        if self.progress_current is None or self.progress_total is None:
            return None
        if self.progress_total <= 0:
            return None
        return max(0.0, min(1.0, float(self.progress_current) / float(self.progress_total)))

    def set_progress(
        self,
        label: str,
        current: int | None,
        total: int | None,
    ) -> bool:
        normalized_label = str(label).strip() or "Working"
        normalized_current = None if current is None else max(0, int(current))
        normalized_total = None if total is None else max(0, int(total))
        changed = (
            self.progress_label != normalized_label
            or self.progress_current != normalized_current
            or self.progress_total != normalized_total
        )
        self.progress_label = normalized_label
        self.progress_current = normalized_current
        self.progress_total = normalized_total
        return changed

    def add_log(self, level: str, message: str) -> None:
        self.logs.append(
            {
                "level": str(level).upper(),
                "message": str(message),
                "time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )

    def to_manifest_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_paths"] = [str(path) for path in self.output_paths]
        payload["elapsed_seconds"] = self.elapsed_seconds
        return payload

    @classmethod
    def from_manifest_record(cls, payload: dict[str, Any]) -> Task:
        return cls(
            action_id=str(payload.get("action_id", "")),
            action_name=str(payload.get("action_name", "")),
            input_item_id=str(payload.get("input_item_id", "")),
            status=str(payload.get("status", "finished")),  # type: ignore[arg-type]
            logs=[dict(entry) for entry in payload.get("logs", [])],
            output_paths=[Path(str(path)) for path in payload.get("output_paths", [])],
            primary_output_item_id=payload.get("primary_output_item_id"),
            error=payload.get("error"),
            settings_snapshot=dict(payload.get("settings_snapshot") or {}),
            settings_hash=payload.get("settings_hash"),
            output_metadata=dict(payload.get("output_metadata") or {}),
            priority=int(payload.get("priority") or 0),
            cancel_capability=str(payload.get("cancel_capability") or "limited"),
            progress_label=payload.get("progress_label"),
            progress_current=(
                None
                if payload.get("progress_current") is None
                else int(payload.get("progress_current"))
            ),
            progress_total=(
                None
                if payload.get("progress_total") is None
                else int(payload.get("progress_total"))
            ),
            task_id=str(payload.get("task_id") or uuid4().hex),
            created_utc=str(payload.get("created_utc", "")),
            started_utc=payload.get("started_utc"),
            finished_utc=payload.get("finished_utc"),
        )


class ProjectStore:
    """Owns project items, task records, discovery, and manifest persistence."""

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.manifest_path = self.project_dir / MANIFEST_NAME
        self.items: list[ProjectItem] = []
        self.tasks: list[Task] = []
        self.workspace_index = WorkspaceIndex()

    def initialize(self) -> bool:
        """Create the project directory when needed and return whether it was created."""

        existed = self.project_dir.exists()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        return not existed

    def load(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.items = [
            ProjectItem.from_manifest_record(record)
            for record in payload.get("items", [])
            if isinstance(record, dict)
        ]
        self.tasks = [
            Task.from_manifest_record(record)
            for record in payload.get("tasks", [])
            if isinstance(record, dict)
        ]
        self.workspace_index = WorkspaceIndex.from_manifest_record(
            payload.get("workspace_index") or {}
        )

    def save(self) -> None:
        payload = {
            "schema_version": 1,
            "project_dir": str(self.project_dir),
            "items": [item.to_manifest_record() for item in self.items],
            "tasks": [task.to_manifest_record() for task in self.tasks[-200:]],
            "workspace_index": self.workspace_index.to_manifest_record(),
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def item_by_id(self, item_id: str) -> ProjectItem | None:
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def find_item_by_path(self, path: str | Path) -> ProjectItem | None:
        resolved = Path(path).expanduser().resolve()
        for item in self.items:
            if item.path.expanduser().resolve() == resolved:
                return item
        return None

    def upsert_item(self, item: ProjectItem) -> ProjectItem:
        existing = self.find_item_by_path(item.path)
        if existing is None:
            self.items.append(item)
            return item
        existing.item_type = item.item_type
        existing.origin = item.origin
        existing.metadata = item.metadata
        existing.validation = item.validation
        existing.relationships.update(item.relationships)
        return existing

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def remove_task(self, task_id: str) -> Task | None:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                return self.tasks.pop(index)
        return None

    def remove_item(self, item_id: str) -> ProjectItem | None:
        removed: ProjectItem | None = None
        kept: list[ProjectItem] = []
        for item in self.items:
            if item.item_id == item_id:
                removed = item
                continue
            kept.append(item)
        if removed is None:
            return None
        self.items = kept
        for item in self.items:
            for relation, related_ids in list(item.relationships.items()):
                item.relationships[relation] = [
                    related_id for related_id in related_ids if related_id != item_id
                ]
                if not item.relationships[relation]:
                    del item.relationships[relation]
        self.workspace_index.forget(removed.path)
        return removed

    def can_delete_generated_file(self, item: ProjectItem) -> bool:
        if item.origin != "generated":
            return False
        try:
            resolved = item.path.expanduser().resolve()
            resolved.relative_to(self.project_dir)
        except (OSError, ValueError):
            return False
        return resolved.exists() and resolved.is_file()

    def delete_generated_item_file(self, item_id: str) -> Path | None:
        item = self.item_by_id(item_id)
        if item is None or not self.can_delete_generated_file(item):
            return None
        resolved = item.path.expanduser().resolve()
        resolved.unlink()
        self.remove_item(item_id)
        return resolved


@dataclass
class WorkspaceIndexEntry:
    """Cached lightweight detection result for one filesystem path."""

    path: str
    size_bytes: int
    modified_time: float
    item_record: dict[str, Any]

    @property
    def signature(self) -> tuple[int, float]:
        return int(self.size_bytes), float(self.modified_time)


class WorkspaceIndex:
    """Versioned cache for lightweight project item detection."""

    schema_version = 1

    def __init__(self, entries: dict[str, WorkspaceIndexEntry] | None = None) -> None:
        self._entries = {} if entries is None else dict(entries)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": {
                key: {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "modified_time": entry.modified_time,
                    "item_record": entry.item_record,
                }
                for key, entry in self._entries.items()
            },
        }

    @classmethod
    def from_manifest_record(cls, payload: dict[str, Any]) -> WorkspaceIndex:
        if int(payload.get("schema_version", 0) or 0) != cls.schema_version:
            return cls()
        entries: dict[str, WorkspaceIndexEntry] = {}
        for key, raw in dict(payload.get("entries") or {}).items():
            if not isinstance(raw, dict):
                continue
            item_record = raw.get("item_record")
            if not isinstance(item_record, dict):
                continue
            entries[str(key)] = WorkspaceIndexEntry(
                path=str(raw.get("path") or key),
                size_bytes=int(raw.get("size_bytes") or 0),
                modified_time=float(raw.get("modified_time") or 0.0),
                item_record=dict(item_record),
            )
        return cls(entries)

    def cached_item(self, path: str | Path) -> ProjectItem | None:
        resolved = Path(path).expanduser().resolve()
        try:
            stat = resolved.stat()
        except OSError:
            return None
        key = str(resolved)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.signature != (int(stat.st_size), float(stat.st_mtime)):
            return None
        return ProjectItem.from_manifest_record(entry.item_record)

    def remember(self, item: ProjectItem) -> None:
        try:
            stat = item.path.expanduser().resolve().stat()
        except OSError:
            return
        resolved = item.path.expanduser().resolve()
        self._entries[str(resolved)] = WorkspaceIndexEntry(
            path=str(resolved),
            size_bytes=int(stat.st_size),
            modified_time=float(stat.st_mtime),
            item_record=item.to_manifest_record(),
        )

    def forget(self, path: str | Path) -> None:
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            resolved = Path(path).expanduser()
        self._entries.pop(str(resolved), None)

    def detect_or_reuse(
        self,
        path: str | Path,
        *,
        origin: ProjectItemOrigin,
        detector: Any,
    ) -> ProjectItem:
        cached = self.cached_item(path)
        if cached is not None:
            cached.origin = origin
            return cached
        item = detector(path, origin=origin)
        self.remember(item)
        return item
