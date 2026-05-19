"""Typed GUI settings and reproducibility helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .actions import Action, validate_action_settings
from .model import ProjectItem

OutputCollisionPolicy = Literal["auto-version", "overwrite"]


@dataclass(frozen=True)
class AutoValue:
    """A GUI setting that can either use an automatic value or a manual value."""

    value: Any = None
    automatic: bool = False
    source: str = ""

    def resolved(self) -> Any:
        return self.value


@dataclass(frozen=True)
class GuiActionSettings:
    """Validated, serializable settings snapshot for one GUI action execution."""

    action_id: str
    values: dict[str, Any]
    settings_hash: str
    output_paths: tuple[Path, ...] = ()
    collision_policy: OutputCollisionPolicy = "auto-version"
    auto_fields: dict[str, str] = field(default_factory=dict)

    def to_backend_dict(self) -> dict[str, Any]:
        return {
            key: value.resolved() if isinstance(value, AutoValue) else value
            for key, value in self.values.items()
        }

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "settings_hash": self.settings_hash,
            "values": _json_safe(self.to_backend_dict()),
            "output_paths": [str(path) for path in self.output_paths],
            "collision_policy": self.collision_policy,
            "auto_fields": dict(self.auto_fields),
        }


def build_gui_action_settings(
    action: Action,
    item: ProjectItem,
    settings: dict[str, Any],
    *,
    project_dir: str | Path,
) -> GuiActionSettings:
    """Validate and normalize GUI settings for execution and manifest storage."""

    backend_settings = {
        key: value.resolved() if isinstance(value, AutoValue) else value
        for key, value in settings.items()
    }
    validate_action_settings(action, item, backend_settings)
    output_paths = action.expected_outputs(
        project_dir=Path(project_dir).expanduser().resolve(),
        item=item,
        settings=backend_settings,
    )
    collision_policy: OutputCollisionPolicy = (
        "overwrite" if bool(backend_settings.get("overwrite")) else "auto-version"
    )
    auto_fields = {
        key: value.source
        for key, value in settings.items()
        if isinstance(value, AutoValue) and value.automatic
    }
    return GuiActionSettings(
        action_id=action.action_id,
        values=dict(backend_settings),
        settings_hash=settings_hash(action.action_id, backend_settings),
        output_paths=tuple(output_paths),
        collision_policy=collision_policy,
        auto_fields=auto_fields,
    )


def settings_hash(action_id: str, settings: dict[str, Any]) -> str:
    payload = {
        "action_id": action_id,
        "settings": _json_safe(settings),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, AutoValue):
        return _json_safe(value.resolved())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
