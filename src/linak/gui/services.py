"""GUI service layer for action execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .actions import Action, ActionContext, ActionExecutionResult
from .settings import GuiActionSettings

CancelCapability = Literal["cooperative", "limited"]


@dataclass(frozen=True)
class ActionServiceDescriptor:
    """Execution properties shown by the GUI before a task runs."""

    action_id: str
    cancel_capability: CancelCapability
    backend_label: str


def descriptor_for_action(action: Action) -> ActionServiceDescriptor:
    # Actions still routed through argparse/CLI internals can only observe cancel requests
    # before and after the backend call. Direct export/pack paths have checkpoints.
    cooperative = {"pack_out_h5", "convert", "export_out_trajectory", "export_out_cube"}
    return ActionServiceDescriptor(
        action_id=action.action_id,
        cancel_capability="cooperative" if action.action_id in cooperative else "limited",
        backend_label="GUI service" if action.action_id in cooperative else "CLI-compatible service",
    )


def execute_action_service(
    action: Action,
    context: ActionContext,
    settings: GuiActionSettings,
) -> ActionExecutionResult:
    """Run one action through the GUI service boundary."""

    if action.backend is None:
        raise ValueError(f"Action '{action.name}' does not have an execution backend.")
    backend_context = ActionContext(
        project_dir=context.project_dir,
        item=context.item,
        settings=settings.to_backend_dict(),
        log=context.log,
        progress=context.progress,
        cancel_requested=context.cancel_requested,
    )
    context.log(
        "INFO",
        f"Settings snapshot {settings.settings_hash}; collision policy: {settings.collision_policy}.",
    )
    return action.backend(backend_context)
