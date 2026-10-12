"""Current-format plot settings checks shared by persistence, CLI, and GUI.

Validation is read-only: obsolete saved profiles must be recreated, not
silently mapped onto a different scientific normalization.
"""

from __future__ import annotations

from collections.abc import Mapping


def validate_source_descriptor(descriptor: Mapping[str, object]) -> None:
    """Require a layer to refer to a loaded source series or a copy of one."""
    kind = str(descriptor.get("source_kind") or "source").strip().lower()
    if kind != "source":
        raise ValueError(
            f"Unsupported layer type '{kind}'. Layers must refer to a source series. "
            "Choose a profile containing source layers only."
        )


def validate_current_plot_settings(settings: Mapping[str, object]) -> None:
    """Reject removed options, including options in per-view saved states."""
    descriptors = settings.get("series_descriptors")
    if isinstance(descriptors, (list, tuple)):
        for descriptor in descriptors:
            if isinstance(descriptor, Mapping):
                validate_source_descriptor(descriptor)
    removed = {"heatmap_normalize", "heatmap_normalization_mode"}
    present = removed.intersection(settings)
    if present:
        raise ValueError(
            "Unsupported orientation plot option(s): "
            + ", ".join(sorted(present))
            + ". Recreate the plot profile using heatmap_value_mode; "
            "normalization is not migrated automatically."
        )
    style = settings.get("style")
    if isinstance(style, Mapping):
        validate_current_plot_settings(style)
    for key in ("orientation_view_states", "density_view_states", "position_view_states"):
        states = settings.get(key)
        if isinstance(states, Mapping):
            for state in states.values():
                if isinstance(state, Mapping):
                    validate_current_plot_settings(state)
