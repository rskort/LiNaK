"""Current-format plot settings checks shared by persistence, CLI, and GUI.

Validation is read-only: obsolete saved profiles must be recreated, not
silently mapped onto a different scientific normalization.
"""

from __future__ import annotations

from collections.abc import Mapping

from .rendering import validate_render_settings


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
    validate_render_settings(settings)
    kind = settings.get("orientation_heatmap_kind")
    quantity = settings.get("orientation_heatmap_quantity")
    if kind is not None and kind not in {"distribution", "spatial"}:
        raise ValueError("Orientation heatmap kind must be distribution or spatial.")
    if quantity is not None and quantity not in {"average", "density", "density-weighted"}:
        raise ValueError("Orientation heatmap quantity must be average, density, or density-weighted.")
    alignments = settings.get("interface_alignments")
    if alignments is not None:
        from ..analysis.interface_alignment import InterfaceAlignment

        if not isinstance(alignments, Mapping):
            raise ValueError("Interface alignments must be a dictionary keyed by reference identity.")
        for identifier, record in alignments.items():
            if not isinstance(record, Mapping):
                raise ValueError("Invalid interface alignment settings.")
            alignment = InterfaceAlignment.from_dict(record)
            if identifier != alignment.reference_id:
                raise ValueError("Interface alignment reference identity does not match its settings key.")
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
    for key in ("orientation_view_states", "orientation_heatmap_states", "density_view_states", "position_view_states"):
        states = settings.get(key)
        if isinstance(states, Mapping):
            for state in states.values():
                if isinstance(state, Mapping):
                    validate_current_plot_settings(state)
