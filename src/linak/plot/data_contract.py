"""Minimal shared plotting vocabulary for LiNaK.

This module defines small structural models that can describe what data a plot
offers and how a future generic plotting layer could map that data onto visual
roles. It intentionally does not connect to the current GUI or renderer yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


PLOT_VIEW_1D_LINE = "plot_1d_line"
PLOT_VIEW_2D_HEATMAP = "plot_2d_heatmap"

PLOT_VIEW_LABEL_1D_LINE = "1D Line"
PLOT_VIEW_LABEL_2D_HEATMAP = "2D Heatmap"

_LEGACY_1D_LINE_VIEW_IDS = frozenset({"line_1d"})
_LEGACY_2D_HEATMAP_VIEW_IDS = frozenset(
    {
        "heatmap_2d",
        "trajectory_2d",
        "scatter_2d",
    }
)
_LEGACY_VIEW_ID_TO_CANONICAL = {
    **{view_id: PLOT_VIEW_1D_LINE for view_id in _LEGACY_1D_LINE_VIEW_IDS},
    **{view_id: PLOT_VIEW_2D_HEATMAP for view_id in _LEGACY_2D_HEATMAP_VIEW_IDS},
    PLOT_VIEW_1D_LINE: PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP: PLOT_VIEW_2D_HEATMAP,
}
_VIEW_LABEL_TO_CANONICAL = {
    "1d": PLOT_VIEW_1D_LINE,
    "1d line": PLOT_VIEW_1D_LINE,
    "line": PLOT_VIEW_1D_LINE,
    "line 1d": PLOT_VIEW_1D_LINE,
    PLOT_VIEW_1D_LINE: PLOT_VIEW_1D_LINE,
    "2d": PLOT_VIEW_2D_HEATMAP,
    "2d heatmap": PLOT_VIEW_2D_HEATMAP,
    "heatmap": PLOT_VIEW_2D_HEATMAP,
    "heatmap 2d": PLOT_VIEW_2D_HEATMAP,
    "2d map": PLOT_VIEW_2D_HEATMAP,
    "trajectory 2d": PLOT_VIEW_2D_HEATMAP,
    "scatter 2d": PLOT_VIEW_2D_HEATMAP,
    PLOT_VIEW_2D_HEATMAP: PLOT_VIEW_2D_HEATMAP,
}


def canonical_plot_view_id(view_type_id: str | None) -> str:
    """Return LiNaK's canonical plot-view token for a legacy or canonical id."""

    token = str(view_type_id or "").strip().lower()
    return _LEGACY_VIEW_ID_TO_CANONICAL.get(token, token)


def plot_view_display_label(view_type_id: str | None) -> str:
    """Return the user-facing plot-view label for a legacy or canonical id."""

    canonical = canonical_plot_view_id(view_type_id)
    if canonical == PLOT_VIEW_1D_LINE:
        return PLOT_VIEW_LABEL_1D_LINE
    if canonical == PLOT_VIEW_2D_HEATMAP:
        return PLOT_VIEW_LABEL_2D_HEATMAP
    return str(view_type_id or "").strip()


def canonical_plot_view_id_from_label(label: str | None) -> str:
    """Resolve a user-facing or legacy view label to a canonical plot-view token."""

    token = str(label or "").strip().lower()
    return _VIEW_LABEL_TO_CANONICAL.get(token, token)


@dataclass(frozen=True)
class PlotDimension:
    """Describe one logical data dimension used by plotted quantities."""

    id: str
    label: str
    kind: str
    length: int | None = None
    unit: str | None = None


@dataclass(frozen=True)
class PlotQuantity:
    """Describe one named quantity together with its dimensions and semantics."""

    id: str
    label: str
    kind: str
    dimensions: tuple[str, ...]
    unit: str | None = None
    source_name: str | None = None


@dataclass(frozen=True)
class PlotViewType:
    """Describe one generic plot/view geometry supported by a data contract."""

    id: str
    label: str
    kind: str
    supported_roles: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlotViewMapping:
    """Assign quantities or dimensions to visual roles for one selected view."""

    view_type_id: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    split_by: str | None = None
    filter_by: str | None = None
    filter_min: float | None = None
    filter_max: float | None = None
    role_assignments: dict[str, str] = field(default_factory=dict)
    fixed_values: dict[str, str] = field(default_factory=dict)

    def resolved_role_assignments(self) -> dict[str, str]:
        """Return explicit mapping fields merged with generic role assignments."""

        resolved = dict(self.role_assignments)
        explicit = {
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "split_by": self.split_by,
            "filter_by": self.filter_by,
        }
        for key, value in explicit.items():
            if value is not None:
                resolved[key] = str(value)
        return resolved

    @classmethod
    def from_mappings(
        cls,
        *,
        view_type_id: str,
        role_assignments: Mapping[str, str] | None = None,
        fixed_values: Mapping[str, str] | None = None,
    ) -> PlotViewMapping:
        """Build a normalized mapping payload from generic mapping inputs."""

        return cls(
            view_type_id=str(view_type_id),
            x=(
                None
                if role_assignments is None or role_assignments.get("x") is None
                else str(role_assignments.get("x"))
            ),
            y=(
                None
                if role_assignments is None or role_assignments.get("y") is None
                else str(role_assignments.get("y"))
            ),
            color=(
                None
                if role_assignments is None or role_assignments.get("color") is None
                else str(role_assignments.get("color"))
            ),
            split_by=(
                None
                if role_assignments is None or role_assignments.get("split_by") is None
                else str(role_assignments.get("split_by"))
            ),
            filter_by=(
                None
                if role_assignments is None or role_assignments.get("filter_by") is None
                else str(role_assignments.get("filter_by"))
            ),
            role_assignments=(
                {}
                if role_assignments is None
                else {
                    str(key): str(value)
                    for key, value in role_assignments.items()
                    if str(key) not in {"x", "y", "color", "split_by", "filter_by"}
                }
            ),
            fixed_values=(
                {}
                if fixed_values is None
                else {str(key): str(value) for key, value in fixed_values.items()}
            ),
        )


@dataclass(frozen=True)
class PlotDataContract:
    """Bundle the dimensions, quantities, and generic views for one plot source."""

    source_id: str
    label: str
    dimensions: tuple[PlotDimension, ...]
    quantities: tuple[PlotQuantity, ...]
    view_types: tuple[PlotViewType, ...]
    default_view_type_id: str | None = None

    @classmethod
    def from_items(
        cls,
        *,
        source_id: str,
        label: str,
        dimensions: Sequence[PlotDimension],
        quantities: Sequence[PlotQuantity],
        view_types: Sequence[PlotViewType],
        default_view_type_id: str | None = None,
    ) -> PlotDataContract:
        """Build a normalized contract from arbitrary dimension/quantity/view sequences."""

        return cls(
            source_id=str(source_id),
            label=str(label),
            dimensions=tuple(dimensions),
            quantities=tuple(quantities),
            view_types=tuple(view_types),
            default_view_type_id=(
                None if default_view_type_id is None else str(default_view_type_id)
            ),
        )
