"""Minimal shared plotting vocabulary for LiNaK.

This module defines small structural models that can describe what data a plot
offers and how a future generic plotting layer could map that data onto visual
roles. It intentionally does not connect to the current GUI or renderer yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


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
