"""Generic plot-mapping helpers for coordination plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

from ..contracts.coordination_contract import (
    coordination_profile_to_plot_data_contract,
    default_coordination_plot_data_contract,
)
from ..data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PlotDataContract,
    PlotViewMapping,
    canonical_plot_view_id,
)
from ..data_validation import MappingStatus, generic_view_type_compatibility

CoordinationMappingPresetName = Literal[
    "coordination_vs_distance",
    "coordination_vs_time",
    "distance_vs_time",
]

_COORDINATION_TIME_AXIS_TOKEN_BY_ID = {
    "time_ps": "ps",
    "time_fs": "fs",
    "step": "step",
    "frame_index": "frame",
}


@dataclass(frozen=True)
class ResolvedCoordinationPlotMapping:
    """Bundle the validated generic mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]

    @property
    def component(self) -> str:
        """Return the resolved renderer component token."""

        return str(self.renderer_options.get("component") or "distance")

    @property
    def time_axis(self) -> str:
        """Return the resolved renderer time-axis token."""

        return str(self.renderer_options.get("time_axis") or "ps")

    @property
    def uses_atom_descriptors(self) -> bool:
        """Return whether the active mapping expands one series per atom."""

        return self.component != "distance"


def coordination_mapping_preset(
    name: CoordinationMappingPresetName,
    *,
    time_axis: str = "ps",
) -> PlotViewMapping:
    """Return one thin default mapping preset for coordination-style data."""

    if name == "coordination_vs_distance":
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_1D_LINE,
            x="distance_to_surface",
            y="coordination_number",
        )
    if name == "coordination_vs_time":
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_1D_LINE,
            x=_coordination_quantity_id_from_token(time_axis),
            y="coordination_number",
            split_by="atom",
        )
    if name == "distance_vs_time":
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_2D_HEATMAP,
            x=_coordination_quantity_id_from_token(time_axis),
            y="distance_to_surface",
            color="coordination_number",
            split_by="atom",
        )
    raise ValueError(f"Unsupported coordination mapping preset '{name}'.")


def coordination_plot_options_to_view_mapping(
    *,
    component: str = "distance",
    time_axis: str = "ps",
) -> PlotViewMapping:
    """Translate current coordination plot options into a generic view mapping."""

    from ...analysis.coordination import _normalize_component_token

    normalized_component = _normalize_component_token(component)
    if normalized_component == "distance":
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_1D_LINE,
            x="distance_to_surface",
            y="coordination_number",
        )
    if normalized_component == "time":
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_1D_LINE,
            x=_coordination_quantity_id_from_token(time_axis),
            y="coordination_number",
            split_by="atom",
        )
    return PlotViewMapping(
        view_type_id=PLOT_VIEW_2D_HEATMAP,
        x=_coordination_quantity_id_from_token(time_axis),
        y="distance_to_surface",
        color="coordination_number",
        split_by="atom",
    )


def coordination_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic coordination mapping back into legacy plot options."""

    view_type_id = str(mapping.view_type_id).strip().lower()
    if canonical_plot_view_id(view_type_id) == PLOT_VIEW_1D_LINE:
        x_quantity = str(mapping.x or "").strip()
        y_quantity = str(mapping.y or "").strip()
        if y_quantity != "coordination_number":
            raise ValueError(
                "Coordination line mappings must place coordination_number on the y role."
            )
        if x_quantity == "distance_to_surface":
            return {
                "component": "distance",
                "time_axis": "ps",
            }
        if x_quantity in _COORDINATION_TIME_AXIS_TOKEN_BY_ID:
            return {
                "component": "time",
                "time_axis": _COORDINATION_TIME_AXIS_TOKEN_BY_ID[x_quantity],
            }
        raise ValueError(f"Unsupported coordination line x quantity '{x_quantity}'.")

    if canonical_plot_view_id(view_type_id) != PLOT_VIEW_2D_HEATMAP:
        raise ValueError(
            "Current coordination plotting only translates generic mappings for "
            "1D Line and 2D Heatmap."
        )

    x_quantity = str(mapping.x or "").strip()
    y_quantity = str(mapping.y or "").strip()
    color_quantity = str(mapping.color or "coordination_number").strip()
    if x_quantity not in _COORDINATION_TIME_AXIS_TOKEN_BY_ID:
        raise ValueError(
            "Coordination 2D Heatmap mappings must place one time quantity on the x role."
        )
    if y_quantity != "distance_to_surface":
        raise ValueError(
            "Coordination 2D Heatmap mappings must place distance_to_surface on the y role."
        )
    if color_quantity != "coordination_number":
        raise ValueError(
            "Coordination 2D Heatmap mappings must use coordination_number as the color quantity."
        )
    if mapping.filter_by is not None and str(mapping.filter_by).strip() != "coordination_number":
        raise ValueError(
            "Current coordination trajectory plotting cannot filter on a quantity other than "
            "coordination_number."
        )
    return {
        "component": "time-distance",
        "time_axis": _COORDINATION_TIME_AXIS_TOKEN_BY_ID[x_quantity],
    }


def resolve_coordination_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
    component: str = "distance",
    time_axis: str = "ps",
) -> ResolvedCoordinationPlotMapping:
    """Resolve legacy coordination options or a mapping into one runtime mapping."""

    resolved_contract = contract
    if resolved_contract is None:
        resolved_contract = (
            default_coordination_plot_data_contract()
            if profile is None
            else coordination_profile_to_plot_data_contract(profile)
        )
    resolved_mapping = mapping
    if resolved_mapping is None:
        resolved_mapping = coordination_plot_options_to_view_mapping(
            component=component,
            time_axis=time_axis,
        )
    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError(
            "Coordination plot mapping is incompatible with the available plot data."
        )
    renderer_options = coordination_view_mapping_to_plot_options(resolved_mapping)
    return ResolvedCoordinationPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=renderer_options,
    )


def _coordination_quantity_id_from_token(token: str) -> str:
    normalized = str(token).strip().lower().replace("_", "-").replace(" ", "-")
    if normalized == "distance":
        return "distance_to_surface"
    if normalized == "coordination":
        return "coordination_number"
    if normalized == "ps":
        return "time_ps"
    if normalized == "fs":
        return "time_fs"
    if normalized == "frame":
        return "frame_index"
    if normalized == "step":
        return "step"
    raise ValueError(f"Unsupported coordination quantity token '{token}'.")
