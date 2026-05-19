"""Generic plot-mapping helpers for position plotting.

This module translates the current position-specific plotting options into the
explicit ``PlotViewMapping`` model without changing renderer behavior yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

from ..contracts.position_contract import (
    default_position_plot_data_contract,
    position_profile_to_plot_data_contract,
)
from ..data_contract import PlotDataContract, PlotViewMapping
from ..data_validation import MappingStatus, generic_view_type_compatibility

PositionMappingPresetName = Literal[
    "distance_vs_time",
    "x_y_trajectory",
    "x_z_trajectory",
    "y_z_trajectory",
]

_POSITION_TIME_AXIS_TOKEN_BY_ID = {
    "time_ps": "ps",
    "time_fs": "fs",
    "step": "step",
    "frame_index": "frame",
}
_POSITION_COMPONENT_TOKEN_BY_ID = {
    "distance_to_surface": "distance",
    "x": "x",
    "y": "y",
    "z": "z",
}
_POSITION_PROJECTION_TOKEN_BY_ID = {
    **_POSITION_TIME_AXIS_TOKEN_BY_ID,
    **_POSITION_COMPONENT_TOKEN_BY_ID,
}
_POSITION_LEGACY_XY_Z_ALIAS_TOKENS = frozenset(
    {
        "xy-z",
        "xy-z-color",
        "xy-z-colormap",
        "trajectory",
        "xyz",
    }
)


@dataclass(frozen=True)
class ResolvedPositionPlotMapping:
    """Bundle the validated generic mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]

    @property
    def uses_profile_descriptors(self) -> bool:
        """Return whether the active mapping should stay at profile granularity in the GUI."""

        return (
            str(self.renderer_options.get("component") or "") == "2d-projection"
            and str(self.renderer_options.get("projection_render_mode") or "color-scale")
            != "line-colors"
        )


def position_mapping_preset(
    name: PositionMappingPresetName,
    *,
    time_axis: str = "ps",
    color_quantity: str = "distance_to_surface",
) -> PlotViewMapping:
    """Return one thin default mapping preset for position-style data."""

    if name == "distance_vs_time":
        return PlotViewMapping(
            view_type_id="line_1d",
            x=_position_quantity_id_from_token(time_axis),
            y="distance_to_surface",
            split_by="atom",
        )
    if name == "x_y_trajectory":
        return PlotViewMapping(
            view_type_id="trajectory_2d",
            x="x",
            y="y",
            color=str(color_quantity),
            split_by="atom",
            fixed_values={"projection_render_mode": "color-scale"},
        )
    if name == "x_z_trajectory":
        return PlotViewMapping(
            view_type_id="trajectory_2d",
            x="x",
            y="z",
            color=str(color_quantity),
            split_by="atom",
            fixed_values={"projection_render_mode": "color-scale"},
        )
    if name == "y_z_trajectory":
        return PlotViewMapping(
            view_type_id="trajectory_2d",
            x="y",
            y="z",
            color=str(color_quantity),
            split_by="atom",
            fixed_values={"projection_render_mode": "color-scale"},
        )
    raise ValueError(f"Unsupported position mapping preset '{name}'.")


def position_plot_options_to_view_mapping(
    *,
    component: str = "distance",
    time_axis: str = "ps",
    map_color: str = "distance",
    projection_x: str | None = None,
    projection_y: str | None = None,
    projection_value: str | None = None,
    projection_render_mode: str | None = None,
    projection_filter_min: float | None = None,
    projection_filter_max: float | None = None,
    xy_z_distance_max: float | None = None,
) -> PlotViewMapping:
    """Translate current position plot options into a generic view mapping."""

    from ...analysis.position import _normalize_component_token, _resolve_projection_settings

    raw_component_token = (
        str(component).strip().lower().replace("_", "-").replace(" ", "-") or "distance"
    )
    normalized_component = _normalize_component_token(component)
    if normalized_component != "2d-projection":
        return PlotViewMapping(
            view_type_id="line_1d",
            x=_position_quantity_id_from_token(time_axis),
            y=_position_quantity_id_from_token(normalized_component),
            split_by="atom",
            fixed_values={"legacy_component": normalized_component},
        )

    (
        resolved_projection_x,
        resolved_projection_y,
        resolved_projection_value,
        resolved_render_mode,
        resolved_filter_min,
        resolved_filter_max,
    ) = _resolve_projection_settings(
        component=component,
        map_color=map_color,
        projection_x=projection_x,
        projection_y=projection_y,
        projection_value=projection_value,
        projection_render_mode=projection_render_mode,
        projection_filter_min=projection_filter_min,
        projection_filter_max=projection_filter_max,
        xy_z_distance_max=xy_z_distance_max,
    )

    use_color_role = resolved_render_mode == "color-scale"
    use_filter_role = resolved_filter_min is not None or resolved_filter_max is not None
    value_quantity_id = _position_quantity_id_from_token(resolved_projection_value)
    fixed_values = {
        "legacy_component": normalized_component,
        "projection_render_mode": resolved_render_mode,
    }
    if raw_component_token in _POSITION_LEGACY_XY_Z_ALIAS_TOKENS:
        fixed_values["legacy_component_alias"] = "xy-z"
        if (
            xy_z_distance_max is not None
            and projection_filter_min is None
            and projection_filter_max is None
            and resolved_projection_value == "distance"
        ):
            fixed_values["legacy_xy_z_distance_cutoff"] = "true"

    return PlotViewMapping(
        view_type_id="trajectory_2d",
        x=_position_quantity_id_from_token(resolved_projection_x),
        y=_position_quantity_id_from_token(resolved_projection_y),
        color=value_quantity_id if use_color_role else None,
        split_by="atom",
        filter_by=value_quantity_id if use_filter_role else None,
        filter_min=resolved_filter_min,
        filter_max=resolved_filter_max,
        fixed_values=fixed_values,
    )


def position_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic position mapping back into legacy plot options."""

    view_type_id = str(mapping.view_type_id).strip().lower()
    if view_type_id == "line_1d":
        x_quantity = str(mapping.x or "").strip()
        y_quantity = str(mapping.y or "").strip()
        if x_quantity not in _POSITION_TIME_AXIS_TOKEN_BY_ID:
            raise ValueError(
                "Position line mappings must place one time quantity on the x role."
            )
        if y_quantity not in _POSITION_COMPONENT_TOKEN_BY_ID:
            raise ValueError(
                "Position line mappings must place one position component on the y role."
            )
        return {
            "component": _POSITION_COMPONENT_TOKEN_BY_ID[y_quantity],
            "time_axis": _POSITION_TIME_AXIS_TOKEN_BY_ID[x_quantity],
            "map_color": "distance",
            "projection_x": "x",
            "projection_y": "y",
            "projection_value": "distance",
            "projection_render_mode": "color-scale",
            "projection_filter_min": None,
            "projection_filter_max": None,
            "xy_z_distance_max": None,
        }

    if view_type_id != "trajectory_2d":
        raise ValueError(
            "Current position plotting only translates generic mappings for "
            "'line_1d' and 'trajectory_2d'."
        )

    x_quantity = str(mapping.x or "").strip()
    y_quantity = str(mapping.y or "").strip()
    if x_quantity not in _POSITION_PROJECTION_TOKEN_BY_ID:
        raise ValueError(f"Unsupported position trajectory x quantity '{x_quantity}'.")
    if y_quantity not in _POSITION_PROJECTION_TOKEN_BY_ID:
        raise ValueError(f"Unsupported position trajectory y quantity '{y_quantity}'.")

    value_quantity = str(mapping.color or mapping.filter_by or "distance_to_surface").strip()
    if mapping.color is not None and mapping.filter_by is not None:
        if str(mapping.color).strip() != str(mapping.filter_by).strip():
            raise ValueError(
                "Current position plotting cannot translate different color and filter "
                "quantities for the same trajectory view."
            )
    if value_quantity not in _POSITION_PROJECTION_TOKEN_BY_ID:
        raise ValueError(
            f"Unsupported position trajectory value quantity '{value_quantity}'."
        )

    render_mode = str(
        mapping.fixed_values.get("projection_render_mode")
        or ("color-scale" if mapping.color is not None else "line-colors")
    ).strip() or "color-scale"
    projection_value = _POSITION_PROJECTION_TOKEN_BY_ID[value_quantity]
    projection_filter_min = mapping.filter_min
    projection_filter_max = mapping.filter_max
    use_legacy_xy_z_cutoff = (
        str(mapping.fixed_values.get("legacy_xy_z_distance_cutoff") or "").strip().lower()
        == "true"
    )
    xy_z_distance_max = None
    if use_legacy_xy_z_cutoff and projection_value == "distance" and projection_filter_min is None:
        xy_z_distance_max = projection_filter_max
        projection_filter_max = None
    elif projection_value == "distance" and projection_filter_min is None:
        xy_z_distance_max = projection_filter_max
    component_token = (
        str(mapping.fixed_values.get("legacy_component_alias") or "").strip() or "2d-projection"
    )
    return {
        "component": component_token,
        "time_axis": "ps",
        "map_color": projection_value if projection_value in {"distance", "z"} else "distance",
        "projection_x": _POSITION_PROJECTION_TOKEN_BY_ID[x_quantity],
        "projection_y": _POSITION_PROJECTION_TOKEN_BY_ID[y_quantity],
        "projection_value": projection_value,
        "projection_render_mode": render_mode,
        "projection_filter_min": projection_filter_min,
        "projection_filter_max": projection_filter_max,
        "xy_z_distance_max": xy_z_distance_max,
    }


def resolve_position_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
    component: str = "distance",
    time_axis: str = "ps",
    map_color: str = "distance",
    projection_x: str | None = None,
    projection_y: str | None = None,
    projection_value: str | None = None,
    projection_render_mode: str | None = None,
    projection_filter_min: float | None = None,
    projection_filter_max: float | None = None,
    xy_z_distance_max: float | None = None,
) -> ResolvedPositionPlotMapping:
    """Resolve legacy position options or an explicit mapping into one validated runtime mapping."""

    resolved_contract = contract
    if resolved_contract is None:
        resolved_contract = (
            default_position_plot_data_contract()
            if profile is None
            else position_profile_to_plot_data_contract(profile)
        )
    resolved_mapping = mapping
    if resolved_mapping is None:
        resolved_mapping = position_plot_options_to_view_mapping(
            component=component,
            time_axis=time_axis,
            map_color=map_color,
            projection_x=projection_x,
            projection_y=projection_y,
            projection_value=projection_value,
            projection_render_mode=projection_render_mode,
            projection_filter_min=projection_filter_min,
            projection_filter_max=projection_filter_max,
            xy_z_distance_max=xy_z_distance_max,
        )
    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError("Position plot mapping is incompatible with the available plot data.")
    renderer_options = position_view_mapping_to_plot_options(resolved_mapping)
    return ResolvedPositionPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=renderer_options,
    )


def _position_quantity_id_from_token(token: str) -> str:
    normalized = str(token).strip().lower().replace("_", "-").replace(" ", "-")
    if normalized == "distance":
        return "distance_to_surface"
    if normalized == "ps":
        return "time_ps"
    if normalized == "fs":
        return "time_fs"
    if normalized == "frame":
        return "frame_index"
    if normalized == "step":
        return "step"
    if normalized in {"x", "y", "z"}:
        return normalized
    raise ValueError(f"Unsupported position quantity token '{token}'.")
