"""Generic plot-mapping helpers for orientation plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.orientation_contract import (
    default_orientation_heatmap_plot_data_contract,
    default_orientation_line_plot_data_contract,
    orientation_heatmap_profile_to_plot_data_contract,
    orientation_line_profile_to_plot_data_contract,
)
from ..data_contract import PlotDataContract, PlotViewMapping
from ..data_validation import MappingStatus, generic_view_type_compatibility

_ORIENTATION_COMPONENTS = frozenset({"average", "density", "density-weighted", "heatmap"})
_ORIENTATION_ANGLES = frozenset({"polar", "azimuthal"})


@dataclass(frozen=True)
class ResolvedOrientationPlotMapping:
    """Bundle the validated orientation mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]

    @property
    def component(self) -> str:
        """Return the resolved renderer component token."""

        return str(self.renderer_options.get("component") or "average")

    @property
    def angle(self) -> str:
        """Return the resolved renderer angle token."""

        return str(self.renderer_options.get("angle") or "polar")

    @property
    def is_heatmap(self) -> bool:
        """Return whether the active mapping targets heatmap rendering."""

        return str(self.mapping.view_type_id).strip().lower() == "heatmap_2d"


def orientation_plot_options_to_view_mapping(
    *,
    component: str = "average",
    angle: str = "polar",
) -> PlotViewMapping:
    """Translate current orientation plot options into a generic view mapping."""

    normalized_component = _normalize_orientation_component(component)
    normalized_angle = _normalize_orientation_angle(angle)
    if normalized_component == "heatmap":
        return PlotViewMapping(
            view_type_id="heatmap_2d",
            x="bin_centers_A",
            y="heatmap_angle_bin_centers",
            role_assignments={
                "z": "heatmap_polar" if normalized_angle == "polar" else "heatmap_azimuthal",
            },
        )

    y_quantity = {
        ("average", "polar"): "cos_polar_mean",
        ("average", "azimuthal"): "cos_azimuthal_mean",
        ("density", "polar"): "density",
        ("density", "azimuthal"): "density",
        ("density-weighted", "polar"): "cos_polar_density",
        ("density-weighted", "azimuthal"): "cos_azimuthal_density",
    }[(normalized_component, normalized_angle)]
    return PlotViewMapping(
        view_type_id="line_1d",
        x="bin_centers_A",
        y=y_quantity,
    )


def orientation_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic orientation mapping back into legacy plot options."""

    view_type_id = str(mapping.view_type_id).strip().lower()
    normalized_angle = "polar"
    if view_type_id == "line_1d":
        if str(mapping.x or "").strip() != "bin_centers_A":
            raise ValueError("Orientation line mappings must place bin_centers_A on the x role.")
        y_quantity = str(mapping.y or "").strip()
        if y_quantity == "density":
            normalized_component = "density"
        elif y_quantity == "cos_polar_mean":
            normalized_component = "average"
            normalized_angle = "polar"
        elif y_quantity == "cos_azimuthal_mean":
            normalized_component = "average"
            normalized_angle = "azimuthal"
        elif y_quantity == "cos_polar_density":
            normalized_component = "density-weighted"
            normalized_angle = "polar"
        elif y_quantity == "cos_azimuthal_density":
            normalized_component = "density-weighted"
            normalized_angle = "azimuthal"
        else:
            raise ValueError(
                "Orientation line mappings must use one supported line quantity on the y role."
            )
        return {"component": normalized_component, "angle": normalized_angle}

    if view_type_id != "heatmap_2d":
        raise ValueError("Current orientation plotting only supports line_1d and heatmap_2d.")
    if str(mapping.x or "").strip() != "bin_centers_A":
        raise ValueError("Orientation heatmap mappings must place bin_centers_A on the x role.")
    if str(mapping.resolved_role_assignments().get("y") or "").strip() != "heatmap_angle_bin_centers":
        raise ValueError(
            "Orientation heatmap mappings must place heatmap_angle_bin_centers on the y role."
        )
    z_quantity = str(mapping.resolved_role_assignments().get("z") or "").strip()
    if z_quantity == "heatmap_polar":
        normalized_angle = "polar"
    elif z_quantity == "heatmap_azimuthal":
        normalized_angle = "azimuthal"
    else:
        raise ValueError(
            "Orientation heatmap mappings must use heatmap_polar or heatmap_azimuthal on the z role."
        )
    return {"component": "heatmap", "angle": normalized_angle}


def resolve_orientation_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
    component: str = "average",
    angle: str = "polar",
) -> ResolvedOrientationPlotMapping:
    """Resolve orientation options or a mapping into one runtime mapping."""

    resolved_mapping = (
        mapping
        if mapping is not None
        else orientation_plot_options_to_view_mapping(component=component, angle=angle)
    )
    resolved_contract = contract
    if resolved_contract is None:
        if str(resolved_mapping.view_type_id).strip().lower() == "heatmap_2d":
            resolved_contract = (
                default_orientation_heatmap_plot_data_contract()
                if profile is None
                else orientation_heatmap_profile_to_plot_data_contract(profile)
            )
        else:
            resolved_contract = (
                default_orientation_line_plot_data_contract()
                if profile is None
                else orientation_line_profile_to_plot_data_contract(profile)
            )

    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError(
            "Orientation plot mapping is incompatible with the available plot data."
        )
    return ResolvedOrientationPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=orientation_view_mapping_to_plot_options(resolved_mapping),
    )


def _normalize_orientation_component(component: str | None) -> str:
    token = "average" if component is None else str(component).strip().lower()
    if token not in _ORIENTATION_COMPONENTS:
        raise ValueError(
            f"Unsupported orientation component '{component}'. "
            "Choose average, density, density-weighted, or heatmap."
        )
    return token


def _normalize_orientation_angle(angle: str | None) -> str:
    token = "polar" if angle is None else str(angle).strip().lower()
    if token not in _ORIENTATION_ANGLES:
        raise ValueError(f"Unsupported orientation angle '{angle}'. Choose polar or azimuthal.")
    return token
