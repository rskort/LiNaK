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
from ..data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PlotDataContract,
    PlotViewMapping,
    canonical_plot_view_id,
)
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

        return canonical_plot_view_id(self.mapping.view_type_id) == PLOT_VIEW_2D_HEATMAP


def orientation_plot_options_to_view_mapping(
    *,
    component: str = "average",
    angle: str = "polar",
    line_x_axis: str | None = None,
    heatmap_x_axis: str | None = None,
    heatmap_y_axis: str | None = None,
) -> PlotViewMapping:
    """Translate current orientation plot options into a generic view mapping."""

    normalized_component = _normalize_orientation_component(component)
    normalized_angle = _normalize_orientation_angle(angle)
    normalized_line_x_axis = _normalize_orientation_line_x_axis(line_x_axis)
    if normalized_component == "heatmap":
        fixed_values: dict[str, str] = {}
        if heatmap_x_axis is not None or heatmap_y_axis is not None:
            normalized_heatmap_x_axis = _normalize_orientation_grid_axis(
                heatmap_x_axis,
                default="x",
                label="2D Heatmap x-axis",
            )
            normalized_heatmap_y_axis = _normalize_orientation_grid_axis(
                heatmap_y_axis,
                default="y",
                label="2D Heatmap y-axis",
            )
            if normalized_heatmap_x_axis == normalized_heatmap_y_axis:
                raise ValueError("Orientation 2D Heatmap axes must be different.")
            fixed_values = {
                "orientation_heatmap_x_axis": normalized_heatmap_x_axis,
                "orientation_heatmap_y_axis": normalized_heatmap_y_axis,
            }
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_2D_HEATMAP,
            x="bin_centers_A",
            y="heatmap_angle_bin_centers",
            role_assignments={
                "z": "heatmap_polar" if normalized_angle == "polar" else "heatmap_azimuthal",
            },
            fixed_values=fixed_values,
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
        view_type_id=PLOT_VIEW_1D_LINE,
        x="bin_centers_A",
        y=y_quantity,
        fixed_values={"orientation_line_x_axis": normalized_line_x_axis},
    )


def orientation_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic orientation mapping back into legacy plot options."""

    view_type_id = _orientation_legacy_view_type_id(mapping.view_type_id)
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
        return {
            "component": normalized_component,
            "angle": normalized_angle,
            "orientation_line_x_axis": _normalize_orientation_line_x_axis(
                mapping.fixed_values.get("orientation_line_x_axis")
            ),
        }

    if view_type_id != "heatmap_2d":
        raise ValueError(
            "Current orientation plotting only supports 1D Line and 2D Heatmap views."
        )
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
    return {
        "component": "heatmap",
        "angle": normalized_angle,
        "orientation_heatmap_x_axis": _normalize_orientation_grid_axis(
            mapping.fixed_values.get("orientation_heatmap_x_axis"),
            default="x",
            label="2D Heatmap x-axis",
        ),
        "orientation_heatmap_y_axis": _normalize_orientation_grid_axis(
            mapping.fixed_values.get("orientation_heatmap_y_axis"),
            default="y",
            label="2D Heatmap y-axis",
        ),
    }


def resolve_orientation_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
    component: str = "average",
    angle: str = "polar",
    line_x_axis: str | None = None,
    heatmap_x_axis: str | None = None,
    heatmap_y_axis: str | None = None,
) -> ResolvedOrientationPlotMapping:
    """Resolve orientation options or a mapping into one runtime mapping."""

    resolved_mapping = (
        mapping
        if mapping is not None
        else orientation_plot_options_to_view_mapping(
            component=component,
            angle=angle,
            line_x_axis=line_x_axis,
            heatmap_x_axis=heatmap_x_axis,
            heatmap_y_axis=heatmap_y_axis,
        )
    )
    resolved_contract = contract
    if resolved_contract is None:
        if _orientation_legacy_view_type_id(resolved_mapping.view_type_id) == "heatmap_2d":
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
            f"Unsupported orientation quantity '{component}'. "
            "Choose average, density, density-weighted, or heatmap."
        )
    return token


def _normalize_orientation_line_x_axis(axis: str | None) -> str:
    token = "distance" if axis is None else str(axis).strip().lower()
    if token in {"", "bin_centers_a", "bin_centers", "distance_to_surface"}:
        token = "distance"
    if token not in {"distance", "x", "y", "z"}:
        raise ValueError(
            f"Unsupported orientation 1D Line x-axis quantity '{axis}'. "
            "Choose distance, x, y, or z."
        )
    return token


def _normalize_orientation_grid_axis(
    axis: str | None,
    *,
    default: str,
    label: str,
) -> str:
    token = default if axis is None else str(axis).strip().lower()
    if token in {"", "distance_to_surface"}:
        token = "distance"
    if token not in {"distance", "x", "y", "z"}:
        raise ValueError(
            f"Unsupported orientation {label} quantity '{axis}'. Choose distance, x, y, or z."
        )
    return token


def _orientation_legacy_view_type_id(view_type: str | None) -> str:
    canonical = canonical_plot_view_id(view_type)
    if canonical == PLOT_VIEW_1D_LINE:
        return "line_1d"
    if canonical == PLOT_VIEW_2D_HEATMAP:
        return "heatmap_2d"
    return str(view_type or "").strip().lower() or "line_1d"


def _normalize_orientation_angle(angle: str | None) -> str:
    token = "polar" if angle is None else str(angle).strip().lower()
    if token not in _ORIENTATION_ANGLES:
        raise ValueError(f"Unsupported orientation angle '{angle}'. Choose polar or azimuthal.")
    return token
