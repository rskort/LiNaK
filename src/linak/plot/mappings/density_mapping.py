"""Generic plot-mapping helpers for density plotting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..contracts.density_contract import (
    default_density_heatmap_plot_data_contract,
    default_density_plot_data_contract,
    density_heatmap_profile_to_plot_data_contract,
    density_profile_to_plot_data_contract,
)
from ..data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PlotDataContract,
    PlotViewMapping,
    canonical_plot_view_id,
)
from ..data_validation import MappingStatus, generic_view_type_compatibility


@dataclass(frozen=True)
class ResolvedDensityPlotMapping:
    """Bundle the validated density mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]

    @property
    def x_mode(self) -> str:
        """Return the resolved renderer x-mode token."""

        return str(self.renderer_options.get("x_mode") or "distance")

    @property
    def quantity(self) -> str:
        """Return the resolved renderer quantity token."""

        return str(self.renderer_options.get("quantity") or "mass")

    @property
    def view_type_id(self) -> str:
        """Return the resolved generic view type id."""

        return canonical_plot_view_id(self.mapping.view_type_id)

    @property
    def is_heatmap(self) -> bool:
        """Return whether the active mapping targets heatmap rendering."""

        return self.view_type_id == PLOT_VIEW_2D_HEATMAP


def density_plot_options_to_view_mapping(
    *,
    view_type: str = "line_1d",
    x_mode: str = "distance",
    quantity: str = "mass",
) -> PlotViewMapping:
    """Translate current density plot options into a generic view mapping."""

    normalized_view_type = _density_legacy_view_type_id(view_type)
    normalized_x_mode = str(x_mode).strip().lower() or "distance"
    normalized_quantity = str(quantity).strip().lower() or "mass"
    if normalized_quantity not in {"mass", "number"}:
        raise ValueError(f"Unsupported density quantity '{quantity}'. Choose 'mass' or 'number'.")
    if normalized_view_type == "heatmap_2d":
        z_quantity = "mass_density_2d" if normalized_quantity == "mass" else "number_density_2d"
        return PlotViewMapping(
            view_type_id=PLOT_VIEW_2D_HEATMAP,
            x="x_bin_center",
            y="y_bin_center",
            role_assignments={"z": z_quantity},
        )
    if normalized_view_type != "line_1d":
        raise ValueError("Current density plotting only supports 1D Line and 2D Heatmap views.")
    if normalized_x_mode == "distance":
        x_quantity = "distance_to_surface"
    elif normalized_x_mode in {"axis", "x", "y", "z"}:
        x_quantity = "axis_coordinate"
    else:
        raise ValueError(
            f"Unsupported density x_mode '{x_mode}'. Choose 'distance', 'x', 'y', 'z', or 'axis'."
        )
    y_quantity = "mass_density" if normalized_quantity == "mass" else "number_density"
    fixed_values: dict[str, str] = {}
    if x_quantity == "axis_coordinate":
        fixed_values["x_mode"] = normalized_x_mode
    return PlotViewMapping(
        view_type_id=PLOT_VIEW_1D_LINE,
        x=x_quantity,
        y=y_quantity,
        fixed_values=fixed_values,
    )


def density_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic density mapping back into legacy plot options."""

    view_type_id = _density_legacy_view_type_id(mapping.view_type_id)
    if view_type_id == "heatmap_2d":
        if str(mapping.x or "").strip() != "x_bin_center":
            raise ValueError("Density heatmap mappings must place x_bin_center on the x role.")
        if str(mapping.y or "").strip() != "y_bin_center":
            raise ValueError("Density heatmap mappings must place y_bin_center on the y role.")
        z_quantity = str(mapping.resolved_role_assignments().get("z") or "").strip()
        if z_quantity == "mass_density_2d":
            quantity = "mass"
        elif z_quantity == "number_density_2d":
            quantity = "number"
        else:
            raise ValueError(
                "Density heatmap mappings must use mass_density_2d or number_density_2d on the z role."
            )
        return {"view_type": "heatmap_2d", "quantity": quantity}
    if view_type_id != "line_1d":
        raise ValueError(
            "Current density plotting only supports 1D Line and 2D Heatmap mappings."
        )
    x_quantity = str(mapping.x or "").strip()
    y_quantity = str(mapping.y or "").strip()
    if x_quantity == "distance_to_surface":
        x_mode = "distance"
    elif x_quantity == "axis_coordinate":
        x_mode = str(mapping.fixed_values.get("x_mode") or "axis").strip().lower() or "axis"
    else:
        raise ValueError(
            "Density line mappings must place axis_coordinate or distance_to_surface on the x role."
        )
    if y_quantity == "mass_density":
        quantity = "mass"
    elif y_quantity == "number_density":
        quantity = "number"
    else:
        raise ValueError(
            "Density line mappings must place mass_density or number_density on the y role."
        )
    return {"view_type": "line_1d", "x_mode": x_mode, "quantity": quantity}


def resolve_density_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
    view_type: str = "line_1d",
    x_mode: str = "distance",
    quantity: str = "mass",
) -> ResolvedDensityPlotMapping:
    """Resolve legacy density options or a mapping into one runtime mapping."""

    resolved_mapping = (
        mapping
        if mapping is not None
        else density_plot_options_to_view_mapping(
            view_type=view_type,
            x_mode=x_mode,
            quantity=quantity,
        )
    )
    resolved_mapping = replace(
        resolved_mapping,
        view_type_id=canonical_plot_view_id(resolved_mapping.view_type_id),
    )
    if contract is not None:
        resolved_contract = contract
    elif profile is not None:
        resolved_contract = (
            density_heatmap_profile_to_plot_data_contract(profile)
            if hasattr(profile, "plane_axes")
            else density_profile_to_plot_data_contract(profile)
        )
    elif _density_legacy_view_type_id(resolved_mapping.view_type_id) == "heatmap_2d":
        resolved_contract = default_density_heatmap_plot_data_contract()
    else:
        resolved_contract = default_density_plot_data_contract()
    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError("Density plot mapping is incompatible with the available plot data.")
    return ResolvedDensityPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=density_view_mapping_to_plot_options(resolved_mapping),
    )


def _density_legacy_view_type_id(view_type: str | None) -> str:
    canonical = canonical_plot_view_id(view_type)
    if canonical == PLOT_VIEW_1D_LINE:
        return "line_1d"
    if canonical == PLOT_VIEW_2D_HEATMAP:
        return "heatmap_2d"
    return str(view_type or "").strip().lower() or "line_1d"
