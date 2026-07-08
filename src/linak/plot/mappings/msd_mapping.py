"""Generic plot-mapping helpers for MSD plotting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..contracts.msd_contract import default_msd_plot_data_contract, msd_profile_to_plot_data_contract
from ..data_contract import (
    PLOT_VIEW_1D_LINE,
    PlotDataContract,
    PlotViewMapping,
    canonical_plot_view_id,
)
from ..data_validation import MappingStatus, generic_view_type_compatibility


@dataclass(frozen=True)
class ResolvedMsdPlotMapping:
    """Bundle the validated MSD mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]


def msd_plot_options_to_view_mapping(*, time_axis: str = "ps") -> PlotViewMapping:
    """Translate current MSD plot options into a generic view mapping."""

    x_quantity = "time_ps" if str(time_axis).strip().lower() != "fs" else "time_fs"
    return PlotViewMapping(
        view_type_id=PLOT_VIEW_1D_LINE,
        x=x_quantity,
        y="msd",
        fixed_values={"time_axis": "fs" if x_quantity == "time_fs" else "ps"},
    )


def msd_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic MSD mapping back into legacy plot options."""

    if canonical_plot_view_id(mapping.view_type_id) != PLOT_VIEW_1D_LINE:
        raise ValueError("Current MSD plotting only supports 1D Line mappings.")
    x_quantity = str(mapping.x or "").strip()
    if x_quantity not in {"time_ps", "time_fs"}:
        raise ValueError("MSD line mappings must use time_fs or time_ps on the x role.")
    if str(mapping.y or "").strip() != "msd":
        raise ValueError("MSD line mappings must place msd on the y role.")
    return {"time_axis": "fs" if x_quantity == "time_fs" else "ps"}


def resolve_msd_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
    time_axis: str = "ps",
) -> ResolvedMsdPlotMapping:
    """Resolve legacy MSD options or a mapping into one runtime mapping."""

    resolved_contract = (
        contract
        if contract is not None
        else default_msd_plot_data_contract() if profile is None else msd_profile_to_plot_data_contract(profile)
    )
    resolved_mapping = mapping if mapping is not None else msd_plot_options_to_view_mapping(time_axis=time_axis)
    resolved_mapping = replace(
        resolved_mapping,
        view_type_id=canonical_plot_view_id(resolved_mapping.view_type_id),
    )
    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError("MSD plot mapping is incompatible with the available plot data.")
    return ResolvedMsdPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=msd_view_mapping_to_plot_options(resolved_mapping),
    )
