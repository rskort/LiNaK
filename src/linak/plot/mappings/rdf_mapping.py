"""Generic plot-mapping helpers for RDF plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.rdf_contract import default_rdf_plot_data_contract, rdf_profile_to_plot_data_contract
from ..data_contract import PlotDataContract, PlotViewMapping
from ..data_validation import MappingStatus, generic_view_type_compatibility


@dataclass(frozen=True)
class ResolvedRdfPlotMapping:
    """Bundle the validated RDF mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]


def rdf_plot_options_to_view_mapping() -> PlotViewMapping:
    """Translate current RDF plot options into a generic view mapping."""

    return PlotViewMapping(view_type_id="line_1d", x="radius", y="g_r")


def rdf_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic RDF mapping back into legacy plot options."""

    if str(mapping.view_type_id).strip().lower() != "line_1d":
        raise ValueError("Current RDF plotting only supports 'line_1d' mappings.")
    if str(mapping.x or "").strip() != "radius":
        raise ValueError("RDF line mappings must place radius on the x role.")
    if str(mapping.y or "").strip() != "g_r":
        raise ValueError("RDF line mappings must place g_r on the y role.")
    return {}


def resolve_rdf_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profile: Any | None = None,
    mapping: PlotViewMapping | None = None,
) -> ResolvedRdfPlotMapping:
    """Resolve an RDF mapping into one runtime mapping."""

    resolved_contract = (
        contract
        if contract is not None
        else default_rdf_plot_data_contract() if profile is None else rdf_profile_to_plot_data_contract(profile)
    )
    resolved_mapping = mapping if mapping is not None else rdf_plot_options_to_view_mapping()
    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError("RDF plot mapping is incompatible with the available plot data.")
    return ResolvedRdfPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=rdf_view_mapping_to_plot_options(resolved_mapping),
    )
