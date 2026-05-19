"""Generic plot-mapping helpers for potential plotting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts.potential_contract import (
    default_potential_plot_data_contract,
    potential_profiles_to_plot_data_contract,
)
from ..data_contract import PlotDataContract, PlotViewMapping
from ..data_validation import MappingStatus, generic_view_type_compatibility


@dataclass(frozen=True)
class ResolvedPotentialPlotMapping:
    """Bundle the validated potential mapping and translated renderer options."""

    contract: PlotDataContract
    mapping: PlotViewMapping
    compatibility: MappingStatus
    renderer_options: dict[str, object]

    @property
    def view_type(self) -> str:
        """Return the resolved renderer view-type token."""

        return str(self.renderer_options.get("view_type") or "line_1d")

    @property
    def y_quantity(self) -> str:
        """Return the resolved renderer y-quantity token for line views."""

        return str(self.renderer_options.get("y_quantity") or "water_bulk_potential")

    @property
    def standard_plot(self) -> str:
        """Return the resolved renderer standard-plot token for line views."""

        return str(self.renderer_options.get("standard_plot") or "").strip().lower()

    @property
    def is_table_view(self) -> bool:
        """Return whether the active mapping represents a table inspection view."""

        return self.view_type == "table_records"


def potential_plot_options_to_view_mapping(
    *,
    y_quantity: str | None = None,
    table_view: bool = False,
) -> PlotViewMapping:
    """Translate current potential plot options into a generic view mapping."""

    if bool(table_view):
        return PlotViewMapping(
            view_type_id="table_records",
            fixed_values={
                "columns": "record_id,water_bulk_potential,efermi,electrode_cshe",
            },
        )
    normalized_y = "water_bulk_potential" if y_quantity is None else str(y_quantity).strip().lower()
    if normalized_y not in {"water_bulk_potential", "efermi", "electrode_cshe"}:
        raise ValueError(
            "Unsupported potential y quantity. Choose water_bulk_potential, efermi, or electrode_cshe."
        )
    return PlotViewMapping(
        view_type_id="line_1d",
        x="record_id",
        y=normalized_y,
        fixed_values=(
            {"standard_plot": "summary"}
            if y_quantity is None
            else {}
        ),
    )


def potential_view_mapping_to_plot_options(mapping: PlotViewMapping) -> dict[str, object]:
    """Translate one generic potential mapping back into renderer options."""

    view_type_id = str(mapping.view_type_id).strip().lower()
    if view_type_id == "table_records":
        return {"view_type": "table_records"}
    if view_type_id != "line_1d":
        raise ValueError(
            "Current potential plotting only supports 'line_1d' and 'table_records' mappings."
        )
    if str(mapping.x or "").strip() != "record_id":
        raise ValueError("Potential line mappings must place record_id on the x role.")
    y_quantity = str(mapping.y or "").strip()
    if y_quantity not in {"water_bulk_potential", "efermi", "electrode_cshe"}:
        raise ValueError(
            "Potential line mappings must place one supported potential quantity on the y role."
        )
    return {
        "view_type": "line_1d",
        "y_quantity": y_quantity,
        "standard_plot": str(mapping.fixed_values.get("standard_plot") or "").strip().lower(),
    }


def resolve_potential_plot_mapping(
    *,
    contract: PlotDataContract | None = None,
    profiles: Sequence[object] | None = None,
    mapping: PlotViewMapping | None = None,
    y_quantity: str | None = None,
    table_view: bool = False,
) -> ResolvedPotentialPlotMapping:
    """Resolve potential options or a mapping into one runtime mapping."""

    resolved_contract = (
        contract
        if contract is not None
        else default_potential_plot_data_contract()
        if profiles is None
        else potential_profiles_to_plot_data_contract(profiles)
    )
    resolved_mapping = (
        mapping
        if mapping is not None
        else potential_plot_options_to_view_mapping(y_quantity=y_quantity, table_view=table_view)
    )
    compatibility = generic_view_type_compatibility(resolved_contract, resolved_mapping)
    if compatibility == "invalid":
        raise ValueError("Potential plot mapping is incompatible with the available plot data.")
    return ResolvedPotentialPlotMapping(
        contract=resolved_contract,
        mapping=resolved_mapping,
        compatibility=compatibility,
        renderer_options=potential_view_mapping_to_plot_options(resolved_mapping),
    )


def potential_table_rows(profiles: Sequence[object]) -> list[dict[str, float | None]]:
    """Build a simple row-oriented inspection table from fixed potential series."""

    if not profiles:
        return []
    series_by_id = {str(getattr(profile, "series_id", "")): profile for profile in profiles}
    x_values = np.asarray(getattr(profiles[0], "x_values", []), dtype=float)
    rows: list[dict[str, float | None]] = []
    for index, record_id in enumerate(x_values.tolist()):
        row = {
            "record_id": float(record_id),
            "water_bulk_potential": _potential_series_value(series_by_id, "water_bulk_potential_ev", index),
            "efermi": _potential_series_value(series_by_id, "efermi_ev", index),
            "electrode_cshe": _potential_series_value(series_by_id, "electrode_cshe_ev", index),
        }
        rows.append(row)
    return rows


def _potential_series_value(
    series_by_id: dict[str, object],
    series_id: str,
    index: int,
) -> float | None:
    profile = series_by_id.get(series_id)
    if profile is None:
        return None
    values = np.asarray(getattr(profile, "y_values", []), dtype=float)
    if index >= values.size:
        return None
    value = float(values[index])
    return None if not np.isfinite(value) else value
