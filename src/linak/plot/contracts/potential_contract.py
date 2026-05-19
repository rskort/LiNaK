"""Plot-data contract adapter for potential summary series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data_contract import PlotDataContract, PlotDimension, PlotQuantity, PlotViewType


def _potential_contract(
    *,
    source_id: str,
    label: str,
    record_length: int | None,
) -> PlotDataContract:
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="record",
                label="Record",
                kind="record_index",
                length=None if record_length is None else int(record_length),
                unit="index",
            ),
        ),
        quantities=(
            PlotQuantity(
                id="record_id",
                label="Record ID",
                kind="identifier",
                dimensions=("record",),
                unit="index",
                source_name="record_id",
            ),
            PlotQuantity(
                id="water_bulk_potential",
                label="Water bulk potential",
                kind="potential",
                dimensions=("record",),
                unit="eV",
                source_name="water_bulk_potential_ev",
            ),
            PlotQuantity(
                id="efermi",
                label="Fermi energy",
                kind="potential",
                dimensions=("record",),
                unit="eV",
                source_name="efermi_ev",
            ),
            PlotQuantity(
                id="electrode_cshe",
                label="Electrode cSHE",
                kind="potential",
                dimensions=("record",),
                unit="eV",
                source_name="electrode_cshe_ev",
            ),
        ),
        view_types=(
            PlotViewType(
                id="line_1d",
                label="Line 1D",
                kind="line_1d",
                supported_roles=("x", "y"),
            ),
            PlotViewType(
                id="table_records",
                label="Table records",
                kind="table_records",
                supported_roles=(),
            ),
        ),
        default_view_type_id="line_1d",
    )


def default_potential_plot_data_contract() -> PlotDataContract:
    """Return a generic potential contract without source-specific lengths."""

    return _potential_contract(
        source_id="potential:generic",
        label="Potential data",
        record_length=None,
    )


def potential_profiles_to_plot_data_contract(profiles: Sequence[object]) -> PlotDataContract:
    """Convert loaded potential plot series into a ``PlotDataContract``."""

    first_profile = profiles[0] if profiles else None
    source_path = (
        ""
        if first_profile is None
        else str(getattr(first_profile, "source_path", "")).strip()
    )
    record_length = (
        None
        if first_profile is None
        else int(np.asarray(getattr(first_profile, "x_values", [])).size)
    )
    return _potential_contract(
        source_id=f"potential:{source_path or 'summary'}",
        label="Potential summary",
        record_length=record_length,
    )
