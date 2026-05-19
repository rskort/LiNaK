"""Plot-data contract adapter for RDF profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..data_contract import PlotDataContract, PlotDimension, PlotQuantity, PlotViewType

if TYPE_CHECKING:
    from ...analysis.rdf import RDFProfile


def _rdf_contract(
    *,
    source_id: str,
    label: str,
    bin_length: int | None,
) -> PlotDataContract:
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="r_bin",
                label="r bin",
                kind="radial_bin",
                length=None if bin_length is None else int(bin_length),
                unit="index",
            ),
        ),
        quantities=(
            PlotQuantity(
                id="radius",
                label="Radius",
                kind="distance",
                dimensions=("r_bin",),
                unit="Angstrom",
                source_name="bin_centers",
            ),
            PlotQuantity(
                id="g_r",
                label="g(r)",
                kind="distribution",
                dimensions=("r_bin",),
                unit=None,
                source_name="g_r",
            ),
        ),
        view_types=(
            PlotViewType(
                id="line_1d",
                label="Line 1D",
                kind="line_1d",
                supported_roles=("x", "y"),
            ),
        ),
        default_view_type_id="line_1d",
    )


def default_rdf_plot_data_contract() -> PlotDataContract:
    """Return a generic RDF contract without profile-specific lengths."""

    return _rdf_contract(
        source_id="rdf:generic",
        label="RDF data",
        bin_length=None,
    )


def rdf_profile_to_plot_data_contract(profile: RDFProfile) -> PlotDataContract:
    """Convert one loaded ``RDFProfile`` into a ``PlotDataContract``."""

    return _rdf_contract(
        source_id=f"rdf:{profile.species_a}:{profile.species_b}",
        label=f"RDF profile: {profile.species_a}-{profile.species_b}",
        bin_length=int(np.asarray(profile.bin_centers).size),
    )
