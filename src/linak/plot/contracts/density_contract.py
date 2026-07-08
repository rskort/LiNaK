"""Plot-data contract adapters for density line and heatmap views."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PlotDataContract,
    PlotDimension,
    PlotQuantity,
    PlotViewType,
    plot_view_display_label,
)

if TYPE_CHECKING:
    from ...analysis.density import DensityHeatmapProfile, DensityProfile


def _density_contract(
    *,
    source_id: str,
    label: str,
    axis_token: str,
    coordinate_mode: str,
    bin_length: int | None,
    supports_axis_coordinates: bool,
    supports_surface_distance: bool,
    supports_number_density: bool,
) -> PlotDataContract:
    quantities = [
        PlotQuantity(
            id="bin_center",
            label=(
                "Distance to surface"
                if str(coordinate_mode).strip().lower() == "distance"
                else f"{axis_token.upper()} coordinate"
            ),
            kind="distance" if str(coordinate_mode).strip().lower() == "distance" else "coordinate",
            dimensions=("bin",),
            unit="Angstrom",
            source_name="bin_centers",
        ),
        PlotQuantity(
            id="mass_density",
            label="Mass density",
            kind="density",
            dimensions=("bin",),
            unit=None,
            source_name="density",
        ),
    ]
    if supports_number_density:
        quantities.append(
            PlotQuantity(
                id="number_density",
                label="Number density",
                kind="density",
                dimensions=("bin",),
                unit=None,
                source_name="number_density",
            )
        )
    if supports_axis_coordinates:
        quantities.append(
            PlotQuantity(
                id="axis_coordinate",
                label=f"{axis_token.upper()} coordinate",
                kind="coordinate",
                dimensions=("bin",),
                unit="Angstrom",
                source_name="bin_centers",
            )
        )
    if supports_surface_distance:
        quantities.append(
            PlotQuantity(
                id="distance_to_surface",
                label="Distance to surface",
                kind="distance",
                dimensions=("bin",),
                unit="Angstrom",
                source_name="bin_centers",
            )
        )
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="bin",
                label="Bin",
                kind="histogram_bin",
                length=None if bin_length is None else int(bin_length),
                unit="index",
            ),
        ),
        quantities=tuple(quantities),
        view_types=(
            PlotViewType(
                id=PLOT_VIEW_1D_LINE,
                label=plot_view_display_label(PLOT_VIEW_1D_LINE),
                kind=PLOT_VIEW_1D_LINE,
                supported_roles=("x", "y"),
            ),
        ),
        default_view_type_id=PLOT_VIEW_1D_LINE,
    )


def _density_heatmap_contract(
    *,
    source_id: str,
    label: str,
    plane_token: str,
    x_axis_token: str,
    y_axis_token: str,
    x_bin_length: int | None,
    y_bin_length: int | None,
    supports_number_density: bool,
) -> PlotDataContract:
    quantities = [
        PlotQuantity(
            id="x_bin_center",
            label=f"{x_axis_token.upper()} coordinate",
            kind="coordinate",
            dimensions=("x_bin",),
            unit="Angstrom",
            source_name="x_bin_centers_A",
        ),
        PlotQuantity(
            id="y_bin_center",
            label=f"{y_axis_token.upper()} coordinate",
            kind="coordinate",
            dimensions=("y_bin",),
            unit="Angstrom",
            source_name="y_bin_centers_A",
        ),
        PlotQuantity(
            id="mass_density_2d",
            label=f"{plane_token.upper()} mass density",
            kind="heatmap",
            dimensions=("x_bin", "y_bin"),
            unit=None,
            source_name="density",
        ),
    ]
    if supports_number_density:
        quantities.append(
            PlotQuantity(
                id="number_density_2d",
                label=f"{plane_token.upper()} number density",
                kind="heatmap",
                dimensions=("x_bin", "y_bin"),
                unit=None,
                source_name="number_density",
            )
        )
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="x_bin",
                label=f"{x_axis_token.upper()} bin",
                kind="histogram_bin",
                length=None if x_bin_length is None else int(x_bin_length),
                unit="index",
            ),
            PlotDimension(
                id="y_bin",
                label=f"{y_axis_token.upper()} bin",
                kind="histogram_bin",
                length=None if y_bin_length is None else int(y_bin_length),
                unit="index",
            ),
        ),
        quantities=tuple(quantities),
        view_types=(
            PlotViewType(
                id=PLOT_VIEW_2D_HEATMAP,
                label=plot_view_display_label(PLOT_VIEW_2D_HEATMAP),
                kind=PLOT_VIEW_2D_HEATMAP,
                supported_roles=("x", "y", "z"),
            ),
        ),
        default_view_type_id=PLOT_VIEW_2D_HEATMAP,
    )


def default_density_plot_data_contract() -> PlotDataContract:
    """Return a generic density contract without profile-specific lengths."""

    return _density_contract(
        source_id="density:generic",
        label="Density data",
        axis_token="z",
        coordinate_mode="distance",
        bin_length=None,
        supports_axis_coordinates=True,
        supports_surface_distance=True,
        supports_number_density=True,
    )


def default_density_heatmap_plot_data_contract() -> PlotDataContract:
    """Return a generic density heatmap contract without profile-specific lengths."""

    return _density_heatmap_contract(
        source_id="density:heatmap:generic",
        label="Density heatmap data",
        plane_token="xy",
        x_axis_token="x",
        y_axis_token="y",
        x_bin_length=None,
        y_bin_length=None,
        supports_number_density=True,
    )


def density_profile_to_plot_data_contract(profile: DensityProfile) -> PlotDataContract:
    """Convert one loaded ``DensityProfile`` into a ``PlotDataContract``."""

    coordinate_mode = str(profile.coordinate_mode).strip().lower() or "axis"
    has_surface_reference = bool(
        profile.surface_position is not None and np.isfinite(float(profile.surface_position))
    )
    return _density_contract(
        source_id=f"density:{profile.species}:{profile.axis}:{coordinate_mode}",
        label=f"Density profile: {profile.species}",
        axis_token=str(profile.axis).strip().lower() or "z",
        coordinate_mode=coordinate_mode,
        bin_length=int(np.asarray(profile.bin_centers).size),
        supports_axis_coordinates=(coordinate_mode != "axis" and has_surface_reference)
        or coordinate_mode == "axis",
        supports_surface_distance=(coordinate_mode == "distance") or has_surface_reference,
        supports_number_density=(
            profile.number_density is not None and profile.number_density_units is not None
        ),
    )


def density_heatmap_profile_to_plot_data_contract(
    profile: DensityHeatmapProfile,
) -> PlotDataContract:
    """Convert one loaded ``DensityHeatmapProfile`` into a ``PlotDataContract``."""

    return _density_heatmap_contract(
        source_id=f"density:heatmap:{profile.species}:{profile.plane}",
        label=f"Density heatmap: {profile.species} {profile.plane.upper()}",
        plane_token=profile.plane,
        x_axis_token=profile.plane_axes[0],
        y_axis_token=profile.plane_axes[1],
        x_bin_length=int(np.asarray(profile.x_bin_centers).size),
        y_bin_length=int(np.asarray(profile.y_bin_centers).size),
        supports_number_density=(
            profile.number_density is not None and profile.number_density_units is not None
        ),
    )
