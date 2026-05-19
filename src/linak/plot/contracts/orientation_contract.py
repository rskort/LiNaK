"""Plot-data contract adapters for orientation line and heatmap views."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..data_contract import PlotDataContract, PlotDimension, PlotQuantity, PlotViewType

if TYPE_CHECKING:
    from ...analysis.orientation import OrientationProfile


def _orientation_line_contract(
    *,
    source_id: str,
    label: str,
    distance_bin_length: int | None,
) -> PlotDataContract:
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="distance_bin",
                label="Distance bin",
                kind="distance_bin",
                length=None if distance_bin_length is None else int(distance_bin_length),
                unit="index",
            ),
        ),
        quantities=(
            PlotQuantity(
                id="bin_centers_A",
                label="Distance bin centers",
                kind="distance",
                dimensions=("distance_bin",),
                unit="Angstrom",
                source_name="bin_centers_A",
            ),
            PlotQuantity(
                id="cos_polar_mean",
                label="Mean cos(polar)",
                kind="orientation_mean",
                dimensions=("distance_bin",),
                unit=None,
                source_name="cos_polar_mean",
            ),
            PlotQuantity(
                id="cos_azimuthal_mean",
                label="Mean cos(azimuthal)",
                kind="orientation_mean",
                dimensions=("distance_bin",),
                unit=None,
                source_name="cos_azimuthal_mean",
            ),
            PlotQuantity(
                id="cos_polar_density",
                label="Density-weighted cos(polar)",
                kind="orientation_density_weighted",
                dimensions=("distance_bin",),
                unit=None,
                source_name="cos_polar_density",
            ),
            PlotQuantity(
                id="cos_azimuthal_density",
                label="Density-weighted cos(azimuthal)",
                kind="orientation_density_weighted",
                dimensions=("distance_bin",),
                unit=None,
                source_name="cos_azimuthal_density",
            ),
            PlotQuantity(
                id="density",
                label="Density",
                kind="density",
                dimensions=("distance_bin",),
                unit=None,
                source_name="density",
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


def _orientation_heatmap_contract(
    *,
    source_id: str,
    label: str,
    distance_bin_length: int | None,
    angle_bin_length: int | None,
) -> PlotDataContract:
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="distance_bin",
                label="Distance bin",
                kind="distance_bin",
                length=None if distance_bin_length is None else int(distance_bin_length),
                unit="index",
            ),
            PlotDimension(
                id="angle_bin",
                label="Angle bin",
                kind="angle_bin",
                length=None if angle_bin_length is None else int(angle_bin_length),
                unit="index",
            ),
        ),
        quantities=(
            PlotQuantity(
                id="bin_centers_A",
                label="Distance bin centers",
                kind="distance",
                dimensions=("distance_bin",),
                unit="Angstrom",
                source_name="bin_centers_A",
            ),
            PlotQuantity(
                id="heatmap_angle_bin_centers",
                label="Heatmap angle bin centers",
                kind="orientation_cosine",
                dimensions=("angle_bin",),
                unit=None,
                source_name="heatmap_angle_bin_centers",
            ),
            PlotQuantity(
                id="heatmap_polar",
                label="Polar heatmap",
                kind="heatmap",
                dimensions=("distance_bin", "angle_bin"),
                unit=None,
                source_name="heatmap_polar",
            ),
            PlotQuantity(
                id="heatmap_azimuthal",
                label="Azimuthal heatmap",
                kind="heatmap",
                dimensions=("distance_bin", "angle_bin"),
                unit=None,
                source_name="heatmap_azimuthal",
            ),
            PlotQuantity(
                id="density",
                label="Density",
                kind="density",
                dimensions=("distance_bin",),
                unit=None,
                source_name="density",
            ),
        ),
        view_types=(
            PlotViewType(
                id="heatmap_2d",
                label="Heatmap 2D",
                kind="heatmap_2d",
                supported_roles=("x", "y", "z"),
            ),
        ),
        default_view_type_id="heatmap_2d",
    )


def default_orientation_line_plot_data_contract() -> PlotDataContract:
    """Return a generic orientation line contract without profile-specific lengths."""

    return _orientation_line_contract(
        source_id="orientation:line:generic",
        label="Orientation line data",
        distance_bin_length=None,
    )


def default_orientation_heatmap_plot_data_contract() -> PlotDataContract:
    """Return a generic orientation heatmap contract without profile-specific lengths."""

    return _orientation_heatmap_contract(
        source_id="orientation:heatmap:generic",
        label="Orientation heatmap data",
        distance_bin_length=None,
        angle_bin_length=None,
    )


def orientation_line_profile_to_plot_data_contract(
    profile: OrientationProfile,
) -> PlotDataContract:
    """Convert one loaded orientation profile into a line-style contract."""

    distance_bin_length = int(np.asarray(profile.bin_centers, dtype=float).size)
    return _orientation_line_contract(
        source_id=f"orientation:line:{profile.axis}:{profile.reference_axis}:{profile.coordinate_mode}",
        label="Orientation line profile",
        distance_bin_length=distance_bin_length,
    )


def orientation_heatmap_profile_to_plot_data_contract(
    profile: OrientationProfile,
) -> PlotDataContract:
    """Convert one loaded orientation profile into a heatmap-style contract."""

    distance_bin_length = int(np.asarray(profile.bin_centers, dtype=float).size)
    angle_bin_length = int(np.asarray(profile.heatmap_angle_bin_centers, dtype=float).size)
    return _orientation_heatmap_contract(
        source_id=(
            f"orientation:heatmap:{profile.axis}:{profile.reference_axis}:{profile.coordinate_mode}"
        ),
        label="Orientation heatmap profile",
        distance_bin_length=distance_bin_length,
        angle_bin_length=angle_bin_length,
    )
