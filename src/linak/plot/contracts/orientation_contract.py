"""Plot-data contract adapters for orientation line and heatmap views."""

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
                id=PLOT_VIEW_1D_LINE,
                label=plot_view_display_label(PLOT_VIEW_1D_LINE),
                kind=PLOT_VIEW_1D_LINE,
                supported_roles=("x", "y"),
            ),
        ),
        default_view_type_id=PLOT_VIEW_1D_LINE,
    )


def _orientation_heatmap_contract(
    *,
    source_id: str,
    label: str,
    distance_bin_length: int | None,
    angle_bin_length: int | None,
    sparse_grid_cell_count: int | None = None,
) -> PlotDataContract:
    dimensions = [
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
    ]
    quantities = [
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
    ]
    if sparse_grid_cell_count is not None:
        dimensions.append(
            PlotDimension(
                id="sparse_grid_cell",
                label="Sparse grid cell",
                kind="sparse_grid_cell",
                length=int(sparse_grid_cell_count),
                unit="index",
            )
        )
        quantities.extend(
            [
                PlotQuantity(
                    id="grid_flat_indices",
                    label="Sparse grid flat indices",
                    kind="sparse_grid_index",
                    dimensions=("sparse_grid_cell",),
                    unit="index",
                    source_name="grid_flat_indices",
                ),
                PlotQuantity(
                    id="grid_entity_sum",
                    label="Sparse grid entity count",
                    kind="orientation_grid_count",
                    dimensions=("sparse_grid_cell",),
                    unit="count",
                    source_name="grid_entity_sum",
                ),
                PlotQuantity(
                    id="grid_cos_polar_sum",
                    label="Sparse grid cos(polar) sum",
                    kind="orientation_grid_sum",
                    dimensions=("sparse_grid_cell",),
                    unit=None,
                    source_name="grid_cos_polar_sum",
                ),
                PlotQuantity(
                    id="grid_count_polar_valid",
                    label="Sparse grid polar valid count",
                    kind="orientation_grid_count",
                    dimensions=("sparse_grid_cell",),
                    unit="count",
                    source_name="grid_count_polar_valid",
                ),
                PlotQuantity(
                    id="grid_cos_azimuthal_sum",
                    label="Sparse grid cos(azimuthal) sum",
                    kind="orientation_grid_sum",
                    dimensions=("sparse_grid_cell",),
                    unit=None,
                    source_name="grid_cos_azimuthal_sum",
                ),
                PlotQuantity(
                    id="grid_count_azimuthal_valid",
                    label="Sparse grid azimuthal valid count",
                    kind="orientation_grid_count",
                    dimensions=("sparse_grid_cell",),
                    unit="count",
                    source_name="grid_count_azimuthal_valid",
                ),
            ]
        )
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=tuple(dimensions),
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
    sparse_grid = getattr(profile, "sparse_grid", None)
    sparse_grid_cell_count = (
        None
        if sparse_grid is None
        else int(np.asarray(getattr(sparse_grid, "flat_indices"), dtype=np.int64).size)
    )
    return _orientation_heatmap_contract(
        source_id=(
            f"orientation:heatmap:{profile.axis}:{profile.reference_axis}:{profile.coordinate_mode}"
        ),
        label="Orientation heatmap profile",
        distance_bin_length=distance_bin_length,
        angle_bin_length=angle_bin_length,
        sparse_grid_cell_count=sparse_grid_cell_count,
    )
