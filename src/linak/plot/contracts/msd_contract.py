"""Plot-data contract adapter for MSD profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_contract import PlotDataContract, PlotDimension, PlotQuantity, PlotViewType

if TYPE_CHECKING:
    from ...analysis.msd import MSDProfile


def _msd_contract(
    *,
    source_id: str,
    label: str,
    frame_length: int | None,
) -> PlotDataContract:
    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=(
            PlotDimension(
                id="frame",
                label="Frame",
                kind="time_index",
                length=None if frame_length is None else int(frame_length),
                unit="index",
            ),
        ),
        quantities=(
            PlotQuantity(
                id="time_fs",
                label="Time (fs)",
                kind="time",
                dimensions=("frame",),
                unit="fs",
                source_name="time_fs",
            ),
            PlotQuantity(
                id="time_ps",
                label="Time (ps)",
                kind="time",
                dimensions=("frame",),
                unit="ps",
                source_name="time_ps",
            ),
            PlotQuantity(
                id="msd",
                label="MSD",
                kind="displacement_squared",
                dimensions=("frame",),
                unit="Angstrom^2",
                source_name="msd_A2",
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


def default_msd_plot_data_contract() -> PlotDataContract:
    """Return a generic MSD contract without profile-specific lengths."""

    return _msd_contract(
        source_id="msd:generic",
        label="MSD data",
        frame_length=None,
    )


def msd_profile_to_plot_data_contract(profile: MSDProfile) -> PlotDataContract:
    """Convert one loaded ``MSDProfile`` into a ``PlotDataContract``."""

    return _msd_contract(
        source_id=f"msd:{profile.species}",
        label=f"MSD profile: {profile.species}",
        frame_length=int(profile.n_frames),
    )
