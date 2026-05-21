"""Plot-data contract adapter for temperature profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_contract import PlotDataContract, PlotDimension, PlotQuantity, PlotViewType

if TYPE_CHECKING:
    from ...analysis.temperature import TemperatureProfile


def _temperature_contract(
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
                id="temperature",
                label="Temperature",
                kind="temperature",
                dimensions=("frame",),
                unit="K",
                source_name="temperature_K",
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


def default_temperature_plot_data_contract() -> PlotDataContract:
    """Return a generic temperature contract without profile-specific lengths."""

    return _temperature_contract(
        source_id="temperature:generic",
        label="Temperature data",
        frame_length=None,
    )


def temperature_profile_to_plot_data_contract(
    profile: TemperatureProfile,
) -> PlotDataContract:
    """Convert one loaded ``TemperatureProfile`` into a ``PlotDataContract``."""

    return _temperature_contract(
        source_id=f"temperature:{profile.default_label}",
        label=f"Temperature profile: {profile.default_label}",
        frame_length=int(profile.n_frames),
    )
