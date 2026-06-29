"""Plot-data contract adapter for atom-resolved position profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_contract import (
    PlotDataContract,
    PlotDimension,
    PlotQuantity,
    PlotViewType,
)

if TYPE_CHECKING:
    from ...analysis.position import PositionProfile


def _position_contract(
    *,
    source_id: str,
    label: str,
    axis_token: str,
    coordinate_mode: str,
    frame_length: int | None,
    atom_length: int | None,
    entity_kind: str = "atom",
) -> PlotDataContract:
    normalized_entity_kind = str(entity_kind or "atom").strip().lower()
    entity_label = "Molecule" if normalized_entity_kind == "molecule" else "Atom"
    dimensions = (
        PlotDimension(
            id="frame",
            label="Frame",
            kind="time_index",
            length=None if frame_length is None else int(frame_length),
            unit="index",
        ),
        PlotDimension(
            id="atom",
            label=entity_label,
            kind="entity_index",
            length=None if atom_length is None else int(atom_length),
            unit="index",
        ),
    )

    quantities = (
        PlotQuantity(
            id="frame_index",
            label="Frame index",
            kind="time_index",
            dimensions=("frame",),
            unit="index",
            source_name="frame_index",
        ),
        PlotQuantity(
            id="step",
            label="Step",
            kind="time_index",
            dimensions=("frame",),
            unit="step",
            source_name="step",
        ),
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
            id="atom_index",
            label=f"{entity_label} index",
            kind="entity_index",
            dimensions=("atom",),
            unit="index",
            source_name="atom_indices",
        ),
        PlotQuantity(
            id="x",
            label="X",
            kind="coordinate",
            dimensions=("frame", "atom"),
            unit="Angstrom",
            source_name="x_A",
        ),
        PlotQuantity(
            id="y",
            label="Y",
            kind="coordinate",
            dimensions=("frame", "atom"),
            unit="Angstrom",
            source_name="y_A",
        ),
        PlotQuantity(
            id="z",
            label="Z",
            kind="coordinate",
            dimensions=("frame", "atom"),
            unit="Angstrom",
            source_name="z_A",
        ),
        PlotQuantity(
            id="distance_to_surface",
            label=(
                "Distance to surface"
                if str(coordinate_mode).strip().lower() == "distance"
                else f"{axis_token.upper()} axis / distance field"
            ),
            kind="distance" if str(coordinate_mode).strip().lower() == "distance" else "coordinate",
            dimensions=("frame", "atom"),
            unit="Angstrom",
            source_name="distance_to_surface_A",
        ),
    )

    view_types = (
        PlotViewType(
            id="line_1d",
            label="Line 1D",
            kind="line_1d",
            supported_roles=("x", "y"),
        ),
        PlotViewType(
            id="scatter_2d",
            label="Scatter 2D",
            kind="scatter_2d",
            supported_roles=("x", "y", "color"),
        ),
        PlotViewType(
            id="trajectory_2d",
            label="Trajectory 2D",
            kind="trajectory_2d",
            supported_roles=("x", "y", "color"),
        ),
    )

    return PlotDataContract.from_items(
        source_id=str(source_id),
        label=str(label),
        dimensions=dimensions,
        quantities=quantities,
        view_types=view_types,
        default_view_type_id="line_1d",
    )


def default_position_plot_data_contract() -> PlotDataContract:
    """Return a generic position contract without profile-specific lengths."""

    return _position_contract(
        source_id="position:generic",
        label="Position data",
        axis_token="z",
        coordinate_mode="distance",
        frame_length=None,
        atom_length=None,
        entity_kind="atom",
    )


def position_profile_to_plot_data_contract(profile: PositionProfile) -> PlotDataContract:
    """Convert one loaded ``PositionProfile`` into a minimal ``PlotDataContract``."""

    axis_token = str(profile.axis).strip().lower() or "z"
    return _position_contract(
        source_id=f"position:{profile.species}:{axis_token}:{profile.coordinate_mode}",
        label=f"Position profile: {profile.species}",
        axis_token=axis_token,
        coordinate_mode=str(profile.coordinate_mode),
        frame_length=int(profile.n_frames),
        atom_length=int(profile.n_atoms),
        entity_kind=str(getattr(profile, "entity_kind", "atom")),
    )
