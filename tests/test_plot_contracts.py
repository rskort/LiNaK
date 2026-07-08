import numpy as np

from linak.analysis.density import DensityProfile
from linak.analysis.msd import MSDProfile
from linak.analysis.potential import PotentialPlotSeries
from linak.analysis.rdf import RDFProfile
from linak.analysis.coordination import CoordinationProfile
from linak.analysis.orientation import OrientationProfile, OrientationSparseGrid
from linak.analysis.position import PositionProfile
from linak.plot.contracts.density_contract import density_profile_to_plot_data_contract
from linak.plot.contracts.msd_contract import msd_profile_to_plot_data_contract
from linak.plot.contracts.potential_contract import potential_profiles_to_plot_data_contract
from linak.plot.contracts.rdf_contract import rdf_profile_to_plot_data_contract
from linak.plot.contracts.coordination_contract import coordination_profile_to_plot_data_contract
from linak.plot.contracts.orientation_contract import (
    orientation_heatmap_profile_to_plot_data_contract,
    orientation_line_profile_to_plot_data_contract,
)
from linak.plot.contracts.position_contract import position_profile_to_plot_data_contract
from linak.plot.data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PLOT_VIEW_LABEL_1D_LINE,
    PLOT_VIEW_LABEL_2D_HEATMAP,
    PlotDataContract,
    PlotDimension,
    PlotQuantity,
    PlotViewType,
    PlotViewMapping,
    canonical_plot_view_id,
    canonical_plot_view_id_from_label,
    plot_view_display_label,
)
from linak.plot.data_validation import generic_view_type_compatibility
from linak.plot.mappings.density_mapping import (
    density_plot_options_to_view_mapping,
    density_view_mapping_to_plot_options,
    resolve_density_plot_mapping,
)
from linak.plot.mappings.msd_mapping import (
    msd_plot_options_to_view_mapping,
    msd_view_mapping_to_plot_options,
    resolve_msd_plot_mapping,
)
from linak.plot.mappings.potential_mapping import (
    potential_plot_options_to_view_mapping,
    potential_view_mapping_to_plot_options,
    resolve_potential_plot_mapping,
)
from linak.plot.mappings.rdf_mapping import (
    rdf_plot_options_to_view_mapping,
    rdf_view_mapping_to_plot_options,
    resolve_rdf_plot_mapping,
)
from linak.plot.mappings.coordination_mapping import (
    coordination_mapping_preset,
    coordination_plot_options_to_view_mapping,
    coordination_view_mapping_to_plot_options,
    resolve_coordination_plot_mapping,
)
from linak.plot.mappings.orientation_mapping import (
    orientation_plot_options_to_view_mapping,
    orientation_view_mapping_to_plot_options,
    resolve_orientation_plot_mapping,
)
from linak.plot.mappings.position_mapping import (
    position_mapping_preset,
    position_plot_options_to_view_mapping,
    resolve_position_plot_mapping,
    position_view_mapping_to_plot_options,
)


def test_position_profile_to_plot_data_contract_exposes_frame_atom_structure():
    profile = PositionProfile(
        species="O",
        axis="z",
        atom_indices=np.array([0, 1]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 10.0, 20.0]),
        time_fs=np.array([0.0, 2.0, 4.0]),
        time_ps=np.array([0.0, 0.002, 0.004]),
        x=np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]]),
        y=np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        z=np.array([[1.0, 2.0], [1.2, 2.2], [1.4, 2.4]]),
        distance_to_surface=np.array([[0.8, 1.8], [0.9, 1.9], [1.0, 2.0]]),
        n_frames=3,
        n_atoms=2,
        coordinate_mode="distance",
        surface_position=0.2,
        surface_position_std=0.0,
        surface_position_per_frame=np.array([0.2, 0.3, 0.4]),
    )

    contract = position_profile_to_plot_data_contract(profile)

    assert contract.source_id == "position:O:z:distance"
    assert contract.label == "Position profile: O"
    assert contract.default_view_type_id == PLOT_VIEW_1D_LINE

    assert [dimension.id for dimension in contract.dimensions] == ["frame", "atom"]
    assert [dimension.length for dimension in contract.dimensions] == [3, 2]
    assert [dimension.kind for dimension in contract.dimensions] == [
        "time_index",
        "entity_index",
    ]

    quantities_by_id = {quantity.id: quantity for quantity in contract.quantities}
    assert set(quantities_by_id) == {
        "frame_index",
        "step",
        "time_fs",
        "time_ps",
        "atom_index",
        "x",
        "y",
        "z",
        "distance_to_surface",
    }
    assert quantities_by_id["time_ps"].dimensions == ("frame",)
    assert quantities_by_id["atom_index"].dimensions == ("atom",)
    assert quantities_by_id["x"].dimensions == ("frame", "atom")
    assert quantities_by_id["distance_to_surface"].dimensions == ("frame", "atom")
    assert quantities_by_id["distance_to_surface"].kind == "distance"
    assert quantities_by_id["distance_to_surface"].source_name == "distance_to_surface_A"

    assert [view.id for view in contract.view_types] == [
        PLOT_VIEW_1D_LINE,
        "scatter_2d",
        "trajectory_2d",
    ]
    assert [view.label for view in contract.view_types] == [
        PLOT_VIEW_LABEL_1D_LINE,
        PLOT_VIEW_LABEL_2D_HEATMAP,
        PLOT_VIEW_LABEL_2D_HEATMAP,
    ]
    assert contract.view_types[0].supported_roles == ("x", "y")
    assert contract.view_types[1].supported_roles == ("x", "y", "color")


def test_plot_view_display_labels_are_global_and_legacy_compatible():
    assert plot_view_display_label("line_1d") == PLOT_VIEW_LABEL_1D_LINE
    assert plot_view_display_label("heatmap_2d") == PLOT_VIEW_LABEL_2D_HEATMAP
    assert plot_view_display_label("trajectory_2d") == PLOT_VIEW_LABEL_2D_HEATMAP
    assert plot_view_display_label("scatter_2d") == PLOT_VIEW_LABEL_2D_HEATMAP

    assert canonical_plot_view_id("line_1d") == "plot_1d_line"
    assert canonical_plot_view_id("heatmap_2d") == "plot_2d_heatmap"
    assert canonical_plot_view_id("trajectory_2d") == "plot_2d_heatmap"

    assert canonical_plot_view_id_from_label("1D") == "plot_1d_line"
    assert canonical_plot_view_id_from_label("Line 1D") == "plot_1d_line"
    assert canonical_plot_view_id_from_label("1D Line") == "plot_1d_line"
    assert canonical_plot_view_id_from_label("2D") == "plot_2d_heatmap"
    assert canonical_plot_view_id_from_label("Heatmap 2D") == "plot_2d_heatmap"
    assert canonical_plot_view_id_from_label("2D Map") == "plot_2d_heatmap"
    assert canonical_plot_view_id_from_label("2D Heatmap") == "plot_2d_heatmap"


def test_canonical_view_tokens_are_accepted_at_mapping_boundaries():
    density_heatmap = density_plot_options_to_view_mapping(
        view_type="plot_2d_heatmap",
        quantity="number",
    )
    assert density_heatmap.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert density_view_mapping_to_plot_options(
        PlotViewMapping(
            view_type_id="plot_2d_heatmap",
            x="x_bin_center",
            y="y_bin_center",
            role_assignments={"z": "mass_density_2d"},
        )
    ) == {"view_type": "heatmap_2d", "quantity": "mass"}
    resolved_density = resolve_density_plot_mapping(mapping=density_heatmap)
    assert resolved_density.mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert resolved_density.compatibility == "valid_preferred"

    assert orientation_view_mapping_to_plot_options(
        PlotViewMapping(
            view_type_id="plot_2d_heatmap",
            x="bin_centers_A",
            y="heatmap_angle_bin_centers",
            role_assignments={"z": "heatmap_polar"},
        )
    ) == {
        "component": "heatmap",
        "angle": "polar",
        "orientation_heatmap_x_axis": "x",
        "orientation_heatmap_y_axis": "y",
    }
    resolved_orientation = resolve_orientation_plot_mapping(
        mapping=PlotViewMapping(
            view_type_id="plot_2d_heatmap",
            x="bin_centers_A",
            y="heatmap_angle_bin_centers",
            role_assignments={"z": "heatmap_polar"},
        )
    )
    assert resolved_orientation.compatibility == "valid_preferred"


def test_canonical_view_tokens_validate_against_canonical_only_contracts():
    line_contract = PlotDataContract.from_items(
        source_id="line",
        label="Line data",
        dimensions=(PlotDimension(id="bin", label="Bin", kind="bin", length=2),),
        quantities=(
            PlotQuantity(id="x", label="X", kind="coordinate", dimensions=("bin",)),
            PlotQuantity(id="y", label="Y", kind="value", dimensions=("bin",)),
        ),
        view_types=(
            PlotViewType(
                id=PLOT_VIEW_1D_LINE,
                label=PLOT_VIEW_LABEL_1D_LINE,
                kind="line_1d",
                supported_roles=("x", "y"),
            ),
        ),
        default_view_type_id=PLOT_VIEW_1D_LINE,
    )
    heatmap_contract = PlotDataContract.from_items(
        source_id="heatmap",
        label="Heatmap data",
        dimensions=(
            PlotDimension(id="x_bin", label="X bin", kind="bin", length=2),
            PlotDimension(id="y_bin", label="Y bin", kind="bin", length=3),
        ),
        quantities=(
            PlotQuantity(id="x", label="X", kind="coordinate", dimensions=("x_bin",)),
            PlotQuantity(id="y", label="Y", kind="coordinate", dimensions=("y_bin",)),
            PlotQuantity(id="z", label="Z", kind="value", dimensions=("x_bin", "y_bin")),
        ),
        view_types=(
            PlotViewType(
                id=PLOT_VIEW_2D_HEATMAP,
                label=PLOT_VIEW_LABEL_2D_HEATMAP,
                kind="heatmap_2d",
                supported_roles=("x", "y", "z"),
            ),
        ),
        default_view_type_id=PLOT_VIEW_2D_HEATMAP,
    )

    assert (
        generic_view_type_compatibility(
            line_contract,
            PlotViewMapping(view_type_id=PLOT_VIEW_1D_LINE, x="x", y="y"),
        )
        == "valid_preferred"
    )
    assert (
        generic_view_type_compatibility(
            heatmap_contract,
            PlotViewMapping(
                view_type_id=PLOT_VIEW_2D_HEATMAP,
                x="x",
                y="y",
                role_assignments={"z": "z"},
            ),
        )
        == "valid_preferred"
    )
    color_heatmap_contract = PlotDataContract.from_items(
        source_id="color-heatmap",
        label="Color heatmap data",
        dimensions=(
            PlotDimension(id="frame", label="Frame", kind="time_index", length=3),
            PlotDimension(id="atom", label="Atom", kind="entity_index", length=2),
        ),
        quantities=(
            PlotQuantity(id="time_ps", label="Time", kind="time", dimensions=("frame",)),
            PlotQuantity(
                id="distance",
                label="Distance",
                kind="distance",
                dimensions=("frame", "atom"),
            ),
            PlotQuantity(
                id="coordination",
                label="Coordination",
                kind="coordination",
                dimensions=("frame", "atom"),
            ),
        ),
        view_types=(
            PlotViewType(
                id=PLOT_VIEW_2D_HEATMAP,
                label=PLOT_VIEW_LABEL_2D_HEATMAP,
                kind=PLOT_VIEW_2D_HEATMAP,
                supported_roles=("x", "y", "color", "split_by"),
            ),
        ),
        default_view_type_id=PLOT_VIEW_2D_HEATMAP,
    )
    assert (
        generic_view_type_compatibility(
            color_heatmap_contract,
            PlotViewMapping(
                view_type_id=PLOT_VIEW_2D_HEATMAP,
                x="time_ps",
                y="distance",
                color="coordination",
                split_by="atom",
            ),
        )
        == "valid_nonpreferred"
    )

    assert rdf_view_mapping_to_plot_options(
        PlotViewMapping(view_type_id="plot_1d_line", x="radius", y="g_r")
    ) == {}
    assert msd_view_mapping_to_plot_options(
        PlotViewMapping(view_type_id="plot_1d_line", x="time_ps", y="msd")
    ) == {"time_axis": "ps"}
    assert coordination_view_mapping_to_plot_options(
        PlotViewMapping(
            view_type_id="plot_1d_line",
            x="distance_to_surface",
            y="coordination_number",
        )
    ) == {"component": "distance", "time_axis": "ps"}
    assert potential_view_mapping_to_plot_options(
        PlotViewMapping(view_type_id="plot_1d_line", x="record_id", y="efermi")
    )["view_type"] == "line_1d"


def test_position_mapping_preset_distance_vs_time_uses_generic_roles():
    mapping = position_mapping_preset("distance_vs_time", time_axis="ps")

    assert mapping.view_type_id == PLOT_VIEW_1D_LINE
    assert mapping.x == "time_ps"
    assert mapping.y == "distance_to_surface"
    assert mapping.split_by == "atom"
    assert mapping.color is None
    assert mapping.filter_by is None


def test_position_mapping_preset_x_z_trajectory_defaults_to_distance_color():
    mapping = position_mapping_preset("x_z_trajectory")

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.x == "x"
    assert mapping.y == "z"
    assert mapping.color == "distance_to_surface"
    assert mapping.split_by == "atom"
    assert mapping.fixed_values["projection_render_mode"] == "color-scale"


def test_position_plot_options_to_view_mapping_translates_legacy_1d_component():
    mapping = position_plot_options_to_view_mapping(component="distance", time_axis="fs")

    assert mapping.view_type_id == PLOT_VIEW_1D_LINE
    assert mapping.x == "time_fs"
    assert mapping.y == "distance_to_surface"
    assert mapping.split_by == "atom"
    assert mapping.fixed_values["legacy_component"] == "distance"


def test_position_plot_options_to_view_mapping_translates_projection_with_filter():
    mapping = position_plot_options_to_view_mapping(
        component="2d-projection",
        projection_x="x",
        projection_y="z",
        projection_value="distance",
        projection_render_mode="color-scale",
        projection_filter_max=2.5,
    )

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.x == "x"
    assert mapping.y == "z"
    assert mapping.color == "distance_to_surface"
    assert mapping.split_by == "atom"
    assert mapping.filter_by == "distance_to_surface"
    assert mapping.filter_min is None
    assert mapping.filter_max == 2.5
    assert mapping.fixed_values["projection_render_mode"] == "color-scale"


def test_position_plot_options_to_view_mapping_accepts_heatmap_alias():
    mapping = position_plot_options_to_view_mapping(
        component="heatmap",
        projection_x="x",
        projection_y="distance",
        projection_value="y",
        projection_render_mode="source-colors",
    )

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.x == "x"
    assert mapping.y == "distance_to_surface"
    assert mapping.color is None
    assert mapping.fixed_values["projection_render_mode"] == "line-colors"


def test_position_view_mapping_to_plot_options_round_trips_distance_vs_time():
    mapping = position_mapping_preset("distance_vs_time", time_axis="fs")

    options = position_view_mapping_to_plot_options(mapping)

    assert options["component"] == "distance"
    assert options["time_axis"] == "fs"
    assert options["projection_x"] == "x"
    assert options["projection_render_mode"] == "color-scale"


def test_position_view_mapping_to_plot_options_round_trips_trajectory_filter():
    mapping = position_plot_options_to_view_mapping(
        component="2d-projection",
        projection_x="x",
        projection_y="z",
        projection_value="distance",
        projection_render_mode="line-colors",
        projection_filter_max=3.5,
    )

    options = position_view_mapping_to_plot_options(mapping)

    assert options["component"] == "2d-projection"
    assert options["projection_x"] == "x"
    assert options["projection_y"] == "z"
    assert options["projection_value"] == "distance"
    assert options["projection_render_mode"] == "line-colors"
    assert options["projection_filter_max"] == 3.5
    assert options["xy_z_distance_max"] == 3.5


def test_resolve_position_plot_mapping_validates_and_marks_profile_descriptor_mode():
    profile = PositionProfile(
        species="O",
        axis="z",
        atom_indices=np.array([0, 1]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 10.0, 20.0]),
        time_fs=np.array([0.0, 2.0, 4.0]),
        time_ps=np.array([0.0, 0.002, 0.004]),
        x=np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]]),
        y=np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        z=np.array([[1.0, 2.0], [1.2, 2.2], [1.4, 2.4]]),
        distance_to_surface=np.array([[0.8, 1.8], [0.9, 1.9], [1.0, 2.0]]),
        n_frames=3,
        n_atoms=2,
        coordinate_mode="distance",
        surface_position=0.2,
        surface_position_std=0.0,
        surface_position_per_frame=np.array([0.2, 0.3, 0.4]),
    )

    resolved = resolve_position_plot_mapping(
        profile=profile,
        component="2d-projection",
        projection_x="x",
        projection_y="z",
        projection_value="distance",
        projection_render_mode="color-scale",
    )

    assert resolved.compatibility == "valid_preferred"
    assert resolved.mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert resolved.renderer_options["component"] == "2d-projection"
    assert resolved.renderer_options["projection_x"] == "x"
    assert resolved.uses_profile_descriptors is True


def test_coordination_profile_to_plot_data_contract_exposes_frame_atom_structure():
    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2, 3]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 10.0, 20.0]),
        time_fs=np.array([0.0, 2.0, 4.0]),
        time_ps=np.array([0.0, 0.002, 0.004]),
        distance_to_surface=np.array([[0.8, 1.2], [0.9, 1.3], [1.0, 1.4]]),
        coordination_number=np.array([[1.0, 0.5], [0.8, 0.4], [0.7, 0.3]]),
        n_frames=3,
        n_atoms=2,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    contract = coordination_profile_to_plot_data_contract(profile)

    assert contract.source_id == "coordination:O:H:z:distance"
    assert contract.label == "Coordination profile: O-H"
    assert [dimension.id for dimension in contract.dimensions] == ["frame", "atom"]
    assert [dimension.length for dimension in contract.dimensions] == [3, 2]

    quantities_by_id = {quantity.id: quantity for quantity in contract.quantities}
    assert set(quantities_by_id) == {
        "frame_index",
        "step",
        "time_fs",
        "time_ps",
        "atom_index",
        "distance_to_surface",
        "coordination_number",
    }
    assert quantities_by_id["time_ps"].dimensions == ("frame",)
    assert quantities_by_id["distance_to_surface"].dimensions == ("frame", "atom")
    assert quantities_by_id["coordination_number"].dimensions == ("frame", "atom")
    assert quantities_by_id["distance_to_surface"].source_name == "distance_to_surface_A"
    assert quantities_by_id["coordination_number"].source_name == "coordination_number"
    assert [(view.id, view.label, view.supported_roles) for view in contract.view_types] == [
        (PLOT_VIEW_1D_LINE, PLOT_VIEW_LABEL_1D_LINE, ("x", "y")),
        ("scatter_2d", PLOT_VIEW_LABEL_2D_HEATMAP, ("x", "y", "color")),
        ("trajectory_2d", PLOT_VIEW_LABEL_2D_HEATMAP, ("x", "y", "color")),
    ]


def test_coordination_mapping_preset_distance_vs_time_uses_generic_roles():
    mapping = coordination_mapping_preset("distance_vs_time", time_axis="fs")

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.x == "time_fs"
    assert mapping.y == "distance_to_surface"
    assert mapping.color == "coordination_number"
    assert mapping.split_by == "atom"


def test_coordination_plot_options_to_view_mapping_translates_time_distance():
    mapping = coordination_plot_options_to_view_mapping(
        component="time-distance",
        time_axis="frame",
    )

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.x == "frame_index"
    assert mapping.y == "distance_to_surface"
    assert mapping.color == "coordination_number"
    assert mapping.split_by == "atom"
    assert mapping.fixed_values == {}


def test_coordination_plot_options_to_view_mapping_accepts_heatmap_alias():
    mapping = coordination_plot_options_to_view_mapping(
        component="heatmap",
        time_axis="frame",
    )

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.x == "frame_index"
    assert mapping.y == "distance_to_surface"
    assert mapping.color == "coordination_number"


def test_coordination_view_mapping_to_plot_options_round_trips_time_mapping():
    mapping = coordination_mapping_preset("coordination_vs_time", time_axis="fs")

    options = coordination_view_mapping_to_plot_options(mapping)

    assert options["component"] == "time"
    assert options["time_axis"] == "fs"


def test_resolve_coordination_plot_mapping_validates_and_marks_atom_descriptor_mode():
    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2, 3]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 10.0, 20.0]),
        time_fs=np.array([0.0, 2.0, 4.0]),
        time_ps=np.array([0.0, 0.002, 0.004]),
        distance_to_surface=np.array([[0.8, 1.2], [0.9, 1.3], [1.0, 1.4]]),
        coordination_number=np.array([[1.0, 0.5], [0.8, 0.4], [0.7, 0.3]]),
        n_frames=3,
        n_atoms=2,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    resolved = resolve_coordination_plot_mapping(
        profile=profile,
        component="time-distance",
        time_axis="ps",
    )

    assert resolved.compatibility == "valid_nonpreferred"
    assert resolved.mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert resolved.renderer_options["component"] == "time-distance"
    assert resolved.renderer_options["time_axis"] == "ps"
    assert resolved.component == "time-distance"
    assert resolved.time_axis == "ps"
    assert resolved.uses_atom_descriptors is True


def test_density_profile_to_plot_data_contract_exposes_available_coordinates_and_quantities():
    profile = DensityProfile(
        axis="z",
        species="O",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        counts_per_frame=np.array([1.0, 1.0]),
        density=np.array([2.0, 3.0]),
        units="g/cm^3",
        n_frames=2,
        number_density=np.array([4.0, 5.0]),
        number_density_units="atom/nm^3",
        coordinate_mode="distance",
        surface_position=1.0,
    )

    contract = density_profile_to_plot_data_contract(profile)

    assert [dimension.id for dimension in contract.dimensions] == ["bin"]
    assert [(view.id, view.label) for view in contract.view_types] == [
        (PLOT_VIEW_1D_LINE, PLOT_VIEW_LABEL_1D_LINE)
    ]
    quantities_by_id = {quantity.id: quantity for quantity in contract.quantities}
    assert {"mass_density", "number_density", "axis_coordinate", "distance_to_surface"} <= set(
        quantities_by_id
    )
    assert quantities_by_id["distance_to_surface"].kind == "distance"
    assert quantities_by_id["axis_coordinate"].kind == "coordinate"


def test_density_view_mapping_round_trips_number_density_against_axis_coordinates():
    mapping = density_plot_options_to_view_mapping(x_mode="z", quantity="number")

    options = density_view_mapping_to_plot_options(mapping)

    assert mapping.fixed_values == {"x_mode": "z"}
    assert options == {"view_type": "line_1d", "x_mode": "z", "quantity": "number"}


def test_density_view_mapping_preserves_legacy_axis_mode_round_trip():
    mapping = density_plot_options_to_view_mapping(x_mode="axis", quantity="mass")

    options = density_view_mapping_to_plot_options(mapping)

    assert mapping.fixed_values == {"x_mode": "axis"}
    assert options == {"view_type": "line_1d", "x_mode": "axis", "quantity": "mass"}


def test_density_heatmap_view_mapping_round_trips_mass_density():
    mapping = density_plot_options_to_view_mapping(view_type="heatmap_2d", quantity="mass")

    options = density_view_mapping_to_plot_options(mapping)

    assert mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert mapping.fixed_values == {}
    assert mapping.resolved_role_assignments()["z"] == "mass_density_2d"
    assert options == {"view_type": "heatmap_2d", "quantity": "mass"}


def test_resolve_density_plot_mapping_validates_loaded_profile_contract():
    profile = DensityProfile(
        axis="z",
        species="O",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        counts_per_frame=np.array([1.0, 1.0]),
        density=np.array([2.0, 3.0]),
        units="g/cm^3",
        n_frames=2,
        coordinate_mode="axis",
    )

    resolved = resolve_density_plot_mapping(
        profile=profile,
        x_mode="z",
        quantity="mass",
    )

    assert resolved.compatibility == "valid_preferred"
    assert resolved.mapping.x == "axis_coordinate"
    assert resolved.renderer_options == {
        "view_type": "line_1d",
        "x_mode": "z",
        "quantity": "mass",
    }
    assert resolved.x_mode == "z"
    assert resolved.quantity == "mass"


def test_msd_profile_to_plot_data_contract_exposes_time_and_msd_quantities():
    profile = MSDProfile(
        species="O",
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        msd=np.array([0.0, 1.0]),
        n_frames=2,
    )

    contract = msd_profile_to_plot_data_contract(profile)

    assert [dimension.id for dimension in contract.dimensions] == ["frame"]
    assert [quantity.id for quantity in contract.quantities] == ["time_fs", "time_ps", "msd"]


def test_msd_view_mapping_round_trips_time_axis():
    mapping = msd_plot_options_to_view_mapping(time_axis="fs")

    options = msd_view_mapping_to_plot_options(mapping)

    assert options == {"time_axis": "fs"}


def test_resolve_msd_plot_mapping_validates_loaded_profile_contract():
    profile = MSDProfile(
        species="O",
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        msd=np.array([0.0, 1.0]),
        n_frames=2,
    )

    resolved = resolve_msd_plot_mapping(profile=profile, time_axis="ps")

    assert resolved.compatibility == "valid_preferred"
    assert resolved.mapping.x == "time_ps"
    assert resolved.renderer_options == {"time_axis": "ps"}


def test_rdf_profile_to_plot_data_contract_exposes_radius_and_distribution():
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 2.0]),
        n_frames=2,
    )

    contract = rdf_profile_to_plot_data_contract(profile)

    assert [dimension.id for dimension in contract.dimensions] == ["r_bin"]
    assert [quantity.id for quantity in contract.quantities] == ["radius", "g_r"]


def test_rdf_view_mapping_round_trips_default_line_mapping():
    mapping = rdf_plot_options_to_view_mapping()

    options = rdf_view_mapping_to_plot_options(mapping)

    assert options == {}


def test_resolve_rdf_plot_mapping_validates_loaded_profile_contract():
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 2.0]),
        n_frames=2,
    )

    resolved = resolve_rdf_plot_mapping(profile=profile)

    assert resolved.compatibility == "valid_preferred"
    assert resolved.mapping.x == "radius"
    assert resolved.mapping.y == "g_r"


def test_potential_profiles_to_plot_data_contract_is_line_only():
    profiles = [
        PotentialPlotSeries(
            series_id="water_bulk_potential_ev",
            default_label="Water bulk",
            x_values=np.array([1.0, 2.0]),
            y_values=np.array([2.0, 2.1]),
            source_path="potential.h5",
            total_rows=2,
            complete_rows=2,
            incomplete_rows=0,
        ),
        PotentialPlotSeries(
            series_id="efermi_ev",
            default_label="Fermi",
            x_values=np.array([1.0, 2.0]),
            y_values=np.array([1.0, 1.1]),
            source_path="potential.h5",
            total_rows=2,
            complete_rows=2,
            incomplete_rows=0,
        ),
        PotentialPlotSeries(
            series_id="electrode_cshe_ev",
            default_label="cSHE",
            x_values=np.array([1.0, 2.0]),
            y_values=np.array([0.2, 0.3]),
            source_path="potential.h5",
            total_rows=2,
            complete_rows=2,
            incomplete_rows=0,
        ),
    ]

    contract = potential_profiles_to_plot_data_contract(profiles)

    assert [dimension.id for dimension in contract.dimensions] == ["record"]
    assert [(view.id, view.label) for view in contract.view_types] == [
        (PLOT_VIEW_1D_LINE, PLOT_VIEW_LABEL_1D_LINE)
    ]


def test_potential_legacy_table_mapping_migrates_to_summary_line_mode():
    legacy_table_options = potential_view_mapping_to_plot_options(
        PlotViewMapping(view_type_id="table_records")
    )
    summary_mapping = potential_plot_options_to_view_mapping()
    summary_options = potential_view_mapping_to_plot_options(summary_mapping)

    assert legacy_table_options == {
        "view_type": "line_1d",
        "y_quantity": "water_bulk_potential",
        "standard_plot": "summary",
    }
    assert summary_options["view_type"] == "line_1d"
    assert summary_options["standard_plot"] == "summary"


def test_resolve_potential_plot_mapping_migrates_legacy_table_mapping_to_line():
    profiles = [
        PotentialPlotSeries(
            series_id="water_bulk_potential_ev",
            default_label="Water bulk",
            x_values=np.array([1.0, 2.0]),
            y_values=np.array([2.0, 2.1]),
            source_path="potential.h5",
            total_rows=2,
            complete_rows=2,
            incomplete_rows=0,
        ),
        PotentialPlotSeries(
            series_id="efermi_ev",
            default_label="Fermi",
            x_values=np.array([1.0, 2.0]),
            y_values=np.array([1.0, 1.1]),
            source_path="potential.h5",
            total_rows=2,
            complete_rows=2,
            incomplete_rows=0,
        ),
        PotentialPlotSeries(
            series_id="electrode_cshe_ev",
            default_label="cSHE",
            x_values=np.array([1.0, 2.0]),
            y_values=np.array([0.2, 0.3]),
            source_path="potential.h5",
            total_rows=2,
            complete_rows=2,
            incomplete_rows=0,
        ),
    ]

    resolved = resolve_potential_plot_mapping(
        profiles=profiles,
        mapping=PlotViewMapping(view_type_id="table_records"),
    )

    assert resolved.compatibility == "valid_nonpreferred"
    assert resolved.mapping.view_type_id == PLOT_VIEW_1D_LINE
    assert resolved.view_type == "line_1d"
    assert resolved.standard_plot == "summary"


def test_orientation_line_profile_to_plot_data_contract_exposes_distance_line_quantities():
    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=2,
        n_molecules_per_frame=1,
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        cos_polar_mean=np.array([0.1, 0.2]),
        cos_azimuthal_mean=np.array([0.3, 0.4]),
        count_total=np.array([1, 1]),
        count_polar_valid=np.array([1, 1]),
        count_azimuthal_valid=np.array([1, 1]),
        cos_polar_density=np.array([0.5, 0.6]),
        cos_azimuthal_density=np.array([0.7, 0.8]),
        density=np.array([0.9, 1.0]),
        heatmap_polar=np.array([[1.0, 0.0], [0.0, 1.0]]),
        heatmap_azimuthal=np.array([[0.0, 1.0], [1.0, 0.0]]),
        heatmap_angle_bin_edges=np.array([-1.0, 0.0, 1.0]),
        heatmap_angle_bin_centers=np.array([-0.5, 0.5]),
        coordinate_mode="distance",
    )

    contract = orientation_line_profile_to_plot_data_contract(profile)

    assert [dimension.id for dimension in contract.dimensions] == ["distance_bin"]
    assert [(view.id, view.label) for view in contract.view_types] == [
        (PLOT_VIEW_1D_LINE, PLOT_VIEW_LABEL_1D_LINE)
    ]
    assert [quantity.id for quantity in contract.quantities] == [
        "bin_centers_A",
        "cos_polar_mean",
        "cos_azimuthal_mean",
        "cos_polar_density",
        "cos_azimuthal_density",
        "density",
    ]


def test_orientation_heatmap_profile_to_plot_data_contract_exposes_distance_angle_grid():
    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=2,
        n_molecules_per_frame=1,
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        cos_polar_mean=np.array([0.1, 0.2]),
        cos_azimuthal_mean=np.array([0.3, 0.4]),
        count_total=np.array([1, 1]),
        count_polar_valid=np.array([1, 1]),
        count_azimuthal_valid=np.array([1, 1]),
        cos_polar_density=np.array([0.5, 0.6]),
        cos_azimuthal_density=np.array([0.7, 0.8]),
        density=np.array([0.9, 1.0]),
        heatmap_polar=np.array([[1.0, 0.0], [0.0, 1.0]]),
        heatmap_azimuthal=np.array([[0.0, 1.0], [1.0, 0.0]]),
        heatmap_angle_bin_edges=np.array([-1.0, 0.0, 1.0]),
        heatmap_angle_bin_centers=np.array([-0.5, 0.5]),
        coordinate_mode="distance",
    )

    contract = orientation_heatmap_profile_to_plot_data_contract(profile)

    assert [dimension.id for dimension in contract.dimensions] == ["distance_bin", "angle_bin"]
    assert [quantity.id for quantity in contract.quantities] == [
        "bin_centers_A",
        "heatmap_angle_bin_centers",
        "heatmap_polar",
        "heatmap_azimuthal",
        "density",
    ]
    assert contract.view_types[0].id == PLOT_VIEW_2D_HEATMAP
    assert contract.view_types[0].label == PLOT_VIEW_LABEL_2D_HEATMAP


def test_orientation_heatmap_profile_to_plot_data_contract_exposes_sparse_grid():
    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=2,
        n_molecules_per_frame=1,
        bin_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([0.5]),
        cos_polar_mean=np.array([1.0]),
        cos_azimuthal_mean=np.array([0.0]),
        count_total=np.array([2]),
        count_polar_valid=np.array([2]),
        count_azimuthal_valid=np.array([2]),
        cos_polar_density=np.array([2.0]),
        cos_azimuthal_density=np.array([0.0]),
        density=np.array([2.0]),
        heatmap_polar=np.array([[2.0, 0.0]]),
        heatmap_azimuthal=np.array([[0.0, 2.0]]),
        heatmap_angle_bin_edges=np.array([-1.0, 0.0, 1.0]),
        heatmap_angle_bin_centers=np.array([-0.5, 0.5]),
        coordinate_mode="distance",
        sparse_grid=OrientationSparseGrid(
            x_edges=np.array([0.0, 1.0]),
            y_edges=np.array([0.0, 1.0]),
            z_edges=np.array([0.0, 1.0]),
            distance_edges=np.array([0.0, 1.0]),
            shape=(1, 1, 1, 1),
            flat_indices=np.array([0], dtype=np.int64),
            entity_sum=np.array([2.0]),
            cos_polar_sum=np.array([2.0]),
            count_polar_valid=np.array([2.0]),
            cos_azimuthal_sum=np.array([0.0]),
            count_azimuthal_valid=np.array([2.0]),
        ),
    )

    contract = orientation_heatmap_profile_to_plot_data_contract(profile)

    dimensions = {dimension.id: dimension for dimension in contract.dimensions}
    assert dimensions["sparse_grid_cell"].length == 1
    quantities = {quantity.id: quantity for quantity in contract.quantities}
    assert quantities["grid_flat_indices"].dimensions == ("sparse_grid_cell",)
    assert quantities["grid_entity_sum"].kind == "orientation_grid_count"
    assert quantities["grid_cos_polar_sum"].kind == "orientation_grid_sum"


def test_orientation_view_mapping_round_trips_line_and_heatmap_modes():
    line_mapping = orientation_plot_options_to_view_mapping(
        component="density-weighted",
        angle="azimuthal",
    )
    assert line_mapping.fixed_values == {"orientation_line_x_axis": "distance"}
    line_options = orientation_view_mapping_to_plot_options(line_mapping)
    assert line_options == {
        "component": "density-weighted",
        "angle": "azimuthal",
        "orientation_line_x_axis": "distance",
    }

    z_line_mapping = orientation_plot_options_to_view_mapping(
        component="average",
        angle="polar",
        line_x_axis="z",
    )
    assert z_line_mapping.fixed_values == {"orientation_line_x_axis": "z"}
    assert orientation_view_mapping_to_plot_options(z_line_mapping) == {
        "component": "average",
        "angle": "polar",
        "orientation_line_x_axis": "z",
    }

    heatmap_mapping = orientation_plot_options_to_view_mapping(
        component="heatmap",
        angle="polar",
    )
    assert heatmap_mapping.fixed_values == {}
    heatmap_options = orientation_view_mapping_to_plot_options(heatmap_mapping)
    assert heatmap_options == {
        "component": "heatmap",
        "angle": "polar",
        "orientation_heatmap_x_axis": "x",
        "orientation_heatmap_y_axis": "y",
    }

    yz_heatmap_mapping = orientation_plot_options_to_view_mapping(
        component="heatmap",
        angle="azimuthal",
        heatmap_x_axis="y",
        heatmap_y_axis="z",
    )
    assert yz_heatmap_mapping.fixed_values == {
        "orientation_heatmap_x_axis": "y",
        "orientation_heatmap_y_axis": "z",
    }
    assert orientation_view_mapping_to_plot_options(yz_heatmap_mapping) == {
        "component": "heatmap",
        "angle": "azimuthal",
        "orientation_heatmap_x_axis": "y",
        "orientation_heatmap_y_axis": "z",
    }


def test_resolve_orientation_plot_mapping_validates_line_and_heatmap_contracts():
    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=2,
        n_molecules_per_frame=1,
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        cos_polar_mean=np.array([0.1, 0.2]),
        cos_azimuthal_mean=np.array([0.3, 0.4]),
        count_total=np.array([1, 1]),
        count_polar_valid=np.array([1, 1]),
        count_azimuthal_valid=np.array([1, 1]),
        cos_polar_density=np.array([0.5, 0.6]),
        cos_azimuthal_density=np.array([0.7, 0.8]),
        density=np.array([0.9, 1.0]),
        heatmap_polar=np.array([[1.0, 0.0], [0.0, 1.0]]),
        heatmap_azimuthal=np.array([[0.0, 1.0], [1.0, 0.0]]),
        heatmap_angle_bin_edges=np.array([-1.0, 0.0, 1.0]),
        heatmap_angle_bin_centers=np.array([-0.5, 0.5]),
        coordinate_mode="distance",
    )

    resolved_line = resolve_orientation_plot_mapping(
        profile=profile,
        component="average",
        angle="polar",
    )
    assert resolved_line.compatibility == "valid_preferred"
    assert resolved_line.mapping.view_type_id == PLOT_VIEW_1D_LINE
    assert resolved_line.renderer_options == {
        "component": "average",
        "angle": "polar",
        "orientation_line_x_axis": "distance",
    }
    assert resolved_line.component == "average"
    assert resolved_line.angle == "polar"

    resolved_heatmap = resolve_orientation_plot_mapping(
        profile=profile,
        component="heatmap",
        angle="azimuthal",
    )
    assert resolved_heatmap.compatibility == "valid_preferred"
    assert resolved_heatmap.mapping.view_type_id == PLOT_VIEW_2D_HEATMAP
    assert resolved_heatmap.renderer_options == {
        "component": "heatmap",
        "angle": "azimuthal",
        "orientation_heatmap_x_axis": "x",
        "orientation_heatmap_y_axis": "y",
    }
    assert resolved_heatmap.component == "heatmap"
    assert resolved_heatmap.angle == "azimuthal"
