import json

import h5py
import pytest

from linak.plot.plot_settings import (
    copy_plot_profile,
    delete_named_plot_profile,
    duplicate_named_plot_profile,
    profile_name_conflict_message,
    read_active_plot_profile_name,
    read_plot_profile,
    read_plot_profile_names,
    rename_named_plot_profile,
    set_active_plot_profile,
    write_plot_profile,
)
from linak.plot.profile_persistence import (
    build_plot_profile_payload,
    flatten_plot_profile_payload,
    plot_profile_requires_legacy_mapping_flatten,
    select_plot_profile_settings,
)
from linak.plot.data_contract import PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP, PlotViewMapping
from linak.storage.hdf5_utils import write_linak_hdf5, write_linak_hdf5_profile_collection


def _saved_density_profile(style: dict[str, object]) -> dict[str, object]:
    return build_plot_profile_payload("plot:density", dict(style))


def _read_flat_density_profile(path, *, profile_name=None):
    payload = read_plot_profile(path, "plot:density", profile_name=profile_name)
    if payload is None:
        return None
    return flatten_plot_profile_payload("plot:density", payload)


def _write_density_hdf5(path):
    write_linak_hdf5(
        path,
        analysis="density",
        datasets={
            "bin_centers_A": [0.5],
            "density": [1.0],
        },
        metadata={
            "axis": "z",
            "species": "O",
        },
    )


def _write_combined_density_hdf5(path):
    write_linak_hdf5_profile_collection(
        path,
        analysis="density",
        profiles=[
            {
                "datasets": {
                    "bin_centers_A": [0.5],
                    "density": [1.0],
                },
                "metadata": {
                    "axis": "z",
                    "species": "O",
                    "profile_uid": "series-a",
                    "origin_hdf5_path": str(path.parent / "a_density.h5"),
                },
            },
            {
                "datasets": {
                    "bin_centers_A": [0.5],
                    "density": [2.0],
                },
                "metadata": {
                    "axis": "z",
                    "species": "H",
                    "profile_uid": "series-b",
                    "origin_hdf5_path": str(path.parent / "b_density.h5"),
                },
            },
        ],
    )


def test_named_plot_profiles_support_active_and_named_lookup(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Default title"}))
    write_plot_profile(
        source,
        "plot:density",
        _saved_density_profile({"title": "Publication title"}),
        profile_name="Publication",
    )

    assert read_plot_profile_names(source, "plot:density") == ["Default", "Publication"]
    assert read_active_plot_profile_name(source, "plot:density") == "Publication"
    assert _read_flat_density_profile(source) == {"title": "Publication title"}
    assert _read_flat_density_profile(source, profile_name="Default") == {
        "title": "Default title"
    }


def test_named_plot_profiles_set_active_and_delete_fallback_to_remaining_profile(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Default title"}))
    write_plot_profile(
        source,
        "plot:density",
        _saved_density_profile({"title": "Publication title"}),
        profile_name="Publication",
    )

    set_active_plot_profile(source, "plot:density", "Default")
    removed, active_after = delete_named_plot_profile(source, "plot:density", "Publication")

    assert removed is True
    assert active_after == "Default"
    assert read_plot_profile_names(source, "plot:density") == ["Default"]
    assert read_active_plot_profile_name(source, "plot:density") == "Default"


def test_copy_plot_profile_without_name_copies_entire_named_store(tmp_path):
    source = tmp_path / "source_density.h5"
    target = tmp_path / "target_density.h5"
    _write_density_hdf5(source)
    _write_density_hdf5(target)

    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Default title"}))
    write_plot_profile(
        source,
        "plot:density",
        _saved_density_profile({"title": "Publication title"}),
        profile_name="Publication",
    )

    copy_plot_profile(source, target, source_key="plot:density")

    assert read_plot_profile_names(target, "plot:density") == ["Default", "Publication"]
    assert read_active_plot_profile_name(target, "plot:density") == "Publication"
    assert _read_flat_density_profile(target, profile_name="Publication") == {
        "title": "Publication title"
    }


def test_duplicate_named_plot_profile_copies_payload_exactly(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)
    payload = _saved_density_profile(
        {
            "title": "Paper title",
            "x_lim": [0.0, 2.0],
            "y_lim": [None, 5.0],
            "_gui_sync_modes": {"x_lim": "manual", "y_lim": "manual"},
            "series_overrides": {"series:0": {"enabled": True}},
        }
    )
    write_plot_profile(source, "plot:density", payload, profile_name="Paper")

    duplicate_named_plot_profile(source, "plot:density", "Paper", "Paper Copy")

    assert read_plot_profile_names(source, "plot:density") == ["Paper", "Paper Copy"]
    assert read_active_plot_profile_name(source, "plot:density") == "Paper Copy"
    assert read_plot_profile(source, "plot:density", profile_name="Paper Copy") == read_plot_profile(
        source,
        "plot:density",
        profile_name="Paper",
    )


def test_combined_hdf5_plot_settings_support_named_profiles(tmp_path):
    source = tmp_path / "combined_density.h5"
    _write_combined_density_hdf5(source)

    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Default title"}))
    write_plot_profile(
        source,
        "plot:density",
        _saved_density_profile({"title": "Publication title"}),
        profile_name="Publication",
    )

    assert read_plot_profile_names(source, "plot:density") == ["Default", "Publication"]
    assert read_active_plot_profile_name(source, "plot:density") == "Publication"
    assert _read_flat_density_profile(source) == {"title": "Publication title"}
    assert _read_flat_density_profile(source, profile_name="Default") == {
        "title": "Default title"
    }
    assert _read_flat_density_profile(source, profile_name="Publication") == {
        "title": "Publication title"
    }


def test_combined_hdf5_plot_settings_allow_set_active_named_profile(tmp_path):
    source = tmp_path / "combined_density.h5"
    _write_combined_density_hdf5(source)
    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Saved"}))
    write_plot_profile(
        source,
        "plot:density",
        _saved_density_profile({"title": "Publication title"}),
        profile_name="Publication",
    )

    set_active_plot_profile(source, "plot:density", "Publication")

    assert read_active_plot_profile_name(source, "plot:density") == "Publication"


def test_combined_hdf5_plot_settings_allow_renaming_default_profile(tmp_path):
    source = tmp_path / "combined_density.h5"
    _write_combined_density_hdf5(source)
    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Default title"}))

    active_profile = rename_named_plot_profile(source, "plot:density", "Default", "My Profile")

    assert active_profile == "My Profile"
    assert read_plot_profile_names(source, "plot:density") == ["My Profile"]
    assert read_active_plot_profile_name(source, "plot:density") == "My Profile"
    assert _read_flat_density_profile(source, profile_name="My Profile") == {
        "title": "Default title"
    }


def test_rename_named_plot_profile_rejects_case_insensitive_conflicts(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)
    write_plot_profile(source, "plot:density", _saved_density_profile({"title": "Default title"}))
    write_plot_profile(
        source,
        "plot:density",
        _saved_density_profile({"title": "Publication title"}),
        profile_name="Publication",
    )

    with pytest.raises(ValueError, match="Choose a different name"):
        rename_named_plot_profile(source, "plot:density", "Default", "publication")

    assert profile_name_conflict_message("Publication") == (
        "Profile 'Publication' already exists. Choose a different name."
    )


def test_read_plot_profile_rejects_legacy_flat_saved_profile_format(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    legacy_payload = {
        "plot:density": {
            "__linak_named_plot_profiles__": 1,
            "active_profile": "Default",
            "profiles": {
                "Default": {
                    "title": "Legacy title",
                    "x_mode": "axis",
                }
            },
        }
    }
    with h5py.File(source, "r+") as handle:
        private = handle.require_group("_linak")
        settings = private.require_group("plot_settings")
        if "profiles_json" in settings:
            del settings["profiles_json"]
        settings.create_dataset(
            "profiles_json",
            data=json.dumps(legacy_payload),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with pytest.raises(ValueError, match="older LiNaK persistence format"):
        read_plot_profile(source, "plot:density")


def test_select_plot_profile_settings_reads_density_payload_without_flattening():
    payload = {
        "source_selection": {"species": "H2O", "axis": "y"},
        "view_mapping": {
            "view_type_id": "line_1d",
            "x": "axis_coordinate",
            "y": "number_density",
            "color": None,
            "split_by": None,
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"x_mode": "axis", "quantity": "number"},
        },
        "style": {"title": "Saved density", "legend": True},
    }

    selected = select_plot_profile_settings(
        "plot:density",
        payload,
        keys=("species", "axis", "view_mapping", "title"),
    )

    assert selected == {
        "species": "H2O",
        "axis": "y",
        "view_mapping": {
            **payload["view_mapping"],
            "view_type_id": PLOT_VIEW_1D_LINE,
        },
        "title": "Saved density",
    }


def test_write_plot_profile_round_trips_structured_density_payload_without_legacy_mapping_keys(
    tmp_path,
):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    settings = {
        "species": "H2O",
        "axis": "y",
        "x_mode": "distance",
        "quantity": "mass",
        "view_mapping": PlotViewMapping(
            view_type_id="line_1d",
            x="axis_coordinate",
            y="number_density",
            fixed_values={"x_mode": "axis"},
        ),
        "title": "Structured density",
        "legend": True,
    }

    write_plot_profile(
        source,
        "plot:density",
        build_plot_profile_payload("plot:density", settings),
        profile_name="Structured",
    )

    payload = read_plot_profile(source, "plot:density", profile_name="Structured")

    assert payload is not None
    assert payload["source_selection"] == {"species": "H2O", "axis": "y"}
    assert payload["view_mapping"]["view_type_id"] == PLOT_VIEW_1D_LINE
    assert payload["view_mapping"]["x"] == "axis_coordinate"
    assert payload["view_mapping"]["y"] == "number_density"
    assert payload["view_mapping"]["fixed_values"] == {"x_mode": "axis"}
    assert payload["style"] == {"title": "Structured density", "legend": True}


def test_select_plot_profile_settings_still_flattens_position_compatibility_keys():
    payload = {
        "source_selection": {"species": "O", "axis": None},
        "view_mapping": {
            "view_type_id": "trajectory_2d",
            "x": "x",
            "y": "z",
            "color": "distance_to_surface",
            "split_by": "atom",
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"projection_render_mode": "line-colors"},
        },
        "style": {},
    }

    selected = select_plot_profile_settings(
        "plot:position",
        payload,
        keys=("species", "axis", "view_mapping", "component"),
    )

    assert selected["species"] == "O"
    assert selected["component"] == "2d-projection"


def test_temperature_plot_profile_round_trips_time_axis_mapping():
    payload = build_plot_profile_payload(
        "plot:temperature",
        {"time_axis": "fs", "title": "Temperature", "legend": True},
    )

    assert payload["source_selection"] == {}
    assert payload["view_mapping"]["view_type_id"] == PLOT_VIEW_1D_LINE
    assert payload["view_mapping"]["x"] == "time_fs"
    assert payload["view_mapping"]["y"] == "temperature"
    assert payload["style"] == {"title": "Temperature", "legend": True}

    flattened = flatten_plot_profile_payload("plot:temperature", payload)

    assert flattened["time_axis"] == "fs"
    assert flattened["title"] == "Temperature"
    assert flattened["legend"] is True


def test_plot_profile_payload_canonicalizes_nested_view_state_ids():
    payload = build_plot_profile_payload(
        "plot:position",
        {
            "position_active_view_type": "trajectory_2d",
            "position_view_states": {
                "line_1d": {
                    "position_active_view_type": "line_1d",
                    "view_mapping": {
                        "view_type_id": "line_1d",
                        "x": "time_ps",
                        "y": "distance_to_surface",
                    },
                    "x_lim": [0.0, 1.0],
                },
                "trajectory_2d": {
                    "position_active_view_type": "trajectory_2d",
                    "view_mapping": {
                        "view_type_id": "trajectory_2d",
                        "x": "x",
                        "y": "z",
                        "color": "distance_to_surface",
                    },
                    "x_lim": [0.0, 2.0],
                },
            },
        },
    )

    style = payload["style"]
    assert style["position_active_view_type"] == PLOT_VIEW_2D_HEATMAP
    assert set(style["position_view_states"]) == {PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP}
    assert (
        style["position_view_states"][PLOT_VIEW_1D_LINE]["position_active_view_type"]
        == PLOT_VIEW_1D_LINE
    )
    assert (
        style["position_view_states"][PLOT_VIEW_2D_HEATMAP]["position_active_view_type"]
        == PLOT_VIEW_2D_HEATMAP
    )
    assert (
        style["position_view_states"][PLOT_VIEW_1D_LINE]["view_mapping"]["view_type_id"]
        == PLOT_VIEW_1D_LINE
    )
    assert (
        style["position_view_states"][PLOT_VIEW_2D_HEATMAP]["view_mapping"]["view_type_id"]
        == PLOT_VIEW_2D_HEATMAP
    )


def test_flatten_plot_profile_payload_canonicalizes_loaded_view_state_ids():
    payload = {
        "source_selection": {},
        "view_mapping": {
            "view_type_id": "trajectory_2d",
            "x": "x",
            "y": "z",
            "color": "distance_to_surface",
            "split_by": "atom",
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"projection_render_mode": "line-colors"},
        },
        "style": {
            "position_active_view_type": "trajectory_2d",
            "position_view_states": {
                "line_1d": {
                    "position_active_view_type": "line_1d",
                    "view_mapping": {
                        "view_type_id": "line_1d",
                        "x": "time_ps",
                        "y": "distance_to_surface",
                    },
                },
                "trajectory_2d": {
                    "position_active_view_type": "trajectory_2d",
                    "view_mapping": {
                        "view_type_id": "trajectory_2d",
                        "x": "x",
                        "y": "z",
                        "color": "distance_to_surface",
                    },
                },
            },
        },
    }

    flattened = flatten_plot_profile_payload("plot:position", payload)

    assert flattened["position_active_view_type"] == PLOT_VIEW_2D_HEATMAP
    assert set(flattened["position_view_states"]) == {PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP}
    assert (
        flattened["position_view_states"][PLOT_VIEW_1D_LINE]["position_active_view_type"]
        == PLOT_VIEW_1D_LINE
    )
    assert (
        flattened["position_view_states"][PLOT_VIEW_2D_HEATMAP]["position_active_view_type"]
        == PLOT_VIEW_2D_HEATMAP
    )
    assert (
        flattened["position_view_states"][PLOT_VIEW_1D_LINE]["view_mapping"]["view_type_id"]
        == PLOT_VIEW_1D_LINE
    )
    assert (
        flattened["position_view_states"][PLOT_VIEW_2D_HEATMAP]["view_mapping"]["view_type_id"]
        == PLOT_VIEW_2D_HEATMAP
    )


def test_plot_profile_requires_legacy_mapping_flatten_only_for_compatibility_keys():
    assert (
        plot_profile_requires_legacy_mapping_flatten(
            profile_key="plot:density",
            keys=("species", "axis", "view_mapping", "title"),
        )
        is False
    )
    assert (
        plot_profile_requires_legacy_mapping_flatten(
            profile_key="plot:position",
            keys=("species", "axis", "view_mapping", "component"),
        )
        is True
    )
