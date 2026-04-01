from linak.plot.plot_settings import (
    copy_plot_profile,
    delete_named_plot_profile,
    read_active_plot_profile_name,
    read_plot_profile,
    read_plot_profile_names,
    set_active_plot_profile,
    write_plot_profile,
)
from linak.storage.hdf5_utils import write_linak_hdf5, write_linak_hdf5_profile_collection


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

    write_plot_profile(source, "plot:density", {"title": "Default title"})
    write_plot_profile(
        source,
        "plot:density",
        {"title": "Publication title"},
        profile_name="Publication",
    )

    assert read_plot_profile_names(source, "plot:density") == ["Default", "Publication"]
    assert read_active_plot_profile_name(source, "plot:density") == "Publication"
    assert read_plot_profile(source, "plot:density") == {"title": "Publication title"}
    assert read_plot_profile(source, "plot:density", profile_name="Default") == {
        "title": "Default title"
    }


def test_named_plot_profiles_set_active_and_delete_fallback_to_remaining_profile(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    write_plot_profile(source, "plot:density", {"title": "Default title"})
    write_plot_profile(
        source,
        "plot:density",
        {"title": "Publication title"},
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

    write_plot_profile(source, "plot:density", {"title": "Default title"})
    write_plot_profile(
        source,
        "plot:density",
        {"title": "Publication title"},
        profile_name="Publication",
    )

    copy_plot_profile(source, target, source_key="plot:density")

    assert read_plot_profile_names(target, "plot:density") == ["Default", "Publication"]
    assert read_active_plot_profile_name(target, "plot:density") == "Publication"
    assert read_plot_profile(target, "plot:density", profile_name="Publication") == {
        "title": "Publication title"
    }


def test_combined_hdf5_plot_settings_support_named_profiles(tmp_path):
    source = tmp_path / "combined_density.h5"
    _write_combined_density_hdf5(source)

    write_plot_profile(source, "plot:density", {"title": "Default title"})
    write_plot_profile(
        source,
        "plot:density",
        {"title": "Publication title"},
        profile_name="Publication",
    )

    assert read_plot_profile_names(source, "plot:density") == ["Default", "Publication"]
    assert read_active_plot_profile_name(source, "plot:density") == "Publication"
    assert read_plot_profile(source, "plot:density") == {"title": "Publication title"}
    assert read_plot_profile(source, "plot:density", profile_name="Default") == {
        "title": "Default title"
    }
    assert read_plot_profile(source, "plot:density", profile_name="Publication") == {
        "title": "Publication title"
    }


def test_combined_hdf5_plot_settings_allow_set_active_named_profile(tmp_path):
    source = tmp_path / "combined_density.h5"
    _write_combined_density_hdf5(source)
    write_plot_profile(source, "plot:density", {"title": "Saved"})
    write_plot_profile(
        source,
        "plot:density",
        {"title": "Publication title"},
        profile_name="Publication",
    )

    set_active_plot_profile(source, "plot:density", "Publication")

    assert read_active_plot_profile_name(source, "plot:density") == "Publication"
