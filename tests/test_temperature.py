from __future__ import annotations

import math
import logging
from pathlib import Path

import numpy as np
import pytest

import linak.analysis.temperature as temperature_module
from linak.analysis.temperature import (
    AMU_TO_KG,
    ATOMIC_VELOCITY_TO_M_PER_S,
    BOLTZMANN_J_PER_K,
    compute_temperature_profiles,
    load_temperature_profiles,
    save_temperature_profiles,
)
from linak.cli import main
from linak.gui.actions import ActionRegistry
from linak.gui.defaults import readiness_for_action
from linak.gui.detection import detect_project_item
from linak.storage.hdf5_utils import read_linak_hdf5_profiles


def _write_cp2k_input(path: Path) -> None:
    path.write_text(
        "&FORCE_EVAL\n"
        "  &SUBSYS\n"
        "    &KIND O\n"
        "    &END KIND\n"
        "    &KIND H\n"
        "    &END KIND\n"
        "    &KIND Au\n"
        "    &END KIND\n"
        "    &KIND K\n"
        "    &END KIND\n"
        "  &END SUBSYS\n"
        "&END FORCE_EVAL\n"
        "&MOTION\n"
        "  &CONSTRAINT\n"
        "    &FIXED_ATOMS\n"
        "      LIST 4\n"
        "    &END FIXED_ATOMS\n"
        "  &END CONSTRAINT\n"
        "  &MD\n"
        "    &THERMAL_REGION\n"
        "      &DEFINE_REGION\n"
        "        TEMPERATURE 320\n"
        "        LIST 3\n"
        "      &END DEFINE_REGION\n"
        "      &DEFINE_REGION\n"
        "        TEMPERATURE 320\n"
        "        LIST 1..2\n"
        "      &END DEFINE_REGION\n"
        "      &DEFINE_REGION\n"
        "        TEMPERATURE 320\n"
        "        LIST 5\n"
        "      &END DEFINE_REGION\n"
        "    &END THERMAL_REGION\n"
        "  &END MD\n"
        "&END MOTION\n",
        encoding="utf-8",
    )


def _write_velocity_xyz(path: Path, *, k_temperature: float = 320.0) -> None:
    # One K atom with only vx set gives exactly the target 3N kinetic temperature.
    k_mass_amu = 39.0983
    velocity_au = (
        math.sqrt(3.0 * BOLTZMANN_J_PER_K * k_temperature / (k_mass_amu * AMU_TO_KG))
        / ATOMIC_VELOCITY_TO_M_PER_S
    )
    path.write_text(
        "5\n"
        " i =        10, time =        5.000, E = -1.0\n"
        "O 0.0000000000 0.0000000000 0.0000000000\n"
        "H 0.0000000000 0.0000000000 0.0000000000\n"
        f"K {velocity_au:.12f} 0.0000000000 0.0000000000\n"
        "Au 0.0000000000 0.0000000000 0.0000000000\n"
        "Au 0.0000000000 0.0000000000 0.0000000000\n",
        encoding="utf-8",
    )


def _write_labeled_velocity_xyz(path: Path, labels: list[str]) -> None:
    rows = "\n".join(f"{label} 0.0000000000 0.0000000000 0.0000000000" for label in labels)
    path.write_text(
        f"{len(labels)}\n i =        10, time =        5.000, E = -1.0\n{rows}\n",
        encoding="utf-8",
    )


def test_temp_table_uses_sibling_velocity_element_order(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    _write_velocity_xyz(tmp_path / "run-vel-1.xyz")
    temp = tmp_path / "run-1.temp"
    temp.write_text("10 5.0 100.0 200.0 300.0 400.0\n", encoding="utf-8")

    profiles = compute_temperature_profiles(temp)

    assert [profile.default_label for profile in profiles] == ["O", "H", "K", "Au"]
    assert [float(profile.temperature_K[0]) for profile in profiles] == [100.0, 200.0, 300.0, 400.0]


def test_tregion_table_tracks_regions_and_atom_indices(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    _write_velocity_xyz(tmp_path / "run-vel-1.xyz")
    tregion = tmp_path / "run-1.tregion"
    tregion.write_text(
        "# Temperature per Region\n"
        "# Step Nr. Time[fs] Temp.[K] ....\n"
        "10 5.0 0.0 320.0 300.0 310.0\n",
        encoding="utf-8",
    )

    profiles = compute_temperature_profiles(tregion)

    assert [profile.default_label for profile in profiles] == [
        "Unassigned [Au1]",
        "Region 1 [K1]",
        "Region 2 [O1 H1]",
        "Region 3 [Au1]",
    ]
    assert profiles[0].region_name == "Unassigned"
    assert profiles[0].region_elements == ("Au",)
    assert profiles[0].region_formula == "Au1"
    assert profiles[0].cp2k_list_expression == "4"
    assert profiles[0].atom_indices is not None
    assert profiles[0].atom_indices.tolist() == [3]
    assert profiles[1].region_name == "Region 1"
    assert profiles[1].region_elements == ("K",)
    assert profiles[1].region_formula == "K1"
    assert profiles[2].cp2k_list_expression == "1..2"
    assert profiles[2].atom_indices is not None
    assert profiles[2].atom_indices.tolist() == [0, 1]
    assert profiles[2].region_elements == ("O", "H")
    assert profiles[2].region_formula == "O1 H1"


def test_tregion_table_keeps_empty_composition_without_symbol_source(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    tregion = tmp_path / "run-1.tregion"
    tregion.write_text("10 5.0 0.0 320.0 300.0 310.0\n", encoding="utf-8")

    profiles = compute_temperature_profiles(tregion)

    assert [profile.default_label for profile in profiles] == [
        "Unassigned",
        "Region 1",
        "Region 2",
        "Region 3",
    ]
    assert all(profile.region_elements == () for profile in profiles)
    assert all(profile.region_formula is None for profile in profiles)


def test_velocity_temperature_uses_atomic_velocity_units_and_regions(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    velocity = tmp_path / "run-vel-1.xyz"
    _write_velocity_xyz(velocity, k_temperature=320.0)

    profiles = compute_temperature_profiles(velocity, input_path=tmp_path / "input.inp")
    by_label = {profile.default_label: profile for profile in profiles}

    assert set(by_label) == {
        "O",
        "H",
        "K",
        "Au",
        "Unassigned [Au1]",
        "Region 1 [K1]",
        "Region 2 [O1 H1]",
        "Region 3 [Au1]",
    }
    assert np.isclose(by_label["K"].temperature_K[0], 320.0, rtol=0, atol=1.0e-5)
    assert np.isclose(by_label["Region 1 [K1]"].temperature_K[0], 320.0, rtol=0, atol=1.0e-5)
    assert by_label["Region 1 [K1]"].region_name == "Region 1"
    assert by_label["Region 1 [K1]"].region_elements == ("K",)
    assert by_label["Region 1 [K1]"].region_formula == "K1"
    assert by_label["Region 1 [K1]"].velocity_unit == "atomic"
    assert by_label["Region 1 [K1]"].dof_mode == "3N"


def test_velocity_temperature_reuses_cached_atomic_masses(monkeypatch, tmp_path: Path) -> None:
    velocity = tmp_path / "run-vel-1.xyz"
    _write_velocity_xyz(velocity)
    cached_mass_flags: list[bool] = []
    original = temperature_module._temperature_for_selection

    def _tracked_temperature_for_selection(**kwargs):
        cached_mass_flags.append(kwargs.get("selected_masses_kg") is not None)
        return original(**kwargs)

    monkeypatch.setattr(
        temperature_module,
        "_temperature_for_selection",
        _tracked_temperature_for_selection,
    )

    profiles = compute_temperature_profiles(velocity, group_by="elements")

    assert len(profiles) == 4
    assert cached_mass_flags == [True, True, True, True]
    assert all(profile.frame_index is profiles[0].frame_index for profile in profiles[1:])
    assert all(profile.time_fs is profiles[0].time_fs for profile in profiles[1:])


def test_velocity_temperature_infers_pt_top_as_pt_without_alias(tmp_path: Path) -> None:
    velocity = tmp_path / "run-vel-1.xyz"
    _write_labeled_velocity_xyz(velocity, ["Pt_top", "Pt"])

    profiles = compute_temperature_profiles(velocity, group_by="elements")

    assert [profile.default_label for profile in profiles] == ["Pt"]
    assert profiles[0].atom_count == 2
    assert profiles[0].atom_indices is not None
    assert profiles[0].atom_indices.tolist() == [0, 1]


def test_velocity_temperature_accepts_explicit_atom_aliases(tmp_path: Path) -> None:
    velocity = tmp_path / "run-vel-1.xyz"
    _write_labeled_velocity_xyz(velocity, ["Qq_top"])

    profiles = compute_temperature_profiles(
        velocity,
        group_by="elements",
        atom_aliases=["Qq_top=O"],
    )

    assert [profile.default_label for profile in profiles] == ["O"]
    assert profiles[0].atom_count == 1
    assert profiles[0].atom_indices is not None
    assert profiles[0].atom_indices.tolist() == [0]


def test_temperature_hdf5_round_trip_and_metadata(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    _write_velocity_xyz(tmp_path / "run-vel-1.xyz")
    tregion = tmp_path / "run-1.tregion"
    tregion.write_text("10 5.0 0.0 320.0 300.0 310.0\n", encoding="utf-8")
    profiles = compute_temperature_profiles(tregion)
    output = tmp_path / "temperature.temperature.h5"

    save_temperature_profiles(profiles, output, additional_metadata={"source_path": str(tregion)})
    loaded = load_temperature_profiles(output)
    payloads = read_linak_hdf5_profiles(output, expected_analysis="temperature")

    assert [profile.default_label for profile in loaded] == [
        "Unassigned [Au1]",
        "Region 1 [K1]",
        "Region 2 [O1 H1]",
        "Region 3 [Au1]",
    ]
    assert payloads[1][1]["selection_kind"] == "region"
    assert payloads[1][1]["region_name"] == "Region 1"
    assert payloads[1][1]["default_label"] == "Region 1 [K1]"
    assert payloads[1][1]["region_elements"] == ["K"]
    assert payloads[1][1]["region_formula"] == "K1"
    assert payloads[1][1]["target_temperature_K"] == 320.0
    assert payloads[1][0]["atom_indices"].tolist() == [2]
    assert loaded[1].region_name == "Region 1"
    assert loaded[1].region_elements == ("K",)
    assert loaded[1].region_formula == "K1"


def test_temperature_cli_compute_and_plot(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    _write_velocity_xyz(tmp_path / "run-vel-1.xyz")
    temp = tmp_path / "run-1.temp"
    temp.write_text("10 5.0 100.0 200.0 300.0 400.0\n", encoding="utf-8")
    output = tmp_path / "temperature.temperature.h5"
    plot = tmp_path / "temperature.png"

    assert main(["compute", "temperature", str(temp), "--output", str(output)]) == 0
    assert output.exists()
    assert main(["plot", "--no-gui", "--no-show", "--output", str(plot), str(output)]) == 0
    assert plot.exists()


def test_temperature_cli_default_output_uses_clean_dot_name(tmp_path: Path) -> None:
    _write_cp2k_input(tmp_path / "input.inp")
    _write_velocity_xyz(tmp_path / "Pt110_1x2_Na4_H-vel-1.xyz")

    assert main(["compute", "temperature", str(tmp_path / "Pt110_1x2_Na4_H-vel-1.xyz")]) == 0

    assert (tmp_path / "LiNaK_outputs" / "Pt110_1x2_Na4_H.temperature.h5").exists()


def test_temperature_cli_velocity_accepts_pt_top_labels(tmp_path: Path) -> None:
    velocity = tmp_path / "run-vel-1.xyz"
    output = tmp_path / "temperature.temperature.h5"
    _write_labeled_velocity_xyz(velocity, ["Pt_top", "Pt"])

    assert (
        main(
            [
                "compute",
                "temperature",
                str(velocity),
                "--group-by",
                "elements",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists()


def test_gui_detects_temperature_sources_and_velocity_readiness(tmp_path: Path) -> None:
    temp = tmp_path / "run-1.temp"
    temp.write_text("10 5.0 100.0\n", encoding="utf-8")
    velocity = tmp_path / "run-vel-1.xyz"
    _write_velocity_xyz(velocity)
    position = tmp_path / "run-pos-1.xyz"
    position.write_text("1\ncomment\nO 0 0 0\n", encoding="utf-8")

    registry = ActionRegistry()
    action = registry.by_id("temperature")
    temp_item = detect_project_item(temp, origin="external")
    velocity_item = detect_project_item(velocity, origin="external")
    position_item = detect_project_item(position, origin="external")

    assert temp_item.item_type == "temperature_file"
    assert action.supports(temp_item)
    assert readiness_for_action(action, velocity_item).available
    assert not readiness_for_action(action, position_item).available


@pytest.mark.parametrize(
    "comment", ["# layers 99..999", "! layers 99..999", "# &END DEFINE_REGION ! note"]
)
def test_temperature_input_comments_preserve_lists_and_results(tmp_path, caplog, comment):
    source = tmp_path / "run-vel-1.xyz"
    _write_velocity_xyz(source)
    input_path = tmp_path / "input.inp"
    _write_cp2k_input(input_path)
    baseline = compute_temperature_profiles(source)
    contents = input_path.read_text(encoding="utf-8")
    input_path.write_text(
        "# &DEFINE_REGION\n" + "\n".join(line + " " + comment for line in contents.splitlines()),
        encoding="utf-8",
    )
    caplog.clear()
    actual = compute_temperature_profiles(source)
    assert [p.default_label for p in actual] == [p.default_label for p in baseline]
    for expected, result in zip(baseline, actual):
        np.testing.assert_array_equal(result.atom_indices, expected.atom_indices)
        np.testing.assert_array_equal(result.temperature_K, expected.temperature_K)
        np.testing.assert_array_equal(result.time_fs, expected.time_fs)
    assert "unsupported CP2K atom-list token" not in caplog.text
    assert temperature_module._parse_cp2k_list_expression("1..3, 5 # layers 99") == (0, 1, 2, 4)


@pytest.mark.parametrize("suffix", ["temp", "tregion"])
def test_temperature_tables_accept_trailing_hash_comments(tmp_path, suffix, caplog):
    _write_cp2k_input(tmp_path / "input.inp")
    source = tmp_path / f"run.{suffix}"
    source.write_text("# headers\n10 5.0 100 200 300 400 # layers 1 2 3\n", encoding="utf-8")
    profiles = compute_temperature_profiles(source)
    assert [p.temperature_K.tolist() for p in profiles] == [[100], [200], [300], [400]]
    assert "non-numeric" not in caplog.text


@pytest.mark.parametrize("group", ["auto", "both", "elements", "regions"])
@pytest.mark.parametrize("input_kind", ["missing", "empty", "unreadable"])
def test_missing_temperature_regions_warn_once_at_compute_boundary(
    tmp_path, monkeypatch, caplog, group, input_kind
):
    source = tmp_path / "run-vel-1.xyz"
    _write_velocity_xyz(source)
    input_path = tmp_path / "input.inp"
    if input_kind != "missing":
        input_path.write_text("&KIND K\n&END KIND\n", encoding="utf-8")
    if input_kind == "unreadable":
        original = temperature_module._parse_cp2k_input

        def unreadable(path):
            if Path(path) == input_path:
                raise PermissionError("test input is unreadable")
            return original(path)

        monkeypatch.setattr(temperature_module, "_parse_cp2k_input", unreadable)
    metadata = temperature_module.discover_temperature_metadata(source, input_path=input_path)
    assert not caplog.records  # Discovery also runs while browsing GUI sources.
    if group == "regions":
        with pytest.raises(ValueError, match="No usable temperature regions"):
            compute_temperature_profiles(source, input_path=input_path, group_by=group)
    else:
        profiles = compute_temperature_profiles(source, input_path=input_path, group_by=group)
        assert profiles and all(p.selection_kind == "element" for p in profiles)
    records = [r for r in caplog.records if r.name == temperature_module.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "--input" in records[0].message
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=temperature_module.__name__):
        temperature_module.compute_temperature_from_velocity_xyz(source, metadata=metadata)
    assert not caplog.records


@pytest.mark.parametrize("suffix", ["temp", "tregion"])
def test_temperature_table_missing_metadata_has_one_actionable_warning(tmp_path, caplog, suffix):
    source = tmp_path / f"run.{suffix}"
    source.write_text("10 5.0 100\n", encoding="utf-8")
    profiles = compute_temperature_profiles(source)
    assert profiles[0].temperature_K.tolist() == [100]
    records = [r for r in caplog.records if r.name == temperature_module.__name__]
    assert len(records) == 1 and "--input" in records[0].message
