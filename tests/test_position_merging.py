from __future__ import annotations

from dataclasses import replace
import json

import h5py
import numpy as np
import pytest
from ase import Atoms

from linak import cli
from linak.analysis import position as position_module
from linak.analysis.position import (
    compute_position_profile,
    load_position_profiles,
    save_position_profiles,
)
from linak.trajectory.io import TrajectoryStoredMetadata, write_trajectory


def _frames():
    frame = Atoms(
        ["Pt"] * 8 + ["O", "H", "H"],
        positions=[[x, y, z] for z in (1, 2) for x, y in ((1, 1), (1, 6), (6, 1), (6, 6))]
        + [[4, 4, 5], [4.9, 4, 5], [4, 4.9, 5]],
        cell=[10, 10, 15],
        pbc=True,
    )
    return [frame, frame.copy()]


def _profile(species, *, axis="z"):
    return compute_position_profile(
        _frames(), species=species, axis=axis, surface_mode="layered", surface_elements=["Pt"]
    )


@pytest.mark.parametrize("output_mode", ["default", "directory", "explicit"])
def test_cli_position_merges_successive_species_and_repeated_runs(tmp_path, output_mode):
    trajectory = tmp_path / "traj.traj.h5"
    write_trajectory(_frames(), trajectory, metadata=TrajectoryStoredMetadata(frame_timestep_fs=1))
    output = tmp_path / "LiNaK_outputs" / "traj.position.h5"
    options = []
    if output_mode == "directory":
        options = ["--output", str(tmp_path / "results") + "/"]
        output = tmp_path / "results" / "traj.position.h5"
    elif output_mode == "explicit":
        output = tmp_path / "custom.position.h5"
        options = ["--output", str(output)]
    base = ["--log-level", "ERROR", "compute", "position", str(trajectory)]
    for species in ("O", "H", "O"):
        assert cli.main(base + ["--species", species] + options) == 0
    profiles = load_position_profiles(output)
    assert [p.species for p in profiles] == ["O", "H"]
    assert list(output.parent.glob("*.position.h5")) == [output]
    assert not list(output.parent.glob("*.position_*.h5"))


def test_position_merge_keeps_existing_arrays_ids_plot_settings_and_molecules(tmp_path):
    output = tmp_path / "sample.position.h5"
    metadata = {"source_path": str(tmp_path / "source.traj.h5")}
    oxygen, hydrogen = _profile("O"), _profile("H")
    save_position_profiles([oxygen], output, additional_metadata=metadata, merge_existing=True)
    with h5py.File(output, "a") as handle:
        handle.require_group("_linak").create_dataset("session_marker", data=[42])
        uid = json.loads(handle.attrs["metadata_json"])["profile_uid"]
    save_position_profiles([hydrogen], output, additional_metadata=metadata, merge_existing=True)
    before = load_position_profiles(output)
    np.testing.assert_allclose(before[0].x, oxygen.x, atol=1e-6)
    changed = replace(oxygen, x=oxygen.x + 0.2)
    save_position_profiles(
        [changed, _profile("mol:H2O")], output, additional_metadata=metadata, merge_existing=True
    )
    after = load_position_profiles(output)
    assert [p.species for p in after] == ["O", "H", "mol:H2O"]
    np.testing.assert_allclose(after[0].x, changed.x, atol=1e-6)
    np.testing.assert_array_equal(after[1].x, before[1].x)
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle["_linak/session_marker"][:], [42])
        assert json.loads(handle["profiles/0000"].attrs["metadata_json"])["profile_uid"] == uid


@pytest.mark.parametrize("difference", ["source", "time", "cell", "surface", "corrupt"])
def test_position_merge_incompatible_output_numbers_before_first_dot(tmp_path, difference):
    output = tmp_path / "sample.test.position.hdf5"
    metadata = {"source_path": str(tmp_path / "source.traj.h5")}
    save_position_profiles([_profile("O")], output, additional_metadata=metadata)
    incoming = _profile("H")
    if difference == "source":
        metadata = {"source_path": str(tmp_path / "other.traj.h5")}
    elif difference == "time":
        incoming = replace(incoming, time_fs=incoming.time_fs + 1)
    elif difference == "cell":
        cell = np.array(incoming.cell_matrices_angstrom, copy=True)
        cell[:, 0, 0] += 1
        incoming = replace(incoming, cell_matrices_angstrom=cell)
    elif difference == "surface":
        incoming = compute_position_profile(_frames(), species="H", surface_mode="none")
    else:
        output.write_bytes(b"invalid HDF5")
    original = output.read_bytes()
    first = save_position_profiles(
        [incoming], output, additional_metadata=metadata, merge_existing=True
    )
    second = save_position_profiles(
        [incoming], output, additional_metadata=metadata, merge_existing=True
    )
    assert first.name == "sample_1.test.position.hdf5"
    assert second.name == "sample_2.test.position.hdf5"
    assert output.read_bytes() == original
    assert load_position_profiles(first)[0].species == "H"


def test_position_merge_write_failure_keeps_original_and_cleans_temporary_files(
    tmp_path, monkeypatch
):
    output = tmp_path / "sample.position.h5"
    save_position_profiles([_profile("O")], output)
    original = output.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("interrupted write")

    monkeypatch.setattr(position_module, "write_profile_collection", fail)
    with pytest.raises(OSError, match="interrupted write"):
        save_position_profiles([_profile("H")], output, merge_existing=True)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_cli_position_incompatible_timestep_uses_intact_compound_extension(tmp_path):
    trajectory = tmp_path / "traj.traj.h5"
    write_trajectory(_frames(), trajectory, metadata=TrajectoryStoredMetadata(frame_timestep_fs=1))
    base = ["--log-level", "ERROR", "compute", "position", str(trajectory)]
    assert cli.main(base + ["--species", "O", "--timestep-fs", "1"]) == 0
    assert cli.main(base + ["--species", "H", "--timestep-fs", "2"]) == 0
    output_directory = tmp_path / "LiNaK_outputs"
    assert {path.name for path in output_directory.iterdir()} == {
        "traj.position.h5",
        "traj_1.position.h5",
    }
    assert load_position_profiles(output_directory / "traj.position.h5")[0].species == "O"
    assert load_position_profiles(output_directory / "traj_1.position.h5")[0].species == "H"


def test_position_merge_preserves_distinct_axes(tmp_path):
    output = tmp_path / "sample.position.h5"
    save_position_profiles([_profile("O")], output)
    save_position_profiles([_profile("O", axis="x")], output, merge_existing=True)
    assert [(p.species, p.axis) for p in load_position_profiles(output)] == [("O", "z"), ("O", "x")]
    assert len(load_position_profiles(output, axis="x")) == 1
