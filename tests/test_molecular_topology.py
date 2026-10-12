from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from collections.abc import Generator, Sequence
from typing import cast

from ase import Atoms
from ase.constraints import FixAtoms
import h5py
import numpy as np
import pytest

from linak.analysis import topology, water
from linak.analysis.common import RAW_SPECIES_ARRAY
from linak.analysis.density import compute_density_profiles
from linak.analysis.position import compute_position_profiles
from linak.trajectory import io
from linak.trajectory.execution import (
    ChunkedTrajectorySequence,
    TrajectoryExecutionOptions,
    prepare_cli_trajectory,
)


def frames() -> list[Atoms]:
    result = []
    for x in (0.9, 3.0, 0.9, 0.9):
        frame = Atoms(
            "OHH", positions=[[5, 5, 5], [5.9, 5, 5], [5 - x, 5, 5]], cell=[10, 10, 10], pbc=True
        )
        frame.set_constraint(FixAtoms(indices=[0]))
        result.append(frame)
    return result


def assert_equal(actual, expected):
    for name in topology.WIDTHS:
        np.testing.assert_array_equal(actual.indices_for(name), expected.indices_for(name))
    for name in topology.DIAGNOSTICS:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def legacy(path: Path):
    io.write_trajectory(frames(), path)
    with h5py.File(path, "r+") as handle:
        del handle["molecular_topology"]


def count_detections(monkeypatch):
    calls = []
    original = water._oh_molecule_topology_with_backend

    def detect(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(water, "_oh_molecule_topology_with_backend", detect)
    return calls


def test_writer_persists_exact_deduplicated_membership_at_storage_precision(tmp_path):
    path = tmp_path / "exact.traj.h5"
    source = frames()
    source[0].positions[1, 0] = 6.27000001
    before = source[0].positions.copy()
    io.write_trajectory(source, path)
    np.testing.assert_array_equal(source[0].positions, before)
    loaded = io.read_trajectory(path)
    for actual in loaded:
        expected, _ = water._oh_molecule_topology_with_backend(
            actual, oh_cutoff=1.27, backend="auto"
        )
        assert_equal(water.oh_molecule_topology(actual), expected)
    with h5py.File(path) as handle:
        profiles = topology.read_profiles(handle, len(source))
        profile = next(iter(profiles.values()))
        assert len(profile.states) == 2
        assert profile.frame_state_index == [0, 1, 0, 0]
        assert handle["positions"].dtype == np.float32
    assert loaded[1].constraints[0].get_indices().tolist() == [0]


@pytest.mark.parametrize("mode", ["eager", "chunks", "sequence"])
def test_warm_profiles_skip_detection_including_copies_slices_and_multiple_passes(
    tmp_path, monkeypatch, mode
):
    path = tmp_path / "warm.traj.h5"
    io.write_trajectory(frames(), path)
    calls = count_detections(monkeypatch)
    loaded: Sequence[Atoms]
    if mode == "eager":
        loaded = io.read_trajectory(path)
    elif mode == "chunks":
        loaded = [
            frame for chunk in io.read_trajectory_chunks(path, chunk_size=1) for frame in chunk
        ]
    else:
        loaded = ChunkedTrajectorySequence([path], [4], chunk_frames=1)
    for _ in range(2):
        for frame in loaded:
            water.oh_molecule_topology(frame.copy())
        for frame in loaded[::2]:
            water.water_molecule_triplets(frame)
        compute_position_profiles(loaded, species="molecules", surface_mode="none")
        compute_density_profiles(loaded, species="molecules", axis="z", bin_width=1)
    assert not calls


def test_legacy_upgrade_uses_materialized_frames_preserves_original_datasets(tmp_path, monkeypatch):
    path = tmp_path / "old.traj.h5"
    legacy(path)
    with h5py.File(path) as handle:
        positions = handle["positions"][:]
        attributes = dict(handle.attrs)
        compression = handle["positions"].compression
    loaded = io.read_trajectory(path)
    calls = count_detections(monkeypatch)

    def no_read(*args, **kwargs):
        raise AssertionError("eager upgrade reread trajectory payload")

    monkeypatch.setattr(io, "_read_trajectory_hdf5_chunks", no_read)
    for frame in loaded:
        water.oh_molecule_topology(frame)
    assert len(calls) == len(loaded)
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["positions"][:], positions)
        assert dict(handle.attrs) == attributes
        assert handle["positions"].compression == compression
        assert "molecular_topology/oh" in handle
    assert sorted(item.name for item in tmp_path.iterdir()) == ["old.traj.h5"]


def test_streaming_legacy_upgrade_waits_for_reader_close(tmp_path, monkeypatch):
    path = tmp_path / "old.traj.h5"
    legacy(path)
    calls = count_detections(monkeypatch)
    reader = io.read_trajectory_chunks(path, chunk_size=1)
    frame = next(reader)[0]
    water.oh_molecule_topology(frame)
    assert len(calls) == 4
    with h5py.File(path) as handle:
        assert "molecular_topology" not in handle
    cast(Generator[list[Atoms], None, None], reader).close()
    with h5py.File(path) as handle:
        assert "molecular_topology/oh" in handle
    for item in ChunkedTrajectorySequence([path], [4], chunk_frames=1):
        water.oh_molecule_topology(item)
    assert len(calls) == 4


@pytest.mark.parametrize(
    "mutation", ["positions", "numbers", "cell", "pbc", "slice", "subprecision"]
)
def test_frame_mutation_recomputes_without_modifying_source(tmp_path, monkeypatch, mutation):
    path = tmp_path / "original.traj.h5"
    io.write_trajectory(frames(), path)
    loaded = io.read_trajectory(path)
    frame = loaded[0].copy()
    before = path.read_bytes()
    water.oh_molecule_topology(frame)
    calls = count_detections(monkeypatch)
    if mutation == "positions":
        frame.positions[1, 0] += 3
    elif mutation == "subprecision":
        frame.positions[1, 0] += 1e-10
    elif mutation == "numbers":
        frame.numbers[1] = 8
    elif mutation == "cell":
        frame.set_cell([3, 3, 3])
    elif mutation == "pbc":
        frame.pbc[:] = False
    else:
        frame = frame[:2]
    actual = water.oh_molecule_topology(frame)
    assert len(calls) == 1
    expected, _ = water._oh_molecule_topology_with_backend(frame, oh_cutoff=1.27, backend="auto")
    assert_equal(actual, expected)
    water.oh_molecule_topology(loaded[0])
    assert len(calls) == 2  # only the explicit live comparison added a second detection
    assert path.read_bytes() == before


def test_mutated_legacy_frames_never_publish_changed_membership(tmp_path, monkeypatch):
    path = tmp_path / "old.traj.h5"
    legacy(path)
    loaded = io.read_trajectory(path)
    before = path.read_bytes()
    loaded[0].positions[1, 0] += 3
    assert water.oh_molecule_topology(loaded[0]).indices_for("mol:H2O").size == 0
    assert path.read_bytes() == before
    # An unmodified frame triggers an upgrade from source, not the edited eager list.
    water.oh_molecule_topology(loaded[1])
    reloaded = io.read_trajectory(path)
    assert water.oh_molecule_topology(reloaded[0]).indices_for("mol:H2O").shape == (1, 3)


@pytest.mark.parametrize("failure", ["read_only", "active_lock", "replace", "write", "interrupt"])
def test_failed_publication_preserves_original_and_reuses_run_cache(
    tmp_path, monkeypatch, caplog, failure
):
    caplog.set_level("WARNING", logger="linak")
    path = tmp_path / "old.traj.h5"
    legacy(path)
    loaded = io.read_trajectory(path)
    before = path.read_bytes()
    lock = path.with_name(path.name + ".topology.lock")
    if failure == "read_only":
        path.chmod(stat.S_IREAD)
    elif failure == "active_lock":
        lock.write_text(json.dumps({"owner": topology.current_owner(), "pid": os.getpid()}))
    elif failure == "replace":

        def failed_replace(*args):
            raise PermissionError("busy source")

        monkeypatch.setattr(topology.os, "replace", failed_replace)
    else:

        def failed_write(*args):
            raise KeyboardInterrupt() if failure == "interrupt" else OSError("disk full")

        monkeypatch.setattr(topology, "write_profile", failed_write)
    calls = count_detections(monkeypatch)
    try:
        if failure == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                water.oh_molecule_topology(loaded[0])
        else:
            for frame in loaded:
                water.oh_molecule_topology(frame)
            assert len(calls) == len(loaded)
            assert caplog.text.count("Cannot save molecular topology") == 1
        assert path.read_bytes() == before
        assert not list(tmp_path.glob(".linak-topology-*"))
        assert lock.exists() == (failure == "active_lock")
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("invalid", ["version", "complete", "offsets", "indices", "geometry"])
def test_invalid_profiles_rebuild_safely(tmp_path, monkeypatch, invalid):
    path = tmp_path / "invalid.traj.h5"
    io.write_trajectory(frames(), path)
    with h5py.File(path, "r+") as handle:
        group = next(iter(handle["molecular_topology/oh"].values()))
        if invalid == "version":
            group.attrs["version"] = 999
        elif invalid == "complete":
            group.attrs["complete"] = False
        elif invalid == "offsets":
            group["mol:H_offsets"][0] = -1
        elif invalid == "indices":
            group["frame_state_index"][0] = 999
        else:
            handle["positions"][0, 1, 0] = 9
    calls = count_detections(monkeypatch)
    loaded = io.read_trajectory(path)
    for frame in loaded:
        actual = water.oh_molecule_topology(frame)
        expected, _ = water._oh_molecule_topology_with_backend(
            frame, oh_cutoff=1.27, backend="auto"
        )
        assert_equal(actual, expected)
    assert len(calls) == 8  # four-frame rebuild and four live comparisons


@pytest.mark.parametrize("backend", ["auto", "ckdtree", "ase"])
def test_cutoff_and_backend_profiles_are_separate_and_reused(tmp_path, monkeypatch, backend):
    path = tmp_path / "profiles.traj.h5"
    io.write_trajectory(frames(), path)
    loaded = io.read_trajectory(path)
    calls = count_detections(monkeypatch)
    water.oh_molecule_topology(loaded[0], oh_cutoff=0.8, backend=backend)
    assert len(calls) == 4
    for frame in io.read_trajectory(path):
        assert (
            water.oh_molecule_topology(frame, oh_cutoff=0.8, backend=backend)
            .indices_for("mol:H2O")
            .size
            == 0
        )
    assert len(calls) == 4
    with h5py.File(path) as handle:
        assert len(handle["molecular_topology/oh"]) == 2


@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True), (False, False, False)])
def test_triclinic_diagnostics_and_all_labels_match_live_detection(tmp_path, pbc):
    from tests.test_density import _mixed_oh_frame

    frame = _mixed_oh_frame()
    frame.set_cell([[30, 0, 0], [1, 30, 0], [2, 1, 30]])
    frame.set_pbc(pbc)
    path = tmp_path / "labels.traj.h5"
    io.write_trajectory([frame], path)
    loaded = io.read_trajectory(path)[0]
    live, _ = water._oh_molecule_topology_with_backend(loaded, oh_cutoff=1.27, backend="auto")
    assert_equal(water.oh_molecule_topology(loaded), live)


@pytest.mark.parametrize("mode", ["eager", "stream"])
def test_xyz_computes_topology_once_and_keeps_source_unchanged(tmp_path, monkeypatch, mode):
    path = tmp_path / "raw.xyz"
    io.write_trajectory(frames(), path)
    before = path.read_bytes()
    calls = count_detections(monkeypatch)
    reads = []
    original = io._iter_xyz_like_frames

    def read(*args, **kwargs):
        reads.append(1)
        yield from original(*args, **kwargs)

    monkeypatch.setattr(io, "_iter_xyz_like_frames", read)
    options = TrajectoryExecutionOptions(mode=mode, temporary_directory=tmp_path)
    with prepare_cli_trajectory(path, options=options) as (loaded, diagnostics):
        first = compute_position_profiles(loaded, species="molecules", surface_mode="none")
        compute_density_profiles(loaded, species="molecules", axis="z", bin_width=1)
        second = compute_position_profiles(loaded, species="molecules", surface_mode="none")
        assert len(calls) == 4
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.entity_counts_per_frame, b.entity_counts_per_frame)
    assert len(reads) == 1
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_cache_metadata_does_not_leak_into_ase_exports(tmp_path):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read

    frame = frames()[0]
    frame.calc = SinglePointCalculator(frame, energy=-12.3, forces=np.ones((3, 3)))
    water.oh_molecule_topology(frame)
    path = tmp_path / "out.xyz"
    io.write_trajectory([frame], path)
    assert topology.FRAME_KEY not in path.read_text()
    restored = read(path)
    assert restored.get_potential_energy() == pytest.approx(-12.3)
    np.testing.assert_array_equal(restored.get_forces(), np.ones((3, 3)))


def test_ambiguous_and_overcoordinated_diagnostics_are_persisted(tmp_path):
    frame = Atoms(
        "OOHOHHHH",
        positions=[
            [0, 0, 0],
            [2, 0, 0],
            [1, 0, 0],
            [8, 0, 0],
            [8.9, 0, 0],
            [7.1, 0, 0],
            [8, 0.9, 0],
            [8, -0.9, 0],
        ],
    )
    path = tmp_path / "diagnostics.traj.h5"
    io.write_trajectory([frame], path)
    actual = water.oh_molecule_topology(io.read_trajectory(path)[0])
    np.testing.assert_array_equal(actual.ambiguous_hydrogen_indices, [2])
    np.testing.assert_array_equal(actual.ambiguous_oxygen_indices, [0, 1])
    np.testing.assert_array_equal(actual.overcoordinated_oxygen_indices, [3])
    profile = topology.TopologyProfile(1.27, "auto")
    profile.append(frame, actual)
    changed = water.OHTopology(
        actual.groups, np.array([], dtype=int), np.array([], dtype=int), np.array([3])
    )
    profile.append(frame, changed)
    assert len(profile.states) == 2  # diagnostics cannot be merged away


def test_raw_aliases_and_changing_atom_order_remain_exact(tmp_path, monkeypatch):
    path = tmp_path / "aliases.xyz"
    path.write_text("2\nfirst\nOw 0 0 0\nHa 0.9 0 0\n2\nsecond\nHb 0.9 0 0\nOw 0 0 0\n")
    loaded = io.read_trajectory(path, atom_aliases={"Ow": "O", "Ha": "H", "Hb": "H"})
    calls = count_detections(monkeypatch)
    for _ in range(2):
        assert water.oh_molecule_topology(loaded[0]).indices_for("mol:OH").tolist() == [[0, 1]]
        assert water.oh_molecule_topology(loaded[1]).indices_for("mol:OH").tolist() == [[1, 0]]
    assert len(calls) == 2
    assert loaded[1].arrays[RAW_SPECIES_ARRAY].tolist() == ["Hb", "Ow"]


def test_variable_topology_writer_has_local_indices_and_exact_labels(tmp_path, monkeypatch):
    source = frames()
    source[1] = source[1][:2]
    source[2] = source[2][[1, 0, 2]]
    path = tmp_path / "filtered.traj.h5"
    io.write_trajectory(
        source, path, metadata=io.TrajectoryStoredMetadata(spatial_filter_metadata={})
    )
    loaded = io.read_trajectory(path)
    expected = [
        water._oh_molecule_topology_with_backend(frame, oh_cutoff=1.27, backend="auto")[0]
        for frame in loaded
    ]
    calls = count_detections(monkeypatch)
    for actual, live in zip(loaded, expected):
        assert_equal(water.oh_molecule_topology(actual), live)
    assert not calls
    assert loaded[2].get_chemical_symbols() == ["H", "O", "H"]


def test_old_float64_trajectory_uses_original_precision(tmp_path, monkeypatch):
    path = tmp_path / "double.traj.h5"
    legacy(path)
    with h5py.File(path, "r+") as handle:
        positions = handle["positions"][:].astype(np.float64)
        positions[0, 1, 0] = 6.27000001
        del handle["positions"]
        handle.create_dataset("positions", data=positions)
    loaded = io.read_trajectory(path)
    assert water.oh_molecule_topology(loaded[0]).indices_for("mol:H2O").size == 0
    calls = count_detections(monkeypatch)
    for frame in io.read_trajectory(path):
        water.oh_molecule_topology(frame)
    assert not calls
    with h5py.File(path) as handle:
        assert handle["positions"].dtype == np.float64


def test_source_change_during_build_does_not_publish(tmp_path, monkeypatch, caplog):
    caplog.set_level("WARNING", logger="linak")
    path = tmp_path / "changed.traj.h5"
    legacy(path)
    loaded = io.read_trajectory(path)
    detector = water._oh_molecule_topology_with_backend
    calls = 0

    def detect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            value = path.stat()
            os.utime(path, ns=(value.st_atime_ns, value.st_mtime_ns + 1_000_000))
        return detector(*args, **kwargs)

    monkeypatch.setattr(water, "_oh_molecule_topology_with_backend", detect)
    water.oh_molecule_topology(loaded[0])
    assert "source changed" in caplog.text
    with h5py.File(path) as handle:
        assert "molecular_topology" not in handle
    assert list(tmp_path.iterdir()) == [path]


def test_dead_owner_lock_can_be_recovered(tmp_path, monkeypatch):
    path = tmp_path / "old.traj.h5"
    legacy(path)
    lock = path.with_name(path.name + ".topology.lock")
    lock.write_text(json.dumps({"owner": topology.current_owner(), "pid": 12345}))
    monkeypatch.setattr(topology, "process_alive", lambda pid: False)
    water.oh_molecule_topology(io.read_trajectory(path)[0])
    assert not lock.exists()
    with h5py.File(path) as handle:
        assert "molecular_topology/oh" in handle


def test_streaming_cell_override_reuses_run_cache_without_upgrading_source(tmp_path, monkeypatch):
    path = tmp_path / "original.traj.h5"
    io.write_trajectory(frames(), path)
    original = path.read_bytes()
    sequence = ChunkedTrajectorySequence([path], [4], chunk_frames=1)
    sequence.set_cell_override([20, 20, 20])
    calls = count_detections(monkeypatch)
    for _ in range(2):
        for frame in sequence:
            water.oh_molecule_topology(frame)
    assert len(calls) == 4
    assert path.read_bytes() == original


def test_returned_membership_is_immutable():
    result = water.oh_molecule_topology(frames()[0])
    with pytest.raises(ValueError):
        result.indices_for("mol:H2O")[0, 0] = 99
    with pytest.raises(TypeError):
        cast(dict[str, np.ndarray], result.groups)["mol:H2O"] = np.array([[99, 0, 0]])


def test_writing_selection_builds_only_selected_frames_without_upgrading_source(
    tmp_path, monkeypatch
):
    path = tmp_path / "old.traj.h5"
    legacy(path)
    loaded = io.read_trajectory(path)
    before = path.read_bytes()
    calls = count_detections(monkeypatch)
    output = tmp_path / "selected.traj.h5"
    io.write_trajectory(loaded[::2], output)
    assert len(calls) == 2
    assert path.read_bytes() == before
    for frame in io.read_trajectory(output):
        water.oh_molecule_topology(frame)
    assert len(calls) == 2


def test_build_exception_closes_progress_and_reports_failed_stage(monkeypatch, caplog):
    closed = []
    original = topology.ProgressBar.close

    def close(bar):
        closed.append(bar.desc)
        original(bar)

    def fail(*args, **kwargs):
        raise RuntimeError("detector failed")

    monkeypatch.setattr(topology.ProgressBar, "close", close)
    monkeypatch.setattr(water, "_oh_molecule_topology_with_backend", fail)
    with pytest.raises(RuntimeError, match="detector failed"):
        topology.build_profile(frames())
    assert closed == ["Molecular topology"]
    assert "Build molecular topology failed after" in caplog.text


def test_orientation_and_molecular_coordination_reuse_stored_memberships(tmp_path, monkeypatch):
    from linak.analysis.orientation import compute_orientation_profile
    from linak.analysis.coordination import (
        CoordinationCutoffResolution,
        compute_coordination_profile,
    )

    frame = Atoms(
        "OHH", positions=[[5, 5, 5], [5.8, 5, 5.4], [4.2, 5, 5.4]], cell=[10, 10, 10], pbc=True
    )
    path = tmp_path / "consumers.traj.h5"
    io.write_trajectory([frame, frame.copy()], path)
    loaded = io.read_trajectory(path)
    calls = count_detections(monkeypatch)
    compute_orientation_profile(loaded)
    compute_coordination_profile(
        loaded,
        species_a="mol:H2O",
        species_b="O",
        surface_mode="none",
        surface_layout=None,
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=3, smoothing_width_A=0, mode="manual"
        ),
    )
    assert not calls


@pytest.mark.parametrize("extension", [".xyz", ".traj.h5"])
def test_real_density_cli_uses_shared_topology(tmp_path, monkeypatch, extension):
    from linak import cli

    path = tmp_path / ("input" + extension)
    io.write_trajectory(frames(), path)
    calls = count_detections(monkeypatch)
    assert (
        cli.main(
            [
                "compute",
                "density",
                str(path),
                "--species",
                "molecules",
                "--output",
                str(tmp_path / "result.density.h5"),
            ]
        )
        == 0
    )
    assert len(calls) == (4 if extension == ".xyz" else 0)
