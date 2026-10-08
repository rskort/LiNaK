from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, cast

from ase import Atoms
import h5py
import numpy as np
import pytest

from linak.analysis.binning import (
    resolve_uniform_bin_width_for_load,
    uniform_bin_width_from_edges,
)
from linak.analysis.density import (
    DensityProfile,
    compute_all_density_profiles,
    load_density_profile,
)
from linak.analysis.coordination import (
    CoordinationCutoffResolution,
    compute_coordination_profiles,
)
from linak.analysis.position import compute_position_profiles
from linak.analysis.water import oh_molecule_topology
from linak.cache import ByteAwareLRU, estimate_object_bytes
from linak.runtime_resources import GIB, MIB, resolve_resource_policy, resolve_worker_count
from linak.storage.hdf5_utils import (
    read_linak_hdf5_profile_headers,
    read_linak_hdf5_profiles_by_index,
    write_linak_hdf5,
)
from linak.storage.policies import storage_array, target_chunk_shape
from linak.trajectory.execution import (
    TrajectoryExecutionOptions,
    cleanup_stale_spools,
    prepare_cli_trajectory,
)
from linak.trajectory.io import write_trajectory
from linak.trajectory.spatial_filter import (
    SpatialFilterOptions,
    apply_spatial_filter,
    apply_spatial_filter_streaming,
)


def _periodic_frames(count: int = 6) -> list[Atoms]:
    frames: list[Atoms] = []
    for index in range(count):
        frame = Atoms(
            "ArAr",
            positions=[[1.0 + 0.1 * index, 1.0, 1.0], [4.0, 4.0, 4.0]],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        )
        frame.info["step"] = index
        frames.append(frame)
    return frames


def test_storage_policy_records_precision_compression_and_target_chunks(tmp_path):
    path = tmp_path / "policy.h5"
    write_linak_hdf5(
        path,
        analysis="density",
        datasets={
            "density": np.linspace(0.0, 1.0, 1000, dtype=np.float64),
            "cell_matrices_angstrom": np.eye(3, dtype=np.float64)[None, ...],
        },
        compression_profile="fast",
    )
    with h5py.File(path, "r") as handle:
        assert handle.attrs["compression_profile"] == "fast"
        assert handle.attrs["storage_precision_policy"] == "mixed-v1"
        assert handle["density"].dtype == np.dtype("float32")
        assert handle["cell_matrices_angstrom"].dtype == np.dtype("float64")
        assert handle["density"].compression == "lzf"
        assert handle["density"].shuffle
        assert np.prod(handle["density"].chunks) * handle["density"].dtype.itemsize <= MIB

    assert (
        storage_array(analysis="potential", name="potential_ev", values=[1.0]).dtype == np.float64
    )
    chunk_shape = target_chunk_shape((10_000, 100, 3), itemsize=4)
    assert chunk_shape is not None
    assert chunk_shape[0] <= 256


def test_mixed_v1_float32_bins_allow_scale_dependent_rounding(tmp_path):
    centers = (0.025 + np.arange(1100, dtype=np.float64) * 0.05).astype(np.float32)
    edges = (np.arange(1101, dtype=np.float64) * 0.05).astype(np.float32)

    assert (
        resolve_uniform_bin_width_for_load(
            metadata={"bin_width_A": 0.05},
            bin_centers=centers,
            source_path=tmp_path / "density.h5",
            analysis_name="Density",
        )
        == 0.05
    )
    assert uniform_bin_width_from_edges(edges, source_label="density bins") == pytest.approx(
        0.05, abs=5.0e-6
    )

    malformed = centers.copy()
    malformed[800] += np.float32(0.01)
    with pytest.raises(ValueError, match="inconsistent bin_centers_A"):
        resolve_uniform_bin_width_for_load(
            metadata={"bin_width_A": 0.05},
            bin_centers=malformed,
            source_path=tmp_path / "density.h5",
            analysis_name="Density",
        )


def test_density_loader_preserves_stored_precision_for_bin_validation(tmp_path):
    path = tmp_path / "large_coordinate_density.h5"
    centers = 0.025 + np.arange(1100, dtype=np.float64) * 0.05
    write_linak_hdf5(
        path,
        analysis="density",
        datasets={
            "bin_centers_A": centers,
            "density": np.ones(centers.shape, dtype=float),
            "counts_per_frame": np.ones(centers.shape, dtype=float),
        },
        metadata={
            "axis": "z",
            "species": "O",
            "bin_width_A": 0.05,
            "units": "g/cm^3",
            "n_frames": 1,
            "coordinate_mode": "axis",
        },
    )

    loaded = load_density_profile(path)

    assert loaded.bin_centers.size == centers.size
    assert np.diff(loaded.bin_edges) == pytest.approx(0.05)


def test_header_inventory_and_selective_payload_are_payload_free_then_selective(tmp_path):
    path = tmp_path / "inventory.h5"
    write_linak_hdf5(
        path,
        analysis="rdf",
        datasets={"bin_centers_A": np.arange(8.0), "g_r": np.ones(8), "unused": np.ones(64)},
    )
    headers = read_linak_hdf5_profile_headers(path, expected_analysis="rdf")
    inventory = headers[0]["dataset_inventory"]
    assert inventory["unused"]["shape"] == [64]
    assert inventory["unused"]["byte_estimate"] == 64 * 4
    payload = read_linak_hdf5_profiles_by_index(
        path,
        [0],
        expected_analysis="rdf",
        dataset_names=("g_r",),
    )[0][0]
    assert set(payload) == {"g_r"}


def test_resource_policy_and_workers_respect_automatic_memory_limit():
    policy = resolve_resource_policy(available_memory_bytes=10 * GIB, environ={})
    assert policy.source == "automatic"
    assert policy.memory_budget_bytes == int(0.55 * 10 * GIB)
    assert (
        resolve_worker_count(
            task_count=20,
            estimated_bytes_per_worker=2 * GIB,
            requested_threads=None,
            policy=policy,
            cpu_count=16,
        )
        == 2
    )
    explicit = resolve_resource_policy(128.0, available_memory_bytes=10 * GIB, environ={})
    assert explicit.memory_budget_bytes == 128 * MIB


def test_byte_aware_lru_counts_shared_buffers_and_evicts_to_budget():
    cache = ByteAwareLRU(budget_bytes=80_000)
    owner = np.ones(10_000, dtype=np.float32)
    shared = owner[::2]
    assert estimate_object_bytes([owner, shared]) < owner.nbytes * 2
    cache.put("profiles", "a", owner)
    cache.put("profiles", "b", np.ones(10_000, dtype=np.float32))
    assert cache.current_bytes <= cache.budget_bytes
    assert cache.stats()["evictions"] >= 1
    released = cache.clear()
    assert released >= 0
    assert cache.current_bytes == 0


def test_ckdtree_topology_matches_ase_across_periodic_boundary():
    frame = Atoms(
        "OHH",
        positions=[[0.1, 2.0, 2.0], [7.85, 2.0, 2.0], [0.95, 2.0, 2.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    reference = oh_molecule_topology(frame, oh_cutoff=1.27, backend="ase")
    accelerated = oh_molecule_topology(frame, oh_cutoff=1.27, backend="ckdtree")
    assert accelerated.signature() == reference.signature()
    np.testing.assert_array_equal(
        accelerated.ambiguous_hydrogen_indices,
        reference.ambiguous_hydrogen_indices,
    )


def test_ckdtree_images_matches_ase_for_triclinic_mixed_pbc_and_ambiguity():
    cell = np.asarray([[8.0, 0.0, 0.0], [1.5, 7.0, 0.0], [0.5, 0.3, 9.0]])
    boundary = Atoms(
        "OHH",
        scaled_positions=[[0.02, 0.3, 0.4], [0.98, 0.3, 0.4], [0.12, 0.3, 0.4]],
        cell=cell,
        pbc=(True, True, False),
    )
    ambiguous = Atoms(
        "OOH",
        positions=[[2.0, 2.0, 2.0], [3.6, 2.0, 2.0], [2.8, 2.0, 2.0]],
        cell=cell,
        pbc=(True, True, False),
    )
    for frame in (boundary, ambiguous):
        reference = oh_molecule_topology(frame, oh_cutoff=1.27, backend="ase")
        accelerated = oh_molecule_topology(frame, oh_cutoff=1.27, backend="ckdtree")
        assert accelerated.signature() == reference.signature()
        np.testing.assert_array_equal(
            accelerated.ambiguous_hydrogen_indices,
            reference.ambiguous_hydrogen_indices,
        )


def test_hdf5_stream_sequence_matches_eager_density(tmp_path):
    source = tmp_path / "source.traj.h5"
    frames = _periodic_frames()
    write_trajectory(frames, source)
    eager = compute_all_density_profiles(
        frames,
        species="Ar",
        surface_mode="none",
        outputs="1d",
        bin_width=0.5,
    )
    with prepare_cli_trajectory(
        source,
        options=TrajectoryExecutionOptions(mode="stream", chunk_target_mib=1.0),
    ) as (streamed, diagnostics):
        actual = compute_all_density_profiles(
            streamed,
            species="Ar",
            surface_mode="none",
            outputs="1d",
            bin_width=0.5,
        )
        assert diagnostics.resolved_mode == "stream"
        assert not isinstance(streamed, list)
    eager_profiles = [profile for profile in eager if isinstance(profile, DensityProfile)]
    actual_profiles = [profile for profile in actual if isinstance(profile, DensityProfile)]
    assert len(actual_profiles) == len(eager_profiles)
    for left, right in zip(eager_profiles, actual_profiles):
        np.testing.assert_allclose(right.bin_centers, left.bin_centers)
        np.testing.assert_allclose(right.density, left.density, equal_nan=True)


def test_hdf5_stream_sequence_matches_eager_position_and_coordination(tmp_path):
    source = tmp_path / "source.traj.h5"
    frames = _periodic_frames()
    write_trajectory(frames, source)
    eager_position = compute_position_profiles(
        frames,
        species="Ar",
        surface_mode="none",
        timestep_fs=2.0,
    )[0]
    cutoff = CoordinationCutoffResolution(
        cutoff_A=5.0,
        smoothing_width_A=0.2,
        mode="direct",
    )
    eager_coordination = compute_coordination_profiles(
        frames,
        ordered_pairs=[("Ar", "Ar")],
        surface_mode="none",
        timestep_fs=2.0,
        cutoff_resolutions={("Ar", "Ar"): cutoff},
    )[0]
    with prepare_cli_trajectory(
        source,
        options=TrajectoryExecutionOptions(mode="stream", chunk_target_mib=1.0),
    ) as (streamed, _diagnostics):
        actual_position = compute_position_profiles(
            streamed,
            species="Ar",
            surface_mode="none",
            timestep_fs=2.0,
        )[0]
        actual_coordination = compute_coordination_profiles(
            streamed,
            ordered_pairs=[("Ar", "Ar")],
            surface_mode="none",
            timestep_fs=2.0,
            cutoff_resolutions={("Ar", "Ar"): cutoff},
        )[0]
    np.testing.assert_allclose(actual_position.x, eager_position.x)
    np.testing.assert_allclose(
        actual_position.distance_to_surface,
        eager_position.distance_to_surface,
    )
    np.testing.assert_allclose(
        actual_coordination.coordination_number,
        eager_coordination.coordination_number,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_cleanup_stale_spools_only_removes_marked_dead_process(tmp_path):
    stale = tmp_path / "linak-spool-stale"
    stale.mkdir()
    (stale / ".linak-spool.json").write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "uid": None,
                "owner": os.getenv("USERNAME") or os.getenv("USER") or "",
                "start_time": time.time() - 25 * 60 * 60,
            }
        ),
        encoding="utf-8",
    )
    unmarked = tmp_path / "linak-spool-unmarked"
    unmarked.mkdir()
    assert cleanup_stale_spools(tmp_path) == 1
    assert not stale.exists()
    assert unmarked.exists()


def test_text_stream_spool_persists_external_cell_override_and_cleans_up(tmp_path):
    source = tmp_path / "plain.xyz"
    raw_frames = [Atoms("Ar", positions=[[1.0 + index, 2.0, 3.0]]) for index in range(3)]
    write_trajectory(raw_frames, source)
    spool_path = None
    with prepare_cli_trajectory(
        source,
        options=TrajectoryExecutionOptions(
            mode="stream",
            chunk_target_mib=1.0,
            temporary_directory=tmp_path,
        ),
    ) as (streamed, diagnostics):
        assert diagnostics.resolved_mode == "stream"
        assert diagnostics.spool_directory is not None
        spool_path = diagnostics.spool_directory
        from linak.trajectory.execution import ChunkedTrajectorySequence

        assert isinstance(streamed, ChunkedTrajectorySequence)
        streamed.set_cell_override([9.0, 10.0, 11.0])
        for _pass in range(2):
            first = next(iter(streamed))
            np.testing.assert_allclose(first.cell.lengths(), [9.0, 10.0, 11.0])
            assert np.all(first.pbc)
    assert spool_path is not None
    assert not Path(spool_path).exists()


def test_streaming_spatial_filter_matches_eager_and_remains_seekable(tmp_path):
    source = tmp_path / "source.traj.h5"
    frames = _periodic_frames()
    write_trajectory(frames, source)
    options = SpatialFilterOptions(x_range="0.0:2.0")
    eager = apply_spatial_filter(frames, options=options)
    with prepare_cli_trajectory(
        source,
        options=TrajectoryExecutionOptions(mode="stream", chunk_target_mib=1.0),
    ) as (streamed, _diagnostics):
        actual = apply_spatial_filter_streaming(
            streamed,
            options=options,
            temporary_directory=tmp_path,
            chunk_frames=2,
        )
        try:
            assert len(actual.frames) == len(eager.frames)
            for expected_frame, actual_frame in zip(eager.frames, actual.frames):
                np.testing.assert_allclose(actual_frame.positions, expected_frame.positions)
            assert sum(1 for _frame in actual.frames) == len(frames)
        finally:
            cast(Any, actual.frames).close()
