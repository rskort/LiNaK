"""Allocation bounds, ordered execution and scientific parallel equivalence."""

from dataclasses import fields, is_dataclass
import importlib.util
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from linak import runtime_resources as resources
from linak.parallel import AUTO_POLICIES, bounded_map, computation, stage_workers
from linak.cancellation import ComputationCancelled, cancellation_scope
from linak.analysis.density import compute_all_density_profiles, save_density_profiles
from linak.analysis.orientation import compute_orientation_profile, save_orientation_profile
from linak.analysis.coordination import compute_coordination_profiles, CoordinationCutoffResolution
from linak.analysis.rdf import compute_rdf
from linak.cell_geometry import bin_volumes, bin_volumes_for_frames
from tests.test_symmetric_surface import _symmetric_frame


def assert_equivalent(a, b):
    if isinstance(a, np.ndarray):
        if a.dtype.kind in "fc":
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-25, equal_nan=True)
        else:
            np.testing.assert_array_equal(a, b)
    elif is_dataclass(a):
        for f in fields(a):
            assert_equivalent(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equivalent(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            assert_equivalent(aa, bb)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, rtol=5e-12, atol=1e-25, equal_nan=True)
    else:
        assert a == b


def trajectory(tilted=False):
    frames = []
    for i in range(37):
        f = _symmetric_frame(crossing_boundary=True)
        f.positions[-6:, 0] += 0.01 * (i % 7)
        if tilted:
            f.set_cell([[10.0, 0, 0], [1.5 + i * 0.001, 10.0, 0], [0, 0, 20.0]])
            f.set_celldisp([1.0, -2.0, 0.25])
        frames.append(f)
    return frames


def test_cpu_budget_respects_affinity_slurm_and_quota(monkeypatch, tmp_path):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(resources.os, "process_cpu_count", lambda: 96, raising=False)
    monkeypatch.setattr(
        resources.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )
    monkeypatch.setattr(resources, "_cgroup_directories", lambda: [tmp_path])
    (tmp_path / "cpu.max").write_text("800000 100000")
    assert resources.detect_cpu_count(environ={"SLURM_CPUS_PER_TASK": "48"}) == 8
    (tmp_path / "cpu.max").write_text("max 100000")
    assert resources.detect_cpu_count(environ={"SLURM_CPUS_PER_TASK": "6"}) == 6
    assert resources.detect_cpu_count(environ={"SLURM_CPUS_PER_TASK": "invalid"}) == 32


def test_memory_budget_respects_cgroup_remaining_and_explicit_worker_request(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_cgroup_directories", lambda: [tmp_path])
    monkeypatch.setattr(resources, "detect_cpu_count", lambda: 48)
    (tmp_path / "memory.max").write_text(str(512 * resources.MIB))
    (tmp_path / "memory.current").write_text(str(384 * resources.MIB))
    policy = resources.resolve_resource_policy(1024, environ={})
    assert policy.memory_budget_bytes == 128 * resources.MIB
    assert (
        resources.resolve_worker_count(
            task_count=100,
            estimated_bytes_per_worker=64 * resources.MIB,
            requested_threads=48,
            policy=policy,
        )
        == 2
    )


def test_stage_policy_and_explicit_budget_are_separate(monkeypatch):
    monkeypatch.setattr(resources, "detect_cpu_count", lambda: 48)
    with computation(None, 512):
        assert stage_workers("density.frames", 100) == AUTO_POLICIES["density.frames"][0]
    with computation(8, 512):
        assert stage_workers("density.frames", 100) == 8
        assert stage_workers("density.frames", 100, serial_reason="ordered provider") == 1


def test_bounded_map_orders_results_and_limits_prefetch():
    reads = []
    main_thread = threading.get_ident()

    def inputs():
        for i in range(20):
            assert threading.get_ident() == main_thread
            reads.append(i)
            yield i

    def task(i):
        time.sleep(0.001 * (3 - i % 3))
        return i

    iterator = bounded_map(task, inputs(), workers=2)
    assert next(iterator) == 0
    assert len(reads) == 4
    assert list(iterator) == list(range(1, 20))


def test_bounded_map_rejects_late_work_after_cancellation():
    cancelled = False
    reads = []

    def inputs():
        for i in range(30):
            reads.append(i)
            yield i

    with cancellation_scope(lambda: cancelled):
        iterator = bounded_map(lambda x: x, inputs(), workers=2)
        next(iterator)
        cancelled = True
        with pytest.raises(ComputationCancelled):
            next(iterator)
    assert len(reads) == 4


def test_inner_work_does_not_start_another_pool():
    with computation(8, 512):
        assert (
            list(bounded_map(lambda _: stage_workers("density.frames", 50), range(4), workers=2))
            == [1] * 4
        )


def test_native_limits_are_restored_after_parallel_work():
    from threadpoolctl import threadpool_info

    before = [(x["prefix"], x["num_threads"]) for x in threadpool_info()]
    with computation(4, 512):
        list(bounded_map(lambda x: np.eye(3) @ np.eye(3), range(4), workers=2))
    assert [(x["prefix"], x["num_threads"]) for x in threadpool_info()] == before


def test_worker_construction_failure_restores_native_limits(monkeypatch):
    from linak import parallel
    from threadpoolctl import threadpool_info

    before = [(x["prefix"], x["num_threads"]) for x in threadpool_info()]

    def fail(**kwargs):
        raise PermissionError("IPC unavailable")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", fail)
    with pytest.raises(PermissionError, match="IPC unavailable"):
        list(bounded_map(lambda x: x, range(4), workers=2))
    assert [(x["prefix"], x["num_threads"]) for x in threadpool_info()] == before


@pytest.mark.parametrize("tilted", [False, True])
def test_density_and_orientation_serial_parallel_match(tilted, tmp_path):
    frames = trajectory(tilted)
    options = dict(surface_layout="symmetric", surface_elements=["Pt"])
    for name, compute, save, extra in (
        (
            "density",
            compute_all_density_profiles,
            save_density_profiles,
            dict(species="all", bin_width=0.5, grid_bin_width=1.0, min_molecule_frames=1),
        ),
        ("orientation", compute_orientation_profile, save_orientation_profile, dict(bin_width=0.5)),
    ):
        serial = compute(frames, threads=1, **options, **extra)
        parallel = compute(frames, threads=2, **options, **extra)
        assert_equivalent(serial, parallel)
        paths = [tmp_path / f"{name}-{count}.h5" for count in (1, 2)]
        save(serial, paths[0])
        save(parallel, paths[1])
        spec = importlib.util.spec_from_file_location(
            "benchmark_compute", Path("benchmarks/benchmark_compute.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.compare_results(*paths)


@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("pair", [("O", "H"), ("mol:H2O", "H")])
def test_coordination_serial_parallel_matches(tilted, pair):
    frames = trajectory(tilted)
    options = dict(
        ordered_pairs=[pair],
        surface_layout="symmetric",
        surface_elements=["Pt"],
        cutoff_resolutions={
            pair: CoordinationCutoffResolution(cutoff_A=1.2, smoothing_width_A=0.2, mode="direct")
        },
    )
    serial = compute_coordination_profiles(frames, threads=1, **options)
    parallel = compute_coordination_profiles(frames, threads=2, **options)
    assert_equivalent(serial, parallel)


@pytest.mark.parametrize("pair", [("O", "H"), ("O", "O")])
def test_small_memory_coordination_tiles_preserve_counts_and_self_exclusion(pair):
    frames = [_symmetric_frame().repeat((4, 4, 1)) for _ in range(3)]
    options = dict(
        ordered_pairs=[pair],
        surface_layout="symmetric",
        surface_elements=["Pt"],
        cutoff_resolutions={
            pair: CoordinationCutoffResolution(cutoff_A=1.2, smoothing_width_A=0.2, mode="direct")
        },
    )
    baseline = compute_coordination_profiles(frames, threads=1, **options)
    tiled = compute_coordination_profiles(frames, threads=2, memory_budget_mib=0.01, **options)
    assert_equivalent(baseline, tiled)


def test_generic_rdf_uses_spawn_workers_without_changing_results():
    frames = trajectory(True)
    options = dict(species_a="O", species_b="H", r_max=2.0, bin_width=0.2)
    serial = compute_rdf(frames, threads=1, **options)
    parallel = compute_rdf(frames, threads=2, **options)
    assert_equivalent(serial, parallel)


def test_repeated_volume_geometry_is_computed_once(monkeypatch):
    from linak import cell_geometry

    frames = trajectory()
    edges = np.linspace(0, 20, 41)
    expected = np.asarray([bin_volumes(f.cell, edges, 2, origin=np.zeros(3)) for f in frames])
    calls = []
    original = cell_geometry.bin_volumes

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cell_geometry, "bin_volumes", counted)
    actual = bin_volumes_for_frames(frames, edges, 2)
    assert len(calls) == 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("budget", [0.125, 64.0])
def test_reordered_elements_match_direct_histogram_independent_of_batch_size(budget):
    frames = []
    for index in range(35):
        frame = _symmetric_frame()
        if index % 2:
            frame = frame[np.arange(len(frame))[::-1]]
        frames.append(frame)
    profiles = compute_all_density_profiles(
        frames,
        species="O",
        outputs="1d",
        surface_mode="none",
        surface_layout=None,
        bin_width=1.0,
        threads=2,
        memory_budget_mib=budget,
    )
    for profile in profiles:
        if profile.coordinate_mode != "axis":
            continue
        axis = {"x": 0, "y": 1, "z": 2}[profile.axis]
        counts = np.zeros(len(profile.bin_centers))
        for frame in frames:
            selected = np.asarray(frame.get_chemical_symbols()) == "O"
            counts += np.histogram(frame.positions[selected, axis], profile.bin_edges)[0]
        np.testing.assert_allclose(profile.entities_per_frame, counts / len(frames), rtol=1e-12)
