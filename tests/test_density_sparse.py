from __future__ import annotations

import inspect
import json
from pathlib import Path

from ase import Atoms
import numpy as np
import pytest

from linak.analysis import density
from linak.analysis import density_sparse
from linak.analysis.density_sparse import SparseReducer, SparseWorkspace, compact_rows, row_dtype
from linak.analysis.surface import SurfaceReference
from linak.runtime_resources import MIB, ResourcePolicy
from benchmarks.density_fixtures import density_payloads, json_value
from benchmarks.surface_fixtures import CASES


@pytest.mark.parametrize("channels", [2, 6])
@pytest.mark.parametrize("budget", [8192, 1024 * 1024])
def test_balanced_reduction_matches_whole_array_and_bounds_buffers(channels, budget):
    random = np.random.default_rng(123)
    rows = np.zeros(1500, dtype=row_dtype(channels))
    rows["group"] = random.integers(0, 4, rows.size)
    rows["bin"] = random.integers(0, 300, rows.size)
    rows["bin"][::11] = np.iinfo(np.uint64).max - 1
    rows["values"] = random.random((rows.size, channels))
    expected = compact_rows(rows)
    with SparseWorkspace(budget) as workspace:
        reducer = SparseReducer(workspace, channels)
        for start in range(0, rows.size, 13):
            reducer.add(rows[start : start + 13])
        run = reducer.finish()
        actual = run.materialize()
        np.testing.assert_array_equal(actual["group"], expected["group"])
        np.testing.assert_array_equal(actual["bin"], expected["bin"])
        np.testing.assert_allclose(actual["values"], expected["values"], rtol=1.0e-14)
        assert workspace.metrics["peak_pending_bytes"] <= workspace.buffer_limit
        assert workspace.metrics["peak_resident_bytes"] <= workspace.budget // 2
        assert workspace.metrics["merged_rows"] < rows.size * 15
        temporary = workspace._temporary
        if budget == 8192:
            assert workspace.metrics["spill_bytes_written"] > 0
            assert temporary is not None
    if temporary is not None:
        assert not Path(temporary.name).exists()
    assert workspace.closed


def test_shared_budget_and_limit_counts_unique_bins_not_pending_rows():
    rows = np.zeros(30, dtype=row_dtype(2))
    rows["bin"] = np.arange(30) % 3
    rows["values"] = 1.0
    with SparseWorkspace(8192) as workspace:
        reducers = [SparseReducer(workspace, 2, max_bins=3) for _ in range(4)]
        for _ in range(40):
            for reducer in reducers:
                reducer.add(rows)
        for reducer in reducers:
            run = reducer.finish()
            np.testing.assert_array_equal(run.materialize()["values"], np.full((3, 2), 400.0))
            run.release()
        assert workspace.pending_bytes == workspace.resident_bytes == 0
        assert workspace.metrics["peak_pending_bytes"] <= workspace.buffer_limit
    with SparseWorkspace(8192) as workspace:
        reducer = SparseReducer(workspace, 2, max_bins=2, species="K")
        reducer.add(rows)
        with pytest.raises(ValueError, match="species='K'.*nonzero bins=3"):
            reducer.finish()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_interrupted_reductions_remove_spools(exception):
    workspace = SparseWorkspace(8192)
    with pytest.raises(exception), workspace:
        reducer = SparseReducer(workspace, 2)
        rows = np.zeros(300, dtype=row_dtype(2))
        rows["bin"] = np.arange(300)
        reducer.add(rows)
        reducer.flush()
        temporary = workspace._temporary
        assert temporary is not None
        raise exception("interrupted")
    assert not Path(temporary.name).exists()
    assert not workspace.runs


def test_unavailable_temporary_storage_reports_error_and_cleans_up(monkeypatch):
    original = Path.open

    def fail(path, *args, **kwargs):
        if path.name.startswith("run_"):
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    workspace = SparseWorkspace(8192)
    with pytest.raises(OSError, match="Density temporary.*failed"), workspace:
        reducer = SparseReducer(workspace, 2)
        rows = np.zeros(300, dtype=row_dtype(2))
        rows["bin"] = np.arange(300)
        reducer.add(rows)
        reducer.finish()
    assert workspace._temporary is not None
    assert not Path(workspace._temporary.name).exists()


def test_grid_invariants_and_exact_geometry_table_are_prepared_once(monkeypatch):
    frames = [
        Atoms("KOO", positions=[[1, 1, 1], [2, 2, 2], [3, 3, 3]], cell=[8, 8, 8], pbc=True)
        for _ in range(80)
    ]
    for index, frame in enumerate(frames):
        frame.set_celldisp([0.0, 0.0, index * 0.001])
    calls = []
    prepare = density._prepare_density_grid_geometry
    validate = density.uniform_bin_width_from_edges

    def prepare_once(*args):
        calls.append("geometry")
        return prepare(*args)

    def validate_once(*args, **kwargs):
        calls.append("width")
        return validate(*args, **kwargs)

    monkeypatch.setattr(density, "_prepare_density_grid_geometry", prepare_once)
    monkeypatch.setattr(density, "uniform_bin_width_from_edges", validate_once)
    profiles = density.compute_all_density_profiles(
        frames, species="elements", outputs="3d", bin_width=0.1, surface_mode="none"
    )
    assert calls.count("geometry") == 1
    assert calls.count("width") == 4 * len(profiles)
    assert len(profiles) == 2
    for profile in profiles:
        assert isinstance(profile, density.DensityGridProfile)
        assert len(profile.geometry_groups) == len(frames)
        assert sum(group["n_frames"] for group in profile.geometry_groups) == len(frames)
    source = inspect.getsource(density._DensitySparseGridAccumulator)
    assert "_merge_chunks" not in source
    assert "counts={}" not in source


def test_empty_geometry_frames_still_count_and_groups_share_prepared_table():
    frames = [Atoms("K", positions=[[1, 1, 1]], cell=[8, 8, 8], pbc=True) for _ in range(4)]
    table = density._prepare_density_grid_geometry(frames, "axis", None, None)
    edges = np.arange(9, dtype=float)
    with SparseWorkspace(8192) as workspace:
        accumulator = density._DensitySparseGridAccumulator(
            species_label="K",
            count_label="atom",
            x_bin_edges=edges,
            y_bin_edges=edges,
            z_bin_edges=edges,
            distance_bin_edges=edges,
            grid_bin_width=1.0,
            max_nonzero_bins=10,
            surface_axis="z",
            distance_coordinate_mode="axis",
            geometry_table=table,
            workspace=workspace,
        )
        result = accumulator.finalize(n_frames=4)
    assert result.flat_indices.size == 0
    assert len(result.geometry_groups) == 1
    assert result.geometry_groups[0]["n_frames"] == 4
    assert result.geometry_groups[0]["flat_indices"].size == 0


def test_one_dimensional_outputs_do_not_show_grid_finalization(caplog):
    frames = [Atoms("K", positions=[[1, 1, 1]], cell=[8, 8, 8], pbc=True)]
    with caplog.at_level("INFO"):
        density.compute_all_density_profiles(frames, species="K", outputs="1d", surface_mode="none")
    assert "Finalize density grids" not in caplog.text


def test_workspace_respects_automatic_and_explicit_budget(monkeypatch):
    monkeypatch.setattr(
        density_sparse, "resolve_resource_policy", lambda: ResourcePolicy(100 * MIB, 1, "automatic")
    )
    with pytest.raises(ValueError, match="free memory.*LINAK_MEMORY_BUDGET_MIB"):
        SparseWorkspace()
    monkeypatch.setattr(
        density_sparse,
        "resolve_resource_policy",
        lambda: ResourcePolicy(100 * MIB, 64 * 1024, "environment"),
    )
    with SparseWorkspace() as workspace:
        assert workspace.budget == 8192


def test_unpack_skips_empty_groups_and_handles_group_crossing_chunk_boundary():
    with SparseWorkspace(8192) as workspace:
        reducer = SparseReducer(workspace, 6)
        rows = np.zeros(120, dtype=row_dtype(6))
        rows["group"] = np.repeat([1, 3], 60)
        rows["bin"] = np.tile(np.arange(60), 2)
        rows["values"] = np.arange(6) + 0.5
        reducer.add(rows)
        run = reducer.finish()
        bins, values, offsets = run.unpack(np.float32, group_count=5)
        np.testing.assert_array_equal(offsets, [0, 0, 60, 60, 120, 120])
        np.testing.assert_array_equal(bins, rows["bin"])
        for channel, vector in enumerate(values):
            np.testing.assert_array_equal(vector, rows["values"][:, channel])
            assert vector.dtype == np.float32
        run.release()


def test_interface_ties_preserve_half_mass_and_half_entity_weights():
    reference = SurfaceReference(
        requested_layout="symmetric",
        resolved_layout="symmetric",
        axis="z",
        top=None,
        bottom=None,
        axis_cut_per_frame_A=np.zeros(1),
        axis_lower_per_frame_A=np.zeros(1),
        axis_period_per_frame_A=np.full(1, 4.0),
        top_population_per_frame=np.zeros(1),
        bottom_population_per_frame=np.zeros(1),
        auto_reason="explicit_symmetric",
    )
    edges = np.arange(5, dtype=float)
    with SparseWorkspace(8192) as workspace:
        accumulator = density._DensitySparseGridAccumulator(
            species_label="K",
            count_label="atom",
            x_bin_edges=edges,
            y_bin_edges=edges,
            z_bin_edges=edges,
            distance_bin_edges=edges,
            grid_bin_width=1.0,
            max_nonzero_bins=10,
            surface_axis="z",
            distance_coordinate_mode="axis",
            surface_reference=reference,
            workspace=workspace,
        )
        accumulator.update(
            positions=np.ones((3, 3)),
            distance_values=np.ones(3),
            masses=np.asarray([2.0, 4.0, 6.0]) * 1e-23,
            frame_index=0,
            validation_plan=density._DensityValidationPlan(True, frozenset()),
            interface_side=np.asarray([-1, 0, 1]),
        )
        result = accumulator.finalize(n_frames=1)
    np.testing.assert_allclose(result.mass_sum_g, [12e-23], rtol=1e-6)
    np.testing.assert_allclose(result.top_mass_sum_g, [8e-23], rtol=1e-6)
    np.testing.assert_allclose(result.bottom_mass_sum_g, [4e-23], rtol=1e-6)
    np.testing.assert_array_equal(result.entity_sum, [3.0])
    np.testing.assert_array_equal(result.top_entity_sum, [1.5])
    np.testing.assert_array_equal(result.bottom_entity_sum, [1.5])


def _assert_metadata(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_metadata(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_metadata(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12, equal_nan=True)
    else:
        assert actual == expected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("spill", [False, True])
def test_density_outputs_match_prechange_baseline(case, spill, monkeypatch):
    monkeypatch.setenv("LINAK_MEMORY_BUDGET_MIB", "0.125" if spill else "64")
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "density_reference_baseline.json").read_text(
            encoding="utf-8"
        )
    )[case]
    actual = density_payloads(case)
    assert len(actual) == len(expected)
    for payload, reference in zip(actual, expected):
        metadata = json.loads(json.dumps(payload["metadata"], default=json_value))
        _assert_metadata(metadata, reference["metadata"])
        assert set(payload["datasets"]) == set(reference["datasets"])
        for name, values in payload["datasets"].items():
            baseline = reference["datasets"][name]
            if values is None:
                assert baseline is None
                continue
            array = np.asarray(values)
            if array.dtype.kind in "f":
                np.testing.assert_allclose(
                    array,
                    np.asarray(baseline),
                    rtol=2.0e-6 if array.dtype.itemsize == 4 else 5.0e-12,
                    atol=1.0e-30,
                    equal_nan=True,
                    err_msg=f"{case}:{name}",
                )
            elif array.dtype.kind in "SU":
                assert json.loads(json.dumps(array, default=json_value)) == baseline
            else:
                np.testing.assert_array_equal(array, baseline, err_msg=f"{case}:{name}")
