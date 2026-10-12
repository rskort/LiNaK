"""Scientific reference captured from the pre-optimization implementation.

The JSON contains the complete SurfaceReference (both estimates, diagnostics,
confidence, candidate membership, populations and layout), not just heights.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
from ase.constraints import FixAtoms

from benchmarks.surface_fixtures import CASES, surface_frames
import linak.analysis.surface as surface
from linak.analysis.topology import prepare_topology


def compare_reference(actual, expected, path="reference"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare_reference(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            compare_reference(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=2e-10, equal_nan=True, err_msg=path
        )
    else:
        assert actual == expected, path


@pytest.mark.parametrize("case", CASES)
def test_surface_matches_complete_baseline_across_systems(case):
    baseline = json.loads(
        (Path(__file__).parent / "fixtures/surface_reference_baseline.json").read_text()
    )
    frames, options = surface_frames(case)
    prepare_topology(frames)
    original_positions = [frame.positions.copy() for frame in frames]
    reference = surface.estimate_surface_layout_reference(frames, **options)
    compare_reference(surface._json_ready(asdict(reference)), baseline[case])
    for frame, positions in zip(frames, original_positions):
        np.testing.assert_array_equal(frame.positions, positions)


def test_periodic_cut_preparation_skips_full_geometry_and_preserves_metadata(monkeypatch):
    frames, options = surface_frames("partial")
    frames[0].info["custom"] = {"source": "test"}
    frames[0].new_array("custom", np.arange(len(frames[0])))
    calls = []
    original = surface._candidate_mobility

    def mobility(positions):
        calls.append(positions.shape)
        return original(positions)

    monkeypatch.setattr(surface, "_candidate_mobility", mobility)
    context_options = surface.SurfaceEstimatorOptions(surface_elements=("Pt",))
    canonical, *_ = surface._canonical_surface_frames(frames, axis="z", options=context_options)
    assert calls == []
    assert canonical[0].info == frames[0].info
    np.testing.assert_array_equal(canonical[0].arrays["custom"], frames[0].arrays["custom"])
    assert canonical[0].constraints == frames[0].constraints
    assert not np.shares_memory(canonical[0].positions, frames[0].positions)
    surface.estimate_surface_layout_reference(frames, surface_elements=["Pt"])
    assert len(calls) == 1  # canonical geometry, reused by independent top/bottom selection


def test_changing_fixed_masks_use_same_candidates_for_periodic_cuts():
    frames, _ = surface_frames("mobile")
    frames[1].set_constraint(FixAtoms(indices=[1, 2, 3]))
    options = surface.SurfaceEstimatorOptions(surface_elements=("Pt",))
    full = surface._build_surface_context(frames, "z", options=options)
    cuts_only = surface._build_surface_context(
        frames, "z", options=options, _prepare_geometry=False
    )
    assert full is not None and cuts_only is not None
    for index in range(len(frames)):
        left = surface._candidate_axis_values_and_indices(full, index)
        right = surface._candidate_axis_values_and_indices(cuts_only, index)
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])


def test_blocked_mobility_is_exact_and_limits_scratch(monkeypatch):
    positions = np.random.default_rng(67).normal(size=(1200, 350, 3))
    medians = np.median(positions, axis=0)
    expected = np.median(np.linalg.norm(positions - medians[None], axis=2), axis=0)
    original_norm = np.linalg.norm
    blocks = []

    def norm(values, *args, **kwargs):
        blocks.append(values.nbytes)
        assert values.nbytes <= 8 * 1024**2
        return original_norm(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", norm)
    np.testing.assert_array_equal(surface._candidate_mobility(positions), expected)
    assert len(blocks) == 2


def test_population_analysis_reuses_exact_reactive_and_variable_topology(monkeypatch):
    import linak.analysis.water as water

    calls = []
    original = water._oh_molecule_topology_with_backend

    def detect(frame, **kwargs):
        calls.append(frame)
        return original(frame, **kwargs)

    monkeypatch.setattr(water, "_oh_molecule_topology_with_backend", detect)
    for case in ("reactive", "variable-count", "reordered"):
        frames, options = surface_frames(case)
        profile = prepare_topology(frames)
        assert len(profile.states) > 1
        calls.clear()
        reference = surface.estimate_surface_layout_reference(frames, **options)
        assert calls == []
        assert reference.top_population_per_frame.size == len(frames)
