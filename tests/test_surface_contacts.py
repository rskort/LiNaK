"""Automatic wetting detection is independent of the periodic coordinate cut."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from linak.analysis.surface import (
    SurfaceEstimatorOptions,
    estimate_surface_layout_reference,
)
from linak.trajectory.io import (
    TrajectoryStoredMetadata,
    read_trajectory,
    read_trajectory_hdf5_surface_cache,
    write_trajectory,
)


def contact_frame(water_z=(6.0,), *, period=40.0):
    symbols = ["Pt"] * 12
    positions = [[x, y, z] for z in (0.0, 2.0, 4.0) for x, y in ((0, 0), (3, 0), (0, 3), (3, 3))]
    for i, z in enumerate(water_z):
        x = 1.0 + (i % 3) * 3.0
        y = 1.0 + (i // 3) * 3.0
        symbols.extend(("O", "H", "H"))
        positions.extend(([x, y, z], [x + 0.8, y, z], [x, y + 0.8, z]))
    return Atoms(symbols, positions=positions, cell=[12, 12, period], pbc=True)


def auto(frames, distance_mode="plane"):
    return estimate_surface_layout_reference(
        frames,
        surface_layout="auto",
        surface_elements=["Pt"],
        surface_options=SurfaceEstimatorOptions(distance_mode=distance_mode),
    )


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
@pytest.mark.parametrize("shift", [0.0, 19.0, -5.0, 80.0])
def test_far_water_across_cut_does_not_wet_bottom(mode, shift):
    frame = contact_frame((6.0, 8.0, 24.0, 27.0))
    frame.positions[:, 2] = np.mod(frame.positions[:, 2] + shift, 40.0)
    reference = auto([frame], mode)
    assert reference.resolved_layout == "single-top"
    contact = reference.validation_diagnostics["layout_contact"]
    assert contact["mean_top"] == 2
    assert contact["mean_bottom"] == 0
    assert contact["bottom_occupancy"] == 0


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
def test_both_wetted_faces_and_fixed_layers(mode):
    frame = contact_frame((6.0, 8.0, 37.0))
    without = auto([frame], mode)
    frame.set_constraint(FixAtoms(indices=list(range(4))))
    with_fixed = auto([frame], mode)
    assert without.resolved_layout == with_fixed.resolved_layout == "symmetric"
    assert (
        without.validation_diagnostics["layout_contact"]
        == with_fixed.validation_diagnostics["layout_contact"]
    )


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
def test_single_bottom_and_persistence_counts_empty_frames(mode):
    frame = contact_frame((37.0,))
    assert auto([frame], mode).resolved_layout == "single-bottom"
    frames = [frame.copy() for _ in range(8)] + [frame[:12] for _ in range(2)]
    contact = auto(frames, mode).validation_diagnostics["layout_contact"]
    assert contact["bottom_occupancy"] == 0.8
    assert auto(frames, mode).resolved_layout != "symmetric"


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
def test_contact_band_boundary_and_half_gap_tie(mode):
    boundary = auto([contact_frame((10.0, 10.001))], mode)
    assert boundary.validation_diagnostics["layout_contact"]["mean_top"] == 1
    # The 8 A solvent gap caps each band to 4 A, meeting at z=8.
    reference = auto([contact_frame((8.0,), period=12.0)], mode)
    contact = reference.validation_diagnostics["layout_contact"]
    assert contact["mean_top"] == contact["mean_bottom"] == 0.5


def test_contact_population_work_uses_one_existing_frame_pass(monkeypatch):
    import linak.analysis.surface as surface

    class CountedFrames(list):
        passes = 0

        def __iter__(self):
            self.passes += 1
            return super().__iter__()

    frame = contact_frame((6.0, 24.0))
    reference = estimate_surface_layout_reference(
        [frame], surface_layout="symmetric", surface_elements=["Pt"]
    )
    frames = CountedFrames([frame])
    monkeypatch.setattr(
        surface, "_build_surface_context", lambda *a, **k: pytest.fail("extra context")
    )
    monkeypatch.setattr(
        surface,
        "_surface_axis_periodic_geometry",
        lambda *a, **k: pytest.fail("extra geometry pass"),
    )
    _top, _bottom, contact = surface._surface_reference_water_populations(
        frames, reference, layout_options=SurfaceEstimatorOptions(surface_elements=("Pt",))
    )
    assert frames.passes == 1
    assert contact["mean_top"] == 1
    assert contact["mean_bottom"] == 0


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
def test_changing_cells_and_rough_extent(mode):
    frames = [contact_frame((6.0, 37.0)), contact_frame((6.0, 41.0), period=44.0)]
    frames[1].positions[8:12, 2] += [0.1, -0.1, 0.2, -0.2]
    reference = estimate_surface_layout_reference(
        frames,
        surface_layout="auto",
        surface_elements=["Pt"],
        # This deliberately tiny moving slab has low estimator confidence;
        # exercise contact geometry independently of that established gate.
        surface_options=SurfaceEstimatorOptions(distance_mode=mode, low_confidence_threshold=0.4),
    )
    assert reference.resolved_layout == "symmetric"
    contact = reference.validation_diagnostics["layout_contact"]
    assert contact["top_occupancy"] == contact["bottom_occupancy"] == 1


@pytest.mark.parametrize("layout", ["auto", "single-top", "symmetric"])
def test_surface_cache_classifier_version(layout, tmp_path):
    frame = contact_frame((6.0, 37.0))
    reference = estimate_surface_layout_reference(
        [frame], surface_layout=layout, surface_elements=["Pt"]
    )
    kwargs = dict(
        axis="z",
        surface_mode="auto",
        surface_layout=layout,
        surface_elements=["Pt"],
        include_fixed_surface_atoms=False,
        rough_surface_envelope_A=None,
        frame_count=1,
    )
    metadata = TrajectoryStoredMetadata(
        surface_cache_status="available",
        surface_cache_axis="z",
        surface_cache_mode="auto",
        surface_cache_layout=layout,
        surface_cache_elements=("Pt",),
        surface_cache_estimate=reference,
    )
    current = tmp_path / "current.traj.h5"
    write_trajectory([frame], current, metadata=metadata)
    assert read_trajectory_hdf5_surface_cache(current, **kwargs) is not None
    old = tmp_path / "old.traj.h5"
    write_trajectory(
        [frame],
        old,
        metadata=replace(
            metadata, surface_cache_estimate=replace(reference, validation_diagnostics={})
        ),
    )
    loaded = read_trajectory_hdf5_surface_cache(old, **kwargs)
    assert (loaded is None) == (layout == "auto")


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
def test_supplied_pt_cs_layout(mode):
    source = Path(__file__).resolve().parents[1] / "data/Pt110_1x2_Cs2/Pt110_1x2_Cs2-pos-1.xyz"
    if not source.exists():
        pytest.skip("Local supplied trajectory is not distributed with the test suite")
    frames = read_trajectory(source)
    for frame in frames:
        frame.set_cell([15.92, 14.07, 51.72])
        frame.set_pbc(True)
    without = auto(frames, mode)
    for frame in frames:
        frame.set_constraint(FixAtoms(indices=list(range(514, 554))))
    with_fixed = auto(frames, mode)
    assert without.resolved_layout == with_fixed.resolved_layout == "single-top"
    contact = without.validation_diagnostics["layout_contact"]
    assert contact["top_occupancy"] == 1
    assert contact["mean_bottom"] == 0
    assert contact == with_fixed.validation_diagnostics["layout_contact"]


@pytest.mark.parametrize("mode", ["plane", "nearest-atom"])
@pytest.mark.parametrize("execution", ["eager", "stream"])
def test_xyz_and_conversion_share_constraints_and_layout(mode, execution, tmp_path):
    import argparse
    from ase.io import write
    from linak.cli import _open_compute_trajectory, _release_compute_resources, main
    from linak.analysis.position import load_position_profiles

    frame = contact_frame((6.0, 8.0, 24.0, 27.0))
    source = tmp_path / "sample.xyz"
    write(source, frame)
    simulation_input = tmp_path / "input.inp"
    simulation_input.write_text(
        "&FORCE_EVAL\n &SUBSYS\n  &CELL\n   ABC 12 12 40\n  &END CELL\n &END SUBSYS\n"
        "&END FORCE_EVAL\n&MOTION\n &MD\n  TIMESTEP 1\n &END MD\n"
        " &CONSTRAINT\n  &FIXED_ATOMS\n   LIST 1..4\n  &END FIXED_ATOMS\n"
        " &END CONSTRAINT\n&END MOTION\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        trajectory=str(source),
        trajectory_execution=execution,
        input=str(simulation_input),
        temporary_directory=str(tmp_path),
    )
    frames = _open_compute_trajectory(args, analysis_name="Position")
    try:
        for actual in (frames[0], next(iter(frames)), frames[0]):
            np.testing.assert_array_equal(actual.constraints[0].get_indices(), np.arange(4))
    finally:
        _release_compute_resources(frames)
    assert not list(tmp_path.glob("linak-spool-*"))
    common = ["--surface-distance-mode", mode, "--surface-elements", "Pt"]
    direct = tmp_path / "direct.position.h5"
    converted = tmp_path / "converted.traj.h5"
    result = tmp_path / "converted.position.h5"
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "compute",
                "position",
                str(source),
                "--species",
                "O",
                "--trajectory-execution",
                execution,
                "--temporary-directory",
                str(tmp_path),
                "--output",
                str(direct),
                *common,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(source),
                "--output",
                str(converted),
                *common,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "compute",
                "position",
                str(converted),
                "--species",
                "O",
                "--output",
                str(result),
                *common,
            ]
        )
        == 0
    )
    a, b = load_position_profiles(direct)[0], load_position_profiles(result)[0]
    assert (
        a.surface_reference.resolved_layout == b.surface_reference.resolved_layout == "single-top"
    )
    np.testing.assert_allclose(a.distance_to_surface, b.distance_to_surface, atol=1e-5)
