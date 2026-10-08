"""Deterministic release probes; timings are reports, never CI pass/fail thresholds.

Run from an installed checkout: python benchmarks/pre_publication.py --frames 200
All generated data is isolated in a TemporaryDirectory; no user inputs are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable, TypeVar

import matplotlib
import numpy as np
from ase import Atoms

from linak.analysis.density import (
    DensityGridProfile,
    DensityProfile,
    compute_all_density_profiles,
    density_grid_to_heatmap_profile,
    plot_density_profiles,
)
from linak.analysis.surface import estimate_surface_layout_reference
from linak.analysis.water import OHTopologyCache, oh_molecule_topology
from linak.trajectory.execution import ChunkedTrajectorySequence
from linak.trajectory.io import write_trajectory

T = TypeVar("T")


def timed(action: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    result = action()
    return result, perf_counter() - start


def symmetric_slab() -> Atoms:
    symbols = []
    positions = []
    for x in (2.0, 7.0, 12.0, 17.0):
        for y in (2.0, 7.0, 12.0, 17.0):
            for z in (12.0, 14.0, 16.0):
                symbols.append("Pt")
                positions.append([x, y, z])
            for z in (8.0, 20.0):
                symbols.extend(["O", "H", "H"])
                positions.extend([[x, y, z], [x + 0.96, y, z], [x - 0.24, y + 0.93, z]])
    return Atoms(
        symbols,
        positions=positions,
        cell=[[20.0, 0.0, 0.0], [2.0, 20.0, 0.0], [0.0, 0.0, 30.0]],
        pbc=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    matplotlib.use("Agg")
    frame = symmetric_slab()
    frames = [frame] * args.frames
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "frames": len(frames),
        "atoms_per_frame": len(frame),
        "geometry": "tilted prismatic symmetric",
    }
    cache = OHTopologyCache(validation_stride=100)
    reference, result["layout_seconds"] = timed(
        lambda: estimate_surface_layout_reference(
            frames, axis="z", surface_layout="auto", mode="layered", surface_elements=["Pt"]
        )
    )
    result["resolved_layout"] = reference.resolved_layout
    _, result["uncached_topology_seconds"] = timed(
        lambda: [oh_molecule_topology(f) for f in frames]
    )
    _, result["cached_topology_seconds"] = timed(
        lambda: [cache.select(f, frame_index=i) for i, f in enumerate(frames)]
    )
    profiles, result["density_compute_seconds"] = timed(
        lambda: compute_all_density_profiles(
            frames,
            species="O",
            surface_axis="z",
            surface_layout="symmetric",
            surface_mode="layered",
            surface_elements=["Pt"],
            bin_width=0.5,
            grid_bin_width=0.5,
            outputs="all",
        )
    )
    grid = next(p for p in profiles if isinstance(p, DensityGridProfile))
    line = next(
        p for p in profiles if isinstance(p, DensityProfile) and p.coordinate_mode == "distance"
    )
    _, result["x_distance_projection_seconds"] = timed(
        lambda: density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="distance")
    )
    with TemporaryDirectory(prefix="linak-release-benchmark-") as name:
        directory = Path(name)
        _, result["stored_line_render_seconds"] = timed(
            lambda: plot_density_profiles(line, output=directory / "density.png", show=False)
        )
        path = directory / "trajectory.traj.h5"
        write_trajectory(frames, path)
        result["trajectory_bytes"] = path.stat().st_size
        sequence = ChunkedTrajectorySequence([path], [len(frames)], chunk_frames=16)
        _, result["indexed_last_frame_seconds"] = timed(lambda: sequence[-1])
        _, result["cached_last_frame_seconds"] = timed(lambda: sequence[-1])
        _, result["replay_to_last_frame_seconds"] = timed(lambda: sum(1 for _ in sequence))
    assert isinstance(result["density_compute_seconds"], float)
    assert np.isfinite(result["density_compute_seconds"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
