"""Measure full conversion and exact live versus persisted molecular analysis.

Usage: python benchmarks/topology_performance.py SOURCE ARTIFACTS
All measurements run sequentially in fresh processes, without profiling.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from conversion_performance import peak_rss_mib


def child(source: Path, artifacts: Path, mode: str) -> dict:
    from linak.analysis import topology, water
    from linak.analysis.density import compute_density_profiles
    from linak import cli, conversion
    from linak.trajectory.io import read_trajectory
    import h5py

    path = artifacts / "cached.traj.h5"
    stages: dict[str, float] = {}
    original_stage = topology.progress_stage

    @contextmanager
    def stage(logger, description):
        start = time.perf_counter()
        with original_stage(logger, description):
            yield
        stages[description] = stages.get(description, 0.0) + time.perf_counter() - start

    topology.progress_stage = stage
    conversion.progress_stage = stage
    detections = 0
    original = water._oh_molecule_topology_with_backend

    def detect(*args, **kwargs):
        nonlocal detections
        detections += 1
        return original(*args, **kwargs)

    water._oh_molecule_topology_with_backend = detect
    started = time.perf_counter()
    if mode == "cold":
        if cli.main(["apply", "convert", str(source), "--output", str(path)]):
            raise RuntimeError("conversion failed")
        elapsed = time.perf_counter() - started
        with h5py.File(path) as handle:
            group = next(iter(handle["molecular_topology/oh"].values()))
            unique_states = len(group["mol:H_offsets"]) - 1
        return {
            "total_s": elapsed,
            "stages_s": stages,
            "detections": detections,
            "peak_rss_mib": peak_rss_mib(),
            "output_bytes": path.stat().st_size,
            "unique_states": unique_states,
        }
    frames = read_trajectory(path)
    read_s = time.perf_counter() - started
    if mode == "live":

        def live(frame, *, cutoff, backend, build_source=True):
            return detect(frame, oh_cutoff=cutoff, backend=backend)[0]

        topology.resolve_topology = live
    start = time.perf_counter()
    profiles = compute_density_profiles(
        frames,
        species="molecules",
        axis="z",
        surface_mode="none",
        surface_layout=None,
        bin_width=0.1,
    )
    analysis_s = time.perf_counter() - start
    # Retain processed scientific output to compare live and cached computations.
    import numpy as np

    np.savez(
        artifacts / f"{mode}_density.npz",
        **{profile.species: profile.counts_per_frame for profile in profiles},
    )
    return {
        "read_s": read_s,
        "analysis_s": analysis_s,
        "total_s": read_s + analysis_s,
        "detections": detections,
        "peak_rss_mib": peak_rss_mib(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--child", choices=("cold", "live", "warm"))
    parser.add_argument("--reuse-conversion", action="store_true")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.child:
        result = child(args.source.resolve(), args.artifacts.resolve(), args.child)
        print(json.dumps(result), flush=True)
        return
    results: dict[str, list[dict]] = {}
    for mode in ("cold", "live", "warm"):
        results[mode] = []
        for index in range(1 if mode == "cold" else 3):
            if mode != "cold" or not args.reuse_conversion:
                with (args.artifacts / f"{mode}_{index}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            str(args.source),
                            str(args.artifacts),
                            "--child",
                            mode,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            lines = (
                (args.artifacts / f"{mode}_{index}.log").read_text(encoding="utf-8").splitlines()
            )
            result = json.loads(lines[-1])
            results[mode].append(result)
            print(json.dumps({"mode": mode, "run": index + 1, **result}), flush=True)
    import h5py

    path = args.artifacts / "cached.traj.h5"
    without = args.artifacts / "without_cache.traj.h5"
    # Repack; merely deleting a group can leave its freed storage in the file.
    with h5py.File(path) as source, h5py.File(without, "w") as target:
        target.attrs.update(dict(source.attrs))
        for key in source:
            if key != "molecular_topology":
                source.copy(key, target)
    results["cold"][0]["cache_growth_bytes"] = path.stat().st_size - without.stat().st_size
    summary = {
        mode: {
            key: statistics.median(item[key] for item in values)
            for key in values[0]
            if isinstance(values[0][key], (int, float))
        }
        for mode, values in results.items()
    }
    (args.artifacts / "results.json").write_text(
        json.dumps({"runs": results, "medians": summary}, indent=2), encoding="utf-8"
    )
    import numpy as np

    with (
        np.load(args.artifacts / "live_density.npz") as live,
        np.load(args.artifacts / "warm_density.npz") as warm,
    ):
        assert live.files == warm.files
        for key in live.files:
            np.testing.assert_array_equal(live[key], warm[key])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
