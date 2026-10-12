"""Profile or measure general surface fixtures using current or snapshotted code.

Run baseline/optimized measurements sequentially, without other benchmarks/tests.
"""

import argparse
import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

# Support both direct script execution and namespace-package test/type imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.conversion_performance import peak_rss_mib
from benchmarks.surface_fixtures import CASES, surface_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.baseline:
        sys.path.insert(0, str(args.baseline.resolve()))
    import numpy as np
    from linak.analysis.surface import estimate_surface_layout_reference, _json_ready
    from linak.analysis.topology import prepare_topology
    from linak.cell_geometry import unwrap_positions

    results = {}
    references = {}
    profiler = cProfile.Profile() if args.profile else None
    for case in args.cases:
        frames, options = surface_frames(case, args.frames, args.grid)
        # Conversion constructs topology before surface analysis; include that
        # same cache lifecycle, measuring its build separately.
        topology_start = time.perf_counter()
        prepare_topology(frames)
        topology_s = time.perf_counter() - topology_start
        times = []
        for _ in range(args.runs):
            start = time.perf_counter()
            if profiler:
                reference = profiler.runcall(estimate_surface_layout_reference, frames, **options)
            else:
                reference = estimate_surface_layout_reference(frames, **options)
            times.append(time.perf_counter() - start)
        references[case] = _json_ready(asdict(reference))
        mic_times = []
        if case != "missing":
            coordinates = np.stack(
                [frame.positions for frame in frames if len(frame) == len(frames[0])]
            )
            geometry_frames = [frame for frame in frames if len(frame) == len(frames[0])]
            for _ in range(args.runs):
                start = time.perf_counter()
                unwrap_positions(
                    coordinates, [f.cell for f in geometry_frames], [f.pbc for f in geometry_frames]
                )
                mic_times.append(time.perf_counter() - start)
        results[case] = {
            "frames": len(frames),
            "atoms": len(frames[0]),
            "surface_runs_s": times,
            "median_surface_s": statistics.median(times),
            "topology_build_s": topology_s,
            "median_unwrap_s": statistics.median(mic_times) if mic_times else None,
        }
    if profiler:
        profiler.dump_stats(str(args.profile))
    if args.reference:
        args.reference.write_text(json.dumps(references, indent=2), encoding="utf-8")
    result = {"cases": results, "peak_rss_mib": peak_rss_mib()}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
