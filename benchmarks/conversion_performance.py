"""Unprofiled conversion timings in fresh processes, including peak resident memory."""

from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def peak_rss_mib():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize / 1024**2
    import importlib

    resource = importlib.import_module("resource")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.child is not None:
        baseline = args.artifacts / "baseline_src"
        if args.label == "before":
            if not (baseline / "linak").is_dir():
                raise SystemExit("Baseline source snapshot missing: " + str(baseline))
            sys.path.insert(0, str(baseline.resolve()))
        from linak import cli, conversion
        from linak.trajectory import io

        timings: dict[str, float] = {}

        def instrument(module, name, stage):
            original = getattr(module, name)

            def wrapped(*pos, **kw):
                key = stage
                if name == "read_trajectory" and Path(pos[0]).resolve() != args.source.resolve():
                    key = "summary_reread"
                start = time.perf_counter()
                try:
                    return original(*pos, **kw)
                finally:
                    timings[key] = timings.get(key, 0.0) + time.perf_counter() - start

            setattr(module, name, wrapped)

        instrument(io, "read_trajectory", "read")
        instrument(io, "write_trajectory", "write")
        instrument(conversion, "_conversion_metadata_with_surface_cache", "surface")
        instrument(conversion, "wrap_trajectory_frames", "wrap")
        output = args.artifacts / f"{args.label}_{args.child}.traj.h5"
        start = time.perf_counter()
        code = cli.main(["apply", "convert", str(args.source), "--output", str(output)])
        elapsed = time.perf_counter() - start
        if code:
            raise SystemExit(code)
        result = {
            "total_s": elapsed,
            "stages_s": timings,
            "peak_rss_mib": peak_rss_mib(),
            "output_bytes": output.stat().st_size,
        }
        (args.artifacts / f"{args.label}_{args.child}.json").write_text(
            json.dumps(result, indent=2)
        )
        return
    results = []
    for index in range(1, 4):
        with (args.artifacts / f"{args.label}_{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    str(args.source),
                    str(args.artifacts),
                    "--label",
                    args.label,
                    "--child",
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result = json.loads((args.artifacts / f"{args.label}_{index}.json").read_text())
        results.append(result)
        print(json.dumps({"run": index, **result}), flush=True)
    summary = {
        "median_total_s": statistics.median(r["total_s"] for r in results),
        "median_peak_rss_mib": statistics.median(r["peak_rss_mib"] for r in results),
        "median_stages_s": {
            key: statistics.median(r["stages_s"].get(key, 0) for r in results)
            for key in results[0]["stages_s"]
        },
    }
    (args.artifacts / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
