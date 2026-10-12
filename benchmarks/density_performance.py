"""Fresh-process density timings, stage throughput, and peak resident memory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.conversion_performance import peak_rss_mib


@contextmanager
def prevent_idle_sleep():
    """Keep Windows benchmark processes awake without changing power settings."""
    restore = None
    if sys.platform == "win32":
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
        set_state.argtypes = (ctypes.c_uint,)
        set_state.restype = ctypes.c_uint
        if set_state(0x80000001):  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            restore = set_state
    try:
        yield
    finally:
        if restore is not None:
            restore(0x80000000)  # Clear this thread's transient sleep requirement.


def replace_hook(target, name, value):
    setattr(target, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.child is None:
        results = []
        for run in range(1, args.runs + 1):
            command = [
                sys.executable,
                __file__,
                str(args.source),
                str(args.artifacts),
                "--label",
                args.label,
                "--child",
                str(run),
            ]
            with (args.artifacts / f"{args.label}_{run}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            results.append(json.loads((args.artifacts / f"{args.label}_{run}.json").read_text()))
            print(json.dumps(results[-1]), flush=True)
        print(
            json.dumps(
                {
                    "medians": {
                        key: statistics.median(r[key] for r in results)
                        for key in ("total_s", "binning_s", "finalize_s", "peak_rss_mib")
                    }
                }
            )
        )
        return
    if args.label == "before":
        baseline = args.artifacts / "baseline_src"
        if not (baseline / "linak").is_dir():
            raise SystemExit(f"Missing baseline: {baseline}")
        sys.path.insert(0, str(baseline.resolve()))
    from linak import cli
    from linak.analysis import density

    timings = {}
    windows = []
    from linak.progress import ProgressBar

    class MeasuredBar(ProgressBar):
        def __enter__(self):
            self.measure_start = self.window_start = time.perf_counter()
            self.window_count = 0
            return super().__enter__()

        def update(self, n=1):
            super().update(n)
            if self.desc == "Binning density frames" and self.count - self.window_count >= 1000:
                now = time.perf_counter()
                windows.append(
                    {
                        "frame": self.count,
                        "frames_s": (self.count - self.window_count) / (now - self.window_start),
                    }
                )
                print(json.dumps({"benchmark_window": windows[-1]}), flush=True)
                self.window_count, self.window_start = self.count, now

        def close(self):
            if not self._closed and hasattr(self, "measure_start"):
                timings[self.desc] = time.perf_counter() - self.measure_start
            super().close()

    replace_hook(density, "ProgressBar", MeasuredBar)
    original_finalize = density._DensitySparseGridAccumulator.finalize
    final_time = 0.0

    def finalize(self, **kwargs):
        nonlocal final_time
        start = time.perf_counter()
        try:
            return original_finalize(self, **kwargs)
        finally:
            final_time += time.perf_counter() - start

    replace_hook(density._DensitySparseGridAccumulator, "finalize", finalize)
    workspace_metrics = {}
    if hasattr(density, "SparseWorkspace"):
        original_close = density.SparseWorkspace.close

        def close(self):
            workspace_metrics.update(self.metrics)
            return original_close(self)

        replace_hook(density.SparseWorkspace, "close", close)
    output = args.artifacts / f"{args.label}_{args.child}.density.h5"
    if output.exists():
        raise FileExistsError(f"Use a fresh benchmark output path: {output}")
    started = time.perf_counter()
    exit_code = cli.main(
        ["compute", "density", str(args.source), "--bin-width", "0.05", "--output", str(output)]
    )
    result = {
        "total_s": time.perf_counter() - started,
        "binning_s": timings.get("Binning density frames", 0.0)
        - (final_time if "Finalize density grids" not in timings else 0.0),
        "finalize_s": timings.get("Finalize density grids", final_time),
        "peak_rss_mib": peak_rss_mib(),
        "windows": windows,
        "workspace": workspace_metrics,
        "output_bytes": output.stat().st_size if output.exists() else 0,
        "exit_code": exit_code,
    }
    (args.artifacts / f"{args.label}_{args.child}.json").write_text(json.dumps(result, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    with prevent_idle_sleep():
        main()
