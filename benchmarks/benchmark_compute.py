"""Run a small, reproducible compute scaling test inside one CPU allocation.

Example: cpu -c 48 python benchmarks/benchmark_compute.py orientation run.traj.h5
Extra analysis arguments follow --. Run from the same environment as linak.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import hashlib
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def _child(arguments):
    # Measure actual fresh-process CLI behavior, including its normal cache use.
    from linak.cli import main

    return main(arguments)


VOLATILE = {"created_utc", "profile_uid", "linak_version"}


def _scientific(value):
    if isinstance(value, dict):
        return {key: _scientific(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, list):
        return [_scientific(item) for item in value]
    return value


def _compare_value(left, right, label):
    import numpy as np

    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), label
        for key in left:
            _compare_value(left[key], right[key], f"{label}/{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), label
        for a, b in zip(left, right):
            _compare_value(a, b, label)
    elif isinstance(left, float) or isinstance(right, float):
        np.testing.assert_allclose(
            left, right, rtol=5e-12, atol=1e-30, equal_nan=True, err_msg=label
        )
    else:
        np.testing.assert_equal(left, right, err_msg=label)


def compare_results(before, after):
    """Compare all scientific attributes and datasets, using bounded reads."""
    import h5py
    import numpy as np

    with h5py.File(before, "r") as left, h5py.File(after, "r") as right:
        names_a: list[str] = []
        names_b: list[str] = []
        left.visit(names_a.append)
        right.visit(names_b.append)
        assert names_a == names_b, "HDF5 object names differ"
        for name in ["/", *names_a]:
            a, b = left[name], right[name]
            assert set(a.attrs) == set(b.attrs), f"{name}: attribute names differ"
            for key in a.attrs:
                if key in VOLATILE:
                    continue
                aa, bb = a.attrs[key], b.attrs[key]
                if key.endswith("_json"):
                    aa, bb = _scientific(json.loads(aa)), _scientific(json.loads(bb))
                _compare_value(aa, bb, f"{name}:{key}")
            if not isinstance(a, h5py.Dataset):
                continue
            assert (a.shape, a.dtype) == (b.shape, b.dtype), name
            for start in range(0, len(a), 65536) if a.ndim else [0]:
                aa = a[start : start + 65536] if a.ndim else a[()]
                bb = b[start : start + 65536] if b.ndim else b[()]
                if a.dtype.kind in "fc":
                    finite = np.asarray(aa)[np.isfinite(aa)]
                    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
                    tolerance = 2e-6 if a.dtype.itemsize <= 4 else 5e-12
                    np.testing.assert_allclose(
                        aa,
                        bb,
                        rtol=tolerance,
                        atol=max(1e-30, scale * tolerance * 0.1),
                        equal_nan=True,
                        err_msg=name,
                    )
                else:
                    np.testing.assert_array_equal(aa, bb, err_msg=name)


def _rss_tree(pid):
    """Linux process-tree RSS; shared pages may be counted more than once."""
    total = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        total += sum(_rss_tree(int(child)) for child in children)
    except (OSError, ValueError):
        pass
    return total


def _run(arguments, log):
    started = time.perf_counter()
    last_update = started
    peak = 0
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                peak = max(peak, _rss_tree(process.pid))
                now = time.perf_counter()
                if now - last_update >= 10:
                    print(f"{log.stem}: running, {now - started:.0f} s elapsed", flush=True)
                    last_update = now
                time.sleep(0.05)
        except BaseException:
            process.terminate()
            process.wait()
            raise
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(f"Compute failed ({process.returncode}). See {log}")
    report = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "Compute execution report: "
        if marker in line:
            report = json.loads(line.split(marker, 1)[1])
    if report is None:
        raise RuntimeError(f"Missing execution report. See {log}")
    return dict(total_seconds=elapsed, peak_rss_mib=peak / 1024**2 if peak else None, **report)


def _summary(records):
    by_limit = defaultdict(list)
    for record in records:
        if record.get("phase") == "measured":
            by_limit[record["cpu_limit"]].append(record)
    baseline = statistics.median(record["total_seconds"] for record in by_limit[1])
    summary = []
    for limit, runs in sorted(by_limit.items()):
        stages = sorted(
            {f"{stage['stage']}={stage['workers']}" for r in runs for stage in r["stages"]}
        )
        timings = {
            key: statistics.median(r["timings"].get(key, 0.0) for r in runs)
            for key in {key for r in runs for key in r["timings"]}
        }
        total = statistics.median(r["total_seconds"] for r in runs)
        memory = [r["peak_rss_mib"] for r in runs if r["peak_rss_mib"] is not None]
        summary.append(
            dict(
                cpu_limit=limit,
                total_seconds=total,
                speedup=baseline / total,
                min_seconds=min(r["total_seconds"] for r in runs),
                max_seconds=max(r["total_seconds"] for r in runs),
                peak_rss_mib=max(memory) if memory else None,
                timings=timings,
                stages=stages,
                results_match=all(r["results_match"] for r in runs),
            )
        )
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--child"]:
        return _child(argv[1:])
    extra = []
    if "--" in argv:
        index = argv.index("--")
        extra, argv = argv[index + 1 :], argv[:index]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis",
        choices=[
            "density",
            "orientation",
            "coordination",
            "rdf",
            "potential",
            "position",
            "temperature",
        ],
    )
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--cpu-counts", default="1,8,48")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    limits = sorted({int(item) for item in args.cpu_counts.split(",")})
    if not limits or min(limits) < 1 or args.repeats < 1:
        parser.error("CPU counts and repeats must be positive.")
    if any(
        token.split("=")[0]
        in {"--threads", "--output", "-o", "--log-level", "--log-file", "--dry-run"}
        for token in extra
    ):
        parser.error(
            "The runner controls threads, output and logging; omit those analysis arguments."
        )
    if args.analysis != "potential" and len(args.sources) != 1:
        parser.error("Only potential accepts multiple sources.")
    sources = [Path(source).expanduser().resolve() for source in args.sources]
    if not all(source.is_file() for source in sources):
        parser.error("Every source must be an existing file.")
    # Importing resource detection does not initialize numerical libraries.
    from linak.runtime_resources import detect_cpu_count

    if os.getenv("LINAK_DISABLE_THREAD_CAP", "").strip().lower() in {"1", "true", "yes", "on"}:
        parser.error("Unset LINAK_DISABLE_THREAD_CAP so the benchmark can control native threads.")
    available = detect_cpu_count()
    skipped = [limit for limit in limits if limit > available]
    limits = sorted({1, *(limit for limit in limits if limit <= available)})
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="linak-compute-benchmark-"))
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        parser.error(f"Report directory must be empty: {directory}. Choose a new directory.")
    print(f"Report: {directory}", flush=True)
    if skipped:
        print(
            f"Skipping {skipped}: only {available} CPUs available; no relabelled results.",
            flush=True,
        )
    inputs = ["-f", *map(str, sources)] if args.analysis == "potential" else [str(sources[0])]
    records = []
    baseline = None
    try:
        for repetition in range(-1, args.repeats):
            order = [1] if repetition < 0 else (limits if repetition % 2 == 0 else limits[::-1])
            for limit in order:
                label = "warmup" if repetition < 0 else f"run-{repetition + 1}-cpu-{limit}"
                output = directory / f"{label}.{args.analysis}.h5"
                if output.exists():
                    raise FileExistsError(f"Refusing to replace an existing output: {output}")
                command = [
                    "--log-level",
                    "DEBUG",
                    "compute",
                    args.analysis,
                    *inputs,
                    *extra,
                    "--threads",
                    str(limit),
                    "--output",
                    str(output),
                ]
                print(f"{label}: starting", flush=True)
                record = _run(command, directory / f"{label}.log")
                if not output.is_file():
                    raise RuntimeError(f"Expected output missing: {output}")
                if repetition < 0:
                    baseline = output
                else:
                    compare_results(baseline, output)
                record.update(
                    cpu_limit=limit,
                    repetition=repetition + 1,
                    results_match=True,
                    phase="warmup" if repetition < 0 else "measured",
                    command=command,
                )
                records.append(record)
                print(f"{label}: {record['total_seconds']:.2f} s; results match", flush=True)
                if repetition >= 0:
                    output.unlink()
    finally:
        import linak

        implementation = hashlib.sha256()
        package = Path(linak.__file__).resolve().parent
        for source_file in sorted(package.rglob("*.py")):
            implementation.update(str(source_file.relative_to(package)).encode("utf-8"))
            implementation.update(source_file.read_bytes())
        payload = dict(
            analysis=args.analysis,
            linak_version=linak.__version__,
            implementation_sha256=implementation.hexdigest(),
            processor=platform.processor(),
            sources=[
                dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                for p in sources
            ],
            arguments=extra,
            host=platform.node(),
            platform=platform.platform(),
            python=platform.python_version(),
            allocated_cpus=available,
            slurm={
                k: v
                for k, v in os.environ.items()
                if k
                in {
                    "SLURM_CPUS_PER_TASK",
                    "SLURM_JOB_ID",
                    "SLURM_JOB_PARTITION",
                    "SLURM_CPU_BIND_LIST",
                }
            },
            skipped_limits=skipped,
            memory_note="Sampled summed process-tree RSS on Linux; shared pages may count twice; unavailable elsewhere.",
            records=records,
        )
        (directory / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = _summary(records)
    if baseline is not None:
        baseline.unlink()
    lines = [
        f"# {args.analysis} compute benchmark",
        "",
        f"Host: {platform.node()}; available CPUs: {available}. One untimed warmup; {args.repeats} measured round(s).",
        "",
        "| CPU limit | Median total s (range) | Calculation s | Reduction s* | Loading s | Writing s | Speedup | Peak RSS MiB | Results |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        timings = row["timings"]
        peak = "-" if row["peak_rss_mib"] is None else f"{row['peak_rss_mib']:.0f}"

        def duration(name, timings=timings):
            return "-" if name not in timings else f"{timings[name]:.2f}"

        lines.append(
            f"| {row['cpu_limit']} | {row['total_seconds']:.2f} ({row['min_seconds']:.2f}-{row['max_seconds']:.2f}) | {duration('calculation')} | {duration('reduction')} | {duration('loading')} | {duration('writing')} | {row['speedup']:.2f}x | {peak} | match |"
        )
    lines.extend(["", "Actual stage workers:", ""])
    for row in summary:
        lines.append(f"- CPU limit {row['cpu_limit']}: " + ", ".join(row["stages"]))
    stage_names = sorted({key for row in summary for key in row["timings"] if "." in key})
    if stage_names:
        lines.extend(
            [
                "",
                "Stage wall time includes its reader/ordered consumer and is a subset of calculation:",
                "",
                "| Stage | "
                + " | ".join(str(row["cpu_limit"]) + " CPUs (s)" for row in summary)
                + " |",
                "| --- | " + " | ".join("---" for _row in summary) + " |",
            ]
        )
        for name in stage_names:
            lines.append(
                "| "
                + name
                + " | "
                + " | ".join(
                    "-" if name not in row["timings"] else f"{row['timings'][name]:.2f}"
                    for row in summary
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Total includes fresh-process imports/preparation. Loading reports eager trajectory loading; streamed reads overlap calculation. *Reduction is included in calculation time; timings can overlap and should not be added. Missing stages are not separately instrumented. Memory is sampled summed RSS, not unique physical memory. No defaults were changed.",
            "",
        ]
    )
    (directory / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
