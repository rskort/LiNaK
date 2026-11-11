# Compute parallelism: implemented changes and benchmarks

Implemented on 2026-11-11. This page records the current implementation and the
measurements still needed to tune automatic defaults. Plotting is unchanged.

`cpu -c 48 linak compute ...` detects the allocation created by your wrapper.
It does not inspect the shell function itself. CPU availability is a ceiling;
each calculation stage can choose fewer workers, including one.

## Shared execution

| Stage | Implemented behavior / change |
| --- | --- |
| Inspect files and validate settings | Remains serial; all compute commands now accept `--threads` and `--memory-budget-mib`. `--dry-run` reports the allocation and budget. |
| Detect resources | CPU limits consider Slurm CPUs per task, affinity and cgroup quotas. Memory limits consider available memory and applicable job/cgroup limits, including existing usage. Explicit concurrency is clamped to CPU/memory limits. |
| Read inputs | The coordinator alone advances trajectory iterators. Bounded task submission overlaps reading with work; it no longer queues an entire trajectory's RDF payloads. Existing eager/streaming input modes remain available where previously supported. |
| Resolve substrate references | Global substrate, mobility, periodic-reference and temporal decisions remain serial. Independent layered-surface fits now support ordered threaded batches after that global preparation; rough estimation and temporal filling remain serial. |
| Resolve molecular membership | Reuses validated topology and shared selections. Worker tasks cannot upgrade/write source topology caches. Existing source-cache creation stays coordinated. |
| Manage workers | A common execution helper preserves input order, checks cancellation, bounds submissions to two tasks per worker, and suppresses nested pools. Batch sizes and distance tiles are memory-bounded. |
| Native threads | CLI bootstrap precedes numerical imports. Explicit backend environment values are preserved and missing values get defaults. Computation scopes limit native pools to one thread per outer worker and restore limits afterward; the existing expert opt-out remains available. |
| Merge and write | Original frame/block indices and scientific formulas are preserved. Ordered accumulation and one output writer remain; timing/worker diagnostics are runtime information, not new HDF5 fields. |

## Per-calculation changes

**Threads** share memory; **processes** have separate Python interpreters.
New frame kernels use threads. Processes remain in RDF's existing generic
backend; additional process backends were not introduced without evidence that
their startup/copying costs pay off. No grids, cutoffs or precision are reduced.

| Calculation | Ordered stages and implemented changes |
| --- | --- |
| **Density** | **1.** Selection/observed-bounds preparation supports threaded frame batches; molecule activity/persistence is reconciled in original order. **2.** Identical full geometry/volume calculations are reused, including across species with matching edges. **3.** Workers prepare selection, distances, line histograms and sparse rows. **4.** Element masks validate atom order on every frame, so selections no longer depend on batch size when atoms reorder. The coordinator applies per-frame normalization, statistics and sparse reductions before final interface folding. Nearest-surface frame work remains serial because its provider owns a mutable search cache. |
| **Orientation** | **1.** Water geometry and angle calculations support threaded frame batches. **2.** Bounds remain global; repeated volume geometry is reused. **3.** Spatial channels share one sparse grouping/reduction rather than repeatedly sorting each channel, with optional threaded bin preparation. **4.** Final mean/density/product calculations retain their existing definitions and equal interface weighting. Nearest-surface frame work remains coordinated. |
| **Coordination** | **1.** Cutoff RDF calculation inherits the execution budget; explicit/saved-cutoff interpretation stays serial. **2.** Molecular selections can be prepared once in shared frame batches. **3.** Orthorhombic distances support threaded frame batches and bounded center/neighbor tiles; general/molecular paths support threaded frame batches. **4.** Results are assembled by original frame and center identity. |
| **RDF** | **1.** Pair selections and backend planning remain shared. **2.** Existing orthorhombic threads use the common budget and lazy bounded payloads; molecular pair batches also support threads. **3.** Generic RDF retains processes, using spawn and transferring configuration once per worker. **4.** Generic results are reduced in original frame order; orthorhombic logical blocks are independent of worker count to stabilize statistics. Fixed automatic caps are now explicit, reviewable policies rather than separate resolver limits. |
| **Potential** | **1.** Existing file preparation uses bounded concurrent submission and the common budget; only reduced profiles are retained after preparation. **2.** A single reference plan is selected globally. **3.** Reduced-profile evaluation and provenance remain serial. **4.** Ordered callbacks and atomic writing remain centralized. |
| **Position** | **1.** Layout/identity validation remains serial. **2.** Atomic coordinate copies remain serial; no generic trajectory splitting was added. **3.** Large nearest-substrate queries can use budgeted SciPy threads, with no nested query pool. **4.** Ordered profile assembly and temporal continuity are preserved. |
| **Temperature** | **1.** Region/unit metadata and input parsing remain serial. **2.** Existing `.temp`/`.tregion` values are loaded directly. **3.** Velocity temperatures use batched NumPy arithmetic with optional threads; cached masses and per-frame/selection COM removal are preserved. **4.** Time/region profiles are assembled in original order. |

Implementation: [execution policies](../src/linak/parallel.py),
[resource detection](../src/linak/runtime_resources.py),
[benchmark runner](../benchmarks/benchmark_compute.py).

## Current automatic defaults

| Stage | Automatic worker ceiling | Status |
| --- | --- | --- |
| Atomic RDF distances | 4 | Previous conservative ceiling retained as a tunable policy. |
| New surface, density, orientation, coordination, molecular RDF, velocity-temperature and nearest-query paths | 1 | Parallel paths implemented and available explicitly; representative scaling evidence pending. |
| Potential file preparation | 1 | Previous conservative default retained pending file/storage benchmarks. |
| Metadata, planar position arithmetic, reference planning, finalization and output | 1 | Intentionally serial. |

`--threads 8` exercises supported parallel paths up to eight workers, still
bounded by tasks, memory, CPU availability and serial-only stages. It is the
**benchmark override**, not a promise of eight active workers everywhere.
Normal CLI use selects automatic policies; direct Python calls to newly extended
analysis APIs remain serial unless `threads` is supplied. Existing RDF/potential
API defaults remain compatible. `LINAK_RDF_THREADS` remains supported.

## Easy benchmark procedure

**Start with one analysis you actually use.** Use a packed trajectory whose usual
calculation takes about 1-5 minutes. Update the cluster environment to this
checkout first (including the new `threadpoolctl` dependency).

From the repository, in the same environment as `linak`:

```bash
cpu -c 48 python benchmarks/benchmark_compute.py orientation /path/run.traj.h5 --report-dir benchmark-O1
```

The runner performs one warmup and **six measured fresh-process runs**: CPU
limits 1, 8 and 48, twice, reversing their order in the second round. It uses
ordinary compute/cache behavior, including normal source topology-cache creation
when needed. Output files are isolated and removed after successful comparison;
logs, `results.md` and `results.json` remain. Failed comparisons retain their
files for inspection. Existing reports are not overwritten.

- Add your usual analysis options **after `--`**, for example
  `-- --surface-elements Pt --surface-layout symmetric --bin-width 0.5`.
- For a shorter initial check, add `--repeats 1` before `--`: three measured runs
  plus warmup. Two rounds are preferable before changing defaults.
- Do not pass output, thread or logging options after `--`; the runner controls
  them. Disable the expert `LINAK_DISABLE_THREAD_CAP` opt-out for controlled tests.
- Counts exceeding the actual allocation are skipped and identified, never
  reported as if that many CPUs had been tested.
- Reports include actual workers, time ranges, total/calculation/loading/writing
  times, per-stage wall times, reduction time where instrumented, and numerical output comparison.
  Reduction is included in calculation; overlapping timings are not additive.
  Peak summed process-tree RSS is sampled on Linux (shared pages may count twice);
  memory is marked unavailable on other platforms. Streamed reading overlaps work.

### Benchmark requests

Replace the analysis/source in the command above and use a different report
directory for each request. **O1 alone is enough to start; the other requests are
only useful for calculations you use.** Use identical scientific options across
CPU limits; the runner does this automatically.

| ID | Priority / calculation | Input and additional options | What the result decides |
| --- | --- | --- | --- |
| O1 | First choice: orientation | Your usual packed trajectory and orientation options. | Whether orientation geometry/binning should use more than one worker. |
| D1 | Next, if used: density | Same trajectory; usual species/outputs/grid widths. | Density selection/binning worker ceilings. |
| C1 | Next, if used: coordination | Usual pair/cutoff, e.g. `-- --species O H --cutoff 1.2`; use a separate run for automatic cutoff RDF if relevant. | Coordination distance/preparation workers. |
| R1 | Optional: RDF | Usual pair, e.g. `-- --species O O`. | Whether the four-worker automatic policy should change; geometry determines thread/process backend. |
| P1 | Optional: potential | Several representative cube/HDF5 inputs passed as separate source arguments. A single field cannot test file concurrency. | File-preparation workers and storage saturation. |
| S1 | Optional: position | A large nearest-atom workload with your usual distance options; ordinary XYZ extraction needs no scaling test. | Whether large nearest queries benefit from threads. |
| T1 | Optional: temperature | A substantial velocity XYZ file and usual region/COM options; `.temp`/`.tregion` need no scaling test. | Velocity-kernel workers. |

### Genoa results to fill in

Copy the medians and worker details from each `results.md`. A report path or
attached `results.json` supplies settings, input identity, software fingerprint,
individual timings and validation details. **Pending means not measured.**

| ID | Total seconds: 1 / 8 / 48 | Calculation seconds: 1 / 8 / 48 | Actual workers at best limit | Results match? | Report / notes | Default decision |
| --- | --- | --- | --- | --- | --- | --- |
| O1 | Pending | Pending | Pending | Pending | | Await results |
| D1 | Pending | Pending | Pending | Pending | | Await results |
| C1 | Pending | Pending | Pending | Pending | | Await results |
| R1 | Pending | Pending | Pending | Pending | | Await results |
| P1 | Pending | Pending | Pending | Pending | | Await results |
| S1 | Pending | Pending | Pending | Pending | | Await results |
| T1 | Pending | Pending | Pending | Pending | | Await results |

**After filling this table, tell me it is ready.** I will review correctness,
actual worker counts, per-stage times, memory and repeat variation, then update the corresponding
`AUTO_POLICIES` entries and this page. Prefer the smallest tested worker count
within 5% of the fastest; raise a default only for a repeatable gain of at least
10% over its current setting. No improvement means retaining the current default.
A result applies to its tested analysis/backend/workload, not every computation.
The Markdown table is a review record, not executable runtime configuration;
editing it alone does not silently change scientific runs.

### Local development results

A short Windows development fixture (1,024 frames, 604 atoms, taken from the local
packed trajectory; 14 available logical CPUs) checks the runner and numerical
agreement. It is **not a Genoa scaling result**. Two rounds at CPU limits 1, 4 and 8 were run for each analysis, without other
benchmark/test jobs alongside them. Values are median seconds; memory measurement
is unavailable on this Windows host. These do not establish Genoa defaults.

| Analysis / options | Total seconds: 1 / 4 / 8 | Calculation seconds: 1 / 4 / 8 | Numerical comparison | Automatic decision |
| --- | --- | --- | --- | --- |
| Orientation, symmetric Pt, 0.5 A bins | 5.10 / 6.23 / 6.16 | 2.55 / 3.65 / 3.65 | All six runs match | Keep 1; more threads slower here |
| Density O, symmetric Pt, 0.5 A line/grid bins | 4.39 / 5.02 / 5.12 | 1.83 / 2.52 / 2.61 | All six runs match | Keep 1; more threads slower here |
| Coordination O-H, cutoff 1.2 A, symmetric Pt | 9.73 / 8.16 / 7.83 | 7.31 / 5.76 / 5.40 | All six runs match | Keep conservative default pending Genoa; local 8 limit is 1.24x faster |

## Validation notes

The broad compute/CLI/geometry regression run passed 922 tests. The final
parallelism-specific suite passed all 20 tests, including real RDF processes,
serial/threaded equivalence, cancellation, allocation/memory limits, small-memory
distance tiling and native-thread restoration. All 18 measured local benchmark
outputs matched their serial reference. Changed files pass Ruff; the new
execution/resource modules and benchmark runner pass their targeted type checks.

The combined regression run encountered seven Qt DLL-load errors in GUI checks;
an affected GUI check passed in a fresh process. The 32 older density metadata-baseline cases were excluded from that run.
Separately, all 30 historical scientific-array comparisons for the other 15
fixtures passed at normal and spilling budgets. The reordered fixture contains
the old stale-mask behavior; it is now checked against direct per-element
histograms at different memory budgets instead. Density from reordered-atom
trajectories may consequently differ from older calculations; existing result
files are not rewritten. These limitations are
separate from the passing computation comparisons and are not recorded as passes.
Genoa/48-CPU performance remains unmeasured until the requests above are run.
