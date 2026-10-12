# General surface preprocessing performance

Measured 2026-10-12 on Windows 11, Python 3.12.0, NumPy 2.3.2,
ASE 3.24.0 and h5py 3.14.0. The baseline is a snapshot of the working
implementation immediately before this optimization, **including** the previous
conversion and exact molecular-topology caching improvements. It is different
from the earlier baseline in `conversion_results.md`.

The controlled three-run benchmark uses `Pt110_Na4_H_cell1-pos-1.xyz`: 8,010
frames and 604 atoms per frame (315,033,300 bytes). The larger
`Au111_K4-pos-1.xyz` and an existing converted trajectory were also located in
the workspace and are measured separately. Measurements below are actual
timings, not projections between differently sized inputs.

## Full conversion

Three unprofiled runs per implementation, each in a fresh process with identical
default options: surface cache and exact topology enabled, float32 position
storage, existing LZF/chunking/shuffle policies. Output paths were explicit and
unique, under `.pytest_surface_performance`. Before/after groups ran sequentially;
OS file caches were not flushed. A few short profiling/validation commands
overlapped some runs; no other full conversion or synthetic benchmark ran in
parallel. Timings are local measurements, with normal run-to-run variability.

| Measurement | Before median | After median |
|---|---:|---:|
| Total conversion | 204.01 s | 93.07 s |
| Read trajectory | 12.94 s | 12.25 s |
| Prepare cell/selection | 7.85 s | 8.21 s |
| Build molecular topology | 54.99 s | 54.41 s |
| Surface preparation | 115.66 s | 5.41 s |
| Surface top estimate | 3.47 s | 3.56 s |
| Surface bottom estimate | 3.59 s | 3.40 s |
| Surface populations | 5.22 s | 3.92 s |
| Complete surface cache | 128.04 s | 16.68 s |
| Write HDF5 | 2.54 s | 2.48 s |
| Peak resident memory | 632.91 MiB | 524.82 MiB |

Total runtime decreased **54.4%** (2.19x faster). Surface preparation decreased
**95.3%** (21.38x faster); complete surface-cache construction is 7.68x faster.
Median peak memory decreased **108.09 MiB / 17.1%**. Individual stage medians
need not sum to the median total. Memory is Windows PeakWorkingSetSize and
includes imports, Python objects and native allocations.

| Run | Before total | After total | Before peak | After peak |
|---|---:|---:|---:|---:|
| 1 | 218.01 s | 96.17 s | 632.31 MiB | 524.82 MiB |
| 2 | 200.99 s | 90.74 s | 633.64 MiB | 524.43 MiB |
| 3 | 204.01 s | 93.07 s | 632.91 MiB | 526.33 MiB |

`compare_conversion.py` compares every dataset and every non-timestamp attribute,
including shapes, dtypes, chunks, compression and shuffle. **All three output
pairs match exactly across all 59 datasets**, including positions, labels,
indices, PBC/cells, topology states/fingerprints, surface choices, confidence and
population diagnostics. All six files are 44,066,508 bytes.

## Reported large input

The workspace's `data/Au111_K4/Au111_K4-pos-1.xyz` matches the reported size:
**23,323 frames x 865 atoms**, 1,312,968,285 bytes. One additional unprofiled
optimized conversion ran in a fresh process, with the same default conversion
options and a separate output file. No benchmark or test ran concurrently with
this conversion.

| Stage | User-reported earlier run | Measured optimized run |
|---|---:|---:|
| Total | approximately 963 s | 376.48 s |
| Read trajectory | 26.82 s | 64.78 s |
| Prepare cell/selection | 8.37 s | 30.83 s |
| Molecular topology | 64.16 s | 221.05 s |
| Surface preparation | 776.77 s | 19.98 s |
| Surface top estimate | 20.06 s | 11.86 s |
| Surface bottom estimate | 22.22 s | 8.92 s |
| Surface populations | 28.28 s | 10.79 s |
| Write HDF5 | 14.95 s | 7.07 s |

This comparison is **observational**, not a controlled before/after experiment:
the earlier times come from the user's report, not a rerun of the baseline in
the current environment. Read/preparation/topology are slower than that earlier
report, so changes in machine conditions cannot be separated from implementation
effects using this single run. The controlled Pt110 comparison above establishes
the optimization's runtime and memory improvement against the current code.

The large optimized run used **1,913.37 MiB peak resident memory**. No earlier
peak-memory measurement is available for this input. Its complete surface-cache
construction took 51.83 s. Its output matches the existing
`data/Au111_K4/LiNaK_outputs/Au111_K4.traj.h5` **exactly across all 59 datasets and
all non-timestamp attributes**, including storage policies. Both files are
157,910,376 bytes. The existing source and converted trajectory were left intact.

## Bottlenecks and implementation

The original `_canonical_surface_frames()` built a complete surface context to
choose periodic cuts. `_build_surface_context()` resolved automatic elements,
unwrapped candidate trajectories and allocated translated 3D coordinates even
though cut selection needed only candidate axis values. A second full context
then repeated that work in canonical coordinates. Raw and canonical automatic
element detection cannot generally be conflated: changed cuts, cells or candidate
constraints can change mobility. The new cut-only preparation skips the unused
candidate geometry, while keeping both automatic element resolutions.

`unwrap_positions()` called ASE `find_mic()` once per frame, repeating cell
validation, coordinate solves and, when necessary, cell reduction/lattice search.
It now groups consecutive **exactly equal** cells and PBC in bounded batches.
Exactly diagonal cells use a separable numerical kernel for arbitrary mobile
coordinates, with ASE handling half-cell image ties. Skewed cells still use
ASE's nearest-image lattice search, not independent fractional rounding.
Changing cells use the previous frame's cell and PBC for each displacement, as
before. MIC scratch blocks contain at most 8,192 vectors, including a single
frame with more atoms. Cumulative reconstruction keeps the original sequential
addition order across chunks. No persistent cache or new dependency is involved.

Surface preparation now compares atomic-number arrays instead of repeatedly
constructing symbol strings. It shares layout validation with element resolution.
Canonical snapshots share unchanged metadata, constraint objects and atom arrays,
and own their transformed positions; the private estimator reads these views
without modifying original coordinates or metadata.

Side-independent translated mobility and axis medians are calculated once per
prepared context. Both sides retain independent selection, confidence, validation
and fallback decisions. Temporal mobility uses atom blocks with an 8 MiB coordinate
scratch target (at least one atom's temporal coordinates); the numerical operation
is identical to the original whole-array reduction. Large 3D arrays are released
after preparation instead of retained through both estimates.

Surface populations use the shared exact topology resolver for every frame,
including changing composition/order. An already prepared profile needs zero
geometric O/H detection calls in population analysis. Centers and distances still
come from current geometry. No topology sampling, fixed layer tracking, skipped
surface-cache work, parallel processing or scientific-definition changes were
introduced.

A diagnostic cProfile comparison on the first 256 frames of the existing cached
Pt110 trajectory (not an unprofiled speed benchmark) recorded:

| Function/work | Before | After |
|---|---:|---:|
| Total profiled surface call | 0.729 s | 0.318 s |
| Both context builds | 0.426 s | 0.064 s |
| ASE `find_mic` calls | 1,532 | 512 |
| Python function calls, including primitives | 1,398,298 | 415,386 |

The remaining 512 MIC calls are the two water bond-vector calculations per frame;
the three trajectory-unwrapping passes no longer call ASE MIC on this verified
diagonal input. Profiling and full-run speedups differ because trajectory length,
crossings, numerical kernels, memory and profiling overhead differ.

## General systems and scaling

Generated fixtures use vibrating/mobile atoms, constraints, independent fluid
environments, changing O/H bonds, periodic crossings and reconstruction. Each
case has 512 frames and 198 atoms (200 for reactive water; variable-count frames
alternate 198/197). Topology is prepared before timing surface work, matching
conversion. Below are medians of three unprofiled surface calls in one fresh
process per implementation; both processes use identical fixtures/options.

| System | Before | After |
|---|---:|---:|
| Fully fixed, explicitly included surface | 0.627 s | 0.312 s |
| Mobile orthorhombic | 2.178 s | 0.311 s |
| Partially constrained | 2.090 s | 0.783 s |
| Constant triclinic | 2.244 s | 0.891 s |
| Changing triclinic | 2.277 s | 1.161 s |
| Symmetric interfaces | 2.269 s | 0.986 s |
| Asymmetric, single fluid interface | 2.124 s | 0.854 s |
| Periodic surface crossings | 1.603 s | 0.865 s |
| Reconstructing surface | 2.113 s | 0.803 s |
| Reactive/ambiguous O-H connectivity | 2.297 s | 1.058 s |
| Variable atom count | 1.444 s | 1.236 s |
| Changing atom order | 1.531 s | 1.239 s |
| Unsupported periodic surface axis | 2.134 s | 1.057 s |
| Missing cell, nonperiodic fallback | 0.978 s | 0.600 s |
| Surface axis X | 2.276 s | 0.386 s |
| Surface axis Y | 2.299 s | 0.370 s |

Peak resident memory over the entire 16-case fixture process was 119.32 MiB
before and 120.31 MiB after (within 1 MiB); this includes fixture construction,
topology preparation and repeated cases. It does not indicate a comparable
increase in full conversion memory, which decreased by 108 MiB.

Additional three-call medians vary frame count and atom count independently:

| Frames x atoms | Mobile before/after | Constant triclinic before/after | Changing triclinic before/after |
|---|---:|---:|---:|
| 128 x 198 | 0.139 / 0.075 s | 0.139 / 0.081 s | 0.140 / 0.103 s |
| 512 x 54 | 0.645 / 0.305 s | 1.713 / 0.289 s | 1.535 / 1.166 s |
| 512 x 198 | 2.178 / 0.311 s | 2.244 / 0.891 s | 2.277 / 1.161 s |

These measurements show workload-dependent gains, not a linear scaling promise.
Changing cell geometries still require separate MIC calculations; common-context
and memory improvements benefit them without assuming a constant lattice.

## Validation and remaining work

1,254 relevant tests pass: 540 conversion/trajectory/scientific-analysis tests,
381 CLI tests, and 333 plotting/GUI tests including offscreen Qt integration.
Targeted Ruff formatting/lint and mypy checks pass.

The new regression suite includes complete baseline `SurfaceReference` fixtures
for all 16 systems: positions, cut/period, top/bottom estimates, candidate indices,
confidence, effective options, provenance, rejection/fill diagnostics, populations
and layout reasons. Numerical comparisons use an absolute 2e-10 tolerance;
integer/string/selection decisions are exact. Original input positions are checked
for mutation. Constraint changes verify cut-only and full preparation select the
same candidates. Reactive and variable-layout populations reuse prepared topology
without detection calls.

MIC tests independently run the original sequential ASE implementation across
diagonal, signed, rotated, strongly skewed and changing cells, full/partial/no
PBC, changing PBC, half-cell ties, empty/single-frame inputs and invalid geometry.
They also verify bounded work on 20,000-atom frames. Existing tests cover aliases,
conversion selections, spatial filtering, storage precision, interrupted writes,
CLI stages/quiet output, scientific analyses and GUI integration.

Cold exact topology building now dominates full conversion (~54 s for Pt110,
221 s in the additional large-input run). Per-frame
top/bottom layer selection and validation remain independent (~3.5 s each).
Further opportunities include cold connectivity kernels, shared layer sorting
with an explicit memory tradeoff, and reusing lattice reduction beyond each MIC
batch. Those are not required to obtain the measured gains above.

Cartesian surface-axis conventions and existing supported/unsupported geometry
decisions are preserved. This change does not extend surface-normal definitions
to new tilted-cell orientations. Unsupported geometry follows the existing safe
fallback. In-memory frames are still required for exact temporal surface analysis.
Linux and other Python/NumPy/ASE versions were not benchmarked in this environment;
the implementation uses portable existing APIs and no platform-specific kernels.

## Reproduction

Keep a pre-change `linak` package snapshot under `ARTIFACTS/baseline_src/linak`.

```powershell
python benchmarks/conversion_performance.py SOURCE.xyz ARTIFACTS --label before
python benchmarks/conversion_performance.py SOURCE.xyz ARTIFACTS --label after
python benchmarks/compare_conversion.py ARTIFACTS/before_1.traj.h5 ARTIFACTS/after_1.traj.h5
python benchmarks/surface_performance.py --baseline ARTIFACTS/baseline_src --frames 512 --grid 8 --runs 3 --output ARTIFACTS/systems_before.json
python benchmarks/surface_performance.py --frames 512 --grid 8 --runs 3 --output ARTIFACTS/systems_after.json
```

For profiling, add `--profile PROFILE_PATH`. To capture complete reference
fixtures, add `--reference JSON_PATH --frames 9 --grid 4 --runs 1`. Golden tests
use the committed pre-optimization reference file under `tests/fixtures`.
