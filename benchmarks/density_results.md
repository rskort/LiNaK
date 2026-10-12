# Bounded density accumulation

Density accumulation now uses numeric sorted runs instead of repeatedly sorting
the entire accumulated grid and retaining geometry-bin entries in Python
dictionaries. Runs are compacted with NumPy and combined in a balanced reduction.
Exact geometry descriptors and frame counts are prepared once and shared across
species. Bin edges, bounds, widths, tolerances and strides are prepared once per
accumulator; entity bin indices are reused for total, geometry and interface
channels. Interface ties still contribute half to each side.

The workspace is shared across species and is limited to the smaller of 256 MiB
and one eighth of the existing command memory budget. Pending rows are limited to
one quarter of that workspace, capped at 8 MiB. Resident runs and bounded merge
scratch are accounted for before allocation. Larger runs use a command-owned
temporary directory, removed on success, exceptions or interruption. Explicit
`LINAK_MEMORY_BUDGET_MIB` settings remain authoritative. An unusably small budget
produces an actionable error rather than silently exceeding it. Input frames and
the scientifically required returned profile arrays are outside the accumulation
workspace; total process memory is therefore larger than the workspace limit.

Returned arrays are filled in bounded blocks without materializing a second full
structured run. The trajectory, scientific selection rules, density HDF5 schema,
storage precision, compression defaults and filenames are unchanged. Final grid
reduction has its own timed **Finalize density grids** stage after **Binning
density frames**. One-dimensional-only computations do not create a sparse
workspace or report a grid-finalization stage.

## Full-trajectory measurements

Windows measurements use the complete cached trajectories and three fresh,
unprofiled processes per implementation. Options are identical: all default
species and outputs, 0.05 Angstrom bins, unchanged surface and topology settings.
The environment is Windows 11, Python 3.12.0, NumPy 2.3.2, ASE 3.24.0 and h5py
3.14.0.
Before/after groups run sequentially. OS file caches are not flushed. Short
regression checks overlapped portions of the first two Au111 baseline runs; no
other full benchmark ran concurrently. Timings vary with system activity, so
individual runs are included below as well as medians. Even the fastest Au111
baseline binning run is slower than the slowest usable new run.

An initial out-of-memory attempt while an unrelated process occupied about 20 GiB
was excluded. The user freed resources before the recorded baseline group. One
new-code Au111 run was interrupted by an apparent machine suspension: a single
1,000-frame window took about 21 minutes with little CPU activity. Its files were
retained under `suspended_after_2` and its timing excluded; a fresh replacement
provides the third usable run. Subsequent helper processes temporarily prevent
idle sleep on Windows and release that requirement on exit, without changing
power settings.

| Trajectory | Metric | Before median | After median |
| --- | --- | ---: | ---: |
| Au111, 23,323 frames × 865 atoms | Binning | 410.20 s | 129.91 s |
| Au111 | Finalization | 6.94 s | 6.01 s |
| Au111 | Total command | 558.94 s | 286.78 s |
| Au111 | Peak resident memory | 15,468 MiB | 3,663 MiB |
| Pt110, 8,010 frames × 604 atoms | Binning | 76.13 s | 35.17 s |
| Pt110 | Finalization | 1.89 s | 1.89 s |
| Pt110 | Total command | 118.79 s | 76.54 s |
| Pt110 | Peak resident memory | 4,506 MiB | 1,321 MiB |

Au111 binning is 3.16× faster; total runtime is 1.95× faster. Peak resident memory
falls by 76.3%. Loading, topology/cache validation, geometry preparation and output
writing remain part of total runtime. Their cost is not represented by the
binning stage alone. Before-code finalization was inside the original binning
progress context; the helper times sparse finalization separately and subtracts
it from the before binning measurement.

| Au111 run | Total before/after (s) | Binning before/after (s) | Finalize before/after (s) | Peak before/after (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 659.90 / 291.04 | 422.72 / 109.58 | 6.94 / 5.81 | 15,508 / 3,664 |
| 2 | 558.94 / 267.67 | 410.20 / 129.91 | 6.95 / 6.01 | 15,468 / 3,663 |
| 3 | 486.29 / 286.78 | 316.01 / 131.90 | 6.54 / 7.17 | 15,468 / 3,588 |

Every usable Au111 new run used the same 256 MiB workspace: peak pending rows
8 MiB, peak resident runs 126.0 MiB, peak live temporary files 1,140.7 MiB, and
4,815.0 MiB written to temporary runs over the command. Temporary files are
removed before returning profiles. All three output comparisons passed across
725 datasets per file. Output sizes remain about 194 MiB; minor compressed-size
differences reflect summation order and random profile identifiers.

Median frame rates for successive 1,000-frame windows show the old accumulated
history slowdown and bounded merge fluctuations in the new implementation:

| Window ending at frame | Before frames/s | After frames/s |
| ---: | ---: | ---: |
| 1,000 | 128.6 | 232.9 |
| 2,000 | 125.1 | 211.2 |
| 3,000 | 108.7 | 192.0 |
| 4,000 | 78.2 | 192.2 |
| 5,000 | 98.6 | 203.7 |
| 6,000 | 95.6 | 193.1 |
| 7,000 | 96.8 | 211.9 |
| 8,000 | 93.8 | 139.6 |
| 9,000 | 89.3 | 207.4 |
| 10,000 | 85.0 | 220.4 |
| 11,000 | 78.1 | 181.5 |
| 12,000 | 62.3 | 218.6 |
| 13,000 | 61.9 | 216.4 |
| 14,000 | 68.1 | 238.9 |
| 15,000 | 66.5 | 116.8 |
| 16,000 | 68.0 | 226.5 |
| 17,000 | 60.9 | 207.9 |
| 18,000 | 58.8 | 160.8 |
| 19,000 | 53.5 | 141.7 |
| 20,000 | 55.8 | 218.7 |
| 21,000 | 48.6 | 238.9 |
| 22,000 | 52.1 | 154.4 |
| 23,000 | 46.4 | 226.7 |

Pt110 binning is 2.16× faster, total runtime is 1.55× faster, and peak resident
memory falls by 70.7%. The three before runs total 147.33, 118.22 and 118.79 s;
the three after runs total 76.54, 76.84 and 75.05 s. Binning takes 77.28, 73.93
and 76.13 s before, versus 35.01, 35.56 and 35.17 s after. Finalization takes
1.89, 1.89 and 1.78 s before, versus 1.89, 1.88 and 1.89 s after. Peak resident
memory is 4,509, 4,495 and 4,506 MiB before, versus 1,276, 1,321 and 1,321 MiB
after. No tests ran concurrently with these six Pt110 measurements.

The Pt110 workspace is 256 MiB, with peak pending rows 8 MiB, resident runs
125.6 MiB, live temporary storage 287.6 MiB and cumulative temporary writes
674.8 MiB. All three full-file comparisons passed across 870 datasets per file,
including all 30 output profiles. Output sizes remain about 56.4 MiB.

| Pt110 window ending at frame | Before frames/s | After frames/s |
| ---: | ---: | ---: |
| 1,000 | 130.8 | 245.7 |
| 2,000 | 122.2 | 229.6 |
| 3,000 | 116.5 | 220.9 |
| 4,000 | 108.0 | 228.2 |
| 5,000 | 104.8 | 207.6 |
| 6,000 | 100.3 | 232.9 |
| 7,000 | 99.2 | 234.5 |
| 8,000 | 87.7 | 215.4 |

These results meet the acceptance criteria for both supplied trajectories:
lower median binning and total runtime, lower peak resident memory, equivalent
scientific outputs, and removal of repeated growing-history sorting and per-bin
Python geometry dictionaries. Balanced merges still do measurable work and may
cause individual throughput dips. Runtime depends on trajectory contents, bin
width, available memory, temporary disk speed and other system activity; no fixed
frames-per-second rate is guaranteed. Linux and other dependency versions were
not benchmarked here.

## Validation

The combined regression run passed **1,012 tests in 55.91 s**, covering density,
binning, trajectory I/O, conversion, topology, surfaces, CLI, progress, plotting,
export and the plot GUI. Targeted Ruff checks and formatting checks pass; mypy
reports no issues in the eight changed Python source/test files. Warning assertion
tests explicitly select their logging level so earlier CLI error-level tests
cannot suppress the expected messages. Product logging behavior is unchanged.

The regression suite compares all density datasets and scientific metadata against
pre-change results for 16 systems: fixed, mobile and partial constraints;
triclinic and changing cells; symmetric interfaces; aliases; reactive and variable
topology; wrapped surfaces; unsupported or missing geometry; and alternate surface
axes. Every case is repeated with a forced 16 KiB workspace to exercise disk spill.
Additional tests cover empty geometry groups, interface tie weights, unsigned bin
keys near the integer limit, shared budgets, overlapping rows at the nonzero-bin
limit, duplicates across read blocks, failed temporary writes and interruption
cleanup. Structural checks prove invariant preparation is shared and the old
growing-prefix merges and per-bin dictionaries have been removed.

Full-file comparisons use bounded HDF5 reads. Integer arrays, labels, indices,
geometry identity/order/frame counts, scientific metadata and storage policies
must match exactly. Random profile identifiers and creation timestamps are
excluded. The non-scientific `linak_version` writer attribute is also excluded:
the copied baseline package uses its fallback version rather than discovering the
workspace's project metadata. Format/schema versions still must match exactly.
Floating-point datasets use relative tolerances of 2e-6 for float32 and
5e-12 for float64, with an absolute tolerance of 1e-30, allowing only small changes
from the reduction order.

## Reproduction

Save the pre-change `linak` package under `ARTIFACTS/baseline_src/linak`, then run:

```powershell
python benchmarks/density_performance.py SOURCE.traj.h5 ARTIFACTS --label before
python benchmarks/density_performance.py SOURCE.traj.h5 ARTIFACTS --label after
python benchmarks/compare_density.py ARTIFACTS/before_1.density.h5 ARTIFACTS/after_1.density.h5
```

Each group runs three fresh processes sequentially with identical options:
`compute density SOURCE --bin-width 0.05 --output UNIQUE_OUTPUT`. Existing source
and result files are preserved. The helper records stage times, 1,000-frame
throughput windows, peak resident memory and temporary-storage metrics. Repeat
the comparison for runs 2 and 3. Benchmark artifacts remain in ignored directories;
no temporary accumulation files are retained.
