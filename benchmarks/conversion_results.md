# CP2K XYZ conversion benchmark

Measured 2026-10-08 on the supplied `Pt110_Na4_H_cell1-pos-1.xyz`: 315,033,300 bytes, 8,010 frames, 604 atoms/frame.

Environment: Windows-11-10.0.26200-SP0, Python 3.12.0, NumPy 2.3.2, h5py 3.14.0, ASE 3.24.0.

Three sequential, unprofiled conversions in fresh processes for each implementation. Identical default options: surface cache enabled, fast/LZF compression, original storage precision. Explicit output paths under an ignored benchmark directory. No tests ran concurrently. The baseline used a source snapshot taken before these conversion changes. OS file caches were not flushed; timings are local measurements and vary between runs.

| Measurement | Before median | After median |
|---|---:|---:|
| Total | 109.58 s | 67.23 s |
| Read trajectory | 10.16 s | 7.65 s |
| PBC wrapping (part of preparation) | 1.55 s | 1.35 s |
| Build surface cache | 85.62 s | 54.40 s |
| Write HDF5 | 3.58 s | 1.15 s |
| Reread output for summary | 4.81 s | 0.00 s |
| Peak resident memory | 635.25 MiB | 629.14 MiB |

Total runtime fell 38.6%; median peak resident memory fell 6.11 MiB. The baseline maximum peak was 670.15 MiB; the optimized maximum was 629.33 MiB. Peak memory is Windows PeakWorkingSetSize (includes imported dependencies and native allocations).

| Run | Before total (s) | After total (s) | Before peak (MiB) | After peak (MiB) |
|---|---:|---:|---:|---:|
| 1 | 115.07 | 69.39 | 670.15 | 629.33 |
| 2 | 109.58 | 64.78 | 634.31 | 629.14 |
| 3 | 95.86 | 67.23 | 635.25 | 628.02 |

The new stage logs report preparation at 3.39 s median, including PBC wrapping. Surface preparation/top/bottom/population median times are 49.30/1.51/1.37/2.29 s respectively (see retained logs for individual runs). The baseline did not expose these internal stage timings.

Validation: `compare_conversion.py` found all 41 datasets and all non-timestamp attributes exactly equal between baseline and optimized output, including surface decisions, labels, indices, PBC and positions. Dataset shapes, dtypes, chunk sizes, compression and shuffle settings also match. HDF5 container size changed by 48 bytes due to allocation order; scientific contents are identical.

Regression tests enforce a single XYZ payload pass, cached label resolution, no output reread, bounded position allocation and incremental writes, and atomic output preservation on interruption. Surface context reuse is compared against independent top/bottom estimates, including fixed atoms and periodic boundary crossing.

Reproduce using `python benchmarks/conversion_performance.py SOURCE ARTIFACTS --label after`. For the baseline, put the original `linak` package under `ARTIFACTS/baseline_src/linak` and use `--label before`. Compare files with `python benchmarks/compare_conversion.py BEFORE.traj.h5 AFTER.traj.h5`.
