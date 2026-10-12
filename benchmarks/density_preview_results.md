# Density preview projection

Measured on the supplied `data/Au111_K4/LiNaK_outputs/Au111_K4.density.h5`,
species K, original 0.05 Å bins. The X/Distance target has 358 × 1207 bins
and 21,403 exact geometry groups.

X/Y combines groups with identical cells and origins; fluctuating surface
heights do not affect that projection. X/Distance retains distinct surface
heights and canonical periodic geometry. Previously each group calculated
normal overlaps in a Python loop and retained its own dense volume grid.
Those volume grids alone would occupy about 68.9 GiB for this file.

Prismatic cells now use compact in-plane area and normal-overlap factors,
vectorized exact periodic distance overlaps, and sparse density normalization.
Coverage still includes empty bins. General polyhedral fallback volume
caching is limited to 16 MiB. No surface geometry is rounded or combined
approximately, and the stored density format is unchanged.

## Measurements

Three fresh processes, sequential runs:

| Run | Load (s) | X/Distance projection (s) | Total (s) | Peak resident memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.50 | 13.43 | 17.93 | 1047.20 |
| 2 | 3.75 | 21.05 | 24.80 | 1047.32 |
| 3 | 7.17 | 39.00 | 46.17 | 1047.03 |
| Median | 4.50 | 21.05 | 24.80 | 1047.20 |

Runtime varied noticeably on this workstation. These times cover loading
and projection, excluding Matplotlib drawing and GUI scheduling.

Reproduce one run with:

```powershell
python benchmarks/density_preview_performance.py data/Au111_K4/LiNaK_outputs/Au111_K4.density.h5 --species K
```

A separate comparison of the first 100 groups took 1.903 s with the old
implementation and 0.084 s with the new implementation. Mass density,
number density and coverage matched at `rtol=1e-12, atol=1e-12`.
A complete old X/Distance run was not attempted because of the demonstrated
69 GiB volume-cache demand. A full new X/Y projection took 5.48 s in that
comparison process; X/Distance took 14.22 s.

## Validation

914 density, sparse accumulation, binning, surface, triclinic, CLI, plotting,
GUI and cancellation regression tests passed. The expanded targeted suite
has 49 passing tests, comparing vector overlaps against scalar calculations
for both single-interface layouts, symmetric layouts, periodic wrapping,
filters, fine and full-period bins, and empty ranges; comparing sparse
normalization with dense normalization including duplicate projected bins;
and checking bounded fallback caching. Ruff and targeted mypy checks passed.
