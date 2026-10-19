# Physical density and nearest-distance distributions

Surface-plane and Cartesian profiles retain their physical-volume normalization:
mass density in **g/cm³**, number density in **entities/nm³**, and the existing
atom/molecule conventions. Surface identification, periodic boundaries and
interface pooling are unchanged.

Nearest surface-atom distance uses ordinary histograms, without geometric volume
integration. For each bin, divide the accumulated population or mass by the
number of frames and the bin width. Molecular centers carry the complete
molecular mass. Integrating a profile therefore recovers the average population
or mass in the selected domain.

| Observable | 1D | 2D |
|---|---|---|
| Atomic number distribution | atoms/Å | atoms/Å² |
| Molecular number distribution | molecules/Å | molecules/Å² |
| Mass distribution | amu/Å | amu/Å² |

Both weighting channels are calculated together. A 2D distribution divides by
both coordinate bin widths. Spatial filters restrict the counted domain; they
do not renormalize to the remaining population. Symmetric combined views retain
the existing equal-interface mean, rather than adding both interfaces. Top and
bottom views retain their respective assignments, including half-weight ties.
Selected reference substrate atoms remain visible in the zero-distance bin.

## Commands and supported views

```console
linak compute density trajectory.traj.h5 --surface-distance-mode nearest-atom --species all
linak plot result.density.h5 --no-gui --quantity number
linak plot result.density.h5 --no-gui --view-type 2d --2d-x-axis x --2d-y-axis distance --quantity mass
```

Use grid-backed output (the default, or `--outputs 3d`) for spatial filtering and
2D projections. Supported nearest-distance views are **Distance**, **X/Distance**
and **Y/Distance**, with either axis order for maps. XY, XZ, YZ, Z/Distance and
duplicate axes are rejected. Surface-plane mode retains its Cartesian physical
density maps. The coordinate definition belongs to the computed data: recompute
with `--surface-distance-mode plane` to obtain physical density instead.

Studio names the coordinates **Surface-plane distance** and **Nearest surface-atom
distance**. Its quantity labels and normalization note distinguish physical
density from mass/number distributions. Unsupported axes are disabled, and
changing an axis repairs its partner before rendering. A file containing only a
saved heatmap exposes only the axes actually stored. Plot labels, colorbars,
saved profiles and metadata-enabled data exports carry the corresponding units.

## Persistence and compatibility

Nearest-distance files record the observable kind, definition version,
normalization, both weighting channels, reference selection, axes, bin widths,
interface policy and scientific units. Derived filtered profiles also retain
their spatial restrictions. Sparse grids retain raw grams and entity counts;
projections convert mass to amu and apply the coordinate-width normalization.

Legacy nearest-distance files without an explicit compatible distribution
definition request recomputation. They are never silently relabelled. Existing
physical-density files remain readable, with their datasets and metadata
unchanged. Plotting saved results does not require the source trajectory.

## Measured complete-trajectory performance

The original benchmark runner and raw timing JSON files were removed during
cleanup; the historical measurements below are retained as a summary.
These were individual fresh-process measurements on Windows with Python 3.12,
using the supplied trajectories, all active species, 0.1 Å line/grid bins,
Pt references including fixed atoms, and the ordinary CLI preparation path.
These measurements are not extrapolated and are not three-run medians.

| Trajectory | Frames | Output | Compute + write | Total | Peak process memory |
|---|---:|---|---:|---:|---:|
| Pt110 Cs | 132 | 1D | 1.75 s | 1.82 s | 117 MiB |
| Pt110 Cs | 132 | Grid + X/Distance + Y/Distance | 1.51 s | 1.58 s | 116 MiB |
| Pt111 Li | 8,001 | 1D | 62.85 s | 63.10 s | 353 MiB |
| Pt111 Li | 8,001 | Grid + X/Distance + Y/Distance | 65.34 s | 66.16 s | 473 MiB |

Each measurement includes **both mass and number channels** in the same pass.
Total includes reading, preparation, analysis, writing and reopening the saved
output; the grid totals additionally include both spatial projections. Each Li
projection took approximately 0.18 s. Figure rendering/export time is excluded.
Peak memory includes loaded trajectories and scientific-library imports. The raw measurement files are no longer retained in this checkout.

The full Li calculation meets the requested one-to-five-minute target. Distinct
selected centers share a nearest-neighbor query per frame/pass across species
and weightings; nearest-distance runs skip physical-volume geometry preparation.

## Simplification and validation

The obsolete shell integration implementation, runner, three JSON benchmark
artifacts, report and experiment-only tests were removed. No Sobol, Monte Carlo,
Voronoi, tetrahedral or radial-ownership normalization remains. Shared production
periodic-neighbor and lattice utilities remain in use.

Focused regressions cover mass/population conservation, complete molecular
masses, variable frame populations, shared queries, filtered 2D marginals,
multi-source averaging, persistence, legacy rejection, labels, exports, CLI
validation and offscreen Studio axis restrictions. Existing nearest-surface
regressions cover fixed/moving substrates, triclinic and partial-periodic cells
and interface behavior. Physical-density baseline datasets remain unchanged.

Validation on this implementation: **1,699 tests passed** in the complete Python
3.12 suite. The 64 focused nearest-surface/distribution regressions also passed
on Python 3.9, 3.10, 3.11, 3.13 and 3.14. Ruff lint/format checks and mypy over
`src`, `tests` and `benchmarks` passed (157 Python files). Offscreen Qt checks
exercise Studio controls and light/dark layouts. Existing dependency deprecation
warnings remain; they are not test failures.

Nearest-distance distributions deliberately do not represent volume density or
signed metal-layer positions. Use Cartesian or surface-plane coordinates for
layer structure. Changing the source coordinate definition requires recomputation.
