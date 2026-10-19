# Nearest-atom surface distances

LiNaK offers two definitions of distance to the surface. `plane` is the existing
signed distance along the selected Cartesian axis, using a frame-wise layered or
rough reference. `nearest-atom` measures the shortest three-dimensional distance
to any eligible substrate atom in the current frame:

\[
d_i(t)=\min_{j\in S(t)}\|r_i(t)-R_j(t)\|_{\mathrm{MIC}}.
\]

Nearest-atom distances are nonnegative. Eligible substrate atoms have distance
zero to themselves. Molecular selections use their existing PBC-aware centers
of mass. No exposed atomic layer or fitted plane is selected for this measurement.

## Selecting the substrate

Automatic element identification uses the existing abundance, mass, and mobility
rules. Use `--surface-elements Pt` to choose explicitly. Python callers can use
`SurfaceEstimatorOptions(distance_mode="nearest-atom", surface_atom_indices=...)`
or an atom mask. Explicit indices and masks require stable atomic elements and
ordering. Element selections are resolved against every frame.

Constrained atoms are excluded unless `--include-fixed-surface-atoms` is supplied.
The established automatic fixed-slab fallback remains available when automatic
selection leaves no candidates. An empty explicit selection fails with a useful
error. A rough surface envelope is incompatible: it would select an exposed layer.
`--surface-mode` continues to select the plane estimator; nearest-atom distances
bypass both layered and rough plane estimation.

## Commands and stored preferences

```bash
linak apply convert trajectory.xyz --surface-distance-mode nearest-atom --surface-elements Pt
linak compute density trajectory.traj.h5
linak compute position trajectory.traj.h5 --species mol:H2O
linak compute cn trajectory.traj.h5 --species K O --cutoff 3.5
linak compute orientation trajectory.traj.h5
```

Conversion records the effective distance mode, substrate elements, fixed-atom
policy, and requested layout in trajectory metadata. Subsequent analyses inherit
these settings. Explicit `--surface-distance-mode plane` selects the existing
plane method. Explicit candidate elements and layout override stored choices.
Existing files without a preference use `plane`.

Plane conversion caches retain their existing behavior. Nearest mode saves
selection settings rather than target-dependent distance matrices. Searches use
current frame coordinates and cell geometry. Incompatible geometry or selections
are rebuilt; a stored scalar plane cannot substitute for nearest distances.
Analysis outputs identify their effective method. Plotting an analysis HDF5 uses
its stored distances/counts and never requires the original trajectory.

Distance-based conversion and analysis filters use the selected definition.
If filtering removes every eligible substrate atom, nearest-atom conversion or
analysis cannot define a surface and reports that selection problem. Filtering
preserves the original substrate element choice; leftover fluid atoms cannot be
automatically reclassified as a replacement substrate. Nearest trajectory files
retain constrained-atom masks in bounded compressed chunks, including after
filtering changes atom indices or frame atom counts.

## Interfaces and periodic cells

Single-top and single-bottom layouts query the complete eligible substrate set.
For symmetric layouts, the slab is made contiguous using its largest periodic
gap; candidates are classified into the positive and negative halves of its
geometric extent. These are candidate sets, not exposed layers. The closest
candidate in each set determines that interface's distance. The smaller distance
assigns each target to an interface; exact ties share equal weight.

Auto layout uses the shared [persistent water-contact classifier](surface-estimation.md):
water within 6 Å outward of the complete substrate faces establishes wetting;
water far across a periodic cut does not. Full-cell
views use stored Cartesian coordinates, top/bottom views retain their individual
counts, combined views average both interface distributions equally, and compare
views show both independently.

Distance queries support orthorhombic and triclinic cells, partial or absent PBC,
translated origins, unwrapped coordinates, moving substrates, and changing cells.
The implementation reduces the periodic lattice, wraps coordinates in its periodic
subspace, and searches a Cartesian KD-tree containing all relevant periodic images.
It does not independently wrap Cartesian components in a skewed cell.

Symmetric interface labeling requires a slab axis decoupled from in-plane
periodic translations. An explicitly symmetric tilted periodic slab fails clearly. Such cells
still support true nearest-atom distances with a single interface. Auto mode uses
a single interface when it cannot establish an unambiguous two-sided axis.

## Density and plotting

Surface-plane distance retains physical mass density in g/cm^3 and number density
in entities/nm^3. Cartesian spatial density maps retain their physical volume
normalization.

Nearest surface-atom distance instead produces ordinary distributions averaged
per frame: **amu/Angstrom** for mass, **atoms/Angstrom** or **molecules/Angstrom**
for population. Molecular centers carry the complete molecular mass. These are
not volumetric densities. No geometric integration is used.

Nearest-distance bin bounds come from observed distances. Species computed together
share their distance bins at the requested resolution. Reference substrate atoms
remain in the zero-distance bin. Combined symmetric views preserve the existing
equal-interface mean; top and bottom views retain their own assignments, including
half-weight ties.

Sparse grids store raw mass and population. They support Distance profiles and
X/Distance or Y/Distance maps (either axis order), normalized by frame count and
the plotted bin widths. Map units are amu/Angstrom^2 and atoms or molecules/Angstrom^2.
Spatial filters restrict the counted domain without renormalizing to its population.
XY/XZ/YZ maps are unavailable in nearest-distance mode; compute surface-plane mode
to obtain Cartesian physical-density maps. Mixing physical density and nearest-
distance distributions on a common distance plot is rejected.

Studio names the coordinates **Surface-plane distance** and **Nearest surface-atom
distance**. Quantity choices, units and available axes follow the saved observable.
Invalid pairs are corrected together before preview; unsupported coordinates are
disabled. Files containing only a saved heatmap expose only its stored axes. The
coordinate definition is source data: recompute the trajectory with the other
`--surface-distance-mode` to change it, rather than relabelling saved bins.

Saved nearest-distance results carry an explicit observable kind, definition
version, weighting channels, coordinate axes, bin widths and units. Unmarked legacy
nearest-density results require recomputation; valid plane-density results remain
readable. See [density and distance distributions](density-distributions.md) for
commands and the measured complete-trajectory performance.

Water-orientation distributions, coordination numbers, Cartesian positions,
molecular topology and their existing selection conventions are unchanged.

## Performance

Only the current frame's trees and geometry key are retained. Target queries are
vectorized in bounded blocks, and species share the frame's substrate structure.
There are no target-by-substrate matrices in the production query path and no
trajectory-wide distance cache. Plane queries bypass this infrastructure.

The historical kernel benchmark used a moving 675-atom metal slab and 900
targets, with an independent bounded brute-force MIC comparison. Its standalone
runner was removed during cleanup; the summary below is retained.

On the local Windows/Python 3.12 environment, three 200-frame runs took 0.178,
0.175, and 0.182 seconds (median 0.178 s, about 1,126 frames/s for search build
and queries). Peak process memory was 109 MiB, including the brute-force check
and imported scientific libraries. The maximum error against the MIC reference
was 7.1e-15 Å. These timings measure the geometric kernel; complete trajectory
analysis also includes I/O, selections, topology, histograms, and output writing.
