# Electrode potential and the bulk-water reference

`linak compute potential` reads CP2K `V_HARTREE_CUBE` fields and their atomic
structures. It builds an explicit water-reference plan from geometry, applies
that same plan to every snapshot, and independently validates its electrostatics.
The Hartree field and electrode-potential trajectory are never smoothed.

## Quick start

```bash
linak compute potential -f run*/**/*v_hartree*.cube \
  --surface-elements Pt --water-padding-ang 5 \
  --save-window-plan water-reference.json --output reference.potential.h5
```

Shell wildcard behavior varies; supply an explicit ordered list of snapshots
when necessary. Input order defines the trajectory order. Use
`--snapshot-interval-fs 500` only when that is the actual interval between inputs.
Otherwise diagnostic plots use snapshot index, not invented physical times.

The command writes schema-v3 HDF5 and, by default, a diagnostic PNG beside it.
`--diagnostics path.png` chooses its location; `--no-diagnostics` skips the figure.
`--strict` returns failure if any reference is rejected or any Fermi energy is
missing. Invalid rows and their measured candidate statistics are still saved.

## Geometry-only selection

LiNaK identifies complete H2O molecules using the shared PBC-aware topology
pipeline. It identifies substrate chemical membership with the shared surface
estimator; `--surface-elements Pt` makes membership explicit. All substrate atoms,
including fixed atoms, define the outer envelope. This deliberately excludes
steps, terraces, reconstructions, and protrusions instead of allowing a mean
surface plane to cut through them. The envelope is conservative: a very rough
or unusually thin liquid gap may not contain a usable bulk region.

For periodic Z, substrate coordinates are made contiguous using the largest
circular substrate gap. The solvent coordinate runs outward from the positive-Z
substrate envelope to the next periodic image of the negative-Z envelope. A
liquid reservoir crossing the cube boundary is therefore continuous. A periodic
two-interface cell often contains **one connected water reservoir**, not two
independent bulk references. LiNaK does not duplicate or arbitrarily split it.
The reported `symmetric` layout means water contacts both faces, not a claim of
atomically identical surfaces. `single-top` and `single-bottom` denote one-sided
contact. Explicit layout requests inconsistent with the geometry are rejected.

Water COM number-density profiles are accumulated on a common fractional
solvent-gap coordinate across the batch. A fixed Gaussian **structural density**
kernel distinguishes molecular sampling holes from separate reservoirs. Its
bandwidth and bin size are explicit parameters, independent of the potential.
The default bin size is 1 A and bandwidth is 1.5 A. There is no temporal filtering
and no filtering of Hartree values. The water-support threshold defaults to
0.015 molecules/A^3. These structural parameters should be checked for the
actual water phase, lateral cell size, and sampling available.

Automatic selection requires exactly one connected water-support interval.
Multiple disconnected reservoirs are ambiguous: use a manually selected window,
or run separate analyses for the physically distinct references. LiNaK does not
rank candidates by width, potential flatness, or resulting electrode potential.

The requested padding is removed from **both edges of this water-support
interval**. Its structural support has fixed fractional bounds, while padding
remains the exact requested distance in Angstrom in each snapshot, even when
the solvent gap changes. No valid interval after padding means an invalid reference.
Padding is never halved, reduced, or replaced with another value.

Use `--z-boundary open` for nonperiodic Z. Cubes do not encode boundary conditions;
periodic Z is the explicit default assumption. An open-boundary cell with two
separate water reservoirs requires a fixed window selecting one reservoir.
The supported geometry is one slab normal to Cartesian Z, with Z decoupled from
in-plane translations; XY skew is supported. Unsupported tilted normals are
rejected rather than treated as Cartesian slices.

## Fixed windows and reusable plans

```bash
# Fixed laboratory-coordinate window for every snapshot:
linak compute potential -f run*/*v_hartree*.cube \
  --bulk-window 5 13 --output fixed.potential.h5

# A fixed interval crossing periodic Z=0:
linak compute potential -f run*/*v_hartree*.cube \
  --bulk-window 35 5 --output wrapped.potential.h5

# Reuse the exact selection definition on another batch:
linak compute potential -f later*/*v_hartree*.cube \
  --window-plan water-reference.json --output later.potential.h5
```

`--bulk-window` is in Cartesian Angstrom coordinates and is mutually exclusive
with `--water-padding-ang`. Its padding is recorded as zero; its limits are
never moved. Under periodic Z a lower limit greater than the upper limit means
a wrapped interval. Open-Z intervals cannot wrap. A manual window still has to
contain water, avoid the substrate, and pass electrostatic and resolution checks.

An automatic plan stores fractional water-support bounds, the selected bounds
at the reference gap, the reference gap,
ensemble water-density profile, all selection parameters, and a content checksum.
It follows actual slab translation and gap deformation, retaining exact
Angstrom padding without choosing a new region at each time step.
A fixed Cartesian window instead stays fixed in the
laboratory frame. Both choices are explicit and reproducible.

Every snapshot is checked against the plan. Loss of water, substrate intrusion,
insufficient width, or violation of the requested substrate clearance invalidates
that snapshot; the window is not moved to repair it. This preserves genuine
structural changes and genuine electrostatic fluctuations.

Appending reuses the persisted plan and requires unchanged selection,
validation, and electrochemical-reference settings. It does not replan from
just the newly appended frames. Use a separate output for convergence comparisons
or `--overwrite` for a deliberate recomputation. Old analysis schemas are rejected
with a recomputation message; there is no compatibility or output-path fallback.

## Averaging and electrostatic validation

Window selection does not inspect Hartree values. Only afterward does LiNaK
integrate the planar profile within the specified interval. Trapezoidal
quadrature interpolates **only at the interval boundaries**, so sub-grid motion
does not add or remove an entire slice abruptly. No profile smoothing, detrending,
plateau fitting, or potential-based window optimization is performed. The exact
integration weights are saved alongside the original planar profile.

The selected interval must pass these explicit, configurable checks:

| Check | Option | Default |
| --- | --- | --- |
| Physical interval width | `--min-window-width-ang` | 4 A |
| Contributing original grid points | `--min-window-grid-points` | 5 |
| Water number density | `--min-water-density` | 0.015 molecules/A^3 |
| Occupied structural bins | fixed check | at least 80% |
| Absolute linear potential gradient | `--max-reference-gradient` | 0.05 eV/A |
| Spatial potential standard deviation | `--max-reference-std` | 0.25 eV |
| Peak-to-peak potential range | `--max-reference-range` | 1 eV |

These are conservative screening defaults, **not universal physical constants**
or proof of bulk thermodynamic behavior. Instantaneous molecular fluctuations,
lateral cell size, pseudopotentials, applied fields, and the intended reference
matter. Inspect the unsmoothed profile, gradients, variation, and window
convergence before justifying different tolerances. Do not relax thresholds merely
to obtain accepted or smoother potentials. Geometry alone does not establish an
electrostatic reference, and a field-bearing liquid may have no unique scalar
bulk potential under this definition.

Dissolved ions are not arbitrarily removed by deleting every slice containing a
non-H/O atom. They belong to the actual electrolyte. Water support, substrate
clearance, and the measured electrostatics determine validity. Adsorbed species
near an interface are handled by geometry and padding rather than selecting a
more desirable clean-potential segment.

Rejected candidates retain their mean, variation, gradient, boundaries, and
sensitivity measurements wherever measurable. Their published
`water_bulk_potential_ev` and `electrode_cshe_ev` are unavailable. Inspecting
rejected candidates is essential: reporting only accepted snapshots could bias a
trajectory toward unusually small fluctuations.

## Units, sign, gauge, and Fermi association

The physical definition remains:

```text
U_cSHE = V_water(CP2K electron-energy convention) - E_F - cshe_offset_ev
```

All three terms use eV numerically for a one-electron potential in volts. The
CLI and Python `PotentialConfig` now both store a **subtractive** offset, default
+0.65 eV. The old signed-additive Python convention is removed. An explicitly
negative CLI offset is subtracted as written. The calibration is not a universal
SHE constant; it must match the intended water-reference method and electronic
structure setup. The repository's original calibration refers to
[Hagopian et al., J. Chem. Phys. 164, 174711 (2026)](https://doi.org/10.1063/5.0322322).

CP2K documents that `V_HARTREE_CUBE` has the opposite sign to the physical
positive-test-charge electrostatic potential. LiNaK retains CP2K's electron-energy
convention consistently with E_F; it does **not** apply an isolated sign flip.
See the [CP2K manual](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/PRINT/V_HARTREE_CUBE.html)
and [CP2K source](https://doxygen.cp2k.org/da/d8a/qs__scf__post__gpw_8F_source.html).

Scalar Hartree values are converted once from Hartree to eV. Standard positive
cube grid counts use Bohr coordinates; consistently negative counts denote
Angstrom coordinates. Mixed unit conventions are rejected. Cube ordering is
X outermost and Z innermost, averaged over X/Y exactly. LiNaK never tries two
orderings and chooses the smoother profile. Density cubes are not Hartree fields.

V_water and E_F must refer to the **same SCF state and electrostatic gauge**.
Adding the same constant to both leaves U unchanged; tests enforce this. LiNaK
never independently recenters a snapshot's Hartree field. Source-specific
`output.out` (or exactly one alternative non-slurm `.out`) can supply E_F for a
single-point calculation. Multiple snapshots sharing one output and multi-step
MD outputs require an explicit map; reusing the output's last Fermi value for
all snapshots is forbidden.

```csv
source,fermi_ev
frame_00000000/Pt-v_hartree-1_0.cube,3.25
frame_00000050/Pt-v_hartree-1_0.cube,3.31
```

Pass `--fermi-map fermi.csv`. Relative paths are relative to that CSV; energies
are in eV in the matching CP2K electron-energy gauge. For logical container fields, use the recorded source identity (the original
cube path retained by conversion), with `::profile:1` when applicable. The map explicitly asserts matching SCF
gauges; numerical values alone cannot prove that assertion. Source/output
association and this assumption are recorded. Arbitrarily aligned energies or
Fermi-relative DOS energies are not substitutes for absolute-gauge E_F.

## Diagnostics and storage

Schema v3 stores one `/records` row per snapshot, including:

- requested padding; actual interval bounds, width, and contributing grid count;
- candidate Hartree mean, spatial standard deviation, range, and gradient;
- H2O density and surface-layout classification;
- `reference_valid`, rejection reasons, status, and Fermi provenance;
- the full selection plan, validation settings, cSHE offset, and gauge assumption;
- declared window-sensitivity results and physical time when supplied.

Automatic `window_min_A`/`window_max_A` are distances from the positive-Z substrate
envelope along the contiguous solvent gap. Fixed-window bounds are unwrapped
Cartesian z coordinates. `reference_diagnostics_json` states the coordinate
system and the exact wrapped `raw_z_intervals_A` for either case.

`/profiles/<record id>` stores the original z grid, unsmoothed planar Hartree
profile, solvent coordinate, water COM distribution, selected integration
support, and `averaging_weights`. Their weighted sum reproduces the reference.
These arrays enable diagnostics without rereading the cube files.

The four-panel figure shows:

1. The time-averaged raw Hartree profile and the gauge-aligned mean V-E_F profile,
   aligned only by the fractional solvent coordinate, with the selected region.
2. The water reference and cSHE trajectory; rejected candidates remain visible.
3. Sensitivity to predeclared ±window shifts and interval contraction/expansion
   (`--sensitivity-ang`, default 1 A). These variants never select the reference;
   invalid variants are marked explicitly. Delta V_water equals Delta U_cSHE at fixed E_F.
4. Potential gradients and spatial variation, including the chosen thresholds.

```bash
# Regenerate diagnostics solely from the HDF5:
linak plot reference.potential.h5 --reference-diagnostics --output checks.png

# The usual potential summary plot is also available:
linak plot reference.potential.h5 --no-gui --output potential.png
```

No time smoothing, acceptance-based recentering, or forced plateau enters the
computed potential. Window sensitivity and rejected snapshots remain part of the
scientific record, rather than being concealed by a smoother trajectory.
