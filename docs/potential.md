# Potential

`linak compute potential` extracts electrode-potential summary quantities from
CP2K Hartree-potential cube files and stores the result as a row-oriented HDF5
table.

## What Is Being Computed

For each input Hartree cube file, LiNaK attempts to compute:

- `efermi_ev`
- `water_bulk_potential_ev`
- `electrode_cshe_ev`

The central derived quantity is:

`electrode_cshe_ev = water_bulk_potential_ev - efermi_ev - cshe_offset_ev`

if both the water-bulk potential and Fermi energy can be resolved.

## Input Model

Each input is assumed to be a CP2K Hartree cube file describing an
electrostatic potential on a three-dimensional grid.

LiNaK processes one cube file at a time and writes one output row per source.

## Step 1: Read The Cube And Compute An In-Plane Average

LiNaK reads the cube header and grid, then computes the `xy`-averaged potential
as a function of `z`.

This converts the full 3D potential field into a 1D profile:

- horizontal coordinate: `z`
- vertical coordinate: average Hartree potential in that `z` slice

The raw Hartree values are converted into electron volts before downstream
analysis.

If `V(x, y, z_k)` is the cube value on one `z` slice, LiNaK computes:

`V_xyavg(z_k) = mean_{x,y} V(x, y, z_k)`

## Step 2: Infer The Water Region From The Cube Header

LiNaK inspects the atoms listed in the cube header and identifies water atoms by
atomic number:

- hydrogen
- oxygen

From those atoms, LiNaK determines the water-region bounds along `z`.

If the water-like atom coordinates are `z_i`, LiNaK defines:

- `z_water_min = min_i z_i`
- `z_water_max = max_i z_i`

If no suitable water atoms are found in the cube header, the water-bulk
potential cannot be resolved and remains unavailable.

## Step 3: Resolve The Water-Bulk Potential

LiNaK takes the water-region interval and shrinks it inward by the configured
padding. It then averages the `xy`-averaged potential within that bulk window.

This is intended to avoid using interfacial edge regions directly.

For padding `p`, the candidate averaging window is:

- `z_min = z_water_min + p`
- `z_max = z_water_max - p`

LiNaK then averages over grid points satisfying:

`z_min <= z_k <= z_max`

### Padding Fallback Logic

If the requested padded window is too narrow or empty, LiNaK progressively
relaxes the padding:

- full padding
- half padding
- quarter padding
- zero padding

If even that fails, LiNaK falls back to the nearest available `z` point around
the water-region midpoint and emits a warning.

The midpoint fallback uses:

`z_mid = 0.5 * (z_water_min + z_water_max)`

and selects the grid point `z_k` minimizing `|z_k - z_mid|`.

This makes the workflow robust to narrow or awkward water regions without
silently discarding the source.

## Step 4: Resolve The Fermi Energy

LiNaK searches for a suitable CP2K output file in the same directory as the
cube source. It then parses the Fermi energy from the output text using built-in
regex patterns covering the supported CP2K output styles.

When several matches exist in the parsed text, LiNaK keeps the last one in the
file, which reflects the final CP2K-reported Fermi value in the current
implementation.

If the Fermi energy cannot be parsed, `efermi_ev` remains unavailable.

## Step 5: Compute cSHE

If both upstream quantities are available, LiNaK computes:

`electrode_cshe_ev = water_bulk_potential_ev - efermi_ev - cshe_offset_ev`

The `cshe_offset_ev` value is configurable and defaults to LiNaK's internal
reference constant.

If either upstream quantity is missing, LiNaK does not fabricate a cSHE value.
Instead, the row is marked as incomplete.

## Batch Behavior

For many cube files, LiNaK can use a thread pool for throughput. Each source is
processed independently.

The workflow is designed to be robust for long-running jobs:

- one row is written per source
- compatible outputs are appended to by default
- already-seen sources can be skipped
- failures are stored as explicit error rows instead of aborting the entire run

This makes reruns resumable.

## Status Model

Each output row carries a status, typically one of:

- `ok`
- `incomplete`
- `error`

This is important for interpretation. A missing cSHE value does not necessarily
mean the file is corrupt; it may simply mean the source lacked enough metadata
to resolve all required terms.

## What Gets Stored

Potential results are stored as a row-oriented HDF5 table with columns such as:

- `id`
- `source`
- `source_dir`
- `output_out`
- `efermi_ev`
- `water_bulk_potential_ev`
- `electrode_cshe_ev`
- `status`
- `error`

At the file level, LiNaK also stores format metadata and column metadata so the
table can be inspected and plotted consistently later.

## Important Assumptions And Limitations

- The workflow is CP2K-specific.
- The cube header must contain chemically meaningful atom information if the
  water region is to be inferred automatically.
- Water-bulk potential quality depends on the spatial validity of the inferred
  water region and chosen padding.
- Fermi parsing depends on supported CP2K output text patterns.
- The analysis produces summary values, not a stored full 1D potential profile
  per source.

## Related Documentation

- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
