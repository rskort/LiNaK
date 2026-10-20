# Potential

## Two-Sided Water References

Potential analysis reconstructs an ASE-compatible periodic structure from the
cube origin, grid vectors, counts, atomic numbers, and coordinates. Only atoms
in detected H2O topology define side windows; slab atoms, ions, adsorbates,
incomplete H/O fragments, and other contaminants are excluded from clean
averaging slices.

Resolved symmetric records store top/bottom bulk-water potentials, cSHE
potentials, window bounds, layout, and provenance. Unsuffixed values are the
equal arithmetic mean of two valid sides. If either symmetric side is invalid,
the combined value is unavailable and the row is incomplete. Without complete
H2O topology, water-bulk and cSHE values remain unavailable for every layout;
isolated hydrogen and oxygen atoms are never substituted for water molecules.

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

LiNaK reconstructs the periodic cell and atom coordinates from the cube header.
O/H topology detection identifies complete H2O molecules. Only their atom
indices determine the water-region bounds along `z`; ions, adsorbed H/O, and
incomplete fragments are excluded.

For a symmetric slab, LiNaK first constructs a contiguous periodic slab image.
Top is the face pointing along positive z; bottom points along negative z.
Water molecules are assigned by their center of mass, and their atoms are
unwrapped around that center before computing bounds. A molecule crossing the
periodic solvent cut therefore cannot expand a side's bounds across the cell.
Each side searches only its own solvent half, outside the slab. Top and bottom
windows cannot use the same physical grid slices.

If the water-like atom coordinates are `z_i`, LiNaK defines:

- `z_water_min = min_i z_i`
- `z_water_max = max_i z_i`

If no suitable water atoms are found in the cube header, the water-bulk
potential cannot be resolved and remains unavailable.

## Step 3: Resolve The Water-Bulk Potential

LiNaK takes the water-region interval and shrinks it inward by the configured
padding. It then removes any `z` slices whose slice bin contains a non-water
atom outside the detected H2O topology, and averages the `xy`-averaged
potential within the widest remaining contiguous clean window.

This is intended to avoid using interfacial edge regions directly.

For padding `p`, the candidate averaging window is:

- `z_min = z_water_min + p`
- `z_max = z_water_max - p`

LiNaK first considers grid points satisfying:

`z_min <= z_k <= z_max`

Within that candidate interval, any slice containing a non-H/O atom is excluded.
If contaminants split the candidate interval, LiNaK uses the widest contiguous
clean span. If spans tie, it chooses the one closest to the water-region
midpoint.

### Padding Fallback Logic

If the requested padded window is too narrow or empty, LiNaK progressively
relaxes the padding:

- full padding
- half padding
- quarter padding
- zero padding

If even zero padding leaves no clean water slice, the water-bulk potential is
left unavailable and the source row remains incomplete. LiNaK does not fall back
to a contaminated slice.

For symmetric references, a clean window must span at least half the unpadded
water extent on its side and at least two grid spacings. An unexpectedly narrow
window triggers the same padding relaxation instead of being accepted just
because it is nonempty. Relaxation is recorded in `water_bulk_provenance`.
Missing, overlapping, incorrectly assigned, or persistently narrow windows are
reported with warnings and stored as unavailable; an invalid symmetric side
also makes the combined potential unavailable and the row incomplete.

The `water_bulk_window_top_min_A`, `water_bulk_window_top_max_A`, and matching
bottom fields store the actual selected grid bounds in canonical periodic z
coordinates, not distances from the slab. Provenance records the coordinate
cut, cell lower bound and period, and slab planes. To recover an original
wrapped cube coordinate, use `z = lower + ((canonical_z + cut) % period)`.

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
reference constant of -0.65. (10.1063/5.0322322)

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

Current potential tables require analysis schema v2, including the top/bottom
potential, cSHE, averaging-window, layout, and provenance columns. Unavailable
values are represented by missing-value entries, not by dropping columns.
Dedicated plotting and combining validate the schema and aligned column shapes
before reading numerical payloads. Unsupported or incomplete files receive a
recomputation message; append planning selects a separate output rather than
modifying an incompatible existing table. Generic HDF5 inspection remains
available.

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
