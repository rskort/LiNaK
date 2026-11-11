# Interface alignment

Symmetric surface handling establishes an outward distance on each side of a
slab. The two sides can still have mirrored steps or different lateral origins.
Studio's **Surface representation → Align interfaces…** registers the negative
interface against the positive interface for density maps, position
projections, and orientation spatial views.

## Using the editor

1. Compute a new symmetric density, position, or orientation HDF5 file. Density alignment
   requires the side-resolved sparse grid (`--outputs all`, the default, or
   `--outputs 3d`). Older results show a disabled control and a recompute message.
2. Open **Align interfaces…** and select the source if several references are
   loaded. Check **Enable alignment for this source**.
3. Choose a compatible reflection or rotation and adjust the lateral shifts in
   Å, or select **Suggest alignment**. Suggestions populate the preview without
   applying the transform to the plot.
4. Inspect the lateral overlay and both cross-sections. Blue atoms belong to
   the positive interface; orange atoms are the mapped negative interface.
   Red outlines identify unmatched atoms. Switch reference frames to inspect
   changes over the sampled trajectory.
5. Review the matching coverage and displacement. Multiple similarly good
   solutions appear as alternatives. If the surfaces retain mismatches, **Use
   despite mismatches** is required before applying.
6. Click **Apply**. **Cancel** discards edits. To turn registration off, uncheck
   the enable control and apply; **Reset** restores the defaults and turns it off.

The status reads **Off**, **Applied**, or **Applied with mismatches**. Save the
plot profile to retain the alignment. **Copy alignment** and **Paste alignment**
transfer it between compatible new density, position, and orientation files; the reference
identity must match. Settings apply across species sharing that reference.

## What the transform means

The positive interface stays fixed. Negative-interface lateral fractional
coordinates use the fixed periodic mapping `u′ = (u M + t) mod 1`. Allowed
matrices preserve the in-plane cell metric, including in tilted prismatic
cells. All stored cell geometries must support the chosen matrix. There is no
scaling, elastic deformation, normal shift, or independent fit of each frame.

Suggestions fit substrate atoms, including fixed atoms, by species, periodic
lateral position, and outward height. The default reference depth is 5 Å and
matching tolerance is 0.3 Å. A good geometric match requires at least 95% of
each interface's selected atoms to match within tolerance in every reference
snapshot. Changing depth or tolerance re-evaluates the fit. Different
terminations, reconstructions, and terrace heights can remain mismatched.
Geometric correspondence alone does not establish that the simulated
interfaces have the same physical conditions.

Use **Distance** for the normal coordinate in registered cross-sections.
Full-cell views and pure distance profiles retain their existing coordinates
and values. Orientation distributions are also unaffected. Coordination has
no registration controls.

## Spatial resolution and plotting behavior

Density registration moves negative-side mass and entity accumulators before
filtering, projection, normalization, and interface averaging. Source-bin
footprints are clipped to the actual cell and remapped by periodic overlap
area. Matching bin boundaries give an exact bin permutation; fractional-bin
shifts distribute contributions conservatively. Source counts are preserved,
but positions inside the original bins are unknown. The dialog therefore
reports the source-bin resolution. Registration cannot restore detail finer
than that resolution.

Each interface retains its own normalization, and the combined map retains the
equal mean of the normalized sides. Original analysis arrays are preserved.

Position registration derives lateral coordinates for negative-side samples.
Axes, color quantities, and filters use the same derived values. Paths break
at periodic seams, interface changes, and excluded samples. Samples whose
interface side is ambiguous are omitted from aligned spatial projections;
their count is recorded in the alignment metadata. Atom identities and time
values are preserved.

Orientation registration moves negative-side cosine sums, valid-orientation
counts, and water counts together, using each stored cell geometry. Filters
and projections act on these registered accumulators before interface means
are combined. Density-weighted maps multiply the folded mean orientation by
the folded density. Registration preserves the existing angle definitions;
it does not rotate or reflect molecular orientation directions. The positive
interface and Full cell stay fixed. Mean-orientation lines with lateral axes
or lateral filters also use registration. Older orientation results require
recomputation to supply per-geometry moments and the substrate reference.

Figure exports use the preview's transform. Data exports include the transform,
reference identity, fit and override information, and density resolution when
**Include metadata comments** is enabled. Position projection exports include
the transformed X/Y/value quantities and a `break_before` column.

## Responsiveness and caching

Rectangular-cell alignment uses analytical periodic interval overlaps, with an
exact permutation when grid boundaries coincide. Other supported geometries
retain conservative polygon overlaps. All water and orientation accumulators
share the same transfer weights; preview grids are never coarsened.

Studio reuses unchanged source arrays across alignment edits. Changing the
angle, quantity, filters, or colors can reuse registered raw accumulators.
The existing memory and cache budgets bound this work and cache storage.
File paths, modification times, sizes, profile selections, and transform
geometry determine reuse; a changed source is loaded again. Debug logging
reports stage timings and cache hits. These optimizations do not require
recomputing files that already support alignment.

Developers can compare the analytical path against the polygon calculation and
measure cold registration, warm registration, and preview drawing separately:

```console
python -m benchmarks.compare_orientation_alignment --grid-size 64 --groups 4
```

## Stored reference geometry

New symmetric density, position, and orientation results automatically contain up to five
evenly spaced substrate snapshots. Capture happens during existing processing
passes and adds no trajectory reread. References retain atom identities,
elements, full cells, origins, periodicity, and both surface references. They
are versioned and stored once per reference under `/interface_references`, with
a deterministic fingerprint shared by matching density, position, and orientation results.

Normal plot loading reads only reference metadata. Atom coordinates are loaded
when the alignment editor or fit validation needs them. Alignment itself is a
saved plot setting, separate from the analysis data. Legacy files have no
attachment or migration workflow: recompute them to use the feature.
