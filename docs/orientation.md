# Orientation

## Symmetric Cells

The top outward normal is `+axis` and the bottom outward normal is `-axis`.
LiNaK flips the bottom polar cosine before pooling while retaining the existing
in-plane azimuthal convention. Folded density, averages, and heatmaps are equal
side means; top/bottom arrays and sparse-grid accumulators are stored.
Spatially filtered line and mean-orientation heatmap projections first
accumulate each side separately, then average the available side means with
equal weight. Selecting a range that includes all data therefore preserves
the folded result even when the side populations differ. Folded sample and
block uncertainties use the same per-frame side averaging and per-interface
density normalization. Full-cell and individual-interface views retain their
own reference and weighting conventions.

`interface_view=all` overlays the three labeled 1D views. Orientation heatmaps
select exactly one of `full`, `folded`, `top`, or `bottom`.
The comparison expands source IDs, labels, and per-series styles together so
GUI preview and export can render all three curves. Interface plots default
to outward distance; returning from Full cell restores the interface axis
selection, defaulting to distance.
`interface_view=full` uses the raw periodic surface-axis coordinate and one
laboratory `+axis` polar reference across the complete cell; it does not flip
the negative-axis half. The analysis stores separate full-cell 1D averages,
density, heatmaps, counts, uncertainty statistics, and laboratory-frame sparse
polar accumulators. Older files without these additive datasets remain readable,
but Full cell is disabled with a recomputation message.

The Studio names the five representations **Full cell (unfolded)**,
**Positive-axis interface**, **Negative-axis interface**, **Combined interfaces
(folded mean)**, and **Compare interface curves**. The displayed `+X/+Y/+Z`
and `−X/−Y/−Z` labels follow the selected surface axis.

`linak compute orientation` computes water-orientation observables as a function
of distance to a trusted shared surface reference or, when no trusted surface
reference is available, along a raw Cartesian axis.

## Aligning symmetric interfaces

**Combined interfaces (folded mean)** averages the two interface means with
equal weight. Without registration, their lateral coordinates retain the
original cell positions, so mirrored steps or shifted terraces may overlap
incorrectly in spatial maps.

For new symmetric orientation results, open **Surface representation → Align
interfaces…** to preview the substrate overlay, suggest or select a compatible
rotation/reflection, and adjust the lateral translation. Apply remains
explicit; Cancel discards dialog changes. Matching diagnostics, mismatch
acknowledgement, and reference validation are shared with density and position.

Registration maps the negative interface's **sample locations** into the
positive interface's lateral coordinates. It preserves the stored polar and
azimuthal angle definitions, including their existing sign conventions; it
does not rotate or reflect molecular orientation directions. Per-geometry
cosine sums, valid-orientation counts, and water counts move together by
conservative periodic overlap before filtering, rebinning, normalization, and
equal-interface averaging. Density-weighted orientation remains the product
of the displayed mean and density. Fractional shifts cannot recover detail
below the original source-bin resolution shown in the dialog.

Alignment applies to spatial maps and mean-orientation lines that use lateral
axes or lateral filters. Use **Distance** for an aligned normal cross-section.
The positive interface stays fixed. Full-cell views, distance-only profiles,
and orientation distributions retain their interpretation. Saved alignment
settings survive these view switches and are included in plot profiles,
preview/export rendering, and quantitative export metadata.

New computations automatically embed substrate reference snapshots and
per-geometry moments during existing trajectory processing. Older results
retain their ordinary plotting capabilities, but alignment is disabled with
**Recompute this orientation file to enable interface alignment.** No
attachment or approximate legacy-normalization workflow is provided. Current
orientation alignment requires a fully periodic, decoupled slab geometry and
surface-plane sparse geometry. See [Interface alignment](interface-alignment.md)
for the matching controls and geometry limitations.

Python plotting and sparse-grid projection functions accept
`interface_alignments={reference_id: alignment.to_dict()}`. Settings can be
copied between density, position, and orientation results only when their
embedded reference fingerprints match.

## What Is Being Computed

LiNaK identifies water molecules, computes PBC-aware water geometry for each
frame, and accumulates orientation statistics in distance bins.

The primary stored outputs are:

- `cos_polar_mean`
- `cos_azimuthal_mean`
- `cos_polar_density`
- `cos_azimuthal_density`
- `heatmap_polar`
- `heatmap_azimuthal`

This analysis is water-specific.

## Water Geometry Used

LiNaK reuses the shared water-geometry pipeline from
[Water Detection And Water Geometry](water-detection.md).

For each detected water molecule the shared geometry builder returns:

- oxygen position `O`
- hydrogen positions `H1` and `H2`
- water center of mass

The O-H vectors used by orientation are explicitly:

- `OH1 = H1 - O`
- `OH2 = H2 - O`

Those vectors come from the shared PBC-aware water geometry routine, which uses
minimum-image handling so each hydrogen is associated with the correct image of
its oxygen.

## Distance Coordinate

Each water molecule is assigned a distance-bin coordinate using its center of
mass along the selected axis.

For a trustworthy reference, let `q` be the molecule COM coordinate in the
canonical periodic slab image. The positive-interface distance is
`d_top = q - q_top`, and the negative-interface distance is
`d_bottom = q_bottom - q`. Symmetric folding uses `max(d_top, d_bottom)`;
single-interface layouts use the distance for their selected side. Periodic
canonicalization keeps solvent in the outward region instead of wrapping a
water/vacuum boundary into negative distances. Full-cell plots instead use the
stored raw Cartesian COM coordinate, with its original origin.

If the surface estimate is incomplete or below the trust threshold, orientation
does not silently use it anyway. Instead, LiNaK bins the raw center-of-mass
axis coordinate and stores:

`coordinate_mode = "axis"`

Interface-relative polar orientation is expressed in the assigned outward
frame. Full-cell polar orientation retains the laboratory reference frame.

## Polar Orientation

LiNaK builds the polar bisector from the two normalized O-H vectors:

`u_1 = OH1 / ||OH1||`

`u_2 = OH2 / ||OH2||`

`b = (u_1 + u_2) / ||u_1 + u_2||`

It then computes:

`cos_polar = b . e_ref`

where `e_ref` is the unit vector along the chosen Cartesian reference axis.

Interpretation:

- `+1`: hydrogens point in the positive reference-axis direction
- `-1`: hydrogens point in the negative reference-axis direction
- `0`: the water bisector is perpendicular to the reference axis

These are laboratory-frame signs, used by Full cell and raw-axis fallback
views. With a trusted interface reference, negative-side polar cosines change
sign before side-resolved/folded accumulation. For the usual choice
`reference_axis == axis`, positive values then mean hydrogens pointing outward
from the selected interface. If a different reference axis is selected, do not
interpret that component as the surface-normal orientation. The in-plane
azimuthal convention is unchanged.

If either O-H vector is degenerate or if the bisector norm is too small, that
sample is marked invalid for polar statistics.

## Azimuthal Orientation

LiNaK also computes a normal to the molecular plane:

`n = OH1 x OH2`

The order `OH1 x OH2` is intentional and follows the stable `H1`, `H2` ordering
from the shared water triplet builder. The sign of this azimuthal descriptor
therefore depends on that deterministic hydrogen ordering.

LiNaK then removes the component of `n` along the reference axis:

`p = n - (n . e_ref) e_ref`

After normalization, it stores the cosine of the angle between the projected
vector and the first available Cartesian axis perpendicular to `reference_axis`.

That makes `cos_azimuthal` a descriptive in-plane orientation observable, not a
unique physically absolute laboratory-frame azimuth unless the system itself
defines a preferred in-plane direction.

If the projected plane normal is too small, that sample is marked invalid for
azimuthal statistics.

## Distance-Bin Accumulation

For each water molecule in each frame, LiNaK:

1. computes the COM position
2. computes `cos_polar`
3. computes `cos_azimuthal`
4. assigns the molecule to a distance bin
5. accumulates separate counts for total, polar-valid, and azimuthal-valid
   samples

Each distance bin stores:

- `count_total`
- `count_polar_valid`
- `count_azimuthal_valid`

The bin means are then:

- `cos_polar_mean = sum(cos_polar over polar-valid samples) / count_polar_valid`
- `cos_azimuthal_mean = sum(cos_azimuthal over azimuthal-valid samples) / count_azimuthal_valid`

Bins with zero valid samples are stored as `NaN`, not as `0`.

## Density-Weighted Orientation

LiNaK also computes a water molecular density profile over the same distance
bins using the water COM assignments.

It then forms:

- `cos_polar_density = cos_polar_mean * density`
- `cos_azimuthal_density = cos_azimuthal_mean * density`

These are density-weighted orientation-bias profiles. They are not pure
orientation observables.

If a usable periodic cell is available, LiNaK stores a volumetric molecular
density using the slab volume:

- `axis_length = ||cell[axis_index]||`
- `cross_section = |det(cell)| / axis_length`
- `bin_volume = cross_section * bin_width`

If the cell varies across frames, LiNaK averages per-frame volume-normalized
histograms. If no usable cell is available, it falls back to a linear molecular
density along the chosen axis.

## 2D Orientation Heatmaps

In Studio, **2D representation** selects **Orientation distribution** (the
default) or **Spatial map**. Each representation retains its own settings.

Orientation distributions use `cos(angle)` on the second axis, rather than the
angle in radians or degrees.

LiNaK stores:

- `heatmap_polar[distance_bin, cosine_bin]`
- `heatmap_azimuthal[distance_bin, cosine_bin]`

These heatmaps store raw counts. Any normalization to probabilities happens
only at plotting time when explicitly requested.

For orientation frequency heatmaps, **Displayed values** controls the
display-time representation:

- `raw_counts`: observations per displayed bin.
- `joint_probability_density`: counts divided by the total count and both bin
  widths, so the complete heatmap integrates to one.
- `conditional_probability_density`: an orientation density within each
  occupied distance row, so each row integrates to one.
- `bulk_relative_enrichment`: the conditional density divided by the pooled
  bulk orientation density; `1` is bulk-like.

Counts from all enabled sources are combined and rebinned by summation before
this transformation. Linear or logarithmic color mapping is applied afterward
and does not change the displayed values. Bulk-relative plots automatically use
the longest contiguous density plateau at least 80% of peak density, or can use
`--heatmap-bulk-reference manual` with `--heatmap-bulk-min` and
`--heatmap-bulk-max`.

For example:

```bash
linak plot traj.orientation.h5 --view-type 2d-heatmap \
  --heatmap-value-mode conditional_probability_density
```

Python plotting calls use `heatmap_value_mode` with the same four values.
Removed `heatmap_normalize` and `heatmap_normalization_mode` options are not
accepted. Saved profiles containing them must be recreated; LiNaK does not
silently select a replacement scientific normalization.

Values exactly equal to `-1` and `+1` are handled deterministically and are
assigned to the first and last cosine bins respectively.

### Spatial maps

Select **Spatial map** to expose spatial X/Y axes, spatial filters, and **Color
quantity**. Both polar and azimuthal orientation support:

| Color quantity | Definition | Units |
| --- | --- | --- |
| Mean orientation | Valid-sample mean of cos(θ), or cos(φ) | Dimensionless |
| Water density | Water molecule count normalized by physical volume and frames | molecules/Å³ |
| Density-weighted orientation | Displayed mean orientation × displayed water density | molecules/Å³ |

A new spatial view starts with X/Y and mean orientation. Choose Cartesian
planes or a lateral coordinate paired with outward **Distance**. Distance and
the raw surface-normal coordinate are redundant. Folded density cross-sections
use Distance; Full cell uses Cartesian coordinates and laboratory-frame polar
orientation. Disabled axis and quantity choices explain the restriction.
Frequency normalization controls apply only to orientation distributions.

Filtering and rebinning operate on raw cosine sums, valid-orientation counts,
water counts, and normalization geometry before deriving values. Enabled
sources pool these accumulators; density uses frame-aware normalization.
Different source grids are coarsened to shared boundaries so stored bins are
never split to invent finer spatial detail. For symmetric cells, interface means and interface densities are folded separately
with equal weight, then multiplied for density-weighted orientation. This is
the same definition as the 1D density-weighted result. A mean of interface
products would give a different quantity.

Density uses the same per-geometry physical volumes as density analysis,
including changing cells, origins, and empty frames. Covered empty bins have
zero density. Mean and density-weighted orientation remain undefined where no
valid orientation exists; bins outside physical coverage stay masked. Values
exported to data files precede colormap and logarithmic rendering and include
quantity, units, axes, filters, interface representation, and normalization
metadata.

New computations store the optional normalization record automatically during
the existing trajectory processing. Older orientation files retain their
distributions and sparse-grid mean maps. **Recompute orientation to enable
density and density-weighted spatial maps.** A fully periodic usable cell and
a physically supported distance definition are required; nearest-atom
distance cannot supply this normalization.

```bash
linak plot traj.orientation.h5 --heatmap-kind spatial \
  --heatmap-quantity density-weighted \
  --heatmap-x-axis x --heatmap-y-axis distance
```

Python plotting accepts `orientation_heatmap_kind="spatial"`,
`orientation_heatmap_quantity="average"`, `"density"`, or `"density-weighted"`,
and `orientation_heatmap_x_axis` / `orientation_heatmap_y_axis`. Mapping builders
also accept `orientation_heatmap_kind` and `orientation_heatmap_quantity`
(with `heatmap_kind` / `heatmap_quantity` shorthand), plus `heatmap_x_axis` and
`heatmap_y_axis`. Existing Python calls with explicit spatial
axes continue to produce spatial mean-orientation maps. Select
`orientation_heatmap_kind="distribution"` explicitly to request a distribution.

## What Gets Stored

The orientation HDF5 profile stores:

- `bin_edges_A`
- `bin_centers_A`
- `cos_polar_mean`
- `cos_azimuthal_mean`
- `count_total`
- `count_polar_valid`
- `count_azimuthal_valid`
- `cos_polar_density`
- `cos_azimuthal_density`
- `density`
- `heatmap_polar`
- `heatmap_azimuthal`
- `heatmap_angle_bin_edges`
- `heatmap_angle_bin_centers`

It also stores metadata such as:

- `analysis = orientation`
- `axis`
- `reference_axis`
- `n_frames`
- `n_molecules_per_frame`
- `coordinate_mode`
- optional representative cell lengths
- versioned optional spatial-density geometry: cells, origins, surface shifts,
  frame counts, and separate interface water-count accumulators, shared by
  polar and azimuthal quantities

When a surface estimate exists, orientation may additionally persist the shared
surface datasets and metadata described in
[Surface Estimation](surface-estimation.md).

## Important Assumptions And Limitations

- The analysis only applies to water molecules.
- Water identification depends on the O-H cutoff and topology-validation logic.
- Full-cell polar sign follows the chosen Cartesian `reference_axis`;
  interface-relative views apply the negative-side outward sign transformation.
- Trusted surfaces determine both the distance coordinate and interface assignment.
- Azimuthal orientation depends on the chosen in-plane Cartesian reference axis.
- Azimuthal sign also depends on the stable `H1`, `H2` ordering inherited from
  the shared water triplet builder.
- Heatmaps store counts rather than automatically normalized probabilities.

## Related Documentation

- [Water Detection And Water Geometry](water-detection.md)
- [Surface Estimation](surface-estimation.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
