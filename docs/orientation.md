# Orientation

`linak compute orientation` computes water-orientation observables as a function
of distance to a trusted shared surface reference or, when no trusted surface
reference is available, along a raw Cartesian axis.

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

If the shared surface estimator returns a complete and sufficiently trustworthy
frame-wise surface reference, LiNaK uses:

`distance_t = COM_axis,t - s_t`

where `s_t` is the scalar surface reference coordinate for frame `t`.

If the surface estimate is incomplete or below the trust threshold, orientation
does not silently use it anyway. Instead, LiNaK bins the raw center-of-mass
axis coordinate and stores:

`coordinate_mode = "axis"`

Surface estimation shifts only the distance coordinate. It does not redefine the
sign of the orientation observables.

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

This sign convention is tied to `reference_axis`. It is not automatically
re-labeled as "toward" or "away from" the surface, because that would be
ambiguous for different surface sides and symmetric slabs.

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

The orientation heatmaps use `cos(angle)` on the second axis, not the angle in
radians or degrees.

LiNaK stores:

- `heatmap_polar[distance_bin, cosine_bin]`
- `heatmap_azimuthal[distance_bin, cosine_bin]`

These heatmaps store raw counts. Any normalization to probabilities happens
only at plotting time when explicitly requested.

Values exactly equal to `-1` and `+1` are handled deterministically and are
assigned to the first and last cosine bins respectively.

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

When a surface estimate exists, orientation may additionally persist the shared
surface datasets and metadata described in
[Surface Estimation](surface-estimation.md).

## Important Assumptions And Limitations

- The analysis only applies to water molecules.
- Water identification depends on the O-H cutoff and topology-validation logic.
- Polar sign follows the chosen Cartesian `reference_axis`.
- Surface estimation affects the distance coordinate only.
- Azimuthal orientation depends on the chosen in-plane Cartesian reference axis.
- Azimuthal sign also depends on the stable `H1`, `H2` ordering inherited from
  the shared water triplet builder.
- Heatmaps store counts rather than automatically normalized probabilities.

## Related Documentation

- [Water Detection And Water Geometry](water-detection.md)
- [Surface Estimation](surface-estimation.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
