# Orientation

`linak compute orientation` computes water-orientation observables as a function
of distance to a surface or, when surface alignment is unavailable, along a raw
Cartesian axis.

## What Is Being Computed

LiNaK identifies water molecules, computes water geometry for each frame, and
then stores orientation statistics in distance bins.

The primary outputs are:

- `cos_polar_mean`
- `cos_azimuthal_mean`
- `cos_polar_density`
- `cos_azimuthal_density`
- `heatmap_polar`
- `heatmap_azimuthal`

This analysis is water-specific.

## Water Geometry Used

LiNaK first identifies water molecules and computes PBC-aware positions for:

- oxygen
- hydrogen 1
- hydrogen 2
- water center of mass

That shared logic is described in
[Water Detection And Water Geometry](water-detection.md).

## Distance Coordinate

Each water molecule is assigned a distance-bin coordinate using its center of
mass along the selected axis.

If a valid frame-wise surface reference exists, LiNaK uses:

`distance = COM_axis - surface_position(frame)`

Otherwise it bins the raw axis coordinate and marks the result as
`coordinate_mode = "axis"`.

Surface estimation is described in [Surface Estimation](surface-estimation.md).

## Polar Orientation

LiNaK builds the water bisector from the two normalized O-H vectors and compares
that bisector to the chosen reference axis.

The stored quantity is:

`cos_polar = bisector dot reference_axis`

If the normalized O-H bond vectors are `u_1` and `u_2`, LiNaK builds the
normalized bisector

`b = (u_1 + u_2) / |u_1 + u_2|`

and then computes:

`cos_polar = b . e_ref`

where `e_ref` is the unit vector along the chosen reference axis.

Interpretation:

- `+1`: H atoms point away from the reference direction
- `-1`: H atoms point toward the reference direction
- `0`: the water bisector is perpendicular to the reference direction

## Azimuthal Orientation

LiNaK also computes the water-plane normal from:

`plane_normal = OH1 x OH2`

After normalization, that normal is projected into the plane perpendicular to
the reference axis. If the remaining in-plane components are `(p_1, p_2)`,
LiNaK stores:

`cos_azimuthal = p_1 / sqrt(p_1^2 + p_2^2)`

It then projects that normal into the plane perpendicular to the reference axis
and stores:

`cos_azimuthal`

relative to the first available Cartesian in-plane axis.

This is useful as an orientation descriptor, but it does not define a unique
global in-plane physical direction in the same strong sense as the polar
quantity.

## Distance-Bin Accumulation

For each water molecule in each frame, LiNaK:

1. assigns the molecule to a distance bin
2. accumulates `cos_polar`
3. accumulates `cos_azimuthal`
4. increments the molecule count in that bin

At the end, LiNaK divides the accumulated angle sums by the count per bin to
obtain:

- `cos_polar_mean`
- `cos_azimuthal_mean`

Explicitly, for distance bin `k`:

- `cos_polar_mean(k) = sum cos_polar / count_k`
- `cos_azimuthal_mean(k) = sum cos_azimuthal / count_k`

## Density-Weighted Orientation

LiNaK also computes a water number density profile over the same distance bins.

It then forms:

- `cos_polar_density = cos_polar_mean * density`
- `cos_azimuthal_density = cos_azimuthal_mean * density`

These density-weighted quantities are useful when you want orientational bias
and molecular population to be represented together.

## 2D Orientation Heatmaps

In addition to 1D averages, LiNaK bins the orientation cosines into a second
angle axis running from `-1` to `+1`.

This yields two 2D histograms:

- `heatmap_polar[distance_bin, angle_bin]`
- `heatmap_azimuthal[distance_bin, angle_bin]`

The heatmaps store counts, not normalized probabilities.

## Density Normalization

If a usable periodic cell is known, LiNaK normalizes the water count per
distance bin by the corresponding slab volume and stores a volumetric number
density. Otherwise it falls back to a linear density along the chosen axis.

## What Gets Stored

The orientation HDF5 profile stores:

- `bin_edges_A`
- `bin_centers_A`
- `cos_polar_mean`
- `cos_azimuthal_mean`
- `cos_polar_density`
- `cos_azimuthal_density`
- `density`
- `heatmap_polar`
- `heatmap_azimuthal`
- `heatmap_angle_bin_edges`
- `heatmap_angle_bin_centers`

and metadata such as:

- `analysis = orientation`
- `axis`
- `reference_axis`
- `n_frames`
- `n_molecules_per_frame`
- `coordinate_mode`
- optional surface summary metadata
- optional cell lengths

## Important Assumptions And Limitations

- The analysis only applies to water molecules.
- Water identification depends on the O-H cutoff and topology-validation logic.
- Azimuthal orientation is reference-frame-dependent in the in-plane direction.
- Heatmaps store counts rather than automatically normalized probabilities.
- Surface-aware interpretation depends on the success of the surface estimator.

## Related Documentation

- [Water Detection And Water Geometry](water-detection.md)
- [Surface Estimation](surface-estimation.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
