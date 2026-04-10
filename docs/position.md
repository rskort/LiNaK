# Position

`linak compute position` stores atom-resolved trajectories for selected species.
Unlike density or RDF, this analysis preserves per-atom and per-frame structure.

## What Is Being Computed

For each selected atom and each frame, LiNaK stores:

- `x_A`
- `y_A`
- `z_A`
- `distance_to_surface_A`

The arrays are shaped as:

- rows: frames
- columns: tracked atoms

This is therefore an atom-resolved time series, not an averaged profile.

## Atom Identity Requirements

Position tracking assumes a stable atom layout across the trajectory:

- same number of atoms in every frame
- same atom ordering in every frame
- same chemical-symbol layout in every frame

LiNaK validates this before computing the result. If the trajectory changes atom
count or atom ordering, the analysis stops with an error rather than silently
mixing atom identities.

## Species Selection

If a specific species is requested, LiNaK identifies the indices of that
species in the reference layout and tracks those exact columns throughout the
trajectory.

If `--species all` is used, LiNaK computes one profile per element species.

## Coordinate Handling

When a usable periodic cell is available, LiNaK stores PBC-corrected
(wrapped) Cartesian coordinates. The source trajectory files are not modified.
When no usable cell is available, LiNaK stores the raw Cartesian coordinates
directly from the trajectory. In both cases the stored arrays are:

- `x_A`
- `y_A`
- `z_A`

It also computes a fourth coordinate-like quantity,
`distance_to_surface_A`, based on the selected analysis axis.

## Surface Distance Logic

LiNaK first estimates a surface reference along the chosen axis using the same
machinery as the density and coordination workflows. See
[Surface Estimation](surface-estimation.md).

If a valid frame-wise surface reference exists, LiNaK stores:

`distance_to_surface(frame, atom) = axis_coordinate(frame, atom) - surface_position(frame)`

If frame-wise alignment is not available, LiNaK still fills
`distance_to_surface_A`, but with the raw axis coordinate instead. In that case
it marks the profile as:

- `coordinate_mode = "axis"`

When the true frame-wise surface distance is available, LiNaK sets:

- `coordinate_mode = "distance"`

This distinction matters. The dataset name `distance_to_surface_A` remains the
same in both cases, but its semantics depend on `coordinate_mode`.

## Time Axes

LiNaK stores four time-like axes:

- `frame_index`
- `step`
- `time_fs`
- `time_ps`

`step` is taken from frame metadata when available. If no usable step metadata
is found, LiNaK falls back to the frame index.

`time_fs` and `time_ps` are generated from the resolved timestep.

In the current implementation:

- `time_fs(frame) = frame_index * timestep_fs`
- `time_ps(frame) = time_fs / 1000`

## What Gets Stored

The position HDF5 profile stores:

- `atom_indices`
- `frame_index`
- `step`
- `time_fs`
- `time_ps`
- `x_A`
- `y_A`
- `z_A`
- `distance_to_surface_A`
- optionally `surface_position_per_frame_A`

and metadata such as:

- `analysis = position`
- `species`
- `axis`
- `n_frames`
- `n_atoms`
- `coordinate_mode`
- optional `surface_position`
- optional `surface_position_std`
- optional `cell_lengths_angstrom`

## Why This Analysis Exists

The position analysis is the shared atom-tracking backbone for several other
features:

- direct time-vs-position plotting
- configurable 2D trajectory projection plotting
- coordination as a function of time and distance to surface

In particular, the coordination workflow reuses the atom tracking and
surface-distance logic produced here.

## Important Assumptions And Limitations

- Atom identity must remain stable across the entire trajectory.
- The result stores per-atom matrices, so file sizes can grow quickly for large
  systems or long trajectories.
- `distance_to_surface_A` must be interpreted together with `coordinate_mode`.
- Surface-aware output quality depends on the success of the surface estimator.

## Plotting Notes

The stored position profile can be plotted in two broad ways:

- 1D time-based components such as `distance`, `x`, `y`, and `z`
- `2d-projection` (legacy alias: `xy-z`)

The 2D projection plot is a plot-only view built from the stored matrices. It
does not change the HDF5 data model. The projection view lets the user choose:

- which stored quantity is used on the X axis
- which stored quantity is used on the Y axis
- which quantity drives the colormap or value filter
- whether rendering uses one colormap-driven overlay (`color-scale`) or one
  normal layer per tracked atom (`line-colors`)

Projection value filters are pointwise masks on the chosen value quantity.
Hidden points do not get bridged by artificial connector lines.

## Related Documentation

- [Surface Estimation](surface-estimation.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
