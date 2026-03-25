# Mean-Squared Displacement

`linak compute msd` computes mean-squared displacement relative to the first
frame of the trajectory.

## What Is Being Computed

For the selected atoms, LiNaK computes:

`MSD(i) = mean_j |r_j(i) - r_j(0)|^2`

where:

- `i` is the frame index
- `j` runs over the selected atoms
- `r_j(0)` is the position of atom `j` in the first frame

This is a single-origin MSD. It is not a sliding-window MSD and it is not a
multi-time-origin estimator.

## Atom Selection

LiNaK selects atoms by species:

- a specific element such as `O`
- `all`, meaning every atom

The selected atom count must remain constant across the trajectory. If the
selected species count changes from frame to frame, LiNaK raises an error
instead of silently computing an inconsistent MSD.

## Two Displacement Modes

The code uses one of two displacement models depending on cell availability.

### Periodic Minimum-Image Accumulation

If every frame has:

- a valid non-zero cell
- periodic boundary conditions enabled

LiNaK unwraps motion incrementally using minimum-image steps:

1. compute the displacement from frame `i-1` to frame `i`
2. apply minimum-image correction to that step
3. accumulate corrected steps into an unwrapped trajectory
4. compare the unwrapped position to the frame-0 reference

This is the physically appropriate mode for diffusive trajectories in periodic
simulation cells, because atoms crossing the periodic boundary are treated as
continuing their trajectories rather than jumping discontinuously across the
box.

### Direct Cartesian Fallback

If LiNaK cannot rely on periodic boundary handling for all frames, it falls
back to direct Cartesian displacement:

`r_j(i) - r_j(0)`

without minimum-image correction.

This preserves a usable result for non-periodic or incompletely specified
trajectories, but it will not correct box-crossing jumps.

## Time Axis

LiNaK stores both:

- `time_fs`
- `time_ps`

using:

- `time_fs = frame_index * timestep_fs`
- `time_ps = time_fs / 1000`

The timestep is resolved elsewhere in the CLI pipeline and may come from:

- an explicit CLI argument
- trajectory metadata
- resolved simulation input metadata
- fallback defaults

## What Gets Stored

The MSD HDF5 profile stores:

- `time_fs`
- `time_ps`
- `msd_A2`

and metadata such as:

- `analysis = msd`
- `species`
- `n_frames`
- `units_map`

## Important Assumptions And Limitations

- MSD is referenced to frame `0`; changing the starting frame changes the
  result.
- The method assumes stable atom identity and ordering across frames.
- The periodic mode requires a usable cell in every frame.
- The fallback direct-displacement mode does not correct periodic crossings.
- The result is species-averaged across the selected atoms; it does not store
  per-atom MSD trajectories.

## Related Documentation

- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
