# Coordination

`linak compute coordination` computes a continuous coordination number for each
selected center atom as a function of time and, when surface alignment is
available, distance to a surface.

## What Is Being Computed

For every frame and every selected center atom of species `A`, LiNaK computes a
continuous coordination number with respect to neighbor atoms of species `B`.

The stored result is atom-resolved and time-resolved:

- `coordination_number[T, N_center]`
- `distance_to_surface_A[T, N_center]`

This is not just one average CN per frame. Each tracked center atom keeps its
own coordination trajectory.

## Step 1: Reuse Position Tracking For Species `A`

LiNaK begins by computing a position profile for species `A`. This provides:

- the tracked center-atom indices
- `frame_index`
- `step`
- `time_fs`
- `time_ps`
- `distance_to_surface_A`

As a result, coordination inherits the same:

- atom-identity assumptions
- timestep handling
- surface-distance logic

documented in [Position](position.md) and [Surface Estimation](surface-estimation.md).

## Step 2: Resolve Neighbor Atoms

LiNaK identifies the neighbor selection for species `B`. If `species_b` is not
given, it defaults to the same species token as `species_a`.

If `A == B`, self-pairs are excluded when computing neighbor contributions for a
given center atom.

## Step 3: Resolve The Coordination Cutoff

The coordination cutoff can be resolved in three ways.

### Direct User Cutoff

If `--cutoff` is given, LiNaK uses that value directly.

### From An Existing RDF File

If `--cutoff-rdf <file>` is used, LiNaK loads the RDF and determines the
cutoff from that curve.

### From A Sampled RDF Of The Current Trajectory

If `--cutoff-from-rdf` is used, LiNaK computes a reference RDF from sampled
trajectory frames and resolves the cutoff from that result.

## How RDF-Based Cutoff Resolution Works

For RDF-based cutoff resolution, LiNaK:

1. computes or loads an RDF curve
2. smooths it with a Gaussian kernel
3. identifies the first RDF peak
4. finds the first minimum after that peak
5. locally refines the minimum position with a quadratic fit

That refined first minimum becomes the coordination cutoff.

If requested, LiNaK also writes a diagnostic figure showing:

- raw RDF
- smoothed RDF
- resolved peak
- selected cutoff

When RDF provenance exists, LiNaK stores it in the coordination HDF5 so the
origin of the cutoff remains inspectable.

### Smoothing And Peak/Minimum Logic

LiNaK smooths the RDF with a Gaussian kernel. In bin units, the kernel is:

`K(n) = exp(-0.5 * (n / sigma_bins)^2)`

normalized so that the kernel sum is `1`.

The smoothing width in bins is computed from:

`sigma_bins = max(smoothing_sigma_A / mean_bin_width, 1.0)`

LiNaK then:

1. finds the first local peak in the smoothed RDF
2. finds the first local minimum after that peak
3. fits a quadratic polynomial locally around that minimum
4. uses the vertex of that quadratic, clipped to the local fit window, as the
   refined cutoff

## Step 4: Convert Pair Distances To Continuous Weights

LiNaK does not use a hard integer cutoff. Instead, each pair distance `r`
contributes a smooth weight controlled by:

- `cutoff_A`
- `cutoff_smoothing_width_A`

Conceptually:

- distances well inside the cutoff contribute approximately `1`
- distances well outside contribute approximately `0`
- distances near the cutoff contribute a fractional weight

The coordination number for one center atom in one frame is the sum of those
weights over all selected neighbors.

If the pair weights are written as `w(r_ij)`, then for center atom `i`:

`CN_i = sum_j w(r_ij)`

This means the stored coordination number is continuous rather than strictly
integer-valued.

## Execution Modes

LiNaK uses one of two internal kernels.

### Framewise Kernel

For smaller or less favorable workloads, LiNaK computes one frame at a time.
When a usable periodic cell exists, it uses a neighbor-list-based distance
search up to the support cutoff. Otherwise it falls back to direct Cartesian
distances.

### Chunked Vectorized Kernel

For larger but still vectorizable workloads, LiNaK stacks multiple frames,
builds center-neighbor distance blocks, optionally applies minimum-image
correction, and evaluates the continuous weights in chunks.

Both kernels target the same quantity. The difference is computational strategy,
not method definition.

## Surface Distance Storage

Because coordination reuses the position workflow for the center atoms,
`distance_to_surface_A` follows the same interpretation rules:

- `coordinate_mode = "distance"` means genuine frame-wise distance to surface
- `coordinate_mode = "axis"` means raw axis coordinate fallback

## What Gets Stored

The coordination HDF5 profile stores:

- `atom_indices`
- `frame_index`
- `step`
- `time_fs`
- `time_ps`
- `distance_to_surface_A`
- `coordination_number`
- optionally `surface_position_per_frame_A`

It may also store RDF provenance for cutoff resolution:

- sampled or loaded RDF bin centers
- raw `g(r)`
- smoothed `g(r)`
- resolved RDF peak position
- resolved RDF minimum position
- RDF source path
- diagnostic plot path

Metadata includes:

- `analysis = coordination`
- `species_a`
- `species_b`
- `axis`
- `n_frames`
- `n_atoms`
- `cutoff_A`
- `cutoff_smoothing_width_A`
- `cutoff_mode`
- `coordinate_mode`
- optional surface summary metadata
- `units_map`

## Important Assumptions And Limitations

- Atom identity must remain stable across frames for the center selection.
- Coordination values depend strongly on the cutoff strategy.
- RDF-derived cutoffs are only as reliable as the RDF quality.
- Continuous coordination numbers are not directly comparable to a hard integer
  CN unless the smoothing model is taken into account.
- Large systems can produce large atom-resolved matrices.

## Related Documentation

- [Position](position.md)
- [RDF](rdf.md)
- [Surface Estimation](surface-estimation.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
