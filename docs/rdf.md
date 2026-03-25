# Radial Distribution Function

`linak compute rdf` computes the radial distribution function `g(r)` for a pair
of species, averaged over all frames.

## What Is Being Computed

LiNaK estimates how often species `B` appears at distance `r` from species `A`,
relative to the distance distribution expected for an ideal spatially uniform
reference state.

The final quantity is the familiar RDF:

`g(r) = observed pair counts / expected pair counts`

after accumulation over all frames and all selected centers.

More concretely, for bin `k` LiNaK stores:

`g_k = C_k / E_k`

where:

- `C_k` is the accumulated observed pair count in bin `k`
- `E_k` is the accumulated expected ideal-gas count in bin `k`

## Species Selection

LiNaK selects:

- centers of species `A`
- neighbors of species `B`

If `A == B`, self-pairs are excluded.

The computation is directional in implementation terms, but for ordinary
species-resolved RDF usage the result corresponds to the standard pair
distribution between the two selections.

## Cell Requirement

RDF requires a usable periodic cell and non-zero volume. The method relies on
geometric shell normalization, which is only well-defined when the simulation
volume is known.

If `r_max` is not explicitly provided, LiNaK chooses it from the trajectory
geometry as half of the smallest cell-vector length encountered, ensuring that
the RDF does not probe beyond the meaningful periodic range.

## Bin Construction

LiNaK constructs uniform bins from `0` to `r_max`.

If the requested bin width does not divide `r_max` exactly, LiNaK adjusts the
effective bin width slightly so that:

- bins remain uniform
- the last bin edge lands exactly at `r_max`

This avoids an irregular final bin.

## Per-Frame Pair Sampling

For each frame, LiNaK first resolves the selected atoms and then collects the
relevant pair distances up to `r_max`.

The code can choose among three internal strategies:

- cutoff neighbor list
- selected-distance submatrix
- full minimum-image distance matrix

This choice is performance-driven. It does not change the intended numerical
quantity being estimated.

## Per-Frame Normalization

For each frame, LiNaK computes:

1. observed counts in each `r` bin
2. expected counts for an ideal uniform reference

The expected counts are based on:

- shell volumes
- the number of selected center atoms
- the number of selected neighbor atoms
- the frame cell volume

For one frame, the expected count is built as:

`E_k^(frame) = N_A * rho_B * V_k`

where:

- `N_A` is the number of selected center atoms
- `rho_B` is the number density of the neighbor selection
- `V_k` is the spherical shell volume of bin `k`

For same-species RDFs, LiNaK uses:

`rho_B = (N_B - 1) / V`

instead of `N_B / V` to exclude self-pairs.

LiNaK accumulates observed and expected contributions separately over all
frames, then divides at the end:

`g(r) = accumulated_observed / accumulated_expected`

for bins where the expected count is non-zero.

This is better than normalizing each frame independently and then averaging,
because it keeps the counting logic explicit and stable.

## Parallel Execution

When beneficial, LiNaK splits the frame list into chunks and processes them in
worker processes. The stored RDF is still the same sum over frame-level observed
and expected contributions.

Parallel execution therefore changes throughput, not the mathematical target.

## What Gets Stored

The RDF HDF5 profile stores:

- `bin_edges_A`
- `bin_centers_A`
- `g_r`

and metadata such as:

- `analysis = rdf`
- `species_a`
- `species_b`
- `n_frames`
- effective `bin_width_A`
- `units_map`

## Important Assumptions And Limitations

- RDF requires a valid periodic cell and non-zero volume.
- The chosen `bin_width` affects smoothness and noise.
- The chosen `r_max` limits the physically interpreted range.
- Small selections or short trajectories can produce noisy RDFs.
- The code assumes the trajectory geometry and species labels are meaningful for
  distance-based pair statistics.

## Why RDF Matters Elsewhere In LiNaK

The coordination workflow can use RDF output to determine a physically
motivated coordination cutoff by locating the first minimum after the first RDF
peak. That logic is documented in [Coordination](coordination.md).

## Related Documentation

- [Coordination](coordination.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
