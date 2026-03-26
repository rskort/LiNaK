# Radial Distribution Function

`linak compute rdf` computes the radial distribution function `g(r)` for a pair
of species, averaged over all frames.

## What Is Being Computed

LiNaK estimates how often species `B` appears at distance `r` from species `A`,
relative to the distance distribution expected for an ideal spatially uniform
reference state.

For bin `k`, LiNaK stores:

`g_k = C_k / E_k`

where:

- `C_k` is the accumulated observed pair count in bin `k`
- `E_k` is the accumulated expected ideal-gas count in bin `k`

LiNaK accumulates `C_k` and `E_k` over frames first, then divides once at the
end. It does not normalize each frame independently and average the resulting
curves.

## Pair-Counting Convention

LiNaK uses an **ordered-pair** counting convention internally.

That means the pair `i -> j` is counted separately from `j -> i` whenever both
belong to the selected center/neighbor relation.

For ordinary species-resolved RDFs this matches the standard interpretation,
because the expected ideal-gas count is built with the same convention.

If `species_a == species_b`, LiNaK excludes self-pairs, but still counts the
ordered non-self pairs:

- `i -> j`
- `j -> i`

So same-species RDFs are self-excluded ordered-pair RDFs.

## Species Selection

LiNaK selects:

- centers of species `A`
- neighbors of species `B`

For frame `t`, let:

- `N_A^(t)` be the number of selected center atoms
- `N_B^(t)` be the number of selected neighbor atoms
- `V^(t)` be the frame volume

Selections are resolved per frame. If atom identities or counts vary across the
trajectory, the RDF normalization follows those per-frame values.

## Cell Requirement

RDF requires:

- a valid `3 x 3` cell
- fully periodic boundary conditions in all three directions
- non-zero volume

LiNaK fails explicitly if any frame does not satisfy those requirements.

## Automatic `r_max`

If `r_max` is not provided, LiNaK chooses a safe default from the periodic cell
geometry of the trajectory.

For each frame it computes the three perpendicular cell heights:

- `h_a = V / ||b x c||`
- `h_b = V / ||c x a||`
- `h_c = V / ||a x b||`

It then uses:

`r_max = 0.5 * min(h_a, h_b, h_c)`

across all frames.

For orthorhombic cells this reduces to half of the shortest box length. For
skewed or triclinic cells it is the safer half-minimum-perpendicular-height
criterion.

## Bin Construction

LiNaK constructs uniform bins from `0` to `r_max`.

If the requested bin width does not divide `r_max` exactly, LiNaK chooses the
nearest integer bin count and then rebuilds a uniform grid so that:

- all bins have the same width
- the final edge lands exactly at `r_max`

This keeps the stored grid deterministic and avoids an irregular final bin.

## Shell Volumes

LiNaK uses the exact spherical-shell volume for each RDF bin:

`V_k = 4/3 pi (r_outer^3 - r_inner^3)`

No thin-shell approximation is used.

## Per-Frame Pair Sampling

For each frame, LiNaK resolves the selected atoms and then collects pair
distances up to `r_max`.

The implementation can choose among three internal strategies:

- cutoff neighbor list
- selected-distance submatrix
- full minimum-image distance matrix

These are performance choices only. They are intended to produce the same
physical pair distances under periodic boundary conditions.

The cutoff neighbor-list path uses a cutoff at least as large as the next
floating-point number above `r_max`, so pairs exactly at `r_max` are not missed
by the search radius itself.

## Histogram Edge Semantics

LiNaK uses NumPy histogram semantics for RDF bins:

- bins are left-inclusive and right-exclusive
- the final bin includes its right edge

So:

- `r = 0` is assigned to the first bin
- distances exactly on interior bin edges go to the higher bin
- `r = r_max` is assigned to the final bin

## Per-Frame Normalization

For frame `t`, the expected count in shell `k` is:

`E_k^(t) = N_A^(t) * rho_B^(t) * V_k`

with:

- cross-species RDF: `rho_B^(t) = N_B^(t) / V^(t)`
- same-species RDF: `rho_B^(t) = (N_B^(t) - 1) / V^(t)`

The same-species form matches the observed self-excluded ordered-pair counting.

LiNaK accumulates:

- `C_k = sum_t C_k^(t)`
- `E_k = sum_t E_k^(t)`

and then stores:

`g_k = C_k / E_k`

for bins where `E_k > 0`.

If a bin has zero expected count, LiNaK stores `NaN` rather than silently
writing `0`.

## Parallel Execution

When beneficial, LiNaK splits frames into chunks and processes them in worker
processes.

Parallel execution changes throughput only. The stored RDF remains the same sum
over frame-level observed and expected contributions.

## What Gets Stored

The RDF HDF5 profile stores:

- `bin_centers_A`
- `g_r`

and metadata such as:

- `analysis = rdf`
- `species_a`
- `species_b`
- `n_frames`
- effective `bin_width_A`
- `units_map`

`bin_edges_A` may appear in older or explicitly written files, but the current
minimal RDF payload stores `bin_centers_A` plus `bin_width_A`, and LiNaK
reconstructs uniform bin edges on load when needed.

## Important Assumptions And Limitations

- RDF requires fully periodic cells with non-zero volume.
- The chosen `bin_width` affects smoothness and noise.
- The chosen `r_max` limits the physically interpreted range.
- Small selections or short trajectories can produce noisy RDFs.
- Ordinary species-resolved usage is the intended interpretation of the RDF
  normalization.

## Why RDF Matters Elsewhere In LiNaK

The coordination workflow can use RDF output to determine a physically
motivated coordination cutoff by locating the first minimum after the first RDF
peak. That logic is documented in [Coordination](coordination.md).

## Related Documentation

- [Coordination](coordination.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
