# Density

`linak compute density` computes a one-dimensional density profile along one
Cartesian axis. Depending on available cell information, LiNaK stores both mass
density and `number_density`. The latter is an entity density:

- atoms for atomic selections
- molecules for `H2O`

Depending on available cell information, those outputs are stored either as
volumetric densities or as linear 1D densities.

## What Is Being Computed

For each selected species or molecule, LiNaK bins positions along one axis:

- `x`
- `y`
- `z`

The result is a histogram-based profile over 1D bins. LiNaK stores:

- `density`: mass density
- `number_density`: number density

The stored profile is a time average over frame-wise histograms, so it
represents the mean sampled distribution rather than a single-frame snapshot.

## Supported Selections

### Atomic Selections

If a chemical symbol such as `O`, `H`, `Li`, or `Pt` is selected, LiNaK bins
the axis coordinate of each matching atom.

### Water Selection

If `H2O` is selected, LiNaK first identifies genuine water molecules and then
uses the water center of mass, not the individual O or H coordinates. Water
detection and geometry handling are described in
[Water Detection And Water Geometry](water-detection.md).

### `all` Selection

If `--species all` is used, LiNaK generates one density profile per detected
element species and, when water molecules are detected, one additional `H2O`
profile.

## Coordinate System Used For Binning

LiNaK first resolves the coordinate values to bin:

- for atomic selections: the selected atomic coordinate along the chosen axis
- for `H2O`: the water center-of-mass coordinate along that axis

LiNaK then decides whether those coordinates should be binned as:

- raw axis coordinates
- surface-relative coordinates

If the shared surface estimator returns a complete and sufficiently trustworthy
frame-wise surface reference, LiNaK shifts the selected coordinates frame by
frame:

`d_t = x_t - s_t`

where `x_t` is the selected axis coordinate and `s_t` is the scalar surface
reference coordinate for frame `t`.

If the surface estimate is incomplete or below the trust threshold, LiNaK does
not silently use it anyway. It falls back to raw axis coordinates and stores

`coordinate_mode = "axis"`

So density uses `coordinate_mode = "distance"` only when the selected surface
estimate is complete and trusted, not merely because some scalar surface summary
exists.

Surface estimation is described in
[Surface Estimation](surface-estimation.md).

## Histogram Range

LiNaK supports two conceptual histogram-range modes.

### `observed`

The histogram spans the minimum and maximum values actually observed in the
selected coordinates for that one profile.

For `compute_density_profiles(..., species="all")`, this means each element
profile uses its own observed-range grid. LiNaK does not currently force a
shared observed-range grid across species in the same analysis call.

### `cell`

If a usable periodic cell is available, LiNaK can instead span the geometric
extent of the simulation cell along the chosen axis. If surface alignment is
active, this cell interval is shifted by the frame-wise surface reference used
for the trusted distance-mode alignment.

If the cell is unusable or unavailable, LiNaK falls back to observed bounds.

## Histogram Construction

After the bounds are known, LiNaK:

1. computes the number of bins from `bin_width`
2. constructs uniform bin edges
3. bins every frame independently
4. accumulates mass and entity counts across frames

For each frame and selected entity, LiNaK contributes:

- its mass to the mass histogram
- `1` to the entity-count histogram

If `M_k^(t)` is the mass accumulated in bin `k` for frame `t`, and `N_k^(t)` is
the entity count, then the stored profile is built from the frame-wise sums of
those quantities.

Bin edges are fixed once per profile before any frame histogram is computed.
They are not recomputed per frame.

LiNaK uses NumPy histogram semantics:

- bins are left-inclusive and right-exclusive
- the final bin includes its right edge

## Volumetric vs Linear Density

LiNaK chooses between two normalization modes.

### Volumetric Mode

If every frame has a usable periodic cell along the chosen axis, LiNaK computes
the cross-sectional area perpendicular to that axis and multiplies it by the
bin width to obtain a slab volume.

For the selected axis vector `a_axis`, LiNaK uses:

- `axis_length = ||a_axis||`
- `cross_section = cell_volume / axis_length`
- `V_slab^(t) = cross_section^(t) * bin_width`

Mass density is then:

`rho_mass(k) = mean_t(M_k^(t) / V_slab^(t))`

Entity density is:

`rho_number(k) = mean_t(N_k^(t) / V_slab^(t))`

When the slab volume is constant across frames, this reduces to:

- `rho_mass(k) = M_k / (n_frames * V_slab)`
- `rho_number(k) = N_k / (n_frames * V_slab)`

Stored units are:

- mass density: `g/cm^3`
- entity density: `atom/nm^3` for atomic selections, `molecule/nm^3` for `H2O`

If slab volume changes between frames, LiNaK normalizes frame by frame before
averaging rather than assuming a fixed cell. If slab volume is constant across
frames, LiNaK can safely accumulate raw histograms first and normalize once at
the end.

### Linear Fallback Mode

If a usable periodic cell is not available in all frames, LiNaK falls back to a
one-dimensional normalization:

- mass density is divided by bin width only
- entity density is divided by bin width only

Stored units are:

- mass density: `g/Angstrom`
- entity density: `atom/Angstrom` for atomic selections, `molecule/Angstrom`
  for `H2O`

This preserves a meaningful 1D mass-per-length or entity-per-length profile
even when volumetric normalization would be physically underdefined. It is not
a volumetric density.

## Surface-Aware Interpretation

When `coordinate_mode = "distance"`, the stored bin centers are already
distance-to-surface values. The stored `surface_position` is only a summary over
the frame-wise surface reference array. It is not the one single value used for
all frames.

When `coordinate_mode = "axis"`, the stored bins should be interpreted as raw
axis coordinates, even though the file may still carry surface summary metadata
or full surface diagnostics.

## What Gets Stored

The density HDF5 profile normally stores:

- `bin_centers_A`
- `density`
- `number_density`

It also stores metadata such as:

- `analysis = density`
- `species`
- `axis`
- `n_frames`
- `coordinate_mode`
- optional nested `surface` metadata
- `units_map`

When a surface estimate exists, density may additionally persist the shared
surface datasets:

- `surface_position_per_frame_A`
- `surface_valid_mask`
- `surface_confidence`
- `surface_provenance`
- `surface_candidate_count`
- `surface_top_layer_size`
- `surface_largest_gap_A`
- `surface_baseline_gap_A`
- `surface_reference_spread_A`
- `surface_jump_rejection_mask`
- `surface_rejection_reason`

and shared surface metadata under `surface`, such as:

- `surface.position`
- `surface.position_std`
- `surface.mode`
- `surface.side`
- `surface.selected_elements`
- `surface.candidate_indices`
- `surface.method_label`
- `surface.valid_fraction`
- `surface.median_confidence`
- `surface.composite_score`
- `surface.low_confidence_threshold`
- `surface.effective_options`

### Notes On Legacy / Reconstructed Fields

LiNaK can still reconstruct or load some additional arrays in compatibility
paths, such as:

- `bin_edges_A`
- `counts_per_frame`
- `entities_per_frame`

But these are not part of the normal current write payload for density files.

## Important Assumptions And Limitations

- The method is histogram-based and therefore depends on the chosen `bin_width`.
- Water density depends on the internal water-detection logic and cutoff.
- Surface-aware density depends on the completeness and confidence of the shared
  surface estimator, not merely on whether some scalar surface summary exists.
- Linear fallback should not be interpreted as a volumetric density.
- The method averages over frames and therefore removes time ordering.

## Related Documentation

- [Surface Estimation](surface-estimation.md)
- [Water Detection And Water Geometry](water-detection.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
