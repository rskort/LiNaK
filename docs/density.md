# Density

`linak compute density` computes a one-dimensional density profile along one
Cartesian axis. Depending on available cell information, LiNaK stores both mass
density and number density either as volumetric densities or as linear
densities.

## What Is Being Computed

For each selected species or molecule, LiNaK bins positions along one axis:

- `x`
- `y`
- `z`

The result is a histogram-based profile over distance bins. LiNaK stores:

- `density`: mass density
- `number_density`: number density

The density is averaged over frames, so the stored profile represents the mean
distribution over the sampled trajectory, not a single-frame snapshot.

## Supported Selections

### Atomic Selections

If a chemical symbol such as `O`, `H`, `Li`, or `Pt` is selected, LiNaK bins
the axis coordinate of each matching atom.

### Water Selection

If `H2O` is selected, LiNaK first identifies genuine water molecules and then
uses the molecule center of mass, not the individual O or H coordinates. Water
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

LiNaK then decides whether to bin:

- raw axis coordinates
- frame-wise distance to a surface reference

If a valid frame-wise surface reference exists, LiNaK shifts the selected
coordinates frame by frame:

`distance_to_surface(frame) = coordinate(frame) - surface_position(frame)`

If no usable frame-wise surface reference exists, LiNaK falls back to binning
raw axis coordinates and marks the result as `coordinate_mode = "axis"`.

Surface estimation is described in more detail in
[Surface Estimation](surface-estimation.md).

## Histogram Range

LiNaK supports two conceptual histogram-range modes.

### `observed`

The histogram spans the minimum and maximum values actually observed in the
selected coordinates.

### `cell`

If a usable periodic cell is available, LiNaK can instead span the geometric
extent of the simulation cell along the chosen axis. If surface alignment is
active, this cell interval is shifted by the representative surface offset.

If the cell is unusable or unavailable, LiNaK falls back to observed bounds.

## Histogram Construction

After the bounds are known, LiNaK:

1. computes the number of bins from `bin_width`
2. constructs uniform bin edges
3. bins every frame independently
4. accumulates mass and counts across frames

For each frame and selected entity, LiNaK contributes:

- its mass to the mass histogram
- `1` to the entity-count histogram

This yields per-bin accumulated mass and accumulated entity counts.

If `M_k^(t)` is the mass accumulated in bin `k` for frame `t`, and `N_k^(t)` is
the entity count, then the stored profile is built from the framewise sums of
those quantities.

## Volumetric vs Linear Density

LiNaK chooses between two normalization modes.

### Volumetric Mode

If every frame has a usable periodic cell along the chosen axis, LiNaK computes
the cross-sectional area perpendicular to that axis and multiplies it by the
bin width to obtain a slab volume.

Mass density is then:

`mass density = mean_frame(mass in bin / slab volume)`

Number density is:

`number density = mean_frame(entity count in bin / slab volume)`

When the slab volume is constant across frames, this is equivalent to:

- `rho_mass(k) = M_k / (n_frames * V_slab)`
- `rho_number(k) = N_k / (n_frames * V_slab)`

Stored units are:

- mass density: `g/cm^3`
- number density: `atom/nm^3`

If slab volume changes between frames, LiNaK normalizes frame by frame before
averaging instead of assuming a fixed cell.

### Linear Fallback Mode

If a usable periodic cell is not available in all frames, LiNaK falls back to a
one-dimensional normalization:

- mass density is divided by bin width only
- number density is divided by bin width only

Stored units are:

- mass density: `g/Angstrom`
- number density: `atom/Angstrom`

This is intentionally a fallback. It preserves a meaningful 1D profile even
when volumetric normalization would be physically underdefined.

## Surface-Aware Interpretation

When `coordinate_mode = "distance"`, the stored bin centers are already
distance-to-surface values. The stored `surface_position` is only a summary
statistic over the frame-wise surface estimates. It is not the one single value
used for all frames.

When `coordinate_mode = "axis"`, the stored bins should be interpreted as raw
axis coordinates.

## What Gets Stored

The density HDF5 profile stores, at minimum:

- `bin_edges_A`
- `bin_centers_A`
- `density`
- `counts_per_frame`
- `number_density`
- `entities_per_frame`

It also stores metadata such as:

- `analysis = density`
- `species`
- `axis`
- `n_frames`
- `coordinate_mode`
- optional `surface_position`
- optional `surface_position_std`
- `units_map`

## Important Assumptions And Limitations

- The method is histogram-based and therefore depends on the chosen `bin_width`.
- Water density depends on the internal water-detection logic and cutoff.
- Surface-aware density depends on the success of the surface estimator.
- Linear-density fallback should not be interpreted as a volumetric density.
- The method averages over frames and therefore removes time ordering.

## Related Documentation

- [Surface Estimation](surface-estimation.md)
- [Water Detection And Water Geometry](water-detection.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
