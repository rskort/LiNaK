# Density

The plane-based and Cartesian physical-density methods are described below.
With `--surface-distance-mode nearest-atom`, LiNaK instead computes mass or
number **distance distributions**, in amu/Å or atoms/molecules per Å. Supported
maps combine nearest distance with X or Y and use squared-coordinate units.
They do not use physical bin volumes or full-cell Cartesian views. See
[density and distance distributions](density-distributions.md) for the definitions,
Studio controls, compatibility rules and measured performance.

## Symmetric Cells

With `surface_layout=symmetric`, density stores folded, top, and bottom
outward-distance profiles. Each side is normalized using its own per-frame
slice volume. Folded density is the equal arithmetic mean of the two normalized
profiles, so identical interfaces retain one-interface magnitude instead of
doubled density. Side-resolved mass and entity accumulators are retained in the
sparse grid without duplicating Cartesian coordinates. Raw X/Y/Z profiles
remain available.

Plotting defaults to `interface_view=folded`; `top` and `bottom` select the
positive- and negative-axis normalized sides, while `all` overlays the three
labeled 1D interface curves. `interface_view=full` loads the existing raw
surface-axis profile and shows the entire periodic cell, including both liquid
regions and the slab, without folding or side masking. In the Studio these are
named **Combined interfaces (folded mean)**, **Positive-axis interface**,
**Negative-axis interface**, **Compare interface curves**, and **Full cell
(unfolded)**. Full-cell and default 1D previews use stored profiles and do not
require a sparse-grid read.

The Studio shows **Surface representation** only for profiles resolved as
`symmetric`. Single-top, single-bottom, and raw-axis profiles have no
two-interface selector. In **Full cell (unfolded)**, the raw surface-axis
coordinate is enforced in the mapping controls; folded distance cannot be
selected until an interface-relative view is restored.

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
extent of the simulation cell along the chosen axis. Symmetric layouts use the
paired folded bounds. A single-interface layout spans one signed slab interval
and the complete positive outward fluid/vacuum arc, including any crossing of
the raw periodic boundary.

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

### Stored Bin Geometry

Current v2 density HDF5 profiles store uniform bin geometry as:

- `bin_centers_A`
- metadata `bin_width_A`

LiNaK reconstructs uniform bin edges from these fields when loading.
Files that only provide older bin-edge-style metadata are rejected and should be
recomputed with the current package.

Density files may also include optional current-schema arrays such as:

- `counts_per_frame`
- `entities_per_frame`

## Important Assumptions And Limitations

- The method is histogram-based and therefore depends on the chosen `bin_width`.
- Water density depends on the internal water-detection logic and cutoff.
- Surface-aware density depends on the completeness and confidence of the shared
  surface estimator, not merely on whether some scalar surface summary exists.
- Linear fallback should not be interpreted as a volumetric density.
- The method averages over frames and therefore removes time ordering.

## Preview Performance And Diagnostics

Density GUI setup reads HDF5 profile headers only. Sparse coordinate and value
datasets are loaded lazily for enabled species when a line or heatmap is first
rendered, and the resulting slice is reused for style-only preview updates. An
INFO message identifies the active species, projected axes, sparse nonzero-bin
count, geometry-group count, and elapsed preparation time. Cache hits are
DEBUG-only so ordinary plotting remains concise.

Range fields are blank when the complete dataset range is active; their
placeholders show the available limits. Only an explicitly narrowed bound
activates sparse-grid filtering. This keeps the default 1D preview on the
stored line profiles, while filtered and 2D views continue to use the exact
sparse-grid projection.

For a tilted prism whose selected surface-normal Cartesian axis is decoupled
from the in-plane lattice vectors, normalization is still exact: LiNaK clips
the in-plane polygon once per unique geometry and combines it with vectorized
normal-interval overlaps. General rotated/triclinic cells retain exact
polyhedral clipping with extent prefiltering. Neither path uses sampling or a
rectangular-cell approximation.

New symmetric sparse grids store aligned per-geometry top/bottom mass and
entity sums plus both surface offsets. Folded distance density is the arithmetic
mean of the two independently volume-normalized sides. A legacy symmetric grid
without this paired normalization data cannot safely reconstruct a distance
view; LiNaK asks for recomputation instead of silently applying one side's
offset.

## Related Documentation

- [Surface Estimation](surface-estimation.md)
- [Water Detection And Water Geometry](water-detection.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
