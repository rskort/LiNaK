# Surface Estimation

Several LiNaK analyses can express coordinates relative to a surface instead of
raw Cartesian position. This page documents the shared surface-reference
estimator used by density, position, coordination, and orientation.

The implementation lives primarily in
`src/linak/analysis/density.py`.

## Goal

For each frame `t`, LiNaK attempts to construct a scalar surface reference
coordinate `s_t` along a chosen axis. Downstream analyses can then convert raw
axis coordinates `x_t` into surface-relative coordinates

`d_t = x_t - s_t`

This is a scalar alignment reference, not a full geometric reconstruction of a
surface mesh.

If LiNaK cannot construct a complete and sufficiently trustworthy frame-wise
surface-reference array, downstream analyses do not silently force
surface-relative coordinates. They fall back to raw axis coordinates and record
the surface diagnostics as metadata.

## User Controls

The shared estimator exposes advanced Python options through
`SurfaceEstimatorOptions`. The most important user-facing controls are:

- `mode = "auto" | "layered" | "rough"`
- `side = "top" | "bottom"`
- `reduction_mode = "median" | "trimmed_mean" | "legacy_mean"`
- `low_confidence_threshold`
- layer-gap and layer-size thresholds
- rough-reference selection fraction, quantile, and optional
  `rough_surface_envelope_A`
- conservative fill limits and neighbor-consistency tolerances

The default reduction is robust:

- layered direct estimates use the median of the detected layer
- rough low-mobility estimates use the median of the selected reference
  coordinates
- tracked fills use the same robust reducer

Legacy mean-based reductions still exist, but only through explicit advanced
Python options.

LiNaK stores effective surface-estimation settings in nested HDF5 metadata under
`surface.effective_options` for reproducibility.

## Candidate Surface-Defining Atoms

Before estimating the surface reference, LiNaK decides which atoms are allowed
to define it. These are the *surface-defining atoms*. They are distinct from
the *analysis target atoms* whose density, position, coordination, or
orientation is later measured relative to the surface.

### Explicit Selection

LiNaK supports three explicit selection styles:

- `surface_elements`
- `surface_atom_indices`
- `surface_atom_mask`

`surface_atom_indices` and `surface_atom_mask` are advanced Python-only
controls. They require a stable atom layout across frames, because the same atom
identity must exist in every frame.

If fixed atoms are excluded, LiNaK removes constrained atoms from the
surface-defining set before estimation.

This only works when ASE constraint metadata is still present in the loaded
frames. Plain XYZ trajectories often do not preserve those constraints, so
"fixed-layer" knowledge may be lost before analysis.

### Automatic Selection

If no explicit selection is provided, LiNaK chooses element types
automatically.

The automatic path:

1. excludes hydrogen when non-hydrogen atoms are present
2. prefers sufficiently abundant species
3. prefers structurally stable species over merely heavy species

When the atom layout is stable across frames, LiNaK computes a
translation-corrected 3D mobility diagnostic.

For candidate atom `i` and frame `t`, let `r_{t,i}` be the Cartesian position.
LiNaK first subtracts a per-frame translation vector based on the median
candidate position:

`r'_{t,i} = r_{t,i} - median_j(r_{t,j})`

It then computes a per-atom mobility as the median Euclidean deviation from the
atom's median translated position:

`m_i = median_t ||r'_{t,i} - median_t(r'_{t,i})||`

When LiNaK aggregates that information to element level, mass is only used as a
weak tie-breaker. Structural stability dominates.

LiNaK can also reject candidate populations that are too dispersed along the
chosen axis to look like a plausible substrate population.

## Layered Mode

Layered mode is intended for slab-like systems where the surface-defining atoms
form separable layers along the chosen axis.

### Per-Frame Layer Detection

For one frame, LiNaK extracts the selected axis coordinates of the
surface-defining atoms:

`z_1, z_2, ..., z_N`

After sorting them in ascending order,

`z_(1) <= z_(2) <= ... <= z_(N)`

LiNaK computes adjacent gaps:

`Delta_k = z_(k+1) - z_(k)`

for `k = 1, ..., N-1`.

It estimates a baseline gap from the lower half of the gap distribution:

`b = median(smallest half of {Delta_k})`

The significance threshold for a layer break is:

`gap_threshold = max(gap_min_A, gap_factor * b)`

With current defaults:

`gap_threshold = max(0.25 Angstrom, 3.0 * b)`

Any gap above this threshold is treated as a significant layer separation.

### Side-Aware Layer Selection

`side="top"` selects the atoms above the last significant gap.

`side="bottom"` selects the atoms below the first significant gap.

If there is no significant gap, LiNaK still checks the single largest gap. If
that largest gap is at least `2 * gap_min_A`, it treats the frame as a
two-layer split. Otherwise the frame is rejected as not clearly layered.

### Per-Frame Rejection Checks

Even if a split is found, LiNaK rejects the frame when:

- there are too few candidate atoms
- the selected layer is too small
- the selected layer is too broad along the chosen axis
- the selected layer overlaps too much with the bulk candidate distribution
- the selected layer implies an unphysical jump relative to neighboring valid
  frames

The minimum accepted layer size is

`n_min = max(minimum_top_layer_atoms, ceil(minimum_top_layer_fraction * N))`

with current defaults:

`n_min = max(2, ceil(0.03 * N))`

The current broadness threshold is controlled by `layered_max_spread_A`.

### Per-Frame Layered Estimate

For a valid frame, the direct layered surface reference is the robust reduction
of the selected layer coordinates:

`s_t = median({z_i in selected layer})`

when `reduction_mode="median"`.

If `reduction_mode="trimmed_mean"`, LiNaK trims the configured fraction from
both tails before averaging. `legacy_mean` restores the older mean-based
behavior.

### Layered Confidence

For each valid layered frame, LiNaK computes a confidence score in `[0, 1]`.
The score depends on:

- the largest detected gap relative to the required threshold
- the selected-layer size relative to the minimum accepted size
- the spread of the selected-layer coordinates
- temporal consistency with the nearest neighboring valid frames

So a frame with a clean layer break, a compact layer, and a stable time-series
position receives higher confidence than a marginal layer split.

### Tracked Fill

If a short missing run remains, LiNaK can attempt a tracked fill by reusing the
nearest valid layer identities from neighboring valid frames.

If the nearest previous and next valid layer atom sets overlap, the overlap is
preferred. Otherwise LiNaK reuses the previous valid set.

For a candidate fill frame, LiNaK computes:

`s_t = robust_reduce({z_i for tracked layer atoms})`

but only accepts the fill if the resulting value is consistent with neighboring
valid frames within the configured fill tolerance. Otherwise the frame remains
missing and records a rejection reason such as `tracked_fill_inconsistent`.

Tracked fills are assigned lower confidence than direct layered detections.

## Rough Mode

Rough mode is intended for systems where no clean layer break exists or where a
layer-based interpretation is unreliable.

### Stable-Layout Low-Mobility Branch

If the atom layout is stable across frames, rough mode prefers a persistent,
low-mobility reference set.

LiNaK starts from the translation-corrected 3D mobility values `m_i` described
earlier. It also computes the median axis coordinate of each candidate atom over
time.

Before it ranks atoms by mobility, it first enforces a geometric *surface
envelope* on the requested side:

- `side="top"` keeps only atoms near the outermost top-side candidate median
- `side="bottom"` keeps only atoms near the outermost bottom-side candidate
  median

The envelope depth is controlled by `rough_surface_envelope_A`. If this option
is omitted, LiNaK derives an adaptive default from the outer candidate spacing.

If the resulting envelope population becomes too small, LiNaK widens the
envelope conservatively before falling back.

Among the remaining candidates, LiNaK ranks atoms primarily by mobility and only
weakly by mass. In simplified form:

`score_i = mobility_rank_i + mass_tiebreak_weight * heavy_rank_i`

Lower mobility rank is better. Higher mass only nudges the ordering.

LiNaK then selects a reference subset whose size is controlled by:

- `rough_reference_fraction`
- `rough_reference_min_atoms`
- `rough_reference_max_soft_cap`

For each frame, the rough low-mobility surface reference is:

`s_t = median({z_i in selected reference set})`

under the default robust reduction.

### Rough Quantile Fallback

If the stable-layout low-mobility branch is unavailable or not usable, LiNaK
falls back to a purely geometric side-aware quantile.

For `side="top"`:

`s_t = q_q({z_i})`

For `side="bottom"`:

`s_t = q_(1-q)({z_i})`

with current default `q = 0.90`.

LiNaK uses the internal labels:

- `upper_reference_quantile`
- `lower_reference_quantile`

instead of the older `rough_axis_quantile:q90` naming.

### Rough Confidence

For each valid rough frame, LiNaK computes a confidence score in `[0, 1]`
based on:

- the spread of the selected reference coordinates
- temporal consistency with neighboring valid frames
- whether the frame came from the preferred low-mobility branch or only from
  the geometric quantile fallback

Low-mobility reference frames are scored more strongly than pure quantile
fallback frames. Rough estimates are also down-ranked when their chosen
reference set sits too far below the geometric outer envelope, which helps
prevent a buried fixed-like subsurface layer from beating the true exposed
surface purely because it is smoother in time.

## Auto Mode

`auto` is not a separate estimator. LiNaK first builds both:

- a layered estimate
- a rough estimate

It then compares them using a composite score rather than using success ratio
alone.

The composite comparison currently combines:

- valid fraction
- median confidence
- temporal smoothness
- within-frame spread quality

The estimate with the stronger composite score is selected. In practice this
means layered mode is preferred only when it is clearly reliable, not merely
barely valid.

## Conservative Gap Filling

After LiNaK constructs the direct estimate, it applies a conservative fill pass.

The current fill policy is intentionally limited:

- only short missing runs are considered
- fills must stay consistent with neighboring valid frames within a configured
  absolute tolerance
- long invalid stretches are not silently invented

If LiNaK cannot justify a fill, the frame stays missing.

This is why a surface estimate may still carry a valid scalar summary
`surface_position`, while downstream analyses still decline to use
surface-relative coordinates.

## Safe Failure Mode

Downstream analyses only switch into

`coordinate_mode = "distance"`

when the selected surface estimate is complete and trusted.

In practical terms, LiNaK requires:

- a frame-wise surface array with the correct length
- no missing frames after conservative filling
- valid fraction `= 1`
- median confidence at least `low_confidence_threshold`

If those conditions are not met, downstream analyses keep

`coordinate_mode = "axis"`

and treat the surface estimate as advisory metadata rather than a reliable
alignment reference.

So `surface_position` and `surface_position_std` may still be present even when
distance-mode was not used.

## Per-Frame Metadata And Diagnostics

The in-memory `SurfaceEstimate` object contains:

- `frame_values`
- `valid_mask`
- `confidence`
- `provenance`
- `candidate_indices` when stable/common across frames
- `selected_elements`
- `mode`
- `side`
- `summary`
- `diagnostics`

The diagnostics currently include:

- `candidate_count_per_frame`
- `top_layer_size_per_frame`
- `largest_gap_A_per_frame`
- `baseline_gap_A_per_frame`
- `reference_spread_A_per_frame`
- `jump_rejection_mask`
- `rejection_reason`
- `effective_options`

### Provenance Labels

Current provenance labels include:

- `direct_layered`
- `direct_rough_low_mobility`
- `direct_rough_quantile`
- `tracked_fill`
- `quantile_fill`
- `missing`

### Rejection Reasons

Current rejection reasons are implementation-facing diagnostics. Common values
include:

- `insufficient_candidates`
- `no_layer_break`
- `top_layer_too_small`
- `top_layer_too_broad`
- `top_layer_overlaps_bulk`
- `bottom_layer_overlaps_bulk`
- `jump_reject`
- `tracked_fill_inconsistent`
- `quantile_fill_inconsistent`

Additional reasons may appear as the implementation evolves.

## What Gets Stored

Surface-aware analyses can persist the following surface datasets in HDF5 when a
surface estimate exists:

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

They can also persist nested summary metadata under `surface`, including keys
such as:

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

Older files may still contain flat summary keys such as `surface_position` or
`surface_mode`. LiNaK loaders accept both the older flat form and the newer
nested surface metadata.

## Important Limitations

- The estimator produces a scalar reference coordinate along one axis, not a
  full surface geometry.
- The result depends strongly on the selected surface-defining atoms.
- Layered mode assumes a physically meaningful gap-separated layer structure.
- Rough mode assumes that a compact, persistent reference population or a
  side-aware quantile is a useful surface proxy.
- Surface-relative interpretation remains conditional on completeness and
  confidence.

## Related Documentation

- [Density](density.md)
- [Position](position.md)
- [Coordination](coordination.md)
- [Orientation](orientation.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)
