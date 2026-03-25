# Surface Estimation

Several LiNaK analyses can express coordinates relative to a surface instead of
raw Cartesian position. This page documents the shared estimator used by
density, position, coordination, and orientation.

The current implementation lives primarily in `src/linak/analysis/density.py`.

## Goal

For each frame `t`, LiNaK attempts to estimate a surface position
`s_t` along a chosen axis. Downstream analyses can then shift raw coordinates
`x_t` into surface-relative coordinates:

`d_t = x_t - s_t`

If LiNaK cannot build a reliable frame-wise surface array `{s_t}`, it falls
back to raw axis coordinates rather than inventing a misleading distance.

## Step 1: Choose Surface Reference Elements

The estimator first decides which atoms are allowed to define the surface.

### User Override

If `--surface-elements` is provided, LiNaK uses exactly those element symbols
after normalization and validation against the trajectory.

### Automatic Selection

If there is no override, LiNaK applies a heuristic.

1. Hydrogen is excluded when non-hydrogen atoms are available.
2. LiNaK counts the abundance of each remaining symbol.
3. It prefers elements that are both abundant and low-mobility.

When the atom layout is stable across frames, LiNaK computes a mobility score
for each candidate atom:

`m_i = median_t |x_{t,i}^{unwrap} - median_t(x_{t,i}^{unwrap})|`

where `x_{t,i}^{unwrap}` is the axis coordinate after periodic unwrapping when
possible.

For each element symbol `E`, LiNaK then reduces the atomic mobilities to one
symbol-level mobility:

`m_E = median_{i in E}(m_i)`

It keeps element symbols with:

- sufficiently low mobility relative to the minimum mobility
- sufficient abundance in the frame

If that mobility-based selection is unavailable or degenerate, LiNaK falls back
to a simpler abundance-and-mass heuristic: keep abundant heavy species and cap
the final selection to a few element types.

## Step 2: Evaluate Surface Modes

LiNaK currently implements two concrete surface estimators:

- `layered`
- `rough`

`auto` evaluates both and chooses between them.

## Layered Mode

Layered mode is intended for slab-like surfaces with clear layer separation.

### 2.1 Per-Frame Candidate Coordinates

For one frame, LiNaK collects the axis coordinates of all atoms belonging to the
selected surface elements. If fixed atoms are disallowed, constrained atoms are
removed first.

Call the resulting sorted coordinates:

`z_(1) <= z_(2) <= ... <= z_(N)`

### 2.2 Gap Analysis

LiNaK computes adjacent gaps:

`Delta_k = z_(k+1) - z_(k)`

for `k = 1, ..., N-1`.

It then estimates a baseline gap from the lower half of the gap distribution:

- take the smallest half of the `Delta_k`
- compute their median

Call that baseline `b`.

The significance threshold for a layer break is:

`gap_threshold = max(0.25 Angstrom, 3.0 * b)`

Any gap satisfying:

`Delta_k >= gap_threshold`

is treated as a significant separation between layers.

### 2.3 Define The Top Layer

If at least one significant gap exists, LiNaK takes the atoms above the last
such gap as the top layer.

If no significant gap exists, LiNaK checks the single largest gap. If that
largest gap is at least `2 * 0.25 = 0.5 Angstrom`, it still treats the atoms
above that gap as a two-layer split. Otherwise the frame is considered
insufficiently layered.

### 2.4 Top-Layer Size Filter

Even if a top cluster is found, LiNaK rejects it if it is too small.

The minimum accepted top-layer size is:

`n_min = max(2, ceil(0.03 * N))`

where `N` is the number of candidate surface atoms in that frame.

### 2.5 Per-Frame Surface Value

If the top layer passes the size test, the frame-wise surface value is the mean
axis coordinate of that layer:

`s_t = mean(z_i for i in top layer)`

### 2.6 Global Acceptance Criterion

Layered mode must succeed often enough across the trajectory.

Let:

`success_ratio = (# frames with valid layered estimate) / (# total frames)`

LiNaK requires:

`success_ratio >= 0.60`

Otherwise layered mode is rejected as a whole.

### 2.7 Summary Metadata

For accepted layered mode, LiNaK stores summary metadata from the valid
frame-wise values:

- representative position: `median_t(s_t)`
- spread: `std_t(s_t)`

So the stored scalar `surface_position` is the median of the valid per-frame
surface positions, not their mean.

### 2.8 Tracked-Layer Gap Filling

If some frames fail after layered mode is accepted overall, LiNaK attempts to
fill those missing frames using the nearest valid top-layer atom identities from
neighboring frames. If the previous and next valid top-layer atom sets overlap,
their intersection is preferred; otherwise the previous valid set is reused.

For a filled frame, the value becomes:

`s_t = mean(z_i for i in tracked top-layer indices)`

This preserves continuity when the layered identification is momentarily
ambiguous but the slab atoms remain consistent.

## Rough Mode

Rough mode is intended for surfaces without a clean layer gap.

### 3.1 Stable-Layout Branch

If the atom layout is stable across frames, LiNaK computes a per-atom mobility
using the same median absolute deviation style measure described earlier:

`m_i = median_t |x_{t,i}^{unwrap} - median_t(x_{t,i}^{unwrap})|`

It then combines mobility and atomic mass into a ranking score. The code uses:

`score_i = mobility_rank_i + 0.35 * heavy_rank_i`

where lower mobility rank is better and higher mass is favored through the
heavy-rank contribution.

LiNaK chooses a reference set of the lowest-scoring atoms. The number of
reference atoms is:

`n_ref = max(min_ref, ceil(0.35 * N))`

with:

`min_ref = min(6, max(3, floor(N / 2)))`

and `N` the number of candidate atoms after filtering.

The frame-wise surface value is then:

`s_t = mean(x_{t,i} for i in reference set)`

This produces a low-mobility mean surface reference.

### 3.2 Quantile Fallback Branch

If the stable-layout low-mobility branch is unavailable, LiNaK falls back to a
purely geometric per-frame quantile:

`s_t = q_0.90({x_{t,i}})`

using the 90th percentile of the candidate surface-atom coordinates in each
frame.

This is the origin of the rough-mode fallback label
`rough_axis_quantile:q90`.

### 3.3 Summary Metadata

As with layered mode, the stored scalar summary is based on the valid per-frame
surface values:

- representative position: `median_t(s_t)`
- spread: `std_t(s_t)`

## Auto Mode

`auto` evaluates both layered and rough estimates and chooses between them.

The current preference logic is:

- choose layered if layered succeeded and either:
  - `layered.success_ratio >= 0.75`, or
  - rough failed entirely
- otherwise choose rough if rough succeeded
- otherwise fall back to layered if only layered succeeded
- otherwise declare surface estimation unavailable

So `auto` is not a black box. It prefers layered behavior only when the layered
signal is sufficiently strong.

## Missing-Frame Filling

After selecting the final estimator, LiNaK checks whether the frame-wise surface
array still has missing values. If so, it computes a frame-local fallback from
the 90th percentile of the selected surface-atom coordinates and fills only the
missing entries where that fallback is finite.

This is a postprocessing step on the chosen estimator, not a third primary
surface mode.

## How Downstream Analyses Use The Result

If the final per-frame array `{s_t}` is fully finite and aligned with the frame
count, downstream analyses switch into surface-distance mode:

`d_t = x_t - s_t`

Otherwise they keep raw axis coordinates and record
`coordinate_mode = "axis"`.

## Important Limitations

- Layered mode assumes a meaningful top-layer gap in the sorted axis
  coordinates.
- Rough mode assumes that low-mobility or high-quantile atoms represent the
  surface.
- Both modes depend on the chosen reference elements.
- The scalar `surface_position` is summary metadata only; the actual alignment
  uses the full frame-wise surface array whenever possible.

## Related Documentation

- [Density](density.md)
- [Position](position.md)
- [Coordination](coordination.md)
- [Orientation](orientation.md)
