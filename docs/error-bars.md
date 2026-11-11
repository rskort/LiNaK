# Error Bars in LiNaK

## 1. Choose the Meaning

Overlays are **off by default**. CLI, Studio, and data exports use the same
statistics backend. A requested estimator is never silently replaced.

| Choice | What it shows | Statistic |
|---|---|---|
| **Variability** | Spread of the defined observations. | Sample standard deviation (`sample_std`). |
| **Uncertainty of the estimate** | Precision of the plotted mean or median. | Block-bootstrap confidence interval (`block_ci`), 95% by default. |
| **Advanced estimators** | Explicit alternatives with different assumptions. | Independent-observation SEM, block SEM, or block SD. |

SD uses `sqrt(sum((y - mean(y))^2) / (n - 1))`; sample SEM is `SD / sqrt(n)`.
Advanced SEM describes the recorded sample/block means; use the confidence
interval for weighted or nonlinear plotted estimators. Sample SEM assumes
independent observations. SD/SEM need at least two finite values;
missing estimates are not replaced with zero. Confidence intervals can be
asymmetric. Bars and shaded bands display the same interval values.

## 2. Observations by Method

| Method | Statistical unit | Confidence-interval calculation |
|---|---|---|
| **Density / distributions** | Normalized density per frame and spatial bin. | Resample frame blocks, retaining mass/number normalization and interface averaging. |
| **RDF** | Observed and expected pair counts per frame. | Every resample recomputes `sum(observed) / sum(expected)`, matching the displayed RDF. |
| **Orientation** | Cosine sums, populations, and density contributions per frame. | Preserve population weighting, cosine-density products, and equal-interface weighting for folded curves. |
| **Coordination** | One ensemble mean per frame and distance bin; one value per frame for a tracked center. | Resample temporal blocks. Center trajectories require a time window to estimate uncertainty. |
| **Position** | One entity's coordinate or distance per frame. | Estimate window means or medians. An instantaneous trajectory point has no sampling-error interval. |
| **Potential** | Ordered records or independent replicates, explicitly selected by the user. | Temporal blocks for ordered records; replicate resampling for independent records within each section. Record IDs do not establish either assumption. |
| **Temperature** | Temperature per frame. | Estimate time-window means or medians using temporal blocks; show thermal fluctuations separately as SD. |

Atoms and pairs within a frame are not independent trajectory samples.
Duplicating identical entities therefore does not shrink a frame-mean confidence
interval. Valid-observation counts and raw entity/pair contributions are reported
separately. Raw median sections use resampled medians; mean SEM is refused for a
median curve.

## 3. Blocking and Reproducibility

- Temporal estimates require at least **100 observations**, divided into
  `min(20, floor(observations / 25))` contiguous blocks: at least four blocks,
  each containing at least 25 observations before missing-value exclusion.
- Bootstrap defaults: **1,000 resamples**, **95% confidence**, **seed 0**.
  The interval is the 2.5th to 97.5th percentile of resampled estimates.
- Spatial bins use the same temporal resamples, preserving within-frame
  dependence. Separate time windows retain their own block lengths. Each bin
  needs four contributing blocks and at least 90% finite resampled estimates.
- Independent replicate groups require at least **four replicates**; temporal
  block merging does not apply to them.
- Consecutive stored blocks can be merged to test longer block lengths.
  Studio reports block-size stability as median interval widths for available
  merge factors 1, 2, and 4. This does **not** establish independence or convergence.
- Counts, block lengths, confidence, resample count, seed, and estimator are
  included in render metadata. Physical durations are reported when a regular
  profile time grid is available, using that profile's configured timestep.

## 4. Controls and Exports

```bash
# Recompute older RDF files first to retain the required block counts/weights.
linak plot trajectory.rdf.h5 --uncertainty estimate
linak plot trajectory.rdf.h5 --uncertainty variability
linak plot trajectory.rdf.h5 --uncertainty estimate --uncertainty-style whiskers
linak plot trajectory.rdf.h5 --uncertainty estimate --uncertainty-confidence 0.95 --uncertainty-resamples 1000 --uncertainty-seed 0 --uncertainty-block-merge 2
```

`--uncertainty mean` remains an alias for `--uncertainty estimate`.

Potential records additionally require `--uncertainty-sampling temporal` or
`--uncertainty-sampling independent` for confidence intervals. Select a section
containing enough records. Position, coordination, and temperature can use
`--time-section-width` to define averaging windows, in the displayed time units.

Studio provides the same choices and settings. Its summary identifies the
observation unit, valid counts, raw contributions, block lengths, and availability
reason. Insufficient data produces an explanation and an action, rather than a
different estimator.

Data exports include interval bounds, method, confidence, sample counts, and
valid-block counts when an overlay is available. Metadata comments additionally
export the full uncertainty provenance and block-size diagnostic.

## 5. Transformations, Old Files, and Limits

- New statistics retain block sums and weights in float64. Existing analysis
  schemas remain readable; older files without block data require recomputation
  for confidence intervals. Their saved SD/SEM remains an explicit advanced option.
- Mean rebinning retains shared resamples and covariance. Normalization is
  reapplied to each resample, including an estimated area, maximum, or reference
  value. Filtering/masking retains supported block data.
- Exact merged-bin valid-frame counts and frame SD/SEM may be unavailable when
  only marginal statistics were retained. They are not inferred by treating
  spatial bins as independent frames. Block confidence intervals remain usable.
- Aggregated sparse-grid slices without retained frame information cannot recover
  uncertainty. Heatmaps/projections do not support these 1D overlays. These cases
  remain explicitly unavailable instead of inventing an interval.
- Folded density confidence intervals retain both interfaces together. Frame SD
  is unavailable when only marginal side moments were retained, because those
  moments cannot recover side covariance. Side-specific statistics remain usable.
- Empty observations, uncovered geometry, and missing molecular centers are
  excluded where the estimator is undefined. Genuine constant observations can
  produce a collapsed empirical interval; this differs from insufficient data.
- Sampling uncertainty excludes force-field/model errors and classification
  errors. Surface detection confidence, reference-region spread, and fit-parameter
  uncertainty remain separate diagnostics.

Implementation: [shared resampling](../src/linak/analysis/uncertainty.py),
[saved statistics](../src/linak/analysis/statistics.py), and
[plotting](../src/linak/plot/plotting.py).
