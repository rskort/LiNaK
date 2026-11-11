"""Estimator-aware temporal resampling shared by compute, plotting and exports."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np


@dataclass(frozen=True)
class BlockData:
    """Sums and weights, indexed by temporal block, plotted bin and component."""

    numerator: np.ndarray
    denominator: np.ndarray
    lengths: np.ndarray
    estimator: str = "ratio"
    unit: str = "frames"
    groups: np.ndarray | None = None
    median_samples: tuple | None = None
    timestep_fs: float | None = None
    lengths_per_bin: np.ndarray | None = None
    independent_bins: bool = False

    def __post_init__(self):
        n, d, lengths = self.numerator, self.denominator, self.lengths
        if n.ndim != 3 or n.shape != d.shape or lengths.shape != (n.shape[0],):
            raise ValueError("Block sufficient statistics have incompatible shapes.")
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("Block sufficient statistics must be finite with nonnegative weights.")
        if np.any(lengths < 0) or self.estimator not in {
            "ratio",
            "side_mean",
            "product",
            "side_product",
            "median",
        }:
            raise ValueError("Invalid block lengths or uncertainty estimator.")
        if self.estimator == "median" and self.median_samples is None:
            raise ValueError("Median intervals require retained within-block observations.")
        if self.independent_bins and self.lengths_per_bin is None:
            raise ValueError("Independent time windows require per-bin block lengths.")
        if self.lengths_per_bin is not None and self.lengths_per_bin.shape != n.shape[:2]:
            raise ValueError("Per-bin block lengths do not match the stored blocks.")
        if self.median_samples is not None:
            if len(self.median_samples) != n.shape[0] or any(
                len(row) != n.shape[1] for row in self.median_samples
            ):
                raise ValueError("Retained median observations do not match the stored blocks.")
            if any(
                np.asarray(values).ndim != 1 or not np.all(np.isfinite(values))
                for row in self.median_samples
                for values in row
            ):
                raise ValueError("Retained median observations must be finite vectors.")


@dataclass(frozen=True)
class Interval:
    lower: np.ndarray
    upper: np.ndarray
    details: dict


def estimate(numerator, denominator, estimator="ratio"):
    ratios = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    if estimator == "median":
        return np.full(ratios.shape[:-1], np.nan)
    if estimator == "ratio":
        return ratios[..., 0]
    if estimator == "product":
        return ratios[..., 0] * ratios[..., 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(ratios[..., :2], axis=-1)
    return mean if estimator == "side_mean" else mean * ratios[..., 2]


def frame_blocks(numerator, denominator=None, *, estimator="ratio", unit="frames"):
    """Keep up to twenty contiguous blocks, with at least 25 frames each."""
    n = np.asarray(numerator, dtype=float)
    if n.ndim == 2:
        n = n[..., None]
    d = np.ones_like(n) if denominator is None else np.asarray(denominator, dtype=float)
    if d.ndim == 2:
        d = d[..., None]
    if n.shape != d.shape or n.ndim != 3:
        raise ValueError("Frame statistics need matching frame/bin/component arrays.")
    if len(n) < 100:
        return None
    groups = np.array_split(np.arange(len(n)), min(20, len(n) // 25))
    valid = np.isfinite(n) & np.isfinite(d) & (d > 0)
    return BlockData(
        np.stack([np.where(valid[g], n[g], 0).sum(axis=0) for g in groups]),
        np.stack([np.where(valid[g], d[g], 0).sum(axis=0) for g in groups]),
        np.asarray([len(g) for g in groups]),
        estimator,
        unit,
    )


def merge_blocks(data: BlockData, factor: int):
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("Block merge must be a positive integer.")
    groups = [
        np.arange(i, min(i + factor, len(data.lengths)))
        for i in range(0, len(data.lengths), factor)
    ]
    median = (
        None
        if data.median_samples is None
        else tuple(
            tuple(
                np.concatenate([data.median_samples[b][k] for b in g])
                for k in range(data.numerator.shape[1])
            )
            for g in groups
        )
    )
    return replace(
        data,
        numerator=np.stack([data.numerator[g].sum(axis=0) for g in groups]),
        denominator=np.stack([data.denominator[g].sum(axis=0) for g in groups]),
        lengths=np.asarray([data.lengths[g].sum() for g in groups]),
        median_samples=median,
        lengths_per_bin=None
        if data.lengths_per_bin is None
        else np.stack([data.lengths_per_bin[g].sum(0) for g in groups]),
    )


def bootstrap_interval(
    data: BlockData, *, confidence=0.95, repeats=1000, seed=0, block_merge=1, normalization=None
):
    """Resample whole temporal blocks together across bins; never resample atoms."""
    if not 0 < confidence < 1 or repeats < 100:
        raise ValueError("Confidence must be between zero and one; use at least 100 resamples.")
    if data.unit == "independent replicates" and block_merge != 1:
        raise ValueError(
            "Block merging requires temporal observations; use block merge 1 for independent replicates."
        )
    data = merge_blocks(data, block_merge)
    blocks = len(data.lengths)
    if blocks < 4:
        raise ValueError(
            "At least four temporal blocks are required. Use a longer trajectory or smaller block merge."
        )
    rng = np.random.default_rng(seed)
    # One weight matrix shared by every bin retains spatial covariance.
    weights = (
        None
        if data.independent_bins
        else rng.multinomial(blocks, np.full(blocks, 1 / blocks), size=repeats)
    )

    def bin_weights(column):
        if weights is not None:
            return weights
        selected = np.flatnonzero(data.lengths_per_bin[:, column] > 0)
        result = np.zeros((repeats, blocks), dtype=int)
        if selected.size:
            result[:, selected] = rng.multinomial(
                len(selected), np.full(len(selected), 1 / len(selected)), size=repeats
            )
        return result

    bins = data.numerator.shape[1]
    lower, upper = np.full(bins, np.nan), np.full(bins, np.nan)
    component_counts = np.sum(data.denominator > 0, axis=0)
    active = component_counts > 0
    valid_blocks = np.min(np.where(active, component_counts, blocks), axis=-1)
    valid_blocks[~np.any(active, axis=-1)] = 0
    # Bound working memory while using the same resamples for all slices.
    draws = np.empty((repeats, bins), dtype=float)
    for start in range(0, bins, 256) if weights is not None else ():
        stop = min(start + 256, bins)
        n = np.einsum("rb,bkc->rkc", weights, data.numerator[:, start:stop])
        d = np.einsum("rb,bkc->rkc", weights, data.denominator[:, start:stop])
        draws[:, start:stop] = estimate(n, d, data.estimator)
    if weights is None and data.estimator != "median":
        for column in range(bins):
            selected_weights = bin_weights(column)
            draws[:, column] = estimate(
                selected_weights @ data.numerator[:, column],
                selected_weights @ data.denominator[:, column],
                data.estimator,
            )
    if data.estimator == "median":
        for column in range(bins):
            selected_weights = bin_weights(column)
            values = np.concatenate([data.median_samples[b][column] for b in range(blocks)])
            block_ids = np.concatenate(
                [np.full(len(data.median_samples[b][column]), b) for b in range(blocks)]
            )
            order = np.argsort(values)
            values, block_ids = values[order], block_ids[order]
            draws[:, column] = np.nan
            if not values.size:
                continue
            for first in range(0, repeats, 64):
                stop = min(first + 64, repeats)
                cumulative = np.cumsum(selected_weights[first:stop, block_ids], axis=1)
                totals = cumulative[:, -1]
                lo = np.argmax(cumulative >= ((totals - 1) // 2 + 1)[:, None], axis=1)
                hi = np.argmax(cumulative >= (totals // 2 + 1)[:, None], axis=1)
                draws[first:stop, column] = np.where(
                    totals > 0, (values[lo] + values[hi]) / 2, np.nan
                )
    if data.groups is not None:
        output_groups = np.unique(data.groups[data.groups >= 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            draws = np.stack(
                [np.nanmean(draws[:, data.groups == g], axis=1) for g in output_groups], axis=1
            )
        valid_blocks = np.asarray([valid_blocks[data.groups == g].min() for g in output_groups])
        lower, upper = np.full(len(output_groups), np.nan), np.full(len(output_groups), np.nan)
    if normalization is not None:
        draws = normalization(draws)
    alpha = (1 - confidence) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower[:], upper[:] = np.nanquantile(draws, [alpha, 1 - alpha], axis=0)
    enough = (valid_blocks >= 4) & (np.isfinite(draws).sum(axis=0) >= repeats * 0.9)
    lower[~enough], upper[~enough] = np.nan, np.nan
    return Interval(
        lower,
        upper,
        {
            "method": "independent replicate bootstrap"
            if data.unit == "independent replicates"
            else "temporal block bootstrap",
            "confidence": confidence,
            "seed": int(seed),
            "resamples": int(repeats),
            "sampling_unit": data.unit,
            "block_count": blocks,
            "block_lengths_frames": data.lengths.tolist(),
            "block_length_unit": "records"
            if data.unit == "ordered records"
            else "replicates"
            if data.unit == "independent replicates"
            else "frames",
            "block_lengths_frames_per_bin": None
            if data.lengths_per_bin is None
            else data.lengths_per_bin.T.tolist(),
            "block_lengths_fs": None
            if data.timestep_fs is None
            else (data.lengths * data.timestep_fs).tolist(),
            "valid_blocks_per_bin": valid_blocks.tolist(),
            "estimator": data.estimator,
            "independence_verified": False,
        },
    )


def block_stability(data: BlockData, **options):
    """Report median interval width at increasing block sizes, without a convergence claim."""
    if data.unit == "independent replicates":
        return []
    result = []
    for factor in (1, 2, 4):
        if (len(data.lengths) + factor - 1) // factor < 4:
            continue
        interval = bootstrap_interval(data, block_merge=factor, **options)
        finite = interval.upper - interval.lower
        finite = finite[np.isfinite(finite)]
        result.append(
            {
                "block_merge": factor,
                "block_lengths_frames": merge_blocks(data, factor).lengths.tolist(),
                "median_interval_width": float(np.median(finite)) if finite.size else None,
            }
        )
    return result


def grouped_frame_statistics(
    x, y, *, bin_width=None, frame_indices=None, temporal=True, independent=False, reducer="mean"
):
    """Group entities by frame first, then estimate each displayed bin's frame mean."""
    from .statistics import SeriesStatistics, _sample_std_and_sem_from_matrix

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ids = np.arange(y.size) if frame_indices is None else np.asarray(frame_indices)
    if ids.shape != y.shape:
        raise ValueError("Frame indices must match raw uncertainty observations.")
    all_frame_ids = np.unique(ids)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, ids = x[valid], y[valid], ids[valid]
    if not y.size:
        return None
    groups = np.floor((x - x.min()) / bin_width).astype(int) if bin_width is not None else x
    groups, inverse = np.unique(groups, return_inverse=True)
    count = len(groups)
    xp, yp = np.empty(count), np.empty(count)
    ns, sd, sem, points = (
        np.zeros(count, int),
        np.full(count, np.nan),
        np.full(count, np.nan),
        np.zeros(count, int),
    )
    # Up to twenty blocks per output bin; missing blocks carry zero weight.
    capacity = max(20, len(np.unique(ids))) if independent else 20
    nums, dens, lengths = (
        np.zeros((capacity, count, 1)),
        np.zeros((capacity, count, 1)),
        np.ones(capacity, int),
    )
    have_blocks = False
    lengths_per_bin = np.zeros((capacity, count), dtype=int)
    median_samples = (
        [[np.empty(0) for _ in range(count)] for _ in range(capacity)]
        if reducer == "median"
        else None
    )
    for column in range(count):
        selected = inverse == column
        points[column] = np.count_nonzero(selected)
        xp[column] = np.mean(x[selected])
        frame_ids, frame_inverse = np.unique(ids[selected], return_inverse=True)
        values = np.bincount(frame_inverse, weights=y[selected]) / np.bincount(frame_inverse)
        yp[column] = np.median(values) if reducer == "median" else np.mean(values)
        n, s, e = _sample_std_and_sem_from_matrix(values[:, None])
        ns[column], sd[column], sem[column] = n[0], s[0], e[0]
        if independent and len(values) >= 4:
            chunks = np.array_split(np.arange(len(values)), len(values))
            # Independent replicate bootstrap retains each replicate, never pools them.
            for b, chunk in enumerate(chunks):
                nums[b, column, 0], dens[b, column, 0] = values[chunk].sum(), len(chunk)
                lengths_per_bin[b, column] = len(chunk)
                if median_samples is not None:
                    median_samples[b][column] = values[chunk]
            have_blocks = True
        elif temporal:
            if frame_indices is None:
                samples = np.full((int(frame_ids[-1] - frame_ids[0]) + 1, 1), np.nan)
                samples[(frame_ids - frame_ids[0]).astype(int), 0] = values
            else:
                all_ids = all_frame_ids
                samples = np.full((len(all_ids), 1), np.nan)
                samples[np.searchsorted(all_ids, frame_ids), 0] = values
            blocks = frame_blocks(samples)
            if blocks is not None:
                size = len(blocks.lengths)
                nums[:size, column] = blocks.numerator[:, 0]
                dens[:size, column] = blocks.denominator[:, 0]
                lengths[:size] = np.maximum(lengths[:size], blocks.lengths)
                lengths_per_bin[:size, column] = blocks.lengths
                if median_samples is not None:
                    start = 0
                    for b, length in enumerate(blocks.lengths):
                        row = samples[start : start + length, 0]
                        median_samples[b][column] = row[np.isfinite(row)]
                        start += length
                have_blocks = True
    if have_blocks:
        used = np.any(lengths_per_bin > 0, axis=1)
        blocks = BlockData(
            nums[used],
            dens[used],
            lengths[used],
            estimator="median" if reducer == "median" else "ratio",
            unit="independent replicates" if independent else "frame means",
            lengths_per_bin=lengths_per_bin[used],
            independent_bins=frame_indices is None,
            median_samples=None
            if median_samples is None
            else tuple(tuple(median_samples[b]) for b in np.flatnonzero(used)),
        )
    else:
        blocks = None
    stats = SeriesStatistics(
        points,
        ns,
        sd,
        sem,
        resampling=blocks,
        sampling_unit="independent replicates"
        if independent
        else "frame means"
        if frame_indices is not None
        else "frames"
        if temporal
        else "records",
    )
    return xp, yp, stats


def subset_blocks(data, mask):
    if data is None:
        return None
    if data.groups is None:
        median = (
            None
            if data.median_samples is None
            else tuple(tuple(row[k] for k in np.flatnonzero(mask)) for row in data.median_samples)
        )
        return replace(
            data,
            numerator=data.numerator[:, mask],
            denominator=data.denominator[:, mask],
            median_samples=median,
            lengths_per_bin=None if data.lengths_per_bin is None else data.lengths_per_bin[:, mask],
        )
    group_ids = np.unique(data.groups[data.groups >= 0])
    kept = group_ids[mask]
    groups = np.full(data.groups.shape, -1, dtype=int)
    for index, group in enumerate(kept):
        groups[data.groups == group] = index
    return replace(data, groups=groups)


def scale_blocks(data, scale):
    if data is None:
        return None
    numerator = data.numerator.copy()
    numerator[..., : 2 if data.estimator in {"side_mean", "side_product"} else 1] *= scale
    return replace(data, numerator=numerator)


def combine_side_statistics(top, bottom):
    """Keep side provenance and covariance by resampling their aligned blocks together."""
    result = {
        **{f"top_{key}": value for key, value in (top or {}).items()},
        **{f"bottom_{key}": value for key, value in (bottom or {}).items()},
    }
    for key in set(top or {}) & set(bottom or {}):
        a, b = top[key], bottom[key]
        data = None
        if a.resampling is not None and b.resampling is not None:
            if not np.array_equal(a.resampling.lengths, b.resampling.lengths):
                raise ValueError("Interface uncertainty blocks are not aligned.")
            data = BlockData(
                np.concatenate([a.resampling.numerator, b.resampling.numerator], axis=2),
                np.concatenate([a.resampling.denominator, b.resampling.denominator], axis=2),
                a.resampling.lengths,
                "side_mean",
            )
        # Marginal sample moments cannot recover side covariance. Do not invent SD.
        result[key] = replace(
            a,
            point_count=a.point_count + b.point_count,
            sample_n=np.minimum(a.sample_n, b.sample_n),
            sample_std=np.full_like(a.sample_std, np.nan),
            sample_sem=np.full_like(a.sample_sem, np.nan),
            block_n=None,
            block_std=None,
            block_sem=None,
            resampling=data,
        )
    return result or None


def time_step_fs(time_values):
    """Return a physical saved-frame interval only for a finite, uniform time grid."""
    time = np.asarray(time_values, dtype=float)
    steps = np.diff(time)
    if not steps.size or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        return None
    return float(steps[0]) if np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9) else None
