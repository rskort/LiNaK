"""Sparse, conservative lateral transfer operators shared by interface views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from time import perf_counter

import numpy as np

from ..cache import PREVIEW_CACHE
from ..cancellation import check_cancelled
from ..runtime_resources import resolve_resource_policy

LOGGER = logging.getLogger("linak")


def expansion_budget() -> int:
    return max(1, min(32 * 1024 * 1024, resolve_resource_policy().memory_budget_bytes // 8))


def _array_key(array) -> tuple:
    value = np.ascontiguousarray(array, dtype=np.float64)
    return value.shape, hashlib.sha256(value.tobytes()).digest()


def interval_compatible(basis, matrix) -> bool:
    matrix = np.asarray(matrix)
    return bool(
        np.array_equal(basis, np.diag(np.diag(basis)))
        and np.all(np.diag(basis) != 0)
        and np.all(np.sum(abs(matrix), axis=0) == 1)
        and np.all(np.sum(abs(matrix), axis=1) == 1)
    )


@dataclass(frozen=True)
class IntervalTransfer:
    offsets: np.ndarray
    targets: np.ndarray
    weights: np.ndarray


def _interval_transfer(
    source, target, source_origin, source_period, target_origin, target_period, sign, translation
):
    """Intersect periodic intervals in batches, including partial boundary bins."""
    lower = np.maximum(source[:-1], min(source_origin, source_origin + source_period))
    upper = np.minimum(source[1:], max(source_origin, source_origin + source_period))
    source_ids = np.flatnonzero(upper > lower)
    fraction0 = (lower[source_ids] - source_origin) / source_period * sign + translation
    fraction1 = (upper[source_ids] - source_origin) / source_period * sign + translation
    lo, hi = np.minimum(fraction0, fraction1), np.maximum(fraction0, fraction1)
    widths = (hi - lo) * abs(target_period)
    starts = np.floor(lo)
    all_sources, all_targets, all_weights = [], [], []
    for image_offset in (0, 1):
        check_cancelled()
        a = np.maximum(lo - starts - image_offset, 0.0)
        b = np.minimum(hi - starts - image_offset, 1.0)
        keep = b > a
        part_ids = np.flatnonzero(keep)
        a, b = a[keep] * target_period + target_origin, b[keep] * target_period + target_origin
        a, b = np.minimum(a, b), np.maximum(a, b)
        first = np.clip(np.searchsorted(target, a, side="right") - 1, 0, len(target) - 1)
        stop = np.clip(np.searchsorted(target, b, side="left"), 0, len(target) - 1)
        counts = np.maximum(stop - first, 0)
        parts = np.repeat(np.arange(len(part_ids)), counts)
        destinations = (
            np.repeat(first, counts)
            + np.arange(int(counts.sum()))
            - np.repeat(np.cumsum(counts) - counts, counts)
        )
        weights = (
            np.maximum(
                0.0,
                np.minimum(b[parts], target[destinations + 1])
                - np.maximum(a[parts], target[destinations]),
            )
            / widths[part_ids[parts]]
        )
        keep = weights > 0
        all_sources.append(source_ids[part_ids[parts[keep]]])
        all_targets.append(destinations[keep])
        all_weights.append(weights[keep])
    n_target = len(target) - 1
    keys = np.concatenate(all_sources) * n_target + np.concatenate(all_targets)
    weights = np.concatenate(all_weights)
    order = np.argsort(keys, kind="stable")
    keys, weights = keys[order], weights[order]
    unique, first = np.unique(keys, return_index=True)
    weights = np.add.reduceat(weights, first) if len(first) else weights
    owners, targets = unique // n_target, unique % n_target
    totals = np.bincount(owners, weights=weights, minlength=len(source) - 1)
    if not np.allclose(totals[source_ids], 1.0, rtol=1e-8, atol=1e-9):
        raise ValueError("Interface remapping did not preserve the full periodic footprint.")
    weights /= totals[owners]
    offsets = np.r_[0, np.cumsum(np.bincount(owners, minlength=len(source) - 1))]
    return IntervalTransfer(offsets, targets, weights)


def _expanded_indices(counts, max_entries):
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts
    total = int(ends[-1]) if len(ends) else 0
    for start in range(0, total, max_entries):
        check_cancelled()
        indices = np.arange(start, min(total, start + max_entries), dtype=np.int64)
        owners = np.searchsorted(ends, indices, side="right")
        yield owners, indices - starts[owners]


@dataclass(frozen=True)
class LateralTransfer:
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    source_axes: tuple[int, int] = (0, 1)
    intervals: tuple[IntervalTransfer, IntervalTransfer] | None = None
    source_bins: np.ndarray | None = None
    offsets: np.ndarray | None = None
    targets: np.ndarray | None = None
    weights: np.ndarray | None = None

    def expand(self, bins, max_entries):
        """Yield bounded source-row/target-bin/weight batches; never densify 4D."""
        if self.intervals is None:
            assert self.source_bins is not None and self.offsets is not None
            assert self.targets is not None and self.weights is not None
            positions = np.searchsorted(self.source_bins, bins)
            if np.any(positions >= len(self.source_bins)) or np.any(
                self.source_bins[positions] != bins
            ):
                raise ValueError("An occupied bin is absent from the interface transfer.")
            counts = np.diff(self.offsets)[positions]
            if np.any(counts == 0):
                raise ValueError("An occupied density bin has no physical cell coverage.")
            for rows, ordinal in _expanded_indices(counts, max_entries):
                indices = self.offsets[positions[rows]] + ordinal
                yield rows, self.targets[indices], self.weights[indices]
            return
        first, second = self.intervals
        coordinates = (bins // self.source_shape[1], bins % self.source_shape[1])
        x, y = (coordinates[axis] for axis in self.source_axes)
        nx, ny = np.diff(first.offsets)[x], np.diff(second.offsets)[y]
        counts = nx * ny
        if np.any(counts == 0):
            raise ValueError("An occupied density bin has no physical cell coverage.")

        def batches():
            for rows, ordinal in _expanded_indices(counts, max_entries):
                a = first.offsets[x[rows]] + ordinal // ny[rows]
                b = second.offsets[y[rows]] + ordinal % ny[rows]
                targets = first.targets[a] * self.target_shape[1] + second.targets[b]
                yield rows, targets, first.weights[a] * second.weights[b]

        if np.all(counts == 1):
            # Coincident boundaries produce an exact permutation, including reflections.
            for rows, targets, _ in batches():
                yield rows, targets, np.ones(len(rows))
            return
        # Match the polygon oracle's small-overlap cutoff and row normalization.
        totals = np.zeros(len(bins))
        for rows, _, weights in batches():
            keep = weights > 1e-13
            np.add.at(totals, rows[keep], weights[keep])
        if not np.allclose(totals, 1.0, rtol=1e-8, atol=1e-9):
            raise ValueError("Interface remapping did not preserve the full periodic footprint.")
        for rows, targets, weights in batches():
            keep = weights > 1e-13
            yield rows[keep], targets[keep], weights[keep] / totals[rows[keep]]


def transfer_operator(source_edges, target_edges, basis, origin, alignment, *, source_bins=None):
    """Compile an exact stencil; normal shifts and displayed quantities are not keys."""
    start = perf_counter()
    matrix = np.asarray(alignment.matrix)
    fast = interval_compatible(basis, matrix)
    key = (
        _array_key(basis),
        _array_key(origin),
        tuple(_array_key(e) for e in source_edges),
        tuple(_array_key(e) for e in target_edges),
        alignment.matrix,
        tuple(float(x) for x in np.asarray(alignment.translation) % 1),
        None
        if fast
        else hashlib.sha256(np.ascontiguousarray(source_bins, dtype=np.int64).tobytes()).digest(),
    )
    cached = PREVIEW_CACHE.get("interface.transfer", key)
    if cached is not None:
        LOGGER.debug("Interface transfer cache hit: analytical=%s", fast)
        return cached
    source_shape = (len(source_edges[0]) - 1, len(source_edges[1]) - 1)
    target_shape = (len(target_edges[0]) - 1, len(target_edges[1]) - 1)
    if fast:
        source_axes = (int(np.flatnonzero(matrix[:, 0])[0]), int(np.flatnonzero(matrix[:, 1])[0]))
        intervals = tuple(
            _interval_transfer(
                source_edges[k],
                target_edges[j],
                origin[k],
                basis[k, k],
                origin[j],
                basis[j, j],
                matrix[k, j],
                alignment.translation[j] % 1,
            )
            for j, k in enumerate(source_axes)
        )
        result = LateralTransfer(source_shape, target_shape, source_axes, intervals)
    else:
        from .interface_density import footprint_weights

        inverse = np.linalg.inv(basis)
        polygon = np.asarray(
            [origin, origin + basis[0], origin + basis[0] + basis[1], origin + basis[1]]
        )
        offsets, targets, weights = [0], [], []
        for i, value in enumerate(source_bins):
            if i % 256 == 0:
                check_cancelled()
            stencil = footprint_weights(
                (int(value // source_shape[1]), int(value % source_shape[1])),
                source_edges,
                target_edges,
                basis,
                origin,
                alignment,
                _inverse=inverse,
                _cell_polygon=polygon,
            )
            for x, y, weight in stencil:
                targets.append(x * target_shape[1] + y)
                weights.append(weight)
            offsets.append(len(targets))
        result = LateralTransfer(
            source_shape,
            target_shape,
            source_bins=np.asarray(source_bins),
            offsets=np.asarray(offsets),
            targets=np.asarray(targets, dtype=np.int64),
            weights=np.asarray(weights),
        )
    check_cancelled()
    PREVIEW_CACHE.put("interface.transfer", key, result)
    LOGGER.debug(
        "Interface transfer built: analytical=%s seconds=%.6f", fast, perf_counter() - start
    )
    return result
