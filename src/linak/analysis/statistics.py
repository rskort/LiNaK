"""Shared uncertainty-statistics helpers for saved and plotted LiNaK series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

_SERIES_STAT_SUFFIXES = (
    "point_count",
    "sample_n",
    "sample_std",
    "sample_sem",
    "block_n",
    "block_std",
    "block_sem",
)
_DEFAULT_ERROR_STAT = "sample_sem"


@dataclass(frozen=True)
class SeriesStatistics:
    """Persisted statistics for one 1-D plotted dataset."""

    point_count: np.ndarray
    sample_n: np.ndarray
    sample_std: np.ndarray
    sample_sem: np.ndarray
    block_n: np.ndarray | None = None
    block_std: np.ndarray | None = None
    block_sem: np.ndarray | None = None


def statistics_suffixes() -> tuple[str, ...]:
    """Return the persisted suffix contract for per-series statistics datasets."""
    return _SERIES_STAT_SUFFIXES


def statistics_default_error_stat() -> str:
    """Return the default error statistic when no block family is available."""
    return _DEFAULT_ERROR_STAT


def resolve_block_slices(frame_count: int) -> tuple[list[slice], list[int]] | None:
    """Return contiguous block slices for block statistics or ``None`` when unavailable."""
    frame_total = int(frame_count)
    if frame_total < 100:
        return None
    block_count = min(20, frame_total // 25)
    if block_count < 4:
        return None
    indices = np.array_split(np.arange(frame_total, dtype=int), block_count)
    if len(indices) < 4:
        return None
    lengths = [int(chunk.size) for chunk in indices]
    if not lengths or min(lengths) < 25:
        return None
    slices = [slice(int(chunk[0]), int(chunk[-1]) + 1) for chunk in indices if chunk.size > 0]
    if len(slices) < 4:
        return None
    return slices, lengths


def _validate_stat_shape(name: str, values: np.ndarray, *, expected_shape: tuple[int, ...]) -> None:
    if values.shape != expected_shape:
        raise ValueError(
            f"Series statistics field '{name}' has shape {values.shape}, expected {expected_shape}."
        )


def statistics_available_stats(stats: SeriesStatistics | None) -> list[str]:
    """Return the available error-stat identifiers for one statistics payload."""
    if stats is None:
        return []
    available = ["sample_std", "sample_sem"]
    if stats.block_std is not None and stats.block_sem is not None and stats.block_n is not None:
        available.extend(["block_std", "block_sem"])
    return available


def default_error_stat_for_statistics(stats: SeriesStatistics | None) -> str | None:
    """Return the preferred error statistic name for one series statistics payload."""
    available = statistics_available_stats(stats)
    if not available:
        return None
    return "block_sem" if "block_sem" in available else "sample_sem"


def build_statistics_metadata(
    *,
    statistics_by_series: Mapping[str, SeriesStatistics] | None,
    block_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return nested metadata describing available statistics families."""
    if not statistics_by_series:
        return {}
    any_block = any(
        stats.block_n is not None and stats.block_std is not None and stats.block_sem is not None
        for stats in statistics_by_series.values()
    )
    families = ["sample", "block"] if any_block else ["sample"]
    metadata: dict[str, Any] = {
        "families": families,
        "default_error_stat": "block_sem" if any_block else _DEFAULT_ERROR_STAT,
    }
    if any_block and block_lengths is not None:
        lengths = [int(length) for length in block_lengths]
        metadata["block_count"] = len(lengths)
        metadata["block_lengths"] = lengths
    return metadata


def statistics_payload_from_series_map(
    statistics_by_series: Mapping[str, SeriesStatistics] | None,
) -> dict[str, np.ndarray]:
    """Flatten one per-series statistics mapping into HDF5 dataset payloads."""
    if not statistics_by_series:
        return {}
    payload: dict[str, np.ndarray] = {}
    for dataset_name, stats in statistics_by_series.items():
        base = str(dataset_name).strip()
        if not base:
            continue
        payload[f"{base}_point_count"] = np.asarray(stats.point_count, dtype=int)
        payload[f"{base}_sample_n"] = np.asarray(stats.sample_n, dtype=int)
        payload[f"{base}_sample_std"] = np.asarray(stats.sample_std, dtype=float)
        payload[f"{base}_sample_sem"] = np.asarray(stats.sample_sem, dtype=float)
        if stats.block_n is not None:
            payload[f"{base}_block_n"] = np.asarray(stats.block_n, dtype=int)
        if stats.block_std is not None:
            payload[f"{base}_block_std"] = np.asarray(stats.block_std, dtype=float)
        if stats.block_sem is not None:
            payload[f"{base}_block_sem"] = np.asarray(stats.block_sem, dtype=float)
    return payload


def statistics_series_map_from_datasets(
    datasets: Mapping[str, np.ndarray],
    *,
    dataset_names: Sequence[str],
) -> dict[str, SeriesStatistics] | None:
    """Load per-series statistics mapping from flat HDF5 datasets."""
    resolved: dict[str, SeriesStatistics] = {}
    for dataset_name in dataset_names:
        base = str(dataset_name).strip()
        if not base:
            continue
        point_count_key = f"{base}_point_count"
        sample_n_key = f"{base}_sample_n"
        sample_std_key = f"{base}_sample_std"
        sample_sem_key = f"{base}_sample_sem"
        if (
            point_count_key not in datasets
            or sample_n_key not in datasets
            or sample_std_key not in datasets
            or sample_sem_key not in datasets
        ):
            continue
        point_count = np.asarray(datasets[point_count_key], dtype=int)
        sample_n = np.asarray(datasets[sample_n_key], dtype=int)
        sample_std = np.asarray(datasets[sample_std_key], dtype=float)
        sample_sem = np.asarray(datasets[sample_sem_key], dtype=float)
        expected_shape = point_count.shape
        _validate_stat_shape("sample_n", sample_n, expected_shape=expected_shape)
        _validate_stat_shape("sample_std", sample_std, expected_shape=expected_shape)
        _validate_stat_shape("sample_sem", sample_sem, expected_shape=expected_shape)
        block_n_key = f"{base}_block_n"
        block_std_key = f"{base}_block_std"
        block_sem_key = f"{base}_block_sem"
        block_n = np.asarray(datasets[block_n_key], dtype=int) if block_n_key in datasets else None
        block_std = (
            np.asarray(datasets[block_std_key], dtype=float) if block_std_key in datasets else None
        )
        block_sem = (
            np.asarray(datasets[block_sem_key], dtype=float) if block_sem_key in datasets else None
        )
        if block_n is not None:
            _validate_stat_shape("block_n", block_n, expected_shape=expected_shape)
        if block_std is not None:
            _validate_stat_shape("block_std", block_std, expected_shape=expected_shape)
        if block_sem is not None:
            _validate_stat_shape("block_sem", block_sem, expected_shape=expected_shape)
        resolved[base] = SeriesStatistics(
            point_count=point_count,
            sample_n=sample_n,
            sample_std=sample_std,
            sample_sem=sample_sem,
            block_n=block_n,
            block_std=block_std,
            block_sem=block_sem,
        )
    return resolved or None


def _sample_std_and_sem_from_matrix(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_matrix = np.asarray(samples, dtype=float)
    if sample_matrix.ndim != 2:
        raise ValueError("Series statistics samples must be a 2D array.")
    finite = np.isfinite(sample_matrix)
    sample_n = np.sum(finite, axis=0).astype(int)
    safe_samples = np.where(finite, sample_matrix, 0.0)
    sample_sum = np.sum(safe_samples, axis=0)
    sample_mean = np.divide(
        sample_sum,
        sample_n,
        out=np.full(sample_sum.shape, np.nan, dtype=float),
        where=sample_n > 0,
    )
    centered = np.where(finite, sample_matrix - sample_mean[np.newaxis, :], 0.0)
    numerator = np.sum(centered**2, axis=0)
    sample_std = np.full(sample_mean.shape, np.nan, dtype=float)
    valid_std = sample_n > 1
    sample_std[valid_std] = np.sqrt(numerator[valid_std] / (sample_n[valid_std] - 1))
    sample_sem = np.full(sample_mean.shape, np.nan, dtype=float)
    sample_sem[valid_std] = sample_std[valid_std] / np.sqrt(sample_n[valid_std].astype(float))
    return sample_n, sample_std, sample_sem


def build_series_statistics(
    *,
    point_count: np.ndarray,
    sample_values: np.ndarray,
    block_values: np.ndarray | None = None,
) -> SeriesStatistics:
    """Build persisted statistics from simple and optional block sample matrices."""
    point_count_array = np.asarray(point_count, dtype=int)
    sample_n, sample_std, sample_sem = _sample_std_and_sem_from_matrix(sample_values)
    if sample_n.shape != point_count_array.shape:
        raise ValueError("Series statistics point_count shape must match the per-bin sample shape.")
    stats = SeriesStatistics(
        point_count=point_count_array,
        sample_n=sample_n,
        sample_std=sample_std,
        sample_sem=sample_sem,
    )
    if block_values is None:
        return stats
    block_n, block_std, block_sem = _sample_std_and_sem_from_matrix(block_values)
    if block_n.shape != point_count_array.shape:
        raise ValueError(
            "Series statistics block sample shape must match the per-bin point_count shape."
        )
    return replace(
        stats,
        block_n=block_n,
        block_std=block_std,
        block_sem=block_sem,
    )


def build_series_statistics_from_moments(
    *,
    point_count: np.ndarray,
    sample_n: np.ndarray,
    sample_sum: np.ndarray,
    sample_sumsq: np.ndarray,
    block_values: np.ndarray | None = None,
) -> SeriesStatistics:
    """Build persisted statistics from per-bin sample moments plus optional block means."""
    point_count_array = np.asarray(point_count, dtype=int)
    sample_n_array = np.asarray(sample_n, dtype=int)
    sample_sum_array = np.asarray(sample_sum, dtype=float)
    sample_sumsq_array = np.asarray(sample_sumsq, dtype=float)
    if not (
        sample_n_array.shape
        == sample_sum_array.shape
        == sample_sumsq_array.shape
        == point_count_array.shape
    ):
        raise ValueError("Sample moment arrays must share the same per-bin shape.")
    sample_std = np.full(sample_sum_array.shape, np.nan, dtype=float)
    valid_std = sample_n_array > 1
    centered_sum = np.full(sample_sum_array.shape, np.nan, dtype=float)
    centered_sum[valid_std] = sample_sumsq_array[valid_std] - (
        (sample_sum_array[valid_std] ** 2) / sample_n_array[valid_std].astype(float)
    )
    centered_sum[valid_std] = np.maximum(centered_sum[valid_std], 0.0)
    sample_std[valid_std] = np.sqrt(
        centered_sum[valid_std] / (sample_n_array[valid_std].astype(float) - 1.0)
    )
    sample_sem = np.full(sample_sum_array.shape, np.nan, dtype=float)
    sample_sem[valid_std] = sample_std[valid_std] / np.sqrt(sample_n_array[valid_std].astype(float))
    stats = SeriesStatistics(
        point_count=point_count_array,
        sample_n=sample_n_array,
        sample_std=sample_std,
        sample_sem=sample_sem,
    )
    if block_values is None:
        return stats
    block_n, block_std, block_sem = _sample_std_and_sem_from_matrix(block_values)
    if block_n.shape != point_count_array.shape:
        raise ValueError(
            "Series statistics block sample shape must match the per-bin point_count shape."
        )
    return replace(
        stats,
        block_n=block_n,
        block_std=block_std,
        block_sem=block_sem,
    )


def block_mean_matrix(
    sample_values: np.ndarray,
    *,
    block_slices: Sequence[slice] | None,
) -> np.ndarray | None:
    """Return one matrix of block-mean values over contiguous sample slices."""
    if block_slices is None:
        return None
    sample_matrix = np.asarray(sample_values, dtype=float)
    if sample_matrix.ndim != 2:
        raise ValueError("Series statistics block mean input must be a 2D array.")
    block_rows: list[np.ndarray] = []
    for block_slice in block_slices:
        block = np.asarray(sample_matrix[block_slice], dtype=float)
        if block.ndim != 2 or block.shape[0] == 0:
            continue
        finite = np.isfinite(block)
        counts = np.sum(finite, axis=0)
        safe = np.where(finite, block, 0.0)
        means = np.divide(
            np.sum(safe, axis=0),
            counts,
            out=np.full(block.shape[1], np.nan, dtype=float),
            where=counts > 0,
        )
        block_rows.append(means)
    if not block_rows:
        return None
    return np.vstack(block_rows)
