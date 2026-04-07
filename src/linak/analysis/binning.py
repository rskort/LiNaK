"""Shared helpers for uniformly binned 1D analysis profiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def uniform_bin_width_from_edges(bin_edges: np.ndarray, *, source_label: str) -> float:
    """Return uniform bin width from edge coordinates or raise for invalid bins."""
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError(f"{source_label} must provide at least two bin edges.")
    widths = np.diff(np.asarray(bin_edges, dtype=float))
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError(f"{source_label} contains invalid bin edges.")
    width = float(widths[0])
    if not np.allclose(widths, width, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(
            f"{source_label} uses non-uniform bins; LiNaK HDF5 stores uniform bins via bin_width_A."
        )
    return width


def resolve_uniform_bin_width_for_load(
    *,
    metadata: Mapping[str, object],
    bin_centers: np.ndarray,
    source_path: Path,
    analysis_name: str,
) -> float:
    """Resolve the required v1 bin width metadata for a binned profile."""
    raw = metadata.get("bin_width_A")
    if raw is None:
        raise ValueError(
            f"{analysis_name} HDF5 '{source_path}' is missing required v1 metadata "
            "bin_width_A. The HDF5 file is either corrupted or originates from the wrong "
            "LiNaK version. Check which LiNaK version generated the file and recompute with "
            "the current LiNaK version if necessary."
        )
    try:
        raw_value: Any = raw
        width = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{analysis_name} HDF5 '{source_path}' has invalid metadata value bin_width_A={raw!r}."
        ) from exc
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(
            f"{analysis_name} HDF5 '{source_path}' has non-positive bin_width_A={raw!r}."
        )
    if bin_centers.size > 1:
        center_steps = np.diff(bin_centers)
        if not np.allclose(center_steps, width, rtol=1.0e-6, atol=1.0e-9):
            raise ValueError(
                f"{analysis_name} HDF5 '{source_path}' has inconsistent "
                "bin_centers_A and bin_width_A."
            )
    return width


def reconstruct_uniform_bin_edges_from_centers(
    bin_centers: np.ndarray,
    *,
    bin_width: float,
) -> np.ndarray:
    """Reconstruct edge coordinates from centers and a uniform bin width."""
    if bin_centers.ndim != 1 or bin_centers.size == 0:
        raise ValueError("Cannot reconstruct bin edges from empty or non-1D bin centers.")
    left_edge = float(bin_centers[0]) - 0.5 * bin_width
    return left_edge + np.arange(bin_centers.size + 1, dtype=float) * bin_width
