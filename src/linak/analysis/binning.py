"""Shared helpers for uniformly binned 1D analysis profiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _coordinate_storage_atol(values: np.ndarray) -> float:
    """Return an absolute comparison tolerance for stored coordinates."""

    array = np.asarray(values)
    tolerance = 1.0e-6
    if array.dtype.kind == "f" and array.dtype.itemsize <= np.dtype(np.float32).itemsize:
        finite = np.asarray(array[np.isfinite(array)], dtype=float)
        scale = float(np.max(np.abs(finite))) if finite.size else 0.0
        tolerance = max(
            tolerance,
            2.0 * float(np.finfo(array.dtype).eps) * scale,
        )
    return tolerance


def uniform_bin_width_from_edges(bin_edges: np.ndarray, *, source_label: str) -> float:
    """Return uniform bin width from edge coordinates or raise for invalid bins."""
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError(f"{source_label} must provide at least two bin edges.")
    stored_edges = np.asarray(bin_edges)
    widths = np.diff(np.asarray(stored_edges, dtype=float))
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError(f"{source_label} contains invalid bin edges.")
    width = float(widths[0])
    # Plot-ready bin coordinates use the mixed-v1 float32 storage tier.  Once
    # reloaded as float64 their adjacent differences retain float32 rounding,
    # so validate against the storage precision rather than an unattainable
    # float64 tolerance.  Classification and accumulation still use the
    # original float64 edges before storage.
    if not np.allclose(
        widths,
        width,
        rtol=5.0e-7,
        atol=_coordinate_storage_atol(stored_edges),
    ):
        raise ValueError(
            f"{source_label} uses non-uniform bins; LiNaK HDF5 stores uniform bins via bin_width_A."
        )
    return width


def reconstruct_uniform_bin_edges_from_edges(
    bin_edges: np.ndarray,
    *,
    bin_width: float | None,
    source_label: str,
) -> tuple[np.ndarray, float]:
    """Validate stored edges at their precision and recover a uniform float64 grid."""

    stored_edges = np.asarray(bin_edges)
    uniform_bin_width_from_edges(stored_edges, source_label=source_label)
    edges = np.asarray(stored_edges, dtype=float)
    width = (
        float(bin_width)
        if bin_width is not None
        else float((edges[-1] - edges[0]) / (edges.size - 1))
    )
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"{source_label} has invalid bin width {bin_width!r}.")
    # Keep the stored origin, but use the metadata width rather than one rounded
    # adjacent difference. This avoids drift and differing widths across sources.
    uniform_edges = float(edges[0]) + np.arange(edges.size, dtype=float) * width
    if not np.allclose(
        edges,
        uniform_edges,
        rtol=0.0,
        atol=_coordinate_storage_atol(stored_edges),
    ):
        raise ValueError(f"{source_label} has bin edges inconsistent with its bin width.")
    return uniform_edges, width


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
        stored_centers = np.asarray(bin_centers)
        center_steps = np.diff(np.asarray(stored_centers, dtype=float))
        if not np.allclose(
            center_steps,
            width,
            rtol=1.0e-6,
            atol=_coordinate_storage_atol(stored_centers),
        ):
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
    if abs(left_edge) <= 1.0e-6:
        left_edge = 0.0
    return left_edge + np.arange(bin_centers.size + 1, dtype=float) * bin_width
