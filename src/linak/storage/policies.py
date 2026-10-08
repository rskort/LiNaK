"""Shared HDF5 storage policies.

The policy in this module deliberately affects only the representation written
to disk.  Scientific calculations continue to operate on their native arrays
and readers upcast where geometry or reductions require float64.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

CompressionProfile = Literal["fast", "balanced", "none"]

DEFAULT_COMPRESSION_PROFILE: CompressionProfile = "fast"
STORAGE_PRECISION_POLICY = "mixed-v1"
TARGET_HDF5_CHUNK_BYTES = 1024 * 1024

_COMPRESSION_PROFILES = frozenset({"fast", "balanced", "none"})
_FLOAT64_ANALYSES = frozenset({"potential", "work_function"})
_FLOAT64_NAME_PARTS = (
    "cell",
    "origin",
    "surface_position",
    "surface_offset",
    "surface_plane",
    "geometry_",
    "pbc_transform",
    "fit_",
    "expected_",
    "work_function",
    "potential",
)
_FLOAT64_NAME_SUFFIXES = ("_sum", "_sumsq", "_moment")


def normalize_compression_profile(value: str | None) -> CompressionProfile:
    """Validate and normalize a public HDF5 compression profile."""

    normalized = DEFAULT_COMPRESSION_PROFILE if value is None else str(value).strip().lower()
    if normalized not in _COMPRESSION_PROFILES:
        choices = ", ".join(sorted(_COMPRESSION_PROFILES))
        raise ValueError(f"Unknown HDF5 compression profile '{value}'. Choose from: {choices}.")
    return normalized  # type: ignore[return-value]


def storage_dtype(*, analysis: str, name: str, dtype: Any) -> np.dtype[Any]:
    """Return the on-disk dtype selected by ``mixed-v1``."""

    resolved = np.dtype(dtype)
    if resolved.kind != "f":
        return resolved
    analysis_key = str(analysis).strip().lower()
    dataset_key = str(name).strip().lower()
    keep_float64 = (
        analysis_key in _FLOAT64_ANALYSES
        or any(part in dataset_key for part in _FLOAT64_NAME_PARTS)
        or dataset_key.endswith(_FLOAT64_NAME_SUFFIXES)
    )
    return np.dtype(np.float64 if keep_float64 else np.float32)


def storage_array(*, analysis: str, name: str, values: Any) -> np.ndarray:
    """Return an array cast according to the fixed ``mixed-v1`` policy."""

    array = np.asarray(values)
    return np.asarray(
        array,
        dtype=storage_dtype(analysis=analysis, name=name, dtype=array.dtype),
    )


def target_chunk_shape(
    shape: tuple[int, ...],
    *,
    itemsize: int,
    target_bytes: int = TARGET_HDF5_CHUNK_BYTES,
) -> tuple[int, ...] | None:
    """Choose a roughly target-sized HDF5 chunk.

    The leading dimension is treated as frame/profile-major and is always
    clamped to 1--256 entries.  Remaining dimensions are reduced only when one
    complete row would exceed the target.
    """

    if not shape or any(int(value) <= 0 for value in shape):
        return None
    chunk = [int(value) for value in shape]
    chunk[0] = min(256, max(1, chunk[0]))
    safe_itemsize = max(1, int(itemsize))
    target = max(safe_itemsize, int(target_bytes))
    while int(np.prod(chunk, dtype=np.int64)) * safe_itemsize > target:
        candidates = [index for index, value in enumerate(chunk) if value > 1]
        if not candidates:
            break
        # Prefer reducing a non-frame dimension when it dominates.  Otherwise
        # reduce the leading dimension to preserve contiguous rows.
        largest = max(candidates, key=lambda index: chunk[index])
        chunk[largest] = max(1, (chunk[largest] + 1) // 2)
    return tuple(chunk)


def dataset_create_options(
    array: np.ndarray,
    *,
    compression_profile: str | None,
) -> dict[str, Any]:
    """Return chunking/compression arguments for ``create_dataset``."""

    return dataset_create_options_for_shape(
        array.shape,
        dtype=array.dtype,
        compression_profile=compression_profile,
    )


def dataset_create_options_for_shape(
    shape: tuple[int, ...],
    *,
    dtype: Any,
    compression_profile: str | None,
) -> dict[str, Any]:
    """Apply the storage policy without allocating an array of the final size."""
    profile = normalize_compression_profile(compression_profile)
    resolved_dtype = np.dtype(dtype)
    if not shape or any(size == 0 for size in shape) or resolved_dtype.kind == "O":
        return {}
    options: dict[str, Any] = {
        "chunks": target_chunk_shape(shape, itemsize=resolved_dtype.itemsize),
    }
    if profile == "fast":
        options["compression"] = "lzf"
    elif profile == "balanced":
        options["compression"] = "gzip"
        options["compression_opts"] = 4
    if profile != "none" and resolved_dtype.kind in "iufc":
        options["shuffle"] = True
    return {key: value for key, value in options.items() if value is not None}


def policy_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    compression_profile: str | None,
) -> dict[str, Any]:
    """Return metadata with storage provenance added without overwriting callers."""

    profile = normalize_compression_profile(compression_profile)
    result = dict(metadata or {})
    result.setdefault("compression_profile", profile)
    result.setdefault("storage_precision_policy", STORAGE_PRECISION_POLICY)
    return result


__all__ = (
    "CompressionProfile",
    "DEFAULT_COMPRESSION_PROFILE",
    "STORAGE_PRECISION_POLICY",
    "TARGET_HDF5_CHUNK_BYTES",
    "dataset_create_options",
    "dataset_create_options_for_shape",
    "normalize_compression_profile",
    "policy_metadata",
    "storage_array",
    "storage_dtype",
    "target_chunk_shape",
)
