"""Plot-facing profile loading and bounded, payload-free header caching.

The small header LRU is separate from the byte-budgeted preview payload cache.
File state participates in identity, and callers receive independent metadata.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

# Metadata inventories are small; numerical datasets never enter this cache.
_HEADER_CACHE_MAX_ENTRIES = 128


@lru_cache(maxsize=_HEADER_CACHE_MAX_ENTRIES)
def _cached_profile_headers(
    source: str,
    analysis: str,
    mtime_ns: int | None,
    size_bytes: int | None,
) -> list[dict[str, Any]]:
    from ..storage.hdf5_utils import read_linak_hdf5_profile_headers

    return read_linak_hdf5_profile_headers(source, expected_analysis=analysis)


def clear_profile_header_cache() -> None:
    """Discard in-memory header metadata; never modify analysis files."""
    _cached_profile_headers.cache_clear()


def read_analysis_profile_payloads(
    *,
    sources: list[str],
    analysis: str,
) -> list[dict[str, Any]]:
    payloads_by_source = read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis=analysis,
    )
    return [
        payload for _source, source_payloads in payloads_by_source for payload in source_payloads
    ]


def read_analysis_profile_payloads_by_source(
    *,
    sources: list[str],
    analysis: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    from ..storage.hdf5_utils import read_linak_hdf5_profiles

    payloads_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source_index, source in enumerate(sources):
        source_path = Path(source).expanduser().resolve()
        profiles = read_linak_hdf5_profiles(source_path, expected_analysis=analysis)
        if not profiles:
            raise ValueError(f"No '{analysis}' profiles found in '{source_path}'.")
        source_payloads: list[dict[str, Any]] = []
        for profile_index, (datasets, metadata) in enumerate(profiles):
            merged_metadata = dict(metadata)
            merged_metadata["origin_hdf5_path"] = str(
                merged_metadata.get("origin_hdf5_path") or source_path
            )
            merged_metadata["source_path"] = str(source_path)
            merged_metadata.setdefault("source_index", source_index)
            merged_metadata.setdefault("source_profile_index", profile_index)
            source_payloads.append(
                {
                    "datasets": datasets,
                    "metadata": merged_metadata,
                }
            )
        payloads_by_source.append((source, source_payloads))
    return payloads_by_source


def read_analysis_profile_headers_by_source(
    *,
    sources: list[str],
    analysis: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    headers_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source_index, source in enumerate(sources):
        source_path = Path(source).expanduser().resolve()
        try:
            stat = source_path.stat()
            header_cache_key: tuple[str, str, int | None, int | None] = (
                str(source_path),
                str(analysis),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except OSError:
            header_cache_key = (str(source_path), str(analysis), None, None)
        headers = deepcopy(_cached_profile_headers(*header_cache_key))
        if not headers:
            raise ValueError(f"No '{analysis}' profiles found in '{source_path}'.")
        source_headers: list[dict[str, Any]] = []
        for profile_index, metadata in enumerate(headers):
            merged_metadata = dict(metadata)
            merged_metadata["origin_hdf5_path"] = str(
                merged_metadata.get("origin_hdf5_path") or source_path
            )
            merged_metadata["source_path"] = str(source_path)
            merged_metadata.setdefault("source_index", source_index)
            merged_metadata.setdefault("source_profile_index", profile_index)
            merged_metadata.setdefault("profile_index", profile_index)
            source_headers.append(merged_metadata)
        headers_by_source.append((source, source_headers))
    return headers_by_source
