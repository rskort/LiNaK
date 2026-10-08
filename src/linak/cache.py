"""Shared byte-aware in-memory caches used by plotting and geometry."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import fields, is_dataclass
import os
import sys
from threading import RLock
from typing import Any

import numpy as np

MIB = 1024 * 1024
DEFAULT_CACHE_BUDGET_BYTES = 512 * MIB


def _cache_budget_from_environment() -> int:
    raw = str(os.getenv("LINAK_CACHE_BUDGET_MIB", "")).strip()
    if not raw:
        return DEFAULT_CACHE_BUDGET_BYTES
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CACHE_BUDGET_BYTES
    return max(1, int(value * MIB)) if value > 0 else DEFAULT_CACHE_BUDGET_BYTES


def estimate_object_bytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Estimate recursively owned bytes without double-counting shared objects/buffers."""

    seen = set() if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    if isinstance(value, np.ndarray):
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        # CPython includes an owned ndarray's data buffer in getsizeof(), but
        # includes only the array header for views.  Normalize to a header
        # estimate and then count the ultimate buffer exactly once.
        header_bytes = sys.getsizeof(value)
        if owner is value:
            header_bytes = max(0, header_bytes - int(value.nbytes))
        owner_id = id(owner)
        if owner_id != object_id:
            if owner_id in seen:
                return header_bytes
            seen.add(owner_id)
        return header_bytes + int(owner.nbytes)
    total = sys.getsizeof(value)
    if isinstance(value, Mapping):
        return total + sum(
            estimate_object_bytes(key, _seen=seen) + estimate_object_bytes(item, _seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return total + sum(estimate_object_bytes(item, _seen=seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return total + sum(
            estimate_object_bytes(getattr(value, field.name), _seen=seen) for field in fields(value)
        )
    return total


class ByteAwareLRU:
    """Thread-safe global LRU whose limit is expressed in estimated bytes."""

    def __init__(self, budget_bytes: int | None = None) -> None:
        self._budget_bytes = max(
            1,
            int(_cache_budget_from_environment() if budget_bytes is None else budget_bytes),
        )
        self._entries: OrderedDict[tuple[str, Hashable], tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def current_bytes(self) -> int:
        return self._bytes

    def set_budget_mib(self, budget_mib: float) -> int:
        if not float(budget_mib) > 0:
            raise ValueError("Cache budget must be a positive number of MiB.")
        with self._lock:
            self._budget_bytes = max(1, int(float(budget_mib) * MIB))
            self._evict_to_budget()
            return self._budget_bytes

    def get(self, namespace: str, key: Hashable, default: Any = None) -> Any:
        cache_key = (str(namespace), key)
        with self._lock:
            entry = self._entries.pop(cache_key, None)
            if entry is None:
                self._misses += 1
                return default
            self._entries[cache_key] = entry
            self._hits += 1
            return entry[0]

    def put(self, namespace: str, key: Hashable, value: Any) -> int:
        cache_key = (str(namespace), key)
        size = estimate_object_bytes((cache_key, value))
        with self._lock:
            previous = self._entries.pop(cache_key, None)
            if previous is not None:
                self._bytes -= previous[1]
            if size > self._budget_bytes:
                return size
            self._entries[cache_key] = (value, size)
            self._bytes += size
            self._evict_to_budget()
        return size

    def _evict_to_budget(self) -> None:
        while self._bytes > self._budget_bytes and self._entries:
            _key, (_value, size) = self._entries.popitem(last=False)
            self._bytes -= size
            self._evictions += 1

    def clear(self, namespace: str | None = None) -> int:
        """Clear all or one namespace and return estimated released bytes."""

        with self._lock:
            if namespace is None:
                released = self._bytes
                self._entries.clear()
                self._bytes = 0
                return released
            prefix = str(namespace)
            keys = [key for key in self._entries if key[0] == prefix]
            released = 0
            for key in keys:
                _value, size = self._entries.pop(key)
                released += size
                self._bytes -= size
            return released

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "budget_bytes": self._budget_bytes,
                "current_bytes": self._bytes,
                "entry_count": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }


PREVIEW_CACHE = ByteAwareLRU()


def clear_preview_cache() -> int:
    """Clear only in-memory preview/geometry data and return bytes released."""

    return PREVIEW_CACHE.clear()


__all__ = (
    "ByteAwareLRU",
    "DEFAULT_CACHE_BUDGET_BYTES",
    "PREVIEW_CACHE",
    "clear_preview_cache",
    "estimate_object_bytes",
)
