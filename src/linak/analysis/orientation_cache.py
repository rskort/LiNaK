"""Read-only, revision-keyed orientation data owned by the plotting GUI."""

from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
import logging
from time import perf_counter

import numpy as np

from ..cache import PREVIEW_CACHE, source_revision
from ..cancellation import check_cancelled

LOGGER = logging.getLogger("linak")


def freeze_arrays(value, seen=None):
    """Freeze GUI-owned buffers, including the owners of geometry-array views."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        if isinstance(value.base, np.ndarray):
            freeze_arrays(value.base, seen)
    elif isinstance(value, Mapping):
        for item in value.values():
            freeze_arrays(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze_arrays(item, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            freeze_arrays(getattr(value, field.name), seen)


def load_gui_orientation_profiles(path, indices, dataset_names):
    """Alignment and display changes never form part of the source-data key."""
    from .orientation import load_orientation_profiles_by_index

    started = perf_counter()
    revision = source_revision(path)
    selection = tuple(sorted(dataset_names))
    keys = [(revision, int(index), selection) for index in indices]
    results = [PREVIEW_CACHE.get("orientation.source", key) for key in keys]
    missing = [position for position, result in enumerate(results) if result is None]
    if missing:
        loaded = load_orientation_profiles_by_index(
            path, [indices[i] for i in missing], dataset_names=selection
        )
        check_cancelled()
        # Do not publish a read that raced a source replacement.
        stable = revision == source_revision(path)
        for position, profile in zip(missing, loaded):
            frozen = replace(profile, _gui_cache_key=keys[position] if stable else None)
            freeze_arrays(frozen)
            results[position] = frozen
        check_cancelled()
        if stable:
            for position in missing:
                PREVIEW_CACHE.put("orientation.source", keys[position], results[position])
    LOGGER.debug(
        "Orientation source data: hits=%d reads=%d seconds=%.6f",
        len(indices) - len(missing),
        len(missing),
        perf_counter() - started,
    )
    return results
