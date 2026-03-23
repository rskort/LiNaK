"""Runtime configuration helpers for native math-library thread counts.

This module must stay lightweight and import only the standard library so it
can run before NumPy/ASE imports in the CLI entry path.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
import os

_BACKEND_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_DISABLE_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class NativeThreadEnvConfiguration:
    """Summary of LiNaK's native-thread environment bootstrap."""

    applied: dict[str, str]
    skipped_reason: str | None = None
    requested_threads: int | None = None
    invalid_value: str | None = None


def _parse_positive_int(raw_value: str) -> int | None:
    stripped = raw_value.strip()
    if not stripped:
        return None
    try:
        parsed = int(stripped)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def configure_native_thread_env(
    environ: MutableMapping[str, str] | None = None,
) -> NativeThreadEnvConfiguration:
    """Apply safe default thread caps for BLAS/OpenMP backends.

    The configuration is only applied when:
    - LiNaK-specific disabling is not requested
    - none of the known backend thread-count environment variables are already set

    Parameters
    ----------
    environ
        Optional mapping used instead of :data:`os.environ`. This exists mainly
        for testing.
    """

    env = os.environ if environ is None else environ

    disable_token = str(env.get("LINAK_DISABLE_THREAD_CAP", "")).strip().lower()
    if disable_token in _DISABLE_TRUE_VALUES:
        return NativeThreadEnvConfiguration(applied={}, skipped_reason="disabled")

    if any(str(env.get(key, "")).strip() for key in _BACKEND_THREAD_ENV_VARS):
        return NativeThreadEnvConfiguration(applied={}, skipped_reason="preconfigured")

    requested_raw = str(env.get("LINAK_NUM_THREADS", "")).strip()
    if requested_raw:
        requested_threads = _parse_positive_int(requested_raw)
        if requested_threads is None:
            return NativeThreadEnvConfiguration(
                applied={},
                skipped_reason="invalid_linak_num_threads",
                invalid_value=requested_raw,
            )
    else:
        requested_threads = 1

    applied = {key: str(requested_threads) for key in _BACKEND_THREAD_ENV_VARS}
    env.update(applied)
    return NativeThreadEnvConfiguration(
        applied=applied,
        requested_threads=requested_threads,
    )


__all__ = (
    "NativeThreadEnvConfiguration",
    "configure_native_thread_env",
)
