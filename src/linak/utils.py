"""Shared utility helpers for LiNaK."""

from __future__ import annotations

import numpy as np

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
SI_PREFIX_SYMBOLS = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "u",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}


def axis_to_index(axis: str) -> int:
    """Return integer axis index for ``x``, ``y``, or ``z``.

    Parameters
    ----------
    axis
        Axis label.

    Returns
    -------
    int
        Axis index (`x` -> 0, `y` -> 1, `z` -> 2).

    Raises
    ------
    ValueError
        If axis is not one of `x`, `y`, `z`.
    """
    normalized = axis.lower()
    if normalized not in AXIS_TO_INDEX:
        raise ValueError(f"Unsupported axis '{axis}'. Choose from x, y, z.")
    return AXIS_TO_INDEX[normalized]


def ensure_positive(name: str, value: float) -> None:
    """Validate that a numeric argument is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value}).")


def choose_engineering_prefix_exponent(values: np.ndarray) -> int:
    """Return a 10^n engineering exponent (n multiple of 3) for numeric values."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        return 0

    finite = np.abs(array[np.isfinite(array)])
    finite = finite[finite > 0]
    if finite.size == 0:
        return 0

    exponent = int(np.floor(np.log10(float(np.max(finite))) / 3.0) * 3)
    exponent = max(min(exponent, max(SI_PREFIX_SYMBOLS)), min(SI_PREFIX_SYMBOLS))
    exponent = 3 * int(round(exponent / 3))
    return exponent if exponent in SI_PREFIX_SYMBOLS else 0


def scale_series_with_si_prefix(
    series: list[np.ndarray],
    unit: str,
) -> tuple[list[np.ndarray], str, float]:
    """Scale numeric series with one shared SI prefix and return scaled unit."""
    if not series:
        return [], unit, 1.0

    flattened = [np.asarray(values, dtype=float).ravel() for values in series]
    non_empty = [values for values in flattened if values.size > 0]
    if not non_empty:
        return [np.asarray(values, dtype=float) for values in series], unit, 1.0
    combined = np.concatenate(non_empty)
    exponent = choose_engineering_prefix_exponent(combined)
    factor = 10.0 ** (-exponent)
    prefix = SI_PREFIX_SYMBOLS.get(exponent, "")
    scaled_unit = f"{prefix}{unit}"
    scaled_series = [np.asarray(values, dtype=float) * factor for values in series]
    return scaled_series, scaled_unit, factor
