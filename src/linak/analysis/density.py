"""Density analysis routines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list

from ..storage.hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles,
    write_linak_hdf5,
)
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    normalize_backend_name as normalize_backend_name,
    plot_line_series,
    plot_multi_line_series,
)
from ..progress import ProgressBar
from ..utils import axis_to_index, ensure_positive

LOGGER = logging.getLogger(__name__)
H2O_VALIDATION_STRIDE = 250
AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24
SURFACE_MOBILITY_FRACTION = 0.35
SURFACE_POSITION_QUANTILE = 0.90
MIN_SURFACE_REFERENCE_ATOMS = 6
SURFACE_LAYER_GAP_MIN = 0.25
SURFACE_LAYER_GAP_FACTOR = 3.0
SURFACE_LAYER_MIN_SUCCESS_RATIO = 0.60
SURFACE_MOBILITY_EPSILON = 1.0e-8
H2O_MASS_G = float(
    (atomic_masses[atomic_numbers["H"]] * 2.0 + atomic_masses[atomic_numbers["O"]]) * AMU_TO_G
)


@dataclass(frozen=True)
class DensityProfile:
    """Container for a 1D density profile."""

    axis: str
    species: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts_per_frame: np.ndarray
    density: np.ndarray
    units: str
    n_frames: int
    entities_per_frame: np.ndarray | None = None
    number_density: np.ndarray | None = None
    number_density_units: str | None = None
    coordinate_mode: str = "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None


@dataclass(frozen=True)
class _SurfaceEstimate:
    position: float
    std: float
    method: str
    success_ratio: float
    per_frame: np.ndarray


def _frame_has_usable_cell(frame: Atoms, axis_index: int) -> bool:
    """Check whether a frame has a finite non-zero cell for volumetric density."""
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return False

    axis_length = np.linalg.norm(cell[axis_index])
    volume = abs(float(np.linalg.det(cell)))
    return axis_length > 0 and volume > 0


def _frames_share_consistent_atom_layout(frames: list[Atoms]) -> bool:
    """Return whether all frames have equal atom counts and symbol ordering."""
    if not frames:
        return False

    first_symbols = np.asarray(frames[0].get_chemical_symbols())
    atom_count = len(first_symbols)
    for frame in frames[1:]:
        frame_symbols = np.asarray(frame.get_chemical_symbols())
        if len(frame_symbols) != atom_count:
            return False
        if not np.array_equal(frame_symbols, first_symbols):
            return False
    return True


def _axis_is_periodic(frame: Atoms, axis_index: int) -> bool:
    """Return whether the selected axis is periodic for this frame."""
    pbc = np.asarray(frame.get_pbc(), dtype=bool)
    return pbc.size == 3 and bool(pbc[axis_index])


def _unwrap_axis_positions(
    axis_matrix: np.ndarray, axis_lengths: np.ndarray | None
) -> np.ndarray:
    """Unwrap per-atom axis trajectories to avoid periodic boundary jumps."""
    if axis_matrix.ndim != 2 or axis_matrix.shape[0] <= 1 or axis_lengths is None:
        return axis_matrix

    unwrapped = np.array(axis_matrix, dtype=float, copy=True)
    for frame_index in range(1, axis_matrix.shape[0]):
        axis_length = float(axis_lengths[frame_index - 1])
        if not np.isfinite(axis_length) or axis_length <= 0.0:
            continue
        delta = axis_matrix[frame_index] - axis_matrix[frame_index - 1]
        delta -= axis_length * np.rint(delta / axis_length)
        unwrapped[frame_index] = unwrapped[frame_index - 1] + delta
    return unwrapped


def _axis_lengths_if_periodic(frames: list[Atoms], axis_index: int) -> np.ndarray | None:
    """Return per-frame axis lengths when all frames are periodic with a usable cell."""
    if not frames:
        return None
    if not all(_frame_has_usable_cell(frame, axis_index) for frame in frames):
        return None
    if not all(_axis_is_periodic(frame, axis_index) for frame in frames):
        return None
    return np.asarray(
        [
            np.linalg.norm(np.asarray(frame.cell.array, dtype=float)[axis_index])
            for frame in frames
        ],
        dtype=float,
    )


def _fixed_atom_mask(frame: Atoms) -> np.ndarray:
    """Return a boolean mask of atoms constrained by index-based ASE constraints."""
    n_atoms = len(frame)
    if n_atoms == 0:
        return np.array([], dtype=bool)

    mask = np.zeros(n_atoms, dtype=bool)
    for constraint in getattr(frame, "constraints", ()) or ():
        get_indices = getattr(constraint, "get_indices", None)
        if get_indices is None:
            continue
        try:
            indices = np.asarray(get_indices(), dtype=int).ravel()
        except Exception:  # pragma: no cover - defensive against third-party constraints.
            continue
        if indices.size == 0:
            continue
        valid = indices[(indices >= 0) & (indices < n_atoms)]
        if valid.size == 0:
            continue
        mask[valid] = True
    return mask


def _normalize_element_symbol(symbol: str) -> str:
    """Normalize and validate an element symbol."""
    token = symbol.strip()
    if not token:
        raise ValueError("Surface element labels cannot be empty.")
    normalized = token[0].upper() + token[1:].lower()
    if normalized not in atomic_numbers:
        raise ValueError(f"Unknown element symbol '{symbol}'.")
    return normalized


def _normalize_surface_elements_argument(
    surface_elements: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Normalize user-provided surface elements while preserving order."""
    if surface_elements is None:
        return None

    tokens: list[str] = []
    for raw in surface_elements:
        for part in str(raw).split(","):
            stripped = part.strip()
            if stripped:
                tokens.append(_normalize_element_symbol(stripped))

    if not tokens:
        return None

    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _resolve_surface_elements(
    frames: list[Atoms],
    surface_elements: list[str] | tuple[str, ...] | None,
    *,
    axis_index: int,
) -> tuple[list[str], str]:
    """Resolve surface reference elements from override or automatic detection."""
    if not frames:
        return [], "none"

    symbols = np.asarray(frames[0].get_chemical_symbols(), dtype=object)
    available = set(str(symbol) for symbol in symbols.tolist())

    normalized_override = _normalize_surface_elements_argument(surface_elements)
    if normalized_override is not None:
        missing = [element for element in normalized_override if element not in available]
        if missing:
            raise ValueError(
                "Surface element override references symbols not present in trajectory: "
                + ", ".join(missing)
            )
        return normalized_override, "user"

    non_h_symbols = symbols[symbols != "H"]
    source_symbols = non_h_symbols if non_h_symbols.size > 0 else symbols
    unique, counts = np.unique(source_symbols, return_counts=True)
    if unique.size == 0:
        return [], "auto"

    count_by_symbol = {str(symbol): int(count) for symbol, count in zip(unique, counts)}
    total = int(sum(count_by_symbol.values()))
    abundance_floor = max(2, int(np.ceil(0.03 * total)))

    # Prefer low-mobility elements when atom ordering is stable across frames.
    if len(frames) > 1 and _frames_share_consistent_atom_layout(frames):
        symbol_matrix = np.asarray(frames[0].get_chemical_symbols(), dtype=object)
        source_mask = symbol_matrix != "H"
        if not np.any(source_mask):
            source_mask = np.ones(symbol_matrix.size, dtype=bool)
        axis_matrix = np.stack(
            [
                np.asarray(frame.positions[:, axis_index], dtype=float)[source_mask]
                for frame in frames
            ],
            axis=0,
        )
        axis_lengths = _axis_lengths_if_periodic(frames, axis_index)
        unwrapped_axis = _unwrap_axis_positions(axis_matrix, axis_lengths)
        medians = np.median(unwrapped_axis, axis=0)
        mobility = np.median(np.abs(unwrapped_axis - medians), axis=0)
        if not np.all(np.isfinite(mobility)):
            mobility = np.nan_to_num(mobility, nan=np.inf)

        symbols_for_mobility = symbol_matrix[source_mask]
        mobility_by_symbol: dict[str, float] = {}
        for symbol in np.unique(symbols_for_mobility):
            symbol_mask = symbols_for_mobility == symbol
            if not np.any(symbol_mask):
                continue
            value = float(np.median(mobility[symbol_mask]))
            if np.isfinite(value):
                mobility_by_symbol[str(symbol)] = value

        if mobility_by_symbol:
            mobility_values = np.asarray(list(mobility_by_symbol.values()), dtype=float)
            if np.ptp(mobility_values) > SURFACE_MOBILITY_EPSILON:
                min_mobility = float(np.min(mobility_values))
                mobility_limit = max(min_mobility + 0.02, min_mobility * 2.5)
                candidates = [
                    symbol
                    for symbol, value in mobility_by_symbol.items()
                    if value <= mobility_limit and count_by_symbol.get(symbol, 0) >= abundance_floor
                ]
                if not candidates:
                    candidates = [min(mobility_by_symbol, key=mobility_by_symbol.get)]

                candidates = sorted(
                    candidates,
                    key=lambda symbol: (
                        mobility_by_symbol[symbol],
                        -count_by_symbol.get(symbol, 0),
                        -float(atomic_masses[atomic_numbers[symbol]]),
                        symbol,
                    ),
                )
                return candidates[:4], "auto"

    abundant_symbols = [
        symbol for symbol, count in count_by_symbol.items() if count >= max(2, int(np.ceil(0.08 * total)))
    ]
    if not abundant_symbols:
        abundant_symbols = [max(count_by_symbol, key=count_by_symbol.get)]

    max_mass = max(float(atomic_masses[atomic_numbers[symbol]]) for symbol in abundant_symbols)
    mass_floor = 0.25 * max_mass
    selected = [
        symbol
        for symbol in abundant_symbols
        if float(atomic_masses[atomic_numbers[symbol]]) >= mass_floor
    ]
    if not selected:
        selected = [max(abundant_symbols, key=lambda symbol: count_by_symbol[symbol])]

    selected = sorted(
        selected,
        key=lambda symbol: (
            -count_by_symbol[symbol],
            -float(atomic_masses[atomic_numbers[symbol]]),
            symbol,
        ),
    )
    return selected[:4], "auto"


def _surface_axis_values_for_frame(
    frame: Atoms,
    *,
    axis_index: int,
    surface_elements: set[str],
    include_fixed_surface_atoms: bool = True,
) -> np.ndarray:
    """Return axis coordinates for the configured surface reference elements."""
    if not surface_elements:
        return np.array([], dtype=float)
    symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
    mask = np.isin(symbols, list(surface_elements))
    if not include_fixed_surface_atoms:
        mask &= ~_fixed_atom_mask(frame)
    if not np.any(mask):
        return np.array([], dtype=float)
    return np.asarray(frame.positions[mask, axis_index], dtype=float)


def _surface_quantile_per_frame(
    frames: list[Atoms],
    *,
    axis_index: int,
    surface_elements: set[str],
    include_fixed_surface_atoms: bool,
) -> np.ndarray:
    """Return per-frame high-quantile surface coordinates for fallback filling."""
    quantiles = np.full(len(frames), np.nan, dtype=float)
    for frame_index, frame in enumerate(frames):
        axis_values = _surface_axis_values_for_frame(
            frame,
            axis_index=axis_index,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
        )
        if axis_values.size == 0:
            continue
        quantiles[frame_index] = float(np.quantile(axis_values, SURFACE_POSITION_QUANTILE))
    return quantiles


def _extract_top_layer(
    sorted_axis_values: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    """Extract top-layer values based on significant gaps in sorted axis positions."""
    if sorted_axis_values.size < 2:
        return None, 1

    diffs = np.diff(sorted_axis_values)
    small_half = max(1, diffs.size // 2)
    baseline = float(np.median(np.partition(diffs, small_half - 1)[:small_half]))
    gap_threshold = max(SURFACE_LAYER_GAP_MIN, SURFACE_LAYER_GAP_FACTOR * baseline)
    significant_gaps = np.where(diffs >= gap_threshold)[0]

    if significant_gaps.size == 0:
        largest_gap_index = int(np.argmax(diffs))
        largest_gap = float(diffs[largest_gap_index])
        if largest_gap >= 2.0 * SURFACE_LAYER_GAP_MIN:
            return sorted_axis_values[largest_gap_index + 1 :], 2
        return None, 1

    top_start = int(significant_gaps[-1] + 1)
    layer_count = int(significant_gaps.size + 1)
    return sorted_axis_values[top_start:], layer_count


def _estimate_surface_position_layered(
    frames: list[Atoms],
    *,
    axis_index: int,
    surface_elements: set[str],
    include_fixed_surface_atoms: bool = True,
) -> _SurfaceEstimate | None:
    """Estimate surface via mean z of the top detected layer in each frame."""
    surface_per_frame = np.full(len(frames), np.nan, dtype=float)
    layer_counts: list[int] = []

    for frame_index, frame in enumerate(frames):
        axis_values = _surface_axis_values_for_frame(
            frame,
            axis_index=axis_index,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
        )
        if axis_values.size < 4:
            continue
        sorted_axis_values = np.sort(axis_values)
        top_layer, layer_count = _extract_top_layer(sorted_axis_values)
        if top_layer is None or top_layer.size == 0:
            continue
        minimum_top_layer_size = max(2, int(np.ceil(0.03 * sorted_axis_values.size)))
        if top_layer.size < minimum_top_layer_size:
            continue
        surface_per_frame[frame_index] = float(np.mean(top_layer))
        layer_counts.append(layer_count)

    valid_mask = np.isfinite(surface_per_frame)
    if not np.any(valid_mask):
        return None

    success_ratio = float(np.count_nonzero(valid_mask)) / len(frames)
    if success_ratio < SURFACE_LAYER_MIN_SUCCESS_RATIO:
        return None

    top_surface_array = surface_per_frame[valid_mask]
    median_layers = int(np.median(np.asarray(layer_counts, dtype=float)))
    method = f"layered_top_layer_mean(median_layers={median_layers})"
    return _SurfaceEstimate(
        position=float(np.median(top_surface_array)),
        std=float(np.std(top_surface_array, ddof=0)),
        method=method,
        success_ratio=success_ratio,
        per_frame=surface_per_frame,
    )


def _estimate_surface_position_rough(
    frames: list[Atoms],
    *,
    axis_index: int,
    surface_elements: set[str],
    include_fixed_surface_atoms: bool = False,
) -> _SurfaceEstimate | None:
    """Estimate surface from low-mobility atoms with a quantile fallback."""
    if _frames_share_consistent_atom_layout(frames):
        symbols = np.asarray(frames[0].get_chemical_symbols(), dtype=object)
        mask = np.isin(symbols, list(surface_elements))
        if not include_fixed_surface_atoms:
            fixed_mask = np.zeros(symbols.size, dtype=bool)
            for frame in frames:
                fixed_mask |= _fixed_atom_mask(frame)
            mask &= ~fixed_mask
        if np.any(mask):
            axis_matrix = np.stack(
                [
                    np.asarray(frame.positions[:, axis_index], dtype=float)[mask]
                    for frame in frames
                ],
                axis=0,
            )
            masses = np.asarray(frames[0].get_masses(), dtype=float)[mask]
            axis_lengths = _axis_lengths_if_periodic(frames, axis_index)
            unwrapped_axis = _unwrap_axis_positions(axis_matrix, axis_lengths)
            medians = np.median(unwrapped_axis, axis=0)
            mobility = np.median(np.abs(unwrapped_axis - medians), axis=0)
            if not np.all(np.isfinite(mobility)):
                mobility = np.nan_to_num(mobility, nan=np.inf)
            if not include_fixed_surface_atoms:
                non_fixed_by_mobility = mobility > SURFACE_MOBILITY_EPSILON
                if np.any(non_fixed_by_mobility):
                    axis_matrix = axis_matrix[:, non_fixed_by_mobility]
                    masses = masses[non_fixed_by_mobility]
                    mobility = mobility[non_fixed_by_mobility]

            atom_count = mobility.size
            minimum_reference = min(
                MIN_SURFACE_REFERENCE_ATOMS,
                max(3, atom_count // 2),
            )
            reference_count = max(
                minimum_reference,
                int(np.ceil(SURFACE_MOBILITY_FRACTION * atom_count)),
            )
            reference_count = min(reference_count, atom_count)
            if reference_count > 0:
                mobility_rank = np.argsort(np.argsort(mobility))
                heavy_rank = np.argsort(np.argsort(-masses))
                combined_score = mobility_rank.astype(float) + 0.35 * heavy_rank.astype(float)
                reference_indices = np.argpartition(combined_score, reference_count - 1)[
                    :reference_count
                ]
                reference_indices.sort()
                surface_per_frame = np.mean(axis_matrix[:, reference_indices], axis=1)
                surface_array = np.asarray(surface_per_frame, dtype=float)
                return _SurfaceEstimate(
                    position=float(np.median(surface_array)),
                    std=float(np.std(surface_array, ddof=0)),
                    method=(
                        f"rough_low_mobility({int(SURFACE_MOBILITY_FRACTION * 100)}%)"
                        "+frame_mean"
                    ),
                    success_ratio=1.0,
                    per_frame=surface_array,
                )

    fallback_surface_per_frame = _surface_quantile_per_frame(
        frames,
        axis_index=axis_index,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )
    valid_mask = np.isfinite(fallback_surface_per_frame)
    if not np.any(valid_mask):
        return None

    surface_array = fallback_surface_per_frame[valid_mask]
    return _SurfaceEstimate(
        position=float(np.median(surface_array)),
        std=float(np.std(surface_array, ddof=0)),
        method=f"rough_axis_quantile:q{int(SURFACE_POSITION_QUANTILE * 100)}",
        success_ratio=float(np.count_nonzero(valid_mask)) / len(frames),
        per_frame=fallback_surface_per_frame,
    )


def _fill_missing_surface_per_frame(
    estimate: _SurfaceEstimate,
    *,
    fallback_per_frame: np.ndarray,
    fallback_label: str,
) -> _SurfaceEstimate:
    """Fill missing frame-wise surface values with a frame-local fallback estimate."""
    primary_per_frame = np.asarray(estimate.per_frame, dtype=float)
    if primary_per_frame.shape[0] == 0 or np.all(np.isfinite(primary_per_frame)):
        return estimate
    if fallback_per_frame.shape[0] != primary_per_frame.shape[0]:
        return estimate

    missing_mask = ~np.isfinite(primary_per_frame)
    fill_mask = missing_mask & np.isfinite(fallback_per_frame)
    if not np.any(fill_mask):
        return estimate

    merged = np.array(primary_per_frame, dtype=float, copy=True)
    merged[fill_mask] = fallback_per_frame[fill_mask]
    valid_mask = np.isfinite(merged)
    if not np.any(valid_mask):
        return estimate

    return _SurfaceEstimate(
        position=float(np.median(merged[valid_mask])),
        std=float(np.std(merged[valid_mask], ddof=0)),
        method=f"{estimate.method}+{fallback_label}_fill",
        success_ratio=float(np.count_nonzero(valid_mask)) / merged.size,
        per_frame=merged,
    )


def _select_surface_estimate(
    frames: list[Atoms],
    axis: str,
    *,
    mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
) -> tuple[_SurfaceEstimate | None, str]:
    if not frames:
        return None, "unavailable:no_frames"

    axis_index = axis_to_index(axis)
    if any(len(frame) == 0 for frame in frames):
        return None, "unavailable:empty_frame"

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"auto", "layered", "rough"}:
        raise ValueError("surface mode must be one of: auto, layered, rough")

    resolved_elements, element_source = _resolve_surface_elements(
        frames,
        surface_elements,
        axis_index=axis_index,
    )
    if not resolved_elements:
        return None, "unavailable:no_surface_elements"
    resolved_element_set = set(resolved_elements)

    LOGGER.info(
        "Surface reference elements (%s): %s",
        element_source,
        ", ".join(resolved_elements),
    )

    layered_estimate = _estimate_surface_position_layered(
        frames,
        axis_index=axis_index,
        surface_elements=resolved_element_set,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )
    rough_estimate = _estimate_surface_position_rough(
        frames,
        axis_index=axis_index,
        surface_elements=resolved_element_set,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )

    selected_estimate: _SurfaceEstimate | None = None
    if normalized_mode == "layered":
        if layered_estimate is not None:
            selected_estimate = layered_estimate
        elif rough_estimate is not None:
            LOGGER.warning(
                "Layered surface estimation failed; falling back to rough surface estimator."
            )
            selected_estimate = rough_estimate
        else:
            return None, "unavailable:layered_and_rough_failed"
    elif normalized_mode == "rough":
        if rough_estimate is not None:
            selected_estimate = rough_estimate
        elif layered_estimate is not None:
            LOGGER.warning(
                "Rough surface estimation failed; falling back to layered surface estimator."
            )
            selected_estimate = layered_estimate
        else:
            return None, "unavailable:rough_and_layered_failed"
    else:
        if layered_estimate is not None and (
            layered_estimate.success_ratio >= 0.75 or rough_estimate is None
        ):
            selected_estimate = layered_estimate
        elif rough_estimate is not None:
            selected_estimate = rough_estimate
        elif layered_estimate is not None:
            selected_estimate = layered_estimate
        else:
            return None, "unavailable:auto_failed"

    missing_before = int(np.count_nonzero(~np.isfinite(selected_estimate.per_frame)))
    if missing_before > 0:
        quantile_per_frame = _surface_quantile_per_frame(
            frames,
            axis_index=axis_index,
            surface_elements=resolved_element_set,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
        )
        selected_estimate = _fill_missing_surface_per_frame(
            selected_estimate,
            fallback_per_frame=quantile_per_frame,
            fallback_label=f"axis_quantile_q{int(SURFACE_POSITION_QUANTILE * 100)}",
        )
        missing_after = int(np.count_nonzero(~np.isfinite(selected_estimate.per_frame)))
        if missing_after < missing_before:
            LOGGER.info(
                "Filled %d frame-wise surface gaps with axis-quantile fallback values.",
                missing_before - missing_after,
            )

    return selected_estimate, selected_estimate.method


def estimate_surface_position(
    frames: list[Atoms],
    axis: str = "z",
    *,
    mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
) -> tuple[float | None, float | None, str]:
    """Estimate a representative surface position along ``axis``.

    ``auto`` mode prefers layered top-layer means when layering is clearly detected
    and falls back to a rough-surface low-mobility estimator otherwise.
    """
    estimate, method = _select_surface_estimate(
        frames,
        axis,
        mode=mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )
    if estimate is None:
        return None, None, method
    return estimate.position, estimate.std, method


def _shift_axis_values_by_surface_per_frame(
    selected_per_frame: list[np.ndarray],
    surface_per_frame: np.ndarray | None,
) -> tuple[list[np.ndarray], str]:
    if surface_per_frame is None:
        return selected_per_frame, "axis"
    if surface_per_frame.shape[0] != len(selected_per_frame):
        return selected_per_frame, "axis"
    if not np.all(np.isfinite(surface_per_frame)):
        return selected_per_frame, "axis"

    shifted = [
        (np.asarray(axis_values, dtype=float) - float(surface_value))
        if axis_values.size > 0
        else np.array([], dtype=float)
        for axis_values, surface_value in zip(selected_per_frame, surface_per_frame)
    ]
    return shifted, "distance"


def _select_axis_values_with_masses(
    frame: Atoms, species: str, axis_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and masses for atoms matching ``species`` in one frame."""
    symbols = np.asarray(frame.get_chemical_symbols())
    mask = symbols == species
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)
    axis_values = np.asarray(frame.positions[mask, axis_index], dtype=float)
    masses = np.asarray(frame.get_masses()[mask], dtype=float) * AMU_TO_G
    return axis_values, masses


def _normalize_species_query(species: str | None) -> tuple[str, str]:
    """Return normalized selection mode and label."""
    if species is None:
        return "all", "ALL"

    species = species.strip()
    if not species or species.lower() == "all" or species == "*":
        return "all", "ALL"
    if species.upper() == "H2O":
        return "h2o", "H2O"

    normalized = species[0].upper() + species[1:].lower()
    return "element", normalized


def available_element_species(frames: list[Atoms]) -> list[str]:
    """Return sorted unique element symbols found across all frames."""
    species_set: set[str] = set()
    for frame in frames:
        species_set.update(frame.get_chemical_symbols())
    return sorted(species_set)


def _select_water_axis_values_with_masses(
    frame: Atoms, axis_index: int, oh_cutoff: float = 1.25
) -> tuple[np.ndarray, np.ndarray]:
    """Return O-axis positions and molecular masses for detected water molecules."""
    oxygen_indices = _water_oxygen_indices(frame, oh_cutoff=oh_cutoff)
    if oxygen_indices.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    axis_values = np.asarray(frame.positions[oxygen_indices, axis_index], dtype=float)
    masses = np.full(oxygen_indices.size, H2O_MASS_G, dtype=float)
    return axis_values, masses


def _water_oxygen_indices(frame: Atoms, oh_cutoff: float = 1.25) -> np.ndarray:
    """Return oxygen indices classified as water oxygens (>=2 H neighbors within cutoff)."""
    pair_i, _ = neighbor_list("ij", frame, {("O", "H"): oh_cutoff})
    if pair_i.size == 0:
        return np.array([], dtype=int)
    hydrogen_neighbor_count = np.bincount(pair_i, minlength=len(frame))
    return np.where(hydrogen_neighbor_count >= 2)[0].astype(int, copy=False)


def _compute_density_profile_from_selected(
    *,
    frames: list[Atoms],
    selected_per_frame: list[np.ndarray],
    selected_masses_per_frame: list[np.ndarray],
    axis: str,
    axis_index: int,
    species_label: str,
    count_label: str,
    bin_width: float,
    coordinate_mode: str = "axis",
    surface_position: float | None = None,
    surface_position_std: float | None = None,
) -> DensityProfile:
    """Build a :class:`DensityProfile` from already-selected axis values."""
    n_selected_total = sum(values.size for values in selected_per_frame)
    if n_selected_total == 0:
        raise ValueError(f"No entities found for selection '{species_label}' in trajectory.")
    LOGGER.info(
        "Selected %d total %s across %d frame(s).",
        n_selected_total,
        count_label,
        len(frames),
    )

    non_empty_selected = [values for values in selected_per_frame if values.size > 0]
    data_min = min(float(np.min(values)) for values in non_empty_selected)
    data_max = max(float(np.max(values)) for values in non_empty_selected)

    if np.isclose(data_min, data_max):
        data_max = data_min + bin_width

    span = data_max - data_min
    n_bins = max(1, int(np.ceil(span / bin_width)))
    bin_edges = data_min + np.arange(n_bins + 1, dtype=float) * bin_width
    if bin_edges[-1] <= data_max:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    LOGGER.debug(
        "Histogram bounds: min=%.6g, max=%.6g, bins=%d.",
        data_min,
        data_max,
        bin_edges.size - 1,
    )

    n_frames = len(frames)
    masses_accum = np.zeros(bin_edges.size - 1, dtype=float)
    entities_accum = np.zeros(bin_edges.size - 1, dtype=float)
    use_volumetric_density = all(_frame_has_usable_cell(frame, axis_index) for frame in frames)
    LOGGER.info("Density mode: %s.", "volumetric" if use_volumetric_density else "linear")

    slice_volumes: np.ndarray | None = None
    variable_slice_volume = False
    if use_volumetric_density:
        slice_volumes = np.empty(n_frames, dtype=float)
        for i, frame in enumerate(frames):
            cell = np.asarray(frame.cell.array, dtype=float)
            axis_length = np.linalg.norm(cell[axis_index])
            volume = abs(float(np.linalg.det(cell)))
            cross_section = volume / axis_length
            slice_volumes[i] = cross_section * bin_width

        variable_slice_volume = not np.allclose(
            slice_volumes, slice_volumes[0], rtol=0.0, atol=1e-12
        )

    density_accum = np.zeros_like(masses_accum) if variable_slice_volume else None
    with ProgressBar(
        desc=f"Binning {species_label} density", total=n_frames, unit="frame"
    ) as progress:
        for frame_index, (axis_values, masses) in enumerate(
            zip(selected_per_frame, selected_masses_per_frame)
        ):
            per_frame_mass, _ = np.histogram(axis_values, bins=bin_edges, weights=masses)
            per_frame_mass = per_frame_mass.astype(float)
            masses_accum += per_frame_mass
            per_frame_entities, _ = np.histogram(axis_values, bins=bin_edges)
            entities_accum += per_frame_entities.astype(float)
            if density_accum is not None and slice_volumes is not None:
                density_accum += per_frame_mass / slice_volumes[frame_index]
            progress.update()

    if use_volumetric_density and slice_volumes is not None:
        if density_accum is None:
            density = (masses_accum / slice_volumes[0]) / n_frames
        else:
            density = density_accum / n_frames
        units = "g/Angstrom^3"
    else:
        density = (masses_accum / n_frames) / bin_width
        units = "g/Angstrom"

    counts_per_frame = masses_accum / n_frames
    entities_per_frame = entities_accum / n_frames
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if use_volumetric_density and slice_volumes is not None:
        if variable_slice_volume:
            number_density_accum = np.zeros_like(entities_accum)
            for frame_index, axis_values in enumerate(selected_per_frame):
                per_frame_entities, _ = np.histogram(axis_values, bins=bin_edges)
                number_density_accum += per_frame_entities.astype(float) / slice_volumes[frame_index]
            number_density = number_density_accum / n_frames
        else:
            number_density = (entities_accum / n_frames) / slice_volumes[0]
        number_density_units = "atoms/Angstrom^3"
    else:
        number_density = entities_per_frame / bin_width
        number_density_units = "atoms/Angstrom"

    return DensityProfile(
        axis=axis.lower(),
        species=species_label,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_per_frame=counts_per_frame,
        density=density,
        units=units,
        n_frames=n_frames,
        entities_per_frame=entities_per_frame,
        number_density=number_density,
        number_density_units=number_density_units,
        coordinate_mode=coordinate_mode,
        surface_position=surface_position,
        surface_position_std=surface_position_std,
    )


def compute_density_profile(
    frames: list[Atoms],
    species: str | None = "all",
    axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
) -> DensityProfile:
    """Compute a 1D species mass-density profile along a Cartesian axis.

    Parameters
    ----------
    frames
        Trajectory frames.
    species
        Selection string: ``all`` (default), element symbol (for example: ``O``),
        or ``H2O`` for water-molecule density.
    axis
        Axis along which to compute density (`x`, `y`, or `z`).
    bin_width
        Histogram bin width in Angstrom.

    Returns
    -------
    DensityProfile
        Binned per-frame mass and mass-density values.
    """
    LOGGER.info(
        "Computing density profile (species=%s, axis=%s, bin_width=%.6g).",
        species,
        axis,
        bin_width,
    )
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("bin_width", bin_width)
    axis_index = axis_to_index(axis)
    surface_estimate, surface_method = _select_surface_estimate(
        frames,
        axis,
        mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    if surface_position is None:
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            axis.lower(),
            axis.lower(),
        )
    else:
        LOGGER.info(
            "Estimated %s-surface at %.6g Angstrom (std=%.4g; method=%s).",
            axis.lower(),
            surface_position,
            0.0 if surface_position_std is None else surface_position_std,
            surface_method,
        )
    selection_mode, species_label = _normalize_species_query(species)
    count_label = "molecules" if selection_mode == "h2o" else "atoms"
    LOGGER.debug("Selection mode: %s (label=%s).", selection_mode, species_label)

    selected_per_frame = []
    selected_masses_per_frame = []
    reference_water_oxygen_indices: np.ndarray | None = None
    use_static_water_indices = selection_mode == "h2o"
    with ProgressBar(
        desc=f"Selecting {species_label} for density", total=len(frames), unit="frame"
    ) as progress:
        for frame_index, frame in enumerate(frames):
            if selection_mode == "all":
                axis_values = np.asarray(frame.positions[:, axis_index], dtype=float)
                masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G
            elif selection_mode == "h2o":
                if use_static_water_indices:
                    dynamic_indices = None
                    if reference_water_oxygen_indices is None:
                        dynamic_indices = _water_oxygen_indices(frame)
                        reference_water_oxygen_indices = dynamic_indices
                    elif frame_index % H2O_VALIDATION_STRIDE == 0:
                        dynamic_indices = _water_oxygen_indices(frame)
                        if not np.array_equal(dynamic_indices, reference_water_oxygen_indices):
                            LOGGER.warning(
                                "Detected H2O topology change at frame %d; "
                                "switching from cached water indices to per-frame detection.",
                                frame_index,
                            )
                            use_static_water_indices = False

                    if use_static_water_indices:
                        indices = reference_water_oxygen_indices
                    else:
                        indices = (
                            dynamic_indices
                            if dynamic_indices is not None
                            else _water_oxygen_indices(frame)
                        )
                    axis_values = np.asarray(frame.positions[indices, axis_index], dtype=float)
                    masses = np.full(indices.size, H2O_MASS_G, dtype=float)
                else:
                    axis_values, masses = _select_water_axis_values_with_masses(frame, axis_index)
            else:
                axis_values, masses = _select_axis_values_with_masses(
                    frame, species_label, axis_index
                )
            selected_per_frame.append(axis_values)
            selected_masses_per_frame.append(masses)
            progress.update()
    selected_for_binning, coordinate_mode = _shift_axis_values_by_surface_per_frame(
        selected_per_frame,
        None if surface_estimate is None else surface_estimate.per_frame,
    )
    profile_surface_position = surface_position if coordinate_mode == "distance" else None
    profile_surface_position_std = surface_position_std if coordinate_mode == "distance" else None
    if surface_position is not None and coordinate_mode != "distance":
        LOGGER.warning(
            "Surface was estimated, but frame-wise surface alignment was unavailable; "
            "density bins remain on raw %s coordinates.",
            axis.lower(),
        )
    return _compute_density_profile_from_selected(
        frames=frames,
        selected_per_frame=selected_for_binning,
        selected_masses_per_frame=selected_masses_per_frame,
        axis=axis,
        axis_index=axis_index,
        species_label=species_label,
        count_label=count_label,
        bin_width=bin_width,
        coordinate_mode=coordinate_mode,
        surface_position=profile_surface_position,
        surface_position_std=profile_surface_position_std,
    )


def compute_density_profiles(
    frames: list[Atoms],
    species: str | None = "all",
    axis: str = "z",
    bin_width: float = 0.1,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
) -> list[DensityProfile]:
    """Compute one or more density profiles based on the species selection policy."""
    ensure_positive("bin_width", bin_width)
    selection_mode, _ = _normalize_species_query(species)
    if selection_mode != "all":
        return [
            compute_density_profile(
                frames=frames,
                species=species,
                axis=axis,
                bin_width=bin_width,
                surface_mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
            )
        ]

    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    surface_estimate, surface_method = _select_surface_estimate(
        frames,
        axis,
        mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )
    surface_position = None if surface_estimate is None else surface_estimate.position
    surface_position_std = None if surface_estimate is None else surface_estimate.std
    if surface_position is None:
        LOGGER.warning(
            "Could not estimate a surface position along %s; distance-to-surface plotting will "
            "fall back to raw %s coordinates.",
            axis.lower(),
            axis.lower(),
        )
    else:
        LOGGER.info(
            "Estimated %s-surface at %.6g Angstrom (std=%.4g; method=%s).",
            axis.lower(),
            surface_position,
            0.0 if surface_position_std is None else surface_position_std,
            surface_method,
        )
    if selection_mode == "all":
        element_species = available_element_species(frames)
        if not element_species:
            raise ValueError("No elements found in trajectory.")

        axis_index = axis_to_index(axis)
        selected_by_species: dict[str, list[np.ndarray]] = {
            element: [] for element in element_species
        }
        selected_masses_by_species: dict[str, list[np.ndarray]] = {
            element: [] for element in element_species
        }
        empty = np.array([], dtype=float)
        with ProgressBar(
            desc="Selecting element data for density", total=len(frames), unit="frame"
        ) as progress:
            for frame in frames:
                symbols = np.asarray(frame.get_chemical_symbols())
                axis_values = np.asarray(frame.positions[:, axis_index], dtype=float)
                masses = np.asarray(frame.get_masses(), dtype=float) * AMU_TO_G

                frame_selected: dict[str, np.ndarray] = {}
                frame_selected_masses: dict[str, np.ndarray] = {}
                for symbol in np.unique(symbols):
                    mask = symbols == symbol
                    frame_selected[str(symbol)] = axis_values[mask]
                    frame_selected_masses[str(symbol)] = masses[mask]

                for element in element_species:
                    selected_by_species[element].append(frame_selected.get(element, empty))
                    selected_masses_by_species[element].append(
                        frame_selected_masses.get(element, empty)
                    )
                progress.update()

        profiles: list[DensityProfile] = []
        per_frame_surface = None if surface_estimate is None else surface_estimate.per_frame
        with ProgressBar(
            desc="Computing element-resolved densities", total=len(element_species), unit="species"
        ) as progress:
            for element in element_species:
                selected_for_binning, coordinate_mode = _shift_axis_values_by_surface_per_frame(
                    selected_by_species[element],
                    per_frame_surface,
                )
                profile_surface_position = surface_position if coordinate_mode == "distance" else None
                profile_surface_position_std = (
                    surface_position_std if coordinate_mode == "distance" else None
                )
                profiles.append(
                    _compute_density_profile_from_selected(
                        frames=frames,
                        selected_per_frame=selected_for_binning,
                        selected_masses_per_frame=selected_masses_by_species[element],
                        axis=axis,
                        axis_index=axis_index,
                        species_label=element,
                        count_label="atoms",
                        bin_width=bin_width,
                        coordinate_mode=coordinate_mode,
                        surface_position=profile_surface_position,
                        surface_position_std=profile_surface_position_std,
                    )
                )
                progress.update()
        return profiles


def _uniform_bin_width_from_edges(
    bin_edges: np.ndarray, *, source_label: str
) -> float:
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


def _resolve_bin_width_for_load(
    *,
    metadata: dict[str, object],
    bin_centers: np.ndarray,
    source_path: Path,
) -> float:
    """Resolve bin width from metadata or infer from equally spaced bin centers."""
    raw = metadata.get("bin_width_A")
    if raw is not None:
        try:
            width = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Density HDF5 '{source_path}' has invalid metadata value bin_width_A={raw!r}."
            ) from exc
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(
                f"Density HDF5 '{source_path}' has non-positive bin_width_A={raw!r}."
            )
        if bin_centers.size > 1:
            center_steps = np.diff(bin_centers)
            if not np.allclose(center_steps, width, rtol=1.0e-6, atol=1.0e-9):
                raise ValueError(
                    f"Density HDF5 '{source_path}' has inconsistent bin_centers_A and bin_width_A."
                )
        return width

    if bin_centers.size <= 1:
        raise ValueError(
            f"Density HDF5 '{source_path}' is missing bin_edges_A and bin_width_A; "
            "cannot reconstruct single-bin edges."
        )
    center_steps = np.diff(bin_centers)
    if not np.all(np.isfinite(center_steps)) or np.any(center_steps <= 0.0):
        raise ValueError(
            f"Density HDF5 '{source_path}' has invalid bin_centers_A spacing."
        )
    inferred = float(center_steps[0])
    if not np.allclose(center_steps, inferred, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError(
            f"Density HDF5 '{source_path}' is missing bin_edges_A/bin_width_A and has non-uniform "
            "bin_centers_A spacing."
        )
    return inferred


def _reconstruct_bin_edges_from_centers(bin_centers: np.ndarray, *, bin_width: float) -> np.ndarray:
    """Reconstruct edge coordinates from bin centers and a uniform bin width."""
    if bin_centers.ndim != 1 or bin_centers.size == 0:
        raise ValueError("Cannot reconstruct bin edges from empty or non-1D bin centers.")
    left_edge = float(bin_centers[0]) - 0.5 * bin_width
    return left_edge + np.arange(bin_centers.size + 1, dtype=float) * bin_width


def save_density_profile(
    profile: DensityProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a density profile to LiNaK HDF5 and return the written path."""
    bin_edges = np.asarray(profile.bin_edges, dtype=float)
    bin_centers = np.asarray(profile.bin_centers, dtype=float)
    if bin_edges.size != bin_centers.size + 1:
        raise ValueError(
            "Density profile bin_edges and bin_centers sizes are inconsistent "
            f"(edges={bin_edges.size}, centers={bin_centers.size})."
        )
    expected_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if not np.allclose(expected_centers, bin_centers, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("Density profile bin_centers are inconsistent with bin_edges.")
    bin_width = _uniform_bin_width_from_edges(
        bin_edges,
        source_label=f"Density profile '{profile.species}'",
    )

    metadata: dict[str, Any] = {
        "axis": profile.axis,
        "species": profile.species,
        "units": profile.units,
        "n_frames": profile.n_frames,
        "number_density_units": profile.number_density_units,
        "coordinate_mode": profile.coordinate_mode,
        "surface_position": profile.surface_position,
        "surface_position_std": profile.surface_position_std,
        "bin_width_A": bin_width,
        "units_map": {
            "bin_width_A": "Angstrom",
            "bin_centers_A": "Angstrom",
            "counts_per_frame": "g",
            "density": profile.units,
            "number_density": profile.number_density_units,
        },
    }
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="density",
        datasets={
            "bin_centers_A": profile.bin_centers,
            "counts_per_frame": profile.counts_per_frame,
            "density": profile.density,
            "number_density": profile.number_density,
        },
        metadata=metadata,
    )
    LOGGER.info("Saved density data to '%s'.", output_path)
    return output_path


def load_density_profile(
    path: str | Path,
    *,
    axis: str | None = None,
    species: str | None = None,
) -> DensityProfile:
    """Load one density profile from LiNaK HDF5.

    For profile-collection files, this returns the first profile.
    """
    profiles = load_density_profiles(path, axis=axis, species=species)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(f"Density HDF5 '{source_path}' does not contain any density profiles.")
    return profiles[0]


def load_density_profiles(
    path: str | Path,
    *,
    axis: str | None = None,
    species: str | None = None,
) -> list[DensityProfile]:
    """Load one or more density profiles from LiNaK HDF5."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Density profile not found: {source_path}")

    if is_hdf5_path(source_path):
        payloads = read_linak_hdf5_profiles(source_path, expected_analysis="density")
        profiles: list[DensityProfile] = []
        for datasets, metadata in payloads:
            required = ("bin_centers_A", "counts_per_frame", "density")
            missing = [name for name in required if name not in datasets]
            if missing:
                raise ValueError(
                    f"Density HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
                )

            axis_meta = str(metadata.get("axis", "")).strip()
            if axis is not None and axis.strip():
                axis_label = "xyz"[axis_to_index(axis)]
            elif axis_meta:
                axis_label = "xyz"[axis_to_index(axis_meta)]
            else:
                axis_label = "z"

            species_meta = str(metadata.get("species", "")).strip()
            if species is not None and species.strip():
                _selection_mode, species_label = _normalize_species_query(species)
            elif species_meta:
                species_label = species_meta
            else:
                species_label = "UNKNOWN"

            units = str(metadata.get("units", "g/cm^3"))
            number_density_units_raw = metadata.get("number_density_units")
            number_density_units = (
                str(number_density_units_raw) if number_density_units_raw is not None else None
            )
            coordinate_mode = str(metadata.get("coordinate_mode", "axis")).strip().lower()
            if coordinate_mode not in {"axis", "distance"}:
                coordinate_mode = "axis"

            surface_position_raw = metadata.get("surface_position")
            surface_position = None
            if surface_position_raw is not None:
                value = float(surface_position_raw)
                if np.isfinite(value):
                    surface_position = value

            surface_std_raw = metadata.get("surface_position_std")
            surface_position_std = None
            if surface_std_raw is not None:
                value = float(surface_std_raw)
                if np.isfinite(value):
                    surface_position_std = value

            number_density = None
            if "number_density" in datasets:
                number_density = np.asarray(datasets["number_density"], dtype=float)

            entities_per_frame = None
            if "entities_per_frame" in datasets:
                entities_per_frame = np.asarray(datasets["entities_per_frame"], dtype=float)

            bin_centers = np.asarray(datasets["bin_centers_A"], dtype=float)
            if "bin_edges_A" in datasets:
                bin_edges = np.asarray(datasets["bin_edges_A"], dtype=float)
            else:
                bin_width = _resolve_bin_width_for_load(
                    metadata=metadata,
                    bin_centers=bin_centers,
                    source_path=source_path,
                )
                bin_edges = _reconstruct_bin_edges_from_centers(bin_centers, bin_width=bin_width)
            if bin_edges.size != bin_centers.size + 1:
                raise ValueError(
                    f"Density HDF5 '{source_path}' has incompatible bin_edges_A/bin_centers_A sizes."
                )

            profiles.append(
                DensityProfile(
                    axis=axis_label,
                    species=species_label,
                    bin_edges=bin_edges,
                    bin_centers=bin_centers,
                    counts_per_frame=np.asarray(datasets["counts_per_frame"], dtype=float),
                    density=np.asarray(datasets["density"], dtype=float),
                    units=units,
                    n_frames=int(metadata.get("n_frames", 0)),
                    entities_per_frame=entities_per_frame,
                    number_density=number_density,
                    number_density_units=number_density_units,
                    coordinate_mode=coordinate_mode,
                    surface_position=surface_position,
                    surface_position_std=surface_position_std,
                )
            )
        return profiles

    raise ValueError(f"Unsupported density profile format for '{source_path}'. Use .h5/.hdf5.")


def _format_plot_density_units(units: str) -> str:
    return units.replace("Angstrom", "A")


def _profile_has_surface_reference(profile: DensityProfile) -> bool:
    if profile.coordinate_mode == "distance":
        return True
    return profile.surface_position is not None and np.isfinite(profile.surface_position)


def _density_x_data(
    profile: DensityProfile,
    *,
    x_mode: str,
) -> tuple[np.ndarray, str]:
    if x_mode == "axis":
        if profile.coordinate_mode == "distance":
            if profile.surface_position is not None and np.isfinite(profile.surface_position):
                return profile.bin_centers + float(profile.surface_position), f"{profile.axis.upper()} (A)"
            LOGGER.warning(
                "Density profile '%s' stores distance-aligned bins with no absolute surface "
                "offset; using distance coordinates.",
                profile.species,
            )
            return profile.bin_centers, "Distance to surface (A)"
        return profile.bin_centers, f"{profile.axis.upper()} (A)"

    if x_mode == "distance":
        if profile.coordinate_mode == "distance":
            return profile.bin_centers, "Distance to surface (A)"
        if _profile_has_surface_reference(profile):
            assert profile.surface_position is not None
            return (
                profile.bin_centers - float(profile.surface_position),
                "Distance to surface (A)",
            )
        LOGGER.warning(
            "Density profile '%s' has no surface reference; falling back to axis coordinates.",
            profile.species,
        )
        return profile.bin_centers, f"{profile.axis.upper()} (A)"

    raise ValueError(f"Unsupported density x_mode '{x_mode}'. Choose 'distance' or 'axis'.")


def _density_y_data(
    profile: DensityProfile,
    *,
    quantity: str,
) -> tuple[np.ndarray, str, str]:
    if quantity == "number":
        if profile.number_density is None or profile.number_density_units is None:
            raise ValueError(
                f"Number-density data unavailable for profile '{profile.species}'. "
                "Use quantity='mass' or recompute volumetric density."
            )
        return profile.number_density, profile.number_density_units, "Number density"

    if quantity == "mass":
        units = profile.units
        density_values = profile.density
        if units == "g/Angstrom^3":
            density_values = profile.density / ANGSTROM3_TO_CM3
            units = "g/cm^3"
        return density_values, units, "Mass density"

    raise ValueError(f"Unsupported density quantity '{quantity}'. Choose 'mass' or 'number'.")


def plot_density_profile(
    profile: DensityProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_mode: str = "distance",
    quantity: str = "mass",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float, float] | list[float] | None = None,
    y_lim: tuple[float, float] | list[float] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    line_label: str | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    capture_state: dict[str, Any] | None = None,
) -> Path | None:
    """Plot and optionally save a density profile."""
    x_values, default_x_label = _density_x_data(profile, x_mode=x_mode)
    density_values, units, y_label_prefix = _density_y_data(profile, quantity=quantity)
    units = _format_plot_density_units(units)
    resolved_line_label = line_label
    if resolved_line_label is None and legend:
        resolved_line_label = profile.species
    resolved_line_color = None
    if line_colors:
        resolved_line_color = line_colors[0]
    line_visible = True if not series_enabled else bool(series_enabled[0])
    line_width_override = None
    if series_line_widths:
        line_width_override = series_line_widths[0]
    line_marker = None
    if series_markers:
        line_marker = series_markers[0]

    return plot_line_series(
        x_values,
        density_values,
        title=title or f"{profile.species} density profile",
        x_label=x_label or default_x_label,
        y_label=y_label or f"{y_label_prefix} ({units})",
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        line_label=resolved_line_label,
        line_color=resolved_line_color,
        line_width_override=line_width_override,
        line_marker=line_marker,
        line_visible=line_visible,
        style=style,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
    )


def plot_density_profiles(
    profiles: list[DensityProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_mode: str = "distance",
    quantity: str = "mass",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float, float] | list[float] | None = None,
    y_lim: tuple[float, float] | list[float] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    capture_state: dict[str, Any] | None = None,
) -> Path | None:
    """Plot one or more density profiles."""
    if not profiles:
        raise ValueError("At least one density profile is required.")
    default_labels = [profile.species for profile in profiles]
    labels = default_labels
    if series_labels is not None:
        if len(series_labels) != len(default_labels):
            raise ValueError(
                "series_labels count must match the number of plotted density series "
                f"({len(default_labels)})."
            )
        labels = [label.strip() for label in series_labels]
        if any(not label for label in labels):
            raise ValueError("series_labels cannot contain empty values.")

    if len(profiles) == 1:
        return plot_density_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            x_mode=x_mode,
            quantity=quantity,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_scale=x_scale,
            y_scale=y_scale,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            line_label=labels[0] if labels else None,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            capture_state=capture_state,
        )

    first = profiles[0]
    if x_mode == "distance" and any(not _profile_has_surface_reference(profile) for profile in profiles):
        LOGGER.warning(
            "At least one profile has no surface reference; combined plot falls back to axis coordinates."
        )
        x_mode = "axis"

    y_resolved = [_density_y_data(profile, quantity=quantity) for profile in profiles]
    y_units = y_resolved[0][1]
    y_label_prefix = y_resolved[0][2]
    if any(units != y_units for _, units, _ in y_resolved[1:]):
        raise ValueError("All density profiles must use the same units for combined plotting.")
    y_series = [values for values, _, _ in y_resolved]
    x_series = [_density_x_data(profile, x_mode=x_mode)[0] for profile in profiles]
    default_x_label = _density_x_data(first, x_mode=x_mode)[1]
    display_units = _format_plot_density_units(y_units)

    return plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or "Element-resolved density profile",
        x_label=x_label or default_x_label,
        y_label=y_label or f"{y_label_prefix} ({display_units})",
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
    )
