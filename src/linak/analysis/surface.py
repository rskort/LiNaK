"""Shared surface-estimation helpers for surface-referenced analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from ..utils import axis_to_index, ensure_positive
from .common import (
    frame_has_usable_cell as _common_frame_has_usable_cell,
    optional_finite_float as _common_optional_finite_float,
)

LOGGER = logging.getLogger(__name__)
# Shared defaults for the conservative layered/rough surface heuristics.
# These are intentionally centralized here so surface-aware analyses stay aligned.
SURFACE_MOBILITY_FRACTION = 0.35
SURFACE_POSITION_QUANTILE = 0.90
MIN_SURFACE_REFERENCE_ATOMS = 6
SURFACE_LAYER_GAP_MIN = 0.25
SURFACE_LAYER_GAP_FACTOR = 3.0
SURFACE_LAYER_MIN_SUCCESS_RATIO = 0.60
SURFACE_MOBILITY_EPSILON = 1.0e-8
_SURFACE_PROVENANCE_MAXLEN = 48
_SURFACE_REASON_MAXLEN = 80


@dataclass(frozen=True)
class SurfaceEstimatorOptions:
    """Tunable parameters for LiNaK's shared surface estimator."""

    mode: str = "auto"
    side: Literal["top", "bottom"] = "top"
    reduction_mode: Literal["median", "trimmed_mean"] = "median"
    trim_fraction: float = 0.10
    gap_min_A: float = 0.25
    gap_factor: float = 3.0
    minimum_top_layer_fraction: float = 0.03
    minimum_top_layer_atoms: int = 2
    required_success_ratio: float = 0.60
    rough_reference_fraction: float = 0.35
    rough_reference_min_atoms: int = 3
    rough_reference_max_soft_cap: int = 6
    rough_surface_envelope_A: float | None = None
    rough_quantile: float = 0.90
    mass_tiebreak_weight: float = 0.10
    candidate_axis_span_max_fraction: float = 0.35
    layered_max_spread_A: float = 0.75
    rough_max_reference_spread_A: float = 1.50
    fill_max_gap: int = 2
    fill_neighbor_tolerance_A: float = 0.75
    jump_reject_tolerance_A: float = 1.50
    low_confidence_threshold: float = 0.55
    debug_diagnostics: bool = False
    surface_elements: tuple[str, ...] | None = None
    surface_atom_indices: tuple[int, ...] | None = None
    surface_atom_mask: np.ndarray | None = None
    include_fixed_surface_atoms: bool = False


@dataclass(frozen=True)
class SurfaceSummary:
    position: float | None
    std: float | None
    valid_fraction: float
    median_confidence: float
    method_label: str
    composite_score: float


@dataclass(frozen=True)
class SurfaceDiagnostics:
    candidate_count_per_frame: np.ndarray
    top_layer_size_per_frame: np.ndarray
    largest_gap_A_per_frame: np.ndarray
    baseline_gap_A_per_frame: np.ndarray
    reference_spread_A_per_frame: np.ndarray
    jump_rejection_mask: np.ndarray
    rejection_reason: np.ndarray
    effective_options: dict[str, Any]


@dataclass(frozen=True)
class SurfaceEstimate:
    frame_values: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    candidate_indices: np.ndarray | None
    selected_elements: tuple[str, ...]
    mode: str
    side: str
    summary: SurfaceSummary
    diagnostics: SurfaceDiagnostics

    @property
    def position(self) -> float | None:
        return self.summary.position

    @property
    def std(self) -> float | None:
        return self.summary.std

    @property
    def method(self) -> str:
        return self.summary.method_label

    @property
    def success_ratio(self) -> float:
        return self.summary.valid_fraction

    @property
    def per_frame(self) -> np.ndarray:
        return self.frame_values


@dataclass(frozen=True)
class _SurfaceAnalysisContext:
    frames: list[Atoms]
    axis_index: int
    axis: str
    options: SurfaceEstimatorOptions
    stable_layout: bool
    selected_elements: tuple[str, ...]
    candidate_indices: np.ndarray | None
    candidate_positions: np.ndarray | None
    candidate_axis_values: np.ndarray | None
    candidate_masses: np.ndarray | None
    translated_candidate_positions: np.ndarray | None
    cell_lengths: np.ndarray | None


def _surface_reference_atoms_label(reference_elements: list[str] | tuple[str, ...]) -> str:
    labels = [str(element).strip() for element in reference_elements if str(element).strip()]
    if not labels:
        return "reference atoms"
    return f"{', '.join(labels)} reference atoms"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _extract_surface_method_argument(method: str, *, key: str) -> str | None:
    marker = f"{key}="
    start = method.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = method.find(")", start)
    if end < 0:
        end = len(method)
    token = method[start:end].strip()
    return token or None


def _describe_surface_estimator(
    method: str,
    *,
    axis: str,
    reference_elements: list[str] | tuple[str, ...],
) -> str:
    axis_label = axis.upper()
    reference_label = _surface_reference_atoms_label(reference_elements)
    if method.startswith("layered_top_layer_"):
        median_layers = _extract_surface_method_argument(method, key="median_layers")
        detail = f" (median layer count {median_layers})" if median_layers is not None else ""
        reducer = method.removeprefix("layered_top_layer_").split("(", 1)[0].replace("_", " ")
        return f"layered top-layer {reducer} on {axis_label} using {reference_label}{detail}"
    if method == "rough_low_mobility_median":
        return f"low-mobility median on {axis_label} using {reference_label}"
    if method == "rough_low_mobility_trimmed_mean":
        return f"low-mobility trimmed mean on {axis_label} using {reference_label}"
    if method == "upper_reference_quantile":
        return f"upper reference quantile on {axis_label} using {reference_label}"
    if method == "lower_reference_quantile":
        return f"lower reference quantile on {axis_label} using {reference_label}"
    if method.startswith("layered_top_layer_mean"):
        median_layers = _extract_surface_method_argument(method, key="median_layers")
        detail = ""
        if median_layers is not None:
            detail = f" (median layer count {median_layers})"
        tracked_fill = _extract_surface_method_argument(method, key="n")
        tracked_detail = ""
        if "+tracked_top_layer_fill" in method and tracked_fill is not None:
            tracked_detail = f"; tracked top-layer mean reused for {tracked_fill} missing frame(s)"
        return (
            f"layered top-layer mean on {axis_label} using {reference_label}"
            f"{detail}{tracked_detail}"
        )
    if method.startswith("rough_low_mobility_mean"):
        fraction = _extract_surface_method_argument(method, key="fraction")
        detail = ""
        if fraction is not None:
            detail = f" (mobility fraction {fraction})"
        return f"low-mobility mean on {axis_label} using {reference_label}{detail}"
    if method.startswith("rough_axis_quantile:q"):
        quantile = method.split(":q", 1)[1].split("+", 1)[0].strip()
        return f"per-frame {axis_label} q{quantile} of {reference_label}"
    return method


def _log_framewise_surface_alignment(
    *,
    logger: logging.Logger,
    axis: str,
    surface_position: float,
    surface_position_std: float | None,
) -> None:
    axis_label = axis.upper()
    logger.info(
        "Frame-wise %s surface alignment active: LiNaK uses each atom's %s coordinate "
        "minus that frame's surface %s; summary of per-frame surface estimates: "
        "median=%.6g Angstrom, std=%.4g Angstrom.",
        axis_label,
        axis_label,
        axis_label,
        surface_position,
        0.0 if surface_position_std is None else surface_position_std,
    )


def _frame_has_usable_cell(frame: Atoms, axis_index: int) -> bool:
    """Check whether a frame has a finite non-zero cell for surface analysis."""
    return _common_frame_has_usable_cell(frame, axis_index=axis_index)


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


def _unwrap_axis_positions(axis_matrix: np.ndarray, axis_lengths: np.ndarray | None) -> np.ndarray:
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
        [np.linalg.norm(np.asarray(frame.cell.array, dtype=float)[axis_index]) for frame in frames],
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


def _default_surface_options() -> SurfaceEstimatorOptions:
    return SurfaceEstimatorOptions(
        gap_min_A=SURFACE_LAYER_GAP_MIN,
        gap_factor=SURFACE_LAYER_GAP_FACTOR,
        required_success_ratio=SURFACE_LAYER_MIN_SUCCESS_RATIO,
        rough_reference_fraction=SURFACE_MOBILITY_FRACTION,
        rough_quantile=SURFACE_POSITION_QUANTILE,
    )


def _normalize_surface_mode_token(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "layered", "rough"}:
        raise ValueError("surface mode must be one of: auto, layered, rough")
    return normalized


def _normalize_surface_side_token(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {"top", "bottom"}:
        raise ValueError("surface side must be one of: top, bottom")
    return normalized


def _validate_surface_options(
    options: SurfaceEstimatorOptions,
    *,
    atom_count: int | None,
) -> SurfaceEstimatorOptions:
    mode = _normalize_surface_mode_token(options.mode)
    side = _normalize_surface_side_token(options.side)
    reduction_mode = str(options.reduction_mode).strip().lower()
    if reduction_mode not in {"median", "trimmed_mean"}:
        raise ValueError("surface reduction_mode must be one of: median, trimmed_mean")

    trim_fraction = float(options.trim_fraction)
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("surface trim_fraction must be in [0, 0.5).")
    if options.minimum_top_layer_atoms < 1:
        raise ValueError("surface minimum_top_layer_atoms must be >= 1.")
    if not 0.0 < float(options.minimum_top_layer_fraction) <= 1.0:
        raise ValueError("surface minimum_top_layer_fraction must be in (0, 1].")
    if not 0.0 < float(options.required_success_ratio) <= 1.0:
        raise ValueError("surface required_success_ratio must be in (0, 1].")
    if not 0.0 < float(options.rough_reference_fraction) <= 1.0:
        raise ValueError("surface rough_reference_fraction must be in (0, 1].")
    if options.rough_reference_min_atoms < 1:
        raise ValueError("surface rough_reference_min_atoms must be >= 1.")
    if options.rough_reference_max_soft_cap < 1:
        raise ValueError("surface rough_reference_max_soft_cap must be >= 1.")
    rough_surface_envelope = (
        None
        if options.rough_surface_envelope_A is None
        else float(options.rough_surface_envelope_A)
    )
    if rough_surface_envelope is not None:
        ensure_positive("surface rough_surface_envelope_A", rough_surface_envelope)
    if not 0.0 < float(options.rough_quantile) < 1.0:
        raise ValueError("surface rough_quantile must be in (0, 1).")
    if not 0.0 <= float(options.mass_tiebreak_weight):
        raise ValueError("surface mass_tiebreak_weight must be >= 0.")
    if not 0.0 < float(options.candidate_axis_span_max_fraction) <= 1.0:
        raise ValueError("surface candidate_axis_span_max_fraction must be in (0, 1].")
    ensure_positive("surface gap_min_A", float(options.gap_min_A))
    ensure_positive("surface gap_factor", float(options.gap_factor))
    ensure_positive("surface layered_max_spread_A", float(options.layered_max_spread_A))
    ensure_positive(
        "surface rough_max_reference_spread_A", float(options.rough_max_reference_spread_A)
    )
    if int(options.fill_max_gap) < 0:
        raise ValueError("surface fill_max_gap must be >= 0.")
    ensure_positive("surface fill_neighbor_tolerance_A", float(options.fill_neighbor_tolerance_A))
    ensure_positive("surface jump_reject_tolerance_A", float(options.jump_reject_tolerance_A))
    if not 0.0 <= float(options.low_confidence_threshold) <= 1.0:
        raise ValueError("surface low_confidence_threshold must be in [0, 1].")

    normalized_elements = _normalize_surface_elements_argument(list(options.surface_elements or ()))
    atom_indices: tuple[int, ...] | None = None
    atom_mask: np.ndarray | None = None
    if options.surface_atom_indices is not None and options.surface_atom_mask is not None:
        raise ValueError("Provide either surface_atom_indices or surface_atom_mask, not both.")
    if options.surface_atom_indices is not None:
        if atom_count is None:
            raise ValueError("surface_atom_indices requires a trajectory with known atom count.")
        indices = np.asarray(options.surface_atom_indices, dtype=int).ravel()
        if indices.size == 0:
            atom_indices = tuple()
        else:
            if np.any(indices < 0) or np.any(indices >= int(atom_count)):
                raise ValueError("surface_atom_indices contains out-of-range atom indices.")
            atom_indices = tuple(int(value) for value in np.unique(indices))
    if options.surface_atom_mask is not None:
        if atom_count is None:
            raise ValueError("surface_atom_mask requires a trajectory with known atom count.")
        mask = np.asarray(options.surface_atom_mask, dtype=bool).ravel()
        if mask.size != int(atom_count):
            raise ValueError("surface_atom_mask must match the atom count of the trajectory.")
        atom_mask = np.asarray(mask, dtype=bool)

    return SurfaceEstimatorOptions(
        mode=mode,
        side=cast(Literal["top", "bottom"], side),
        reduction_mode=cast(Literal["median", "trimmed_mean"], reduction_mode),
        trim_fraction=trim_fraction,
        gap_min_A=float(options.gap_min_A),
        gap_factor=float(options.gap_factor),
        minimum_top_layer_fraction=float(options.minimum_top_layer_fraction),
        minimum_top_layer_atoms=int(options.minimum_top_layer_atoms),
        required_success_ratio=float(options.required_success_ratio),
        rough_reference_fraction=float(options.rough_reference_fraction),
        rough_reference_min_atoms=int(options.rough_reference_min_atoms),
        rough_reference_max_soft_cap=int(options.rough_reference_max_soft_cap),
        rough_surface_envelope_A=rough_surface_envelope,
        rough_quantile=float(options.rough_quantile),
        mass_tiebreak_weight=float(options.mass_tiebreak_weight),
        candidate_axis_span_max_fraction=float(options.candidate_axis_span_max_fraction),
        layered_max_spread_A=float(options.layered_max_spread_A),
        rough_max_reference_spread_A=float(options.rough_max_reference_spread_A),
        fill_max_gap=int(options.fill_max_gap),
        fill_neighbor_tolerance_A=float(options.fill_neighbor_tolerance_A),
        jump_reject_tolerance_A=float(options.jump_reject_tolerance_A),
        low_confidence_threshold=float(options.low_confidence_threshold),
        debug_diagnostics=bool(options.debug_diagnostics),
        surface_elements=None if normalized_elements is None else tuple(normalized_elements),
        surface_atom_indices=atom_indices,
        surface_atom_mask=atom_mask,
        include_fixed_surface_atoms=bool(options.include_fixed_surface_atoms),
    )


def _resolve_surface_options(
    *,
    frames: list[Atoms],
    mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
) -> SurfaceEstimatorOptions:
    base = _default_surface_options()
    atom_count = None if not frames else len(frames[0])
    if surface_options is None:
        merged = SurfaceEstimatorOptions(
            **{
                **asdict(base),
                "mode": mode,
                "surface_elements": None if surface_elements is None else tuple(surface_elements),
                "include_fixed_surface_atoms": include_fixed_surface_atoms,
            }
        )
        return _validate_surface_options(merged, atom_count=atom_count)

    if surface_elements is not None and surface_options.surface_elements is not None:
        normalized_simple = _normalize_surface_elements_argument(surface_elements) or []
        normalized_advanced = (
            _normalize_surface_elements_argument(surface_options.surface_elements) or []
        )
        if normalized_simple != normalized_advanced:
            raise ValueError("surface_elements conflicts with surface_options.surface_elements.")
    if bool(include_fixed_surface_atoms) != bool(surface_options.include_fixed_surface_atoms):
        raise ValueError(
            "include_fixed_surface_atoms conflicts with surface_options.include_fixed_surface_atoms."
        )
    if _normalize_surface_mode_token(mode) != _normalize_surface_mode_token(surface_options.mode):
        raise ValueError("surface_mode conflicts with surface_options.mode.")

    return _validate_surface_options(surface_options, atom_count=atom_count)


def _string_array(values: Sequence[str], *, maxlen: int) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=f"S{maxlen}")


def _decode_string_array(values: np.ndarray | Sequence[Any]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8", errors="replace")
    if array.dtype.kind == "O":
        return np.asarray([str(item) for item in array.tolist()], dtype=object)
    return array.astype(str)


def _optional_finite_float(value: Any) -> float | None:
    return _common_optional_finite_float(value)


def _robust_reduce(values: np.ndarray, options: SurfaceEstimatorOptions) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    if options.reduction_mode == "trimmed_mean":
        trim = int(np.floor(float(options.trim_fraction) * array.size))
        if trim > 0 and (array.size - 2 * trim) >= 1:
            sorted_values = np.sort(array)
            return float(np.mean(sorted_values[trim : array.size - trim]))
    return float(np.median(array))


def _surface_temporal_component(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tolerance_A: float,
) -> np.ndarray:
    component = np.zeros(values.shape, dtype=float)
    for index, value in enumerate(values):
        if not valid_mask[index] or not np.isfinite(value):
            continue
        neighbors: list[float] = []
        for offset in (-1, 1):
            cursor = index + offset
            while 0 <= cursor < values.size:
                if valid_mask[cursor] and np.isfinite(values[cursor]):
                    neighbors.append(float(values[cursor]))
                    break
                cursor += offset
        if not neighbors:
            component[index] = 0.5
            continue
        delta = max(abs(float(value) - neighbor) for neighbor in neighbors)
        component[index] = float(np.clip(1.0 - delta / max(tolerance_A, 1.0e-12), 0.0, 1.0))
    return component


def _surface_smoothness_score(
    values: np.ndarray, valid_mask: np.ndarray, *, tolerance_A: float
) -> float:
    valid_values = np.asarray(values[valid_mask], dtype=float)
    if valid_values.size <= 1:
        return 1.0 if valid_values.size == values.size and valid_values.size > 0 else 0.0
    deltas = np.abs(np.diff(valid_values))
    if deltas.size == 0:
        return 1.0
    return float(np.clip(1.0 - np.median(deltas) / max(tolerance_A, 1.0e-12), 0.0, 1.0))


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
                    candidates = [
                        min(
                            candidates or mobility_by_symbol,
                            key=lambda symbol: mobility_by_symbol[symbol],
                        )
                    ]

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
        symbol
        for symbol, count in count_by_symbol.items()
        if count >= max(2, int(np.ceil(0.08 * total)))
    ]
    if not abundant_symbols:
        abundant_symbols = [max(count_by_symbol, key=lambda symbol: count_by_symbol[symbol])]

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


def _nearest_valid_layer_indices(
    top_layer_indices_by_frame: list[np.ndarray | None],
    *,
    frame_index: int,
) -> np.ndarray | None:
    previous: np.ndarray | None = None
    next_: np.ndarray | None = None

    for candidate_index in range(frame_index - 1, -1, -1):
        candidate = top_layer_indices_by_frame[candidate_index]
        if candidate is not None and candidate.size > 0:
            previous = candidate
            break
    for candidate_index in range(frame_index + 1, len(top_layer_indices_by_frame)):
        candidate = top_layer_indices_by_frame[candidate_index]
        if candidate is not None and candidate.size > 0:
            next_ = candidate
            break

    if previous is None and next_ is None:
        return None
    if previous is None:
        return np.asarray(next_, dtype=int)
    if next_ is None:
        return np.asarray(previous, dtype=int)
    if np.array_equal(previous, next_):
        return np.asarray(previous, dtype=int)

    shared = np.intersect1d(previous, next_, assume_unique=False)
    if shared.size >= 2:
        return np.asarray(shared, dtype=int)
    return np.asarray(previous, dtype=int)


def _cell_lengths_if_periodic_all(frames: list[Atoms]) -> np.ndarray | None:
    if not frames:
        return None
    lengths: list[np.ndarray] = []
    for frame in frames:
        pbc = np.asarray(frame.get_pbc(), dtype=bool)
        if pbc.size != 3 or not bool(np.all(pbc)):
            return None
        cell_lengths = np.asarray(frame.cell.lengths(), dtype=float)
        if (
            cell_lengths.shape != (3,)
            or np.any(~np.isfinite(cell_lengths))
            or np.any(cell_lengths <= 0.0)
        ):
            return None
        lengths.append(cell_lengths)
    return np.asarray(lengths, dtype=float)


def _unwrap_positions_matrix(
    positions: np.ndarray,
    cell_lengths: np.ndarray | None,
) -> np.ndarray:
    if positions.ndim != 3 or positions.shape[0] <= 1 or cell_lengths is None:
        return np.asarray(positions, dtype=float)
    unwrapped = np.array(positions, dtype=float, copy=True)
    for frame_index in range(1, positions.shape[0]):
        lengths = np.asarray(cell_lengths[frame_index - 1], dtype=float)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
            continue
        delta = positions[frame_index] - positions[frame_index - 1]
        delta -= lengths[np.newaxis, :] * np.rint(delta / lengths[np.newaxis, :])
        unwrapped[frame_index] = unwrapped[frame_index - 1] + delta
    return unwrapped


def _candidate_mask_for_frame(
    frame: Atoms,
    *,
    selected_elements: tuple[str, ...],
    atom_indices: tuple[int, ...] | None,
    atom_mask: np.ndarray | None,
    include_fixed_surface_atoms: bool,
) -> np.ndarray:
    n_atoms = len(frame)
    if atom_mask is not None:
        mask = np.asarray(atom_mask, dtype=bool).copy()
    elif atom_indices is not None:
        mask = np.zeros(n_atoms, dtype=bool)
        if atom_indices:
            mask[np.asarray(atom_indices, dtype=int)] = True
    elif selected_elements:
        symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
        mask = np.isin(symbols, list(selected_elements))
    else:
        mask = np.zeros(n_atoms, dtype=bool)
    if not include_fixed_surface_atoms:
        mask &= ~_fixed_atom_mask(frame)
    return mask


def _build_surface_context(
    frames: list[Atoms],
    axis: str,
    *,
    options: SurfaceEstimatorOptions,
) -> _SurfaceAnalysisContext | None:
    axis_index = axis_to_index(axis)
    if not frames:
        return None
    stable_layout = _frames_share_consistent_atom_layout(frames)
    if (
        options.surface_atom_indices is not None or options.surface_atom_mask is not None
    ) and not stable_layout:
        raise ValueError(
            "Explicit surface atom indices/masks require a stable atom layout across frames."
        )

    if options.surface_atom_indices is not None or options.surface_atom_mask is not None:
        mask = _candidate_mask_for_frame(
            frames[0],
            selected_elements=tuple(),
            atom_indices=options.surface_atom_indices,
            atom_mask=options.surface_atom_mask,
            include_fixed_surface_atoms=options.include_fixed_surface_atoms,
        )
        candidate_indices = np.flatnonzero(mask) if stable_layout else None
        selected_elements = (
            tuple(
                np.unique(
                    np.asarray(frames[0].get_chemical_symbols(), dtype=object)[candidate_indices]
                ).tolist()
            )
            if candidate_indices is not None and candidate_indices.size > 0
            else tuple()
        )
    else:
        resolved_elements, _source = _resolve_surface_elements(
            frames,
            list(options.surface_elements) if options.surface_elements is not None else None,
            axis_index=axis_index,
        )
        if not resolved_elements:
            return None
        selected_elements = tuple(resolved_elements)
        candidate_indices = None
        if stable_layout:
            mask = _candidate_mask_for_frame(
                frames[0],
                selected_elements=selected_elements,
                atom_indices=None,
                atom_mask=None,
                include_fixed_surface_atoms=options.include_fixed_surface_atoms,
            )
            if not np.any(mask):
                return None
            if not options.include_fixed_surface_atoms:
                fixed_mask = np.zeros(len(frames[0]), dtype=bool)
                for frame in frames:
                    fixed_mask |= _fixed_atom_mask(frame)
                mask &= ~fixed_mask
            candidate_indices = np.flatnonzero(mask)

    if candidate_indices is not None and candidate_indices.size == 0:
        return None

    candidate_positions: np.ndarray | None = None
    candidate_axis_values: np.ndarray | None = None
    candidate_masses: np.ndarray | None = None
    translated_candidate_positions: np.ndarray | None = None
    cell_lengths = _cell_lengths_if_periodic_all(frames)
    if stable_layout and candidate_indices is not None and candidate_indices.size > 0:
        candidate_positions = np.stack(
            [np.asarray(frame.positions[candidate_indices], dtype=float) for frame in frames],
            axis=0,
        )
        candidate_axis_values = np.asarray(candidate_positions[:, :, axis_index], dtype=float)
        candidate_masses = np.asarray(frames[0].get_masses(), dtype=float)[candidate_indices]
        unwrapped = _unwrap_positions_matrix(candidate_positions, cell_lengths)
        translated_candidate_positions = unwrapped - np.median(unwrapped, axis=1, keepdims=True)

    return _SurfaceAnalysisContext(
        frames=frames,
        axis_index=axis_index,
        axis=axis.lower(),
        options=options,
        stable_layout=stable_layout,
        selected_elements=selected_elements,
        candidate_indices=None
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int),
        candidate_positions=candidate_positions,
        candidate_axis_values=candidate_axis_values,
        candidate_masses=candidate_masses,
        translated_candidate_positions=translated_candidate_positions,
        cell_lengths=cell_lengths,
    )


def _candidate_axis_values_and_indices(
    context: _SurfaceAnalysisContext,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if context.candidate_indices is not None and context.candidate_axis_values is not None:
        return (
            np.asarray(context.candidate_axis_values[frame_index], dtype=float),
            np.asarray(context.candidate_indices, dtype=int),
        )
    frame = context.frames[frame_index]
    mask = _candidate_mask_for_frame(
        frame,
        selected_elements=context.selected_elements,
        atom_indices=context.options.surface_atom_indices,
        atom_mask=context.options.surface_atom_mask,
        include_fixed_surface_atoms=context.options.include_fixed_surface_atoms,
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    return (
        np.asarray(frame.positions[indices, context.axis_index], dtype=float),
        np.asarray(indices, dtype=int),
    )


def _axis_span_fraction(
    axis_values: np.ndarray,
    cell_lengths: np.ndarray | None,
    *,
    axis_index: int,
) -> float:
    if axis_values.size == 0:
        return 1.0
    span = float(np.ptp(axis_values))
    if cell_lengths is None:
        return span
    axis_length = float(np.median(cell_lengths[:, axis_index]))
    if not np.isfinite(axis_length) or axis_length <= 0.0:
        return span
    return span / axis_length


def _side_outer_anchor(values: np.ndarray, *, side: str) -> float | None:
    if values.size == 0:
        return None
    if side == "top":
        return float(np.max(values))
    return float(np.min(values))


def _surface_proximity_depth(
    values: np.ndarray,
    *,
    side: str,
    anchor: float | None = None,
) -> float:
    if values.size == 0:
        return float("inf")
    resolved_anchor = _side_outer_anchor(values, side=side) if anchor is None else float(anchor)
    if resolved_anchor is None or not np.isfinite(resolved_anchor):
        return float("inf")
    center = float(np.median(np.asarray(values, dtype=float)))
    if side == "top":
        return max(0.0, resolved_anchor - center)
    return max(0.0, center - resolved_anchor)


def _surface_envelope_mask(
    values: np.ndarray,
    *,
    side: str,
    envelope_A: float,
    anchor: float | None = None,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    resolved_anchor = _side_outer_anchor(values, side=side) if anchor is None else float(anchor)
    if resolved_anchor is None or not np.isfinite(resolved_anchor):
        return np.ones(values.size, dtype=bool)
    envelope = max(float(envelope_A), 1.0e-12)
    if side == "top":
        return np.asarray(values, dtype=float) >= resolved_anchor - envelope
    return np.asarray(values, dtype=float) <= resolved_anchor + envelope


def _adaptive_rough_surface_envelope_A(
    axis_values: np.ndarray,
    *,
    side: str,
    options: SurfaceEstimatorOptions,
) -> float:
    if axis_values.size == 0:
        return max(float(options.rough_max_reference_spread_A), float(options.gap_min_A))
    ordered = np.sort(np.asarray(axis_values, dtype=float))
    if side == "top":
        ordered = ordered[::-1]
    diffs = np.abs(np.diff(ordered))
    positive = diffs[diffs > 1.0e-6]
    if positive.size == 0:
        return max(float(options.gap_min_A), 0.5 * float(options.rough_max_reference_spread_A))
    baseline_count = max(1, diffs.size // 2)
    baseline = float(np.median(np.partition(diffs, baseline_count - 1)[:baseline_count]))
    gap_threshold = max(float(options.gap_min_A), float(options.gap_factor) * baseline)
    significant = positive[positive >= gap_threshold]
    outer_gap = float(significant[0]) if significant.size > 0 else float(np.max(positive))
    envelope = max(0.75 * outer_gap, 3.0 * baseline, float(options.gap_min_A))
    return float(envelope)


def _resolve_rough_surface_envelope_A(
    axis_values: np.ndarray,
    *,
    side: str,
    options: SurfaceEstimatorOptions,
) -> float:
    if options.rough_surface_envelope_A is not None:
        return float(options.rough_surface_envelope_A)
    return _adaptive_rough_surface_envelope_A(axis_values, side=side, options=options)


def _side_quantile(values: np.ndarray, *, side: str, quantile: float) -> float:
    if side == "top":
        return float(np.quantile(values, float(quantile)))
    return float(np.quantile(values, 1.0 - float(quantile)))


def _extract_layer_selection(
    axis_values: np.ndarray,
    *,
    side: str,
    options: SurfaceEstimatorOptions,
) -> tuple[np.ndarray | None, np.ndarray | None, int, float, float]:
    if axis_values.size < 2:
        return None, None, 1, float("nan"), float("nan")
    sorted_order = np.argsort(axis_values)
    sorted_axis_values = np.asarray(axis_values[sorted_order], dtype=float)
    diffs = np.diff(sorted_axis_values)
    small_half = max(1, diffs.size // 2)
    baseline = float(np.median(np.partition(diffs, small_half - 1)[:small_half]))
    gap_threshold = max(float(options.gap_min_A), float(options.gap_factor) * baseline)
    significant_gaps = np.where(diffs >= gap_threshold)[0]
    largest_gap = float(np.max(diffs)) if diffs.size else float("nan")

    split_index: int | None = None
    layer_count = int(significant_gaps.size + 1) if significant_gaps.size > 0 else 1
    if significant_gaps.size > 0:
        split_index = (
            int(significant_gaps[-1] + 1) if side == "top" else int(significant_gaps[0] + 1)
        )
    elif diffs.size > 0:
        largest_gap_index = int(np.argmax(diffs))
        if float(diffs[largest_gap_index]) >= 2.0 * float(options.gap_min_A):
            split_index = int(largest_gap_index + 1)
            layer_count = 2
    if split_index is None:
        return None, None, layer_count, largest_gap, baseline

    if side == "top":
        local_indices = sorted_order[split_index:]
    else:
        local_indices = sorted_order[:split_index]
    if local_indices.size == 0:
        return None, None, layer_count, largest_gap, baseline
    return (
        np.asarray(axis_values[local_indices], dtype=float),
        np.asarray(local_indices, dtype=int),
        layer_count,
        largest_gap,
        baseline,
    )


def _build_surface_summary(
    values: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    method_label: str,
    composite_score: float,
) -> SurfaceSummary:
    if np.any(valid_mask):
        valid_values = np.asarray(values[valid_mask], dtype=float)
        position = float(np.median(valid_values))
        std = float(np.std(valid_values, ddof=0))
        median_confidence = float(np.median(np.asarray(confidence[valid_mask], dtype=float)))
    else:
        position = None
        std = None
        median_confidence = 0.0
    return SurfaceSummary(
        position=position,
        std=std,
        valid_fraction=float(np.count_nonzero(valid_mask)) / float(max(1, values.size)),
        median_confidence=median_confidence,
        method_label=method_label,
        composite_score=float(np.clip(composite_score, 0.0, 1.0)),
    )


def _estimate_composite_score(
    values: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    spread: np.ndarray,
    *,
    spread_limit: float,
    jump_tolerance_A: float,
) -> float:
    valid_fraction = float(np.count_nonzero(valid_mask)) / float(max(1, values.size))
    median_confidence = float(np.median(confidence[valid_mask])) if np.any(valid_mask) else 0.0
    smoothness = _surface_smoothness_score(values, valid_mask, tolerance_A=jump_tolerance_A)
    if np.any(valid_mask):
        spread_values = np.asarray(spread[valid_mask], dtype=float)
        finite_spread = spread_values[np.isfinite(spread_values)]
        if finite_spread.size > 0:
            spread_quality = float(
                np.clip(
                    1.0 - np.median(finite_spread) / max(spread_limit, 1.0e-12),
                    0.0,
                    1.0,
                )
            )
        else:
            spread_quality = 0.0
    else:
        spread_quality = 0.0
    return float(
        0.35 * valid_fraction + 0.35 * median_confidence + 0.15 * smoothness + 0.15 * spread_quality
    )


def _make_surface_estimate(
    *,
    values: np.ndarray,
    confidence: np.ndarray,
    provenance: np.ndarray,
    candidate_indices: np.ndarray | None,
    selected_elements: tuple[str, ...],
    mode: str,
    side: str,
    diagnostics: SurfaceDiagnostics,
    spread: np.ndarray,
    spread_limit: float,
    jump_tolerance_A: float,
    method_label: str,
) -> SurfaceEstimate:
    valid_mask = np.isfinite(values)
    summary = _build_surface_summary(
        values,
        valid_mask,
        confidence,
        method_label=method_label,
        composite_score=_estimate_composite_score(
            values,
            valid_mask,
            confidence,
            spread,
            spread_limit=spread_limit,
            jump_tolerance_A=jump_tolerance_A,
        ),
    )
    return SurfaceEstimate(
        frame_values=np.asarray(values, dtype=float),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        confidence=np.asarray(np.clip(confidence, 0.0, 1.0), dtype=float),
        provenance=np.asarray(provenance, dtype=object),
        candidate_indices=None
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int),
        selected_elements=tuple(selected_elements),
        mode=mode,
        side=side,
        summary=summary,
        diagnostics=diagnostics,
    )


def _surface_rejection_summary(reasons: np.ndarray) -> dict[str, int]:
    values = np.asarray(reasons, dtype=object)
    counts: dict[str, int] = {}
    for item in values:
        token = str(item).strip()
        if not token:
            continue
        counts[token] = counts.get(token, 0) + 1
    return counts


def _log_surface_estimate_candidate_debug(
    *,
    axis: str,
    label: str,
    estimate: SurfaceEstimate | None,
) -> None:
    axis_label = axis.upper()
    if estimate is None:
        LOGGER.debug("Surface estimator candidate %s along %s: unavailable.", label, axis_label)
        return
    LOGGER.debug(
        "Surface estimator candidate %s along %s: method=%s, score=%.3f, valid_fraction=%.3f, "
        "median_confidence=%.3f, position=%s.",
        label,
        axis_label,
        estimate.summary.method_label,
        estimate.summary.composite_score,
        estimate.summary.valid_fraction,
        estimate.summary.median_confidence,
        "None" if estimate.summary.position is None else f"{estimate.summary.position:.6g}",
    )
    rejection_counts = _surface_rejection_summary(estimate.diagnostics.rejection_reason)
    if rejection_counts:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(rejection_counts.items()))
        LOGGER.debug(
            "Surface estimator candidate %s along %s rejection summary: %s.",
            label,
            axis_label,
            summary,
        )


def _build_layered_estimate(context: _SurfaceAnalysisContext) -> SurfaceEstimate | None:
    frame_count = len(context.frames)
    values = np.full(frame_count, np.nan, dtype=float)
    confidence = np.zeros(frame_count, dtype=float)
    provenance = np.asarray(["missing"] * frame_count, dtype=object)
    candidate_count = np.zeros(frame_count, dtype=int)
    top_layer_size = np.zeros(frame_count, dtype=int)
    largest_gap = np.full(frame_count, np.nan, dtype=float)
    baseline_gap = np.full(frame_count, np.nan, dtype=float)
    reference_spread = np.full(frame_count, np.nan, dtype=float)
    jump_rejection_mask = np.zeros(frame_count, dtype=bool)
    rejection_reason = np.asarray(["missing"] * frame_count, dtype=object)
    layer_counts: list[int] = []
    top_layer_indices_by_frame: list[np.ndarray | None] = [None] * frame_count

    for frame_index in range(frame_count):
        axis_values, _frame_indices = _candidate_axis_values_and_indices(context, frame_index)
        candidate_count[frame_index] = int(axis_values.size)
        if axis_values.size < max(4, int(context.options.minimum_top_layer_atoms)):
            rejection_reason[frame_index] = "insufficient_candidates"
            continue
        layer_values, local_indices, layer_count, largest_gap_value, baseline_value = (
            _extract_layer_selection(
                axis_values,
                side=context.options.side,
                options=context.options,
            )
        )
        largest_gap[frame_index] = largest_gap_value
        baseline_gap[frame_index] = baseline_value
        layer_counts.append(layer_count)
        if layer_values is None or local_indices is None:
            rejection_reason[frame_index] = "no_layer_break"
            continue
        top_layer_indices_by_frame[frame_index] = np.asarray(
            _frame_indices[local_indices],
            dtype=int,
        )
        top_layer_size[frame_index] = int(layer_values.size)
        minimum_top_layer_size = max(
            int(context.options.minimum_top_layer_atoms),
            int(np.ceil(float(context.options.minimum_top_layer_fraction) * axis_values.size)),
        )
        if layer_values.size < minimum_top_layer_size:
            rejection_reason[frame_index] = "top_layer_too_small"
            continue
        spread = float(np.std(layer_values, ddof=0))
        reference_spread[frame_index] = spread
        if spread > float(context.options.layered_max_spread_A):
            rejection_reason[frame_index] = "top_layer_too_broad"
            continue
        bulk_values = np.delete(np.asarray(axis_values, dtype=float), local_indices)
        if bulk_values.size > 0:
            if context.options.side == "top":
                if float(np.min(layer_values)) <= float(np.quantile(bulk_values, 0.75)):
                    rejection_reason[frame_index] = "top_layer_overlaps_bulk"
                    continue
            else:
                if float(np.max(layer_values)) >= float(np.quantile(bulk_values, 0.25)):
                    rejection_reason[frame_index] = "bottom_layer_overlaps_bulk"
                    continue
        values[frame_index] = _robust_reduce(layer_values, context.options)
        provenance[frame_index] = "direct_layered"
        rejection_reason[frame_index] = ""

    cursor = 0
    while cursor < frame_count:
        if np.isfinite(values[cursor]):
            cursor += 1
            continue
        start = cursor
        while cursor < frame_count and not np.isfinite(values[cursor]):
            cursor += 1
        stop = cursor
        if stop - start > int(context.options.fill_max_gap):
            continue
        for frame_index in range(start, stop):
            tracked_indices = _nearest_valid_layer_indices(
                top_layer_indices_by_frame,
                frame_index=frame_index,
            )
            if tracked_indices is None or tracked_indices.size == 0:
                continue
            frame_axis_values = np.asarray(
                context.frames[frame_index].positions[:, context.axis_index],
                dtype=float,
            )
            if np.any(tracked_indices >= frame_axis_values.size):
                continue
            tracked_values = np.asarray(frame_axis_values[tracked_indices], dtype=float)
            if tracked_values.size == 0 or not np.all(np.isfinite(tracked_values)):
                continue
            left = frame_index - 1
            while left >= 0 and not np.isfinite(values[left]):
                left -= 1
            right = frame_index + 1
            while right < frame_count and not np.isfinite(values[right]):
                right += 1
            neighbor_values = [
                float(values[index])
                for index in (left, right)
                if 0 <= index < frame_count and np.isfinite(values[index])
            ]
            candidate_value = _robust_reduce(tracked_values, context.options)
            if neighbor_values and max(
                abs(candidate_value - neighbor) for neighbor in neighbor_values
            ) > float(context.options.fill_neighbor_tolerance_A):
                rejection_reason[frame_index] = "tracked_fill_inconsistent"
                continue
            values[frame_index] = candidate_value
            provenance[frame_index] = "tracked_fill"
            top_layer_size[frame_index] = int(tracked_values.size)
            reference_spread[frame_index] = float(np.std(tracked_values, ddof=0))
            rejection_reason[frame_index] = ""

    valid_mask = np.isfinite(values)
    temporal = _surface_temporal_component(
        values, valid_mask, tolerance_A=float(context.options.jump_reject_tolerance_A)
    )
    for frame_index in np.flatnonzero(valid_mask):
        if temporal[frame_index] <= 0.0 and np.count_nonzero(valid_mask) > 1:
            values[frame_index] = np.nan
            provenance[frame_index] = "missing"
            rejection_reason[frame_index] = "jump_reject"
            jump_rejection_mask[frame_index] = True
            continue
        gap_ratio = 0.0
        if np.isfinite(largest_gap[frame_index]):
            threshold = max(
                float(context.options.gap_min_A),
                float(context.options.gap_factor) * max(float(baseline_gap[frame_index]), 1.0e-12),
            )
            gap_ratio = float(
                np.clip(largest_gap[frame_index] / max(threshold, 1.0e-12), 0.0, 1.5) / 1.5
            )
        size_score = float(
            np.clip(
                top_layer_size[frame_index]
                / max(
                    float(context.options.minimum_top_layer_atoms),
                    float(context.options.minimum_top_layer_fraction)
                    * max(candidate_count[frame_index], 1),
                ),
                0.0,
                1.0,
            )
        )
        spread_score = float(
            np.clip(
                1.0
                - reference_spread[frame_index]
                / max(float(context.options.layered_max_spread_A), 1.0e-12),
                0.0,
                1.0,
            )
        )
        base_confidence = float(
            0.35 * gap_ratio
            + 0.25 * size_score
            + 0.20 * spread_score
            + 0.20 * temporal[frame_index]
        )
        if provenance[frame_index] == "tracked_fill":
            base_confidence *= 0.75
        confidence[frame_index] = base_confidence

    diagnostics = SurfaceDiagnostics(
        candidate_count_per_frame=np.asarray(candidate_count, dtype=int),
        top_layer_size_per_frame=np.asarray(top_layer_size, dtype=int),
        largest_gap_A_per_frame=np.asarray(largest_gap, dtype=float),
        baseline_gap_A_per_frame=np.asarray(baseline_gap, dtype=float),
        reference_spread_A_per_frame=np.asarray(reference_spread, dtype=float),
        jump_rejection_mask=np.asarray(jump_rejection_mask, dtype=bool),
        rejection_reason=np.asarray(rejection_reason, dtype=object),
        effective_options={
            str(key): _json_ready(value) for key, value in asdict(context.options).items()
        },
    )
    median_layers = int(np.median(np.asarray(layer_counts, dtype=float))) if layer_counts else 0
    return _make_surface_estimate(
        values=values,
        confidence=confidence,
        provenance=provenance,
        candidate_indices=context.candidate_indices,
        selected_elements=context.selected_elements,
        mode="layered",
        side=context.options.side,
        diagnostics=diagnostics,
        spread=reference_spread,
        spread_limit=float(context.options.layered_max_spread_A),
        jump_tolerance_A=float(context.options.jump_reject_tolerance_A),
        method_label=f"layered_top_layer_{context.options.reduction_mode}(median_layers={median_layers})",
    )


def _build_rough_estimate(context: _SurfaceAnalysisContext) -> SurfaceEstimate | None:
    frame_count = len(context.frames)
    values = np.full(frame_count, np.nan, dtype=float)
    confidence = np.zeros(frame_count, dtype=float)
    provenance = np.asarray(["missing"] * frame_count, dtype=object)
    candidate_count = np.zeros(frame_count, dtype=int)
    top_layer_size = np.zeros(frame_count, dtype=int)
    largest_gap = np.full(frame_count, np.nan, dtype=float)
    baseline_gap = np.full(frame_count, np.nan, dtype=float)
    reference_spread = np.full(frame_count, np.nan, dtype=float)
    jump_rejection_mask = np.zeros(frame_count, dtype=bool)
    rejection_reason = np.asarray(["missing"] * frame_count, dtype=object)
    effective_options = {
        str(key): _json_ready(value) for key, value in asdict(context.options).items()
    }

    used_low_mobility = False
    if (
        context.stable_layout
        and context.candidate_indices is not None
        and context.translated_candidate_positions is not None
        and context.candidate_axis_values is not None
        and context.candidate_positions is not None
        and context.candidate_masses is not None
    ):
        candidate_count[:] = int(context.candidate_indices.size)
        median_positions = np.median(context.translated_candidate_positions, axis=0)
        mobility = np.median(
            np.linalg.norm(
                context.translated_candidate_positions - median_positions[np.newaxis, :, :], axis=2
            ),
            axis=0,
        )
        mobility = np.nan_to_num(mobility, nan=np.inf)
        axis_medians = np.median(context.candidate_axis_values, axis=0)
        side_anchor = _side_outer_anchor(axis_medians, side=context.options.side)
        resolved_envelope = _resolve_rough_surface_envelope_A(
            axis_medians,
            side=context.options.side,
            options=context.options,
        )
        effective_options["rough_surface_envelope_A_effective"] = resolved_envelope
        envelope_mask = np.zeros(axis_medians.size, dtype=bool)
        envelope_used = None
        for multiplier in (1.0, 1.5, 2.0, 3.0):
            candidate_mask = _surface_envelope_mask(
                axis_medians,
                side=context.options.side,
                envelope_A=resolved_envelope * multiplier,
                anchor=side_anchor,
            )
            if np.count_nonzero(candidate_mask) < int(context.options.rough_reference_min_atoms):
                continue
            candidate_axis_values = context.candidate_axis_values[:, candidate_mask]
            if _axis_span_fraction(
                candidate_axis_values,
                context.cell_lengths,
                axis_index=context.axis_index,
            ) > float(context.options.candidate_axis_span_max_fraction):
                continue
            envelope_mask = np.asarray(candidate_mask, dtype=bool)
            envelope_used = float(resolved_envelope * multiplier)
            break
        if not np.any(envelope_mask):
            envelope_mask = _surface_envelope_mask(
                axis_medians,
                side=context.options.side,
                envelope_A=resolved_envelope,
                anchor=side_anchor,
            )
            envelope_used = float(resolved_envelope)
        effective_options["rough_surface_envelope_A_used"] = envelope_used
        effective_options["rough_surface_envelope_candidate_count"] = int(
            np.count_nonzero(envelope_mask)
        )
        if np.any(envelope_mask):
            envelope_indices = np.flatnonzero(envelope_mask)
            envelope_mobility = mobility[envelope_mask]
            envelope_masses = context.candidate_masses[envelope_mask]
            atom_count = int(envelope_indices.size)
            if atom_count >= int(context.options.rough_reference_min_atoms):
                desired = int(np.ceil(float(context.options.rough_reference_fraction) * atom_count))
                desired = max(desired, int(context.options.rough_reference_min_atoms))
                desired = min(
                    desired, int(context.options.rough_reference_max_soft_cap), atom_count
                )
                mobility_rank = np.argsort(np.argsort(envelope_mobility))
                heavy_rank = np.argsort(np.argsort(-envelope_masses))
                combined_score = mobility_rank.astype(float) + float(
                    context.options.mass_tiebreak_weight
                ) * heavy_rank.astype(float)
                chosen_local = np.argpartition(combined_score, desired - 1)[:desired]
                chosen_local.sort()
                chosen_indices = envelope_indices[chosen_local]
                effective_options["rough_reference_atom_indices"] = chosen_indices.astype(int)
                reference_values = context.candidate_axis_values[:, chosen_indices]
                reference_spread = np.std(reference_values, axis=1, ddof=0)
                reference_depth = _surface_proximity_depth(
                    axis_medians[chosen_indices],
                    side=context.options.side,
                    anchor=side_anchor,
                )
                effective_options["rough_reference_depth_A"] = reference_depth
                for frame_index in range(frame_count):
                    if reference_spread[frame_index] > float(
                        context.options.rough_max_reference_spread_A
                    ):
                        rejection_reason[frame_index] = "reference_set_too_broad"
                        continue
                    values[frame_index] = _robust_reduce(
                        reference_values[frame_index], context.options
                    )
                    provenance[frame_index] = "direct_rough_low_mobility"
                    rejection_reason[frame_index] = ""
                used_low_mobility = np.any(np.isfinite(values))
                if used_low_mobility and envelope_used is not None:
                    proximity_score = float(
                        np.clip(1.0 - reference_depth / max(envelope_used, 1.0e-12), 0.0, 1.0)
                    )
                    effective_options["rough_surface_proximity_score"] = proximity_score

    if not used_low_mobility:
        for frame_index in range(frame_count):
            axis_values, _frame_indices = _candidate_axis_values_and_indices(context, frame_index)
            candidate_count[frame_index] = int(axis_values.size)
            if axis_values.size == 0:
                rejection_reason[frame_index] = "insufficient_candidates"
                continue
            reference_spread[frame_index] = float(np.std(axis_values, ddof=0))
            values[frame_index] = _side_quantile(
                axis_values,
                side=context.options.side,
                quantile=float(context.options.rough_quantile),
            )
            provenance[frame_index] = "direct_rough_quantile"
            rejection_reason[frame_index] = ""

    valid_mask = np.isfinite(values)
    temporal = _surface_temporal_component(
        values, valid_mask, tolerance_A=float(context.options.jump_reject_tolerance_A)
    )
    for frame_index in np.flatnonzero(valid_mask):
        if temporal[frame_index] <= 0.0 and np.count_nonzero(valid_mask) > 1:
            values[frame_index] = np.nan
            provenance[frame_index] = "missing"
            rejection_reason[frame_index] = "jump_reject"
            jump_rejection_mask[frame_index] = True
            continue
        spread_score = float(
            np.clip(
                1.0
                - reference_spread[frame_index]
                / max(float(context.options.rough_max_reference_spread_A), 1.0e-12),
                0.0,
                1.0,
            )
        )
        source_quality = 1.0 if provenance[frame_index] == "direct_rough_low_mobility" else 0.6
        proximity_score = float(effective_options.get("rough_surface_proximity_score", 1.0))
        confidence[frame_index] = float(
            0.35 * spread_score
            + 0.20 * temporal[frame_index]
            + 0.25 * source_quality
            + 0.20 * proximity_score
        )

    diagnostics = SurfaceDiagnostics(
        candidate_count_per_frame=np.asarray(candidate_count, dtype=int),
        top_layer_size_per_frame=np.asarray(top_layer_size, dtype=int),
        largest_gap_A_per_frame=np.asarray(largest_gap, dtype=float),
        baseline_gap_A_per_frame=np.asarray(baseline_gap, dtype=float),
        reference_spread_A_per_frame=np.asarray(reference_spread, dtype=float),
        jump_rejection_mask=np.asarray(jump_rejection_mask, dtype=bool),
        rejection_reason=np.asarray(rejection_reason, dtype=object),
        effective_options=effective_options,
    )
    if used_low_mobility:
        reducer = context.options.reduction_mode
        label = f"rough_low_mobility_{reducer}"
    else:
        label = (
            "upper_reference_quantile"
            if context.options.side == "top"
            else "lower_reference_quantile"
        )
    return _make_surface_estimate(
        values=values,
        confidence=confidence,
        provenance=provenance,
        candidate_indices=context.candidate_indices,
        selected_elements=context.selected_elements,
        mode="rough",
        side=context.options.side,
        diagnostics=diagnostics,
        spread=reference_spread,
        spread_limit=float(context.options.rough_max_reference_spread_A),
        jump_tolerance_A=float(context.options.jump_reject_tolerance_A),
        method_label=label,
    )


def _conservative_fill_surface_estimate(
    estimate: SurfaceEstimate,
    *,
    context: _SurfaceAnalysisContext,
) -> SurfaceEstimate:
    values = np.asarray(estimate.frame_values, dtype=float).copy()
    confidence = np.asarray(estimate.confidence, dtype=float).copy()
    provenance = np.asarray(estimate.provenance, dtype=object).copy()
    rejection_reason = np.asarray(estimate.diagnostics.rejection_reason, dtype=object).copy()
    jump_rejection_mask = np.asarray(estimate.diagnostics.jump_rejection_mask, dtype=bool).copy()
    frame_count = values.size
    cursor = 0
    while cursor < frame_count:
        if np.isfinite(values[cursor]):
            cursor += 1
            continue
        start = cursor
        while cursor < frame_count and not np.isfinite(values[cursor]):
            cursor += 1
        stop = cursor
        run_length = stop - start
        if run_length > int(context.options.fill_max_gap):
            continue
        left = start - 1
        while left >= 0 and not np.isfinite(values[left]):
            left -= 1
        right = stop
        while right < frame_count and not np.isfinite(values[right]):
            right += 1
        neighbor_values = [
            float(values[index])
            for index in (left, right)
            if 0 <= index < frame_count and np.isfinite(values[index])
        ]
        for frame_index in range(start, stop):
            axis_values, _frame_indices = _candidate_axis_values_and_indices(context, frame_index)
            if axis_values.size == 0:
                continue
            candidate_value = _side_quantile(
                axis_values,
                side=context.options.side,
                quantile=float(context.options.rough_quantile),
            )
            if neighbor_values and max(
                abs(candidate_value - neighbor) for neighbor in neighbor_values
            ) > float(context.options.fill_neighbor_tolerance_A):
                rejection_reason[frame_index] = "quantile_fill_inconsistent"
                continue
            values[frame_index] = float(candidate_value)
            provenance[frame_index] = "quantile_fill"
            confidence[frame_index] = 0.45
            rejection_reason[frame_index] = ""
    diagnostics = SurfaceDiagnostics(
        candidate_count_per_frame=np.asarray(
            estimate.diagnostics.candidate_count_per_frame, dtype=int
        ),
        top_layer_size_per_frame=np.asarray(
            estimate.diagnostics.top_layer_size_per_frame, dtype=int
        ),
        largest_gap_A_per_frame=np.asarray(
            estimate.diagnostics.largest_gap_A_per_frame, dtype=float
        ),
        baseline_gap_A_per_frame=np.asarray(
            estimate.diagnostics.baseline_gap_A_per_frame, dtype=float
        ),
        reference_spread_A_per_frame=np.asarray(
            estimate.diagnostics.reference_spread_A_per_frame, dtype=float
        ),
        jump_rejection_mask=jump_rejection_mask,
        rejection_reason=rejection_reason,
        effective_options=dict(estimate.diagnostics.effective_options),
    )
    spread_limit = (
        float(context.options.layered_max_spread_A)
        if estimate.mode == "layered"
        else float(context.options.rough_max_reference_spread_A)
    )
    return _make_surface_estimate(
        values=values,
        confidence=confidence,
        provenance=provenance,
        candidate_indices=estimate.candidate_indices,
        selected_elements=estimate.selected_elements,
        mode=estimate.mode,
        side=estimate.side,
        diagnostics=diagnostics,
        spread=np.asarray(diagnostics.reference_spread_A_per_frame, dtype=float),
        spread_limit=spread_limit,
        jump_tolerance_A=float(context.options.jump_reject_tolerance_A),
        method_label=estimate.method,
    )


def _surface_estimate_supports_distance_mode(
    estimate: SurfaceEstimate | None,
    *,
    frame_count: int,
) -> bool:
    if estimate is None:
        return False
    if estimate.frame_values.shape[0] != frame_count:
        return False
    if not np.all(estimate.valid_mask):
        return False
    if estimate.summary.valid_fraction < 1.0:
        return False
    if estimate.summary.median_confidence < float(
        estimate.diagnostics.effective_options.get("low_confidence_threshold", 0.55)
    ):
        return False
    return True


def _select_surface_estimate_object(
    context: _SurfaceAnalysisContext,
) -> SurfaceEstimate | None:
    layered_estimate = _build_layered_estimate(context)
    rough_estimate = _build_rough_estimate(context)
    if layered_estimate is not None:
        layered_estimate = _conservative_fill_surface_estimate(layered_estimate, context=context)
    if rough_estimate is not None:
        rough_estimate = _conservative_fill_surface_estimate(rough_estimate, context=context)
    _log_surface_estimate_candidate_debug(
        axis=context.axis,
        label="layered",
        estimate=layered_estimate,
    )
    _log_surface_estimate_candidate_debug(
        axis=context.axis,
        label="rough",
        estimate=rough_estimate,
    )
    if context.options.mode == "layered":
        return layered_estimate
    if context.options.mode == "rough":
        return rough_estimate
    candidates = [
        estimate for estimate in (layered_estimate, rough_estimate) if estimate is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item.summary.composite_score,
            item.summary.valid_fraction,
            item.summary.median_confidence,
        ),
        reverse=True,
    )
    LOGGER.debug(
        "Surface estimator auto selection along %s chose %s (score=%.3f).",
        context.axis.upper(),
        candidates[0].summary.method_label,
        candidates[0].summary.composite_score,
    )
    return candidates[0]


def _any_frame_has_fixed_atoms(frames: list[Atoms]) -> bool:
    """Return True if any frame contains index-based ASE constraints."""
    for frame in frames:
        if np.any(_fixed_atom_mask(frame)):
            return True
    return False


def _inject_effective_option(
    estimate: SurfaceEstimate,
    key: str,
    value: Any,
) -> SurfaceEstimate:
    """Return a copy of *estimate* with an extra key in effective_options."""
    updated_options = dict(estimate.diagnostics.effective_options)
    updated_options[key] = _json_ready(value)
    diagnostics = SurfaceDiagnostics(
        candidate_count_per_frame=estimate.diagnostics.candidate_count_per_frame,
        top_layer_size_per_frame=estimate.diagnostics.top_layer_size_per_frame,
        largest_gap_A_per_frame=estimate.diagnostics.largest_gap_A_per_frame,
        baseline_gap_A_per_frame=estimate.diagnostics.baseline_gap_A_per_frame,
        reference_spread_A_per_frame=estimate.diagnostics.reference_spread_A_per_frame,
        jump_rejection_mask=estimate.diagnostics.jump_rejection_mask,
        rejection_reason=estimate.diagnostics.rejection_reason,
        effective_options=updated_options,
    )
    return SurfaceEstimate(
        frame_values=estimate.frame_values,
        valid_mask=estimate.valid_mask,
        confidence=estimate.confidence,
        provenance=estimate.provenance,
        candidate_indices=estimate.candidate_indices,
        selected_elements=estimate.selected_elements,
        mode=estimate.mode,
        side=estimate.side,
        summary=estimate.summary,
        diagnostics=diagnostics,
    )


def estimate_surface_reference(
    frames: list[Atoms],
    axis: str = "z",
    *,
    mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    logger: logging.Logger | None = None,
    _fixed_atom_fallback_retry: bool = False,
) -> SurfaceEstimate | None:
    if not frames:
        return None
    if any(len(frame) == 0 for frame in frames):
        return None
    options = _resolve_surface_options(
        frames=frames,
        mode=mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
    )
    active_logger = LOGGER if logger is None else logger
    context = _build_surface_context(frames, axis, options=options)
    if context is None:
        # --- fixed-atom fallback: context failed ---
        if (
            not include_fixed_surface_atoms
            and surface_options is None
            and _any_frame_has_fixed_atoms(frames)
        ):
            active_logger.info(
                "Surface estimation along %s found no candidates with fixed atoms excluded; "
                "retrying with include_fixed_surface_atoms=True.",
                axis.upper(),
            )
            return estimate_surface_reference(
                frames,
                axis,
                mode=mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=True,
                surface_options=None,
                logger=logger,
                _fixed_atom_fallback_retry=True,
            )
        return None
    reference_source = "automatic"
    if options.surface_atom_indices is not None or options.surface_atom_mask is not None:
        reference_source = "explicit atom-mask"
    elif options.surface_elements is not None:
        reference_source = "user-selected"
    active_logger.info(
        "Surface reference along %s: %s %s.",
        axis.upper(),
        reference_source,
        _surface_reference_atoms_label(list(context.selected_elements)),
    )
    estimate = _select_surface_estimate_object(context)
    if estimate is None:
        # --- fixed-atom fallback: estimate selection failed ---
        if (
            not include_fixed_surface_atoms
            and surface_options is None
            and _any_frame_has_fixed_atoms(frames)
        ):
            active_logger.info(
                "Surface estimation along %s produced no usable estimate with fixed atoms "
                "excluded; retrying with include_fixed_surface_atoms=True.",
                axis.upper(),
            )
            return estimate_surface_reference(
                frames,
                axis,
                mode=mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=True,
                surface_options=None,
                logger=logger,
                _fixed_atom_fallback_retry=True,
            )
        return None
    # Tag the estimate when the fixed-atom fallback path was used.
    if _fixed_atom_fallback_retry:
        estimate = _inject_effective_option(estimate, "fixed_atom_fallback", True)
    active_logger.info(
        "Surface estimator along %s: %s.",
        axis.upper(),
        _describe_surface_estimator(
            estimate.method,
            axis=axis,
            reference_elements=list(estimate.selected_elements),
        ),
    )
    active_logger.debug(
        "Surface estimator along %s chose %s using %s.",
        axis.upper(),
        estimate.mode,
        _surface_reference_atoms_label(list(estimate.selected_elements)),
    )
    return estimate


def _select_surface_estimate(
    frames: list[Atoms],
    axis: str,
    *,
    mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    logger: logging.Logger | None = None,
) -> tuple[SurfaceEstimate | None, str]:
    estimate = estimate_surface_reference(
        frames,
        axis,
        mode=mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
        logger=logger,
    )
    if estimate is None:
        return None, "unavailable:no_surface_reference"
    return estimate, estimate.method


def estimate_surface_position(
    frames: list[Atoms],
    axis: str = "z",
    *,
    mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    logger: logging.Logger | None = None,
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
        surface_options=surface_options,
        logger=logger,
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


def _surface_effective_options_payload(estimate: SurfaceEstimate) -> dict[str, Any]:
    return {
        str(key): _json_ready(value)
        for key, value in dict(estimate.diagnostics.effective_options).items()
    }


def _surface_metadata_view(metadata: Mapping[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {}
    nested = metadata.get("surface")
    if isinstance(nested, Mapping):
        view.update({str(key): value for key, value in nested.items()})

    flat_aliases = {
        "surface_position": "position",
        "surface_position_std": "position_std",
        "surface_mode": "mode",
        "surface_side": "side",
        "surface_selected_elements": "selected_elements",
        "surface_candidate_indices": "candidate_indices",
        "surface_method_label": "method_label",
        "surface_valid_fraction": "valid_fraction",
        "surface_median_confidence": "median_confidence",
        "surface_composite_score": "composite_score",
        "surface_low_confidence_threshold": "low_confidence_threshold",
        "surface_effective_options": "effective_options",
    }
    for flat_key, nested_key in flat_aliases.items():
        if nested_key not in view and flat_key in metadata:
            view[nested_key] = metadata.get(flat_key)
    return view


def _surface_metadata_payload(
    *,
    surface_position: float | None,
    surface_position_std: float | None,
    estimate: SurfaceEstimate | None,
) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    if surface_position is not None and np.isfinite(surface_position):
        nested["position"] = float(surface_position)
    if surface_position_std is not None and np.isfinite(surface_position_std):
        nested["position_std"] = float(surface_position_std)
    if estimate is not None:
        nested.update(
            {
                "mode": estimate.mode,
                "side": estimate.side,
                "selected_elements": list(estimate.selected_elements),
                "method_label": estimate.summary.method_label,
                "valid_fraction": estimate.summary.valid_fraction,
                "median_confidence": estimate.summary.median_confidence,
                "composite_score": estimate.summary.composite_score,
                "low_confidence_threshold": estimate.diagnostics.effective_options.get(
                    "low_confidence_threshold"
                ),
                "effective_options": _surface_effective_options_payload(estimate),
            }
        )
        if estimate.candidate_indices is not None:
            nested["candidate_indices"] = [
                int(value) for value in np.asarray(estimate.candidate_indices, dtype=int)
            ]
    return {"surface": nested} if nested else {}


def _surface_estimate_metadata(estimate: SurfaceEstimate | None) -> dict[str, Any]:
    return _surface_metadata_payload(
        surface_position=None if estimate is None else estimate.summary.position,
        surface_position_std=None if estimate is None else estimate.summary.std,
        estimate=estimate,
    )


def _surface_estimate_datasets(estimate: SurfaceEstimate | None) -> dict[str, np.ndarray | None]:
    if estimate is None:
        return {
            "surface_position_per_frame_A": None,
            "surface_valid_mask": None,
            "surface_confidence": None,
            "surface_provenance": None,
            "surface_candidate_count": None,
            "surface_top_layer_size": None,
            "surface_largest_gap_A": None,
            "surface_baseline_gap_A": None,
            "surface_reference_spread_A": None,
            "surface_jump_rejection_mask": None,
            "surface_rejection_reason": None,
        }
    diagnostics = estimate.diagnostics
    return {
        "surface_position_per_frame_A": np.asarray(estimate.frame_values, dtype=float),
        "surface_valid_mask": np.asarray(estimate.valid_mask, dtype=bool),
        "surface_confidence": np.asarray(estimate.confidence, dtype=float),
        "surface_provenance": _string_array(estimate.provenance, maxlen=_SURFACE_PROVENANCE_MAXLEN),
        "surface_candidate_count": np.asarray(diagnostics.candidate_count_per_frame, dtype=int),
        "surface_top_layer_size": np.asarray(diagnostics.top_layer_size_per_frame, dtype=int),
        "surface_largest_gap_A": np.asarray(diagnostics.largest_gap_A_per_frame, dtype=float),
        "surface_baseline_gap_A": np.asarray(diagnostics.baseline_gap_A_per_frame, dtype=float),
        "surface_reference_spread_A": np.asarray(
            diagnostics.reference_spread_A_per_frame, dtype=float
        ),
        "surface_jump_rejection_mask": np.asarray(diagnostics.jump_rejection_mask, dtype=bool),
        "surface_rejection_reason": _string_array(
            diagnostics.rejection_reason,
            maxlen=_SURFACE_REASON_MAXLEN,
        ),
    }


def _surface_estimate_from_payload(
    *,
    datasets: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> SurfaceEstimate | None:
    if "surface_position_per_frame_A" not in datasets:
        return None
    frame_values = np.asarray(datasets["surface_position_per_frame_A"], dtype=float)
    valid_mask = (
        np.asarray(datasets["surface_valid_mask"], dtype=bool)
        if "surface_valid_mask" in datasets
        else np.isfinite(frame_values)
    )
    confidence = (
        np.asarray(datasets["surface_confidence"], dtype=float)
        if "surface_confidence" in datasets
        else np.where(valid_mask, 1.0, 0.0)
    )
    provenance = (
        _decode_string_array(np.asarray(datasets["surface_provenance"]))
        if "surface_provenance" in datasets
        else np.where(valid_mask, "loaded", "missing").astype(object)
    )
    candidate_count = (
        np.asarray(datasets["surface_candidate_count"], dtype=int)
        if "surface_candidate_count" in datasets
        else np.zeros(frame_values.shape, dtype=int)
    )
    top_layer_size = (
        np.asarray(datasets["surface_top_layer_size"], dtype=int)
        if "surface_top_layer_size" in datasets
        else np.zeros(frame_values.shape, dtype=int)
    )
    largest_gap = (
        np.asarray(datasets["surface_largest_gap_A"], dtype=float)
        if "surface_largest_gap_A" in datasets
        else np.full(frame_values.shape, np.nan, dtype=float)
    )
    baseline_gap = (
        np.asarray(datasets["surface_baseline_gap_A"], dtype=float)
        if "surface_baseline_gap_A" in datasets
        else np.full(frame_values.shape, np.nan, dtype=float)
    )
    reference_spread = (
        np.asarray(datasets["surface_reference_spread_A"], dtype=float)
        if "surface_reference_spread_A" in datasets
        else np.full(frame_values.shape, np.nan, dtype=float)
    )
    jump_rejection_mask = (
        np.asarray(datasets["surface_jump_rejection_mask"], dtype=bool)
        if "surface_jump_rejection_mask" in datasets
        else np.zeros(frame_values.shape, dtype=bool)
    )
    rejection_reason = (
        _decode_string_array(np.asarray(datasets["surface_rejection_reason"]))
        if "surface_rejection_reason" in datasets
        else np.asarray([""] * frame_values.size, dtype=object)
    )
    surface_metadata = _surface_metadata_view(metadata)
    effective_options_raw = surface_metadata.get("effective_options")
    if isinstance(effective_options_raw, Mapping):
        effective_options = {
            str(key): _json_ready(value) for key, value in effective_options_raw.items()
        }
    else:
        effective_options = {}
    summary = SurfaceSummary(
        position=(
            float(np.median(frame_values[valid_mask]))
            if np.any(valid_mask)
            else _optional_finite_float(surface_metadata.get("position"))
        ),
        std=(
            float(np.std(frame_values[valid_mask], ddof=0))
            if np.any(valid_mask)
            else _optional_finite_float(surface_metadata.get("position_std"))
        ),
        valid_fraction=float(
            surface_metadata.get(
                "valid_fraction", np.count_nonzero(valid_mask) / max(1, frame_values.size)
            )
        ),
        median_confidence=float(
            surface_metadata.get(
                "median_confidence",
                np.median(confidence[valid_mask]) if np.any(valid_mask) else 0.0,
            )
        ),
        method_label=str(
            surface_metadata.get("method_label", surface_metadata.get("mode", "loaded"))
        ),
        composite_score=float(surface_metadata.get("composite_score", 0.0)),
    )
    diagnostics = SurfaceDiagnostics(
        candidate_count_per_frame=candidate_count,
        top_layer_size_per_frame=top_layer_size,
        largest_gap_A_per_frame=largest_gap,
        baseline_gap_A_per_frame=baseline_gap,
        reference_spread_A_per_frame=reference_spread,
        jump_rejection_mask=jump_rejection_mask,
        rejection_reason=np.asarray(rejection_reason, dtype=object),
        effective_options=effective_options,
    )
    candidate_indices_raw = surface_metadata.get("candidate_indices")
    candidate_indices = None
    if isinstance(candidate_indices_raw, Sequence) and not isinstance(
        candidate_indices_raw, (str, bytes)
    ):
        try:
            candidate_indices = np.asarray(candidate_indices_raw, dtype=int)
        except (TypeError, ValueError):
            candidate_indices = None
    return SurfaceEstimate(
        frame_values=frame_values,
        valid_mask=valid_mask,
        confidence=confidence,
        provenance=np.asarray(provenance, dtype=object),
        candidate_indices=candidate_indices,
        selected_elements=tuple(
            str(value) for value in surface_metadata.get("selected_elements", ()) if str(value)
        ),
        mode=str(surface_metadata.get("mode", "loaded")),
        side=str(surface_metadata.get("side", "top")),
        summary=summary,
        diagnostics=diagnostics,
    )


def _cell_histogram_bounds(
    *,
    frames: list[Atoms],
    axis_index: int,
    coordinate_mode: str,
    surface_per_frame: np.ndarray | None,
) -> tuple[float, float] | None:
    if not frames:
        return None
    if not all(_frame_has_usable_cell(frame, axis_index) for frame in frames):
        return None

    lower = float("inf")
    upper = float("-inf")
    for frame_index, frame in enumerate(frames):
        cell = np.asarray(frame.cell.array, dtype=float)
        axis_length = float(np.linalg.norm(cell[axis_index]))
        frame_min = 0.0
        frame_max = axis_length
        if coordinate_mode == "distance" and surface_per_frame is not None:
            frame_surface = float(surface_per_frame[frame_index])
            frame_min -= frame_surface
            frame_max -= frame_surface
        lower = min(lower, frame_min)
        upper = max(upper, frame_max)

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper
