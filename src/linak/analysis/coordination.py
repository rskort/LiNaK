"""Continuous coordination-number analysis routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np

from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    _apply_minor_tick_modes,
    _axis_tick_params,
    _extract_tick_controls,
    _render_plot_annotations,
    _sanitize_line_collection_kwargs,
    configure_matplotlib_backend,
    format_axis_label_units,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
)
from ..progress import ProgressBar
from ..storage.hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles_by_index,
    read_linak_hdf5_profiles,
    write_linak_hdf5,
    write_linak_hdf5_profile_collection,
)
from ..utils import ensure_positive
from .position import (
    _build_xy_segments,
    compute_position_profile,
)
from .density import (
    _surface_estimate_datasets,
    _surface_estimate_from_payload,
    _surface_metadata_payload,
    _surface_metadata_view,
    SurfaceEstimate,
    SurfaceEstimatorOptions,
    available_element_species,
)
from .rdf import (
    _accumulate_rdf_pair_collection,
    _auto_r_max_from_frames,
    _canonical_rdf_pair,
    _normalize_species as _normalize_rdf_species,
)
from .schema import build_profile_metadata

LOGGER = logging.getLogger(__name__)
_DEFAULT_DISTANCE_BIN_WIDTH_A = 0.25
_DEFAULT_CUTOFF_SMOOTHING_WIDTH_A = 0.50
_DEFAULT_RDF_BIN_WIDTH_A = 0.05
_DEFAULT_RDF_SMOOTHING_SIGMA_A = 0.10
_MIN_FIT_POINTS = 3
_FIT_HALF_WINDOW_POINTS = 3
_COORD_DENSE_PAIR_THRESHOLD = 200_000
_COORD_MAX_DISTANCE_VALUES_PER_CHUNK = 2_000_000
_COORD_BACKEND_DENSE = "dense_chunked_orthorhombic"
_COORD_BACKEND_NEIGHBOR = "framewise_periodic_neighbor_list"
_COORD_BACKEND_GENERIC = "framewise_generic"


@dataclass(frozen=True)
class CoordinationCutoffResolution:
    """Resolved cutoff and optional RDF provenance."""

    cutoff_A: float
    smoothing_width_A: float
    mode: str
    rdf_bin_centers_A: np.ndarray | None = None
    rdf_g_r: np.ndarray | None = None
    rdf_g_r_smoothed: np.ndarray | None = None
    rdf_peak_A: float | None = None
    rdf_minimum_A: float | None = None
    rdf_sampled_frame_index: np.ndarray | None = None
    rdf_source_path: str | None = None
    diagnostic_plot_path: str | None = None


@dataclass(frozen=True)
class CoordinationProfile:
    """Container for a continuous coordination-number trajectory."""

    species_a: str
    species_b: str
    axis: str
    atom_indices: np.ndarray
    frame_index: np.ndarray
    step: np.ndarray
    time_fs: np.ndarray
    time_ps: np.ndarray
    distance_to_surface: np.ndarray
    coordination_number: np.ndarray
    n_frames: int
    n_atoms: int
    coordinate_mode: str = "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_position_per_frame: np.ndarray | None = None
    surface_estimate: SurfaceEstimate | None = None
    cell_lengths_angstrom: tuple[float, float, float] | None = None
    cutoff_A: float | None = None
    cutoff_smoothing_width_A: float | None = None
    cutoff_mode: str = "direct"
    cutoff_rdf_bin_centers_A: np.ndarray | None = None
    cutoff_rdf_g_r: np.ndarray | None = None
    cutoff_rdf_g_r_smoothed: np.ndarray | None = None
    cutoff_rdf_peak_A: float | None = None
    cutoff_rdf_minimum_A: float | None = None
    cutoff_rdf_sampled_frame_index: np.ndarray | None = None
    cutoff_rdf_source_path: str | None = None
    cutoff_diagnostic_plot_path: str | None = None


@dataclass(frozen=True)
class _CoordinationSelectionCache:
    """Stable per-run center/neighbor selection metadata reused across frames."""

    center_indices: np.ndarray
    neighbor_indices: np.ndarray
    center_lookup: np.ndarray
    center_mask: np.ndarray
    neighbor_mask: np.ndarray
    same_selection: bool
    center_count: int
    neighbor_count: int
    pair_count: int


def _normalize_species(species: str | None) -> str:
    return _normalize_rdf_species(species)


def _ordered_coordination_pairs_from_frames(frames: list[Atoms]) -> list[tuple[str, str]]:
    """Return the ordered center->neighbor species pairs for bare collection mode."""
    species_labels = available_element_species(frames)
    if not species_labels:
        raise ValueError("No elements found in trajectory.")
    return [(species_a, species_b) for species_a in species_labels for species_b in species_labels]


def _unique_physical_coordination_pairs(
    ordered_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return unique unordered RDF reference pairs needed by coordination pairs."""
    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for species_a, species_b in ordered_pairs:
        pair = _canonical_rdf_pair(species_a, species_b)
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)
    return unique_pairs


def _normalize_axis(axis: str | None) -> str:
    token = "z" if axis is None else str(axis).strip().lower()
    if token not in {"x", "y", "z"}:
        raise ValueError("Axis must be one of: x, y, z.")
    return token


def _normalize_component_token(component: str) -> str:
    token = str(component).strip().lower().replace("_", "-")
    if token in {"distance", "time", "time-distance"}:
        return token
    if token in {"distance-time", "trajectory", "heatmap"}:
        return "time-distance"
    raise ValueError(
        f"Unsupported coordination component '{component}'. "
        "Choose 'distance', 'time', or 'time-distance'."
    )


def _frame_has_usable_cell(frame: Atoms) -> bool:
    if not bool(np.all(frame.get_pbc())):
        return False
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return False
    volume = abs(float(np.linalg.det(cell)))
    return volume > 0.0


def _select_indices(frame: Atoms, species: str) -> np.ndarray:
    if species == "ALL":
        return np.arange(len(frame), dtype=int)
    symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
    return np.where(symbols == species)[0].astype(int, copy=False)


def _build_coordination_selection_cache(
    *,
    frame: Atoms,
    center_species: str,
    neighbor_species: str,
    center_indices: np.ndarray | None = None,
) -> _CoordinationSelectionCache:
    """Build stable selection metadata reused across one coordination run."""
    resolved_center_indices = (
        _select_indices(frame, center_species)
        if center_indices is None
        else np.asarray(center_indices, dtype=int)
    )
    resolved_neighbor_indices = _select_indices(frame, neighbor_species)
    same_selection = center_species == neighbor_species
    if same_selection and not np.array_equal(resolved_center_indices, resolved_neighbor_indices):
        raise ValueError(
            "Same-species coordination requires identical center and neighbor index ordering."
        )

    center_count = int(resolved_center_indices.size)
    neighbor_count = int(resolved_neighbor_indices.size)
    if center_count == 0:
        raise ValueError(
            f"No center atoms found for species '{center_species}' in the reference frame."
        )
    if neighbor_count == 0:
        raise ValueError(
            f"No neighbor atoms found for species '{neighbor_species}' in the reference frame."
        )

    center_lookup = np.full(len(frame), -1, dtype=int)
    center_lookup[resolved_center_indices] = np.arange(center_count, dtype=int)
    center_mask = np.zeros(len(frame), dtype=bool)
    center_mask[resolved_center_indices] = True
    neighbor_mask = np.zeros(len(frame), dtype=bool)
    neighbor_mask[resolved_neighbor_indices] = True

    return _CoordinationSelectionCache(
        center_indices=resolved_center_indices,
        neighbor_indices=resolved_neighbor_indices,
        center_lookup=center_lookup,
        center_mask=center_mask,
        neighbor_mask=neighbor_mask,
        same_selection=same_selection,
        center_count=center_count,
        neighbor_count=neighbor_count,
        pair_count=max(
            0,
            center_count * neighbor_count - (center_count if same_selection else 0),
        ),
    )


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _ensure_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return parsed


def _optional_cell_lengths(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None
    if len(items) < 3:
        return None
    parsed: list[float] = []
    for raw in items[:3]:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        parsed.append(numeric)
    return (parsed[0], parsed[1], parsed[2])


def _gaussian_kernel(*, sigma_bins: float) -> np.ndarray:
    sigma = max(float(sigma_bins), 1.0e-12)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)


def _gaussian_smooth(values: np.ndarray, *, sigma_bins: float) -> np.ndarray:
    if values.size <= 2:
        return np.asarray(values, dtype=float)
    if float(sigma_bins) <= 1.0e-12:
        return np.asarray(values, dtype=float)
    kernel = _gaussian_kernel(sigma_bins=sigma_bins)
    padded = np.pad(np.asarray(values, dtype=float), (kernel.size // 2,), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _find_first_peak_index(x: np.ndarray, y: np.ndarray) -> int:
    if x.size != y.size or x.size < 3:
        raise ValueError("Need at least three RDF points to resolve a peak.")
    candidates = [
        index
        for index in range(1, y.size - 1)
        if y[index] >= y[index - 1] and y[index] > y[index + 1]
    ]
    if not candidates:
        return int(np.argmax(y))
    positive_candidates = [index for index in candidates if x[index] > 0.0]
    return positive_candidates[0] if positive_candidates else candidates[0]


def _find_first_minimum_index(y: np.ndarray, *, start_index: int) -> int | None:
    if y.size < 3:
        raise ValueError("Need at least three RDF points to resolve a minimum.")
    candidates = [
        index
        for index in range(max(1, start_index + 1), y.size - 1)
        if y[index] <= y[index - 1] and y[index] < y[index + 1]
    ]
    if candidates:
        return candidates[0]
    if start_index + 1 >= y.size - 1:
        return None
    discrete_minimum = int(start_index + 1 + np.argmin(y[start_index + 1 :]))
    if discrete_minimum >= y.size - 1:
        return None
    return discrete_minimum


def _validate_resolved_cutoff(
    *,
    x: np.ndarray,
    smoothed: np.ndarray,
    peak_index: int,
    minimum_index: int | None,
    cutoff_A: float,
) -> None:
    if peak_index < 0 or peak_index >= x.size - 1:
        raise ValueError("Unable to resolve a valid first RDF peak before the cutoff region.")
    if minimum_index is None:
        raise ValueError("Unable to resolve a valid first RDF minimum after the first peak.")
    if minimum_index <= peak_index:
        raise ValueError("Resolved RDF minimum does not lie after the first peak.")
    if not np.isfinite(cutoff_A) or cutoff_A <= 0.0:
        raise ValueError("Resolved coordination cutoff must be finite and positive.")
    lower = float(np.min(x))
    upper = float(np.max(x))
    if cutoff_A < lower or cutoff_A > upper:
        raise ValueError("Resolved coordination cutoff lies outside the RDF range.")
    if cutoff_A <= float(x[peak_index]):
        raise ValueError("Resolved coordination cutoff must lie after the first RDF peak.")
    if not np.all(np.isfinite(smoothed)):
        raise ValueError("Smoothed RDF contains non-finite values in the cutoff-resolution window.")


def _fit_local_quadratic_minimum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center_index: int,
) -> float:
    start = max(0, center_index - _FIT_HALF_WINDOW_POINTS)
    stop = min(x.size, center_index + _FIT_HALF_WINDOW_POINTS + 1)
    x_window = np.asarray(x[start:stop], dtype=float)
    y_window = np.asarray(y[start:stop], dtype=float)
    if x_window.size < _MIN_FIT_POINTS:
        return float(x[center_index])

    coeffs = np.polyfit(x_window, y_window, deg=2)
    a, b, _c = coeffs
    if not np.isfinite(a) or abs(a) <= 1.0e-12 or a <= 0.0:
        return float(x[center_index])

    vertex = -b / (2.0 * a)
    lower = float(np.min(x_window))
    upper = float(np.max(x_window))
    if not np.isfinite(vertex):
        return float(x[center_index])
    return float(np.clip(vertex, lower, upper))


def _compute_reference_rdf_pairs(
    frames: list[Atoms],
    *,
    pairs: Sequence[tuple[str, str]],
    r_max: float | None,
    bin_width: float,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Compute full-trajectory reference RDF curves for one or more physical pairs."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    unique_pairs = _unique_physical_coordination_pairs(
        [(str(species_a), str(species_b)) for species_a, species_b in pairs]
    )
    if not unique_pairs:
        raise ValueError("At least one coordination pair is required.")

    (
        accumulated_pairs,
        bin_edges,
        _shell_volumes,
        counts_by_pair,
        expected_by_pair,
        _selection_cache_by_pair,
        _statistics_by_pair,
    ) = _accumulate_rdf_pair_collection(
        frames,
        pairs=unique_pairs,
        r_max=r_max,
        bin_width=float(bin_width),
        progress_desc="Coordination cutoff RDF",
    )
    if accumulated_pairs != unique_pairs:
        raise ValueError("Full coordination RDF accumulation changed pair ordering.")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    resolved: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for pair in unique_pairs:
        counts = np.asarray(counts_by_pair[pair], dtype=float)
        expected = np.asarray(expected_by_pair[pair], dtype=float)
        g_r = np.full_like(counts, np.nan, dtype=float)
        finite_mask = expected > 0.0
        g_r[finite_mask] = counts[finite_mask] / expected[finite_mask]
        resolved[pair] = (np.asarray(bin_centers, dtype=float), g_r)
    return resolved


def _resolve_reference_rdf_r_max(
    frames: list[Atoms],
    *,
    r_max: float | None,
) -> float:
    if r_max is not None:
        ensure_positive("r_max", r_max)
        return float(r_max)
    resolved_r_max = _auto_r_max_from_frames(frames)
    ensure_positive("r_max", resolved_r_max)
    return float(resolved_r_max)


def _build_reference_rdf_config(
    selected_frames: list[tuple[int, Atoms]],
    *,
    species_a: str,
    species_b: str,
    r_max: float | None,
    bin_width: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not selected_frames:
        raise ValueError("No frames were selected for RDF cutoff resolution.")
    ensure_positive("bin_width", bin_width)

    frame_objects = [frame for _frame_index, frame in selected_frames]
    resolved_r_max = _resolve_reference_rdf_r_max(frame_objects, r_max=r_max)
    n_bins = max(1, int(np.floor(float(resolved_r_max) / float(bin_width))))
    effective_r_max = float(n_bins) * float(bin_width)
    bin_edges = np.linspace(0.0, effective_r_max, n_bins + 1, dtype=float)
    return (
        bin_edges,
        {
            "species_a": _normalize_species(species_a),
            "species_b": _normalize_species(species_b),
            "r_max": float(effective_r_max),
            "bin_width": float(bin_width),
        },
    )


def _accumulate_reference_rdf_contributions(
    selected_frames: list[tuple[int, Atoms]],
    *,
    config: Mapping[str, Any] | Any,
    progress: ProgressBar | None = None,
    worker_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    del worker_count
    if isinstance(config, Mapping):
        resolved_species_a = str(config["species_a"])
        resolved_species_b = str(config["species_b"])
        resolved_r_max = float(config["r_max"])
        resolved_bin_width = float(config["bin_width"])
    else:
        resolved_species_a = str(config.species_a)
        resolved_species_b = str(config.species_b)
        resolved_r_max = float(config.r_max)
        resolved_bin_width = float(config.bin_width)

    if not selected_frames:
        n_bins = max(1, int(np.floor(float(resolved_r_max) / float(resolved_bin_width))))
        counts_accum = np.zeros(n_bins, dtype=float)
        return counts_accum, np.zeros_like(counts_accum)

    frame_objects = [frame for _frame_index, frame in selected_frames]
    (
        unique_pairs,
        bin_edges,
        _shell_volumes,
        counts_by_pair,
        expected_by_pair,
        _selection_cache,
        _statistics_by_pair,
    ) = _accumulate_rdf_pair_collection(
        frame_objects,
        pairs=[(resolved_species_a, resolved_species_b)],
        r_max=resolved_r_max,
        bin_width=resolved_bin_width,
        progress_desc=None,
    )
    pair = _canonical_rdf_pair(resolved_species_a, resolved_species_b)
    if progress is not None:
        progress.update(len(selected_frames))
    if unique_pairs != [pair]:
        raise ValueError("Reference RDF accumulation returned unexpected pair selection.")
    return (
        np.asarray(counts_by_pair[pair], dtype=float),
        np.asarray(expected_by_pair[pair], dtype=float),
    )


def _compute_reference_rdf(
    frames: list[Atoms],
    *,
    species_a: str,
    species_b: str,
    frame_indices: np.ndarray,
    r_max: float | None,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    selected_frames = [(int(index), frames[int(index)]) for index in frame_indices.tolist()]
    bin_edges, config = _build_reference_rdf_config(
        selected_frames,
        species_a=species_a,
        species_b=species_b,
        r_max=r_max,
        bin_width=float(bin_width),
    )
    with ProgressBar(
        desc=f"Coordination cutoff RDF ({species_a}-{species_b})",
        total=len(selected_frames),
        unit="frame",
    ) as progress:
        counts_accum, expected_accum = _accumulate_reference_rdf_contributions(
            selected_frames,
            config=config,
            progress=progress,
        )

    g_r = np.full_like(counts_accum, np.nan, dtype=float)
    non_zero = expected_accum > 0.0
    g_r[non_zero] = counts_accum[non_zero] / expected_accum[non_zero]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return np.asarray(bin_centers, dtype=float), np.asarray(g_r, dtype=float)


def _resolve_cutoff_from_rdf_curve(
    *,
    bin_centers_A: np.ndarray,
    g_r: np.ndarray,
    smoothing_sigma_A: float,
) -> tuple[np.ndarray, float, float]:
    if bin_centers_A.size != g_r.size:
        raise ValueError("RDF bin centers and g(r) arrays must have matching sizes.")
    finite_mask = np.isfinite(bin_centers_A) & np.isfinite(g_r)
    finite_bin_centers = np.asarray(bin_centers_A[finite_mask], dtype=float)
    finite_g_r = np.asarray(g_r[finite_mask], dtype=float)
    if finite_bin_centers.size < 3:
        raise ValueError("Need at least three finite RDF bins to resolve a coordination cutoff.")

    diffs = np.diff(finite_bin_centers)
    if diffs.size == 0:
        raise ValueError("Need at least two RDF bins to resolve smoothing width.")
    if np.any(~np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError("RDF bin centers must be strictly increasing in the valid cutoff region.")
    mean_bin_width = float(np.mean(diffs))
    sigma_bins = float(smoothing_sigma_A) / max(mean_bin_width, 1.0e-12)
    smoothed = _gaussian_smooth(finite_g_r, sigma_bins=sigma_bins)
    peak_index = _find_first_peak_index(finite_bin_centers, smoothed)
    minimum_index = _find_first_minimum_index(smoothed, start_index=peak_index)
    if minimum_index is None:
        raise ValueError("Unable to resolve a valid first RDF minimum after the first peak.")
    cutoff_A = _fit_local_quadratic_minimum(
        finite_bin_centers,
        np.asarray(smoothed, dtype=float),
        center_index=minimum_index,
    )
    _validate_resolved_cutoff(
        x=finite_bin_centers,
        smoothed=np.asarray(smoothed, dtype=float),
        peak_index=peak_index,
        minimum_index=minimum_index,
        cutoff_A=cutoff_A,
    )
    full_smoothed = np.full(g_r.shape, np.nan, dtype=float)
    full_smoothed[finite_mask] = smoothed
    return full_smoothed, float(finite_bin_centers[peak_index]), cutoff_A


def _save_cutoff_diagnostic_plot(
    *,
    output: str | Path,
    bin_centers_A: np.ndarray,
    g_r: np.ndarray,
    g_r_smoothed: np.ndarray,
    peak_A: float,
    minimum_A: float,
    species_a: str,
    species_b: str,
) -> Path:
    import matplotlib

    pyplot_module = matplotlib.pyplot if hasattr(matplotlib, "pyplot") else None
    if pyplot_module is None:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=DEFAULT_PLOT_STYLE.figure_size)
    ax.plot(bin_centers_A, g_r, label="RDF", color="#4c6a92", lw=1.6)
    ax.plot(bin_centers_A, g_r_smoothed, label="Smoothed RDF", color="#d07b29", lw=2.2)
    ax.axvline(minimum_A, color="#b22222", linestyle="--", lw=1.5, label="Cutoff")
    ax.scatter(
        [peak_A],
        [float(np.interp(peak_A, bin_centers_A, g_r_smoothed))],
        color="#2f855a",
        zorder=3,
    )
    ax.scatter(
        [minimum_A],
        [float(np.interp(minimum_A, bin_centers_A, g_r_smoothed))],
        color="#b22222",
        zorder=3,
    )
    ax.set_xlabel(
        format_axis_label_units("r (Angstrom)"),
        fontsize=DEFAULT_PLOT_STYLE.label_font_size,
    )
    ax.set_ylabel("g(r)", fontsize=DEFAULT_PLOT_STYLE.label_font_size)
    ax.set_title(
        f"{species_a}-{species_b} cutoff reference RDF",
        fontsize=DEFAULT_PLOT_STYLE.title_font_size,
    )
    ax.grid(
        DEFAULT_PLOT_STYLE.grid,
        linestyle=DEFAULT_PLOT_STYLE.grid_linestyle,
        linewidth=DEFAULT_PLOT_STYLE.grid_linewidth,
        alpha=DEFAULT_PLOT_STYLE.grid_alpha,
    )
    ax.legend(fontsize=DEFAULT_PLOT_STYLE.legend_font_size, loc="best")
    ax.tick_params(axis="both", labelsize=DEFAULT_PLOT_STYLE.tick_font_size)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_PLOT_STYLE.dpi)
    plt.close(fig)
    return output_path


def resolve_coordination_cutoff(
    *,
    frames: list[Atoms],
    species_a: str | None,
    species_b: str | None,
    cutoff_A: float | None,
    cutoff_rdf_path: str | Path | None,
    cutoff_from_rdf: bool,
    cutoff_smoothing_width_A: float = _DEFAULT_CUTOFF_SMOOTHING_WIDTH_A,
    diagnostic_plot_output: str | Path | None = None,
) -> CoordinationCutoffResolution:
    label_a = _normalize_species(species_a)
    label_b = _normalize_species(species_b if species_b is not None else species_a)
    resolutions = resolve_coordination_cutoffs(
        frames=frames,
        ordered_pairs=[(label_a, label_b)],
        cutoff_A=cutoff_A,
        cutoff_rdf_path=cutoff_rdf_path,
        cutoff_from_rdf=cutoff_from_rdf,
        cutoff_smoothing_width_A=cutoff_smoothing_width_A,
        diagnostic_plot_outputs=(
            None if diagnostic_plot_output is None else {(label_a, label_b): diagnostic_plot_output}
        ),
    )
    return resolutions[(label_a, label_b)]


def resolve_coordination_cutoffs(
    *,
    frames: list[Atoms],
    ordered_pairs: Sequence[tuple[str, str]],
    cutoff_A: float | None,
    cutoff_rdf_path: str | Path | None,
    cutoff_from_rdf: bool,
    cutoff_smoothing_width_A: float = _DEFAULT_CUTOFF_SMOOTHING_WIDTH_A,
    diagnostic_plot_outputs: Mapping[tuple[str, str], str | Path] | None = None,
) -> dict[tuple[str, str], CoordinationCutoffResolution]:
    cutoff_smoothing_width_A = _ensure_nonnegative(
        "cutoff_smoothing_width_A", cutoff_smoothing_width_A
    )
    normalized_pairs = [
        (_normalize_species(species_a), _normalize_species(species_b))
        for species_a, species_b in ordered_pairs
    ]
    if not normalized_pairs:
        raise ValueError("At least one coordination pair is required.")
    unique_physical_pairs = _unique_physical_coordination_pairs(normalized_pairs)
    diagnostic_outputs = {
        (_normalize_species(species_a), _normalize_species(species_b)): output
        for (species_a, species_b), output in (diagnostic_plot_outputs or {}).items()
    }

    if cutoff_A is not None:
        ensure_positive("cutoff_A", cutoff_A)
        return {
            pair: CoordinationCutoffResolution(
                cutoff_A=float(cutoff_A),
                smoothing_width_A=float(cutoff_smoothing_width_A),
                mode="direct",
            )
            for pair in normalized_pairs
        }

    if cutoff_rdf_path is not None:
        from .rdf import load_rdf_profile

        source_path = Path(cutoff_rdf_path).expanduser().resolve()
        resolved_by_physical_pair: dict[tuple[str, str], CoordinationCutoffResolution] = {}
        for physical_pair in unique_physical_pairs:
            rdf_profile = load_rdf_profile(
                source_path,
                species_a=physical_pair[0],
                species_b=physical_pair[1],
            )
            smoothed, peak_A, minimum_A = _resolve_cutoff_from_rdf_curve(
                bin_centers_A=np.asarray(rdf_profile.bin_centers, dtype=float),
                g_r=np.asarray(rdf_profile.g_r, dtype=float),
                smoothing_sigma_A=float(_DEFAULT_RDF_SMOOTHING_SIGMA_A),
            )
            resolved_by_physical_pair[physical_pair] = CoordinationCutoffResolution(
                cutoff_A=float(minimum_A),
                smoothing_width_A=float(cutoff_smoothing_width_A),
                mode="rdf_file",
                rdf_bin_centers_A=np.asarray(rdf_profile.bin_centers, dtype=float),
                rdf_g_r=np.asarray(rdf_profile.g_r, dtype=float),
                rdf_g_r_smoothed=smoothed,
                rdf_peak_A=peak_A,
                rdf_minimum_A=float(minimum_A),
                rdf_source_path=str(source_path),
            )

        resolved_by_ordered_pair: dict[tuple[str, str], CoordinationCutoffResolution] = {}
        for pair in normalized_pairs:
            physical_pair = _canonical_rdf_pair(*pair)
            base = resolved_by_physical_pair[physical_pair]
            if base.rdf_peak_A is None or base.rdf_minimum_A is None:
                raise ValueError(
                    f"RDF file cutoff resolution for pair {pair[0]}-{pair[1]} is incomplete."
                )
            diagnostic_path = None
            diagnostic_output = diagnostic_outputs.get(pair)
            if diagnostic_output is not None:
                diagnostic_path = _save_cutoff_diagnostic_plot(
                    output=diagnostic_output,
                    bin_centers_A=np.asarray(base.rdf_bin_centers_A, dtype=float),
                    g_r=np.asarray(base.rdf_g_r, dtype=float),
                    g_r_smoothed=np.asarray(base.rdf_g_r_smoothed, dtype=float),
                    peak_A=float(base.rdf_peak_A),
                    minimum_A=float(base.rdf_minimum_A),
                    species_a=pair[0],
                    species_b=pair[1],
                )
            resolved_by_ordered_pair[pair] = CoordinationCutoffResolution(
                cutoff_A=float(base.cutoff_A),
                smoothing_width_A=float(base.smoothing_width_A),
                mode=base.mode,
                rdf_bin_centers_A=np.asarray(base.rdf_bin_centers_A, dtype=float),
                rdf_g_r=np.asarray(base.rdf_g_r, dtype=float),
                rdf_g_r_smoothed=np.asarray(base.rdf_g_r_smoothed, dtype=float),
                rdf_peak_A=base.rdf_peak_A,
                rdf_minimum_A=base.rdf_minimum_A,
                rdf_source_path=base.rdf_source_path,
                diagnostic_plot_path=None if diagnostic_path is None else str(diagnostic_path),
            )
        return resolved_by_ordered_pair

    LOGGER.info(
        "Resolving coordination cutoff from full-trajectory RDF for %d physical pair(s).",
        len(unique_physical_pairs),
    )
    resolved_curves = _compute_reference_rdf_pairs(
        frames,
        pairs=unique_physical_pairs,
        r_max=None,
        bin_width=float(_DEFAULT_RDF_BIN_WIDTH_A),
    )
    resolved_by_ordered_pair = {}
    for pair in normalized_pairs:
        physical_pair = _canonical_rdf_pair(*pair)
        bin_centers_A, g_r = resolved_curves[physical_pair]
        smoothed, peak_A, minimum_A = _resolve_cutoff_from_rdf_curve(
            bin_centers_A=np.asarray(bin_centers_A, dtype=float),
            g_r=np.asarray(g_r, dtype=float),
            smoothing_sigma_A=float(_DEFAULT_RDF_SMOOTHING_SIGMA_A),
        )
        diagnostic_path = None
        diagnostic_output = diagnostic_outputs.get(pair)
        if diagnostic_output is not None:
            diagnostic_path = _save_cutoff_diagnostic_plot(
                output=diagnostic_output,
                bin_centers_A=bin_centers_A,
                g_r=g_r,
                g_r_smoothed=smoothed,
                peak_A=peak_A,
                minimum_A=minimum_A,
                species_a=pair[0],
                species_b=pair[1],
            )
        resolved_by_ordered_pair[pair] = CoordinationCutoffResolution(
            cutoff_A=float(minimum_A),
            smoothing_width_A=float(cutoff_smoothing_width_A),
            mode="full_rdf",
            rdf_bin_centers_A=np.asarray(bin_centers_A, dtype=float),
            rdf_g_r=np.asarray(g_r, dtype=float),
            rdf_g_r_smoothed=np.asarray(smoothed, dtype=float),
            rdf_peak_A=peak_A,
            rdf_minimum_A=float(minimum_A),
            diagnostic_plot_path=None if diagnostic_path is None else str(diagnostic_path),
        )
        LOGGER.info(
            "Resolved coordination cutoff for %s-%s from full RDF: cutoff=%.6g A (peak=%.6g A).",
            pair[0],
            pair[1],
            float(minimum_A),
            float(peak_A),
        )
    return resolved_by_ordered_pair


def _continuous_coordination_weights(
    distances: np.ndarray,
    *,
    cutoff_A: float,
    smoothing_width_A: float,
) -> np.ndarray:
    """Return cosine-taper coordination weights for pair distances.

    With ``Delta = smoothing_width_A`` the weight is:
    - ``1`` for ``r <= cutoff_A - Delta/2``
    - ``0.5 * (1 + cos(pi * (r - (cutoff_A - Delta/2)) / Delta))`` inside the
      transition interval
    - ``0`` for ``r >= cutoff_A + Delta/2``

    ``Delta <= eps`` is treated as the hard-cutoff limit.
    """
    distances = np.asarray(distances, dtype=float)
    weights = np.zeros(distances.shape, dtype=float)
    half_width = 0.5 * float(smoothing_width_A)
    lower = float(cutoff_A) - half_width
    upper = float(cutoff_A) + half_width
    if half_width <= 1.0e-12:
        weights[distances <= float(cutoff_A)] = 1.0
        return weights

    inside = distances <= lower
    transition = (distances > lower) & (distances < upper)
    weights[inside] = 1.0
    if np.any(transition):
        scaled = (distances[transition] - lower) / max(upper - lower, 1.0e-12)
        weights[transition] = 0.5 * (1.0 + np.cos(np.pi * scaled))
    return weights


def _deduplicate_ordered_pairs(
    i_selected: np.ndarray,
    j_selected: np.ndarray,
    distances: np.ndarray,
    *,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distances.size <= 1:
        return i_selected, j_selected, distances
    pair_ids = i_selected.astype(np.int64) * int(n_atoms) + j_selected.astype(np.int64)
    order = np.argsort(pair_ids, kind="mergesort")
    sorted_ids = pair_ids[order]
    sorted_i = i_selected[order]
    sorted_j = j_selected[order]
    sorted_d = distances[order]
    unique_starts = np.empty(sorted_ids.size, dtype=bool)
    unique_starts[0] = True
    unique_starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
    starts = np.flatnonzero(unique_starts)
    dedup_i = sorted_i[starts]
    dedup_j = sorted_j[starts]
    dedup_d = np.minimum.reduceat(sorted_d, starts)
    return dedup_i, dedup_j, dedup_d


def _expand_time_distance_render_layers(
    profiles: list[CoordinationProfile],
    *,
    series_ids: list[str] | None,
    render_series_descriptors: list[dict[str, Any]] | None,
    series_enabled: list[bool] | None,
    series_line_widths: list[float | None] | None,
    series_markers: list[str | None] | None,
    series_normalization_modes: list[str | None] | None,
    series_normalization_values: list[float | None] | None,
    series_normalization_x_refs: list[float | None] | None,
) -> tuple[
    list[CoordinationProfile],
    list[bool] | None,
    list[float | None] | None,
    list[str | None] | None,
    list[str | None] | None,
    list[float | None] | None,
    list[float | None] | None,
]:
    if not render_series_descriptors:
        return (
            profiles,
            series_enabled,
            series_line_widths,
            series_markers,
            series_normalization_modes,
            series_normalization_values,
            series_normalization_x_refs,
        )

    source_ids = (
        [str(series_id) for series_id in series_ids]
        if series_ids is not None
        else [f"series:{index}" for index in range(len(profiles))]
    )
    if len(source_ids) != len(profiles):
        return (
            profiles,
            series_enabled,
            series_line_widths,
            series_markers,
            series_normalization_modes,
            series_normalization_values,
            series_normalization_x_refs,
        )

    source_by_id = {
        series_id: (index, profile)
        for index, (series_id, profile) in enumerate(zip(source_ids, profiles))
    }
    render_indices: list[int] = []
    source_indices: list[int] = []
    expanded_profiles: list[CoordinationProfile] = []
    for render_index, descriptor in enumerate(render_series_descriptors):
        if str(descriptor.get("source_kind") or "source").strip().lower() == "group":
            continue
        source_id = str(
            descriptor.get("source_series_id") or descriptor.get("series_id") or ""
        ).strip()
        source_entry = source_by_id.get(source_id)
        if source_entry is None:
            continue
        source_index, source_profile = source_entry
        render_indices.append(render_index)
        source_indices.append(source_index)
        expanded_profiles.append(source_profile)

    if not expanded_profiles:
        return (
            profiles,
            series_enabled,
            series_line_widths,
            series_markers,
            series_normalization_modes,
            series_normalization_values,
            series_normalization_x_refs,
        )

    def _expand_values(values: list[Any] | None) -> list[Any] | None:
        if values is None:
            return None
        if len(values) == len(render_series_descriptors):
            return [values[index] for index in render_indices]
        if len(values) == len(profiles):
            return [values[index] for index in source_indices]
        if len(values) == len(expanded_profiles):
            return list(values)
        return values

    return (
        expanded_profiles,
        _expand_values(series_enabled),
        _expand_values(series_line_widths),
        _expand_values(series_markers),
        _expand_values(series_normalization_modes),
        _expand_values(series_normalization_values),
        _expand_values(series_normalization_x_refs),
    )


def _compute_coordination_frame_values(
    frame: Atoms,
    *,
    selection_cache: _CoordinationSelectionCache | None = None,
    center_indices: np.ndarray | None = None,
    neighbor_indices: np.ndarray | None = None,
    same_selection: bool | None = None,
    cutoff_A: float,
    smoothing_width_A: float,
) -> np.ndarray:
    if selection_cache is None:
        if center_indices is None or neighbor_indices is None or same_selection is None:
            raise TypeError(
                "Provide either selection_cache or center_indices/neighbor_indices/same_selection."
            )
        resolved_center_indices = np.asarray(center_indices, dtype=int)
        resolved_neighbor_indices = np.asarray(neighbor_indices, dtype=int)
        if same_selection and not np.array_equal(
            resolved_center_indices, resolved_neighbor_indices
        ):
            raise ValueError(
                "Same-species coordination requires identical center and neighbor index ordering."
            )
        center_lookup = np.full(len(frame), -1, dtype=int)
        center_lookup[resolved_center_indices] = np.arange(resolved_center_indices.size, dtype=int)
        center_mask = np.zeros(len(frame), dtype=bool)
        center_mask[resolved_center_indices] = True
        neighbor_mask = np.zeros(len(frame), dtype=bool)
        neighbor_mask[resolved_neighbor_indices] = True
        selection_cache = _CoordinationSelectionCache(
            center_indices=resolved_center_indices,
            neighbor_indices=resolved_neighbor_indices,
            center_lookup=center_lookup,
            center_mask=center_mask,
            neighbor_mask=neighbor_mask,
            same_selection=bool(same_selection),
            center_count=int(resolved_center_indices.size),
            neighbor_count=int(resolved_neighbor_indices.size),
            pair_count=max(
                0,
                int(resolved_center_indices.size) * int(resolved_neighbor_indices.size)
                - (int(resolved_center_indices.size) if same_selection else 0),
            ),
        )

    n_centers = int(selection_cache.center_count)
    if n_centers == 0:
        raise ValueError("Coordination calculation requires at least one center atom.")
    support_cutoff = float(cutoff_A) + 0.5 * float(smoothing_width_A)
    if support_cutoff <= 0.0:
        raise ValueError("Coordination support cutoff must be positive.")

    use_neighbor_list = _frame_has_usable_cell(frame)
    weights_accum = np.zeros(n_centers, dtype=float)

    if use_neighbor_list:
        i_pairs, j_pairs, pair_distances = neighbor_list(
            "ijd",
            frame,
            float(np.nextafter(support_cutoff, np.inf)),
        )
        mask = selection_cache.center_mask[i_pairs] & selection_cache.neighbor_mask[j_pairs]
        if selection_cache.same_selection:
            mask &= i_pairs != j_pairs
        if np.any(mask):
            i_selected = np.asarray(i_pairs[mask], dtype=int)
            j_selected = np.asarray(j_pairs[mask], dtype=int)
            distances = np.asarray(pair_distances[mask], dtype=float)
            i_selected, j_selected, distances = _deduplicate_ordered_pairs(
                i_selected,
                j_selected,
                distances,
                n_atoms=len(frame),
            )
            weights = _continuous_coordination_weights(
                distances,
                cutoff_A=float(cutoff_A),
                smoothing_width_A=float(smoothing_width_A),
            )
            np.add.at(weights_accum, selection_cache.center_lookup[i_selected], weights)
        return weights_accum

    center_indices = selection_cache.center_indices
    neighbor_indices = selection_cache.neighbor_indices
    if selection_cache.same_selection:
        pair_distances = np.asarray(frame.get_all_distances(mic=False), dtype=float)[
            np.ix_(center_indices, neighbor_indices)
        ]
        pair_distances = pair_distances.copy()
        pair_distances[np.eye(n_centers, dtype=bool)] = np.inf
    else:
        center_positions = np.asarray(frame.positions[center_indices], dtype=float)
        neighbor_positions = np.asarray(frame.positions[neighbor_indices], dtype=float)
        deltas = center_positions[:, np.newaxis, :] - neighbor_positions[np.newaxis, :, :]
        pair_distances = np.linalg.norm(deltas, axis=2)

    weights_matrix = _continuous_coordination_weights(
        pair_distances,
        cutoff_A=float(cutoff_A),
        smoothing_width_A=float(smoothing_width_A),
    )
    return np.sum(weights_matrix, axis=1)


def _resolve_coordination_chunk_size(
    *,
    frame_count: int,
    pair_count: int,
) -> int:
    if frame_count <= 0:
        return 1
    if pair_count <= 0:
        return frame_count
    max_chunk = max(1, _COORD_MAX_DISTANCE_VALUES_PER_CHUNK // pair_count)
    return max(1, min(frame_count, max_chunk))


def _can_vectorize_coordination_kernel(
    frames: list[Atoms],
    *,
    pair_count: int,
) -> bool:
    if pair_count <= 0 or pair_count > _COORD_DENSE_PAIR_THRESHOLD:
        return False
    if not frames:
        return False
    usable_pbc = [_frame_has_usable_cell(frame) for frame in frames]
    if all(usable_pbc):
        return all(_frame_has_axis_aligned_orthorhombic_cell(frame) for frame in frames)
    return not any(usable_pbc)


def _frame_has_axis_aligned_orthorhombic_cell(frame: Atoms) -> bool:
    if not _frame_has_usable_cell(frame):
        return False
    cell = np.asarray(frame.cell.array, dtype=float)
    diagonal = np.diag(np.diag(cell))
    if not np.allclose(cell, diagonal, rtol=0.0, atol=1.0e-12):
        return False
    return bool(np.all(np.diag(diagonal) > 0.0))


def _validate_coordination_matrix(
    values: np.ndarray,
    *,
    frame_count: int,
    center_count: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (frame_count, center_count):
        raise ValueError(
            "Coordination matrix shape mismatch: "
            f"expected {(frame_count, center_count)}, got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Coordination matrix contains non-finite values.")
    if np.any(matrix < 0.0):
        raise ValueError("Coordination matrix contains negative values.")
    return matrix


def _compute_coordination_values_chunked(
    frames: list[Atoms],
    *,
    selection_cache: _CoordinationSelectionCache | None = None,
    center_indices: np.ndarray | None = None,
    neighbor_indices: np.ndarray | None = None,
    same_selection: bool | None = None,
    cutoff_A: float,
    smoothing_width_A: float,
) -> np.ndarray:
    if selection_cache is None:
        if center_indices is None or neighbor_indices is None or same_selection is None:
            raise TypeError(
                "Provide either selection_cache or center_indices/neighbor_indices/same_selection."
            )
        if not frames:
            raise ValueError("At least one trajectory frame is required.")
        resolved_center_indices = np.asarray(center_indices, dtype=int)
        resolved_neighbor_indices = np.asarray(neighbor_indices, dtype=int)
        if same_selection and not np.array_equal(
            resolved_center_indices, resolved_neighbor_indices
        ):
            raise ValueError(
                "Same-species coordination requires identical center and neighbor index ordering."
            )
        center_lookup = np.full(len(frames[0]), -1, dtype=int)
        center_lookup[resolved_center_indices] = np.arange(resolved_center_indices.size, dtype=int)
        center_mask = np.zeros(len(frames[0]), dtype=bool)
        center_mask[resolved_center_indices] = True
        neighbor_mask = np.zeros(len(frames[0]), dtype=bool)
        neighbor_mask[resolved_neighbor_indices] = True
        selection_cache = _CoordinationSelectionCache(
            center_indices=resolved_center_indices,
            neighbor_indices=resolved_neighbor_indices,
            center_lookup=center_lookup,
            center_mask=center_mask,
            neighbor_mask=neighbor_mask,
            same_selection=bool(same_selection),
            center_count=int(resolved_center_indices.size),
            neighbor_count=int(resolved_neighbor_indices.size),
            pair_count=max(
                0,
                int(resolved_center_indices.size) * int(resolved_neighbor_indices.size)
                - (int(resolved_center_indices.size) if same_selection else 0),
            ),
        )

    frame_count = len(frames)
    center_indices = selection_cache.center_indices
    neighbor_indices = selection_cache.neighbor_indices
    center_count = int(selection_cache.center_count)
    neighbor_count = int(selection_cache.neighbor_count)
    pair_count = int(selection_cache.pair_count)
    chunk_size = _resolve_coordination_chunk_size(frame_count=frame_count, pair_count=pair_count)
    use_mic = bool(frames) and all(_frame_has_usable_cell(frame) for frame in frames)

    coordination = np.zeros((frame_count, center_count), dtype=np.float32)
    with ProgressBar(
        desc=f"Computing coordination ({center_count}x{neighbor_count})",
        total=frame_count,
        unit="frame",
    ) as progress:
        for start in range(0, frame_count, chunk_size):
            stop = min(frame_count, start + chunk_size)
            center_positions = np.stack(
                [
                    np.asarray(frame.positions[center_indices], dtype=float)
                    for frame in frames[start:stop]
                ],
                axis=0,
            )
            neighbor_positions = np.stack(
                [
                    np.asarray(frame.positions[neighbor_indices], dtype=float)
                    for frame in frames[start:stop]
                ],
                axis=0,
            )
            deltas = center_positions[:, :, np.newaxis, :] - neighbor_positions[:, np.newaxis, :, :]
            if use_mic:
                cell_lengths = np.stack(
                    [
                        np.asarray(np.diag(frame.cell.array), dtype=float)
                        for frame in frames[start:stop]
                    ],
                    axis=0,
                )
                deltas -= cell_lengths[:, np.newaxis, np.newaxis, :] * np.round(
                    deltas / cell_lengths[:, np.newaxis, np.newaxis, :]
                )
            distances = np.linalg.norm(deltas, axis=3)
            if selection_cache.same_selection:
                diagonal = np.arange(center_count, dtype=int)
                distances[:, diagonal, diagonal] = np.inf
            weights = _continuous_coordination_weights(
                distances,
                cutoff_A=float(cutoff_A),
                smoothing_width_A=float(smoothing_width_A),
            )
            coordination[start:stop, :] = np.sum(weights, axis=2, dtype=float).astype(
                np.float32,
                copy=False,
            )
            progress.update(stop - start)
    return coordination


def _compute_coordination_profile_from_position_profile(
    frames: list[Atoms],
    *,
    position_profile: Any,
    species_b: str,
    cutoff_resolution: CoordinationCutoffResolution,
) -> CoordinationProfile:
    """Compute one coordination profile reusing a precomputed center position profile."""
    label_a = _normalize_species(str(position_profile.species))
    label_b = _normalize_species(species_b)

    selection_cache = _build_coordination_selection_cache(
        frame=frames[0],
        center_species=label_a,
        neighbor_species=label_b,
        center_indices=np.asarray(position_profile.atom_indices, dtype=int),
    )
    center_indices = selection_cache.center_indices
    center_count = int(selection_cache.center_count)
    neighbor_count = int(selection_cache.neighbor_count)
    pair_count = int(selection_cache.pair_count)
    if _can_vectorize_coordination_kernel(frames, pair_count=pair_count):
        backend = _COORD_BACKEND_DENSE
        coordination = _compute_coordination_values_chunked(
            frames,
            selection_cache=selection_cache,
            cutoff_A=float(cutoff_resolution.cutoff_A),
            smoothing_width_A=float(cutoff_resolution.smoothing_width_A),
        )
    else:
        backend = (
            _COORD_BACKEND_NEIGHBOR
            if bool(frames) and all(_frame_has_usable_cell(frame) for frame in frames)
            else _COORD_BACKEND_GENERIC
        )
        coordination = np.zeros(
            (len(frames), center_count),
            dtype=np.float32,
        )
        with ProgressBar(
            desc=f"Computing coordination ({center_count}x{neighbor_count})",
            total=len(frames),
            unit="frame",
        ) as progress:
            for frame_index, frame in enumerate(frames):
                coordination[frame_index, :] = _compute_coordination_frame_values(
                    frame,
                    selection_cache=selection_cache,
                    cutoff_A=float(cutoff_resolution.cutoff_A),
                    smoothing_width_A=float(cutoff_resolution.smoothing_width_A),
                )
                progress.update()

    LOGGER.info(
        "Computed coordination for %s-%s over %d frame(s) using %s (pair_count=%d, cutoff=%.6g A).",
        label_a,
        label_b,
        len(frames),
        backend,
        pair_count,
        float(cutoff_resolution.cutoff_A),
    )

    coordination = _validate_coordination_matrix(
        coordination,
        frame_count=len(frames),
        center_count=center_count,
    )
    distance_to_surface = np.asarray(position_profile.distance_to_surface, dtype=float)
    if distance_to_surface.shape != coordination.shape:
        raise ValueError(
            "Coordination/position alignment failed: distance-to-surface shape "
            f"{distance_to_surface.shape} does not match coordination shape {coordination.shape}."
        )
    if not np.array_equal(center_indices, np.asarray(position_profile.atom_indices, dtype=int)):
        raise ValueError(
            "Coordination center ordering no longer matches the tracked position profile."
        )

    return CoordinationProfile(
        species_a=label_a,
        species_b=label_b,
        axis=str(position_profile.axis),
        atom_indices=center_indices,
        frame_index=np.asarray(position_profile.frame_index, dtype=int),
        step=np.asarray(position_profile.step, dtype=float),
        time_fs=np.asarray(position_profile.time_fs, dtype=float),
        time_ps=np.asarray(position_profile.time_ps, dtype=float),
        distance_to_surface=np.asarray(distance_to_surface, dtype=np.float32),
        coordination_number=np.asarray(coordination, dtype=np.float32),
        n_frames=int(position_profile.n_frames),
        n_atoms=int(position_profile.n_atoms),
        coordinate_mode=str(position_profile.coordinate_mode),
        surface_position=position_profile.surface_position,
        surface_position_std=position_profile.surface_position_std,
        surface_position_per_frame=(
            None
            if position_profile.surface_position_per_frame is None
            else np.asarray(position_profile.surface_position_per_frame, dtype=float)
        ),
        surface_estimate=position_profile.surface_estimate,
        cell_lengths_angstrom=position_profile.cell_lengths_angstrom,
        cutoff_A=float(cutoff_resolution.cutoff_A),
        cutoff_smoothing_width_A=float(cutoff_resolution.smoothing_width_A),
        cutoff_mode=str(cutoff_resolution.mode),
        cutoff_rdf_bin_centers_A=(
            None
            if cutoff_resolution.rdf_bin_centers_A is None
            else np.asarray(cutoff_resolution.rdf_bin_centers_A, dtype=float)
        ),
        cutoff_rdf_g_r=(
            None
            if cutoff_resolution.rdf_g_r is None
            else np.asarray(cutoff_resolution.rdf_g_r, dtype=float)
        ),
        cutoff_rdf_g_r_smoothed=(
            None
            if cutoff_resolution.rdf_g_r_smoothed is None
            else np.asarray(cutoff_resolution.rdf_g_r_smoothed, dtype=float)
        ),
        cutoff_rdf_peak_A=cutoff_resolution.rdf_peak_A,
        cutoff_rdf_minimum_A=cutoff_resolution.rdf_minimum_A,
        cutoff_rdf_sampled_frame_index=(
            None
            if cutoff_resolution.rdf_sampled_frame_index is None
            else np.asarray(cutoff_resolution.rdf_sampled_frame_index, dtype=int)
        ),
        cutoff_rdf_source_path=cutoff_resolution.rdf_source_path,
        cutoff_diagnostic_plot_path=cutoff_resolution.diagnostic_plot_path,
    )


def compute_coordination_profile(
    frames: list[Atoms],
    *,
    species_a: str | None = "all",
    species_b: str | None = None,
    axis: str = "z",
    timestep_fs: float = 1.0,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
    cutoff_resolution: CoordinationCutoffResolution,
) -> CoordinationProfile:
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    axis_label = _normalize_axis(axis)
    label_a = _normalize_species(species_a)
    label_b = _normalize_species(species_b if species_b is not None else species_a)
    ensure_positive("timestep_fs", timestep_fs)

    LOGGER.info(
        "Resolving coordination center trajectories for %d frame(s) using species_a=%s.",
        len(frames),
        label_a,
    )
    position_profile = compute_position_profile(
        frames,
        species=label_a,
        axis=axis_label,
        timestep_fs=float(timestep_fs),
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
    )
    return _compute_coordination_profile_from_position_profile(
        frames,
        position_profile=position_profile,
        species_b=label_b,
        cutoff_resolution=cutoff_resolution,
    )


def compute_coordination_profiles(
    frames: list[Atoms],
    *,
    ordered_pairs: Sequence[tuple[str, str]] | None = None,
    axis: str = "z",
    timestep_fs: float = 1.0,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
    precomputed_surface_estimate: SurfaceEstimate | None = None,
    cutoff_resolutions: Mapping[tuple[str, str], CoordinationCutoffResolution],
) -> list[CoordinationProfile]:
    """Compute ordered coordination profiles while reusing center trajectories."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    axis_label = _normalize_axis(axis)
    ensure_positive("timestep_fs", timestep_fs)
    resolved_pairs = (
        [
            (_normalize_species(species_a), _normalize_species(species_b))
            for species_a, species_b in ordered_pairs
        ]
        if ordered_pairs is not None
        else _ordered_coordination_pairs_from_frames(frames)
    )
    if not resolved_pairs:
        raise ValueError("At least one coordination pair is required.")

    center_species_in_order: list[str] = []
    for species_a, _species_b in resolved_pairs:
        if species_a not in center_species_in_order:
            center_species_in_order.append(species_a)

    position_profiles_by_species: dict[str, Any] = {}
    for species_a in center_species_in_order:
        LOGGER.info(
            "Resolving coordination center trajectories for %d frame(s) using species_a=%s.",
            len(frames),
            species_a,
        )
        position_profiles_by_species[species_a] = compute_position_profile(
            frames,
            species=species_a,
            axis=axis_label,
            timestep_fs=float(timestep_fs),
            surface_mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            surface_options=surface_options,
            precomputed_surface_estimate=precomputed_surface_estimate,
        )

    profiles: list[CoordinationProfile] = []
    for pair in resolved_pairs:
        if pair not in cutoff_resolutions:
            raise ValueError(
                f"Missing coordination cutoff resolution for pair {pair[0]}-{pair[1]}."
            )
        profiles.append(
            _compute_coordination_profile_from_position_profile(
                frames,
                position_profile=position_profiles_by_species[pair[0]],
                species_b=pair[1],
                cutoff_resolution=cutoff_resolutions[pair],
            )
        )
    return profiles


def save_coordination_profile(
    profile: CoordinationProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    metadata = build_profile_metadata(
        analysis="coordination",
        metadata={
            "species_a": profile.species_a,
            "species_b": profile.species_b,
            "axis": profile.axis,
            "n_frames": int(profile.n_frames),
            "n_atoms": int(profile.n_atoms),
            "coordinate_mode": profile.coordinate_mode,
            "cell_lengths_angstrom": (
                None
                if profile.cell_lengths_angstrom is None
                else [float(value) for value in profile.cell_lengths_angstrom]
            ),
            "cutoff_A": profile.cutoff_A,
            "cutoff_smoothing_width_A": profile.cutoff_smoothing_width_A,
            "cutoff_mode": profile.cutoff_mode,
            "cutoff_rdf_peak_A": profile.cutoff_rdf_peak_A,
            "cutoff_rdf_minimum_A": profile.cutoff_rdf_minimum_A,
            "cutoff_rdf_source_path": profile.cutoff_rdf_source_path,
            "cutoff_diagnostic_plot_path": profile.cutoff_diagnostic_plot_path,
            **_surface_metadata_payload(
                surface_position=profile.surface_position,
                surface_position_std=profile.surface_position_std,
                estimate=profile.surface_estimate,
            ),
        },
    )
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="coordination",
        datasets={
            "frame_index": profile.frame_index,
            "step": profile.step,
            "time_fs": profile.time_fs,
            "time_ps": profile.time_ps,
            "atom_indices": profile.atom_indices,
            "distance_to_surface_A": profile.distance_to_surface,
            "coordination_number": profile.coordination_number,
            "surface_position_per_frame_A": profile.surface_position_per_frame,
            "cutoff_rdf_bin_centers_A": profile.cutoff_rdf_bin_centers_A,
            "cutoff_rdf_g_r": profile.cutoff_rdf_g_r,
            "cutoff_rdf_g_r_smoothed": profile.cutoff_rdf_g_r_smoothed,
            **_surface_estimate_datasets(profile.surface_estimate),
        }
        | (
            {}
            if profile.cutoff_rdf_sampled_frame_index is None
            else {
                "cutoff_rdf_sampled_frame_index": profile.cutoff_rdf_sampled_frame_index,
            }
        ),
        metadata=metadata,
    )
    LOGGER.info("Saved coordination data to '%s'.", output_path)
    return output_path


def _coordination_profile_hdf5_payload(profile: CoordinationProfile) -> dict[str, Any]:
    """Return LiNaK HDF5 payload pieces for one coordination profile."""
    return {
        "datasets": {
            "frame_index": profile.frame_index,
            "step": profile.step,
            "time_fs": profile.time_fs,
            "time_ps": profile.time_ps,
            "atom_indices": profile.atom_indices,
            "distance_to_surface_A": profile.distance_to_surface,
            "coordination_number": profile.coordination_number,
            "surface_position_per_frame_A": profile.surface_position_per_frame,
            "cutoff_rdf_bin_centers_A": profile.cutoff_rdf_bin_centers_A,
            "cutoff_rdf_g_r": profile.cutoff_rdf_g_r,
            "cutoff_rdf_g_r_smoothed": profile.cutoff_rdf_g_r_smoothed,
            **_surface_estimate_datasets(profile.surface_estimate),
        }
        | (
            {}
            if profile.cutoff_rdf_sampled_frame_index is None
            else {
                "cutoff_rdf_sampled_frame_index": profile.cutoff_rdf_sampled_frame_index,
            }
        ),
        "metadata": build_profile_metadata(
            analysis="coordination",
            metadata={
                "species_a": profile.species_a,
                "species_b": profile.species_b,
                "axis": profile.axis,
                "n_frames": int(profile.n_frames),
                "n_atoms": int(profile.n_atoms),
                "coordinate_mode": profile.coordinate_mode,
                "cell_lengths_angstrom": (
                    None
                    if profile.cell_lengths_angstrom is None
                    else [float(value) for value in profile.cell_lengths_angstrom]
                ),
                "cutoff_A": profile.cutoff_A,
                "cutoff_smoothing_width_A": profile.cutoff_smoothing_width_A,
                "cutoff_mode": profile.cutoff_mode,
                "cutoff_rdf_peak_A": profile.cutoff_rdf_peak_A,
                "cutoff_rdf_minimum_A": profile.cutoff_rdf_minimum_A,
                "cutoff_rdf_source_path": profile.cutoff_rdf_source_path,
                "cutoff_diagnostic_plot_path": profile.cutoff_diagnostic_plot_path,
                **_surface_metadata_payload(
                    surface_position=profile.surface_position,
                    surface_position_std=profile.surface_position_std,
                    estimate=profile.surface_estimate,
                ),
            },
        ),
    }


def save_coordination_profiles(
    profiles: list[CoordinationProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one or more coordination profiles to LiNaK HDF5 and return the written path."""
    if not profiles:
        raise ValueError("At least one coordination profile is required.")
    if len(profiles) == 1:
        return save_coordination_profile(
            profiles[0],
            output,
            additional_metadata=additional_metadata,
        )

    output_path = write_linak_hdf5_profile_collection(
        output,
        analysis="coordination",
        profiles=[_coordination_profile_hdf5_payload(profile) for profile in profiles],
        metadata=dict(additional_metadata or {}),
    )
    LOGGER.info("Saved %d coordination profiles to '%s'.", len(profiles), output_path)
    return output_path


def load_coordination_profile(
    path: str | Path,
    *,
    species_a: str | None = None,
    species_b: str | None = None,
    axis: str | None = None,
) -> CoordinationProfile:
    profiles = load_coordination_profiles(path, species_a=species_a, species_b=species_b, axis=axis)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(
            f"Coordination HDF5 '{source_path}' does not contain matching coordination profiles."
        )
    return profiles[0]


def load_coordination_profiles(
    path: str | Path,
    *,
    species_a: str | None = None,
    species_b: str | None = None,
    axis: str | None = None,
) -> list[CoordinationProfile]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Coordination profile not found: {source_path}")
    if not is_hdf5_path(source_path):
        raise ValueError(
            f"Unsupported coordination profile format for '{source_path}'. Use .h5/.hdf5."
        )

    payloads = read_linak_hdf5_profiles(source_path, expected_analysis="coordination")
    return _load_coordination_profiles_from_payloads(
        source_path,
        payloads,
        species_a=species_a,
        species_b=species_b,
        axis=axis,
    )


def _load_coordination_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species_a: str | None = None,
    species_b: str | None = None,
    axis: str | None = None,
) -> list[CoordinationProfile]:
    wanted_species_a = (
        None if species_a is None or not str(species_a).strip() else _normalize_species(species_a)
    )
    wanted_species_b = (
        None if species_b is None or not str(species_b).strip() else _normalize_species(species_b)
    )
    wanted_axis = None if axis is None or not str(axis).strip() else _normalize_axis(axis)
    profiles: list[CoordinationProfile] = []
    for datasets, metadata in payloads:
        required = (
            "frame_index",
            "step",
            "time_fs",
            "time_ps",
            "atom_indices",
            "distance_to_surface_A",
            "coordination_number",
        )
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Coordination HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )

        resolved_species_a = str(metadata.get("species_a", "")).strip() or "UNKNOWN"
        resolved_species_b = str(metadata.get("species_b", "")).strip() or resolved_species_a
        resolved_axis = str(metadata.get("axis", "z")).strip().lower()
        if resolved_axis not in {"x", "y", "z"}:
            resolved_axis = "z"

        if (
            wanted_species_a is not None
            and _normalize_species(resolved_species_a) != wanted_species_a
        ):
            continue
        if (
            wanted_species_b is not None
            and _normalize_species(resolved_species_b) != wanted_species_b
        ):
            continue
        if wanted_axis is not None and resolved_axis != wanted_axis:
            continue

        frame_index = np.asarray(datasets["frame_index"], dtype=int)
        step = np.asarray(datasets["step"], dtype=float)
        time_fs = np.asarray(datasets["time_fs"], dtype=float)
        time_ps = np.asarray(datasets["time_ps"], dtype=float)
        atom_indices = np.asarray(datasets["atom_indices"], dtype=int)
        distance_values = np.asarray(datasets["distance_to_surface_A"], dtype=float)
        coordination_values = np.asarray(datasets["coordination_number"], dtype=float)
        if distance_values.ndim != 2:
            raise ValueError(
                f"Coordination HDF5 '{source_path}' dataset 'distance_to_surface_A' must be 2D."
            )
        if coordination_values.shape != distance_values.shape:
            raise ValueError(
                f"Coordination HDF5 '{source_path}' dataset 'coordination_number' shape mismatch: "
                f"expected {distance_values.shape}, got {coordination_values.shape}."
            )
        if atom_indices.size != distance_values.shape[1]:
            raise ValueError(
                f"Coordination HDF5 '{source_path}' has inconsistent atom index count "
                f"({atom_indices.size}) for matrix width {distance_values.shape[1]}."
            )
        if frame_index.size != distance_values.shape[0]:
            raise ValueError(
                f"Coordination HDF5 '{source_path}' has inconsistent frame index count "
                f"({frame_index.size}) for matrix height {distance_values.shape[0]}."
            )

        surface_per_frame = None
        if "surface_position_per_frame_A" in datasets:
            candidate = np.asarray(datasets["surface_position_per_frame_A"], dtype=float)
            if candidate.shape == (distance_values.shape[0],):
                surface_per_frame = candidate
        surface_estimate = _surface_estimate_from_payload(
            datasets=datasets,
            metadata=metadata,
        )

        cutoff_rdf_bin_centers = (
            np.asarray(datasets["cutoff_rdf_bin_centers_A"], dtype=float)
            if "cutoff_rdf_bin_centers_A" in datasets
            else None
        )
        cutoff_rdf_g_r = (
            np.asarray(datasets["cutoff_rdf_g_r"], dtype=float)
            if "cutoff_rdf_g_r" in datasets
            else None
        )
        cutoff_rdf_g_r_smoothed = (
            np.asarray(datasets["cutoff_rdf_g_r_smoothed"], dtype=float)
            if "cutoff_rdf_g_r_smoothed" in datasets
            else None
        )
        cutoff_rdf_sampled_frame_index = (
            np.asarray(datasets["cutoff_rdf_sampled_frame_index"], dtype=int)
            if "cutoff_rdf_sampled_frame_index" in datasets
            else None
        )
        surface_metadata = _surface_metadata_view(metadata)
        profiles.append(
            CoordinationProfile(
                species_a=resolved_species_a,
                species_b=resolved_species_b,
                axis=resolved_axis,
                atom_indices=atom_indices,
                frame_index=frame_index,
                step=step,
                time_fs=time_fs,
                time_ps=time_ps,
                distance_to_surface=distance_values,
                coordination_number=coordination_values,
                n_frames=int(metadata.get("n_frames", distance_values.shape[0])),
                n_atoms=int(metadata.get("n_atoms", distance_values.shape[1])),
                coordinate_mode=str(metadata.get("coordinate_mode", "axis")).strip().lower()
                or "axis",
                surface_position=_optional_finite_float(
                    surface_metadata.get("position", metadata.get("surface_position"))
                ),
                surface_position_std=_optional_finite_float(
                    surface_metadata.get("position_std", metadata.get("surface_position_std"))
                ),
                surface_position_per_frame=surface_per_frame,
                surface_estimate=surface_estimate,
                cell_lengths_angstrom=(
                    _optional_cell_lengths(metadata.get("cell_lengths_angstrom"))
                    or _optional_cell_lengths(metadata.get("pbc_cell_angstrom"))
                    or _optional_cell_lengths(metadata.get("resolved_cell_angstrom"))
                ),
                cutoff_A=_optional_finite_float(metadata.get("cutoff_A")),
                cutoff_smoothing_width_A=_optional_finite_float(
                    metadata.get("cutoff_smoothing_width_A")
                ),
                cutoff_mode=str(metadata.get("cutoff_mode", "direct")).strip().lower() or "direct",
                cutoff_rdf_bin_centers_A=cutoff_rdf_bin_centers,
                cutoff_rdf_g_r=cutoff_rdf_g_r,
                cutoff_rdf_g_r_smoothed=cutoff_rdf_g_r_smoothed,
                cutoff_rdf_peak_A=_optional_finite_float(metadata.get("cutoff_rdf_peak_A")),
                cutoff_rdf_minimum_A=_optional_finite_float(metadata.get("cutoff_rdf_minimum_A")),
                cutoff_rdf_sampled_frame_index=cutoff_rdf_sampled_frame_index,
                cutoff_rdf_source_path=str(metadata.get("cutoff_rdf_source_path", "")).strip()
                or None,
                cutoff_diagnostic_plot_path=str(
                    metadata.get("cutoff_diagnostic_plot_path", "")
                ).strip()
                or None,
            )
        )
    return profiles


def load_coordination_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species_a: str | None = None,
    species_b: str | None = None,
    axis: str | None = None,
) -> list[CoordinationProfile]:
    """Load selected coordination profiles by profile index from LiNaK HDF5."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Coordination profile not found: {source_path}")
    if not is_hdf5_path(source_path):
        raise ValueError(
            f"Unsupported coordination profile format for '{source_path}'. Use .h5/.hdf5."
        )
    payloads = read_linak_hdf5_profiles_by_index(
        source_path,
        profile_indices,
        expected_analysis="coordination",
    )
    return _load_coordination_profiles_from_payloads(
        source_path,
        payloads,
        species_a=species_a,
        species_b=species_b,
        axis=axis,
    )


def _coordination_time_data(
    profile: CoordinationProfile,
    *,
    time_axis: str,
) -> tuple[np.ndarray, str]:
    normalized = str(time_axis).strip().lower()
    if normalized == "ps":
        return profile.time_ps, "Time (ps)"
    if normalized == "fs":
        return profile.time_fs, "Time (fs)"
    if normalized == "step":
        return profile.step, "Timestep"
    if normalized == "frame":
        return profile.frame_index.astype(float), "Frame index"
    raise ValueError(
        f"Unsupported coordination time_axis '{time_axis}'. Choose 'ps', 'fs', 'step', or 'frame'."
    )


def _coordination_distance_label(profile: CoordinationProfile) -> str:
    if profile.coordinate_mode != "distance":
        return f"{profile.axis.upper()} (A)"
    return "Distance to the surface ($\\mathrm{\\AA}$)"


def _default_coordination_series_labels(profile: CoordinationProfile) -> list[str]:
    prefix = profile.species_a if profile.species_a != "ALL" else "A"
    return [f"{prefix}[{int(atom_index)}]" for atom_index in profile.atom_indices.tolist()]


def _coordination_distance_series(
    profile: CoordinationProfile,
    *,
    bin_width_A: float,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    ensure_positive("x_bin_width", bin_width_A)
    token = str(reducer).strip().lower()
    if token not in {"mean", "median", "sum", "min", "max"}:
        raise ValueError("x_bin_reducer must be one of: mean, median, sum, min, max.")

    distances = np.asarray(profile.distance_to_surface, dtype=float).ravel()
    cn_values = np.asarray(profile.coordination_number, dtype=float).ravel()
    mask = np.isfinite(distances) & np.isfinite(cn_values)
    if not np.any(mask):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    distances = distances[mask]
    cn_values = cn_values[mask]
    start = float(np.floor(np.min(distances) / float(bin_width_A)) * float(bin_width_A))
    bin_index = np.floor((distances - start) / float(bin_width_A)).astype(np.int64)
    unique_bins = np.unique(bin_index)
    x_out = np.empty(unique_bins.size, dtype=float)
    y_out = np.empty(unique_bins.size, dtype=float)
    for out_index, group_id in enumerate(unique_bins):
        group = cn_values[bin_index == group_id]
        x_out[out_index] = start + (float(group_id) + 0.5) * float(bin_width_A)
        if token == "mean":
            y_out[out_index] = float(np.mean(group))
        elif token == "median":
            y_out[out_index] = float(np.median(group))
        elif token == "sum":
            y_out[out_index] = float(np.sum(group))
        elif token == "min":
            y_out[out_index] = float(np.min(group))
        else:
            y_out[out_index] = float(np.max(group))
    return x_out, y_out


def _plot_coordination_time_distance_projection(
    profiles: list[CoordinationProfile],
    *,
    time_axis: str,
    output: str | Path | None,
    show: bool,
    show_blocking: bool,
    preferred_backend: str | None,
    style: PlotStyle,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    x_scale: str,
    y_scale: str,
    x_lim: tuple[float | None, float | None] | list[float | None] | None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None,
    x_ticks: list[float] | tuple[float, ...] | None,
    y_ticks: list[float] | tuple[float, ...] | None,
    x_tick_rotation: float | None,
    y_tick_rotation: float | None,
    x_label_font_size: int | None,
    y_label_font_size: int | None,
    x_label_pad: float | None,
    y_label_pad: float | None,
    title_visible: bool | None,
    ticks_visible: bool | None,
    line_colors: list[str] | None,
    series_ids: list[str] | None,
    series_enabled: list[bool] | None,
    series_line_widths: list[float | None] | None,
    series_markers: list[str | None] | None,
    render_series_descriptors: list[dict[str, Any]] | None,
    series_normalization_modes: list[str | None] | None,
    series_normalization_values: list[float | None] | None,
    series_normalization_x_refs: list[float | None] | None,
    x_bin_width: float | None,
    x_bin_reducer: str | None,
    annotations: list[dict[str, Any]] | None,
    capture_state: dict[str, Any] | None,
    suppress_output_log: bool,
    matplotlib_rc: dict[str, Any] | None,
    figure_kwargs: dict[str, Any] | None,
    axes_kwargs: dict[str, Any] | None,
    line_kwargs: dict[str, Any] | None,
    grid_kwargs: dict[str, Any] | None,
    tick_params_kwargs: dict[str, Any] | None,
    tight_layout_kwargs: dict[str, Any] | None,
    savefig_kwargs: dict[str, Any] | None,
) -> Path | None:
    if not profiles:
        raise ValueError("At least one coordination profile is required.")

    (
        profiles,
        series_enabled,
        series_line_widths,
        series_markers,
        series_normalization_modes,
        series_normalization_values,
        series_normalization_x_refs,
    ) = _expand_time_distance_render_layers(
        profiles,
        series_ids=series_ids,
        render_series_descriptors=render_series_descriptors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
    )

    series_total = sum(max(0, int(profile.n_atoms)) for profile in profiles)
    if series_enabled is not None and len(series_enabled) != series_total:
        raise ValueError(
            "series_enabled count must match the number of plotted coordination atom series "
            f"({series_total})."
        )
    if line_colors is not None:
        LOGGER.warning(
            "Coordination component 'time-distance' ignores fixed line colors and uses coordination-number colors."
        )
    if series_markers is not None:
        LOGGER.warning("Coordination component 'time-distance' ignores per-series markers.")
    if series_normalization_modes is not None:
        LOGGER.warning(
            "Coordination component 'time-distance' ignores per-series normalization settings."
        )
    if x_bin_width is not None:
        LOGGER.warning(
            "Coordination component 'time-distance' ignores x-bin settings (received %.6g; reducer=%s).",
            x_bin_width,
            x_bin_reducer or "mean",
        )

    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    segment_blocks: list[np.ndarray] = []
    segment_color_blocks: list[np.ndarray] = []
    point_x_values: list[float] = []
    point_y_values: list[float] = []
    point_color_values: list[float] = []
    series_index = 0
    default_x_label = "Time (ps)"
    default_y_label = _coordination_distance_label(profiles[0])
    for profile in profiles:
        x_values_template, resolved_x_label = _coordination_time_data(profile, time_axis=time_axis)
        default_x_label = resolved_x_label
        default_y_label = _coordination_distance_label(profile)
        for atom_column in range(profile.n_atoms):
            is_enabled = True if series_enabled is None else bool(series_enabled[series_index])
            series_index += 1
            if not is_enabled:
                continue
            x_values = np.asarray(x_values_template, dtype=float)
            y_values = np.asarray(profile.distance_to_surface[:, atom_column], dtype=float)
            color_values = np.asarray(profile.coordination_number[:, atom_column], dtype=float)
            if x_values.size == 0:
                continue
            if x_values.size == 1:
                point_x_values.append(float(x_values[0]))
                point_y_values.append(float(y_values[0]))
                point_color_values.append(float(color_values[0]))
                continue
            segments, segment_colors = _build_xy_segments(
                x_values,
                y_values,
                color_values,
                cell_lengths_xy=None,
            )
            if segments.size == 0:
                continue
            segment_blocks.append(segments)
            segment_color_blocks.append(segment_colors)

    if not segment_blocks and not point_x_values:
        raise ValueError(
            "No enabled atom trajectories available for coordination 'time-distance' plotting."
        )

    color_values_flat: list[np.ndarray] = []
    if segment_color_blocks:
        color_values_flat.extend(segment_color_blocks)
    if point_color_values:
        color_values_flat.append(np.asarray(point_color_values, dtype=float))
    concatenated_colors = np.concatenate(color_values_flat, axis=0)
    color_min = float(np.min(concatenated_colors))
    color_max = float(np.max(concatenated_colors))
    if np.isclose(color_min, color_max):
        color_max = color_min + 1.0
    norm = mcolors.Normalize(vmin=color_min, vmax=color_max)

    line_collection_kwargs = _sanitize_line_collection_kwargs(line_kwargs)
    explicit_line_width = None
    if series_line_widths:
        for value in series_line_widths:
            if value is not None:
                explicit_line_width = float(value)
                break
    line_collection_kwargs.setdefault(
        "linewidths",
        style.line_width if explicit_line_width is None else explicit_line_width,
    )
    marker_size = max(9.0, (style.line_width * 7.0) ** 2)

    active_backend = configure_matplotlib_backend(
        interactive=show, preferred_backend=preferred_backend
    )
    del active_backend
    rc_context_args: dict[str, Any] = {"font.family": style.font_family, "text.parse_math": True}
    if matplotlib_rc is not None:
        rc_context_args.update(dict(matplotlib_rc))
    with plt.rc_context(rc_context_args):
        fig, ax = plt.subplots(figsize=style.figure_size)
        if figure_kwargs is not None:
            fig.set(**dict(figure_kwargs))

        mappable = None
        if segment_blocks:
            segments_all = np.concatenate(segment_blocks, axis=0)
            segment_color_all = np.concatenate(segment_color_blocks, axis=0)
            collection = LineCollection(
                segments_all,
                cmap="turbo",
                norm=norm,
                **line_collection_kwargs,
            )
            collection.set_array(segment_color_all)
            ax.add_collection(collection)
            mappable = collection
        if point_x_values:
            scatter = ax.scatter(
                np.asarray(point_x_values, dtype=float),
                np.asarray(point_y_values, dtype=float),
                c=np.asarray(point_color_values, dtype=float),
                cmap="turbo",
                norm=norm,
                s=marker_size,
                edgecolors="none",
            )
            if mappable is None:
                mappable = scatter

        ax.autoscale()
        colorbar = fig.colorbar(mappable, ax=ax) if mappable is not None else None
        if colorbar is not None:
            colorbar.set_label("Coordination number", fontsize=style.label_font_size)
            colorbar.ax.tick_params(labelsize=style.tick_font_size)

        xlabel_kwargs: dict[str, Any] = {"fontsize": x_label_font_size or style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": y_label_font_size or style.label_font_size}
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(
            format_axis_label_units(resolve_explicit_plot_text(x_label, default_x_label)),
            **xlabel_kwargs,
        )
        ax.set_ylabel(
            format_axis_label_units(resolve_explicit_plot_text(y_label, default_y_label)),
            **ylabel_kwargs,
        )
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size)
        else:
            ax.set_title(
                title
                or f"{profiles[0].species_a}-{profiles[0].species_b} distance vs time colored by CN",
                fontsize=style.title_font_size,
            )

        ax.tick_params(axis="both", labelsize=style.tick_font_size)
        resolved_tick_params, tick_axis_hint, minor_ticks_mode = _extract_tick_controls(
            tick_params_kwargs
        )
        x_axis_tick_params = _axis_tick_params(tick_params_kwargs, "x")
        y_axis_tick_params = _axis_tick_params(tick_params_kwargs, "y")
        if x_tick_rotation is not None:
            x_axis_tick_params["rotation"] = float(x_tick_rotation)
        if y_tick_rotation is not None:
            y_axis_tick_params["rotation"] = float(y_tick_rotation)
        if resolved_tick_params:
            ax.tick_params(**resolved_tick_params)
        if x_axis_tick_params:
            ax.tick_params(axis="x", **x_axis_tick_params)
        if y_axis_tick_params:
            ax.tick_params(axis="y", **y_axis_tick_params)
        _apply_minor_tick_modes(
            ax,
            tick_params_kwargs=tick_params_kwargs,
            fallback_mode=minor_ticks_mode,
        )
        if ticks_visible is False:
            if tick_axis_hint in {"both", "x"}:
                ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            if tick_axis_hint in {"both", "y"}:
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        if style.grid:
            resolved_grid_kwargs: dict[str, Any] = {
                "linestyle": style.grid_linestyle,
                "linewidth": style.grid_linewidth,
                "alpha": style.grid_alpha,
            }
            if grid_kwargs is not None:
                resolved_grid_kwargs.update(dict(grid_kwargs))
            ax.grid(True, **resolved_grid_kwargs)
        elif grid_kwargs is not None:
            ax.grid(**dict(grid_kwargs))

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        if x_ticks is not None:
            ax.set_xticks([float(value) for value in x_ticks])
        if y_ticks is not None:
            ax.set_yticks([float(value) for value in y_ticks])
        if x_lim is not None:
            left = None if x_lim[0] is None else float(x_lim[0])
            right = None if x_lim[1] is None else float(x_lim[1])
            ax.set_xlim(left=left, right=right)
        if y_lim is not None:
            bottom = None if y_lim[0] is None else float(y_lim[0])
            top = None if y_lim[1] is None else float(y_lim[1])
            ax.set_ylim(bottom=bottom, top=top)
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))
        annotation_summaries = _render_plot_annotations(ax, annotations)

        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()

        if capture_state is not None:
            capture_state.clear()
            capture_state.update(
                {
                    "title": str(ax.get_title()),
                    "title_visible": bool(
                        ax.title.get_visible() and bool(str(ax.get_title()).strip())
                    ),
                    "x_label": str(ax.get_xlabel()),
                    "y_label": str(ax.get_ylabel()),
                    "x_scale": str(ax.get_xscale()),
                    "y_scale": str(ax.get_yscale()),
                    "x_lim": [float(value) for value in ax.get_xlim()],
                    "y_lim": [float(value) for value in ax.get_ylim()],
                    "x_ticks": [float(value) for value in ax.get_xticks()],
                    "y_ticks": [float(value) for value in ax.get_yticks()],
                    "legend": False,
                    "legend_title": None,
                    "legend_loc": "best",
                    "series_labels": None,
                    "line_colors": None,
                    "markers": bool(point_x_values),
                    "component": "time-distance",
                    "time_axis": str(time_axis).strip().lower(),
                    "annotations_summary": annotation_summaries,
                }
            )

        output_path = None
        if output is not None:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs: dict[str, Any] = {}
            if savefig_kwargs is not None:
                save_kwargs.update(dict(savefig_kwargs))
            save_kwargs.setdefault("dpi", style.dpi)
            fig.savefig(output_path, **save_kwargs)
            if not suppress_output_log:
                LOGGER.info("Saved plot to '%s'.", output_path)

        if show:
            plt.show(block=show_blocking)
            if not show_blocking:
                plt.pause(0.001)

        if not (show and not show_blocking):
            plt.close(fig)
        return output_path


def plot_coordination_profile(
    profile: CoordinationProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    component: str = "distance",
    time_axis: str = "ps",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_font_size: int | None = None,
    y_label_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    line_label: str | None = None,
    line_colors: list[str] | None = None,
    error_config: dict[str, Any] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_cumulative_configs: list[dict[str, Any] | None] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    cumulative_config: dict[str, Any] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> Path | None:
    normalized_component = _normalize_component_token(component)
    if normalized_component == "time-distance":
        return _plot_coordination_time_distance_projection(
            [profile],
            time_axis=time_axis,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
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
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            line_colors=line_colors,
            series_ids=[series_id] if series_id is not None else None,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            render_series_descriptors=render_series_descriptors,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            annotations=annotations,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    if normalized_component == "time":
        x_values, default_x = _coordination_time_data(profile, time_axis=time_axis)
        labels = _default_coordination_series_labels(profile)
        effective_legend = (profile.n_atoms <= 12) if legend is None else legend
        return plot_multi_line_series(
            [np.asarray(x_values, dtype=float) for _ in range(profile.n_atoms)],
            [
                np.asarray(profile.coordination_number[:, column], dtype=float)
                for column in range(profile.n_atoms)
            ],
            labels if line_label is None else [line_label, *labels[1:]],
            title=title or f"{profile.species_a}-{profile.species_b} coordination vs time",
            x_label=resolve_explicit_plot_text(x_label, default_x),
            y_label=resolve_explicit_plot_text(y_label, "Coordination number"),
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            series_ids=[series_id] if series_id is not None else None,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            series_cumulative_configs=series_cumulative_configs,
            series_error_configs=[error_config] * profile.n_atoms
            if error_config is not None
            else None,
            series_raw_statistics=[True] * profile.n_atoms,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            render_series_descriptors=render_series_descriptors,
            series_overrides_by_id=series_overrides_by_id,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            min_bin_points=min_bin_points,
            analysis_name="coordination",
            annotations=annotations,
            integration_config=integration_config,
            x_scale=x_scale,
            y_scale=y_scale,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=effective_legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            capture_state=capture_state,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            suppress_output_log=suppress_output_log,
        )

    bin_width = _DEFAULT_DISTANCE_BIN_WIDTH_A if x_bin_width is None else float(x_bin_width)
    reducer = "mean" if x_bin_reducer is None else str(x_bin_reducer).strip().lower()
    x_values = np.asarray(profile.distance_to_surface, dtype=float).reshape(-1)
    y_values = np.asarray(profile.coordination_number, dtype=float).reshape(-1)
    line_label_resolved = line_label or f"{profile.species_a}-{profile.species_b}"
    effective_legend = False if legend is None else legend
    return plot_line_series(
        x_values,
        y_values,
        title=title or f"{profile.species_a}-{profile.species_b} coordination vs distance",
        x_label=resolve_explicit_plot_text(x_label, _coordination_distance_label(profile)),
        y_label=resolve_explicit_plot_text(y_label, "Coordination number"),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_id=series_id,
        line_label=line_label_resolved,
        line_color=line_colors[0] if line_colors else None,
        line_width_override=series_line_widths[0] if series_line_widths else None,
        line_marker=series_markers[0] if series_markers else None,
        line_visible=True if not series_enabled else bool(series_enabled[0]),
        show_in_legend=True if not series_show_in_legend else bool(series_show_in_legend[0]),
        fit_config=None if not series_fit_configs else series_fit_configs[0],
        cumulative_config=cumulative_config,
        raw_point_statistics=True,
        error_config=error_config,
        normalization_mode=series_normalization_modes[0] if series_normalization_modes else None,
        normalization_value=series_normalization_values[0] if series_normalization_values else None,
        normalization_x_ref=series_normalization_x_refs[0] if series_normalization_x_refs else None,
        x_bin_width=bin_width,
        x_bin_reducer=reducer,
        min_bin_points=min_bin_points,
        analysis_name="coordination",
        annotations=annotations,
        integration_config=integration_config,
        style=style,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        x_label_font_size=x_label_font_size,
        y_label_font_size=y_label_font_size,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=effective_legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )


def plot_coordination_profiles(
    profiles: list[CoordinationProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    component: str = "distance",
    time_axis: str = "ps",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_font_size: int | None = None,
    y_label_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    series_ids: list[str] | None = None,
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    series_error_configs: list[dict[str, Any] | None] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_cumulative_configs: list[dict[str, Any] | None] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> Path | None:
    if not profiles:
        raise ValueError("At least one coordination profile is required.")
    normalized_component = _normalize_component_token(component)

    if normalized_component == "time-distance":
        return _plot_coordination_time_distance_projection(
            profiles,
            time_axis=time_axis,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
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
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            line_colors=line_colors,
            series_ids=series_ids,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            render_series_descriptors=render_series_descriptors,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            annotations=annotations,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    use_gui_render_layers = bool(render_series_descriptors) or bool(series_overrides_by_id)
    if len(profiles) == 1 and not use_gui_render_layers:
        single_series_labels = series_labels
        if (
            normalized_component == "distance"
            and series_labels is not None
            and len(series_labels) == 1
        ):
            single_series_labels = [series_labels[0]]
        return plot_coordination_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            component=normalized_component,
            time_axis=time_axis,
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
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            x_axis_scale=x_axis_scale,
            x_axis_offset=x_axis_offset,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            series_id=None if not series_ids else str(series_ids[0]),
            line_label=None if not single_series_labels else single_series_labels[0],
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            series_cumulative_configs=series_cumulative_configs,
            render_series_descriptors=render_series_descriptors,
            series_overrides_by_id=series_overrides_by_id,
            cumulative_config=None
            if not series_cumulative_configs
            else series_cumulative_configs[0],
            error_config=None if not series_error_configs else series_error_configs[0],
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            min_bin_points=min_bin_points,
            annotations=annotations,
            integration_config=integration_config,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    if normalized_component == "time":
        default_labels: list[str] = []
        x_series: list[np.ndarray] = []
        y_series: list[np.ndarray] = []
        default_x_label = "Time (ps)"
        for profile in profiles:
            x_values, default_x_label = _coordination_time_data(profile, time_axis=time_axis)
            for column, atom_index in enumerate(profile.atom_indices.tolist()):
                x_series.append(np.asarray(x_values, dtype=float))
                y_series.append(np.asarray(profile.coordination_number[:, column], dtype=float))
                prefix = profile.species_a if profile.species_a != "ALL" else "A"
                default_labels.append(f"{prefix}[{int(atom_index)}]")
        labels = resolve_series_labels(default_labels, series_labels, series_kind="coordination")
        effective_legend = (len(labels) <= 12) if legend is None else legend
        return plot_multi_line_series(
            x_series,
            y_series,
            labels,
            title=title or "Continuous coordination vs time",
            x_label=resolve_explicit_plot_text(x_label, default_x_label),
            y_label=resolve_explicit_plot_text(y_label, "Coordination number"),
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            series_ids=series_ids,
            style=style,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            series_cumulative_configs=series_cumulative_configs,
            series_error_configs=series_error_configs,
            series_raw_statistics=[True] * len(labels),
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            render_series_descriptors=render_series_descriptors,
            series_overrides_by_id=series_overrides_by_id,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            min_bin_points=min_bin_points,
            analysis_name="coordination",
            annotations=annotations,
            integration_config=integration_config,
            x_scale=x_scale,
            y_scale=y_scale,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            x_axis_scale=x_axis_scale,
            x_axis_offset=x_axis_offset,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=effective_legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            capture_state=capture_state,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            series_line_kwargs=series_line_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
            suppress_output_log=suppress_output_log,
        )

    reducer = "mean" if x_bin_reducer is None else str(x_bin_reducer).strip().lower()
    bin_width = _DEFAULT_DISTANCE_BIN_WIDTH_A if x_bin_width is None else float(x_bin_width)
    x_series = []
    y_series = []
    default_labels = []
    default_x_label = _coordination_distance_label(profiles[0])
    for profile in profiles:
        x_series.append(np.asarray(profile.distance_to_surface, dtype=float).reshape(-1))
        y_series.append(np.asarray(profile.coordination_number, dtype=float).reshape(-1))
        default_labels.append(f"{profile.species_a}-{profile.species_b}")
        default_x_label = _coordination_distance_label(profile)
    labels = resolve_series_labels(default_labels, series_labels, series_kind="coordination")
    effective_legend = True if legend is None else legend
    return plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or "Continuous coordination vs distance",
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, "Coordination number"),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_ids=series_ids,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_cumulative_configs=series_cumulative_configs,
        series_error_configs=series_error_configs,
        series_raw_statistics=[True] * len(labels),
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=bin_width,
        x_bin_reducer=reducer,
        min_bin_points=min_bin_points,
        analysis_name="coordination",
        annotations=annotations,
        integration_config=integration_config,
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
        legend=effective_legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        series_line_kwargs=series_line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )
