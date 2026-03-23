"""Continuous coordination-number analysis routines."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import repeat
import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np

from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
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
    read_linak_hdf5_profiles,
    write_linak_hdf5,
)
from ..utils import ensure_positive
from .position import (
    _build_xy_segments,
    compute_position_profile,
)
from .rdf import (
    _RDFWorkerConfig,
    _build_uniform_rdf_bins,
    _compute_rdf_chunk_contributions,
    _compute_rdf_frame_contribution,
    _normalize_species as _normalize_rdf_species,
    _resolve_rdf_chunk_size,
    _resolve_rdf_selection_cache,
    _resolve_rdf_worker_count,
    _should_parallelize_rdf,
)
from .schema import build_profile_metadata

LOGGER = logging.getLogger(__name__)
_DEFAULT_DISTANCE_BIN_WIDTH_A = 0.25
_DEFAULT_CUTOFF_SMOOTHING_WIDTH_A = 0.50
_DEFAULT_RDF_BIN_WIDTH_A = 0.05
_DEFAULT_RDF_SMOOTHING_SIGMA_A = 0.10
_DEFAULT_RDF_CONVERGENCE_BATCH_SIZE = 250
_DEFAULT_RDF_CONVERGENCE_MIN_FRAMES = 1_000
_DEFAULT_RDF_CONVERGENCE_MAX_FRAMES = 5_000
_DEFAULT_RDF_CONVERGENCE_WINDOW = 3
_DEFAULT_RDF_CONVERGENCE_TOLERANCE_A = 1.0e-3
_MIN_FIT_POINTS = 3
_FIT_HALF_WINDOW_POINTS = 3
_COORD_VECTORIZE_PAIR_THRESHOLD = 50_000
_COORD_MAX_DISTANCE_VALUES_PER_CHUNK = 2_000_000


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


def _normalize_species(species: str | None) -> str:
    return _normalize_rdf_species(species)


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


def _find_first_minimum_index(y: np.ndarray, *, start_index: int) -> int:
    if y.size < 3:
        raise ValueError("Need at least three RDF points to resolve a minimum.")
    candidates = [
        index
        for index in range(max(1, start_index + 1), y.size - 1)
        if y[index] <= y[index - 1] and y[index] < y[index + 1]
    ]
    if candidates:
        return candidates[0]
    if start_index + 1 >= y.size:
        return y.size - 1
    return int(start_index + 1 + np.argmin(y[start_index + 1 :]))


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


def _resolve_converged_sampled_rdf(
    frames: list[Atoms],
    *,
    species_a: str,
    species_b: str,
    batch_size: int = _DEFAULT_RDF_CONVERGENCE_BATCH_SIZE,
    tolerance_A: float = _DEFAULT_RDF_CONVERGENCE_TOLERANCE_A,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not frames:
        raise ValueError("At least one trajectory frame is required.")
    ensure_positive("batch_size", batch_size)
    ensure_positive("tolerance_A", tolerance_A)

    frame_order = np.asarray(np.random.default_rng(0).permutation(len(frames)), dtype=int)
    max_sampled_frames = min(len(frame_order), int(_DEFAULT_RDF_CONVERGENCE_MAX_FRAMES))
    min_frames_before_check = min(max_sampled_frames, int(_DEFAULT_RDF_CONVERGENCE_MIN_FRAMES))
    convergence_window = max(2, int(_DEFAULT_RDF_CONVERGENCE_WINDOW))
    selected_index_batches: list[np.ndarray] = []
    sampled_frame_count = 0
    previous_cutoff_A: float | None = None
    recent_cutoffs_A: list[float] = []
    selected_frame_batch = [
        (frame_index, frames[frame_index]) for frame_index in frame_order.tolist()
    ]
    (
        bin_edges,
        config,
    ) = _build_reference_rdf_config(
        selected_frame_batch,
        species_a=species_a,
        species_b=species_b,
        r_max=None,
        bin_width=float(_DEFAULT_RDF_BIN_WIDTH_A),
    )
    counts_accum = np.zeros(bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    final_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    final_g_r = np.zeros_like(counts_accum)
    final_smoothed = np.empty(0, dtype=float)
    final_peak_A = 0.0
    final_minimum_A = 0.0
    max_batch_frames = min(int(batch_size), max_sampled_frames)
    worker_count = _resolve_rdf_worker_count(None, max_batch_frames)
    use_parallel = _should_parallelize_rdf(max_batch_frames, worker_count)

    executor_context = (
        ProcessPoolExecutor(max_workers=worker_count) if use_parallel else nullcontext(None)
    )
    with executor_context as executor:
        for step_index, start in enumerate(range(0, max_sampled_frames, int(batch_size)), start=1):
            stop = min(start + int(batch_size), max_sampled_frames)
            next_batch = frame_order[start:stop]
            batch_frames = selected_frame_batch[start:stop]
            if next_batch.size == 0:
                break
            batch_counts, batch_expected = _accumulate_reference_rdf_contributions(
                batch_frames,
                config=config,
                progress=None,
                executor=executor,
                worker_count=worker_count,
            )
            counts_accum += batch_counts
            expected_accum += batch_expected
            selected_index_batches.append(next_batch.astype(int))
            sampled_frame_count += int(next_batch.size)
            non_zero = expected_accum > 0.0
            final_g_r = np.zeros_like(counts_accum)
            final_g_r[non_zero] = counts_accum[non_zero] / expected_accum[non_zero]
            final_smoothed, final_peak_A, final_minimum_A = _resolve_cutoff_from_rdf_curve(
                bin_centers_A=final_bin_centers,
                g_r=final_g_r,
                smoothing_sigma_A=float(_DEFAULT_RDF_SMOOTHING_SIGMA_A),
            )
            delta_A = (
                None
                if previous_cutoff_A is None
                else abs(float(final_minimum_A) - float(previous_cutoff_A))
            )
            previous_cutoff_A = float(final_minimum_A)
            recent_cutoffs_A.append(float(final_minimum_A))
            if len(recent_cutoffs_A) > convergence_window:
                recent_cutoffs_A = recent_cutoffs_A[-convergence_window:]
            recent_span_A = (
                None
                if len(recent_cutoffs_A) < convergence_window
                else max(recent_cutoffs_A) - min(recent_cutoffs_A)
            )
            LOGGER.info(
                "Coordination cutoff RDF step %d: sampled=%d frame(s), cutoff=%.6g A%s%s",
                step_index,
                sampled_frame_count,
                final_minimum_A,
                "" if delta_A is None else f", delta={delta_A:.6g} A",
                "" if recent_span_A is None else f", recent_span={recent_span_A:.6g} A",
            )
            if (
                sampled_frame_count >= min_frames_before_check
                and recent_span_A is not None
                and recent_span_A <= float(tolerance_A)
            ):
                return (
                    np.sort(np.concatenate(selected_index_batches)).astype(int, copy=False),
                    np.asarray(final_bin_centers, dtype=float),
                    np.asarray(final_g_r, dtype=float),
                    np.asarray(final_smoothed, dtype=float),
                    float(final_peak_A),
                    float(final_minimum_A),
                )

    LOGGER.info(
        "Coordination cutoff RDF did not converge within %.6g A after sampling %d frame(s); using the final sampled cutoff.",
        float(tolerance_A),
        sampled_frame_count,
    )
    return (
        np.sort(np.concatenate(selected_index_batches)).astype(int, copy=False),
        np.asarray(final_bin_centers, dtype=float),
        np.asarray(final_g_r, dtype=float),
        np.asarray(final_smoothed, dtype=float),
        float(final_peak_A),
        float(final_minimum_A),
    )


def _resolve_reference_rdf_r_max(
    frames: list[Atoms],
    *,
    r_max: float | None,
) -> float:
    if r_max is not None:
        ensure_positive("r_max", r_max)
        return float(r_max)
    min_cell_lengths = []
    for frame in frames:
        cell = np.asarray(frame.cell.array, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError("RDF cutoff detection requires valid periodic cell vectors.")
        cell_lengths = np.linalg.norm(cell, axis=1)
        min_cell_lengths.append(float(np.min(cell_lengths)))
    resolved_r_max = 0.5 * min(min_cell_lengths)
    ensure_positive("r_max", resolved_r_max)
    return float(resolved_r_max)


def _build_reference_rdf_config(
    selected_frames: list[tuple[int, Atoms]],
    *,
    species_a: str,
    species_b: str,
    r_max: float | None,
    bin_width: float,
) -> tuple[np.ndarray, _RDFWorkerConfig]:
    if not selected_frames:
        raise ValueError("No frames were selected for RDF cutoff resolution.")
    ensure_positive("bin_width", bin_width)

    frame_objects = [frame for _frame_index, frame in selected_frames]
    resolved_r_max = _resolve_reference_rdf_r_max(frame_objects, r_max=r_max)
    bin_edges, _effective_bin_width = _build_uniform_rdf_bins(
        r_max=resolved_r_max,
        target_bin_width=float(bin_width),
    )
    shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    max_sphere_volume = (4.0 / 3.0) * np.pi * (resolved_r_max**3)
    same_selection = species_a == species_b
    selection_cache = _resolve_rdf_selection_cache(
        frame_objects,
        label_a=species_a,
        label_b=species_b,
    )
    return (
        bin_edges,
        _RDFWorkerConfig(
            label_a=species_a,
            label_b=species_b,
            same_selection=same_selection,
            r_max=resolved_r_max,
            bin_edges=bin_edges,
            shell_volumes=shell_volumes,
            max_sphere_volume=max_sphere_volume,
            selection_cache=selection_cache,
        ),
    )


def _accumulate_reference_rdf_contributions(
    selected_frames: list[tuple[int, Atoms]],
    *,
    config: _RDFWorkerConfig,
    progress: ProgressBar | None = None,
    executor: ProcessPoolExecutor | None = None,
    worker_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts_accum = np.zeros(config.bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    if not selected_frames:
        return counts_accum, expected_accum

    resolved_worker_count = (
        _resolve_rdf_worker_count(None, len(selected_frames))
        if worker_count is None
        else worker_count
    )
    resolved_worker_count = min(resolved_worker_count, max(1, len(selected_frames)))
    use_parallel = _should_parallelize_rdf(len(selected_frames), resolved_worker_count)
    if not use_parallel:
        for frame_index, frame in selected_frames:
            counts, expected = _compute_rdf_frame_contribution(
                frame_index,
                frame,
                label_a=config.label_a,
                label_b=config.label_b,
                same_selection=config.same_selection,
                r_max=config.r_max,
                bin_edges=config.bin_edges,
                shell_volumes=config.shell_volumes,
                max_sphere_volume=config.max_sphere_volume,
                selection_cache=config.selection_cache,
            )
            counts_accum += counts
            expected_accum += expected
            if progress is not None:
                progress.update()
        return counts_accum, expected_accum

    chunk_size = _resolve_rdf_chunk_size(len(selected_frames), resolved_worker_count)
    chunks = [
        selected_frames[start : start + chunk_size]
        for start in range(0, len(selected_frames), chunk_size)
    ]
    if executor is None:
        with ProcessPoolExecutor(max_workers=resolved_worker_count) as local_executor:
            for counts, expected, processed_frames in local_executor.map(
                _compute_rdf_chunk_contributions,
                chunks,
                repeat(config),
            ):
                counts_accum += counts
                expected_accum += expected
                if progress is not None:
                    progress.update(processed_frames)
        return counts_accum, expected_accum

    for counts, expected, processed_frames in executor.map(
        _compute_rdf_chunk_contributions,
        chunks,
        repeat(config),
    ):
        counts_accum += counts
        expected_accum += expected
        if progress is not None:
            progress.update(processed_frames)
    return counts_accum, expected_accum


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

    g_r = np.zeros_like(counts_accum)
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
    if bin_centers_A.size < 3:
        raise ValueError("Need at least three RDF bins to resolve a coordination cutoff.")

    diffs = np.diff(bin_centers_A)
    if diffs.size == 0:
        raise ValueError("Need at least two RDF bins to resolve smoothing width.")
    mean_bin_width = float(np.mean(diffs))
    sigma_bins = max(float(smoothing_sigma_A) / max(mean_bin_width, 1.0e-12), 1.0)
    smoothed = _gaussian_smooth(np.asarray(g_r, dtype=float), sigma_bins=sigma_bins)
    peak_index = _find_first_peak_index(bin_centers_A, smoothed)
    minimum_index = _find_first_minimum_index(smoothed, start_index=peak_index)
    cutoff_A = _fit_local_quadratic_minimum(
        np.asarray(bin_centers_A, dtype=float),
        np.asarray(smoothed, dtype=float),
        center_index=minimum_index,
    )
    return np.asarray(smoothed, dtype=float), float(bin_centers_A[peak_index]), cutoff_A


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
    ensure_positive("cutoff_smoothing_width_A", cutoff_smoothing_width_A)

    label_a = _normalize_species(species_a)
    label_b = _normalize_species(species_b if species_b is not None else species_a)

    if cutoff_A is not None:
        ensure_positive("cutoff_A", cutoff_A)
        return CoordinationCutoffResolution(
            cutoff_A=float(cutoff_A),
            smoothing_width_A=float(cutoff_smoothing_width_A),
            mode="direct",
        )

    if cutoff_rdf_path is not None:
        from .rdf import load_rdf_profile

        source_path = Path(cutoff_rdf_path).expanduser().resolve()
        rdf_profile = load_rdf_profile(source_path, species_a=label_a, species_b=label_b)
        smoothed, peak_A, minimum_A = _resolve_cutoff_from_rdf_curve(
            bin_centers_A=np.asarray(rdf_profile.bin_centers, dtype=float),
            g_r=np.asarray(rdf_profile.g_r, dtype=float),
            smoothing_sigma_A=float(_DEFAULT_RDF_SMOOTHING_SIGMA_A),
        )
        diagnostic_path = None
        if diagnostic_plot_output is not None:
            diagnostic_path = _save_cutoff_diagnostic_plot(
                output=diagnostic_plot_output,
                bin_centers_A=np.asarray(rdf_profile.bin_centers, dtype=float),
                g_r=np.asarray(rdf_profile.g_r, dtype=float),
                g_r_smoothed=smoothed,
                peak_A=peak_A,
                minimum_A=minimum_A,
                species_a=label_a,
                species_b=label_b,
            )
        return CoordinationCutoffResolution(
            cutoff_A=float(minimum_A),
            smoothing_width_A=float(cutoff_smoothing_width_A),
            mode="rdf_file",
            rdf_bin_centers_A=np.asarray(rdf_profile.bin_centers, dtype=float),
            rdf_g_r=np.asarray(rdf_profile.g_r, dtype=float),
            rdf_g_r_smoothed=smoothed,
            rdf_peak_A=peak_A,
            rdf_minimum_A=float(minimum_A),
            rdf_source_path=str(source_path),
            diagnostic_plot_path=None if diagnostic_path is None else str(diagnostic_path),
        )

    # Default to an internally sampled RDF whenever no higher-priority cutoff source is provided.
    LOGGER.info(
        "Resolving coordination cutoff from sampled RDF using random %d-frame batches (check after %d frame(s), %d-step span <= %.6g A, cap=%d frame(s)).",
        int(_DEFAULT_RDF_CONVERGENCE_BATCH_SIZE),
        int(_DEFAULT_RDF_CONVERGENCE_MIN_FRAMES),
        int(_DEFAULT_RDF_CONVERGENCE_WINDOW),
        float(_DEFAULT_RDF_CONVERGENCE_TOLERANCE_A),
        int(_DEFAULT_RDF_CONVERGENCE_MAX_FRAMES),
    )
    sampled_indices, bin_centers_A, g_r, smoothed, peak_A, minimum_A = (
        _resolve_converged_sampled_rdf(
            frames,
            species_a=label_a,
            species_b=label_b,
        )
    )
    diagnostic_path = None
    if diagnostic_plot_output is not None:
        diagnostic_path = _save_cutoff_diagnostic_plot(
            output=diagnostic_plot_output,
            bin_centers_A=bin_centers_A,
            g_r=g_r,
            g_r_smoothed=smoothed,
            peak_A=peak_A,
            minimum_A=minimum_A,
            species_a=label_a,
            species_b=label_b,
        )
    return CoordinationCutoffResolution(
        cutoff_A=float(minimum_A),
        smoothing_width_A=float(cutoff_smoothing_width_A),
        mode="sampled_rdf",
        rdf_bin_centers_A=np.asarray(bin_centers_A, dtype=float),
        rdf_g_r=np.asarray(g_r, dtype=float),
        rdf_g_r_smoothed=np.asarray(smoothed, dtype=float),
        rdf_peak_A=peak_A,
        rdf_minimum_A=float(minimum_A),
        rdf_sampled_frame_index=np.asarray(sampled_indices, dtype=int),
        diagnostic_plot_path=None if diagnostic_path is None else str(diagnostic_path),
    )


def _continuous_coordination_weights(
    distances: np.ndarray,
    *,
    cutoff_A: float,
    smoothing_width_A: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    weights = np.zeros(distances.shape, dtype=float)
    half_width = 0.5 * float(smoothing_width_A)
    lower = float(cutoff_A) - half_width
    upper = float(cutoff_A) + half_width
    if half_width <= 0.0:
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


def _compute_coordination_frame_values(
    frame: Atoms,
    *,
    center_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    same_selection: bool,
    cutoff_A: float,
    smoothing_width_A: float,
) -> np.ndarray:
    n_centers = int(center_indices.size)
    if n_centers == 0:
        raise ValueError("Coordination calculation requires at least one center atom.")
    support_cutoff = float(cutoff_A) + 0.5 * float(smoothing_width_A)
    if support_cutoff <= 0.0:
        raise ValueError("Coordination support cutoff must be positive.")

    use_neighbor_list = _frame_has_usable_cell(frame)
    center_lookup = np.full(len(frame), -1, dtype=int)
    center_lookup[np.asarray(center_indices, dtype=int)] = np.arange(n_centers, dtype=int)
    weights_accum = np.zeros(n_centers, dtype=float)

    if use_neighbor_list:
        center_mask = np.zeros(len(frame), dtype=bool)
        neighbor_mask = np.zeros(len(frame), dtype=bool)
        center_mask[np.asarray(center_indices, dtype=int)] = True
        neighbor_mask[np.asarray(neighbor_indices, dtype=int)] = True
        i_pairs, j_pairs, pair_distances = neighbor_list(
            "ijd",
            frame,
            float(np.nextafter(support_cutoff, np.inf)),
        )
        mask = center_mask[i_pairs] & neighbor_mask[j_pairs] & (pair_distances > 0.0)
        if same_selection:
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
            np.add.at(weights_accum, center_lookup[i_selected], weights)
        return weights_accum

    if same_selection:
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
    if pair_count <= 0 or pair_count > _COORD_VECTORIZE_PAIR_THRESHOLD:
        return False
    if not frames:
        return False
    usable_pbc = [_frame_has_usable_cell(frame) for frame in frames]
    return all(usable_pbc) or not any(usable_pbc)


def _compute_coordination_values_chunked(
    frames: list[Atoms],
    *,
    center_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    same_selection: bool,
    cutoff_A: float,
    smoothing_width_A: float,
) -> np.ndarray:
    frame_count = len(frames)
    center_count = int(center_indices.size)
    neighbor_count = int(neighbor_indices.size)
    pair_count = max(0, center_count * neighbor_count - (center_count if same_selection else 0))
    chunk_size = _resolve_coordination_chunk_size(frame_count=frame_count, pair_count=pair_count)
    use_mic = bool(frames) and all(_frame_has_usable_cell(frame) for frame in frames)

    coordination = np.zeros((frame_count, center_count), dtype=np.float32)
    with ProgressBar(
        desc=f"Coordination values ({center_count}x{neighbor_count})",
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
                    [np.asarray(frame.cell.lengths(), dtype=float) for frame in frames[start:stop]],
                    axis=0,
                )
                deltas -= cell_lengths[:, np.newaxis, np.newaxis, :] * np.round(
                    deltas / cell_lengths[:, np.newaxis, np.newaxis, :]
                )
            distances = np.linalg.norm(deltas, axis=3)
            if same_selection:
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
    )

    neighbor_indices = _select_indices(frames[0], label_b)
    if neighbor_indices.size == 0:
        raise ValueError(f"No atoms found for species '{label_b}' in frame 0.")

    same_selection = label_a == label_b
    center_indices = np.asarray(position_profile.atom_indices, dtype=int)
    center_count = int(center_indices.size)
    neighbor_count = int(neighbor_indices.size)
    pair_count = max(0, center_count * neighbor_count - (center_count if same_selection else 0))
    LOGGER.info(
        "Computing coordination values for %d frame(s), %d center atom(s), %d neighbor atom(s).",
        len(frames),
        center_count,
        neighbor_count,
    )
    if _can_vectorize_coordination_kernel(frames, pair_count=pair_count):
        chunk_size = _resolve_coordination_chunk_size(
            frame_count=len(frames),
            pair_count=max(1, pair_count),
        )
        LOGGER.info(
            "Using chunked vectorized coordination kernel (pair_count=%d, chunk_size=%d frame(s)).",
            pair_count,
            chunk_size,
        )
        coordination = _compute_coordination_values_chunked(
            frames,
            center_indices=center_indices,
            neighbor_indices=np.asarray(neighbor_indices, dtype=int),
            same_selection=same_selection,
            cutoff_A=float(cutoff_resolution.cutoff_A),
            smoothing_width_A=float(cutoff_resolution.smoothing_width_A),
        )
    else:
        LOGGER.info(
            "Using framewise coordination kernel (pair_count=%d per frame).",
            pair_count,
        )
        coordination = np.zeros(
            (len(frames), center_count),
            dtype=np.float32,
        )
        with ProgressBar(
            desc=f"Coordination values ({center_count}x{neighbor_count})",
            total=len(frames),
            unit="frame",
        ) as progress:
            for frame_index, frame in enumerate(frames):
                coordination[frame_index, :] = _compute_coordination_frame_values(
                    frame,
                    center_indices=center_indices,
                    neighbor_indices=np.asarray(neighbor_indices, dtype=int),
                    same_selection=same_selection,
                    cutoff_A=float(cutoff_resolution.cutoff_A),
                    smoothing_width_A=float(cutoff_resolution.smoothing_width_A),
                )
                progress.update()

    return CoordinationProfile(
        species_a=label_a,
        species_b=label_b,
        axis=axis_label,
        atom_indices=center_indices,
        frame_index=np.asarray(position_profile.frame_index, dtype=int),
        step=np.asarray(position_profile.step, dtype=float),
        time_fs=np.asarray(position_profile.time_fs, dtype=float),
        time_ps=np.asarray(position_profile.time_ps, dtype=float),
        distance_to_surface=np.asarray(position_profile.distance_to_surface, dtype=np.float32),
        coordination_number=np.asarray(coordination, dtype=np.float32),
        n_frames=int(position_profile.n_frames),
        n_atoms=int(position_profile.n_atoms),
        coordinate_mode=position_profile.coordinate_mode,
        surface_position=position_profile.surface_position,
        surface_position_std=position_profile.surface_position_std,
        surface_position_per_frame=(
            None
            if position_profile.surface_position_per_frame is None
            else np.asarray(position_profile.surface_position_per_frame, dtype=float)
        ),
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
            "surface_position": profile.surface_position,
            "surface_position_std": profile.surface_position_std,
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
            "cutoff_rdf_sampled_frame_index": profile.cutoff_rdf_sampled_frame_index,
        },
        metadata=metadata,
    )
    LOGGER.info("Saved coordination data to '%s'.", output_path)
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

    wanted_species_a = (
        None if species_a is None or not str(species_a).strip() else _normalize_species(species_a)
    )
    wanted_species_b = (
        None if species_b is None or not str(species_b).strip() else _normalize_species(species_b)
    )
    wanted_axis = None if axis is None or not str(axis).strip() else _normalize_axis(axis)

    payloads = read_linak_hdf5_profiles(source_path, expected_analysis="coordination")
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
                surface_position=_optional_finite_float(metadata.get("surface_position")),
                surface_position_std=_optional_finite_float(metadata.get("surface_position_std")),
                surface_position_per_frame=surface_per_frame,
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
    x_label_pad: float | None,
    y_label_pad: float | None,
    title_visible: bool | None,
    ticks_visible: bool | None,
    line_colors: list[str] | None,
    series_enabled: list[bool] | None,
    series_line_widths: list[float | None] | None,
    series_markers: list[str | None] | None,
    series_normalization_modes: list[str] | None,
    series_normalization_values: list[float | None] | None,
    series_normalization_x_refs: list[float | None] | None,
    x_bin_width: float | None,
    x_bin_reducer: str | None,
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
                cmap="viridis",
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
                cmap="viridis",
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

        xlabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
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
        resolved_tick_params = dict(tick_params_kwargs) if tick_params_kwargs is not None else {}
        tick_axis_hint = str(resolved_tick_params.pop("_ticks_axis", "both")).strip().lower()
        if tick_axis_hint not in {"x", "y", "both"}:
            tick_axis_hint = "both"
        minor_ticks_mode = (
            str(resolved_tick_params.pop("_minor_ticks_mode", "auto")).strip().lower()
        )
        if minor_ticks_mode == "on":
            ax.minorticks_on()
        elif minor_ticks_mode == "off":
            ax.minorticks_off()
        if resolved_tick_params:
            ax.tick_params(**resolved_tick_params)
        if ticks_visible is False:
            if tick_axis_hint in {"both", "x"}:
                ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            if tick_axis_hint in {"both", "y"}:
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        if x_tick_rotation is not None:
            ax.tick_params(axis="x", rotation=float(x_tick_rotation))
        if y_tick_rotation is not None:
            ax.tick_params(axis="y", rotation=float(y_tick_rotation))

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
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    line_label: str | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_fit_enabled: list[bool] | None = None,
    series_fit_labels: list[str | None] | None = None,
    series_fit_show_in_legend: list[bool] | None = None,
    series_normalization_modes: list[str] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
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
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
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
            series_fit_enabled=series_fit_enabled,
            series_fit_labels=series_fit_labels,
            series_fit_show_in_legend=series_fit_show_in_legend,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            x_scale=x_scale,
            y_scale=y_scale,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
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
    x_values, y_values = _coordination_distance_series(
        profile,
        bin_width_A=bin_width,
        reducer=reducer,
    )
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
        fit_enabled=True if series_fit_enabled and bool(series_fit_enabled[0]) else False,
        fit_label=(
            None if not series_fit_labels or not series_fit_labels[0] else str(series_fit_labels[0])
        ),
        fit_show_in_legend=(
            True if not series_fit_show_in_legend else bool(series_fit_show_in_legend[0])
        ),
        normalization_mode=series_normalization_modes[0] if series_normalization_modes else None,
        normalization_value=series_normalization_values[0] if series_normalization_values else None,
        normalization_x_ref=series_normalization_x_refs[0] if series_normalization_x_refs else None,
        x_bin_width=None,
        x_bin_reducer=None,
        style=style,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
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
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    series_ids: list[str] | None = None,
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_fit_enabled: list[bool] | None = None,
    series_fit_labels: list[str | None] | None = None,
    series_fit_show_in_legend: list[bool] | None = None,
    series_normalization_modes: list[str] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
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
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
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

    if len(profiles) == 1:
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
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
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
            series_fit_enabled=series_fit_enabled,
            series_fit_labels=series_fit_labels,
            series_fit_show_in_legend=series_fit_show_in_legend,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
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
            series_fit_enabled=series_fit_enabled,
            series_fit_labels=series_fit_labels,
            series_fit_show_in_legend=series_fit_show_in_legend,
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            x_scale=x_scale,
            y_scale=y_scale,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
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
        x_values, y_values = _coordination_distance_series(
            profile,
            bin_width_A=bin_width,
            reducer=reducer,
        )
        x_series.append(x_values)
        y_series.append(y_values)
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
        series_fit_enabled=series_fit_enabled,
        series_fit_labels=series_fit_labels,
        series_fit_show_in_legend=series_fit_show_in_legend,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        x_bin_width=None,
        x_bin_reducer=None,
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
