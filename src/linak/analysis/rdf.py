"""RDF analysis routines."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import combinations_with_replacement, repeat
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.rdf_mapping import resolve_rdf_plot_mapping
from ..storage.hdf5_utils import (
    resolve_hdf5_output_path,
    write_linak_hdf5,
)
from .binning import (
    reconstruct_uniform_bin_edges_from_centers,
    resolve_uniform_bin_width_for_load,
    uniform_bin_width_from_edges,
)
from .schema import build_profile_metadata, default_plot_labels
from .statistics import (
    SeriesStatistics,
    block_mean_matrix,
    build_series_statistics,
    build_series_statistics_from_moments,
    build_statistics_metadata,
    statistics_payload_from_series_map,
    statistics_series_map_from_datasets,
    resolve_block_slices,
)
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
    resolve_single_series_options,
)
from ..progress import ProgressBar
from ..utils import ensure_positive
from .common import (
    available_element_species as _available_element_species,
    normalize_species_label as _normalize_species,
    read_profile_payloads,
    read_profile_payloads_by_index,
    use_multi_series_plot,
    write_profile_collection,
)

LOGGER = logging.getLogger(__name__)
_NEIGHBORLIST_DENSITY_THRESHOLD = 0.25
_SELECTED_MATRIX_FRACTION_THRESHOLD = 0.40
_DEFAULT_RDF_THREAD_CAP = 4
_RDF_MIN_PARALLEL_FRAMES_PER_WORKER = 4
_RDF_MIN_CHUNK_SIZE = 16
_RDF_MAX_CHUNK_SIZE = 128
_RDF_DENSE_PAIR_THRESHOLD = 250_000
_RDF_FAST_MAX_DISTANCE_VALUES_PER_CHUNK = 2_000_000
_RDF_FAST_TARGET_CHUNKS_PER_WORKER = 4
_RDF_BACKEND_DENSE = "chunked_dense_matrix"
_RDF_BACKEND_SPARSE = "chunked_sparse_cutoff"
_RDF_BACKEND_GENERIC = "framewise_generic_fallback"
_RDF_SAME_PAIR_RTOL = 1.0e-6
_RDF_SAME_PAIR_ATOL = 1.0e-8


@dataclass(frozen=True)
class RDFProfile:
    """Container for a radial distribution function profile."""

    species_a: str
    species_b: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    g_r: np.ndarray
    n_frames: int
    series_statistics: dict[str, SeriesStatistics] | None = None
    atom_indices_a: np.ndarray | None = None
    atom_indices_b: np.ndarray | None = None
    selection_kind_a: str = "species"
    selection_kind_b: str = "species"


@dataclass(frozen=True)
class _RDFSelectionCache:
    """Stable per-frame selection metadata reused when atom identities do not change."""

    mask_a: np.ndarray
    mask_b: np.ndarray
    indices_a: np.ndarray
    indices_b: np.ndarray
    count_a: int
    count_b: int
    overlap_count: int


@dataclass(frozen=True)
class _RDFResolvedSelector:
    """Normalized one-side RDF selector used by compute and persistence layers."""

    selection_kind: str
    label: str
    species_label: str | None
    atom_indices: np.ndarray | None


@dataclass(frozen=True)
class _RDFWorkerConfig:
    """Shared RDF worker configuration for one chunk of frames."""

    label_a: str
    label_b: str
    same_selection: bool
    r_max: float
    bin_edges: np.ndarray
    shell_volumes: np.ndarray
    max_sphere_volume: float
    selection_cache: _RDFSelectionCache | None = None
    strategy_override: str | None = None
    collect_statistics: bool = False
    block_index_by_frame: np.ndarray | None = None


@dataclass(frozen=True)
class _RDFOrthorhombicChunk:
    """Compact orthorhombic chunk payload used by fast RDF backends."""

    positions_a: np.ndarray
    positions_b: np.ndarray
    cell_lengths: np.ndarray
    volumes: np.ndarray
    frame_count: int
    frame_start: int


@dataclass(frozen=True)
class _RDFOrthorhombicConfig:
    """Shared config for orthorhombic RDF chunk kernels."""

    r_max: float
    bin_edges: np.ndarray
    shell_volumes: np.ndarray
    same_selection: bool
    count_a: int
    count_b: int
    overlap_count: int
    indices_a: np.ndarray
    indices_b: np.ndarray
    self_pair_rows: np.ndarray
    self_pair_cols: np.ndarray
    collect_statistics: bool = False
    block_index_by_frame: np.ndarray | None = None


@dataclass(frozen=True)
class _RDFBackendResolution:
    """Resolved RDF execution backend for one compute_rdf call."""

    backend: str
    worker_count: int
    use_parallel: bool
    chunk_size: int
    selection_cache: _RDFSelectionCache | None = None
    cell_lengths: np.ndarray | None = None
    volumes: np.ndarray | None = None
    generic_strategy_override: str | None = None
    pair_count: int = 0


@dataclass
class _RDFStatisticsMoments:
    """Compact per-bin RDF statistics moments accumulated during the main compute pass."""

    point_count: np.ndarray
    sample_n: np.ndarray
    sample_sum: np.ndarray
    sample_sumsq: np.ndarray
    block_sum: np.ndarray | None = None
    block_n: np.ndarray | None = None


@dataclass(frozen=True)
class _RDFPairJob:
    """Prepared one-pair RDF job reused by pairwise RDF collection compute."""

    species_a: str
    species_b: str
    backend: str
    selection_cache: _RDFSelectionCache
    orthorhombic_config: _RDFOrthorhombicConfig


@dataclass(frozen=True)
class _RDFOrthorhombicMultiChunk:
    """Shared species-position chunk reused across multiple RDF pairs."""

    positions_by_species: Mapping[str, np.ndarray]
    cell_lengths: np.ndarray
    volumes: np.ndarray
    frame_count: int
    frame_start: int


def _resolve_rdf_block_index_by_frame(frame_count: int) -> np.ndarray | None:
    """Return one frame->block index map or ``None`` when block stats are unavailable."""
    block_resolution = resolve_block_slices(int(frame_count))
    if block_resolution is None:
        return None
    block_slices, _block_lengths = block_resolution
    block_index_by_frame = np.full(int(frame_count), -1, dtype=int)
    for block_index, block_slice in enumerate(block_slices):
        block_index_by_frame[block_slice] = int(block_index)
    return block_index_by_frame


def _empty_rdf_statistics_moments(
    *,
    n_bins: int,
    n_blocks: int | None,
) -> _RDFStatisticsMoments:
    """Allocate empty per-bin RDF statistics moments."""
    block_shape = None if n_blocks is None else (int(n_blocks), int(n_bins))
    return _RDFStatisticsMoments(
        point_count=np.zeros(int(n_bins), dtype=int),
        sample_n=np.zeros(int(n_bins), dtype=int),
        sample_sum=np.zeros(int(n_bins), dtype=float),
        sample_sumsq=np.zeros(int(n_bins), dtype=float),
        block_sum=None if block_shape is None else np.zeros(block_shape, dtype=float),
        block_n=None if block_shape is None else np.zeros(block_shape, dtype=int),
    )


def _update_rdf_statistics_moments(
    moments: _RDFStatisticsMoments | None,
    *,
    counts: np.ndarray,
    expected: np.ndarray,
    frame_index: int,
    block_index_by_frame: np.ndarray | None,
) -> None:
    """Accumulate one frame's exact RDF values into per-bin sample/block moments."""
    if moments is None:
        return
    counts_array = np.asarray(counts, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    moments.point_count += np.rint(counts_array).astype(int)
    finite = expected_array > 0.0
    if not np.any(finite):
        return
    g_values = np.full(counts_array.shape, np.nan, dtype=float)
    g_values[finite] = counts_array[finite] / expected_array[finite]
    moments.sample_n[finite] += 1
    moments.sample_sum[finite] += g_values[finite]
    moments.sample_sumsq[finite] += g_values[finite] ** 2
    if (
        block_index_by_frame is None
        or moments.block_sum is None
        or moments.block_n is None
        or frame_index < 0
        or frame_index >= block_index_by_frame.size
    ):
        return
    block_index = int(block_index_by_frame[frame_index])
    if block_index < 0:
        return
    moments.block_sum[block_index, finite] += g_values[finite]
    moments.block_n[block_index, finite] += 1


def _merge_rdf_statistics_moments(
    destination: _RDFStatisticsMoments | None,
    source: _RDFStatisticsMoments | None,
) -> _RDFStatisticsMoments | None:
    """Merge one chunk-local RDF statistics payload into the running total."""
    if destination is None:
        return source
    if source is None:
        return destination
    destination.point_count += source.point_count
    destination.sample_n += source.sample_n
    destination.sample_sum += source.sample_sum
    destination.sample_sumsq += source.sample_sumsq
    if destination.block_sum is not None and source.block_sum is not None:
        destination.block_sum += source.block_sum
    if destination.block_n is not None and source.block_n is not None:
        destination.block_n += source.block_n
    return destination


def _block_mean_matrix_from_rdf_moments(
    moments: _RDFStatisticsMoments,
) -> np.ndarray | None:
    """Return block-mean values reconstructed from accumulated block sums/counts."""
    if moments.block_sum is None or moments.block_n is None:
        return None
    block_values = np.full(moments.block_sum.shape, np.nan, dtype=float)
    valid = moments.block_n > 0
    block_values[valid] = moments.block_sum[valid] / moments.block_n[valid].astype(float)
    return block_values


def _finalize_rdf_statistics_moments(
    moments: _RDFStatisticsMoments,
) -> SeriesStatistics:
    """Build persisted RDF series statistics from accumulated moments."""
    return build_series_statistics_from_moments(
        point_count=moments.point_count,
        sample_n=moments.sample_n,
        sample_sum=moments.sample_sum,
        sample_sumsq=moments.sample_sumsq,
        block_values=_block_mean_matrix_from_rdf_moments(moments),
    )


def _format_atom_selector_label(atom_indices: Sequence[int] | np.ndarray) -> str:
    values = [int(value) for value in np.asarray(atom_indices, dtype=int).tolist()]
    if not values:
        return "atoms[]"

    chunks: list[str] = []
    start = values[0]
    end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        chunks.append(str(start) if start == end else f"{start}..{end}")
        start = value
        end = value
    chunks.append(str(start) if start == end else f"{start}..{end}")
    return f"atoms[{','.join(chunks)}]"


def _resolve_rdf_selector(
    *,
    species: str | None,
    atom_indices: Sequence[int] | np.ndarray | None,
) -> _RDFResolvedSelector:
    if atom_indices is not None:
        raw_indices = np.asarray(atom_indices, dtype=int).reshape(-1)
        ordered_indices: list[int] = []
        seen: set[int] = set()
        for raw_value in raw_indices.tolist():
            value = int(raw_value)
            if value in seen:
                continue
            seen.add(value)
            ordered_indices.append(value)
        resolved_indices = np.asarray(ordered_indices, dtype=int)
        if resolved_indices.size == 0:
            raise ValueError("RDF atom-index selection produced no atoms.")
        if np.any(resolved_indices < 0):
            raise ValueError("RDF atom-index selections must use indices >= 0.")
        return _RDFResolvedSelector(
            selection_kind="atoms",
            label=_format_atom_selector_label(resolved_indices),
            species_label=None,
            atom_indices=np.asarray(resolved_indices, dtype=int),
        )

    normalized_species = _normalize_species(species)
    return _RDFResolvedSelector(
        selection_kind="species",
        label=normalized_species,
        species_label=normalized_species,
        atom_indices=None,
    )


def _canonical_rdf_pair(species_a: str, species_b: str) -> tuple[str, str]:
    """Return the unique unordered storage key for a physical RDF pair."""
    label_a = _normalize_species(species_a)
    label_b = _normalize_species(species_b)
    return (label_a, label_b) if label_a <= label_b else (label_b, label_a)


def _ordered_unique_unordered_rdf_pairs(
    pairs: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return unique physical RDF pairs preserving first-seen canonical order."""
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for species_a, species_b in pairs:
        pair = _canonical_rdf_pair(species_a, species_b)
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append(pair)
    return ordered


def _pair_request_is_cross_species(
    wanted_species_a: str | None,
    wanted_species_b: str | None,
) -> bool:
    return (
        wanted_species_a is not None
        and wanted_species_b is not None
        and wanted_species_a != wanted_species_b
    )


def _rdf_pair_matches_request(
    *,
    stored_species_a: str,
    stored_species_b: str,
    wanted_species_a: str | None,
    wanted_species_b: str | None,
) -> bool:
    """Return whether a stored RDF pair satisfies a requested pair filter."""
    stored_a = _normalize_species(stored_species_a)
    stored_b = _normalize_species(stored_species_b)
    if wanted_species_a is not None and wanted_species_b is not None:
        if stored_a == wanted_species_a and stored_b == wanted_species_b:
            return True
        return _pair_request_is_cross_species(wanted_species_a, wanted_species_b) and (
            stored_a == wanted_species_b and stored_b == wanted_species_a
        )
    if wanted_species_a is not None and stored_a != wanted_species_a:
        return False
    if wanted_species_b is not None and stored_b != wanted_species_b:
        return False
    return True


def _rdf_pair_matches_exact_request(
    *,
    stored_species_a: str,
    stored_species_b: str,
    wanted_species_a: str | None,
    wanted_species_b: str | None,
) -> bool:
    """Return whether a stored RDF pair matches the requested exact order."""
    stored_a = _normalize_species(stored_species_a)
    stored_b = _normalize_species(stored_species_b)
    if wanted_species_a is not None and stored_a != wanted_species_a:
        return False
    if wanted_species_b is not None and stored_b != wanted_species_b:
        return False
    return True


def _select_mask(numbers: np.ndarray, species: str) -> np.ndarray:
    """Return atom-selection mask for one frame."""
    if species == "ALL":
        return np.ones(numbers.size, dtype=bool)

    atomic_number = atomic_numbers.get(species)
    if atomic_number is None:
        return np.zeros(numbers.size, dtype=bool)
    return numbers == atomic_number


def _frame_volume(frame: Atoms) -> float:
    """Return absolute frame volume and validate it is usable."""
    pbc = np.asarray(frame.get_pbc(), dtype=bool)
    if pbc.shape != (3,) or not bool(np.all(pbc)):
        raise ValueError("RDF requires fully periodic boundary conditions in all frames.")
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("RDF requires valid periodic cell vectors in all frames.")

    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0:
        raise ValueError("RDF requires non-zero cell volume in all frames.")
    return volume


def _frame_perpendicular_heights(frame: Atoms) -> tuple[float, float, float]:
    """Return perpendicular cell heights for one periodic cell."""
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("RDF requires valid periodic cell vectors in all frames.")
    volume = _frame_volume(frame)
    a_vec, b_vec, c_vec = np.asarray(cell, dtype=float)
    face_areas = np.asarray(
        [
            np.linalg.norm(np.cross(b_vec, c_vec)),
            np.linalg.norm(np.cross(c_vec, a_vec)),
            np.linalg.norm(np.cross(a_vec, b_vec)),
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(face_areas)) or np.any(face_areas <= 0.0):
        raise ValueError("RDF requires valid periodic cell vectors in all frames.")
    heights = volume / face_areas
    return float(heights[0]), float(heights[1]), float(heights[2])


def _frame_has_axis_aligned_orthorhombic_cell(frame: Atoms) -> bool:
    """Return whether a frame uses a positive diagonal orthorhombic cell."""
    pbc = np.asarray(frame.get_pbc(), dtype=bool)
    if pbc.shape != (3,) or not bool(np.all(pbc)):
        return False
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return False
    diagonal = np.diag(np.diag(cell))
    if not np.allclose(cell, diagonal, rtol=0.0, atol=1.0e-12):
        return False
    return bool(np.all(np.diag(diagonal) > 0.0))


def _resolve_rdf_orthorhombic_geometry(
    frames: list[Atoms],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return per-frame orthorhombic cell lengths/volumes when available."""
    if not frames:
        return None

    cell_lengths = np.empty((len(frames), 3), dtype=float)
    volumes = np.empty(len(frames), dtype=float)
    for frame_index, frame in enumerate(frames):
        if not _frame_has_axis_aligned_orthorhombic_cell(frame):
            return None
        diagonal = np.asarray(np.diag(frame.cell.array), dtype=float)
        cell_lengths[frame_index, :] = diagonal
        volumes[frame_index] = float(np.prod(diagonal))
    return cell_lengths, volumes


def _histogram_rdf_distances(
    sampled_distances: np.ndarray,
    *,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Histogram RDF distances with NumPy-compatible edge semantics."""
    distances = np.asarray(sampled_distances, dtype=float).reshape(-1)
    n_bins = max(0, int(bin_edges.size) - 1)
    if n_bins == 0 or distances.size == 0:
        return np.zeros(n_bins, dtype=float)

    finite = np.isfinite(distances) & (distances >= 0.0) & (distances <= float(bin_edges[-1]))
    distances = distances[finite]
    if distances.size == 0:
        return np.zeros(n_bins, dtype=float)

    bin_indices = np.searchsorted(bin_edges, distances, side="right") - 1
    last_index = n_bins - 1
    right_edge_mask = np.isclose(
        distances,
        float(bin_edges[-1]),
        rtol=0.0,
        atol=np.finfo(float).eps * max(1.0, abs(float(bin_edges[-1]))),
    )
    bin_indices[right_edge_mask] = last_index
    valid_bins = (bin_indices >= 0) & (bin_indices <= last_index)
    return np.bincount(bin_indices[valid_bins], minlength=n_bins).astype(float, copy=False)


def _resolve_rdf_self_pair_coordinates(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column coordinates for literal self-pairs shared by two selections."""
    if indices_a.size == 0 or indices_b.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    _shared, rows, cols = np.intersect1d(
        np.asarray(indices_a, dtype=int),
        np.asarray(indices_b, dtype=int),
        assume_unique=True,
        return_indices=True,
    )
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _resolve_rdf_orthorhombic_selection_cache(
    frames: list[Atoms],
    *,
    label_a: str,
    label_b: str,
    selection_cache: _RDFSelectionCache | None,
) -> _RDFSelectionCache | None:
    """Resolve stable selection metadata required by orthorhombic fast backends."""
    if selection_cache is not None:
        return selection_cache
    if not frames or not (label_a == "ALL" and label_b == "ALL"):
        return None

    atom_count = len(frames[0])
    for frame_index, frame in enumerate(frames[1:], start=1):
        if len(frame) != atom_count:
            LOGGER.warning(
                "RDF atom count changed at frame %d; falling back to framewise RDF backend.",
                frame_index,
            )
            return None
    mask = np.ones(atom_count, dtype=bool)
    indices = np.arange(atom_count, dtype=int)
    return _RDFSelectionCache(
        mask_a=mask,
        mask_b=mask,
        indices_a=indices,
        indices_b=indices,
        count_a=int(atom_count),
        count_b=int(atom_count),
        overlap_count=int(atom_count),
    )


def _sample_distances_neighbor_list(
    frame: Atoms,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    r_max: float,
) -> np.ndarray:
    """Collect selected ordered-pair distances using a cutoff neighbor list."""
    i_pairs, j_pairs, pair_distances = neighbor_list(
        "ijd", frame, float(np.nextafter(r_max, np.inf))
    )
    # Keep ordered i->j pairs and exclude only literal self-pairs.
    pair_mask = mask_a[i_pairs] & mask_b[j_pairs] & (i_pairs != j_pairs)
    if not np.any(pair_mask):
        return np.empty(0, dtype=float)

    i_selected = i_pairs[pair_mask]
    j_selected = j_pairs[pair_mask]
    sampled_distances = pair_distances[pair_mask]
    if sampled_distances.size > 1:
        # neighbor_list can return multiple periodic images for one ordered pair.
        # Reduce to minimum distance to match MIC matrix semantics.
        n_atoms = mask_a.size
        pair_ids = i_selected.astype(np.int64) * n_atoms + j_selected.astype(np.int64)
        order = np.argsort(pair_ids, kind="mergesort")
        sorted_ids = pair_ids[order]
        sorted_distances = sampled_distances[order]
        unique_starts = np.empty(sorted_ids.size, dtype=bool)
        unique_starts[0] = True
        unique_starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
        sampled_distances = np.minimum.reduceat(sorted_distances, np.flatnonzero(unique_starts))
    finite = np.isfinite(sampled_distances)
    return sampled_distances[finite & (sampled_distances >= 0.0) & (sampled_distances <= r_max)]


def _sample_distances_selected_matrix(
    frame: Atoms,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    *,
    same_selection: bool,
    r_max: float,
) -> np.ndarray:
    """Collect selected ordered-pair distances using only the relevant submatrix."""
    _, pair_distances = get_distances(
        frame.positions[indices_a],
        frame.positions[indices_b],
        cell=frame.cell.array,
        pbc=frame.get_pbc(),
    )
    same_atom_mask = indices_a[:, np.newaxis] != indices_b[np.newaxis, :]
    pair_distances = pair_distances[same_atom_mask]
    return pair_distances[
        np.isfinite(pair_distances) & (pair_distances >= 0.0) & (pair_distances <= r_max)
    ]


def _sample_distances_full_matrix(
    frame: Atoms,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    *,
    same_selection: bool,
    r_max: float,
) -> np.ndarray:
    """Collect selected ordered-pair distances from a full MIC matrix."""
    all_distances = np.asarray(frame.get_all_distances(mic=True), dtype=float)
    pair_distances = all_distances[np.ix_(indices_a, indices_b)]
    same_atom_mask = indices_a[:, np.newaxis] != indices_b[np.newaxis, :]
    pair_distances = pair_distances[same_atom_mask]
    return pair_distances[
        np.isfinite(pair_distances) & (pair_distances >= 0.0) & (pair_distances <= r_max)
    ]


def _resolve_rdf_worker_count(threads: int | None, n_frames: int) -> int:
    """Resolve requested RDF worker count."""
    if threads is None:
        env_threads = os.getenv("LINAK_RDF_THREADS")
        if env_threads:
            try:
                threads = int(env_threads)
            except ValueError:
                LOGGER.warning(
                    "Ignoring invalid LINAK_RDF_THREADS=%r; using default thread count.",
                    env_threads,
                )
                threads = None

    if threads is None:
        threads = min(_DEFAULT_RDF_THREAD_CAP, os.cpu_count() or 1)

    if threads < 1:
        raise ValueError("RDF threads must be >= 1.")
    return min(threads, max(1, n_frames))


def _build_uniform_rdf_bins(
    *,
    r_max: float,
    target_bin_width: float,
) -> tuple[np.ndarray, float]:
    """Construct uniform RDF bin edges that end exactly at ``r_max``."""
    ratio = r_max / target_bin_width
    n_bins = max(1, int(np.round(ratio)))
    effective_bin_width = float(r_max) / float(n_bins)
    bin_edges = np.linspace(0.0, float(r_max), n_bins + 1, dtype=float)
    return bin_edges, effective_bin_width


def _resolve_auto_r_max_for_bin_width(*, auto_r_max: float, target_bin_width: float) -> float:
    """Round an auto-derived safe ``r_max`` down to a compatible uniform-bin endpoint."""
    n_bins = max(1, int(np.floor(float(auto_r_max) / float(target_bin_width))))
    resolved = float(n_bins) * float(target_bin_width)
    return resolved


def _shell_volumes_from_edges(bin_edges: np.ndarray) -> np.ndarray:
    """Return exact spherical shell volumes for uniform RDF bins."""
    edges = np.asarray(bin_edges, dtype=float)
    return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def _auto_r_max_from_frames(frames: list[Atoms]) -> float:
    """Return a safe default RDF cutoff from the minimum periodic cell height."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")
    min_height = float(min(min(_frame_perpendicular_heights(frame)) for frame in frames))
    return 0.5 * min_height


def _normalize_strategy_override(strategy_override: str | None) -> str | None:
    if strategy_override is None:
        return None
    token = str(strategy_override).strip().lower()
    if token not in {
        "neighbor_list",
        "selected_matrix",
        "full_matrix",
        _RDF_BACKEND_DENSE,
        _RDF_BACKEND_SPARSE,
        _RDF_BACKEND_GENERIC,
    }:
        raise ValueError(
            "RDF strategy override must be one of: "
            "neighbor_list, selected_matrix, full_matrix, "
            "chunked_dense_matrix, chunked_sparse_cutoff, framewise_generic_fallback."
        )
    return token


def _resolve_rdf_chunk_size(n_frames: int, worker_count: int) -> int:
    target_chunks = max(worker_count * 8, 1)
    chunk_size = max(_RDF_MIN_CHUNK_SIZE, (n_frames + target_chunks - 1) // target_chunks)
    return min(_RDF_MAX_CHUNK_SIZE, chunk_size)


def _resolve_rdf_fast_chunk_size(
    *,
    frame_count: int,
    pair_count: int,
    worker_count: int,
) -> int:
    """Resolve chunk size for orthorhombic fast RDF kernels."""
    if frame_count <= 0:
        return 1
    if pair_count <= 0:
        return frame_count

    target_chunks = max(worker_count * _RDF_FAST_TARGET_CHUNKS_PER_WORKER, 1)
    parallel_chunk = max(1, (frame_count + target_chunks - 1) // target_chunks)
    memory_chunk = max(1, _RDF_FAST_MAX_DISTANCE_VALUES_PER_CHUNK // max(1, pair_count))
    return max(1, min(frame_count, parallel_chunk, memory_chunk))


def _should_parallelize_rdf(n_frames: int, worker_count: int) -> bool:
    if worker_count <= 1:
        return False
    return n_frames >= max(_RDF_MIN_CHUNK_SIZE, worker_count * _RDF_MIN_PARALLEL_FRAMES_PER_WORKER)


def _resolve_rdf_backend(
    frames: list[Atoms],
    *,
    label_a: str,
    label_b: str,
    r_max: float,
    worker_count: int,
    selection_cache: _RDFSelectionCache | None,
    strategy_override: str | None,
) -> _RDFBackendResolution:
    """Resolve the fastest exact RDF backend that preserves current semantics."""
    generic_strategy_override = None
    if strategy_override in {"neighbor_list", "selected_matrix", "full_matrix"}:
        generic_strategy_override = strategy_override
        strategy_override = _RDF_BACKEND_GENERIC

    orthorhombic_geometry = _resolve_rdf_orthorhombic_geometry(frames)
    orthorhombic_selection = _resolve_rdf_orthorhombic_selection_cache(
        frames,
        label_a=label_a,
        label_b=label_b,
        selection_cache=selection_cache,
    )
    orthorhombic_ready = orthorhombic_geometry is not None and orthorhombic_selection is not None

    if strategy_override in {_RDF_BACKEND_DENSE, _RDF_BACKEND_SPARSE} and not orthorhombic_ready:
        raise ValueError(
            f"RDF backend override '{strategy_override}' requires axis-aligned orthorhombic "
            "cells and stable atom selections across frames."
        )

    if strategy_override == _RDF_BACKEND_GENERIC or not orthorhombic_ready:
        chunk_size = (
            _resolve_rdf_chunk_size(len(frames), worker_count)
            if _should_parallelize_rdf(len(frames), worker_count)
            else len(frames)
        )
        return _RDFBackendResolution(
            backend=_RDF_BACKEND_GENERIC,
            worker_count=worker_count,
            use_parallel=_should_parallelize_rdf(len(frames), worker_count),
            chunk_size=chunk_size,
            selection_cache=selection_cache,
            generic_strategy_override=generic_strategy_override,
        )

    if orthorhombic_geometry is None or orthorhombic_selection is None:
        raise ValueError("Orthorhombic RDF backend resolution requires geometry and selections.")
    cell_lengths, volumes = orthorhombic_geometry
    pair_count = int(orthorhombic_selection.count_a * orthorhombic_selection.count_b)
    mean_volume = float(np.mean(volumes))
    max_sphere_volume = (4.0 / 3.0) * np.pi * (float(r_max) ** 3)
    neighbor_density = min(1.0, max_sphere_volume / max(mean_volume, 1.0e-12))

    if strategy_override == _RDF_BACKEND_DENSE:
        backend = _RDF_BACKEND_DENSE
    elif strategy_override == _RDF_BACKEND_SPARSE:
        backend = _RDF_BACKEND_SPARSE
    elif (
        pair_count <= _RDF_DENSE_PAIR_THRESHOLD
        or neighbor_density > _NEIGHBORLIST_DENSITY_THRESHOLD
    ):
        backend = _RDF_BACKEND_DENSE
    else:
        backend = _RDF_BACKEND_SPARSE

    use_parallel = _should_parallelize_rdf(len(frames), worker_count)
    chunk_size = _resolve_rdf_fast_chunk_size(
        frame_count=len(frames),
        pair_count=max(1, pair_count),
        worker_count=worker_count,
    )
    return _RDFBackendResolution(
        backend=backend,
        worker_count=worker_count,
        use_parallel=use_parallel,
        chunk_size=chunk_size,
        selection_cache=orthorhombic_selection,
        cell_lengths=cell_lengths,
        volumes=volumes,
        pair_count=pair_count,
    )


def _iter_rdf_frame_chunks(
    frames: list[Atoms],
    chunk_size: int,
) -> list[list[tuple[int, Atoms]]]:
    return [
        list(enumerate(frames[start : start + chunk_size], start))
        for start in range(0, len(frames), chunk_size)
    ]


def _iter_rdf_orthorhombic_chunks(
    frames: list[Atoms],
    *,
    selection_cache: _RDFSelectionCache,
    cell_lengths: np.ndarray,
    volumes: np.ndarray,
    chunk_size: int,
) -> list[_RDFOrthorhombicChunk]:
    """Build compact orthorhombic chunk payloads from selected positions only."""
    chunks: list[_RDFOrthorhombicChunk] = []
    indices_a = np.asarray(selection_cache.indices_a, dtype=int)
    indices_b = np.asarray(selection_cache.indices_b, dtype=int)
    for start in range(0, len(frames), chunk_size):
        stop = min(len(frames), start + chunk_size)
        positions_a = np.stack(
            [np.asarray(frame.positions[indices_a], dtype=float) for frame in frames[start:stop]],
            axis=0,
        )
        positions_b = np.stack(
            [np.asarray(frame.positions[indices_b], dtype=float) for frame in frames[start:stop]],
            axis=0,
        )
        chunks.append(
            _RDFOrthorhombicChunk(
                positions_a=positions_a,
                positions_b=positions_b,
                cell_lengths=np.asarray(cell_lengths[start:stop], dtype=float),
                volumes=np.asarray(volumes[start:stop], dtype=float),
                frame_count=stop - start,
                frame_start=start,
            )
        )
    return chunks


def _iter_rdf_orthorhombic_multi_chunks(
    frames: list[Atoms],
    *,
    selection_caches_by_species: Mapping[str, _RDFSelectionCache],
    cell_lengths: np.ndarray,
    volumes: np.ndarray,
    chunk_size: int,
) -> list[_RDFOrthorhombicMultiChunk]:
    """Build compact orthorhombic chunk payloads shared across RDF species pairs."""
    chunks: list[_RDFOrthorhombicMultiChunk] = []
    for start in range(0, len(frames), chunk_size):
        stop = min(len(frames), start + chunk_size)
        positions_by_species = {
            species: np.stack(
                [
                    np.asarray(frame.positions[cache.indices_a], dtype=float)
                    for frame in frames[start:stop]
                ],
                axis=0,
            )
            for species, cache in selection_caches_by_species.items()
        }
        chunks.append(
            _RDFOrthorhombicMultiChunk(
                positions_by_species=positions_by_species,
                cell_lengths=np.asarray(cell_lengths[start:stop], dtype=float),
                volumes=np.asarray(volumes[start:stop], dtype=float),
                frame_count=stop - start,
                frame_start=start,
            )
        )
    return chunks


def _resolve_rdf_expected_counts_from_volumes(
    volumes: np.ndarray,
    *,
    config: _RDFOrthorhombicConfig,
) -> np.ndarray:
    """Return accumulated expected RDF counts for one chunk of frames."""
    valid_ordered_pair_count = (config.count_a * config.count_b) - config.overlap_count
    if config.count_a <= 0 or config.count_b <= 0 or valid_ordered_pair_count <= 0:
        return np.zeros_like(config.shell_volumes, dtype=float)
    scale = float(valid_ordered_pair_count) * float(np.sum(1.0 / np.asarray(volumes, dtype=float)))
    return np.asarray(config.shell_volumes, dtype=float) * scale


def _resolve_rdf_expected_counts_from_volume(
    volume: float,
    *,
    config: _RDFOrthorhombicConfig,
) -> np.ndarray:
    """Return expected RDF counts for one orthorhombic frame volume."""
    valid_ordered_pair_count = (config.count_a * config.count_b) - config.overlap_count
    if config.count_a <= 0 or config.count_b <= 0 or valid_ordered_pair_count <= 0:
        return np.zeros_like(config.shell_volumes, dtype=float)
    scale = float(valid_ordered_pair_count) / float(volume)
    return np.asarray(config.shell_volumes, dtype=float) * scale


def _wrap_orthorhombic_positions(positions: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    """Wrap positions into an orthorhombic unit cell for periodic KD-tree queries."""
    wrapped = np.mod(np.asarray(positions, dtype=float), np.asarray(cell_lengths, dtype=float))
    return np.asarray(wrapped, dtype=float)


def _compute_orthorhombic_pair_distances(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    *,
    cell_lengths: np.ndarray,
) -> np.ndarray:
    """Compute exact orthorhombic MIC distances for selected pairs."""
    deltas = np.asarray(positions_a, dtype=float) - np.asarray(positions_b, dtype=float)
    lengths = np.asarray(cell_lengths, dtype=float)
    deltas -= lengths * np.round(deltas / lengths)
    return np.linalg.norm(deltas, axis=1)


def _collect_sparse_query_pairs(
    neighbor_lists: list[list[int]],
    *,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten KD-tree neighbor-list output into ordered local pair indices."""
    pair_i: list[np.ndarray] = []
    pair_j: list[np.ndarray] = []
    for local_i, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        neighbor_indices = np.asarray(neighbors, dtype=int)
        keep = np.asarray(indices_b[neighbor_indices] != indices_a[local_i], dtype=bool)
        if not np.any(keep):
            continue
        filtered = neighbor_indices[keep]
        pair_i.append(np.full(filtered.size, local_i, dtype=int))
        pair_j.append(filtered)

    if not pair_i:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(pair_i), np.concatenate(pair_j)


def _compute_rdf_dense_orthorhombic_chunk_contributions(
    chunk: _RDFOrthorhombicChunk,
    config: _RDFOrthorhombicConfig,
) -> tuple[np.ndarray, np.ndarray, int, _RDFStatisticsMoments | None]:
    """Compute one dense orthorhombic RDF chunk exactly."""
    deltas = chunk.positions_a[:, :, np.newaxis, :] - chunk.positions_b[:, np.newaxis, :, :]
    deltas -= chunk.cell_lengths[:, np.newaxis, np.newaxis, :] * np.round(
        deltas / chunk.cell_lengths[:, np.newaxis, np.newaxis, :]
    )
    distances = np.linalg.norm(deltas, axis=3)
    if config.self_pair_rows.size:
        distances[:, config.self_pair_rows, config.self_pair_cols] = np.inf
    counts = np.zeros(config.bin_edges.size - 1, dtype=float)
    moments = (
        _empty_rdf_statistics_moments(
            n_bins=config.bin_edges.size - 1,
            n_blocks=(
                None
                if config.block_index_by_frame is None
                else int(np.max(config.block_index_by_frame)) + 1
            ),
        )
        if config.collect_statistics
        else None
    )
    for frame_local in range(chunk.frame_count):
        frame_counts = _histogram_rdf_distances(distances[frame_local], bin_edges=config.bin_edges)
        counts += frame_counts
        if moments is not None:
            frame_expected = _resolve_rdf_expected_counts_from_volume(
                float(chunk.volumes[frame_local]),
                config=config,
            )
            _update_rdf_statistics_moments(
                moments,
                counts=frame_counts,
                expected=frame_expected,
                frame_index=chunk.frame_start + frame_local,
                block_index_by_frame=config.block_index_by_frame,
            )
    expected = _resolve_rdf_expected_counts_from_volumes(chunk.volumes, config=config)
    return counts, expected, chunk.frame_count, moments


def _compute_rdf_sparse_orthorhombic_chunk_contributions(
    chunk: _RDFOrthorhombicChunk,
    config: _RDFOrthorhombicConfig,
) -> tuple[np.ndarray, np.ndarray, int, _RDFStatisticsMoments | None]:
    """Compute one sparse orthorhombic RDF chunk exactly via periodic KD-tree queries."""
    counts_accum = np.zeros(config.bin_edges.size - 1, dtype=float)
    expected_accum = _resolve_rdf_expected_counts_from_volumes(chunk.volumes, config=config)
    moments = (
        _empty_rdf_statistics_moments(
            n_bins=config.bin_edges.size - 1,
            n_blocks=(
                None
                if config.block_index_by_frame is None
                else int(np.max(config.block_index_by_frame)) + 1
            ),
        )
        if config.collect_statistics
        else None
    )

    for frame_local in range(chunk.frame_count):
        lengths = np.asarray(chunk.cell_lengths[frame_local], dtype=float)
        positions_a = _wrap_orthorhombic_positions(chunk.positions_a[frame_local], lengths)
        positions_b = _wrap_orthorhombic_positions(chunk.positions_b[frame_local], lengths)
        tree_a = cKDTree(positions_a, boxsize=lengths)
        if config.same_selection:
            neighbor_lists = tree_a.query_ball_tree(tree_a, r=float(config.r_max))
        else:
            tree_b = cKDTree(positions_b, boxsize=lengths)
            neighbor_lists = tree_a.query_ball_tree(tree_b, r=float(config.r_max))
        pair_i, pair_j = _collect_sparse_query_pairs(
            neighbor_lists,
            indices_a=config.indices_a,
            indices_b=config.indices_b,
        )
        if pair_i.size == 0:
            if moments is not None:
                _update_rdf_statistics_moments(
                    moments,
                    counts=np.zeros(config.bin_edges.size - 1, dtype=float),
                    expected=_resolve_rdf_expected_counts_from_volume(
                        float(chunk.volumes[frame_local]),
                        config=config,
                    ),
                    frame_index=chunk.frame_start + frame_local,
                    block_index_by_frame=config.block_index_by_frame,
                )
            continue
        pair_distances = _compute_orthorhombic_pair_distances(
            positions_a[pair_i],
            positions_b[pair_j],
            cell_lengths=lengths,
        )
        frame_counts = _histogram_rdf_distances(pair_distances, bin_edges=config.bin_edges)
        counts_accum += frame_counts
        if moments is not None:
            _update_rdf_statistics_moments(
                moments,
                counts=frame_counts,
                expected=_resolve_rdf_expected_counts_from_volume(
                    float(chunk.volumes[frame_local]),
                    config=config,
                ),
                frame_index=chunk.frame_start + frame_local,
                block_index_by_frame=config.block_index_by_frame,
            )
    return counts_accum, expected_accum, chunk.frame_count, moments


def _resolve_rdf_pairwise_jobs(
    *,
    pairs: list[tuple[str, str]],
    bin_edges: np.ndarray,
    shell_volumes: np.ndarray,
    mean_volume: float,
    r_max: float,
    selection_caches_by_species: Mapping[str, _RDFSelectionCache],
    collect_statistics: bool = False,
    block_index_by_frame: np.ndarray | None = None,
) -> list[_RDFPairJob]:
    """Resolve prepared orthorhombic RDF jobs for one pairwise collection compute."""
    jobs: list[_RDFPairJob] = []
    max_sphere_volume = (4.0 / 3.0) * np.pi * (float(r_max) ** 3)
    neighbor_density = min(1.0, max_sphere_volume / max(float(mean_volume), 1.0e-12))
    for species_a, species_b in pairs:
        cache_a = selection_caches_by_species[species_a]
        cache_b = selection_caches_by_species[species_b]
        same_selection = species_a == species_b
        selection_cache = _build_rdf_pair_selection_cache(
            cache_a,
            cache_b,
            same_selection=same_selection,
        )
        pair_count = int(selection_cache.count_a * selection_cache.count_b)
        backend = (
            _RDF_BACKEND_DENSE
            if pair_count <= _RDF_DENSE_PAIR_THRESHOLD
            or neighbor_density > _NEIGHBORLIST_DENSITY_THRESHOLD
            else _RDF_BACKEND_SPARSE
        )
        self_pair_rows, self_pair_cols = _resolve_rdf_self_pair_coordinates(
            selection_cache.indices_a,
            selection_cache.indices_b,
        )
        jobs.append(
            _RDFPairJob(
                species_a=species_a,
                species_b=species_b,
                backend=backend,
                selection_cache=selection_cache,
                orthorhombic_config=_RDFOrthorhombicConfig(
                    r_max=float(r_max),
                    bin_edges=np.asarray(bin_edges, dtype=float),
                    shell_volumes=np.asarray(shell_volumes, dtype=float),
                    same_selection=same_selection,
                    count_a=int(selection_cache.count_a),
                    count_b=int(selection_cache.count_b),
                    overlap_count=int(selection_cache.overlap_count),
                    indices_a=np.asarray(selection_cache.indices_a, dtype=int),
                    indices_b=np.asarray(selection_cache.indices_b, dtype=int),
                    self_pair_rows=self_pair_rows,
                    self_pair_cols=self_pair_cols,
                    collect_statistics=collect_statistics,
                    block_index_by_frame=block_index_by_frame,
                ),
            )
        )
    return jobs


def _resolve_rdf_pair_selection_caches(
    *,
    pairs: Sequence[tuple[str, str]],
    selection_caches_by_species: Mapping[str, _RDFSelectionCache] | None,
) -> dict[tuple[str, str], _RDFSelectionCache | None]:
    """Resolve pair-selection caches from shared per-species caches when available."""
    if selection_caches_by_species is None:
        return {pair: None for pair in pairs}

    resolved: dict[tuple[str, str], _RDFSelectionCache | None] = {}
    for species_a, species_b in pairs:
        cache_a = selection_caches_by_species[species_a]
        cache_b = selection_caches_by_species[species_b]
        resolved[(species_a, species_b)] = _build_rdf_pair_selection_cache(
            cache_a,
            cache_b,
            same_selection=(species_a == species_b),
        )
    return resolved


def _accumulate_rdf_pair_collection(
    frames: list[Atoms],
    *,
    pairs: Sequence[tuple[str, str]],
    r_max: float | None,
    bin_width: float,
    threads: int | None = None,
    progress_desc: str | None = None,
    collect_statistics: bool = False,
) -> tuple[
    list[tuple[str, str]],
    np.ndarray,
    np.ndarray,
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], _RDFSelectionCache | None],
    dict[tuple[str, str], _RDFStatisticsMoments | None],
]:
    """Accumulate exact RDF observed/expected counts for one or more physical pairs."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    unique_pairs = _ordered_unique_unordered_rdf_pairs(pairs)
    if not unique_pairs:
        raise ValueError("At least one RDF pair is required.")

    ensure_positive("bin_width", bin_width)
    requested_bin_width = float(bin_width)
    auto_r_max_raw = None
    if r_max is None:
        auto_r_max_raw = _auto_r_max_from_frames(frames)
        r_max = _resolve_auto_r_max_for_bin_width(
            auto_r_max=float(auto_r_max_raw),
            target_bin_width=requested_bin_width,
        )
    ensure_positive("r_max", r_max)
    resolved_r_max = float(r_max)
    bin_edges, _effective_bin_width = _build_uniform_rdf_bins(
        r_max=resolved_r_max,
        target_bin_width=requested_bin_width,
    )
    if auto_r_max_raw is not None and not np.isclose(
        resolved_r_max,
        float(auto_r_max_raw),
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        LOGGER.debug(
            "Rounded auto RDF r_max down from %.6g to %.6g Angstrom to match bin_width=%.6g.",
            float(auto_r_max_raw),
            resolved_r_max,
            requested_bin_width,
        )

    shell_volumes = _shell_volumes_from_edges(bin_edges)
    worker_count = _resolve_rdf_worker_count(threads, len(frames))
    species_labels = sorted({label for pair in unique_pairs for label in pair})
    orthorhombic_geometry = _resolve_rdf_orthorhombic_geometry(frames)
    selection_caches_by_species = _resolve_rdf_selection_caches_by_species(
        frames,
        species_labels=species_labels,
    )
    selection_cache_by_pair = _resolve_rdf_pair_selection_caches(
        pairs=unique_pairs,
        selection_caches_by_species=selection_caches_by_species,
    )

    counts_by_pair = {pair: np.zeros(bin_edges.size - 1, dtype=float) for pair in unique_pairs}
    expected_by_pair = {pair: np.zeros(bin_edges.size - 1, dtype=float) for pair in unique_pairs}
    block_index_by_frame = (
        _resolve_rdf_block_index_by_frame(len(frames)) if collect_statistics else None
    )
    moments_by_pair = {
        pair: (
            _empty_rdf_statistics_moments(
                n_bins=bin_edges.size - 1,
                n_blocks=(
                    None if block_index_by_frame is None else int(np.max(block_index_by_frame)) + 1
                ),
            )
            if collect_statistics
            else None
        )
        for pair in unique_pairs
    }

    if orthorhombic_geometry is not None and selection_caches_by_species is not None:
        cell_lengths, volumes = orthorhombic_geometry
        mean_volume = float(np.mean(volumes))
        jobs = _resolve_rdf_pairwise_jobs(
            pairs=list(unique_pairs),
            bin_edges=bin_edges,
            shell_volumes=shell_volumes,
            mean_volume=mean_volume,
            r_max=resolved_r_max,
            selection_caches_by_species=selection_caches_by_species,
            collect_statistics=collect_statistics,
            block_index_by_frame=block_index_by_frame,
        )
        use_parallel = _should_parallelize_rdf(len(frames), worker_count)
        max_pair_count = max(
            int(job.selection_cache.count_a * job.selection_cache.count_b) for job in jobs
        )
        chunk_size = _resolve_rdf_fast_chunk_size(
            frame_count=len(frames),
            pair_count=max(1, max_pair_count),
            worker_count=worker_count,
        )
        chunks = _iter_rdf_orthorhombic_multi_chunks(
            frames,
            selection_caches_by_species=selection_caches_by_species,
            cell_lengths=cell_lengths,
            volumes=volumes,
            chunk_size=chunk_size,
        )

        progress_cm: ProgressBar | None
        if progress_desc is None:
            progress_cm = None
        else:
            progress_cm = ProgressBar(desc=progress_desc, total=len(frames), unit="frame")

        progress_context = nullcontext(progress_cm) if progress_cm is None else progress_cm
        with progress_context as progress:
            if use_parallel:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    for chunk_results in executor.map(
                        _compute_rdf_pairwise_orthorhombic_chunk_contributions,
                        chunks,
                        repeat(tuple(jobs)),
                    ):
                        processed_frames = 0
                        for (
                            species_a,
                            species_b,
                            counts,
                            expected,
                            processed,
                            chunk_moments,
                        ) in chunk_results:
                            counts_by_pair[(species_a, species_b)] += counts
                            expected_by_pair[(species_a, species_b)] += expected
                            moments_by_pair[(species_a, species_b)] = _merge_rdf_statistics_moments(
                                moments_by_pair[(species_a, species_b)],
                                chunk_moments,
                            )
                            processed_frames = max(processed_frames, processed)
                        if progress is not None:
                            progress.update(processed_frames)
            else:
                for chunk in chunks:
                    chunk_results = _compute_rdf_pairwise_orthorhombic_chunk_contributions(
                        chunk,
                        tuple(jobs),
                    )
                    processed_frames = 0
                    for (
                        species_a,
                        species_b,
                        counts,
                        expected,
                        processed,
                        chunk_moments,
                    ) in chunk_results:
                        counts_by_pair[(species_a, species_b)] += counts
                        expected_by_pair[(species_a, species_b)] += expected
                        moments_by_pair[(species_a, species_b)] = _merge_rdf_statistics_moments(
                            moments_by_pair[(species_a, species_b)],
                            chunk_moments,
                        )
                        processed_frames = max(processed_frames, processed)
                    if progress is not None:
                        progress.update(processed_frames)

        return (
            unique_pairs,
            np.asarray(bin_edges, dtype=float),
            np.asarray(shell_volumes, dtype=float),
            counts_by_pair,
            expected_by_pair,
            selection_cache_by_pair,
            moments_by_pair,
        )

    total_work = len(frames) * len(unique_pairs)
    progress_cm = (
        None
        if progress_desc is None
        else ProgressBar(desc=progress_desc, total=total_work, unit="frame")
    )
    progress_context = nullcontext(progress_cm) if progress_cm is None else progress_cm
    with progress_context as progress:
        for species_a, species_b in unique_pairs:
            same_selection = species_a == species_b
            selection_cache = selection_cache_by_pair[(species_a, species_b)]
            backend_resolution = _resolve_rdf_backend(
                frames,
                label_a=species_a,
                label_b=species_b,
                r_max=resolved_r_max,
                worker_count=worker_count,
                selection_cache=selection_cache,
                strategy_override=None,
            )
            generic_config = _RDFWorkerConfig(
                label_a=species_a,
                label_b=species_b,
                same_selection=same_selection,
                r_max=resolved_r_max,
                bin_edges=np.asarray(bin_edges, dtype=float),
                shell_volumes=np.asarray(shell_volumes, dtype=float),
                max_sphere_volume=(4.0 / 3.0) * np.pi * (resolved_r_max**3),
                selection_cache=selection_cache,
                strategy_override=backend_resolution.generic_strategy_override,
                collect_statistics=collect_statistics,
                block_index_by_frame=block_index_by_frame,
            )

            progress_proxy: Any
            if progress is None:

                class _NullProgressBar:
                    def update(self, _n: int = 1) -> None:
                        return None

                progress_proxy = _NullProgressBar()
            else:
                progress_proxy = progress

            counts_accum, expected_accum, moments = _compute_rdf_generic_backend(
                frames,
                config=generic_config,
                worker_count=backend_resolution.worker_count,
                use_parallel=backend_resolution.use_parallel,
                chunk_size=backend_resolution.chunk_size,
                progress=progress_proxy,
            )
            counts_by_pair[(species_a, species_b)] = counts_accum
            expected_by_pair[(species_a, species_b)] = expected_accum
            moments_by_pair[(species_a, species_b)] = moments

    return (
        unique_pairs,
        np.asarray(bin_edges, dtype=float),
        np.asarray(shell_volumes, dtype=float),
        counts_by_pair,
        expected_by_pair,
        selection_cache_by_pair,
        moments_by_pair,
    )


def _compute_rdf_pairwise_orthorhombic_chunk_contributions(
    chunk: _RDFOrthorhombicMultiChunk,
    jobs: tuple[_RDFPairJob, ...],
) -> list[tuple[str, str, np.ndarray, np.ndarray, int, _RDFStatisticsMoments | None]]:
    """Compute exact orthorhombic RDF contributions for all requested pairs in one chunk."""
    results: list[tuple[str, str, np.ndarray, np.ndarray, int, _RDFStatisticsMoments | None]] = []
    for job in jobs:
        pair_chunk = _RDFOrthorhombicChunk(
            positions_a=np.asarray(chunk.positions_by_species[job.species_a], dtype=float),
            positions_b=np.asarray(chunk.positions_by_species[job.species_b], dtype=float),
            cell_lengths=np.asarray(chunk.cell_lengths, dtype=float),
            volumes=np.asarray(chunk.volumes, dtype=float),
            frame_count=chunk.frame_count,
            frame_start=chunk.frame_start,
        )
        if job.backend == _RDF_BACKEND_DENSE:
            counts, expected, processed, moments = (
                _compute_rdf_dense_orthorhombic_chunk_contributions(
                    pair_chunk,
                    job.orthorhombic_config,
                )
            )
        else:
            counts, expected, processed, moments = (
                _compute_rdf_sparse_orthorhombic_chunk_contributions(
                    pair_chunk,
                    job.orthorhombic_config,
                )
            )
        results.append((job.species_a, job.species_b, counts, expected, processed, moments))
    return results


def _resolve_rdf_selection_cache(
    frames: list[Atoms],
    *,
    label_a: str,
    label_b: str,
    atom_indices_a: np.ndarray | None = None,
    atom_indices_b: np.ndarray | None = None,
) -> _RDFSelectionCache | None:
    """Reuse RDF selections when atom identities remain fixed across all frames."""
    if not frames:
        return None
    if atom_indices_a is None and atom_indices_b is None and label_a == "ALL" and label_b == "ALL":
        return None

    reference_numbers = np.asarray(frames[0].numbers, dtype=int)
    for frame_index, frame in enumerate(frames[1:], start=1):
        current_numbers = np.asarray(frame.numbers, dtype=int)
        if current_numbers.shape != reference_numbers.shape or not np.array_equal(
            current_numbers, reference_numbers
        ):
            if atom_indices_a is not None or atom_indices_b is not None:
                raise ValueError(
                    "Explicit RDF atom-index selections require stable atom identities/order "
                    f"across frames; mismatch detected at frame {frame_index}."
                )
            LOGGER.warning(
                "RDF atom identities/order changed at frame %d; "
                "falling back to per-frame species selection.",
                frame_index,
            )
            return None

    if atom_indices_a is not None:
        indices_a = np.asarray(atom_indices_a, dtype=int).reshape(-1)
        if np.any(indices_a >= reference_numbers.size):
            raise ValueError(
                f"RDF atom-index selection A contains out-of-range indices for {reference_numbers.size} atoms."
            )
        mask_a = np.zeros(reference_numbers.size, dtype=bool)
        mask_a[indices_a] = True
    else:
        mask_a = _select_mask(reference_numbers, label_a)
        indices_a = np.flatnonzero(mask_a)

    if atom_indices_b is not None:
        indices_b = np.asarray(atom_indices_b, dtype=int).reshape(-1)
        if np.any(indices_b >= reference_numbers.size):
            raise ValueError(
                f"RDF atom-index selection B contains out-of-range indices for {reference_numbers.size} atoms."
            )
        mask_b = np.zeros(reference_numbers.size, dtype=bool)
        mask_b[indices_b] = True
    else:
        mask_b = _select_mask(reference_numbers, label_b)
        indices_b = np.flatnonzero(mask_b)

    count_a = int(indices_a.size)
    count_b = int(indices_b.size)
    if count_a == 0 or count_b == 0:
        raise ValueError(
            f"RDF selection produced no atoms in frame 0 (species_a={label_a}, species_b={label_b})."
        )
    overlap_count = int(np.intersect1d(indices_a, indices_b, assume_unique=True).size)
    return _RDFSelectionCache(
        mask_a=mask_a,
        mask_b=mask_b,
        indices_a=np.asarray(indices_a, dtype=int),
        indices_b=np.asarray(indices_b, dtype=int),
        count_a=count_a,
        count_b=count_b,
        overlap_count=overlap_count,
    )


def _resolve_rdf_selection_caches_by_species(
    frames: list[Atoms],
    *,
    species_labels: list[str],
) -> dict[str, _RDFSelectionCache] | None:
    """Resolve stable per-species selection caches reused across many RDF pairs."""
    if not frames:
        return {}

    reference_numbers = np.asarray(frames[0].numbers, dtype=int)
    for frame_index, frame in enumerate(frames[1:], start=1):
        current_numbers = np.asarray(frame.numbers, dtype=int)
        if current_numbers.shape != reference_numbers.shape or not np.array_equal(
            current_numbers,
            reference_numbers,
        ):
            LOGGER.warning(
                "RDF atom identities/order changed at frame %d; "
                "falling back to per-pair framewise species selection.",
                frame_index,
            )
            return None

    caches: dict[str, _RDFSelectionCache] = {}
    for label in species_labels:
        mask = _select_mask(reference_numbers, label)
        indices = np.flatnonzero(mask)
        count = int(indices.size)
        if count == 0:
            raise ValueError(f"RDF selection produced no atoms in frame 0 (species={label}).")
        caches[label] = _RDFSelectionCache(
            mask_a=mask,
            mask_b=mask,
            indices_a=indices,
            indices_b=indices,
            count_a=count,
            count_b=count,
            overlap_count=count,
        )
    return caches


def _build_rdf_pair_selection_cache(
    cache_a: _RDFSelectionCache,
    cache_b: _RDFSelectionCache,
    *,
    same_selection: bool,
) -> _RDFSelectionCache:
    """Combine single-species stable caches into one pair-selection cache."""
    if same_selection:
        return cache_a
    return _RDFSelectionCache(
        mask_a=np.asarray(cache_a.mask_a, dtype=bool),
        mask_b=np.asarray(cache_b.mask_a, dtype=bool),
        indices_a=np.asarray(cache_a.indices_a, dtype=int),
        indices_b=np.asarray(cache_b.indices_a, dtype=int),
        count_a=int(cache_a.count_a),
        count_b=int(cache_b.count_a),
        overlap_count=int(
            np.intersect1d(cache_a.indices_a, cache_b.indices_a, assume_unique=True).size
        ),
    )


def _compute_rdf_frame_contribution(
    frame_index: int,
    frame: Atoms,
    *,
    label_a: str,
    label_b: str,
    same_selection: bool,
    r_max: float,
    bin_edges: np.ndarray,
    shell_volumes: np.ndarray,
    max_sphere_volume: float,
    selection_cache: _RDFSelectionCache | None = None,
    strategy_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram counts and expected shell counts for one frame."""
    volume = _frame_volume(frame)
    numbers = np.asarray(frame.numbers, dtype=int)
    n_atoms = numbers.size
    neighbor_density = min(1.0, max_sphere_volume / volume)

    if label_a == "ALL" and label_b == "ALL":
        if n_atoms == 0:
            raise ValueError(
                f"RDF selection produced no atoms in frame {frame_index} "
                f"(species_a={label_a}, species_b={label_b})."
            )
        if neighbor_density <= _NEIGHBORLIST_DENSITY_THRESHOLD:
            full_mask = np.ones(n_atoms, dtype=bool)
            sampled_distances = _sample_distances_neighbor_list(
                frame,
                full_mask,
                full_mask,
                r_max=r_max,
            )
        else:
            all_distances = np.asarray(frame.get_all_distances(mic=True), dtype=float)
            off_diagonal = ~np.eye(n_atoms, dtype=bool)
            sampled_distances = all_distances[off_diagonal]
            sampled_distances = sampled_distances[
                np.isfinite(sampled_distances)
                & (sampled_distances >= 0.0)
                & (sampled_distances <= r_max)
            ]
        counts = _histogram_rdf_distances(sampled_distances, bin_edges=bin_edges)
        valid_ordered_pair_count = n_atoms * n_atoms - n_atoms
        expected = (float(valid_ordered_pair_count) / volume) * shell_volumes
        return counts, expected

    if selection_cache is not None:
        mask_a = selection_cache.mask_a
        mask_b = selection_cache.mask_b
        indices_a = selection_cache.indices_a
        indices_b = selection_cache.indices_b
        count_a = selection_cache.count_a
        count_b = selection_cache.count_b
        overlap_count = selection_cache.overlap_count
    else:
        mask_a = _select_mask(numbers, label_a)
        mask_b = _select_mask(numbers, label_b)
        indices_a = np.flatnonzero(mask_a)
        indices_b = np.flatnonzero(mask_b)
        count_a = int(indices_a.size)
        count_b = int(indices_b.size)
        overlap_count = int(np.intersect1d(indices_a, indices_b, assume_unique=True).size)
    if count_a == 0 or count_b == 0:
        raise ValueError(
            f"RDF selection produced no atoms in frame {frame_index} "
            f"(species_a={label_a}, species_b={label_b})."
        )

    pair_fraction = (count_a * count_b) / float(n_atoms * n_atoms)
    resolved_strategy = strategy_override
    if resolved_strategy is None:
        use_neighbor_list = neighbor_density <= _NEIGHBORLIST_DENSITY_THRESHOLD
        use_selected_matrix = pair_fraction <= _SELECTED_MATRIX_FRACTION_THRESHOLD
        if use_neighbor_list:
            resolved_strategy = "neighbor_list"
        elif use_selected_matrix:
            resolved_strategy = "selected_matrix"
        else:
            resolved_strategy = "full_matrix"
    if resolved_strategy == "neighbor_list":
        sampled_distances = _sample_distances_neighbor_list(
            frame,
            mask_a,
            mask_b,
            r_max=r_max,
        )
    else:
        if resolved_strategy == "selected_matrix":
            sampled_distances = _sample_distances_selected_matrix(
                frame,
                indices_a,
                indices_b,
                same_selection=same_selection,
                r_max=r_max,
            )
        else:
            sampled_distances = _sample_distances_full_matrix(
                frame,
                indices_a,
                indices_b,
                same_selection=same_selection,
                r_max=r_max,
            )

    counts = _histogram_rdf_distances(sampled_distances, bin_edges=bin_edges)
    valid_ordered_pair_count = (count_a * count_b) - overlap_count
    expected = (float(valid_ordered_pair_count) / volume) * shell_volumes
    return counts, expected


def _compute_rdf_chunk_contributions(
    chunk: list[tuple[int, Atoms]],
    config: _RDFWorkerConfig,
) -> tuple[np.ndarray, np.ndarray, int, _RDFStatisticsMoments | None]:
    """Compute accumulated RDF contributions for one chunk of frames."""
    counts_accum = np.zeros(config.bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    moments = (
        _empty_rdf_statistics_moments(
            n_bins=config.bin_edges.size - 1,
            n_blocks=(
                None
                if config.block_index_by_frame is None
                else int(np.max(config.block_index_by_frame)) + 1
            ),
        )
        if config.collect_statistics
        else None
    )
    for frame_index, frame in chunk:
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
            strategy_override=config.strategy_override,
        )
        counts_accum += counts
        expected_accum += expected
        _update_rdf_statistics_moments(
            moments,
            counts=counts,
            expected=expected,
            frame_index=frame_index,
            block_index_by_frame=config.block_index_by_frame,
        )
    return counts_accum, expected_accum, len(chunk), moments


def _compute_rdf_generic_backend(
    frames: list[Atoms],
    *,
    config: _RDFWorkerConfig,
    worker_count: int,
    use_parallel: bool,
    chunk_size: int,
    progress: Any,
) -> tuple[np.ndarray, np.ndarray, _RDFStatisticsMoments | None]:
    """Accumulate RDF contributions with the exact framewise backend."""
    counts_accum = np.zeros(config.bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    moments = (
        _empty_rdf_statistics_moments(
            n_bins=config.bin_edges.size - 1,
            n_blocks=(
                None
                if config.block_index_by_frame is None
                else int(np.max(config.block_index_by_frame)) + 1
            ),
        )
        if config.collect_statistics
        else None
    )
    if not use_parallel:
        for frame_index, frame in enumerate(frames):
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
                strategy_override=config.strategy_override,
            )
            counts_accum += counts
            expected_accum += expected
            _update_rdf_statistics_moments(
                moments,
                counts=counts,
                expected=expected,
                frame_index=frame_index,
                block_index_by_frame=config.block_index_by_frame,
            )
            progress.update()
        return counts_accum, expected_accum, moments

    chunks = _iter_rdf_frame_chunks(frames, chunk_size)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for counts, expected, processed_frames, chunk_moments in executor.map(
            _compute_rdf_chunk_contributions,
            chunks,
            repeat(config),
        ):
            counts_accum += counts
            expected_accum += expected
            moments = _merge_rdf_statistics_moments(moments, chunk_moments)
            progress.update(processed_frames)
    return counts_accum, expected_accum, moments


def _compute_rdf_orthorhombic_backend(
    frames: list[Atoms],
    *,
    config: _RDFOrthorhombicConfig,
    backend: str,
    selection_cache: _RDFSelectionCache,
    cell_lengths: np.ndarray,
    volumes: np.ndarray,
    worker_count: int,
    use_parallel: bool,
    chunk_size: int,
    progress: Any,
) -> tuple[np.ndarray, np.ndarray, _RDFStatisticsMoments | None]:
    """Accumulate RDF contributions with an exact orthorhombic fast backend."""
    counts_accum = np.zeros(config.bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    moments = (
        _empty_rdf_statistics_moments(
            n_bins=config.bin_edges.size - 1,
            n_blocks=(
                None
                if config.block_index_by_frame is None
                else int(np.max(config.block_index_by_frame)) + 1
            ),
        )
        if config.collect_statistics
        else None
    )
    chunks = _iter_rdf_orthorhombic_chunks(
        frames,
        selection_cache=selection_cache,
        cell_lengths=cell_lengths,
        volumes=volumes,
        chunk_size=chunk_size,
    )
    worker = (
        _compute_rdf_dense_orthorhombic_chunk_contributions
        if backend == _RDF_BACKEND_DENSE
        else _compute_rdf_sparse_orthorhombic_chunk_contributions
    )

    if not use_parallel:
        for chunk in chunks:
            counts, expected, processed_frames, chunk_moments = worker(chunk, config)
            counts_accum += counts
            expected_accum += expected
            moments = _merge_rdf_statistics_moments(moments, chunk_moments)
            progress.update(processed_frames)
        return counts_accum, expected_accum, moments

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for counts, expected, processed_frames, chunk_moments in executor.map(
            worker, chunks, repeat(config)
        ):
            counts_accum += counts
            expected_accum += expected
            moments = _merge_rdf_statistics_moments(moments, chunk_moments)
            progress.update(processed_frames)
    return counts_accum, expected_accum, moments


def _compute_rdf_statistics_profile(
    frames: list[Atoms],
    *,
    label_a: str,
    label_b: str,
    r_max: float,
    bin_edges: np.ndarray,
    shell_volumes: np.ndarray,
    selection_cache: _RDFSelectionCache | None = None,
    strategy_override: str | None = None,
) -> SeriesStatistics:
    """Compute persisted RDF uncertainty statistics from exact per-frame RDF values."""
    max_sphere_volume = (4.0 / 3.0) * np.pi * (float(r_max) ** 3)
    same_selection = label_a == label_b
    sample_rows: list[np.ndarray] = []
    point_count = np.zeros(bin_edges.size - 1, dtype=int)
    for frame_index, frame in enumerate(frames):
        counts, expected = _compute_rdf_frame_contribution(
            frame_index,
            frame,
            label_a=label_a,
            label_b=label_b,
            same_selection=same_selection,
            r_max=float(r_max),
            bin_edges=np.asarray(bin_edges, dtype=float),
            shell_volumes=np.asarray(shell_volumes, dtype=float),
            max_sphere_volume=max_sphere_volume,
            selection_cache=selection_cache,
            strategy_override=strategy_override,
        )
        point_count += np.rint(counts).astype(int)
        g_values = np.full(counts.shape, np.nan, dtype=float)
        finite = expected > 0.0
        g_values[finite] = counts[finite] / expected[finite]
        sample_rows.append(g_values)
    sample_matrix = np.vstack(sample_rows)
    block_resolution = resolve_block_slices(len(frames))
    block_slices = None if block_resolution is None else block_resolution[0]
    return build_series_statistics(
        point_count=point_count,
        sample_values=sample_matrix,
        block_values=block_mean_matrix(sample_matrix, block_slices=block_slices),
    )


def compute_rdf(
    frames: list[Atoms],
    species_a: str | None = "all",
    species_b: str | None = None,
    atom_indices_a: Sequence[int] | np.ndarray | None = None,
    atom_indices_b: Sequence[int] | np.ndarray | None = None,
    r_max: float | None = None,
    bin_width: float = 0.05,
    threads: int | None = None,
    _strategy_override: str | None = None,
) -> RDFProfile:
    """Compute a radial distribution function averaged across frames."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("bin_width", bin_width)
    selector_a = _resolve_rdf_selector(species=species_a, atom_indices=atom_indices_a)
    selector_b = _resolve_rdf_selector(
        species=species_b if species_b is not None else species_a,
        atom_indices=atom_indices_b if atom_indices_b is not None else atom_indices_a,
    )
    label_a = str(selector_a.label)
    label_b = str(selector_b.label)
    strategy_override = _normalize_strategy_override(_strategy_override)

    requested_bin_width = float(bin_width)
    auto_r_max_raw = None
    if r_max is None:
        auto_r_max_raw = _auto_r_max_from_frames(frames)
        r_max = _resolve_auto_r_max_for_bin_width(
            auto_r_max=float(auto_r_max_raw),
            target_bin_width=requested_bin_width,
        )

    ensure_positive("r_max", r_max)

    bin_edges, _effective_bin_width = _build_uniform_rdf_bins(
        r_max=float(r_max),
        target_bin_width=requested_bin_width,
    )
    if auto_r_max_raw is not None and not np.isclose(
        float(r_max), float(auto_r_max_raw), rtol=1.0e-9, atol=1.0e-12
    ):
        LOGGER.debug(
            "Rounded auto RDF r_max down from %.6g to %.6g Angstrom to match bin_width=%.6g.",
            float(auto_r_max_raw),
            float(r_max),
            requested_bin_width,
        )

    same_selection = label_a == label_b
    shell_volumes = _shell_volumes_from_edges(bin_edges)

    LOGGER.debug(
        "Computing RDF (species_a=%s, species_b=%s, r_max=%.6g, bin_width=%.6g).",
        label_a,
        label_b,
        r_max,
        bin_width,
    )

    max_sphere_volume = (4.0 / 3.0) * np.pi * (r_max**3)
    worker_count = _resolve_rdf_worker_count(threads, len(frames))
    block_index_by_frame = _resolve_rdf_block_index_by_frame(len(frames))
    selection_cache = _resolve_rdf_selection_cache(
        frames,
        label_a=label_a,
        label_b=label_b,
        atom_indices_a=selector_a.atom_indices,
        atom_indices_b=selector_b.atom_indices,
    )
    backend_resolution = _resolve_rdf_backend(
        frames,
        label_a=label_a,
        label_b=label_b,
        r_max=float(r_max),
        worker_count=worker_count,
        selection_cache=selection_cache,
        strategy_override=strategy_override,
    )
    generic_config = _RDFWorkerConfig(
        label_a=label_a,
        label_b=label_b,
        same_selection=same_selection,
        r_max=r_max,
        bin_edges=bin_edges,
        shell_volumes=shell_volumes,
        max_sphere_volume=max_sphere_volume,
        selection_cache=backend_resolution.selection_cache,
        strategy_override=backend_resolution.generic_strategy_override,
        collect_statistics=True,
        block_index_by_frame=block_index_by_frame,
    )
    orth_backend_inputs: (
        tuple[_RDFOrthorhombicConfig, _RDFSelectionCache, np.ndarray, np.ndarray] | None
    ) = None
    if backend_resolution.backend == _RDF_BACKEND_GENERIC:
        if backend_resolution.use_parallel:
            LOGGER.debug(
                "Using RDF backend: generic framewise fallback "
                "(workers=%d, chunk_size=%d frame(s)).",
                backend_resolution.worker_count,
                backend_resolution.chunk_size,
            )
        else:
            LOGGER.debug("Using RDF backend: generic framewise fallback.")
    else:
        backend_label = (
            "dense orthorhombic chunked"
            if backend_resolution.backend == _RDF_BACKEND_DENSE
            else "sparse orthorhombic cutoff"
        )
        LOGGER.debug(
            "Using RDF backend: %s (workers=%d, chunk_size=%d frame(s), pair_count=%d).",
            backend_label,
            backend_resolution.worker_count,
            backend_resolution.chunk_size,
            backend_resolution.pair_count,
        )

        orth_selection = backend_resolution.selection_cache
        orth_cell_lengths = backend_resolution.cell_lengths
        orth_volumes = backend_resolution.volumes
        if orth_selection is None:
            raise ValueError("Orthorhombic RDF backend requires resolved stable selections.")
        if orth_cell_lengths is None or orth_volumes is None:
            raise ValueError("Orthorhombic RDF backend requires per-frame cell geometry.")
        self_pair_rows, self_pair_cols = _resolve_rdf_self_pair_coordinates(
            orth_selection.indices_a,
            orth_selection.indices_b,
        )
        orthorhombic_config = _RDFOrthorhombicConfig(
            r_max=float(r_max),
            bin_edges=np.asarray(bin_edges, dtype=float),
            shell_volumes=np.asarray(shell_volumes, dtype=float),
            same_selection=same_selection,
            count_a=int(orth_selection.count_a),
            count_b=int(orth_selection.count_b),
            overlap_count=int(orth_selection.overlap_count),
            indices_a=np.asarray(orth_selection.indices_a, dtype=int),
            indices_b=np.asarray(orth_selection.indices_b, dtype=int),
            self_pair_rows=self_pair_rows,
            self_pair_cols=self_pair_cols,
            collect_statistics=True,
            block_index_by_frame=block_index_by_frame,
        )
        orth_backend_inputs = (
            orthorhombic_config,
            orth_selection,
            orth_cell_lengths,
            orth_volumes,
        )

    with ProgressBar(
        desc=f"Computing RDF {label_a}-{label_b}", total=len(frames), unit="frame"
    ) as progress:
        if backend_resolution.backend == _RDF_BACKEND_GENERIC:
            counts_accum, expected_accum, moments = _compute_rdf_generic_backend(
                frames,
                config=generic_config,
                worker_count=backend_resolution.worker_count,
                use_parallel=backend_resolution.use_parallel,
                chunk_size=backend_resolution.chunk_size,
                progress=progress,
            )
        else:
            if orth_backend_inputs is None:
                raise ValueError("Orthorhombic RDF backend inputs were not initialized.")
            orthorhombic_config, orth_selection, orth_cell_lengths, orth_volumes = (
                orth_backend_inputs
            )
            counts_accum, expected_accum, moments = _compute_rdf_orthorhombic_backend(
                frames,
                config=orthorhombic_config,
                backend=backend_resolution.backend,
                selection_cache=orth_selection,
                cell_lengths=orth_cell_lengths,
                volumes=orth_volumes,
                worker_count=backend_resolution.worker_count,
                use_parallel=backend_resolution.use_parallel,
                chunk_size=backend_resolution.chunk_size,
                progress=progress,
            )

    g_r = np.full_like(counts_accum, np.nan, dtype=float)
    non_zero = expected_accum > 0.0
    g_r[non_zero] = counts_accum[non_zero] / expected_accum[non_zero]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if moments is None:
        raise ValueError("RDF statistics moments were not collected during compute.")
    with ProgressBar(
        desc=f"Finalizing RDF statistics {label_a}-{label_b}",
        total=1,
        unit="profile",
    ) as progress:
        statistics = _finalize_rdf_statistics_moments(moments)
        progress.update()
    LOGGER.debug(
        "Computed RDF profile with saved %s statistics.",
        "sample+block" if statistics.block_sem is not None else "sample",
    )

    return RDFProfile(
        species_a=label_a,
        species_b=label_b,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        g_r=g_r,
        n_frames=len(frames),
        series_statistics={"g_r": statistics},
        atom_indices_a=None
        if selector_a.atom_indices is None
        else np.asarray(selector_a.atom_indices, dtype=int),
        atom_indices_b=None
        if selector_b.atom_indices is None
        else np.asarray(selector_b.atom_indices, dtype=int),
        selection_kind_a=str(selector_a.selection_kind),
        selection_kind_b=str(selector_b.selection_kind),
    )


def compute_rdf_profiles(
    frames: list[Atoms],
    *,
    r_max: float | None = None,
    bin_width: float = 0.05,
    threads: int | None = None,
) -> list[RDFProfile]:
    """Compute RDFs for all unique unordered element-species pairs in a trajectory."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    species_labels = _available_element_species(frames)
    if not species_labels:
        raise ValueError("No elements found in trajectory.")
    pairs = list(combinations_with_replacement(species_labels, 2))
    requested_bin_width = float(bin_width)
    LOGGER.debug(
        "Computing pairwise RDF collection (%d profile(s), species=%s, r_max=%s, bin_width=%.6g).",
        len(pairs),
        ", ".join(species_labels),
        "auto" if r_max is None else f"{float(r_max):.6g}",
        requested_bin_width,
    )
    (
        unique_pairs,
        bin_edges,
        shell_volumes,
        counts_by_pair,
        expected_by_pair,
        selection_cache_by_pair,
        moments_by_pair,
    ) = _accumulate_rdf_pair_collection(
        frames,
        pairs=pairs,
        r_max=r_max,
        bin_width=requested_bin_width,
        threads=threads,
        progress_desc="Computing RDF pair collection",
        collect_statistics=True,
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    resolved_profiles: list[RDFProfile] = []
    with ProgressBar(
        desc="Finalizing RDF statistics",
        total=len(unique_pairs),
        unit="profile",
    ) as progress:
        for species_a, species_b in unique_pairs:
            counts_accum = counts_by_pair[(species_a, species_b)]
            expected_accum = expected_by_pair[(species_a, species_b)]
            g_r = np.full_like(counts_accum, np.nan, dtype=float)
            non_zero = expected_accum > 0.0
            g_r[non_zero] = counts_accum[non_zero] / expected_accum[non_zero]
            moments = moments_by_pair[(species_a, species_b)]
            if moments is None:
                raise ValueError(
                    f"RDF statistics moments were not collected for pair {species_a}-{species_b}."
                )
            resolved_profiles.append(
                RDFProfile(
                    species_a=species_a,
                    species_b=species_b,
                    bin_edges=np.asarray(bin_edges, dtype=float),
                    bin_centers=np.asarray(bin_centers, dtype=float),
                    g_r=g_r,
                    n_frames=len(frames),
                    series_statistics={"g_r": _finalize_rdf_statistics_moments(moments)},
                )
            )
            progress.update()
    any_block_statistics = any(
        profile.series_statistics is not None
        and profile.series_statistics["g_r"].block_sem is not None
        for profile in resolved_profiles
    )
    LOGGER.debug(
        "Computed %d RDF profiles with saved %s statistics.",
        len(resolved_profiles),
        "sample+block" if any_block_statistics else "sample",
    )
    return resolved_profiles


def _rdf_profile_hdf5_payload(profile: RDFProfile) -> dict[str, Any]:
    """Return LiNaK HDF5 payload pieces for one RDF profile."""
    bin_edges = np.asarray(profile.bin_edges, dtype=float)
    bin_centers = np.asarray(profile.bin_centers, dtype=float)
    if bin_edges.size != bin_centers.size + 1:
        raise ValueError(
            "RDF profile bin_edges and bin_centers sizes are inconsistent "
            f"(edges={bin_edges.size}, centers={bin_centers.size})."
        )
    expected_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if not np.allclose(expected_centers, bin_centers, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("RDF profile bin_centers are inconsistent with bin_edges.")
    bin_width = uniform_bin_width_from_edges(
        bin_edges,
        source_label=f"RDF profile '{profile.species_a}-{profile.species_b}'",
    )
    metadata_payload = {
        "species_a": profile.species_a,
        "species_b": profile.species_b,
        "selection_kind_a": str(profile.selection_kind_a),
        "selection_kind_b": str(profile.selection_kind_b),
        "n_frames": profile.n_frames,
        "bin_width_A": bin_width,
    }
    if profile.series_statistics:
        block_resolution = resolve_block_slices(int(profile.n_frames))
        block_lengths = None if block_resolution is None else block_resolution[1]
        metadata_payload["statistics"] = build_statistics_metadata(
            statistics_by_series=profile.series_statistics,
            block_lengths=block_lengths,
        )
    return {
        "datasets": {
            "bin_centers_A": np.asarray(profile.bin_centers, dtype=float),
            "g_r": np.asarray(profile.g_r, dtype=float),
            **(
                {}
                if profile.atom_indices_a is None
                else {"atom_indices_a": np.asarray(profile.atom_indices_a, dtype=int)}
            ),
            **(
                {}
                if profile.atom_indices_b is None
                else {"atom_indices_b": np.asarray(profile.atom_indices_b, dtype=int)}
            ),
            **statistics_payload_from_series_map(profile.series_statistics),
        },
        "metadata": build_profile_metadata(
            analysis="rdf",
            metadata=metadata_payload,
        ),
    }

def _rdf_pair_storage_key(species_a: str, species_b: str) -> tuple[str, str]:
    normalized_a = _normalize_species(species_a)
    normalized_b = _normalize_species(species_b)
    return tuple(sorted((normalized_a, normalized_b)))


def _rdf_payload_pair_key(payload: Mapping[str, Any]) -> tuple[str, str]:
    metadata = payload.get("metadata", {})
    species_a = str(metadata.get("species_a", "")).strip() or "UNKNOWN"
    species_b = str(metadata.get("species_b", "")).strip() or species_a
    return _rdf_pair_storage_key(species_a, species_b)


def _resolve_non_overwriting_rdf_output_path(path: str | Path) -> Path:
    resolved = resolve_hdf5_output_path(path)
    if not resolved.exists():
        return resolved
    stem = resolved.stem
    suffix = resolved.suffix
    parent = resolved.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _rdf_collection_compatibility_error(
    existing_payloads: Sequence[tuple[dict[str, np.ndarray], dict[str, Any]]],
    incoming_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_source_path: str | None,
) -> str | None:
    if not existing_payloads:
        return None
    if not incoming_payloads:
        return "no incoming RDF payloads were provided"

    incoming_source = str(expected_source_path or "").strip()
    incoming_reference = incoming_payloads[0]
    incoming_reference_datasets = incoming_reference.get("datasets", {})
    incoming_reference_metadata = incoming_reference.get("metadata", {})
    incoming_centers = np.asarray(incoming_reference_datasets.get("bin_centers_A", []), dtype=float)
    incoming_n_frames = int(incoming_reference_metadata.get("n_frames", 0) or 0)

    seen_pair_keys: set[tuple[str, str]] = set()
    for index, (datasets, metadata) in enumerate(existing_payloads):
        pair_key = _rdf_pair_storage_key(
            str(metadata.get("species_a", "")).strip() or "UNKNOWN",
            str(metadata.get("species_b", "")).strip() or str(metadata.get("species_a", "")).strip() or "UNKNOWN",
        )
        if pair_key in seen_pair_keys:
            return f"stored RDF collection already contains duplicate pair '{pair_key[0]}-{pair_key[1]}'"
        seen_pair_keys.add(pair_key)

        if incoming_source:
            stored_source = str(metadata.get("source_path", "")).strip()
            if stored_source and Path(stored_source).expanduser().resolve() != Path(
                incoming_source
            ).expanduser().resolve():
                return "stored source_path does not match the requested trajectory source"

        stored_centers = np.asarray(datasets.get("bin_centers_A", []), dtype=float)
        if stored_centers.shape != incoming_centers.shape or not np.allclose(
            stored_centers,
            incoming_centers,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            return f"stored RDF bin geometry is incompatible at profile index {index}"

        stored_n_frames = int(metadata.get("n_frames", 0) or 0)
        if stored_n_frames != incoming_n_frames:
            return f"stored n_frames={stored_n_frames} is incompatible with incoming n_frames={incoming_n_frames}"

    return None


def _rdf_payloads_match_within_tolerance(
    existing_payload: Mapping[str, Any],
    incoming_payload: Mapping[str, Any],
) -> bool:
    existing_datasets = existing_payload.get("datasets", {})
    incoming_datasets = incoming_payload.get("datasets", {})
    existing_centers = np.asarray(existing_datasets.get("bin_centers_A", []), dtype=float)
    incoming_centers = np.asarray(incoming_datasets.get("bin_centers_A", []), dtype=float)
    existing_g_r = np.asarray(existing_datasets.get("g_r", []), dtype=float)
    incoming_g_r = np.asarray(incoming_datasets.get("g_r", []), dtype=float)
    return (
        existing_centers.shape == incoming_centers.shape
        and existing_g_r.shape == incoming_g_r.shape
        and np.allclose(existing_centers, incoming_centers, rtol=1.0e-9, atol=1.0e-12)
        and np.allclose(
            existing_g_r,
            incoming_g_r,
            rtol=_RDF_SAME_PAIR_RTOL,
            atol=_RDF_SAME_PAIR_ATOL,
            equal_nan=True,
        )
    )


def _save_rdf_profile_collection(
    profiles: Sequence[RDFProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
    merge_existing: bool,
) -> Path:
    payloads = [_rdf_profile_hdf5_payload(profile) for profile in profiles]
    root_metadata = dict(additional_metadata or {})
    output_path = resolve_hdf5_output_path(output)
    if not merge_existing or not output_path.exists():
        written_path = write_profile_collection(
            output_path,
            analysis="rdf",
            profiles=payloads,
            metadata=root_metadata,
        )
        LOGGER.info("Saved %d RDF profile(s) to '%s'.", len(payloads), written_path)
        return written_path

    try:
        _source_path, existing_payloads = read_profile_payloads(
            output_path,
            analysis="rdf",
            label="RDF",
        )
    except Exception as exc:
        fallback_path = _resolve_non_overwriting_rdf_output_path(output_path)
        LOGGER.warning(
            "Existing RDF output '%s' could not be merged (%s). Writing fallback file '%s'.",
            output_path,
            exc,
            fallback_path,
        )
        written_path = write_profile_collection(
            fallback_path,
            analysis="rdf",
            profiles=payloads,
            metadata=root_metadata,
        )
        LOGGER.info("Saved %d RDF profile(s) to fallback '%s'.", len(payloads), written_path)
        return written_path

    compatibility_error = _rdf_collection_compatibility_error(
        existing_payloads,
        payloads,
        expected_source_path=str(root_metadata.get("source_path", "")).strip() or None,
    )
    if compatibility_error is not None:
        fallback_path = _resolve_non_overwriting_rdf_output_path(output_path)
        LOGGER.warning(
            "Existing RDF output '%s' is incompatible (%s). Writing fallback file '%s'.",
            output_path,
            compatibility_error,
            fallback_path,
        )
        written_path = write_profile_collection(
            fallback_path,
            analysis="rdf",
            profiles=payloads,
            metadata=root_metadata,
        )
        LOGGER.info("Saved %d RDF profile(s) to fallback '%s'.", len(payloads), written_path)
        return written_path

    merged_payloads = [
        {"datasets": dict(datasets), "metadata": dict(metadata)}
        for datasets, metadata in existing_payloads
    ]
    existing_index_by_pair = {
        _rdf_payload_pair_key(payload): index for index, payload in enumerate(merged_payloads)
    }
    appended_labels: list[str] = []
    identical_labels: list[str] = []
    replaced_labels: list[str] = []

    for payload in payloads:
        metadata = payload.get("metadata", {})
        species_a = str(metadata.get("species_a", "")).strip() or "UNKNOWN"
        species_b = str(metadata.get("species_b", "")).strip() or species_a
        label = f"{species_a}-{species_b}"
        pair_key = _rdf_pair_storage_key(species_a, species_b)
        existing_index = existing_index_by_pair.get(pair_key)
        if existing_index is None:
            existing_index_by_pair[pair_key] = len(merged_payloads)
            merged_payloads.append(payload)
            appended_labels.append(label)
            continue
        if _rdf_payloads_match_within_tolerance(merged_payloads[existing_index], payload):
            identical_labels.append(label)
        else:
            replaced_labels.append(label)
        merged_payloads[existing_index] = payload

    written_path = write_profile_collection(
        output_path,
        analysis="rdf",
        profiles=merged_payloads,
        metadata=root_metadata,
    )
    if appended_labels:
        LOGGER.info(
            "Appended RDF profile(s) %s to '%s'.",
            ", ".join(appended_labels),
            written_path,
        )
    if identical_labels:
        LOGGER.info(
            "RDF profile(s) already up to date in '%s': %s.",
            written_path,
            ", ".join(identical_labels),
        )
    if replaced_labels:
        LOGGER.warning(
            "Replaced existing RDF profile(s) in '%s' with newly computed data: %s.",
            written_path,
            ", ".join(replaced_labels),
        )
    if not appended_labels and not identical_labels and not replaced_labels:
        LOGGER.info("Saved RDF profile collection to '%s'.", written_path)
    return written_path


def save_rdf_profile(
    profile: RDFProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save RDF profile to LiNaK HDF5 and return written path."""
    payload = _rdf_profile_hdf5_payload(profile)
    metadata = dict(payload["metadata"])
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="rdf",
        datasets=payload["datasets"],
        metadata=metadata,
    )
    LOGGER.info("Saved RDF data to '%s'.", output_path)
    return output_path


def save_rdf_profiles(
    profiles: list[RDFProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
    force_collection: bool = False,
    merge_existing: bool = False,
) -> Path:
    """Save one or more RDF profiles to LiNaK HDF5 and return the written path."""
    if not profiles:
        raise ValueError("At least one RDF profile is required.")
    if len(profiles) == 1 and not force_collection and not merge_existing:
        return save_rdf_profile(
            profiles[0],
            output,
            additional_metadata=additional_metadata,
        )
    return _save_rdf_profile_collection(
        profiles,
        output,
        additional_metadata=additional_metadata,
        merge_existing=merge_existing,
    )


def load_rdf_profile(
    path: str | Path,
    *,
    species_a: str | None = None,
    species_b: str | None = None,
) -> RDFProfile:
    """Load one RDF profile from LiNaK HDF5.

    For profile-collection files, this returns the first profile.
    """
    profiles = load_rdf_profiles(path, species_a=species_a, species_b=species_b)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(f"RDF HDF5 '{source_path}' does not contain any RDF profiles.")
    return profiles[0]


def load_rdf_profiles(
    path: str | Path,
    *,
    species_a: str | None = None,
    species_b: str | None = None,
) -> list[RDFProfile]:
    """Load one or more RDF profiles from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads(
        path,
        analysis="rdf",
        label="RDF",
    )
    return _load_rdf_profiles_from_payloads(
        source_path,
        payloads,
        species_a=species_a,
        species_b=species_b,
    )


def _load_rdf_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species_a: str | None = None,
    species_b: str | None = None,
) -> list[RDFProfile]:
    exact_profiles: list[RDFProfile] = []
    reversed_profiles: list[RDFProfile] = []
    wanted_species_a = (
        None if species_a is None or not species_a.strip() else _normalize_species(species_a)
    )
    wanted_species_b = (
        None if species_b is None or not species_b.strip() else _normalize_species(species_b)
    )
    for datasets, metadata in payloads:
        required = ("bin_centers_A", "g_r")
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"RDF HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )

        resolved_species_a = str(metadata.get("species_a", "")).strip() or "UNKNOWN"
        resolved_species_b = str(metadata.get("species_b", "")).strip() or resolved_species_a
        selection_kind_a = str(metadata.get("selection_kind_a", "species") or "species")
        selection_kind_b = str(metadata.get("selection_kind_b", "species") or "species")
        if not _rdf_pair_matches_request(
            stored_species_a=resolved_species_a,
            stored_species_b=resolved_species_b,
            wanted_species_a=wanted_species_a,
            wanted_species_b=wanted_species_b,
        ):
            continue

        bin_centers = np.asarray(datasets["bin_centers_A"], dtype=float)
        bin_width = resolve_uniform_bin_width_for_load(
            metadata=metadata,
            bin_centers=bin_centers,
            source_path=source_path,
            analysis_name="RDF",
        )
        bin_edges = reconstruct_uniform_bin_edges_from_centers(
            bin_centers,
            bin_width=bin_width,
        )
        if bin_edges.size != bin_centers.size + 1:
            raise ValueError(f"RDF HDF5 '{source_path}' has incompatible bin geometry.")
        g_r = np.asarray(datasets["g_r"], dtype=float)
        n_frames = int(metadata.get("n_frames", 0))
        atom_indices_a = (
            None
            if "atom_indices_a" not in datasets
            else np.asarray(datasets["atom_indices_a"], dtype=int)
        )
        atom_indices_b = (
            None
            if "atom_indices_b" not in datasets
            else np.asarray(datasets["atom_indices_b"], dtype=int)
        )

        profile = RDFProfile(
            species_a=resolved_species_a,
            species_b=resolved_species_b,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            g_r=g_r,
            n_frames=n_frames,
            series_statistics=statistics_series_map_from_datasets(
                datasets,
                dataset_names=("g_r",),
            ),
            atom_indices_a=atom_indices_a,
            atom_indices_b=atom_indices_b,
            selection_kind_a=selection_kind_a,
            selection_kind_b=selection_kind_b,
        )
        if _rdf_pair_matches_exact_request(
            stored_species_a=resolved_species_a,
            stored_species_b=resolved_species_b,
            wanted_species_a=wanted_species_a,
            wanted_species_b=wanted_species_b,
        ):
            exact_profiles.append(profile)
        elif _pair_request_is_cross_species(wanted_species_a, wanted_species_b):
            reversed_profiles.append(
                RDFProfile(
                    species_a=str(wanted_species_a),
                    species_b=str(wanted_species_b),
                    bin_edges=bin_edges,
                    bin_centers=bin_centers,
                    g_r=g_r,
                    n_frames=n_frames,
                    series_statistics=statistics_series_map_from_datasets(
                        datasets,
                        dataset_names=("g_r",),
                    ),
                    atom_indices_a=atom_indices_a,
                    atom_indices_b=atom_indices_b,
                    selection_kind_a=selection_kind_a,
                    selection_kind_b=selection_kind_b,
                )
            )
        else:
            exact_profiles.append(profile)

    if exact_profiles:
        return exact_profiles
    if reversed_profiles:
        return reversed_profiles
    return []


def load_rdf_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species_a: str | None = None,
    species_b: str | None = None,
) -> list[RDFProfile]:
    """Load selected RDF profiles by profile index from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="rdf",
        label="RDF",
    )
    return _load_rdf_profiles_from_payloads(
        source_path,
        payloads,
        species_a=species_a,
        species_b=species_b,
    )


def plot_rdf_profile(
    profile: RDFProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
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
    title_pad: float | None = None,
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
    """Plot RDF profile using shared LiNaK plotting style."""
    resolve_rdf_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
    )
    schema_labels = default_plot_labels("rdf")
    default_x = "r (Angstrom)" if schema_labels is None else schema_labels[0]
    default_y = "g(r)" if schema_labels is None else schema_labels[1]
    resolved_label = line_label
    if resolved_label is None and legend:
        resolved_label = f"{profile.species_a}-{profile.species_b}"
    single_series = resolve_single_series_options(
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
    )
    return plot_line_series(
        profile.bin_centers,
        profile.g_r,
        title=title or f"{profile.species_a}-{profile.species_b} radial distribution function",
        x_label=resolve_explicit_plot_text(x_label, default_x),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_id=series_id,
        line_label=resolved_label,
        line_color=single_series.line_color,
        line_width_override=single_series.line_width_override,
        line_marker=single_series.line_marker,
        line_visible=single_series.line_visible,
        show_in_legend=True if not series_show_in_legend else bool(series_show_in_legend[0]),
        fit_config=None if not series_fit_configs else series_fit_configs[0],
        cumulative_config=cumulative_config,
        series_statistics=None
        if profile.series_statistics is None
        else profile.series_statistics.get("g_r"),
        error_config=error_config,
        normalization_mode=single_series.normalization_mode,
        normalization_value=single_series.normalization_value,
        normalization_x_ref=single_series.normalization_x_ref,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="rdf",
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
        title_pad=title_pad,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
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


def plot_rdf_profiles(
    profiles: list[RDFProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
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
    title_pad: float | None = None,
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
    """Plot one or more RDF profiles."""
    schema_labels = default_plot_labels("rdf")
    default_x = "r (Angstrom)" if schema_labels is None else schema_labels[0]
    default_y = "g(r)" if schema_labels is None else schema_labels[1]
    if not profiles:
        raise ValueError("At least one RDF profile is required.")
    first_profile = profiles[0]
    resolved_mapping = resolve_rdf_plot_mapping(
        contract=data_contract,
        profile=first_profile,
        mapping=view_mapping,
    )
    default_labels = [f"{profile.species_a}-{profile.species_b}" for profile in profiles]
    labels = resolve_series_labels(
        default_labels,
        series_labels,
        series_kind="RDF",
    )

    if not use_multi_series_plot(
        profile_count=len(profiles),
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
    ):
        return plot_rdf_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            data_contract=resolved_mapping.contract,
            view_mapping=resolved_mapping.mapping,
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
            title_pad=title_pad,
            x_axis_scale=x_axis_scale,
            x_axis_offset=x_axis_offset,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            line_label=labels[0] if labels else None,
            line_colors=line_colors,
            error_config=None if not series_error_configs else series_error_configs[0],
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            cumulative_config=None
            if not series_cumulative_configs
            else series_cumulative_configs[0],
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

    return plot_multi_line_series(
        [profile.bin_centers for profile in profiles],
        [profile.g_r for profile in profiles],
        labels,
        title=title or "Radial distribution function",
        x_label=resolve_explicit_plot_text(x_label, default_x),
        y_label=resolve_explicit_plot_text(y_label, default_y),
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
        series_statistics_data=[
            None if profile.series_statistics is None else profile.series_statistics.get("g_r")
            for profile in profiles
        ],
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="rdf",
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
        title_pad=title_pad,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
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
