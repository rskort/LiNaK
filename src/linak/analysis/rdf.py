"""RDF analysis routines."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import repeat
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list

from ..storage.hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles_by_index,
    read_linak_hdf5_profiles,
    write_linak_hdf5,
)
from .binning import (
    reconstruct_uniform_bin_edges_from_centers,
    resolve_uniform_bin_width_for_load,
    uniform_bin_width_from_edges,
)
from .schema import build_profile_metadata, default_plot_labels
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

LOGGER = logging.getLogger(__name__)
_NEIGHBORLIST_DENSITY_THRESHOLD = 0.25
_SELECTED_MATRIX_FRACTION_THRESHOLD = 0.40
_DEFAULT_RDF_THREAD_CAP = 4
_RDF_MIN_PARALLEL_FRAMES_PER_WORKER = 4
_RDF_MIN_CHUNK_SIZE = 16
_RDF_MAX_CHUNK_SIZE = 128


@dataclass(frozen=True)
class RDFProfile:
    """Container for a radial distribution function profile."""

    species_a: str
    species_b: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    g_r: np.ndarray
    n_frames: int


@dataclass(frozen=True)
class _RDFSelectionCache:
    """Stable per-frame selection metadata reused when atom identities do not change."""

    mask_a: np.ndarray
    mask_b: np.ndarray
    indices_a: np.ndarray
    indices_b: np.ndarray
    count_a: int
    count_b: int


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


def _normalize_species(species: str | None) -> str:
    """Normalize species selection for atom-resolved analyses."""
    if species is None:
        return "ALL"

    species = species.strip()
    if not species or species.lower() == "all" or species == "*":
        return "ALL"

    return species[0].upper() + species[1:].lower()


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
    if token not in {"neighbor_list", "selected_matrix", "full_matrix"}:
        raise ValueError(
            "RDF strategy override must be one of: neighbor_list, selected_matrix, full_matrix."
        )
    return token


def _resolve_rdf_chunk_size(n_frames: int, worker_count: int) -> int:
    target_chunks = max(worker_count * 8, 1)
    chunk_size = max(_RDF_MIN_CHUNK_SIZE, (n_frames + target_chunks - 1) // target_chunks)
    return min(_RDF_MAX_CHUNK_SIZE, chunk_size)


def _should_parallelize_rdf(n_frames: int, worker_count: int) -> bool:
    if worker_count <= 1:
        return False
    return n_frames >= max(_RDF_MIN_CHUNK_SIZE, worker_count * _RDF_MIN_PARALLEL_FRAMES_PER_WORKER)


def _iter_rdf_frame_chunks(
    frames: list[Atoms],
    chunk_size: int,
) -> list[list[tuple[int, Atoms]]]:
    return [
        list(enumerate(frames[start : start + chunk_size], start))
        for start in range(0, len(frames), chunk_size)
    ]


def _resolve_rdf_selection_cache(
    frames: list[Atoms],
    *,
    label_a: str,
    label_b: str,
) -> _RDFSelectionCache | None:
    """Reuse species selections when atom identities remain fixed across all frames."""
    if not frames:
        return None
    if label_a == "ALL" and label_b == "ALL":
        return None

    reference_numbers = np.asarray(frames[0].numbers, dtype=int)
    for frame_index, frame in enumerate(frames[1:], start=1):
        current_numbers = np.asarray(frame.numbers, dtype=int)
        if current_numbers.shape != reference_numbers.shape or not np.array_equal(
            current_numbers, reference_numbers
        ):
            LOGGER.warning(
                "RDF atom identities/order changed at frame %d; "
                "falling back to per-frame species selection.",
                frame_index,
            )
            return None

    mask_a = _select_mask(reference_numbers, label_a)
    mask_b = _select_mask(reference_numbers, label_b)
    indices_a = np.flatnonzero(mask_a)
    indices_b = np.flatnonzero(mask_b)
    count_a = int(indices_a.size)
    count_b = int(indices_b.size)
    if count_a == 0 or count_b == 0:
        raise ValueError(
            f"RDF selection produced no atoms in frame 0 (species_a={label_a}, species_b={label_b})."
        )
    return _RDFSelectionCache(
        mask_a=mask_a,
        mask_b=mask_b,
        indices_a=indices_a,
        indices_b=indices_b,
        count_a=count_a,
        count_b=count_b,
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
        counts = np.histogram(sampled_distances, bins=bin_edges)[0].astype(float)
        # Both observed and expected counts use ordered pairs: i->j and j->i are distinct.
        rho_b = (n_atoms - 1) / volume
        expected = (n_atoms * rho_b) * shell_volumes
        return counts, expected

    if selection_cache is not None:
        mask_a = selection_cache.mask_a
        mask_b = selection_cache.mask_b
        indices_a = selection_cache.indices_a
        indices_b = selection_cache.indices_b
        count_a = selection_cache.count_a
        count_b = selection_cache.count_b
    else:
        mask_a = _select_mask(numbers, label_a)
        mask_b = _select_mask(numbers, label_b)
        indices_a = np.flatnonzero(mask_a)
        indices_b = np.flatnonzero(mask_b)
        count_a = int(indices_a.size)
        count_b = int(indices_b.size)
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

    counts = np.histogram(sampled_distances, bins=bin_edges)[0].astype(float)
    # Expected counts must match the same ordered-pair convention used above.
    if same_selection:
        rho_b = (count_b - 1) / volume
    else:
        rho_b = count_b / volume
    expected = (count_a * rho_b) * shell_volumes
    return counts, expected


def _compute_rdf_chunk_contributions(
    chunk: list[tuple[int, Atoms]],
    config: _RDFWorkerConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute accumulated RDF contributions for one chunk of frames."""
    counts_accum = np.zeros(config.bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
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
    return counts_accum, expected_accum, len(chunk)


def compute_rdf(
    frames: list[Atoms],
    species_a: str | None = "all",
    species_b: str | None = None,
    r_max: float | None = None,
    bin_width: float = 0.05,
    threads: int | None = None,
    _strategy_override: str | None = None,
) -> RDFProfile:
    """Compute a radial distribution function averaged across frames."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("bin_width", bin_width)
    label_a = _normalize_species(species_a)
    label_b = _normalize_species(species_b if species_b is not None else species_a)
    strategy_override = _normalize_strategy_override(_strategy_override)

    if r_max is None:
        r_max = _auto_r_max_from_frames(frames)

    ensure_positive("r_max", r_max)

    requested_bin_width = float(bin_width)
    bin_edges, effective_bin_width = _build_uniform_rdf_bins(
        r_max=float(r_max),
        target_bin_width=requested_bin_width,
    )
    if not np.isclose(effective_bin_width, requested_bin_width, rtol=1.0e-9, atol=1.0e-12):
        LOGGER.info(
            "Adjusted RDF bin width from %.6g to %.6g Angstrom so uniform bins end at r_max=%.6g.",
            requested_bin_width,
            effective_bin_width,
            r_max,
        )

    counts_accum = np.zeros(bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    same_selection = label_a == label_b
    shell_volumes = _shell_volumes_from_edges(bin_edges)

    LOGGER.info(
        "Computing RDF (species_a=%s, species_b=%s, r_max=%.6g, bin_width=%.6g).",
        label_a,
        label_b,
        r_max,
        bin_width,
    )

    max_sphere_volume = (4.0 / 3.0) * np.pi * (r_max**3)
    worker_count = _resolve_rdf_worker_count(threads, len(frames))
    selection_cache = _resolve_rdf_selection_cache(frames, label_a=label_a, label_b=label_b)
    config = _RDFWorkerConfig(
        label_a=label_a,
        label_b=label_b,
        same_selection=same_selection,
        r_max=r_max,
        bin_edges=bin_edges,
        shell_volumes=shell_volumes,
        max_sphere_volume=max_sphere_volume,
        selection_cache=selection_cache,
        strategy_override=strategy_override,
    )

    use_parallel = _should_parallelize_rdf(len(frames), worker_count)
    chunk_size = _resolve_rdf_chunk_size(len(frames), worker_count) if use_parallel else len(frames)
    if use_parallel:
        LOGGER.info(
            "Using %d worker process(es) for RDF frame processing (chunk_size=%d).",
            worker_count,
            chunk_size,
        )
    with ProgressBar(
        desc=f"Computing RDF {label_a}-{label_b}", total=len(frames), unit="frame"
    ) as progress:
        if not use_parallel:
            for i, frame in enumerate(frames):
                counts, expected = _compute_rdf_frame_contribution(
                    i,
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
                progress.update()
        else:
            chunks = _iter_rdf_frame_chunks(frames, chunk_size)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for counts, expected, processed_frames in executor.map(
                    _compute_rdf_chunk_contributions,
                    chunks,
                    repeat(config),
                ):
                    counts_accum += counts
                    expected_accum += expected
                    progress.update(processed_frames)

    g_r = np.full_like(counts_accum, np.nan, dtype=float)
    non_zero = expected_accum > 0.0
    g_r[non_zero] = counts_accum[non_zero] / expected_accum[non_zero]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return RDFProfile(
        species_a=label_a,
        species_b=label_b,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        g_r=g_r,
        n_frames=len(frames),
    )


def save_rdf_profile(
    profile: RDFProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save RDF profile to LiNaK HDF5 and return written path."""
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

    metadata = build_profile_metadata(
        analysis="rdf",
        metadata={
            "species_a": profile.species_a,
            "species_b": profile.species_b,
            "n_frames": profile.n_frames,
            "bin_width_A": bin_width,
        },
    )
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="rdf",
        datasets={
            "bin_centers_A": profile.bin_centers,
            "g_r": profile.g_r,
        },
        metadata=metadata,
    )
    LOGGER.info("Saved RDF data to '%s'.", output_path)
    return output_path


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
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"RDF profile not found: {source_path}")

    if is_hdf5_path(source_path):
        payloads = read_linak_hdf5_profiles(source_path, expected_analysis="rdf")
        return _load_rdf_profiles_from_payloads(
            source_path,
            payloads,
            species_a=species_a,
            species_b=species_b,
        )

    raise ValueError(f"Unsupported RDF profile format for '{source_path}'. Use .h5/.hdf5.")


def _load_rdf_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species_a: str | None = None,
    species_b: str | None = None,
) -> list[RDFProfile]:
    profiles: list[RDFProfile] = []
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

        bin_centers = np.asarray(datasets["bin_centers_A"], dtype=float)
        if "bin_edges_A" in datasets:
            bin_edges = np.asarray(datasets["bin_edges_A"], dtype=float)
        else:
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
            raise ValueError(
                f"RDF HDF5 '{source_path}' has incompatible bin_edges_A/bin_centers_A sizes."
            )
        g_r = np.asarray(datasets["g_r"], dtype=float)
        n_frames = int(metadata.get("n_frames", 0))

        profiles.append(
            RDFProfile(
                species_a=resolved_species_a,
                species_b=resolved_species_b,
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                g_r=g_r,
                n_frames=n_frames,
            )
        )
    return profiles


def load_rdf_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species_a: str | None = None,
    species_b: str | None = None,
) -> list[RDFProfile]:
    """Load selected RDF profiles by profile index from LiNaK HDF5."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"RDF profile not found: {source_path}")
    if not is_hdf5_path(source_path):
        raise ValueError(f"Unsupported RDF profile format for '{source_path}'. Use .h5/.hdf5.")
    payloads = read_linak_hdf5_profiles_by_index(
        source_path,
        profile_indices,
        expected_analysis="rdf",
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
    """Plot RDF profile using shared LiNaK plotting style."""
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
        fit_enabled=True if series_fit_enabled and bool(series_fit_enabled[0]) else False,
        fit_label=(
            None if not series_fit_labels or not series_fit_labels[0] else str(series_fit_labels[0])
        ),
        fit_show_in_legend=(
            True if not series_fit_show_in_legend else bool(series_fit_show_in_legend[0])
        ),
        normalization_mode=single_series.normalization_mode,
        normalization_value=single_series.normalization_value,
        normalization_x_ref=single_series.normalization_x_ref,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
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
    """Plot one or more RDF profiles."""
    schema_labels = default_plot_labels("rdf")
    default_x = "r (Angstrom)" if schema_labels is None else schema_labels[0]
    default_y = "g(r)" if schema_labels is None else schema_labels[1]
    if not profiles:
        raise ValueError("At least one RDF profile is required.")
    default_labels = [f"{profile.species_a}-{profile.species_b}" for profile in profiles]
    labels = resolve_series_labels(
        default_labels,
        series_labels,
        series_kind="RDF",
    )

    if len(profiles) == 1:
        return plot_rdf_profile(
            profiles[0],
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
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            line_label=labels[0] if labels else None,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
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
