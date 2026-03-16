"""RDF analysis routines."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list

from .hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles,
    write_linak_hdf5,
)
from .plotting import DEFAULT_PLOT_STYLE, PlotStyle, plot_line_series, plot_multi_line_series
from .progress import ProgressBar
from .utils import ensure_positive

LOGGER = logging.getLogger(__name__)
_NEIGHBORLIST_DENSITY_THRESHOLD = 0.25
_SELECTED_MATRIX_FRACTION_THRESHOLD = 0.40
_DEFAULT_RDF_THREAD_CAP = 4


@dataclass(frozen=True)
class RDFProfile:
    """Container for a radial distribution function profile."""

    species_a: str
    species_b: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    g_r: np.ndarray
    n_frames: int


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
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("RDF requires valid periodic cell vectors in all frames.")

    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0:
        raise ValueError("RDF requires non-zero cell volume in all frames.")
    return volume


def _sample_distances_neighbor_list(
    frame: Atoms,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    r_max: float,
) -> np.ndarray:
    """Collect selected pair distances using a cutoff neighbor list."""
    i_pairs, j_pairs, pair_distances = neighbor_list(
        "ijd", frame, float(np.nextafter(r_max, np.inf))
    )
    pair_mask = mask_a[i_pairs] & mask_b[j_pairs] & (i_pairs != j_pairs) & (pair_distances > 0.0)
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
    return sampled_distances[sampled_distances <= r_max]


def _sample_distances_selected_matrix(
    frame: Atoms,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    *,
    same_selection: bool,
    r_max: float,
) -> np.ndarray:
    """Collect selected pair distances using only the relevant submatrix."""
    _, pair_distances = get_distances(
        frame.positions[indices_a],
        frame.positions[indices_b],
        cell=frame.cell.array,
        pbc=frame.get_pbc(),
    )
    if same_selection:
        pair_distances = pair_distances[~np.eye(indices_a.size, dtype=bool)]
    else:
        pair_distances = pair_distances.ravel()
    return pair_distances[(pair_distances > 0.0) & (pair_distances <= r_max)]


def _sample_distances_full_matrix(
    frame: Atoms,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    *,
    same_selection: bool,
    r_max: float,
) -> np.ndarray:
    """Collect selected pair distances from full MIC matrix."""
    all_distances = np.asarray(frame.get_all_distances(mic=True), dtype=float)
    pair_distances = all_distances[np.ix_(indices_a, indices_b)]
    if same_selection:
        pair_distances = pair_distances[~np.eye(indices_a.size, dtype=bool)]
    else:
        pair_distances = pair_distances.ravel()
    return pair_distances[(pair_distances > 0.0) & (pair_distances <= r_max)]


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
            sampled_distances = all_distances[(all_distances > 0.0) & (all_distances <= r_max)]
        counts = np.histogram(sampled_distances, bins=bin_edges)[0].astype(float)
        rho_b = (n_atoms - 1) / volume
        expected = (n_atoms * rho_b) * shell_volumes
        return counts, expected

    mask_a = _select_mask(numbers, label_a)
    mask_b = _select_mask(numbers, label_b)
    count_a = int(mask_a.sum())
    count_b = int(mask_b.sum())
    if count_a == 0 or count_b == 0:
        raise ValueError(
            f"RDF selection produced no atoms in frame {frame_index} "
            f"(species_a={label_a}, species_b={label_b})."
        )

    pair_fraction = (count_a * count_b) / float(n_atoms * n_atoms)
    use_neighbor_list = neighbor_density <= _NEIGHBORLIST_DENSITY_THRESHOLD
    use_selected_matrix = pair_fraction <= _SELECTED_MATRIX_FRACTION_THRESHOLD
    if use_neighbor_list:
        sampled_distances = _sample_distances_neighbor_list(
            frame,
            mask_a,
            mask_b,
            r_max=r_max,
        )
    else:
        indices_a = np.flatnonzero(mask_a)
        indices_b = np.flatnonzero(mask_b)
        if use_selected_matrix:
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
    if same_selection:
        rho_b = (count_b - 1) / volume
    else:
        rho_b = count_b / volume
    expected = (count_a * rho_b) * shell_volumes
    return counts, expected


def compute_rdf(
    frames: list[Atoms],
    species_a: str | None = "all",
    species_b: str | None = None,
    r_max: float | None = None,
    bin_width: float = 0.05,
    threads: int | None = None,
) -> RDFProfile:
    """Compute a radial distribution function averaged across frames."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("bin_width", bin_width)
    label_a = _normalize_species(species_a)
    label_b = _normalize_species(species_b if species_b is not None else species_a)

    if r_max is None:
        min_cell_lengths = []
        for frame in frames:
            cell = np.asarray(frame.cell.array, dtype=float)
            if cell.shape != (3, 3):
                raise ValueError("RDF requires valid periodic cell vectors in all frames.")
            cell_lengths = np.linalg.norm(cell, axis=1)
            min_cell_lengths.append(float(np.min(cell_lengths)))
        r_max = 0.5 * min(min_cell_lengths)

    ensure_positive("r_max", r_max)

    bin_edges = np.arange(0.0, r_max, bin_width, dtype=float)
    if bin_edges.size == 0:
        bin_edges = np.array([0.0], dtype=float)
    if not np.isclose(bin_edges[-1], r_max):
        bin_edges = np.append(bin_edges, r_max)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, r_max], dtype=float)

    counts_accum = np.zeros(bin_edges.size - 1, dtype=float)
    expected_accum = np.zeros_like(counts_accum)
    same_selection = label_a == label_b
    shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

    LOGGER.info(
        "Computing RDF (species_a=%s, species_b=%s, r_max=%.6g, bin_width=%.6g).",
        label_a,
        label_b,
        r_max,
        bin_width,
    )

    max_sphere_volume = (4.0 / 3.0) * np.pi * (r_max**3)
    worker_count = _resolve_rdf_worker_count(threads, len(frames))
    if worker_count > 1:
        LOGGER.info("Using %d thread(s) for RDF frame processing.", worker_count)
    with ProgressBar(
        desc=f"Computing RDF {label_a}-{label_b}", total=len(frames), unit="frame"
    ) as progress:
        if worker_count == 1:
            for i, frame in enumerate(frames):
                counts, expected = _compute_rdf_frame_contribution(
                    i,
                    frame,
                    label_a=label_a,
                    label_b=label_b,
                    same_selection=same_selection,
                    r_max=r_max,
                    bin_edges=bin_edges,
                    shell_volumes=shell_volumes,
                    max_sphere_volume=max_sphere_volume,
                )
                counts_accum += counts
                expected_accum += expected
                progress.update()
        else:

            def _frame_contribution_from_item(
                item: tuple[int, Atoms],
            ) -> tuple[np.ndarray, np.ndarray]:
                frame_index, frame_item = item
                return _compute_rdf_frame_contribution(
                    frame_index,
                    frame_item,
                    label_a=label_a,
                    label_b=label_b,
                    same_selection=same_selection,
                    r_max=r_max,
                    bin_edges=bin_edges,
                    shell_volumes=shell_volumes,
                    max_sphere_volume=max_sphere_volume,
                )

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for counts, expected in executor.map(
                    _frame_contribution_from_item, enumerate(frames)
                ):
                    counts_accum += counts
                    expected_accum += expected
                    progress.update()

    g_r = np.zeros_like(counts_accum)
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


def _uniform_bin_width_from_edges(bin_edges: np.ndarray, *, source_label: str) -> float:
    """Return a uniform bin width from edge coordinates."""
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
    """Resolve RDF bin width from metadata or infer from equally spaced centers."""
    raw = metadata.get("bin_width_A")
    if raw is not None:
        try:
            width = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RDF HDF5 '{source_path}' has invalid metadata value bin_width_A={raw!r}."
            ) from exc
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(
                f"RDF HDF5 '{source_path}' has non-positive bin_width_A={raw!r}."
            )
        if bin_centers.size > 1:
            center_steps = np.diff(bin_centers)
            if not np.allclose(center_steps, width, rtol=1.0e-6, atol=1.0e-9):
                raise ValueError(
                    f"RDF HDF5 '{source_path}' has inconsistent bin_centers_A and bin_width_A."
                )
        return width

    if bin_centers.size <= 1:
        raise ValueError(
            f"RDF HDF5 '{source_path}' is missing bin_edges_A and bin_width_A; "
            "cannot reconstruct single-bin edges."
        )
    center_steps = np.diff(bin_centers)
    if not np.all(np.isfinite(center_steps)) or np.any(center_steps <= 0.0):
        raise ValueError(f"RDF HDF5 '{source_path}' has invalid bin_centers_A spacing.")
    inferred = float(center_steps[0])
    if not np.allclose(center_steps, inferred, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError(
            f"RDF HDF5 '{source_path}' is missing bin_edges_A/bin_width_A and has non-uniform "
            "bin_centers_A spacing."
        )
    return inferred


def _reconstruct_bin_edges_from_centers(bin_centers: np.ndarray, *, bin_width: float) -> np.ndarray:
    """Reconstruct RDF edge coordinates from centers and uniform bin width."""
    if bin_centers.ndim != 1 or bin_centers.size == 0:
        raise ValueError("Cannot reconstruct bin edges from empty or non-1D bin centers.")
    left_edge = float(bin_centers[0]) - 0.5 * bin_width
    return left_edge + np.arange(bin_centers.size + 1, dtype=float) * bin_width


def save_rdf_profile(profile: RDFProfile, output: str | Path) -> Path:
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
    bin_width = _uniform_bin_width_from_edges(
        bin_edges,
        source_label=f"RDF profile '{profile.species_a}-{profile.species_b}'",
    )

    output_path = write_linak_hdf5(
        output,
        analysis="rdf",
        datasets={
            "bin_centers_A": profile.bin_centers,
            "g_r": profile.g_r,
        },
        metadata={
            "species_a": profile.species_a,
            "species_b": profile.species_b,
            "n_frames": profile.n_frames,
            "bin_width_A": bin_width,
            "units": {
                "bin_width_A": "Angstrom",
                "bin_centers_A": "Angstrom",
                "g_r": "dimensionless",
            },
        },
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
        profiles: list[RDFProfile] = []
        for datasets, metadata in payloads:
            required = ("bin_centers_A", "g_r")
            missing = [name for name in required if name not in datasets]
            if missing:
                raise ValueError(
                    f"RDF HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
                )

            meta_species_a = str(metadata.get("species_a", "")).strip()
            meta_species_b = str(metadata.get("species_b", "")).strip()
            if species_a is not None and species_a.strip():
                resolved_species_a = _normalize_species(species_a)
            elif meta_species_a:
                resolved_species_a = meta_species_a
            else:
                resolved_species_a = "UNKNOWN"

            if species_b is not None and species_b.strip():
                resolved_species_b = _normalize_species(species_b)
            elif meta_species_b:
                resolved_species_b = meta_species_b
            else:
                resolved_species_b = resolved_species_a

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

    raise ValueError(f"Unsupported RDF profile format for '{source_path}'. Use .h5/.hdf5.")


def plot_rdf_profile(
    profile: RDFProfile,
    output: str | Path | None = None,
    show: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
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
    capture_state: dict[str, Any] | None = None,
) -> Path | None:
    """Plot RDF profile using shared LiNaK plotting style."""
    resolved_label = line_label
    if resolved_label is None and legend:
        resolved_label = f"{profile.species_a}-{profile.species_b}"
    resolved_line_color = None
    if line_colors:
        resolved_line_color = line_colors[0]
    return plot_line_series(
        profile.bin_centers,
        profile.g_r,
        title=title or f"{profile.species_a}-{profile.species_b} radial distribution function",
        x_label=x_label or "r (Angstrom)",
        y_label=y_label or "g(r)",
        output=output,
        show=show,
        preferred_backend=preferred_backend,
        line_label=resolved_label,
        line_color=resolved_line_color,
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


def plot_rdf_profiles(
    profiles: list[RDFProfile],
    output: str | Path | None = None,
    show: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
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
    capture_state: dict[str, Any] | None = None,
) -> Path | None:
    """Plot one or more RDF profiles."""
    if not profiles:
        raise ValueError("At least one RDF profile is required.")
    default_labels = [f"{profile.species_a}-{profile.species_b}" for profile in profiles]
    labels = default_labels
    if series_labels is not None:
        if len(series_labels) != len(default_labels):
            raise ValueError(
                "series_labels count must match the number of plotted RDF series "
                f"({len(default_labels)})."
            )
        labels = [label.strip() for label in series_labels]
        if any(not label for label in labels):
            raise ValueError("series_labels cannot contain empty values.")

    if len(profiles) == 1:
        return plot_rdf_profile(
            profiles[0],
            output=output,
            show=show,
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
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            line_label=labels[0] if labels else None,
            line_colors=line_colors,
            capture_state=capture_state,
        )

    return plot_multi_line_series(
        [profile.bin_centers for profile in profiles],
        [profile.g_r for profile in profiles],
        labels,
        title=title or "Radial distribution function",
        x_label=x_label or "r (Angstrom)",
        y_label=y_label or "g(r)",
        output=output,
        show=show,
        preferred_backend=preferred_backend,
        style=style,
        line_colors=line_colors,
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
