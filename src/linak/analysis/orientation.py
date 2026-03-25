"""Water orientation analysis.

Computes the orientation of water molecules (polar and azimuthal angles)
as a function of distance to a surface.  The primary quantity is
``cos(polar)``, the cosine of the angle between the water bisector and
the surface-normal direction:

* ``+1`` → H-atoms pointing **away** from the surface
* ``−1`` → H-atoms pointing **toward** the surface
* ``0``  → bisector parallel to the surface

An azimuthal angle (molecular-plane normal projected onto the plane
perpendicular to the reference axis) is also computed and stored for
potential future use, but it is secondary and has no unique in-plane
reference direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..storage.hdf5_utils import (
    read_linak_hdf5_profiles,
    read_linak_hdf5_profiles_by_index,
    write_linak_hdf5,
)
from .schema import build_profile_metadata
from .water import (
    H2O_OH_CUTOFF_A,
    H2O_VALIDATION_STRIDE,
    WaterGeometry,
    water_molecule_triplets,
    water_triplet_geometry,
)
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
)
from ..progress import ProgressBar
from ..utils import axis_to_index

LOGGER = logging.getLogger(__name__)

# Default number of equally-spaced bins over the cos(angle) range [-1, +1].
_DEFAULT_ANGLE_BIN_COUNT: int = 50


# ───────────────────────────── dataclass ──────────────────────────────────

@dataclass(frozen=True)
class OrientationProfile:
    """Container for a water-orientation analysis result."""

    axis: str
    """Spatial axis used for distance binning (``"x"``, ``"y"``, ``"z"``)."""

    reference_axis: str
    """Axis treated as the surface normal for angle computation."""

    n_frames: int
    n_molecules_per_frame: int

    # 1-D distance bins (Å) ─────────────────────────────────────────────
    bin_edges: np.ndarray
    bin_centers: np.ndarray

    # Mean cos(angle) per distance bin ──────────────────────────────────
    cos_polar_mean: np.ndarray
    cos_azimuthal_mean: np.ndarray

    # Density-weighted: cos(angle) × number-density per distance bin ───
    cos_polar_density: np.ndarray
    cos_azimuthal_density: np.ndarray

    # H₂O number-density per distance bin (molecules / Å³ or Å⁻¹) ─────
    density: np.ndarray

    # 2-D heatmaps (n_dist_bins × n_angle_bins) ────────────────────────
    heatmap_polar: np.ndarray
    heatmap_azimuthal: np.ndarray
    heatmap_angle_bin_edges: np.ndarray
    heatmap_angle_bin_centers: np.ndarray

    # Surface / coordinate metadata ─────────────────────────────────────
    coordinate_mode: str  # "distance" or "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None
    cell_lengths_angstrom: tuple[float, float, float] | None = None


# ───────────────────────────── compute ────────────────────────────────────

def compute_orientation_profile(
    frames: list[Atoms],
    *,
    axis: str = "z",
    reference_axis: str = "z",
    bin_width: float = 0.1,
    angle_bin_count: int = _DEFAULT_ANGLE_BIN_COUNT,
    surface_mode: str = "auto",
    surface_elements: list[str] | None = None,
    include_fixed_surface_atoms: bool = False,
    binning: str = "cell",
    oh_cutoff: float = H2O_OH_CUTOFF_A,
) -> OrientationProfile:
    """Compute water-orientation profiles from a trajectory.

    Parameters
    ----------
    frames
        Trajectory frames (ASE ``Atoms`` list).
    axis
        Spatial axis for distance binning (default ``"z"``).
    reference_axis
        Axis treated as the surface normal (default ``"z"``).
    bin_width
        Spatial bin width in Å (default 0.1).
    angle_bin_count
        Number of bins over ``cos(angle) ∈ [-1, +1]`` for heatmaps.
    surface_mode / surface_elements / include_fixed_surface_atoms
        Forwarded to the surface estimator in *density.py*.
    binning
        ``"cell"`` (span full cell) or ``"observed"`` (data range only).
    oh_cutoff
        O–H cutoff for water-molecule detection.

    Returns
    -------
    OrientationProfile
    """
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    axis_index = axis_to_index(axis)
    ref_index = axis_to_index(reference_axis)
    ref_vec = np.zeros(3, dtype=float)
    ref_vec[ref_index] = 1.0

    # ── surface estimation (reuse density machinery) ──────────────────
    from .density import _select_surface_estimate

    surface_estimate, _method = _select_surface_estimate(
        frames,
        axis,
        mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
    )
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_per_frame: np.ndarray | None = None
    if surface_estimate is not None:
        surface_position = surface_estimate.position
        surface_position_std = surface_estimate.std
        surface_per_frame = surface_estimate.per_frame
        if not np.all(np.isfinite(surface_per_frame)):
            surface_per_frame = None

    coordinate_mode = "distance" if surface_per_frame is not None else "axis"

    # ── determine histogram bounds ────────────────────────────────────
    dist_min, dist_max = _determine_distance_bounds(
        frames, axis_index, binning, coordinate_mode, surface_per_frame,
    )
    dist_bin_edges = np.arange(dist_min, dist_max + bin_width, bin_width)
    n_dist_bins = len(dist_bin_edges) - 1
    if n_dist_bins < 1:
        raise ValueError(
            f"No distance bins produced (range [{dist_min:.3f}, {dist_max:.3f}], "
            f"bin_width={bin_width})."
        )
    dist_bin_centers = 0.5 * (dist_bin_edges[:-1] + dist_bin_edges[1:])

    # ── angle (cos) bin edges ─────────────────────────────────────────
    angle_bin_edges = np.linspace(-1.0, 1.0, angle_bin_count + 1)
    angle_bin_centers = 0.5 * (angle_bin_edges[:-1] + angle_bin_edges[1:])

    # ── accumulators ──────────────────────────────────────────────────
    cos_polar_sum = np.zeros(n_dist_bins, dtype=float)
    cos_azimuthal_sum = np.zeros(n_dist_bins, dtype=float)
    count = np.zeros(n_dist_bins, dtype=float)
    heatmap_polar = np.zeros((n_dist_bins, angle_bin_count), dtype=float)
    heatmap_azimuthal = np.zeros((n_dist_bins, angle_bin_count), dtype=float)

    # ── cell lengths for volume normalisation ─────────────────────────
    cell_lengths = _extract_cell_lengths(frames[0], axis_index)

    # ── frame loop ────────────────────────────────────────────────────
    cached_triplets: np.ndarray | None = None
    n_molecules_per_frame = 0

    with ProgressBar(
        desc="Computing orientation", total=len(frames), unit="frame",
    ) as progress:
        for frame_idx, frame in enumerate(frames):
            # water detection (cached, periodic re-validation)
            if cached_triplets is None:
                cached_triplets = water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
            elif frame_idx % H2O_VALIDATION_STRIDE == 0:
                validated = water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
                if not np.array_equal(validated, cached_triplets):
                    LOGGER.warning(
                        "H2O topology change at frame %d; refreshing water triplets.",
                        frame_idx,
                    )
                    cached_triplets = validated

            geom = water_triplet_geometry(frame, cached_triplets)
            n_mol = geom.com_positions.shape[0]
            if n_mol == 0:
                progress.update()
                continue
            if frame_idx == 0:
                n_molecules_per_frame = n_mol

            # distance coordinate
            dist_values = geom.com_positions[:, axis_index]
            if surface_per_frame is not None:
                dist_values = dist_values - float(surface_per_frame[frame_idx])

            # bisector and plane-normal vectors
            oh1 = geom.hydrogen1_positions - geom.oxygen_positions  # (n_mol, 3)
            oh2 = geom.hydrogen2_positions - geom.oxygen_positions
            oh1_norm = oh1 / np.linalg.norm(oh1, axis=1, keepdims=True)
            oh2_norm = oh2 / np.linalg.norm(oh2, axis=1, keepdims=True)

            bisector = oh1_norm + oh2_norm  # unnormalised bisector
            bisector_len = np.linalg.norm(bisector, axis=1, keepdims=True)
            # guard against degenerate geometry (180° H-O-H)
            bisector_len = np.maximum(bisector_len, 1.0e-12)
            bisector = bisector / bisector_len

            plane_normal = np.cross(oh1, oh2)
            pn_len = np.linalg.norm(plane_normal, axis=1, keepdims=True)
            pn_len = np.maximum(pn_len, 1.0e-12)
            plane_normal = plane_normal / pn_len

            # cos(polar) = bisector · ref_axis
            cos_polar = bisector @ ref_vec  # (n_mol,)

            # cos(azimuthal): project plane_normal onto plane ⊥ ref_axis,
            # then measure cos of angle from first in-plane Cartesian axis.
            # Pick the two Cartesian axes that span the perpendicular plane.
            in_plane_axes = [i for i in range(3) if i != ref_index]
            proj = plane_normal[:, in_plane_axes]  # (n_mol, 2)
            proj_len = np.linalg.norm(proj, axis=1)
            proj_len = np.maximum(proj_len, 1.0e-12)
            cos_azimuthal = proj[:, 0] / proj_len  # angle from first in-plane axis

            # bin assignment
            dist_idx = np.searchsorted(dist_bin_edges, dist_values, side="right") - 1
            valid = (dist_idx >= 0) & (dist_idx < n_dist_bins)
            dist_idx_v = dist_idx[valid]
            cos_polar_v = cos_polar[valid]
            cos_azimuthal_v = cos_azimuthal[valid]

            # 1-D accumulation
            np.add.at(cos_polar_sum, dist_idx_v, cos_polar_v)
            np.add.at(cos_azimuthal_sum, dist_idx_v, cos_azimuthal_v)
            np.add.at(count, dist_idx_v, 1.0)

            # 2-D heatmap accumulation
            angle_idx_polar = np.searchsorted(angle_bin_edges, cos_polar_v, side="right") - 1
            angle_idx_polar = np.clip(angle_idx_polar, 0, angle_bin_count - 1)
            angle_idx_azi = np.searchsorted(angle_bin_edges, cos_azimuthal_v, side="right") - 1
            angle_idx_azi = np.clip(angle_idx_azi, 0, angle_bin_count - 1)

            np.add.at(heatmap_polar, (dist_idx_v, angle_idx_polar), 1.0)
            np.add.at(heatmap_azimuthal, (dist_idx_v, angle_idx_azi), 1.0)

            progress.update()

    # ── finalise averages ─────────────────────────────────────────────
    n_frames = len(frames)
    safe_count = np.where(count > 0, count, 1.0)

    cos_polar_mean = cos_polar_sum / safe_count
    cos_azimuthal_mean = cos_azimuthal_sum / safe_count

    # number density: count / (n_frames * bin_volume_or_length)
    density = _compute_number_density(count, n_frames, bin_width, cell_lengths, axis_index)

    cos_polar_density = cos_polar_mean * density
    cos_azimuthal_density = cos_azimuthal_mean * density

    return OrientationProfile(
        axis=axis,
        reference_axis=reference_axis,
        n_frames=n_frames,
        n_molecules_per_frame=n_molecules_per_frame,
        bin_edges=dist_bin_edges,
        bin_centers=dist_bin_centers,
        cos_polar_mean=cos_polar_mean,
        cos_azimuthal_mean=cos_azimuthal_mean,
        cos_polar_density=cos_polar_density,
        cos_azimuthal_density=cos_azimuthal_density,
        density=density,
        heatmap_polar=heatmap_polar,
        heatmap_azimuthal=heatmap_azimuthal,
        heatmap_angle_bin_edges=angle_bin_edges,
        heatmap_angle_bin_centers=angle_bin_centers,
        coordinate_mode=coordinate_mode,
        surface_position=surface_position,
        surface_position_std=surface_position_std,
        cell_lengths_angstrom=cell_lengths,
    )


# ─────────────────────── internal helpers ─────────────────────────────────

def _determine_distance_bounds(
    frames: list[Atoms],
    axis_index: int,
    binning: str,
    coordinate_mode: str,
    surface_per_frame: np.ndarray | None,
) -> tuple[float, float]:
    """Return ``(min, max)`` of the distance coordinate across all frames."""
    normalized_binning = binning.strip().lower()
    if normalized_binning == "cell":
        cell = np.asarray(frames[0].cell.array, dtype=float)
        axis_len = float(np.linalg.norm(cell[axis_index]))
        if axis_len > 0:
            if coordinate_mode == "distance" and surface_per_frame is not None:
                offsets = surface_per_frame[np.isfinite(surface_per_frame)]
                if offsets.size > 0:
                    mean_offset = float(np.mean(offsets))
                    return -mean_offset, axis_len - mean_offset
            return 0.0, axis_len

    # fallback: scan data (only invoked under "observed" or if cell is unusable)
    global_min = float("inf")
    global_max = float("-inf")
    for fi, frame in enumerate(frames):
        pos = np.asarray(frame.positions[:, axis_index], dtype=float)
        if surface_per_frame is not None and np.isfinite(surface_per_frame[fi]):
            pos = pos - float(surface_per_frame[fi])
        if pos.size > 0:
            global_min = min(global_min, float(np.min(pos)))
            global_max = max(global_max, float(np.max(pos)))
    if not np.isfinite(global_min):
        global_min = 0.0
    if not np.isfinite(global_max):
        global_max = global_min + 1.0
    return global_min, global_max


def _extract_cell_lengths(
    frame: Atoms, axis_index: int,
) -> tuple[float, float, float] | None:
    """Extract cell lengths from a frame; return None if not periodic."""
    if not bool(np.all(frame.get_pbc())):
        return None
    cell = np.asarray(frame.cell.array, dtype=float)
    lengths = tuple(float(np.linalg.norm(cell[i])) for i in range(3))
    if any(length <= 0.0 for length in lengths):
        return None
    return (lengths[0], lengths[1], lengths[2])


def _compute_number_density(
    count: np.ndarray,
    n_frames: int,
    bin_width: float,
    cell_lengths: tuple[float, float, float] | None,
    axis_index: int,
) -> np.ndarray:
    """Convert raw molecule counts into number density (molecules / Å³ or Å⁻¹).

    If cell dimensions are known the volume of each bin slab is used;
    otherwise only bin-width normalisation is applied (linear density).
    """
    if cell_lengths is not None:
        cross_axes = [i for i in range(3) if i != axis_index]
        cross_area = cell_lengths[cross_axes[0]] * cell_lengths[cross_axes[1]]
        bin_volume = cross_area * bin_width
    else:
        bin_volume = bin_width  # linear fallback

    density = count / (n_frames * bin_volume)
    return density


# ─────────────────────── HDF5 save / load ─────────────────────────────────

_ANALYSIS_NAME = "orientation"


def _orientation_profile_hdf5_payload(
    profile: OrientationProfile,
) -> dict[str, Any]:
    metadata = build_profile_metadata(
        analysis=_ANALYSIS_NAME,
        metadata={
            "species": "H2O",
            "axis": profile.axis,
            "reference_axis": profile.reference_axis,
            "n_frames": int(profile.n_frames),
            "n_molecules_per_frame": int(profile.n_molecules_per_frame),
            "coordinate_mode": profile.coordinate_mode,
            "surface_position": profile.surface_position,
            "surface_position_std": profile.surface_position_std,
            "cell_lengths_angstrom": (
                None
                if profile.cell_lengths_angstrom is None
                else [float(v) for v in profile.cell_lengths_angstrom]
            ),
        },
    )
    datasets: dict[str, np.ndarray | None] = {
        "bin_edges_A": profile.bin_edges,
        "bin_centers_A": profile.bin_centers,
        "cos_polar_mean": profile.cos_polar_mean,
        "cos_azimuthal_mean": profile.cos_azimuthal_mean,
        "cos_polar_density": profile.cos_polar_density,
        "cos_azimuthal_density": profile.cos_azimuthal_density,
        "density": profile.density,
        "heatmap_polar": profile.heatmap_polar,
        "heatmap_azimuthal": profile.heatmap_azimuthal,
        "heatmap_angle_bin_edges": profile.heatmap_angle_bin_edges,
        "heatmap_angle_bin_centers": profile.heatmap_angle_bin_centers,
    }
    return {"datasets": datasets, "metadata": metadata}


def save_orientation_profile(
    profile: OrientationProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save an orientation profile to LiNaK HDF5."""
    payload = _orientation_profile_hdf5_payload(profile)
    metadata = dict(payload["metadata"])
    if additional_metadata:
        metadata.update(dict(additional_metadata))
    output_path = write_linak_hdf5(
        output,
        analysis=_ANALYSIS_NAME,
        datasets=payload["datasets"],
        metadata=metadata,
    )
    try:
        _display = os.path.relpath(output_path)
    except ValueError:
        _display = str(output_path)
    LOGGER.info("Saved orientation data to '%s'.", _display)
    return output_path


def _build_orientation_profile_from_hdf5(
    datasets: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> OrientationProfile:
    def _get(name: str) -> np.ndarray:
        arr = datasets.get(name)
        if arr is None:
            raise ValueError(f"Missing dataset '{name}' in orientation HDF5.")
        return np.asarray(arr, dtype=float)

    cell_raw = metadata.get("cell_lengths_angstrom")
    cell_lengths: tuple[float, float, float] | None = None
    if cell_raw is not None:
        try:
            cell_lengths = (float(cell_raw[0]), float(cell_raw[1]), float(cell_raw[2]))
        except (TypeError, IndexError, ValueError):
            cell_lengths = None

    return OrientationProfile(
        axis=str(metadata.get("axis", "z")),
        reference_axis=str(metadata.get("reference_axis", "z")),
        n_frames=int(metadata.get("n_frames", 0)),
        n_molecules_per_frame=int(metadata.get("n_molecules_per_frame", 0)),
        bin_edges=_get("bin_edges_A"),
        bin_centers=_get("bin_centers_A"),
        cos_polar_mean=_get("cos_polar_mean"),
        cos_azimuthal_mean=_get("cos_azimuthal_mean"),
        cos_polar_density=_get("cos_polar_density"),
        cos_azimuthal_density=_get("cos_azimuthal_density"),
        density=_get("density"),
        heatmap_polar=_get("heatmap_polar"),
        heatmap_azimuthal=_get("heatmap_azimuthal"),
        heatmap_angle_bin_edges=_get("heatmap_angle_bin_edges"),
        heatmap_angle_bin_centers=_get("heatmap_angle_bin_centers"),
        coordinate_mode=str(metadata.get("coordinate_mode", "axis")),
        surface_position=metadata.get("surface_position"),
        surface_position_std=metadata.get("surface_position_std"),
        cell_lengths_angstrom=cell_lengths,
    )


def load_orientation_profile(path: str | Path) -> OrientationProfile:
    """Load a single orientation profile from LiNaK HDF5."""
    profiles = load_orientation_profiles(path)
    if not profiles:
        raise ValueError(f"No orientation profiles found in '{path}'.")
    return profiles[0]


def load_orientation_profiles(path: str | Path) -> list[OrientationProfile]:
    """Load all orientation profiles from a LiNaK HDF5 file."""
    raw_profiles = read_linak_hdf5_profiles(path, expected_analysis=_ANALYSIS_NAME)
    return [
        _build_orientation_profile_from_hdf5(datasets, metadata)
        for datasets, metadata in raw_profiles
    ]


def load_orientation_profiles_by_index(
    path: str | Path,
    indices: list[int],
) -> list[OrientationProfile]:
    """Load selected orientation profiles by index."""
    raw = read_linak_hdf5_profiles_by_index(
        path, indices, expected_analysis=_ANALYSIS_NAME,
    )
    return [
        _build_orientation_profile_from_hdf5(datasets, metadata)
        for datasets, metadata in raw
    ]


# ──────────────────────── plotting helpers ────────────────────────────────

_ANGLE_CHOICES = ("polar", "azimuthal")
_COMPONENT_CHOICES = ("average", "density-weighted", "heatmap")


def _normalize_angle_token(angle: str | None) -> str:
    token = "polar" if angle is None else str(angle).strip().lower()
    if token not in _ANGLE_CHOICES:
        raise ValueError(f"angle must be one of: {', '.join(_ANGLE_CHOICES)}")
    return token


def _normalize_component_token(component: str | None) -> str:
    token = "average" if component is None else str(component).strip().lower()
    if token not in _COMPONENT_CHOICES:
        raise ValueError(f"component must be one of: {', '.join(_COMPONENT_CHOICES)}")
    return token


def _distance_label(profile: OrientationProfile) -> str:
    if profile.coordinate_mode == "distance":
        return f"Distance to surface along {profile.axis.upper()} (Å)"
    return f"{profile.axis.upper()} (Å)"


def _y_label_for_component(component: str, angle: str) -> str:
    angle_label = "θ" if angle == "polar" else "φ"
    if component == "average":
        return f"⟨cos({angle_label})⟩"
    if component == "density-weighted":
        return f"ρ · ⟨cos({angle_label})⟩ (Å⁻³)"
    return f"cos({angle_label})"


def _select_1d_data(
    profile: OrientationProfile,
    component: str,
    angle: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = profile.bin_centers
    if component == "average":
        y = profile.cos_polar_mean if angle == "polar" else profile.cos_azimuthal_mean
    elif component == "density-weighted":
        y = profile.cos_polar_density if angle == "polar" else profile.cos_azimuthal_density
    else:
        raise ValueError(f"Cannot produce 1D data for component '{component}'.")
    return x, y


def _select_heatmap_data(
    profile: OrientationProfile,
    angle: str,
) -> np.ndarray:
    if angle == "polar":
        return profile.heatmap_polar
    return profile.heatmap_azimuthal


# ─────────────────── public plot functions ────────────────────────────────

def plot_orientation_profile(
    profile: OrientationProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    component: str = "average",
    angle: str = "polar",
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
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    y_bin_width: float | None = None,
    y_bin_reducer: str | None = None,
    heatmap_normalize: bool = False,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
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
    """Plot a single orientation profile."""
    norm_component = _normalize_component_token(component)
    norm_angle = _normalize_angle_token(angle)

    if norm_component == "heatmap":
        return _plot_orientation_heatmap(
            [profile],
            angle=norm_angle,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_lim=x_lim,
            y_lim=y_lim,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            tick_params_kwargs=tick_params_kwargs,
            grid_kwargs=grid_kwargs,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_cmap=heatmap_cmap,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            y_bin_width=y_bin_width,
            y_bin_reducer=y_bin_reducer,
            heatmap_normalize=heatmap_normalize,
            heatmap_colorbar_enabled=heatmap_colorbar_enabled,
            heatmap_colorbar_label=heatmap_colorbar_label,
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    x, y = _select_1d_data(profile, norm_component, norm_angle)
    default_title = f"H₂O orientation ({norm_angle})"
    default_y = _y_label_for_component(norm_component, norm_angle)
    return plot_line_series(
        x,
        y,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, _distance_label(profile)),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_id=None,
        line_label=line_label or f"cos({norm_angle})",
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


def plot_orientation_profiles(
    profiles: list[OrientationProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    component: str = "average",
    angle: str = "polar",
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
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    y_bin_width: float | None = None,
    y_bin_reducer: str | None = None,
    heatmap_normalize: bool = False,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
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
    """Plot one or more orientation profiles overlaid."""
    if not profiles:
        raise ValueError("At least one orientation profile is required.")

    norm_component = _normalize_component_token(component)
    norm_angle = _normalize_angle_token(angle)

    if norm_component == "heatmap":
        return _plot_orientation_heatmap(
            profiles,
            angle=norm_angle,
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_lim=x_lim,
            y_lim=y_lim,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            tick_params_kwargs=tick_params_kwargs,
            grid_kwargs=grid_kwargs,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_cmap=heatmap_cmap,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            y_bin_width=y_bin_width,
            y_bin_reducer=y_bin_reducer,
            heatmap_normalize=heatmap_normalize,
            heatmap_colorbar_enabled=heatmap_colorbar_enabled,
            heatmap_colorbar_label=heatmap_colorbar_label,
            heatmap_colorbar_label_size=heatmap_colorbar_label_size,
            heatmap_colorbar_tick_size=heatmap_colorbar_tick_size,
            heatmap_colorbar_position=heatmap_colorbar_position,
            heatmap_colorbar_pad=heatmap_colorbar_pad,
            heatmap_colorbar_shrink=heatmap_colorbar_shrink,
            heatmap_colorbar_aspect=heatmap_colorbar_aspect,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            savefig_kwargs=savefig_kwargs,
        )

    # Build series for overlay
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    labels: list[str] = []
    for i, profile in enumerate(profiles):
        x, y = _select_1d_data(profile, norm_component, norm_angle)
        x_arrays.append(x)
        y_arrays.append(y)
        labels.append(f"cos({norm_angle}) [{i}]" if len(profiles) > 1 else f"cos({norm_angle})")

    if series_labels is not None:
        labels = list(series_labels) + labels[len(series_labels):]

    default_title = f"H₂O orientation ({norm_angle})"
    ref_profile = profiles[0]
    default_y = _y_label_for_component(norm_component, norm_angle)
    return plot_multi_line_series(
        x_arrays,
        y_arrays,
        labels,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, _distance_label(ref_profile)),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        series_ids=series_ids,
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
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )


# ──────────────────── heatmap rebinning ───────────────────────────────────

_REDUCERS: dict[str, Any] = {
    "sum": np.sum,
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
}


def _rebin_heatmap_axis(
    data: np.ndarray,
    edges: np.ndarray,
    bin_width: float,
    *,
    axis: int,
    reducer: str = "sum",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge bins along *axis* so each new bin spans approximately *bin_width*.

    Returns the rebinned data array and new edge array.
    """
    n_bins = data.shape[axis]
    old_widths = np.diff(edges)
    reduce_fn = _REDUCERS.get(reducer, np.sum)

    # Determine grouping: greedily merge consecutive bins until width >= bin_width
    groups: list[list[int]] = []
    current: list[int] = []
    current_width = 0.0
    for i in range(n_bins):
        current.append(i)
        current_width += old_widths[i]
        if current_width >= bin_width - 1e-12:
            groups.append(current)
            current = []
            current_width = 0.0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    new_edges = [edges[groups[0][0]]]
    slices: list[np.ndarray] = []
    for group in groups:
        new_edges.append(edges[group[-1] + 1])
        idx = np.array(group)
        chunk = np.take(data, idx, axis=axis)
        reduced = reduce_fn(chunk, axis=axis, keepdims=True)
        slices.append(reduced)

    rebinned = np.concatenate(slices, axis=axis)
    return rebinned, np.array(new_edges)


# ──────────────────── heatmap renderer ────────────────────────────────────

def _plot_orientation_heatmap(
    profiles: list[OrientationProfile],
    *,
    angle: str,
    output: str | Path | None,
    show: bool,
    show_blocking: bool,
    preferred_backend: str | None,
    style: PlotStyle,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    x_lim: tuple[float | None, float | None] | list[float | None] | None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None,
    title_visible: bool | None,
    ticks_visible: bool | None,
    x_tick_rotation: float | None,
    y_tick_rotation: float | None,
    tick_params_kwargs: dict[str, Any] | None,
    grid_kwargs: dict[str, Any] | None,
    heatmap_vmin: float | None,
    heatmap_vmax: float | None,
    heatmap_cmap: str | None,
    x_bin_width: float | None,
    x_bin_reducer: str | None,
    y_bin_width: float | None,
    y_bin_reducer: str | None,
    heatmap_normalize: bool,
    heatmap_colorbar_label: str | None,
    heatmap_colorbar_label_size: int | None,
    heatmap_colorbar_tick_size: int | None,
    heatmap_colorbar_enabled: bool,
    heatmap_colorbar_position: str,
    heatmap_colorbar_pad: float | None,
    heatmap_colorbar_shrink: float | None,
    heatmap_colorbar_aspect: float | None,
    x_ticks: list[float] | tuple[float, ...] | None,
    y_ticks: list[float] | tuple[float, ...] | None,
    capture_state: dict[str, Any] | None,
    suppress_output_log: bool,
    matplotlib_rc: dict[str, Any] | None,
    figure_kwargs: dict[str, Any] | None,
    savefig_kwargs: dict[str, Any] | None,
) -> Path | None:
    """Render a 2-D heatmap of orientation frequency vs distance."""
    import matplotlib
    import matplotlib.pyplot as plt

    if preferred_backend:
        matplotlib.use(preferred_backend)
    if matplotlib_rc:
        plt.rcParams.update(matplotlib_rc)

    # Sum heatmaps if multiple profiles
    ref = profiles[0]
    heatmap = _select_heatmap_data(ref, angle).copy()
    for p in profiles[1:]:
        extra = _select_heatmap_data(p, angle)
        if extra.shape == heatmap.shape:
            heatmap += extra

    x_edges = ref.bin_edges
    y_edges = ref.heatmap_angle_bin_edges

    # Rebin distance axis (x)
    if x_bin_width is not None and x_bin_width > 0:
        heatmap, x_edges = _rebin_heatmap_axis(
            heatmap, x_edges, x_bin_width, axis=0, reducer=x_bin_reducer or "sum",
        )
    # Rebin angle axis (y)
    if y_bin_width is not None and y_bin_width > 0:
        heatmap, y_edges = _rebin_heatmap_axis(
            heatmap, y_edges, y_bin_width, axis=1, reducer=y_bin_reducer or "sum",
        )

    # Optionally normalise each distance bin to a probability distribution
    if heatmap_normalize:
        row_sums = heatmap.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        heatmap_plot = heatmap / row_sums
    else:
        heatmap_plot = heatmap

    fig_kw = dict(figsize=style.figure_size, dpi=style.dpi)
    if figure_kwargs:
        fig_kw.update(figure_kwargs)
    fig, ax = plt.subplots(**fig_kw)

    angle_symbol = "θ" if angle == "polar" else "φ"
    default_title = f"H₂O orientation heatmap ({angle})"
    default_x = _distance_label(ref)
    default_y = f"cos({angle_symbol})"

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        heatmap_plot.T,
        shading="flat",
        cmap=heatmap_cmap or "viridis",
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )
    default_cb_label = "Probability" if heatmap_normalize else "Frequency"
    if heatmap_colorbar_enabled:
        cb_kw: dict[str, Any] = {}
        position = heatmap_colorbar_position if heatmap_colorbar_position in {"right", "left", "top", "bottom"} else "right"
        cb_kw["location"] = position
        if heatmap_colorbar_pad is not None:
            cb_kw["pad"] = heatmap_colorbar_pad
        if heatmap_colorbar_shrink is not None:
            cb_kw["shrink"] = heatmap_colorbar_shrink
        if heatmap_colorbar_aspect is not None:
            cb_kw["aspect"] = heatmap_colorbar_aspect
        cbar = fig.colorbar(mesh, ax=ax, label=heatmap_colorbar_label if heatmap_colorbar_label is not None else default_cb_label, **cb_kw)
        cb_is_vertical = position in {"right", "left"}
        if heatmap_colorbar_label_size is not None:
            cbar.set_label(cbar.ax.get_ylabel() if cb_is_vertical else cbar.ax.get_xlabel(), fontsize=heatmap_colorbar_label_size)
        if heatmap_colorbar_tick_size is not None:
            cbar.ax.tick_params(labelsize=heatmap_colorbar_tick_size)

    ax.set_xlabel(x_label or default_x, fontsize=style.label_font_size)
    ax.set_ylabel(y_label or default_y, fontsize=style.label_font_size)
    if title_visible is not False:
        ax.set_title(title or default_title, fontsize=style.title_font_size)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_ticks is not None:
        ax.set_xticks([float(v) for v in x_ticks])
    if y_ticks is not None:
        ax.set_yticks([float(v) for v in y_ticks])

    # Ticks
    ax.tick_params(axis="both", labelsize=style.tick_font_size)
    tick_axis_hint = "both"
    minor_ticks_mode = "off"
    cleaned_tick_kw: dict[str, Any] = {}
    if isinstance(tick_params_kwargs, dict):
        cleaned_tick_kw = {k: v for k, v in tick_params_kwargs.items() if not k.startswith("_")}
        raw_axis = tick_params_kwargs.get("_ticks_axis")
        if raw_axis in {"both", "x", "y"}:
            tick_axis_hint = raw_axis
        raw_minor = tick_params_kwargs.get("_minor_ticks_mode")
        if raw_minor in {"on", "off"}:
            minor_ticks_mode = raw_minor

    if ticks_visible is False:
        if tick_axis_hint in {"both", "x"}:
            ax.tick_params(
                axis="x", which="both",
                bottom=False, top=False, labelbottom=False,
            )
        if tick_axis_hint in {"both", "y"}:
            ax.tick_params(
                axis="y", which="both",
                left=False, right=False, labelleft=False,
            )
    else:
        if ticks_visible is True and tick_axis_hint in {"x", "y"}:
            if tick_axis_hint == "x":
                ax.tick_params(
                    axis="y", which="both",
                    left=False, right=False, labelleft=False,
                )
            else:
                ax.tick_params(
                    axis="x", which="both",
                    bottom=False, top=False, labelbottom=False,
                )
        if x_tick_rotation is not None:
            ax.tick_params(axis="x", rotation=float(x_tick_rotation))
        if y_tick_rotation is not None:
            ax.tick_params(axis="y", rotation=float(y_tick_rotation))
    if minor_ticks_mode == "on":
        ax.minorticks_on()
    elif minor_ticks_mode == "off":
        ax.minorticks_off()
    if cleaned_tick_kw:
        ax.tick_params(**cleaned_tick_kw)

    # Grid (default off for heatmaps)
    if style.grid:
        resolved_grid_kw: dict[str, Any] = {
            "linestyle": style.grid_linestyle,
            "linewidth": style.grid_linewidth,
            "alpha": style.grid_alpha,
        }
        if grid_kwargs is not None:
            resolved_grid_kw.update(dict(grid_kwargs))
        ax.grid(True, **resolved_grid_kw)
    else:
        ax.grid(False)

    fig.tight_layout()

    saved_path: Path | None = None
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sfig_kw: dict[str, Any] = {}
        if savefig_kwargs:
            sfig_kw.update(savefig_kwargs)
        fig.savefig(str(output_path), **sfig_kw)
        if not suppress_output_log:
            LOGGER.info("Saved orientation heatmap to '%s'.", output_path)
        saved_path = output_path

    if capture_state is not None:
        capture_state["figure"] = fig
        capture_state["axes"] = ax

    if show:
        plt.show(block=show_blocking)
    else:
        plt.close(fig)

    return saved_path
