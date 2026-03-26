"""Atom-resolved position analysis routines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..storage.hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles_by_index,
    read_linak_hdf5_profiles,
    write_linak_hdf5,
)
from .density import (
    _log_framewise_surface_alignment,
    _select_surface_estimate,
    _surface_estimate_datasets,
    _surface_estimate_from_payload,
    _surface_estimate_supports_distance_mode,
    _surface_metadata_payload,
    _surface_metadata_view,
    SurfaceEstimate,
    SurfaceEstimatorOptions,
    available_element_species,
)
from .schema import build_profile_metadata, default_plot_labels
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
    resolve_single_series_options,
)
from ..utils import axis_to_index, ensure_positive

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionProfile:
    """Container for atom-resolved positions."""

    species: str
    axis: str
    atom_indices: np.ndarray
    frame_index: np.ndarray
    step: np.ndarray
    time_fs: np.ndarray
    time_ps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    distance_to_surface: np.ndarray
    n_frames: int
    n_atoms: int
    coordinate_mode: str = "axis"
    surface_position: float | None = None
    surface_position_std: float | None = None
    surface_position_per_frame: np.ndarray | None = None
    surface_estimate: SurfaceEstimate | None = None
    cell_lengths_angstrom: tuple[float, float, float] | None = None


def _normalize_species(species: str | None) -> str:
    if species is None:
        return "ALL"
    token = species.strip()
    if not token or token.lower() == "all" or token == "*":
        return "ALL"
    return token[0].upper() + token[1:].lower()


def _validate_stable_atom_layout(frames: list[Atoms]) -> np.ndarray:
    if not frames:
        raise ValueError("At least one trajectory frame is required.")
    reference_symbols = np.asarray(frames[0].get_chemical_symbols(), dtype=object)
    for frame_index, frame in enumerate(frames[1:], start=1):
        symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
        if symbols.size != reference_symbols.size:
            raise ValueError(
                "Atom-resolved position tracking requires all frames to have the same atom count "
                f"(frame 0: {reference_symbols.size}, frame {frame_index}: {symbols.size})."
            )
        if not np.array_equal(symbols, reference_symbols):
            raise ValueError(
                "Atom-resolved position tracking requires a stable atom ordering/symbol layout "
                f"across frames (mismatch at frame {frame_index})."
            )
    return reference_symbols


def _resolve_step_values(frames: list[Atoms]) -> np.ndarray:
    values = np.zeros(len(frames), dtype=float)
    all_have_steps = True
    for index, frame in enumerate(frames):
        info = getattr(frame, "info", None)
        if not isinstance(info, dict) or "timestep" not in info:
            all_have_steps = False
            break
        raw = info.get("timestep")
        parsed: float | None = None
        if isinstance(raw, (int, float, np.integer, np.floating)):
            parsed = float(raw)
        elif isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                try:
                    parsed = float(stripped)
                except ValueError:
                    parsed = None
        if parsed is None or not np.isfinite(parsed):
            all_have_steps = False
            break
        values[index] = parsed
    if all_have_steps:
        return values
    return np.arange(len(frames), dtype=float)


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


def _frame_has_usable_cell(frame: Atoms) -> bool:
    if not bool(np.all(frame.get_pbc())):
        return False
    lengths = np.asarray(frame.cell.lengths(), dtype=float)
    if lengths.shape != (3,):
        return False
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        return False
    return True


def _resolve_cell_lengths_from_frames(
    frames: list[Atoms],
) -> tuple[float, float, float] | None:
    if not frames:
        return None
    if not all(_frame_has_usable_cell(frame) for frame in frames):
        return None
    lengths = np.asarray(frames[0].cell.lengths(), dtype=float)
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]))


def _resolve_surface_distance_values(
    *,
    frames: list[Atoms],
    axis: str,
    axis_values_all: np.ndarray,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
) -> tuple[np.ndarray, str, float | None, float | None, np.ndarray | None, SurfaceEstimate | None]:
    surface_estimate, _surface_method = _select_surface_estimate(
        frames,
        axis,
        mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
    )
    if surface_estimate is None:
        LOGGER.warning(
            "Could not estimate a surface position along %s; storing raw %s coordinates "
            "for distance-to-surface values.",
            axis.lower(),
            axis.lower(),
        )
        return np.array(axis_values_all, copy=True), "axis", None, None, None, None

    surface_per_frame = np.asarray(surface_estimate.per_frame, dtype=float)
    if _surface_estimate_supports_distance_mode(
        surface_estimate, frame_count=axis_values_all.shape[0]
    ):
        _log_framewise_surface_alignment(
            logger=LOGGER,
            axis=axis,
            surface_position=surface_estimate.position,
            surface_position_std=surface_estimate.std,
        )
        return (
            axis_values_all - surface_per_frame[:, np.newaxis],
            "distance",
            float(surface_estimate.position),
            float(surface_estimate.std),
            surface_per_frame,
            surface_estimate,
        )

    LOGGER.warning(
        "Surface position was estimated for %s, but frame-wise alignment was unavailable; "
        "storing raw %s coordinates for distance-to-surface values.",
        axis.lower(),
        axis.lower(),
    )
    return np.array(axis_values_all, copy=True), "axis", None, None, None, surface_estimate


def _compute_position_profiles_for_labels(
    *,
    frames: list[Atoms],
    species_labels: list[str],
    axis: str,
    timestep_fs: float,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    surface_options: SurfaceEstimatorOptions | None,
) -> list[PositionProfile]:
    ensure_positive("timestep_fs", timestep_fs)
    symbols = _validate_stable_atom_layout(frames)
    cell_lengths_angstrom = _resolve_cell_lengths_from_frames(frames)
    positions = np.stack([np.asarray(frame.positions, dtype=float) for frame in frames], axis=0)
    axis_index = axis_to_index(axis)
    axis_values_all = positions[:, :, axis_index]
    (
        distance_to_surface_all,
        coordinate_mode,
        surface_position,
        surface_position_std,
        surface_position_per_frame,
        surface_estimate,
    ) = _resolve_surface_distance_values(
        frames=frames,
        axis=axis,
        axis_values_all=axis_values_all,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
    )

    frame_index = np.arange(len(frames), dtype=int)
    step = _resolve_step_values(frames)
    time_fs = frame_index.astype(float) * float(timestep_fs)
    time_ps = time_fs / 1000.0

    profiles: list[PositionProfile] = []
    for species_label in species_labels:
        if species_label == "ALL":
            atom_indices = np.arange(symbols.size, dtype=int)
        else:
            atom_indices = np.where(symbols == species_label)[0].astype(int, copy=False)
        if atom_indices.size == 0:
            raise ValueError(f"No atoms found for species '{species_label}' in frame 0.")

        selected = positions[:, atom_indices, :]
        profiles.append(
            PositionProfile(
                species=species_label,
                axis=axis.lower(),
                atom_indices=np.asarray(atom_indices, dtype=int),
                frame_index=np.asarray(frame_index, dtype=int),
                step=np.asarray(step, dtype=float),
                time_fs=np.asarray(time_fs, dtype=float),
                time_ps=np.asarray(time_ps, dtype=float),
                x=np.asarray(selected[:, :, 0], dtype=float),
                y=np.asarray(selected[:, :, 1], dtype=float),
                z=np.asarray(selected[:, :, 2], dtype=float),
                distance_to_surface=np.asarray(
                    distance_to_surface_all[:, atom_indices],
                    dtype=float,
                ),
                n_frames=len(frames),
                n_atoms=int(atom_indices.size),
                coordinate_mode=coordinate_mode,
                surface_position=surface_position,
                surface_position_std=surface_position_std,
                surface_position_per_frame=(
                    None
                    if surface_position_per_frame is None
                    else np.asarray(surface_position_per_frame, dtype=float)
                ),
                surface_estimate=surface_estimate,
                cell_lengths_angstrom=cell_lengths_angstrom,
            )
        )
    return profiles


def compute_position_profile(
    frames: list[Atoms],
    species: str | None = "all",
    *,
    axis: str = "z",
    timestep_fs: float = 1.0,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
) -> PositionProfile:
    """Compute one atom-resolved position profile."""
    species_label = _normalize_species(species)
    profiles = _compute_position_profiles_for_labels(
        frames=frames,
        species_labels=[species_label],
        axis=axis,
        timestep_fs=timestep_fs,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
    )
    return profiles[0]


def compute_position_profiles(
    frames: list[Atoms],
    species: str | None = "all",
    *,
    axis: str = "z",
    timestep_fs: float = 1.0,
    surface_mode: str = "auto",
    surface_elements: list[str] | tuple[str, ...] | None = None,
    include_fixed_surface_atoms: bool = False,
    surface_options: SurfaceEstimatorOptions | None = None,
) -> list[PositionProfile]:
    """Compute one or more atom-resolved position profiles."""
    species_label = _normalize_species(species)
    if species_label != "ALL":
        return [
            compute_position_profile(
                frames=frames,
                species=species_label,
                axis=axis,
                timestep_fs=timestep_fs,
                surface_mode=surface_mode,
                surface_elements=surface_elements,
                include_fixed_surface_atoms=include_fixed_surface_atoms,
                surface_options=surface_options,
            )
        ]

    element_species = available_element_species(frames)
    if not element_species:
        raise ValueError("No elements found in trajectory.")
    return _compute_position_profiles_for_labels(
        frames=frames,
        species_labels=element_species,
        axis=axis,
        timestep_fs=timestep_fs,
        surface_mode=surface_mode,
        surface_elements=surface_elements,
        include_fixed_surface_atoms=include_fixed_surface_atoms,
        surface_options=surface_options,
    )


def save_position_profile(
    profile: PositionProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one position profile to LiNaK HDF5 and return the written path."""
    metadata = build_profile_metadata(
        analysis="position",
        metadata={
            "species": profile.species,
            "axis": profile.axis,
            "n_frames": int(profile.n_frames),
            "n_atoms": int(profile.n_atoms),
            "coordinate_mode": profile.coordinate_mode,
            "cell_lengths_angstrom": (
                None
                if profile.cell_lengths_angstrom is None
                else [float(value) for value in profile.cell_lengths_angstrom]
            ),
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
        analysis="position",
        datasets={
            "frame_index": profile.frame_index,
            "step": profile.step,
            "time_fs": profile.time_fs,
            "time_ps": profile.time_ps,
            "atom_indices": profile.atom_indices,
            "x_A": profile.x,
            "y_A": profile.y,
            "z_A": profile.z,
            "distance_to_surface_A": profile.distance_to_surface,
            "surface_position_per_frame_A": profile.surface_position_per_frame,
            **_surface_estimate_datasets(profile.surface_estimate),
        },
        metadata=metadata,
    )
    LOGGER.info("Saved position data to '%s'.", output_path)
    return output_path


def load_position_profile(
    path: str | Path,
    *,
    species: str | None = None,
    axis: str | None = None,
) -> PositionProfile:
    """Load one position profile from LiNaK HDF5."""
    profiles = load_position_profiles(path, species=species, axis=axis)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(
            f"Position HDF5 '{source_path}' does not contain matching position profiles."
        )
    return profiles[0]


def load_position_profiles(
    path: str | Path,
    *,
    species: str | None = None,
    axis: str | None = None,
) -> list[PositionProfile]:
    """Load one or more position profiles from LiNaK HDF5."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Position profile not found: {source_path}")

    if not is_hdf5_path(source_path):
        raise ValueError(f"Unsupported position profile format for '{source_path}'. Use .h5/.hdf5.")

    payloads = read_linak_hdf5_profiles(source_path, expected_analysis="position")
    return _load_position_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
        axis=axis,
    )


def _load_position_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species: str | None = None,
    axis: str | None = None,
) -> list[PositionProfile]:
    wanted_species = None if species is None or not species.strip() else _normalize_species(species)
    wanted_axis = None if axis is None or not axis.strip() else axis.strip().lower()
    profiles: list[PositionProfile] = []
    for datasets, metadata in payloads:
        required = (
            "frame_index",
            "step",
            "time_fs",
            "time_ps",
            "atom_indices",
            "x_A",
            "y_A",
            "z_A",
            "distance_to_surface_A",
        )
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Position HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )

        resolved_species = str(metadata.get("species", "")).strip() or "UNKNOWN"
        resolved_axis = str(metadata.get("axis", "z")).strip().lower()
        if resolved_axis not in {"x", "y", "z"}:
            resolved_axis = "z"

        if wanted_species is not None and wanted_species != "ALL":
            if _normalize_species(resolved_species) != wanted_species:
                continue
        if wanted_axis is not None and resolved_axis != wanted_axis:
            continue

        frame_index = np.asarray(datasets["frame_index"], dtype=int)
        step = np.asarray(datasets["step"], dtype=float)
        time_fs = np.asarray(datasets["time_fs"], dtype=float)
        time_ps = np.asarray(datasets["time_ps"], dtype=float)
        atom_indices = np.asarray(datasets["atom_indices"], dtype=int)
        x_values = np.asarray(datasets["x_A"], dtype=float)
        y_values = np.asarray(datasets["y_A"], dtype=float)
        z_values = np.asarray(datasets["z_A"], dtype=float)
        distance_values = np.asarray(datasets["distance_to_surface_A"], dtype=float)

        if x_values.ndim != 2:
            raise ValueError(f"Position HDF5 '{source_path}' dataset 'x_A' must be 2D.")
        expected_shape = x_values.shape
        for dataset_name, values in (
            ("y_A", y_values),
            ("z_A", z_values),
            ("distance_to_surface_A", distance_values),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"Position HDF5 '{source_path}' dataset '{dataset_name}' shape mismatch: "
                    f"expected {expected_shape}, got {values.shape}."
                )
        if atom_indices.size != expected_shape[1]:
            raise ValueError(
                f"Position HDF5 '{source_path}' has inconsistent atom index count "
                f"({atom_indices.size}) for matrix width {expected_shape[1]}."
            )
        if frame_index.size != expected_shape[0]:
            raise ValueError(
                f"Position HDF5 '{source_path}' has inconsistent frame index count "
                f"({frame_index.size}) for matrix height {expected_shape[0]}."
            )

        coordinate_mode = str(metadata.get("coordinate_mode", "axis")).strip().lower()
        if coordinate_mode not in {"axis", "distance"}:
            coordinate_mode = "axis"

        surface_per_frame = None
        if "surface_position_per_frame_A" in datasets:
            candidate = np.asarray(datasets["surface_position_per_frame_A"], dtype=float)
            if candidate.shape == (expected_shape[0],):
                surface_per_frame = candidate
        surface_estimate = _surface_estimate_from_payload(
            datasets=datasets,
            metadata=metadata,
        )

        n_frames = int(metadata.get("n_frames", expected_shape[0]))
        n_atoms = int(metadata.get("n_atoms", expected_shape[1]))
        cell_lengths_angstrom = (
            _optional_cell_lengths(metadata.get("cell_lengths_angstrom"))
            or _optional_cell_lengths(metadata.get("pbc_cell_angstrom"))
            or _optional_cell_lengths(metadata.get("resolved_cell_angstrom"))
        )
        surface_metadata = _surface_metadata_view(metadata)
        profiles.append(
            PositionProfile(
                species=resolved_species,
                axis=resolved_axis,
                atom_indices=atom_indices,
                frame_index=frame_index,
                step=step,
                time_fs=time_fs,
                time_ps=time_ps,
                x=x_values,
                y=y_values,
                z=z_values,
                distance_to_surface=distance_values,
                n_frames=n_frames,
                n_atoms=n_atoms,
                coordinate_mode=coordinate_mode,
                surface_position=_optional_finite_float(
                    surface_metadata.get("position", metadata.get("surface_position"))
                ),
                surface_position_std=_optional_finite_float(
                    surface_metadata.get("position_std", metadata.get("surface_position_std"))
                ),
                surface_position_per_frame=surface_per_frame,
                surface_estimate=surface_estimate,
                cell_lengths_angstrom=cell_lengths_angstrom,
            )
        )
    return profiles


def load_position_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species: str | None = None,
    axis: str | None = None,
) -> list[PositionProfile]:
    """Load selected position profiles by profile index from LiNaK HDF5."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Position profile not found: {source_path}")
    if not is_hdf5_path(source_path):
        raise ValueError(f"Unsupported position profile format for '{source_path}'. Use .h5/.hdf5.")
    payloads = read_linak_hdf5_profiles_by_index(
        source_path,
        profile_indices,
        expected_analysis="position",
    )
    return _load_position_profiles_from_payloads(
        source_path,
        payloads,
        species=species,
        axis=axis,
    )


def _position_time_data(
    profile: PositionProfile,
    *,
    time_axis: str,
) -> tuple[np.ndarray, str]:
    normalized = time_axis.strip().lower()
    if normalized == "ps":
        return profile.time_ps, "Time (ps)"
    if normalized == "fs":
        return profile.time_fs, "Time (fs)"
    if normalized == "step":
        return profile.step, "Timestep"
    if normalized == "frame":
        return profile.frame_index.astype(float), "Frame index"
    raise ValueError(
        f"Unsupported position time_axis '{time_axis}'. Choose 'ps', 'fs', 'step', or 'frame'."
    )


def _normalize_component_token(component: str) -> str:
    token = component.strip().lower().replace("_", "-")
    if token in {"distance", "x", "y", "z", "xy-z"}:
        return token
    if token in {"xy-z-color", "xy-z-colormap", "trajectory", "xyz"}:
        return "xy-z"
    raise ValueError(
        f"Unsupported position component '{component}'. "
        "Choose 'distance', 'x', 'y', 'z', or 'xy-z'."
    )


def _normalize_map_color_token(map_color: str) -> str:
    token = map_color.strip().lower().replace("_", "-")
    if token in {"distance", "z"}:
        return token
    if token in {"surface-distance", "dist"}:
        return "distance"
    raise ValueError(f"Unsupported position map_color '{map_color}'. Choose 'distance' or 'z'.")


def _position_component_data(
    profile: PositionProfile,
    *,
    component: str,
) -> tuple[np.ndarray, str]:
    normalized = _normalize_component_token(component)
    if normalized == "distance":
        if profile.coordinate_mode != "distance":
            LOGGER.warning(
                "Position profile '%s' has no valid surface-distance reference; using %s-axis values.",
                profile.species,
                profile.axis.upper(),
            )
            return profile.distance_to_surface, f"{profile.axis.upper()} (A)"
        return profile.distance_to_surface, "Distance to the surface ($\\mathrm{\\AA}$)"
    if normalized == "x":
        return profile.x, "X (A)"
    if normalized == "y":
        return profile.y, "Y (A)"
    if normalized == "z":
        return profile.z, "Z (A)"
    raise ValueError("Component 'xy-z' must be rendered via XY trajectory plotting.")


def _position_map_color_data(
    profile: PositionProfile,
    *,
    map_color: str,
) -> tuple[np.ndarray, str]:
    normalized = _normalize_map_color_token(map_color)
    if normalized == "distance":
        return _position_component_data(profile, component="distance")
    return np.asarray(profile.z, dtype=float), "Z (A)"


def _build_xy_segments(
    x_values: np.ndarray,
    y_values: np.ndarray,
    color_values: np.ndarray,
    *,
    cell_lengths_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size != y_values.size or x_values.size != color_values.size:
        raise ValueError(
            "x, y, and color arrays must have matching length for XY segment building."
        )
    if x_values.size < 2:
        return (
            np.empty((0, 2, 2), dtype=float),
            np.empty((0,), dtype=float),
        )

    points = np.column_stack((x_values, y_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_colors = 0.5 * (color_values[:-1] + color_values[1:])

    if cell_lengths_xy is None:
        return np.asarray(segments, dtype=float), np.asarray(segment_colors, dtype=float)

    x_length, y_length = cell_lengths_xy
    if x_length <= 0.0 or y_length <= 0.0:
        return np.asarray(segments, dtype=float), np.asarray(segment_colors, dtype=float)

    dx = np.abs(np.diff(x_values))
    dy = np.abs(np.diff(y_values))
    # Break PBC-jump connectors so trajectories do not draw artificial lines across the box.
    keep = (dx <= (0.5 * x_length + 1e-12)) & (dy <= (0.5 * y_length + 1e-12))
    return np.asarray(segments[keep], dtype=float), np.asarray(segment_colors[keep], dtype=float)


def _default_position_series_labels(profile: PositionProfile) -> list[str]:
    return [f"{profile.species}[{int(atom_index)}]" for atom_index in profile.atom_indices.tolist()]


def _first_non_none(values: list[float | None] | None) -> float | None:
    if not values:
        return None
    for value in values:
        if value is not None:
            return float(value)
    return None


def _plot_position_xy_z_projection(
    profiles: list[PositionProfile],
    *,
    map_color: str,
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
        raise ValueError("At least one position profile is required.")

    series_total = sum(max(0, int(profile.n_atoms)) for profile in profiles)
    if series_enabled is not None and len(series_enabled) != series_total:
        raise ValueError(
            "series_enabled count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if line_colors is not None and len(line_colors) != series_total:
        raise ValueError(
            "line_colors count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_line_widths is not None and len(series_line_widths) != series_total:
        raise ValueError(
            "series_line_widths count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_markers is not None and len(series_markers) != series_total:
        raise ValueError(
            "series_markers count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_normalization_modes is not None and len(series_normalization_modes) != series_total:
        raise ValueError(
            "series_normalization_modes count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_normalization_values is not None and len(series_normalization_values) != series_total:
        raise ValueError(
            "series_normalization_values count must match the number of plotted position atom series "
            f"({series_total})."
        )
    if series_normalization_x_refs is not None and len(series_normalization_x_refs) != series_total:
        raise ValueError(
            "series_normalization_x_refs count must match the number of plotted position atom series "
            f"({series_total})."
        )

    if x_bin_width is not None:
        LOGGER.warning(
            "Position component 'xy-z' ignores time-section/x-bin settings (received %.6g; reducer=%s).",
            x_bin_width,
            x_bin_reducer or "mean",
        )
    if series_normalization_modes is not None:
        LOGGER.warning("Position component 'xy-z' ignores per-series y-normalization settings.")
    if line_colors is not None:
        LOGGER.warning(
            "Position component 'xy-z' ignores per-series fixed line colors and uses %s colormap values.",
            _normalize_map_color_token(map_color),
        )

    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    segment_blocks: list[np.ndarray] = []
    segment_color_blocks: list[np.ndarray] = []
    point_x_values: list[float] = []
    point_y_values: list[float] = []
    point_color_values: list[float] = []
    color_label_reference: str | None = None

    series_index = 0
    for profile in profiles:
        x_matrix = np.asarray(profile.x, dtype=float)
        y_matrix = np.asarray(profile.y, dtype=float)
        color_matrix, color_label = _position_map_color_data(profile, map_color=map_color)
        cell_lengths_xy = None
        if profile.cell_lengths_angstrom is not None:
            cell_lengths_xy = (
                float(profile.cell_lengths_angstrom[0]),
                float(profile.cell_lengths_angstrom[1]),
            )
        if color_label_reference is None:
            color_label_reference = color_label
        elif color_label != color_label_reference:
            color_label_reference = "Color value (A)"
        if not (x_matrix.shape == y_matrix.shape == color_matrix.shape):
            raise ValueError(
                f"Position profile '{profile.species}' has inconsistent x/y/color matrix shapes."
            )
        for atom_column in range(x_matrix.shape[1]):
            is_enabled = True if series_enabled is None else bool(series_enabled[series_index])
            series_index += 1
            if not is_enabled:
                continue

            x_values = x_matrix[:, atom_column]
            y_values = y_matrix[:, atom_column]
            color_values = color_matrix[:, atom_column]
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
                cell_lengths_xy=cell_lengths_xy,
            )
            if segments.size == 0:
                continue
            segment_blocks.append(segments)
            segment_color_blocks.append(segment_colors)

    if not segment_blocks and not point_x_values:
        raise ValueError("No enabled atom trajectories available for 'xy-z' position plotting.")

    color_samples: list[np.ndarray] = []
    if segment_color_blocks:
        color_samples.extend(segment_color_blocks)
    if point_color_values:
        color_samples.append(np.asarray(point_color_values, dtype=float))
    color_all = np.concatenate(color_samples)
    color_min = float(np.nanmin(color_all))
    color_max = float(np.nanmax(color_all))
    if not np.isfinite(color_min) or not np.isfinite(color_max):
        raise ValueError("Cannot render 'xy-z' projection because color values are non-finite.")
    if color_min == color_max:
        color_min -= 0.5
        color_max += 0.5
    norm = mcolors.Normalize(vmin=color_min, vmax=color_max)

    line_collection_kwargs = _sanitize_line_collection_kwargs(line_kwargs)
    explicit_line_width = _first_non_none(series_line_widths)
    line_collection_kwargs.setdefault(
        "linewidths",
        style.line_width if explicit_line_width is None else explicit_line_width,
    )

    marker_size = max(9.0, (style.line_width * 7.0) ** 2)
    active_backend = configure_matplotlib_backend(
        interactive=show,
        preferred_backend=preferred_backend,
    )
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
            colorbar.set_label(
                color_label_reference or "Color value (A)",
                fontsize=style.label_font_size,
            )
            colorbar.ax.tick_params(labelsize=style.tick_font_size)

        xlabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(
            format_axis_label_units(resolve_explicit_plot_text(x_label, "X (A)")),
            **xlabel_kwargs,
        )
        ax.set_ylabel(
            format_axis_label_units(resolve_explicit_plot_text(y_label, "Y (A)")),
            **ylabel_kwargs,
        )
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size)
        else:
            ax.set_title(
                title
                or (
                    "XY trajectories colored by distance to surface"
                    if _normalize_map_color_token(map_color) == "distance"
                    else "XY trajectories colored by Z"
                ),
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
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )
            if tick_axis_hint in {"both", "y"}:
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                )
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
        default_cell_x: float | None = None
        default_cell_y: float | None = None
        profile_lengths = [
            profile.cell_lengths_angstrom
            for profile in profiles
            if profile.cell_lengths_angstrom is not None
        ]
        if profile_lengths:
            default_cell_x = max(float(lengths[0]) for lengths in profile_lengths)
            default_cell_y = max(float(lengths[1]) for lengths in profile_lengths)
        effective_x_lim = x_lim
        effective_y_lim = y_lim
        if effective_x_lim is None and default_cell_x is not None:
            effective_x_lim = (0.0, default_cell_x)
        if effective_y_lim is None and default_cell_y is not None:
            effective_y_lim = (0.0, default_cell_y)

        if effective_x_lim is not None:
            left = None if effective_x_lim[0] is None else float(effective_x_lim[0])
            right = None if effective_x_lim[1] is None else float(effective_x_lim[1])
            ax.set_xlim(left=left, right=right)
        if effective_y_lim is not None:
            bottom = None if effective_y_lim[0] is None else float(effective_y_lim[0])
            top = None if effective_y_lim[1] is None else float(effective_y_lim[1])
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
            if show_blocking:
                LOGGER.info(
                    "Showing interactive plot window using backend '%s'. Close the window to continue.",
                    active_backend,
                )
            else:
                LOGGER.info(
                    "Showing interactive plot window using backend '%s'.",
                    active_backend,
                )
            plt.show(block=show_blocking)
            if not show_blocking:
                plt.pause(0.001)

        if not (show and not show_blocking):
            plt.close(fig)
        return output_path


def plot_position_profile(
    profile: PositionProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    component: str = "distance",
    map_color: str = "distance",
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
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
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
    """Plot one atom-resolved position profile."""
    normalized_component = _normalize_component_token(component)
    if normalized_component == "xy-z":
        return _plot_position_xy_z_projection(
            [profile],
            map_color=map_color,
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

    x_values, default_x_label = _position_time_data(profile, time_axis=time_axis)
    matrix, default_y_label = _position_component_data(profile, component=normalized_component)
    default_labels = _default_position_series_labels(profile)
    effective_legend = (profile.n_atoms <= 12) if legend is None else legend
    schema_labels = default_plot_labels("position")
    default_title = (
        f"{profile.species} atom-resolved positions"
        if schema_labels is not None
        else "Atom-resolved positions"
    )
    labels = resolve_series_labels(default_labels, series_labels, series_kind="position")

    if matrix.shape[1] == 1:
        resolved_label = line_label
        if resolved_label is None and effective_legend:
            resolved_label = labels[0]
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
            np.asarray(x_values, dtype=float),
            np.asarray(matrix[:, 0], dtype=float),
            title=title or default_title,
            x_label=resolve_explicit_plot_text(x_label, default_x_label),
            y_label=resolve_explicit_plot_text(y_label, default_y_label),
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
                None
                if not series_fit_labels or not series_fit_labels[0]
                else str(series_fit_labels[0])
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

    return plot_multi_line_series(
        [np.asarray(x_values, dtype=float) for _ in range(matrix.shape[1])],
        [np.asarray(matrix[:, col], dtype=float) for col in range(matrix.shape[1])],
        labels,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, default_y_label),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_line_kwargs=series_line_kwargs,
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


def plot_position_profiles(
    profiles: list[PositionProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    component: str = "distance",
    map_color: str = "distance",
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
    """Plot one or more atom-resolved position profiles."""
    if not profiles:
        raise ValueError("At least one position profile is required.")
    normalized_component = _normalize_component_token(component)
    if normalized_component == "xy-z":
        return _plot_position_xy_z_projection(
            profiles,
            map_color=map_color,
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
        return plot_position_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            component=normalized_component,
            map_color=map_color,
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
            series_labels=series_labels,
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
            series_line_kwargs=series_line_kwargs,
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

    first_profile = profiles[0]
    _x_template, default_x_label = _position_time_data(first_profile, time_axis=time_axis)
    _matrix, default_y_label = _position_component_data(
        first_profile,
        component=normalized_component,
    )
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []
    default_labels: list[str] = []
    for profile in profiles:
        x_values, _x_label = _position_time_data(profile, time_axis=time_axis)
        matrix, _y_label = _position_component_data(profile, component=normalized_component)
        for column, atom_index in enumerate(profile.atom_indices.tolist()):
            x_series.append(np.asarray(x_values, dtype=float))
            y_series.append(np.asarray(matrix[:, column], dtype=float))
            default_labels.append(f"{profile.species}[{int(atom_index)}]")

    labels = resolve_series_labels(default_labels, series_labels, series_kind="position")
    effective_legend = (len(labels) <= 12) if legend is None else legend
    schema_labels = default_plot_labels("position")
    default_title = "Atom-resolved positions" if schema_labels is not None else "Position profile"

    return plot_multi_line_series(
        x_series,
        y_series,
        labels,
        title=title or default_title,
        x_label=resolve_explicit_plot_text(x_label, default_x_label),
        y_label=resolve_explicit_plot_text(y_label, default_y_label),
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
