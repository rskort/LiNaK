"""MSD analysis routines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from ..storage.hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles,
    write_linak_hdf5,
)
from .schema import build_profile_metadata, default_plot_labels
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_line_series,
    plot_multi_line_series,
    resolve_series_labels,
    resolve_single_series_options,
)
from ..progress import ProgressBar
from ..utils import ensure_positive

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MSDProfile:
    """Container for a mean-squared displacement profile."""

    species: str
    time_fs: np.ndarray
    time_ps: np.ndarray
    msd: np.ndarray
    n_frames: int


def _normalize_species(species: str | None) -> str:
    """Normalize species selection for atom-resolved analyses."""
    if species is None:
        return "ALL"

    species = species.strip()
    if not species or species.lower() == "all" or species == "*":
        return "ALL"

    return species[0].upper() + species[1:].lower()


def _select_indices(frame: Atoms, species: str) -> np.ndarray:
    """Return selected atom indices for one frame."""
    if species == "ALL":
        return np.arange(len(frame), dtype=int)

    symbols = np.asarray(frame.get_chemical_symbols())
    return np.where(symbols == species)[0]


def _frame_has_usable_cell(frame: Atoms) -> bool:
    """Check whether a frame has finite non-zero cell and enabled periodicity."""
    if not bool(np.all(frame.get_pbc())):
        return False
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return False
    volume = abs(float(np.linalg.det(cell)))
    return volume > 0.0


def compute_msd(
    frames: list[Atoms],
    species: str | None = "all",
    timestep_fs: float = 1.0,
) -> MSDProfile:
    """Compute mean-squared displacement from the first frame reference."""
    LOGGER.info(
        "Computing MSD (species=%s, timestep_fs=%.6g).",
        species,
        timestep_fs,
    )
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("timestep_fs", timestep_fs)
    species_label = _normalize_species(species)

    reference = frames[0]
    reference_indices = _select_indices(reference, species_label)
    if reference_indices.size == 0:
        raise ValueError(f"No atoms found for species '{species_label}' in frame 0.")

    reference_positions = np.asarray(reference.positions[reference_indices], dtype=float)
    msd = np.zeros(len(frames), dtype=float)
    use_pbc_mic = all(_frame_has_usable_cell(frame) for frame in frames)

    if use_pbc_mic:
        LOGGER.info("MSD mode: periodic minimum-image accumulation.")
        prev_positions = np.asarray(reference.positions[reference_indices], dtype=float)
        unwrapped_positions = reference_positions.copy()
        msd[0] = 0.0

        with ProgressBar(
            desc=f"Computing MSD for {species_label}", total=max(1, len(frames) - 1), unit="frame"
        ) as progress:
            for i in range(1, len(frames)):
                frame = frames[i]
                if len(frame) != len(reference):
                    raise ValueError("All frames must contain the same number of atoms for MSD.")

                frame_indices = _select_indices(frame, species_label)
                if frame_indices.size != reference_indices.size:
                    raise ValueError(
                        f"Frame {i} does not preserve the selected atom count for species "
                        f"'{species_label}'."
                    )

                current_positions = np.asarray(frame.positions[frame_indices], dtype=float)
                step_vectors = current_positions - prev_positions
                mic_steps, _ = find_mic(step_vectors, cell=frame.cell, pbc=frame.get_pbc())
                unwrapped_positions += np.asarray(mic_steps, dtype=float)

                displacements = unwrapped_positions - reference_positions
                msd[i] = float(np.mean(np.sum(displacements**2, axis=1)))
                prev_positions = current_positions
                progress.update()
    else:
        LOGGER.warning("MSD mode: direct displacement (no usable periodic cell in all frames).")
        with ProgressBar(
            desc=f"Computing MSD for {species_label}", total=len(frames), unit="frame"
        ) as progress:
            for i, frame in enumerate(frames):
                if len(frame) != len(reference):
                    raise ValueError("All frames must contain the same number of atoms for MSD.")

                frame_indices = _select_indices(frame, species_label)
                if frame_indices.size != reference_indices.size:
                    raise ValueError(
                        f"Frame {i} does not preserve the selected atom count for species "
                        f"'{species_label}'."
                    )

                displacements = (
                    np.asarray(frame.positions[frame_indices], dtype=float) - reference_positions
                )
                msd[i] = float(np.mean(np.sum(displacements**2, axis=1)))
                progress.update()

    time_fs = np.arange(len(frames), dtype=float) * timestep_fs
    time_ps = time_fs / 1000.0

    return MSDProfile(
        species=species_label,
        time_fs=time_fs,
        time_ps=time_ps,
        msd=msd,
        n_frames=len(frames),
    )


def save_msd_profile(
    profile: MSDProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save MSD profile to LiNaK HDF5 and return written path."""
    metadata = build_profile_metadata(
        analysis="msd",
        metadata={
            "species": profile.species,
            "n_frames": profile.n_frames,
        },
    )
    if additional_metadata:
        metadata.update(dict(additional_metadata))

    output_path = write_linak_hdf5(
        output,
        analysis="msd",
        datasets={
            "time_fs": profile.time_fs,
            "time_ps": profile.time_ps,
            "msd_A2": profile.msd,
        },
        metadata=metadata,
    )
    LOGGER.info("Saved MSD data to '%s'.", output_path)
    return output_path


def load_msd_profile(path: str | Path, *, species: str | None = None) -> MSDProfile:
    """Load one MSD profile from LiNaK HDF5.

    For profile-collection files, this returns the first profile.
    """
    profiles = load_msd_profiles(path, species=species)
    if not profiles:
        source_path = Path(path).expanduser().resolve()
        raise ValueError(f"MSD HDF5 '{source_path}' does not contain any MSD profiles.")
    return profiles[0]


def load_msd_profiles(path: str | Path, *, species: str | None = None) -> list[MSDProfile]:
    """Load one or more MSD profiles from LiNaK HDF5."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"MSD profile not found: {source_path}")

    if is_hdf5_path(source_path):
        payloads = read_linak_hdf5_profiles(source_path, expected_analysis="msd")
        profiles: list[MSDProfile] = []
        for datasets, metadata in payloads:
            required = ("time_fs", "time_ps", "msd_A2")
            missing = [name for name in required if name not in datasets]
            if missing:
                raise ValueError(
                    f"MSD HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
                )

            meta_species = str(metadata.get("species", "")).strip()
            if species is not None and species.strip():
                resolved_species = _normalize_species(species)
            elif meta_species:
                resolved_species = meta_species
            else:
                resolved_species = "UNKNOWN"

            time_fs = np.asarray(datasets["time_fs"], dtype=float)
            time_ps = np.asarray(datasets["time_ps"], dtype=float)
            msd = np.asarray(datasets["msd_A2"], dtype=float)
            n_frames = int(metadata.get("n_frames", time_fs.size))

            profiles.append(
                MSDProfile(
                    species=resolved_species,
                    time_fs=time_fs,
                    time_ps=time_ps,
                    msd=msd,
                    n_frames=n_frames,
                )
            )
        return profiles

    raise ValueError(f"Unsupported MSD profile format for '{source_path}'. Use .h5/.hdf5.")


def plot_msd_profile(
    profile: MSDProfile,
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
    line_label: str | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
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
    """Plot MSD profile using shared LiNaK plotting style."""
    schema_labels = default_plot_labels("msd")
    default_x = "Time (ps)" if schema_labels is None else schema_labels[0]
    default_y = "MSD (Angstrom^2)" if schema_labels is None else schema_labels[1]
    resolved_label = line_label
    if resolved_label is None and legend:
        resolved_label = profile.species
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
        profile.time_ps,
        profile.msd,
        title=title or f"{profile.species} mean squared displacement",
        x_label=x_label or default_x,
        y_label=y_label or default_y,
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        line_label=resolved_label,
        line_color=single_series.line_color,
        line_width_override=single_series.line_width_override,
        line_marker=single_series.line_marker,
        line_visible=single_series.line_visible,
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


def plot_msd_profiles(
    profiles: list[MSDProfile],
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
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
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
    """Plot one or more MSD profiles."""
    schema_labels = default_plot_labels("msd")
    default_x = "Time (ps)" if schema_labels is None else schema_labels[0]
    default_y = "MSD (Angstrom^2)" if schema_labels is None else schema_labels[1]
    if not profiles:
        raise ValueError("At least one MSD profile is required.")
    default_labels = [profile.species for profile in profiles]
    labels = resolve_series_labels(
        default_labels,
        series_labels,
        series_kind="MSD",
    )

    if len(profiles) == 1:
        return plot_msd_profile(
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
        [profile.time_ps for profile in profiles],
        [profile.msd for profile in profiles],
        labels,
        title=title or "Mean squared displacement",
        x_label=x_label or default_x,
        y_label=y_label or default_y,
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
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
