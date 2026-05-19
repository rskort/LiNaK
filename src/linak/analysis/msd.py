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

from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.msd_mapping import resolve_msd_plot_mapping
from ..storage.hdf5_utils import write_linak_hdf5
from .common import (
    frame_has_usable_cell as _common_frame_has_usable_cell,
    normalize_species_label as _normalize_species,
    read_profile_payloads,
    read_profile_payloads_by_index,
    select_species_indices as _select_indices,
    use_multi_series_plot,
)
from .schema import build_profile_metadata, default_plot_labels
from .statistics import (
    SeriesStatistics,
    build_series_statistics,
    build_statistics_metadata,
    statistics_payload_from_series_map,
    statistics_series_map_from_datasets,
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

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MSDProfile:
    """Container for a mean-squared displacement profile."""

    species: str
    time_fs: np.ndarray
    time_ps: np.ndarray
    msd: np.ndarray
    n_frames: int
    series_statistics: dict[str, SeriesStatistics] | None = None


def _frame_has_usable_cell(frame: Atoms) -> bool:
    """Preserve MSD's stricter periodic-cell requirement."""
    return _common_frame_has_usable_cell(frame, require_all_pbc=True)


def compute_msd(
    frames: list[Atoms],
    species: str | None = "all",
    timestep_fs: float = 1.0,
) -> MSDProfile:
    """Compute mean-squared displacement from the first frame reference."""
    LOGGER.debug(
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
    squared_samples_by_lag = np.zeros((len(frames), reference_indices.size), dtype=float)
    use_pbc_mic = all(_frame_has_usable_cell(frame) for frame in frames)

    if use_pbc_mic:
        LOGGER.debug("MSD mode: periodic minimum-image accumulation.")
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
                squared_displacements = np.sum(displacements**2, axis=1)
                squared_samples_by_lag[i, :] = np.asarray(squared_displacements, dtype=float)
                msd[i] = float(np.mean(squared_displacements))
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
                squared_displacements = np.sum(displacements**2, axis=1)
                squared_samples_by_lag[i, :] = np.asarray(squared_displacements, dtype=float)
                msd[i] = float(np.mean(squared_displacements))
                progress.update()

    time_fs = np.arange(len(frames), dtype=float) * timestep_fs
    time_ps = time_fs / 1000.0
    statistics = build_series_statistics(
        point_count=np.full(len(frames), reference_indices.size, dtype=int),
        sample_values=squared_samples_by_lag.T,
        block_values=None,
    )

    return MSDProfile(
        species=species_label,
        time_fs=time_fs,
        time_ps=time_ps,
        msd=msd,
        n_frames=len(frames),
        series_statistics={"msd_A2": statistics},
    )


def save_msd_profile(
    profile: MSDProfile,
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save MSD profile to LiNaK HDF5 and return written path."""
    metadata_payload = {
        "species": profile.species,
        "n_frames": profile.n_frames,
    }
    if profile.series_statistics:
        metadata_payload["statistics"] = build_statistics_metadata(
            statistics_by_series=profile.series_statistics,
            block_lengths=None,
        )
    metadata = build_profile_metadata(
        analysis="msd",
        metadata=metadata_payload,
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
            **statistics_payload_from_series_map(profile.series_statistics),
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


def _load_msd_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *,
    species: str | None = None,
) -> list[MSDProfile]:
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
        series_statistics = statistics_series_map_from_datasets(
            datasets,
            dataset_names=("msd_A2",),
        )

        profiles.append(
            MSDProfile(
                species=resolved_species,
                time_fs=time_fs,
                time_ps=time_ps,
                msd=msd,
                n_frames=n_frames,
                series_statistics=series_statistics,
            )
        )
    return profiles


def load_msd_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    species: str | None = None,
) -> list[MSDProfile]:
    """Load selected MSD profiles by profile index from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="msd",
        label="MSD",
    )
    return _load_msd_profiles_from_payloads(source_path, payloads, species=species)


def load_msd_profiles(path: str | Path, *, species: str | None = None) -> list[MSDProfile]:
    """Load one or more MSD profiles from LiNaK HDF5."""
    source_path, payloads = read_profile_payloads(
        path,
        analysis="msd",
        label="MSD",
    )
    return _load_msd_profiles_from_payloads(source_path, payloads, species=species)


def plot_msd_profile(
    profile: MSDProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
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
    """Plot MSD profile using shared LiNaK plotting style."""
    resolved_mapping = resolve_msd_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
        time_axis=time_axis,
    )
    runtime_time_axis = str(resolved_mapping.renderer_options.get("time_axis") or "ps")
    x_values = profile.time_fs if runtime_time_axis == "fs" else profile.time_ps
    schema_labels = default_plot_labels("msd")
    if schema_labels is None:
        default_x = "Time (fs)" if runtime_time_axis == "fs" else "Time (ps)"
    else:
        default_x = (
            schema_labels[0].replace("(ps)", "(fs)")
            if runtime_time_axis == "fs"
            else schema_labels[0]
        )
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
        x_values,
        profile.msd,
        title=title or f"{profile.species} mean squared displacement",
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
        else profile.series_statistics.get("msd_A2"),
        error_config=error_config,
        normalization_mode=single_series.normalization_mode,
        normalization_value=single_series.normalization_value,
        normalization_x_ref=single_series.normalization_x_ref,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="msd",
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


def plot_msd_profiles(
    profiles: list[MSDProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
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
    """Plot one or more MSD profiles."""
    if not profiles:
        raise ValueError("At least one MSD profile is required.")
    first_profile = profiles[0]
    resolved_mapping = resolve_msd_plot_mapping(
        contract=data_contract,
        profile=first_profile,
        mapping=view_mapping,
        time_axis=time_axis,
    )
    runtime_time_axis = str(resolved_mapping.renderer_options.get("time_axis") or "ps")
    schema_labels = default_plot_labels("msd")
    if schema_labels is None:
        default_x = "Time (fs)" if runtime_time_axis == "fs" else "Time (ps)"
    else:
        default_x = (
            schema_labels[0].replace("(ps)", "(fs)")
            if runtime_time_axis == "fs"
            else schema_labels[0]
        )
    default_y = "MSD (Angstrom^2)" if schema_labels is None else schema_labels[1]
    default_labels = [profile.species for profile in profiles]
    labels = resolve_series_labels(
        default_labels,
        series_labels,
        series_kind="MSD",
    )

    if not use_multi_series_plot(
        profile_count=len(profiles),
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
    ):
        return plot_msd_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            data_contract=resolved_mapping.contract,
            view_mapping=resolved_mapping.mapping,
            time_axis=runtime_time_axis,
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
        [
            profile.time_fs if runtime_time_axis == "fs" else profile.time_ps
            for profile in profiles
        ],
        [profile.msd for profile in profiles],
        labels,
        title=title or "Mean squared displacement",
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
            None if profile.series_statistics is None else profile.series_statistics.get("msd_A2")
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
        analysis_name="msd",
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
