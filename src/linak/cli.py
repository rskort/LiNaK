"""Command-line interface for LiNaK."""

from __future__ import annotations

import argparse
from copy import deepcopy
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, metadata as package_metadata
import importlib
import json
import logging
from math import isclose
import os
from pathlib import Path
import re
import shutil
import shlex
import sys
import tempfile
import textwrap
from time import perf_counter
from typing import Any, Callable, TYPE_CHECKING

from . import __version__
from .runtime_threads import configure_native_thread_env

_NATIVE_THREAD_ENV_CONFIGURATION = configure_native_thread_env()

np = importlib.import_module("numpy")

if TYPE_CHECKING:
    from ase import Atoms
    from .plot.plotting import PlotStyle
    from .analysis.potential import PotentialComputationFailure, PotentialRecord

LOGGER = logging.getLogger(__name__)
DEFAULT_INTERACTIVE_BACKEND = "QtAgg"
_PROJECT_AUTHOR_LINE = re.compile(r'^\s*authors\s*=\s*\[\{\s*name\s*=\s*"([^"]+)"')
_TABULAR_COMMAND = "hdf5"
_TABULAR_COMMAND_ALIASES = (
    "hd",
    "h5",
)
_TABULAR_COMMAND_TOKENS = {_TABULAR_COMMAND, *_TABULAR_COMMAND_ALIASES}
_PLOT_PROFILE_DENSITY = "plot:density"
_PLOT_PROFILE_MSD = "plot:msd"
_PLOT_PROFILE_RDF = "plot:rdf"
_PLOT_PROFILE_POSITION = "plot:position"
_PLOT_PROFILE_COORDINATION = "plot:coordination"
_PLOT_PROFILE_POTENTIAL = "plot:potential"
_PLOT_PROFILE_ORIENTATION = "plot:orientation"
_PLOT_PROFILE_TABLE = "plot:table"
_ANALYSIS_TO_PROFILE_KEY = {
    "density": _PLOT_PROFILE_DENSITY,
    "msd": _PLOT_PROFILE_MSD,
    "rdf": _PLOT_PROFILE_RDF,
    "position": _PLOT_PROFILE_POSITION,
    "coordination": _PLOT_PROFILE_COORDINATION,
    "potential": _PLOT_PROFILE_POTENTIAL,
    "orientation": _PLOT_PROFILE_ORIENTATION,
    "table": _PLOT_PROFILE_TABLE,
}
_PROFILE_KEY_TO_ANALYSIS = {value: key for key, value in _ANALYSIS_TO_PROFILE_KEY.items()}
_LINAK_OUTPUT_DIRNAME = "LiNaK_outputs"


@dataclass(frozen=True)
class _GuiPlotRenderContext:
    profile: Any
    plot_source_label: str
    plotter_kwargs: dict[str, Any] | None
    fallback_labels_by_source: list[list[str]]
    default_series_labels: list[str]
    series_descriptors: list[dict[str, Any]]
    profile_filter_options: dict[str, Any] | None = None

    @property
    def series_count(self) -> int:
        return len(self.series_descriptors)


@dataclass
class _LazyGuiSeriesCatalog:
    sources: list[str]
    plot_source_label: str
    plotter_kwargs: dict[str, Any] | None
    descriptor_segments_by_source: list[list[dict[str, Any]]]
    profile_filter_options: dict[str, Any] | None
    load_profiles: Callable[[list[dict[str, Any]]], list[Any]]
    default_series_labels: list[str] = field(default_factory=list)
    _active_profiles_by_series_id: dict[str, Any] = field(default_factory=dict)

    @property
    def series_descriptors(self) -> list[dict[str, Any]]:
        return [
            dict(descriptor)
            for segment in self.descriptor_segments_by_source
            for descriptor in segment
        ]

    @property
    def fallback_labels_by_source(self) -> list[list[str]]:
        return [
            [
                str(descriptor.get("default_label") or f"Series {index + 1}")
                for index, descriptor in enumerate(segment)
            ]
            for segment in self.descriptor_segments_by_source
        ]

    def build_initial_context(self) -> _GuiPlotRenderContext:
        return _GuiPlotRenderContext(
            profile=[],
            plot_source_label=self.plot_source_label,
            plotter_kwargs=self.plotter_kwargs,
            fallback_labels_by_source=self.fallback_labels_by_source,
            default_series_labels=list(self.default_series_labels),
            series_descriptors=self.series_descriptors,
            profile_filter_options=deepcopy(self.profile_filter_options),
        )

    def build_render_context(self, args: argparse.Namespace) -> _GuiPlotRenderContext:
        active_descriptors_by_source, active_ids = _filter_active_gui_descriptor_segments(
            args=args,
            descriptor_segments_by_source=self.descriptor_segments_by_source,
        )
        active_id_set = set(active_ids)
        for series_id in list(self._active_profiles_by_series_id):
            if series_id not in active_id_set:
                self._active_profiles_by_series_id.pop(series_id, None)

        active_descriptors = [
            dict(descriptor) for segment in active_descriptors_by_source for descriptor in segment
        ]
        missing_descriptors = [
            descriptor
            for descriptor in active_descriptors
            if str(descriptor.get("series_id") or "") not in self._active_profiles_by_series_id
        ]
        if missing_descriptors:
            loaded_profiles = self.load_profiles(missing_descriptors)
            if len(loaded_profiles) != len(missing_descriptors):
                raise ValueError("Lazy GUI series loader returned mismatched profile count.")
            for descriptor, profile in zip(missing_descriptors, loaded_profiles):
                series_id = str(descriptor.get("series_id") or "").strip()
                if not series_id:
                    raise ValueError("Lazy GUI descriptor is missing a series_id.")
                self._active_profiles_by_series_id[series_id] = profile

        return _GuiPlotRenderContext(
            profile=[
                self._active_profiles_by_series_id[str(descriptor.get("series_id") or "")]
                for descriptor in active_descriptors
            ],
            plot_source_label=self.plot_source_label,
            plotter_kwargs=self.plotter_kwargs,
            fallback_labels_by_source=[
                [
                    str(descriptor.get("default_label") or f"Series {index + 1}")
                    for index, descriptor in enumerate(segment)
                ]
                for segment in active_descriptors_by_source
            ],
            default_series_labels=[
                str(descriptor.get("default_label") or f"Series {index + 1}")
                for index, descriptor in enumerate(active_descriptors)
            ],
            series_descriptors=active_descriptors,
            profile_filter_options=deepcopy(self.profile_filter_options),
        )


_PERSISTED_PLOT_SETTING_OPTION_FLAGS = {
    "axis": ("--axis",),
    "backend": ("--backend",),
    "bins": ("--bins",),
    "dpi": ("--dpi",),
    "figsize": ("--figsize",),
    "file_labels": ("--file-labels",),
    "font_family": ("--font-family",),
    "grid": ("--grid", "--no-grid"),
    "grid_alpha": ("--grid-alpha",),
    "grid_linestyle": ("--grid-linestyle",),
    "grid_linewidth": ("--grid-linewidth",),
    "group": ("--group",),
    "kind": ("--kind",),
    "label_font_size": ("--label-font-size",),
    "legend": ("--legend", "--no-legend"),
    "legend_font_size": ("--legend-font-size",),
    "legend_loc": ("--legend-loc",),
    "legend_title": ("--legend-title",),
    "line_color": ("--line-color",),
    "line_colors": ("--line-colors",),
    "line_width": ("--line-width",),
    "quantity": ("--quantity",),
    "series_labels": ("--labels", "--series-labels"),
    "species": ("--species",),
    "species_a": ("--species-a",),
    "species_b": ("--species-b",),
    "tick_font_size": ("--tick-font-size",),
    "title": ("--title",),
    "title_visible": ("--title-mode",),
    "title_font_size": ("--title-font-size",),
    "x": ("--x",),
    "x_label": ("--x-label",),
    "x_lim": ("--x-min", "--x-max"),
    "x_mode": ("--x-mode",),
    "x_bin_width": ("--x-bin-width",),
    "x_bin_reducer": ("--x-bin-reducer",),
    "x_scale": ("--x-scale",),
    "x_tick_rotation": ("--x-tick-rotation",),
    "x_ticks": ("--x-ticks",),
    "y": ("--y",),
    "y_label": ("--y-label",),
    "y_lim": ("--y-min", "--y-max"),
    "y_scale": ("--y-scale",),
    "y_tick_rotation": ("--y-tick-rotation",),
    "y_ticks": ("--y-ticks",),
    "ticks": ("--ticks",),
    "markers": ("--markers",),
    "component": ("--component",),
    "map_color": ("--map-color",),
    "time_axis": ("--time-axis",),
    "time_section_width": ("--time-section-width",),
}

_PLOT_SETTINGS_COMMON_KEYS = (
    "title",
    "title_visible",
    "x_label",
    "y_label",
    "x_scale",
    "y_scale",
    "x_lim",
    "y_lim",
    "x_ticks",
    "y_ticks",
    "x_tick_rotation",
    "y_tick_rotation",
    "x_label_pad",
    "y_label_pad",
    "series_labels",
    "series_order",
    "series_descriptors",
    "series_overrides",
    "series_enabled",
    "series_line_widths",
    "series_markers",
    "series_normalization_modes",
    "series_normalization_values",
    "series_normalization_x_refs",
    "x_bin_width",
    "x_bin_reducer",
    "matplotlib_rc",
    "figure_kwargs",
    "axes_kwargs",
    "line_kwargs",
    "series_line_kwargs",
    "grid_kwargs",
    "legend_kwargs",
    "tick_params_kwargs",
    "tight_layout_kwargs",
    "savefig_kwargs",
    "legend",
    "legend_title",
    "legend_loc",
    "ticks",
    "markers",
    "figsize",
    "dpi",
    "font_family",
    "title_font_size",
    "label_font_size",
    "tick_font_size",
    "legend_font_size",
    "line_width",
    "line_color",
    "line_colors",
    "grid",
    "grid_linestyle",
    "grid_linewidth",
    "grid_alpha",
)
_PLOT_SETTINGS_DENSITY_KEYS = (
    "species",
    "axis",
    "x_mode",
    "quantity",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_MSD_KEYS = (
    "species",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_RDF_KEYS = (
    "species_a",
    "species_b",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_POSITION_KEYS = (
    "species",
    "axis",
    "component",
    "map_color",
    "time_axis",
    "time_section_width",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_COORDINATION_KEYS = (
    "species_a",
    "species_b",
    "axis",
    "component",
    "time_axis",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_POTENTIAL_KEYS = (*_PLOT_SETTINGS_COMMON_KEYS,)
_PLOT_SETTINGS_ORIENTATION_KEYS = (
    "component",
    "angle",
    "heatmap_vmin",
    "heatmap_vmax",
    "heatmap_cmap",
    "heatmap_normalize",
    "heatmap_colorbar_enabled",
    "heatmap_colorbar_label",
    "heatmap_colorbar_label_size",
    "heatmap_colorbar_tick_size",
    "heatmap_colorbar_position",
    "heatmap_colorbar_pad",
    "heatmap_colorbar_shrink",
    "heatmap_colorbar_aspect",
    "y_bin_width",
    "y_bin_reducer",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_TABLE_KEYS = (
    "kind",
    "group",
    "x",
    "y",
    "bins",
    "file_labels",
    *_PLOT_SETTINGS_COMMON_KEYS,
)


def _project_pyproject_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pyproject.toml"


def _read_project_author(default: str = "Unknown") -> str:
    path = _project_pyproject_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        match = _PROJECT_AUTHOR_LINE.match(line.strip())
        if match:
            return match.group(1)
    try:
        metadata = package_metadata("LiNaK")
    except PackageNotFoundError:
        return default
    for key in ("Author", "Author-email", "Maintainer", "Maintainer-email"):
        value = str(metadata[key] if key in metadata else "").strip()
        if value:
            return value
    return default


class _ProgressAwareStreamHandler(logging.StreamHandler):
    """Ensure log lines do not collide with active terminal progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        from .progress import ProgressBar

        ProgressBar.prepare_for_external_write(self.stream)
        super().emit(record)


class _LiNaKConsoleFormatter(logging.Formatter):
    """Compact, branded formatter for terminal CLI logs."""

    _COLOR_RESET = "\x1b[0m"
    _LEVEL_COLORS = {
        "DEBUG": "\x1b[38;5;244m",
        "INFO": "\x1b[38;5;39m",
        "WARNING": "\x1b[38;5;214m",
        "ERROR": "\x1b[38;5;196m",
        "CRITICAL": "\x1b[1;38;5;196m",
    }
    _LEVEL_LABELS = {
        "DEBUG": "DBG",
        "INFO": "INF",
        "WARNING": "WRN",
        "ERROR": "ERR",
        "CRITICAL": "CRT",
    }
    _BRAND_COLOR = "\x1b[1;38;5;45m"
    _TIME_COLOR = "\x1b[38;5;242m"
    _SCOPE_COLOR = "\x1b[38;5;246m"

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    @staticmethod
    def _short_name(logger_name: str) -> str:
        normalized = logger_name[6:] if logger_name.startswith("linak.") else logger_name
        return normalized.split(".")[-1]

    def format(self, record: logging.LogRecord) -> str:
        brand = "LiNaK"
        timestamp = self.formatTime(record, self.datefmt)
        level = self._LEVEL_LABELS.get(record.levelname, record.levelname[:3].upper())
        scope = self._short_name(record.name)

        if self.use_color:
            brand = f"{self._BRAND_COLOR}{brand}{self._COLOR_RESET}"
            timestamp = f"{self._TIME_COLOR}{timestamp}{self._COLOR_RESET}"
            color = self._LEVEL_COLORS.get(record.levelname, "")
            if color:
                level = f"{color}{level}{self._COLOR_RESET}"
            scope = f"{self._SCOPE_COLOR}{scope}{self._COLOR_RESET}"

        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"
        return f"{brand} {timestamp} {level} {scope}: {message}"


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure console and optional file logging."""
    linak_level = getattr(logging, level.upper(), logging.INFO)
    supports_color = (
        bool(getattr(sys.stderr, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
    )

    console_handler = _ProgressAwareStreamHandler(sys.stderr)
    console_handler.setFormatter(_LiNaKConsoleFormatter(use_color=supports_color))
    handlers: list[logging.Handler] = [console_handler]

    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="LiNaK %(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.WARNING,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("linak").setLevel(linak_level)


def _add_dry_run_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help=(
            "Preview planned actions and resolved output paths without reading/writing "
            "trajectory data or running heavy analysis."
        ),
    )


def _format_cli_invocation(argv: list[str]) -> str:
    if not argv:
        return "linak"
    return "linak " + " ".join(shlex.quote(token) for token in argv)


def _command_scope(args: argparse.Namespace) -> str:
    parts: list[str] = []
    for key in ("command", "plot_command", "compute_command", "apply_command", "csv_command"):
        value = getattr(args, key, None)
        if value:
            parts.append(value)
    return " ".join(parts) if parts else "unknown"


def _log_run_banner(args: argparse.Namespace, argv: list[str]) -> None:
    run_command = _format_cli_invocation(argv)
    LOGGER.info(
        "Session start | command=%s | mode=%s | args=%d",
        _command_scope(args),
        "dry-run" if getattr(args, "dry_run", False) else "execute",
        len(argv),
    )
    LOGGER.debug("Run command (full): %s", run_command)


def _log_dry_run_plan(title: str, lines: list[str]) -> None:
    LOGGER.info("Dry-run plan for %s:", title)
    for line in lines:
        _log_wrapped_info(f"  - {line}")


def _log_wrapped_info(message: str, *, width: int | None = None) -> None:
    if width is None:
        terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
        width = max(72, min(140, terminal_width - 20))

    wrapped = textwrap.wrap(
        message,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped:
        LOGGER.info("")
        return
    for chunk in wrapped:
        LOGGER.info("%s", chunk)


def _compact_path_for_log(path: str | Path, *, max_chars: int = 36) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    return "..." + text[-(max_chars - 3) :]


def _display_path(path: str | Path) -> str:
    """Return a human-friendly path for log messages, preferring relative form."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return str(path)
    abs_str = str(path)
    return rel if len(rel) <= len(abs_str) else abs_str


def _describe_cell_resolution(
    cell_arg: tuple[float, float, float] | None, input_path: str | None
) -> str:
    if cell_arg is not None:
        return f"explicit --cell {cell_arg[0]:.6g} {cell_arg[1]:.6g} {cell_arg[2]:.6g}"
    if input_path is not None:
        return f"simulation input metadata from {Path(input_path).expanduser()}"
    return "automatic trajectory/input discovery"


def _preview_resolve_cell_without_trajectory_read(
    trajectory: str | Path,
    *,
    cell: tuple[float, float, float] | None,
    input_path: str | None,
) -> tuple[tuple[float, float, float] | None, str]:
    """Resolve cell for dry-run using explicit, input, or auto-detected sources."""
    from .pbc import extract_cell_from_simulation_input, find_unique_simulation_input

    if cell is not None:
        return cell, "explicit --cell"

    if input_path is not None:
        resolved_input = Path(input_path).expanduser().resolve()
        return extract_cell_from_simulation_input(
            resolved_input
        ), f"explicit --input ({resolved_input})"

    trajectory_path = Path(trajectory).expanduser().resolve()

    try:
        auto_input = find_unique_simulation_input(trajectory_path.parent)
        return extract_cell_from_simulation_input(auto_input), f"auto-detected ({auto_input})"
    except (FileNotFoundError, ValueError):
        pass

    return None, "unresolved from input sources"


def _preview_resolve_msd_timestep_without_trajectory_read(
    trajectory: str | Path,
    *,
    timestep_fs: float | None,
    input_path: str | None,
) -> tuple[float, str]:
    """Resolve MSD timestep for dry-run without loading trajectory frames."""
    from .pbc import extract_frame_timestep_fs_from_simulation_input, find_unique_simulation_input

    if timestep_fs is not None:
        return float(timestep_fs), "explicit --timestep-fs"

    if input_path is not None:
        resolved_input = Path(input_path).expanduser().resolve()
        frame_timestep_fs, _, _ = extract_frame_timestep_fs_from_simulation_input(resolved_input)
        return frame_timestep_fs, f"explicit --input ({resolved_input})"

    trajectory_path = Path(trajectory).expanduser().resolve()

    try:
        auto_input = find_unique_simulation_input(trajectory_path.parent)
        frame_timestep_fs, _, _ = extract_frame_timestep_fs_from_simulation_input(auto_input)
        return frame_timestep_fs, f"auto-detected ({auto_input})"
    except (FileNotFoundError, ValueError):
        pass

    return 1.0, "fallback default"


def _format_cell_values(cell: tuple[float, float, float]) -> str:
    return f"{cell[0]:.6g} {cell[1]:.6g} {cell[2]:.6g} Angstrom"


def _describe_cell_resolution_preview(
    trajectory: str | Path,
    *,
    cell: tuple[float, float, float] | None,
    input_path: str | None,
    include_trajectory_fallback_note: bool = True,
) -> str:
    resolved_cell, cell_source = _preview_resolve_cell_without_trajectory_read(
        trajectory,
        cell=cell,
        input_path=input_path,
    )
    if resolved_cell is not None:
        return f"resolved {_format_cell_values(resolved_cell)} ({cell_source})"
    if include_trajectory_fallback_note:
        return (
            "unresolved from input sources; execution may still use "
            "trajectory-embedded periodic cell after loading frames"
        )
    return "unresolved from input sources"


def _summarize_source_resolution_previews(previews: list[str], *, limit: int = 3) -> str:
    if len(previews) <= limit:
        return "; ".join(previews)
    head = "; ".join(previews[:limit])
    return f"{head}; ... (+{len(previews) - limit} more)"


def _summarize_sources(sources: list[str], *, limit: int = 4) -> str:
    if len(sources) <= limit:
        return ", ".join(sources)
    preview = ", ".join(sources[:limit])
    return f"{preview}, ... (+{len(sources) - limit} more)"


def _sanitize_token(value: str) -> str:
    """Convert free text into a deterministic filename-safe token."""
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    token = token.strip("_")
    return token or "all"


def _linak_output_dir_for_parent(parent: Path) -> Path:
    if parent.name.lower() == _LINAK_OUTPUT_DIRNAME.lower():
        return parent
    return parent / _LINAK_OUTPUT_DIRNAME


def _linak_output_dir_for_source(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    return _linak_output_dir_for_parent(source_path.parent)


def _linak_output_dir_for_sources(sources: Sequence[str | Path]) -> Path:
    resolved_sources = [Path(source).expanduser().resolve() for source in sources]
    if not resolved_sources:
        return _linak_output_dir_for_parent(Path.cwd())
    if len(resolved_sources) == 1:
        return _linak_output_dir_for_parent(resolved_sources[0].parent)
    return _linak_output_dir_for_parent(Path.cwd())


def _resolve_non_overwriting_hdf5_path(path: str | Path) -> Path:
    from .storage.hdf5_utils import resolve_hdf5_output_path

    resolved = resolve_hdf5_output_path(path)
    return _unique_path_with_numeric_suffix(resolved) if resolved.exists() else resolved


def _output_request_looks_like_directory(value: str | Path) -> bool:
    text = str(value).strip()
    return text.endswith(("/", "\\"))


def _resolve_single_analysis_hdf5_output_path(
    base_output: str | None,
    default_output: str | Path,
) -> Path:
    if base_output is None:
        return _resolve_non_overwriting_hdf5_path(default_output)

    base_path = Path(base_output).expanduser()
    default_path = Path(default_output).expanduser()
    if _output_request_looks_like_directory(base_output) or (
        base_path.exists() and base_path.is_dir()
    ):
        return _resolve_non_overwriting_hdf5_path(base_path / default_path.name)

    if not base_path.suffix:
        base_path = base_path.with_suffix(".h5")
    return _resolve_non_overwriting_hdf5_path(base_path)


def _default_density_hdf5_output_path(source: str | Path, species: str, axis: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "trajectory"
    normalized_species = str(species).strip().lower()
    if normalized_species in {"", "all", "*"}:
        filename = f"{stem}_density_{axis.lower()}.h5"
    else:
        filename = f"{stem}_density_{_sanitize_token(species)}_{axis.lower()}.h5"
    return _linak_output_dir_for_source(source_path) / filename


def _density_hdf5_output_path(
    base_output: str | None,
    source: str | Path,
    *,
    species: str,
    axis: str,
) -> Path:
    return _resolve_single_analysis_hdf5_output_path(
        base_output,
        _default_density_hdf5_output_path(source, species, axis),
    )


def _default_orientation_hdf5_output_path(source: str | Path, axis: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "trajectory"
    filename = f"{stem}_orientation_{axis.lower()}.h5"
    return _linak_output_dir_for_source(source_path) / filename


def _orientation_hdf5_output_path(
    base_output: str | None,
    source: str | Path,
    *,
    axis: str,
) -> Path:
    return _resolve_single_analysis_hdf5_output_path(
        base_output,
        _default_orientation_hdf5_output_path(source, axis),
    )


def _normalize_source_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(path) for path in value]


def _resolve_source_arguments(
    *,
    positional: Any,
    files: Any,
    source_label: str,
    allow_multiple: bool,
) -> list[str]:
    positional_sources = _normalize_source_values(positional)
    option_sources = _normalize_source_values(files)

    if positional_sources and option_sources:
        raise ValueError("Use either positional SOURCE arguments or -f/--files, not both.")

    sources = option_sources or positional_sources
    if not sources:
        if allow_multiple:
            raise ValueError(f"Provide at least one {source_label} via SOURCE or -f/--files.")
        raise ValueError(f"Provide one {source_label} via SOURCE or -f/--files.")

    if len(positional_sources) > 1:
        raise ValueError("Use -f/--files when passing multiple input files.")

    if not allow_multiple and len(sources) != 1:
        raise ValueError(f"This command accepts exactly one {source_label}.")

    return sources


def _resolve_single_source_argument(
    args: argparse.Namespace,
    *,
    positional_attr: str,
    files_attr: str = "files",
    source_label: str,
) -> str:
    sources = _resolve_source_arguments(
        positional=getattr(args, positional_attr, None),
        files=getattr(args, files_attr, None),
        source_label=source_label,
        allow_multiple=False,
    )
    source = sources[0]
    setattr(args, positional_attr, source)
    return source


def _validate_hdf5_only_sources(sources: list[str], *, command_name: str) -> None:
    non_hdf5 = [source for source in sources if not _is_hdf5_source(source)]
    if not non_hdf5:
        return
    raise ValueError(
        f"{command_name} only accepts HDF5 input (.h5/.hdf5). "
        "Use `linak compute ...` to generate HDF5 from trajectories first. "
        f"Non-HDF5 source(s): {_summarize_sources(non_hdf5)}"
    )


def _validate_csv_only_sources(sources: list[str], *, command_name: str) -> None:
    non_hdf5 = [source for source in sources if not _is_hdf5_source(source)]
    if not non_hdf5:
        return
    raise ValueError(
        f"{command_name} only accepts HDF5 input (.h5/.hdf5). "
        f"Non-HDF5 source(s): {_summarize_sources(non_hdf5)}"
    )


def _resolve_plot_sources(args: argparse.Namespace) -> list[str]:
    return _resolve_source_arguments(
        positional=getattr(args, "source", None),
        files=getattr(args, "files", None),
        source_label="input file",
        allow_multiple=True,
    )


def _resolve_csv_plot_sources(args: argparse.Namespace) -> list[str]:
    sources = _resolve_source_arguments(
        positional=getattr(args, "source", None),
        files=getattr(args, "files", None),
        source_label="HDF5 input file",
        allow_multiple=True,
    )
    _validate_csv_only_sources(sources, command_name=f"linak {_TABULAR_COMMAND} plot")
    return sources


def _default_msd_hdf5_output_path(source: str | Path, species: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "trajectory"
    return _linak_output_dir_for_source(source_path) / f"{stem}_msd_{_sanitize_token(species)}.h5"


def _default_position_hdf5_output_path(source: str | Path, species: str, axis: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "trajectory"
    return _linak_output_dir_for_source(source_path) / (
        f"{stem}_position_{_sanitize_token(species)}_{axis.lower()}.h5"
    )


def _position_hdf5_output_paths(
    base_output: str | None,
    source: str | Path,
    profiles: list[Any],
    *,
    axis: str,
) -> list[Path]:
    if not profiles:
        return []

    if base_output is None:
        paths = [
            _default_position_hdf5_output_path(source, profile.species, axis)
            for profile in profiles
        ]
        return [_resolve_non_overwriting_hdf5_path(path) for path in paths]

    base_path = Path(base_output).expanduser()
    if len(profiles) == 1:
        default_path = _default_position_hdf5_output_path(source, profiles[0].species, axis)
        if _output_request_looks_like_directory(base_output) or (
            base_path.exists() and base_path.is_dir()
        ):
            paths = [base_path / default_path.name]
        else:
            paths = [base_path if base_path.suffix else base_path.with_suffix(".h5")]
        return [_resolve_non_overwriting_hdf5_path(path) for path in paths]

    if base_path.suffix.lower() in {".h5", ".hdf5"}:
        paths = [
            base_path.with_name(
                f"{base_path.stem}_{_sanitize_token(profile.species)}{base_path.suffix}"
            )
            for profile in profiles
        ]
        return [_resolve_non_overwriting_hdf5_path(path) for path in paths]

    base_path.mkdir(parents=True, exist_ok=True)
    paths = [
        base_path / f"position_{_sanitize_token(profile.species)}_{axis.lower()}.h5"
        for profile in profiles
    ]
    return [_resolve_non_overwriting_hdf5_path(path) for path in paths]


def _default_rdf_hdf5_output_path(source: str | Path, species_a: str, species_b: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "trajectory"
    return _linak_output_dir_for_source(source_path) / (
        f"{stem}_rdf_{_sanitize_token(species_a)}_{_sanitize_token(species_b)}.h5"
    )


def _default_coordination_hdf5_output_path(
    source: str | Path,
    species_a: str,
    species_b: str,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "trajectory"
    return _linak_output_dir_for_source(source_path) / (
        f"{stem}_coordination_{_sanitize_token(species_a)}_{_sanitize_token(species_b)}.h5"
    )


def _default_potential_hdf5_output_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or source_path.name or "source"
    for suffix in "-v_hartree-1_0":
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)] or "source"
            break
    return _linak_output_dir_for_source(source_path) / f"{stem}_potential.h5"


def _default_potential_hdf5_output_for_sources(sources: list[str]) -> Path:
    if len(sources) == 1:
        return _default_potential_hdf5_output_path(sources[0])
    return _linak_output_dir_for_sources(sources) / "linak_potential.h5"


def _default_combined_analysis_hdf5_path(sources: list[str], *, analysis: str) -> Path:
    return _linak_output_dir_for_sources(sources) / f"linak_{analysis}_combined.h5"


def _unique_path_with_numeric_suffix(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find available output filename for '{path}'.")


def _default_plot_output_path(source: str | Path, analysis_name: str) -> Path:
    stem = Path(source).stem or "profile"
    return Path.cwd() / f"{stem}_{analysis_name.lower()}.png"


def _default_pbc_output_path(trajectory: str | Path) -> Path:
    input_path = Path(trajectory).expanduser().resolve()
    if input_path.suffix.lower() in {".dump", ".lmp"}:
        output_name = f"{input_path.stem}_pbc.xyz"
        return input_path.with_name(output_name)
    if input_path.suffix:
        output_name = f"{input_path.stem}_pbc{input_path.suffix}"
    else:
        output_name = f"{input_path.name}_pbc"
    return input_path.with_name(output_name)


def _is_csv_source(path: str | Path) -> bool:
    return _is_hdf5_source(path)


def _is_hdf5_source(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".h5", ".hdf5"}


def _parse_backend(value: str) -> str:
    """Argparse type wrapper for backend normalization with useful errors."""
    from .plot.plotting import normalize_backend_name

    try:
        return normalize_backend_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _normalize_cell_args(args: argparse.Namespace) -> tuple[float, float, float] | None:
    return tuple(args.cell) if args.cell is not None else None


def _frame_has_usable_periodic_cell(frame: Atoms) -> bool:
    if not all(bool(value) for value in frame.get_pbc()):
        return False
    lengths = frame.cell.lengths()
    if any(length <= 0.0 for length in lengths):
        return False
    return abs(float(frame.get_volume())) > 0.0


def _frames_have_usable_periodic_cell(frames: list[Atoms]) -> bool:
    return bool(frames) and all(_frame_has_usable_periodic_cell(frame) for frame in frames)


def _cell_lengths_from_frame(frame: Atoms) -> tuple[float, float, float]:
    raw_lengths = frame.cell.lengths()
    lengths = (
        float(raw_lengths[0]),
        float(raw_lengths[1]),
        float(raw_lengths[2]),
    )
    if any(value <= 0.0 for value in lengths):
        raise ValueError("Trajectory frame has non-positive cell length(s).")
    return lengths


def _flatten_profiles_by_source(source_profiles: list[tuple[str, list[Any]]]) -> list[Any]:
    flattened: list[Any] = []
    for _, profiles in source_profiles:
        flattened.extend(profiles)
    return flattened


def _metadata_source_label(metadata: dict[str, Any], *, fallback_source: str) -> str:
    origin_path = str(metadata.get("origin_hdf5_path") or "").strip()
    if origin_path:
        return Path(origin_path).name or origin_path
    return Path(fallback_source).name or fallback_source


def _should_prefix_combined_source_labels(
    *,
    sources: list[str],
    metadata_items: list[dict[str, Any]],
) -> bool:
    if len(sources) > 1:
        return True
    source_labels = {
        _metadata_source_label(metadata, fallback_source=sources[0]) for metadata in metadata_items
    }
    return len(source_labels) > 1


def _position_series_labels_for_profile(profile: Any) -> list[str]:
    atom_indices = getattr(profile, "atom_indices", None)
    species = str(getattr(profile, "species", "UNKNOWN"))
    if atom_indices is None:
        return [species]
    labels: list[str] = []
    for raw_index in list(atom_indices):
        try:
            labels.append(f"{species}[{int(raw_index)}]")
        except (TypeError, ValueError):
            labels.append(f"{species}[{raw_index}]")
    return labels


def _coordination_series_labels_for_profile(profile: Any) -> list[str]:
    atom_indices = getattr(profile, "atom_indices", None)
    species = str(getattr(profile, "species_a", "UNKNOWN"))
    if species == "ALL":
        species = "A"
    if atom_indices is None:
        return [species]
    labels: list[str] = []
    for raw_index in list(atom_indices):
        try:
            labels.append(f"{species}[{int(raw_index)}]")
        except (TypeError, ValueError):
            labels.append(f"{species}[{raw_index}]")
    return labels


def _split_position_profile_into_atom_series(profile: Any) -> list[Any]:
    atom_indices = getattr(profile, "atom_indices", None)
    if atom_indices is None:
        return [profile]
    atom_index_array = list(atom_indices)
    if len(atom_index_array) <= 1:
        return [profile]

    split_profiles: list[Any] = []
    for column, raw_atom_index in enumerate(atom_index_array):
        split_profiles.append(
            replace(
                profile,
                atom_indices=np.asarray([int(raw_atom_index)], dtype=int),
                x=np.asarray(profile.x[:, [column]], dtype=float),
                y=np.asarray(profile.y[:, [column]], dtype=float),
                z=np.asarray(profile.z[:, [column]], dtype=float),
                distance_to_surface=np.asarray(
                    profile.distance_to_surface[:, [column]], dtype=float
                ),
                n_atoms=1,
            )
        )
    return split_profiles


def _split_coordination_profile_into_atom_series(profile: Any) -> list[Any]:
    atom_indices = getattr(profile, "atom_indices", None)
    if atom_indices is None:
        return [profile]
    atom_index_array = list(atom_indices)
    if len(atom_index_array) <= 1:
        return [profile]

    split_profiles: list[Any] = []
    for column, raw_atom_index in enumerate(atom_index_array):
        split_profiles.append(
            replace(
                profile,
                atom_indices=np.asarray([int(raw_atom_index)], dtype=int),
                distance_to_surface=np.asarray(
                    profile.distance_to_surface[:, [column]], dtype=float
                ),
                coordination_number=np.asarray(
                    profile.coordination_number[:, [column]], dtype=float
                ),
                n_atoms=1,
            )
        )
    return split_profiles


def _extract_position_profile_atom_series(profile: Any, atom_index: int) -> Any:
    raw_atom_indices = getattr(profile, "atom_indices", None)
    atom_indices = [] if raw_atom_indices is None else list(raw_atom_indices)
    for column, raw_atom_index in enumerate(atom_indices):
        try:
            resolved_atom_index = int(raw_atom_index)
        except (TypeError, ValueError):
            continue
        if resolved_atom_index != int(atom_index):
            continue
        return replace(
            profile,
            atom_indices=np.asarray([resolved_atom_index], dtype=int),
            x=np.asarray(profile.x[:, [column]], dtype=float),
            y=np.asarray(profile.y[:, [column]], dtype=float),
            z=np.asarray(profile.z[:, [column]], dtype=float),
            distance_to_surface=np.asarray(profile.distance_to_surface[:, [column]], dtype=float),
            n_atoms=1,
        )
    raise ValueError(f"Position profile does not contain atom index {atom_index}.")


def _extract_coordination_profile_atom_series(profile: Any, atom_index: int) -> Any:
    raw_atom_indices = getattr(profile, "atom_indices", None)
    atom_indices = [] if raw_atom_indices is None else list(raw_atom_indices)
    for column, raw_atom_index in enumerate(atom_indices):
        try:
            resolved_atom_index = int(raw_atom_index)
        except (TypeError, ValueError):
            continue
        if resolved_atom_index != int(atom_index):
            continue
        return replace(
            profile,
            atom_indices=np.asarray([resolved_atom_index], dtype=int),
            distance_to_surface=np.asarray(profile.distance_to_surface[:, [column]], dtype=float),
            coordination_number=np.asarray(profile.coordination_number[:, [column]], dtype=float),
            n_atoms=1,
        )
    raise ValueError(f"Coordination profile does not contain atom index {atom_index}.")


def _ordered_common_items_by_source(
    items_by_source: Sequence[Sequence[tuple[str, ...]]],
) -> list[tuple[str, ...]]:
    if not items_by_source:
        return []
    common_items = set(items_by_source[0])
    for source_items in items_by_source[1:]:
        common_items &= set(source_items)
    ordered: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items_by_source[0]:
        if item in common_items and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _build_rdf_profile_filter_options(
    raw_payloads_by_source: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    pairs_by_source: list[list[tuple[str, str]]] = []
    for _source, payloads in raw_payloads_by_source:
        source_pairs: list[tuple[str, str]] = []
        for payload in payloads:
            metadata = payload.get("metadata", {})
            species_a = str(metadata.get("species_a", "")).strip()
            species_b = str(metadata.get("species_b", "")).strip() or species_a
            if not species_a:
                continue
            source_pairs.append((species_a, species_b))
        pairs_by_source.append(source_pairs)

    common_pairs = _ordered_common_items_by_source(pairs_by_source)
    species_a_options: list[str] = []
    species_b_by_species_a: dict[str, list[str]] = {"": []}
    for species_a, species_b in common_pairs:
        if species_a not in species_a_options:
            species_a_options.append(species_a)
        global_species_b = species_b_by_species_a[""]
        if species_b not in global_species_b:
            global_species_b.append(species_b)
        species_b_by_species_a.setdefault(species_a, [])
        if species_b not in species_b_by_species_a[species_a]:
            species_b_by_species_a[species_a].append(species_b)

    return {
        "species_a": species_a_options,
        "species_b_by_species_a": species_b_by_species_a,
    }


def _build_coordination_profile_filter_options(
    raw_payloads_by_source: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    triples_by_source: list[list[tuple[str, str, str]]] = []
    for _source, payloads in raw_payloads_by_source:
        source_triples: list[tuple[str, str, str]] = []
        for payload in payloads:
            metadata = payload.get("metadata", {})
            species_a = str(metadata.get("species_a", "")).strip()
            species_b = str(metadata.get("species_b", "")).strip() or species_a
            axis = str(metadata.get("axis", "z")).strip().lower() or "z"
            if not species_a:
                continue
            source_triples.append((species_a, species_b, axis))
        triples_by_source.append(source_triples)

    common_triples = _ordered_common_items_by_source(triples_by_source)
    species_a_options: list[str] = []
    species_b_by_species_a: dict[str, list[str]] = {"": []}
    axes: list[str] = []
    axes_by_species_pair: dict[str, dict[str, list[str]]] = {"": {"": []}}
    for species_a, species_b, axis in common_triples:
        if species_a not in species_a_options:
            species_a_options.append(species_a)
        global_species_b = species_b_by_species_a[""]
        if species_b not in global_species_b:
            global_species_b.append(species_b)
        species_b_by_species_a.setdefault(species_a, [])
        if species_b not in species_b_by_species_a[species_a]:
            species_b_by_species_a[species_a].append(species_b)
        if axis not in axes:
            axes.append(axis)
        axes_by_species_pair.setdefault("", {}).setdefault("", [])
        if axis not in axes_by_species_pair[""][""]:
            axes_by_species_pair[""][""].append(axis)
        axes_by_species_pair.setdefault(species_a, {}).setdefault(species_b, [])
        if axis not in axes_by_species_pair[species_a][species_b]:
            axes_by_species_pair[species_a][species_b].append(axis)

    return {
        "species_a": species_a_options,
        "species_b_by_species_a": species_b_by_species_a,
        "axes": axes,
        "axes_by_species_pair": axes_by_species_pair,
    }


def _load_density_plot_profiles(
    *,
    sources: list[str],
    species: str | None,
    axis: str | None,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.density import load_density_profiles

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="density",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(payload.get("metadata", {}))
            for _source, source_payloads in raw_payloads_by_source
            for payload in source_payloads
        ],
    )
    profiles_by_source: list[tuple[str, list[Any]]] = []
    for source in sources:
        profiles = load_density_profiles(source, axis=axis, species=species)
        profiles_by_source.append((source, profiles))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("Density profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                rendered_species = f"{source_label}:{profile.species}"
                source_labels.append(rendered_species)
                source_ids.append(
                    _profile_uid_from_payload(
                        payload, fallback_prefix="density", index=profile_index
                    )
                )
                source_origins.append(
                    str(payload.get("metadata", {}).get("origin_hdf5_path") or source)
                )
                plot_profiles.append(replace(profile, species=rendered_species))
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        plot_profiles.extend(flattened)
        fallback_labels_by_source.append([profile.species for profile in flattened])
        raw_payloads = raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("Density profile metadata does not match loaded profiles.")
        series_id_segments_by_source.append(
            [
                _profile_uid_from_payload(payload, fallback_prefix="density", index=profile_index)
                for profile_index, payload in enumerate(raw_payloads)
            ]
        )
        origin_path_segments_by_source.append(
            [
                str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])
                for payload in raw_payloads
            ]
        )

    return (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    )


def _load_orientation_plot_profiles(
    *,
    sources: list[str],
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.orientation import load_orientation_profiles

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="orientation",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(payload.get("metadata", {}))
            for _source, source_payloads in raw_payloads_by_source
            for payload in source_payloads
        ],
    )
    profiles_by_source: list[tuple[str, list[Any]]] = []
    for source in sources:
        profiles = load_orientation_profiles(source)
        profiles_by_source.append((source, profiles))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("Orientation profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                rendered_label = f"{source_label}:orientation"
                source_labels.append(rendered_label)
                source_ids.append(
                    _profile_uid_from_payload(
                        payload, fallback_prefix="orientation", index=profile_index
                    )
                )
                source_origins.append(
                    str(payload.get("metadata", {}).get("origin_hdf5_path") or source)
                )
                plot_profiles.append(profile)
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        plot_profiles.extend(flattened)
        fallback_labels_by_source.append([f"orientation [{i}]" for i in range(len(flattened))])
        raw_payloads = raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("Orientation profile metadata does not match loaded profiles.")
        series_id_segments_by_source.append(
            [
                _profile_uid_from_payload(
                    payload, fallback_prefix="orientation", index=profile_index
                )
                for profile_index, payload in enumerate(raw_payloads)
            ]
        )
        origin_path_segments_by_source.append(
            [
                str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])
                for payload in raw_payloads
            ]
        )

    return (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    )


def _load_msd_plot_profiles(
    *,
    sources: list[str],
    species: str | None,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.msd import load_msd_profiles

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="msd",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(payload.get("metadata", {}))
            for _source, source_payloads in raw_payloads_by_source
            for payload in source_payloads
        ],
    )
    profiles_by_source: list[tuple[str, list[Any]]] = []
    for source in sources:
        profiles = load_msd_profiles(source, species=species)
        profiles_by_source.append((source, profiles))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("MSD profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                rendered_species = f"{source_label}:{profile.species}"
                source_labels.append(rendered_species)
                source_ids.append(
                    _profile_uid_from_payload(payload, fallback_prefix="msd", index=profile_index)
                )
                source_origins.append(
                    str(payload.get("metadata", {}).get("origin_hdf5_path") or source)
                )
                plot_profiles.append(replace(profile, species=rendered_species))
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        plot_profiles.extend(flattened)
        fallback_labels_by_source.append([profile.species for profile in flattened])
        raw_payloads = raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("MSD profile metadata does not match loaded profiles.")
        series_id_segments_by_source.append(
            [
                _profile_uid_from_payload(payload, fallback_prefix="msd", index=profile_index)
                for profile_index, payload in enumerate(raw_payloads)
            ]
        )
        origin_path_segments_by_source.append(
            [
                str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])
                for payload in raw_payloads
            ]
        )

    return (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    )


def _load_rdf_plot_profiles(
    *,
    sources: list[str],
    species_a: str | None,
    species_b: str | None,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.rdf import load_rdf_profiles, _normalize_species as _normalize_rdf_species

    resolved_species_b = species_b if species_b is not None else species_a
    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="rdf",
    )
    wanted_species_a = (
        None
        if species_a is None or not str(species_a).strip()
        else _normalize_rdf_species(species_a)
    )
    wanted_species_b = (
        None
        if resolved_species_b is None or not str(resolved_species_b).strip()
        else _normalize_rdf_species(resolved_species_b)
    )
    filtered_raw_payloads_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source, payloads in raw_payloads_by_source:
        filtered_payloads: list[dict[str, Any]] = []
        for payload in payloads:
            metadata = payload.get("metadata", {})
            meta_species_a = str(metadata.get("species_a", "")).strip() or "UNKNOWN"
            meta_species_b = str(metadata.get("species_b", "")).strip() or meta_species_a
            if (
                wanted_species_a is not None
                and _normalize_rdf_species(meta_species_a) != wanted_species_a
            ):
                continue
            if (
                wanted_species_b is not None
                and _normalize_rdf_species(meta_species_b) != wanted_species_b
            ):
                continue
            filtered_payloads.append(payload)
        filtered_raw_payloads_by_source.append((source, filtered_payloads))
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(payload.get("metadata", {}))
            for _source, source_payloads in filtered_raw_payloads_by_source
            for payload in source_payloads
        ],
    )
    profiles_by_source: list[tuple[str, list[Any]]] = []
    for source in sources:
        profiles = load_rdf_profiles(source, species_a=species_a, species_b=resolved_species_b)
        profiles_by_source.append((source, profiles))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = filtered_raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("RDF profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                rendered_species_a = f"{source_label}:{profile.species_a}"
                source_labels.append(f"{rendered_species_a}-{profile.species_b}")
                source_ids.append(
                    _profile_uid_from_payload(payload, fallback_prefix="rdf", index=profile_index)
                )
                source_origins.append(
                    str(payload.get("metadata", {}).get("origin_hdf5_path") or source)
                )
                plot_profiles.append(replace(profile, species_a=rendered_species_a))
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        plot_profiles.extend(flattened)
        fallback_labels_by_source.append(
            [f"{profile.species_a}-{profile.species_b}" for profile in flattened]
        )
        raw_payloads = filtered_raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("RDF profile metadata does not match loaded profiles.")
        series_id_segments_by_source.append(
            [
                _profile_uid_from_payload(payload, fallback_prefix="rdf", index=profile_index)
                for profile_index, payload in enumerate(raw_payloads)
            ]
        )
        origin_path_segments_by_source.append(
            [
                str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])
                for payload in raw_payloads
            ]
        )

    return (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    )


def _load_position_plot_profiles(
    *,
    sources: list[str],
    species: str | None,
    axis: str | None,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.position import load_position_profiles

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="position",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(payload.get("metadata", {}))
            for _source, source_payloads in raw_payloads_by_source
            for payload in source_payloads
        ],
    )
    profiles_by_source: list[tuple[str, list[Any]]] = []
    for source in sources:
        profiles = load_position_profiles(source, species=species, axis=axis)
        profiles_by_source.append((source, profiles))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("Position profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                profile_uid = _profile_uid_from_payload(
                    payload,
                    fallback_prefix="position",
                    index=profile_index,
                )
                rendered_species = f"{source_label}:{profile.species}"
                rendered_profile = replace(profile, species=rendered_species)
                atom_profiles = _split_position_profile_into_atom_series(rendered_profile)
                plot_profiles.extend(atom_profiles)
                source_labels.extend(
                    [
                        label
                        for item in atom_profiles
                        for label in _position_series_labels_for_profile(item)
                    ]
                )
                source_ids.extend(
                    [
                        f"{profile_uid}:atom:{int(atom_profile.atom_indices[0])}"
                        for atom_profile in atom_profiles
                    ]
                )
                source_origins.extend(
                    [str(payload.get("metadata", {}).get("origin_hdf5_path") or source)]
                    * len(atom_profiles)
                )
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        flattened_source_labels: list[str] = []
        flattened_source_ids: list[str] = []
        flattened_source_origins: list[str] = []
        raw_payloads = raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("Position profile metadata does not match loaded profiles.")
        for profile_index, profile in enumerate(flattened):
            payload = raw_payloads[profile_index]
            profile_uid = _profile_uid_from_payload(
                payload,
                fallback_prefix="position",
                index=profile_index,
            )
            atom_profiles = _split_position_profile_into_atom_series(profile)
            plot_profiles.extend(atom_profiles)
            flattened_source_labels.extend(
                [
                    label
                    for item in atom_profiles
                    for label in _position_series_labels_for_profile(item)
                ]
            )
            flattened_source_ids.extend(
                [
                    f"{profile_uid}:atom:{int(atom_profile.atom_indices[0])}"
                    for atom_profile in atom_profiles
                ]
            )
            flattened_source_origins.extend(
                [str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])]
                * len(atom_profiles)
            )
        fallback_labels_by_source.append(flattened_source_labels)
        series_id_segments_by_source.append(flattened_source_ids)
        origin_path_segments_by_source.append(flattened_source_origins)

    return (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    )


def _load_coordination_plot_profiles(
    *,
    sources: list[str],
    species_a: str | None,
    species_b: str | None,
    axis: str | None,
    component: str,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.coordination import (
        _normalize_axis as _normalize_coordination_axis,
        _normalize_species as _normalize_coordination_species,
        load_coordination_profiles,
    )

    normalized_component = str(component).strip().lower().replace("_", "-")
    resolved_species_b = species_b if species_b is not None else species_a
    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="coordination",
    )
    wanted_species_a = (
        None
        if species_a is None or not str(species_a).strip()
        else _normalize_coordination_species(species_a)
    )
    wanted_species_b = (
        None
        if resolved_species_b is None or not str(resolved_species_b).strip()
        else _normalize_coordination_species(resolved_species_b)
    )
    wanted_axis = (
        None if axis is None or not str(axis).strip() else _normalize_coordination_axis(axis)
    )
    filtered_raw_payloads_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source, payloads in raw_payloads_by_source:
        filtered_payloads: list[dict[str, Any]] = []
        for payload in payloads:
            metadata = payload.get("metadata", {})
            meta_species_a = str(metadata.get("species_a", "")).strip() or "UNKNOWN"
            meta_species_b = str(metadata.get("species_b", "")).strip() or meta_species_a
            meta_axis = str(metadata.get("axis", "z")).strip().lower() or "z"
            if (
                wanted_species_a is not None
                and _normalize_coordination_species(meta_species_a) != wanted_species_a
            ):
                continue
            if (
                wanted_species_b is not None
                and _normalize_coordination_species(meta_species_b) != wanted_species_b
            ):
                continue
            if wanted_axis is not None and _normalize_coordination_axis(meta_axis) != wanted_axis:
                continue
            filtered_payloads.append(payload)
        filtered_raw_payloads_by_source.append((source, filtered_payloads))
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(payload.get("metadata", {}))
            for _source, source_payloads in filtered_raw_payloads_by_source
            for payload in source_payloads
        ],
    )
    profiles_by_source: list[tuple[str, list[Any]]] = []
    for source in sources:
        profiles = load_coordination_profiles(
            source,
            species_a=species_a,
            species_b=resolved_species_b,
            axis=axis,
        )
        profiles_by_source.append((source, profiles))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = filtered_raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("Coordination profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                profile_uid = _profile_uid_from_payload(
                    payload,
                    fallback_prefix="coordination",
                    index=profile_index,
                )
                rendered_species_a = f"{source_label}:{profile.species_a}"
                rendered_profile = replace(profile, species_a=rendered_species_a)
                if normalized_component == "distance":
                    plot_profiles.append(rendered_profile)
                    source_labels.append(f"{rendered_species_a}-{profile.species_b}")
                    source_ids.append(profile_uid)
                    source_origins.append(
                        str(payload.get("metadata", {}).get("origin_hdf5_path") or source)
                    )
                else:
                    atom_profiles = _split_coordination_profile_into_atom_series(rendered_profile)
                    plot_profiles.extend(atom_profiles)
                    source_labels.extend(
                        [
                            label
                            for item in atom_profiles
                            for label in _coordination_series_labels_for_profile(item)
                        ]
                    )
                    source_ids.extend(
                        [
                            f"{profile_uid}:atom:{int(atom_profile.atom_indices[0])}"
                            for atom_profile in atom_profiles
                        ]
                    )
                    source_origins.extend(
                        [str(payload.get("metadata", {}).get("origin_hdf5_path") or source)]
                        * len(atom_profiles)
                    )
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        flattened_source_labels: list[str] = []
        flattened_source_ids: list[str] = []
        flattened_source_origins: list[str] = []
        raw_payloads = filtered_raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("Coordination profile metadata does not match loaded profiles.")
        for profile_index, profile in enumerate(flattened):
            payload = raw_payloads[profile_index]
            profile_uid = _profile_uid_from_payload(
                payload,
                fallback_prefix="coordination",
                index=profile_index,
            )
            if normalized_component == "distance":
                plot_profiles.append(profile)
                flattened_source_labels.append(f"{profile.species_a}-{profile.species_b}")
                flattened_source_ids.append(profile_uid)
                flattened_source_origins.append(
                    str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])
                )
            else:
                atom_profiles = _split_coordination_profile_into_atom_series(profile)
                plot_profiles.extend(atom_profiles)
                flattened_source_labels.extend(
                    [
                        label
                        for item in atom_profiles
                        for label in _coordination_series_labels_for_profile(item)
                    ]
                )
                flattened_source_ids.extend(
                    [
                        f"{profile_uid}:atom:{int(atom_profile.atom_indices[0])}"
                        for atom_profile in atom_profiles
                    ]
                )
                flattened_source_origins.extend(
                    [str(payload.get("metadata", {}).get("origin_hdf5_path") or sources[0])]
                    * len(atom_profiles)
                )
        fallback_labels_by_source.append(flattened_source_labels)
        series_id_segments_by_source.append(flattened_source_ids)
        origin_path_segments_by_source.append(flattened_source_origins)

    return (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a value > 0, got {value}.")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected a value >= 0, got {value}.")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a value > 0, got {value}.")
    return parsed


def _add_csv_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        nargs="?",
        metavar="SOURCE",
        help="Input HDF5 path (legacy positional form)",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="Input HDF5 file path(s). Use -f/--files even for one file; required for multiple.",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Optional HDF5 group path to read tabular datasets from (default: auto; prefers /records).",
    )


def _add_csv_plot_source_options(parser: argparse.ArgumentParser) -> None:
    input_group = parser.add_argument_group("Input files")
    input_group.add_argument(
        "source",
        nargs="*",
        metavar="SOURCE",
        help="Input HDF5 file path(s) (legacy positional form; use -f/--files for multiple)",
    )
    input_group.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="Input HDF5 file path(s). Use -f/--files even for one file; required for multiple.",
    )
    input_group.add_argument(
        "--group",
        default=None,
        help="Optional HDF5 group path to read tabular datasets from (default: auto; prefers /records).",
    )


def _add_csv_write_options(parser: argparse.ArgumentParser) -> None:
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        help="Output HDF5 path (default: auto-generated next to input)",
    )
    output_group.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input HDF5 file in place",
    )


def _add_csv_plot_options(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(x_lim=None)
    parser.set_defaults(y_lim=None)
    render_group = parser.add_argument_group("Render and output")
    render_group.add_argument(
        "-o",
        "--output",
        help="Output image path (PNG, PDF, SVG, ...)",
    )
    render_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Show interactive plot window (default: enabled)",
    )
    render_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive plot window",
    )
    render_group.add_argument(
        "--backend",
        type=_parse_backend,
        default=DEFAULT_INTERACTIVE_BACKEND,
        metavar="BACKEND",
        help=(
            "Preferred Matplotlib backend when interactive plotting is enabled "
            f"(default: {DEFAULT_INTERACTIVE_BACKEND})"
        ),
    )
    render_group.add_argument(
        "--settings-source",
        metavar="PATH_OR_INDEX",
        default=None,
        help=(
            "When plotting multiple input HDF5 files, select which source provides persisted "
            "plot settings (default: first input). Accepts a 1-based index or one of the input paths."
        ),
    )
    render_group.add_argument(
        "--settings-profile",
        metavar="NAME",
        default=None,
        help=(
            "Optional named saved profile inside the selected plot-settings source. "
            "Defaults to that file's active saved profile."
        ),
    )

    axis_group = parser.add_argument_group("Axes and title")
    axis_group.add_argument(
        "--title",
        default=None,
        help="Optional plot title (default: inferred from file and plot type)",
    )
    _add_toggle_state_argument(
        axis_group,
        flag="title-mode",
        dest="title_visible",
        feature_name="Title display",
    )
    axis_group.add_argument("--x-label", help="Custom x-axis label")
    axis_group.add_argument("--y-label", help="Custom y-axis label")
    axis_group.add_argument(
        "--x-scale",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
        help="X-axis scale (default: linear)",
    )
    axis_group.add_argument(
        "--y-scale",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
        help="Y-axis scale (default: linear)",
    )
    axis_group.add_argument("--x-min", type=float, metavar="XMIN", help="Lower x-axis limit")
    axis_group.add_argument("--x-max", type=float, metavar="XMAX", help="Upper x-axis limit")
    axis_group.add_argument("--y-min", type=float, metavar="YMIN", help="Lower y-axis limit")
    axis_group.add_argument("--y-max", type=float, metavar="YMAX", help="Upper y-axis limit")
    axis_group.add_argument(
        "--x-ticks",
        nargs="+",
        type=float,
        metavar="XTICK",
        help="Explicit x-axis tick positions",
    )
    axis_group.add_argument(
        "--y-ticks",
        nargs="+",
        type=float,
        metavar="YTICK",
        help="Explicit y-axis tick positions",
    )
    axis_group.add_argument(
        "--x-tick-rotation",
        type=float,
        help="X-axis tick-label rotation in degrees",
    )
    axis_group.add_argument(
        "--y-tick-rotation",
        type=float,
        help="Y-axis tick-label rotation in degrees",
    )

    data_group = parser.add_argument_group("Data transforms (plot-only)")
    data_group.add_argument(
        "--x-bin-width",
        type=_positive_float,
        default=None,
        help=(
            "Optional x-bin width for display-only rebinning. "
            "This does not modify source HDF5 data."
        ),
    )
    data_group.add_argument(
        "--x-bin-reducer",
        choices=["mean", "median", "sum", "min", "max"],
        default=None,
        help="Reducer applied during x rebinning (default when set: mean).",
    )

    legend_group = parser.add_argument_group("Series labels and legend")
    legend_group.add_argument(
        "--labels",
        "--series-labels",
        dest="series_labels",
        nargs="+",
        metavar="LABEL",
        help=(
            "Custom labels for plotted series. Count must match the number of rendered series "
            "(used for legends, or box-plot tick labels)."
        ),
    )
    legend_group.add_argument(
        "--file-labels",
        nargs="+",
        metavar="LABEL",
        help="Optional labels for each input file (used when plotting multiple HDF5 files).",
    )
    _add_toggle_state_argument(
        legend_group,
        flag="legend",
        dest="legend",
        feature_name="Legend display",
    )
    legend_group.add_argument(
        "--no-legend",
        dest="legend",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    legend_group.add_argument(
        "--legend-title",
        help="Optional legend title",
    )
    legend_group.add_argument(
        "--legend-loc",
        default="best",
        help="Matplotlib legend location (default: best)",
    )

    style_group = parser.add_argument_group("Figure style")
    style_group.add_argument(
        "--figsize",
        nargs=2,
        type=_positive_float,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 7 4)",
    )
    style_group.add_argument(
        "--dpi",
        type=_positive_int,
        help="Figure DPI when saving output (default: 200)",
    )
    style_group.add_argument(
        "--font-family",
        help="Matplotlib font family (default: DejaVu Sans)",
    )
    style_group.add_argument(
        "--title-font-size",
        type=_positive_int,
        help="Title font size (default: 14)",
    )
    style_group.add_argument(
        "--label-font-size",
        type=_positive_int,
        help="Axis label font size (default: 12)",
    )
    style_group.add_argument(
        "--tick-font-size",
        type=_positive_int,
        help="Tick label font size (default: 10)",
    )
    style_group.add_argument(
        "--legend-font-size",
        type=_positive_int,
        help="Legend font size (default: 10)",
    )
    style_group.add_argument(
        "--line-width",
        type=_positive_float,
        help="Main line width (default: 2.0)",
    )
    style_group.add_argument("--line-color", help="Main line color (default: #1f77b4)")
    style_group.add_argument(
        "--line-colors",
        nargs="+",
        metavar="COLOR",
        help="Per-series line colors (count must match rendered series count).",
    )
    _add_toggle_state_argument(
        style_group,
        flag="grid",
        dest="grid",
        feature_name="Grid display",
    )
    style_group.add_argument(
        "--no-grid",
        dest="grid",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    _add_toggle_state_argument(
        style_group,
        flag="ticks",
        dest="ticks",
        feature_name="Tick display",
    )
    _add_toggle_state_argument(
        style_group,
        flag="markers",
        dest="markers",
        feature_name="Line markers",
    )
    style_group.add_argument(
        "--grid-linestyle",
        help="Grid linestyle (default: --)",
    )
    style_group.add_argument(
        "--grid-linewidth",
        type=_positive_float,
        help="Grid line width (default: 0.8)",
    )
    style_group.add_argument(
        "--grid-alpha",
        type=_non_negative_float,
        help="Grid alpha transparency (default: 0.35)",
    )


def _ensure_prompt_capable_terminal() -> None:
    if not _interactive_prompts_available():
        raise ValueError(
            "Interactive prompt unavailable in non-interactive mode. "
            "Provide explicit CLI arguments instead."
        )


def _interactive_prompts_available() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _resolve_column_tokens(raw: str, candidates: list[str]) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No column selection provided.")

    resolved: list[str] = []
    lowered = {name.lower(): name for name in candidates}
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if index < 1 or index > len(candidates):
                raise ValueError(f"Column index {index} is out of range 1..{len(candidates)}.")
            resolved.append(candidates[index - 1])
            continue
        if token in candidates:
            resolved.append(token)
            continue
        normalized = lowered.get(token.lower())
        if normalized is None:
            raise ValueError(f"Unknown column '{token}'.")
        resolved.append(normalized)

    unique: list[str] = []
    for name in resolved:
        if name not in unique:
            unique.append(name)
    return unique


def _prompt_for_columns(
    *,
    columns: list[str],
    prompt: str,
    allow_multiple: bool,
) -> list[str]:
    _ensure_prompt_capable_terminal()
    print("Available columns:")
    for index, name in enumerate(columns, start=1):
        print(f"  {index:>2}. {name}")

    while True:
        suffix = " (name/index, comma-separated)" if allow_multiple else " (name/index)"
        raw = input(f"{prompt}{suffix}: ").strip()
        try:
            resolved = _resolve_column_tokens(raw, columns)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")
            continue
        if not allow_multiple and len(resolved) != 1:
            print("Please select exactly one column.")
            continue
        return resolved


def _prompt_for_value(prompt: str, *, allowed: set[str] | None = None) -> str:
    _ensure_prompt_capable_terminal()
    while True:
        value = input(f"{prompt}: ").strip()
        if not value:
            print("A value is required.")
            continue
        if allowed is not None and value.lower() not in allowed:
            print(f"Please choose one of: {', '.join(sorted(allowed))}")
            continue
        return value


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    _ensure_prompt_capable_terminal()
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _runtime_option_was_provided(args: argparse.Namespace, setting_key: str) -> bool:
    flags = _PERSISTED_PLOT_SETTING_OPTION_FLAGS.get(setting_key)
    if not flags:
        return False
    runtime_argv = tuple(getattr(args, "_runtime_argv", ()))
    for token in runtime_argv:
        for flag in flags:
            if token == flag or token.startswith(f"{flag}="):
                return True
    return False


def _json_ready_setting(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_ready_setting(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready_setting(item) for key, item in value.items()}
    return value


def _collect_plot_settings_from_args(
    args: argparse.Namespace, *, keys: tuple[str, ...]
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key in keys:
        if hasattr(args, key):
            settings[key] = _json_ready_setting(getattr(args, key))
    return settings


def _apply_saved_plot_settings(
    *,
    args: argparse.Namespace,
    source_path: Path,
    profile_key: str,
    keys: tuple[str, ...],
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    from .plot.plot_settings import read_plot_profile

    try:
        saved = read_plot_profile(source_path, profile_key, profile_name=profile_name)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.debug("Could not read saved plot settings from '%s': %s", source_path, exc)
        return None
    if saved is None:
        return None

    for key in keys:
        if key not in saved:
            continue
        if _runtime_option_was_provided(args, key):
            continue
        setattr(args, key, deepcopy(saved[key]))
    return saved


def _set_nested_setting(settings: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    node = settings
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = deepcopy(value)


def _delete_nested_setting(settings: dict[str, Any], dotted_path: str) -> None:
    keys = dotted_path.split(".")
    node: dict[str, Any] = settings
    trail: list[tuple[dict[str, Any], str]] = []
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            return
        trail.append((node, key))
        node = child

    if keys[-1] not in node:
        return
    del node[keys[-1]]

    for parent, key in reversed(trail):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
        else:
            break


def _profile_key_from_analysis(analysis: str | None) -> str:
    normalized = (analysis or "").strip().lower()
    if normalized == "density":
        return _PLOT_PROFILE_DENSITY
    if normalized == "msd":
        return _PLOT_PROFILE_MSD
    if normalized == "rdf":
        return _PLOT_PROFILE_RDF
    if normalized == "position":
        return _PLOT_PROFILE_POSITION
    if normalized == "coordination":
        return _PLOT_PROFILE_COORDINATION
    if normalized == "potential":
        return _PLOT_PROFILE_POTENTIAL
    if normalized == "orientation":
        return _PLOT_PROFILE_ORIENTATION
    return _PLOT_PROFILE_TABLE


def _resolve_plot_profile_key(
    *,
    profile_token: str | None,
    source_path: Path,
) -> str:
    if profile_token is None or profile_token == "auto":
        from .plot.plot_settings import read_hdf5_analysis

        return _profile_key_from_analysis(read_hdf5_analysis(source_path))

    normalized = profile_token.strip().lower()
    if normalized == "density":
        return _PLOT_PROFILE_DENSITY
    if normalized == "msd":
        return _PLOT_PROFILE_MSD
    if normalized == "rdf":
        return _PLOT_PROFILE_RDF
    if normalized == "position":
        return _PLOT_PROFILE_POSITION
    if normalized == "coordination":
        return _PLOT_PROFILE_COORDINATION
    if normalized == "potential":
        return _PLOT_PROFILE_POTENTIAL
    if normalized == "orientation":
        return _PLOT_PROFILE_ORIENTATION
    if normalized in {"table", "hdf5"}:
        return _PLOT_PROFILE_TABLE
    raise ValueError(f"Unsupported plot profile '{profile_token}'.")


def _default_csv_output_path(source: str | Path, suffix: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "data"
    return _linak_output_dir_for_source(source_path) / f"{stem}_{suffix}.h5"


def _resolve_csv_output_path(args: argparse.Namespace, *, suffix: str) -> Path:
    source = _resolve_single_source_argument(
        args,
        positional_attr="source",
        source_label="HDF5 input file",
    )
    source_path = Path(source).expanduser().resolve()
    if getattr(args, "inplace", False):
        return source_path
    if args.output:
        return _resolve_non_overwriting_hdf5_path(args.output)
    return _resolve_non_overwriting_hdf5_path(_default_csv_output_path(source_path, suffix))


def _default_csv_plot_output_path(source: str | Path, kind: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    stem = source_path.stem or "data"
    return source_path.with_name(f"{stem}_{kind}.png")


def _default_csv_plot_output_for_sources(sources: list[Path], kind: str) -> Path:
    if len(sources) == 1:
        return _default_csv_plot_output_path(sources[0], kind)
    return Path.cwd() / f"linak_{kind}.png"


def _print_csv_preview_for_interactive(
    *,
    frame: Any,
    source_path: Path,
    rows: int = 8,
    heading: str = "Preview before interactive selection",
) -> None:
    from .storage.csv_tools import format_frame_preview

    print(f"{heading}: {source_path} (head {rows})")
    print(format_frame_preview(frame, rows=rows, tail=False, show_index=False))
    print("")


def _load_csv_frame_from_source(
    source: str | Path,
    *,
    group: str | None = None,
) -> tuple[Any, Path]:
    try:
        from .storage.hdf5_table import read_hdf5_frame
    except ModuleNotFoundError as exc:
        if exc.name in {"pandas", "h5py"}:
            raise ValueError(
                "HDF5 tabular commands require pandas and h5py. "
                "Install dependencies and rerun (for example: pip install pandas h5py)."
            ) from exc
        raise

    frame, source_info = read_hdf5_frame(source, group=group)
    source_path = source_info.source_path
    frame.attrs["linak_hdf5_source_info"] = source_info
    LOGGER.info(
        "Loaded HDF5 '%s' (analysis='%s', group='%s') with %d row(s) and %d column(s).",
        source_path,
        source_info.analysis or "unknown",
        source_info.container,
        len(frame),
        len(frame.columns),
    )
    if source_info.skipped_datasets:
        LOGGER.info("Skipped %d non-tabular dataset(s).", len(source_info.skipped_datasets))
    return frame, source_path


def _load_csv_frame(args: argparse.Namespace) -> tuple[Any, Path]:
    source = _resolve_single_source_argument(
        args,
        positional_attr="source",
        source_label="HDF5 input file",
    )
    return _load_csv_frame_from_source(
        source,
        group=getattr(args, "group", None),
    )


def _print_hdf5_metadata_overview(frame: Any) -> None:
    source_info = frame.attrs.get("linak_hdf5_source_info")
    if source_info is None:
        return

    from .storage.hdf5_table import format_hdf5_metadata_overview

    print(format_hdf5_metadata_overview(source_info))
    print("")


def _validate_csv_columns(frame: Any, requested: list[str]) -> list[str]:
    unknown = [column for column in requested if column not in frame.columns]
    if unknown:
        raise ValueError(f"Unknown column(s): {', '.join(unknown)}")
    unique: list[str] = []
    for column in requested:
        if column not in unique:
            unique.append(column)
    return unique


def _format_float(value: float | int | None, digits: int) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def _format_column_statistics(
    stats: dict[str, object], *, digits: int, metrics: list[str] | None
) -> str:
    kind = stats["kind"]
    if kind == "numeric":
        default_metrics = [
            "count",
            "missing",
            "distinct",
            "min",
            "max",
            "mean",
            "median",
            "std",
            "sum",
            "q05",
            "q25",
            "q75",
            "q95",
            "iqr",
        ]
    else:
        default_metrics = [
            "count",
            "missing",
            "distinct",
            "mode",
            "mode_count",
            "numeric_ratio",
        ]

    selected = metrics if metrics is not None else default_metrics
    unavailable = [metric for metric in selected if metric not in stats]
    if unavailable:
        raise ValueError(
            f"Metrics not available for column '{stats['column']}' ({kind}): {', '.join(unavailable)}."
        )

    lines = [f"Column: {stats['column']} ({kind})"]
    for metric in selected:
        value = stats[metric]
        if isinstance(value, float):
            rendered = _format_float(value, digits)
        elif isinstance(value, int):
            rendered = str(value)
        elif isinstance(value, list):
            rendered = ", ".join(f"{name}:{count}" for name, count in value) if value else "NA"
        else:
            rendered = "NA" if value is None else str(value)
        lines.append(f"  {metric:>12}: {rendered}")
    return "\n".join(lines)


def _add_plot_common_options(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(x_lim=None)
    parser.set_defaults(y_lim=None)
    render_group = parser.add_argument_group("General plot options")
    render_group.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Open an interactive plot-settings window with form controls, preview, and "
            "save actions (default when interactive plotting is enabled). Use --no-gui "
            "for direct Matplotlib rendering."
        ),
    )
    render_group.add_argument("-o", "--output", help="Output image path (PNG, PDF, SVG, ...)")
    render_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Show interactive plot window (default: enabled)",
    )
    render_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive plot window",
    )
    render_group.add_argument(
        "--backend",
        type=_parse_backend,
        default=DEFAULT_INTERACTIVE_BACKEND,
        metavar="BACKEND",
        help=(
            "Preferred Matplotlib backend when interactive plotting is enabled "
            f"(default: {DEFAULT_INTERACTIVE_BACKEND})"
        ),
    )
    render_group.add_argument(
        "--settings-source",
        metavar="PATH_OR_INDEX",
        default=None,
        help=(
            "When plotting multiple input HDF5 files, select which source provides persisted "
            "plot settings (default: first input). Accepts a 1-based index or one of the input paths."
        ),
    )
    render_group.add_argument(
        "--settings-profile",
        metavar="NAME",
        default=None,
        help=(
            "Optional named saved profile inside the selected plot-settings source. "
            "Defaults to that file's active saved profile."
        ),
    )

    axis_group = parser.add_argument_group("Axes and title")
    axis_group.add_argument(
        "--title",
        default=None,
        help="Optional plot title (default: inferred from data and analysis type)",
    )
    _add_toggle_state_argument(
        axis_group,
        flag="title-mode",
        dest="title_visible",
        feature_name="Title display",
    )
    axis_group.add_argument("--x-label", help="Custom x-axis label")
    axis_group.add_argument("--y-label", help="Custom y-axis label")
    axis_group.add_argument(
        "--x-scale",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
        help="X-axis scale (default: linear)",
    )
    axis_group.add_argument(
        "--y-scale",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
        help="Y-axis scale (default: linear)",
    )
    axis_group.add_argument("--x-min", type=float, metavar="XMIN", help="Lower x-axis limit")
    axis_group.add_argument("--x-max", type=float, metavar="XMAX", help="Upper x-axis limit")
    axis_group.add_argument("--y-min", type=float, metavar="YMIN", help="Lower y-axis limit")
    axis_group.add_argument("--y-max", type=float, metavar="YMAX", help="Upper y-axis limit")
    axis_group.add_argument(
        "--x-ticks",
        nargs="+",
        type=float,
        metavar="XTICK",
        help="Explicit x-axis tick positions",
    )
    axis_group.add_argument(
        "--y-ticks",
        nargs="+",
        type=float,
        metavar="YTICK",
        help="Explicit y-axis tick positions",
    )
    axis_group.add_argument(
        "--x-tick-rotation",
        type=float,
        help="X-axis tick-label rotation in degrees",
    )
    axis_group.add_argument(
        "--y-tick-rotation",
        type=float,
        help="Y-axis tick-label rotation in degrees",
    )

    legend_group = parser.add_argument_group("Series labels and legend")
    legend_group.add_argument(
        "--labels",
        "--series-labels",
        dest="series_labels",
        nargs="+",
        metavar="LABEL",
        help=(
            "Custom labels for plotted series. Count must match the rendered series count "
            "(used for legends and stored plot profiles). For multi-file plots, stored labels "
            "are merged per source automatically unless this flag is provided."
        ),
    )
    _add_toggle_state_argument(
        legend_group,
        flag="legend",
        dest="legend",
        feature_name="Legend display",
    )
    legend_group.add_argument(
        "--no-legend",
        dest="legend",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    legend_group.add_argument(
        "--legend-title",
        help="Optional legend title",
    )
    legend_group.add_argument(
        "--legend-loc",
        default="best",
        help="Matplotlib legend location (default: best)",
    )

    style_group = parser.add_argument_group("Plot style options")
    style_group.add_argument(
        "--figsize",
        nargs=2,
        type=_positive_float,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 7 4)",
    )
    style_group.add_argument(
        "--dpi",
        type=_positive_int,
        help="Figure DPI when saving output (default: 200)",
    )
    style_group.add_argument(
        "--font-family",
        help="Matplotlib font family (default: DejaVu Sans)",
    )
    style_group.add_argument(
        "--title-font-size",
        type=_positive_int,
        help="Title font size (default: 14)",
    )
    style_group.add_argument(
        "--label-font-size",
        type=_positive_int,
        help="Axis label font size (default: 12)",
    )
    style_group.add_argument(
        "--tick-font-size",
        type=_positive_int,
        help="Tick label font size (default: 10)",
    )
    style_group.add_argument(
        "--legend-font-size",
        type=_positive_int,
        help="Legend font size (default: 10)",
    )
    style_group.add_argument(
        "--line-width",
        type=_positive_float,
        help="Main line width (default: 2.0)",
    )
    style_group.add_argument("--line-color", help="Main line color (default: #1f77b4)")
    style_group.add_argument(
        "--line-colors",
        nargs="+",
        metavar="COLOR",
        help=(
            "Per-series line colors (count must match rendered series count). For multi-file "
            "plots, stored per-source colors are merged automatically unless this flag is provided."
        ),
    )
    _add_toggle_state_argument(
        style_group,
        flag="grid",
        dest="grid",
        feature_name="Grid display",
    )
    style_group.add_argument(
        "--no-grid",
        dest="grid",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    _add_toggle_state_argument(
        style_group,
        flag="ticks",
        dest="ticks",
        feature_name="Tick display",
    )
    _add_toggle_state_argument(
        style_group,
        flag="markers",
        dest="markers",
        feature_name="Line markers",
    )
    style_group.add_argument(
        "--grid-linestyle",
        help="Grid linestyle (default: --)",
    )
    style_group.add_argument(
        "--grid-linewidth",
        type=_positive_float,
        help="Grid line width (default: 0.8)",
    )
    style_group.add_argument(
        "--grid-alpha",
        type=_non_negative_float,
        help="Grid alpha transparency (default: 0.35)",
    )


def _add_plot_source_options(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "source",
        nargs="*",
        help=help_text,
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="Input HDF5 file path(s). Use -f/--files even for one file; required for multiple.",
    )


def _resolve_apply_output_path(args: argparse.Namespace) -> Path:
    if args.overwrite:
        return Path(args.trajectory).expanduser().resolve()
    if args.output is not None:
        return Path(args.output).expanduser().resolve()
    return _default_pbc_output_path(args.trajectory)


def _add_cell_resolution_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Cell / PBC options")
    group.add_argument(
        "--cell",
        nargs=3,
        type=_positive_float,
        metavar=("A", "B", "C"),
        help="Explicit orthorhombic cell lengths in Angstrom.",
    )
    group.add_argument(
        "-i",
        "--input",
        "--cp2k-input",
        "--lammps-input",
        dest="input",
        metavar="PATH",
        help=(
            "Path to simulation input file (.inp for CP2K, .lmp for LAMMPS). "
            "Used if automatic input discovery fails."
        ),
    )


def _set_cell_on_frames(frames: list[Atoms], cell: tuple[float, float, float]) -> None:
    for frame in frames:
        frame.set_cell(cell)
        frame.set_pbc((True, True, True))


def _preflight_resolve_cell(
    trajectory: str | Path,
    *,
    cell: tuple[float, float, float] | None,
    input_path: str | None,
    analysis_name: str,
) -> tuple[Any | None, Exception | None]:
    from .resolution import resolve_analysis_cell

    try:
        return resolve_analysis_cell(trajectory, cell=cell, input_path=input_path), None
    except (FileNotFoundError, ValueError) as exc:
        if cell is not None:
            raise
        LOGGER.info(
            "Could not resolve cell from simulation input before loading trajectory for %s "
            "analysis; checking trajectory metadata after load. %s",
            analysis_name,
            exc,
        )
        return None, exc


def _resolve_and_apply_required_cell(
    frames: list[Atoms],
    trajectory: str | Path,
    *,
    cell: tuple[float, float, float] | None,
    input_path: str | None,
    analysis_name: str,
    pre_resolved: Any | None = None,
    preflight_error: Exception | None = None,
) -> tuple[tuple[float, float, float], str, str | None]:
    from .resolution import resolve_analysis_cell

    has_trajectory_cell = _frames_have_usable_periodic_cell(frames)
    if cell is None and input_path is None and has_trajectory_cell:
        resolved = _cell_lengths_from_frame(frames[0])
        LOGGER.info(
            "Using periodic cell already present in trajectory for %s analysis: "
            "A=%.6g, B=%.6g, C=%.6g Angstrom.",
            analysis_name,
            resolved[0],
            resolved[1],
            resolved[2],
        )
        return resolved, "trajectory metadata", None

    cell_resolution = pre_resolved
    if cell_resolution is None and preflight_error is None:
        try:
            cell_resolution = resolve_analysis_cell(
                trajectory,
                cell=cell,
                input_path=input_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            preflight_error = exc

    if cell_resolution is None:
        resolved_error = preflight_error or ValueError("Could not resolve analysis cell.")
        if cell is None and has_trajectory_cell:
            resolved = _cell_lengths_from_frame(frames[0])
            LOGGER.info(
                "Could not resolve cell from simulation input for %s analysis; using "
                "periodic cell already present in trajectory. %s",
                analysis_name,
                resolved_error,
            )
            return resolved, "trajectory metadata", None
        raise resolved_error

    resolved_cell = cell_resolution.cell_angstrom
    LOGGER.info(
        "Using cell for %s analysis: A=%.6g, B=%.6g, C=%.6g Angstrom.",
        analysis_name,
        resolved_cell[0],
        resolved_cell[1],
        resolved_cell[2],
    )
    _set_cell_on_frames(frames, resolved_cell)
    return (
        resolved_cell,
        cell_resolution.source,
        str(cell_resolution.input_path) if cell_resolution.input_path is not None else None,
    )


def _maybe_apply_density_cell(
    frames: list[Atoms],
    trajectory: str | Path,
    *,
    cell: tuple[float, float, float] | None,
    input_path: str | None,
    pre_resolved: Any | None = None,
    preflight_error: Exception | None = None,
    analysis_label: str = "density analysis",
) -> tuple[tuple[float, float, float] | None, str, str | None]:
    """Try to resolve/apply a periodic cell for density; return None on fallback."""
    from .resolution import resolve_analysis_cell

    has_trajectory_cell = _frames_have_usable_periodic_cell(frames)
    if cell is None and input_path is None and has_trajectory_cell:
        resolved = _cell_lengths_from_frame(frames[0])
        LOGGER.info(
            "Using periodic cell already present in trajectory for %s: "
            "A=%.6g, B=%.6g, C=%.6g Angstrom.",
            analysis_label,
            resolved[0],
            resolved[1],
            resolved[2],
        )
        return resolved, "trajectory metadata", None

    cell_resolution = pre_resolved
    if cell_resolution is None and preflight_error is None:
        try:
            cell_resolution = resolve_analysis_cell(
                trajectory,
                cell=cell,
                input_path=input_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            preflight_error = exc

    if cell_resolution is None:
        resolved_error = preflight_error or ValueError("Could not resolve density cell.")
        if cell is None and has_trajectory_cell:
            resolved = _cell_lengths_from_frame(frames[0])
            LOGGER.info(
                "Could not resolve cell from simulation input for %s; using "
                "periodic cell already present in trajectory. %s",
                analysis_label,
                resolved_error,
            )
            return resolved, "trajectory metadata", None
        LOGGER.info(
            "No periodic cell resolved for %s; using linear density. %s",
            analysis_label,
            resolved_error,
        )
        return None, "unresolved", None
    resolved_cell = cell_resolution.cell_angstrom
    LOGGER.info(
        "Using cell for %s: A=%.6g, B=%.6g, C=%.6g Angstrom.",
        analysis_label,
        resolved_cell[0],
        resolved_cell[1],
        resolved_cell[2],
    )
    _set_cell_on_frames(frames, resolved_cell)
    return (
        resolved_cell,
        cell_resolution.source,
        str(cell_resolution.input_path) if cell_resolution.input_path is not None else None,
    )


def _preflight_resolve_analysis_timestep_fs(
    trajectory: str | Path,
    *,
    timestep_fs: float | None,
    input_path: str | None,
    analysis_name: str,
) -> tuple[Any | None, Exception | None]:
    from .resolution import resolve_analysis_timestep_fs

    try:
        return (
            resolve_analysis_timestep_fs(
                trajectory,
                timestep_fs=timestep_fs,
                input_path=input_path,
            ),
            None,
        )
    except ValueError as exc:
        if timestep_fs is not None:
            raise
        LOGGER.info(
            "Could not resolve timestep from simulation input before loading trajectory for %s "
            "analysis; checking trajectory metadata after load. %s",
            analysis_name,
            exc,
        )
        return None, exc


def _resolve_analysis_timestep_fs(
    trajectory: str | Path,
    *,
    timestep_fs: float | None,
    input_path: str | None,
    analysis_name: str,
    frames: list[Atoms] | None = None,
    pre_resolved: Any | None = None,
    preflight_error: Exception | None = None,
) -> tuple[float, str, str | None, float | None, int | None]:
    from .resolution import (
        TimestepResolution,
        _extract_metadata_timestep_details,
        resolve_analysis_timestep_fs,
    )

    resolved = pre_resolved
    if resolved is None and preflight_error is None:
        try:
            resolved = resolve_analysis_timestep_fs(
                trajectory,
                timestep_fs=timestep_fs,
                input_path=input_path,
                frames=frames,
            )
        except ValueError as exc:
            preflight_error = exc

    if resolved is None and frames is not None:
        metadata_timestep, metadata_md_timestep, metadata_stride = (
            _extract_metadata_timestep_details(frames)
        )
        if metadata_timestep is not None:
            resolved = TimestepResolution(
                frame_timestep_fs=metadata_timestep,
                source="trajectory metadata",
                md_timestep_fs=metadata_md_timestep,
                trajectory_stride_md=metadata_stride,
            )

    if resolved is None:
        resolved_error = preflight_error or ValueError("Could not resolve analysis timestep.")
        if timestep_fs is not None or input_path is not None:
            raise resolved_error
        LOGGER.info(
            "No timestep resolved for %s analysis; using default 0.5 fs. %s",
            analysis_name,
            resolved_error,
        )
        return 0.5, "fallback default", None, None, None

    LOGGER.info(
        "Using timestep for %s analysis: %.6g fs.",
        analysis_name,
        resolved.frame_timestep_fs,
    )
    return (
        resolved.frame_timestep_fs,
        resolved.source,
        str(resolved.input_path) if resolved.input_path is not None else None,
        resolved.md_timestep_fs,
        resolved.trajectory_stride_md,
    )


def _resolve_combined_msd_timestep_fs(timesteps_by_source: list[tuple[str, float]]) -> float:
    if not timesteps_by_source:
        raise ValueError("No trajectories available to resolve combined MSD timestep.")

    reference_source, reference_timestep = timesteps_by_source[0]
    for source, timestep in timesteps_by_source[1:]:
        if isclose(reference_timestep, timestep, rel_tol=0.0, abs_tol=1e-9):
            continue
        raise ValueError(
            "Cannot combine trajectories with different timestep values for MSD "
            f"({Path(reference_source).name}: {reference_timestep:.6g} fs, "
            f"{Path(source).name}: {timestep:.6g} fs). "
            "Use --timestep-fs to force one shared value."
        )
    return reference_timestep


def _build_plot_style(args: argparse.Namespace) -> PlotStyle:
    from .plot.plotting import with_style_overrides

    figure_size = tuple(args.figsize) if args.figsize is not None else None
    return with_style_overrides(
        figure_size=figure_size,
        dpi=args.dpi,
        font_family=args.font_family,
        title_font_size=args.title_font_size,
        label_font_size=args.label_font_size,
        tick_font_size=args.tick_font_size,
        legend_font_size=args.legend_font_size,
        line_width=args.line_width,
        line_color=args.line_color,
        grid=args.grid,
        grid_linestyle=args.grid_linestyle,
        grid_linewidth=args.grid_linewidth,
        grid_alpha=args.grid_alpha,
    )


def _normalize_series_setting_list(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    cleaned: list[str] = []
    for item in value:
        token = str(item).strip()
        if not token:
            return None
        cleaned.append(token)
    return cleaned or None


def _normalize_line_color_setting_list(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    cleaned = [str(item).strip() for item in value]
    return cleaned if any(cleaned) else None


def _coerce_series_override_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in value.items():
        series_id = str(raw_key).strip()
        if not series_id or not isinstance(raw_value, dict):
            continue
        overrides[series_id] = dict(raw_value)
    return overrides


def _coerce_series_order(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in value:
        token = str(raw).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        resolved.append(token)
    return resolved


def _resolve_series_id_order(
    series_ids: list[str],
    requested_order: list[str] | None,
) -> list[str]:
    if not series_ids:
        return []
    available = set(series_ids)
    resolved: list[str] = []
    seen: set[str] = set()
    for series_id in _coerce_series_order(requested_order):
        if series_id not in available or series_id in seen:
            continue
        seen.add(series_id)
        resolved.append(series_id)
    for series_id in series_ids:
        if series_id in seen:
            continue
        resolved.append(series_id)
    return resolved


def _resolve_gui_series_enabled_by_id(
    args: argparse.Namespace,
    descriptors: list[dict[str, Any]],
) -> dict[str, bool]:
    overrides = _coerce_series_override_map(getattr(args, "series_overrides", None))
    raw_enabled = getattr(args, "series_enabled", None)
    enabled_list = (
        raw_enabled
        if isinstance(raw_enabled, list) and len(raw_enabled) == len(descriptors)
        else None
    )
    enabled_by_id: dict[str, bool] = {}
    for index, descriptor in enumerate(descriptors):
        series_id = str(descriptor.get("series_id") or f"series:{index}")
        entry = overrides.get(series_id, {})
        if isinstance(entry, dict) and "enabled" in entry:
            enabled_by_id[series_id] = bool(entry.get("enabled"))
        elif enabled_list is not None:
            enabled_by_id[series_id] = bool(enabled_list[index])
        else:
            enabled_by_id[series_id] = True
    return enabled_by_id


def _filter_active_gui_descriptor_segments(
    *,
    args: argparse.Namespace,
    descriptor_segments_by_source: list[list[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    all_descriptors = [
        dict(descriptor) for segment in descriptor_segments_by_source for descriptor in segment
    ]
    if not all_descriptors:
        return [list() for _segment in descriptor_segments_by_source], []

    natural_ids = [
        str(descriptor.get("series_id") or f"series:{index}")
        for index, descriptor in enumerate(all_descriptors)
    ]
    enabled_by_id = _resolve_gui_series_enabled_by_id(args, all_descriptors)
    resolved_order = _resolve_series_id_order(natural_ids, getattr(args, "series_order", None))
    active_ids = [series_id for series_id in resolved_order if enabled_by_id.get(series_id, True)]
    active_id_set = set(active_ids)
    filtered_segments: list[list[dict[str, Any]]] = []
    for segment in descriptor_segments_by_source:
        filtered_segments.append(
            [
                dict(descriptor)
                for descriptor in segment
                if str(descriptor.get("series_id") or "") in active_id_set
            ]
        )
    return filtered_segments, active_ids


def _reorder_series_values(values: Any, indices: list[int]) -> Any:
    if not isinstance(values, list) or len(values) != len(indices):
        return values
    return [deepcopy(values[index]) for index in indices]


def _default_series_family_colors(
    series_descriptors: list[dict[str, Any]] | None,
    count: int,
) -> list[str]:
    _ = series_descriptors
    return _default_multi_series_colors(count)


def _default_multi_series_colors(count: int) -> list[str]:
    if count <= 0:
        return []

    colors: list[str] = []
    try:
        import matplotlib

        prop_cycle = matplotlib.rcParams.get("axes.prop_cycle")
        if prop_cycle is not None:
            by_key = prop_cycle.by_key()
            raw_colors = by_key.get("color", [])
            colors = [str(item).strip() for item in raw_colors if str(item).strip()]
    except Exception:
        colors = []

    if not colors:
        from .plot.plotting import DEFAULT_PLOT_STYLE

        colors = [DEFAULT_PLOT_STYLE.line_color]

    return [colors[index % len(colors)] for index in range(count)]


def _read_plot_profile_safe(
    source: str | Path,
    *,
    profile_key: str,
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    from .plot.plot_settings import read_plot_profile

    source_path = Path(source).expanduser().resolve()
    try:
        return read_plot_profile(source_path, profile_key, profile_name=profile_name)
    except (FileNotFoundError, OSError, ValueError) as exc:
        LOGGER.debug(
            "Could not read plot profile '%s' from '%s': %s", profile_key, source_path, exc
        )
        return None


def _resolve_multi_source_series_settings(
    *,
    sources: list[str],
    profile_key: str,
    fallback_labels_by_source: list[list[str]],
    series_descriptors: list[dict[str, Any]] | None = None,
    profile_name: str | None = None,
) -> tuple[list[str], list[str] | None]:
    if len(sources) != len(fallback_labels_by_source):
        raise ValueError("sources and fallback_labels_by_source must have equal lengths.")

    saved_label_segments: list[list[str] | None] = [None] * len(sources)
    saved_color_segments: list[list[str] | None] = [None] * len(sources)
    total_series = sum(len(labels) for labels in fallback_labels_by_source)

    for index, source in enumerate(sources):
        expected_count = len(fallback_labels_by_source[index])
        if expected_count == 0:
            continue
        saved_profile = _read_plot_profile_safe(
            source,
            profile_key=profile_key,
            profile_name=profile_name,
        )
        if not isinstance(saved_profile, dict):
            continue

        source_name = Path(source).name or str(source)
        saved_labels = _normalize_series_setting_list(saved_profile.get("series_labels"))
        if saved_labels is not None:
            if len(saved_labels) == expected_count:
                saved_label_segments[index] = saved_labels
            else:
                LOGGER.info(
                    "Ignoring saved series_labels from '%s': expected %d value(s) for current "
                    "series selection, got %d.",
                    source_name,
                    expected_count,
                    len(saved_labels),
                )

        saved_colors = _normalize_line_color_setting_list(saved_profile.get("line_colors"))
        if saved_colors is not None:
            if len(saved_colors) == expected_count:
                saved_color_segments[index] = saved_colors
            else:
                LOGGER.info(
                    "Ignoring saved line_colors from '%s': expected %d value(s) for current "
                    "series selection, got %d.",
                    source_name,
                    expected_count,
                    len(saved_colors),
                )

    merged_labels: list[str] = []
    for fallback_labels, saved_labels in zip(fallback_labels_by_source, saved_label_segments):
        merged_labels.extend(saved_labels if saved_labels is not None else fallback_labels)

    if not any(segment is not None for segment in saved_color_segments):
        return merged_labels, None

    merged_colors = _default_series_family_colors(series_descriptors, total_series)
    offset = 0
    for expected_labels, saved_colors in zip(fallback_labels_by_source, saved_color_segments):
        expected_count = len(expected_labels)
        if saved_colors is not None:
            merged_colors[offset : offset + expected_count] = saved_colors
        offset += expected_count

    return merged_labels, merged_colors


def _apply_effective_series_settings(
    *,
    args: argparse.Namespace,
    sources: list[str],
    profile_key: str,
    fallback_labels_by_source: list[list[str]],
    series_descriptors: list[dict[str, Any]] | None = None,
    allow_saved_multi_source_merge: bool = True,
    materialize_default_colors: bool = True,
) -> None:
    total_series = sum(len(labels) for labels in fallback_labels_by_source)
    if total_series <= 0:
        return

    explicit_labels = _runtime_option_was_provided(args, "series_labels")
    explicit_line_colors = _runtime_option_was_provided(args, "line_colors")

    merged_labels: list[str] | None = None
    merged_colors: list[str] | None = None
    if len(sources) > 1 and allow_saved_multi_source_merge:
        merged_labels, merged_colors = _resolve_multi_source_series_settings(
            sources=sources,
            profile_key=profile_key,
            fallback_labels_by_source=fallback_labels_by_source,
            series_descriptors=series_descriptors,
        )
    overrides = _coerce_series_override_map(getattr(args, "series_overrides", None))
    ordered_descriptors = list(series_descriptors) if isinstance(series_descriptors, list) else []

    if overrides and len(ordered_descriptors) == total_series:
        from .plot.fitting import coerce_fit_config

        override_labels: list[str] = []
        override_colors: list[str] = []
        override_enabled: list[bool] = []
        override_show_in_legend: list[bool] = []
        override_fit_configs: list[dict[str, Any] | None] = []
        override_fit_enabled: list[bool] = []
        override_fit_labels: list[str | None] = []
        override_fit_show_in_legend: list[bool] = []
        override_widths: list[float | None] = []
        override_markers: list[str | None] = []
        override_line_kwargs: list[dict[str, Any] | None] = []
        override_norm_modes: list[str | None] = []
        override_norm_values: list[float | None] = []
        override_norm_x_refs: list[float | None] = []
        any_color = False
        any_disabled = False
        any_hidden_in_legend = False
        any_fit = False
        any_fit_label = False
        any_fit_hidden_in_legend = False
        any_width = False
        any_marker = False
        any_line_kwargs = False
        any_norm = False
        for descriptor in ordered_descriptors:
            default_label = str(descriptor.get("default_label") or "Series").strip() or "Series"
            series_id = str(descriptor.get("series_id") or "").strip()
            entry = overrides.get(series_id, {})
            label_override = str(entry.get("label_override") or "").strip()
            override_labels.append(label_override or default_label)

            color = str(entry.get("color") or "").strip()
            override_colors.append(color)
            any_color = any_color or bool(color)

            enabled = bool(entry.get("enabled", True))
            override_enabled.append(enabled)
            any_disabled = any_disabled or (enabled is False)

            show_in_legend = bool(entry.get("show_in_legend", True))
            override_show_in_legend.append(show_in_legend)
            any_hidden_in_legend = any_hidden_in_legend or (show_in_legend is False)

            fit_label = entry.get("fit_label_override")
            fit_show_in_legend = bool(entry.get("fit_show_in_legend", True))
            fit_config = coerce_fit_config(
                entry.get("fit"),
                legacy_enabled=entry.get("fit_enabled", False),
                legacy_label=fit_label,
                legacy_show_in_legend=fit_show_in_legend,
            )
            override_fit_configs.append(fit_config if fit_config.get("fit_enabled") else None)
            fit_enabled = bool(fit_config.get("fit_enabled", False))
            override_fit_enabled.append(fit_enabled)
            any_fit = any_fit or fit_enabled

            fit_label_value = None if fit_label in {None, ""} else str(fit_label)
            override_fit_labels.append(fit_label_value)
            any_fit_label = any_fit_label or bool(fit_label_value)

            override_fit_show_in_legend.append(fit_show_in_legend)
            any_fit_hidden_in_legend = any_fit_hidden_in_legend or (fit_show_in_legend is False)

            raw_width = entry.get("line_width")
            width_value = None if raw_width in {None, ""} else float(str(raw_width))
            override_widths.append(width_value)
            any_width = any_width or (width_value is not None)

            marker = entry.get("marker")
            marker_value = None if marker in {None, ""} else str(marker)
            override_markers.append(marker_value)
            any_marker = any_marker or bool(marker_value)

            line_kwargs = entry.get("line_kwargs")
            line_kwargs_value = dict(line_kwargs) if isinstance(line_kwargs, dict) else None
            if entry.get("alpha") is not None:
                if line_kwargs_value is None:
                    line_kwargs_value = {}
                line_kwargs_value["alpha"] = float(entry["alpha"])
            override_line_kwargs.append(line_kwargs_value)
            any_line_kwargs = any_line_kwargs or (line_kwargs_value is not None)

            mode_value = str(entry.get("normalization_mode") or "").strip().lower() or None
            if mode_value == "none":
                mode_value = None
            override_norm_modes.append(mode_value)
            override_norm_values.append(
                None
                if entry.get("normalization_value") is None
                else float(entry["normalization_value"])
            )
            override_norm_x_refs.append(
                None
                if entry.get("normalization_x_ref") is None
                else float(entry["normalization_x_ref"])
            )
            any_norm = any_norm or (mode_value is not None)

        merged_labels = override_labels
        merged_colors = override_colors if any_color else None
        args.series_enabled = override_enabled if any_disabled else None
        args.series_show_in_legend = override_show_in_legend if any_hidden_in_legend else None
        args.series_fit_configs = override_fit_configs if any_fit else None
        args.series_fit_enabled = override_fit_enabled if any_fit else None
        args.series_fit_labels = override_fit_labels if any_fit_label else None
        args.series_fit_show_in_legend = (
            override_fit_show_in_legend if any_fit_hidden_in_legend else None
        )
        args.series_line_widths = override_widths if any_width else None
        args.series_markers = override_markers if any_marker else None
        args.series_line_kwargs = override_line_kwargs if any_line_kwargs else None
        args.series_normalization_modes = override_norm_modes if any_norm else None
        args.series_normalization_values = override_norm_values if any_norm else None
        args.series_normalization_x_refs = override_norm_x_refs if any_norm else None
    else:
        args.series_fit_configs = None
        args.series_fit_enabled = None
        args.series_fit_labels = None
        args.series_fit_show_in_legend = None

    if not explicit_labels:
        if merged_labels is not None:
            args.series_labels = merged_labels
        else:
            normalized_labels = _normalize_series_setting_list(getattr(args, "series_labels", None))
            if normalized_labels is None:
                args.series_labels = None
            elif len(normalized_labels) == total_series:
                args.series_labels = normalized_labels
            else:
                LOGGER.info(
                    "Ignoring saved series_labels: expected %d value(s) for current series "
                    "selection, got %d.",
                    total_series,
                    len(normalized_labels),
                )
                args.series_labels = None

    if not explicit_line_colors:
        if merged_colors is not None:
            args.line_colors = merged_colors
        else:
            normalized_colors = _normalize_line_color_setting_list(
                getattr(args, "line_colors", None)
            )
            if normalized_colors is None:
                args.line_colors = None
            elif len(normalized_colors) == total_series:
                args.line_colors = normalized_colors
            else:
                LOGGER.info(
                    "Ignoring saved line_colors: expected %d value(s) for current series "
                    "selection, got %d.",
                    total_series,
                    len(normalized_colors),
                )
                args.line_colors = None

    ordered_descriptors = list(series_descriptors) if isinstance(series_descriptors, list) else []
    if len(ordered_descriptors) != total_series:
        return

    natural_ids = [
        str(item.get("series_id") or f"series:{index}")
        for index, item in enumerate(ordered_descriptors)
    ]
    resolved_order = _resolve_series_id_order(natural_ids, getattr(args, "series_order", None))
    if resolved_order == natural_ids:
        return
    index_by_id = {series_id: index for index, series_id in enumerate(natural_ids)}
    indices = [index_by_id[series_id] for series_id in resolved_order]

    for attr in (
        "series_labels",
        "line_colors",
        "series_enabled",
        "series_show_in_legend",
        "series_fit_configs",
        "series_fit_enabled",
        "series_fit_labels",
        "series_fit_show_in_legend",
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
    ):
        setattr(args, attr, _reorder_series_values(getattr(args, attr, None), indices))


def _persist_effective_series_settings(
    *,
    source_path: Path,
    profile_key: str,
    series_labels: list[str] | None,
    line_colors: list[str] | None,
    profile_name: str | None = None,
) -> None:
    from .plot.plot_settings import read_plot_profile, write_plot_profile

    existing = read_plot_profile(source_path, profile_key, profile_name=profile_name) or {}
    if not isinstance(existing, dict):
        existing = {}
    updated = dict(existing)

    if series_labels is None:
        updated.pop("series_labels", None)
    else:
        updated["series_labels"] = list(series_labels)

    if line_colors is None:
        updated.pop("line_colors", None)
    else:
        updated["line_colors"] = list(line_colors)

    if updated == existing:
        return
    write_plot_profile(
        source_path,
        profile_key,
        updated,
        profile_name=profile_name,
    )


def _flatten_series_labels_by_source(fallback_labels_by_source: list[list[str]]) -> list[str]:
    return [label for source_labels in fallback_labels_by_source for label in source_labels]


def _segment_gui_series_descriptors(
    descriptors: list[dict[str, Any]],
    fallback_labels_by_source: list[list[str]],
) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    offset = 0
    for labels in fallback_labels_by_source:
        count = len(labels)
        segments.append([dict(item) for item in descriptors[offset : offset + count]])
        offset += count
    if offset != len(descriptors):
        raise ValueError("Descriptor count does not match fallback_labels_by_source.")
    return segments


def _apply_descriptor_extra_segments(
    descriptors: list[dict[str, Any]],
    extra_segments_by_source: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    flattened_extras = [dict(extra) for segment in extra_segments_by_source for extra in segment]
    if len(flattened_extras) != len(descriptors):
        raise ValueError("Descriptor extras must align with the descriptor count.")
    updated: list[dict[str, Any]] = []
    for descriptor, extra in zip(descriptors, flattened_extras):
        merged = dict(descriptor)
        merged.update(extra)
        updated.append(merged)
    return updated


def _build_gui_descriptor_segments(
    *,
    sources: list[str],
    fallback_labels_by_source: list[list[str]],
    series_id_segments_by_source: list[list[str]],
    origin_path_segments_by_source: list[list[str]],
    load_source_path_segments_by_source: list[list[str]],
    extra_segments_by_source: list[list[dict[str, Any]]] | None = None,
) -> list[list[dict[str, Any]]]:
    descriptors = _build_gui_series_descriptors(
        sources=sources,
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
        load_source_path_segments_by_source=load_source_path_segments_by_source,
    )
    if extra_segments_by_source is not None:
        descriptors = _apply_descriptor_extra_segments(descriptors, extra_segments_by_source)
    return _segment_gui_series_descriptors(descriptors, fallback_labels_by_source)


def _build_gui_series_descriptors(
    *,
    sources: list[str],
    fallback_labels_by_source: list[list[str]],
    series_id_segments_by_source: list[list[str]] | None = None,
    origin_path_segments_by_source: list[list[str]] | None = None,
    load_source_path_segments_by_source: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    if len(sources) != len(fallback_labels_by_source):
        raise ValueError("sources and fallback_labels_by_source must have equal lengths.")
    if series_id_segments_by_source is not None and len(series_id_segments_by_source) != len(
        sources
    ):
        raise ValueError("series_id_segments_by_source must align with sources.")
    if origin_path_segments_by_source is not None and len(origin_path_segments_by_source) != len(
        sources
    ):
        raise ValueError("origin_path_segments_by_source must align with sources.")
    if load_source_path_segments_by_source is not None and len(
        load_source_path_segments_by_source
    ) != len(sources):
        raise ValueError("load_source_path_segments_by_source must align with sources.")

    descriptors: list[dict[str, Any]] = []
    source_group_indices: dict[str, int] = {}
    for source_index, (source, labels) in enumerate(zip(sources, fallback_labels_by_source)):
        id_segment = (
            series_id_segments_by_source[source_index]
            if series_id_segments_by_source is not None
            else None
        )
        origin_segment = (
            origin_path_segments_by_source[source_index]
            if origin_path_segments_by_source is not None
            else None
        )
        load_source_segment = (
            load_source_path_segments_by_source[source_index]
            if load_source_path_segments_by_source is not None
            else None
        )
        if id_segment is not None and len(id_segment) != len(labels):
            raise ValueError("series id segments must align with fallback labels.")
        if origin_segment is not None and len(origin_segment) != len(labels):
            raise ValueError("origin path segments must align with fallback labels.")
        if load_source_segment is not None and len(load_source_segment) != len(labels):
            raise ValueError("load source path segments must align with fallback labels.")
        for local_index, default_label in enumerate(labels):
            resolved_source_path = (
                Path(origin_segment[local_index]).expanduser()
                if origin_segment is not None
                else Path(source).expanduser()
            )
            resolved_load_source_path = (
                Path(load_source_segment[local_index]).expanduser()
                if load_source_segment is not None
                else Path(source).expanduser()
            )
            source_name = resolved_source_path.name or str(resolved_source_path)
            source_directory = (
                str(resolved_source_path.parent)
                if str(resolved_source_path.parent) not in {"", "."}
                else ""
            )
            source_group_key = str(resolved_source_path)
            resolved_source_index = source_group_indices.setdefault(
                source_group_key,
                len(source_group_indices),
            )
            descriptors.append(
                {
                    "series_id": (
                        str(id_segment[local_index]).strip()
                        if id_segment is not None
                        else f"series:{source_index}:{local_index}"
                    ),
                    "source_index": resolved_source_index,
                    "series_index": local_index,
                    "source_name": source_name,
                    "source_directory": source_directory,
                    "source_path": str(resolved_source_path),
                    "load_source_path": str(resolved_load_source_path),
                    "default_label": str(default_label).strip() or f"Series {len(descriptors) + 1}",
                }
            )
    return descriptors


def _resolve_gui_default_series_labels(
    *,
    args: argparse.Namespace,
    sources: list[str],
    profile_key: str,
    fallback_labels_by_source: list[list[str]],
) -> list[str]:
    default_args = deepcopy(args)
    default_args.series_labels = None
    default_args.line_colors = None
    default_args.series_overrides = None
    default_args._runtime_argv = ()
    _apply_effective_series_settings(
        args=default_args,
        sources=sources,
        profile_key=profile_key,
        fallback_labels_by_source=fallback_labels_by_source,
        allow_saved_multi_source_merge=False,
    )
    labels = _normalize_series_setting_list(getattr(default_args, "series_labels", None))
    if labels is not None:
        return labels
    return _flatten_series_labels_by_source(fallback_labels_by_source)


def _profile_uid_from_payload(payload: dict[str, Any], *, fallback_prefix: str, index: int) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        token = str(metadata.get("profile_uid") or "").strip()
        if token:
            return token
    return f"{fallback_prefix}:{index}"


def _merge_gui_only_plot_settings(
    target: dict[str, Any],
    saved: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(saved, dict):
        return target
    merged = dict(target)
    for key in (
        "series_order",
        "series_overrides",
        "series_enabled",
        "series_show_in_legend",
        "series_alpha",
        "_gui_locked_fields",
        "_gui_sync_modes",
    ):
        if key in saved:
            merged[key] = deepcopy(saved[key])
    return merged


def _strip_redundant_series_lists_for_gui(settings: dict[str, Any]) -> dict[str, Any]:
    """Drop positional per-series lists when ID-keyed overrides are present.

    GUI initialization consumes `series_descriptors` in natural source order. When the CLI has already
    materialized display-order lists like `series_enabled`, those positional lists no longer align with
    descriptor order. The ID-keyed `series_overrides` payload is the authoritative representation.
    """
    if not isinstance(settings.get("series_overrides"), dict):
        return settings
    cleaned = dict(settings)
    for key in (
        "series_labels",
        "line_colors",
        "series_enabled",
        "series_show_in_legend",
        "series_alpha",
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
    ):
        cleaned.pop(key, None)
    return cleaned


def _without_preview_series_state(settings: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(settings, dict):
        return {}
    blocked = {
        "series_order",
        "series_labels",
        "line_colors",
        "line_color",
        "line_kwargs",
        "series_enabled",
        "series_fit_configs",
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
        "markers",
    }
    return {key: deepcopy(value) for key, value in settings.items() if key not in blocked}


def _build_density_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_density_plot_profiles(
        sources=sources,
        species=args.species,
        axis=args.axis,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
        plotter_kwargs={
            "x_mode": args.x_mode,
            "quantity": args.quantity,
        },
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_DENSITY,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
    )


def _build_msd_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_msd_plot_profiles(
        sources=sources,
        species=args.species,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_msd",
        plotter_kwargs=None,
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_MSD,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
    )


def _build_rdf_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="rdf",
    )
    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_rdf_plot_profiles(
        sources=sources,
        species_a=args.species_a,
        species_b=args.species_b,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_rdf",
        plotter_kwargs=None,
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_RDF,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
        profile_filter_options=_build_rdf_profile_filter_options(raw_payloads_by_source),
    )


def _build_position_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_position_plot_profiles(
        sources=sources,
        species=args.species,
        axis=args.axis,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_position",
        plotter_kwargs={
            "component": args.component,
            "map_color": args.map_color,
            "time_axis": args.time_axis,
        },
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_POSITION,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
    )


def _build_coordination_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="coordination",
    )
    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_coordination_plot_profiles(
        sources=sources,
        species_a=args.species_a,
        species_b=args.species_b,
        axis=args.axis,
        component=args.component,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_coordination",
        plotter_kwargs={
            "component": args.component,
            "time_axis": args.time_axis,
        },
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_COORDINATION,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
        profile_filter_options=_build_coordination_profile_filter_options(raw_payloads_by_source),
    )


def _headers_by_source_as_metadata_payloads(
    headers_by_source: list[tuple[str, list[dict[str, Any]]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (source, [{"metadata": dict(header)} for header in headers])
        for source, headers in headers_by_source
    ]


def _group_descriptors_by_load_source(
    descriptors: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for descriptor in descriptors:
        source_path = str(
            descriptor.get("load_source_path") or descriptor.get("source_path") or ""
        ).strip()
        if source_path not in grouped:
            grouped[source_path] = []
            order.append(source_path)
        grouped[source_path].append(descriptor)
    return [(source_path, grouped[source_path]) for source_path in order]


def _build_density_gui_lazy_catalog(
    args: argparse.Namespace,
    *,
    sources: list[str],
    active_profiles_by_series_id: dict[str, Any] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .analysis.density import load_density_profiles_by_index, _normalize_species_query

    headers_by_source = _read_analysis_profile_headers_by_source(
        sources=sources,
        analysis="density",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(header) for _source, headers in headers_by_source for header in headers
        ],
    )
    resolved_species_label: str | None = None
    if args.species is not None and str(args.species).strip():
        _selection_mode, resolved_species_label = _normalize_species_query(args.species)

    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    load_source_path_segments_by_source: list[list[str]] = []
    extra_segments_by_source: list[list[dict[str, Any]]] = []
    for source, headers in headers_by_source:
        source_labels: list[str] = []
        source_ids: list[str] = []
        source_origins: list[str] = []
        source_load_paths: list[str] = []
        source_extras: list[dict[str, Any]] = []
        for header in headers:
            source_label = _metadata_source_label(header, fallback_source=source)
            base_species = (
                resolved_species_label or str(header.get("species", "")).strip() or "UNKNOWN"
            )
            rendered_species = (
                f"{source_label}:{base_species}" if prefix_source_labels else base_species
            )
            profile_index = int(header.get("profile_index", len(source_labels)))
            profile_uid = _profile_uid_from_payload(
                {"metadata": header},
                fallback_prefix="density",
                index=profile_index,
            )
            source_labels.append(rendered_species)
            source_ids.append(profile_uid)
            source_origins.append(str(header.get("origin_hdf5_path") or source))
            source_load_paths.append(str(header.get("source_path") or source))
            source_extras.append(
                {
                    "profile_index": profile_index,
                    "profile_uid": profile_uid,
                    "rendered_species": rendered_species,
                }
            )
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
        load_source_path_segments_by_source.append(source_load_paths)
        extra_segments_by_source.append(source_extras)

    descriptor_segments = _build_gui_descriptor_segments(
        sources=sources,
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
        load_source_path_segments_by_source=load_source_path_segments_by_source,
        extra_segments_by_source=extra_segments_by_source,
    )

    def _load_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
        loaded_by_id: dict[str, Any] = {}
        for load_source_path, source_descriptors in _group_descriptors_by_load_source(descriptors):
            indices = [int(descriptor["profile_index"]) for descriptor in source_descriptors]
            profiles = load_density_profiles_by_index(
                load_source_path,
                indices,
                axis=args.axis,
                species=args.species,
            )
            if len(profiles) != len(source_descriptors):
                raise ValueError("Lazy density loader returned mismatched profile count.")
            for descriptor, profile in zip(source_descriptors, profiles):
                loaded_by_id[str(descriptor["series_id"])] = replace(
                    profile,
                    species=str(descriptor.get("rendered_species") or profile.species),
                )
        return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
        plotter_kwargs={
            "x_mode": args.x_mode,
            "quantity": args.quantity,
        },
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options=None,
        load_profiles=_load_profiles,
        _active_profiles_by_series_id=(
            active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
        ),
    )


def _build_msd_gui_lazy_catalog(
    args: argparse.Namespace,
    *,
    sources: list[str],
    active_profiles_by_series_id: dict[str, Any] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .analysis.msd import (
        load_msd_profiles_by_index,
        _normalize_species as _normalize_msd_species,
    )

    headers_by_source = _read_analysis_profile_headers_by_source(
        sources=sources,
        analysis="msd",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(header) for _source, headers in headers_by_source for header in headers
        ],
    )
    resolved_species = (
        _normalize_msd_species(args.species)
        if args.species is not None and str(args.species).strip()
        else None
    )

    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    load_source_path_segments_by_source: list[list[str]] = []
    extra_segments_by_source: list[list[dict[str, Any]]] = []
    for source, headers in headers_by_source:
        source_labels: list[str] = []
        source_ids: list[str] = []
        source_origins: list[str] = []
        source_load_paths: list[str] = []
        source_extras: list[dict[str, Any]] = []
        for header in headers:
            source_label = _metadata_source_label(header, fallback_source=source)
            base_species = resolved_species or str(header.get("species", "")).strip() or "UNKNOWN"
            rendered_species = (
                f"{source_label}:{base_species}" if prefix_source_labels else base_species
            )
            profile_index = int(header.get("profile_index", len(source_labels)))
            profile_uid = _profile_uid_from_payload(
                {"metadata": header},
                fallback_prefix="msd",
                index=profile_index,
            )
            source_labels.append(rendered_species)
            source_ids.append(profile_uid)
            source_origins.append(str(header.get("origin_hdf5_path") or source))
            source_load_paths.append(str(header.get("source_path") or source))
            source_extras.append(
                {
                    "profile_index": profile_index,
                    "profile_uid": profile_uid,
                    "rendered_species": rendered_species,
                }
            )
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
        load_source_path_segments_by_source.append(source_load_paths)
        extra_segments_by_source.append(source_extras)

    descriptor_segments = _build_gui_descriptor_segments(
        sources=sources,
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
        load_source_path_segments_by_source=load_source_path_segments_by_source,
        extra_segments_by_source=extra_segments_by_source,
    )

    def _load_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
        loaded_by_id: dict[str, Any] = {}
        for load_source_path, source_descriptors in _group_descriptors_by_load_source(descriptors):
            indices = [int(descriptor["profile_index"]) for descriptor in source_descriptors]
            profiles = load_msd_profiles_by_index(
                load_source_path,
                indices,
                species=args.species,
            )
            if len(profiles) != len(source_descriptors):
                raise ValueError("Lazy MSD loader returned mismatched profile count.")
            for descriptor, profile in zip(source_descriptors, profiles):
                loaded_by_id[str(descriptor["series_id"])] = replace(
                    profile,
                    species=str(descriptor.get("rendered_species") or profile.species),
                )
        return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_msd",
        plotter_kwargs=None,
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options=None,
        load_profiles=_load_profiles,
        _active_profiles_by_series_id=(
            active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
        ),
    )


def _build_rdf_gui_lazy_catalog(
    args: argparse.Namespace,
    *,
    sources: list[str],
    active_profiles_by_series_id: dict[str, Any] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .analysis.rdf import (
        _normalize_species as _normalize_rdf_species,
        load_rdf_profiles_by_index,
    )

    resolved_species_b = args.species_b if args.species_b is not None else args.species_a
    wanted_species_a = (
        None
        if args.species_a is None or not str(args.species_a).strip()
        else _normalize_rdf_species(args.species_a)
    )
    wanted_species_b = (
        None
        if resolved_species_b is None or not str(resolved_species_b).strip()
        else _normalize_rdf_species(resolved_species_b)
    )
    headers_by_source = _read_analysis_profile_headers_by_source(
        sources=sources,
        analysis="rdf",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(header) for _source, headers in headers_by_source for header in headers
        ],
    )

    filtered_headers_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    load_source_path_segments_by_source: list[list[str]] = []
    extra_segments_by_source: list[list[dict[str, Any]]] = []
    for source, headers in headers_by_source:
        filtered_headers: list[dict[str, Any]] = []
        source_labels: list[str] = []
        source_ids: list[str] = []
        source_origins: list[str] = []
        source_load_paths: list[str] = []
        source_extras: list[dict[str, Any]] = []
        for header in headers:
            resolved_a = str(header.get("species_a", "")).strip() or "UNKNOWN"
            resolved_b = str(header.get("species_b", "")).strip() or resolved_a
            if (
                wanted_species_a is not None
                and _normalize_rdf_species(resolved_a) != wanted_species_a
            ):
                continue
            if (
                wanted_species_b is not None
                and _normalize_rdf_species(resolved_b) != wanted_species_b
            ):
                continue
            source_label = _metadata_source_label(header, fallback_source=source)
            rendered_species_a = (
                f"{source_label}:{resolved_a}" if prefix_source_labels else resolved_a
            )
            profile_index = int(header.get("profile_index", len(source_labels)))
            profile_uid = _profile_uid_from_payload(
                {"metadata": header},
                fallback_prefix="rdf",
                index=profile_index,
            )
            filtered_headers.append(header)
            source_labels.append(f"{rendered_species_a}-{resolved_b}")
            source_ids.append(profile_uid)
            source_origins.append(str(header.get("origin_hdf5_path") or source))
            source_load_paths.append(str(header.get("source_path") or source))
            source_extras.append(
                {
                    "profile_index": profile_index,
                    "profile_uid": profile_uid,
                    "rendered_species_a": rendered_species_a,
                    "rendered_species_b": resolved_b,
                }
            )
        filtered_headers_by_source.append((source, filtered_headers))
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
        load_source_path_segments_by_source.append(source_load_paths)
        extra_segments_by_source.append(source_extras)

    descriptor_segments = _build_gui_descriptor_segments(
        sources=sources,
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
        load_source_path_segments_by_source=load_source_path_segments_by_source,
        extra_segments_by_source=extra_segments_by_source,
    )

    def _load_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
        loaded_by_id: dict[str, Any] = {}
        for load_source_path, source_descriptors in _group_descriptors_by_load_source(descriptors):
            indices = [int(descriptor["profile_index"]) for descriptor in source_descriptors]
            profiles = load_rdf_profiles_by_index(
                load_source_path,
                indices,
                species_a=args.species_a,
                species_b=resolved_species_b,
            )
            if len(profiles) != len(source_descriptors):
                raise ValueError("Lazy RDF loader returned mismatched profile count.")
            for descriptor, profile in zip(source_descriptors, profiles):
                loaded_by_id[str(descriptor["series_id"])] = replace(
                    profile,
                    species_a=str(descriptor.get("rendered_species_a") or profile.species_a),
                    species_b=str(descriptor.get("rendered_species_b") or profile.species_b),
                )
        return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_rdf",
        plotter_kwargs=None,
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options=_build_rdf_profile_filter_options(
            _headers_by_source_as_metadata_payloads(filtered_headers_by_source)
        ),
        load_profiles=_load_profiles,
        _active_profiles_by_series_id=(
            active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
        ),
    )


def _build_position_gui_lazy_catalog(
    args: argparse.Namespace,
    *,
    sources: list[str],
    active_profiles_by_series_id: dict[str, Any] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .analysis.position import _normalize_species as _normalize_position_species
    from .analysis.position import load_position_profiles_by_index
    from .storage.hdf5_utils import read_linak_hdf5_profiles_by_index

    wanted_species = (
        None
        if args.species is None or not str(args.species).strip()
        else _normalize_position_species(args.species)
    )
    wanted_axis = (
        None if args.axis is None or not str(args.axis).strip() else str(args.axis).strip().lower()
    )
    headers_by_source = _read_analysis_profile_headers_by_source(
        sources=sources,
        analysis="position",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(header) for _source, headers in headers_by_source for header in headers
        ],
    )

    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    load_source_path_segments_by_source: list[list[str]] = []
    extra_segments_by_source: list[list[dict[str, Any]]] = []
    for source, headers in headers_by_source:
        source_path = Path(source).expanduser().resolve()
        lightweight_payloads = read_linak_hdf5_profiles_by_index(
            source_path,
            list(range(len(headers))),
            expected_analysis="position",
            dataset_names=("atom_indices",),
        )
        source_labels: list[str] = []
        source_ids: list[str] = []
        source_origins: list[str] = []
        source_load_paths: list[str] = []
        source_extras: list[dict[str, Any]] = []
        for header, (datasets, _metadata) in zip(headers, lightweight_payloads):
            source_label = _metadata_source_label(header, fallback_source=str(source_path))
            resolved_species = str(header.get("species", "")).strip() or "UNKNOWN"
            resolved_axis = str(header.get("axis", "z")).strip().lower() or "z"
            if wanted_species is not None and wanted_species != "ALL":
                if _normalize_position_species(resolved_species) != wanted_species:
                    continue
            if wanted_axis is not None and resolved_axis != wanted_axis:
                continue
            rendered_species = (
                f"{source_label}:{resolved_species}" if prefix_source_labels else resolved_species
            )
            profile_index = int(header.get("profile_index", 0))
            profile_uid = _profile_uid_from_payload(
                {"metadata": header},
                fallback_prefix="position",
                index=profile_index,
            )
            atom_indices = np.asarray(datasets.get("atom_indices", []), dtype=int)
            for atom_index in atom_indices.tolist():
                atom_token = int(atom_index)
                source_labels.append(f"{rendered_species}[{atom_token}]")
                source_ids.append(f"{profile_uid}:atom:{atom_token}")
                source_origins.append(str(header.get("origin_hdf5_path") or source))
                source_load_paths.append(str(header.get("source_path") or source))
                source_extras.append(
                    {
                        "profile_index": profile_index,
                        "profile_uid": profile_uid,
                        "atom_index": atom_token,
                        "rendered_species": rendered_species,
                    }
                )
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
        load_source_path_segments_by_source.append(source_load_paths)
        extra_segments_by_source.append(source_extras)

    descriptor_segments = _build_gui_descriptor_segments(
        sources=sources,
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
        load_source_path_segments_by_source=load_source_path_segments_by_source,
        extra_segments_by_source=extra_segments_by_source,
    )

    def _load_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
        loaded_by_id: dict[str, Any] = {}
        for load_source_path, source_descriptors in _group_descriptors_by_load_source(descriptors):
            grouped_parents: dict[int, list[dict[str, Any]]] = {}
            parent_order: list[int] = []
            for descriptor in source_descriptors:
                profile_index = int(descriptor["profile_index"])
                if profile_index not in grouped_parents:
                    grouped_parents[profile_index] = []
                    parent_order.append(profile_index)
                grouped_parents[profile_index].append(descriptor)
            parent_profiles = load_position_profiles_by_index(
                load_source_path,
                parent_order,
                species=args.species,
                axis=args.axis,
            )
            if len(parent_profiles) != len(parent_order):
                raise ValueError("Lazy position loader returned mismatched parent profile count.")
            parent_by_index = {
                profile_index: profile
                for profile_index, profile in zip(parent_order, parent_profiles)
            }
            for profile_index in parent_order:
                parent_profile = parent_by_index[profile_index]
                for descriptor in grouped_parents[profile_index]:
                    child_profile = _extract_position_profile_atom_series(
                        parent_profile,
                        int(descriptor["atom_index"]),
                    )
                    loaded_by_id[str(descriptor["series_id"])] = replace(
                        child_profile,
                        species=str(descriptor.get("rendered_species") or child_profile.species),
                    )
        return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_position",
        plotter_kwargs={
            "component": args.component,
            "map_color": args.map_color,
            "time_axis": args.time_axis,
        },
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options=None,
        load_profiles=_load_profiles,
        _active_profiles_by_series_id=(
            active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
        ),
    )


def _build_coordination_gui_lazy_catalog(
    args: argparse.Namespace,
    *,
    sources: list[str],
    active_profiles_by_series_id: dict[str, Any] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .analysis.coordination import (
        _normalize_axis as _normalize_coordination_axis,
        _normalize_species as _normalize_coordination_species,
        load_coordination_profiles_by_index,
    )
    from .storage.hdf5_utils import read_linak_hdf5_profiles_by_index

    normalized_component = str(args.component).strip().lower().replace("_", "-")
    resolved_species_b = args.species_b if args.species_b is not None else args.species_a
    wanted_species_a = (
        None
        if args.species_a is None or not str(args.species_a).strip()
        else _normalize_coordination_species(args.species_a)
    )
    wanted_species_b = (
        None
        if resolved_species_b is None or not str(resolved_species_b).strip()
        else _normalize_coordination_species(resolved_species_b)
    )
    wanted_axis = (
        None
        if args.axis is None or not str(args.axis).strip()
        else _normalize_coordination_axis(args.axis)
    )
    headers_by_source = _read_analysis_profile_headers_by_source(
        sources=sources,
        analysis="coordination",
    )
    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(header) for _source, headers in headers_by_source for header in headers
        ],
    )

    filtered_headers_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    load_source_path_segments_by_source: list[list[str]] = []
    extra_segments_by_source: list[list[dict[str, Any]]] = []
    for source, headers in headers_by_source:
        source_path = Path(source).expanduser().resolve()
        matching_headers: list[dict[str, Any]] = []
        matching_indices: list[int] = []
        for header in headers:
            resolved_a = str(header.get("species_a", "")).strip() or "UNKNOWN"
            resolved_b = str(header.get("species_b", "")).strip() or resolved_a
            resolved_axis = str(header.get("axis", "z")).strip().lower() or "z"
            if (
                wanted_species_a is not None
                and _normalize_coordination_species(resolved_a) != wanted_species_a
            ):
                continue
            if (
                wanted_species_b is not None
                and _normalize_coordination_species(resolved_b) != wanted_species_b
            ):
                continue
            if wanted_axis is not None and resolved_axis != wanted_axis:
                continue
            matching_headers.append(header)
            matching_indices.append(int(header.get("profile_index", len(matching_headers) - 1)))

        lightweight_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if normalized_component != "distance" and matching_indices:
            lightweight_payloads = read_linak_hdf5_profiles_by_index(
                source_path,
                matching_indices,
                expected_analysis="coordination",
                dataset_names=("atom_indices",),
            )
        payloads_by_index = {
            int(metadata.get("profile_index", profile_index)): datasets
            for profile_index, (datasets, metadata) in enumerate(lightweight_payloads)
        }

        source_labels: list[str] = []
        source_ids: list[str] = []
        source_origins: list[str] = []
        source_load_paths: list[str] = []
        source_extras: list[dict[str, Any]] = []
        for header in matching_headers:
            source_label = _metadata_source_label(header, fallback_source=str(source_path))
            resolved_a = str(header.get("species_a", "")).strip() or "UNKNOWN"
            resolved_b = str(header.get("species_b", "")).strip() or resolved_a
            rendered_species_a = (
                f"{source_label}:{resolved_a}" if prefix_source_labels else resolved_a
            )
            profile_index = int(header.get("profile_index", 0))
            profile_uid = _profile_uid_from_payload(
                {"metadata": header},
                fallback_prefix="coordination",
                index=profile_index,
            )
            if normalized_component == "distance":
                source_labels.append(f"{rendered_species_a}-{resolved_b}")
                source_ids.append(profile_uid)
                source_origins.append(str(header.get("origin_hdf5_path") or source))
                source_load_paths.append(str(header.get("source_path") or source))
                source_extras.append(
                    {
                        "profile_index": profile_index,
                        "profile_uid": profile_uid,
                        "rendered_species_a": rendered_species_a,
                        "rendered_species_b": resolved_b,
                    }
                )
                continue

            atom_indices = np.asarray(
                payloads_by_index.get(profile_index, {}).get("atom_indices", []), dtype=int
            )
            for atom_index in atom_indices.tolist():
                atom_token = int(atom_index)
                source_labels.append(f"{rendered_species_a}[{atom_token}]")
                source_ids.append(f"{profile_uid}:atom:{atom_token}")
                source_origins.append(str(header.get("origin_hdf5_path") or source))
                source_load_paths.append(str(header.get("source_path") or source))
                source_extras.append(
                    {
                        "profile_index": profile_index,
                        "profile_uid": profile_uid,
                        "atom_index": atom_token,
                        "rendered_species_a": rendered_species_a,
                        "rendered_species_b": resolved_b,
                    }
                )
        filtered_headers_by_source.append((source, matching_headers))
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
        load_source_path_segments_by_source.append(source_load_paths)
        extra_segments_by_source.append(source_extras)

    descriptor_segments = _build_gui_descriptor_segments(
        sources=sources,
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
        load_source_path_segments_by_source=load_source_path_segments_by_source,
        extra_segments_by_source=extra_segments_by_source,
    )

    def _load_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
        loaded_by_id: dict[str, Any] = {}
        for load_source_path, source_descriptors in _group_descriptors_by_load_source(descriptors):
            grouped_parents: dict[int, list[dict[str, Any]]] = {}
            parent_order: list[int] = []
            for descriptor in source_descriptors:
                profile_index = int(descriptor["profile_index"])
                if profile_index not in grouped_parents:
                    grouped_parents[profile_index] = []
                    parent_order.append(profile_index)
                grouped_parents[profile_index].append(descriptor)
            parent_profiles = load_coordination_profiles_by_index(
                load_source_path,
                parent_order,
                species_a=args.species_a,
                species_b=resolved_species_b,
                axis=args.axis,
            )
            if len(parent_profiles) != len(parent_order):
                raise ValueError(
                    "Lazy coordination loader returned mismatched parent profile count."
                )
            parent_by_index = {
                profile_index: profile
                for profile_index, profile in zip(parent_order, parent_profiles)
            }
            for profile_index in parent_order:
                parent_profile = parent_by_index[profile_index]
                for descriptor in grouped_parents[profile_index]:
                    if normalized_component == "distance":
                        loaded_by_id[str(descriptor["series_id"])] = replace(
                            parent_profile,
                            species_a=str(
                                descriptor.get("rendered_species_a") or parent_profile.species_a
                            ),
                            species_b=str(
                                descriptor.get("rendered_species_b") or parent_profile.species_b
                            ),
                        )
                        continue
                    child_profile = _extract_coordination_profile_atom_series(
                        parent_profile,
                        int(descriptor["atom_index"]),
                    )
                    loaded_by_id[str(descriptor["series_id"])] = replace(
                        child_profile,
                        species_a=str(
                            descriptor.get("rendered_species_a") or child_profile.species_a
                        ),
                        species_b=str(
                            descriptor.get("rendered_species_b") or child_profile.species_b
                        ),
                    )
        return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_coordination",
        plotter_kwargs={
            "component": args.component,
            "time_axis": args.time_axis,
        },
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options=_build_coordination_profile_filter_options(
            _headers_by_source_as_metadata_payloads(filtered_headers_by_source)
        ),
        load_profiles=_load_profiles,
        _active_profiles_by_series_id=(
            active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
        ),
    )


def _build_potential_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .analysis.potential import load_potential_plot_profiles

    resolved_sources = [Path(source).expanduser().resolve() for source in sources]
    flattened_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    total_rows = 0
    complete_rows = 0
    incomplete_rows = 0
    for source_path in resolved_sources:
        plot_profiles, summary = load_potential_plot_profiles(source_path)
        total_rows += int(summary.get("total_rows") or 0)
        complete_rows += int(summary.get("complete_rows") or 0)
        incomplete_rows += int(summary.get("incomplete_rows") or 0)
        label_prefix = source_path.stem or source_path.name or str(source_path)
        fallback_labels_by_source.append(
            [f"{label_prefix}: {profile.default_label}" for profile in plot_profiles]
        )
        series_id_segments_by_source.append(
            [f"{source_path}::{profile.series_id}" for profile in plot_profiles]
        )
        origin_path_segments_by_source.append([str(source_path) for _profile in plot_profiles])
        for profile in plot_profiles:
            flattened_profiles.append(
                replace(
                    profile,
                    series_id=f"{source_path}::{profile.series_id}",
                    default_label=f"{label_prefix}: {profile.default_label}",
                    source_path=str(source_path),
                )
            )
    descriptors = _build_gui_series_descriptors(
        sources=[str(source_path) for source_path in resolved_sources],
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
    )
    return _GuiPlotRenderContext(
        profile=flattened_profiles,
        plot_source_label=(
            str(resolved_sources[0]) if len(resolved_sources) == 1 else "multi_source_potential"
        ),
        plotter_kwargs=None,
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=[str(source_path) for source_path in resolved_sources],
            profile_key=_PLOT_PROFILE_POTENTIAL,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=descriptors,
        profile_filter_options={
            "potential_summary": {
                "x_axis_label": "Record ID",
                "total_rows": total_rows,
                "complete_rows": complete_rows,
                "incomplete_rows": incomplete_rows,
            }
        },
    )


def _build_orientation_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_orientation_plot_profiles(sources=sources)
    raw_component = getattr(args, "component", "average")
    orientation_component = (
        raw_component if raw_component in {"average", "density-weighted", "heatmap"} else "average"
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_orientation",
        plotter_kwargs={
            "component": orientation_component,
            "angle": getattr(args, "angle", "polar"),
            "heatmap_vmin": getattr(args, "heatmap_vmin", None),
            "heatmap_vmax": getattr(args, "heatmap_vmax", None),
            "heatmap_cmap": getattr(args, "heatmap_cmap", None),
            "y_bin_width": getattr(args, "y_bin_width", None),
            "y_bin_reducer": getattr(args, "y_bin_reducer", None),
            "heatmap_normalize": getattr(args, "heatmap_normalize", False),
            "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
            "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
            "heatmap_colorbar_label_size": getattr(args, "heatmap_colorbar_label_size", None),
            "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
            "heatmap_colorbar_position": getattr(args, "heatmap_colorbar_position", "right"),
            "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
            "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
            "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
        },
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_ORIENTATION,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
    )


def _parse_toggle_state(raw: str) -> bool | None:
    token = raw.strip().lower()
    if token in {"on", "true", "yes", "1"}:
        return True
    if token in {"off", "false", "no", "0"}:
        return False
    if token in {"auto", "default"}:
        return None
    raise argparse.ArgumentTypeError("Expected one of: on, off, auto")


def _add_toggle_state_argument(
    group: argparse._ArgumentGroup,
    *,
    flag: str,
    dest: str,
    feature_name: str,
) -> None:
    group.add_argument(
        f"--{flag}",
        dest=dest,
        nargs="?",
        const="on",
        default=None,
        type=_parse_toggle_state,
        metavar="{on|off|auto}",
        help=(
            f"{feature_name}. Use `--{flag}` for on, `--{flag} off` for off, "
            f"or `--{flag} auto` for default behavior."
        ),
    )


def _resolve_x_lim(args: argparse.Namespace) -> list[float | None] | None:
    """Resolve x-axis limits from explicit min/max bounds or persisted x_lim."""
    x_min = getattr(args, "x_min", None)
    x_max = getattr(args, "x_max", None)
    if x_min is not None or x_max is not None:
        return [
            None if x_min is None else float(x_min),
            None if x_max is None else float(x_max),
        ]

    raw = getattr(args, "x_lim", None)
    if raw is None:
        return None
    return [
        None if raw[0] is None else float(raw[0]),
        None if raw[1] is None else float(raw[1]),
    ]


def _resolve_y_lim(args: argparse.Namespace) -> list[float | None] | None:
    """Resolve y-axis limits from explicit min/max bounds or persisted y_lim."""
    y_min = getattr(args, "y_min", None)
    y_max = getattr(args, "y_max", None)
    if y_min is not None or y_max is not None:
        return [
            None if y_min is None else float(y_min),
            None if y_max is None else float(y_max),
        ]

    raw = getattr(args, "y_lim", None)
    if raw is None:
        return None
    return [
        None if raw[0] is None else float(raw[0]),
        None if raw[1] is None else float(raw[1]),
    ]


def _resolve_gui_mode(args: argparse.Namespace) -> bool:
    raw = getattr(args, "gui", None)
    use_gui = bool(getattr(args, "show", True)) if raw is None else bool(raw)
    args.gui = use_gui
    return use_gui


def _render_profile_plot(
    *,
    args: argparse.Namespace,
    source: str,
    analysis_name: str,
    profile: Any,
    plotter: Callable[..., Path | None],
    plotter_kwargs: dict[str, Any] | None = None,
    series_descriptors: list[dict[str, Any]] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    from .plot.plotting import configure_matplotlib_backend

    interactive_requested = bool(args.show)
    if interactive_requested:
        try:
            configure_matplotlib_backend(
                interactive=True,
                preferred_backend=args.backend,
            )
        except RuntimeError:
            if not args.show:
                configure_matplotlib_backend(interactive=False, preferred_backend=args.backend)
    else:
        configure_matplotlib_backend(interactive=False, preferred_backend=args.backend)

    extra_kwargs = {} if plotter_kwargs is None else dict(plotter_kwargs)
    style = _build_plot_style(args)
    captured_state: dict[str, Any] = {}
    ordered_profile = profile
    ordered_descriptors = list(series_descriptors or [])
    if (
        isinstance(profile, list)
        and ordered_descriptors
        and len(profile) == len(ordered_descriptors)
    ):
        natural_ids = [
            str(descriptor.get("series_id") or f"series:{index}")
            for index, descriptor in enumerate(ordered_descriptors)
        ]
        resolved_order = _resolve_series_id_order(natural_ids, getattr(args, "series_order", None))
        if resolved_order != natural_ids:
            index_by_id = {series_id: index for index, series_id in enumerate(natural_ids)}
            indices = [index_by_id[series_id] for series_id in resolved_order]
            ordered_profile = [profile[index] for index in indices]
            ordered_descriptors = [ordered_descriptors[index] for index in indices]

    shared_kwargs = {
        "series_ids": [
            str(descriptor.get("series_id") or f"series:{index}")
            for index, descriptor in enumerate(ordered_descriptors)
        ]
        if ordered_descriptors
        else None,
        "title": args.title,
        "x_label": args.x_label,
        "y_label": args.y_label,
        "x_scale": args.x_scale,
        "y_scale": args.y_scale,
        "x_lim": _resolve_x_lim(args),
        "y_lim": _resolve_y_lim(args),
        "x_ticks": args.x_ticks,
        "y_ticks": args.y_ticks,
        "x_tick_rotation": args.x_tick_rotation,
        "y_tick_rotation": args.y_tick_rotation,
        "x_label_pad": getattr(args, "x_label_pad", None),
        "y_label_pad": getattr(args, "y_label_pad", None),
        "title_visible": args.title_visible,
        "ticks_visible": args.ticks,
        "markers": args.markers,
        "legend": args.legend,
        "legend_title": args.legend_title,
        "legend_loc": args.legend_loc,
        "series_labels": args.series_labels,
        "series_enabled": getattr(args, "series_enabled", None),
        "series_show_in_legend": getattr(args, "series_show_in_legend", None),
        "series_line_widths": getattr(args, "series_line_widths", None),
        "series_markers": getattr(args, "series_markers", None),
        "series_normalization_modes": getattr(args, "series_normalization_modes", None),
        "series_normalization_values": getattr(args, "series_normalization_values", None),
        "series_normalization_x_refs": getattr(args, "series_normalization_x_refs", None),
        "x_bin_width": getattr(args, "x_bin_width", None)
        if getattr(args, "x_bin_width", None) is not None
        else getattr(args, "time_section_width", None),
        "x_bin_reducer": getattr(args, "x_bin_reducer", None),
        "matplotlib_rc": getattr(args, "matplotlib_rc", None),
        "figure_kwargs": getattr(args, "figure_kwargs", None),
        "axes_kwargs": getattr(args, "axes_kwargs", None),
        "line_kwargs": getattr(args, "line_kwargs", None),
        "series_line_kwargs": getattr(args, "series_line_kwargs", None),
        "grid_kwargs": getattr(args, "grid_kwargs", None),
        "legend_kwargs": getattr(args, "legend_kwargs", None),
        "tick_params_kwargs": getattr(args, "tick_params_kwargs", None),
        "tight_layout_kwargs": getattr(args, "tight_layout_kwargs", None),
        "savefig_kwargs": getattr(args, "savefig_kwargs", None),
        "line_colors": (
            args.line_colors
            if getattr(args, "line_colors", None) is not None
            else _default_series_family_colors(ordered_descriptors, len(ordered_descriptors))
            if ordered_descriptors
            else None
        ),
        "show_blocking": not bool(getattr(args, "gui", False)),
        "capture_state": captured_state,
        "suppress_output_log": bool(getattr(args, "_suppress_output_log", False)),
    }
    shared_kwargs["series_fit_configs"] = getattr(args, "series_fit_configs", None)
    shared_kwargs["series_fit_enabled"] = getattr(args, "series_fit_enabled", None)
    shared_kwargs["series_fit_labels"] = getattr(args, "series_fit_labels", None)
    shared_kwargs["series_fit_show_in_legend"] = getattr(
        args,
        "series_fit_show_in_legend",
        None,
    )

    def _render_with_options(show: bool, output: str | Path | None) -> Path | None:
        call_kwargs = dict(shared_kwargs)
        if not isinstance(profile, list):
            series_ids = call_kwargs.pop("series_ids", None)
            if isinstance(series_ids, list) and series_ids:
                call_kwargs["series_id"] = str(series_ids[0])
            labels = call_kwargs.pop("series_labels", None)
            if isinstance(labels, list) and labels:
                call_kwargs["line_label"] = str(labels[0])
            series_show_in_legend = call_kwargs.pop("series_show_in_legend", None)
            if isinstance(series_show_in_legend, list) and series_show_in_legend:
                call_kwargs["show_in_legend"] = bool(series_show_in_legend[0])
            fit_configs = call_kwargs.pop("series_fit_configs", None)
            if isinstance(fit_configs, list) and fit_configs:
                first_fit_config = fit_configs[0]
                if isinstance(first_fit_config, dict):
                    call_kwargs["fit_config"] = dict(first_fit_config)
            fit_enabled = call_kwargs.pop("series_fit_enabled", None)
            if isinstance(fit_enabled, list) and fit_enabled:
                call_kwargs["fit_enabled"] = bool(fit_enabled[0])
            fit_labels = call_kwargs.pop("series_fit_labels", None)
            if isinstance(fit_labels, list) and fit_labels:
                first_fit_label = fit_labels[0]
                if first_fit_label is not None:
                    call_kwargs["fit_label"] = str(first_fit_label)
            fit_show_in_legend = call_kwargs.pop("series_fit_show_in_legend", None)
            if isinstance(fit_show_in_legend, list) and fit_show_in_legend:
                call_kwargs["fit_show_in_legend"] = bool(fit_show_in_legend[0])
            per_series_line_kwargs = call_kwargs.pop("series_line_kwargs", None)
            if isinstance(per_series_line_kwargs, list) and per_series_line_kwargs:
                first_kwargs = per_series_line_kwargs[0]
                if isinstance(first_kwargs, dict):
                    merged_line_kwargs: dict[str, Any] = {}
                    if isinstance(call_kwargs.get("line_kwargs"), dict):
                        merged_line_kwargs.update(dict(call_kwargs["line_kwargs"]))
                    merged_line_kwargs.update(first_kwargs)
                    call_kwargs["line_kwargs"] = merged_line_kwargs
        return plotter(
            ordered_profile,
            output=output,
            show=show,
            preferred_backend=args.backend,
            style=style,
            **extra_kwargs,
            **call_kwargs,
        )

    if args.show:
        try:
            saved = _render_with_options(True, args.output)
            return saved, captured_state
        except RuntimeError as exc:
            fallback_output = args.output or _default_plot_output_path(source, analysis_name)
            LOGGER.warning("Interactive plotting unavailable: %s", exc)
            LOGGER.warning(
                "Falling back to non-interactive render. Plot will be saved to '%s'.",
                fallback_output,
            )
            saved = _render_with_options(False, fallback_output)
            return saved, captured_state

    saved_path = _render_with_options(False, args.output)
    if saved_path is None and not bool(getattr(args, "_suppress_output_log", False)):
        LOGGER.warning("No interactive display or output path requested. Nothing was rendered.")
    return saved_path, captured_state


def _collect_plot_settings_for_persistence(
    args: argparse.Namespace, *, keys: tuple[str, ...]
) -> dict[str, Any]:
    candidate = _collect_plot_settings_from_args(args, keys=keys)
    if "x_lim" in candidate:
        candidate["x_lim"] = _resolve_x_lim(args)
    if "y_lim" in candidate:
        candidate["y_lim"] = _resolve_y_lim(args)
    return candidate


def _derive_gui_locked_fields(settings: dict[str, Any]) -> dict[str, bool]:
    x_lim = settings.get("x_lim")
    y_lim = settings.get("y_lim")
    return {
        "title": settings.get("title") is not None,
        "x_label": settings.get("x_label") is not None,
        "y_label": settings.get("y_label") is not None,
        "x_lim": isinstance(x_lim, (list, tuple)) and any(value is not None for value in x_lim[:2]),
        "y_lim": isinstance(y_lim, (list, tuple)) and any(value is not None for value in y_lim[:2]),
        "x_ticks": settings.get("x_ticks") is not None,
        "y_ticks": settings.get("y_ticks") is not None,
        "x_label_pad": settings.get("x_label_pad") is not None,
        "y_label_pad": settings.get("y_label_pad") is not None,
    }


def _derive_gui_sync_modes(settings: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key in ("title", "x_label", "y_label"):
        if settings.get(key) is None:
            resolved[key] = "auto"
        else:
            resolved[key] = "manual"
    return resolved


def _apply_gui_settings_to_args(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        setattr(args, key, value)
    if isinstance(settings.get("series_overrides"), dict):
        for key in (
            "series_labels",
            "line_colors",
            "series_enabled",
            "series_show_in_legend",
            "series_alpha",
            "series_line_widths",
            "series_markers",
            "series_line_kwargs",
            "series_normalization_modes",
            "series_normalization_values",
            "series_normalization_x_refs",
        ):
            setattr(args, key, None)
    if "x_bin_width" in settings and hasattr(args, "time_section_width"):
        args.time_section_width = settings.get("x_bin_width")
    if hasattr(args, "x_lim"):
        args.x_lim = None
    if hasattr(args, "y_lim"):
        args.y_lim = None


def _open_plot_settings_gui(
    *,
    title: str,
    initial_settings: dict[str, Any],
    on_preview: Callable[[dict[str, Any]], dict[str, Any] | None],
    on_save: Callable[[str, dict[str, Any]], str],
    on_save_figure: Callable[[dict[str, Any], str], str | tuple[str, dict[str, Any]]] | None = None,
    on_import_hdf5: Callable[[str, str], dict[str, Any]] | None = None,
    analysis_name: str | None = None,
    on_resolve_series_defaults: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    initial_profile_name: str | None = None,
    available_profile_names: list[str] | None = None,
    default_profile_settings: dict[str, Any] | None = None,
    on_load_profile: Callable[[str], dict[str, Any]] | None = None,
    on_delete_profile: Callable[[str], tuple[str | None, str]] | None = None,
    on_set_active_profile: Callable[[str], str] | None = None,
    allow_named_profiles: bool = True,
) -> None:
    from .plot.plot_gui import launch_plot_settings_panel

    launch_plot_settings_panel(
        title=title,
        initial_settings=initial_settings,
        on_preview=on_preview,
        on_save=on_save,
        on_save_figure=on_save_figure,
        on_import_hdf5=on_import_hdf5,
        analysis_name=analysis_name,
        on_resolve_series_defaults=on_resolve_series_defaults,
        initial_profile_name=initial_profile_name,
        available_profile_names=available_profile_names,
        default_profile_settings=default_profile_settings,
        on_load_profile=on_load_profile,
        on_delete_profile=on_delete_profile,
        on_set_active_profile=on_set_active_profile,
        allow_named_profiles=allow_named_profiles,
    )


def _is_gui_preview_output_path(path: str | Path) -> bool:
    resolved = Path(path).expanduser()
    return (
        resolved.parent == Path(tempfile.gettempdir())
        and resolved.name.startswith("linak_preview_")
        and resolved.suffix.lower() == ".png"
    )


def _launch_profile_plot_gui(
    *,
    args: argparse.Namespace,
    default_args: argparse.Namespace,
    source_path: Path,
    profile_key: str,
    setting_keys: tuple[str, ...],
    gui_title: str,
    analysis_name: str,
    plotter: Callable[..., Path | None],
    initial_context: _GuiPlotRenderContext,
    build_context: Callable[[argparse.Namespace], _GuiPlotRenderContext],
    build_full_context: Callable[[argparse.Namespace], _GuiPlotRenderContext] | None = None,
) -> None:
    from .plot.plot_settings import (
        delete_named_plot_profile,
        read_active_plot_profile_name,
        read_plot_profile,
        read_plot_profile_names,
        set_active_plot_profile,
        supports_named_plot_profiles,
        write_plot_profile,
    )

    allow_named_profiles = supports_named_plot_profiles(source_path)
    if build_full_context is None:
        build_full_context = build_context
    initial_settings = _collect_plot_settings_for_persistence(args, keys=setting_keys)
    initial_settings["_gui_locked_fields"] = _derive_gui_locked_fields(initial_settings)
    initial_settings["_gui_sync_modes"] = _derive_gui_sync_modes(initial_settings)
    initial_settings["series_count"] = max(1, int(initial_context.series_count))
    initial_settings["series_descriptors"] = deepcopy(initial_context.series_descriptors)
    initial_settings["_profile_filter_options"] = deepcopy(initial_context.profile_filter_options)
    if initial_context.default_series_labels and not initial_settings.get("series_labels"):
        initial_settings["series_labels"] = list(initial_context.default_series_labels)
    default_settings = _collect_plot_settings_for_persistence(default_args, keys=setting_keys)
    default_settings["series_count"] = max(1, int(initial_context.series_count))
    default_settings["series_descriptors"] = deepcopy(initial_context.series_descriptors)
    default_settings["_profile_filter_options"] = deepcopy(initial_context.profile_filter_options)
    if initial_context.default_series_labels and not default_settings.get("series_labels"):
        default_settings["series_labels"] = list(initial_context.default_series_labels)
    available_profile_names = (
        read_plot_profile_names(source_path, profile_key) if allow_named_profiles else ["Default"]
    )
    if not available_profile_names:
        available_profile_names = ["Default"]
    initial_profile_name = (
        read_active_plot_profile_name(source_path, profile_key) or available_profile_names[0]
    )
    if not allow_named_profiles:
        initial_profile_name = "Default"
    initial_saved_profile = read_plot_profile(
        source_path,
        profile_key,
        profile_name=(initial_profile_name if allow_named_profiles else None),
    )
    initial_settings = _merge_gui_only_plot_settings(initial_settings, initial_saved_profile)
    initial_settings = _strip_redundant_series_lists_for_gui(initial_settings)
    gui_render_sources = [
        f"gui-series-source:{index}"
        for index in range(len(initial_context.fallback_labels_by_source))
    ]

    initial_preview_args = deepcopy(args)
    initial_preview_args.show = False
    initial_preview_args.output = None
    initial_preview_args._suppress_output_log = True
    initial_preview_context = build_context(initial_preview_args)
    initial_render_state: dict[str, Any] = {}
    if initial_preview_context.series_count > 0:
        _apply_effective_series_settings(
            args=initial_preview_args,
            sources=gui_render_sources,
            profile_key=profile_key,
            fallback_labels_by_source=initial_preview_context.fallback_labels_by_source,
            series_descriptors=initial_preview_context.series_descriptors,
            allow_saved_multi_source_merge=False,
            materialize_default_colors=True,
        )
        _initial_saved_path, initial_render_state = _render_profile_plot(
            args=initial_preview_args,
            source=initial_preview_context.plot_source_label,
            analysis_name=analysis_name,
            profile=initial_preview_context.profile,
            plotter=plotter,
            plotter_kwargs=initial_preview_context.plotter_kwargs,
            series_descriptors=initial_preview_context.series_descriptors,
        )
    if initial_render_state:
        initial_settings.update(_without_preview_series_state(initial_render_state))

    def _preview(gui_settings: dict[str, Any]) -> dict[str, Any]:
        preview_args = deepcopy(args)
        _apply_gui_settings_to_args(preview_args, gui_settings)
        preview_args.show = True
        preview_args.output = None
        context = build_context(preview_args)
        if context.series_count <= 0:
            raise ValueError("No series are enabled. Turn on at least one series to preview.")
        _apply_effective_series_settings(
            args=preview_args,
            sources=gui_render_sources,
            profile_key=profile_key,
            fallback_labels_by_source=context.fallback_labels_by_source,
            series_descriptors=context.series_descriptors,
            allow_saved_multi_source_merge=False,
            materialize_default_colors=True,
        )
        _saved_path, render_state = _render_profile_plot(
            args=preview_args,
            source=context.plot_source_label,
            analysis_name=analysis_name,
            profile=context.profile,
            plotter=plotter,
            plotter_kwargs=context.plotter_kwargs,
            series_descriptors=context.series_descriptors,
        )
        return render_state

    def _save(profile_name: str, gui_settings: dict[str, Any]) -> str:
        save_args = deepcopy(args)
        _apply_gui_settings_to_args(save_args, gui_settings)
        candidate = _collect_plot_settings_for_persistence(save_args, keys=setting_keys)
        if isinstance(gui_settings.get("series_descriptors"), list):
            candidate["series_descriptors"] = deepcopy(gui_settings["series_descriptors"])
        else:
            save_context = build_full_context(save_args)
            candidate["series_descriptors"] = deepcopy(save_context.series_descriptors)
        if gui_settings.get("series_order") is not None:
            candidate["series_order"] = deepcopy(gui_settings["series_order"])
        else:
            candidate.pop("series_order", None)
        if "series_overrides" in gui_settings:
            candidate["series_overrides"] = deepcopy(gui_settings["series_overrides"])
            for key in (
                "series_labels",
                "line_colors",
                "series_enabled",
                "series_show_in_legend",
                "series_alpha",
                "series_line_widths",
                "series_markers",
                "series_line_kwargs",
                "series_normalization_modes",
                "series_normalization_values",
                "series_normalization_x_refs",
            ):
                candidate.pop(key, None)
        if "_gui_locked_fields" in gui_settings:
            candidate["_gui_locked_fields"] = deepcopy(gui_settings["_gui_locked_fields"])
        if "_gui_sync_modes" in gui_settings:
            candidate["_gui_sync_modes"] = deepcopy(gui_settings["_gui_sync_modes"])
        write_plot_profile(
            source_path,
            profile_key,
            candidate,
            profile_name=profile_name,
        )
        return f"Saved '{profile_name}' to {source_path.name}."

    def _save_figure(gui_settings: dict[str, Any], output_path: str) -> tuple[str, dict[str, Any]]:
        save_args = deepcopy(args)
        _apply_gui_settings_to_args(save_args, gui_settings)
        save_args.show = False
        save_args.output = output_path
        save_args._suppress_output_log = _is_gui_preview_output_path(output_path)
        context = build_context(save_args)
        if context.series_count <= 0:
            raise ValueError("No series are enabled. Turn on at least one series before exporting.")
        _apply_effective_series_settings(
            args=save_args,
            sources=gui_render_sources,
            profile_key=profile_key,
            fallback_labels_by_source=context.fallback_labels_by_source,
            series_descriptors=context.series_descriptors,
            allow_saved_multi_source_merge=False,
            materialize_default_colors=True,
        )
        saved_path, render_state = _render_profile_plot(
            args=save_args,
            source=context.plot_source_label,
            analysis_name=analysis_name,
            profile=context.profile,
            plotter=plotter,
            plotter_kwargs=context.plotter_kwargs,
            series_descriptors=context.series_descriptors,
        )
        if saved_path is None:
            raise ValueError("No output was generated for the requested figure path.")
        return f"Saved figure to '{saved_path}'.", render_state

    def _import_hdf5(source_hdf5_path: str, profile_name: str) -> dict[str, Any]:
        imported_path = Path(source_hdf5_path).expanduser().resolve()
        imported = read_plot_profile(imported_path, profile_key, profile_name=profile_name)
        if imported is None:
            imported = read_plot_profile(imported_path, profile_key)
        if imported is None:
            raise ValueError(
                f"No plot settings profile '{profile_key}' found in '{imported_path}'."
            )
        LOGGER.info(
            "Loaded plot settings template from '%s' (%s).",
            imported_path,
            profile_key,
        )
        return imported

    def _load_profile(profile_name: str) -> dict[str, Any]:
        loaded = read_plot_profile(source_path, profile_key, profile_name=profile_name)
        if loaded is None:
            raise ValueError(
                f"No saved profile '{profile_name}' found in '{source_path.name}' ({profile_key})."
            )
        load_args = deepcopy(args)
        for key, value in loaded.items():
            setattr(load_args, key, deepcopy(value))
        context = build_full_context(load_args)
        merged = _merge_gui_only_plot_settings(loaded, loaded)
        merged["series_descriptors"] = deepcopy(context.series_descriptors)
        merged["_profile_filter_options"] = deepcopy(context.profile_filter_options)
        return merged

    def _delete_profile(profile_name: str) -> tuple[str | None, str]:
        removed, active_profile = delete_named_plot_profile(
            source_path,
            profile_key,
            profile_name,
        )
        if not removed:
            raise ValueError(
                f"No saved profile '{profile_name}' found in '{source_path.name}' ({profile_key})."
            )
        if active_profile is None:
            return None, f"Deleted profile '{profile_name}' from '{source_path.name}'."
        return (
            active_profile,
            f"Deleted profile '{profile_name}' from '{source_path.name}'. "
            f"Active profile is now '{active_profile}'.",
        )

    def _set_active_profile(profile_name: str) -> str:
        set_active_plot_profile(source_path, profile_key, profile_name)
        return f"Selected profile '{profile_name}' in '{source_path.name}'."

    def _resolve_series_defaults(gui_settings: dict[str, Any]) -> dict[str, Any]:
        resolved_args = deepcopy(args)
        _apply_gui_settings_to_args(resolved_args, gui_settings)
        context = build_full_context(resolved_args)
        return {
            "series_count": context.series_count,
            "series_labels": list(context.default_series_labels),
            "series_descriptors": deepcopy(context.series_descriptors),
            "_profile_filter_options": deepcopy(context.profile_filter_options),
        }

    _open_plot_settings_gui(
        title=gui_title,
        initial_settings=initial_settings,
        on_preview=_preview,
        on_save=_save,
        on_save_figure=_save_figure,
        on_import_hdf5=_import_hdf5,
        analysis_name=analysis_name,
        on_resolve_series_defaults=_resolve_series_defaults,
        initial_profile_name=initial_profile_name,
        available_profile_names=available_profile_names,
        default_profile_settings=default_settings,
        on_load_profile=_load_profile if allow_named_profiles else None,
        on_delete_profile=_delete_profile if allow_named_profiles else None,
        on_set_active_profile=_set_active_profile if allow_named_profiles else None,
        allow_named_profiles=allow_named_profiles,
    )


def _handle_root_overview(_args: argparse.Namespace) -> int:
    author = _read_project_author(default="Unknown")
    print(
        "\n".join(
            [
                "LiNaK Command Center",
                "====================",
                f"Version      : {__version__}",
                f"Author       : {author}",
                "",
                "Core workflow",
                "  1) Compute analysis HDF5 from trajectory data",
                "     linak compute density /path/to/traj.xyz --species O --axis z",
                "  2) Plot from HDF5 only",
                "     linak plot /path/to/traj_density_o_z.h5",
                "",
                "Fast HDF5 plotting shorthand",
                (
                    "  linak plot /path/to/data.h5    "
                    "# auto-detects density/msd/rdf/position/coordination/potential from HDF5 metadata, "
                    f"or falls back to: linak {_TABULAR_COMMAND} plot ..."
                ),
                "",
                "Command groups",
                "  compute   trajectory -> HDF5",
                "  plot      LiNaK analysis HDF5 -> figure",
                (
                    f"  {_TABULAR_COMMAND:<8} inspect/transform/plot tabular HDF5 "
                    f"(aliases: {', '.join(_TABULAR_COMMAND_ALIASES)})"
                ),
                "  apply     trajectory transformations",
                "",
                "Need details?",
                "  linak <command> --help",
                "  linak compute --help",
                "  linak plot --help",
                f"  linak {_TABULAR_COMMAND} --help",
                "  linak apply --help",
            ]
        )
    )
    return 0


def _handle_plot_overview(_args: argparse.Namespace) -> int:
    print(
        "\n".join(
            [
                "LiNaK Plot Usage (HDF5-only)",
                "============================",
                "Plot accepts LiNaK density/MSD/RDF/position/coordination/potential HDF5 inputs and auto-detects the analysis.",
                "",
                "Examples",
                "  linak compute density /path/to/traj.xyz --species O --axis z",
                "  linak plot /path/to/traj_density_o_z.h5",
                "  linak plot -f run1_density.h5 run2_density.h5 --no-show --output density.png",
                "  linak plot /path/to/traj_msd_o.h5 --no-show --output msd.png",
                "  linak plot /path/to/traj_rdf_o_h.h5 --species-a O --species-b H",
                "  linak plot /path/to/traj_position_o_z.h5 --component distance",
                "  linak plot /path/to/traj_coordination_o_h.h5 --component distance",
                "  linak plot /path/to/potentials.h5",
                "",
                "Legacy syntax removed",
                "  linak plot density ...    # removed",
                "  linak plot msd ...        # removed",
                "  linak plot rdf ...        # removed",
                "  linak plot position ...   # removed",
                "  linak plot coordination ... # removed",
                "",
                "Generic HDF5 table plotting",
                "  linak plot /path/to/data.h5             # falls back to hdf5 plot when not LiNaK analysis",
                f"  linak {_TABULAR_COMMAND} plot /path/to/data.h5 --help",
            ]
        )
    )
    return 0


def _handle_compute_overview(_args: argparse.Namespace) -> int:
    print(
        "\n".join(
            [
                "LiNaK Compute Usage",
                "===================",
                "Compute commands read trajectory files and write HDF5 outputs.",
                "",
                "Examples",
                "  linak compute density /path/to/traj.xyz --species O --axis z",
                "  linak compute msd /path/to/traj.xyz --species O",
                "  linak compute position /path/to/traj.xyz --species O",
                "  linak compute rdf /path/to/traj.xyz --species-a O --species-b H",
                "  linak compute coordination /path/to/traj.xyz --species-a O --species-b H --cutoff-from-rdf",
                "  linak compute potential -f /path/to/*.cube",
                "",
                "Need command options?",
                "  linak compute density --help",
                "  linak compute msd --help",
                "  linak compute position --help",
                "  linak compute rdf --help",
                "  linak compute coordination --help",
                "  linak compute potential --help",
            ]
        )
    )
    return 0


def _handle_apply_overview(_args: argparse.Namespace) -> int:
    print(
        "\n".join(
            [
                "LiNaK Apply Usage",
                "=================",
                "Apply commands transform trajectory files and related simulation artifacts.",
                "",
                "Examples",
                "  linak apply pbc /path/to/traj.xyz --cell 10 10 10",
                "  linak apply compress /path/to/output.out",
                "",
                "Need command options?",
                "  linak apply pbc --help",
                "  linak apply compress --help",
            ]
        )
    )
    return 0


def _handle_csv_overview(_args: argparse.Namespace) -> int:
    alias_lines = [f"  linak {alias} ..." for alias in _TABULAR_COMMAND_ALIASES]
    print(
        "\n".join(
            [
                "LiNaK HDF5 Table Usage",
                "========================",
                "Inspect, transform, and plot tabular data from HDF5 files.",
                "",
                "Subcommands",
                "  interactive, info, preview, get, sort, filter, dedupe, combine, plot, plot-settings",
                "",
                "Examples",
                f"  linak {_TABULAR_COMMAND} preview -f /path/to/data.h5",
                f"  linak {_TABULAR_COMMAND} get /path/to/data.h5 --column value",
                (
                    f"  linak {_TABULAR_COMMAND} combine -f run1_density.h5 run2_density.h5 "
                    "-o combined_density.h5"
                ),
                (
                    f"  linak {_TABULAR_COMMAND} plot /path/to/data.h5 "
                    "--kind line --x step --y value"
                ),
                (
                    f"  linak {_TABULAR_COMMAND} plot-settings /path/to/data.h5 --profile auto --show-all"
                ),
                "",
                "Quick start",
                f"  linak {_TABULAR_COMMAND} interactive /path/to/data.h5",
            ]
            + (["", "Aliases", *alias_lines] if alias_lines else [])
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create and return the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="linak",
        description=(
            "LiNaK: modular molecular dynamics analysis toolkit. "
            "Start with `linak` for a guided overview."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path; logs are always shown in the terminal",
    )
    parser.set_defaults(handler=_handle_root_overview)

    commands = parser.add_subparsers(dest="command", required=False)

    plot_parser = commands.add_parser(
        "plot",
        help="Generate plots from precomputed LiNaK HDF5 data.",
        description=(
            "Plot LiNaK analysis HDF5 data by auto-detecting density, MSD, RDF, position, coordination, potential, or orientation from HDF5 "
            "metadata. If the input HDF5 is not a supported LiNaK analysis file, LiNaK falls "
            f"back to `{_TABULAR_COMMAND} plot`."
        ),
        epilog=(
            "Trajectory inputs are intentionally not supported here: run `linak compute ...` "
            "first. For generic tabular HDF5 plotting, use `linak hdf5 plot` directly."
        ),
    )
    plot_parser.set_defaults(handler=_handle_plot)
    _add_plot_common_options(plot_parser)
    _add_plot_source_options(
        plot_parser,
        help_text="LiNaK analysis HDF5 input (use `linak hdf5 plot` for generic tables)",
    )
    common_analysis_group = plot_parser.add_argument_group("Analysis selection filters")
    common_analysis_group.add_argument(
        "--species",
        default=None,
        help="Optional species override for density/MSD/position loaded profile labels (default: use file metadata)",
    )
    density_group = plot_parser.add_argument_group("Density plot options")
    density_group.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default=None,
        help=(
            "Optional axis override for loaded density/position/coordination profiles "
            "(default: use file metadata)"
        ),
    )
    density_group.add_argument(
        "--x-mode",
        choices=["distance", "axis"],
        default="distance",
        help="Density x-axis mode: distance to surface (default) or raw axis coordinate.",
    )
    density_group.add_argument(
        "--quantity",
        choices=["mass", "number"],
        default="mass",
        help="Density quantity to plot (default: mass; use number for atoms/A^3).",
    )
    rdf_group = plot_parser.add_argument_group("RDF plot options")
    rdf_group.add_argument(
        "--species-a",
        default=None,
        help="Optional first-species override for RDF/coordination profiles (default: use file metadata)",
    )
    rdf_group.add_argument(
        "--species-b",
        default=None,
        help="Optional second-species override for RDF/coordination profiles (default: use file metadata or species-a)",
    )
    position_group = plot_parser.add_argument_group("Position plot options")
    position_group.add_argument(
        "--component",
        choices=[
            "distance",
            "x",
            "y",
            "z",
            "xy-z",
            "time",
            "time-distance",
            "average",
            "density-weighted",
            "heatmap",
        ],
        default="distance",
        help=(
            "Plot component. Position supports distance/x/y/z and xy-z. "
            "Coordination supports distance, time, and time-distance. "
            "Orientation supports average, density-weighted, and heatmap."
        ),
    )
    position_group.add_argument(
        "--map-color",
        choices=["distance", "z"],
        default="distance",
        help=(
            "Color source for --component xy-z (default: distance). "
            "Ignored by time-axis components."
        ),
    )
    position_group.add_argument(
        "--time-axis",
        choices=["ps", "fs", "step", "frame"],
        default="ps",
        help="Time axis for position/coordination plots (default: ps).",
    )
    position_group.add_argument(
        "--time-section-width",
        type=_positive_float,
        default=None,
        help=(
            "Optional time-section width for display-only rebinning in position plots. "
            "Equivalent to x-bin width."
        ),
    )
    orientation_group = plot_parser.add_argument_group("Orientation plot options")
    orientation_group.add_argument(
        "--angle",
        choices=["polar", "azimuthal"],
        default="polar",
        help=(
            "Which angle component to plot for orientation analysis "
            "(default: polar). Ignored by non-orientation analyses."
        ),
    )
    _add_dry_run_option(plot_parser)

    compute_parser = commands.add_parser(
        "compute",
        help="Compute analysis data and save HDF5 outputs.",
        description=(
            "Compute analysis data from trajectories. Commands under `compute` write HDF5 outputs "
            "and do not plot by default."
        ),
    )
    compute_parser.set_defaults(handler=_handle_compute_overview)
    compute_commands = compute_parser.add_subparsers(dest="compute_command", required=False)

    compute_density = compute_commands.add_parser(
        "density",
        help="Compute 1D mass-density profile and save HDF5.",
    )
    compute_density.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(legacy positional form)"
        ),
    )
    compute_density.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_density.add_argument(
        "--species",
        default="all",
        help=(
            "Chemical symbol (e.g. O), H2O, or all resolved density series together "
            "(elements plus H2O when present; default: all)"
        ),
    )
    compute_density.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Axis for profile (default: z)",
    )
    compute_density.add_argument(
        "--bin-width",
        type=_positive_float,
        default=0.1,
        help="Histogram bin width in Angstrom (default: 0.1)",
    )
    compute_density.add_argument(
        "--surface-mode",
        choices=["auto", "layered", "rough"],
        default="auto",
        help=(
            "Surface detection mode (default: auto). "
            "'layered' uses top-layer mean; 'rough' uses low-mobility frame-wise mean."
        ),
    )
    compute_density.add_argument(
        "--surface-elements",
        nargs="+",
        metavar="ELEM",
        help=(
            "Optional element symbols used to detect the reference surface "
            "(default: automatic detection)."
        ),
    )
    compute_density.add_argument(
        "--include-fixed-surface-atoms",
        action="store_true",
        help=(
            "Allow atoms marked by ASE constraints to be used in surface detection "
            "(default: constrained atoms are excluded)."
        ),
    )
    compute_density.add_argument(
        "--rough-surface-envelope",
        type=_positive_float,
        default=None,
        help=(
            "Restrict rough-mode reference selection to atoms within this depth from the "
            "outer surface in Angstrom (default: adaptive)."
        ),
    )
    _add_cell_resolution_options(compute_density)
    compute_density.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5)",
    )
    _add_dry_run_option(compute_density)
    compute_density.set_defaults(handler=_handle_compute_density)

    compute_msd = compute_commands.add_parser(
        "msd",
        aliases=["MSD"],
        help="Compute MSD and save HDF5.",
    )
    compute_msd.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(legacy positional form)"
        ),
    )
    compute_msd.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_msd.add_argument(
        "--species",
        default="all",
        help="Species for MSD (default: all)",
    )
    compute_msd.add_argument(
        "--timestep-fs",
        type=_positive_float,
        default=None,
        help=(
            "Timestep between frames in fs "
            "(default: auto from metadata or simulation input; fallback 1.0)"
        ),
    )
    _add_cell_resolution_options(compute_msd)
    compute_msd.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5)",
    )
    _add_dry_run_option(compute_msd)
    compute_msd.set_defaults(handler=_handle_compute_msd, compute_command="msd")

    compute_position = compute_commands.add_parser(
        "position",
        help="Compute atom-resolved positions and save HDF5.",
    )
    compute_position.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(legacy positional form)"
        ),
    )
    compute_position.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_position.add_argument(
        "--species",
        default=None,
        help=(
            "Optional species to track (for example O). "
            "If omitted, LiNaK warns and writes one output per species."
        ),
    )
    compute_position.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Surface-reference axis for distance-to-surface (default: z)",
    )
    compute_position.add_argument(
        "--timestep-fs",
        type=_positive_float,
        default=None,
        help=(
            "Timestep between frames in fs "
            "(default: auto from metadata or simulation input; fallback 1.0)"
        ),
    )
    compute_position.add_argument(
        "--surface-mode",
        choices=["auto", "layered", "rough"],
        default="auto",
        help=(
            "Surface detection mode (default: auto). "
            "'layered' uses top-layer mean; 'rough' uses low-mobility frame-wise mean."
        ),
    )
    compute_position.add_argument(
        "--surface-elements",
        nargs="+",
        metavar="ELEM",
        help=(
            "Optional element symbols used to detect the reference surface "
            "(default: automatic detection)."
        ),
    )
    compute_position.add_argument(
        "--include-fixed-surface-atoms",
        action="store_true",
        help=(
            "Allow atoms marked by ASE constraints to be used in surface detection "
            "(default: constrained atoms are excluded)."
        ),
    )
    compute_position.add_argument(
        "--rough-surface-envelope",
        type=_positive_float,
        default=None,
        help=(
            "Restrict rough-mode reference selection to atoms within this depth from the "
            "outer surface in Angstrom (default: adaptive)."
        ),
    )
    _add_cell_resolution_options(compute_position)
    compute_position.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5; one file per species when needed)",
    )
    _add_dry_run_option(compute_position)
    compute_position.set_defaults(handler=_handle_compute_position, compute_command="position")

    compute_rdf = compute_commands.add_parser(
        "rdf",
        aliases=["RDF"],
        help="Compute RDF and save HDF5.",
    )
    compute_rdf.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(legacy positional form)"
        ),
    )
    compute_rdf.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_rdf.add_argument(
        "--species-a",
        default="all",
        help="First species for RDF (default: all)",
    )
    compute_rdf.add_argument(
        "--species-b",
        default=None,
        help="Second species for RDF (default: same as species-a)",
    )
    compute_rdf.add_argument(
        "--r-max",
        type=_positive_float,
        default=None,
        help="Maximum RDF radius in Angstrom (default: auto)",
    )
    compute_rdf.add_argument(
        "--bin-width",
        type=_positive_float,
        default=0.05,
        help="RDF bin width in Angstrom (default: 0.05)",
    )
    compute_rdf.add_argument(
        "--threads",
        type=int,
        default=None,
        help=("Number of threads for RDF compute (default: auto; set 1 to disable parallelism)"),
    )
    _add_cell_resolution_options(compute_rdf)
    compute_rdf.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5)",
    )
    _add_dry_run_option(compute_rdf)
    compute_rdf.set_defaults(handler=_handle_compute_rdf, compute_command="rdf")

    compute_coordination = compute_commands.add_parser(
        "coordination",
        aliases=["cn", "coord"],
        help="Compute continuous coordination numbers and save HDF5.",
    )
    compute_coordination.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(legacy positional form)"
        ),
    )
    compute_coordination.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_coordination.add_argument(
        "--species-a",
        default="all",
        help="Center species for coordination analysis (default: all)",
    )
    compute_coordination.add_argument(
        "--species-b",
        default=None,
        help="Neighbor species for coordination analysis (default: same as species-a)",
    )
    compute_coordination.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Surface-reference axis for distance-to-surface (default: z)",
    )
    compute_coordination.add_argument(
        "--timestep-fs",
        type=_positive_float,
        default=None,
        help=(
            "Timestep between frames in fs "
            "(default: auto from metadata or simulation input; fallback 1.0)"
        ),
    )
    compute_coordination.add_argument(
        "--surface-mode",
        choices=["auto", "layered", "rough"],
        default="auto",
        help=(
            "Surface detection mode (default: auto). "
            "'layered' uses top-layer mean; 'rough' uses low-mobility frame-wise mean."
        ),
    )
    compute_coordination.add_argument(
        "--surface-elements",
        nargs="+",
        metavar="ELEM",
        help=(
            "Optional element symbols used to detect the reference surface "
            "(default: automatic detection)."
        ),
    )
    compute_coordination.add_argument(
        "--include-fixed-surface-atoms",
        action="store_true",
        help=(
            "Allow atoms marked by ASE constraints to be used in surface detection "
            "(default: constrained atoms are excluded)."
        ),
    )
    compute_coordination.add_argument(
        "--rough-surface-envelope",
        type=_positive_float,
        default=None,
        help=(
            "Restrict rough-mode reference selection to atoms within this depth from the "
            "outer surface in Angstrom (default: adaptive)."
        ),
    )
    compute_coordination.add_argument(
        "--cutoff",
        type=_positive_float,
        default=None,
        help="Direct coordination cutoff in Angstrom (highest priority cutoff source).",
    )
    compute_coordination.add_argument(
        "--cutoff-rdf",
        default=None,
        help="Use an existing RDF HDF5 file to determine the coordination cutoff.",
    )
    compute_coordination.add_argument(
        "--cutoff-from-rdf",
        action="store_true",
        help=(
            "Recompute an average RDF from sampled frames and determine the coordination cutoff "
            "(used automatically when neither --cutoff nor --cutoff-rdf is provided)."
        ),
    )
    compute_coordination.add_argument(
        "--cutoff-smoothing-width",
        type=_positive_float,
        default=0.20,
        help="Width of the cosine taper around the cutoff in Angstrom (default: 0.20).",
    )
    _add_cell_resolution_options(compute_coordination)
    compute_coordination.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5)",
    )
    _add_dry_run_option(compute_coordination)
    compute_coordination.set_defaults(
        handler=_handle_compute_coordination,
        compute_command="coordination",
    )

    compute_potential = compute_commands.add_parser(
        "potential",
        help="Compute CP2K electrode cSHE potentials from Hartree cube files and save HDF5.",
        description=(
            "Compute CP2K cSHE from Hartree cube files. "
            "For each cube: parse E_Fermi from nearby output (.out), "
            "compute water-bulk potential from O/H z-bounds in the cube header, "
            "and report U_cSHE = V_bulk - E_F - offset."
        ),
        epilog=(
            "Examples:\n"
            "  linak compute potential /path/to/*-v_hartree-1_0.cube\n"
            "  linak compute potential -f run1/*-v_hartree-1_0.cube run2/*-v_hartree-1_0.cube "
            "--output potentials.h5\n"
            "  linak compute potential -f *.cube --threads 4 --water-padding-ang 4.0"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    input_group = compute_potential.add_argument_group("Input")
    input_group.add_argument(
        "source",
        nargs="*",
        help="Hartree cube file path(s) (legacy positional form).",
    )
    input_group.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="One or more Hartree cube files.",
    )
    analysis_group = compute_potential.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--water-padding-ang",
        type=_non_negative_float,
        default=5.0,
        help="Padding removed from O/H z-bounds for water-bulk averaging (default: 5.0).",
    )
    analysis_group.add_argument(
        "--cshe-offset-ev",
        type=float,
        default=0.81,
        help="cSHE offset in eV applied as U_cSHE = V_bulk - E_F - offset (default: 0.81).",
    )
    execution_group = compute_potential.add_argument_group("Execution")
    execution_group.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Number of threads for potential compute "
            "(default: auto=1; increase only if benchmarking shows a gain)"
        ),
    )
    execution_group.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any source fails or yields incomplete cSHE data.",
    )
    execution_group.add_argument(
        "--include-failures",
        dest="include_failures",
        action="store_true",
        default=True,
        help="Include failed sources as status=error rows in the HDF5 output (default: enabled).",
    )
    execution_group.add_argument(
        "--no-include-failures",
        dest="include_failures",
        action="store_false",
        help="Do not write failed sources to HDF5.",
    )
    output_group = compute_potential.add_argument_group("Output HDF5")
    output_group.add_argument(
        "--append",
        dest="append",
        action="store_true",
        default=True,
        help="Append to existing HDF5 when schema is compatible (default: enabled).",
    )
    output_group.add_argument(
        "--no-append",
        dest="append",
        action="store_false",
        help="Do not append to an existing HDF5 file (a new fallback file is created unless --overwrite is used).",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the HDF5 output path if it already exists.",
    )
    output_group.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5).",
    )
    _add_dry_run_option(compute_potential)
    compute_potential.set_defaults(handler=_handle_compute_potential)

    # ── compute orientation ───────────────────────────────────────────
    compute_orientation = compute_commands.add_parser(
        "orientation",
        help="Compute H2O orientation vs distance-to-surface and save HDF5.",
    )
    compute_orientation.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(legacy positional form)"
        ),
    )
    compute_orientation.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_orientation.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Spatial axis for distance binning (default: z).",
    )
    compute_orientation.add_argument(
        "--reference-axis",
        choices=["x", "y", "z"],
        default="z",
        help=(
            "Axis treated as the surface normal for angle computation (default: z). "
            "The polar angle (cos(theta)) is measured between the water bisector and this axis."
        ),
    )
    compute_orientation.add_argument(
        "--bin-width",
        type=_positive_float,
        default=0.01,
        help="Distance histogram bin width in Angstrom (default: 0.01).",
    )
    compute_orientation.add_argument(
        "--angle-bins",
        type=int,
        default=100,
        help="Number of cos(angle) bins for heatmaps over [-1, +1] (default: 100).",
    )
    compute_orientation.add_argument(
        "--surface-mode",
        choices=["auto", "layered", "rough"],
        default="auto",
        help=(
            "Surface detection mode (default: auto). "
            "'layered' uses top-layer mean; 'rough' uses low-mobility frame-wise mean."
        ),
    )
    compute_orientation.add_argument(
        "--surface-elements",
        nargs="+",
        metavar="ELEM",
        help="Element symbols used to detect the reference surface (default: auto).",
    )
    compute_orientation.add_argument(
        "--include-fixed-surface-atoms",
        action="store_true",
        help=(
            "Allow atoms marked by ASE constraints to be used in surface detection "
            "(default: constrained atoms are excluded)."
        ),
    )
    compute_orientation.add_argument(
        "--rough-surface-envelope",
        type=_positive_float,
        default=None,
        help=(
            "Restrict rough-mode reference selection to atoms within this depth from the "
            "outer surface in Angstrom (default: adaptive)."
        ),
    )
    compute_orientation.add_argument(
        "--oh-cutoff",
        type=_positive_float,
        default=1.25,
        help="O-H cutoff in Angstrom for water-molecule detection (default: 1.25).",
    )
    _add_cell_resolution_options(compute_orientation)
    compute_orientation.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5).",
    )
    _add_dry_run_option(compute_orientation)
    compute_orientation.set_defaults(handler=_handle_compute_orientation)

    apply_parser = commands.add_parser(
        "apply",
        help="Apply transformations to trajectory files.",
        description="Apply post-processing transforms to trajectories and CP2K output artifacts.",
    )
    apply_parser.set_defaults(handler=_handle_apply_overview)
    apply_commands = apply_parser.add_subparsers(dest="apply_command", required=False)

    apply_pbc = apply_commands.add_parser(
        "pbc",
        help="Apply periodic boundary conditions by wrapping positions into a cell.",
        description=(
            "Apply orthorhombic PBC to a trajectory and wrap atom positions. "
            "Cell dimensions are resolved in this order: --cell, --input, "
            "or automatic .inp/.lmp simulation-input discovery in the output directory."
        ),
    )
    apply_pbc.add_argument(
        "trajectory", nargs="?", help="Input trajectory path (legacy positional form)"
    )
    apply_pbc.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    output_group = apply_pbc.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        help="Output trajectory path (default: auto-generated next to input)",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input trajectory in place",
    )
    apply_pbc.add_argument(
        "-i",
        "--input",
        "--cp2k-input",
        "--lammps-input",
        dest="input",
        help=(
            "Path to simulation input file (.inp for CP2K, .lmp for LAMMPS). "
            "If omitted, LiNaK searches for one .inp/.lmp file in the output directory."
        ),
    )
    apply_pbc.add_argument(
        "--cell",
        nargs=3,
        type=_positive_float,
        metavar=("A", "B", "C"),
        help="Explicit orthorhombic cell lengths in Angstrom (overrides auto-discovery).",
    )
    _add_dry_run_option(apply_pbc)
    apply_pbc.set_defaults(handler=_handle_apply_pbc)

    from .storage.compress import DROP_SECTION_CHOICES

    apply_compress = apply_commands.add_parser(
        "compress",
        help="Compress CP2K output into structured files and back up the raw .out source.",
        description=(
            "Extract key CP2K data from one output file into a compact directory, then move the "
            "original raw .out into a backup directory. This keeps analysis-friendly files near the "
            "run while preserving the full source output."
        ),
        epilog=(
            "What `linak apply compress` creates\n"
            "  <stem>/README.txt          human-readable generated/skipped file report\n"
            "  <stem>/manifest.json       machine-readable file/row metadata\n"
            "  <stem>/summary.txt         compact CP2K run summary\n"
            "  <stem>/*.csv               parsed tables (SCF, charges, forces, MD, ...)\n"
            "  <stem>/*.txt               setup, warnings, timing, and performance snippets\n"
            "  <backup-dir>/<unique>.out  moved original CP2K output\n"
            "  <backup-dir>/<unique>.out.meta.json  source/backup/output linkage metadata\n\n"
            "Defaults\n"
            "  backup dir: <input-dir>/.linak_backups\n"
            "  output dir: <input-stem> (auto-suffixed if already present)\n\n"
            "Examples\n"
            "  linak apply compress /path/to/output.out\n"
            "  linak apply compress /path/to/output.out --backup-dir ./private_backups\n"
            "  linak apply compress /path/to/output.out --drop mulliken hirshfeld\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    apply_compress.add_argument(
        "output_file",
        nargs="?",
        metavar="OUTPUT_OUT",
        help="Input CP2K output file path (legacy positional form).",
    )
    apply_compress.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "CP2K output file path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    apply_compress.add_argument(
        "--backup-dir",
        metavar="PATH",
        help=(
            "Backup directory for the moved raw .out file "
            "(default: hidden .linak_backups next to input)."
        ),
    )
    apply_compress.add_argument(
        "--drop",
        nargs="+",
        choices=list(DROP_SECTION_CHOICES),
        metavar="SECTION",
        help=("Optional outputs to skip. Choices: " + ", ".join(DROP_SECTION_CHOICES) + "."),
    )
    _add_dry_run_option(apply_compress)
    apply_compress.set_defaults(handler=_handle_apply_compress)

    csv_parser = commands.add_parser(
        _TABULAR_COMMAND,
        aliases=list(_TABULAR_COMMAND_ALIASES),
        help="Inspect, query, transform, and plot tabular HDF5 data.",
        description=(
            "Work with generic tabular datasets in HDF5 files. "
            "Most subcommands are semi-interactive: if required choices such as columns are omitted, "
            "LiNaK prompts with available options. "
            f"Short alias: {', '.join(_TABULAR_COMMAND_ALIASES)}."
        ),
    )
    csv_parser.set_defaults(handler=_handle_csv_overview)
    csv_commands = csv_parser.add_subparsers(dest="csv_command", required=False)

    csv_interactive = csv_commands.add_parser(
        "interactive",
        help="Interactive HDF5 assistant for one file.",
    )
    _add_csv_source_options(csv_interactive)
    csv_interactive.add_argument(
        "--rows",
        type=_positive_int,
        default=8,
        help="Preview rows shown at startup (default: 8)",
    )
    csv_interactive.set_defaults(handler=_handle_csv_interactive)

    csv_info = csv_commands.add_parser(
        "info",
        help="Show HDF5 table shape, inferred types, and data-quality summary.",
    )
    _add_csv_source_options(csv_info)
    csv_info.set_defaults(handler=_handle_csv_info)

    csv_preview = csv_commands.add_parser(
        "preview",
        help="Print a head/tail preview of HDF5 table rows.",
    )
    _add_csv_source_options(csv_preview)
    csv_preview.add_argument(
        "--rows",
        type=_positive_int,
        default=10,
        help="Rows to preview (default: 10)",
    )
    csv_preview.add_argument(
        "--tail",
        action="store_true",
        help="Show final rows instead of first rows",
    )
    csv_preview.add_argument(
        "--show-index",
        action="store_true",
        help="Include row index in preview output",
    )
    csv_preview.set_defaults(handler=_handle_csv_preview)

    csv_get = csv_commands.add_parser(
        "get",
        help="Compute useful statistics for one or more columns.",
    )
    _add_csv_source_options(csv_get)
    csv_get.add_argument(
        "--column",
        nargs="+",
        help="Column(s) to analyze. If omitted, LiNaK prompts interactively.",
    )
    csv_get.add_argument(
        "--all-columns",
        action="store_true",
        help="Compute statistics for every column.",
    )
    csv_get.add_argument(
        "--metric",
        nargs="+",
        help=(
            "Optional metric subset. Common metrics: count missing distinct min max "
            "mean median std sum q05 q25 q75 q95 iqr mode mode_count numeric_ratio."
        ),
    )
    csv_get.add_argument(
        "--round",
        dest="round_digits",
        type=_positive_int,
        default=6,
        help="Significant digits for floating-point output (default: 6)",
    )
    csv_get.set_defaults(handler=_handle_csv_get)

    csv_sort = csv_commands.add_parser(
        "sort",
        help="Sort HDF5 table rows by one or more columns and write a new HDF5 file.",
    )
    _add_csv_source_options(csv_sort)
    csv_sort.add_argument(
        "--by",
        nargs="+",
        help="Sort key column(s). If omitted, LiNaK prompts interactively.",
    )
    csv_sort.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order",
    )
    csv_sort.add_argument(
        "--na-position",
        choices=["first", "last"],
        default="last",
        help="Placement of missing values in sorted output (default: last)",
    )
    csv_sort.add_argument(
        "--mode",
        choices=["auto", "numeric", "string"],
        default="auto",
        help="Sort mode for key columns (default: auto)",
    )
    _add_csv_write_options(csv_sort)
    _add_dry_run_option(csv_sort)
    csv_sort.set_defaults(handler=_handle_csv_sort)

    csv_filter = csv_commands.add_parser(
        "filter",
        help="Filter rows with numeric/text predicates and write HDF5 output.",
    )
    _add_csv_source_options(csv_filter)
    csv_filter.add_argument(
        "--column",
        help="Column used for filtering. If omitted, LiNaK prompts interactively.",
    )
    csv_filter.add_argument(
        "--op",
        "--operator",
        dest="operator",
        choices=[
            "eq",
            "ne",
            "gt",
            "ge",
            "lt",
            "le",
            "contains",
            "startswith",
            "endswith",
            "regex",
            "in",
            "not-in",
        ],
        help="Filter operator",
    )
    csv_filter.add_argument(
        "--value",
        help="Filter value (for in/not-in use comma-separated values)",
    )
    csv_filter.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive text matching",
    )
    csv_filter.add_argument(
        "--invert",
        action="store_true",
        help="Invert filter selection",
    )
    _add_csv_write_options(csv_filter)
    _add_dry_run_option(csv_filter)
    csv_filter.set_defaults(handler=_handle_csv_filter)

    csv_dedupe = csv_commands.add_parser(
        "dedupe",
        help="Drop duplicate rows and write HDF5 output.",
    )
    _add_csv_source_options(csv_dedupe)
    csv_dedupe.add_argument(
        "--subset",
        nargs="+",
        help="Subset columns for duplicate detection (default: all columns)",
    )
    csv_dedupe.add_argument(
        "--keep",
        choices=["first", "last", "none"],
        default="first",
        help="Which duplicate occurrence to keep (default: first)",
    )
    _add_csv_write_options(csv_dedupe)
    _add_dry_run_option(csv_dedupe)
    csv_dedupe.set_defaults(handler=_handle_csv_dedupe)

    csv_combine = csv_commands.add_parser(
        "combine",
        help="Combine multiple LiNaK analysis HDF5 files into one multi-profile HDF5.",
        description=(
            "Combine multiple density/MSD/RDF/position/coordination LiNaK HDF5 files into one combined HDF5 file "
            "that can be plotted directly with `linak plot /path/to/combined.h5`."
        ),
    )
    csv_combine.add_argument(
        "source",
        nargs="*",
        metavar="SOURCE",
        help="Input HDF5 file path(s) (legacy positional form; use -f/--files for multiple)",
    )
    csv_combine.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="Input HDF5 file path(s). Use -f/--files even for one file; required for multiple.",
    )
    csv_combine.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Output combined HDF5 path (default: auto-generated next to first input).",
    )
    csv_combine.add_argument(
        "--settings-source",
        metavar="PATH_OR_INDEX",
        default=None,
        help=(
            "Input used as plot-settings source when multiple files are provided "
            "(default: first input). Accepts a 1-based index or one of the input paths."
        ),
    )
    _add_dry_run_option(csv_combine)
    csv_combine.set_defaults(handler=_handle_csv_combine)

    csv_plot = csv_commands.add_parser(
        "plot",
        help="Plot HDF5 table columns with line/scatter/bar/hist/box charts.",
        description=(
            "Plot one or more HDF5 files in a single figure.\n\n"
            "Source rules:\n"
            "  - Single file: SOURCE or -f FILE\n"
            "  - Multiple files: -f FILE1 FILE2 ... (required)"
        ),
        epilog=(
            "Examples:\n"
            f"  linak {_TABULAR_COMMAND} plot data.h5 --kind line --x step --y value\n"
            f"  linak {_TABULAR_COMMAND} plot -f run1.h5 run2.h5 "
            "--x step --y value --labels run1 run2\n"
            f"  linak {_TABULAR_COMMAND} plot data.h5 --kind hist --y energy --bins 50\n"
            "Tip: if --x/--y is omitted, LiNaK previews the data and prompts interactively."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_csv_plot_source_options(csv_plot)
    selection_group = csv_plot.add_argument_group("Data selection")
    selection_group.add_argument(
        "--kind",
        choices=["line", "scatter", "bar", "hist", "box"],
        default="line",
        help="Plot type (default: line)",
    )
    selection_group.add_argument(
        "--x",
        help="X-axis column for line/scatter/bar plots",
    )
    selection_group.add_argument(
        "--y",
        nargs="+",
        help=(
            "Y-axis column(s) for line/scatter/bar, or numeric column(s) for hist/box. "
            "If omitted, LiNaK prompts interactively."
        ),
    )
    selection_group.add_argument(
        "--bins",
        type=_positive_int,
        default=30,
        help="Number of bins for histogram plots (default: 30)",
    )
    _add_csv_plot_options(csv_plot)
    _add_dry_run_option(csv_plot)
    csv_plot.set_defaults(handler=_handle_csv_plot)

    csv_plot_settings = csv_commands.add_parser(
        "plot-settings",
        help="Inspect/edit/copy persisted plot-setting profiles stored in HDF5 files.",
        description=(
            "Manage saved plot-setting profiles in LiNaK HDF5 files.\n\n"
            "Profiles:\n"
            "  - plot:density\n"
            "  - plot:msd\n"
            "  - plot:rdf\n"
            "  - plot:position\n"
            "  - plot:coordination\n"
            "  - plot:potential\n"
            "  - plot:table\n\n"
            "Use this command to inspect settings, set/unset individual keys, "
            "import/export profiles between files, or delete stale profiles."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    csv_plot_settings.add_argument(
        "source",
        nargs="?",
        metavar="SOURCE",
        help="Target HDF5 file (legacy positional form)",
    )
    csv_plot_settings.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="Target HDF5 file path (single file expected).",
    )
    csv_plot_settings.add_argument(
        "--profile",
        choices=["auto", "density", "msd", "rdf", "position", "coordination", "potential", "table"],
        default="auto",
        help=(
            "Profile to target. 'auto' resolves from HDF5 analysis metadata; "
            "if unresolved, it falls back to the table profile."
        ),
    )
    csv_plot_settings.add_argument(
        "--name",
        metavar="NAME",
        help=(
            "Named saved profile inside the selected analysis profile key. "
            "Defaults to the active saved profile."
        ),
    )
    csv_plot_settings.add_argument(
        "--set",
        nargs="+",
        metavar="KEY=VALUE",
        help=(
            "Set one or more settings using dotted keys. "
            "VALUE accepts JSON (for example: axis.x_lim=[0,10])."
        ),
    )
    csv_plot_settings.add_argument(
        "--unset",
        nargs="+",
        metavar="KEY",
        help="Remove one or more dotted-key settings from the selected profile.",
    )
    csv_plot_settings.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Delete the selected named profile when --name is given; otherwise delete "
            "the whole analysis profile key from the HDF5 file."
        ),
    )
    csv_plot_settings.add_argument(
        "--copy-name",
        metavar="NAME",
        help="Duplicate the selected named profile inside SOURCE under a new NAME.",
    )
    csv_plot_settings.add_argument(
        "--set-active",
        metavar="NAME",
        help="Set the active named profile for the selected analysis profile key.",
    )
    csv_plot_settings.add_argument(
        "--import-from",
        metavar="PATH",
        help="Import the selected profile from another HDF5 file into SOURCE.",
    )
    csv_plot_settings.add_argument(
        "--export-to",
        nargs="+",
        metavar="PATH",
        help="Apply the selected profile from SOURCE to one or more target HDF5 files.",
    )
    csv_plot_settings.add_argument(
        "--show-all",
        action="store_true",
        help="List every saved profile key in SOURCE instead of printing one profile payload.",
    )
    csv_plot_settings.set_defaults(handler=_handle_csv_plot_settings)

    return parser


def _resolve_metric_selection(metrics: list[str] | None) -> list[str] | None:
    if metrics is None:
        return None
    return [metric.strip().lower() for metric in metrics if metric.strip()]


def _parse_plot_setting_assignment(token: str) -> tuple[str, Any]:
    if "=" not in token:
        raise ValueError(f"Expected KEY=VALUE assignment, got '{token}'.")
    key, raw_value = token.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ValueError(f"Invalid assignment '{token}': key cannot be empty.")
    if not raw_value:
        raise ValueError(f"Invalid assignment '{token}': value cannot be empty.")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def _handle_csv_plot_settings(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 plot-settings management.")
    source = _resolve_single_source_argument(
        args,
        positional_attr="source",
        source_label="HDF5 input file",
    )
    source_path = Path(source).expanduser().resolve()
    profile_key = _resolve_plot_profile_key(
        profile_token=args.profile,
        source_path=source_path,
    )

    from .plot.plot_settings import (
        copy_plot_profile,
        delete_plot_profile,
        delete_named_plot_profile,
        read_active_plot_profile_name,
        read_plot_profile,
        read_plot_profile_stores,
        set_active_plot_profile,
        supports_named_plot_profiles,
        write_plot_profile,
    )

    named_profiles_supported = supports_named_plot_profiles(source_path)
    selected_name = getattr(args, "name", None)
    if not named_profiles_supported and selected_name not in {None, "Default"}:
        raise ValueError(
            "Combined HDF5 plot settings use one fixed profile 'Default'; named profiles are unsupported."
        )
    if not named_profiles_supported and args.copy_name is not None:
        raise ValueError("Combined HDF5 plot settings do not support creating named copies.")
    if not named_profiles_supported and args.set_active is not None:
        raise ValueError("Combined HDF5 plot settings always use the fixed profile 'Default'.")
    active_name = read_active_plot_profile_name(source_path, profile_key)
    resolved_name = selected_name or active_name

    if args.delete:
        if selected_name is None:
            removed = delete_plot_profile(source_path, profile_key)
            print(
                f"Removed plot profile '{profile_key}' from {source_path}"
                if removed
                else f"No plot profile '{profile_key}' found in {source_path}"
            )
        else:
            removed, active_after_delete = delete_named_plot_profile(
                source_path,
                profile_key,
                selected_name,
            )
            if removed:
                if active_after_delete is None:
                    print(
                        f"Removed named profile '{selected_name}' from {profile_key} in {source_path}"
                    )
                else:
                    print(
                        f"Removed named profile '{selected_name}' from {profile_key} in {source_path}; "
                        f"active profile is now '{active_after_delete}'"
                    )
            else:
                print(
                    f"No named profile '{selected_name}' found in {profile_key} for {source_path}"
                )

    if args.set_active is not None:
        set_active_plot_profile(source_path, profile_key, args.set_active)
        print(f"Selected named profile '{args.set_active}' for {profile_key} in {source_path}")

    if args.import_from is not None:
        import_path = Path(args.import_from).expanduser().resolve()
        copy_plot_profile(
            import_path,
            source_path,
            source_key=profile_key,
            target_key=profile_key,
            source_name=selected_name,
            target_name=selected_name,
        )
        if selected_name is None:
            print(f"Imported plot profile '{profile_key}' from {import_path} into {source_path}")
        else:
            print(
                f"Imported named profile '{selected_name}' for {profile_key} from {import_path} into {source_path}"
            )

    if args.copy_name is not None:
        if resolved_name is None:
            raise ValueError(
                f"No active named profile found for '{profile_key}' in '{source_path}'. Use --name or create one first."
            )
        copy_plot_profile(
            source_path,
            source_path,
            source_key=profile_key,
            target_key=profile_key,
            source_name=resolved_name,
            target_name=args.copy_name,
        )
        print(
            f"Copied named profile '{resolved_name}' to '{args.copy_name}' for {profile_key} in {source_path}"
        )

    if args.set or args.unset:
        current = (
            read_plot_profile(
                source_path,
                profile_key,
                profile_name=selected_name,
            )
            or {}
        )
        for assignment in args.set or []:
            key, value = _parse_plot_setting_assignment(assignment)
            _set_nested_setting(current, key, value)
        for dotted in args.unset or []:
            _delete_nested_setting(current, dotted)
        write_plot_profile(
            source_path,
            profile_key,
            current,
            profile_name=selected_name,
        )
        if selected_name is None:
            print(f"Updated plot profile '{profile_key}' in {source_path}")
        else:
            print(f"Updated named profile '{selected_name}' for {profile_key} in {source_path}")

    if args.export_to:
        for raw_target in args.export_to:
            target_path = Path(raw_target).expanduser().resolve()
            if target_path == source_path:
                continue
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Cannot export plot settings: target HDF5 does not exist: {target_path}"
                )
            copy_plot_profile(
                source_path,
                target_path,
                source_key=profile_key,
                target_key=profile_key,
                source_name=selected_name,
                target_name=selected_name,
            )
            if selected_name is None:
                print(f"Applied plot profile '{profile_key}' to {target_path}")
            else:
                print(f"Applied named profile '{selected_name}' for {profile_key} to {target_path}")

    if args.show_all:
        all_profiles = read_plot_profile_stores(source_path)
        if not all_profiles:
            print(f"No saved plot-setting profiles found in {source_path}")
        else:
            print(f"Saved plot-setting profiles in {source_path}:")
            for key in sorted(all_profiles):
                store = all_profiles[key]
                active_marker = store.active_profile or "<none>"
                names = ", ".join(store.profiles.keys())
                print(f"  - {key} [active: {active_marker}]")
                print(f"    names: {names}")
        LOGGER.info("HDF5 plot-settings management finished in %.2f s.", perf_counter() - start)
        return 0

    active_name = read_active_plot_profile_name(source_path, profile_key)
    resolved_name = selected_name or active_name
    selected_profile = read_plot_profile(
        source_path,
        profile_key,
        profile_name=selected_name,
    )
    print("HDF5 plot-settings")
    print(f"Source file         : {source_path}")
    print(f"Requested profile   : {args.profile}")
    print(f"Resolved profile key: {profile_key}")
    print(f"Requested name      : {selected_name if selected_name is not None else '<active>'}")
    print(f"Resolved name       : {resolved_name if resolved_name is not None else '<none>'}")
    if selected_profile is None:
        print("Profile payload     : <none>")
        print("Tip                 : use --set KEY=VALUE to create/update this profile.")
    else:
        print("Profile payload (JSON)")
        print(json.dumps(selected_profile, indent=2, sort_keys=True))

    LOGGER.info("HDF5 plot-settings management finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_csv_info(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 info.")

    from .storage.csv_tools import format_profiles_table, infer_numeric_columns, profile_columns

    frame, source_path = _load_csv_frame(args)
    profiles = profile_columns(frame)
    numeric_columns = infer_numeric_columns(frame)

    print(f"HDF5 file: {source_path}")
    _print_hdf5_metadata_overview(frame)
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(
        f"Numeric-like columns ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'none'}"
    )
    print("")
    print(format_profiles_table(profiles))

    LOGGER.info("HDF5 info finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_csv_preview(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 preview.")

    from .storage.csv_tools import format_frame_preview

    frame, source_path = _load_csv_frame(args)
    preview = format_frame_preview(
        frame,
        rows=args.rows,
        tail=args.tail,
        show_index=args.show_index,
    )
    print(f"Preview: {source_path} ({'tail' if args.tail else 'head'} {args.rows})")
    _print_hdf5_metadata_overview(frame)
    print(preview)

    LOGGER.info("HDF5 preview finished in %.2f s.", perf_counter() - start)
    return 0


def _resolve_get_columns(args: argparse.Namespace, frame: Any) -> list[str]:
    if args.all_columns:
        return list(frame.columns)
    if args.column:
        return _validate_csv_columns(frame, args.column)
    return _prompt_for_columns(
        columns=list(frame.columns),
        prompt="Select column(s) to analyze",
        allow_multiple=True,
    )


def _handle_csv_get(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 statistics.")

    from .storage.csv_tools import compute_column_statistics

    frame, source_path = _load_csv_frame(args)
    print(f"HDF5 file: {source_path}")
    _print_hdf5_metadata_overview(frame)
    if not args.all_columns and not args.column and _interactive_prompts_available():
        _print_csv_preview_for_interactive(frame=frame, source_path=source_path)
    columns = _resolve_get_columns(args, frame)
    metrics = _resolve_metric_selection(args.metric)

    blocks: list[str] = []
    for column in columns:
        stats = compute_column_statistics(frame, column)
        blocks.append(
            _format_column_statistics(
                stats,
                digits=args.round_digits,
                metrics=metrics,
            )
        )
    print("\n\n".join(blocks))

    LOGGER.info(
        "HDF5 statistics finished in %.2f s for %d column(s).", perf_counter() - start, len(columns)
    )
    return 0


def _resolve_sort_columns(args: argparse.Namespace, frame: Any) -> list[str]:
    if args.by:
        return _validate_csv_columns(frame, args.by)
    return _prompt_for_columns(
        columns=list(frame.columns),
        prompt="Select sort column(s)",
        allow_multiple=True,
    )


def _handle_csv_sort(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 sort.")

    from .storage.csv_tools import sort_frame
    from .storage.hdf5_table import write_hdf5_frame

    output_path = _resolve_csv_output_path(args, suffix="sorted")
    if args.dry_run:
        sort_columns_label = (
            ", ".join(args.by) if args.by else "interactive (resolved at execution)"
        )
        plan = [
            f"source: {Path(args.source).expanduser().resolve()}",
            "rows/columns: not inspected in dry-run",
            f"sort columns: {sort_columns_label}",
            f"descending: {'yes' if args.descending else 'no'}",
            f"mode: {args.mode}, na_position: {args.na_position}",
            f"output: {output_path}",
        ]
        _log_dry_run_plan(f"{_TABULAR_COMMAND} sort", plan)
        LOGGER.info("HDF5 sort dry run finished in %.2f s.", perf_counter() - start)
        return 0

    frame, source_path = _load_csv_frame(args)
    if not args.by and _interactive_prompts_available():
        _print_csv_preview_for_interactive(frame=frame, source_path=source_path)
    sort_columns = _resolve_sort_columns(args, frame)

    sorted_frame = sort_frame(
        frame,
        columns=sort_columns,
        descending=args.descending,
        na_position=args.na_position,
        mode=args.mode,
    )
    source_info = frame.attrs.get("linak_hdf5_source_info")
    written = write_hdf5_frame(sorted_frame, output_path, source_info=source_info)
    print(f"Wrote sorted HDF5: {written}")

    LOGGER.info("HDF5 sort finished in %.2f s.", perf_counter() - start)
    return 0


def _resolve_filter_inputs(args: argparse.Namespace, frame: Any) -> tuple[str, str, str]:
    column = args.column
    if column is None:
        column = _prompt_for_columns(
            columns=list(frame.columns),
            prompt="Select filter column",
            allow_multiple=False,
        )[0]
    elif column not in frame.columns:
        raise ValueError(f"Unknown column '{column}'.")

    operator = args.operator
    allowed = {
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "contains",
        "startswith",
        "endswith",
        "regex",
        "in",
        "not-in",
    }
    if operator is None:
        operator = _prompt_for_value(
            "Operator (eq/ne/gt/ge/lt/le/contains/startswith/endswith/regex/in/not-in)",
            allowed=allowed,
        ).lower()
    value = args.value if args.value is not None else _prompt_for_value("Filter value")
    return column, operator, value


def _handle_csv_filter(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 filter.")

    from .storage.csv_tools import filter_frame
    from .storage.hdf5_table import write_hdf5_frame

    output_path = _resolve_csv_output_path(args, suffix="filtered")

    if args.dry_run:
        filter_column = args.column or "interactive (resolved at execution)"
        filter_operator = args.operator or "interactive (resolved at execution)"
        filter_value = (
            args.value if args.value is not None else "interactive (resolved at execution)"
        )
        plan = [
            f"source: {Path(args.source).expanduser().resolve()}",
            "rows/columns: not inspected in dry-run",
            f"filter: {filter_column} {filter_operator} {filter_value}",
            f"case_sensitive: {'yes' if args.case_sensitive else 'no'}",
            f"invert: {'yes' if args.invert else 'no'}",
            f"output: {output_path}",
        ]
        _log_dry_run_plan(f"{_TABULAR_COMMAND} filter", plan)
        LOGGER.info("HDF5 filter dry run finished in %.2f s.", perf_counter() - start)
        return 0

    frame, source_path = _load_csv_frame(args)
    if (
        args.column is None or args.operator is None or args.value is None
    ) and _interactive_prompts_available():
        _print_csv_preview_for_interactive(frame=frame, source_path=source_path)
    column, operator, value = _resolve_filter_inputs(args, frame)

    filtered_frame = filter_frame(
        frame,
        column=column,
        operator=operator,
        value=value,
        case_sensitive=args.case_sensitive,
        invert=args.invert,
    )
    source_info = frame.attrs.get("linak_hdf5_source_info")
    written = write_hdf5_frame(filtered_frame, output_path, source_info=source_info)
    print(f"Rows kept: {len(filtered_frame)} / {len(frame)}")
    print(f"Wrote filtered HDF5: {written}")

    LOGGER.info("HDF5 filter finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_csv_dedupe(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 dedupe.")

    from .storage.csv_tools import deduplicate_frame
    from .storage.hdf5_table import write_hdf5_frame

    output_path = _resolve_csv_output_path(args, suffix="deduped")

    if args.dry_run:
        subset_label = ", ".join(args.subset) if args.subset else "all columns"
        plan = [
            f"source: {Path(args.source).expanduser().resolve()}",
            "rows/columns: not inspected in dry-run",
            f"subset: {subset_label}",
            f"keep: {args.keep}",
            f"output: {output_path}",
        ]
        _log_dry_run_plan(f"{_TABULAR_COMMAND} dedupe", plan)
        LOGGER.info("HDF5 dedupe dry run finished in %.2f s.", perf_counter() - start)
        return 0

    frame, source_path = _load_csv_frame(args)
    subset = _validate_csv_columns(frame, args.subset) if args.subset else None

    deduped = deduplicate_frame(
        frame,
        subset=subset,
        keep=args.keep,
    )
    source_info = frame.attrs.get("linak_hdf5_source_info")
    written = write_hdf5_frame(deduped, output_path, source_info=source_info)
    print(f"Rows after dedupe: {len(deduped)} / {len(frame)}")
    print(f"Wrote deduped HDF5: {written}")

    LOGGER.info("HDF5 dedupe finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_csv_combine(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 combine.")
    sources = _resolve_source_arguments(
        positional=getattr(args, "source", None),
        files=getattr(args, "files", None),
        source_label="HDF5 input file",
        allow_multiple=True,
    )
    _validate_hdf5_only_sources(sources, command_name=f"linak {_TABULAR_COMMAND} combine")
    if len(sources) < 2:
        raise ValueError(
            f"linak {_TABULAR_COMMAND} combine requires at least two HDF5 input files."
        )

    detected_analysis = _resolve_auto_plot_analysis_from_sources(sources)
    if detected_analysis not in {"density", "msd", "rdf", "position", "coordination"}:
        raise ValueError(
            "HDF5 combine currently supports LiNaK density/MSD/RDF/position/coordination analysis files only."
        )

    settings_source_path = _resolve_plot_settings_source_path(
        sources,
        setting_source_token=getattr(args, "settings_source", None),
    )
    if args.output:
        output_path = _resolve_non_overwriting_hdf5_path(args.output)
    else:
        output_path = _default_combined_analysis_hdf5_path(
            sources,
            analysis=detected_analysis,
        )
    output_path = _resolve_non_overwriting_hdf5_path(output_path)

    if args.dry_run:
        plan = [
            f"analysis: {detected_analysis}",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            f"plot-settings source: {settings_source_path}",
            f"output combined HDF5: {output_path}",
        ]
        _log_dry_run_plan(f"{_TABULAR_COMMAND} combine", plan)
        LOGGER.info("HDF5 combine dry run finished in %.2f s.", perf_counter() - start)
        return 0

    written = _combine_analysis_hdf5_sources(
        sources=sources,
        analysis=detected_analysis,
        output=output_path,
    )
    print(f"Wrote combined HDF5: {written}")
    LOGGER.info("HDF5 combine finished in %.2f s.", perf_counter() - start)
    return 0


def _resolve_plot_columns(args: argparse.Namespace, frame: Any) -> tuple[str | None, list[str]]:
    from .storage.csv_tools import infer_numeric_columns

    numeric_columns = infer_numeric_columns(frame)
    kind = args.kind

    if kind in {"line", "scatter", "bar"}:
        x_column = args.x
        if x_column is None:
            x_column = _prompt_for_columns(
                columns=list(frame.columns),
                prompt="Select x-axis column",
                allow_multiple=False,
            )[0]
        if x_column not in frame.columns:
            raise ValueError(f"Unknown x-axis column '{x_column}'.")

        if args.y:
            y_columns = _validate_csv_columns(frame, args.y)
        else:
            candidates = numeric_columns or list(frame.columns)
            y_columns = _prompt_for_columns(
                columns=candidates,
                prompt="Select y-axis column(s)",
                allow_multiple=True,
            )
        if kind == "bar" and len(y_columns) > 1:
            raise ValueError("Bar plots currently support exactly one --y column.")
        return x_column, y_columns

    if args.y:
        y_columns = _validate_csv_columns(frame, args.y)
    else:
        if not numeric_columns:
            raise ValueError("No numeric-like columns available for hist/box plot.")
        y_columns = _prompt_for_columns(
            columns=numeric_columns,
            prompt="Select numeric column(s) to plot",
            allow_multiple=True,
        )
    return None, y_columns


def _csv_plot_requires_interactive_selection(args: argparse.Namespace) -> bool:
    if args.kind in {"line", "scatter", "bar"}:
        return args.x is None or not args.y
    return not args.y


def _validate_plot_columns_across_frames(
    *,
    frames_by_source: list[tuple[Any, Path]],
    x_column: str | None,
    y_columns: list[str],
) -> None:
    required = list(y_columns)
    if x_column is not None:
        required.append(x_column)
    for frame, source_path in frames_by_source:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"HDF5 table '{source_path}' is missing required column(s): {', '.join(missing)}."
            )


def _resolve_csv_plot_source_labels(
    args: argparse.Namespace,
    source_paths: list[Path],
) -> list[str]:
    if args.file_labels is None:
        return [path.stem or path.name for path in source_paths]

    labels = [label.strip() for label in args.file_labels]
    if len(labels) != len(source_paths):
        raise ValueError(
            f"--file-labels count must match the number of input HDF5 files ({len(source_paths)})."
        )
    if any(not label for label in labels):
        raise ValueError("--file-labels cannot contain empty values.")
    return labels


def _resolve_csv_plot_series_labels(
    args: argparse.Namespace,
    default_labels: list[str],
) -> list[str]:
    if args.series_labels is None:
        return default_labels

    labels = [label.strip() for label in args.series_labels]
    if len(labels) != len(default_labels):
        raise ValueError(
            "--labels/--series-labels count must match rendered series count "
            f"({len(default_labels)})."
        )
    if any(not label for label in labels):
        raise ValueError("--labels/--series-labels cannot contain empty values.")
    return labels


def _should_render_csv_legend(args: argparse.Namespace, *, series_count: int) -> bool:
    if args.legend is not None:
        return args.legend and series_count > 0
    return series_count > 1


def _apply_csv_axis_controls(
    *,
    args: argparse.Namespace,
    ax: Any,
    kind: str,
    title: str,
    style: PlotStyle,
    default_x_label: str | None,
    default_y_label: str | None,
) -> None:
    from .plot.plotting import format_axis_label_units

    x_label = args.x_label if args.x_label is not None else default_x_label
    y_label = args.y_label if args.y_label is not None else default_y_label
    if x_label is not None:
        ax.set_xlabel(format_axis_label_units(x_label), fontsize=style.label_font_size)
    if y_label is not None:
        ax.set_ylabel(format_axis_label_units(y_label), fontsize=style.label_font_size)

    if args.title_visible is not False:
        ax.set_title(title, fontsize=style.title_font_size)
    else:
        ax.set_title("", fontsize=style.title_font_size)
    ax.tick_params(axis="both", labelsize=style.tick_font_size)
    if style.grid:
        ax.grid(
            True,
            linestyle=style.grid_linestyle,
            linewidth=style.grid_linewidth,
            alpha=style.grid_alpha,
        )
    if args.ticks is not False:
        if args.x_tick_rotation is not None:
            ax.tick_params(axis="x", rotation=float(args.x_tick_rotation))
        if args.y_tick_rotation is not None:
            ax.tick_params(axis="y", rotation=float(args.y_tick_rotation))
    else:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    if kind in {"bar", "box"} and args.x_scale != "linear":
        raise ValueError("--x-scale is only supported for numeric x-axes (line/scatter/hist).")

    try:
        ax.set_xscale(args.x_scale)
        ax.set_yscale(args.y_scale)
    except ValueError as exc:
        raise ValueError(
            f"Could not apply axis scales x={args.x_scale}, y={args.y_scale}: {exc}"
        ) from exc

    resolved_x_lim = _resolve_x_lim(args)
    if args.x_ticks is not None:
        ax.set_xticks([float(value) for value in args.x_ticks])
    if args.y_ticks is not None:
        ax.set_yticks([float(value) for value in args.y_ticks])
    # Apply explicit limits after ticks so tick placement cannot widen bounds.
    if resolved_x_lim is not None:
        ax.set_xlim(left=resolved_x_lim[0], right=resolved_x_lim[1])
    resolved_y_lim = _resolve_y_lim(args)
    if resolved_y_lim is not None:
        ax.set_ylim(bottom=resolved_y_lim[0], top=resolved_y_lim[1])


def _render_csv_plot(
    *,
    args: argparse.Namespace,
    frames_by_source: list[tuple[Any, Path]],
    x_column: str | None,
    y_columns: list[str],
) -> tuple[Path | None, dict[str, Any]]:
    import pandas as pd

    from .plot.plotting import configure_matplotlib_backend

    style = _build_plot_style(args)
    source_paths = [source_path for _, source_path in frames_by_source]
    title = args.title or (
        f"{source_paths[0].name} ({args.kind})"
        if len(source_paths) == 1
        else f"{len(source_paths)} HDF5 files ({args.kind})"
    )
    source_labels = _resolve_csv_plot_source_labels(args, source_paths)
    multi_source = len(frames_by_source) > 1

    def _draw(show: bool, output: str | Path | None) -> tuple[Path | None, dict[str, Any]]:
        active_backend = configure_matplotlib_backend(
            interactive=show,
            preferred_backend=args.backend,
        )
        import matplotlib.pyplot as plt

        with plt.rc_context({"font.family": style.font_family}):
            fig, ax = plt.subplots(figsize=style.figure_size)
            kind = args.kind
            rendered_handles: list[Any] = []
            rendered_labels: list[str] = []
            rendered_colors: list[str] = []
            box_data: list[Any] = []
            box_labels: list[str] = []
            final_labels: list[str] | None = None

            def _label_for(source_label: str, y_column: str) -> str:
                if multi_source:
                    return f"{source_label}:{y_column}"
                return y_column

            if kind == "line":
                total_series = len(frames_by_source) * len(y_columns)
                if args.line_colors is not None and len(args.line_colors) != total_series:
                    raise ValueError(
                        f"--line-colors count must match rendered series count ({total_series})."
                    )
                color_index = 0
                for (frame, _source_path), source_label in zip(frames_by_source, source_labels):
                    x_values = frame[x_column] if x_column is not None else frame.index
                    for y_column in y_columns:
                        y_numeric = pd.to_numeric(frame[y_column], errors="coerce")
                        mask = y_numeric.notna()
                        if x_column is not None:
                            mask = mask & frame[x_column].notna()
                        label = _label_for(source_label, y_column)
                        line_kwargs: dict[str, Any] = {
                            "lw": style.line_width,
                            "label": label,
                        }
                        if args.markers is True:
                            line_kwargs["marker"] = "o"
                        elif args.markers is False:
                            line_kwargs["marker"] = ""
                        if args.line_colors is not None:
                            line_kwargs["color"] = args.line_colors[color_index]
                        elif total_series == 1 and args.line_color is not None:
                            line_kwargs["color"] = args.line_color
                        (line_handle,) = ax.plot(
                            x_values[mask],
                            y_numeric[mask],
                            **line_kwargs,
                        )
                        rendered_handles.append(line_handle)
                        rendered_labels.append(label)
                        rendered_colors.append(str(line_handle.get_color()))
                        color_index += 1
                default_x_label = x_column or "index"
                default_y_label = "value"
            elif kind == "scatter":
                if x_column is None:
                    raise ValueError("Scatter plot requires an x-axis column.")
                for (frame, source_path), source_label in zip(frames_by_source, source_labels):
                    x_numeric = pd.to_numeric(frame[x_column], errors="coerce")
                    if int(x_numeric.notna().sum()) == 0:
                        raise ValueError(
                            f"Scatter x-axis column '{x_column}' must be numeric in '{source_path}'."
                        )
                    for y_column in y_columns:
                        y_numeric = pd.to_numeric(frame[y_column], errors="coerce")
                        mask = x_numeric.notna() & y_numeric.notna()
                        label = _label_for(source_label, y_column)
                        points = ax.scatter(
                            x_numeric[mask],
                            y_numeric[mask],
                            alpha=0.75,
                            label=label,
                        )
                        rendered_handles.append(points)
                        rendered_labels.append(label)
                default_x_label = x_column
                default_y_label = "value"
            elif kind == "bar":
                if x_column is None:
                    raise ValueError("Bar plot requires an x-axis column.")
                total_series = len(frames_by_source) * len(y_columns)
                for (frame, _source_path), source_label in zip(frames_by_source, source_labels):
                    for y_column in y_columns:
                        y_numeric = pd.to_numeric(frame[y_column], errors="coerce")
                        mask = frame[x_column].notna() & y_numeric.notna()
                        label = _label_for(source_label, y_column)
                        bar_kwargs: dict[str, Any] = {"label": label}
                        if total_series == 1:
                            bar_kwargs["color"] = style.line_color
                        else:
                            bar_kwargs["alpha"] = 0.65
                        bars = ax.bar(
                            frame.loc[mask, x_column].astype("string"),
                            y_numeric[mask],
                            **bar_kwargs,
                        )
                        rendered_handles.append(bars)
                        rendered_labels.append(label)
                default_x_label = x_column
                default_y_label = y_columns[0] if len(y_columns) == 1 else "value"
                ax.tick_params(axis="x", rotation=35, labelsize=style.tick_font_size)
            elif kind == "hist":
                total_series = len(frames_by_source) * len(y_columns)
                for (frame, _source_path), source_label in zip(frames_by_source, source_labels):
                    for y_column in y_columns:
                        y_numeric = pd.to_numeric(frame[y_column], errors="coerce").dropna()
                        if len(y_numeric) == 0:
                            continue
                        label = _label_for(source_label, y_column)
                        _counts, _edges, patches = ax.hist(
                            y_numeric,
                            bins=args.bins,
                            alpha=0.6 if total_series == 1 else 0.5,
                            label=label,
                        )
                        if patches:
                            rendered_handles.append(patches[0])
                            rendered_labels.append(label)
                default_x_label = "value"
                default_y_label = "count"
            else:  # box
                for (frame, _source_path), source_label in zip(frames_by_source, source_labels):
                    for column in y_columns:
                        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
                        if len(values) == 0:
                            continue
                        box_data.append(values)
                        box_labels.append(_label_for(source_label, column))
                if not box_data:
                    raise ValueError("No numeric data available for box plot.")
                final_box_labels = _resolve_csv_plot_series_labels(args, box_labels)
                ax.boxplot(box_data, tick_labels=final_box_labels)
                final_labels = final_box_labels
                default_x_label = "series"
                default_y_label = "value"
                ax.tick_params(axis="x", rotation=30, labelsize=style.tick_font_size)

            if kind != "box":
                if not rendered_labels:
                    raise ValueError(
                        "No plottable data found for the requested HDF5 plot selection."
                    )
                final_labels = _resolve_csv_plot_series_labels(args, rendered_labels)
                if _should_render_csv_legend(args, series_count=len(final_labels)):
                    ax.legend(
                        rendered_handles,
                        final_labels,
                        fontsize=style.legend_font_size,
                        title=args.legend_title,
                        loc=args.legend_loc,
                    )

            _apply_csv_axis_controls(
                args=args,
                ax=ax,
                kind=kind,
                title=title,
                style=style,
                default_x_label=default_x_label,
                default_y_label=default_y_label,
            )

            fig.tight_layout()
            output_path = None
            if output is not None:
                output_path = Path(output).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=style.dpi)
                LOGGER.info("Saved HDF5 plot to '%s'.", output_path)

            if show:
                LOGGER.info(
                    "Showing HDF5 plot using backend '%s'. Close the window to continue.",
                    active_backend,
                )
                plt.show()

            legend = ax.get_legend()
            legend_title = None
            if legend is not None:
                title_obj = legend.get_title()
                if title_obj is not None:
                    legend_title = str(title_obj.get_text()) or None
            captured_state = {
                "title": str(ax.get_title()),
                "title_visible": bool(ax.title.get_visible() and bool(str(ax.get_title()).strip())),
                "x_label": str(ax.get_xlabel()),
                "y_label": str(ax.get_ylabel()),
                "x_scale": str(ax.get_xscale()),
                "y_scale": str(ax.get_yscale()),
                "x_lim": [float(value) for value in ax.get_xlim()],
                "y_lim": [float(value) for value in ax.get_ylim()],
                "x_ticks": [float(value) for value in ax.get_xticks()],
                "y_ticks": [float(value) for value in ax.get_yticks()],
                "ticks": bool(
                    any(
                        label.get_visible() for label in ax.get_xticklabels() + ax.get_yticklabels()
                    )
                ),
                "legend": legend is not None,
                "legend_title": legend_title,
                "legend_loc": args.legend_loc,
                "series_labels": final_labels if final_labels is not None else None,
                "line_colors": rendered_colors if rendered_colors else None,
                "markers": bool(args.markers) if args.markers is not None else False,
                "figsize": [float(style.figure_size[0]), float(style.figure_size[1])],
                "dpi": int(style.dpi),
                "font_family": style.font_family,
                "title_font_size": int(style.title_font_size),
                "label_font_size": int(style.label_font_size),
                "tick_font_size": int(style.tick_font_size),
                "legend_font_size": int(style.legend_font_size),
                "line_width": float(style.line_width),
                "grid": bool(style.grid),
                "grid_linestyle": style.grid_linestyle,
                "grid_linewidth": float(style.grid_linewidth),
                "grid_alpha": float(style.grid_alpha),
            }
            plt.close(fig)
            return output_path, captured_state

    if args.show:
        try:
            return _draw(True, args.output)
        except RuntimeError as exc:
            fallback_output = args.output or _default_csv_plot_output_for_sources(
                source_paths,
                f"hdf5_{args.kind}" if len(source_paths) == 1 else f"multi_hdf5_{args.kind}",
            )
            LOGGER.warning("Interactive plotting unavailable: %s", exc)
            LOGGER.warning(
                "Falling back to non-interactive render. Plot will be saved to '%s'.",
                fallback_output,
            )
            return _draw(False, fallback_output)
    return _draw(False, args.output)


def _handle_csv_plot(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 plot.")
    sources = _resolve_csv_plot_sources(args)
    if len(sources) == 1:
        _apply_saved_plot_settings(
            args=args,
            source_path=Path(sources[0]).expanduser().resolve(),
            profile_key=_PLOT_PROFILE_TABLE,
            keys=_PLOT_SETTINGS_TABLE_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    _resolve_csv_plot_source_labels(
        args, [Path(source).expanduser().resolve() for source in sources]
    )

    if args.dry_run:
        if args.output:
            render_target = f"save plot to {Path(args.output).expanduser().resolve()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"
        y_preview = ", ".join(args.y) if args.y else "interactive (resolved at execution)"
        plan = [
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            "rows/columns: not inspected in dry-run",
            f"kind={args.kind}",
            f"x={args.x if args.x is not None else 'auto/interactive'}",
            f"y={y_preview}",
            f"bins={args.bins if args.kind == 'hist' else 'n/a'}",
            f"legend={'auto' if args.legend is None else ('on' if args.legend else 'off')}",
            f"render target: {render_target}",
        ]
        _log_dry_run_plan(f"{_TABULAR_COMMAND} plot", plan)
        LOGGER.info("HDF5 plot dry run finished in %.2f s.", perf_counter() - start)
        return 0

    frames_by_source = [_load_csv_frame_from_source(source, group=args.group) for source in sources]
    if _csv_plot_requires_interactive_selection(args) and _interactive_prompts_available():
        for index, (frame, source_path) in enumerate(frames_by_source, start=1):
            heading = (
                "Preview before interactive plot selection"
                if len(frames_by_source) == 1
                else f"Preview [{index}/{len(frames_by_source)}] before interactive plot selection"
            )
            _print_csv_preview_for_interactive(
                frame=frame,
                source_path=source_path,
                heading=heading,
            )
    x_column, y_columns = _resolve_plot_columns(args, frames_by_source[0][0])
    if not y_columns:
        raise ValueError("No y columns were selected for plotting.")
    _validate_plot_columns_across_frames(
        frames_by_source=frames_by_source,
        x_column=x_column,
        y_columns=y_columns,
    )

    saved_path, _rendered_state = _render_csv_plot(
        args=args,
        frames_by_source=frames_by_source,
        x_column=x_column,
        y_columns=y_columns,
    )
    if saved_path is None and not args.show:
        LOGGER.warning("No interactive display or output path requested. Nothing was rendered.")

    LOGGER.info("HDF5 plot finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_csv_interactive(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting HDF5 interactive assistant.")

    from .storage.csv_tools import format_frame_preview

    frame, source_path = _load_csv_frame(args)
    print(f"Interactive HDF5 assistant for: {source_path}")
    _print_hdf5_metadata_overview(frame)
    print(f"Rows: {len(frame)} | Columns: {len(frame.columns)}")
    print("")
    print(format_frame_preview(frame, rows=args.rows, tail=False, show_index=False))
    print("")

    action = _prompt_for_value(
        "Action (info/preview/get/sort/filter/dedupe/plot/plot-settings)",
        allowed={
            "info",
            "preview",
            "get",
            "sort",
            "filter",
            "dedupe",
            "plot",
            "plot-settings",
        },
    ).lower()

    if action == "info":
        delegated = argparse.Namespace(
            source=args.source,
            group=args.group,
        )
        rc = _handle_csv_info(delegated)
    elif action == "preview":
        rows = int(_prompt_for_value("Rows to preview"))
        tail = _prompt_yes_no("Show tail rows", default=False)
        delegated = argparse.Namespace(
            source=args.source,
            group=args.group,
            rows=rows,
            tail=tail,
            show_index=False,
        )
        rc = _handle_csv_preview(delegated)
    elif action == "get":
        columns = _prompt_for_columns(
            columns=list(frame.columns),
            prompt="Select column(s) to analyze",
            allow_multiple=True,
        )
        delegated = argparse.Namespace(
            source=args.source,
            group=args.group,
            column=columns,
            all_columns=False,
            metric=None,
            round_digits=6,
        )
        rc = _handle_csv_get(delegated)
    elif action == "sort":
        sort_columns = _prompt_for_columns(
            columns=list(frame.columns),
            prompt="Select sort column(s)",
            allow_multiple=True,
        )
        descending = _prompt_yes_no("Sort descending", default=False)
        inplace = _prompt_yes_no("Overwrite input HDF5", default=False)
        delegated = argparse.Namespace(
            source=args.source,
            group=args.group,
            by=sort_columns,
            descending=descending,
            na_position="last",
            mode="auto",
            output=None,
            inplace=inplace,
            dry_run=False,
        )
        rc = _handle_csv_sort(delegated)
    elif action == "filter":
        column = _prompt_for_columns(
            columns=list(frame.columns),
            prompt="Select filter column",
            allow_multiple=False,
        )[0]
        operator = _prompt_for_value(
            "Operator (eq/ne/gt/ge/lt/le/contains/startswith/endswith/regex/in/not-in)",
            allowed={
                "eq",
                "ne",
                "gt",
                "ge",
                "lt",
                "le",
                "contains",
                "startswith",
                "endswith",
                "regex",
                "in",
                "not-in",
            },
        ).lower()
        value = _prompt_for_value("Filter value")
        delegated = argparse.Namespace(
            source=args.source,
            group=args.group,
            column=column,
            operator=operator,
            value=value,
            case_sensitive=False,
            invert=False,
            output=None,
            inplace=False,
            dry_run=False,
        )
        rc = _handle_csv_filter(delegated)
    elif action == "dedupe":
        subset = _prompt_for_columns(
            columns=list(frame.columns),
            prompt="Select dedupe subset column(s)",
            allow_multiple=True,
        )
        keep = _prompt_for_value(
            "Keep (first/last/none)", allowed={"first", "last", "none"}
        ).lower()
        inplace = _prompt_yes_no("Overwrite input HDF5", default=False)
        delegated = argparse.Namespace(
            source=args.source,
            group=args.group,
            subset=subset,
            keep=keep,
            output=None,
            inplace=inplace,
            dry_run=False,
        )
        rc = _handle_csv_dedupe(delegated)
    elif action == "plot":
        kind = _prompt_for_value(
            "Plot kind (line/scatter/bar/hist/box)",
            allowed={"line", "scatter", "bar", "hist", "box"},
        ).lower()
        delegated = argparse.Namespace(
            source=[args.source],
            files=None,
            group=args.group,
            kind=kind,
            x=None,
            y=None,
            bins=30,
            output=None,
            show=True,
            backend=DEFAULT_INTERACTIVE_BACKEND,
            title=None,
            x_label=None,
            y_label=None,
            x_scale="linear",
            y_scale="linear",
            title_visible=None,
            ticks=None,
            markers=None,
            x_min=None,
            x_max=None,
            x_lim=None,
            y_min=None,
            y_max=None,
            y_lim=None,
            x_ticks=None,
            y_ticks=None,
            x_tick_rotation=None,
            y_tick_rotation=None,
            series_labels=None,
            file_labels=None,
            legend=None,
            legend_title=None,
            legend_loc="best",
            figsize=None,
            dpi=None,
            font_family=None,
            title_font_size=None,
            label_font_size=None,
            tick_font_size=None,
            legend_font_size=None,
            line_width=None,
            line_color=None,
            line_colors=None,
            grid=None,
            grid_linestyle=None,
            grid_linewidth=None,
            grid_alpha=None,
            dry_run=False,
        )
        rc = _handle_csv_plot(delegated)
    else:
        delegated = argparse.Namespace(
            source=args.source,
            files=None,
            profile="auto",
            set=None,
            unset=None,
            delete=False,
            import_from=None,
            export_to=None,
            show_all=False,
        )
        rc = _handle_csv_plot_settings(delegated)

    LOGGER.info("HDF5 interactive assistant finished in %.2f s.", perf_counter() - start)
    return rc


def _detect_plot_analysis_from_hdf5_source(source: str | Path) -> str | None:
    from .plot.plot_settings import read_hdf5_analysis, read_plot_profiles

    source_path = Path(source).expanduser().resolve()
    analysis = read_hdf5_analysis(source_path)
    if analysis in {
        "density",
        "msd",
        "rdf",
        "position",
        "coordination",
        "potential",
        "orientation",
        "table",
    }:
        return analysis

    try:
        profiles = read_plot_profiles(source_path)
    except Exception:
        return None
    for profile_key in (
        _PLOT_PROFILE_DENSITY,
        _PLOT_PROFILE_MSD,
        _PLOT_PROFILE_RDF,
        _PLOT_PROFILE_POSITION,
        _PLOT_PROFILE_COORDINATION,
        _PLOT_PROFILE_POTENTIAL,
        _PLOT_PROFILE_ORIENTATION,
        _PLOT_PROFILE_TABLE,
    ):
        if profile_key in profiles:
            return _PROFILE_KEY_TO_ANALYSIS.get(profile_key)
    return None


def _resolve_plot_settings_source_path(
    sources: list[str],
    *,
    setting_source_token: str | None,
) -> Path:
    if not sources:
        raise ValueError("No HDF5 sources were provided.")

    resolved_sources = [Path(source).expanduser().resolve() for source in sources]
    if setting_source_token is None:
        return resolved_sources[0]

    token = setting_source_token.strip()
    if not token:
        return resolved_sources[0]

    if token.isdigit():
        one_based = int(token)
        if one_based < 1 or one_based > len(resolved_sources):
            raise ValueError(
                "--settings-source index is out of range. "
                f"Expected 1..{len(resolved_sources)}, got {one_based}."
            )
        return resolved_sources[one_based - 1]

    requested = Path(token).expanduser().resolve()
    if requested in resolved_sources:
        return requested

    raise ValueError(
        f"--settings-source path must match one of the provided input files. Got '{requested}'."
    )


def _resolve_auto_plot_analysis_from_sources(sources: list[str]) -> str | None:
    detected = [_detect_plot_analysis_from_hdf5_source(source) for source in sources]
    available = [name for name in detected if name is not None]
    if not available:
        return None
    first = available[0]
    if any(name != first for name in available[1:]):
        LOGGER.warning(
            "Input files report mixed analysis/profile metadata. "
            "Using the first detected analysis '%s'.",
            first,
        )
    return first


def _read_analysis_profile_payloads(
    *,
    sources: list[str],
    analysis: str,
) -> list[dict[str, Any]]:
    payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis=analysis,
    )
    return [
        payload for _source, source_payloads in payloads_by_source for payload in source_payloads
    ]


def _read_analysis_profile_payloads_by_source(
    *,
    sources: list[str],
    analysis: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    from .storage.hdf5_utils import read_linak_hdf5_profiles

    payloads_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source_index, source in enumerate(sources):
        source_path = Path(source).expanduser().resolve()
        profiles = read_linak_hdf5_profiles(source_path, expected_analysis=analysis)
        if not profiles:
            raise ValueError(f"No '{analysis}' profiles found in '{source_path}'.")
        source_payloads: list[dict[str, Any]] = []
        for profile_index, (datasets, metadata) in enumerate(profiles):
            merged_metadata = dict(metadata)
            merged_metadata["origin_hdf5_path"] = str(
                merged_metadata.get("origin_hdf5_path") or source_path
            )
            merged_metadata["source_path"] = str(source_path)
            merged_metadata.setdefault("source_index", source_index)
            merged_metadata.setdefault("source_profile_index", profile_index)
            source_payloads.append(
                {
                    "datasets": datasets,
                    "metadata": merged_metadata,
                }
            )
        payloads_by_source.append((source, source_payloads))
    return payloads_by_source


def _read_analysis_profile_headers_by_source(
    *,
    sources: list[str],
    analysis: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    from .storage.hdf5_utils import read_linak_hdf5_profile_headers

    headers_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source_index, source in enumerate(sources):
        source_path = Path(source).expanduser().resolve()
        headers = read_linak_hdf5_profile_headers(source_path, expected_analysis=analysis)
        if not headers:
            raise ValueError(f"No '{analysis}' profiles found in '{source_path}'.")
        source_headers: list[dict[str, Any]] = []
        for profile_index, metadata in enumerate(headers):
            merged_metadata = dict(metadata)
            merged_metadata["origin_hdf5_path"] = str(
                merged_metadata.get("origin_hdf5_path") or source_path
            )
            merged_metadata["source_path"] = str(source_path)
            merged_metadata.setdefault("source_index", source_index)
            merged_metadata.setdefault("source_profile_index", profile_index)
            merged_metadata.setdefault("profile_index", profile_index)
            source_headers.append(merged_metadata)
        headers_by_source.append((source, source_headers))
    return headers_by_source


def _combine_analysis_hdf5_sources(
    *,
    sources: list[str],
    analysis: str,
    output: str | Path | None,
    settings_source_path: str | Path | None = None,
) -> Path:
    from .storage.hdf5_utils import write_linak_hdf5_profile_collection

    del settings_source_path
    payloads = _read_analysis_profile_payloads(sources=sources, analysis=analysis)
    if output is None:
        output_path = _resolve_non_overwriting_hdf5_path(
            _default_combined_analysis_hdf5_path(sources, analysis=analysis)
        )
    else:
        output_path = _resolve_non_overwriting_hdf5_path(output)

    combined_metadata: dict[str, Any] = {
        "analysis": analysis,
        "source_files": [str(Path(source).expanduser().resolve()) for source in sources],
    }

    written_path = write_linak_hdf5_profile_collection(
        output_path,
        analysis=analysis,
        profiles=payloads,
        metadata=combined_metadata,
    )
    return written_path


def _resolve_plot_hdf5_sources(args: argparse.Namespace, *, command_name: str) -> list[str]:
    sources = _resolve_plot_sources(args)
    _validate_hdf5_only_sources(sources, command_name=command_name)
    return sources


def _legacy_plot_subcommand(args: argparse.Namespace) -> str | None:
    positional_sources = _normalize_source_values(getattr(args, "source", None))
    if positional_sources and positional_sources[0] in {
        "density",
        "msd",
        "rdf",
        "position",
        "coordination",
    }:
        return positional_sources[0]
    return None


def _handle_plot(args: argparse.Namespace) -> int:
    if _legacy_plot_subcommand(args) is not None:
        raise ValueError(
            "Explicit `linak plot density|msd|rdf|position|coordination|potential` subcommands were removed. "
            "Use `linak plot /path/to/file.h5` for LiNaK analysis HDF5, or "
            f"`linak {_TABULAR_COMMAND} plot ...` for generic HDF5 tables."
        )

    if not _normalize_source_values(getattr(args, "source", None)) and not _normalize_source_values(
        getattr(args, "files", None)
    ):
        return _handle_plot_overview(args)

    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    detected_analysis = _resolve_auto_plot_analysis_from_sources(sources)
    if detected_analysis == "density":
        args.plot_command = "density"
        return _handle_plot_density(args)
    if detected_analysis == "msd":
        args.plot_command = "msd"
        return _handle_plot_msd(args)
    if detected_analysis == "rdf":
        args.plot_command = "rdf"
        return _handle_plot_rdf(args)
    if detected_analysis == "position":
        args.plot_command = "position"
        return _handle_plot_position(args)
    if detected_analysis == "coordination":
        args.plot_command = "coordination"
        return _handle_plot_coordination(args)
    if detected_analysis == "potential":
        args.plot_command = "potential"
        return _handle_plot_potential(args)
    if detected_analysis == "orientation":
        args.plot_command = "orientation"
        return _handle_plot_orientation(args)

    raise ValueError(
        "Could not detect a LiNaK density/MSD/RDF/position/coordination/potential/orientation analysis from the provided HDF5 input. "
        f"Use `linak {_TABULAR_COMMAND} plot ...` for generic HDF5 plotting."
    )


def _handle_plot_density(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting density plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_DENSITY,
            keys=_PLOT_SETTINGS_DENSITY_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d density HDF5 input file(s).", len(sources))

    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"

        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            (
                f"species={args.species}, axis={args.axis}, "
                f"x_mode={args.x_mode}, quantity={args.quantity}"
            ),
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot density", plan)
        LOGGER.info("Density plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    if use_gui:
        from .analysis.density import plot_density_profiles

        gui_sources = list(sources)
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="density",
                output=None,
            )
            gui_sources = [str(gui_settings_path)]
            LOGGER.info(
                "Created combined density HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None

        active_profiles_by_series_id: dict[str, Any] = {}

        def _build_catalog(current_args: argparse.Namespace) -> _LazyGuiSeriesCatalog:
            catalog = _build_density_gui_lazy_catalog(
                current_args,
                sources=gui_sources,
                active_profiles_by_series_id=active_profiles_by_series_id,
            )
            catalog.default_series_labels = _resolve_gui_default_series_labels(
                args=current_args,
                sources=gui_sources,
                profile_key=_PLOT_PROFILE_DENSITY,
                fallback_labels_by_source=catalog.fallback_labels_by_source,
            )
            return catalog

        initial_catalog = _build_catalog(args)
        initial_context = initial_catalog.build_initial_context()
        _apply_effective_series_settings(
            args=args,
            sources=gui_sources,
            profile_key=_PLOT_PROFILE_DENSITY,
            fallback_labels_by_source=initial_context.fallback_labels_by_source,
            series_descriptors=initial_context.series_descriptors,
            allow_saved_multi_source_merge=not (use_gui and len(sources) > 1),
            materialize_default_colors=False,
        )
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_DENSITY,
            setting_keys=_PLOT_SETTINGS_DENSITY_KEYS,
            gui_title="LiNaK Plot Controls: Density",
            analysis_name="density",
            plotter=plot_density_profiles,
            initial_context=initial_context,
            build_context=lambda current_args: _build_catalog(current_args).build_render_context(
                current_args
            ),
            build_full_context=lambda current_args: _build_catalog(
                current_args
            ).build_initial_context(),
        )
        LOGGER.info("Density GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.density import plot_density_profiles

    render_context = _build_density_gui_context(args, sources=sources)

    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_DENSITY,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=True,
        materialize_default_colors=True,
    )

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="density",
        profile=render_context.profile,
        plotter=plot_density_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("Density plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_plot_msd(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting MSD plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_MSD,
            keys=_PLOT_SETTINGS_MSD_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d MSD HDF5 input file(s).", len(sources))

    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"

        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            f"species={args.species}",
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot msd", plan)
        LOGGER.info("MSD plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    if use_gui:
        from .analysis.msd import plot_msd_profiles

        gui_sources = list(sources)
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="msd",
                output=None,
            )
            gui_sources = [str(gui_settings_path)]
            LOGGER.info(
                "Created combined MSD HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None

        active_profiles_by_series_id: dict[str, Any] = {}

        def _build_catalog(current_args: argparse.Namespace) -> _LazyGuiSeriesCatalog:
            catalog = _build_msd_gui_lazy_catalog(
                current_args,
                sources=gui_sources,
                active_profiles_by_series_id=active_profiles_by_series_id,
            )
            catalog.default_series_labels = _resolve_gui_default_series_labels(
                args=current_args,
                sources=gui_sources,
                profile_key=_PLOT_PROFILE_MSD,
                fallback_labels_by_source=catalog.fallback_labels_by_source,
            )
            return catalog

        initial_catalog = _build_catalog(args)
        initial_context = initial_catalog.build_initial_context()
        _apply_effective_series_settings(
            args=args,
            sources=gui_sources,
            profile_key=_PLOT_PROFILE_MSD,
            fallback_labels_by_source=initial_context.fallback_labels_by_source,
            series_descriptors=initial_context.series_descriptors,
            allow_saved_multi_source_merge=not (use_gui and len(sources) > 1),
            materialize_default_colors=False,
        )
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_MSD,
            setting_keys=_PLOT_SETTINGS_MSD_KEYS,
            gui_title="LiNaK Plot Controls: MSD",
            analysis_name="msd",
            plotter=plot_msd_profiles,
            initial_context=initial_context,
            build_context=lambda current_args: _build_catalog(current_args).build_render_context(
                current_args
            ),
            build_full_context=lambda current_args: _build_catalog(
                current_args
            ).build_initial_context(),
        )
        LOGGER.info("MSD GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.msd import plot_msd_profiles

    render_context = _build_msd_gui_context(args, sources=sources)

    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_MSD,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=True,
        materialize_default_colors=True,
    )

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="msd",
        profile=render_context.profile,
        plotter=plot_msd_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("MSD plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_plot_rdf(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting RDF plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_RDF,
            keys=_PLOT_SETTINGS_RDF_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d RDF HDF5 input file(s).", len(sources))

    species_b = args.species_b if args.species_b is not None else args.species_a
    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"

        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            f"species_a={args.species_a}, species_b={species_b}",
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot rdf", plan)
        LOGGER.info("RDF plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    if use_gui:
        from .analysis.rdf import plot_rdf_profiles

        gui_sources = list(sources)
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="rdf",
                output=None,
            )
            gui_sources = [str(gui_settings_path)]
            LOGGER.info(
                "Created combined RDF HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None

        active_profiles_by_series_id: dict[str, Any] = {}

        def _build_catalog(current_args: argparse.Namespace) -> _LazyGuiSeriesCatalog:
            catalog = _build_rdf_gui_lazy_catalog(
                current_args,
                sources=gui_sources,
                active_profiles_by_series_id=active_profiles_by_series_id,
            )
            catalog.default_series_labels = _resolve_gui_default_series_labels(
                args=current_args,
                sources=gui_sources,
                profile_key=_PLOT_PROFILE_RDF,
                fallback_labels_by_source=catalog.fallback_labels_by_source,
            )
            return catalog

        initial_catalog = _build_catalog(args)
        initial_context = initial_catalog.build_initial_context()
        _apply_effective_series_settings(
            args=args,
            sources=gui_sources,
            profile_key=_PLOT_PROFILE_RDF,
            fallback_labels_by_source=initial_context.fallback_labels_by_source,
            series_descriptors=initial_context.series_descriptors,
            allow_saved_multi_source_merge=not (use_gui and len(sources) > 1),
            materialize_default_colors=False,
        )
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_RDF,
            setting_keys=_PLOT_SETTINGS_RDF_KEYS,
            gui_title="LiNaK Plot Controls: RDF",
            analysis_name="rdf",
            plotter=plot_rdf_profiles,
            initial_context=initial_context,
            build_context=lambda current_args: _build_catalog(current_args).build_render_context(
                current_args
            ),
            build_full_context=lambda current_args: _build_catalog(
                current_args
            ).build_initial_context(),
        )
        LOGGER.info("RDF GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.rdf import plot_rdf_profiles

    render_context = _build_rdf_gui_context(args, sources=sources)

    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_RDF,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=True,
        materialize_default_colors=True,
    )

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="rdf",
        profile=render_context.profile,
        plotter=plot_rdf_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("RDF plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_plot_position(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting position plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if not hasattr(args, "x_bin_width"):
        args.x_bin_width = None
    if getattr(args, "time_section_width", None) is not None:
        args.x_bin_width = args.time_section_width
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_POSITION,
            keys=_PLOT_SETTINGS_POSITION_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    if (
        getattr(args, "time_section_width", None) is None
        and getattr(args, "x_bin_width", None) is not None
    ):
        args.time_section_width = args.x_bin_width

    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d position HDF5 input file(s).", len(sources))

    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"

        section_preview = (
            f"{args.time_section_width:.6g}" if args.time_section_width is not None else "off"
        )
        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            (
                f"species={args.species}, axis={args.axis}, component={args.component}, "
                f"map_color={args.map_color}, time_axis={args.time_axis}, "
                f"time_section_width={section_preview}"
            ),
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot position", plan)
        LOGGER.info("Position plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    if use_gui:
        from .analysis.position import plot_position_profiles

        gui_sources = list(sources)
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="position",
                output=None,
            )
            gui_sources = [str(gui_settings_path)]
            LOGGER.info(
                "Created combined position HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None

        active_profiles_by_series_id: dict[str, Any] = {}

        def _build_catalog(current_args: argparse.Namespace) -> _LazyGuiSeriesCatalog:
            catalog = _build_position_gui_lazy_catalog(
                current_args,
                sources=gui_sources,
                active_profiles_by_series_id=active_profiles_by_series_id,
            )
            catalog.default_series_labels = _resolve_gui_default_series_labels(
                args=current_args,
                sources=gui_sources,
                profile_key=_PLOT_PROFILE_POSITION,
                fallback_labels_by_source=catalog.fallback_labels_by_source,
            )
            return catalog

        initial_catalog = _build_catalog(args)
        initial_context = initial_catalog.build_initial_context()
        _apply_effective_series_settings(
            args=args,
            sources=gui_sources,
            profile_key=_PLOT_PROFILE_POSITION,
            fallback_labels_by_source=initial_context.fallback_labels_by_source,
            series_descriptors=initial_context.series_descriptors,
            allow_saved_multi_source_merge=not (use_gui and len(sources) > 1),
            materialize_default_colors=False,
        )
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_POSITION,
            setting_keys=_PLOT_SETTINGS_POSITION_KEYS,
            gui_title="LiNaK Plot Controls: Position",
            analysis_name="position",
            plotter=plot_position_profiles,
            initial_context=initial_context,
            build_context=lambda current_args: _build_catalog(current_args).build_render_context(
                current_args
            ),
            build_full_context=lambda current_args: _build_catalog(
                current_args
            ).build_initial_context(),
        )
        LOGGER.info("Position GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.position import plot_position_profiles

    render_context = _build_position_gui_context(args, sources=sources)

    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_POSITION,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=True,
        materialize_default_colors=True,
    )

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="position",
        profile=render_context.profile,
        plotter=plot_position_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("Position plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_plot_coordination(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting coordination plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_COORDINATION,
            keys=_PLOT_SETTINGS_COORDINATION_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d coordination HDF5 input file(s).", len(sources))

    species_b = args.species_b if args.species_b is not None else args.species_a
    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"
        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            (
                f"species_a={args.species_a}, species_b={species_b}, axis={args.axis}, "
                f"component={args.component}, time_axis={args.time_axis}, "
                f"x_bin_width={getattr(args, 'x_bin_width', None)}"
            ),
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot coordination", plan)
        LOGGER.info("Coordination plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    if use_gui:
        from .analysis.coordination import plot_coordination_profiles

        gui_sources = list(sources)
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="coordination",
                output=None,
            )
            gui_sources = [str(gui_settings_path)]
            LOGGER.info(
                "Created combined coordination HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None

        active_profiles_by_series_id: dict[str, Any] = {}

        def _build_catalog(current_args: argparse.Namespace) -> _LazyGuiSeriesCatalog:
            catalog = _build_coordination_gui_lazy_catalog(
                current_args,
                sources=gui_sources,
                active_profiles_by_series_id=active_profiles_by_series_id,
            )
            catalog.default_series_labels = _resolve_gui_default_series_labels(
                args=current_args,
                sources=gui_sources,
                profile_key=_PLOT_PROFILE_COORDINATION,
                fallback_labels_by_source=catalog.fallback_labels_by_source,
            )
            return catalog

        initial_catalog = _build_catalog(args)
        initial_context = initial_catalog.build_initial_context()
        _apply_effective_series_settings(
            args=args,
            sources=gui_sources,
            profile_key=_PLOT_PROFILE_COORDINATION,
            fallback_labels_by_source=initial_context.fallback_labels_by_source,
            series_descriptors=initial_context.series_descriptors,
            allow_saved_multi_source_merge=not (use_gui and len(sources) > 1),
            materialize_default_colors=False,
        )
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_COORDINATION,
            setting_keys=_PLOT_SETTINGS_COORDINATION_KEYS,
            gui_title="LiNaK Plot Controls: Coordination",
            analysis_name="coordination",
            plotter=plot_coordination_profiles,
            initial_context=initial_context,
            build_context=lambda current_args: _build_catalog(current_args).build_render_context(
                current_args
            ),
            build_full_context=lambda current_args: _build_catalog(
                current_args
            ).build_initial_context(),
        )
        LOGGER.info("Coordination GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.coordination import plot_coordination_profiles

    render_context = _build_coordination_gui_context(args, sources=sources)

    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_COORDINATION,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=True,
        materialize_default_colors=True,
    )

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="coordination",
        profile=render_context.profile,
        plotter=plot_coordination_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("Coordination plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_plot_potential(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting potential plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_POTENTIAL,
            keys=_PLOT_SETTINGS_POTENTIAL_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d potential HDF5 input file(s).", len(sources))

    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"
        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            "series: Water bulk, Fermi, cSHE per source",
            "x-axis: record id",
            f"render target: {render_target}",
            (
                f"plot-settings source: {settings_source_path}"
                if settings_source_path is not None
                else "plot-settings source: combined GUI/session context"
            ),
        ]
        _log_dry_run_plan("plot potential", plan)
        LOGGER.info("Potential plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.potential import combine_potential_hdf5_sources, plot_potential_profiles

    render_context = _build_potential_gui_context(args, sources=sources)
    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_POTENTIAL,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=False,
        materialize_default_colors=not use_gui,
    )

    if use_gui:
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = combine_potential_hdf5_sources(
                sources=sources,
                output=_resolve_non_overwriting_hdf5_path(
                    _default_combined_analysis_hdf5_path(sources, analysis="potential")
                ),
            )
            LOGGER.info(
                "Created combined potential HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_POTENTIAL,
            setting_keys=_PLOT_SETTINGS_POTENTIAL_KEYS,
            gui_title="LiNaK Plot Controls: Hartree Potential",
            analysis_name="potential",
            plotter=plot_potential_profiles,
            initial_context=render_context,
            build_context=lambda current_args: _build_potential_gui_context(
                current_args,
                sources=sources,
            ),
        )
        LOGGER.info("Potential GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="potential",
        profile=render_context.profile,
        plotter=plot_potential_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("Potential plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_plot_orientation(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting orientation plotting.")
    sources = _resolve_plot_hdf5_sources(args, command_name="linak plot")
    settings_source_path = (
        _resolve_plot_settings_source_path(
            sources,
            setting_source_token=getattr(args, "settings_source", None),
        )
        if len(sources) == 1
        else None
    )
    default_args = deepcopy(args)
    if len(sources) == 1:
        assert settings_source_path is not None
        _apply_saved_plot_settings(
            args=args,
            source_path=settings_source_path,
            profile_key=_PLOT_PROFILE_ORIENTATION,
            keys=_PLOT_SETTINGS_ORIENTATION_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d orientation HDF5 input file(s).", len(sources))

    if args.dry_run:
        if use_gui:
            render_target = "interactive GUI controls"
        elif args.output:
            render_target = f"save plot to {Path(args.output).expanduser()}"
        elif args.show:
            render_target = f"interactive display via backend {args.backend}"
        else:
            render_target = "no render target (--no-show without --output)"
        component = getattr(args, "component", "average")
        angle = getattr(args, "angle", "polar")
        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            f"component={component}, angle={angle}",
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot orientation", plan)
        LOGGER.info("Orientation plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.orientation import plot_orientation_profiles

    render_context = _build_orientation_gui_context(args, sources=sources)
    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_ORIENTATION,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=False,
        materialize_default_colors=not use_gui,
    )

    if use_gui:
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="orientation",
                output=None,
            )
            LOGGER.info(
                "Created combined orientation HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_ORIENTATION,
            setting_keys=_PLOT_SETTINGS_ORIENTATION_KEYS,
            gui_title="LiNaK Plot Controls: Water Orientation",
            analysis_name="orientation",
            plotter=plot_orientation_profiles,
            initial_context=render_context,
            build_context=lambda current_args: _build_orientation_gui_context(
                current_args,
                sources=sources,
            ),
        )
        LOGGER.info("Orientation GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="orientation",
        profile=render_context.profile,
        plotter=plot_orientation_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )

    LOGGER.info("Orientation plotting finished in %.2f s.", perf_counter() - start)
    return 0


def _surface_options_from_cli_args(args: argparse.Namespace):
    rough_surface_envelope = getattr(args, "rough_surface_envelope", None)
    if rough_surface_envelope is None:
        return None
    from .analysis.density import SurfaceEstimatorOptions

    return SurfaceEstimatorOptions(
        mode=str(getattr(args, "surface_mode", "auto")),
        surface_elements=(
            None
            if getattr(args, "surface_elements", None) is None
            else tuple(str(value) for value in args.surface_elements)
        ),
        include_fixed_surface_atoms=bool(getattr(args, "include_fixed_surface_atoms", False)),
        rough_surface_envelope_A=float(rough_surface_envelope),
    )


def _describe_surface_cli_options(args: argparse.Namespace) -> str:
    rough_surface_envelope = getattr(args, "rough_surface_envelope", None)
    rough_surface_text = (
        "adaptive" if rough_surface_envelope is None else f"{float(rough_surface_envelope):.6g}"
    )
    return (
        f"surface_mode={args.surface_mode}, "
        f"surface_elements={args.surface_elements if args.surface_elements else 'auto'}, "
        f"include_fixed_surface_atoms={args.include_fixed_surface_atoms}, "
        f"rough_surface_envelope_A={rough_surface_text}"
    )


def _handle_compute_density(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting density compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    if args.dry_run:
        source_path = Path(args.trajectory).expanduser().resolve()
        cell_preview = _describe_cell_resolution_preview(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        if args.output:
            output_preview = str(
                _density_hdf5_output_path(
                    args.output,
                    args.trajectory,
                    species=args.species,
                    axis=args.axis,
                )
            )
        else:
            output_preview = str(
                _default_density_hdf5_output_path(args.trajectory, args.species, args.axis)
            )

        plan = [
            f"trajectory source: {source_path}",
            (
                f"species={args.species}, axis={args.axis}, bin_width={args.bin_width}, "
                f"{_describe_surface_cli_options(args)}"
            ),
            (
                "density mode: volumetric if a periodic cell is available after resolution, "
                "otherwise linear fallback"
            ),
            f"cell resolution: {cell_preview}",
            f"output HDF5 target: {output_preview}",
        ]
        _log_dry_run_plan("compute density", plan)
        LOGGER.info("Density compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.density import compute_density_profiles, save_density_profiles
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="density",
    )
    frames = read_trajectory(args.trajectory)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    profiles = compute_density_profiles(
        frames=frames,
        species=args.species,
        axis=args.axis,
        bin_width=args.bin_width,
        surface_mode=args.surface_mode,
        surface_elements=args.surface_elements,
        include_fixed_surface_atoms=args.include_fixed_surface_atoms,
        binning="cell",
        surface_options=_surface_options_from_cli_args(args),
    )
    output_path = _density_hdf5_output_path(
        args.output,
        args.trajectory,
        species=args.species,
        axis=args.axis,
    )
    density_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
    }
    if cell_input_path is not None:
        density_metadata["input_path"] = cell_input_path
    if resolved_cell is not None:
        density_metadata["resolved_cell_angstrom"] = list(resolved_cell)
    save_density_profiles(
        profiles,
        output_path,
        additional_metadata=density_metadata,
    )

    LOGGER.info("Density compute finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_compute_msd(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting MSD compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    if args.dry_run:
        source_path = Path(args.trajectory).expanduser().resolve()
        resolved_cell, cell_source = _preview_resolve_cell_without_trajectory_read(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        resolved_timestep_fs, timestep_source = (
            _preview_resolve_msd_timestep_without_trajectory_read(
                args.trajectory,
                timestep_fs=args.timestep_fs,
                input_path=args.input,
            )
        )
        output_preview = str(
            _resolve_single_analysis_hdf5_output_path(
                args.output,
                _default_msd_hdf5_output_path(args.trajectory, args.species),
            )
        )

        if resolved_cell is None:
            cell_preview = (
                "unresolved from input sources; execution may still use "
                "trajectory-embedded periodic cell after loading frames"
            )
        else:
            cell_preview = (
                f"resolved {resolved_cell[0]:.6g} {resolved_cell[1]:.6g} "
                f"{resolved_cell[2]:.6g} Angstrom ({cell_source})"
            )

        plan = [
            f"trajectory source: {source_path}",
            f"species={args.species}",
            f"cell resolution: {cell_preview}",
            f"timestep resolution: {resolved_timestep_fs:.6g} fs ({timestep_source})",
            f"output HDF5 target: {output_preview}",
        ]
        _log_dry_run_plan("compute msd", plan)
        LOGGER.info("MSD compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .trajectory.io import read_trajectory
    from .analysis.msd import compute_msd, save_msd_profile

    source_path = Path(args.trajectory).expanduser().resolve()
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="MSD",
    )
    pre_resolved_timestep, preflight_timestep_error = _preflight_resolve_analysis_timestep_fs(
        args.trajectory,
        timestep_fs=args.timestep_fs,
        input_path=args.input,
        analysis_name="MSD",
    )
    frames = read_trajectory(args.trajectory)
    resolved_cell, cell_source, cell_input_path = _resolve_and_apply_required_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="MSD",
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    timestep_fs, timestep_source, timestep_input_path, md_timestep_fs, trajectory_stride_md = (
        _resolve_analysis_timestep_fs(
            args.trajectory,
            timestep_fs=args.timestep_fs,
            input_path=args.input,
            analysis_name="MSD",
            frames=frames,
            pre_resolved=pre_resolved_timestep,
            preflight_error=preflight_timestep_error,
        )
    )
    profile = compute_msd(
        frames=frames,
        species=args.species,
        timestep_fs=timestep_fs,
    )
    output = _resolve_single_analysis_hdf5_output_path(
        args.output,
        _default_msd_hdf5_output_path(args.trajectory, profile.species),
    )
    msd_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
        "resolved_cell_angstrom": list(resolved_cell),
        "timestep_source": timestep_source,
        "frame_timestep_fs": float(timestep_fs),
    }
    if cell_input_path is not None:
        msd_metadata["cell_input_path"] = cell_input_path
    if timestep_input_path is not None:
        msd_metadata["timestep_input_path"] = timestep_input_path
    if md_timestep_fs is not None:
        msd_metadata["md_timestep_fs"] = float(md_timestep_fs)
    if trajectory_stride_md is not None:
        msd_metadata["trajectory_stride_md"] = int(trajectory_stride_md)
    save_msd_profile(profile, output, additional_metadata=msd_metadata)

    LOGGER.info("MSD compute finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_compute_position(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting position compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )
    species_token = args.species if args.species is not None else "all"
    if args.species is None:
        LOGGER.warning(
            "No --species provided for position analysis; computing one output per species."
        )

    if args.dry_run:
        source_path = Path(args.trajectory).expanduser().resolve()
        cell_preview = _describe_cell_resolution_preview(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        resolved_timestep_fs, timestep_source = (
            _preview_resolve_msd_timestep_without_trajectory_read(
                args.trajectory,
                timestep_fs=args.timestep_fs,
                input_path=args.input,
            )
        )
        if args.output:
            output_preview = str(
                _resolve_single_analysis_hdf5_output_path(
                    args.output,
                    _default_position_hdf5_output_path(args.trajectory, species_token, args.axis),
                )
            )
        elif species_token.lower() in {"all", "*"}:
            output_preview = str(
                _linak_output_dir_for_source(source_path)
                / f"{source_path.stem or 'trajectory'}_position_<species>_{args.axis.lower()}.h5"
            )
        else:
            output_preview = str(
                _default_position_hdf5_output_path(args.trajectory, species_token, args.axis)
            )

        plan = [
            f"trajectory source: {source_path}",
            (
                f"species={species_token}, axis={args.axis}, timestep_fs={args.timestep_fs or 'auto'}, "
                f"{_describe_surface_cli_options(args)}"
            ),
            f"cell resolution: {cell_preview}",
            (
                "coordinate handling: apply in-memory PBC wrapping before writing HDF5 "
                "when a usable periodic cell is available"
            ),
            f"timestep resolution: {resolved_timestep_fs:.6g} fs ({timestep_source})",
            f"output HDF5 target: {output_preview}",
        ]
        _log_dry_run_plan("compute position", plan)
        LOGGER.info("Position compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.position import compute_position_profiles, save_position_profile
    from .pbc import apply_pbc_to_frames
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="position",
    )
    pre_resolved_timestep, preflight_timestep_error = _preflight_resolve_analysis_timestep_fs(
        args.trajectory,
        timestep_fs=args.timestep_fs,
        input_path=args.input,
        analysis_name="position",
    )
    frames = read_trajectory(args.trajectory)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    pbc_corrected_positions = False
    pbc_cell: tuple[float, float, float] | None = None
    analysis_frames = frames
    if _frames_have_usable_periodic_cell(frames):
        pbc_cell = _cell_lengths_from_frame(frames[0])
        analysis_frames = apply_pbc_to_frames(frames, pbc_cell)
        pbc_corrected_positions = True
        LOGGER.info(
            "Position analysis stores PBC-corrected coordinates in HDF5 "
            "(A=%.6g, B=%.6g, C=%.6g Angstrom).",
            pbc_cell[0],
            pbc_cell[1],
            pbc_cell[2],
        )
    else:
        LOGGER.warning(
            "No usable periodic cell available for position analysis; storing raw coordinates "
            "without PBC correction."
        )
    timestep_fs, timestep_source, timestep_input_path, md_timestep_fs, trajectory_stride_md = (
        _resolve_analysis_timestep_fs(
            args.trajectory,
            timestep_fs=args.timestep_fs,
            input_path=args.input,
            analysis_name="position",
            frames=frames,
            pre_resolved=pre_resolved_timestep,
            preflight_error=preflight_timestep_error,
        )
    )
    profiles = compute_position_profiles(
        frames=analysis_frames,
        species=species_token,
        axis=args.axis,
        timestep_fs=timestep_fs,
        surface_mode=args.surface_mode,
        surface_elements=args.surface_elements,
        include_fixed_surface_atoms=args.include_fixed_surface_atoms,
        surface_options=_surface_options_from_cli_args(args),
    )
    outputs = _position_hdf5_output_paths(
        args.output,
        args.trajectory,
        profiles,
        axis=args.axis,
    )
    for profile, output in zip(profiles, outputs):
        position_metadata: dict[str, Any] = {
            "source_path": str(source_path),
            "cell_source": cell_source,
            "timestep_source": timestep_source,
            "frame_timestep_fs": float(timestep_fs),
            "positions_pbc_corrected": bool(pbc_corrected_positions),
        }
        if pbc_cell is not None:
            position_metadata["pbc_cell_angstrom"] = list(pbc_cell)
        if resolved_cell is not None:
            position_metadata["resolved_cell_angstrom"] = list(resolved_cell)
        if cell_input_path is not None:
            position_metadata["cell_input_path"] = cell_input_path
        if timestep_input_path is not None:
            position_metadata["timestep_input_path"] = timestep_input_path
        if md_timestep_fs is not None:
            position_metadata["md_timestep_fs"] = float(md_timestep_fs)
        if trajectory_stride_md is not None:
            position_metadata["trajectory_stride_md"] = int(trajectory_stride_md)
        save_position_profile(profile, output, additional_metadata=position_metadata)

    LOGGER.info("Position compute finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_compute_rdf(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting RDF compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    if args.dry_run:
        source_path = Path(args.trajectory).expanduser().resolve()
        resolved_cell, cell_source = _preview_resolve_cell_without_trajectory_read(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        if resolved_cell is None:
            cell_preview = (
                "unresolved from input sources; execution may still use "
                "trajectory-embedded periodic cell after loading frames"
            )
            r_max_preview = (
                f"{args.r_max:.6g} (explicit)"
                if args.r_max is not None
                else "auto (resolved from loaded trajectory cell at execution)"
            )
        else:
            cell_preview = f"resolved {_format_cell_values(resolved_cell)} ({cell_source})"
            r_max_preview = (
                f"{args.r_max:.6g} (explicit)"
                if args.r_max is not None
                else f"{0.5 * min(resolved_cell):.6g} (half min resolved perpendicular cell height)"
            )
        species_b = args.species_b if args.species_b is not None else args.species_a
        output_preview = str(
            _resolve_single_analysis_hdf5_output_path(
                args.output,
                _default_rdf_hdf5_output_path(args.trajectory, args.species_a, species_b),
            )
        )
        plan = [
            f"trajectory source: {source_path}",
            (
                f"species_a={args.species_a}, species_b={species_b}, r_max="
                f"{args.r_max if args.r_max is not None else 'auto'}, bin_width={args.bin_width}, "
                f"threads={args.threads if args.threads is not None else 'auto'}"
            ),
            f"cell resolution: {cell_preview}",
            f"r_max resolution: {r_max_preview}",
            f"output HDF5 target: {output_preview}",
        ]
        _log_dry_run_plan("compute rdf", plan)
        LOGGER.info("RDF compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .trajectory.io import read_trajectory
    from .analysis.rdf import compute_rdf, save_rdf_profile

    source_path = Path(args.trajectory).expanduser().resolve()
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="RDF",
    )
    frames = read_trajectory(args.trajectory)
    resolved_cell, cell_source, cell_input_path = _resolve_and_apply_required_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="RDF",
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    profile = compute_rdf(
        frames=frames,
        species_a=args.species_a,
        species_b=args.species_b,
        r_max=args.r_max,
        bin_width=args.bin_width,
        threads=args.threads,
    )
    output = _resolve_single_analysis_hdf5_output_path(
        args.output,
        _default_rdf_hdf5_output_path(args.trajectory, profile.species_a, profile.species_b),
    )
    rdf_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
        "resolved_cell_angstrom": list(resolved_cell),
    }
    if cell_input_path is not None:
        rdf_metadata["input_path"] = cell_input_path
    save_rdf_profile(profile, output, additional_metadata=rdf_metadata)

    LOGGER.info("RDF compute finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_compute_coordination(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting coordination compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    species_b = args.species_b if args.species_b is not None else args.species_a
    output_path = _resolve_single_analysis_hdf5_output_path(
        args.output,
        _default_coordination_hdf5_output_path(args.trajectory, args.species_a, species_b),
    )
    use_cutoff_from_rdf = bool(args.cutoff_from_rdf) or (
        args.cutoff is None and args.cutoff_rdf is None
    )
    diagnostic_plot_output = None
    if args.cutoff_rdf or use_cutoff_from_rdf:
        diagnostic_plot_output = output_path.with_name(f"{output_path.stem}_cutoff_rdf.png")

    if args.dry_run:
        source_path = Path(args.trajectory).expanduser().resolve()
        cell_preview = _describe_cell_resolution_preview(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        resolved_timestep_fs, timestep_source = (
            _preview_resolve_msd_timestep_without_trajectory_read(
                args.trajectory,
                timestep_fs=args.timestep_fs,
                input_path=args.input,
            )
        )
        if args.cutoff is not None:
            cutoff_preview = f"direct cutoff={args.cutoff:.6g} A"
        elif args.cutoff_rdf:
            cutoff_preview = f"RDF file={Path(args.cutoff_rdf).expanduser().resolve()}"
        else:
            cutoff_preview = "sampled RDF convergence (random batches until cutoff stabilizes)"
        plan = [
            f"trajectory source: {source_path}",
            (
                f"species_a={args.species_a}, species_b={species_b}, axis={args.axis}, "
                f"{_describe_surface_cli_options(args)}"
            ),
            (
                f"coordination cutoff: {cutoff_preview}, smoothing_width="
                f"{args.cutoff_smoothing_width:.6g} A"
            ),
            f"cell resolution: {cell_preview}",
            f"timestep resolution: {resolved_timestep_fs:.6g} fs ({timestep_source})",
            f"output HDF5 target: {output_path}",
            (
                "cutoff diagnostic PNG: none"
                if diagnostic_plot_output is None
                else str(diagnostic_plot_output)
            ),
        ]
        _log_dry_run_plan("compute coordination", plan)
        LOGGER.info("Coordination compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.coordination import (
        compute_coordination_profile,
        resolve_coordination_cutoff,
        save_coordination_profile,
    )
    from .pbc import apply_pbc_to_frames
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="coordination",
    )
    pre_resolved_timestep, preflight_timestep_error = _preflight_resolve_analysis_timestep_fs(
        args.trajectory,
        timestep_fs=args.timestep_fs,
        input_path=args.input,
        analysis_name="coordination",
    )
    frames = read_trajectory(args.trajectory)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    pbc_corrected_positions = False
    pbc_cell: tuple[float, float, float] | None = None
    analysis_frames = frames
    if _frames_have_usable_periodic_cell(frames):
        pbc_cell = _cell_lengths_from_frame(frames[0])
        analysis_frames = apply_pbc_to_frames(frames, pbc_cell)
        pbc_corrected_positions = True
    else:
        LOGGER.warning(
            "No usable periodic cell available for coordination analysis; using raw coordinates "
            "without PBC correction."
        )

    timestep_fs, timestep_source, timestep_input_path, md_timestep_fs, trajectory_stride_md = (
        _resolve_analysis_timestep_fs(
            args.trajectory,
            timestep_fs=args.timestep_fs,
            input_path=args.input,
            analysis_name="coordination",
            frames=frames,
            pre_resolved=pre_resolved_timestep,
            preflight_error=preflight_timestep_error,
        )
    )
    cutoff_resolution = resolve_coordination_cutoff(
        frames=analysis_frames,
        species_a=args.species_a,
        species_b=args.species_b,
        cutoff_A=args.cutoff,
        cutoff_rdf_path=args.cutoff_rdf,
        cutoff_from_rdf=use_cutoff_from_rdf,
        cutoff_smoothing_width_A=args.cutoff_smoothing_width,
        diagnostic_plot_output=diagnostic_plot_output,
    )
    profile = compute_coordination_profile(
        frames=analysis_frames,
        species_a=args.species_a,
        species_b=args.species_b,
        axis=args.axis,
        timestep_fs=timestep_fs,
        surface_mode=args.surface_mode,
        surface_elements=args.surface_elements,
        include_fixed_surface_atoms=args.include_fixed_surface_atoms,
        surface_options=_surface_options_from_cli_args(args),
        cutoff_resolution=cutoff_resolution,
    )
    coordination_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
        "timestep_source": timestep_source,
        "frame_timestep_fs": float(timestep_fs),
        "positions_pbc_corrected": bool(pbc_corrected_positions),
    }
    if pbc_cell is not None:
        coordination_metadata["pbc_cell_angstrom"] = list(pbc_cell)
    if resolved_cell is not None:
        coordination_metadata["resolved_cell_angstrom"] = list(resolved_cell)
    if cell_input_path is not None:
        coordination_metadata["cell_input_path"] = cell_input_path
    if timestep_input_path is not None:
        coordination_metadata["timestep_input_path"] = timestep_input_path
    if md_timestep_fs is not None:
        coordination_metadata["md_timestep_fs"] = float(md_timestep_fs)
    if trajectory_stride_md is not None:
        coordination_metadata["trajectory_stride_md"] = int(trajectory_stride_md)
    save_coordination_profile(profile, output_path, additional_metadata=coordination_metadata)

    LOGGER.info("Coordination compute finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_compute_potential(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting potential compute.")

    from .analysis.potential import (
        PotentialCsvAppender,
        PotentialConfig,
        compute_potential_records,
        error_record_for_source,
        plan_potential_csv_output,
        summarize_potential_statistics,
        validate_hartree_cube_source,
    )

    raw_sources = _resolve_plot_sources(args)
    if len(raw_sources) > 1:
        LOGGER.info("Received %d potential input file(s).", len(raw_sources))

    validated_sources: list[str] = []
    duplicate_sources: list[str] = []
    seen_sources: set[str] = set()
    for source in raw_sources:
        resolved = validate_hartree_cube_source(source)
        key = str(resolved)
        if key in seen_sources:
            duplicate_sources.append(key)
            continue
        seen_sources.add(key)
        validated_sources.append(key)

    if not validated_sources:
        raise ValueError("No unique valid Hartree cube inputs were provided.")

    default_output = _default_potential_hdf5_output_for_sources(validated_sources)
    if args.output:
        from .storage.hdf5_utils import resolve_hdf5_output_path

        output_target = resolve_hdf5_output_path(args.output)
    else:
        output_target = default_output

    if args.dry_run:
        plan = [
            f"validated input files: {len(validated_sources)}",
            (
                f"skipped duplicates from CLI input: {len(duplicate_sources)}"
                if duplicate_sources
                else "skipped duplicates from CLI input: 0"
            ),
            "already present in HDF5 and would be skipped: not inspected in dry-run",
            "files that would be computed: not inspected in dry-run",
            (f"water_padding_ang={args.water_padding_ang}, cshe_offset_ev={args.cshe_offset_ev}"),
            f"threads={args.threads if args.threads is not None else 'auto'}, strict={'yes' if args.strict else 'no'}",
            f"HDF5 requested target: {_compact_path_for_log(output_target)}",
            "HDF5 actual target (append/fallback behavior): not inspected in dry-run",
            (
                "dry-run behavior: inputs validated, no file contents inspected, "
                "no compute executed, no file written"
            ),
        ]
        _log_dry_run_plan("compute potential", plan)
        LOGGER.info("Potential compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    hdf5_plan_preview = plan_potential_csv_output(
        output_target,
        append=args.append,
        overwrite=args.overwrite,
    )
    existing_sources = (
        hdf5_plan_preview.existing_source_keys if (args.append and not args.overwrite) else set()
    )
    skip_existing = [source for source in validated_sources if source in existing_sources]
    sources_to_compute = [source for source in validated_sources if source not in existing_sources]

    config = PotentialConfig(
        water_padding_ang=args.water_padding_ang,
        cshe_offset_ev=args.cshe_offset_ev,
    )

    if duplicate_sources:
        LOGGER.info("Skipping %d duplicate CLI input file(s).", len(duplicate_sources))
        LOGGER.debug("Duplicate input files skipped: %s", duplicate_sources)
    if skip_existing:
        LOGGER.info(
            "Skipping %d input file(s) already present in '%s'.",
            len(skip_existing),
            _compact_path_for_log(
                hdf5_plan_preview.target_path if hdf5_plan_preview.mode == "a" else output_target
            ),
        )
        LOGGER.debug("HDF5-existing source keys skipped: %s", skip_existing)
    if hdf5_plan_preview.used_fallback_path:
        LOGGER.warning(
            "HDF5 target switched to fallback '%s'.",
            _compact_path_for_log(hdf5_plan_preview.target_path),
        )
        LOGGER.debug("HDF5 fallback target path: %s", hdf5_plan_preview.target_path)

    if not sources_to_compute:
        LOGGER.info("No new input files to compute after pre-checks. HDF5 left unchanged.")
        LOGGER.info("Potential compute finished in %.2f s.", perf_counter() - start)
        return 0

    with PotentialCsvAppender(
        output=output_target,
        append=args.append,
        overwrite=args.overwrite,
        sync_on_write=True,
    ) as csv_appender:
        if csv_appender.used_fallback_path and csv_appender.path != hdf5_plan_preview.target_path:
            LOGGER.warning(
                "HDF5 write fallback in use: '%s'.",
                _compact_path_for_log(csv_appender.path),
            )
            LOGGER.debug("HDF5 appender fallback path: %s", csv_appender.path)

        def _persist_record(record: PotentialRecord) -> None:
            csv_appender.append_record(record)

        def _persist_failure(failure: PotentialComputationFailure) -> None:
            if not args.include_failures:
                return
            csv_appender.append_record(error_record_for_source(failure.source, failure.error))

        records, failures = compute_potential_records(
            sources_to_compute,
            config=config,
            threads=args.threads,
            on_record=_persist_record,
            on_failure=_persist_failure,
        )

        rows_written = csv_appender.rows_written
        write_path = csv_appender.path
        write_used_fallback = csv_appender.used_fallback_path

    if failures:
        for failure in failures:
            LOGGER.error(
                "Potential compute failed for '%s': %s",
                _compact_path_for_log(failure.source),
                failure.error,
            )
            LOGGER.debug("Potential compute failure source: %s", failure.source)

    if not records and not (args.include_failures and failures):
        raise ValueError("No potential records were produced for HDF5 export.")
    if write_used_fallback:
        LOGGER.warning(
            "Used fallback HDF5 path for this run: '%s'.",
            _compact_path_for_log(write_path),
        )
        LOGGER.debug("Write-result fallback path: %s", write_path)

    stats = summarize_potential_statistics(records)
    for key, label in (
        ("efermi_ev", "E_Fermi"),
        ("water_bulk_potential_ev", "Water bulk potential"),
        ("electrode_cshe_ev", "Electrode potential cSHE"),
    ):
        mean, std, count = stats[key]
        if count == 0 or mean is None or std is None:
            LOGGER.info("%s avg+-std (n=%d): NA", label, count)
        else:
            LOGGER.info("%s avg+-std (n=%d): %.6f +- %.6f eV", label, count, mean, std)

    incomplete_count = sum(1 for record in records if not record.is_complete())
    if incomplete_count:
        LOGGER.warning(
            "Computed %d incomplete potential row(s) (missing E_Fermi and/or water bulk and/or cSHE).",
            incomplete_count,
        )

    if not records:
        LOGGER.error("No potential source completed successfully.")
        LOGGER.info("Potential compute finished in %.2f s.", perf_counter() - start)
        return 1

    if args.strict and (failures or incomplete_count):
        LOGGER.error(
            "Strict mode enabled: failing run due to %d failure(s) and %d incomplete row(s).",
            len(failures),
            incomplete_count,
        )
        LOGGER.info("Potential compute finished in %.2f s.", perf_counter() - start)
        return 1

    LOGGER.info(
        "Potential compute finished in %.2f s. rows=%d success=%d errors=%d skipped_existing=%d.",
        perf_counter() - start,
        rows_written,
        len(records),
        len(failures),
        len(skip_existing),
    )
    return 0


def _handle_compute_orientation(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting orientation compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    if args.dry_run:
        source_path = Path(args.trajectory).expanduser().resolve()
        cell_preview = _describe_cell_resolution_preview(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        output_preview = str(
            _orientation_hdf5_output_path(
                args.output,
                args.trajectory,
                axis=args.axis,
            )
        )
        plan = [
            f"trajectory source: {source_path}",
            (
                f"axis={args.axis}, reference_axis={args.reference_axis}, "
                f"bin_width={args.bin_width}, angle_bins={args.angle_bins}, "
                f"oh_cutoff={args.oh_cutoff}, "
                f"{_describe_surface_cli_options(args)}"
            ),
            f"cell resolution: {cell_preview}",
            f"output HDF5 target: {output_preview}",
        ]
        _log_dry_run_plan("compute orientation", plan)
        LOGGER.info("Orientation compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.orientation import compute_orientation_profile, save_orientation_profile
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="orientation",
    )
    frames = read_trajectory(args.trajectory)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
        analysis_label="orientation analysis",
    )
    profile = compute_orientation_profile(
        frames=frames,
        axis=args.axis,
        reference_axis=args.reference_axis,
        bin_width=args.bin_width,
        angle_bin_count=args.angle_bins,
        surface_mode=args.surface_mode,
        surface_elements=args.surface_elements,
        include_fixed_surface_atoms=args.include_fixed_surface_atoms,
        surface_options=_surface_options_from_cli_args(args),
        oh_cutoff=args.oh_cutoff,
    )
    LOGGER.info(
        "Orientation result: %d frames, %d H\u2082O/frame, %d distance bins "
        "(%.2f\u2013%.2f \u00c5, width %.2g), %d angle bins, mode=%s.",
        profile.n_frames,
        profile.n_molecules_per_frame,
        len(profile.bin_centers),
        float(profile.bin_edges[0]),
        float(profile.bin_edges[-1]),
        args.bin_width,
        len(profile.heatmap_angle_bin_centers),
        profile.coordinate_mode,
    )
    output_path = _orientation_hdf5_output_path(
        args.output,
        args.trajectory,
        axis=args.axis,
    )
    orientation_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
    }
    if cell_input_path is not None:
        orientation_metadata["input_path"] = cell_input_path
    if resolved_cell is not None:
        orientation_metadata["resolved_cell_angstrom"] = list(resolved_cell)
    save_orientation_profile(
        profile,
        output_path,
        additional_metadata=orientation_metadata,
    )

    LOGGER.info("Orientation compute finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_apply_pbc(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting PBC application.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )
    output_path = _resolve_apply_output_path(args)

    if args.dry_run:
        cell_preview = _describe_cell_resolution_preview(
            args.trajectory,
            cell=_normalize_cell_args(args),
            input_path=args.input,
        )
        plan = [
            f"input trajectory: {Path(args.trajectory).expanduser().resolve()}",
            f"output trajectory: {output_path}",
            f"overwrite input: {'yes' if args.overwrite else 'no'}",
            f"cell resolution: {cell_preview}",
            "operation: wrap atom positions into resolved orthorhombic periodic cell",
        ]
        _log_dry_run_plan("apply pbc", plan)
        LOGGER.info("PBC dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .trajectory.io import read_trajectory, write_trajectory
    from .pbc import apply_pbc_to_frames, resolve_cell_dimensions

    frames = read_trajectory(args.trajectory)

    cell_arg = _normalize_cell_args(args)
    if cell_arg is None and args.input is None and _frames_have_usable_periodic_cell(frames):
        cell = _cell_lengths_from_frame(frames[0])
        LOGGER.info(
            "Using periodic cell already present in trajectory: A=%.6g, B=%.6g, C=%.6g Angstrom.",
            cell[0],
            cell[1],
            cell[2],
        )
    else:
        cell = resolve_cell_dimensions(
            output_path=output_path,
            input_path=args.input,
            cell=cell_arg,
        )
    LOGGER.info(
        "Using orthorhombic cell lengths: A=%.6g, B=%.6g, C=%.6g Angstrom.",
        cell[0],
        cell[1],
        cell[2],
    )

    wrapped_frames = apply_pbc_to_frames(frames, cell)
    write_trajectory(wrapped_frames, output_path)

    LOGGER.info("PBC application finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_apply_compress(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting CP2K output compression.")
    _resolve_single_source_argument(
        args,
        positional_attr="output_file",
        source_label="CP2K output file",
    )

    from .storage.compress import (
        build_parser_options_from_drop_sections,
        compress_cp2k_output,
        default_backup_dir_for_input,
    )

    input_path = Path(args.output_file).expanduser().resolve()
    backup_dir = (
        Path(args.backup_dir).expanduser().resolve()
        if args.backup_dir
        else default_backup_dir_for_input(input_path).resolve()
    )
    output_target = input_path.with_suffix("")
    drop_sections = sorted(set(args.drop or []))

    if args.dry_run:
        dropped_text = ", ".join(drop_sections) if drop_sections else "none"
        plan = [
            f"input CP2K output: {input_path}",
            f"output directory target: {output_target} (auto-unique suffix if path exists)",
            f"backup directory: {backup_dir}",
            "backup file naming: compress_output__<timestamp>__<stem>__<digest>.out",
            f"optional outputs dropped: {dropped_text}",
            (
                "generated artifacts: README.txt, manifest.json, summary.txt, CSV extracts "
                "(SCF/charges/forces/MD), setup/performance/warning notes, and backup_info.txt"
            ),
            "original .out handling: moved to backup directory with sidecar .meta.json",
        ]
        _log_dry_run_plan("apply compress", plan)
        LOGGER.info("Compression dry run finished in %.2f s.", perf_counter() - start)
        return 0

    options = build_parser_options_from_drop_sections(drop_sections)
    result = compress_cp2k_output(
        input_path,
        backup_dir=backup_dir,
        options=options,
    )
    LOGGER.info(
        "Compression finished in %.2f s. generated=%d skipped=%d output=%s backup=%s",
        perf_counter() - start,
        result.generated_count,
        result.skipped_count,
        result.output_dir,
        result.backup_path,
    )
    print(result.output_dir)
    return 0


def _find_primary_command_index(argv: list[str]) -> int | None:
    command_index = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"--log-level", "--log-file"}:
            index += 2
            continue
        if token.startswith("--log-level=") or token.startswith("--log-file="):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        command_index = index
        break
    return command_index


def _rewrite_implicit_plot_csv(argv: list[str]) -> list[str]:
    """Allow ``linak plot`` to fall back to ``linak hdf5 plot`` for non-analysis HDF5 sources."""
    if not argv:
        return argv

    command_index = _find_primary_command_index(argv)
    if command_index is None:
        return argv
    if argv[command_index] != "plot":
        return argv
    if len(argv) <= command_index + 1:
        return argv

    known_subcommands = {"density", "msd", "rdf", "position", "coordination", "potential"}
    next_token = argv[command_index + 1]
    if next_token in known_subcommands or next_token in {"-h", "--help"}:
        return argv

    trailing_tokens = argv[command_index + 1 :]
    source_tokens: list[str] = []

    if next_token in {"-f", "--files"}:
        index = command_index + 2
        while index < len(argv):
            candidate = argv[index]
            if candidate.startswith("-"):
                break
            source_tokens.append(candidate)
            index += 1
    else:
        file_option_index = None
        for index, token in enumerate(trailing_tokens):
            if token in {"-f", "--files"}:
                file_option_index = index
                break
        if file_option_index is not None:
            index = command_index + 1 + file_option_index + 1
            while index < len(argv):
                candidate = argv[index]
                if candidate.startswith("-"):
                    break
                source_tokens.append(candidate)
                index += 1
        else:
            hdf5_positional = next(
                (
                    token
                    for token in trailing_tokens
                    if not token.startswith("-") and Path(token).suffix.lower() in {".h5", ".hdf5"}
                ),
                None,
            )
            if hdf5_positional is not None:
                source_tokens = [hdf5_positional]

    if not source_tokens:
        return argv

    if any(Path(token).suffix.lower() not in {".h5", ".hdf5"} for token in source_tokens):
        return argv

    if any(not Path(token).expanduser().exists() for token in source_tokens):
        return argv

    detected_subcommand = _resolve_auto_plot_analysis_from_sources(source_tokens)

    rewritten = list(argv)
    if detected_subcommand in {
        "density",
        "msd",
        "rdf",
        "position",
        "coordination",
        "potential",
        "orientation",
    }:
        return rewritten

    rewritten[command_index] = _TABULAR_COMMAND
    rewritten.insert(command_index + 1, "plot")
    return rewritten


def _rewrite_implicit_csv_interactive(argv: list[str]) -> list[str]:
    """Allow ``linak hdf5 /path/to/file.h5`` as shorthand for ``hdf5 interactive``."""
    if not argv:
        return argv

    command_index = _find_primary_command_index(argv)
    if command_index is None:
        return argv
    if argv[command_index] not in _TABULAR_COMMAND_TOKENS:
        return argv
    if len(argv) <= command_index + 1:
        return argv

    known_subcommands = {
        "interactive",
        "info",
        "preview",
        "get",
        "sort",
        "filter",
        "dedupe",
        "combine",
        "plot",
        "plot-settings",
    }
    next_token = argv[command_index + 1]
    if next_token in known_subcommands or next_token in {"-h", "--help"}:
        return argv

    rewritten = list(argv)
    rewritten.insert(command_index + 1, "interactive")
    return rewritten


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by the ``linak`` console script."""
    runtime_argv = list(argv) if argv is not None else sys.argv[1:]
    runtime_argv = _rewrite_implicit_plot_csv(runtime_argv)
    runtime_argv = _rewrite_implicit_csv_interactive(runtime_argv)
    parser = build_parser()
    args = parser.parse_args(runtime_argv)
    args._runtime_argv = tuple(runtime_argv)
    configure_logging(level=args.log_level, log_file=args.log_file)
    _log_run_banner(args, runtime_argv)
    try:
        return args.handler(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
