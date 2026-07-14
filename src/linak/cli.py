"""Command-line interface for LiNaK."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, metadata as package_metadata
import importlib
import inspect
import json
import logging
import math
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
_PLOT_PROFILE_TEMPERATURE = "plot:temperature"
_PLOT_PROFILE_TABLE = "plot:table"
_ANALYSIS_TO_PROFILE_KEY = {
    "density": _PLOT_PROFILE_DENSITY,
    "msd": _PLOT_PROFILE_MSD,
    "rdf": _PLOT_PROFILE_RDF,
    "position": _PLOT_PROFILE_POSITION,
    "coordination": _PLOT_PROFILE_COORDINATION,
    "potential": _PLOT_PROFILE_POTENTIAL,
    "orientation": _PLOT_PROFILE_ORIENTATION,
    "temperature": _PLOT_PROFILE_TEMPERATURE,
    "table": _PLOT_PROFILE_TABLE,
}
_PROFILE_KEY_TO_ANALYSIS = {value: key for key, value in _ANALYSIS_TO_PROFILE_KEY.items()}
_LINAK_OUTPUT_DIRNAME = "LiNaK_outputs"
_GUI_COMPLEXITY_MAX_SERIES = 128
_GUI_COMPLEXITY_MAX_POINTS = 1_000_000
_POSITION_GUI_AUTO_DISPLAY_SERIES = 64
_POSITION_GUI_AUTO_DISPLAY_TRIGGER_SERIES = _GUI_COMPLEXITY_MAX_SERIES
_POSITION_GUI_AUTO_DISPLAY_TARGET_POINTS = 200_000
_ANALYSIS_PROFILE_HEADER_CACHE: dict[tuple[str, str, int | None, int | None], list[dict[str, Any]]] = {}


@dataclass(frozen=True)
class _PlotComplexityEstimate:
    analysis_name: str
    series_count: int
    estimated_total_points: int | None = None
    entity_expanded: bool = False

    @property
    def exceeds_limits(self) -> bool:
        if self.series_count > _GUI_COMPLEXITY_MAX_SERIES:
            return True
        if (
            self.estimated_total_points is not None
            and self.estimated_total_points > _GUI_COMPLEXITY_MAX_POINTS
        ):
            return True
        return False


@dataclass(frozen=True)
class _GuiPlotRenderContext:
    # Current bridge object between analysis-specific data loading and the
    # shared CLI/GUI rendering path.
    profile: Any
    plot_source_label: str
    plotter_kwargs: dict[str, Any] | None
    fallback_labels_by_source: list[list[str]]
    default_series_labels: list[str]
    series_descriptors: list[dict[str, Any]]
    profile_filter_options: dict[str, Any] | None = None
    estimated_total_points: int | None = None

    @property
    def series_count(self) -> int:
        return len(self.series_descriptors)


def _cached_gui_profile_matches_descriptor(profile: Any, descriptor: dict[str, Any]) -> bool:
    active_mode = str(descriptor.get("active_coordinate_mode") or "").strip().lower()
    if not active_mode:
        return True
    profile_coordinate_mode = str(getattr(profile, "coordinate_mode", "") or "").strip().lower()
    profile_axis = str(getattr(profile, "axis", "") or "").strip().lower()
    if active_mode == "distance":
        return profile_coordinate_mode == "distance"
    if active_mode in {"x", "y", "z"}:
        return profile_coordinate_mode != "distance" and profile_axis == active_mode
    return True


@dataclass
class _LazyGuiSeriesCatalog:
    sources: list[str]
    plot_source_label: str
    plotter_kwargs: dict[str, Any] | None
    descriptor_segments_by_source: list[list[dict[str, Any]]]
    profile_filter_options: dict[str, Any] | None
    load_profiles: Callable[[list[dict[str, Any]]], list[Any]]
    default_series_labels: list[str] = field(default_factory=list)
    estimated_total_points: int | None = None
    estimate_render_points: (
        Callable[[list[Any], argparse.Namespace, list[dict[str, Any]]], int | None] | None
    ) = None
    combined_profile_loader: bool = False
    _active_profiles_by_series_id: dict[str, Any] = field(default_factory=dict)
    _active_profile_cache_keys_by_series_id: dict[str, Any] = field(default_factory=dict)
    _combined_profile_cache_key: str | None = None
    _combined_profiles: list[Any] | None = None

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
            estimated_total_points=self.estimated_total_points,
        )

    def build_render_context(self, args: argparse.Namespace) -> _GuiPlotRenderContext:
        # GUI preview re-enters the normal plot path by rebuilding a fresh
        # render context from the currently active descriptor/filter state.
        active_descriptors_by_source, active_ids = _filter_active_gui_descriptor_segments(
            args=args,
            descriptor_segments_by_source=self.descriptor_segments_by_source,
        )
        active_id_set = set(active_ids)
        for series_id in list(self._active_profiles_by_series_id):
            if series_id not in active_id_set:
                self._active_profiles_by_series_id.pop(series_id, None)
                self._active_profile_cache_keys_by_series_id.pop(series_id, None)

        active_descriptors = [
            dict(descriptor) for segment in active_descriptors_by_source for descriptor in segment
        ]
        if self.combined_profile_loader:
            combined_cache_key = json.dumps(
                [
                    {
                        "series_id": str(descriptor.get("series_id") or ""),
                        "slice_key": descriptor.get("density_grid_slice_key"),
                        "enabled": True,
                    }
                    for descriptor in active_descriptors
                ],
                sort_keys=True,
                default=str,
            )
            if self._combined_profile_cache_key != combined_cache_key or self._combined_profiles is None:
                self._combined_profiles = list(self.load_profiles(active_descriptors))
                self._combined_profile_cache_key = combined_cache_key
            estimated_total_points = (
                self.estimate_render_points(
                    list(self._combined_profiles),
                    args,
                    active_descriptors,
                )
                if self.estimate_render_points is not None
                else _estimate_total_points_from_loaded_profiles(list(self._combined_profiles))
            )
            return _GuiPlotRenderContext(
                profile=list(self._combined_profiles),
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
                estimated_total_points=estimated_total_points,
            )
        missing_descriptors = [
            descriptor
            for descriptor in active_descriptors
            if str(descriptor.get("series_id") or "") not in self._active_profiles_by_series_id
        ]
        stale_descriptors = [
            descriptor
            for descriptor in active_descriptors
            if str(descriptor.get("series_id") or "") in self._active_profiles_by_series_id
            and (
                self._active_profile_cache_keys_by_series_id.get(
                    str(descriptor.get("series_id") or "")
                )
                != descriptor.get("density_grid_slice_key")
                or not _cached_gui_profile_matches_descriptor(
                    self._active_profiles_by_series_id[str(descriptor.get("series_id") or "")],
                    descriptor,
                )
            )
        ]
        descriptors_to_load = missing_descriptors + stale_descriptors
        if descriptors_to_load:
            loaded_profiles = self.load_profiles(descriptors_to_load)
            if len(loaded_profiles) != len(descriptors_to_load):
                raise ValueError("Lazy GUI series loader returned mismatched profile count.")
            for descriptor, profile in zip(descriptors_to_load, loaded_profiles):
                series_id = str(descriptor.get("series_id") or "").strip()
                if not series_id:
                    raise ValueError("Lazy GUI descriptor is missing a series_id.")
                self._active_profiles_by_series_id[series_id] = profile
                self._active_profile_cache_keys_by_series_id[series_id] = descriptor.get(
                    "density_grid_slice_key"
                )

        estimated_total_points = (
            self.estimate_render_points(
                [
                    self._active_profiles_by_series_id[str(descriptor.get("series_id") or "")]
                    for descriptor in active_descriptors
                ],
                args,
                active_descriptors,
            )
            if self.estimate_render_points is not None
            else _estimate_total_points_from_loaded_profiles(
                [
                    self._active_profiles_by_series_id[str(descriptor.get("series_id") or "")]
                    for descriptor in active_descriptors
                ]
            )
        )

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
            estimated_total_points=estimated_total_points,
        )


def _estimate_points_for_loaded_profile(profile: Any) -> int | None:
    for attr_name in (
        "coordination_number",
        "distance_to_surface",
        "x",
        "density",
        "number_density",
        "g_r",
        "msd",
        "values",
        "bin_centers",
    ):
        values = getattr(profile, attr_name, None)
        if isinstance(values, np.ndarray) and values.ndim >= 1 and values.size > 0:
            return int(values.size)
    return None


def _estimate_total_points_from_loaded_profiles(profile: Any) -> int | None:
    if profile is None:
        return None
    profiles = profile if isinstance(profile, list) else [profile]
    total = 0
    counted_any = False
    for item in profiles:
        count = _estimate_points_for_loaded_profile(item)
        if count is None:
            continue
        total += count
        counted_any = True
    return total if counted_any else None


def _serialize_plot_data_contract(contract: Any) -> dict[str, Any]:
    return {
        "source_id": str(getattr(contract, "source_id", "") or ""),
        "label": str(getattr(contract, "label", "") or ""),
        "default_view_type_id": (
            None
            if getattr(contract, "default_view_type_id", None) is None
            else str(contract.default_view_type_id)
        ),
        "dimensions": [
            {
                "id": str(getattr(dimension, "id", "") or ""),
                "label": str(getattr(dimension, "label", "") or ""),
                "kind": str(getattr(dimension, "kind", "") or ""),
                "length": getattr(dimension, "length", None),
                "unit": (None if getattr(dimension, "unit", None) is None else str(dimension.unit)),
            }
            for dimension in getattr(contract, "dimensions", ())
        ],
        "quantities": [
            {
                "id": str(getattr(quantity, "id", "") or ""),
                "label": str(getattr(quantity, "label", "") or ""),
                "kind": str(getattr(quantity, "kind", "") or ""),
                "dimensions": [
                    str(token)
                    for token in tuple(getattr(quantity, "dimensions", ()) or ())
                    if str(token).strip()
                ],
                "unit": None if getattr(quantity, "unit", None) is None else str(quantity.unit),
                "source_name": (
                    None
                    if getattr(quantity, "source_name", None) is None
                    else str(quantity.source_name)
                ),
            }
            for quantity in getattr(contract, "quantities", ())
        ],
        "view_types": [
            {
                "id": str(getattr(view_type, "id", "") or ""),
                "label": str(getattr(view_type, "label", "") or ""),
                "kind": str(getattr(view_type, "kind", "") or ""),
                "supported_roles": [
                    str(token)
                    for token in tuple(getattr(view_type, "supported_roles", ()) or ())
                    if str(token).strip()
                ],
            }
            for view_type in getattr(contract, "view_types", ())
        ],
    }


def _build_position_plot_gui_filter_options(reference_profile: Any | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "position_mapping_presets": [
            {"id": "distance_vs_time", "label": "Distance vs time"},
            {"id": "x_y_trajectory", "label": "X/Y view"},
            {"id": "x_z_trajectory", "label": "X/Z view"},
            {"id": "y_z_trajectory", "label": "Y/Z view"},
        ],
    }
    return _populate_position_plot_gui_filter_options(options, reference_profile)


def _populate_position_plot_gui_filter_options(
    options: dict[str, Any],
    reference_profile: Any | None,
) -> dict[str, Any]:
    if reference_profile is None:
        return options
    from .plot.contracts.position_contract import position_profile_to_plot_data_contract

    options["position_plot_contract"] = _serialize_plot_data_contract(
        position_profile_to_plot_data_contract(reference_profile)
    )
    return options


def _position_species_display_label_for_gui(species_label: str) -> str:
    from .analysis.common import is_molecule_species_label, molecule_display_label, species_selector_raw_label

    label = str(species_label).strip()
    if is_molecule_species_label(label):
        return molecule_display_label(label)
    if label.lower().startswith("species:"):
        return f"{species_selector_raw_label(label)} (species)"
    return label


def _build_position_species_options_from_headers(
    headers_by_source: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, str]]:
    from .analysis.position import _normalize_species as _normalize_position_species

    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for _source, headers in headers_by_source:
        for header in headers:
            raw_species = str(header.get("species", "")).strip()
            if not raw_species:
                continue
            try:
                species = _normalize_position_species(raw_species)
            except ValueError:
                species = raw_species
            if species in seen:
                continue
            seen.add(species)
            options.append(
                {
                    "value": species,
                    "label": _position_species_display_label_for_gui(species),
                }
            )
    return options


def _position_enabled_species_set(value: Any) -> set[str] | None:
    if not isinstance(value, (list, tuple, set)):
        return None
    enabled = {str(item).strip() for item in value if str(item).strip()}
    return enabled


def _series_descriptors_are_entity_expanded(
    series_descriptors: list[dict[str, Any]] | None,
) -> bool:
    for descriptor in series_descriptors or []:
        if descriptor.get("atom_index") is not None:
            return True
        series_id = str(descriptor.get("series_id") or "").strip()
        if ":atom:" in series_id:
            return True
    return False


def _estimate_plot_complexity(
    *,
    analysis_name: str,
    series_descriptors: list[dict[str, Any]] | None,
    profile: Any = None,
    estimated_total_points: int | None = None,
) -> _PlotComplexityEstimate:
    point_count = estimated_total_points
    if point_count is None:
        point_count = _estimate_total_points_from_loaded_profiles(profile)
    return _PlotComplexityEstimate(
        analysis_name=analysis_name,
        series_count=len(series_descriptors or []),
        estimated_total_points=point_count,
        entity_expanded=_series_descriptors_are_entity_expanded(series_descriptors),
    )


def _log_plot_complexity_debug(
    *,
    analysis_name: str,
    stage: str,
    raw_series_count: int | None = None,
    raw_point_count: int | None = None,
    final_series_count: int | None = None,
    final_point_count: int | None = None,
) -> None:
    LOGGER.debug(
        "%s GUI complexity at %s: raw_series=%s, raw_points=%s, final_series=%s, final_points=%s",
        analysis_name,
        stage,
        "NA" if raw_series_count is None else raw_series_count,
        "NA" if raw_point_count is None else raw_point_count,
        "NA" if final_series_count is None else final_series_count,
        "NA" if final_point_count is None else final_point_count,
    )


def _format_plot_complexity_message(
    estimate: _PlotComplexityEstimate,
    *,
    interactive_gui: bool,
) -> str:
    render_target = "interactive GUI controls" if interactive_gui else "non-GUI plotting"
    details = [f"{estimate.series_count} series"]
    if estimate.estimated_total_points is not None:
        details.append(f"~{estimate.estimated_total_points:,} plotted points")
    if estimate.entity_expanded:
        details.append("per-entity expanded")
    message = (
        f"{estimate.analysis_name.capitalize()} plot is too large for {render_target}: "
        f"{', '.join(details)}."
    )
    suggestions = ["Use --no-gui"]
    if estimate.analysis_name in {"position", "coordination"}:
        suggestions.append("filter species/axis or narrow the input sources")
    else:
        suggestions.append("narrow the input sources or filter the data")
    if estimate.analysis_name == "position":
        suggestions.append(
            "use --view-type 2d-heatmap with --heatmap-filter-min/--heatmap-filter-max when that lighter view is sufficient"
        )
    if interactive_gui:
        return f"{message} {'; '.join(suggestions)}. Or use --force-gui to proceed with rendering anyway, but be aware that the GUI may become unresponsive or crash."
    return f"{message} Proceeding anyway. {'; '.join(suggestions)}."


def _raise_or_warn_for_plot_complexity(
    estimate: _PlotComplexityEstimate,
    *,
    interactive_gui: bool,
) -> None:
    if not estimate.exceeds_limits:
        return
    message = _format_plot_complexity_message(estimate, interactive_gui=interactive_gui)
    if interactive_gui:
        raise ValueError(message)
    LOGGER.warning(message)


def _warn_for_non_gui_plot_complexity(
    *,
    analysis_name: str,
    render_context: _GuiPlotRenderContext,
) -> None:
    _raise_or_warn_for_plot_complexity(
        _estimate_plot_complexity(
            analysis_name=analysis_name,
            series_descriptors=render_context.series_descriptors,
            profile=render_context.profile,
            estimated_total_points=render_context.estimated_total_points,
        ),
        interactive_gui=False,
    )


@dataclass(frozen=True)
class _ResolvedPositionProjectionEstimate:
    mapping: Any
    projection_x: str
    projection_y: str
    projection_value: str
    render_mode: str
    filter_min: float | None
    filter_max: float | None

    @property
    def is_projection(self) -> bool:
        from .plot.data_contract import PLOT_VIEW_2D_HEATMAP, canonical_plot_view_id

        return (
            canonical_plot_view_id(getattr(self.mapping, "view_type_id", None))
            == PLOT_VIEW_2D_HEATMAP
        )


def _position_projection_token_from_quantity_id(quantity_id: str | None) -> str:
    token = str(quantity_id or "").strip()
    return {
        "distance_to_surface": "distance",
        "time_ps": "ps",
        "time_fs": "fs",
        "frame_index": "frame",
        "step": "step",
        "x": "x",
        "y": "y",
        "z": "z",
    }.get(token, "distance")


def _position_projection_uses_profile_descriptors(
    args: argparse.Namespace,
    *,
    resolved_projection: _ResolvedPositionProjectionEstimate | None = None,
) -> bool:
    if resolved_projection is None:
        resolved_projection = _resolve_position_projection_estimation_settings(args)
    return resolved_projection.is_projection and resolved_projection.render_mode != "line-colors"


def _coordination_plot_uses_atom_descriptors(
    args: argparse.Namespace,
) -> bool:
    from .plot.mappings.coordination_mapping import resolve_coordination_plot_mapping

    resolved_mapping = resolve_coordination_plot_mapping(
        mapping=_coerce_runtime_view_mapping(getattr(args, "view_mapping", None)),
        component=getattr(args, "component", "distance"),
        time_axis=getattr(args, "time_axis", "ps"),
    )
    return resolved_mapping.uses_atom_descriptors


_PERSISTED_PLOT_SETTING_OPTION_FLAGS = {
    "axis": ("--axis",),
    "backend": ("--backend",),
    "bins": ("--bins",),
    "dpi": ("--dpi",),
    "figsize": ("--figsize",),
    "file_labels": ("--file-labels",),
    "font_family": ("--font-family",),
    "font_color": (),
    "font_size": ("--font-size",),
    "border": ("--border", "--no-border"),
    "grid": ("--grid", "--no-grid"),
    "grid_alpha": ("--grid-alpha",),
    "grid_linestyle": ("--grid-linestyle",),
    "grid_linewidth": ("--grid-linewidth",),
    "group": ("--group",),
    "kind": ("--kind",),
    "label_font_size": ("--label-font-size",),
    "x_label_font_size": (),
    "y_label_font_size": (),
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
    "x_tick_font_size": (),
    "y_tick_font_size": (),
    "title": ("--title",),
    "title_visible": ("--title-mode",),
    "title_font_size": ("--title-font-size",),
    "title_pad": (),
    "x": ("--x",),
    "x_label": ("--x-label",),
    "x_lim": ("--x-min", "--x-max"),
    "x_mode": ("--x-mode",),
    "x_bin_width": ("--x-bin-width",),
    "x_bin_reducer": ("--x-bin-reducer",),
    "x_scale": ("--x-scale",),
    "x_axis_scale": (),
    "x_axis_offset": (),
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
    "projection_x": ("--heatmap-x",),
    "projection_y": ("--heatmap-y",),
    "projection_value": ("--heatmap-value",),
    "projection_render_mode": ("--heatmap-render-mode",),
    "projection_filter_min": ("--heatmap-filter-min",),
    "projection_filter_max": ("--heatmap-filter-max",),
    "xy_z_distance_max": ("--xy-z-distance-max",),
    "time_axis": ("--time-axis",),
    "time_section_width": ("--time-section-width",),
}

_PLOT_SETTINGS_COMMON_KEYS = (
    "title",
    "title_visible",
    "x_label",
    "y_label",
    "x_scale",
    "x_axis_scale",
    "x_axis_offset",
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
    "series_show_in_legend",
    "series_alpha",
    "series_line_widths",
    "series_markers",
    "series_normalization_modes",
    "series_normalization_values",
    "series_normalization_x_refs",
    "annotations",
    "integration_config",
    "x_bin_width",
    "x_bin_reducer",
    "min_bin_points",
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
    "font_color",
    "font_size",
    "title_font_size",
    "title_pad",
    "label_font_size",
    "x_label_font_size",
    "y_label_font_size",
    "tick_font_size",
    "x_tick_font_size",
    "y_tick_font_size",
    "legend_font_size",
    "figure_alpha",
    "line_width",
    "line_color",
    "line_colors",
    "border",
    "grid",
    "grid_linestyle",
    "grid_linewidth",
    "grid_alpha",
    "plot_data_format",
    "plot_data_delimiter",
    "plot_data_include_metadata",
    "plot_data_enabled_only",
    "_gui_sync_modes",
)
_PLOT_SETTINGS_DENSITY_KEYS = (
    "species",
    "density_enabled_species",
    "density_active_view_type",
    "density_view_states",
    "axis",
    "plane",
    "density_2d_x_axis",
    "density_2d_y_axis",
    "density_filter_x_min",
    "density_filter_x_max",
    "density_filter_y_min",
    "density_filter_y_max",
    "density_filter_z_min",
    "density_filter_z_max",
    "density_filter_distance_min",
    "density_filter_distance_max",
    "view_mapping",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_MSD_KEYS = (
    "species",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_TEMPERATURE_KEYS = (
    "time_axis",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_RDF_KEYS = (
    "species_a",
    "species_b",
    "view_mapping",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_POSITION_KEYS = (
    "species",
    "position_enabled_species",
    "position_active_view_type",
    "position_view_states",
    "axis",
    "view_mapping",
    "plot_view_type",
    "plot_y_quantity",
    "component",
    "map_color",
    "projection_x",
    "projection_y",
    "projection_value",
    "projection_render_mode",
    "projection_filter_min",
    "projection_filter_max",
    "xy_z_distance_max",
    "time_axis",
    "time_section_width",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_COORDINATION_KEYS = (
    "species_a",
    "species_b",
    "axis",
    "plot_view_type",
    "plot_x_quantity",
    "view_mapping",
    *_PLOT_SETTINGS_COMMON_KEYS,
)
_PLOT_SETTINGS_POTENTIAL_KEYS = ("view_mapping", *_PLOT_SETTINGS_COMMON_KEYS)
_PLOT_SETTINGS_ORIENTATION_KEYS = (
    "view_mapping",
    "orientation_active_view_type",
    "orientation_view_states",
    "plot_view_type",
    "plot_y_quantity",
    "orientation_line_x_axis",
    "orientation_heatmap_x_axis",
    "orientation_heatmap_y_axis",
    "orientation_filter_x_min",
    "orientation_filter_x_max",
    "orientation_filter_y_min",
    "orientation_filter_y_max",
    "orientation_filter_z_min",
    "orientation_filter_z_max",
    "orientation_filter_distance_min",
    "orientation_filter_distance_max",
    "heatmap_vmin",
    "heatmap_vmax",
    "heatmap_cmap",
    "heatmap_normalize",
    "heatmap_normalization_mode",
    "heatmap_log_scale",
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


def _add_atom_alias_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--atom-alias",
        action="append",
        default=None,
        metavar="RAW=ELEMENT",
        help=(
            "Map a non-standard atom label to an element while preserving the raw species label, "
            "for example --atom-alias Ow=O --atom-alias Pt_top=Pt. Repeat as needed."
        ),
    )


def _read_trajectory_with_optional_atom_aliases(read_trajectory_func: Any, path: Any, args: argparse.Namespace):
    atom_aliases = getattr(args, "atom_alias", None)
    if atom_aliases:
        return read_trajectory_func(path, atom_aliases=atom_aliases)
    return read_trajectory_func(path)


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


def _preview_resolve_cell_without_trajectory_read(
    trajectory: str | Path,
    *,
    cell: tuple[float, float, float] | None,
    input_path: str | None,
) -> tuple[tuple[float, float, float] | None, str]:
    """Resolve cell for dry-run without loading trajectory frames."""
    from .resolution import resolve_analysis_cell

    try:
        resolved = resolve_analysis_cell(trajectory, cell=cell, input_path=input_path)
    except (FileNotFoundError, ValueError):
        return None, "unresolved from input and trajectory HDF5 metadata"
    return resolved.cell_angstrom, resolved.source


def _preview_resolve_msd_timestep_without_trajectory_read(
    trajectory: str | Path,
    *,
    timestep_fs: float | None,
    input_path: str | None,
) -> tuple[float, str]:
    """Resolve MSD timestep for dry-run without loading trajectory frames."""
    from .resolution import resolve_analysis_timestep_fs

    try:
        resolved = resolve_analysis_timestep_fs(
            trajectory,
            timestep_fs=timestep_fs,
            input_path=input_path,
        )
    except ValueError:
        return 1.0, "fallback default"
    return resolved.frame_timestep_fs, resolved.source


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


def _resolve_requested_analysis_hdf5_output_path(
    base_output: str | None,
    default_output: str | Path,
) -> Path:
    from .storage.hdf5_utils import resolve_hdf5_output_path

    if base_output is None:
        return resolve_hdf5_output_path(default_output)

    base_path = Path(base_output).expanduser()
    default_path = Path(default_output).expanduser()
    if _output_request_looks_like_directory(base_output) or (
        base_path.exists() and base_path.is_dir()
    ):
        return resolve_hdf5_output_path(base_path / default_path.name)

    if not base_path.suffix:
        base_path = base_path.with_suffix(".h5")
    return resolve_hdf5_output_path(base_path)


def _preflight_prepare_output_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"{label} points to a directory, not a file: '{resolved}'.")
    parent = resolved.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create parent directory for {label} '{resolved}': {exc}") from exc
    if not os.access(parent, os.W_OK):
        raise ValueError(f"Cannot write {label} '{resolved}': parent directory is not writable.")
    if resolved.exists() and not os.access(resolved, os.W_OK):
        raise ValueError(f"Cannot write {label} '{resolved}': file is not writable.")
    return resolved


def _preflight_existing_file_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: '{resolved}'.")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: '{resolved}'.")
    if not os.access(resolved, os.R_OK):
        raise ValueError(f"{label} is not readable: '{resolved}'.")
    return resolved


def _analysis_source_stem(source: str | Path, *, default: str) -> str:
    from .analysis.output_naming import analysis_source_base

    return analysis_source_base(source, default=default)


def _default_analysis_hdf5_output_path(
    source: str | Path,
    analysis: str,
    *,
    default: str = "trajectory",
) -> Path:
    from .analysis.output_naming import analysis_hdf5_filename

    source_path = Path(source).expanduser().resolve()
    return _linak_output_dir_for_source(source_path) / analysis_hdf5_filename(
        source_path,
        analysis,
        default=default,
    )


def _default_density_hdf5_output_path(source: str | Path, species: str) -> Path:
    return _default_analysis_hdf5_output_path(source, "density")


def _density_hdf5_output_path(
    base_output: str | None,
    source: str | Path,
    *,
    species: str,
) -> Path:
    return _resolve_single_analysis_hdf5_output_path(
        base_output,
        _default_density_hdf5_output_path(source, species),
    )


def _default_orientation_hdf5_output_path(source: str | Path, axis: str) -> Path:
    return _default_analysis_hdf5_output_path(source, "orientation")


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


def _is_non_analysis_hdf5(source: str | Path) -> str | None:
    """Return a human-readable label if *source* is a non-analysis LiNaK HDF5, else ``None``."""
    from .cube_io import is_linak_cube_hdf5
    from .trajectory.io import is_linak_trajectory_hdf5

    if is_linak_trajectory_hdf5(source):
        return "trajectory"
    if is_linak_cube_hdf5(source):
        return "cube"
    return None


def _validate_no_non_analysis_hdf5_sources(sources: list[str], *, command_name: str) -> None:
    bad: list[tuple[str, str]] = []
    for source in sources:
        if not _is_hdf5_source(source):
            continue
        kind = _is_non_analysis_hdf5(source)
        if kind is not None:
            bad.append((source, kind))
    if not bad:
        return
    details = ", ".join(f"{Path(s).name} ({k})" for s, k in bad)
    raise ValueError(
        f"{command_name} only accepts LiNaK analysis HDF5 files (density, MSD, RDF, etc.). "
        f"The following file(s) are not analysis outputs: {details}. "
        "Use `linak compute ...` to generate analysis HDF5 from trajectories first."
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
    return _default_analysis_hdf5_output_path(source, "msd")


def _default_temperature_hdf5_output_path(source: str | Path) -> Path:
    return _default_analysis_hdf5_output_path(source, "temperature", default="temperature")


def _default_position_hdf5_output_path(source: str | Path, species: str, axis: str) -> Path:
    return _default_analysis_hdf5_output_path(source, "position")


def _position_hdf5_output_path(
    base_output: str | None,
    source: str | Path,
    profiles: list[Any],
    *,
    axis: str,
) -> Path | None:
    if not profiles:
        return None

    default_path = _default_position_hdf5_output_path(source, "all", axis)
    if base_output is None:
        return _resolve_non_overwriting_hdf5_path(default_path)

    base_path = Path(base_output).expanduser()
    if _output_request_looks_like_directory(base_output) or (base_path.exists() and base_path.is_dir()):
        return _resolve_non_overwriting_hdf5_path(base_path / default_path.name)
    if not base_path.suffix:
        base_path = base_path.with_suffix(".h5")
    return _resolve_non_overwriting_hdf5_path(base_path)


def _default_rdf_collection_hdf5_output_path(source: str | Path) -> Path:
    return _default_analysis_hdf5_output_path(source, "rdf")


def _default_coordination_hdf5_output_path(
    source: str | Path,
    species_a: str,
    species_b: str,
) -> Path:
    return _default_analysis_hdf5_output_path(source, "coordination")


def _default_coordination_collection_hdf5_output_path(source: str | Path) -> Path:
    return _default_analysis_hdf5_output_path(source, "coordination")


def _default_potential_hdf5_output_path(source: str | Path) -> Path:
    return _default_analysis_hdf5_output_path(source, "potential", default="source")


def _default_potential_hdf5_output_for_sources(sources: list[str]) -> Path:
    if len(sources) == 1:
        return _default_potential_hdf5_output_path(sources[0])
    from .analysis.output_naming import combined_analysis_hdf5_filename

    return _linak_output_dir_for_sources(sources) / combined_analysis_hdf5_filename("potential")


def _default_combined_analysis_hdf5_path(sources: list[str], *, analysis: str) -> Path:
    from .analysis.output_naming import combined_analysis_hdf5_filename

    return _linak_output_dir_for_sources(sources) / combined_analysis_hdf5_filename(analysis)


def _unique_path_with_numeric_suffix(path: Path) -> Path:
    from .analysis.output_naming import numbered_hdf5_path

    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = numbered_hdf5_path(path, index)
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


def _is_hdf5_source(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".h5", ".hdf5"}


def _is_raw_text_trajectory_source(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".xyz", ".extxyz", ".dump", ".lmp"}


def _maybe_log_trajectory_convert_hint(path: str | Path) -> None:
    source_path = Path(path).expanduser().resolve()
    if not _is_raw_text_trajectory_source(source_path):
        return
    LOGGER.info(
        "For faster repeated analysis, convert once with "
        "`linak apply convert %s` and compute from the resulting `.traj.h5`.",
        _display_path(source_path),
    )


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


def _coerce_runtime_view_mapping(value: Any) -> Any | None:
    if value is None:
        return None
    from .plot.data_contract import PlotViewMapping
    from .plot.profile_persistence import deserialize_plot_view_mapping

    if isinstance(value, PlotViewMapping):
        return value
    if isinstance(value, dict):
        return deserialize_plot_view_mapping(value)
    raise ValueError("view_mapping must be a PlotViewMapping or mapping payload dictionary.")


def _public_plot_view_type(value: Any | None) -> str | None:
    from .plot.data_contract import PLOT_VIEW_1D_LINE, PLOT_VIEW_2D_HEATMAP

    token = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not token:
        return None
    if token in {"1d", "line", "1d-line", "line-1d", "plot-1d-line"}:
        return PLOT_VIEW_1D_LINE
    if token in {"2d", "heatmap", "2d-heatmap", "heatmap-2d", "plot-2d-heatmap"}:
        return PLOT_VIEW_2D_HEATMAP
    raise ValueError("View type must be one of: 1D Line, 2D Heatmap.")


def _position_component_from_public_args(args: argparse.Namespace) -> str:
    view_type = _public_plot_view_type(getattr(args, "plot_view_type", None))
    from .plot.data_contract import PLOT_VIEW_2D_HEATMAP

    y_quantity = str(getattr(args, "plot_y_quantity", "") or "").strip().lower()
    if view_type == PLOT_VIEW_2D_HEATMAP:
        return "heatmap"
    if y_quantity:
        return y_quantity
    return getattr(args, "component", "distance")


def _coordination_component_from_public_args(args: argparse.Namespace) -> str:
    view_type = _public_plot_view_type(getattr(args, "plot_view_type", None))
    from .plot.data_contract import PLOT_VIEW_2D_HEATMAP

    x_quantity = str(getattr(args, "plot_x_quantity", "") or "").strip().lower()
    if view_type == PLOT_VIEW_2D_HEATMAP:
        return "time-distance"
    if x_quantity == "time":
        return "time"
    if x_quantity == "distance":
        return "distance"
    return getattr(args, "component", "distance")


def _orientation_component_from_public_args(args: argparse.Namespace) -> str:
    view_type = _public_plot_view_type(getattr(args, "plot_view_type", None))
    from .plot.data_contract import PLOT_VIEW_2D_HEATMAP

    y_quantity = str(getattr(args, "plot_y_quantity", "") or "").strip().lower()
    if view_type == PLOT_VIEW_2D_HEATMAP:
        return "heatmap"
    if y_quantity:
        return y_quantity
    return getattr(args, "component", "average")


def _public_mapping_view_label(view_type_id: Any) -> str:
    from .plot.data_contract import plot_view_display_label

    return plot_view_display_label(str(view_type_id or "line_1d").strip() or "line_1d")


def _canonical_mapping_view_id(view_type_id: Any) -> str:
    from .plot.data_contract import canonical_plot_view_id

    return canonical_plot_view_id(str(view_type_id or "line_1d").strip() or "line_1d")


def _mapping_is_2d_heatmap(mapping: Any) -> bool:
    from .plot.data_contract import PLOT_VIEW_2D_HEATMAP

    return (
        _canonical_mapping_view_id(getattr(mapping, "view_type_id", None))
        == PLOT_VIEW_2D_HEATMAP
    )


def _resolve_position_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.position_mapping import resolve_position_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_position_plot_mapping(
        contract=data_contract,
        mapping=mapping,
        component=_position_component_from_public_args(args),
        time_axis=getattr(args, "time_axis", "ps"),
        map_color=getattr(args, "map_color", "distance"),
        projection_x=getattr(args, "projection_x", None),
        projection_y=getattr(args, "projection_y", None),
        projection_value=getattr(args, "projection_value", None),
        projection_render_mode=getattr(args, "projection_render_mode", None),
        projection_filter_min=getattr(args, "projection_filter_min", None),
        projection_filter_max=getattr(args, "projection_filter_max", None),
        xy_z_distance_max=getattr(args, "xy_z_distance_max", None),
    )
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_coordination_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.coordination_mapping import resolve_coordination_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_coordination_plot_mapping(
        contract=data_contract,
        mapping=mapping,
        component=_coordination_component_from_public_args(args),
        time_axis=getattr(args, "time_axis", "ps"),
    )
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_density_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
    view_type: str | None = None,
) -> dict[str, Any]:
    from .plot.mappings.density_mapping import resolve_density_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_density_plot_mapping(
        contract=data_contract,
        mapping=mapping,
        view_type=view_type or "line_1d",
        x_mode=getattr(args, "x_mode", "distance"),
        quantity=getattr(args, "quantity", "mass"),
    )
    resolved_mapping = resolved.mapping
    if mapping is None:
        fixed_values = dict(resolved_mapping.fixed_values)
        fixed_values["quantity"] = str(getattr(args, "quantity", "mass") or "mass").strip().lower()
        resolved_mapping = replace(resolved_mapping, fixed_values=fixed_values)
    payload: dict[str, Any] = {"view_mapping": resolved_mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_msd_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.msd_mapping import resolve_msd_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_msd_plot_mapping(
        contract=data_contract,
        mapping=mapping,
        time_axis=getattr(args, "time_axis", "ps"),
    )
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_temperature_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.temperature_mapping import resolve_temperature_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_temperature_plot_mapping(
        contract=data_contract,
        mapping=mapping,
        time_axis=getattr(args, "time_axis", "ps"),
    )
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_rdf_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.rdf_mapping import resolve_rdf_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_rdf_plot_mapping(contract=data_contract, mapping=mapping)
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_potential_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.potential_mapping import resolve_potential_plot_mapping

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    resolved = resolve_potential_plot_mapping(
        contract=data_contract,
        mapping=mapping,
        y_quantity=getattr(args, "y_quantity", None),
    )
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_orientation_plotter_kwargs(
    args: argparse.Namespace,
    *,
    data_contract: Any | None = None,
) -> dict[str, Any]:
    from .plot.mappings.orientation_mapping import (
        _ORIENTATION_COMPONENTS,
        resolve_orientation_plot_mapping,
    )

    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    if mapping is None:
        raw_component = _orientation_component_from_public_args(args)
        component = raw_component if raw_component in _ORIENTATION_COMPONENTS else "average"
        resolved = resolve_orientation_plot_mapping(
            contract=data_contract,
            component=component,
            angle=getattr(args, "angle", "polar"),
            line_x_axis=getattr(args, "orientation_line_x_axis", None),
            heatmap_x_axis=getattr(args, "orientation_heatmap_x_axis", None),
            heatmap_y_axis=getattr(args, "orientation_heatmap_y_axis", None),
        )
    else:
        resolved = resolve_orientation_plot_mapping(
            contract=data_contract,
            mapping=mapping,
        )
    payload: dict[str, Any] = {"view_mapping": resolved.mapping}
    if data_contract is not None:
        payload["data_contract"] = resolved.contract
    return payload


def _resolve_position_projection_estimation_settings(
    args: argparse.Namespace,
) -> _ResolvedPositionProjectionEstimate:
    from .plot.mappings.position_mapping import resolve_position_plot_mapping

    resolved_mapping = resolve_position_plot_mapping(
        mapping=_coerce_runtime_view_mapping(getattr(args, "view_mapping", None)),
        component=_position_component_from_public_args(args),
        time_axis=getattr(args, "time_axis", "ps"),
        map_color=getattr(args, "map_color", "distance"),
        projection_x=getattr(args, "projection_x", None),
        projection_y=getattr(args, "projection_y", None),
        projection_value=getattr(args, "projection_value", None),
        projection_render_mode=getattr(args, "projection_render_mode", None),
        projection_filter_min=getattr(args, "projection_filter_min", None),
        projection_filter_max=getattr(args, "projection_filter_max", None),
        xy_z_distance_max=getattr(args, "xy_z_distance_max", None),
    )
    mapping = resolved_mapping.mapping
    value_quantity = mapping.color or mapping.filter_by or "distance_to_surface"
    render_mode = (
        str(
            mapping.fixed_values.get("projection_render_mode")
            or ("color-scale" if mapping.color is not None else "line-colors")
        ).strip()
        or "color-scale"
    )
    return _ResolvedPositionProjectionEstimate(
        mapping=mapping,
        projection_x=_position_projection_token_from_quantity_id(mapping.x),
        projection_y=_position_projection_token_from_quantity_id(mapping.y),
        projection_value=_position_projection_token_from_quantity_id(str(value_quantity)),
        render_mode=render_mode,
        filter_min=mapping.filter_min,
        filter_max=mapping.filter_max,
    )


def _position_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_position_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    view_label = _public_mapping_view_label(view_type_id)
    if _mapping_is_2d_heatmap(mapping):
        render_mode = (
            str(
                mapping.fixed_values.get("projection_render_mode")
                or ("color-scale" if mapping.color is not None else "line-colors")
            ).strip()
            or "color-scale"
        )
        value_role = str(mapping.color or mapping.filter_by or "distance_to_surface")
        filter_text = ""
        if mapping.filter_min is not None or mapping.filter_max is not None:
            filter_text = (
                f", filter={value_role}["
                f"{'' if mapping.filter_min is None else mapping.filter_min}, "
                f"{'' if mapping.filter_max is None else mapping.filter_max}]"
            )
        return (
            f"view type={view_label}, x={mapping.x}, y={mapping.y}, "
            f"value={value_role}, render_mode={render_mode}{filter_text}"
        )
    return f"view type={view_label}, x={mapping.x}, y={mapping.y}"


def _density_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_density_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    view_label = _public_mapping_view_label(view_type_id)
    if _mapping_is_2d_heatmap(mapping):
        resolved_roles = mapping.resolved_role_assignments()
        z_role = resolved_roles.get("z")
        return (
            f"view type={view_label}, x={mapping.x}, y={mapping.y}, "
            f"z={z_role if z_role is not None else '<unassigned>'}"
        )
    quantity = str(mapping.y or "").strip() or "density"
    x_mode = str(mapping.fixed_values.get("x_mode") or "distance").strip() or "distance"
    return f"view type={view_label}, x={mapping.x}, y={quantity}, x_mode={x_mode}"


def _msd_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_msd_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    return f"view type={_public_mapping_view_label(view_type_id)}, x={mapping.x}, y={mapping.y}"


def _rdf_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_rdf_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    return f"view type={_public_mapping_view_label(view_type_id)}, x={mapping.x}, y={mapping.y}"


def _coordination_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_coordination_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    parts = [f"view type={_public_mapping_view_label(view_type_id)}", f"x={mapping.x}", f"y={mapping.y}"]
    resolved_roles = mapping.resolved_role_assignments()
    if "color" in resolved_roles:
        parts.append(f"color={resolved_roles['color']}")
    return ", ".join(parts)


def _orientation_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_orientation_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    parts = [f"view type={_public_mapping_view_label(view_type_id)}", f"x={mapping.x}", f"y={mapping.y}"]
    resolved_roles = mapping.resolved_role_assignments()
    if "z" in resolved_roles:
        parts.append(f"z={resolved_roles['z']}")
    return ", ".join(parts)


def _potential_mapping_summary_for_dry_run(args: argparse.Namespace) -> str:
    mapping = _resolve_potential_plotter_kwargs(args).get("view_mapping")
    if mapping is None:
        return "view type=<unresolved>"
    view_type_id = str(getattr(mapping, "view_type_id", "")).strip().lower() or "line_1d"
    view_label = _public_mapping_view_label(view_type_id)
    fixed_values = getattr(mapping, "fixed_values", {})
    standard_plot = str(fixed_values.get("standard_plot") or "").strip()
    if standard_plot:
        return f"view type={view_label}, x={mapping.x}, standard_plot={standard_plot}"
    return f"view type={view_label}, x={mapping.x}, y={mapping.y}"


def _estimate_position_projection_point_counts(
    profile: Any,
    *,
    resolved_projection: _ResolvedPositionProjectionEstimate,
) -> tuple[int, int]:
    from .analysis.position import _position_projection_quantity_data

    x_matrix, _ = _position_projection_quantity_data(
        profile,
        quantity=resolved_projection.projection_x,
    )
    y_matrix, _ = _position_projection_quantity_data(
        profile,
        quantity=resolved_projection.projection_y,
    )
    value_matrix, _ = _position_projection_quantity_data(
        profile,
        quantity=resolved_projection.projection_value,
    )
    visible_mask = np.isfinite(x_matrix) & np.isfinite(y_matrix) & np.isfinite(value_matrix)
    raw_candidate_points = int(np.count_nonzero(visible_mask))
    filter_min = resolved_projection.filter_min
    filter_max = resolved_projection.filter_max
    if filter_min is not None:
        visible_mask &= value_matrix >= float(filter_min)
    if filter_max is not None:
        visible_mask &= value_matrix <= float(filter_max)
    final_visible_points = int(np.count_nonzero(visible_mask))
    return raw_candidate_points, final_visible_points


def _position_projection_dataset_name(quantity: str) -> str:
    token = str(quantity or "").strip().lower().replace("_", "-").replace(" ", "-")
    if token in {"distance", "surface-distance", "dist", "distance-to-surface"}:
        return "distance_to_surface_A"
    if token in {"x", "y", "z"}:
        return f"{token}_A"
    if token == "ps":
        return "time_ps"
    if token == "fs":
        return "time_fs"
    if token == "step":
        return "step"
    if token == "frame":
        return "frame_index"
    return "distance_to_surface_A"


def _position_projection_estimate_dataset_names(
    resolved_projection: _ResolvedPositionProjectionEstimate,
) -> tuple[str, ...]:
    names = {
        "atom_indices",
        _position_projection_dataset_name(resolved_projection.projection_x),
        _position_projection_dataset_name(resolved_projection.projection_y),
        _position_projection_dataset_name(resolved_projection.projection_value),
    }
    return tuple(sorted(names))


def _position_projection_matrix_from_payload(
    datasets: Mapping[str, Any],
    *,
    quantity: str,
    n_frames: int,
    n_atoms: int,
) -> np.ndarray | None:
    dataset_name = _position_projection_dataset_name(quantity)
    if dataset_name not in datasets:
        return None
    values = np.asarray(datasets[dataset_name], dtype=float)
    if dataset_name in {"x_A", "y_A", "z_A", "distance_to_surface_A"}:
        if values.ndim != 2:
            return None
        return values
    if values.ndim == 2:
        return values
    if values.ndim != 1 or n_frames <= 0 or n_atoms <= 0 or values.size != n_frames:
        return None
    return np.repeat(values[:, np.newaxis], n_atoms, axis=1)


def _estimate_position_projection_point_counts_from_payload(
    datasets: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    resolved_projection: _ResolvedPositionProjectionEstimate,
) -> tuple[int, int]:
    atom_indices = np.asarray(datasets.get("atom_indices", []), dtype=int)
    try:
        n_frames = int(metadata.get("n_frames", 0) or 0)
    except (TypeError, ValueError):
        n_frames = 0
    try:
        n_atoms = int(metadata.get("n_atoms", 0) or 0)
    except (TypeError, ValueError):
        n_atoms = 0
    if atom_indices.size > 0:
        n_atoms = int(atom_indices.size)

    x_matrix = _position_projection_matrix_from_payload(
        datasets,
        quantity=resolved_projection.projection_x,
        n_frames=n_frames,
        n_atoms=n_atoms,
    )
    y_matrix = _position_projection_matrix_from_payload(
        datasets,
        quantity=resolved_projection.projection_y,
        n_frames=n_frames,
        n_atoms=n_atoms,
    )
    value_matrix = _position_projection_matrix_from_payload(
        datasets,
        quantity=resolved_projection.projection_value,
        n_frames=n_frames,
        n_atoms=n_atoms,
    )
    if x_matrix is None or y_matrix is None or value_matrix is None:
        fallback_points = max(0, int(n_frames)) * max(0, int(n_atoms))
        return fallback_points, fallback_points
    if x_matrix.shape != y_matrix.shape or x_matrix.shape != value_matrix.shape:
        fallback_points = max(0, int(n_frames)) * max(0, int(n_atoms))
        return fallback_points, fallback_points

    visible_mask = np.isfinite(x_matrix) & np.isfinite(y_matrix) & np.isfinite(value_matrix)
    raw_candidate_points = int(np.count_nonzero(visible_mask))
    filter_min = resolved_projection.filter_min
    filter_max = resolved_projection.filter_max
    if filter_min is not None:
        visible_mask &= value_matrix >= float(filter_min)
    if filter_max is not None:
        visible_mask &= value_matrix <= float(filter_max)
    return raw_candidate_points, int(np.count_nonzero(visible_mask))


def _estimate_position_gui_point_counts(
    profiles: Sequence[Any],
    *,
    resolved_projection: _ResolvedPositionProjectionEstimate,
) -> tuple[int, int]:
    if not resolved_projection.is_projection:
        raw_total = 0
        for profile in profiles:
            points = _estimate_points_for_loaded_profile(profile)
            if points is not None:
                raw_total += int(points)
        return raw_total, raw_total

    raw_total = 0
    final_total = 0
    for profile in profiles:
        raw_points, final_points = _estimate_position_projection_point_counts(
            profile,
            resolved_projection=resolved_projection,
        )
        raw_total += raw_points
        final_total += final_points
    return raw_total, final_total


def _coerce_positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(resolved) or resolved <= 0:
        return None
    return resolved


def _position_descriptor_n_frames(descriptor: Mapping[str, Any]) -> int:
    try:
        n_frames = int(descriptor.get("n_frames", 0) or 0)
    except (TypeError, ValueError):
        n_frames = 0
    return max(0, n_frames)


def _position_descriptor_time_span(
    descriptor: Mapping[str, Any],
    *,
    time_axis: str,
) -> float:
    n_frames = _position_descriptor_n_frames(descriptor)
    if n_frames <= 1:
        return 0.0
    axis = str(time_axis or "ps").strip().lower()
    try:
        timestep_fs = float(descriptor.get("frame_timestep_fs", 1000.0) or 1000.0)
    except (TypeError, ValueError):
        timestep_fs = 1000.0
    if not np.isfinite(timestep_fs) or timestep_fs <= 0:
        timestep_fs = 1000.0
    if axis == "fs":
        return float(n_frames - 1) * timestep_fs
    if axis == "ps":
        return float(n_frames - 1) * timestep_fs / 1000.0
    return float(n_frames - 1)


def _estimate_position_line_points_from_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    x_bin_width: Any,
    time_axis: str,
) -> int:
    width = _coerce_positive_float_or_none(x_bin_width)
    total = 0
    for descriptor in descriptors:
        n_frames = _position_descriptor_n_frames(descriptor)
        if n_frames <= 0:
            continue
        if width is None:
            total += n_frames
            continue
        span = _position_descriptor_time_span(descriptor, time_axis=time_axis)
        if span <= 0:
            total += 1
        else:
            total += min(n_frames, max(1, int(math.ceil(span / width)) + 1))
    return int(total)


def _round_up_display_bin_width(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    normalized = value / scale
    for step in (1.0, 2.0, 5.0, 10.0):
        if normalized <= step:
            return step * scale
    return 10.0 * scale


def _position_descriptor_profile_key(descriptor: Mapping[str, Any]) -> str:
    source = str(descriptor.get("load_source_path") or descriptor.get("origin_path") or "")
    profile_uid = str(
        descriptor.get("profile_uid")
        or descriptor.get("source_series_id")
        or descriptor.get("series_id")
        or ""
    )
    rendered_species = str(descriptor.get("rendered_species") or descriptor.get("default_label") or "")
    return f"{source}|{profile_uid}|{rendered_species}"


def _select_round_robin_position_series_ids(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> set[str]:
    if limit <= 0:
        return set()
    grouped: dict[str, list[str]] = {}
    group_order: list[str] = []
    for index, descriptor in enumerate(descriptors):
        series_id = str(descriptor.get("series_id") or f"series:{index}")
        group_key = _position_descriptor_profile_key(descriptor)
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        grouped[group_key].append(series_id)

    selected: set[str] = set()
    round_index = 0
    while len(selected) < limit:
        added = False
        for group_key in group_order:
            group_ids = grouped[group_key]
            if round_index >= len(group_ids):
                continue
            selected.add(group_ids[round_index])
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        round_index += 1
    return selected


def _format_gui_count(value: int | None) -> str:
    return "unknown" if value is None else f"{int(value):,}"


def _apply_position_gui_auto_display_reduction(
    settings: dict[str, Any],
    *,
    args: argparse.Namespace,
    initial_context: _GuiPlotRenderContext,
) -> dict[str, Any]:
    descriptors = [
        dict(item) for item in settings.get("series_descriptors", []) if isinstance(item, dict)
    ]
    if not descriptors:
        return settings
    resolved_projection = _resolve_position_projection_estimation_settings(args)
    time_axis = str(settings.get("time_axis") or getattr(args, "time_axis", "ps") or "ps")
    raw_series = len(descriptors)
    raw_points = initial_context.estimated_total_points
    if raw_points is None:
        raw_points = _estimate_position_line_points_from_descriptors(
            descriptors,
            x_bin_width=None,
            time_axis=time_axis,
        )
    if (
        raw_series <= _POSITION_GUI_AUTO_DISPLAY_TRIGGER_SERIES
        and raw_points <= _GUI_COMPLEXITY_MAX_POINTS
    ):
        return settings

    reduced = dict(settings)
    existing_overrides = _coerce_series_override_map(reduced.get("series_overrides"))
    enabled_ids = {str(descriptor.get("series_id") or "") for descriptor in descriptors}
    if raw_series > _POSITION_GUI_AUTO_DISPLAY_TRIGGER_SERIES:
        enabled_ids = _select_round_robin_position_series_ids(
            descriptors,
            limit=_POSITION_GUI_AUTO_DISPLAY_SERIES,
        )
    enabled_ids.discard("")

    overrides: dict[str, dict[str, Any]] = {}
    for index, descriptor in enumerate(descriptors):
        series_id = str(descriptor.get("series_id") or f"series:{index}")
        entry = dict(existing_overrides.get(series_id, {}))
        entry["enabled"] = series_id in enabled_ids
        overrides[series_id] = entry
    reduced["series_overrides"] = overrides
    reduced.pop("series_enabled", None)

    enabled_descriptors = [
        descriptor
        for index, descriptor in enumerate(descriptors)
        if str(descriptor.get("series_id") or f"series:{index}") in enabled_ids
    ]
    initial_series = len(enabled_descriptors)
    existing_width = _coerce_positive_float_or_none(
        reduced.get("x_bin_width", getattr(args, "x_bin_width", None))
    )
    final_width = existing_width
    width_was_raised = False
    if not resolved_projection.is_projection and initial_series > 0:
        target_bins_per_series = max(
            1,
            int(
                math.floor(
                    _POSITION_GUI_AUTO_DISPLAY_TARGET_POINTS / max(initial_series, 1)
                )
            ),
        )
        max_span = max(
            (
                _position_descriptor_time_span(descriptor, time_axis=time_axis)
                for descriptor in enabled_descriptors
            ),
            default=0.0,
        )
        if max_span > 0:
            required_width = _round_up_display_bin_width(max_span / target_bins_per_series)
            candidate_points = _estimate_position_line_points_from_descriptors(
                enabled_descriptors,
                x_bin_width=existing_width,
                time_axis=time_axis,
            )
            if existing_width is None or candidate_points > _POSITION_GUI_AUTO_DISPLAY_TARGET_POINTS:
                final_width = max(existing_width or 0.0, required_width)
                width_was_raised = existing_width is not None and final_width > existing_width
                reduced["x_bin_width"] = float(final_width)
                reduced["time_section_width"] = float(final_width)
        if not reduced.get("x_bin_reducer"):
            reduced["x_bin_reducer"] = "mean"

    estimated_points = (
        _estimate_position_line_points_from_descriptors(
            enabled_descriptors,
            x_bin_width=final_width,
            time_axis=time_axis,
        )
        if not resolved_projection.is_projection
        else min(raw_points, _POSITION_GUI_AUTO_DISPLAY_TARGET_POINTS)
    )
    if not enabled_descriptors and descriptors:
        estimated_points = 0

    width_label = "off"
    if final_width is not None:
        width_label = f"{float(final_width):.6g} {time_axis}"
    note = (
        "Position GUI auto-reduced display: "
        f"raw={raw_series:,} series, ~{_format_gui_count(raw_points)} points; "
        f"initial={initial_series:,} series, ~{_format_gui_count(estimated_points)} points, "
        f"time_section_width={width_label}. "
        "HDF5 data is unchanged; enable more series or clear X bin size in the GUI to render raw data."
    )
    reduced["_auto_display_note"] = note
    LOGGER.warning(note)
    if width_was_raised:
        LOGGER.warning(
            "Raised requested position GUI time_section_width from %.6g to %.6g %s for initial responsiveness.",
            float(existing_width),
            float(final_width),
            time_axis,
        )
    return reduced


def _log_position_projection_guard_debug(
    *,
    stage: str,
    resolved_projection: _ResolvedPositionProjectionEstimate,
    raw_candidate_points: int,
    final_visible_points: int,
) -> None:
    if not resolved_projection.is_projection:
        return
    LOGGER.debug(
        "position projection guard at %s: value=%s, render_mode=%s, filter_min=%s, "
        "filter_max=%s, raw_candidate_points=%d, final_visible_points=%d",
        stage,
        resolved_projection.projection_value,
        resolved_projection.render_mode,
        resolved_projection.filter_min,
        resolved_projection.filter_max,
        raw_candidate_points,
        final_visible_points,
    )


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
            source_pairs.extend(_rdf_pair_aliases(species_a, species_b))
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


def _load_density_heatmap_plot_profiles(
    *,
    sources: list[str],
    species: str | None,
    plane: str | None,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.density import load_density_heatmap_profiles, _density_payload_matches_selection
    from .analysis.common import is_molecule_species_label, molecule_display_label

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
    filtered_payloads_by_source: list[tuple[str, list[dict[str, Any]]]] = []
    for source in sources:
        profiles = load_density_heatmap_profiles(source, species=species, plane=plane)
        profiles_by_source.append((source, profiles))
    for source, source_payloads in raw_payloads_by_source:
        filtered_payloads = [
            payload
            for payload in source_payloads
            if _density_payload_matches_selection(
                dict(payload.get("metadata", {})),
                species=species,
                plane=plane,
                profile_kind="heatmap_2d",
            )
        ]
        filtered_payloads_by_source.append((source, filtered_payloads))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = filtered_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("Density heatmap metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                display_species = (
                    molecule_display_label(profile.species)
                    if is_molecule_species_label(profile.species)
                    else profile.species
                )
                rendered_species = f"{source_label}:{display_species}"
                source_labels.append(f"{rendered_species} {profile.plane.upper()}")
                source_ids.append(
                    _profile_uid_from_payload(
                        payload, fallback_prefix="density_heatmap", index=profile_index
                    )
                )
                source_origins.append(str(metadata.get("origin_hdf5_path") or source))
                plot_profiles.append(replace(profile, species=rendered_species))
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        plot_profiles.extend(flattened)
        fallback_labels_by_source.append(
            [
                (
                    f"{molecule_display_label(profile.species)} {profile.plane.upper()}"
                    if is_molecule_species_label(profile.species)
                    else f"{profile.species} {profile.plane.upper()}"
                )
                for profile in flattened
            ]
        )
        raw_payloads = filtered_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("Density heatmap metadata does not match loaded profiles.")
        series_id_segments_by_source.append(
            [
                _profile_uid_from_payload(
                    payload, fallback_prefix="density_heatmap", index=profile_index
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


def _normalize_density_x_mode(x_mode: str | None) -> str:
    normalized = str(x_mode or "distance").strip().lower() or "distance"
    if normalized not in {"distance", "x", "y", "z", "axis"}:
        raise ValueError(
            f"Unsupported density x_mode '{x_mode}'. Choose 'distance', 'x', 'y', 'z', or legacy 'axis'."
        )
    return normalized


def _resolve_density_plot_axis_and_x_mode(
    *,
    axis: str | None,
    x_mode: str | None,
) -> tuple[str | None, str]:
    resolved_x_mode = _normalize_density_x_mode(x_mode)
    resolved_axis = None if axis is None or not str(axis).strip() else str(axis).strip().lower()
    if resolved_axis is not None and resolved_axis not in {"x", "y", "z"}:
        raise ValueError("Density plot axis override must be one of x, y, or z.")
    return resolved_axis, resolved_x_mode


def _density_selected_view_type(
    args: argparse.Namespace,
    filter_options: Mapping[str, Any] | None = None,
) -> str:
    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    if mapping is not None:
        return _canonical_mapping_view_id(getattr(mapping, "view_type_id", None))
    view_types_raw = None if filter_options is None else filter_options.get("density_view_types")
    if isinstance(view_types_raw, (list, tuple, set)):
        view_types = {str(value).strip().lower() for value in view_types_raw}
        if "heatmap_2d" in view_types and "line_1d" not in view_types:
            return _canonical_mapping_view_id("heatmap_2d")
    return _canonical_mapping_view_id("line_1d")


def _build_density_profile_filter_options(
    raw_payloads_by_source: list[tuple[str, list[dict[str, Any]]]],
    *,
    axis: str | None,
    species: str | None,
) -> dict[str, Any] | None:
    from .analysis.density import _density_payload_matches_selection
    from .analysis.common import is_molecule_species_label, molecule_display_label, normalize_species_query

    available_modes: list[str] = []
    seen_modes: set[str] = set()
    heatmap_sources: list[dict[str, str]] = []
    seen_heatmap_sources: set[tuple[str, str]] = set()
    grid_sources: list[dict[str, str]] = []
    seen_grid_sources: set[str] = set()
    species_options: list[dict[str, str]] = []
    seen_species: set[str] = set()
    axis_ranges: dict[str, list[float]] = {}
    axis_range_priorities: dict[str, int] = {}
    axis_bin_width_values: dict[str, list[float]] = {axis_id: [] for axis_id in ("x", "y", "z", "distance")}

    def _add_mode(mode: str) -> None:
        if mode not in seen_modes:
            available_modes.append(mode)
            seen_modes.add(mode)

    def _add_species(raw_species: str) -> str:
        candidate = str(raw_species).strip() or "UNKNOWN"
        if candidate != "UNKNOWN":
            _selection_mode, candidate = normalize_species_query(
                candidate,
                allow_h2o=True,
                allow_molecules=True,
            )
        if candidate not in seen_species:
            seen_species.add(candidate)
            species_options.append(
                {
                    "value": candidate,
                    "label": molecule_display_label(candidate)
                    if is_molecule_species_label(candidate)
                    else candidate,
                }
            )
        return candidate

    def _coerce_cell_lengths(metadata: Mapping[str, Any]) -> tuple[float, float, float] | None:
        raw_cell = metadata.get("resolved_cell_angstrom") or metadata.get("pbc_cell_angstrom")
        if not isinstance(raw_cell, Sequence) or isinstance(raw_cell, (str, bytes)):
            return None
        if len(raw_cell) != 3:
            return None
        try:
            cell = tuple(float(value) for value in raw_cell)
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(value) or value <= 0.0 for value in cell):
            return None
        return cell

    def _optional_metadata_float(metadata: Mapping[str, Any], key: str) -> float | None:
        try:
            value = float(metadata.get(key))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _merge_axis_range(axis_id: str, lower: float, upper: float, *, priority: int = 0) -> None:
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
            return
        axis_key = str(axis_id).strip().lower()
        current_priority = axis_range_priorities.get(axis_key)
        if current_priority is None or int(priority) > current_priority:
            axis_ranges[axis_key] = [float(lower), float(upper)]
            axis_range_priorities[axis_key] = int(priority)
            return
        if int(priority) < current_priority:
            return
        current = axis_ranges.get(axis_key)
        if current is None:
            axis_ranges[axis_key] = [float(lower), float(upper)]
        else:
            current[0] = min(current[0], float(lower))
            current[1] = max(current[1], float(upper))

    def _add_axis_bin_width(axis_id: str, width: Any) -> None:
        try:
            value = float(width)
        except (TypeError, ValueError):
            return
        if math.isfinite(value) and value > 0.0:
            axis_bin_width_values.setdefault(str(axis_id).strip().lower(), []).append(value)

    def _add_cell_axis_ranges(metadata: Mapping[str, Any]) -> None:
        cell = _coerce_cell_lengths(metadata)
        if cell is None:
            return
        _merge_axis_range("x", 0.0, cell[0], priority=0)
        _merge_axis_range("y", 0.0, cell[1], priority=0)
        _merge_axis_range("z", 0.0, cell[2], priority=0)
        surface_axis = str(metadata.get("surface_axis") or axis or "z").strip().lower()
        distance_index = {"x": 0, "y": 1, "z": 2}.get(surface_axis, 2)
        _merge_axis_range("distance", 0.0, cell[distance_index], priority=0)

    def _add_grid_distance_range(metadata: Mapping[str, Any]) -> None:
        visible_lower = _optional_metadata_float(metadata, "visible_distance_min_A")
        visible_upper = _optional_metadata_float(metadata, "visible_distance_max_A")
        if visible_lower is not None and visible_upper is not None and visible_lower < visible_upper:
            _merge_axis_range("distance", visible_lower, visible_upper, priority=2)
            return
        grid_shape = metadata.get("grid_shape")
        grid_bin_width = _optional_metadata_float(metadata, "grid_bin_width_A")
        if (
            isinstance(grid_shape, Sequence)
            and not isinstance(grid_shape, (str, bytes))
            and len(grid_shape) >= 4
            and grid_bin_width is not None
            and grid_bin_width > 0.0
        ):
            try:
                distance_bins = int(grid_shape[3])
            except (TypeError, ValueError):
                return
            if distance_bins > 0:
                _merge_axis_range("distance", 0.0, float(distance_bins) * grid_bin_width, priority=1)

    for _source, source_payloads in raw_payloads_by_source:
        for payload in source_payloads:
            metadata = dict(payload.get("metadata", {}))
            _add_cell_axis_ranges(metadata)
            profile_kind = str(metadata.get("profile_kind", "line_1d")).strip().lower() or "line_1d"
            if profile_kind == "grid_3d_sparse":
                if not _density_payload_matches_selection(
                    metadata,
                    species=species,
                    profile_kind="grid_3d_sparse",
                ):
                    continue
                grid_bin_width = metadata.get("grid_bin_width_A")
                for axis_id in ("x", "y", "z", "distance"):
                    _add_axis_bin_width(axis_id, grid_bin_width)
                _add_grid_distance_range(metadata)
                grid_species = _add_species(str(metadata.get("species", "")).strip() or "UNKNOWN")
                grid_key = grid_species
                if grid_key not in seen_grid_sources:
                    grid_sources.append(
                        {
                            "label": grid_species,
                            "species": grid_species,
                        }
                    )
                    seen_grid_sources.add(grid_key)
                for mode in ("distance", "x", "y", "z"):
                    _add_mode(mode)
                continue
            if profile_kind == "heatmap_2d":
                if not _density_payload_matches_selection(
                    metadata,
                    species=species,
                    profile_kind="heatmap_2d",
                ):
                    continue
                heatmap_species = _add_species(str(metadata.get("species", "")).strip() or "UNKNOWN")
                heatmap_plane = str(metadata.get("plane", "")).strip().lower() or "xy"
                plane_axes = metadata.get("plane_axes")
                if not isinstance(plane_axes, Sequence) or isinstance(plane_axes, (str, bytes)):
                    plane_axes = tuple(heatmap_plane[:2])
                if len(plane_axes) >= 1:
                    _add_axis_bin_width(str(plane_axes[0]).strip().lower(), metadata.get("x_bin_width_A"))
                if len(plane_axes) >= 2:
                    _add_axis_bin_width(str(plane_axes[1]).strip().lower(), metadata.get("y_bin_width_A"))
                source_key = (heatmap_species, heatmap_plane)
                if source_key not in seen_heatmap_sources:
                    heatmap_sources.append(
                        {
                            "label": f"{heatmap_species} {heatmap_plane.upper()}",
                            "species": heatmap_species,
                            "plane": heatmap_plane,
                        }
                    )
                    seen_heatmap_sources.add(source_key)
                continue
            if not _density_payload_matches_selection(metadata, axis=axis, species=species):
                continue
            _add_species(str(metadata.get("species", "")).strip() or "UNKNOWN")
            coordinate_mode = str(metadata.get("coordinate_mode", "axis")).strip().lower()
            axis_value = str(metadata.get("axis", "z")).strip().lower() or "z"
            if coordinate_mode == "distance":
                mode = "distance"
            else:
                mode = axis_value if axis_value in {"x", "y", "z"} else "z"
            _add_axis_bin_width(mode, metadata.get("bin_width_A"))
            _add_mode(mode)
    species_options = _deduplicate_density_species_options(species_options)
    allowed_species_options = {
        str(option.get("value") or "").strip()
        for option in species_options
        if str(option.get("value") or "").strip()
    }
    if allowed_species_options:
        heatmap_sources = [
            source
            for source in heatmap_sources
            if str(source.get("species") or "").strip() in allowed_species_options
        ]
        grid_sources = [
            source
            for source in grid_sources
            if str(source.get("species") or "").strip() in allowed_species_options
        ]
    payload: dict[str, Any] = {}
    if available_modes:
        payload.update(
            {
                "density_x_modes": list(available_modes),
                "available_modes": list(available_modes),
            }
        )
    if species_options:
        payload["density_species_options"] = species_options
    if heatmap_sources:
        payload["density_heatmap_sources"] = heatmap_sources
    if grid_sources:
        payload["density_grid_sources"] = grid_sources
    if axis_ranges:
        payload["density_axis_ranges"] = axis_ranges
    default_axis_bin_widths = {
        axis_id: max(values)
        for axis_id, values in axis_bin_width_values.items()
        if values
    }
    if default_axis_bin_widths:
        payload["density_default_axis_bin_widths_A"] = default_axis_bin_widths
        if "x" in default_axis_bin_widths:
            payload["density_default_x_bin_width_A"] = default_axis_bin_widths["x"]
        if "y" in default_axis_bin_widths:
            payload["density_default_y_bin_width_A"] = default_axis_bin_widths["y"]
    has_2d = bool(heatmap_sources or grid_sources)
    if available_modes and has_2d:
        payload["density_view_types"] = ["line_1d", "heatmap_2d"]
    elif has_2d:
        payload["density_view_types"] = ["heatmap_2d"]
    elif available_modes:
        payload["density_view_types"] = ["line_1d"]
    return payload or None


def _resolve_density_outputs_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "heatmap_planes", None):
        raise ValueError(
            "density --heatmap-planes is no longer used. Use --outputs 3d for sparse "
            "grid-backed 2D slicing in the GUI."
        )
    requested_outputs = getattr(args, "outputs", None)
    requested = str(requested_outputs or "all").strip().lower()
    if requested == "line":
        return "1d"
    if requested not in {"1d", "3d", "all"}:
        raise ValueError("density --outputs must be one of: 1d, 3d, all (legacy alias: line).")
    return requested


def _density_profile_mode_from_metadata(metadata: Mapping[str, Any]) -> str:
    coordinate_mode = str(metadata.get("coordinate_mode", "axis")).strip().lower()
    axis_value = str(metadata.get("axis", "z")).strip().lower() or "z"
    if coordinate_mode == "distance":
        return "distance"
    return axis_value if axis_value in {"x", "y", "z"} else "z"


def _density_effective_render_mode(*, axis: str | None, x_mode: str) -> str:
    normalized_x_mode = _normalize_density_x_mode(x_mode)
    if normalized_x_mode in {"distance", "x", "y", "z"}:
        return normalized_x_mode
    if axis in {"x", "y", "z"}:
        return axis
    return "z"


def _density_grid_filters_from_args(args: argparse.Namespace) -> dict[str, tuple[float | None, float | None]]:
    filters: dict[str, tuple[float | None, float | None]] = {}
    for axis_id in ("x", "y", "z", "distance"):
        lower = getattr(args, f"density_filter_{axis_id}_min", None)
        upper = getattr(args, f"density_filter_{axis_id}_max", None)
        if lower is None and upper is None:
            continue
        filters[axis_id] = (
            None if lower is None else float(lower),
            None if upper is None else float(upper),
        )
    return filters


def _density_grid_2d_axes_from_args(args: argparse.Namespace) -> tuple[str, str]:
    x_axis = str(getattr(args, "density_2d_x_axis", None) or "x").strip().lower()
    y_axis = str(getattr(args, "density_2d_y_axis", None) or "y").strip().lower()
    return x_axis, y_axis


def _density_logical_series_id(*, source_path: str, species: str) -> str:
    return f"density:{Path(source_path).expanduser().resolve()}:{species.strip()}"


def _deduplicate_grouped_density_species_labels(labels: Sequence[str]) -> list[str]:
    """Hide raw-species duplicates unless an element is split into raw labels."""

    from .analysis.common import (
        infer_element_from_raw_label,
        is_molecule_species_label,
        normalize_species_query,
        species_selector_raw_label,
    )

    ordered_labels: list[str] = []
    seen: set[str] = set()
    for raw_label in labels:
        candidate = str(raw_label).strip()
        if not candidate:
            continue
        if candidate != "UNKNOWN":
            try:
                _mode, candidate = normalize_species_query(
                    candidate,
                    allow_h2o=True,
                    allow_molecules=True,
                )
            except ValueError:
                pass
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_labels.append(candidate)

    element_labels: set[str] = set()
    raw_species_by_element: dict[str, set[str]] = {}
    raw_species_to_element: dict[str, str | None] = {}
    for label in ordered_labels:
        if label == "UNKNOWN" or is_molecule_species_label(label):
            continue
        if label.lower().startswith("species:"):
            raw_species = species_selector_raw_label(label)
            element = infer_element_from_raw_label(raw_species)
            raw_species_to_element[label] = element
            if element is not None:
                raw_species_by_element.setdefault(element, set()).add(raw_species)
            continue
        element_labels.add(label)

    kept: list[str] = []
    for label in ordered_labels:
        if not label.lower().startswith("species:"):
            kept.append(label)
            continue
        element = raw_species_to_element.get(label)
        if element is None:
            kept.append(label)
            continue
        if len(raw_species_by_element.get(element, set())) > 1:
            kept.append(label)
            continue
        if element not in element_labels:
            kept.append(label)
    return kept


def _deduplicate_density_species_options(
    species_options: list[dict[str, str]],
) -> list[dict[str, str]]:
    allowed_labels = set(
        _deduplicate_grouped_density_species_labels(
            [str(option.get("value") or "") for option in species_options]
        )
    )
    return [
        dict(option)
        for option in species_options
        if str(option.get("value") or "").strip() in allowed_labels
    ]


def _deduplicate_density_descriptor_segments_by_species(
    descriptor_segments_by_source: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    labels = [
        _density_descriptor_species(descriptor)
        for segment in descriptor_segments_by_source
        for descriptor in segment
    ]
    allowed_labels = set(_deduplicate_grouped_density_species_labels(labels))
    return [
        [
            dict(descriptor)
            for descriptor in segment
            if not _density_descriptor_species(descriptor)
            or _density_descriptor_species(descriptor) in allowed_labels
        ]
        for segment in descriptor_segments_by_source
    ]


def _flatten_descriptor_segments(
    descriptor_segments_by_source: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [dict(descriptor) for segment in descriptor_segments_by_source for descriptor in segment]


def _build_density_logical_descriptor_segments(
    *,
    sources: list[str],
    metadata_by_source: list[tuple[str, list[dict[str, Any]]]],
    axis: str | None,
    species: str | None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any] | None]:
    from .analysis.density import _density_payload_matches_selection, _normalize_species_query
    from .analysis.common import is_molecule_species_label, molecule_display_label

    prefix_source_labels = _should_prefix_combined_source_labels(
        sources=sources,
        metadata_items=[
            dict(metadata)
            for _source, source_metadata in metadata_by_source
            for metadata in source_metadata
        ],
    )
    resolved_species_label: str | None = None
    if species is not None and str(species).strip():
        _selection_mode, candidate_species_label = _normalize_species_query(
            species,
            allow_h2o=True,
            allow_molecules=True,
        )
        if _selection_mode not in {"all", "elements", "molecules"}:
            resolved_species_label = candidate_species_label

    source_group_indices: dict[str, int] = {}
    seen_modes: set[str] = set()
    available_modes: list[str] = []
    descriptor_segments_by_source: list[list[dict[str, Any]]] = []
    for source, metadata_items in metadata_by_source:
        grouped_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
        descriptor_order: list[tuple[str, str]] = []
        for metadata in metadata_items:
            profile_kind = str(metadata.get("profile_kind", "line_1d")).strip().lower() or "line_1d"
            if profile_kind == "heatmap_2d":
                continue
            if profile_kind == "grid_3d_sparse":
                if not _density_payload_matches_selection(
                    metadata,
                    species=species,
                    profile_kind="grid_3d_sparse",
                ):
                    continue
            elif not _density_payload_matches_selection(metadata, axis=axis, species=species):
                continue

            mode = _density_profile_mode_from_metadata(metadata)
            if profile_kind == "grid_3d_sparse":
                mode = "grid"
                for grid_mode in ("distance", "x", "y", "z"):
                    if grid_mode not in seen_modes:
                        available_modes.append(grid_mode)
                        seen_modes.add(grid_mode)
            elif mode not in seen_modes:
                available_modes.append(mode)
                seen_modes.add(mode)

            raw_base_species = (
                resolved_species_label or str(metadata.get("species", "")).strip() or "UNKNOWN"
            )
            if raw_base_species != "UNKNOWN":
                _metadata_mode, base_species = _normalize_species_query(
                    raw_base_species,
                    allow_h2o=True,
                    allow_molecules=True,
                )
            else:
                base_species = raw_base_species
            display_species = (
                molecule_display_label(base_species)
                if is_molecule_species_label(base_species)
                else base_species
            )
            source_label = _metadata_source_label(dict(metadata), fallback_source=source)
            rendered_species = (
                f"{source_label}:{display_species}" if prefix_source_labels else display_species
            )
            resolved_source_path = Path(
                str(metadata.get("origin_hdf5_path") or source)
            ).expanduser()
            resolved_load_source_path = Path(
                str(metadata.get("source_path") or source)
            ).expanduser()
            logical_key = (str(resolved_source_path), base_species)
            if logical_key not in grouped_descriptors:
                source_group_key = str(resolved_source_path)
                resolved_source_index = source_group_indices.setdefault(
                    source_group_key,
                    len(source_group_indices),
                )
                grouped_descriptors[logical_key] = {
                    "series_id": _density_logical_series_id(
                        source_path=str(resolved_source_path),
                        species=base_species,
                    ),
                    "source_kind": "source",
                    "source_series_id": _density_logical_series_id(
                        source_path=str(resolved_source_path),
                        species=base_species,
                    ),
                    "is_generated": False,
                    "source_index": resolved_source_index,
                    "series_index": len(descriptor_order),
                    "source_name": resolved_source_path.name or str(resolved_source_path),
                    "source_directory": (
                        str(resolved_source_path.parent)
                        if str(resolved_source_path.parent) not in {"", "."}
                        else ""
                    ),
                    "source_path": str(resolved_source_path),
                    "load_source_path": str(resolved_load_source_path),
                    "default_label": rendered_species,
                    "density_species": base_species,
                    "density_backing_profiles_by_mode": {},
                    "_density_axis_fallbacks": {},
                }
                descriptor_order.append(logical_key)

            active_profile = {
                "profile_index": int(metadata.get("profile_index", 0)),
                "profile_uid": _profile_uid_from_payload(
                    {"metadata": dict(metadata)},
                    fallback_prefix="density_grid" if profile_kind == "grid_3d_sparse" else "density",
                    index=int(metadata.get("profile_index", 0)),
                ),
                "coordinate_mode": mode,
                "profile_kind": profile_kind,
            }
            if profile_kind == "grid_3d_sparse":
                grouped_descriptors[logical_key]["density_grid_profile"] = active_profile
            else:
                grouped_descriptors[logical_key]["density_backing_profiles_by_mode"][mode] = (
                    active_profile
                )
            if mode == "distance":
                axis_mode = str(metadata.get("axis", "")).strip().lower()
                if axis_mode in {"x", "y", "z"}:
                    grouped_descriptors[logical_key]["_density_axis_fallbacks"].setdefault(
                        axis_mode,
                        dict(active_profile),
                    )

        for key in descriptor_order:
            descriptor = grouped_descriptors[key]
            fallback_profiles = descriptor.pop("_density_axis_fallbacks", {})
            if not isinstance(fallback_profiles, dict):
                continue
            backing_profiles = descriptor.get("density_backing_profiles_by_mode", {})
            if not isinstance(backing_profiles, dict):
                continue
            for fallback_mode, fallback_profile in fallback_profiles.items():
                if fallback_mode in backing_profiles:
                    continue
                backing_profiles[fallback_mode] = dict(fallback_profile)
                if fallback_mode not in seen_modes:
                    available_modes.append(fallback_mode)
                    seen_modes.add(fallback_mode)

        descriptor_segments_by_source.append(
            [dict(grouped_descriptors[key]) for key in descriptor_order]
        )

    return (
        descriptor_segments_by_source,
        {"available_modes": available_modes, "density_x_modes": available_modes}
        if available_modes
        else None,
    )


def _resolve_density_render_descriptor_segments(
    descriptor_segments_by_source: list[list[dict[str, Any]]],
    *,
    axis: str | None,
    x_mode: str,
    x_bin_width: float | None = None,
    grid_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> list[list[dict[str, Any]]]:
    active_mode = _density_effective_render_mode(axis=axis, x_mode=x_mode)
    normalized_grid_filters = dict(grid_filters or {})
    force_grid = bool(normalized_grid_filters)
    resolved_segments: list[list[dict[str, Any]]] = []
    for segment in descriptor_segments_by_source:
        resolved_segment: list[dict[str, Any]] = []
        for descriptor in segment:
            backing_profiles = descriptor.get("density_backing_profiles_by_mode")
            if not isinstance(backing_profiles, dict):
                continue
            active_profile = backing_profiles.get(active_mode)
            resolved_descriptor = dict(descriptor)
            if isinstance(active_profile, dict) and not force_grid:
                resolved_descriptor["profile_kind"] = str(active_profile.get("profile_kind") or "line_1d")
                resolved_descriptor["profile_index"] = int(active_profile["profile_index"])
                resolved_descriptor["profile_uid"] = str(active_profile["profile_uid"])
                resolved_descriptor["active_coordinate_mode"] = active_mode
                resolved_segment.append(resolved_descriptor)
                continue
            grid_profile = descriptor.get("density_grid_profile")
            if isinstance(grid_profile, dict):
                slice_key = json.dumps(
                    {
                        "kind": "line_1d",
                        "x_mode": active_mode,
                        "filters": normalized_grid_filters,
                        "x_bin_width": x_bin_width,
                    },
                    sort_keys=True,
                    default=str,
                )
                resolved_descriptor["profile_kind"] = "grid_3d_sparse"
                resolved_descriptor["profile_index"] = int(grid_profile["profile_index"])
                resolved_descriptor["profile_uid"] = str(grid_profile["profile_uid"])
                resolved_descriptor["active_coordinate_mode"] = active_mode
                resolved_descriptor["density_grid_filters"] = normalized_grid_filters
                resolved_descriptor["density_grid_x_bin_width"] = x_bin_width
                resolved_descriptor["density_grid_slice_key"] = slice_key
                resolved_segment.append(resolved_descriptor)
                continue
            if force_grid and isinstance(active_profile, dict):
                raise ValueError(
                    "Density ranges require a 3D density grid. Recompute density with "
                    "--outputs 3d or --outputs all."
                )
        resolved_segments.append(resolved_segment)
    return resolved_segments


def _load_density_profiles_for_render_descriptors(
    descriptors: list[dict[str, Any]],
    *,
    axis: str | None,
    species: str | None,
) -> list[Any]:
    from .analysis.density import (
        density_grid_to_line_profile,
        load_density_grid_profiles_by_index,
        load_density_profiles_by_index,
    )

    loaded_by_id: dict[str, Any] = {}
    for load_source_path, source_descriptors in _group_descriptors_by_load_source(descriptors):
        line_descriptors = [
            descriptor
            for descriptor in source_descriptors
            if str(descriptor.get("profile_kind", "line_1d")).strip().lower() != "grid_3d_sparse"
        ]
        grid_descriptors = [
            descriptor
            for descriptor in source_descriptors
            if str(descriptor.get("profile_kind", "line_1d")).strip().lower() == "grid_3d_sparse"
        ]
        if line_descriptors:
            indices = [int(descriptor["profile_index"]) for descriptor in line_descriptors]
            profiles = load_density_profiles_by_index(
                load_source_path,
                indices,
                axis=axis,
                species=species,
            )
            if len(profiles) != len(line_descriptors):
                raise ValueError("Density profile metadata does not match loaded profiles.")
            for descriptor, profile in zip(line_descriptors, profiles):
                loaded_by_id[str(descriptor["series_id"])] = replace(
                    profile,
                    species=str(descriptor.get("default_label") or profile.species),
                )
        if grid_descriptors:
            indices = [int(descriptor["profile_index"]) for descriptor in grid_descriptors]
            grid_profiles = load_density_grid_profiles_by_index(
                load_source_path,
                indices,
                species=species,
            )
            if len(grid_profiles) != len(grid_descriptors):
                raise ValueError("Density sparse-grid metadata does not match loaded profiles.")
            for descriptor, grid_profile in zip(grid_descriptors, grid_profiles):
                derived_profile = density_grid_to_line_profile(
                    grid_profile,
                    x_mode=str(descriptor.get("active_coordinate_mode") or "distance"),
                    filters=descriptor.get("density_grid_filters") or {},
                    x_bin_width=descriptor.get("density_grid_x_bin_width"),
                )
                loaded_by_id[str(descriptor["series_id"])] = replace(
                    derived_profile,
                    species=str(descriptor.get("default_label") or derived_profile.species),
                )
    return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]


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


def _load_temperature_plot_profiles(
    *,
    sources: list[str],
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.temperature import load_temperature_profiles

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="temperature",
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
        profiles_by_source.append((source, load_temperature_profiles(source)))

    plot_profiles: list[Any] = []
    fallback_labels_by_source: list[list[str]] = []
    series_id_segments_by_source: list[list[str]] = []
    origin_path_segments_by_source: list[list[str]] = []
    if prefix_source_labels:
        for source_index, (source, profiles) in enumerate(profiles_by_source):
            raw_payloads = raw_payloads_by_source[source_index][1]
            if len(raw_payloads) != len(profiles):
                raise ValueError("Temperature profile metadata does not match loaded profiles.")
            source_labels: list[str] = []
            source_ids: list[str] = []
            source_origins: list[str] = []
            for profile_index, profile in enumerate(profiles):
                payload = raw_payloads[profile_index]
                metadata = dict(payload.get("metadata", {}))
                source_label = _metadata_source_label(metadata, fallback_source=source)
                rendered_label = f"{source_label}:{profile.default_label}"
                source_labels.append(rendered_label)
                source_ids.append(
                    _profile_uid_from_payload(
                        payload,
                        fallback_prefix="temperature",
                        index=profile_index,
                    )
                )
                source_origins.append(str(metadata.get("origin_hdf5_path") or source))
                plot_profiles.append(replace(profile, default_label=rendered_label))
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)
    else:
        flattened = _flatten_profiles_by_source(profiles_by_source)
        plot_profiles.extend(flattened)
        fallback_labels_by_source.append([profile.default_label for profile in flattened])
        raw_payloads = raw_payloads_by_source[0][1]
        if len(raw_payloads) != len(flattened):
            raise ValueError("Temperature profile metadata does not match loaded profiles.")
        series_id_segments_by_source.append(
            [
                _profile_uid_from_payload(
                    payload,
                    fallback_prefix="temperature",
                    index=profile_index,
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
            if not _rdf_pair_matches_cli_filter(
                stored_species_a=meta_species_a,
                stored_species_b=meta_species_b,
                wanted_species_a=wanted_species_a,
                wanted_species_b=wanted_species_b,
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
                rendered_species_b = profile.species_b
                source_labels.append(f"{rendered_species_a}-{rendered_species_b}")
                source_ids.append(
                    _profile_uid_from_payload(payload, fallback_prefix="rdf", index=profile_index)
                )
                source_origins.append(
                    str(payload.get("metadata", {}).get("origin_hdf5_path") or source)
                )
                plot_profiles.append(
                    replace(
                        profile,
                        species_a=rendered_species_a,
                        species_b=rendered_species_b,
                    )
                )
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
    expand_atom_descriptors: bool,
) -> tuple[list[Any], list[list[str]], list[list[str]], list[list[str]]]:
    from .analysis.coordination import (
        _normalize_axis as _normalize_coordination_axis,
        _normalize_species as _normalize_coordination_species,
        load_coordination_profiles,
    )

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
                if not expand_atom_descriptors:
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
            if not expand_atom_descriptors:
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
        help="Input HDF5 path",
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
        help="Input HDF5 file path(s); use -f/--files for multiple",
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
        "--font-size",
        type=_positive_int,
        help="Base font size used when title/label/tick/legend font sizes are unset (default: 12)",
    )
    style_group.add_argument(
        "--title-font-size",
        type=_positive_int,
        help="Title font size (default: inherited from base font size, normally 14)",
    )
    style_group.add_argument(
        "--label-font-size",
        type=_positive_int,
        help="Axis label font size (default: inherited from base font size, normally 12)",
    )
    style_group.add_argument(
        "--tick-font-size",
        type=_positive_int,
        help="Tick label font size (default: inherited from base font size, normally 10)",
    )
    style_group.add_argument(
        "--legend-font-size",
        type=_positive_int,
        help="Legend font size (default: inherited from base font size, normally 10)",
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
        flag="border",
        dest="border",
        feature_name="Plot border",
    )
    style_group.add_argument(
        "--no-border",
        dest="border",
        action="store_false",
        help=argparse.SUPPRESS,
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


def _runtime_view_mapping_was_provided(
    args: argparse.Namespace,
    *,
    profile_key: str,
) -> bool:
    if getattr(args, "view_mapping", None) is not None:
        return True
    mapping_keys_by_profile = {
        "plot:density": ("x_mode", "quantity"),
        "plot:msd": ("time_axis",),
        "plot:rdf": (),
        "plot:position": (
            "component",
            "map_color",
            "projection_x",
            "projection_y",
            "projection_value",
            "projection_render_mode",
            "projection_filter_min",
            "projection_filter_max",
            "xy_z_distance_max",
            "time_axis",
        ),
        "plot:coordination": ("component", "time_axis"),
        "plot:potential": ("y_quantity", "view_type"),
        "plot:orientation": ("component", "angle"),
        "plot:temperature": ("time_axis",),
        "plot:table": ("kind", "x", "y", "bins"),
    }
    return any(
        _runtime_option_was_provided(args, key)
        for key in mapping_keys_by_profile.get(str(profile_key), ())
    )


def _runtime_flag_was_provided(args: argparse.Namespace, *flags: str) -> bool:
    runtime_argv = tuple(getattr(args, "_runtime_argv", ()))
    for token in runtime_argv:
        for flag in flags:
            if token == flag or token.startswith(f"{flag}="):
                return True
    return False


def _parse_atom_index_selection_tokens(raw_tokens: Sequence[str] | None) -> tuple[int, ...]:
    if raw_tokens is None:
        return ()

    resolved: list[int] = []
    seen: set[int] = set()
    for raw_token in raw_tokens:
        token = str(raw_token).strip()
        if not token:
            continue
        normalized = token.replace("{", ",").replace("}", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if not parts:
            continue
        for part in parts:
            if ".." in part:
                start_text, end_text = part.split("..", 1)
                if not start_text.strip() or not end_text.strip():
                    raise ValueError(f"Malformed atom-index range '{part}'.")
                try:
                    start = int(start_text)
                    end = int(end_text)
                except ValueError as exc:
                    raise ValueError(f"Malformed atom-index range '{part}'.") from exc
                if start < 0 or end < 0:
                    raise ValueError(f"Atom indices must be >= 0, got '{part}'.")
                if end < start:
                    raise ValueError(f"Atom-index range end must be >= start, got '{part}'.")
                for value in range(start, end + 1):
                    if value not in seen:
                        seen.add(value)
                        resolved.append(value)
                continue
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(f"Malformed atom index '{part}'.") from exc
            if value < 0:
                raise ValueError(f"Atom indices must be >= 0, got '{part}'.")
            if value not in seen:
                seen.add(value)
                resolved.append(value)
    return tuple(resolved)


def _format_atom_index_selection_label(atom_indices: Sequence[int] | None) -> str:
    if atom_indices is None:
        return "atoms[]"
    values = [int(value) for value in atom_indices]
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


def _resolve_compute_rdf_selectors(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, tuple[int, ...] | None, tuple[int, ...] | None]:
    explicit_species_a = _runtime_flag_was_provided(args, "--species-a")
    explicit_species_b = _runtime_flag_was_provided(args, "--species-b")
    explicit_atoms_a = getattr(args, "atoms_a", None) is not None
    explicit_atoms_b = getattr(args, "atoms_b", None) is not None

    explicit_selector_a = explicit_species_a or explicit_atoms_a
    explicit_selector_b = explicit_species_b or explicit_atoms_b
    if explicit_atoms_b and not explicit_selector_a:
        raise ValueError(
            "RDF atom selector B requires an explicit selector A. Provide --species-a or --atoms-a."
        )

    pairwise_default_mode = not explicit_selector_a and not explicit_selector_b
    if pairwise_default_mode:
        return "pairwise_collection", None, None, None, None

    atoms_a = _parse_atom_index_selection_tokens(getattr(args, "atoms_a", None))
    atoms_b = _parse_atom_index_selection_tokens(getattr(args, "atoms_b", None))
    if explicit_atoms_a and not atoms_a:
        raise ValueError("Provide at least one atom index via --atoms-a.")
    if explicit_atoms_b and not atoms_b:
        raise ValueError("Provide at least one atom index via --atoms-b.")

    if explicit_atoms_a or explicit_atoms_b:
        if explicit_atoms_a:
            selector_species_a = None
            selector_atoms_a = atoms_a
        else:
            selector_species_a = str(args.species_a)
            selector_atoms_a = None

        if explicit_atoms_b:
            selector_species_b = None
            selector_atoms_b = atoms_b
        elif explicit_species_b:
            selector_species_b = str(args.species_b)
            selector_atoms_b = None
        elif selector_atoms_a is not None:
            selector_species_b = None
            selector_atoms_b = selector_atoms_a
        else:
            selector_species_b = selector_species_a
            selector_atoms_b = None

        return (
            "single_pair",
            selector_species_a,
            selector_species_b,
            selector_atoms_a,
            selector_atoms_b,
        )

    if explicit_species_a and explicit_species_b:
        return "single_pair", str(args.species_a), str(args.species_b), None, None
    if explicit_species_a:
        return "species_collection", str(args.species_a), None, None, None
    if explicit_species_b:
        return "species_collection", None, str(args.species_b), None, None
    raise ValueError("Unable to resolve RDF selector mode.")


def _describe_compute_rdf_selector(
    *,
    species: str | None,
    atom_indices: Sequence[int] | None,
) -> str:
    if atom_indices is not None:
        return _format_atom_index_selection_label(atom_indices)
    return str(species)


def _rdf_profile_matches_species_filter(
    profile: Any,
    *,
    species_a: str | None,
    species_b: str | None,
) -> bool:
    label_a = str(getattr(profile, "species_a", "")).strip()
    label_b = str(getattr(profile, "species_b", "")).strip()
    if species_a is not None and species_a not in {label_a, label_b}:
        return False
    if species_b is not None and species_b not in {label_a, label_b}:
        return False
    return True


def _resolve_compute_coordination_pairs(
    frames: Sequence[Any],
    *,
    species_a: str | None,
    species_b: str | None,
) -> list[tuple[str, str]]:
    from .analysis.coordination import _ordered_coordination_pairs_from_frames
    from .analysis.rdf import _normalize_species as _normalize_rdf_species

    available_pairs = _ordered_coordination_pairs_from_frames(list(frames))
    normalized_species_a = (
        None
        if species_a is None or not str(species_a).strip()
        else _normalize_rdf_species(species_a)
    )
    normalized_species_b = (
        None
        if species_b is None or not str(species_b).strip()
        else _normalize_rdf_species(species_b)
    )

    if normalized_species_a is None and normalized_species_b is None:
        raise ValueError("Provide at least one coordination species selector.")

    filtered = [
        (pair_a, pair_b)
        for pair_a, pair_b in available_pairs
        if (normalized_species_a is None or pair_a == normalized_species_a)
        and (normalized_species_b is None or pair_b == normalized_species_b)
    ]
    if filtered:
        return filtered

    if normalized_species_a is not None and normalized_species_b is not None:
        raise ValueError(
            f"No coordination pairs found for species_a={normalized_species_a} and "
            f"species_b={normalized_species_b}."
        )
    if normalized_species_a is not None:
        raise ValueError(
            f"No coordination center species '{normalized_species_a}' found in the trajectory."
        )
    raise ValueError(
        f"No coordination neighbor species '{normalized_species_b}' found in the trajectory."
    )


def _normalize_rdf_pair_tokens(species_a: str, species_b: str) -> tuple[str, str]:
    from .analysis.rdf import _normalize_species as _normalize_rdf_species

    return _normalize_rdf_species(species_a), _normalize_rdf_species(species_b)


def _rdf_pair_aliases(species_a: str, species_b: str) -> list[tuple[str, str]]:
    normalized_a, normalized_b = _normalize_rdf_pair_tokens(species_a, species_b)
    if normalized_a == normalized_b:
        return [(normalized_a, normalized_b)]
    return [(normalized_a, normalized_b), (normalized_b, normalized_a)]


def _rdf_pair_matches_cli_filter(
    *,
    stored_species_a: str,
    stored_species_b: str,
    wanted_species_a: str | None,
    wanted_species_b: str | None,
) -> bool:
    normalized_a, normalized_b = _normalize_rdf_pair_tokens(stored_species_a, stored_species_b)
    if wanted_species_a is not None and wanted_species_b is not None:
        if normalized_a == wanted_species_a and normalized_b == wanted_species_b:
            return True
        return (
            wanted_species_a != wanted_species_b
            and normalized_a == wanted_species_b
            and normalized_b == wanted_species_a
        )
    if wanted_species_a is not None and normalized_a != wanted_species_a:
        return False
    if wanted_species_b is not None and normalized_b != wanted_species_b:
        return False
    return True


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


def _build_saved_plot_profile_payload(
    *,
    profile_key: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    from .plot.profile_persistence import build_plot_profile_payload

    return build_plot_profile_payload(profile_key, settings)


def _read_plot_profile_for_apply(
    path: str | Path,
    *,
    profile_key: str,
    keys: tuple[str, ...],
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    from .plot.plot_settings import read_plot_profile
    from .plot.profile_persistence import select_plot_profile_settings

    payload = read_plot_profile(path, profile_key, profile_name=profile_name)
    if payload is None:
        return None
    return select_plot_profile_settings(
        profile_key,
        payload,
        keys=keys,
    )


def _read_flat_plot_profile(
    path: str | Path,
    *,
    profile_key: str,
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    from .plot.plot_settings import read_plot_profile
    from .plot.profile_persistence import flatten_plot_profile_payload

    payload = read_plot_profile(path, profile_key, profile_name=profile_name)
    if payload is None:
        return None
    return flatten_plot_profile_payload(profile_key, payload)


def _write_flat_plot_profile(
    path: str | Path,
    *,
    profile_key: str,
    settings: dict[str, Any],
    profile_name: str | None = None,
    set_active: bool = True,
) -> None:
    from .plot.plot_settings import write_plot_profile

    write_plot_profile(
        path,
        profile_key,
        _build_saved_plot_profile_payload(profile_key=profile_key, settings=settings),
        profile_name=profile_name,
        set_active=set_active,
    )


def _apply_saved_plot_settings(
    *,
    args: argparse.Namespace,
    source_path: Path,
    profile_key: str,
    keys: tuple[str, ...],
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    try:
        saved = _read_plot_profile_for_apply(
            source_path,
            profile_key=profile_key,
            keys=keys,
            profile_name=profile_name,
        )
    except FileNotFoundError as exc:
        LOGGER.debug("Could not read saved plot settings from '%s': %s", source_path, exc)
        return None
    if saved is None:
        return None

    for key in keys:
        if key not in saved:
            continue
        if key == "view_mapping":
            if _runtime_view_mapping_was_provided(args, profile_key=profile_key):
                continue
            setattr(args, key, deepcopy(saved[key]))
            continue
        if _runtime_option_was_provided(args, key):
            continue
        else:
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
    if normalized == "temperature":
        return _PLOT_PROFILE_TEMPERATURE
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
    if normalized == "temperature":
        return _PLOT_PROFILE_TEMPERATURE
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
    kind = _is_non_analysis_hdf5(source)
    if kind is not None:
        raise ValueError(
            f"This command only accepts LiNaK analysis HDF5 files, but received a {kind} "
            f"HDF5 file: {Path(source).expanduser().resolve()}. "
            "Use `linak compute ...` to generate analysis HDF5 from trajectories first."
        )

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
    render_group.add_argument(
        "--force-gui",
        action="store_true",
        help=(
            "Bypass the interactive GUI size guard and open plot controls even when the "
            "estimated plot complexity is very large."
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
        "--font-size",
        type=_positive_int,
        help="Base font size used when title/label/tick/legend font sizes are unset (default: 12)",
    )
    style_group.add_argument(
        "--title-font-size",
        type=_positive_int,
        help="Title font size (default: inherited from base font size, normally 14)",
    )
    style_group.add_argument(
        "--label-font-size",
        type=_positive_int,
        help="Axis label font size (default: inherited from base font size, normally 12)",
    )
    style_group.add_argument(
        "--tick-font-size",
        type=_positive_int,
        help="Tick label font size (default: inherited from base font size, normally 10)",
    )
    style_group.add_argument(
        "--legend-font-size",
        type=_positive_int,
        help="Legend font size (default: inherited from base font size, normally 10)",
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
        flag="border",
        dest="border",
        feature_name="Plot border",
    )
    style_group.add_argument(
        "--no-border",
        dest="border",
        action="store_false",
        help=argparse.SUPPRESS,
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


_PLOT_PARSER_DESCRIPTION = (
    "Plot LiNaK analysis HDF5 data by auto-detecting density, MSD, RDF, position, "
    "coordination, potential, or orientation from HDF5 metadata. If the input HDF5 is "
    f"not a supported LiNaK analysis file, LiNaK falls back to `{_TABULAR_COMMAND} plot`."
)

_PLOT_PARSER_EPILOG = (
    "Trajectory inputs are intentionally not supported here: run `linak compute ...` "
    "first. For generic tabular HDF5 plotting, use `linak hdf5 plot` directly."
)


def _plot_parser_description(*, analysis: str | None = None) -> str:
    if analysis is None:
        return _PLOT_PARSER_DESCRIPTION
    return (
        f"Detected analysis from input: {analysis}. Showing shared plot options plus "
        f"{analysis}-specific options.\n\n{_PLOT_PARSER_DESCRIPTION}"
    )


def _add_species_override_options(
    parser: argparse.ArgumentParser,
    *,
    group_title: str = "Analysis selection filters",
) -> None:
    group = parser.add_argument_group(group_title)
    group.add_argument(
        "--species",
        default=None,
        help=(
            "Optional species override for density/MSD/position loaded profile labels "
            "(default: use file metadata)"
        ),
    )


def _add_density_plot_options(
    parser: argparse.ArgumentParser,
    *,
    include_axis: bool,
    group_title: str = "Density plot options",
) -> None:
    group = parser.add_argument_group(group_title)
    if include_axis:
        group.add_argument(
            "--axis",
            choices=["x", "y", "z"],
            default=None,
            help=(
                "Optional axis override for loaded density/position/coordination profiles "
                "(default: use file metadata)"
            ),
        )
    group.add_argument(
        "--x-mode",
        choices=["distance", "x", "y", "z", "axis"],
        default="distance",
        help=(
            "Density x-axis values to plot: distance to surface (default) or the stored X/Y/Z "
            "coordinate axis. Legacy 'axis' remains accepted for existing scripts/settings."
        ),
    )
    group.add_argument(
        "--quantity",
        choices=["mass", "number"],
        default="mass",
        help="Density quantity to plot (default: mass; use number for entity number density).",
    )


def _add_rdf_plot_options(
    parser: argparse.ArgumentParser,
    *,
    group_title: str = "RDF plot options",
) -> None:
    group = parser.add_argument_group(group_title)
    group.add_argument(
        "--species-a",
        default=None,
        help=(
            "Optional first-species override for RDF/coordination profiles "
            "(default: use file metadata)"
        ),
    )
    group.add_argument(
        "--species-b",
        default=None,
        help=(
            "Optional second-species override for RDF/coordination profiles "
            "(default: use file metadata or species-a)"
        ),
    )


def _add_temperature_plot_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Temperature plot options")
    group.add_argument(
        "--time-axis",
        choices=["ps", "fs"],
        default="ps",
        help="Time axis for temperature plots (default: ps).",
    )


def _add_position_plot_options(
    parser: argparse.ArgumentParser,
    *,
    include_axis: bool,
    include_component: bool,
    include_map_color: bool,
    include_projection: bool,
    include_time_axis: bool,
    include_time_section_width: bool,
    include_line_y_quantity: bool = False,
    include_line_x_quantity: bool = False,
    group_title: str = "Position plot options",
) -> None:
    group = parser.add_argument_group(group_title)
    if include_axis:
        group.add_argument(
            "--axis",
            choices=["x", "y", "z"],
            default=None,
            help=(
                "Optional axis override for loaded density/position/coordination profiles "
                "(default: use file metadata)"
            ),
        )
    if include_component:
        group.add_argument(
            "--view-type",
            dest="plot_view_type",
            choices=["1d-line", "2d-heatmap", "line", "heatmap", "1d", "2d"],
            default=None,
            help="Plot view type: 1D Line or 2D Heatmap.",
        )
        if include_line_y_quantity:
            group.add_argument(
                "--y-quantity",
                dest="plot_y_quantity",
                choices=["distance", "x", "y", "z"],
                default=None,
                help="Y quantity for 1D Line position plots.",
            )
        if include_line_x_quantity:
            group.add_argument(
                "--x-quantity",
                dest="plot_x_quantity",
                choices=["distance", "time"],
                default=None,
                help="X-axis quantity for 1D Line coordination plots.",
            )
        group.add_argument(
            "--component",
            choices=[
                "distance",
                "x",
                "y",
                "z",
                "xy-z",
                "2d-projection",
                "time",
                "time-distance",
                "average",
                "density-weighted",
                "heatmap",
            ],
            metavar="COMPONENT",
            default="distance",
            help=argparse.SUPPRESS,
        )
    if include_map_color:
        group.add_argument(
            "--map-color",
            choices=["distance", "z"],
            default="distance",
            help=(
                "Legacy compatibility color source for 2D Heatmap mode (default: distance). "
                "Equivalent to --heatmap-value when heatmap-specific settings are not set."
            ),
        )
    if include_projection:
        group.add_argument(
            "--heatmap-x",
            dest="projection_x",
            choices=["x", "y", "z", "distance", "ps", "fs", "step", "frame"],
            default=None,
            help="X-axis quantity for a 2D Heatmap (default: x).",
        )
        group.add_argument(
            "--heatmap-y",
            dest="projection_y",
            choices=["x", "y", "z", "distance", "ps", "fs", "step", "frame"],
            default=None,
            help="Y-axis quantity for a 2D Heatmap (default: y).",
        )
        group.add_argument(
            "--heatmap-value",
            dest="projection_value",
            choices=["x", "y", "z", "distance", "ps", "fs", "step", "frame"],
            default=None,
            help=(
                "Color/filter quantity for a 2D Heatmap. "
                "Defaults to --map-color compatibility behavior."
            ),
        )
        group.add_argument(
            "--heatmap-render-mode",
            dest="projection_render_mode",
            choices=["color-scale", "source-colors", "line-colors"],
            default=None,
            help=(
                "How a 2D Heatmap is rendered: a continuous color scale or source colors. "
                "'line-colors' is accepted as a legacy alias for source colors."
            ),
        )
        group.add_argument(
            "--heatmap-filter-min",
            dest="projection_filter_min",
            type=float,
            default=None,
            help="Optional lower bound for the selected 2D Heatmap value quantity.",
        )
        group.add_argument(
            "--heatmap-filter-max",
            dest="projection_filter_max",
            type=float,
            default=None,
            help="Optional upper bound for the selected 2D Heatmap value quantity.",
        )
        group.add_argument(
            "--projection-x",
            dest="projection_x",
            choices=["x", "y", "z", "distance", "ps", "fs", "step", "frame"],
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        group.add_argument(
            "--projection-y",
            dest="projection_y",
            choices=["x", "y", "z", "distance", "ps", "fs", "step", "frame"],
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        group.add_argument(
            "--projection-value",
            dest="projection_value",
            choices=["x", "y", "z", "distance", "ps", "fs", "step", "frame"],
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        group.add_argument(
            "--projection-render-mode",
            dest="projection_render_mode",
            choices=["color-scale", "source-colors", "line-colors"],
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        group.add_argument(
            "--projection-filter-min",
            dest="projection_filter_min",
            type=float,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        group.add_argument(
            "--projection-filter-max",
            dest="projection_filter_max",
            type=float,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        group.add_argument(
            "--xy-z-distance-max",
            type=_positive_float,
            default=None,
            help=argparse.SUPPRESS,
        )
    if include_time_axis:
        group.add_argument(
            "--time-axis",
            choices=["ps", "fs", "step", "frame"],
            default="ps",
            help="Time axis for position/coordination plots (default: ps).",
        )
    if include_time_section_width:
        group.add_argument(
            "--time-section-width",
            type=_positive_float,
            default=None,
            help=(
                "Optional time-section width for display-only rebinning in position plots. "
                "Equivalent to x-bin width."
            ),
        )


def _add_orientation_plot_options(
    parser: argparse.ArgumentParser,
    *,
    include_component: bool,
    group_title: str = "Orientation plot options",
) -> None:
    group = parser.add_argument_group(group_title)
    if include_component:
        group.add_argument(
            "--view-type",
            dest="plot_view_type",
            choices=["1d-line", "2d-heatmap", "line", "heatmap", "1d", "2d"],
            default=None,
            help="Plot view type: 1D Line or 2D Heatmap.",
        )
        group.add_argument(
            "--y-quantity",
            dest="plot_y_quantity",
            choices=["average", "density", "density-weighted"],
            default=None,
            help="Y quantity for 1D Line orientation plots.",
        )
        group.add_argument(
            "--x-quantity",
            dest="orientation_line_x_axis",
            choices=["distance", "x", "y", "z"],
            default=None,
            help="X-axis quantity for 1D Line orientation plots (default: distance).",
        )
        group.add_argument(
            "--component",
            choices=["average", "density", "density-weighted", "heatmap"],
            default="average",
            help=argparse.SUPPRESS,
        )
    group.add_argument(
        "--angle",
        choices=["polar", "azimuthal"],
        default="polar",
        help=(
            "Orientation angle quantity for 1D Line and 2D Heatmap plots "
            "(default: polar)."
        ),
    )


def _configure_plot_parser(parser: argparse.ArgumentParser, *, analysis: str | None = None) -> None:
    _add_plot_common_options(parser)
    _add_plot_source_options(
        parser,
        help_text="LiNaK analysis HDF5 input (use `linak hdf5 plot` for generic tables)",
    )

    if analysis is None:
        _add_species_override_options(parser)
        _add_density_plot_options(parser, include_axis=True)
        _add_rdf_plot_options(parser)
        _add_position_plot_options(
            parser,
            include_axis=False,
            include_component=True,
            include_map_color=True,
            include_projection=True,
            include_time_axis=True,
            include_time_section_width=True,
            include_line_y_quantity=True,
        )
        _add_orientation_plot_options(parser, include_component=False)
        return

    normalized = str(analysis).strip().lower()
    if normalized == "density":
        _add_species_override_options(parser)
        _add_density_plot_options(parser, include_axis=True)
        return
    if normalized == "msd":
        _add_species_override_options(parser)
        return
    if normalized == "temperature":
        _add_temperature_plot_options(parser)
        return
    if normalized == "rdf":
        _add_rdf_plot_options(parser)
        return
    if normalized == "position":
        _add_species_override_options(parser)
        _add_position_plot_options(
            parser,
            include_axis=True,
            include_component=True,
            include_map_color=True,
            include_projection=True,
            include_time_axis=True,
            include_time_section_width=True,
            include_line_y_quantity=True,
            group_title="Position plot options",
        )
        return
    if normalized == "coordination":
        _add_rdf_plot_options(parser, group_title="Coordination plot options")
        _add_position_plot_options(
            parser,
            include_axis=True,
            include_component=True,
            include_map_color=False,
            include_projection=False,
            include_time_axis=True,
            include_time_section_width=False,
            include_line_x_quantity=True,
            group_title="Coordination plot options",
        )
        return
    if normalized == "orientation":
        _add_orientation_plot_options(parser, include_component=True)
        return
    if normalized == "potential":
        return


def build_plot_parser(*, analysis: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak plot",
        description=_plot_parser_description(analysis=analysis),
        epilog=_PLOT_PARSER_EPILOG,
    )
    _configure_plot_parser(parser, analysis=analysis)
    _add_dry_run_option(parser)
    return parser


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
            "Could not resolve cell from trajectory HDF5 metadata or simulation input before "
            "loading trajectory for %s analysis; checking loaded-frame metadata after load. %s",
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

    if cell_resolution is None and cell is None and input_path is None and has_trajectory_cell:
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

    if cell_resolution is None and cell is None and input_path is None and has_trajectory_cell:
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
            "Could not resolve timestep from trajectory HDF5 metadata or simulation input "
            "before loading trajectory for %s analysis; checking loaded-frame metadata after "
            "load. %s",
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


def _build_plot_style(args: argparse.Namespace) -> PlotStyle:
    from .plot.plotting import with_style_overrides

    figure_size = tuple(args.figsize) if args.figsize is not None else None
    return with_style_overrides(
        figure_size=figure_size,
        dpi=args.dpi,
        font_family=args.font_family,
        font_color=getattr(args, "font_color", None),
        font_size=getattr(args, "font_size", None),
        title_font_size=args.title_font_size,
        title_pad=getattr(args, "title_pad", None),
        label_font_size=args.label_font_size,
        tick_font_size=args.tick_font_size,
        legend_font_size=args.legend_font_size,
        line_width=args.line_width,
        line_color=args.line_color,
        axes_border=getattr(args, "border", None),
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
            enabled_by_id[series_id] = bool(descriptor.get("enabled", True))
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
    *,
    target_descriptors: list[dict[str, Any]] | None = None,
) -> list[str]:
    if count <= 0:
        return []

    def _is_group(descriptor: dict[str, Any]) -> bool:
        return str(descriptor.get("source_kind") or "source").strip().lower() == "group"

    def _family_id(descriptor: dict[str, Any], index: int) -> str:
        return str(
            descriptor.get("source_series_id") or descriptor.get("series_id") or f"series:{index}"
        ).strip()

    palette_descriptors = [
        dict(descriptor)
        for descriptor in (series_descriptors or [])
        if isinstance(descriptor, dict) and not _is_group(descriptor)
    ]
    target_items = [
        dict(descriptor)
        for descriptor in (
            target_descriptors if isinstance(target_descriptors, list) else palette_descriptors
        )
        if isinstance(descriptor, dict) and not _is_group(descriptor)
    ]

    family_order: list[str] = []
    for index, descriptor in enumerate(palette_descriptors):
        family = _family_id(descriptor, index)
        if family and family not in family_order:
            family_order.append(family)

    palette = _default_multi_series_colors(max(len(family_order), len(target_items), count))
    family_colors = {family: palette[index] for index, family in enumerate(family_order)}

    resolved: list[str] = []
    for index, descriptor in enumerate(target_items):
        family = _family_id(descriptor, index)
        resolved.append(family_colors.get(family, palette[index % len(palette)]))

    if len(resolved) == count:
        return resolved
    if len(resolved) > count:
        return resolved[:count]
    return resolved + palette[len(resolved) : count]


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
    source_path = Path(source).expanduser().resolve()
    try:
        return _read_flat_plot_profile(
            source_path,
            profile_key=profile_key,
            profile_name=profile_name,
        )
    except (FileNotFoundError, OSError) as exc:
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
    source_ordered_descriptors = [
        d
        for d in ordered_descriptors
        if str(d.get("source_kind") or "source").strip().lower() != "group"
    ]

    if overrides and len(source_ordered_descriptors) == total_series:
        from .plot.fitting import coerce_fit_config

        override_labels: list[str] = []
        override_colors: list[str] = []
        override_enabled: list[bool] = []
        override_show_in_legend: list[bool] = []
        override_fit_configs: list[dict[str, Any] | None] = []
        override_error_configs: list[dict[str, Any] | None] = []
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
        any_error = False
        any_width = False
        any_marker = False
        any_line_kwargs = False
        for descriptor in source_ordered_descriptors:
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

            fit_config = coerce_fit_config(entry.get("fit"))
            override_fit_configs.append(fit_config if fit_config.get("fit_enabled") else None)
            fit_enabled = bool(fit_config.get("fit_enabled", False))
            any_fit = any_fit or fit_enabled

            error_config = entry.get("error")
            error_config_value = dict(error_config) if isinstance(error_config, dict) else None
            override_error_configs.append(error_config_value)
            any_error = any_error or (error_config_value is not None)

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

        merged_labels = override_labels
        merged_colors = override_colors if any_color else None
        args.series_enabled = override_enabled if any_disabled else None
        args.series_show_in_legend = override_show_in_legend if any_hidden_in_legend else None
        args.series_fit_configs = override_fit_configs if any_fit else None
        args.series_error_configs = override_error_configs if any_error else None
        args.series_line_widths = override_widths if any_width else None
        args.series_markers = override_markers if any_marker else None
        args.series_line_kwargs = override_line_kwargs if any_line_kwargs else None
        args.series_normalization_modes = override_norm_modes
        args.series_normalization_values = override_norm_values
        args.series_normalization_x_refs = override_norm_x_refs
    else:
        args.series_fit_configs = None
        args.series_error_configs = None

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

    source_for_reorder = [
        d
        for d in (list(series_descriptors) if isinstance(series_descriptors, list) else [])
        if str(d.get("source_kind") or "source").strip().lower() != "group"
    ]
    if len(source_for_reorder) != total_series:
        return

    natural_ids = [
        str(item.get("series_id") or f"series:{index}")
        for index, item in enumerate(source_for_reorder)
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
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
    ):
        setattr(args, attr, _reorder_series_values(getattr(args, attr, None), indices))


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
                    "source_kind": "source",
                    "source_series_id": (
                        str(id_segment[local_index]).strip()
                        if id_segment is not None
                        else f"series:{source_index}:{local_index}"
                    ),
                    "is_generated": False,
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
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
        "_gui_sync_modes",
    ):
        if key in saved:
            merged[key] = deepcopy(saved[key])
    return merged


def _materialize_gui_series_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    """Normalize positional per-series GUI state into ID-keyed overrides.

    The GUI should start from the same `series_descriptors + series_overrides`
    model whether it is opening for the first time from runtime args or
    reopening a saved profile. Existing override entries remain authoritative;
    positional lists only fill missing fields.
    """
    descriptors = settings.get("series_descriptors")
    if not isinstance(descriptors, list) or not descriptors:
        return settings

    normalized_descriptors = [dict(item) for item in descriptors if isinstance(item, dict)]
    if not normalized_descriptors:
        return settings

    def _list_or_none(key: str) -> list[Any] | None:
        value = settings.get(key)
        return list(value) if isinstance(value, (list, tuple)) else None

    raw_labels = _list_or_none("series_labels") or []
    raw_colors = _list_or_none("line_colors") or []
    raw_enabled = _list_or_none("series_enabled") or []
    raw_show_in_legend = _list_or_none("series_show_in_legend") or []
    raw_alpha = _list_or_none("series_alpha") or []
    raw_widths = _list_or_none("series_line_widths") or []
    raw_markers = _list_or_none("series_markers") or []
    raw_line_kwargs = _list_or_none("series_line_kwargs") or []
    raw_norm_modes = _list_or_none("series_normalization_modes") or []
    raw_norm_values = _list_or_none("series_normalization_values") or []
    raw_norm_x_refs = _list_or_none("series_normalization_x_refs") or []
    raw_fit_configs = _list_or_none("series_fit_configs") or []
    raw_error_configs = _list_or_none("series_error_configs") or []

    overrides = _coerce_series_override_map(settings.get("series_overrides"))
    has_existing_overrides = bool(overrides)
    any_entry = bool(overrides)

    for index, descriptor in enumerate(normalized_descriptors):
        series_id = str(descriptor.get("series_id") or f"series:{index}").strip()
        if not series_id:
            continue
        entry = dict(overrides.get(series_id, {}))
        default_label = str(descriptor.get("default_label") or "").strip()

        if not has_existing_overrides:
            if index < len(raw_labels):
                label_value = str(raw_labels[index] or "").strip()
                if label_value and label_value != default_label and "label_override" not in entry:
                    entry["label_override"] = label_value

            if index < len(raw_colors):
                color_value = str(raw_colors[index] or "").strip()
                if color_value and "color" not in entry:
                    entry["color"] = color_value

            if "enabled" not in entry:
                entry["enabled"] = bool(raw_enabled[index]) if index < len(raw_enabled) else True

            if "show_in_legend" not in entry:
                entry["show_in_legend"] = (
                    bool(raw_show_in_legend[index]) if index < len(raw_show_in_legend) else True
                )

            if index < len(raw_alpha) and raw_alpha[index] is not None and "alpha" not in entry:
                entry["alpha"] = float(raw_alpha[index])

            if (
                index < len(raw_widths)
                and raw_widths[index] is not None
                and "line_width" not in entry
            ):
                entry["line_width"] = float(raw_widths[index])

            if index < len(raw_markers):
                marker_value = raw_markers[index]
                if marker_value not in {None, ""} and "marker" not in entry:
                    entry["marker"] = str(marker_value)

            if index < len(raw_line_kwargs):
                line_kwargs_value = raw_line_kwargs[index]
                if isinstance(line_kwargs_value, dict) and "line_kwargs" not in entry:
                    entry["line_kwargs"] = deepcopy(line_kwargs_value)

            if index < len(raw_fit_configs):
                fit_config = raw_fit_configs[index]
                if isinstance(fit_config, dict) and "fit" not in entry:
                    entry["fit"] = deepcopy(fit_config)

            if index < len(raw_error_configs):
                error_config = raw_error_configs[index]
                if isinstance(error_config, dict) and "error" not in entry:
                    entry["error"] = deepcopy(error_config)
        else:
            entry.setdefault("enabled", True)
            entry.setdefault("show_in_legend", True)

        if index < len(raw_norm_modes):
            mode_value = raw_norm_modes[index]
            if mode_value not in {None, ""} and "normalization_mode" not in entry:
                entry["normalization_mode"] = str(mode_value)

        if index < len(raw_norm_values) and raw_norm_values[index] is not None:
            entry.setdefault("normalization_value", float(raw_norm_values[index]))

        if index < len(raw_norm_x_refs) and raw_norm_x_refs[index] is not None:
            entry.setdefault("normalization_x_ref", float(raw_norm_x_refs[index]))

        if entry:
            overrides[series_id] = entry
            any_entry = True

    if not any_entry:
        return settings

    normalized = dict(settings)
    normalized["series_overrides"] = overrides
    normalized.pop("series_normalization_modes", None)
    normalized.pop("series_normalization_values", None)
    normalized.pop("series_normalization_x_refs", None)
    return normalized


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
        "series_fit_configs",
        "series_error_configs",
    ):
        cleaned.pop(key, None)
    return cleaned


def _without_preview_series_state(settings: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(settings, dict):
        return {}
    blocked = {
        "series_order",
        "series_overrides",
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
        "integration_config",
    }
    return {key: deepcopy(value) for key, value in settings.items() if key not in blocked}


def _merge_preview_defaults_into_gui_settings(
    settings: dict[str, Any],
    preview_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge preview-derived defaults without overwriting explicit manual GUI choices."""
    if not isinstance(preview_state, dict):
        return dict(settings)
    merged = dict(settings)
    preview_defaults = _without_preview_series_state(preview_state)
    explicit_sync_modes = _derive_gui_sync_modes(settings)
    guarded_keys = {
        "title": "title",
        "title_visible": "title",
        "x_label": "x_label",
        "y_label": "y_label",
        "x_lim": "x_lim",
        "y_lim": "y_lim",
        "x_ticks": "x_ticks",
        "y_ticks": "y_ticks",
        "x_label_pad": "x_label_pad",
        "y_label_pad": "y_label_pad",
    }
    for key, value in preview_defaults.items():
        mode_key = guarded_keys.get(key)
        if mode_key is not None and explicit_sync_modes.get(mode_key, "auto") != "auto":
            continue
        merged[key] = deepcopy(value)
    return merged


def _build_density_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.density_contract import (
        default_density_heatmap_plot_data_contract,
        default_density_plot_data_contract,
        density_heatmap_profile_to_plot_data_contract,
        density_profile_to_plot_data_contract,
    )
    from .plot.mappings.density_mapping import resolve_density_plot_mapping

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="density",
    )
    filter_options = (
        _build_density_profile_filter_options(
            raw_payloads_by_source,
            axis=None,
            species=args.species,
        )
        or {}
    )
    selected_view_type = _density_selected_view_type(args, filter_options)
    resolved_density_mapping = resolve_density_plot_mapping(
        mapping=_coerce_runtime_view_mapping(getattr(args, "view_mapping", None)),
        view_type=selected_view_type,
        x_mode=getattr(args, "x_mode", None),
        quantity=getattr(args, "quantity", "mass"),
    )
    selected_view_type = resolved_density_mapping.view_type_id
    _load_axis, resolved_x_mode = _resolve_density_plot_axis_and_x_mode(
        axis=getattr(args, "axis", None),
        x_mode=resolved_density_mapping.x_mode,
    )
    line_contract = default_density_plot_data_contract()
    heatmap_contract = default_density_heatmap_plot_data_contract()
    common_profile_filter_options = {
        **filter_options,
        "density_plot_contract": _serialize_plot_data_contract(line_contract),
        "density_heatmap_plot_contract": _serialize_plot_data_contract(heatmap_contract),
    }
    if _canonical_mapping_view_id(selected_view_type) == "plot_2d_heatmap":
        (
            heatmap_profiles,
            fallback_labels_by_source,
            series_id_segments_by_source,
            origin_path_segments_by_source,
        ) = _load_density_heatmap_plot_profiles(
            sources=sources,
            species=args.species,
            plane=getattr(args, "plane", None),
        )
        if not heatmap_profiles:
            return _GuiPlotRenderContext(
                profile=[],
                plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
                plotter_kwargs={
                    **_resolve_density_plotter_kwargs(
                        args,
                        data_contract=heatmap_contract,
                        view_type=selected_view_type,
                    ),
                    "heatmap_vmin": getattr(args, "heatmap_vmin", None),
                    "heatmap_vmax": getattr(args, "heatmap_vmax", None),
                    "heatmap_cmap": getattr(args, "heatmap_cmap", None),
                    "heatmap_log_scale": getattr(args, "heatmap_log_scale", False),
                    "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
                    "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
                    "heatmap_colorbar_label_size": getattr(
                        args, "heatmap_colorbar_label_size", None
                    ),
                    "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
                    "heatmap_colorbar_position": getattr(
                        args, "heatmap_colorbar_position", "right"
                    ),
                    "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
                    "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
                    "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
                },
                fallback_labels_by_source=fallback_labels_by_source,
                default_series_labels=None,
                series_descriptors=[],
                profile_filter_options=common_profile_filter_options,
                estimated_total_points=None,
            )
        active_profile = heatmap_profiles[0]
        heatmap_contract = density_heatmap_profile_to_plot_data_contract(active_profile)
        selected_label = next(
            (
                label
                for segment in fallback_labels_by_source
                for label in segment
                if str(label).strip()
            ),
            f"{active_profile.species} {active_profile.plane.upper()}",
        )
        selected_series_id = next(
            (
                series_id
                for segment in series_id_segments_by_source
                for series_id in segment
                if str(series_id).strip()
            ),
            f"density-heatmap:{active_profile.species}:{active_profile.plane}",
        )
        selected_origin = next(
            (
                origin
                for segment in origin_path_segments_by_source
                for origin in segment
                if str(origin).strip()
            ),
            sources[0],
        )
        heatmap_descriptors = _build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=[[selected_label]],
            series_id_segments_by_source=[[selected_series_id]],
            origin_path_segments_by_source=[[selected_origin]],
        )
        return _GuiPlotRenderContext(
            profile=active_profile,
            plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
            plotter_kwargs={
                **_resolve_density_plotter_kwargs(
                    args,
                    data_contract=heatmap_contract,
                    view_type=selected_view_type,
                ),
                "heatmap_vmin": getattr(args, "heatmap_vmin", None),
                "heatmap_vmax": getattr(args, "heatmap_vmax", None),
                "heatmap_cmap": getattr(args, "heatmap_cmap", None),
                "heatmap_log_scale": getattr(args, "heatmap_log_scale", False),
                "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
                "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
                "heatmap_colorbar_label_size": getattr(args, "heatmap_colorbar_label_size", None),
                "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
                "heatmap_colorbar_position": getattr(args, "heatmap_colorbar_position", "right"),
                "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
                "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
                "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
            },
            fallback_labels_by_source=[[selected_label]],
            default_series_labels=[selected_label],
            series_descriptors=heatmap_descriptors,
            profile_filter_options={
                **common_profile_filter_options,
                "density_heatmap_plot_contract": _serialize_plot_data_contract(heatmap_contract),
            },
            estimated_total_points=_estimate_total_points_from_loaded_profiles(active_profile),
        )
    base_filter_options = dict(filter_options)
    logical_descriptor_segments, logical_filter_options = _build_density_logical_descriptor_segments(
        sources=sources,
        metadata_by_source=[
            (source, [dict(payload.get("metadata", {})) for payload in source_payloads])
            for source, source_payloads in raw_payloads_by_source
        ],
        axis=None,
        species=args.species,
    )
    logical_descriptor_segments = _deduplicate_density_descriptor_segments_by_species(
        logical_descriptor_segments
    )
    logical_descriptor_segments = _filter_density_descriptor_segments_by_enabled_species(
        logical_descriptor_segments,
        getattr(args, "density_enabled_species", None),
    )
    render_descriptor_segments = _resolve_density_render_descriptor_segments(
        logical_descriptor_segments,
        axis=getattr(args, "axis", None),
        x_mode=resolved_x_mode,
        x_bin_width=getattr(args, "x_bin_width", None),
        grid_filters=_density_grid_filters_from_args(args),
    )
    render_descriptors = _flatten_descriptor_segments(render_descriptor_segments)
    plot_profiles = _load_density_profiles_for_render_descriptors(
        render_descriptors,
        axis=None,
        species=None,
    )
    density_contract = (
        None if not plot_profiles else density_profile_to_plot_data_contract(plot_profiles[0])
    )
    fallback_labels_by_source = [
        [
            str(descriptor.get("default_label") or f"Series {index + 1}")
            for index, descriptor in enumerate(segment)
        ]
        for segment in render_descriptor_segments
    ]
    profile_filter_options = dict(common_profile_filter_options)
    if density_contract is not None:
        profile_filter_options = {
            **profile_filter_options,
            "density_plot_contract": _serialize_plot_data_contract(density_contract),
        }
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
        plotter_kwargs=_resolve_density_plotter_kwargs(
            args,
            data_contract=density_contract,
            view_type=selected_view_type,
        ),
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_DENSITY,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=render_descriptors,
        profile_filter_options=profile_filter_options,
        estimated_total_points=_estimate_total_points_from_loaded_profiles(plot_profiles),
    )


def _build_density_gui_logical_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.density_contract import (
        default_density_heatmap_plot_data_contract,
        default_density_plot_data_contract,
    )
    from .plot.mappings.density_mapping import resolve_density_plot_mapping

    raw_payloads_by_source = _read_analysis_profile_payloads_by_source(
        sources=sources,
        analysis="density",
    )
    filter_options = (
        _build_density_profile_filter_options(
            raw_payloads_by_source,
            axis=None,
            species=args.species,
        )
        or {}
    )
    selected_view_type = _density_selected_view_type(args, filter_options)
    resolved_density_mapping = resolve_density_plot_mapping(
        mapping=_coerce_runtime_view_mapping(getattr(args, "view_mapping", None)),
        view_type=selected_view_type,
        x_mode=getattr(args, "x_mode", None),
        quantity=getattr(args, "quantity", "mass"),
    )
    selected_view_type = resolved_density_mapping.view_type_id
    _load_axis, resolved_x_mode = _resolve_density_plot_axis_and_x_mode(
        axis=getattr(args, "axis", None),
        x_mode=resolved_density_mapping.x_mode,
    )
    if _canonical_mapping_view_id(selected_view_type) == "plot_2d_heatmap":
        heatmap_contract = default_density_heatmap_plot_data_contract()
        heatmap_descriptor_context = _build_density_gui_lazy_catalog(
            args,
            sources=sources,
        ).build_initial_context()
        return _GuiPlotRenderContext(
            profile=[],
            plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
            plotter_kwargs={
                **_resolve_density_plotter_kwargs(
                    args,
                    data_contract=heatmap_contract,
                    view_type=selected_view_type,
                ),
                "heatmap_vmin": getattr(args, "heatmap_vmin", None),
                "heatmap_vmax": getattr(args, "heatmap_vmax", None),
                "heatmap_cmap": getattr(args, "heatmap_cmap", None),
                "heatmap_log_scale": getattr(args, "heatmap_log_scale", False),
                "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
                "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
                "heatmap_colorbar_label_size": getattr(args, "heatmap_colorbar_label_size", None),
                "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
                "heatmap_colorbar_position": getattr(args, "heatmap_colorbar_position", "right"),
                "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
                "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
                "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
            },
            fallback_labels_by_source=heatmap_descriptor_context.fallback_labels_by_source,
            default_series_labels=list(heatmap_descriptor_context.default_series_labels),
            series_descriptors=heatmap_descriptor_context.series_descriptors,
            profile_filter_options={
                **filter_options,
                "density_plot_contract": _serialize_plot_data_contract(
                    default_density_plot_data_contract()
                ),
                "density_heatmap_plot_contract": _serialize_plot_data_contract(heatmap_contract),
            },
            estimated_total_points=None,
        )
    base_filter_options = dict(filter_options)
    logical_descriptor_segments, logical_filter_options = _build_density_logical_descriptor_segments(
        sources=sources,
        metadata_by_source=[
            (source, [dict(payload.get("metadata", {})) for payload in source_payloads])
            for source, source_payloads in raw_payloads_by_source
        ],
        axis=None,
        species=args.species,
    )
    logical_descriptor_segments = _deduplicate_density_descriptor_segments_by_species(
        logical_descriptor_segments
    )
    logical_descriptor_segments = _filter_density_descriptor_segments_by_enabled_species(
        logical_descriptor_segments,
        getattr(args, "density_enabled_species", None),
    )
    logical_descriptors = _flatten_descriptor_segments(logical_descriptor_segments)
    fallback_labels_by_source = [
        [
            str(descriptor.get("default_label") or f"Series {index + 1}")
            for index, descriptor in enumerate(segment)
        ]
        for segment in logical_descriptor_segments
    ]
    density_contract = default_density_plot_data_contract()
    profile_filter_options = {
        **base_filter_options,
        **(logical_filter_options or {}),
    }
    profile_filter_options = {
        **profile_filter_options,
        "density_plot_contract": _serialize_plot_data_contract(density_contract),
        "density_heatmap_plot_contract": _serialize_plot_data_contract(
            default_density_heatmap_plot_data_contract()
        ),
    }
    return _GuiPlotRenderContext(
        profile=[],
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
        plotter_kwargs=_resolve_density_plotter_kwargs(
            args,
            data_contract=density_contract,
            view_type=selected_view_type,
        ),
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_DENSITY,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=logical_descriptors,
        profile_filter_options=profile_filter_options,
        estimated_total_points=None,
    )


def _build_msd_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.msd_contract import msd_profile_to_plot_data_contract

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
        plotter_kwargs=_resolve_msd_plotter_kwargs(
            args,
            data_contract=(
                None if not plot_profiles else msd_profile_to_plot_data_contract(plot_profiles[0])
            ),
        ),
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
        estimated_total_points=_estimate_total_points_from_loaded_profiles(plot_profiles),
    )


def _build_temperature_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.temperature_contract import temperature_profile_to_plot_data_contract

    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_temperature_plot_profiles(sources=sources)
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_temperature",
        plotter_kwargs=_resolve_temperature_plotter_kwargs(
            args,
            data_contract=(
                None
                if not plot_profiles
                else temperature_profile_to_plot_data_contract(plot_profiles[0])
            ),
        ),
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=sources,
            profile_key=_PLOT_PROFILE_TEMPERATURE,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=_build_gui_series_descriptors(
            sources=sources,
            fallback_labels_by_source=fallback_labels_by_source,
            series_id_segments_by_source=series_id_segments_by_source,
            origin_path_segments_by_source=origin_path_segments_by_source,
        ),
        estimated_total_points=_estimate_total_points_from_loaded_profiles(plot_profiles),
    )


def _build_rdf_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.rdf_contract import rdf_profile_to_plot_data_contract

    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_rdf_plot_profiles(
        sources=sources,
        species_a=None,
        species_b=None,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_rdf",
        plotter_kwargs=_resolve_rdf_plotter_kwargs(
            args,
            data_contract=(
                None if not plot_profiles else rdf_profile_to_plot_data_contract(plot_profiles[0])
            ),
        ),
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
        profile_filter_options=None,
        estimated_total_points=_estimate_total_points_from_loaded_profiles(plot_profiles),
    )


def _build_position_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.position_contract import position_profile_to_plot_data_contract

    # Position is the clearest current example of analysis-specific view
    # mapping entering the flow: component/projection choices influence both
    # profile loading and whether render series stay profile-level or expand to
    # per-atom descriptors.
    resolved_projection = _resolve_position_projection_estimation_settings(args)
    projection_mode = resolved_projection.is_projection
    if _position_projection_uses_profile_descriptors(args, resolved_projection=resolved_projection):
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
            profiles_by_source.append(
                (
                    source,
                    load_position_profiles(source, species=args.species, axis=args.axis),
                )
            )

        plot_profiles: list[Any] = []
        fallback_labels_by_source: list[list[str]] = []
        series_id_segments_by_source: list[list[str]] = []
        origin_path_segments_by_source: list[list[str]] = []
        raw_estimated_total_points = 0
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
                profile_uid = _profile_uid_from_payload(
                    payload,
                    fallback_prefix="position",
                    index=profile_index,
                )
                source_label = _metadata_source_label(metadata, fallback_source=source)
                rendered_species = (
                    f"{source_label}:{profile.species}" if prefix_source_labels else profile.species
                )
                rendered_profile = replace(profile, species=rendered_species)
                plot_profiles.append(rendered_profile)
                source_labels.append(rendered_species)
                source_ids.append(str(profile_uid))
                source_origins.append(str(metadata.get("origin_hdf5_path") or source))
                points = _estimate_points_for_loaded_profile(rendered_profile)
                if points is not None:
                    raw_estimated_total_points += int(points)
            fallback_labels_by_source.append(source_labels)
            series_id_segments_by_source.append(source_ids)
            origin_path_segments_by_source.append(source_origins)

        raw_candidate_points = raw_estimated_total_points
        estimated_total_points = raw_estimated_total_points
        if projection_mode:
            raw_candidate_points, estimated_total_points = _estimate_position_gui_point_counts(
                plot_profiles,
                resolved_projection=resolved_projection,
            )
            _log_position_projection_guard_debug(
                stage="full_context",
                resolved_projection=resolved_projection,
                raw_candidate_points=raw_candidate_points,
                final_visible_points=estimated_total_points,
            )
        _log_plot_complexity_debug(
            analysis_name="position",
            stage="full_context",
            raw_series_count=len(plot_profiles),
            raw_point_count=raw_candidate_points,
            final_series_count=len(plot_profiles),
            final_point_count=estimated_total_points,
        )
        reference_profile = None
        for _source, profiles in profiles_by_source:
            if profiles:
                reference_profile = profiles[0]
                break

        return _GuiPlotRenderContext(
            profile=plot_profiles,
            plot_source_label=sources[0] if len(sources) == 1 else "multi_source_position",
            plotter_kwargs=_resolve_position_plotter_kwargs(
                args,
                data_contract=(
                    None
                    if reference_profile is None
                    else position_profile_to_plot_data_contract(reference_profile)
                ),
            ),
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
            profile_filter_options=_build_position_plot_gui_filter_options(reference_profile),
            estimated_total_points=estimated_total_points,
        )

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
    raw_estimated_total_points = _estimate_total_points_from_loaded_profiles(plot_profiles) or 0
    raw_candidate_points = raw_estimated_total_points
    estimated_total_points = raw_estimated_total_points
    if projection_mode:
        raw_candidate_points, estimated_total_points = _estimate_position_gui_point_counts(
            plot_profiles,
            resolved_projection=resolved_projection,
        )
        _log_position_projection_guard_debug(
            stage="full_context",
            resolved_projection=resolved_projection,
            raw_candidate_points=raw_candidate_points,
            final_visible_points=estimated_total_points,
        )
    _log_plot_complexity_debug(
        analysis_name="position",
        stage="full_context",
        raw_series_count=len(plot_profiles),
        raw_point_count=raw_candidate_points,
        final_series_count=len(plot_profiles),
        final_point_count=estimated_total_points,
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_position",
        plotter_kwargs=_resolve_position_plotter_kwargs(
            args,
            data_contract=(
                None
                if not plot_profiles
                else position_profile_to_plot_data_contract(plot_profiles[0])
            ),
        ),
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
        profile_filter_options=_build_position_plot_gui_filter_options(None),
        estimated_total_points=estimated_total_points,
    )


def _build_coordination_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.coordination_contract import coordination_profile_to_plot_data_contract
    from .analysis.coordination import load_coordination_profiles

    expand_atom_descriptors = _coordination_plot_uses_atom_descriptors(args)
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
        expand_atom_descriptors=expand_atom_descriptors,
    )
    reference_profile = None
    for source in sources:
        reference_profiles = load_coordination_profiles(
            source,
            species_a=args.species_a,
            species_b=args.species_b if args.species_b is not None else args.species_a,
            axis=args.axis,
        )
        if reference_profiles:
            reference_profile = reference_profiles[0]
            break
    reference_contract = (
        None
        if reference_profile is None
        else coordination_profile_to_plot_data_contract(reference_profile)
    )
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_coordination",
        plotter_kwargs={
            **_resolve_coordination_plotter_kwargs(
                args,
                data_contract=reference_contract,
            ),
            "component": getattr(args, "component", "distance"),
            "time_axis": getattr(args, "time_axis", "ps"),
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
        profile_filter_options={
            **_build_coordination_profile_filter_options(raw_payloads_by_source),
            **(
                {}
                if reference_contract is None
                else {
                    "coordination_plot_contract": _serialize_plot_data_contract(reference_contract)
                }
            ),
        },
        estimated_total_points=_estimate_total_points_from_loaded_profiles(plot_profiles),
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
    active_profile_cache_keys_by_series_id: dict[str, Any] | None = None,
    density_grid_profile_cache: dict[str, Any] | None = None,
    density_grid_slice_cache: dict[str, list[Any]] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .analysis.density import (
        _density_payload_matches_selection,
        density_grids_to_averaged_heatmap_profile,
        load_density_grid_profiles_by_index,
        load_density_heatmap_profiles_by_index,
    )
    from .plot.contracts.density_contract import (
        default_density_heatmap_plot_data_contract,
        default_density_plot_data_contract,
    )
    from .plot.mappings.density_mapping import resolve_density_plot_mapping

    headers_by_source = _read_analysis_profile_headers_by_source(
        sources=sources,
        analysis="density",
    )
    filter_options = (
        _build_density_profile_filter_options(
            _headers_by_source_as_metadata_payloads(headers_by_source),
            axis=None,
            species=args.species,
        )
        or {}
    )
    selected_view_type = _density_selected_view_type(args, filter_options)
    resolved_density_mapping = resolve_density_plot_mapping(
        mapping=_coerce_runtime_view_mapping(getattr(args, "view_mapping", None)),
        view_type=selected_view_type,
        x_mode=getattr(args, "x_mode", None),
        quantity=getattr(args, "quantity", "mass"),
    )
    selected_view_type = resolved_density_mapping.view_type_id
    _load_axis, resolved_x_mode = _resolve_density_plot_axis_and_x_mode(
        axis=getattr(args, "axis", None),
        x_mode=resolved_density_mapping.x_mode,
    )
    if _canonical_mapping_view_id(selected_view_type) == "plot_2d_heatmap":
        grid_x_axis, grid_y_axis = _density_grid_2d_axes_from_args(args)
        grid_filters = _density_grid_filters_from_args(args)
        grid_slice_key = json.dumps(
            {
                "kind": "heatmap_2d",
                "x": grid_x_axis,
                "y": grid_y_axis,
                "filters": grid_filters,
                "x_bin_width": getattr(args, "x_bin_width", None),
                "y_bin_width": getattr(args, "y_bin_width", None),
            },
            sort_keys=True,
            default=str,
        )
        grid_descriptors: list[list[dict[str, Any]]] = []
        for source_index, (source, headers) in enumerate(headers_by_source):
            source_grid_descriptors: list[dict[str, Any]] = []
            for header in headers:
                header = dict(header)
                if not _density_payload_matches_selection(
                    header,
                    species=args.species,
                    profile_kind="grid_3d_sparse",
                ):
                    continue
                profile_index = int(header.get("profile_index", 0))
                profile_uid = _profile_uid_from_payload(
                    {"metadata": header},
                    fallback_prefix="density_grid",
                    index=profile_index,
                )
                species_label = str(header.get("species") or "UNKNOWN")
                resolved_load_source_path = Path(source).expanduser()
                resolved_source_path = resolved_load_source_path
                series_id = _density_logical_series_id(
                    source_path=str(resolved_source_path),
                    species=f"{species_label}:grid:profile:{profile_index}:{profile_uid}",
                )
                source_grid_descriptors.append(
                    {
                        "series_id": series_id,
                        "source_kind": "source",
                        "source_series_id": series_id,
                        "is_generated": False,
                        "source_index": source_index,
                        "series_index": len(source_grid_descriptors),
                        "source_name": resolved_source_path.name or str(source),
                        "source_directory": str(resolved_source_path.parent),
                        "source_path": str(resolved_source_path),
                        "load_source_path": str(resolved_load_source_path),
                        "default_label": f"{species_label} {grid_x_axis.upper()}/{grid_y_axis.upper()}",
                        "density_species": species_label,
                        "profile_kind": "grid_3d_sparse",
                        "profile_index": profile_index,
                        "profile_uid": profile_uid,
                        "density_grid_x_axis": grid_x_axis,
                        "density_grid_y_axis": grid_y_axis,
                        "density_grid_filters": grid_filters,
                        "density_grid_x_bin_width": getattr(args, "x_bin_width", None),
                        "density_grid_y_bin_width": getattr(args, "y_bin_width", None),
                        "density_grid_slice_key": grid_slice_key,
                    }
                )
            if source_grid_descriptors:
                grid_descriptors.append(source_grid_descriptors)
        grid_descriptors = _deduplicate_density_descriptor_segments_by_species(
            grid_descriptors
        )
        active_2d_species = _density_2d_single_species_from_segments(
            grid_descriptors,
            getattr(args, "density_enabled_species", None),
        )
        grid_descriptors = _filter_density_descriptor_segments_to_single_species(
            grid_descriptors,
            active_2d_species,
        )
        if grid_descriptors:
            def _load_grid_heatmap_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
                if not descriptors:
                    raise ValueError("No density 2D source series are enabled.")
                active_species = {
                    str(descriptor.get("density_species") or "").strip()
                    for descriptor in descriptors
                    if str(descriptor.get("density_species") or "").strip()
                }
                if len(active_species) > 1:
                    raise ValueError(
                        "Density 2D rendering supports one mapped species at a time. "
                        "Select one species in Data/Mapping, then use the series list "
                        "to choose which source files contribute."
                    )
                active_species_label = next(iter(active_species), None)
                first_descriptor = descriptors[0]
                slice_cache_payload = []
                for descriptor in descriptors:
                    load_source_path = str(descriptor.get("load_source_path") or "")
                    try:
                        stat = Path(load_source_path).expanduser().stat()
                        source_state = {
                            "path": str(Path(load_source_path).expanduser().resolve()),
                            "mtime_ns": int(stat.st_mtime_ns),
                            "size": int(stat.st_size),
                        }
                    except OSError:
                        source_state = {"path": load_source_path, "mtime_ns": None, "size": None}
                    slice_cache_payload.append(
                        {
                            "source": source_state,
                            "profile_index": int(descriptor.get("profile_index", 0)),
                            "profile_uid": str(descriptor.get("profile_uid") or ""),
                            "series_id": str(descriptor.get("series_id") or ""),
                            "slice_key": descriptor.get("density_grid_slice_key"),
                            "species": str(descriptor.get("density_species") or ""),
                        }
                    )
                slice_cache_key = json.dumps(slice_cache_payload, sort_keys=True, default=str)
                if density_grid_slice_cache is not None and slice_cache_key in density_grid_slice_cache:
                    LOGGER.debug("Density sparse-grid slice cache hit: key=%s.", slice_cache_key[:96])
                    return list(density_grid_slice_cache[slice_cache_key])

                LOGGER.debug("Density sparse-grid slice cache miss: key=%s.", slice_cache_key[:96])
                loaded_by_descriptor_key: dict[str, Any] = {}
                missing_by_source: dict[str, list[dict[str, Any]]] = {}
                source_order: list[str] = []
                for descriptor in descriptors:
                    load_source_path = str(descriptor.get("load_source_path") or "")
                    profile_index = int(descriptor.get("profile_index", 0))
                    try:
                        stat = Path(load_source_path).expanduser().stat()
                        grid_source_state = {
                            "path": str(Path(load_source_path).expanduser().resolve()),
                            "mtime_ns": int(stat.st_mtime_ns),
                            "size": int(stat.st_size),
                        }
                    except OSError:
                        grid_source_state = {
                            "path": load_source_path,
                            "mtime_ns": None,
                            "size": None,
                        }
                    descriptor_cache_key = json.dumps(
                        {
                            "source": grid_source_state,
                            "profile_index": profile_index,
                            "profile_uid": str(descriptor.get("profile_uid") or ""),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    if (
                        density_grid_profile_cache is not None
                        and descriptor_cache_key in density_grid_profile_cache
                    ):
                        loaded_by_descriptor_key[descriptor_cache_key] = density_grid_profile_cache[
                            descriptor_cache_key
                        ]
                        continue
                    if load_source_path not in missing_by_source:
                        missing_by_source[load_source_path] = []
                        source_order.append(load_source_path)
                    descriptor["_density_grid_profile_cache_key"] = descriptor_cache_key
                    missing_by_source[load_source_path].append(descriptor)

                for load_source_path in source_order:
                    source_descriptors = missing_by_source[load_source_path]
                    indices = [int(descriptor["profile_index"]) for descriptor in source_descriptors]
                    grid_profiles = load_density_grid_profiles_by_index(
                        load_source_path,
                        indices,
                        species=args.species,
                    )
                    if len(grid_profiles) != len(source_descriptors):
                        raise ValueError("Density sparse-grid metadata does not match loaded profiles.")
                    for descriptor, grid_profile in zip(source_descriptors, grid_profiles):
                        descriptor_cache_key = str(
                            descriptor.get("_density_grid_profile_cache_key") or ""
                        )
                        loaded_by_descriptor_key[descriptor_cache_key] = grid_profile
                        if density_grid_profile_cache is not None and descriptor_cache_key:
                            density_grid_profile_cache[descriptor_cache_key] = grid_profile

                loaded_grids: list[Any] = []
                for payload_item, descriptor in zip(slice_cache_payload, descriptors):
                    profile_index = int(descriptor.get("profile_index", 0))
                    load_source_path = str(descriptor.get("load_source_path") or "")
                    try:
                        stat = Path(load_source_path).expanduser().stat()
                        grid_source_state = {
                            "path": str(Path(load_source_path).expanduser().resolve()),
                            "mtime_ns": int(stat.st_mtime_ns),
                            "size": int(stat.st_size),
                        }
                    except OSError:
                        grid_source_state = payload_item["source"]
                    descriptor_cache_key = json.dumps(
                        {
                            "source": grid_source_state,
                            "profile_index": profile_index,
                            "profile_uid": str(descriptor.get("profile_uid") or ""),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    if descriptor_cache_key not in loaded_by_descriptor_key:
                        raise ValueError("Density sparse-grid cache is missing an enabled profile.")
                    loaded_grids.append(loaded_by_descriptor_key[descriptor_cache_key])

                combined_profiles = [
                    density_grids_to_averaged_heatmap_profile(
                        loaded_grids,
                        x_axis=str(first_descriptor.get("density_grid_x_axis") or "x"),
                        y_axis=str(first_descriptor.get("density_grid_y_axis") or "y"),
                        filters=first_descriptor.get("density_grid_filters") or {},
                        x_bin_width=first_descriptor.get("density_grid_x_bin_width"),
                        y_bin_width=first_descriptor.get("density_grid_y_bin_width"),
                        species=active_species_label,
                    )
                ]
                if density_grid_slice_cache is not None:
                    density_grid_slice_cache[slice_cache_key] = list(combined_profiles)
                return combined_profiles

            return _LazyGuiSeriesCatalog(
                sources=list(sources),
                plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
                plotter_kwargs={
                    **_resolve_density_plotter_kwargs(
                        args,
                        data_contract=default_density_heatmap_plot_data_contract(),
                        view_type=selected_view_type,
                    ),
                    "heatmap_vmin": getattr(args, "heatmap_vmin", None),
                    "heatmap_vmax": getattr(args, "heatmap_vmax", None),
                    "heatmap_cmap": getattr(args, "heatmap_cmap", None),
                    "heatmap_log_scale": getattr(args, "heatmap_log_scale", False),
                    "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
                    "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
                    "heatmap_colorbar_label_size": getattr(args, "heatmap_colorbar_label_size", None),
                    "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
                    "heatmap_colorbar_position": getattr(args, "heatmap_colorbar_position", "right"),
                    "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
                    "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
                    "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
                },
                descriptor_segments_by_source=grid_descriptors,
                default_series_labels=[
                    str(descriptor.get("default_label") or "")
                    for segment in grid_descriptors
                    for descriptor in segment
                ],
                profile_filter_options={
                    **filter_options,
                    "density_plot_contract": _serialize_plot_data_contract(
                        default_density_plot_data_contract()
                    ),
                    "density_heatmap_plot_contract": _serialize_plot_data_contract(
                        default_density_heatmap_plot_data_contract()
                    ),
                    "density_grid_2d_allowed_pairs": [
                        ["x", "y"],
                        ["y", "x"],
                        ["x", "z"],
                        ["z", "x"],
                        ["x", "distance"],
                        ["distance", "x"],
                        ["y", "z"],
                        ["z", "y"],
                        ["y", "distance"],
                        ["distance", "y"],
                    ],
                },
                load_profiles=_load_grid_heatmap_profiles,
                combined_profile_loader=True,
                _active_profiles_by_series_id=(
                    active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
                ),
                _active_profile_cache_keys_by_series_id=(
                    active_profile_cache_keys_by_series_id
                    if active_profile_cache_keys_by_series_id is not None
                    else {}
                ),
            )
        if grid_filters:
            raise ValueError(
                "Density ranges require a 3D density grid. Recompute density with "
                "--outputs 3d or --outputs all."
            )

        selected_descriptors: list[list[dict[str, Any]]] = []
        for source, headers in headers_by_source:
            matching_headers = [
                dict(header)
                for header in headers
                if _density_payload_matches_selection(
                    dict(header),
                    species=args.species,
                    plane=getattr(args, "plane", None),
                    profile_kind="heatmap_2d",
                )
            ]
            if not matching_headers:
                continue
            header = matching_headers[0]
            selected_descriptors.append(
                [
                    {
                        "series_id": _profile_uid_from_payload(
                            {"metadata": header},
                            fallback_prefix="density_heatmap",
                            index=int(header.get("profile_index", 0)),
                        ),
                        "source_kind": "source",
                        "source_series_id": _profile_uid_from_payload(
                            {"metadata": header},
                            fallback_prefix="density_heatmap",
                            index=int(header.get("profile_index", 0)),
                        ),
                        "is_generated": False,
                        "source_index": 0,
                        "series_index": 0,
                        "source_name": Path(str(header.get("origin_hdf5_path") or source)).name
                        or str(source),
                        "source_directory": str(
                            Path(str(header.get("origin_hdf5_path") or source)).expanduser().parent
                        ),
                        "source_path": str(
                            Path(str(header.get("origin_hdf5_path") or source)).expanduser()
                        ),
                        "load_source_path": str(Path(source).expanduser()),
                        "default_label": f"{str(header.get('species') or 'UNKNOWN')} {str(header.get('plane') or 'xy').upper()}",
                        "density_species": str(header.get("species") or "UNKNOWN"),
                        "profile_index": int(header.get("profile_index", 0)),
                        "profile_uid": _profile_uid_from_payload(
                            {"metadata": header},
                            fallback_prefix="density_heatmap",
                            index=int(header.get("profile_index", 0)),
                        ),
                    }
                ]
            )
            break
        selected_descriptors = _deduplicate_density_descriptor_segments_by_species(
            selected_descriptors
        )
        selected_descriptors = _filter_density_descriptor_segments_by_enabled_species(
            selected_descriptors,
            getattr(args, "density_enabled_species", None),
        )

        def _load_heatmap_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
            loaded: list[Any] = []
            for load_source_path, source_descriptors in _group_descriptors_by_load_source(
                descriptors
            ):
                indices = [int(descriptor["profile_index"]) for descriptor in source_descriptors]
                loaded.extend(
                    load_density_heatmap_profiles_by_index(
                        load_source_path,
                        indices,
                        species=args.species,
                        plane=getattr(args, "plane", None),
                    )
                )
            return loaded

        return _LazyGuiSeriesCatalog(
            sources=list(sources),
            plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
            plotter_kwargs={
                **_resolve_density_plotter_kwargs(
                    args,
                    data_contract=default_density_heatmap_plot_data_contract(),
                    view_type=selected_view_type,
                ),
                "heatmap_vmin": getattr(args, "heatmap_vmin", None),
                "heatmap_vmax": getattr(args, "heatmap_vmax", None),
                "heatmap_cmap": getattr(args, "heatmap_cmap", None),
                "heatmap_log_scale": getattr(args, "heatmap_log_scale", False),
                "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
                "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
                "heatmap_colorbar_label_size": getattr(args, "heatmap_colorbar_label_size", None),
                "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
                "heatmap_colorbar_position": getattr(args, "heatmap_colorbar_position", "right"),
                "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
                "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
                "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
            },
            descriptor_segments_by_source=selected_descriptors,
            default_series_labels=[
                str(descriptor.get("default_label") or "")
                for segment in selected_descriptors
                for descriptor in segment
            ],
            profile_filter_options={
                **filter_options,
                "density_plot_contract": _serialize_plot_data_contract(
                    default_density_plot_data_contract()
                ),
                "density_heatmap_plot_contract": _serialize_plot_data_contract(
                    default_density_heatmap_plot_data_contract()
                ),
            },
            load_profiles=_load_heatmap_profiles,
            _active_profiles_by_series_id=(
                active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
            ),
            _active_profile_cache_keys_by_series_id=(
                active_profile_cache_keys_by_series_id
                if active_profile_cache_keys_by_series_id is not None
                else {}
            ),
        )
    logical_descriptor_segments, logical_filter_options = _build_density_logical_descriptor_segments(
        sources=sources,
        metadata_by_source=headers_by_source,
        axis=None,
        species=args.species,
    )
    logical_descriptor_segments = _deduplicate_density_descriptor_segments_by_species(
        logical_descriptor_segments
    )
    filter_options = {
        **filter_options,
        **(logical_filter_options or {}),
    }
    logical_descriptor_segments = _filter_density_descriptor_segments_by_enabled_species(
        logical_descriptor_segments,
        getattr(args, "density_enabled_species", None),
    )
    descriptor_segments = _resolve_density_render_descriptor_segments(
        logical_descriptor_segments,
        axis=getattr(args, "axis", None),
        x_mode=resolved_x_mode,
        x_bin_width=getattr(args, "x_bin_width", None),
        grid_filters=_density_grid_filters_from_args(args),
    )
    density_contract = default_density_plot_data_contract()

    def _load_profiles(descriptors: list[dict[str, Any]]) -> list[Any]:
        loaded = _load_density_profiles_for_render_descriptors(
            descriptors,
            axis=None,
            species=None,
        )
        if len(loaded) != len(descriptors):
            raise ValueError("Lazy density loader returned mismatched profile count.")
        return loaded

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_density",
        plotter_kwargs=_resolve_density_plotter_kwargs(
            args,
            data_contract=density_contract,
            view_type=selected_view_type,
        ),
        descriptor_segments_by_source=descriptor_segments,
        default_series_labels=[
            str(descriptor.get("default_label") or "")
            for segment in descriptor_segments
            for descriptor in segment
        ],
        profile_filter_options={
            **(filter_options or {}),
            "density_plot_contract": _serialize_plot_data_contract(density_contract),
            "density_heatmap_plot_contract": _serialize_plot_data_contract(
                default_density_heatmap_plot_data_contract()
            ),
        },
        load_profiles=_load_profiles,
        _active_profiles_by_series_id=(
            active_profiles_by_series_id if active_profiles_by_series_id is not None else {}
        ),
        _active_profile_cache_keys_by_series_id=(
            active_profile_cache_keys_by_series_id
            if active_profile_cache_keys_by_series_id is not None
            else {}
        ),
    )


def _build_msd_gui_lazy_catalog(
    args: argparse.Namespace,
    *,
    sources: list[str],
    active_profiles_by_series_id: dict[str, Any] | None = None,
) -> _LazyGuiSeriesCatalog:
    from .plot.contracts.msd_contract import default_msd_plot_data_contract
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
        plotter_kwargs=_resolve_msd_plotter_kwargs(
            args,
            data_contract=default_msd_plot_data_contract(),
        ),
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
    from .plot.contracts.rdf_contract import default_rdf_plot_data_contract
    from .analysis.rdf import load_rdf_profiles_by_index

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
            resolved_a = str(header.get("species_a", "")).strip() or "UNKNOWN"
            resolved_b = str(header.get("species_b", "")).strip() or resolved_a
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
            profiles = load_rdf_profiles_by_index(load_source_path, indices)
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
        plotter_kwargs=_resolve_rdf_plotter_kwargs(
            args,
            data_contract=default_rdf_plot_data_contract(),
        ),
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options=None,
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
    enabled_position_species = _position_enabled_species_set(
        getattr(args, "position_enabled_species", None)
    )
    position_species_options = _build_position_species_options_from_headers(headers_by_source)
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
    raw_estimated_total_points = 0
    resolved_projection = _resolve_position_projection_estimation_settings(args)
    profile_level_projection = _position_projection_uses_profile_descriptors(
        args,
        resolved_projection=resolved_projection,
    )
    projection_estimate_indices_by_source: list[tuple[str, list[int]]] = []
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
        projection_estimate_indices: list[int] = []
        for header, (datasets, _metadata) in zip(headers, lightweight_payloads):
            source_label = _metadata_source_label(header, fallback_source=str(source_path))
            resolved_species = str(header.get("species", "")).strip() or "UNKNOWN"
            resolved_axis = str(header.get("axis", "z")).strip().lower() or "z"
            try:
                normalized_resolved_species = _normalize_position_species(resolved_species)
            except ValueError:
                normalized_resolved_species = resolved_species
            if (
                enabled_position_species is not None
                and normalized_resolved_species not in enabled_position_species
            ):
                continue
            if wanted_species is not None and wanted_species != "ALL":
                if normalized_resolved_species != wanted_species:
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
            n_frames = int(header.get("n_frames", 0) or 0)
            frame_timestep_fs = _coerce_positive_float_or_none(
                header.get("frame_timestep_fs")
            )
            atom_indices = np.asarray(datasets.get("atom_indices", []), dtype=int)
            raw_estimated_total_points += n_frames * int(atom_indices.size)
            projection_estimate_indices.append(profile_index)
            if profile_level_projection:
                source_labels.append(rendered_species)
                source_ids.append(str(profile_uid))
                source_origins.append(str(header.get("origin_hdf5_path") or source))
                source_load_paths.append(str(header.get("source_path") or source))
                source_extras.append(
                    {
                        "profile_index": profile_index,
                        "profile_uid": profile_uid,
                        "position_species": normalized_resolved_species,
                        "rendered_species": rendered_species,
                        "n_frames": n_frames,
                        "frame_timestep_fs": frame_timestep_fs,
                    }
                )
            else:
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
                            "position_species": normalized_resolved_species,
                            "atom_index": atom_token,
                            "rendered_species": rendered_species,
                            "n_frames": n_frames,
                            "frame_timestep_fs": frame_timestep_fs,
                        }
                    )
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
        load_source_path_segments_by_source.append(source_load_paths)
        extra_segments_by_source.append(source_extras)
        projection_estimate_indices_by_source.append((str(source_path), projection_estimate_indices))

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
                    if profile_level_projection:
                        loaded_by_id[str(descriptor["series_id"])] = replace(
                            parent_profile,
                            species=str(
                                descriptor.get("rendered_species") or parent_profile.species
                            ),
                        )
                    else:
                        child_profile = _extract_position_profile_atom_series(
                            parent_profile,
                            int(descriptor["atom_index"]),
                        )
                        loaded_by_id[str(descriptor["series_id"])] = replace(
                            child_profile,
                            species=str(
                                descriptor.get("rendered_species") or child_profile.species
                            ),
                        )
        return [loaded_by_id[str(descriptor["series_id"])] for descriptor in descriptors]

    raw_candidate_points = raw_estimated_total_points
    estimated_total_points = raw_estimated_total_points
    if resolved_projection.is_projection:
        raw_candidate_points = 0
        estimated_total_points = 0
        dataset_names = _position_projection_estimate_dataset_names(resolved_projection)
        for source, indices in projection_estimate_indices_by_source:
            if not indices:
                continue
            source_path = Path(source).expanduser().resolve()
            payloads = read_linak_hdf5_profiles_by_index(
                source_path,
                indices,
                expected_analysis="position",
                dataset_names=dataset_names,
            )
            for datasets, metadata in payloads:
                profile_raw_points, profile_final_points = (
                    _estimate_position_projection_point_counts_from_payload(
                        datasets,
                        metadata,
                        resolved_projection=resolved_projection,
                    )
                )
                raw_candidate_points += profile_raw_points
                estimated_total_points += profile_final_points
        _log_position_projection_guard_debug(
            stage="lazy_catalog",
            resolved_projection=resolved_projection,
            raw_candidate_points=raw_candidate_points,
            final_visible_points=estimated_total_points,
        )
    descriptor_count = sum(len(segment) for segment in descriptor_segments)
    _log_plot_complexity_debug(
        analysis_name="position",
        stage="lazy_catalog",
        raw_series_count=descriptor_count,
        raw_point_count=raw_candidate_points,
        final_series_count=descriptor_count,
        final_point_count=estimated_total_points,
    )

    def _estimate_render_points(
        active_profiles: list[Any],
        current_args: argparse.Namespace,
        active_descriptors: list[dict[str, Any]],
    ) -> int | None:
        resolved_current_projection = _resolve_position_projection_estimation_settings(current_args)
        if not resolved_current_projection.is_projection:
            return _estimate_position_line_points_from_descriptors(
                active_descriptors,
                x_bin_width=getattr(current_args, "time_section_width", None)
                if getattr(current_args, "time_section_width", None) is not None
                else getattr(current_args, "x_bin_width", None),
                time_axis=str(getattr(current_args, "time_axis", "ps") or "ps"),
            )
        raw_points, final_points = _estimate_position_gui_point_counts(
            active_profiles,
            resolved_projection=resolved_current_projection,
        )
        _log_position_projection_guard_debug(
            stage="render_context",
            resolved_projection=resolved_current_projection,
            raw_candidate_points=raw_points,
            final_visible_points=final_points,
        )
        return final_points

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_position",
        plotter_kwargs=_resolve_position_plotter_kwargs(
            args,
            data_contract=None,
        ),
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options={
            **_build_position_plot_gui_filter_options(None),
            "position_species_options": position_species_options,
        },
        load_profiles=_load_profiles,
        estimated_total_points=estimated_total_points,
        estimate_render_points=_estimate_render_points,
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
    from .plot.contracts.coordination_contract import coordination_profile_to_plot_data_contract
    from .analysis.coordination import (
        _normalize_axis as _normalize_coordination_axis,
        _normalize_species as _normalize_coordination_species,
        load_coordination_profiles_by_index,
    )
    from .storage.hdf5_utils import read_linak_hdf5_profiles_by_index

    expand_atom_descriptors = _coordination_plot_uses_atom_descriptors(args)
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
    estimated_total_points = 0
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
        if expand_atom_descriptors and matching_indices:
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
            if not expand_atom_descriptors:
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
            estimated_total_points += int(header.get("n_frames", 0) or 0) * max(
                int(atom_indices.size),
                1,
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
                    if not expand_atom_descriptors:
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

    reference_profile = None
    for source, headers in filtered_headers_by_source:
        if not headers:
            continue
        source_path = Path(source).expanduser().resolve()
        reference_profiles = load_coordination_profiles_by_index(
            source_path,
            [int(headers[0].get("profile_index", 0))],
            species_a=args.species_a,
            species_b=resolved_species_b,
            axis=args.axis,
        )
        if reference_profiles:
            reference_profile = reference_profiles[0]
            break
    reference_contract = (
        None
        if reference_profile is None
        else coordination_profile_to_plot_data_contract(reference_profile)
    )

    return _LazyGuiSeriesCatalog(
        sources=list(sources),
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_coordination",
        plotter_kwargs={
            **_resolve_coordination_plotter_kwargs(
                args,
                data_contract=reference_contract,
            ),
            "component": getattr(args, "component", "distance"),
            "time_axis": getattr(args, "time_axis", "ps"),
        },
        descriptor_segments_by_source=descriptor_segments,
        profile_filter_options={
            **_build_coordination_profile_filter_options(
                _headers_by_source_as_metadata_payloads(filtered_headers_by_source)
            ),
            **(
                {}
                if reference_contract is None
                else {
                    "coordination_plot_contract": _serialize_plot_data_contract(reference_contract)
                }
            ),
        },
        load_profiles=_load_profiles,
        estimated_total_points=estimated_total_points or None,
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
    from .plot.contracts.potential_contract import potential_profiles_to_plot_data_contract

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
        source_labels: list[str] = []
        source_ids: list[str] = []
        source_origins: list[str] = []
        for profile in plot_profiles:
            profile_source_path = Path(profile.source_path or source_path).expanduser().resolve()
            label_prefix = profile_source_path.stem or profile_source_path.name or str(
                profile_source_path
            )
            source_token = f"{profile_source_path}::"
            if str(profile.series_id).startswith(source_token):
                rendered_series_id = str(profile.series_id)
                rendered_label = profile.default_label
            else:
                rendered_series_id = f"{profile_source_path}::{profile.series_id}"
                rendered_label = f"{label_prefix}: {profile.default_label}"
            source_labels.append(rendered_label)
            source_ids.append(rendered_series_id)
            source_origins.append(str(profile_source_path))
            flattened_profiles.append(
                replace(
                    profile,
                    series_id=rendered_series_id,
                    default_label=rendered_label,
                    source_path=str(profile_source_path),
                )
            )
        fallback_labels_by_source.append(source_labels)
        series_id_segments_by_source.append(source_ids)
        origin_path_segments_by_source.append(source_origins)
    descriptors = _build_gui_series_descriptors(
        sources=[str(source_path) for source_path in resolved_sources],
        fallback_labels_by_source=fallback_labels_by_source,
        series_id_segments_by_source=series_id_segments_by_source,
        origin_path_segments_by_source=origin_path_segments_by_source,
    )
    potential_contract = potential_profiles_to_plot_data_contract(flattened_profiles)
    return _GuiPlotRenderContext(
        profile=flattened_profiles,
        plot_source_label=(
            str(resolved_sources[0]) if len(resolved_sources) == 1 else "multi_source_potential"
        ),
        plotter_kwargs=_resolve_potential_plotter_kwargs(
            args,
            data_contract=potential_contract,
        ),
        fallback_labels_by_source=fallback_labels_by_source,
        default_series_labels=_resolve_gui_default_series_labels(
            args=args,
            sources=[str(source_path) for source_path in resolved_sources],
            profile_key=_PLOT_PROFILE_POTENTIAL,
            fallback_labels_by_source=fallback_labels_by_source,
        ),
        series_descriptors=descriptors,
        profile_filter_options={
            "potential_plot_contract": _serialize_plot_data_contract(potential_contract),
            "potential_summary": {
                "x_axis_label": "Record ID",
                "total_rows": total_rows,
                "complete_rows": complete_rows,
                "incomplete_rows": incomplete_rows,
            },
        },
        estimated_total_points=_estimate_total_points_from_loaded_profiles(flattened_profiles),
    )


def _build_orientation_gui_context(
    args: argparse.Namespace,
    *,
    sources: list[str],
) -> _GuiPlotRenderContext:
    from .plot.contracts.orientation_contract import (
        default_orientation_heatmap_plot_data_contract,
        default_orientation_line_plot_data_contract,
        orientation_heatmap_profile_to_plot_data_contract,
        orientation_line_profile_to_plot_data_contract,
    )

    (
        plot_profiles,
        fallback_labels_by_source,
        series_id_segments_by_source,
        origin_path_segments_by_source,
    ) = _load_orientation_plot_profiles(sources=sources)
    reference_profile = None if not plot_profiles else plot_profiles[0]
    if reference_profile is None:
        line_contract = default_orientation_line_plot_data_contract()
        heatmap_contract = default_orientation_heatmap_plot_data_contract()
    else:
        line_contract = orientation_line_profile_to_plot_data_contract(reference_profile)
        heatmap_contract = orientation_heatmap_profile_to_plot_data_contract(reference_profile)
    orientation_axis_ranges: dict[str, list[float]] = {}
    orientation_axis_bin_widths: dict[str, float] = {}

    def _merge_orientation_axis_range(axis_id: str, edges: Any) -> None:
        try:
            values = np.asarray(edges, dtype=float)
        except (TypeError, ValueError):
            return
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return
        lower = float(finite[0])
        upper = float(finite[-1])
        if lower >= upper:
            return
        current = orientation_axis_ranges.get(axis_id)
        if current is None:
            orientation_axis_ranges[axis_id] = [lower, upper]
        else:
            current[0] = min(current[0], lower)
            current[1] = max(current[1], upper)
        widths = np.diff(finite)
        positive = widths[np.isfinite(widths) & (widths > 0.0)]
        if positive.size:
            orientation_axis_bin_widths[axis_id] = float(np.median(positive))

    for profile in plot_profiles:
        grid = getattr(profile, "sparse_grid", None)
        if grid is not None:
            _merge_orientation_axis_range("x", getattr(grid, "x_edges", None))
            _merge_orientation_axis_range("y", getattr(grid, "y_edges", None))
            _merge_orientation_axis_range("z", getattr(grid, "z_edges", None))
            _merge_orientation_axis_range("distance", getattr(grid, "distance_edges", None))
            continue
        metadata = dict(getattr(profile, "metadata", {}) or {})
        raw_cell = metadata.get("resolved_cell_angstrom") or metadata.get("pbc_cell_angstrom")
        if isinstance(raw_cell, Sequence) and not isinstance(raw_cell, (str, bytes)) and len(raw_cell) == 3:
            try:
                cell = [float(value) for value in raw_cell]
            except (TypeError, ValueError):
                cell = []
            if len(cell) == 3 and all(math.isfinite(value) and value > 0.0 for value in cell):
                for axis_id, value in zip(("x", "y", "z"), cell):
                    _merge_orientation_axis_range(axis_id, [0.0, value])
        _merge_orientation_axis_range("distance", getattr(profile, "bin_edges", None))
    resolved_mapping = _resolve_orientation_plotter_kwargs(args).get("view_mapping")
    reference_contract = None
    if reference_profile is not None and resolved_mapping is not None:
        if _canonical_mapping_view_id(getattr(resolved_mapping, "view_type_id", None)) == "plot_2d_heatmap":
            reference_contract = heatmap_contract
        else:
            reference_contract = line_contract
    return _GuiPlotRenderContext(
        profile=plot_profiles,
        plot_source_label=sources[0] if len(sources) == 1 else "multi_source_orientation",
        plotter_kwargs={
            **_resolve_orientation_plotter_kwargs(args, data_contract=reference_contract),
            "heatmap_vmin": getattr(args, "heatmap_vmin", None),
            "heatmap_vmax": getattr(args, "heatmap_vmax", None),
            "heatmap_cmap": getattr(args, "heatmap_cmap", None),
            "y_bin_width": getattr(args, "y_bin_width", None),
            "y_bin_reducer": getattr(args, "y_bin_reducer", None),
            "heatmap_normalize": getattr(args, "heatmap_normalize", False),
            "heatmap_normalization_mode": getattr(args, "heatmap_normalization_mode", None),
            "heatmap_log_scale": getattr(args, "heatmap_log_scale", False),
            "heatmap_colorbar_enabled": getattr(args, "heatmap_colorbar_enabled", True),
            "heatmap_colorbar_label": getattr(args, "heatmap_colorbar_label", None),
            "heatmap_colorbar_label_size": getattr(args, "heatmap_colorbar_label_size", None),
            "heatmap_colorbar_tick_size": getattr(args, "heatmap_colorbar_tick_size", None),
            "heatmap_colorbar_position": getattr(args, "heatmap_colorbar_position", "right"),
            "heatmap_colorbar_pad": getattr(args, "heatmap_colorbar_pad", None),
            "heatmap_colorbar_shrink": getattr(args, "heatmap_colorbar_shrink", None),
            "heatmap_colorbar_aspect": getattr(args, "heatmap_colorbar_aspect", None),
            "orientation_line_x_axis": getattr(args, "orientation_line_x_axis", None),
            "orientation_heatmap_x_axis": getattr(args, "orientation_heatmap_x_axis", None),
            "orientation_heatmap_y_axis": getattr(args, "orientation_heatmap_y_axis", None),
            "orientation_filter_x_min": getattr(args, "orientation_filter_x_min", None),
            "orientation_filter_x_max": getattr(args, "orientation_filter_x_max", None),
            "orientation_filter_y_min": getattr(args, "orientation_filter_y_min", None),
            "orientation_filter_y_max": getattr(args, "orientation_filter_y_max", None),
            "orientation_filter_z_min": getattr(args, "orientation_filter_z_min", None),
            "orientation_filter_z_max": getattr(args, "orientation_filter_z_max", None),
            "orientation_filter_distance_min": getattr(args, "orientation_filter_distance_min", None),
            "orientation_filter_distance_max": getattr(args, "orientation_filter_distance_max", None),
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
        profile_filter_options={
            "orientation_line_plot_contract": _serialize_plot_data_contract(line_contract),
            "orientation_heatmap_plot_contract": _serialize_plot_data_contract(heatmap_contract),
            "orientation_axis_ranges": orientation_axis_ranges,
            "orientation_default_axis_bin_widths_A": orientation_axis_bin_widths,
        },
        estimated_total_points=_estimate_total_points_from_loaded_profiles(plot_profiles),
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


def _filter_plotter_kwargs(
    plotter: Callable[..., Path | None], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return kwargs accepted by the target plotter signature."""
    try:
        signature = inspect.signature(plotter)
    except (TypeError, ValueError):
        return kwargs

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    filtered = {name: value for name, value in kwargs.items() if name in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        LOGGER.debug(
            "Dropping unsupported plot kwargs for %s: %s",
            getattr(plotter, "__name__", repr(plotter)),
            ", ".join(dropped),
        )
    return filtered


def _render_profile_plot(
    *,
    args: argparse.Namespace,
    source: str,
    analysis_name: str,
    profile: Any,
    plotter: Callable[..., Path | None],
    plotter_kwargs: dict[str, Any] | None = None,
    series_descriptors: list[dict[str, Any]] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    keep_figure_open: bool = False,
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
    ordered_render_descriptors = list(render_series_descriptors or ordered_descriptors)
    source_ordered_descriptors = [
        d
        for d in ordered_descriptors
        if str(d.get("source_kind") or "source").strip().lower() != "group"
    ]
    group_ordered_descriptors = [
        d
        for d in ordered_descriptors
        if str(d.get("source_kind") or "source").strip().lower() == "group"
    ]
    if (
        isinstance(profile, list)
        and source_ordered_descriptors
        and len(profile) == len(source_ordered_descriptors)
    ):
        natural_ids = [
            str(descriptor.get("series_id") or f"series:{index}")
            for index, descriptor in enumerate(source_ordered_descriptors)
        ]
        resolved_order = _resolve_series_id_order(natural_ids, getattr(args, "series_order", None))
        if resolved_order != natural_ids:
            index_by_id = {series_id: index for index, series_id in enumerate(natural_ids)}
            indices = [index_by_id[series_id] for series_id in resolved_order]
            ordered_profile = [profile[index] for index in indices]
            source_ordered_descriptors = [source_ordered_descriptors[index] for index in indices]
            ordered_descriptors = source_ordered_descriptors + group_ordered_descriptors
            render_descriptor_by_id = {
                str(descriptor.get("series_id") or ""): dict(descriptor)
                for descriptor in ordered_render_descriptors
                if str(descriptor.get("series_id") or "").strip()
            }
            ordered_source_render_descriptors = [
                render_descriptor_by_id.get(
                    str(descriptor.get("series_id") or ""),
                    dict(descriptor),
                )
                for descriptor in source_ordered_descriptors
            ]
            ordered_source_ids = {
                str(descriptor.get("series_id") or "")
                for descriptor in ordered_source_render_descriptors
            }
            ordered_render_descriptors = ordered_source_render_descriptors + [
                dict(descriptor)
                for descriptor in ordered_render_descriptors
                if str(descriptor.get("series_id") or "") not in ordered_source_ids
            ]

            def _reorder_source_list(value: Any) -> Any:
                if isinstance(value, list) and len(value) == len(indices):
                    return [value[index] for index in indices]
                return value

            for attr_name in (
                "series_labels",
                "line_colors",
                "series_enabled",
                "series_show_in_legend",
                "series_line_widths",
                "series_markers",
                "series_fit_configs",
                "series_error_configs",
                "series_normalization_modes",
                "series_normalization_values",
                "series_normalization_x_refs",
                "series_line_kwargs",
            ):
                if hasattr(args, attr_name):
                    setattr(args, attr_name, _reorder_source_list(getattr(args, attr_name)))

    # Final transition point: generic style options from argparse are merged
    # here with analysis-specific plotter kwargs and series identity metadata
    # before dispatch to the selected analysis plotter.
    shared_kwargs = {
        "series_ids": [
            str(d.get("series_id") or f"series:{i}")
            for i, d in enumerate(source_ordered_descriptors)
        ]
        if source_ordered_descriptors
        else None,
        "title": args.title,
        "x_label": args.x_label,
        "y_label": args.y_label,
        "x_scale": args.x_scale,
        "x_axis_scale": getattr(args, "x_axis_scale", None),
        "x_axis_offset": getattr(args, "x_axis_offset", None),
        "y_scale": args.y_scale,
        "x_lim": _resolve_x_lim(args),
        "y_lim": _resolve_y_lim(args),
        "x_ticks": args.x_ticks,
        "y_ticks": args.y_ticks,
        "x_tick_rotation": args.x_tick_rotation,
        "y_tick_rotation": args.y_tick_rotation,
        "x_label_font_size": getattr(args, "x_label_font_size", None),
        "y_label_font_size": getattr(args, "y_label_font_size", None),
        "x_tick_font_size": getattr(args, "x_tick_font_size", None),
        "y_tick_font_size": getattr(args, "y_tick_font_size", None),
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
        "annotations": getattr(args, "annotations", None),
        "integration_config": getattr(args, "integration_config", None),
        "x_bin_width": getattr(args, "x_bin_width", None)
        if getattr(args, "x_bin_width", None) is not None
        else getattr(args, "time_section_width", None),
        "x_bin_reducer": getattr(args, "x_bin_reducer", None),
        "min_bin_points": getattr(args, "min_bin_points", None),
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
            else _default_series_family_colors(
                ordered_render_descriptors or ordered_descriptors,
                len(source_ordered_descriptors),
                target_descriptors=source_ordered_descriptors,
            )
            if source_ordered_descriptors
            else None
        ),
        "show_blocking": not bool(getattr(args, "gui", False)),
        "keep_figure_open": bool(keep_figure_open),
        "capture_state": captured_state,
        "suppress_output_log": bool(getattr(args, "_suppress_output_log", False)),
    }
    shared_kwargs["series_fit_configs"] = getattr(args, "series_fit_configs", None)
    shared_kwargs["series_error_configs"] = getattr(args, "series_error_configs", None)
    shared_kwargs["render_series_descriptors"] = ordered_render_descriptors or None
    shared_kwargs["series_overrides_by_id"] = getattr(args, "series_overrides", None)

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
            error_configs = call_kwargs.pop("series_error_configs", None)
            if isinstance(error_configs, list) and error_configs:
                first_error_config = error_configs[0]
                if isinstance(first_error_config, dict):
                    call_kwargs["error_config"] = dict(first_error_config)
            render_descriptors = call_kwargs.get("render_series_descriptors")
            override_map = call_kwargs.get("series_overrides_by_id")
            if (
                isinstance(render_descriptors, list)
                and render_descriptors
                and isinstance(override_map, dict)
            ):
                first_series_id = str(render_descriptors[0].get("series_id") or "").strip()
                first_override = override_map.get(first_series_id)
                if isinstance(first_override, dict) and isinstance(
                    first_override.get("cumulative"), dict
                ):
                    call_kwargs["cumulative_config"] = dict(first_override["cumulative"])
            per_series_line_kwargs = call_kwargs.pop("series_line_kwargs", None)
            if isinstance(per_series_line_kwargs, list) and per_series_line_kwargs:
                first_kwargs = per_series_line_kwargs[0]
                if isinstance(first_kwargs, dict):
                    merged_line_kwargs: dict[str, Any] = {}
                    if isinstance(call_kwargs.get("line_kwargs"), dict):
                        merged_line_kwargs.update(dict(call_kwargs["line_kwargs"]))
                    merged_line_kwargs.update(first_kwargs)
                    call_kwargs["line_kwargs"] = merged_line_kwargs
        merged_kwargs = dict(extra_kwargs)
        merged_kwargs.update(call_kwargs)
        return plotter(
            ordered_profile,
            output=output,
            show=show,
            preferred_backend=args.backend,
            style=style,
            **_filter_plotter_kwargs(plotter, merged_kwargs),
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
    resolved_view_mapping = _resolve_plot_settings_view_mapping(args, keys=keys)
    if resolved_view_mapping is not None:
        from .plot.profile_persistence import serialize_plot_view_mapping

        candidate["view_mapping"] = serialize_plot_view_mapping(resolved_view_mapping)
    if "x_lim" in candidate:
        candidate["x_lim"] = _resolve_x_lim(args)
    if "y_lim" in candidate:
        candidate["y_lim"] = _resolve_y_lim(args)
    return candidate


def _resolve_plot_settings_view_mapping(
    args: argparse.Namespace,
    *,
    keys: tuple[str, ...],
) -> Any | None:
    mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
    if mapping is not None:
        return mapping
    if keys is _PLOT_SETTINGS_DENSITY_KEYS:
        return _resolve_density_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_MSD_KEYS:
        return _resolve_msd_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_RDF_KEYS:
        return _resolve_rdf_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_POSITION_KEYS:
        return _resolve_position_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_COORDINATION_KEYS:
        return _resolve_coordination_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_POTENTIAL_KEYS:
        return _resolve_potential_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_ORIENTATION_KEYS:
        return _resolve_orientation_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_TEMPERATURE_KEYS:
        return _resolve_temperature_plotter_kwargs(args).get("view_mapping")
    if keys is _PLOT_SETTINGS_TABLE_KEYS:
        from .plot.profile_persistence import (
            build_plot_profile_payload,
            deserialize_plot_view_mapping,
        )

        settings = _collect_plot_settings_from_args(args, keys=keys)
        payload = build_plot_profile_payload("plot:table", settings)
        raw_mapping = payload.get("view_mapping")
        if isinstance(raw_mapping, dict):
            return deserialize_plot_view_mapping(raw_mapping)
    return None


def _derive_gui_sync_modes(settings: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    x_lim = settings.get("x_lim")
    y_lim = settings.get("y_lim")
    inferred = {
        "title": "off"
        if settings.get("title_visible") is False
        else "manual"
        if settings.get("title") is not None
        else "auto",
        "x_label": "manual" if settings.get("x_label") is not None else "auto",
        "y_label": "manual" if settings.get("y_label") is not None else "auto",
        "x_lim": "manual"
        if isinstance(x_lim, (list, tuple)) and any(value is not None for value in x_lim[:2])
        else "auto",
        "y_lim": "manual"
        if isinstance(y_lim, (list, tuple)) and any(value is not None for value in y_lim[:2])
        else "auto",
        "x_ticks": "manual" if settings.get("x_ticks") is not None else "auto",
        "y_ticks": "manual" if settings.get("y_ticks") is not None else "auto",
        "x_label_pad": "manual" if settings.get("x_label_pad") is not None else "auto",
        "y_label_pad": "manual" if settings.get("y_label_pad") is not None else "auto",
    }
    for key, mode in inferred.items():
        if mode != "auto":
            resolved[key] = mode
    return resolved


def _apply_gui_settings_to_args(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    always_forward = {
        "title",
        "title_visible",
        "x_label",
        "y_label",
        "x_lim",
        "y_lim",
        "x_ticks",
        "y_ticks",
        "x_scale",
        "y_scale",
        "x_axis_scale",
        "x_axis_offset",
        "x_tick_rotation",
        "y_tick_rotation",
        "x_label_font_size",
        "y_label_font_size",
        "x_tick_font_size",
        "y_tick_font_size",
        "x_label_pad",
        "y_label_pad",
        "series_overrides",
        "series_order",
        "annotations",
        "title_pad",
        "heatmap_vmin",
        "heatmap_vmax",
        "heatmap_cmap",
        "heatmap_normalize",
        "heatmap_normalization_mode",
        "heatmap_log_scale",
        "heatmap_colorbar_enabled",
        "heatmap_colorbar_label",
        "heatmap_colorbar_label_size",
        "heatmap_colorbar_tick_size",
        "heatmap_colorbar_position",
        "heatmap_colorbar_pad",
        "heatmap_colorbar_shrink",
        "heatmap_colorbar_aspect",
        "view_mapping",
        "density_enabled_species",
        "density_active_view_type",
        "density_view_states",
        "position_enabled_species",
        "position_active_view_type",
        "position_view_states",
        "orientation_active_view_type",
        "orientation_view_states",
        "density_2d_x_axis",
        "density_2d_y_axis",
        "density_filter_x_min",
        "density_filter_x_max",
        "density_filter_y_min",
        "density_filter_y_max",
        "density_filter_z_min",
        "density_filter_z_max",
        "density_filter_distance_min",
        "density_filter_distance_max",
        "orientation_line_x_axis",
        "orientation_heatmap_x_axis",
        "orientation_heatmap_y_axis",
        "orientation_filter_x_min",
        "orientation_filter_x_max",
        "orientation_filter_y_min",
        "orientation_filter_y_max",
        "orientation_filter_z_min",
        "orientation_filter_z_max",
        "orientation_filter_distance_min",
        "orientation_filter_distance_max",
        "projection_x",
        "projection_y",
        "projection_value",
        "projection_render_mode",
        "projection_filter_min",
        "projection_filter_max",
        "x_bin_width",
        "x_bin_reducer",
        "y_bin_width",
        "y_bin_reducer",
        "xy_z_distance_max",
        "figure_kwargs",
        "axes_kwargs",
        "line_kwargs",
        "series_line_kwargs",
        "grid_kwargs",
        "legend_kwargs",
        "tick_params_kwargs",
        "tight_layout_kwargs",
        "savefig_kwargs",
        "series_show_in_legend",
        "series_alpha",
        "plot_data_format",
        "plot_data_delimiter",
        "plot_data_include_metadata",
        "plot_data_enabled_only",
        "_gui_sync_modes",
    }
    for key, value in settings.items():
        if key not in always_forward and not hasattr(args, key):
            continue
        setattr(args, key, deepcopy(value))
    if ("x_mode" in settings or "quantity" in settings) and hasattr(args, "view_mapping"):
        from .plot.mappings.density_mapping import density_plot_options_to_view_mapping

        current_mapping = _coerce_runtime_view_mapping(getattr(args, "view_mapping", None))
        view_type = (
            str(getattr(current_mapping, "view_type_id", "") or "line_1d").strip().lower()
            if current_mapping is not None
            else "line_1d"
        )
        args.view_mapping = density_plot_options_to_view_mapping(
            view_type=view_type,
            x_mode=getattr(args, "x_mode", "distance"),
            quantity=getattr(args, "quantity", "mass"),
        )
    if isinstance(settings.get("series_overrides"), dict):
        _clear_gui_positional_series_args(args)
    if "x_bin_width" in settings and hasattr(args, "time_section_width"):
        args.time_section_width = settings.get("x_bin_width")
    if "x_min" not in settings and hasattr(args, "x_min"):
        args.x_min = None
    if "x_max" not in settings and hasattr(args, "x_max"):
        args.x_max = None
    if "y_min" not in settings and hasattr(args, "y_min"):
        args.y_min = None
    if "y_max" not in settings and hasattr(args, "y_max"):
        args.y_max = None
    if (
        "x_lim" not in settings
        and settings.get("x_min") is None
        and settings.get("x_max") is None
        and hasattr(args, "x_lim")
    ):
        args.x_lim = None
    if (
        "y_lim" not in settings
        and settings.get("y_min") is None
        and settings.get("y_max") is None
        and hasattr(args, "y_lim")
    ):
        args.y_lim = None


def _clear_gui_positional_series_args(args: argparse.Namespace) -> None:
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
        "series_fit_configs",
        "series_error_configs",
    ):
        setattr(args, key, None)


def _open_plot_settings_gui(
    *,
    title: str,
    initial_settings: dict[str, Any],
    on_preview: Callable[[dict[str, Any]], dict[str, Any] | None],
    on_save: Callable[[str, dict[str, Any]], str],
    on_preview_figure: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    on_save_figure: Callable[[dict[str, Any], str], str | tuple[str, dict[str, Any]]] | None = None,
    on_save_data: Callable[[dict[str, Any], str], str | tuple[str, dict[str, Any]]] | None = None,
    on_import_hdf5: Callable[[str, str | None], dict[str, Any]] | None = None,
    on_list_import_hdf5_profiles: Callable[[str], dict[str, Any]] | None = None,
    analysis_name: str | None = None,
    on_resolve_series_defaults: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    initial_profile_name: str | None = None,
    available_profile_names: list[str] | None = None,
    default_profile_settings: dict[str, Any] | None = None,
    on_load_profile: Callable[[str], dict[str, Any]] | None = None,
    on_rename_profile: Callable[[str, str], str] | None = None,
    on_duplicate_profile: Callable[[str, str], str] | None = None,
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
        on_preview_figure=on_preview_figure,
        on_save_figure=on_save_figure,
        on_save_data=on_save_data,
        on_import_hdf5=on_import_hdf5,
        on_list_import_hdf5_profiles=on_list_import_hdf5_profiles,
        analysis_name=analysis_name,
        on_resolve_series_defaults=on_resolve_series_defaults,
        initial_profile_name=initial_profile_name,
        available_profile_names=available_profile_names,
        default_profile_settings=default_profile_settings,
        on_load_profile=on_load_profile,
        on_rename_profile=on_rename_profile,
        on_duplicate_profile=on_duplicate_profile,
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


def _plot_data_export_delimiter(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".dat":
        return " "
    if suffix in {".tsv", ".txt"}:
        return "\t"
    return ","


def _plot_data_export_delimiter_from_setting(value: Any, path: str | Path) -> str:
    token = str(value or "auto").strip().lower()
    if token in {"", "auto"}:
        return _plot_data_export_delimiter(path)
    if token == "comma":
        return ","
    if token == "tab":
        return "\t"
    if token == "space":
        return " "
    if len(token) == 1:
        return token
    return _plot_data_export_delimiter(path)


def _format_plot_data_export_value(value: Any) -> str:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return format(float(value), ".15g")
    return str(value)


def _write_plotted_xy_data_export(
    render_state: Mapping[str, Any],
    output_path: str | Path,
    *,
    delimiter: Any = None,
    include_metadata: bool = False,
) -> Path:
    series_payload = render_state.get("plotted_xy_series")
    if not isinstance(series_payload, list) or not series_payload:
        raise ValueError("No line xy data is available for this preview.")

    normalized_series: list[dict[str, Any]] = []
    for raw_series in series_payload:
        if not isinstance(raw_series, Mapping):
            continue
        x_values = raw_series.get("x")
        y_values = raw_series.get("y")
        if not isinstance(x_values, Sequence) or isinstance(x_values, (str, bytes)):
            continue
        if not isinstance(y_values, Sequence) or isinstance(y_values, (str, bytes)):
            continue
        if len(x_values) != len(y_values) or len(x_values) == 0:
            continue
        normalized_series.append(
            {
                "series_id": str(raw_series.get("series_id") or ""),
                "series_label": str(raw_series.get("series_label") or ""),
                "series_kind": str(raw_series.get("series_kind") or "source"),
                "x": list(x_values),
                "y": list(y_values),
            }
        )

    if not normalized_series:
        raise ValueError("No line xy data is available for this preview.")

    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = _plot_data_export_delimiter_from_setting(delimiter, target_path)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        if include_metadata:
            handle.write("# LiNaK plotted data export\n")
            handle.write(f"# series_count={len(normalized_series)}\n")
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        if len(normalized_series) == 1:
            writer.writerow(["x", "y"])
            series = normalized_series[0]
            for x_value, y_value in zip(series["x"], series["y"]):
                writer.writerow(
                    [
                        _format_plot_data_export_value(x_value),
                        _format_plot_data_export_value(y_value),
                    ]
                )
            return target_path

        writer.writerow(["series_id", "series_label", "series_kind", "point_index", "x", "y"])
        for series in normalized_series:
            for point_index, (x_value, y_value) in enumerate(zip(series["x"], series["y"]), start=1):
                writer.writerow(
                    [
                        series["series_id"],
                        series["series_label"],
                        series["series_kind"],
                        str(point_index),
                        _format_plot_data_export_value(x_value),
                        _format_plot_data_export_value(y_value),
                    ]
                )
    return target_path


def _descriptor_is_generated_layer(descriptor: dict[str, Any]) -> bool:
    return str(descriptor.get("source_kind") or "source").strip().lower() == "group" or bool(
        descriptor.get("is_generated", False)
    )


def _merge_gui_series_descriptors(
    current_descriptors: list[dict[str, Any]],
    saved_descriptors: Any,
) -> list[dict[str, Any]]:
    current = [dict(descriptor) for descriptor in current_descriptors]
    if not isinstance(saved_descriptors, list):
        return current
    current_ids = {str(item.get("series_id") or "").strip() for item in current}
    merged = list(current)
    generated_sources: list[dict[str, Any]] = []
    generated_groups: list[dict[str, Any]] = []
    for raw_descriptor in saved_descriptors:
        if not isinstance(raw_descriptor, dict):
            continue
        descriptor = dict(raw_descriptor)
        if not _descriptor_is_generated_layer(descriptor):
            continue
        source_kind = str(descriptor.get("source_kind") or "source").strip().lower()
        descriptor["source_kind"] = "group" if source_kind == "group" else "source"
        descriptor["is_generated"] = True
        if descriptor["source_kind"] == "group":
            generated_groups.append(descriptor)
            continue
        source_series_id = str(descriptor.get("source_series_id") or "").strip()
        if source_series_id in current_ids:
            descriptor["source_series_id"] = source_series_id
            generated_sources.append(descriptor)
    generated_ids = set(current_ids)
    for descriptor in generated_sources:
        series_id = str(descriptor.get("series_id") or "").strip()
        if series_id and series_id not in generated_ids:
            merged.append(descriptor)
            generated_ids.add(series_id)
    for descriptor in generated_groups:
        series_id = str(descriptor.get("series_id") or "").strip()
        member_ids = [
            str(member_id).strip()
            for member_id in descriptor.get("member_series_ids", [])
            if str(member_id).strip() in generated_ids
        ]
        if series_id and series_id not in generated_ids:
            descriptor["member_series_ids"] = member_ids
            merged.append(descriptor)
            generated_ids.add(series_id)
    return merged


def _gui_series_descriptors_from_settings(
    gui_settings: dict[str, Any],
    fallback_descriptors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    descriptors = gui_settings.get("series_descriptors")
    if isinstance(descriptors, list):
        resolved = [dict(descriptor) for descriptor in descriptors if isinstance(descriptor, dict)]
    else:
        resolved = [dict(descriptor) for descriptor in fallback_descriptors]
    return _filter_density_descriptors_by_enabled_species(
        resolved,
        gui_settings.get("density_enabled_species"),
    )


def _density_enabled_species_set(value: Any) -> set[str] | None:
    if not isinstance(value, (list, tuple, set)):
        return None
    enabled = {str(item).strip() for item in value if str(item).strip()}
    return enabled


def _density_descriptor_species(descriptor: Mapping[str, Any]) -> str:
    return str(descriptor.get("density_species") or descriptor.get("default_label") or "").strip()


def _filter_density_descriptors_by_enabled_species(
    descriptors: list[dict[str, Any]],
    enabled_species_value: Any,
) -> list[dict[str, Any]]:
    enabled_species = _density_enabled_species_set(enabled_species_value)
    if enabled_species is None:
        return [dict(descriptor) for descriptor in descriptors]
    descriptor_by_id = {
        str(descriptor.get("series_id") or "").strip(): dict(descriptor)
        for descriptor in descriptors
        if str(descriptor.get("series_id") or "").strip()
    }
    filtered: list[dict[str, Any]] = []
    for descriptor in descriptors:
        source_kind = str(descriptor.get("source_kind") or "source").strip().lower()
        if source_kind == "group":
            member_ids = [
                str(member_id).strip()
                for member_id in descriptor.get("member_series_ids", [])
                if str(member_id).strip()
            ]
            kept_members = [
                member_id
                for member_id in member_ids
                if _density_descriptor_species(descriptor_by_id.get(member_id, {}))
                in enabled_species
            ]
            if not kept_members:
                continue
            updated = dict(descriptor)
            updated["member_series_ids"] = kept_members
            filtered.append(updated)
            continue
        species = _density_descriptor_species(descriptor)
        if species and species not in enabled_species:
            continue
        filtered.append(dict(descriptor))
    return filtered


def _filter_density_descriptor_segments_by_enabled_species(
    descriptor_segments_by_source: list[list[dict[str, Any]]],
    enabled_species_value: Any,
) -> list[list[dict[str, Any]]]:
    enabled_species = _density_enabled_species_set(enabled_species_value)
    if enabled_species is None:
        return [[dict(descriptor) for descriptor in segment] for segment in descriptor_segments_by_source]
    return [
        [
            dict(descriptor)
            for descriptor in segment
            if not _density_descriptor_species(descriptor)
            or _density_descriptor_species(descriptor) in enabled_species
        ]
        for segment in descriptor_segments_by_source
    ]


def _density_2d_single_species_from_segments(
    descriptor_segments_by_source: list[list[dict[str, Any]]],
    enabled_species_value: Any,
) -> str | None:
    available_species: list[str] = []
    available_set: set[str] = set()
    for segment in descriptor_segments_by_source:
        for descriptor in segment:
            species = _density_descriptor_species(descriptor)
            if species and species not in available_set:
                available_species.append(species)
                available_set.add(species)
    if not available_species:
        return None
    if isinstance(enabled_species_value, (list, tuple)):
        for item in enabled_species_value:
            species = str(item).strip()
            if species in available_set:
                return species
    enabled_species = _density_enabled_species_set(enabled_species_value)
    if enabled_species:
        for species in available_species:
            if species in enabled_species:
                return species
    return available_species[0]


def _filter_density_descriptor_segments_to_single_species(
    descriptor_segments_by_source: list[list[dict[str, Any]]],
    species: str | None,
) -> list[list[dict[str, Any]]]:
    if not species:
        return []
    return [
        [
            dict(descriptor)
            for descriptor in segment
            if _density_descriptor_species(descriptor) == species
        ]
        for segment in descriptor_segments_by_source
    ]


def _required_source_ids_for_gui_render(gui_settings: dict[str, Any]) -> set[str]:
    descriptors = _gui_series_descriptors_from_settings(gui_settings, [])
    if not descriptors:
        return set()
    overrides = _coerce_series_override_map(gui_settings.get("series_overrides"))
    descriptor_by_id = {
        str(descriptor.get("series_id") or "").strip(): descriptor
        for descriptor in descriptors
        if str(descriptor.get("series_id") or "").strip()
    }

    def _descriptor_enabled(series_id: str) -> bool:
        entry = overrides.get(series_id)
        return not (isinstance(entry, dict) and entry.get("enabled") is False)

    required: set[str] = set()
    for descriptor in descriptors:
        series_id = str(descriptor.get("series_id") or "").strip()
        if not series_id or not _descriptor_enabled(series_id):
            continue
        source_kind = str(descriptor.get("source_kind") or "source").strip().lower()
        if source_kind != "group":
            source_id = str(descriptor.get("source_series_id") or series_id).strip()
            if source_id:
                required.add(source_id)
            continue
        for member_id in descriptor.get("member_series_ids", []):
            member_key = str(member_id).strip()
            member_descriptor = descriptor_by_id.get(member_key)
            if member_descriptor is None:
                continue
            if not _descriptor_enabled(member_key):
                continue
            source_id = str(
                member_descriptor.get("source_series_id")
                or member_descriptor.get("series_id")
                or ""
            ).strip()
            if source_id:
                required.add(source_id)
    return required


def _force_source_ids_enabled_for_gui_loading(
    args: argparse.Namespace,
    required_source_ids: set[str],
) -> None:
    if not required_source_ids:
        return
    overrides = _coerce_series_override_map(getattr(args, "series_overrides", None))
    for source_id in required_source_ids:
        entry = dict(overrides.get(source_id, {}))
        entry["enabled"] = True
        overrides[source_id] = entry
    args.series_overrides = overrides
    args.series_enabled = None


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
        duplicate_named_plot_profile,
        rename_named_plot_profile,
        read_active_plot_profile_name,
        read_plot_profile_names,
        set_active_plot_profile,
        supports_named_plot_profiles,
    )

    allow_named_profiles = supports_named_plot_profiles(source_path)
    if build_full_context is None:
        build_full_context = build_context
    # The GUI currently persists one broad settings payload that mixes source
    # filters, view-mapping choices, layer state, and pure figure style.
    initial_settings = _collect_plot_settings_for_persistence(args, keys=setting_keys)
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
    initial_saved_profile = _read_plot_profile_for_apply(
        source_path,
        profile_key=profile_key,
        keys=setting_keys,
        profile_name=initial_profile_name,
    )
    initial_settings = _merge_gui_only_plot_settings(initial_settings, initial_saved_profile)
    initial_settings["series_descriptors"] = _merge_gui_series_descriptors(
        initial_context.series_descriptors,
        initial_saved_profile.get("series_descriptors")
        if isinstance(initial_saved_profile, dict)
        else None,
    )
    initial_settings = _materialize_gui_series_overrides(initial_settings)
    if analysis_name == "position":
        initial_settings = _apply_position_gui_auto_display_reduction(
            initial_settings,
            args=args,
            initial_context=initial_context,
        )
    initial_settings = _strip_redundant_series_lists_for_gui(initial_settings)

    context_cache: dict[str, _GuiPlotRenderContext] = {}

    def _source_state_signature() -> dict[str, Any]:
        try:
            stat = source_path.stat()
        except OSError:
            return {"path": str(source_path), "mtime_ns": None, "size": None}
        return {
            "path": str(source_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }

    def _gui_context_data_signature(
        settings: dict[str, Any],
        *,
        kind: str,
        required_source_ids: set[str],
    ) -> str:
        data_keys = (
            "species",
            "axis",
            "component",
            "time_axis",
            "map_color",
            "view_mapping",
            "density_enabled_species",
            "density_2d_x_axis",
            "density_2d_y_axis",
            "density_filter_x_min",
            "density_filter_x_max",
            "density_filter_y_min",
            "density_filter_y_max",
            "density_filter_z_min",
            "density_filter_z_max",
            "density_filter_distance_min",
            "density_filter_distance_max",
            "orientation_line_x_axis",
            "orientation_heatmap_x_axis",
            "orientation_heatmap_y_axis",
            "orientation_filter_x_min",
            "orientation_filter_x_max",
            "orientation_filter_y_min",
            "orientation_filter_y_max",
            "orientation_filter_z_min",
            "orientation_filter_z_max",
            "orientation_filter_distance_min",
            "orientation_filter_distance_max",
            "projection_x",
            "projection_y",
            "projection_value",
            "projection_render_mode",
            "projection_filter_min",
            "projection_filter_max",
            "xy_z_distance_max",
            "x_bin_width",
            "y_bin_width",
            "x_bin_reducer",
            "y_bin_reducer",
            "min_bin_points",
        )
        payload = {
            "analysis": analysis_name,
            "kind": kind,
            "source": _source_state_signature(),
            "required_source_ids": sorted(required_source_ids),
            "settings": {
                key: deepcopy(settings.get(key))
                for key in data_keys
                if key in settings
            },
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _cached_gui_context(
        *,
        kind: str,
        builder: Callable[[argparse.Namespace], _GuiPlotRenderContext],
        current_args: argparse.Namespace,
        settings: dict[str, Any],
        required_source_ids: set[str],
    ) -> _GuiPlotRenderContext:
        cache_key = _gui_context_data_signature(
            settings,
            kind=kind,
            required_source_ids=required_source_ids,
        )
        cached = context_cache.get(cache_key)
        if cached is not None:
            LOGGER.debug(
                "GUI %s context cache hit: analysis=%s key=%s.",
                kind,
                analysis_name,
                cache_key[:96],
            )
            return cached
        LOGGER.debug(
            "GUI %s context cache miss: analysis=%s key=%s.",
            kind,
            analysis_name,
            cache_key[:96],
        )
        context = builder(current_args)
        context_cache[cache_key] = context
        return context

    effective_guard_args = deepcopy(args)
    _apply_gui_settings_to_args(effective_guard_args, initial_settings)
    effective_context = (
        build_context(effective_guard_args)
        if analysis_name == "position"
        else build_full_context(effective_guard_args)
    )
    effective_series_descriptors = _merge_gui_series_descriptors(
        effective_context.series_descriptors,
        initial_saved_profile.get("series_descriptors")
        if isinstance(initial_saved_profile, dict)
        else None,
    )
    _log_plot_complexity_debug(
        analysis_name=analysis_name,
        stage="gui_launch_guard",
        raw_series_count=len(initial_context.series_descriptors),
        raw_point_count=initial_context.estimated_total_points,
        final_series_count=len(effective_series_descriptors),
        final_point_count=effective_context.estimated_total_points,
    )
    if not getattr(args, "force_gui", False):
        _raise_or_warn_for_plot_complexity(
            _estimate_plot_complexity(
                analysis_name=analysis_name,
                series_descriptors=effective_series_descriptors,
                profile=effective_context.profile,
                estimated_total_points=effective_context.estimated_total_points,
            ),
            interactive_gui=analysis_name != "position",
        )
    else:
        LOGGER.debug(
            "Bypassing %s GUI complexity guard because --force-gui was provided.", analysis_name
        )
    gui_render_sources = [
        f"gui-series-source:{index}"
        for index in range(len(initial_context.fallback_labels_by_source))
    ]

    def _render_gui_preview_settings(
        gui_settings: dict[str, Any],
        *,
        show: bool,
        output: str | None,
        empty_error_message: str,
        keep_figure_open: bool = False,
    ) -> tuple[Path | None, dict[str, Any]]:
        # Preview/export do not use a separate render path. They replay GUI
        # settings back into argparse-like state, rebuild context, and then
        # call the same renderer bridge used by non-GUI plotting.
        preview_args = deepcopy(args)
        _apply_gui_settings_to_args(preview_args, gui_settings)
        preview_args.show = show
        preview_args.output = output
        preview_args._suppress_output_log = output is None or _is_gui_preview_output_path(output)
        load_args = deepcopy(preview_args)
        required_source_ids = _required_source_ids_for_gui_render(gui_settings)
        _force_source_ids_enabled_for_gui_loading(
            load_args,
            required_source_ids,
        )
        context = _cached_gui_context(
            kind="render",
            builder=build_context,
            current_args=load_args,
            settings=gui_settings,
            required_source_ids=required_source_ids,
        )
        if context.series_count <= 0:
            raise ValueError(empty_error_message)
        if output is None:
            _log_plot_complexity_debug(
                analysis_name=analysis_name,
                stage="gui_preview_guard",
                raw_series_count=len(initial_context.series_descriptors),
                raw_point_count=initial_context.estimated_total_points,
                final_series_count=len(context.series_descriptors),
                final_point_count=context.estimated_total_points,
            )
            if not getattr(preview_args, "force_gui", False):
                _raise_or_warn_for_plot_complexity(
                    _estimate_plot_complexity(
                        analysis_name=analysis_name,
                        series_descriptors=context.series_descriptors,
                        profile=context.profile,
                        estimated_total_points=context.estimated_total_points,
                    ),
                    interactive_gui=analysis_name != "position",
                )
            else:
                LOGGER.debug(
                    "Bypassing %s GUI preview complexity guard because --force-gui was provided.",
                    analysis_name,
                )
        _apply_effective_series_settings(
            args=preview_args,
            sources=gui_render_sources,
            profile_key=profile_key,
            fallback_labels_by_source=context.fallback_labels_by_source,
            series_descriptors=context.series_descriptors,
            allow_saved_multi_source_merge=False,
            materialize_default_colors=True,
        )
        if isinstance(gui_settings.get("series_overrides"), dict):
            _clear_gui_positional_series_args(preview_args)
        saved_path, render_state = _render_profile_plot(
            args=preview_args,
            source=context.plot_source_label,
            analysis_name=analysis_name,
            profile=context.profile,
            plotter=plotter,
            plotter_kwargs=context.plotter_kwargs,
            series_descriptors=context.series_descriptors,
            render_series_descriptors=context.series_descriptors,
            keep_figure_open=keep_figure_open,
        )
        render_state = dict(render_state or {})
        if analysis_name == "density":
            descriptor_context = _cached_gui_context(
                kind="full",
                builder=build_full_context,
                current_args=preview_args,
                settings=gui_settings,
                required_source_ids=required_source_ids,
            )
            descriptor_source = (
                descriptor_context
                if descriptor_context.series_descriptors
                else context
            )
            render_state["series_descriptors"] = deepcopy(
                descriptor_source.series_descriptors
            )
            render_state["series_labels"] = list(descriptor_source.default_series_labels)
            render_state["_profile_filter_options"] = deepcopy(
                descriptor_source.profile_filter_options
            )
        return saved_path, render_state

    initial_render_state: dict[str, Any] = {}
    initial_required_source_ids = _required_source_ids_for_gui_render(initial_settings)
    if initial_required_source_ids:
        _initial_saved_path, initial_render_state = _render_gui_preview_settings(
            initial_settings,
            show=False,
            output=None,
            empty_error_message="No series are enabled. Turn on at least one series to preview.",
        )
    if initial_render_state:
        initial_settings = _merge_preview_defaults_into_gui_settings(
            initial_settings,
            initial_render_state,
        )
        # Render-state values are matplotlib defaults, not explicit user choices.
        # Explicitly mark any synced field without a stored mode as "auto" so
        # _derive_synced_field_modes in the GUI does not infer "manual" from them.
        _render_sync_modes = dict(initial_settings.get("_gui_sync_modes") or {})
        for _k in (
            "title",
            "x_label",
            "y_label",
            "x_lim",
            "y_lim",
            "x_ticks",
            "y_ticks",
            "x_label_pad",
            "y_label_pad",
        ):
            _render_sync_modes.setdefault(_k, "auto")
        initial_settings["_gui_sync_modes"] = _render_sync_modes

    def _preview(gui_settings: dict[str, Any]) -> dict[str, Any]:
        _saved_path, render_state = _render_gui_preview_settings(
            gui_settings,
            show=True,
            output=None,
            empty_error_message="No series are enabled. Turn on at least one series to preview.",
        )
        return render_state

    def _preview_figure(gui_settings: dict[str, Any]) -> dict[str, Any]:
        _saved_path, render_state = _render_gui_preview_settings(
            gui_settings,
            show=False,
            output=None,
            empty_error_message="No series are enabled. Turn on at least one series to preview.",
            keep_figure_open=True,
        )
        figure = render_state.get("figure") if isinstance(render_state, dict) else None
        if figure is None:
            raise RuntimeError("Preview renderer did not return a Matplotlib figure.")
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
                "series_fit_configs",
                "series_error_configs",
            ):
                candidate.pop(key, None)
        if "_gui_sync_modes" in gui_settings:
            candidate["_gui_sync_modes"] = deepcopy(gui_settings["_gui_sync_modes"])
        _write_flat_plot_profile(
            source_path,
            profile_key=profile_key,
            settings=candidate,
            profile_name=profile_name,
        )
        return f"Saved '{profile_name}' to {source_path.name}."

    def _save_figure(gui_settings: dict[str, Any], output_path: str) -> tuple[str, dict[str, Any]]:
        saved_path, render_state = _render_gui_preview_settings(
            gui_settings,
            show=False,
            output=output_path,
            empty_error_message=(
                "No series are enabled. Turn on at least one series before exporting."
            ),
        )
        if saved_path is None:
            raise ValueError("No output was generated for the requested figure path.")
        return f"Saved figure to '{saved_path}'.", render_state

    def _save_data(gui_settings: dict[str, Any], output_path: str) -> tuple[str, dict[str, Any]]:
        _saved_path, render_state = _render_gui_preview_settings(
            gui_settings,
            show=False,
            output=None,
            empty_error_message=(
                "No series are enabled. Turn on at least one series before exporting data."
            ),
        )
        saved_path = _write_plotted_xy_data_export(
            render_state,
            output_path,
            delimiter=gui_settings.get("plot_data_delimiter"),
            include_metadata=bool(gui_settings.get("plot_data_include_metadata", False)),
        )
        return f"Saved data to '{saved_path}'.", render_state

    def _list_import_hdf5_profiles(source_hdf5_path: str) -> dict[str, Any]:
        imported_path = Path(source_hdf5_path).expanduser().resolve()
        available_names = read_plot_profile_names(imported_path, profile_key)
        if not available_names:
            if _read_flat_plot_profile(imported_path, profile_key=profile_key) is None:
                raise ValueError(
                    f"No plot settings profile '{profile_key}' found in '{imported_path}'."
                )
            available_names = ["Default"]
        active_name = read_active_plot_profile_name(imported_path, profile_key)
        if active_name is None and "Default" in available_names:
            active_name = "Default"
        return {
            "available_names": available_names,
            "active_name": active_name,
        }

    def _import_hdf5(source_hdf5_path: str, profile_name: str | None) -> dict[str, Any]:
        imported_path = Path(source_hdf5_path).expanduser().resolve()
        imported = _read_plot_profile_for_apply(
            imported_path,
            profile_key=profile_key,
            keys=setting_keys,
            profile_name=profile_name,
        )
        if imported is None:
            raise ValueError(
                f"No plot settings profile '{profile_name or profile_key}' found in '{imported_path}'."
            )
        LOGGER.info(
            "Loaded plot settings template from '%s' (%s).",
            imported_path,
            profile_key,
        )
        imported = _materialize_gui_series_overrides(imported)
        return _strip_redundant_series_lists_for_gui(imported)

    def _load_profile(profile_name: str) -> dict[str, Any]:
        loaded = _read_plot_profile_for_apply(
            source_path,
            profile_key=profile_key,
            keys=setting_keys,
            profile_name=profile_name,
        )
        if loaded is None:
            raise ValueError(
                f"No saved profile '{profile_name}' found in '{source_path.name}' ({profile_key})."
            )
        load_args = deepcopy(args)
        for key, value in loaded.items():
            setattr(load_args, key, deepcopy(value))
        context = build_full_context(load_args)
        merged = _merge_gui_only_plot_settings(loaded, loaded)
        merged["series_descriptors"] = _merge_gui_series_descriptors(
            context.series_descriptors,
            loaded.get("series_descriptors"),
        )
        merged["_profile_filter_options"] = deepcopy(context.profile_filter_options)
        merged = _materialize_gui_series_overrides(merged)
        return _strip_redundant_series_lists_for_gui(merged)

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

    def _rename_profile(current_name: str, new_name: str) -> str:
        _active_profile = rename_named_plot_profile(
            source_path,
            profile_key,
            current_name,
            new_name,
        )
        return f"Renamed profile '{current_name}' to '{new_name}' in '{source_path.name}'."

    def _duplicate_profile(current_name: str, new_name: str) -> str:
        duplicate_named_plot_profile(
            source_path,
            profile_key,
            current_name,
            new_name,
        )
        return f"Duplicated profile '{current_name}' as '{new_name}' in '{source_path.name}'."

    def _set_active_profile(profile_name: str) -> str:
        set_active_plot_profile(source_path, profile_key, profile_name)
        return f"Selected profile '{profile_name}' in '{source_path.name}'."

    def _resolve_series_defaults(gui_settings: dict[str, Any]) -> dict[str, Any]:
        resolved_args = deepcopy(args)
        _apply_gui_settings_to_args(resolved_args, gui_settings)
        context = build_full_context(resolved_args)
        merged_descriptors = _merge_gui_series_descriptors(
            context.series_descriptors,
            gui_settings.get("series_descriptors"),
        )
        return {
            "series_count": len(merged_descriptors),
            "series_labels": list(context.default_series_labels),
            "series_descriptors": merged_descriptors,
            "_profile_filter_options": deepcopy(context.profile_filter_options),
        }

    _open_plot_settings_gui(
        title=gui_title,
        initial_settings=initial_settings,
        on_preview=_preview,
        on_preview_figure=_preview_figure,
        on_save=_save,
        on_save_figure=_save_figure,
        on_save_data=_save_data,
        on_import_hdf5=_import_hdf5,
        on_list_import_hdf5_profiles=_list_import_hdf5_profiles,
        analysis_name=analysis_name,
        on_resolve_series_defaults=_resolve_series_defaults,
        initial_profile_name=initial_profile_name,
        available_profile_names=available_profile_names,
        default_profile_settings=default_settings,
        on_load_profile=_load_profile,
        on_rename_profile=_rename_profile,
        on_duplicate_profile=_duplicate_profile,
        on_delete_profile=_delete_profile,
        on_set_active_profile=_set_active_profile,
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
                "  1) Pack a simulation directory into one .out.h5 container",
                "     linak apply pack /path/to/simulation_dir --output run.out.h5",
                "  2) Compute analysis HDF5 from trajectory or .out.h5 data",
                "     linak compute density /path/to/traj.xyz",
                "  3) Plot from HDF5 only",
                "     linak plot /path/to/traj.density.h5",
                "",
                "Fast HDF5 plotting shorthand",
                (
                    "  linak plot /path/to/data.h5    "
                    "# auto-detects density/msd/rdf/position/coordination/potential from HDF5 metadata, "
                    f"or falls back to: linak {_TABULAR_COMMAND} plot ..."
                ),
                "",
                "Command groups",
                "  compute   trajectory/.out.h5 -> HDF5",
                "  plot      LiNaK analysis HDF5 -> figure",
                (
                    f"  {_TABULAR_COMMAND:<8} inspect/transform/plot tabular HDF5 "
                    f"(aliases: {', '.join(_TABULAR_COMMAND_ALIASES)})"
                ),
                "  apply     trajectory transformations and directory packing",
                "  project   experimental workspace UI (WIP; not the recommended path)",
                "",
                "Need details?",
                "  linak <command> --help",
                "  linak compute --help",
                "  linak plot --help",
                f"  linak {_TABULAR_COMMAND} --help",
                "  linak apply --help",
                "  linak project --help",
            ]
        )
    )
    return 0


def _handle_project(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).expanduser().resolve()
    created = not project_dir.exists()
    if created:
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created LiNaK project directory: {project_dir}")
    from .gui import launch_project_workspace

    print(f"Opening LiNaK project workspace: {project_dir}")
    launch_project_workspace(str(project_dir))
    return 0


def _handle_plot_overview(_args: argparse.Namespace) -> int:
    print(
        "\n".join(
            [
                "LiNaK Plot Usage (HDF5-only)",
                "============================",
                "Plot accepts LiNaK density/MSD/RDF/position/coordination/potential/temperature HDF5 inputs and auto-detects the analysis.",
                "",
                "Examples",
                "  linak compute density /path/to/traj.xyz",
                "  linak plot /path/to/traj.density.h5",
                "  linak plot -f run1.density.h5 run2.density.h5 --no-show --output density.png",
                "  linak plot /path/to/traj.msd.h5 --no-show --output msd.png",
                "  linak plot /path/to/traj.rdf.h5 --species-a O --species-b H",
                "  linak plot /path/to/traj.position.h5 --view-type 1d-line --y-quantity distance",
                "  linak plot /path/to/traj.coordination.h5 --view-type 1d-line --x-quantity distance",
                "  linak plot /path/to/potentials.h5",
                "  linak plot /path/to/run.temperature.h5",
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
                "Compute commands read trajectory files or .out.h5 containers and write HDF5 outputs.",
                "",
                "Examples",
                "  linak compute density /path/to/traj.xyz --species O --axis z",
                "  linak compute msd /path/to/traj.xyz --species O",
                "  linak compute position /path/to/traj.xyz --species O",
                "  linak compute rdf /path/to/traj.xyz --species-a O --species-b H",
                "  linak compute rdf /path/to/traj.xyz --atoms-a 0 100 200..210 --atoms-b 5 6 7",
                "  linak compute coordination /path/to/traj.xyz --species-a O --species-b H --cutoff-from-rdf",
                "  linak compute temperature /path/to/run-vel-1.xyz --input /path/to/input.inp",
                "  linak compute potential -f /path/to/*.cube",
                "",
                "Need command options?",
                "  linak compute density --help",
                "  linak compute msd --help",
                "  linak compute position --help",
                "  linak compute rdf --help",
                "  linak compute coordination --help",
                "  linak compute temperature --help",
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
                "  linak apply convert /path/to/traj.xyz",
                "  linak apply pack /path/to/simulation_dir --output run.out.h5",
                "  linak apply pbc /path/to/traj.xyz --cell 10 10 10",
                "  linak apply compress /path/to/output.out",
                "",
                "Need command options?",
                "  linak apply convert --help",
                "  linak apply pack --help",
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
                    f"  linak {_TABULAR_COMMAND} combine -f run1.density.h5 run2.density.h5 "
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

    project_parser = commands.add_parser(
        "project",
        help="Open a LiNaK project workspace.",
        description=(
            "Open a structured LiNaK project workspace. The project directory stores "
            "generated outputs and the workspace manifest; imported input files are "
            "referenced in place and are not copied."
        ),
    )
    project_parser.add_argument(
        "project_dir",
        help="Project directory where LiNaK outputs and workspace metadata are stored.",
    )
    project_parser.set_defaults(handler=_handle_project)

    plot_parser = commands.add_parser(
        "plot",
        help="Generate plots from precomputed LiNaK HDF5 data.",
        description=_plot_parser_description(),
        epilog=_PLOT_PARSER_EPILOG,
    )
    plot_parser.set_defaults(handler=_handle_plot)
    _configure_plot_parser(plot_parser)
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
            "(positional form)"
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
            "Density selector: element symbol (e.g. O), element:<symbol>, species:<raw_label>, "
            "group selector elements/molecules/all, or O/H molecule selector "
            "mol:H, mol:O, mol:OH, mol:H2O, mol:H3O "
            "(aliases OH, HO, H2O, H3O and legacy mol:HO accepted). H and O are elements; "
            "use mol:H and mol:O for free molecular H/O. Default: all."
        ),
    )
    _add_atom_alias_option(compute_density)
    compute_density.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Surface/distance reference axis (default: z)",
    )
    compute_density.add_argument(
        "--bin-width",
        type=_positive_float,
        default=0.05,
        help="Histogram bin width in Angstrom (default: 0.05)",
    )
    compute_density.add_argument(
        "--oh-cutoff",
        type=_positive_float,
        default=1.27,
        help="O-H cutoff in Angstrom for O/H molecule classification (default: 1.27).",
    )
    compute_density.add_argument(
        "--min-molecule-frames",
        type=int,
        default=5,
        help=(
            "Minimum number of frames a non-water O/H molecule type must appear in before "
            "group selectors create a density profile (default: 5). Explicit molecule selectors "
            "are always honored."
        ),
    )
    compute_density.add_argument(
        "--outputs",
        default="all",
        help=(
            "Density outputs to compute: '1d' writes 1D distance/X/Y/Z profiles, "
            "'3d' writes sparse grid data for GUI slicing/filtering, and 'all' writes both "
            "(default: all)."
        ),
    )
    compute_density.add_argument(
        "--grid-bin-width",
        type=_positive_float,
        default=None,
        help=(
            "Sparse 3D density grid bin width in Angstrom (default: same as --bin-width). "
            "Increase this to reduce 3D grid file size."
        ),
    )
    compute_density.add_argument(
        "--grid-max-nonzero-bins",
        type=int,
        default=20_000_000,
        help=(
            "Maximum nonzero cells allowed per sparse density grid profile before compute stops "
            "with guidance to increase --grid-bin-width or request fewer species (default: 20000000)."
        ),
    )
    compute_density.add_argument(
        "--heatmap-planes",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
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
    _add_spatial_filter_cli_args(compute_density)
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
            "(positional form)"
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
        help=(
            "Species for MSD: element symbol, element:<symbol>, species:<raw_label>, or all "
            "(default: all)."
        ),
    )
    _add_atom_alias_option(compute_msd)
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

    compute_temperature = compute_commands.add_parser(
        "temperature",
        help="Compute temperature profiles from CP2K .temp/.tregion or velocity XYZ.",
    )
    compute_temperature.add_argument(
        "source",
        nargs="?",
        help="Path to .temp, .tregion, or *-vel-*.xyz temperature source.",
    )
    compute_temperature.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Temperature source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    compute_temperature.add_argument(
        "--input",
        help="Optional CP2K input.inp path used to resolve elements and thermal regions.",
    )
    _add_atom_alias_option(compute_temperature)
    compute_temperature.add_argument(
        "--group-by",
        choices=["auto", "elements", "regions", "both"],
        default="auto",
        help=(
            "Velocity grouping mode (default: auto; both when regions are known, "
            "otherwise elements). Ignored for .temp/.tregion tables."
        ),
    )
    compute_temperature.add_argument(
        "--velocity-unit",
        choices=["auto", "atomic", "angstrom/fs"],
        default="auto",
        help="Velocity XYZ unit (default: auto, interpreted as CP2K atomic velocity units).",
    )
    compute_temperature.add_argument(
        "--remove-com",
        action="store_true",
        help="Remove center-of-mass velocity from each velocity-derived selection.",
    )
    compute_temperature.add_argument(
        "-o",
        "--output",
        "--save-data",
        dest="output",
        help="HDF5 output path (default: auto-generated .h5)",
    )
    _add_dry_run_option(compute_temperature)
    compute_temperature.set_defaults(
        handler=_handle_compute_temperature,
        compute_command="temperature",
    )

    compute_position = compute_commands.add_parser(
        "position",
        help="Compute atom-resolved positions and save HDF5.",
    )
    compute_position.add_argument(
        "trajectory",
        nargs="?",
        help=(
            "Path to trajectory file (ASE-supported; .dump supported) or LAMMPS input .lmp "
            "(positional form)"
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
            "Position selector: element symbol (e.g. O), element:<symbol>, species:<raw_label>, "
            "group selector elements/molecules/all, or O/H molecule selector "
            "mol:H, mol:O, mol:OH, mol:H2O, mol:H3O "
            "(aliases OH, HO, H2O, H3O and legacy mol:HO accepted). H and O are elements; "
            "use mol:H and mol:O for free molecular H/O. If omitted, LiNaK warns and writes "
            "one output per element and active O/H molecule."
        ),
    )
    _add_atom_alias_option(compute_position)
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
    compute_position.add_argument(
        "--oh-cutoff",
        type=_positive_float,
        default=1.27,
        help="O-H cutoff in Angstrom for O/H molecule classification (default: 1.27).",
    )
    compute_position.add_argument(
        "--min-molecule-frames",
        type=_positive_int,
        default=5,
        help=(
            "Minimum number of frames a non-water O/H molecule type must appear in before "
            "group selectors create a position profile (default: 5). Explicit molecule selectors "
            "are always honored."
        ),
    )
    compute_position.add_argument(
        "--oh-topology-stride",
        type=_positive_int,
        default=100,
        help=(
            "Frame stride for validating cached O/H molecule topology before switching to "
            "per-frame detection after a detected change (default: 100)."
        ),
    )
    _add_cell_resolution_options(compute_position)
    _add_spatial_filter_cli_args(compute_position)
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
            "(positional form)"
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
    rdf_selection_group_a = compute_rdf.add_mutually_exclusive_group()
    rdf_selection_group_a.add_argument(
        "--species-a",
        default="all",
        help=(
            "First RDF selector by species: element symbol, element:<symbol>, "
            "or species:<raw_label> (default: all when no explicit selectors are provided)."
        ),
    )
    rdf_selection_group_a.add_argument(
        "--atoms-a",
        nargs="+",
        metavar="INDEX",
        help=(
            "First RDF selector by 0-based atom indices. Accepts integers, inclusive ranges like "
            "200..210, comma-separated tokens, and optional quoted brace groups like '{200,210}'."
        ),
    )
    rdf_selection_group_b = compute_rdf.add_mutually_exclusive_group()
    rdf_selection_group_b.add_argument(
        "--species-b",
        default=None,
        help=(
            "Second RDF selector by species: element symbol, element:<symbol>, "
            "or species:<raw_label> "
            "(default: same as selector A in single-pair mode; when used alone, "
            "write all RDF pairs involving that species)"
        ),
    )
    _add_atom_alias_option(compute_rdf)
    rdf_selection_group_b.add_argument(
        "--atoms-b",
        nargs="+",
        metavar="INDEX",
        help=(
            "Second RDF selector by 0-based atom indices. Accepts integers, inclusive ranges like "
            "200..210, comma-separated tokens, and optional quoted brace groups like '{200,210}'."
        ),
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
    _add_spatial_filter_cli_args(compute_rdf)
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
            "(positional form)"
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
        help=(
            "Center species for coordination analysis: element symbol, element:<symbol>, "
            "or species:<raw_label> (required unless --species-b is provided)."
        ),
    )
    compute_coordination.add_argument(
        "--species-b",
        help=(
            "Neighbor species for coordination analysis: element symbol, element:<symbol>, "
            "or species:<raw_label> (required unless --species-a is provided)."
        ),
    )
    _add_atom_alias_option(compute_coordination)
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
            "Recompute an average RDF from all frames and determine the coordination cutoff "
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
    _add_spatial_filter_cli_args(compute_coordination)
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
        help="Hartree cube file path(s).",
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
            "(positional form)"
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
    _add_atom_alias_option(compute_orientation)
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
        default=1.27,
        help="O-H cutoff in Angstrom for water-molecule detection (default: 1.27).",
    )
    _add_cell_resolution_options(compute_orientation)
    _add_spatial_filter_cli_args(compute_orientation)
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

    apply_convert = apply_commands.add_parser(
        "convert",
        help="Convert one supported file into another supported format.",
        description=(
            "Convert one supported trajectory- or cube-family file into another supported "
            "format. Without --target-file-type, LiNaK converts to its preferred HDF5 "
            "working format for that file family."
        ),
    )
    apply_convert.add_argument("trajectory", nargs="?", help="Input trajectory path")
    apply_convert.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help=(
            "Trajectory source path(s). Use -f/--files even for one file; "
            "this command accepts exactly one source."
        ),
    )
    apply_convert.add_argument(
        "-o",
        "--output",
        help="Output path (default: family-specific target path next to the input)",
    )
    apply_convert.add_argument(
        "--input",
        help=(
            "Optional simulation input file used to embed cell/timestep/fixed-atom metadata "
            "into the converted trajectory HDF5."
        ),
    )
    _add_atom_alias_option(apply_convert)
    apply_convert.add_argument(
        "--select",
        help=(
            "Compact partial-trajectory selector, for example: first:1000f, last:5ps, "
            "first:50%%, first:500step, or range:1000f:5000f"
        ),
    )
    _add_spatial_filter_cli_args(apply_convert)
    apply_convert.add_argument(
        "--target-file-type",
        dest="target_file_type",
        help="Target file type, for example: traj.h5, xyz, cube.h5, cube",
    )
    apply_convert.add_argument(
        "--format",
        dest="target_file_type",
        help=argparse.SUPPRESS,
    )
    _add_dry_run_option(apply_convert)
    apply_convert.set_defaults(handler=_handle_apply_convert)

    apply_combine = apply_commands.add_parser(
        "combine",
        help="Combine multiple compatible inputs into one output file.",
        description=(
            "Combine multiple supported inputs while preserving input order. "
            "For trajectory-like inputs, LiNaK writes one combined `.traj.h5` by default. "
            "Use --no-convert to keep a raw combined trajectory output such as `.xyz`."
        ),
    )
    apply_combine.add_argument("trajectory", nargs="*", help="Input file path(s)")
    apply_combine.add_argument(
        "-f",
        "--files",
        nargs="+",
        metavar="PATH",
        help="Input path(s). Use -f/--files when passing multiple files.",
    )
    apply_combine.add_argument(
        "-o",
        "--output",
        help="Output path (default: <cwd>/<first-input>_combined.<family-default>)",
    )
    apply_combine.add_argument(
        "--input",
        help=(
            "Optional simulation input file used to embed cell/timestep/fixed-atom metadata "
            "into the combined trajectory HDF5."
        ),
    )
    _add_atom_alias_option(apply_combine)
    apply_combine.add_argument(
        "--cell",
        nargs=3,
        type=_positive_float,
        metavar=("A", "B", "C"),
        help="Explicit orthorhombic cell lengths in Angstrom for the combined trajectory.",
    )
    apply_combine.add_argument(
        "--no-convert",
        action="store_true",
        help="Disable default HDF5 conversion and write a raw combined trajectory when supported.",
    )
    _add_dry_run_option(apply_combine)
    apply_combine.set_defaults(handler=_handle_apply_combine)

    apply_pbc = apply_commands.add_parser(
        "pbc",
        help="Apply periodic boundary conditions by wrapping positions into a cell.",
        description=(
            "Apply orthorhombic PBC to a trajectory and wrap atom positions. "
            "Cell dimensions are resolved in this order: --cell, --input, "
            "or automatic .inp/.lmp simulation-input discovery in the output directory."
        ),
    )
    apply_pbc.add_argument("trajectory", nargs="?", help="Input trajectory path")
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
    _add_atom_alias_option(apply_pbc)
    _add_dry_run_option(apply_pbc)
    apply_pbc.set_defaults(handler=_handle_apply_pbc)

    from .storage.compress import DROP_SECTION_CHOICES

    apply_pack = apply_commands.add_parser(
        "pack",
        help="Pack a simulation output directory into one LiNaK .out.h5 container.",
        description=(
            "Recursively scan a simulation output directory and write a single LiNaK "
            ".out.h5 container with trajectory, cube, CP2K singlepoint tables, system "
            "metadata, and provenance where available."
        ),
    )
    apply_pack.add_argument(
        "source_dir",
        metavar="SIM_DIR",
        help="Simulation output directory to scan and pack.",
    )
    apply_pack.add_argument(
        "-o",
        "--output",
        help="Output .out.h5 path (default: <simulation_dir>.out.h5 next to the source).",
    )
    apply_pack.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the requested output path instead of auto-versioning collisions.",
    )
    apply_pack.add_argument(
        "--include",
        nargs="+",
        metavar="GLOB",
        help="Only include files matching these relative-path or filename glob patterns.",
    )
    apply_pack.add_argument(
        "--exclude",
        nargs="+",
        metavar="GLOB",
        help="Skip files matching these relative-path or filename glob patterns.",
    )
    apply_pack.add_argument(
        "--drop",
        nargs="+",
        choices=list(DROP_SECTION_CHOICES),
        metavar="SECTION",
        help=(
            "Optional CP2K singlepoint sections to skip. Choices: "
            + ", ".join(DROP_SECTION_CHOICES)
            + "."
        ),
    )
    _add_dry_run_option(apply_pack)
    apply_pack.set_defaults(handler=_handle_apply_pack)

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
        help="Input CP2K output file path.",
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
            "Combine multiple density/MSD/RDF/position/coordination/temperature LiNaK HDF5 files into one combined HDF5 file "
            "that can be plotted directly with `linak plot /path/to/combined.h5`."
        ),
    )
    csv_combine.add_argument(
        "source",
        nargs="*",
        metavar="SOURCE",
        help="Input HDF5 file path(s); use -f/--files for multiple",
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
        help="Target HDF5 file",
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
        read_plot_profile_stores,
        set_active_plot_profile,
    )

    selected_name = getattr(args, "name", None)
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
            _read_flat_plot_profile(
                source_path,
                profile_key=profile_key,
                profile_name=selected_name,
            )
            or {}
        )
        for assignment in args.set or []:
            key, value = _parse_plot_setting_assignment(assignment)
            _set_nested_setting(current, key, value)
        for dotted in args.unset or []:
            _delete_nested_setting(current, dotted)
        _write_flat_plot_profile(
            source_path,
            profile_key=profile_key,
            settings=current,
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
    selected_profile = _read_flat_plot_profile(
        source_path,
        profile_key=profile_key,
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
    if detected_analysis not in {"density", "msd", "rdf", "position", "coordination", "temperature"}:
        raise ValueError(
            "HDF5 combine currently supports LiNaK density/MSD/RDF/position/coordination/temperature analysis files only."
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
                "font_size": int(style.base_font_size),
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
            font_size=None,
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
        "temperature",
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
        _PLOT_PROFILE_TEMPERATURE,
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


def _resolve_uniform_plot_analysis_from_sources(sources: list[str]) -> str | None:
    if not sources:
        return None
    detected = [_detect_plot_analysis_from_hdf5_source(source) for source in sources]
    if any(name in {None, "table"} for name in detected):
        return None
    first = detected[0]
    if any(name != first for name in detected[1:]):
        return None
    return first


def _extract_plot_help_sources(argv: list[str]) -> list[str]:
    probe = argparse.ArgumentParser(add_help=False)
    _add_plot_source_options(
        probe,
        help_text="LiNaK analysis HDF5 input (use `linak hdf5 plot` for generic tables)",
    )
    try:
        args, _unknown = probe.parse_known_args(argv)
    except SystemExit:
        return []

    positional_sources = [
        source
        for source in _normalize_source_values(getattr(args, "source", None))
        if _is_hdf5_source(source)
    ]
    option_sources = [
        source
        for source in _normalize_source_values(getattr(args, "files", None))
        if _is_hdf5_source(source)
    ]
    if option_sources and positional_sources:
        return [*option_sources, *positional_sources]
    return option_sources or positional_sources


def _maybe_handle_analysis_specific_plot_help(argv: list[str]) -> int | None:
    root_probe = argparse.ArgumentParser(add_help=False)
    root_probe.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    root_probe.add_argument("--log-file")
    root_probe.add_argument("command", nargs="?")
    try:
        args, remaining = root_probe.parse_known_args(argv)
    except SystemExit:
        return None

    if getattr(args, "command", None) != "plot":
        return None
    if "-h" not in remaining and "--help" not in remaining:
        return None

    sources = _extract_plot_help_sources(remaining)
    if not sources:
        return None
    detected_analysis = _resolve_uniform_plot_analysis_from_sources(sources)
    build_plot_parser(analysis=detected_analysis).print_help()
    return 0


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
        try:
            stat = source_path.stat()
            header_cache_key = (
                str(source_path),
                str(analysis),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except OSError:
            header_cache_key = (str(source_path), str(analysis), None, None)
        cached_headers = _ANALYSIS_PROFILE_HEADER_CACHE.get(header_cache_key)
        if cached_headers is None:
            headers = read_linak_hdf5_profile_headers(source_path, expected_analysis=analysis)
            _ANALYSIS_PROFILE_HEADER_CACHE[header_cache_key] = [
                dict(header) for header in headers
            ]
            if len(_ANALYSIS_PROFILE_HEADER_CACHE) > 128:
                oldest_key = next(iter(_ANALYSIS_PROFILE_HEADER_CACHE))
                _ANALYSIS_PROFILE_HEADER_CACHE.pop(oldest_key, None)
        else:
            headers = [dict(header) for header in cached_headers]
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
    _validate_no_non_analysis_hdf5_sources(sources, command_name=command_name)
    return sources


def _handle_plot(args: argparse.Namespace) -> int:
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
    if detected_analysis == "temperature":
        args.plot_command = "temperature"
        return _handle_plot_temperature(args)

    raise ValueError(
        "Could not detect a LiNaK density/MSD/RDF/position/coordination/potential/orientation/temperature analysis from the provided HDF5 input. "
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
    resolved_axis, resolved_x_mode = _resolve_density_plot_axis_and_x_mode(
        axis=getattr(args, "axis", None),
        x_mode=getattr(args, "x_mode", None),
    )
    args.axis = resolved_axis
    args.x_mode = resolved_x_mode
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
            f"species={args.species}, axis={args.axis}",
            _density_mapping_summary_for_dry_run(args),
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
        active_profile_cache_keys_by_series_id: dict[str, Any] = {}
        density_grid_profile_cache: dict[str, Any] = {}
        density_grid_slice_cache: dict[str, list[Any]] = {}

        def _build_catalog(current_args: argparse.Namespace) -> _LazyGuiSeriesCatalog:
            catalog = _build_density_gui_lazy_catalog(
                current_args,
                sources=gui_sources,
                active_profiles_by_series_id=active_profiles_by_series_id,
                active_profile_cache_keys_by_series_id=active_profile_cache_keys_by_series_id,
                density_grid_profile_cache=density_grid_profile_cache,
                density_grid_slice_cache=density_grid_slice_cache,
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
            build_full_context=lambda current_args: _build_density_gui_logical_context(
                current_args,
                sources=gui_sources,
            ),
        )
        LOGGER.info("Density GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.density import plot_density_profiles

    render_context = _build_density_gui_lazy_catalog(args, sources=sources).build_render_context(args)
    _warn_for_non_gui_plot_complexity(analysis_name="density", render_context=render_context)

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
            _msd_mapping_summary_for_dry_run(args),
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
    _warn_for_non_gui_plot_complexity(analysis_name="msd", render_context=render_context)

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


def _handle_plot_temperature(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting temperature plotting.")
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
            profile_key=_PLOT_PROFILE_TEMPERATURE,
            keys=_PLOT_SETTINGS_TEMPERATURE_KEYS,
            profile_name=getattr(args, "settings_profile", None),
        )
    use_gui = _resolve_gui_mode(args)
    if len(sources) > 1:
        LOGGER.info("Processing %d temperature HDF5 input file(s).", len(sources))

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
            f"time_axis={getattr(args, 'time_axis', 'ps')}",
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot temperature", plan)
        LOGGER.info("Temperature plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    if use_gui:
        from .analysis.temperature import plot_temperature_profiles

        gui_sources = list(sources)
        gui_settings_path = settings_source_path
        if len(sources) > 1:
            gui_settings_path = _combine_analysis_hdf5_sources(
                sources=sources,
                analysis="temperature",
                output=None,
            )
            gui_sources = [str(gui_settings_path)]
            LOGGER.info(
                "Created combined temperature HDF5 for GUI controls: '%s'.",
                gui_settings_path,
            )
        assert gui_settings_path is not None
        initial_context = _build_temperature_gui_context(args, sources=gui_sources)
        _apply_effective_series_settings(
            args=args,
            sources=gui_sources,
            profile_key=_PLOT_PROFILE_TEMPERATURE,
            fallback_labels_by_source=initial_context.fallback_labels_by_source,
            series_descriptors=initial_context.series_descriptors,
            allow_saved_multi_source_merge=not (use_gui and len(sources) > 1),
            materialize_default_colors=False,
        )
        _launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=gui_settings_path,
            profile_key=_PLOT_PROFILE_TEMPERATURE,
            setting_keys=_PLOT_SETTINGS_TEMPERATURE_KEYS,
            gui_title="LiNaK Plot Controls: Temperature",
            analysis_name="temperature",
            plotter=plot_temperature_profiles,
            initial_context=initial_context,
            build_context=lambda current_args: _build_temperature_gui_context(
                current_args,
                sources=gui_sources,
            ),
            build_full_context=lambda current_args: _build_temperature_gui_context(
                current_args,
                sources=gui_sources,
            ),
        )
        LOGGER.info("Temperature GUI plotting session finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.temperature import plot_temperature_profiles

    render_context = _build_temperature_gui_context(args, sources=sources)
    _warn_for_non_gui_plot_complexity(analysis_name="temperature", render_context=render_context)
    _apply_effective_series_settings(
        args=args,
        sources=sources,
        profile_key=_PLOT_PROFILE_TEMPERATURE,
        fallback_labels_by_source=render_context.fallback_labels_by_source,
        series_descriptors=render_context.series_descriptors,
        allow_saved_multi_source_merge=True,
        materialize_default_colors=True,
    )
    _saved_path, _rendered_state = _render_profile_plot(
        args=args,
        source=render_context.plot_source_label,
        analysis_name="temperature",
        profile=render_context.profile,
        plotter=plot_temperature_profiles,
        plotter_kwargs=render_context.plotter_kwargs,
        series_descriptors=render_context.series_descriptors,
    )
    LOGGER.info("Temperature plotting finished in %.2f s.", perf_counter() - start)
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
            _rdf_mapping_summary_for_dry_run(args),
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
    _warn_for_non_gui_plot_complexity(analysis_name="rdf", render_context=render_context)

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
            f"species={args.species}, axis={args.axis}",
            _position_mapping_summary_for_dry_run(args),
            f"time_section_width={section_preview}",
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
    _warn_for_non_gui_plot_complexity(analysis_name="position", render_context=render_context)

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
            f"species_a={args.species_a}, species_b={species_b}, axis={args.axis}",
            _coordination_mapping_summary_for_dry_run(args),
            f"x_bin_width={getattr(args, 'x_bin_width', None)}",
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
    _warn_for_non_gui_plot_complexity(
        analysis_name="coordination",
        render_context=render_context,
    )

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
            _potential_mapping_summary_for_dry_run(args),
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
    if not use_gui:
        _warn_for_non_gui_plot_complexity(analysis_name="potential", render_context=render_context)
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
        plan = [
            "input mode: HDF5 only",
            f"sources ({len(sources)}): {_summarize_sources(sources)}",
            _orientation_mapping_summary_for_dry_run(args),
            f"render target: {render_target}",
        ]
        if settings_source_path is not None:
            plan.insert(-1, f"plot-settings source: {settings_source_path}")
        _log_dry_run_plan("plot orientation", plan)
        LOGGER.info("Orientation plotting dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.orientation import plot_orientation_profiles

    render_context = _build_orientation_gui_context(args, sources=sources)
    if not use_gui:
        _warn_for_non_gui_plot_complexity(
            analysis_name="orientation",
            render_context=render_context,
        )
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


def _add_spatial_filter_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x-range", help="Keep only atoms within the resolved X range <min:max>.")
    parser.add_argument("--y-range", help="Keep only atoms within the resolved Y range <min:max>.")
    parser.add_argument("--z-range", help="Keep only atoms within the resolved Z range <min:max>.")
    parser.add_argument(
        "--distance-range",
        help="Keep only atoms within the resolved distance-to-surface range <min:max>.",
    )
    parser.add_argument(
        "--keep-molecules-intact",
        action="store_true",
        help="Keep or discard full molecules based on a PBC-aware molecule center instead of atom-wise filtering.",
    )


def _spatial_filter_is_active(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in ("x_range", "y_range", "z_range", "distance_range")
    )


def _spatial_filter_surface_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    surface_axis = "z"
    for attr_name in ("surface_axis", "axis"):
        value = getattr(args, attr_name, None)
        if value is not None:
            surface_axis = str(value).strip().lower() or "z"
            break
    return {
        "surface_axis": surface_axis,
        "surface_mode": str(getattr(args, "surface_mode", "auto")).strip().lower() or "auto",
        "surface_elements": getattr(args, "surface_elements", None),
        "include_fixed_surface_atoms": bool(getattr(args, "include_fixed_surface_atoms", False)),
        "rough_surface_envelope_A": getattr(args, "rough_surface_envelope", None),
    }


def _apply_spatial_filter_from_cli_args(
    *,
    frames: list[Any],
    args: argparse.Namespace,
    precomputed_surface_estimate: Any | None = None,
) -> Any | None:
    if not _spatial_filter_is_active(args):
        return None
    from .trajectory.spatial_filter import (
        apply_spatial_filter,
        spatial_filter_options_from_mapping,
    )

    spatial_options = spatial_filter_options_from_mapping(
        {
            "x_range": getattr(args, "x_range", None),
            "y_range": getattr(args, "y_range", None),
            "z_range": getattr(args, "z_range", None),
            "distance_range": getattr(args, "distance_range", None),
            "keep_molecules_intact": bool(getattr(args, "keep_molecules_intact", False)),
        },
        **_spatial_filter_surface_config_from_args(args),
    )
    return apply_spatial_filter(
        frames,
        options=spatial_options,
        precomputed_surface_estimate=precomputed_surface_estimate,
    )


def _trajectory_hdf5_pbc_cache_matches(
    trajectory: str | Path,
    resolved_cell: tuple[float, float, float] | None,
) -> tuple[bool, tuple[float, float, float] | None]:
    from .trajectory.io import read_trajectory_hdf5_metadata

    metadata = read_trajectory_hdf5_metadata(trajectory)
    if metadata is None or not metadata.pbc_applied:
        return False, None
    cached_cell = metadata.pbc_cell_angstrom
    if cached_cell is not None and resolved_cell is not None:
        if not np.allclose(
            np.asarray(cached_cell),
            np.asarray(resolved_cell),
            rtol=1e-9,
            atol=1e-9,
        ):
            LOGGER.debug(
                "Conversion-cached PBC cell does not match requested cell for '%s' "
                "(cached=%s, requested=%s); recomputing PBC wrapping.",
                trajectory,
                cached_cell,
                resolved_cell,
            )
            return False, cached_cell
    return True, cached_cell


def _matching_cached_surface_from_trajectory(
    trajectory: str | Path,
    args: argparse.Namespace,
    frames: list[Any],
) -> Any | None:
    from .trajectory.io import read_trajectory_hdf5_surface_cache

    estimate = read_trajectory_hdf5_surface_cache(
        trajectory,
        axis=str(getattr(args, "axis", "z")),
        surface_mode=str(getattr(args, "surface_mode", "auto")),
        surface_elements=getattr(args, "surface_elements", None),
        include_fixed_surface_atoms=bool(getattr(args, "include_fixed_surface_atoms", False)),
        rough_surface_envelope_A=getattr(args, "rough_surface_envelope", None),
        frame_count=len(frames),
    )
    if estimate is not None:
        LOGGER.info(
            "Using conversion-cached surface positions from '%s' for axis %s.",
            _display_path(trajectory),
            str(getattr(args, "axis", "z")).upper(),
        )
    return estimate


def _handle_compute_density(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting density compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    surface_axis: str = args.axis
    density_outputs = _resolve_density_outputs_from_args(args)
    min_molecule_frames = int(getattr(args, "min_molecule_frames", 5))
    if min_molecule_frames < 1:
        raise ValueError("--min-molecule-frames must be >= 1.")
    oh_cutoff = float(getattr(args, "oh_cutoff", 1.27))
    grid_bin_width = getattr(args, "grid_bin_width", None)
    grid_max_nonzero_bins = int(getattr(args, "grid_max_nonzero_bins", 20_000_000))
    if grid_max_nonzero_bins < 1:
        raise ValueError("--grid-max-nonzero-bins must be >= 1.")

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
                )
            )
        else:
            output_preview = str(_default_density_hdf5_output_path(args.trajectory, args.species))

        plan = [
            f"trajectory source: {source_path}",
            (
                f"species={args.species}, raw axes=x/y/z, "
                f"distance axis={surface_axis}, "
                f"bin_width={args.bin_width}, "
                f"oh_cutoff={oh_cutoff}, "
                f"min_molecule_frames={min_molecule_frames}, "
                f"outputs={density_outputs}, "
                f"grid_bin_width={grid_bin_width if grid_bin_width is not None else args.bin_width}, "
                f"grid_max_nonzero_bins={grid_max_nonzero_bins}, "
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

    from .analysis.density import compute_all_density_profiles, save_density_profiles
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    default_output_path = _resolve_single_analysis_hdf5_output_path(
        None,
        _default_density_hdf5_output_path(args.trajectory, args.species),
    )
    output_path = (
        _preflight_prepare_output_path(
            _density_hdf5_output_path(
                args.output,
                args.trajectory,
                species=args.species,
            ),
            label="density HDF5 output",
        )
        if args.output is not None
        else None
    )
    _maybe_log_trajectory_convert_hint(source_path)
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="density",
    )
    LOGGER.info("Density preflight checks passed; loading trajectory.")
    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    cached_surface_estimate = _matching_cached_surface_from_trajectory(
        args.trajectory,
        args,
        frames,
    )
    spatial_filter_result = _apply_spatial_filter_from_cli_args(
        frames=frames,
        args=args,
        precomputed_surface_estimate=cached_surface_estimate,
    )
    if spatial_filter_result is not None:
        frames = spatial_filter_result.frames
        if spatial_filter_result.surface_estimate is not None:
            cached_surface_estimate = spatial_filter_result.surface_estimate
        if output_path is None:
            output_path = _preflight_prepare_output_path(
                default_output_path,
                label="density HDF5 output",
            )
    elif output_path is None:
        output_path = _preflight_prepare_output_path(
            default_output_path,
            label="density HDF5 output",
        )
    all_profiles = compute_all_density_profiles(
        frames=frames,
        species=args.species,
        surface_axis=surface_axis,
        bin_width=args.bin_width,
        surface_mode=args.surface_mode,
        surface_elements=args.surface_elements,
        include_fixed_surface_atoms=args.include_fixed_surface_atoms,
        binning="cell",
        surface_options=_surface_options_from_cli_args(args),
        precomputed_surface_estimate=cached_surface_estimate,
        outputs=density_outputs,
        grid_bin_width=grid_bin_width,
        grid_max_nonzero_bins=grid_max_nonzero_bins,
        oh_cutoff=oh_cutoff,
        min_molecule_frames=min_molecule_frames,
    )
    density_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
        "surface_axis": surface_axis,
        "oh_cutoff_A": oh_cutoff,
        "min_molecule_frames": min_molecule_frames,
        "grid_bin_width_A": grid_bin_width if grid_bin_width is not None else args.bin_width,
        "grid_max_nonzero_bins": grid_max_nonzero_bins,
    }
    if cell_input_path is not None:
        density_metadata["input_path"] = cell_input_path
    if resolved_cell is not None:
        density_metadata["resolved_cell_angstrom"] = list(resolved_cell)
    if spatial_filter_result is not None:
        density_metadata["spatial_filter"] = spatial_filter_result.metadata
    save_density_profiles(
        all_profiles,
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
    output = _preflight_prepare_output_path(
        _resolve_single_analysis_hdf5_output_path(
            args.output,
            _default_msd_hdf5_output_path(args.trajectory, args.species),
        ),
        label="MSD HDF5 output",
    )
    _maybe_log_trajectory_convert_hint(source_path)
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
    LOGGER.info("MSD preflight checks passed; loading trajectory.")
    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)
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


def _handle_compute_temperature(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting temperature compute.")
    _resolve_single_source_argument(
        args,
        positional_attr="source",
        source_label="temperature input file",
    )
    source_path = Path(args.source).expanduser().resolve()
    output = _resolve_single_analysis_hdf5_output_path(
        args.output,
        _default_temperature_hdf5_output_path(args.source),
    )

    if args.dry_run:
        input_preview = (
            str(Path(args.input).expanduser().resolve()) if args.input else "auto sibling input.inp"
        )
        plan = [
            f"temperature source: {source_path}",
            f"group_by={args.group_by}, velocity_unit={args.velocity_unit}, remove_com={bool(args.remove_com)}",
            f"metadata input: {input_preview}",
            f"output HDF5 target: {output}",
        ]
        _log_dry_run_plan("compute temperature", plan)
        LOGGER.info("Temperature compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.temperature import compute_temperature_profiles, save_temperature_profiles

    output_path = _preflight_prepare_output_path(output, label="temperature HDF5 output")
    profiles = compute_temperature_profiles(
        args.source,
        input_path=args.input,
        group_by=args.group_by,
        velocity_unit=args.velocity_unit,
        remove_com=bool(args.remove_com),
        atom_aliases=getattr(args, "atom_alias", None),
    )
    metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "group_by": args.group_by,
        "velocity_unit": args.velocity_unit,
        "remove_com": bool(args.remove_com),
    }
    if args.input:
        metadata["input_path"] = str(Path(args.input).expanduser().resolve())
    save_temperature_profiles(
        profiles,
        output_path,
        additional_metadata=metadata,
    )
    LOGGER.info("Temperature compute finished in %.2f s.", perf_counter() - start)
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
            "No --species provided for position analysis; computing one output per element "
            "and active O/H molecule selection."
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
            output_preview = str(_default_position_hdf5_output_path(args.trajectory, "all", args.axis))
        else:
            output_preview = str(
                _default_position_hdf5_output_path(args.trajectory, species_token, args.axis)
            )

        plan = [
            f"trajectory source: {source_path}",
            (
                f"species={species_token}, axis={args.axis}, timestep_fs={args.timestep_fs or 'auto'}, "
                f"oh_cutoff={args.oh_cutoff}, min_molecule_frames={args.min_molecule_frames}, "
                f"oh_topology_stride={args.oh_topology_stride}, {_describe_surface_cli_options(args)}"
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

    from .analysis.position import compute_position_profiles, save_position_profiles
    from .pbc import apply_pbc_to_frames
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    _maybe_log_trajectory_convert_hint(source_path)
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
    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
        analysis_label="position analysis",
    )
    pbc_corrected_positions = False
    pbc_cell: tuple[float, float, float] | None = None
    analysis_frames = frames
    pbc_cache_matches, cached_pbc_cell = _trajectory_hdf5_pbc_cache_matches(
        args.trajectory,
        resolved_cell,
    )
    if pbc_cache_matches:
        pbc_cell = cached_pbc_cell or (
            _cell_lengths_from_frame(frames[0])
            if _frames_have_usable_periodic_cell(frames)
            else None
        )
        pbc_corrected_positions = True
        LOGGER.info("Using conversion-cached PBC-wrapped coordinates for position analysis.")
    elif _frames_have_usable_periodic_cell(frames):
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
    cached_surface_estimate = _matching_cached_surface_from_trajectory(
        args.trajectory,
        args,
        analysis_frames,
    )
    spatial_filter_result = _apply_spatial_filter_from_cli_args(
        frames=analysis_frames,
        args=args,
        precomputed_surface_estimate=cached_surface_estimate,
    )
    if spatial_filter_result is not None:
        analysis_frames = spatial_filter_result.frames
        if spatial_filter_result.surface_estimate is not None:
            cached_surface_estimate = spatial_filter_result.surface_estimate
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
        precomputed_surface_estimate=cached_surface_estimate,
        oh_cutoff=args.oh_cutoff,
        min_molecule_frames=args.min_molecule_frames,
        oh_topology_stride=args.oh_topology_stride,
    )
    output_path = _position_hdf5_output_path(
        args.output,
        args.trajectory,
        profiles,
        axis=args.axis,
    )
    if output_path is None:
        raise ValueError("No position profiles were computed.")
    position_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
        "timestep_source": timestep_source,
        "frame_timestep_fs": float(timestep_fs),
        "positions_pbc_corrected": bool(pbc_corrected_positions),
        "oh_cutoff_A": float(args.oh_cutoff),
        "min_molecule_frames": int(args.min_molecule_frames),
        "oh_topology_stride": int(args.oh_topology_stride),
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
    if spatial_filter_result is not None:
        position_metadata["spatial_filter"] = spatial_filter_result.metadata
    save_position_profiles(profiles, output_path, additional_metadata=position_metadata)

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

    (
        selector_mode,
        selector_species_a,
        selector_species_b,
        selector_atoms_a,
        selector_atoms_b,
    ) = _resolve_compute_rdf_selectors(args)
    collection_mode = selector_mode in {"pairwise_collection", "species_collection"}
    default_output = _resolve_requested_analysis_hdf5_output_path(
        None,
        _default_rdf_collection_hdf5_output_path(args.trajectory),
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
                else (
                    f"{(max(1, int(np.floor((0.5 * min(resolved_cell)) / args.bin_width))) * args.bin_width):.6g} "
                    f"(auto rounded down from {0.5 * min(resolved_cell):.6g} to match bin_width={args.bin_width:.6g})"
                )
            )
        output_preview = str(
            _resolve_requested_analysis_hdf5_output_path(
                args.output,
                default_output,
            )
        )
        if selector_mode == "pairwise_collection":
            mode_preview = (
                "mode=pairwise element collection, species resolved from trajectory at execution, "
                f"r_max={args.r_max if args.r_max is not None else 'auto'}, "
                f"bin_width={args.bin_width}, threads={args.threads if args.threads is not None else 'auto'}"
            )
        elif selector_mode == "species_collection":
            selected_species = (
                selector_species_a if selector_species_a is not None else selector_species_b
            )
            selector_label = "species_a" if selector_species_a is not None else "species_b"
            mode_preview = (
                "mode=filtered pair collection, "
                f"{selector_label}={selected_species}, matching RDF pairs resolved from the trajectory at execution, "
                f"r_max={args.r_max if args.r_max is not None else 'auto'}, "
                f"bin_width={args.bin_width}, threads={args.threads if args.threads is not None else 'auto'}"
            )
        else:
            mode_preview = (
                "mode=single pair, "
                f"selector_a={_describe_compute_rdf_selector(species=selector_species_a, atom_indices=selector_atoms_a)}, "
                f"selector_b={_describe_compute_rdf_selector(species=selector_species_b, atom_indices=selector_atoms_b)}, r_max="
                f"{args.r_max if args.r_max is not None else 'auto'}, bin_width={args.bin_width}, "
                f"threads={args.threads if args.threads is not None else 'auto'}"
            )
        plan = [
            f"trajectory source: {source_path}",
            mode_preview,
            f"cell resolution: {cell_preview}",
            f"r_max resolution: {r_max_preview}",
            f"output HDF5 target: {output_preview}",
            "collection behavior: existing canonical RDF output is merged when compatible; otherwise a fallback suffix file is used",
        ]
        _log_dry_run_plan("compute rdf", plan)
        LOGGER.info("RDF compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .trajectory.io import read_trajectory
    from .analysis.rdf import (
        compute_rdf,
        compute_rdf_profiles,
        save_rdf_profiles,
    )

    source_path = Path(args.trajectory).expanduser().resolve()
    output_path = (
        _preflight_prepare_output_path(
            _resolve_requested_analysis_hdf5_output_path(args.output, default_output),
            label="RDF HDF5 output",
        )
        if args.output is not None
        else None
    )
    _maybe_log_trajectory_convert_hint(source_path)
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="RDF",
    )
    LOGGER.info("RDF preflight checks passed; loading trajectory.")
    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)
    resolved_cell, cell_source, cell_input_path = _resolve_and_apply_required_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="RDF",
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
    )
    rdf_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
        "resolved_cell_angstrom": list(resolved_cell),
    }
    spatial_filter_result = _apply_spatial_filter_from_cli_args(frames=frames, args=args)
    if spatial_filter_result is not None:
        frames = spatial_filter_result.frames
        rdf_metadata["spatial_filter"] = spatial_filter_result.metadata
        if output_path is None:
            output_path = _preflight_prepare_output_path(
                default_output,
                label="RDF HDF5 output",
            )
    elif output_path is None:
        output_path = _preflight_prepare_output_path(default_output, label="RDF HDF5 output")
    if cell_input_path is not None:
        rdf_metadata["input_path"] = cell_input_path
    if collection_mode:
        profiles = compute_rdf_profiles(
            frames=frames,
            r_max=args.r_max,
            bin_width=args.bin_width,
            threads=args.threads,
        )
        if selector_mode == "species_collection":
            profiles = [
                profile
                for profile in profiles
                if _rdf_profile_matches_species_filter(
                    profile,
                    species_a=selector_species_a,
                    species_b=selector_species_b,
                )
            ]
            if not profiles:
                selected_species = selector_species_a or selector_species_b
                raise ValueError(
                    f"No RDF pairs involving species '{selected_species}' were found in the trajectory."
                )
        save_rdf_profiles(
            profiles,
            output_path,
            additional_metadata=rdf_metadata,
            force_collection=True,
            merge_existing=True,
        )
    else:
        profile = compute_rdf(
            frames=frames,
            species_a=selector_species_a,
            species_b=selector_species_b,
            atom_indices_a=selector_atoms_a,
            atom_indices_b=selector_atoms_b,
            r_max=args.r_max,
            bin_width=args.bin_width,
            threads=args.threads,
        )
        save_rdf_profiles(
            [profile],
            output_path,
            additional_metadata=rdf_metadata,
            force_collection=True,
            merge_existing=True,
        )

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
    if args.species_a is None and args.species_b is None:
        if args.cutoff is not None:
            print(
                "Error: Provide at least one coordination selector via --species-a or --species-b.",
                file=sys.stderr,
            )
            return 2
        raise ValueError(
            "Provide at least one coordination selector via --species-a or --species-b."
        )
    single_pair_mode = args.species_a is not None and args.species_b is not None
    default_output = (
        _default_coordination_hdf5_output_path(args.trajectory, args.species_a, args.species_b)
        if single_pair_mode
        else _default_coordination_collection_hdf5_output_path(args.trajectory)
    )
    output_path = _resolve_single_analysis_hdf5_output_path(args.output, default_output)
    default_output = _resolve_single_analysis_hdf5_output_path(None, default_output)
    use_cutoff_from_rdf = bool(args.cutoff_from_rdf) or (
        args.cutoff is None and args.cutoff_rdf is None
    )
    single_diagnostic_plot_output = None
    if (args.cutoff_rdf or use_cutoff_from_rdf) and single_pair_mode:
        single_diagnostic_plot_output = output_path.with_name(f"{output_path.stem}_cutoff_rdf.png")

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
        if single_pair_mode:
            selection_preview = f"species_a={args.species_a}, species_b={args.species_b}"
        elif args.species_a is not None:
            selection_preview = f"species_a={args.species_a}, species_b=all matching stored species"
        else:
            selection_preview = f"species_a=all matching stored species, species_b={args.species_b}"
        if args.cutoff is not None:
            cutoff_preview = f"direct cutoff={args.cutoff:.6g} A"
        elif args.cutoff_rdf:
            cutoff_preview = f"RDF file={Path(args.cutoff_rdf).expanduser().resolve()}"
        else:
            cutoff_preview = "full-trajectory RDF cutoff"
        plan = [
            f"trajectory source: {source_path}",
            (f"{selection_preview}, axis={args.axis}, {_describe_surface_cli_options(args)}"),
            (
                f"coordination cutoff: {cutoff_preview}, smoothing_width="
                f"{args.cutoff_smoothing_width:.6g} A"
            ),
            f"cell resolution: {cell_preview}",
            f"timestep resolution: {resolved_timestep_fs:.6g} fs ({timestep_source})",
            f"output HDF5 target: {output_path}",
            (
                "cutoff diagnostic PNG: none"
                if single_diagnostic_plot_output is None
                else f"cutoff diagnostic PNG: {single_diagnostic_plot_output}"
            ),
        ]
        _log_dry_run_plan("compute coordination", plan)
        LOGGER.info("Coordination compute dry run finished in %.2f s.", perf_counter() - start)
        return 0

    from .analysis.coordination import (
        compute_coordination_profiles,
        resolve_coordination_cutoff,
        resolve_coordination_cutoffs,
        save_coordination_profiles,
    )
    from .pbc import apply_pbc_to_frames
    from .trajectory.io import read_trajectory

    source_path = Path(args.trajectory).expanduser().resolve()
    output_path = (
        _preflight_prepare_output_path(output_path, label="coordination HDF5 output")
        if args.output is not None
        else None
    )
    if args.cutoff_rdf is not None:
        _preflight_existing_file_path(args.cutoff_rdf, label="coordination cutoff RDF file")
    if single_diagnostic_plot_output is not None:
        _preflight_prepare_output_path(
            single_diagnostic_plot_output,
            label="coordination cutoff diagnostic PNG",
        )
    _maybe_log_trajectory_convert_hint(source_path)
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
    LOGGER.info("Coordination preflight checks passed; loading trajectory.")
    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)
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
    pbc_cache_matches, cached_pbc_cell = _trajectory_hdf5_pbc_cache_matches(
        args.trajectory,
        resolved_cell,
    )
    if pbc_cache_matches:
        pbc_cell = cached_pbc_cell or (
            _cell_lengths_from_frame(frames[0])
            if _frames_have_usable_periodic_cell(frames)
            else None
        )
        pbc_corrected_positions = True
        LOGGER.info("Using conversion-cached PBC-wrapped coordinates for coordination analysis.")
    elif _frames_have_usable_periodic_cell(frames):
        pbc_cell = _cell_lengths_from_frame(frames[0])
        analysis_frames = apply_pbc_to_frames(frames, pbc_cell)
        pbc_corrected_positions = True
    else:
        LOGGER.warning(
            "No usable periodic cell available for coordination analysis; using raw coordinates "
            "without PBC correction."
        )
    cached_surface_estimate = _matching_cached_surface_from_trajectory(
        args.trajectory,
        args,
        analysis_frames,
    )
    spatial_filter_result = _apply_spatial_filter_from_cli_args(
        frames=analysis_frames,
        args=args,
        precomputed_surface_estimate=cached_surface_estimate,
    )
    if spatial_filter_result is not None:
        analysis_frames = spatial_filter_result.frames
        if spatial_filter_result.surface_estimate is not None:
            cached_surface_estimate = spatial_filter_result.surface_estimate
        if output_path is None:
            output_path = _preflight_prepare_output_path(
                default_output,
                label="coordination HDF5 output",
            )
    elif output_path is None:
        output_path = _preflight_prepare_output_path(
            default_output,
            label="coordination HDF5 output",
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
    if spatial_filter_result is not None:
        coordination_metadata["spatial_filter"] = spatial_filter_result.metadata
    ordered_pairs = _resolve_compute_coordination_pairs(
        frames=analysis_frames,
        species_a=args.species_a,
        species_b=args.species_b,
    )
    diagnostic_plot_outputs: dict[tuple[str, str], Path] | None = None
    if args.cutoff_rdf or use_cutoff_from_rdf:
        if len(ordered_pairs) == 1 and single_diagnostic_plot_output is not None:
            diagnostic_plot_outputs = {ordered_pairs[0]: single_diagnostic_plot_output}
        else:
            diagnostic_plot_outputs = {}
            for pair_species_a, pair_species_b in ordered_pairs:
                diagnostic_candidate = output_path.with_name(
                    f"{output_path.stem}_{_sanitize_token(pair_species_a)}_{_sanitize_token(pair_species_b)}_cutoff_rdf.png"
                )
                diagnostic_plot_outputs[(pair_species_a, pair_species_b)] = (
                    _preflight_prepare_output_path(
                        diagnostic_candidate,
                        label="coordination cutoff diagnostic PNG",
                    )
                )
    if len(ordered_pairs) == 1:
        pair_species_a, pair_species_b = ordered_pairs[0]
        cutoff_resolutions = {
            ordered_pairs[0]: resolve_coordination_cutoff(
                frames=analysis_frames,
                species_a=pair_species_a,
                species_b=pair_species_b,
                cutoff_A=args.cutoff,
                cutoff_rdf_path=args.cutoff_rdf,
                cutoff_from_rdf=use_cutoff_from_rdf,
                cutoff_smoothing_width_A=args.cutoff_smoothing_width,
                diagnostic_plot_output=(
                    None
                    if diagnostic_plot_outputs is None
                    else diagnostic_plot_outputs.get(ordered_pairs[0])
                ),
            )
        }
    else:
        cutoff_resolutions = resolve_coordination_cutoffs(
            frames=analysis_frames,
            ordered_pairs=ordered_pairs,
            cutoff_A=args.cutoff,
            cutoff_rdf_path=args.cutoff_rdf,
            cutoff_from_rdf=use_cutoff_from_rdf,
            cutoff_smoothing_width_A=args.cutoff_smoothing_width,
            diagnostic_plot_outputs=diagnostic_plot_outputs,
        )
    profiles = compute_coordination_profiles(
        frames=analysis_frames,
        ordered_pairs=ordered_pairs,
        axis=args.axis,
        timestep_fs=timestep_fs,
        surface_mode=args.surface_mode,
        surface_elements=args.surface_elements,
        include_fixed_surface_atoms=args.include_fixed_surface_atoms,
        surface_options=_surface_options_from_cli_args(args),
        precomputed_surface_estimate=cached_surface_estimate,
        cutoff_resolutions=cutoff_resolutions,
    )
    save_coordination_profiles(profiles, output_path, additional_metadata=coordination_metadata)

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
        expand_hartree_cube_sources,
        plan_potential_csv_output,
        summarize_potential_statistics,
        validate_hartree_cube_source,
    )
    from .progress import ProgressBar

    raw_sources = _resolve_plot_sources(args)
    if len(raw_sources) > 1:
        LOGGER.info("Received %d potential input file(s).", len(raw_sources))

    validated_sources: list[str] = []
    duplicate_sources: list[str] = []
    seen_sources: set[str] = set()
    expanded_sources: list[Any] = []
    expanded_source_labels: list[str] = []
    seen_expanded_sources: set[str] = set()
    with ProgressBar(
        desc="Loading potential inputs",
        total=len(raw_sources),
        unit="file",
    ) as progress:
        for source in raw_sources:
            resolved = validate_hartree_cube_source(source)
            key = str(resolved)
            if key in seen_sources:
                duplicate_sources.append(key)
                progress.update()
                continue
            seen_sources.add(key)
            validated_sources.append(key)
            for dataset in expand_hartree_cube_sources(resolved):
                dataset_source = str(Path(dataset.source_path or key).expanduser().resolve())
                dataset_profile_index = int(dataset.source_profile_index or 0)
                expanded_key = f"{dataset_source}::profile:{dataset_profile_index}"
                if expanded_key in seen_expanded_sources:
                    duplicate_sources.append(expanded_key)
                    continue
                seen_expanded_sources.add(expanded_key)
                expanded_sources.append(dataset)
                expanded_source_labels.append(
                    dataset_source if dataset_profile_index == 0 else expanded_key
                )
            progress.update()

    if not expanded_sources:
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
    skip_existing = [
        source_label for source_label in expanded_source_labels if source_label in existing_sources
    ]
    sources_to_compute = [
        source
        for source, source_label in zip(expanded_sources, expanded_source_labels)
        if source_label not in existing_sources
    ]

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
    default_output_path = _resolve_single_analysis_hdf5_output_path(
        None,
        _default_orientation_hdf5_output_path(args.trajectory, args.axis),
    )
    output_path = (
        _preflight_prepare_output_path(
            _orientation_hdf5_output_path(
                args.output,
                args.trajectory,
                axis=args.axis,
            ),
            label="orientation HDF5 output",
        )
        if args.output is not None
        else None
    )
    _maybe_log_trajectory_convert_hint(source_path)
    pre_resolved_cell, preflight_cell_error = _preflight_resolve_cell(
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        analysis_name="orientation",
    )
    LOGGER.info("Orientation preflight checks passed; loading trajectory.")
    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)
    resolved_cell, cell_source, cell_input_path = _maybe_apply_density_cell(
        frames,
        args.trajectory,
        cell=_normalize_cell_args(args),
        input_path=args.input,
        pre_resolved=pre_resolved_cell,
        preflight_error=preflight_cell_error,
        analysis_label="orientation analysis",
    )
    cached_surface_estimate = _matching_cached_surface_from_trajectory(
        args.trajectory,
        args,
        frames,
    )
    spatial_filter_result = _apply_spatial_filter_from_cli_args(
        frames=frames,
        args=args,
        precomputed_surface_estimate=cached_surface_estimate,
    )
    if spatial_filter_result is not None:
        frames = spatial_filter_result.frames
        if spatial_filter_result.surface_estimate is not None:
            cached_surface_estimate = spatial_filter_result.surface_estimate
        if output_path is None:
            output_path = _preflight_prepare_output_path(
                default_output_path,
                label="orientation HDF5 output",
            )
    elif output_path is None:
        output_path = _preflight_prepare_output_path(
            default_output_path,
            label="orientation HDF5 output",
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
        precomputed_surface_estimate=cached_surface_estimate,
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
    orientation_metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "cell_source": cell_source,
    }
    if cell_input_path is not None:
        orientation_metadata["input_path"] = cell_input_path
    if resolved_cell is not None:
        orientation_metadata["resolved_cell_angstrom"] = list(resolved_cell)
    if spatial_filter_result is not None:
        orientation_metadata["spatial_filter"] = spatial_filter_result.metadata
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

    frames = _read_trajectory_with_optional_atom_aliases(read_trajectory, args.trajectory, args)

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
    write_trajectory(wrapped_frames, output_path, raw_species_as_symbols=True)

    LOGGER.info("PBC application finished in %.2f s.", perf_counter() - start)
    return 0


def _handle_apply_convert(args: argparse.Namespace) -> int:
    from .conversion import (
        CONVERSION_REGISTRY,
        CubeConversionOptions,
        TrajectoryConversionOptions,
        parse_trajectory_selection,
    )

    start = perf_counter()
    LOGGER.info("Starting file conversion.")
    _resolve_single_source_argument(
        args,
        positional_attr="trajectory",
        source_label="trajectory input file",
    )

    request = CONVERSION_REGISTRY.build_default_request(
        args.trajectory,
        output_path=args.output,
        target_selector=args.target_file_type,
        uniquify_default_output=True,
        output_name_suffix=(
            None
            if args.select is None or args.output is not None
            else parse_trajectory_selection(args.select).suffix
        ),
    )
    if (
        args.output is None
        and args.target_file_type is None
        and args.select is None
        and request.source_path == request.target_path
        and request.source_file_type == request.target_file_type
    ):
        LOGGER.info(
            "Conversion skipped: '%s' is already LiNaK's preferred %s working format.",
            request.source_path,
            request.family,
        )
        print(request.source_path)
        return 0

    options: TrajectoryConversionOptions | CubeConversionOptions
    if request.family == "trajectory":
        options = TrajectoryConversionOptions(
            input_path=args.input,
            select=args.select,
            x_range=getattr(args, "x_range", None),
            y_range=getattr(args, "y_range", None),
            z_range=getattr(args, "z_range", None),
            distance_range=getattr(args, "distance_range", None),
            keep_molecules_intact=bool(getattr(args, "keep_molecules_intact", False)),
            output_was_default=bool(args.output is None),
            atom_aliases=tuple(getattr(args, "atom_alias", None) or ()),
        )
    else:
        options = CubeConversionOptions()

    if args.dry_run:
        plan = CONVERSION_REGISTRY.describe_plan(request, options=options)
        _log_dry_run_plan("apply convert", plan)
        LOGGER.info("File conversion dry run finished in %.2f s.", perf_counter() - start)
        return 0

    # Conversion routing, metadata discovery, and family-specific execution live
    # in `linak.conversion`; the CLI only resolves arguments and reports results.
    result = CONVERSION_REGISTRY.execute(request, options=options)
    converted_path = result.output_path
    if request.family == "trajectory":
        from .trajectory.io import read_trajectory

        frames = read_trajectory(converted_path)
        LOGGER.info(
            "File conversion finished in %.2f s. family=%s frames=%d atoms/frame=%d output=%s",
            perf_counter() - start,
            request.family,
            len(frames),
            len(frames[0]),
            converted_path,
        )
    else:
        LOGGER.info(
            "File conversion finished in %.2f s. family=%s output=%s",
            perf_counter() - start,
            request.family,
            converted_path,
        )
    print(converted_path)
    return 0


def _handle_apply_combine(args: argparse.Namespace) -> int:
    from .conversion import CONVERSION_REGISTRY, TrajectoryConversionOptions

    start = perf_counter()
    LOGGER.info("Starting file combine.")
    sources = _resolve_source_arguments(
        positional=getattr(args, "trajectory", None),
        files=getattr(args, "files", None),
        source_label="input file",
        allow_multiple=True,
    )
    if len(sources) < 2:
        raise ValueError("linak apply combine requires at least two input files.")

    request = CONVERSION_REGISTRY.build_combine_request(
        sources,
        output_path=args.output,
        no_convert=bool(args.no_convert),
        uniquify_default_output=True,
    )
    options = TrajectoryConversionOptions(
        input_path=getattr(args, "input", None),
        cell=(
            None
            if getattr(args, "cell", None) is None
            else tuple(float(value) for value in args.cell)
        ),
        atom_aliases=tuple(getattr(args, "atom_alias", None) or ()),
    )

    if args.dry_run:
        plan = CONVERSION_REGISTRY.describe_combine_plan(request, options=options)
        _log_dry_run_plan("apply combine", plan)
        LOGGER.info("File combine dry run finished in %.2f s.", perf_counter() - start)
        return 0

    result = CONVERSION_REGISTRY.execute_combine(request, options=options)
    LOGGER.info(
        "File combine finished in %.2f s. family=%s sources=%d output=%s",
        perf_counter() - start,
        request.family,
        len(request.source_paths),
        result.output_path,
    )
    print(result.output_path)
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


def _handle_apply_pack(args: argparse.Namespace) -> int:
    start = perf_counter()
    LOGGER.info("Starting simulation directory pack.")

    from .out_h5 import (
        OutH5PackOptions,
        default_out_h5_output_path,
        discover_simulation_directory,
        pack_simulation_directory,
        unique_out_h5_output_path,
    )

    source_dir = Path(args.source_dir).expanduser().resolve()
    options = OutH5PackOptions(
        include=tuple(args.include or ()),
        exclude=tuple(args.exclude or ()),
        overwrite=bool(args.overwrite),
        drop_sections=tuple(args.drop or ()),
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_out_h5_output_path(source_dir)
    )
    if output_path.exists() and not args.overwrite:
        output_path = unique_out_h5_output_path(output_path)

    if args.dry_run:
        discovery = discover_simulation_directory(source_dir, options=options)
        plan = [
            f"source directory: {discovery.source_dir}",
            f"output container: {output_path}",
            f"trajectory candidates: {len(discovery.trajectories)}",
            f"cube candidates: {len(discovery.cubes)}",
            f"CP2K output candidates: {len(discovery.cp2k_outputs)}",
            f"skipped candidates: {len(discovery.skipped)}",
            "container schema: linak-out-hdf5 v1",
            "raw source handling: referenced in provenance, not copied into project directories",
        ]
        _log_dry_run_plan("apply pack", plan)
        LOGGER.info("Directory pack dry run finished in %.2f s.", perf_counter() - start)
        return 0

    result = pack_simulation_directory(
        source_dir,
        output_path,
        options=options,
        logger=lambda level, message: getattr(LOGGER, str(level).lower(), LOGGER.info)(message),
    )
    LOGGER.info(
        "Directory pack finished in %.2f s. output=%s frames=%s cubes=%d cp2k_outputs=%d",
        perf_counter() - start,
        result.output_path,
        result.summary.frame_count,
        result.summary.cube_count,
        result.summary.cp2k_output_count,
    )
    print(result.output_path)
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

    known_subcommands = {
        "density",
        "msd",
        "rdf",
        "position",
        "coordination",
        "potential",
        "orientation",
        "temperature",
    }
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
        "temperature",
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
    specialized_help_rc = _maybe_handle_analysis_specific_plot_help(runtime_argv)
    if specialized_help_rc is not None:
        return specialized_help_rc
    runtime_argv = _rewrite_implicit_plot_csv(runtime_argv)
    runtime_argv = _rewrite_implicit_csv_interactive(runtime_argv)
    parser = build_parser()
    try:
        args = parser.parse_args(runtime_argv)
    except SystemExit as exc:
        return int(exc.code or 0)
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
