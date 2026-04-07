"""PySide6 GUI panel for interactive plot settings."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
import html
from pathlib import Path
import re
import tempfile
from collections.abc import Sequence
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from .fitting import coerce_fit_config, default_fit_config, supported_fit_types
from .plotting import (
    _describe_error_provenance,
    _coerce_plot_annotations,
    DEFAULT_PLOT_STYLE,
    default_plot_font_sizes,
    default_series_colors,
    resolve_series_error_availability,
)

_LEGEND_LOCATIONS = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
)
_BIN_REDUCERS = ("mean", "median", "sum", "min", "max")
_NORMALIZATION_MODES = ("none", "max", "area", "value_at_x", "factor")
_GRID_AXES = ("both", "x", "y")
_GRID_WHICH = ("major", "minor", "both")
_TICK_AXES = ("both", "x", "y")
_TICK_VISIBILITY_MODES = ("both", "x", "y", "none")
_TICK_DIRECTIONS = ("out", "in", "inout")
_MINOR_TICKS_MODES = ("off", "on")
_TOGGLE_MODES = ("on", "off")
_BORDER_MODES = ("on", "off", "custom")
_SYNC_MODES = ("Auto", "Manual")
_TEXT_SYNC_MODES = ("Auto", "Manual", "Off")
_FIT_TYPES = supported_fit_types()
_FIT_RANGE_MODES = ("visible", "manual")
_ERROR_STATS = ("sample_sem", "sample_std", "block_sem", "block_std")
_ERROR_STYLES = ("band", "whiskers")
_GROUP_REDUCERS = ("mean", "median", "sum", "min", "max")
_ANNOTATION_TYPES = ("text", "line", "arrow")
_ANNOTATION_COORD_SYSTEMS = ("data", "axes")
_ANNOTATION_LINE_STYLES = ("-", "--", "-.", ":")
_ANNOTATION_ARROW_STYLES = ("->", "-|>", "<->", "simple", "fancy")
_ANNOTATION_HORIZONTAL_ALIGN = ("left", "center", "right")
_ANNOTATION_VERTICAL_ALIGN = ("top", "center", "bottom", "baseline")
_PROFILE_FILTER_METADATA_LABEL = "Use stored metadata"
_PROFILE_FILTER_SPECIES_B_AUTO_LABEL = "Same as Species A / stored metadata"
_AUTO_PREVIEW_DEBOUNCE_MS = 1000
_HEATMAP_NORMALIZATION_LABEL_BY_MODE = {
    "counts": "Raw frequencies",
    "global_probability": "Global normalization",
    "bulk_water_reference": "Normalize to water bulk",
}
_HEATMAP_NORMALIZATION_MODE_BY_LABEL = {
    label: mode for mode, label in _HEATMAP_NORMALIZATION_LABEL_BY_MODE.items()
}
_TRI_STATE_SYNC_FIELD_KEYS = frozenset({"title", "x_label", "y_label"})
_SERIES_SPECIFIC_SETTINGS = frozenset(
    {
        "series_order",
        "series_labels",
        "series_descriptors",
        "series_overrides",
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
    }
)
_SYNCED_FIELD_KEYS = frozenset(
    {
        "title",
        "x_label",
        "y_label",
        "x_lim",
        "y_lim",
        "x_ticks",
        "y_ticks",
        "x_label_pad",
        "y_label_pad",
    }
)
_NEW_PROFILE_RESET_KEYS = frozenset(
    {
        "series_order",
        "series_overrides",
        "series_enabled",
        "series_show_in_legend",
        "series_alpha",
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
        "line_colors",
    }
)
_TOOLTIPS: dict[str, str] = {
    "shared.sync_mode": "Auto follows the preview; Manual keeps your typed value.",
    "shared.color_picker": "Opens a color picker.",
    "preview.refresh": "Renders the current settings again.",
    "preview.fit": "Fits the preview image to the panel.",
    "preview.actual_size": "Shows the preview at actual size.",
    "preview.reset": "Restores the default plot settings.",
    "preview.auto_update": "Refreshes the preview after each change.",
    "profiles.selector": "Chooses which saved plot profile is active.",
    "profiles.new": "Creates a new profile from the defaults.",
    "profiles.rename": "Renames the current profile.",
    "profiles.duplicate": "Copies the current profile under a new name.",
    "profiles.delete": "Deletes the current profile.",
    "profiles.save": "Saves the current settings to this profile.",
    "profiles.reset": "Restores the default plot settings.",
    "profiles.import": "Loads profile settings from a JSON file.",
    "profiles.export_json": "Saves this profile to a JSON file.",
    "export.transparent": "Saves the figure with a transparent background.",
    "export.figure": "Saves the current figure to an image file.",
    "data.density.x_values": "Chooses what is shown on the x-axis.",
    "data.density.quantity": "Chooses whether density is mass or number based.",
    "data.profile.species": "Filters the stored profile by species.",
    "data.profile.axis": "Filters the stored profile by axis.",
    "data.rdf.species_a": "Chooses the first species in the stored RDF profile.",
    "data.rdf.species_b": "Chooses the second species in the stored RDF profile.",
    "data.coordination.species_a": "Chooses the center species in the stored profile.",
    "data.coordination.species_b": "Chooses the neighbor species in the stored profile.",
    "data.coordination.axis": "Chooses which stored axis profile to load.",
    "data.position.component": "Chooses which position component to plot.",
    "data.position.color_by": "Chooses what colors the xy-z map.",
    "data.position.time_axis": "Chooses the time unit on the x-axis.",
    "data.coordination.component": "Chooses which coordination view to plot.",
    "data.coordination.time_axis": "Chooses the time unit on the x-axis.",
    "data.orientation.component": "Chooses the averaging mode: mean cos(angle), density-weighted, or 2-D heatmap.",
    "data.orientation.angle": "Chooses which angle to display: polar (bisector vs surface normal) or azimuthal (molecular-plane normal).",
    "data.potential.x_axis": "Shows which value is used on the x-axis.",
    "data.potential.total_rows": "Shows how many records were loaded.",
    "data.potential.complete_rows": "Shows how many records have complete values.",
    "data.potential.incomplete_rows": "Shows how many records have missing values.",
    "data.section.width": "Groups nearby x-values into wider sections.",
    "data.section.reducer": "Chooses how each section is summarized.",
    "data.section.min_points": "Requires at least this many contributing raw points in a plotted bin before LiNaK shows the value, uncertainty, or fit input for that bin.",
    "data.section.y_width": "Groups nearby y-values (angle bins) into wider sections. Only used for heatmaps.",
    "data.section.y_reducer": "Chooses how each y-section is summarized.",
    "series.all_on": "Turns every series on.",
    "series.all_off": "Turns every series off.",
    "series.duplicate": "Duplicates the selected base or grouped series.",
    "series.add_group": "Adds a grouped series that can aggregate several loaded base series.",
    "series.show_in_legend": "Shows or hides this series in the legend.",
    "series.label": "Sets the legend name for this series.",
    "series.color": "Sets the line color for this series.",
    "series.alpha": "Sets the line transparency for this series.",
    "series.line_width": "Sets the line width for this series.",
    "series.marker": "Sets the marker shape for this series.",
    "series.error_enabled": "Turns uncertainty display for this series on or off.",
    "series.error_stat": "Chooses the uncertainty statistic. sample_* uses sample spread or SEM, block_* uses contiguous frame-block spread or SEM when saved by the analysis.",
    "series.error_style": "Chooses whether uncertainty is shown as a shaded band or whiskers around the current series.",
    "series.error_color": "Sets a separate error-overlay color. Leave blank to follow the base series color.",
    "series.error.summary": "Shows which uncertainty is currently rendered, what data it is computed from, and any fallback or availability reason.",
    "series.fit_enabled": "Turns fitting for this series on or off.",
    "series.fit_type": "Chooses the fitting model.",
    "series.fit_degree": "Sets the polynomial degree.",
    "series.fit_range_mode": "Chooses whether fit range follows the view or manual limits.",
    "series.fit_x_min": "Sets the minimum x value used for fitting.",
    "series.fit_x_max": "Sets the maximum x value used for fitting.",
    "series.fit_show_in_legend": "Shows or hides the fit in the legend.",
    "series.fit_label": "Sets the legend name for the fitted series.",
    "series.line_kwargs_json": "Adds extra Matplotlib line options for this series.",
    "series.norm.mode": "Chooses how this series is normalized.",
    "series.norm.target": "Sets the value used by the normalization.",
    "series.norm.reference_x": "Sets the x-value used for value-at-x normalization.",
    "series.meta.default_label": "Shows the original label of this series.",
    "series.meta.source_file": "Shows which file this series came from.",
    "series.meta.source_directory": "Shows where the source file is located.",
    "series.meta.series_id": "Shows the internal ID of this series.",
    "series.fit.summary": "Shows the current fit result for this series.",
    "series.fit.warning": "Shows why the fit could not be made.",
    "series.cumulative_enabled": "Turns the cumulative-average derived line for this series on or off.",
    "series.cumulative_show_in_legend": "Shows or hides the cumulative-average line in the legend.",
    "series.cumulative_label": "Sets the legend name for the cumulative-average line.",
    "series.cumulative.summary": "Shows the current cumulative-average derived-line status for this series.",
    "series.group.reducer": "Chooses how the selected member series are aggregated into one grouped line.",
    "series.group.members": "Chooses which loaded non-group base series contribute to this grouped line.",
    "annotations.list": "Shows the figure-level annotations applied on top of the current plot.",
    "annotations.add_text": "Adds a new text annotation.",
    "annotations.add_line": "Adds a new line annotation.",
    "annotations.add_arrow": "Adds a new arrow annotation.",
    "annotations.duplicate": "Duplicates the selected annotation.",
    "annotations.delete": "Deletes the selected annotation.",
    "annotations.move_up": "Moves the selected annotation earlier in the draw order.",
    "annotations.move_down": "Moves the selected annotation later in the draw order.",
    "annotations.enabled": "Shows or hides this annotation.",
    "annotations.type": "Chooses whether the annotation is text, a line, or an arrow.",
    "annotations.name": "Sets the annotation name shown in the list.",
    "annotations.coord_system": "Chooses whether coordinates follow the data axes or the axes frame.",
    "annotations.color": "Sets the annotation color.",
    "annotations.alpha": "Sets the annotation opacity between 0 and 1.",
    "annotations.zorder": "Sets the annotation draw order. Higher values render on top.",
    "annotations.text": "Text shown by this annotation.",
    "annotations.x": "X position for a text annotation.",
    "annotations.y": "Y position for a text annotation.",
    "annotations.font_size": "Font size used by the text annotation.",
    "annotations.rotation": "Text rotation in degrees.",
    "annotations.horizontal_align": "Horizontal alignment for the text anchor.",
    "annotations.vertical_align": "Vertical alignment for the text anchor.",
    "annotations.x1": "Start x position for a line or arrow.",
    "annotations.y1": "Start y position for a line or arrow.",
    "annotations.x2": "End x position for a line or arrow.",
    "annotations.y2": "End y position for a line or arrow.",
    "annotations.line_width": "Line width for a line or arrow.",
    "annotations.line_style": "Line style for a line or arrow.",
    "annotations.arrow_style": "Arrow head style for an arrow annotation.",
    "annotations.mutation_scale": "Arrow head size for an arrow annotation.",
    "annotations.summary": "Shows the current preview summary for the selected annotation.",
    "figure.text.title": "Sets the plot title.",
    "figure.text.x_label": "Sets the x-axis label.",
    "figure.text.y_label": "Sets the y-axis label.",
    "figure.text.title_font": "Sets the title font size.",
    "figure.text.label_font": "Sets the axis label font size.",
    "figure.legend.enabled": "Shows or hides the legend.",
    "figure.legend.title": "Sets the legend title.",
    "figure.legend.location": "Chooses where the legend is placed.",
    "figure.legend.frame": "Shows or hides the legend box.",
    "figure.legend.columns": "Sets how many columns the legend uses.",
    "figure.legend.font": "Sets the legend font size.",
    "figure.axes.x_scale": "Chooses the x-axis scale.",
    "figure.axes.y_scale": "Chooses the y-axis scale.",
    "figure.axes.border": "Controls the plot border: 'on' shows all four spines, 'off' hides them all, 'custom' lets you choose each side individually.",
    "figure.axes.border_sides": "Choose which individual spines to show when border mode is set to 'custom'.",
    "figure.axes.label_font": "Sets the axis label font size.",
    "figure.axes.x_limits": "Sets the x-axis limits.",
    "figure.axes.y_limits": "Sets the y-axis limits.",
    "figure.axes.x_label_pad": "Sets the space below the x-axis label.",
    "figure.axes.y_label_pad": "Sets the space beside the y-axis label.",
    "figure.ticks.show": "Chooses which axes show ticks.",
    "figure.ticks.x_ticks": "Sets the x-axis tick positions.",
    "figure.ticks.y_ticks": "Sets the y-axis tick positions.",
    "figure.ticks.x_rotation": "Rotates the x-axis tick labels.",
    "figure.ticks.y_rotation": "Rotates the y-axis tick labels.",
    "figure.ticks.font": "Sets the tick label font size.",
    "figure.ticks.direction": "Chooses whether ticks point in or out.",
    "figure.ticks.length": "Sets the tick length.",
    "figure.ticks.width": "Sets the tick width.",
    "figure.ticks.minor": "Shows or hides minor ticks.",
    "figure.grid.show": "Shows or hides the grid.",
    "figure.grid.line_style": "Sets the grid line style.",
    "figure.grid.line_width": "Sets the grid line width.",
    "figure.grid.alpha": "Sets the grid transparency.",
    "figure.grid.color": "Sets the grid color.",
    "figure.grid.axis": "Chooses which axis gets grid lines.",
    "figure.grid.lines": "Chooses major, minor, or both grid lines.",
    "figure.lines.width": "Sets the default line width.",
    "figure.lines.style": "Sets the default line style.",
    "figure.lines.alpha": "Sets the default line transparency.",
    "figure.lines.markers": "Shows or hides markers on lines.",
    "figure.lines.marker_size": "Sets the default marker size.",
    "figure.heatmap.vmin": "Minimum value for the colorbar range. Leave blank for auto.",
    "figure.heatmap.vmax": "Maximum value for the colorbar range. Leave blank for auto.",
    "figure.heatmap.cmap": "Matplotlib colormap name for the heatmap.",
    "figure.heatmap.normalize_mode": "Choose raw frequencies, whole-heatmap global probability, or bulk-water-referenced normalization.",
    "figure.heatmap.log_scale": "Use a logarithmic color scale. Zero and negative cells are masked.",
    "figure.heatmap.colorbar_enabled": "Show or hide the colorbar.",
    "figure.heatmap.colorbar_label": "Colorbar label text. 'none' hides the label; blank uses the default.",
    "figure.heatmap.colorbar_label_size": "Font size for the colorbar label.",
    "figure.heatmap.colorbar_tick_size": "Font size for the colorbar tick labels.",
    "figure.heatmap.colorbar_position": "Where the colorbar is placed relative to the plot.",
    "figure.heatmap.colorbar_pad": "Space between the plot and the colorbar.",
    "figure.heatmap.colorbar_shrink": "Fraction of the axes height (or width) used by the colorbar.",
    "figure.heatmap.colorbar_aspect": "Ratio of the long to short dimension of the colorbar.",
    "figure.canvas.width": "Sets the figure width.",
    "figure.canvas.height": "Sets the figure height.",
    "figure.canvas.dpi": "Sets the render resolution.",
    "figure.canvas.font_family": "Sets the main font family.",
    "figure.canvas.font_size": "Sets the default font size used when title, label, tick, and legend font fields are left blank.",
    "figure.canvas.facecolor": "Sets the figure background color.",
    "advanced.matplotlib_guide_link": "Opens the official Matplotlib user guide.",
    "advanced.rcparams": "Sets raw Matplotlib rcParams.",
    "advanced.figure_kwargs": "Sets raw figure options.",
    "advanced.axes_kwargs": "Sets raw axes options.",
    "advanced.tight_layout_kwargs": "Sets raw tight-layout options.",
    "advanced.savefig_kwargs": "Sets raw savefig options.",
    "advanced.legend_kwargs": "Sets raw legend options.",
    "advanced.grid_kwargs": "Sets raw grid options.",
    "advanced.tick_params_kwargs": "Sets raw tick options.",
    "advanced.line_kwargs": "Sets raw default line options.",
}


def _without_series_specific_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without per-series plot controls."""
    return {key: value for key, value in settings.items() if key not in _SERIES_SPECIFIC_SETTINGS}


def _without_new_profile_series_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(settings)
    for key in _NEW_PROFILE_RESET_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _toggle_to_mode(value: bool | None, *, auto_mode: str = "on") -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return auto_mode


def _mode_to_toggle(value: str) -> bool | None:
    token = value.strip().lower()
    if token == "on":
        return True
    if token == "off":
        return False
    return None


def _border_setting_to_mode(value: Any) -> str:
    if value is False:
        return "off"
    if isinstance(value, dict):
        return "custom"
    return "on"


def _border_spines_from_setting(value: Any) -> dict[str, bool]:
    if isinstance(value, dict):
        return {s: bool(value.get(s, True)) for s in ("left", "right", "top", "bottom")}
    visible = value is not False
    return {"left": visible, "right": visible, "top": visible, "bottom": visible}


def _lock_to_sync_mode(locked: bool) -> str:
    return "Manual" if bool(locked) else "Auto"


def _sync_mode_to_lock(value: str) -> bool:
    return str(value).strip().lower() == "manual"


def _preview_button_enabled(*, auto_update_enabled: bool, preview_loading: bool) -> bool:
    return (not auto_update_enabled) and (not preview_loading)


def _explicit_text(value: str) -> str:
    return str(value).strip()


def _optional_float(value: str, *, field_name: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a float.") from exc


def _optional_int(value: str, *, field_name: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _display_positive_int(value: Any, *, fallback: int) -> str:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    if parsed <= 0:
        parsed = fallback
    return str(parsed)


def _display_optional_positive_int(value: Any) -> str:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return ""
    return str(parsed) if parsed > 0 else ""


def _font_size_placeholder_text(size: int) -> str:
    return f"Auto ({int(size)})"


def _annotation_type_label(annotation_type: str) -> str:
    token = str(annotation_type).strip().lower()
    if token == "text":
        return "Text"
    if token == "line":
        return "Line"
    if token == "arrow":
        return "Arrow"
    return "Annotation"


def _default_annotation_name(annotation_type: str, *, index: int) -> str:
    return f"{_annotation_type_label(annotation_type)} {index}"


def _annotation_defaults_for_gui(annotation_type: str, *, index: int) -> dict[str, Any]:
    normalized_type = str(annotation_type).strip().lower()
    if normalized_type not in _ANNOTATION_TYPES:
        normalized_type = "text"
    defaults: dict[str, Any] = {
        "id": f"annotation:{uuid4().hex}",
        "type": normalized_type,
        "enabled": True,
        "name": _default_annotation_name(normalized_type, index=index),
        "coord_system": "axes" if normalized_type == "text" else "data",
        "color": "#000000",
        "alpha": "1.0",
        "zorder": "5",
        "text": "",
        "x": "0.5",
        "y": "0.5",
        "font_size": "12",
        "rotation": "0",
        "horizontal_align": "center",
        "vertical_align": "center",
        "x1": "0",
        "y1": "0",
        "x2": "1",
        "y2": "1",
        "line_width": "1.5",
        "line_style": "-",
        "arrow_style": "->",
        "mutation_scale": "12",
    }
    if normalized_type == "text":
        defaults["text"] = defaults["name"]
    return defaults


def _annotation_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _truncate_annotation_label(value: str, *, limit: int = 42) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}..."


def _annotation_fallback_title(entry: dict[str, Any], *, index: int) -> str:
    annotation_type = str(entry.get("type") or "text").strip().lower()
    if annotation_type == "text":
        text_value = _annotation_string(entry.get("text"))
        if text_value:
            return _truncate_annotation_label(text_value)
        return _default_annotation_name(annotation_type, index=index)
    if annotation_type in {"line", "arrow"}:
        return (
            f"{_annotation_type_label(annotation_type)} "
            f"({_annotation_string(entry.get('x1') or '0')}, {_annotation_string(entry.get('y1') or '0')} -> "
            f"{_annotation_string(entry.get('x2') or '1')}, {_annotation_string(entry.get('y2') or '1')})"
        )
    return _default_annotation_name(annotation_type, index=index)


def _annotation_primary_title(entry: dict[str, Any], *, index: int) -> str:
    name = str(entry.get("name") or "").strip()
    if name:
        return _truncate_annotation_label(name)
    return _annotation_fallback_title(entry, index=index)


def _annotation_display_text_from_entry(entry: dict[str, Any], *, index: int) -> str:
    if index < 0:
        index = 0
    suffix = "" if bool(entry.get("enabled", True)) else " (off)"
    title = _annotation_primary_title(entry, index=index + 1)
    annotation_type = _annotation_type_label(str(entry.get("type") or "text"))
    return f"{index + 1}: {title} [{annotation_type}]{suffix}"


def _current_error_statistics_mode(
    *,
    analysis_name: str | None,
    error_supported: bool,
    x_bin_width_active: bool,
) -> str | None:
    if not error_supported:
        return None
    normalized = str(analysis_name or "").strip().lower()
    if normalized in {"position", "coordination", "potential"}:
        return "raw_grouped"
    if x_bin_width_active:
        return "saved_rebinned_sample"
    return "direct"


def _inferred_available_error_stats(
    *,
    analysis_name: str | None,
    error_supported: bool,
    x_bin_width_active: bool,
) -> list[str]:
    statistics_mode = _current_error_statistics_mode(
        analysis_name=analysis_name,
        error_supported=error_supported,
        x_bin_width_active=x_bin_width_active,
    )
    if statistics_mode is None:
        return []
    if statistics_mode in {"raw_grouped", "saved_rebinned_sample"}:
        return ["sample_std", "sample_sem"]
    if str(analysis_name or "").strip().lower() == "msd":
        return ["sample_std", "sample_sem"]
    return ["sample_std", "sample_sem", "block_std", "block_sem"]


def _coerce_annotation_for_gui(value: Any, *, index: int) -> dict[str, Any]:
    normalized_type = "text"
    if isinstance(value, dict):
        candidate_type = str(value.get("type") or "text").strip().lower()
        if candidate_type in _ANNOTATION_TYPES:
            normalized_type = candidate_type
    entry = _annotation_defaults_for_gui(normalized_type, index=index)
    if not isinstance(value, dict):
        return entry

    if value.get("id") is not None:
        entry["id"] = _annotation_string(value.get("id")) or entry["id"]
    if value.get("enabled") is not None:
        entry["enabled"] = bool(value.get("enabled"))
    if value.get("name") is not None:
        entry["name"] = _annotation_string(value.get("name")) or entry["name"]
    coord_system = _annotation_string(value.get("coord_system")).lower()
    if coord_system in _ANNOTATION_COORD_SYSTEMS:
        entry["coord_system"] = coord_system
    if value.get("color") is not None:
        entry["color"] = _annotation_string(value.get("color")) or entry["color"]
    for key in (
        "alpha",
        "zorder",
        "text",
        "x",
        "y",
        "font_size",
        "rotation",
        "x1",
        "y1",
        "x2",
        "y2",
        "line_width",
        "mutation_scale",
    ):
        if value.get(key) is not None:
            entry[key] = _annotation_string(value.get(key))
    horizontal = _annotation_string(value.get("horizontal_align")).lower()
    if horizontal in _ANNOTATION_HORIZONTAL_ALIGN:
        entry["horizontal_align"] = horizontal
    vertical = _annotation_string(value.get("vertical_align")).lower()
    if vertical in _ANNOTATION_VERTICAL_ALIGN:
        entry["vertical_align"] = vertical
    line_style = _annotation_string(value.get("line_style"))
    if line_style in _ANNOTATION_LINE_STYLES:
        entry["line_style"] = line_style
    arrow_style = _annotation_string(value.get("arrow_style"))
    if arrow_style in _ANNOTATION_ARROW_STYLES:
        entry["arrow_style"] = arrow_style
    return entry


def _optional_float_list(value: str, *, field_name: str) -> list[float] | None:
    stripped = value.strip()
    if not stripped:
        return None
    tokens = [token for token in re.split(r"[,\s]+", stripped) if token]
    parsed: list[float] = []
    for token in tokens:
        try:
            parsed.append(float(token))
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must contain only float values (comma or space separated)."
            ) from exc
    return parsed


def _format_float_list(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    rendered: list[str] = []
    for item in value:
        if item is None:
            continue
        try:
            rendered.append(f"{float(item):g}")
        except (TypeError, ValueError):
            rendered.append(str(item))
    return ", ".join(rendered)


def _format_float_value(value: Any, *, decimals: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not numeric.is_integer():
        return f"{numeric:.{decimals}f}".rstrip("0").rstrip(".")
    return str(int(numeric))


def _format_json_block(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except TypeError:
        return str(value)


def _optional_json_dict(value: str, *, field_name: str) -> dict[str, Any] | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed


def _resolve_series_line_colors(colors: list[str]) -> list[str] | None:
    """Return explicit per-series colors while preserving blank entries as no-override."""
    normalized = [str(color).strip() for color in colors]
    return normalized if any(normalized) else None


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


def _resolve_series_id_order(series_ids: list[str], requested_order: list[str] | None) -> list[str]:
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


def _partition_series_ids_by_enabled_state(
    series_ids: list[str], enabled_by_id: dict[str, bool]
) -> list[str]:
    enabled_ids = [series_id for series_id in series_ids if enabled_by_id.get(series_id, True)]
    disabled_ids = [series_id for series_id in series_ids if not enabled_by_id.get(series_id, True)]
    return enabled_ids + disabled_ids


def _capture_series_list_view_anchor(
    rows: list[tuple[str, int, int]], *, viewport_height: int, scroll_value: int
) -> dict[str, int | str] | None:
    for row_id, top, bottom in rows:
        if bottom > 0 and top < viewport_height:
            return {
                "row_id": row_id,
                "offset": int(top),
                "scroll_value": int(scroll_value),
            }
    if scroll_value > 0:
        return {
            "row_id": "",
            "offset": 0,
            "scroll_value": int(scroll_value),
        }
    return None


def _restore_series_list_anchor_scroll_value(
    anchor: dict[str, int | str] | None,
    *,
    row_tops: dict[str, int],
    current_scroll_value: int,
    maximum: int,
) -> int | None:
    if anchor is None:
        return None
    fallback = min(max(int(anchor.get("scroll_value", 0)), 0), max(0, int(maximum)))
    row_id = str(anchor.get("row_id") or "").strip()
    if not row_id:
        return fallback
    top = row_tops.get(row_id)
    if top is None:
        return fallback
    offset = int(anchor.get("offset", 0))
    content_top = int(top) + int(current_scroll_value)
    return min(max(content_top - offset, 0), max(0, int(maximum)))


def _format_series_display_text(index: int, label: str, *, enabled: bool) -> str:
    resolved_label = str(label).strip() or f"Series {index + 1}"
    suffix = "" if enabled else " (off)"
    return f"{index + 1}: {resolved_label}{suffix}"


def _coerce_series_descriptors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    descriptors: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            continue
        series_id = str(raw.get("series_id") or f"series:{index}").strip() or f"series:{index}"
        default_label = str(raw.get("default_label") or "").strip()
        descriptors.append(
            {
                "series_id": series_id,
                "default_label": default_label,
                "source_kind": (
                    "group"
                    if str(raw.get("source_kind") or "").strip().lower() == "group"
                    else "source"
                ),
                "source_series_id": str(raw.get("source_series_id") or "").strip() or None,
                "member_series_ids": [
                    str(item).strip()
                    for item in raw.get("member_series_ids", [])
                    if str(item).strip()
                ]
                if isinstance(raw.get("member_series_ids"), (list, tuple))
                else [],
                "group_reducer": (
                    str(raw.get("group_reducer") or "mean").strip().lower() or "mean"
                ),
                "source_name": str(raw.get("source_name") or "").strip(),
                "source_directory": str(raw.get("source_directory") or "").strip(),
                "source_path": str(raw.get("source_path") or "").strip(),
                "source_index": raw.get("source_index"),
                "series_index": raw.get("series_index"),
            }
        )
    return descriptors


def _coerce_series_overrides(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in value.items():
        series_id = str(raw_key).strip()
        if not series_id or not isinstance(raw_value, dict):
            continue
        overrides[series_id] = dict(raw_value)
    return overrides


def _coerce_sync_mode_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    resolved: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if key not in _TRI_STATE_SYNC_FIELD_KEYS:
            continue
        mode = str(raw_value).strip().lower()
        if mode in {"auto", "manual", "off"}:
            resolved[key] = mode
    return resolved


def _fit_defaults_for_gui() -> dict[str, Any]:
    config = default_fit_config()
    config["fit_type"] = "linear"
    config["fit_degree"] = 2
    config["fit_range_mode"] = "visible"
    return config


def _cumulative_defaults_for_gui() -> dict[str, Any]:
    return {
        "enabled": False,
        "label_override": None,
        "show_in_legend": True,
    }


def _coerce_series_cumulative_config(value: Any) -> dict[str, Any]:
    config = _cumulative_defaults_for_gui()
    if not isinstance(value, dict):
        return config
    if "enabled" in value:
        config["enabled"] = bool(value.get("enabled"))
    if value.get("label_override") is not None:
        config["label_override"] = str(value.get("label_override")).strip() or None
    if "show_in_legend" in value:
        config["show_in_legend"] = bool(value.get("show_in_legend"))
    return config


def _coerce_series_fit_config(value: Any) -> dict[str, Any]:
    config = _fit_defaults_for_gui()
    coerced = coerce_fit_config(value)
    config.update(coerced)
    if config.get("fit_type") == "polynomial" and not config.get("fit_degree"):
        config["fit_degree"] = 2
    return config


def _error_defaults_for_gui() -> dict[str, Any]:
    return {
        "enabled": False,
        "stat": "block_sem",
        "style": "band",
        "color": None,
        "label_override": None,
        "show_in_legend": False,
    }


def _coerce_series_error_config(value: Any) -> dict[str, Any]:
    config = _error_defaults_for_gui()
    if not isinstance(value, dict):
        return config
    if "enabled" in value:
        config["enabled"] = bool(value.get("enabled"))
    stat = str(value.get("stat") or "").strip().lower()
    if stat in _ERROR_STATS:
        config["stat"] = stat
    style = str(value.get("style") or "").strip().lower()
    if style in _ERROR_STYLES:
        config["style"] = style
    if value.get("color") is not None:
        config["color"] = str(value.get("color")).strip() or None
    if value.get("label_override") is not None:
        config["label_override"] = str(value.get("label_override")).strip() or None
    if "show_in_legend" in value:
        config["show_in_legend"] = bool(value.get("show_in_legend"))
    return config


def _resolve_error_stat_for_available(
    configured_stat: str | None,
    available_stats: Sequence[str] | None,
) -> str:
    """Return the effective error statistic for one series and available-stat set."""
    configured = str(configured_stat or "").strip().lower()
    available = [
        token
        for token in (str(value).strip().lower() for value in (available_stats or ()))
        if token in _ERROR_STATS
    ]
    if available:
        if configured in available:
            return configured
        for candidate in ("block_sem", "sample_sem", "block_std", "sample_std"):
            if candidate in available:
                return candidate
        return available[0]
    if configured in _ERROR_STATS:
        return configured
    return "sample_sem"


def _default_error_series_label(base_label: str, stat: str) -> str:
    """Return the default derived label for one error-bar child series."""
    resolved_label = str(base_label).strip() or "Series"
    resolved_stat = _resolve_error_stat_for_available(stat, None)
    return f"{resolved_label} {resolved_stat}"


def _error_supported_for_view(
    analysis: str,
    *,
    orientation_heatmap: bool = False,
    position_component: str = "distance",
    coordination_component: str = "distance",
) -> bool:
    """Return whether one GUI view supports 1-D error overlays and min-bin masking."""
    normalized_analysis = str(analysis).strip().lower()
    if normalized_analysis in {"density", "msd", "rdf", "potential"}:
        return True
    if normalized_analysis == "orientation":
        return not bool(orientation_heatmap)
    if normalized_analysis == "position":
        return str(position_component).strip().lower() != "xy-z"
    if normalized_analysis == "coordination":
        return str(coordination_component).strip().lower() != "time-distance"
    return False


@dataclass(frozen=True)
class _LayerInspectorCapabilities:
    plot_family: str
    layer_kind: str
    show_visibility_label: bool
    show_style: bool
    show_markers: bool
    show_derived_lines: bool
    show_uncertainty: bool
    show_normalization: bool
    show_group_members: bool
    show_metadata: bool
    show_fit_editor: bool
    show_cumulative_editor: bool


@dataclass(frozen=True)
class _FigureInspectorCapabilities:
    plot_family: str
    show_legend: bool
    show_lines: bool
    show_heatmap: bool
    show_colorbar: bool


def _plot_family_for_view(
    analysis: str,
    *,
    orientation_heatmap: bool = False,
    position_component: str = "distance",
    coordination_component: str = "distance",
) -> str:
    normalized_analysis = str(analysis).strip().lower()
    if normalized_analysis == "orientation" and orientation_heatmap:
        return "heatmap"
    if normalized_analysis == "position" and str(position_component).strip().lower() == "xy-z":
        return "projection2d"
    if normalized_analysis == "coordination":
        component = str(coordination_component).strip().lower()
        if component == "time-distance":
            return "time-distance"
    return "line"


def _coerce_degree_text(value: Any) -> int | None:
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _derive_synced_field_modes(settings: dict[str, Any]) -> dict[str, str]:
    metadata = _coerce_sync_mode_map(settings.get("_gui_sync_modes"))
    if metadata:
        return metadata
    resolved: dict[str, str] = {}
    for key in _TRI_STATE_SYNC_FIELD_KEYS:
        resolved[key] = "manual" if settings.get(key) is not None else "auto"
    return resolved


def _coerce_profile_filter_options(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _derive_synced_field_locks(settings: dict[str, Any]) -> dict[str, bool]:
    modes = _coerce_sync_mode_map(settings.get("_gui_sync_modes"))
    if modes:
        return {key: mode != "auto" for key, mode in modes.items()}
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


def _derive_warning_messages(
    settings: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> list[str]:
    messages: list[str] = []
    if error:
        messages.append(f"Fix invalid setting: {error}")
    if not isinstance(settings, dict):
        return messages

    normalization_modes = settings.get("series_normalization_modes")
    if isinstance(normalization_modes, (list, tuple)):
        enabled_values = settings.get("series_enabled")
        visible_modes = [
            str(mode).strip().lower()
            for index, mode in enumerate(normalization_modes)
            if not isinstance(enabled_values, (list, tuple))
            or index >= len(enabled_values)
            or bool(enabled_values[index])
        ]
        normalized_count = sum(1 for mode in visible_modes if mode != "none")
        if len(visible_modes) > 1 and 0 < normalized_count < len(visible_modes):
            messages.append(
                "Only part of the plotted series is normalized; compare y-axis values carefully."
            )

    return messages


def _extract_dict_value(settings: dict[str, Any], *, key: str, nested_key: str) -> Any:
    raw = settings.get(key)
    if not isinstance(raw, dict):
        return None
    return raw.get(nested_key)


def _extract_dict_text(settings: dict[str, Any], *, key: str, nested_key: str) -> str:
    value = _extract_dict_value(settings, key=key, nested_key=nested_key)
    if value is None:
        return ""
    return str(value)


def _extract_dict_mode(
    settings: dict[str, Any], *, key: str, nested_key: str, auto_mode: str = "on"
) -> str:
    value = _extract_dict_value(settings, key=key, nested_key=nested_key)
    if isinstance(value, bool):
        return _toggle_to_mode(value, auto_mode=auto_mode)
    return auto_mode


def _figure_filetype_filters() -> tuple[str, str]:
    filetypes: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        try:
            supported = fig.canvas.get_supported_filetypes()
        finally:
            plt.close(fig)
        filetypes = {
            str(ext).strip().lower(): str(label).strip() or f"{str(ext).upper()} file"
            for ext, label in supported.items()
            if str(ext).strip()
        }
    except Exception:
        filetypes = {}

    required = {
        "png": "PNG image",
        "jpg": "JPEG image",
        "jpeg": "JPEG image",
        "svg": "SVG image",
    }
    merged = dict(filetypes)
    for ext, label in required.items():
        merged.setdefault(ext, label)

    if not merged:
        merged = dict(required)

    ordered = dict(sorted(merged.items(), key=lambda item: item[0]))
    extensions = list(ordered.keys())
    all_patterns = " ".join(f"*.{ext}" for ext in extensions)
    filters = [f"All supported ({all_patterns})"]
    for ext, label in ordered.items():
        filters.append(f"{label} (*.{ext})")

    default_name = "linak_plot.png" if "png" in ordered else f"linak_plot.{extensions[0]}"
    return ";;".join(filters), default_name


def _resolve_asset_path(filename: str, *, module_path: Path | None = None) -> Path:
    """Resolve shared asset files from a source tree or an installed distribution."""
    resolved_module_path = (
        module_path.resolve() if module_path is not None else Path(__file__).resolve()
    )
    for parent in resolved_module_path.parents:
        repo_candidate = parent / "assets" / filename
        if repo_candidate.exists():
            return repo_candidate
        installed_candidate = parent / "share" / "linak" / "assets" / filename
        if installed_candidate.exists():
            return installed_candidate
    return resolved_module_path.parent / "assets" / filename


def _default_gui_artwork_path() -> Path:
    return _resolve_asset_path("logo_simple.svg")


def _extract_limit(
    settings: dict[str, Any],
    *,
    key: str,
    index: int,
) -> str:
    raw = settings.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return ""
    value = raw[index]
    if value is None:
        return ""
    return _format_float_value(value)


def _extract_figsize_dimension(
    settings: dict[str, Any],
    *,
    index: int,
    fallback: float,
) -> str:
    raw = settings.get("figsize")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        value = raw[index]
        if value is not None:
            return str(value)
    return str(fallback)


def launch_plot_settings_panel(
    *,
    title: str,
    initial_settings: dict[str, Any],
    on_preview: Callable[[dict[str, Any]], dict[str, Any] | None],
    on_save: Callable[[str, dict[str, Any]], str],
    on_save_figure: Callable[[dict[str, Any], str], str | tuple[str, dict[str, Any]]] | None = None,
    on_import_hdf5: Callable[[str, str | None], dict[str, Any]] | None = None,
    on_list_import_hdf5_profiles: Callable[[str], dict[str, Any]] | None = None,
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
    """Open a PySide6 panel that previews and persists plot settings."""
    try:
        from PySide6.QtCore import QEvent, QTimer, Qt
        from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QCheckBox,
            QComboBox,
            QColorDialog,
            QFrame,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSplitter,
            QStackedWidget,
            QTabWidget,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

        try:
            from PySide6.QtSvg import QSvgRenderer
        except Exception:  # pragma: no cover - optional Qt module
            QSvgRenderer = None
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PySide6 is unavailable; cannot open GUI plot controls. "
            "Install PySide6 or use CLI plot flags."
        ) from exc

    defaults = DEFAULT_PLOT_STYLE

    class _ScrollSafeComboBox(QComboBox):
        def wheelEvent(self, event: Any) -> None:
            if self.view().isVisible():
                super().wheelEvent(event)
                return
            event.ignore()

    class _SeriesGripWidget(QWidget):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFixedSize(12, 16)

        def paintEvent(self, event: Any) -> None:  # pragma: no cover - UI paint
            super().paintEvent(event)
            painter = QPainter(self)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen = QPen(self.palette().color(QPalette.ColorRole.Mid))
                pen.setWidth(2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                for y_pos in (4, 8, 12):
                    painter.drawLine(2, y_pos, 10, y_pos)
            finally:
                painter.end()

    class _SeriesRowWidget(QWidget):
        def __init__(
            self,
            *,
            on_select: Callable[[], None],
            on_toggle: Callable[[bool], None],
            on_move_up: Callable[[], None],
            on_move_down: Callable[[], None],
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._on_select = on_select
            self._on_toggle = on_toggle
            self._on_move_up = on_move_up
            self._on_move_down = on_move_down
            self.setObjectName("seriesRowWidget")

            layout = QHBoxLayout(self)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(8)

            self.checkbox = QCheckBox(self)
            self.checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.checkbox.toggled.connect(self._on_toggle)
            layout.addWidget(self.checkbox)

            self.color_swatch = QFrame(self)
            self.color_swatch.setObjectName("seriesRowSwatch")
            self.color_swatch.setFixedSize(12, 12)
            layout.addWidget(self.color_swatch)

            self.text_label = QLabel(self)
            self.text_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            layout.addWidget(self.text_label, stretch=1)

            self.move_up_button = QToolButton(self)
            self.move_up_button.setObjectName("seriesRowButton")
            self.move_up_button.setText("▴")
            self.move_up_button.setAutoRaise(True)
            self.move_up_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.move_up_button.setFixedSize(22, 22)
            self.move_up_button.setArrowType(Qt.ArrowType.UpArrow)
            self.move_up_button.setText("")
            self.move_up_button.clicked.connect(self._handle_move_up_clicked)
            layout.addWidget(self.move_up_button)

            self.move_down_button = QToolButton(self)
            self.move_down_button.setObjectName("seriesRowButton")
            self.move_down_button.setText("▾")
            self.move_down_button.setAutoRaise(True)
            self.move_down_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.move_down_button.setFixedSize(22, 22)
            self.move_down_button.setArrowType(Qt.ArrowType.DownArrow)
            self.move_down_button.setText("")
            self.move_down_button.clicked.connect(self._handle_move_down_clicked)
            layout.addWidget(self.move_down_button)

            self.grip_widget = _SeriesGripWidget(self)
            layout.addWidget(self.grip_widget)

            for target in (self, self.text_label, self.grip_widget, self.color_swatch):
                target.installEventFilter(self)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # pragma: no cover - UI flow
            if watched in {
                self,
                self.text_label,
                self.grip_widget,
                self.color_swatch,
            } and event.type() in {
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonDblClick,
            }:
                QTimer.singleShot(0, self._on_select)
            return super().eventFilter(watched, event)

        def _handle_move_up_clicked(self) -> None:
            self._on_select()
            self._on_move_up()

        def _handle_move_down_clicked(self) -> None:
            self._on_select()
            self._on_move_down()

        def update_content(
            self,
            *,
            text: str,
            checked: bool,
            enabled: bool,
            selected: bool,
            color_token: str,
            kind: str,
            can_move_up: bool,
            can_move_down: bool,
            tooltip_text: str,
        ) -> None:
            self.checkbox.blockSignals(True)
            try:
                self.checkbox.setChecked(checked)
            finally:
                self.checkbox.blockSignals(False)
            self.text_label.setText(text)
            self.setToolTip(tooltip_text)
            self.text_label.setToolTip(tooltip_text)

            if not color_token:
                self.color_swatch.setVisible(False)
            else:
                self.color_swatch.setVisible(True)
                swatch_color = QColor(color_token)
                if swatch_color.isValid():
                    self.color_swatch.setStyleSheet(
                        f"background-color: {swatch_color.name()}; border: none; border-radius: 6px;"
                    )
                else:
                    border = self.palette().color(QPalette.ColorRole.Mid).name()
                    background = self.palette().color(QPalette.ColorRole.Base).name()
                    self.color_swatch.setStyleSheet(
                        "background-color: "
                        f"{background}; border: 1px solid {border}; border-radius: 6px;"
                    )

            text_color = self.palette().color(QPalette.ColorRole.Text)
            if not enabled:
                text_color.setAlpha(150)
            background = "transparent"
            border = "transparent"
            if selected:
                background = self.palette().color(QPalette.ColorRole.Highlight).lighter(160).name()
                border = self.palette().color(QPalette.ColorRole.Highlight).name()
            self.setStyleSheet(
                "QWidget#seriesRowWidget {"
                f"background-color: {background};"
                f"border: 1px solid {border};"
                "border-radius: 8px;"
                "}"
                "QToolButton#seriesRowButton {"
                "padding: 0px;"
                "margin: 0px;"
                "border: none;"
                "background: transparent;"
                f"color: {text_color.name()};"
                "}"
                "QToolButton#seriesRowButton:hover {"
                "border-radius: 6px;"
                f"background-color: {self.palette().color(QPalette.ColorRole.Button).name()};"
                "}"
                "QToolButton#seriesRowButton:disabled {"
                f"color: {self.palette().color(QPalette.ColorRole.Mid).name()};"
                "}"
            )

            label_font_style = "italic" if not enabled else "normal"
            label_font_weight = (
                "700" if kind == "fit" else "600" if kind in {"error", "cumulative"} else "400"
            )
            self.text_label.setStyleSheet(
                "border: none;"
                f"color: {text_color.name()};"
                f"font-style: {label_font_style};"
                f"font-weight: {label_font_weight};"
            )

            control_enabled = kind == "base"
            self.move_up_button.setEnabled(control_enabled and can_move_up)
            self.move_down_button.setEnabled(control_enabled and can_move_down)
            self.move_up_button.setVisible(control_enabled)
            self.move_down_button.setVisible(control_enabled)
            self.grip_widget.setVisible(control_enabled)

    class _PlotSettingsWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(title)
            self.resize(980, 760)
            self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
            self._analysis_name = (analysis_name or "").strip().lower() or None
            self._allow_named_profiles = bool(allow_named_profiles)
            self._on_resolve_series_defaults = on_resolve_series_defaults
            self._saved_signature: str | None = None
            self._profile_filter_options = _coerce_profile_filter_options(
                initial_settings.get("_profile_filter_options")
            )
            normalized_profile_names: list[str] = []
            for raw_name in available_profile_names or []:
                candidate = str(raw_name).strip()
                if candidate and candidate not in normalized_profile_names:
                    normalized_profile_names.append(candidate)
            requested_profile_name = str(initial_profile_name or "").strip()
            if requested_profile_name and requested_profile_name not in normalized_profile_names:
                normalized_profile_names.insert(0, requested_profile_name)
            if not normalized_profile_names:
                normalized_profile_names = ["Default"]
            self._profile_names = normalized_profile_names
            self._current_profile_name = requested_profile_name or self._profile_names[0]
            self._default_profile_settings = dict(
                default_profile_settings
                if isinstance(default_profile_settings, dict)
                else initial_settings
            )
            self._profile_selector_syncing = False
            self._title_rows: list[tuple[QFormLayout, QWidget]] = []
            self._title_detail_widgets: list[QWidget] = []
            self._x_label_detail_widgets: list[QWidget] = []
            self._y_label_detail_widgets: list[QWidget] = []
            self._legend_rows: list[tuple[QFormLayout, QWidget]] = []
            self._ticks_rows: list[tuple[QFormLayout, QWidget]] = []
            self._grid_rows: list[tuple[QFormLayout, QWidget]] = []
            self._marker_rows: list[tuple[QFormLayout, QWidget]] = []
            self._colorbar_rows: list[tuple[QFormLayout, QWidget]] = []
            self._border_custom_rows: list[tuple[QFormLayout, QWidget]] = []
            self._x_bin_reducer_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_value_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_x_ref_row: tuple[QFormLayout, QWidget] | None = None
            self._position_map_color_row: tuple[QFormLayout, QWidget] | None = None
            self._position_time_axis_row: tuple[QFormLayout, QWidget] | None = None
            self._coordination_time_axis_row: tuple[QFormLayout, QWidget] | None = None
            self._axes_ticks_group: QGroupBox | None = None
            self._tick_appearance_group: QGroupBox | None = None
            self._grid_group: QGroupBox | None = None
            self._data_transform_group: QGroupBox | None = None
            self._normalization_group: QGroupBox | None = None
            self._series_syncing = False
            self._series_active_index = 0
            self._series_natural_order_data: list[str] = []
            self._series_descriptors_data: list[dict[str, Any]] = []
            self._series_labels_data: list[str] = []
            self._series_label_overrides_data: list[str] = []
            self._series_colors_data: list[str] = []
            self._series_enabled_data: list[bool] = []
            self._series_show_in_legend_data: list[bool] = []
            self._series_alpha_data: list[str] = []
            self._series_error_enabled_data: list[bool] = []
            self._series_error_stats_data: list[str] = []
            self._series_error_styles_data: list[str] = []
            self._series_error_colors_data: list[str] = []
            self._series_error_label_overrides_data: list[str] = []
            self._series_error_show_in_legend_data: list[bool] = []
            self._series_fit_enabled_data: list[bool] = []
            self._series_fit_label_overrides_data: list[str] = []
            self._series_fit_show_in_legend_data: list[bool] = []
            self._series_fit_types_data: list[str] = []
            self._series_fit_degrees_data: list[str] = []
            self._series_fit_range_modes_data: list[str] = []
            self._series_fit_x_mins_data: list[str] = []
            self._series_fit_x_maxs_data: list[str] = []
            self._series_cumulative_enabled_data: list[bool] = []
            self._series_cumulative_label_overrides_data: list[str] = []
            self._series_cumulative_show_in_legend_data: list[bool] = []
            self._series_line_widths_data: list[str] = []
            self._series_markers_data: list[str] = []
            self._series_line_kwargs_data: list[str] = []
            self._normalization_syncing = False
            self._series_normalization_modes_data: list[str] = []
            self._series_normalization_values_data: list[str] = []
            self._series_normalization_x_refs_data: list[str] = []
            self._annotations_data: list[dict[str, Any]] = []
            self._annotation_active_index = 0
            self._annotation_syncing = False
            self._series_active_is_error_child = False
            self._series_active_is_fit_child = False
            self._series_active_is_cumulative_child = False
            self._series_display_rows: list[dict[str, Any]] = []
            self._last_preview_state: dict[str, Any] = {}
            self._synced_field_locks: dict[str, bool] = {}
            self._synced_field_modes: dict[str, str] = {}
            self._advanced_json_syncing = False
            self._suspend_preview_events = False
            self._preview_pixmap: QPixmap | None = None
            self._preview_zoom_factor = 1.0
            self._preview_frame: QFrame | None = None
            self._preview_scroll: QScrollArea | None = None
            self._preview_label: QLabel | None = None
            self._nav_list: QListWidget | None = None
            self._page_stack: QStackedWidget | None = None
            self._page_title_label: QLabel | None = None
            self._page_note_label: QLabel | None = None
            self._layers_tabs: QTabWidget | None = None
            self._header_state_label: QLabel | None = None
            self._warning_summary_label: QLabel | None = None
            self._status_source_badge: QLabel | None = None
            self._status_mode_badge: QLabel | None = None
            self._status_layers_badge: QLabel | None = None
            self._status_preview_badge: QLabel | None = None
            self._status_profile_badge: QLabel | None = None
            self._status_warning_badge: QLabel | None = None
            self._potential_summary_x_axis_label: QLabel | None = None
            self._potential_summary_total_rows_label: QLabel | None = None
            self._potential_summary_complete_rows_label: QLabel | None = None
            self._potential_summary_incomplete_rows_label: QLabel | None = None
            self._series_meta_default_label: QLabel | None = None
            self._series_meta_source_name: QLabel | None = None
            self._series_meta_source_dir: QLabel | None = None
            self._series_meta_series_id: QLabel | None = None
            self._series_stats_label: QLabel | None = None
            self._series_fit_mode: QComboBox | None = None
            self._series_fit_summary: QLabel | None = None
            self._series_fit_warning: QLabel | None = None
            self._series_fit_style_note: QLabel | None = None
            self._series_fit_summary_group: QGroupBox | None = None
            self._series_visibility_group: QGroupBox | None = None
            self._series_style_group: QGroupBox | None = None
            self._series_uncertainty_group: QGroupBox | None = None
            self._series_derived_group: QGroupBox | None = None
            self._series_metadata_group: QGroupBox | None = None
            self._series_fit_group: QGroupBox | None = None
            self._series_cumulative_group: QGroupBox | None = None
            self._series_error_detail_rows: list[tuple[QFormLayout, QWidget]] = []
            self._series_error_detail_widgets: list[QWidget] = []
            self._series_fit_detail_rows: list[tuple[QFormLayout, QWidget]] = []
            self._series_fit_detail_widgets: list[QWidget] = []
            self._series_cumulative_detail_rows: list[tuple[QFormLayout, QWidget]] = []
            self._series_cumulative_detail_widgets: list[QWidget] = []
            self._normalization_copy_button: QPushButton | None = None
            self._normalization_actions_widget: QWidget | None = None
            self._normalization_hint_label: QLabel | None = None
            self._annotation_common_detail_rows: list[tuple[QFormLayout, QWidget]] = []
            self._annotation_summary_group: QGroupBox | None = None
            self._figure_tabs: QTabWidget | None = None
            self._figure_legend_section: QWidget | None = None
            self._figure_lines_group: QGroupBox | None = None
            self._figure_heatmap_group: QGroupBox | None = None
            self._figure_colorbar_group: QGroupBox | None = None
            self._y_bin_width_row: tuple[QFormLayout, QWidget] | None = None
            self._y_bin_reducer_row: tuple[QFormLayout, QWidget] | None = None
            self._min_bin_points_row: tuple[QFormLayout, QWidget] | None = None
            self._tooltip_disabled_reasons: dict[int, str | None] = {}
            self._gui_artwork_path = _default_gui_artwork_path()
            self._figure_save_filters, self._figure_default_name = _figure_filetype_filters()
            self._preview_image_path = (
                Path(tempfile.gettempdir()) / f"linak_preview_{uuid4().hex}.png"
            )
            self._preview_timer = QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.timeout.connect(self._handle_debounced_preview)
            self._preview_loading = False
            self._preview_error: str | None = None
            self._status_label = QLabel("Ready.")
            self._build_ui()
            self._suspend_preview_events = True
            self._populate(initial_settings)
            self._suspend_preview_events = False
            self._refresh_widget_states()
            self._bind_live_preview_signals()
            try:
                self._saved_signature = self._signature(self._collect_settings())
            except Exception:
                self._saved_signature = None
            self._update_embedded_preview(interactive=False)

        def _signature(self, settings: dict[str, Any]) -> str:
            return json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)

        def _configure_horizontal_growth(self, widget: QWidget) -> QWidget:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                widget.sizePolicy().verticalPolicy(),
            )
            return widget

        def _combo(
            self,
            values: tuple[str, ...],
            *,
            editable: bool = False,
            minimum_contents_length: int = 12,
        ) -> QComboBox:
            widget = _ScrollSafeComboBox()
            widget.addItems(list(values))
            widget.setEditable(editable)
            widget.setMinimumContentsLength(max(8, minimum_contents_length))
            widget.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._configure_horizontal_growth(widget)
            return widget

        def _line(self, placeholder: str = "") -> QLineEdit:
            widget = QLineEdit()
            if placeholder:
                widget.setPlaceholderText(placeholder)
            self._configure_horizontal_growth(widget)
            return widget

        def _register_tooltip(self, widget: QWidget, tooltip_id: str | None) -> None:
            if tooltip_id is None:
                return
            widget.setProperty("tooltipId", tooltip_id)

        def _base_tooltip(self, tooltip_id: str | None) -> str:
            if tooltip_id is None:
                return ""
            return _TOOLTIPS.get(tooltip_id, "")

        def _apply_widget_tooltip(
            self,
            widget: QWidget | None,
            *,
            tooltip_id: str | None = None,
            disabled_reason: str | None = None,
        ) -> None:
            if widget is None:
                return
            resolved_id = tooltip_id or str(widget.property("tooltipId") or "").strip() or None
            if resolved_id is None:
                return
            base = self._base_tooltip(resolved_id).strip()
            reason = str(disabled_reason or "").strip()
            text = base
            if reason:
                text = f"{base} Disabled because {reason}" if base else f"Disabled because {reason}"
            if text:
                escaped = html.escape(text).replace("\n", "<br/>")
                text = (
                    f"<qt><div style='max-width: 320px; white-space: normal;'>{escaped}</div></qt>"
                )
            widget.setToolTip(text)
            self._tooltip_disabled_reasons[id(widget)] = reason or None

        def _sync_mode_widget(self, *, allow_off: bool = False) -> QComboBox:
            widget = self._combo(_TEXT_SYNC_MODES if allow_off else _SYNC_MODES)
            self._register_tooltip(widget, "shared.sync_mode")
            self._apply_widget_tooltip(widget)
            return widget

        def _lockable_line(
            self,
            *,
            placeholder: str = "",
            allow_off: bool = False,
        ) -> tuple[QWidget, QLineEdit, QComboBox]:
            container = QWidget()
            self._configure_horizontal_growth(container)
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            field = self._line(placeholder)
            lock = self._sync_mode_widget(allow_off=allow_off)
            row.addWidget(field, stretch=1)
            row.addWidget(lock)
            return container, field, lock

        def _lockable_pair(
            self,
            *,
            first_placeholder: str = "",
            second_placeholder: str = "",
        ) -> tuple[QWidget, QLineEdit, QLineEdit, QComboBox]:
            container = QWidget()
            self._configure_horizontal_growth(container)
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            first = self._line(first_placeholder)
            second = self._line(second_placeholder)
            lock = self._sync_mode_widget()
            row.addWidget(first, stretch=1)
            row.addWidget(second, stretch=1)
            row.addWidget(lock)
            return container, first, second, lock

        def _color_field(
            self,
            *,
            placeholder: str = "",
            tooltip_id: str | None = None,
        ) -> tuple[QWidget, QLineEdit]:
            container = QWidget()
            self._configure_horizontal_growth(container)
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            field = self._line(placeholder)
            pick_button = QPushButton("Pick")
            self._register_tooltip(field, tooltip_id)
            self._register_tooltip(pick_button, "shared.color_picker")

            def _pick_color() -> None:
                initial = QColor(field.text().strip())
                if not initial.isValid():
                    initial = QColor("white")
                selected = QColorDialog.getColor(initial, self, "Select color")
                if not selected.isValid():
                    return
                field.setText(str(selected.name()))

            pick_button.clicked.connect(_pick_color)
            self._apply_widget_tooltip(field, tooltip_id=tooltip_id)
            self._apply_widget_tooltip(pick_button)
            row.addWidget(field, stretch=1)
            row.addWidget(pick_button)
            return container, field

        def _set_form_row_visible(
            self,
            form: QFormLayout,
            field: QWidget,
            visible: bool,
        ) -> None:
            try:
                form.setRowVisible(field, visible)
            except Exception:
                label = form.labelForField(field)
                if label is not None:
                    label.setVisible(visible)
                field.setVisible(visible)

        def _set_rows_visible(
            self,
            rows: list[tuple[QFormLayout, QWidget]],
            visible: bool,
        ) -> None:
            for form, field in rows:
                self._set_form_row_visible(form, field, visible)

        def _set_form_row_enabled(
            self,
            form: QFormLayout,
            field: QWidget,
            enabled: bool,
            *,
            disabled_reason: str | None = None,
        ) -> None:
            label = form.labelForField(field)
            if label is not None:
                label.setEnabled(enabled)
                self._apply_widget_tooltip(
                    label,
                    disabled_reason=None if enabled else disabled_reason,
                )
            field.setEnabled(enabled)
            self._apply_widget_tooltip(
                field,
                disabled_reason=None if enabled else disabled_reason,
            )

        def _set_rows_enabled(
            self,
            rows: list[tuple[QFormLayout, QWidget]],
            enabled: bool,
            *,
            disabled_reason: str | None = None,
        ) -> None:
            for form, field in rows:
                self._set_form_row_enabled(
                    form,
                    field,
                    enabled,
                    disabled_reason=disabled_reason,
                )

        def _add_form_row(
            self,
            form: QFormLayout,
            label: str,
            field: QWidget,
            *,
            tooltip_id: str | None = None,
        ) -> None:
            if label:
                label_widget = QLabel(label)
                if tooltip_id is not None:
                    self._register_tooltip(label_widget, tooltip_id)
                    self._register_tooltip(field, tooltip_id)
                    self._apply_widget_tooltip(label_widget)
                    self._apply_widget_tooltip(field)
                form.addRow(label_widget, field)
            else:
                if tooltip_id is not None:
                    self._register_tooltip(field, tooltip_id)
                    self._apply_widget_tooltip(field)
                form.addRow(label, field)

        def _make_scrollable_tab(self, tab: QWidget) -> QWidget:
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea(tab)
            if tab.objectName() == "plotSubtabPage":
                scroll.setObjectName("plotSubtabScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMinimumWidth(0)
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            if scroll.viewport() is not None and tab.objectName() == "plotSubtabPage":
                scroll.viewport().setObjectName("plotSubtabViewport")
            content = QWidget(scroll)
            if tab.objectName() == "plotSubtabPage":
                content.setObjectName("plotSubtabContent")
            content.setMinimumWidth(0)
            content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            scroll.setWidget(content)
            tab_layout.addWidget(scroll)
            return content

        def _set_synced_field_lock(self, key: str, locked: bool) -> None:
            mode_widget = getattr(self, f"_{key}_lock", None)
            if mode_widget is None:
                return
            self._synced_field_locks[key] = bool(locked)
            if key in _TRI_STATE_SYNC_FIELD_KEYS:
                self._synced_field_modes[key] = "manual" if bool(locked) else "auto"
            mode_widget.blockSignals(True)
            try:
                self._set_combo_value(mode_widget, _lock_to_sync_mode(locked))
            finally:
                mode_widget.blockSignals(False)

        def _synced_field_mode(self, key: str) -> str:
            if key in _TRI_STATE_SYNC_FIELD_KEYS:
                token = str(self._synced_field_modes.get(key, "auto")).strip().lower()
                if token in {"auto", "manual", "off"}:
                    return token
                return "auto"
            return "manual" if self._synced_field_locks.get(key, False) else "auto"

        def _set_synced_field_mode(self, key: str, mode: str) -> None:
            normalized = str(mode).strip().lower()
            if key not in _TRI_STATE_SYNC_FIELD_KEYS:
                self._set_synced_field_lock(key, normalized == "manual")
                return
            if normalized not in {"auto", "manual", "off"}:
                normalized = "auto"
            self._synced_field_modes[key] = normalized
            self._synced_field_locks[key] = normalized == "manual"
            mode_widget = getattr(self, f"_{key}_lock", None)
            if mode_widget is not None:
                mode_widget.blockSignals(True)
                try:
                    self._set_combo_value(mode_widget, normalized.title())
                finally:
                    mode_widget.blockSignals(False)

        def _connect_lockable_line(
            self,
            key: str,
            field: QLineEdit,
            lock: QComboBox,
            *,
            allow_off: bool = False,
        ) -> None:
            setattr(self, f"_{key}_lock", lock)
            field.textEdited.connect(
                lambda _text, sync_key=key: self._handle_synced_field_edit(sync_key)
            )
            if allow_off:
                lock.currentTextChanged.connect(
                    lambda value, sync_key=key: self._handle_synced_field_mode_changed(
                        sync_key, value
                    )
                )
            else:
                lock.currentTextChanged.connect(
                    lambda value, sync_key=key: self._handle_synced_field_lock_toggled(
                        sync_key, _sync_mode_to_lock(value)
                    )
                )

        def _apply_preview_state_to_synced_fields(self, settings: dict[str, Any]) -> None:
            self._last_preview_state = dict(settings)
            self._suspend_preview_events = True
            try:
                if self._synced_field_mode("title") == "auto":
                    self.title_text.setText(str(settings.get("title") or ""))
                if self._synced_field_mode("x_label") == "auto":
                    self.x_label.setText(str(settings.get("x_label") or ""))
                if self._synced_field_mode("y_label") == "auto":
                    self.y_label.setText(str(settings.get("y_label") or ""))
                if not self._synced_field_locks.get("x_lim", False):
                    self.x_min.setText(_extract_limit(settings, key="x_lim", index=0))
                    self.x_max.setText(_extract_limit(settings, key="x_lim", index=1))
                if not self._synced_field_locks.get("y_lim", False):
                    self.y_min.setText(_extract_limit(settings, key="y_lim", index=0))
                    self.y_max.setText(_extract_limit(settings, key="y_lim", index=1))
                if not self._synced_field_locks.get("x_ticks", False):
                    self.x_ticks.setText(_format_float_list(settings.get("x_ticks")))
                if not self._synced_field_locks.get("y_ticks", False):
                    self.y_ticks.setText(_format_float_list(settings.get("y_ticks")))
                if not self._synced_field_locks.get("x_label_pad", False):
                    x_label_pad = settings.get("x_label_pad")
                    self.x_label_pad.setText("" if x_label_pad is None else str(x_label_pad))
                if not self._synced_field_locks.get("y_label_pad", False):
                    y_label_pad = settings.get("y_label_pad")
                    self.y_label_pad.setText("" if y_label_pad is None else str(y_label_pad))
            finally:
                self._suspend_preview_events = False
            if 0 <= self._series_active_index < len(self._series_labels_data):
                self._refresh_error_stat_choices(self._series_active_index)
            self._update_potential_summary_panel(settings)
            self._update_series_metadata_panel(self._series_active_index)
            self._update_series_error_summary(self._series_active_index)
            self._update_series_fit_summary(self._series_active_index)
            self._update_annotation_preview_summary(self._annotation_active_index)

        def _handle_synced_field_edit(self, key: str) -> None:
            if key in _TRI_STATE_SYNC_FIELD_KEYS:
                self._set_synced_field_mode(key, "manual")
            else:
                self._set_synced_field_lock(key, True)

        def _handle_synced_field_lock_toggled(self, key: str, checked: bool) -> None:
            self._synced_field_locks[key] = bool(checked)
            if checked:
                self._schedule_preview_update()
                return
            self._apply_preview_state_to_synced_fields(self._last_preview_state)
            self._schedule_preview_update()

        def _handle_synced_field_mode_changed(self, key: str, value: str) -> None:
            normalized = str(value).strip().lower()
            if normalized not in {"auto", "manual", "off"}:
                normalized = "auto"
            self._set_synced_field_mode(key, normalized)
            if normalized == "auto":
                self._apply_preview_state_to_synced_fields(self._last_preview_state)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _build_rasterized_window_icon(self) -> QIcon | None:
            if not self._gui_artwork_path.exists():
                return None
            if self._gui_artwork_path.suffix.lower() != ".svg" or QSvgRenderer is None:
                icon = QIcon(str(self._gui_artwork_path))
                return None if icon.isNull() else icon
            renderer = QSvgRenderer(str(self._gui_artwork_path))
            if not renderer.isValid():
                fallback = QIcon(str(self._gui_artwork_path))
                return None if fallback.isNull() else fallback

            icon = QIcon()
            for size in (16, 20, 24, 32, 48, 64, 128, 256):
                pixmap = QPixmap(size, size)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                try:
                    renderer.render(painter)
                finally:
                    painter.end()
                icon.addPixmap(pixmap)
            return None if icon.isNull() else icon

        def _apply_window_icon(self) -> None:
            icon = self._build_rasterized_window_icon()
            if icon is None:
                return
            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)

        @staticmethod
        def _normalize_profile_name(name: str) -> str:
            normalized = str(name).strip()
            if not normalized:
                raise ValueError("Profile name cannot be empty.")
            return normalized

        def _profile_name_exists(self, name: str) -> bool:
            lowered = name.casefold()
            return any(existing.casefold() == lowered for existing in self._profile_names)

        def _sync_profile_selector(self) -> None:
            self._profile_selector_syncing = True
            try:
                self._profile_selector.clear()
                for name in self._profile_names:
                    self._profile_selector.addItem(name)
                if self._current_profile_name in self._profile_names:
                    self._profile_selector.setCurrentText(self._current_profile_name)
                elif self._profile_names:
                    self._current_profile_name = self._profile_names[0]
                    self._profile_selector.setCurrentText(self._current_profile_name)
            finally:
                self._profile_selector_syncing = False
            delete_button = getattr(self, "_profile_delete_button", None)
            if delete_button is not None:
                delete_button.setEnabled(
                    self._allow_named_profiles
                    and len(self._profile_names) > 1
                    and on_delete_profile is not None
                )

        def _resolved_default_profile_settings(self) -> dict[str, Any]:
            settings = _without_new_profile_series_overrides(self._default_profile_settings)
            if self._on_resolve_series_defaults is not None:
                try:
                    resolved = self._on_resolve_series_defaults(settings)
                except Exception:
                    resolved = None
                if isinstance(resolved, dict):
                    if "series_count" in resolved:
                        settings["series_count"] = resolved["series_count"]
                    if "series_labels" in resolved:
                        settings["series_labels"] = list(resolved["series_labels"])
                    if "series_descriptors" in resolved:
                        settings["series_descriptors"] = list(resolved["series_descriptors"])
            return settings

        def _load_settings_into_editor(
            self,
            settings: dict[str, Any],
            *,
            profile_name: str | None = None,
            status_message: str | None = None,
            mark_saved: bool,
        ) -> None:
            if profile_name is not None:
                self._current_profile_name = profile_name
                self._sync_profile_selector()
            self._suspend_preview_events = True
            try:
                self._populate(settings)
            finally:
                self._suspend_preview_events = False
            self._refresh_widget_states()
            if status_message:
                self._status_label.setText(status_message)
            if mark_saved:
                self._saved_signature = self._signature(self._collect_settings())
            self._refresh_shell_state()

        def _save_current_profile(self, *, status_prefix: str | None = None) -> bool:
            try:
                settings = self._collect_settings()
                message = on_save(self._current_profile_name, settings)
                self._status_label.setText(
                    message if status_prefix is None else f"{status_prefix}{message}"
                )
                self._saved_signature = self._signature(settings)
                self._refresh_shell_state()
                return True
            except Exception as exc:
                self._report_error("Save failed", exc)
                return False

        def _confirm_context_change(self, *, action_label: str) -> bool:
            try:
                settings = self._collect_settings()
                current_signature = self._signature(settings)
            except Exception as exc:
                decision = QMessageBox.question(
                    self,
                    "Invalid settings",
                    "Current profile contains invalid values "
                    f"({exc}). Continue without saving before {action_label}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                return decision == QMessageBox.StandardButton.Yes

            if self._saved_signature == current_signature:
                return True

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Unsaved profile changes")
            box.setText(
                f"Save changes to profile '{self._current_profile_name}' before {action_label}?"
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Save)
            decision = box.exec()
            if decision == QMessageBox.StandardButton.Cancel:
                return False
            if decision == QMessageBox.StandardButton.Save:
                return self._save_current_profile()
            return True

        def _prompt_profile_name(self, *, title_text: str, default_value: str) -> str | None:
            raw_value, accepted = QInputDialog.getText(
                self,
                title_text,
                "Profile name",
                text=default_value,
            )
            if not accepted:
                self._status_label.setText(f"{title_text} canceled.")
                return None
            name = self._normalize_profile_name(raw_value)
            if self._profile_name_exists(name):
                raise ValueError(f"A profile named '{name}' already exists.")
            return name

        def _next_duplicate_profile_name(self) -> str:
            base = f"{self._current_profile_name} Copy"
            if not self._profile_name_exists(base):
                return base
            for index in range(2, 1000):
                candidate = f"{base} {index}"
                if not self._profile_name_exists(candidate):
                    return candidate
            raise ValueError("Could not find an available duplicate profile name.")

        def _handle_profile_selection_request(self, index: int) -> None:
            if not self._allow_named_profiles:
                return
            if self._profile_selector_syncing or index < 0:
                return
            requested = self._profile_selector.itemText(index).strip()
            if not requested or requested == self._current_profile_name:
                return
            if not self._confirm_context_change(action_label=f"switching to '{requested}'"):
                self._sync_profile_selector()
                return
            try:
                if on_load_profile is None:
                    raise ValueError("Profile loading is unavailable for this plot session.")
                loaded = on_load_profile(requested)
                message = f"Loaded profile '{requested}'."
                if on_set_active_profile is not None:
                    message = on_set_active_profile(requested)
                self._load_settings_into_editor(
                    loaded,
                    profile_name=requested,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._sync_profile_selector()
                self._report_error("Load profile failed", exc)

        def _handle_new_profile(self) -> None:
            if not self._allow_named_profiles:
                self._status_label.setText("Combined plot files use one saved settings document.")
                self._refresh_shell_state()
                return
            if not self._confirm_context_change(action_label="creating a new profile"):
                return
            try:
                name = self._prompt_profile_name(
                    title_text="Create profile",
                    default_value="New Profile",
                )
                if name is None:
                    return
                settings = self._resolved_default_profile_settings()
                message = on_save(name, settings)
                if on_set_active_profile is not None:
                    message = on_set_active_profile(name)
                self._profile_names.append(name)
                self._load_settings_into_editor(
                    settings,
                    profile_name=name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Create profile failed", exc)

        def _handle_duplicate_profile(self) -> None:
            if not self._allow_named_profiles:
                self._status_label.setText("Combined plot files use one saved settings document.")
                self._refresh_shell_state()
                return
            try:
                settings = self._collect_settings()
                name = self._prompt_profile_name(
                    title_text="Duplicate profile",
                    default_value=self._next_duplicate_profile_name(),
                )
                if name is None:
                    return
                message = on_save(name, settings)
                if on_set_active_profile is not None:
                    message = on_set_active_profile(name)
                self._profile_names.append(name)
                self._load_settings_into_editor(
                    settings,
                    profile_name=name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Duplicate profile failed", exc)

        def _handle_rename_profile(self) -> None:
            if not self._allow_named_profiles:
                self._status_label.setText("Combined plot files use one saved settings document.")
                self._refresh_shell_state()
                return
            if on_delete_profile is None:
                self._status_label.setText("Rename profile is unavailable.")
                return
            try:
                current_name = self._current_profile_name
                settings = self._collect_settings()
                raw_value, accepted = QInputDialog.getText(
                    self,
                    "Rename profile",
                    "Profile name",
                    text=current_name,
                )
                if not accepted:
                    self._status_label.setText("Rename profile canceled.")
                    return
                name = self._normalize_profile_name(raw_value)
                if name == current_name:
                    self._status_label.setText("Profile name unchanged.")
                    return
                if self._profile_name_exists(name):
                    raise ValueError(f"A profile named '{name}' already exists.")
                message = on_save(name, settings)
                if on_set_active_profile is not None:
                    on_set_active_profile(name)
                on_delete_profile(current_name)
                self._profile_names = [
                    name if entry == current_name else entry for entry in self._profile_names
                ]
                message = f"Renamed profile '{current_name}' to '{name}'."
                self._load_settings_into_editor(
                    settings,
                    profile_name=name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Rename profile failed", exc)

        def _handle_delete_profile(self) -> None:
            if not self._allow_named_profiles:
                self._status_label.setText("Combined plot files use one saved settings document.")
                self._refresh_shell_state()
                return
            if on_delete_profile is None:
                self._status_label.setText("Delete profile is unavailable.")
                return
            if len(self._profile_names) <= 1:
                self._status_label.setText("At least one profile must remain available.")
                return
            if not self._confirm_context_change(action_label="deleting the current profile"):
                return
            decision = QMessageBox.question(
                self,
                "Delete profile",
                f"Delete saved profile '{self._current_profile_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if decision != QMessageBox.StandardButton.Yes:
                self._status_label.setText("Delete profile canceled.")
                return
            try:
                deleted_name = self._current_profile_name
                next_profile_name, message = on_delete_profile(deleted_name)
                self._profile_names = [name for name in self._profile_names if name != deleted_name]
                if next_profile_name is None:
                    self._current_profile_name = self._profile_names[0]
                else:
                    self._current_profile_name = next_profile_name
                    if next_profile_name not in self._profile_names:
                        self._profile_names.append(next_profile_name)
                if on_load_profile is None:
                    raise ValueError("Profile loading is unavailable after deletion.")
                loaded = on_load_profile(self._current_profile_name)
                self._load_settings_into_editor(
                    loaded,
                    profile_name=self._current_profile_name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Delete profile failed", exc)

        def _build_ui(self) -> None:
            root = QWidget(self)
            root.setObjectName("windowRoot")
            self.setCentralWidget(root)
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(12, 12, 12, 12)
            root_layout.setSpacing(12)
            self._apply_window_icon()

            header = QFrame(root)
            header.setObjectName("appHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(14, 12, 14, 12)
            header_layout.setSpacing(10)

            title_block = QVBoxLayout()
            title_block.setSpacing(2)
            app_title = QLabel("LiNaK Studio")
            app_title.setObjectName("appTitle")
            subtitle = QLabel(title)
            subtitle.setObjectName("appSubtitle")
            title_block.addWidget(app_title)
            title_block.addWidget(subtitle)
            header_layout.addLayout(title_block, stretch=1)

            header_layout.addStretch(1)

            self._save_button = QPushButton("Save Profile")
            self._save_button.setProperty("role", "primary")
            self._save_button.clicked.connect(self._handle_save)
            self._register_tooltip(self._save_button, "profiles.save")
            self._apply_widget_tooltip(self._save_button)
            header_layout.addWidget(self._save_button)

            self._exit_button = QPushButton("Exit")
            self._exit_button.clicked.connect(self.close)
            header_layout.addWidget(self._exit_button)
            root_layout.addWidget(header)

            splitter = QSplitter(Qt.Orientation.Horizontal, root)
            splitter.setChildrenCollapsible(False)
            root_layout.addWidget(splitter, stretch=1)

            left_panel = QWidget(splitter)
            left_panel.setMinimumWidth(0)
            left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            left_layout = QHBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(12)

            nav_panel = QFrame(left_panel)
            nav_panel.setObjectName("navPanel")
            nav_panel.setMinimumWidth(96)
            nav_panel.setMaximumWidth(150)
            nav_layout = QVBoxLayout(nav_panel)
            nav_layout.setContentsMargins(14, 14, 14, 14)
            nav_layout.setSpacing(10)
            workspace_label = QLabel("Workspace")
            workspace_label.setObjectName("pageTitle")
            nav_layout.addWidget(workspace_label)
            self._nav_list = QListWidget(nav_panel)
            self._nav_list.setObjectName("navList")
            self._nav_list.currentRowChanged.connect(self._handle_navigation_change)
            nav_layout.addWidget(self._nav_list, stretch=1)
            left_layout.addWidget(nav_panel)

            inspector_panel = QFrame(left_panel)
            inspector_panel.setObjectName("inspectorPanel")
            inspector_panel.setMinimumWidth(0)
            inspector_panel.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            inspector_layout = QVBoxLayout(inspector_panel)
            inspector_layout.setContentsMargins(14, 14, 14, 14)
            inspector_layout.setSpacing(10)
            self._page_title_label = QLabel("Data")
            self._page_title_label.setObjectName("pageTitle")
            inspector_layout.addWidget(self._page_title_label)
            self._page_note_label = QLabel(
                "Choose which stored data and view mode are sent into the plot."
            )
            self._page_note_label.setWordWrap(True)
            self._page_note_label.setObjectName("sectionNote")
            inspector_layout.addWidget(self._page_note_label)
            self._page_stack = QStackedWidget(inspector_panel)
            self._page_stack.setMinimumWidth(0)
            self._page_stack.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            inspector_layout.addWidget(self._page_stack, stretch=1)
            left_layout.addWidget(inspector_panel, stretch=1)

            self._register_workspace_page("Profiles", self._build_profiles_page())
            self._register_workspace_page("Data", self._build_content_page())
            self._register_workspace_page("Layers", self._build_layers_page())
            self._register_workspace_page("Figure", self._build_figure_page())
            self._register_workspace_page("Advanced", self._build_advanced_page())
            self._sync_profile_selector()
            if self._nav_list is not None:
                self._nav_list.setCurrentRow(0)

            right_panel = QFrame(splitter)
            right_panel.setObjectName("previewPanel")
            right_panel.setMinimumWidth(320)
            right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(14, 14, 14, 14)
            right_layout.setSpacing(10)

            preview_title = QLabel("Figure Preview")
            preview_title.setObjectName("pageTitle")
            right_layout.addWidget(preview_title)

            preview_controls = QHBoxLayout()
            self._preview_button = QPushButton("Refresh Preview")
            self._preview_button.clicked.connect(self._handle_preview)
            self._register_tooltip(self._preview_button, "preview.refresh")
            self._apply_widget_tooltip(self._preview_button)
            preview_controls.addWidget(self._preview_button)
            fit_button = QPushButton("Fit")
            fit_button.clicked.connect(self._handle_fit_preview)
            self._register_tooltip(fit_button, "preview.fit")
            self._apply_widget_tooltip(fit_button)
            preview_controls.addWidget(fit_button)
            actual_size_button = QPushButton("100%")
            actual_size_button.clicked.connect(self._handle_actual_size_preview)
            self._register_tooltip(actual_size_button, "preview.actual_size")
            self._apply_widget_tooltip(actual_size_button)
            preview_controls.addWidget(actual_size_button)
            transparent_label = QLabel("Transparent save")
            preview_controls.addWidget(transparent_label)
            self.transparent_mode = self._combo(_TOGGLE_MODES, minimum_contents_length=4)
            self.transparent_mode.setEnabled(on_save_figure is not None)
            self._register_tooltip(self.transparent_mode, "export.transparent")
            self._apply_widget_tooltip(self.transparent_mode)
            preview_controls.addWidget(self.transparent_mode)
            self._save_figure_button = QPushButton("Export Figure")
            self._save_figure_button.setEnabled(on_save_figure is not None)
            self._save_figure_button.clicked.connect(self._handle_save_figure)
            self._register_tooltip(self._save_figure_button, "export.figure")
            self._apply_widget_tooltip(self._save_figure_button)
            preview_controls.addWidget(self._save_figure_button)
            self._auto_preview_checkbox = QCheckBox("Auto update")
            self._auto_preview_checkbox.setChecked(on_save_figure is not None)
            self._auto_preview_checkbox.setEnabled(on_save_figure is not None)
            self._auto_preview_checkbox.toggled.connect(self._handle_auto_preview_toggle)
            self._register_tooltip(self._auto_preview_checkbox, "preview.auto_update")
            self._apply_widget_tooltip(self._auto_preview_checkbox)
            preview_controls.addWidget(self._auto_preview_checkbox)
            preview_controls.addStretch(1)
            self._preview_status = QLabel("Preview ready.")
            preview_controls.addWidget(self._preview_status)
            right_layout.addLayout(preview_controls)

            self._preview_frame = QFrame(right_panel)
            self._preview_frame.setObjectName("previewFrame")
            self._preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
            self._preview_frame.installEventFilter(self)
            preview_frame_layout = QVBoxLayout(self._preview_frame)
            preview_frame_layout.setContentsMargins(6, 6, 6, 6)
            self._preview_scroll = QScrollArea(self._preview_frame)
            self._preview_scroll.setWidgetResizable(False)
            self._preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._preview_scroll.viewport().installEventFilter(self)

            self._preview_label = QLabel("Preview will appear here.\nUse mouse wheel to zoom.")
            self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._preview_label.setWordWrap(True)
            self._preview_label.setMinimumSize(1, 1)
            self._preview_label.installEventFilter(self)
            self._preview_scroll.setWidget(self._preview_label)
            preview_frame_layout.addWidget(self._preview_scroll, stretch=1)
            right_layout.addWidget(self._preview_frame, stretch=1)

            self._status_label.setObjectName("statusBar")
            self._status_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            right_layout.addWidget(self._status_label)

            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([760, 840])

            self._apply_theme_styles()

        def _is_dark_theme(self) -> bool:
            app = QApplication.instance()
            palette = app.palette() if app is not None else self.palette()
            window_color = palette.color(QPalette.ColorRole.Window)
            text_color = palette.color(QPalette.ColorRole.WindowText)
            return window_color.lightness() < text_color.lightness()

        def _theme_tokens(self) -> dict[str, str]:
            if self._is_dark_theme():
                return {
                    "window_bg": "#0b1220",
                    "header_bg": "#131d2d",
                    "panel_bg": "#101826",
                    "panel_elevated": "#162233",
                    "card_bg": "#142030",
                    "input_bg": "#0e1725",
                    "button_bg": "#182334",
                    "button_hover": "#1e2d42",
                    "button_pressed": "#25364c",
                    "disabled_bg": "#0f1724",
                    "disabled_text": "#607086",
                    "text": "#edf3fb",
                    "heading": "#f7fbff",
                    "muted_text": "#9caec5",
                    "border": "#31425a",
                    "border_soft": "#3f5270",
                    "input_border": "#425774",
                    "accent": "#2aa7b8",
                    "accent_hover": "#34bed1",
                    "accent_text": "#07151b",
                    "accent_soft": "#163e47",
                    "nav_hover": "#182637",
                    "nav_selected": "#18424b",
                    "nav_selected_border": "#2aa7b8",
                    "nav_selected_text": "#f6fbff",
                    "badge_bg": "#1a2637",
                    "badge_border": "#465a76",
                    "badge_text": "#edf3fb",
                    "warning_bg": "#33250e",
                    "warning_border": "#b98934",
                    "warning_text": "#f5d9a1",
                    "splitter": "#42556f",
                    "scrollbar_track": "#0f1724",
                    "scrollbar_thumb": "#40536d",
                    "scrollbar_thumb_hover": "#56708f",
                }
            return {
                "window_bg": "#eef3f8",
                "header_bg": "#f9fbfe",
                "panel_bg": "#fdfefe",
                "panel_elevated": "#ffffff",
                "card_bg": "#f8fafc",
                "input_bg": "#ffffff",
                "button_bg": "#f4f7fb",
                "button_hover": "#e9f0f8",
                "button_pressed": "#dbe6f2",
                "disabled_bg": "#eef2f6",
                "disabled_text": "#8190a3",
                "text": "#142033",
                "heading": "#0f1728",
                "muted_text": "#556274",
                "border": "#c8d3e0",
                "border_soft": "#d9e2ec",
                "input_border": "#bcc9d8",
                "accent": "#0f8f95",
                "accent_hover": "#0c7a80",
                "accent_text": "#f8feff",
                "accent_soft": "#d9f0f2",
                "nav_hover": "#edf4fa",
                "nav_selected": "#0f8f95",
                "nav_selected_border": "#0c7a80",
                "nav_selected_text": "#f8feff",
                "badge_bg": "#edf4fa",
                "badge_border": "#c3d1df",
                "badge_text": "#142033",
                "warning_bg": "#fff3d9",
                "warning_border": "#d8a94f",
                "warning_text": "#7c5400",
                "splitter": "#cad5e1",
                "scrollbar_track": "#edf2f7",
                "scrollbar_thumb": "#b9c5d4",
                "scrollbar_thumb_hover": "#95a6bb",
            }

        def _apply_theme_styles(self) -> None:
            colors = self._theme_tokens()
            self.setStyleSheet(
                f"QWidget#windowRoot {{"
                f"  background-color: {colors['window_bg']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QFrame#appHeader {{"
                f"  background-color: {colors['header_bg']};"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 14px;"
                f"}}"
                f"QFrame#navPanel, QFrame#inspectorPanel, QFrame#previewPanel {{"
                f"  background-color: {colors['panel_bg']};"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 14px;"
                f"}}"
                f"QFrame#previewFrame {{"
                f"  background-color: {colors['panel_elevated']};"
                f"  border: 1px solid {colors['border_soft']};"
                f"  border-radius: 12px;"
                f"}}"
                f"QGroupBox {{"
                f"  background-color: {colors['card_bg']};"
                f"  border: 1px solid {colors['border_soft']};"
                f"  border-radius: 12px;"
                f"  margin-top: 14px;"
                f"}}"
                f"QGroupBox::title {{"
                f"  subcontrol-origin: margin;"
                f"  left: 10px;"
                f"  padding: 0 5px;"
                f"  color: {colors['heading']};"
                f"  font-weight: 600;"
                f"}}"
                f"QPushButton {{"
                f"  padding: 7px 12px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QPushButton:hover {{"
                f"  border-color: {colors['accent']};"
                f"  background-color: {colors['button_hover']};"
                f"}}"
                f"QPushButton:pressed {{"
                f"  background-color: {colors['button_pressed']};"
                f"}}"
                f"QPushButton:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f'QPushButton[role="primary"] {{'
                f"  background-color: {colors['accent']};"
                f"  color: {colors['accent_text']};"
                f"  border-color: {colors['accent']};"
                f"  font-weight: 700;"
                f"}}"
                f'QPushButton[role="primary"]:hover {{'
                f"  background-color: {colors['accent_hover']};"
                f"  border-color: {colors['accent_hover']};"
                f"}}"
                f'QPushButton[role="primary"]:pressed {{'
                f"  background-color: {colors['accent_hover']};"
                f"}}"
                f'QPushButton[role="primary"]:disabled {{'
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QLineEdit, QComboBox, QPlainTextEdit, QListWidget {{"
                f"  border: 1px solid {colors['input_border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['input_bg']};"
                f"  color: {colors['text']};"
                f"  outline: none;"
                f"  selection-background-color: {colors['accent_soft']};"
                f"  selection-color: {colors['text']};"
                f"}}"
                f"QLineEdit, QComboBox {{ padding: 6px 8px; min-height: 18px; }}"
                f"QPlainTextEdit {{ padding: 6px; }}"
                f"QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QListWidget:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {{"
                f"  border-color: {colors['accent']};"
                f"  outline: none;"
                f"}}"
                f"QComboBox::drop-down {{ border: none; width: 24px; }}"
                f"QComboBox QAbstractItemView {{"
                f"  border: 1px solid {colors['border']};"
                f"  background-color: {colors['panel_elevated']};"
                f"  color: {colors['text']};"
                f"  selection-background-color: {colors['accent']};"
                f"  selection-color: {colors['accent_text']};"
                f"}}"
                f"QCheckBox {{ spacing: 6px; }}"
                f"QCheckBox::indicator {{"
                f"  width: 16px;"
                f"  height: 16px;"
                f"  border-radius: 4px;"
                f"  border: 1px solid {colors['input_border']};"
                f"  background-color: {colors['input_bg']};"
                f"}}"
                f"QCheckBox::indicator:hover {{ border-color: {colors['accent']}; }}"
                f"QCheckBox::indicator:checked {{"
                f"  background-color: {colors['accent']};"
                f"  border-color: {colors['accent']};"
                f"}}"
                f"QLabel#appTitle {{ font-size: 22px; font-weight: 700; color: {colors['heading']}; }}"
                f"QLabel#appSubtitle {{ color: {colors['muted_text']}; font-size: 12px; }}"
                f"QLabel#pageTitle {{ font-size: 17px; font-weight: 700; color: {colors['heading']}; }}"
                # f"QLabel#sectionTitle {{ font-size: 17px; font-weight: 700; color: {colors['heading']}; }}"
                f"QLabel#sectionNote {{ color: {colors['muted_text']}; }}"
                f"QLabel#stateBadge {{"
                f"  padding: 5px 12px;"
                f"  border: 1px solid {colors['badge_border']};"
                f"  border-radius: 999px;"
                f"  background-color: {colors['badge_bg']};"
                f"  color: {colors['badge_text']};"
                f"  font-weight: 600;"
                f"}}"
                f"QLabel#warningSummary, QLabel#inlineWarning {{"
                f"  padding: 8px 10px;"
                f"  border: 1px solid {colors['warning_border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['warning_bg']};"
                f"  color: {colors['warning_text']};"
                f"}}"
                f"QLabel#statusBar {{"
                f"  color: {colors['muted_text']};"
                f"  font-size: 12px;"
                f"  padding: 4px 6px 2px 6px;"
                f"}}"
                f"QFrame#inspectorPanel QStackedWidget {{"
                f"  background: transparent;"
                f"}}"
                f"QFrame#inspectorPanel QStackedWidget > QWidget {{"
                f"  background: transparent;"
                f"}}"
                f"QFrame#inspectorPanel QAbstractScrollArea::viewport {{"
                f"  background: transparent;"
                f"}}"
                f"QFrame#inspectorPanel QScrollArea > QWidget > QWidget {{"
                f"  background: transparent;"
                f"}}"
                f"QTabWidget#plotSubtabs::pane {{"
                f"  margin-top: 8px;"
                f"  border: none;"
                f"  background-color: transparent;"
                f"}}"
                f"QWidget#plotSubtabPage, QWidget#plotSubtabContent {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QScrollArea#plotSubtabScroll {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QWidget#plotSubtabViewport {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QScrollArea#plotSubtabScroll > QWidget > QWidget {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QTabWidget#plotSubtabs {{"
                f"  background: transparent;"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab {{"
                f"  padding: 7px 14px;"
                f"  margin-right: 6px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 10px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"  font-weight: 600;"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:hover {{"
                f"  background-color: {colors['nav_hover']};"
                f"  border-color: {colors['accent']};"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:selected {{"
                f"  background-color: {colors['accent_soft']};"
                f"  color: {colors['heading']};"
                f"  border-color: {colors['accent']};"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:!selected {{"
                f"  margin-top: 0px;"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:disabled {{"
                f"  color: {colors['disabled_text']};"
                f"  background-color: {colors['disabled_bg']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QListWidget#navList {{ border: none; background: transparent; padding: 4px; }}"
                f"QListWidget#navList::item {{"
                f"  padding: 12px 14px;"
                f"  margin: 3px 0;"
                f"  border-radius: 10px;"
                f"}}"
                f"QListWidget#navList::item:hover {{ background-color: {colors['nav_hover']}; }}"
                f"QListWidget#navList::item:selected {{"
                f"  background-color: {colors['nav_selected']};"
                f"  color: {colors['nav_selected_text']};"
                f"  border: 1px solid {colors['nav_selected_border']};"
                f"}}"
                f"QListWidget#seriesList::item:selected {{"
                f"  background: transparent;"
                f"  color: transparent;"
                f"  border: none;"
                f"}}"
                f"QListWidget#seriesList::item {{"
                f"  padding: 0px;"
                f"  margin: 0px;"
                f"  border: none;"
                f"  background: transparent;"
                f"  color: transparent;"
                f"}}"
                f"QListWidget#seriesList::indicator {{"
                f"  width: 0px;"
                f"  height: 0px;"
                f"  border: none;"
                f"  background: transparent;"
                f"}}"
                f"QListWidget#annotationList {{"
                f"  background-color: {colors['input_bg']};"
                f"}}"
                f"QListWidget#annotationList::item {{"
                f"  padding: 8px 10px;"
                f"  margin: 2px 0;"
                f"  border-radius: 8px;"
                f"  border: 1px solid transparent;"
                f"  color: {colors['text']};"
                f"}}"
                f"QListWidget#annotationList::item:hover {{"
                f"  background-color: {colors['nav_hover']};"
                f"}}"
                f"QListWidget#annotationList::item:selected {{"
                f"  background-color: {colors['accent_soft']};"
                f"  color: {colors['heading']};"
                f"  border: 1px solid {colors['accent']};"
                f"}}"
                f"QScrollArea {{ border: none; background: transparent; }}"
                f"QScrollBar:vertical {{"
                f"  background: {colors['scrollbar_track']};"
                f"  width: 12px;"
                f"  margin: 2px;"
                f"  border-radius: 6px;"
                f"}}"
                f"QScrollBar::handle:vertical {{"
                f"  background: {colors['scrollbar_thumb']};"
                f"  min-height: 24px;"
                f"  border-radius: 6px;"
                f"}}"
                f"QScrollBar::handle:vertical:hover {{ background: {colors['scrollbar_thumb_hover']}; }}"
                f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, "
                f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{"
                f"  height: 0px;"
                f"  background: transparent;"
                f"}}"
                f"QSplitter::handle {{"
                f"  background-color: {colors['splitter']};"
                f"  width: 5px;"
                f"  margin: 6px 4px;"
                f"  border-radius: 2px;"
                f"}}"
            )

        def _register_workspace_page(self, label: str, page: QWidget) -> None:
            if self._page_stack is None or self._nav_list is None:
                return
            self._page_stack.addWidget(page)
            self._nav_list.addItem(label)

        @staticmethod
        def _workspace_note(label: str) -> str:
            notes = {
                "Profiles": "Save, load, duplicate, import, export, or reset plot presets.",
                "Data": "Choose which stored data and view mode are sent into the plot.",
                "Layers": (
                    "Edit plotted layers here. Unsupported controls stay hidden so the inspector only shows meaningful options."
                ),
                "Figure": (
                    "Adjust figure-wide styling, layout, axes, and heatmap presentation. "
                    "Use the preview toolbar to export the current figure."
                ),
                "Advanced": "Use raw Matplotlib overrides only when the standard controls are not enough.",
            }
            return notes.get(label, "")

        def _handle_navigation_change(self, index: int) -> None:
            if self._page_stack is None or self._nav_list is None or index < 0:
                return
            self._page_stack.setCurrentIndex(index)
            item = self._nav_list.item(index)
            if item is not None:
                if self._page_title_label is not None:
                    self._page_title_label.setText(item.text())
                if self._page_note_label is not None:
                    self._page_note_label.setText(self._workspace_note(item.text()))

        def _humanized_analysis_name(self) -> str:
            if self._analysis_name is None:
                return "Generic"
            return str(self._analysis_name).replace("-", " ").strip().title() or "Generic"

        def _safe_collect_settings(self) -> tuple[dict[str, Any] | None, str | None]:
            try:
                return self._collect_settings(), None
            except Exception as exc:
                return None, str(exc)

        def _advanced_editor(self, section_key: str) -> Any | None:
            return getattr(self, f"{section_key}_json", None)

        def _set_line_text_if_different(self, widget: QLineEdit, value: str) -> None:
            text = str(value)
            if widget.text() == text:
                return
            widget.setText(text)

        def _set_plaintext_if_different(self, widget: QPlainTextEdit, value: Any) -> None:
            rendered = _format_json_block(value)
            if widget.toPlainText().strip() == rendered.strip():
                return
            widget.blockSignals(True)
            try:
                widget.setPlainText(rendered)
            finally:
                widget.blockSignals(False)

        def _apply_advanced_json_to_standard_controls(self, section_key: str) -> None:
            editor = self._advanced_editor(section_key)
            if editor is None or self._advanced_json_syncing:
                return

            raw_text = editor.toPlainText()
            try:
                parsed = _optional_json_dict(raw_text, field_name=section_key.replace("_", " "))
            except ValueError:
                self._refresh_shell_state()
                return
            if parsed is None:
                parsed = {}

            self._advanced_json_syncing = True
            self._suspend_preview_events = True
            try:
                if section_key == "figure_kwargs":
                    self._set_line_text_if_different(
                        self.figure_facecolor,
                        str(parsed.get("facecolor") or ""),
                    )
                elif section_key == "line_kwargs":
                    self._set_combo_value(
                        self.line_style,
                        str(parsed.get("linestyle") or ""),
                    )
                    self._set_line_text_if_different(
                        self.line_alpha,
                        "" if parsed.get("alpha") is None else str(parsed.get("alpha")),
                    )
                    self._set_line_text_if_different(
                        self.marker_size,
                        "" if parsed.get("markersize") is None else str(parsed.get("markersize")),
                    )
                elif section_key == "legend_kwargs":
                    frameon = parsed.get("frameon")
                    if isinstance(frameon, bool):
                        self._set_combo_value(
                            self.legend_frame_mode,
                            _toggle_to_mode(frameon, auto_mode="on"),
                        )
                    else:
                        self._set_combo_value(self.legend_frame_mode, "on")
                    ncols = parsed.get("ncols")
                    self._set_line_text_if_different(
                        self.legend_columns,
                        "" if ncols is None else str(ncols),
                    )
                elif section_key == "grid_kwargs":
                    self._set_line_text_if_different(
                        self.grid_color,
                        str(parsed.get("color") or ""),
                    )
                    axis_value = str(parsed.get("axis") or "both").strip().lower()
                    which_value = str(parsed.get("which") or "major").strip().lower()
                    if axis_value in _GRID_AXES:
                        self._set_combo_value(self.grid_axis, axis_value)
                    if which_value in _GRID_WHICH:
                        self._set_combo_value(self.grid_which, which_value)
                elif section_key == "tick_params_kwargs":
                    direction = str(parsed.get("direction") or "out").strip().lower()
                    axis_value = (
                        str(parsed.get("_ticks_axis", parsed.get("axis", "both"))).strip().lower()
                    )
                    minor_value = str(parsed.get("_minor_ticks_mode") or "off").strip().lower()
                    if direction in _TICK_DIRECTIONS:
                        self._set_combo_value(self.tick_direction, direction)
                    if axis_value in _TICK_AXES:
                        current_ticks_value = (
                            self.ticks_visibility.currentText().strip().lower()
                            if hasattr(self, "ticks_visibility")
                            else "both"
                        )
                        if current_ticks_value != "none":
                            self._set_combo_value(self.ticks_visibility, axis_value)
                    if minor_value in _MINOR_TICKS_MODES:
                        self._set_combo_value(self.minor_ticks_mode, minor_value)
                    self._set_line_text_if_different(
                        self.tick_length,
                        "" if parsed.get("length") is None else str(parsed.get("length")),
                    )
                    self._set_line_text_if_different(
                        self.tick_width,
                        "" if parsed.get("width") is None else str(parsed.get("width")),
                    )
                elif section_key == "savefig_kwargs":
                    transparent = parsed.get("transparent")
                    if isinstance(transparent, bool):
                        self._set_combo_value(
                            self.transparent_mode,
                            _toggle_to_mode(transparent, auto_mode="off"),
                        )
                    else:
                        self._set_combo_value(self.transparent_mode, "off")
            finally:
                self._suspend_preview_events = False
                self._advanced_json_syncing = False

            self._refresh_widget_states()
            self._schedule_preview_update()

        def _sync_standard_controls_to_advanced_json(self) -> None:
            if self._advanced_json_syncing:
                return
            settings, error = self._safe_collect_settings()
            if error or settings is None:
                return

            self._advanced_json_syncing = True
            try:
                if hasattr(self, "figure_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.figure_kwargs_json, settings.get("figure_kwargs")
                    )
                if hasattr(self, "axes_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.axes_kwargs_json, settings.get("axes_kwargs")
                    )
                if hasattr(self, "line_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.line_kwargs_json, settings.get("line_kwargs")
                    )
                if hasattr(self, "legend_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.legend_kwargs_json, settings.get("legend_kwargs")
                    )
                if hasattr(self, "grid_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.grid_kwargs_json, settings.get("grid_kwargs")
                    )
                if hasattr(self, "tick_params_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.tick_params_kwargs_json, settings.get("tick_params_kwargs")
                    )
                if hasattr(self, "savefig_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.savefig_kwargs_json, settings.get("savefig_kwargs")
                    )
            finally:
                self._advanced_json_syncing = False

        def _handle_advanced_editor_changed(self, section_key: str) -> None:
            if self._advanced_json_syncing:
                return
            self._apply_advanced_json_to_standard_controls(section_key)
            self._refresh_shell_state()

        def _refresh_disabled_tooltips(self) -> None:
            if hasattr(self, "_preview_button") and self._preview_button is not None:
                reason = "Auto Update is on." if self._auto_preview_checkbox.isChecked() else None
                self._apply_widget_tooltip(
                    self._preview_button,
                    disabled_reason=reason,
                )
            if hasattr(self, "_save_button") and self._save_button is not None:
                settings, error = self._safe_collect_settings()
                save_reason: str | None = None
                if error:
                    save_reason = "the current settings are invalid."
                elif settings is not None:
                    signature = self._signature(settings)
                    if self._saved_signature is not None and signature == self._saved_signature:
                        save_reason = "this profile is already saved."
                self._apply_widget_tooltip(self._save_button, disabled_reason=save_reason)

        def _refresh_shell_state(self) -> None:
            self._update_header_state()
            self._update_status_strip()
            self._update_warning_panel()
            profiles_label = getattr(self, "_profiles_current_label", None)
            if profiles_label is not None:
                profiles_label.setText(self._current_profile_name)
            self._refresh_disabled_tooltips()

        def _update_header_state(self) -> None:
            if not hasattr(self, "_save_button") or self._save_button is None:
                return
            settings, error = self._safe_collect_settings()
            if error:
                self._save_button.setEnabled(False)
                return
            if settings is None:
                self._save_button.setEnabled(False)
                return
            signature = self._signature(settings)
            self._save_button.setEnabled(
                self._saved_signature is None or signature != self._saved_signature
            )

        def _update_status_strip(self) -> None:
            source_badge = self._status_source_badge
            mode_badge = self._status_mode_badge
            layers_badge = self._status_layers_badge
            preview_badge = self._status_preview_badge
            profile_badge = self._status_profile_badge
            warning_badge = self._status_warning_badge
            if any(
                label is None
                for label in (
                    source_badge,
                    mode_badge,
                    layers_badge,
                    preview_badge,
                    profile_badge,
                    warning_badge,
                )
            ):
                return
            assert source_badge is not None
            assert mode_badge is not None
            assert layers_badge is not None
            assert preview_badge is not None
            assert profile_badge is not None
            assert warning_badge is not None
            settings, error = self._safe_collect_settings()
            descriptor = self._series_descriptor(self._series_active_index)
            source_name = str(descriptor.get("source_name") or "").strip() or "Current session"
            plot_family = self._current_plot_family()
            series_count = len(self._series_labels_data)
            visible_count = sum(1 for value in self._series_enabled_data if value)
            layer_count = series_count + len(self._annotations_data)
            preview_mode = "Auto" if self._auto_preview_checkbox.isChecked() else "Manual"
            warnings = _derive_warning_messages(settings, error=error)
            if self._preview_error:
                warnings = [f"Preview paused: {self._preview_error}"] + list(warnings)
            source_badge.setText(f"Source: {source_name}")
            mode_badge.setText(f"Mode: {self._humanized_analysis_name()} / {plot_family}")
            layers_badge.setText(
                f"Layers: {visible_count}/{series_count} series on, {len(self._annotations_data)} annotations"
                if series_count
                else f"Layers: {layer_count}"
            )
            preview_badge.setText(f"Preview: {preview_mode}")
            profile_badge.setText(f"Profile: {self._current_profile_name}")
            warning_badge.setText(f"Warnings: {len(warnings)}" if warnings else "Warnings: 0")

        def _update_warning_panel(self) -> None:
            settings, error = self._safe_collect_settings()
            warnings = _derive_warning_messages(settings, error=error)
            if self._preview_error:
                warnings = [f"Preview paused: {self._preview_error}"] + list(warnings)
            if self._warning_summary_label is None:
                current_status = self._status_label.text().strip()
                if warnings and current_status in {"Ready.", "Preview updated.", "Preview ready."}:
                    self._status_label.setText(f"Warning: {warnings[0]}")
                elif not warnings and current_status.startswith("Warning:"):
                    self._status_label.setText("Ready.")
                return
            if warnings:
                self._warning_summary_label.setText(
                    "\n".join(f"- {message}" for message in warnings[:4])
                )
                self._warning_summary_label.show()
            else:
                self._warning_summary_label.setText("")
                self._warning_summary_label.hide()

        def _handle_fit_preview(self) -> None:
            self._set_preview_zoom(1.0)
            self._refresh_preview_pixmap()
            self._preview_status.setText("Preview fit to workspace.")

        def _handle_actual_size_preview(self) -> None:
            if (
                self._preview_pixmap is None
                or self._preview_pixmap.isNull()
                or self._preview_scroll is None
            ):
                return
            viewport = self._preview_scroll.viewport().size()
            source = self._preview_pixmap.size()
            fit_scale = min(
                viewport.width() / max(1, source.width()),
                viewport.height() / max(1, source.height()),
            )
            if fit_scale <= 0.0:
                return
            self._set_preview_zoom(1.0 / fit_scale)
            self._refresh_preview_pixmap()
            self._preview_status.setText("Preview shown at 100%.")

        def _build_content_page(self) -> QWidget:
            self._tab_data = QWidget()
            self._tab_data_content = self._make_scrollable_tab(self._tab_data)
            self._build_data_tab()
            return self._tab_data

        def _build_layers_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            hint = QLabel(
                "Layers collect plotted series, grouped series, derived lines, and annotations. The inspector only shows controls that matter for the selected layer."
            )
            hint.setWordWrap(True)
            hint.setObjectName("sectionNote")
            layout.addWidget(hint)
            tabs = QTabWidget(page)
            tabs.setObjectName("plotSubtabs")
            tabs.setDocumentMode(True)
            self._layers_tabs = tabs
            self._tab_series = QWidget()
            self._tab_series_content = self._make_scrollable_tab(self._tab_series)
            self._build_series_tab()
            self._tab_annotations = QWidget()
            self._tab_annotations_content = self._make_scrollable_tab(self._tab_annotations)
            self._build_annotations_tab()
            tabs.addTab(self._tab_series, "Plot Layers")
            tabs.addTab(self._tab_annotations, "Annotations")
            tabs.currentChanged.connect(self._refresh_widget_states)
            layout.addWidget(tabs, stretch=1)
            return page

        def _build_figure_page(self) -> QWidget:
            page = QWidget()
            content = self._make_scrollable_tab(page)
            layout = QVBoxLayout(content)
            layout.setSpacing(12)
            hint = QLabel(
                "Figure controls the visual presentation of the current plot. "
                "All figure-wide settings are collected here in one place."
            )
            hint.setWordWrap(True)
            hint.setObjectName("sectionNote")
            layout.addWidget(hint)
            self._figure_tabs = None
            self._tab_text_content = QGroupBox("Text & Labels")
            layout.addWidget(self._tab_text_content)
            self._tab_legend_content = QGroupBox("Legend")
            layout.addWidget(self._tab_legend_content)
            self._tab_axes_content = QGroupBox("Axes & Limits")
            layout.addWidget(self._tab_axes_content)
            self._tab_ticks_grid_content = QGroupBox("Ticks & Grid")
            layout.addWidget(self._tab_ticks_grid_content)
            self._tab_lines_content = QGroupBox("Line / Marker Style")
            layout.addWidget(self._tab_lines_content)
            self._tab_heatmap_content = QGroupBox("Heatmap / Colorbar")
            layout.addWidget(self._tab_heatmap_content)
            self._tab_canvas_content = QGroupBox("Canvas / Font / Figure Size")
            layout.addWidget(self._tab_canvas_content)

            self._build_text_tab()
            self._build_legend_tab()
            self._build_axes_tab()
            self._build_ticks_grid_tab()
            self._build_lines_tab()
            self._build_heatmap_tab()
            self._build_canvas_tab()
            layout.addStretch(1)
            return page

        def _build_profiles_page(self) -> QWidget:
            page = QWidget()
            content = self._make_scrollable_tab(page)
            layout = QVBoxLayout(content)
            layout.setSpacing(12)

            selection_group = QGroupBox("Profile Selection")
            selection_form = QFormLayout(selection_group)
            self._profile_selector = _ScrollSafeComboBox()
            self._profile_selector.setMinimumContentsLength(14)
            self._profile_selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._configure_horizontal_growth(self._profile_selector)
            self._profile_selector.currentIndexChanged.connect(
                self._handle_profile_selection_request
            )
            self._profile_selector.setEnabled(self._allow_named_profiles)
            self._add_form_row(
                selection_form,
                "Profile",
                self._profile_selector,
                tooltip_id="profiles.selector",
            )
            layout.addWidget(selection_group)

            current_group = QGroupBox("Current Profile")
            current_layout = QFormLayout(current_group)
            self._profiles_current_label = QLabel(self._current_profile_name)
            self._profiles_current_label.setWordWrap(True)
            current_note = QLabel(
                "Profiles store reusable plotting presets inside the current HDF5 source."
                if self._allow_named_profiles
                else "Combined plot files store one shared settings document for all plotted series."
            )
            current_note.setWordWrap(True)
            current_layout.addRow("Active profile", self._profiles_current_label)
            current_layout.addRow("", current_note)
            layout.addWidget(current_group)

            manage_group = QGroupBox("Manage Profiles")
            manage_layout = QGridLayout(manage_group)

            def _page_button(label: str, callback: Callable[[], None]) -> QPushButton:
                button = QPushButton(label)
                button.clicked.connect(callback)
                return button

            new_profile_button = _page_button("New Profile", self._handle_new_profile)
            self._register_tooltip(new_profile_button, "profiles.new")
            self._apply_widget_tooltip(new_profile_button)
            new_profile_button.setEnabled(self._allow_named_profiles)
            manage_layout.addWidget(new_profile_button, 0, 0)
            rename_button = _page_button("Rename", self._handle_rename_profile)
            self._register_tooltip(rename_button, "profiles.rename")
            self._apply_widget_tooltip(rename_button)
            rename_button.setEnabled(self._allow_named_profiles and on_delete_profile is not None)
            manage_layout.addWidget(rename_button, 0, 1)
            duplicate_button = _page_button("Duplicate", self._handle_duplicate_profile)
            self._register_tooltip(duplicate_button, "profiles.duplicate")
            self._apply_widget_tooltip(duplicate_button)
            duplicate_button.setEnabled(self._allow_named_profiles)
            manage_layout.addWidget(duplicate_button, 1, 0)
            self._profile_delete_button = _page_button("Delete", self._handle_delete_profile)
            self._register_tooltip(self._profile_delete_button, "profiles.delete")
            self._apply_widget_tooltip(self._profile_delete_button)
            self._profile_delete_button.setEnabled(
                self._allow_named_profiles
                and len(self._profile_names) > 1
                and on_delete_profile is not None
            )
            manage_layout.addWidget(self._profile_delete_button, 1, 1)
            save_profile_button = _page_button("Save Profile", self._handle_save)
            self._register_tooltip(save_profile_button, "profiles.save")
            self._apply_widget_tooltip(save_profile_button)
            manage_layout.addWidget(save_profile_button, 2, 0)
            reset_profile_button = _page_button("Reset Profile to Defaults", self._handle_reset)
            self._register_tooltip(reset_profile_button, "profiles.reset")
            self._apply_widget_tooltip(reset_profile_button)
            manage_layout.addWidget(reset_profile_button, 2, 1)
            layout.addWidget(manage_group)

            transfer_group = QGroupBox("Transfer Profiles")
            transfer_layout = QGridLayout(transfer_group)
            import_button = _page_button("Import Profile", self._handle_import_json)
            self._register_tooltip(import_button, "profiles.import")
            self._apply_widget_tooltip(import_button)
            transfer_layout.addWidget(import_button, 0, 0)
            export_json_button = _page_button("Export Profile JSON", self._handle_export_json)
            self._register_tooltip(export_json_button, "profiles.export_json")
            self._apply_widget_tooltip(export_json_button)
            transfer_layout.addWidget(export_json_button, 0, 1)
            layout.addWidget(transfer_group)
            layout.addStretch(1)
            self._sync_profile_selector()
            return page

        def _build_export_page(self) -> QWidget:
            page = QWidget()
            content = self._make_scrollable_tab(page)
            layout = QVBoxLayout(content)
            layout.setSpacing(12)

            export_group = QGroupBox("Figure Export")
            export_form = QFormLayout(export_group)
            self.transparent_mode = self._combo(_TOGGLE_MODES)
            self._add_form_row(
                export_form,
                "Transparent save",
                self.transparent_mode,
                tooltip_id="export.transparent",
            )
            export_note = QLabel(
                "Export creates an image file from the current preview. It does not replace saving the current plot profile."
            )
            export_note.setWordWrap(True)
            export_form.addRow("", export_note)
            layout.addWidget(export_group)

            actions_group = QGroupBox("Export Actions")
            actions_layout = QGridLayout(actions_group)

            def _page_button(label: str, callback: Callable[[], None]) -> QPushButton:
                button = QPushButton(label)
                button.clicked.connect(callback)
                return button

            export_button = _page_button("Export Figure", self._handle_save_figure)
            export_button.setEnabled(on_save_figure is not None)
            self._register_tooltip(export_button, "export.figure")
            self._apply_widget_tooltip(export_button)
            actions_layout.addWidget(export_button, 0, 0)
            layout.addWidget(actions_group)
            layout.addStretch(1)
            return page

        def _build_advanced_page(self) -> QWidget:
            self._tab_advanced = QWidget()
            self._tab_advanced_content = self._make_scrollable_tab(self._tab_advanced)
            self._build_advanced_tab()
            return self._tab_advanced

        def _build_text_tab(self) -> None:
            form = QFormLayout(self._tab_text_content)
            title_row, self.title_text, title_lock = self._lockable_line(
                placeholder="Leave blank to hide the title",
                allow_off=True,
            )
            x_label_row, self.x_label, x_label_lock = self._lockable_line(
                placeholder="Matplotlib mathtext supported, e.g. Distance ($A$)",
                allow_off=True,
            )
            y_label_row, self.y_label, y_label_lock = self._lockable_line(
                placeholder="e.g. Density ($g/cm^3$)",
                allow_off=True,
            )
            self._connect_lockable_line("title", self.title_text, title_lock, allow_off=True)
            self._connect_lockable_line("x_label", self.x_label, x_label_lock, allow_off=True)
            self._connect_lockable_line("y_label", self.y_label, y_label_lock, allow_off=True)
            self.title_font = self._line()

            self._add_form_row(form, "Title", title_row, tooltip_id="figure.text.title")
            self._add_form_row(form, "X label", x_label_row, tooltip_id="figure.text.x_label")
            self._add_form_row(form, "Y label", y_label_row, tooltip_id="figure.text.y_label")
            self._add_form_row(
                form,
                "Title font",
                self.title_font,
                tooltip_id="figure.text.title_font",
            )
            math_hint = QLabel("Math labels: e.g. $cm^3$, $\\Delta G$, $\\rho$.")
            math_hint.setWordWrap(True)
            self._add_form_row(form, "", math_hint)
            self._title_rows = [(form, self.title_font)]
            self._title_detail_widgets = [self.title_text]
            self._x_label_detail_widgets = [self.x_label]
            self._y_label_detail_widgets = [self.y_label]

        def _build_legend_tab(self) -> None:
            form = QFormLayout(self._tab_legend_content)
            self._figure_legend_section = self._tab_legend_content
            self.legend_mode = self._combo(_TOGGLE_MODES)
            self.legend_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.legend_title = self._line()
            self.legend_loc = self._combo(_LEGEND_LOCATIONS)
            self.legend_frame_mode = self._combo(_TOGGLE_MODES)
            self.legend_columns = self._line("1")
            self.legend_font = self._line()
            self._add_form_row(form, "Legend", self.legend_mode, tooltip_id="figure.legend.enabled")
            self._add_form_row(
                form,
                "Legend title",
                self.legend_title,
                tooltip_id="figure.legend.title",
            )
            self._add_form_row(
                form,
                "Legend location",
                self.legend_loc,
                tooltip_id="figure.legend.location",
            )
            self._add_form_row(
                form,
                "Legend frame",
                self.legend_frame_mode,
                tooltip_id="figure.legend.frame",
            )
            self._add_form_row(
                form,
                "Legend columns",
                self.legend_columns,
                tooltip_id="figure.legend.columns",
            )
            self._add_form_row(
                form,
                "Legend font",
                self.legend_font,
                tooltip_id="figure.legend.font",
            )

            self._legend_rows = [
                (form, self.legend_title),
                (form, self.legend_loc),
                (form, self.legend_frame_mode),
                (form, self.legend_columns),
                (form, self.legend_font),
            ]

        def _build_axes_tab(self) -> None:
            layout = QVBoxLayout(self._tab_axes_content)
            self.x_scale = self._combo(("linear", "log", "symlog", "logit"))
            self.y_scale = self._combo(("linear", "log", "symlog", "logit"))
            self.axes_border_mode = self._combo(_BORDER_MODES)
            top_form = QFormLayout()
            self._add_form_row(top_form, "X scale", self.x_scale, tooltip_id="figure.axes.x_scale")
            self._add_form_row(top_form, "Y scale", self.y_scale, tooltip_id="figure.axes.y_scale")
            self._add_form_row(
                top_form,
                "Plot border",
                self.axes_border_mode,
                tooltip_id="figure.axes.border",
            )
            sides_widget = QWidget()
            sides_layout = QHBoxLayout(sides_widget)
            sides_layout.setContentsMargins(0, 0, 0, 0)
            self.border_left = QCheckBox("Left")
            self.border_right = QCheckBox("Right")
            self.border_top = QCheckBox("Top")
            self.border_bottom = QCheckBox("Bottom")
            for cb in (self.border_left, self.border_right, self.border_top, self.border_bottom):
                cb.setChecked(True)
                sides_layout.addWidget(cb)
            sides_layout.addStretch(1)
            self._add_form_row(
                top_form,
                "Sides",
                sides_widget,
                tooltip_id="figure.axes.border_sides",
            )
            self._border_custom_rows = [(top_form, sides_widget)]
            self.axes_border_mode.currentTextChanged.connect(self._refresh_widget_states)
            layout.addLayout(top_form)

            limits = QGroupBox("Limits")
            limits_form = QFormLayout(limits)
            x_limits_row, self.x_min, self.x_max, x_limits_lock = self._lockable_pair()
            y_limits_row, self.y_min, self.y_max, y_limits_lock = self._lockable_pair()
            self._connect_lockable_line("x_lim", self.x_min, x_limits_lock)
            self.x_max.textEdited.connect(lambda _text: self._handle_synced_field_edit("x_lim"))
            self._connect_lockable_line("y_lim", self.y_min, y_limits_lock)
            self.y_max.textEdited.connect(lambda _text: self._handle_synced_field_edit("y_lim"))
            self._add_form_row(
                limits_form,
                "X min / max",
                x_limits_row,
                tooltip_id="figure.axes.x_limits",
            )
            self._add_form_row(
                limits_form,
                "Y min / max",
                y_limits_row,
                tooltip_id="figure.axes.y_limits",
            )
            layout.addWidget(limits)

            label_spacing = QGroupBox("Label Spacing")
            label_spacing_form = QFormLayout(label_spacing)
            x_label_pad_row, self.x_label_pad, x_label_pad_lock = self._lockable_line(
                placeholder="points"
            )
            y_label_pad_row, self.y_label_pad, y_label_pad_lock = self._lockable_line(
                placeholder="points"
            )
            self._connect_lockable_line("x_label_pad", self.x_label_pad, x_label_pad_lock)
            self._connect_lockable_line("y_label_pad", self.y_label_pad, y_label_pad_lock)
            self._add_form_row(
                label_spacing_form,
                "X label pad",
                x_label_pad_row,
                tooltip_id="figure.axes.x_label_pad",
            )
            self._add_form_row(
                label_spacing_form,
                "Y label pad",
                y_label_pad_row,
                tooltip_id="figure.axes.y_label_pad",
            )
            self.label_font = self._line()
            self._add_form_row(
                label_spacing_form,
                "Label font",
                self.label_font,
                tooltip_id="figure.axes.label_font",
            )
            layout.addWidget(label_spacing)
            layout.addStretch(1)

        def _build_ticks_grid_tab(self) -> None:
            layout = QVBoxLayout(self._tab_ticks_grid_content)
            ticks = QGroupBox("Ticks")
            ticks_form = QFormLayout(ticks)
            self.ticks_visibility = self._combo(_TICK_VISIBILITY_MODES)
            self.ticks_visibility.currentTextChanged.connect(self._refresh_widget_states)
            self.tick_font = self._line()
            self.tick_direction = self._combo(_TICK_DIRECTIONS)
            self.tick_length = self._line("points")
            self.tick_width = self._line("points")
            self.minor_ticks_mode = self._combo(_MINOR_TICKS_MODES)
            x_ticks_row, self.x_ticks, x_ticks_lock = self._lockable_line(
                placeholder="e.g. 0, 1, 2"
            )
            y_ticks_row, self.y_ticks, y_ticks_lock = self._lockable_line(
                placeholder="e.g. 0, 5, 10"
            )
            self._connect_lockable_line("x_ticks", self.x_ticks, x_ticks_lock)
            self._connect_lockable_line("y_ticks", self.y_ticks, y_ticks_lock)
            self.x_tick_rotation = self._line("degrees")
            self.y_tick_rotation = self._line("degrees")
            self._add_form_row(
                ticks_form,
                "Show ticks",
                self.ticks_visibility,
                tooltip_id="figure.ticks.show",
            )
            self._add_form_row(
                ticks_form,
                "X ticks",
                x_ticks_row,
                tooltip_id="figure.ticks.x_ticks",
            )
            self._add_form_row(
                ticks_form,
                "Y ticks",
                y_ticks_row,
                tooltip_id="figure.ticks.y_ticks",
            )
            self._add_form_row(
                ticks_form,
                "X rotation",
                self.x_tick_rotation,
                tooltip_id="figure.ticks.x_rotation",
            )
            self._add_form_row(
                ticks_form,
                "Y rotation",
                self.y_tick_rotation,
                tooltip_id="figure.ticks.y_rotation",
            )
            self._add_form_row(
                ticks_form,
                "Tick font",
                self.tick_font,
                tooltip_id="figure.ticks.font",
            )
            self._add_form_row(
                ticks_form,
                "Direction",
                self.tick_direction,
                tooltip_id="figure.ticks.direction",
            )
            self._add_form_row(
                ticks_form,
                "Length",
                self.tick_length,
                tooltip_id="figure.ticks.length",
            )
            self._add_form_row(
                ticks_form,
                "Width",
                self.tick_width,
                tooltip_id="figure.ticks.width",
            )
            self._add_form_row(
                ticks_form,
                "Minor ticks",
                self.minor_ticks_mode,
                tooltip_id="figure.ticks.minor",
            )
            layout.addWidget(ticks)

            grid = QGroupBox("Grid")
            grid_form = QFormLayout(grid)
            self.grid_mode = self._combo(_TOGGLE_MODES)
            self.grid_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.grid_linestyle = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.grid_linewidth = self._line()
            self.grid_alpha = self._line()
            grid_color_row, self.grid_color = self._color_field(
                placeholder="#dddddd",
                tooltip_id="figure.grid.color",
            )
            self.grid_axis = self._combo(_GRID_AXES)
            self.grid_which = self._combo(_GRID_WHICH)
            self._add_form_row(
                grid_form, "Show grid", self.grid_mode, tooltip_id="figure.grid.show"
            )
            self._add_form_row(
                grid_form,
                "Line style",
                self.grid_linestyle,
                tooltip_id="figure.grid.line_style",
            )
            self._add_form_row(
                grid_form,
                "Line width",
                self.grid_linewidth,
                tooltip_id="figure.grid.line_width",
            )
            self._add_form_row(
                grid_form,
                "Alpha",
                self.grid_alpha,
                tooltip_id="figure.grid.alpha",
            )
            self._add_form_row(grid_form, "Color", grid_color_row, tooltip_id="figure.grid.color")
            self._add_form_row(grid_form, "Axis", self.grid_axis, tooltip_id="figure.grid.axis")
            self._add_form_row(grid_form, "Lines", self.grid_which, tooltip_id="figure.grid.lines")
            layout.addWidget(grid)
            layout.addStretch(1)

            self._ticks_rows = [
                (ticks_form, x_ticks_row),
                (ticks_form, y_ticks_row),
                (ticks_form, self.x_tick_rotation),
                (ticks_form, self.y_tick_rotation),
                (ticks_form, self.tick_font),
                (ticks_form, self.tick_direction),
                (ticks_form, self.tick_length),
                (ticks_form, self.tick_width),
                (ticks_form, self.minor_ticks_mode),
            ]
            self._grid_rows = [
                (grid_form, self.grid_linestyle),
                (grid_form, self.grid_linewidth),
                (grid_form, self.grid_alpha),
                (grid_form, grid_color_row),
                (grid_form, self.grid_axis),
                (grid_form, self.grid_which),
            ]

        def _build_lines_tab(self) -> None:
            layout = QVBoxLayout(self._tab_lines_content)
            lines = QGroupBox("Lines and Markers")
            self._figure_lines_group = lines
            lines_form = QFormLayout(lines)
            self.line_width = self._line()
            self.line_style = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.line_alpha = self._line("0.0 - 1.0")
            self.markers_mode = self._combo(_TOGGLE_MODES)
            self.markers_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.marker_size = self._line("e.g. 5")
            self._add_form_row(
                lines_form,
                "Line width",
                self.line_width,
                tooltip_id="figure.lines.width",
            )
            self._add_form_row(
                lines_form,
                "Line style",
                self.line_style,
                tooltip_id="figure.lines.style",
            )
            self._add_form_row(
                lines_form,
                "Line alpha",
                self.line_alpha,
                tooltip_id="figure.lines.alpha",
            )
            self._add_form_row(
                lines_form,
                "Show markers",
                self.markers_mode,
                tooltip_id="figure.lines.markers",
            )
            self._add_form_row(
                lines_form,
                "Marker size",
                self.marker_size,
                tooltip_id="figure.lines.marker_size",
            )
            layout.addWidget(lines)
            layout.addStretch(1)
            self._marker_rows = [(lines_form, self.marker_size)]

        def _build_heatmap_tab(self) -> None:
            layout = QVBoxLayout(self._tab_heatmap_content)
            group = QGroupBox("Heatmap Rendering")
            self._figure_heatmap_group = group
            form = QFormLayout(group)
            self.heatmap_cmap = self._combo(
                (
                    "turbo",
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",
                    "coolwarm",
                    "RdBu_r",
                    "seismic",
                ),
                editable=True,
            )
            self.heatmap_cmap.currentTextChanged.connect(self._schedule_preview_update)
            self.heatmap_vmin = self._line("auto")
            self.heatmap_vmin.textChanged.connect(self._schedule_preview_update)
            self.heatmap_vmax = self._line("auto")
            self.heatmap_vmax.textChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                form, "Colormap", self.heatmap_cmap, tooltip_id="figure.heatmap.cmap"
            )
            self._add_form_row(
                form, "Color min", self.heatmap_vmin, tooltip_id="figure.heatmap.vmin"
            )
            self._add_form_row(
                form, "Color max", self.heatmap_vmax, tooltip_id="figure.heatmap.vmax"
            )
            self.heatmap_normalization_mode = self._combo(
                tuple(_HEATMAP_NORMALIZATION_LABEL_BY_MODE.values())
            )
            self.heatmap_normalization_mode.currentTextChanged.connect(
                self._schedule_preview_update
            )
            self._add_form_row(
                form,
                "Normalization",
                self.heatmap_normalization_mode,
                tooltip_id="figure.heatmap.normalize_mode",
            )
            self.heatmap_log_scale = QCheckBox("Use log color scale")
            self.heatmap_log_scale.stateChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                form, "Color scale", self.heatmap_log_scale, tooltip_id="figure.heatmap.log_scale"
            )
            layout.addWidget(group)

            cb_group = QGroupBox("Colorbar")
            self._figure_colorbar_group = cb_group
            cb_form = QFormLayout(cb_group)
            self.heatmap_colorbar_enabled = QCheckBox("Show colorbar")
            self.heatmap_colorbar_enabled.setChecked(True)
            self.heatmap_colorbar_enabled.stateChanged.connect(self._refresh_widget_states)
            self.heatmap_colorbar_enabled.stateChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                cb_form,
                "",
                self.heatmap_colorbar_enabled,
                tooltip_id="figure.heatmap.colorbar_enabled",
            )
            self.heatmap_colorbar_label = self._line("auto")
            self.heatmap_colorbar_label.textChanged.connect(self._schedule_preview_update)
            self.heatmap_colorbar_label_size = self._line("")
            self.heatmap_colorbar_label_size.textChanged.connect(self._schedule_preview_update)
            self.heatmap_colorbar_tick_size = self._line("")
            self.heatmap_colorbar_tick_size.textChanged.connect(self._schedule_preview_update)
            self.heatmap_colorbar_position = self._combo(("right", "left", "top", "bottom"))
            self.heatmap_colorbar_position.currentTextChanged.connect(self._schedule_preview_update)
            self.heatmap_colorbar_pad = self._line("0.05")
            self.heatmap_colorbar_pad.textChanged.connect(self._schedule_preview_update)
            self.heatmap_colorbar_shrink = self._line("1.0")
            self.heatmap_colorbar_shrink.textChanged.connect(self._schedule_preview_update)
            self.heatmap_colorbar_aspect = self._line("20")
            self.heatmap_colorbar_aspect.textChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                cb_form,
                "Label",
                self.heatmap_colorbar_label,
                tooltip_id="figure.heatmap.colorbar_label",
            )
            self._add_form_row(
                cb_form,
                "Label size",
                self.heatmap_colorbar_label_size,
                tooltip_id="figure.heatmap.colorbar_label_size",
            )
            self._add_form_row(
                cb_form,
                "Tick size",
                self.heatmap_colorbar_tick_size,
                tooltip_id="figure.heatmap.colorbar_tick_size",
            )
            self._add_form_row(
                cb_form,
                "Position",
                self.heatmap_colorbar_position,
                tooltip_id="figure.heatmap.colorbar_position",
            )
            self._add_form_row(
                cb_form,
                "Padding",
                self.heatmap_colorbar_pad,
                tooltip_id="figure.heatmap.colorbar_pad",
            )
            self._add_form_row(
                cb_form,
                "Shrink",
                self.heatmap_colorbar_shrink,
                tooltip_id="figure.heatmap.colorbar_shrink",
            )
            self._add_form_row(
                cb_form,
                "Aspect",
                self.heatmap_colorbar_aspect,
                tooltip_id="figure.heatmap.colorbar_aspect",
            )
            self._colorbar_rows = [
                (cb_form, self.heatmap_colorbar_label),
                (cb_form, self.heatmap_colorbar_label_size),
                (cb_form, self.heatmap_colorbar_tick_size),
                (cb_form, self.heatmap_colorbar_position),
                (cb_form, self.heatmap_colorbar_pad),
                (cb_form, self.heatmap_colorbar_shrink),
                (cb_form, self.heatmap_colorbar_aspect),
            ]
            layout.addWidget(cb_group)
            layout.addStretch(1)

        def _build_canvas_tab(self) -> None:
            form = QFormLayout(self._tab_canvas_content)
            self.fig_width = self._line()
            self.fig_height = self._line()
            self.dpi = self._line()
            self.font_family = self._line()
            self.base_font_size = self._line()
            self.base_font_size.setPlaceholderText(str(DEFAULT_PLOT_STYLE.base_font_size))
            self.base_font_size.textChanged.connect(self._handle_base_font_size_changed)
            figure_facecolor_row, self.figure_facecolor = self._color_field(
                placeholder="#ffffff",
                tooltip_id="figure.canvas.facecolor",
            )
            self._add_form_row(
                form,
                "Figure width",
                self.fig_width,
                tooltip_id="figure.canvas.width",
            )
            self._add_form_row(
                form,
                "Figure height",
                self.fig_height,
                tooltip_id="figure.canvas.height",
            )
            self._add_form_row(form, "DPI", self.dpi, tooltip_id="figure.canvas.dpi")
            self._add_form_row(
                form,
                "Font family",
                self.font_family,
                tooltip_id="figure.canvas.font_family",
            )
            self._add_form_row(
                form,
                "Base font size",
                self.base_font_size,
                tooltip_id="figure.canvas.font_size",
            )
            self._add_form_row(
                form,
                "Figure facecolor",
                figure_facecolor_row,
                tooltip_id="figure.canvas.facecolor",
            )

        def _resolved_base_font_size_value(self) -> int:
            raw = self.base_font_size.text().strip()
            if not raw:
                return int(DEFAULT_PLOT_STYLE.base_font_size)
            try:
                parsed = int(raw)
            except ValueError:
                return int(DEFAULT_PLOT_STYLE.base_font_size)
            return max(1, parsed)

        def _refresh_font_size_placeholders(self) -> None:
            defaults = default_plot_font_sizes(self._resolved_base_font_size_value())
            self.title_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["title_font_size"])
            )
            self.label_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["label_font_size"])
            )
            self.tick_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["tick_font_size"])
            )
            self.legend_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["legend_font_size"])
            )

        def _handle_base_font_size_changed(self, *_unused: object) -> None:
            self._refresh_font_size_placeholders()

        def _build_series_tab(self) -> None:
            layout = QVBoxLayout(self._tab_series_content)

            selector_row = QHBoxLayout()
            hint = QLabel(
                "Edit the currently loaded plot layers here. Use Data to choose which stored profiles and view modes are loaded."
            )
            hint.setObjectName("sectionNote")
            hint.setWordWrap(True)
            selector_row.addWidget(hint, stretch=1)
            enable_all_button = QPushButton("All on")
            enable_all_button.clicked.connect(lambda: self._set_all_series_enabled(True))
            self._register_tooltip(enable_all_button, "series.all_on")
            self._apply_widget_tooltip(enable_all_button)
            disable_all_button = QPushButton("All off")
            disable_all_button.clicked.connect(lambda: self._set_all_series_enabled(False))
            self._register_tooltip(disable_all_button, "series.all_off")
            self._apply_widget_tooltip(disable_all_button)
            selector_row.addWidget(enable_all_button)
            selector_row.addWidget(disable_all_button)
            self._series_duplicate_button = QPushButton("Duplicate")
            self._series_duplicate_button.clicked.connect(self._duplicate_selected_series)
            self._register_tooltip(self._series_duplicate_button, "series.duplicate")
            self._apply_widget_tooltip(self._series_duplicate_button)
            selector_row.addWidget(self._series_duplicate_button)
            self._series_add_group_button = QPushButton("Add Group")
            self._series_add_group_button.clicked.connect(self._add_group_series)
            self._register_tooltip(self._series_add_group_button, "series.add_group")
            self._apply_widget_tooltip(self._series_add_group_button)
            selector_row.addWidget(self._series_add_group_button)
            layout.addLayout(selector_row)

            self.series_list = QListWidget()
            self.series_list.setObjectName("seriesList")
            self.series_list.setAlternatingRowColors(True)
            self.series_list.setMinimumHeight(180)
            self.series_list.setDragEnabled(True)
            self.series_list.setAcceptDrops(True)
            self.series_list.setDropIndicatorShown(True)
            self.series_list.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.series_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.series_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.series_list.setSpacing(2)
            self.series_list.currentRowChanged.connect(self._handle_series_list_selection_change)
            self.series_list.model().rowsMoved.connect(self._handle_series_list_rows_moved)
            layout.addWidget(self.series_list)

            panel = QGroupBox("Selected Layer")
            panel_layout = QVBoxLayout(panel)
            panel_note = QLabel(
                "Only controls that apply to the selected layer and current plot mode are shown."
            )
            panel_note.setWordWrap(True)
            panel_layout.addWidget(panel_note)
            visibility_group = QGroupBox("Visibility & Label")
            self._series_visibility_group = visibility_group
            visibility_form = QFormLayout(visibility_group)
            self.series_show_in_legend = self._combo(("on", "off"))
            self.series_show_in_legend.currentTextChanged.connect(self._on_series_editor_changed)
            self.series_label = self._line()
            self.series_label.textChanged.connect(self._on_series_label_changed)
            self._add_form_row(
                visibility_form,
                "Show in legend",
                self.series_show_in_legend,
                tooltip_id="series.show_in_legend",
            )
            self._add_form_row(
                visibility_form, "Label", self.series_label, tooltip_id="series.label"
            )
            panel_layout.addWidget(visibility_group)

            style_group = QGroupBox("Style")
            self._series_style_group = style_group
            style_layout = QVBoxLayout(style_group)
            style_note = QLabel(
                "Style affects the selected layer only. Derived fit and cumulative rows inherit their base-series styling."
            )
            style_note.setWordWrap(True)
            style_layout.addWidget(style_note)
            style_form = QFormLayout()
            series_color_row, self.series_color = self._color_field(
                placeholder="#1f77b4",
                tooltip_id="series.color",
            )
            self.series_color.textChanged.connect(self._on_series_editor_changed)
            self.series_alpha = self._line("0.0 - 1.0")
            self.series_alpha.textChanged.connect(self._on_series_editor_changed)
            self.series_line_width = self._line("blank: use global line width")
            self.series_line_width.textChanged.connect(self._on_series_editor_changed)
            self.series_marker = self._combo(
                ("", "o", "s", "^", "v", "d", "x", "+", ".", "*"), editable=True
            )
            self.series_marker.currentTextChanged.connect(self._on_series_editor_changed)
            self.series_line_kwargs_json = QPlainTextEdit()
            self.series_line_kwargs_json.setPlaceholderText('{"linestyle": "--", "alpha": 0.8}')
            self.series_line_kwargs_json.setFixedHeight(84)
            self._configure_horizontal_growth(self.series_line_kwargs_json)
            self.series_line_kwargs_json.textChanged.connect(self._on_series_editor_changed)
            self._add_form_row(style_form, "Color", series_color_row, tooltip_id="series.color")
            self._add_form_row(style_form, "Alpha", self.series_alpha, tooltip_id="series.alpha")
            self._add_form_row(
                style_form,
                "Line width",
                self.series_line_width,
                tooltip_id="series.line_width",
            )
            self._add_form_row(style_form, "Marker", self.series_marker, tooltip_id="series.marker")
            self._add_form_row(
                style_form,
                "Extra line kwargs (JSON)",
                self.series_line_kwargs_json,
                tooltip_id="series.line_kwargs_json",
            )
            style_layout.addLayout(style_form)
            panel_layout.addWidget(style_group)
            derived_group = QGroupBox("Derived Lines")
            self._series_derived_group = derived_group
            derived_layout = QVBoxLayout(derived_group)
            derived_note = QLabel(
                "Derived lines are computed from the currently displayed data after transforms, sectioning, and masking."
            )
            derived_note.setWordWrap(True)
            derived_layout.addWidget(derived_note)

            if self._analysis_name in {
                "density",
                "msd",
                "rdf",
                "potential",
                "position",
                "coordination",
                "orientation",
            }:
                error_group = QGroupBox("Uncertainty")
                self._series_uncertainty_group = error_group
                error_layout = QVBoxLayout(error_group)
                error_note = QLabel(
                    "Error overlays belong to the base series. Leave the error color blank to follow the base series color."
                )
                error_note.setWordWrap(True)
                error_layout.addWidget(error_note)
                error_form = QFormLayout()
                self._series_error_mode = self._combo(_TOGGLE_MODES)
                self._series_error_mode.currentTextChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    error_form,
                    "Enabled",
                    self._series_error_mode,
                    tooltip_id="series.error_enabled",
                )
                self._series_error_stat = self._combo(_ERROR_STATS)
                self._series_error_stat.currentTextChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    error_form,
                    "Statistic",
                    self._series_error_stat,
                    tooltip_id="series.error_stat",
                )
                self._series_error_style = self._combo(_ERROR_STYLES)
                self._series_error_style.currentTextChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    error_form,
                    "Style",
                    self._series_error_style,
                    tooltip_id="series.error_style",
                )
                error_color_row, self._series_error_color = self._color_field(
                    placeholder="Blank: follow series color",
                    tooltip_id="series.error_color",
                )
                self._series_error_color.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    error_form,
                    "Color",
                    error_color_row,
                    tooltip_id="series.error_color",
                )
                self._series_error_show_in_legend = self._combo(("on", "off"))
                self._series_error_show_in_legend.currentTextChanged.connect(
                    self._on_series_editor_changed
                )
                self._add_form_row(
                    error_form,
                    "Show in legend",
                    self._series_error_show_in_legend,
                    tooltip_id="series.show_in_legend",
                )
                self._series_error_label = self._line()
                self._series_error_label.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    error_form,
                    "Label",
                    self._series_error_label,
                    tooltip_id="series.label",
                )
                error_layout.addLayout(error_form)
                self._series_error_detail_rows = [
                    (error_form, self._series_error_stat),
                    (error_form, self._series_error_style),
                    (error_form, error_color_row),
                    (error_form, self._series_error_show_in_legend),
                    (error_form, self._series_error_label),
                ]
                self._series_error_summary = QLabel("No error overlay configured for this series.")
                self._series_error_summary.setWordWrap(True)
                self._register_tooltip(self._series_error_summary, "series.error.summary")
                self._apply_widget_tooltip(self._series_error_summary)
                error_layout.addWidget(self._series_error_summary)
                self._series_error_explanation = QLabel("")
                self._series_error_explanation.setWordWrap(True)
                self._register_tooltip(self._series_error_explanation, "series.error.summary")
                self._apply_widget_tooltip(self._series_error_explanation)
                error_layout.addWidget(self._series_error_explanation)
                self._series_error_warning = QLabel("")
                self._series_error_warning.setObjectName("inlineWarning")
                self._series_error_warning.setWordWrap(True)
                self._series_error_warning.hide()
                self._register_tooltip(self._series_error_warning, "series.error.summary")
                self._apply_widget_tooltip(self._series_error_warning)
                error_layout.addWidget(self._series_error_warning)
                self._series_error_style_note = QLabel(
                    "Bands use the default transparency and whiskers use the default capsize."
                )
                self._series_error_style_note.setWordWrap(True)
                self._series_error_style_note.hide()
                error_layout.addWidget(self._series_error_style_note)
                self._series_error_detail_widgets = [
                    error_note,
                    self._series_error_summary,
                    self._series_error_explanation,
                    self._series_error_warning,
                    self._series_error_style_note,
                ]
                panel_layout.addWidget(error_group)
            else:
                self._series_uncertainty_group = None

            cumulative_group = QGroupBox("Cumulative Average")
            self._series_cumulative_group = cumulative_group
            cumulative_layout = QVBoxLayout(cumulative_group)
            cumulative_note = QLabel(
                "Cumulative lines are running means of the currently displayed y-values, ordered by plotted x."
            )
            cumulative_note.setWordWrap(True)
            cumulative_layout.addWidget(cumulative_note)
            cumulative_form = QFormLayout()
            self._series_cumulative_mode = self._combo(_TOGGLE_MODES)
            self._series_cumulative_mode.currentTextChanged.connect(self._on_series_editor_changed)
            self._add_form_row(
                cumulative_form,
                "Enabled",
                self._series_cumulative_mode,
                tooltip_id="series.cumulative_enabled",
            )
            self._series_cumulative_show_in_legend = self._combo(_TOGGLE_MODES)
            self._series_cumulative_show_in_legend.currentTextChanged.connect(
                self._on_series_editor_changed
            )
            self._add_form_row(
                cumulative_form,
                "Show in legend",
                self._series_cumulative_show_in_legend,
                tooltip_id="series.cumulative_show_in_legend",
            )
            self._series_cumulative_label = self._line()
            self._series_cumulative_label.textChanged.connect(self._on_series_editor_changed)
            self._add_form_row(
                cumulative_form,
                "Label",
                self._series_cumulative_label,
                tooltip_id="series.cumulative_label",
            )
            self._series_cumulative_detail_rows = [
                (cumulative_form, self._series_cumulative_show_in_legend),
                (cumulative_form, self._series_cumulative_label),
            ]
            cumulative_layout.addLayout(cumulative_form)
            self._series_cumulative_summary = QLabel(
                "No cumulative-average line configured for this series."
            )
            self._series_cumulative_summary.setWordWrap(True)
            self._register_tooltip(self._series_cumulative_summary, "series.cumulative.summary")
            self._apply_widget_tooltip(self._series_cumulative_summary)
            cumulative_layout.addWidget(self._series_cumulative_summary)
            self._series_cumulative_style_note = QLabel(
                "Cumulative rows inherit color, alpha, and width from their base series."
            )
            self._series_cumulative_style_note.setWordWrap(True)
            self._series_cumulative_style_note.hide()
            cumulative_layout.addWidget(self._series_cumulative_style_note)
            self._series_cumulative_detail_widgets = [
                cumulative_note,
                self._series_cumulative_summary,
                self._series_cumulative_style_note,
            ]
            derived_layout.addWidget(cumulative_group)

            if self._analysis_name in {
                "density",
                "msd",
                "rdf",
                "potential",
                "position",
                "coordination",
            }:
                fit_group = QGroupBox("Fit")
                self._series_fit_group = fit_group
                fit_layout = QVBoxLayout(fit_group)
                fit_note = QLabel(
                    "Fits are derived child series based on the currently displayed data."
                )
                fit_note.setWordWrap(True)
                fit_layout.addWidget(fit_note)
                fit_form = QFormLayout()
                self._series_fit_mode = self._combo(_TOGGLE_MODES)
                self._series_fit_mode.currentTextChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "Enabled",
                    self._series_fit_mode,
                    tooltip_id="series.fit_enabled",
                )
                self._series_fit_type = self._combo(_FIT_TYPES)
                self._series_fit_type.currentTextChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "Type",
                    self._series_fit_type,
                    tooltip_id="series.fit_type",
                )
                self._series_fit_degree = self._line("2")
                self._series_fit_degree.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "Polynomial degree",
                    self._series_fit_degree,
                    tooltip_id="series.fit_degree",
                )
                self._series_fit_range_mode = self._combo(
                    tuple(mode.title() for mode in _FIT_RANGE_MODES)
                )
                self._series_fit_range_mode.currentTextChanged.connect(
                    self._on_series_editor_changed
                )
                self._add_form_row(
                    fit_form,
                    "Range",
                    self._series_fit_range_mode,
                    tooltip_id="series.fit_range_mode",
                )
                self._series_fit_x_min = self._line("Visible range")
                self._series_fit_x_min.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "X min",
                    self._series_fit_x_min,
                    tooltip_id="series.fit_x_min",
                )
                self._series_fit_x_max = self._line("Visible range")
                self._series_fit_x_max.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "X max",
                    self._series_fit_x_max,
                    tooltip_id="series.fit_x_max",
                )
                self._series_fit_show_in_legend = self._combo(_TOGGLE_MODES)
                self._series_fit_show_in_legend.currentTextChanged.connect(
                    self._on_series_editor_changed
                )
                self._add_form_row(
                    fit_form,
                    "Show in legend",
                    self._series_fit_show_in_legend,
                    tooltip_id="series.fit_show_in_legend",
                )
                self._series_fit_label = self._line()
                self._series_fit_label.textChanged.connect(self._on_series_fit_label_changed)
                self._add_form_row(
                    fit_form,
                    "Label",
                    self._series_fit_label,
                    tooltip_id="series.fit_label",
                )
                self._series_fit_detail_rows = [
                    (fit_form, self._series_fit_type),
                    (fit_form, self._series_fit_degree),
                    (fit_form, self._series_fit_range_mode),
                    (fit_form, self._series_fit_x_min),
                    (fit_form, self._series_fit_x_max),
                    (fit_form, self._series_fit_show_in_legend),
                    (fit_form, self._series_fit_label),
                ]
                fit_layout.addLayout(fit_form)

                fit_summary = QGroupBox("Fit Summary")
                self._series_fit_summary_group = fit_summary
                fit_summary_layout = QVBoxLayout(fit_summary)
                self._series_fit_summary = QLabel("No fit configured for this series.")
                self._series_fit_summary.setWordWrap(True)
                self._register_tooltip(self._series_fit_summary, "series.fit.summary")
                self._apply_widget_tooltip(self._series_fit_summary)
                fit_summary_layout.addWidget(self._series_fit_summary)
                self._series_fit_warning = QLabel("")
                self._series_fit_warning.setObjectName("inlineWarning")
                self._series_fit_warning.setWordWrap(True)
                self._series_fit_warning.hide()
                self._register_tooltip(self._series_fit_warning, "series.fit.warning")
                self._apply_widget_tooltip(self._series_fit_warning)
                fit_summary_layout.addWidget(self._series_fit_warning)
                self._series_fit_style_note = QLabel(
                    "Fit rows inherit color, alpha, and width from their base series."
                )
                self._series_fit_style_note.setWordWrap(True)
                self._series_fit_style_note.hide()
                fit_summary_layout.addWidget(self._series_fit_style_note)
                fit_layout.addWidget(fit_summary)
                self._series_fit_detail_widgets = [fit_note, fit_summary]
                derived_layout.addWidget(fit_group)

            panel_layout.addWidget(derived_group)

            group_group = QGroupBox("Group Members")
            self._series_group_group = group_group
            group_layout = QVBoxLayout(group_group)
            group_note = QLabel(
                "Grouped series aggregate several loaded base series after their current plot transforms. Grouped lines do not show error overlays in this pass."
            )
            group_note.setWordWrap(True)
            group_layout.addWidget(group_note)
            group_form = QFormLayout()
            self._series_group_reducer = self._combo(_GROUP_REDUCERS)
            self._series_group_reducer.currentTextChanged.connect(self._on_series_editor_changed)
            self._add_form_row(
                group_form,
                "Reducer",
                self._series_group_reducer,
                tooltip_id="series.group.reducer",
            )
            group_layout.addLayout(group_form)
            self._series_group_members = QListWidget()
            self._series_group_members.setSelectionMode(
                QAbstractItemView.SelectionMode.MultiSelection
            )
            self._series_group_members.itemSelectionChanged.connect(self._on_series_editor_changed)
            self._register_tooltip(self._series_group_members, "series.group.members")
            self._apply_widget_tooltip(self._series_group_members)
            group_layout.addWidget(self._series_group_members)
            self._series_group_summary = QLabel("This series is not grouped.")
            self._series_group_summary.setWordWrap(True)
            group_layout.addWidget(self._series_group_summary)
            panel_layout.addWidget(group_group)

            normalize_group = QGroupBox("Normalization")
            self._normalization_group = normalize_group
            normalize_layout = QVBoxLayout(normalize_group)
            normalize_form = QFormLayout()
            self.norm_mode = self._combo(_NORMALIZATION_MODES)
            self.norm_mode.currentTextChanged.connect(self._on_normalization_editor_changed)
            self.norm_value = self._line("Target value (required unless mode=none)")
            self.norm_value.textChanged.connect(self._on_normalization_editor_changed)
            self.norm_x_ref = self._line("Reference x (required for value_at_x)")
            self.norm_x_ref.textChanged.connect(self._on_normalization_editor_changed)
            self._add_form_row(
                normalize_form,
                "Mode",
                self.norm_mode,
                tooltip_id="series.norm.mode",
            )
            self._add_form_row(
                normalize_form,
                "Target",
                self.norm_value,
                tooltip_id="series.norm.target",
            )
            self._add_form_row(
                normalize_form,
                "Reference x",
                self.norm_x_ref,
                tooltip_id="series.norm.reference_x",
            )
            self._norm_value_row = (normalize_form, self.norm_value)
            self._norm_x_ref_row = (normalize_form, self.norm_x_ref)
            normalize_layout.addLayout(normalize_form)

            normalization_actions_widget = QWidget()
            normalization_actions = QHBoxLayout(normalization_actions_widget)
            normalization_actions.setContentsMargins(0, 0, 0, 0)
            normalization_actions.addStretch(1)
            self._normalization_copy_button = QPushButton("Copy settings to all series")
            self._normalization_copy_button.clicked.connect(
                self._copy_normalization_settings_to_all_series
            )
            normalization_actions.addWidget(self._normalization_copy_button)
            self._normalization_actions_widget = normalization_actions_widget
            normalize_layout.addWidget(normalization_actions_widget)

            self.normalization_warning = QLabel("")
            self.normalization_warning.setObjectName("inlineWarning")
            self.normalization_warning.setWordWrap(True)
            self.normalization_warning.hide()
            normalize_layout.addWidget(self.normalization_warning)

            norm_hint = QLabel(
                "Normalization affects only the displayed figure. Stored HDF5 datasets remain unchanged."
            )
            norm_hint.setWordWrap(True)
            self._normalization_hint_label = norm_hint
            normalize_layout.addWidget(norm_hint)
            panel_layout.addWidget(normalize_group)

            metadata_group = QGroupBox("Source Metadata")
            self._series_metadata_group = metadata_group
            metadata_form = QFormLayout(metadata_group)
            self._series_meta_default_label = QLabel("")
            self._series_meta_default_label.setWordWrap(True)
            self._series_meta_source_name = QLabel("")
            self._series_meta_source_name.setWordWrap(True)
            self._series_meta_source_dir = QLabel("")
            self._series_meta_source_dir.setWordWrap(True)
            self._series_meta_series_id = QLabel("")
            self._series_meta_series_id.setWordWrap(True)
            self._add_form_row(
                metadata_form,
                "Default label",
                self._series_meta_default_label,
                tooltip_id="series.meta.default_label",
            )
            self._add_form_row(
                metadata_form,
                "Source file",
                self._series_meta_source_name,
                tooltip_id="series.meta.source_file",
            )
            self._add_form_row(
                metadata_form,
                "Source directory",
                self._series_meta_source_dir,
                tooltip_id="series.meta.source_directory",
            )
            self._add_form_row(
                metadata_form,
                "Series id",
                self._series_meta_series_id,
                tooltip_id="series.meta.series_id",
            )
            self._series_stats_label = QLabel("No series statistics available yet.")
            self._series_stats_label.setWordWrap(True)
            metadata_form.addRow(QLabel("Series stats"), self._series_stats_label)
            panel_layout.addWidget(metadata_group)
            layout.addWidget(panel)
            layout.addStretch(1)

        def _build_annotations_tab(self) -> None:
            layout = QVBoxLayout(self._tab_annotations_content)

            hint = QLabel(
                "Annotations are figure-level overlays drawn on top of the current axes. Use data coordinates for plot-tied callouts and axes coordinates for labels that stay fixed to the frame."
            )
            hint.setObjectName("sectionNote")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            actions = QHBoxLayout()
            add_text_button = QPushButton("Add Text")
            add_text_button.clicked.connect(lambda: self._add_annotation("text"))
            self._register_tooltip(add_text_button, "annotations.add_text")
            self._apply_widget_tooltip(add_text_button)
            actions.addWidget(add_text_button)
            add_line_button = QPushButton("Add Line")
            add_line_button.clicked.connect(lambda: self._add_annotation("line"))
            self._register_tooltip(add_line_button, "annotations.add_line")
            self._apply_widget_tooltip(add_line_button)
            actions.addWidget(add_line_button)
            add_arrow_button = QPushButton("Add Arrow")
            add_arrow_button.clicked.connect(lambda: self._add_annotation("arrow"))
            self._register_tooltip(add_arrow_button, "annotations.add_arrow")
            self._apply_widget_tooltip(add_arrow_button)
            actions.addWidget(add_arrow_button)
            actions.addStretch(1)
            self._annotation_duplicate_button = QPushButton("Duplicate")
            self._annotation_duplicate_button.clicked.connect(self._duplicate_annotation)
            self._register_tooltip(self._annotation_duplicate_button, "annotations.duplicate")
            self._apply_widget_tooltip(self._annotation_duplicate_button)
            actions.addWidget(self._annotation_duplicate_button)
            self._annotation_delete_button = QPushButton("Delete")
            self._annotation_delete_button.clicked.connect(self._delete_annotation)
            self._register_tooltip(self._annotation_delete_button, "annotations.delete")
            self._apply_widget_tooltip(self._annotation_delete_button)
            actions.addWidget(self._annotation_delete_button)
            self._annotation_move_up_button = QPushButton("Move Up")
            self._annotation_move_up_button.clicked.connect(lambda: self._move_annotation(-1))
            self._register_tooltip(self._annotation_move_up_button, "annotations.move_up")
            self._apply_widget_tooltip(self._annotation_move_up_button)
            actions.addWidget(self._annotation_move_up_button)
            self._annotation_move_down_button = QPushButton("Move Down")
            self._annotation_move_down_button.clicked.connect(lambda: self._move_annotation(1))
            self._register_tooltip(self._annotation_move_down_button, "annotations.move_down")
            self._apply_widget_tooltip(self._annotation_move_down_button)
            actions.addWidget(self._annotation_move_down_button)
            layout.addLayout(actions)

            self.annotation_list = QListWidget()
            self.annotation_list.setObjectName("annotationList")
            self.annotation_list.setAlternatingRowColors(True)
            self.annotation_list.setMinimumHeight(180)
            self.annotation_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.annotation_list.currentRowChanged.connect(
                self._handle_annotation_list_selection_change
            )
            self._register_tooltip(self.annotation_list, "annotations.list")
            self._apply_widget_tooltip(self.annotation_list)
            list_group = QGroupBox("Annotations")
            list_layout = QVBoxLayout(list_group)
            list_layout.addWidget(self.annotation_list)

            panel = QGroupBox("Selected Annotation")
            panel_layout = QVBoxLayout(panel)
            panel_form = QFormLayout()

            self._annotation_enabled_mode = self._combo(_TOGGLE_MODES)
            self._annotation_enabled_mode.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._add_form_row(
                panel_form,
                "Enabled",
                self._annotation_enabled_mode,
                tooltip_id="annotations.enabled",
            )
            self._annotation_type = self._combo(_ANNOTATION_TYPES)
            self._annotation_type.currentTextChanged.connect(self._on_annotation_type_changed)
            self._add_form_row(
                panel_form,
                "Type",
                self._annotation_type,
                tooltip_id="annotations.type",
            )
            self._annotation_name = self._line()
            self._annotation_name.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Name",
                self._annotation_name,
                tooltip_id="annotations.name",
            )

            self._annotation_text = self._line()
            self._annotation_text.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Text",
                self._annotation_text,
                tooltip_id="annotations.text",
            )

            self._annotation_coord_system = self._combo(_ANNOTATION_COORD_SYSTEMS)
            self._annotation_coord_system.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._add_form_row(
                panel_form,
                "Coordinates",
                self._annotation_coord_system,
                tooltip_id="annotations.coord_system",
            )
            self._annotation_x = self._line("0.5")
            self._annotation_x.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(panel_form, "X", self._annotation_x, tooltip_id="annotations.x")
            self._annotation_y = self._line("0.5")
            self._annotation_y.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(panel_form, "Y", self._annotation_y, tooltip_id="annotations.y")

            self._annotation_horizontal_align = self._combo(_ANNOTATION_HORIZONTAL_ALIGN)
            self._annotation_horizontal_align.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._add_form_row(
                panel_form,
                "Horizontal align",
                self._annotation_horizontal_align,
                tooltip_id="annotations.horizontal_align",
            )
            self._annotation_vertical_align = self._combo(_ANNOTATION_VERTICAL_ALIGN)
            self._annotation_vertical_align.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._add_form_row(
                panel_form,
                "Vertical align",
                self._annotation_vertical_align,
                tooltip_id="annotations.vertical_align",
            )
            self._annotation_font_size = self._line("12")
            self._annotation_font_size.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Font size",
                self._annotation_font_size,
                tooltip_id="annotations.font_size",
            )
            self._annotation_rotation = self._line("0")
            self._annotation_rotation.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Rotation",
                self._annotation_rotation,
                tooltip_id="annotations.rotation",
            )

            annotation_color_row, self._annotation_color = self._color_field(
                placeholder="#000000",
                tooltip_id="annotations.color",
            )
            self._annotation_color.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Color",
                annotation_color_row,
                tooltip_id="annotations.color",
            )
            self._annotation_alpha = self._line("1.0")
            self._annotation_alpha.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Alpha",
                self._annotation_alpha,
                tooltip_id="annotations.alpha",
            )
            self._annotation_zorder = self._line("5")
            self._annotation_zorder.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Z-order",
                self._annotation_zorder,
                tooltip_id="annotations.zorder",
            )

            self._annotation_x1 = self._line("0")
            self._annotation_x1.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(panel_form, "X1", self._annotation_x1, tooltip_id="annotations.x1")
            self._annotation_y1 = self._line("0")
            self._annotation_y1.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(panel_form, "Y1", self._annotation_y1, tooltip_id="annotations.y1")
            self._annotation_x2 = self._line("1")
            self._annotation_x2.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(panel_form, "X2", self._annotation_x2, tooltip_id="annotations.x2")
            self._annotation_y2 = self._line("1")
            self._annotation_y2.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(panel_form, "Y2", self._annotation_y2, tooltip_id="annotations.y2")
            self._annotation_line_width = self._line("1.5")
            self._annotation_line_width.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Line width",
                self._annotation_line_width,
                tooltip_id="annotations.line_width",
            )
            self._annotation_line_style = self._combo(_ANNOTATION_LINE_STYLES)
            self._annotation_line_style.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._add_form_row(
                panel_form,
                "Line style",
                self._annotation_line_style,
                tooltip_id="annotations.line_style",
            )
            self._annotation_arrow_style = self._combo(_ANNOTATION_ARROW_STYLES)
            self._annotation_arrow_style.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._add_form_row(
                panel_form,
                "Arrow style",
                self._annotation_arrow_style,
                tooltip_id="annotations.arrow_style",
            )
            self._annotation_mutation_scale = self._line("12")
            self._annotation_mutation_scale.textChanged.connect(self._on_annotation_editor_changed)
            self._add_form_row(
                panel_form,
                "Mutation scale",
                self._annotation_mutation_scale,
                tooltip_id="annotations.mutation_scale",
            )
            panel_layout.addLayout(panel_form)

            self._annotation_text_rows = [
                (panel_form, self._annotation_text),
                (panel_form, self._annotation_x),
                (panel_form, self._annotation_y),
                (panel_form, self._annotation_font_size),
                (panel_form, self._annotation_rotation),
                (panel_form, self._annotation_horizontal_align),
                (panel_form, self._annotation_vertical_align),
            ]
            self._annotation_line_rows = [
                (panel_form, self._annotation_x1),
                (panel_form, self._annotation_y1),
                (panel_form, self._annotation_x2),
                (panel_form, self._annotation_y2),
                (panel_form, self._annotation_line_width),
                (panel_form, self._annotation_line_style),
            ]
            self._annotation_arrow_rows = [
                (panel_form, self._annotation_arrow_style),
                (panel_form, self._annotation_mutation_scale),
            ]
            self._annotation_common_detail_rows = [
                (panel_form, self._annotation_type),
                (panel_form, self._annotation_name),
                (panel_form, self._annotation_coord_system),
                (panel_form, annotation_color_row),
                (panel_form, self._annotation_alpha),
                (panel_form, self._annotation_zorder),
            ]

            summary_group = QGroupBox("Preview Summary")
            self._annotation_summary_group = summary_group
            summary_layout = QVBoxLayout(summary_group)
            self._annotation_preview_summary = QLabel(
                "Preview status will update after the next rendered preview."
            )
            self._annotation_preview_summary.setWordWrap(True)
            self._register_tooltip(self._annotation_preview_summary, "annotations.summary")
            self._apply_widget_tooltip(self._annotation_preview_summary)
            summary_layout.addWidget(self._annotation_preview_summary)
            panel_layout.addWidget(summary_group)

            content_splitter = QSplitter(Qt.Orientation.Horizontal)
            content_splitter.addWidget(list_group)
            content_splitter.addWidget(panel)
            content_splitter.setStretchFactor(0, 0)
            content_splitter.setStretchFactor(1, 1)
            content_splitter.setChildrenCollapsible(False)
            content_splitter.setSizes([260, 640])
            layout.addWidget(content_splitter, stretch=1)
            layout.addStretch(1)

        def _annotation_display_text(self, index: int) -> str:
            if index < 0 or index >= len(self._annotations_data):
                return f"{index + 1}: Annotation"
            return _annotation_display_text_from_entry(self._annotations_data[index], index=index)

        def _refresh_annotation_editor_rows(self) -> None:
            annotation_type = (
                self._annotation_type.currentText().strip().lower()
                if hasattr(self, "_annotation_type")
                else "text"
            )
            annotation_enabled = bool(self._annotations_data) and (
                self._annotation_enabled_mode.currentText().strip().lower() != "off"
                if hasattr(self, "_annotation_enabled_mode")
                else False
            )
            self._set_rows_visible(self._annotation_common_detail_rows, annotation_enabled)
            self._set_rows_visible(
                self._annotation_text_rows,
                annotation_enabled and annotation_type == "text",
            )
            self._set_rows_visible(
                self._annotation_line_rows,
                annotation_enabled and annotation_type in {"line", "arrow"},
            )
            self._set_rows_visible(
                self._annotation_arrow_rows,
                annotation_enabled and annotation_type == "arrow",
            )
            if self._annotation_summary_group is not None:
                self._annotation_summary_group.setVisible(annotation_enabled)

        def _refresh_annotation_list(self) -> None:
            if not hasattr(self, "annotation_list"):
                return
            self._annotation_syncing = True
            try:
                self.annotation_list.clear()
                for index in range(len(self._annotations_data)):
                    entry = self._annotations_data[index]
                    item = QListWidgetItem(self._annotation_display_text(index))
                    tooltip_text = "\n".join(
                        [
                            _annotation_primary_title(entry, index=index + 1),
                            f"Type: {_annotation_type_label(str(entry.get('type') or 'text'))}",
                            f"Coordinates: {str(entry.get('coord_system') or 'axes')}",
                        ]
                    )
                    item.setToolTip(tooltip_text)
                    self.annotation_list.addItem(item)
                if self._annotations_data:
                    target = min(
                        max(self._annotation_active_index, 0), len(self._annotations_data) - 1
                    )
                    self._annotation_active_index = target
                    self.annotation_list.setCurrentRow(target)
                else:
                    self._annotation_active_index = 0
                    self.annotation_list.setCurrentRow(-1)
            finally:
                self._annotation_syncing = False

        def _clear_annotation_editor(self) -> None:
            if not hasattr(self, "_annotation_name"):
                return
            self._annotation_syncing = True
            try:
                self._set_combo_value(self._annotation_enabled_mode, "on")
                self._set_combo_value(self._annotation_type, "text")
                self._annotation_name.setText("")
                self._set_combo_value(self._annotation_coord_system, "axes")
                self._annotation_color.setText("#000000")
                self._annotation_alpha.setText("1.0")
                self._annotation_zorder.setText("5")
                self._annotation_text.setText("")
                self._annotation_x.setText("0.5")
                self._annotation_y.setText("0.5")
                self._annotation_font_size.setText("12")
                self._annotation_rotation.setText("0")
                self._set_combo_value(self._annotation_horizontal_align, "center")
                self._set_combo_value(self._annotation_vertical_align, "center")
                self._annotation_x1.setText("0")
                self._annotation_y1.setText("0")
                self._annotation_x2.setText("1")
                self._annotation_y2.setText("1")
                self._annotation_line_width.setText("1.5")
                self._set_combo_value(self._annotation_line_style, "-")
                self._set_combo_value(self._annotation_arrow_style, "->")
                self._annotation_mutation_scale.setText("12")
            finally:
                self._annotation_syncing = False
            self._refresh_annotation_editor_rows()

        def _load_annotation_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._annotations_data):
                self._clear_annotation_editor()
                if getattr(self, "_annotation_preview_summary", None) is not None:
                    self._annotation_preview_summary.setText("No annotation selected.")
                return
            entry = dict(self._annotations_data[index])
            self._annotation_syncing = True
            try:
                self._set_combo_value(
                    self._annotation_enabled_mode,
                    "on" if bool(entry.get("enabled", True)) else "off",
                )
                self._set_combo_value(self._annotation_type, str(entry.get("type") or "text"))
                self._annotation_name.setPlaceholderText(
                    _default_annotation_name(
                        str(entry.get("type") or "text"),
                        index=index + 1,
                    )
                )
                self._annotation_name.setText(str(entry.get("name") or ""))
                self._set_combo_value(
                    self._annotation_coord_system,
                    str(entry.get("coord_system") or "axes"),
                )
                self._annotation_color.setText(str(entry.get("color") or "#000000"))
                self._annotation_alpha.setText(str(entry.get("alpha") or "1.0"))
                self._annotation_zorder.setText(str(entry.get("zorder") or "5"))
                self._annotation_text.setText(str(entry.get("text") or ""))
                self._annotation_x.setText(str(entry.get("x") or "0.5"))
                self._annotation_y.setText(str(entry.get("y") or "0.5"))
                self._annotation_font_size.setText(str(entry.get("font_size") or "12"))
                self._annotation_rotation.setText(str(entry.get("rotation") or "0"))
                self._set_combo_value(
                    self._annotation_horizontal_align,
                    str(entry.get("horizontal_align") or "center"),
                )
                self._set_combo_value(
                    self._annotation_vertical_align,
                    str(entry.get("vertical_align") or "center"),
                )
                self._annotation_x1.setText(str(entry.get("x1") or "0"))
                self._annotation_y1.setText(str(entry.get("y1") or "0"))
                self._annotation_x2.setText(str(entry.get("x2") or "1"))
                self._annotation_y2.setText(str(entry.get("y2") or "1"))
                self._annotation_line_width.setText(str(entry.get("line_width") or "1.5"))
                self._set_combo_value(
                    self._annotation_line_style,
                    str(entry.get("line_style") or "-"),
                )
                self._set_combo_value(
                    self._annotation_arrow_style,
                    str(entry.get("arrow_style") or "->"),
                )
                self._annotation_mutation_scale.setText(str(entry.get("mutation_scale") or "12"))
            finally:
                self._annotation_syncing = False
            self._refresh_annotation_editor_rows()
            self._update_annotation_preview_summary(index)

        def _persist_annotation_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._annotations_data):
                return
            current = dict(self._annotations_data[index])
            current["enabled"] = (
                self._annotation_enabled_mode.currentText().strip().lower() != "off"
            )
            current["type"] = self._annotation_type.currentText().strip().lower() or "text"
            current["name"] = self._annotation_name.text().strip()
            current["coord_system"] = (
                self._annotation_coord_system.currentText().strip().lower() or "axes"
            )
            current["color"] = self._annotation_color.text().strip() or "#000000"
            current["alpha"] = self._annotation_alpha.text().strip() or "1.0"
            current["zorder"] = self._annotation_zorder.text().strip() or "5"
            current["text"] = self._annotation_text.text().strip()
            current["x"] = self._annotation_x.text().strip() or "0.5"
            current["y"] = self._annotation_y.text().strip() or "0.5"
            current["font_size"] = self._annotation_font_size.text().strip() or "12"
            current["rotation"] = self._annotation_rotation.text().strip() or "0"
            current["horizontal_align"] = (
                self._annotation_horizontal_align.currentText().strip().lower() or "center"
            )
            current["vertical_align"] = (
                self._annotation_vertical_align.currentText().strip().lower() or "center"
            )
            current["x1"] = self._annotation_x1.text().strip() or "0"
            current["y1"] = self._annotation_y1.text().strip() or "0"
            current["x2"] = self._annotation_x2.text().strip() or "1"
            current["y2"] = self._annotation_y2.text().strip() or "1"
            current["line_width"] = self._annotation_line_width.text().strip() or "1.5"
            current["line_style"] = self._annotation_line_style.currentText().strip() or "-"
            current["arrow_style"] = self._annotation_arrow_style.currentText().strip() or "->"
            current["mutation_scale"] = self._annotation_mutation_scale.text().strip() or "12"
            self._annotations_data[index] = current

            self._annotation_syncing = True
            try:
                self._refresh_annotation_list()
            finally:
                self._annotation_syncing = False
            self._update_annotation_preview_summary(index)

        def _persist_active_annotation_editor(self) -> None:
            self._persist_annotation_editor(self._annotation_active_index)

        def _handle_annotation_list_selection_change(self, index: int) -> None:
            if self._annotation_syncing:
                return
            self._persist_active_annotation_editor()
            self._annotation_active_index = max(index, 0)
            self._load_annotation_into_editor(index)
            self._refresh_widget_states()

        def _on_annotation_type_changed(self, *_unused: object) -> None:
            if self._annotation_syncing:
                return
            self._refresh_annotation_editor_rows()
            self._on_annotation_editor_changed()

        def _on_annotation_editor_changed(self, *_unused: object) -> None:
            if self._annotation_syncing:
                return
            self._refresh_annotation_editor_rows()
            self._persist_active_annotation_editor()
            self._schedule_preview_update()

        def _add_annotation(self, annotation_type: str) -> None:
            self._persist_active_annotation_editor()
            entry = _annotation_defaults_for_gui(
                annotation_type,
                index=sum(
                    1
                    for value in self._annotations_data
                    if str(value.get("type") or "").strip().lower() == annotation_type
                )
                + 1,
            )
            self._annotations_data.append(entry)
            self._annotation_active_index = len(self._annotations_data) - 1
            self._refresh_annotation_list()
            self._load_annotation_into_editor(self._annotation_active_index)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _duplicate_annotation(self) -> None:
            if not self._annotations_data:
                return
            self._persist_active_annotation_editor()
            source = dict(self._annotations_data[self._annotation_active_index])
            source["id"] = f"annotation:{uuid4().hex}"
            source["name"] = (
                f"{str(source.get('name') or _annotation_type_label(str(source.get('type') or 'text'))).strip()} Copy"
            )
            insert_at = self._annotation_active_index + 1
            self._annotations_data.insert(insert_at, source)
            self._annotation_active_index = insert_at
            self._refresh_annotation_list()
            self._load_annotation_into_editor(insert_at)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _delete_annotation(self) -> None:
            if not self._annotations_data:
                return
            self._annotations_data.pop(self._annotation_active_index)
            if self._annotations_data:
                self._annotation_active_index = min(
                    self._annotation_active_index,
                    len(self._annotations_data) - 1,
                )
                self._refresh_annotation_list()
                self._load_annotation_into_editor(self._annotation_active_index)
            else:
                self._annotation_active_index = 0
                self._refresh_annotation_list()
                self._clear_annotation_editor()
                if getattr(self, "_annotation_preview_summary", None) is not None:
                    self._annotation_preview_summary.setText("No annotations configured.")
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _move_annotation(self, delta: int) -> None:
            if not self._annotations_data or delta == 0:
                return
            current = self._annotation_active_index
            target = min(max(current + delta, 0), len(self._annotations_data) - 1)
            if target == current:
                return
            self._persist_active_annotation_editor()
            moving = self._annotations_data.pop(current)
            self._annotations_data.insert(target, moving)
            self._annotation_active_index = target
            self._refresh_annotation_list()
            self._load_annotation_into_editor(target)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _initialize_annotation_data(self, settings: dict[str, Any]) -> None:
            raw_annotations = settings.get("annotations")
            self._annotations_data = []
            if isinstance(raw_annotations, list):
                for index, value in enumerate(raw_annotations, start=1):
                    self._annotations_data.append(_coerce_annotation_for_gui(value, index=index))
            self._annotation_active_index = 0
            self._refresh_annotation_list()
            if self._annotations_data:
                self._load_annotation_into_editor(0)
            else:
                self._clear_annotation_editor()
                if getattr(self, "_annotation_preview_summary", None) is not None:
                    self._annotation_preview_summary.setText("No annotations configured.")

        def _annotation_summary_for_active(self) -> dict[str, Any] | None:
            if self._annotation_active_index < 0 or self._annotation_active_index >= len(
                self._annotations_data
            ):
                return None
            annotation_id = str(
                self._annotations_data[self._annotation_active_index].get("id")
                or f"annotation:{self._annotation_active_index}"
            )
            raw = self._last_preview_state.get("annotations_summary")
            if not isinstance(raw, list):
                return None
            for item in raw:
                if isinstance(item, dict) and str(item.get("id") or "") == annotation_id:
                    return dict(item)
            return None

        def _update_annotation_preview_summary(self, index: int) -> None:
            if getattr(self, "_annotation_preview_summary", None) is None:
                return
            if index < 0 or index >= len(self._annotations_data):
                self._annotation_preview_summary.setText("No annotation selected.")
                return
            entry = self._annotations_data[index]
            summary = self._annotation_summary_for_active()
            title = _annotation_primary_title(entry, index=index + 1)
            lines = [
                f"Title: {title}",
                f"Type: {_annotation_type_label(str(entry.get('type') or 'text'))}",
                f"Coordinates: {str(entry.get('coord_system') or 'axes')}",
            ]
            if isinstance(summary, dict):
                status = str(summary.get("status") or "ok").strip().lower()
                if status == "ok":
                    lines.append("Preview: Rendered in the current preview.")
                elif status == "disabled":
                    lines.append("Preview: Disabled and hidden from the current preview.")
                else:
                    lines.append(f"Preview: {status}")
            else:
                lines.append("Preview: Waiting for the next rendered preview of this annotation.")
            self._annotation_preview_summary.setText("\n".join(lines))

        def _build_analysis_data_sections(self, layout: QVBoxLayout) -> None:
            analysis = self._analysis_name
            if analysis is None:
                return

            if analysis in {"msd", "position"}:
                selection = QGroupBox("Profile Selection")
                selection_form = QFormLayout(selection)
                self.analysis_species = self._line("Leave blank to use file metadata")
                self.analysis_species.textChanged.connect(self._handle_series_identity_change)
                self._add_form_row(
                    selection_form,
                    "Species",
                    self.analysis_species,
                    tooltip_id="data.profile.species",
                )
                if analysis == "position":
                    self.analysis_axis = self._combo(("", "x", "y", "z"))
                    self.analysis_axis.currentTextChanged.connect(
                        self._handle_series_identity_change
                    )
                    self._add_form_row(
                        selection_form,
                        "Axis",
                        self.analysis_axis,
                        tooltip_id="data.profile.axis",
                    )
                selection_note = QLabel(
                    "Chooses which stored profile is loaded from the current source."
                )
                selection_note.setObjectName("sectionNote")
                selection_note.setWordWrap(True)
                selection_form.addRow(selection_note)
                layout.addWidget(selection)

            if analysis == "rdf":
                selection = QGroupBox("Profile Selection")
                selection_form = QFormLayout(selection)
                rdf_species_a_options = [
                    _PROFILE_FILTER_METADATA_LABEL,
                    *[
                        str(value)
                        for value in self._profile_filter_options.get("species_a", [])
                        if str(value).strip()
                    ],
                ]
                self.rdf_species_a = self._combo(tuple(rdf_species_a_options))
                self.rdf_species_a.currentTextChanged.connect(
                    self._handle_rdf_profile_selection_change
                )
                self.rdf_species_b = self._combo(
                    tuple(self._rdf_species_b_choices(None)),
                )
                self.rdf_species_b.currentTextChanged.connect(self._handle_series_identity_change)
                self._add_form_row(
                    selection_form,
                    "Species A",
                    self.rdf_species_a,
                    tooltip_id="data.rdf.species_a",
                )
                self._add_form_row(
                    selection_form,
                    "Species B",
                    self.rdf_species_b,
                    tooltip_id="data.rdf.species_b",
                )
                selection_note = QLabel(
                    "Chooses which stored RDF profile is loaded from the current source."
                )
                selection_note.setObjectName("sectionNote")
                selection_note.setWordWrap(True)
                selection_form.addRow(selection_note)
                layout.addWidget(selection)

            if analysis == "coordination":
                selection = QGroupBox("Profile Selection")
                selection_form = QFormLayout(selection)
                coord_species_a_options = [
                    _PROFILE_FILTER_METADATA_LABEL,
                    *[
                        str(value)
                        for value in self._profile_filter_options.get("species_a", [])
                        if str(value).strip()
                    ],
                ]
                self.coord_species_a = self._combo(tuple(coord_species_a_options))
                self.coord_species_a.currentTextChanged.connect(
                    self._handle_coordination_profile_selection_change
                )
                self.coord_species_b = self._combo(
                    tuple(self._coordination_species_b_choices(None)),
                )
                self.coord_species_b.currentTextChanged.connect(
                    self._handle_coordination_profile_selection_change
                )
                self.analysis_axis = self._combo(tuple(self._coordination_axis_choices(None, None)))
                self.analysis_axis.currentTextChanged.connect(self._handle_series_identity_change)
                self._add_form_row(
                    selection_form,
                    "Species A",
                    self.coord_species_a,
                    tooltip_id="data.coordination.species_a",
                )
                self._add_form_row(
                    selection_form,
                    "Species B",
                    self.coord_species_b,
                    tooltip_id="data.coordination.species_b",
                )
                self._add_form_row(
                    selection_form,
                    "Axis",
                    self.analysis_axis,
                    tooltip_id="data.coordination.axis",
                )
                note = QLabel(
                    "Chooses which stored coordination profile(s) are loaded from the current source."
                )
                note.setWordWrap(True)
                note.setObjectName("sectionNote")
                selection_form.addRow(note)
                layout.addWidget(selection)

            if analysis == "density":
                view = QGroupBox("Density View")
                view_form = QFormLayout(view)
                self.density_x_mode = self._combo(("distance", "axis"))
                self.density_x_mode.currentTextChanged.connect(self._schedule_preview_update)
                self.density_quantity = self._combo(("mass", "number"))
                self.density_quantity.currentTextChanged.connect(self._schedule_preview_update)
                self._add_form_row(
                    view_form,
                    "X values",
                    self.density_x_mode,
                    tooltip_id="data.density.x_values",
                )
                self._add_form_row(
                    view_form,
                    "Quantity",
                    self.density_quantity,
                    tooltip_id="data.density.quantity",
                )
                layout.addWidget(view)

            if analysis == "potential":
                summary = QGroupBox("Potential Summary")
                summary_form = QFormLayout(summary)
                summary_note = QLabel(
                    "Potential HDF5 plots use record id on the x-axis and plot Water bulk, Fermi, and cSHE as fixed series. Missing values remain as gaps."
                )
                summary_note.setWordWrap(True)
                self._potential_summary_x_axis_label = QLabel("")
                self._potential_summary_total_rows_label = QLabel("")
                self._potential_summary_complete_rows_label = QLabel("")
                self._potential_summary_incomplete_rows_label = QLabel("")
                self._add_form_row(
                    summary_form,
                    "X-axis",
                    self._potential_summary_x_axis_label,
                    tooltip_id="data.potential.x_axis",
                )
                self._add_form_row(
                    summary_form,
                    "Total rows",
                    self._potential_summary_total_rows_label,
                    tooltip_id="data.potential.total_rows",
                )
                self._add_form_row(
                    summary_form,
                    "Complete rows",
                    self._potential_summary_complete_rows_label,
                    tooltip_id="data.potential.complete_rows",
                )
                self._add_form_row(
                    summary_form,
                    "Incomplete rows",
                    self._potential_summary_incomplete_rows_label,
                    tooltip_id="data.potential.incomplete_rows",
                )
                summary_form.addRow("", summary_note)
                layout.addWidget(summary)

            if analysis == "position":
                view = QGroupBox("Position View")
                view_form = QFormLayout(view)
                self.position_component = self._combo(("distance", "x", "y", "z", "xy-z"))
                self.position_component.currentTextChanged.connect(
                    self._handle_series_identity_change
                )
                self.position_map_color = self._combo(("distance", "z"))
                self.position_map_color.currentTextChanged.connect(self._schedule_preview_update)
                self.position_time_axis = self._combo(("ps", "fs", "step", "frame"))
                self.position_time_axis.currentTextChanged.connect(self._schedule_preview_update)
                self._add_form_row(
                    view_form,
                    "Component",
                    self.position_component,
                    tooltip_id="data.position.component",
                )
                self._add_form_row(
                    view_form,
                    "Color by",
                    self.position_map_color,
                    tooltip_id="data.position.color_by",
                )
                self._add_form_row(
                    view_form,
                    "Time axis",
                    self.position_time_axis,
                    tooltip_id="data.position.time_axis",
                )
                self._position_map_color_row = (view_form, self.position_map_color)
                self._position_time_axis_row = (view_form, self.position_time_axis)
                layout.addWidget(view)

            if analysis == "coordination":
                view = QGroupBox("Coordination View")
                view_form = QFormLayout(view)
                self.coordination_component = self._combo(("distance", "time", "time-distance"))
                self.coordination_component.currentTextChanged.connect(
                    self._handle_series_identity_change
                )
                self.coordination_time_axis = self._combo(("ps", "fs", "step", "frame"))
                self.coordination_time_axis.currentTextChanged.connect(
                    self._schedule_preview_update
                )
                self._add_form_row(
                    view_form,
                    "Component",
                    self.coordination_component,
                    tooltip_id="data.coordination.component",
                )
                self._add_form_row(
                    view_form,
                    "Time axis",
                    self.coordination_time_axis,
                    tooltip_id="data.coordination.time_axis",
                )
                self._coordination_time_axis_row = (view_form, self.coordination_time_axis)
                layout.addWidget(view)

            if analysis == "orientation":
                view = QGroupBox("Orientation View")
                view_form = QFormLayout(view)
                self.orientation_component = self._combo(
                    ("average", "density", "density-weighted", "heatmap")
                )
                self.orientation_component.currentTextChanged.connect(
                    self._handle_series_identity_change
                )
                self.orientation_angle = self._combo(("polar", "azimuthal"))
                self.orientation_angle.currentTextChanged.connect(self._schedule_preview_update)
                self._add_form_row(
                    view_form,
                    "Component",
                    self.orientation_component,
                    tooltip_id="data.orientation.component",
                )
                self._add_form_row(
                    view_form,
                    "Angle",
                    self.orientation_angle,
                    tooltip_id="data.orientation.angle",
                )
                layout.addWidget(view)

        def _resize_list_with_defaults(
            self,
            values: list[Any],
            *,
            target_size: int,
            default_factory: Callable[[], Any],
        ) -> list[Any]:
            resized = list(values[:target_size])
            while len(resized) < target_size:
                resized.append(default_factory())
            return resized

        def _apply_series_defaults(
            self,
            labels: list[str],
            *,
            descriptors: list[dict[str, Any]] | None = None,
        ) -> None:
            if not labels and not descriptors:
                return

            selected_series = self._series_active_index
            existing_order = self._current_series_id_order()
            normalized_descriptors = _coerce_series_descriptors(descriptors)
            count = max(len(labels), len(normalized_descriptors))
            if count <= 0:
                return

            existing_by_id: dict[str, dict[str, Any]] = {}
            for index, descriptor in enumerate(self._series_descriptors_data):
                series_id = str(descriptor.get("series_id") or f"series:{index}")
                existing_by_id[series_id] = {
                    "label_override": self._series_label_overrides_data[index]
                    if index < len(self._series_label_overrides_data)
                    else "",
                    "color": self._series_colors_data[index]
                    if index < len(self._series_colors_data)
                    else "",
                    "enabled": self._series_enabled_data[index]
                    if index < len(self._series_enabled_data)
                    else True,
                    "show_in_legend": self._series_show_in_legend_data[index]
                    if index < len(self._series_show_in_legend_data)
                    else True,
                    "alpha": self._series_alpha_data[index]
                    if index < len(self._series_alpha_data)
                    else "",
                    "error_enabled": self._series_error_enabled_data[index]
                    if index < len(self._series_error_enabled_data)
                    else False,
                    "error_stat": self._series_error_stats_data[index]
                    if index < len(self._series_error_stats_data)
                    else "block_sem",
                    "error_style": self._series_error_styles_data[index]
                    if index < len(self._series_error_styles_data)
                    else "band",
                    "error_color": self._series_error_colors_data[index]
                    if index < len(self._series_error_colors_data)
                    else "",
                    "error_label_override": self._series_error_label_overrides_data[index]
                    if index < len(self._series_error_label_overrides_data)
                    else "",
                    "error_show_in_legend": self._series_error_show_in_legend_data[index]
                    if index < len(self._series_error_show_in_legend_data)
                    else False,
                    "fit_enabled": self._series_fit_enabled_data[index]
                    if index < len(self._series_fit_enabled_data)
                    else False,
                    "fit_label_override": self._series_fit_label_overrides_data[index]
                    if index < len(self._series_fit_label_overrides_data)
                    else "",
                    "fit_show_in_legend": self._series_fit_show_in_legend_data[index]
                    if index < len(self._series_fit_show_in_legend_data)
                    else True,
                    "fit_type": self._series_fit_types_data[index]
                    if index < len(self._series_fit_types_data)
                    else "linear",
                    "fit_degree": self._series_fit_degrees_data[index]
                    if index < len(self._series_fit_degrees_data)
                    else "2",
                    "fit_range_mode": self._series_fit_range_modes_data[index]
                    if index < len(self._series_fit_range_modes_data)
                    else "visible",
                    "fit_x_min": self._series_fit_x_mins_data[index]
                    if index < len(self._series_fit_x_mins_data)
                    else "",
                    "fit_x_max": self._series_fit_x_maxs_data[index]
                    if index < len(self._series_fit_x_maxs_data)
                    else "",
                    "cumulative_enabled": self._series_cumulative_enabled_data[index]
                    if index < len(self._series_cumulative_enabled_data)
                    else False,
                    "cumulative_label_override": self._series_cumulative_label_overrides_data[index]
                    if index < len(self._series_cumulative_label_overrides_data)
                    else "",
                    "cumulative_show_in_legend": self._series_cumulative_show_in_legend_data[index]
                    if index < len(self._series_cumulative_show_in_legend_data)
                    else True,
                    "line_width": self._series_line_widths_data[index]
                    if index < len(self._series_line_widths_data)
                    else "",
                    "marker": self._series_markers_data[index]
                    if index < len(self._series_markers_data)
                    else "",
                    "line_kwargs": self._series_line_kwargs_data[index]
                    if index < len(self._series_line_kwargs_data)
                    else "",
                    "normalization_mode": self._series_normalization_modes_data[index]
                    if index < len(self._series_normalization_modes_data)
                    else "none",
                    "normalization_value": self._series_normalization_values_data[index]
                    if index < len(self._series_normalization_values_data)
                    else "",
                    "normalization_x_ref": self._series_normalization_x_refs_data[index]
                    if index < len(self._series_normalization_x_refs_data)
                    else "",
                }

            new_descriptors: list[dict[str, Any]] = []
            new_default_labels: list[str] = []
            new_label_overrides: list[str] = []
            new_colors: list[str] = []
            new_enabled: list[bool] = []
            new_show_in_legend: list[bool] = []
            new_alpha: list[str] = []
            new_error_enabled: list[bool] = []
            new_error_stats: list[str] = []
            new_error_styles: list[str] = []
            new_error_colors: list[str] = []
            new_error_label_overrides: list[str] = []
            new_error_show_in_legend: list[bool] = []
            new_fit_enabled: list[bool] = []
            new_fit_label_overrides: list[str] = []
            new_fit_show_in_legend: list[bool] = []
            new_fit_types: list[str] = []
            new_fit_degrees: list[str] = []
            new_fit_range_modes: list[str] = []
            new_fit_x_mins: list[str] = []
            new_fit_x_maxs: list[str] = []
            new_cumulative_enabled: list[bool] = []
            new_cumulative_label_overrides: list[str] = []
            new_cumulative_show_in_legend: list[bool] = []
            new_widths: list[str] = []
            new_markers: list[str] = []
            new_line_kwargs: list[str] = []
            new_norm_modes: list[str] = []
            new_norm_values: list[str] = []
            new_norm_x_refs: list[str] = []

            for index in range(count):
                descriptor = (
                    dict(normalized_descriptors[index])
                    if index < len(normalized_descriptors)
                    else {"series_id": f"series:{index}"}
                )
                default_label = str(descriptor.get("default_label") or "").strip()
                if not default_label and index < len(labels):
                    default_label = str(labels[index]).strip()
                if not default_label:
                    default_label = f"Series {index + 1}"
                descriptor["series_id"] = str(descriptor.get("series_id") or f"series:{index}")
                descriptor["default_label"] = default_label
                previous = existing_by_id.get(descriptor["series_id"], {})

                new_descriptors.append(descriptor)
                new_default_labels.append(default_label)
                new_label_overrides.append(str(previous.get("label_override") or "").strip())
                new_colors.append(str(previous.get("color") or "").strip())
                new_enabled.append(bool(previous.get("enabled", True)))
                new_show_in_legend.append(bool(previous.get("show_in_legend", True)))
                new_alpha.append(str(previous.get("alpha") or "").strip())
                new_error_enabled.append(bool(previous.get("error_enabled", False)))
                new_error_stats.append(
                    str(previous.get("error_stat") or "block_sem").strip() or "block_sem"
                )
                new_error_styles.append(
                    str(previous.get("error_style") or "band").strip() or "band"
                )
                new_error_colors.append(str(previous.get("error_color") or "").strip())
                new_error_label_overrides.append(
                    str(previous.get("error_label_override") or "").strip()
                )
                new_error_show_in_legend.append(bool(previous.get("error_show_in_legend", False)))
                new_fit_enabled.append(bool(previous.get("fit_enabled", False)))
                new_fit_label_overrides.append(
                    str(previous.get("fit_label_override") or "").strip()
                )
                new_fit_show_in_legend.append(bool(previous.get("fit_show_in_legend", True)))
                new_fit_types.append(str(previous.get("fit_type") or "linear").strip() or "linear")
                new_fit_degrees.append(str(previous.get("fit_degree") or "2").strip() or "2")
                new_fit_range_modes.append(
                    str(previous.get("fit_range_mode") or "visible").strip() or "visible"
                )
                new_fit_x_mins.append(str(previous.get("fit_x_min") or "").strip())
                new_fit_x_maxs.append(str(previous.get("fit_x_max") or "").strip())
                new_cumulative_enabled.append(bool(previous.get("cumulative_enabled", False)))
                new_cumulative_label_overrides.append(
                    str(previous.get("cumulative_label_override") or "").strip()
                )
                new_cumulative_show_in_legend.append(
                    bool(previous.get("cumulative_show_in_legend", True))
                )
                new_widths.append(str(previous.get("line_width") or "").strip())
                new_markers.append(str(previous.get("marker") or "").strip())
                new_line_kwargs.append(str(previous.get("line_kwargs") or "").strip())
                new_norm_modes.append(str(previous.get("normalization_mode") or "none"))
                new_norm_values.append(str(previous.get("normalization_value") or "").strip())
                new_norm_x_refs.append(str(previous.get("normalization_x_ref") or "").strip())

            self._series_descriptors_data = new_descriptors
            self._series_labels_data = new_default_labels
            self._series_label_overrides_data = new_label_overrides
            self._series_colors_data = new_colors
            self._series_enabled_data = new_enabled
            self._series_show_in_legend_data = new_show_in_legend
            self._series_alpha_data = new_alpha
            self._series_error_enabled_data = new_error_enabled
            self._series_error_stats_data = new_error_stats
            self._series_error_styles_data = new_error_styles
            self._series_error_colors_data = new_error_colors
            self._series_error_label_overrides_data = new_error_label_overrides
            self._series_error_show_in_legend_data = new_error_show_in_legend
            self._series_fit_enabled_data = new_fit_enabled
            self._series_fit_label_overrides_data = new_fit_label_overrides
            self._series_fit_show_in_legend_data = new_fit_show_in_legend
            self._series_fit_types_data = new_fit_types
            self._series_fit_degrees_data = new_fit_degrees
            self._series_fit_range_modes_data = new_fit_range_modes
            self._series_fit_x_mins_data = new_fit_x_mins
            self._series_fit_x_maxs_data = new_fit_x_maxs
            self._series_cumulative_enabled_data = new_cumulative_enabled
            self._series_cumulative_label_overrides_data = new_cumulative_label_overrides
            self._series_cumulative_show_in_legend_data = new_cumulative_show_in_legend
            self._series_line_widths_data = new_widths
            self._series_markers_data = new_markers
            self._series_line_kwargs_data = new_line_kwargs
            self._series_normalization_modes_data = new_norm_modes
            self._series_normalization_values_data = new_norm_values
            self._series_normalization_x_refs_data = new_norm_x_refs
            self._series_natural_order_data = [
                str(descriptor.get("series_id") or f"series:{index}")
                for index, descriptor in enumerate(self._series_descriptors_data)
            ]
            self._apply_series_id_order(existing_order)

            next_series_index = min(selected_series, count - 1)

            self._series_syncing = True
            self._normalization_syncing = True
            try:
                self._sync_series_selection_widgets(next_series_index)
                self._series_active_index = next_series_index
                self._load_series_into_editor(next_series_index)
            finally:
                self._normalization_syncing = False
                self._series_syncing = False

            self._update_normalization_warning()

        def _series_display_text(self, index: int) -> str:
            return _format_series_display_text(
                index,
                self._effective_series_label(index),
                enabled=self._series_enabled_data[index],
            )

        def _series_descriptor(self, index: int) -> dict[str, Any]:
            if 0 <= index < len(self._series_descriptors_data):
                return dict(self._series_descriptors_data[index])
            return {
                "series_id": f"series:{index}",
                "default_label": self._series_labels_data[index]
                if 0 <= index < len(self._series_labels_data)
                else f"Series {index + 1}",
                "source_name": "",
                "source_directory": "",
                "source_path": "",
            }

        def _current_series_id_order(self) -> list[str]:
            return [
                str(descriptor.get("series_id") or f"series:{index}")
                for index, descriptor in enumerate(self._series_descriptors_data)
            ]

        def _apply_series_id_order(self, requested_order: list[str] | None) -> None:
            current_ids = self._current_series_id_order()
            resolved_order = _resolve_series_id_order(current_ids, requested_order)
            if resolved_order == current_ids:
                return
            index_by_id = {series_id: index for index, series_id in enumerate(current_ids)}
            indices = [index_by_id[series_id] for series_id in resolved_order]

            def _reorder(values: list[Any]) -> list[Any]:
                return [values[index] for index in indices]

            self._series_descriptors_data = _reorder(self._series_descriptors_data)
            self._series_labels_data = _reorder(self._series_labels_data)
            self._series_label_overrides_data = _reorder(self._series_label_overrides_data)
            self._series_colors_data = _reorder(self._series_colors_data)
            self._series_enabled_data = _reorder(self._series_enabled_data)
            self._series_show_in_legend_data = _reorder(self._series_show_in_legend_data)
            self._series_alpha_data = _reorder(self._series_alpha_data)
            self._series_error_enabled_data = _reorder(self._series_error_enabled_data)
            self._series_error_stats_data = _reorder(self._series_error_stats_data)
            self._series_error_styles_data = _reorder(self._series_error_styles_data)
            self._series_error_label_overrides_data = _reorder(
                self._series_error_label_overrides_data
            )
            self._series_error_show_in_legend_data = _reorder(
                self._series_error_show_in_legend_data
            )
            self._series_fit_enabled_data = _reorder(self._series_fit_enabled_data)
            self._series_fit_label_overrides_data = _reorder(self._series_fit_label_overrides_data)
            self._series_fit_show_in_legend_data = _reorder(self._series_fit_show_in_legend_data)
            self._series_fit_types_data = _reorder(self._series_fit_types_data)
            self._series_fit_degrees_data = _reorder(self._series_fit_degrees_data)
            self._series_fit_range_modes_data = _reorder(self._series_fit_range_modes_data)
            self._series_fit_x_mins_data = _reorder(self._series_fit_x_mins_data)
            self._series_fit_x_maxs_data = _reorder(self._series_fit_x_maxs_data)
            self._series_cumulative_enabled_data = _reorder(self._series_cumulative_enabled_data)
            self._series_cumulative_label_overrides_data = _reorder(
                self._series_cumulative_label_overrides_data
            )
            self._series_cumulative_show_in_legend_data = _reorder(
                self._series_cumulative_show_in_legend_data
            )
            self._series_line_widths_data = _reorder(self._series_line_widths_data)
            self._series_markers_data = _reorder(self._series_markers_data)
            self._series_line_kwargs_data = _reorder(self._series_line_kwargs_data)
            self._series_normalization_modes_data = _reorder(self._series_normalization_modes_data)
            self._series_normalization_values_data = _reorder(
                self._series_normalization_values_data
            )
            self._series_normalization_x_refs_data = _reorder(
                self._series_normalization_x_refs_data
            )

        def _enabled_partitioned_series_id_order(self) -> list[str]:
            current_ids = self._current_series_id_order()
            enabled_by_id = {
                str(descriptor.get("series_id") or f"series:{index}"): bool(
                    self._series_enabled_data[index]
                )
                for index, descriptor in enumerate(self._series_descriptors_data)
                if index < len(self._series_enabled_data)
            }
            return _partition_series_ids_by_enabled_state(current_ids, enabled_by_id)

        def _restore_active_series_from_id(self, selected_id: str) -> None:
            if not selected_id:
                return
            try:
                if selected_id.startswith("cumulative::"):
                    selected_base_id = selected_id.removeprefix("cumulative::")
                    self._series_active_index = self._current_series_id_order().index(
                        selected_base_id
                    )
                    self._set_active_series_child_kind(
                        "cumulative"
                        if (
                            self._series_active_index < len(self._series_cumulative_enabled_data)
                            and bool(
                                self._series_cumulative_enabled_data[self._series_active_index]
                            )
                            and not self._is_orientation_heatmap_mode()
                        )
                        else "base"
                    )
                elif selected_id.startswith("fit::"):
                    selected_base_id = selected_id.removeprefix("fit::")
                    self._series_active_index = self._current_series_id_order().index(
                        selected_base_id
                    )
                    self._set_active_series_child_kind(
                        "fit"
                        if (
                            self._series_active_index < len(self._series_fit_enabled_data)
                            and bool(self._series_fit_enabled_data[self._series_active_index])
                            and self._fit_supported_for_current_view()
                        )
                        else "base"
                    )
                else:
                    self._series_active_index = self._current_series_id_order().index(selected_id)
                    self._set_active_series_child_kind("base")
            except ValueError:
                self._series_active_index = 0
                self._set_active_series_child_kind("base")

        def _effective_series_label(self, index: int) -> str:
            override = ""
            if 0 <= index < len(self._series_label_overrides_data):
                override = self._series_label_overrides_data[index].strip()
            if override:
                return override
            if 0 <= index < len(self._series_labels_data):
                label = self._series_labels_data[index].strip()
                if label:
                    return label
            return f"Series {index + 1}"

        def _effective_series_color(self, index: int) -> str:
            explicit_color = ""
            if 0 <= index < len(self._series_colors_data):
                explicit_color = self._series_colors_data[index].strip()
            if explicit_color:
                return explicit_color
            default_colors = default_series_colors(len(self._series_descriptors_data))
            if 0 <= index < len(default_colors):
                return default_colors[index]
            return ""

        def _fit_supported_for_current_view(self) -> bool:
            analysis = self._analysis_name
            if analysis in {"density", "msd", "rdf", "potential"}:
                return True
            if analysis == "position":
                component = (
                    self.position_component.currentText().strip().lower()
                    if hasattr(self, "position_component")
                    else "distance"
                )
                return component != "xy-z"
            if analysis == "coordination":
                component = (
                    self.coordination_component.currentText().strip().lower()
                    if hasattr(self, "coordination_component")
                    else "distance"
                )
                return component != "time-distance"
            return False

        def _error_supported_for_current_view(self) -> bool:
            return _error_supported_for_view(
                str(self._analysis_name or ""),
                orientation_heatmap=self._is_orientation_heatmap_mode(),
                position_component=(
                    self.position_component.currentText().strip().lower()
                    if hasattr(self, "position_component")
                    else "distance"
                ),
                coordination_component=(
                    self.coordination_component.currentText().strip().lower()
                    if hasattr(self, "coordination_component")
                    else "distance"
                ),
            )

        def _current_plot_family(self) -> str:
            return _plot_family_for_view(
                str(self._analysis_name or ""),
                orientation_heatmap=self._is_orientation_heatmap_mode(),
                position_component=(
                    self.position_component.currentText().strip().lower()
                    if hasattr(self, "position_component")
                    else "distance"
                ),
                coordination_component=(
                    self.coordination_component.currentText().strip().lower()
                    if hasattr(self, "coordination_component")
                    else "distance"
                ),
            )

        def _current_layer_kind(self) -> str:
            if self._layers_tabs is not None and self._layers_tabs.currentIndex() == 1:
                return "annotation"
            if self._series_active_is_cumulative_child:
                return "cumulative"
            if self._series_active_is_fit_child:
                return "fit"
            if self._series_is_group(self._series_active_index):
                return "group"
            return "source"

        def _current_layer_capabilities(self) -> _LayerInspectorCapabilities:
            plot_family = self._current_plot_family()
            layer_kind = self._current_layer_kind()
            is_line_family = plot_family == "line"
            if layer_kind == "annotation":
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=False,
                    show_style=False,
                    show_markers=False,
                    show_derived_lines=False,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_group_members=False,
                    show_metadata=False,
                    show_fit_editor=False,
                    show_cumulative_editor=False,
                )
            if layer_kind == "group":
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=True,
                    show_style=is_line_family,
                    show_markers=is_line_family,
                    show_derived_lines=is_line_family,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_group_members=True,
                    show_metadata=True,
                    show_fit_editor=is_line_family,
                    show_cumulative_editor=is_line_family,
                )
            if layer_kind == "fit":
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=False,
                    show_style=False,
                    show_markers=False,
                    show_derived_lines=True,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_group_members=False,
                    show_metadata=False,
                    show_fit_editor=is_line_family,
                    show_cumulative_editor=False,
                )
            if layer_kind == "cumulative":
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=False,
                    show_style=False,
                    show_markers=False,
                    show_derived_lines=True,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_group_members=False,
                    show_metadata=False,
                    show_fit_editor=False,
                    show_cumulative_editor=is_line_family,
                )
            return _LayerInspectorCapabilities(
                plot_family=plot_family,
                layer_kind=layer_kind,
                show_visibility_label=is_line_family,
                show_style=is_line_family,
                show_markers=is_line_family,
                show_derived_lines=is_line_family,
                show_uncertainty=self._error_supported_for_current_view(),
                show_normalization=is_line_family,
                show_group_members=False,
                show_metadata=True,
                show_fit_editor=is_line_family and self._fit_supported_for_current_view(),
                show_cumulative_editor=is_line_family,
            )

        def _current_figure_capabilities(self) -> _FigureInspectorCapabilities:
            plot_family = self._current_plot_family()
            show_heatmap = plot_family == "heatmap"
            return _FigureInspectorCapabilities(
                plot_family=plot_family,
                show_legend=plot_family == "line",
                show_lines=plot_family == "line",
                show_heatmap=show_heatmap,
                show_colorbar=show_heatmap,
            )

        def _fit_child_series_id(self, index: int) -> str:
            return f"fit::{self._series_descriptor(index).get('series_id') or f'series:{index}'}"

        def _cumulative_child_series_id(self, index: int) -> str:
            return f"cumulative::{self._series_descriptor(index).get('series_id') or f'series:{index}'}"

        def _available_error_stats_for_series(self, index: int) -> list[str]:
            if self._series_is_group(index):
                return []
            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            raw = self._last_preview_state.get("series_available_error_stats")
            resolved: list[str] = []
            values = raw.get(series_id) if isinstance(raw, dict) else None
            if isinstance(values, list):
                for value in values:
                    token = str(value).strip().lower()
                    if token in _ERROR_STATS and token not in resolved:
                        resolved.append(token)
            if resolved:
                return resolved
            summary = self._preview_error_summary_for_series(index)
            if isinstance(summary, dict):
                token = str(summary.get("stat") or "").strip().lower()
                if token in _ERROR_STATS:
                    resolved.append(token)
                if resolved or str(summary.get("status") or "").strip().lower() in {
                    "unavailable",
                    "empty",
                }:
                    return resolved
            return _inferred_available_error_stats(
                analysis_name=self._analysis_name,
                error_supported=self._error_supported_for_current_view(),
                x_bin_width_active=bool(
                    getattr(self, "x_bin_width", None) and self.x_bin_width.text().strip()
                ),
            )

        def _current_error_statistics_mode_for_series(self, index: int) -> str | None:
            if self._series_is_group(index):
                return None
            _ = index
            return _current_error_statistics_mode(
                analysis_name=self._analysis_name,
                error_supported=self._error_supported_for_current_view(),
                x_bin_width_active=bool(
                    getattr(self, "x_bin_width", None) and self.x_bin_width.text().strip()
                ),
            )

        def _current_error_provenance_for_series(self, index: int) -> str:
            return _describe_error_provenance(
                analysis_name=self._analysis_name,
                requested_stat=self._resolved_error_stat_for_series(index),
                statistics_mode=self._current_error_statistics_mode_for_series(index) or "direct",
            )

        def _preview_error_summary_for_series(self, index: int) -> dict[str, Any] | None:
            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            raw = self._last_preview_state.get("series_error_summaries")
            summary = raw.get(series_id) if isinstance(raw, dict) else None
            return summary if isinstance(summary, dict) else None

        def _error_availability_for_series(self, index: int) -> Any:
            if self._series_is_group(index):
                return resolve_series_error_availability(
                    supported_for_view=False,
                    available_stats=[],
                    error_reason="Grouped series do not render error overlays.",
                )
            summary = self._preview_error_summary_for_series(index)
            reason = summary.get("reason") if isinstance(summary, dict) else None
            if reason is None and not self._error_supported_for_current_view():
                reason = "Error overlays are only available for 1-D line-based views."
            return resolve_series_error_availability(
                supported_for_view=self._error_supported_for_current_view(),
                available_stats=self._available_error_stats_for_series(index),
                error_status=summary.get("status") if isinstance(summary, dict) else None,
                error_reason=reason,
            )

        def _default_error_stat_for_series(self, index: int) -> str:
            availability = self._error_availability_for_series(index)
            return str(availability.default_stat or "sample_sem")

        def _resolved_error_stat_for_series(self, index: int) -> str:
            configured = (
                self._series_error_stats_data[index].strip().lower()
                if 0 <= index < len(self._series_error_stats_data)
                else None
            )
            available = self._error_availability_for_series(index).available_stats
            if not available:
                return (
                    configured
                    if configured in _ERROR_STATS
                    else self._default_error_stat_for_series(index)
                )
            return _resolve_error_stat_for_available(
                configured,
                available,
            )

        def _series_error_config(self, index: int) -> dict[str, Any]:
            config = _error_defaults_for_gui()
            if 0 <= index < len(self._series_error_enabled_data):
                config["enabled"] = bool(self._series_error_enabled_data[index])
            config["stat"] = self._resolved_error_stat_for_series(index)
            if 0 <= index < len(self._series_error_styles_data):
                token = self._series_error_styles_data[index].strip().lower()
                if token in _ERROR_STYLES:
                    config["style"] = token
            if 0 <= index < len(self._series_error_colors_data):
                config["color"] = self._series_error_colors_data[index].strip() or None
            if 0 <= index < len(self._series_error_label_overrides_data):
                config["label_override"] = (
                    self._series_error_label_overrides_data[index].strip() or None
                )
            if 0 <= index < len(self._series_error_show_in_legend_data):
                config["show_in_legend"] = bool(self._series_error_show_in_legend_data[index])
            return config

        def _error_effective_label(self, index: int) -> str:
            override = ""
            if 0 <= index < len(self._series_error_label_overrides_data):
                override = self._series_error_label_overrides_data[index].strip()
            return override or _default_error_series_label(
                self._effective_series_label(index),
                self._resolved_error_stat_for_series(index),
            )

        def _cumulative_effective_label(self, index: int) -> str:
            override = ""
            if 0 <= index < len(self._series_cumulative_label_overrides_data):
                override = self._series_cumulative_label_overrides_data[index].strip()
            return override or f"{self._effective_series_label(index)} cumulative average"

        def _active_series_child_kind(self) -> str:
            if self._series_active_is_cumulative_child:
                return "cumulative"
            if self._series_active_is_fit_child:
                return "fit"
            return "base"

        def _set_active_series_child_kind(self, kind: str) -> None:
            normalized = str(kind).strip().lower()
            self._series_active_is_error_child = False
            self._series_active_is_cumulative_child = normalized == "cumulative"
            self._series_active_is_fit_child = normalized == "fit"

        def _active_series_row_id(self) -> str:
            if self._series_active_is_cumulative_child:
                return self._cumulative_child_series_id(self._series_active_index)
            if self._series_active_is_fit_child:
                return self._fit_child_series_id(self._series_active_index)
            return str(self._series_descriptor(self._series_active_index).get("series_id") or "")

        def _series_fit_config(self, index: int) -> dict[str, Any]:
            config = _fit_defaults_for_gui()

            def _soft_float(value: str) -> float | None:
                stripped = str(value).strip()
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError:
                    return None

            if 0 <= index < len(self._series_fit_enabled_data):
                config["fit_enabled"] = bool(self._series_fit_enabled_data[index])
            if 0 <= index < len(self._series_fit_label_overrides_data):
                config["fit_label_override"] = (
                    self._series_fit_label_overrides_data[index].strip() or None
                )
            if 0 <= index < len(self._series_fit_show_in_legend_data):
                config["fit_show_in_legend"] = bool(self._series_fit_show_in_legend_data[index])
            if 0 <= index < len(self._series_fit_types_data):
                token = self._series_fit_types_data[index].strip().lower()
                if token in _FIT_TYPES:
                    config["fit_type"] = token
            if 0 <= index < len(self._series_fit_degrees_data):
                degree = _coerce_degree_text(self._series_fit_degrees_data[index])
                if degree is not None:
                    config["fit_degree"] = degree
            if 0 <= index < len(self._series_fit_range_modes_data):
                token = self._series_fit_range_modes_data[index].strip().lower()
                if token in _FIT_RANGE_MODES:
                    config["fit_range_mode"] = token
            if 0 <= index < len(self._series_fit_x_mins_data):
                config["fit_x_min"] = _soft_float(self._series_fit_x_mins_data[index])
            if 0 <= index < len(self._series_fit_x_maxs_data):
                config["fit_x_max"] = _soft_float(self._series_fit_x_maxs_data[index])
            return config

        def _fit_effective_label(self, index: int) -> str:
            override = ""
            if 0 <= index < len(self._series_fit_label_overrides_data):
                override = self._series_fit_label_overrides_data[index].strip()
            return override or f"{self._effective_series_label(index)} fit"

        def _series_is_group(self, index: int) -> bool:
            descriptor = self._series_descriptor(index)
            return str(descriptor.get("source_kind") or "source").strip().lower() == "group"

        def _group_member_candidate_indices(self) -> list[int]:
            candidates: list[int] = []
            for index, descriptor in enumerate(self._series_descriptors_data):
                if str(descriptor.get("source_kind") or "source").strip().lower() != "group":
                    candidates.append(index)
            return candidates

        def _next_group_label(self) -> str:
            existing = {
                self._effective_series_label(index).strip().lower()
                for index in range(len(self._series_labels_data))
                if self._effective_series_label(index).strip()
            }
            for number in range(1, 1000):
                candidate = f"Group {number}"
                if candidate.strip().lower() not in existing:
                    return candidate
            return f"Group {len(self._series_labels_data) + 1}"

        def _clone_series_at_index(self, index: int) -> None:
            if index < 0 or index >= len(self._series_descriptors_data):
                return
            descriptor = dict(self._series_descriptor(index))
            source_kind = str(descriptor.get("source_kind") or "source").strip().lower()
            base_label = self._effective_series_label(index)
            descriptor["series_id"] = f"{source_kind}:{uuid4().hex}"
            descriptor["default_label"] = f"{base_label} Copy"
            insert_at = index + 1

            def _insert(values: list[Any], copied: Any) -> None:
                values.insert(insert_at, deepcopy(copied))

            for values, copied in (
                (self._series_descriptors_data, descriptor),
                (self._series_labels_data, f"{base_label} Copy"),
                (self._series_label_overrides_data, f"{base_label} Copy"),
                (self._series_colors_data, self._series_colors_data[index]),
                (self._series_enabled_data, self._series_enabled_data[index]),
                (self._series_show_in_legend_data, self._series_show_in_legend_data[index]),
                (self._series_alpha_data, self._series_alpha_data[index]),
                (self._series_error_enabled_data, self._series_error_enabled_data[index]),
                (self._series_error_stats_data, self._series_error_stats_data[index]),
                (self._series_error_styles_data, self._series_error_styles_data[index]),
                (self._series_error_colors_data, self._series_error_colors_data[index]),
                (
                    self._series_error_label_overrides_data,
                    self._series_error_label_overrides_data[index],
                ),
                (
                    self._series_error_show_in_legend_data,
                    self._series_error_show_in_legend_data[index],
                ),
                (self._series_fit_enabled_data, self._series_fit_enabled_data[index]),
                (
                    self._series_fit_label_overrides_data,
                    self._series_fit_label_overrides_data[index],
                ),
                (
                    self._series_fit_show_in_legend_data,
                    self._series_fit_show_in_legend_data[index],
                ),
                (self._series_fit_types_data, self._series_fit_types_data[index]),
                (self._series_fit_degrees_data, self._series_fit_degrees_data[index]),
                (self._series_fit_range_modes_data, self._series_fit_range_modes_data[index]),
                (self._series_fit_x_mins_data, self._series_fit_x_mins_data[index]),
                (self._series_fit_x_maxs_data, self._series_fit_x_maxs_data[index]),
                (self._series_cumulative_enabled_data, self._series_cumulative_enabled_data[index]),
                (
                    self._series_cumulative_label_overrides_data,
                    self._series_cumulative_label_overrides_data[index],
                ),
                (
                    self._series_cumulative_show_in_legend_data,
                    self._series_cumulative_show_in_legend_data[index],
                ),
                (self._series_line_widths_data, self._series_line_widths_data[index]),
                (self._series_markers_data, self._series_markers_data[index]),
                (self._series_line_kwargs_data, self._series_line_kwargs_data[index]),
                (
                    self._series_normalization_modes_data,
                    self._series_normalization_modes_data[index],
                ),
                (
                    self._series_normalization_values_data,
                    self._series_normalization_values_data[index],
                ),
                (
                    self._series_normalization_x_refs_data,
                    self._series_normalization_x_refs_data[index],
                ),
            ):
                _insert(values, copied)
            self._series_active_index = insert_at
            self._set_active_series_child_kind("base")
            self._sync_series_selection_widgets(self._series_active_index)
            self._load_series_into_editor(self._series_active_index)
            self._schedule_preview_update()

        def _duplicate_selected_series(self) -> None:
            self._persist_active_series_editor()
            self._clone_series_at_index(self._series_active_index)

        def _add_group_series(self) -> None:
            self._persist_active_series_editor()
            base_members = [
                str(self._series_descriptor(index).get("series_id") or f"series:{index}")
                for index in self._group_member_candidate_indices()
            ]
            active_series_id = str(
                self._series_descriptor(self._series_active_index).get("series_id")
                or f"series:{self._series_active_index}"
            )
            if active_series_id in base_members:
                member_series_ids = [active_series_id]
            else:
                member_series_ids = base_members[:1]
            descriptor = {
                "series_id": f"group:{uuid4().hex}",
                "default_label": self._next_group_label(),
                "source_kind": "group",
                "member_series_ids": member_series_ids,
                "group_reducer": "mean",
                "source_name": "Grouped series",
                "source_directory": "",
                "source_path": "",
            }
            self._series_descriptors_data.append(descriptor)
            self._series_labels_data.append(str(descriptor["default_label"]))
            self._series_label_overrides_data.append("")
            self._series_colors_data.append("")
            self._series_enabled_data.append(True)
            self._series_show_in_legend_data.append(True)
            self._series_alpha_data.append("")
            self._series_error_enabled_data.append(False)
            self._series_error_stats_data.append("sample_sem")
            self._series_error_styles_data.append("band")
            self._series_error_colors_data.append("")
            self._series_error_label_overrides_data.append("")
            self._series_error_show_in_legend_data.append(False)
            self._series_fit_enabled_data.append(False)
            self._series_fit_label_overrides_data.append("")
            self._series_fit_show_in_legend_data.append(True)
            self._series_fit_types_data.append("linear")
            self._series_fit_degrees_data.append("2")
            self._series_fit_range_modes_data.append("visible")
            self._series_fit_x_mins_data.append("")
            self._series_fit_x_maxs_data.append("")
            self._series_cumulative_enabled_data.append(False)
            self._series_cumulative_label_overrides_data.append("")
            self._series_cumulative_show_in_legend_data.append(True)
            self._series_line_widths_data.append("")
            self._series_markers_data.append("")
            self._series_line_kwargs_data.append("")
            self._series_normalization_modes_data.append("none")
            self._series_normalization_values_data.append("")
            self._series_normalization_x_refs_data.append("")
            self._series_active_index = len(self._series_descriptors_data) - 1
            self._set_active_series_child_kind("base")
            self._sync_series_selection_widgets(self._series_active_index)
            self._load_series_into_editor(self._series_active_index)
            self._schedule_preview_update()

        def _rebuild_series_display_rows(self) -> None:
            rows: list[dict[str, Any]] = []
            for index in range(len(self._series_labels_data)):
                rows.append({"kind": "base", "base_index": index})
                if (
                    index < len(self._series_cumulative_enabled_data)
                    and self._series_cumulative_enabled_data[index]
                    and not self._is_orientation_heatmap_mode()
                ):
                    rows.append({"kind": "cumulative", "base_index": index})
                if (
                    index < len(self._series_fit_enabled_data)
                    and self._series_fit_enabled_data[index]
                    and self._fit_supported_for_current_view()
                ):
                    rows.append({"kind": "fit", "base_index": index})
            self._series_display_rows = rows

        def _display_row(self, row: int) -> dict[str, Any]:
            if 0 <= row < len(self._series_display_rows):
                return dict(self._series_display_rows[row])
            return {"kind": "base", "base_index": max(0, self._series_active_index)}

        def _display_row_for_selection(self, base_index: int, *, kind: str = "base") -> int:
            for row, descriptor in enumerate(self._series_display_rows):
                if (
                    int(descriptor.get("base_index", -1)) == base_index
                    and str(descriptor.get("kind") or "base") == kind
                ):
                    return row
            for row, descriptor in enumerate(self._series_display_rows):
                if int(descriptor.get("base_index", -1)) == base_index:
                    return row
            return 0

        def _display_row_text(self, row: int) -> str:
            descriptor = self._display_row(row)
            base_index = int(descriptor.get("base_index", 0))
            kind = str(descriptor.get("kind") or "base")
            base_label = self._effective_series_label(base_index)
            if self._series_is_group(base_index):
                base_label = f"{base_label} [Group]"
            if kind == "cumulative":
                enabled = base_index < len(self._series_cumulative_enabled_data) and bool(
                    self._series_cumulative_enabled_data[base_index]
                )
                suffix = "" if enabled else " (off)"
                return (
                    f"{base_index + 1}.1: Cumulative - "
                    f"{self._cumulative_effective_label(base_index)}{suffix}"
                )
            if kind == "fit":
                enabled = base_index < len(self._series_fit_enabled_data) and bool(
                    self._series_fit_enabled_data[base_index]
                )
                suffix = "" if enabled else " (off)"
                return f"{base_index + 1}.2: Fit - {self._fit_effective_label(base_index)}{suffix}"
            return _format_series_display_text(
                base_index,
                base_label,
                enabled=self._series_enabled_data[base_index],
            )

        def _series_row_tooltip(self, base_index: int, *, kind: str) -> str:
            descriptor = self._series_descriptor(base_index)
            tooltip_lines = [
                f"Default label: {descriptor.get('default_label') or self._effective_series_label(base_index)}",
                f"Source file: {descriptor.get('source_name') or 'Current session'}",
            ]
            source_directory = str(descriptor.get("source_directory") or "").strip()
            if source_directory:
                tooltip_lines.append(f"Source directory: {source_directory}")
            if kind == "fit":
                fit_type = str(self._series_fit_config(base_index).get("fit_type") or "fit")
                tooltip_lines.append(f"Derived series: {fit_type} fit")
                tooltip_lines.append(f"Series id: {self._fit_child_series_id(base_index)}")
            elif kind == "cumulative":
                tooltip_lines.append("Derived series: cumulative average")
                tooltip_lines.append(f"Series id: {self._cumulative_child_series_id(base_index)}")
            else:
                if self._series_is_group(base_index):
                    tooltip_lines.append(
                        f"Grouped members: {', '.join(self._series_descriptor(base_index).get('member_series_ids', [])) or 'none'}"
                    )
                tooltip_lines.append(
                    f"Series id: {descriptor.get('series_id') or f'series:{base_index}'}"
                )
            return "\n".join(tooltip_lines)

        def _handle_series_row_widget_toggle(self, row: int, checked: bool) -> None:
            if self._series_syncing:
                return
            row_descriptor = self._display_row(row)
            index = int(row_descriptor.get("base_index", -1))
            if index < 0 or index >= len(self._series_enabled_data):
                return
            selected_id = self._active_series_row_id()
            row_kind = str(row_descriptor.get("kind") or "base")
            if row_kind == "fit":
                self._series_fit_enabled_data[index] = checked
                if (
                    index == self._series_active_index
                    and self._series_active_is_fit_child
                    and not checked
                ):
                    self._set_active_series_child_kind("base")
            elif row_kind == "cumulative":
                self._series_cumulative_enabled_data[index] = checked
                if (
                    index == self._series_active_index
                    and self._series_active_is_cumulative_child
                    and not checked
                ):
                    self._set_active_series_child_kind("base")
            else:
                if checked and self._is_orientation_heatmap_mode():
                    for i in range(len(self._series_enabled_data)):
                        if i != index:
                            self._series_enabled_data[i] = False
                self._series_enabled_data[index] = checked
                self._apply_series_id_order(self._enabled_partitioned_series_id_order())
                self._restore_active_series_from_id(selected_id)
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
            finally:
                self._series_syncing = False
            if index == self._series_active_index:
                if self._series_active_is_cumulative_child:
                    self._load_cumulative_series_into_editor(index)
                elif self._series_active_is_fit_child:
                    self._load_fit_series_into_editor(index)
                else:
                    self._load_series_into_editor(index)
            self._refresh_series_list_widgets()
            self._schedule_preview_update()

        def _move_series_by_delta(self, series_id: str, delta: int) -> None:
            if self._series_syncing or delta == 0:
                return
            current_ids = self._current_series_id_order()
            try:
                current_index = current_ids.index(series_id)
            except ValueError:
                return
            target_index = min(max(current_index + delta, 0), len(current_ids) - 1)
            if target_index == current_index:
                return
            selected_id = self._active_series_row_id()
            self._persist_active_series_editor()
            moving_id = current_ids.pop(current_index)
            current_ids.insert(target_index, moving_id)
            self._apply_series_id_order(current_ids)
            self._restore_active_series_from_id(selected_id)
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
                if self._series_active_is_cumulative_child:
                    self._load_cumulative_series_into_editor(self._series_active_index)
                elif self._series_active_is_fit_child:
                    self._load_fit_series_into_editor(self._series_active_index)
                else:
                    self._load_series_into_editor(self._series_active_index)
            finally:
                self._series_syncing = False
            self._schedule_preview_update()

        def _update_series_metadata_panel(self, index: int) -> None:
            descriptor = self._series_descriptor(index)
            if self._series_meta_default_label is not None:
                self._series_meta_default_label.setText(
                    str(descriptor.get("default_label") or f"Series {index + 1}")
                )
            if self._series_meta_source_name is not None:
                self._series_meta_source_name.setText(str(descriptor.get("source_name") or ""))
                source_path = str(descriptor.get("source_path") or "").strip()
                self._series_meta_source_name.setToolTip(source_path)
            if self._series_meta_source_dir is not None:
                source_directory = str(descriptor.get("source_directory") or "").strip()
                self._series_meta_source_dir.setText(source_directory or "Current session")
                self._series_meta_source_dir.setToolTip(str(descriptor.get("source_path") or ""))
            if self._series_meta_series_id is not None:
                self._series_meta_series_id.setText(str(descriptor.get("series_id") or ""))
            if self._series_stats_label is not None:
                fit_stats = None
                if self._series_active_is_fit_child:
                    fit_summary = (
                        self._last_preview_state.get("series_fit_summaries", {}).get(
                            self._series_descriptor(index).get("series_id"),
                            {},
                        )
                        if isinstance(self._last_preview_state.get("series_fit_summaries"), dict)
                        else {}
                    )
                    if isinstance(fit_summary, dict) and fit_summary.get("status") == "ok":
                        y_fit = fit_summary.get("y_fit")
                        if isinstance(y_fit, list) and y_fit:
                            numeric = [float(value) for value in y_fit]
                            array = np.asarray(numeric, dtype=float)
                            fit_stats = {
                                "point_count": int(array.size),
                                "min": float(np.min(array)),
                                "max": float(np.max(array)),
                                "mean": float(np.mean(array)),
                                "std": float(np.std(array, ddof=0)),
                            }
                stats_map = self._last_preview_state.get("series_statistics")
                series_id = str(descriptor.get("series_id") or "")
                stats = fit_stats
                if stats is None and isinstance(stats_map, dict):
                    maybe_stats = stats_map.get(series_id)
                    if isinstance(maybe_stats, dict):
                        stats = maybe_stats
                if not isinstance(stats, dict) or not stats:
                    self._series_stats_label.setText("No series statistics available yet.")
                else:
                    point_count = stats.get("point_count")
                    self._series_stats_label.setText(
                        "\n".join(
                            [
                                f"Points: {point_count if point_count is not None else 'n/a'}",
                                f"Min: {_format_float_value(stats.get('min')) if stats.get('min') is not None else 'n/a'}",
                                f"Max: {_format_float_value(stats.get('max')) if stats.get('max') is not None else 'n/a'}",
                                f"Mean: {_format_float_value(stats.get('mean')) if stats.get('mean') is not None else 'n/a'}",
                                f"Std: {_format_float_value(stats.get('std')) if stats.get('std') is not None else 'n/a'}",
                            ]
                        )
                    )

        def _refresh_series_list_widgets(self) -> None:
            if not hasattr(self, "series_list") or self.series_list is None:
                return
            for index in range(self.series_list.count()):
                self._apply_series_list_item_visuals(self.series_list.item(index), index)

        def _current_series_list_view_anchor(self) -> dict[str, int | str] | None:
            if not hasattr(self, "series_list") or self.series_list is None:
                return None
            scrollbar = self.series_list.verticalScrollBar()
            viewport = self.series_list.viewport()
            viewport_height = viewport.height() if viewport is not None else 0
            rows: list[tuple[str, int, int]] = []
            for row in range(self.series_list.count()):
                item = self.series_list.item(row)
                if item is None:
                    continue
                row_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                if not row_id:
                    continue
                rect = self.series_list.visualItemRect(item)
                rows.append((row_id, int(rect.top()), int(rect.bottom())))
            return _capture_series_list_view_anchor(
                rows,
                viewport_height=viewport_height,
                scroll_value=scrollbar.value(),
            )

        def _restore_series_list_view_anchor(self, anchor: dict[str, int | str] | None) -> None:
            if anchor is None or not hasattr(self, "series_list") or self.series_list is None:
                return
            scrollbar = self.series_list.verticalScrollBar()
            row_tops: dict[str, int] = {}
            for row in range(self.series_list.count()):
                item = self.series_list.item(row)
                if item is None:
                    continue
                row_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                if not row_id:
                    continue
                rect = self.series_list.visualItemRect(item)
                row_tops[row_id] = int(rect.top())
            restored_value = _restore_series_list_anchor_scroll_value(
                anchor,
                row_tops=row_tops,
                current_scroll_value=scrollbar.value(),
                maximum=scrollbar.maximum(),
            )
            if restored_value is not None:
                scrollbar.setValue(restored_value)

        def _refresh_active_series_list_widgets(self) -> None:
            if not hasattr(self, "series_list") or self.series_list is None:
                return
            current_row = self.series_list.currentRow()
            if current_row >= 0:
                self._apply_series_list_item_visuals(
                    self.series_list.item(current_row), current_row
                )
            if (
                self._series_active_index < len(self._series_fit_enabled_data)
                and self._series_fit_enabled_data[self._series_active_index]
                and self._fit_supported_for_current_view()
            ):
                fit_row = self._display_row_for_selection(
                    self._series_active_index,
                    kind="fit",
                )
                if fit_row >= 0 and fit_row != current_row:
                    self._apply_series_list_item_visuals(self.series_list.item(fit_row), fit_row)
            if (
                self._series_active_index < len(self._series_cumulative_enabled_data)
                and self._series_cumulative_enabled_data[self._series_active_index]
                and not self._is_orientation_heatmap_mode()
            ):
                cumulative_row = self._display_row_for_selection(
                    self._series_active_index,
                    kind="cumulative",
                )
                if cumulative_row >= 0 and cumulative_row != current_row:
                    self._apply_series_list_item_visuals(
                        self.series_list.item(cumulative_row), cumulative_row
                    )

        def _apply_series_list_item_visuals(self, item: Any, index: int) -> None:
            if item is None or index < 0:
                return
            row_descriptor = self._display_row(index)
            base_index = int(row_descriptor.get("base_index", -1))
            if base_index < 0 or base_index >= len(self._series_enabled_data):
                return
            kind = str(row_descriptor.get("kind") or "base")
            if kind == "fit":
                enabled = bool(self._series_fit_enabled_data[base_index])
            elif kind == "cumulative":
                enabled = bool(self._series_cumulative_enabled_data[base_index])
            else:
                enabled = self._series_enabled_data[base_index]
            item.setText(self._display_row_text(index).replace("Â·", "-"))
            item.setData(
                Qt.ItemDataRole.UserRole,
                self._fit_child_series_id(base_index)
                if kind == "fit"
                else self._cumulative_child_series_id(base_index)
                if kind == "cumulative"
                else str(
                    self._series_descriptor(base_index).get("series_id") or f"series:{base_index}"
                ),
            )
            item.setData(Qt.ItemDataRole.UserRole + 1, kind)
            item.setToolTip(self._series_row_tooltip(base_index, kind=kind))
            item.setText("")
            row_widget = self.series_list.itemWidget(item)
            if isinstance(row_widget, _SeriesRowWidget):
                heatmap_active = self._is_orientation_heatmap_mode()
                row_widget.update_content(
                    text=self._display_row_text(index).replace("·", "-"),
                    checked=enabled,
                    enabled=enabled,
                    selected=self.series_list.currentRow() == index,
                    color_token="" if heatmap_active else self._effective_series_color(base_index),
                    kind=kind,
                    can_move_up=base_index > 0,
                    can_move_down=base_index < len(self._series_labels_data) - 1,
                    tooltip_text=item.toolTip(),
                )

        def _sync_series_selection_widgets(self, selected_index: int) -> None:
            view_anchor = self._current_series_list_view_anchor()
            self._rebuild_series_display_rows()
            self.series_list.clear()
            for index in range(len(self._series_display_rows)):
                item = QListWidgetItem()
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                )
                row_descriptor = self._display_row(index)
                if str(row_descriptor.get("kind") or "base") == "base":
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                    base_index = int(row_descriptor.get("base_index", 0))
                    base_series_id = str(
                        self._series_descriptor(base_index).get("series_id")
                        or f"series:{base_index}"
                    )
                else:
                    base_index = int(row_descriptor.get("base_index", 0))
                    base_series_id = str(
                        self._series_descriptor(base_index).get("series_id")
                        or f"series:{base_index}"
                    )
                self.series_list.addItem(item)

                def _select_row(row: int = index) -> None:
                    self.series_list.setCurrentRow(row)

                def _toggle_row(checked: bool, row: int = index) -> None:
                    self._handle_series_row_widget_toggle(row, checked)

                def _move_base_up(series_id: str = base_series_id) -> None:
                    self._move_series_by_delta(series_id, -1)

                def _move_base_down(series_id: str = base_series_id) -> None:
                    self._move_series_by_delta(series_id, 1)

                row_widget = _SeriesRowWidget(
                    on_select=_select_row,
                    on_toggle=_toggle_row,
                    on_move_up=_move_base_up,
                    on_move_down=_move_base_down,
                    parent=self.series_list,
                )
                item.setSizeHint(row_widget.sizeHint())
                self.series_list.setItemWidget(item, row_widget)
                self._apply_series_list_item_visuals(item, index)
            if self.series_list.count() > 0:
                row = self._display_row_for_selection(
                    selected_index,
                    kind=self._active_series_child_kind(),
                )
                self.series_list.setCurrentRow(row)
                self._restore_series_list_view_anchor(view_anchor)
                self._refresh_series_list_widgets()

        def _handle_series_identity_change(self, *_unused: object) -> None:
            self._refresh_widget_states()
            if self._suspend_preview_events:
                return
            if self._on_resolve_series_defaults is not None:
                try:
                    settings = self._collect_settings()
                    resolved = self._on_resolve_series_defaults(settings)
                    raw_labels = resolved.get("series_labels")
                    if isinstance(raw_labels, (list, tuple)):
                        labels = [str(label) for label in raw_labels]
                        if labels:
                            raw_descriptors = resolved.get("series_descriptors")
                            descriptors = (
                                list(raw_descriptors)
                                if isinstance(raw_descriptors, (list, tuple))
                                else None
                            )
                            self._apply_series_defaults(labels, descriptors=descriptors)
                except Exception:
                    pass
            self._schedule_preview_update()

        def _build_binning_section(self, layout: QVBoxLayout) -> None:
            binning_title = (
                "Time Sectioning (plot-only)"
                if self._analysis_name == "position"
                else (
                    "Distance / Time Binning (plot-only)"
                    if self._analysis_name == "coordination"
                    else "X Rebinning / Sectioning (plot-only)"
                )
            )
            binning = QGroupBox(binning_title)
            self._data_transform_group = binning
            binning_form = QFormLayout(binning)
            self.x_bin_width = self._line("Leave blank to use the data width")
            self.x_bin_width.textChanged.connect(self._refresh_widget_states)
            self.x_bin_reducer = self._combo(_BIN_REDUCERS)
            width_label = (
                "Time section width" if self._analysis_name == "position" else "Section width"
            )
            self._add_form_row(
                binning_form,
                width_label,
                self.x_bin_width,
                tooltip_id="data.section.width",
            )
            self._add_form_row(
                binning_form,
                "Reducer",
                self.x_bin_reducer,
                tooltip_id="data.section.reducer",
            )
            self._x_bin_reducer_row = (binning_form, self.x_bin_reducer)
            self.min_bin_points = self._line("Leave blank to disable masking")
            self.min_bin_points.textChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                binning_form,
                "Min points per bin",
                self.min_bin_points,
                tooltip_id="data.section.min_points",
            )
            self._min_bin_points_row = (binning_form, self.min_bin_points)

            if self._analysis_name == "orientation":
                self.y_bin_width = self._line("Leave blank to use the data width")
                self.y_bin_width.textChanged.connect(self._refresh_widget_states)
                self.y_bin_reducer = self._combo(_BIN_REDUCERS)
                self._add_form_row(
                    binning_form,
                    "Y section width",
                    self.y_bin_width,
                    tooltip_id="data.section.y_width",
                )
                self._add_form_row(
                    binning_form,
                    "Y reducer",
                    self.y_bin_reducer,
                    tooltip_id="data.section.y_reducer",
                )
                self._y_bin_width_row = (binning_form, self.y_bin_width)
                self._y_bin_reducer_row = (binning_form, self.y_bin_reducer)

            layout.addWidget(binning)

        def _build_data_tab(self) -> None:
            layout = QVBoxLayout(self._tab_data_content)
            self._build_analysis_data_sections(layout)
            self._build_binning_section(layout)
            hint = QLabel(
                "Data controls decide what gets plotted. Layer styling, annotations, uncertainty, and derived lines live in the Layers workspace."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)

        def _build_advanced_tab(self) -> None:
            layout = QVBoxLayout(self._tab_advanced_content)
            guide_link = QLabel(
                '<a href="https://matplotlib.org/stable/users/index.html">Matplotlib user guide</a>'
            )
            guide_link.setOpenExternalLinks(True)
            self._register_tooltip(guide_link, "advanced.matplotlib_guide_link")
            self._apply_widget_tooltip(guide_link)
            layout.addWidget(guide_link)

            def _json_editor(
                placeholder: str,
                *,
                sync_section: str | None = None,
            ) -> QPlainTextEdit:
                editor = QPlainTextEdit()
                editor.setPlaceholderText(placeholder)
                editor.setFixedHeight(84)
                self._configure_horizontal_growth(editor)
                if sync_section is None:
                    editor.textChanged.connect(self._schedule_preview_update)
                else:
                    editor.textChanged.connect(
                        lambda section=sync_section: self._handle_advanced_editor_changed(section)
                    )
                return editor

            rc_group = QGroupBox("Matplotlib rcParams")
            rc_form = QFormLayout(rc_group)
            self.matplotlib_rc_json = _json_editor('{"axes.facecolor": "#f8f8f8"}')
            self._add_form_row(
                rc_form,
                "rcParams (JSON object)",
                self.matplotlib_rc_json,
                tooltip_id="advanced.rcparams",
            )
            layout.addWidget(rc_group)

            render_group = QGroupBox("Figure / Axes / Layout")
            render_form = QFormLayout(render_group)
            self.figure_kwargs_json = _json_editor(
                '{"facecolor": "white"}',
                sync_section="figure_kwargs",
            )
            self.axes_kwargs_json = _json_editor(
                '{"xmargin": 0.02, "ymargin": 0.05}',
                sync_section="axes_kwargs",
            )
            self.tight_layout_kwargs_json = _json_editor('{"pad": 0.6}')
            self.savefig_kwargs_json = _json_editor(
                '{"transparent": false}',
                sync_section="savefig_kwargs",
            )
            self._add_form_row(
                render_form,
                "Figure kwargs",
                self.figure_kwargs_json,
                tooltip_id="advanced.figure_kwargs",
            )
            self._add_form_row(
                render_form,
                "Axes kwargs",
                self.axes_kwargs_json,
                tooltip_id="advanced.axes_kwargs",
            )
            self._add_form_row(
                render_form,
                "tight_layout kwargs",
                self.tight_layout_kwargs_json,
                tooltip_id="advanced.tight_layout_kwargs",
            )
            self._add_form_row(
                render_form,
                "savefig kwargs",
                self.savefig_kwargs_json,
                tooltip_id="advanced.savefig_kwargs",
            )
            layout.addWidget(render_group)

            style_group = QGroupBox("Raw Matplotlib kwargs")
            style_form = QFormLayout(style_group)
            self.legend_kwargs_json = _json_editor(
                '{"frameon": true}',
                sync_section="legend_kwargs",
            )
            self.grid_kwargs_json = _json_editor(
                '{"color": "#dddddd"}',
                sync_section="grid_kwargs",
            )
            self.tick_params_kwargs_json = _json_editor(
                '{"direction": "out"}',
                sync_section="tick_params_kwargs",
            )
            self.line_kwargs_json = _json_editor(
                '{"linestyle": "-", "alpha": 1.0}',
                sync_section="line_kwargs",
            )
            self._add_form_row(
                style_form,
                "Legend kwargs",
                self.legend_kwargs_json,
                tooltip_id="advanced.legend_kwargs",
            )
            self._add_form_row(
                style_form,
                "Grid kwargs",
                self.grid_kwargs_json,
                tooltip_id="advanced.grid_kwargs",
            )
            self._add_form_row(
                style_form,
                "Tick params kwargs",
                self.tick_params_kwargs_json,
                tooltip_id="advanced.tick_params_kwargs",
            )
            self._add_form_row(
                style_form,
                "Global line kwargs",
                self.line_kwargs_json,
                tooltip_id="advanced.line_kwargs",
            )
            layout.addWidget(style_group)

            hint = QLabel(
                "Advanced JSON fields map directly onto Matplotlib API kwargs. "
                "Recognized keys stay synchronized with the standard controls; extra keys remain available here."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)

        def _series_count_from_settings(self, settings: dict[str, Any]) -> int:
            candidates = [1]
            raw_count = settings.get("series_count")
            if isinstance(raw_count, int) and raw_count > 0:
                candidates.append(raw_count)
            raw_descriptors = settings.get("series_descriptors")
            if isinstance(raw_descriptors, (list, tuple)):
                candidates.append(len(raw_descriptors))
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
                raw = settings.get(key)
                if isinstance(raw, (list, tuple)):
                    candidates.append(len(raw))
            return max(candidates)

        def _coerce_series_bool(self, value: Any, *, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            token = str(value).strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
            return default

        def _initialize_series_data(self, settings: dict[str, Any]) -> None:
            count = self._series_count_from_settings(settings)
            descriptors = _coerce_series_descriptors(settings.get("series_descriptors"))
            if descriptors:
                count = max(count, len(descriptors))
            overrides_by_id = _coerce_series_overrides(settings.get("series_overrides"))
            raw_labels = settings.get("series_labels")
            raw_colors = settings.get("line_colors")
            raw_enabled = settings.get("series_enabled")
            raw_show_in_legend = settings.get("series_show_in_legend")
            raw_alpha = settings.get("series_alpha")
            raw_widths = settings.get("series_line_widths")
            raw_markers = settings.get("series_markers")
            raw_line_kwargs = settings.get("series_line_kwargs")
            raw_norm_modes = settings.get("series_normalization_modes")
            raw_norm_values = settings.get("series_normalization_values")
            raw_norm_x_refs = settings.get("series_normalization_x_refs")

            self._series_descriptors_data = []
            self._series_labels_data = []
            self._series_label_overrides_data = []
            self._series_colors_data = []
            self._series_enabled_data = []
            self._series_show_in_legend_data = []
            self._series_alpha_data = []
            self._series_error_enabled_data = []
            self._series_error_stats_data = []
            self._series_error_styles_data = []
            self._series_error_colors_data = []
            self._series_error_label_overrides_data = []
            self._series_error_show_in_legend_data = []
            self._series_fit_enabled_data = []
            self._series_fit_label_overrides_data = []
            self._series_fit_show_in_legend_data = []
            self._series_fit_types_data = []
            self._series_fit_degrees_data = []
            self._series_fit_range_modes_data = []
            self._series_fit_x_mins_data = []
            self._series_fit_x_maxs_data = []
            self._series_cumulative_enabled_data = []
            self._series_cumulative_label_overrides_data = []
            self._series_cumulative_show_in_legend_data = []
            self._series_line_widths_data = []
            self._series_markers_data = []
            self._series_line_kwargs_data = []
            self._series_normalization_modes_data = []
            self._series_normalization_values_data = []
            self._series_normalization_x_refs_data = []

            for index in range(count):
                descriptor = (
                    dict(descriptors[index])
                    if index < len(descriptors)
                    else {"series_id": f"series:{index}"}
                )
                fallback_label = f"Series {index + 1}"
                default_label = str(descriptor.get("default_label") or "").strip()
                if (
                    not default_label
                    and isinstance(raw_labels, (list, tuple))
                    and index < len(raw_labels)
                ):
                    token = str(raw_labels[index]).strip()
                    if token:
                        default_label = token
                if not default_label:
                    default_label = fallback_label
                descriptor["series_id"] = str(descriptor.get("series_id") or f"series:{index}")
                descriptor["default_label"] = default_label
                self._series_descriptors_data.append(descriptor)
                self._series_labels_data.append(default_label)

                override = ""
                series_override = overrides_by_id.get(descriptor["series_id"])
                if isinstance(series_override, dict):
                    raw_override = series_override.get("label_override")
                    if raw_override is not None:
                        override = str(raw_override).strip()
                elif isinstance(raw_labels, (list, tuple)) and index < len(raw_labels):
                    token = str(raw_labels[index]).strip()
                    if token and token != default_label:
                        override = token
                self._series_label_overrides_data.append(override)

                color = ""
                if isinstance(series_override, dict) and series_override.get("color") is not None:
                    color = str(series_override.get("color")).strip()
                elif isinstance(raw_colors, (list, tuple)) and index < len(raw_colors):
                    color = str(raw_colors[index]).strip()
                self._series_colors_data.append(color)

                enabled = True
                if isinstance(series_override, dict) and "enabled" in series_override:
                    enabled = self._coerce_series_bool(series_override.get("enabled"), default=True)
                elif isinstance(raw_enabled, (list, tuple)) and index < len(raw_enabled):
                    enabled = self._coerce_series_bool(raw_enabled[index], default=True)
                self._series_enabled_data.append(enabled)

                show_in_legend = True
                if isinstance(series_override, dict) and "show_in_legend" in series_override:
                    show_in_legend = self._coerce_series_bool(
                        series_override.get("show_in_legend"),
                        default=True,
                    )
                elif isinstance(raw_show_in_legend, (list, tuple)) and index < len(
                    raw_show_in_legend
                ):
                    show_in_legend = self._coerce_series_bool(
                        raw_show_in_legend[index],
                        default=True,
                    )
                self._series_show_in_legend_data.append(show_in_legend)

                alpha = ""
                if isinstance(series_override, dict) and series_override.get("alpha") is not None:
                    alpha = str(series_override.get("alpha")).strip()
                elif isinstance(raw_alpha, (list, tuple)) and index < len(raw_alpha):
                    raw_alpha_value = raw_alpha[index]
                    if raw_alpha_value is not None:
                        alpha = str(raw_alpha_value).strip()
                self._series_alpha_data.append(alpha)

                error_config = _error_defaults_for_gui()
                if isinstance(series_override, dict):
                    error_config = _coerce_series_error_config(series_override.get("error"))
                self._series_error_enabled_data.append(bool(error_config.get("enabled", False)))
                self._series_error_stats_data.append(
                    str(error_config.get("stat") or "block_sem").strip() or "block_sem"
                )
                self._series_error_styles_data.append(
                    str(error_config.get("style") or "band").strip() or "band"
                )
                self._series_error_colors_data.append(str(error_config.get("color") or "").strip())
                self._series_error_label_overrides_data.append(
                    str(error_config.get("label_override") or "").strip()
                )
                self._series_error_show_in_legend_data.append(
                    bool(error_config.get("show_in_legend", False))
                )

                fit_config = _fit_defaults_for_gui()
                if isinstance(series_override, dict):
                    fit_config = _coerce_series_fit_config(
                        series_override.get("fit"),
                    )
                    if "fit_enabled" in series_override:
                        fit_config["fit_enabled"] = self._coerce_series_bool(
                            series_override.get("fit_enabled"),
                            default=bool(fit_config.get("fit_enabled")),
                        )
                    if series_override.get("fit_label_override") is not None:
                        fit_config["fit_label_override"] = str(
                            series_override.get("fit_label_override")
                        ).strip()
                    if "fit_show_in_legend" in series_override:
                        fit_config["fit_show_in_legend"] = self._coerce_series_bool(
                            series_override.get("fit_show_in_legend"),
                            default=bool(fit_config.get("fit_show_in_legend", True)),
                        )
                self._series_fit_enabled_data.append(bool(fit_config.get("fit_enabled", False)))
                self._series_fit_label_overrides_data.append(
                    str(fit_config.get("fit_label_override") or "").strip()
                )
                self._series_fit_show_in_legend_data.append(
                    bool(fit_config.get("fit_show_in_legend", True))
                )
                self._series_fit_types_data.append(
                    str(fit_config.get("fit_type") or "linear").strip() or "linear"
                )
                self._series_fit_degrees_data.append(
                    str(fit_config.get("fit_degree") or "2").strip() or "2"
                )
                self._series_fit_range_modes_data.append(
                    str(fit_config.get("fit_range_mode") or "visible").strip() or "visible"
                )
                self._series_fit_x_mins_data.append(
                    ""
                    if fit_config.get("fit_x_min") is None
                    else str(fit_config.get("fit_x_min")).strip()
                )
                self._series_fit_x_maxs_data.append(
                    ""
                    if fit_config.get("fit_x_max") is None
                    else str(fit_config.get("fit_x_max")).strip()
                )

                cumulative_config = _cumulative_defaults_for_gui()
                if isinstance(series_override, dict):
                    cumulative_config = _coerce_series_cumulative_config(
                        series_override.get("cumulative")
                    )
                self._series_cumulative_enabled_data.append(
                    bool(cumulative_config.get("enabled", False))
                )
                self._series_cumulative_label_overrides_data.append(
                    str(cumulative_config.get("label_override") or "").strip()
                )
                self._series_cumulative_show_in_legend_data.append(
                    bool(cumulative_config.get("show_in_legend", True))
                )

                width = ""
                if (
                    isinstance(series_override, dict)
                    and series_override.get("line_width") is not None
                ):
                    width = str(series_override.get("line_width")).strip()
                elif isinstance(raw_widths, (list, tuple)) and index < len(raw_widths):
                    width = str(raw_widths[index]).strip()
                    if width.lower() == "none":
                        width = ""
                self._series_line_widths_data.append(width)

                marker = ""
                if isinstance(series_override, dict) and series_override.get("marker") is not None:
                    marker = str(series_override.get("marker")).strip()
                elif isinstance(raw_markers, (list, tuple)) and index < len(raw_markers):
                    marker = str(raw_markers[index]).strip()
                    if marker.lower() == "none":
                        marker = ""
                self._series_markers_data.append(marker)

                line_kwargs_text = ""
                if isinstance(series_override, dict) and isinstance(
                    series_override.get("line_kwargs"),
                    dict,
                ):
                    line_kwargs_text = _format_json_block(series_override.get("line_kwargs"))
                elif isinstance(raw_line_kwargs, (list, tuple)) and index < len(raw_line_kwargs):
                    value = raw_line_kwargs[index]
                    if isinstance(value, dict):
                        line_kwargs_text = _format_json_block(value)
                        if not alpha and value.get("alpha") is not None:
                            self._series_alpha_data[index] = str(value.get("alpha")).strip()
                self._series_line_kwargs_data.append(line_kwargs_text)

                mode = "none"
                if isinstance(series_override, dict):
                    token = str(series_override.get("normalization_mode") or "").strip().lower()
                    if token in _NORMALIZATION_MODES:
                        mode = token
                elif isinstance(raw_norm_modes, (list, tuple)) and index < len(raw_norm_modes):
                    token = str(raw_norm_modes[index]).strip().lower()
                    if token in _NORMALIZATION_MODES:
                        mode = token
                self._series_normalization_modes_data.append(mode)

                norm_value = ""
                if (
                    isinstance(series_override, dict)
                    and series_override.get("normalization_value") is not None
                ):
                    norm_value = str(series_override.get("normalization_value")).strip()
                elif isinstance(raw_norm_values, (list, tuple)) and index < len(raw_norm_values):
                    raw_value = raw_norm_values[index]
                    if raw_value is not None:
                        norm_value = str(raw_value).strip()
                self._series_normalization_values_data.append(norm_value)

                norm_x_ref = ""
                if (
                    isinstance(series_override, dict)
                    and series_override.get("normalization_x_ref") is not None
                ):
                    norm_x_ref = str(series_override.get("normalization_x_ref")).strip()
                elif isinstance(raw_norm_x_refs, (list, tuple)) and index < len(raw_norm_x_refs):
                    raw_x_ref = raw_norm_x_refs[index]
                    if raw_x_ref is not None:
                        norm_x_ref = str(raw_x_ref).strip()
                self._series_normalization_x_refs_data.append(norm_x_ref)

            self._series_syncing = True
            try:
                self._series_natural_order_data = [
                    str(descriptor.get("series_id") or f"series:{index}")
                    for index, descriptor in enumerate(self._series_descriptors_data)
                ]
                self._apply_series_id_order(_coerce_series_order(settings.get("series_order")))
                selected = self._series_active_index if self._series_active_index < count else 0
                self._sync_series_selection_widgets(selected)
                self._series_active_index = selected
                self._set_active_series_child_kind("base")
                self._load_series_into_editor(selected)
            finally:
                self._series_syncing = False

        def _load_series_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            self._series_syncing = True
            try:
                self._set_combo_value(
                    self.series_show_in_legend,
                    "on" if self._series_show_in_legend_data[index] else "off",
                )
                self.series_label.setPlaceholderText(self._series_labels_data[index])
                self.series_label.setText(self._series_label_overrides_data[index])
                self.series_color.setText(self._series_colors_data[index])
                self.series_alpha.setText(self._series_alpha_data[index])
                self.series_line_width.setText(self._series_line_widths_data[index])
                self._set_combo_value(self.series_marker, self._series_markers_data[index])
                if hasattr(self, "_series_error_mode"):
                    self._set_combo_value(
                        self._series_error_mode,
                        "on" if self._series_error_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_error_stat"):
                    self._refresh_error_stat_choices(index)
                if hasattr(self, "_series_error_style"):
                    self._set_combo_value(
                        self._series_error_style, self._series_error_styles_data[index]
                    )
                if hasattr(self, "_series_error_color"):
                    self._series_error_color.setText(self._series_error_colors_data[index])
                if hasattr(self, "_series_error_show_in_legend"):
                    self._set_combo_value(
                        self._series_error_show_in_legend,
                        "on" if self._series_error_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_error_label"):
                    self._series_error_label.setPlaceholderText(self._error_effective_label(index))
                    self._series_error_label.setText(self._series_error_label_overrides_data[index])
                if hasattr(self, "_series_cumulative_mode"):
                    self._set_combo_value(
                        self._series_cumulative_mode,
                        "on" if self._series_cumulative_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_cumulative_show_in_legend"):
                    self._set_combo_value(
                        self._series_cumulative_show_in_legend,
                        "on" if self._series_cumulative_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_cumulative_label"):
                    self._series_cumulative_label.setPlaceholderText(
                        self._cumulative_effective_label(index)
                    )
                    self._series_cumulative_label.setText(
                        self._series_cumulative_label_overrides_data[index]
                    )
                if self._series_fit_mode is not None:
                    self._set_combo_value(
                        self._series_fit_mode,
                        "on" if self._series_fit_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_type"):
                    self._set_combo_value(self._series_fit_type, self._series_fit_types_data[index])
                if hasattr(self, "_series_fit_degree"):
                    self._series_fit_degree.setText(self._series_fit_degrees_data[index])
                if hasattr(self, "_series_fit_range_mode"):
                    self._set_combo_value(
                        self._series_fit_range_mode,
                        self._series_fit_range_modes_data[index],
                    )
                if hasattr(self, "_series_fit_x_min"):
                    self._series_fit_x_min.setText(self._series_fit_x_mins_data[index])
                if hasattr(self, "_series_fit_x_max"):
                    self._series_fit_x_max.setText(self._series_fit_x_maxs_data[index])
                if hasattr(self, "_series_fit_show_in_legend"):
                    self._set_combo_value(
                        self._series_fit_show_in_legend,
                        "on" if self._series_fit_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_label"):
                    self._series_fit_label.setPlaceholderText(self._fit_effective_label(index))
                    self._series_fit_label.setText(self._series_fit_label_overrides_data[index])
                if hasattr(self, "_series_group_reducer"):
                    descriptor = self._series_descriptor(index)
                    reducer = (
                        str(descriptor.get("group_reducer") or "mean").strip().lower() or "mean"
                    )
                    self._set_combo_value(self._series_group_reducer, reducer)
                if hasattr(self, "_series_group_members"):
                    self._series_group_members.blockSignals(True)
                    try:
                        self._series_group_members.clear()
                        descriptor = self._series_descriptor(index)
                        selected_ids = {
                            str(value).strip()
                            for value in descriptor.get("member_series_ids", [])
                            if str(value).strip()
                        }
                        for candidate_index in self._group_member_candidate_indices():
                            candidate_descriptor = self._series_descriptor(candidate_index)
                            candidate_id = str(
                                candidate_descriptor.get("series_id") or f"series:{candidate_index}"
                            )
                            item = QListWidgetItem(self._effective_series_label(candidate_index))
                            item.setData(Qt.ItemDataRole.UserRole, candidate_id)
                            item.setSelected(candidate_id in selected_ids)
                            self._series_group_members.addItem(item)
                    finally:
                        self._series_group_members.blockSignals(False)
                self.series_line_kwargs_json.setPlainText(self._series_line_kwargs_data[index])
                self._load_normalization_into_editor(index)
                is_group = self._series_is_group(index)
                for widget in (
                    self.series_color,
                    self.series_alpha,
                    self.series_line_width,
                    self.series_marker,
                    self.series_line_kwargs_json,
                ):
                    widget.setEnabled(True)
                for widget in (
                    self.norm_mode,
                    self.norm_value,
                    self.norm_x_ref,
                ):
                    widget.setEnabled(not is_group)
                if self._series_fit_mode is not None:
                    self._series_fit_mode.setEnabled(self._fit_supported_for_current_view())
                if hasattr(self, "_series_error_mode"):
                    self._series_error_mode.setEnabled(self._error_supported_for_current_view())
                fit_supported = self._fit_supported_for_current_view()
                error_supported = self._error_supported_for_current_view()
                for widget in (
                    getattr(self, "_series_error_stat", None),
                    getattr(self, "_series_error_style", None),
                    getattr(self, "_series_error_color", None),
                    getattr(self, "_series_error_show_in_legend", None),
                    getattr(self, "_series_error_label", None),
                ):
                    if widget is not None:
                        widget.setEnabled(error_supported)
                for widget in (
                    getattr(self, "_series_fit_type", None),
                    getattr(self, "_series_fit_degree", None),
                    getattr(self, "_series_fit_range_mode", None),
                    getattr(self, "_series_fit_x_min", None),
                    getattr(self, "_series_fit_x_max", None),
                    getattr(self, "_series_fit_show_in_legend", None),
                    getattr(self, "_series_fit_label", None),
                ):
                    if widget is not None:
                        widget.setEnabled(fit_supported)
            finally:
                self._series_syncing = False
            self._update_series_metadata_panel(self._series_active_index)
            self._update_series_group_summary(self._series_active_index)
            self._update_series_error_summary(self._series_active_index)
            self._update_series_cumulative_summary(self._series_active_index)
            self._update_series_fit_summary(self._series_active_index)
            if getattr(self, "_series_error_style_note", None) is not None:
                self._series_error_style_note.hide()
            if getattr(self, "_series_cumulative_style_note", None) is not None:
                self._series_cumulative_style_note.hide()
            if self._series_fit_style_note is not None:
                self._series_fit_style_note.hide()
            self._refresh_widget_states()

        def _refresh_error_stat_choices(self, index: int) -> None:
            if not hasattr(self, "_series_error_stat"):
                return
            availability = self._error_availability_for_series(index)
            available = list(availability.available_stats)
            if not available:
                available = [self._default_error_stat_for_series(index)]
            current = self._resolved_error_stat_for_series(index)
            if current not in available:
                current = available[0]
            if 0 <= index < len(self._series_error_stats_data):
                self._series_error_stats_data[index] = current
            self._series_error_stat.blockSignals(True)
            try:
                self._series_error_stat.clear()
                self._series_error_stat.addItems(tuple(available))
                self._set_combo_value(self._series_error_stat, current)
            finally:
                self._series_error_stat.blockSignals(False)

        def _load_fit_series_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            self._series_syncing = True
            try:
                self._set_combo_value(
                    self.series_show_in_legend,
                    "on" if self._series_fit_show_in_legend_data[index] else "off",
                )
                self.series_label.setPlaceholderText(self._fit_effective_label(index))
                self.series_label.setText(self._series_fit_label_overrides_data[index])
                self.series_color.setText(self._effective_series_color(index))
                self.series_alpha.setText(self._series_alpha_data[index])
                self.series_line_width.setText(self._series_line_widths_data[index])
                self._set_combo_value(self.series_marker, "")
                if hasattr(self, "_series_error_mode"):
                    self._set_combo_value(
                        self._series_error_mode,
                        "on" if self._series_error_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_error_stat"):
                    self._refresh_error_stat_choices(index)
                if hasattr(self, "_series_error_style"):
                    self._set_combo_value(
                        self._series_error_style, self._series_error_styles_data[index]
                    )
                if hasattr(self, "_series_error_color"):
                    self._series_error_color.setText(self._series_error_colors_data[index])
                if hasattr(self, "_series_error_show_in_legend"):
                    self._set_combo_value(
                        self._series_error_show_in_legend,
                        "on" if self._series_error_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_error_label"):
                    self._series_error_label.setPlaceholderText(self._error_effective_label(index))
                    self._series_error_label.setText(self._series_error_label_overrides_data[index])
                if self._series_fit_mode is not None:
                    self._set_combo_value(self._series_fit_mode, "on")
                if hasattr(self, "_series_fit_type"):
                    self._set_combo_value(self._series_fit_type, self._series_fit_types_data[index])
                if hasattr(self, "_series_fit_degree"):
                    self._series_fit_degree.setText(self._series_fit_degrees_data[index])
                if hasattr(self, "_series_fit_range_mode"):
                    self._set_combo_value(
                        self._series_fit_range_mode,
                        self._series_fit_range_modes_data[index],
                    )
                if hasattr(self, "_series_fit_x_min"):
                    self._series_fit_x_min.setText(self._series_fit_x_mins_data[index])
                if hasattr(self, "_series_fit_x_max"):
                    self._series_fit_x_max.setText(self._series_fit_x_maxs_data[index])
                if hasattr(self, "_series_fit_show_in_legend"):
                    self._set_combo_value(
                        self._series_fit_show_in_legend,
                        "on" if self._series_fit_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_label"):
                    self._series_fit_label.setPlaceholderText(self._fit_effective_label(index))
                    self._series_fit_label.setText(self._series_fit_label_overrides_data[index])
                self.series_line_kwargs_json.setPlainText(json.dumps({"linestyle": "--"}, indent=2))
                self._load_normalization_into_editor(index)
                for widget in (
                    self.series_color,
                    self.series_alpha,
                    self.series_line_width,
                    self.series_marker,
                    self.series_line_kwargs_json,
                    self.norm_mode,
                    self.norm_value,
                    self.norm_x_ref,
                ):
                    widget.setEnabled(False)
                if self._series_fit_mode is not None:
                    self._series_fit_mode.setEnabled(False)
                for widget in (
                    getattr(self, "_series_fit_type", None),
                    getattr(self, "_series_fit_degree", None),
                    getattr(self, "_series_fit_range_mode", None),
                    getattr(self, "_series_fit_x_min", None),
                    getattr(self, "_series_fit_x_max", None),
                ):
                    if widget is not None:
                        widget.setEnabled(False)
                for widget in (
                    getattr(self, "_series_fit_show_in_legend", None),
                    getattr(self, "_series_fit_label", None),
                ):
                    if widget is not None:
                        widget.setEnabled(True)
            finally:
                self._series_syncing = False
            self._update_series_metadata_panel(index)
            self._update_series_group_summary(index)
            self._update_series_error_summary(index)
            self._update_series_cumulative_summary(index)
            self._update_series_fit_summary(index)
            if getattr(self, "_series_error_style_note", None) is not None:
                self._series_error_style_note.hide()
            if getattr(self, "_series_cumulative_style_note", None) is not None:
                self._series_cumulative_style_note.hide()
            if self._series_fit_style_note is not None:
                self._series_fit_style_note.show()
            self._refresh_widget_states()

        def _load_cumulative_series_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            self._series_syncing = True
            try:
                self._set_combo_value(
                    self.series_show_in_legend,
                    "on" if self._series_cumulative_show_in_legend_data[index] else "off",
                )
                self.series_label.setPlaceholderText(self._cumulative_effective_label(index))
                self.series_label.setText(self._series_cumulative_label_overrides_data[index])
                self.series_color.setText(self._effective_series_color(index))
                self.series_alpha.setText(self._series_alpha_data[index])
                self.series_line_width.setText(self._series_line_widths_data[index])
                self._set_combo_value(self.series_marker, "")
                if hasattr(self, "_series_error_mode"):
                    self._set_combo_value(
                        self._series_error_mode,
                        "on" if self._series_error_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_error_stat"):
                    self._refresh_error_stat_choices(index)
                if hasattr(self, "_series_error_style"):
                    self._set_combo_value(
                        self._series_error_style, self._series_error_styles_data[index]
                    )
                if hasattr(self, "_series_error_color"):
                    self._series_error_color.setText(self._series_error_colors_data[index])
                if hasattr(self, "_series_error_show_in_legend"):
                    self._set_combo_value(
                        self._series_error_show_in_legend,
                        "on" if self._series_error_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_error_label"):
                    self._series_error_label.setPlaceholderText(self._error_effective_label(index))
                    self._series_error_label.setText(self._series_error_label_overrides_data[index])
                if hasattr(self, "_series_cumulative_mode"):
                    self._set_combo_value(self._series_cumulative_mode, "on")
                if hasattr(self, "_series_cumulative_show_in_legend"):
                    self._set_combo_value(
                        self._series_cumulative_show_in_legend,
                        "on" if self._series_cumulative_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_cumulative_label"):
                    self._series_cumulative_label.setPlaceholderText(
                        self._cumulative_effective_label(index)
                    )
                    self._series_cumulative_label.setText(
                        self._series_cumulative_label_overrides_data[index]
                    )
                if self._series_fit_mode is not None:
                    self._set_combo_value(
                        self._series_fit_mode,
                        "on" if self._series_fit_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_type"):
                    self._set_combo_value(self._series_fit_type, self._series_fit_types_data[index])
                if hasattr(self, "_series_fit_degree"):
                    self._series_fit_degree.setText(self._series_fit_degrees_data[index])
                if hasattr(self, "_series_fit_range_mode"):
                    self._set_combo_value(
                        self._series_fit_range_mode,
                        self._series_fit_range_modes_data[index],
                    )
                if hasattr(self, "_series_fit_x_min"):
                    self._series_fit_x_min.setText(self._series_fit_x_mins_data[index])
                if hasattr(self, "_series_fit_x_max"):
                    self._series_fit_x_max.setText(self._series_fit_x_maxs_data[index])
                if hasattr(self, "_series_fit_show_in_legend"):
                    self._set_combo_value(
                        self._series_fit_show_in_legend,
                        "on" if self._series_fit_show_in_legend_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_label"):
                    self._series_fit_label.setPlaceholderText(self._fit_effective_label(index))
                    self._series_fit_label.setText(self._series_fit_label_overrides_data[index])
                self.series_line_kwargs_json.setPlainText(json.dumps({"linestyle": ":"}, indent=2))
                self._load_normalization_into_editor(index)
                for widget in (
                    self.series_color,
                    self.series_alpha,
                    self.series_line_width,
                    self.series_marker,
                    self.series_line_kwargs_json,
                    self.norm_mode,
                    self.norm_value,
                    self.norm_x_ref,
                ):
                    widget.setEnabled(False)
                if hasattr(self, "_series_cumulative_mode"):
                    self._series_cumulative_mode.setEnabled(False)
                if hasattr(self, "_series_cumulative_show_in_legend"):
                    self._series_cumulative_show_in_legend.setEnabled(True)
                if hasattr(self, "_series_cumulative_label"):
                    self._series_cumulative_label.setEnabled(True)
            finally:
                self._series_syncing = False
            self._update_series_metadata_panel(index)
            self._update_series_group_summary(index)
            self._update_series_error_summary(index)
            self._update_series_cumulative_summary(index)
            self._update_series_fit_summary(index)
            if getattr(self, "_series_error_style_note", None) is not None:
                self._series_error_style_note.hide()
            if getattr(self, "_series_cumulative_style_note", None) is not None:
                self._series_cumulative_style_note.show()
            if self._series_fit_style_note is not None:
                self._series_fit_style_note.hide()
            self._refresh_widget_states()

        def _persist_series_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            label_value = self.series_label.text().strip()
            self._series_label_overrides_data[index] = label_value
            self._series_colors_data[index] = self.series_color.text().strip()
            self._series_show_in_legend_data[index] = (
                self.series_show_in_legend.currentText().strip().lower() != "off"
            )
            self._series_alpha_data[index] = self.series_alpha.text().strip()
            if hasattr(self, "_series_error_mode"):
                self._series_error_enabled_data[index] = (
                    self._series_error_mode.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_error_stat"):
                token = self._series_error_stat.currentText().strip().lower()
                self._series_error_stats_data[index] = (
                    token if token in _ERROR_STATS else self._default_error_stat_for_series(index)
                )
            if hasattr(self, "_series_error_style"):
                token = self._series_error_style.currentText().strip().lower()
                self._series_error_styles_data[index] = token if token in _ERROR_STYLES else "band"
            if hasattr(self, "_series_error_color"):
                self._series_error_colors_data[index] = self._series_error_color.text().strip()
            if hasattr(self, "_series_error_show_in_legend"):
                self._series_error_show_in_legend_data[index] = (
                    self._series_error_show_in_legend.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_error_label"):
                self._series_error_label_overrides_data[index] = (
                    self._series_error_label.text().strip()
                )
            if hasattr(self, "_series_cumulative_mode"):
                self._series_cumulative_enabled_data[index] = (
                    self._series_cumulative_mode.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_cumulative_show_in_legend"):
                self._series_cumulative_show_in_legend_data[index] = (
                    self._series_cumulative_show_in_legend.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_cumulative_label"):
                self._series_cumulative_label_overrides_data[index] = (
                    self._series_cumulative_label.text().strip()
                )
            if self._series_fit_mode is not None:
                self._series_fit_enabled_data[index] = (
                    self._series_fit_mode.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_fit_type"):
                self._series_fit_types_data[index] = (
                    self._series_fit_type.currentText().strip().lower() or "linear"
                )
            if hasattr(self, "_series_fit_degree"):
                self._series_fit_degrees_data[index] = self._series_fit_degree.text().strip() or "2"
            if hasattr(self, "_series_fit_range_mode"):
                self._series_fit_range_modes_data[index] = (
                    self._series_fit_range_mode.currentText().strip().lower() or "visible"
                )
            if hasattr(self, "_series_fit_x_min"):
                self._series_fit_x_mins_data[index] = self._series_fit_x_min.text().strip()
            if hasattr(self, "_series_fit_x_max"):
                self._series_fit_x_maxs_data[index] = self._series_fit_x_max.text().strip()
            if hasattr(self, "_series_fit_show_in_legend"):
                self._series_fit_show_in_legend_data[index] = (
                    self._series_fit_show_in_legend.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_fit_label"):
                self._series_fit_label_overrides_data[index] = self._series_fit_label.text().strip()
            self._series_line_widths_data[index] = self.series_line_width.text().strip()
            self._series_markers_data[index] = self.series_marker.currentText().strip()
            self._series_line_kwargs_data[index] = (
                self.series_line_kwargs_json.toPlainText().strip()
            )
            if self._series_is_group(index):
                descriptor = dict(self._series_descriptors_data[index])
                if hasattr(self, "_series_group_reducer"):
                    descriptor["group_reducer"] = (
                        self._series_group_reducer.currentText().strip().lower() or "mean"
                    )
                if hasattr(self, "_series_group_members"):
                    descriptor["member_series_ids"] = [
                        str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                        for item in self._series_group_members.selectedItems()
                        if str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                    ]
                self._series_descriptors_data[index] = descriptor
            self._persist_normalization_editor(index)
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
            finally:
                self._series_syncing = False
            self._update_series_metadata_panel(index)
            self._update_series_group_summary(index)
            self._update_series_error_summary(index)
            self._update_series_cumulative_summary(index)
            self._update_series_fit_summary(index)

        def _persist_fit_series_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            if hasattr(self, "_series_fit_show_in_legend"):
                self._series_fit_show_in_legend_data[index] = (
                    self._series_fit_show_in_legend.currentText().strip().lower() != "off"
                )
            else:
                self._series_fit_show_in_legend_data[index] = (
                    self.series_show_in_legend.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_fit_label"):
                self._series_fit_label_overrides_data[index] = self._series_fit_label.text().strip()
            else:
                self._series_fit_label_overrides_data[index] = self.series_label.text().strip()
            selection_is_fit_child = (
                self._series_fit_enabled_data[index] and self._fit_supported_for_current_view()
            )
            self._set_active_series_child_kind("fit" if selection_is_fit_child else "base")
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
                if selection_is_fit_child:
                    self._load_fit_series_into_editor(index)
                else:
                    self._load_series_into_editor(index)
            finally:
                self._series_syncing = False
            self._update_series_error_summary(index)
            self._update_series_cumulative_summary(index)
            self._update_series_fit_summary(index)

        def _persist_cumulative_series_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            if hasattr(self, "_series_cumulative_show_in_legend"):
                self._series_cumulative_show_in_legend_data[index] = (
                    self._series_cumulative_show_in_legend.currentText().strip().lower() != "off"
                )
            else:
                self._series_cumulative_show_in_legend_data[index] = (
                    self.series_show_in_legend.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_cumulative_label"):
                self._series_cumulative_label_overrides_data[index] = (
                    self._series_cumulative_label.text().strip()
                )
            else:
                self._series_cumulative_label_overrides_data[index] = (
                    self.series_label.text().strip()
                )
            selection_is_cumulative_child = bool(self._series_cumulative_enabled_data[index]) and (
                not self._is_orientation_heatmap_mode()
            )
            self._set_active_series_child_kind(
                "cumulative" if selection_is_cumulative_child else "base"
            )
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
                if selection_is_cumulative_child:
                    self._load_cumulative_series_into_editor(index)
                else:
                    self._load_series_into_editor(index)
            finally:
                self._series_syncing = False
            self._update_series_error_summary(index)
            self._update_series_cumulative_summary(index)
            self._update_series_fit_summary(index)

        def _persist_active_series_editor(self) -> None:
            if self._series_active_is_cumulative_child:
                self._persist_cumulative_series_editor(self._series_active_index)
            elif self._series_active_is_fit_child:
                self._persist_fit_series_editor(self._series_active_index)
            else:
                self._persist_series_editor(self._series_active_index)

        def _handle_series_list_selection_change(self, index: int) -> None:
            if self._series_syncing or index < 0:
                return
            selected_descriptor = self._display_row(index)
            target_base_index = int(selected_descriptor.get("base_index", 0))
            target_kind = str(selected_descriptor.get("kind") or "base")
            self._persist_active_series_editor()
            self._series_active_index = target_base_index
            if target_kind == "fit":
                self._set_active_series_child_kind(
                    "fit"
                    if (
                        target_base_index < len(self._series_fit_enabled_data)
                        and bool(self._series_fit_enabled_data[target_base_index])
                        and self._fit_supported_for_current_view()
                    )
                    else "base"
                )
            elif target_kind == "cumulative":
                self._set_active_series_child_kind(
                    "cumulative"
                    if (
                        target_base_index < len(self._series_cumulative_enabled_data)
                        and bool(self._series_cumulative_enabled_data[target_base_index])
                        and not self._is_orientation_heatmap_mode()
                    )
                    else "base"
                )
            else:
                self._set_active_series_child_kind("base")
            self._series_syncing = True
            try:
                self.series_list.setCurrentRow(
                    self._display_row_for_selection(
                        self._series_active_index,
                        kind=self._active_series_child_kind(),
                    )
                )
            finally:
                self._series_syncing = False
            if self._series_active_is_cumulative_child:
                self._load_cumulative_series_into_editor(self._series_active_index)
            elif self._series_active_is_fit_child:
                self._load_fit_series_into_editor(self._series_active_index)
            else:
                self._load_series_into_editor(self._series_active_index)
            self._refresh_series_list_widgets()

        def _handle_series_list_rows_moved(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            selected_item = self.series_list.currentItem()
            selected_id = (
                str(selected_item.data(Qt.ItemDataRole.UserRole)).strip()
                if selected_item is not None
                else ""
            )
            desired_order: list[str] = []
            for row in range(self.series_list.count()):
                item = self.series_list.item(row)
                if item is None:
                    continue
                item_id = str(item.data(Qt.ItemDataRole.UserRole)).strip()
                if item_id.startswith("fit::") or not item_id:
                    continue
                if item_id not in desired_order:
                    desired_order.append(item_id)
            self._persist_active_series_editor()
            self._apply_series_id_order(desired_order)
            self._restore_active_series_from_id(selected_id)
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
                if self._series_active_is_fit_child:
                    self._load_fit_series_into_editor(self._series_active_index)
                else:
                    self._load_series_into_editor(self._series_active_index)
            finally:
                self._series_syncing = False
            self._schedule_preview_update()

        def _set_all_series_enabled(self, enabled: bool) -> None:
            if not self._series_enabled_data:
                return
            self._persist_active_series_editor()
            self._series_enabled_data = [enabled] * len(self._series_enabled_data)
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
            finally:
                self._series_syncing = False
            self._refresh_series_list_widgets()
            self._schedule_preview_update()

        def _on_series_editor_changed(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            self._persist_active_series_editor()
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _on_series_label_changed(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            index = self._series_active_index
            if index < 0 or index >= len(self._series_labels_data):
                return
            if self._series_active_is_fit_child:
                self._series_fit_label_overrides_data[index] = self.series_label.text().strip()
            else:
                self._series_label_overrides_data[index] = self.series_label.text().strip()
                if hasattr(self, "_series_fit_label") and self._series_fit_label is not None:
                    self._series_fit_label.setPlaceholderText(self._fit_effective_label(index))
                if hasattr(self, "_series_error_label") and self._series_error_label is not None:
                    self._series_error_label.setPlaceholderText(self._error_effective_label(index))
            self._refresh_active_series_list_widgets()
            self._schedule_preview_update()

        def _on_series_fit_label_changed(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            index = self._series_active_index
            if index < 0 or index >= len(self._series_fit_label_overrides_data):
                return
            self._series_fit_label_overrides_data[index] = self._series_fit_label.text().strip()
            self._refresh_active_series_list_widgets()
            self._schedule_preview_update()

        def _initialize_normalization_data(self, settings: dict[str, Any]) -> None:
            count = len(self._series_labels_data)
            if (
                settings.get("series_overrides") is not None
                and len(self._series_normalization_modes_data) == count
                and len(self._series_normalization_values_data) == count
                and len(self._series_normalization_x_refs_data) == count
            ):
                self._normalization_syncing = True
                try:
                    self._load_normalization_into_editor(self._series_active_index)
                finally:
                    self._normalization_syncing = False
                self._update_normalization_warning()
                return
            raw_modes = settings.get("series_normalization_modes")
            raw_values = settings.get("series_normalization_values")
            raw_x_refs = settings.get("series_normalization_x_refs")

            self._series_normalization_modes_data = []
            self._series_normalization_values_data = []
            self._series_normalization_x_refs_data = []

            for index in range(count):
                mode = "none"
                if isinstance(raw_modes, (list, tuple)) and index < len(raw_modes):
                    token = str(raw_modes[index]).strip().lower()
                    if token in _NORMALIZATION_MODES:
                        mode = token
                value = ""
                if isinstance(raw_values, (list, tuple)) and index < len(raw_values):
                    raw_value = raw_values[index]
                    if raw_value is not None:
                        value = str(raw_value).strip()
                x_ref = ""
                if isinstance(raw_x_refs, (list, tuple)) and index < len(raw_x_refs):
                    raw_x_ref = raw_x_refs[index]
                    if raw_x_ref is not None:
                        x_ref = str(raw_x_ref).strip()
                self._series_normalization_modes_data.append(mode)
                self._series_normalization_values_data.append(value)
                self._series_normalization_x_refs_data.append(x_ref)

            self._normalization_syncing = True
            try:
                self._load_normalization_into_editor(self._series_active_index)
            finally:
                self._normalization_syncing = False
            self._update_normalization_warning()

        def _load_normalization_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_normalization_modes_data):
                return
            self._normalization_syncing = True
            try:
                self._set_combo_value(self.norm_mode, self._series_normalization_modes_data[index])
                self.norm_value.setText(self._series_normalization_values_data[index])
                self.norm_x_ref.setText(self._series_normalization_x_refs_data[index])
            finally:
                self._normalization_syncing = False

        def _persist_normalization_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_normalization_modes_data):
                return
            mode = self.norm_mode.currentText().strip().lower()
            if mode not in _NORMALIZATION_MODES:
                mode = "none"
            self._series_normalization_modes_data[index] = mode
            self._series_normalization_values_data[index] = self.norm_value.text().strip()
            self._series_normalization_x_refs_data[index] = self.norm_x_ref.text().strip()
            self._update_normalization_warning()

        def _copy_normalization_settings_to_all_series(self) -> None:
            if self._series_active_is_fit_child:
                self._status_label.setText("Normalization is edited on the base series only.")
                return
            if not self._series_normalization_modes_data:
                return
            self._persist_normalization_editor(self._series_active_index)
            mode = self._series_normalization_modes_data[self._series_active_index]
            value = self._series_normalization_values_data[self._series_active_index]
            x_ref = self._series_normalization_x_refs_data[self._series_active_index]
            for index in range(len(self._series_normalization_modes_data)):
                self._series_normalization_modes_data[index] = mode
                self._series_normalization_values_data[index] = value
                self._series_normalization_x_refs_data[index] = x_ref
            self._load_normalization_into_editor(self._series_active_index)
            self._update_normalization_warning()
            self._schedule_preview_update()
            self._status_label.setText("Copied normalization settings to all base series.")

        def _on_normalization_editor_changed(self, *_unused: object) -> None:
            if self._normalization_syncing:
                return
            self._persist_normalization_editor(self._series_active_index)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _update_normalization_warning(self) -> None:
            if not hasattr(self, "normalization_warning"):
                return
            visible_modes = [
                mode
                for index, mode in enumerate(self._series_normalization_modes_data)
                if index >= len(self._series_enabled_data) or self._series_enabled_data[index]
            ]
            series_count = len(visible_modes)
            normalized_count = sum(1 for mode in visible_modes if mode != "none")
            if series_count > 1 and 0 < normalized_count < series_count:
                self.normalization_warning.setText(
                    "Warning: only part of the plotted series is normalized. "
                    "Interpret y-axis comparisons carefully."
                )
                self.normalization_warning.show()
                return
            self.normalization_warning.setText("")
            self.normalization_warning.hide()

        def _update_potential_summary_panel(self, settings: dict[str, Any]) -> None:
            if self._analysis_name != "potential":
                return
            summary = settings.get("potential_summary")
            if not isinstance(summary, dict):
                summary = self._profile_filter_options.get("potential_summary")
            if not isinstance(summary, dict):
                summary = {}
            if self._potential_summary_x_axis_label is not None:
                self._potential_summary_x_axis_label.setText(
                    str(summary.get("x_axis_label") or "Record ID")
                )
            if self._potential_summary_total_rows_label is not None:
                total_rows = summary.get("total_rows")
                self._potential_summary_total_rows_label.setText(
                    "" if total_rows is None else str(total_rows)
                )
            if self._potential_summary_complete_rows_label is not None:
                complete_rows = summary.get("complete_rows")
                self._potential_summary_complete_rows_label.setText(
                    "" if complete_rows is None else str(complete_rows)
                )
            if self._potential_summary_incomplete_rows_label is not None:
                incomplete_rows = summary.get("incomplete_rows")
                self._potential_summary_incomplete_rows_label.setText(
                    "" if incomplete_rows is None else str(incomplete_rows)
                )

        def _update_series_fit_summary_unused(self, index: int) -> None:
            if self._series_fit_summary is None or self._series_fit_warning is None:
                return
            if index < 0 or index >= len(self._series_labels_data):
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_summary.setText("No fit available for this selection.")
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.hide()
                return

            fit_enabled = index < len(self._series_fit_enabled_data) and bool(
                self._series_fit_enabled_data[index]
            )
            if not fit_enabled:
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_summary.setText("No fit configured for this series.")
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.hide()
                return

            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            fit_summaries = self._last_preview_state.get("series_fit_summaries")
            summary = fit_summaries.get(series_id) if isinstance(fit_summaries, dict) else None
            if not isinstance(summary, dict):
                self._series_fit_summary.show()
                self._series_fit_summary.setText(
                    "Fit summary will appear after the next preview refresh."
                )
                self._series_fit_warning.setText("")
                self._series_fit_warning.hide()
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.show()
                return

            status = str(summary.get("status") or "").strip().lower()
            if status == "ok":
                slope = summary.get("slope")
                intercept = summary.get("intercept")
                r_squared = summary.get("r_squared")
                point_count = summary.get("point_count")
                self._series_fit_summary.setText(
                    "\n".join(
                        [
                            f"Slope: {_format_float_value(slope)}",
                            f"Intercept: {_format_float_value(intercept)}",
                            f"R²: {_format_float_value(r_squared)}",
                            f"Valid points: {point_count}",
                        ]
                    )
                )
                self._series_fit_summary.setText(
                    self._series_fit_summary.text().replace("Â²", "^2")
                )
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.show()
                return

            if status in {"off", "disabled"}:
                self._series_fit_summary.hide()
                self._series_fit_warning.hide()
                self._series_fit_summary.setText("")
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.hide()
                return

            reason = str(summary.get("reason") or "Linear fit is not available for this series.")
            self._series_fit_summary.hide()
            self._series_fit_warning.setText(reason)
            self._series_fit_warning.show()
            if self._series_fit_style_note is not None:
                self._series_fit_style_note.show()

        def _update_series_cumulative_summary(self, index: int) -> None:
            summary_label = getattr(self, "_series_cumulative_summary", None)
            if summary_label is None:
                return
            if index < 0 or index >= len(self._series_labels_data):
                summary_label.setText("No cumulative-average line available for this selection.")
                summary_label.show()
                if getattr(self, "_series_cumulative_style_note", None) is not None:
                    self._series_cumulative_style_note.hide()
                return

            if self._is_orientation_heatmap_mode():
                summary_label.setText(
                    "Cumulative-average lines are only available for 1-D line-based views."
                )
                summary_label.show()
                if getattr(self, "_series_cumulative_style_note", None) is not None:
                    self._series_cumulative_style_note.hide()
                return

            enabled = index < len(self._series_cumulative_enabled_data) and bool(
                self._series_cumulative_enabled_data[index]
            )
            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            raw = self._last_preview_state.get("series_cumulative_summaries")
            summary = raw.get(series_id) if isinstance(raw, dict) else None
            if not enabled:
                summary_label.setText("No cumulative-average line configured for this series.")
                summary_label.show()
                if getattr(self, "_series_cumulative_style_note", None) is not None:
                    self._series_cumulative_style_note.hide()
                return
            if not isinstance(summary, dict):
                summary_label.setText(
                    "Cumulative-average line is enabled. Render details will update with the preview."
                )
                summary_label.show()
                if getattr(self, "_series_cumulative_style_note", None) is not None:
                    self._series_cumulative_style_note.show()
                return
            status = str(summary.get("status") or "").strip().lower()
            if status == "ok":
                summary_label.setText(
                    "\n".join(
                        [
                            f"Label: {summary.get('label') or self._cumulative_effective_label(index)}",
                            f"Visible points: {summary.get('point_count') if summary.get('point_count') is not None else 'n/a'}",
                            "Running mean of the currently displayed y-values ordered by x.",
                        ]
                    )
                )
                summary_label.show()
                if getattr(self, "_series_cumulative_style_note", None) is not None:
                    self._series_cumulative_style_note.show()
                return
            if status == "empty":
                summary_label.setText(
                    "Cumulative-average line is enabled, but no finite plotted points remain."
                )
                summary_label.show()
                if getattr(self, "_series_cumulative_style_note", None) is not None:
                    self._series_cumulative_style_note.show()
                return
            summary_label.setText("Cumulative-average line is disabled for this series.")
            summary_label.show()
            if getattr(self, "_series_cumulative_style_note", None) is not None:
                self._series_cumulative_style_note.hide()

        def _update_series_group_summary(self, index: int) -> None:
            summary_label = getattr(self, "_series_group_summary", None)
            if summary_label is None:
                return
            if index < 0 or index >= len(self._series_descriptors_data):
                summary_label.setText("No grouped-series settings available.")
                return
            descriptor = self._series_descriptor(index)
            if not self._series_is_group(index):
                summary_label.setText("This series is not grouped.")
                return
            member_ids = [
                str(value).strip()
                for value in descriptor.get("member_series_ids", [])
                if str(value).strip()
            ]
            group_map = self._last_preview_state.get("series_group_summaries")
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            summary = group_map.get(series_id) if isinstance(group_map, dict) else None
            reducer = str(descriptor.get("group_reducer") or "mean")
            lines = [
                f"Reducer: {reducer}",
                f"Members: {len(member_ids)}",
            ]
            if isinstance(summary, dict):
                reason = str(summary.get("reason") or "").strip()
                status = str(summary.get("status") or "").strip().lower()
                if status == "ok":
                    lines.append("Grouped line rendered from the current member series.")
                elif reason:
                    lines.append(reason)
            elif not member_ids:
                lines.append("Select at least one member series.")
            summary_label.setText("\n".join(lines))

        def _update_series_error_summary(self, index: int) -> None:
            if (
                getattr(self, "_series_error_summary", None) is None
                or getattr(self, "_series_error_warning", None) is None
            ):
                return
            explanation_label = getattr(self, "_series_error_explanation", None)
            if index < 0 or index >= len(self._series_labels_data):
                self._series_error_summary.setText("No error overlay available for this selection.")
                self._series_error_summary.show()
                if explanation_label is not None:
                    explanation_label.setText("")
                    explanation_label.hide()
                self._series_error_warning.hide()
                self._series_error_warning.setText("")
                if getattr(self, "_series_error_style_note", None) is not None:
                    self._series_error_style_note.hide()
                return

            if not self._error_supported_for_current_view():
                self._series_error_summary.setText(
                    "Error overlays are only available for 1-D line-based views."
                )
                self._series_error_summary.show()
                if explanation_label is not None:
                    explanation_label.setText("")
                    explanation_label.hide()
                self._series_error_warning.hide()
                self._series_error_warning.setText("")
                if getattr(self, "_series_error_style_note", None) is not None:
                    self._series_error_style_note.hide()
                return

            descriptor = self._series_descriptor(index)
            summary = self._preview_error_summary_for_series(index)
            availability = self._error_availability_for_series(index)
            available = list(availability.available_stats)
            masked_map = self._last_preview_state.get("series_masked_bin_counts")
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            masked_count = masked_map.get(series_id) if isinstance(masked_map, dict) else None
            if not isinstance(summary, dict):
                available_text = ", ".join(available) if available else "none yet"
                configured_stat = self._series_error_stats_data[index]
                resolved_stat = self._resolved_error_stat_for_series(index)
                statistics_mode = self._current_error_statistics_mode_for_series(index)
                statistics_mode_text = {
                    "direct": "Stored direct statistics",
                    "raw_grouped": "Raw grouped statistics",
                    "saved_rebinned_sample": "Rebinned sample statistics from stored data",
                }.get(statistics_mode or "", "Not yet rendered")
                lines = [
                    "Rendered uncertainty details are not available for the current preview state yet.",
                    f"Configured statistic: {configured_stat or 'sample_sem'}",
                    f"Effective statistic: {resolved_stat}",
                    f"Statistics mode: {statistics_mode_text}",
                    f"Available stats: {available_text}",
                ]
                if availability.reason:
                    lines.append(f"Availability: {availability.reason}")
                self._series_error_summary.setText("\n".join(lines))
                self._series_error_summary.show()
                if explanation_label is not None:
                    provenance = self._current_error_provenance_for_series(index)
                    if provenance:
                        explanation_label.setText(f"What it shows: {provenance}")
                        explanation_label.show()
                    else:
                        explanation_label.setText("")
                        explanation_label.hide()
                self._series_error_warning.hide()
                self._series_error_warning.setText("")
                if getattr(self, "_series_error_style_note", None) is not None:
                    self._series_error_style_note.show()
                return

            status = str(summary.get("status") or "").strip().lower()
            if status == "ok":
                statistics_mode = str(summary.get("statistics_mode") or "").strip().lower()
                statistics_mode_text = {
                    "direct": "Stored direct statistics",
                    "raw_grouped": "Raw grouped statistics",
                    "saved_rebinned_sample": "Rebinned sample statistics from stored data",
                }.get(statistics_mode, "Statistics mode unavailable")
                lines = [
                    f"Statistic: {summary.get('stat') or self._resolved_error_stat_for_series(index)}",
                    f"Style: {summary.get('style') or self._series_error_styles_data[index]}",
                    f"Color: {summary.get('color') or self._series_error_colors_data[index] or 'Follow series color'}",
                    f"Statistics mode: {statistics_mode_text}",
                    f"Visible bins: {summary.get('point_count') if summary.get('point_count') is not None else 'n/a'}",
                ]
                reason = str(summary.get("reason") or "").strip()
                if reason:
                    lines.append(reason)
                if masked_count is not None:
                    lines.append(f"Masked bins: {masked_count}")
                if available:
                    lines.append(f"Available stats: {', '.join(available)}")
                self._series_error_summary.setText("\n".join(lines))
                self._series_error_summary.show()
                provenance = str(summary.get("provenance") or "").strip()
                if explanation_label is not None:
                    if provenance:
                        explanation_label.setText(f"What it shows: {provenance}")
                        explanation_label.show()
                    else:
                        explanation_label.setText("")
                        explanation_label.hide()
                self._series_error_warning.hide()
                self._series_error_warning.setText("")
                if getattr(self, "_series_error_style_note", None) is not None:
                    self._series_error_style_note.show()
                return

            if status == "off":
                available_text = ", ".join(available) if available else "none"
                self._series_error_summary.setText(
                    "\n".join(
                        [
                            "No error overlay is currently rendered for this series.",
                            f"Available stats: {available_text}",
                        ]
                    )
                )
                self._series_error_summary.show()
                provenance = str(summary.get("provenance") or "").strip()
                if explanation_label is not None:
                    if provenance:
                        explanation_label.setText(f"What it would show: {provenance}")
                        explanation_label.show()
                    else:
                        explanation_label.setText("")
                        explanation_label.hide()
                self._series_error_warning.hide()
                self._series_error_warning.setText("")
                if getattr(self, "_series_error_style_note", None) is not None:
                    self._series_error_style_note.hide()
                return

            if status == "disabled":
                reason = str(
                    summary.get("reason") or "Error overlay is not active for this series."
                )
                self._series_error_summary.setText(reason)
                self._series_error_summary.show()
                provenance = str(summary.get("provenance") or "").strip()
                if explanation_label is not None:
                    if provenance:
                        explanation_label.setText(f"What it would show: {provenance}")
                        explanation_label.show()
                    else:
                        explanation_label.setText("")
                        explanation_label.hide()
                self._series_error_warning.hide()
                self._series_error_warning.setText("")
                if getattr(self, "_series_error_style_note", None) is not None:
                    self._series_error_style_note.show()
                return

            reason = str(summary.get("reason") or "Error overlay is unavailable for this series.")
            self._series_error_summary.hide()
            if explanation_label is not None:
                explanation_label.setText("")
                explanation_label.hide()
            self._series_error_warning.setText(reason)
            self._series_error_warning.show()
            if getattr(self, "_series_error_style_note", None) is not None:
                self._series_error_style_note.show()

        def _update_series_fit_summary(self, index: int) -> None:
            if self._series_fit_summary is None or self._series_fit_warning is None:
                return
            if index < 0 or index >= len(self._series_labels_data):
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_summary.setText("No fit available for this selection.")
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.hide()
                return

            fit_enabled = index < len(self._series_fit_enabled_data) and bool(
                self._series_fit_enabled_data[index]
            )
            if not fit_enabled:
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_summary.setText("No fit configured for this series.")
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.hide()
                return

            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or f"series:{index}")
            fit_summaries = self._last_preview_state.get("series_fit_summaries")
            summary = fit_summaries.get(series_id) if isinstance(fit_summaries, dict) else None
            if not isinstance(summary, dict):
                self._series_fit_summary.show()
                self._series_fit_summary.setText(
                    "Fit summary will appear after the next preview refresh."
                )
                self._series_fit_warning.setText("")
                self._series_fit_warning.hide()
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.show()
                return

            status = str(summary.get("status") or "").strip().lower()
            if status == "ok":
                fit_type = str(summary.get("fit_type") or "fit").strip()
                equation = str(summary.get("equation") or "").strip()
                parameters = summary.get("parameters")
                parameter_order = summary.get("parameter_order")
                r_squared = summary.get("r_squared")
                rmse = summary.get("rmse")
                fit_point_count = summary.get("fit_point_count")
                display_point_count = summary.get("display_point_count")
                lines = [f"Type: {fit_type}"]
                if equation:
                    lines.append(f"Equation: {equation}")
                if isinstance(parameters, dict) and isinstance(parameter_order, list):
                    for key in parameter_order:
                        if key in parameters:
                            lines.append(f"{key}: {_format_float_value(parameters[key])}")
                elif isinstance(parameters, dict):
                    for key, value in parameters.items():
                        lines.append(f"{key}: {_format_float_value(value)}")
                if r_squared is not None:
                    lines.append(f"R^2: {_format_float_value(r_squared)}")
                if rmse is not None:
                    lines.append(f"RMSE: {_format_float_value(rmse)}")
                lines.append(f"Fit points: {fit_point_count}")
                lines.append(f"Displayed points: {display_point_count}")
                self._series_fit_summary.setText("\n".join(lines))
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.show()
                return

            if status in {"off", "disabled"}:
                self._series_fit_summary.show()
                self._series_fit_warning.hide()
                self._series_fit_summary.setText("No fit is currently rendered for this series.")
                self._series_fit_warning.setText("")
                if self._series_fit_style_note is not None:
                    self._series_fit_style_note.hide()
                return

            reason = str(summary.get("reason") or "Fit is not available for this series.")
            self._series_fit_summary.show()
            self._series_fit_summary.setText("No valid fit is currently available.")
            self._series_fit_warning.setText(reason)
            self._series_fit_warning.show()
            if self._series_fit_style_note is not None:
                self._series_fit_style_note.show()

        def _bind_live_preview_signals(self) -> None:
            line_widgets = (
                self.title_text,
                self.x_label,
                self.y_label,
                self.legend_title,
                self.legend_columns,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.x_ticks,
                self.y_ticks,
                self.x_tick_rotation,
                self.y_tick_rotation,
                self.x_label_pad,
                self.y_label_pad,
                self.fig_width,
                self.fig_height,
                self.dpi,
                self.font_family,
                self.base_font_size,
                self.title_font,
                self.label_font,
                self.tick_font,
                self.legend_font,
                self.line_width,
                self.line_alpha,
                self.marker_size,
                self.figure_facecolor,
                self.grid_linewidth,
                self.grid_alpha,
                self.grid_color,
                self.tick_length,
                self.tick_width,
                self.x_bin_width,
                self.min_bin_points,
            )
            for widget in line_widgets:
                widget.textChanged.connect(self._schedule_preview_update)
            if hasattr(self, "y_bin_width"):
                self.y_bin_width.textChanged.connect(self._schedule_preview_update)

            combo_widgets = (
                self.legend_mode,
                self.legend_loc,
                self.legend_frame_mode,
                self.ticks_visibility,
                self.grid_mode,
                self.markers_mode,
                self.x_scale,
                self.y_scale,
                self.transparent_mode,
                self.line_style,
                self.grid_linestyle,
                self.grid_axis,
                self.grid_which,
                self.tick_direction,
                self.minor_ticks_mode,
                self.x_bin_reducer,
                self.axes_border_mode,
            )
            for widget in combo_widgets:
                widget.currentTextChanged.connect(self._schedule_preview_update)
            if hasattr(self, "y_bin_reducer"):
                self.y_bin_reducer.currentTextChanged.connect(self._schedule_preview_update)
            for cb in (self.border_left, self.border_right, self.border_top, self.border_bottom):
                cb.toggled.connect(self._schedule_preview_update)

        def _handle_auto_preview_toggle(self, checked: bool) -> None:
            if not checked:
                self._preview_timer.stop()
                self._preview_status.setText("Auto update paused.")
                self._refresh_shell_state()
                return
            self._preview_status.setText("Auto update enabled.")
            self._refresh_shell_state()
            self._schedule_preview_update()

        def _schedule_preview_update(self, *_unused: object) -> None:
            if self._suspend_preview_events:
                return
            if not self._auto_preview_checkbox.isChecked():
                return
            if self._preview_loading:
                return
            self._preview_timer.start(_AUTO_PREVIEW_DEBOUNCE_MS)

        def _handle_debounced_preview(self) -> None:
            if self._preview_loading:
                return
            self._update_embedded_preview(interactive=False)

        def _set_preview_loading(self, active: bool) -> None:
            self._preview_loading = bool(active)
            for widget in (
                self._preview_frame,
                self._preview_scroll,
                self._preview_scroll.viewport() if self._preview_scroll is not None else None,
                self._preview_label,
            ):
                if widget is None:
                    continue
                if active:
                    widget.setCursor(Qt.CursorShape.WaitCursor)
                else:
                    widget.unsetCursor()
            if self._preview_button is not None:
                self._preview_button.setEnabled(
                    _preview_button_enabled(
                        auto_update_enabled=self._auto_preview_checkbox.isChecked(),
                        preview_loading=self._preview_loading,
                    )
                )

        def _set_preview_zoom(self, value: float) -> None:
            self._preview_zoom_factor = max(0.2, min(20.0, float(value)))

        def _zoom_preview_at_viewport_pos(self, viewport_pos: Any, *, direction: int) -> bool:
            if (
                self._preview_pixmap is None
                or self._preview_pixmap.isNull()
                or self._preview_scroll is None
                or self._preview_label is None
            ):
                return False
            step = 1.12 if direction > 0 else (1.0 / 1.12)
            old_zoom = self._preview_zoom_factor
            self._set_preview_zoom(old_zoom * step)
            if abs(self._preview_zoom_factor - old_zoom) < 1.0e-9:
                return False

            hbar = self._preview_scroll.horizontalScrollBar()
            vbar = self._preview_scroll.verticalScrollBar()
            old_width = max(1, self._preview_label.width())
            old_height = max(1, self._preview_label.height())
            if hasattr(viewport_pos, "x") and hasattr(viewport_pos, "y"):
                cursor_x = float(viewport_pos.x())
                cursor_y = float(viewport_pos.y())
            else:
                cursor_x = 0.0
                cursor_y = 0.0

            relative_x = (hbar.value() + cursor_x) / old_width
            relative_y = (vbar.value() + cursor_y) / old_height

            self._refresh_preview_pixmap()

            new_width = max(1, self._preview_label.width())
            new_height = max(1, self._preview_label.height())
            hbar.setValue(int(relative_x * new_width - cursor_x))
            vbar.setValue(int(relative_y * new_height - cursor_y))
            return True

        def _refresh_preview_pixmap(self) -> None:
            if (
                self._preview_pixmap is None
                or self._preview_pixmap.isNull()
                or self._preview_scroll is None
                or self._preview_label is None
            ):
                return
            target_size = self._preview_scroll.viewport().size()
            if target_size.width() < 2 or target_size.height() < 2:
                return

            source_size = self._preview_pixmap.size()
            fit_scale = min(
                target_size.width() / max(1, source_size.width()),
                target_size.height() / max(1, source_size.height()),
            )
            if fit_scale <= 0.0:
                return
            effective_scale = fit_scale * self._preview_zoom_factor
            scaled = self._preview_pixmap.scaled(
                max(1, int(source_size.width() * effective_scale)),
                max(1, int(source_size.height() * effective_scale)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)
            self._preview_label.resize(scaled.size())

        def _update_embedded_preview(self, *, interactive: bool) -> bool:
            if self._preview_loading:
                return False
            self._preview_timer.stop()
            self._set_preview_loading(True)
            if on_save_figure is None:
                try:
                    settings = self._collect_settings()
                    render_state = on_preview(settings)
                    if isinstance(render_state, dict) and render_state:
                        self._apply_preview_state_to_synced_fields(render_state)
                    self._preview_error = None
                    self._preview_status.setText("External preview opened.")
                    self._refresh_shell_state()
                    return True
                except Exception as exc:
                    if interactive:
                        self._report_error("Preview failed", exc)
                    else:
                        self._preview_error = str(exc)
                        self._preview_status.setText("Preview paused.")
                        self._refresh_shell_state()
                    return False
                finally:
                    self._set_preview_loading(False)

            try:
                settings = self._collect_settings()
                save_result = on_save_figure(settings, str(self._preview_image_path))
                render_state = None
                if isinstance(save_result, tuple):
                    _message, render_state = save_result
                elif isinstance(save_result, dict):
                    render_state = save_result
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_state_to_synced_fields(render_state)
                pixmap = QPixmap(str(self._preview_image_path))
                if pixmap.isNull():
                    raise RuntimeError("Could not load rendered preview image.")
                self._preview_pixmap = pixmap
                self._refresh_preview_pixmap()
                self._preview_error = None
                self._preview_status.setText("Preview updated.")
                self._refresh_shell_state()
                return True
            except Exception as exc:
                if interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_error = str(exc)
                    self._preview_status.setText("Preview paused.")
                    self._refresh_shell_state()
                return False
            finally:
                self._set_preview_loading(False)

        def _set_combo_value(self, widget: QComboBox, value: str) -> None:
            if not value:
                if widget.count() > 0:
                    widget.setCurrentIndex(0)
                return
            index = widget.findText(value)
            if index < 0:
                lowered = value.strip().casefold()
                for candidate_index in range(widget.count()):
                    candidate = widget.itemText(candidate_index).strip().casefold()
                    if candidate == lowered:
                        index = candidate_index
                        break
            if index < 0:
                if widget.isEditable():
                    widget.setEditText(value)
                elif widget.count() > 0:
                    widget.setCurrentIndex(0)
                return
            widget.setCurrentIndex(index)

        def _set_combo_items(
            self,
            widget: QComboBox,
            values: list[str],
            *,
            preferred_value: str | None = None,
        ) -> None:
            current_value = widget.currentText()
            target_value = current_value if preferred_value is None else preferred_value
            widget.blockSignals(True)
            try:
                widget.clear()
                widget.addItems(values)
                self._set_combo_value(widget, target_value)
            finally:
                widget.blockSignals(False)

        def _profile_filter_display_value(self, value: str | None, *, default_label: str) -> str:
            token = str(value or "").strip()
            return token or default_label

        def _selected_profile_filter_value(
            self,
            widget: QComboBox,
            *,
            default_label: str,
        ) -> str | None:
            token = widget.currentText().strip()
            return None if token == default_label or token == "" else token

        def _rdf_species_b_choices(self, species_a: str | None) -> list[str]:
            mapping = self._profile_filter_options.get("species_b_by_species_a", {})
            if not isinstance(mapping, dict):
                return [_PROFILE_FILTER_SPECIES_B_AUTO_LABEL]
            key = "" if species_a is None else str(species_a)
            values = mapping.get(key)
            if not isinstance(values, list):
                values = mapping.get("", [])
            resolved = [str(value).strip() for value in values if str(value).strip()]
            return [_PROFILE_FILTER_SPECIES_B_AUTO_LABEL, *resolved]

        def _coordination_species_b_choices(self, species_a: str | None) -> list[str]:
            return self._rdf_species_b_choices(species_a)

        def _coordination_axis_choices(
            self,
            species_a: str | None,
            species_b: str | None,
        ) -> list[str]:
            axes_by_pair = self._profile_filter_options.get("axes_by_species_pair", {})
            if not isinstance(axes_by_pair, dict):
                return ["", "x", "y", "z"]
            species_a_key = "" if species_a is None else str(species_a)
            species_b_key = "" if species_b is None else str(species_b)
            axis_values: list[str] | None = None
            pair_mapping = axes_by_pair.get(species_a_key)
            if isinstance(pair_mapping, dict):
                candidate = pair_mapping.get(species_b_key)
                if isinstance(candidate, list):
                    axis_values = candidate
            if axis_values is None:
                global_mapping = axes_by_pair.get("", {})
                if isinstance(global_mapping, dict):
                    candidate = global_mapping.get("", [])
                    if isinstance(candidate, list):
                        axis_values = candidate
            resolved = [
                str(value).strip().lower() for value in (axis_values or []) if str(value).strip()
            ]
            resolved = [value for value in resolved if value in {"x", "y", "z"}]
            return ["", *resolved] if resolved else ["", "x", "y", "z"]

        def _handle_rdf_profile_selection_change(self, *_unused: object) -> None:
            if not hasattr(self, "rdf_species_a") or not hasattr(self, "rdf_species_b"):
                return
            species_a = self._selected_profile_filter_value(
                self.rdf_species_a,
                default_label=_PROFILE_FILTER_METADATA_LABEL,
            )
            current_species_b = self.rdf_species_b.currentText()
            self._set_combo_items(
                self.rdf_species_b,
                self._rdf_species_b_choices(species_a),
                preferred_value=current_species_b,
            )
            self._handle_series_identity_change()

        def _handle_coordination_profile_selection_change(self, *_unused: object) -> None:
            if not hasattr(self, "coord_species_a") or not hasattr(self, "coord_species_b"):
                return
            species_a = self._selected_profile_filter_value(
                self.coord_species_a,
                default_label=_PROFILE_FILTER_METADATA_LABEL,
            )
            current_species_b = self.coord_species_b.currentText()
            self._set_combo_items(
                self.coord_species_b,
                self._coordination_species_b_choices(species_a),
                preferred_value=current_species_b,
            )
            if hasattr(self, "analysis_axis"):
                species_b = self._selected_profile_filter_value(
                    self.coord_species_b,
                    default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                )
                current_axis = self.analysis_axis.currentText()
                self._set_combo_items(
                    self.analysis_axis,
                    self._coordination_axis_choices(species_a, species_b),
                    preferred_value=current_axis,
                )
            self._handle_series_identity_change()

        def _populate(self, settings: dict[str, Any]) -> None:
            synced_locks = _derive_synced_field_locks(settings)
            synced_modes = _derive_synced_field_modes(settings)
            title_visible = settings.get("title_visible")
            if title_visible is False:
                self.title_text.setText("")
            else:
                self.title_text.setText(str(settings.get("title") or ""))
            self.x_label.setText(str(settings.get("x_label") or ""))
            self.y_label.setText(str(settings.get("y_label") or ""))

            self._set_combo_value(
                self.legend_mode,
                _toggle_to_mode(settings.get("legend"), auto_mode="on"),
            )
            self.legend_title.setText(str(settings.get("legend_title") or ""))
            self._set_combo_value(self.legend_loc, str(settings.get("legend_loc") or "best"))
            self._set_combo_value(
                self.legend_frame_mode,
                _extract_dict_mode(
                    settings,
                    key="legend_kwargs",
                    nested_key="frameon",
                    auto_mode="on",
                ),
            )
            self.legend_columns.setText(
                _extract_dict_text(settings, key="legend_kwargs", nested_key="ncols")
            )
            legend_font_size = settings.get("legend_font_size")
            if legend_font_size is None:
                legend_font_size = _extract_dict_text(
                    settings,
                    key="legend_kwargs",
                    nested_key="fontsize",
                )
            self.legend_font.setText(_display_optional_positive_int(legend_font_size))

            self._set_combo_value(
                self.grid_mode,
                _toggle_to_mode(settings.get("grid"), auto_mode="on"),
            )
            self._set_combo_value(
                self.markers_mode,
                _toggle_to_mode(settings.get("markers"), auto_mode="off"),
            )
            tick_params_settings = settings.get("tick_params_kwargs")
            tick_axis_mode = "both"
            minor_ticks_mode = "off"
            if isinstance(tick_params_settings, dict):
                raw_tick_axis = (
                    str(
                        tick_params_settings.get(
                            "_ticks_axis", tick_params_settings.get("axis", "both")
                        )
                    )
                    .strip()
                    .lower()
                )
                if raw_tick_axis in _TICK_AXES:
                    tick_axis_mode = raw_tick_axis
                raw_minor_mode = (
                    str(tick_params_settings.get("_minor_ticks_mode", "off")).strip().lower()
                )
                if raw_minor_mode in _MINOR_TICKS_MODES:
                    minor_ticks_mode = raw_minor_mode
            ticks_visibility_mode = "none" if settings.get("ticks") is False else tick_axis_mode
            self._set_combo_value(self.ticks_visibility, ticks_visibility_mode)

            self._set_combo_value(self.x_scale, str(settings.get("x_scale") or "linear"))
            self._set_combo_value(self.y_scale, str(settings.get("y_scale") or "linear"))
            self.x_min.setText(_extract_limit(settings, key="x_lim", index=0))
            self.x_max.setText(_extract_limit(settings, key="x_lim", index=1))
            self.y_min.setText(_extract_limit(settings, key="y_lim", index=0))
            self.y_max.setText(_extract_limit(settings, key="y_lim", index=1))
            self.x_ticks.setText(_format_float_list(settings.get("x_ticks")))
            self.y_ticks.setText(_format_float_list(settings.get("y_ticks")))
            self.x_tick_rotation.setText(str(settings.get("x_tick_rotation") or ""))
            self.y_tick_rotation.setText(str(settings.get("y_tick_rotation") or ""))
            self.x_label_pad.setText(
                "" if settings.get("x_label_pad") is None else str(settings.get("x_label_pad"))
            )
            self.y_label_pad.setText(
                "" if settings.get("y_label_pad") is None else str(settings.get("y_label_pad"))
            )

            self.fig_width.setText(
                _extract_figsize_dimension(settings, index=0, fallback=defaults.figure_size[0])
            )
            self.fig_height.setText(
                _extract_figsize_dimension(settings, index=1, fallback=defaults.figure_size[1])
            )
            self.dpi.setText(str(settings.get("dpi") or defaults.dpi))
            self.font_family.setText(str(settings.get("font_family") or ""))
            self.base_font_size.setText(_display_optional_positive_int(settings.get("font_size")))
            self.figure_facecolor.setText(
                _extract_dict_text(settings, key="figure_kwargs", nested_key="facecolor")
            )
            self._set_combo_value(
                self.transparent_mode,
                _extract_dict_mode(
                    settings,
                    key="savefig_kwargs",
                    nested_key="transparent",
                    auto_mode="off",
                ),
            )
            self.title_font.setText(_display_optional_positive_int(settings.get("title_font_size")))
            self.label_font.setText(_display_optional_positive_int(settings.get("label_font_size")))
            self.tick_font.setText(_display_optional_positive_int(settings.get("tick_font_size")))
            self.line_width.setText(str(settings.get("line_width") or defaults.line_width))
            self._set_combo_value(
                self.line_style,
                _extract_dict_text(settings, key="line_kwargs", nested_key="linestyle"),
            )
            self.line_alpha.setText(
                _extract_dict_text(settings, key="line_kwargs", nested_key="alpha")
            )
            self.marker_size.setText(
                _extract_dict_text(settings, key="line_kwargs", nested_key="markersize")
            )
            self._set_combo_value(
                self.axes_border_mode,
                _border_setting_to_mode(settings.get("border")),
            )
            _border_spines = _border_spines_from_setting(settings.get("border"))
            self.border_left.setChecked(_border_spines["left"])
            self.border_right.setChecked(_border_spines["right"])
            self.border_top.setChecked(_border_spines["top"])
            self.border_bottom.setChecked(_border_spines["bottom"])
            self._set_combo_value(self.grid_linestyle, str(settings.get("grid_linestyle") or ""))
            self.grid_linewidth.setText(
                str(settings.get("grid_linewidth") or defaults.grid_linewidth)
            )
            self.grid_alpha.setText(str(settings.get("grid_alpha") or defaults.grid_alpha))
            self.grid_color.setText(
                _extract_dict_text(settings, key="grid_kwargs", nested_key="color")
            )
            self._set_combo_value(
                self.grid_axis,
                str(_extract_dict_value(settings, key="grid_kwargs", nested_key="axis") or "both"),
            )
            self._set_combo_value(
                self.grid_which,
                str(
                    _extract_dict_value(settings, key="grid_kwargs", nested_key="which") or "major"
                ),
            )
            self._set_combo_value(
                self.tick_direction,
                str(
                    _extract_dict_value(settings, key="tick_params_kwargs", nested_key="direction")
                    or "out"
                ),
            )
            self.tick_length.setText(
                _extract_dict_text(settings, key="tick_params_kwargs", nested_key="length")
            )
            self._refresh_font_size_placeholders()
            self.tick_width.setText(
                _extract_dict_text(settings, key="tick_params_kwargs", nested_key="width")
            )
            self._set_combo_value(self.minor_ticks_mode, minor_ticks_mode)

            if hasattr(self, "analysis_species"):
                self.analysis_species.setText(str(settings.get("species") or ""))
            if hasattr(self, "analysis_axis"):
                self._set_combo_value(self.analysis_axis, str(settings.get("axis") or ""))
            if hasattr(self, "density_x_mode"):
                self._set_combo_value(
                    self.density_x_mode,
                    str(settings.get("x_mode") or "distance"),
                )
            if hasattr(self, "density_quantity"):
                self._set_combo_value(
                    self.density_quantity,
                    str(settings.get("quantity") or "mass"),
                )
            if hasattr(self, "rdf_species_a"):
                self._set_combo_value(
                    self.rdf_species_a,
                    self._profile_filter_display_value(
                        settings.get("species_a"),
                        default_label=_PROFILE_FILTER_METADATA_LABEL,
                    ),
                )
            if hasattr(self, "rdf_species_b"):
                species_a_value = settings.get("species_a")
                self._set_combo_items(
                    self.rdf_species_b,
                    self._rdf_species_b_choices(
                        None if species_a_value in {None, ""} else str(species_a_value)
                    ),
                    preferred_value=self._profile_filter_display_value(
                        settings.get("species_b"),
                        default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                    ),
                )
            if hasattr(self, "coord_species_a"):
                self._set_combo_value(
                    self.coord_species_a,
                    self._profile_filter_display_value(
                        settings.get("species_a"),
                        default_label=_PROFILE_FILTER_METADATA_LABEL,
                    ),
                )
            if hasattr(self, "coord_species_b"):
                species_a_value = settings.get("species_a")
                species_b_value = settings.get("species_b")
                self._set_combo_items(
                    self.coord_species_b,
                    self._coordination_species_b_choices(
                        None if species_a_value in {None, ""} else str(species_a_value)
                    ),
                    preferred_value=self._profile_filter_display_value(
                        species_b_value,
                        default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                    ),
                )
                if hasattr(self, "analysis_axis") and self._analysis_name == "coordination":
                    self._set_combo_items(
                        self.analysis_axis,
                        self._coordination_axis_choices(
                            None if species_a_value in {None, ""} else str(species_a_value),
                            None if species_b_value in {None, ""} else str(species_b_value),
                        ),
                        preferred_value=str(settings.get("axis") or ""),
                    )
            if hasattr(self, "position_component"):
                self._set_combo_value(
                    self.position_component,
                    str(settings.get("component") or "distance"),
                )
            if hasattr(self, "position_map_color"):
                self._set_combo_value(
                    self.position_map_color,
                    str(settings.get("map_color") or "distance"),
                )
            if hasattr(self, "position_time_axis"):
                self._set_combo_value(
                    self.position_time_axis,
                    str(settings.get("time_axis") or "ps"),
                )
            if hasattr(self, "coordination_component"):
                self._set_combo_value(
                    self.coordination_component,
                    str(settings.get("component") or "distance"),
                )
            if hasattr(self, "coordination_time_axis"):
                self._set_combo_value(
                    self.coordination_time_axis,
                    str(settings.get("time_axis") or "ps"),
                )
            if hasattr(self, "orientation_component"):
                self.orientation_component.blockSignals(True)
                try:
                    self._set_combo_value(
                        self.orientation_component,
                        str(settings.get("component") or "average"),
                    )
                finally:
                    self.orientation_component.blockSignals(False)
            if hasattr(self, "orientation_angle"):
                self._set_combo_value(
                    self.orientation_angle,
                    str(settings.get("angle") or "polar"),
                )
            if hasattr(self, "heatmap_vmin"):
                vmin_raw = settings.get("heatmap_vmin")
                self.heatmap_vmin.setText(str(vmin_raw) if vmin_raw is not None else "")
            if hasattr(self, "heatmap_vmax"):
                vmax_raw = settings.get("heatmap_vmax")
                self.heatmap_vmax.setText(str(vmax_raw) if vmax_raw is not None else "")
            if hasattr(self, "heatmap_cmap"):
                self._set_combo_value(
                    self.heatmap_cmap,
                    str(settings.get("heatmap_cmap") or "turbo"),
                )
            if hasattr(self, "heatmap_normalization_mode"):
                raw_mode = settings.get("heatmap_normalization_mode")
                if raw_mode is None:
                    raw_mode = (
                        "global_probability"
                        if bool(settings.get("heatmap_normalize", False))
                        else "counts"
                    )
                self._set_combo_value(
                    self.heatmap_normalization_mode,
                    _HEATMAP_NORMALIZATION_LABEL_BY_MODE.get(
                        str(raw_mode).strip().lower(),
                        _HEATMAP_NORMALIZATION_LABEL_BY_MODE["counts"],
                    ),
                )
            if hasattr(self, "heatmap_log_scale"):
                self.heatmap_log_scale.setChecked(bool(settings.get("heatmap_log_scale", False)))
            if hasattr(self, "heatmap_colorbar_label"):
                raw_cb_label = settings.get("heatmap_colorbar_label")
                if raw_cb_label is None:
                    self.heatmap_colorbar_label.setText("")
                elif raw_cb_label == "":
                    self.heatmap_colorbar_label.setText("none")
                else:
                    self.heatmap_colorbar_label.setText(str(raw_cb_label))
            if hasattr(self, "heatmap_colorbar_label_size"):
                raw = settings.get("heatmap_colorbar_label_size")
                self.heatmap_colorbar_label_size.setText(str(raw) if raw is not None else "")
            if hasattr(self, "heatmap_colorbar_tick_size"):
                raw = settings.get("heatmap_colorbar_tick_size")
                self.heatmap_colorbar_tick_size.setText(str(raw) if raw is not None else "")
            if hasattr(self, "heatmap_colorbar_enabled"):
                self.heatmap_colorbar_enabled.setChecked(
                    bool(settings.get("heatmap_colorbar_enabled", True))
                )
            if hasattr(self, "heatmap_colorbar_position"):
                self._set_combo_value(
                    self.heatmap_colorbar_position,
                    str(settings.get("heatmap_colorbar_position") or "right"),
                )
            if hasattr(self, "heatmap_colorbar_pad"):
                raw = settings.get("heatmap_colorbar_pad")
                self.heatmap_colorbar_pad.setText(str(raw) if raw is not None else "0.05")
            if hasattr(self, "heatmap_colorbar_shrink"):
                raw = settings.get("heatmap_colorbar_shrink")
                self.heatmap_colorbar_shrink.setText(str(raw) if raw is not None else "1.0")
            if hasattr(self, "heatmap_colorbar_aspect"):
                raw = settings.get("heatmap_colorbar_aspect")
                self.heatmap_colorbar_aspect.setText(str(raw) if raw is not None else "20")
            self.x_bin_width.setText(str(settings.get("x_bin_width") or ""))
            self._set_combo_value(self.x_bin_reducer, str(settings.get("x_bin_reducer") or "mean"))
            self.min_bin_points.setText(str(settings.get("min_bin_points") or ""))
            if hasattr(self, "y_bin_width"):
                self.y_bin_width.setText(str(settings.get("y_bin_width") or ""))
            if hasattr(self, "y_bin_reducer"):
                self._set_combo_value(
                    self.y_bin_reducer, str(settings.get("y_bin_reducer") or "mean")
                )
            self._initialize_annotation_data(settings)
            self._initialize_series_data(settings)
            self._initialize_normalization_data(settings)
            self._update_potential_summary_panel(settings)
            self.matplotlib_rc_json.setPlainText(_format_json_block(settings.get("matplotlib_rc")))
            self.figure_kwargs_json.setPlainText(_format_json_block(settings.get("figure_kwargs")))
            self.axes_kwargs_json.setPlainText(_format_json_block(settings.get("axes_kwargs")))
            self.line_kwargs_json.setPlainText(_format_json_block(settings.get("line_kwargs")))
            self.legend_kwargs_json.setPlainText(_format_json_block(settings.get("legend_kwargs")))
            self.grid_kwargs_json.setPlainText(_format_json_block(settings.get("grid_kwargs")))
            self.tick_params_kwargs_json.setPlainText(
                _format_json_block(settings.get("tick_params_kwargs"))
            )
            self.tight_layout_kwargs_json.setPlainText(
                _format_json_block(settings.get("tight_layout_kwargs"))
            )
            self.savefig_kwargs_json.setPlainText(
                _format_json_block(settings.get("savefig_kwargs"))
            )
            for key in _SYNCED_FIELD_KEYS:
                if key in _TRI_STATE_SYNC_FIELD_KEYS:
                    self._set_synced_field_mode(key, synced_modes.get(key, "auto"))
                else:
                    self._set_synced_field_lock(key, synced_locks.get(key, False))
            self._apply_preview_state_to_synced_fields(settings)

        def _is_orientation_heatmap_mode(self) -> bool:
            return (
                self._analysis_name == "orientation"
                and hasattr(self, "orientation_component")
                and self.orientation_component.currentText().strip().lower() == "heatmap"
            )

        def _refresh_widget_states(self, *_unused: object) -> None:
            if all(
                hasattr(self, name)
                for name in (
                    "title_font",
                    "label_font",
                    "tick_font",
                    "legend_font",
                    "base_font_size",
                )
            ):
                self._refresh_font_size_placeholders()
            title_enabled = self._synced_field_mode("title") != "off"
            x_label_enabled = self._synced_field_mode("x_label") != "off"
            y_label_enabled = self._synced_field_mode("y_label") != "off"
            legend_enabled = self.legend_mode.currentText().strip().lower() != "off"
            grid_enabled = self.grid_mode.currentText().strip().lower() != "off"
            ticks_mode = self.ticks_visibility.currentText().strip().lower()
            ticks_enabled = ticks_mode != "none"
            markers_enabled = self.markers_mode.currentText().strip().lower() != "off"
            colorbar_enabled = (
                self.heatmap_colorbar_enabled.isChecked()
                if hasattr(self, "heatmap_colorbar_enabled")
                else True
            )
            rebin_enabled = (
                bool(self.x_bin_width.text().strip()) if hasattr(self, "x_bin_width") else False
            )
            norm_mode = (
                self.norm_mode.currentText().strip().lower()
                if hasattr(self, "norm_mode")
                else "none"
            )
            norm_enabled = norm_mode != "none"
            norm_x_ref_enabled = norm_mode == "value_at_x"
            position_component = (
                self.position_component.currentText().strip().lower()
                if hasattr(self, "position_component")
                else ""
            )
            position_xy_projection = position_component == "xy-z"
            coordination_component = (
                self.coordination_component.currentText().strip().lower()
                if hasattr(self, "coordination_component")
                else ""
            )
            coordination_time_distance = coordination_component == "time-distance"
            coordination_distance = coordination_component == "distance"
            layer_caps = self._current_layer_capabilities()
            figure_caps = self._current_figure_capabilities()
            fit_supported = bool(layer_caps.show_fit_editor)
            cumulative_supported = bool(layer_caps.show_cumulative_editor)
            selected_group = layer_caps.layer_kind == "group"

            for widget in self._title_detail_widgets:
                widget.setVisible(title_enabled)
            for widget in self._x_label_detail_widgets:
                widget.setVisible(x_label_enabled)
            for widget in self._y_label_detail_widgets:
                widget.setVisible(y_label_enabled)
            self._set_rows_visible(self._title_rows, title_enabled)
            self._set_rows_visible(self._legend_rows, legend_enabled)
            self._set_rows_visible(self._grid_rows, grid_enabled)
            self._set_rows_visible(self._marker_rows, markers_enabled)
            border_custom = (
                hasattr(self, "axes_border_mode")
                and self.axes_border_mode.currentText().strip().lower() == "custom"
            )
            self._set_rows_visible(self._border_custom_rows, border_custom)
            self._set_rows_visible(self._ticks_rows, ticks_enabled)
            self._set_rows_visible(self._colorbar_rows, colorbar_enabled)
            self._set_rows_enabled(
                self._title_rows,
                title_enabled,
                disabled_reason="the title is currently off.",
            )
            self._set_rows_enabled(
                self._legend_rows,
                legend_enabled,
                disabled_reason="the legend is currently off.",
            )
            self._set_rows_enabled(
                self._grid_rows,
                grid_enabled,
                disabled_reason="the grid is currently off.",
            )
            self._set_rows_enabled(
                self._marker_rows,
                markers_enabled,
                disabled_reason="markers are currently off.",
            )
            self._set_rows_enabled(
                self._colorbar_rows,
                colorbar_enabled,
                disabled_reason="the colorbar is currently off.",
            )
            for form, field in self._ticks_rows:
                self._set_form_row_enabled(
                    form,
                    field,
                    ticks_enabled,
                    disabled_reason="ticks are currently off.",
                )

            if self._series_visibility_group is not None:
                self._series_visibility_group.setVisible(layer_caps.show_visibility_label)
            if self._series_style_group is not None:
                self._series_style_group.setVisible(layer_caps.show_style)
            if self._series_uncertainty_group is not None:
                self._series_uncertainty_group.setVisible(layer_caps.show_uncertainty)
            if self._series_derived_group is not None:
                self._series_derived_group.setVisible(layer_caps.show_derived_lines)
            if self._series_fit_group is not None:
                self._series_fit_group.setVisible(layer_caps.show_fit_editor)
            if self._series_cumulative_group is not None:
                self._series_cumulative_group.setVisible(layer_caps.show_cumulative_editor)
            if self._series_group_group is not None:
                self._series_group_group.setVisible(layer_caps.show_group_members)
            if self._normalization_group is not None:
                self._normalization_group.setVisible(layer_caps.show_normalization)
            if self._series_metadata_group is not None:
                self._series_metadata_group.setVisible(layer_caps.show_metadata)
            if self._figure_legend_section is not None:
                self._figure_legend_section.setVisible(figure_caps.show_legend)
            if self._figure_lines_group is not None:
                self._figure_lines_group.setVisible(figure_caps.show_lines)
            if self._figure_heatmap_group is not None:
                self._figure_heatmap_group.setVisible(figure_caps.show_heatmap)
            if self._figure_colorbar_group is not None:
                self._figure_colorbar_group.setVisible(figure_caps.show_colorbar)

            if self._position_map_color_row is not None:
                self._set_form_row_visible(
                    self._position_map_color_row[0],
                    self._position_map_color_row[1],
                    position_xy_projection,
                )
            if self._position_time_axis_row is not None:
                self._set_form_row_visible(
                    self._position_time_axis_row[0],
                    self._position_time_axis_row[1],
                    not position_xy_projection,
                )
            if self._coordination_time_axis_row is not None:
                self._set_form_row_visible(
                    self._coordination_time_axis_row[0],
                    self._coordination_time_axis_row[1],
                    not coordination_distance,
                )
            if self._data_transform_group is not None and self._analysis_name == "position":
                self._data_transform_group.setVisible(not position_xy_projection)
            if self._data_transform_group is not None and self._analysis_name == "coordination":
                self._data_transform_group.setVisible(not coordination_time_distance)
            # ── orientation heatmap mode ──────────────────────────────
            is_heatmap = self._is_orientation_heatmap_mode()
            if self._data_transform_group is not None and self._analysis_name == "orientation":
                self._data_transform_group.setEnabled(True)
                self._data_transform_group.setToolTip("")
            if self._y_bin_width_row is not None:
                self._set_form_row_visible(
                    self._y_bin_width_row[0],
                    self._y_bin_width_row[1],
                    is_heatmap,
                )
            if self._y_bin_reducer_row is not None:
                y_rebin_enabled = (
                    is_heatmap
                    and bool(getattr(self, "y_bin_width", None))
                    and bool(self.y_bin_width.text().strip())
                )
                self._set_form_row_visible(
                    self._y_bin_reducer_row[0],
                    self._y_bin_reducer_row[1],
                    y_rebin_enabled,
                )
                if is_heatmap:
                    self._set_form_row_enabled(
                        self._y_bin_reducer_row[0],
                        self._y_bin_reducer_row[1],
                        y_rebin_enabled,
                        disabled_reason="set a Y section width first.",
                    )
            if self._normalization_group is not None and self._analysis_name == "orientation":
                self._normalization_group.setEnabled(not is_heatmap)
                self._normalization_group.setToolTip(
                    "" if not is_heatmap else "Normalization is unavailable for heatmap views."
                )
            if is_heatmap:
                heatmap_tip = "This control applies to line plots only."
                for widget_name in (
                    "series_color",
                    "series_alpha",
                    "series_line_width",
                    "series_marker",
                    "series_show_in_legend",
                    "series_line_kwargs_json",
                ):
                    widget = getattr(self, widget_name, None)
                    if widget is not None:
                        widget.setEnabled(False)
                        self._apply_widget_tooltip(widget, disabled_reason=heatmap_tip)
                self._set_rows_enabled(
                    self._legend_rows,
                    False,
                    disabled_reason="legend is unavailable for heatmap views.",
                )
                self._set_rows_enabled(
                    self._marker_rows,
                    False,
                    disabled_reason="markers are unavailable for heatmap views.",
                )
            if self._x_bin_reducer_row is not None:
                self._set_form_row_visible(
                    self._x_bin_reducer_row[0],
                    self._x_bin_reducer_row[1],
                    rebin_enabled,
                )
                self._set_form_row_enabled(
                    self._x_bin_reducer_row[0],
                    self._x_bin_reducer_row[1],
                    rebin_enabled,
                    disabled_reason="set a section width first.",
                )
            if self._norm_value_row is not None:
                self._set_form_row_visible(
                    self._norm_value_row[0],
                    self._norm_value_row[1],
                    norm_enabled,
                )
                self._set_form_row_enabled(
                    self._norm_value_row[0],
                    self._norm_value_row[1],
                    norm_enabled,
                    disabled_reason="normalization is currently off.",
                )
            if self._norm_x_ref_row is not None:
                self._set_form_row_visible(
                    self._norm_x_ref_row[0],
                    self._norm_x_ref_row[1],
                    norm_x_ref_enabled,
                )
                self._set_form_row_enabled(
                    self._norm_x_ref_row[0],
                    self._norm_x_ref_row[1],
                    norm_x_ref_enabled,
                    disabled_reason="reference x is only used for value_at_x normalization.",
                )
            for widget in (
                self._normalization_actions_widget,
                self._normalization_hint_label,
            ):
                if widget is not None:
                    widget.setVisible(norm_enabled)
            if self._normalization_copy_button is not None:
                normalization_copy_enabled = (
                    not self._series_active_is_fit_child
                    and not self._series_active_is_cumulative_child
                    and not selected_group
                    and norm_enabled
                )
                self._normalization_copy_button.setEnabled(normalization_copy_enabled)
                self._apply_widget_tooltip(
                    self._normalization_copy_button,
                    disabled_reason=(
                        None
                        if normalization_copy_enabled
                        else "normalization is edited on non-group base series only."
                    ),
                )
            if self._normalization_group is not None and selected_group:
                self._normalization_group.setEnabled(False)
                self._normalization_group.setToolTip(
                    "Grouped series aggregate already-transformed member series and do not apply an extra normalization step."
                )
            annotation_selected = bool(self._annotations_data)
            series_selected = bool(self._series_descriptors_data)
            if getattr(self, "_series_duplicate_button", None) is not None:
                self._series_duplicate_button.setEnabled(series_selected)
            if getattr(self, "_series_add_group_button", None) is not None:
                self._series_add_group_button.setEnabled(
                    bool(self._group_member_candidate_indices())
                )
            for button in (
                getattr(self, "_annotation_duplicate_button", None),
                getattr(self, "_annotation_delete_button", None),
            ):
                if button is not None:
                    button.setEnabled(annotation_selected)
            if getattr(self, "_annotation_move_up_button", None) is not None:
                self._annotation_move_up_button.setEnabled(
                    annotation_selected and self._annotation_active_index > 0
                )
            if getattr(self, "_annotation_move_down_button", None) is not None:
                self._annotation_move_down_button.setEnabled(
                    annotation_selected
                    and self._annotation_active_index < len(self._annotations_data) - 1
                )
            for widget in (
                getattr(self, "_annotation_enabled_mode", None),
                getattr(self, "_annotation_type", None),
                getattr(self, "_annotation_name", None),
                getattr(self, "_annotation_coord_system", None),
                getattr(self, "_annotation_color", None),
                getattr(self, "_annotation_alpha", None),
                getattr(self, "_annotation_zorder", None),
                getattr(self, "_annotation_text", None),
                getattr(self, "_annotation_x", None),
                getattr(self, "_annotation_y", None),
                getattr(self, "_annotation_font_size", None),
                getattr(self, "_annotation_rotation", None),
                getattr(self, "_annotation_horizontal_align", None),
                getattr(self, "_annotation_vertical_align", None),
                getattr(self, "_annotation_x1", None),
                getattr(self, "_annotation_y1", None),
                getattr(self, "_annotation_x2", None),
                getattr(self, "_annotation_y2", None),
                getattr(self, "_annotation_line_width", None),
                getattr(self, "_annotation_line_style", None),
                getattr(self, "_annotation_arrow_style", None),
                getattr(self, "_annotation_mutation_scale", None),
            ):
                if widget is not None:
                    widget.setEnabled(annotation_selected)
            title_mode = self._synced_field_mode("title")
            x_label_mode = self._synced_field_mode("x_label")
            y_label_mode = self._synced_field_mode("y_label")
            self.title_text.setEnabled(title_mode != "off")
            self.x_label.setEnabled(x_label_mode != "off")
            self.y_label.setEnabled(y_label_mode != "off")
            self._apply_widget_tooltip(
                self.title_text,
                disabled_reason=None if title_mode != "off" else "its sync mode is Off.",
            )
            self._apply_widget_tooltip(
                self.x_label,
                disabled_reason=None if x_label_mode != "off" else "its sync mode is Off.",
            )
            self._apply_widget_tooltip(
                self.y_label,
                disabled_reason=None if y_label_mode != "off" else "its sync mode is Off.",
            )
            error_supported = self._error_supported_for_current_view()
            error_availability = self._error_availability_for_series(self._series_active_index)
            error_controls_available = (
                error_supported
                and not self._series_active_is_fit_child
                and not self._series_active_is_cumulative_child
                and not selected_group
            )
            error_active = (
                error_controls_available
                and hasattr(self, "_series_error_mode")
                and self._series_error_mode.currentText().strip().lower() != "off"
            )
            self._set_rows_visible(self._series_error_detail_rows, error_active)
            for widget in self._series_error_detail_widgets:
                widget.setVisible(error_active)
            if hasattr(self, "_series_error_mode"):
                self._series_error_mode.setEnabled(error_controls_available)
                self._apply_widget_tooltip(
                    self._series_error_mode,
                    disabled_reason=(
                        "error settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else "grouped series do not render error overlays."
                        if selected_group
                        else None
                        if error_supported
                        else "error overlays are only available for 1-D line-based views."
                    ),
                )
            if hasattr(self, "_series_error_stat"):
                self._series_error_stat.setEnabled(
                    error_controls_available
                    and error_active
                    and error_availability.selector_enabled
                )
                self._apply_widget_tooltip(
                    self._series_error_stat,
                    disabled_reason=(
                        "error settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else "grouped series do not render error overlays."
                        if selected_group
                        else error_availability.reason
                        if error_supported and not error_availability.selector_enabled
                        else None
                        if error_supported
                        else "error overlays are only available for 1-D line-based views."
                    ),
                )
            if hasattr(self, "_series_error_style"):
                self._series_error_style.setEnabled(error_controls_available and error_active)
                self._apply_widget_tooltip(
                    self._series_error_style,
                    disabled_reason=(
                        "error settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else "grouped series do not render error overlays."
                        if selected_group
                        else None
                        if error_supported
                        else "error overlays are only available for 1-D line-based views."
                    ),
                )
            for widget in (
                getattr(self, "_series_error_color", None),
                getattr(self, "_series_error_show_in_legend", None),
                getattr(self, "_series_error_label", None),
            ):
                if widget is not None:
                    widget.setEnabled(error_controls_available and error_active)
                    self._apply_widget_tooltip(
                        widget,
                        disabled_reason=(
                            "error settings are edited on the base series only."
                            if self._series_active_is_fit_child
                            or self._series_active_is_cumulative_child
                            else "grouped series do not render error overlays."
                            if selected_group
                            else None
                            if error_supported
                            else "error overlays are only available for 1-D line-based views."
                        ),
                    )
            if self._min_bin_points_row is not None:
                self._set_form_row_visible(
                    self._min_bin_points_row[0],
                    self._min_bin_points_row[1],
                    error_supported,
                )
                self._set_form_row_enabled(
                    self._min_bin_points_row[0],
                    self._min_bin_points_row[1],
                    error_supported,
                    disabled_reason=(
                        None
                        if error_supported
                        else "minimum bin counts are only available for 1-D line-based views."
                    ),
                )
            if self._series_fit_mode is not None:
                self._series_fit_mode.setEnabled(
                    fit_supported and not self._series_active_is_cumulative_child
                )
                self._apply_widget_tooltip(
                    self._series_fit_mode,
                    disabled_reason=(
                        "fit settings are edited on the base series only."
                        if self._series_active_is_cumulative_child
                        else None
                        if fit_supported
                        else "fitting is only available for line-based views."
                    ),
                )
            fit_active = (
                fit_supported
                and self._series_fit_mode is not None
                and self._series_fit_mode.currentText().strip().lower() != "off"
            )
            fit_type = (
                self._series_fit_type.currentText().strip().lower()
                if hasattr(self, "_series_fit_type")
                else "linear"
            )
            fit_manual_range = (
                hasattr(self, "_series_fit_range_mode")
                and self._series_fit_range_mode.currentText().strip().lower() == "manual"
            )
            polynomial_selected = fit_type == "polynomial"
            for form, field in self._series_fit_detail_rows:
                visible = fit_active
                if field is getattr(self, "_series_fit_degree", None):
                    visible = fit_active and polynomial_selected
                elif field is getattr(self, "_series_fit_x_min", None) or field is getattr(
                    self,
                    "_series_fit_x_max",
                    None,
                ):
                    visible = fit_active and fit_manual_range
                self._set_form_row_visible(form, field, visible)
            for widget in self._series_fit_detail_widgets:
                widget.setVisible(fit_active)
            if hasattr(self, "_series_fit_type"):
                self._series_fit_type.setEnabled(
                    fit_supported
                    and not self._series_active_is_fit_child
                    and not self._series_active_is_cumulative_child
                )
                self._apply_widget_tooltip(
                    self._series_fit_type,
                    disabled_reason=(
                        "fit settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else None
                        if fit_supported
                        else "fitting is only available for line-based views."
                    ),
                )
            if hasattr(self, "_series_fit_degree"):
                self._series_fit_degree.setEnabled(
                    fit_supported
                    and not self._series_active_is_fit_child
                    and fit_active
                    and polynomial_selected
                )
                self._apply_widget_tooltip(
                    self._series_fit_degree,
                    disabled_reason=(
                        "fit settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        else "turn fitting on first."
                        if not fit_active
                        else "polynomial degree is only used for polynomial fits."
                        if not polynomial_selected
                        else None
                    ),
                )
            if hasattr(self, "_series_fit_range_mode"):
                self._series_fit_range_mode.setEnabled(
                    fit_supported and not self._series_active_is_fit_child and fit_active
                )
                self._apply_widget_tooltip(
                    self._series_fit_range_mode,
                    disabled_reason=(
                        "fit settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        else "turn fitting on first."
                        if not fit_active
                        else None
                    ),
                )
            for widget in (
                getattr(self, "_series_fit_x_min", None),
                getattr(self, "_series_fit_x_max", None),
            ):
                if widget is not None:
                    widget.setEnabled(
                        fit_supported
                        and not self._series_active_is_fit_child
                        and not self._series_active_is_cumulative_child
                        and fit_active
                        and fit_manual_range
                    )
                    self._apply_widget_tooltip(
                        widget,
                        disabled_reason=(
                            "fit settings are edited on the base series only."
                            if self._series_active_is_fit_child
                            or self._series_active_is_cumulative_child
                            else "turn fitting on first."
                            if not fit_active
                            else "manual fit range is not selected."
                            if not fit_manual_range
                            else None
                        ),
                    )
            if hasattr(self, "_series_fit_show_in_legend"):
                self._series_fit_show_in_legend.setEnabled(
                    fit_supported and fit_active and not self._series_active_is_cumulative_child
                )
            if hasattr(self, "_series_fit_label"):
                self._series_fit_label.setEnabled(
                    fit_supported and fit_active and not self._series_active_is_cumulative_child
                )
            if hasattr(self, "_series_cumulative_mode"):
                self._series_cumulative_mode.setEnabled(
                    cumulative_supported and not self._series_active_is_fit_child
                )
                self._apply_widget_tooltip(
                    self._series_cumulative_mode,
                    disabled_reason=(
                        "cumulative settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        else None
                        if cumulative_supported
                        else "cumulative-average lines are only available for 1-D line-based views."
                    ),
                )
            cumulative_active = (
                cumulative_supported
                and hasattr(self, "_series_cumulative_mode")
                and self._series_cumulative_mode.currentText().strip().lower() != "off"
            )
            self._set_rows_visible(self._series_cumulative_detail_rows, cumulative_active)
            for widget in self._series_cumulative_detail_widgets:
                widget.setVisible(cumulative_active)
            for widget in (
                getattr(self, "_series_cumulative_show_in_legend", None),
                getattr(self, "_series_cumulative_label", None),
            ):
                if widget is not None:
                    widget.setEnabled(cumulative_supported and cumulative_active)
            if hasattr(self, "_series_group_group") and self._series_group_group is not None:
                self._series_group_group.setEnabled(not self._series_active_is_fit_child)
            if hasattr(self, "_series_group_reducer"):
                self._series_group_reducer.setEnabled(
                    selected_group and not self._series_active_is_fit_child
                )
            if hasattr(self, "_series_group_members"):
                self._series_group_members.setEnabled(
                    selected_group and not self._series_active_is_fit_child
                )
            self._preview_button.setEnabled(
                _preview_button_enabled(
                    auto_update_enabled=self._auto_preview_checkbox.isChecked(),
                    preview_loading=self._preview_loading,
                )
            )
            self._update_normalization_warning()
            self._update_series_error_summary(self._series_active_index)
            if hasattr(self, "normalization_warning"):
                self.normalization_warning.setVisible(
                    norm_enabled and bool(self.normalization_warning.text().strip())
                )
            if not error_active:
                for widget in self._series_error_detail_widgets:
                    widget.setVisible(False)
            self._update_series_fit_summary(self._series_active_index)
            if not fit_active:
                for widget in self._series_fit_detail_widgets:
                    widget.setVisible(False)
            if not cumulative_active:
                for widget in self._series_cumulative_detail_widgets:
                    widget.setVisible(False)
            self._sync_standard_controls_to_advanced_json()
            self._refresh_shell_state()

        def _collect_settings(self) -> dict[str, Any]:
            self._persist_active_series_editor()
            self._persist_active_annotation_editor()

            def _synced_text(key: str, widget: QLineEdit) -> str | None:
                mode = self._synced_field_mode(key)
                if mode == "off":
                    return ""
                if mode != "manual":
                    return None
                return _explicit_text(widget.text())

            def _synced_float(key: str, widget: QLineEdit, *, field_name: str) -> float | None:
                if not self._synced_field_locks.get(key, False):
                    return None
                return _optional_float(widget.text(), field_name=field_name)

            def _synced_float_list(
                key: str,
                widget: QLineEdit,
                *,
                field_name: str,
            ) -> list[float] | None:
                if not self._synced_field_locks.get(key, False):
                    return None
                return _optional_float_list(widget.text(), field_name=field_name)

            def _optional_positive_int_or_none(value: str, *, field_name: str) -> int | None:
                parsed = _optional_int(value, field_name=field_name)
                if parsed is None or parsed <= 0:
                    return None
                return parsed

            fig_width = _optional_float(self.fig_width.text(), field_name="figure width")
            fig_height = _optional_float(self.fig_height.text(), field_name="figure height")
            if (fig_width is None) != (fig_height is None):
                raise ValueError(
                    "Figure width and figure height must both be set or both be blank."
                )
            figsize: list[float] | None = None
            if fig_width is not None and fig_height is not None:
                figsize = [fig_width, fig_height]

            series_labels = [
                self._effective_series_label(index)
                for index in range(len(self._series_labels_data))
            ]
            line_colors = [color.strip() for color in self._series_colors_data]
            line_colors_value = _resolve_series_line_colors(line_colors)

            series_enabled = [bool(value) for value in self._series_enabled_data]
            series_enabled_value: list[bool] | None = None
            if any(value is False for value in series_enabled):
                series_enabled_value = series_enabled

            series_show_in_legend = [bool(value) for value in self._series_show_in_legend_data]
            series_show_in_legend_value: list[bool] | None = None
            if any(value is False for value in series_show_in_legend):
                series_show_in_legend_value = series_show_in_legend

            series_alpha: list[float | None] = []
            has_series_alpha = False
            for index, raw_alpha in enumerate(self._series_alpha_data):
                token = raw_alpha.strip()
                if not token:
                    series_alpha.append(None)
                    continue
                try:
                    parsed_alpha = float(token)
                except ValueError as exc:
                    raise ValueError(f"Series {index + 1} alpha must be a float.") from exc
                series_alpha.append(parsed_alpha)
                has_series_alpha = True
            series_alpha_value: list[float | None] | None = (
                series_alpha if has_series_alpha else None
            )

            series_line_widths: list[float | None] = []
            has_custom_series_width = False
            for index, raw_width in enumerate(self._series_line_widths_data):
                token = raw_width.strip()
                if not token:
                    series_line_widths.append(None)
                    continue
                try:
                    series_line_widths.append(float(token))
                    has_custom_series_width = True
                except ValueError as exc:
                    raise ValueError(f"Series {index + 1} line width must be a float.") from exc
            series_line_widths_value: list[float | None] | None = None
            if has_custom_series_width:
                series_line_widths_value = series_line_widths

            series_markers = [marker.strip() for marker in self._series_markers_data]
            series_markers_value: list[str] | None = None
            if any(marker for marker in series_markers):
                series_markers_value = series_markers

            series_line_kwargs: list[dict[str, Any] | None] = []
            has_series_line_kwargs = False
            for index, raw in enumerate(self._series_line_kwargs_data):
                token = raw.strip()
                if not token:
                    series_line_kwargs.append(None)
                    continue
                parsed = _optional_json_dict(
                    token,
                    field_name=f"Series {index + 1} line kwargs",
                )
                series_line_kwargs.append(parsed)
                has_series_line_kwargs = True
            series_line_kwargs_value: list[dict[str, Any] | None] | None = None
            if has_series_line_kwargs:
                series_line_kwargs_value = series_line_kwargs

            normalization_modes = [
                mode.strip().lower() for mode in self._series_normalization_modes_data
            ]
            normalization_values: list[float | None] = []
            normalization_x_refs: list[float | None] = []
            has_normalization = False
            for index, mode in enumerate(normalization_modes):
                if mode not in _NORMALIZATION_MODES:
                    raise ValueError(f"Series {index + 1} normalization mode is invalid.")
                value = _optional_float(
                    self._series_normalization_values_data[index],
                    field_name=f"Series {index + 1} normalization target",
                )
                x_ref = _optional_float(
                    self._series_normalization_x_refs_data[index],
                    field_name=f"Series {index + 1} normalization reference x",
                )
                if mode == "none":
                    value = None
                    x_ref = None
                elif mode == "value_at_x":
                    if value is None or x_ref is None:
                        raise ValueError(
                            f"Series {index + 1} normalization mode 'value_at_x' requires target and reference x."
                        )
                    has_normalization = True
                else:
                    if value is None:
                        raise ValueError(
                            f"Series {index + 1} normalization mode '{mode}' requires a target value."
                        )
                    if x_ref is not None:
                        raise ValueError(
                            f"Series {index + 1} normalization reference x is only valid for mode 'value_at_x'."
                        )
                    has_normalization = True
                normalization_values.append(value)
                normalization_x_refs.append(x_ref)

            normalization_modes_value: list[str] | None = None
            normalization_values_value: list[float | None] | None = None
            normalization_x_refs_value: list[float | None] | None = None
            if has_normalization:
                normalization_modes_value = normalization_modes
                normalization_values_value = normalization_values
                normalization_x_refs_value = normalization_x_refs

            raw_annotations: list[dict[str, Any]] = []
            for annotation_entry in self._annotations_data:
                annotation_type = (
                    str(annotation_entry.get("type") or "text").strip().lower() or "text"
                )
                payload: dict[str, Any] = {
                    "id": str(annotation_entry.get("id") or f"annotation:{uuid4().hex}").strip(),
                    "type": annotation_type,
                    "enabled": bool(annotation_entry.get("enabled", True)),
                    "name": str(annotation_entry.get("name") or "").strip(),
                    "coord_system": str(annotation_entry.get("coord_system") or "axes")
                    .strip()
                    .lower(),
                    "color": str(annotation_entry.get("color") or "#000000").strip() or "#000000",
                    "alpha": str(annotation_entry.get("alpha") or "1.0").strip() or "1.0",
                    "zorder": str(annotation_entry.get("zorder") or "5").strip() or "5",
                }
                if annotation_type == "text":
                    payload.update(
                        {
                            "text": str(annotation_entry.get("text") or "").strip(),
                            "x": str(annotation_entry.get("x") or "0.5").strip() or "0.5",
                            "y": str(annotation_entry.get("y") or "0.5").strip() or "0.5",
                            "font_size": str(annotation_entry.get("font_size") or "12").strip()
                            or "12",
                            "rotation": str(annotation_entry.get("rotation") or "0").strip() or "0",
                            "horizontal_align": str(
                                annotation_entry.get("horizontal_align") or "center"
                            )
                            .strip()
                            .lower()
                            or "center",
                            "vertical_align": str(
                                annotation_entry.get("vertical_align") or "center"
                            )
                            .strip()
                            .lower()
                            or "center",
                        }
                    )
                else:
                    payload.update(
                        {
                            "x1": str(annotation_entry.get("x1") or "0").strip() or "0",
                            "y1": str(annotation_entry.get("y1") or "0").strip() or "0",
                            "x2": str(annotation_entry.get("x2") or "1").strip() or "1",
                            "y2": str(annotation_entry.get("y2") or "1").strip() or "1",
                            "line_width": str(annotation_entry.get("line_width") or "1.5").strip()
                            or "1.5",
                            "line_style": str(annotation_entry.get("line_style") or "-").strip()
                            or "-",
                        }
                    )
                    if annotation_type == "arrow":
                        payload["arrow_style"] = (
                            str(annotation_entry.get("arrow_style") or "->").strip() or "->"
                        )
                        payload["mutation_scale"] = (
                            str(annotation_entry.get("mutation_scale") or "12").strip() or "12"
                        )
                raw_annotations.append(payload)
            annotations_value = (
                _coerce_plot_annotations(raw_annotations) if raw_annotations else None
            )

            series_overrides: dict[str, dict[str, Any]] = {}
            for index, descriptor in enumerate(self._series_descriptors_data):
                series_id = str(descriptor.get("series_id") or f"series:{index}")
                entry: dict[str, Any] = {}
                label_override = self._series_label_overrides_data[index].strip()
                if label_override:
                    entry["label_override"] = label_override
                if self._series_enabled_data[index] is False:
                    entry["enabled"] = False
                if self._series_show_in_legend_data[index] is False:
                    entry["show_in_legend"] = False
                fit_type = (
                    self._series_fit_types_data[index].strip().lower()
                    if index < len(self._series_fit_types_data)
                    else "linear"
                )
                if fit_type not in _FIT_TYPES:
                    raise ValueError(f"Series {index + 1} fit type is invalid.")
                fit_degree_value = None
                if fit_type == "polynomial":
                    fit_degree_value = _optional_int(
                        self._series_fit_degrees_data[index],
                        field_name=f"Series {index + 1} polynomial degree",
                    )
                    if fit_degree_value is None or fit_degree_value < 1:
                        raise ValueError(
                            f"Series {index + 1} polynomial degree must be an integer >= 1."
                        )
                fit_range_mode = (
                    self._series_fit_range_modes_data[index].strip().lower()
                    if index < len(self._series_fit_range_modes_data)
                    else "visible"
                )
                if fit_range_mode not in _FIT_RANGE_MODES:
                    raise ValueError(f"Series {index + 1} fit range mode is invalid.")
                fit_x_min_value = _optional_float(
                    self._series_fit_x_mins_data[index],
                    field_name=f"Series {index + 1} fit x min",
                )
                fit_x_max_value = _optional_float(
                    self._series_fit_x_maxs_data[index],
                    field_name=f"Series {index + 1} fit x max",
                )
                fit_label_override_value = (
                    self._series_fit_label_overrides_data[index].strip() or None
                )
                fit_enabled_value = bool(self._series_fit_enabled_data[index])
                fit_show_in_legend_value = bool(self._series_fit_show_in_legend_data[index])
                fit_defaults = _fit_defaults_for_gui()
                fit_config_payload = {
                    "fit_enabled": fit_enabled_value,
                    "fit_type": fit_type,
                    "fit_degree": fit_degree_value if fit_type == "polynomial" else None,
                    "fit_range_mode": fit_range_mode,
                    "fit_x_min": fit_x_min_value,
                    "fit_x_max": fit_x_max_value,
                    "fit_initial_guess": None,
                    "fit_bounds": None,
                    "fit_label_override": fit_label_override_value,
                    "fit_show_in_legend": fit_show_in_legend_value,
                }
                if fit_config_payload != {
                    "fit_enabled": fit_defaults["fit_enabled"],
                    "fit_type": fit_defaults["fit_type"],
                    "fit_degree": fit_defaults["fit_degree"],
                    "fit_range_mode": fit_defaults["fit_range_mode"],
                    "fit_x_min": fit_defaults["fit_x_min"],
                    "fit_x_max": fit_defaults["fit_x_max"],
                    "fit_initial_guess": fit_defaults["fit_initial_guess"],
                    "fit_bounds": fit_defaults["fit_bounds"],
                    "fit_label_override": fit_defaults["fit_label_override"],
                    "fit_show_in_legend": fit_defaults["fit_show_in_legend"],
                }:
                    entry["fit"] = fit_config_payload
                cumulative_enabled_value = bool(self._series_cumulative_enabled_data[index])
                cumulative_label_override_value = (
                    self._series_cumulative_label_overrides_data[index].strip() or None
                )
                cumulative_show_in_legend_value = bool(
                    self._series_cumulative_show_in_legend_data[index]
                )
                cumulative_payload = {
                    "enabled": cumulative_enabled_value,
                    "label_override": cumulative_label_override_value,
                    "show_in_legend": cumulative_show_in_legend_value,
                }
                if cumulative_payload != _cumulative_defaults_for_gui():
                    entry["cumulative"] = cumulative_payload
                error_enabled_value = bool(self._series_error_enabled_data[index])
                error_stat = self._resolved_error_stat_for_series(index)
                error_style = self._series_error_styles_data[index].strip().lower()
                if error_style not in _ERROR_STYLES:
                    error_style = "band"
                error_color_value = self._series_error_colors_data[index].strip() or None
                error_label_override_value = (
                    self._series_error_label_overrides_data[index].strip() or None
                )
                error_show_in_legend_value = bool(self._series_error_show_in_legend_data[index])
                error_config_payload = {
                    "enabled": error_enabled_value,
                    "stat": error_stat,
                    "style": error_style,
                    "color": error_color_value,
                    "label_override": error_label_override_value,
                    "show_in_legend": error_show_in_legend_value,
                }
                if error_config_payload != _error_defaults_for_gui():
                    entry["error"] = error_config_payload
                alpha_token = self._series_alpha_data[index].strip()
                if alpha_token:
                    try:
                        entry["alpha"] = float(alpha_token)
                    except ValueError as exc:
                        raise ValueError(f"Series {index + 1} alpha must be a float.") from exc
                if line_colors[index]:
                    entry["color"] = line_colors[index]
                width_token = self._series_line_widths_data[index].strip()
                if width_token:
                    try:
                        entry["line_width"] = float(width_token)
                    except ValueError as exc:
                        raise ValueError(f"Series {index + 1} line width must be a float.") from exc
                marker_token = self._series_markers_data[index].strip()
                if marker_token:
                    entry["marker"] = marker_token
                line_kwargs_token = self._series_line_kwargs_data[index].strip()
                if line_kwargs_token:
                    entry["line_kwargs"] = _optional_json_dict(
                        line_kwargs_token,
                        field_name=f"Series {index + 1} line kwargs",
                    )
                mode = normalization_modes[index]
                if mode != "none":
                    entry["normalization_mode"] = mode
                    if normalization_values[index] is not None:
                        entry["normalization_value"] = normalization_values[index]
                    if normalization_x_refs[index] is not None:
                        entry["normalization_x_ref"] = normalization_x_refs[index]
                if entry:
                    series_overrides[series_id] = entry

            matplotlib_rc_value = _optional_json_dict(
                self.matplotlib_rc_json.toPlainText(),
                field_name="Matplotlib rcParams",
            )
            figure_kwargs_value = _optional_json_dict(
                self.figure_kwargs_json.toPlainText(),
                field_name="Figure kwargs",
            )
            axes_kwargs_value = _optional_json_dict(
                self.axes_kwargs_json.toPlainText(),
                field_name="Axes kwargs",
            )
            line_kwargs_value = _optional_json_dict(
                self.line_kwargs_json.toPlainText(),
                field_name="Global line kwargs",
            )
            legend_kwargs_value = _optional_json_dict(
                self.legend_kwargs_json.toPlainText(),
                field_name="Legend kwargs",
            )
            grid_kwargs_value = _optional_json_dict(
                self.grid_kwargs_json.toPlainText(),
                field_name="Grid kwargs",
            )
            tick_params_kwargs_value = _optional_json_dict(
                self.tick_params_kwargs_json.toPlainText(),
                field_name="Tick params kwargs",
            )
            tight_layout_kwargs_value = _optional_json_dict(
                self.tight_layout_kwargs_json.toPlainText(),
                field_name="tight_layout kwargs",
            )
            savefig_kwargs_value = _optional_json_dict(
                self.savefig_kwargs_json.toPlainText(),
                field_name="savefig kwargs",
            )
            figure_kwargs_merged = (
                dict(figure_kwargs_value) if isinstance(figure_kwargs_value, dict) else {}
            )
            facecolor = _explicit_text(self.figure_facecolor.text())
            if facecolor:
                figure_kwargs_merged["facecolor"] = facecolor
            else:
                figure_kwargs_merged.pop("facecolor", None)
            figure_kwargs_value = figure_kwargs_merged or None

            axes_kwargs_value = (
                dict(axes_kwargs_value) if isinstance(axes_kwargs_value, dict) else None
            )
            x_label_pad = _synced_float(
                "x_label_pad",
                self.x_label_pad,
                field_name="x-label pad",
            )
            y_label_pad = _synced_float(
                "y_label_pad",
                self.y_label_pad,
                field_name="y-label pad",
            )

            line_kwargs_merged = (
                dict(line_kwargs_value) if isinstance(line_kwargs_value, dict) else {}
            )
            line_style = self.line_style.currentText().strip()
            line_alpha = _optional_float(self.line_alpha.text(), field_name="line-alpha")
            marker_size = _optional_float(self.marker_size.text(), field_name="marker-size")
            if line_style:
                line_kwargs_merged["linestyle"] = line_style
            else:
                line_kwargs_merged.pop("linestyle", None)
            if line_alpha is not None:
                line_kwargs_merged["alpha"] = line_alpha
            else:
                line_kwargs_merged.pop("alpha", None)
            if marker_size is not None:
                line_kwargs_merged["markersize"] = marker_size
            else:
                line_kwargs_merged.pop("markersize", None)
            line_kwargs_value = line_kwargs_merged or None

            legend_kwargs_merged = (
                dict(legend_kwargs_value) if isinstance(legend_kwargs_value, dict) else {}
            )
            legend_frame = _mode_to_toggle(self.legend_frame_mode.currentText())
            legend_columns = _optional_int(self.legend_columns.text(), field_name="legend-columns")
            if legend_columns is not None and legend_columns < 1:
                raise ValueError("legend-columns must be >= 1.")
            if legend_frame is not None:
                legend_kwargs_merged["frameon"] = legend_frame
            if legend_columns is not None:
                legend_kwargs_merged["ncols"] = legend_columns
            else:
                legend_kwargs_merged.pop("ncols", None)
            legend_kwargs_value = legend_kwargs_merged or None

            grid_kwargs_merged = (
                dict(grid_kwargs_value) if isinstance(grid_kwargs_value, dict) else {}
            )
            grid_color = _explicit_text(self.grid_color.text())
            if grid_color:
                grid_kwargs_merged["color"] = grid_color
            else:
                grid_kwargs_merged.pop("color", None)
            grid_axis = self.grid_axis.currentText().strip().lower() or "both"
            if grid_axis not in _GRID_AXES:
                raise ValueError("Grid axis must be both, x, or y.")
            grid_kwargs_merged["axis"] = grid_axis
            grid_which = self.grid_which.currentText().strip().lower() or "major"
            if grid_which not in _GRID_WHICH:
                raise ValueError("Grid lines selector must be major, minor, or both.")
            grid_kwargs_merged["which"] = grid_which
            grid_kwargs_value = grid_kwargs_merged or None

            tick_params_kwargs_merged = (
                dict(tick_params_kwargs_value) if isinstance(tick_params_kwargs_value, dict) else {}
            )
            tick_direction = self.tick_direction.currentText().strip().lower() or "out"
            if tick_direction not in _TICK_DIRECTIONS:
                raise ValueError("Tick direction must be out, in, or inout.")
            tick_params_kwargs_merged["direction"] = tick_direction
            tick_length = _optional_float(self.tick_length.text(), field_name="tick-length")
            if tick_length is None or tick_length <= 0:
                tick_params_kwargs_merged.pop("length", None)
            else:
                tick_params_kwargs_merged["length"] = tick_length
            tick_width = _optional_float(self.tick_width.text(), field_name="tick-width")
            if tick_width is None or tick_width <= 0:
                tick_params_kwargs_merged.pop("width", None)
            else:
                tick_params_kwargs_merged["width"] = tick_width
            ticks_axis = self.ticks_visibility.currentText().strip().lower() or "both"
            if ticks_axis not in _TICK_VISIBILITY_MODES:
                raise ValueError("Ticks must be both, x, y, or none.")
            resolved_ticks_axis = "both" if ticks_axis == "none" else ticks_axis
            tick_params_kwargs_merged["_ticks_axis"] = resolved_ticks_axis
            tick_params_kwargs_merged["axis"] = resolved_ticks_axis
            minor_ticks_mode = self.minor_ticks_mode.currentText().strip().lower() or "off"
            if minor_ticks_mode not in _MINOR_TICKS_MODES:
                raise ValueError("Minor ticks mode must be on or off.")
            tick_params_kwargs_merged["_minor_ticks_mode"] = minor_ticks_mode
            tick_params_kwargs_value = tick_params_kwargs_merged or None

            savefig_kwargs_merged = (
                dict(savefig_kwargs_value) if isinstance(savefig_kwargs_value, dict) else {}
            )
            transparent = _mode_to_toggle(self.transparent_mode.currentText())
            if transparent is not None:
                savefig_kwargs_merged["transparent"] = transparent
            savefig_kwargs_value = savefig_kwargs_merged or None

            x_bin_width = _optional_float(self.x_bin_width.text(), field_name="x-bin-width")
            if x_bin_width is not None and x_bin_width <= 0:
                raise ValueError("x-bin-width must be positive.")
            min_bin_points = _optional_int(
                self.min_bin_points.text(),
                field_name="min-points-per-bin",
            )
            if min_bin_points is not None and min_bin_points < 1:
                raise ValueError("min-points-per-bin must be >= 1.")

            y_bin_width: float | None = None
            if hasattr(self, "y_bin_width"):
                y_bin_width = _optional_float(self.y_bin_width.text(), field_name="y-bin-width")
                if y_bin_width is not None and y_bin_width <= 0:
                    raise ValueError("y-bin-width must be positive.")

            x_min = _synced_float("x_lim", self.x_min, field_name="x-min")
            x_max = _synced_float("x_lim", self.x_max, field_name="x-max")
            y_min = _synced_float("y_lim", self.y_min, field_name="y-min")
            y_max = _synced_float("y_lim", self.y_max, field_name="y-max")
            title_value = _synced_text("title", self.title_text)
            title_visible_value: bool | None = None
            title_mode = self._synced_field_mode("title")
            if title_mode == "off":
                title_visible_value = False
                title_value = None
            elif title_mode == "manual":
                title_visible_value = bool(title_value)
                if not title_visible_value:
                    title_value = None

            settings = {
                "title": title_value,
                "x_label": _synced_text("x_label", self.x_label),
                "y_label": _synced_text("y_label", self.y_label),
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "x_scale": self.x_scale.currentText().strip() or "linear",
                "y_scale": self.y_scale.currentText().strip() or "linear",
                "x_ticks": _synced_float_list("x_ticks", self.x_ticks, field_name="x-ticks"),
                "y_ticks": _synced_float_list("y_ticks", self.y_ticks, field_name="y-ticks"),
                "x_tick_rotation": _optional_float(
                    self.x_tick_rotation.text(), field_name="x-tick-rotation"
                ),
                "y_tick_rotation": _optional_float(
                    self.y_tick_rotation.text(), field_name="y-tick-rotation"
                ),
                "title_visible": title_visible_value,
                "legend": _mode_to_toggle(self.legend_mode.currentText()),
                "border": (
                    False if self.axes_border_mode.currentText().strip().lower() == "off"
                    else {"left": self.border_left.isChecked(), "right": self.border_right.isChecked(),
                          "top": self.border_top.isChecked(), "bottom": self.border_bottom.isChecked()}
                    if self.axes_border_mode.currentText().strip().lower() == "custom"
                    else True
                ),
                "grid": _mode_to_toggle(self.grid_mode.currentText()),
                "ticks": self.ticks_visibility.currentText().strip().lower() != "none",
                "markers": _mode_to_toggle(self.markers_mode.currentText()),
                "legend_title": _explicit_text(self.legend_title.text()) or None,
                "legend_loc": self.legend_loc.currentText().strip() or "best",
                "figsize": figsize,
                "dpi": _optional_positive_int_or_none(self.dpi.text(), field_name="dpi"),
                "font_family": _explicit_text(self.font_family.text()) or None,
                "font_size": _optional_positive_int_or_none(
                    self.base_font_size.text(), field_name="font-size"
                ),
                "title_font_size": _optional_positive_int_or_none(
                    self.title_font.text(), field_name="title-font-size"
                ),
                "label_font_size": _optional_positive_int_or_none(
                    self.label_font.text(), field_name="label-font-size"
                ),
                "tick_font_size": _optional_positive_int_or_none(
                    self.tick_font.text(), field_name="tick-font-size"
                ),
                "legend_font_size": _optional_positive_int_or_none(
                    self.legend_font.text(), field_name="legend-font-size"
                ),
                "line_width": _optional_float(self.line_width.text(), field_name="line-width"),
                "line_colors": line_colors_value,
                "series_labels": series_labels,
                "series_order": (
                    self._current_series_id_order()
                    if self._current_series_id_order() != self._series_natural_order_data
                    else None
                ),
                "series_descriptors": deepcopy(self._series_descriptors_data),
                "series_overrides": series_overrides or None,
                "series_enabled": series_enabled_value,
                "series_show_in_legend": series_show_in_legend_value,
                "series_alpha": series_alpha_value,
                "series_line_widths": series_line_widths_value,
                "series_markers": series_markers_value,
                "series_line_kwargs": series_line_kwargs_value,
                "series_normalization_modes": normalization_modes_value,
                "series_normalization_values": normalization_values_value,
                "series_normalization_x_refs": normalization_x_refs_value,
                "annotations": annotations_value,
                "x_bin_width": x_bin_width,
                "x_bin_reducer": (self.x_bin_reducer.currentText().strip() or "mean")
                if x_bin_width is not None
                else None,
                "min_bin_points": min_bin_points,
                "y_bin_width": y_bin_width,
                "y_bin_reducer": (self.y_bin_reducer.currentText().strip() or "mean")
                if hasattr(self, "y_bin_reducer") and y_bin_width is not None
                else None,
                "grid_linestyle": _explicit_text(self.grid_linestyle.currentText()) or None,
                "grid_linewidth": _optional_float(
                    self.grid_linewidth.text(), field_name="grid-linewidth"
                ),
                "grid_alpha": _optional_float(self.grid_alpha.text(), field_name="grid-alpha"),
                "matplotlib_rc": matplotlib_rc_value,
                "figure_kwargs": figure_kwargs_value,
                "axes_kwargs": axes_kwargs_value,
                "x_label_pad": x_label_pad,
                "y_label_pad": y_label_pad,
                "line_kwargs": line_kwargs_value,
                "legend_kwargs": legend_kwargs_value,
                "grid_kwargs": grid_kwargs_value,
                "tick_params_kwargs": tick_params_kwargs_value,
                "tight_layout_kwargs": tight_layout_kwargs_value,
                "savefig_kwargs": savefig_kwargs_value,
                "_gui_sync_modes": {
                    key: self._synced_field_mode(key)
                    for key in _TRI_STATE_SYNC_FIELD_KEYS
                    if self._synced_field_mode(key) != "auto"
                }
                or None,
            }
            if hasattr(self, "analysis_species"):
                settings["species"] = _explicit_text(self.analysis_species.text()) or None
            if hasattr(self, "analysis_axis"):
                axis_value = self.analysis_axis.currentText().strip().lower()
                settings["axis"] = None if axis_value == "" else axis_value
            if hasattr(self, "density_x_mode"):
                settings["x_mode"] = self.density_x_mode.currentText().strip() or "distance"
            if hasattr(self, "density_quantity"):
                settings["quantity"] = self.density_quantity.currentText().strip() or "mass"
            if hasattr(self, "rdf_species_a"):
                settings["species_a"] = self._selected_profile_filter_value(
                    self.rdf_species_a,
                    default_label=_PROFILE_FILTER_METADATA_LABEL,
                )
            if hasattr(self, "rdf_species_b"):
                settings["species_b"] = self._selected_profile_filter_value(
                    self.rdf_species_b,
                    default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                )
            if hasattr(self, "coord_species_a"):
                settings["species_a"] = self._selected_profile_filter_value(
                    self.coord_species_a,
                    default_label=_PROFILE_FILTER_METADATA_LABEL,
                )
            if hasattr(self, "coord_species_b"):
                settings["species_b"] = self._selected_profile_filter_value(
                    self.coord_species_b,
                    default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                )
            if hasattr(self, "position_component"):
                settings["component"] = self.position_component.currentText().strip() or "distance"
            if hasattr(self, "position_map_color"):
                settings["map_color"] = self.position_map_color.currentText().strip() or "distance"
            if hasattr(self, "position_time_axis"):
                settings["time_axis"] = self.position_time_axis.currentText().strip() or "ps"
            if hasattr(self, "coordination_component"):
                settings["component"] = (
                    self.coordination_component.currentText().strip() or "distance"
                )
            if hasattr(self, "coordination_time_axis"):
                settings["time_axis"] = self.coordination_time_axis.currentText().strip() or "ps"
            if hasattr(self, "orientation_component"):
                settings["component"] = (
                    self.orientation_component.currentText().strip() or "average"
                )
            if hasattr(self, "orientation_angle"):
                settings["angle"] = self.orientation_angle.currentText().strip() or "polar"
            if hasattr(self, "heatmap_vmin"):
                settings["heatmap_vmin"] = _optional_float(
                    self.heatmap_vmin.text(), field_name="heatmap vmin"
                )
            if hasattr(self, "heatmap_vmax"):
                settings["heatmap_vmax"] = _optional_float(
                    self.heatmap_vmax.text(), field_name="heatmap vmax"
                )
            if hasattr(self, "heatmap_cmap"):
                settings["heatmap_cmap"] = self.heatmap_cmap.currentText().strip() or None
            if hasattr(self, "heatmap_normalization_mode"):
                normalization_label = self.heatmap_normalization_mode.currentText().strip()
                normalization_mode = _HEATMAP_NORMALIZATION_MODE_BY_LABEL.get(
                    normalization_label,
                    "counts",
                )
                settings["heatmap_normalization_mode"] = normalization_mode
                settings["heatmap_normalize"] = normalization_mode == "global_probability"
            if hasattr(self, "heatmap_log_scale"):
                settings["heatmap_log_scale"] = self.heatmap_log_scale.isChecked()
            if hasattr(self, "heatmap_colorbar_label"):
                raw = self.heatmap_colorbar_label.text().strip()
                if raw.lower() in {"none", "off"}:
                    settings["heatmap_colorbar_label"] = ""
                elif raw and raw.lower() != "auto":
                    settings["heatmap_colorbar_label"] = raw
                else:
                    settings["heatmap_colorbar_label"] = None
            if hasattr(self, "heatmap_colorbar_label_size"):
                settings["heatmap_colorbar_label_size"] = _optional_positive_int_or_none(
                    self.heatmap_colorbar_label_size.text(), field_name="colorbar label size"
                )
            if hasattr(self, "heatmap_colorbar_tick_size"):
                settings["heatmap_colorbar_tick_size"] = _optional_positive_int_or_none(
                    self.heatmap_colorbar_tick_size.text(), field_name="colorbar tick size"
                )
            if hasattr(self, "heatmap_colorbar_enabled"):
                settings["heatmap_colorbar_enabled"] = self.heatmap_colorbar_enabled.isChecked()
            if hasattr(self, "heatmap_colorbar_position"):
                val = self.heatmap_colorbar_position.currentText().strip().lower()
                settings["heatmap_colorbar_position"] = (
                    val if val in {"right", "left", "top", "bottom"} else "right"
                )
            if hasattr(self, "heatmap_colorbar_pad"):
                settings["heatmap_colorbar_pad"] = _optional_float(
                    self.heatmap_colorbar_pad.text(), field_name="colorbar padding"
                )
            if hasattr(self, "heatmap_colorbar_shrink"):
                settings["heatmap_colorbar_shrink"] = _optional_float(
                    self.heatmap_colorbar_shrink.text(), field_name="colorbar shrink"
                )
            if hasattr(self, "heatmap_colorbar_aspect"):
                settings["heatmap_colorbar_aspect"] = _optional_float(
                    self.heatmap_colorbar_aspect.text(), field_name="colorbar aspect"
                )
            return settings

        def _report_error(self, title_text: str, exc: Exception) -> None:
            self._status_label.setText(f"{title_text}: {exc}")
            self._refresh_shell_state()
            QMessageBox.critical(self, title_text, str(exc))

        def _handle_preview(self) -> None:
            self._preview_timer.stop()
            if self._preview_loading:
                return
            self._update_embedded_preview(interactive=True)

        def _handle_save(self) -> None:
            try:
                settings = self._collect_settings()
                message = on_save(self._current_profile_name, settings)
                self._status_label.setText(message)
                self._saved_signature = self._signature(settings)
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Save failed", exc)

        def _handle_save_figure(self) -> None:
            if on_save_figure is None:
                self._status_label.setText("Save-figure action is not available.")
                return
            try:
                settings = self._collect_settings()
                output_path, _selected = QFileDialog.getSaveFileName(
                    self,
                    "Save Figure",
                    self._figure_default_name,
                    self._figure_save_filters,
                )
                if not output_path:
                    self._status_label.setText("Save figure canceled.")
                    return
                result = on_save_figure(settings, output_path)
                message = result[0] if isinstance(result, tuple) else result
                render_state = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_state_to_synced_fields(render_state)
                self._status_label.setText(message)
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Save figure failed", exc)

        def _confirm_reset_defaults(self) -> bool:
            decision = QMessageBox.question(
                self,
                "Reset profile to defaults",
                "Reset the current profile values to the default plot settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return decision == QMessageBox.StandardButton.Yes

        def _handle_reset(self) -> None:
            if not self._confirm_reset_defaults():
                self._status_label.setText("Reset canceled.")
                self._refresh_shell_state()
                return
            baseline = self._resolved_default_profile_settings()
            self._suspend_preview_events = True
            try:
                self._populate(baseline)
            finally:
                self._suspend_preview_events = False
            self._refresh_widget_states()
            self._status_label.setText("Current profile reset to style defaults.")
            self._schedule_preview_update()
            self._refresh_shell_state()

        def _handle_import_json(self) -> None:
            if not self._confirm_context_change(action_label="importing settings"):
                return
            path_str, _selected = QFileDialog.getOpenFileName(
                self,
                "Import Plot Settings",
                "",
                "JSON files (*.json);;HDF5 files (*.h5 *.hdf5);;All files (*)",
            )
            if not path_str:
                return
            try:
                source_path = Path(path_str)
                suffix = source_path.suffix.lower()
                if suffix in {".h5", ".hdf5"}:
                    if on_import_hdf5 is None:
                        raise ValueError("HDF5 import is unavailable for this plot session.")
                    selected_profile_name: str | None = None
                    if on_list_import_hdf5_profiles is not None:
                        listing = on_list_import_hdf5_profiles(path_str)
                        available_names = listing.get("available_names")
                        active_name = str(listing.get("active_name") or "").strip() or None
                        if isinstance(available_names, list):
                            normalized_names = [
                                str(name).strip() for name in available_names if str(name).strip()
                            ]
                            if len(normalized_names) > 1:
                                if active_name in normalized_names:
                                    default_name = active_name
                                elif "Default" in normalized_names:
                                    default_name = "Default"
                                else:
                                    default_name = normalized_names[0]
                                selected_profile_name, accepted = QInputDialog.getItem(
                                    self,
                                    "Choose imported profile",
                                    "Saved profile",
                                    normalized_names,
                                    normalized_names.index(default_name),
                                    False,
                                )
                                if not accepted:
                                    self._status_label.setText("Import canceled.")
                                    self._refresh_shell_state()
                                    return
                                selected_profile_name = (
                                    str(selected_profile_name).strip() or default_name
                                )
                            elif len(normalized_names) == 1:
                                selected_profile_name = normalized_names[0]
                    payload = on_import_hdf5(path_str, selected_profile_name)
                    source_label = source_path.name
                else:
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                    source_label = source_path.name
                if not isinstance(payload, dict):
                    raise ValueError("JSON root must be an object with setting keys.")
                payload = _without_series_specific_settings(payload)
                merged = self._resolved_default_profile_settings()
                merged.update(payload)
                self._load_settings_into_editor(
                    merged,
                    status_message=f"Imported non-series settings into '{self._current_profile_name}' from '{source_label}'.",
                    mark_saved=False,
                )
                self._schedule_preview_update()
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Import failed", exc)

        def _handle_export_json(self) -> None:
            try:
                settings = self._collect_settings()
                settings = _without_series_specific_settings(settings)
                path_str, _selected = QFileDialog.getSaveFileName(
                    self,
                    "Export Plot Settings JSON",
                    f"{self._current_profile_name.lower().replace(' ', '_')}_plot_settings.json",
                    "JSON files (*.json)",
                )
                if not path_str:
                    self._status_label.setText("Export canceled.")
                    return
                output_path = Path(path_str)
                output_path.write_text(
                    json.dumps(settings, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                self._status_label.setText(f"Exported settings to '{output_path.name}'.")
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Export failed", exc)

        def _cleanup_preview_artifacts(self) -> None:
            self._preview_timer.stop()
            try:
                self._preview_image_path.unlink(missing_ok=True)
            except OSError:
                pass

        def _refresh_preview_after_layout(self) -> None:
            self._refresh_preview_pixmap()
            if self._preview_pixmap is None or self._preview_pixmap.isNull():
                self._update_embedded_preview(interactive=False)

        def showEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            super().showEvent(event)
            QTimer.singleShot(0, self._refresh_preview_after_layout)

        def changeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            if event.type() in {
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
            }:
                self._apply_theme_styles()
                if hasattr(self, "series_list") and self.series_list is not None:
                    for index in range(self.series_list.count()):
                        self._apply_series_list_item_visuals(self.series_list.item(index), index)
            super().changeEvent(event)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # pragma: no cover - UI flow
            preview_scroll = self._preview_scroll
            preview_label = self._preview_label
            preview_frame = self._preview_frame
            preview_viewport = preview_scroll.viewport() if preview_scroll is not None else None

            if watched in {preview_viewport, preview_label}:
                if event.type() == QEvent.Type.Wheel:
                    delta = int(event.angleDelta().y())
                    if delta != 0 and self._zoom_preview_at_viewport_pos(
                        event.position(),
                        direction=1 if delta > 0 else -1,
                    ):
                        event.accept()
                        return True
                if event.type() == QEvent.Type.MouseButtonDblClick:
                    self._set_preview_zoom(1.0)
                    self._refresh_preview_pixmap()
                    event.accept()
                    return True
            if watched in {preview_frame, preview_viewport} and event.type() == QEvent.Type.Resize:
                QTimer.singleShot(0, self._refresh_preview_pixmap)
            return super().eventFilter(watched, event)

        def resizeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            super().resizeEvent(event)
            self._refresh_preview_pixmap()

        def closeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            try:
                settings = self._collect_settings()
                current_signature = self._signature(settings)
            except Exception as exc:
                decision = QMessageBox.question(
                    self,
                    "Invalid settings",
                    "Current settings contain invalid values "
                    f"({exc}). Close anyway without saving?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if decision == QMessageBox.StandardButton.Yes:
                    self._cleanup_preview_artifacts()
                    event.accept()
                else:
                    event.ignore()
                return

            if self._saved_signature == current_signature:
                self._cleanup_preview_artifacts()
                event.accept()
                return

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Unsaved plot settings")
            box.setText("Save settings before closing?")
            box.setStandardButtons(
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Yes)
            decision = box.exec()

            if decision == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if decision == QMessageBox.StandardButton.Yes:
                try:
                    message = on_save(self._current_profile_name, settings)
                    self._status_label.setText(message)
                    self._saved_signature = current_signature
                except Exception as exc:
                    self._report_error("Save failed", exc)
                    event.ignore()
                    return
            self._cleanup_preview_artifacts()
            event.accept()

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = _PlotSettingsWindow()
    window.showMaximized()

    # Ensure window is on a visible screen (guards against stale geometry).
    if QApplication.screenAt(window.geometry().center()) is None:
        primary = QApplication.primaryScreen()
        if primary is not None:
            avail = primary.availableGeometry()
            window.move(avail.x() + 50, avail.y() + 50)

    app.exec()
