"""PySide6 GUI panel for interactive plot settings."""

from __future__ import annotations

import json
import queue
from copy import deepcopy
from dataclasses import dataclass
import html
from pathlib import Path
import re
import tempfile
import threading
import warnings
from collections.abc import Sequence
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from .data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PlotDataContract,
    PlotDimension,
    PlotQuantity,
    PlotViewMapping,
    PlotViewType,
    canonical_plot_view_id,
    plot_view_display_label,
)
from .contracts.position_contract import default_position_plot_data_contract
from .contracts.density_contract import (
    default_density_heatmap_plot_data_contract,
    default_density_plot_data_contract,
)
from .contracts.coordination_contract import default_coordination_plot_data_contract
from .contracts.orientation_contract import (
    default_orientation_heatmap_plot_data_contract,
    default_orientation_line_plot_data_contract,
)
from .contracts.potential_contract import default_potential_plot_data_contract
from .data_validation import generic_view_type_compatibility, visual_role_compatibility
from .fitting import coerce_fit_config, default_fit_config, supported_fit_types
from .mappings.position_mapping import (
    _position_quantity_id_from_token,
    position_mapping_preset,
    position_plot_options_to_view_mapping,
    position_view_mapping_to_plot_options,
)
from .mappings.density_mapping import (
    density_plot_options_to_view_mapping,
    resolve_density_plot_mapping,
    density_view_mapping_to_plot_options,
)
from .mappings.coordination_mapping import (
    coordination_plot_options_to_view_mapping,
    resolve_coordination_plot_mapping,
    coordination_view_mapping_to_plot_options,
)
from .mappings.orientation_mapping import (
    orientation_plot_options_to_view_mapping,
    resolve_orientation_plot_mapping,
    orientation_view_mapping_to_plot_options,
)
from .mappings.potential_mapping import (
    potential_plot_options_to_view_mapping,
    resolve_potential_plot_mapping,
    potential_view_mapping_to_plot_options,
)
from .profile_persistence import deserialize_plot_view_mapping, serialize_plot_view_mapping
from .plot_settings import profile_name_conflict_message
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
_TICK_DIRECTIONS = ("out", "in", "inout")
_MINOR_TICKS_MODES = ("off", "on")
_TOGGLE_MODES = ("on", "off")
_BORDER_MODES = ("on", "off", "custom")
_SYNC_MODES = ("Auto", "Manual")
_TEXT_SYNC_MODES = ("Auto", "Manual", "Off")
_MARKER_TYPES = (
    "",
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
)
_FIT_TYPES = supported_fit_types()
_ERROR_STATS = ("sample_sem", "sample_std", "block_sem", "block_std")
_ERROR_STAT_DISPLAY: dict[str, str] = {
    "sample_sem": "Sample SEM",
    "sample_std": "Sample Std. Dev.",
    "block_sem": "Block SEM",
    "block_std": "Block Std. Dev.",
}
_ERROR_STAT_INTERNAL: dict[str, str] = {v: k for k, v in _ERROR_STAT_DISPLAY.items()}
_ERROR_STYLES = ("band", "whiskers")
_ERROR_STYLE_DISPLAY: dict[str, str] = {
    "band": "Shaded band",
    "whiskers": "Whiskers",
}
_ERROR_STYLE_INTERNAL: dict[str, str] = {v: k for k, v in _ERROR_STYLE_DISPLAY.items()}
_GROUP_REDUCERS = ("mean", "median", "sum", "min", "max")
_INTEGRATION_SOURCES = ("Plotted data", "Raw profile data")
_INTEGRATION_SOURCE_BY_LABEL = {
    "Plotted data": "plotted",
    "Raw profile data": "raw",
}
_INTEGRATION_SOURCE_LABEL_BY_MODE = {
    value: label for label, value in _INTEGRATION_SOURCE_BY_LABEL.items()
}
_INTEGRATION_COLOR_MODES = ("Auto", "Custom")
_ANNOTATION_TYPES = ("text", "line", "arrow")
_ANNOTATION_COORD_SYSTEMS = ("axes", "data")
_ANNOTATION_LINE_STYLES = ("-", "--", "-.", ":")
_ANNOTATION_ARROW_STYLES = ("->", "-|>", "<->", "simple", "fancy")
_ANNOTATION_HORIZONTAL_ALIGN = ("left", "center", "right")
_ANNOTATION_VERTICAL_ALIGN = ("top", "center", "bottom", "baseline")
_POSITION_COMPONENT_LABELS = ("distance", "x", "y", "z", "2D Heatmap")
_POSITION_PROJECTION_QUANTITIES = ("x", "y", "z", "distance", "ps", "fs", "step", "frame")
_POSITION_PROJECTION_RENDER_MODES = ("Continuous quantity", "Species / layer")
_POSITION_RENDER_MODE_BACKEND_BY_LABEL = {
    "Continuous quantity": "color-scale",
    "Species / layer": "line-colors",
    "color-scale": "color-scale",
    "source colors": "line-colors",
    "line-colors": "line-colors",
}
_POSITION_RENDER_MODE_LABEL_BY_BACKEND = {
    "color-scale": "Continuous quantity",
    "line-colors": "Species / layer",
}

_PUBLIC_PLOT_VIEW_LABEL_BY_ID = {
    PLOT_VIEW_1D_LINE: plot_view_display_label(PLOT_VIEW_1D_LINE),
    PLOT_VIEW_2D_HEATMAP: plot_view_display_label(PLOT_VIEW_2D_HEATMAP),
}
_LEGACY_PLOT_VIEW_LABEL_ALIASES = {
    "Line 1D": PLOT_VIEW_1D_LINE,
    "1D": PLOT_VIEW_1D_LINE,
    "Heatmap 2D": PLOT_VIEW_2D_HEATMAP,
    "2D Map": PLOT_VIEW_2D_HEATMAP,
    "Trajectory 2D": PLOT_VIEW_2D_HEATMAP,
    "2D": PLOT_VIEW_2D_HEATMAP,
    "line_1d": PLOT_VIEW_1D_LINE,
    "heatmap_2d": PLOT_VIEW_2D_HEATMAP,
    "trajectory_2d": PLOT_VIEW_2D_HEATMAP,
    "plot_1d_line": PLOT_VIEW_1D_LINE,
    "plot_2d_heatmap": PLOT_VIEW_2D_HEATMAP,
}


def _plot_view_label_by_id(*, include_heatmap: bool) -> dict[str, str]:
    view_ids = [PLOT_VIEW_1D_LINE]
    if include_heatmap:
        view_ids.append(PLOT_VIEW_2D_HEATMAP)
    return {view_id: _PUBLIC_PLOT_VIEW_LABEL_BY_ID[view_id] for view_id in view_ids}


def _plot_view_id_by_label(*, include_heatmap: bool) -> dict[str, str]:
    label_by_id = _plot_view_label_by_id(include_heatmap=include_heatmap)
    id_by_label = {label: view_id for view_id, label in label_by_id.items()}
    for label, view_id in _LEGACY_PLOT_VIEW_LABEL_ALIASES.items():
        if include_heatmap or view_id == PLOT_VIEW_1D_LINE:
            id_by_label[label] = view_id
    return id_by_label


def _contract_has_public_heatmap_view(contract: PlotDataContract) -> bool:
    """Return whether a contract exposes a true public 2D Heatmap view.

    Legacy trajectory/scatter views are still accepted by compatibility adapters,
    but they should not make the normal GUI advertise `2D Heatmap`.
    """

    available = {str(view_type.id).strip() for view_type in contract.view_types}
    return bool({PLOT_VIEW_2D_HEATMAP, "heatmap_2d"} & available)


_POSITION_GUI_VIEW_TYPE_LABEL_BY_ID = _plot_view_label_by_id(include_heatmap=True)
_POSITION_GUI_VIEW_TYPE_ID_BY_LABEL = _plot_view_id_by_label(include_heatmap=True)
_POSITION_GUI_PRESET_LABEL_BY_ID = {
    "distance_vs_time": "Distance vs time",
    "x_y_trajectory": "X/Y view",
    "x_z_trajectory": "X/Z view",
    "y_z_trajectory": "Y/Z view",
}
_POSITION_GUI_PRESET_ID_BY_LABEL = {
    label: preset_id for preset_id, label in _POSITION_GUI_PRESET_LABEL_BY_ID.items()
}
_POSITION_GUI_TOKEN_BY_QUANTITY_ID = {
    "distance_to_surface": "distance",
    "x": "x",
    "y": "y",
    "z": "z",
    "time_ps": "ps",
    "time_fs": "fs",
    "step": "step",
    "frame_index": "frame",
}
_POTENTIAL_VIEW_TYPE_LABEL_BY_ID = _plot_view_label_by_id(include_heatmap=False)
_POTENTIAL_VIEW_TYPE_ID_BY_LABEL = _plot_view_id_by_label(include_heatmap=False)
_POTENTIAL_SERIES_LABEL_BY_ID = {
    "summary": "Summary (all series)",
}
_POTENTIAL_SERIES_ID_BY_LABEL = {
    label: quantity_id for quantity_id, label in _POTENTIAL_SERIES_LABEL_BY_ID.items()
}
_COORDINATION_VIEW_TYPE_LABEL_BY_ID = _plot_view_label_by_id(include_heatmap=True)
_COORDINATION_VIEW_TYPE_ID_BY_LABEL = _plot_view_id_by_label(include_heatmap=True)
_COORDINATION_LINE_X_QUANTITY_LABEL_BY_BACKEND = {
    "distance": "distance",
    "time": "time",
}
_COORDINATION_LINE_X_QUANTITY_BACKEND_BY_LABEL = {
    label: backend for backend, label in _COORDINATION_LINE_X_QUANTITY_LABEL_BY_BACKEND.items()
}
_ORIENTATION_VIEW_TYPE_LABEL_BY_ID = _plot_view_label_by_id(include_heatmap=True)
_ORIENTATION_VIEW_TYPE_ID_BY_LABEL = _plot_view_id_by_label(include_heatmap=True)
_ORIENTATION_LINE_QUANTITY_LABEL_BY_BACKEND = {
    "average": "Mean orientation",
    "density": "H2O density",
    "density-weighted": "Density-weighted orientation",
}
_ORIENTATION_LINE_QUANTITY_BACKEND_BY_LABEL = {
    label: backend for backend, label in _ORIENTATION_LINE_QUANTITY_LABEL_BY_BACKEND.items()
}
_ORIENTATION_LINE_QUANTITY_BACKEND_BY_LABEL.update(
    {
        "average": "average",
        "density": "density",
        "density-weighted": "density-weighted",
    }
)
_DENSITY_X_MODE_LABELS = ("Distance", "X", "Y", "Z")
_DENSITY_VIEW_TYPE_LABEL_BY_ID = _plot_view_label_by_id(include_heatmap=True)
_DENSITY_VIEW_TYPE_ID_BY_LABEL = _plot_view_id_by_label(include_heatmap=True)
_DENSITY_X_MODE_BY_LABEL = {
    "distance": "distance",
    "x": "x",
    "y": "y",
    "z": "z",
    "axis": "axis",
}
_AUTO_PREVIEW_DEBOUNCE_MS = 1000
_AUTO_PREVIEW_STYLE_DEBOUNCE_MS = 100
_AUTO_PREVIEW_SERIES_DEBOUNCE_MS = 150
_AUTO_PREVIEW_DATA_DEBOUNCE_MS = 650
_WORKSPACE_PANEL_WIDTH = 760
_WORKSPACE_PANEL_MIN_WIDTH = 520
_HEATMAP_VALUE_LABEL_BY_MODE = {
    "raw_counts": "Observation count per bin",
    "joint_probability_density": "Joint probability density",
    "conditional_probability_density": "Orientation distribution at each distance",
    "bulk_relative_enrichment": "Relative to bulk orientation",
}
_HEATMAP_VALUE_MODE_BY_LABEL = {
    label: mode for mode, label in _HEATMAP_VALUE_LABEL_BY_MODE.items()
}
_HEATMAP_VALUE_DESCRIPTION_BY_MODE = {
    "raw_counts": "C\u1d62\u2c7c. The color is the number of observations in each displayed bin.",
    "joint_probability_density": (
        "C\u1d62\u2c7c / (N \u0394x\u1d62 \u0394y\u2c7c). The complete heatmap integrates to one."
    ),
    "conditional_probability_density": (
        "C\u1d62\u2c7c / (C\u1d62* \u0394y\u2c7c). Every occupied distance row integrates to one."
    ),
    "bulk_relative_enrichment": (
        "The conditional orientation density divided by the pooled bulk distribution. "
        "A value of 1 is bulk-like."
    ),
}
_TEXT_SYNC_FIELD_KEYS = frozenset({"title", "x_label", "y_label"})
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
    "preview.fit": "Resets the preview view to the plotted data bounds.",
    "preview.actual_size": "Resets the preview zoom or view.",
    "preview.reset": "Restores the default plot settings.",
    "preview.auto_update": "Refreshes the preview after each change.",
    "profiles.selector": "Chooses which saved plot profile is active.",
    "profiles.new": "Creates a new profile from the defaults.",
    "profiles.rename": "Renames the current profile.",
    "profiles.duplicate": "Copies the current profile under a new name.",
    "profiles.delete": "Deletes the current profile.",
    "profiles.save": "Saves the current settings to this profile.",
    "profiles.undo": "Reverts the most recent GUI change in this session.",
    "profiles.redo": "Reapplies the most recently undone GUI change in this session.",
    "profiles.reset": "Restores the default plot settings.",
    "profiles.import": "Loads profile settings from a JSON file.",
    "profiles.export_json": "Saves this profile to a JSON file.",
    "export.transparent": "Saves the figure with a transparent background.",
    "export.figure": "Saves the current figure to an image file.",
    "export.data": "Saves the current preview line data to a text data file.",
    "data.density.x_values": "Chooses which source quantity is assigned to the x role.",
    "data.density.quantity": "Chooses which density quantity is assigned to the y role.",
    "data.profile.species": "Filters the stored profile by species.",
    "data.profile.axis": "Filters the stored profile by axis.",
    "data.rdf.layer_count": "Shows how many RDF profiles were loaded as plot layers from the current source.",
    "data.rdf.pairs": "Lists the RDF profile labels currently available as layers.",
    "data.coordination.species_a": "Chooses the center species in the stored profile.",
    "data.coordination.species_b": "Chooses the neighbor species in the stored profile.",
    "data.coordination.axis": "Chooses which stored axis profile to load.",
    "data.density.source.contract": "Shows the shared plot-data contract detected for the current density source.",
    "data.density.source.dimensions": "Shows the logical dimensions available for density plotting.",
    "data.density.source.quantities": "Shows the quantities exposed by the current density plot-data contract.",
    "data.density.target": "Chooses which loaded density species or molecule target is visible in the preview.",
    "data.density.summary.status": "Shows whether the current density mapping is preferred, supported, or invalid for the active contract.",
    "data.density.summary.mapping": "Shows the current density mapping in generic view-role form.",
    "data.density.summary.backend": "Shows how the current density mapping is translated into backend plotting options.",
    "data.position.component": "Chooses the active position view type or line Y quantity.",
    "data.position.source.contract": "Shows the shared plot-data contract detected for the current position source.",
    "data.position.source.dimensions": "Shows the logical dimensions available for position plotting.",
    "data.position.source.quantities": "Shows the quantities exposed by the current position plot-data contract.",
    "data.position.mapping.preset": "Applies a default mapping for common position views.",
    "data.position.mapping.view_type": "Chooses which generic plot view to map onto the current position data.",
    "data.position.mapping.x": "Chooses which quantity is assigned to the x visual role.",
    "data.position.mapping.y": "Chooses which quantity is assigned to the y visual role.",
    "data.position.mapping.value": "Chooses which quantity supplies heatmap color values and optional range filtering.",
    "data.position.mapping.split_by": "Shows which dimension the current position mapping splits into separate plotted series.",
    "data.position.color_by": "Chooses which quantity is used for heatmap coloring or range filtering.",
    "data.position.xy_z_distance_max": "Legacy distance cutoff for the 2D heatmap view. Use the range controls for the general case.",
    "data.position.projection_x": "Chooses which quantity is shown on the horizontal axis in 2D Heatmap mode.",
    "data.position.projection_y": "Chooses which quantity is shown on the vertical axis in 2D Heatmap mode.",
    "data.position.projection_render_mode": "Chooses between a continuous colormap and source-colored paths.",
    "data.position.projection_range_min": "Optional lower bound for the selected 2D Heatmap value quantity.",
    "data.position.projection_range_max": "Optional upper bound for the selected 2D Heatmap value quantity.",
    "data.position.time_axis": "Chooses the time unit on the x-axis.",
    "data.position.summary.status": "Shows whether the current generic mapping is preferred, merely supported, or invalid for the current contract.",
    "data.position.summary.mapping": "Shows the current position mapping in a compact generic form.",
    "data.position.summary.backend": "Shows how the current generic mapping is translated back into the existing position plotting backend.",
    "data.coordination.view_type": "Chooses whether coordination is shown as a 1D Line or 2D Heatmap.",
    "data.coordination.x_quantity": "Chooses the x-axis quantity for 1D Line coordination plots.",
    "data.coordination.time_axis": "Chooses which time quantity is assigned to the x role when time is used.",
    "data.coordination.source.contract": "Shows the shared plot-data contract detected for the current coordination source.",
    "data.coordination.source.dimensions": "Shows the logical dimensions available for coordination plotting.",
    "data.coordination.source.quantities": "Shows the quantities exposed by the current coordination plot-data contract.",
    "data.coordination.summary.status": "Shows whether the current coordination mapping is preferred, supported, or invalid for the active contract.",
    "data.coordination.summary.mapping": "Shows the current coordination mapping in generic view-role form.",
    "data.coordination.summary.backend": "Shows how the current coordination mapping is translated into backend plotting options.",
    "data.orientation.view_type": "Chooses whether orientation is shown as a 1D Line or 2D Heatmap.",
    "data.orientation.y_quantity": "Chooses the line Y quantity for orientation 1D Line plots.",
    "data.orientation.angle": "Chooses the orientation angle quantity used by the active mapping.",
    "data.orientation.source.contract": "Shows the shared plot-data contract detected for the current orientation source.",
    "data.orientation.source.dimensions": "Shows the logical dimensions available for the active orientation view type.",
    "data.orientation.source.quantities": "Shows the quantities exposed by the active orientation plot-data contract.",
    "data.orientation.summary.status": "Shows whether the current orientation mapping is preferred, supported, or invalid for the active contract.",
    "data.orientation.summary.mapping": "Shows the current orientation mapping in generic view-role form.",
    "data.orientation.summary.backend": "Shows how the current orientation mapping is translated into backend plotting options.",
    "data.potential.x_axis": "Shows which value is used on the x-axis.",
    "data.potential.total_rows": "Shows how many records were loaded.",
    "data.potential.complete_rows": "Shows how many records have complete values.",
    "data.potential.incomplete_rows": "Shows how many records have missing values.",
    "data.potential.source.contract": "Shows the shared plot-data contract detected for the current potential source.",
    "data.potential.source.dimensions": "Shows the logical dimensions available for potential plotting.",
    "data.potential.source.quantities": "Shows the quantities exposed by the current potential plot-data contract.",
    "data.potential.mapping.view_type": "Shows the fixed potential plot view.",
    "data.potential.mapping.series": "Chooses which potential quantity is assigned to the y role in line mode.",
    "data.potential.summary.status": "Shows whether the current potential mapping is preferred, supported, or invalid for the active contract.",
    "data.potential.summary.mapping": "Shows the current potential mapping in generic view-role form.",
    "data.potential.summary.backend": "Shows how the current potential mapping is translated into backend plotting options.",
    "data.section.width": "Groups nearby x-values into wider display bins. The helper note below shows the source bin size and requested display bin size.",
    "data.section.reducer": "Chooses how each section is summarized.",
    "data.section.min_points": "Requires at least this many contributing raw points in a plotted bin before LiNaK shows the value, uncertainty, or fit input for that bin. The helper note below summarizes the current points-per-bin distribution.",
    "data.section.y_width": "Groups nearby y-values into wider display bins for 2D views.",
    "data.section.y_reducer": "Chooses how each y-section is summarized.",
    "series.all_on": "Turns every series on.",
    "series.all_off": "Turns every series off.",
    "series.duplicate": "Duplicates the selected base or grouped series.",
    "series.add_group": "Adds a grouped series that can aggregate several loaded base series.",
    "series.show_in_legend": "Shows or hides this series in the legend.",
    "series.show_raw_line": "When off, the raw data line is hidden. Trendlines and cumulative averages remain visible if enabled.",
    "series.label": "Sets the legend name for this series.",
    "series.color": "Sets the line color for this series.",
    "series.alpha": "Sets the line transparency for this series.",
    "series.line_width": "Sets the line width for this series.",
    "series.marker": "Sets the marker shape for this series.",
    "series.error_enabled": "Toggle the uncertainty overlay for this series.",
    "series.error_stat": "Select the uncertainty measure. Sample-based statistics reflect frame-to-frame variation; block-based statistics use averages over contiguous trajectory blocks.",
    "series.error_style": "Display uncertainty as a shaded band or as whisker bars.",
    "series.error_color": "Set a custom color for the uncertainty overlay, or leave blank to match the series color.",
    "series.error.summary": "Summary of the active uncertainty overlay, including the statistic used, its data source, and any availability notes.",
    "series.fit_enabled": "Turns fitting for this series on or off.",
    "series.fit_type": "Chooses the fitting model.",
    "series.fit_degree": "Sets the polynomial degree.",
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
    "figure.text.title": "Sets the plot title.",
    "figure.text.x_label": "Sets the x-axis label.",
    "figure.text.y_label": "Sets the y-axis label.",
    "figure.text.title_font": "Sets the title font size.",
    "figure.text.title_pad": "Sets the space between the graph and the title in points.",
    "figure.text.label_font": "Sets the axis label font size.",
    "figure.legend.enabled": "Shows or hides the legend.",
    "figure.legend.title": "Sets the legend title.",
    "figure.legend.location": "Chooses where the legend is placed.",
    "figure.legend.frame": "Shows or hides the legend box.",
    "figure.legend.columns": "Sets how many columns the legend uses.",
    "figure.legend.font": "Sets the legend font size.",
    "figure.axes.x_scale": "Chooses the x-axis scale.",
    "figure.axes.x_axis_scale": "Multiplies displayed x-values by this factor. Use 0.2 to show 100 count units as 20 display units.",
    "figure.axes.x_axis_offset": "Adds this offset after x-axis scaling.",
    "figure.axes.y_scale": "Chooses the y-axis scale.",
    "figure.axes.border": "Controls the plot border: 'on' shows all four spines, 'off' hides them all, 'custom' lets you choose each side individually.",
    "figure.axes.border_sides": "Choose which individual spines to show when border mode is set to 'custom'.",
    "figure.axes.label_font": "Sets the axis label font size.",
    "figure.axes.x_limits": "Sets the x-axis limits.",
    "figure.axes.y_limits": "Sets the y-axis limits.",
    "figure.axes.x_label_pad": "Sets the space below the x-axis label.",
    "figure.axes.y_label_pad": "Sets the space beside the y-axis label.",
    "figure.axes.x_label_font": "Sets the x-axis label font size.",
    "figure.axes.y_label_font": "Sets the y-axis label font size.",
    "figure.ticks.show": "Shows or hides ticks for this axis.",
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
    "figure.lines.marker_type": "Sets the default marker shape.",
    "figure.lines.marker_color": "Sets the default marker color.",
    "figure.integration.enabled": "Turns on a shaded integral region for line plots.",
    "figure.integration.source": "Chooses whether the integral uses the plotted data or stored profile data before GUI transforms.",
    "figure.integration.range": "Sets the x-range for integration. Leave blank to use the target series range.",
    "figure.integration.baseline": "Sets the baseline subtracted before integration and used for the shaded fill.",
    "figure.integration.color_mode": "Chooses whether the shaded fill uses the target line color or a custom color.",
    "figure.integration.color": "Sets the custom integration fill color.",
    "figure.integration.alpha": "Sets the integration fill opacity between 0 and 1.",
    "figure.integration.summary": "Shows integration results from the latest preview; this text is not drawn into the figure.",
    "figure.heatmap.vmin": "Minimum value for the colorbar range. Leave blank for auto.",
    "figure.heatmap.vmax": "Maximum value for the colorbar range. Leave blank for auto.",
    "figure.heatmap.cmap": "Matplotlib colormap name for the heatmap.",
    "figure.heatmap.value_mode": "Choose the scientific values represented by the colors. This transformation runs after count aggregation and rebinning.",
    "figure.heatmap.bulk_reference": "Select the bulk orientation reference automatically from the density plateau or enter a manual distance range.",
    "figure.heatmap.bulk_range": "Distance bounds for the pooled manual bulk orientation reference.",
    "figure.heatmap.log_scale": "Choose linear or logarithmic color mapping. Logarithmic mapping masks zero and negative cells but does not transform the represented data.",
    "figure.heatmap.trajectory_width": "Set one uniform stroke width for all continuously colored 2D position trajectories.",
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
    "figure.canvas.font_color": "Sets the base color for title, labels, ticks, and legend text.",
    "figure.canvas.facecolor": "Sets the figure background color.",
    "figure.canvas.alpha": "Sets the figure background opacity from 0 to 1.",
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
    is_text = normalized_type == "text"
    defaults: dict[str, Any] = {
        "id": f"annotation:{uuid4().hex}",
        "type": normalized_type,
        "enabled": True,
        "name": _default_annotation_name(normalized_type, index=index),
        "coord_system": "axes",
        "color": "#000000",
        "alpha": "1.0",
        "zorder": "5",
        "text": _default_annotation_name(normalized_type, index=index) if is_text else "",
        "x": "0.5",
        "y": "0.92" if is_text else "0.5",
        "font_size": "12",
        "rotation": "0",
        "horizontal_align": "center",
        "vertical_align": "center",
        "x1": "0.15",
        "y1": "0.2",
        "x2": "0.85",
        "y2": "0.8",
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


def _partition_series_ids_for_display_order(
    series_ids: list[str],
    *,
    enabled_by_id: dict[str, bool],
    group_by_id: dict[str, bool],
) -> list[str]:
    enabled_non_group_ids = [
        series_id
        for series_id in series_ids
        if enabled_by_id.get(series_id, True) and not group_by_id.get(series_id, False)
    ]
    enabled_group_ids = [
        series_id
        for series_id in series_ids
        if enabled_by_id.get(series_id, True) and group_by_id.get(series_id, False)
    ]
    disabled_ids = [series_id for series_id in series_ids if not enabled_by_id.get(series_id, True)]
    return enabled_non_group_ids + enabled_group_ids + disabled_ids


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
        source_kind = (
            "group" if str(raw.get("source_kind") or "").strip().lower() == "group" else "source"
        )
        source_series_id = str(raw.get("source_series_id") or "").strip()
        descriptors.append(
            {
                "series_id": series_id,
                "default_label": default_label,
                "source_kind": source_kind,
                "source_series_id": (
                    None if source_kind == "group" else source_series_id or series_id
                ),
                "is_generated": bool(raw.get("is_generated", source_kind == "group")),
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
        if key not in _SYNCED_FIELD_KEYS:
            continue
        mode = str(raw_value).strip().lower()
        allowed_modes = (
            {"auto", "manual", "off"} if key in _TEXT_SYNC_FIELD_KEYS else {"auto", "manual"}
        )
        if mode in allowed_modes:
            resolved[key] = mode
    return resolved


def _fit_defaults_for_gui() -> dict[str, Any]:
    config = default_fit_config()
    config["fit_type"] = "linear"
    config["fit_degree"] = 2
    config["fit_range_mode"] = "visible"
    return config


def _fit_range_mode_from_limits(x_min: Any, x_max: Any) -> str:
    return "manual" if str(x_min or "").strip() or str(x_max or "").strip() else "visible"


def _integration_defaults_for_gui() -> dict[str, Any]:
    return {
        "enabled": False,
        "source": "plotted",
        "x_min": None,
        "x_max": None,
        "baseline": 0.0,
        "color": None,
        "alpha": 0.25,
    }


def _coerce_series_integration_config(value: Any) -> dict[str, Any]:
    config = _integration_defaults_for_gui()
    if not isinstance(value, dict):
        return config
    if "enabled" in value:
        config["enabled"] = bool(value.get("enabled"))
    source = str(value.get("source") or "").strip().lower()
    if source in ("plotted", "raw"):
        config["source"] = source
    for key in ("x_min", "x_max", "baseline"):
        if value.get(key) is not None:
            try:
                config[key] = float(value[key])
            except (ValueError, TypeError):
                pass
    if value.get("color") is not None:
        config["color"] = str(value.get("color")).strip() or None
    alpha_value = value.get("alpha")
    if alpha_value is not None:
        try:
            config["alpha"] = float(alpha_value)
        except (ValueError, TypeError):
            pass
    return config


def _cumulative_defaults_for_gui() -> dict[str, Any]:
    return {
        "enabled": False,
        "label_override": None,
        "show_in_legend": True,
        "color": None,
        "alpha": None,
        "line_width": None,
        "line_style": None,
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
    if value.get("color") is not None:
        config["color"] = str(value.get("color")).strip() or None
    alpha_value = value.get("alpha")
    if alpha_value is not None:
        try:
            config["alpha"] = str(float(alpha_value))
        except (ValueError, TypeError):
            pass
    line_width_value = value.get("line_width")
    if line_width_value is not None:
        try:
            config["line_width"] = str(float(line_width_value))
        except (ValueError, TypeError):
            pass
    if value.get("line_style") is not None:
        config["line_style"] = str(value.get("line_style")).strip() or None
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
    """Return the default derived label for one uncertainty child series."""
    resolved_label = str(base_label).strip() or "Series"
    resolved_stat = _resolve_error_stat_for_available(stat, None)
    friendly = _ERROR_STAT_DISPLAY.get(resolved_stat, resolved_stat)
    return f"{resolved_label} \u00b1{friendly}"


def _error_supported_for_view(
    analysis: str,
    *,
    orientation_heatmap: bool = False,
    position_component: str = "distance",
    coordination_component: str = "distance",
) -> bool:
    """Return whether one GUI view supports 1-D error overlays and min-bin masking."""
    normalized_analysis = str(analysis).strip().lower()
    if normalized_analysis in {"density", "msd", "rdf", "potential", "temperature"}:
        return True
    if normalized_analysis == "orientation":
        return not bool(orientation_heatmap)
    if normalized_analysis == "position":
        return not _is_position_projection_component(position_component)
    if normalized_analysis == "coordination":
        return str(coordination_component).strip().lower() != "time-distance"
    return False


def _is_position_projection_component(value: Any) -> bool:
    token = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    return token in {
        "xy-z",
        "xy-z-color",
        "xy-z-colormap",
        "trajectory",
        "xyz",
        "2d-projection",
        "2dprojection",
        "projection-2d",
        "projection2d",
        "projection",
        "2d",
    }


def _plot_data_contract_from_payload(value: Any) -> PlotDataContract | None:
    if not isinstance(value, dict):
        return None
    dimensions = tuple(
        PlotDimension(
            id=str(item.get("id") or ""),
            label=str(item.get("label") or ""),
            kind=str(item.get("kind") or ""),
            length=(
                None if item.get("length") is None else int(item.get("length"))
            ),
            unit=None if item.get("unit") is None else str(item.get("unit")),
        )
        for item in value.get("dimensions", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    )
    quantities = tuple(
        PlotQuantity(
            id=str(item.get("id") or ""),
            label=str(item.get("label") or ""),
            kind=str(item.get("kind") or ""),
            dimensions=tuple(str(token) for token in item.get("dimensions", []) if str(token).strip()),
            unit=None if item.get("unit") is None else str(item.get("unit")),
            source_name=(
                None if item.get("source_name") is None else str(item.get("source_name"))
            ),
        )
        for item in value.get("quantities", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    )
    view_types = tuple(
        PlotViewType(
            id=str(item.get("id") or ""),
            label=str(item.get("label") or ""),
            kind=str(item.get("kind") or ""),
            supported_roles=tuple(
                str(token) for token in item.get("supported_roles", []) if str(token).strip()
            ),
        )
        for item in value.get("view_types", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    )
    if not quantities:
        return None
    return PlotDataContract.from_items(
        source_id=str(value.get("source_id") or "position"),
        label=str(value.get("label") or "Position data"),
        dimensions=dimensions,
        quantities=quantities,
        view_types=view_types,
        default_view_type_id=(
            None
            if value.get("default_view_type_id") is None
            else str(value.get("default_view_type_id"))
        ),
    )


def _fallback_position_plot_data_contract() -> PlotDataContract:
    return default_position_plot_data_contract()


def _fallback_density_plot_data_contract() -> PlotDataContract:
    return default_density_plot_data_contract()


def _fallback_density_heatmap_plot_data_contract() -> PlotDataContract:
    return default_density_heatmap_plot_data_contract()


def _fallback_coordination_plot_data_contract() -> PlotDataContract:
    return default_coordination_plot_data_contract()


def _fallback_orientation_line_plot_data_contract() -> PlotDataContract:
    return default_orientation_line_plot_data_contract()


def _fallback_orientation_heatmap_plot_data_contract() -> PlotDataContract:
    return default_orientation_heatmap_plot_data_contract()


def _fallback_potential_plot_data_contract() -> PlotDataContract:
    return default_potential_plot_data_contract()


def _position_quantity_token(quantity_id: str | None) -> str:
    token = _POSITION_GUI_TOKEN_BY_QUANTITY_ID.get(str(quantity_id or "").strip())
    if token is None:
        raise ValueError(f"Unsupported position quantity id '{quantity_id}'.")
    return token


def _mapping_status_label(status: str) -> str:
    if status == "valid_preferred":
        return "Preferred"
    if status == "valid_nonpreferred":
        return "Supported"
    return "Invalid"


def _contract_dimensions_text(contract: PlotDataContract) -> str:
    parts: list[str] = []
    for dimension in contract.dimensions:
        token = str(dimension.id)
        if dimension.length is not None:
            token = f"{token}={dimension.length}"
        parts.append(token)
    return ", ".join(parts) if parts else "n/a"


def _contract_quantities_text(contract: PlotDataContract) -> str:
    quantities = [str(quantity.id) for quantity in contract.quantities if str(quantity.id).strip()]
    return ", ".join(quantities) if quantities else "n/a"


def _mapping_summary_text(mapping: PlotViewMapping) -> str:
    roles = mapping.resolved_role_assignments()
    ordered_roles = ("x", "y", "z", "color", "split_by", "filter_by")
    parts: list[str] = []
    for role in ordered_roles:
        value = roles.get(role)
        if value is None:
            continue
        token = f"{role}={value}"
        if role == "filter_by" and (mapping.filter_min is not None or mapping.filter_max is not None):
            token += f" [{'' if mapping.filter_min is None else mapping.filter_min}, {'' if mapping.filter_max is None else mapping.filter_max}]"
        parts.append(token)
    if not parts:
        return mapping.view_type_id
    return f"{mapping.view_type_id}: " + ", ".join(parts)


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
    show_integration: bool
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
    show_axis_transforms: bool
    show_advanced_legend: bool
    show_advanced_lines: bool


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
    if normalized_analysis == "position" and _is_position_projection_component(position_component):
        return "heatmap"
    if normalized_analysis == "coordination":
        component = str(coordination_component).strip().lower()
        if component == "time-distance":
            return "heatmap"
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
    for key in _SYNCED_FIELD_KEYS:
        resolved[key] = metadata.get(key, inferred[key])
    return resolved


def _coerce_profile_filter_options(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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

    if not _settings_use_heatmap_rendering(settings):
        visible_modes: list[str] = []
        series_overrides = settings.get("series_overrides")
        series_descriptors = settings.get("series_descriptors")
        if isinstance(series_overrides, dict) and isinstance(series_descriptors, list):
            for raw_descriptor in series_descriptors:
                if not isinstance(raw_descriptor, dict):
                    continue
                series_id = str(raw_descriptor.get("series_id") or "").strip()
                if not series_id:
                    continue
                entry = series_overrides.get(series_id)
                if not isinstance(entry, dict):
                    continue
                if entry.get("enabled") is False:
                    continue
                visible_modes.append(str(entry.get("normalization_mode") or "none").strip().lower())
        else:
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


def _settings_use_heatmap_rendering(settings: dict[str, Any]) -> bool:
    mapping = settings.get("view_mapping")
    if isinstance(mapping, PlotViewMapping):
        return canonical_plot_view_id(mapping.view_type_id) == PLOT_VIEW_2D_HEATMAP
    if isinstance(mapping, dict):
        try:
            resolved_mapping = deserialize_plot_view_mapping(mapping)
        except ValueError:
            resolved_mapping = None
        if resolved_mapping is not None:
            return canonical_plot_view_id(resolved_mapping.view_type_id) == PLOT_VIEW_2D_HEATMAP
    if str(settings.get("component") or "").strip().lower() == "heatmap":
        return True
    return False


def _density_backend_summary_text(*, view_type_id: str, x_mode: str, quantity: str) -> str:
    normalized_view_type = canonical_plot_view_id(view_type_id)
    normalized_quantity = str(quantity or "").strip().lower() or "mass"
    if normalized_view_type == PLOT_VIEW_2D_HEATMAP:
        return (
            f"view type={plot_view_display_label(PLOT_VIEW_2D_HEATMAP)}, "
            f"source field={normalized_quantity}_density_2d"
        )
    return (
        f"view type={plot_view_display_label(PLOT_VIEW_1D_LINE)}, "
        f"x role={str(x_mode or '').strip().lower() or 'distance'}, "
        f"y role={normalized_quantity}_density"
    )


def _coordination_backend_summary_text(*, component: str, time_axis: str) -> str:
    normalized_component = str(component or "").strip().lower() or "distance"
    if normalized_component == "distance":
        return (
            f"view type={plot_view_display_label(PLOT_VIEW_1D_LINE)}, "
            "x role=distance_to_surface, y role=coordination_number"
        )
    view_type = PLOT_VIEW_2D_HEATMAP if normalized_component == "time-distance" else PLOT_VIEW_1D_LINE
    y_role = "distance_to_surface" if normalized_component == "time-distance" else "coordination_number"
    color_role = (
        ", color role=coordination_number"
        if normalized_component == "time-distance"
        else ""
    )
    return (
        f"view type={plot_view_display_label(view_type)}, "
        f"x role=time ({str(time_axis or '').strip().lower() or 'ps'}), "
        f"y role={y_role}{color_role}"
    )


def _orientation_backend_summary_text(*, component: str, angle: str, is_heatmap: bool) -> str:
    normalized_component = str(component or "").strip().lower() or "average"
    normalized_angle = str(angle or "").strip().lower() or "polar"
    view_type = PLOT_VIEW_2D_HEATMAP if is_heatmap else PLOT_VIEW_1D_LINE
    if is_heatmap:
        return (
            f"view type={plot_view_display_label(view_type)}, "
            f"color quantity=mean cos({normalized_angle})"
        )
    return (
        f"view type={plot_view_display_label(view_type)}, "
        f"Y quantity={normalized_component}, angle quantity={normalized_angle}"
    )


def _potential_backend_summary_text(
    *,
    view_type: str,
    y_quantity: str,
    standard_plot: str,
) -> str:
    normalized_standard_plot = str(standard_plot or "").strip().lower()
    if normalized_standard_plot == "summary":
        return f"view type={plot_view_display_label(PLOT_VIEW_1D_LINE)}, y role=summary"
    return (
        f"view type={plot_view_display_label(PLOT_VIEW_1D_LINE)}, "
        f"y role={str(y_quantity or '').strip().lower() or 'water_bulk_potential'}"
    )


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


def _data_filetype_filters() -> tuple[str, str]:
    return (
        "CSV data (*.csv);;DAT data (*.dat);;TSV data (*.tsv);;Text data (*.txt);;All files (*)",
        "linak_plot_data.csv",
    )


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
    """Open a PySide6 panel that previews and persists plot settings."""
    try:
        from PySide6.QtCore import (
            QAbstractListModel,
            QEasingCurve,
            QEvent,
            QModelIndex,
            QObject,
            QPropertyAnimation,
            QRect,
            QSize,
            QTimer,
            Qt,
            Signal,
        )
        from PySide6.QtGui import (
            QBrush,
            QColor,
            QDoubleValidator,
            QIcon,
            QIntValidator,
            QKeySequence,
            QPainter,
            QPalette,
            QPen,
            QPixmap,
            QPixmapCache,
            QShortcut,
        )
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
            QListView,
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
            QStyle,
            QStyledItemDelegate,
            QStyleOptionViewItem,
            QTabWidget,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

        try:
            from PySide6.QtSvg import QSvgRenderer
        except Exception:  # pragma: no cover - optional Qt module
            QSvgRenderer = None
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
                NavigationToolbar2QT,
            )
        except Exception:  # pragma: no cover - matplotlib Qt backend availability
            FigureCanvas = None
            NavigationToolbar2QT = None
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PySide6 is unavailable; cannot open GUI plot controls. "
            "Install PySide6 or use CLI plot flags."
        ) from exc

    if NavigationToolbar2QT is not None:

        class _LiNaKNavigationToolbar(NavigationToolbar2QT):  # type: ignore[misc, valid-type]
            def __init__(
                self,
                canvas: Any,
                parent: QWidget | None,
                *,
                on_linak_save_figure: Callable[[], None],
            ) -> None:
                super().__init__(canvas, parent)
                self._on_linak_save_figure = on_linak_save_figure

            def save_figure(self, *args: Any, **kwargs: Any) -> None:
                self._on_linak_save_figure()

    else:  # pragma: no cover - matplotlib Qt backend availability
        _LiNaKNavigationToolbar = None

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
            self._line_color = "#93a4b8"

        def set_line_color(self, color: str) -> None:
            self._line_color = color
            self.update()

        def paintEvent(self, event: Any) -> None:  # pragma: no cover - UI paint
            super().paintEvent(event)
            painter = QPainter(self)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen_color = QColor(self._line_color)
                if not pen_color.isValid():
                    pen_color = QColor("#93a4b8")
                pen = QPen(pen_color)
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

            self.kind_badge = QLabel(self)
            self.kind_badge.setObjectName("seriesRowKindBadge")
            layout.addWidget(self.kind_badge)

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

            for target in (self, self.text_label, self.color_swatch):
                target.installEventFilter(self)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # pragma: no cover - UI flow
            if watched in {
                self,
                self.text_label,
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
            layer_role: str,
            can_move_up: bool,
            can_move_down: bool,
            tooltip_text: str,
            theme: dict[str, str],
        ) -> None:
            self.checkbox.blockSignals(True)
            try:
                self.checkbox.setChecked(checked)
            finally:
                self.checkbox.blockSignals(False)
            self.checkbox.setEnabled(kind == "base")
            self.checkbox.setVisible(True)
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
                        "background-color: "
                        f"{swatch_color.name()}; "
                        f"border: 1px solid {theme['series_row_swatch_border']}; "
                        "border-radius: 6px;"
                    )
                else:
                    self.color_swatch.setStyleSheet(
                        "background-color: "
                        f"{theme['series_row_swatch_bg']}; "
                        f"border: 1px solid {theme['series_row_swatch_border']}; "
                        "border-radius: 6px;"
                    )

            text_color = theme["series_row_selected_text"] if selected else theme["series_row_text"]
            if not enabled:
                text_color = theme["series_row_disabled_text"]
            normalized_role = layer_role.strip().lower()
            badge_label = {
                "group": "Group",
                "copy": "Copy",
                "original": "Original",
            }.get(normalized_role, layer_role.strip().title() or "Layer")
            badge_prefix = (
                "series_badge_group"
                if normalized_role == "group"
                else "series_badge_copy"
                if normalized_role == "copy"
                else "series_badge_original"
            )
            badge_bg = theme[f"{badge_prefix}_bg"]
            badge_border = theme[f"{badge_prefix}_border"]
            badge_text = theme[f"{badge_prefix}_text"]
            self.kind_badge.setText(badge_label)
            self.kind_badge.setStyleSheet(
                "padding: 2px 8px;"
                "border-radius: 999px;"
                f"background-color: {badge_bg};"
                f"border: 1px solid {badge_border};"
                f"color: {badge_text};"
                "font-weight: 700;"
                "font-size: 10px;"
            )

            if selected:
                row_bg = theme["series_row_selected_bg"]
                left_border = f"5px solid {theme['series_row_selected_border']}"
                edge_border = f"1px solid {theme['series_row_selected_border']}"
                tl_radius = "4px"
                bl_radius = "4px"
                tr_radius = "8px"
                br_radius = "8px"
            else:
                row_bg = theme["series_row_bg"]
                left_border = "4px solid transparent"
                edge_border = "1px solid transparent"
                tl_radius = "4px"
                bl_radius = "4px"
                tr_radius = "8px"
                br_radius = "8px"

            self.setStyleSheet(
                "QWidget#seriesRowWidget {"
                f"background-color: {row_bg};"
                f"border-left: {left_border};"
                f"border-top: {edge_border};"
                f"border-right: {edge_border};"
                f"border-bottom: {edge_border};"
                f"border-top-left-radius: {tl_radius};"
                f"border-bottom-left-radius: {bl_radius};"
                f"border-top-right-radius: {tr_radius};"
                f"border-bottom-right-radius: {br_radius};"
                "}"
                "QToolButton#seriesRowButton {"
                "padding: 0px;"
                "margin: 0px;"
                "border: none;"
                "background: transparent;"
                f"color: {text_color};"
                "}"
                "QToolButton#seriesRowButton:hover {"
                "border-radius: 6px;"
                f"background-color: {theme['series_row_button_hover']};"
                "}"
                "QToolButton#seriesRowButton:disabled {"
                f"color: {theme['series_row_disabled_text']};"
                "}"
            )

            label_font_style = "italic" if not enabled else "normal"
            if kind == "fit":
                label_font_weight = "700"
            elif kind in {"error", "cumulative"}:
                label_font_weight = "700" if selected else "600"
            else:
                label_font_weight = "600" if selected else "400"
            self.text_label.setStyleSheet(
                "border: none;"
                f"color: {text_color};"
                f"font-style: {label_font_style};"
                f"font-weight: {label_font_weight};"
            )

            control_enabled = kind == "base"
            self.move_up_button.setEnabled(control_enabled and can_move_up)
            self.move_down_button.setEnabled(control_enabled and can_move_down)
            self.move_up_button.setVisible(control_enabled)
            self.move_down_button.setVisible(control_enabled)

    _SERIES_ROW_STATE_ROLE = int(Qt.ItemDataRole.UserRole) + 10

    class _SeriesListItem:
        def __init__(self) -> None:
            self._text = ""
            self._tooltip = ""
            self._data: dict[int, Any] = {}

        def setText(self, text: str) -> None:
            self._text = str(text)

        def text(self) -> str:
            return self._text

        def setToolTip(self, text: str) -> None:
            self._tooltip = str(text)

        def toolTip(self) -> str:
            return self._tooltip

        def setData(self, role: Any, value: Any) -> None:
            self._data[int(role)] = value

        def data(self, role: Any) -> Any:
            return self._data.get(int(role))

    class _SeriesListModel(QAbstractListModel):
        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._items: list[_SeriesListItem] = []

        def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
            if parent.isValid():
                return 0
            return len(self._items)

        def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)) -> Any:
            if not index.isValid():
                return None
            row = int(index.row())
            if row < 0 or row >= len(self._items):
                return None
            item = self._items[row]
            if role == int(Qt.ItemDataRole.DisplayRole):
                state = item.data(_SERIES_ROW_STATE_ROLE)
                if isinstance(state, dict):
                    return str(state.get("text") or "")
                return item.text()
            if role == int(Qt.ItemDataRole.ToolTipRole):
                return item.toolTip()
            return item.data(role)

        def flags(self, index: QModelIndex) -> Qt.ItemFlag:
            if not index.isValid():
                return Qt.ItemFlag.NoItemFlags
            flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            row = int(index.row())
            item = self._items[row] if 0 <= row < len(self._items) else None
            kind = "" if item is None else str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
            if kind == "base":
                flags |= Qt.ItemFlag.ItemIsDragEnabled
            return flags

        def add_item(self, item: _SeriesListItem) -> None:
            row = len(self._items)
            self.beginInsertRows(QModelIndex(), row, row)
            self._items.append(item)
            self.endInsertRows()

        def clear_items(self) -> None:
            if not self._items:
                return
            self.beginResetModel()
            self._items = []
            self.endResetModel()

        def item(self, row: int) -> _SeriesListItem | None:
            if 0 <= row < len(self._items):
                return self._items[row]
            return None

        def index_of_item(self, item: _SeriesListItem) -> int:
            try:
                return self._items.index(item)
            except ValueError:
                return -1

        def notify_row(self, row: int) -> None:
            if row < 0 or row >= len(self._items):
                return
            model_index = self.index(row, 0)
            self.dataChanged.emit(model_index, model_index, [])

    class _SeriesListDelegate(QStyledItemDelegate):
        def __init__(self, owner: Any, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._owner = owner

        def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
            return QSize(max(260, int(option.rect.width())), 38)

        def _row_state(self, index: QModelIndex) -> dict[str, Any]:
            state = index.data(_SERIES_ROW_STATE_ROLE)
            return dict(state) if isinstance(state, dict) else {}

        def _checkbox_rect(self, rect: QRect) -> QRect:
            return QRect(rect.left() + 10, rect.top() + 11, 16, 16)

        def _up_rect(self, rect: QRect) -> QRect:
            return QRect(rect.right() - 52, rect.top() + 8, 20, 22)

        def _down_rect(self, rect: QRect) -> QRect:
            return QRect(rect.right() - 28, rect.top() + 8, 20, 22)

        def paint(
            self,
            painter: QPainter,
            option: QStyleOptionViewItem,
            index: QModelIndex,
        ) -> None:  # pragma: no cover - UI paint
            state = self._row_state(index)
            theme = state.get("theme") if isinstance(state.get("theme"), dict) else {}
            selected = bool(option.state & QStyle.StateFlag.State_Selected)
            enabled = bool(state.get("enabled", True))
            kind = str(state.get("kind") or "base")
            layer_role = str(state.get("layer_role") or "Layer")
            rect = option.rect.adjusted(2, 2, -2, -2)

            painter.save()
            try:
                row_bg = (
                    theme.get("series_row_selected_bg", "#14515a")
                    if selected
                    else theme.get("series_row_bg", "#132033")
                )
                border = (
                    theme.get("series_row_selected_border", "#2fb7c9")
                    if selected
                    else "transparent"
                )
                painter.setPen(QPen(QColor(border)))
                painter.setBrush(QBrush(QColor(row_bg)))
                painter.drawRoundedRect(rect, 6, 6)

                text_color = (
                    theme.get("series_row_selected_text", "#ffffff")
                    if selected
                    else theme.get("series_row_text", "#e6edf7")
                )
                if not enabled:
                    text_color = theme.get("series_row_disabled_text", "#8290a3")

                checkbox_rect = self._checkbox_rect(option.rect)
                painter.setPen(QPen(QColor(theme.get("series_row_swatch_border", "#46627f"))))
                painter.setBrush(QBrush(QColor("#2fb7c9" if bool(state.get("checked", True)) else "#0f1a2a")))
                painter.drawRoundedRect(checkbox_rect, 3, 3)
                if bool(state.get("checked", True)):
                    painter.setPen(QPen(QColor("#ffffff"), 2))
                    painter.drawLine(
                        checkbox_rect.left() + 4,
                        checkbox_rect.center().y(),
                        checkbox_rect.center().x() - 1,
                        checkbox_rect.bottom() - 4,
                    )
                    painter.drawLine(
                        checkbox_rect.center().x() - 1,
                        checkbox_rect.bottom() - 4,
                        checkbox_rect.right() - 3,
                        checkbox_rect.top() + 4,
                    )

                x_cursor = checkbox_rect.right() + 12
                color_token = str(state.get("color_token") or "").strip()
                if color_token:
                    swatch_color = QColor(color_token)
                    if not swatch_color.isValid():
                        swatch_color = QColor(theme.get("series_row_swatch_bg", "#20304a"))
                    swatch_rect = QRect(x_cursor, option.rect.top() + 13, 12, 12)
                    painter.setPen(QPen(QColor(theme.get("series_row_swatch_border", "#46627f"))))
                    painter.setBrush(QBrush(swatch_color))
                    painter.drawRoundedRect(swatch_rect, 6, 6)
                    x_cursor = swatch_rect.right() + 10

                badge_text = {
                    "group": "Group",
                    "copy": "Copy",
                    "original": "Original",
                    "fit": "Fit",
                    "cumulative": "Cumulative",
                }.get(layer_role.strip().lower(), layer_role.strip().title() or "Layer")
                badge_rect = QRect(option.rect.right() - 150, option.rect.top() + 9, 78, 20)
                if kind != "base":
                    badge_rect.setWidth(96)
                    badge_rect.moveLeft(option.rect.right() - 168)
                badge_prefix = (
                    "series_badge_group"
                    if layer_role.strip().lower() == "group"
                    else "series_badge_copy"
                    if layer_role.strip().lower() == "copy"
                    else "series_badge_original"
                )
                painter.setPen(QPen(QColor(theme.get(f"{badge_prefix}_border", "#4e6380"))))
                painter.setBrush(QBrush(QColor(theme.get(f"{badge_prefix}_bg", "#1d2b42"))))
                painter.drawRoundedRect(badge_rect, 6, 6)
                painter.setPen(QPen(QColor(theme.get(f"{badge_prefix}_text", text_color))))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

                text_right = badge_rect.left() - 8
                if kind == "base":
                    text_right = min(text_right, self._up_rect(option.rect).left() - 8)
                text_rect = QRect(
                    x_cursor,
                    option.rect.top() + 4,
                    max(20, text_right - x_cursor),
                    option.rect.height() - 8,
                )
                font = painter.font()
                font.setBold(selected or kind in {"fit", "cumulative"})
                font.setItalic(not enabled)
                painter.setFont(font)
                painter.setPen(QPen(QColor(text_color)))
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine,
                    str(state.get("text") or ""),
                )

                if kind == "base":
                    painter.setFont(option.font)
                    arrow_color = QColor(
                        text_color
                        if bool(state.get("can_move_up", False))
                        else theme.get("series_row_disabled_text", "#8290a3")
                    )
                    painter.setPen(QPen(arrow_color))
                    painter.drawText(self._up_rect(option.rect), Qt.AlignmentFlag.AlignCenter, "^")
                    arrow_color = QColor(
                        text_color
                        if bool(state.get("can_move_down", False))
                        else theme.get("series_row_disabled_text", "#8290a3")
                    )
                    painter.setPen(QPen(arrow_color))
                    painter.drawText(self._down_rect(option.rect), Qt.AlignmentFlag.AlignCenter, "v")
            finally:
                painter.restore()

        def editorEvent(
            self,
            event: QEvent,
            model: QAbstractListModel,
            option: QStyleOptionViewItem,
            index: QModelIndex,
        ) -> bool:  # pragma: no cover - UI flow
            if event.type() != QEvent.Type.MouseButtonRelease:
                return super().editorEvent(event, model, option, index)
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            row = int(index.row())
            state = self._row_state(index)
            if self._checkbox_rect(option.rect).contains(pos):
                self._owner._handle_series_row_widget_toggle(
                    row,
                    not bool(state.get("checked", True)),
                )
                return True
            kind = str(state.get("kind") or "base")
            series_id = str(state.get("series_id") or "")
            if kind == "base" and series_id:
                if self._up_rect(option.rect).contains(pos) and bool(state.get("can_move_up")):
                    self._owner._move_series_by_delta(series_id, -1)
                    return True
                if self._down_rect(option.rect).contains(pos) and bool(state.get("can_move_down")):
                    self._owner._move_series_by_delta(series_id, 1)
                    return True
            return super().editorEvent(event, model, option, index)

    class _SeriesListView(QListView):
        currentRowChanged = Signal(int)

        def __init__(self, owner: Any, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._series_model = _SeriesListModel(self)
            self.setModel(self._series_model)
            self.setItemDelegate(_SeriesListDelegate(owner, self))
            self.selectionModel().currentChanged.connect(self._emit_current_row_changed)

        def _emit_current_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
            self.currentRowChanged.emit(int(current.row()) if current.isValid() else -1)

        def count(self) -> int:
            return self._series_model.rowCount()

        def item(self, row: int) -> _SeriesListItem | None:
            return self._series_model.item(row)

        def addItem(self, item: _SeriesListItem) -> None:
            self._series_model.add_item(item)

        def clear(self) -> None:
            self._series_model.clear_items()

        def itemWidget(self, _item: _SeriesListItem) -> None:
            return None

        def setItemWidget(self, _item: _SeriesListItem, _widget: QWidget) -> None:
            return None

        def removeItemWidget(self, _item: _SeriesListItem) -> None:
            return None

        def currentRow(self) -> int:
            current = self.currentIndex()
            return int(current.row()) if current.isValid() else -1

        def currentItem(self) -> _SeriesListItem | None:
            return self.item(self.currentRow())

        def setCurrentRow(self, row: int) -> None:
            if row < 0 or row >= self.count():
                self.clearSelection()
                return
            model_index = self._series_model.index(row, 0)
            self.setCurrentIndex(model_index)

        def visualItemRect(self, item: _SeriesListItem) -> QRect:
            row = self._series_model.index_of_item(item)
            if row < 0:
                return QRect()
            return self.visualRect(self._series_model.index(row, 0))

        def notifyRowChanged(self, row: int) -> None:
            self._series_model.notify_row(row)

    class _AnnotationRowWidget(QWidget):
        def __init__(
            self,
            *,
            on_select: Callable[[], None],
            on_move_up: Callable[[], None],
            on_move_down: Callable[[], None],
        ) -> None:
            super().__init__()
            self._on_select = on_select
            self._on_move_up = on_move_up
            self._on_move_down = on_move_down
            self.setObjectName("annotationRowWidget")

            layout = QHBoxLayout(self)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(8)

            self.text_label = QLabel(self)
            self.text_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            layout.addWidget(self.text_label, stretch=1)

            self.move_up_button = QToolButton(self)
            self.move_up_button.setObjectName("seriesRowButton")
            self.move_up_button.setAutoRaise(True)
            self.move_up_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.move_up_button.setFixedSize(22, 22)
            self.move_up_button.setArrowType(Qt.ArrowType.UpArrow)
            self.move_up_button.setText("")
            self.move_up_button.clicked.connect(self._handle_move_up_clicked)
            layout.addWidget(self.move_up_button)

            self.move_down_button = QToolButton(self)
            self.move_down_button.setObjectName("seriesRowButton")
            self.move_down_button.setAutoRaise(True)
            self.move_down_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.move_down_button.setFixedSize(22, 22)
            self.move_down_button.setArrowType(Qt.ArrowType.DownArrow)
            self.move_down_button.setText("")
            self.move_down_button.clicked.connect(self._handle_move_down_clicked)
            layout.addWidget(self.move_down_button)

            for target in (self, self.text_label):
                target.installEventFilter(self)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # pragma: no cover - UI flow
            if watched in {self, self.text_label} and event.type() in {
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
            enabled: bool,
            selected: bool,
            can_move_up: bool,
            can_move_down: bool,
            tooltip_text: str,
            theme: dict[str, str],
        ) -> None:
            self.text_label.setText(text)
            self.setToolTip(tooltip_text)
            self.text_label.setToolTip(tooltip_text)
            text_color = theme["series_row_selected_text"] if selected else theme["series_row_text"]
            if not enabled:
                text_color = theme["series_row_disabled_text"]
            if selected:
                row_bg = theme["series_row_selected_bg"]
                left_border = f"5px solid {theme['series_row_selected_border']}"
                edge_border = f"1px solid {theme['series_row_selected_border']}"
            else:
                row_bg = theme["series_row_bg"]
                left_border = "4px solid transparent"
                edge_border = "1px solid transparent"
            self.setStyleSheet(
                "QWidget#annotationRowWidget {"
                f"background-color: {row_bg};"
                f"border-left: {left_border};"
                f"border-top: {edge_border};"
                f"border-right: {edge_border};"
                f"border-bottom: {edge_border};"
                "border-top-left-radius: 4px;"
                "border-bottom-left-radius: 4px;"
                "border-top-right-radius: 8px;"
                "border-bottom-right-radius: 8px;"
                "}"
                "QToolButton#seriesRowButton {"
                "padding: 0px;"
                "margin: 0px;"
                "border: none;"
                "background: transparent;"
                f"color: {text_color};"
                "}"
                "QToolButton#seriesRowButton:hover {"
                "border-radius: 6px;"
                f"background-color: {theme['series_row_button_hover']};"
                "}"
                "QToolButton#seriesRowButton:disabled {"
                f"color: {theme['series_row_disabled_text']};"
                "}"
            )
            self.text_label.setStyleSheet(f"color: {text_color};")
            self.move_up_button.setEnabled(can_move_up)
            self.move_down_button.setEnabled(can_move_down)

    class _PreviewPane(QFrame):
        def __init__(
            self,
            *,
            title_text: str,
            object_name: str,
            on_refresh: Callable[[], None],
            on_fit: Callable[[], None],
            on_actual_size: Callable[[], None],
            on_save_figure_callback: Callable[[], None],
            on_save_data_callback: Callable[[], None],
            on_auto_update: Callable[[bool], None],
            on_detach: Callable[[], None] | None,
            on_dock: Callable[[], None] | None,
            register_tooltip: Callable[[QWidget, str], None],
            apply_tooltip: Callable[[QWidget], None],
            event_filter_owner: QWidget,
            auto_update_enabled: bool,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName(object_name)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)

            preview_title = QLabel(title_text)
            preview_title.setObjectName("pageTitle")
            layout.addWidget(preview_title)

            # The embedded Matplotlib toolbar owns visible preview actions.
            # Keep these hidden widgets so the existing preview-loading and
            # callback wiring can remain small and predictable.
            self.preview_button = QPushButton("Refresh Preview")
            self.preview_button.clicked.connect(on_refresh)
            register_tooltip(self.preview_button, "preview.refresh")
            apply_tooltip(self.preview_button)
            self.preview_button.setVisible(False)

            self.fit_button = QPushButton("Fit")
            self.fit_button.clicked.connect(on_fit)
            register_tooltip(self.fit_button, "preview.fit")
            apply_tooltip(self.fit_button)
            self.fit_button.setVisible(False)

            self.actual_size_button = QPushButton("Reset View")
            self.actual_size_button.clicked.connect(on_actual_size)
            register_tooltip(self.actual_size_button, "preview.actual_size")
            apply_tooltip(self.actual_size_button)
            self.actual_size_button.setVisible(False)

            self.save_figure_button = QPushButton("Export Figure")
            self.save_figure_button.setEnabled(auto_update_enabled)
            self.save_figure_button.clicked.connect(on_save_figure_callback)
            register_tooltip(self.save_figure_button, "export.figure")
            apply_tooltip(self.save_figure_button)
            self.save_figure_button.setVisible(False)

            self.save_data_button = QPushButton("Export Data")
            self.save_data_button.setEnabled(auto_update_enabled)
            self.save_data_button.clicked.connect(on_save_data_callback)
            register_tooltip(self.save_data_button, "export.data")
            apply_tooltip(self.save_data_button)
            self.save_data_button.setVisible(False)

            self.auto_preview_checkbox = QCheckBox("Auto update")
            self.auto_preview_checkbox.setChecked(True)
            self.auto_preview_checkbox.setEnabled(False)
            self.auto_preview_checkbox.toggled.connect(on_auto_update)
            register_tooltip(self.auto_preview_checkbox, "preview.auto_update")
            apply_tooltip(self.auto_preview_checkbox)
            self.auto_preview_checkbox.setVisible(False)

            self.detach_button: QPushButton | None = None
            if on_detach is not None:
                self.detach_button = QPushButton("Detach Preview")
                self.detach_button.clicked.connect(on_detach)
                self.detach_button.setVisible(False)

            self.dock_button: QPushButton | None = None
            if on_dock is not None:
                self.dock_button = QPushButton("Dock Back")
                self.dock_button.clicked.connect(on_dock)
                self.dock_button.setVisible(False)

            self.preview_status = QLabel("Preview ready.")
            self.preview_status.setVisible(False)

            self.preview_frame = QFrame(self)
            self.preview_frame.setObjectName("previewFrame")
            self.preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
            self.preview_frame.installEventFilter(event_filter_owner)
            preview_frame_layout = QVBoxLayout(self.preview_frame)
            preview_frame_layout.setContentsMargins(6, 6, 6, 6)

            self.preview_scroll = QScrollArea(self.preview_frame)
            self.preview_scroll.setWidgetResizable(False)
            self.preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_scroll.viewport().installEventFilter(event_filter_owner)

            self.preview_label = QLabel("Loading...")
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setWordWrap(True)
            self.preview_label.setMinimumSize(3, 3)
            self.preview_label.installEventFilter(event_filter_owner)
            self.preview_scroll.setWidget(self.preview_label)
            preview_frame_layout.addWidget(self.preview_scroll, stretch=1)

            self.preview_canvas_container = QWidget(self.preview_frame)
            self.preview_canvas_layout = QVBoxLayout(self.preview_canvas_container)
            self.preview_canvas_layout.setContentsMargins(0, 0, 0, 0)
            self.preview_canvas_layout.setSpacing(4)
            self.preview_canvas_container.setVisible(False)
            preview_frame_layout.addWidget(self.preview_canvas_container, stretch=1)
            layout.addWidget(self.preview_frame, stretch=1)

    class _PreviewWorkerBridge(QObject):
        finished = Signal(int, object)
        failed = Signal(int, object)

    class _CollapsibleSection(QFrame):
        def __init__(
            self,
            *,
            title: str,
            section_id: str,
            state_store: dict[str, bool],
            default_expanded: bool = False,
            subsection: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._section_id = str(section_id).strip()
            self._state_store = state_store
            self._subsection = subsection
            self._body_widget: QWidget | None = None
            self._expanded = bool(state_store.get(self._section_id, default_expanded))
            self._collapse_after_animation = False
            self.setObjectName("collapsibleSubsection" if subsection else "collapsibleSection")

            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            self.header_frame = QFrame(self)
            self.header_frame.setObjectName(
                "collapsibleSubsectionHeader" if subsection else "collapsibleSectionHeader"
            )
            header_layout = QHBoxLayout(self.header_frame)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(8)

            self.toggle_button = QToolButton(self.header_frame)
            self.toggle_button.setObjectName("collapsibleToggle")
            self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.toggle_button.setCheckable(True)
            self.toggle_button.setChecked(self._expanded)
            self.toggle_button.setText(title)
            self.toggle_button.clicked.connect(self._handle_toggle_clicked)
            self.toggle_button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            header_layout.addWidget(
                self.toggle_button,
                stretch=1,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )

            root_layout.addWidget(self.header_frame)

            self.body_frame = QFrame(self)
            self.body_frame.setObjectName(
                "collapsibleSubsectionBody" if subsection else "collapsibleSectionBody"
            )
            self.body_layout = QVBoxLayout(self.body_frame)
            self.body_layout.setContentsMargins(0, 0, 0, 0)
            self.body_layout.setSpacing(0)
            root_layout.addWidget(self.body_frame)

            self._animation = QPropertyAnimation(self.body_frame, b"maximumHeight", self)
            self._animation.setDuration(160)
            self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._animation.finished.connect(self._handle_animation_finished)
            self._apply_expanded_state(self._expanded, animate=False, persist=False)

        def set_body_widget(self, widget: QWidget) -> None:
            self._body_widget = widget
            self.body_layout.addWidget(widget)
            self._apply_expanded_state(self._expanded, animate=False, persist=False)

        def is_expanded(self) -> bool:
            return self._expanded

        def set_expanded(self, expanded: bool, *, animate: bool = True) -> None:
            self._apply_expanded_state(bool(expanded), animate=animate, persist=True)

        def _target_body_height(self) -> int:
            if self._body_widget is None:
                return 0
            hint = self._body_widget.sizeHint()
            height = int(hint.height()) if hint is not None else 0
            if height <= 0:
                layout_hint = self.body_layout.sizeHint()
                height = int(layout_hint.height()) if layout_hint is not None else 0
            return max(0, height)

        def _handle_toggle_clicked(self, checked: bool) -> None:
            self._apply_expanded_state(bool(checked), animate=True, persist=True)

        def _apply_expanded_state(
            self,
            expanded: bool,
            *,
            animate: bool,
            persist: bool,
        ) -> None:
            self._expanded = bool(expanded)
            if persist and self._section_id:
                self._state_store[self._section_id] = self._expanded
            self.toggle_button.blockSignals(True)
            try:
                self.toggle_button.setChecked(self._expanded)
                self.toggle_button.setArrowType(
                    Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
                )
            finally:
                self.toggle_button.blockSignals(False)

            target_height = self._target_body_height()
            can_animate = (
                animate
                and self.isVisible()
                and target_height > 0
                and self._body_widget is not None
                and not self._body_widget.isWindow()
            )
            if not can_animate:
                self._animation.stop()
                self._collapse_after_animation = False
                self.body_frame.setVisible(self._expanded)
                self.body_frame.setMaximumHeight(16777215 if self._expanded else 0)
                return

            start_height = max(0, int(self.body_frame.height()))
            end_height = target_height if self._expanded else 0
            if start_height == end_height:
                self.body_frame.setVisible(self._expanded)
                self.body_frame.setMaximumHeight(16777215 if self._expanded else 0)
                return

            if self._expanded:
                self.body_frame.setVisible(True)
            self._collapse_after_animation = not self._expanded
            self._animation.stop()
            self.body_frame.setMaximumHeight(start_height)
            self._animation.setStartValue(start_height)
            self._animation.setEndValue(end_height)
            self._animation.start()

        def _handle_animation_finished(self) -> None:
            if self._collapse_after_animation:
                self.body_frame.setVisible(False)
                self.body_frame.setMaximumHeight(0)
                self._collapse_after_animation = False
                return
            self.body_frame.setVisible(True)
            self.body_frame.setMaximumHeight(16777215)

    class _StaticSection(QFrame):
        def __init__(
            self,
            *,
            title: str,
            subsection: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("collapsibleSubsection" if subsection else "collapsibleSection")

            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            self.header_frame = QFrame(self)
            self.header_frame.setObjectName(
                "collapsibleSubsectionHeader" if subsection else "collapsibleSectionHeader"
            )
            header_layout = QHBoxLayout(self.header_frame)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(8)

            self.header_label = QLabel(title, self.header_frame)
            self.header_label.setObjectName(
                "staticSubsectionHeaderLabel" if subsection else "staticSectionHeaderLabel"
            )
            self.header_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.header_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            header_layout.addWidget(self.header_label, stretch=1)

            root_layout.addWidget(self.header_frame)

            self.body_frame = QFrame(self)
            self.body_frame.setObjectName(
                "collapsibleSubsectionBody" if subsection else "collapsibleSectionBody"
            )
            self.body_layout = QVBoxLayout(self.body_frame)
            self.body_layout.setContentsMargins(0, 0, 0, 0)
            self.body_layout.setSpacing(0)
            root_layout.addWidget(self.body_frame)

        def set_body_widget(self, widget: QWidget) -> None:
            self.body_layout.addWidget(widget)

    class _DetachedPreviewWindow(QMainWindow):
        def __init__(
            self,
            *,
            on_dock_requested: Callable[[], None],
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._on_dock_requested = on_dock_requested
            self._allow_close = False
            self.setWindowTitle("LiNaK Figure Preview")

        def close_from_dock(self) -> None:
            self._allow_close = True
            self.close()

        def closeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            if self._allow_close:
                super().closeEvent(event)
                return
            event.ignore()
            QTimer.singleShot(0, self._on_dock_requested)

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
            self._undo_stack: list[dict[str, Any]] = []
            self._redo_stack: list[dict[str, Any]] = []
            self._undo_syncing = False
            self._undo_current_settings: dict[str, Any] | None = None
            self._undo_current_signature: str | None = None
            self._undo_text_edit_widget_id: int | None = None
            self._undo_text_edit_signature: str | None = None
            self._undo_text_edit_settings: dict[str, Any] | None = None
            self._profile_filter_options = _coerce_profile_filter_options(
                initial_settings.get("_profile_filter_options")
            )
            self._position_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("position_plot_contract")
                )
                if self._analysis_name == "position"
                else None
            )
            if self._analysis_name == "position" and self._position_data_contract is None:
                self._position_data_contract = _fallback_position_plot_data_contract()
            self._density_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("density_plot_contract")
                )
                if self._analysis_name == "density"
                else None
            )
            self._density_heatmap_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("density_heatmap_plot_contract")
                )
                if self._analysis_name == "density"
                else None
            )
            if self._analysis_name == "density" and self._density_data_contract is None:
                self._density_data_contract = _fallback_density_plot_data_contract()
            if self._analysis_name == "density" and self._density_heatmap_data_contract is None:
                self._density_heatmap_data_contract = _fallback_density_heatmap_plot_data_contract()
            self._coordination_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("coordination_plot_contract")
                )
                if self._analysis_name == "coordination"
                else None
            )
            if (
                self._analysis_name == "coordination"
                and self._coordination_data_contract is None
            ):
                self._coordination_data_contract = _fallback_coordination_plot_data_contract()
            self._orientation_line_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("orientation_line_plot_contract")
                )
                if self._analysis_name == "orientation"
                else None
            )
            self._orientation_heatmap_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("orientation_heatmap_plot_contract")
                )
                if self._analysis_name == "orientation"
                else None
            )
            if (
                self._analysis_name == "orientation"
                and self._orientation_line_data_contract is None
            ):
                self._orientation_line_data_contract = (
                    _fallback_orientation_line_plot_data_contract()
                )
            if (
                self._analysis_name == "orientation"
                and self._orientation_heatmap_data_contract is None
            ):
                self._orientation_heatmap_data_contract = (
                    _fallback_orientation_heatmap_plot_data_contract()
                )
            self._potential_data_contract = (
                _plot_data_contract_from_payload(
                    self._profile_filter_options.get("potential_plot_contract")
                )
                if self._analysis_name == "potential"
                else None
            )
            if self._analysis_name == "potential" and self._potential_data_contract is None:
                self._potential_data_contract = _fallback_potential_plot_data_contract()
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
            self._x_ticks_rows: list[tuple[QFormLayout, QWidget]] = []
            self._y_ticks_rows: list[tuple[QFormLayout, QWidget]] = []
            self._marker_rows: list[tuple[QFormLayout, QWidget]] = []
            self._integration_rows: list[tuple[QFormLayout, QWidget]] = []
            self._integration_custom_color_row: tuple[QFormLayout, QWidget] | None = None
            self._colorbar_rows: list[tuple[QFormLayout, QWidget]] = []
            self._border_custom_rows: list[tuple[QFormLayout, QWidget]] = []
            self._x_bin_reducer_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_value_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_x_ref_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_x_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_y_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_render_mode_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_value_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_filter_min_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_filter_max_row: tuple[QFormLayout, QWidget] | None = None
            self._position_mapping_split_by_row: tuple[QFormLayout, QWidget] | None = None
            self._position_species_checkboxes: dict[str, QCheckBox] = {}
            self._density_mapping_1d_rows: list[tuple[QFormLayout, QWidget]] = []
            self._density_mapping_2d_rows: list[tuple[QFormLayout, QWidget]] = []
            self._density_filter_rows: dict[str, tuple[QFormLayout, QWidget]] = {}
            self._orientation_mapping_2d_rows: list[tuple[QFormLayout, QWidget]] = []
            self._orientation_filter_rows: dict[str, tuple[QFormLayout, QWidget]] = {}
            self._orientation_line_quantity_row: tuple[QFormLayout, QWidget] | None = None
            self._orientation_line_x_axis_row: tuple[QFormLayout, QWidget] | None = None
            self._density_species_checkbox_syncing = False
            self._density_previous_view_type_id: str | None = None
            self._density_1d_enabled_species_snapshot: set[str] | None = None
            self._coordination_line_x_quantity_row: tuple[QFormLayout, QWidget] | None = None
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
            self._series_show_raw_line_data: list[bool] = []
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
            self._series_fit_color_data: list[str] = []
            self._series_fit_alpha_data: list[str] = []
            self._series_fit_line_width_data: list[str] = []
            self._series_fit_line_style_data: list[str] = []
            self._series_cumulative_enabled_data: list[bool] = []
            self._series_cumulative_label_overrides_data: list[str] = []
            self._series_cumulative_show_in_legend_data: list[bool] = []
            self._series_cumulative_color_data: list[str] = []
            self._series_cumulative_alpha_data: list[str] = []
            self._series_cumulative_line_width_data: list[str] = []
            self._series_cumulative_line_style_data: list[str] = []
            self._series_integration_enabled_data: list[bool] = []
            self._series_integration_source_data: list[str] = []
            self._series_integration_x_min_data: list[str] = []
            self._series_integration_x_max_data: list[str] = []
            self._series_integration_baseline_data: list[str] = []
            self._series_integration_color_mode_data: list[str] = []
            self._series_integration_color_data: list[str] = []
            self._series_integration_alpha_data: list[str] = []
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
            self._synced_field_modes: dict[str, str] = {}
            self._density_active_view_type = PLOT_VIEW_1D_LINE
            self._density_view_states: dict[str, dict[str, Any]] = {}
            self._density_view_state_switching = False
            self._position_active_view_type = PLOT_VIEW_1D_LINE
            self._position_view_states: dict[str, dict[str, Any]] = {}
            self._position_view_state_switching = False
            self._orientation_active_view_type = PLOT_VIEW_1D_LINE
            self._orientation_view_states: dict[str, dict[str, Any]] = {}
            self._orientation_view_state_switching = False
            self._collapsible_section_state: dict[str, bool] = {}
            self._advanced_json_syncing = False
            self._suspend_preview_events = False
            self._preview_pixmap: QPixmap | None = None
            self._preview_figure: Any | None = None
            self._preview_canvas: Any | None = None
            self._preview_toolbar: Any | None = None
            self._preview_axis_callback_ids: list[tuple[Any, int]] = []
            self._canvas_axis_limit_syncing = False
            self._preview_zoom_factor = 1.0
            self._splitter: QSplitter | None = None
            self._preview_splitter_sizes: list[int] | None = None
            self._embedded_preview_pane: _PreviewPane | None = None
            self._detached_preview_window: _DetachedPreviewWindow | None = None
            self._detached_preview_pane: _PreviewPane | None = None
            self._active_preview_pane: _PreviewPane | None = None
            self._preview_frame: QFrame | None = None
            self._preview_scroll: QScrollArea | None = None
            self._preview_label: QLabel | None = None
            self._preview_canvas_container: QWidget | None = None
            self._preview_canvas_layout: QVBoxLayout | None = None
            self._preview_canvas_scroll: QScrollArea | None = None
            self._preview_status: QLabel | None = None
            self._preview_button: QPushButton | None = None
            self._undo_button: QPushButton | None = None
            self._redo_button: QPushButton | None = None
            self._header_detach_preview_button: QPushButton | None = None
            self._undo_shortcut: QShortcut | None = None
            self._redo_shortcut: QShortcut | None = None
            self._save_figure_button: QPushButton | None = None
            self._save_data_button: QPushButton | None = None
            self._data_export_summary_label: QLabel | None = None
            self._data_export_format: QComboBox | None = None
            self._data_export_delimiter: QComboBox | None = None
            self._data_export_include_metadata: QCheckBox | None = None
            self._data_export_enabled_only: QCheckBox | None = None
            self._data_export_button: QPushButton | None = None
            self._auto_preview_checkbox: QCheckBox | None = None
            self._detach_preview_button: QPushButton | None = None
            self._dock_preview_button: QPushButton | None = None
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
            self._series_show_in_legend_row: tuple[QFormLayout, QWidget] | None = None
            self._series_show_raw_line_row: tuple[QFormLayout, QWidget] | None = None
            self._series_uncertainty_group: QGroupBox | None = None
            self._series_derived_group: QGroupBox | None = None
            self._series_metadata_group: QGroupBox | None = None
            self._selected_layer_card: QFrame | None = None
            self._selected_layer_title: QLabel | None = None
            self._selected_layer_badge: QLabel | None = None
            self._selected_layer_state: QLabel | None = None
            self._selected_layer_source: QLabel | None = None
            self._selected_layer_swatch: QFrame | None = None
            self._series_delete_button: QPushButton | None = None
            self._series_fit_group: QGroupBox | None = None
            self._series_cumulative_group: QGroupBox | None = None
            self._series_integration_group: QGroupBox | None = None
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
            self._figure_tabs: QTabWidget | None = None
            self._figure_legend_section: QWidget | None = None
            self._figure_lines_section: QGroupBox | None = None
            self._figure_lines_group: QGroupBox | None = None
            self._figure_heatmap_section: QGroupBox | None = None
            self._figure_heatmap_group: QGroupBox | None = None
            self._heatmap_value_group: QWidget | None = None
            self._heatmap_bulk_rows: list[tuple[QFormLayout, QWidget]] = []
            self._heatmap_bulk_manual_rows: list[tuple[QFormLayout, QWidget]] = []
            self._heatmap_trajectory_group: QWidget | None = None
            self._figure_colorbar_group: QGroupBox | None = None
            self._position_projection_stroke_row: tuple[QFormLayout, QWidget] | None = None
            self._x_axis_transform_rows: list[tuple[QFormLayout, QWidget]] = []
            self._advanced_legend_kwargs_rows: list[tuple[QFormLayout, QWidget]] = []
            self._advanced_line_kwargs_rows: list[tuple[QFormLayout, QWidget]] = []
            self._y_bin_width_row: tuple[QFormLayout, QWidget] | None = None
            self._y_bin_reducer_row: tuple[QFormLayout, QWidget] | None = None
            self._min_bin_points_row: tuple[QFormLayout, QWidget] | None = None
            self._binning_helper_label: QLabel | None = None
            self._density_target_filter: QComboBox | None = None
            self._tooltip_disabled_reasons: dict[int, str | None] = {}
            self._gui_artwork_path = _default_gui_artwork_path()
            self._figure_save_filters, self._figure_default_name = _figure_filetype_filters()
            self._data_save_filters, self._data_default_name = _data_filetype_filters()
            self._preview_image_path = (
                Path(tempfile.gettempdir()) / f"linak_preview_{uuid4().hex}.png"
            )
            self._preview_temp_paths: set[Path] = {self._preview_image_path}
            self._preview_generation = 0
            self._active_preview_generation: int | None = None
            self._active_preview_image_path: Path | None = None
            self._active_preview_interactive = False
            self._pending_preview_request: tuple[dict[str, Any], bool] | None = None
            self._preview_worker_queue: queue.Queue[tuple[int, dict[str, Any], Path | None] | None] = queue.Queue()
            self._preview_worker_stop = threading.Event()
            self._preview_worker_thread: threading.Thread | None = threading.Thread(
                target=self._preview_worker_loop,
                name="LiNaKPreviewWorker",
                daemon=True,
            )
            self._preview_worker_bridge = _PreviewWorkerBridge(self)
            self._preview_worker_bridge.finished.connect(self._handle_preview_worker_finished)
            self._preview_worker_bridge.failed.connect(self._handle_preview_worker_failed)
            self._preview_worker_thread.start()
            self._closing = False
            self._preview_timer = QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.timeout.connect(self._handle_debounced_preview)
            self._preview_loading = False
            self._preview_error: str | None = None
            self._theme_mode = "system"
            self._theme_switch: QCheckBox | None = None
            self._status_label = QLabel("Ready.")
            self._build_ui()
            self._install_undo_event_filters()
            self._bind_undo_change_signals()
            self._suspend_preview_events = True
            _previous_series_syncing = self._series_syncing
            _previous_annotation_syncing = self._annotation_syncing
            self._series_syncing = True
            self._annotation_syncing = True
            try:
                self._populate(initial_settings)
            finally:
                self._series_syncing = _previous_series_syncing
                self._annotation_syncing = _previous_annotation_syncing
                self._suspend_preview_events = False
            self._refresh_widget_states()
            self._bind_live_preview_signals()
            try:
                self._saved_signature = self._signature(self._collect_settings())
            except Exception:
                self._saved_signature = None
            self._reset_undo_history()
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

        def _bounded_float_line(
            self,
            placeholder: str = "",
            *,
            bottom: float | None = None,
            top: float | None = None,
            decimals: int = 6,
        ) -> QLineEdit:
            widget = self._line(placeholder)
            validator = QDoubleValidator(widget)
            validator.setDecimals(decimals)
            if bottom is not None:
                validator.setBottom(float(bottom))
            if top is not None:
                validator.setTop(float(top))
            widget.setValidator(validator)
            return widget

        def _positive_int_line(self, placeholder: str = "") -> QLineEdit:
            widget = self._line(placeholder)
            widget.setValidator(QIntValidator(1, 1_000_000, widget))
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

        def _prepare_collapsible_body(self, widget: QWidget) -> QWidget:
            if isinstance(widget, QGroupBox):
                widget.setProperty("collapsibleBody", True)
                widget.setTitle("")
            return widget

        def _make_collapsible_section(
            self,
            *,
            title: str,
            section_id: str,
            body_widget: QWidget,
            default_expanded: bool = False,
            subsection: bool = False,
        ) -> _CollapsibleSection:
            section = _CollapsibleSection(
                title=title,
                section_id=section_id,
                state_store=self._collapsible_section_state,
                default_expanded=default_expanded,
                subsection=subsection,
                parent=self,
            )
            section.set_body_widget(self._prepare_collapsible_body(body_widget))
            return section

        def _make_static_section(
            self,
            *,
            title: str,
            body_widget: QWidget,
            subsection: bool = False,
        ) -> _StaticSection:
            section = _StaticSection(
                title=title,
                subsection=subsection,
                parent=self,
            )
            section.set_body_widget(self._prepare_collapsible_body(body_widget))
            return section

        def _synced_field_mode(self, key: str) -> str:
            token = str(self._synced_field_modes.get(key, "auto")).strip().lower()
            allowed_modes = (
                {"auto", "manual", "off"} if key in _TEXT_SYNC_FIELD_KEYS else {"auto", "manual"}
            )
            return token if token in allowed_modes else "auto"

        def _set_synced_field_mode(self, key: str, mode: str) -> None:
            normalized = str(mode).strip().lower()
            allowed_modes = (
                {"auto", "manual", "off"} if key in _TEXT_SYNC_FIELD_KEYS else {"auto", "manual"}
            )
            if normalized not in allowed_modes:
                normalized = "auto"
            self._synced_field_modes[key] = normalized
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
            lock.currentTextChanged.connect(
                lambda value, sync_key=key: self._handle_synced_field_mode_changed(sync_key, value)
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
                if self._synced_field_mode("x_lim") == "auto":
                    self.x_min.setText(_extract_limit(settings, key="x_lim", index=0))
                    self.x_max.setText(_extract_limit(settings, key="x_lim", index=1))
                if self._synced_field_mode("y_lim") == "auto":
                    self.y_min.setText(_extract_limit(settings, key="y_lim", index=0))
                    self.y_max.setText(_extract_limit(settings, key="y_lim", index=1))
                if self._synced_field_mode("x_ticks") == "auto":
                    self.x_ticks.setText(_format_float_list(settings.get("x_ticks")))
                if self._synced_field_mode("y_ticks") == "auto":
                    self.y_ticks.setText(_format_float_list(settings.get("y_ticks")))
                if self._synced_field_mode("x_label_pad") == "auto":
                    x_label_pad = settings.get("x_label_pad")
                    self.x_label_pad.setText("" if x_label_pad is None else str(x_label_pad))
                if self._synced_field_mode("y_label_pad") == "auto":
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
            self._update_integration_summary()
            self._update_heatmap_value_summary()

        def _set_axis_limit_fields_from_canvas(self, ax: Any) -> None:
            if self._canvas_axis_limit_syncing:
                return
            self._canvas_axis_limit_syncing = True
            self._suspend_preview_events = True
            try:
                x_lim = [float(value) for value in ax.get_xlim()]
                y_lim = [float(value) for value in ax.get_ylim()]
                self._set_synced_field_mode("x_lim", "manual")
                self._set_synced_field_mode("y_lim", "manual")
                self.x_min.setText(_format_float_value(x_lim[0]))
                self.x_max.setText(_format_float_value(x_lim[1]))
                self.y_min.setText(_format_float_value(y_lim[0]))
                self.y_max.setText(_format_float_value(y_lim[1]))
                self._last_preview_state["x_lim"] = x_lim
                self._last_preview_state["y_lim"] = y_lim
                modes = dict(self._last_preview_state.get("_gui_sync_modes") or {})
                modes["x_lim"] = "manual"
                modes["y_lim"] = "manual"
                self._last_preview_state["_gui_sync_modes"] = modes
            finally:
                self._suspend_preview_events = False
                self._canvas_axis_limit_syncing = False
            if self._preview_status is not None:
                self._preview_status.setText("Axis limits updated from preview.")
            self._refresh_shell_state()

        def _handle_canvas_axis_limits_changed(self, ax: Any) -> None:
            self._set_axis_limit_fields_from_canvas(ax)

        def _apply_axis_limit_fields_to_canvas(self, key: str) -> bool:
            if self._preview_canvas is None or self._preview_figure is None:
                return False
            axes = list(getattr(self._preview_figure, "axes", []) or [])
            if not axes:
                return False
            ax = axes[0]

            def _field_value(widget: QLineEdit) -> float | None:
                text = widget.text().strip()
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return None

            x_lim = [_field_value(self.x_min), _field_value(self.x_max)]
            y_lim = [_field_value(self.y_min), _field_value(self.y_max)]
            self._canvas_axis_limit_syncing = True
            try:
                if key == "x_lim":
                    ax.set_xlim(left=x_lim[0], right=x_lim[1])
                    self._last_preview_state["x_lim"] = list(ax.get_xlim())
                elif key == "y_lim":
                    ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
                    self._last_preview_state["y_lim"] = list(ax.get_ylim())
                else:
                    return False
                modes = dict(self._last_preview_state.get("_gui_sync_modes") or {})
                modes[key] = "manual"
                self._last_preview_state["_gui_sync_modes"] = modes
                self._preview_canvas.draw_idle()
            finally:
                self._canvas_axis_limit_syncing = False
            if self._preview_status is not None:
                self._preview_status.setText("Axis limits updated.")
            self._refresh_shell_state()
            return True

        def _apply_text_fields_to_canvas(self) -> bool:
            if self._preview_canvas is None or self._preview_figure is None:
                return False
            axes = list(getattr(self._preview_figure, "axes", []) or [])
            if not axes:
                return False
            ax = axes[0]

            def _optional_float_text(widget: QLineEdit) -> float | None:
                text = widget.text().strip()
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return None

            def _optional_int_text(widget: QLineEdit) -> int | None:
                value = _optional_float_text(widget)
                if value is None or value <= 0:
                    return None
                return int(round(value))

            title_kwargs: dict[str, Any] = {}
            title_size = _optional_int_text(self.title_font)
            if title_size is not None:
                title_kwargs["fontsize"] = title_size
            title_pad = _optional_float_text(self.title_pad)
            if title_pad is not None:
                title_kwargs["pad"] = title_pad
            label_kwargs: dict[str, Any] = {}
            x_label_size = _optional_int_text(self.x_label_font)
            y_label_size = _optional_int_text(self.y_label_font)
            x_label_pad = _optional_float_text(self.x_label_pad)
            y_label_pad = _optional_float_text(self.y_label_pad)

            title = "" if self._synced_field_mode("title") == "off" else self.title_text.text()
            x_label = "" if self._synced_field_mode("x_label") == "off" else self.x_label.text()
            y_label = "" if self._synced_field_mode("y_label") == "off" else self.y_label.text()
            ax.set_title(title, **title_kwargs)
            x_kwargs = dict(label_kwargs)
            y_kwargs = dict(label_kwargs)
            if x_label_size is not None:
                x_kwargs["fontsize"] = x_label_size
            if y_label_size is not None:
                y_kwargs["fontsize"] = y_label_size
            if x_label_pad is not None:
                x_kwargs["labelpad"] = x_label_pad
            if y_label_pad is not None:
                y_kwargs["labelpad"] = y_label_pad
            ax.set_xlabel(x_label, **x_kwargs)
            ax.set_ylabel(y_label, **y_kwargs)
            try:
                self._preview_figure.tight_layout()
            except Exception:
                pass
            self._last_preview_state["title"] = title
            self._last_preview_state["x_label"] = x_label
            self._last_preview_state["y_label"] = y_label
            if x_label_pad is not None:
                self._last_preview_state["x_label_pad"] = x_label_pad
            if y_label_pad is not None:
                self._last_preview_state["y_label_pad"] = y_label_pad
            self._preview_canvas.draw_idle()
            if self._preview_status is not None:
                self._preview_status.setText("Preview text updated.")
            self._refresh_shell_state()
            return True

        def _apply_axis_style_fields_to_canvas(self) -> bool:
            if self._preview_canvas is None or self._preview_figure is None:
                return False
            axes = list(getattr(self._preview_figure, "axes", []) or [])
            if not axes:
                return False
            ax = axes[0]

            x_scale = self.x_scale.currentText().strip() or "linear"
            y_scale = self.y_scale.currentText().strip() or "linear"
            try:
                ax.set_xscale(x_scale)
                ax.set_yscale(y_scale)
            except Exception:
                return False

            grid_mode = self.grid_mode.currentText().strip().lower()
            grid_enabled = grid_mode != "off"
            grid_kwargs: dict[str, Any] = {
                "axis": self.grid_axis.currentText().strip().lower() or "both",
                "which": self.grid_which.currentText().strip().lower() or "major",
            }
            linestyle = self.grid_linestyle.currentText().strip()
            if linestyle:
                grid_kwargs["linestyle"] = linestyle
            for key, widget in (
                ("linewidth", self.grid_linewidth),
                ("alpha", self.grid_alpha),
            ):
                text = widget.text().strip()
                if text:
                    try:
                        grid_kwargs[key] = float(text)
                    except ValueError:
                        return False
            color = self.grid_color.text().strip()
            if color:
                grid_kwargs["color"] = color
            if grid_kwargs["axis"] not in _GRID_AXES:
                grid_kwargs["axis"] = "both"
            if grid_kwargs["which"] not in _GRID_WHICH:
                grid_kwargs["which"] = "major"
            try:
                if grid_enabled:
                    ax.grid(True, **grid_kwargs)
                else:
                    ax.grid(False)
                    for axis in (ax.xaxis, ax.yaxis):
                        for gridline in axis.get_gridlines():
                            gridline.set_visible(False)
            except Exception:
                return False
            self._last_preview_state["x_scale"] = x_scale
            self._last_preview_state["y_scale"] = y_scale
            self._last_preview_state["grid"] = bool(grid_enabled)
            self._last_preview_state["grid_kwargs"] = dict(grid_kwargs)
            self._preview_canvas.draw_idle()
            if self._preview_status is not None:
                self._preview_status.setText("Preview axes updated.")
            self._refresh_shell_state()
            return True

        def _apply_heatmap_style_fields_to_canvas(self) -> bool:
            if self._preview_canvas is None:
                return False
            mesh = self._last_preview_state.get("heatmap_artist")
            if mesh is None:
                return False
            colorbar = self._last_preview_state.get("heatmap_colorbar")

            cmap = self.heatmap_cmap.currentText().strip()
            if cmap:
                try:
                    mesh.set_cmap(cmap)
                except Exception:
                    return False

            def _optional_float_text(widget: QLineEdit) -> float | None:
                text = widget.text().strip()
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return None

            vmin = _optional_float_text(self.heatmap_vmin)
            vmax = _optional_float_text(self.heatmap_vmax)
            try:
                mesh.set_clim(vmin=vmin, vmax=vmax)
            except Exception:
                return False

            projection_width = (
                _optional_float_text(self.projection_line_width)
                if self._analysis_name == "position"
                and self._current_position_is_projection_view()
                and self._current_position_uses_continuous_color()
                else None
            )
            if projection_width is not None:
                if projection_width <= 0.0:
                    return False
                try:
                    if hasattr(mesh, "set_linewidths"):
                        mesh.set_linewidths([projection_width])
                    point_artist = self._last_preview_state.get("heatmap_point_artist")
                    if point_artist is not None and hasattr(point_artist, "set_sizes"):
                        point_count = len(point_artist.get_offsets())
                        point_artist.set_sizes(
                            [max(1.0, projection_width**2)] * point_count
                        )
                except Exception:
                    return False

            if colorbar is not None:
                colorbar_enabled = self.heatmap_colorbar_enabled.isChecked()
                try:
                    colorbar.ax.set_visible(bool(colorbar_enabled))
                except Exception:
                    pass
                raw_label = self.heatmap_colorbar_label.text().strip()
                if raw_label.casefold() == "none":
                    label = ""
                elif raw_label:
                    label = raw_label
                else:
                    label = None
                label_size_text = self.heatmap_colorbar_label_size.text().strip()
                tick_size_text = self.heatmap_colorbar_tick_size.text().strip()
                label_size = int(label_size_text) if label_size_text.isdigit() else None
                tick_size = int(tick_size_text) if tick_size_text.isdigit() else None
                if label is not None:
                    try:
                        colorbar.set_label(label, fontsize=label_size)
                    except TypeError:
                        colorbar.set_label(label)
                elif label_size is not None:
                    current_label = (
                        colorbar.ax.get_ylabel()
                        if colorbar.ax.yaxis.get_visible()
                        else colorbar.ax.get_xlabel()
                    )
                    try:
                        colorbar.set_label(current_label, fontsize=label_size)
                    except TypeError:
                        colorbar.set_label(current_label)
                if tick_size is not None:
                    colorbar.ax.tick_params(labelsize=tick_size)
                try:
                    colorbar.update_normal(mesh)
                except Exception:
                    pass

            self._last_preview_state["heatmap_cmap"] = cmap or None
            self._last_preview_state["heatmap_vmin"] = vmin
            self._last_preview_state["heatmap_vmax"] = vmax
            self._last_preview_state["projection_line_width"] = projection_width
            self._last_preview_state["heatmap_colorbar_enabled"] = (
                self.heatmap_colorbar_enabled.isChecked()
            )
            self._preview_canvas.draw_idle()
            if self._preview_status is not None:
                self._preview_status.setText("Preview heatmap style updated.")
            self._refresh_shell_state()
            return True

        def _apply_line_style_fields_to_canvas(self) -> bool:
            if self._preview_canvas is None or self._preview_figure is None:
                return False
            axes = list(getattr(self._preview_figure, "axes", []) or [])
            if not axes:
                return False
            lines = [
                line
                for ax in axes
                for line in list(getattr(ax, "lines", []) or [])
            ]
            if not lines:
                return False

            def _optional_float_text(widget: QLineEdit) -> float | None:
                text = widget.text().strip()
                if not text:
                    return None
                try:
                    return float(text)
                except ValueError:
                    return None

            line_width = _optional_float_text(self.line_width)
            line_alpha = _optional_float_text(self.line_alpha)
            marker_size = _optional_float_text(self.marker_size)
            line_style = self.line_style.currentText().strip()
            marker_mode = self.markers_mode.currentText().strip().lower()
            marker_type = self.marker_type.currentText().strip() or "o"
            marker = "" if marker_mode == "off" else marker_type
            marker_color = self.marker_color.text().strip()
            try:
                for line in lines:
                    if line_width is not None:
                        line.set_linewidth(line_width)
                    if line_alpha is not None:
                        line.set_alpha(line_alpha)
                    if line_style:
                        line.set_linestyle(line_style)
                    line.set_marker(marker)
                    if marker_size is not None:
                        line.set_markersize(marker_size)
                    if marker_color:
                        line.set_markerfacecolor(marker_color)
                        line.set_markeredgecolor(marker_color)
            except Exception:
                return False
            self._last_preview_state["line_width"] = line_width
            self._last_preview_state["markers"] = marker_mode != "off"
            self._last_preview_state["line_kwargs"] = {
                "linestyle": line_style or None,
                "alpha": line_alpha,
                "marker": marker,
                "markersize": marker_size,
            }
            self._preview_canvas.draw_idle()
            if self._preview_status is not None:
                self._preview_status.setText("Preview line style updated.")
            self._refresh_shell_state()
            return True

        def _optional_float_from_line(self, widget: QLineEdit) -> float | None:
            text = widget.text().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        def _optional_int_from_line(self, widget: QLineEdit) -> int | None:
            value = self._optional_float_from_line(widget)
            if value is None or value <= 0:
                return None
            return int(round(value))

        def _preview_export_figsize(self) -> tuple[float, float]:
            fig_width = self._optional_float_from_line(self.fig_width)
            fig_height = self._optional_float_from_line(self.fig_height)
            if (
                fig_width is not None
                and fig_height is not None
                and fig_width > 0.0
                and fig_height > 0.0
            ):
                return float(fig_width), float(fig_height)

            raw_figsize = self._last_preview_state.get("figsize")
            if isinstance(raw_figsize, (list, tuple)) and len(raw_figsize) >= 2:
                try:
                    state_width = float(raw_figsize[0])
                    state_height = float(raw_figsize[1])
                except (TypeError, ValueError):
                    state_width = 0.0
                    state_height = 0.0
                if state_width > 0.0 and state_height > 0.0:
                    return state_width, state_height

            if self._preview_figure is not None:
                try:
                    current_size = self._preview_figure.get_size_inches()
                    current_width = float(current_size[0])
                    current_height = float(current_size[1])
                except Exception:
                    current_width = 0.0
                    current_height = 0.0
                if current_width > 0.0 and current_height > 0.0:
                    return current_width, current_height

            return tuple(float(value) for value in DEFAULT_PLOT_STYLE.figure_size)

        def _resize_preview_canvas_to_figure(self) -> None:
            if self._preview_canvas is None or self._preview_figure is None:
                return
            available_width = 800
            available_height = 520
            candidates: list[QWidget] = []
            if self._preview_canvas_scroll is not None:
                viewport = self._preview_canvas_scroll.viewport()
                if viewport is not None:
                    candidates.append(viewport)
                candidates.append(self._preview_canvas_scroll)
            if self._preview_canvas_container is not None:
                candidates.append(self._preview_canvas_container)
            for widget in candidates:
                width = int(widget.width()) - 8
                height = int(widget.height()) - 8
                if width >= 160 and height >= 120:
                    available_width = width
                    available_height = height
                    break
            available_width = max(320, available_width)
            available_height = max(240, available_height)
            export_width, export_height = self._preview_export_figsize()
            export_aspect = max(1.0e-9, export_width / export_height)
            available_aspect = available_width / available_height
            if available_aspect > export_aspect:
                target_height = available_height
                target_width = int(round(target_height * export_aspect))
            else:
                target_width = available_width
                target_height = int(round(target_width / export_aspect))
            target_width = max(1, min(available_width, target_width))
            target_height = max(1, min(available_height, target_height))
            try:
                preview_dpi = max(target_width / export_width, target_height / export_height, 1.0)
                self._preview_figure.set_size_inches(export_width, export_height, forward=False)
                self._preview_figure.set_dpi(preview_dpi)
            except Exception:
                pass
            self._preview_canvas.setFixedSize(QSize(target_width, target_height))
            self._preview_canvas.resize(target_width, target_height)
            if self._preview_canvas_scroll is not None:
                self._preview_canvas_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._preview_canvas_scroll.updateGeometry()
                self._preview_canvas_scroll.viewport().update()
            self._preview_canvas.updateGeometry()
            self._preview_canvas.update()
            QTimer.singleShot(0, self._redraw_preview_canvas)

        def _redraw_preview_canvas(self) -> None:
            if self._preview_canvas is None:
                return
            try:
                self._preview_canvas.draw_idle()
            except Exception:
                pass

        def _apply_canvas_style_fields_to_canvas(self) -> bool:
            if self._preview_canvas is None or self._preview_figure is None:
                return False
            fig = self._preview_figure
            axes = list(getattr(fig, "axes", []) or [])

            fig_width = self._optional_float_from_line(self.fig_width)
            fig_height = self._optional_float_from_line(self.fig_height)
            dpi = self._optional_int_from_line(self.dpi)
            if (fig_width is None) != (fig_height is None):
                return False

            facecolor = self.figure_facecolor.text().strip()
            if facecolor:
                try:
                    fig.patch.set_facecolor(facecolor)
                except ValueError:
                    return False
            alpha = self._optional_float_from_line(self.figure_alpha)
            if alpha is not None:
                try:
                    fig.patch.set_alpha(float(alpha))
                except Exception:
                    return False

            font_family = self.font_family.text().strip()
            font_color = self.font_color.text().strip()
            base_sizes = default_plot_font_sizes(self._resolved_base_font_size_value())
            x_tick_size = self._optional_int_from_line(self.x_tick_font) or base_sizes[
                "tick_font_size"
            ]
            y_tick_size = self._optional_int_from_line(self.y_tick_font) or base_sizes[
                "tick_font_size"
            ]
            legend_size = self._optional_int_from_line(self.legend_font) or base_sizes[
                "legend_font_size"
            ]
            title_size = self._optional_int_from_line(self.title_font) or base_sizes[
                "title_font_size"
            ]
            label_size = base_sizes["label_font_size"]
            x_label_size = self._optional_int_from_line(self.x_label_font) or label_size
            y_label_size = self._optional_int_from_line(self.y_label_font) or label_size

            for ax in axes:
                text_artists = [
                    ax.title,
                    ax.xaxis.label,
                    ax.yaxis.label,
                    *list(ax.get_xticklabels()),
                    *list(ax.get_yticklabels()),
                ]
                legend = ax.get_legend()
                if legend is not None:
                    text_artists.extend(list(legend.get_texts()))
                    title = legend.get_title()
                    if title is not None:
                        text_artists.append(title)
                for artist in text_artists:
                    if font_family:
                        try:
                            artist.set_fontfamily(font_family)
                        except Exception:
                            pass
                    if font_color:
                        try:
                            artist.set_color(font_color)
                        except ValueError:
                            return False
                ax.title.set_fontsize(title_size)
                ax.xaxis.label.set_fontsize(x_label_size)
                ax.yaxis.label.set_fontsize(y_label_size)
                ax.tick_params(axis="x", labelsize=x_tick_size)
                ax.tick_params(axis="y", labelsize=y_tick_size)
                if font_color:
                    try:
                        ax.tick_params(axis="both", colors=font_color)
                    except ValueError:
                        return False
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(legend_size)
                    legend_title = legend.get_title()
                    if legend_title is not None:
                        legend_title.set_fontsize(legend_size)

            if fig_width is not None and fig_height is not None:
                self._last_preview_state["figsize"] = [float(fig_width), float(fig_height)]
            else:
                self._last_preview_state["figsize"] = list(fig.get_size_inches())
            if dpi is not None:
                self._last_preview_state["dpi"] = int(dpi)
            else:
                try:
                    state_dpi = int(self._last_preview_state.get("dpi") or DEFAULT_PLOT_STYLE.dpi)
                except (TypeError, ValueError):
                    state_dpi = int(DEFAULT_PLOT_STYLE.dpi)
                self._last_preview_state["dpi"] = state_dpi
            self._last_preview_state["font_family"] = font_family or None
            self._last_preview_state["font_color"] = font_color or None
            self._last_preview_state["figure_kwargs"] = {
                "facecolor": facecolor or None,
                "alpha": alpha,
            }
            self._resize_preview_canvas_to_figure()
            try:
                fig.tight_layout()
            except Exception:
                pass
            self._preview_canvas.draw_idle()
            if self._preview_status is not None:
                self._preview_status.setText("Preview canvas updated.")
            self._refresh_shell_state()
            return True

        def _apply_series_artist_updates_to_canvas(self) -> bool:
            if self._preview_canvas is None or self._preview_figure is None:
                return False
            line_artists = self._last_preview_state.get("line_artists")
            plotted_series = self._last_preview_state.get("plotted_xy_series")
            if not isinstance(line_artists, list) or not isinstance(plotted_series, list):
                return False
            if not line_artists or len(line_artists) < len(plotted_series):
                return False
            artist_by_series_id: dict[str, Any] = {}
            for artist, payload in zip(line_artists, plotted_series):
                if not isinstance(payload, dict):
                    continue
                series_id = str(payload.get("series_id") or "").strip()
                if series_id:
                    artist_by_series_id[series_id] = artist
            if not artist_by_series_id:
                return False

            missing_enabled_artist = False
            for index, descriptor in enumerate(self._series_descriptors_data):
                series_id = str(descriptor.get("series_id") or f"series:{index}")
                artist = artist_by_series_id.get(series_id)
                enabled = bool(
                    self._series_enabled_data[index]
                    if index < len(self._series_enabled_data)
                    else True
                )
                if enabled and artist is None:
                    missing_enabled_artist = True
                    break
                if artist is None:
                    continue
                show_raw = bool(
                    self._series_show_raw_line_data[index]
                    if index < len(self._series_show_raw_line_data)
                    else True
                )
                artist.set_visible(enabled and show_raw)
                artist.set_label(self._effective_series_label(index))
                color = self._effective_series_color(index).strip()
                if color:
                    try:
                        artist.set_color(color)
                    except ValueError:
                        pass
                alpha = (
                    self._series_alpha_data[index].strip()
                    if index < len(self._series_alpha_data)
                    else ""
                )
                if alpha:
                    try:
                        artist.set_alpha(float(alpha))
                    except ValueError:
                        pass
                width = (
                    self._series_line_widths_data[index].strip()
                    if index < len(self._series_line_widths_data)
                    else ""
                )
                if width:
                    try:
                        artist.set_linewidth(float(width))
                    except ValueError:
                        pass
                marker = (
                    self._series_markers_data[index].strip()
                    if index < len(self._series_markers_data)
                    else ""
                )
                if marker:
                    artist.set_marker(marker)
                line_kwargs = (
                    self._series_line_kwargs_data[index].strip()
                    if index < len(self._series_line_kwargs_data)
                    else ""
                )
                if line_kwargs:
                    try:
                        parsed_kwargs = _optional_json_dict(
                            line_kwargs,
                            field_name="Series Matplotlib line options",
                        )
                    except ValueError:
                        parsed_kwargs = None
                    if parsed_kwargs:
                        for key, value in parsed_kwargs.items():
                            if key == "label":
                                continue
                            setter = getattr(artist, f"set_{key}", None)
                            if callable(setter):
                                try:
                                    setter(value)
                                except Exception:
                                    pass
                artist.set_zorder(float(index + 2))

                fit_artist = artist_by_series_id.get(f"{series_id}::fit")
                if fit_artist is not None:
                    fit_enabled = bool(
                        index < len(self._series_fit_enabled_data)
                        and self._series_fit_enabled_data[index]
                    )
                    fit_artist.set_visible(fit_enabled)
                    fit_artist.set_label(self._fit_effective_label(index))
                    fit_color = (
                        self._series_fit_color_data[index].strip()
                        if index < len(self._series_fit_color_data)
                        else ""
                    )
                    if fit_color:
                        try:
                            fit_artist.set_color(fit_color)
                        except ValueError:
                            pass
                    fit_artist.set_zorder(float(index + 2.5))

                cumulative_artist = artist_by_series_id.get(f"{series_id}::cumulative")
                if cumulative_artist is not None:
                    cumulative_enabled = bool(
                        index < len(self._series_cumulative_enabled_data)
                        and self._series_cumulative_enabled_data[index]
                    )
                    cumulative_artist.set_visible(cumulative_enabled)
                    cumulative_artist.set_label(self._cumulative_effective_label(index))
                    cumulative_color = (
                        self._series_cumulative_color_data[index].strip()
                        if index < len(self._series_cumulative_color_data)
                        else ""
                    )
                    if cumulative_color:
                        try:
                            cumulative_artist.set_color(cumulative_color)
                        except ValueError:
                            pass
                    cumulative_artist.set_zorder(float(index + 2.25))
            if missing_enabled_artist:
                return False

            for ax in self._preview_figure.axes:
                legend = ax.get_legend()
                if legend is None:
                    continue
                handles, labels = ax.get_legend_handles_labels()
                visible_pairs = [
                    (handle, label)
                    for handle, label in zip(handles, labels)
                    if getattr(handle, "get_visible", lambda: True)()
                    and not str(label).startswith("_")
                ]
                legend.remove()
                if visible_pairs and self.legend_mode.currentText().strip().lower() != "off":
                    ax.legend(
                        [handle for handle, _label in visible_pairs],
                        [label for _handle, label in visible_pairs],
                        loc=self.legend_loc.currentText().strip() or "best",
                    )
            self._preview_canvas.draw_idle()
            if self._preview_status is not None:
                self._preview_status.setText("Preview series styling updated.")
            self._refresh_shell_state()
            return True

        def _handle_synced_field_edit(self, key: str) -> None:
            self._set_synced_field_mode(key, "manual")
            if key in {"x_lim", "y_lim"} and self._apply_axis_limit_fields_to_canvas(key):
                self._preview_timer.stop()
                return
            if key in {"title", "x_label", "y_label"} and self._apply_text_fields_to_canvas():
                self._preview_timer.stop()
                return
            self._schedule_preview_update()

        def _handle_synced_field_mode_changed(self, key: str, value: str) -> None:
            normalized = str(value).strip().lower()
            if normalized not in {"auto", "manual", "off"}:
                normalized = "auto"
            self._set_synced_field_mode(key, normalized)
            if self._synced_field_mode(key) == "auto":
                self._apply_preview_state_to_synced_fields(self._last_preview_state)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _apply_preview_series_state(self, render_state: dict[str, Any]) -> None:
            raw_descriptors = render_state.get("series_descriptors")
            if not isinstance(raw_descriptors, list):
                return
            descriptors = [
                dict(descriptor)
                for descriptor in raw_descriptors
                if isinstance(descriptor, dict)
            ]
            if not descriptors:
                return
            labels = [
                str(label)
                for label in render_state.get("series_labels", [])
                if str(label).strip()
            ]
            if not labels:
                labels = [
                    str(descriptor.get("default_label") or f"Series {index + 1}")
                    for index, descriptor in enumerate(descriptors)
                ]
            current_ids = [
                str(descriptor.get("series_id") or "").strip()
                for descriptor in self._series_descriptors_data
            ]
            next_ids = [
                str(descriptor.get("series_id") or "").strip()
                for descriptor in descriptors
            ]
            if current_ids == next_ids:
                return
            self._apply_series_defaults(labels, descriptors=descriptors)

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
                raise ValueError("Profile name cannot be empty. Enter a name to continue.")
            return normalized

        def _profile_name_exists(self, name: str) -> bool:
            lowered = name.casefold()
            return any(existing.casefold() == lowered for existing in self._profile_names)

        @staticmethod
        def _profile_unavailable_message(action: str) -> str:
            return f"Cannot {action} profiles in this session."

        @staticmethod
        def _profile_name_unchanged_message(name: str) -> str:
            return f"Profile name is still '{name}'. Enter a different name to rename it."

        def _set_profile_names(self, names: Sequence[str], *, active_name: str | None = None) -> None:
            normalized_names: list[str] = []
            seen_names: set[str] = set()
            for raw_name in names:
                candidate = str(raw_name).strip()
                if not candidate:
                    continue
                lowered = candidate.casefold()
                if lowered in seen_names:
                    continue
                seen_names.add(lowered)
                normalized_names.append(candidate)
            if not normalized_names:
                normalized_names = ["Default"]
            self._profile_names = normalized_names
            if active_name is not None and any(
                existing.casefold() == str(active_name).strip().casefold()
                for existing in normalized_names
            ):
                self._current_profile_name = next(
                    existing
                    for existing in normalized_names
                    if existing.casefold() == str(active_name).strip().casefold()
                )
            elif self._current_profile_name not in normalized_names:
                self._current_profile_name = normalized_names[0]
            self._sync_profile_selector()

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
            self._reset_undo_history()
            self._refresh_shell_state()

        def _reset_undo_history(self, settings: dict[str, Any] | None = None) -> None:
            baseline_settings = dict(settings) if isinstance(settings, dict) else None
            if baseline_settings is None:
                baseline_settings, error = self._safe_collect_history_settings()
                if error or baseline_settings is None:
                    self._undo_stack = []
                    self._redo_stack = []
                    self._undo_current_settings = None
                    self._undo_current_signature = None
                    self._undo_text_edit_widget_id = None
                    self._undo_text_edit_signature = None
                    self._undo_text_edit_settings = None
                    return
            self._undo_stack = []
            self._redo_stack = []
            self._undo_current_settings = deepcopy(baseline_settings)
            self._undo_current_signature = self._signature(baseline_settings)
            self._undo_text_edit_widget_id = None
            self._undo_text_edit_signature = None
            self._undo_text_edit_settings = None

        def _history_snapshot(self, settings: dict[str, Any]) -> dict[str, Any]:
            snapshot = deepcopy(settings)
            if isinstance(self._last_preview_state, dict) and self._last_preview_state:
                snapshot["_undo_preview_state"] = deepcopy(self._last_preview_state)
            return snapshot

        @staticmethod
        def _history_signature(settings: dict[str, Any]) -> str:
            filtered = {
                key: value for key, value in settings.items() if key != "_undo_preview_state"
            }
            return json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str)

        def _safe_collect_history_settings(self) -> tuple[dict[str, Any] | None, str | None]:
            settings, error = self._safe_collect_settings()
            if error or settings is None:
                return None, error
            return self._history_snapshot(settings), None

        def _sender_is_text_editor(self) -> bool:
            sender = self.sender()
            return isinstance(sender, (QLineEdit, QPlainTextEdit))

        def _install_undo_event_filters(self) -> None:
            for widget in self.findChildren(QLineEdit):
                widget.installEventFilter(self)
            for widget in self.findChildren(QPlainTextEdit):
                widget.installEventFilter(self)

        def _bind_undo_change_signals(self) -> None:
            for widget in self.findChildren(QComboBox):
                widget.currentTextChanged.connect(self._record_history_after_non_text_change)
            for widget in self.findChildren(QCheckBox):
                if widget in {self._theme_switch, self._auto_preview_checkbox}:
                    continue
                widget.toggled.connect(self._record_history_after_non_text_change)

        def _set_undo_current(self, settings: dict[str, Any]) -> None:
            self._undo_current_settings = deepcopy(settings)
            self._undo_current_signature = self._history_signature(settings)

        def _push_history_snapshot(
            self,
            stack: list[dict[str, Any]],
            settings: dict[str, Any],
        ) -> None:
            signature = self._history_signature(settings)
            if stack:
                top_signature = self._history_signature(stack[-1])
                if top_signature == signature:
                    return
            stack.append(deepcopy(settings))
            if len(stack) > 100:
                del stack[:-100]

        def _begin_text_undo_edit(self, widget: QWidget | None) -> None:
            if (
                widget is None
                or self._undo_syncing
                or self._suspend_preview_events
                or self._normalization_syncing
                or self._series_syncing
                or self._annotation_syncing
            ):
                return
            widget_id = id(widget)
            if self._undo_text_edit_widget_id == widget_id:
                return
            if self._undo_text_edit_widget_id is not None:
                self._finalize_text_undo_edit()
            settings, error = self._safe_collect_history_settings()
            if error or settings is None:
                return
            self._undo_text_edit_widget_id = widget_id
            self._undo_text_edit_settings = deepcopy(settings)
            self._undo_text_edit_signature = self._history_signature(settings)

        def _finalize_text_undo_edit(self, widget: QWidget | None = None) -> None:
            if self._undo_text_edit_widget_id is None:
                return
            if widget is not None and id(widget) != self._undo_text_edit_widget_id:
                return
            settings, error = self._safe_collect_history_settings()
            baseline_settings = self._undo_text_edit_settings
            baseline_signature = self._undo_text_edit_signature
            self._undo_text_edit_widget_id = None
            self._undo_text_edit_settings = None
            self._undo_text_edit_signature = None
            if error or settings is None or baseline_settings is None or baseline_signature is None:
                return
            current_signature = self._history_signature(settings)
            if current_signature == baseline_signature:
                self._set_undo_current(settings)
                return
            self._push_history_snapshot(self._undo_stack, baseline_settings)
            self._redo_stack = []
            self._set_undo_current(settings)

        def _record_history_after_non_text_change(self, *_unused: object) -> None:
            if (
                self._undo_syncing
                or self._suspend_preview_events
                or self._normalization_syncing
                or self._series_syncing
                or self._annotation_syncing
            ):
                return
            self._finalize_text_undo_edit()
            settings, error = self._safe_collect_history_settings()
            if error or settings is None:
                return
            signature = self._history_signature(settings)
            if self._undo_current_signature is None or self._undo_current_settings is None:
                self._set_undo_current(settings)
                return
            if signature == self._undo_current_signature:
                return
            self._push_history_snapshot(self._undo_stack, self._undo_current_settings)
            self._redo_stack = []
            self._set_undo_current(settings)

        def _restore_history_state(self, settings: dict[str, Any], *, status_text: str) -> None:
            preview_state = settings.get("_undo_preview_state")
            restore_settings = {
                key: value for key, value in settings.items() if key != "_undo_preview_state"
            }
            self._undo_syncing = True
            self._suspend_preview_events = True
            try:
                self._populate(restore_settings)
                if isinstance(preview_state, dict) and preview_state:
                    self._apply_preview_state_to_synced_fields(preview_state)
            finally:
                self._suspend_preview_events = False
                self._undo_syncing = False
            self._refresh_widget_states()
            self._set_undo_current(settings)
            self._undo_text_edit_widget_id = None
            self._undo_text_edit_signature = None
            self._undo_text_edit_settings = None
            self._status_label.setText(status_text)
            self._refresh_shell_state()
            self._schedule_preview_update()

        def _handle_undo(self) -> None:
            self._finalize_text_undo_edit()
            if not self._undo_stack:
                self._status_label.setText("Nothing to undo.")
                self._refresh_shell_state()
                return
            current_settings = self._undo_current_settings
            settings, error = self._safe_collect_history_settings()
            if not error and settings is not None:
                current_settings = deepcopy(settings)
            target_settings = deepcopy(self._undo_stack.pop())
            if current_settings is not None:
                self._push_history_snapshot(self._redo_stack, current_settings)
            self._restore_history_state(target_settings, status_text="Undid last change.")

        def _handle_redo(self) -> None:
            self._finalize_text_undo_edit()
            if not self._redo_stack:
                self._status_label.setText("Nothing to redo.")
                self._refresh_shell_state()
                return
            current_settings = self._undo_current_settings
            settings, error = self._safe_collect_history_settings()
            if not error and settings is not None:
                current_settings = deepcopy(settings)
            target_settings = deepcopy(self._redo_stack.pop())
            if current_settings is not None:
                self._push_history_snapshot(self._undo_stack, current_settings)
            self._restore_history_state(target_settings, status_text="Redid last change.")

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
                raise ValueError(profile_name_conflict_message(name))
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
                self._status_label.setText(self._profile_unavailable_message("create"))
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
                self._set_profile_names([*self._profile_names, name], active_name=name)
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
                self._status_label.setText(self._profile_unavailable_message("duplicate"))
                self._refresh_shell_state()
                return
            if on_duplicate_profile is None:
                self._status_label.setText(self._profile_unavailable_message("duplicate"))
                self._refresh_shell_state()
                return
            try:
                name = self._prompt_profile_name(
                    title_text="Duplicate profile",
                    default_value=self._next_duplicate_profile_name(),
                )
                if name is None:
                    return
                current_name = self._current_profile_name
                settings = self._collect_settings()
                on_save(current_name, settings)
                message = on_duplicate_profile(current_name, name)
                self._set_profile_names([*self._profile_names, name], active_name=name)
                self._saved_signature = self._signature(settings)
                self._status_label.setText(message)
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Duplicate profile failed", exc)

        def _handle_rename_profile(self) -> None:
            if not self._allow_named_profiles:
                self._status_label.setText(self._profile_unavailable_message("rename"))
                self._refresh_shell_state()
                return
            if on_rename_profile is None:
                self._status_label.setText(self._profile_unavailable_message("rename"))
                return
            try:
                current_name = self._current_profile_name
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
                    self._status_label.setText(self._profile_name_unchanged_message(current_name))
                    return
                if self._profile_name_exists(name):
                    raise ValueError(profile_name_conflict_message(name))
                message = on_rename_profile(current_name, name)
                self._set_profile_names(
                    [name if entry.casefold() == current_name.casefold() else entry for entry in self._profile_names],
                    active_name=name,
                )
                if on_load_profile is None:
                    raise ValueError("Profile loading is unavailable after renaming.")
                loaded = on_load_profile(name)
                self._load_settings_into_editor(loaded, profile_name=name, status_message=message, mark_saved=True)
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Rename profile failed", exc)

        def _handle_delete_profile(self) -> None:
            if not self._allow_named_profiles:
                self._status_label.setText(self._profile_unavailable_message("delete"))
                self._refresh_shell_state()
                return
            if on_delete_profile is None:
                self._status_label.setText(self._profile_unavailable_message("delete"))
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
                remaining_names = [
                    name for name in self._profile_names if name.casefold() != deleted_name.casefold()
                ]
                self._set_profile_names(
                    remaining_names,
                    active_name=next_profile_name,
                )
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

            self._theme_switch = QCheckBox("Dark mode")
            self._theme_switch.setObjectName("themeSwitch")
            self._sync_theme_switch_label()
            self._theme_switch.toggled.connect(self._handle_theme_switch_toggled)
            header_layout.addWidget(self._theme_switch)

            self._undo_button = QPushButton("Undo")
            self._undo_button.clicked.connect(self._handle_undo)
            self._register_tooltip(self._undo_button, "profiles.undo")
            self._apply_widget_tooltip(
                self._undo_button, disabled_reason="there is nothing to undo."
            )
            header_layout.addWidget(self._undo_button)
            self._undo_button.setVisible(False)

            self._redo_button = QPushButton("Redo")
            self._redo_button.clicked.connect(self._handle_redo)
            self._register_tooltip(self._redo_button, "profiles.redo")
            self._apply_widget_tooltip(
                self._redo_button, disabled_reason="there is nothing to redo."
            )
            header_layout.addWidget(self._redo_button)
            self._redo_button.setVisible(False)

            self._header_detach_preview_button = QPushButton("Detach Preview")
            self._header_detach_preview_button.clicked.connect(
                self._handle_header_preview_detach_toggle
            )
            header_layout.addWidget(self._header_detach_preview_button)

            self._save_button = QPushButton("Save Profile")
            self._save_button.setProperty("role", "primary")
            self._save_button.clicked.connect(self._handle_save)
            self._register_tooltip(self._save_button, "profiles.save")
            self._apply_widget_tooltip(self._save_button)
            header_layout.addWidget(self._save_button)

            self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
            self._undo_shortcut.activated.connect(self._handle_undo)
            self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
            self._redo_shortcut.activated.connect(self._handle_redo)

            self._exit_button = QPushButton("Exit")
            self._exit_button.clicked.connect(self.close)
            header_layout.addWidget(self._exit_button)
            root_layout.addWidget(header)

            splitter = QSplitter(Qt.Orientation.Horizontal, root)
            splitter.setChildrenCollapsible(False)
            self._splitter = splitter
            root_layout.addWidget(splitter, stretch=1)

            left_panel = QWidget(splitter)
            left_panel.setMinimumWidth(_WORKSPACE_PANEL_MIN_WIDTH)
            left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            self._workspace_panel = left_panel
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

            right_panel = _PreviewPane(
                title_text="Figure Preview",
                object_name="previewPanel",
                on_refresh=self._handle_preview,
                on_fit=self._handle_fit_preview,
                on_actual_size=self._handle_actual_size_preview,
                on_save_figure_callback=self._handle_save_figure,
                on_save_data_callback=self._handle_save_data,
                on_auto_update=self._handle_auto_preview_toggle,
                on_detach=self._handle_detach_preview,
                on_dock=None,
                register_tooltip=self._register_tooltip,
                apply_tooltip=self._apply_widget_tooltip,
                event_filter_owner=self,
                auto_update_enabled=on_save_figure is not None,
                parent=splitter,
            )
            right_panel.setMinimumWidth(320)
            self._embedded_preview_pane = right_panel
            self._activate_preview_pane(right_panel)

            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([_WORKSPACE_PANEL_WIDTH, 840])
            self._preview_splitter_sizes = splitter.sizes()

            self._status_label.setObjectName("statusBar")
            self._status_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            root_layout.addWidget(self._status_label)

            self._apply_theme_styles()

        def _is_dark_theme(self) -> bool:
            if self._theme_mode == "dark":
                return True
            if self._theme_mode == "light":
                return False
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
                    "series_row_bg": "transparent",
                    "series_row_text": "#edf3fb",
                    "series_row_disabled_text": "#7f8fa6",
                    "series_row_selected_bg": "#123c46",
                    "series_row_selected_border": "#34bed1",
                    "series_row_selected_text": "#f7fbff",
                    "series_row_button_hover": "#203349",
                    "series_row_swatch_bg": "#0e1725",
                    "series_row_swatch_border": "#607086",
                    "series_badge_original_bg": "#1f2d40",
                    "series_badge_original_border": "#566b88",
                    "series_badge_original_text": "#dce8f7",
                    "series_badge_copy_bg": "#463a16",
                    "series_badge_copy_border": "#b9973d",
                    "series_badge_copy_text": "#ffe7a3",
                    "series_badge_group_bg": "#173f35",
                    "series_badge_group_border": "#39b990",
                    "series_badge_group_text": "#cffff0",
                    "selected_card_bg": "#102f3a",
                    "selected_card_border": "#34bed1",
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
                    "tooltip_bg": "#1a2435",
                    "tooltip_border": "#4a607e",
                    "tooltip_text": "#f7fbff",
                    "placeholder_text": "#7f8fa6",
                    "item_hover": "#1c2c40",
                    "item_selected_bg": "#2aa7b8",
                    "item_selected_text": "#07151b",
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
                "series_row_bg": "transparent",
                "series_row_text": "#142033",
                "series_row_disabled_text": "#6d7b8e",
                "series_row_selected_bg": "#cbecef",
                "series_row_selected_border": "#0c7a80",
                "series_row_selected_text": "#082f34",
                "series_row_button_hover": "#d8edf0",
                "series_row_swatch_bg": "#ffffff",
                "series_row_swatch_border": "#93a4b8",
                "series_badge_original_bg": "#e7edf5",
                "series_badge_original_border": "#9aacbf",
                "series_badge_original_text": "#23324a",
                "series_badge_copy_bg": "#fff0bd",
                "series_badge_copy_border": "#c6941f",
                "series_badge_copy_text": "#4f3600",
                "series_badge_group_bg": "#d8f3e9",
                "series_badge_group_border": "#15946f",
                "series_badge_group_text": "#053e31",
                "selected_card_bg": "#d7f1f4",
                "selected_card_border": "#087982",
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
                "tooltip_bg": "#f8fbff",
                "tooltip_border": "#b7c6d8",
                "tooltip_text": "#102033",
                "placeholder_text": "#7b8797",
                "item_hover": "#edf4fa",
                "item_selected_bg": "#0f8f95",
                "item_selected_text": "#f8feff",
                "splitter": "#cad5e1",
                "scrollbar_track": "#edf2f7",
                "scrollbar_thumb": "#b9c5d4",
                "scrollbar_thumb_hover": "#95a6bb",
            }

        def _sync_theme_switch_label(self) -> None:
            if self._theme_switch is None:
                return
            is_dark = self._is_dark_theme()
            self._theme_switch.blockSignals(True)
            try:
                self._theme_switch.setChecked(is_dark)
                self._theme_switch.setText("Dark mode" if is_dark else "Light mode")
            finally:
                self._theme_switch.blockSignals(False)

        def _handle_theme_switch_toggled(self, checked: bool) -> None:
            self._theme_mode = "dark" if checked else "light"
            self._sync_theme_switch_label()
            self._apply_theme_styles()
            if hasattr(self, "series_list") and self.series_list is not None:
                for index in range(self.series_list.count()):
                    self._apply_series_list_item_visuals(self.series_list.item(index), index)
            self._update_selected_layer_card(self._series_active_index)
            self._refresh_shell_state()

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
                f"QFrame#navPanel, QFrame#inspectorPanel, QFrame#previewPanel, "
                f"QFrame#detachedPreviewPanel {{"
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
                f'QGroupBox[collapsibleBody="true"] {{'
                f"  margin-top: 0px;"
                f"  border: none;"
                f"  border-radius: 0px;"
                f"  background: transparent;"
                f"}}"
                f'QGroupBox[collapsibleBody="true"]::title {{'
                f"  color: transparent;"
                f"  padding: 0px;"
                f"  left: 0px;"
                f"  height: 0px;"
                f"}}"
                f"QFrame#collapsibleSection, QFrame#collapsibleSubsection {{"
                f"  background-color: {colors['card_bg']};"
                f"  border: 1px solid {colors['border_soft']};"
                f"  border-radius: 12px;"
                f"}}"
                f"QFrame#collapsibleSubsection {{"
                f"  background-color: {colors['panel_elevated']};"
                f"  border-radius: 10px;"
                f"}}"
                f"QFrame#collapsibleSectionHeader, QFrame#collapsibleSubsectionHeader {{"
                f"  background: transparent;"
                f"  border: none;"
                f"}}"
                f"QFrame#collapsibleSectionBody, QFrame#collapsibleSubsectionBody {{"
                f"  background: transparent;"
                f"  border: none;"
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
                f"QToolButton {{"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 7px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QToolButton:hover {{"
                f"  border-color: {colors['accent']};"
                f"  background-color: {colors['button_hover']};"
                f"}}"
                f"QToolButton:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QToolButton#collapsibleToggle {{"
                f"  padding: 10px 12px;"
                f"  border: none;"
                f"  border-radius: 10px;"
                f"  background: transparent;"
                f"  color: {colors['heading']};"
                f"  font-weight: 600;"
                f"  text-align: left;"
                f"}}"
                f"QToolButton#collapsibleToggle:hover {{"
                f"  background-color: {colors['nav_hover']};"
                f"  border: none;"
                f"}}"
                f"QToolButton#collapsibleToggle:pressed {{"
                f"  background-color: {colors['button_hover']};"
                f"}}"
                f"QToolButton#collapsibleToggle:disabled {{"
                f"  background: transparent;"
                f"  color: {colors['disabled_text']};"
                f"  border: none;"
                f"}}"
                f"QLabel#staticSectionHeaderLabel, QLabel#staticSubsectionHeaderLabel {{"
                f"  padding: 10px 12px;"
                f"  color: {colors['heading']};"
                f"  font-weight: 600;"
                f"}}"
                f"QMenu {{"
                f"  background-color: {colors['panel_elevated']};"
                f"  color: {colors['text']};"
                f"  border: 1px solid {colors['border']};"
                f"}}"
                f"QMenu::item {{"
                f"  padding: 6px 20px 6px 12px;"
                f"  color: {colors['text']};"
                f"  background: transparent;"
                f"}}"
                f"QMenu::item:disabled {{"
                f"  color: {colors['disabled_text']};"
                f"}}"
                f"QMenu::item:selected {{"
                f"  background-color: {colors['item_selected_bg']};"
                f"  color: {colors['item_selected_text']};"
                f"}}"
                f"QMenu::separator {{"
                f"  height: 1px;"
                f"  margin: 5px 8px;"
                f"  background: {colors['border_soft']};"
                f"}}"
                f"QToolTip {{"
                f"  background-color: {colors['tooltip_bg']};"
                f"  color: {colors['tooltip_text']};"
                f"  border: 1px solid {colors['tooltip_border']};"
                f"  padding: 6px 8px;"
                f"}}"
                f"QMessageBox {{"
                f"  background-color: {colors['panel_bg']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QMessageBox QLabel {{"
                f"  color: {colors['text']};"
                f"  background: transparent;"
                f"}}"
                f"QMessageBox QTextEdit, QMessageBox QPlainTextEdit {{"
                f"  border: 1px solid {colors['input_border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['input_bg']};"
                f"  color: {colors['text']};"
                f"  selection-background-color: {colors['accent_soft']};"
                f"  selection-color: {colors['text']};"
                f"}}"
                f"QMessageBox QPushButton {{"
                f"  padding: 7px 12px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"  min-width: 88px;"
                f"}}"
                f"QMessageBox QPushButton:hover {{"
                f"  border-color: {colors['accent']};"
                f"  background-color: {colors['button_hover']};"
                f"}}"
                f"QMessageBox QPushButton:pressed {{"
                f"  background-color: {colors['button_pressed']};"
                f"}}"
                f"QMessageBox QPushButton:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QDialogButtonBox QPushButton {{"
                f"  padding: 7px 12px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"  min-width: 88px;"
                f"}}"
                f"QDialogButtonBox QPushButton:hover {{"
                f"  border-color: {colors['accent']};"
                f"  background-color: {colors['button_hover']};"
                f"}}"
                f"QDialogButtonBox QPushButton:pressed {{"
                f"  background-color: {colors['button_pressed']};"
                f"}}"
                f"QDialogButtonBox QPushButton:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QLabel {{ color: {colors['text']}; }}"
                f"QLineEdit, QComboBox, QPlainTextEdit, QListWidget {{"
                f"  border: 1px solid {colors['input_border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['input_bg']};"
                f"  color: {colors['text']};"
                f"  outline: none;"
                f"  selection-background-color: {colors['accent_soft']};"
                f"  selection-color: {colors['text']};"
                f"}}"
                f"QLineEdit, QPlainTextEdit {{"
                f"  placeholder-text-color: {colors['placeholder_text']};"
                f"}}"
                f"QLineEdit, QComboBox {{ padding: 6px 8px; min-height: 18px; }}"
                f"QPlainTextEdit {{ padding: 6px; }}"
                f"QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QListWidget:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QLabel:disabled, QCheckBox:disabled, QGroupBox:disabled {{"
                f"  color: {colors['disabled_text']};"
                f"}}"
                f"QGroupBox:disabled::title {{"
                f"  color: {colors['disabled_text']};"
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
                f"  outline: none;"
                f"  selection-background-color: {colors['item_selected_bg']};"
                f"  selection-color: {colors['item_selected_text']};"
                f"  alternate-background-color: {colors['card_bg']};"
                f"}}"
                f"QComboBox QAbstractItemView::item {{"
                f"  padding: 6px 10px;"
                f"  margin: 0px;"
                f"  border: none;"
                f"  show-decoration-selected: 1;"
                f"}}"
                f"QComboBox QAbstractItemView::item:selected {{"
                f"  background-color: {colors['item_selected_bg']};"
                f"  color: {colors['item_selected_text']};"
                f"  border: none;"
                f"}}"
                f"QAbstractItemView {{"
                f"  background-color: {colors['panel_elevated']};"
                f"  color: {colors['text']};"
                f"  selection-background-color: {colors['item_selected_bg']};"
                f"  selection-color: {colors['item_selected_text']};"
                f"}}"
                f"QAbstractItemView::item:hover {{"
                f"  background-color: {colors['item_hover']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QAbstractItemView::item:selected {{"
                f"  background-color: {colors['item_selected_bg']};"
                f"  color: {colors['item_selected_text']};"
                f"}}"
                f"QAbstractItemView::item:selected:active, QAbstractItemView::item:selected:!active {{"
                f"  background-color: {colors['item_selected_bg']};"
                f"  color: {colors['item_selected_text']};"
                f"}}"
                f"QCheckBox {{ spacing: 6px; }}"
                f"QCheckBox#themeSwitch {{"
                f"  padding: 6px 10px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 999px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"  font-weight: 600;"
                f"}}"
                f"QCheckBox#themeSwitch:hover {{"
                f"  border-color: {colors['accent']};"
                f"  background-color: {colors['button_hover']};"
                f"}}"
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
                f"QCheckBox::indicator:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QAbstractItemView::indicator {{"
                f"  width: 16px;"
                f"  height: 16px;"
                f"  border-radius: 4px;"
                f"  border: 1px solid {colors['input_border']};"
                f"  background-color: {colors['input_bg']};"
                f"}}"
                f"QAbstractItemView::indicator:hover {{"
                f"  border-color: {colors['accent']};"
                f"}}"
                f"QAbstractItemView::indicator:checked {{"
                f"  background-color: {colors['accent']};"
                f"  border-color: {colors['accent']};"
                f"}}"
                f"QAbstractItemView::indicator:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QCheckBox#themeSwitch::indicator {{"
                f"  width: 34px;"
                f"  height: 18px;"
                f"  border-radius: 9px;"
                f"  background-color: {colors['input_bg']};"
                f"  border: 1px solid {colors['input_border']};"
                f"}}"
                f"QCheckBox#themeSwitch::indicator:checked {{"
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
                f"QFrame#selectedLayerCard {{"
                f"  background-color: {colors['selected_card_bg']};"
                f"  border: 1px solid {colors['selected_card_border']};"
                f"  border-radius: 8px;"
                f"}}"
                f"QLabel#selectedLayerTitle {{"
                f"  color: {colors['heading']};"
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
                f"QTabWidget#plotSubtabs QTabBar {{"
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
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 12px;"
                f"  padding: 6px;"
                f"}}"
                f"QListWidget#annotationList::item {{"
                f"  padding: 0px;"
                f"  margin: 0px;"
                f"  border: none;"
                f"  background: transparent;"
                f"  color: transparent;"
                f"}}"
                f"QListWidget#annotationList::item:hover {{"
                f"  background: transparent;"
                f"}}"
                f"QListWidget#annotationList::item:selected {{"
                f"  background: transparent;"
                f"  color: transparent;"
                f"  border: none;"
                f"}}"
                f"QListWidget#annotationList::indicator {{"
                f"  width: 0px;"
                f"  height: 0px;"
                f"  border: none;"
                f"  background: transparent;"
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
                f"QScrollBar:horizontal {{"
                f"  background: {colors['scrollbar_track']};"
                f"  height: 12px;"
                f"  margin: 2px;"
                f"  border-radius: 6px;"
                f"}}"
                f"QScrollBar::handle:horizontal {{"
                f"  background: {colors['scrollbar_thumb']};"
                f"  min-width: 24px;"
                f"  border-radius: 6px;"
                f"}}"
                f"QScrollBar::handle:horizontal:hover {{ background: {colors['scrollbar_thumb_hover']}; }}"
                f"QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, "
                f"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{"
                f"  width: 0px;"
                f"  background: transparent;"
                f"}}"
                f"QSplitter::handle {{"
                f"  background-color: {colors['splitter']};"
                f"  width: 5px;"
                f"  margin: 6px 4px;"
                f"  border-radius: 2px;"
                f"}}"
            )
            if self._detached_preview_window is not None:
                self._detached_preview_window.setStyleSheet(self.styleSheet())

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
                    self._set_line_text_if_different(
                        self.figure_alpha,
                        "" if parsed.get("alpha") is None else str(parsed.get("alpha")),
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
                    marker = str(parsed.get("marker") or "").strip()
                    self._set_combo_value(self.marker_type, marker)
                    marker_color = (
                        parsed.get("markerfacecolor")
                        if parsed.get("markerfacecolor") is not None
                        else parsed.get("markeredgecolor")
                    )
                    self._set_line_text_if_different(
                        self.marker_color,
                        "" if marker_color is None else str(marker_color),
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
                    x_params = parsed.get("_x_tick_params")
                    y_params = parsed.get("_y_tick_params")
                    x_params = x_params if isinstance(x_params, dict) else {}
                    y_params = y_params if isinstance(y_params, dict) else {}
                    direction = (
                        str(x_params.get("direction", parsed.get("direction") or "out"))
                        .strip()
                        .lower()
                    )
                    y_direction = (
                        str(y_params.get("direction", parsed.get("direction") or "out"))
                        .strip()
                        .lower()
                    )
                    axis_value = (
                        str(parsed.get("_ticks_axis", parsed.get("axis", "both"))).strip().lower()
                    )
                    minor_value = (
                        str(
                            parsed.get(
                                "_x_minor_ticks_mode",
                                parsed.get("_minor_ticks_mode") or "off",
                            )
                        )
                        .strip()
                        .lower()
                    )
                    y_minor_value = (
                        str(
                            parsed.get(
                                "_y_minor_ticks_mode",
                                parsed.get("_minor_ticks_mode") or "off",
                            )
                        )
                        .strip()
                        .lower()
                    )
                    if direction in _TICK_DIRECTIONS:
                        self._set_combo_value(self.x_tick_direction, direction)
                    if y_direction in _TICK_DIRECTIONS:
                        self._set_combo_value(self.y_tick_direction, y_direction)
                    if axis_value in _TICK_AXES:
                        x_ticks_visible = parsed.get("_x_ticks_visible")
                        y_ticks_visible = parsed.get("_y_ticks_visible")
                        self._set_combo_value(
                            self.x_ticks_mode,
                            (
                                "on"
                                if bool(x_ticks_visible)
                                else "off"
                                if x_ticks_visible is not None
                                else ("on" if axis_value in {"x", "both"} else "off")
                            ),
                        )
                        self._set_combo_value(
                            self.y_ticks_mode,
                            (
                                "on"
                                if bool(y_ticks_visible)
                                else "off"
                                if y_ticks_visible is not None
                                else ("on" if axis_value in {"y", "both"} else "off")
                            ),
                        )
                    if minor_value in _MINOR_TICKS_MODES:
                        self._set_combo_value(self.x_minor_ticks_mode, minor_value)
                    if y_minor_value in _MINOR_TICKS_MODES:
                        self._set_combo_value(self.y_minor_ticks_mode, y_minor_value)
                    self._set_line_text_if_different(
                        self.x_tick_length,
                        ""
                        if x_params.get("length", parsed.get("length")) is None
                        else str(x_params.get("length", parsed.get("length"))),
                    )
                    self._set_line_text_if_different(
                        self.y_tick_length,
                        ""
                        if y_params.get("length", parsed.get("length")) is None
                        else str(y_params.get("length", parsed.get("length"))),
                    )
                    self._set_line_text_if_different(
                        self.x_tick_width,
                        ""
                        if x_params.get("width", parsed.get("width")) is None
                        else str(x_params.get("width", parsed.get("width"))),
                    )
                    self._set_line_text_if_different(
                        self.y_tick_width,
                        ""
                        if y_params.get("width", parsed.get("width")) is None
                        else str(y_params.get("width", parsed.get("width"))),
                    )
                    self._set_line_text_if_different(
                        self.x_tick_font,
                        "" if x_params.get("labelsize") is None else str(x_params.get("labelsize")),
                    )
                    self._set_line_text_if_different(
                        self.y_tick_font,
                        "" if y_params.get("labelsize") is None else str(y_params.get("labelsize")),
                    )
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
            if (
                hasattr(self, "_preview_button")
                and self._preview_button is not None
                and self._auto_preview_checkbox is not None
            ):
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
            if hasattr(self, "_undo_button") and self._undo_button is not None:
                undo_reason = None if self._undo_stack else "there is nothing to undo."
                self._apply_widget_tooltip(self._undo_button, disabled_reason=undo_reason)
            if hasattr(self, "_redo_button") and self._redo_button is not None:
                redo_reason = None if self._redo_stack else "there is nothing to redo."
                self._apply_widget_tooltip(self._redo_button, disabled_reason=redo_reason)

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
            if hasattr(self, "_undo_button") and self._undo_button is not None:
                self._undo_button.setEnabled(bool(self._undo_stack))
            if hasattr(self, "_redo_button") and self._redo_button is not None:
                self._redo_button.setEnabled(bool(self._redo_stack))
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
            auto_preview_enabled = (
                self._auto_preview_checkbox is not None and self._auto_preview_checkbox.isChecked()
            )
            preview_mode = "Auto" if auto_preview_enabled else "Manual"
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
            if self._preview_canvas is not None and self._preview_figure is not None:
                axes = list(getattr(self._preview_figure, "axes", []) or [])
                self._canvas_axis_limit_syncing = True
                try:
                    for ax in axes:
                        try:
                            ax.relim()
                        except Exception:
                            pass
                        try:
                            ax.autoscale_view()
                        except Exception:
                            pass
                    self._preview_canvas.draw_idle()
                finally:
                    self._canvas_axis_limit_syncing = False
                if axes:
                    self._set_axis_limit_fields_from_canvas(axes[0])
                if self._preview_status is not None:
                    self._preview_status.setText("Preview view reset.")
                return
            self._set_preview_zoom(1.0)
            self._refresh_preview_pixmap()
            if self._preview_status is not None:
                self._preview_status.setText("Preview fit to workspace.")

        def _handle_actual_size_preview(self) -> None:
            if self._preview_canvas is not None:
                if self._preview_toolbar is not None:
                    try:
                        self._preview_toolbar.home()
                    except Exception:
                        pass
                else:
                    self._handle_fit_preview()
                    return
                if self._preview_status is not None:
                    self._preview_status.setText("Preview view reset.")
                return
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
            if self._preview_status is not None:
                self._preview_status.setText("Preview shown at 100%.")

        def _activate_preview_pane(
            self,
            pane: _PreviewPane,
            *,
            auto_update_checked: bool | None = None,
        ) -> None:
            if auto_update_checked is None and self._auto_preview_checkbox is not None:
                auto_update_checked = self._auto_preview_checkbox.isChecked()
            if auto_update_checked is None:
                auto_update_checked = on_save_figure is not None

            self._active_preview_pane = pane
            self._preview_frame = pane.preview_frame
            self._preview_scroll = pane.preview_scroll
            self._preview_label = pane.preview_label
            self._preview_canvas_container = pane.preview_canvas_container
            self._preview_canvas_layout = pane.preview_canvas_layout
            self._preview_canvas_scroll = None
            self._preview_status = self._status_label
            self._preview_button = pane.preview_button
            self._save_figure_button = pane.save_figure_button
            self._save_data_button = pane.save_data_button
            self._auto_preview_checkbox = pane.auto_preview_checkbox
            self._detach_preview_button = pane.detach_button
            self._dock_preview_button = pane.dock_button

            self._auto_preview_checkbox.blockSignals(True)
            try:
                self._auto_preview_checkbox.setChecked(True)
                self._auto_preview_checkbox.setEnabled(False)
            finally:
                self._auto_preview_checkbox.blockSignals(False)
            self._save_figure_button.setEnabled(on_save_figure is not None)
            self._save_data_button.setEnabled(on_save_data is not None)
            if self._data_export_button is not None:
                self._data_export_button.setEnabled(on_save_data is not None)
            self._set_preview_loading(self._preview_loading)
            if self._preview_figure is not None:
                self._install_preview_figure(self._preview_figure, close_previous=False)
            else:
                self._refresh_preview_pixmap()
            self._refresh_header_preview_detach_button()

        def _refresh_header_preview_detach_button(self) -> None:
            if self._header_detach_preview_button is None:
                return
            detached = self._detached_preview_window is not None
            self._header_detach_preview_button.setText(
                "Dock Preview" if detached else "Detach Preview"
            )
            self._header_detach_preview_button.setEnabled(self._embedded_preview_pane is not None)

        def _handle_header_preview_detach_toggle(self) -> None:
            if self._detached_preview_window is None:
                self._handle_detach_preview()
            else:
                self._handle_dock_preview()

        def _handle_detach_preview(self) -> None:
            if self._detached_preview_window is not None:
                self._detached_preview_window.raise_()
                self._detached_preview_window.activateWindow()
                return
            if self._embedded_preview_pane is None:
                return
            auto_update_checked = (
                self._auto_preview_checkbox.isChecked()
                if self._auto_preview_checkbox is not None
                else on_save_figure is not None
            )
            if self._splitter is not None:
                self._preview_splitter_sizes = self._splitter.sizes()

            detached_window = _DetachedPreviewWindow(
                on_dock_requested=self._handle_dock_preview,
                parent=self,
            )
            detached_window.setStyleSheet(self.styleSheet())
            detached_pane = _PreviewPane(
                title_text="Figure Preview",
                object_name="detachedPreviewPanel",
                on_refresh=self._handle_preview,
                on_fit=self._handle_fit_preview,
                on_actual_size=self._handle_actual_size_preview,
                on_save_figure_callback=self._handle_save_figure,
                on_save_data_callback=self._handle_save_data,
                on_auto_update=self._handle_auto_preview_toggle,
                on_detach=None,
                on_dock=self._handle_dock_preview,
                register_tooltip=self._register_tooltip,
                apply_tooltip=self._apply_widget_tooltip,
                event_filter_owner=self,
                auto_update_enabled=on_save_figure is not None,
                parent=detached_window,
            )
            detached_window.setCentralWidget(detached_pane)
            self._detached_preview_window = detached_window
            self._detached_preview_pane = detached_pane
            self._embedded_preview_pane.setVisible(False)
            if self._splitter is not None:
                left_width = (
                    self._splitter.sizes()[0] if self._splitter.sizes() else _WORKSPACE_PANEL_WIDTH
                )
                self._splitter.setSizes([max(_WORKSPACE_PANEL_MIN_WIDTH, left_width), 0])

            self._activate_preview_pane(detached_pane, auto_update_checked=auto_update_checked)
            detached_window.resize(1100, 820)
            detached_window.show()
            self._status_label.setText("Preview detached to a separate window.")
            self._refresh_header_preview_detach_button()
            self._refresh_shell_state()
            QTimer.singleShot(0, self._refresh_preview_pixmap)

        def _handle_dock_preview(self) -> None:
            if self._detached_preview_window is None or self._embedded_preview_pane is None:
                return
            auto_update_checked = (
                self._auto_preview_checkbox.isChecked()
                if self._auto_preview_checkbox is not None
                else on_save_figure is not None
            )
            detached_window = self._detached_preview_window
            self._detached_preview_window = None
            self._detached_preview_pane = None

            self._embedded_preview_pane.setVisible(True)
            self._activate_preview_pane(
                self._embedded_preview_pane,
                auto_update_checked=auto_update_checked,
            )
            if self._splitter is not None:
                if (
                    isinstance(self._preview_splitter_sizes, list)
                    and len(self._preview_splitter_sizes) > 1
                ):
                    left_width = max(
                        _WORKSPACE_PANEL_MIN_WIDTH, int(self._preview_splitter_sizes[0])
                    )
                    right_width = max(1, int(self._preview_splitter_sizes[1]))
                    self._splitter.setSizes([left_width, right_width])
                else:
                    self._splitter.setSizes([_WORKSPACE_PANEL_WIDTH, 840])

            detached_window.close_from_dock()
            detached_window.deleteLater()
            self._status_label.setText("Preview docked in the workspace.")
            self._refresh_header_preview_detach_button()
            self._refresh_shell_state()
            QTimer.singleShot(0, self._refresh_preview_pixmap)

        def _build_content_page(self) -> QWidget:
            self._tab_data = QWidget()
            self._tab_data_content = self._make_scrollable_tab(self._tab_data)
            self._build_data_tab()
            return self._tab_data

        def _build_layers_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            tabs = QTabWidget(page)
            tabs.setObjectName("plotSubtabs")
            self._layers_tabs = tabs
            self._tab_series = QWidget()
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
            # hint = QLabel(
            #     "Figure controls the visual presentation of the current plot. "
            #     "All figure-wide settings are collected here in one place."
            # )
            # hint.setWordWrap(True)
            # hint.setObjectName("sectionNote")
            # layout.addWidget(hint)
            self._figure_tabs = None
            self._tab_canvas_content = QGroupBox("Canvas and Typography")
            self._tab_lines_content = QGroupBox("Lines and Markers")
            self._tab_axes_content = QGroupBox("Axes, Ticks and Grid")
            self._tab_legend_content = QGroupBox("Legend")
            self._tab_heatmap_content = QGroupBox("Heatmap and Colorbar")

            self._build_canvas_tab()
            self._build_lines_tab()
            self._build_axes_tab()
            self._build_legend_tab()
            self._build_heatmap_tab()
            layout.addWidget(
                self._make_collapsible_section(
                    title="Canvas and Typography",
                    section_id="figure.canvas",
                    body_widget=self._tab_canvas_content,
                )
            )
            self._figure_lines_section = self._make_collapsible_section(
                title="Lines and Markers",
                section_id="figure.lines",
                body_widget=self._tab_lines_content,
            )
            layout.addWidget(self._figure_lines_section)
            layout.addWidget(
                self._make_collapsible_section(
                    title="Axes and Ticks",
                    section_id="figure.axes",
                    body_widget=self._tab_axes_content,
                )
            )
            self._figure_legend_section = self._make_collapsible_section(
                title="Legend",
                section_id="figure.legend",
                body_widget=self._tab_legend_content,
            )
            layout.addWidget(self._figure_legend_section)
            self._figure_heatmap_section = self._make_collapsible_section(
                title="Heatmap and Colorbar",
                section_id="figure.heatmap",
                body_widget=self._tab_heatmap_content,
            )
            layout.addWidget(self._figure_heatmap_section)
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
            layout.addWidget(
                self._make_collapsible_section(
                    title="Profile Selection",
                    section_id="profiles.selection",
                    body_widget=selection_group,
                )
            )

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
            layout.addWidget(
                self._make_collapsible_section(
                    title="Manage Profiles",
                    section_id="profiles.manage",
                    body_widget=manage_group,
                )
            )

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
            layout.addWidget(
                self._make_collapsible_section(
                    title="Transfer Profiles",
                    section_id="profiles.transfer",
                    body_widget=transfer_group,
                )
            )
            layout.addStretch(1)
            self._sync_profile_selector()
            return page

        def _build_advanced_page(self) -> QWidget:
            self._tab_advanced = QWidget()
            self._tab_advanced_content = self._make_scrollable_tab(self._tab_advanced)
            self._build_advanced_tab()
            return self._tab_advanced

        def _build_axes_tab(self) -> None:
            layout = QVBoxLayout(self._tab_axes_content)
            layout.setSpacing(12)

            title_box = QGroupBox("Title")
            title_layout = QVBoxLayout(title_box)
            top_form = QFormLayout()

            title_row, self.title_text, title_lock = self._lockable_line(
                placeholder="Leave blank to hide the title",
                allow_off=True,
            )
            x_label_row, self.x_label, x_label_lock = self._lockable_line(
                placeholder="Matplotlib mathtext supported, e.g. Distance ($A$)",
                allow_off=True,
            )
            y_label_row, self.y_label, y_label_lock = self._lockable_line(
                placeholder="e.g. Density (g/cm$^3$)",
                allow_off=True,
            )

            self._connect_lockable_line("title", self.title_text, title_lock, allow_off=True)
            self._connect_lockable_line("x_label", self.x_label, x_label_lock, allow_off=True)
            self._connect_lockable_line("y_label", self.y_label, y_label_lock, allow_off=True)

            self.title_font = self._positive_int_line()
            self.title_pad = self._bounded_float_line("6.0")

            self._add_form_row(top_form, "Title", title_row, tooltip_id="figure.text.title")
            self._add_form_row(
                top_form,
                "Title font",
                self.title_font,
                tooltip_id="figure.text.title_font",
            )
            self._add_form_row(
                top_form,
                "Title pad",
                self.title_pad,
                tooltip_id="figure.text.title_pad",
            )

            title_layout.addLayout(top_form)
            layout.addWidget(
                self._make_collapsible_section(
                    title="Title",
                    section_id="figure.axes.title",
                    body_widget=title_box,
                    subsection=True,
                )
            )

            x_axis_box = QGroupBox("X-axis")
            x_axis_layout = QVBoxLayout(x_axis_box)
            x_axis_form = QFormLayout()

            self.x_label_font = self._positive_int_line()
            self.x_scale = self._combo(("linear", "log", "symlog", "logit"))
            self.x_axis_scale = self._bounded_float_line("1.0")
            self.x_axis_offset = self._bounded_float_line("0.0")
            x_limits_row, self.x_min, self.x_max, x_limits_lock = self._lockable_pair()
            self._connect_lockable_line("x_lim", self.x_min, x_limits_lock)
            self.x_max.textEdited.connect(lambda _text: self._handle_synced_field_edit("x_lim"))

            x_label_pad_row, self.x_label_pad, x_label_pad_lock = self._lockable_line(
                placeholder="points"
            )
            self._connect_lockable_line("x_label_pad", self.x_label_pad, x_label_pad_lock)

            self._add_form_row(
                x_axis_form, "X label", x_label_row, tooltip_id="figure.text.x_label"
            )
            self._add_form_row(
                x_axis_form,
                "X label font",
                self.x_label_font,
                tooltip_id="figure.axes.x_label_font",
            )
            self._add_form_row(
                x_axis_form,
                "X scale",
                self.x_scale,
                tooltip_id="figure.axes.x_scale",
            )
            self._add_form_row(
                x_axis_form,
                "X scale factor",
                self.x_axis_scale,
                tooltip_id="figure.axes.x_axis_scale",
            )
            self._x_axis_transform_rows.append((x_axis_form, self.x_axis_scale))
            self._add_form_row(
                x_axis_form,
                "X offset",
                self.x_axis_offset,
                tooltip_id="figure.axes.x_axis_offset",
            )
            self._x_axis_transform_rows.append((x_axis_form, self.x_axis_offset))
            self._add_form_row(
                x_axis_form,
                "X min / max",
                x_limits_row,
                tooltip_id="figure.axes.x_limits",
            )
            self._add_form_row(
                x_axis_form,
                "X label pad",
                x_label_pad_row,
                tooltip_id="figure.axes.x_label_pad",
            )
            x_axis_layout.addLayout(x_axis_form)

            x_ticks_box = QGroupBox("X ticks")
            x_ticks_form = QFormLayout(x_ticks_box)
            self.x_ticks_mode = self._combo(_TOGGLE_MODES)
            self.x_ticks_mode.currentTextChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                x_ticks_form,
                "Show ticks",
                self.x_ticks_mode,
                tooltip_id="figure.ticks.show",
            )
            x_ticks_row, self.x_ticks, x_ticks_lock = self._lockable_line(
                placeholder="e.g. 0, 1, 2"
            )
            self._connect_lockable_line("x_ticks", self.x_ticks, x_ticks_lock)
            self.x_tick_rotation = self._bounded_float_line("degrees")
            self.x_tick_font = self._positive_int_line()
            self.x_tick_direction = self._combo(_TICK_DIRECTIONS)
            self.x_tick_length = self._bounded_float_line("points", bottom=0.0)
            self.x_tick_width = self._bounded_float_line("points", bottom=0.0)
            self.x_minor_ticks_mode = self._combo(_MINOR_TICKS_MODES)
            self._add_form_row(
                x_ticks_form,
                "Ticks",
                x_ticks_row,
                tooltip_id="figure.ticks.x_ticks",
            )
            self._add_form_row(
                x_ticks_form,
                "Rotation",
                self.x_tick_rotation,
                tooltip_id="figure.ticks.x_rotation",
            )
            self._add_form_row(
                x_ticks_form,
                "Font size",
                self.x_tick_font,
                tooltip_id="figure.ticks.font",
            )
            self._add_form_row(
                x_ticks_form,
                "Direction",
                self.x_tick_direction,
                tooltip_id="figure.ticks.direction",
            )
            self._add_form_row(
                x_ticks_form,
                "Length",
                self.x_tick_length,
                tooltip_id="figure.ticks.length",
            )
            self._add_form_row(
                x_ticks_form,
                "Width",
                self.x_tick_width,
                tooltip_id="figure.ticks.width",
            )
            self._add_form_row(
                x_ticks_form,
                "Minor ticks",
                self.x_minor_ticks_mode,
                tooltip_id="figure.ticks.minor",
            )
            self._x_ticks_rows = [
                (x_ticks_form, x_ticks_row),
                (x_ticks_form, self.x_tick_rotation),
                (x_ticks_form, self.x_tick_font),
                (x_ticks_form, self.x_tick_direction),
                (x_ticks_form, self.x_tick_length),
                (x_ticks_form, self.x_tick_width),
                (x_ticks_form, self.x_minor_ticks_mode),
            ]
            self._x_ticks_group = self._make_collapsible_section(
                title="X ticks",
                section_id="figure.axes.x_ticks",
                body_widget=x_ticks_box,
                subsection=True,
            )
            x_axis_layout.addWidget(self._x_ticks_group)
            layout.addWidget(
                self._make_collapsible_section(
                    title="X-axis",
                    section_id="figure.axes.x",
                    body_widget=x_axis_box,
                    subsection=True,
                )
            )

            y_axis_box = QGroupBox("Y-axis")
            y_axis_layout = QVBoxLayout(y_axis_box)
            y_axis_form = QFormLayout()

            self.y_label_font = self._positive_int_line()
            self.y_scale = self._combo(("linear", "log", "symlog", "logit"))
            y_limits_row, self.y_min, self.y_max, y_limits_lock = self._lockable_pair()
            self._connect_lockable_line("y_lim", self.y_min, y_limits_lock)
            self.y_max.textEdited.connect(lambda _text: self._handle_synced_field_edit("y_lim"))

            y_label_pad_row, self.y_label_pad, y_label_pad_lock = self._lockable_line(
                placeholder="points"
            )
            self._connect_lockable_line("y_label_pad", self.y_label_pad, y_label_pad_lock)

            self._add_form_row(
                y_axis_form, "Y label", y_label_row, tooltip_id="figure.text.y_label"
            )
            self._add_form_row(
                y_axis_form,
                "Y label font",
                self.y_label_font,
                tooltip_id="figure.axes.y_label_font",
            )
            self._add_form_row(
                y_axis_form,
                "Y scale",
                self.y_scale,
                tooltip_id="figure.axes.y_scale",
            )
            self._add_form_row(
                y_axis_form,
                "Y min / max",
                y_limits_row,
                tooltip_id="figure.axes.y_limits",
            )
            self._add_form_row(
                y_axis_form,
                "Y label pad",
                y_label_pad_row,
                tooltip_id="figure.axes.y_label_pad",
            )
            y_axis_layout.addLayout(y_axis_form)

            y_ticks_box = QGroupBox("Y ticks")
            y_ticks_form = QFormLayout(y_ticks_box)
            self.y_ticks_mode = self._combo(_TOGGLE_MODES)
            self.y_ticks_mode.currentTextChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                y_ticks_form,
                "Show ticks",
                self.y_ticks_mode,
                tooltip_id="figure.ticks.show",
            )
            y_ticks_row, self.y_ticks, y_ticks_lock = self._lockable_line(
                placeholder="e.g. 0, 5, 10"
            )
            self._connect_lockable_line("y_ticks", self.y_ticks, y_ticks_lock)
            self.y_tick_rotation = self._bounded_float_line("degrees")
            self.y_tick_font = self._positive_int_line()
            self.y_tick_direction = self._combo(_TICK_DIRECTIONS)
            self.y_tick_length = self._bounded_float_line("points", bottom=0.0)
            self.y_tick_width = self._bounded_float_line("points", bottom=0.0)
            self.y_minor_ticks_mode = self._combo(_MINOR_TICKS_MODES)
            self._add_form_row(
                y_ticks_form,
                "Ticks",
                y_ticks_row,
                tooltip_id="figure.ticks.y_ticks",
            )
            self._add_form_row(
                y_ticks_form,
                "Rotation",
                self.y_tick_rotation,
                tooltip_id="figure.ticks.y_rotation",
            )
            self._add_form_row(
                y_ticks_form,
                "Font size",
                self.y_tick_font,
                tooltip_id="figure.ticks.font",
            )
            self._add_form_row(
                y_ticks_form,
                "Direction",
                self.y_tick_direction,
                tooltip_id="figure.ticks.direction",
            )
            self._add_form_row(
                y_ticks_form,
                "Length",
                self.y_tick_length,
                tooltip_id="figure.ticks.length",
            )
            self._add_form_row(
                y_ticks_form,
                "Width",
                self.y_tick_width,
                tooltip_id="figure.ticks.width",
            )
            self._add_form_row(
                y_ticks_form,
                "Minor ticks",
                self.y_minor_ticks_mode,
                tooltip_id="figure.ticks.minor",
            )
            self._y_ticks_rows = [
                (y_ticks_form, y_ticks_row),
                (y_ticks_form, self.y_tick_rotation),
                (y_ticks_form, self.y_tick_font),
                (y_ticks_form, self.y_tick_direction),
                (y_ticks_form, self.y_tick_length),
                (y_ticks_form, self.y_tick_width),
                (y_ticks_form, self.y_minor_ticks_mode),
            ]
            self._y_ticks_group = self._make_collapsible_section(
                title="Y ticks",
                section_id="figure.axes.y_ticks",
                body_widget=y_ticks_box,
                subsection=True,
            )
            y_axis_layout.addWidget(self._y_ticks_group)
            layout.addWidget(
                self._make_collapsible_section(
                    title="Y-axis",
                    section_id="figure.axes.y",
                    body_widget=y_axis_box,
                    subsection=True,
                )
            )

            grid_box = QGroupBox("Grid")
            grid_form = QFormLayout(grid_box)
            self.grid_mode = self._combo(_TOGGLE_MODES)
            self.grid_mode.currentTextChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                grid_form,
                "Show grid",
                self.grid_mode,
                tooltip_id="figure.grid.show",
            )
            self.grid_linestyle = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.grid_linewidth = self._bounded_float_line(bottom=0.0)
            self.grid_alpha = self._bounded_float_line(bottom=0.0, top=1.0)
            grid_color_row, self.grid_color = self._color_field(
                placeholder="#dddddd",
                tooltip_id="figure.grid.color",
            )
            self.grid_axis = self._combo(_GRID_AXES)
            self.grid_which = self._combo(_GRID_WHICH)
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
            self._grid_rows = [
                (grid_form, self.grid_linestyle),
                (grid_form, self.grid_linewidth),
                (grid_form, self.grid_alpha),
                (grid_form, grid_color_row),
                (grid_form, self.grid_axis),
                (grid_form, self.grid_which),
            ]
            layout.addWidget(
                self._make_collapsible_section(
                    title="Grid",
                    section_id="figure.axes.grid",
                    body_widget=grid_box,
                    subsection=True,
                )
            )

            border_box = QGroupBox("Border")
            border_form = QFormLayout(border_box)

            self.axes_border_mode = self._combo(_BORDER_MODES)

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
                border_form,
                "Plot border",
                self.axes_border_mode,
                tooltip_id="figure.axes.border",
            )
            self._add_form_row(
                border_form,
                "Sides",
                sides_widget,
                tooltip_id="figure.axes.border_sides",
            )

            self._border_custom_rows = [(border_form, sides_widget)]
            self.axes_border_mode.currentTextChanged.connect(self._refresh_widget_states)

            layout.addWidget(
                self._make_collapsible_section(
                    title="Border",
                    section_id="figure.axes.border",
                    body_widget=border_box,
                    subsection=True,
                )
            )

            self._title_rows = [(top_form, self.title_font), (top_form, self.title_pad)]
            self._title_detail_widgets = [self.title_text]
            self._x_label_detail_widgets = [self.x_label]
            self._y_label_detail_widgets = [self.y_label]
            self._ticks_rows = self._x_ticks_rows + self._y_ticks_rows

            layout.addStretch(1)

        def _build_legend_tab(self) -> None:
            form = QFormLayout(self._tab_legend_content)
            self._figure_legend_section = self._tab_legend_content
            self.legend_mode = self._combo(_TOGGLE_MODES)
            self.legend_mode.currentTextChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                form,
                "Legend",
                self.legend_mode,
                tooltip_id="figure.legend.enabled",
            )
            self.legend_title = self._line()
            self.legend_title.textChanged.connect(self._refresh_widget_states)
            self.legend_title_font = self._positive_int_line()
            self.legend_loc = self._combo(_LEGEND_LOCATIONS)
            self.legend_frame_mode = self._combo(_TOGGLE_MODES)
            self.legend_columns = self._positive_int_line("1")
            self.legend_font = self._positive_int_line()
            self._add_form_row(
                form,
                "Legend title",
                self.legend_title,
                tooltip_id="figure.legend.title",
            )
            self._add_form_row(
                form,
                "Legend title font",
                self.legend_title_font,
                tooltip_id="figure.legend.font",
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
                (form, self.legend_title_font),
                (form, self.legend_loc),
                (form, self.legend_frame_mode),
                (form, self.legend_columns),
                (form, self.legend_font),
            ]

        def _build_lines_tab(self) -> None:
            layout = QVBoxLayout(self._tab_lines_content)
            lines = QWidget()
            self._figure_lines_group = lines
            lines_layout = QVBoxLayout(lines)
            lines_layout.setContentsMargins(0, 0, 0, 0)
            lines_form = QFormLayout()
            self.line_width = self._bounded_float_line(bottom=0.0)
            self.line_style = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.line_alpha = self._bounded_float_line("0.0 - 1.0", bottom=0.0, top=1.0)
            self.markers_mode = self._combo(_TOGGLE_MODES)
            self.markers_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.marker_size = self._bounded_float_line("e.g. 5", bottom=0.0)
            self.marker_type = self._combo(_MARKER_TYPES)
            marker_color_row, self.marker_color = self._color_field(
                placeholder="auto",
                tooltip_id="figure.lines.marker_color",
            )
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
            self._add_form_row(
                lines_form,
                "Marker type",
                self.marker_type,
                tooltip_id="figure.lines.marker_type",
            )
            self._add_form_row(
                lines_form,
                "Marker color",
                marker_color_row,
                tooltip_id="figure.lines.marker_color",
            )
            lines_layout.addLayout(lines_form)
            layout.addWidget(lines)
            layout.addStretch(1)
            self._marker_rows = [
                (lines_form, self.marker_size),
                (lines_form, self.marker_type),
                (lines_form, marker_color_row),
            ]

        def _build_integration_tab(self) -> None:
            layout = QVBoxLayout(self._tab_integration_content)
            form = QFormLayout()
            self.integration_mode = self._combo(_TOGGLE_MODES)
            self.integration_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.integration_source = self._combo(_INTEGRATION_SOURCES)
            range_widget = QWidget()
            range_layout = QHBoxLayout(range_widget)
            range_layout.setContentsMargins(0, 0, 0, 0)
            self.integration_x_min = self._bounded_float_line("min")
            self.integration_x_max = self._bounded_float_line("max")
            range_layout.addWidget(self.integration_x_min, stretch=1)
            range_layout.addWidget(self.integration_x_max, stretch=1)
            self.integration_baseline = self._bounded_float_line("0.0")
            self.integration_color_mode = self._combo(_INTEGRATION_COLOR_MODES)
            self.integration_color_mode.currentTextChanged.connect(self._refresh_widget_states)
            integration_color_row, self.integration_color = self._color_field(
                placeholder="#4d9de0",
                tooltip_id="figure.integration.color",
            )
            self.integration_alpha = self._bounded_float_line("0.0 - 1.0", bottom=0.0, top=1.0)

            self._register_tooltip(self.integration_mode, "figure.integration.enabled")
            self._apply_widget_tooltip(self.integration_mode)
            self._add_form_row(
                form,
                "Integration",
                self.integration_mode,
                tooltip_id="figure.integration.enabled",
            )
            self._add_form_row(
                form,
                "Data source",
                self.integration_source,
                tooltip_id="figure.integration.source",
            )
            self._add_form_row(
                form,
                "X min / max",
                range_widget,
                tooltip_id="figure.integration.range",
            )
            self._add_form_row(
                form,
                "Baseline",
                self.integration_baseline,
                tooltip_id="figure.integration.baseline",
            )
            self._add_form_row(
                form,
                "Fill color",
                self.integration_color_mode,
                tooltip_id="figure.integration.color_mode",
            )
            self._add_form_row(
                form,
                "Custom color",
                integration_color_row,
                tooltip_id="figure.integration.color",
            )
            self._add_form_row(
                form,
                "Fill alpha",
                self.integration_alpha,
                tooltip_id="figure.integration.alpha",
            )
            layout.addLayout(form)

            self._integration_summary_label = QLabel(
                "Turn integration on to show the area summary after preview refresh."
            )
            self._integration_summary_label.setWordWrap(True)
            self._integration_summary_label.setObjectName("sectionNote")
            self._register_tooltip(self._integration_summary_label, "figure.integration.summary")
            self._apply_widget_tooltip(self._integration_summary_label)
            layout.addWidget(self._integration_summary_label)
            layout.addStretch(1)

            self._integration_rows = [
                (form, self.integration_source),
                (form, range_widget),
                (form, self.integration_baseline),
                (form, self.integration_color_mode),
                (form, integration_color_row),
                (form, self.integration_alpha),
            ]
            self._integration_custom_color_row = (form, integration_color_row)

        def _build_heatmap_tab(self) -> None:
            layout = QVBoxLayout(self._tab_heatmap_content)

            value_group = QGroupBox("Data Representation")
            value_form = QFormLayout(value_group)
            self.heatmap_value_mode = self._combo(
                tuple(_HEATMAP_VALUE_LABEL_BY_MODE.values())
            )
            self.heatmap_value_mode.currentTextChanged.connect(
                self._refresh_widget_states
            )
            self.heatmap_value_mode.currentTextChanged.connect(
                self._schedule_preview_update
            )
            self._add_form_row(
                value_form,
                "Displayed values",
                self.heatmap_value_mode,
                tooltip_id="figure.heatmap.value_mode",
            )
            self.heatmap_value_description = QLabel("")
            self.heatmap_value_description.setWordWrap(True)
            self.heatmap_value_description.setObjectName("sectionNote")
            value_form.addRow(self.heatmap_value_description)
            self.heatmap_value_pipeline = QLabel(
                "Applied after sources are combined and counts are rebinned; "
                "applied before color limits and color mapping."
            )
            self.heatmap_value_pipeline.setWordWrap(True)
            self.heatmap_value_pipeline.setObjectName("sectionNote")
            value_form.addRow(self.heatmap_value_pipeline)

            self.heatmap_bulk_reference_mode = self._combo(("Automatic", "Manual"))
            self.heatmap_bulk_reference_mode.currentTextChanged.connect(
                self._refresh_widget_states
            )
            self.heatmap_bulk_reference_mode.currentTextChanged.connect(
                self._schedule_preview_update
            )
            self.heatmap_bulk_min = self._line("auto")
            self.heatmap_bulk_min.textChanged.connect(self._schedule_preview_update)
            self.heatmap_bulk_max = self._line("auto")
            self.heatmap_bulk_max.textChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                value_form,
                "Bulk reference",
                self.heatmap_bulk_reference_mode,
                tooltip_id="figure.heatmap.bulk_reference",
            )
            self._add_form_row(
                value_form,
                "Minimum distance",
                self.heatmap_bulk_min,
                tooltip_id="figure.heatmap.bulk_range",
            )
            self._add_form_row(
                value_form,
                "Maximum distance",
                self.heatmap_bulk_max,
                tooltip_id="figure.heatmap.bulk_range",
            )
            self.heatmap_bulk_summary = QLabel(
                "The resolved bulk range will appear after preview."
            )
            self.heatmap_bulk_summary.setWordWrap(True)
            self.heatmap_bulk_summary.setObjectName("sectionNote")
            value_form.addRow(self.heatmap_bulk_summary)
            self._heatmap_bulk_rows = [
                (value_form, self.heatmap_bulk_reference_mode),
                (value_form, self.heatmap_bulk_summary),
            ]
            self._heatmap_bulk_manual_rows = [
                (value_form, self.heatmap_bulk_min),
                (value_form, self.heatmap_bulk_max),
            ]
            self._heatmap_value_group = self._make_collapsible_section(
                title="Data Representation",
                section_id="figure.heatmap.value_representation",
                body_widget=value_group,
                subsection=True,
            )
            layout.addWidget(self._heatmap_value_group)

            group = QGroupBox("Color Mapping")
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
            self.heatmap_log_scale = self._combo(("Linear", "Logarithmic"))
            self.heatmap_log_scale.currentTextChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                form, "Color scale", self.heatmap_log_scale, tooltip_id="figure.heatmap.log_scale"
            )

            self._figure_heatmap_group = self._make_collapsible_section(
                title="Color Mapping",
                section_id="figure.heatmap.rendering",
                body_widget=group,
                subsection=True,
            )
            layout.addWidget(self._figure_heatmap_group)

            trajectory_group = QGroupBox("Trajectory Rendering")
            trajectory_form = QFormLayout(trajectory_group)
            self.projection_line_width = self._bounded_float_line(bottom=0.01)
            self.projection_line_width.textChanged.connect(self._schedule_preview_update)
            self._add_form_row(
                trajectory_form,
                "Trajectory line width",
                self.projection_line_width,
                tooltip_id="figure.heatmap.trajectory_width",
            )
            self._position_projection_stroke_row = (
                trajectory_form,
                self.projection_line_width,
            )
            self._heatmap_trajectory_group = self._make_collapsible_section(
                title="Trajectory Rendering",
                section_id="figure.heatmap.trajectory",
                body_widget=trajectory_group,
                subsection=True,
            )
            layout.addWidget(self._heatmap_trajectory_group)

            cb_group = QGroupBox("Colorbar")
            cb_form = QFormLayout(cb_group)
            self.heatmap_colorbar_enabled = QCheckBox("Show colorbar")
            self.heatmap_colorbar_enabled.setChecked(True)
            self.heatmap_colorbar_enabled.stateChanged.connect(self._refresh_widget_states)
            self.heatmap_colorbar_enabled.stateChanged.connect(self._schedule_preview_update)
            self._register_tooltip(self.heatmap_colorbar_enabled, "figure.heatmap.colorbar_enabled")
            self._apply_widget_tooltip(self.heatmap_colorbar_enabled)
            self._add_form_row(
                cb_form,
                "Colorbar",
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
            self._figure_colorbar_group = self._make_collapsible_section(
                title="Colorbar",
                section_id="figure.heatmap.colorbar",
                body_widget=cb_group,
                subsection=True,
            )
            layout.addWidget(self._figure_colorbar_group)
            layout.addStretch(1)

        def _build_canvas_tab(self) -> None:
            form = QFormLayout(self._tab_canvas_content)
            self.fig_width = self._bounded_float_line(bottom=0.0)
            self.fig_height = self._bounded_float_line(bottom=0.0)
            self.dpi = self._positive_int_line()
            self.font_family = self._line()
            self.base_font_size = self._positive_int_line()
            self.base_font_size.setPlaceholderText(str(DEFAULT_PLOT_STYLE.base_font_size))
            self.base_font_size.textChanged.connect(self._handle_base_font_size_changed)
            self.base_font_size.editingFinished.connect(self._on_base_font_size_committed)
            self._last_resolved_base_font_size: int = int(DEFAULT_PLOT_STYLE.base_font_size)
            figure_facecolor_row, self.figure_facecolor = self._color_field(
                placeholder="#ffffff",
                tooltip_id="figure.canvas.facecolor",
            )
            self.figure_alpha = self._bounded_float_line("0.0 - 1.0", bottom=0.0, top=1.0)
            font_color_row, self.font_color = self._color_field(
                placeholder="#000000",
                tooltip_id="figure.canvas.font_color",
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
                "Base font color",
                font_color_row,
                tooltip_id="figure.canvas.font_color",
            )
            self._add_form_row(
                form,
                "Figure facecolor",
                figure_facecolor_row,
                tooltip_id="figure.canvas.facecolor",
            )
            self._add_form_row(
                form,
                "Figure alpha",
                self.figure_alpha,
                tooltip_id="figure.canvas.alpha",
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
            self.x_label_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["label_font_size"])
            )
            self.y_label_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["label_font_size"])
            )
            self.x_tick_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["tick_font_size"])
            )
            self.y_tick_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["tick_font_size"])
            )
            self.legend_font.setPlaceholderText(
                _font_size_placeholder_text(defaults["legend_font_size"])
            )

        def _handle_base_font_size_changed(self, *_unused: object) -> None:
            self._refresh_font_size_placeholders()
            self._propagate_base_font_size_change_to_auto_fields()

        def _propagate_base_font_size_change_to_auto_fields(self) -> None:
            new_base = self._resolved_base_font_size_value()
            old_base = self._last_resolved_base_font_size
            if old_base == new_base:
                return
            old_auto = default_plot_font_sizes(old_base)
            new_auto = default_plot_font_sizes(new_base)
            pairs = [
                (self.title_font, "title_font_size"),
                (self.x_label_font, "x_label_font_size"),
                (self.y_label_font, "y_label_font_size"),
                (self.x_tick_font, "x_tick_font_size"),
                (self.y_tick_font, "y_tick_font_size"),
                (self.legend_font, "legend_font_size"),
            ]
            for widget, key in pairs:
                fallback_key = (
                    "label_font_size"
                    if key in {"x_label_font_size", "y_label_font_size"}
                    else "tick_font_size"
                    if key in {"x_tick_font_size", "y_tick_font_size"}
                    else key
                )
                if widget.text().strip() == str(old_auto[fallback_key]):
                    widget.setText(str(new_auto[fallback_key]))
            self._last_resolved_base_font_size = new_base

        def _on_base_font_size_committed(self) -> None:
            self._propagate_base_font_size_change_to_auto_fields()

        def _build_series_tab(self) -> None:
            tab_layout = QVBoxLayout(self._tab_series)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(4)

            self.series_list = _SeriesListView(self)
            self.series_list.setObjectName("seriesList")
            self.series_list.setAlternatingRowColors(True)
            self.series_list.setMinimumHeight(180)
            self.series_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.series_list.setSpacing(2)
            self.series_list.setUniformItemSizes(True)
            self.series_list.currentRowChanged.connect(self._handle_series_list_selection_change)
            self._selected_layer_card = QFrame()
            self._selected_layer_card.setObjectName("selectedLayerCard")
            selected_card_layout = QHBoxLayout(self._selected_layer_card)
            selected_card_layout.setContentsMargins(8, 6, 8, 6)
            selected_card_layout.setSpacing(8)
            self._selected_layer_swatch = QFrame()
            self._selected_layer_swatch.setObjectName("selectedLayerSwatch")
            self._selected_layer_swatch.setFixedSize(14, 14)
            selected_card_layout.addWidget(self._selected_layer_swatch)
            self._selected_layer_title = QLabel("No layer selected")
            self._selected_layer_title.setObjectName("selectedLayerTitle")
            selected_card_layout.addWidget(self._selected_layer_title, stretch=1)
            self._selected_layer_badge = None
            self._selected_layer_state = None
            self._selected_layer_source = None
            self._series_delete_button = QPushButton("Delete Layer")
            self._series_delete_button.clicked.connect(self._delete_selected_series)
            selected_card_layout.addWidget(self._series_delete_button)
            tab_layout.addWidget(self._selected_layer_card)

            scroll = QScrollArea(self._tab_series)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMinimumWidth(0)
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._tab_series_content = QWidget(scroll)
            self._tab_series_content.setMinimumWidth(0)
            self._tab_series_content.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            scroll.setWidget(self._tab_series_content)
            tab_layout.addWidget(scroll, stretch=1)

            layout = QVBoxLayout(self._tab_series_content)
            selector_row = QHBoxLayout()
            selector_row.addStretch(1)
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
            layout.addWidget(self.series_list)

            group_group = QGroupBox("Group Members")
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
            self._series_group_members.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self._series_group_members.itemChanged.connect(self._on_series_editor_changed)
            self._register_tooltip(self._series_group_members, "series.group.members")
            self._apply_widget_tooltip(self._series_group_members)
            group_layout.addWidget(self._series_group_members)
            self._series_group_summary = QLabel("This series is not grouped.")
            self._series_group_summary.setWordWrap(True)
            group_layout.addWidget(self._series_group_summary)
            self._series_group_group = self._make_collapsible_section(
                title="Group Members",
                section_id="layers.group_members",
                body_widget=group_group,
            )
            layout.addWidget(self._series_group_group)

            visibility_group = QGroupBox("Visibility and Label")
            visibility_form = QFormLayout(visibility_group)
            self.series_show_in_legend = self._combo(("on", "off"))
            self.series_show_in_legend.currentTextChanged.connect(self._on_series_editor_changed)
            self._series_show_raw_line = self._combo(("on", "off"))
            self._series_show_raw_line.currentTextChanged.connect(self._on_series_editor_changed)
            self.series_label = self._line()
            self.series_label.textChanged.connect(self._on_series_label_changed)
            self._add_form_row(
                visibility_form,
                "Show in legend",
                self.series_show_in_legend,
                tooltip_id="series.show_in_legend",
            )
            self._series_show_in_legend_row = (visibility_form, self.series_show_in_legend)
            self._add_form_row(
                visibility_form,
                "Raw data",
                self._series_show_raw_line,
                tooltip_id="series.show_raw_line",
            )
            self._series_show_raw_line_row = (visibility_form, self._series_show_raw_line)
            self._add_form_row(
                visibility_form, "Label", self.series_label, tooltip_id="series.label"
            )
            self._series_visibility_group = self._make_collapsible_section(
                title="Visibility and Label",
                section_id="layers.visibility",
                body_widget=visibility_group,
            )
            layout.addWidget(self._series_visibility_group)

            style_group = QGroupBox("Style")
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
            self.series_marker = self._combo(_MARKER_TYPES)
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
            self._series_style_group = self._make_collapsible_section(
                title="Style",
                section_id="layers.style",
                body_widget=style_group,
            )
            layout.addWidget(self._series_style_group)

            integration_group = QGroupBox("Integral")
            self._tab_integration_content = integration_group
            self._build_integration_tab()

            derived_group = QGroupBox("Derivations")
            derived_layout = QVBoxLayout(derived_group)
            derived_note = QLabel(
                "Derivations are computed from the currently displayed data after transforms, sectioning, and masking."
            )
            derived_note.setWordWrap(True)
            derived_layout.addWidget(derived_note)
            self._series_integration_group = self._make_collapsible_section(
                title="Integral",
                section_id="layers.derived.integral",
                body_widget=integration_group,
                subsection=True,
            )
            derived_layout.addWidget(self._series_integration_group)

            if self._analysis_name in {
                "density",
                "msd",
                "rdf",
                "potential",
                "position",
                "coordination",
                "orientation",
                "temperature",
            }:
                error_group = QGroupBox("Uncertainty")
                error_layout = QVBoxLayout(error_group)
                error_note = QLabel(
                    "Uncertainty overlays are tied to the base series. Leave the color field blank to match the series color automatically."
                )
                error_note.setWordWrap(True)
                error_layout.addWidget(error_note)
                error_form = QFormLayout()
                self._series_error_mode = self._combo(_TOGGLE_MODES)
                self._series_error_mode.currentTextChanged.connect(self._on_series_editor_changed)
                self._register_tooltip(self._series_error_mode, "series.error_enabled")
                self._apply_widget_tooltip(self._series_error_mode)
                self._add_form_row(
                    error_form,
                    "Enabled",
                    self._series_error_mode,
                    tooltip_id="series.error_enabled",
                )
                self._series_error_stat = self._combo(
                    tuple(_ERROR_STAT_DISPLAY[s] for s in _ERROR_STATS)
                )
                self._series_error_stat.currentTextChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    error_form,
                    "Statistic",
                    self._series_error_stat,
                    tooltip_id="series.error_stat",
                )
                self._series_error_style = self._combo(
                    tuple(_ERROR_STYLE_DISPLAY[s] for s in _ERROR_STYLES)
                )
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
                self._series_error_summary = QLabel(
                    "No uncertainty overlay configured for this series."
                )
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
                    "Shaded bands use semi-transparent fill; whiskers use thin caps at each point."
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
                self._series_uncertainty_group = self._make_collapsible_section(
                    title="Uncertainty",
                    section_id="layers.derived.uncertainty",
                    body_widget=error_group,
                    subsection=True,
                )
                derived_layout.addWidget(self._series_uncertainty_group)
            else:
                self._series_uncertainty_group = None

            cumulative_group = QGroupBox("Cumulative Average")
            cumulative_layout = QVBoxLayout(cumulative_group)
            cumulative_note = QLabel(
                "Cumulative lines are running means of the currently displayed y-values, ordered by plotted x."
            )
            cumulative_note.setWordWrap(True)
            cumulative_layout.addWidget(cumulative_note)
            cumulative_form = QFormLayout()
            self._series_cumulative_mode = self._combo(_TOGGLE_MODES)
            self._series_cumulative_mode.currentTextChanged.connect(self._on_series_editor_changed)
            self._register_tooltip(self._series_cumulative_mode, "series.cumulative_enabled")
            self._apply_widget_tooltip(self._series_cumulative_mode)
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
            self._series_cumulative_color_row, self._series_cumulative_color = self._color_field(
                placeholder="inherit from base series",
            )
            self._series_cumulative_color.textChanged.connect(self._on_series_editor_changed)
            self._add_form_row(
                cumulative_form,
                "Color",
                self._series_cumulative_color_row,
            )
            self._series_cumulative_alpha = self._line("0.0 - 1.0")
            self._series_cumulative_alpha.textChanged.connect(self._on_series_editor_changed)
            self._add_form_row(
                cumulative_form,
                "Alpha",
                self._series_cumulative_alpha,
            )
            self._series_cumulative_line_width = self._line("blank: inherit from base")
            self._series_cumulative_line_width.textChanged.connect(self._on_series_editor_changed)
            self._add_form_row(
                cumulative_form,
                "Line width",
                self._series_cumulative_line_width,
            )
            self._series_cumulative_line_style = self._combo(("", "-", "--", "-.", ":"))
            self._series_cumulative_line_style.currentTextChanged.connect(
                self._on_series_editor_changed
            )
            self._add_form_row(
                cumulative_form,
                "Line style",
                self._series_cumulative_line_style,
            )
            self._series_cumulative_detail_rows = [
                (cumulative_form, self._series_cumulative_show_in_legend),
                (cumulative_form, self._series_cumulative_label),
                (cumulative_form, self._series_cumulative_color_row),
                (cumulative_form, self._series_cumulative_alpha),
                (cumulative_form, self._series_cumulative_line_width),
                (cumulative_form, self._series_cumulative_line_style),
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
                "Leave Color/Alpha/Line width/Line style blank to inherit from the base series."
            )
            self._series_cumulative_style_note.setWordWrap(True)
            self._series_cumulative_style_note.hide()
            cumulative_layout.addWidget(self._series_cumulative_style_note)
            self._series_cumulative_detail_widgets = [
                cumulative_note,
                self._series_cumulative_summary,
                self._series_cumulative_style_note,
            ]
            self._series_cumulative_group = self._make_collapsible_section(
                title="Cumulative Average",
                section_id="layers.derived.cumulative",
                body_widget=cumulative_group,
                subsection=True,
            )
            derived_layout.addWidget(self._series_cumulative_group)

            if self._analysis_name in {
                "density",
                "msd",
                "rdf",
                "potential",
                "position",
                "coordination",
                "orientation",
                "temperature",
            }:
                fit_group = QGroupBox("Fit")
                fit_layout = QVBoxLayout(fit_group)
                fit_note = QLabel(
                    "Fits are derived child series based on the currently displayed data."
                )
                fit_note.setWordWrap(True)
                fit_layout.addWidget(fit_note)
                fit_form = QFormLayout()
                self._series_fit_mode = self._combo(_TOGGLE_MODES)
                self._set_combo_value(self._series_fit_mode, "off")
                self._series_fit_mode.currentTextChanged.connect(self._on_series_editor_changed)
                self._register_tooltip(self._series_fit_mode, "series.fit_enabled")
                self._apply_widget_tooltip(self._series_fit_mode)
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
                self._series_fit_x_min = self._line("Auto: full range")
                self._series_fit_x_min.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "X min",
                    self._series_fit_x_min,
                    tooltip_id="series.fit_x_min",
                )
                self._series_fit_x_max = self._line("Auto: full range")
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
                self._series_fit_color_row, self._series_fit_color = self._color_field(
                    placeholder="inherit from base series",
                )
                self._series_fit_color.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "Color",
                    self._series_fit_color_row,
                )
                self._series_fit_alpha = self._line("0.0 - 1.0")
                self._series_fit_alpha.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "Alpha",
                    self._series_fit_alpha,
                )
                self._series_fit_line_width = self._line("blank: inherit from base")
                self._series_fit_line_width.textChanged.connect(self._on_series_editor_changed)
                self._add_form_row(
                    fit_form,
                    "Line width",
                    self._series_fit_line_width,
                )
                self._series_fit_line_style = self._combo(("", "-", "--", "-.", ":"))
                self._series_fit_line_style.currentTextChanged.connect(
                    self._on_series_editor_changed
                )
                self._add_form_row(
                    fit_form,
                    "Line style",
                    self._series_fit_line_style,
                )
                self._series_fit_detail_rows = [
                    (fit_form, self._series_fit_type),
                    (fit_form, self._series_fit_degree),
                    (fit_form, self._series_fit_x_min),
                    (fit_form, self._series_fit_x_max),
                    (fit_form, self._series_fit_show_in_legend),
                    (fit_form, self._series_fit_label),
                    (fit_form, self._series_fit_color_row),
                    (fit_form, self._series_fit_alpha),
                    (fit_form, self._series_fit_line_width),
                    (fit_form, self._series_fit_line_style),
                ]
                fit_layout.addLayout(fit_form)

                fit_summary = QWidget()
                self._series_fit_summary_group = fit_summary
                fit_summary_layout = QVBoxLayout(fit_summary)
                fit_summary_layout.setContentsMargins(0, 0, 0, 0)
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
                    "Leave Color/Alpha/Line width/Line style blank to inherit from the base series."
                )
                self._series_fit_style_note.setWordWrap(True)
                self._series_fit_style_note.hide()
                fit_summary_layout.addWidget(self._series_fit_style_note)
                fit_layout.addWidget(fit_summary)
                self._series_fit_detail_widgets = [fit_note, fit_summary]
                self._series_fit_group = self._make_collapsible_section(
                    title="Fit",
                    section_id="layers.derived.fit",
                    body_widget=fit_group,
                    subsection=True,
                )
                derived_layout.addWidget(self._series_fit_group)

            self._series_derived_group = self._make_collapsible_section(
                title="Derivations",
                section_id="layers.derived",
                body_widget=derived_group,
            )
            layout.addWidget(self._series_derived_group)

            normalize_group = QGroupBox("Normalization")
            normalize_layout = QVBoxLayout(normalize_group)
            normalize_form = QFormLayout()
            self.norm_mode = self._combo(_NORMALIZATION_MODES)
            self.norm_mode.currentTextChanged.connect(self._on_normalization_editor_changed)
            self.norm_value = self._line("Target value (required unless mode=none)")
            self.norm_value.textChanged.connect(self._on_normalization_editor_changed)
            self.norm_x_ref = self._line("Reference x (required for value_at_x)")
            self.norm_x_ref.textChanged.connect(self._on_normalization_editor_changed)
            self._register_tooltip(self.norm_mode, "series.norm.mode")
            self._apply_widget_tooltip(self.norm_mode)
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
            self._normalization_copy_button = QPushButton("Copy settings to all layers")
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
            self._normalization_group = self._make_collapsible_section(
                title="Normalization",
                section_id="layers.normalization",
                body_widget=normalize_group,
            )
            layout.addWidget(self._normalization_group)

            metadata_group = QGroupBox("Source Metadata")
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
            self._series_metadata_group = self._make_collapsible_section(
                title="Source Metadata",
                section_id="layers.metadata",
                body_widget=metadata_group,
            )
            layout.addWidget(self._series_metadata_group)
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
            layout.addLayout(actions)

            self.annotation_list = QListWidget()
            self.annotation_list.setObjectName("annotationList")
            self.annotation_list.setAlternatingRowColors(False)
            self.annotation_list.setMinimumHeight(180)
            self.annotation_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.annotation_list.currentRowChanged.connect(
                self._handle_annotation_list_selection_change
            )
            self._register_tooltip(self.annotation_list, "annotations.list")
            self._apply_widget_tooltip(self.annotation_list)
            list_group = QGroupBox("")
            list_layout = QVBoxLayout(list_group)
            list_layout.addWidget(self.annotation_list)

            panel = QGroupBox("")
            panel_layout = QVBoxLayout(panel)
            panel_form = QFormLayout()

            self._annotation_enabled_mode = self._combo(_TOGGLE_MODES)
            self._annotation_enabled_mode.currentTextChanged.connect(
                self._on_annotation_editor_changed
            )
            self._register_tooltip(self._annotation_enabled_mode, "annotations.enabled")
            self._apply_widget_tooltip(self._annotation_enabled_mode)
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

            content_splitter = QSplitter(Qt.Orientation.Horizontal)
            content_splitter.addWidget(
                self._make_static_section(
                    title="Annotations",
                    body_widget=list_group,
                )
            )
            content_splitter.addWidget(
                self._make_static_section(
                    title="Selected Annotation",
                    body_widget=panel,
                )
            )
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

        def _refresh_annotation_list(self) -> None:
            if not hasattr(self, "annotation_list"):
                return
            self._annotation_syncing = True
            try:
                while self.annotation_list.count():
                    item = self.annotation_list.takeItem(0)
                    if item is None:
                        continue
                    widget = self.annotation_list.itemWidget(item)
                    if widget is not None:
                        self.annotation_list.removeItemWidget(item)
                        widget.deleteLater()
                for index in range(len(self._annotations_data)):
                    entry = self._annotations_data[index]
                    item = QListWidgetItem()
                    tooltip_text = "\n".join(
                        [
                            _annotation_primary_title(entry, index=index + 1),
                            f"Type: {_annotation_type_label(str(entry.get('type') or 'text'))}",
                            f"Coordinates: {str(entry.get('coord_system') or 'axes')}",
                        ]
                    )
                    item.setToolTip(tooltip_text)
                    self.annotation_list.addItem(item)

                    def _select_row(current: int = index) -> None:
                        self.annotation_list.setCurrentRow(current)

                    def _move_row_up(current: int = index) -> None:
                        self._handle_annotation_row_move(current, -1)

                    def _move_row_down(current: int = index) -> None:
                        self._handle_annotation_row_move(current, 1)

                    row_widget = _AnnotationRowWidget(
                        on_select=_select_row,
                        on_move_up=_move_row_up,
                        on_move_down=_move_row_down,
                    )
                    row_widget.update_content(
                        text=self._annotation_display_text(index),
                        enabled=bool(entry.get("enabled", True)),
                        selected=index == self._annotation_active_index,
                        can_move_up=index > 0,
                        can_move_down=index < len(self._annotations_data) - 1,
                        tooltip_text=tooltip_text,
                        theme=self._theme_tokens(),
                    )
                    item.setSizeHint(row_widget.sizeHint())
                    self.annotation_list.setItemWidget(item, row_widget)
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

        def _handle_annotation_row_move(self, index: int, delta: int) -> None:
            if self._annotation_syncing:
                return
            if index < 0 or index >= len(self._annotations_data):
                return
            self.annotation_list.setCurrentRow(index)
            self._move_annotation(delta)

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
            if not self._sender_is_text_editor():
                self._record_history_after_non_text_change()
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
            self._record_history_after_non_text_change()
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
            self._record_history_after_non_text_change()
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
            self._refresh_widget_states()
            self._record_history_after_non_text_change()
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
            self._record_history_after_non_text_change()
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

        def _update_integration_summary(self) -> None:
            summary_label = getattr(self, "_integration_summary_label", None)
            if summary_label is None:
                return
            if not hasattr(self, "integration_mode"):
                return
            if self.integration_mode.currentText().strip().lower() == "off":
                summary_label.setText(
                    "Turn integration on to show the area summary after preview refresh."
                )
                return
            raw = self._last_preview_state.get("integration_summaries")
            if not isinstance(raw, list):
                summary_label.setText("Integration summary will appear after preview refresh.")
                return
            lines: list[str] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or item.get("series_id") or "Series").strip()
                status = str(item.get("status") or "").strip().lower()
                source = str(item.get("source") or "plotted").strip()
                if status == "ok":
                    lines.append(
                        f"{label}: area={_format_float_value(item.get('signed_area'))}, "
                        f"absolute area={_format_float_value(item.get('absolute_area'))}, "
                        f"x={_format_float_value(item.get('x_min'))}.."
                        f"{_format_float_value(item.get('x_max'))}, "
                        f"points={item.get('point_count')}, source={source}"
                    )
                else:
                    reason = str(item.get("reason") or "Integration is unavailable.").strip()
                    lines.append(f"{label}: {reason}")
            summary_label.setText("\n".join(lines) if lines else "No integration result available.")

        def _update_heatmap_value_summary(self) -> None:
            if not hasattr(self, "heatmap_value_mode"):
                return
            label = self.heatmap_value_mode.currentText().strip()
            mode = _HEATMAP_VALUE_MODE_BY_LABEL.get(label, "raw_counts")
            if hasattr(self, "heatmap_value_description"):
                self.heatmap_value_description.setText(
                    _HEATMAP_VALUE_DESCRIPTION_BY_MODE[mode]
                )
            if not hasattr(self, "heatmap_bulk_summary"):
                return
            if mode != "bulk_relative_enrichment":
                self.heatmap_bulk_summary.setText("")
                return
            reference = self._last_preview_state.get("heatmap_bulk_reference")
            if not isinstance(reference, dict):
                self.heatmap_bulk_summary.setText(
                    "The resolved bulk range will appear after preview."
                )
                return
            lower = _format_float_value(reference.get("resolved_min"))
            upper = _format_float_value(reference.get("resolved_max"))
            row_count = int(reference.get("row_count") or 0)
            resolved_mode = str(reference.get("mode") or "auto").strip().capitalize()
            self.heatmap_bulk_summary.setText(
                f"{resolved_mode} reference: {lower} to {upper} \u00c5 "
                f"({row_count} contributing distance bins)."
            )

        def _build_analysis_data_sections(self, layout: QVBoxLayout) -> None:
            analysis = self._analysis_name
            if analysis is None:
                return

            if analysis == "msd":
                selection_title = "Profile Selection"
                selection = QGroupBox(selection_title)
                selection_form = QFormLayout(selection)
                self.analysis_species = self._line("Leave blank to use file metadata")
                self.analysis_species.textChanged.connect(self._handle_series_identity_change)
                self._add_form_row(
                    selection_form,
                    "Species",
                    self.analysis_species,
                    tooltip_id="data.profile.species",
                )
                selection_note = QLabel(
                    "Chooses which stored profile is loaded from the current source."
                )
                selection_note.setObjectName("sectionNote")
                selection_note.setWordWrap(True)
                selection_form.addRow(selection_note)
                layout.addWidget(
                    self._make_collapsible_section(
                        title=selection_title,
                        section_id=f"data.{analysis}.selection",
                        body_widget=selection,
                    )
                )

            if analysis == "rdf":
                self._rdf_source_pair_count_label = QLabel("")
                self._rdf_source_pair_list_label = QLabel("")

            if analysis == "coordination":
                coord_species_a_options = [
                    str(value)
                    for value in self._profile_filter_options.get("species_a", [])
                    if str(value).strip()
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
                self._coordination_source_contract_label = QLabel("")
                self._coordination_source_dimensions_label = QLabel("")
                self._coordination_source_quantities_label = QLabel("")

            if analysis == "density":
                mapping = QGroupBox("Mapping")
                mapping_form = QFormLayout(mapping)
                self._density_species_checkboxes = {}
                species_widget = QWidget(mapping)
                species_layout = QGridLayout(species_widget)
                species_layout.setContentsMargins(0, 0, 0, 0)
                species_layout.setSpacing(4)
                species_options = self._profile_filter_options.get("density_species_options")
                if isinstance(species_options, list) and species_options:
                    resolved_species_options = [
                        (
                            str(option.get("value") or option.get("label") or "").strip(),
                            str(option.get("label") or option.get("value") or "").strip(),
                        )
                        for option in species_options
                        if isinstance(option, dict)
                    ]
                else:
                    resolved_species_options = [
                        (label, label)
                        for label in self._density_target_filter_labels()
                        if label != "All targets"
                    ]
                option_count = max(1, len(resolved_species_options))
                species_columns = 1
                if option_count > 12:
                    species_columns = 4
                elif option_count > 6:
                    species_columns = 3
                elif option_count > 2:
                    species_columns = 2
                grid_index = 0
                for value, label in resolved_species_options:
                    if not value:
                        continue
                    checkbox = QCheckBox(label or value)
                    checkbox.setChecked(True)
                    checkbox.setProperty("density_species", value)
                    checkbox.stateChanged.connect(self._handle_density_species_checkbox_changed)
                    species_layout.addWidget(
                        checkbox,
                        grid_index // species_columns,
                        grid_index % species_columns,
                    )
                    grid_index += 1
                    self._density_species_checkboxes[value] = checkbox
                if not self._density_species_checkboxes:
                    species_layout.addWidget(QLabel("No species filters available"), 0, 0)
                self._add_form_row(
                    mapping_form,
                    "Species",
                    species_widget,
                    tooltip_id="data.density.target",
                )
                available_density_view_types = self._profile_filter_options.get("density_view_types")
                if isinstance(available_density_view_types, list) and available_density_view_types:
                    density_view_type_labels = tuple(
                        _DENSITY_VIEW_TYPE_LABEL_BY_ID.get(
                            canonical_plot_view_id(str(view_type).strip().lower()),
                            plot_view_display_label(PLOT_VIEW_1D_LINE),
                        )
                        for view_type in available_density_view_types
                    )
                else:
                    density_view_type_labels = (
                        _DENSITY_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_1D_LINE],
                    )
                self.density_view_type = self._combo(density_view_type_labels)
                self.density_view_type.currentTextChanged.connect(self._handle_density_mapping_change)
                available_modes = self._profile_filter_options.get("available_modes")
                if isinstance(available_modes, list) and available_modes:
                    mode_labels: list[str] = []
                    for mode in available_modes:
                        if mode == "distance":
                            mode_labels.append("Distance")
                        elif mode in {"x", "y", "z"}:
                            mode_labels.append(mode.upper())
                    density_x_mode_labels = tuple(mode_labels) if mode_labels else _DENSITY_X_MODE_LABELS
                else:
                    density_x_mode_labels = _DENSITY_X_MODE_LABELS
                self.density_x_mode = self._combo(density_x_mode_labels)
                self.density_x_mode.currentTextChanged.connect(self._handle_density_mapping_change)
                self.density_2d_x_axis = self._combo(_DENSITY_X_MODE_LABELS)
                self.density_2d_y_axis = self._combo(("Y", "X", "Z", "Distance"))
                self.density_2d_x_axis.currentTextChanged.connect(self._handle_density_mapping_change)
                self.density_2d_y_axis.currentTextChanged.connect(self._handle_density_mapping_change)
                self.density_quantity = self._combo(("mass", "number"))
                self.density_quantity.currentTextChanged.connect(self._handle_density_mapping_change)
                self._add_form_row(
                    mapping_form,
                    "View type",
                    self.density_view_type,
                    tooltip_id="data.density.summary.mapping",
                )
                self._add_form_row(
                    mapping_form,
                    "X-axis quantity",
                    self.density_x_mode,
                    tooltip_id="data.density.x_values",
                )
                self._density_mapping_1d_rows.append((mapping_form, self.density_x_mode))
                self._add_form_row(
                    mapping_form,
                    "X-axis quantity",
                    self.density_2d_x_axis,
                    tooltip_id="data.density.source.quantities",
                )
                self._density_mapping_2d_rows.append((mapping_form, self.density_2d_x_axis))
                self._add_form_row(
                    mapping_form,
                    "Y-axis quantity",
                    self.density_2d_y_axis,
                    tooltip_id="data.density.source.quantities",
                )
                self._density_mapping_2d_rows.append((mapping_form, self.density_2d_y_axis))
                self._add_form_row(
                    mapping_form,
                    "Y quantity",
                    self.density_quantity,
                    tooltip_id="data.density.quantity",
                )
                self._density_filter_widgets = {}
                for axis_label, axis_id in (("X range", "x"), ("Y range", "y"), ("Z range", "z"), ("Distance range", "distance")):
                    range_widget = QWidget(mapping)
                    range_layout = QHBoxLayout(range_widget)
                    range_layout.setContentsMargins(0, 0, 0, 0)
                    default_range = self._density_axis_range_defaults(axis_id)
                    lower = self._bounded_float_line(
                        self._density_axis_range_text(axis_id, 0) or "min",
                        bottom=default_range[0] if default_range is not None else None,
                        top=default_range[1] if default_range is not None else None,
                    )
                    upper = self._bounded_float_line(
                        self._density_axis_range_text(axis_id, 1) or "max",
                        bottom=default_range[0] if default_range is not None else None,
                        top=default_range[1] if default_range is not None else None,
                    )
                    lower.textChanged.connect(self._handle_density_range_change)
                    upper.textChanged.connect(self._handle_density_range_change)
                    range_layout.addWidget(lower)
                    range_layout.addWidget(upper)
                    self._density_filter_widgets[axis_id] = (lower, upper)
                    self._add_form_row(
                        mapping_form,
                        axis_label,
                        range_widget,
                        tooltip_id="data.density.source.quantities",
                    )
                    self._density_filter_rows[axis_id] = (mapping_form, range_widget)
                layout.addWidget(
                    self._make_collapsible_section(
                        title="Mapping",
                        section_id="data.density.view",
                        body_widget=mapping,
                    )
                )

            if analysis == "potential":
                self._potential_source_contract_label = QLabel("")
                self._potential_source_dimensions_label = QLabel("")
                self._potential_source_quantities_label = QLabel("")

                mapping = QGroupBox("Mapping")
                mapping_form = QFormLayout(mapping)
                self.potential_view_type = self._combo(
                    list(_POTENTIAL_VIEW_TYPE_LABEL_BY_ID.values())
                )
                self.potential_view_type.currentTextChanged.connect(
                    self._handle_potential_mapping_change
                )
                self.potential_series_mode = self._combo(
                    list(_POTENTIAL_SERIES_LABEL_BY_ID.values())
                )
                self.potential_series_mode.currentTextChanged.connect(
                    self._handle_potential_mapping_change
                )
                self._add_form_row(
                    mapping_form,
                    "View type",
                    self.potential_view_type,
                    tooltip_id="data.potential.mapping.view_type",
                )
                self._add_form_row(
                    mapping_form,
                    "Y quantity",
                    self.potential_series_mode,
                    tooltip_id="data.potential.mapping.series",
                )
                mapping_note = QLabel(
                    "Potential uses a 1D Line view with record_id on the x-axis and the selected potential quantity on y. Tabular values belong in data export."
                )
                mapping_note.setWordWrap(True)
                mapping_note.setObjectName("sectionNote")
                mapping_form.addRow(mapping_note)
                layout.addWidget(
                    self._make_collapsible_section(
                        title="Mapping",
                        section_id="data.potential.mapping",
                        body_widget=mapping,
                    )
                )

                self._potential_summary_x_axis_label = QLabel("")
                self._potential_summary_total_rows_label = QLabel("")
                self._potential_summary_complete_rows_label = QLabel("")
                self._potential_summary_incomplete_rows_label = QLabel("")
                self._potential_mapping_status_label = QLabel("")
                self._potential_mapping_summary_label = QLabel("")
                self._potential_backend_summary_label = QLabel("")

            if analysis == "position":
                self._build_position_mapping_sections(layout)

            if analysis == "coordination":
                view = QGroupBox("Mapping")
                view_form = QFormLayout(view)
                self.coordination_component = self._combo(
                    tuple(
                        _COORDINATION_VIEW_TYPE_LABEL_BY_ID[view_type_id]
                        for view_type_id in self._coordination_supported_view_type_ids()
                    )
                )
                self.coordination_component.currentTextChanged.connect(self._handle_coordination_mapping_change)
                self.coordination_line_x_quantity = self._combo(
                    tuple(_COORDINATION_LINE_X_QUANTITY_LABEL_BY_BACKEND.values())
                )
                self.coordination_line_x_quantity.currentTextChanged.connect(
                    self._handle_coordination_mapping_change
                )
                self.coordination_time_axis = self._combo(("ps", "fs", "step", "frame"))
                self.coordination_time_axis.currentTextChanged.connect(self._handle_coordination_mapping_change)
                self._add_form_row(
                    view_form,
                    "Species A",
                    self.coord_species_a,
                    tooltip_id="data.coordination.species_a",
                )
                self._add_form_row(
                    view_form,
                    "Species B",
                    self.coord_species_b,
                    tooltip_id="data.coordination.species_b",
                )
                self._add_form_row(
                    view_form,
                    "Axis",
                    self.analysis_axis,
                    tooltip_id="data.coordination.axis",
                )
                self._add_form_row(
                    view_form,
                    "View type",
                    self.coordination_component,
                    tooltip_id="data.coordination.view_type",
                )
                self._add_form_row(
                    view_form,
                    "X-axis quantity",
                    self.coordination_line_x_quantity,
                    tooltip_id="data.coordination.x_quantity",
                )
                self._coordination_line_x_quantity_row = (
                    view_form,
                    self.coordination_line_x_quantity,
                )
                self._add_form_row(
                    view_form,
                    "Time unit",
                    self.coordination_time_axis,
                    tooltip_id="data.coordination.time_axis",
                )
                self._coordination_time_axis_row = (view_form, self.coordination_time_axis)
                note = QLabel(
                    "These controls generate a 1D Line coordination profile or a 2D Heatmap with time on X and distance on Y."
                )
                note.setWordWrap(True)
                note.setObjectName("sectionNote")
                view_form.addRow(note)
                layout.addWidget(
                    self._make_collapsible_section(
                        title="Mapping",
                        section_id="data.coordination.view",
                        body_widget=view,
                    )
                )

                self._coordination_mapping_status_label = QLabel("")
                self._coordination_mapping_summary_label = QLabel("")
                self._coordination_backend_summary_label = QLabel("")

            if analysis == "orientation":
                self._orientation_source_contract_label = QLabel("")
                self._orientation_source_dimensions_label = QLabel("")
                self._orientation_source_quantities_label = QLabel("")

                view = QGroupBox("Mapping")
                view_form = QFormLayout(view)
                self.orientation_view_type = self._combo(
                    (
                        _ORIENTATION_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_1D_LINE],
                        _ORIENTATION_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_2D_HEATMAP],
                    )
                )
                self.orientation_view_type.currentTextChanged.connect(
                    self._handle_orientation_mapping_change
                )
                self.orientation_component = self._combo(
                    tuple(_ORIENTATION_LINE_QUANTITY_LABEL_BY_BACKEND.values())
                )
                self.orientation_component.currentTextChanged.connect(self._handle_orientation_mapping_change)
                self.orientation_line_x_axis = self._combo(_DENSITY_X_MODE_LABELS)
                self._set_combo_value(self.orientation_line_x_axis, "Distance")
                self.orientation_line_x_axis.currentTextChanged.connect(
                    self._handle_orientation_mapping_change
                )
                self.orientation_angle = self._combo(("polar", "azimuthal"))
                self.orientation_angle.currentTextChanged.connect(self._handle_orientation_mapping_change)
                self.orientation_heatmap_x_axis = self._combo(_DENSITY_X_MODE_LABELS)
                self.orientation_heatmap_y_axis = self._combo(("Y", "X", "Z", "Distance"))
                self._set_combo_value(self.orientation_heatmap_x_axis, "X")
                self._set_combo_value(self.orientation_heatmap_y_axis, "Y")
                self.orientation_heatmap_x_axis.currentTextChanged.connect(
                    self._handle_orientation_mapping_change
                )
                self.orientation_heatmap_y_axis.currentTextChanged.connect(
                    self._handle_orientation_mapping_change
                )
                self._add_form_row(
                    view_form,
                    "View type",
                    self.orientation_view_type,
                    tooltip_id="data.orientation.view_type",
                )
                self._add_form_row(
                    view_form,
                    "Y quantity",
                    self.orientation_component,
                    tooltip_id="data.orientation.y_quantity",
                )
                self._orientation_line_quantity_row = (view_form, self.orientation_component)
                self._add_form_row(
                    view_form,
                    "X-axis quantity",
                    self.orientation_line_x_axis,
                    tooltip_id="data.orientation.source.quantities",
                )
                self._orientation_line_x_axis_row = (view_form, self.orientation_line_x_axis)
                self._add_form_row(
                    view_form,
                    "Angle quantity",
                    self.orientation_angle,
                    tooltip_id="data.orientation.angle",
                )
                self._add_form_row(
                    view_form,
                    "X-axis quantity",
                    self.orientation_heatmap_x_axis,
                    tooltip_id="data.orientation.source.quantities",
                )
                self._orientation_mapping_2d_rows.append((view_form, self.orientation_heatmap_x_axis))
                self._add_form_row(
                    view_form,
                    "Y-axis quantity",
                    self.orientation_heatmap_y_axis,
                    tooltip_id="data.orientation.source.quantities",
                )
                self._orientation_mapping_2d_rows.append((view_form, self.orientation_heatmap_y_axis))
                self._orientation_filter_widgets = {}
                for axis_label, axis_id in (("X range", "x"), ("Y range", "y"), ("Z range", "z"), ("Distance range", "distance")):
                    range_widget = QWidget(view)
                    range_layout = QHBoxLayout(range_widget)
                    range_layout.setContentsMargins(0, 0, 0, 0)
                    default_range = self._orientation_axis_range_defaults(axis_id)
                    lower = self._bounded_float_line(
                        self._orientation_axis_range_text(axis_id, 0) or "min",
                        bottom=default_range[0] if default_range is not None else None,
                        top=default_range[1] if default_range is not None else None,
                    )
                    upper = self._bounded_float_line(
                        self._orientation_axis_range_text(axis_id, 1) or "max",
                        bottom=default_range[0] if default_range is not None else None,
                        top=default_range[1] if default_range is not None else None,
                    )
                    lower.textChanged.connect(self._handle_orientation_mapping_change)
                    upper.textChanged.connect(self._handle_orientation_mapping_change)
                    range_layout.addWidget(lower)
                    range_layout.addWidget(upper)
                    self._orientation_filter_widgets[axis_id] = (lower, upper)
                    self._add_form_row(
                        view_form,
                        axis_label,
                        range_widget,
                        tooltip_id="data.orientation.source.quantities",
                    )
                    self._orientation_filter_rows[axis_id] = (view_form, range_widget)
                note = QLabel(
                    "1D presets show orientation along distance; 2D Heatmap uses the stored sparse spatial grid when available."
                )
                note.setWordWrap(True)
                note.setObjectName("sectionNote")
                view_form.addRow(note)
                layout.addWidget(
                    self._make_collapsible_section(
                        title="Mapping",
                        section_id="data.orientation.view",
                        body_widget=view,
                    )
                )

                self._orientation_mapping_status_label = QLabel("")
                self._orientation_mapping_summary_label = QLabel("")
                self._orientation_backend_summary_label = QLabel("")

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
                    "show_raw_line": self._series_show_raw_line_data[index]
                    if index < len(self._series_show_raw_line_data)
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
                    "fit_range_mode": _fit_range_mode_from_limits(
                        self._series_fit_x_mins_data[index]
                        if index < len(self._series_fit_x_mins_data)
                        else "",
                        self._series_fit_x_maxs_data[index]
                        if index < len(self._series_fit_x_maxs_data)
                        else "",
                    ),
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
                    "cumulative_color": self._series_cumulative_color_data[index]
                    if index < len(self._series_cumulative_color_data)
                    else "",
                    "cumulative_alpha": self._series_cumulative_alpha_data[index]
                    if index < len(self._series_cumulative_alpha_data)
                    else "",
                    "cumulative_line_width": self._series_cumulative_line_width_data[index]
                    if index < len(self._series_cumulative_line_width_data)
                    else "",
                    "cumulative_line_style": self._series_cumulative_line_style_data[index]
                    if index < len(self._series_cumulative_line_style_data)
                    else "",
                    "integration_enabled": self._series_integration_enabled_data[index]
                    if index < len(self._series_integration_enabled_data)
                    else False,
                    "integration_source": self._series_integration_source_data[index]
                    if index < len(self._series_integration_source_data)
                    else "plotted",
                    "integration_x_min": self._series_integration_x_min_data[index]
                    if index < len(self._series_integration_x_min_data)
                    else "",
                    "integration_x_max": self._series_integration_x_max_data[index]
                    if index < len(self._series_integration_x_max_data)
                    else "",
                    "integration_baseline": self._series_integration_baseline_data[index]
                    if index < len(self._series_integration_baseline_data)
                    else "0.0",
                    "integration_color_mode": self._series_integration_color_mode_data[index]
                    if index < len(self._series_integration_color_mode_data)
                    else "Auto",
                    "integration_color": self._series_integration_color_data[index]
                    if index < len(self._series_integration_color_data)
                    else "",
                    "integration_alpha": self._series_integration_alpha_data[index]
                    if index < len(self._series_integration_alpha_data)
                    else "0.25",
                    "fit_color": self._series_fit_color_data[index]
                    if index < len(self._series_fit_color_data)
                    else "",
                    "fit_alpha": self._series_fit_alpha_data[index]
                    if index < len(self._series_fit_alpha_data)
                    else "",
                    "fit_line_width": self._series_fit_line_width_data[index]
                    if index < len(self._series_fit_line_width_data)
                    else "",
                    "fit_line_style": self._series_fit_line_style_data[index]
                    if index < len(self._series_fit_line_style_data)
                    else "",
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
            new_show_raw_line: list[bool] = []
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
            new_cumulative_colors: list[str] = []
            new_cumulative_alphas: list[str] = []
            new_cumulative_line_widths: list[str] = []
            new_cumulative_line_styles: list[str] = []
            new_integration_enabled: list[bool] = []
            new_integration_source: list[str] = []
            new_integration_x_min: list[str] = []
            new_integration_x_max: list[str] = []
            new_integration_baseline: list[str] = []
            new_integration_color_mode: list[str] = []
            new_integration_color: list[str] = []
            new_integration_alpha: list[str] = []
            new_fit_colors: list[str] = []
            new_fit_alphas: list[str] = []
            new_fit_line_widths: list[str] = []
            new_fit_line_styles: list[str] = []
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
                source_kind = (
                    "group"
                    if str(descriptor.get("source_kind") or "").strip().lower() == "group"
                    else "source"
                )
                descriptor["source_kind"] = source_kind
                descriptor["is_generated"] = bool(
                    descriptor.get("is_generated", source_kind == "group")
                )
                if source_kind != "group":
                    descriptor["source_series_id"] = (
                        str(descriptor.get("source_series_id") or "").strip()
                        or descriptor["series_id"]
                    )
                previous = existing_by_id.get(descriptor["series_id"], {})

                new_descriptors.append(descriptor)
                new_default_labels.append(default_label)
                new_label_overrides.append(str(previous.get("label_override") or "").strip())
                new_colors.append(str(previous.get("color") or "").strip())
                new_enabled.append(bool(previous.get("enabled", True)))
                new_show_in_legend.append(bool(previous.get("show_in_legend", True)))
                new_show_raw_line.append(bool(previous.get("show_raw_line", True)))
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
                new_cumulative_colors.append(str(previous.get("cumulative_color") or "").strip())
                new_cumulative_alphas.append(str(previous.get("cumulative_alpha") or "").strip())
                new_cumulative_line_widths.append(
                    str(previous.get("cumulative_line_width") or "").strip()
                )
                new_cumulative_line_styles.append(
                    str(previous.get("cumulative_line_style") or "").strip()
                )
                new_integration_enabled.append(bool(previous.get("integration_enabled", False)))
                new_integration_source.append(
                    str(previous.get("integration_source") or "Plotted data").strip()
                    or "Plotted data"
                )
                new_integration_x_min.append(str(previous.get("integration_x_min") or "").strip())
                new_integration_x_max.append(str(previous.get("integration_x_max") or "").strip())
                new_integration_baseline.append(
                    str(previous.get("integration_baseline") or "0.0").strip() or "0.0"
                )
                new_integration_color_mode.append(
                    str(previous.get("integration_color_mode") or "Auto").strip() or "Auto"
                )
                new_integration_color.append(str(previous.get("integration_color") or "").strip())
                new_integration_alpha.append(
                    str(previous.get("integration_alpha") or "0.25").strip() or "0.25"
                )
                new_fit_colors.append(str(previous.get("fit_color") or "").strip())
                new_fit_alphas.append(str(previous.get("fit_alpha") or "").strip())
                new_fit_line_widths.append(str(previous.get("fit_line_width") or "").strip())
                new_fit_line_styles.append(str(previous.get("fit_line_style") or "").strip())
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
            self._series_show_raw_line_data = new_show_raw_line
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
            self._series_cumulative_color_data = new_cumulative_colors
            self._series_cumulative_alpha_data = new_cumulative_alphas
            self._series_cumulative_line_width_data = new_cumulative_line_widths
            self._series_cumulative_line_style_data = new_cumulative_line_styles
            self._series_integration_enabled_data = new_integration_enabled
            self._series_integration_source_data = new_integration_source
            self._series_integration_x_min_data = new_integration_x_min
            self._series_integration_x_max_data = new_integration_x_max
            self._series_integration_baseline_data = new_integration_baseline
            self._series_integration_color_mode_data = new_integration_color_mode
            self._series_integration_color_data = new_integration_color
            self._series_integration_alpha_data = new_integration_alpha
            self._series_fit_color_data = new_fit_colors
            self._series_fit_alpha_data = new_fit_alphas
            self._series_fit_line_width_data = new_fit_line_widths
            self._series_fit_line_style_data = new_fit_line_styles
            self._series_line_widths_data = new_widths
            self._series_markers_data = new_markers
            self._series_line_kwargs_data = new_line_kwargs
            self._series_normalization_modes_data = new_norm_modes
            self._series_normalization_values_data = new_norm_values
            self._series_normalization_x_refs_data = new_norm_x_refs
            self._validate_series_state_lengths()
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
            self._update_rdf_source_summary()

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
                "source_kind": "source",
                "source_series_id": f"series:{index}",
                "is_generated": False,
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

        def _series_state_attr_names(self) -> tuple[str, ...]:
            return (
                "_series_descriptors_data",
                "_series_labels_data",
                "_series_label_overrides_data",
                "_series_colors_data",
                "_series_enabled_data",
                "_series_show_in_legend_data",
                "_series_show_raw_line_data",
                "_series_alpha_data",
                "_series_error_enabled_data",
                "_series_error_stats_data",
                "_series_error_styles_data",
                "_series_error_colors_data",
                "_series_error_label_overrides_data",
                "_series_error_show_in_legend_data",
                "_series_fit_enabled_data",
                "_series_fit_label_overrides_data",
                "_series_fit_show_in_legend_data",
                "_series_fit_types_data",
                "_series_fit_degrees_data",
                "_series_fit_range_modes_data",
                "_series_fit_x_mins_data",
                "_series_fit_x_maxs_data",
                "_series_fit_color_data",
                "_series_fit_alpha_data",
                "_series_fit_line_width_data",
                "_series_fit_line_style_data",
                "_series_cumulative_enabled_data",
                "_series_cumulative_label_overrides_data",
                "_series_cumulative_show_in_legend_data",
                "_series_cumulative_color_data",
                "_series_cumulative_alpha_data",
                "_series_cumulative_line_width_data",
                "_series_cumulative_line_style_data",
                "_series_integration_enabled_data",
                "_series_integration_source_data",
                "_series_integration_x_min_data",
                "_series_integration_x_max_data",
                "_series_integration_baseline_data",
                "_series_integration_color_mode_data",
                "_series_integration_color_data",
                "_series_integration_alpha_data",
                "_series_line_widths_data",
                "_series_markers_data",
                "_series_line_kwargs_data",
                "_series_normalization_modes_data",
                "_series_normalization_values_data",
                "_series_normalization_x_refs_data",
            )

        def _iter_series_state_lists(self) -> list[tuple[str, list[Any]]]:
            state_lists: list[tuple[str, list[Any]]] = []
            for name in self._series_state_attr_names():
                values = getattr(self, name)
                if not isinstance(values, list):
                    raise RuntimeError(f"Internal GUI layer state '{name}' is not a list.")
                state_lists.append((name, values))
            return state_lists

        def _validate_series_state_lengths(self) -> None:
            expected = len(self._series_descriptors_data)
            mismatched = [
                f"{name}={len(values)}"
                for name, values in self._iter_series_state_lists()
                if len(values) != expected
            ]
            if mismatched:
                raise RuntimeError(
                    "Internal GUI layer state is inconsistent: "
                    + ", ".join(mismatched)
                    + f" (expected {expected})."
                )

        def _apply_series_id_order(self, requested_order: list[str] | None) -> None:
            current_ids = self._current_series_id_order()
            resolved_order = _resolve_series_id_order(current_ids, requested_order)
            if resolved_order == current_ids:
                return
            index_by_id = {series_id: index for index, series_id in enumerate(current_ids)}
            indices = [index_by_id[series_id] for series_id in resolved_order]
            for name, values in self._iter_series_state_lists():
                setattr(self, name, [values[index] for index in indices])
            self._validate_series_state_lengths()

        def _enabled_partitioned_series_id_order(self) -> list[str]:
            current_ids = self._current_series_id_order()
            enabled_by_id = {
                str(descriptor.get("series_id") or f"series:{index}"): bool(
                    self._series_enabled_data[index]
                )
                for index, descriptor in enumerate(self._series_descriptors_data)
                if index < len(self._series_enabled_data)
            }
            group_by_id = {
                str(descriptor.get("series_id") or f"series:{index}"): (
                    str(descriptor.get("source_kind") or "source").strip().lower() == "group"
                )
                for index, descriptor in enumerate(self._series_descriptors_data)
            }
            return _partition_series_ids_for_display_order(
                current_ids,
                enabled_by_id=enabled_by_id,
                group_by_id=group_by_id,
            )

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

        def _effective_series_state(self, index: int) -> dict[str, Any]:
            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or f"series:{index}").strip()
            source_kind = str(descriptor.get("source_kind") or "source").strip().lower()
            is_group = source_kind == "group"
            is_generated = is_group or bool(descriptor.get("is_generated", False))
            return {
                "series_id": series_id,
                "descriptor": descriptor,
                "enabled": bool(
                    self._series_enabled_data[index]
                    if 0 <= index < len(self._series_enabled_data)
                    else True
                ),
                "label": self._effective_series_label(index),
                "color": self._effective_series_color(index),
                "show_in_legend": bool(
                    self._series_show_in_legend_data[index]
                    if 0 <= index < len(self._series_show_in_legend_data)
                    else True
                ),
                "source_kind": source_kind,
                "is_group": is_group,
                "is_generated": is_generated,
                "layer_role": "group" if is_group else "copy" if is_generated else "original",
                "source_series_id": str(
                    descriptor.get("source_series_id") or descriptor.get("series_id") or series_id
                ).strip(),
                "member_series_ids": [
                    str(member_id).strip()
                    for member_id in descriptor.get("member_series_ids", [])
                    if str(member_id).strip()
                ],
            }

        def _effective_series_color(self, index: int) -> str:
            explicit_color = ""
            if 0 <= index < len(self._series_colors_data):
                explicit_color = self._series_colors_data[index].strip()
            if explicit_color:
                return explicit_color
            if self._series_is_generated(index):
                return self._default_generated_series_color(index)
            original_source_ids: list[str] = []
            for descriptor_index, descriptor in enumerate(self._series_descriptors_data):
                if self._series_is_generated(descriptor_index):
                    continue
                source_id = str(
                    descriptor.get("source_series_id")
                    or descriptor.get("series_id")
                    or f"series:{descriptor_index}"
                ).strip()
                if source_id:
                    original_source_ids.append(source_id)
            if original_source_ids:
                default_colors = default_series_colors(len(original_source_ids))
                source_id = str(
                    self._series_descriptor(index).get("source_series_id")
                    or self._series_descriptor(index).get("series_id")
                    or f"series:{index}"
                ).strip()
                if source_id in original_source_ids:
                    return default_colors[original_source_ids.index(source_id)]
            return ""

        def _series_layer_role_label(self, index: int) -> str:
            return {
                "group": "Group",
                "copy": "Copy",
                "original": "Original",
            }.get(self._series_layer_role(index), "Layer")

        def _series_source_summary(self, index: int) -> str:
            descriptor = self._series_descriptor(index)
            if self._series_is_group(index):
                member_ids = [
                    str(value).strip()
                    for value in descriptor.get("member_series_ids", [])
                    if str(value).strip()
                ]
                return f"{len(member_ids)} member(s): {', '.join(member_ids) if member_ids else 'none'}"
            if self._series_is_generated(index):
                source_id = str(descriptor.get("source_series_id") or "").strip()
                return f"Copy of {source_id or 'source series'}"
            source_name = str(descriptor.get("source_name") or "").strip()
            return f"Original data series{f' from {source_name}' if source_name else ''}"

        def _update_selected_layer_card(self, index: int | None = None) -> None:
            if self._selected_layer_card is None:
                return
            if index is None:
                index = self._series_active_index
            if index < 0 or index >= len(self._series_descriptors_data):
                if self._selected_layer_title is not None:
                    self._selected_layer_title.setText("No layer selected")
                return
            state = self._effective_series_state(index)
            label = str(state["label"])
            if self._selected_layer_title is not None:
                self._selected_layer_title.setText(label)
            if self._selected_layer_swatch is not None:
                color = str(state["color"])
                swatch_color = QColor(color)
                if not swatch_color.isValid():
                    swatch_color = QColor(self._theme_tokens()["accent"])
                self._selected_layer_swatch.setStyleSheet(
                    f"background-color: {swatch_color.name()};"
                    f"border: 1px solid {self._theme_tokens()['border']};"
                    "border-radius: 4px;"
                )
            if self._series_delete_button is not None:
                can_delete = self._series_is_generated(index)
                self._series_delete_button.setEnabled(can_delete)
                self._series_delete_button.setToolTip(
                    "Delete this generated layer."
                    if can_delete
                    else "Original data series cannot be deleted here; turn them off instead."
                )

        def _coerce_settings_view_mapping(self, settings: dict[str, Any]) -> PlotViewMapping | None:
            raw = settings.get("view_mapping")
            if isinstance(raw, PlotViewMapping):
                return raw
            if isinstance(raw, dict):
                try:
                    return deserialize_plot_view_mapping(raw)
                except ValueError:
                    return None
            return None

        def _density_contract(self) -> PlotDataContract:
            if self._is_density_heatmap_mode():
                contract = getattr(self, "_density_heatmap_data_contract", None)
                if isinstance(contract, PlotDataContract):
                    return contract
                return _fallback_density_heatmap_plot_data_contract()
            contract = self._density_data_contract
            if isinstance(contract, PlotDataContract):
                return contract
            return _fallback_density_plot_data_contract()

        def _density_target_filter_labels(self) -> tuple[str, ...]:
            labels = ["All targets"]
            seen = {"All targets"}
            for descriptor in self._series_descriptors_data:
                target = str(
                    descriptor.get("density_species")
                    or descriptor.get("default_label")
                    or ""
                ).strip()
                if not target or target in seen:
                    continue
                labels.append(target)
                seen.add(target)
            return tuple(labels)

        def _density_target_for_descriptor(self, descriptor: dict[str, Any]) -> str:
            return str(
                descriptor.get("density_species")
                or descriptor.get("default_label")
                or ""
            ).strip()

        def _enabled_density_species(self) -> set[str] | None:
            checkboxes = getattr(self, "_density_species_checkboxes", {})
            if not isinstance(checkboxes, dict) or not checkboxes:
                return None
            return {
                str(value)
                for value, checkbox in checkboxes.items()
                if getattr(checkbox, "isChecked", lambda: False)()
            }

        def _apply_density_enabled_species_settings(self, value: Any) -> None:
            checkboxes = getattr(self, "_density_species_checkboxes", {})
            if not isinstance(checkboxes, dict) or not checkboxes:
                return
            if not isinstance(value, (list, tuple, set)):
                return
            enabled = {str(item).strip() for item in value if str(item).strip()}
            if not enabled:
                return
            available = set(checkboxes)
            selected = enabled & available
            if not selected:
                return
            self._density_species_checkbox_syncing = True
            try:
                for species, checkbox in checkboxes.items():
                    checkbox.blockSignals(True)
                    try:
                        checkbox.setChecked(str(species) in selected)
                    finally:
                        checkbox.blockSignals(False)
            finally:
                self._density_species_checkbox_syncing = False

        def _density_descriptor_passes_species_filter(self, descriptor: dict[str, Any]) -> bool:
            enabled_species = self._enabled_density_species()
            if enabled_species is None:
                return True
            target = self._density_target_for_descriptor(descriptor)
            return not target or target in enabled_species

        def _position_species_options(self) -> list[tuple[str, str]]:
            raw_options = self._profile_filter_options.get("position_species_options")
            options: list[tuple[str, str]] = []
            seen: set[str] = set()
            if isinstance(raw_options, list):
                for option in raw_options:
                    if not isinstance(option, dict):
                        continue
                    value = str(option.get("value") or "").strip()
                    if not value or value in seen:
                        continue
                    label = str(option.get("label") or value).strip() or value
                    options.append((value, label))
                    seen.add(value)
            if options:
                return options
            for descriptor in self._series_descriptors_data:
                value = str(
                    descriptor.get("position_species")
                    or descriptor.get("rendered_species")
                    or descriptor.get("default_label")
                    or ""
                ).strip()
                if not value or value in seen:
                    continue
                options.append((value, value))
                seen.add(value)
            return options

        def _enabled_position_species(self) -> set[str] | None:
            checkboxes = getattr(self, "_position_species_checkboxes", {})
            if not isinstance(checkboxes, dict) or not checkboxes:
                return None
            return {
                str(value)
                for value, checkbox in checkboxes.items()
                if getattr(checkbox, "isChecked", lambda: False)()
            }

        def _density_axis_range_defaults(self, axis_id: str) -> tuple[float, float] | None:
            ranges = self._profile_filter_options.get("density_axis_ranges")
            if not isinstance(ranges, dict):
                return None
            raw_range = ranges.get(str(axis_id).strip().lower())
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                return None
            try:
                lower = float(raw_range[0])
                upper = float(raw_range[1])
            except (TypeError, ValueError):
                return None
            if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
                return None
            return lower, upper

        def _density_axis_range_text(self, axis_id: str, index: int) -> str:
            defaults = self._density_axis_range_defaults(axis_id)
            if defaults is None:
                return ""
            return f"{defaults[index]:.6g}"

        def _density_selected_axis_for_bin_role(self, role: str) -> str | None:
            if self._analysis_name != "density":
                return None
            if self._is_density_heatmap_mode():
                widget = (
                    getattr(self, "density_2d_y_axis", None)
                    if str(role).strip().lower() == "y"
                    else getattr(self, "density_2d_x_axis", None)
                )
                if widget is None:
                    return "y" if str(role).strip().lower() == "y" else "x"
                return _DENSITY_X_MODE_BY_LABEL.get(
                    widget.currentText().strip().lower(),
                    "y" if str(role).strip().lower() == "y" else "x",
                )
            if str(role).strip().lower() == "y":
                return None
            return self._selected_density_x_mode() if hasattr(self, "density_x_mode") else "distance"

        def _density_default_bin_width(self, role: str) -> float | None:
            defaults = self._profile_filter_options.get("density_default_axis_bin_widths_A")
            if not isinstance(defaults, dict):
                return None
            axis_id = self._density_selected_axis_for_bin_role(role)
            if axis_id is None:
                return None
            try:
                value = float(defaults.get(axis_id))
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value > 0.0 else None

        def _density_default_bin_width_text(self, role: str) -> str:
            value = self._density_default_bin_width(role)
            return "" if value is None else f"{value:.6g}"

        def _apply_density_default_bin_width_texts(self) -> None:
            if self._analysis_name != "density":
                return
            previous = getattr(self, "_density_auto_bin_width_texts", {})
            if not isinstance(previous, dict):
                previous = {}
            updated: dict[str, str] = {}
            for role, widget_name in (("x", "x_bin_width"), ("y", "y_bin_width")):
                widget = getattr(self, widget_name, None)
                if widget is None:
                    continue
                default_text = self._density_default_bin_width_text(role)
                updated[role] = default_text
                current_text = widget.text().strip()
                previous_text = str(previous.get(role) or "")
                if default_text and (not current_text or current_text == previous_text):
                    widget.setText(default_text)
                elif not default_text and current_text == previous_text:
                    widget.setText("")
            self._density_auto_bin_width_texts = updated

        def _density_effective_filter_values(
            self,
            axis_id: str,
            lower: float | None,
            upper: float | None,
        ) -> tuple[float | None, float | None]:
            defaults = self._density_axis_range_defaults(axis_id)
            if defaults is None:
                return lower, upper
            if lower is not None and abs(float(lower) - defaults[0]) <= 1.0e-9:
                lower = None
            if upper is not None and abs(float(upper) - defaults[1]) <= 1.0e-9:
                upper = None
            return lower, upper

        def _orientation_axis_range_defaults(self, axis_id: str) -> tuple[float, float] | None:
            ranges = self._profile_filter_options.get("orientation_axis_ranges")
            if not isinstance(ranges, dict):
                return None
            raw_range = ranges.get(str(axis_id).strip().lower())
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                return None
            try:
                lower = float(raw_range[0])
                upper = float(raw_range[1])
            except (TypeError, ValueError):
                return None
            if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
                return None
            return lower, upper

        def _orientation_axis_range_text(self, axis_id: str, index: int) -> str:
            defaults = self._orientation_axis_range_defaults(axis_id)
            if defaults is None:
                return ""
            return f"{defaults[index]:.6g}"

        def _orientation_effective_filter_values(
            self,
            axis_id: str,
            lower: float | None,
            upper: float | None,
        ) -> tuple[float | None, float | None]:
            defaults = self._orientation_axis_range_defaults(axis_id)
            if defaults is None:
                return lower, upper
            if lower is not None and abs(float(lower) - defaults[0]) <= 1.0e-9:
                lower = None
            if upper is not None and abs(float(upper) - defaults[1]) <= 1.0e-9:
                upper = None
            return lower, upper

        def _density_current_view_type_id(self) -> str:
            if not hasattr(self, "density_view_type"):
                return PLOT_VIEW_1D_LINE
            return _DENSITY_VIEW_TYPE_ID_BY_LABEL.get(
                self.density_view_type.currentText().strip(),
                PLOT_VIEW_1D_LINE,
            )

        def _density_active_view_type_id(self) -> str:
            if self._analysis_name != "density":
                return PLOT_VIEW_1D_LINE
            return self._density_current_view_type_id()

        def _normalize_density_view_type_id(self, value: Any) -> str:
            token = str(value or "").strip().lower()
            if token in {"2d", "2d heatmap", "heatmap", "heatmap_2d", "projection2d", "plot_2d_heatmap"}:
                return PLOT_VIEW_2D_HEATMAP
            return PLOT_VIEW_1D_LINE

        def _clean_density_view_state(self, state: dict[str, Any]) -> dict[str, Any]:
            cleaned = deepcopy(state)
            for key in (
                "_profile_filter_options",
                "data_contract",
                "source_selection",
                "style",
                "density_view_states",
            ):
                cleaned.pop(key, None)
            return cleaned

        def _default_density_view_state(self, view_type: str) -> dict[str, Any]:
            defaults = self._clean_density_view_state(deepcopy(self._default_profile_settings))
            defaults["_density_view_state_initialized"] = True
            normalized = self._normalize_density_view_type_id(view_type)
            if normalized == PLOT_VIEW_2D_HEATMAP:
                defaults.update(
                    {
                        "x_lim": None,
                        "y_lim": None,
                        "x_min": None,
                        "x_max": None,
                        "y_min": None,
                        "y_max": None,
                        "density_2d_x_axis": "x",
                        "density_2d_y_axis": "y",
                        "x_bin_width": None,
                        "y_bin_width": None,
                        "x_bin_reducer": None,
                        "y_bin_reducer": None,
                        "legend": False,
                        "markers": False,
                        "annotations": None,
                        "_gui_sync_modes": None,
                    }
                )
                defaults["view_mapping"] = serialize_plot_view_mapping(
                    density_plot_options_to_view_mapping(
                        view_type=PLOT_VIEW_2D_HEATMAP,
                        quantity=str(defaults.get("quantity") or "mass"),
                    )
                )
            else:
                defaults.update(
                    {
                        "x_lim": None,
                        "y_lim": None,
                        "x_min": None,
                        "x_max": None,
                        "y_min": None,
                        "y_max": None,
                        "x_bin_width": None,
                        "y_bin_width": None,
                        "y_bin_reducer": None,
                        "annotations": None,
                        "_gui_sync_modes": None,
                    }
                )
                defaults["view_mapping"] = serialize_plot_view_mapping(
                    density_plot_options_to_view_mapping(
                        view_type=PLOT_VIEW_1D_LINE,
                        x_mode=str(defaults.get("x_mode") or "distance"),
                        quantity=str(defaults.get("quantity") or "mass"),
                    )
                )
            return defaults

        def _snapshot_density_view_state(self, view_type: str) -> dict[str, Any]:
            normalized = self._normalize_density_view_type_id(view_type)
            if self._analysis_name != "density" or not hasattr(self, "density_view_type"):
                return {}
            try:
                snapshot = self._collect_settings()
            except Exception:
                snapshot = {}
            if not isinstance(snapshot, dict):
                snapshot = {}
            snapshot = deepcopy(snapshot)
            snapshot = self._clean_density_view_state(snapshot)
            snapshot["_density_view_state_initialized"] = True
            snapshot["density_active_view_type"] = normalized
            return snapshot

        def _active_density_settings_with_view_state(
            self,
            settings: dict[str, Any],
            view_type: str,
            state: dict[str, Any],
        ) -> dict[str, Any]:
            merged = dict(settings)
            merged.update(deepcopy(state))
            normalized = self._normalize_density_view_type_id(view_type)
            merged["density_active_view_type"] = normalized
            if normalized == PLOT_VIEW_2D_HEATMAP:
                merged["view_mapping"] = serialize_plot_view_mapping(
                    density_plot_options_to_view_mapping(
                        view_type=PLOT_VIEW_2D_HEATMAP,
                        quantity=str(merged.get("quantity") or "mass"),
                    )
                )
            else:
                merged["view_mapping"] = serialize_plot_view_mapping(
                    density_plot_options_to_view_mapping(
                        view_type=PLOT_VIEW_1D_LINE,
                        x_mode=str(merged.get("x_mode") or "distance"),
                        quantity=str(merged.get("quantity") or "mass"),
                    )
                )
            return merged

        def _apply_density_view_state(self, view_type: str, state: dict[str, Any]) -> None:
            if self._analysis_name != "density":
                return
            normalized = self._normalize_density_view_type_id(view_type)
            restore_state = (
                deepcopy(state)
                if isinstance(state, dict) and state.get("_density_view_state_initialized")
                else self._default_density_view_state(normalized)
            )
            restore_settings = self._active_density_settings_with_view_state(
                self._collect_settings() if hasattr(self, "density_view_type") else {},
                normalized,
                restore_state,
            )
            previous_switching = self._density_view_state_switching
            previous_suspend = self._suspend_preview_events
            self._density_view_state_switching = True
            self._suspend_preview_events = True
            try:
                self._populate(restore_settings)
                self._density_active_view_type = normalized
            finally:
                self._suspend_preview_events = previous_suspend
                self._density_view_state_switching = previous_switching

        def _merge_density_view_state_into_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
            if self._analysis_name != "density":
                return settings
            active = self._density_active_view_type_id()
            states = deepcopy(self._density_view_states)
            active_state = self._clean_density_view_state(deepcopy(settings))
            active_state["_density_view_state_initialized"] = True
            active_state["density_active_view_type"] = active
            states[active] = active_state
            states[active]["_density_view_state_initialized"] = True
            merged = dict(settings)
            merged["density_active_view_type"] = active
            merged["density_view_states"] = states
            return merged

        def _normalize_position_view_type_id(self, value: Any) -> str:
            token = str(value or "").strip().lower()
            if token in {
                "2d",
                "2d heatmap",
                "2d map",
                "trajectory_2d",
                "scatter_2d",
                "projection",
                "projection2d",
                "2d-projection",
                "plot_2d_heatmap",
            }:
                return PLOT_VIEW_2D_HEATMAP
            return PLOT_VIEW_1D_LINE

        def _position_active_view_type_id(self) -> str:
            if self._analysis_name != "position":
                return PLOT_VIEW_1D_LINE
            return self._normalize_position_view_type_id(self._position_view_type_id())

        def _clean_position_view_state(self, state: dict[str, Any]) -> dict[str, Any]:
            cleaned = deepcopy(state)
            for key in (
                "_profile_filter_options",
                "data_contract",
                "source_selection",
                "style",
                "position_view_states",
            ):
                cleaned.pop(key, None)
            return cleaned

        def _default_position_view_state(self, view_type: str) -> dict[str, Any]:
            defaults = self._clean_position_view_state(deepcopy(self._default_profile_settings))
            normalized = self._normalize_position_view_type_id(view_type)
            defaults["_position_view_state_initialized"] = True
            defaults["position_active_view_type"] = normalized
            defaults.update(
                {
                    "x_lim": None,
                    "y_lim": None,
                    "x_min": None,
                    "x_max": None,
                    "y_min": None,
                    "y_max": None,
                    "annotations": None,
                    "_gui_sync_modes": None,
                }
            )
            if normalized == PLOT_VIEW_2D_HEATMAP:
                defaults.update(
                    {
                        "component": "2d-projection",
                        "projection_x": "x",
                        "projection_y": "y",
                        "projection_value": "distance",
                        "projection_render_mode": "color-scale",
                        "projection_filter_min": None,
                        "projection_filter_max": None,
                        "legend": False,
                        "markers": False,
                    }
                )
                defaults["view_mapping"] = serialize_plot_view_mapping(
                    position_plot_options_to_view_mapping(
                        component="2d-projection",
                        projection_x="x",
                        projection_y="y",
                        projection_value="distance",
                        projection_render_mode="color-scale",
                    )
                )
            else:
                defaults.update(
                    {
                        "component": "distance",
                        "time_axis": "ps",
                        "projection_filter_min": None,
                        "projection_filter_max": None,
                    }
                )
                defaults["view_mapping"] = serialize_plot_view_mapping(
                    position_plot_options_to_view_mapping(component="distance", time_axis="ps")
                )
            return defaults

        def _snapshot_position_view_state(self, view_type: str) -> dict[str, Any]:
            normalized = self._normalize_position_view_type_id(view_type)
            if self._analysis_name != "position" or not hasattr(self, "position_view_type"):
                return {}
            try:
                snapshot = self._collect_settings()
            except Exception:
                snapshot = {}
            if not isinstance(snapshot, dict):
                snapshot = {}
            snapshot = self._clean_position_view_state(snapshot)
            snapshot["_position_view_state_initialized"] = True
            snapshot["position_active_view_type"] = normalized
            return snapshot

        def _active_position_settings_with_view_state(
            self,
            settings: dict[str, Any],
            view_type: str,
            state: dict[str, Any],
        ) -> dict[str, Any]:
            merged = dict(settings)
            merged.update(deepcopy(state))
            normalized = self._normalize_position_view_type_id(view_type)
            merged["position_active_view_type"] = normalized
            return merged

        def _apply_position_view_state(self, view_type: str, state: dict[str, Any]) -> None:
            if self._analysis_name != "position":
                return
            normalized = self._normalize_position_view_type_id(view_type)
            restore_state = (
                deepcopy(state)
                if isinstance(state, dict) and state.get("_position_view_state_initialized")
                else self._default_position_view_state(normalized)
            )
            restore_settings = self._active_position_settings_with_view_state(
                self._collect_settings() if hasattr(self, "position_view_type") else {},
                normalized,
                restore_state,
            )
            previous_switching = self._position_view_state_switching
            previous_suspend = self._suspend_preview_events
            self._position_view_state_switching = True
            self._suspend_preview_events = True
            try:
                self._populate(restore_settings)
                self._position_active_view_type = normalized
            finally:
                self._suspend_preview_events = previous_suspend
                self._position_view_state_switching = previous_switching

        def _merge_position_view_state_into_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
            if self._analysis_name != "position":
                return settings
            active = self._position_active_view_type_id()
            states = deepcopy(self._position_view_states)
            active_state = self._clean_position_view_state(deepcopy(settings))
            active_state["_position_view_state_initialized"] = True
            active_state["position_active_view_type"] = active
            states[active] = active_state
            merged = dict(settings)
            merged["position_active_view_type"] = active
            merged["position_view_states"] = states
            return merged

        def _normalize_orientation_view_type_id(self, value: Any) -> str:
            token = str(value or "").strip().lower()
            if token in {"2d", "2d heatmap", "heatmap", "heatmap_2d", "plot_2d_heatmap"}:
                return PLOT_VIEW_2D_HEATMAP
            return PLOT_VIEW_1D_LINE

        def _orientation_active_view_type_id(self) -> str:
            if self._analysis_name != "orientation":
                return PLOT_VIEW_1D_LINE
            if hasattr(self, "orientation_view_type"):
                return self._normalize_orientation_view_type_id(
                    _ORIENTATION_VIEW_TYPE_ID_BY_LABEL.get(
                        self.orientation_view_type.currentText().strip(),
                        PLOT_VIEW_1D_LINE,
                    )
                )
            return self._orientation_active_view_type

        def _clean_orientation_view_state(self, state: dict[str, Any]) -> dict[str, Any]:
            cleaned = deepcopy(state)
            for key in (
                "_profile_filter_options",
                "data_contract",
                "source_selection",
                "style",
                "orientation_view_states",
                "heatmap_normalize",
                "heatmap_normalization_mode",
            ):
                cleaned.pop(key, None)
            return cleaned

        def _default_orientation_view_state(self, view_type: str) -> dict[str, Any]:
            defaults = self._clean_orientation_view_state(deepcopy(self._default_profile_settings))
            normalized = self._normalize_orientation_view_type_id(view_type)
            defaults["_orientation_view_state_initialized"] = True
            defaults["orientation_active_view_type"] = normalized
            defaults.update(
                {
                    "x_lim": None,
                    "y_lim": None,
                    "x_min": None,
                    "x_max": None,
                    "y_min": None,
                    "y_max": None,
                    "annotations": None,
                    "_gui_sync_modes": None,
                }
            )
            if normalized == PLOT_VIEW_2D_HEATMAP:
                defaults.update(
                    {
                        "component": "heatmap",
                        "orientation_line_x_axis": "distance",
                        "orientation_heatmap_x_axis": None,
                        "orientation_heatmap_y_axis": None,
                        "orientation_filter_x_min": None,
                        "orientation_filter_x_max": None,
                        "orientation_filter_y_min": None,
                        "orientation_filter_y_max": None,
                        "orientation_filter_z_min": None,
                        "orientation_filter_z_max": None,
                        "orientation_filter_distance_min": None,
                        "orientation_filter_distance_max": None,
                        "x_bin_width": None,
                        "y_bin_width": None,
                        "x_bin_reducer": None,
                        "y_bin_reducer": None,
                        "heatmap_value_mode": "raw_counts",
                        "heatmap_bulk_reference_mode": "auto",
                        "heatmap_bulk_min": None,
                        "heatmap_bulk_max": None,
                        "legend": False,
                        "markers": False,
                    }
                )
                defaults["view_mapping"] = serialize_plot_view_mapping(
                    orientation_plot_options_to_view_mapping(
                        component="heatmap",
                        angle=str(defaults.get("angle") or "polar"),
                    )
                )
            else:
                defaults.update(
                    {
                        "component": "average",
                        "orientation_line_x_axis": "distance",
                        "orientation_filter_x_min": None,
                        "orientation_filter_x_max": None,
                        "orientation_filter_y_min": None,
                        "orientation_filter_y_max": None,
                        "orientation_filter_z_min": None,
                        "orientation_filter_z_max": None,
                        "orientation_filter_distance_min": None,
                        "orientation_filter_distance_max": None,
                        "x_bin_width": None,
                        "y_bin_width": None,
                        "y_bin_reducer": None,
                    }
                )
                defaults["view_mapping"] = serialize_plot_view_mapping(
                    orientation_plot_options_to_view_mapping(
                        component=str(defaults.get("component") or "average"),
                        angle=str(defaults.get("angle") or "polar"),
                        line_x_axis=str(defaults.get("orientation_line_x_axis") or "distance"),
                    )
                )
            return defaults

        def _snapshot_orientation_view_state(self, view_type: str) -> dict[str, Any]:
            normalized = self._normalize_orientation_view_type_id(view_type)
            if self._analysis_name != "orientation" or not hasattr(self, "orientation_view_type"):
                return {}
            try:
                snapshot = self._collect_settings()
            except Exception:
                snapshot = {}
            if not isinstance(snapshot, dict):
                snapshot = {}
            snapshot = deepcopy(snapshot)
            snapshot = self._clean_orientation_view_state(snapshot)
            snapshot["_orientation_view_state_initialized"] = True
            snapshot["orientation_active_view_type"] = normalized
            return snapshot

        def _active_orientation_settings_with_view_state(
            self,
            settings: dict[str, Any],
            view_type: str,
            state: dict[str, Any],
        ) -> dict[str, Any]:
            merged = dict(settings)
            merged.update(deepcopy(state))
            normalized = self._normalize_orientation_view_type_id(view_type)
            merged["orientation_active_view_type"] = normalized
            if normalized == PLOT_VIEW_2D_HEATMAP:
                merged["view_mapping"] = serialize_plot_view_mapping(
                    orientation_plot_options_to_view_mapping(
                        component="heatmap",
                        angle=str(merged.get("angle") or "polar"),
                    )
                )
            else:
                component = str(merged.get("component") or "average")
                if component == "heatmap":
                    component = "average"
                merged["view_mapping"] = serialize_plot_view_mapping(
                    orientation_plot_options_to_view_mapping(
                        component=component,
                        angle=str(merged.get("angle") or "polar"),
                        line_x_axis=str(merged.get("orientation_line_x_axis") or "distance"),
                    )
                )
            return merged

        def _apply_orientation_view_state(self, view_type: str, state: dict[str, Any]) -> None:
            if self._analysis_name != "orientation":
                return
            normalized = self._normalize_orientation_view_type_id(view_type)
            restore_state = (
                deepcopy(state)
                if isinstance(state, dict) and state.get("_orientation_view_state_initialized")
                else self._default_orientation_view_state(normalized)
            )
            restore_settings = self._active_orientation_settings_with_view_state(
                self._collect_settings() if hasattr(self, "orientation_view_type") else {},
                normalized,
                restore_state,
            )
            previous_switching = self._orientation_view_state_switching
            previous_suspend = self._suspend_preview_events
            self._orientation_view_state_switching = True
            self._suspend_preview_events = True
            try:
                self._populate(restore_settings)
                self._orientation_active_view_type = normalized
            finally:
                self._suspend_preview_events = previous_suspend
                self._orientation_view_state_switching = previous_switching

        def _merge_orientation_view_state_into_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
            if self._analysis_name != "orientation":
                return settings
            active = self._orientation_active_view_type_id()
            states = deepcopy(self._orientation_view_states)
            active_state = self._clean_orientation_view_state(deepcopy(settings))
            active_state["_orientation_view_state_initialized"] = True
            active_state["orientation_active_view_type"] = active
            states[active] = active_state
            merged = dict(settings)
            merged["orientation_active_view_type"] = active
            merged["orientation_view_states"] = states
            return merged

        def _sync_density_species_selection_for_view_type(
            self,
            *,
            record_snapshot: bool = True,
        ) -> None:
            if self._analysis_name != "density":
                return
            checkboxes = getattr(self, "_density_species_checkboxes", {})
            if not isinstance(checkboxes, dict) or not checkboxes:
                return
            view_type_id = self._density_current_view_type_id()
            previous_view_type_id = self._density_previous_view_type_id
            self._density_previous_view_type_id = view_type_id
            if canonical_plot_view_id(view_type_id) == PLOT_VIEW_2D_HEATMAP:
                if previous_view_type_id != view_type_id and record_snapshot:
                    self._density_1d_enabled_species_snapshot = self._enabled_density_species()
                active_value = next(
                    (
                        str(value)
                        for value, checkbox in checkboxes.items()
                        if checkbox.isChecked()
                    ),
                    next(iter(checkboxes)),
                )
                self._density_species_checkbox_syncing = True
                try:
                    for value, checkbox in checkboxes.items():
                        checkbox.blockSignals(True)
                        try:
                            checkbox.setChecked(str(value) == active_value)
                        finally:
                            checkbox.blockSignals(False)
                finally:
                    self._density_species_checkbox_syncing = False
                return
            if (
                canonical_plot_view_id(previous_view_type_id) == PLOT_VIEW_2D_HEATMAP
                and self._density_1d_enabled_species_snapshot is not None
            ):
                enabled = set(self._density_1d_enabled_species_snapshot)
                self._density_species_checkbox_syncing = True
                try:
                    for value, checkbox in checkboxes.items():
                        checkbox.blockSignals(True)
                        try:
                            checkbox.setChecked(str(value) in enabled)
                        finally:
                            checkbox.blockSignals(False)
                finally:
                    self._density_species_checkbox_syncing = False

        def _coordination_contract(self) -> PlotDataContract:
            contract = self._coordination_data_contract
            if isinstance(contract, PlotDataContract):
                return contract
            return _fallback_coordination_plot_data_contract()

        def _coordination_supported_view_type_ids(self) -> list[str]:
            contract = self._coordination_contract()
            ordered = [PLOT_VIEW_1D_LINE]
            if _contract_has_public_heatmap_view(contract):
                ordered.append(PLOT_VIEW_2D_HEATMAP)
            return ordered

        def _orientation_line_contract(self) -> PlotDataContract:
            contract = self._orientation_line_data_contract
            if isinstance(contract, PlotDataContract):
                return contract
            return _fallback_orientation_line_plot_data_contract()

        def _orientation_heatmap_contract(self) -> PlotDataContract:
            contract = self._orientation_heatmap_data_contract
            if isinstance(contract, PlotDataContract):
                return contract
            return _fallback_orientation_heatmap_plot_data_contract()

        def _potential_contract(self) -> PlotDataContract:
            contract = self._potential_data_contract
            if isinstance(contract, PlotDataContract):
                return contract
            return _fallback_potential_plot_data_contract()

        def _current_density_mapping(self) -> PlotViewMapping:
            view_type = (
                _DENSITY_VIEW_TYPE_ID_BY_LABEL.get(
                    self.density_view_type.currentText().strip(),
                    PLOT_VIEW_1D_LINE,
                )
                if hasattr(self, "density_view_type")
                else PLOT_VIEW_1D_LINE
            )
            x_mode = self._selected_density_x_mode() if hasattr(self, "density_x_mode") else "distance"
            quantity = (
                self.density_quantity.currentText().strip() or "mass"
                if hasattr(self, "density_quantity")
                else "mass"
            )
            return density_plot_options_to_view_mapping(
                view_type=view_type,
                x_mode=x_mode,
                quantity=quantity,
            )

        def _current_coordination_mapping(self) -> PlotViewMapping:
            view_type_label = (
                self.coordination_component.currentText().strip()
                if hasattr(self, "coordination_component")
                else ""
            )
            view_type_id = _COORDINATION_VIEW_TYPE_ID_BY_LABEL.get(
                view_type_label,
                PLOT_VIEW_1D_LINE,
            )
            line_x_label = (
                self.coordination_line_x_quantity.currentText().strip()
                if hasattr(self, "coordination_line_x_quantity")
                else ""
            )
            line_x_quantity = _COORDINATION_LINE_X_QUANTITY_BACKEND_BY_LABEL.get(
                line_x_label,
                line_x_label or "distance",
            )
            component = (
                "time-distance"
                if canonical_plot_view_id(view_type_id) == PLOT_VIEW_2D_HEATMAP
                else line_x_quantity
            )
            time_x_role = (
                self.coordination_time_axis.currentText().strip() or "ps"
                if hasattr(self, "coordination_time_axis")
                else "ps"
            )
            return coordination_plot_options_to_view_mapping(
                component=component,
                time_axis=time_x_role,
            )

        def _current_orientation_mapping(self) -> PlotViewMapping:
            view_type = (
                _ORIENTATION_VIEW_TYPE_ID_BY_LABEL.get(
                    self.orientation_view_type.currentText().strip(),
                    PLOT_VIEW_1D_LINE,
                )
                if hasattr(self, "orientation_view_type")
                else PLOT_VIEW_1D_LINE
            )
            if canonical_plot_view_id(view_type) == PLOT_VIEW_2D_HEATMAP:
                component = "heatmap"
            else:
                line_quantity_label = (
                    self.orientation_component.currentText().strip()
                    if hasattr(self, "orientation_component")
                    else ""
                )
                component = _ORIENTATION_LINE_QUANTITY_BACKEND_BY_LABEL.get(
                    line_quantity_label,
                    line_quantity_label or "average",
                )
            angle_role = (
                self.orientation_angle.currentText().strip() or "polar"
                if hasattr(self, "orientation_angle")
                else "polar"
            )
            return orientation_plot_options_to_view_mapping(
                component=component,
                angle=angle_role,
                line_x_axis=(
                    _DENSITY_X_MODE_BY_LABEL.get(
                        self.orientation_line_x_axis.currentText().strip().lower(),
                        "distance",
                    )
                    if hasattr(self, "orientation_line_x_axis")
                    else "distance"
                ),
            )

        def _set_orientation_view_type_from_component(self, component: str) -> None:
            if not hasattr(self, "orientation_view_type"):
                return
            view_type_id = (
                PLOT_VIEW_2D_HEATMAP
                if str(component).strip().lower() == "heatmap"
                else PLOT_VIEW_1D_LINE
            )
            self._set_combo_value(
                self.orientation_view_type,
                _ORIENTATION_VIEW_TYPE_LABEL_BY_ID.get(
                    view_type_id,
                    _ORIENTATION_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_1D_LINE],
                ),
            )

        def _set_orientation_line_quantity(self, component: str) -> None:
            if not hasattr(self, "orientation_component"):
                return
            backend = str(component or "average").strip().lower()
            if backend == "heatmap":
                backend = "average"
            self._set_combo_value(
                self.orientation_component,
                _ORIENTATION_LINE_QUANTITY_LABEL_BY_BACKEND.get(
                    backend,
                    _ORIENTATION_LINE_QUANTITY_LABEL_BY_BACKEND["average"],
                ),
            )

        def _current_potential_mapping(self) -> PlotViewMapping:
            if hasattr(self, "potential_view_type"):
                view_type_id = _POTENTIAL_VIEW_TYPE_ID_BY_LABEL.get(
                    self.potential_view_type.currentText().strip(),
                    "line_1d",
                )
                series_token = _POTENTIAL_SERIES_ID_BY_LABEL.get(
                    self.potential_series_mode.currentText().strip(),
                    "summary",
                ) if hasattr(self, "potential_series_mode") else "summary"
                y_quantity = None if series_token == "summary" else series_token
                return potential_plot_options_to_view_mapping(y_quantity=y_quantity)
            return potential_plot_options_to_view_mapping()

        def _update_density_contract_summary(self) -> None:
            if self._analysis_name != "density":
                return
            contract = self._density_contract()
            if hasattr(self, "_density_source_contract_label"):
                self._density_source_contract_label.setText(
                    str(contract.label or contract.source_id or "Density data")
                )
            if hasattr(self, "_density_source_dimensions_label"):
                self._density_source_dimensions_label.setText(_contract_dimensions_text(contract))
            if hasattr(self, "_density_source_quantities_label"):
                self._density_source_quantities_label.setText(_contract_quantities_text(contract))
            if not hasattr(self, "_density_mapping_status_label"):
                return
            mapping = self._current_density_mapping()
            compatibility = generic_view_type_compatibility(contract, mapping)
            self._density_mapping_status_label.setText(_mapping_status_label(compatibility))
            self._density_mapping_summary_label.setText(_mapping_summary_text(mapping))
            try:
                resolved = resolve_density_plot_mapping(
                    contract=contract,
                    mapping=mapping,
                )
                self._density_backend_summary_label.setText(
                    _density_backend_summary_text(
                        view_type_id=resolved.view_type_id,
                        x_mode=resolved.x_mode,
                        quantity=resolved.quantity,
                    )
                )
            except ValueError as exc:
                self._density_backend_summary_label.setText(str(exc))

        def _update_coordination_contract_summary(self) -> None:
            if self._analysis_name != "coordination":
                return
            contract = self._coordination_contract()
            if hasattr(self, "_coordination_source_contract_label"):
                self._coordination_source_contract_label.setText(
                    str(contract.label or contract.source_id or "Coordination data")
                )
            if hasattr(self, "_coordination_source_dimensions_label"):
                self._coordination_source_dimensions_label.setText(_contract_dimensions_text(contract))
            if hasattr(self, "_coordination_source_quantities_label"):
                self._coordination_source_quantities_label.setText(_contract_quantities_text(contract))
            if not hasattr(self, "_coordination_mapping_status_label"):
                return
            mapping = self._current_coordination_mapping()
            compatibility = generic_view_type_compatibility(contract, mapping)
            self._coordination_mapping_status_label.setText(_mapping_status_label(compatibility))
            self._coordination_mapping_summary_label.setText(_mapping_summary_text(mapping))
            try:
                resolved = resolve_coordination_plot_mapping(
                    contract=contract,
                    mapping=mapping,
                )
                self._coordination_backend_summary_label.setText(
                    _coordination_backend_summary_text(
                        component=resolved.component,
                        time_axis=resolved.time_axis,
                    )
                )
            except ValueError as exc:
                self._coordination_backend_summary_label.setText(str(exc))

        def _active_orientation_contract(self) -> PlotDataContract:
            mapping = self._current_orientation_mapping()
            if canonical_plot_view_id(mapping.view_type_id) == PLOT_VIEW_2D_HEATMAP:
                return self._orientation_heatmap_contract()
            return self._orientation_line_contract()

        def _update_orientation_contract_summary(self) -> None:
            if self._analysis_name != "orientation":
                return
            contract = self._active_orientation_contract()
            if hasattr(self, "_orientation_source_contract_label"):
                self._orientation_source_contract_label.setText(
                    str(contract.label or contract.source_id or "Orientation data")
                )
            if hasattr(self, "_orientation_source_dimensions_label"):
                self._orientation_source_dimensions_label.setText(_contract_dimensions_text(contract))
            if hasattr(self, "_orientation_source_quantities_label"):
                self._orientation_source_quantities_label.setText(_contract_quantities_text(contract))
            if not hasattr(self, "_orientation_mapping_status_label"):
                return
            mapping = self._current_orientation_mapping()
            compatibility = generic_view_type_compatibility(contract, mapping)
            self._orientation_mapping_status_label.setText(_mapping_status_label(compatibility))
            self._orientation_mapping_summary_label.setText(_mapping_summary_text(mapping))
            try:
                resolved = resolve_orientation_plot_mapping(
                    contract=contract,
                    mapping=mapping,
                )
                self._orientation_backend_summary_label.setText(
                    _orientation_backend_summary_text(
                        component=resolved.component,
                        angle=resolved.angle,
                        is_heatmap=resolved.is_heatmap,
                    )
                )
            except ValueError as exc:
                self._orientation_backend_summary_label.setText(str(exc))

        def _update_potential_contract_summary(self) -> None:
            if self._analysis_name != "potential":
                return
            contract = self._potential_contract()
            if hasattr(self, "_potential_source_contract_label"):
                self._potential_source_contract_label.setText(
                    str(contract.label or contract.source_id or "Potential data")
                )
            if hasattr(self, "_potential_source_dimensions_label"):
                self._potential_source_dimensions_label.setText(_contract_dimensions_text(contract))
            if hasattr(self, "_potential_source_quantities_label"):
                self._potential_source_quantities_label.setText(_contract_quantities_text(contract))
            if not hasattr(self, "_potential_mapping_status_label"):
                return
            mapping = self._current_potential_mapping()
            compatibility = generic_view_type_compatibility(contract, mapping)
            self._potential_mapping_status_label.setText(_mapping_status_label(compatibility))
            self._potential_mapping_summary_label.setText(_mapping_summary_text(mapping))
            try:
                resolved = resolve_potential_plot_mapping(
                    contract=contract,
                    mapping=mapping,
                )
                self._potential_backend_summary_label.setText(
                    _potential_backend_summary_text(
                        view_type=resolved.view_type,
                        y_quantity=resolved.y_quantity,
                        standard_plot=resolved.standard_plot,
                    )
                )
            except ValueError as exc:
                self._potential_backend_summary_label.setText(str(exc))

        def _handle_density_mapping_change(self, *_unused: object) -> None:
            if (
                self._analysis_name == "density"
                and not self._density_view_state_switching
                and hasattr(self, "density_view_type")
                and self.sender() is self.density_view_type
            ):
                previous_view = self._density_active_view_type
                next_view = self._density_current_view_type_id()
                if previous_view != next_view:
                    self._density_view_states[previous_view] = (
                        self._snapshot_density_view_state(previous_view)
                    )
                    target_state = self._density_view_states.get(next_view)
                    if not isinstance(target_state, dict):
                        target_state = self._default_density_view_state(next_view)
                    self._apply_density_view_state(next_view, target_state)
            self._sync_density_species_selection_for_view_type()
            self._apply_density_default_bin_width_texts()
            self._update_density_contract_summary()
            self._handle_series_identity_change()

        def _handle_density_range_change(self, *_unused: object) -> None:
            if self._analysis_name != "density":
                return
            self._update_density_contract_summary()
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _handle_density_binning_change(self, *_unused: object) -> None:
            if self._analysis_name != "density":
                return
            self._handle_series_identity_change()

        def _handle_density_target_filter_changed(self, *_unused: object) -> None:
            if self._analysis_name != "density" or self._density_target_filter is None:
                return
            selected = self._density_target_filter.currentText().strip()
            show_all = not selected or selected == "All targets"
            for index, descriptor in enumerate(self._series_descriptors_data):
                if index >= len(self._series_enabled_data):
                    continue
                if str(descriptor.get("source_kind") or "source").strip().lower() == "group":
                    continue
                target = self._density_target_for_descriptor(descriptor)
                self._series_enabled_data[index] = show_all or target == selected
            self._refresh_series_list_widgets()
            self._refresh_widget_states()
            self._record_history_after_non_text_change()
            self._schedule_preview_update()

        def _handle_density_species_checkbox_changed(self, *_unused: object) -> None:
            if self._analysis_name != "density":
                return
            if self._density_species_checkbox_syncing:
                return
            if self._is_density_heatmap_mode():
                checkboxes = getattr(self, "_density_species_checkboxes", {})
                sender = self.sender()
                if isinstance(checkboxes, dict) and checkboxes:
                    active_value = next(
                        (
                            str(value)
                            for value, checkbox in checkboxes.items()
                            if checkbox is sender and checkbox.isChecked()
                        ),
                        None,
                    )
                    if active_value is None:
                        active_value = next(
                            (
                                str(value)
                                for value, checkbox in checkboxes.items()
                                if checkbox.isChecked()
                            ),
                            next(iter(checkboxes)),
                        )
                    self._density_species_checkbox_syncing = True
                    try:
                        for value, checkbox in checkboxes.items():
                            checkbox.blockSignals(True)
                            try:
                                checkbox.setChecked(str(value) == active_value)
                            finally:
                                checkbox.blockSignals(False)
                    finally:
                        self._density_species_checkbox_syncing = False
            self._record_history_after_non_text_change()
            self._handle_series_identity_change()

        def _handle_position_species_checkbox_changed(self, *_unused: object) -> None:
            if self._analysis_name != "position":
                return
            self._record_history_after_non_text_change()
            self._handle_series_identity_change()

        def _handle_coordination_mapping_change(self, *_unused: object) -> None:
            self._update_coordination_contract_summary()
            self._handle_series_identity_change()

        def _handle_orientation_mapping_change(self, *_unused: object) -> None:
            if self._analysis_name == "orientation" and not self._orientation_view_state_switching:
                previous_view = self._orientation_active_view_type
                next_view = self._orientation_active_view_type_id()
                if previous_view != next_view:
                    self._orientation_view_states[previous_view] = (
                        self._snapshot_orientation_view_state(previous_view)
                    )
                    target_state = self._orientation_view_states.get(next_view)
                    if target_state is None:
                        target_state = self._default_orientation_view_state(next_view)
                    self._apply_orientation_view_state(next_view, target_state)
                    self._handle_series_identity_change()
                    return
            self._update_orientation_contract_summary()
            self._handle_series_identity_change()

        def _handle_potential_mapping_change(self, *_unused: object) -> None:
            self._update_potential_contract_summary()
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _position_contract(self) -> PlotDataContract:
            contract = self._position_data_contract
            if isinstance(contract, PlotDataContract):
                return contract
            return _fallback_position_plot_data_contract()

        def _position_supported_view_type_ids(self) -> list[str]:
            contract = self._position_contract()
            ordered = [PLOT_VIEW_1D_LINE]
            if _contract_has_public_heatmap_view(contract):
                ordered.append(PLOT_VIEW_2D_HEATMAP)
            return ordered

        def _position_mapping_candidate_quantity_ids(
            self,
            *,
            view_type_id: str,
            role: str,
        ) -> list[str]:
            canonical_view_type_id = canonical_plot_view_id(view_type_id)
            validation_view_type_id = (
                PLOT_VIEW_2D_HEATMAP
                if canonical_view_type_id == PLOT_VIEW_2D_HEATMAP
                else PLOT_VIEW_1D_LINE
            )
            if canonical_view_type_id == PLOT_VIEW_1D_LINE:
                backend_supported = (
                    {"time_ps", "time_fs", "step", "frame_index"}
                    if role == "x"
                    else {"distance_to_surface", "x", "y", "z"}
                    if role == "y"
                    else set()
                )
            elif canonical_view_type_id == PLOT_VIEW_2D_HEATMAP:
                backend_supported = set(_POSITION_GUI_TOKEN_BY_QUANTITY_ID)
            else:
                backend_supported = set()
            candidates: list[str] = []
            contract = self._position_contract()
            for quantity in contract.quantities:
                quantity_id = str(quantity.id).strip()
                if quantity_id not in backend_supported:
                    continue
                status = visual_role_compatibility(
                    contract,
                    view_type_id=validation_view_type_id,
                    role=role,
                    quantity_id=quantity_id,
                )
                if status == "invalid":
                    continue
                candidates.append(quantity_id)
            return candidates

        def _position_view_type_id(self) -> str:
            if hasattr(self, "position_view_type"):
                return _POSITION_GUI_VIEW_TYPE_ID_BY_LABEL.get(
                    self.position_view_type.currentText().strip(),
                    PLOT_VIEW_1D_LINE,
                )
            return PLOT_VIEW_1D_LINE

        def _current_position_mapping(
            self,
            *,
            strict: bool = False,
        ) -> PlotViewMapping:
            def _parse_optional_bound(text: str, *, field_name: str) -> float | None:
                if strict:
                    return _optional_float(text, field_name=field_name)
                stripped = str(text).strip()
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError:
                    return None

            if hasattr(self, "position_view_type"):
                view_type_id = self._position_view_type_id()
                if canonical_plot_view_id(view_type_id) == PLOT_VIEW_1D_LINE:
                    x_token = self.position_mapping_x.currentText().strip() or "ps"
                    y_token = self.position_mapping_y.currentText().strip() or "distance"
                    return PlotViewMapping(
                        view_type_id=PLOT_VIEW_1D_LINE,
                        x=_position_quantity_id_from_token(x_token),
                        y=_position_quantity_id_from_token(y_token),
                        split_by="atom",
                    )
                value_token = self.position_mapping_value.currentText().strip() or "distance"
                filter_min = _parse_optional_bound(
                    self.position_mapping_filter_min.text(),
                    field_name="2D Heatmap range minimum",
                )
                filter_max = _parse_optional_bound(
                    self.position_mapping_filter_max.text(),
                    field_name="2D Heatmap range maximum",
                )
                render_mode_label = (
                    self.position_mapping_render_mode.currentText().strip() or "color-scale"
                )
                render_mode = _POSITION_RENDER_MODE_BACKEND_BY_LABEL.get(
                    render_mode_label,
                    render_mode_label,
                )
                value_id = _position_quantity_id_from_token(value_token)
                return PlotViewMapping(
                    view_type_id=PLOT_VIEW_2D_HEATMAP,
                    x=_position_quantity_id_from_token(
                        self.position_mapping_x.currentText().strip() or "x"
                    ),
                    y=_position_quantity_id_from_token(
                        self.position_mapping_y.currentText().strip() or "y"
                    ),
                    color=value_id if render_mode == "color-scale" else None,
                    split_by="atom",
                    filter_by=value_id if filter_min is not None or filter_max is not None else None,
                    filter_min=filter_min,
                    filter_max=filter_max,
                    fixed_values={"projection_render_mode": render_mode},
                )
            return position_plot_options_to_view_mapping(component="distance", time_axis="ps")

        def _current_position_is_projection_view(self) -> bool:
            return canonical_plot_view_id(self._position_view_type_id()) == PLOT_VIEW_2D_HEATMAP

        def _current_position_uses_continuous_color(self) -> bool:
            if not self._current_position_is_projection_view():
                return False
            mapping = self._current_position_mapping()
            render_mode = str(
                mapping.fixed_values.get("projection_render_mode")
                or ("color-scale" if mapping.color is not None else "line-colors")
            ).strip()
            return render_mode == "color-scale"

        def _set_position_mapping_combo_items(
            self,
            mapping: PlotViewMapping | None = None,
        ) -> None:
            if not hasattr(self, "position_mapping_x"):
                return
            active_mapping = mapping or self._current_position_mapping()
            view_type_id = str(active_mapping.view_type_id).strip() or PLOT_VIEW_1D_LINE
            self._set_combo_items(
                self.position_mapping_x,
                [
                    _position_quantity_token(quantity_id)
                    for quantity_id in self._position_mapping_candidate_quantity_ids(
                        view_type_id=view_type_id,
                        role="x",
                    )
                ],
                preferred_value=_position_quantity_token(active_mapping.x or "time_ps"),
            )
            self._set_combo_items(
                self.position_mapping_y,
                [
                    _position_quantity_token(quantity_id)
                    for quantity_id in self._position_mapping_candidate_quantity_ids(
                        view_type_id=view_type_id,
                        role="y",
                    )
                ],
                preferred_value=_position_quantity_token(active_mapping.y or "distance_to_surface"),
            )
            if hasattr(self, "position_mapping_value"):
                value_quantity = active_mapping.color or active_mapping.filter_by or "distance_to_surface"
                self._set_combo_items(
                    self.position_mapping_value,
                    [
                        _position_quantity_token(quantity_id)
                        for quantity_id in self._position_mapping_candidate_quantity_ids(
                            view_type_id=view_type_id,
                            role="color",
                        )
                    ],
                    preferred_value=_position_quantity_token(value_quantity),
                )

        def _set_position_preset_label(self, label: str) -> None:
            if not hasattr(self, "position_mapping_preset"):
                return
            self.position_mapping_preset.blockSignals(True)
            try:
                self._set_combo_value(self.position_mapping_preset, label)
            finally:
                self.position_mapping_preset.blockSignals(False)

        def _apply_position_mapping_controls(self, mapping: PlotViewMapping) -> None:
            if not hasattr(self, "position_view_type"):
                return
            self._set_combo_value(
                self.position_view_type,
                _POSITION_GUI_VIEW_TYPE_LABEL_BY_ID.get(
                    mapping.view_type_id,
                    plot_view_display_label(PLOT_VIEW_1D_LINE),
                ),
            )
            self._set_position_mapping_combo_items(mapping)
            if hasattr(self, "position_mapping_render_mode"):
                self._set_combo_value(
                    self.position_mapping_render_mode,
                    _POSITION_RENDER_MODE_LABEL_BY_BACKEND.get(
                        str(mapping.fixed_values.get("projection_render_mode") or "color-scale"),
                        str(mapping.fixed_values.get("projection_render_mode") or "color-scale"),
                    ),
                )
            if hasattr(self, "position_mapping_filter_min"):
                self.position_mapping_filter_min.setText(
                    "" if mapping.filter_min is None else str(mapping.filter_min)
                )
            if hasattr(self, "position_mapping_filter_max"):
                self.position_mapping_filter_max.setText(
                    "" if mapping.filter_max is None else str(mapping.filter_max)
                )
            self._update_position_contract_summary()

        def _handle_position_mapping_preset(self, *_unused: object) -> None:
            preset_id = _POSITION_GUI_PRESET_ID_BY_LABEL.get(
                self.position_mapping_preset.currentText().strip()
            )
            if preset_id is None:
                return
            self._apply_position_mapping_controls(position_mapping_preset(preset_id))
            self._handle_series_identity_change()

        def _handle_position_mapping_view_change(self, *_unused: object) -> None:
            if self._analysis_name == "position" and not self._position_view_state_switching:
                previous_view = self._position_active_view_type
                next_view = self._position_view_type_id()
                if previous_view != next_view:
                    self._position_view_states[previous_view] = (
                        self._snapshot_position_view_state(previous_view)
                    )
                    target_state = self._position_view_states.get(next_view)
                    if target_state is None:
                        target_state = self._default_position_view_state(next_view)
                    self._apply_position_view_state(next_view, target_state)
                    self._handle_series_identity_change()
                    return
            self._set_position_preset_label("Custom")
            self._set_position_mapping_combo_items()
            self._update_position_contract_summary()
            self._handle_series_identity_change()

        def _handle_position_mapping_preview_change(self, *_unused: object) -> None:
            self._set_position_preset_label("Custom")
            self._update_position_contract_summary()
            self._schedule_preview_update()

        def _build_position_mapping_sections(self, layout: QVBoxLayout) -> None:
            mapping_group = QGroupBox("Mapping")
            mapping_form = QFormLayout(mapping_group)
            self._position_species_checkboxes = {}
            species_widget = QWidget(mapping_group)
            species_layout = QGridLayout(species_widget)
            species_layout.setContentsMargins(0, 0, 0, 0)
            species_layout.setSpacing(4)
            position_species_options = self._position_species_options()
            option_count = max(1, len(position_species_options))
            species_columns = 1
            if option_count >= 12:
                species_columns = 4
            elif option_count >= 7:
                species_columns = 3
            elif option_count >= 4:
                species_columns = 2
            for grid_index, (value, label) in enumerate(position_species_options):
                checkbox = QCheckBox(label)
                checkbox.setChecked(True)
                checkbox.setProperty("position_species", value)
                checkbox.stateChanged.connect(self._handle_position_species_checkbox_changed)
                species_layout.addWidget(
                    checkbox,
                    grid_index // species_columns,
                    grid_index % species_columns,
                )
                self._position_species_checkboxes[value] = checkbox
            if not self._position_species_checkboxes:
                species_layout.addWidget(QLabel("No species filters available"), 0, 0)
            self.position_mapping_preset = self._combo(
                ["Custom", *list(_POSITION_GUI_PRESET_LABEL_BY_ID.values())]
            )
            self.position_mapping_preset.currentTextChanged.connect(
                self._handle_position_mapping_preset
            )
            self.position_view_type = self._combo(
                [
                    _POSITION_GUI_VIEW_TYPE_LABEL_BY_ID[view_type_id]
                    for view_type_id in self._position_supported_view_type_ids()
                ]
            )
            self.position_view_type.currentTextChanged.connect(
                self._handle_position_mapping_view_change
            )
            self.position_mapping_x = self._combo(["ps", "fs", "step", "frame"])
            self.position_mapping_x.currentTextChanged.connect(
                self._handle_position_mapping_preview_change
            )
            self.position_mapping_y = self._combo(["distance", "x", "y", "z"])
            self.position_mapping_y.currentTextChanged.connect(
                self._handle_position_mapping_preview_change
            )
            self.position_mapping_render_mode = self._combo(_POSITION_PROJECTION_RENDER_MODES)
            self.position_mapping_render_mode.currentTextChanged.connect(
                lambda *_unused: self._set_position_preset_label("Custom")
            )
            self.position_mapping_render_mode.currentTextChanged.connect(
                lambda *_unused: self._update_position_contract_summary()
            )
            self.position_mapping_render_mode.currentTextChanged.connect(
                self._refresh_widget_states
            )
            self.position_mapping_render_mode.currentTextChanged.connect(
                self._handle_series_identity_change
            )
            self.position_mapping_value = self._combo(list(_POSITION_PROJECTION_QUANTITIES))
            self.position_mapping_value.currentTextChanged.connect(
                self._handle_position_mapping_preview_change
            )
            self.position_mapping_filter_min = self._line("")
            self.position_mapping_filter_min.textChanged.connect(
                self._handle_position_mapping_preview_change
            )
            self.position_mapping_filter_max = self._line("")
            self.position_mapping_filter_max.textChanged.connect(
                self._handle_position_mapping_preview_change
            )
            self.position_mapping_split_by = QLabel("atom")
            self._add_form_row(
                mapping_form,
                "Species",
                species_widget,
                tooltip_id="data.profile.species",
            )
            self._add_form_row(
                mapping_form,
                "Preset",
                self.position_mapping_preset,
                tooltip_id="data.position.mapping.preset",
            )
            self._add_form_row(
                mapping_form,
                "View type",
                self.position_view_type,
                tooltip_id="data.position.mapping.view_type",
            )
            self._add_form_row(
                mapping_form,
                "X-axis quantity",
                self.position_mapping_x,
                tooltip_id="data.position.mapping.x",
            )
            self._add_form_row(
                mapping_form,
                "Y-axis quantity",
                self.position_mapping_y,
                tooltip_id="data.position.mapping.y",
            )
            self._add_form_row(
                mapping_form,
                "Color mode",
                self.position_mapping_render_mode,
                tooltip_id="data.position.projection_render_mode",
            )
            self._add_form_row(
                mapping_form,
                "Value quantity",
                self.position_mapping_value,
                tooltip_id="data.position.mapping.value",
            )
            self._add_form_row(
                mapping_form,
                "Split by",
                self.position_mapping_split_by,
                tooltip_id="data.position.mapping.split_by",
            )
            self._add_form_row(
                mapping_form,
                "Value filter min",
                self.position_mapping_filter_min,
                tooltip_id="data.position.projection_range_min",
            )
            self._add_form_row(
                mapping_form,
                "Value filter max",
                self.position_mapping_filter_max,
                tooltip_id="data.position.projection_range_max",
            )
            self._position_mapping_x_row = (mapping_form, self.position_mapping_x)
            self._position_mapping_y_row = (mapping_form, self.position_mapping_y)
            self._position_mapping_render_mode_row = (
                mapping_form,
                self.position_mapping_render_mode,
            )
            self._position_mapping_value_row = (mapping_form, self.position_mapping_value)
            self._position_mapping_split_by_row = (mapping_form, self.position_mapping_split_by)
            self._position_mapping_filter_min_row = (
                mapping_form,
                self.position_mapping_filter_min,
            )
            self._position_mapping_filter_max_row = (
                mapping_form,
                self.position_mapping_filter_max,
            )
            layout.addWidget(
                self._make_collapsible_section(
                    title="Mapping",
                    section_id="data.position.mapping",
                    body_widget=mapping_group,
                )
            )

            previous_suspend = self._suspend_preview_events
            self._suspend_preview_events = True
            try:
                self._apply_position_mapping_controls(position_mapping_preset("distance_vs_time"))
                self._set_position_preset_label("Custom")
            finally:
                self._suspend_preview_events = previous_suspend

        def _update_position_contract_summary(self) -> None:
            if self._analysis_name != "position":
                return
            contract = self._position_contract()
            if hasattr(self, "_position_source_contract_label"):
                self._position_source_contract_label.setText(
                    str(contract.label or contract.source_id or "Position data")
                )
            if hasattr(self, "_position_source_dimensions_label"):
                dimension_tokens: list[str] = []
                for dimension in contract.dimensions:
                    token = str(dimension.id)
                    if dimension.length is not None:
                        token = f"{token}={dimension.length}"
                    dimension_tokens.append(token)
                self._position_source_dimensions_label.setText(", ".join(dimension_tokens))
            if hasattr(self, "_position_source_quantities_label"):
                self._position_source_quantities_label.setText(
                    ", ".join(
                        _position_quantity_token(quantity.id)
                        if quantity.id in _POSITION_GUI_TOKEN_BY_QUANTITY_ID
                        else str(quantity.id)
                        for quantity in contract.quantities
                    )
                )
            if not hasattr(self, "_position_mapping_status_label"):
                return
            mapping = self._current_position_mapping()
            compatibility = generic_view_type_compatibility(contract, mapping)
            self._position_mapping_status_label.setText(
                _mapping_status_label(compatibility)
            )
            view_label = plot_view_display_label(mapping.view_type_id)
            role_parts = [
                f"x-axis={_position_quantity_token(mapping.x)}",
                f"y-axis={_position_quantity_token(mapping.y)}",
            ]
            if mapping.color is not None:
                role_parts.append(f"color quantity={_position_quantity_token(mapping.color)}")
            if mapping.filter_by is not None:
                filter_text = f"range quantity={_position_quantity_token(mapping.filter_by)}"
                if mapping.filter_min is not None or mapping.filter_max is not None:
                    filter_text += (
                        f" [{'' if mapping.filter_min is None else mapping.filter_min}, "
                        f"{'' if mapping.filter_max is None else mapping.filter_max}]"
                    )
                role_parts.append(filter_text)
            role_parts.append(f"series={mapping.split_by or 'atom'}")
            self._position_mapping_summary_label.setText(
                f"{view_label}: " + ", ".join(role_parts)
            )
            try:
                legacy = position_view_mapping_to_plot_options(mapping)
                backend_parts = [
                    f"view={view_label}",
                ]
                if legacy.get("component") == "2d-projection":
                    render_mode = _POSITION_RENDER_MODE_LABEL_BY_BACKEND.get(
                        str(legacy.get("projection_render_mode") or "color-scale"),
                        str(legacy.get("projection_render_mode") or "color-scale"),
                    )
                    backend_parts.extend(
                        [
                            f"x-axis={legacy.get('projection_x')}",
                            f"y-axis={legacy.get('projection_y')}",
                            f"color quantity={legacy.get('projection_value')}",
                            f"rendering={render_mode}",
                        ]
                    )
                else:
                    backend_parts.append(f"x-axis time unit={legacy.get('time_axis')}")
                self._position_backend_summary_label.setText(", ".join(backend_parts))
            except ValueError as exc:
                self._position_backend_summary_label.setText(str(exc))

        def _fit_supported_for_current_view(self) -> bool:
            analysis = self._analysis_name
            if analysis in {"density", "msd", "rdf", "temperature"}:
                return True
            if analysis == "potential":
                return True
            if analysis == "position":
                return not self._current_position_is_projection_view()
            if analysis == "coordination":
                return (
                    canonical_plot_view_id(self._current_coordination_mapping().view_type_id)
                    == PLOT_VIEW_1D_LINE
                )
            if analysis == "orientation":
                return not self._is_orientation_heatmap_mode()
            return False

        def _error_supported_for_current_view(self) -> bool:
            analysis = str(self._analysis_name or "")
            if analysis == "potential":
                return True
            if analysis == "coordination":
                return (
                    canonical_plot_view_id(self._current_coordination_mapping().view_type_id)
                    == PLOT_VIEW_1D_LINE
                )
            if analysis == "orientation":
                return not self._is_orientation_heatmap_mode()
            if analysis == "position":
                return not self._current_position_is_projection_view()
            return _error_supported_for_view(
                analysis,
                orientation_heatmap=self._is_orientation_heatmap_mode(),
            )

        def _current_plot_family(self) -> str:
            analysis = str(self._analysis_name or "")
            if analysis == "coordination":
                mapping = self._current_coordination_mapping()
                if canonical_plot_view_id(mapping.view_type_id) == PLOT_VIEW_2D_HEATMAP:
                    return "heatmap"
            if analysis == "density" and self._is_density_heatmap_mode():
                return "heatmap"
            if analysis == "orientation" and self._is_orientation_heatmap_mode():
                return "heatmap"
            if analysis == "position" and self._current_position_is_projection_view():
                return (
                    "heatmap"
                    if self._current_position_uses_continuous_color()
                    else "line"
                )
            return _plot_family_for_view(
                analysis,
                orientation_heatmap=self._is_orientation_heatmap_mode(),
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
            is_heatmap_family = plot_family == "heatmap"
            active_layer_enabled = (
                0 <= self._series_active_index < len(self._series_enabled_data)
                and bool(self._series_enabled_data[self._series_active_index])
            )
            if (
                self._analysis_name == "position"
                and self._current_position_is_projection_view()
                and layer_kind == "source"
            ):
                categorical_color = not self._current_position_uses_continuous_color()
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=True,
                    show_style=categorical_color,
                    show_markers=False,
                    show_derived_lines=False,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_integration=False,
                    show_group_members=False,
                    show_metadata=True,
                    show_fit_editor=False,
                    show_cumulative_editor=False,
                )
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
                    show_integration=False,
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
                    show_derived_lines=is_line_family and active_layer_enabled,
                    show_uncertainty=False,
                    show_normalization=is_line_family,
                    show_integration=is_line_family and active_layer_enabled,
                    show_group_members=True,
                    show_metadata=True,
                    show_fit_editor=is_line_family and active_layer_enabled,
                    show_cumulative_editor=is_line_family and active_layer_enabled,
                )
            if layer_kind == "fit":
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=False,
                    show_style=False,
                    show_markers=False,
                    show_derived_lines=active_layer_enabled,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_integration=False,
                    show_group_members=False,
                    show_metadata=False,
                    show_fit_editor=is_line_family and active_layer_enabled,
                    show_cumulative_editor=False,
                )
            if layer_kind == "cumulative":
                return _LayerInspectorCapabilities(
                    plot_family=plot_family,
                    layer_kind=layer_kind,
                    show_visibility_label=False,
                    show_style=False,
                    show_markers=False,
                    show_derived_lines=active_layer_enabled,
                    show_uncertainty=False,
                    show_normalization=False,
                    show_integration=False,
                    show_group_members=False,
                    show_metadata=False,
                    show_fit_editor=False,
                    show_cumulative_editor=is_line_family and active_layer_enabled,
                )
            return _LayerInspectorCapabilities(
                plot_family=plot_family,
                layer_kind=layer_kind,
                show_visibility_label=is_line_family or is_heatmap_family,
                show_style=is_line_family,
                show_markers=is_line_family,
                show_derived_lines=is_line_family and active_layer_enabled,
                show_uncertainty=self._error_supported_for_current_view()
                and active_layer_enabled,
                show_normalization=is_line_family,
                show_integration=is_line_family and active_layer_enabled,
                show_group_members=False,
                show_metadata=True,
                show_fit_editor=is_line_family
                and active_layer_enabled
                and self._fit_supported_for_current_view(),
                show_cumulative_editor=is_line_family and active_layer_enabled,
            )

        def _current_figure_capabilities(self) -> _FigureInspectorCapabilities:
            plot_family = self._current_plot_family()
            if (
                self._analysis_name == "position"
                and self._current_position_is_projection_view()
            ):
                continuous_color = self._current_position_uses_continuous_color()
                return _FigureInspectorCapabilities(
                    plot_family=plot_family,
                    show_legend=not continuous_color,
                    show_lines=not continuous_color,
                    show_heatmap=continuous_color,
                    show_colorbar=continuous_color,
                    show_axis_transforms=False,
                    show_advanced_legend=not continuous_color,
                    show_advanced_lines=not continuous_color,
                )
            is_line = plot_family == "line"
            show_heatmap = plot_family == "heatmap"
            return _FigureInspectorCapabilities(
                plot_family=plot_family,
                show_legend=is_line,
                show_lines=is_line,
                show_heatmap=show_heatmap,
                show_colorbar=show_heatmap,
                show_axis_transforms=is_line,
                show_advanced_legend=is_line,
                show_advanced_lines=is_line,
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
                    error_reason="Grouped series do not support uncertainty overlays.",
                )
            summary = self._preview_error_summary_for_series(index)
            reason = summary.get("reason") if isinstance(summary, dict) else None
            if reason is None and not self._error_supported_for_current_view():
                reason = "Uncertainty overlays are only available for 1-D line plots."
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
            if 0 <= index < len(self._series_fit_x_mins_data):
                config["fit_x_min"] = _soft_float(self._series_fit_x_mins_data[index])
            if 0 <= index < len(self._series_fit_x_maxs_data):
                config["fit_x_max"] = _soft_float(self._series_fit_x_maxs_data[index])
            config["fit_range_mode"] = _fit_range_mode_from_limits(
                self._series_fit_x_mins_data[index]
                if 0 <= index < len(self._series_fit_x_mins_data)
                else "",
                self._series_fit_x_maxs_data[index]
                if 0 <= index < len(self._series_fit_x_maxs_data)
                else "",
            )
            if 0 <= index < len(self._series_fit_color_data):
                config["fit_color"] = self._series_fit_color_data[index].strip() or None
            if 0 <= index < len(self._series_fit_alpha_data):
                raw_fa = self._series_fit_alpha_data[index].strip()
                try:
                    config["fit_alpha"] = float(raw_fa) if raw_fa else None
                except ValueError:
                    config["fit_alpha"] = None
            if 0 <= index < len(self._series_fit_line_width_data):
                raw_flw = self._series_fit_line_width_data[index].strip()
                try:
                    config["fit_line_width"] = float(raw_flw) if raw_flw else None
                except ValueError:
                    config["fit_line_width"] = None
            if 0 <= index < len(self._series_fit_line_style_data):
                config["fit_line_style"] = self._series_fit_line_style_data[index].strip() or None
            return config

        def _fit_effective_label(self, index: int) -> str:
            override = ""
            if 0 <= index < len(self._series_fit_label_overrides_data):
                override = self._series_fit_label_overrides_data[index].strip()
            return override or f"{self._effective_series_label(index)} fit"

        def _series_is_group(self, index: int) -> bool:
            descriptor = self._series_descriptor(index)
            return str(descriptor.get("source_kind") or "source").strip().lower() == "group"

        def _series_is_generated(self, index: int) -> bool:
            descriptor = self._series_descriptor(index)
            return self._series_is_group(index) or bool(descriptor.get("is_generated", False))

        def _series_layer_role(self, index: int) -> str:
            if self._series_is_group(index):
                return "group"
            return "copy" if self._series_is_generated(index) else "original"

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

        def _default_generated_series_color(self, index: int) -> str:
            generated_series_ids = [
                str(descriptor.get("series_id") or f"series:{descriptor_index}")
                for descriptor_index, descriptor in enumerate(self._series_descriptors_data)
                if self._series_is_generated(descriptor_index)
            ]
            if not generated_series_ids:
                return ""
            current_id = str(
                self._series_descriptor(index).get("series_id") or f"series:{index}"
            ).strip()
            if current_id not in generated_series_ids:
                return ""
            default_colors = default_series_colors(max(1, len(generated_series_ids)))
            return default_colors[generated_series_ids.index(current_id)]

        def _clone_series_at_index(self, index: int) -> None:
            if index < 0 or index >= len(self._series_descriptors_data):
                return
            descriptor = dict(self._series_descriptor(index))
            source_kind = str(descriptor.get("source_kind") or "source").strip().lower()
            base_label = self._effective_series_label(index)
            descriptor["series_id"] = f"{source_kind}:{uuid4().hex}"
            descriptor["default_label"] = f"{base_label} Copy"
            descriptor["is_generated"] = True
            if source_kind != "group":
                descriptor["source_kind"] = "source"
                descriptor["source_series_id"] = str(
                    descriptor.get("source_series_id") or ""
                ).strip() or str(
                    self._series_descriptor(index).get("series_id") or f"series:{index}"
                )
            insert_at = index + 1

            def _insert(values: list[Any], copied: Any) -> None:
                values.insert(insert_at, deepcopy(copied))

            for values, copied in (
                (self._series_descriptors_data, descriptor),
                (self._series_labels_data, f"{base_label} Copy"),
                (self._series_label_overrides_data, f"{base_label} Copy"),
                (self._series_colors_data, ""),
                (self._series_enabled_data, self._series_enabled_data[index]),
                (self._series_show_in_legend_data, self._series_show_in_legend_data[index]),
                (self._series_alpha_data, self._series_alpha_data[index]),
                (self._series_error_enabled_data, False),
                (self._series_error_stats_data, "block_sem"),
                (self._series_error_styles_data, "band"),
                (self._series_error_colors_data, ""),
                (self._series_error_label_overrides_data, ""),
                (self._series_error_show_in_legend_data, False),
                (self._series_fit_enabled_data, False),
                (self._series_fit_label_overrides_data, ""),
                (self._series_fit_show_in_legend_data, True),
                (self._series_fit_types_data, "linear"),
                (self._series_fit_degrees_data, "2"),
                (self._series_fit_range_modes_data, "visible"),
                (self._series_fit_x_mins_data, ""),
                (self._series_fit_x_maxs_data, ""),
                (self._series_fit_color_data, ""),
                (self._series_fit_alpha_data, ""),
                (self._series_fit_line_width_data, ""),
                (self._series_fit_line_style_data, ""),
                (self._series_cumulative_enabled_data, False),
                (self._series_cumulative_label_overrides_data, ""),
                (self._series_cumulative_show_in_legend_data, True),
                (self._series_cumulative_color_data, ""),
                (self._series_cumulative_alpha_data, ""),
                (self._series_cumulative_line_width_data, ""),
                (self._series_cumulative_line_style_data, ""),
                (self._series_integration_enabled_data, False),
                (self._series_integration_source_data, "Plotted data"),
                (self._series_integration_x_min_data, ""),
                (self._series_integration_x_max_data, ""),
                (self._series_integration_baseline_data, "0.0"),
                (self._series_integration_color_mode_data, "Auto"),
                (self._series_integration_color_data, ""),
                (self._series_integration_alpha_data, "0.25"),
                (self._series_show_raw_line_data, self._series_show_raw_line_data[index]),
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
            self._validate_series_state_lengths()
            self._series_active_index = insert_at
            self._set_active_series_child_kind("base")
            self._sync_series_selection_widgets(self._series_active_index)
            self._load_series_into_editor(self._series_active_index)
            self._record_history_after_non_text_change()
            self._schedule_preview_update()

        def _duplicate_selected_series(self) -> None:
            self._persist_active_series_editor()
            self._set_active_series_child_kind("base")
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
                "is_generated": True,
                "member_series_ids": member_series_ids,
                "group_reducer": "mean",
                "source_name": "Grouped series",
                "source_directory": "",
                "source_path": "",
            }
            self._series_descriptors_data.append(descriptor)
            self._series_labels_data.append(str(descriptor["default_label"]))
            self._series_label_overrides_data.append("")
            self._series_colors_data.append(
                self._default_generated_series_color(len(self._series_descriptors_data) - 1)
            )
            self._series_enabled_data.append(True)
            self._series_show_in_legend_data.append(True)
            self._series_show_raw_line_data.append(True)
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
            self._series_fit_color_data.append("")
            self._series_fit_alpha_data.append("")
            self._series_fit_line_width_data.append("")
            self._series_fit_line_style_data.append("")
            self._series_cumulative_enabled_data.append(False)
            self._series_cumulative_label_overrides_data.append("")
            self._series_cumulative_show_in_legend_data.append(True)
            self._series_cumulative_color_data.append("")
            self._series_cumulative_alpha_data.append("")
            self._series_cumulative_line_width_data.append("")
            self._series_cumulative_line_style_data.append("")
            self._series_integration_enabled_data.append(False)
            self._series_integration_source_data.append("Plotted data")
            self._series_integration_x_min_data.append("")
            self._series_integration_x_max_data.append("")
            self._series_integration_baseline_data.append("0.0")
            self._series_integration_color_mode_data.append("Auto")
            self._series_integration_color_data.append("")
            self._series_integration_alpha_data.append("0.25")
            self._series_line_widths_data.append("")
            self._series_markers_data.append("")
            self._series_line_kwargs_data.append("")
            self._series_normalization_modes_data.append("none")
            self._series_normalization_values_data.append("")
            self._series_normalization_x_refs_data.append("")
            self._validate_series_state_lengths()
            self._apply_series_id_order(self._enabled_partitioned_series_id_order())
            self._series_active_index = self._current_series_id_order().index(
                str(descriptor["series_id"])
            )
            self._set_active_series_child_kind("base")
            self._sync_series_selection_widgets(self._series_active_index)
            self._load_series_into_editor(self._series_active_index)
            self._record_history_after_non_text_change()
            self._schedule_preview_update()

        def _delete_series_at_index(self, index: int) -> None:
            if index < 0 or index >= len(self._series_descriptors_data):
                return
            if not self._series_is_generated(index):
                return
            removed_id = str(
                self._series_descriptors_data[index].get("series_id") or f"series:{index}"
            )
            for _name, values in self._iter_series_state_lists():
                if index < len(values):
                    values.pop(index)
            for descriptor_index, descriptor in enumerate(self._series_descriptors_data):
                if str(descriptor.get("source_kind") or "source").strip().lower() != "group":
                    continue
                member_ids = [
                    str(member_id).strip()
                    for member_id in descriptor.get("member_series_ids", [])
                    if str(member_id).strip() and str(member_id).strip() != removed_id
                ]
                updated = dict(descriptor)
                updated["member_series_ids"] = member_ids
                self._series_descriptors_data[descriptor_index] = updated
            self._validate_series_state_lengths()
            self._series_active_index = min(index, max(0, len(self._series_descriptors_data) - 1))
            self._set_active_series_child_kind("base")
            self._sync_series_selection_widgets(self._series_active_index)
            if self._series_descriptors_data:
                self._load_series_into_editor(self._series_active_index)
            self._record_history_after_non_text_change()
            self._schedule_preview_update()

        def _delete_selected_series(self) -> None:
            self._persist_active_series_editor()
            self._set_active_series_child_kind("base")
            self._delete_series_at_index(self._series_active_index)

        def _rebuild_series_display_rows(self) -> None:
            rows: list[dict[str, Any]] = []
            for index in range(len(self._series_labels_data)):
                rows.append({"kind": "base", "base_index": index})
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
            badges: list[str] = []
            if (
                base_index < len(self._series_cumulative_enabled_data)
                and self._series_cumulative_enabled_data[base_index]
                and not self._is_orientation_heatmap_mode()
            ):
                badges.append("~")
            if (
                base_index < len(self._series_fit_enabled_data)
                and self._series_fit_enabled_data[base_index]
                and self._fit_supported_for_current_view()
            ):
                badges.append("\u2197")
            if badges:
                base_label = base_label + "  \u00b7 " + " ".join(badges)
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
                if not checked:
                    self._series_fit_enabled_data[index] = False
                    self._series_cumulative_enabled_data[index] = False
                    self._series_error_enabled_data[index] = False
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
            self._record_history_after_non_text_change()
            self._schedule_or_apply_series_preview_update(force_full_render=True)

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
            self._record_history_after_non_text_change()
            self._schedule_or_apply_series_preview_update()

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
            self._update_binning_helper_summary(index)

        def _update_binning_helper_summary(self, index: int) -> None:
            if self._binning_helper_label is None:
                return
            auto_note = str(getattr(self, "_auto_display_note", "") or "").strip()
            if not hasattr(self, "x_bin_width") or not hasattr(self, "min_bin_points"):
                self._binning_helper_label.hide()
                self._binning_helper_label.setText("")
                return

            if index < 0 or index >= len(self._series_labels_data):
                lines = []
                if auto_note:
                    lines.append(auto_note)
                lines.append(
                    "Refresh preview to inspect source bin size, requested display bin size, "
                    "and bin occupancy for the current layer."
                )
                self._binning_helper_label.setText("\n".join(lines))
                self._binning_helper_label.show()
                return

            descriptor = self._series_descriptor(index)
            series_id = str(descriptor.get("series_id") or "")
            point_counts_map = self._last_preview_state.get("series_point_counts")
            source_widths_map = self._last_preview_state.get("series_source_bin_widths")
            masked_map = self._last_preview_state.get("series_masked_bin_counts")

            point_counts_raw = (
                point_counts_map.get(series_id) if isinstance(point_counts_map, dict) else None
            )
            point_counts = (
                np.asarray(point_counts_raw, dtype=float)
                if isinstance(point_counts_raw, list) and point_counts_raw
                else np.empty(0, dtype=float)
            )
            source_width_raw = (
                source_widths_map.get(series_id) if isinstance(source_widths_map, dict) else None
            )
            source_width = (
                float(source_width_raw)
                if source_width_raw is not None and np.isfinite(source_width_raw)
                else None
            )
            masked_count = masked_map.get(series_id) if isinstance(masked_map, dict) else None
            min_points_text = self.min_bin_points.text().strip()
            requested_width_text = self.x_bin_width.text().strip()
            requested_width: float | None = None
            if requested_width_text:
                try:
                    requested_width = float(requested_width_text)
                except ValueError:
                    requested_width = None

            lines: list[str] = []
            if source_width is not None:
                lines.append(f"Source bin size: {_format_float_value(source_width)}")

            if requested_width_text:
                lines.append(f"Requested display bin size: {requested_width_text}")
                if source_width is not None and requested_width is not None:
                    status = (
                        "valid"
                        if requested_width >= source_width
                        else "smaller than source bin size"
                    )
                    lines.append(f"Requested bin size status: {status}")
            elif source_width is not None:
                lines.append("Requested display bin size: source bin size")

            if point_counts.size > 0:
                lines.append(f"Visible bins: {int(point_counts.size)}")
                lines.append(
                    "Points per visible bin: "
                    f"avg {_format_float_value(float(np.mean(point_counts)))}, "
                    f"median {_format_float_value(float(np.median(point_counts)))}"
                )
            if masked_count is not None:
                if min_points_text:
                    lines.append(
                        f"Masked bins at threshold {min_points_text}: {int(masked_count)}"
                    )
                else:
                    lines.append(f"Masked bins: {int(masked_count)}")

            if not lines:
                lines.append(
                    "Refresh preview to inspect source bin size, requested display bin size, "
                    "and bin occupancy for the current layer."
                )
            if auto_note:
                lines.insert(0, auto_note)
            self._binning_helper_label.setText("\n".join(lines))
            self._binning_helper_label.show()

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

        def _clear_series_list_widget_items(self) -> None:
            if not hasattr(self, "series_list") or self.series_list is None:
                return
            self.series_list.clear()

        def _apply_series_list_item_visuals(self, item: Any, index: int) -> None:
            if item is None or index < 0:
                return
            row_descriptor = self._display_row(index)
            base_index = int(row_descriptor.get("base_index", -1))
            if base_index < 0 or base_index >= len(self._series_enabled_data):
                return
            base_state = self._effective_series_state(base_index)
            kind = str(row_descriptor.get("kind") or "base")
            if kind == "fit":
                enabled = bool(self._series_fit_enabled_data[base_index])
            elif kind == "cumulative":
                enabled = bool(self._series_cumulative_enabled_data[base_index])
            else:
                enabled = bool(base_state["enabled"])
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
            heatmap_active = self._is_orientation_heatmap_mode()
            color_token = (
                ""
                if heatmap_active
                else (
                    (
                        self._series_fit_color_data[base_index].strip()
                        if base_index < len(self._series_fit_color_data)
                        else ""
                    )
                    or self._effective_series_color(base_index)
                )
                if kind == "fit"
                else (
                    (
                        self._series_cumulative_color_data[base_index].strip()
                        if base_index < len(self._series_cumulative_color_data)
                        else ""
                    )
                    or self._effective_series_color(base_index)
                )
                if kind == "cumulative"
                else str(base_state["color"])
            )
            layer_role = (
                "fit"
                if kind == "fit"
                else "cumulative"
                if kind == "cumulative"
                else str(base_state["layer_role"])
            )
            series_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            item.setData(
                _SERIES_ROW_STATE_ROLE,
                {
                    "text": self._display_row_text(index).replace("Â·", "-"),
                    "checked": enabled,
                    "enabled": enabled,
                    "selected": self.series_list.currentRow() == index,
                    "color_token": color_token,
                    "kind": kind,
                    "layer_role": layer_role,
                    "can_move_up": base_index > 0,
                    "can_move_down": base_index < len(self._series_labels_data) - 1,
                    "tooltip_text": item.toolTip(),
                    "theme": self._theme_tokens(),
                    "series_id": series_id,
                },
            )
            if hasattr(self.series_list, "notifyRowChanged"):
                self.series_list.notifyRowChanged(index)
            row_widget = self.series_list.itemWidget(item)
            if isinstance(row_widget, _SeriesRowWidget):
                heatmap_active = self._is_orientation_heatmap_mode()
                row_widget.update_content(
                    text=self._display_row_text(index).replace("·", "-"),
                    checked=enabled,
                    enabled=enabled,
                    selected=self.series_list.currentRow() == index,
                    color_token=(
                        ""
                        if heatmap_active
                        else (
                            (
                                self._series_fit_color_data[base_index].strip()
                                if base_index < len(self._series_fit_color_data)
                                else ""
                            )
                            or self._effective_series_color(base_index)
                        )
                        if kind == "fit"
                        else (
                            (
                                self._series_cumulative_color_data[base_index].strip()
                                if base_index < len(self._series_cumulative_color_data)
                                else ""
                            )
                            or self._effective_series_color(base_index)
                        )
                        if kind == "cumulative"
                        else str(base_state["color"])
                    ),
                    kind=kind,
                    layer_role=(
                        "fit"
                        if kind == "fit"
                        else "cumulative"
                        if kind == "cumulative"
                        else str(base_state["layer_role"])
                    ),
                    can_move_up=base_index > 0,
                    can_move_down=base_index < len(self._series_labels_data) - 1,
                    tooltip_text=item.toolTip(),
                    theme=self._theme_tokens(),
                )

        def _sync_series_selection_widgets(self, selected_index: int) -> None:
            view_anchor = self._current_series_list_view_anchor()
            self._rebuild_series_display_rows()
            old_signal_block = self.series_list.blockSignals(True)
            self.series_list.setUpdatesEnabled(False)
            try:
                self._clear_series_list_widget_items()
                for index in range(len(self._series_display_rows)):
                    item = _SeriesListItem()
                    row_descriptor = self._display_row(index)
                    base_index = int(row_descriptor.get("base_index", 0))
                    self.series_list.addItem(item)
                    self._apply_series_list_item_visuals(item, index)
                if self.series_list.count() > 0:
                    row = self._display_row_for_selection(
                        selected_index,
                        kind=self._active_series_child_kind(),
                    )
                    self.series_list.setCurrentRow(row)
                    self._restore_series_list_view_anchor(view_anchor)
                    self._refresh_series_list_widgets()
            finally:
                self.series_list.setUpdatesEnabled(True)
                self.series_list.blockSignals(old_signal_block)

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
                "Density Binning"
                if self._analysis_name == "density"
                else "Position Binning / Sampling"
                if self._analysis_name == "position"
                else "Display Binning / Sectioning"
            )
            binning = QGroupBox(binning_title)
            self._data_transform_group = binning
            binning_form = QFormLayout(binning)
            self.x_bin_width = self._line("Leave blank to use the data width")
            self.x_bin_width.textChanged.connect(self._refresh_widget_states)
            if self._analysis_name == "density":
                self.x_bin_width.textChanged.connect(self._handle_density_binning_change)
            self.x_bin_reducer = self._combo(_BIN_REDUCERS)
            self._add_form_row(
                binning_form,
                "X-axis bin size" if self._analysis_name == "density" else "X bin size",
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
            if self._analysis_name != "density":
                self._add_form_row(
                    binning_form,
                    "Min points per bin",
                    self.min_bin_points,
                    tooltip_id="data.section.min_points",
                )
                self._min_bin_points_row = (binning_form, self.min_bin_points)
            self._binning_helper_label = QLabel("")
            self._binning_helper_label.setObjectName("sectionNote")
            self._binning_helper_label.setWordWrap(True)
            binning_form.addRow(self._binning_helper_label)

            if self._analysis_name in {"density", "orientation"}:
                self.y_bin_width = self._line("Leave blank to use the data width")
                self.y_bin_width.textChanged.connect(self._refresh_widget_states)
                if self._analysis_name == "density":
                    self.y_bin_width.textChanged.connect(self._handle_density_binning_change)
                self.y_bin_reducer = self._combo(_BIN_REDUCERS)
                self._add_form_row(
                    binning_form,
                    "Y-axis bin size" if self._analysis_name == "density" else "Y bin size",
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

            layout.addWidget(
                self._make_collapsible_section(
                    title=binning_title,
                    section_id="data.binning",
                    body_widget=binning,
                )
            )

        def _build_data_export_section(self, layout: QVBoxLayout) -> None:
            if on_save_data is None:
                return

            group = QGroupBox("Export Data")
            form = QFormLayout(group)

            self._data_export_summary_label = QLabel("")
            self._data_export_summary_label.setObjectName("sectionNote")
            self._data_export_summary_label.setWordWrap(True)
            form.addRow(self._data_export_summary_label)

            self._data_export_format = self._combo(("Auto", "CSV", "DAT", "TSV", "TXT"))
            self._data_export_format.currentTextChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                form,
                "Format",
                self._data_export_format,
                tooltip_id="export.data",
            )

            self._data_export_delimiter = self._combo(("Auto", "Comma", "Tab", "Space"))
            self._data_export_delimiter.currentTextChanged.connect(self._refresh_widget_states)
            self._add_form_row(
                form,
                "Delimiter",
                self._data_export_delimiter,
                tooltip_id="export.data",
            )

            self._data_export_enabled_only = QCheckBox("Enabled series only")
            self._data_export_enabled_only.setChecked(True)
            self._data_export_enabled_only.setEnabled(False)
            form.addRow("", self._data_export_enabled_only)

            self._data_export_include_metadata = QCheckBox("Include metadata comments")
            self._data_export_include_metadata.setChecked(False)
            self._data_export_include_metadata.toggled.connect(self._refresh_widget_states)
            form.addRow("", self._data_export_include_metadata)

            self._data_export_button = QPushButton("Export Data")
            self._data_export_button.clicked.connect(self._handle_save_data)
            self._register_tooltip(self._data_export_button, "export.data")
            self._apply_widget_tooltip(self._data_export_button)
            form.addRow("", self._data_export_button)

            layout.addWidget(
                self._make_collapsible_section(
                    title="Export Data",
                    section_id="data.export",
                    body_widget=group,
                )
            )

        def _update_data_export_summary(self) -> None:
            if self._data_export_summary_label is None:
                return
            enabled_count = sum(1 for value in self._series_enabled_data if value)
            total_count = len(self._series_enabled_data)
            view_type = "2D" if self._is_density_heatmap_mode() else "1D"
            species_text = ""
            if self._analysis_name == "density":
                species = self._enabled_density_species()
                if species:
                    species_text = "; species=" + ",".join(sorted(species))
            axes_text = ""
            if self._analysis_name == "density" and self._is_density_heatmap_mode():
                x_axis_widget = getattr(self, "density_2d_x_axis", None)
                y_axis_widget = getattr(self, "density_2d_y_axis", None)
                axes_text = (
                    f"; axes={x_axis_widget.currentText()}/"
                    f"{y_axis_widget.currentText()}"
                    if x_axis_widget is not None and y_axis_widget is not None
                    else ""
                )
            elif hasattr(self, "density_x_quantity") and self.density_x_quantity is not None:
                axes_text = f"; x={self.density_x_quantity.currentText()}"
            self._data_export_summary_label.setText(
                f"Exports the current plotted data from {enabled_count}/{total_count} "
                f"enabled series; view={view_type}{species_text}{axes_text}."
            )

        def _build_data_tab(self) -> None:
            layout = QVBoxLayout(self._tab_data_content)
            self._build_analysis_data_sections(layout)
            self._build_binning_section(layout)
            self._build_data_export_section(layout)
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
            layout.addWidget(
                self._make_collapsible_section(
                    title="Matplotlib rcParams",
                    section_id="advanced.rcparams",
                    body_widget=rc_group,
                )
            )

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
            layout.addWidget(
                self._make_collapsible_section(
                    title="Figure / Axes / Layout",
                    section_id="advanced.render",
                    body_widget=render_group,
                )
            )

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
            self._advanced_legend_kwargs_rows.append(
                (style_form, self.legend_kwargs_json)
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
            self._advanced_line_kwargs_rows.append((style_form, self.line_kwargs_json))
            layout.addWidget(
                self._make_collapsible_section(
                    title="Raw Matplotlib kwargs",
                    section_id="advanced.style",
                    body_widget=style_group,
                )
            )

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
            self._series_show_raw_line_data = []
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
            self._series_fit_color_data = []
            self._series_fit_alpha_data = []
            self._series_fit_line_width_data = []
            self._series_fit_line_style_data = []
            self._series_cumulative_enabled_data = []
            self._series_cumulative_label_overrides_data = []
            self._series_cumulative_show_in_legend_data = []
            self._series_cumulative_color_data = []
            self._series_cumulative_alpha_data = []
            self._series_cumulative_line_width_data = []
            self._series_cumulative_line_style_data = []
            self._series_integration_enabled_data = []
            self._series_integration_source_data = []
            self._series_integration_x_min_data = []
            self._series_integration_x_max_data = []
            self._series_integration_baseline_data = []
            self._series_integration_color_mode_data = []
            self._series_integration_color_data = []
            self._series_integration_alpha_data = []
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
                source_kind = (
                    "group"
                    if str(descriptor.get("source_kind") or "").strip().lower() == "group"
                    else "source"
                )
                descriptor["source_kind"] = source_kind
                descriptor["is_generated"] = bool(
                    descriptor.get("is_generated", source_kind == "group")
                )
                if source_kind != "group":
                    descriptor["source_series_id"] = (
                        str(descriptor.get("source_series_id") or "").strip()
                        or descriptor["series_id"]
                    )
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

                show_raw_line = True
                if isinstance(series_override, dict) and "show_raw_line" in series_override:
                    show_raw_line = self._coerce_series_bool(
                        series_override.get("show_raw_line"), default=True
                    )
                self._series_show_raw_line_data.append(show_raw_line)

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
                self._series_error_enabled_data.append(
                    bool(error_config.get("enabled", False)) and enabled
                )
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
                    # Legacy compat: apply root-level fit keys only when no
                    # "fit" sub-dict exists so new-format saves stay
                    # authoritative.
                    if not isinstance(series_override.get("fit"), dict):
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
                self._series_fit_enabled_data.append(
                    bool(fit_config.get("fit_enabled", False)) and enabled
                )
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
                    bool(cumulative_config.get("enabled", False)) and enabled
                )
                self._series_cumulative_label_overrides_data.append(
                    str(cumulative_config.get("label_override") or "").strip()
                )
                self._series_cumulative_show_in_legend_data.append(
                    bool(cumulative_config.get("show_in_legend", True))
                )
                self._series_cumulative_color_data.append(
                    str(cumulative_config.get("color") or "").strip()
                )
                self._series_cumulative_alpha_data.append(
                    str(cumulative_config.get("alpha") or "").strip()
                )
                self._series_cumulative_line_width_data.append(
                    str(cumulative_config.get("line_width") or "").strip()
                )
                self._series_cumulative_line_style_data.append(
                    str(cumulative_config.get("line_style") or "").strip()
                )

                integration_config = _integration_defaults_for_gui()
                if isinstance(series_override, dict):
                    integration_config = _coerce_series_integration_config(
                        series_override.get("integration")
                    )
                self._series_integration_enabled_data.append(
                    bool(integration_config.get("enabled", False))
                )
                self._series_integration_source_data.append(
                    _INTEGRATION_SOURCE_LABEL_BY_MODE.get(
                        str(integration_config.get("source") or "plotted").strip().lower(),
                        "Plotted data",
                    )
                )
                self._series_integration_x_min_data.append(
                    ""
                    if integration_config.get("x_min") is None
                    else str(integration_config["x_min"])
                )
                self._series_integration_x_max_data.append(
                    ""
                    if integration_config.get("x_max") is None
                    else str(integration_config["x_max"])
                )
                self._series_integration_baseline_data.append(
                    "0.0"
                    if integration_config.get("baseline") is None
                    else str(integration_config["baseline"])
                )
                integration_color = str(integration_config.get("color") or "").strip()
                self._series_integration_color_mode_data.append(
                    "Custom" if integration_color else "Auto"
                )
                self._series_integration_color_data.append(integration_color)
                self._series_integration_alpha_data.append(
                    "0.25"
                    if integration_config.get("alpha") is None
                    else str(integration_config["alpha"])
                )

                fit_color_raw = ""
                fit_alpha_raw = ""
                fit_line_width_raw = ""
                fit_line_style_raw = ""
                if isinstance(series_override, dict):
                    fit_override = series_override.get("fit")
                    fit_override_dict = fit_override if isinstance(fit_override, dict) else {}
                    fit_color_value = fit_override_dict.get("fit_color")
                    if fit_color_value is None:
                        fit_color_value = series_override.get("fit_color")
                    fit_alpha_value = fit_override_dict.get("fit_alpha")
                    if fit_alpha_value is None:
                        fit_alpha_value = series_override.get("fit_alpha")
                    fit_line_width_value = fit_override_dict.get("fit_line_width")
                    if fit_line_width_value is None:
                        fit_line_width_value = series_override.get("fit_line_width")
                    fit_line_style_value = fit_override_dict.get("fit_line_style")
                    if fit_line_style_value is None:
                        fit_line_style_value = series_override.get("fit_line_style")
                    fit_color_raw = str(fit_color_value or "").strip()
                    fit_alpha_raw = str("" if fit_alpha_value is None else fit_alpha_value).strip()
                    fit_line_width_raw = str(
                        "" if fit_line_width_value is None else fit_line_width_value
                    ).strip()
                    fit_line_style_raw = str(
                        "" if fit_line_style_value is None else fit_line_style_value
                    ).strip()
                self._series_fit_color_data.append(fit_color_raw)
                self._series_fit_alpha_data.append(fit_alpha_raw)
                self._series_fit_line_width_data.append(fit_line_width_raw)
                self._series_fit_line_style_data.append(fit_line_style_raw)

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
                self._validate_series_state_lengths()
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
                if hasattr(self, "_series_show_raw_line") and index < len(
                    self._series_show_raw_line_data
                ):
                    self._set_combo_value(
                        self._series_show_raw_line,
                        "on" if self._series_show_raw_line_data[index] else "off",
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
                        self._series_error_style,
                        _ERROR_STYLE_DISPLAY.get(
                            self._series_error_styles_data[index],
                            self._series_error_styles_data[index],
                        ),
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
                if hasattr(self, "_series_cumulative_color"):
                    self._series_cumulative_color.setText(self._series_cumulative_color_data[index])
                if hasattr(self, "_series_cumulative_alpha"):
                    self._series_cumulative_alpha.setText(self._series_cumulative_alpha_data[index])
                if hasattr(self, "_series_cumulative_line_width"):
                    self._series_cumulative_line_width.setText(
                        self._series_cumulative_line_width_data[index]
                    )
                if hasattr(self, "_series_cumulative_line_style"):
                    self._set_combo_value(
                        self._series_cumulative_line_style,
                        self._series_cumulative_line_style_data[index],
                    )
                if hasattr(self, "integration_mode"):
                    self._set_combo_value(
                        self.integration_mode,
                        "on" if self._series_integration_enabled_data[index] else "off",
                    )
                if hasattr(self, "integration_source"):
                    self._set_combo_value(
                        self.integration_source,
                        self._series_integration_source_data[index],
                    )
                if hasattr(self, "integration_x_min"):
                    self.integration_x_min.setText(self._series_integration_x_min_data[index])
                if hasattr(self, "integration_x_max"):
                    self.integration_x_max.setText(self._series_integration_x_max_data[index])
                if hasattr(self, "integration_baseline"):
                    self.integration_baseline.setText(self._series_integration_baseline_data[index])
                if hasattr(self, "integration_color_mode"):
                    self._set_combo_value(
                        self.integration_color_mode,
                        self._series_integration_color_mode_data[index],
                    )
                if hasattr(self, "integration_color"):
                    self.integration_color.setText(self._series_integration_color_data[index])
                if hasattr(self, "integration_alpha"):
                    self.integration_alpha.setText(self._series_integration_alpha_data[index])
                if self._series_fit_mode is not None:
                    self._set_combo_value(
                        self._series_fit_mode,
                        "on" if self._series_fit_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_type"):
                    self._set_combo_value(self._series_fit_type, self._series_fit_types_data[index])
                if hasattr(self, "_series_fit_degree"):
                    self._series_fit_degree.setText(self._series_fit_degrees_data[index])
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
                if hasattr(self, "_series_fit_color"):
                    self._series_fit_color.setText(self._series_fit_color_data[index])
                if hasattr(self, "_series_fit_alpha"):
                    self._series_fit_alpha.setText(self._series_fit_alpha_data[index])
                if hasattr(self, "_series_fit_line_width"):
                    self._series_fit_line_width.setText(self._series_fit_line_width_data[index])
                if hasattr(self, "_series_fit_line_style"):
                    self._set_combo_value(
                        self._series_fit_line_style, self._series_fit_line_style_data[index]
                    )
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
                            candidate_role = self._series_layer_role(candidate_index).title()
                            item = QListWidgetItem(
                                f"{self._effective_series_label(candidate_index)} [{candidate_role}]"
                            )
                            item.setData(Qt.ItemDataRole.UserRole, candidate_id)
                            item.setFlags(
                                item.flags()
                                | Qt.ItemFlag.ItemIsUserCheckable
                                | Qt.ItemFlag.ItemIsEnabled
                            )
                            item.setCheckState(
                                Qt.CheckState.Checked
                                if candidate_id in selected_ids
                                else Qt.CheckState.Unchecked
                            )
                            self._series_group_members.addItem(item)
                    finally:
                        self._series_group_members.blockSignals(False)
                self.series_line_kwargs_json.setPlainText(self._series_line_kwargs_data[index])
                self._load_normalization_into_editor(index)
                for widget in (
                    self.series_color,
                    self.series_alpha,
                    self.series_line_width,
                    self.series_marker,
                    self.series_line_kwargs_json,
                ):
                    widget.setEnabled(True)
                if hasattr(self, "_series_show_raw_line"):
                    self._series_show_raw_line.setEnabled(True)
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
                    getattr(self, "_series_fit_x_min", None),
                    getattr(self, "_series_fit_x_max", None),
                    getattr(self, "_series_fit_show_in_legend", None),
                    getattr(self, "_series_fit_label", None),
                    getattr(self, "_series_fit_color_row", None),
                    getattr(self, "_series_fit_alpha", None),
                    getattr(self, "_series_fit_line_width", None),
                    getattr(self, "_series_fit_line_style", None),
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
            self._update_selected_layer_card(self._series_active_index)
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
            display_available = tuple(_ERROR_STAT_DISPLAY.get(s, s) for s in available)
            display_current = _ERROR_STAT_DISPLAY.get(current, current)
            self._series_error_stat.blockSignals(True)
            try:
                self._series_error_stat.clear()
                self._series_error_stat.addItems(display_available)
                self._set_combo_value(self._series_error_stat, display_current)
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
                        self._series_error_style,
                        _ERROR_STYLE_DISPLAY.get(
                            self._series_error_styles_data[index],
                            self._series_error_styles_data[index],
                        ),
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
                    self._set_combo_value(
                        self._series_fit_mode,
                        "on" if self._series_fit_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_type"):
                    self._set_combo_value(self._series_fit_type, self._series_fit_types_data[index])
                if hasattr(self, "_series_fit_degree"):
                    self._series_fit_degree.setText(self._series_fit_degrees_data[index])
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
                if hasattr(self, "_series_fit_color"):
                    self._series_fit_color.setText(self._series_fit_color_data[index])
                if hasattr(self, "_series_fit_alpha"):
                    self._series_fit_alpha.setText(self._series_fit_alpha_data[index])
                if hasattr(self, "_series_fit_line_width"):
                    self._series_fit_line_width.setText(self._series_fit_line_width_data[index])
                if hasattr(self, "_series_fit_line_style"):
                    self._set_combo_value(
                        self._series_fit_line_style, self._series_fit_line_style_data[index]
                    )
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
                if hasattr(self, "_series_show_raw_line"):
                    self._series_show_raw_line.setEnabled(False)
                for widget in (
                    getattr(self, "_series_fit_type", None),
                    getattr(self, "_series_fit_degree", None),
                    getattr(self, "_series_fit_x_min", None),
                    getattr(self, "_series_fit_x_max", None),
                ):
                    if widget is not None:
                        widget.setEnabled(False)
                for widget in (
                    getattr(self, "_series_fit_show_in_legend", None),
                    getattr(self, "_series_fit_label", None),
                    getattr(self, "_series_fit_color_row", None),
                    getattr(self, "_series_fit_alpha", None),
                    getattr(self, "_series_fit_line_width", None),
                    getattr(self, "_series_fit_line_style", None),
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
            self._update_selected_layer_card(index)
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
                        self._series_error_style,
                        _ERROR_STYLE_DISPLAY.get(
                            self._series_error_styles_data[index],
                            self._series_error_styles_data[index],
                        ),
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
                if hasattr(self, "_series_cumulative_color"):
                    self._series_cumulative_color.setText(self._series_cumulative_color_data[index])
                if hasattr(self, "_series_cumulative_alpha"):
                    self._series_cumulative_alpha.setText(self._series_cumulative_alpha_data[index])
                if hasattr(self, "_series_cumulative_line_width"):
                    self._series_cumulative_line_width.setText(
                        self._series_cumulative_line_width_data[index]
                    )
                if hasattr(self, "_series_cumulative_line_style"):
                    self._set_combo_value(
                        self._series_cumulative_line_style,
                        self._series_cumulative_line_style_data[index],
                    )
                if hasattr(self, "integration_mode"):
                    self._set_combo_value(
                        self.integration_mode,
                        "on" if self._series_integration_enabled_data[index] else "off",
                    )
                if hasattr(self, "integration_source"):
                    self._set_combo_value(
                        self.integration_source,
                        self._series_integration_source_data[index],
                    )
                if hasattr(self, "integration_x_min"):
                    self.integration_x_min.setText(self._series_integration_x_min_data[index])
                if hasattr(self, "integration_x_max"):
                    self.integration_x_max.setText(self._series_integration_x_max_data[index])
                if hasattr(self, "integration_baseline"):
                    self.integration_baseline.setText(self._series_integration_baseline_data[index])
                if hasattr(self, "integration_color_mode"):
                    self._set_combo_value(
                        self.integration_color_mode,
                        self._series_integration_color_mode_data[index],
                    )
                if hasattr(self, "integration_color"):
                    self.integration_color.setText(self._series_integration_color_data[index])
                if hasattr(self, "integration_alpha"):
                    self.integration_alpha.setText(self._series_integration_alpha_data[index])
                if self._series_fit_mode is not None:
                    self._set_combo_value(
                        self._series_fit_mode,
                        "on" if self._series_fit_enabled_data[index] else "off",
                    )
                if hasattr(self, "_series_fit_type"):
                    self._set_combo_value(self._series_fit_type, self._series_fit_types_data[index])
                if hasattr(self, "_series_fit_degree"):
                    self._series_fit_degree.setText(self._series_fit_degrees_data[index])
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
                if hasattr(self, "_series_show_raw_line"):
                    self._series_show_raw_line.setEnabled(False)
                for widget in (
                    getattr(self, "_series_cumulative_show_in_legend", None),
                    getattr(self, "_series_cumulative_label", None),
                    getattr(self, "_series_cumulative_color_row", None),
                    getattr(self, "_series_cumulative_alpha", None),
                    getattr(self, "_series_cumulative_line_width", None),
                    getattr(self, "_series_cumulative_line_style", None),
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
            self._update_selected_layer_card(index)
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
            if hasattr(self, "_series_show_raw_line") and index < len(
                self._series_show_raw_line_data
            ):
                self._series_show_raw_line_data[index] = (
                    self._series_show_raw_line.currentText().strip().lower() != "off"
                )
            self._series_alpha_data[index] = self.series_alpha.text().strip()
            if hasattr(self, "_series_error_mode"):
                self._series_error_enabled_data[index] = (
                    self._series_error_mode.currentText().strip().lower() != "off"
                )
            if hasattr(self, "_series_error_stat"):
                display_text = self._series_error_stat.currentText().strip()
                token = _ERROR_STAT_INTERNAL.get(display_text, display_text).lower()
                self._series_error_stats_data[index] = (
                    token if token in _ERROR_STATS else self._default_error_stat_for_series(index)
                )
            if hasattr(self, "_series_error_style"):
                display_text = self._series_error_style.currentText().strip()
                token = _ERROR_STYLE_INTERNAL.get(display_text, display_text).lower()
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
            if hasattr(self, "_series_cumulative_color"):
                self._series_cumulative_color_data[index] = (
                    self._series_cumulative_color.text().strip()
                )
            if hasattr(self, "_series_cumulative_alpha"):
                self._series_cumulative_alpha_data[index] = (
                    self._series_cumulative_alpha.text().strip()
                )
            if hasattr(self, "_series_cumulative_line_width"):
                self._series_cumulative_line_width_data[index] = (
                    self._series_cumulative_line_width.text().strip()
                )
            if hasattr(self, "_series_cumulative_line_style"):
                self._series_cumulative_line_style_data[index] = (
                    self._series_cumulative_line_style.currentText().strip()
                )
            if hasattr(self, "integration_mode"):
                self._series_integration_enabled_data[index] = (
                    self.integration_mode.currentText().strip().lower() != "off"
                )
            if hasattr(self, "integration_source"):
                self._series_integration_source_data[index] = (
                    self.integration_source.currentText().strip()
                )
            if hasattr(self, "integration_x_min"):
                self._series_integration_x_min_data[index] = self.integration_x_min.text().strip()
            if hasattr(self, "integration_x_max"):
                self._series_integration_x_max_data[index] = self.integration_x_max.text().strip()
            if hasattr(self, "integration_baseline"):
                self._series_integration_baseline_data[index] = (
                    self.integration_baseline.text().strip()
                )
            if hasattr(self, "integration_color_mode"):
                self._series_integration_color_mode_data[index] = (
                    self.integration_color_mode.currentText().strip()
                )
            if hasattr(self, "integration_color"):
                self._series_integration_color_data[index] = self.integration_color.text().strip()
            if hasattr(self, "integration_alpha"):
                self._series_integration_alpha_data[index] = self.integration_alpha.text().strip()
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
            if hasattr(self, "_series_fit_color") and index < len(self._series_fit_color_data):
                self._series_fit_color_data[index] = self._series_fit_color.text().strip()
            if hasattr(self, "_series_fit_alpha") and index < len(self._series_fit_alpha_data):
                self._series_fit_alpha_data[index] = self._series_fit_alpha.text().strip()
            if hasattr(self, "_series_fit_line_width") and index < len(
                self._series_fit_line_width_data
            ):
                self._series_fit_line_width_data[index] = self._series_fit_line_width.text().strip()
            if hasattr(self, "_series_fit_line_style") and index < len(
                self._series_fit_line_style_data
            ):
                self._series_fit_line_style_data[index] = (
                    self._series_fit_line_style.currentText().strip()
                )
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
                        for item_index in range(self._series_group_members.count())
                        for item in [self._series_group_members.item(item_index)]
                        if item is not None
                        and item.checkState() == Qt.CheckState.Checked
                        and str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
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
            self._update_selected_layer_card(index)

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
            if hasattr(self, "_series_fit_color"):
                self._series_fit_color_data[index] = self._series_fit_color.text().strip()
            if hasattr(self, "_series_fit_alpha"):
                self._series_fit_alpha_data[index] = self._series_fit_alpha.text().strip()
            if hasattr(self, "_series_fit_line_width"):
                self._series_fit_line_width_data[index] = self._series_fit_line_width.text().strip()
            if hasattr(self, "_series_fit_line_style"):
                self._series_fit_line_style_data[index] = (
                    self._series_fit_line_style.currentText().strip()
                )
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
            if hasattr(self, "_series_cumulative_color"):
                self._series_cumulative_color_data[index] = (
                    self._series_cumulative_color.text().strip()
                )
            if hasattr(self, "_series_cumulative_alpha"):
                self._series_cumulative_alpha_data[index] = (
                    self._series_cumulative_alpha.text().strip()
                )
            if hasattr(self, "_series_cumulative_line_width"):
                self._series_cumulative_line_width_data[index] = (
                    self._series_cumulative_line_width.text().strip()
                )
            if hasattr(self, "_series_cumulative_line_style"):
                self._series_cumulative_line_style_data[index] = (
                    self._series_cumulative_line_style.currentText().strip()
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

        def _schedule_or_apply_series_preview_update(self, *, force_full_render: bool = False) -> None:
            if not force_full_render and self._apply_series_artist_updates_to_canvas():
                self._preview_timer.stop()
                return
            self._schedule_preview_update()

        def _fit_preview_requires_full_render_sender(self, sender: object | None) -> bool:
            return sender in {
                getattr(self, "_series_fit_mode", None),
                getattr(self, "_series_fit_type", None),
                getattr(self, "_series_fit_degree", None),
                getattr(self, "_series_fit_x_min", None),
                getattr(self, "_series_fit_x_max", None),
                getattr(self, "_series_fit_label", None),
                getattr(self, "_series_fit_color", None),
                getattr(self, "_series_fit_alpha", None),
                getattr(self, "_series_fit_line_width", None),
                getattr(self, "_series_fit_line_style", None),
            }

        def _derived_preview_requires_full_render_sender(self, sender: object | None) -> bool:
            return self._fit_preview_requires_full_render_sender(sender) or sender in {
                getattr(self, "_series_error_mode", None),
                getattr(self, "_series_error_stat", None),
                getattr(self, "_series_error_style", None),
                getattr(self, "_series_error_color", None),
                getattr(self, "_series_error_show_in_legend", None),
                getattr(self, "_series_error_label", None),
                getattr(self, "_series_cumulative_mode", None),
                getattr(self, "_series_cumulative_show_in_legend", None),
                getattr(self, "_series_cumulative_label", None),
                getattr(self, "_series_cumulative_color", None),
                getattr(self, "_series_cumulative_alpha", None),
                getattr(self, "_series_cumulative_line_width", None),
                getattr(self, "_series_cumulative_line_style", None),
                getattr(self, "integration_mode", None),
                getattr(self, "integration_source", None),
                getattr(self, "integration_x_min", None),
                getattr(self, "integration_x_max", None),
                getattr(self, "integration_baseline", None),
                getattr(self, "integration_color_mode", None),
                getattr(self, "integration_color", None),
                getattr(self, "integration_alpha", None),
            }

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
            self._record_history_after_non_text_change()
            self._schedule_or_apply_series_preview_update()

        def _set_all_series_enabled(self, enabled: bool) -> None:
            if not self._series_enabled_data:
                return
            self._persist_active_series_editor()
            selected_id = self._active_series_row_id()
            self._series_enabled_data = [enabled] * len(self._series_enabled_data)
            self._apply_series_id_order(self._enabled_partitioned_series_id_order())
            self._restore_active_series_from_id(selected_id)
            self._series_syncing = True
            try:
                self._sync_series_selection_widgets(self._series_active_index)
            finally:
                self._series_syncing = False
            self._refresh_series_list_widgets()
            self._record_history_after_non_text_change()
            self._schedule_or_apply_series_preview_update()

        def _on_series_editor_changed(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            sender = self.sender()
            self._persist_active_series_editor()
            self._refresh_widget_states()
            if not self._sender_is_text_editor():
                self._record_history_after_non_text_change()
            self._schedule_or_apply_series_preview_update(
                force_full_render=self._derived_preview_requires_full_render_sender(sender)
            )

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
            self._update_selected_layer_card(index)
            self._refresh_active_series_list_widgets()
            self._schedule_or_apply_series_preview_update()

        def _on_series_fit_label_changed(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            index = self._series_active_index
            if index < 0 or index >= len(self._series_fit_label_overrides_data):
                return
            self._series_fit_label_overrides_data[index] = self._series_fit_label.text().strip()
            self._refresh_active_series_list_widgets()
            self._schedule_or_apply_series_preview_update(force_full_render=True)

        def _initialize_normalization_data(self, settings: dict[str, Any]) -> None:
            count = len(self._series_labels_data)
            overrides_by_id = _coerce_series_overrides(settings.get("series_overrides"))
            if overrides_by_id:
                self._series_normalization_modes_data = []
                self._series_normalization_values_data = []
                self._series_normalization_x_refs_data = []
                for index in range(count):
                    descriptor = (
                        self._series_descriptors_data[index]
                        if index < len(self._series_descriptors_data)
                        else {"series_id": f"series:{index}"}
                    )
                    series_id = str(descriptor.get("series_id") or f"series:{index}")
                    entry = overrides_by_id.get(series_id, {})
                    mode = str(entry.get("normalization_mode") or "none").strip().lower()
                    if mode not in _NORMALIZATION_MODES:
                        mode = "none"
                    value = (
                        ""
                        if entry.get("normalization_value") is None
                        else str(entry.get("normalization_value")).strip()
                    )
                    x_ref = (
                        ""
                        if entry.get("normalization_x_ref") is None
                        else str(entry.get("normalization_x_ref")).strip()
                    )
                    self._series_normalization_modes_data.append(mode)
                    self._series_normalization_values_data.append(value)
                    self._series_normalization_x_refs_data.append(x_ref)

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
            if mode == "none":
                value = ""
                x_ref = ""
            for index in range(len(self._series_normalization_modes_data)):
                self._series_normalization_modes_data[index] = mode
                self._series_normalization_values_data[index] = value
                self._series_normalization_x_refs_data[index] = x_ref
            self._load_normalization_into_editor(self._series_active_index)
            self._update_normalization_warning()
            self._record_history_after_non_text_change()
            self._schedule_preview_update()
            self._status_label.setText("Copied normalization settings to all layers.")

        def _on_normalization_editor_changed(self, *_unused: object) -> None:
            if self._normalization_syncing:
                return
            self._persist_normalization_editor(self._series_active_index)
            self._refresh_widget_states()
            if not self._sender_is_text_editor():
                self._record_history_after_non_text_change()
            self._schedule_preview_update()

        def _update_normalization_warning(self) -> None:
            if not hasattr(self, "normalization_warning"):
                return
            layer_caps = self._current_layer_capabilities()
            if not layer_caps.show_normalization or self._is_orientation_heatmap_mode():
                self.normalization_warning.setText("")
                self.normalization_warning.hide()
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
                self._series_error_summary.setText(
                    "No uncertainty data available for this selection."
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

            if not self._error_supported_for_current_view():
                self._series_error_summary.setText(
                    "Uncertainty overlays are only available for 1-D line plots."
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
                available_display = (
                    [_ERROR_STAT_DISPLAY.get(s, s) for s in available]
                    if available
                    else ["none yet"]
                )
                configured_stat = self._series_error_stats_data[index]
                resolved_stat = self._resolved_error_stat_for_series(index)
                statistics_mode = self._current_error_statistics_mode_for_series(index)
                statistics_mode_text = {
                    "direct": "Stored statistics",
                    "raw_grouped": "Grouped raw data",
                    "saved_rebinned_sample": "Rebinned from stored data",
                }.get(statistics_mode or "", "Pending")
                lines = [
                    "Uncertainty details will appear after the next render.",
                    f"Selected: {_ERROR_STAT_DISPLAY.get(configured_stat or 'sample_sem', configured_stat or 'Sample SEM')}",
                    f"Effective: {_ERROR_STAT_DISPLAY.get(resolved_stat, resolved_stat)}",
                    f"Source: {statistics_mode_text}",
                    f"Available: {', '.join(available_display)}",
                ]
                if availability.reason:
                    lines.append(availability.reason)
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
                    "direct": "Stored statistics",
                    "raw_grouped": "Grouped raw data",
                    "saved_rebinned_sample": "Rebinned from stored data",
                }.get(statistics_mode, "Unknown")
                raw_stat = summary.get("stat") or self._resolved_error_stat_for_series(index)
                raw_style = summary.get("style") or self._series_error_styles_data[index]
                lines = [
                    f"Statistic: {_ERROR_STAT_DISPLAY.get(raw_stat, raw_stat)}",
                    f"Style: {_ERROR_STYLE_DISPLAY.get(raw_style, raw_style)}",
                    f"Color: {summary.get('color') or self._series_error_colors_data[index] or 'Auto (matches series)'}",
                    f"Source: {statistics_mode_text}",
                    f"Visible bins: {summary.get('point_count') if summary.get('point_count') is not None else 'n/a'}",
                ]
                reason = str(summary.get("reason") or "").strip()
                if reason:
                    lines.append(reason)
                if masked_count is not None:
                    lines.append(f"Masked bins: {masked_count}")
                if available:
                    available_display = [_ERROR_STAT_DISPLAY.get(s, s) for s in available]
                    lines.append(f"Available: {', '.join(available_display)}")
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
                available_display = (
                    [_ERROR_STAT_DISPLAY.get(s, s) for s in available] if available else ["none"]
                )
                self._series_error_summary.setText(
                    "\n".join(
                        [
                            "Uncertainty is not currently shown for this series.",
                            f"Available: {', '.join(available_display)}",
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
                    summary.get("reason") or "Uncertainty overlay is not active for this series."
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

            reason = str(
                summary.get("reason") or "Uncertainty data is unavailable for this series."
            )
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
                characteristic_point = summary.get("characteristic_point")
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
                if isinstance(characteristic_point, dict):
                    label = str(characteristic_point.get("label") or "Characteristic point").strip()
                    point_x = characteristic_point.get("x")
                    point_y = characteristic_point.get("y")
                    if point_x is not None and point_y is not None:
                        lines.append(
                            f"{label}: x={_format_float_value(point_x)}, y={_format_float_value(point_y)}"
                        )
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
                self.legend_title_font,
                self.legend_columns,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.x_ticks,
                self.y_ticks,
                self.x_tick_rotation,
                self.y_tick_rotation,
                self.x_axis_scale,
                self.x_axis_offset,
                self.x_label_pad,
                self.y_label_pad,
                self.fig_width,
                self.fig_height,
                self.dpi,
                self.font_family,
                self.base_font_size,
                self.title_font,
                self.title_pad,
                self.x_label_font,
                self.y_label_font,
                self.x_tick_font,
                self.y_tick_font,
                self.legend_font,
                self.font_color,
                self.line_width,
                self.line_alpha,
                self.marker_size,
                self.marker_color,
                self.figure_facecolor,
                self.figure_alpha,
                self.grid_linewidth,
                self.grid_alpha,
                self.grid_color,
                self.x_tick_length,
                self.x_tick_width,
                self.y_tick_length,
                self.y_tick_width,
                self.integration_x_min,
                self.integration_x_max,
                self.integration_baseline,
                self.integration_color,
                self.integration_alpha,
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
                self.x_ticks_mode,
                self.y_ticks_mode,
                self.grid_mode,
                self.markers_mode,
                self.x_scale,
                self.y_scale,
                self.line_style,
                self.marker_type,
                self.grid_linestyle,
                self.grid_axis,
                self.grid_which,
                self.x_tick_direction,
                self.y_tick_direction,
                self.x_minor_ticks_mode,
                self.y_minor_ticks_mode,
                self.integration_mode,
                self.integration_source,
                self.integration_color_mode,
                self.x_bin_reducer,
                self.axes_border_mode,
            )
            for widget in combo_widgets:
                widget.currentTextChanged.connect(self._record_history_after_non_text_change)
                widget.currentTextChanged.connect(self._schedule_preview_update)
            if hasattr(self, "y_bin_reducer"):
                self.y_bin_reducer.currentTextChanged.connect(
                    self._record_history_after_non_text_change
                )
                self.y_bin_reducer.currentTextChanged.connect(self._schedule_preview_update)
            for cb in (self.border_left, self.border_right, self.border_top, self.border_bottom):
                cb.toggled.connect(self._record_history_after_non_text_change)
                cb.toggled.connect(self._schedule_preview_update)

        def _handle_auto_preview_toggle(self, checked: bool) -> None:
            if not checked:
                self._preview_timer.stop()
                if self._preview_status is not None:
                    self._preview_status.setText("Auto update paused.")
                self._refresh_shell_state()
                return
            if self._preview_status is not None:
                self._preview_status.setText("Auto update enabled.")
            self._refresh_shell_state()
            self._schedule_preview_update()

        def _preview_debounce_ms_for_sender(self, sender: object | None) -> int:
            if sender is None:
                return _AUTO_PREVIEW_DEBOUNCE_MS
            data_widgets = {
                getattr(self, "analysis_species", None),
                getattr(self, "analysis_axis", None),
                getattr(self, "density_view_type", None),
                getattr(self, "density_1d_x_axis", None),
                getattr(self, "density_1d_z_quantity", None),
                getattr(self, "density_2d_x_axis", None),
                getattr(self, "density_2d_y_axis", None),
                getattr(self, "density_2d_z_quantity", None),
                getattr(self, "position_view_type", None),
                getattr(self, "position_quantity", None),
                getattr(self, "position_x_quantity", None),
                getattr(self, "position_y_quantity", None),
                getattr(self, "position_projection_filter_quantity", None),
                getattr(self, "position_projection_filter_min", None),
                getattr(self, "position_projection_filter_max", None),
                getattr(self, "coordination_x_quantity", None),
                getattr(self, "coordination_y_quantity", None),
                getattr(self, "orientation_view_type", None),
                getattr(self, "orientation_component", None),
                getattr(self, "orientation_line_x_axis", None),
                getattr(self, "orientation_angle", None),
                getattr(self, "orientation_heatmap_x_axis", None),
                getattr(self, "orientation_heatmap_y_axis", None),
                getattr(self, "potential_quantity", None),
                getattr(self, "x_bin_width", None),
                getattr(self, "y_bin_width", None),
                getattr(self, "x_bin_reducer", None),
                getattr(self, "y_bin_reducer", None),
                getattr(self, "min_bin_points", None),
            }
            if hasattr(self, "_density_filter_widgets"):
                for lower, upper in self._density_filter_widgets.values():
                    data_widgets.add(lower)
                    data_widgets.add(upper)
            if hasattr(self, "_orientation_filter_widgets"):
                for lower, upper in self._orientation_filter_widgets.values():
                    data_widgets.add(lower)
                    data_widgets.add(upper)
            if sender in data_widgets:
                return _AUTO_PREVIEW_DATA_DEBOUNCE_MS
            series_widgets = {
                getattr(self, "series_list", None),
                getattr(self, "series_name_input", None),
                getattr(self, "series_color_input", None),
                getattr(self, "_series_fit_mode", None),
                getattr(self, "_series_fit_type", None),
                getattr(self, "_series_fit_degree", None),
                getattr(self, "_series_fit_x_min", None),
                getattr(self, "_series_fit_x_max", None),
                getattr(self, "_series_fit_label", None),
                getattr(self, "_series_fit_color", None),
                getattr(self, "_series_fit_alpha", None),
                getattr(self, "_series_fit_line_width", None),
                getattr(self, "_series_fit_line_style", None),
            }
            if sender in series_widgets:
                return _AUTO_PREVIEW_SERIES_DEBOUNCE_MS
            style_widgets = {
                getattr(self, "title_text", None),
                getattr(self, "x_label", None),
                getattr(self, "y_label", None),
                getattr(self, "title_font", None),
                getattr(self, "title_pad", None),
                getattr(self, "x_label_font", None),
                getattr(self, "y_label_font", None),
                getattr(self, "x_label_pad", None),
                getattr(self, "y_label_pad", None),
                getattr(self, "x_scale", None),
                getattr(self, "y_scale", None),
                getattr(self, "grid_mode", None),
                getattr(self, "grid_linestyle", None),
                getattr(self, "grid_linewidth", None),
                getattr(self, "grid_alpha", None),
                getattr(self, "grid_color", None),
                getattr(self, "grid_axis", None),
                getattr(self, "grid_which", None),
                getattr(self, "heatmap_cmap", None),
                getattr(self, "heatmap_vmin", None),
                getattr(self, "heatmap_vmax", None),
                getattr(self, "projection_line_width", None),
                getattr(self, "heatmap_colorbar_enabled", None),
                getattr(self, "heatmap_colorbar_label", None),
                getattr(self, "heatmap_colorbar_label_size", None),
                getattr(self, "heatmap_colorbar_tick_size", None),
                getattr(self, "line_width", None),
                getattr(self, "line_style", None),
                getattr(self, "line_alpha", None),
                getattr(self, "markers_mode", None),
                getattr(self, "marker_size", None),
                getattr(self, "marker_type", None),
                getattr(self, "marker_color", None),
            }
            if sender in style_widgets:
                return _AUTO_PREVIEW_STYLE_DEBOUNCE_MS
            return _AUTO_PREVIEW_DEBOUNCE_MS

        def _schedule_preview_update(self, *_unused: object) -> None:
            if self._suspend_preview_events:
                return
            sender = self.sender()
            if sender in {
                getattr(self, "x_min", None),
                getattr(self, "x_max", None),
            } and self._apply_axis_limit_fields_to_canvas("x_lim"):
                self._preview_timer.stop()
                return
            if sender in {
                getattr(self, "y_min", None),
                getattr(self, "y_max", None),
            } and self._apply_axis_limit_fields_to_canvas("y_lim"):
                self._preview_timer.stop()
                return
            if sender in {
                getattr(self, "title_text", None),
                getattr(self, "x_label", None),
                getattr(self, "y_label", None),
                getattr(self, "title_font", None),
                getattr(self, "title_pad", None),
                getattr(self, "x_label_font", None),
                getattr(self, "y_label_font", None),
                getattr(self, "x_label_pad", None),
                getattr(self, "y_label_pad", None),
            } and self._apply_text_fields_to_canvas():
                self._preview_timer.stop()
                return
            if sender in {
                getattr(self, "fig_width", None),
                getattr(self, "fig_height", None),
                getattr(self, "dpi", None),
                getattr(self, "font_family", None),
                getattr(self, "base_font_size", None),
                getattr(self, "font_color", None),
                getattr(self, "figure_facecolor", None),
                getattr(self, "figure_alpha", None),
                getattr(self, "x_tick_font", None),
                getattr(self, "y_tick_font", None),
                getattr(self, "legend_font", None),
            } and self._apply_canvas_style_fields_to_canvas():
                self._preview_timer.stop()
                return
            if sender in {
                getattr(self, "x_scale", None),
                getattr(self, "y_scale", None),
                getattr(self, "grid_mode", None),
                getattr(self, "grid_linestyle", None),
                getattr(self, "grid_linewidth", None),
                getattr(self, "grid_alpha", None),
                getattr(self, "grid_color", None),
                getattr(self, "grid_axis", None),
                getattr(self, "grid_which", None),
            } and self._apply_axis_style_fields_to_canvas():
                self._preview_timer.stop()
                return
            if sender in {
                getattr(self, "heatmap_cmap", None),
                getattr(self, "heatmap_vmin", None),
                getattr(self, "heatmap_vmax", None),
                getattr(self, "projection_line_width", None),
                getattr(self, "heatmap_colorbar_enabled", None),
                getattr(self, "heatmap_colorbar_label", None),
                getattr(self, "heatmap_colorbar_label_size", None),
                getattr(self, "heatmap_colorbar_tick_size", None),
            } and self._apply_heatmap_style_fields_to_canvas():
                self._preview_timer.stop()
                return
            if sender in {
                getattr(self, "line_width", None),
                getattr(self, "line_style", None),
                getattr(self, "line_alpha", None),
                getattr(self, "markers_mode", None),
                getattr(self, "marker_size", None),
                getattr(self, "marker_type", None),
                getattr(self, "marker_color", None),
            } and self._apply_line_style_fields_to_canvas():
                self._preview_timer.stop()
                return
            if self._auto_preview_checkbox is None:
                return
            if not self._auto_preview_checkbox.isChecked():
                return
            if self._preview_loading:
                if self._preview_status is not None:
                    self._preview_status.setText(
                        "Preview updating..."
                    )
            self._preview_timer.start(self._preview_debounce_ms_for_sender(sender))

        def _handle_debounced_preview(self) -> None:
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
                auto_update_enabled = (
                    self._auto_preview_checkbox is not None
                    and self._auto_preview_checkbox.isChecked()
                )
                self._preview_button.setEnabled(
                    _preview_button_enabled(
                        auto_update_enabled=auto_update_enabled,
                        preview_loading=self._preview_loading,
                    )
                )
            if self._save_figure_button is not None:
                self._save_figure_button.setEnabled(
                    (not self._preview_loading) and on_save_figure is not None
                )
            if self._save_data_button is not None:
                self._save_data_button.setEnabled(
                    (not self._preview_loading) and on_save_data is not None
                )
            if self._data_export_button is not None:
                self._data_export_button.setEnabled(
                    (not self._preview_loading) and on_save_data is not None
                )

        def _cleanup_preview_canvas(self, *, close_figure: bool = True) -> None:
            self._disconnect_preview_axis_callbacks()
            canvas = self._preview_canvas
            toolbar = self._preview_toolbar
            canvas_scroll = self._preview_canvas_scroll
            figure = self._preview_figure
            self._preview_canvas = None
            self._preview_toolbar = None
            self._preview_canvas_scroll = None
            if toolbar is not None:
                toolbar.setParent(None)
                toolbar.deleteLater()
            if canvas_scroll is not None:
                canvas_scroll.setParent(None)
                canvas_scroll.deleteLater()
            if canvas is not None:
                canvas.setParent(None)
                canvas.deleteLater()
            if close_figure and figure is not None:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:
                    pass
                self._preview_figure = None

        def _disconnect_preview_axis_callbacks(self) -> None:
            for ax, callback_id in list(self._preview_axis_callback_ids):
                try:
                    ax.callbacks.disconnect(callback_id)
                except Exception:
                    pass
            self._preview_axis_callback_ids = []

        def _connect_preview_axis_callbacks(self, figure: Any) -> None:
            self._disconnect_preview_axis_callbacks()
            axes = list(getattr(figure, "axes", []) or [])
            if not axes:
                return
            primary_ax = axes[0]
            try:
                x_callback_id = primary_ax.callbacks.connect(
                    "xlim_changed",
                    self._handle_canvas_axis_limits_changed,
                )
                y_callback_id = primary_ax.callbacks.connect(
                    "ylim_changed",
                    self._handle_canvas_axis_limits_changed,
                )
            except Exception:
                self._preview_axis_callback_ids = []
                return
            self._preview_axis_callback_ids = [
                (primary_ax, int(x_callback_id)),
                (primary_ax, int(y_callback_id)),
            ]

        def _install_preview_figure(
            self,
            figure: Any,
            *,
            close_previous: bool = True,
        ) -> None:
            if FigureCanvas is None:
                raise RuntimeError("Matplotlib Qt canvas is unavailable.")
            if self._preview_canvas_container is None or self._preview_canvas_layout is None:
                raise RuntimeError("Preview canvas container is unavailable.")
            if close_previous:
                self._cleanup_preview_canvas(close_figure=True)
            else:
                self._disconnect_preview_axis_callbacks()
                old_canvas = self._preview_canvas
                old_toolbar = self._preview_toolbar
                old_canvas_scroll = self._preview_canvas_scroll
                self._preview_canvas = None
                self._preview_toolbar = None
                self._preview_canvas_scroll = None
                if old_toolbar is not None:
                    old_toolbar.setParent(None)
                    old_toolbar.deleteLater()
                if old_canvas_scroll is not None:
                    old_canvas_scroll.setParent(None)
                    old_canvas_scroll.deleteLater()
                if old_canvas is not None:
                    old_canvas.setParent(None)
                    old_canvas.deleteLater()
            self._preview_figure = figure
            canvas = FigureCanvas(figure)
            toolbar = (
                _LiNaKNavigationToolbar(
                    canvas,
                    self._preview_canvas_container,
                    on_linak_save_figure=self._handle_save_figure,
                )
                if _LiNaKNavigationToolbar is not None
                else None
            )
            if toolbar is not None:
                self._preview_canvas_layout.addWidget(toolbar)
            canvas_scroll = QScrollArea(self._preview_canvas_container)
            canvas_scroll.setWidgetResizable(False)
            canvas_scroll.setFrameShape(QFrame.Shape.NoFrame)
            canvas_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            canvas_scroll.viewport().installEventFilter(self)
            canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            canvas_scroll.setWidget(canvas)
            self._preview_canvas_layout.addWidget(canvas_scroll, stretch=1)
            self._preview_canvas = canvas
            self._preview_toolbar = toolbar
            self._preview_canvas_scroll = canvas_scroll
            if self._preview_scroll is not None:
                self._preview_scroll.setVisible(False)
            self._preview_canvas_container.setVisible(True)
            self._resize_preview_canvas_to_figure()
            canvas.draw_idle()
            self._connect_preview_axis_callbacks(figure)

        def _show_preview_image_fallback(self) -> None:
            if self._preview_canvas_container is not None:
                self._preview_canvas_container.setVisible(False)
            if self._preview_scroll is not None:
                self._preview_scroll.setVisible(True)

        def _new_preview_image_path(self) -> Path:
            path = Path(tempfile.gettempdir()) / f"linak_preview_{uuid4().hex}.png"
            self._preview_temp_paths.add(path)
            return path

        def _cleanup_preview_temp_path(self, path: str | Path | None) -> None:
            if path is None:
                return
            resolved = Path(path)
            QPixmapCache.remove(str(resolved))
            try:
                resolved.unlink(missing_ok=True)
            except OSError:
                pass
            self._preview_temp_paths.discard(resolved)

        def _parse_preview_worker_result(
            self,
            save_result: str | tuple[str, dict[str, Any]] | dict[str, Any] | None,
        ) -> tuple[str, dict[str, Any]]:
            if isinstance(save_result, tuple):
                message = str(save_result[0] or "")
                render_state = save_result[1] if len(save_result) > 1 else {}
                return message, dict(render_state) if isinstance(render_state, dict) else {}
            if isinstance(save_result, dict):
                return "", dict(save_result)
            return str(save_result or ""), {}

        def _preview_worker_loop(self) -> None:
            while not self._preview_worker_stop.is_set():
                try:
                    job = self._preview_worker_queue.get()
                except Exception:
                    continue
                if job is None:
                    self._preview_worker_queue.task_done()
                    break
                generation, settings, image_path = job
                try:
                    if self._preview_worker_job_cancelled(generation):
                        self._cleanup_preview_temp_path(image_path)
                        continue
                    self._run_preview_worker(
                        generation=generation,
                        settings=settings,
                        image_path=image_path,
                    )
                finally:
                    self._preview_worker_queue.task_done()

        def _run_preview_worker(
            self,
            *,
            generation: int,
            settings: dict[str, Any],
            image_path: Path | None,
        ) -> None:
            try:
                if self._preview_worker_job_cancelled(generation):
                    self._cleanup_preview_temp_path(image_path)
                    return
                try:
                    from .plotting import configure_matplotlib_backend

                    configure_matplotlib_backend(interactive=False)
                except Exception:
                    pass
                if self._preview_worker_job_cancelled(generation):
                    self._cleanup_preview_temp_path(image_path)
                    return
                if on_preview_figure is not None and FigureCanvas is not None:
                    render_state = on_preview_figure(settings)
                    payload = {
                        "figure": (
                            render_state.get("figure")
                            if isinstance(render_state, dict)
                            else None
                        ),
                        "image_path": None,
                        "message": "Preview updated.",
                        "render_state": dict(render_state)
                        if isinstance(render_state, dict)
                        else {},
                    }
                elif image_path is None or on_save_figure is None:
                    render_state = on_preview(settings)
                    payload = {
                        "figure": None,
                        "image_path": None,
                        "message": "External preview opened.",
                        "render_state": dict(render_state)
                        if isinstance(render_state, dict)
                        else {},
                    }
                else:
                    save_result = on_save_figure(settings, str(image_path))
                    message, render_state = self._parse_preview_worker_result(save_result)
                    payload = {
                        "figure": None,
                        "image_path": str(image_path),
                        "message": message,
                        "render_state": render_state,
                    }
                try:
                    self._preview_worker_bridge.finished.emit(generation, payload)
                except RuntimeError:
                    pass
            except Exception as exc:
                try:
                    self._preview_worker_bridge.failed.emit(generation, exc)
                except RuntimeError:
                    pass

        def _start_preview_worker(
            self,
            settings: dict[str, Any],
            *,
            interactive: bool,
        ) -> bool:
            if self._closing:
                return False
            self._preview_generation += 1
            generation = self._preview_generation
            image_path = (
                None
                if on_preview_figure is not None and FigureCanvas is not None
                else self._new_preview_image_path()
                if on_save_figure is not None
                else None
            )
            self._active_preview_generation = generation
            self._active_preview_image_path = image_path
            self._active_preview_interactive = bool(interactive)
            self._set_preview_loading(True)
            if self._preview_status is not None:
                self._preview_status.setText("Preview updating...")
            self._preview_worker_queue.put((generation, deepcopy(settings), image_path))
            return True

        def _queue_pending_preview(
            self,
            settings: dict[str, Any],
            *,
            interactive: bool,
        ) -> None:
            self._pending_preview_request = (deepcopy(settings), bool(interactive))
            if self._preview_status is not None:
                self._preview_status.setText(
                    "Preview updating..."
                )

        def _start_pending_preview_if_available(self) -> bool:
            pending = self._pending_preview_request
            if pending is None:
                return False
            self._pending_preview_request = None
            settings, interactive = pending
            return self._start_preview_worker(settings, interactive=interactive)

        def _finish_active_preview_generation(self, generation: int) -> None:
            if self._active_preview_generation == generation:
                self._active_preview_generation = None
                self._active_preview_image_path = None
                self._active_preview_interactive = False

        def _preview_worker_job_cancelled(self, generation: int) -> bool:
            return self._closing or generation != self._active_preview_generation

        def _preview_worker_result_is_stale(self, generation: int) -> bool:
            return (
                self._preview_worker_job_cancelled(generation)
                or self._pending_preview_request is not None
            )

        def _handle_preview_worker_finished(self, generation: int, payload: object) -> None:
            payload_dict = dict(payload) if isinstance(payload, dict) else {}
            image_path = payload_dict.get("image_path")
            figure = payload_dict.get("figure")
            if self._preview_worker_result_is_stale(generation):
                if figure is not None:
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(figure)
                    except Exception:
                        pass
                self._cleanup_preview_temp_path(image_path)
                self._finish_active_preview_generation(generation)
                self._set_preview_loading(False)
                self._start_pending_preview_if_available()
                return

            try:
                render_state = payload_dict.get("render_state")
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_series_state(render_state)
                    self._apply_preview_state_to_synced_fields(render_state)
                if figure is not None:
                    self._install_preview_figure(figure)
                    self._preview_pixmap = None
                    self._cleanup_preview_temp_path(self._preview_image_path)
                    self._preview_image_path = None
                elif image_path is not None:
                    self._cleanup_preview_canvas(close_figure=True)
                    self._show_preview_image_fallback()
                    image_path_obj = Path(str(image_path))
                    previous_path = self._preview_image_path
                    QPixmapCache.remove(str(image_path_obj))
                    pixmap = QPixmap()
                    if not pixmap.load(str(image_path_obj)):
                        raise RuntimeError("Could not load rendered preview image.")
                    if pixmap.isNull():
                        raise RuntimeError("Could not load rendered preview image.")
                    self._preview_image_path = image_path_obj
                    if previous_path != image_path_obj:
                        self._cleanup_preview_temp_path(previous_path)
                    self._preview_pixmap = self._preview_display_pixmap(pixmap)
                    self._refresh_preview_pixmap()
                self._preview_error = None
                if self._preview_status is not None:
                    self._preview_status.setText(
                        "Preview updated."
                        if figure is not None or image_path is not None
                        else str(payload_dict.get("message") or "External preview opened.")
                    )
                self._refresh_shell_state()
            except Exception as exc:
                if self._active_preview_interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_error = str(exc)
                    if self._preview_status is not None:
                        self._preview_status.setText("Preview paused.")
                    self._refresh_shell_state()
            finally:
                self._finish_active_preview_generation(generation)
                self._set_preview_loading(False)
                self._start_pending_preview_if_available()

        def _handle_preview_worker_failed(self, generation: int, exc: object) -> None:
            image_path = (
                self._active_preview_image_path
                if generation == self._active_preview_generation
                else None
            )
            if self._preview_worker_result_is_stale(generation):
                self._cleanup_preview_temp_path(image_path)
                self._finish_active_preview_generation(generation)
                self._set_preview_loading(False)
                self._start_pending_preview_if_available()
                return
            error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
            try:
                self._cleanup_preview_temp_path(image_path)
                if self._active_preview_interactive:
                    self._report_error("Preview failed", error)
                else:
                    self._preview_error = str(error)
                    if self._preview_status is not None:
                        self._preview_status.setText("Preview paused.")
                    self._refresh_shell_state()
            finally:
                self._finish_active_preview_generation(generation)
                self._set_preview_loading(False)
                self._start_pending_preview_if_available()

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

        def _preview_transparency_matte_colors(self) -> tuple[QColor, QColor]:
            if self._theme_mode == "dark":
                return QColor("#8793a3"), QColor("#aeb7c3")
            return QColor("#f4f7fb"), QColor("#dce4ee")

        def _preview_display_pixmap(self, pixmap: QPixmap) -> QPixmap:
            if pixmap.isNull():
                return pixmap
            matte_a, matte_b = self._preview_transparency_matte_colors()
            composed = QPixmap(pixmap.size())
            painter = QPainter(composed)
            try:
                tile_size = 18
                painter.fillRect(composed.rect(), matte_a)
                painter.setBrush(QBrush(matte_b))
                painter.setPen(Qt.PenStyle.NoPen)
                width = max(1, composed.width())
                height = max(1, composed.height())
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                            painter.drawRect(x, y, tile_size, tile_size)
                painter.drawPixmap(0, 0, pixmap)
            finally:
                painter.end()
            return composed

        def _update_embedded_preview(self, *, interactive: bool) -> bool:
            self._preview_timer.stop()
            try:
                settings = self._collect_settings()
            except Exception as exc:
                if interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_error = str(exc)
                    if self._preview_status is not None:
                        self._preview_status.setText("Preview paused.")
                    self._refresh_shell_state()
                return False
            if self._preview_loading:
                self._queue_pending_preview(settings, interactive=interactive)
                return False
            return self._start_preview_worker(settings, interactive=interactive)

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

        def _profile_filter_display_value(
            self,
            value: str | None,
            *,
            default_label: str = "",
        ) -> str:
            token = str(value or "").strip()
            return token or default_label

        def _selected_profile_filter_value(
            self,
            widget: QComboBox,
            *,
            default_label: str = "",
        ) -> str | None:
            token = widget.currentText().strip()
            return None if token == default_label or token == "" else token

        def _density_x_mode_display_label(
            self,
            x_mode: str | None,
            *,
            axis: str | None,
        ) -> str:
            normalized_mode = str(x_mode or "distance").strip().lower() or "distance"
            normalized_axis = str(axis or "").strip().lower()
            if normalized_mode == "axis" and normalized_axis in {"x", "y", "z"}:
                return normalized_axis.upper()
            if normalized_mode in {"x", "y", "z"}:
                return normalized_mode.upper()
            return "Distance"

        def _selected_density_x_mode(self) -> str:
            label = self.density_x_mode.currentText().strip().lower()
            return _DENSITY_X_MODE_BY_LABEL.get(label, "distance")

        def _is_density_heatmap_mode(self) -> bool:
            if self._analysis_name != "density":
                return False
            return (
                canonical_plot_view_id(self._current_density_mapping().view_type_id)
                == PLOT_VIEW_2D_HEATMAP
            )

        def _selected_density_heatmap_source(self) -> dict[str, str] | None:
            entries = self._profile_filter_options.get("density_heatmap_sources")
            if not isinstance(entries, list) or not entries:
                return None
            selected_label = (
                self.density_heatmap_source.currentText().strip()
                if hasattr(self, "density_heatmap_source")
                else ""
            )
            for entry in entries:
                if str(entry.get("label") or "").strip() == selected_label:
                    return {
                        "label": str(entry.get("label") or ""),
                        "species": str(entry.get("species") or ""),
                        "plane": str(entry.get("plane") or ""),
                    }
            first = entries[0]
            return {
                "label": str(first.get("label") or ""),
                "species": str(first.get("species") or ""),
                "plane": str(first.get("plane") or ""),
            }

        def _rdf_species_b_choices(self, species_a: str | None) -> list[str]:
            mapping = self._profile_filter_options.get("species_b_by_species_a", {})
            if not isinstance(mapping, dict):
                return []
            key = "" if species_a is None else str(species_a)
            values = mapping.get(key)
            if not isinstance(values, list):
                values = mapping.get("", [])
            return [str(value).strip() for value in values if str(value).strip()]

        def _coordination_species_b_choices(self, species_a: str | None) -> list[str]:
            return self._rdf_species_b_choices(species_a)

        def _coordination_axis_choices(
            self,
            species_a: str | None,
            species_b: str | None,
        ) -> list[str]:
            axes_by_pair = self._profile_filter_options.get("axes_by_species_pair", {})
            if not isinstance(axes_by_pair, dict):
                return ["x", "y", "z"]
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
            return resolved if resolved else ["x", "y", "z"]

        def _update_rdf_source_summary(self) -> None:
            count_label = getattr(self, "_rdf_source_pair_count_label", None)
            pair_list_label = getattr(self, "_rdf_source_pair_list_label", None)
            if count_label is None or pair_list_label is None:
                return
            pair_labels = [
                self._effective_series_label(index)
                for index, descriptor in enumerate(self._series_descriptors_data)
                if str(descriptor.get("source_kind") or "source").strip().lower() != "group"
            ]
            if not pair_labels:
                count_label.setText("No RDF layers are available.")
                pair_list_label.setText("No RDF profiles were loaded from the selected source.")
                return
            count_label.setText(f"{len(pair_labels)} RDF layer(s)")
            preview_pairs = pair_labels[:8]
            rendered = ", ".join(preview_pairs)
            if len(pair_labels) > len(preview_pairs):
                rendered = f"{rendered}, +{len(pair_labels) - len(preview_pairs)} more"
            pair_list_label.setText(rendered)

        def _handle_coordination_profile_selection_change(self, *_unused: object) -> None:
            if not hasattr(self, "coord_species_a") or not hasattr(self, "coord_species_b"):
                return
            species_a = self._selected_profile_filter_value(self.coord_species_a)
            current_species_b = self.coord_species_b.currentText()
            self._set_combo_items(
                self.coord_species_b,
                self._coordination_species_b_choices(species_a),
                preferred_value=current_species_b,
            )
            if hasattr(self, "analysis_axis"):
                species_b = self._selected_profile_filter_value(self.coord_species_b)
                current_axis = self.analysis_axis.currentText()
                self._set_combo_items(
                    self.analysis_axis,
                    self._coordination_axis_choices(species_a, species_b),
                    preferred_value=current_axis,
                )
            self._handle_series_identity_change()

        def _populate(self, settings: dict[str, Any]) -> None:
            if self._analysis_name == "density":
                raw_states = settings.get("density_view_states")
                if isinstance(raw_states, dict):
                    self._density_view_states = {
                        self._normalize_density_view_type_id(key): deepcopy(value)
                        for key, value in raw_states.items()
                        if isinstance(value, dict)
                    }
                elif not self._density_view_state_switching:
                    self._density_view_states = {}
                settings_mapping_for_view = self._coerce_settings_view_mapping(settings)
                active_view = self._normalize_density_view_type_id(
                    settings.get("density_active_view_type")
                    or (
                        getattr(settings_mapping_for_view, "view_type_id", None)
                        if settings_mapping_for_view is not None
                        else None
                    )
                    or PLOT_VIEW_1D_LINE
                )
                self._density_active_view_type = active_view
                active_state = self._density_view_states.get(active_view)
                if isinstance(active_state, dict) and active_state.get(
                    "_density_view_state_initialized"
                ):
                    settings = self._active_density_settings_with_view_state(
                        settings,
                        active_view,
                        active_state,
                    )
                elif not self._density_view_state_switching:
                    initial_state = deepcopy(settings)
                    initial_state["_density_view_state_initialized"] = True
                    initial_state["density_active_view_type"] = active_view
                    self._density_view_states[active_view] = initial_state
            if self._analysis_name == "position":
                raw_states = settings.get("position_view_states")
                if isinstance(raw_states, dict):
                    self._position_view_states = {
                        self._normalize_position_view_type_id(key): deepcopy(value)
                        for key, value in raw_states.items()
                        if isinstance(value, dict)
                    }
                elif not self._position_view_state_switching:
                    self._position_view_states = {}
                settings_mapping_for_view = self._coerce_settings_view_mapping(settings)
                active_view = self._normalize_position_view_type_id(
                    settings.get("position_active_view_type")
                    or (
                        getattr(settings_mapping_for_view, "view_type_id", None)
                        if settings_mapping_for_view is not None
                        else None
                    )
                    or PLOT_VIEW_1D_LINE
                )
                self._position_active_view_type = active_view
                active_state = self._position_view_states.get(active_view)
                if isinstance(active_state, dict) and active_state.get(
                    "_position_view_state_initialized"
                ):
                    settings = self._active_position_settings_with_view_state(
                        settings,
                        active_view,
                        active_state,
                    )
                elif not self._position_view_state_switching:
                    initial_state = deepcopy(settings)
                    initial_state["_position_view_state_initialized"] = True
                    initial_state["position_active_view_type"] = active_view
                    self._position_view_states[active_view] = initial_state
            if self._analysis_name == "orientation":
                raw_states = settings.get("orientation_view_states")
                if isinstance(raw_states, dict):
                    migrated_states: dict[str, dict[str, Any]] = {}
                    migrated_legacy_state = False
                    for key, value in raw_states.items():
                        if not isinstance(value, dict):
                            continue
                        state = deepcopy(value)
                        if "heatmap_value_mode" not in state:
                            legacy_mode = state.get("heatmap_normalization_mode")
                            if legacy_mode is None and state.get("heatmap_normalize"):
                                legacy_mode = "global_probability"
                            legacy_mapping = {
                                "counts": "raw_counts",
                                "global_probability": "joint_probability_density",
                                "bulk_water_reference": "bulk_relative_enrichment",
                            }
                            if legacy_mode in legacy_mapping:
                                state["heatmap_value_mode"] = legacy_mapping[
                                    str(legacy_mode)
                                ]
                                migrated_legacy_state = True
                        state.pop("heatmap_normalize", None)
                        state.pop("heatmap_normalization_mode", None)
                        migrated_states[
                            self._normalize_orientation_view_type_id(key)
                        ] = state
                    self._orientation_view_states = migrated_states
                    if migrated_legacy_state:
                        warnings.warn(
                            "Legacy orientation heatmap normalization settings were "
                            "migrated to displayed-value modes. The next save writes only "
                            "the new fields.",
                            UserWarning,
                            stacklevel=2,
                        )
                elif not self._orientation_view_state_switching:
                    self._orientation_view_states = {}
                settings_mapping_for_view = self._coerce_settings_view_mapping(settings)
                active_view = self._normalize_orientation_view_type_id(
                    settings.get("orientation_active_view_type")
                    or (
                        getattr(settings_mapping_for_view, "view_type_id", None)
                        if settings_mapping_for_view is not None
                        else None
                    )
                    or PLOT_VIEW_1D_LINE
                )
                self._orientation_active_view_type = active_view
                active_state = self._orientation_view_states.get(active_view)
                if isinstance(active_state, dict) and active_state.get(
                    "_orientation_view_state_initialized"
                ):
                    settings = self._active_orientation_settings_with_view_state(
                        settings,
                        active_view,
                        active_state,
                    )
                elif not self._orientation_view_state_switching:
                    initial_state = deepcopy(settings)
                    initial_state["_orientation_view_state_initialized"] = True
                    initial_state["orientation_active_view_type"] = active_view
                    self._orientation_view_states[active_view] = initial_state
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
            self.legend_title_font.setText(
                _extract_dict_text(settings, key="legend_kwargs", nested_key="title_fontsize")
            )
            legend_font_size = settings.get("legend_font_size")
            if legend_font_size is None:
                legend_font_size = _extract_dict_text(
                    settings,
                    key="legend_kwargs",
                    nested_key="fontsize",
                )
            if legend_font_size in {None, ""}:
                legend_font_size = default_plot_font_sizes(self._resolved_base_font_size_value())[
                    "legend_font_size"
                ]
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
            x_minor_ticks_mode = "off"
            y_minor_ticks_mode = "off"
            x_tick_params: dict[str, Any] = {}
            y_tick_params: dict[str, Any] = {}
            if isinstance(tick_params_settings, dict):
                raw_x_tick_params = tick_params_settings.get("_x_tick_params")
                raw_y_tick_params = tick_params_settings.get("_y_tick_params")
                if isinstance(raw_x_tick_params, dict):
                    x_tick_params = dict(raw_x_tick_params)
                if isinstance(raw_y_tick_params, dict):
                    y_tick_params = dict(raw_y_tick_params)
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
                raw_x_minor = (
                    str(tick_params_settings.get("_x_minor_ticks_mode", raw_minor_mode))
                    .strip()
                    .lower()
                )
                raw_y_minor = (
                    str(tick_params_settings.get("_y_minor_ticks_mode", raw_minor_mode))
                    .strip()
                    .lower()
                )
                if raw_x_minor in _MINOR_TICKS_MODES:
                    x_minor_ticks_mode = raw_x_minor
                if raw_y_minor in _MINOR_TICKS_MODES:
                    y_minor_ticks_mode = raw_y_minor
            ticks_enabled = settings.get("ticks") is not False
            self._set_combo_value(
                self.x_ticks_mode,
                "on" if ticks_enabled and tick_axis_mode in {"x", "both"} else "off",
            )
            self._set_combo_value(
                self.y_ticks_mode,
                "on" if ticks_enabled and tick_axis_mode in {"y", "both"} else "off",
            )

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
            self.x_axis_scale.setText(str(settings.get("x_axis_scale") or "1.0"))
            self.x_axis_offset.setText(str(settings.get("x_axis_offset") or "0.0"))
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
            self.base_font_size.setText(
                _display_optional_positive_int(settings.get("font_size") or defaults.base_font_size)
            )
            self.font_color.setText(str(settings.get("font_color") or defaults.font_color))
            self.figure_facecolor.setText(
                _extract_dict_text(settings, key="figure_kwargs", nested_key="facecolor")
                or "#ffffff"
            )
            figure_alpha_value = settings.get("figure_alpha")
            if figure_alpha_value is None:
                figure_alpha_value = _extract_dict_text(
                    settings, key="figure_kwargs", nested_key="alpha"
                )
            self.figure_alpha.setText(
                "1.0" if figure_alpha_value in {None, ""} else str(figure_alpha_value)
            )
            self.title_font.setText(
                _display_optional_positive_int(
                    settings.get("title_font_size")
                    or default_plot_font_sizes(self._resolved_base_font_size_value())[
                        "title_font_size"
                    ]
                )
            )
            self.title_pad.setText(
                "6.0" if settings.get("title_pad") in {None, ""} else str(settings.get("title_pad"))
            )
            label_font_size = (
                settings.get("label_font_size")
                or default_plot_font_sizes(self._resolved_base_font_size_value())["label_font_size"]
            )
            tick_font_size = (
                settings.get("tick_font_size")
                or default_plot_font_sizes(self._resolved_base_font_size_value())["tick_font_size"]
            )
            self.x_label_font.setText(
                _display_optional_positive_int(settings.get("x_label_font_size") or label_font_size)
            )
            self.y_label_font.setText(
                _display_optional_positive_int(settings.get("y_label_font_size") or label_font_size)
            )
            self.x_tick_font.setText(
                _display_optional_positive_int(
                    settings.get("x_tick_font_size")
                    or x_tick_params.get("labelsize")
                    or tick_font_size
                )
            )
            self.y_tick_font.setText(
                _display_optional_positive_int(
                    settings.get("y_tick_font_size")
                    or y_tick_params.get("labelsize")
                    or tick_font_size
                )
            )
            self.line_width.setText(str(settings.get("line_width") or defaults.line_width))
            self.projection_line_width.setText(
                str(
                    settings.get("projection_line_width")
                    or settings.get("line_width")
                    or defaults.line_width
                )
            )
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
                self.marker_type,
                _extract_dict_text(settings, key="line_kwargs", nested_key="marker") or "o",
            )
            marker_color = _extract_dict_text(
                settings,
                key="line_kwargs",
                nested_key="markerfacecolor",
            ) or _extract_dict_text(settings, key="line_kwargs", nested_key="markeredgecolor")
            self.marker_color.setText(marker_color)

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
                self.x_tick_direction,
                str(
                    x_tick_params.get("direction")
                    or _extract_dict_value(
                        settings, key="tick_params_kwargs", nested_key="direction"
                    )
                    or "out"
                ),
            )
            self._set_combo_value(
                self.y_tick_direction,
                str(
                    y_tick_params.get("direction")
                    or _extract_dict_value(
                        settings, key="tick_params_kwargs", nested_key="direction"
                    )
                    or "out"
                ),
            )
            self.x_tick_length.setText(
                str(
                    x_tick_params.get("length")
                    or _extract_dict_text(settings, key="tick_params_kwargs", nested_key="length")
                    or ""
                )
            )
            self.y_tick_length.setText(
                str(
                    y_tick_params.get("length")
                    or _extract_dict_text(settings, key="tick_params_kwargs", nested_key="length")
                    or ""
                )
            )
            self._refresh_font_size_placeholders()
            self._last_resolved_base_font_size = self._resolved_base_font_size_value()
            self.x_tick_width.setText(
                str(
                    x_tick_params.get("width")
                    or _extract_dict_text(settings, key="tick_params_kwargs", nested_key="width")
                    or ""
                )
            )
            self.y_tick_width.setText(
                str(
                    y_tick_params.get("width")
                    or _extract_dict_text(settings, key="tick_params_kwargs", nested_key="width")
                    or ""
                )
            )
            self._set_combo_value(self.x_minor_ticks_mode, x_minor_ticks_mode)
            self._set_combo_value(self.y_minor_ticks_mode, y_minor_ticks_mode)

            if hasattr(self, "analysis_species"):
                self.analysis_species.setText(str(settings.get("species") or ""))
            if hasattr(self, "analysis_axis"):
                self._set_combo_value(self.analysis_axis, str(settings.get("axis") or ""))
            settings_mapping = self._coerce_settings_view_mapping(settings)
            if hasattr(self, "density_view_type"):
                density_view_type_id = (
                    canonical_plot_view_id(getattr(settings_mapping, "view_type_id", None))
                    if settings_mapping is not None
                    else PLOT_VIEW_1D_LINE
                ) or PLOT_VIEW_1D_LINE
                self._set_combo_value(
                    self.density_view_type,
                    _DENSITY_VIEW_TYPE_LABEL_BY_ID.get(
                        density_view_type_id,
                        _DENSITY_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_1D_LINE],
                    ),
                )
            if hasattr(self, "density_x_mode"):
                density_options: dict[str, object]
                if settings_mapping is not None:
                    try:
                        density_options = density_view_mapping_to_plot_options(settings_mapping)
                    except ValueError:
                        # Compatibility fallback is only used when no
                        # usable mapping-native state can be restored.
                        density_options = {
                            "x_mode": str(settings.get("x_mode") or "distance"),
                            "quantity": str(settings.get("quantity") or "mass"),
                        }
                else:
                    density_options = {
                        "x_mode": str(settings.get("x_mode") or "distance"),
                        "quantity": str(settings.get("quantity") or "mass"),
                    }
                self._set_combo_value(
                    self.density_x_mode,
                    self._density_x_mode_display_label(
                        str(density_options.get("x_mode") or "distance"),
                        axis=str(settings.get("axis") or ""),
                    ),
                )
            if hasattr(self, "density_2d_x_axis"):
                self._set_combo_value(
                    self.density_2d_x_axis,
                    self._density_x_mode_display_label(
                        str(settings.get("density_2d_x_axis") or "x"),
                        axis=None,
                    ),
                )
            if hasattr(self, "density_2d_y_axis"):
                self._set_combo_value(
                    self.density_2d_y_axis,
                    self._density_x_mode_display_label(
                        str(settings.get("density_2d_y_axis") or "y"),
                        axis=None,
                    ),
                )
            if hasattr(self, "_density_filter_widgets"):
                for axis_id, widgets in self._density_filter_widgets.items():
                    lower_widget, upper_widget = widgets
                    lower = settings.get(f"density_filter_{axis_id}_min")
                    upper = settings.get(f"density_filter_{axis_id}_max")
                    lower_text = (
                        self._density_axis_range_text(axis_id, 0)
                        if lower is None
                        else str(lower)
                    )
                    upper_text = (
                        self._density_axis_range_text(axis_id, 1)
                        if upper is None
                        else str(upper)
                    )
                    lower_widget.setText(lower_text)
                    upper_widget.setText(upper_text)
            if hasattr(self, "density_quantity"):
                self._set_combo_value(
                    self.density_quantity,
                    str(
                        (
                            density_options.get("quantity")
                            if "density_options" in locals()
                            else settings.get("quantity")
                        )
                        or "mass"
                    ),
                )
            if hasattr(self, "_density_species_checkboxes"):
                self._apply_density_enabled_species_settings(
                    settings.get("density_enabled_species")
                )
                self._sync_density_species_selection_for_view_type(record_snapshot=False)
            if hasattr(self, "coord_species_a"):
                self._set_combo_value(
                    self.coord_species_a,
                    self._profile_filter_display_value(settings.get("species_a")),
                )
            if hasattr(self, "coord_species_b"):
                species_a_value = settings.get("species_a")
                species_b_value = settings.get("species_b")
                self._set_combo_items(
                    self.coord_species_b,
                    self._coordination_species_b_choices(
                        None if species_a_value in {None, ""} else str(species_a_value)
                    ),
                    preferred_value=self._profile_filter_display_value(species_b_value),
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
            if hasattr(self, "position_view_type"):
                position_mapping = settings_mapping
                if position_mapping is None:
                    # Position still has a broader compatibility restore surface than
                    # the other migrated analyses, so keep one explicit fallback.
                    position_mapping = position_plot_options_to_view_mapping(
                        component=str(settings.get("component") or "distance"),
                        time_axis=str(settings.get("time_axis") or "ps"),
                        map_color=str(settings.get("map_color") or "distance"),
                        projection_x=(
                            None
                            if settings.get("projection_x") is None
                            else str(settings.get("projection_x"))
                        ),
                        projection_y=(
                            None
                            if settings.get("projection_y") is None
                            else str(settings.get("projection_y"))
                        ),
                        projection_value=(
                            None
                            if settings.get("projection_value") is None
                            else str(settings.get("projection_value"))
                        ),
                        projection_render_mode=(
                            None
                            if settings.get("projection_render_mode") is None
                            else str(settings.get("projection_render_mode"))
                        ),
                        projection_filter_min=settings.get("projection_filter_min"),
                        projection_filter_max=settings.get("projection_filter_max"),
                        xy_z_distance_max=settings.get("xy_z_distance_max"),
                    )
                self._apply_position_mapping_controls(
                    position_mapping
                )
            if hasattr(self, "coordination_component"):
                coordination_options: dict[str, object]
                if settings_mapping is not None:
                    try:
                        coordination_options = coordination_view_mapping_to_plot_options(settings_mapping)
                    except ValueError:
                        # Compatibility fallback is only used when no
                        # usable mapping-native state can be restored.
                        coordination_options = {
                            "component": str(settings.get("component") or "distance"),
                            "time_axis": str(settings.get("time_axis") or "ps"),
                        }
                else:
                    coordination_options = {
                        "component": str(settings.get("component") or "distance"),
                        "time_axis": str(settings.get("time_axis") or "ps"),
                    }
                coordination_component = str(coordination_options.get("component") or "distance")
                coordination_view_type_id = (
                    PLOT_VIEW_2D_HEATMAP
                    if coordination_component == "time-distance"
                    else PLOT_VIEW_1D_LINE
                )
                self._set_combo_value(
                    self.coordination_component,
                    _COORDINATION_VIEW_TYPE_LABEL_BY_ID.get(
                        coordination_view_type_id,
                        _COORDINATION_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_1D_LINE],
                    ),
                )
            if hasattr(self, "coordination_line_x_quantity"):
                line_component = (
                    coordination_component
                    if "coordination_component" in locals()
                    else str(settings.get("component") or "distance")
                )
                if line_component == "time-distance":
                    line_component = "time"
                self._set_combo_value(
                    self.coordination_line_x_quantity,
                    _COORDINATION_LINE_X_QUANTITY_LABEL_BY_BACKEND.get(
                        line_component,
                        _COORDINATION_LINE_X_QUANTITY_LABEL_BY_BACKEND["distance"],
                    ),
                )
            if hasattr(self, "coordination_time_axis"):
                self._set_combo_value(
                    self.coordination_time_axis,
                    str(
                        (
                            coordination_options.get("time_axis")
                            if "coordination_options" in locals()
                            else settings.get("time_axis")
                        )
                        or "ps"
                    ),
                )
            if hasattr(self, "orientation_component"):
                orientation_options: dict[str, object]
                if settings_mapping is not None:
                    try:
                        orientation_options = orientation_view_mapping_to_plot_options(settings_mapping)
                    except ValueError:
                        # Compatibility fallback is only used when no
                        # usable mapping-native state can be restored.
                        orientation_options = {
                            "component": str(settings.get("component") or "average"),
                            "angle": str(settings.get("angle") or "polar"),
                        }
                else:
                    orientation_options = {
                        "component": str(settings.get("component") or "average"),
                        "angle": str(settings.get("angle") or "polar"),
                    }
                self._set_orientation_view_type_from_component(
                    str(orientation_options.get("component") or "average")
                )
                self.orientation_component.blockSignals(True)
                try:
                    self._set_orientation_line_quantity(
                        str(orientation_options.get("component") or "average")
                    )
                finally:
                    self.orientation_component.blockSignals(False)
            if hasattr(self, "orientation_angle"):
                self._set_combo_value(
                    self.orientation_angle,
                    str(
                        (
                            orientation_options.get("angle")
                            if "orientation_options" in locals()
                            else settings.get("angle")
                        )
                        or "polar"
                    ),
                )
            if hasattr(self, "orientation_line_x_axis"):
                self._set_combo_value(
                    self.orientation_line_x_axis,
                    self._density_x_mode_display_label(
                        str(settings.get("orientation_line_x_axis") or "distance"),
                        axis=None,
                    ),
                )
            if hasattr(self, "orientation_heatmap_x_axis"):
                self._set_combo_value(
                    self.orientation_heatmap_x_axis,
                    self._density_x_mode_display_label(
                        str(settings.get("orientation_heatmap_x_axis") or "x"),
                        axis=None,
                    ),
                )
            if hasattr(self, "orientation_heatmap_y_axis"):
                self._set_combo_value(
                    self.orientation_heatmap_y_axis,
                    self._density_x_mode_display_label(
                        str(settings.get("orientation_heatmap_y_axis") or "y"),
                        axis=None,
                    ),
                )
            if hasattr(self, "_orientation_filter_widgets"):
                for axis_id, widgets in self._orientation_filter_widgets.items():
                    lower_widget, upper_widget = widgets
                    lower = settings.get(f"orientation_filter_{axis_id}_min")
                    upper = settings.get(f"orientation_filter_{axis_id}_max")
                    lower_text = (
                        self._orientation_axis_range_text(axis_id, 0)
                        if lower is None
                        else str(lower)
                    )
                    upper_text = (
                        self._orientation_axis_range_text(axis_id, 1)
                        if upper is None
                        else str(upper)
                    )
                    lower_widget.setText(lower_text)
                    upper_widget.setText(upper_text)
            if hasattr(self, "potential_view_type"):
                potential_options: dict[str, object]
                if settings_mapping is not None:
                    try:
                        potential_options = potential_view_mapping_to_plot_options(settings_mapping)
                    except ValueError:
                        # Compatibility fallback is only used when no
                        # usable mapping-native state can be restored.
                        potential_options = {
                            "view_type": "line_1d",
                            "y_quantity": settings.get("y_quantity"),
                        }
                else:
                    potential_options = {
                        "view_type": "line_1d",
                        "y_quantity": settings.get("y_quantity"),
                    }
                self._set_combo_value(
                    self.potential_view_type,
                    _POTENTIAL_VIEW_TYPE_LABEL_BY_ID.get(
                        canonical_plot_view_id(
                            str(potential_options.get("view_type") or PLOT_VIEW_1D_LINE)
                        ),
                        _POTENTIAL_VIEW_TYPE_LABEL_BY_ID[PLOT_VIEW_1D_LINE],
                    ),
                )
                series_token = str(
                    potential_options.get("y_quantity")
                    or ("summary" if str(potential_options.get("standard_plot") or "").strip().lower() == "summary" else "summary")
                )
                self._set_combo_value(
                    self.potential_series_mode,
                    _POTENTIAL_SERIES_LABEL_BY_ID.get(
                        series_token,
                        _POTENTIAL_SERIES_LABEL_BY_ID["summary"],
                    ),
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
            if hasattr(self, "heatmap_value_mode"):
                raw_mode = settings.get("heatmap_value_mode")
                if raw_mode is None:
                    legacy_mode = settings.get("heatmap_normalization_mode")
                    legacy_settings_present = (
                        legacy_mode is not None
                        or bool(settings.get("heatmap_normalize", False))
                    )
                    if legacy_mode is None and bool(settings.get("heatmap_normalize", False)):
                        legacy_mode = "global_probability"
                    legacy_mapping = {
                        "counts": "raw_counts",
                        "global_probability": "joint_probability_density",
                        "bulk_water_reference": "bulk_relative_enrichment",
                    }
                    raw_mode = legacy_mapping.get(str(legacy_mode), "raw_counts")
                    if legacy_settings_present:
                        warnings.warn(
                            "This plot profile used legacy heatmap normalization. It was "
                            "migrated to the nearest displayed-value representation; the "
                            "updated semantics will be saved using only the new fields.",
                            UserWarning,
                            stacklevel=2,
                        )
                self._set_combo_value(
                    self.heatmap_value_mode,
                    _HEATMAP_VALUE_LABEL_BY_MODE.get(
                        str(raw_mode).strip().lower(),
                        _HEATMAP_VALUE_LABEL_BY_MODE["raw_counts"],
                    ),
                )
            if hasattr(self, "heatmap_bulk_reference_mode"):
                bulk_mode = str(
                    settings.get("heatmap_bulk_reference_mode") or "auto"
                ).strip().lower()
                self._set_combo_value(
                    self.heatmap_bulk_reference_mode,
                    "Manual" if bulk_mode == "manual" else "Automatic",
                )
                bulk_min = settings.get("heatmap_bulk_min")
                bulk_max = settings.get("heatmap_bulk_max")
                self.heatmap_bulk_min.setText(
                    "" if bulk_min is None else str(bulk_min)
                )
                self.heatmap_bulk_max.setText(
                    "" if bulk_max is None else str(bulk_max)
                )
            if hasattr(self, "heatmap_log_scale"):
                self._set_combo_value(
                    self.heatmap_log_scale,
                    "Logarithmic"
                    if bool(settings.get("heatmap_log_scale", False))
                    else "Linear",
                )
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
            self._auto_display_note = str(settings.get("_auto_display_note") or "").strip()
            self.x_bin_width.setText(str(settings.get("x_bin_width") or ""))
            self._set_combo_value(self.x_bin_reducer, str(settings.get("x_bin_reducer") or "mean"))
            self.min_bin_points.setText(str(settings.get("min_bin_points") or ""))
            if hasattr(self, "y_bin_width"):
                self.y_bin_width.setText(str(settings.get("y_bin_width") or ""))
            if hasattr(self, "y_bin_reducer"):
                self._set_combo_value(
                    self.y_bin_reducer, str(settings.get("y_bin_reducer") or "mean")
                )
            self._apply_density_default_bin_width_texts()
            self._initialize_annotation_data(settings)
            self._initialize_series_data(settings)
            self._initialize_normalization_data(settings)
            self._update_potential_summary_panel(settings)
            self._update_density_contract_summary()
            self._update_coordination_contract_summary()
            self._update_orientation_contract_summary()
            self._update_potential_contract_summary()
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
                self._set_synced_field_mode(key, synced_modes.get(key, "auto"))
            self._apply_preview_state_to_synced_fields(settings)

        def _is_orientation_heatmap_mode(self) -> bool:
            if self._analysis_name != "orientation":
                return False
            return (
                canonical_plot_view_id(self._current_orientation_mapping().view_type_id)
                == PLOT_VIEW_2D_HEATMAP
            )

        def _refresh_widget_states(self, *_unused: object) -> None:
            if all(
                hasattr(self, name)
                for name in (
                    "title_font",
                    "x_label_font",
                    "y_label_font",
                    "x_tick_font",
                    "y_tick_font",
                    "legend_font",
                    "base_font_size",
                )
            ):
                self._refresh_font_size_placeholders()
            title_enabled = self._synced_field_mode("title") != "off"
            x_label_enabled = self._synced_field_mode("x_label") != "off"
            y_label_enabled = self._synced_field_mode("y_label") != "off"
            legend_enabled = self.legend_mode.currentText().strip().lower() != "off"
            legend_title_present = (
                bool(self.legend_title.text().strip()) if hasattr(self, "legend_title") else False
            )
            grid_enabled = self.grid_mode.currentText().strip().lower() != "off"
            x_ticks_enabled = self.x_ticks_mode.currentText().strip().lower() != "off"
            y_ticks_enabled = self.y_ticks_mode.currentText().strip().lower() != "off"
            markers_enabled = self.markers_mode.currentText().strip().lower() != "off"
            integration_enabled = (
                hasattr(self, "integration_mode")
                and self.integration_mode.currentText().strip().lower() != "off"
            )
            integration_custom_color = (
                integration_enabled
                and hasattr(self, "integration_color_mode")
                and self.integration_color_mode.currentText().strip().lower() == "custom"
            )
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
            position_xy_projection = self._current_position_is_projection_view()
            coordination_mapping = (
                self._current_coordination_mapping()
                if self._analysis_name == "coordination" and hasattr(self, "coordination_component")
                else None
            )
            coordination_view_type_id = (
                str(getattr(coordination_mapping, "view_type_id", "") or "").strip().lower()
                if coordination_mapping is not None
                else ""
            )
            coordination_time_distance = (
                canonical_plot_view_id(coordination_view_type_id) == PLOT_VIEW_2D_HEATMAP
            )
            coordination_line_time = (
                coordination_mapping is not None
                and canonical_plot_view_id(coordination_view_type_id) == PLOT_VIEW_1D_LINE
                and str(getattr(coordination_mapping, "x", "") or "").strip().lower()
                in {"time_ps", "time_fs", "step", "frame_index"}
            )
            layer_caps = self._current_layer_capabilities()
            figure_caps = self._current_figure_capabilities()
            fit_supported = bool(layer_caps.show_fit_editor)
            cumulative_supported = bool(layer_caps.show_cumulative_editor)
            selected_group = layer_caps.layer_kind == "group"
            heatmap_layer_mode = layer_caps.plot_family == "heatmap"

            for widget in self._title_detail_widgets:
                widget.setVisible(title_enabled)
            for widget in self._x_label_detail_widgets:
                widget.setVisible(x_label_enabled)
            for widget in self._y_label_detail_widgets:
                widget.setVisible(y_label_enabled)
            self._set_rows_visible(self._title_rows, title_enabled)
            self._set_rows_visible(self._legend_rows, legend_enabled)
            if hasattr(self, "legend_title_font"):
                self._set_form_row_visible(
                    self._legend_rows[0][0],
                    self.legend_title_font,
                    legend_enabled and legend_title_present,
                )
            self._set_rows_visible(self._grid_rows, grid_enabled)
            self._set_rows_visible(self._marker_rows, markers_enabled)
            self._set_rows_visible(self._integration_rows, integration_enabled)
            if self._integration_custom_color_row is not None:
                self._set_form_row_visible(
                    self._integration_custom_color_row[0],
                    self._integration_custom_color_row[1],
                    integration_custom_color,
                )
            if getattr(self, "_integration_summary_label", None) is not None:
                self._integration_summary_label.setVisible(integration_enabled)
            border_custom = (
                hasattr(self, "axes_border_mode")
                and self.axes_border_mode.currentText().strip().lower() == "custom"
            )
            self._set_rows_visible(self._border_custom_rows, border_custom)
            self._set_rows_visible(self._x_ticks_rows, x_ticks_enabled)
            self._set_rows_visible(self._y_ticks_rows, y_ticks_enabled)
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
            for form, field in self._x_ticks_rows:
                self._set_form_row_enabled(
                    form,
                    field,
                    x_ticks_enabled,
                    disabled_reason="ticks are currently off.",
                )
            for form, field in self._y_ticks_rows:
                self._set_form_row_enabled(
                    form,
                    field,
                    y_ticks_enabled,
                    disabled_reason="ticks are currently off.",
                )

            if self._series_visibility_group is not None:
                self._series_visibility_group.setVisible(layer_caps.show_visibility_label)
            if self._series_show_in_legend_row is not None:
                self._set_form_row_visible(
                    self._series_show_in_legend_row[0],
                    self._series_show_in_legend_row[1],
                    layer_caps.show_visibility_label and not heatmap_layer_mode,
                )
            if self._series_show_raw_line_row is not None:
                self._set_form_row_visible(
                    self._series_show_raw_line_row[0],
                    self._series_show_raw_line_row[1],
                    layer_caps.show_visibility_label and not heatmap_layer_mode,
                )
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
            if self._series_integration_group is not None:
                self._series_integration_group.setVisible(layer_caps.show_integration)
            if self._series_metadata_group is not None:
                self._series_metadata_group.setVisible(layer_caps.show_metadata)
            self._update_selected_layer_card(self._series_active_index)
            if self._figure_legend_section is not None:
                self._figure_legend_section.setVisible(figure_caps.show_legend)
            if self._figure_lines_section is not None:
                self._figure_lines_section.setVisible(figure_caps.show_lines)
            if self._figure_lines_group is not None:
                self._figure_lines_group.setVisible(figure_caps.show_lines)
            if self._figure_heatmap_group is not None:
                self._figure_heatmap_group.setVisible(figure_caps.show_heatmap)
            orientation_frequency_heatmap = (
                self._analysis_name == "orientation"
                and self._is_orientation_heatmap_mode()
            )
            if self._heatmap_value_group is not None:
                self._heatmap_value_group.setVisible(orientation_frequency_heatmap)
            selected_heatmap_value_mode = "raw_counts"
            if hasattr(self, "heatmap_value_mode"):
                selected_heatmap_value_mode = _HEATMAP_VALUE_MODE_BY_LABEL.get(
                    self.heatmap_value_mode.currentText().strip(),
                    "raw_counts",
                )
            bulk_relative = (
                orientation_frequency_heatmap
                and selected_heatmap_value_mode == "bulk_relative_enrichment"
            )
            self._set_rows_visible(self._heatmap_bulk_rows, bulk_relative)
            bulk_manual = (
                bulk_relative
                and hasattr(self, "heatmap_bulk_reference_mode")
                and self.heatmap_bulk_reference_mode.currentText().strip().lower()
                == "manual"
            )
            self._set_rows_visible(self._heatmap_bulk_manual_rows, bulk_manual)
            if self._heatmap_trajectory_group is not None:
                self._heatmap_trajectory_group.setVisible(
                    position_xy_projection
                    and self._current_position_uses_continuous_color()
                )
            self._update_heatmap_value_summary()
            if self._figure_colorbar_group is not None:
                self._figure_colorbar_group.setVisible(figure_caps.show_colorbar)
            if self._figure_heatmap_section is not None:
                self._figure_heatmap_section.setVisible(figure_caps.show_heatmap)
            if self._position_projection_stroke_row is not None:
                self._set_form_row_visible(
                    self._position_projection_stroke_row[0],
                    self._position_projection_stroke_row[1],
                    position_xy_projection
                    and self._current_position_uses_continuous_color(),
                )
            self._set_rows_visible(
                self._x_axis_transform_rows,
                figure_caps.show_axis_transforms,
            )
            self._set_rows_visible(
                self._advanced_legend_kwargs_rows,
                figure_caps.show_advanced_legend,
            )
            self._set_rows_visible(
                self._advanced_line_kwargs_rows,
                figure_caps.show_advanced_lines,
            )

            if self._position_mapping_x_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_x_row[0],
                    self._position_mapping_x_row[1],
                    True,
                )
            if self._position_mapping_y_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_y_row[0],
                    self._position_mapping_y_row[1],
                    True,
                )
            if self._position_mapping_render_mode_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_render_mode_row[0],
                    self._position_mapping_render_mode_row[1],
                    position_xy_projection,
                )
            if self._position_mapping_value_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_value_row[0],
                    self._position_mapping_value_row[1],
                    position_xy_projection,
                )
            if self._position_mapping_filter_min_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_filter_min_row[0],
                    self._position_mapping_filter_min_row[1],
                    position_xy_projection,
                )
            if self._position_mapping_filter_max_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_filter_max_row[0],
                    self._position_mapping_filter_max_row[1],
                    position_xy_projection,
                )
            if self._position_mapping_split_by_row is not None:
                self._set_form_row_visible(
                    self._position_mapping_split_by_row[0],
                    self._position_mapping_split_by_row[1],
                    True,
                )
            if self._coordination_time_axis_row is not None:
                coordination_time_axis_row = self._coordination_time_axis_row
                assert coordination_time_axis_row is not None
                self._set_form_row_visible(
                    coordination_time_axis_row[0],
                    coordination_time_axis_row[1],
                    coordination_time_distance or coordination_line_time,
                )
            if self._coordination_line_x_quantity_row is not None:
                coordination_line_x_quantity_row = self._coordination_line_x_quantity_row
                assert coordination_line_x_quantity_row is not None
                self._set_form_row_visible(
                    coordination_line_x_quantity_row[0],
                    coordination_line_x_quantity_row[1],
                    not coordination_time_distance,
                )
            density_heatmap_mode = (
                self._analysis_name == "density" and self._is_density_heatmap_mode()
            )
            if self._analysis_name == "density":
                self._sync_density_species_selection_for_view_type()
                self._set_rows_visible(self._density_mapping_1d_rows, not density_heatmap_mode)
                self._set_rows_visible(self._density_mapping_2d_rows, density_heatmap_mode)
                for axis_id, row in self._density_filter_rows.items():
                    self._set_form_row_visible(row[0], row[1], True)
                if hasattr(self, "density_quantity"):
                    label = None
                    try:
                        parent_layout = self.density_quantity.parentWidget().layout()
                        if isinstance(parent_layout, QFormLayout):
                            label = parent_layout.labelForField(self.density_quantity)
                    except Exception:
                        label = None
                    if isinstance(label, QLabel):
                        label.setText("Color quantity" if density_heatmap_mode else "Y quantity")
            if hasattr(self, "density_x_mode"):
                self.density_x_mode.setEnabled(not density_heatmap_mode)
            for widget_name in ("density_2d_x_axis", "density_2d_y_axis"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setEnabled(density_heatmap_mode)
            if self._data_transform_group is not None and self._analysis_name == "position":
                self._data_transform_group.setVisible(not position_xy_projection)
            self._update_position_contract_summary()
            self._update_density_contract_summary()
            self._update_coordination_contract_summary()
            self._update_orientation_contract_summary()
            self._update_potential_contract_summary()
            if self._data_transform_group is not None and self._analysis_name == "coordination":
                self._data_transform_group.setVisible(not coordination_time_distance)
            if self._data_transform_group is not None and self._analysis_name == "potential":
                self._data_transform_group.setVisible(True)
            if hasattr(self, "potential_series_mode"):
                self.potential_series_mode.setEnabled(True)
                self._apply_widget_tooltip(self.potential_series_mode, disabled_reason=None)
            # ── orientation heatmap mode ──────────────────────────────
            is_heatmap = self._is_orientation_heatmap_mode()
            if self._analysis_name == "orientation":
                orientation_line_component = "average"
                if hasattr(self, "orientation_component"):
                    orientation_line_component = _ORIENTATION_LINE_QUANTITY_BACKEND_BY_LABEL.get(
                        self.orientation_component.currentText().strip(),
                        "average",
                    )
                orientation_grid_line_controls = (
                    not is_heatmap and orientation_line_component == "average"
                )
                if self._orientation_line_quantity_row is not None:
                    self._set_form_row_visible(
                        self._orientation_line_quantity_row[0],
                        self._orientation_line_quantity_row[1],
                        not is_heatmap,
                    )
                if self._orientation_line_x_axis_row is not None:
                    self._set_form_row_visible(
                        self._orientation_line_x_axis_row[0],
                        self._orientation_line_x_axis_row[1],
                        not is_heatmap,
                    )
                    if hasattr(self, "orientation_line_x_axis"):
                        self.orientation_line_x_axis.setEnabled(orientation_grid_line_controls)
                        self._apply_widget_tooltip(
                            self.orientation_line_x_axis,
                            disabled_reason=(
                                "Only Mean orientation supports alternate orientation grid axes."
                                if not is_heatmap and not orientation_grid_line_controls
                                else None
                            ),
                        )
                self._set_rows_visible(self._orientation_mapping_2d_rows, False)
                for row in self._orientation_filter_rows.values():
                    self._set_form_row_visible(row[0], row[1], orientation_grid_line_controls)
                for widget_name in ("orientation_heatmap_x_axis", "orientation_heatmap_y_axis"):
                    widget = getattr(self, widget_name, None)
                    if widget is not None:
                        widget.setEnabled(False)
                if hasattr(self, "orientation_angle"):
                    label = None
                    try:
                        parent_layout = self.orientation_angle.parentWidget().layout()
                        if isinstance(parent_layout, QFormLayout):
                            label = parent_layout.labelForField(self.orientation_angle)
                    except Exception:
                        label = None
                    if isinstance(label, QLabel):
                        label.setText("Angle quantity")
            two_dimensional_binning = is_heatmap or density_heatmap_mode
            if self._data_transform_group is not None and self._analysis_name == "orientation":
                self._data_transform_group.setEnabled(True)
                self._data_transform_group.setToolTip("")
            if self._y_bin_width_row is not None:
                self._set_form_row_visible(
                    self._y_bin_width_row[0],
                    self._y_bin_width_row[1],
                    two_dimensional_binning,
                )
            if self._y_bin_reducer_row is not None:
                y_rebin_enabled = (
                    two_dimensional_binning
                    and bool(getattr(self, "y_bin_width", None))
                    and bool(self.y_bin_width.text().strip())
                )
                self._set_form_row_visible(
                    self._y_bin_reducer_row[0],
                    self._y_bin_reducer_row[1],
                    y_rebin_enabled and not orientation_frequency_heatmap,
                )
                if two_dimensional_binning:
                    self._set_form_row_enabled(
                        self._y_bin_reducer_row[0],
                        self._y_bin_reducer_row[1],
                        y_rebin_enabled,
                        disabled_reason="set a Y bin size first.",
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
                    rebin_enabled and not orientation_frequency_heatmap,
                )
                self._set_form_row_enabled(
                    self._x_bin_reducer_row[0],
                    self._x_bin_reducer_row[1],
                    rebin_enabled,
                    disabled_reason="set an X bin size first.",
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
            normalization_actions_visible = layer_caps.show_normalization and not is_heatmap
            for widget in (
                self._normalization_actions_widget,
                self._normalization_hint_label,
            ):
                if widget is not None:
                    if normalization_actions_visible == norm_enabled:
                        widget.setVisible(norm_enabled)
                    else:
                        widget.setVisible(normalization_actions_visible)
            if self._normalization_copy_button is not None:
                normalization_copy_enabled = (
                    layer_caps.show_normalization
                    and not is_heatmap
                    and not self._series_active_is_fit_child
                    and not self._series_active_is_cumulative_child
                )
                normalization_copy_disabled_reason = None
                if not layer_caps.show_normalization or is_heatmap:
                    normalization_copy_disabled_reason = (
                        "Normalization is unavailable for the current layer."
                    )
                elif self._series_active_is_fit_child:
                    normalization_copy_disabled_reason = (
                        "Normalization is edited on the base series only."
                    )
                elif self._series_active_is_cumulative_child:
                    normalization_copy_disabled_reason = (
                        "Normalization is edited on the base series only."
                    )
                self._normalization_copy_button.setEnabled(normalization_copy_enabled)
                self._apply_widget_tooltip(
                    self._normalization_copy_button,
                    disabled_reason=normalization_copy_disabled_reason,
                )
            if self._normalization_group is not None and not (
                self._analysis_name == "orientation" and is_heatmap
            ):
                self._normalization_group.setEnabled(layer_caps.show_normalization)
                self._normalization_group.setToolTip("")
            annotation_selected = bool(self._annotations_data)
            series_selected = bool(self._series_descriptors_data)
            if getattr(self, "_series_duplicate_button", None) is not None:
                self._series_duplicate_button.setVisible(not heatmap_layer_mode)
                self._series_duplicate_button.setEnabled(series_selected and not heatmap_layer_mode)
            if getattr(self, "_series_add_group_button", None) is not None:
                self._series_add_group_button.setVisible(not heatmap_layer_mode)
                self._series_add_group_button.setEnabled(
                    bool(self._group_member_candidate_indices()) and not heatmap_layer_mode
                )
            delete_button = getattr(self, "_series_delete_button", None)
            if delete_button is not None:
                delete_button.setVisible(not heatmap_layer_mode)
                can_delete_series = (
                    series_selected
                    and not heatmap_layer_mode
                    and not self._series_active_is_fit_child
                    and not self._series_active_is_cumulative_child
                    and self._series_is_generated(self._series_active_index)
                )
                delete_button.setEnabled(can_delete_series)
            for button in (
                getattr(self, "_annotation_duplicate_button", None),
                getattr(self, "_annotation_delete_button", None),
            ):
                if button is not None:
                    button.setEnabled(annotation_selected)
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
                        "Uncertainty settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else "Grouped series do not support uncertainty overlays."
                        if selected_group
                        else None
                        if error_supported
                        else "Uncertainty overlays are only available for 1-D line plots."
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
                        "Uncertainty settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else "Grouped series do not support uncertainty overlays."
                        if selected_group
                        else error_availability.reason
                        if error_supported and not error_availability.selector_enabled
                        else None
                        if error_supported
                        else "Uncertainty overlays are only available for 1-D line plots."
                    ),
                )
            if hasattr(self, "_series_error_style"):
                self._series_error_style.setEnabled(error_controls_available and error_active)
                self._apply_widget_tooltip(
                    self._series_error_style,
                    disabled_reason=(
                        "Uncertainty settings are edited on the base series only."
                        if self._series_active_is_fit_child
                        or self._series_active_is_cumulative_child
                        else "Grouped series do not support uncertainty overlays."
                        if selected_group
                        else None
                        if error_supported
                        else "Uncertainty overlays are only available for 1-D line plots."
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
                            "Uncertainty settings are edited on the base series only."
                            if self._series_active_is_fit_child
                            or self._series_active_is_cumulative_child
                            else "Grouped series do not support uncertainty overlays."
                            if selected_group
                            else None
                            if error_supported
                            else "Uncertainty overlays are only available for 1-D line plots."
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
            polynomial_selected = fit_type == "polynomial"
            for form, field in self._series_fit_detail_rows:
                visible = fit_active
                if field is getattr(self, "_series_fit_degree", None):
                    visible = fit_active and polynomial_selected
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
                    )
                    self._apply_widget_tooltip(
                        widget,
                        disabled_reason=(
                            "fit settings are edited on the base series only."
                            if self._series_active_is_fit_child
                            or self._series_active_is_cumulative_child
                            else "turn fitting on first."
                            if not fit_active
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
            if self._preview_button is not None:
                auto_update_enabled = (
                    self._auto_preview_checkbox is not None
                    and self._auto_preview_checkbox.isChecked()
                )
                self._preview_button.setEnabled(
                    _preview_button_enabled(
                        auto_update_enabled=auto_update_enabled,
                        preview_loading=self._preview_loading,
                    )
                )
            self._update_normalization_warning()
            self._update_series_error_summary(self._series_active_index)
            self._update_integration_summary()
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
            self._update_data_export_summary()
            self._refresh_shell_state()

        def _collect_settings(self) -> dict[str, Any]:
            self._persist_active_series_editor()
            self._persist_active_annotation_editor()
            self._validate_series_state_lengths()
            # Plot Studio currently serializes one settings payload for
            # preview/save that includes data selection, view mapping,
            # per-series state, and pure style fields together.
            resolved_view_mapping: PlotViewMapping | None = None

            def _synced_text(key: str, widget: QLineEdit) -> str | None:
                mode = self._synced_field_mode(key)
                if mode == "off":
                    return ""
                if mode != "manual":
                    return None
                return _explicit_text(widget.text())

            def _synced_float(key: str, widget: QLineEdit, *, field_name: str) -> float | None:
                if self._synced_field_mode(key) != "manual":
                    return None
                return _optional_float(widget.text(), field_name=field_name)

            def _synced_float_list(
                key: str,
                widget: QLineEdit,
                *,
                field_name: str,
            ) -> list[float] | None:
                if self._synced_field_mode(key) != "manual":
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
            x_label_font_size_value = _optional_positive_int_or_none(
                self.x_label_font.text(), field_name="x-label-font-size"
            )
            y_label_font_size_value = _optional_positive_int_or_none(
                self.y_label_font.text(), field_name="y-label-font-size"
            )
            shared_label_font_size_value = (
                x_label_font_size_value
                if x_label_font_size_value is not None
                and x_label_font_size_value == y_label_font_size_value
                else None
            )
            x_tick_font_size_value = _optional_positive_int_or_none(
                self.x_tick_font.text(), field_name="x-tick-font-size"
            )
            y_tick_font_size_value = _optional_positive_int_or_none(
                self.y_tick_font.text(), field_name="y-tick-font-size"
            )
            shared_tick_font_size_value = (
                x_tick_font_size_value
                if x_tick_font_size_value is not None
                and x_tick_font_size_value == y_tick_font_size_value
                else None
            )

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
            density_enabled_species = None
            if self._analysis_name == "density":
                enabled_species = self._enabled_density_species()
                if enabled_species is not None:
                    density_enabled_species = sorted(enabled_species)
            position_enabled_species = None
            if self._analysis_name == "position":
                enabled_species = self._enabled_position_species()
                if enabled_species is not None:
                    position_enabled_species = sorted(enabled_species)

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
                if (
                    index < len(self._series_show_raw_line_data)
                    and not self._series_show_raw_line_data[index]
                ):
                    entry["show_raw_line"] = False
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
                fit_x_min_value = _optional_float(
                    self._series_fit_x_mins_data[index],
                    field_name=f"Series {index + 1} fit x min",
                )
                fit_x_max_value = _optional_float(
                    self._series_fit_x_maxs_data[index],
                    field_name=f"Series {index + 1} fit x max",
                )
                fit_range_mode = (
                    "manual"
                    if fit_x_min_value is not None or fit_x_max_value is not None
                    else "visible"
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
                fit_color_out = self._series_fit_color_data[index].strip() or None
                fit_alpha_out = self._series_fit_alpha_data[index].strip() or None
                fit_line_width_out = self._series_fit_line_width_data[index].strip() or None
                fit_line_style_out = self._series_fit_line_style_data[index].strip() or None
                fit_config_payload["fit_color"] = fit_color_out
                fit_config_payload["fit_alpha"] = fit_alpha_out
                fit_config_payload["fit_line_width"] = fit_line_width_out
                fit_config_payload["fit_line_style"] = fit_line_style_out
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
                    "fit_color": fit_defaults["fit_color"],
                    "fit_alpha": fit_defaults["fit_alpha"],
                    "fit_line_width": fit_defaults["fit_line_width"],
                    "fit_line_style": fit_defaults["fit_line_style"],
                }:
                    entry["fit"] = fit_config_payload
                cumulative_enabled_value = bool(self._series_cumulative_enabled_data[index])
                cumulative_label_override_value = (
                    self._series_cumulative_label_overrides_data[index].strip() or None
                )
                cumulative_show_in_legend_value = bool(
                    self._series_cumulative_show_in_legend_data[index]
                )
                cumulative_color_value = self._series_cumulative_color_data[index].strip() or None
                cumulative_alpha_value = self._series_cumulative_alpha_data[index].strip() or None
                cumulative_line_width_value = (
                    self._series_cumulative_line_width_data[index].strip() or None
                )
                cumulative_line_style_value = (
                    self._series_cumulative_line_style_data[index].strip() or None
                )
                cumulative_payload = {
                    "enabled": cumulative_enabled_value,
                    "label_override": cumulative_label_override_value,
                    "show_in_legend": cumulative_show_in_legend_value,
                    "color": cumulative_color_value,
                    "alpha": cumulative_alpha_value,
                    "line_width": cumulative_line_width_value,
                    "line_style": cumulative_line_style_value,
                }
                if cumulative_payload != _cumulative_defaults_for_gui():
                    entry["cumulative"] = cumulative_payload
                integration_enabled_value = bool(self._series_integration_enabled_data[index])
                integration_source_label = self._series_integration_source_data[index].strip()
                integration_source_value = _INTEGRATION_SOURCE_BY_LABEL.get(
                    integration_source_label, "plotted"
                )
                integration_x_min_value = _optional_float(
                    self._series_integration_x_min_data[index],
                    field_name=f"Series {index + 1} integration x-min",
                )
                integration_x_max_value = _optional_float(
                    self._series_integration_x_max_data[index],
                    field_name=f"Series {index + 1} integration x-max",
                )
                if (
                    integration_x_min_value is not None
                    and integration_x_max_value is not None
                    and integration_x_min_value >= integration_x_max_value
                ):
                    raise ValueError(
                        f"Series {index + 1} integration x-min must be smaller than x-max."
                    )
                integration_baseline_value = _optional_float(
                    self._series_integration_baseline_data[index],
                    field_name=f"Series {index + 1} integration baseline",
                )
                integration_alpha_value = _optional_float(
                    self._series_integration_alpha_data[index],
                    field_name=f"Series {index + 1} integration alpha",
                )
                if (
                    integration_alpha_value is not None
                    and not 0.0 <= integration_alpha_value <= 1.0
                ):
                    raise ValueError(
                        f"Series {index + 1} integration alpha must be between 0 and 1."
                    )
                integration_color_value = (
                    _explicit_text(self._series_integration_color_data[index])
                    if self._series_integration_color_mode_data[index].strip().lower() == "custom"
                    else ""
                )
                integration_payload: dict[str, Any] = {
                    "enabled": integration_enabled_value,
                    "source": integration_source_value,
                    "x_min": integration_x_min_value,
                    "x_max": integration_x_max_value,
                    "baseline": (
                        0.0 if integration_baseline_value is None else integration_baseline_value
                    ),
                    "color": integration_color_value or None,
                    "alpha": (0.25 if integration_alpha_value is None else integration_alpha_value),
                }
                integration_defaults = _integration_defaults_for_gui()
                integration_default_payload = {
                    "enabled": integration_defaults["enabled"],
                    "source": integration_defaults["source"],
                    "x_min": integration_defaults["x_min"],
                    "x_max": integration_defaults["x_max"],
                    "baseline": integration_defaults["baseline"],
                    "color": integration_defaults["color"],
                    "alpha": integration_defaults["alpha"],
                }
                if integration_payload != integration_default_payload:
                    entry["integration"] = integration_payload
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
                elif self._series_is_generated(index):
                    generated_color = self._effective_series_color(index).strip()
                    if generated_color:
                        entry["color"] = generated_color
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
            figure_alpha = _optional_float(self.figure_alpha.text(), field_name="figure-alpha")
            if figure_alpha is not None and not 0.0 <= figure_alpha <= 1.0:
                raise ValueError("figure-alpha must be between 0 and 1.")
            if figure_alpha is not None:
                figure_kwargs_merged["alpha"] = figure_alpha
            else:
                figure_kwargs_merged.pop("alpha", None)
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
            if line_alpha is not None and not 0.0 <= line_alpha <= 1.0:
                raise ValueError("line-alpha must be between 0 and 1.")
            marker_size = _optional_float(self.marker_size.text(), field_name="marker-size")
            marker_color = _explicit_text(self.marker_color.text())
            marker_type = self.marker_type.currentText().strip()
            markers_on = self.markers_mode.currentText().strip().lower() != "off"
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
            if markers_on:
                if marker_type:
                    line_kwargs_merged["marker"] = marker_type
                if marker_color:
                    line_kwargs_merged["markerfacecolor"] = marker_color
                    line_kwargs_merged["markeredgecolor"] = marker_color
            else:
                for key in ("marker", "markersize", "markerfacecolor", "markeredgecolor"):
                    line_kwargs_merged.pop(key, None)
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
            legend_title_font_size = _optional_positive_int_or_none(
                self.legend_title_font.text(),
                field_name="legend-title-font-size",
            )
            if legend_title_font_size is not None:
                legend_kwargs_merged["title_fontsize"] = legend_title_font_size
            else:
                legend_kwargs_merged.pop("title_fontsize", None)
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
            for key in (
                "direction",
                "length",
                "width",
                "_minor_ticks_mode",
                "_x_tick_params",
                "_y_tick_params",
                "_x_minor_ticks_mode",
                "_y_minor_ticks_mode",
            ):
                tick_params_kwargs_merged.pop(key, None)

            def _axis_tick_params(axis: str) -> dict[str, Any]:
                direction_widget = self.x_tick_direction if axis == "x" else self.y_tick_direction
                length_widget = self.x_tick_length if axis == "x" else self.y_tick_length
                width_widget = self.x_tick_width if axis == "x" else self.y_tick_width
                font_widget = self.x_tick_font if axis == "x" else self.y_tick_font
                direction = direction_widget.currentText().strip().lower() or "out"
                if direction not in _TICK_DIRECTIONS:
                    raise ValueError(f"{axis}-tick direction must be out, in, or inout.")
                params: dict[str, Any] = {"direction": direction}
                length = _optional_float(length_widget.text(), field_name=f"{axis}-tick-length")
                if length is not None:
                    if length <= 0:
                        raise ValueError(f"{axis}-tick-length must be positive.")
                    params["length"] = length
                width = _optional_float(width_widget.text(), field_name=f"{axis}-tick-width")
                if width is not None:
                    if width <= 0:
                        raise ValueError(f"{axis}-tick-width must be positive.")
                    params["width"] = width
                font_size = _optional_positive_int_or_none(
                    font_widget.text(),
                    field_name=f"{axis}-tick-font-size",
                )
                if font_size is not None:
                    params["labelsize"] = font_size
                return params

            x_ticks_on = self.x_ticks_mode.currentText().strip().lower() != "off"
            y_ticks_on = self.y_ticks_mode.currentText().strip().lower() != "off"
            resolved_ticks_axis = (
                "both"
                if x_ticks_on and y_ticks_on
                else "x"
                if x_ticks_on
                else "y"
                if y_ticks_on
                else "both"
            )
            tick_params_kwargs_merged["_ticks_axis"] = resolved_ticks_axis
            tick_params_kwargs_merged["axis"] = resolved_ticks_axis
            tick_params_kwargs_merged["_x_ticks_visible"] = x_ticks_on
            tick_params_kwargs_merged["_y_ticks_visible"] = y_ticks_on
            tick_params_kwargs_merged["_x_tick_params"] = _axis_tick_params("x")
            tick_params_kwargs_merged["_y_tick_params"] = _axis_tick_params("y")
            for axis, widget in (
                ("x", self.x_minor_ticks_mode),
                ("y", self.y_minor_ticks_mode),
            ):
                minor_ticks_mode = widget.currentText().strip().lower() or "off"
                if minor_ticks_mode not in _MINOR_TICKS_MODES:
                    raise ValueError(f"{axis}-minor ticks mode must be on or off.")
                tick_params_kwargs_merged[f"_{axis}_minor_ticks_mode"] = minor_ticks_mode
            tick_params_kwargs_value = tick_params_kwargs_merged or None

            savefig_kwargs_value = (
                dict(savefig_kwargs_value) if isinstance(savefig_kwargs_value, dict) else None
            )

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
            x_axis_scale = _optional_float(
                self.x_axis_scale.text(),
                field_name="x-axis-scale-factor",
            )
            if x_axis_scale is None:
                x_axis_scale = 1.0
            if x_axis_scale == 0.0:
                raise ValueError("x-axis-scale-factor must not be zero.")
            x_axis_offset = _optional_float(
                self.x_axis_offset.text(),
                field_name="x-axis-offset",
            )
            if x_axis_offset is None:
                x_axis_offset = 0.0

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
            data_export_format = (
                self._data_export_format.currentText().strip().lower()
                if self._data_export_format is not None
                else "auto"
            )
            data_export_delimiter = (
                self._data_export_delimiter.currentText().strip().lower()
                if self._data_export_delimiter is not None
                else "auto"
            )
            data_export_include_metadata = (
                bool(self._data_export_include_metadata.isChecked())
                if self._data_export_include_metadata is not None
                else False
            )
            data_export_enabled_only = (
                bool(self._data_export_enabled_only.isChecked())
                if self._data_export_enabled_only is not None
                else True
            )
            projection_line_width_value = _optional_float(
                self.projection_line_width.text(),
                field_name="trajectory-line-width",
            )
            if (
                projection_line_width_value is not None
                and projection_line_width_value <= 0.0
            ):
                raise ValueError("Trajectory line width must be positive.")

            settings = {
                "title": title_value,
                "x_label": _synced_text("x_label", self.x_label),
                "y_label": _synced_text("y_label", self.y_label),
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "x_scale": self.x_scale.currentText().strip() or "linear",
                "x_axis_scale": x_axis_scale,
                "x_axis_offset": x_axis_offset,
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
                    False
                    if self.axes_border_mode.currentText().strip().lower() == "off"
                    else {
                        "left": self.border_left.isChecked(),
                        "right": self.border_right.isChecked(),
                        "top": self.border_top.isChecked(),
                        "bottom": self.border_bottom.isChecked(),
                    }
                    if self.axes_border_mode.currentText().strip().lower() == "custom"
                    else True
                ),
                "grid": _mode_to_toggle(self.grid_mode.currentText()),
                "ticks": x_ticks_on or y_ticks_on,
                "markers": _mode_to_toggle(self.markers_mode.currentText()),
                "legend_title": _explicit_text(self.legend_title.text()) or None,
                "legend_loc": self.legend_loc.currentText().strip() or "best",
                "figsize": figsize,
                "dpi": _optional_positive_int_or_none(self.dpi.text(), field_name="dpi"),
                "font_family": _explicit_text(self.font_family.text()) or None,
                "font_color": _explicit_text(self.font_color.text()) or None,
                "font_size": _optional_positive_int_or_none(
                    self.base_font_size.text(), field_name="font-size"
                ),
                "title_font_size": _optional_positive_int_or_none(
                    self.title_font.text(), field_name="title-font-size"
                ),
                "title_pad": _optional_float(self.title_pad.text(), field_name="title-pad"),
                "label_font_size": shared_label_font_size_value,
                "x_label_font_size": x_label_font_size_value,
                "y_label_font_size": y_label_font_size_value,
                "tick_font_size": shared_tick_font_size_value,
                "x_tick_font_size": x_tick_font_size_value,
                "y_tick_font_size": y_tick_font_size_value,
                "figure_alpha": figure_alpha,
                "legend_font_size": _optional_positive_int_or_none(
                    self.legend_font.text(), field_name="legend-font-size"
                ),
                "line_width": _optional_float(self.line_width.text(), field_name="line-width"),
                "projection_line_width": projection_line_width_value,
                "line_colors": None if series_overrides else line_colors_value,
                "series_labels": series_labels,
                "series_order": (
                    self._current_series_id_order()
                    if self._current_series_id_order() != self._series_natural_order_data
                    else None
                ),
                "series_descriptors": deepcopy(self._series_descriptors_data),
                "density_enabled_species": density_enabled_species,
                "position_enabled_species": position_enabled_species,
                "series_overrides": series_overrides or None,
                "series_enabled": None if series_overrides else series_enabled_value,
                "series_show_in_legend": (
                    None if series_overrides else series_show_in_legend_value
                ),
                "series_alpha": None if series_overrides else series_alpha_value,
                "series_line_widths": None if series_overrides else series_line_widths_value,
                "series_markers": None if series_overrides else series_markers_value,
                "series_line_kwargs": None if series_overrides else series_line_kwargs_value,
                "series_normalization_modes": (
                    None if series_overrides else normalization_modes_value
                ),
                "series_normalization_values": (
                    None if series_overrides else normalization_values_value
                ),
                "series_normalization_x_refs": (
                    None if series_overrides else normalization_x_refs_value
                ),
                "annotations": annotations_value,
                "x_bin_width": x_bin_width,
                "x_bin_reducer": (
                    "sum"
                    if self._analysis_name == "orientation"
                    and self._is_orientation_heatmap_mode()
                    else self.x_bin_reducer.currentText().strip() or "mean"
                )
                if x_bin_width is not None
                else None,
                "min_bin_points": min_bin_points,
                "y_bin_width": y_bin_width,
                "y_bin_reducer": (
                    "sum"
                    if self._analysis_name == "orientation"
                    and self._is_orientation_heatmap_mode()
                    else self.y_bin_reducer.currentText().strip() or "mean"
                )
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
                "plot_data_format": data_export_format,
                "plot_data_delimiter": data_export_delimiter,
                "plot_data_include_metadata": data_export_include_metadata,
                "plot_data_enabled_only": data_export_enabled_only,
                "_gui_sync_modes": {
                    key: self._synced_field_mode(key)
                    for key in _SYNCED_FIELD_KEYS
                    if self._synced_field_mode(key) != "auto"
                }
                or None,
            }
            if hasattr(self, "analysis_species"):
                settings["species"] = _explicit_text(self.analysis_species.text()) or None
            if hasattr(self, "analysis_axis"):
                axis_value = self.analysis_axis.currentText().strip().lower()
                settings["axis"] = None if axis_value == "" else axis_value
            if hasattr(self, "density_x_mode") or hasattr(self, "density_quantity"):
                resolved_view_mapping = self._current_density_mapping()
                if (
                    generic_view_type_compatibility(
                        self._density_contract(),
                        resolved_view_mapping,
                    )
                    == "invalid"
                ):
                    raise ValueError(
                        "The selected density mapping is incompatible with the current plot-data contract."
                    )
                if self._is_density_heatmap_mode():
                    settings["density_2d_x_axis"] = _DENSITY_X_MODE_BY_LABEL.get(
                        self.density_2d_x_axis.currentText().strip().lower(),
                        "x",
                    ) if hasattr(self, "density_2d_x_axis") else "x"
                    settings["density_2d_y_axis"] = _DENSITY_X_MODE_BY_LABEL.get(
                        self.density_2d_y_axis.currentText().strip().lower(),
                        "y",
                    ) if hasattr(self, "density_2d_y_axis") else "y"
                    if settings["density_2d_x_axis"] == settings["density_2d_y_axis"]:
                        raise ValueError("Density 2D axes must be different.")
                    if {
                        settings["density_2d_x_axis"],
                        settings["density_2d_y_axis"],
                    } == {"z", "distance"}:
                        raise ValueError("Density 2D plotting does not support Z versus distance.")
                    settings["axis"] = None
                    if hasattr(self, "_density_filter_widgets"):
                        for axis_id, widgets in self._density_filter_widgets.items():
                            lower_widget, upper_widget = widgets
                            lower_value = _optional_float(
                                lower_widget.text(),
                                field_name=f"{axis_id}-filter-min",
                            )
                            upper_value = _optional_float(
                                upper_widget.text(),
                                field_name=f"{axis_id}-filter-max",
                            )
                            if (
                                lower_value is not None
                                and upper_value is not None
                                and lower_value > upper_value
                            ):
                                raise ValueError(f"{axis_id.upper()} range minimum must not exceed maximum.")
                            lower_value, upper_value = self._density_effective_filter_values(
                                axis_id,
                                lower_value,
                                upper_value,
                            )
                            settings[f"density_filter_{axis_id}_min"] = lower_value
                            settings[f"density_filter_{axis_id}_max"] = upper_value
                else:
                    settings["plane"] = None
                    if hasattr(self, "_density_filter_widgets"):
                        for axis_id, widgets in self._density_filter_widgets.items():
                            lower_widget, upper_widget = widgets
                            lower_value = _optional_float(
                                lower_widget.text(),
                                field_name=f"{axis_id}-filter-min",
                            )
                            upper_value = _optional_float(
                                upper_widget.text(),
                                field_name=f"{axis_id}-filter-max",
                            )
                            if (
                                lower_value is not None
                                and upper_value is not None
                                and lower_value > upper_value
                            ):
                                raise ValueError(f"{axis_id.upper()} range minimum must not exceed maximum.")
                            lower_value, upper_value = self._density_effective_filter_values(
                                axis_id,
                                lower_value,
                                upper_value,
                            )
                            settings[f"density_filter_{axis_id}_min"] = lower_value
                            settings[f"density_filter_{axis_id}_max"] = upper_value
            if hasattr(self, "coord_species_a"):
                settings["species_a"] = self._selected_profile_filter_value(self.coord_species_a)
            if hasattr(self, "coord_species_b"):
                settings["species_b"] = self._selected_profile_filter_value(self.coord_species_b)
            if hasattr(self, "position_view_type"):
                mapping = self._current_position_mapping(strict=True)
                if (
                    mapping.filter_min is not None
                    and mapping.filter_max is not None
                    and mapping.filter_min > mapping.filter_max
                ):
                    raise ValueError(
                        "2D Heatmap range minimum must not exceed the range maximum."
                    )
                compatibility = generic_view_type_compatibility(
                    self._position_contract(),
                    mapping,
                )
                if (
                    compatibility == "invalid"
                ):
                    raise ValueError(
                        "The selected position mapping is incompatible with the current plot-data contract."
                    )
                resolved_view_mapping = mapping
            if hasattr(self, "coordination_component") or hasattr(self, "coordination_time_axis"):
                resolved_view_mapping = self._current_coordination_mapping()
                if (
                    generic_view_type_compatibility(
                        self._coordination_contract(),
                        resolved_view_mapping,
                    )
                    == "invalid"
                ):
                    raise ValueError(
                        "The selected coordination mapping is incompatible with the current plot-data contract."
                    )
            if hasattr(self, "orientation_component") or hasattr(self, "orientation_angle"):
                resolved_view_mapping = self._current_orientation_mapping()
                if (
                    generic_view_type_compatibility(
                        self._active_orientation_contract(),
                        resolved_view_mapping,
                    )
                    == "invalid"
                ):
                    raise ValueError(
                        "The selected orientation mapping is incompatible with the current plot-data contract."
                    )
                orientation_heatmap_mode = self._is_orientation_heatmap_mode()
                orientation_line_component = "heatmap" if orientation_heatmap_mode else "average"
                if not orientation_heatmap_mode and hasattr(self, "orientation_component"):
                    orientation_line_component = _ORIENTATION_LINE_QUANTITY_BACKEND_BY_LABEL.get(
                        self.orientation_component.currentText().strip(),
                        self.orientation_component.currentText().strip().lower() or "average",
                    )
                orientation_grid_line_enabled = (
                    not orientation_heatmap_mode and orientation_line_component == "average"
                )
                if orientation_heatmap_mode:
                    settings["orientation_line_x_axis"] = _DENSITY_X_MODE_BY_LABEL.get(
                        self.orientation_line_x_axis.currentText().strip().lower(),
                        "distance",
                    ) if hasattr(self, "orientation_line_x_axis") else "distance"
                    settings["orientation_heatmap_x_axis"] = None
                    settings["orientation_heatmap_y_axis"] = None
                else:
                    if orientation_grid_line_enabled:
                        settings["orientation_line_x_axis"] = _DENSITY_X_MODE_BY_LABEL.get(
                            self.orientation_line_x_axis.currentText().strip().lower(),
                            "distance",
                        ) if hasattr(self, "orientation_line_x_axis") else "distance"
                    else:
                        settings["orientation_line_x_axis"] = "distance"
                    settings["orientation_heatmap_x_axis"] = None
                    settings["orientation_heatmap_y_axis"] = None
                if hasattr(self, "_orientation_filter_widgets"):
                    for axis_id, widgets in self._orientation_filter_widgets.items():
                        if orientation_grid_line_enabled:
                            lower_widget, upper_widget = widgets
                            lower_value = _optional_float(
                                lower_widget.text(),
                                field_name=f"orientation-{axis_id}-filter-min",
                            )
                            upper_value = _optional_float(
                                upper_widget.text(),
                                field_name=f"orientation-{axis_id}-filter-max",
                            )
                            if (
                                lower_value is not None
                                and upper_value is not None
                                and lower_value > upper_value
                            ):
                                raise ValueError(
                                    f"Orientation {axis_id.upper()} range minimum must not exceed maximum."
                                )
                            lower_value, upper_value = self._orientation_effective_filter_values(
                                axis_id,
                                lower_value,
                                upper_value,
                            )
                        else:
                            lower_value = None
                            upper_value = None
                        settings[f"orientation_filter_{axis_id}_min"] = lower_value
                        settings[f"orientation_filter_{axis_id}_max"] = upper_value
            if hasattr(self, "potential_view_type"):
                potential_mapping = self._current_potential_mapping()
                if (
                    generic_view_type_compatibility(
                        self._potential_contract(),
                        potential_mapping,
                    )
                    == "invalid"
                ):
                    raise ValueError(
                        "The selected potential mapping is incompatible with the current plot-data contract."
                    )
                resolved_view_mapping = potential_mapping
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
            if (
                hasattr(self, "heatmap_value_mode")
                and self._analysis_name == "orientation"
            ):
                value_label = self.heatmap_value_mode.currentText().strip()
                value_mode = _HEATMAP_VALUE_MODE_BY_LABEL.get(
                    value_label,
                    "raw_counts",
                )
                settings["heatmap_value_mode"] = value_mode
                bulk_reference_mode = (
                    "manual"
                    if self.heatmap_bulk_reference_mode.currentText().strip().lower()
                    == "manual"
                    else "auto"
                )
                settings["heatmap_bulk_reference_mode"] = bulk_reference_mode
                settings["heatmap_bulk_min"] = None
                settings["heatmap_bulk_max"] = None
                if (
                    value_mode == "bulk_relative_enrichment"
                    and bulk_reference_mode == "manual"
                ):
                    bulk_min = _optional_float(
                        self.heatmap_bulk_min.text(),
                        field_name="bulk-reference minimum distance",
                    )
                    bulk_max = _optional_float(
                        self.heatmap_bulk_max.text(),
                        field_name="bulk-reference maximum distance",
                    )
                    if bulk_min is None or bulk_max is None or bulk_min >= bulk_max:
                        raise ValueError(
                            "Manual bulk reference requires finite minimum and maximum "
                            "distances with minimum < maximum."
                        )
                    settings["heatmap_bulk_min"] = bulk_min
                    settings["heatmap_bulk_max"] = bulk_max
            if hasattr(self, "heatmap_log_scale"):
                settings["heatmap_log_scale"] = (
                    self.heatmap_log_scale.currentText().strip().lower()
                    == "logarithmic"
                )
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
            if resolved_view_mapping is not None:
                settings["view_mapping"] = serialize_plot_view_mapping(resolved_view_mapping)
            settings = self._merge_density_view_state_into_settings(settings)
            settings = self._merge_position_view_state_into_settings(settings)
            settings = self._merge_orientation_view_state_into_settings(settings)
            return settings

        def _report_error(self, title_text: str, exc: Exception) -> None:
            self._status_label.setText(f"{title_text}: {exc}")
            self._refresh_shell_state()
            QMessageBox.critical(self, title_text, str(exc))

        def _handle_preview(self) -> None:
            self._preview_timer.stop()
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

        def _save_current_profile_for_export(self, settings: dict[str, Any]) -> None:
            # Export uses the same resolved settings path as Save Profile so a
            # reopened GUI reproduces the exported figure/data for the active profile.
            on_save(self._current_profile_name, settings)
            self._saved_signature = self._signature(settings)

        def _sync_preview_canvas_axis_limits_for_export(self) -> None:
            if self._preview_figure is None:
                return
            axes = list(getattr(self._preview_figure, "axes", []) or [])
            if not axes:
                return
            self._set_axis_limit_fields_from_canvas(axes[0])

        def _handle_save_figure(self) -> None:
            if on_save_figure is None:
                self._status_label.setText("Save-figure action is not available.")
                return
            try:
                self._sync_preview_canvas_axis_limits_for_export()
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
                self._save_current_profile_for_export(settings)
                result = on_save_figure(settings, output_path)
                message = result[0] if isinstance(result, tuple) else result
                render_state = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_state_to_synced_fields(render_state)
                self._status_label.setText(message)
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Save figure failed", exc)

        def _handle_save_data(self) -> None:
            if on_save_data is None:
                self._status_label.setText("Save-data action is not available.")
                return
            try:
                self._sync_preview_canvas_axis_limits_for_export()
                settings = self._collect_settings()
                data_default_name = self._data_default_name
                selected_filter = ""
                requested_format = str(settings.get("plot_data_format") or "auto").lower()
                format_map = {
                    "csv": ("CSV data (*.csv)", ".csv"),
                    "dat": ("DAT data (*.dat)", ".dat"),
                    "tsv": ("TSV data (*.tsv)", ".tsv"),
                    "txt": ("Text data (*.txt)", ".txt"),
                }
                if requested_format in format_map:
                    selected_filter, suffix = format_map[requested_format]
                    data_default_name = str(Path(data_default_name).with_suffix(suffix))
                output_path, _selected = QFileDialog.getSaveFileName(
                    self,
                    "Save Data",
                    data_default_name,
                    self._data_save_filters,
                    selected_filter,
                )
                if not output_path:
                    self._status_label.setText("Save data canceled.")
                    return
                self._save_current_profile_for_export(settings)
                result = on_save_data(settings, output_path)
                message = result[0] if isinstance(result, tuple) else result
                render_state = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_state_to_synced_fields(render_state)
                self._status_label.setText(message)
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Save data failed", exc)

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
            self._reset_undo_history()
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
            self._closing = True
            self._preview_timer.stop()
            self._preview_generation += 1
            self._active_preview_generation = None
            self._active_preview_image_path = None
            self._pending_preview_request = None
            self._preview_worker_stop.set()
            try:
                self._preview_worker_queue.put_nowait(None)
            except Exception:
                pass
            worker = self._preview_worker_thread
            if worker is not None and worker.is_alive():
                worker.join(timeout=0.5)
            if self._detached_preview_window is not None:
                detached_window = self._detached_preview_window
                self._detached_preview_window = None
                self._detached_preview_pane = None
                detached_window.close_from_dock()
                detached_window.deleteLater()
            self._cleanup_preview_canvas(close_figure=True)
            if self._preview_image_path is not None:
                QPixmapCache.remove(str(self._preview_image_path))
                try:
                    self._preview_image_path.unlink(missing_ok=True)
                except OSError:
                    pass
            for path in list(self._preview_temp_paths):
                self._cleanup_preview_temp_path(path)

        def _refresh_preview_after_layout(self) -> None:
            if self._preview_figure is not None and self._preview_canvas is not None:
                try:
                    self._preview_canvas.draw_idle()
                except Exception:
                    pass
                return
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
                self._sync_theme_switch_label()
                if hasattr(self, "series_list") and self.series_list is not None:
                    for index in range(self.series_list.count()):
                        self._apply_series_list_item_visuals(self.series_list.item(index), index)
            super().changeEvent(event)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # pragma: no cover - UI flow
            preview_scroll = self._preview_scroll
            preview_label = self._preview_label
            preview_frame = self._preview_frame
            preview_canvas_scroll = self._preview_canvas_scroll
            preview_viewport = preview_scroll.viewport() if preview_scroll is not None else None
            preview_canvas_viewport = (
                preview_canvas_scroll.viewport()
                if preview_canvas_scroll is not None
                else None
            )

            if isinstance(watched, (QLineEdit, QPlainTextEdit)):
                if event.type() == QEvent.Type.KeyPress:
                    key = int(event.key())
                    modifiers = event.modifiers()
                    is_ctrl_edit_shortcut = bool(
                        modifiers & Qt.KeyboardModifier.ControlModifier
                    ) and key in {
                        int(Qt.Key.Key_V),
                        int(Qt.Key.Key_X),
                    }
                    if key in {
                        int(Qt.Key.Key_Return),
                        int(Qt.Key.Key_Enter),
                    }:
                        self._finalize_text_undo_edit(watched)
                    elif (
                        key
                        in {
                            int(Qt.Key.Key_Backspace),
                            int(Qt.Key.Key_Delete),
                        }
                        or is_ctrl_edit_shortcut
                        or (
                            not (modifiers & Qt.KeyboardModifier.ControlModifier)
                            and not (modifiers & Qt.KeyboardModifier.AltModifier)
                            and event.text()
                        )
                    ):
                        self._begin_text_undo_edit(watched)
                elif event.type() == QEvent.Type.FocusOut:
                    self._finalize_text_undo_edit(watched)

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
            if watched in {preview_frame, preview_canvas_viewport} and event.type() == QEvent.Type.Resize:
                QTimer.singleShot(0, self._resize_preview_canvas_to_figure)
            return super().eventFilter(watched, event)

        def resizeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            super().resizeEvent(event)
            self._refresh_preview_pixmap()
            self._resize_preview_canvas_to_figure()

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
