"""Shared plotting helpers and style definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import difflib
import logging
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np

from ..analysis.statistics import (
    SeriesStatistics,
    statistics_available_stats,
)
from .fitting import execute_series_fit, resolve_series_fit_configs

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERACTIVE_BACKEND = "QtAgg"
CANONICAL_INTERACTIVE_BACKENDS = ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg", "MacOSX")
BACKEND_ALIASES = {
    "tkagg": "TkAgg",
    "qtagg": "QtAgg",
    "qt5agg": "QtAgg",
    "qt6agg": "QtAgg",
    "gtk3agg": "GTK3Agg",
    "wxagg": "WXAgg",
    "macosx": "MacOSX",
}
INTERACTIVE_BACKENDS = {
    "gtk3agg",
    "gtk4agg",
    "macosx",
    "nbagg",
    "qtagg",
    "qt5agg",
    "tkagg",
    "webagg",
    "wxagg",
}
_BACKEND_CONFIGURED = False
_CONFIGURED_BACKEND: str | None = None
_DEFAULT_BASE_FONT_SIZE = 12


def default_plot_font_sizes(base_font_size: int) -> dict[str, int]:
    """Return the default role-specific font sizes for one base font size."""
    base = max(1, int(base_font_size))
    return {
        "title_font_size": base + 2,
        "label_font_size": base,
        "tick_font_size": max(1, base - 2),
        "legend_font_size": max(1, base - 2),
    }


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate with NumPy's trapezoid API across NumPy 1.x and 2.x."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return float(trapz(y, x))
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _import_pyplot() -> Any:
    # Import pyplot lazily to guarantee backend selection happens first.
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class PlotStyle:
    """Plot style settings reused across all analysis plots."""

    figure_size: tuple[float, float] = (7.0, 4.0)
    dpi: int = 200
    font_family: str = "DejaVu Sans"
    font_color: str = "#000000"
    base_font_size: int = _DEFAULT_BASE_FONT_SIZE
    title_font_size: int = 14
    title_pad: float = 6.0
    label_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 10
    line_width: float = 2.0
    line_color: str = "#1f77b4"
    axes_border: bool | dict[str, bool] = True
    grid: bool = True
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.8
    grid_alpha: float = 0.35

    def __post_init__(self) -> None:
        base = max(1, int(self.base_font_size))
        object.__setattr__(self, "base_font_size", base)
        default_sizes = default_plot_font_sizes(base)
        canonical_defaults = default_plot_font_sizes(_DEFAULT_BASE_FONT_SIZE)
        for key in ("title_font_size", "label_font_size", "tick_font_size", "legend_font_size"):
            raw_value = int(getattr(self, key))
            if base != _DEFAULT_BASE_FONT_SIZE and raw_value == canonical_defaults[key]:
                raw_value = default_sizes[key]
            object.__setattr__(self, key, max(1, raw_value))


DEFAULT_PLOT_STYLE = PlotStyle()


_MOJIBAKE_MARKERS = ("Ã", "â", "Î", "Ï", "Â")


def normalize_plot_text(value: str) -> str:
    """Return plot text exactly as stored.

    Some older plot settings were persisted after UTF-8 text had been decoded with
    a Windows single-byte code page. That produces strings such as ``Hâ‚‚O``,
    ``Ã…``, or ``âŸ¨cos(Î¸)âŸ©``. When such settings are re-applied, they override
    clean defaults. This helper repairs those strings conservatively.
    """
    text = str(value)
    return text


def format_axis_label_units(label: str) -> str:
    """Return the axis label exactly as provided by the caller."""
    return normalize_plot_text(str(label))


def resolve_explicit_plot_text(value: str | None, default: str) -> str:
    """Preserve explicit blank strings while still filling missing values from defaults."""
    return normalize_plot_text(default if value is None else str(value))


def _series_statistics(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float | int | None]:
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    finite_y = y_values[finite_mask]
    if finite_y.size == 0:
        return {
            "point_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "point_count": int(finite_y.size),
        "min": float(np.min(finite_y)),
        "max": float(np.max(finite_y)),
        "mean": float(np.mean(finite_y)),
        "std": float(np.std(finite_y, ddof=0)),
    }


def default_series_colors(count: int) -> list[str]:
    """Return deterministic default series colors for a given series count."""
    if count <= 0:
        return []

    colors: list[str] = []
    prop_cycle = matplotlib.rcParams.get("axes.prop_cycle")
    if prop_cycle is not None:
        by_key = prop_cycle.by_key()
        raw_colors = by_key.get("color", [])
        colors = [str(item).strip() for item in raw_colors if str(item).strip()]

    if not colors:
        colors = [DEFAULT_PLOT_STYLE.line_color]

    return [colors[index % len(colors)] for index in range(count)]


def resolve_series_colors(
    line_colors: list[str] | None,
    *,
    series_count: int,
) -> list[str] | None:
    """Fill blank per-series colors with the indexed default palette."""
    if line_colors is None:
        return None
    if len(line_colors) != series_count:
        raise ValueError(
            f"line_colors count must match the number of plotted series ({series_count})."
        )

    normalized = [str(color).strip() for color in line_colors]
    if not any(normalized):
        return None

    defaults = default_series_colors(series_count)
    return [color or defaults[index] for index, color in enumerate(normalized)]


def _auto_axis_limits_from_values(
    values: np.ndarray,
    *,
    scale: str,
    clamp_nonnegative_to_zero: bool,
    padding_fraction: float = 0.05,
) -> list[float] | None:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None

    normalized_scale = str(scale).strip().lower()
    if normalized_scale == "log":
        data = data[data > 0.0]
        if data.size == 0:
            return None

    lower = float(np.min(data))
    upper = float(np.max(data))
    if lower == upper:
        if normalized_scale == "log":
            pad = max(abs(lower) * padding_fraction, 1.0e-6)
            lower = max(lower - pad, lower * 0.5, 1.0e-12)
            upper = upper + pad
        else:
            pad = max(abs(lower) * padding_fraction, 1.0)
            lower -= pad
            upper += pad
    elif normalized_scale == "log":
        log_lower = float(np.log10(lower))
        log_upper = float(np.log10(upper))
        log_pad = max((log_upper - log_lower) * padding_fraction, 0.05)
        lower = float(10 ** (log_lower - log_pad))
        upper = float(10 ** (log_upper + log_pad))
    else:
        pad = (upper - lower) * padding_fraction
        lower -= pad
        upper += pad

    if clamp_nonnegative_to_zero and lower >= 0.0 and normalized_scale != "log":
        lower = 0.0
    return [lower, upper]


def _merge_axis_limits(
    requested: tuple[float | None, float | None] | list[float | None] | None,
    auto: list[float] | None,
) -> list[float | None] | None:
    if requested is None:
        return None if auto is None else [float(auto[0]), float(auto[1])]

    resolved: list[float | None] = [
        None if requested[0] is None else float(requested[0]),
        None if requested[1] is None else float(requested[1]),
    ]
    if auto is None:
        return resolved
    if resolved[0] is None:
        resolved[0] = float(auto[0])
    if resolved[1] is None:
        resolved[1] = float(auto[1])
    return resolved


def _union_axis_limits(
    first: list[float] | None,
    second: list[float] | None,
) -> list[float] | None:
    if first is None:
        return None if second is None else [float(second[0]), float(second[1])]
    if second is None:
        return [float(first[0]), float(first[1])]
    return [
        float(min(first[0], second[0])),
        float(max(first[1], second[1])),
    ]


def _axes_artist_auto_limits(ax: Any) -> tuple[list[float] | None, list[float] | None]:
    if not bool(ax.has_data()):
        return None, None
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    auto_x = (
        [float(x_left), float(x_right)] if np.isfinite(x_left) and np.isfinite(x_right) else None
    )
    auto_y = (
        [float(y_bottom), float(y_top)] if np.isfinite(y_bottom) and np.isfinite(y_top) else None
    )
    return auto_x, auto_y


def _density_visible_auto_limits(
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    *,
    x_scale: str,
    y_scale: str,
    x_window: tuple[float | None, float | None] | list[float | None] | None = None,
    y_window: tuple[float | None, float | None] | list[float | None] | None = None,
) -> tuple[list[float] | None, list[float] | None]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    nonzero_x: list[np.ndarray] = []
    nonzero_y: list[np.ndarray] = []

    for x_values, y_values in zip(x_series, y_series):
        x_data = np.asarray(x_values, dtype=float)
        y_data = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        if x_window is not None:
            if x_window[0] is not None:
                finite_mask &= x_data >= float(x_window[0])
            if x_window[1] is not None:
                finite_mask &= x_data <= float(x_window[1])
        if y_window is not None:
            if y_window[0] is not None:
                finite_mask &= y_data >= float(y_window[0])
            if y_window[1] is not None:
                finite_mask &= y_data <= float(y_window[1])
        if not np.any(finite_mask):
            continue
        x_finite = x_data[finite_mask]
        y_finite = y_data[finite_mask]
        all_x.append(x_finite)
        all_y.append(y_finite)

        nonzero_mask = y_finite != 0.0
        if np.any(nonzero_mask):
            nonzero_x.append(x_finite[nonzero_mask])
            nonzero_y.append(y_finite[nonzero_mask])

    if not all_x:
        return None, None

    x_focus = np.concatenate(nonzero_x) if nonzero_x else np.concatenate(all_x)
    y_focus = np.concatenate(nonzero_y) if nonzero_y else np.concatenate(all_y)
    auto_x = _auto_axis_limits_from_values(
        x_focus,
        scale=x_scale,
        clamp_nonnegative_to_zero=False,
    )
    auto_y = _auto_axis_limits_from_values(
        y_focus,
        scale=y_scale,
        clamp_nonnegative_to_zero=True,
    )
    return auto_x, auto_y


def _visible_series_auto_limits(
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    *,
    x_scale: str,
    y_scale: str,
    x_window: tuple[float | None, float | None] | list[float | None] | None = None,
    y_window: tuple[float | None, float | None] | list[float | None] | None = None,
    clamp_y_nonnegative_to_zero: bool = False,
) -> tuple[list[float] | None, list[float] | None]:
    visible_x: list[np.ndarray] = []
    visible_y: list[np.ndarray] = []

    for x_values, y_values in zip(x_series, y_series):
        x_data = np.asarray(x_values, dtype=float)
        y_data = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        if x_window is not None:
            if x_window[0] is not None:
                finite_mask &= x_data >= float(x_window[0])
            if x_window[1] is not None:
                finite_mask &= x_data <= float(x_window[1])
        if y_window is not None:
            if y_window[0] is not None:
                finite_mask &= y_data >= float(y_window[0])
            if y_window[1] is not None:
                finite_mask &= y_data <= float(y_window[1])
        if not np.any(finite_mask):
            continue
        visible_x.append(x_data[finite_mask])
        visible_y.append(y_data[finite_mask])

    if not visible_x:
        return None, None

    auto_x = _auto_axis_limits_from_values(
        np.concatenate(visible_x),
        scale=x_scale,
        clamp_nonnegative_to_zero=False,
    )
    auto_y = _auto_axis_limits_from_values(
        np.concatenate(visible_y),
        scale=y_scale,
        clamp_nonnegative_to_zero=clamp_y_nonnegative_to_zero,
    )
    return auto_x, auto_y


@dataclass(frozen=True)
class SingleSeriesPlotOptions:
    """Resolved plotting options for one rendered series."""

    line_color: str | None = None
    line_visible: bool = True
    line_width_override: float | None = None
    line_marker: str | None = None
    normalization_mode: str | None = None
    normalization_value: float | None = None
    normalization_x_ref: float | None = None


@dataclass(frozen=True)
class SeriesErrorConfig:
    """Requested uncertainty overlay for one rendered 1-D series."""

    enabled: bool = False
    stat: str | None = None
    style: str = "band"
    color: str | None = None
    label_override: str | None = None
    show_in_legend: bool = False


@dataclass(frozen=True)
class SeriesCumulativeConfig:
    """Requested cumulative-average derived line for one rendered 1-D series."""

    enabled: bool = False
    label_override: str | None = None
    show_in_legend: bool = True
    color: str | None = None
    alpha: float | None = None
    line_width: float | None = None
    line_style: str | None = None


@dataclass(frozen=True)
class PreparedLineSeries:
    """Prepared 1-D series data plus optional uncertainty metadata."""

    x: np.ndarray
    y: np.ndarray
    statistics: SeriesStatistics | None
    available_error_stats: list[str]
    error_config: SeriesErrorConfig
    masked_bin_count: int
    error_status: str
    statistics_mode: str
    error_reason: str | None = None


@dataclass(frozen=True)
class IntegrationConfig:
    """Requested shaded integral overlay for rendered 1-D series."""

    enabled: bool = False
    source: str = "plotted"
    target: str = "selected"
    target_series_id: str | None = None
    x_min: float | None = None
    x_max: float | None = None
    baseline: float = 0.0
    color: str | None = None
    alpha: float = 0.25


def _coerce_x_axis_linear_transform(
    scale: float | None,
    offset: float | None,
) -> tuple[float, float]:
    resolved_scale = 1.0 if scale is None else float(scale)
    resolved_offset = 0.0 if offset is None else float(offset)
    if not np.isfinite(resolved_scale):
        raise ValueError("X-axis scale factor must be finite.")
    if resolved_scale == 0.0:
        raise ValueError("X-axis scale factor must not be zero.")
    if not np.isfinite(resolved_offset):
        raise ValueError("X-axis offset must be finite.")
    return resolved_scale, resolved_offset


def _base_x_values(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    raw_x = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    if raw_x.size == 0 and y_array.size > 0:
        raw_x = np.arange(1, y_array.size + 1, dtype=float)
    return raw_x


def _display_x_values(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    scale: float,
    offset: float,
) -> np.ndarray:
    return scale * _base_x_values(x_values, y_values) + offset


def _error_provenance_family(
    *,
    requested_stat: str | None,
    statistics_mode: str,
) -> str:
    normalized_mode = str(statistics_mode).strip().lower()
    if normalized_mode == "raw_grouped":
        return "raw_grouped"
    if normalized_mode == "saved_rebinned_sample":
        return "sample"
    token = str(requested_stat or "").strip().lower()
    if token.startswith("block_"):
        return "block"
    return "sample"


def _describe_error_provenance(
    *,
    analysis_name: str | None,
    requested_stat: str | None,
    statistics_mode: str,
) -> str:
    normalized_analysis = str(analysis_name or "").strip().lower()
    family = _error_provenance_family(
        requested_stat=requested_stat,
        statistics_mode=statistics_mode,
    )
    if statistics_mode == "raw_grouped":
        return (
            "Computed from the currently grouped raw plotted points for this series. "
            "The spread updates whenever section width or grouping changes."
        )
    if statistics_mode == "saved_rebinned_sample":
        return (
            "Reconstructed from stored sample statistics after x rebinning. "
            "Block-based uncertainty is not preserved across rebinned bins."
        )
    if normalized_analysis == "density":
        if family == "block":
            return (
                "Computed from saved density bin statistics over contiguous frame blocks. "
                "It measures how block-averaged bin values vary across the trajectory."
            )
        return (
            "Computed from saved frame-to-frame density bin values. "
            "It measures how each bin varies across trajectory frames."
        )
    if normalized_analysis == "rdf":
        if family == "block":
            return (
                "Computed from saved RDF statistics over contiguous frame blocks. "
                "It measures how block-averaged g(r) varies across the trajectory."
            )
        return (
            "Computed from saved per-frame g(r) values in bins with finite expected counts. "
            "It measures frame-to-frame variation in the RDF."
        )
    if normalized_analysis == "msd":
        return (
            "Computed from saved per-lag squared-displacement samples across the selected atoms. "
            "Only sample-based uncertainty is available for MSD."
        )
    if normalized_analysis == "orientation":
        if family == "block":
            return (
                "Computed from saved orientation statistics over contiguous frame blocks. "
                "It measures how block-averaged line observables vary across the trajectory."
            )
        return (
            "Computed from saved per-frame orientation line-observable values. "
            "It measures frame-to-frame variation in the selected orientation quantity."
        )
    if normalized_analysis in {"position", "coordination", "potential"}:
        return (
            "Computed at plot time from the currently grouped raw points for this series. "
            "It reflects the spread of the values visible in the current grouping."
        )
    if family == "block":
        return "Computed from saved contiguous frame-block statistics for this series."
    return "Computed from saved sample statistics for this series."


_ANNOTATION_TYPES = {"text", "line", "arrow"}
_ANNOTATION_COORD_SYSTEMS = {"data", "axes"}
_ANNOTATION_LINE_STYLES = {"-", "--", "-.", ":"}
_ANNOTATION_H_ALIGNS = {"left", "center", "right"}
_ANNOTATION_V_ALIGNS = {"top", "center", "bottom", "baseline"}
_ANNOTATION_ARROW_STYLES = {"->", "-|>", "<->", "simple", "fancy"}


def _empty_series_statistics() -> SeriesStatistics:
    empty_int = np.empty(0, dtype=int)
    empty_float = np.empty(0, dtype=float)
    return SeriesStatistics(
        point_count=empty_int,
        sample_n=empty_int,
        sample_std=empty_float,
        sample_sem=empty_float,
    )


def _annotation_numeric(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _annotation_positive_numeric(value: Any, *, field_name: str) -> float:
    numeric = _annotation_numeric(value, field_name=field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be > 0.")
    return numeric


def _coerce_plot_annotation(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Annotation {index + 1} must be a mapping.")

    annotation_type = str(value.get("type") or "text").strip().lower()
    if annotation_type not in _ANNOTATION_TYPES:
        raise ValueError(f"Annotation {index + 1} type must be one of: text, line, arrow.")

    coord_system = str(value.get("coord_system") or "axes").strip().lower()
    if coord_system not in _ANNOTATION_COORD_SYSTEMS:
        raise ValueError(f"Annotation {index + 1} coord_system must be one of: data, axes.")

    color = str(value.get("color") or "#000000").strip() or "#000000"
    alpha = (
        1.0
        if value.get("alpha") in {None, ""}
        else _annotation_numeric(value.get("alpha"), field_name=f"Annotation {index + 1} alpha")
    )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Annotation {index + 1} alpha must be between 0 and 1.")
    zorder = (
        5.0
        if value.get("zorder") in {None, ""}
        else _annotation_numeric(
            value.get("zorder"),
            field_name=f"Annotation {index + 1} z-order",
        )
    )

    normalized: dict[str, Any] = {
        "id": str(value.get("id") or f"annotation:{index}").strip() or f"annotation:{index}",
        "type": annotation_type,
        "enabled": bool(value.get("enabled", True)),
        "name": normalize_plot_text(
            str(value.get("name") or f"{annotation_type.title()} {index + 1}").strip()
            or f"{annotation_type.title()} {index + 1}"
        ),
        "coord_system": coord_system,
        "color": color,
        "alpha": float(alpha),
        "zorder": float(zorder),
    }

    if annotation_type == "text":
        text_value = normalize_plot_text(str(value.get("text") or ""))
        if normalized["enabled"] and not text_value.strip():
            raise ValueError(f"Annotation {index + 1} text cannot be blank when enabled.")
        h_align = str(value.get("horizontal_align") or "center").strip().lower()
        if h_align not in _ANNOTATION_H_ALIGNS:
            raise ValueError(
                f"Annotation {index + 1} horizontal alignment must be left, center, or right."
            )
        v_align = str(value.get("vertical_align") or "center").strip().lower()
        if v_align not in _ANNOTATION_V_ALIGNS:
            raise ValueError(
                "Annotation "
                f"{index + 1} vertical alignment must be top, center, bottom, or baseline."
            )
        normalized.update(
            {
                "x": _annotation_numeric(
                    value.get("x", 0.5), field_name=f"Annotation {index + 1} x"
                ),
                "y": _annotation_numeric(
                    value.get("y", 0.5), field_name=f"Annotation {index + 1} y"
                ),
                "text": text_value,
                "font_size": (
                    12
                    if value.get("font_size") in {None, ""}
                    else int(
                        _annotation_positive_numeric(
                            value.get("font_size"),
                            field_name=f"Annotation {index + 1} font size",
                        )
                    )
                ),
                "rotation": (
                    0.0
                    if value.get("rotation") in {None, ""}
                    else _annotation_numeric(
                        value.get("rotation"),
                        field_name=f"Annotation {index + 1} rotation",
                    )
                ),
                "horizontal_align": h_align,
                "vertical_align": v_align,
            }
        )
        return normalized

    line_style = str(value.get("line_style") or "-").strip()
    if line_style not in _ANNOTATION_LINE_STYLES:
        raise ValueError(f"Annotation {index + 1} line style must be one of: -, --, -., :.")
    normalized.update(
        {
            "x1": _annotation_numeric(
                value.get("x1", 0.0), field_name=f"Annotation {index + 1} x1"
            ),
            "y1": _annotation_numeric(
                value.get("y1", 0.0), field_name=f"Annotation {index + 1} y1"
            ),
            "x2": _annotation_numeric(
                value.get("x2", 1.0), field_name=f"Annotation {index + 1} x2"
            ),
            "y2": _annotation_numeric(
                value.get("y2", 1.0), field_name=f"Annotation {index + 1} y2"
            ),
            "line_width": (
                1.5
                if value.get("line_width") in {None, ""}
                else _annotation_positive_numeric(
                    value.get("line_width"),
                    field_name=f"Annotation {index + 1} line width",
                )
            ),
            "line_style": line_style,
        }
    )
    if annotation_type == "arrow":
        arrow_style = str(value.get("arrow_style") or "->").strip()
        if arrow_style not in _ANNOTATION_ARROW_STYLES:
            raise ValueError(
                f"Annotation {index + 1} arrow style must be one of: ->, -|>, <->, simple, fancy."
            )
        normalized["arrow_style"] = arrow_style
        normalized["mutation_scale"] = (
            12.0
            if value.get("mutation_scale") in {None, ""}
            else _annotation_positive_numeric(
                value.get("mutation_scale"),
                field_name=f"Annotation {index + 1} mutation scale",
            )
        )
    return normalized


def _coerce_plot_annotations(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if not isinstance(value, list):
        raise ValueError("annotations must be a list of annotation objects.")
    return [_coerce_plot_annotation(item, index=index) for index, item in enumerate(value)]


def _render_plot_annotations(
    ax: Any, annotations: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    resolved = _coerce_plot_annotations(annotations)
    summaries: list[dict[str, Any]] = []
    for annotation in resolved:
        summary = {
            "id": str(annotation["id"]),
            "type": str(annotation["type"]),
            "name": str(annotation["name"]),
            "enabled": bool(annotation["enabled"]),
            "coord_system": str(annotation["coord_system"]),
        }
        if not bool(annotation["enabled"]):
            summary["status"] = "disabled"
            summaries.append(summary)
            continue

        transform = ax.transData if annotation["coord_system"] == "data" else ax.transAxes
        if annotation["type"] == "text":
            ax.text(
                float(annotation["x"]),
                float(annotation["y"]),
                normalize_plot_text(str(annotation["text"])),
                transform=transform,
                color=str(annotation["color"]),
                alpha=float(annotation["alpha"]),
                fontsize=int(annotation["font_size"]),
                rotation=float(annotation["rotation"]),
                ha=str(annotation["horizontal_align"]),
                va=str(annotation["vertical_align"]),
                zorder=float(annotation["zorder"]),
                clip_on=False,
            )
        elif annotation["type"] == "line":
            line_artist = matplotlib.lines.Line2D(
                [float(annotation["x1"]), float(annotation["x2"])],
                [float(annotation["y1"]), float(annotation["y2"])],
                color=str(annotation["color"]),
                alpha=float(annotation["alpha"]),
                linewidth=float(annotation["line_width"]),
                linestyle=str(annotation["line_style"]),
                transform=transform,
                zorder=float(annotation["zorder"]),
                clip_on=False,
            )
            ax.add_line(line_artist)
        else:
            ax.annotate(
                "",
                xy=(float(annotation["x2"]), float(annotation["y2"])),
                xytext=(float(annotation["x1"]), float(annotation["y1"])),
                xycoords=transform,
                textcoords=transform,
                arrowprops={
                    "arrowstyle": str(annotation["arrow_style"]),
                    "color": str(annotation["color"]),
                    "alpha": float(annotation["alpha"]),
                    "linewidth": float(annotation["line_width"]),
                    "linestyle": str(annotation["line_style"]),
                    "mutation_scale": float(annotation["mutation_scale"]),
                },
                zorder=float(annotation["zorder"]),
                annotation_clip=False,
            )
        summary["status"] = "ok"
        summaries.append(summary)
    return summaries


def with_style_overrides(
    *,
    base_style: PlotStyle = DEFAULT_PLOT_STYLE,
    figure_size: tuple[float, float] | None = None,
    dpi: int | None = None,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    title_font_size: int | None = None,
    title_pad: float | None = None,
    label_font_size: int | None = None,
    tick_font_size: int | None = None,
    legend_font_size: int | None = None,
    line_width: float | None = None,
    line_color: str | None = None,
    axes_border: bool | dict[str, bool] | None = None,
    grid: bool | None = None,
    grid_linestyle: str | None = None,
    grid_linewidth: float | None = None,
    grid_alpha: float | None = None,
) -> PlotStyle:
    """Return a :class:`PlotStyle` with explicit overrides applied."""
    updates: dict[str, Any] = {}
    current_font_defaults = default_plot_font_sizes(base_style.base_font_size)
    if figure_size is not None:
        updates["figure_size"] = figure_size
    if dpi is not None:
        updates["dpi"] = dpi
    if font_family is not None:
        updates["font_family"] = font_family
    if font_color is not None:
        updates["font_color"] = str(font_color)
    target_base_font_size = base_style.base_font_size if font_size is None else int(font_size)
    if font_size is not None:
        updates["base_font_size"] = target_base_font_size
    target_font_defaults = default_plot_font_sizes(target_base_font_size)
    for key, explicit_value in (
        ("title_font_size", title_font_size),
        ("label_font_size", label_font_size),
        ("tick_font_size", tick_font_size),
        ("legend_font_size", legend_font_size),
    ):
        if explicit_value is not None:
            updates[key] = int(explicit_value)
            continue
        if font_size is None:
            continue
        if int(getattr(base_style, key)) == int(current_font_defaults[key]):
            updates[key] = int(target_font_defaults[key])
    if title_pad is not None:
        updates["title_pad"] = float(title_pad)
    if line_width is not None:
        updates["line_width"] = line_width
    if line_color is not None:
        updates["line_color"] = line_color
    if axes_border is not None:
        updates["axes_border"] = axes_border
    if grid is not None:
        updates["grid"] = grid
    if grid_linestyle is not None:
        updates["grid_linestyle"] = grid_linestyle
    if grid_linewidth is not None:
        updates["grid_linewidth"] = grid_linewidth
    if grid_alpha is not None:
        updates["grid_alpha"] = grid_alpha
    return replace(base_style, **updates)


def resolve_series_labels(
    default_labels: list[str],
    series_labels: list[str] | None,
    *,
    series_kind: str,
) -> list[str]:
    """Return validated series labels for multi-line plots."""
    if series_labels is None:
        return list(default_labels)
    if len(series_labels) != len(default_labels):
        raise ValueError(
            "series_labels count must match the number of plotted "
            f"{series_kind} series ({len(default_labels)})."
        )
    labels = [label.strip() for label in series_labels]
    if any(not label for label in labels):
        raise ValueError("series_labels cannot contain empty values.")
    return labels


def resolve_single_series_options(
    *,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
) -> SingleSeriesPlotOptions:
    """Resolve single-series overrides from list-based plot settings."""
    return SingleSeriesPlotOptions(
        line_color=line_colors[0] if line_colors else None,
        line_visible=True if not series_enabled else bool(series_enabled[0]),
        line_width_override=series_line_widths[0] if series_line_widths else None,
        line_marker=series_markers[0] if series_markers else None,
        normalization_mode=series_normalization_modes[0] if series_normalization_modes else None,
        normalization_value=series_normalization_values[0] if series_normalization_values else None,
        normalization_x_ref=series_normalization_x_refs[0] if series_normalization_x_refs else None,
    )


def _normalize_error_stat_name(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    if token not in {"sample_std", "sample_sem", "block_std", "block_sem"}:
        raise ValueError(
            "Error statistic must be one of: sample_std, sample_sem, block_std, block_sem."
        )
    return token


_FRIENDLY_STAT_LABELS: dict[str, str] = {
    "sample_sem": "Sample SEM",
    "sample_std": "Sample Std. Dev.",
    "block_sem": "Block SEM",
    "block_std": "Block Std. Dev.",
}


def _friendly_stat_label(stat: str) -> str:
    return _FRIENDLY_STAT_LABELS.get(stat, stat)


def _normalize_error_style_name(value: str | None) -> str:
    token = "band" if value is None else str(value).strip().lower()
    if token not in {"band", "whiskers"}:
        raise ValueError("Error style must be one of: band, whiskers.")
    return token


def _resolve_effective_error_stat(
    configured_stat: str | None,
    available_stats: Sequence[str] | None,
) -> str | None:
    configured = _normalize_error_stat_name(configured_stat)
    available = [
        token
        for token in (str(value).strip().lower() for value in (available_stats or ()))
        if token in {"sample_std", "sample_sem", "block_std", "block_sem"}
    ]
    if configured is not None and configured in available:
        return configured
    for candidate in ("block_sem", "sample_sem", "block_std", "sample_std"):
        if candidate in available:
            return candidate
    return configured


def _coerce_error_config(value: Any) -> SeriesErrorConfig:
    if not isinstance(value, dict):
        return SeriesErrorConfig()
    return SeriesErrorConfig(
        enabled=bool(value.get("enabled", False)),
        stat=_normalize_error_stat_name(value.get("stat")),
        style=_normalize_error_style_name(value.get("style")),
        color=None if value.get("color") in {None, ""} else str(value.get("color")),
        label_override=(
            None if value.get("label_override") in {None, ""} else str(value.get("label_override"))
        ),
        show_in_legend=bool(value.get("show_in_legend", False)),
    )


@dataclass(frozen=True)
class SeriesErrorAvailability:
    """Resolved availability state for one GUI/plot uncertainty selector."""

    available_stats: list[str]
    default_stat: str | None
    selector_enabled: bool
    reason: str | None = None


def _coerce_cumulative_config(value: Any) -> SeriesCumulativeConfig:
    if not isinstance(value, dict):
        return SeriesCumulativeConfig()
    _color = value.get("color")
    _alpha_raw = value.get("alpha")
    _lw_raw = value.get("line_width")
    _ls = value.get("line_style")
    try:
        _alpha = float(_alpha_raw) if _alpha_raw is not None and str(_alpha_raw).strip() else None
    except (ValueError, TypeError):
        _alpha = None
    try:
        _lw = float(_lw_raw) if _lw_raw is not None and str(_lw_raw).strip() else None
    except (ValueError, TypeError):
        _lw = None
    return SeriesCumulativeConfig(
        enabled=bool(value.get("enabled", False)),
        label_override=(
            None if value.get("label_override") in {None, ""} else str(value.get("label_override"))
        ),
        show_in_legend=bool(value.get("show_in_legend", True)),
        color=str(_color).strip() or None if _color is not None and str(_color).strip() else None,
        alpha=_alpha,
        line_width=_lw,
        line_style=str(_ls).strip() or None if _ls is not None and str(_ls).strip() else None,
    )


def _coerce_integration_config(value: Any) -> IntegrationConfig:
    if not isinstance(value, dict):
        return IntegrationConfig()
    enabled = bool(value.get("enabled", False))
    source = str(value.get("source") or "plotted").strip().lower()
    if source not in {"plotted", "raw"}:
        raise ValueError("Integration data source must be 'plotted' or 'raw'.")
    target = str(value.get("target") or "selected").strip().lower()
    if target != "selected":
        raise ValueError("Integration target must be 'selected'.")

    def _optional_numeric(key: str) -> float | None:
        raw = value.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Integration {key} must be numeric.") from exc

    x_min = _optional_numeric("x_min")
    x_max = _optional_numeric("x_max")
    if x_min is not None and x_max is not None and x_min >= x_max:
        raise ValueError("Integration x-min must be smaller than integration x-max.")
    baseline = _optional_numeric("baseline")
    alpha = _optional_numeric("alpha")
    if alpha is None:
        alpha = 0.25
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Integration fill alpha must be between 0 and 1.")
    color = str(value.get("color") or "").strip() or None
    target_series_id = str(value.get("target_series_id") or "").strip() or None
    return IntegrationConfig(
        enabled=enabled,
        source=source,
        target=target,
        target_series_id=target_series_id,
        x_min=x_min,
        x_max=x_max,
        baseline=0.0 if baseline is None else baseline,
        color=color,
        alpha=float(alpha),
    )


def _integration_region(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_min: float | None,
    x_max: float | None,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_finite = np.asarray(x_values[finite_mask], dtype=float)
    y_finite = np.asarray(y_values[finite_mask], dtype=float)
    if x_finite.size < 2:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            {
                "status": "empty",
                "reason": "Fewer than two finite points are available for integration.",
                "point_count": int(x_finite.size),
            },
        )

    order = np.argsort(x_finite, kind="mergesort")
    x_sorted = x_finite[order]
    y_sorted = y_finite[order]
    x_unique, unique_index = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_index]
    if x_unique.size < 2:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            {
                "status": "empty",
                "reason": "Integration requires at least two distinct finite x-values.",
                "point_count": int(x_unique.size),
            },
        )

    data_min = float(x_unique[0])
    data_max = float(x_unique[-1])
    left = data_min if x_min is None else max(float(x_min), data_min)
    right = data_max if x_max is None else min(float(x_max), data_max)
    if left >= right:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            {
                "status": "empty",
                "reason": "Requested integration range does not overlap the available data.",
                "point_count": 0,
            },
        )

    inside_mask = (x_unique > left) & (x_unique < right)
    region_x = np.concatenate(
        (
            np.asarray([left], dtype=float),
            x_unique[inside_mask],
            np.asarray([right], dtype=float),
        )
    )
    region_y = np.interp(region_x, x_unique, y_unique)
    delta_y = region_y - float(baseline)
    signed_area = _trapezoid_integral(delta_y, region_x)
    absolute_area = _trapezoid_integral(np.abs(delta_y), region_x)
    return (
        region_x,
        region_y,
        {
            "status": "ok",
            "x_min": float(region_x[0]),
            "x_max": float(region_x[-1]),
            "point_count": int(region_x.size),
            "baseline": float(baseline),
            "area": signed_area,
            "signed_area": signed_area,
            "absolute_area": absolute_area,
        },
    )


def resolve_series_error_availability(
    *,
    supported_for_view: bool,
    available_stats: Sequence[str] | None,
    error_status: str | None = None,
    error_reason: str | None = None,
) -> SeriesErrorAvailability:
    """Resolve a user-facing availability state for one uncertainty selector."""
    if not supported_for_view:
        return SeriesErrorAvailability(
            available_stats=[],
            default_stat=None,
            selector_enabled=False,
            reason="Error overlays are only available for 1-D line-based views.",
        )

    resolved = [
        token
        for token in (str(value).strip().lower() for value in (available_stats or ()))
        if token in {"sample_std", "sample_sem", "block_std", "block_sem"}
    ]
    unique_available = list(dict.fromkeys(resolved))
    default_stat = (
        "block_sem"
        if "block_sem" in unique_available
        else (
            "sample_sem"
            if "sample_sem" in unique_available
            else unique_available[0]
            if unique_available
            else None
        )
    )
    normalized_status = str(error_status or "").strip().lower()
    normalized_reason = str(error_reason or "").strip() or None
    if normalized_status == "rebinned_saved_profile" and not unique_available:
        return SeriesErrorAvailability(
            available_stats=[],
            default_stat=None,
            selector_enabled=False,
            reason=normalized_reason
            or "Saved-profile uncertainty is unavailable after x rebinning.",
        )
    if unique_available:
        if len(unique_available) == 1:
            return SeriesErrorAvailability(
                available_stats=unique_available,
                default_stat=default_stat,
                selector_enabled=False,
                reason=normalized_reason
                or f"Only '{unique_available[0]}' is available for this series.",
            )
        return SeriesErrorAvailability(
            available_stats=unique_available,
            default_stat=default_stat,
            selector_enabled=True,
            reason=normalized_reason if normalized_status == "rebinned_saved_profile" else None,
        )
    return SeriesErrorAvailability(
        available_stats=[],
        default_stat=None,
        selector_enabled=False,
        reason=normalized_reason or "No uncertainty statistics are available for this series.",
    )


def _coerce_error_config_list(
    series_count: int,
    configs: list[dict[str, Any] | None] | None,
) -> list[SeriesErrorConfig]:
    if configs is None:
        return [SeriesErrorConfig() for _ in range(series_count)]
    if len(configs) != series_count:
        raise ValueError(
            f"series_error_configs count must match the number of plotted series ({series_count})."
        )
    return [_coerce_error_config(config) for config in configs]


def _coerce_cumulative_config_list(
    series_count: int,
    configs: list[dict[str, Any] | None] | None,
) -> list[SeriesCumulativeConfig]:
    if configs is None:
        return [SeriesCumulativeConfig() for _ in range(series_count)]
    if len(configs) != series_count:
        raise ValueError(
            "series_cumulative_configs count must match the number of plotted series "
            f"({series_count})."
        )
    return [_coerce_cumulative_config(config) for config in configs]


def _scale_series_statistics(stats: SeriesStatistics, *, scale: float) -> SeriesStatistics:
    magnitude = abs(float(scale))
    return replace(
        stats,
        sample_std=np.asarray(stats.sample_std, dtype=float) * magnitude,
        sample_sem=np.asarray(stats.sample_sem, dtype=float) * magnitude,
        block_std=(
            None
            if stats.block_std is None
            else np.asarray(stats.block_std, dtype=float) * magnitude
        ),
        block_sem=(
            None
            if stats.block_sem is None
            else np.asarray(stats.block_sem, dtype=float) * magnitude
        ),
    )


def _group_xy_series_with_statistics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_width: float | None,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray, SeriesStatistics]:
    if x.shape != y.shape:
        raise ValueError("x and y data must have the same shape.")
    if x.size == 0:
        return (np.empty(0, dtype=float), np.empty(0, dtype=float), _empty_series_statistics())
    if bin_width is not None and bin_width <= 0:
        raise ValueError("x_bin_width must be positive.")

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return (np.empty(0, dtype=float), np.empty(0, dtype=float), _empty_series_statistics())

    x_clean = np.asarray(x[mask], dtype=float)
    y_clean = np.asarray(y[mask], dtype=float)
    order = np.argsort(x_clean, kind="mergesort")
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]

    if bin_width is None:
        group_ids = (
            np.cumsum(np.concatenate(([True], x_sorted[1:] != x_sorted[:-1]))).astype(np.int64) - 1
        )
    else:
        start = float(x_sorted[0])
        group_ids = np.floor((x_sorted - start) / float(bin_width)).astype(np.int64)
    unique_groups = np.unique(group_ids)

    x_out = np.empty(unique_groups.size, dtype=float)
    y_out = np.empty(unique_groups.size, dtype=float)
    point_count = np.empty(unique_groups.size, dtype=int)
    sample_std = np.full(unique_groups.size, np.nan, dtype=float)
    sample_sem = np.full(unique_groups.size, np.nan, dtype=float)
    for out_index, group_id in enumerate(unique_groups):
        group_mask = group_ids == group_id
        x_group = x_sorted[group_mask]
        y_group = y_sorted[group_mask]
        x_out[out_index] = float(np.mean(x_group))
        y_out[out_index] = _reduce_values(y_group, reducer=reducer)
        point_count[out_index] = int(y_group.size)
        if y_group.size > 1:
            sample_std[out_index] = float(np.std(y_group, ddof=1))
            sample_sem[out_index] = sample_std[out_index] / np.sqrt(float(y_group.size))

    statistics = SeriesStatistics(
        point_count=point_count,
        sample_n=point_count.copy(),
        sample_std=sample_std,
        sample_sem=sample_sem,
    )
    return x_out, y_out, statistics


def _subset_series_statistics(stats: SeriesStatistics, mask: np.ndarray) -> SeriesStatistics:
    return SeriesStatistics(
        point_count=np.asarray(stats.point_count, dtype=int)[mask],
        sample_n=np.asarray(stats.sample_n, dtype=int)[mask],
        sample_std=np.asarray(stats.sample_std, dtype=float)[mask],
        sample_sem=np.asarray(stats.sample_sem, dtype=float)[mask],
        block_n=None if stats.block_n is None else np.asarray(stats.block_n, dtype=int)[mask],
        block_std=(
            None if stats.block_std is None else np.asarray(stats.block_std, dtype=float)[mask]
        ),
        block_sem=(
            None if stats.block_sem is None else np.asarray(stats.block_sem, dtype=float)[mask]
        ),
    )


def _rebin_persisted_series_statistics(
    x: np.ndarray,
    y: np.ndarray,
    stats: SeriesStatistics,
    *,
    bin_width: float,
    reducer: str,
) -> tuple[SeriesStatistics | None, str | None]:
    if x.shape != y.shape:
        raise ValueError("x and y data must have the same shape.")
    if bin_width <= 0:
        raise ValueError("x_bin_width must be positive.")
    if reducer != "mean":
        return None, "Saved-profile uncertainty can only be rebinned for mean sectioning."
    if x.shape != np.asarray(stats.sample_n).shape:
        return None, "Saved-profile statistics shape does not match the plotted series."

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return _empty_series_statistics(), None

    x_clean = np.asarray(x[mask], dtype=float)
    y_clean = np.asarray(y[mask], dtype=float)
    stats_clean = _subset_series_statistics(stats, mask)
    sample_n = np.asarray(stats_clean.sample_n, dtype=int)
    sample_std = np.asarray(stats_clean.sample_std, dtype=float)
    point_count = np.asarray(stats_clean.point_count, dtype=int)

    if sample_n.shape != y_clean.shape or sample_std.shape != y_clean.shape:
        return None, "Saved-profile statistics shape does not match the rebinned series."

    order = np.argsort(x_clean, kind="mergesort")
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]
    sample_n_sorted = sample_n[order]
    sample_std_sorted = sample_std[order]
    point_count_sorted = point_count[order]

    start = float(x_sorted[0])
    bin_index = np.floor((x_sorted - start) / float(bin_width)).astype(np.int64)
    unique_bins = np.unique(bin_index)

    rebinned_point_count = np.empty(unique_bins.size, dtype=int)
    rebinned_sample_n = np.empty(unique_bins.size, dtype=int)
    rebinned_sample_std = np.full(unique_bins.size, np.nan, dtype=float)
    rebinned_sample_sem = np.full(unique_bins.size, np.nan, dtype=float)

    for out_index, group_id in enumerate(unique_bins):
        group_mask = bin_index == group_id
        y_group = y_sorted[group_mask]
        n_group = sample_n_sorted[group_mask]
        std_group = sample_std_sorted[group_mask]
        point_group = point_count_sorted[group_mask]

        finite_counts = np.isfinite(y_group) & (n_group > 0)
        if not np.any(finite_counts):
            rebinned_point_count[out_index] = int(np.sum(point_group))
            rebinned_sample_n[out_index] = 0
            continue

        y_valid = np.asarray(y_group[finite_counts], dtype=float)
        n_valid = np.asarray(n_group[finite_counts], dtype=int)
        std_valid = np.asarray(std_group[finite_counts], dtype=float)
        point_valid = np.asarray(point_group[finite_counts], dtype=int)

        sample_sum = y_valid * n_valid.astype(float)
        sample_sumsq = np.empty(y_valid.size, dtype=float)
        for idx, (mean_value, n_value, std_value) in enumerate(zip(y_valid, n_valid, std_valid)):
            if n_value <= 1 or not np.isfinite(std_value):
                sample_sumsq[idx] = float(n_value) * float(mean_value) ** 2
            else:
                sample_sumsq[idx] = (float(std_value) ** 2) * float(n_value - 1) + float(
                    n_value
                ) * float(mean_value) ** 2

        total_n = int(np.sum(n_valid))
        rebinned_sample_n[out_index] = total_n
        rebinned_point_count[out_index] = int(np.sum(point_valid))
        if total_n <= 0:
            continue
        total_sum = float(np.sum(sample_sum))
        total_sumsq = float(np.sum(sample_sumsq))
        mean_value = total_sum / float(total_n)
        if total_n > 1:
            variance_numerator = max(total_sumsq - float(total_n) * mean_value**2, 0.0)
            std_value = np.sqrt(variance_numerator / float(total_n - 1))
            rebinned_sample_std[out_index] = float(std_value)
            rebinned_sample_sem[out_index] = float(std_value) / np.sqrt(float(total_n))

    return (
        SeriesStatistics(
            point_count=rebinned_point_count,
            sample_n=rebinned_sample_n,
            sample_std=rebinned_sample_std,
            sample_sem=rebinned_sample_sem,
        ),
        None,
    )


def _statistics_error_values(stats: SeriesStatistics, *, stat_name: str) -> np.ndarray | None:
    if stat_name == "sample_std":
        return np.asarray(stats.sample_std, dtype=float)
    if stat_name == "sample_sem":
        return np.asarray(stats.sample_sem, dtype=float)
    if stat_name == "block_std":
        return None if stats.block_std is None else np.asarray(stats.block_std, dtype=float)
    if stat_name == "block_sem":
        return None if stats.block_sem is None else np.asarray(stats.block_sem, dtype=float)
    raise ValueError(f"Unsupported error statistic '{stat_name}'.")


def _is_interactive_backend(backend: str) -> bool:
    """Return whether a backend name refers to an interactive backend."""
    return backend.lower().split(".")[-1] in INTERACTIVE_BACKENDS


def normalize_backend_name(name: str) -> str:
    """Normalize and validate a backend name."""
    stripped = name.strip()
    if not stripped:
        raise ValueError("Backend name cannot be empty.")

    lowered = stripped.lower()
    if lowered in BACKEND_ALIASES:
        return BACKEND_ALIASES[lowered]

    options = ", ".join(CANONICAL_INTERACTIVE_BACKENDS)
    suggestion = difflib.get_close_matches(stripped, CANONICAL_INTERACTIVE_BACKENDS, n=1)
    if suggestion:
        raise ValueError(
            f"Unsupported backend '{name}'. Did you mean '{suggestion[0]}'? "
            f"Allowed values: {options}."
        )

    raise ValueError(f"Unsupported backend '{name}'. Allowed values: {options}.")


def _has_graphical_display() -> bool:
    """Return whether a graphical display appears available."""
    if sys.platform in {"win32", "darwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def configure_matplotlib_backend(
    *,
    interactive: bool,
    preferred_backend: str | None = None,
) -> str:
    """Configure Matplotlib backend before pyplot import/figure creation."""
    global _BACKEND_CONFIGURED, _CONFIGURED_BACKEND

    if _BACKEND_CONFIGURED and _CONFIGURED_BACKEND is not None:
        configured_is_interactive = _is_interactive_backend(_CONFIGURED_BACKEND)
        if configured_is_interactive == bool(interactive):
            return _CONFIGURED_BACKEND

    if interactive:
        candidates = list(CANONICAL_INTERACTIVE_BACKENDS)
        if preferred_backend:
            normalized = normalize_backend_name(preferred_backend)
            candidates = [normalized, *[c for c in candidates if c != normalized]]

        if not _has_graphical_display():
            raise RuntimeError(
                "Interactive plotting requested but no graphical display is available. "
                "Use X11/Wayland forwarding, or run with --no-show and --output."
            )

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                matplotlib.use(candidate, force=True)
                active = matplotlib.get_backend()
                if _is_interactive_backend(active):
                    _BACKEND_CONFIGURED = True
                    _CONFIGURED_BACKEND = active
                    LOGGER.debug("Configured Matplotlib backend '%s'.", active)
                    return active
            except Exception as exc:  # pragma: no cover - environment dependent
                last_error = exc
                LOGGER.debug("Could not activate backend '%s': %s", candidate, exc)

        active = matplotlib.get_backend()
        message = (
            f"Interactive plotting requested, but active backend '{active}' is non-interactive. "
            f"Attempted backends: {', '.join(candidates)}. "
            "Install an interactive backend (Tk/Qt/GTK) or set MPLBACKEND accordingly."
        )
        if last_error is not None:
            message = f"{message} Last backend error: {last_error}"
        raise RuntimeError(message)

    matplotlib.use("Agg", force=True)
    active = matplotlib.get_backend()
    _BACKEND_CONFIGURED = True
    _CONFIGURED_BACKEND = active
    LOGGER.debug("Configured Matplotlib backend '%s'.", active)
    return active


def ensure_interactive_backend(preferred_backend: str | None = None) -> str:
    """Backwards-compatible wrapper for interactive backend configuration."""
    return configure_matplotlib_backend(
        interactive=True,
        preferred_backend=preferred_backend,
    )


def _to_float_list(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


def _normalize_tick_axis(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"x", "y", "both"}:
        return token
    return "both"


def _normalize_minor_ticks_mode(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"on", "off", "auto"}:
        return token
    return "auto"


def _sanitize_line_collection_kwargs(line_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip line-only marker kwargs that ``LineCollection`` does not accept."""
    resolved = {} if line_kwargs is None else dict(line_kwargs)
    if "lw" in resolved and "linewidths" not in resolved:
        resolved["linewidths"] = resolved.pop("lw")
    for key in (
        "label",
        "color",
        "c",
        "marker",
        "markersize",
        "ms",
        "markeredgecolor",
        "mec",
        "markeredgewidth",
        "mew",
        "markerfacecolor",
        "mfc",
        "markerfacecoloralt",
        "fillstyle",
        "markevery",
    ):
        resolved.pop(key, None)
    return resolved


def _extract_tick_controls(
    tick_params_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(tick_params_kwargs, dict):
        return {}, "both", "auto"

    resolved = dict(tick_params_kwargs)
    axis_hint_raw = resolved.pop("_ticks_axis", None)
    minor_mode_raw = resolved.pop("_minor_ticks_mode", None)
    resolved.pop("_x_tick_params", None)
    resolved.pop("_y_tick_params", None)
    resolved.pop("_x_minor_ticks_mode", None)
    resolved.pop("_y_minor_ticks_mode", None)
    resolved.pop("_x_ticks_visible", None)
    resolved.pop("_y_ticks_visible", None)

    axis_hint = _normalize_tick_axis(axis_hint_raw) if axis_hint_raw is not None else "both"
    if axis_hint_raw is None and "axis" in resolved:
        axis_hint = _normalize_tick_axis(resolved["axis"])

    minor_mode = _normalize_minor_ticks_mode(minor_mode_raw)
    return resolved, axis_hint, minor_mode


def _resolve_tick_visibility(
    tick_params_kwargs: dict[str, Any] | None,
    ticks_visible: bool | None,
    tick_axis_hint: str,
) -> tuple[bool, bool]:
    if isinstance(tick_params_kwargs, dict):
        raw_x_visible = tick_params_kwargs.get("_x_ticks_visible")
        raw_y_visible = tick_params_kwargs.get("_y_ticks_visible")
        if raw_x_visible is not None or raw_y_visible is not None:
            return (
                True if raw_x_visible is None else bool(raw_x_visible),
                True if raw_y_visible is None else bool(raw_y_visible),
            )

    if ticks_visible is False:
        return (
            tick_axis_hint not in {"both", "x"},
            tick_axis_hint not in {"both", "y"},
        )
    if ticks_visible is True and tick_axis_hint in {"x", "y"}:
        return (
            tick_axis_hint == "x",
            tick_axis_hint == "y",
        )
    return True, True


def _visible_axes_data_bounds(ax: Any) -> tuple[float, float, float, float] | None:
    x_mins: list[float] = []
    x_maxs: list[float] = []
    y_mins: list[float] = []
    y_maxs: list[float] = []

    for line in getattr(ax, "lines", ()):
        if not bool(line.get_visible()):
            continue
        x_values = np.asarray(line.get_xdata(orig=False), dtype=float)
        y_values = np.asarray(line.get_ydata(orig=False), dtype=float)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            continue
        x_visible = x_values[finite_mask]
        y_visible = y_values[finite_mask]
        x_mins.append(float(np.min(x_visible)))
        x_maxs.append(float(np.max(x_visible)))
        y_mins.append(float(np.min(y_visible)))
        y_maxs.append(float(np.max(y_visible)))

    for collection in getattr(ax, "collections", ()):
        if not bool(collection.get_visible()):
            continue
        try:
            data_limits = collection.get_datalim(ax.transData)
            points = np.asarray(data_limits.get_points(), dtype=float)
        except Exception:
            continue
        if points.shape != (2, 2) or not np.all(np.isfinite(points)):
            continue
        x_mins.append(float(points[0, 0]))
        x_maxs.append(float(points[1, 0]))
        y_mins.append(float(points[0, 1]))
        y_maxs.append(float(points[1, 1]))

    if not x_mins or not y_mins:
        return None
    return min(x_mins), max(x_maxs), min(y_mins), max(y_maxs)


def _apply_figure_kwargs(fig: Any, figure_kwargs: dict[str, Any] | None) -> float | None:
    if figure_kwargs is None:
        return None
    resolved = dict(figure_kwargs)
    alpha = resolved.pop("alpha", None)
    if resolved:
        fig.set(**resolved)
    if alpha is not None:
        resolved_alpha = float(alpha)
        fig.patch.set_alpha(resolved_alpha)
        return resolved_alpha
    return None


def _apply_axes_face_alpha(ax: Any, alpha: float | None) -> None:
    if alpha is None:
        return
    ax.patch.set_alpha(float(alpha))


def _axis_tick_params(
    tick_params_kwargs: dict[str, Any] | None,
    axis: str,
) -> dict[str, Any]:
    if not isinstance(tick_params_kwargs, dict):
        return {}
    raw = tick_params_kwargs.get(f"_{axis}_tick_params")
    if not isinstance(raw, dict):
        return {}
    allowed = {"direction", "length", "width", "labelsize", "colors", "color"}
    return {
        key: value
        for key, value in raw.items()
        if key in allowed and value is not None and str(value).strip() != ""
    }


def _minor_tick_mode(tick_params_kwargs: dict[str, Any] | None, axis: str, fallback: str) -> str:
    if not isinstance(tick_params_kwargs, dict):
        return fallback
    raw = tick_params_kwargs.get(f"_{axis}_minor_ticks_mode", fallback)
    return _normalize_minor_ticks_mode(raw)


def _apply_minor_tick_modes(
    ax: Any,
    *,
    tick_params_kwargs: dict[str, Any] | None,
    fallback_mode: str,
) -> None:
    x_mode = _minor_tick_mode(tick_params_kwargs, "x", fallback_mode)
    y_mode = _minor_tick_mode(tick_params_kwargs, "y", fallback_mode)
    if x_mode == "on" or y_mode == "on":
        ax.minorticks_on()
    elif x_mode == "off" and y_mode == "off":
        ax.minorticks_off()
    if x_mode == "off" and y_mode == "on":
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    if y_mode == "off" and x_mode == "on":
        ax.tick_params(axis="y", which="minor", left=False, right=False)


def _capture_plot_state(
    *,
    ax: Any,
    style: PlotStyle,
    line_colors: list[str],
    line_labels: list[str],
    line_markers: list[str],
    legend_loc: str,
    grid_kwargs: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None,
    annotation_summaries: list[dict[str, Any]] | None = None,
) -> None:
    if capture_state is None:
        return

    legend = ax.get_legend()
    legend_title = None
    if legend is not None:
        title_obj = legend.get_title()
        if title_obj is not None:
            title_text = str(title_obj.get_text())
            legend_title = title_text if title_text else None
    figure = ax.figure
    try:
        import matplotlib.colors as mcolors

        facecolor_value = str(mcolors.to_hex(figure.get_facecolor(), keep_alpha=False))
    except Exception:
        facecolor_value = str(figure.get_facecolor())
    first_line = ax.lines[0] if getattr(ax, "lines", None) else None
    legend_kwargs = None
    if legend is not None:
        legend_title_fontsize = None
        title_obj = legend.get_title()
        if title_obj is not None:
            try:
                legend_title_fontsize = int(round(float(title_obj.get_fontsize())))
            except (TypeError, ValueError):
                legend_title_fontsize = None
        legend_kwargs = {
            "frameon": bool(legend.get_frame_on()),
            "ncols": int(getattr(legend, "_ncols", 1)),
        }
        if legend_title_fontsize is not None and legend_title is not None:
            legend_kwargs["title_fontsize"] = legend_title_fontsize
    axes_kwargs = {
        "xmargin": float(ax.get_xmargin()),
        "ymargin": float(ax.get_ymargin()),
    }
    line_kwargs = None
    line_color = None
    if first_line is not None:
        line_color = str(first_line.get_color())
        line_alpha = first_line.get_alpha()
        line_kwargs = {
            "linestyle": str(first_line.get_linestyle()),
            "alpha": None if line_alpha is None else float(line_alpha),
            "markersize": float(first_line.get_markersize()),
        }
    axes_border_states = {name: spine.get_visible() for name, spine in ax.spines.items()}
    axes_border_state: bool | dict[str, bool]
    if all(axes_border_states.values()):
        axes_border_state = True
    elif not any(axes_border_states.values()):
        axes_border_state = False
    else:
        axes_border_state = dict(axes_border_states)

    capture_state.clear()
    capture_state.update(
        {
            "figure": figure,
            "axes": ax,
            "title": str(ax.get_title()),
            "title_visible": bool(ax.title.get_visible() and bool(str(ax.get_title()).strip())),
            "x_label": str(ax.get_xlabel()),
            "y_label": str(ax.get_ylabel()),
            "x_scale": str(ax.get_xscale()),
            "y_scale": str(ax.get_yscale()),
            "x_lim": _to_float_list(ax.get_xlim()),
            "y_lim": _to_float_list(ax.get_ylim()),
            "x_ticks": _to_float_list(ax.get_xticks()),
            "y_ticks": _to_float_list(ax.get_yticks()),
            "ticks": bool(
                any(label.get_visible() for label in ax.get_xticklabels() + ax.get_yticklabels())
            ),
            "legend": legend is not None,
            "legend_title": legend_title,
            "legend_loc": legend_loc,
            "legend_kwargs": legend_kwargs,
            "line_colors": list(line_colors),
            "line_color": line_color,
            "markers": any(
                marker not in {"", "None", "none", " ", "NoneType"} for marker in line_markers
            ),
            "series_labels": [normalize_plot_text(str(label)) for label in line_labels],
            "figsize": [float(style.figure_size[0]), float(style.figure_size[1])],
            "dpi": int(style.dpi),
            "font_family": style.font_family,
            "font_size": int(style.base_font_size),
            "title_font_size": int(style.title_font_size),
            "title_pad": float(style.title_pad),
            "label_font_size": int(style.label_font_size),
            "tick_font_size": int(style.tick_font_size),
            "legend_font_size": int(style.legend_font_size),
            "line_width": float(style.line_width),
            "line_kwargs": line_kwargs,
            "axes_kwargs": axes_kwargs,
            "axes_border": axes_border_state,
            "x_margin": float(ax.get_xmargin()),
            "y_margin": float(ax.get_ymargin()),
            "x_label_pad": float(ax.xaxis.labelpad),
            "y_label_pad": float(ax.yaxis.labelpad),
            "figure_kwargs": {
                "facecolor": facecolor_value,
            },
            "grid": bool(style.grid),
            "grid_linestyle": style.grid_linestyle,
            "grid_linewidth": float(style.grid_linewidth),
            "grid_alpha": float(style.grid_alpha),
            "grid_kwargs": None if grid_kwargs is None else dict(grid_kwargs),
            "annotations_summary": list(annotation_summaries or []),
        }
    )


def _apply_axes_border(ax: Any, *, visible: bool | dict[str, bool]) -> None:
    if isinstance(visible, dict):
        for name, spine in ax.spines.items():
            spine.set_visible(bool(visible.get(name, True)))
    else:
        for spine in ax.spines.values():
            spine.set_visible(bool(visible))


def _capture_heatmap_state(
    *,
    ax: Any,
    style: PlotStyle,
    capture_state: dict[str, Any] | None,
    mesh: Any,
    colorbar: Any,
    legend_loc: str = "best",
    extra_state: dict[str, Any] | None = None,
    annotation_summaries: list[dict[str, Any]] | None = None,
) -> None:
    _capture_plot_state(
        ax=ax,
        style=style,
        line_colors=[],
        line_labels=[],
        line_markers=[],
        legend_loc=legend_loc,
        capture_state=capture_state,
        annotation_summaries=annotation_summaries,
    )
    if capture_state is None:
        return
    capture_state["figure"] = ax.figure
    capture_state["axes"] = ax
    capture_state["heatmap_artist"] = mesh
    capture_state["heatmap_colorbar"] = colorbar
    if extra_state is not None:
        capture_state.update(dict(extra_state))


def _resolve_reducer_name(value: str | None) -> str:
    token = "mean" if value is None else str(value).strip().lower()
    if token not in {"mean", "median", "sum", "min", "max"}:
        raise ValueError("x_bin_reducer must be one of: mean, median, sum, min, max.")
    return token


def _reduce_values(values: np.ndarray, *, reducer: str) -> float:
    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "median":
        return float(np.median(values))
    if reducer == "sum":
        return float(np.sum(values))
    if reducer == "min":
        return float(np.min(values))
    if reducer == "max":
        return float(np.max(values))
    raise ValueError(f"Unsupported reducer '{reducer}'.")


def _rebin_xy_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_width: float,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    if x.shape != y.shape:
        raise ValueError("x and y data must have the same shape.")
    if x.size == 0:
        return x, y
    if bin_width <= 0:
        raise ValueError("x_bin_width must be positive.")
    _validate_section_width_request(x, bin_width=bin_width)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return x, y

    x_clean = np.asarray(x[mask], dtype=float)
    y_clean = np.asarray(y[mask], dtype=float)
    order = np.argsort(x_clean, kind="mergesort")
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]

    start = float(x_sorted[0])
    bin_index = np.floor((x_sorted - start) / float(bin_width)).astype(np.int64)
    unique_bins = np.unique(bin_index)

    x_out = np.empty(unique_bins.size, dtype=float)
    y_out = np.empty(unique_bins.size, dtype=float)
    for out_index, group_id in enumerate(unique_bins):
        group_mask = bin_index == group_id
        x_group = x_sorted[group_mask]
        y_group = y_sorted[group_mask]
        x_out[out_index] = float(np.mean(x_group))
        y_out[out_index] = _reduce_values(y_group, reducer=reducer)

    return x_out, y_out


def _resolve_meaningful_source_bin_width(x: np.ndarray) -> float | None:
    finite_x = np.asarray(x[np.isfinite(x)], dtype=float)
    if finite_x.size < 2:
        return None
    unique_sorted = np.unique(np.sort(finite_x, kind="mergesort"))
    if unique_sorted.size < 2:
        return None
    diffs = np.diff(unique_sorted)
    if np.any(~np.isfinite(diffs)) or np.any(diffs <= 0.0):
        return None
    first = float(diffs[0])
    if np.allclose(diffs, first, rtol=1.0e-6, atol=1.0e-12):
        return first
    return None


def _validate_section_width_request(x: np.ndarray, *, bin_width: float) -> None:
    if bin_width <= 0.0:
        raise ValueError("x_bin_width must be positive.")

    finite_x = np.asarray(x[np.isfinite(x)], dtype=float)
    if finite_x.size == 0:
        return

    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))
    x_range = float(x_max - x_min)
    if x_range <= 1.0e-12:
        raise ValueError(
            f"Section width {float(bin_width):.6g} cannot be applied because the available x-range "
            f"is degenerate ({x_range:.6g})."
        )
    if float(bin_width) > x_range:
        raise ValueError(
            f"Section width {float(bin_width):.6g} is larger than the available x-range "
            f"{x_range:.6g}. Use a width <= {x_range:.6g}."
        )

    source_bin_width = _resolve_meaningful_source_bin_width(finite_x)
    if source_bin_width is not None and float(bin_width) + 1.0e-12 < source_bin_width:
        raise ValueError(
            f"Section width {float(bin_width):.6g} is smaller than the data bin width "
            f"{source_bin_width:.6g}. Use a width >= {source_bin_width:.6g}."
        )


def _normalize_mode(value: str | None) -> str:
    token = "none" if value is None else str(value).strip().lower()
    if token not in {"none", "max", "area", "value_at_x", "factor"}:
        raise ValueError("Normalization mode must be one of: none, max, area, value_at_x, factor.")
    return token


def _normalize_series_values(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    target_value: float | None,
    reference_x: float | None,
    label: str,
) -> tuple[np.ndarray, bool, float]:
    if mode == "none":
        return y, False, 1.0

    if target_value is None:
        raise ValueError(f"Series '{label}' normalization mode '{mode}' requires a target value.")

    source_value: float
    if mode == "max":
        source_value = float(np.max(y))
    elif mode == "area":
        source_value = _trapezoid_integral(y, x)
    elif mode == "value_at_x":
        if reference_x is None:
            raise ValueError(
                f"Series '{label}' normalization mode 'value_at_x' requires a reference x value."
            )
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        y_sorted = y[order]
        source_value = float(
            np.interp(float(reference_x), x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])
        )
    elif mode == "factor":
        source_value = 1.0
    else:
        raise ValueError(f"Unsupported normalization mode '{mode}'.")

    if not np.isfinite(source_value) or abs(source_value) <= 1e-15:
        raise ValueError(
            f"Series '{label}' normalization source value is zero/invalid; cannot normalize."
        )

    if mode == "factor":
        scale = float(target_value)
    else:
        scale = float(target_value) / source_value
    return y * scale, True, scale


def _prepare_plot_series_data(
    *,
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    labels: list[str],
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    series_count = len(labels)
    if len(x_series) != series_count or len(y_series) != series_count:
        raise ValueError("x_series, y_series, and labels must have equal lengths.")

    if series_normalization_modes is not None and len(series_normalization_modes) != series_count:
        raise ValueError(
            "series_normalization_modes count must match the number of plotted series "
            f"({series_count})."
        )
    if series_normalization_values is not None and len(series_normalization_values) != series_count:
        raise ValueError(
            "series_normalization_values count must match the number of plotted series "
            f"({series_count})."
        )
    if series_normalization_x_refs is not None and len(series_normalization_x_refs) != series_count:
        raise ValueError(
            "series_normalization_x_refs count must match the number of plotted series "
            f"({series_count})."
        )

    reducer = _resolve_reducer_name(x_bin_reducer) if x_bin_width is not None else "mean"
    transformed_x: list[np.ndarray] = []
    transformed_y: list[np.ndarray] = []
    normalized_count = 0
    for index, (x_values, y_values, label) in enumerate(zip(x_series, y_series, labels)):
        x_data = np.asarray(x_values, dtype=float)
        y_data = np.asarray(y_values, dtype=float)
        if x_data.shape != y_data.shape:
            raise ValueError(f"Series '{label}' x/y arrays must have the same shape.")

        if x_bin_width is not None:
            x_data, y_data = _rebin_xy_series(
                x_data,
                y_data,
                bin_width=float(x_bin_width),
                reducer=reducer,
            )

        mode = _normalize_mode(
            series_normalization_modes[index] if series_normalization_modes is not None else None
        )
        target_value = (
            series_normalization_values[index] if series_normalization_values is not None else None
        )
        reference_x = (
            series_normalization_x_refs[index] if series_normalization_x_refs is not None else None
        )
        y_data, applied, _scale = _normalize_series_values(
            x_data,
            y_data,
            mode=mode,
            target_value=target_value,
            reference_x=reference_x,
            label=label,
        )
        if applied:
            normalized_count += 1

        transformed_x.append(x_data)
        transformed_y.append(y_data)
    return transformed_x, transformed_y, normalized_count


def _prepare_line_render_series(
    *,
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    labels: list[str],
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    series_statistics_data: list[SeriesStatistics | None] | None = None,
    series_raw_statistics: list[bool] | None = None,
    series_error_configs: list[dict[str, Any] | None] | None = None,
    min_bin_points: int | None = None,
) -> tuple[list[PreparedLineSeries], int]:
    series_count = len(labels)
    if len(x_series) != series_count or len(y_series) != series_count:
        raise ValueError("x_series, y_series, and labels must have equal lengths.")
    if series_statistics_data is not None and len(series_statistics_data) != series_count:
        raise ValueError(
            "series_statistics_data count must match the number of plotted series "
            f"({series_count})."
        )
    if series_raw_statistics is not None and len(series_raw_statistics) != series_count:
        raise ValueError(
            f"series_raw_statistics count must match the number of plotted series ({series_count})."
        )
    if series_normalization_modes is not None and len(series_normalization_modes) != series_count:
        raise ValueError(
            "series_normalization_modes count must match the number of plotted series "
            f"({series_count})."
        )
    if series_normalization_values is not None and len(series_normalization_values) != series_count:
        raise ValueError(
            "series_normalization_values count must match the number of plotted series "
            f"({series_count})."
        )
    if series_normalization_x_refs is not None and len(series_normalization_x_refs) != series_count:
        raise ValueError(
            "series_normalization_x_refs count must match the number of plotted series "
            f"({series_count})."
        )
    if min_bin_points is not None and int(min_bin_points) < 1:
        raise ValueError("min_bin_points must be >= 1 when provided.")

    reducer = _resolve_reducer_name(x_bin_reducer) if x_bin_width is not None else "mean"
    resolved_error_configs = _coerce_error_config_list(series_count, series_error_configs)
    prepared: list[PreparedLineSeries] = []
    normalized_count = 0
    for index, (x_values, y_values, label) in enumerate(zip(x_series, y_series, labels)):
        x_data = np.asarray(x_values, dtype=float)
        y_data = np.asarray(y_values, dtype=float)
        if x_data.shape != y_data.shape:
            raise ValueError(f"Series '{label}' x/y arrays must have the same shape.")

        persisted_stats = None if series_statistics_data is None else series_statistics_data[index]
        raw_statistics_enabled = (
            False if series_raw_statistics is None else bool(series_raw_statistics[index])
        )
        error_config = resolved_error_configs[index]
        requires_raw_grouping = raw_statistics_enabled and (
            error_config.enabled or min_bin_points is not None
        )

        x_plot = np.asarray(x_data, dtype=float)
        y_plot = np.asarray(y_data, dtype=float)
        plot_stats: SeriesStatistics | None = persisted_stats
        error_status = "off"
        error_reason: str | None = None
        statistics_mode = "direct"
        if x_bin_width is not None:
            if raw_statistics_enabled:
                x_plot, y_plot, plot_stats = _group_xy_series_with_statistics(
                    x_plot,
                    y_plot,
                    bin_width=float(x_bin_width),
                    reducer=reducer,
                )
                statistics_mode = "raw_grouped"
            else:
                x_plot, y_plot = _rebin_xy_series(
                    x_plot,
                    y_plot,
                    bin_width=float(x_bin_width),
                    reducer=reducer,
                )
                if persisted_stats is not None:
                    rebinned_stats, rebinned_reason = _rebin_persisted_series_statistics(
                        x_data,
                        y_data,
                        persisted_stats,
                        bin_width=float(x_bin_width),
                        reducer=reducer,
                    )
                    if rebinned_stats is None:
                        error_status = "rebinned_saved_profile"
                        error_reason = (
                            rebinned_reason
                            or "Saved-profile uncertainty cannot be rebinned for this series."
                        )
                        plot_stats = None
                    else:
                        plot_stats = rebinned_stats
                        statistics_mode = "saved_rebinned_sample"
        elif requires_raw_grouping:
            x_plot, y_plot, plot_stats = _group_xy_series_with_statistics(
                x_plot,
                y_plot,
                bin_width=None,
                reducer="mean",
            )
            statistics_mode = "raw_grouped"

        mode = _normalize_mode(
            series_normalization_modes[index] if series_normalization_modes is not None else None
        )
        target_value = (
            series_normalization_values[index] if series_normalization_values is not None else None
        )
        reference_x = (
            series_normalization_x_refs[index] if series_normalization_x_refs is not None else None
        )
        y_plot, applied, scale = _normalize_series_values(
            x_plot,
            y_plot,
            mode=mode,
            target_value=target_value,
            reference_x=reference_x,
            label=label,
        )
        if applied:
            normalized_count += 1
            if plot_stats is not None:
                plot_stats = _scale_series_statistics(plot_stats, scale=scale)

        masked_bin_count = 0
        if min_bin_points is not None and plot_stats is not None:
            mask = np.asarray(plot_stats.point_count, dtype=int) >= int(min_bin_points)
            if mask.shape == y_plot.shape:
                masked_bin_count = int(mask.size - np.count_nonzero(mask))
                x_plot = x_plot[mask]
                y_plot = y_plot[mask]
                plot_stats = _subset_series_statistics(plot_stats, mask)

        available_error_stats = statistics_available_stats(plot_stats)
        error_availability = resolve_series_error_availability(
            supported_for_view=True,
            available_stats=available_error_stats,
            error_status=error_status if error_status != "off" else None,
            error_reason=error_reason,
        )
        if error_config.enabled:
            if plot_stats is None or not error_availability.available_stats:
                if error_status == "off":
                    error_status = "unavailable"
                    error_reason = (
                        error_availability.reason
                        or "No uncertainty statistics are available for this series."
                    )
            else:
                requested_stat = error_config.stat or error_availability.default_stat
                if requested_stat is None:
                    error_status = "unavailable"
                    error_reason = (
                        error_availability.reason
                        or "No uncertainty statistics are available for this series."
                    )
                elif requested_stat not in error_availability.available_stats:
                    fallback_stat = error_availability.default_stat
                    if (
                        fallback_stat is not None
                        and fallback_stat in error_availability.available_stats
                    ):
                        requested_stat = fallback_stat
                        error_status = "ok"
                        if statistics_mode == "saved_rebinned_sample":
                            error_reason = (
                                "Block uncertainty is unavailable after x rebinning; "
                                f"using {requested_stat}."
                            )
                    else:
                        error_status = "unavailable"
                        error_reason = f"Requested error statistic '{requested_stat}' is unavailable for this series."
                else:
                    error_status = "ok"
                    if statistics_mode == "saved_rebinned_sample" and error_config.stat in {
                        "block_sem",
                        "block_std",
                    }:
                        resolved_stat = requested_stat
                        error_reason = (
                            "Block uncertainty is unavailable after x rebinning; "
                            f"using {resolved_stat}."
                        )
        prepared.append(
            PreparedLineSeries(
                x=np.asarray(x_plot, dtype=float),
                y=np.asarray(y_plot, dtype=float),
                statistics=plot_stats,
                available_error_stats=error_availability.available_stats,
                error_config=error_config,
                masked_bin_count=masked_bin_count,
                error_status=error_status,
                statistics_mode=statistics_mode,
                error_reason=error_reason,
            )
        )
    return prepared, normalized_count


def _series_override_entry(
    overrides_by_id: dict[str, dict[str, Any]] | None,
    series_id: str,
) -> dict[str, Any]:
    if not isinstance(overrides_by_id, dict):
        return {}
    raw = overrides_by_id.get(str(series_id))
    return dict(raw) if isinstance(raw, dict) else {}


def _descriptor_source_kind(descriptor: dict[str, Any]) -> str:
    token = str(descriptor.get("source_kind") or "source").strip().lower()
    return "group" if token == "group" else "source"


def _build_cumulative_series(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.shape != y_values.shape:
        raise ValueError("Cumulative series requires x and y arrays with matching shapes.")
    if x_values.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite_mask):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    x_clean = np.asarray(x_values[finite_mask], dtype=float)
    y_clean = np.asarray(y_values[finite_mask], dtype=float)
    order = np.argsort(x_clean, kind="mergesort")
    x_sorted = x_clean[order]
    y_sorted = y_clean[order]
    running = np.cumsum(y_sorted, dtype=float)
    counts = np.arange(1, y_sorted.size + 1, dtype=float)
    return x_sorted, running / counts


def _reduce_group_stack(values: np.ndarray, *, reducer: str) -> float:
    if values.size == 0:
        return float("nan")
    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "median":
        return float(np.median(values))
    if reducer == "sum":
        return float(np.sum(values))
    if reducer == "min":
        return float(np.min(values))
    if reducer == "max":
        return float(np.max(values))
    raise ValueError(f"Unsupported grouped-series reducer '{reducer}'.")


def _aggregate_grouped_prepared_series(
    prepared_members: list[PreparedLineSeries],
    *,
    reducer: str,
    x_bin_width: float | None,
    x_bin_reducer: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if not prepared_members:
        return None, None, "Grouped series has no member series."

    for prepared in prepared_members:
        x_values = np.asarray(prepared.x, dtype=float)
        y_values = np.asarray(prepared.y, dtype=float)
        if x_values.shape != y_values.shape:
            raise ValueError(
                "Internal plotting error: grouped-series x/y point counts do not match."
            )

    first_x = np.asarray(prepared_members[0].x, dtype=float)
    same_grid = True
    for prepared in prepared_members[1:]:
        candidate_x = np.asarray(prepared.x, dtype=float)
        if candidate_x.shape != first_x.shape or not np.allclose(
            candidate_x, first_x, equal_nan=True
        ):
            same_grid = False
            break

    if same_grid:
        stack = np.vstack([np.asarray(prepared.y, dtype=float) for prepared in prepared_members])
        aggregated = np.apply_along_axis(_reduce_group_stack, 0, stack, reducer=reducer)
        return np.asarray(first_x, dtype=float), np.asarray(aggregated, dtype=float), None

    if x_bin_width is None:
        return (
            None,
            None,
            "Grouped series requires matching x grids, or active x rebinning/sectioning.",
        )

    global_x: list[np.ndarray] = []
    global_y: list[np.ndarray] = []
    for prepared in prepared_members:
        finite_mask = np.isfinite(prepared.x) & np.isfinite(prepared.y)
        if np.any(finite_mask):
            global_x.append(np.asarray(prepared.x[finite_mask], dtype=float))
            global_y.append(np.asarray(prepared.y[finite_mask], dtype=float))
    if not global_x:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), None

    start = float(min(float(np.min(values)) for values in global_x))
    per_series_bins: list[dict[int, float]] = []
    per_series_x_bins: list[dict[int, float]] = []
    seen_bins: set[int] = set()
    if len(global_x) != len(global_y):
        raise ValueError(
            "Internal plotting error: grouped-series x/y collection counts do not match."
        )
    for x_values, y_values in zip(global_x, global_y):
        if len(x_values) != len(y_values):
            raise ValueError(
                "Internal plotting error: grouped-series x/y point counts do not match."
            )
        bin_index = np.floor((x_values - start) / float(x_bin_width)).astype(np.int64)
        bin_values: dict[int, list[float]] = {}
        bin_positions: dict[int, list[float]] = {}
        if len(bin_index) != len(x_values):
            raise ValueError(
                "Internal plotting error: grouped-series bin index count does not match x values."
            )
        for current_x, current_y, current_bin in zip(x_values, y_values, bin_index):
            key = int(current_bin)
            bin_values.setdefault(key, []).append(float(current_y))
            bin_positions.setdefault(key, []).append(float(current_x))
            seen_bins.add(key)
        reduced_y = {
            key: _reduce_values(np.asarray(values, dtype=float), reducer=x_bin_reducer)
            for key, values in bin_values.items()
        }
        reduced_x = {
            key: float(np.mean(np.asarray(values, dtype=float)))
            for key, values in bin_positions.items()
        }
        per_series_bins.append(reduced_y)
        per_series_x_bins.append(reduced_x)

    if not seen_bins:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), None

    ordered_bins = sorted(seen_bins)
    x_out: list[float] = []
    y_out: list[float] = []
    for current_bin in ordered_bins:
        y_candidates = [
            values[current_bin]
            for values in per_series_bins
            if current_bin in values and np.isfinite(values[current_bin])
        ]
        if not y_candidates:
            continue
        x_candidates = [
            positions[current_bin]
            for positions in per_series_x_bins
            if current_bin in positions and np.isfinite(positions[current_bin])
        ]
        x_out.append(float(np.mean(np.asarray(x_candidates, dtype=float))))
        y_out.append(_reduce_group_stack(np.asarray(y_candidates, dtype=float), reducer=reducer))
    return np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float), None


def plot_line_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    series_id: str | None = None,
    title: str,
    x_label: str,
    y_label: str,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    line_label: str | None = None,
    line_color: str | None = None,
    line_width_override: float | None = None,
    line_marker: str | None = None,
    line_visible: bool = True,
    show_in_legend: bool = True,
    fit_config: dict[str, Any] | None = None,
    fit_enabled: bool = False,
    fit_label: str | None = None,
    fit_show_in_legend: bool = True,
    cumulative_config: dict[str, Any] | None = None,
    series_statistics: SeriesStatistics | None = None,
    raw_point_statistics: bool = False,
    error_config: dict[str, Any] | None = None,
    normalization_mode: str | None = None,
    normalization_value: float | None = None,
    normalization_x_ref: float | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
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
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_pad: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    analysis_name: str | None = None,
    capture_state: dict[str, Any] | None = None,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
) -> Path | None:
    """Plot a single line using the shared LiNaK style."""
    resolved_fit_config = None if fit_config is None else dict(fit_config)
    if fit_enabled or fit_label not in {None, ""} or fit_show_in_legend is False:
        resolved_fit_config = dict(resolved_fit_config or {})
        if fit_enabled:
            resolved_fit_config["fit_enabled"] = True
        if fit_label not in {None, ""}:
            resolved_fit_config["fit_label_override"] = str(fit_label)
        if fit_show_in_legend is False:
            resolved_fit_config["fit_show_in_legend"] = False

    return plot_multi_line_series(
        [np.asarray(x, dtype=float)],
        [np.asarray(y, dtype=float)],
        [line_label or "Series"],
        series_ids=[str(series_id or "series")],
        title=title,
        x_label=x_label,
        y_label=y_label,
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        style=style,
        line_colors=None if line_color is None else [line_color],
        series_enabled=[bool(line_visible)],
        series_show_in_legend=[bool(show_in_legend)],
        series_line_widths=[line_width_override],
        series_markers=[line_marker],
        series_fit_configs=[resolved_fit_config],
        series_cumulative_configs=[cumulative_config],
        series_statistics_data=[series_statistics],
        series_raw_statistics=[raw_point_statistics],
        series_error_configs=[error_config],
        series_normalization_modes=[normalization_mode],
        series_normalization_values=[normalization_value],
        series_normalization_x_refs=[normalization_x_ref],
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        line_kwargs=line_kwargs,
        annotations=annotations,
        integration_config=integration_config,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
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
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
        title_pad=title_pad,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        analysis_name=analysis_name,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )


def plot_heatmap_series(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_values: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_font_size: int | None = None,
    y_label_font_size: int | None = None,
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_pad: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    capture_state: dict[str, Any] | None = None,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_cmap: str | None = None,
    heatmap_log_scale: bool = False,
    heatmap_colorbar_enabled: bool = True,
    heatmap_colorbar_label: str | None = None,
    heatmap_colorbar_label_size: int | None = None,
    heatmap_colorbar_tick_size: int | None = None,
    heatmap_colorbar_position: str = "right",
    heatmap_colorbar_pad: float | None = None,
    heatmap_colorbar_shrink: float | None = None,
    heatmap_colorbar_aspect: float | None = None,
    annotations: list[dict[str, Any]] | None = None,
    capture_state_extra: dict[str, Any] | None = None,
) -> Path | None:
    """Plot a pcolormesh heatmap using the shared LiNaK plot contract."""
    from matplotlib.colors import LogNorm

    active_backend = configure_matplotlib_backend(
        interactive=show,
        preferred_backend=preferred_backend,
    )
    plt = _import_pyplot()

    rc_context_args: dict[str, Any] = {"font.family": style.font_family, "text.parse_math": True}
    if matplotlib_rc is not None:
        rc_context_args.update(dict(matplotlib_rc))
    with plt.rc_context(rc_context_args):
        fig, ax = plt.subplots(figsize=style.figure_size)
        figure_alpha = _apply_figure_kwargs(fig, figure_kwargs)

        heatmap_array = np.asarray(z_values, dtype=float)
        mesh_kwargs: dict[str, Any] = {
            "shading": "flat",
            "cmap": heatmap_cmap or "turbo",
        }
        resolved_vmin = None if heatmap_vmin is None else float(heatmap_vmin)
        resolved_vmax = None if heatmap_vmax is None else float(heatmap_vmax)
        if heatmap_log_scale:
            if resolved_vmin is not None and resolved_vmin <= 0.0:
                raise ValueError("Heatmap log scale requires a positive heatmap_vmin.")
            if resolved_vmax is not None and resolved_vmax <= 0.0:
                raise ValueError("Heatmap log scale requires a positive heatmap_vmax.")
            positive = heatmap_array[np.isfinite(heatmap_array) & (heatmap_array > 0.0)]
            if positive.size == 0:
                raise ValueError("Heatmap log scale requires at least one positive heatmap value.")
            if resolved_vmin is None:
                resolved_vmin = float(np.min(positive))
            if resolved_vmax is None:
                resolved_vmax = float(np.max(positive))
            if resolved_vmax < resolved_vmin:
                raise ValueError("Heatmap log scale requires heatmap_vmax >= heatmap_vmin.")
            mesh_kwargs["norm"] = LogNorm(vmin=resolved_vmin, vmax=resolved_vmax)
            heatmap_array = np.ma.masked_less_equal(np.ma.masked_invalid(heatmap_array), 0.0)
        else:
            mesh_kwargs["vmin"] = resolved_vmin
            mesh_kwargs["vmax"] = resolved_vmax

        mesh = ax.pcolormesh(
            np.asarray(x_edges, dtype=float),
            np.asarray(y_edges, dtype=float),
            heatmap_array.T,
            **mesh_kwargs,
        )

        colorbar = None
        if heatmap_colorbar_enabled:
            cb_kw: dict[str, Any] = {}
            position = (
                heatmap_colorbar_position
                if heatmap_colorbar_position in {"right", "left", "top", "bottom"}
                else "right"
            )
            cb_kw["location"] = position
            if heatmap_colorbar_pad is not None:
                cb_kw["pad"] = float(heatmap_colorbar_pad)
            if heatmap_colorbar_shrink is not None:
                cb_kw["shrink"] = float(heatmap_colorbar_shrink)
            if heatmap_colorbar_aspect is not None:
                cb_kw["aspect"] = float(heatmap_colorbar_aspect)
            colorbar = fig.colorbar(
                mesh,
                ax=ax,
                label=heatmap_colorbar_label,
                **cb_kw,
            )
            cb_is_vertical = position in {"right", "left"}
            if heatmap_colorbar_label_size is not None:
                colorbar.set_label(
                    colorbar.ax.get_ylabel() if cb_is_vertical else colorbar.ax.get_xlabel(),
                    fontsize=heatmap_colorbar_label_size,
                )
            if heatmap_colorbar_tick_size is not None:
                colorbar.ax.tick_params(labelsize=heatmap_colorbar_tick_size)
            colorbar.ax.tick_params(colors=style.font_color)
            label_axis = colorbar.ax.yaxis if cb_is_vertical else colorbar.ax.xaxis
            label_axis.label.set_color(style.font_color)

        xlabel_kwargs: dict[str, Any] = {
            "fontsize": x_label_font_size or style.label_font_size,
            "color": style.font_color,
        }
        ylabel_kwargs: dict[str, Any] = {
            "fontsize": y_label_font_size or style.label_font_size,
            "color": style.font_color,
        }
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(format_axis_label_units(x_label), **xlabel_kwargs)
        ax.set_ylabel(format_axis_label_units(y_label), **ylabel_kwargs)
        if title_visible is False:
            ax.set_title(
                "",
                fontsize=style.title_font_size,
                color=style.font_color,
                pad=style.title_pad,
            )
        else:
            ax.set_title(
                normalize_plot_text(title),
                fontsize=style.title_font_size,
                color=style.font_color,
                pad=style.title_pad,
            )
        ax.tick_params(axis="both", labelsize=style.tick_font_size, colors=style.font_color)
        resolved_tick_params_kwargs, tick_axis_hint, minor_ticks_mode = _extract_tick_controls(
            tick_params_kwargs
        )

        if style.grid:
            resolved_grid_kwargs: dict[str, Any] = {
                "linestyle": style.grid_linestyle,
                "linewidth": style.grid_linewidth,
                "alpha": style.grid_alpha,
            }
            if grid_kwargs is not None:
                resolved_grid_kwargs.update(dict(grid_kwargs))
            ax.grid(True, **resolved_grid_kwargs)
        else:
            ax.grid(False)
        x_ticks_visible, y_ticks_visible = _resolve_tick_visibility(
            tick_params_kwargs,
            ticks_visible,
            tick_axis_hint,
        )
        if not x_ticks_visible:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        if not y_ticks_visible:
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        if resolved_tick_params_kwargs:
            ax.tick_params(**resolved_tick_params_kwargs)
        x_axis_tick_params = _axis_tick_params(tick_params_kwargs, "x")
        y_axis_tick_params = _axis_tick_params(tick_params_kwargs, "y")
        if x_tick_font_size is not None:
            x_axis_tick_params["labelsize"] = int(x_tick_font_size)
        if y_tick_font_size is not None:
            y_axis_tick_params["labelsize"] = int(y_tick_font_size)
        if x_tick_rotation is not None:
            x_axis_tick_params["rotation"] = float(x_tick_rotation)
        if y_tick_rotation is not None:
            y_axis_tick_params["rotation"] = float(y_tick_rotation)
        if x_axis_tick_params:
            ax.tick_params(axis="x", **x_axis_tick_params)
        if y_axis_tick_params:
            ax.tick_params(axis="y", **y_axis_tick_params)
        _apply_minor_tick_modes(
            ax,
            tick_params_kwargs=tick_params_kwargs,
            fallback_mode=minor_ticks_mode,
        )
        if x_ticks is not None:
            ax.set_xticks([float(value) for value in x_ticks])
        if y_ticks is not None:
            ax.set_yticks([float(value) for value in y_ticks])
        visible_bounds = _visible_axes_data_bounds(ax)
        if visible_bounds is not None:
            auto_left, auto_right, auto_bottom, auto_top = visible_bounds
            if x_lim is None:
                ax.set_xlim(left=auto_left, right=auto_right)
            else:
                left = auto_left if x_lim[0] is None else float(x_lim[0])
                right = auto_right if x_lim[1] is None else float(x_lim[1])
                ax.set_xlim(left=left, right=right)
            if y_lim is None:
                ax.set_ylim(bottom=auto_bottom, top=auto_top)
            else:
                bottom = auto_bottom if y_lim[0] is None else float(y_lim[0])
                top = auto_top if y_lim[1] is None else float(y_lim[1])
                ax.set_ylim(bottom=bottom, top=top)
        else:
            if x_lim is not None:
                left_value: float | None = None if x_lim[0] is None else float(x_lim[0])
                right_value: float | None = None if x_lim[1] is None else float(x_lim[1])
                ax.set_xlim(left=left_value, right=right_value)
            if y_lim is not None:
                bottom_value: float | None = None if y_lim[0] is None else float(y_lim[0])
                top_value: float | None = None if y_lim[1] is None else float(y_lim[1])
                ax.set_ylim(bottom=bottom_value, top=top_value)
        _apply_axes_border(ax, visible=style.axes_border)
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))
        _apply_axes_face_alpha(ax, figure_alpha)

        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()
        annotation_summaries = _render_plot_annotations(ax, annotations)
        _capture_heatmap_state(
            ax=ax,
            style=style,
            capture_state=capture_state,
            mesh=mesh,
            colorbar=colorbar,
            extra_state=capture_state_extra,
            annotation_summaries=annotation_summaries,
        )
        if capture_state is not None:
            capture_state["annotations_summary"] = list(annotation_summaries)

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


def plot_multi_line_series(
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    labels: list[str],
    *,
    series_ids: list[str] | None = None,
    title: str,
    x_label: str,
    y_label: str,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_cumulative_configs: list[dict[str, Any] | None] | None = None,
    series_statistics_data: list[SeriesStatistics | None] | None = None,
    series_raw_statistics: list[bool] | None = None,
    series_error_configs: list[dict[str, Any] | None] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
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
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_pad: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    analysis_name: str | None = None,
    capture_state: dict[str, Any] | None = None,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
) -> Path | None:
    """Plot multiple line series in a single axes using the shared LiNaK style."""
    if not (len(x_series) == len(y_series) == len(labels)):
        raise ValueError("x_series, y_series, and labels must have equal lengths.")
    if not x_series:
        raise ValueError("At least one series is required for multi-line plotting.")

    source_count = len(labels)
    if series_ids is not None and len(series_ids) != source_count:
        raise ValueError("series_ids count must match the number of plotted series.")
    source_ids = (
        [str(value) for value in series_ids]
        if series_ids is not None
        else [f"series:{index}" for index in range(source_count)]
    )
    if series_enabled is not None and len(series_enabled) != source_count:
        raise ValueError(
            f"series_enabled count must match the number of plotted series ({source_count})."
        )
    if series_show_in_legend is not None and len(series_show_in_legend) != source_count:
        raise ValueError(
            f"series_show_in_legend count must match the number of plotted series ({source_count})."
        )
    if series_line_widths is not None and len(series_line_widths) != source_count:
        raise ValueError(
            f"series_line_widths count must match the number of plotted series ({source_count})."
        )
    if series_markers is not None and len(series_markers) != source_count:
        raise ValueError(
            f"series_markers count must match the number of plotted series ({source_count})."
        )
    if series_line_kwargs is not None and len(series_line_kwargs) != source_count:
        raise ValueError(
            f"series_line_kwargs count must match the number of plotted series ({source_count})."
        )
    resolved_x_axis_scale, resolved_x_axis_offset = _coerce_x_axis_linear_transform(
        x_axis_scale,
        x_axis_offset,
    )
    if title_pad is not None:
        style = with_style_overrides(base_style=style, title_pad=title_pad)

    resolved_line_colors = resolve_series_colors(line_colors, series_count=source_count)
    default_source_colors = default_series_colors(source_count)
    resolved_fit_configs_source = resolve_series_fit_configs(
        series_count=source_count,
        series_fit_configs=series_fit_configs,
    )
    resolved_cumulative_configs_source = _coerce_cumulative_config_list(
        source_count,
        series_cumulative_configs,
    )
    resolved_error_configs_source = _coerce_error_config_list(source_count, series_error_configs)
    source_entries: dict[str, dict[str, Any]] = {}
    for index in range(source_count):
        raw_x = np.asarray(x_series[index], dtype=float)
        y_array = np.asarray(y_series[index], dtype=float)
        base_x = _base_x_values(raw_x, y_array)
        source_entries[source_ids[index]] = {
            "series_id": source_ids[index],
            "label": str(labels[index]),
            "x": _display_x_values(
                base_x,
                y_array,
                scale=resolved_x_axis_scale,
                offset=resolved_x_axis_offset,
            ),
            "raw_x": base_x,
            "y": y_array,
            "statistics": (
                None if series_statistics_data is None else series_statistics_data[index]
            ),
            "raw_statistics": (
                False if series_raw_statistics is None else bool(series_raw_statistics[index])
            ),
            "line_visible": True if series_enabled is None else bool(series_enabled[index]),
            "show_in_legend": (
                True if series_show_in_legend is None else bool(series_show_in_legend[index])
            ),
            "line_color": (
                resolved_line_colors[index]
                if resolved_line_colors is not None
                else default_source_colors[index]
            ),
            "line_width": None if series_line_widths is None else series_line_widths[index],
            "marker": None if series_markers is None else series_markers[index],
            "line_kwargs": None if series_line_kwargs is None else series_line_kwargs[index],
            "fit_config": dict(resolved_fit_configs_source[index]),
            "cumulative_config": resolved_cumulative_configs_source[index],
            "error_config": resolved_error_configs_source[index],
            "normalization_mode": (
                None if series_normalization_modes is None else series_normalization_modes[index]
            ),
            "normalization_value": (
                None if series_normalization_values is None else series_normalization_values[index]
            ),
            "normalization_x_ref": (
                None if series_normalization_x_refs is None else series_normalization_x_refs[index]
            ),
        }

    resolved_render_descriptors = (
        [dict(value) for value in render_series_descriptors]
        if isinstance(render_series_descriptors, list) and render_series_descriptors
        else [
            {
                "series_id": series_id,
                "default_label": source_entries[series_id]["label"],
                "source_kind": "source",
                "source_series_id": series_id,
            }
            for series_id in source_ids
        ]
    )
    overrides = (
        {
            str(key): dict(value)
            for key, value in series_overrides_by_id.items()
            if isinstance(value, dict)
        }
        if isinstance(series_overrides_by_id, dict)
        else {}
    )
    reducer_name = _resolve_reducer_name(x_bin_reducer) if x_bin_width is not None else "mean"

    def _build_source_render_item(
        *,
        current_id: str,
        descriptor: dict[str, Any],
    ) -> dict[str, Any] | None:
        current_override = _series_override_entry(overrides, current_id)
        source_series_id = str(
            descriptor.get("source_series_id") or descriptor.get("source_id") or current_id
        ).strip()
        source_entry = source_entries.get(source_series_id)
        if source_entry is None:
            return None
        raw_override_line_kwargs = current_override.get("line_kwargs")
        source_line_kwargs_value = source_entry["line_kwargs"]
        line_kwargs_value = (
            dict(raw_override_line_kwargs)
            if isinstance(raw_override_line_kwargs, dict)
            else (
                dict(source_line_kwargs_value)
                if isinstance(source_line_kwargs_value, dict)
                else None
            )
        )
        if current_override.get("alpha") is not None:
            if line_kwargs_value is None:
                line_kwargs_value = {}
            line_kwargs_value["alpha"] = float(current_override["alpha"])
        fit_override = current_override.get("fit")
        fit_config = resolve_series_fit_configs(
            series_count=1,
            series_fit_configs=[
                fit_override if isinstance(fit_override, dict) else source_entry["fit_config"]
            ],
        )[0]
        cumulative_value = (
            current_override.get("cumulative")
            if isinstance(current_override.get("cumulative"), dict)
            else {
                "enabled": source_entry["cumulative_config"].enabled,
                "label_override": source_entry["cumulative_config"].label_override,
                "show_in_legend": source_entry["cumulative_config"].show_in_legend,
            }
        )
        return {
            "series_id": current_id,
            "label": (
                str(current_override.get("label_override") or "").strip()
                or str(descriptor.get("default_label") or source_entry["label"])
            ),
            "kind": "source",
            "source_series_id": source_series_id,
            "x": np.asarray(source_entry["x"], dtype=float),
            "raw_x": np.asarray(source_entry["raw_x"], dtype=float),
            "y": np.asarray(source_entry["y"], dtype=float),
            "statistics": source_entry["statistics"],
            "raw_statistics": bool(source_entry["raw_statistics"]),
            "series_enabled": bool(current_override.get("enabled", True)),
            "line_visible": bool(current_override.get("enabled", True))
            and bool(source_entry["line_visible"])
            and bool(current_override.get("show_raw_line", True)),
            "show_in_legend": bool(
                current_override.get("show_in_legend", source_entry["show_in_legend"])
            ),
            "line_color": (
                None
                if current_override.get("color") in {None, ""}
                else str(current_override.get("color"))
            )
            or source_entry["line_color"],
            "line_width": (
                current_override.get("line_width")
                if current_override.get("line_width") not in {None, ""}
                else source_entry["line_width"]
            ),
            "marker": (
                None
                if current_override.get("marker") in {None, ""}
                else str(current_override.get("marker"))
            )
            if current_override.get("marker") not in {None, ""}
            else source_entry["marker"],
            "line_kwargs": line_kwargs_value,
            "fit_config": fit_config,
            "cumulative_config": _coerce_cumulative_config(cumulative_value),
            "integration_config": _coerce_integration_config(
                current_override.get("integration")
                if isinstance(current_override.get("integration"), dict)
                else None
            ),
            "error_config": _coerce_error_config(
                current_override.get("error")
                if isinstance(current_override.get("error"), dict)
                else {
                    "enabled": source_entry["error_config"].enabled,
                    "stat": source_entry["error_config"].stat,
                    "style": source_entry["error_config"].style,
                    "color": source_entry["error_config"].color,
                    "label_override": source_entry["error_config"].label_override,
                    "show_in_legend": source_entry["error_config"].show_in_legend,
                }
            ),
            "normalization_mode": (
                str(current_override.get("normalization_mode") or "").strip()
                or source_entry["normalization_mode"]
            ),
            "normalization_value": (
                current_override.get("normalization_value")
                if current_override.get("normalization_value") is not None
                else source_entry["normalization_value"]
            ),
            "normalization_x_ref": (
                current_override.get("normalization_x_ref")
                if current_override.get("normalization_x_ref") is not None
                else source_entry["normalization_x_ref"]
            ),
        }

    render_items: list[dict[str, Any]] = []
    source_render_items: list[dict[str, Any]] = []
    for descriptor in resolved_render_descriptors:
        current_id = str(descriptor.get("series_id") or "").strip()
        if not current_id:
            continue
        current_override = _series_override_entry(overrides, current_id)
        kind = _descriptor_source_kind(descriptor)
        if kind == "group":
            raw_override_line_kwargs = current_override.get("line_kwargs")
            line_kwargs_value = (
                dict(raw_override_line_kwargs)
                if isinstance(raw_override_line_kwargs, dict)
                else None
            )
            if current_override.get("alpha") is not None:
                if line_kwargs_value is None:
                    line_kwargs_value = {}
                line_kwargs_value["alpha"] = float(current_override["alpha"])
            fit_override = current_override.get("fit")
            fit_config = resolve_series_fit_configs(
                series_count=1,
                series_fit_configs=[fit_override if isinstance(fit_override, dict) else None],
            )[0]
            render_items.append(
                {
                    "series_id": current_id,
                    "label": (
                        str(current_override.get("label_override") or "").strip()
                        or str(descriptor.get("default_label") or current_id)
                    ),
                    "kind": "group",
                    "member_series_ids": [
                        str(value).strip()
                        for value in descriptor.get("member_series_ids", [])
                        if str(value).strip()
                    ],
                    "group_reducer": _resolve_reducer_name(
                        str(descriptor.get("group_reducer") or "mean").strip().lower() or "mean"
                    ),
                    "series_enabled": bool(current_override.get("enabled", True)),
                    "line_visible": bool(current_override.get("enabled", True))
                    and bool(current_override.get("show_raw_line", True)),
                    "show_in_legend": bool(current_override.get("show_in_legend", True)),
                    "line_color": (
                        None
                        if current_override.get("color") in {None, ""}
                        else str(current_override.get("color"))
                    ),
                    "line_width": current_override.get("line_width"),
                    "marker": (
                        None
                        if current_override.get("marker") in {None, ""}
                        else str(current_override.get("marker"))
                    ),
                    "line_kwargs": line_kwargs_value,
                    "fit_config": fit_config,
                    "cumulative_config": _coerce_cumulative_config(
                        current_override.get("cumulative")
                    ),
                    "normalization_mode": (
                        str(current_override.get("normalization_mode") or "").strip() or None
                    ),
                    "normalization_value": current_override.get("normalization_value"),
                    "normalization_x_ref": current_override.get("normalization_x_ref"),
                }
            )
            continue

        item = _build_source_render_item(current_id=current_id, descriptor=descriptor)
        if item is None:
            continue
        render_items.append(item)
        source_render_items.append(item)

    prepared_source_ids = {
        str(item["source_series_id"]): item
        for item in source_render_items
        if str(item.get("source_series_id") or "").strip()
    }
    for source_id, source_entry in source_entries.items():
        if source_id in prepared_source_ids:
            continue
        fallback_descriptor = {
            "series_id": source_id,
            "default_label": source_entry["label"],
            "source_kind": "source",
            "source_series_id": source_id,
        }
        item = _build_source_render_item(current_id=source_id, descriptor=fallback_descriptor)
        if item is not None:
            source_render_items.append(item)

    prepared_series, normalized_count = _prepare_line_render_series(
        x_series=[item["x"] for item in source_render_items],
        y_series=[item["y"] for item in source_render_items],
        labels=[item["label"] for item in source_render_items],
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        series_statistics_data=[item["statistics"] for item in source_render_items],
        series_raw_statistics=[item["raw_statistics"] for item in source_render_items],
        series_error_configs=[
            {
                "enabled": item["error_config"].enabled,
                "stat": item["error_config"].stat,
                "style": item["error_config"].style,
                "color": item["error_config"].color,
                "label_override": item["error_config"].label_override,
                "show_in_legend": item["error_config"].show_in_legend,
            }
            for item in source_render_items
        ],
        min_bin_points=min_bin_points,
        series_normalization_modes=[item["normalization_mode"] for item in source_render_items],
        series_normalization_values=[item["normalization_value"] for item in source_render_items],
        series_normalization_x_refs=[item["normalization_x_ref"] for item in source_render_items],
    )
    if len(source_render_items) != len(prepared_series):
        raise ValueError(
            "Internal plotting error: prepared series count does not match render items."
        )
    for item, prepared in zip(source_render_items, prepared_series):
        item["prepared"] = prepared
    if len(source_render_items) > 1 and 0 < normalized_count < len(source_render_items):
        LOGGER.warning(
            "Only %d/%d plotted series are normalized. Interpret y-axis comparisons with care.",
            normalized_count,
            len(source_render_items),
        )

    prepared_by_id = {
        str(item["series_id"]): item["prepared"]
        for item in source_render_items
        if isinstance(item.get("prepared"), PreparedLineSeries)
        and bool(item.get("series_enabled", True))
    }
    for item in render_items:
        if item["kind"] != "group":
            continue
        member_prepared = [
            prepared_by_id[member_id]
            for member_id in item["member_series_ids"]
            if member_id in prepared_by_id
        ]
        x_group, y_group, group_reason = _aggregate_grouped_prepared_series(
            member_prepared,
            reducer=item["group_reducer"],
            x_bin_width=x_bin_width,
            x_bin_reducer=reducer_name,
        )
        item["group_reason"] = group_reason
        if x_group is None or y_group is None:
            item["prepared"] = PreparedLineSeries(
                x=np.empty(0, dtype=float),
                y=np.empty(0, dtype=float),
                statistics=None,
                available_error_stats=[],
                error_config=SeriesErrorConfig(),
                masked_bin_count=0,
                error_status="unavailable",
                statistics_mode="direct",
                error_reason=group_reason,
            )
            continue
        group_prepared, _ = _prepare_line_render_series(
            x_series=[x_group],
            y_series=[y_group],
            labels=[item["label"]],
            series_statistics_data=[None],
            series_raw_statistics=[False],
            series_error_configs=[{"enabled": False}],
            series_normalization_modes=[item["normalization_mode"]],
            series_normalization_values=[item["normalization_value"]],
            series_normalization_x_refs=[item["normalization_x_ref"]],
        )
        item["prepared"] = group_prepared[0]

    active_backend = configure_matplotlib_backend(
        interactive=show,
        preferred_backend=preferred_backend,
    )
    plt = _import_pyplot()

    rc_context_args: dict[str, Any] = {"font.family": style.font_family, "text.parse_math": True}
    if matplotlib_rc is not None:
        rc_context_args.update(dict(matplotlib_rc))
    with plt.rc_context(rc_context_args):
        fig, ax = plt.subplots(figsize=style.figure_size)
        figure_alpha = _apply_figure_kwargs(fig, figure_kwargs)
        rendered_colors: list[str] = []
        rendered_markers: list[str] = []
        rendered_labels: list[str] = []
        fit_summaries: dict[str, dict[str, Any]] = {}
        cumulative_summaries: dict[str, dict[str, Any]] = {}
        series_stats: dict[str, dict[str, float | int | None]] = {}
        error_summaries: dict[str, dict[str, Any]] = {}
        available_error_stats_map: dict[str, list[str]] = {}
        point_counts_map: dict[str, list[int]] = {}
        source_bin_widths: dict[str, float | None] = {}
        masked_bin_counts: dict[str, int] = {}
        grouped_summaries: dict[str, dict[str, Any]] = {}
        visible_x_series: list[np.ndarray] = []
        visible_y_series: list[np.ndarray] = []
        density_visible_x_series: list[np.ndarray] = []
        density_visible_y_series: list[np.ndarray] = []
        has_visible_overlay_bounds = False
        integration_summaries: list[dict[str, Any]] = []
        integration_seen_ids: set[str] = set()
        for item in render_items:
            prepared_item = item.get("prepared")
            if not isinstance(prepared_item, PreparedLineSeries):
                continue
            label = str(item["label"])
            x_values = prepared_item.x
            y_values = prepared_item.y
            fit_key = str(item["series_id"])
            available_error_stats_map[fit_key] = list(prepared_item.available_error_stats)
            x_source = item.get("x")
            if x_source is None:
                x_source = prepared_item.x
            source_bin_widths[fit_key] = _resolve_meaningful_source_bin_width(
                np.asarray(x_source, dtype=float)
            )
            if prepared_item.statistics is not None:
                point_counts_map[fit_key] = np.asarray(
                    prepared_item.statistics.point_count, dtype=int
                ).tolist()
            masked_bin_counts[fit_key] = int(prepared_item.masked_bin_count)
            is_group = str(item.get("kind") or "source") == "group"
            layer_enabled = bool(item.get("series_enabled", True))
            line_visible = bool(item.get("line_visible", True))
            if is_group:
                grouped_summaries[fit_key] = {
                    "status": "ok" if item.get("group_reason") is None else "unavailable",
                    "reason": item.get("group_reason"),
                    "reducer": str(item.get("group_reducer") or "mean"),
                    "member_count": len(item.get("member_series_ids", [])),
                }
            kwargs: dict[str, Any] = {
                "lw": style.line_width,
                "label": label if bool(item.get("show_in_legend", True)) else "_nolegend_",
            }
            raw_line_width = item.get("line_width")
            if raw_line_width not in {None, ""}:
                kwargs["lw"] = float(str(raw_line_width))
            marker_value = "o" if markers else ""
            if item.get("marker") is not None:
                marker_value = str(item["marker"])
            kwargs["marker"] = marker_value
            color_token = str(item.get("line_color") or "").strip()
            if color_token:
                kwargs["color"] = color_token
            if line_kwargs is not None:
                resolved_line_kwargs = dict(line_kwargs)
                resolved_line_kwargs.pop("label", None)
                kwargs.update(resolved_line_kwargs)
            if item.get("line_kwargs") is not None:
                resolved_item_line_kwargs = dict(item["line_kwargs"])
                resolved_item_line_kwargs.pop("label", None)
                kwargs.update(resolved_item_line_kwargs)
            artist = None
            if line_visible:
                (artist,) = ax.plot(x_values, y_values, **kwargs)
                rendered_colors.append(str(artist.get_color()))
                rendered_markers.append(str(artist.get_marker()))
                rendered_labels.append(str(artist.get_label()))
                visible_x_series.append(np.asarray(x_values, dtype=float))
                visible_y_series.append(np.asarray(y_values, dtype=float))
                if str(analysis_name or "").strip().lower() == "density":
                    density_visible_x_series.append(np.asarray(x_values, dtype=float))
                    density_visible_y_series.append(np.asarray(y_values, dtype=float))
            item_integration_config: IntegrationConfig = item.get(
                "integration_config", IntegrationConfig()
            )
            if item_integration_config.enabled:
                integration_seen_ids.add(fit_key)
                summary: dict[str, Any] = {
                    "series_id": fit_key,
                    "label": label,
                    "source": item_integration_config.source,
                    "target": "self",
                    "enabled": True,
                }
                if not layer_enabled:
                    summary.update(
                        {
                            "status": "unavailable",
                            "reason": "Target series is disabled.",
                        }
                    )
                    integration_summaries.append(summary)
                elif item_integration_config.source == "raw" and is_group:
                    summary.update(
                        {
                            "status": "unavailable",
                            "reason": (
                                "Raw-profile integration is unavailable for grouped series; "
                                "use plotted data instead."
                            ),
                        }
                    )
                    integration_summaries.append(summary)
                else:
                    integration_x = (
                        np.asarray(item.get("raw_x"), dtype=float)
                        if item_integration_config.source == "raw"
                        else x_values
                    )
                    integration_y = (
                        np.asarray(item.get("y"), dtype=float)
                        if item_integration_config.source == "raw"
                        else y_values
                    )
                    region_x, region_y, region_summary = _integration_region(
                        integration_x,
                        integration_y,
                        x_min=item_integration_config.x_min,
                        x_max=item_integration_config.x_max,
                        baseline=item_integration_config.baseline,
                    )
                    summary.update(region_summary)
                    if region_summary.get("status") == "ok":
                        fill_color = item_integration_config.color or str(
                            artist.get_color()
                            if artist is not None
                            else kwargs.get("color", style.line_color)
                        )
                        summary["color"] = fill_color
                        summary["alpha"] = float(item_integration_config.alpha)
                        if artist is not None:
                            zorder = float(artist.get_zorder()) - 0.5
                        else:
                            try:
                                zorder = float(kwargs.get("zorder", 2)) - 0.5
                            except (TypeError, ValueError):
                                zorder = 1.5
                        ax.fill_between(
                            region_x,
                            np.full_like(region_x, item_integration_config.baseline),
                            region_y,
                            color=fill_color,
                            alpha=item_integration_config.alpha,
                            label="_nolegend_",
                            zorder=zorder,
                        )
                        has_visible_overlay_bounds = True
                    integration_summaries.append(summary)
            series_stats[fit_key] = _series_statistics(x_values, y_values)
            if (
                not is_group
                and layer_enabled
                and prepared_item.error_config.enabled
                and prepared_item.error_status == "ok"
                and prepared_item.statistics is not None
            ):
                requested_stat = _resolve_effective_error_stat(
                    prepared_item.error_config.stat,
                    prepared_item.available_error_stats,
                )
                if requested_stat is not None:
                    error_values = _statistics_error_values(
                        prepared_item.statistics,
                        stat_name=requested_stat,
                    )
                    if error_values is not None:
                        finite_mask = (
                            np.isfinite(x_values)
                            & np.isfinite(y_values)
                            & np.isfinite(error_values)
                        )
                        if np.any(finite_mask):
                            error_label = (
                                prepared_item.error_config.label_override
                                or f"{label} \u00b1{_friendly_stat_label(requested_stat)}"
                            )
                            error_color = (
                                str(prepared_item.error_config.color).strip()
                                if prepared_item.error_config.color is not None
                                and str(prepared_item.error_config.color).strip()
                                else str(
                                    artist.get_color()
                                    if artist is not None
                                    else kwargs.get("color", style.line_color)
                                )
                            )
                            if prepared_item.error_config.style == "band":
                                ax.fill_between(
                                    x_values[finite_mask],
                                    y_values[finite_mask] - error_values[finite_mask],
                                    y_values[finite_mask] + error_values[finite_mask],
                                    color=error_color,
                                    alpha=0.20,
                                    label=error_label
                                    if prepared_item.error_config.show_in_legend
                                    else "_nolegend_",
                                )
                            else:
                                ax.errorbar(
                                    x_values[finite_mask],
                                    y_values[finite_mask],
                                    yerr=error_values[finite_mask],
                                    fmt="none",
                                    ecolor=error_color,
                                    elinewidth=float(
                                        artist.get_linewidth()
                                        if artist is not None
                                        else kwargs["lw"]
                                    ),
                                    capsize=2.0,
                                    label=error_label
                                    if prepared_item.error_config.show_in_legend
                                    else "_nolegend_",
                                )
                            has_visible_overlay_bounds = True
                            error_summaries[fit_key] = {
                                "enabled": True,
                                "status": "ok",
                                "stat": requested_stat,
                                "style": prepared_item.error_config.style,
                                "color": error_color,
                                "statistics_mode": prepared_item.statistics_mode,
                                "provenance_family": _error_provenance_family(
                                    requested_stat=requested_stat,
                                    statistics_mode=prepared_item.statistics_mode,
                                ),
                                "provenance": _describe_error_provenance(
                                    analysis_name=analysis_name,
                                    requested_stat=requested_stat,
                                    statistics_mode=prepared_item.statistics_mode,
                                ),
                                "reason": prepared_item.error_reason,
                                "point_count": int(np.count_nonzero(finite_mask)),
                            }
                        else:
                            error_summaries[fit_key] = {
                                "enabled": True,
                                "status": "empty",
                                "reason": "No finite error values remain after masking.",
                            }
                    else:
                        error_summaries[fit_key] = {
                            "enabled": True,
                            "status": "unavailable",
                            "reason": f"Requested error statistic '{requested_stat}' is unavailable.",
                        }
                else:
                    error_summaries[fit_key] = {
                        "enabled": True,
                        "status": "unavailable",
                        "reason": "No uncertainty statistics are available for this series.",
                    }
            else:
                error_summaries[fit_key] = {
                    "enabled": False if is_group else bool(prepared_item.error_config.enabled),
                    "status": (
                        "disabled" if is_group or not layer_enabled else prepared_item.error_status
                    ),
                    "statistics_mode": prepared_item.statistics_mode,
                    "provenance_family": _error_provenance_family(
                        requested_stat=prepared_item.error_config.stat,
                        statistics_mode=prepared_item.statistics_mode,
                    ),
                    "provenance": _describe_error_provenance(
                        analysis_name=analysis_name,
                        requested_stat=prepared_item.error_config.stat,
                        statistics_mode=prepared_item.statistics_mode,
                    ),
                    "reason": (
                        "Grouped series do not render error overlays."
                        if is_group
                        else "Series is disabled."
                        if not layer_enabled and prepared_item.error_config.enabled
                        else prepared_item.error_reason
                    ),
                }
            fit_config = dict(item.get("fit_config") or {})
            if not layer_enabled:
                fit_summaries[fit_key] = {
                    "fit_enabled": bool(fit_config.get("fit_enabled")),
                    "status": "disabled",
                    "fit_type": str(fit_config.get("fit_type") or "linear"),
                    "point_count": 0,
                }
            elif not bool(fit_config.get("fit_enabled")):
                fit_summaries[fit_key] = {
                    "fit_enabled": False,
                    "status": "off",
                    "fit_type": str(fit_config.get("fit_type") or "linear"),
                    "point_count": 0,
                }
            else:
                fit_summary = execute_series_fit(
                    x_values,
                    y_values,
                    fit_config=fit_config,
                    visible_x_lim=x_lim,
                )
                fit_render_label = (
                    str(fit_config.get("fit_label_override") or "").strip() or f"{label} fit"
                )
                fit_summary["fit_enabled"] = True
                fit_summary["label"] = fit_render_label
                fit_summaries[fit_key] = fit_summary
                if fit_summary.get("status") == "ok":
                    _fit_base_color = str(
                        artist.get_color()
                        if artist is not None
                        else kwargs.get("color", style.line_color)
                    )
                    _fit_base_lw = float(
                        artist.get_linewidth() if artist is not None else kwargs["lw"]
                    )
                    _fit_color_override = str(fit_config.get("fit_color") or "").strip() or None
                    _fit_alpha_override = fit_config.get("fit_alpha")
                    _fit_lw_override = fit_config.get("fit_line_width")
                    _fit_ls_override = str(fit_config.get("fit_line_style") or "").strip() or None
                    fit_kwargs: dict[str, Any] = {
                        "color": _fit_color_override if _fit_color_override else _fit_base_color,
                        "linestyle": _fit_ls_override if _fit_ls_override else "--",
                        "linewidth": float(_fit_lw_override)
                        if _fit_lw_override is not None
                        else _fit_base_lw,
                        "marker": "",
                        "label": (
                            fit_render_label
                            if bool(fit_config.get("fit_show_in_legend", True))
                            else "_nolegend_"
                        ),
                    }
                    if _fit_alpha_override is not None:
                        try:
                            fit_kwargs["alpha"] = float(_fit_alpha_override)
                        except (ValueError, TypeError):
                            pass
                    elif artist is not None and artist.get_alpha() is not None:
                        fit_kwargs["alpha"] = float(artist.get_alpha())
                    ax.plot(
                        np.asarray(fit_summary.get("x_fit", []), dtype=float),
                        np.asarray(fit_summary.get("y_fit", []), dtype=float),
                        **fit_kwargs,
                    )
                    visible_x_series.append(np.asarray(fit_summary.get("x_fit", []), dtype=float))
                    visible_y_series.append(np.asarray(fit_summary.get("y_fit", []), dtype=float))
                    has_visible_overlay_bounds = True

            cumulative_config = item.get("cumulative_config")
            if (
                layer_enabled
                and isinstance(cumulative_config, SeriesCumulativeConfig)
                and cumulative_config.enabled
            ):
                cumulative_x, cumulative_y = _build_cumulative_series(x_values, y_values)
                cumulative_label = cumulative_config.label_override or f"{label} cumulative average"
                cumulative_summaries[fit_key] = {
                    "enabled": True,
                    "status": "ok" if cumulative_x.size else "empty",
                    "label": cumulative_label,
                    "point_count": int(cumulative_x.size),
                }
                if cumulative_x.size:
                    _base_color = str(
                        artist.get_color()
                        if artist is not None
                        else kwargs.get("color", style.line_color)
                    )
                    _base_lw = float(artist.get_linewidth() if artist is not None else kwargs["lw"])
                    _base_ls = ":"
                    cumulative_kwargs: dict[str, Any] = {
                        "color": cumulative_config.color
                        if cumulative_config.color
                        else _base_color,
                        "linestyle": cumulative_config.line_style
                        if cumulative_config.line_style
                        else _base_ls,
                        "linewidth": cumulative_config.line_width
                        if cumulative_config.line_width is not None
                        else _base_lw,
                        "marker": "",
                        "label": (
                            cumulative_label if cumulative_config.show_in_legend else "_nolegend_"
                        ),
                    }
                    if cumulative_config.alpha is not None:
                        cumulative_kwargs["alpha"] = float(cumulative_config.alpha)
                    elif artist is not None and artist.get_alpha() is not None:
                        cumulative_kwargs["alpha"] = float(artist.get_alpha())
                    ax.plot(cumulative_x, cumulative_y, **cumulative_kwargs)
                    visible_x_series.append(np.asarray(cumulative_x, dtype=float))
                    visible_y_series.append(np.asarray(cumulative_y, dtype=float))
                    has_visible_overlay_bounds = True
            else:
                cumulative_summaries[fit_key] = {
                    "enabled": (
                        bool(cumulative_config.enabled)
                        if isinstance(cumulative_config, SeriesCumulativeConfig)
                        else False
                    ),
                    "status": "disabled" if not layer_enabled else "off",
                    "point_count": 0,
                }

        handles, legend_labels = ax.get_legend_handles_labels()
        should_show_legend = bool(handles) if legend is None else bool(legend and handles)
        if should_show_legend:
            resolved_legend_kwargs: dict[str, Any] = {
                "fontsize": style.legend_font_size,
                "title": legend_title,
                "loc": legend_loc,
            }
            if legend_kwargs is not None:
                resolved_legend_kwargs.update(dict(legend_kwargs))
            if "ncols" in resolved_legend_kwargs and "ncol" not in resolved_legend_kwargs:
                resolved_legend_kwargs["ncol"] = resolved_legend_kwargs["ncols"]
            legend_obj = ax.legend(**resolved_legend_kwargs)
            for text in legend_obj.get_texts():
                text.set_color(style.font_color)
            legend_obj.get_title().set_color(style.font_color)
        auto_x_lim, auto_y_lim = _axes_artist_auto_limits(ax)
        requested_x_window = (
            x_lim
            if x_lim is not None and any(bound is not None for bound in x_lim[:2])
            else None
        )
        requested_y_window = (
            y_lim
            if y_lim is not None and any(bound is not None for bound in y_lim[:2])
            else None
        )
        constrain_auto_x = requested_y_window is not None and (
            x_lim is None or any(bound is None for bound in x_lim[:2])
        )
        constrain_auto_y = requested_x_window is not None and (
            y_lim is None or any(bound is None for bound in y_lim[:2])
        )
        if str(analysis_name or "").strip().lower() == "density":
            density_auto_x_lim, density_auto_y_lim = _density_visible_auto_limits(
                density_visible_x_series,
                density_visible_y_series,
                x_scale=x_scale,
                y_scale=y_scale,
                x_window=requested_x_window if constrain_auto_y else None,
                y_window=requested_y_window if constrain_auto_x else None,
            )
            overlay_auto_x_lim, overlay_auto_y_lim = _visible_series_auto_limits(
                visible_x_series,
                visible_y_series,
                x_scale=x_scale,
                y_scale=y_scale,
                x_window=requested_x_window if constrain_auto_y else None,
                y_window=requested_y_window if constrain_auto_x else None,
                clamp_y_nonnegative_to_zero=True,
            )
            auto_x_lim = density_auto_x_lim if density_auto_x_lim is not None else auto_x_lim
            if constrain_auto_x and overlay_auto_x_lim is not None:
                auto_x_lim = overlay_auto_x_lim
            auto_y_lim = (
                _union_axis_limits(density_auto_y_lim, overlay_auto_y_lim or auto_y_lim)
                if has_visible_overlay_bounds
                else density_auto_y_lim
                if density_auto_y_lim is not None
                else overlay_auto_y_lim
                if overlay_auto_y_lim is not None
                else auto_y_lim
            )
        elif constrain_auto_x or constrain_auto_y:
            constrained_auto_x_lim, constrained_auto_y_lim = _visible_series_auto_limits(
                visible_x_series,
                visible_y_series,
                x_scale=x_scale,
                y_scale=y_scale,
                x_window=requested_x_window if constrain_auto_y else None,
                y_window=requested_y_window if constrain_auto_x else None,
                clamp_y_nonnegative_to_zero=False,
            )
            if constrain_auto_x and constrained_auto_x_lim is not None:
                auto_x_lim = constrained_auto_x_lim
            if constrain_auto_y and constrained_auto_y_lim is not None:
                auto_y_lim = constrained_auto_y_lim
        x_lim = _merge_axis_limits(x_lim, auto_x_lim)
        y_lim = _merge_axis_limits(y_lim, auto_y_lim)
        xlabel_kwargs: dict[str, Any] = {
            "fontsize": x_label_font_size or style.label_font_size,
            "color": style.font_color,
        }
        ylabel_kwargs: dict[str, Any] = {
            "fontsize": y_label_font_size or style.label_font_size,
            "color": style.font_color,
        }
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(format_axis_label_units(x_label), **xlabel_kwargs)
        ax.set_ylabel(format_axis_label_units(y_label), **ylabel_kwargs)
        if title_visible is False:
            ax.set_title(
                "",
                fontsize=style.title_font_size,
                color=style.font_color,
                pad=style.title_pad,
            )
        else:
            ax.set_title(
                normalize_plot_text(title),
                fontsize=style.title_font_size,
                color=style.font_color,
                pad=style.title_pad,
            )
        ax.tick_params(axis="both", labelsize=style.tick_font_size, colors=style.font_color)
        resolved_tick_params_kwargs, tick_axis_hint, minor_ticks_mode = _extract_tick_controls(
            tick_params_kwargs
        )

        if style.grid:
            resolved_grid_kwargs: dict[str, Any] = {
                "linestyle": style.grid_linestyle,
                "linewidth": style.grid_linewidth,
                "alpha": style.grid_alpha,
            }
            if grid_kwargs is not None:
                resolved_grid_kwargs.update(dict(grid_kwargs))
            ax.grid(True, **resolved_grid_kwargs)
        else:
            ax.grid(False)
        x_ticks_visible, y_ticks_visible = _resolve_tick_visibility(
            tick_params_kwargs,
            ticks_visible,
            tick_axis_hint,
        )
        if not x_ticks_visible:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
        if not y_ticks_visible:
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
            )
        if resolved_tick_params_kwargs:
            ax.tick_params(**resolved_tick_params_kwargs)
        x_axis_tick_params = _axis_tick_params(tick_params_kwargs, "x")
        y_axis_tick_params = _axis_tick_params(tick_params_kwargs, "y")
        if x_tick_font_size is not None:
            x_axis_tick_params["labelsize"] = int(x_tick_font_size)
        if y_tick_font_size is not None:
            y_axis_tick_params["labelsize"] = int(y_tick_font_size)
        if x_tick_rotation is not None:
            x_axis_tick_params["rotation"] = float(x_tick_rotation)
        if y_tick_rotation is not None:
            y_axis_tick_params["rotation"] = float(y_tick_rotation)
        if x_axis_tick_params:
            ax.tick_params(axis="x", **x_axis_tick_params)
        if y_axis_tick_params:
            ax.tick_params(axis="y", **y_axis_tick_params)
        _apply_minor_tick_modes(
            ax,
            tick_params_kwargs=tick_params_kwargs,
            fallback_mode=minor_ticks_mode,
        )
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        if x_ticks is not None:
            ax.set_xticks([float(value) for value in x_ticks])
        if y_ticks is not None:
            ax.set_yticks([float(value) for value in y_ticks])
        # Apply explicit limits after ticks so tick placement cannot widen bounds.
        if x_lim is not None:
            left = None if x_lim[0] is None else float(x_lim[0])
            right = None if x_lim[1] is None else float(x_lim[1])
            ax.set_xlim(left=left, right=right)
        if y_lim is not None:
            bottom = None if y_lim[0] is None else float(y_lim[0])
            top = None if y_lim[1] is None else float(y_lim[1])
            ax.set_ylim(bottom=bottom, top=top)
        _apply_axes_border(ax, visible=style.axes_border)
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))
        _apply_axes_face_alpha(ax, figure_alpha)

        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()
        annotation_summaries = _render_plot_annotations(ax, annotations)
        _capture_plot_state(
            ax=ax,
            style=style,
            line_colors=rendered_colors,
            line_labels=rendered_labels or legend_labels,
            line_markers=rendered_markers,
            legend_loc=legend_loc,
            grid_kwargs=grid_kwargs,
            capture_state=capture_state,
            annotation_summaries=annotation_summaries,
        )
        if capture_state is not None:
            capture_state["series_fit_summaries"] = fit_summaries
            capture_state["series_cumulative_summaries"] = cumulative_summaries
            capture_state["series_statistics"] = series_stats
            capture_state["series_error_summaries"] = error_summaries
            capture_state["series_available_error_stats"] = available_error_stats_map
            capture_state["series_point_counts"] = point_counts_map
            capture_state["series_source_bin_widths"] = source_bin_widths
            capture_state["series_masked_bin_counts"] = masked_bin_counts
            capture_state["series_group_summaries"] = grouped_summaries
            capture_state["integration_summaries"] = list(integration_summaries)
            capture_state["annotations_summary"] = list(annotation_summaries)
            capture_state["x_axis_scale"] = float(resolved_x_axis_scale)
            capture_state["x_axis_offset"] = float(resolved_x_axis_offset)

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
                # Ensure the window is realized in GUI-preview mode.
                plt.pause(0.001)

        if not (show and not show_blocking):
            plt.close(fig)
        return output_path
