"""Shared plotting helpers and style definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import difflib
import logging
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np

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
    title_font_size: int = 14
    label_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 10
    line_width: float = 2.0
    line_color: str = "#1f77b4"
    grid: bool = True
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.8
    grid_alpha: float = 0.35


DEFAULT_PLOT_STYLE = PlotStyle()


def format_axis_label_units(label: str) -> str:
    """Return the axis label exactly as provided by the caller."""
    return str(label)


def resolve_explicit_plot_text(value: str | None, default: str) -> str:
    """Preserve explicit blank strings while still filling missing values from defaults."""
    return default if value is None else str(value)


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


def with_style_overrides(
    *,
    base_style: PlotStyle = DEFAULT_PLOT_STYLE,
    figure_size: tuple[float, float] | None = None,
    dpi: int | None = None,
    font_family: str | None = None,
    title_font_size: int | None = None,
    label_font_size: int | None = None,
    tick_font_size: int | None = None,
    legend_font_size: int | None = None,
    line_width: float | None = None,
    line_color: str | None = None,
    grid: bool | None = None,
    grid_linestyle: str | None = None,
    grid_linewidth: float | None = None,
    grid_alpha: float | None = None,
) -> PlotStyle:
    """Return a :class:`PlotStyle` with explicit overrides applied."""
    updates: dict[str, Any] = {}
    if figure_size is not None:
        updates["figure_size"] = figure_size
    if dpi is not None:
        updates["dpi"] = dpi
    if font_family is not None:
        updates["font_family"] = font_family
    if title_font_size is not None:
        updates["title_font_size"] = title_font_size
    if label_font_size is not None:
        updates["label_font_size"] = label_font_size
    if tick_font_size is not None:
        updates["tick_font_size"] = tick_font_size
    if legend_font_size is not None:
        updates["legend_font_size"] = legend_font_size
    if line_width is not None:
        updates["line_width"] = line_width
    if line_color is not None:
        updates["line_color"] = line_color
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
    series_normalization_modes: list[str] | None = None,
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
                    LOGGER.info("Configured Matplotlib backend '%s'.", active)
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
    LOGGER.info("Configured Matplotlib backend '%s'.", active)
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

    axis_hint = _normalize_tick_axis(axis_hint_raw) if axis_hint_raw is not None else "both"
    if axis_hint_raw is None and "axis" in resolved:
        axis_hint = _normalize_tick_axis(resolved["axis"])

    minor_mode = _normalize_minor_ticks_mode(minor_mode_raw)
    return resolved, axis_hint, minor_mode


def _capture_plot_state(
    *,
    ax: Any,
    style: PlotStyle,
    line_colors: list[str],
    line_labels: list[str],
    line_markers: list[str],
    legend_loc: str,
    capture_state: dict[str, Any] | None,
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
        legend_kwargs = {
            "frameon": bool(legend.get_frame_on()),
            "ncols": int(getattr(legend, "_ncols", 1)),
        }
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

    capture_state.clear()
    capture_state.update(
        {
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
            "series_labels": list(line_labels),
            "figsize": [float(style.figure_size[0]), float(style.figure_size[1])],
            "dpi": int(style.dpi),
            "font_family": style.font_family,
            "title_font_size": int(style.title_font_size),
            "label_font_size": int(style.label_font_size),
            "tick_font_size": int(style.tick_font_size),
            "legend_font_size": int(style.legend_font_size),
            "line_width": float(style.line_width),
            "line_kwargs": line_kwargs,
            "axes_kwargs": axes_kwargs,
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
        }
    )


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
) -> tuple[np.ndarray, bool]:
    if mode == "none":
        return y, False

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
    return y * scale, True


def _prepare_plot_series_data(
    *,
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    labels: list[str],
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    series_normalization_modes: list[str] | None = None,
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
        y_data, applied = _normalize_series_values(
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
    normalization_mode: str | None = None,
    normalization_value: float | None = None,
    normalization_x_ref: float | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
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
    transformed_x, transformed_y, _normalized_count = _prepare_plot_series_data(
        x_series=[np.asarray(x, dtype=float)],
        y_series=[np.asarray(y, dtype=float)],
        labels=["series"],
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        series_normalization_modes=None if normalization_mode is None else [normalization_mode],
        series_normalization_values=None if normalization_value is None else [normalization_value],
        series_normalization_x_refs=None if normalization_x_ref is None else [normalization_x_ref],
    )
    x_plot = transformed_x[0]
    y_plot = transformed_y[0]

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
        if figure_kwargs is not None:
            fig.set(**dict(figure_kwargs))

        color = line_color or style.line_color
        marker = ("o" if markers else "") if line_marker is None else str(line_marker)
        line_artist = None
        fit_summary_key = str(series_id or "series")
        fit_summaries: dict[str, dict[str, Any]] = {}
        series_stats: dict[str, dict[str, float | int | None]] = {}
        resolved_fit_config = resolve_series_fit_configs(
            series_count=1,
            series_fit_configs=[fit_config],
            series_fit_enabled=[fit_enabled],
            series_fit_labels=[fit_label],
            series_fit_show_in_legend=[fit_show_in_legend],
        )[0]
        if not bool(resolved_fit_config.get("fit_enabled")):
            fit_summaries[fit_summary_key] = {
                "fit_enabled": False,
                "status": "off",
                "fit_type": str(resolved_fit_config.get("fit_type") or "linear"),
                "point_count": 0,
            }
        if line_visible:
            resolved_line_kwargs: dict[str, Any] = {
                "lw": style.line_width
                if line_width_override is None
                else float(line_width_override),
                "color": color,
                "label": line_label if show_in_legend else "_nolegend_",
                "marker": marker,
            }
            if line_kwargs is not None:
                resolved_line_kwargs.update(dict(line_kwargs))
            if line_label is not None:
                resolved_line_kwargs["label"] = line_label
            (line_artist,) = ax.plot(
                x_plot,
                y_plot,
                **resolved_line_kwargs,
            )
            series_stats[fit_summary_key] = _series_statistics(x_plot, y_plot)

        if line_artist is not None and bool(resolved_fit_config.get("fit_enabled")):
            fit_summary = execute_series_fit(
                x_plot,
                y_plot,
                fit_config=resolved_fit_config,
                visible_x_lim=x_lim,
            )
            fit_render_label = (
                str(resolved_fit_config.get("fit_label_override") or "").strip()
                or f"{line_label or 'Series'} fit"
            )
            fit_summary["fit_enabled"] = True
            fit_summary["label"] = fit_render_label
            fit_summaries[fit_summary_key] = fit_summary
            if fit_summary.get("status") == "ok":
                fit_kwargs: dict[str, Any] = {
                    "color": str(line_artist.get_color()),
                    "linestyle": "--",
                    "linewidth": float(line_artist.get_linewidth()),
                    "marker": "",
                    "label": (
                        fit_render_label
                        if bool(resolved_fit_config.get("fit_show_in_legend", True))
                        else "_nolegend_"
                    ),
                }
                line_alpha = line_artist.get_alpha()
                if line_alpha is not None:
                    fit_kwargs["alpha"] = float(line_alpha)
                ax.plot(
                    np.asarray(fit_summary.get("x_fit", []), dtype=float),
                    np.asarray(fit_summary.get("y_fit", []), dtype=float),
                    **fit_kwargs,
                )
        elif bool(resolved_fit_config.get("fit_enabled")):
            fit_summaries[fit_summary_key] = {
                "fit_enabled": True,
                "status": "disabled",
                "fit_type": str(resolved_fit_config.get("fit_type") or "linear"),
            }

        should_show_legend = (
            bool(ax.get_legend_handles_labels()[1])
            if legend is None
            else bool(legend and ax.get_legend_handles_labels()[1])
        )
        if should_show_legend:
            resolved_legend_kwargs: dict[str, Any] = {
                "fontsize": style.legend_font_size,
                "title": legend_title,
                "loc": legend_loc,
            }
            if legend_kwargs is not None:
                resolved_legend_kwargs.update(dict(legend_kwargs))
            ax.legend(**resolved_legend_kwargs)

        xlabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(format_axis_label_units(x_label), **xlabel_kwargs)
        ax.set_ylabel(format_axis_label_units(y_label), **ylabel_kwargs)
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size)
        else:
            ax.set_title(title, fontsize=style.title_font_size)
        ax.tick_params(axis="both", labelsize=style.tick_font_size)
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
        else:
            if ticks_visible is True and tick_axis_hint in {"x", "y"}:
                if tick_axis_hint == "x":
                    ax.tick_params(
                        axis="y",
                        which="both",
                        left=False,
                        right=False,
                        labelleft=False,
                    )
                else:
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
            if x_tick_rotation is not None:
                ax.tick_params(axis="x", rotation=float(x_tick_rotation))
            if y_tick_rotation is not None:
                ax.tick_params(axis="y", rotation=float(y_tick_rotation))
        if minor_ticks_mode == "on":
            ax.minorticks_on()
        elif minor_ticks_mode == "off":
            ax.minorticks_off()
        if resolved_tick_params_kwargs:
            ax.tick_params(**resolved_tick_params_kwargs)
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
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))

        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()
        _capture_plot_state(
            ax=ax,
            style=style,
            line_colors=[str(line_artist.get_color())] if line_artist is not None else [],
            line_labels=[str(line_label or line_artist.get_label())]
            if line_artist is not None
            else [],
            line_markers=[str(line_artist.get_marker())] if line_artist is not None else [],
            legend_loc=legend_loc,
            capture_state=capture_state,
        )
        if capture_state is not None:
            capture_state["series_fit_summaries"] = fit_summaries
            capture_state["series_statistics"] = series_stats

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
    series_fit_enabled: list[bool] | None = None,
    series_fit_labels: list[str | None] | None = None,
    series_fit_show_in_legend: list[bool] | None = None,
    series_normalization_modes: list[str] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
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
    if series_ids is not None and len(series_ids) != len(labels):
        raise ValueError("series_ids count must match the number of plotted series.")
    if not x_series:
        raise ValueError("At least one series is required for multi-line plotting.")

    transformed_x_series, transformed_y_series, normalized_count = _prepare_plot_series_data(
        x_series=[np.asarray(values, dtype=float) for values in x_series],
        y_series=[np.asarray(values, dtype=float) for values in y_series],
        labels=labels,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
    )
    if len(labels) > 1 and 0 < normalized_count < len(labels):
        LOGGER.warning(
            "Only %d/%d plotted series are normalized. Interpret y-axis comparisons with care.",
            normalized_count,
            len(labels),
        )

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
        if figure_kwargs is not None:
            fig.set(**dict(figure_kwargs))
        rendered_colors: list[str] = []
        rendered_markers: list[str] = []
        rendered_labels: list[str] = []
        resolved_line_colors = resolve_series_colors(line_colors, series_count=len(labels))
        if series_enabled is not None and len(series_enabled) != len(labels):
            raise ValueError(
                f"series_enabled count must match the number of plotted series ({len(labels)})."
            )
        if series_show_in_legend is not None and len(series_show_in_legend) != len(labels):
            raise ValueError(
                f"series_show_in_legend count must match the number of plotted series ({len(labels)})."
            )
        if series_line_widths is not None and len(series_line_widths) != len(labels):
            raise ValueError(
                f"series_line_widths count must match the number of plotted series ({len(labels)})."
            )
        if series_markers is not None and len(series_markers) != len(labels):
            raise ValueError(
                f"series_markers count must match the number of plotted series ({len(labels)})."
            )
        if series_line_kwargs is not None and len(series_line_kwargs) != len(labels):
            raise ValueError(
                f"series_line_kwargs count must match the number of plotted series ({len(labels)})."
            )
        if series_fit_enabled is not None and len(series_fit_enabled) != len(labels):
            raise ValueError(
                f"series_fit_enabled count must match the number of plotted series ({len(labels)})."
            )
        if series_fit_configs is not None and len(series_fit_configs) != len(labels):
            raise ValueError(
                f"series_fit_configs count must match the number of plotted series ({len(labels)})."
            )
        if series_fit_labels is not None and len(series_fit_labels) != len(labels):
            raise ValueError(
                f"series_fit_labels count must match the number of plotted series ({len(labels)})."
            )
        if series_fit_show_in_legend is not None and len(series_fit_show_in_legend) != len(labels):
            raise ValueError(
                f"series_fit_show_in_legend count must match the number of plotted series ({len(labels)})."
            )
        resolved_fit_configs = resolve_series_fit_configs(
            series_count=len(labels),
            series_fit_configs=series_fit_configs,
            series_fit_enabled=series_fit_enabled,
            series_fit_labels=series_fit_labels,
            series_fit_show_in_legend=series_fit_show_in_legend,
        )
        fit_summaries: dict[str, dict[str, Any]] = {}
        series_stats: dict[str, dict[str, float | int | None]] = {}
        for index, (x_values, y_values, label) in enumerate(
            zip(transformed_x_series, transformed_y_series, labels)
        ):
            if series_enabled is not None and not bool(series_enabled[index]):
                if bool(resolved_fit_configs[index].get("fit_enabled")):
                    fit_key = (
                        str(series_ids[index]) if series_ids is not None else f"series:{index}"
                    )
                    fit_summaries[fit_key] = {
                        "fit_enabled": True,
                        "status": "disabled",
                        "fit_type": str(resolved_fit_configs[index].get("fit_type") or "linear"),
                    }
                continue
            kwargs: dict[str, Any] = {
                "lw": style.line_width,
                "label": (
                    label
                    if series_show_in_legend is None or bool(series_show_in_legend[index])
                    else "_nolegend_"
                ),
            }
            line_width_override = None if series_line_widths is None else series_line_widths[index]
            if line_width_override is not None:
                kwargs["lw"] = float(line_width_override)
            marker_value = "o" if markers else ""
            if series_markers is not None and series_markers[index] is not None:
                marker_value = str(series_markers[index])
            kwargs["marker"] = marker_value
            if resolved_line_colors is not None:
                color_token = str(resolved_line_colors[index]).strip()
                if color_token:
                    kwargs["color"] = color_token
            if line_kwargs is not None:
                kwargs.update(dict(line_kwargs))
            line_kwargs_override = None if series_line_kwargs is None else series_line_kwargs[index]
            if line_kwargs_override is not None:
                kwargs.update(dict(line_kwargs_override))
            kwargs["label"] = (
                label
                if series_show_in_legend is None or bool(series_show_in_legend[index])
                else "_nolegend_"
            )
            (artist,) = ax.plot(x_values, y_values, **kwargs)
            rendered_colors.append(str(artist.get_color()))
            rendered_markers.append(str(artist.get_marker()))
            rendered_labels.append(str(artist.get_label()))
            fit_key = str(series_ids[index]) if series_ids is not None else f"series:{index}"
            series_stats[fit_key] = _series_statistics(x_values, y_values)
            fit_config = resolved_fit_configs[index]
            if not bool(fit_config.get("fit_enabled")):
                fit_summaries[fit_key] = {
                    "fit_enabled": False,
                    "status": "off",
                    "fit_type": str(fit_config.get("fit_type") or "linear"),
                    "point_count": 0,
                }
                continue
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
            if fit_summary.get("status") != "ok":
                continue
            fit_kwargs: dict[str, Any] = {
                "color": str(artist.get_color()),
                "linestyle": "--",
                "linewidth": float(artist.get_linewidth()),
                "marker": "",
                "label": (
                    fit_render_label
                    if bool(fit_config.get("fit_show_in_legend", True))
                    else "_nolegend_"
                ),
            }
            line_alpha = artist.get_alpha()
            if line_alpha is not None:
                fit_kwargs["alpha"] = float(line_alpha)
            ax.plot(
                np.asarray(fit_summary.get("x_fit", []), dtype=float),
                np.asarray(fit_summary.get("y_fit", []), dtype=float),
                **fit_kwargs,
            )

        should_show_legend = (
            len(rendered_labels) > 1 if legend is None else bool(legend and rendered_labels)
        )
        if should_show_legend:
            resolved_legend_kwargs: dict[str, Any] = {
                "fontsize": style.legend_font_size,
                "title": legend_title,
                "loc": legend_loc,
            }
            if legend_kwargs is not None:
                resolved_legend_kwargs.update(dict(legend_kwargs))
            ax.legend(**resolved_legend_kwargs)
        xlabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        ylabel_kwargs: dict[str, Any] = {"fontsize": style.label_font_size}
        if x_label_pad is not None:
            xlabel_kwargs["labelpad"] = float(x_label_pad)
        if y_label_pad is not None:
            ylabel_kwargs["labelpad"] = float(y_label_pad)
        ax.set_xlabel(format_axis_label_units(x_label), **xlabel_kwargs)
        ax.set_ylabel(format_axis_label_units(y_label), **ylabel_kwargs)
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size)
        else:
            ax.set_title(title, fontsize=style.title_font_size)
        ax.tick_params(axis="both", labelsize=style.tick_font_size)
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
        else:
            if ticks_visible is True and tick_axis_hint in {"x", "y"}:
                if tick_axis_hint == "x":
                    ax.tick_params(
                        axis="y",
                        which="both",
                        left=False,
                        right=False,
                        labelleft=False,
                    )
                else:
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
            if x_tick_rotation is not None:
                ax.tick_params(axis="x", rotation=float(x_tick_rotation))
            if y_tick_rotation is not None:
                ax.tick_params(axis="y", rotation=float(y_tick_rotation))
        if minor_ticks_mode == "on":
            ax.minorticks_on()
        elif minor_ticks_mode == "off":
            ax.minorticks_off()
        if resolved_tick_params_kwargs:
            ax.tick_params(**resolved_tick_params_kwargs)
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
        if axes_kwargs is not None:
            ax.set(**dict(axes_kwargs))

        if tight_layout_kwargs is not None:
            fig.tight_layout(**dict(tight_layout_kwargs))
        else:
            fig.tight_layout()
        _capture_plot_state(
            ax=ax,
            style=style,
            line_colors=rendered_colors,
            line_labels=rendered_labels,
            line_markers=rendered_markers,
            legend_loc=legend_loc,
            capture_state=capture_state,
        )
        if capture_state is not None:
            capture_state["series_fit_summaries"] = fit_summaries
            capture_state["series_statistics"] = series_stats

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
