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
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERACTIVE_BACKEND = "TkAgg"
CANONICAL_INTERACTIVE_BACKENDS = ("TkAgg", "QtAgg", "GTK3Agg", "WXAgg", "MacOSX")
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


@dataclass(frozen=True)
class PlotStyle:
    """Plot style settings reused across all analysis plots."""

    figure_size: tuple[float, float] = (7.0, 4.0)
    dpi: int = 200
    font_family: str = "DejaVu Sans"
    title_font_size: int = 14
    label_font_size: int = 12
    tick_font_size: int = 10
    line_width: float = 2.0
    line_color: str = "#1f77b4"
    grid: bool = True
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.8
    grid_alpha: float = 0.35


DEFAULT_PLOT_STYLE = PlotStyle()


def with_style_overrides(
    *,
    base_style: PlotStyle = DEFAULT_PLOT_STYLE,
    figure_size: tuple[float, float] | None = None,
    dpi: int | None = None,
    font_family: str | None = None,
    title_font_size: int | None = None,
    label_font_size: int | None = None,
    tick_font_size: int | None = None,
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


def ensure_interactive_backend(preferred_backend: str | None = None) -> str:
    """Ensure Matplotlib uses an interactive backend or raise RuntimeError."""
    current = matplotlib.get_backend()
    if _is_interactive_backend(current):
        return current

    LOGGER.info("Current Matplotlib backend '%s' is non-interactive.", current)
    candidates = list(CANONICAL_INTERACTIVE_BACKENDS)
    if preferred_backend:
        preferred_backend = normalize_backend_name(preferred_backend)
        candidates = [preferred_backend, *[c for c in candidates if c != preferred_backend]]
    LOGGER.info("Trying interactive backends in order: %s", ", ".join(candidates))

    for i, candidate in enumerate(candidates):
        try:
            plt.switch_backend(candidate)
            new_backend = matplotlib.get_backend()
            if _is_interactive_backend(new_backend):
                LOGGER.info("Using interactive Matplotlib backend '%s'.", new_backend)
                return new_backend
        except Exception as exc:  # pragma: no cover - environment dependent
            if i == 0:
                LOGGER.info("Could not activate preferred backend '%s': %s", candidate, exc)
            else:
                LOGGER.debug("Could not activate backend '%s': %s", candidate, exc)

    active = matplotlib.get_backend()
    if not _has_graphical_display():
        raise RuntimeError(
            "Interactive plotting requested but no graphical display is available. "
            "Use X11/Wayland forwarding, or run with --no-show and --output."
        )

    raise RuntimeError(
        f"Interactive plotting requested, but active backend '{active}' is non-interactive. "
        f"Attempted backends: {', '.join(candidates)}. "
        "Install an interactive backend (Tk/Qt/GTK) or set MPLBACKEND accordingly."
    )


def _to_float_list(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


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
            "line_colors": list(line_colors),
            "markers": any(marker not in {"", "None", "none", " ", "NoneType"} for marker in line_markers),
            "series_labels": list(line_labels),
            "figsize": [float(style.figure_size[0]), float(style.figure_size[1])],
            "dpi": int(style.dpi),
            "font_family": style.font_family,
            "title_font_size": int(style.title_font_size),
            "label_font_size": int(style.label_font_size),
            "tick_font_size": int(style.tick_font_size),
            "line_width": float(style.line_width),
            "grid": bool(style.grid),
            "grid_linestyle": style.grid_linestyle,
            "grid_linewidth": float(style.grid_linewidth),
            "grid_alpha": float(style.grid_alpha),
        }
    )


def plot_line_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    output: str | Path | None = None,
    show: bool = True,
    preferred_backend: str | None = None,
    line_label: str | None = None,
    line_color: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    capture_state: dict[str, Any] | None = None,
) -> Path | None:
    """Plot a single line using the shared LiNaK style."""
    with plt.rc_context({"font.family": style.font_family}):
        fig, ax = plt.subplots(figsize=style.figure_size)

        color = line_color or style.line_color
        marker = "o" if markers else ""
        (line_artist,) = ax.plot(x, y, lw=style.line_width, color=color, label=line_label, marker=marker)

        should_show_legend = bool(line_label) if legend is None else bool(legend and line_label)
        if should_show_legend:
            ax.legend(
                fontsize=style.tick_font_size,
                title=legend_title,
                loc=legend_loc,
            )

        ax.set_xlabel(x_label, fontsize=style.label_font_size)
        ax.set_ylabel(y_label, fontsize=style.label_font_size)
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size)
        else:
            ax.set_title(title, fontsize=style.title_font_size)
        ax.tick_params(axis="both", labelsize=style.tick_font_size)

        if style.grid:
            ax.grid(
                True,
                linestyle=style.grid_linestyle,
                linewidth=style.grid_linewidth,
                alpha=style.grid_alpha,
            )
        if ticks_visible is False:
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
        else:
            if x_tick_rotation is not None:
                ax.tick_params(axis="x", rotation=float(x_tick_rotation))
            if y_tick_rotation is not None:
                ax.tick_params(axis="y", rotation=float(y_tick_rotation))
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

        fig.tight_layout()
        _capture_plot_state(
            ax=ax,
            style=style,
            line_colors=[str(line_artist.get_color())],
            line_labels=[str(line_label) if line_label else "series_1"],
            line_markers=[str(line_artist.get_marker())],
            legend_loc=legend_loc,
            capture_state=capture_state,
        )

        output_path = None
        if output is not None:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=style.dpi)
            LOGGER.info("Saved plot to '%s'.", output_path)

        if show:
            backend = ensure_interactive_backend(preferred_backend=preferred_backend)
            LOGGER.info(
                "Showing interactive plot window using backend '%s'. Close the window to continue.",
                backend,
            )
            plt.show()

        plt.close(fig)
        return output_path


def plot_multi_line_series(
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    labels: list[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
    output: str | Path | None = None,
    show: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    line_colors: list[str] | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    capture_state: dict[str, Any] | None = None,
) -> Path | None:
    """Plot multiple line series in a single axes using the shared LiNaK style."""
    if not (len(x_series) == len(y_series) == len(labels)):
        raise ValueError("x_series, y_series, and labels must have equal lengths.")
    if not x_series:
        raise ValueError("At least one series is required for multi-line plotting.")

    with plt.rc_context({"font.family": style.font_family}):
        fig, ax = plt.subplots(figsize=style.figure_size)
        rendered_colors: list[str] = []
        rendered_markers: list[str] = []
        if line_colors is not None and len(line_colors) != len(labels):
            raise ValueError(
                "line_colors count must match the number of plotted series "
                f"({len(labels)})."
            )
        for index, (x_values, y_values, label) in enumerate(zip(x_series, y_series, labels)):
            kwargs: dict[str, Any] = {"lw": style.line_width, "label": label}
            kwargs["marker"] = "o" if markers else ""
            if line_colors is not None:
                kwargs["color"] = line_colors[index]
            (artist,) = ax.plot(x_values, y_values, **kwargs)
            rendered_colors.append(str(artist.get_color()))
            rendered_markers.append(str(artist.get_marker()))

        should_show_legend = len(labels) > 1 if legend is None else bool(legend)
        if should_show_legend:
            ax.legend(
                fontsize=style.tick_font_size,
                title=legend_title,
                loc=legend_loc,
            )
        ax.set_xlabel(x_label, fontsize=style.label_font_size)
        ax.set_ylabel(y_label, fontsize=style.label_font_size)
        if title_visible is False:
            ax.set_title("", fontsize=style.title_font_size)
        else:
            ax.set_title(title, fontsize=style.title_font_size)
        ax.tick_params(axis="both", labelsize=style.tick_font_size)

        if style.grid:
            ax.grid(
                True,
                linestyle=style.grid_linestyle,
                linewidth=style.grid_linewidth,
                alpha=style.grid_alpha,
            )
        if ticks_visible is False:
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
        else:
            if x_tick_rotation is not None:
                ax.tick_params(axis="x", rotation=float(x_tick_rotation))
            if y_tick_rotation is not None:
                ax.tick_params(axis="y", rotation=float(y_tick_rotation))
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

        fig.tight_layout()
        _capture_plot_state(
            ax=ax,
            style=style,
            line_colors=rendered_colors,
            line_labels=[str(label) for label in labels],
            line_markers=rendered_markers,
            legend_loc=legend_loc,
            capture_state=capture_state,
        )

        output_path = None
        if output is not None:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=style.dpi)
            LOGGER.info("Saved plot to '%s'.", output_path)

        if show:
            backend = ensure_interactive_backend(preferred_backend=preferred_backend)
            LOGGER.info(
                "Showing interactive plot window using backend '%s'. Close the window to continue.",
                backend,
            )
            plt.show()

        plt.close(fig)
        return output_path
