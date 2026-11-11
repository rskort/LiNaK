"""Rendering safeguards shared by interactive and exported plots."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.font_manager import font_scalings
from matplotlib.text import Text


def positive_render_value(value: Any, name: str) -> float:
    """Reject invalid values before they reach Qt, Agg, or FreeType."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


def _validate_font_size(value: Any, name: str) -> None:
    if isinstance(value, str) and value in font_scalings:
        return
    positive_render_value(value, name)


def validate_render_settings(
    settings: Mapping[str, Any], *, allow_figure_dpi: bool = False
) -> None:
    """Validate numerical text and figure settings, including advanced overrides."""
    for key, value in settings.items():
        if value is None:
            continue
        if key == "dpi" or key == "font_size" or key.endswith("_font_size"):
            if key == "dpi" and allow_figure_dpi and value == "figure":
                continue
            positive_render_value(value, key)
        elif key == "figsize":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("figsize must contain a width and height in inches.")
            for dimension in value:
                positive_render_value(dimension, "Figure dimensions")
        elif key in {"figure_kwargs", "savefig_kwargs", "style"} and isinstance(value, Mapping):
            validate_render_settings(value, allow_figure_dpi=key == "savefig_kwargs")
        elif key == "matplotlib_rc" and isinstance(value, Mapping):
            for rc_key in ("figure.dpi", "savefig.dpi"):
                if rc_key in value and not (rc_key == "savefig.dpi" and value[rc_key] == "figure"):
                    positive_render_value(value[rc_key], rc_key)
            for rc_key in (
                "font.size",
                "axes.labelsize",
                "axes.titlesize",
                "xtick.labelsize",
                "ytick.labelsize",
                "legend.fontsize",
                "legend.title_fontsize",
                "figure.titlesize",
                "figure.labelsize",
            ):
                if rc_key in value and value[rc_key] is not None:
                    _validate_font_size(value[rc_key], rc_key)
        elif key in {"legend_kwargs", "text_kwargs"} and isinstance(value, Mapping):
            for font_key in ("fontsize", "title_fontsize"):
                font_value = value.get(font_key)
                # Matplotlib also supports named relative font sizes.
                if font_value is not None:
                    _validate_font_size(font_value, font_key)
        elif key == "annotations" and isinstance(value, (list, tuple)):
            for annotation in value:
                if isinstance(annotation, Mapping):
                    validate_render_settings(annotation)


def preview_canvas_geometry(
    figure_size: tuple[float, float],
    available_size: tuple[int, int],
    *,
    minimum_dpi: float = 72.0,
) -> tuple[int, int, float]:
    """Fit a canvas where safe; use a scrollable minimum size in tiny viewports.

    Sizes and DPI here are logical Qt pixels. Device scaling is applied once
    by the caller, keeping the physical figure and widget dimensions in sync.
    """
    width, height = (positive_render_value(value, "Figure dimensions") for value in figure_size)
    floor = positive_render_value(minimum_dpi, "Minimum preview DPI")
    available_width, available_height = (max(1, int(value)) for value in available_size)
    dpi = max(min(available_width / width, available_height / height), floor)
    return max(1, math.ceil(width * dpi)), max(1, math.ceil(height * dpi)), dpi


def font_rendering_error(error: RuntimeError, dpi: float) -> ValueError:
    return ValueError(
        f"Could not render plot text at {dpi:g} DPI: {error}. "
        "Increase the rendering DPI or font size (including mathtext superscripts), "
        "or check the selected font and Matplotlib/FreeType installation."
    )


class PlotFigure(Figure):
    """Figure with consistent validation and actionable font errors on all backends."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        validate_render_settings({"matplotlib_rc": rcParams})
        validate_render_settings(kwargs)
        super().__init__(*args, **kwargs)

    def _validate_rendering(self) -> None:
        positive_render_value(self.dpi, "Rendering DPI")
        for dimension in self.get_size_inches():
            positive_render_value(dimension, "Figure dimensions")
        for artist in self.findobj(match=Text):
            positive_render_value(artist.get_fontsize(), "Text font size")

    def draw(self, renderer: Any) -> None:
        self._validate_rendering()
        try:
            super().draw(renderer)
        except RuntimeError as exc:
            if "invalid ppem" not in str(exc).lower():
                raise
            raise font_rendering_error(exc, self.dpi) from exc

    def tight_layout(self, **kwargs: Any) -> None:
        self._validate_rendering()
        try:
            super().tight_layout(**kwargs)
        except RuntimeError as exc:
            if "invalid ppem" not in str(exc).lower():
                raise
            raise font_rendering_error(exc, self.dpi) from exc

    def savefig(self, fname: Any, **kwargs: Any) -> None:
        validate_render_settings(kwargs, allow_figure_dpi=True)
        super().savefig(fname, **kwargs)
