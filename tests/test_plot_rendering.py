"""Regressions for tiny previews and invalid FreeType font sizes."""

from io import BytesIO

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from linak.plot.plotting import PlotStyle, plot_heatmap_series, plot_multi_line_series
from linak.plot.rendering import PlotFigure, preview_canvas_geometry
from linak.plot.settings_validation import validate_current_plot_settings


@pytest.mark.parametrize("viewport", [(1, 1), (7, 4), (30, 20), (800, 520), (1600, 900)])
@pytest.mark.parametrize("ratio", [1, 1.25, 2])
def test_small_preview_renders_mathtext_and_preserves_export_geometry(viewport, ratio):
    width, height, dpi = preview_canvas_geometry((7, 4), viewport)
    assert dpi >= 72
    assert width / height == pytest.approx(7 / 4, abs=0.01)
    assert width == pytest.approx(7 * dpi, abs=1)
    assert height == pytest.approx(4 * dpi, abs=1)
    fig = PlotFigure(figsize=(7, 4), dpi=dpi * ratio)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], label=r"$H_2O$")
    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)", fontsize=12)
    ax.set_ylabel(r"Density (g/cm$^3$)", fontsize=12)
    ax.legend(fontsize=10)
    fig.canvas.draw()
    assert fig.canvas.get_width_height() == (int(7 * dpi * ratio), int(4 * dpi * ratio))
    # A preview's safe DPI must never replace the requested export DPI.
    data = BytesIO()
    fig.savefig(data, format="png", dpi=200)
    data.seek(0)
    with Image.open(data) as image:
        assert image.size == (1400, 800)
    assert fig.dpi == pytest.approx(dpi * ratio)


@pytest.mark.parametrize(
    "settings",
    [
        {"dpi": 0},
        {"font_size": float("nan")},
        {"x_label_font_size": -1},
        {"figsize": [7, float("inf")]},
        {"figsize": [0, 4]},
        {"figure_kwargs": {"dpi": -1}},
        {"savefig_kwargs": {"dpi": float("nan")}},
        {"matplotlib_rc": {"figure.dpi": float("inf")}},
        {"matplotlib_rc": {"xtick.labelsize": float("nan")}},
        {"legend_kwargs": {"fontsize": "nan"}},
        {"annotations": [{"font_size": "nan"}]},
        {"figure_kwargs": {"dpi": "figure"}},
        {"style": {"legend_font_size": -1}},
    ],
)
def test_saved_and_advanced_settings_reject_invalid_render_values(settings):
    with pytest.raises(ValueError, match="finite positive"):
        validate_current_plot_settings(settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"dpi": 0},
        {"figure_size": (float("nan"), 4)},
        {"base_font_size": -1},
        {"label_font_size": float("inf")},
    ],
)
def test_python_plot_style_rejects_invalid_render_values(settings):
    with pytest.raises(ValueError, match="finite positive"):
        PlotStyle(**settings)


def test_valid_relative_fonts_and_inherited_export_dpi_are_supported():
    validate_current_plot_settings(
        {
            "savefig_kwargs": {"dpi": "figure"},
            "legend_kwargs": {"fontsize": "small", "title_fontsize": "large"},
            "matplotlib_rc": {"axes.titlesize": "large", "savefig.dpi": "figure"},
        }
    )
    fig = PlotFigure(figsize=(2, 1), dpi=100)
    FigureCanvasAgg(fig)
    fig.savefig(BytesIO(), format="png", dpi="figure")


@pytest.mark.parametrize("operation", ["draw", "tight_layout", "savefig"])
def test_font_size_failure_has_actionable_error(monkeypatch, operation):
    fig = PlotFigure(figsize=(7, 4))
    FigureCanvasAgg(fig)

    def fail(*args, **kwargs):
        raise RuntimeError("Could not set the fontsize (invalid ppem value; error code 0x97)")

    # Simulate a font-specific failure independently of the installed FreeType version.
    monkeypatch.setattr(Figure, "tight_layout" if operation == "tight_layout" else "draw", fail)
    with pytest.raises(ValueError, match="Increase the rendering DPI or font size"):
        if operation == "savefig":
            fig.savefig(BytesIO(), format="png")
        elif operation == "tight_layout":
            fig.tight_layout()
        else:
            fig.canvas.draw()


def test_unrelated_render_failure_is_not_reclassified(monkeypatch):
    fig = PlotFigure()
    FigureCanvasAgg(fig)

    def fail(*args, **kwargs):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(Figure, "draw", fail)
    with pytest.raises(RuntimeError, match="^unrelated failure$"):
        fig.canvas.draw()


@pytest.mark.parametrize("heatmap", [False, True])
def test_shared_plot_functions_use_safe_figure_and_keep_mathtext(tmp_path, heatmap):
    state = {}
    options = dict(
        title="Density",
        x_label=r"Distance ($\mathrm{\AA}$)",
        y_label=r"Density (g/cm$^3$)",
        show=False,
        capture_state=state,
        output=tmp_path / "density.svg",
    )
    if heatmap:
        plot_heatmap_series(np.arange(3), np.arange(3), np.ones((2, 2)), **options)
    else:
        plot_multi_line_series([np.arange(3)], [np.arange(3)], ["OH"], **options)
    assert isinstance(state["axes"].figure, PlotFigure)
    assert state["axes"].get_xlabel() == options["x_label"]
    assert options["output"].stat().st_size > 0
