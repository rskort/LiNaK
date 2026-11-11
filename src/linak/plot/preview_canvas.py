"""Qt canvas that reports errors from deferred draws to the plotting GUI."""

from collections.abc import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class PlotPreviewCanvas(FigureCanvasQTAgg):
    def __init__(
        self, figure, *, on_error: Callable[[Exception], None], on_recovered: Callable[[], None]
    ):
        self._on_draw_error = on_error
        self._on_draw_recovered = on_recovered
        self._draw_failed = False
        super().__init__(figure)

    def draw(self):
        # Qt's draw_idle executes later, outside the GUI caller's try/except.
        try:
            super().draw()
        except Exception as exc:
            self._draw_failed = True
            self._on_draw_error(exc)
        else:
            if self._draw_failed:
                self._draw_failed = False
                self._on_draw_recovered()
