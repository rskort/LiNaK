"""Qt-independent preparation and progress for one Studio session."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ..cancellation import check_cancelled

LOGGER = logging.getLogger("linak")


@dataclass(frozen=True)
class PlotGuiSession:
    title: str
    initial_settings: dict[str, Any]
    callbacks: dict[str, Any]
    options: dict[str, Any]
    cleanups: tuple[Callable[[], None], ...] = ()

    @classmethod
    def from_parameters(cls, parameters):
        values = dict(parameters)
        title, settings = values.pop("title"), values.pop("initial_settings")
        callbacks = {key: values.pop(key) for key in tuple(values) if key.startswith("on_")}
        return cls(title, settings, callbacks, values)

    def parameters(self):
        return dict(
            title=self.title,
            initial_settings=self.initial_settings,
            **self.callbacks,
            **self.options,
        )

    def discard(self):
        for cleanup in reversed(self.cleanups):
            try:
                cleanup()
            except OSError:
                LOGGER.debug("Could not remove a cancelled Studio output", exc_info=True)


@dataclass(frozen=True)
class PlotProgress:
    stage: str
    source: str = ""
    index: int | None = None
    total: int | None = None

    @property
    def title(self) -> str:
        """Short, friendly captions tied to actual work, rather than a timer."""
        stage = self.stage.rstrip(". …").casefold()
        return {
            "checking files": "Reading your results",
            "preparing combined results": "Bringing results together",
            "preparing controls": "Setting up your workspace",
            "loading plot data": "Gathering your data",
            "updating preview": "Updating your plot",
            "rendering preview": "Building your plot",
            "aligning interfaces": "Lining up the interfaces",
            "checking interface alignment": "Checking the interfaces",
            "projecting orientation": "Shaping your map",
            "calculating density": "Measuring the density",
            "drawing plot": "Drawing the details",
            "displaying plot": "Bringing your plot into view",
        }.get(stage, self.stage.rstrip(". …"))

    @property
    def message(self):
        suffix = f" ({self.index}/{self.total})" if self.index is not None and self.total else ""
        return f"{self.stage}{suffix}" + (f" — {self.source}" if self.source else "")


@dataclass
class _Preparation:
    sessions: list[PlotGuiSession] = field(default_factory=list)
    cleanups: list[Callable[[], None]] = field(default_factory=list)


_PREPARATION: ContextVar[_Preparation | None] = ContextVar("plot_session_preparation", default=None)
_PROGRESS: ContextVar[Callable[[PlotProgress], None] | None] = ContextVar(
    "plot_progress", default=None
)


def preparing_session() -> bool:
    return _PREPARATION.get() is not None


def capture_session(session: PlotGuiSession) -> bool:
    preparation = _PREPARATION.get()
    if preparation is None:
        return False
    check_cancelled()
    preparation.sessions.append(session)
    return True


def own_preparation_output(path):
    """Only register a newly created output, never an existing source."""
    preparation = _PREPARATION.get()
    if preparation is not None:
        output = Path(path)
        preparation.cleanups.append(lambda: output.unlink(missing_ok=True))


@contextmanager
def progress_scope(callback):
    token = _PROGRESS.set(callback)
    try:
        yield
    finally:
        _PROGRESS.reset(token)


def report_progress(stage, source="", index=None, total=None):
    check_cancelled()
    update = PlotProgress(stage, str(source), index, total)
    callback = _PROGRESS.get()
    if callback is not None:
        callback(update)
    LOGGER.debug("Studio: %s", update.message)


def prepare_session(builder) -> PlotGuiSession:
    started = perf_counter()
    preparation = _Preparation()
    token = _PREPARATION.set(preparation)
    try:
        result = builder()
        check_cancelled()
        if result != 0 or len(preparation.sessions) != 1:
            raise RuntimeError("Studio preparation did not produce a plotting session.")
        LOGGER.debug("Studio session preparation seconds=%.6f", perf_counter() - started)
        from dataclasses import replace

        return replace(preparation.sessions[0], cleanups=tuple(preparation.cleanups))
    except BaseException:
        for cleanup in reversed(preparation.cleanups):
            cleanup()
        raise
    finally:
        _PREPARATION.reset(token)
