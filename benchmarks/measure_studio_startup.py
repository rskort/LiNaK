"""Measure fresh-process and warm Studio startup through the first canvas draw.

Run from the repository root with one existing analysis HDF5 file. Each child
opens it twice without saving plot settings or modifying analysis files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter
from typing import Any


def child(path):
    started = perf_counter()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import linak.cli as cli
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow
    from linak.plot.preview_canvas import PlotPreviewCanvas

    imports_s = perf_counter() - started
    application = QApplication.instance() or QApplication([])
    results: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    original_paint, original_draw = QMainWindow.paintEvent, PlotPreviewCanvas.draw

    def paint(window, event):
        original_paint(window, event)
        if hasattr(window, "_session_ready"):
            state.setdefault("first_paint_s", perf_counter() - state["started"])

    def draw(canvas):
        original_draw(canvas)
        if not canvas._draw_failed:
            state.setdefault("first_preview_s", perf_counter() - state["started"])

    QMainWindow.paintEvent, PlotPreviewCanvas.draw = paint, draw

    def check():
        windows = [
            window
            for window in application.topLevelWidgets()
            if hasattr(window, "_session_ready") and window.isVisible()
        ]
        if windows:
            window = windows[0]
            failed = window._startup_failed or (window._session_ready and window._preview_error)
            if failed or perf_counter() - state["started"] > 60:
                state["error"] = (
                    window._startup_label.text()
                    if window._startup_failed
                    else str(getattr(window, "_preview_error", None) or "Timeout")
                )
            if "first_preview_s" in state or "error" in state:
                if window._session_ready:
                    window._saved_signature = window._signature(window._collect_settings())
                window.close()
                window.deleteLater()
                return
        QTimer.singleShot(10, check)

    for temperature in ("fresh_process", "warm"):
        # First process timing includes the imports and Qt application setup.
        state = {"started": started if not results else perf_counter()}
        QTimer.singleShot(0, check)
        rc = cli.main(["--log-level", "ERROR", "plot", str(path), "--gui"])
        if rc or "error" in state:
            raise RuntimeError(state.get("error", f"Studio returned {rc}"))
        results.append({"run": temperature, **{k: v for k, v in state.items() if k != "started"}})
        application.processEvents()
    return {"imports_s": imports_s, "runs": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeats must be positive.")
    if args.child:
        print(json.dumps(child(args.source)))
        return
    samples = []
    for _ in range(args.repeats):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.measure_studio_startup",
                str(args.source.resolve()),
                "--child",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=150,
        )
        samples.append(json.loads(completed.stdout))
    summary = {
        "source": str(args.source.resolve()),
        "fresh_processes": args.repeats,
        "imports_median_s": median(sample["imports_s"] for sample in samples),
    }
    for index, run in enumerate(("fresh_process", "warm")):
        summary[run] = {
            key: median(sample["runs"][index][key] for sample in samples)
            for key in ("first_paint_s", "first_preview_s")
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
