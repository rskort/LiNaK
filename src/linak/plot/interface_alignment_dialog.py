"""Studio's explicit, reversible interface registration editor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import queue
import threading
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from ..analysis.interface_alignment import (
    InterfaceAlignment,
    InterfaceReference,
    apply_interface_alignment,
    evaluate_interface_alignment,
    lattice_symmetries,
    snapshot_sides,
    suggest_interface_alignment,
    transform_label,
    validate_alignment,
)
from ..cancellation import ComputationCancelled, cancellation_scope


class InterfaceAlignmentDialog(QDialog):
    """One source at a time; only Apply publishes a new settings dictionary."""

    def __init__(self, references: list[dict[str, Any]], alignments: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align interfaces")
        self.resize(1040, 780)
        self.result_alignments = deepcopy(alignments)
        self._references = references
        if not references:
            raise ValueError("No interface reference is available.")
        self._reference: InterfaceReference = references[0]["reference"]
        self._diagnostics = None
        self._suggestions: list[InterfaceAlignment] = []
        self._generation = 0
        self._cancel = threading.Event()
        self._results: queue.Queue = queue.Queue()
        self._closing = False
        self._loading = False
        layout = QVBoxLayout(self)
        note = QLabel(
            "Map the negative-axis interface onto the positive-axis interface. "
            "Preview substrate atoms and terrace heights before applying. "
            "The same transform is used throughout the trajectory."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.source_combo = QComboBox()
        for item in references:
            self.source_combo.addItem(item["label"])
        form.addRow("Source", self.source_combo)
        self.enabled_check = QCheckBox("Enable alignment for this source")
        form.addRow(self.enabled_check)
        self.transform_combo = QComboBox()
        form.addRow("Transform", self.transform_combo)
        self.shift_controls = [self._spin(-1e6, 1e6, 0.1, 5) for _ in range(2)]
        self.shift_labels = [QLabel(), QLabel()]
        for index, control in enumerate(self.shift_controls):
            row = QHBoxLayout()
            minus, plus = QPushButton("−"), QPushButton("+")
            minus.setMaximumWidth(32)
            plus.setMaximumWidth(32)
            minus.clicked.connect(lambda _=False, c=control: c.stepDown())
            plus.clicked.connect(lambda _=False, c=control: c.stepUp())
            row.addWidget(control)
            row.addWidget(minus)
            row.addWidget(plus)
            form.addRow(self.shift_labels[index], row)
        self.fractional_check = QCheckBox("Show fractional translation")
        self.fractional_label = QLabel()
        self.fractional_label.setVisible(False)
        self.fractional_check.toggled.connect(self.fractional_label.setVisible)
        form.addRow(self.fractional_check, self.fractional_label)
        self.depth_spin = self._spin(0.01, 10000, 0.5, 2)
        self.depth_spin.setSuffix(" Å")
        self.tolerance_spin = self._spin(0.001, 10, 0.05, 3)
        self.tolerance_spin.setSuffix(" Å")
        form.addRow("Substrate depth", self.depth_spin)
        form.addRow("Matching tolerance", self.tolerance_spin)
        self.frame_combo = QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._draw_preview)
        form.addRow("Reference frame", self.frame_combo)
        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        form.addRow(self.resolution_label)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.suggest_button = QPushButton("Suggest alignment")
        self.suggest_button.clicked.connect(lambda: self._start_job(True))
        self.cancel_suggestion_button = QPushButton("Cancel suggestion")
        self.cancel_suggestion_button.setVisible(False)
        self.cancel_suggestion_button.clicked.connect(self._cancel_suggestion)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset)
        self.copy_button = QPushButton("Copy alignment")
        self.copy_button.clicked.connect(self._copy)
        self.paste_button = QPushButton("Paste alignment")
        self.paste_button.clicked.connect(self._paste)
        for button in (
            self.suggest_button,
            self.cancel_suggestion_button,
            self.reset_button,
            self.copy_button,
            self.paste_button,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.suggestions_combo = QComboBox()
        self.suggestions_combo.setVisible(False)
        self.suggestions_combo.currentIndexChanged.connect(self._choose_suggestion)
        layout.addWidget(self.suggestions_combo)
        self.diagnostics_label = QLabel("Loading reference geometry…")
        self.diagnostics_label.setWordWrap(True)
        self.diagnostics_label.setTextInteractionFlags(
            self.diagnostics_label.textInteractionFlags()
        )
        layout.addWidget(self.diagnostics_label)
        self.figure = Figure(figsize=(10, 3.1), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(1, 3)
        layout.addWidget(self.canvas, 1)
        self.mismatch_check = QCheckBox("Use despite mismatches")
        self.mismatch_check.toggled.connect(self._refresh_apply)
        layout.addWidget(self.mismatch_check)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(lambda: self._start_job(False))
        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._poll_results)
        self._poll.start()
        self.source_combo.currentIndexChanged.connect(self._load_source)
        self.transform_combo.currentIndexChanged.connect(self._changed)
        for control in [*self.shift_controls, self.depth_spin, self.tolerance_spin]:
            control.valueChanged.connect(self._changed)
        self.enabled_check.toggled.connect(self._refresh_apply)
        self._load_source()

    @staticmethod
    def _spin(lower, upper, step, decimals):
        control = QDoubleSpinBox()
        control.setRange(lower, upper)
        control.setDecimals(decimals)
        control.setSingleStep(step)
        control.setKeyboardTracking(False)
        return control

    def _load_source(self, *_):
        self._loading = True
        self._cancel.set()
        item = self._references[self.source_combo.currentIndex()]
        self._reference = item["reference"]
        self._suggestions = []
        self.frame_combo.clear()
        self.transform_combo.clear()
        for matrix in lattice_symmetries(self._reference):
            self.transform_combo.addItem(transform_label(self._reference, matrix), matrix.tolist())
        for label, index in zip(self.shift_labels, self._reference.lateral_indices):
            label.setText(f"Δ{'XYZ'[index]} (Å)")
        resolution = item.get("resolution")
        self.resolution_label.setText(
            f"Spatial source-bin resolution: {resolution} Å. Fractional shifts use conservative bin overlap; "
            "detail below this resolution cannot be recovered."
            if resolution
            else ""
        )
        self.suggestions_combo.setVisible(False)
        saved = self.result_alignments.get(self._reference.reference_id)
        self._set_alignment(
            InterfaceAlignment.from_dict(saved)
            if saved
            else InterfaceAlignment(self._reference.reference_id)
        )
        self.enabled_check.setChecked(saved is not None)
        self._loading = False
        self._changed(clear_override=False)

    def _alignment(self):
        reference = self._reference
        axes = list(reference.lateral_indices)
        basis = reference.cells[0][np.ix_(axes, axes)]
        shift = np.asarray([control.value() for control in self.shift_controls]) @ np.linalg.inv(
            basis
        )
        return InterfaceAlignment.from_dict(
            {
                "version": 1,
                "reference_id": reference.reference_id,
                "matrix": self.transform_combo.currentData(),
                "translation": (shift % 1).tolist(),
                "depth_A": self.depth_spin.value(),
                "tolerance_A": self.tolerance_spin.value(),
                "allow_mismatches": self.mismatch_check.isChecked(),
            }
        )

    def _set_alignment(self, alignment):
        previous = self._loading
        self._loading = True
        for index in range(self.transform_combo.count()):
            if np.array_equal(self.transform_combo.itemData(index), alignment.matrix):
                self.transform_combo.setCurrentIndex(index)
                break
        axes = list(self._reference.lateral_indices)
        shift = np.asarray(alignment.translation) @ self._reference.cells[0][np.ix_(axes, axes)]
        for control, value in zip(self.shift_controls, shift):
            control.setValue(float(value))
        self.depth_spin.setValue(alignment.depth_A)
        self.tolerance_spin.setValue(alignment.tolerance_A)
        self.mismatch_check.setChecked(alignment.allow_mismatches)
        self._loading = previous

    def _changed(self, *_, clear_override=True, from_suggestion=False):
        if self._loading or self._closing:
            return
        self._generation += 1
        self._cancel.set()
        self._diagnostics = None
        if not from_suggestion:
            self._suggestions = []
            self.suggestions_combo.setVisible(False)
        if clear_override:
            self.mismatch_check.setChecked(False)
        alignment = self._alignment()
        self.fractional_label.setText(
            f"a: {alignment.translation[0]:.6f}, b: {alignment.translation[1]:.6f}"
        )
        self._refresh_apply()
        self._debounce.start()

    def _start_job(self, suggest=False):
        if self._closing:
            return
        self._debounce.stop()
        self._cancel.set()
        self._cancel = threading.Event()
        event = self._cancel
        self._generation += 1
        generation = self._generation
        reference, alignment = self._reference, self._alignment()
        self.suggest_button.setEnabled(False)
        self.cancel_suggestion_button.setVisible(suggest)
        self.diagnostics_label.setText(
            "Searching lattice-compatible alignments…"
            if suggest
            else "Evaluating substrate correspondence…"
        )
        self._diagnostics = None
        self._refresh_apply()
        results = self._results

        def work():
            try:
                with cancellation_scope(event.is_set):
                    loaded = reference.materialize()
                    value = (
                        suggest_interface_alignment(
                            loaded, depth_A=alignment.depth_A, tolerance_A=alignment.tolerance_A
                        )
                        if suggest
                        else evaluate_interface_alignment(loaded, alignment)
                    )
                results.put((generation, suggest, loaded, value, None))
            except ComputationCancelled:
                pass
            except Exception as exc:
                results.put((generation, suggest, reference, None, str(exc)))

        threading.Thread(target=work, name="linak-interface-alignment", daemon=True).start()

    def _poll_results(self):
        while not self._results.empty():
            generation, suggested, reference, value, error = self._results.get_nowait()
            if generation != self._generation or self._closing:
                continue
            self.suggest_button.setEnabled(True)
            self.cancel_suggestion_button.setVisible(False)
            if error:
                self.diagnostics_label.setText(error)
                self._refresh_apply()
                continue
            self._reference = reference
            if self.frame_combo.count() != len(reference.snapshots):
                self.frame_combo.blockSignals(True)
                self.frame_combo.clear()
                for snapshot in reference.snapshots:
                    self.frame_combo.addItem(str(snapshot.frame_index))
                self.frame_combo.blockSignals(False)
            if suggested:
                self._suggestions = value
                self.suggestions_combo.blockSignals(True)
                self.suggestions_combo.clear()
                for alignment in value:
                    diagnostics = alignment.diagnostics
                    coverage = min(diagnostics["top_coverage"] + diagnostics["bottom_coverage"])
                    label = transform_label(reference, np.asarray(alignment.matrix))
                    self.suggestions_combo.addItem(
                        f"{label}; shift {alignment.translation[0]:.4f}, {alignment.translation[1]:.4f}; match {coverage:.1%}"
                    )
                self.suggestions_combo.blockSignals(False)
                self.suggestions_combo.setVisible(len(value) > 1)
                if value:
                    self._choose_suggestion(0)
                else:
                    self.diagnostics_label.setText(
                        "No matching substrate candidates were found. Adjust the depth or tolerance."
                    )
            else:
                self._diagnostics = value
                coverage_top = min(value.top_coverage)
                coverage_bottom = min(value.bottom_coverage)
                rmsds = [x for x in value.rmsd_A if x is not None]
                rmsd_text = (
                    f"RMS displacement {min(rmsds):.3f}–{max(rmsds):.3f} Å"
                    if rmsds
                    else "No matched atoms"
                )
                ambiguous = bool(
                    self._suggestions and (self._suggestions[0].diagnostics or {}).get("ambiguous")
                )
                prefix = (
                    "Several alignments fit similarly. Review the alternatives. "
                    if ambiguous
                    else ""
                )
                self.diagnostics_label.setText(
                    prefix
                    + f"Minimum match: positive {coverage_top:.1%}, negative {coverage_bottom:.1%}. "
                    f"{rmsd_text} across {len(value.rmsd_A)} reference frame(s). "
                    + (
                        "Good geometric match."
                        if value.good_match
                        else "The surfaces retain geometric mismatches."
                    )
                )
                self._draw_preview()
                self._refresh_apply()

    def _choose_suggestion(self, index):
        if index < 0 or index >= len(self._suggestions):
            return
        self._set_alignment(self._suggestions[index])
        self.enabled_check.setChecked(True)
        self._changed(from_suggestion=True)

    def _draw_preview(self, *_):
        if not self._reference or not self._reference.snapshots or self._diagnostics is None:
            return
        index = max(0, self.frame_combo.currentIndex())
        snapshot = self._reference.snapshots[index]
        alignment = self._alignment()
        (top, _, _), (bottom, _, _) = snapshot_sides(self._reference, snapshot, alignment.depth_A)
        bottom = apply_interface_alignment(
            bottom, snapshot.cell, snapshot.origin, self._reference.axis, alignment
        )
        axes = self._reference.lateral_indices
        normal = "xyz".index(self._reference.axis)
        ti, bi = self._diagnostics.matched_indices[index]
        for plot, (x, y) in zip(
            self.axes, ((axes[0], axes[1]), (axes[0], normal), (axes[1], normal))
        ):
            plot.clear()
            for points, matched, color, label, marker in (
                (top, ti, "#2468bd", "Positive", "o"),
                (bottom, bi, "#d57922", "Negative (mapped)", "x"),
            ):
                plot.scatter(
                    points[:, x], points[:, y], s=20, color=color, marker=marker, label=label
                )
                unmatched = np.ones(len(points), bool)
                unmatched[matched] = False
                plot.scatter(
                    points[unmatched, x],
                    points[unmatched, y],
                    s=52,
                    facecolors="none",
                    edgecolors="#ba2636",
                    linewidths=1,
                )
            plot.set_xlabel(f"{'XYZ'[x]} (Å)")
            plot.set_ylabel("Outward height (Å)" if y == normal else f"{'XYZ'[y]} (Å)")
            plot.grid(alpha=0.2)
        self.axes[0].set_aspect("equal", adjustable="datalim")
        self.axes[0].legend(fontsize=8)
        self.canvas.draw_idle()

    def _refresh_apply(self, *_):
        good = self._diagnostics is not None and self._diagnostics.good_match
        self.mismatch_check.setVisible(self._diagnostics is not None and not good)
        self.apply_button.setEnabled(
            not self.enabled_check.isChecked()
            or (self._diagnostics is not None and (good or self.mismatch_check.isChecked()))
        )

    def _apply(self):
        if not self.enabled_check.isChecked():
            self.result_alignments.pop(self._reference.reference_id, None)
        else:
            if self._diagnostics is None or (
                not self._diagnostics.good_match and not self.mismatch_check.isChecked()
            ):
                return
            alignment = replace(self._alignment(), diagnostics=self._diagnostics.to_dict())
            self.result_alignments[alignment.reference_id] = alignment.to_dict()
        self.accept()

    def _reset(self):
        self._suggestions = []
        self.suggestions_combo.setVisible(False)
        self._set_alignment(InterfaceAlignment(self._reference.reference_id))
        self.enabled_check.setChecked(False)
        self._changed()

    def _copy(self):
        QApplication.clipboard().setText(json.dumps(self._alignment().to_dict(), allow_nan=False))

    def _paste(self):
        try:
            alignment = InterfaceAlignment.from_dict(json.loads(QApplication.clipboard().text()))
            validate_alignment(self._reference, alignment)
        except (ValueError, TypeError) as exc:
            self.diagnostics_label.setText(f"Cannot paste alignment: {exc}")
            return
        self._set_alignment(alignment)
        self.enabled_check.setChecked(True)
        self._changed()

    def _cancel_suggestion(self):
        self._cancel.set()
        self._generation += 1
        self.cancel_suggestion_button.setVisible(False)
        self.suggest_button.setEnabled(True)
        self._changed()

    def done(self, result):
        self._closing = True
        self._cancel.set()
        self._debounce.stop()
        self._poll.stop()
        super().done(result)
