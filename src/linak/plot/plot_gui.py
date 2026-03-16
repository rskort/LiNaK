"""PySide6 GUI panel for interactive plot settings."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable
from uuid import uuid4

from .plotting import DEFAULT_PLOT_STYLE

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


def _toggle_to_mode(value: bool | None) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "auto"


def _mode_to_toggle(value: str) -> bool | None:
    token = value.strip().lower()
    if token == "on":
        return True
    if token == "off":
        return False
    return None


def _optional_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


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
    return str(value)


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
    on_preview: Callable[[dict[str, Any]], None],
    on_save: Callable[[dict[str, Any]], str],
    on_save_figure: Callable[[dict[str, Any], str], str] | None = None,
) -> None:
    """Open a PySide6 panel that previews and persists plot settings."""
    try:
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSplitter,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PySide6 is unavailable; cannot open GUI plot controls. "
            "Install PySide6 or use CLI plot flags."
        ) from exc

    defaults = DEFAULT_PLOT_STYLE

    class _PlotSettingsWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(title)
            self.resize(980, 760)
            self._saved_signature: str | None = None
            self._title_widgets: list[QWidget] = []
            self._legend_widgets: list[QWidget] = []
            self._grid_widgets: list[QWidget] = []
            self._ticks_widgets: list[QWidget] = []
            self._series_syncing = False
            self._series_active_index = 0
            self._series_labels_data: list[str] = []
            self._series_colors_data: list[str] = []
            self._series_enabled_data: list[bool] = []
            self._series_line_widths_data: list[str] = []
            self._series_markers_data: list[str] = []
            self._suspend_preview_events = False
            self._preview_pixmap: QPixmap | None = None
            self._preview_image_path = (
                Path(tempfile.gettempdir()) / f"linak_preview_{uuid4().hex}.png"
            )
            self._preview_timer = QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.timeout.connect(self._handle_debounced_preview)
            self._status_label = QLabel("Ready.")
            self._build_ui()
            self._suspend_preview_events = True
            self._populate(initial_settings)
            self._suspend_preview_events = False
            self._refresh_widget_states()
            self._bind_live_preview_signals()
            try:
                self._saved_signature = self._signature(self._collect_settings())
            except Exception:
                self._saved_signature = None
            self._update_embedded_preview(interactive=False)

        def _signature(self, settings: dict[str, Any]) -> str:
            return json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)

        def _combo(self, values: tuple[str, ...], *, editable: bool = False) -> QComboBox:
            widget = QComboBox()
            widget.addItems(list(values))
            widget.setEditable(editable)
            return widget

        def _line(self, placeholder: str = "") -> QLineEdit:
            widget = QLineEdit()
            if placeholder:
                widget.setPlaceholderText(placeholder)
            return widget

        def _add_form_row(
            self,
            form: QFormLayout,
            label: str,
            field: QWidget,
        ) -> None:
            form.addRow(label, field)

        def _build_ui(self) -> None:
            root = QWidget(self)
            self.setCentralWidget(root)
            root_layout = QVBoxLayout(root)

            splitter = QSplitter(Qt.Orientation.Horizontal, root)
            root_layout.addWidget(splitter, stretch=1)

            left_panel = QWidget(splitter)
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            splitter.addWidget(left_panel)

            right_panel = QWidget(splitter)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([480, 640])

            tabs = QTabWidget(left_panel)
            left_layout.addWidget(tabs, stretch=1)

            self._tab_general = QWidget()
            self._tab_axes = QWidget()
            self._tab_style = QWidget()
            self._tab_series = QWidget()
            tabs.addTab(self._tab_general, "General")
            tabs.addTab(self._tab_axes, "Axes")
            tabs.addTab(self._tab_style, "Style")
            tabs.addTab(self._tab_series, "Series")

            self._build_general_tab()
            self._build_axes_tab()
            self._build_style_tab()
            self._build_series_tab()

            actions = QHBoxLayout()
            self._preview_button = QPushButton("Refresh Preview")
            self._preview_button.clicked.connect(self._handle_preview)
            actions.addWidget(self._preview_button)

            self._save_figure_button = QPushButton("Save Figure PNG")
            self._save_figure_button.setEnabled(on_save_figure is not None)
            self._save_figure_button.clicked.connect(self._handle_save_figure)
            actions.addWidget(self._save_figure_button)

            self._save_button = QPushButton("Save Settings")
            self._save_button.clicked.connect(self._handle_save)
            actions.addWidget(self._save_button)

            self._reset_button = QPushButton("Reset to Defaults")
            self._reset_button.clicked.connect(self._handle_reset)
            actions.addWidget(self._reset_button)

            self._import_button = QPushButton("Import JSON")
            self._import_button.clicked.connect(self._handle_import_json)
            actions.addWidget(self._import_button)

            self._export_button = QPushButton("Export JSON")
            self._export_button.clicked.connect(self._handle_export_json)
            actions.addWidget(self._export_button)

            self._close_button = QPushButton("Close")
            self._close_button.clicked.connect(self.close)
            actions.addWidget(self._close_button)

            left_layout.addLayout(actions)

            preview_controls = QHBoxLayout()
            self._auto_preview_checkbox = QCheckBox("Auto update")
            self._auto_preview_checkbox.setChecked(on_save_figure is not None)
            self._auto_preview_checkbox.setEnabled(on_save_figure is not None)
            self._auto_preview_checkbox.toggled.connect(self._handle_auto_preview_toggle)
            preview_controls.addWidget(self._auto_preview_checkbox)
            preview_controls.addStretch(1)
            self._preview_status = QLabel("Preview ready.")
            preview_controls.addWidget(self._preview_status)
            right_layout.addLayout(preview_controls)

            self._preview_label = QLabel("Preview will appear here.")
            self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._preview_label.setWordWrap(True)
            self._preview_label.setMinimumSize(420, 320)
            self._preview_label.setStyleSheet("QLabel { border: 1px solid palette(mid); }")
            right_layout.addWidget(self._preview_label, stretch=1)

            status_row = QHBoxLayout()
            status_row.addStretch(1)
            self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            status_row.addWidget(self._status_label)
            right_layout.addLayout(status_row)

            # Keep styling neutral and let Qt/native theme own colors.
            # Explicit color/background rules can become unreadable on mixed system themes.
            self.setStyleSheet(
                "QTabWidget::pane { border: 1px solid palette(mid); }"
                "QGroupBox { border: 1px solid palette(midlight); border-radius: 5px; margin-top: 8px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
                "QPushButton { padding: 6px 10px; }"
            )

        def _build_general_tab(self) -> None:
            form = QFormLayout(self._tab_general)
            self.title_mode = self._combo(("auto", "on", "off"))
            self.title_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.title_text = self._line()
            self.x_label = self._line("Matplotlib mathtext supported, e.g. Distance ($A$)")
            self.y_label = self._line("e.g. Density ($g/cm^3$)")

            self.legend_mode = self._combo(("auto", "on", "off"))
            self.legend_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.legend_title = self._line()
            self.legend_loc = self._combo(_LEGEND_LOCATIONS)

            self.ticks_mode = self._combo(("auto", "on", "off"))
            self.ticks_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.grid_mode = self._combo(("auto", "on", "off"))
            self.grid_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.markers_mode = self._combo(("auto", "on", "off"))

            self._add_form_row(form, "Title", self.title_mode)
            self._add_form_row(form, "Title text", self.title_text)
            self._add_form_row(form, "X label", self.x_label)
            self._add_form_row(form, "Y label", self.y_label)
            math_hint = QLabel("Math labels: use Matplotlib mathtext, e.g. $cm^3$, $\\Delta G$, $\\rho$.")
            math_hint.setWordWrap(True)
            self._add_form_row(form, "", math_hint)
            self._add_form_row(form, "Legend", self.legend_mode)
            self._add_form_row(form, "Legend title", self.legend_title)
            self._add_form_row(form, "Legend location", self.legend_loc)
            self._add_form_row(form, "Ticks", self.ticks_mode)
            self._add_form_row(form, "Grid", self.grid_mode)
            self._add_form_row(form, "Markers", self.markers_mode)

            self._title_widgets = [self.title_text]
            self._legend_widgets = [self.legend_title, self.legend_loc]

        def _build_axes_tab(self) -> None:
            layout = QVBoxLayout(self._tab_axes)
            top_form = QFormLayout()
            self.x_scale = self._combo(("linear", "log", "symlog", "logit"))
            self.y_scale = self._combo(("linear", "log", "symlog", "logit"))
            top_form.addRow("X scale", self.x_scale)
            top_form.addRow("Y scale", self.y_scale)
            layout.addLayout(top_form)

            limits = QGroupBox("Limits")
            limits_layout = QGridLayout(limits)
            self.x_min = self._line()
            self.x_max = self._line()
            self.y_min = self._line()
            self.y_max = self._line()
            limits_layout.addWidget(QLabel("X min"), 0, 0)
            limits_layout.addWidget(self.x_min, 0, 1)
            limits_layout.addWidget(QLabel("X max"), 0, 2)
            limits_layout.addWidget(self.x_max, 0, 3)
            limits_layout.addWidget(QLabel("Y min"), 1, 0)
            limits_layout.addWidget(self.y_min, 1, 1)
            limits_layout.addWidget(QLabel("Y max"), 1, 2)
            limits_layout.addWidget(self.y_max, 1, 3)
            layout.addWidget(limits)

            ticks = QGroupBox("Ticks")
            ticks_layout = QGridLayout(ticks)
            self.x_ticks = self._line("e.g. 0, 1, 2")
            self.y_ticks = self._line("e.g. 0, 5, 10")
            self.x_tick_rotation = self._line("degrees")
            self.y_tick_rotation = self._line("degrees")
            ticks_layout.addWidget(QLabel("X ticks"), 0, 0)
            ticks_layout.addWidget(self.x_ticks, 0, 1)
            ticks_layout.addWidget(QLabel("Y ticks"), 0, 2)
            ticks_layout.addWidget(self.y_ticks, 0, 3)
            ticks_layout.addWidget(QLabel("X rotation"), 1, 0)
            ticks_layout.addWidget(self.x_tick_rotation, 1, 1)
            ticks_layout.addWidget(QLabel("Y rotation"), 1, 2)
            ticks_layout.addWidget(self.y_tick_rotation, 1, 3)
            layout.addWidget(ticks)
            layout.addStretch(1)

            self._ticks_widgets = [
                self.x_ticks,
                self.y_ticks,
                self.x_tick_rotation,
                self.y_tick_rotation,
            ]

        def _build_style_tab(self) -> None:
            layout = QVBoxLayout(self._tab_style)

            metrics = QGroupBox("Figure and Typography")
            metrics_form = QFormLayout(metrics)
            self.fig_width = self._line()
            self.fig_height = self._line()
            self.dpi = self._line()
            self.font_family = self._line()
            metrics_form.addRow("Figure width", self.fig_width)
            metrics_form.addRow("Figure height", self.fig_height)
            metrics_form.addRow("DPI", self.dpi)
            metrics_form.addRow("Font family", self.font_family)
            layout.addWidget(metrics)

            fonts = QGroupBox("Font Sizes")
            fonts_form = QFormLayout(fonts)
            self.title_font = self._line()
            self.label_font = self._line()
            self.tick_font = self._line()
            fonts_form.addRow("Title", self.title_font)
            fonts_form.addRow("Labels", self.label_font)
            fonts_form.addRow("Ticks", self.tick_font)
            layout.addWidget(fonts)

            lines = QGroupBox("Lines and Colors")
            lines_form = QFormLayout(lines)
            self.line_width = self._line()
            self.line_color = self._line()
            lines_form.addRow("Line width", self.line_width)
            lines_form.addRow("Default line color", self.line_color)
            layout.addWidget(lines)

            grid = QGroupBox("Grid")
            grid_form = QFormLayout(grid)
            self.grid_linestyle = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.grid_linewidth = self._line()
            self.grid_alpha = self._line()
            grid_form.addRow("Line style", self.grid_linestyle)
            grid_form.addRow("Line width", self.grid_linewidth)
            grid_form.addRow("Alpha", self.grid_alpha)
            layout.addWidget(grid)
            layout.addStretch(1)

            self._grid_widgets = [self.grid_linestyle, self.grid_linewidth, self.grid_alpha]
            self._title_widgets.append(self.title_font)
            self._ticks_widgets.append(self.tick_font)

        def _build_series_tab(self) -> None:
            layout = QVBoxLayout(self._tab_series)

            selector_row = QHBoxLayout()
            selector_row.addWidget(QLabel("Series"))
            self.series_selector = QComboBox()
            self.series_selector.currentIndexChanged.connect(self._handle_series_selection_change)
            selector_row.addWidget(self.series_selector, stretch=1)
            layout.addLayout(selector_row)

            panel = QGroupBox("Selected Series")
            panel_form = QFormLayout(panel)
            self.series_enabled = self._combo(("on", "off"))
            self.series_enabled.currentTextChanged.connect(self._on_series_editor_changed)
            self.series_label = self._line()
            self.series_label.textChanged.connect(self._on_series_editor_changed)
            self.series_color = self._line("#1f77b4")
            self.series_color.textChanged.connect(self._on_series_editor_changed)
            self.series_line_width = self._line("blank: use global line width")
            self.series_line_width.textChanged.connect(self._on_series_editor_changed)
            self.series_marker = self._combo(("", "o", "s", "^", "v", "d", "x", "+", ".", "*"), editable=True)
            self.series_marker.currentTextChanged.connect(self._on_series_editor_changed)
            panel_form.addRow("Enabled", self.series_enabled)
            panel_form.addRow("Label", self.series_label)
            panel_form.addRow("Color", self.series_color)
            panel_form.addRow("Line width", self.series_line_width)
            panel_form.addRow("Marker", self.series_marker)
            layout.addWidget(panel)

            hint = QLabel(
                "Per-series settings are persisted into the plot profile. "
                "Disable a series to hide it without deleting its metadata."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)

        def _series_count_from_settings(self, settings: dict[str, Any]) -> int:
            candidates = [1]
            raw_count = settings.get("series_count")
            if isinstance(raw_count, int) and raw_count > 0:
                candidates.append(raw_count)
            for key in (
                "series_labels",
                "line_colors",
                "series_enabled",
                "series_line_widths",
                "series_markers",
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
            raw_labels = settings.get("series_labels")
            raw_colors = settings.get("line_colors")
            raw_enabled = settings.get("series_enabled")
            raw_widths = settings.get("series_line_widths")
            raw_markers = settings.get("series_markers")

            self._series_labels_data = []
            self._series_colors_data = []
            self._series_enabled_data = []
            self._series_line_widths_data = []
            self._series_markers_data = []

            for index in range(count):
                fallback_label = f"Series {index + 1}"
                label = fallback_label
                if isinstance(raw_labels, (list, tuple)) and index < len(raw_labels):
                    token = str(raw_labels[index]).strip()
                    if token:
                        label = token
                self._series_labels_data.append(label)

                color = ""
                if isinstance(raw_colors, (list, tuple)) and index < len(raw_colors):
                    color = str(raw_colors[index]).strip()
                self._series_colors_data.append(color)

                enabled = True
                if isinstance(raw_enabled, (list, tuple)) and index < len(raw_enabled):
                    enabled = self._coerce_series_bool(raw_enabled[index], default=True)
                self._series_enabled_data.append(enabled)

                width = ""
                if isinstance(raw_widths, (list, tuple)) and index < len(raw_widths):
                    width = str(raw_widths[index]).strip()
                    if width.lower() == "none":
                        width = ""
                self._series_line_widths_data.append(width)

                marker = ""
                if isinstance(raw_markers, (list, tuple)) and index < len(raw_markers):
                    marker = str(raw_markers[index]).strip()
                    if marker.lower() == "none":
                        marker = ""
                self._series_markers_data.append(marker)

            self._series_syncing = True
            try:
                selected = self._series_active_index if self._series_active_index < count else 0
                self.series_selector.clear()
                for index in range(count):
                    display_label = self._series_labels_data[index] or f"Series {index + 1}"
                    self.series_selector.addItem(f"{index + 1}: {display_label}")
                self.series_selector.setCurrentIndex(selected)
                self._series_active_index = selected
                self._load_series_into_editor(selected)
            finally:
                self._series_syncing = False

        def _load_series_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            self._series_syncing = True
            try:
                self._set_combo_value(self.series_enabled, "on" if self._series_enabled_data[index] else "off")
                self.series_label.setText(self._series_labels_data[index])
                self.series_color.setText(self._series_colors_data[index])
                self.series_line_width.setText(self._series_line_widths_data[index])
                self._set_combo_value(self.series_marker, self._series_markers_data[index])
            finally:
                self._series_syncing = False

        def _persist_series_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            label_value = self.series_label.text().strip()
            self._series_labels_data[index] = label_value or f"Series {index + 1}"
            self._series_colors_data[index] = self.series_color.text().strip()
            self._series_enabled_data[index] = self.series_enabled.currentText().strip().lower() != "off"
            self._series_line_widths_data[index] = self.series_line_width.text().strip()
            self._series_markers_data[index] = self.series_marker.currentText().strip()
            self.series_selector.setItemText(index, f"{index + 1}: {self._series_labels_data[index]}")

        def _handle_series_selection_change(self, index: int) -> None:
            if self._series_syncing:
                return
            self._persist_series_editor(self._series_active_index)
            self._series_active_index = index
            self._load_series_into_editor(index)

        def _on_series_editor_changed(self, *_unused: object) -> None:
            if self._series_syncing:
                return
            self._persist_series_editor(self._series_active_index)
            self._schedule_preview_update()

        def _bind_live_preview_signals(self) -> None:
            line_widgets = (
                self.title_text,
                self.x_label,
                self.y_label,
                self.legend_title,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.x_ticks,
                self.y_ticks,
                self.x_tick_rotation,
                self.y_tick_rotation,
                self.fig_width,
                self.fig_height,
                self.dpi,
                self.font_family,
                self.title_font,
                self.label_font,
                self.tick_font,
                self.line_width,
                self.line_color,
                self.grid_linewidth,
                self.grid_alpha,
            )
            for widget in line_widgets:
                widget.textChanged.connect(self._schedule_preview_update)

            combo_widgets = (
                self.title_mode,
                self.legend_mode,
                self.legend_loc,
                self.ticks_mode,
                self.grid_mode,
                self.markers_mode,
                self.x_scale,
                self.y_scale,
                self.grid_linestyle,
            )
            for widget in combo_widgets:
                widget.currentTextChanged.connect(self._schedule_preview_update)

        def _handle_auto_preview_toggle(self, checked: bool) -> None:
            if not checked:
                self._preview_timer.stop()
                self._preview_status.setText("Auto update paused.")
                return
            self._preview_status.setText("Auto update enabled.")
            self._schedule_preview_update()

        def _schedule_preview_update(self, *_unused: object) -> None:
            if self._suspend_preview_events:
                return
            if not self._auto_preview_checkbox.isChecked():
                return
            self._preview_timer.start(220)

        def _handle_debounced_preview(self) -> None:
            self._update_embedded_preview(interactive=False)

        def _refresh_preview_pixmap(self) -> None:
            if self._preview_pixmap is None or self._preview_pixmap.isNull():
                return
            target_size = self._preview_label.size()
            if target_size.width() < 2 or target_size.height() < 2:
                return
            scaled = self._preview_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)

        def _update_embedded_preview(self, *, interactive: bool) -> bool:
            if on_save_figure is None:
                try:
                    settings = self._collect_settings()
                    on_preview(settings)
                    self._status_label.setText("Preview opened.")
                    self._preview_status.setText("External preview opened.")
                    return True
                except Exception as exc:
                    if interactive:
                        self._report_error("Preview failed", exc)
                    else:
                        self._preview_status.setText(f"Preview paused: {exc}")
                    return False

            try:
                settings = self._collect_settings()
            except Exception as exc:
                if interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_status.setText(f"Preview paused: {exc}")
                return False

            try:
                on_save_figure(settings, str(self._preview_image_path))
                pixmap = QPixmap(str(self._preview_image_path))
                if pixmap.isNull():
                    raise RuntimeError("Could not load rendered preview image.")
                self._preview_pixmap = pixmap
                self._refresh_preview_pixmap()
                self._status_label.setText("Preview updated.")
                self._preview_status.setText("Preview updated.")
                return True
            except Exception as exc:
                if interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_status.setText(f"Preview paused: {exc}")
                return False

        def _set_combo_value(self, widget: QComboBox, value: str) -> None:
            if not value:
                if widget.count() > 0:
                    widget.setCurrentIndex(0)
                return
            index = widget.findText(value)
            if index < 0:
                if widget.isEditable():
                    widget.setEditText(value)
                elif widget.count() > 0:
                    widget.setCurrentIndex(0)
                return
            widget.setCurrentIndex(index)

        def _populate(self, settings: dict[str, Any]) -> None:
            self._set_combo_value(self.title_mode, _toggle_to_mode(settings.get("title_visible")))
            self.title_text.setText(str(settings.get("title") or ""))
            self.x_label.setText(str(settings.get("x_label") or ""))
            self.y_label.setText(str(settings.get("y_label") or ""))

            self._set_combo_value(self.legend_mode, _toggle_to_mode(settings.get("legend")))
            self.legend_title.setText(str(settings.get("legend_title") or ""))
            self._set_combo_value(self.legend_loc, str(settings.get("legend_loc") or "best"))

            self._set_combo_value(self.ticks_mode, _toggle_to_mode(settings.get("ticks")))
            self._set_combo_value(self.grid_mode, _toggle_to_mode(settings.get("grid")))
            self._set_combo_value(self.markers_mode, _toggle_to_mode(settings.get("markers")))

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

            self.fig_width.setText(_extract_figsize_dimension(settings, index=0, fallback=defaults.figure_size[0]))
            self.fig_height.setText(_extract_figsize_dimension(settings, index=1, fallback=defaults.figure_size[1]))
            self.dpi.setText(str(settings.get("dpi") or defaults.dpi))
            self.font_family.setText(str(settings.get("font_family") or ""))
            self.title_font.setText(str(settings.get("title_font_size") or defaults.title_font_size))
            self.label_font.setText(str(settings.get("label_font_size") or defaults.label_font_size))
            self.tick_font.setText(str(settings.get("tick_font_size") or defaults.tick_font_size))
            self.line_width.setText(str(settings.get("line_width") or defaults.line_width))
            self.line_color.setText(str(settings.get("line_color") or ""))
            self._set_combo_value(self.grid_linestyle, str(settings.get("grid_linestyle") or ""))
            self.grid_linewidth.setText(str(settings.get("grid_linewidth") or defaults.grid_linewidth))
            self.grid_alpha.setText(str(settings.get("grid_alpha") or defaults.grid_alpha))
            self._initialize_series_data(settings)

        def _refresh_widget_states(self, *_unused: object) -> None:
            title_enabled = self.title_mode.currentText().strip().lower() != "off"
            legend_enabled = self.legend_mode.currentText().strip().lower() != "off"
            grid_enabled = self.grid_mode.currentText().strip().lower() != "off"
            ticks_enabled = self.ticks_mode.currentText().strip().lower() != "off"

            for widget in self._title_widgets:
                widget.setEnabled(title_enabled)
            for widget in self._legend_widgets:
                widget.setEnabled(legend_enabled)
            for widget in self._grid_widgets:
                widget.setEnabled(grid_enabled)
            for widget in self._ticks_widgets:
                widget.setEnabled(ticks_enabled)

        def _collect_settings(self) -> dict[str, Any]:
            self._persist_series_editor(self._series_active_index)
            fig_width = _optional_float(self.fig_width.text(), field_name="figure width")
            fig_height = _optional_float(self.fig_height.text(), field_name="figure height")
            if (fig_width is None) != (fig_height is None):
                raise ValueError("Figure width and figure height must both be set or both be blank.")
            figsize: list[float] | None = None
            if fig_width is not None and fig_height is not None:
                figsize = [fig_width, fig_height]

            series_labels = [label.strip() or f"Series {index + 1}" for index, label in enumerate(self._series_labels_data)]
            line_colors = [color.strip() for color in self._series_colors_data]
            line_colors_value: list[str] | None = None
            if any(color for color in line_colors):
                if any(not color for color in line_colors):
                    raise ValueError(
                        "When using per-series colors, please set a color value for every series."
                    )
                line_colors_value = [color for color in line_colors]

            series_enabled = [bool(value) for value in self._series_enabled_data]
            series_enabled_value: list[bool] | None = None
            if any(value is False for value in series_enabled):
                series_enabled_value = series_enabled

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
                    raise ValueError(
                        f"Series {index + 1} line width must be a float."
                    ) from exc
            series_line_widths_value: list[float | None] | None = None
            if has_custom_series_width:
                series_line_widths_value = series_line_widths

            series_markers = [marker.strip() for marker in self._series_markers_data]
            series_markers_value: list[str] | None = None
            if any(marker for marker in series_markers):
                series_markers_value = series_markers

            return {
                "title": _optional_text(self.title_text.text()),
                "x_label": _optional_text(self.x_label.text()),
                "y_label": _optional_text(self.y_label.text()),
                "x_min": _optional_float(self.x_min.text(), field_name="x-min"),
                "x_max": _optional_float(self.x_max.text(), field_name="x-max"),
                "y_min": _optional_float(self.y_min.text(), field_name="y-min"),
                "y_max": _optional_float(self.y_max.text(), field_name="y-max"),
                "x_scale": self.x_scale.currentText().strip() or "linear",
                "y_scale": self.y_scale.currentText().strip() or "linear",
                "x_ticks": _optional_float_list(self.x_ticks.text(), field_name="x-ticks"),
                "y_ticks": _optional_float_list(self.y_ticks.text(), field_name="y-ticks"),
                "x_tick_rotation": _optional_float(
                    self.x_tick_rotation.text(), field_name="x-tick-rotation"
                ),
                "y_tick_rotation": _optional_float(
                    self.y_tick_rotation.text(), field_name="y-tick-rotation"
                ),
                "title_visible": _mode_to_toggle(self.title_mode.currentText()),
                "legend": _mode_to_toggle(self.legend_mode.currentText()),
                "grid": _mode_to_toggle(self.grid_mode.currentText()),
                "ticks": _mode_to_toggle(self.ticks_mode.currentText()),
                "markers": _mode_to_toggle(self.markers_mode.currentText()),
                "legend_title": _optional_text(self.legend_title.text()),
                "legend_loc": self.legend_loc.currentText().strip() or "best",
                "figsize": figsize,
                "dpi": _optional_int(self.dpi.text(), field_name="dpi"),
                "font_family": _optional_text(self.font_family.text()),
                "title_font_size": _optional_int(self.title_font.text(), field_name="title-font-size"),
                "label_font_size": _optional_int(self.label_font.text(), field_name="label-font-size"),
                "tick_font_size": _optional_int(self.tick_font.text(), field_name="tick-font-size"),
                "line_width": _optional_float(self.line_width.text(), field_name="line-width"),
                "line_color": _optional_text(self.line_color.text()),
                "line_colors": line_colors_value,
                "series_labels": series_labels,
                "series_enabled": series_enabled_value,
                "series_line_widths": series_line_widths_value,
                "series_markers": series_markers_value,
                "grid_linestyle": _optional_text(self.grid_linestyle.currentText()),
                "grid_linewidth": _optional_float(
                    self.grid_linewidth.text(), field_name="grid-linewidth"
                ),
                "grid_alpha": _optional_float(self.grid_alpha.text(), field_name="grid-alpha"),
            }

        def _report_error(self, title_text: str, exc: Exception) -> None:
            self._status_label.setText(f"{title_text}: {exc}")
            QMessageBox.critical(self, title_text, str(exc))

        def _handle_preview(self) -> None:
            self._preview_timer.stop()
            self._update_embedded_preview(interactive=True)

        def _handle_save(self) -> None:
            try:
                settings = self._collect_settings()
                message = on_save(settings)
                self._status_label.setText(message)
                self._saved_signature = self._signature(settings)
            except Exception as exc:
                self._report_error("Save failed", exc)

        def _handle_save_figure(self) -> None:
            if on_save_figure is None:
                self._status_label.setText("Save-figure action is not available.")
                return
            try:
                settings = self._collect_settings()
                output_path, _selected = QFileDialog.getSaveFileName(
                    self,
                    "Save Figure PNG",
                    "linak_plot.png",
                    "PNG image (*.png)",
                )
                if not output_path:
                    self._status_label.setText("Save figure canceled.")
                    return
                message = on_save_figure(settings, output_path)
                self._status_label.setText(message)
            except Exception as exc:
                self._report_error("Save figure failed", exc)

        def _handle_reset(self) -> None:
            baseline = dict(initial_settings)
            baseline["figsize"] = [defaults.figure_size[0], defaults.figure_size[1]]
            baseline["dpi"] = defaults.dpi
            baseline["title_font_size"] = defaults.title_font_size
            baseline["label_font_size"] = defaults.label_font_size
            baseline["tick_font_size"] = defaults.tick_font_size
            baseline["line_width"] = defaults.line_width
            baseline["line_color"] = defaults.line_color
            baseline["grid"] = defaults.grid
            baseline["grid_linestyle"] = defaults.grid_linestyle
            baseline["grid_linewidth"] = defaults.grid_linewidth
            baseline["grid_alpha"] = defaults.grid_alpha
            baseline.pop("line_colors", None)
            baseline.pop("series_enabled", None)
            baseline.pop("series_line_widths", None)
            baseline.pop("series_markers", None)
            self._suspend_preview_events = True
            try:
                self._populate(baseline)
            finally:
                self._suspend_preview_events = False
            self._refresh_widget_states()
            self._status_label.setText("Values reset to style defaults.")
            self._schedule_preview_update()

        def _handle_import_json(self) -> None:
            path_str, _selected = QFileDialog.getOpenFileName(
                self,
                "Import Plot Settings JSON",
                "",
                "JSON files (*.json)",
            )
            if not path_str:
                return
            try:
                payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON root must be an object with setting keys.")
                merged = dict(initial_settings)
                merged.update(payload)
                self._suspend_preview_events = True
                try:
                    self._populate(merged)
                finally:
                    self._suspend_preview_events = False
                self._refresh_widget_states()
                self._status_label.setText(f"Imported settings from '{Path(path_str).name}'.")
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Import failed", exc)

        def _handle_export_json(self) -> None:
            try:
                settings = self._collect_settings()
                path_str, _selected = QFileDialog.getSaveFileName(
                    self,
                    "Export Plot Settings JSON",
                    "linak_plot_settings.json",
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
            except Exception as exc:
                self._report_error("Export failed", exc)

        def _cleanup_preview_artifacts(self) -> None:
            self._preview_timer.stop()
            try:
                self._preview_image_path.unlink(missing_ok=True)
            except OSError:
                pass

        def resizeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            super().resizeEvent(event)
            self._refresh_preview_pixmap()

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
                    message = on_save(settings)
                    self._status_label.setText(message)
                    self._saved_signature = current_signature
                except Exception as exc:
                    self._report_error("Save failed", exc)
                    event.ignore()
                    return
            self._cleanup_preview_artifacts()
            event.accept()

    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    window = _PlotSettingsWindow()
    window.show()

    if created_app:
        app.exec()
