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
_BIN_REDUCERS = ("mean", "median", "sum", "min", "max")
_NORMALIZATION_MODES = ("none", "max", "area", "value_at_x", "factor")
_GRID_AXES = ("auto", "both", "x", "y")
_GRID_WHICH = ("auto", "major", "minor", "both")
_TICK_AXES = ("auto", "both", "x", "y")
_TICK_DIRECTIONS = ("auto", "out", "in", "inout")
_MINOR_TICKS_MODES = ("auto", "on", "off")
_SERIES_SPECIFIC_SETTINGS = frozenset(
    {
        "series_labels",
        "line_colors",
        "series_enabled",
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
    }
)


def _without_series_specific_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without per-series plot controls."""
    return {key: value for key, value in settings.items() if key not in _SERIES_SPECIFIC_SETTINGS}


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


def _format_json_block(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except TypeError:
        return str(value)


def _optional_json_dict(value: str, *, field_name: str) -> dict[str, Any] | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed


def _extract_dict_value(settings: dict[str, Any], *, key: str, nested_key: str) -> Any:
    raw = settings.get(key)
    if not isinstance(raw, dict):
        return None
    return raw.get(nested_key)


def _extract_dict_text(settings: dict[str, Any], *, key: str, nested_key: str) -> str:
    value = _extract_dict_value(settings, key=key, nested_key=nested_key)
    if value is None:
        return ""
    return str(value)


def _extract_dict_mode(settings: dict[str, Any], *, key: str, nested_key: str) -> str:
    value = _extract_dict_value(settings, key=key, nested_key=nested_key)
    if isinstance(value, bool):
        return _toggle_to_mode(value)
    return "auto"


def _figure_filetype_filters() -> tuple[str, str]:
    filetypes: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        try:
            supported = fig.canvas.get_supported_filetypes()
        finally:
            plt.close(fig)
        filetypes = {
            str(ext).strip().lower(): str(label).strip() or f"{str(ext).upper()} file"
            for ext, label in supported.items()
            if str(ext).strip()
        }
    except Exception:
        filetypes = {}

    required = {
        "png": "PNG image",
        "jpg": "JPEG image",
        "jpeg": "JPEG image",
        "svg": "SVG image",
    }
    merged = dict(filetypes)
    for ext, label in required.items():
        merged.setdefault(ext, label)

    if not merged:
        merged = dict(required)

    ordered = dict(sorted(merged.items(), key=lambda item: item[0]))
    extensions = list(ordered.keys())
    all_patterns = " ".join(f"*.{ext}" for ext in extensions)
    filters = [f"All supported ({all_patterns})"]
    for ext, label in ordered.items():
        filters.append(f"{label} (*.{ext})")

    default_name = "linak_plot.png" if "png" in ordered else f"linak_plot.{extensions[0]}"
    return ";;".join(filters), default_name


def _default_gui_artwork_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "linak_gui_banner.svg"


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
    on_import_hdf5: Callable[[str], dict[str, Any]] | None = None,
) -> None:
    """Open a PySide6 panel that previews and persists plot settings."""
    try:
        from PySide6.QtCore import QEvent, QTimer, Qt
        from PySide6.QtGui import QColor, QIcon, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QColorDialog,
            QFrame,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
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
            self._title_rows: list[tuple[QFormLayout, QWidget]] = []
            self._legend_rows: list[tuple[QFormLayout, QWidget]] = []
            self._ticks_rows: list[tuple[QFormLayout, QWidget]] = []
            self._grid_rows: list[tuple[QFormLayout, QWidget]] = []
            self._x_bin_reducer_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_value_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_x_ref_row: tuple[QFormLayout, QWidget] | None = None
            self._axes_ticks_group: QGroupBox | None = None
            self._tick_appearance_group: QGroupBox | None = None
            self._grid_group: QGroupBox | None = None
            self._series_syncing = False
            self._series_active_index = 0
            self._series_labels_data: list[str] = []
            self._series_colors_data: list[str] = []
            self._series_enabled_data: list[bool] = []
            self._series_line_widths_data: list[str] = []
            self._series_markers_data: list[str] = []
            self._series_line_kwargs_data: list[str] = []
            self._normalization_syncing = False
            self._normalization_active_index = 0
            self._series_normalization_modes_data: list[str] = []
            self._series_normalization_values_data: list[str] = []
            self._series_normalization_x_refs_data: list[str] = []
            self._suspend_preview_events = False
            self._preview_pixmap: QPixmap | None = None
            self._preview_zoom_factor = 1.0
            self._preview_frame: QFrame | None = None
            self._preview_scroll: QScrollArea | None = None
            self._preview_label: QLabel | None = None
            self._gui_artwork_path = _default_gui_artwork_path()
            self._figure_save_filters, self._figure_default_name = _figure_filetype_filters()
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

        def _color_field(self, *, placeholder: str = "") -> tuple[QWidget, QLineEdit]:
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            field = self._line(placeholder)
            pick_button = QPushButton("Pick")

            def _pick_color() -> None:
                initial = QColor(field.text().strip())
                if not initial.isValid():
                    initial = QColor("white")
                selected = QColorDialog.getColor(initial, self, "Select color")
                if not selected.isValid():
                    return
                field.setText(str(selected.name()))

            pick_button.clicked.connect(_pick_color)
            row.addWidget(field, stretch=1)
            row.addWidget(pick_button)
            return container, field

        def _set_form_row_visible(
            self,
            form: QFormLayout,
            field: QWidget,
            visible: bool,
        ) -> None:
            try:
                form.setRowVisible(field, visible)
            except Exception:
                label = form.labelForField(field)
                if label is not None:
                    label.setVisible(visible)
                field.setVisible(visible)

        def _set_rows_visible(
            self,
            rows: list[tuple[QFormLayout, QWidget]],
            visible: bool,
        ) -> None:
            for form, field in rows:
                self._set_form_row_visible(form, field, visible)

        def _add_form_row(
            self,
            form: QFormLayout,
            label: str,
            field: QWidget,
        ) -> None:
            form.addRow(label, field)

        def _make_scrollable_tab(self, tab: QWidget) -> QWidget:
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea(tab)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            content = QWidget(scroll)
            scroll.setWidget(content)
            tab_layout.addWidget(scroll)
            return content

        def _apply_window_icon(self) -> None:
            if not self._gui_artwork_path.exists():
                return
            if self._gui_artwork_path.suffix.lower() == ".svg":
                try:
                    payload = self._gui_artwork_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    return
                # Guard against complex editor-specific SVG payloads that Qt may reject.
                if "sodipodi:namedview" in payload or "inkscape:version" in payload:
                    return
            icon = QIcon(str(self._gui_artwork_path))
            if icon.isNull():
                return
            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)

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
            self._apply_window_icon()

            tabs = QTabWidget(left_panel)
            left_layout.addWidget(tabs, stretch=1)

            self._tab_general = QWidget()
            self._tab_axes = QWidget()
            self._tab_style = QWidget()
            self._tab_series = QWidget()
            self._tab_data = QWidget()
            self._tab_advanced = QWidget()
            tabs.addTab(self._tab_general, "General")
            tabs.addTab(self._tab_axes, "Axes")
            tabs.addTab(self._tab_style, "Style")
            tabs.addTab(self._tab_series, "Series")
            tabs.addTab(self._tab_data, "Data")
            tabs.addTab(self._tab_advanced, "Advanced")
            self._tab_general_content = self._make_scrollable_tab(self._tab_general)
            self._tab_axes_content = self._make_scrollable_tab(self._tab_axes)
            self._tab_style_content = self._make_scrollable_tab(self._tab_style)
            self._tab_series_content = self._make_scrollable_tab(self._tab_series)
            self._tab_data_content = self._make_scrollable_tab(self._tab_data)
            self._tab_advanced_content = self._make_scrollable_tab(self._tab_advanced)

            self._build_general_tab()
            self._build_axes_tab()
            self._build_style_tab()
            self._build_series_tab()
            self._build_data_tab()
            self._build_advanced_tab()

            actions = QHBoxLayout()
            self._preview_button = QPushButton("Refresh Preview")
            self._preview_button.clicked.connect(self._handle_preview)
            actions.addWidget(self._preview_button)

            self._save_figure_button = QPushButton("Save Figure")
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

            self._preview_frame = QFrame(right_panel)
            self._preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
            self._preview_frame.installEventFilter(self)
            preview_frame_layout = QVBoxLayout(self._preview_frame)
            preview_frame_layout.setContentsMargins(6, 6, 6, 6)
            self._preview_scroll = QScrollArea(self._preview_frame)
            self._preview_scroll.setWidgetResizable(False)
            self._preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self._preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._preview_scroll.viewport().installEventFilter(self)

            self._preview_label = QLabel("Preview will appear here.\nUse mouse wheel to zoom.")
            self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._preview_label.setWordWrap(True)
            self._preview_label.setMinimumSize(1, 1)
            self._preview_label.installEventFilter(self)
            self._preview_scroll.setWidget(self._preview_label)
            preview_frame_layout.addWidget(self._preview_scroll, stretch=1)
            right_layout.addWidget(self._preview_frame, stretch=1)

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
            form = QFormLayout(self._tab_general_content)
            self.title_mode = self._combo(("auto", "on", "off"))
            self.title_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.title_text = self._line()
            self.x_label = self._line("Matplotlib mathtext supported, e.g. Distance ($A$)")
            self.y_label = self._line("e.g. Density ($g/cm^3$)")

            self.legend_mode = self._combo(("auto", "on", "off"))
            self.legend_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.legend_title = self._line()
            self.legend_loc = self._combo(_LEGEND_LOCATIONS)
            self.legend_frame_mode = self._combo(("auto", "on", "off"))
            self.legend_columns = self._line("1")

            self.ticks_mode = self._combo(("auto", "on", "off"))
            self.ticks_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.ticks_axis = self._combo(_TICK_AXES)
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
            self._add_form_row(form, "Legend frame", self.legend_frame_mode)
            self._add_form_row(form, "Legend columns", self.legend_columns)
            self._add_form_row(form, "Ticks", self.ticks_mode)
            self._add_form_row(form, "Tick axis", self.ticks_axis)
            self._add_form_row(form, "Grid", self.grid_mode)
            self._add_form_row(form, "Markers", self.markers_mode)

            self._title_rows = [(form, self.title_text)]
            self._legend_rows = [
                (form, self.legend_title),
                (form, self.legend_loc),
                (form, self.legend_frame_mode),
                (form, self.legend_columns),
            ]
            self._ticks_rows = [(form, self.ticks_axis)]

        def _build_axes_tab(self) -> None:
            layout = QVBoxLayout(self._tab_axes_content)
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
            self._axes_ticks_group = ticks
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

            margins = QGroupBox("Margins")
            margins_form = QFormLayout(margins)
            self.x_margin = self._line("e.g. 0.02")
            self.y_margin = self._line("e.g. 0.05")
            margins_form.addRow("X margin", self.x_margin)
            margins_form.addRow("Y margin", self.y_margin)
            layout.addWidget(margins)
            layout.addStretch(1)

        def _build_style_tab(self) -> None:
            layout = QVBoxLayout(self._tab_style_content)

            metrics = QGroupBox("Figure and Typography")
            metrics_form = QFormLayout(metrics)
            self.fig_width = self._line()
            self.fig_height = self._line()
            self.dpi = self._line()
            self.font_family = self._line()
            figure_facecolor_row, self.figure_facecolor = self._color_field(placeholder="#ffffff")
            self.transparent_mode = self._combo(("auto", "on", "off"))
            metrics_form.addRow("Figure width", self.fig_width)
            metrics_form.addRow("Figure height", self.fig_height)
            metrics_form.addRow("DPI", self.dpi)
            metrics_form.addRow("Font family", self.font_family)
            metrics_form.addRow("Figure facecolor", figure_facecolor_row)
            metrics_form.addRow("Transparent save", self.transparent_mode)
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
            line_color_row, self.line_color = self._color_field(placeholder="#1f77b4")
            self.line_style = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.line_alpha = self._line("0.0 - 1.0")
            self.marker_size = self._line("e.g. 5")
            lines_form.addRow("Line width", self.line_width)
            lines_form.addRow("Default line color", line_color_row)
            lines_form.addRow("Line style", self.line_style)
            lines_form.addRow("Line alpha", self.line_alpha)
            lines_form.addRow("Marker size", self.marker_size)
            layout.addWidget(lines)

            grid = QGroupBox("Grid")
            self._grid_group = grid
            grid_form = QFormLayout(grid)
            self.grid_linestyle = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.grid_linewidth = self._line()
            self.grid_alpha = self._line()
            grid_color_row, self.grid_color = self._color_field(placeholder="#dddddd")
            self.grid_axis = self._combo(_GRID_AXES)
            self.grid_which = self._combo(_GRID_WHICH)
            grid_form.addRow("Line style", self.grid_linestyle)
            grid_form.addRow("Line width", self.grid_linewidth)
            grid_form.addRow("Alpha", self.grid_alpha)
            grid_form.addRow("Color", grid_color_row)
            grid_form.addRow("Axis", self.grid_axis)
            grid_form.addRow("Lines", self.grid_which)
            layout.addWidget(grid)

            tick_appearance = QGroupBox("Tick Appearance")
            self._tick_appearance_group = tick_appearance
            tick_appearance_form = QFormLayout(tick_appearance)
            self.tick_direction = self._combo(_TICK_DIRECTIONS)
            self.tick_length = self._line("points")
            self.tick_width = self._line("points")
            self.minor_ticks_mode = self._combo(_MINOR_TICKS_MODES)
            tick_appearance_form.addRow("Direction", self.tick_direction)
            tick_appearance_form.addRow("Length", self.tick_length)
            tick_appearance_form.addRow("Width", self.tick_width)
            tick_appearance_form.addRow("Minor ticks", self.minor_ticks_mode)
            layout.addWidget(tick_appearance)
            layout.addStretch(1)

            self._grid_rows = [
                (grid_form, self.grid_linestyle),
                (grid_form, self.grid_linewidth),
                (grid_form, self.grid_alpha),
                (grid_form, grid_color_row),
                (grid_form, self.grid_axis),
                (grid_form, self.grid_which),
            ]
            self._title_rows.append((fonts_form, self.title_font))
            self._ticks_rows.append((fonts_form, self.tick_font))

        def _build_series_tab(self) -> None:
            layout = QVBoxLayout(self._tab_series_content)

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
            series_color_row, self.series_color = self._color_field(placeholder="#1f77b4")
            self.series_color.textChanged.connect(self._on_series_editor_changed)
            self.series_line_width = self._line("blank: use global line width")
            self.series_line_width.textChanged.connect(self._on_series_editor_changed)
            self.series_marker = self._combo(("", "o", "s", "^", "v", "d", "x", "+", ".", "*"), editable=True)
            self.series_marker.currentTextChanged.connect(self._on_series_editor_changed)
            self.series_line_kwargs_json = QPlainTextEdit()
            self.series_line_kwargs_json.setPlaceholderText('{"linestyle": "--", "alpha": 0.8}')
            self.series_line_kwargs_json.setFixedHeight(84)
            self.series_line_kwargs_json.textChanged.connect(self._on_series_editor_changed)
            panel_form.addRow("Enabled", self.series_enabled)
            panel_form.addRow("Label", self.series_label)
            panel_form.addRow("Color", series_color_row)
            panel_form.addRow("Line width", self.series_line_width)
            panel_form.addRow("Marker", self.series_marker)
            panel_form.addRow("Extra line kwargs (JSON)", self.series_line_kwargs_json)
            layout.addWidget(panel)

            hint = QLabel(
                "Per-series settings are persisted into the plot profile. "
                "Disable a series to hide it without deleting its metadata."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)

        def _build_data_tab(self) -> None:
            layout = QVBoxLayout(self._tab_data_content)

            binning = QGroupBox("X Rebinning (plot-only)")
            binning_form = QFormLayout(binning)
            self.x_bin_width = self._line("Leave blank to disable")
            self.x_bin_width.textChanged.connect(self._refresh_widget_states)
            self.x_bin_reducer = self._combo(_BIN_REDUCERS)
            binning_form.addRow("Bin width", self.x_bin_width)
            binning_form.addRow("Reducer", self.x_bin_reducer)
            self._x_bin_reducer_row = (binning_form, self.x_bin_reducer)
            layout.addWidget(binning)

            normalize_group = QGroupBox("Per-Series Normalization")
            normalize_layout = QVBoxLayout(normalize_group)
            selector_row = QHBoxLayout()
            selector_row.addWidget(QLabel("Series"))
            self.norm_series_selector = QComboBox()
            self.norm_series_selector.currentIndexChanged.connect(self._handle_normalization_selection_change)
            selector_row.addWidget(self.norm_series_selector, stretch=1)
            normalize_layout.addLayout(selector_row)

            normalize_form = QFormLayout()
            self.norm_mode = self._combo(_NORMALIZATION_MODES)
            self.norm_mode.currentTextChanged.connect(self._on_normalization_editor_changed)
            self.norm_value = self._line("Target value (required unless mode=none)")
            self.norm_value.textChanged.connect(self._on_normalization_editor_changed)
            self.norm_x_ref = self._line("Reference x (required for value_at_x)")
            self.norm_x_ref.textChanged.connect(self._on_normalization_editor_changed)
            normalize_form.addRow("Mode", self.norm_mode)
            normalize_form.addRow("Target", self.norm_value)
            normalize_form.addRow("Reference x", self.norm_x_ref)
            self._norm_value_row = (normalize_form, self.norm_value)
            self._norm_x_ref_row = (normalize_form, self.norm_x_ref)
            normalize_layout.addLayout(normalize_form)

            self.norm_copy_to_all = QPushButton("Copy Current Settings To All Series")
            self.norm_copy_to_all.clicked.connect(self._copy_normalization_to_all)
            normalize_layout.addWidget(self.norm_copy_to_all)

            self.normalization_warning = QLabel("")
            self.normalization_warning.setWordWrap(True)
            normalize_layout.addWidget(self.normalization_warning)

            hint = QLabel(
                "Normalization and x rebinning affect only the displayed figure. "
                "Stored HDF5 datasets remain unchanged."
            )
            hint.setWordWrap(True)
            normalize_layout.addWidget(hint)
            layout.addWidget(normalize_group)
            layout.addStretch(1)

        def _build_advanced_tab(self) -> None:
            layout = QVBoxLayout(self._tab_advanced_content)

            def _json_editor(placeholder: str) -> QPlainTextEdit:
                editor = QPlainTextEdit()
                editor.setPlaceholderText(placeholder)
                editor.setFixedHeight(84)
                editor.textChanged.connect(self._schedule_preview_update)
                return editor

            rc_group = QGroupBox("Matplotlib rcParams")
            rc_form = QFormLayout(rc_group)
            self.matplotlib_rc_json = _json_editor('{"axes.facecolor": "#f8f8f8"}')
            rc_form.addRow("rcParams (JSON object)", self.matplotlib_rc_json)
            layout.addWidget(rc_group)

            render_group = QGroupBox("Figure / Axes / Layout")
            render_form = QFormLayout(render_group)
            self.figure_kwargs_json = _json_editor('{"facecolor": "white"}')
            self.axes_kwargs_json = _json_editor('{"xmargin": 0.02, "ymargin": 0.05}')
            self.tight_layout_kwargs_json = _json_editor('{"pad": 0.6}')
            self.savefig_kwargs_json = _json_editor('{"transparent": false}')
            render_form.addRow("Figure kwargs", self.figure_kwargs_json)
            render_form.addRow("Axes kwargs", self.axes_kwargs_json)
            render_form.addRow("tight_layout kwargs", self.tight_layout_kwargs_json)
            render_form.addRow("savefig kwargs", self.savefig_kwargs_json)
            layout.addWidget(render_group)

            style_group = QGroupBox("Raw Matplotlib kwargs")
            style_form = QFormLayout(style_group)
            self.legend_kwargs_json = _json_editor('{"frameon": true}')
            self.grid_kwargs_json = _json_editor('{"color": "#dddddd"}')
            self.tick_params_kwargs_json = _json_editor('{"direction": "out"}')
            self.line_kwargs_json = _json_editor('{"linestyle": "-", "alpha": 1.0}')
            style_form.addRow("Legend kwargs", self.legend_kwargs_json)
            style_form.addRow("Grid kwargs", self.grid_kwargs_json)
            style_form.addRow("Tick params kwargs", self.tick_params_kwargs_json)
            style_form.addRow("Global line kwargs", self.line_kwargs_json)
            layout.addWidget(style_group)

            hint = QLabel(
                "Advanced JSON fields map directly onto Matplotlib API kwargs. "
                "Use JSON objects; invalid keys/values are reported at preview time."
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
                "series_line_kwargs",
                "series_normalization_modes",
                "series_normalization_values",
                "series_normalization_x_refs",
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
            raw_line_kwargs = settings.get("series_line_kwargs")

            self._series_labels_data = []
            self._series_colors_data = []
            self._series_enabled_data = []
            self._series_line_widths_data = []
            self._series_markers_data = []
            self._series_line_kwargs_data = []

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

                line_kwargs_text = ""
                if isinstance(raw_line_kwargs, (list, tuple)) and index < len(raw_line_kwargs):
                    value = raw_line_kwargs[index]
                    if isinstance(value, dict):
                        line_kwargs_text = _format_json_block(value)
                self._series_line_kwargs_data.append(line_kwargs_text)

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
                self._sync_normalization_selector_labels()
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
                self.series_line_kwargs_json.setPlainText(self._series_line_kwargs_data[index])
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
            self._series_line_kwargs_data[index] = self.series_line_kwargs_json.toPlainText().strip()
            label_text = f"{index + 1}: {self._series_labels_data[index]}"
            self.series_selector.setItemText(index, label_text)
            if self.norm_series_selector.count() > index:
                self.norm_series_selector.setItemText(index, label_text)

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

        def _sync_normalization_selector_labels(self) -> None:
            if not hasattr(self, "norm_series_selector"):
                return
            self._normalization_syncing = True
            try:
                selected = (
                    self._normalization_active_index
                    if self._normalization_active_index < len(self._series_labels_data)
                    else 0
                )
                self.norm_series_selector.clear()
                for index, label in enumerate(self._series_labels_data):
                    self.norm_series_selector.addItem(f"{index + 1}: {label}")
                self.norm_series_selector.setCurrentIndex(selected)
                self._normalization_active_index = selected
            finally:
                self._normalization_syncing = False

        def _initialize_normalization_data(self, settings: dict[str, Any]) -> None:
            count = len(self._series_labels_data)
            raw_modes = settings.get("series_normalization_modes")
            raw_values = settings.get("series_normalization_values")
            raw_x_refs = settings.get("series_normalization_x_refs")

            self._series_normalization_modes_data = []
            self._series_normalization_values_data = []
            self._series_normalization_x_refs_data = []

            for index in range(count):
                mode = "none"
                if isinstance(raw_modes, (list, tuple)) and index < len(raw_modes):
                    token = str(raw_modes[index]).strip().lower()
                    if token in _NORMALIZATION_MODES:
                        mode = token
                value = ""
                if isinstance(raw_values, (list, tuple)) and index < len(raw_values):
                    raw_value = raw_values[index]
                    if raw_value is not None:
                        value = str(raw_value).strip()
                x_ref = ""
                if isinstance(raw_x_refs, (list, tuple)) and index < len(raw_x_refs):
                    raw_x_ref = raw_x_refs[index]
                    if raw_x_ref is not None:
                        x_ref = str(raw_x_ref).strip()
                self._series_normalization_modes_data.append(mode)
                self._series_normalization_values_data.append(value)
                self._series_normalization_x_refs_data.append(x_ref)

            self._sync_normalization_selector_labels()
            self._normalization_syncing = True
            try:
                self._load_normalization_into_editor(self._normalization_active_index)
            finally:
                self._normalization_syncing = False
            self._update_normalization_warning()

        def _load_normalization_into_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_normalization_modes_data):
                return
            self._normalization_syncing = True
            try:
                self._set_combo_value(self.norm_mode, self._series_normalization_modes_data[index])
                self.norm_value.setText(self._series_normalization_values_data[index])
                self.norm_x_ref.setText(self._series_normalization_x_refs_data[index])
            finally:
                self._normalization_syncing = False

        def _persist_normalization_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_normalization_modes_data):
                return
            mode = self.norm_mode.currentText().strip().lower()
            if mode not in _NORMALIZATION_MODES:
                mode = "none"
            self._series_normalization_modes_data[index] = mode
            self._series_normalization_values_data[index] = self.norm_value.text().strip()
            self._series_normalization_x_refs_data[index] = self.norm_x_ref.text().strip()
            self._update_normalization_warning()

        def _handle_normalization_selection_change(self, index: int) -> None:
            if self._normalization_syncing:
                return
            self._persist_normalization_editor(self._normalization_active_index)
            self._normalization_active_index = index
            self._load_normalization_into_editor(index)
            self._refresh_widget_states()

        def _on_normalization_editor_changed(self, *_unused: object) -> None:
            if self._normalization_syncing:
                return
            self._persist_normalization_editor(self._normalization_active_index)
            self._refresh_widget_states()
            self._schedule_preview_update()

        def _copy_normalization_to_all(self) -> None:
            self._persist_normalization_editor(self._normalization_active_index)
            if not self._series_normalization_modes_data:
                return
            mode = self.norm_mode.currentText().strip().lower()
            value = self.norm_value.text().strip()
            x_ref = self.norm_x_ref.text().strip()
            for index in range(len(self._series_normalization_modes_data)):
                self._series_normalization_modes_data[index] = mode
                self._series_normalization_values_data[index] = value
                self._series_normalization_x_refs_data[index] = x_ref
            self._load_normalization_into_editor(self._normalization_active_index)
            self._update_normalization_warning()
            self._schedule_preview_update()

        def _update_normalization_warning(self) -> None:
            if not hasattr(self, "normalization_warning"):
                return
            series_count = len(self._series_normalization_modes_data)
            normalized_count = sum(
                1 for mode in self._series_normalization_modes_data if mode != "none"
            )
            if series_count > 1 and 0 < normalized_count < series_count:
                self.normalization_warning.setText(
                    "Warning: only part of the plotted series is normalized. "
                    "Interpret y-axis comparisons carefully."
                )
                self.normalization_warning.setStyleSheet("QLabel { color: #a16207; }")
                return
            self.normalization_warning.setText("")
            self.normalization_warning.setStyleSheet("")

        def _bind_live_preview_signals(self) -> None:
            line_widgets = (
                self.title_text,
                self.x_label,
                self.y_label,
                self.legend_title,
                self.legend_columns,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.x_ticks,
                self.y_ticks,
                self.x_tick_rotation,
                self.y_tick_rotation,
                self.x_margin,
                self.y_margin,
                self.fig_width,
                self.fig_height,
                self.dpi,
                self.font_family,
                self.title_font,
                self.label_font,
                self.tick_font,
                self.line_width,
                self.line_color,
                self.line_alpha,
                self.marker_size,
                self.figure_facecolor,
                self.grid_linewidth,
                self.grid_alpha,
                self.grid_color,
                self.tick_length,
                self.tick_width,
                self.x_bin_width,
            )
            for widget in line_widgets:
                widget.textChanged.connect(self._schedule_preview_update)

            combo_widgets = (
                self.title_mode,
                self.legend_mode,
                self.legend_loc,
                self.legend_frame_mode,
                self.ticks_mode,
                self.ticks_axis,
                self.grid_mode,
                self.markers_mode,
                self.x_scale,
                self.y_scale,
                self.transparent_mode,
                self.line_style,
                self.grid_linestyle,
                self.grid_axis,
                self.grid_which,
                self.tick_direction,
                self.minor_ticks_mode,
                self.x_bin_reducer,
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

        def _set_preview_zoom(self, value: float) -> None:
            self._preview_zoom_factor = max(0.2, min(20.0, float(value)))

        def _zoom_preview_at_viewport_pos(self, viewport_pos: Any, *, direction: int) -> bool:
            if (
                self._preview_pixmap is None
                or self._preview_pixmap.isNull()
                or self._preview_scroll is None
                or self._preview_label is None
            ):
                return False
            step = 1.12 if direction > 0 else (1.0 / 1.12)
            old_zoom = self._preview_zoom_factor
            self._set_preview_zoom(old_zoom * step)
            if abs(self._preview_zoom_factor - old_zoom) < 1.0e-9:
                return False

            hbar = self._preview_scroll.horizontalScrollBar()
            vbar = self._preview_scroll.verticalScrollBar()
            old_width = max(1, self._preview_label.width())
            old_height = max(1, self._preview_label.height())
            if hasattr(viewport_pos, "x") and hasattr(viewport_pos, "y"):
                cursor_x = float(viewport_pos.x())
                cursor_y = float(viewport_pos.y())
            else:
                cursor_x = 0.0
                cursor_y = 0.0

            relative_x = (hbar.value() + cursor_x) / old_width
            relative_y = (vbar.value() + cursor_y) / old_height

            self._refresh_preview_pixmap()

            new_width = max(1, self._preview_label.width())
            new_height = max(1, self._preview_label.height())
            hbar.setValue(int(relative_x * new_width - cursor_x))
            vbar.setValue(int(relative_y * new_height - cursor_y))
            return True

        def _refresh_preview_pixmap(self) -> None:
            if (
                self._preview_pixmap is None
                or self._preview_pixmap.isNull()
                or self._preview_scroll is None
                or self._preview_label is None
            ):
                return
            target_size = self._preview_scroll.viewport().size()
            if target_size.width() < 2 or target_size.height() < 2:
                return

            source_size = self._preview_pixmap.size()
            fit_scale = min(
                target_size.width() / max(1, source_size.width()),
                target_size.height() / max(1, source_size.height()),
            )
            if fit_scale <= 0.0:
                return
            effective_scale = fit_scale * self._preview_zoom_factor
            scaled = self._preview_pixmap.scaled(
                max(1, int(source_size.width() * effective_scale)),
                max(1, int(source_size.height() * effective_scale)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)
            self._preview_label.resize(scaled.size())

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
            self._set_combo_value(
                self.legend_frame_mode,
                _extract_dict_mode(settings, key="legend_kwargs", nested_key="frameon"),
            )
            self.legend_columns.setText(_extract_dict_text(settings, key="legend_kwargs", nested_key="ncols"))

            self._set_combo_value(self.ticks_mode, _toggle_to_mode(settings.get("ticks")))
            self._set_combo_value(self.grid_mode, _toggle_to_mode(settings.get("grid")))
            self._set_combo_value(self.markers_mode, _toggle_to_mode(settings.get("markers")))
            tick_params_settings = settings.get("tick_params_kwargs")
            tick_axis_mode = "auto"
            minor_ticks_mode = "auto"
            if isinstance(tick_params_settings, dict):
                raw_tick_axis = str(
                    tick_params_settings.get("_ticks_axis", tick_params_settings.get("axis", "auto"))
                ).strip().lower()
                if raw_tick_axis in _TICK_AXES:
                    tick_axis_mode = raw_tick_axis
                raw_minor_mode = str(
                    tick_params_settings.get("_minor_ticks_mode", "auto")
                ).strip().lower()
                if raw_minor_mode in {"auto", "on", "off"}:
                    minor_ticks_mode = raw_minor_mode
            self._set_combo_value(self.ticks_axis, tick_axis_mode)

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
            self.x_margin.setText(_extract_dict_text(settings, key="axes_kwargs", nested_key="xmargin"))
            self.y_margin.setText(_extract_dict_text(settings, key="axes_kwargs", nested_key="ymargin"))

            self.fig_width.setText(_extract_figsize_dimension(settings, index=0, fallback=defaults.figure_size[0]))
            self.fig_height.setText(_extract_figsize_dimension(settings, index=1, fallback=defaults.figure_size[1]))
            self.dpi.setText(str(settings.get("dpi") or defaults.dpi))
            self.font_family.setText(str(settings.get("font_family") or ""))
            self.figure_facecolor.setText(
                _extract_dict_text(settings, key="figure_kwargs", nested_key="facecolor")
            )
            self._set_combo_value(
                self.transparent_mode,
                _extract_dict_mode(settings, key="savefig_kwargs", nested_key="transparent"),
            )
            self.title_font.setText(str(settings.get("title_font_size") or defaults.title_font_size))
            self.label_font.setText(str(settings.get("label_font_size") or defaults.label_font_size))
            self.tick_font.setText(str(settings.get("tick_font_size") or defaults.tick_font_size))
            self.line_width.setText(str(settings.get("line_width") or defaults.line_width))
            self.line_color.setText(str(settings.get("line_color") or ""))
            self._set_combo_value(self.line_style, _extract_dict_text(settings, key="line_kwargs", nested_key="linestyle"))
            self.line_alpha.setText(_extract_dict_text(settings, key="line_kwargs", nested_key="alpha"))
            self.marker_size.setText(_extract_dict_text(settings, key="line_kwargs", nested_key="markersize"))
            self._set_combo_value(self.grid_linestyle, str(settings.get("grid_linestyle") or ""))
            self.grid_linewidth.setText(str(settings.get("grid_linewidth") or defaults.grid_linewidth))
            self.grid_alpha.setText(str(settings.get("grid_alpha") or defaults.grid_alpha))
            self.grid_color.setText(_extract_dict_text(settings, key="grid_kwargs", nested_key="color"))
            self._set_combo_value(
                self.grid_axis,
                str(_extract_dict_value(settings, key="grid_kwargs", nested_key="axis") or "auto"),
            )
            self._set_combo_value(
                self.grid_which,
                str(_extract_dict_value(settings, key="grid_kwargs", nested_key="which") or "auto"),
            )
            self._set_combo_value(
                self.tick_direction,
                str(_extract_dict_value(settings, key="tick_params_kwargs", nested_key="direction") or "auto"),
            )
            self.tick_length.setText(
                _extract_dict_text(settings, key="tick_params_kwargs", nested_key="length")
            )
            self.tick_width.setText(
                _extract_dict_text(settings, key="tick_params_kwargs", nested_key="width")
            )
            self._set_combo_value(self.minor_ticks_mode, minor_ticks_mode)

            self.x_bin_width.setText(str(settings.get("x_bin_width") or ""))
            self._set_combo_value(self.x_bin_reducer, str(settings.get("x_bin_reducer") or "mean"))
            self._initialize_series_data(settings)
            self._initialize_normalization_data(settings)
            self.matplotlib_rc_json.setPlainText(_format_json_block(settings.get("matplotlib_rc")))
            self.figure_kwargs_json.setPlainText(_format_json_block(settings.get("figure_kwargs")))
            self.axes_kwargs_json.setPlainText(_format_json_block(settings.get("axes_kwargs")))
            self.line_kwargs_json.setPlainText(_format_json_block(settings.get("line_kwargs")))
            self.legend_kwargs_json.setPlainText(_format_json_block(settings.get("legend_kwargs")))
            self.grid_kwargs_json.setPlainText(_format_json_block(settings.get("grid_kwargs")))
            self.tick_params_kwargs_json.setPlainText(
                _format_json_block(settings.get("tick_params_kwargs"))
            )
            self.tight_layout_kwargs_json.setPlainText(
                _format_json_block(settings.get("tight_layout_kwargs"))
            )
            self.savefig_kwargs_json.setPlainText(_format_json_block(settings.get("savefig_kwargs")))

        def _refresh_widget_states(self, *_unused: object) -> None:
            title_enabled = self.title_mode.currentText().strip().lower() != "off"
            legend_enabled = self.legend_mode.currentText().strip().lower() != "off"
            grid_enabled = self.grid_mode.currentText().strip().lower() != "off"
            ticks_mode = self.ticks_mode.currentText().strip().lower()
            ticks_enabled = ticks_mode != "off"
            rebin_enabled = bool(self.x_bin_width.text().strip()) if hasattr(self, "x_bin_width") else False
            norm_mode = (
                self.norm_mode.currentText().strip().lower()
                if hasattr(self, "norm_mode")
                else "none"
            )
            norm_enabled = norm_mode != "none"
            norm_x_ref_enabled = norm_mode == "value_at_x"

            self._set_rows_visible(self._title_rows, title_enabled)
            self._set_rows_visible(self._legend_rows, legend_enabled)
            self._set_rows_visible(self._grid_rows, grid_enabled)
            for form, field in self._ticks_rows:
                row_visible = ticks_enabled
                if field is self.ticks_axis:
                    row_visible = ticks_mode != "auto"
                self._set_form_row_visible(form, field, row_visible)

            if self._axes_ticks_group is not None:
                self._axes_ticks_group.setVisible(ticks_enabled)
            if self._tick_appearance_group is not None:
                self._tick_appearance_group.setVisible(ticks_enabled)
            if self._grid_group is not None:
                self._grid_group.setVisible(grid_enabled)
            if self._x_bin_reducer_row is not None:
                self._set_form_row_visible(
                    self._x_bin_reducer_row[0],
                    self._x_bin_reducer_row[1],
                    rebin_enabled,
                )
            if self._norm_value_row is not None:
                self._set_form_row_visible(
                    self._norm_value_row[0],
                    self._norm_value_row[1],
                    norm_enabled,
                )
            if self._norm_x_ref_row is not None:
                self._set_form_row_visible(
                    self._norm_x_ref_row[0],
                    self._norm_x_ref_row[1],
                    norm_x_ref_enabled,
                )
            self._update_normalization_warning()

        def _collect_settings(self) -> dict[str, Any]:
            self._persist_series_editor(self._series_active_index)
            self._persist_normalization_editor(self._normalization_active_index)
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

            series_line_kwargs: list[dict[str, Any] | None] = []
            has_series_line_kwargs = False
            for index, raw in enumerate(self._series_line_kwargs_data):
                token = raw.strip()
                if not token:
                    series_line_kwargs.append(None)
                    continue
                parsed = _optional_json_dict(
                    token,
                    field_name=f"Series {index + 1} line kwargs",
                )
                series_line_kwargs.append(parsed)
                has_series_line_kwargs = True
            series_line_kwargs_value: list[dict[str, Any] | None] | None = None
            if has_series_line_kwargs:
                series_line_kwargs_value = series_line_kwargs

            normalization_modes = [mode.strip().lower() for mode in self._series_normalization_modes_data]
            normalization_values: list[float | None] = []
            normalization_x_refs: list[float | None] = []
            has_normalization = False
            for index, mode in enumerate(normalization_modes):
                if mode not in _NORMALIZATION_MODES:
                    raise ValueError(f"Series {index + 1} normalization mode is invalid.")
                value = _optional_float(
                    self._series_normalization_values_data[index],
                    field_name=f"Series {index + 1} normalization target",
                )
                x_ref = _optional_float(
                    self._series_normalization_x_refs_data[index],
                    field_name=f"Series {index + 1} normalization reference x",
                )
                if mode == "none":
                    if value is not None or x_ref is not None:
                        raise ValueError(
                            f"Series {index + 1} normalization mode is 'none'; remove target/reference values."
                        )
                elif mode == "value_at_x":
                    if value is None or x_ref is None:
                        raise ValueError(
                            f"Series {index + 1} normalization mode 'value_at_x' requires target and reference x."
                        )
                    has_normalization = True
                else:
                    if value is None:
                        raise ValueError(
                            f"Series {index + 1} normalization mode '{mode}' requires a target value."
                        )
                    if x_ref is not None:
                        raise ValueError(
                            f"Series {index + 1} normalization reference x is only valid for mode 'value_at_x'."
                        )
                    has_normalization = True
                normalization_values.append(value)
                normalization_x_refs.append(x_ref)

            normalization_modes_value: list[str] | None = None
            normalization_values_value: list[float | None] | None = None
            normalization_x_refs_value: list[float | None] | None = None
            if has_normalization:
                normalization_modes_value = normalization_modes
                normalization_values_value = normalization_values
                normalization_x_refs_value = normalization_x_refs

            matplotlib_rc_value = _optional_json_dict(
                self.matplotlib_rc_json.toPlainText(),
                field_name="Matplotlib rcParams",
            )
            figure_kwargs_value = _optional_json_dict(
                self.figure_kwargs_json.toPlainText(),
                field_name="Figure kwargs",
            )
            axes_kwargs_value = _optional_json_dict(
                self.axes_kwargs_json.toPlainText(),
                field_name="Axes kwargs",
            )
            line_kwargs_value = _optional_json_dict(
                self.line_kwargs_json.toPlainText(),
                field_name="Global line kwargs",
            )
            legend_kwargs_value = _optional_json_dict(
                self.legend_kwargs_json.toPlainText(),
                field_name="Legend kwargs",
            )
            grid_kwargs_value = _optional_json_dict(
                self.grid_kwargs_json.toPlainText(),
                field_name="Grid kwargs",
            )
            tick_params_kwargs_value = _optional_json_dict(
                self.tick_params_kwargs_json.toPlainText(),
                field_name="Tick params kwargs",
            )
            tight_layout_kwargs_value = _optional_json_dict(
                self.tight_layout_kwargs_json.toPlainText(),
                field_name="tight_layout kwargs",
            )
            savefig_kwargs_value = _optional_json_dict(
                self.savefig_kwargs_json.toPlainText(),
                field_name="savefig kwargs",
            )
            figure_kwargs_merged = (
                dict(figure_kwargs_value) if isinstance(figure_kwargs_value, dict) else {}
            )
            facecolor = _optional_text(self.figure_facecolor.text())
            if facecolor is not None:
                figure_kwargs_merged["facecolor"] = facecolor
            figure_kwargs_value = figure_kwargs_merged or None

            axes_kwargs_merged = dict(axes_kwargs_value) if isinstance(axes_kwargs_value, dict) else {}
            x_margin = _optional_float(self.x_margin.text(), field_name="x-margin")
            y_margin = _optional_float(self.y_margin.text(), field_name="y-margin")
            if x_margin is not None:
                axes_kwargs_merged["xmargin"] = x_margin
            else:
                axes_kwargs_merged.pop("xmargin", None)
            if y_margin is not None:
                axes_kwargs_merged["ymargin"] = y_margin
            else:
                axes_kwargs_merged.pop("ymargin", None)
            axes_kwargs_value = axes_kwargs_merged or None

            line_kwargs_merged = dict(line_kwargs_value) if isinstance(line_kwargs_value, dict) else {}
            line_style = _optional_text(self.line_style.currentText())
            line_alpha = _optional_float(self.line_alpha.text(), field_name="line-alpha")
            marker_size = _optional_float(self.marker_size.text(), field_name="marker-size")
            if line_style is not None:
                line_kwargs_merged["linestyle"] = line_style
            if line_alpha is not None:
                line_kwargs_merged["alpha"] = line_alpha
            if marker_size is not None:
                line_kwargs_merged["markersize"] = marker_size
            line_kwargs_value = line_kwargs_merged or None

            legend_kwargs_merged = (
                dict(legend_kwargs_value) if isinstance(legend_kwargs_value, dict) else {}
            )
            legend_frame = _mode_to_toggle(self.legend_frame_mode.currentText())
            legend_columns = _optional_int(self.legend_columns.text(), field_name="legend-columns")
            if legend_columns is not None and legend_columns < 1:
                raise ValueError("legend-columns must be >= 1.")
            if legend_frame is not None:
                legend_kwargs_merged["frameon"] = legend_frame
            if legend_columns is not None:
                legend_kwargs_merged["ncols"] = legend_columns
            legend_kwargs_value = legend_kwargs_merged or None

            grid_kwargs_merged = dict(grid_kwargs_value) if isinstance(grid_kwargs_value, dict) else {}
            grid_color = _optional_text(self.grid_color.text())
            if grid_color is not None:
                grid_kwargs_merged["color"] = grid_color
            grid_axis = self.grid_axis.currentText().strip().lower() or "auto"
            if grid_axis not in _GRID_AXES:
                raise ValueError("Grid axis must be auto, both, x, or y.")
            if grid_axis == "auto":
                grid_kwargs_merged.pop("axis", None)
            else:
                grid_kwargs_merged["axis"] = grid_axis
            grid_which = self.grid_which.currentText().strip().lower() or "auto"
            if grid_which not in _GRID_WHICH:
                raise ValueError("Grid lines selector must be auto, major, minor, or both.")
            if grid_which == "auto":
                grid_kwargs_merged.pop("which", None)
            else:
                grid_kwargs_merged["which"] = grid_which
            grid_kwargs_value = grid_kwargs_merged or None

            tick_params_kwargs_merged = (
                dict(tick_params_kwargs_value) if isinstance(tick_params_kwargs_value, dict) else {}
            )
            tick_direction = self.tick_direction.currentText().strip().lower() or "auto"
            if tick_direction not in _TICK_DIRECTIONS:
                raise ValueError("Tick direction must be auto, out, in, or inout.")
            if tick_direction == "auto":
                tick_params_kwargs_merged.pop("direction", None)
            else:
                tick_params_kwargs_merged["direction"] = tick_direction
            tick_length = _optional_float(self.tick_length.text(), field_name="tick-length")
            if tick_length is None:
                tick_params_kwargs_merged.pop("length", None)
            else:
                tick_params_kwargs_merged["length"] = tick_length
            tick_width = _optional_float(self.tick_width.text(), field_name="tick-width")
            if tick_width is None:
                tick_params_kwargs_merged.pop("width", None)
            else:
                tick_params_kwargs_merged["width"] = tick_width
            ticks_axis = self.ticks_axis.currentText().strip().lower() or "auto"
            if ticks_axis not in _TICK_AXES:
                raise ValueError("Tick axis must be auto, both, x, or y.")
            if ticks_axis == "auto":
                tick_params_kwargs_merged.pop("_ticks_axis", None)
            else:
                tick_params_kwargs_merged["_ticks_axis"] = ticks_axis
                tick_params_kwargs_merged.setdefault("axis", ticks_axis)
            minor_ticks_mode = self.minor_ticks_mode.currentText().strip().lower() or "auto"
            if minor_ticks_mode not in {"auto", "on", "off"}:
                raise ValueError("Minor ticks mode must be auto, on, or off.")
            if minor_ticks_mode == "auto":
                tick_params_kwargs_merged.pop("_minor_ticks_mode", None)
            else:
                tick_params_kwargs_merged["_minor_ticks_mode"] = minor_ticks_mode
            tick_params_kwargs_value = tick_params_kwargs_merged or None

            savefig_kwargs_merged = (
                dict(savefig_kwargs_value) if isinstance(savefig_kwargs_value, dict) else {}
            )
            transparent = _mode_to_toggle(self.transparent_mode.currentText())
            if transparent is not None:
                savefig_kwargs_merged["transparent"] = transparent
            savefig_kwargs_value = savefig_kwargs_merged or None

            x_bin_width = _optional_float(self.x_bin_width.text(), field_name="x-bin-width")
            if x_bin_width is not None and x_bin_width <= 0:
                raise ValueError("x-bin-width must be positive.")

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
                "series_line_kwargs": series_line_kwargs_value,
                "series_normalization_modes": normalization_modes_value,
                "series_normalization_values": normalization_values_value,
                "series_normalization_x_refs": normalization_x_refs_value,
                "x_bin_width": x_bin_width,
                "x_bin_reducer": (
                    self.x_bin_reducer.currentText().strip() or "mean"
                )
                if x_bin_width is not None
                else None,
                "grid_linestyle": _optional_text(self.grid_linestyle.currentText()),
                "grid_linewidth": _optional_float(
                    self.grid_linewidth.text(), field_name="grid-linewidth"
                ),
                "grid_alpha": _optional_float(self.grid_alpha.text(), field_name="grid-alpha"),
                "matplotlib_rc": matplotlib_rc_value,
                "figure_kwargs": figure_kwargs_value,
                "axes_kwargs": axes_kwargs_value,
                "line_kwargs": line_kwargs_value,
                "legend_kwargs": legend_kwargs_value,
                "grid_kwargs": grid_kwargs_value,
                "tick_params_kwargs": tick_params_kwargs_value,
                "tight_layout_kwargs": tight_layout_kwargs_value,
                "savefig_kwargs": savefig_kwargs_value,
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
                    "Save Figure",
                    self._figure_default_name,
                    self._figure_save_filters,
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
            baseline.pop("series_line_kwargs", None)
            baseline.pop("series_normalization_modes", None)
            baseline.pop("series_normalization_values", None)
            baseline.pop("series_normalization_x_refs", None)
            baseline.pop("x_bin_width", None)
            baseline.pop("x_bin_reducer", None)
            baseline.pop("matplotlib_rc", None)
            baseline.pop("figure_kwargs", None)
            baseline.pop("axes_kwargs", None)
            baseline.pop("line_kwargs", None)
            baseline.pop("legend_kwargs", None)
            baseline.pop("grid_kwargs", None)
            baseline.pop("tick_params_kwargs", None)
            baseline.pop("tight_layout_kwargs", None)
            baseline.pop("savefig_kwargs", None)
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
                "Import Plot Settings",
                "",
                "JSON files (*.json);;HDF5 files (*.h5 *.hdf5);;All files (*)",
            )
            if not path_str:
                return
            try:
                source_path = Path(path_str)
                suffix = source_path.suffix.lower()
                if suffix in {".h5", ".hdf5"}:
                    if on_import_hdf5 is None:
                        raise ValueError(
                            "HDF5 import is unavailable for this plot session."
                        )
                    payload = on_import_hdf5(path_str)
                    source_label = source_path.name
                else:
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                    source_label = source_path.name
                if not isinstance(payload, dict):
                    raise ValueError("JSON root must be an object with setting keys.")
                payload = _without_series_specific_settings(payload)
                merged = dict(initial_settings)
                merged.update(payload)
                self._suspend_preview_events = True
                try:
                    self._populate(merged)
                finally:
                    self._suspend_preview_events = False
                self._refresh_widget_states()
                self._status_label.setText(
                    f"Imported non-series settings from '{source_label}'."
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Import failed", exc)

        def _handle_export_json(self) -> None:
            try:
                settings = self._collect_settings()
                settings = _without_series_specific_settings(settings)
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

        def _refresh_preview_after_layout(self) -> None:
            self._refresh_preview_pixmap()
            if self._preview_pixmap is None or self._preview_pixmap.isNull():
                self._update_embedded_preview(interactive=False)

        def showEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            super().showEvent(event)
            QTimer.singleShot(0, self._refresh_preview_after_layout)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # pragma: no cover - UI flow
            preview_scroll = self._preview_scroll
            preview_label = self._preview_label
            preview_frame = self._preview_frame
            preview_viewport = preview_scroll.viewport() if preview_scroll is not None else None

            if watched in {preview_viewport, preview_label}:
                if event.type() == QEvent.Type.Wheel:
                    delta = int(event.angleDelta().y())
                    if delta != 0 and self._zoom_preview_at_viewport_pos(
                        event.position(),
                        direction=1 if delta > 0 else -1,
                    ):
                        event.accept()
                        return True
                if event.type() == QEvent.Type.MouseButtonDblClick:
                    self._set_preview_zoom(1.0)
                    self._refresh_preview_pixmap()
                    event.accept()
                    return True
            if watched in {preview_frame, preview_viewport} and event.type() == QEvent.Type.Resize:
                QTimer.singleShot(0, self._refresh_preview_pixmap)
            return super().eventFilter(watched, event)

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
