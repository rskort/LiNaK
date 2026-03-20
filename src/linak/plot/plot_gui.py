"""PySide6 GUI panel for interactive plot settings."""

from __future__ import annotations

import json
from copy import deepcopy
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
_GRID_AXES = ("both", "x", "y")
_GRID_WHICH = ("major", "minor", "both")
_TICK_AXES = ("both", "x", "y")
_TICK_VISIBILITY_MODES = ("both", "x", "y", "none")
_TICK_DIRECTIONS = ("out", "in", "inout")
_MINOR_TICKS_MODES = ("off", "on")
_TOGGLE_MODES = ("on", "off")
_SYNC_MODES = ("Auto", "Manual")
_PROFILE_FILTER_METADATA_LABEL = "Use stored metadata"
_PROFILE_FILTER_SPECIES_B_AUTO_LABEL = "Same as Species A / stored metadata"
_SERIES_SPECIFIC_SETTINGS = frozenset(
    {
        "series_labels",
        "series_descriptors",
        "series_overrides",
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
_SYNCED_FIELD_KEYS = frozenset(
    {
        "title",
        "x_label",
        "y_label",
        "x_lim",
        "y_lim",
        "x_ticks",
        "y_ticks",
        "x_label_pad",
        "y_label_pad",
    }
)
_NEW_PROFILE_RESET_KEYS = frozenset(
    {
        "series_overrides",
        "series_enabled",
        "series_line_widths",
        "series_markers",
        "series_line_kwargs",
        "series_normalization_modes",
        "series_normalization_values",
        "series_normalization_x_refs",
        "line_colors",
    }
)


def _without_series_specific_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without per-series plot controls."""
    return {key: value for key, value in settings.items() if key not in _SERIES_SPECIFIC_SETTINGS}


def _without_new_profile_series_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(settings)
    for key in _NEW_PROFILE_RESET_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _toggle_to_mode(value: bool | None, *, auto_mode: str = "on") -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return auto_mode


def _mode_to_toggle(value: str) -> bool | None:
    token = value.strip().lower()
    if token == "on":
        return True
    if token == "off":
        return False
    return None


def _lock_to_sync_mode(locked: bool) -> str:
    return "Manual" if bool(locked) else "Auto"


def _sync_mode_to_lock(value: str) -> bool:
    return str(value).strip().lower() == "manual"


def _explicit_text(value: str) -> str:
    return str(value).strip()


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


def _format_float_value(value: Any, *, decimals: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not numeric.is_integer():
        return f"{numeric:.{decimals}f}".rstrip("0").rstrip(".")
    return str(int(numeric))


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


def _resolve_series_line_colors(colors: list[str]) -> list[str] | None:
    """Return explicit per-series colors while preserving blank entries as no-override."""
    normalized = [str(color).strip() for color in colors]
    return normalized if any(normalized) else None


def _format_series_display_text(index: int, label: str, *, enabled: bool) -> str:
    resolved_label = str(label).strip() or f"Series {index + 1}"
    suffix = "" if enabled else " (off)"
    return f"{index + 1}: {resolved_label}{suffix}"


def _coerce_series_descriptors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    descriptors: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            continue
        series_id = str(raw.get("series_id") or f"series:{index}").strip() or f"series:{index}"
        default_label = str(raw.get("default_label") or "").strip()
        descriptors.append(
            {
                "series_id": series_id,
                "default_label": default_label,
                "source_name": str(raw.get("source_name") or "").strip(),
                "source_directory": str(raw.get("source_directory") or "").strip(),
                "source_path": str(raw.get("source_path") or "").strip(),
                "source_index": raw.get("source_index"),
                "series_index": raw.get("series_index"),
            }
        )
    return descriptors


def _coerce_series_overrides(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in value.items():
        series_id = str(raw_key).strip()
        if not series_id or not isinstance(raw_value, dict):
            continue
        overrides[series_id] = dict(raw_value)
    return overrides


def _coerce_lock_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    resolved: dict[str, bool] = {}
    for key, raw in value.items():
        token = str(key).strip()
        if token in _SYNCED_FIELD_KEYS:
            resolved[token] = bool(raw)
    return resolved


def _coerce_profile_filter_options(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _derive_synced_field_locks(settings: dict[str, Any]) -> dict[str, bool]:
    metadata = _coerce_lock_map(settings.get("_gui_locked_fields"))
    if metadata:
        return metadata
    x_lim = settings.get("x_lim")
    y_lim = settings.get("y_lim")
    return {
        "title": settings.get("title") is not None,
        "x_label": settings.get("x_label") is not None,
        "y_label": settings.get("y_label") is not None,
        "x_lim": isinstance(x_lim, (list, tuple)) and any(value is not None for value in x_lim[:2]),
        "y_lim": isinstance(y_lim, (list, tuple)) and any(value is not None for value in y_lim[:2]),
        "x_ticks": settings.get("x_ticks") is not None,
        "y_ticks": settings.get("y_ticks") is not None,
        "x_label_pad": settings.get("x_label_pad") is not None,
        "y_label_pad": settings.get("y_label_pad") is not None,
    }


def _derive_warning_messages(
    settings: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> list[str]:
    messages: list[str] = []
    if error:
        messages.append(f"Fix invalid setting: {error}")
    if not isinstance(settings, dict):
        return messages

    legend_enabled = settings.get("legend") is not False
    legend_kwargs = settings.get("legend_kwargs")
    legend_columns = legend_kwargs.get("ncols") if isinstance(legend_kwargs, dict) else None
    if not legend_enabled and (settings.get("legend_title") or legend_columns is not None):
        messages.append("Legend is off; legend title and layout options will not be used.")

    ticks_enabled = settings.get("ticks") is not False
    tick_params = settings.get("tick_params_kwargs")
    if not ticks_enabled and isinstance(tick_params, dict):
        tick_keys = {"direction", "length", "width", "axis", "_ticks_axis", "_minor_ticks_mode"}
        if any(key in tick_params for key in tick_keys):
            messages.append("Ticks are off; tick appearance options will not be used.")

    normalization_modes = settings.get("series_normalization_modes")
    if isinstance(normalization_modes, (list, tuple)):
        normalized_count = sum(
            1 for mode in normalization_modes if str(mode).strip().lower() != "none"
        )
        if len(normalization_modes) > 1 and 0 < normalized_count < len(normalization_modes):
            messages.append(
                "Only part of the plotted series is normalized; compare y-axis values carefully."
            )

    return messages


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


def _extract_dict_mode(
    settings: dict[str, Any], *, key: str, nested_key: str, auto_mode: str = "on"
) -> str:
    value = _extract_dict_value(settings, key=key, nested_key=nested_key)
    if isinstance(value, bool):
        return _toggle_to_mode(value, auto_mode=auto_mode)
    return auto_mode


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


def _resolve_asset_path(filename: str, *, module_path: Path | None = None) -> Path:
    """Resolve shared asset files from a source tree or an installed distribution."""
    resolved_module_path = (
        module_path.resolve() if module_path is not None else Path(__file__).resolve()
    )
    for parent in resolved_module_path.parents:
        repo_candidate = parent / "assets" / filename
        if repo_candidate.exists():
            return repo_candidate
        installed_candidate = parent / "share" / "linak" / "assets" / filename
        if installed_candidate.exists():
            return installed_candidate
    return resolved_module_path.parent / "assets" / filename


def _default_gui_artwork_path() -> Path:
    return _resolve_asset_path("logo_simple.svg")


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
    return _format_float_value(value)


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
    on_preview: Callable[[dict[str, Any]], dict[str, Any] | None],
    on_save: Callable[[str, dict[str, Any]], str],
    on_save_figure: Callable[[dict[str, Any], str], str | tuple[str, dict[str, Any]]] | None = None,
    on_import_hdf5: Callable[[str, str], dict[str, Any]] | None = None,
    analysis_name: str | None = None,
    on_resolve_series_defaults: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    initial_profile_name: str | None = None,
    available_profile_names: list[str] | None = None,
    default_profile_settings: dict[str, Any] | None = None,
    on_load_profile: Callable[[str], dict[str, Any]] | None = None,
    on_delete_profile: Callable[[str], tuple[str | None, str]] | None = None,
    on_set_active_profile: Callable[[str], str] | None = None,
) -> None:
    """Open a PySide6 panel that previews and persists plot settings."""
    try:
        from PySide6.QtCore import QEvent, QTimer, Qt
        from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap
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
            QInputDialog,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSplitter,
            QStackedWidget,
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
            self._analysis_name = (analysis_name or "").strip().lower() or None
            self._on_resolve_series_defaults = on_resolve_series_defaults
            self._saved_signature: str | None = None
            self._profile_filter_options = _coerce_profile_filter_options(
                initial_settings.get("_profile_filter_options")
            )
            normalized_profile_names: list[str] = []
            for raw_name in available_profile_names or []:
                candidate = str(raw_name).strip()
                if candidate and candidate not in normalized_profile_names:
                    normalized_profile_names.append(candidate)
            requested_profile_name = str(initial_profile_name or "").strip()
            if requested_profile_name and requested_profile_name not in normalized_profile_names:
                normalized_profile_names.insert(0, requested_profile_name)
            if not normalized_profile_names:
                normalized_profile_names = ["Default"]
            self._profile_names = normalized_profile_names
            self._current_profile_name = requested_profile_name or self._profile_names[0]
            self._default_profile_settings = dict(
                default_profile_settings
                if isinstance(default_profile_settings, dict)
                else initial_settings
            )
            self._profile_selector_syncing = False
            self._title_rows: list[tuple[QFormLayout, QWidget]] = []
            self._legend_rows: list[tuple[QFormLayout, QWidget]] = []
            self._ticks_rows: list[tuple[QFormLayout, QWidget]] = []
            self._grid_rows: list[tuple[QFormLayout, QWidget]] = []
            self._marker_rows: list[tuple[QFormLayout, QWidget]] = []
            self._x_bin_reducer_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_value_row: tuple[QFormLayout, QWidget] | None = None
            self._norm_x_ref_row: tuple[QFormLayout, QWidget] | None = None
            self._position_map_color_row: tuple[QFormLayout, QWidget] | None = None
            self._position_time_axis_row: tuple[QFormLayout, QWidget] | None = None
            self._coordination_time_axis_row: tuple[QFormLayout, QWidget] | None = None
            self._single_series_line_color_row: tuple[QFormLayout, QWidget] | None = None
            self._axes_ticks_group: QGroupBox | None = None
            self._tick_appearance_group: QGroupBox | None = None
            self._grid_group: QGroupBox | None = None
            self._data_transform_group: QGroupBox | None = None
            self._normalization_group: QGroupBox | None = None
            self._series_syncing = False
            self._series_active_index = 0
            self._series_descriptors_data: list[dict[str, Any]] = []
            self._series_labels_data: list[str] = []
            self._series_label_overrides_data: list[str] = []
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
            self._last_preview_state: dict[str, Any] = {}
            self._synced_field_locks: dict[str, bool] = {}
            self._advanced_json_syncing = False
            self._suspend_preview_events = False
            self._preview_pixmap: QPixmap | None = None
            self._preview_zoom_factor = 1.0
            self._preview_frame: QFrame | None = None
            self._preview_scroll: QScrollArea | None = None
            self._preview_label: QLabel | None = None
            self._nav_list: QListWidget | None = None
            self._page_stack: QStackedWidget | None = None
            self._page_title_label: QLabel | None = None
            self._header_state_label: QLabel | None = None
            self._overview_session_label: QLabel | None = None
            self._overview_analysis_label: QLabel | None = None
            self._overview_profile_label: QLabel | None = None
            self._overview_series_label: QLabel | None = None
            self._overview_preview_label: QLabel | None = None
            self._overview_override_label: QLabel | None = None
            self._overview_warning_label: QLabel | None = None
            self._warning_summary_label: QLabel | None = None
            self._series_meta_default_label: QLabel | None = None
            self._series_meta_source_name: QLabel | None = None
            self._series_meta_source_dir: QLabel | None = None
            self._series_meta_series_id: QLabel | None = None
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

        def _configure_horizontal_growth(self, widget: QWidget) -> QWidget:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                widget.sizePolicy().verticalPolicy(),
            )
            return widget

        def _combo(
            self,
            values: tuple[str, ...],
            *,
            editable: bool = False,
            minimum_contents_length: int = 12,
        ) -> QComboBox:
            widget = QComboBox()
            widget.addItems(list(values))
            widget.setEditable(editable)
            widget.setMinimumContentsLength(max(8, minimum_contents_length))
            widget.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._configure_horizontal_growth(widget)
            return widget

        def _line(self, placeholder: str = "") -> QLineEdit:
            widget = QLineEdit()
            if placeholder:
                widget.setPlaceholderText(placeholder)
            self._configure_horizontal_growth(widget)
            return widget

        def _sync_mode_widget(self) -> QComboBox:
            widget = self._combo(_SYNC_MODES)
            widget.setToolTip(
                "Auto follows the latest preview-derived value. Manual keeps the value you type."
            )
            return widget

        def _lockable_line(
            self,
            *,
            placeholder: str = "",
        ) -> tuple[QWidget, QLineEdit, QComboBox]:
            container = QWidget()
            self._configure_horizontal_growth(container)
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            field = self._line(placeholder)
            lock = self._sync_mode_widget()
            row.addWidget(field, stretch=1)
            row.addWidget(lock)
            return container, field, lock

        def _lockable_pair(
            self,
            *,
            first_placeholder: str = "",
            second_placeholder: str = "",
        ) -> tuple[QWidget, QLineEdit, QLineEdit, QComboBox]:
            container = QWidget()
            self._configure_horizontal_growth(container)
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            first = self._line(first_placeholder)
            second = self._line(second_placeholder)
            lock = self._sync_mode_widget()
            row.addWidget(first, stretch=1)
            row.addWidget(second, stretch=1)
            row.addWidget(lock)
            return container, first, second, lock

        def _color_field(self, *, placeholder: str = "") -> tuple[QWidget, QLineEdit]:
            container = QWidget()
            self._configure_horizontal_growth(container)
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

        def _set_form_row_enabled(
            self,
            form: QFormLayout,
            field: QWidget,
            enabled: bool,
        ) -> None:
            label = form.labelForField(field)
            if label is not None:
                label.setEnabled(enabled)
            field.setEnabled(enabled)

        def _set_rows_enabled(
            self,
            rows: list[tuple[QFormLayout, QWidget]],
            enabled: bool,
        ) -> None:
            for form, field in rows:
                self._set_form_row_enabled(form, field, enabled)

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
            if tab.objectName() == "plotSubtabPage":
                scroll.setObjectName("plotSubtabScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMinimumWidth(0)
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            if scroll.viewport() is not None and tab.objectName() == "plotSubtabPage":
                scroll.viewport().setObjectName("plotSubtabViewport")
            content = QWidget(scroll)
            if tab.objectName() == "plotSubtabPage":
                content.setObjectName("plotSubtabContent")
            content.setMinimumWidth(0)
            content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            scroll.setWidget(content)
            tab_layout.addWidget(scroll)
            return content

        def _set_synced_field_lock(self, key: str, locked: bool) -> None:
            mode_widget = getattr(self, f"_{key}_lock", None)
            if mode_widget is None:
                return
            self._synced_field_locks[key] = bool(locked)
            mode_widget.blockSignals(True)
            try:
                self._set_combo_value(mode_widget, _lock_to_sync_mode(locked))
            finally:
                mode_widget.blockSignals(False)

        def _connect_lockable_line(self, key: str, field: QLineEdit, lock: QComboBox) -> None:
            setattr(self, f"_{key}_lock", lock)
            field.textEdited.connect(
                lambda _text, sync_key=key: self._handle_synced_field_edit(sync_key)
            )
            lock.currentTextChanged.connect(
                lambda value, sync_key=key: self._handle_synced_field_lock_toggled(
                    sync_key, _sync_mode_to_lock(value)
                )
            )

        def _apply_preview_state_to_synced_fields(self, settings: dict[str, Any]) -> None:
            self._last_preview_state = dict(settings)
            self._suspend_preview_events = True
            try:
                if not self._synced_field_locks.get("title", False):
                    self.title_text.setText(str(settings.get("title") or ""))
                if not self._synced_field_locks.get("x_label", False):
                    self.x_label.setText(str(settings.get("x_label") or ""))
                if not self._synced_field_locks.get("y_label", False):
                    self.y_label.setText(str(settings.get("y_label") or ""))
                if not self._synced_field_locks.get("x_lim", False):
                    self.x_min.setText(_extract_limit(settings, key="x_lim", index=0))
                    self.x_max.setText(_extract_limit(settings, key="x_lim", index=1))
                if not self._synced_field_locks.get("y_lim", False):
                    self.y_min.setText(_extract_limit(settings, key="y_lim", index=0))
                    self.y_max.setText(_extract_limit(settings, key="y_lim", index=1))
                if not self._synced_field_locks.get("x_ticks", False):
                    self.x_ticks.setText(_format_float_list(settings.get("x_ticks")))
                if not self._synced_field_locks.get("y_ticks", False):
                    self.y_ticks.setText(_format_float_list(settings.get("y_ticks")))
                if not self._synced_field_locks.get("x_label_pad", False):
                    x_label_pad = settings.get("x_label_pad")
                    self.x_label_pad.setText("" if x_label_pad is None else str(x_label_pad))
                if not self._synced_field_locks.get("y_label_pad", False):
                    y_label_pad = settings.get("y_label_pad")
                    self.y_label_pad.setText("" if y_label_pad is None else str(y_label_pad))
            finally:
                self._suspend_preview_events = False

        def _handle_synced_field_edit(self, key: str) -> None:
            self._set_synced_field_lock(key, True)

        def _handle_synced_field_lock_toggled(self, key: str, checked: bool) -> None:
            self._synced_field_locks[key] = bool(checked)
            if checked:
                self._schedule_preview_update()
                return
            self._apply_preview_state_to_synced_fields(self._last_preview_state)
            self._schedule_preview_update()

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

        @staticmethod
        def _normalize_profile_name(name: str) -> str:
            normalized = str(name).strip()
            if not normalized:
                raise ValueError("Profile name cannot be empty.")
            return normalized

        def _profile_name_exists(self, name: str) -> bool:
            lowered = name.casefold()
            return any(existing.casefold() == lowered for existing in self._profile_names)

        def _sync_profile_selector(self) -> None:
            self._profile_selector_syncing = True
            try:
                self._profile_selector.clear()
                for name in self._profile_names:
                    self._profile_selector.addItem(name)
                if self._current_profile_name in self._profile_names:
                    self._profile_selector.setCurrentText(self._current_profile_name)
                elif self._profile_names:
                    self._current_profile_name = self._profile_names[0]
                    self._profile_selector.setCurrentText(self._current_profile_name)
            finally:
                self._profile_selector_syncing = False
            delete_button = getattr(self, "_profile_delete_button", None)
            if delete_button is not None:
                delete_button.setEnabled(
                    len(self._profile_names) > 1 and on_delete_profile is not None
                )

        def _resolved_default_profile_settings(self) -> dict[str, Any]:
            settings = _without_new_profile_series_overrides(self._default_profile_settings)
            if self._on_resolve_series_defaults is not None:
                try:
                    resolved = self._on_resolve_series_defaults(settings)
                except Exception:
                    resolved = None
                if isinstance(resolved, dict):
                    if "series_count" in resolved:
                        settings["series_count"] = resolved["series_count"]
                    if "series_labels" in resolved:
                        settings["series_labels"] = list(resolved["series_labels"])
                    if "series_descriptors" in resolved:
                        settings["series_descriptors"] = list(resolved["series_descriptors"])
            return settings

        def _load_settings_into_editor(
            self,
            settings: dict[str, Any],
            *,
            profile_name: str | None = None,
            status_message: str | None = None,
            mark_saved: bool,
        ) -> None:
            if profile_name is not None:
                self._current_profile_name = profile_name
                self._sync_profile_selector()
            self._suspend_preview_events = True
            try:
                self._populate(settings)
            finally:
                self._suspend_preview_events = False
            self._refresh_widget_states()
            if status_message:
                self._status_label.setText(status_message)
            if mark_saved:
                self._saved_signature = self._signature(self._collect_settings())
            self._refresh_shell_state()

        def _save_current_profile(self, *, status_prefix: str | None = None) -> bool:
            try:
                settings = self._collect_settings()
                message = on_save(self._current_profile_name, settings)
                self._status_label.setText(
                    message if status_prefix is None else f"{status_prefix}{message}"
                )
                self._saved_signature = self._signature(settings)
                self._refresh_shell_state()
                return True
            except Exception as exc:
                self._report_error("Save failed", exc)
                return False

        def _confirm_context_change(self, *, action_label: str) -> bool:
            try:
                settings = self._collect_settings()
                current_signature = self._signature(settings)
            except Exception as exc:
                decision = QMessageBox.question(
                    self,
                    "Invalid settings",
                    "Current profile contains invalid values "
                    f"({exc}). Continue without saving before {action_label}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                return decision == QMessageBox.StandardButton.Yes

            if self._saved_signature == current_signature:
                return True

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Unsaved profile changes")
            box.setText(
                f"Save changes to profile '{self._current_profile_name}' before {action_label}?"
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Save)
            decision = box.exec()
            if decision == QMessageBox.StandardButton.Cancel:
                return False
            if decision == QMessageBox.StandardButton.Save:
                return self._save_current_profile()
            return True

        def _prompt_profile_name(self, *, title_text: str, default_value: str) -> str | None:
            raw_value, accepted = QInputDialog.getText(
                self,
                title_text,
                "Profile name",
                text=default_value,
            )
            if not accepted:
                self._status_label.setText(f"{title_text} canceled.")
                return None
            name = self._normalize_profile_name(raw_value)
            if self._profile_name_exists(name):
                raise ValueError(f"A profile named '{name}' already exists.")
            return name

        def _next_duplicate_profile_name(self) -> str:
            base = f"{self._current_profile_name} Copy"
            if not self._profile_name_exists(base):
                return base
            for index in range(2, 1000):
                candidate = f"{base} {index}"
                if not self._profile_name_exists(candidate):
                    return candidate
            raise ValueError("Could not find an available duplicate profile name.")

        def _handle_profile_selection_request(self, index: int) -> None:
            if self._profile_selector_syncing or index < 0:
                return
            requested = self._profile_selector.itemText(index).strip()
            if not requested or requested == self._current_profile_name:
                return
            if not self._confirm_context_change(action_label=f"switching to '{requested}'"):
                self._sync_profile_selector()
                return
            try:
                if on_load_profile is None:
                    raise ValueError("Profile loading is unavailable for this plot session.")
                loaded = on_load_profile(requested)
                message = f"Loaded profile '{requested}'."
                if on_set_active_profile is not None:
                    message = on_set_active_profile(requested)
                self._load_settings_into_editor(
                    loaded,
                    profile_name=requested,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._sync_profile_selector()
                self._report_error("Load profile failed", exc)

        def _handle_new_profile(self) -> None:
            if not self._confirm_context_change(action_label="creating a new profile"):
                return
            try:
                name = self._prompt_profile_name(
                    title_text="Create profile",
                    default_value="New Profile",
                )
                if name is None:
                    return
                settings = self._resolved_default_profile_settings()
                message = on_save(name, settings)
                if on_set_active_profile is not None:
                    message = on_set_active_profile(name)
                self._profile_names.append(name)
                self._load_settings_into_editor(
                    settings,
                    profile_name=name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Create profile failed", exc)

        def _handle_duplicate_profile(self) -> None:
            try:
                settings = self._collect_settings()
                name = self._prompt_profile_name(
                    title_text="Duplicate profile",
                    default_value=self._next_duplicate_profile_name(),
                )
                if name is None:
                    return
                message = on_save(name, settings)
                if on_set_active_profile is not None:
                    message = on_set_active_profile(name)
                self._profile_names.append(name)
                self._load_settings_into_editor(
                    settings,
                    profile_name=name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Duplicate profile failed", exc)

        def _handle_rename_profile(self) -> None:
            if on_delete_profile is None:
                self._status_label.setText("Rename profile is unavailable.")
                return
            try:
                current_name = self._current_profile_name
                settings = self._collect_settings()
                raw_value, accepted = QInputDialog.getText(
                    self,
                    "Rename profile",
                    "Profile name",
                    text=current_name,
                )
                if not accepted:
                    self._status_label.setText("Rename profile canceled.")
                    return
                name = self._normalize_profile_name(raw_value)
                if name == current_name:
                    self._status_label.setText("Profile name unchanged.")
                    return
                if self._profile_name_exists(name):
                    raise ValueError(f"A profile named '{name}' already exists.")
                message = on_save(name, settings)
                if on_set_active_profile is not None:
                    on_set_active_profile(name)
                on_delete_profile(current_name)
                self._profile_names = [
                    name if entry == current_name else entry for entry in self._profile_names
                ]
                message = f"Renamed profile '{current_name}' to '{name}'."
                self._load_settings_into_editor(
                    settings,
                    profile_name=name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Rename profile failed", exc)

        def _handle_delete_profile(self) -> None:
            if on_delete_profile is None:
                self._status_label.setText("Delete profile is unavailable.")
                return
            if len(self._profile_names) <= 1:
                self._status_label.setText("At least one profile must remain available.")
                return
            if not self._confirm_context_change(action_label="deleting the current profile"):
                return
            decision = QMessageBox.question(
                self,
                "Delete profile",
                f"Delete saved profile '{self._current_profile_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if decision != QMessageBox.StandardButton.Yes:
                self._status_label.setText("Delete profile canceled.")
                return
            try:
                deleted_name = self._current_profile_name
                next_profile_name, message = on_delete_profile(deleted_name)
                self._profile_names = [name for name in self._profile_names if name != deleted_name]
                if next_profile_name is None:
                    self._current_profile_name = self._profile_names[0]
                else:
                    self._current_profile_name = next_profile_name
                    if next_profile_name not in self._profile_names:
                        self._profile_names.append(next_profile_name)
                if on_load_profile is None:
                    raise ValueError("Profile loading is unavailable after deletion.")
                loaded = on_load_profile(self._current_profile_name)
                self._load_settings_into_editor(
                    loaded,
                    profile_name=self._current_profile_name,
                    status_message=message,
                    mark_saved=True,
                )
                self._schedule_preview_update()
            except Exception as exc:
                self._report_error("Delete profile failed", exc)

        def _build_ui(self) -> None:
            root = QWidget(self)
            root.setObjectName("windowRoot")
            self.setCentralWidget(root)
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(12, 12, 12, 12)
            root_layout.setSpacing(12)
            self._apply_window_icon()

            header = QFrame(root)
            header.setObjectName("appHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(14, 12, 14, 12)
            header_layout.setSpacing(10)

            title_block = QVBoxLayout()
            title_block.setSpacing(2)
            app_title = QLabel("LiNaK Studio")
            app_title.setObjectName("appTitle")
            subtitle = QLabel(title)
            subtitle.setObjectName("appSubtitle")
            title_block.addWidget(app_title)
            title_block.addWidget(subtitle)
            header_layout.addLayout(title_block, stretch=1)

            header_layout.addStretch(1)

            self._save_button = QPushButton("Save Profile")
            self._save_button.setProperty("role", "primary")
            self._save_button.clicked.connect(self._handle_save)
            header_layout.addWidget(self._save_button)

            self._save_figure_button = QPushButton("Export Figure")
            self._save_figure_button.setEnabled(on_save_figure is not None)
            self._save_figure_button.clicked.connect(self._handle_save_figure)
            header_layout.addWidget(self._save_figure_button)

            self._header_state_label = QLabel("Ready")
            self._header_state_label.setObjectName("stateBadge")
            header_layout.addWidget(self._header_state_label)
            root_layout.addWidget(header)

            splitter = QSplitter(Qt.Orientation.Horizontal, root)
            splitter.setChildrenCollapsible(False)
            root_layout.addWidget(splitter, stretch=1)

            left_panel = QWidget(splitter)
            left_panel.setMinimumWidth(0)
            left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            left_layout = QHBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(12)

            nav_panel = QFrame(left_panel)
            nav_panel.setObjectName("navPanel")
            nav_panel.setMinimumWidth(96)
            nav_panel.setMaximumWidth(150)
            nav_layout = QVBoxLayout(nav_panel)
            nav_layout.setContentsMargins(14, 14, 14, 14)
            nav_layout.setSpacing(10)
            workspace_label = QLabel("Workspace")
            workspace_label.setObjectName("pageTitle")
            nav_layout.addWidget(workspace_label)
            self._nav_list = QListWidget(nav_panel)
            self._nav_list.setObjectName("navList")
            self._nav_list.currentRowChanged.connect(self._handle_navigation_change)
            nav_layout.addWidget(self._nav_list, stretch=1)
            left_layout.addWidget(nav_panel)

            inspector_panel = QFrame(left_panel)
            inspector_panel.setObjectName("inspectorPanel")
            inspector_panel.setMinimumWidth(0)
            inspector_panel.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            inspector_layout = QVBoxLayout(inspector_panel)
            inspector_layout.setContentsMargins(14, 14, 14, 14)
            inspector_layout.setSpacing(10)
            self._page_title_label = QLabel("Overview")
            self._page_title_label.setObjectName("pageTitle")
            inspector_layout.addWidget(self._page_title_label)
            inspector_note = QLabel(
                "Preview-first controls grouped by task: data, series, figure, profiles, export, and advanced overrides."
            )
            inspector_note.setWordWrap(True)
            inspector_note.setObjectName("sectionNote")
            inspector_layout.addWidget(inspector_note)
            self._page_stack = QStackedWidget(inspector_panel)
            self._page_stack.setMinimumWidth(0)
            self._page_stack.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            inspector_layout.addWidget(self._page_stack, stretch=1)
            left_layout.addWidget(inspector_panel, stretch=1)

            self._register_workspace_page("Overview", self._build_overview_page())
            self._register_workspace_page("Data", self._build_content_page())
            self._register_workspace_page("Series", self._build_series_page())
            self._register_workspace_page("Figure", self._build_figure_page())
            self._register_workspace_page("Profiles", self._build_profiles_page())
            self._register_workspace_page("Export", self._build_export_page())
            self._register_workspace_page("Advanced", self._build_advanced_page())
            self._sync_profile_selector()
            if self._nav_list is not None:
                self._nav_list.setCurrentRow(0)

            right_panel = QFrame(splitter)
            right_panel.setObjectName("previewPanel")
            right_panel.setMinimumWidth(320)
            right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(14, 14, 14, 14)
            right_layout.setSpacing(10)

            preview_title = QLabel("Preview")
            preview_title.setObjectName("pageTitle")
            right_layout.addWidget(preview_title)

            preview_controls = QHBoxLayout()
            self._preview_button = QPushButton("Refresh Preview")
            self._preview_button.clicked.connect(self._handle_preview)
            preview_controls.addWidget(self._preview_button)
            fit_button = QPushButton("Fit")
            fit_button.clicked.connect(self._handle_fit_preview)
            preview_controls.addWidget(fit_button)
            actual_size_button = QPushButton("100%")
            actual_size_button.clicked.connect(self._handle_actual_size_preview)
            preview_controls.addWidget(actual_size_button)
            self._reset_button = QPushButton("Reset to Defaults")
            self._reset_button.clicked.connect(self._handle_reset)
            preview_controls.addWidget(self._reset_button)
            self._auto_preview_checkbox = QCheckBox("Auto update")
            self._auto_preview_checkbox.setChecked(on_save_figure is not None)
            self._auto_preview_checkbox.setEnabled(on_save_figure is not None)
            self._auto_preview_checkbox.toggled.connect(self._handle_auto_preview_toggle)
            preview_controls.addWidget(self._auto_preview_checkbox)
            preview_controls.addStretch(1)
            self._preview_status = QLabel("Preview ready.")
            preview_controls.addWidget(self._preview_status)
            right_layout.addLayout(preview_controls)

            self._warning_summary_label = QLabel("")
            self._warning_summary_label.setWordWrap(True)
            self._warning_summary_label.setObjectName("warningSummary")
            self._warning_summary_label.hide()
            right_layout.addWidget(self._warning_summary_label)

            self._preview_frame = QFrame(right_panel)
            self._preview_frame.setObjectName("previewFrame")
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
            self._status_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            status_row.addWidget(self._status_label)
            self._close_button = QPushButton("Close")
            self._close_button.clicked.connect(self.close)
            status_row.addWidget(self._close_button)
            right_layout.addLayout(status_row)

            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([760, 840])

            self._apply_theme_styles()

        def _is_dark_theme(self) -> bool:
            app = QApplication.instance()
            palette = app.palette() if app is not None else self.palette()
            window_color = palette.color(QPalette.ColorRole.Window)
            text_color = palette.color(QPalette.ColorRole.WindowText)
            return window_color.lightness() < text_color.lightness()

        def _theme_tokens(self) -> dict[str, str]:
            if self._is_dark_theme():
                return {
                    "window_bg": "#0b1220",
                    "header_bg": "#131d2d",
                    "panel_bg": "#101826",
                    "panel_elevated": "#162233",
                    "card_bg": "#142030",
                    "input_bg": "#0e1725",
                    "button_bg": "#182334",
                    "button_hover": "#1e2d42",
                    "button_pressed": "#25364c",
                    "disabled_bg": "#0f1724",
                    "disabled_text": "#607086",
                    "text": "#edf3fb",
                    "heading": "#f7fbff",
                    "muted_text": "#9caec5",
                    "border": "#31425a",
                    "border_soft": "#3f5270",
                    "input_border": "#425774",
                    "accent": "#2aa7b8",
                    "accent_hover": "#34bed1",
                    "accent_text": "#07151b",
                    "accent_soft": "#163e47",
                    "nav_hover": "#182637",
                    "nav_selected": "#18424b",
                    "nav_selected_border": "#2aa7b8",
                    "nav_selected_text": "#f6fbff",
                    "badge_bg": "#1a2637",
                    "badge_border": "#465a76",
                    "badge_text": "#edf3fb",
                    "warning_bg": "#33250e",
                    "warning_border": "#b98934",
                    "warning_text": "#f5d9a1",
                    "splitter": "#42556f",
                    "scrollbar_track": "#0f1724",
                    "scrollbar_thumb": "#40536d",
                    "scrollbar_thumb_hover": "#56708f",
                }
            return {
                "window_bg": "#eef3f8",
                "header_bg": "#f9fbfe",
                "panel_bg": "#fdfefe",
                "panel_elevated": "#ffffff",
                "card_bg": "#f8fafc",
                "input_bg": "#ffffff",
                "button_bg": "#f4f7fb",
                "button_hover": "#e9f0f8",
                "button_pressed": "#dbe6f2",
                "disabled_bg": "#eef2f6",
                "disabled_text": "#8190a3",
                "text": "#142033",
                "heading": "#0f1728",
                "muted_text": "#556274",
                "border": "#c8d3e0",
                "border_soft": "#d9e2ec",
                "input_border": "#bcc9d8",
                "accent": "#0f8f95",
                "accent_hover": "#0c7a80",
                "accent_text": "#f8feff",
                "accent_soft": "#d9f0f2",
                "nav_hover": "#edf4fa",
                "nav_selected": "#0f8f95",
                "nav_selected_border": "#0c7a80",
                "nav_selected_text": "#f8feff",
                "badge_bg": "#edf4fa",
                "badge_border": "#c3d1df",
                "badge_text": "#142033",
                "warning_bg": "#fff3d9",
                "warning_border": "#d8a94f",
                "warning_text": "#7c5400",
                "splitter": "#cad5e1",
                "scrollbar_track": "#edf2f7",
                "scrollbar_thumb": "#b9c5d4",
                "scrollbar_thumb_hover": "#95a6bb",
            }

        def _apply_theme_styles(self) -> None:
            colors = self._theme_tokens()
            self.setStyleSheet(
                f"QWidget#windowRoot {{"
                f"  background-color: {colors['window_bg']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QFrame#appHeader {{"
                f"  background-color: {colors['header_bg']};"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 14px;"
                f"}}"
                f"QFrame#navPanel, QFrame#inspectorPanel, QFrame#previewPanel {{"
                f"  background-color: {colors['panel_bg']};"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 14px;"
                f"}}"
                f"QFrame#previewFrame {{"
                f"  background-color: {colors['panel_elevated']};"
                f"  border: 1px solid {colors['border_soft']};"
                f"  border-radius: 12px;"
                f"}}"
                f"QGroupBox {{"
                f"  background-color: {colors['card_bg']};"
                f"  border: 1px solid {colors['border_soft']};"
                f"  border-radius: 12px;"
                f"  margin-top: 14px;"
                f"}}"
                f"QGroupBox::title {{"
                f"  subcontrol-origin: margin;"
                f"  left: 10px;"
                f"  padding: 0 5px;"
                f"  color: {colors['muted_text']};"
                f"  font-weight: 600;"
                f"}}"
                f"QPushButton {{"
                f"  padding: 7px 12px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['text']};"
                f"}}"
                f"QPushButton:hover {{"
                f"  border-color: {colors['accent']};"
                f"  background-color: {colors['button_hover']};"
                f"}}"
                f"QPushButton:pressed {{"
                f"  background-color: {colors['button_pressed']};"
                f"}}"
                f"QPushButton:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f'QPushButton[role="primary"] {{'
                f"  background-color: {colors['accent']};"
                f"  color: {colors['accent_text']};"
                f"  border-color: {colors['accent']};"
                f"  font-weight: 700;"
                f"}}"
                f'QPushButton[role="primary"]:hover {{'
                f"  background-color: {colors['accent_hover']};"
                f"  border-color: {colors['accent_hover']};"
                f"}}"
                f'QPushButton[role="primary"]:pressed {{'
                f"  background-color: {colors['accent_hover']};"
                f"}}"
                f"QLineEdit, QComboBox, QPlainTextEdit, QListWidget {{"
                f"  border: 1px solid {colors['input_border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['input_bg']};"
                f"  color: {colors['text']};"
                f"  outline: none;"
                f"  selection-background-color: {colors['accent_soft']};"
                f"}}"
                f"QLineEdit, QComboBox {{ padding: 6px 8px; min-height: 18px; }}"
                f"QPlainTextEdit {{ padding: 6px; }}"
                f"QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QListWidget:disabled {{"
                f"  background-color: {colors['disabled_bg']};"
                f"  color: {colors['disabled_text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {{"
                f"  border-color: {colors['accent']};"
                f"  outline: none;"
                f"}}"
                f"QComboBox::drop-down {{ border: none; width: 24px; }}"
                f"QComboBox QAbstractItemView {{"
                f"  border: 1px solid {colors['border']};"
                f"  background-color: {colors['panel_elevated']};"
                f"  color: {colors['text']};"
                f"  selection-background-color: {colors['accent']};"
                f"  selection-color: {colors['accent_text']};"
                f"}}"
                f"QCheckBox {{ spacing: 6px; }}"
                f"QCheckBox::indicator {{"
                f"  width: 16px;"
                f"  height: 16px;"
                f"  border-radius: 4px;"
                f"  border: 1px solid {colors['input_border']};"
                f"  background-color: {colors['input_bg']};"
                f"}}"
                f"QCheckBox::indicator:hover {{ border-color: {colors['accent']}; }}"
                f"QCheckBox::indicator:checked {{"
                f"  background-color: {colors['accent']};"
                f"  border-color: {colors['accent']};"
                f"}}"
                f"QLabel#appTitle {{ font-size: 22px; font-weight: 700; color: {colors['heading']}; }}"
                f"QLabel#appSubtitle {{ color: {colors['muted_text']}; font-size: 12px; }}"
                f"QLabel#pageTitle {{ font-size: 17px; font-weight: 700; color: {colors['heading']}; }}"
                # f"QLabel#sectionTitle {{ font-size: 17px; font-weight: 700; color: {colors['heading']}; }}"
                f"QLabel#sectionNote {{ color: {colors['muted_text']}; }}"
                f"QLabel#stateBadge {{"
                f"  padding: 5px 12px;"
                f"  border: 1px solid {colors['badge_border']};"
                f"  border-radius: 999px;"
                f"  background-color: {colors['badge_bg']};"
                f"  color: {colors['badge_text']};"
                f"  font-weight: 600;"
                f"}}"
                f"QLabel#warningSummary, QLabel#inlineWarning {{"
                f"  padding: 8px 10px;"
                f"  border: 1px solid {colors['warning_border']};"
                f"  border-radius: 8px;"
                f"  background-color: {colors['warning_bg']};"
                f"  color: {colors['warning_text']};"
                f"}}"
                f"QFrame#inspectorPanel QStackedWidget {{"
                f"  background: transparent;"
                f"}}"
                f"QFrame#inspectorPanel QStackedWidget > QWidget {{"
                f"  background: transparent;"
                f"}}"
                f"QFrame#inspectorPanel QAbstractScrollArea::viewport {{"
                f"  background: transparent;"
                f"}}"
                f"QFrame#inspectorPanel QScrollArea > QWidget > QWidget {{"
                f"  background: transparent;"
                f"}}"
                f"QTabWidget#plotSubtabs::pane {{"
                f"  margin-top: 0px;"
                f"  top: -1px;"
                f"  border: 1px solid {colors['border_soft']};"
                f"  border-radius: 12px;"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QWidget#plotSubtabPage, QWidget#plotSubtabContent {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QScrollArea#plotSubtabScroll {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QWidget#plotSubtabViewport {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QScrollArea#plotSubtabScroll > QWidget > QWidget {{"
                f"  background-color: {colors['card_bg']};"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab {{"
                f"  padding: 8px 16px;"
                f"  margin-right: 8px;"
                f"  border: 1px solid {colors['border']};"
                f"  border-bottom: none;"
                f"  border-top-left-radius: 10px;"
                f"  border-top-right-radius: 10px;"
                f"  background-color: {colors['button_bg']};"
                f"  color: {colors['muted_text']};"
                f"  font-weight: 600;"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:hover {{"
                f"  background-color: {colors['nav_hover']};"
                f"  color: {colors['text']};"
                f"  border-color: {colors['border_soft']};"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:selected {{"
                f"  background-color: {colors['panel_elevated']};"
                f"  color: {colors['heading']};"
                f"  border-color: {colors['accent']};"
                f"  margin-bottom: -1px;"
                f"}}"
                f"QTabWidget#plotSubtabs QTabBar::tab:!selected {{"
                f"  margin-top: 4px;"
                f"}}"
                f"QListWidget#navList {{ border: none; background: transparent; padding: 4px; }}"
                f"QListWidget#navList::item {{"
                f"  padding: 12px 14px;"
                f"  margin: 3px 0;"
                f"  border-radius: 10px;"
                f"}}"
                f"QListWidget#navList::item:hover {{ background-color: {colors['nav_hover']}; }}"
                f"QListWidget#navList::item:selected {{"
                f"  background-color: {colors['nav_selected']};"
                f"  color: {colors['nav_selected_text']};"
                f"  border: 1px solid {colors['nav_selected_border']};"
                f"}}"
                f"QScrollArea {{ border: none; background: transparent; }}"
                f"QScrollBar:vertical {{"
                f"  background: {colors['scrollbar_track']};"
                f"  width: 12px;"
                f"  margin: 2px;"
                f"  border-radius: 6px;"
                f"}}"
                f"QScrollBar::handle:vertical {{"
                f"  background: {colors['scrollbar_thumb']};"
                f"  min-height: 24px;"
                f"  border-radius: 6px;"
                f"}}"
                f"QScrollBar::handle:vertical:hover {{ background: {colors['scrollbar_thumb_hover']}; }}"
                f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, "
                f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{"
                f"  height: 0px;"
                f"  background: transparent;"
                f"}}"
                f"QSplitter::handle {{"
                f"  background-color: {colors['splitter']};"
                f"  width: 5px;"
                f"  margin: 6px 4px;"
                f"  border-radius: 2px;"
                f"}}"
            )

        def _register_workspace_page(self, label: str, page: QWidget) -> None:
            if self._page_stack is None or self._nav_list is None:
                return
            self._page_stack.addWidget(page)
            self._nav_list.addItem(label)

        def _handle_navigation_change(self, index: int) -> None:
            if self._page_stack is None or self._nav_list is None or index < 0:
                return
            self._page_stack.setCurrentIndex(index)
            item = self._nav_list.item(index)
            if item is not None and self._page_title_label is not None:
                self._page_title_label.setText(item.text())

        def _humanized_analysis_name(self) -> str:
            if self._analysis_name is None:
                return "Generic"
            return str(self._analysis_name).replace("-", " ").strip().title() or "Generic"

        def _safe_collect_settings(self) -> tuple[dict[str, Any] | None, str | None]:
            try:
                return self._collect_settings(), None
            except Exception as exc:
                return None, str(exc)

        def _advanced_editor(self, section_key: str) -> Any | None:
            return getattr(self, f"{section_key}_json", None)

        def _set_line_text_if_different(self, widget: QLineEdit, value: str) -> None:
            text = str(value)
            if widget.text() == text:
                return
            widget.setText(text)

        def _set_plaintext_if_different(self, widget: QPlainTextEdit, value: Any) -> None:
            rendered = _format_json_block(value)
            if widget.toPlainText().strip() == rendered.strip():
                return
            widget.blockSignals(True)
            try:
                widget.setPlainText(rendered)
            finally:
                widget.blockSignals(False)

        def _apply_advanced_json_to_standard_controls(self, section_key: str) -> None:
            editor = self._advanced_editor(section_key)
            if editor is None or self._advanced_json_syncing:
                return

            raw_text = editor.toPlainText()
            try:
                parsed = _optional_json_dict(raw_text, field_name=section_key.replace("_", " "))
            except ValueError:
                self._refresh_shell_state()
                return
            if parsed is None:
                parsed = {}

            self._advanced_json_syncing = True
            self._suspend_preview_events = True
            try:
                if section_key == "figure_kwargs":
                    self._set_line_text_if_different(
                        self.figure_facecolor,
                        str(parsed.get("facecolor") or ""),
                    )
                elif section_key == "line_kwargs":
                    self._set_combo_value(
                        self.line_style,
                        str(parsed.get("linestyle") or ""),
                    )
                    self._set_line_text_if_different(
                        self.line_alpha,
                        "" if parsed.get("alpha") is None else str(parsed.get("alpha")),
                    )
                    self._set_line_text_if_different(
                        self.marker_size,
                        "" if parsed.get("markersize") is None else str(parsed.get("markersize")),
                    )
                elif section_key == "legend_kwargs":
                    frameon = parsed.get("frameon")
                    if isinstance(frameon, bool):
                        self._set_combo_value(
                            self.legend_frame_mode,
                            _toggle_to_mode(frameon, auto_mode="on"),
                        )
                    else:
                        self._set_combo_value(self.legend_frame_mode, "on")
                    ncols = parsed.get("ncols")
                    self._set_line_text_if_different(
                        self.legend_columns,
                        "" if ncols is None else str(ncols),
                    )
                elif section_key == "grid_kwargs":
                    self._set_line_text_if_different(
                        self.grid_color,
                        str(parsed.get("color") or ""),
                    )
                    axis_value = str(parsed.get("axis") or "both").strip().lower()
                    which_value = str(parsed.get("which") or "major").strip().lower()
                    if axis_value in _GRID_AXES:
                        self._set_combo_value(self.grid_axis, axis_value)
                    if which_value in _GRID_WHICH:
                        self._set_combo_value(self.grid_which, which_value)
                elif section_key == "tick_params_kwargs":
                    direction = str(parsed.get("direction") or "out").strip().lower()
                    axis_value = (
                        str(parsed.get("_ticks_axis", parsed.get("axis", "both"))).strip().lower()
                    )
                    minor_value = str(parsed.get("_minor_ticks_mode") or "off").strip().lower()
                    if direction in _TICK_DIRECTIONS:
                        self._set_combo_value(self.tick_direction, direction)
                    if axis_value in _TICK_AXES:
                        current_ticks_value = (
                            self.ticks_visibility.currentText().strip().lower()
                            if hasattr(self, "ticks_visibility")
                            else "both"
                        )
                        if current_ticks_value != "none":
                            self._set_combo_value(self.ticks_visibility, axis_value)
                    if minor_value in _MINOR_TICKS_MODES:
                        self._set_combo_value(self.minor_ticks_mode, minor_value)
                    self._set_line_text_if_different(
                        self.tick_length,
                        "" if parsed.get("length") is None else str(parsed.get("length")),
                    )
                    self._set_line_text_if_different(
                        self.tick_width,
                        "" if parsed.get("width") is None else str(parsed.get("width")),
                    )
                elif section_key == "savefig_kwargs":
                    transparent = parsed.get("transparent")
                    if isinstance(transparent, bool):
                        self._set_combo_value(
                            self.transparent_mode,
                            _toggle_to_mode(transparent, auto_mode="off"),
                        )
                    else:
                        self._set_combo_value(self.transparent_mode, "off")
            finally:
                self._suspend_preview_events = False
                self._advanced_json_syncing = False

            self._refresh_widget_states()
            self._schedule_preview_update()

        def _sync_standard_controls_to_advanced_json(self) -> None:
            if self._advanced_json_syncing:
                return
            settings, error = self._safe_collect_settings()
            if error or settings is None:
                return

            self._advanced_json_syncing = True
            try:
                if hasattr(self, "figure_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.figure_kwargs_json, settings.get("figure_kwargs")
                    )
                if hasattr(self, "axes_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.axes_kwargs_json, settings.get("axes_kwargs")
                    )
                if hasattr(self, "line_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.line_kwargs_json, settings.get("line_kwargs")
                    )
                if hasattr(self, "legend_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.legend_kwargs_json, settings.get("legend_kwargs")
                    )
                if hasattr(self, "grid_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.grid_kwargs_json, settings.get("grid_kwargs")
                    )
                if hasattr(self, "tick_params_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.tick_params_kwargs_json, settings.get("tick_params_kwargs")
                    )
                if hasattr(self, "savefig_kwargs_json"):
                    self._set_plaintext_if_different(
                        self.savefig_kwargs_json, settings.get("savefig_kwargs")
                    )
            finally:
                self._advanced_json_syncing = False

        def _handle_advanced_editor_changed(self, section_key: str) -> None:
            if self._advanced_json_syncing:
                return
            self._apply_advanced_json_to_standard_controls(section_key)
            self._refresh_shell_state()

        def _refresh_shell_state(self) -> None:
            self._update_header_state()
            self._update_overview_panel()
            self._update_warning_panel()
            profiles_label = getattr(self, "_profiles_current_label", None)
            if profiles_label is not None:
                profiles_label.setText(self._current_profile_name)

        def _update_header_state(self) -> None:
            if self._header_state_label is None:
                return
            settings, error = self._safe_collect_settings()
            if error:
                self._header_state_label.setText("Invalid")
                return
            if settings is None:
                self._header_state_label.setText("Unknown")
                return
            signature = self._signature(settings)
            if self._saved_signature is not None and signature == self._saved_signature:
                self._header_state_label.setText("Saved")
            else:
                self._header_state_label.setText("Unsaved")

        def _update_overview_panel(self) -> None:
            if (
                self._overview_session_label is None
                or self._overview_analysis_label is None
                or self._overview_profile_label is None
                or self._overview_series_label is None
                or self._overview_preview_label is None
                or self._overview_override_label is None
            ):
                return
            settings, error = self._safe_collect_settings()
            self._overview_session_label.setText(title)
            self._overview_analysis_label.setText(self._humanized_analysis_name())
            self._overview_profile_label.setText(self._current_profile_name)
            series_count = len(self._series_labels_data)
            visible_count = sum(1 for value in self._series_enabled_data if value)
            if series_count:
                self._overview_series_label.setText(f"{visible_count} of {series_count} visible")
            else:
                self._overview_series_label.setText("No series detected yet")
            preview_mode = "Auto" if self._auto_preview_checkbox.isChecked() else "Manual"
            self._overview_preview_label.setText(preview_mode)
            manual_overrides = sum(1 for value in self._synced_field_locks.values() if value)
            self._overview_override_label.setText(f"{manual_overrides} manual axis/label overrides")
            warnings = _derive_warning_messages(settings, error=error)
            if self._overview_warning_label is not None:
                self._overview_warning_label.setText(
                    "\n".join(f"- {message}" for message in warnings[:4])
                    if warnings
                    else "No active warnings."
                )

        def _update_warning_panel(self) -> None:
            if self._warning_summary_label is None:
                return
            settings, error = self._safe_collect_settings()
            warnings = _derive_warning_messages(settings, error=error)
            if warnings:
                self._warning_summary_label.setText(
                    "\n".join(f"- {message}" for message in warnings[:4])
                )
                self._warning_summary_label.show()
            else:
                self._warning_summary_label.setText("")
                self._warning_summary_label.hide()

        def _handle_fit_preview(self) -> None:
            self._set_preview_zoom(1.0)
            self._refresh_preview_pixmap()
            self._preview_status.setText("Preview fit to workspace.")

        def _handle_actual_size_preview(self) -> None:
            if (
                self._preview_pixmap is None
                or self._preview_pixmap.isNull()
                or self._preview_scroll is None
            ):
                return
            viewport = self._preview_scroll.viewport().size()
            source = self._preview_pixmap.size()
            fit_scale = min(
                viewport.width() / max(1, source.width()),
                viewport.height() / max(1, source.height()),
            )
            if fit_scale <= 0.0:
                return
            self._set_preview_zoom(1.0 / fit_scale)
            self._refresh_preview_pixmap()
            self._preview_status.setText("Preview shown at 100%.")

        def _build_overview_page(self) -> QWidget:
            page = QWidget()
            content = self._make_scrollable_tab(page)
            layout = QVBoxLayout(content)
            layout.setSpacing(12)

            intro = QLabel(
                "LiNaK Studio keeps the preview visible while data, series, styling, profiles, and export actions stay grouped by task."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            summary = QGroupBox("Session Summary")
            summary_grid = QGridLayout(summary)
            summary_grid.addWidget(QLabel("Session"), 0, 0)
            self._overview_session_label = QLabel(title)
            self._overview_session_label.setWordWrap(True)
            summary_grid.addWidget(self._overview_session_label, 0, 1)
            summary_grid.addWidget(QLabel("Analysis"), 1, 0)
            self._overview_analysis_label = QLabel(self._humanized_analysis_name())
            summary_grid.addWidget(self._overview_analysis_label, 1, 1)
            summary_grid.addWidget(QLabel("Profile"), 2, 0)
            self._overview_profile_label = QLabel(self._current_profile_name)
            summary_grid.addWidget(self._overview_profile_label, 2, 1)
            summary_grid.addWidget(QLabel("Series"), 3, 0)
            self._overview_series_label = QLabel("")
            summary_grid.addWidget(self._overview_series_label, 3, 1)
            summary_grid.addWidget(QLabel("Preview mode"), 4, 0)
            self._overview_preview_label = QLabel("")
            summary_grid.addWidget(self._overview_preview_label, 4, 1)
            summary_grid.addWidget(QLabel("Manual overrides"), 5, 0)
            self._overview_override_label = QLabel("")
            summary_grid.addWidget(self._overview_override_label, 5, 1)
            layout.addWidget(summary)

            quick_actions = QGroupBox("Quick Actions")
            quick_layout = QGridLayout(quick_actions)

            def _action_button(label: str, callback: Callable[[], None]) -> QPushButton:
                button = QPushButton(label)
                button.clicked.connect(callback)
                return button

            quick_layout.addWidget(_action_button("Refresh Preview", self._handle_preview), 0, 0)
            quick_layout.addWidget(_action_button("Save Profile", self._handle_save), 0, 1)
            quick_layout.addWidget(_action_button("Export Figure", self._handle_save_figure), 1, 0)
            quick_layout.addWidget(_action_button("Reset Defaults", self._handle_reset), 1, 1)
            layout.addWidget(quick_actions)

            warnings_group = QGroupBox("Warnings")
            warnings_layout = QVBoxLayout(warnings_group)
            self._overview_warning_label = QLabel("No active warnings.")
            self._overview_warning_label.setWordWrap(True)
            warnings_layout.addWidget(self._overview_warning_label)
            layout.addWidget(warnings_group)
            layout.addStretch(1)
            return page

        def _build_content_page(self) -> QWidget:
            self._tab_data = QWidget()
            self._tab_data_content = self._make_scrollable_tab(self._tab_data)
            self._build_data_tab()
            return self._tab_data

        def _build_series_page(self) -> QWidget:
            self._tab_series = QWidget()
            self._tab_series_content = self._make_scrollable_tab(self._tab_series)
            self._build_series_tab()
            return self._tab_series

        def _build_figure_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            hint = QLabel(
                "Figure controls chart structure and visual treatment. Use Auto and Manual on synced fields to follow or keep preview-derived values."
            )
            hint.setWordWrap(True)
            hint.setObjectName("sectionNote")
            layout.addWidget(hint)
            tabs = QTabWidget(page)
            tabs.setObjectName("plotSubtabs")
            tabs.setDocumentMode(True)

            self._tab_text = QWidget()
            self._tab_text.setObjectName("plotSubtabPage")
            self._tab_legend = QWidget()
            self._tab_legend.setObjectName("plotSubtabPage")
            self._tab_axes = QWidget()
            self._tab_axes.setObjectName("plotSubtabPage")
            self._tab_ticks_grid = QWidget()
            self._tab_ticks_grid.setObjectName("plotSubtabPage")
            self._tab_lines = QWidget()
            self._tab_lines.setObjectName("plotSubtabPage")
            self._tab_canvas = QWidget()
            self._tab_canvas.setObjectName("plotSubtabPage")

            tabs.addTab(self._tab_text, "Text")
            tabs.addTab(self._tab_legend, "Legend")
            tabs.addTab(self._tab_axes, "Axes")
            tabs.addTab(self._tab_ticks_grid, "Ticks && Grid")
            tabs.addTab(self._tab_lines, "Lines && Markers")
            tabs.addTab(self._tab_canvas, "Canvas")

            self._tab_text_content = self._make_scrollable_tab(self._tab_text)
            self._tab_text_content.setObjectName("plotSubtabContent")
            self._tab_legend_content = self._make_scrollable_tab(self._tab_legend)
            self._tab_legend_content.setObjectName("plotSubtabContent")
            self._tab_axes_content = self._make_scrollable_tab(self._tab_axes)
            self._tab_axes_content.setObjectName("plotSubtabContent")
            self._tab_ticks_grid_content = self._make_scrollable_tab(self._tab_ticks_grid)
            self._tab_ticks_grid_content.setObjectName("plotSubtabContent")
            self._tab_lines_content = self._make_scrollable_tab(self._tab_lines)
            self._tab_lines_content.setObjectName("plotSubtabContent")
            self._tab_canvas_content = self._make_scrollable_tab(self._tab_canvas)
            self._tab_canvas_content.setObjectName("plotSubtabContent")

            self._build_text_tab()
            self._build_legend_tab()
            self._build_axes_tab()
            self._build_ticks_grid_tab()
            self._build_lines_tab()
            self._build_canvas_tab()
            layout.addWidget(tabs, stretch=1)
            return page

        def _build_profiles_page(self) -> QWidget:
            page = QWidget()
            content = self._make_scrollable_tab(page)
            layout = QVBoxLayout(content)
            layout.setSpacing(12)

            selection_group = QGroupBox("Profile Selection")
            selection_form = QFormLayout(selection_group)
            self._profile_selector = QComboBox()
            self._profile_selector.setMinimumContentsLength(14)
            self._profile_selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._configure_horizontal_growth(self._profile_selector)
            self._profile_selector.currentIndexChanged.connect(
                self._handle_profile_selection_request
            )
            selection_form.addRow("Profile", self._profile_selector)
            layout.addWidget(selection_group)

            current_group = QGroupBox("Current Profile")
            current_layout = QFormLayout(current_group)
            self._profiles_current_label = QLabel(self._current_profile_name)
            self._profiles_current_label.setWordWrap(True)
            current_note = QLabel(
                "Profiles store reusable plotting presets inside the current HDF5 source."
            )
            current_note.setWordWrap(True)
            current_layout.addRow("Active profile", self._profiles_current_label)
            current_layout.addRow("", current_note)
            layout.addWidget(current_group)

            manage_group = QGroupBox("Manage Profiles")
            manage_layout = QGridLayout(manage_group)

            def _page_button(label: str, callback: Callable[[], None]) -> QPushButton:
                button = QPushButton(label)
                button.clicked.connect(callback)
                return button

            manage_layout.addWidget(_page_button("New Profile", self._handle_new_profile), 0, 0)
            manage_layout.addWidget(_page_button("Rename", self._handle_rename_profile), 0, 1)
            manage_layout.addWidget(_page_button("Duplicate", self._handle_duplicate_profile), 1, 0)
            self._profile_delete_button = _page_button("Delete", self._handle_delete_profile)
            manage_layout.addWidget(self._profile_delete_button, 1, 1)
            manage_layout.addWidget(_page_button("Save Profile", self._handle_save), 2, 0)
            manage_layout.addWidget(_page_button("Reset Defaults", self._handle_reset), 2, 1)
            layout.addWidget(manage_group)

            transfer_group = QGroupBox("Transfer Profiles")
            transfer_layout = QGridLayout(transfer_group)
            transfer_layout.addWidget(
                _page_button("Import Profile", self._handle_import_json), 0, 0
            )
            transfer_layout.addWidget(
                _page_button("Export Profile JSON", self._handle_export_json),
                0,
                1,
            )
            layout.addWidget(transfer_group)
            layout.addStretch(1)
            self._sync_profile_selector()
            return page

        def _build_export_page(self) -> QWidget:
            page = QWidget()
            content = self._make_scrollable_tab(page)
            layout = QVBoxLayout(content)
            layout.setSpacing(12)

            export_group = QGroupBox("Figure Export")
            export_form = QFormLayout(export_group)
            self.transparent_mode = self._combo(_TOGGLE_MODES)
            export_form.addRow("Transparent save", self.transparent_mode)
            export_note = QLabel(
                "Export creates an image file from the current preview. It does not replace saving the current plot profile."
            )
            export_note.setWordWrap(True)
            export_form.addRow("", export_note)
            layout.addWidget(export_group)

            actions_group = QGroupBox("Export Actions")
            actions_layout = QGridLayout(actions_group)

            def _page_button(label: str, callback: Callable[[], None]) -> QPushButton:
                button = QPushButton(label)
                button.clicked.connect(callback)
                return button

            export_button = _page_button("Export Figure", self._handle_save_figure)
            export_button.setEnabled(on_save_figure is not None)
            actions_layout.addWidget(export_button, 0, 0)
            actions_layout.addWidget(_page_button("Refresh Preview", self._handle_preview), 0, 1)
            layout.addWidget(actions_group)
            layout.addStretch(1)
            return page

        def _build_advanced_page(self) -> QWidget:
            self._tab_advanced = QWidget()
            self._tab_advanced_content = self._make_scrollable_tab(self._tab_advanced)
            self._build_advanced_tab()
            return self._tab_advanced

        def _build_text_tab(self) -> None:
            form = QFormLayout(self._tab_text_content)
            title_row, self.title_text, title_lock = self._lockable_line(
                placeholder="Leave blank to hide the title"
            )
            x_label_row, self.x_label, x_label_lock = self._lockable_line(
                placeholder="Matplotlib mathtext supported, e.g. Distance ($A$)"
            )
            y_label_row, self.y_label, y_label_lock = self._lockable_line(
                placeholder="e.g. Density ($g/cm^3$)"
            )
            self._connect_lockable_line("title", self.title_text, title_lock)
            self._connect_lockable_line("x_label", self.x_label, x_label_lock)
            self._connect_lockable_line("y_label", self.y_label, y_label_lock)
            self.title_font = self._line()
            self.label_font = self._line()

            self._add_form_row(form, "Title", title_row)
            self._add_form_row(form, "X label", x_label_row)
            self._add_form_row(form, "Y label", y_label_row)
            self._add_form_row(form, "Title font", self.title_font)
            self._add_form_row(form, "Label font", self.label_font)
            math_hint = QLabel("Math labels: e.g. $cm^3$, $\\Delta G$, $\\rho$.")
            math_hint.setWordWrap(True)
            self._add_form_row(form, "", math_hint)
            self._title_rows = [(form, self.title_font)]

        def _build_legend_tab(self) -> None:
            form = QFormLayout(self._tab_legend_content)
            self.legend_mode = self._combo(_TOGGLE_MODES)
            self.legend_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.legend_title = self._line()
            self.legend_loc = self._combo(_LEGEND_LOCATIONS)
            self.legend_frame_mode = self._combo(_TOGGLE_MODES)
            self.legend_columns = self._line("1")
            self.legend_font = self._line()
            self._add_form_row(form, "Legend", self.legend_mode)
            self._add_form_row(form, "Legend title", self.legend_title)
            self._add_form_row(form, "Legend location", self.legend_loc)
            self._add_form_row(form, "Legend frame", self.legend_frame_mode)
            self._add_form_row(form, "Legend columns", self.legend_columns)
            self._add_form_row(form, "Legend font", self.legend_font)

            self._legend_rows = [
                (form, self.legend_title),
                (form, self.legend_loc),
                (form, self.legend_frame_mode),
                (form, self.legend_columns),
                (form, self.legend_font),
            ]

        def _build_axes_tab(self) -> None:
            layout = QVBoxLayout(self._tab_axes_content)
            self.x_scale = self._combo(("linear", "log", "symlog", "logit"))
            self.y_scale = self._combo(("linear", "log", "symlog", "logit"))
            top_form = QFormLayout()
            top_form.addRow("X scale", self.x_scale)
            top_form.addRow("Y scale", self.y_scale)
            layout.addLayout(top_form)

            limits = QGroupBox("Limits")
            limits_form = QFormLayout(limits)
            x_limits_row, self.x_min, self.x_max, x_limits_lock = self._lockable_pair()
            y_limits_row, self.y_min, self.y_max, y_limits_lock = self._lockable_pair()
            self._connect_lockable_line("x_lim", self.x_min, x_limits_lock)
            self.x_max.textEdited.connect(lambda _text: self._handle_synced_field_edit("x_lim"))
            self._connect_lockable_line("y_lim", self.y_min, y_limits_lock)
            self.y_max.textEdited.connect(lambda _text: self._handle_synced_field_edit("y_lim"))
            limits_form.addRow("X min / max", x_limits_row)
            limits_form.addRow("Y min / max", y_limits_row)
            layout.addWidget(limits)

            label_spacing = QGroupBox("Label Spacing")
            label_spacing_form = QFormLayout(label_spacing)
            x_label_pad_row, self.x_label_pad, x_label_pad_lock = self._lockable_line(
                placeholder="points"
            )
            y_label_pad_row, self.y_label_pad, y_label_pad_lock = self._lockable_line(
                placeholder="points"
            )
            self._connect_lockable_line("x_label_pad", self.x_label_pad, x_label_pad_lock)
            self._connect_lockable_line("y_label_pad", self.y_label_pad, y_label_pad_lock)
            label_spacing_form.addRow("X label pad", x_label_pad_row)
            label_spacing_form.addRow("Y label pad", y_label_pad_row)
            layout.addWidget(label_spacing)
            layout.addStretch(1)

        def _build_ticks_grid_tab(self) -> None:
            layout = QVBoxLayout(self._tab_ticks_grid_content)
            ticks = QGroupBox("Ticks")
            ticks_form = QFormLayout(ticks)
            self.ticks_visibility = self._combo(_TICK_VISIBILITY_MODES)
            self.ticks_visibility.currentTextChanged.connect(self._refresh_widget_states)
            self.tick_font = self._line()
            self.tick_direction = self._combo(_TICK_DIRECTIONS)
            self.tick_length = self._line("points")
            self.tick_width = self._line("points")
            self.minor_ticks_mode = self._combo(_MINOR_TICKS_MODES)
            x_ticks_row, self.x_ticks, x_ticks_lock = self._lockable_line(
                placeholder="e.g. 0, 1, 2"
            )
            y_ticks_row, self.y_ticks, y_ticks_lock = self._lockable_line(
                placeholder="e.g. 0, 5, 10"
            )
            self._connect_lockable_line("x_ticks", self.x_ticks, x_ticks_lock)
            self._connect_lockable_line("y_ticks", self.y_ticks, y_ticks_lock)
            self.x_tick_rotation = self._line("degrees")
            self.y_tick_rotation = self._line("degrees")
            ticks_form.addRow("Show ticks", self.ticks_visibility)
            ticks_form.addRow("X ticks", x_ticks_row)
            ticks_form.addRow("Y ticks", y_ticks_row)
            ticks_form.addRow("X rotation", self.x_tick_rotation)
            ticks_form.addRow("Y rotation", self.y_tick_rotation)
            ticks_form.addRow("Tick font", self.tick_font)
            ticks_form.addRow("Direction", self.tick_direction)
            ticks_form.addRow("Length", self.tick_length)
            ticks_form.addRow("Width", self.tick_width)
            ticks_form.addRow("Minor ticks", self.minor_ticks_mode)
            layout.addWidget(ticks)

            grid = QGroupBox("Grid")
            grid_form = QFormLayout(grid)
            self.grid_mode = self._combo(_TOGGLE_MODES)
            self.grid_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.grid_linestyle = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.grid_linewidth = self._line()
            self.grid_alpha = self._line()
            grid_color_row, self.grid_color = self._color_field(placeholder="#dddddd")
            self.grid_axis = self._combo(_GRID_AXES)
            self.grid_which = self._combo(_GRID_WHICH)
            grid_form.addRow("Show grid", self.grid_mode)
            grid_form.addRow("Line style", self.grid_linestyle)
            grid_form.addRow("Line width", self.grid_linewidth)
            grid_form.addRow("Alpha", self.grid_alpha)
            grid_form.addRow("Color", grid_color_row)
            grid_form.addRow("Axis", self.grid_axis)
            grid_form.addRow("Lines", self.grid_which)
            layout.addWidget(grid)
            layout.addStretch(1)

            self._ticks_rows = [
                (ticks_form, x_ticks_row),
                (ticks_form, y_ticks_row),
                (ticks_form, self.x_tick_rotation),
                (ticks_form, self.y_tick_rotation),
                (ticks_form, self.tick_font),
                (ticks_form, self.tick_direction),
                (ticks_form, self.tick_length),
                (ticks_form, self.tick_width),
                (ticks_form, self.minor_ticks_mode),
            ]
            self._grid_rows = [
                (grid_form, self.grid_linestyle),
                (grid_form, self.grid_linewidth),
                (grid_form, self.grid_alpha),
                (grid_form, grid_color_row),
                (grid_form, self.grid_axis),
                (grid_form, self.grid_which),
            ]

        def _build_lines_tab(self) -> None:
            layout = QVBoxLayout(self._tab_lines_content)
            lines = QGroupBox("Lines and Markers")
            lines_form = QFormLayout(lines)
            self.line_width = self._line()
            line_color_row, self.line_color = self._color_field(placeholder="#1f77b4")
            self.line_style = self._combo(("-", "--", "-.", ":", ""), editable=True)
            self.line_alpha = self._line("0.0 - 1.0")
            self.markers_mode = self._combo(_TOGGLE_MODES)
            self.markers_mode.currentTextChanged.connect(self._refresh_widget_states)
            self.marker_size = self._line("e.g. 5")
            lines_form.addRow("Line width", self.line_width)
            lines_form.addRow("Single-series color", line_color_row)
            lines_form.addRow("Line style", self.line_style)
            lines_form.addRow("Line alpha", self.line_alpha)
            lines_form.addRow("Show markers", self.markers_mode)
            lines_form.addRow("Marker size", self.marker_size)
            layout.addWidget(lines)
            layout.addStretch(1)
            self._single_series_line_color_row = (lines_form, line_color_row)
            self._marker_rows = [(lines_form, self.marker_size)]

        def _build_canvas_tab(self) -> None:
            form = QFormLayout(self._tab_canvas_content)
            self.fig_width = self._line()
            self.fig_height = self._line()
            self.dpi = self._line()
            self.font_family = self._line()
            figure_facecolor_row, self.figure_facecolor = self._color_field(placeholder="#ffffff")
            form.addRow("Figure width", self.fig_width)
            form.addRow("Figure height", self.fig_height)
            form.addRow("DPI", self.dpi)
            form.addRow("Font family", self.font_family)
            form.addRow("Figure facecolor", figure_facecolor_row)

        def _build_series_tab(self) -> None:
            layout = QVBoxLayout(self._tab_series_content)

            selector_row = QHBoxLayout()
            selector_row.addWidget(QLabel("Series"))
            self.series_selector = QComboBox()
            self.series_selector.setMinimumContentsLength(12)
            self.series_selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._configure_horizontal_growth(self.series_selector)
            self.series_selector.currentIndexChanged.connect(self._handle_series_selection_change)
            selector_row.addWidget(self.series_selector, stretch=1)
            enable_all_button = QPushButton("All on")
            enable_all_button.clicked.connect(lambda: self._set_all_series_enabled(True))
            disable_all_button = QPushButton("All off")
            disable_all_button.clicked.connect(lambda: self._set_all_series_enabled(False))
            selector_row.addWidget(enable_all_button)
            selector_row.addWidget(disable_all_button)
            layout.addLayout(selector_row)

            self.series_list = QListWidget()
            self.series_list.setAlternatingRowColors(True)
            self.series_list.setMinimumHeight(180)
            self.series_list.currentRowChanged.connect(self._handle_series_list_selection_change)
            self.series_list.itemChanged.connect(self._handle_series_list_item_changed)
            layout.addWidget(self.series_list)

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
            self.series_marker = self._combo(
                ("", "o", "s", "^", "v", "d", "x", "+", ".", "*"), editable=True
            )
            self.series_marker.currentTextChanged.connect(self._on_series_editor_changed)
            self.series_line_kwargs_json = QPlainTextEdit()
            self.series_line_kwargs_json.setPlaceholderText('{"linestyle": "--", "alpha": 0.8}')
            self.series_line_kwargs_json.setFixedHeight(84)
            self._configure_horizontal_growth(self.series_line_kwargs_json)
            self.series_line_kwargs_json.textChanged.connect(self._on_series_editor_changed)
            panel_form.addRow("Enabled", self.series_enabled)
            panel_form.addRow("Label", self.series_label)
            panel_form.addRow("Color", series_color_row)
            panel_form.addRow("Line width", self.series_line_width)
            panel_form.addRow("Marker", self.series_marker)
            panel_form.addRow("Extra line kwargs (JSON)", self.series_line_kwargs_json)
            layout.addWidget(panel)

            metadata_group = QGroupBox("Source Metadata")
            metadata_form = QFormLayout(metadata_group)
            self._series_meta_default_label = QLabel("")
            self._series_meta_default_label.setWordWrap(True)
            self._series_meta_source_name = QLabel("")
            self._series_meta_source_name.setWordWrap(True)
            self._series_meta_source_dir = QLabel("")
            self._series_meta_source_dir.setWordWrap(True)
            self._series_meta_series_id = QLabel("")
            self._series_meta_series_id.setWordWrap(True)
            metadata_form.addRow("Default label", self._series_meta_default_label)
            metadata_form.addRow("Source file", self._series_meta_source_name)
            metadata_form.addRow("Source directory", self._series_meta_source_dir)
            metadata_form.addRow("Series id", self._series_meta_series_id)
            layout.addWidget(metadata_group)

            # hint = QLabel(
            #     "Use the checklist to enable or disable multiple series directly, then edit the "
            #     "selected entry below for labels, colors, and line options. Source metadata stays "
            #     "read-only so display labels can stay short without losing provenance."
            # )
            # hint.setWordWrap(True)
            # layout.addWidget(hint)

            self._build_normalization_section(layout)
            layout.addStretch(1)

        def _build_analysis_data_sections(self, layout: QVBoxLayout) -> None:
            analysis = self._analysis_name
            if analysis is None:
                return

            if analysis in {"msd", "position"}:
                selection = QGroupBox("Profile Selection")
                selection_form = QFormLayout(selection)
                self.analysis_species = self._line("Leave blank to use file metadata")
                self.analysis_species.textChanged.connect(self._handle_series_identity_change)
                selection_form.addRow("Species", self.analysis_species)
                if analysis == "position":
                    self.analysis_axis = self._combo(("", "x", "y", "z"))
                    self.analysis_axis.currentTextChanged.connect(
                        self._handle_series_identity_change
                    )
                    selection_form.addRow("Axis", self.analysis_axis)
                layout.addWidget(selection)

            if analysis == "rdf":
                selection = QGroupBox("Profile Selection")
                selection_form = QFormLayout(selection)
                rdf_species_a_options = [
                    _PROFILE_FILTER_METADATA_LABEL,
                    *[
                        str(value)
                        for value in self._profile_filter_options.get("species_a", [])
                        if str(value).strip()
                    ],
                ]
                self.rdf_species_a = self._combo(tuple(rdf_species_a_options))
                self.rdf_species_a.currentTextChanged.connect(
                    self._handle_rdf_profile_selection_change
                )
                self.rdf_species_b = self._combo(
                    tuple(self._rdf_species_b_choices(None)),
                )
                self.rdf_species_b.currentTextChanged.connect(self._handle_series_identity_change)
                selection_form.addRow("Species A", self.rdf_species_a)
                selection_form.addRow("Species B", self.rdf_species_b)
                layout.addWidget(selection)

            if analysis == "coordination":
                selection = QGroupBox("Profile Selection")
                selection_form = QFormLayout(selection)
                coord_species_a_options = [
                    _PROFILE_FILTER_METADATA_LABEL,
                    *[
                        str(value)
                        for value in self._profile_filter_options.get("species_a", [])
                        if str(value).strip()
                    ],
                ]
                self.coord_species_a = self._combo(tuple(coord_species_a_options))
                self.coord_species_a.currentTextChanged.connect(
                    self._handle_coordination_profile_selection_change
                )
                self.coord_species_b = self._combo(
                    tuple(self._coordination_species_b_choices(None)),
                )
                self.coord_species_b.currentTextChanged.connect(
                    self._handle_coordination_profile_selection_change
                )
                self.analysis_axis = self._combo(tuple(self._coordination_axis_choices(None, None)))
                self.analysis_axis.currentTextChanged.connect(self._handle_series_identity_change)
                selection_form.addRow("Species A", self.coord_species_a)
                selection_form.addRow("Species B", self.coord_species_b)
                selection_form.addRow("Axis", self.analysis_axis)
                note = QLabel(
                    "Filters which stored coordination profile(s) are loaded from the HDF5. "
                )
                note.setWordWrap(True)
                note.setObjectName("sectionNote")
                selection_form.addRow(note)
                layout.addWidget(selection)

            if analysis == "density":
                view = QGroupBox("Density View")
                view_form = QFormLayout(view)
                self.density_x_mode = self._combo(("distance", "axis"))
                self.density_x_mode.currentTextChanged.connect(self._schedule_preview_update)
                self.density_quantity = self._combo(("mass", "number"))
                self.density_quantity.currentTextChanged.connect(self._schedule_preview_update)
                view_form.addRow("X values", self.density_x_mode)
                view_form.addRow("Quantity", self.density_quantity)
                layout.addWidget(view)

            if analysis == "position":
                view = QGroupBox("Position View")
                view_form = QFormLayout(view)
                self.position_component = self._combo(("distance", "x", "y", "z", "xy-z"))
                self.position_component.currentTextChanged.connect(
                    self._handle_series_identity_change
                )
                self.position_map_color = self._combo(("distance", "z"))
                self.position_map_color.currentTextChanged.connect(self._schedule_preview_update)
                self.position_time_axis = self._combo(("ps", "fs", "step", "frame"))
                self.position_time_axis.currentTextChanged.connect(self._schedule_preview_update)
                view_form.addRow("Component", self.position_component)
                view_form.addRow("Color by", self.position_map_color)
                view_form.addRow("Time axis", self.position_time_axis)
                self._position_map_color_row = (view_form, self.position_map_color)
                self._position_time_axis_row = (view_form, self.position_time_axis)
                layout.addWidget(view)

            if analysis == "coordination":
                view = QGroupBox("Coordination View")
                view_form = QFormLayout(view)
                self.coordination_component = self._combo(("distance", "time", "time-distance"))
                self.coordination_component.currentTextChanged.connect(
                    self._handle_series_identity_change
                )
                self.coordination_time_axis = self._combo(("ps", "fs", "step", "frame"))
                self.coordination_time_axis.currentTextChanged.connect(
                    self._schedule_preview_update
                )
                view_form.addRow("Component", self.coordination_component)
                view_form.addRow("Time axis", self.coordination_time_axis)
                self._coordination_time_axis_row = (view_form, self.coordination_time_axis)
                layout.addWidget(view)

        def _resize_list_with_defaults(
            self,
            values: list[Any],
            *,
            target_size: int,
            default_factory: Callable[[], Any],
        ) -> list[Any]:
            resized = list(values[:target_size])
            while len(resized) < target_size:
                resized.append(default_factory())
            return resized

        def _apply_series_defaults(
            self,
            labels: list[str],
            *,
            descriptors: list[dict[str, Any]] | None = None,
        ) -> None:
            if not labels and not descriptors:
                return

            selected_series = self._series_active_index
            selected_norm = self._normalization_active_index
            normalized_descriptors = _coerce_series_descriptors(descriptors)
            count = max(len(labels), len(normalized_descriptors))
            if count <= 0:
                return

            existing_by_id: dict[str, dict[str, Any]] = {}
            for index, descriptor in enumerate(self._series_descriptors_data):
                series_id = str(descriptor.get("series_id") or f"series:{index}")
                existing_by_id[series_id] = {
                    "label_override": self._series_label_overrides_data[index]
                    if index < len(self._series_label_overrides_data)
                    else "",
                    "color": self._series_colors_data[index]
                    if index < len(self._series_colors_data)
                    else "",
                    "enabled": self._series_enabled_data[index]
                    if index < len(self._series_enabled_data)
                    else True,
                    "line_width": self._series_line_widths_data[index]
                    if index < len(self._series_line_widths_data)
                    else "",
                    "marker": self._series_markers_data[index]
                    if index < len(self._series_markers_data)
                    else "",
                    "line_kwargs": self._series_line_kwargs_data[index]
                    if index < len(self._series_line_kwargs_data)
                    else "",
                    "normalization_mode": self._series_normalization_modes_data[index]
                    if index < len(self._series_normalization_modes_data)
                    else "none",
                    "normalization_value": self._series_normalization_values_data[index]
                    if index < len(self._series_normalization_values_data)
                    else "",
                    "normalization_x_ref": self._series_normalization_x_refs_data[index]
                    if index < len(self._series_normalization_x_refs_data)
                    else "",
                }

            new_descriptors: list[dict[str, Any]] = []
            new_default_labels: list[str] = []
            new_label_overrides: list[str] = []
            new_colors: list[str] = []
            new_enabled: list[bool] = []
            new_widths: list[str] = []
            new_markers: list[str] = []
            new_line_kwargs: list[str] = []
            new_norm_modes: list[str] = []
            new_norm_values: list[str] = []
            new_norm_x_refs: list[str] = []

            for index in range(count):
                descriptor = (
                    dict(normalized_descriptors[index])
                    if index < len(normalized_descriptors)
                    else {"series_id": f"series:{index}"}
                )
                default_label = str(descriptor.get("default_label") or "").strip()
                if not default_label and index < len(labels):
                    default_label = str(labels[index]).strip()
                if not default_label:
                    default_label = f"Series {index + 1}"
                descriptor["series_id"] = str(descriptor.get("series_id") or f"series:{index}")
                descriptor["default_label"] = default_label
                previous = existing_by_id.get(descriptor["series_id"], {})

                new_descriptors.append(descriptor)
                new_default_labels.append(default_label)
                new_label_overrides.append(str(previous.get("label_override") or "").strip())
                new_colors.append(str(previous.get("color") or "").strip())
                new_enabled.append(bool(previous.get("enabled", True)))
                new_widths.append(str(previous.get("line_width") or "").strip())
                new_markers.append(str(previous.get("marker") or "").strip())
                new_line_kwargs.append(str(previous.get("line_kwargs") or "").strip())
                new_norm_modes.append(str(previous.get("normalization_mode") or "none"))
                new_norm_values.append(str(previous.get("normalization_value") or "").strip())
                new_norm_x_refs.append(str(previous.get("normalization_x_ref") or "").strip())

            self._series_descriptors_data = new_descriptors
            self._series_labels_data = new_default_labels
            self._series_label_overrides_data = new_label_overrides
            self._series_colors_data = new_colors
            self._series_enabled_data = new_enabled
            self._series_line_widths_data = new_widths
            self._series_markers_data = new_markers
            self._series_line_kwargs_data = new_line_kwargs
            self._series_normalization_modes_data = new_norm_modes
            self._series_normalization_values_data = new_norm_values
            self._series_normalization_x_refs_data = new_norm_x_refs

            next_series_index = min(selected_series, count - 1)
            next_norm_index = min(selected_norm, count - 1)

            self._series_syncing = True
            self._normalization_syncing = True
            try:
                self._sync_series_selection_widgets(next_series_index)
                self._series_active_index = next_series_index
                self._load_series_into_editor(next_series_index)
                self._sync_normalization_selector_labels()
                self._normalization_active_index = next_norm_index
                if self.norm_series_selector.count() > 0:
                    self.norm_series_selector.setCurrentIndex(next_norm_index)
                self._load_normalization_into_editor(next_norm_index)
            finally:
                self._normalization_syncing = False
                self._series_syncing = False

            self._update_normalization_warning()

        def _series_display_text(self, index: int) -> str:
            return _format_series_display_text(
                index,
                self._effective_series_label(index),
                enabled=self._series_enabled_data[index],
            )

        def _series_descriptor(self, index: int) -> dict[str, Any]:
            if 0 <= index < len(self._series_descriptors_data):
                return dict(self._series_descriptors_data[index])
            return {
                "series_id": f"series:{index}",
                "default_label": self._series_labels_data[index]
                if 0 <= index < len(self._series_labels_data)
                else f"Series {index + 1}",
                "source_name": "",
                "source_directory": "",
                "source_path": "",
            }

        def _effective_series_label(self, index: int) -> str:
            override = ""
            if 0 <= index < len(self._series_label_overrides_data):
                override = self._series_label_overrides_data[index].strip()
            if override:
                return override
            if 0 <= index < len(self._series_labels_data):
                label = self._series_labels_data[index].strip()
                if label:
                    return label
            return f"Series {index + 1}"

        def _update_series_metadata_panel(self, index: int) -> None:
            descriptor = self._series_descriptor(index)
            if self._series_meta_default_label is not None:
                self._series_meta_default_label.setText(
                    str(descriptor.get("default_label") or f"Series {index + 1}")
                )
            if self._series_meta_source_name is not None:
                self._series_meta_source_name.setText(str(descriptor.get("source_name") or ""))
                source_path = str(descriptor.get("source_path") or "").strip()
                self._series_meta_source_name.setToolTip(source_path)
            if self._series_meta_source_dir is not None:
                source_directory = str(descriptor.get("source_directory") or "").strip()
                self._series_meta_source_dir.setText(source_directory or "Current session")
                self._series_meta_source_dir.setToolTip(source_directory)
            if self._series_meta_series_id is not None:
                self._series_meta_series_id.setText(str(descriptor.get("series_id") or ""))

        def _apply_series_list_item_visuals(self, item: Any, index: int) -> None:
            if item is None or index < 0 or index >= len(self._series_enabled_data):
                return
            enabled = self._series_enabled_data[index]
            item.setText(self._series_display_text(index))
            descriptor = self._series_descriptor(index)
            tooltip_lines = [
                f"Default label: {descriptor.get('default_label') or self._effective_series_label(index)}",
                f"Source file: {descriptor.get('source_name') or 'Current session'}",
            ]
            source_directory = str(descriptor.get("source_directory") or "").strip()
            if source_directory:
                tooltip_lines.append(f"Source directory: {source_directory}")
            tooltip_lines.append(f"Series id: {descriptor.get('series_id') or f'series:{index}'}")
            item.setToolTip("\n".join(tooltip_lines))
            font = item.font()
            font.setItalic(not enabled)
            item.setFont(font)
            base_color = self.series_list.palette().color(self.series_list.foregroundRole())
            text_color = QColor(base_color)
            if not enabled:
                text_color.setAlpha(150)
            item.setData(Qt.ItemDataRole.ForegroundRole, text_color)

        def _sync_series_selection_widgets(self, selected_index: int) -> None:
            self.series_selector.clear()
            self.series_list.clear()
            for index in range(len(self._series_labels_data)):
                label_text = self._series_display_text(index)
                self.series_selector.addItem(label_text)
                item = QListWidgetItem(label_text)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if self._series_enabled_data[index]
                    else Qt.CheckState.Unchecked
                )
                self._apply_series_list_item_visuals(item, index)
                self.series_list.addItem(item)
            if self.series_selector.count() > 0:
                self.series_selector.setCurrentIndex(selected_index)
            if self.series_list.count() > 0:
                self.series_list.setCurrentRow(selected_index)

        def _handle_series_identity_change(self, *_unused: object) -> None:
            self._refresh_widget_states()
            if self._suspend_preview_events:
                return
            if self._on_resolve_series_defaults is not None:
                try:
                    settings = self._collect_settings()
                    resolved = self._on_resolve_series_defaults(settings)
                    raw_labels = resolved.get("series_labels")
                    if isinstance(raw_labels, (list, tuple)):
                        labels = [str(label) for label in raw_labels]
                        if labels:
                            raw_descriptors = resolved.get("series_descriptors")
                            descriptors = (
                                list(raw_descriptors)
                                if isinstance(raw_descriptors, (list, tuple))
                                else None
                            )
                            self._apply_series_defaults(labels, descriptors=descriptors)
                except Exception:
                    pass
            self._schedule_preview_update()

        def _build_binning_section(self, layout: QVBoxLayout) -> None:
            binning_title = (
                "Time Sectioning (plot-only)"
                if self._analysis_name == "position"
                else (
                    "Distance / Time Binning (plot-only)"
                    if self._analysis_name == "coordination"
                    else "X Rebinning / Sectioning (plot-only)"
                )
            )
            binning = QGroupBox(binning_title)
            self._data_transform_group = binning
            binning_form = QFormLayout(binning)
            self.x_bin_width = self._line("Leave blank to use the data width")
            self.x_bin_width.textChanged.connect(self._refresh_widget_states)
            self.x_bin_reducer = self._combo(_BIN_REDUCERS)
            width_label = (
                "Time section width" if self._analysis_name == "position" else "Section width"
            )
            binning_form.addRow(width_label, self.x_bin_width)
            binning_form.addRow("Reducer", self.x_bin_reducer)
            self._x_bin_reducer_row = (binning_form, self.x_bin_reducer)
            layout.addWidget(binning)

        def _build_normalization_section(self, layout: QVBoxLayout) -> None:
            normalize_group = QGroupBox("Per-Series Normalization")
            self._normalization_group = normalize_group
            normalize_layout = QVBoxLayout(normalize_group)
            selector_row = QHBoxLayout()
            selector_row.addWidget(QLabel("Series"))
            self.norm_series_selector = QComboBox()
            self.norm_series_selector.setMinimumContentsLength(12)
            self.norm_series_selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._configure_horizontal_growth(self.norm_series_selector)
            self.norm_series_selector.currentIndexChanged.connect(
                self._handle_normalization_selection_change
            )
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
            self.normalization_warning.setObjectName("inlineWarning")
            self.normalization_warning.setWordWrap(True)
            self.normalization_warning.hide()
            normalize_layout.addWidget(self.normalization_warning)

            hint_text = "Normalization affects only the displayed figure. Stored HDF5 datasets remain unchanged."
            hint = QLabel(hint_text)
            hint.setWordWrap(True)
            normalize_layout.addWidget(hint)
            layout.addWidget(normalize_group)

        def _build_data_tab(self) -> None:
            layout = QVBoxLayout(self._tab_data_content)
            self._build_analysis_data_sections(layout)
            self._build_binning_section(layout)
            hint = QLabel(
                "Data controls decide what gets plotted. Series styling and normalization live in the Series workspace."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)

        def _build_advanced_tab(self) -> None:
            layout = QVBoxLayout(self._tab_advanced_content)

            def _json_editor(
                placeholder: str,
                *,
                sync_section: str | None = None,
            ) -> QPlainTextEdit:
                editor = QPlainTextEdit()
                editor.setPlaceholderText(placeholder)
                editor.setFixedHeight(84)
                self._configure_horizontal_growth(editor)
                if sync_section is None:
                    editor.textChanged.connect(self._schedule_preview_update)
                else:
                    editor.textChanged.connect(
                        lambda section=sync_section: self._handle_advanced_editor_changed(section)
                    )
                return editor

            rc_group = QGroupBox("Matplotlib rcParams")
            rc_form = QFormLayout(rc_group)
            self.matplotlib_rc_json = _json_editor('{"axes.facecolor": "#f8f8f8"}')
            rc_form.addRow("rcParams (JSON object)", self.matplotlib_rc_json)
            layout.addWidget(rc_group)

            render_group = QGroupBox("Figure / Axes / Layout")
            render_form = QFormLayout(render_group)
            self.figure_kwargs_json = _json_editor(
                '{"facecolor": "white"}',
                sync_section="figure_kwargs",
            )
            self.axes_kwargs_json = _json_editor(
                '{"xmargin": 0.02, "ymargin": 0.05}',
                sync_section="axes_kwargs",
            )
            self.tight_layout_kwargs_json = _json_editor('{"pad": 0.6}')
            self.savefig_kwargs_json = _json_editor(
                '{"transparent": false}',
                sync_section="savefig_kwargs",
            )
            render_form.addRow("Figure kwargs", self.figure_kwargs_json)
            render_form.addRow("Axes kwargs", self.axes_kwargs_json)
            render_form.addRow("tight_layout kwargs", self.tight_layout_kwargs_json)
            render_form.addRow("savefig kwargs", self.savefig_kwargs_json)
            layout.addWidget(render_group)

            style_group = QGroupBox("Raw Matplotlib kwargs")
            style_form = QFormLayout(style_group)
            self.legend_kwargs_json = _json_editor(
                '{"frameon": true}',
                sync_section="legend_kwargs",
            )
            self.grid_kwargs_json = _json_editor(
                '{"color": "#dddddd"}',
                sync_section="grid_kwargs",
            )
            self.tick_params_kwargs_json = _json_editor(
                '{"direction": "out"}',
                sync_section="tick_params_kwargs",
            )
            self.line_kwargs_json = _json_editor(
                '{"linestyle": "-", "alpha": 1.0}',
                sync_section="line_kwargs",
            )
            style_form.addRow("Legend kwargs", self.legend_kwargs_json)
            style_form.addRow("Grid kwargs", self.grid_kwargs_json)
            style_form.addRow("Tick params kwargs", self.tick_params_kwargs_json)
            style_form.addRow("Global line kwargs", self.line_kwargs_json)
            layout.addWidget(style_group)

            hint = QLabel(
                "Advanced JSON fields map directly onto Matplotlib API kwargs. "
                "Recognized keys stay synchronized with the standard controls; extra keys remain available here."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
            layout.addStretch(1)

        def _series_count_from_settings(self, settings: dict[str, Any]) -> int:
            candidates = [1]
            raw_count = settings.get("series_count")
            if isinstance(raw_count, int) and raw_count > 0:
                candidates.append(raw_count)
            raw_descriptors = settings.get("series_descriptors")
            if isinstance(raw_descriptors, (list, tuple)):
                candidates.append(len(raw_descriptors))
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
            descriptors = _coerce_series_descriptors(settings.get("series_descriptors"))
            if descriptors:
                count = max(count, len(descriptors))
            overrides_by_id = _coerce_series_overrides(settings.get("series_overrides"))
            raw_labels = settings.get("series_labels")
            raw_colors = settings.get("line_colors")
            raw_enabled = settings.get("series_enabled")
            raw_widths = settings.get("series_line_widths")
            raw_markers = settings.get("series_markers")
            raw_line_kwargs = settings.get("series_line_kwargs")
            raw_norm_modes = settings.get("series_normalization_modes")
            raw_norm_values = settings.get("series_normalization_values")
            raw_norm_x_refs = settings.get("series_normalization_x_refs")

            self._series_descriptors_data = []
            self._series_labels_data = []
            self._series_label_overrides_data = []
            self._series_colors_data = []
            self._series_enabled_data = []
            self._series_line_widths_data = []
            self._series_markers_data = []
            self._series_line_kwargs_data = []
            self._series_normalization_modes_data = []
            self._series_normalization_values_data = []
            self._series_normalization_x_refs_data = []

            for index in range(count):
                descriptor = (
                    dict(descriptors[index])
                    if index < len(descriptors)
                    else {"series_id": f"series:{index}"}
                )
                fallback_label = f"Series {index + 1}"
                default_label = str(descriptor.get("default_label") or "").strip()
                if (
                    not default_label
                    and isinstance(raw_labels, (list, tuple))
                    and index < len(raw_labels)
                ):
                    token = str(raw_labels[index]).strip()
                    if token:
                        default_label = token
                if not default_label:
                    default_label = fallback_label
                descriptor["series_id"] = str(descriptor.get("series_id") or f"series:{index}")
                descriptor["default_label"] = default_label
                self._series_descriptors_data.append(descriptor)
                self._series_labels_data.append(default_label)

                override = ""
                series_override = overrides_by_id.get(descriptor["series_id"])
                if isinstance(series_override, dict):
                    raw_override = series_override.get("label_override")
                    if raw_override is not None:
                        override = str(raw_override).strip()
                elif isinstance(raw_labels, (list, tuple)) and index < len(raw_labels):
                    token = str(raw_labels[index]).strip()
                    if token and token != default_label:
                        override = token
                self._series_label_overrides_data.append(override)

                color = ""
                if isinstance(series_override, dict) and series_override.get("color") is not None:
                    color = str(series_override.get("color")).strip()
                elif isinstance(raw_colors, (list, tuple)) and index < len(raw_colors):
                    color = str(raw_colors[index]).strip()
                self._series_colors_data.append(color)

                enabled = True
                if isinstance(series_override, dict) and "enabled" in series_override:
                    enabled = self._coerce_series_bool(series_override.get("enabled"), default=True)
                elif isinstance(raw_enabled, (list, tuple)) and index < len(raw_enabled):
                    enabled = self._coerce_series_bool(raw_enabled[index], default=True)
                self._series_enabled_data.append(enabled)

                width = ""
                if (
                    isinstance(series_override, dict)
                    and series_override.get("line_width") is not None
                ):
                    width = str(series_override.get("line_width")).strip()
                elif isinstance(raw_widths, (list, tuple)) and index < len(raw_widths):
                    width = str(raw_widths[index]).strip()
                    if width.lower() == "none":
                        width = ""
                self._series_line_widths_data.append(width)

                marker = ""
                if isinstance(series_override, dict) and series_override.get("marker") is not None:
                    marker = str(series_override.get("marker")).strip()
                elif isinstance(raw_markers, (list, tuple)) and index < len(raw_markers):
                    marker = str(raw_markers[index]).strip()
                    if marker.lower() == "none":
                        marker = ""
                self._series_markers_data.append(marker)

                line_kwargs_text = ""
                if isinstance(series_override, dict) and isinstance(
                    series_override.get("line_kwargs"),
                    dict,
                ):
                    line_kwargs_text = _format_json_block(series_override.get("line_kwargs"))
                elif isinstance(raw_line_kwargs, (list, tuple)) and index < len(raw_line_kwargs):
                    value = raw_line_kwargs[index]
                    if isinstance(value, dict):
                        line_kwargs_text = _format_json_block(value)
                self._series_line_kwargs_data.append(line_kwargs_text)

                mode = "none"
                if isinstance(series_override, dict):
                    token = str(series_override.get("normalization_mode") or "").strip().lower()
                    if token in _NORMALIZATION_MODES:
                        mode = token
                elif isinstance(raw_norm_modes, (list, tuple)) and index < len(raw_norm_modes):
                    token = str(raw_norm_modes[index]).strip().lower()
                    if token in _NORMALIZATION_MODES:
                        mode = token
                self._series_normalization_modes_data.append(mode)

                norm_value = ""
                if (
                    isinstance(series_override, dict)
                    and series_override.get("normalization_value") is not None
                ):
                    norm_value = str(series_override.get("normalization_value")).strip()
                elif isinstance(raw_norm_values, (list, tuple)) and index < len(raw_norm_values):
                    raw_value = raw_norm_values[index]
                    if raw_value is not None:
                        norm_value = str(raw_value).strip()
                self._series_normalization_values_data.append(norm_value)

                norm_x_ref = ""
                if (
                    isinstance(series_override, dict)
                    and series_override.get("normalization_x_ref") is not None
                ):
                    norm_x_ref = str(series_override.get("normalization_x_ref")).strip()
                elif isinstance(raw_norm_x_refs, (list, tuple)) and index < len(raw_norm_x_refs):
                    raw_x_ref = raw_norm_x_refs[index]
                    if raw_x_ref is not None:
                        norm_x_ref = str(raw_x_ref).strip()
                self._series_normalization_x_refs_data.append(norm_x_ref)

            self._series_syncing = True
            try:
                selected = self._series_active_index if self._series_active_index < count else 0
                self._sync_series_selection_widgets(selected)
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
                self._set_combo_value(
                    self.series_enabled, "on" if self._series_enabled_data[index] else "off"
                )
                self.series_label.setPlaceholderText(self._series_labels_data[index])
                self.series_label.setText(self._series_label_overrides_data[index])
                self.series_color.setText(self._series_colors_data[index])
                self.series_line_width.setText(self._series_line_widths_data[index])
                self._set_combo_value(self.series_marker, self._series_markers_data[index])
                self.series_line_kwargs_json.setPlainText(self._series_line_kwargs_data[index])
            finally:
                self._series_syncing = False
            self._update_series_metadata_panel(index)

        def _persist_series_editor(self, index: int) -> None:
            if index < 0 or index >= len(self._series_labels_data):
                return
            label_value = self.series_label.text().strip()
            self._series_label_overrides_data[index] = label_value
            self._series_colors_data[index] = self.series_color.text().strip()
            self._series_enabled_data[index] = (
                self.series_enabled.currentText().strip().lower() != "off"
            )
            self._series_line_widths_data[index] = self.series_line_width.text().strip()
            self._series_markers_data[index] = self.series_marker.currentText().strip()
            self._series_line_kwargs_data[index] = (
                self.series_line_kwargs_json.toPlainText().strip()
            )
            label_text = self._series_display_text(index)
            self._series_syncing = True
            try:
                self.series_selector.setItemText(index, label_text)
                item = self.series_list.item(index)
                if item is not None:
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if self._series_enabled_data[index]
                        else Qt.CheckState.Unchecked
                    )
                    self._apply_series_list_item_visuals(item, index)
            finally:
                self._series_syncing = False
            if self.norm_series_selector.count() > index:
                self.norm_series_selector.setItemText(index, label_text)
            self._update_series_metadata_panel(index)

        def _handle_series_selection_change(self, index: int) -> None:
            if self._series_syncing or index < 0:
                return
            self._persist_series_editor(self._series_active_index)
            self._series_active_index = index
            self._series_syncing = True
            try:
                if self.series_list.currentRow() != index:
                    self.series_list.setCurrentRow(index)
            finally:
                self._series_syncing = False
            self._load_series_into_editor(index)

        def _handle_series_list_selection_change(self, index: int) -> None:
            if self._series_syncing or index < 0:
                return
            self._persist_series_editor(self._series_active_index)
            self._series_active_index = index
            self._series_syncing = True
            try:
                if self.series_selector.currentIndex() != index:
                    self.series_selector.setCurrentIndex(index)
            finally:
                self._series_syncing = False
            self._load_series_into_editor(index)

        def _handle_series_list_item_changed(self, item: Any) -> None:
            if self._series_syncing:
                return
            index = self.series_list.row(item)
            if index < 0 or index >= len(self._series_enabled_data):
                return
            enabled = item.checkState() == Qt.CheckState.Checked
            self._series_enabled_data[index] = enabled
            self._series_syncing = True
            try:
                self._apply_series_list_item_visuals(item, index)
            finally:
                self._series_syncing = False
            if index == self._series_active_index:
                self._series_syncing = True
                try:
                    self._set_combo_value(self.series_enabled, "on" if enabled else "off")
                finally:
                    self._series_syncing = False
            self._schedule_preview_update()

        def _set_all_series_enabled(self, enabled: bool) -> None:
            if not self._series_enabled_data:
                return
            self._persist_series_editor(self._series_active_index)
            self._series_enabled_data = [enabled] * len(self._series_enabled_data)
            self._series_syncing = True
            try:
                for index in range(self.series_list.count()):
                    item = self.series_list.item(index)
                    if item is not None:
                        item.setCheckState(
                            Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
                        )
                        self._apply_series_list_item_visuals(item, index)
                self._set_combo_value(self.series_enabled, "on" if enabled else "off")
            finally:
                self._series_syncing = False
            self._schedule_preview_update()

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
                for index in range(len(self._series_labels_data)):
                    self.norm_series_selector.addItem(
                        f"{index + 1}: {self._effective_series_label(index)}"
                    )
                self.norm_series_selector.setCurrentIndex(selected)
                self._normalization_active_index = selected
            finally:
                self._normalization_syncing = False

        def _initialize_normalization_data(self, settings: dict[str, Any]) -> None:
            count = len(self._series_labels_data)
            if (
                settings.get("series_overrides") is not None
                and len(self._series_normalization_modes_data) == count
                and len(self._series_normalization_values_data) == count
                and len(self._series_normalization_x_refs_data) == count
            ):
                self._sync_normalization_selector_labels()
                self._normalization_syncing = True
                try:
                    self._load_normalization_into_editor(self._normalization_active_index)
                finally:
                    self._normalization_syncing = False
                self._update_normalization_warning()
                return
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
                self.normalization_warning.show()
                return
            self.normalization_warning.setText("")
            self.normalization_warning.hide()

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
                self.x_label_pad,
                self.y_label_pad,
                self.fig_width,
                self.fig_height,
                self.dpi,
                self.font_family,
                self.title_font,
                self.label_font,
                self.tick_font,
                self.legend_font,
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
                self.legend_mode,
                self.legend_loc,
                self.legend_frame_mode,
                self.ticks_visibility,
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
                self._refresh_shell_state()
                return
            self._preview_status.setText("Auto update enabled.")
            self._refresh_shell_state()
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
                    render_state = on_preview(settings)
                    if isinstance(render_state, dict) and render_state:
                        self._apply_preview_state_to_synced_fields(render_state)
                    self._status_label.setText("Preview opened.")
                    self._preview_status.setText("External preview opened.")
                    self._refresh_shell_state()
                    return True
                except Exception as exc:
                    if interactive:
                        self._report_error("Preview failed", exc)
                    else:
                        self._preview_status.setText(f"Preview paused: {exc}")
                        self._refresh_shell_state()
                    return False

            try:
                settings = self._collect_settings()
            except Exception as exc:
                if interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_status.setText(f"Preview paused: {exc}")
                    self._refresh_shell_state()
                return False

            try:
                save_result = on_save_figure(settings, str(self._preview_image_path))
                render_state = None
                if isinstance(save_result, tuple):
                    _message, render_state = save_result
                elif isinstance(save_result, dict):
                    render_state = save_result
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_state_to_synced_fields(render_state)
                pixmap = QPixmap(str(self._preview_image_path))
                if pixmap.isNull():
                    raise RuntimeError("Could not load rendered preview image.")
                self._preview_pixmap = pixmap
                self._refresh_preview_pixmap()
                self._status_label.setText("Preview updated.")
                self._preview_status.setText("Preview updated.")
                self._refresh_shell_state()
                return True
            except Exception as exc:
                if interactive:
                    self._report_error("Preview failed", exc)
                else:
                    self._preview_status.setText(f"Preview paused: {exc}")
                    self._refresh_shell_state()
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

        def _set_combo_items(
            self,
            widget: QComboBox,
            values: list[str],
            *,
            preferred_value: str | None = None,
        ) -> None:
            current_value = widget.currentText()
            target_value = current_value if preferred_value is None else preferred_value
            widget.blockSignals(True)
            try:
                widget.clear()
                widget.addItems(values)
                self._set_combo_value(widget, target_value)
            finally:
                widget.blockSignals(False)

        def _profile_filter_display_value(self, value: str | None, *, default_label: str) -> str:
            token = str(value or "").strip()
            return token or default_label

        def _selected_profile_filter_value(
            self,
            widget: QComboBox,
            *,
            default_label: str,
        ) -> str | None:
            token = widget.currentText().strip()
            return None if token == default_label or token == "" else token

        def _rdf_species_b_choices(self, species_a: str | None) -> list[str]:
            mapping = self._profile_filter_options.get("species_b_by_species_a", {})
            if not isinstance(mapping, dict):
                return [_PROFILE_FILTER_SPECIES_B_AUTO_LABEL]
            key = "" if species_a is None else str(species_a)
            values = mapping.get(key)
            if not isinstance(values, list):
                values = mapping.get("", [])
            resolved = [str(value).strip() for value in values if str(value).strip()]
            return [_PROFILE_FILTER_SPECIES_B_AUTO_LABEL, *resolved]

        def _coordination_species_b_choices(self, species_a: str | None) -> list[str]:
            return self._rdf_species_b_choices(species_a)

        def _coordination_axis_choices(
            self,
            species_a: str | None,
            species_b: str | None,
        ) -> list[str]:
            axes_by_pair = self._profile_filter_options.get("axes_by_species_pair", {})
            if not isinstance(axes_by_pair, dict):
                return ["", "x", "y", "z"]
            species_a_key = "" if species_a is None else str(species_a)
            species_b_key = "" if species_b is None else str(species_b)
            axis_values: list[str] | None = None
            pair_mapping = axes_by_pair.get(species_a_key)
            if isinstance(pair_mapping, dict):
                candidate = pair_mapping.get(species_b_key)
                if isinstance(candidate, list):
                    axis_values = candidate
            if axis_values is None:
                global_mapping = axes_by_pair.get("", {})
                if isinstance(global_mapping, dict):
                    candidate = global_mapping.get("", [])
                    if isinstance(candidate, list):
                        axis_values = candidate
            resolved = [
                str(value).strip().lower() for value in (axis_values or []) if str(value).strip()
            ]
            resolved = [value for value in resolved if value in {"x", "y", "z"}]
            return ["", *resolved] if resolved else ["", "x", "y", "z"]

        def _handle_rdf_profile_selection_change(self, *_unused: object) -> None:
            if not hasattr(self, "rdf_species_a") or not hasattr(self, "rdf_species_b"):
                return
            species_a = self._selected_profile_filter_value(
                self.rdf_species_a,
                default_label=_PROFILE_FILTER_METADATA_LABEL,
            )
            current_species_b = self.rdf_species_b.currentText()
            self._set_combo_items(
                self.rdf_species_b,
                self._rdf_species_b_choices(species_a),
                preferred_value=current_species_b,
            )
            self._handle_series_identity_change()

        def _handle_coordination_profile_selection_change(self, *_unused: object) -> None:
            if not hasattr(self, "coord_species_a") or not hasattr(self, "coord_species_b"):
                return
            species_a = self._selected_profile_filter_value(
                self.coord_species_a,
                default_label=_PROFILE_FILTER_METADATA_LABEL,
            )
            current_species_b = self.coord_species_b.currentText()
            self._set_combo_items(
                self.coord_species_b,
                self._coordination_species_b_choices(species_a),
                preferred_value=current_species_b,
            )
            if hasattr(self, "analysis_axis"):
                species_b = self._selected_profile_filter_value(
                    self.coord_species_b,
                    default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                )
                current_axis = self.analysis_axis.currentText()
                self._set_combo_items(
                    self.analysis_axis,
                    self._coordination_axis_choices(species_a, species_b),
                    preferred_value=current_axis,
                )
            self._handle_series_identity_change()

        def _populate(self, settings: dict[str, Any]) -> None:
            synced_locks = _derive_synced_field_locks(settings)
            title_visible = settings.get("title_visible")
            if title_visible is False:
                self.title_text.setText("")
            else:
                self.title_text.setText(str(settings.get("title") or ""))
            self.x_label.setText(str(settings.get("x_label") or ""))
            self.y_label.setText(str(settings.get("y_label") or ""))

            self._set_combo_value(
                self.legend_mode,
                _toggle_to_mode(settings.get("legend"), auto_mode="on"),
            )
            self.legend_title.setText(str(settings.get("legend_title") or ""))
            self._set_combo_value(self.legend_loc, str(settings.get("legend_loc") or "best"))
            self._set_combo_value(
                self.legend_frame_mode,
                _extract_dict_mode(
                    settings,
                    key="legend_kwargs",
                    nested_key="frameon",
                    auto_mode="on",
                ),
            )
            self.legend_columns.setText(
                _extract_dict_text(settings, key="legend_kwargs", nested_key="ncols")
            )
            legend_font_size = settings.get("legend_font_size")
            if legend_font_size is None:
                legend_font_size = _extract_dict_text(
                    settings,
                    key="legend_kwargs",
                    nested_key="fontsize",
                )
            self.legend_font.setText("" if legend_font_size is None else str(legend_font_size))

            self._set_combo_value(
                self.grid_mode,
                _toggle_to_mode(settings.get("grid"), auto_mode="on"),
            )
            self._set_combo_value(
                self.markers_mode,
                _toggle_to_mode(settings.get("markers"), auto_mode="off"),
            )
            tick_params_settings = settings.get("tick_params_kwargs")
            tick_axis_mode = "both"
            minor_ticks_mode = "off"
            if isinstance(tick_params_settings, dict):
                raw_tick_axis = (
                    str(
                        tick_params_settings.get(
                            "_ticks_axis", tick_params_settings.get("axis", "both")
                        )
                    )
                    .strip()
                    .lower()
                )
                if raw_tick_axis in _TICK_AXES:
                    tick_axis_mode = raw_tick_axis
                raw_minor_mode = (
                    str(tick_params_settings.get("_minor_ticks_mode", "off")).strip().lower()
                )
                if raw_minor_mode in _MINOR_TICKS_MODES:
                    minor_ticks_mode = raw_minor_mode
            ticks_visibility_mode = "none" if settings.get("ticks") is False else tick_axis_mode
            self._set_combo_value(self.ticks_visibility, ticks_visibility_mode)

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
            self.x_label_pad.setText(
                "" if settings.get("x_label_pad") is None else str(settings.get("x_label_pad"))
            )
            self.y_label_pad.setText(
                "" if settings.get("y_label_pad") is None else str(settings.get("y_label_pad"))
            )

            self.fig_width.setText(
                _extract_figsize_dimension(settings, index=0, fallback=defaults.figure_size[0])
            )
            self.fig_height.setText(
                _extract_figsize_dimension(settings, index=1, fallback=defaults.figure_size[1])
            )
            self.dpi.setText(str(settings.get("dpi") or defaults.dpi))
            self.font_family.setText(str(settings.get("font_family") or ""))
            self.figure_facecolor.setText(
                _extract_dict_text(settings, key="figure_kwargs", nested_key="facecolor")
            )
            self._set_combo_value(
                self.transparent_mode,
                _extract_dict_mode(
                    settings,
                    key="savefig_kwargs",
                    nested_key="transparent",
                    auto_mode="off",
                ),
            )
            self.title_font.setText(
                str(settings.get("title_font_size") or defaults.title_font_size)
            )
            self.label_font.setText(
                str(settings.get("label_font_size") or defaults.label_font_size)
            )
            self.tick_font.setText(str(settings.get("tick_font_size") or defaults.tick_font_size))
            self.line_width.setText(str(settings.get("line_width") or defaults.line_width))
            single_series_color = settings.get("line_color")
            if single_series_color is None:
                raw_line_colors = settings.get("line_colors")
                if isinstance(raw_line_colors, (list, tuple)) and len(raw_line_colors) == 1:
                    single_series_color = raw_line_colors[0]
            self.line_color.setText(str(single_series_color or ""))
            self._set_combo_value(
                self.line_style,
                _extract_dict_text(settings, key="line_kwargs", nested_key="linestyle"),
            )
            self.line_alpha.setText(
                _extract_dict_text(settings, key="line_kwargs", nested_key="alpha")
            )
            self.marker_size.setText(
                _extract_dict_text(settings, key="line_kwargs", nested_key="markersize")
            )
            self._set_combo_value(self.grid_linestyle, str(settings.get("grid_linestyle") or ""))
            self.grid_linewidth.setText(
                str(settings.get("grid_linewidth") or defaults.grid_linewidth)
            )
            self.grid_alpha.setText(str(settings.get("grid_alpha") or defaults.grid_alpha))
            self.grid_color.setText(
                _extract_dict_text(settings, key="grid_kwargs", nested_key="color")
            )
            self._set_combo_value(
                self.grid_axis,
                str(_extract_dict_value(settings, key="grid_kwargs", nested_key="axis") or "both"),
            )
            self._set_combo_value(
                self.grid_which,
                str(
                    _extract_dict_value(settings, key="grid_kwargs", nested_key="which") or "major"
                ),
            )
            self._set_combo_value(
                self.tick_direction,
                str(
                    _extract_dict_value(settings, key="tick_params_kwargs", nested_key="direction")
                    or "out"
                ),
            )
            self.tick_length.setText(
                _extract_dict_text(settings, key="tick_params_kwargs", nested_key="length")
            )
            self.tick_width.setText(
                _extract_dict_text(settings, key="tick_params_kwargs", nested_key="width")
            )
            self._set_combo_value(self.minor_ticks_mode, minor_ticks_mode)

            if hasattr(self, "analysis_species"):
                self.analysis_species.setText(str(settings.get("species") or ""))
            if hasattr(self, "analysis_axis"):
                self._set_combo_value(self.analysis_axis, str(settings.get("axis") or ""))
            if hasattr(self, "density_x_mode"):
                self._set_combo_value(
                    self.density_x_mode,
                    str(settings.get("x_mode") or "distance"),
                )
            if hasattr(self, "density_quantity"):
                self._set_combo_value(
                    self.density_quantity,
                    str(settings.get("quantity") or "mass"),
                )
            if hasattr(self, "rdf_species_a"):
                self._set_combo_value(
                    self.rdf_species_a,
                    self._profile_filter_display_value(
                        settings.get("species_a"),
                        default_label=_PROFILE_FILTER_METADATA_LABEL,
                    ),
                )
            if hasattr(self, "rdf_species_b"):
                species_a_value = settings.get("species_a")
                self._set_combo_items(
                    self.rdf_species_b,
                    self._rdf_species_b_choices(
                        None if species_a_value in {None, ""} else str(species_a_value)
                    ),
                    preferred_value=self._profile_filter_display_value(
                        settings.get("species_b"),
                        default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                    ),
                )
            if hasattr(self, "coord_species_a"):
                self._set_combo_value(
                    self.coord_species_a,
                    self._profile_filter_display_value(
                        settings.get("species_a"),
                        default_label=_PROFILE_FILTER_METADATA_LABEL,
                    ),
                )
            if hasattr(self, "coord_species_b"):
                species_a_value = settings.get("species_a")
                species_b_value = settings.get("species_b")
                self._set_combo_items(
                    self.coord_species_b,
                    self._coordination_species_b_choices(
                        None if species_a_value in {None, ""} else str(species_a_value)
                    ),
                    preferred_value=self._profile_filter_display_value(
                        species_b_value,
                        default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                    ),
                )
                if hasattr(self, "analysis_axis") and self._analysis_name == "coordination":
                    self._set_combo_items(
                        self.analysis_axis,
                        self._coordination_axis_choices(
                            None if species_a_value in {None, ""} else str(species_a_value),
                            None if species_b_value in {None, ""} else str(species_b_value),
                        ),
                        preferred_value=str(settings.get("axis") or ""),
                    )
            if hasattr(self, "position_component"):
                self._set_combo_value(
                    self.position_component,
                    str(settings.get("component") or "distance"),
                )
            if hasattr(self, "position_map_color"):
                self._set_combo_value(
                    self.position_map_color,
                    str(settings.get("map_color") or "distance"),
                )
            if hasattr(self, "position_time_axis"):
                self._set_combo_value(
                    self.position_time_axis,
                    str(settings.get("time_axis") or "ps"),
                )
            if hasattr(self, "coordination_component"):
                self._set_combo_value(
                    self.coordination_component,
                    str(settings.get("component") or "distance"),
                )
            if hasattr(self, "coordination_time_axis"):
                self._set_combo_value(
                    self.coordination_time_axis,
                    str(settings.get("time_axis") or "ps"),
                )
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
            self.savefig_kwargs_json.setPlainText(
                _format_json_block(settings.get("savefig_kwargs"))
            )
            for key in _SYNCED_FIELD_KEYS:
                self._set_synced_field_lock(key, synced_locks.get(key, False))
            self._apply_preview_state_to_synced_fields(settings)

        def _refresh_widget_states(self, *_unused: object) -> None:
            title_enabled = bool(self.title_text.text().strip())
            legend_enabled = self.legend_mode.currentText().strip().lower() != "off"
            grid_enabled = self.grid_mode.currentText().strip().lower() != "off"
            ticks_mode = self.ticks_visibility.currentText().strip().lower()
            ticks_enabled = ticks_mode != "none"
            markers_enabled = self.markers_mode.currentText().strip().lower() != "off"
            rebin_enabled = (
                bool(self.x_bin_width.text().strip()) if hasattr(self, "x_bin_width") else False
            )
            norm_mode = (
                self.norm_mode.currentText().strip().lower()
                if hasattr(self, "norm_mode")
                else "none"
            )
            norm_enabled = norm_mode != "none"
            norm_x_ref_enabled = norm_mode == "value_at_x"
            position_component = (
                self.position_component.currentText().strip().lower()
                if hasattr(self, "position_component")
                else ""
            )
            position_xy_projection = position_component == "xy-z"
            coordination_component = (
                self.coordination_component.currentText().strip().lower()
                if hasattr(self, "coordination_component")
                else ""
            )
            coordination_time_distance = coordination_component == "time-distance"
            coordination_distance = coordination_component == "distance"

            self._set_rows_enabled(self._title_rows, title_enabled)
            self._set_rows_enabled(self._legend_rows, legend_enabled)
            self._set_rows_enabled(self._grid_rows, grid_enabled)
            self._set_rows_enabled(self._marker_rows, markers_enabled)
            for form, field in self._ticks_rows:
                self._set_form_row_enabled(form, field, ticks_enabled)

            if self._single_series_line_color_row is not None:
                self._set_form_row_enabled(
                    self._single_series_line_color_row[0],
                    self._single_series_line_color_row[1],
                    len(self._series_labels_data) <= 1,
                )
            if self._position_map_color_row is not None:
                self._set_form_row_enabled(
                    self._position_map_color_row[0],
                    self._position_map_color_row[1],
                    position_xy_projection,
                )
            if self._position_time_axis_row is not None:
                self._set_form_row_enabled(
                    self._position_time_axis_row[0],
                    self._position_time_axis_row[1],
                    not position_xy_projection,
                )
            if self._coordination_time_axis_row is not None:
                self._set_form_row_enabled(
                    self._coordination_time_axis_row[0],
                    self._coordination_time_axis_row[1],
                    not coordination_distance,
                )
            if self._data_transform_group is not None and self._analysis_name == "position":
                self._data_transform_group.setEnabled(not position_xy_projection)
                self._data_transform_group.setToolTip(
                    ""
                    if not position_xy_projection
                    else "Time sectioning is unavailable for the xy-z projection."
                )
            if self._normalization_group is not None and self._analysis_name == "position":
                self._normalization_group.setEnabled(not position_xy_projection)
                self._normalization_group.setToolTip(
                    ""
                    if not position_xy_projection
                    else "Normalization is unavailable for the xy-z projection."
                )
            if self._data_transform_group is not None and self._analysis_name == "coordination":
                self._data_transform_group.setEnabled(not coordination_time_distance)
                self._data_transform_group.setToolTip(
                    ""
                    if not coordination_time_distance
                    else "Sectioning is unavailable for the time-distance view."
                )
            if self._normalization_group is not None and self._analysis_name == "coordination":
                self._normalization_group.setEnabled(not coordination_time_distance)
                self._normalization_group.setToolTip(
                    ""
                    if not coordination_time_distance
                    else "Normalization is unavailable for the time-distance view."
                )
            if self._x_bin_reducer_row is not None:
                self._set_form_row_enabled(
                    self._x_bin_reducer_row[0],
                    self._x_bin_reducer_row[1],
                    rebin_enabled,
                )
            if self._norm_value_row is not None:
                self._set_form_row_enabled(
                    self._norm_value_row[0],
                    self._norm_value_row[1],
                    norm_enabled,
                )
            if self._norm_x_ref_row is not None:
                self._set_form_row_enabled(
                    self._norm_x_ref_row[0],
                    self._norm_x_ref_row[1],
                    norm_x_ref_enabled,
                )
            self._update_normalization_warning()
            self._sync_standard_controls_to_advanced_json()
            self._refresh_shell_state()

        def _collect_settings(self) -> dict[str, Any]:
            self._persist_series_editor(self._series_active_index)
            self._persist_normalization_editor(self._normalization_active_index)

            def _synced_text(key: str, widget: QLineEdit) -> str | None:
                if not self._synced_field_locks.get(key, False):
                    return None
                return _explicit_text(widget.text())

            def _synced_float(key: str, widget: QLineEdit, *, field_name: str) -> float | None:
                if not self._synced_field_locks.get(key, False):
                    return None
                return _optional_float(widget.text(), field_name=field_name)

            def _synced_float_list(
                key: str,
                widget: QLineEdit,
                *,
                field_name: str,
            ) -> list[float] | None:
                if not self._synced_field_locks.get(key, False):
                    return None
                return _optional_float_list(widget.text(), field_name=field_name)

            def _optional_positive_int_or_none(value: str, *, field_name: str) -> int | None:
                parsed = _optional_int(value, field_name=field_name)
                if parsed is None or parsed <= 0:
                    return None
                return parsed

            fig_width = _optional_float(self.fig_width.text(), field_name="figure width")
            fig_height = _optional_float(self.fig_height.text(), field_name="figure height")
            if (fig_width is None) != (fig_height is None):
                raise ValueError(
                    "Figure width and figure height must both be set or both be blank."
                )
            figsize: list[float] | None = None
            if fig_width is not None and fig_height is not None:
                figsize = [fig_width, fig_height]

            series_labels = [
                self._effective_series_label(index)
                for index in range(len(self._series_labels_data))
            ]
            line_colors = [color.strip() for color in self._series_colors_data]
            line_colors_value = _resolve_series_line_colors(line_colors)

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
                    raise ValueError(f"Series {index + 1} line width must be a float.") from exc
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

            normalization_modes = [
                mode.strip().lower() for mode in self._series_normalization_modes_data
            ]
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
                    value = None
                    x_ref = None
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

            series_overrides: dict[str, dict[str, Any]] = {}
            for index, descriptor in enumerate(self._series_descriptors_data):
                series_id = str(descriptor.get("series_id") or f"series:{index}")
                entry: dict[str, Any] = {}
                label_override = self._series_label_overrides_data[index].strip()
                if label_override:
                    entry["label_override"] = label_override
                if self._series_enabled_data[index] is False:
                    entry["enabled"] = False
                if line_colors[index]:
                    entry["color"] = line_colors[index]
                width_token = self._series_line_widths_data[index].strip()
                if width_token:
                    try:
                        entry["line_width"] = float(width_token)
                    except ValueError as exc:
                        raise ValueError(f"Series {index + 1} line width must be a float.") from exc
                marker_token = self._series_markers_data[index].strip()
                if marker_token:
                    entry["marker"] = marker_token
                line_kwargs_token = self._series_line_kwargs_data[index].strip()
                if line_kwargs_token:
                    entry["line_kwargs"] = _optional_json_dict(
                        line_kwargs_token,
                        field_name=f"Series {index + 1} line kwargs",
                    )
                mode = normalization_modes[index]
                if mode != "none":
                    entry["normalization_mode"] = mode
                    if normalization_values[index] is not None:
                        entry["normalization_value"] = normalization_values[index]
                    if normalization_x_refs[index] is not None:
                        entry["normalization_x_ref"] = normalization_x_refs[index]
                if entry:
                    series_overrides[series_id] = entry

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
            facecolor = _explicit_text(self.figure_facecolor.text())
            if facecolor:
                figure_kwargs_merged["facecolor"] = facecolor
            else:
                figure_kwargs_merged.pop("facecolor", None)
            figure_kwargs_value = figure_kwargs_merged or None

            axes_kwargs_value = (
                dict(axes_kwargs_value) if isinstance(axes_kwargs_value, dict) else None
            )
            x_label_pad = _synced_float(
                "x_label_pad",
                self.x_label_pad,
                field_name="x-label pad",
            )
            y_label_pad = _synced_float(
                "y_label_pad",
                self.y_label_pad,
                field_name="y-label pad",
            )

            line_kwargs_merged = (
                dict(line_kwargs_value) if isinstance(line_kwargs_value, dict) else {}
            )
            line_style = self.line_style.currentText().strip()
            line_alpha = _optional_float(self.line_alpha.text(), field_name="line-alpha")
            marker_size = _optional_float(self.marker_size.text(), field_name="marker-size")
            if line_style:
                line_kwargs_merged["linestyle"] = line_style
            else:
                line_kwargs_merged.pop("linestyle", None)
            if line_alpha is not None:
                line_kwargs_merged["alpha"] = line_alpha
            else:
                line_kwargs_merged.pop("alpha", None)
            if marker_size is not None:
                line_kwargs_merged["markersize"] = marker_size
            else:
                line_kwargs_merged.pop("markersize", None)
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
            else:
                legend_kwargs_merged.pop("ncols", None)
            legend_kwargs_value = legend_kwargs_merged or None

            grid_kwargs_merged = (
                dict(grid_kwargs_value) if isinstance(grid_kwargs_value, dict) else {}
            )
            grid_color = _explicit_text(self.grid_color.text())
            if grid_color:
                grid_kwargs_merged["color"] = grid_color
            else:
                grid_kwargs_merged.pop("color", None)
            grid_axis = self.grid_axis.currentText().strip().lower() or "both"
            if grid_axis not in _GRID_AXES:
                raise ValueError("Grid axis must be both, x, or y.")
            grid_kwargs_merged["axis"] = grid_axis
            grid_which = self.grid_which.currentText().strip().lower() or "major"
            if grid_which not in _GRID_WHICH:
                raise ValueError("Grid lines selector must be major, minor, or both.")
            grid_kwargs_merged["which"] = grid_which
            grid_kwargs_value = grid_kwargs_merged or None

            tick_params_kwargs_merged = (
                dict(tick_params_kwargs_value) if isinstance(tick_params_kwargs_value, dict) else {}
            )
            tick_direction = self.tick_direction.currentText().strip().lower() or "out"
            if tick_direction not in _TICK_DIRECTIONS:
                raise ValueError("Tick direction must be out, in, or inout.")
            tick_params_kwargs_merged["direction"] = tick_direction
            tick_length = _optional_float(self.tick_length.text(), field_name="tick-length")
            if tick_length is None or tick_length <= 0:
                tick_params_kwargs_merged.pop("length", None)
            else:
                tick_params_kwargs_merged["length"] = tick_length
            tick_width = _optional_float(self.tick_width.text(), field_name="tick-width")
            if tick_width is None or tick_width <= 0:
                tick_params_kwargs_merged.pop("width", None)
            else:
                tick_params_kwargs_merged["width"] = tick_width
            ticks_axis = self.ticks_visibility.currentText().strip().lower() or "both"
            if ticks_axis not in _TICK_VISIBILITY_MODES:
                raise ValueError("Ticks must be both, x, y, or none.")
            resolved_ticks_axis = "both" if ticks_axis == "none" else ticks_axis
            tick_params_kwargs_merged["_ticks_axis"] = resolved_ticks_axis
            tick_params_kwargs_merged["axis"] = resolved_ticks_axis
            minor_ticks_mode = self.minor_ticks_mode.currentText().strip().lower() or "off"
            if minor_ticks_mode not in _MINOR_TICKS_MODES:
                raise ValueError("Minor ticks mode must be on or off.")
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

            x_min = _synced_float("x_lim", self.x_min, field_name="x-min")
            x_max = _synced_float("x_lim", self.x_max, field_name="x-max")
            y_min = _synced_float("y_lim", self.y_min, field_name="y-min")
            y_max = _synced_float("y_lim", self.y_max, field_name="y-max")
            title_value = _synced_text("title", self.title_text)
            title_visible_value: bool | None = None
            if self._synced_field_locks.get("title", False):
                title_visible_value = bool(title_value)
                if not title_visible_value:
                    title_value = None

            settings = {
                "title": title_value,
                "x_label": _synced_text("x_label", self.x_label),
                "y_label": _synced_text("y_label", self.y_label),
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "x_scale": self.x_scale.currentText().strip() or "linear",
                "y_scale": self.y_scale.currentText().strip() or "linear",
                "x_ticks": _synced_float_list("x_ticks", self.x_ticks, field_name="x-ticks"),
                "y_ticks": _synced_float_list("y_ticks", self.y_ticks, field_name="y-ticks"),
                "x_tick_rotation": _optional_float(
                    self.x_tick_rotation.text(), field_name="x-tick-rotation"
                ),
                "y_tick_rotation": _optional_float(
                    self.y_tick_rotation.text(), field_name="y-tick-rotation"
                ),
                "title_visible": title_visible_value,
                "legend": _mode_to_toggle(self.legend_mode.currentText()),
                "grid": _mode_to_toggle(self.grid_mode.currentText()),
                "ticks": self.ticks_visibility.currentText().strip().lower() != "none",
                "markers": _mode_to_toggle(self.markers_mode.currentText()),
                "legend_title": _explicit_text(self.legend_title.text()) or None,
                "legend_loc": self.legend_loc.currentText().strip() or "best",
                "figsize": figsize,
                "dpi": _optional_positive_int_or_none(self.dpi.text(), field_name="dpi"),
                "font_family": _explicit_text(self.font_family.text()) or None,
                "title_font_size": _optional_positive_int_or_none(
                    self.title_font.text(), field_name="title-font-size"
                ),
                "label_font_size": _optional_positive_int_or_none(
                    self.label_font.text(), field_name="label-font-size"
                ),
                "tick_font_size": _optional_positive_int_or_none(
                    self.tick_font.text(), field_name="tick-font-size"
                ),
                "legend_font_size": _optional_positive_int_or_none(
                    self.legend_font.text(), field_name="legend-font-size"
                ),
                "line_width": _optional_float(self.line_width.text(), field_name="line-width"),
                "line_color": _explicit_text(self.line_color.text()) or None,
                "line_colors": line_colors_value,
                "series_labels": series_labels,
                "series_descriptors": deepcopy(self._series_descriptors_data),
                "series_overrides": series_overrides or None,
                "series_enabled": series_enabled_value,
                "series_line_widths": series_line_widths_value,
                "series_markers": series_markers_value,
                "series_line_kwargs": series_line_kwargs_value,
                "series_normalization_modes": normalization_modes_value,
                "series_normalization_values": normalization_values_value,
                "series_normalization_x_refs": normalization_x_refs_value,
                "x_bin_width": x_bin_width,
                "x_bin_reducer": (self.x_bin_reducer.currentText().strip() or "mean")
                if x_bin_width is not None
                else None,
                "grid_linestyle": _explicit_text(self.grid_linestyle.currentText()) or None,
                "grid_linewidth": _optional_float(
                    self.grid_linewidth.text(), field_name="grid-linewidth"
                ),
                "grid_alpha": _optional_float(self.grid_alpha.text(), field_name="grid-alpha"),
                "matplotlib_rc": matplotlib_rc_value,
                "figure_kwargs": figure_kwargs_value,
                "axes_kwargs": axes_kwargs_value,
                "x_label_pad": x_label_pad,
                "y_label_pad": y_label_pad,
                "line_kwargs": line_kwargs_value,
                "legend_kwargs": legend_kwargs_value,
                "grid_kwargs": grid_kwargs_value,
                "tick_params_kwargs": tick_params_kwargs_value,
                "tight_layout_kwargs": tight_layout_kwargs_value,
                "savefig_kwargs": savefig_kwargs_value,
                "_gui_locked_fields": dict(self._synced_field_locks),
            }
            if hasattr(self, "analysis_species"):
                settings["species"] = _explicit_text(self.analysis_species.text()) or None
            if hasattr(self, "analysis_axis"):
                axis_value = self.analysis_axis.currentText().strip().lower()
                settings["axis"] = None if axis_value == "" else axis_value
            if hasattr(self, "density_x_mode"):
                settings["x_mode"] = self.density_x_mode.currentText().strip() or "distance"
            if hasattr(self, "density_quantity"):
                settings["quantity"] = self.density_quantity.currentText().strip() or "mass"
            if hasattr(self, "rdf_species_a"):
                settings["species_a"] = self._selected_profile_filter_value(
                    self.rdf_species_a,
                    default_label=_PROFILE_FILTER_METADATA_LABEL,
                )
            if hasattr(self, "rdf_species_b"):
                settings["species_b"] = self._selected_profile_filter_value(
                    self.rdf_species_b,
                    default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                )
            if hasattr(self, "coord_species_a"):
                settings["species_a"] = self._selected_profile_filter_value(
                    self.coord_species_a,
                    default_label=_PROFILE_FILTER_METADATA_LABEL,
                )
            if hasattr(self, "coord_species_b"):
                settings["species_b"] = self._selected_profile_filter_value(
                    self.coord_species_b,
                    default_label=_PROFILE_FILTER_SPECIES_B_AUTO_LABEL,
                )
            if hasattr(self, "position_component"):
                settings["component"] = self.position_component.currentText().strip() or "distance"
            if hasattr(self, "position_map_color"):
                settings["map_color"] = self.position_map_color.currentText().strip() or "distance"
            if hasattr(self, "position_time_axis"):
                settings["time_axis"] = self.position_time_axis.currentText().strip() or "ps"
            if hasattr(self, "coordination_component"):
                settings["component"] = (
                    self.coordination_component.currentText().strip() or "distance"
                )
            if hasattr(self, "coordination_time_axis"):
                settings["time_axis"] = self.coordination_time_axis.currentText().strip() or "ps"
            return settings

        def _report_error(self, title_text: str, exc: Exception) -> None:
            self._status_label.setText(f"{title_text}: {exc}")
            self._refresh_shell_state()
            QMessageBox.critical(self, title_text, str(exc))

        def _handle_preview(self) -> None:
            self._preview_timer.stop()
            self._update_embedded_preview(interactive=True)

        def _handle_save(self) -> None:
            try:
                settings = self._collect_settings()
                message = on_save(self._current_profile_name, settings)
                self._status_label.setText(message)
                self._saved_signature = self._signature(settings)
                self._refresh_shell_state()
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
                result = on_save_figure(settings, output_path)
                message = result[0] if isinstance(result, tuple) else result
                render_state = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                if isinstance(render_state, dict) and render_state:
                    self._apply_preview_state_to_synced_fields(render_state)
                self._status_label.setText(message)
                self._refresh_shell_state()
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
            baseline.pop("series_overrides", None)
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
            self._refresh_shell_state()

        def _handle_import_json(self) -> None:
            if not self._confirm_context_change(action_label="importing settings"):
                return
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
                        raise ValueError("HDF5 import is unavailable for this plot session.")
                    payload = on_import_hdf5(path_str, self._current_profile_name)
                    source_label = source_path.name
                else:
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                    source_label = source_path.name
                if not isinstance(payload, dict):
                    raise ValueError("JSON root must be an object with setting keys.")
                payload = _without_series_specific_settings(payload)
                merged = self._resolved_default_profile_settings()
                merged.update(payload)
                self._load_settings_into_editor(
                    merged,
                    status_message=f"Imported non-series settings into '{self._current_profile_name}' from '{source_label}'.",
                    mark_saved=False,
                )
                self._schedule_preview_update()
                self._refresh_shell_state()
            except Exception as exc:
                self._report_error("Import failed", exc)

        def _handle_export_json(self) -> None:
            try:
                settings = self._collect_settings()
                settings = _without_series_specific_settings(settings)
                path_str, _selected = QFileDialog.getSaveFileName(
                    self,
                    "Export Plot Settings JSON",
                    f"{self._current_profile_name.lower().replace(' ', '_')}_plot_settings.json",
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
                self._refresh_shell_state()
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

        def changeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
            if event.type() in {
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
            }:
                self._apply_theme_styles()
                if hasattr(self, "series_list") and self.series_list is not None:
                    for index in range(self.series_list.count()):
                        self._apply_series_list_item_visuals(self.series_list.item(index), index)
            super().changeEvent(event)

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
                    message = on_save(self._current_profile_name, settings)
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
