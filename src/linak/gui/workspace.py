"""PySide6 project workspace for LiNaK."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .actions import Action, ActionRegistry, SettingField, validate_action_settings
from .components import grouped_item_rows, task_detail_display
from .detection import detect_project_item, discover_generated_items_cached
from .defaults import (
    default_settings_for_action,
    defaults_validate,
    out_h5_gui_summary_for_item,
    readiness_for_action,
)
from .model import ProjectItem, ProjectStore, Task
from .services import descriptor_for_action
from .styles import badge_style
from .tasks import TaskManager
from .theme import is_dark_theme, workspace_stylesheet
from .viewers import open_project_item
from .viewmodels import (
    Badge,
    display_for_item,
    display_for_task,
    filter_items,
    relationship_names,
    suggested_actions_for_item,
)
from .widgets import CollapsibleSection

_OPEN_PROJECT_WORKSPACES: list[ProjectWorkspaceWindow] = []


def _require_pyside6() -> None:
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ValueError(
            "The LiNaK GUI requires PySide6. Install LiNaK with GUI dependencies and rerun."
        ) from exc


def _coerce_field_value(field: SettingField, widget: Any) -> Any:
    from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QDoubleSpinBox):
        if not field.required and widget.property("linak_empty"):
            return None
        return widget.value()
    if isinstance(widget, QSpinBox):
        if not field.required and widget.property("linak_empty"):
            return None
        return widget.value()
    if isinstance(widget, QLineEdit):
        text = widget.text().strip()
        if text == "":
            if field.required:
                raise ValueError(f"{field.label} is required.")
            return None
        if field.kind == "float":
            value = float(text)
            if field.minimum is not None and value < field.minimum:
                raise ValueError(f"{field.label} must be >= {field.minimum}.")
            return value
        if field.kind == "int":
            value = int(text)
            if field.minimum is not None and value < field.minimum:
                raise ValueError(f"{field.label} must be >= {int(field.minimum)}.")
            return value
        return text
    return None


def _field_auto_source(field: SettingField, summary: Any | None) -> str:
    """Return explicit auto-source text for settings backed by `.out.h5`."""

    if summary is None:
        return ""
    if field.key == "cell" and summary.cell_angstrom is not None:
        return (
            "Auto from .out.h5: "
            + " ".join(f"{value:.6g}" for value in summary.cell_angstrom)
            + " A"
        )
    if field.key == "timestep_fs":
        timestep = summary.timestep_fs
        if timestep is None and summary.timestep_candidates_fs:
            timestep = summary.timestep_candidates_fs[0]
        if timestep is not None:
            return f"Auto from .out.h5: {timestep:g} fs"
    if field.widget == "species" and summary.species:
        return "Auto from .out.h5: " + ", ".join(summary.species)
    if field.key == "input" and summary.trajectory_source_path:
        return f"Auto from .out.h5: {Path(summary.trajectory_source_path).name}"
    return ""


class SettingsDialog:
    """Shared typed settings dialog generated from action schemas."""

    def __init__(
        self,
        parent: Any,
        *,
        action: Action,
        item: ProjectItem,
        project_dir: Path,
        theme_colors: dict[str, str] | None = None,
    ) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle(f"{action.name} settings")
        self.dialog.resize(720, 620)
        if parent is not None and hasattr(parent, "styleSheet"):
            self.dialog.setStyleSheet(parent.styleSheet())
        defaults = default_settings_for_action(action, item, project_dir)
        self._fields = [
            replace(field, default=defaults.get(field.key, field.default))
            for field in action.settings_schema(item)
        ]
        self._widgets: dict[str, Any] = {}
        self._result: dict[str, Any] = {}
        self._action = action
        self._item = item
        self._project_dir = project_dir
        self._theme_colors = theme_colors or {}
        self._section_state: dict[str, bool] = {}

        root = QVBoxLayout(self.dialog)
        summary = out_h5_gui_summary_for_item(item)
        source_hint = "\n".join(summary.detail_lines()[:6]) if summary is not None else ""
        intro_text = f"{action.description}\nInput: {item.display_name}"
        if source_hint:
            intro_text += f"\n\nDetected from .out.h5:\n{source_hint}"
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        intro.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(intro)
        self._summary_label = QLabel(action.summary(project_dir=project_dir, item=item, settings=defaults))
        self._summary_label.setWordWrap(True)
        self._summary_label.setObjectName("MutedText")
        root.addWidget(self._summary_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_body = QWidget()
        scroll_layout = QVBoxLayout(scroll_body)
        grouped: dict[str, list[SettingField]] = defaultdict(list)
        for field in self._fields:
            grouped[field.group].append(field)

        ordered_groups = ("Input", "Selection", "Geometry", "Time", "Binning", "Output", "Surface", "Cell / Metadata", "Cell", "Cutoff", "Water detection", "Analysis", "Discovery", "CP2K", "Execution", "Advanced", "General")
        group_names = [name for name in ordered_groups if name in grouped]
        group_names.extend(name for name in grouped if name not in group_names)
        for group_name in group_names:
            fields = grouped[group_name]
            body = QWidget()
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(12, 0, 12, 12)
            form_host = QWidget()
            form = QFormLayout(form_host)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            for field in fields:
                widget_kind = field.widget if field.widget != "auto" else field.kind
                if field.kind == "bool":
                    widget = QCheckBox()
                    widget.setChecked(bool(field.default))
                elif widget_kind in {"choice", "axis"}:
                    widget = QComboBox()
                    widget.addItems(list(field.choices))
                    if field.default is not None:
                        index = widget.findText(str(field.default))
                        if index >= 0:
                            widget.setCurrentIndex(index)
                elif widget_kind == "species":
                    widget = QComboBox()
                    widget.setEditable(True)
                    species = ["all"]
                    if summary is not None:
                        species.extend(value for value in summary.species if value not in species)
                    widget.addItems(species)
                    if field.default is not None:
                        widget.setCurrentText(str(field.default))
                    if field.placeholder:
                        widget.lineEdit().setPlaceholderText(field.placeholder)
                elif field.kind == "float":
                    widget = QDoubleSpinBox()
                    widget.setDecimals(6)
                    widget.setRange(
                        -1.0e12 if field.minimum is None else float(field.minimum),
                        1.0e12,
                    )
                    widget.setSpecialValueText("auto") if not field.required else None
                    if field.unit:
                        widget.setSuffix(f" {field.unit}")
                    if field.default is None:
                        widget.setProperty("linak_empty", True)
                        widget.setValue(field.minimum if field.minimum is not None else 0.0)
                    else:
                        widget.setProperty("linak_empty", False)
                        widget.setValue(float(field.default))
                    widget.valueChanged.connect(lambda _value, target=widget: target.setProperty("linak_empty", False))
                elif field.kind == "int":
                    widget = QSpinBox()
                    widget.setRange(int(field.minimum or 0), 2_147_483_647)
                    widget.setSpecialValueText("auto") if not field.required else None
                    if field.default is None:
                        widget.setProperty("linak_empty", True)
                        widget.setValue(int(field.minimum or 0))
                    else:
                        widget.setProperty("linak_empty", False)
                        widget.setValue(int(field.default))
                    widget.valueChanged.connect(lambda _value, target=widget: target.setProperty("linak_empty", False))
                else:
                    line = QLineEdit("" if field.default is None else str(field.default))
                    if field.placeholder:
                        line.setPlaceholderText(field.placeholder)
                    widget = line
                    if field.kind == "path":
                        row = QWidget()
                        row_layout = QHBoxLayout(row)
                        row_layout.setContentsMargins(0, 0, 0, 0)
                        row_layout.addWidget(line, 1)
                        browse = QPushButton("Browse")
                        browse.clicked.connect(
                            lambda _checked=False, target=line: self._browse_path(target)
                        )
                        row_layout.addWidget(browse)
                        widget = row
                        self._widgets[field.key] = line
                if field.key not in self._widgets:
                    self._widgets[field.key] = widget
                hint_lines = [line for line in (field.help_text, field.description) if line]
                auto_source = _field_auto_source(field, summary)
                display_widget = widget
                if auto_source:
                    hint_lines.append(auto_source)
                    mode = QComboBox()
                    mode.addItems(["Auto", "Manual"])
                    mode.setToolTip(auto_source)
                    auto_target = widget
                    mode.currentTextChanged.connect(
                        lambda value, target=auto_target: target.setEnabled(value == "Manual")
                    )
                    auto_target.setEnabled(False)
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.addWidget(mode)
                    row_layout.addWidget(widget, 1)
                    display_widget = row
                    auto_label = QLabel(auto_source)
                    auto_label.setObjectName("MutedText")
                    auto_label.setWordWrap(True)
                    form.addRow("", auto_label)
                if hint_lines:
                    widget.setToolTip("\n".join(hint_lines))
                if field.kind in {"text", "path"}:
                    target = self._widgets[field.key]
                    if isinstance(target, QLineEdit):
                        target.textChanged.connect(self._refresh_summary)
                elif field.kind in {"float", "int"}:
                    widget.valueChanged.connect(self._refresh_summary)
                elif field.kind == "choice" or widget_kind in {"species", "axis"}:
                    widget.currentTextChanged.connect(self._refresh_summary)
                elif field.kind == "bool":
                    widget.stateChanged.connect(self._refresh_summary)
                form.addRow(field.label, display_widget)
            body_layout.addWidget(form_host)
            section = CollapsibleSection(
                title=group_name,
                section_id=f"settings:{action.action_id}:{group_name}",
                state_store=self._section_state,
                default_expanded=group_name not in {"Advanced", "Execution", "CP2K"},
            )
            section.set_body_widget(body)
            scroll_layout.addWidget(section)
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_body)
        root.addWidget(scroll, 1)
        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        root.addWidget(self._validation_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.dialog.reject)
        root.addWidget(buttons)

    def _browse_path(self, line_edit: Any) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _selected = QFileDialog.getOpenFileName(self.dialog, "Select file")
        if path:
            line_edit.setText(path)

    def _accept(self) -> None:
        try:
            values = self._collect_values()
            validate_action_settings(self._action, self._item, values)
            self._result = values
        except Exception as exc:
            self._validation_label.setText(f"Invalid settings: {exc}")
            self._validation_label.setStyleSheet(
                badge_style("danger", colors=self._theme_colors or None)
            )
            return
        self.dialog.accept()

    def _collect_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field in self._fields:
            values[field.key] = _coerce_field_value(field, self._widgets[field.key])
        return values

    def _refresh_summary(self, *_args: Any) -> None:
        try:
            values = self._collect_values()
            validate_action_settings(self._action, self._item, values)
            self._validation_label.setText("Settings are valid.")
            self._validation_label.setStyleSheet(
                badge_style("success", colors=self._theme_colors or None)
            )
            self._summary_label.setText(
                self._action.summary(
                    project_dir=self._project_dir,
                    item=self._item,
                    settings=values,
                )
            )
        except Exception as exc:
            self._validation_label.setText(f"Check settings: {exc}")
            self._validation_label.setStyleSheet(
                badge_style("warning", colors=self._theme_colors or None)
            )

    def exec(self) -> tuple[bool, dict[str, Any]]:
        accepted = bool(self.dialog.exec())
        return accepted, dict(self._result)


def _task_status_text(task: Task) -> str:
    if task.status == "failed" and task.error:
        return f"{task.action_name}: failed - {task.error}"
    return f"{task.action_name}: {task.status}"


class ProjectWorkspaceWindow:
    """Main project workspace window."""

    def __init__(self, store: ProjectStore, *, created_project_dir: bool = False) -> None:
        _require_pyside6()
        from PySide6.QtCore import QObject, Qt, Signal
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QFrame,
            QHBoxLayout,
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
            QVBoxLayout,
            QWidget,
        )

        class _ProjectMainWindow(QMainWindow):
            def __init__(self, owner: ProjectWorkspaceWindow) -> None:
                super().__init__()
                self._owner = owner

            def changeEvent(self, event: Any) -> None:  # pragma: no cover - UI flow
                from PySide6.QtCore import QEvent

                if event.type() in {
                    QEvent.Type.PaletteChange,
                    QEvent.Type.ApplicationPaletteChange,
                }:
                    self._owner._apply_theme_styles()
                    self._owner._sync_theme_switch_label()
                super().changeEvent(event)

        class _Bridge(QObject):
            task_updated = Signal(object)
            task_finished = Signal(object)

        self._qt = {
            "QListWidgetItem": QListWidgetItem,
            "QMessageBox": QMessageBox,
            "QFrame": QFrame,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QLineEdit": QLineEdit,
            "QPushButton": QPushButton,
            "QScrollArea": QScrollArea,
            "QSizePolicy": QSizePolicy,
            "QVBoxLayout": QVBoxLayout,
            "QWidget": QWidget,
            "QComboBox": QComboBox,
            "Qt": Qt,
        }
        self.store = store
        self.registry = ActionRegistry()
        self.task_manager = TaskManager(store)
        self._theme_mode = "system"
        self._theme_switch: Any | None = None
        self._theme_colors: dict[str, str] = {}
        self._tooltips: dict[int, str] = {}
        self._section_state: dict[str, bool] = {}
        self.bridge = _Bridge()
        self.bridge.task_updated.connect(self._handle_task_updated)
        self.bridge.task_finished.connect(self._handle_task_finished)

        self.window = _ProjectMainWindow(self)
        self.window.setWindowTitle(f"LiNaK Workspace - {store.project_dir}")
        self.window.resize(1280, 820)

        central = QWidget()
        central.setObjectName("windowRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        app_header = QFrame()
        app_header.setObjectName("appHeader")
        header_layout = QHBoxLayout(app_header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        app_title = QLabel("LiNaK Project Workspace")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel(str(store.project_dir))
        app_subtitle.setObjectName("appSubtitle")
        app_subtitle.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title_stack.addWidget(app_title)
        title_stack.addWidget(app_subtitle)
        header_layout.addLayout(title_stack, 1)
        self._theme_switch = QCheckBox("Dark mode")
        self._theme_switch.setObjectName("themeSwitch")
        self._theme_switch.toggled.connect(self._handle_theme_switch_toggled)
        header_layout.addWidget(self._theme_switch)
        root.addWidget(app_header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)

        left = QFrame()
        left.setObjectName("navPanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(8)
        header = QLabel("Project items")
        header.setObjectName("SectionTitle")
        self.item_search = QLineEdit()
        self.item_search.setPlaceholderText("Search files, analysis, paths")
        self.item_search.textChanged.connect(self._refresh_items)
        self.origin_filter = QComboBox()
        self.origin_filter.addItems(["all", "external", "generated"])
        self.origin_filter.currentTextChanged.connect(self._refresh_items)
        self.type_filter = QComboBox()
        self.type_filter.addItem("all")
        self.type_filter.currentTextChanged.connect(self._refresh_items)
        self.group_filter = QComboBox()
        self.group_filter.addItems(["Workflow", "Type", "Source run", "Flat"])
        self.group_filter.currentTextChanged.connect(self._refresh_items)
        self.item_list = QListWidget()
        self.item_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.item_list.currentItemChanged.connect(self._selected_item_changed)
        import_button = QPushButton("Import file")
        import_button.setObjectName("PrimaryButton")
        import_button.clicked.connect(self._import_file)
        import_dir_button = QPushButton("Convert directory")
        import_dir_button.setObjectName("PrimaryButton")
        import_dir_button.clicked.connect(self._import_directory)
        rescan_button = QPushButton("Rescan project")
        rescan_button.setObjectName("SecondaryButton")
        rescan_button.clicked.connect(self._rescan_project)
        remove_button = QPushButton("Remove")
        remove_button.setObjectName("SecondaryButton")
        remove_button.clicked.connect(self._remove_selected_item)
        self._register_tooltip(import_button, "Reference an external input file without copying it.")
        self._register_tooltip(import_dir_button, "Select a simulation directory and queue a .out.h5 packing task.")
        self._register_tooltip(rescan_button, "Refresh generated outputs and cached metadata in this project.")
        self._register_tooltip(remove_button, "Remove the selected item reference or generated output after confirmation.")
        left_layout.addWidget(header)
        left_layout.addWidget(self.item_search)
        filters = QHBoxLayout()
        filters.addWidget(self.origin_filter)
        filters.addWidget(self.type_filter)
        filters.addWidget(self.group_filter)
        left_layout.addLayout(filters)
        left_layout.addWidget(self.item_list, 1)
        left_buttons = QHBoxLayout()
        left_buttons.addWidget(import_button)
        left_buttons.addWidget(import_dir_button)
        left_buttons.addWidget(rescan_button)
        left_buttons.addWidget(remove_button)
        left_layout.addLayout(left_buttons)

        action_panel = QFrame()
        action_panel.setObjectName("inspectorPanel")
        action_panel_layout = QVBoxLayout(action_panel)
        action_panel_layout.setContentsMargins(14, 14, 14, 14)
        action_panel_label = QLabel("Inspector")
        action_panel_label.setObjectName("SectionTitle")
        action_panel_layout.addWidget(action_panel_label)
        self.actions_scroll = QScrollArea()
        self.actions_scroll.setWidgetResizable(True)
        self.actions_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.actions_body = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_body)
        self.actions_layout.addStretch(1)
        self.actions_scroll.setWidget(self.actions_body)
        action_panel_layout.addWidget(self.actions_scroll, 1)

        top.addWidget(left)
        top.addWidget(action_panel)
        top.setSizes([390, 890])

        bottom = QSplitter(Qt.Orientation.Horizontal)
        task_panel = QFrame()
        task_panel.setObjectName("taskPanel")
        task_layout = QVBoxLayout(task_panel)
        task_layout.setContentsMargins(14, 14, 14, 14)
        task_label = QLabel("Tasks")
        task_label.setObjectName("SectionTitle")
        self.task_list = QListWidget()
        self.task_list.currentItemChanged.connect(self._selected_task_changed)
        task_buttons = QHBoxLayout()
        self.cancel_task_button = QPushButton("Cancel")
        self.cancel_task_button.clicked.connect(self._cancel_selected_task)
        self.remove_task_button = QPushButton("Remove completed")
        self.remove_task_button.clicked.connect(self._remove_selected_task)
        self.pause_queue_button = QPushButton("Pause queue")
        self.pause_queue_button.clicked.connect(self._toggle_queue_pause)
        self.cancel_queued_button = QPushButton("Cancel queued")
        self.cancel_queued_button.clicked.connect(self._cancel_all_queued)
        self.move_task_up_button = QPushButton("Move up")
        self.move_task_up_button.clicked.connect(lambda: self._move_selected_task(-1))
        self.move_task_down_button = QPushButton("Move down")
        self.move_task_down_button.clicked.connect(lambda: self._move_selected_task(1))
        self.retry_task_button = QPushButton("Retry")
        self.retry_task_button.clicked.connect(self._retry_selected_task)
        task_buttons.addWidget(self.cancel_task_button)
        task_buttons.addWidget(self.remove_task_button)
        task_buttons.addWidget(self.pause_queue_button)
        task_buttons.addWidget(self.cancel_queued_button)
        task_buttons.addWidget(self.move_task_up_button)
        task_buttons.addWidget(self.move_task_down_button)
        task_buttons.addWidget(self.retry_task_button)
        task_layout.addWidget(task_label)
        task_layout.addWidget(self.task_list)
        task_layout.addLayout(task_buttons)

        log_panel = QFrame()
        log_panel.setObjectName("logPanel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(14, 14, 14, 14)
        log_label = QLabel("Task log")
        log_label.setObjectName("SectionTitle")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_view)
        bottom.addWidget(task_panel)
        bottom.addWidget(log_panel)
        bottom.setSizes([340, 940])

        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setSizes([560, 260])
        root.addWidget(splitter)
        self.window.setCentralWidget(central)
        self._apply_theme_styles()
        self._sync_theme_switch_label()

        self._load_project_items()
        self._refresh_items()
        self._refresh_tasks()
        if created_project_dir:
            self.window.statusBar().showMessage(
                f"Created project directory: {store.project_dir}", 8000
            )

    def show(self) -> None:
        self.window.show()

    def _is_dark_theme(self) -> bool:
        return is_dark_theme(self._theme_mode, self.window)

    def _theme_tokens(self) -> dict[str, str]:
        from .theme import plot_like_theme_tokens

        return plot_like_theme_tokens(self._is_dark_theme())

    def _sync_theme_switch_label(self) -> None:
        if self._theme_switch is None:
            return
        is_dark = self._is_dark_theme()
        self._theme_switch.blockSignals(True)
        try:
            self._theme_switch.setChecked(is_dark)
            self._theme_switch.setText("Dark mode" if is_dark else "Light mode")
        finally:
            self._theme_switch.blockSignals(False)

    def _handle_theme_switch_toggled(self, checked: bool) -> None:
        self._theme_mode = "dark" if checked else "light"
        self._sync_theme_switch_label()
        self._apply_theme_styles()
        self._refresh_items()
        self._refresh_tasks()
        self._render_actions(self._selected_item())

    def _apply_theme_styles(self) -> None:
        colors = self._theme_tokens()
        self._theme_colors = colors
        self.window.setStyleSheet(workspace_stylesheet(colors))

    def _register_tooltip(self, widget: Any, text: str | None) -> None:
        if not text:
            return
        self._tooltips[id(widget)] = text
        self._apply_widget_tooltip(widget)

    def _apply_widget_tooltip(self, widget: Any) -> None:
        text = self._tooltips.get(id(widget), "")
        if text:
            widget.setToolTip(text)

    def _load_project_items(self) -> None:
        for item in discover_generated_items_cached(
            self.store.project_dir,
            index=self.store.workspace_index,
        ):
            self.store.upsert_item(item)
        self._rebuild_relationships()
        self._refresh_type_filter_options()
        self.store.save()

    def _refresh_type_filter_options(self) -> None:
        current = self.type_filter.currentText() if hasattr(self, "type_filter") else "all"
        self.type_filter.blockSignals(True)
        self.type_filter.clear()
        self.type_filter.addItem("all")
        for item_type in sorted({item.item_type for item in self.store.items}):
            self.type_filter.addItem(item_type)
        index = self.type_filter.findText(current)
        self.type_filter.setCurrentIndex(index if index >= 0 else 0)
        self.type_filter.blockSignals(False)

    def _rebuild_relationships(self) -> None:
        path_to_item = {str(item.path.expanduser().resolve()): item for item in self.store.items}
        for item in self.store.items:
            metadata = item.metadata.get("profile_metadata")
            if not isinstance(metadata, dict):
                continue
            source_path = str(metadata.get("source_path") or "").strip()
            if source_path and source_path in path_to_item:
                source = path_to_item[source_path]
                item.relationships.setdefault("inputs", [])
                if source.item_id not in item.relationships["inputs"]:
                    item.relationships["inputs"].append(source.item_id)
                source.relationships.setdefault("outputs", [])
                if item.item_id not in source.relationships["outputs"]:
                    source.relationships["outputs"].append(item.item_id)

    def _refresh_items(self, *_args: Any) -> None:
        QListWidgetItem = self._qt["QListWidgetItem"]
        Qt = self._qt["Qt"]
        selected_id = self._selected_item_id()
        self.item_list.clear()
        query = self.item_search.text() if hasattr(self, "item_search") else ""
        origin = self.origin_filter.currentText() if hasattr(self, "origin_filter") else "all"
        item_type = self.type_filter.currentText() if hasattr(self, "type_filter") else "all"
        grouping = self.group_filter.currentText() if hasattr(self, "group_filter") else "Workflow"
        sorted_items = filter_items(
            self.store.items,
            query=query,
            origin=origin,
            item_type=item_type,
        )
        visible_ids = {item.item_id for item in sorted_items}
        grouped_rows = [
            row
            for row in grouped_item_rows(self.store, mode=grouping)
            if row.item.item_id in visible_ids
        ]
        for row in grouped_rows:
            item = row.item
            display = display_for_item(item)
            meta = " | ".join(display.metadata_lines[:2])
            text = f"{row.group_label}\n{display.icon} {display.title}\n{display.subtitle}"
            if meta:
                text = f"{text}\n{meta}"
            if row.de_emphasized:
                text = f"{text}\n(raw source covered by .out.h5)"
            list_item = QListWidgetItem(text)
            list_item.setData(Qt.ItemDataRole.UserRole, item.item_id)
            list_item.setToolTip(display.tooltip)
            self.item_list.addItem(list_item)
            if item.item_id == selected_id:
                self.item_list.setCurrentItem(list_item)
        if self.item_list.count() == 0:
            self._render_empty_actions("Import an input file or generate outputs into this project directory.")
        elif self.item_list.currentItem() is None:
            self.item_list.setCurrentRow(0)

    def _refresh_tasks(self) -> None:
        QListWidgetItem = self._qt["QListWidgetItem"]
        Qt = self._qt["Qt"]
        selected_id = self._selected_task_id()
        self.task_list.clear()
        for task in reversed(self.store.tasks):
            display = display_for_task(task)
            counts = ", ".join(
                f"{level.lower()}={count}"
                for level, count in sorted(display.log_counts.items())
                if count
            )
            subtitle = display.subtitle
            if counts:
                subtitle = f"{subtitle}\n{counts}" if subtitle else counts
            if display.output_labels:
                output_text = "Outputs: " + ", ".join(display.output_labels[:3])
                subtitle = f"{subtitle}\n{output_text}" if subtitle else output_text
            item = QListWidgetItem(
                f"{display.title} [{display.status_badge.text}]"
                + (f"\n{subtitle}" if subtitle else "")
            )
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            self.task_list.addItem(item)
            if task.task_id == selected_id:
                self.task_list.setCurrentItem(item)
        if self.task_list.currentItem() is None and self.task_list.count() > 0:
            self.task_list.setCurrentRow(0)
        self._refresh_task_buttons(self._selected_task())

    def _selected_item_id(self) -> str | None:
        Qt = self._qt["Qt"]
        current = self.item_list.currentItem()
        return None if current is None else str(current.data(Qt.ItemDataRole.UserRole))

    def _selected_task_id(self) -> str | None:
        Qt = self._qt["Qt"]
        current = self.task_list.currentItem()
        return None if current is None else str(current.data(Qt.ItemDataRole.UserRole))

    def _selected_item(self) -> ProjectItem | None:
        selected_id = self._selected_item_id()
        return None if selected_id is None else self.store.item_by_id(selected_id)

    def _selected_items(self) -> list[ProjectItem]:
        Qt = self._qt["Qt"]
        items: list[ProjectItem] = []
        for current in self.item_list.selectedItems():
            item = self.store.item_by_id(str(current.data(Qt.ItemDataRole.UserRole)))
            if item is not None:
                items.append(item)
        return items

    def _selected_task(self) -> Task | None:
        selected_id = self._selected_task_id()
        if selected_id is None:
            return None
        for task in self.store.tasks:
            if task.task_id == selected_id:
                return task
        return None

    def _selected_item_changed(self, _current: Any, _previous: Any) -> None:
        self._render_actions(self._selected_item())

    def _selected_task_changed(self, _current: Any, _previous: Any) -> None:
        task = self._selected_task()
        self._render_task_log(task)
        self._refresh_task_buttons(task)

    def _refresh_task_buttons(self, task: Task | None) -> None:
        if task is None:
            self.cancel_task_button.setEnabled(False)
            self.remove_task_button.setEnabled(False)
            self.move_task_up_button.setEnabled(False)
            self.move_task_down_button.setEnabled(False)
            self.retry_task_button.setEnabled(False)
            return
        self.cancel_task_button.setEnabled(task.status in {"queued", "pending", "running", "canceling"})
        self.remove_task_button.setEnabled(task.status in {"finished", "failed", "canceled"})
        self.move_task_up_button.setEnabled(task.status == "queued")
        self.move_task_down_button.setEnabled(task.status == "queued")
        self.retry_task_button.setEnabled(task.status in {"failed", "canceled", "finished"})
        self.pause_queue_button.setText(
            "Resume queue" if self.task_manager.is_queue_paused() else "Pause queue"
        )

    def _render_task_log(self, task: Task | None) -> None:
        if task is None:
            self.log_view.setPlainText("")
            return
        lines = []
        for entry in task.logs:
            level = str(entry.get("level", "INFO")).upper()
            time = str(entry.get("time_utc", ""))
            message = str(entry.get("message", ""))
            lines.append(f"[{level}] {time} {message}".rstrip())
        if task.output_paths:
            lines.append("")
            lines.append("Outputs:")
            lines.extend(f"  {path}" for path in task.output_paths)
        detail = task_detail_display(task)
        if detail.settings_hash:
            lines.append("")
            lines.append(f"Settings hash: {detail.settings_hash}")
        if detail.cancel_capability:
            lines.append(f"Cancel mode: {detail.cancel_capability}")
        if detail.settings_lines:
            lines.append("")
            lines.append("Settings:")
            lines.extend(f"  {line}" for line in detail.settings_lines[:20])
        if task.error:
            lines.append("")
            lines.append(f"Failure: {task.error}")
        self.log_view.setPlainText("\n".join(lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _clear_actions_layout(self) -> None:
        while self.actions_layout.count():
            item = self.actions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_empty_actions(self, message: str) -> None:
        QLabel = self._qt["QLabel"]
        self._clear_actions_layout()
        label = QLabel(message)
        label.setWordWrap(True)
        self.actions_layout.addWidget(label)
        self.actions_layout.addStretch(1)

    def _badge_label(self, badge: Badge, *, progress_fraction: float | None = None) -> Any:
        QLabel = self._qt["QLabel"]
        label = QLabel(badge.text)
        label.setStyleSheet(
            badge_style(
                badge.tone,
                progress_fraction=progress_fraction,
                colors=self._theme_colors or self._theme_tokens(),
            )
        )
        return label

    def _card(self, title: str, lines: list[str]) -> Any:
        QFrame = self._qt["QFrame"]
        QLabel = self._qt["QLabel"]
        QVBoxLayout = self._qt["QVBoxLayout"]
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        if lines:
            for line in lines:
                label = QLabel(line)
                label.setWordWrap(True)
                label.setTextInteractionFlags(self._qt["Qt"].TextInteractionFlag.TextSelectableByMouse)
                layout.addWidget(label)
        else:
            empty = QLabel("None")
            empty.setObjectName("MutedText")
            layout.addWidget(empty)
        return card

    def _render_item_detail(self, item: ProjectItem) -> None:
        QLabel = self._qt["QLabel"]
        QHBoxLayout = self._qt["QHBoxLayout"]
        QVBoxLayout = self._qt["QVBoxLayout"]
        QWidget = self._qt["QWidget"]
        display = display_for_item(item)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        title = QLabel(f"{display.icon} {display.title}")
        title.setObjectName("SectionTitle")
        header_layout.addWidget(title)
        subtitle = QLabel(display.subtitle)
        subtitle.setObjectName("MutedText")
        header_layout.addWidget(subtitle)
        badges = QWidget()
        badge_layout = QHBoxLayout(badges)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        for badge in display.badges:
            badge_layout.addWidget(self._badge_label(badge))
        badge_layout.addStretch(1)
        header_layout.addWidget(badges)
        self.actions_layout.addWidget(header)

        metadata_lines = list(display.metadata_lines)
        metadata_lines.append(f"Path: {item.path}")
        self.actions_layout.addWidget(self._card("Metadata", metadata_lines))

        summary = out_h5_gui_summary_for_item(item)
        if summary is not None:
            self.actions_layout.addWidget(
                self._card(".out.h5 contents", list(summary.detail_lines()))
            )

        inputs = relationship_names(self.store, item, "inputs")
        outputs = relationship_names(self.store, item, "outputs")
        relationship_lines = [f"Input: {name}" for name in inputs]
        relationship_lines.extend(f"Output: {name}" for name in outputs)
        self.actions_layout.addWidget(self._card("Relationships", relationship_lines))

        suggestions = suggested_actions_for_item(item, self.registry)
        suggestion_lines = [f"{action.category}: {action.name}" for action in suggestions]
        self.actions_layout.addWidget(self._card("Suggested next actions", suggestion_lines))

    def _render_actions(self, item: ProjectItem | None) -> None:
        if item is None:
            self._render_empty_actions("Select a project item to see available actions.")
            return
        actions = self.registry.available_for(item)
        if not actions:
            self._render_empty_actions(
                f"No valid workspace actions are available for {item.display_name}."
            )
            return

        QFrame = self._qt["QFrame"]
        QHBoxLayout = self._qt["QHBoxLayout"]
        QLabel = self._qt["QLabel"]
        QPushButton = self._qt["QPushButton"]
        QSizePolicy = self._qt["QSizePolicy"]
        QVBoxLayout = self._qt["QVBoxLayout"]
        QWidget = self._qt["QWidget"]

        self._clear_actions_layout()
        self._render_item_detail(item)

        grouped: dict[str, list[Action]] = defaultdict(list)
        for action in actions:
            grouped[action.category].append(action)
        for category in ("Compute", "Apply", "Convert", "Open"):
            category_actions = grouped.get(category, [])
            if not category_actions:
                continue
            group_body = QWidget()
            group_layout = QVBoxLayout(group_body)
            group_layout.setContentsMargins(12, 0, 12, 12)
            for action in category_actions:
                readiness = readiness_for_action(action, item)
                row = QFrame()
                row.setObjectName("actionRow")
                row.setFrameShape(QFrame.Shape.StyledPanel)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(12, 10, 12, 10)
                description = action.description
                descriptor = descriptor_for_action(action)
                try:
                    preview_settings = default_settings_for_action(action, item, self.store.project_dir)
                    preview_outputs = action.expected_outputs(
                        project_dir=self.store.project_dir,
                        item=item,
                        settings=preview_settings,
                    )
                except Exception:
                    preview_outputs = ()
                if preview_outputs:
                    description += (
                        "<br><span>Output: "
                        + ", ".join(path.name for path in preview_outputs)
                        + "</span>"
                    )
                description += (
                    f"<br><span>Cancel: {descriptor.cancel_capability}</span>"
                )
                if not readiness.available:
                    description = f"{description}<br><span>Blocked: {readiness.reason}</span>"
                text = QLabel(f"<b>{action.name}</b><br>{description}")
                text.setWordWrap(True)
                text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                latest_task = self._latest_task_for(action, item)
                status = self._badge_label(
                    self._action_status_badge(action, item),
                    progress_fraction=(
                        latest_task.progress_fraction
                        if latest_task is not None and latest_task.status == "running"
                        else None
                    ),
                )
                status.setMinimumWidth(115)
                row_layout.addWidget(text, 1)
                row_layout.addWidget(status)
                if action.action_id == "open_plot":
                    open_button = QPushButton("Open")
                    open_button.setEnabled(readiness.available)
                    if not readiness.available:
                        self._register_tooltip(open_button, readiness.reason)
                    open_button.clicked.connect(lambda _checked=False, selected=item: self._open_item(selected))
                    row_layout.addWidget(open_button)
                else:
                    run_default_button = QPushButton("Run defaults")
                    can_run_defaults = readiness.available and defaults_validate(
                        action,
                        item,
                        self.store.project_dir,
                    )
                    run_default_button.setEnabled(can_run_defaults)
                    if not can_run_defaults:
                        self._register_tooltip(run_default_button, readiness.reason)
                    run_default_button.clicked.connect(
                        lambda _checked=False, selected_action=action, selected=item: self._run_with_defaults(selected_action, selected)
                    )
                    row_layout.addWidget(run_default_button)
                    batch_items = [
                        selected
                        for selected in self._selected_items()
                        if action.supports(selected)
                        and readiness_for_action(action, selected).available
                    ]
                    if len(batch_items) > 1:
                        batch_button = QPushButton(f"Run batch ({len(batch_items)})")
                        batch_button.setEnabled(can_run_defaults)
                        batch_button.clicked.connect(
                            lambda _checked=False, selected_action=action: self._run_defaults_for_items(selected_action, self._selected_items())
                        )
                        row_layout.addWidget(batch_button)
                    settings_button = QPushButton("Configure")
                    settings_button.setEnabled(readiness.available)
                    if not readiness.available:
                        self._register_tooltip(settings_button, readiness.reason)
                    settings_button.clicked.connect(
                        lambda _checked=False, selected_action=action, selected=item: self._configure_and_run(selected_action, selected)
                    )
                    row_layout.addWidget(settings_button)
                    output = self._latest_output_for(action, item)
                    if output is not None:
                        open_button = QPushButton("Open")
                        open_button.clicked.connect(lambda _checked=False, path=output: open_project_item(path))
                        row_layout.addWidget(open_button)
                group_layout.addWidget(row)
            section = CollapsibleSection(
                title=category,
                section_id=f"actions:{category}",
                state_store=self._section_state,
                default_expanded=True,
            )
            section.set_body_widget(group_body)
            self.actions_layout.addWidget(section)
        self.actions_layout.addStretch(1)

    def _action_status_label(self, action: Action, item: ProjectItem) -> str:
        latest = self._latest_task_for(action, item)
        if latest is None:
            return "Ready"
        if latest.status == "finished":
            return "Finished"
        if latest.status == "failed":
            return "Failed"
        return latest.status.capitalize()

    def _action_status_badge(self, action: Action, item: ProjectItem) -> Badge:
        latest = self._latest_task_for(action, item)
        if latest is None:
            return Badge("ready", "neutral")
        tone = {
            "queued": "queued",
            "pending": "neutral",
            "running": "running",
            "canceling": "canceling",
            "canceled": "canceled",
            "finished": "success",
            "failed": "danger",
        }.get(latest.status, "neutral")
        return Badge(self._action_status_label(action, item).lower(), tone)

    def _latest_task_for(self, action: Action, item: ProjectItem) -> Task | None:
        for task in reversed(self.store.tasks):
            if task.action_id == action.action_id and task.input_item_id == item.item_id:
                return task
        return None

    def _latest_output_for(self, action: Action, item: ProjectItem) -> Path | None:
        task = self._latest_task_for(action, item)
        if task is None:
            return None
        for path in reversed(task.output_paths):
            if path.exists():
                return path
        return None

    def _configure_and_run(self, action: Action, item: ProjectItem) -> None:
        dialog = SettingsDialog(
            self.window,
            action=action,
            item=item,
            project_dir=self.store.project_dir,
            theme_colors=self._theme_colors or self._theme_tokens(),
        )
        accepted, settings = dialog.exec()
        if not accepted:
            return
        self._enqueue_task(action, item, settings)

    def _run_with_defaults(self, action: Action, item: ProjectItem) -> None:
        settings = default_settings_for_action(action, item, self.store.project_dir)
        self._enqueue_task(action, item, settings)

    def _run_defaults_for_items(self, action: Action, items: list[ProjectItem]) -> None:
        queued = 0
        for item in items:
            if not action.supports(item) or not readiness_for_action(action, item).available:
                continue
            settings = default_settings_for_action(action, item, self.store.project_dir)
            self._enqueue_task(action, item, settings, select_task=False)
            queued += 1
        self._refresh_tasks()
        self.window.statusBar().showMessage(f"Queued {queued} {action.name} task(s).", 6000)

    def _enqueue_task(
        self,
        action: Action,
        item: ProjectItem,
        settings: dict[str, Any],
        *,
        select_task: bool = True,
    ) -> None:
        QMessageBox = self._qt["QMessageBox"]
        try:
            task = self.task_manager.start(
                action=action,
                item=item,
                settings=settings,
                on_update=self.bridge.task_updated.emit,
                on_finished=self.bridge.task_finished.emit,
            )
        except Exception as exc:
            QMessageBox.critical(self.window, "Could not start task", str(exc))
            return
        self._refresh_tasks()
        if select_task:
            self._select_task(task.task_id)
        self._render_actions(self._selected_item())
        self.window.statusBar().showMessage(
            f"{task.action_name} {task.status} for {item.display_name}.",
            6000,
        )

    def _select_task(self, task_id: str) -> None:
        Qt = self._qt["Qt"]
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            if str(item.data(Qt.ItemDataRole.UserRole)) == task_id:
                self.task_list.setCurrentItem(item)
                return

    def _handle_task_updated(self, task: Task) -> None:
        self._refresh_tasks()
        self._select_task(task.task_id)
        self._render_task_log(task)

    def _handle_task_finished(self, task: Task) -> None:
        if task.status != "finished":
            self._refresh_tasks()
            self._select_task(task.task_id)
            self._render_actions(self._selected_item())
            return
        input_item = self.store.item_by_id(task.input_item_id)
        for output_path in task.output_paths:
            item = detect_project_item(output_path, origin="generated")
            item.metadata["source_item_id"] = task.input_item_id
            item.metadata["source_path"] = str(input_item.path) if input_item is not None else ""
            item.metadata["action_id"] = task.action_id
            item.metadata["settings_hash"] = task.settings_hash
            item.metadata["output_type"] = item.item_type
            if input_item is not None:
                item.relationships.setdefault("inputs", [])
                if input_item.item_id not in item.relationships["inputs"]:
                    item.relationships["inputs"].append(input_item.item_id)
            generated = self.store.upsert_item(item)
            self.store.workspace_index.remember(generated)
            if task.primary_output_item_id is None:
                task.primary_output_item_id = generated.item_id
            if input_item is not None:
                input_item.relationships.setdefault("outputs", [])
                if generated.item_id not in input_item.relationships["outputs"]:
                    input_item.relationships["outputs"].append(generated.item_id)
        self._load_project_items()
        self._refresh_items()
        self._refresh_tasks()
        self._select_task(task.task_id)
        self._render_actions(self._selected_item())

    def _import_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        QMessageBox = self._qt["QMessageBox"]
        path, _selected = QFileDialog.getOpenFileName(
            self.window,
            "Import LiNaK input",
            "",
            "LiNaK inputs (*.xyz *.extxyz *.dump *.lmp *.out.h5 *.out.hdf5 *.traj.h5 *.traj.hdf5 *.cube *.cube.h5 *.cube.hdf5 *.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return
        item = self.store.workspace_index.detect_or_reuse(
            path,
            origin="external",
            detector=detect_project_item,
        )
        if item.validation.state == "invalid":
            QMessageBox.warning(self.window, "Import failed", item.validation.message)
            return
        self.store.upsert_item(item)
        self.store.workspace_index.remember(item)
        self._rebuild_relationships()
        self.store.save()
        self._refresh_items()
        self.window.statusBar().showMessage(f"Imported reference: {item.path}", 6000)

    def _import_directory(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        QMessageBox = self._qt["QMessageBox"]
        path = QFileDialog.getExistingDirectory(
            self.window,
            "Select simulation output directory",
            "",
        )
        if not path:
            return
        item = self.store.workspace_index.detect_or_reuse(
            path,
            origin="external",
            detector=detect_project_item,
        )
        if item.validation.state == "invalid":
            QMessageBox.warning(self.window, "Import failed", item.validation.message)
            return
        self.store.upsert_item(item)
        self.store.workspace_index.remember(item)
        self._rebuild_relationships()
        self.store.save()
        self._refresh_type_filter_options()
        self._refresh_items()
        self.window.statusBar().showMessage(
            f"Imported simulation directory reference: {item.path}",
            6000,
        )

    def _rescan_project(self) -> None:
        self._load_project_items()
        self._refresh_items()
        self.window.statusBar().showMessage("Project scan complete.", 5000)

    def _remove_selected_item(self) -> None:
        QMessageBox = self._qt["QMessageBox"]
        item = self._selected_item()
        if item is None:
            return
        if self.store.can_delete_generated_file(item):
            message = QMessageBox(self.window)
            message.setWindowTitle("Remove project item")
            message.setText(f"Remove {item.display_name} from the project?")
            message.setInformativeText(
                "Generated outputs can also be deleted from disk. External references are never deleted."
            )
            remove_only = message.addButton("Remove from project", QMessageBox.ButtonRole.AcceptRole)
            delete_file = message.addButton("Delete file and remove", QMessageBox.ButtonRole.DestructiveRole)
            message.addButton(QMessageBox.StandardButton.Cancel)
            message.exec()
            clicked = message.clickedButton()
            if clicked == delete_file:
                deleted = self.store.delete_generated_item_file(item.item_id)
                if deleted is None:
                    QMessageBox.warning(self.window, "Remove failed", "The generated file could not be deleted safely.")
                    return
            elif clicked == remove_only:
                self.store.remove_item(item.item_id)
            else:
                return
        else:
            reply = QMessageBox.question(
                self.window,
                "Remove project item",
                f"Remove {item.display_name} from the project manifest?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.store.remove_item(item.item_id)
        self._rebuild_relationships()
        self.store.save()
        self._refresh_type_filter_options()
        self._refresh_items()
        self.window.statusBar().showMessage(f"Removed {item.display_name}.", 5000)

    def _cancel_selected_task(self) -> None:
        task = self._selected_task()
        if task is None:
            return
        if self.task_manager.cancel(task.task_id):
            self._refresh_tasks()
            self._select_task(task.task_id)
            self._render_task_log(task)
            self.window.statusBar().showMessage(f"Cancel requested for {task.action_name}.", 5000)

    def _toggle_queue_pause(self) -> None:
        if self.task_manager.is_queue_paused():
            self.task_manager.resume_queue()
            self.window.statusBar().showMessage("Task queue resumed.", 5000)
        else:
            self.task_manager.pause_queue()
            self.window.statusBar().showMessage("Task queue paused.", 5000)
        self._refresh_tasks()

    def _cancel_all_queued(self) -> None:
        count = self.task_manager.cancel_all_queued()
        self._refresh_tasks()
        self.window.statusBar().showMessage(f"Canceled {count} queued task(s).", 5000)

    def _move_selected_task(self, direction: int) -> None:
        task = self._selected_task()
        if task is None:
            return
        if self.task_manager.reorder_queued(task.task_id, direction):
            self._refresh_tasks()
            self._select_task(task.task_id)

    def _retry_selected_task(self) -> None:
        QMessageBox = self._qt["QMessageBox"]
        task = self._selected_task()
        if task is None:
            return
        item = self.store.item_by_id(task.input_item_id)
        if item is None:
            QMessageBox.warning(self.window, "Retry failed", "The original input item is missing.")
            return
        try:
            action = self.registry.by_id(task.action_id)
        except KeyError:
            QMessageBox.warning(self.window, "Retry failed", "The original action is unavailable.")
            return
        settings = dict(task.settings_snapshot)
        if not settings:
            settings = default_settings_for_action(action, item, self.store.project_dir)
        self._enqueue_task(action, item, settings)

    def _remove_selected_task(self) -> None:
        task = self._selected_task()
        if task is None or task.status not in {"finished", "failed", "canceled"}:
            return
        removed = self.store.remove_task(task.task_id)
        if removed is None:
            return
        self.store.save()
        self._refresh_tasks()
        self._render_task_log(self._selected_task())

    def _open_item(self, item: ProjectItem) -> None:
        try:
            open_project_item(item.path)
        except Exception as exc:
            self._qt["QMessageBox"].critical(self.window, "Open failed", str(exc))


def launch_project_workspace(project_dir: str | Path) -> None:
    """Open the project workspace GUI."""

    _require_pyside6()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    store = ProjectStore(project_dir)
    created = store.initialize()
    store.load()

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])

    window = ProjectWorkspaceWindow(store, created_project_dir=created)
    _OPEN_PROJECT_WORKSPACES.append(window)
    window.window.destroyed.connect(
        lambda _obj=None, workspace=window: _OPEN_PROJECT_WORKSPACES.remove(workspace)
        if workspace in _OPEN_PROJECT_WORKSPACES
        else None
    )
    window.show()
    window.window.showNormal()
    window.window.raise_()
    window.window.activateWindow()
    QTimer.singleShot(0, window.window.raise_)
    QTimer.singleShot(0, window.window.activateWindow)
    if owns_app:
        app.exec()
