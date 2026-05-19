"""Reusable Qt widgets aligned with the plotting GUI interaction patterns."""

from __future__ import annotations

from typing import Any


def require_qt() -> dict[str, Any]:
    from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QSizePolicy,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    return {
        "QEasingCurve": QEasingCurve,
        "QFrame": QFrame,
        "QHBoxLayout": QHBoxLayout,
        "QPropertyAnimation": QPropertyAnimation,
        "QSizePolicy": QSizePolicy,
        "Qt": Qt,
        "QToolButton": QToolButton,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
    }


class CollapsibleSection:
    """Factory-backed collapsible section matching the plotting GUI behavior."""

    def __new__(
        cls,
        *,
        title: str,
        section_id: str,
        state_store: dict[str, bool],
        default_expanded: bool = False,
        subsection: bool = False,
        parent: Any | None = None,
    ) -> Any:
        qt = require_qt()
        QFrame = qt["QFrame"]
        QHBoxLayout = qt["QHBoxLayout"]
        QPropertyAnimation = qt["QPropertyAnimation"]
        QEasingCurve = qt["QEasingCurve"]
        QSizePolicy = qt["QSizePolicy"]
        Qt = qt["Qt"]
        QToolButton = qt["QToolButton"]
        QVBoxLayout = qt["QVBoxLayout"]

        class _Section(QFrame):
            def __init__(self) -> None:
                super().__init__(parent)
                self._section_id = str(section_id).strip()
                self._state_store = state_store
                self._body_widget: Any | None = None
                self._expanded = bool(state_store.get(self._section_id, default_expanded))
                self._collapse_after_animation = False
                self.setObjectName("collapsibleSubsection" if subsection else "collapsibleSection")

                root_layout = QVBoxLayout(self)
                root_layout.setContentsMargins(0, 0, 0, 0)
                root_layout.setSpacing(0)

                self.header_frame = QFrame(self)
                self.header_frame.setObjectName(
                    "collapsibleSubsectionHeader" if subsection else "collapsibleSectionHeader"
                )
                header_layout = QHBoxLayout(self.header_frame)
                header_layout.setContentsMargins(0, 0, 0, 0)
                header_layout.setSpacing(8)

                self.toggle_button = QToolButton(self.header_frame)
                self.toggle_button.setObjectName("collapsibleToggle")
                self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                self.toggle_button.setCheckable(True)
                self.toggle_button.setChecked(self._expanded)
                self.toggle_button.setText(title)
                self.toggle_button.clicked.connect(self._handle_toggle_clicked)
                self.toggle_button.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                header_layout.addWidget(self.toggle_button, 1, Qt.AlignmentFlag.AlignVCenter)
                root_layout.addWidget(self.header_frame)

                self.body_frame = QFrame(self)
                self.body_frame.setObjectName(
                    "collapsibleSubsectionBody" if subsection else "collapsibleSectionBody"
                )
                self.body_layout = QVBoxLayout(self.body_frame)
                self.body_layout.setContentsMargins(0, 0, 0, 0)
                self.body_layout.setSpacing(0)
                root_layout.addWidget(self.body_frame)

                self._animation = QPropertyAnimation(self.body_frame, b"maximumHeight", self)
                self._animation.setDuration(160)
                self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
                self._animation.finished.connect(self._handle_animation_finished)
                self._apply_expanded_state(self._expanded, animate=False, persist=False)

            def set_body_widget(self, widget: Any) -> None:
                self._body_widget = widget
                self.body_layout.addWidget(widget)
                self._apply_expanded_state(self._expanded, animate=False, persist=False)

            def _target_body_height(self) -> int:
                if self._body_widget is None:
                    return 0
                hint = self._body_widget.sizeHint()
                height = int(hint.height()) if hint is not None else 0
                if height <= 0:
                    layout_hint = self.body_layout.sizeHint()
                    height = int(layout_hint.height()) if layout_hint is not None else 0
                return max(0, height)

            def _handle_toggle_clicked(self, checked: bool) -> None:
                self._apply_expanded_state(bool(checked), animate=True, persist=True)

            def _apply_expanded_state(
                self,
                expanded: bool,
                *,
                animate: bool,
                persist: bool,
            ) -> None:
                self._expanded = bool(expanded)
                if persist and self._section_id:
                    self._state_store[self._section_id] = self._expanded
                self.toggle_button.blockSignals(True)
                try:
                    self.toggle_button.setChecked(self._expanded)
                    self.toggle_button.setArrowType(
                        Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
                    )
                finally:
                    self.toggle_button.blockSignals(False)

                target_height = self._target_body_height()
                can_animate = animate and self.isVisible() and target_height > 0 and self._body_widget is not None
                if not can_animate:
                    self._animation.stop()
                    self._collapse_after_animation = False
                    self.body_frame.setVisible(self._expanded)
                    self.body_frame.setMaximumHeight(16777215 if self._expanded else 0)
                    return
                start_height = max(0, int(self.body_frame.height()))
                self.body_frame.setVisible(True)
                self._animation.stop()
                self._animation.setStartValue(start_height)
                self._animation.setEndValue(target_height if self._expanded else 0)
                self._collapse_after_animation = not self._expanded
                self._animation.start()

            def _handle_animation_finished(self) -> None:
                if self._collapse_after_animation:
                    self.body_frame.setVisible(False)
                    self._collapse_after_animation = False
                if self._expanded:
                    self.body_frame.setMaximumHeight(16777215)

        return _Section()
