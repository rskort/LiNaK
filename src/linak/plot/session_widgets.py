"""Qt widgets for plot-session diagnostics and preview activity.

Imported only by the GUI launcher, keeping Qt optional for CLI use.
"""

from __future__ import annotations

import logging
import json
import math
import re
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPointF, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .session_log import SessionLog, SessionMessage


class JsonOptionsEditor(QWidget):
    """An optional JSON object with local validation and reversible editing aids."""

    def __init__(self, description: str, example: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.example = example
        self.is_valid = True
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(6)
        explanation = QLabel(description, self)
        explanation.setWordWrap(True)
        explanation.setObjectName("sectionNote")
        layout.addWidget(explanation)
        self.editor = QPlainTextEdit(self)
        self.editor.setFixedHeight(100)
        self.editor.setPlaceholderText(example)
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance(" ") * 4)
        self.editor.setMinimumWidth(0)
        layout.addWidget(self.editor)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.format_button = QPushButton("Format JSON", self)
        self.format_button.setToolTip("Indent this JSON object. Its values stay the same.")
        self.format_button.clicked.connect(self.format_json)
        self.example_button = QPushButton("Use example", self)
        self.example_button.setToolTip("Insert the example into an empty field.\n" + example)
        self.example_button.clicked.connect(self.use_example)
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.setToolTip("Clear this field. You can undo this edit with Ctrl+Z.")
        self.clear_button.clicked.connect(lambda: self._replace_text(""))
        for button in (self.format_button, self.example_button, self.clear_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.validation_label = QLabel(self)
        self.validation_label.setTextFormat(Qt.TextFormat.PlainText)
        self.validation_label.setWordWrap(True)
        self.validation_label.setAccessibleName("JSON validation")
        layout.addWidget(self.validation_label)
        self.editor.textChanged.connect(self.validate)
        self.validate()

    def validate(self) -> None:
        raw = self.editor.toPlainText().strip()
        self.is_valid = False
        try:
            value = json.loads(raw) if raw else {}
            if not isinstance(value, dict):
                raise ValueError('Use a JSON object: {"option": value}.')
        except json.JSONDecodeError as error:
            message = f"Line {error.lineno}, column {error.colno}: {error.msg}"
        except ValueError as error:
            message = str(error)
        else:
            self.is_valid = True
            count = len(value)
            message = (
                f"{count} option{'s' if count != 1 else ''} configured"
                if count
                else "No extra options"
            )
        self.validation_label.setText(message)
        self.validation_label.setStyleSheet("" if self.is_valid else "color: #b91c1c;")
        self.format_button.setEnabled(self.is_valid and bool(raw))
        self.example_button.setEnabled(not raw)
        self.clear_button.setEnabled(bool(raw))

    def _replace_text(self, text: str) -> None:
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)
        cursor.endEditBlock()
        self.editor.setTextCursor(cursor)

    def format_json(self) -> None:
        if self.is_valid and self.editor.toPlainText().strip():
            self._replace_text(json.dumps(json.loads(self.editor.toPlainText()), indent=2))

    def use_example(self) -> None:
        if not self.editor.toPlainText().strip():
            self._replace_text(self.example)


class SessionFooter(QLabel):
    activated = Signal()

    def __init__(self, session: SessionLog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._message = ""
        self._display = ""
        self._level = logging.INFO
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(False)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Latest message; open session logs")

    def sizeHint(self) -> QSize:
        return QSize(0, self.fontMetrics().height() + 20)

    def text(self) -> str:
        # Existing validation code needs the full message, rather than elided text.
        return self._message

    def setText(self, text: str) -> None:
        latest = self.session.latest()
        if latest is not None and latest.source == "GUI" and latest.message == text:
            return
        lowered = str(text).lower()
        level = (
            logging.ERROR
            if "failed" in lowered or lowered.startswith("invalid ")
            else (logging.WARNING if lowered.startswith("warning:") else logging.INFO)
        )
        self.session.publish(str(text), level=level)

    def display_message(self, entry: SessionMessage | None) -> None:
        self._message = entry.message if entry else "Ready."
        self._level = entry.level if entry else logging.INFO
        plain = re.sub(r"\x1b\[[0-9;]*m", "", self._message)
        self._display = " ".join(plain.split())
        if self._level >= logging.WARNING:
            self._display = re.sub(r"^(warning|error):\s*", "", self._display, flags=re.IGNORECASE)
        if self._level >= logging.ERROR:
            self._display = "Error · " + self._display
        elif self._level >= logging.WARNING:
            self._display = "Warning · " + self._display
        self.setToolTip(
            (entry.formatted() if entry else "Ready.") + "\nClick to open Advanced → Logs."
        )
        self._elide()

    def _elide(self) -> None:
        # One plain-text line, including messages containing HTML or newlines.
        color = (
            "#b91c1c"
            if self._level >= logging.ERROR
            else ("#92400e" if self._level >= logging.WARNING else "")
        )
        self.setStyleSheet(f"color: {color};" if color else "")
        available = max(0, self.contentsRect().width() - 12)
        super().setText(
            self.fontMetrics().elidedText(self._display, Qt.TextElideMode.ElideRight, available)
        )
        self.setAccessibleDescription(self._display)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._elide()

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if self.hasFocus():
            painter = QPainter(self)
            painter.setPen(QPen(self.palette().highlight().color(), 1))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 3, 3)
            painter.end()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class SessionDiagnostics(QObject):
    changed = Signal()

    def __init__(self, session: SessionLog, parent: QWidget) -> None:
        super().__init__(parent)
        self.session = session
        self._revision = -1
        self._count = 0
        self._closed = False
        self.footer = SessionFooter(session, parent)
        self.body = QWidget(parent)
        self.body.setMinimumWidth(0)
        self.body.setMaximumWidth(960)
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.copy_button = QPushButton("Copy all", self.body)
        self.copy_button.clicked.connect(self.copy_all)
        actions.addWidget(self.copy_button)
        self.clear_button = QPushButton("Clear", self.body)
        self.clear_button.clicked.connect(session.clear)
        actions.addWidget(self.clear_button)
        layout.addLayout(actions)
        self.viewer = QPlainTextEdit(self.body)
        self.viewer.setReadOnly(True)
        self.viewer.setFixedHeight(240)
        self.viewer.setMinimumWidth(0)
        self.viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.viewer.setPlaceholderText("Messages from this plot window session appear here.")
        self.viewer.setAccessibleName("Session logs")
        layout.addWidget(self.viewer)
        self.changed.connect(self._drain)
        session.subscribe(self._notify)
        self._drain()

    def _notify(self) -> None:
        # Only signal emission occurs on a worker thread. AutoConnection delivers
        # the slot to the GUI thread, which drains entries in collection order.
        if self._closed:
            return
        try:
            self.changed.emit()
        except RuntimeError:
            pass

    def _drain(self) -> None:
        if self._closed:
            return
        revision, count, new_entries = self.session.read_since(self._revision, self._count)
        scroll = self.viewer.verticalScrollBar()
        position = scroll.value()
        following = position >= scroll.maximum() - 2
        cleared = revision != self._revision
        if cleared:
            self.viewer.clear()
            following = True
        for entry in new_entries:
            self.viewer.appendPlainText(entry.formatted())
        if new_entries or cleared:
            self.footer.display_message(new_entries[-1] if new_entries else None)
        self._revision, self._count = revision, count
        scroll.setValue(scroll.maximum() if following else position)
        self.copy_button.setEnabled(count > 0)
        self.clear_button.setEnabled(count > 0)

    def copy_all(self) -> None:
        QApplication.clipboard().setText(
            "\n".join(entry.formatted() for entry in self.session.snapshot())
        )

    def shutdown(self) -> None:
        self._closed = True
        self.session.close()


class PreviewLoadingOverlay(QWidget):
    """Dim retained content and animate a small spinner only during rendering."""

    def __init__(self, target: QWidget) -> None:
        super().__init__(target)
        self._target = target
        self._active = False
        self._angle = 0
        self.delay = QTimer(self)
        self.delay.setSingleShot(True)
        self.delay.setInterval(150)
        self.delay.timeout.connect(self._show_loading)
        self.animation = QTimer(self)
        self.animation.setInterval(80)
        self.animation.timeout.connect(self._tick)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.setAccessibleName("Updating preview")
        target.installEventFilter(self)
        self.hide()

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        if active:
            self.delay.start()
        else:
            self.delay.stop()
            self.animation.stop()
            self.hide()

    def _show_loading(self) -> None:
        if not self._active:
            return
        self.setGeometry(self._target.rect())
        self.show()
        self.raise_()
        if self._target.isVisible():
            self.animation.start()

    def _tick(self) -> None:
        self._angle = (self._angle + 1) % 12
        self.update()

    def eventFilter(self, watched: QObject, event: Any) -> bool:
        if watched is self._target:
            if event.type() == QEvent.Type.Resize:
                self.setGeometry(self._target.rect())
            elif event.type() == QEvent.Type.Hide:
                self.animation.stop()
            elif event.type() == QEvent.Type.Show and self._active and not self.delay.isActive():
                self._show_loading()
        return False

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        background = QColor("#ffffff")
        background.setAlpha(170)
        painter.fillRect(self.rect(), background)
        foreground = QColor("#334155")
        center_x, center_y = self.width() // 2, self.height() // 2 - 12
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index in range(12):
            color = QColor(foreground)
            color.setAlpha(45 + ((index - self._angle) % 12) * 18)
            pen = QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            angle = math.radians(index * 30)
            painter.drawLine(
                QPointF(center_x + 10 * math.cos(angle), center_y + 10 * math.sin(angle)),
                QPointF(center_x + 16 * math.cos(angle), center_y + 16 * math.sin(angle)),
            )
        painter.setPen(foreground)
        painter.drawText(
            self.rect().adjusted(0, 36, 0, 0), Qt.AlignmentFlag.AlignCenter, "Updating preview…"
        )
        painter.end()

    def shutdown(self) -> None:
        self.set_active(False)
        self._target.removeEventFilter(self)
