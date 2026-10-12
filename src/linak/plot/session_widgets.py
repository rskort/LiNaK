"""Qt widgets for plot-session diagnostics and preview activity.

Imported only by the GUI launcher, keeping Qt optional for CLI use.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPointF, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
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


class SessionFooter(QLabel):
    activated = Signal()

    def __init__(self, session: SessionLog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._message = ""
        self._display = ""
        self._level = logging.INFO
        self._dark = False
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

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self._elide()

    def _elide(self) -> None:
        # One plain-text line, including messages containing HTML or newlines.
        color = (
            ("#f87171" if self._dark else "#b91c1c")
            if self._level >= logging.ERROR
            else (
                ("#fbbf24" if self._dark else "#92400e") if self._level >= logging.WARNING else ""
            )
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

    def __init__(self, target: QWidget, is_dark: Callable[[], bool]) -> None:
        super().__init__(target)
        self._target = target
        self._is_dark = is_dark
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
        dark = self._is_dark()
        background = QColor("#101827" if dark else "#ffffff")
        background.setAlpha(170)
        painter.fillRect(self.rect(), background)
        foreground = QColor("#e5e7eb" if dark else "#334155")
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
