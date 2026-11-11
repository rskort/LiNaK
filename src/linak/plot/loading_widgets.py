"""One compact activity presentation for Studio startup and plot previews."""

from __future__ import annotations

from time import perf_counter

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .gui_session import PlotProgress


class _ActivitySpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.angle = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ring = QRectF(3, 3, 22, 22)
        painter.setPen(QPen(QColor("#e2e8ee"), 2.5))
        painter.drawEllipse(ring)
        painter.setPen(QPen(QColor("#0f8f95"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(ring, -self.angle * 16, -100 * 16)


class StudioActivityIndicator(QFrame):
    """A layout keeps the spinner, caption and quiet detail line apart at any DPI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("studioActivity")
        self.setMaximumWidth(380)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setStyleSheet("""
            QFrame#studioActivity {
                background: #ffffff; border: 1px solid #dfe7ed; border-radius: 12px;
            }
            QLabel#activityTitle {
                color: #223448; font-size: 14px; font-weight: 600;
                background: transparent; border: none;
            }
            QLabel#activityDetail {
                color: #64748b; font-size: 12px; font-weight: 400;
                background: transparent; border: none;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(12)
        self.spinner = _ActivitySpinner(self)
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.title_label = QLabel(self)
        self.title_label.setObjectName("activityTitle")
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        self.detail_label = QLabel(self)
        self.detail_label.setObjectName("activityDetail")
        self.detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setMinimumWidth(0)
        self.detail_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.detail_label)
        self._progress = PlotProgress("Loading plot data")
        self._started = perf_counter()
        self._last_second = -1
        self._active = False
        self.animation = QTimer(self)
        self.animation.setInterval(40)
        self.animation.timeout.connect(self._tick)
        self.set_progress(self._progress)

    def sizeHint(self):
        # Keep the card steady as short captions change; reserve room for two lines.
        return QSize(320, 156)

    def set_progress(self, progress: PlotProgress) -> None:
        self._progress = progress
        if self.title_label.text() != progress.title:
            self.title_label.setText(progress.title)
        self.setAccessibleName(progress.title)
        self.setToolTip(progress.message)
        self._refresh_detail()

    def set_active(self, active: bool) -> None:
        if active and not self._active:
            self._started = perf_counter()
            self._last_second = -1
        self._active = active
        self.spinner.setVisible(active)
        if active and self.isVisible():
            self.animation.start()
        else:
            self.animation.stop()
        self._refresh_detail()

    def _refresh_detail(self):
        from pathlib import Path

        progress = self._progress
        parts = []
        if progress.index is not None and progress.total:
            parts.append(f"{progress.index} of {progress.total}")
        elapsed = int(perf_counter() - self._started)
        if self._active and elapsed >= 10:
            parts.append(f"{elapsed} s")
        suffix = " · ".join(parts)
        metrics = self.detail_label.fontMetrics()
        available = max(0, self.width() - 56)
        if progress.source:
            separator = " · " if suffix else ""
            file_space = max(0, available - metrics.horizontalAdvance(separator + suffix))
            filename = metrics.elidedText(
                Path(progress.source).name, Qt.TextElideMode.ElideMiddle, file_space
            )
            text = filename + (separator if filename else "") + suffix
        else:
            text = suffix
        self.detail_label.setText(text)
        # Reserve one line even without details so phase changes never move the card.
        self.detail_label.setMinimumHeight(self.detail_label.fontMetrics().height())
        self.setAccessibleDescription(
            progress.message + (f"; {elapsed} seconds" if self._active else "")
        )

    def _tick(self):
        self.spinner.angle = (self.spinner.angle + 12) % 360
        self.spinner.update()
        second = int(perf_counter() - self._started)
        if second != self._last_second:
            self._last_second = second
            self._refresh_detail()

    def showEvent(self, event):
        super().showEvent(event)
        if self._active:
            self.animation.start()
        self._refresh_detail()

    def hideEvent(self, event):
        self.animation.stop()
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_detail()


class PreviewLoadingOverlay(QWidget):
    """One delayed indicator over a retained plot; no text painted over text."""

    def __init__(self, target: QWidget):
        super().__init__(target)
        self._target = target
        self._active = False
        self.indicator = StudioActivityIndicator(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addStretch()
        layout.addWidget(self.indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        self.animation = self.indicator.animation
        self.delay = QTimer(self)
        self.delay.setSingleShot(True)
        self.delay.setInterval(150)
        self.delay.timeout.connect(self._show_loading)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.WaitCursor)
        target.installEventFilter(self)
        self.hide()

    def set_message(self, message: str) -> None:
        self.set_progress(PlotProgress(message))

    def set_progress(self, progress: PlotProgress) -> None:
        self.indicator.set_progress(progress)
        self.setAccessibleName(progress.title)
        self.setToolTip(progress.message)

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        self.indicator.set_active(active)
        if active:
            self.set_progress(PlotProgress("Loading plot data"))
            self.delay.start()
        else:
            self.delay.stop()
            self.hide()

    def _show_loading(self):
        if self._active and self._target.isVisible():
            self.setGeometry(self._target.rect())
            self.show()
            self.raise_()

    def eventFilter(self, watched: QObject, event):
        if watched is self._target:
            if event.type() == QEvent.Type.Resize:
                self.setGeometry(self._target.rect())
            elif event.type() == QEvent.Type.Hide:
                self.hide()
            elif event.type() == QEvent.Type.Show and self._active and not self.delay.isActive():
                self._show_loading()
        return False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255, 220))

    def shutdown(self):
        self.set_active(False)
        self._target.removeEventFilter(self)
