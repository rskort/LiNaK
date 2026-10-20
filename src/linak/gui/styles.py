"""Shared light styling for LiNaK GUI windows and widgets."""

from __future__ import annotations

from typing import Any

_LIGHT_COLORS: dict[str, str] = {
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
    "danger_bg": "#fde8ea",
    "danger_border": "#cc4f5a",
    "danger_text": "#7d1820",
    "success_bg": "#d8f3e9",
    "success_border": "#15946f",
    "success_text": "#053e31",
    "info_bg": "#d7f1f4",
    "info_border": "#087982",
    "info_text": "#082f34",
    "tooltip_bg": "#f8fbff",
    "tooltip_border": "#b7c6d8",
    "tooltip_text": "#102033",
    "placeholder_text": "#7b8797",
    "item_hover": "#edf4fa",
    "item_selected_bg": "#0f8f95",
    "item_selected_text": "#f8feff",
    "splitter": "#cad5e1",
    "scrollbar_track": "#edf2f7",
    "scrollbar_thumb": "#b9c5d4",
    "scrollbar_thumb_hover": "#95a6bb",
    "series_row_bg": "transparent",
    "series_row_text": "#142033",
    "series_row_disabled_text": "#6d7b8e",
    "series_row_selected_bg": "#cbecef",
    "series_row_selected_border": "#0c7a80",
    "series_row_selected_text": "#082f34",
    "series_row_button_hover": "#d8edf0",
    "series_row_swatch_bg": "#ffffff",
    "series_row_swatch_border": "#93a4b8",
    "series_badge_original_bg": "#e7edf5",
    "series_badge_original_border": "#9aacbf",
    "series_badge_original_text": "#23324a",
    "series_badge_copy_bg": "#fff0bd",
    "series_badge_copy_border": "#c6941f",
    "series_badge_copy_text": "#4f3600",
    "selected_card_bg": "#d7f1f4",
    "selected_card_border": "#087982",
}


def light_colors() -> dict[str, str]:
    """Return an independent copy of the fixed GUI colours."""
    return _LIGHT_COLORS.copy()


def apply_light_palette(widget: Any) -> None:
    """Give a GUI window and its owned dialogs explicit, system-independent colours."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    colors = light_colors()
    palette = QPalette()
    roles = {
        "Window": "window_bg",
        "WindowText": "text",
        "Base": "input_bg",
        "AlternateBase": "card_bg",
        "Text": "text",
        "Button": "button_bg",
        "ButtonText": "text",
        "BrightText": "accent_text",
        "Highlight": "accent",
        "HighlightedText": "accent_text",
        "ToolTipBase": "tooltip_bg",
        "ToolTipText": "tooltip_text",
        "PlaceholderText": "placeholder_text",
        "Link": "accent",
        "LinkVisited": "accent_hover",
        "Light": "panel_elevated",
        "Midlight": "header_bg",
        "Mid": "border",
        "Dark": "border_soft",
        "Shadow": "muted_text",
    }
    for role, token in roles.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(colors[token]))
    for role in ("WindowText", "Text", "ButtonText", "PlaceholderText"):
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            getattr(QPalette.ColorRole, role),
            QColor(colors["disabled_text"]),
        )
    for role in ("Base", "Button"):
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            getattr(QPalette.ColorRole, role),
            QColor(colors["disabled_bg"]),
        )
    # Qt popups are top-level windows and use application defaults, not the parent palette.
    app = QApplication.instance()
    if app is not None:
        app.setPalette(palette)
    widget.setPalette(palette)
    widget.setAttribute(Qt.WidgetAttribute.WA_WindowPropagation, True)


def workspace_stylesheet(colors: dict[str, str]) -> str:
    """Return a project-workspace stylesheet using plotting GUI token names."""

    return (
        f"QWidget#windowRoot {{ background-color: {colors['window_bg']}; color: {colors['text']}; }}"
        f"QFrame#appHeader {{ background-color: {colors['header_bg']}; border: 1px solid {colors['border']}; border-radius: 14px; }}"
        f"QFrame#navPanel, QFrame#inspectorPanel, QFrame#previewPanel, QFrame#taskPanel, QFrame#logPanel {{"
        f" background-color: {colors['panel_bg']}; border: 1px solid {colors['border']}; border-radius: 14px; }}"
        f"QFrame#previewFrame, QFrame#Card, QFrame#actionRow {{"
        f" background-color: {colors['card_bg']}; border: 1px solid {colors['border_soft']}; border-radius: 10px; }}"
        f"QFrame#actionRow:hover {{ background-color: {colors['panel_elevated']}; border-color: {colors['border']}; }}"
        f"QFrame#collapsibleSection, QFrame#collapsibleSubsection {{"
        f" background-color: {colors['card_bg']}; border: 1px solid {colors['border_soft']}; border-radius: 12px; }}"
        f"QFrame#collapsibleSubsection {{ background-color: {colors['panel_elevated']}; border-radius: 10px; }}"
        f"QFrame#collapsibleSectionHeader, QFrame#collapsibleSubsectionHeader, "
        f"QFrame#collapsibleSectionBody, QFrame#collapsibleSubsectionBody {{ background: transparent; border: none; }}"
        f"QWidget {{ color: {colors['text']}; font-size: 10pt; }}"
        f"QLabel {{ color: {colors['text']}; background: transparent; }}"
        f"QLabel#appTitle {{ font-size: 22px; font-weight: 700; color: {colors['heading']}; }}"
        f"QLabel#appSubtitle, QLabel#MutedText, QLabel#sectionNote {{ color: {colors['muted_text']}; }}"
        f"QLabel#SectionTitle, QLabel#pageTitle {{ font-size: 16px; font-weight: 700; color: {colors['heading']}; }}"
        f"QLabel#cardTitle {{ font-weight: 700; color: {colors['heading']}; }}"
        f"QLabel#inlineWarning {{ padding: 8px 10px; border-radius: 8px; background-color: {colors['warning_bg']};"
        f" border: 1px solid {colors['warning_border']}; color: {colors['warning_text']}; }}"
        f"QPushButton {{ padding: 7px 12px; border: 1px solid {colors['border']}; border-radius: 8px;"
        f" background-color: {colors['button_bg']}; color: {colors['text']}; }}"
        f"QPushButton:hover {{ border-color: {colors['accent']}; background-color: {colors['button_hover']}; }}"
        f"QPushButton:pressed {{ background-color: {colors['button_pressed']}; }}"
        f"QPushButton:disabled {{ background-color: {colors['disabled_bg']}; color: {colors['disabled_text']}; border-color: {colors['border_soft']}; }}"
        f"QPushButton#PrimaryButton, QPushButton[role='primary'] {{ background-color: {colors['accent']}; color: {colors['accent_text']};"
        f" border-color: {colors['accent']}; font-weight: 700; }}"
        f"QPushButton#PrimaryButton:hover, QPushButton[role='primary']:hover {{ background-color: {colors['accent_hover']}; border-color: {colors['accent_hover']}; }}"
        f"QPushButton#SecondaryButton {{ background-color: {colors['button_bg']}; }}"
        f"QToolButton {{ border: 1px solid {colors['border']}; border-radius: 7px; background-color: {colors['button_bg']}; color: {colors['text']}; }}"
        f"QToolButton:hover {{ border-color: {colors['accent']}; background-color: {colors['button_hover']}; }}"
        f"QToolButton#collapsibleToggle {{ padding: 10px 12px; border: none; border-radius: 10px; background: transparent;"
        f" color: {colors['heading']}; font-weight: 600; text-align: left; }}"
        f"QToolButton#collapsibleToggle:hover {{ background-color: {colors['nav_hover']}; border: none; }}"
        f"QLineEdit, QComboBox, QPlainTextEdit, QListWidget {{ border: 1px solid {colors['input_border']}; border-radius: 8px;"
        f" background-color: {colors['input_bg']}; color: {colors['text']}; outline: none;"
        f" selection-background-color: {colors['accent_soft']}; selection-color: {colors['text']}; }}"
        f"QLineEdit, QComboBox {{ padding: 6px 8px; min-height: 18px; }}"
        f"QPlainTextEdit {{ padding: 6px; font-family: Consolas, monospace; }}"
        f"QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QListWidget:disabled {{"
        f" background-color: {colors['disabled_bg']}; color: {colors['disabled_text']}; border-color: {colors['border_soft']}; }}"
        f"QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {{ border-color: {colors['accent']}; }}"
        f"QComboBox::drop-down {{ border: none; width: 24px; }}"
        f"QComboBox QAbstractItemView, QAbstractItemView {{ background-color: {colors['panel_elevated']}; color: {colors['text']};"
        f" border: 1px solid {colors['border']}; selection-background-color: {colors['item_selected_bg']}; selection-color: {colors['item_selected_text']}; }}"
        f"QAbstractItemView::item {{ padding: 7px 9px; }}"
        f"QAbstractItemView::item:hover {{ background-color: {colors['item_hover']}; color: {colors['text']}; }}"
        f"QAbstractItemView::item:selected, QAbstractItemView::item:selected:active, QAbstractItemView::item:selected:!active {{"
        f" background-color: {colors['item_selected_bg']}; color: {colors['item_selected_text']}; }}"
        f"QGroupBox {{ background-color: {colors['card_bg']}; border: 1px solid {colors['border_soft']}; border-radius: 12px; margin-top: 14px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; color: {colors['heading']}; font-weight: 600; }}"
        f"QCheckBox {{ spacing: 6px; }}"
        f"QCheckBox::indicator, QAbstractItemView::indicator {{ width: 16px; height: 16px; border-radius: 4px;"
        f" border: 1px solid {colors['input_border']}; background-color: {colors['input_bg']}; }}"
        f"QCheckBox::indicator:hover, QAbstractItemView::indicator:hover {{ border-color: {colors['accent']}; }}"
        f"QCheckBox::indicator:checked, QAbstractItemView::indicator:checked {{ background-color: {colors['accent']}; border-color: {colors['accent']}; }}"
        f"QSplitter::handle {{ background-color: {colors['splitter']}; }}"
        f"QScrollBar:vertical, QScrollBar:horizontal {{ background: {colors['scrollbar_track']}; border: none; margin: 0px; }}"
        f"QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {colors['scrollbar_thumb']}; border-radius: 4px; min-height: 24px; min-width: 24px; }}"
        f"QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {colors['scrollbar_thumb_hover']}; }}"
        f"QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}"
        f"QToolTip {{ background-color: {colors['tooltip_bg']}; color: {colors['tooltip_text']};"
        f" border: 1px solid {colors['tooltip_border']}; padding: 6px 8px; }}"
        f"QMessageBox {{ background-color: {colors['panel_bg']}; color: {colors['text']}; }}"
        f"QMessageBox QLabel {{ color: {colors['text']}; background: transparent; }}"
        f"QMessageBox QPushButton, QDialogButtonBox QPushButton {{ padding: 7px 12px; border: 1px solid {colors['border']};"
        f" border-radius: 8px; background-color: {colors['button_bg']}; color: {colors['text']}; min-width: 88px; }}"
        f"QMessageBox QPushButton:hover, QDialogButtonBox QPushButton:hover {{ border-color: {colors['accent']}; background-color: {colors['button_hover']}; }}"
        f"QMessageBox QPushButton:pressed, QDialogButtonBox QPushButton:pressed {{ background-color: {colors['button_pressed']}; }}"
    )


WORKSPACE_STYLESHEET = workspace_stylesheet(light_colors())


def _badge_colors(colors: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    return {
        "success": (colors["success_text"], colors["success_bg"], colors["success_border"]),
        "warning": (colors["warning_text"], colors["warning_bg"], colors["warning_border"]),
        "danger": (colors["danger_text"], colors["danger_bg"], colors["danger_border"]),
        "info": (colors["info_text"], colors["info_bg"], colors["info_border"]),
        "neutral": (colors["badge_text"], colors["badge_bg"], colors["badge_border"]),
        "queued": (colors["muted_text"], colors["panel_elevated"], colors["border"]),
        "canceling": (colors["warning_text"], colors["warning_bg"], colors["warning_border"]),
        "canceled": (colors["muted_text"], colors["button_bg"], colors["border_soft"]),
        "external": (colors["badge_text"], colors["badge_bg"], colors["badge_border"]),
        "generated": (colors["info_text"], colors["info_bg"], colors["info_border"]),
        "running": (colors["info_text"], colors["info_bg"], colors["info_border"]),
    }


def _clamp_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def badge_style(
    tone: str,
    *,
    progress_fraction: float | None = None,
    colors: dict[str, str] | None = None,
) -> str:
    tokens = colors or light_colors()
    badge_colors = _badge_colors(tokens)
    fg, bg, border = badge_colors.get(tone, badge_colors["neutral"])
    fraction = _clamp_fraction(progress_fraction)
    if tone == "running" and fraction is not None:
        stop = int(round(fraction * 100.0))
        return (
            f"color: {tokens['heading']}; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {tokens['accent']}, stop:{fraction:.3f} {tokens['accent_hover']}, "
            f"stop:{fraction:.3f} {tokens['info_bg']}, stop:1 {tokens['info_bg']}); "
            f"border: 1px solid {tokens['accent']}; border-radius: 999px; "
            "padding: 3px 8px; font-weight: 700;"
            f" /* progress {stop}% */"
        )
    return (
        f"color: {fg}; background: {bg}; border: 1px solid {border}; "
        "border-radius: 999px; padding: 3px 8px; font-weight: 600;"
    )
