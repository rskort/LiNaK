"""Plotting-GUI-aligned theme helpers for LiNaK Qt tools."""

from __future__ import annotations

from typing import Any, Literal

ThemeMode = Literal["system", "light", "dark"]


def plot_like_theme_tokens(is_dark: bool) -> dict[str, str]:
    """Return the same core color tokens used by the plotting GUI."""

    if is_dark:
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
            "danger_bg": "#3b1518",
            "danger_border": "#d05b63",
            "danger_text": "#ffd6d9",
            "success_bg": "#12352d",
            "success_border": "#39b990",
            "success_text": "#cffff0",
            "info_bg": "#123c46",
            "info_border": "#34bed1",
            "info_text": "#e0fbff",
            "tooltip_bg": "#1a2435",
            "tooltip_border": "#4a607e",
            "tooltip_text": "#f7fbff",
            "placeholder_text": "#7f8fa6",
            "item_hover": "#1c2c40",
            "item_selected_bg": "#2aa7b8",
            "item_selected_text": "#07151b",
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
    }


def is_dark_theme(mode: ThemeMode, widget: Any) -> bool:
    """Resolve explicit or system theme state like the plotting GUI."""

    if mode == "dark":
        return True
    if mode == "light":
        return False
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    palette = app.palette() if app is not None else widget.palette()
    window_color = palette.color(QPalette.ColorRole.Window)
    text_color = palette.color(QPalette.ColorRole.WindowText)
    return window_color.lightness() < text_color.lightness()


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
        f"QCheckBox#themeSwitch {{ padding: 6px 10px; border: 1px solid {colors['border']}; border-radius: 999px;"
        f" background-color: {colors['button_bg']}; color: {colors['text']}; font-weight: 600; }}"
        f"QCheckBox#themeSwitch:hover {{ border-color: {colors['accent']}; background-color: {colors['button_hover']}; }}"
        f"QCheckBox::indicator, QAbstractItemView::indicator {{ width: 16px; height: 16px; border-radius: 4px;"
        f" border: 1px solid {colors['input_border']}; background-color: {colors['input_bg']}; }}"
        f"QCheckBox::indicator:hover, QAbstractItemView::indicator:hover {{ border-color: {colors['accent']}; }}"
        f"QCheckBox::indicator:checked, QAbstractItemView::indicator:checked {{ background-color: {colors['accent']}; border-color: {colors['accent']}; }}"
        f"QCheckBox#themeSwitch::indicator {{ width: 34px; height: 18px; border-radius: 9px;"
        f" background-color: {colors['input_bg']}; border: 1px solid {colors['input_border']}; }}"
        f"QCheckBox#themeSwitch::indicator:checked {{ background-color: {colors['accent']}; border-color: {colors['accent']}; }}"
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
