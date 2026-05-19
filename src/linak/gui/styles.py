"""Shared styling helpers for the project workspace."""

from __future__ import annotations

from .theme import plot_like_theme_tokens, workspace_stylesheet

WORKSPACE_STYLESHEET = workspace_stylesheet(plot_like_theme_tokens(False))


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
    tokens = colors or plot_like_theme_tokens(False)
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
