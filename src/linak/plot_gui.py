"""Tkinter GUI panel for interactive plot settings."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

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


def _optional_string_list(value: str) -> list[str] | None:
    stripped = value.strip()
    if not stripped:
        return None
    values = [token.strip() for token in stripped.split(",")]
    values = [token for token in values if token]
    return values or None


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


def _format_string_list(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return ", ".join(str(item) for item in value if str(item).strip())


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


def _set_widget_enabled(widget: Any, *, enabled: bool) -> None:
    try:
        if enabled:
            widget.state(["!disabled"])
        else:
            widget.state(["disabled"])
        return
    except Exception:
        pass

    try:
        widget.configure(state="normal" if enabled else "disabled")
    except Exception:
        return


def launch_plot_settings_panel(
    *,
    title: str,
    initial_settings: dict[str, Any],
    on_preview: Callable[[dict[str, Any]], None],
    on_save: Callable[[dict[str, Any]], str],
    on_save_figure: Callable[[dict[str, Any], str], str] | None = None,
) -> None:
    """Open a Tk panel that previews and persists plot settings."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter is unavailable; cannot open GUI plot controls. "
            "Use CLI plot flags or install Tk support."
        ) from exc

    root = tk.Tk()
    root.title(title)
    root.minsize(920, 700)

    main = ttk.Frame(root, padding=12)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(0, weight=1)

    notebook = ttk.Notebook(main)
    notebook.grid(row=0, column=0, sticky="nsew")

    defaults = DEFAULT_PLOT_STYLE

    title_var = tk.StringVar(value=str(initial_settings.get("title") or ""))
    x_label_var = tk.StringVar(value=str(initial_settings.get("x_label") or ""))
    y_label_var = tk.StringVar(value=str(initial_settings.get("y_label") or ""))
    title_mode_var = tk.StringVar(value=_toggle_to_mode(initial_settings.get("title_visible")))
    legend_mode_var = tk.StringVar(value=_toggle_to_mode(initial_settings.get("legend")))
    grid_mode_var = tk.StringVar(value=_toggle_to_mode(initial_settings.get("grid")))
    ticks_mode_var = tk.StringVar(value=_toggle_to_mode(initial_settings.get("ticks")))
    markers_mode_var = tk.StringVar(value=_toggle_to_mode(initial_settings.get("markers")))
    legend_title_var = tk.StringVar(value=str(initial_settings.get("legend_title") or ""))
    legend_loc_var = tk.StringVar(value=str(initial_settings.get("legend_loc") or "best"))

    x_scale_var = tk.StringVar(value=str(initial_settings.get("x_scale") or "linear"))
    y_scale_var = tk.StringVar(value=str(initial_settings.get("y_scale") or "linear"))
    x_min_var = tk.StringVar(value=_extract_limit(initial_settings, key="x_lim", index=0))
    x_max_var = tk.StringVar(value=_extract_limit(initial_settings, key="x_lim", index=1))
    y_min_var = tk.StringVar(value=_extract_limit(initial_settings, key="y_lim", index=0))
    y_max_var = tk.StringVar(value=_extract_limit(initial_settings, key="y_lim", index=1))
    x_ticks_var = tk.StringVar(value=_format_float_list(initial_settings.get("x_ticks")))
    y_ticks_var = tk.StringVar(value=_format_float_list(initial_settings.get("y_ticks")))
    x_tick_rotation_var = tk.StringVar(value=str(initial_settings.get("x_tick_rotation") or ""))
    y_tick_rotation_var = tk.StringVar(value=str(initial_settings.get("y_tick_rotation") or ""))

    fig_width_var = tk.StringVar(
        value=_extract_figsize_dimension(
            initial_settings, index=0, fallback=defaults.figure_size[0]
        )
    )
    fig_height_var = tk.StringVar(
        value=_extract_figsize_dimension(
            initial_settings, index=1, fallback=defaults.figure_size[1]
        )
    )
    dpi_var = tk.StringVar(value=str(initial_settings.get("dpi") or defaults.dpi))
    font_family_var = tk.StringVar(value=str(initial_settings.get("font_family") or ""))
    title_font_var = tk.StringVar(
        value=str(initial_settings.get("title_font_size") or defaults.title_font_size)
    )
    label_font_var = tk.StringVar(
        value=str(initial_settings.get("label_font_size") or defaults.label_font_size)
    )
    tick_font_var = tk.StringVar(
        value=str(initial_settings.get("tick_font_size") or defaults.tick_font_size)
    )
    line_width_var = tk.StringVar(value=str(initial_settings.get("line_width") or defaults.line_width))
    line_color_var = tk.StringVar(value=str(initial_settings.get("line_color") or ""))
    line_colors_var = tk.StringVar(value=_format_string_list(initial_settings.get("line_colors")))
    grid_linestyle_var = tk.StringVar(value=str(initial_settings.get("grid_linestyle") or ""))
    grid_linewidth_var = tk.StringVar(
        value=str(initial_settings.get("grid_linewidth") or defaults.grid_linewidth)
    )
    grid_alpha_var = tk.StringVar(value=str(initial_settings.get("grid_alpha") or defaults.grid_alpha))
    series_labels_var = tk.StringVar(value=_format_string_list(initial_settings.get("series_labels")))

    status_var = tk.StringVar(value="Ready.")

    def _signature(settings: dict[str, Any]) -> str:
        return json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)

    def _add_entry_row(
        parent: Any,
        *,
        row: int,
        label: str,
        variable: Any,
        width: int = 30,
    ) -> Any:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        widget = ttk.Entry(parent, textvariable=variable, width=width)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        return widget

    def _add_combo_row(
        parent: Any,
        *,
        row: int,
        label: str,
        variable: Any,
        values: tuple[str, ...],
        width: int = 18,
    ) -> Any:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        widget = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=width,
        )
        widget.grid(row=row, column=1, sticky="w", pady=4)
        return widget

    tab_general = ttk.Frame(notebook, padding=10)
    tab_axes = ttk.Frame(notebook, padding=10)
    tab_style = ttk.Frame(notebook, padding=10)
    tab_series = ttk.Frame(notebook, padding=10)
    for tab in (tab_general, tab_axes, tab_style, tab_series):
        tab.columnconfigure(1, weight=1)
    notebook.add(tab_general, text="General")
    notebook.add(tab_axes, text="Axes")
    notebook.add(tab_style, text="Style")
    notebook.add(tab_series, text="Series")

    _add_combo_row(
        tab_general,
        row=0,
        label="Title",
        variable=title_mode_var,
        values=("auto", "on", "off"),
    )
    title_entry = _add_entry_row(tab_general, row=1, label="Title text", variable=title_var, width=48)
    _add_entry_row(tab_general, row=2, label="X label", variable=x_label_var, width=48)
    _add_entry_row(tab_general, row=3, label="Y label", variable=y_label_var, width=48)

    _add_combo_row(
        tab_general,
        row=4,
        label="Legend",
        variable=legend_mode_var,
        values=("auto", "on", "off"),
    )
    legend_title_entry = _add_entry_row(
        tab_general, row=5, label="Legend title", variable=legend_title_var, width=40
    )
    legend_loc_combo = _add_combo_row(
        tab_general,
        row=6,
        label="Legend location",
        variable=legend_loc_var,
        values=_LEGEND_LOCATIONS,
        width=16,
    )

    _add_combo_row(
        tab_general,
        row=7,
        label="Ticks",
        variable=ticks_mode_var,
        values=("auto", "on", "off"),
    )
    _add_combo_row(
        tab_general,
        row=8,
        label="Grid",
        variable=grid_mode_var,
        values=("auto", "on", "off"),
    )
    _add_combo_row(
        tab_general,
        row=9,
        label="Markers",
        variable=markers_mode_var,
        values=("auto", "on", "off"),
    )
    _add_combo_row(
        tab_axes,
        row=0,
        label="X scale",
        variable=x_scale_var,
        values=("linear", "log", "symlog", "logit"),
    )
    _add_combo_row(
        tab_axes,
        row=1,
        label="Y scale",
        variable=y_scale_var,
        values=("linear", "log", "symlog", "logit"),
    )

    axis_limits = ttk.LabelFrame(tab_axes, text="Limits", padding=8)
    axis_limits.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    for col in range(4):
        axis_limits.columnconfigure(col, weight=1)
    ttk.Label(axis_limits, text="X min").grid(row=0, column=0, sticky="w")
    ttk.Entry(axis_limits, textvariable=x_min_var, width=14).grid(
        row=0, column=1, sticky="w", padx=(6, 12)
    )
    ttk.Label(axis_limits, text="X max").grid(row=0, column=2, sticky="w")
    ttk.Entry(axis_limits, textvariable=x_max_var, width=14).grid(
        row=0, column=3, sticky="w", padx=(6, 0)
    )
    ttk.Label(axis_limits, text="Y min").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(axis_limits, textvariable=y_min_var, width=14).grid(
        row=1, column=1, sticky="w", padx=(6, 12), pady=(6, 0)
    )
    ttk.Label(axis_limits, text="Y max").grid(row=1, column=2, sticky="w", pady=(6, 0))
    ttk.Entry(axis_limits, textvariable=y_max_var, width=14).grid(
        row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0)
    )

    axis_ticks = ttk.LabelFrame(tab_axes, text="Ticks", padding=8)
    axis_ticks.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    axis_ticks.columnconfigure(1, weight=1)
    axis_ticks.columnconfigure(3, weight=1)
    ttk.Label(axis_ticks, text="X ticks").grid(row=0, column=0, sticky="w")
    x_ticks_entry = ttk.Entry(axis_ticks, textvariable=x_ticks_var)
    x_ticks_entry.grid(row=0, column=1, sticky="ew", padx=(6, 12))
    ttk.Label(axis_ticks, text="Y ticks").grid(row=0, column=2, sticky="w")
    y_ticks_entry = ttk.Entry(axis_ticks, textvariable=y_ticks_var)
    y_ticks_entry.grid(row=0, column=3, sticky="ew", padx=(6, 0))
    ttk.Label(axis_ticks, text="X rotation").grid(row=1, column=0, sticky="w", pady=(6, 0))
    x_tick_rotation_entry = ttk.Entry(axis_ticks, textvariable=x_tick_rotation_var, width=14)
    x_tick_rotation_entry.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(6, 0))
    ttk.Label(axis_ticks, text="Y rotation").grid(row=1, column=2, sticky="w", pady=(6, 0))
    y_tick_rotation_entry = ttk.Entry(axis_ticks, textvariable=y_tick_rotation_var, width=14)
    y_tick_rotation_entry.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0))

    style_metrics = ttk.LabelFrame(tab_style, text="Figure & Typography", padding=8)
    style_metrics.grid(row=0, column=0, columnspan=2, sticky="ew")
    for col in range(4):
        style_metrics.columnconfigure(col, weight=1)
    ttk.Label(style_metrics, text="Figure width").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(
        style_metrics, from_=2.0, to=30.0, increment=0.1, textvariable=fig_width_var, width=12
    ).grid(row=0, column=1, sticky="w", padx=(6, 12))
    ttk.Label(style_metrics, text="Figure height").grid(row=0, column=2, sticky="w")
    ttk.Spinbox(
        style_metrics,
        from_=2.0,
        to=30.0,
        increment=0.1,
        textvariable=fig_height_var,
        width=12,
    ).grid(row=0, column=3, sticky="w", padx=(6, 0))
    ttk.Label(style_metrics, text="DPI").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Spinbox(
        style_metrics, from_=50, to=1200, increment=10, textvariable=dpi_var, width=12
    ).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(6, 0))
    _add_entry_row(style_metrics, row=2, label="Font family", variable=font_family_var, width=28)

    font_controls = ttk.LabelFrame(tab_style, text="Font Sizes", padding=8)
    font_controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    for col in range(6):
        font_controls.columnconfigure(col, weight=1)
    ttk.Label(font_controls, text="Title").grid(row=0, column=0, sticky="w")
    title_font_entry = ttk.Spinbox(
        font_controls, from_=6, to=60, increment=1, textvariable=title_font_var, width=10
    )
    title_font_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))
    ttk.Label(font_controls, text="Labels").grid(row=0, column=2, sticky="w")
    ttk.Spinbox(
        font_controls, from_=6, to=60, increment=1, textvariable=label_font_var, width=10
    ).grid(row=0, column=3, sticky="w", padx=(6, 12))
    ttk.Label(font_controls, text="Ticks").grid(row=0, column=4, sticky="w")
    tick_font_entry = ttk.Spinbox(
        font_controls, from_=6, to=60, increment=1, textvariable=tick_font_var, width=10
    )
    tick_font_entry.grid(row=0, column=5, sticky="w", padx=(6, 0))

    line_controls = ttk.LabelFrame(tab_style, text="Lines & Colors", padding=8)
    line_controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    line_controls.columnconfigure(1, weight=1)
    line_controls.columnconfigure(3, weight=1)
    ttk.Label(line_controls, text="Line width").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(
        line_controls,
        from_=0.1,
        to=20.0,
        increment=0.1,
        textvariable=line_width_var,
        width=12,
    ).grid(row=0, column=1, sticky="w", padx=(6, 12))
    ttk.Label(line_controls, text="Default line color").grid(row=0, column=2, sticky="w")
    ttk.Entry(line_controls, textvariable=line_color_var).grid(
        row=0, column=3, sticky="ew", padx=(6, 0)
    )
    ttk.Label(line_controls, text="Per-series colors").grid(row=1, column=0, sticky="w", pady=(6, 0))
    line_colors_entry = ttk.Entry(line_controls, textvariable=line_colors_var)
    line_colors_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(6, 0))

    grid_controls = ttk.LabelFrame(tab_style, text="Grid", padding=8)
    grid_controls.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    for col in range(6):
        grid_controls.columnconfigure(col, weight=1)
    ttk.Label(grid_controls, text="Line style").grid(row=0, column=0, sticky="w")
    grid_linestyle_entry = ttk.Combobox(
        grid_controls,
        textvariable=grid_linestyle_var,
        values=("-", "--", "-.", ":", ""),
        width=10,
    )
    grid_linestyle_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))
    ttk.Label(grid_controls, text="Line width").grid(row=0, column=2, sticky="w")
    grid_linewidth_entry = ttk.Spinbox(
        grid_controls,
        from_=0.1,
        to=10.0,
        increment=0.1,
        textvariable=grid_linewidth_var,
        width=10,
    )
    grid_linewidth_entry.grid(row=0, column=3, sticky="w", padx=(6, 12))
    ttk.Label(grid_controls, text="Alpha").grid(row=0, column=4, sticky="w")
    grid_alpha_entry = ttk.Spinbox(
        grid_controls,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=grid_alpha_var,
        width=10,
    )
    grid_alpha_entry.grid(row=0, column=5, sticky="w", padx=(6, 0))
    _add_entry_row(
        tab_series,
        row=0,
        label="Series labels",
        variable=series_labels_var,
        width=64,
    )
    ttk.Label(
        tab_series,
        text="Use comma-separated values, e.g. A, B, C.",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

    title_widgets = [title_entry, title_font_entry]
    legend_widgets = [legend_title_entry, legend_loc_combo]
    grid_widgets = [grid_linestyle_entry, grid_linewidth_entry, grid_alpha_entry]
    ticks_widgets = [
        x_ticks_entry,
        y_ticks_entry,
        x_tick_rotation_entry,
        y_tick_rotation_entry,
        tick_font_entry,
    ]

    def _refresh_widget_states(*_unused: object) -> None:
        title_enabled = title_mode_var.get().strip().lower() != "off"
        legend_enabled = legend_mode_var.get().strip().lower() != "off"
        grid_enabled = grid_mode_var.get().strip().lower() != "off"
        ticks_enabled = ticks_mode_var.get().strip().lower() != "off"

        for widget in title_widgets:
            _set_widget_enabled(widget, enabled=title_enabled)
        for widget in legend_widgets:
            _set_widget_enabled(widget, enabled=legend_enabled)
        for widget in grid_widgets:
            _set_widget_enabled(widget, enabled=grid_enabled)
        for widget in ticks_widgets:
            _set_widget_enabled(widget, enabled=ticks_enabled)

    for variable in (title_mode_var, legend_mode_var, grid_mode_var, ticks_mode_var):
        variable.trace_add("write", _refresh_widget_states)
    _refresh_widget_states()

    def _collect_settings() -> dict[str, Any]:
        fig_width = _optional_float(fig_width_var.get(), field_name="figure width")
        fig_height = _optional_float(fig_height_var.get(), field_name="figure height")
        if (fig_width is None) != (fig_height is None):
            raise ValueError("Figure width and figure height must both be set or both be blank.")
        figsize: list[float] | None = None
        if fig_width is not None and fig_height is not None:
            figsize = [fig_width, fig_height]

        settings = {
            "title": _optional_text(title_var.get()),
            "x_label": _optional_text(x_label_var.get()),
            "y_label": _optional_text(y_label_var.get()),
            "x_min": _optional_float(x_min_var.get(), field_name="x-min"),
            "x_max": _optional_float(x_max_var.get(), field_name="x-max"),
            "y_min": _optional_float(y_min_var.get(), field_name="y-min"),
            "y_max": _optional_float(y_max_var.get(), field_name="y-max"),
            "x_scale": x_scale_var.get().strip() or "linear",
            "y_scale": y_scale_var.get().strip() or "linear",
            "x_ticks": _optional_float_list(x_ticks_var.get(), field_name="x-ticks"),
            "y_ticks": _optional_float_list(y_ticks_var.get(), field_name="y-ticks"),
            "x_tick_rotation": _optional_float(
                x_tick_rotation_var.get(), field_name="x-tick-rotation"
            ),
            "y_tick_rotation": _optional_float(
                y_tick_rotation_var.get(), field_name="y-tick-rotation"
            ),
            "title_visible": _mode_to_toggle(title_mode_var.get()),
            "legend": _mode_to_toggle(legend_mode_var.get()),
            "grid": _mode_to_toggle(grid_mode_var.get()),
            "ticks": _mode_to_toggle(ticks_mode_var.get()),
            "markers": _mode_to_toggle(markers_mode_var.get()),
            "legend_title": _optional_text(legend_title_var.get()),
            "legend_loc": legend_loc_var.get().strip() or "best",
            "figsize": figsize,
            "dpi": _optional_int(dpi_var.get(), field_name="dpi"),
            "font_family": _optional_text(font_family_var.get()),
            "title_font_size": _optional_int(title_font_var.get(), field_name="title-font-size"),
            "label_font_size": _optional_int(label_font_var.get(), field_name="label-font-size"),
            "tick_font_size": _optional_int(tick_font_var.get(), field_name="tick-font-size"),
            "line_width": _optional_float(line_width_var.get(), field_name="line-width"),
            "line_color": _optional_text(line_color_var.get()),
            "line_colors": _optional_string_list(line_colors_var.get()),
            "series_labels": _optional_string_list(series_labels_var.get()),
            "grid_linestyle": _optional_text(grid_linestyle_var.get()),
            "grid_linewidth": _optional_float(
                grid_linewidth_var.get(), field_name="grid-linewidth"
            ),
            "grid_alpha": _optional_float(grid_alpha_var.get(), field_name="grid-alpha"),
        }
        return settings

    def _handle_preview() -> None:
        try:
            settings = _collect_settings()
            on_preview(settings)
            status_var.set("Preview opened.")
        except Exception as exc:
            status_var.set(f"Preview failed: {exc}")
            messagebox.showerror("Preview failed", str(exc), parent=root)

    def _handle_save() -> None:
        try:
            settings = _collect_settings()
            message = on_save(settings)
            status_var.set(message)
            saved_signature["value"] = _signature(settings)
        except Exception as exc:
            status_var.set(f"Save failed: {exc}")
            messagebox.showerror("Save failed", str(exc), parent=root)

    def _handle_save_figure() -> None:
        if on_save_figure is None:
            status_var.set("Save-figure action is not available.")
            return
        try:
            settings = _collect_settings()
            output_path = filedialog.asksaveasfilename(
                parent=root,
                title="Save Figure PNG",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
                initialfile="linak_plot.png",
            )
            if not output_path:
                status_var.set("Save figure canceled.")
                return
            message = on_save_figure(settings, output_path)
            status_var.set(message)
        except Exception as exc:
            status_var.set(f"Save figure failed: {exc}")
            messagebox.showerror("Save figure failed", str(exc), parent=root)

    saved_signature: dict[str, str | None] = {"value": None}
    try:
        saved_signature["value"] = _signature(_collect_settings())
    except Exception:
        saved_signature["value"] = None

    def _handle_close() -> None:
        try:
            settings = _collect_settings()
            current_signature = _signature(settings)
        except Exception as exc:
            close_anyway = messagebox.askyesno(
                "Invalid settings",
                f"Current settings contain invalid values ({exc}). Close anyway without saving?",
                parent=root,
            )
            if close_anyway:
                root.destroy()
            return

        if saved_signature["value"] == current_signature:
            root.destroy()
            return

        decision = messagebox.askyesnocancel(
            "Unsaved plot settings",
            "Save settings before closing?",
            parent=root,
        )
        if decision is None:
            return
        if decision:
            try:
                message = on_save(settings)
                status_var.set(message)
                saved_signature["value"] = current_signature
            except Exception as exc:
                status_var.set(f"Save failed: {exc}")
                messagebox.showerror("Save failed", str(exc), parent=root)
                return
        root.destroy()

    actions = ttk.Frame(main)
    actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    actions.columnconfigure(4, weight=1)
    ttk.Button(actions, text="Preview Figure", command=_handle_preview).grid(
        row=0, column=0, sticky="w"
    )
    save_figure_button = ttk.Button(
        actions,
        text="Save Figure PNG",
        command=_handle_save_figure,
    )
    save_figure_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
    if on_save_figure is None:
        _set_widget_enabled(save_figure_button, enabled=False)
    ttk.Button(actions, text="Save Settings", command=_handle_save).grid(
        row=0, column=2, sticky="w", padx=(8, 0)
    )
    ttk.Button(actions, text="Close", command=_handle_close).grid(
        row=0, column=3, sticky="w", padx=(8, 0)
    )
    ttk.Label(actions, textvariable=status_var).grid(row=0, column=4, sticky="e")

    root.protocol("WM_DELETE_WINDOW", _handle_close)
    root.mainloop()
