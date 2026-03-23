import numpy as np

import linak.plot.plotting as plotting_module


def test_format_axis_label_units_wraps_trailing_units_in_mathrm():
    assert plotting_module.format_axis_label_units("Density (g/cm^3)") == (
        "Density ($\\mathrm{g/cm^3}$)"
    )
    assert plotting_module.format_axis_label_units("Time (ps)") == "Time ($\\mathrm{ps}$)"


def test_format_axis_label_units_preserves_existing_math_and_non_unit_parentheses():
    assert plotting_module.format_axis_label_units(
        "Distance to the surface ($\\mathrm{\\AA}$)"
    ) == ("Distance to the surface ($\\mathrm{\\AA}$)")
    assert plotting_module.format_axis_label_units("g(r)") == "g(r)"


def test_normalize_series_values_area_uses_numpy_compatibility(monkeypatch):
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 1.0, 1.0], dtype=float)

    monkeypatch.delattr(plotting_module.np, "trapezoid", raising=False)
    trapz_calls = 0

    def fallback_trapz(y_values, x_values):
        nonlocal trapz_calls
        trapz_calls += 1
        return 2.0

    monkeypatch.setattr(plotting_module.np, "trapz", fallback_trapz, raising=False)

    normalized, changed = plotting_module._normalize_series_values(
        x,
        y,
        mode="area",
        target_value=1.0,
        reference_x=None,
        label="series",
    )

    assert changed is True
    assert trapz_calls == 1
    np.testing.assert_allclose(normalized, np.array([0.5, 0.5, 0.5]))


def test_normalize_series_values_none_ignores_stale_target_and_reference():
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    normalized, changed = plotting_module._normalize_series_values(
        x,
        y,
        mode="none",
        target_value=5.0,
        reference_x=1.25,
        label="series",
    )

    assert changed is False
    np.testing.assert_allclose(normalized, y)


def test_sanitize_line_collection_kwargs_removes_marker_only_fields():
    sanitized = plotting_module._sanitize_line_collection_kwargs(
        {
            "label": "Series A",
            "color": "#ff0000",
            "marker": "o",
            "markersize": 8.0,
            "markeredgecolor": "#000000",
            "markevery": 2,
            "lw": 1.5,
            "alpha": 0.7,
        }
    )

    assert sanitized == {
        "linewidths": 1.5,
        "alpha": 0.7,
    }
