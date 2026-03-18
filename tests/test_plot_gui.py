from linak.plot.plot_gui import (
    _extract_dict_mode,
    _resolve_asset_path,
    _resolve_series_line_colors,
    _toggle_to_mode,
    _without_series_specific_settings,
)
from linak.plot.plotting import default_series_colors


def test_without_series_specific_settings_removes_per_series_keys():
    settings = {
        "title": "Demo",
        "grid": True,
        "line_width": 2.0,
        "series_labels": ["run-a", "run-b"],
        "line_colors": ["#ff0000", "#00ff00"],
        "series_normalization_modes": ["max", "none"],
    }

    filtered = _without_series_specific_settings(settings)

    assert filtered["title"] == "Demo"
    assert filtered["grid"] is True
    assert filtered["line_width"] == 2.0
    assert "series_labels" not in filtered
    assert "line_colors" not in filtered
    assert "series_normalization_modes" not in filtered


def test_without_series_specific_settings_keeps_non_series_limits():
    settings = {
        "x_lim": [0.0, 10.0],
        "y_lim": [0.0, 1.0],
        "font_family": "DejaVu Sans",
        "series_enabled": [True, False],
    }

    filtered = _without_series_specific_settings(settings)

    assert filtered["x_lim"] == [0.0, 10.0]
    assert filtered["y_lim"] == [0.0, 1.0]
    assert filtered["font_family"] == "DejaVu Sans"
    assert "series_enabled" not in filtered


def test_resolve_series_line_colors_keeps_none_when_all_blank():
    assert _resolve_series_line_colors(["", "   "]) is None


def test_resolve_series_line_colors_fills_blanks_with_default_palette():
    defaults = default_series_colors(3)

    resolved = _resolve_series_line_colors(["", "#ff5500", ""])

    assert resolved == [defaults[0], "#ff5500", defaults[2]]


def test_toggle_to_mode_materializes_auto_to_effective_default():
    assert _toggle_to_mode(None, auto_mode="on") == "on"
    assert _toggle_to_mode(None, auto_mode="off") == "off"


def test_extract_dict_mode_materializes_missing_bool_to_effective_default():
    assert _extract_dict_mode({}, key="legend_kwargs", nested_key="frameon", auto_mode="on") == "on"
    assert (
        _extract_dict_mode({}, key="savefig_kwargs", nested_key="transparent", auto_mode="off")
        == "off"
    )


def test_resolve_asset_path_prefers_repo_root_assets(tmp_path):
    module_path = tmp_path / "project" / "src" / "linak" / "plot" / "plot_gui.py"
    expected = tmp_path / "project" / "assets" / "linak_gui_banner.svg"
    expected.parent.mkdir(parents=True)
    expected.write_text("<svg />", encoding="utf-8")

    resolved = _resolve_asset_path("linak_gui_banner.svg", module_path=module_path)

    assert resolved == expected.resolve()


def test_resolve_asset_path_finds_installed_share_assets(tmp_path):
    module_path = tmp_path / "venv" / "Lib" / "site-packages" / "linak" / "plot" / "plot_gui.py"
    expected = tmp_path / "venv" / "share" / "linak" / "assets" / "linak_gui_banner.svg"
    expected.parent.mkdir(parents=True)
    expected.write_text("<svg />", encoding="utf-8")

    resolved = _resolve_asset_path("linak_gui_banner.svg", module_path=module_path)

    assert resolved == expected.resolve()
