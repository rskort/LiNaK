from linak.plot.plot_gui import (
    _coerce_series_descriptors,
    _coerce_series_overrides,
    _derive_synced_field_locks,
    _derive_warning_messages,
    _extract_limit,
    _extract_dict_mode,
    _format_series_display_text,
    _lock_to_sync_mode,
    _resolve_asset_path,
    _resolve_series_line_colors,
    _sync_mode_to_lock,
    _toggle_to_mode,
    _without_new_profile_series_overrides,
    _without_series_specific_settings,
)


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


def test_without_new_profile_series_overrides_resets_series_customizations_only():
    settings = {
        "title": "Demo",
        "series_labels": ["Au", "H2O"],
        "series_descriptors": [{"series_id": "a"}, {"series_id": "b"}],
        "series_overrides": {"a": {"enabled": False}},
        "series_enabled": [False, True],
        "line_colors": ["#ff0000", "#00ff00"],
        "series_markers": ["o", ""],
    }

    cleaned = _without_new_profile_series_overrides(settings)

    assert cleaned["title"] == "Demo"
    assert cleaned["series_labels"] == ["Au", "H2O"]
    assert cleaned["series_descriptors"] == [{"series_id": "a"}, {"series_id": "b"}]
    assert "series_overrides" not in cleaned
    assert "series_enabled" not in cleaned
    assert "line_colors" not in cleaned
    assert "series_markers" not in cleaned


def test_resolve_series_line_colors_keeps_none_when_all_blank():
    assert _resolve_series_line_colors(["", "   "]) is None


def test_resolve_series_line_colors_preserves_blank_entries():
    resolved = _resolve_series_line_colors(["", "#ff5500", ""])

    assert resolved == ["", "#ff5500", ""]


def test_format_series_display_text_marks_disabled_entries():
    assert _format_series_display_text(0, "Au", enabled=True) == "1: Au"
    assert _format_series_display_text(1, "H2O", enabled=False) == "2: H2O (off)"


def test_coerce_series_descriptors_preserves_identity_and_metadata():
    descriptors = _coerce_series_descriptors(
        [
            {
                "series_id": "series:0:3",
                "default_label": "O",
                "source_name": "density.h5",
                "source_directory": "runs/run_04",
                "source_path": "/tmp/runs/run_04/density.h5",
            }
        ]
    )

    assert descriptors == [
        {
            "series_id": "series:0:3",
            "default_label": "O",
            "source_name": "density.h5",
            "source_directory": "runs/run_04",
            "source_path": "/tmp/runs/run_04/density.h5",
            "source_index": None,
            "series_index": None,
        }
    ]


def test_coerce_series_overrides_keeps_sparse_label_override_mapping():
    overrides = _coerce_series_overrides(
        {
            "series:0:3": {
                "label_override": "H2O",
                "enabled": False,
            }
        }
    )

    assert overrides["series:0:3"]["label_override"] == "H2O"
    assert overrides["series:0:3"]["enabled"] is False


def test_derive_synced_field_locks_prefers_explicit_metadata():
    locks = _derive_synced_field_locks(
        {
            "_gui_locked_fields": {
                "x_label": True,
                "y_label": False,
                "ignored": True,
            },
            "x_label": None,
            "y_label": "Density",
        }
    )

    assert locks["x_label"] is True
    assert locks["y_label"] is False


def test_derive_synced_field_locks_detects_explicit_limits_ticks_and_label_padding():
    locks = _derive_synced_field_locks(
        {
            "title": "",
            "x_label": None,
            "y_label": "",
            "x_lim": [0.0, 5.0],
            "y_lim": [None, None],
            "x_ticks": [0.0, 1.0, 2.0],
            "y_ticks": None,
            "x_label_pad": 6.0,
        }
    )

    assert locks == {
        "title": True,
        "x_label": False,
        "y_label": True,
        "x_lim": True,
        "y_lim": False,
        "x_ticks": True,
        "y_ticks": False,
        "x_label_pad": True,
        "y_label_pad": False,
    }


def test_extract_limit_rounds_auto_values_for_gui_display():
    settings = {"x_lim": [0.123456789, 12.0]}

    assert _extract_limit(settings, key="x_lim", index=0) == "0.123457"
    assert _extract_limit(settings, key="x_lim", index=1) == "12"


def test_toggle_to_mode_materializes_auto_to_effective_default():
    assert _toggle_to_mode(None, auto_mode="on") == "on"
    assert _toggle_to_mode(None, auto_mode="off") == "off"


def test_extract_dict_mode_materializes_missing_bool_to_effective_default():
    assert _extract_dict_mode({}, key="legend_kwargs", nested_key="frameon", auto_mode="on") == "on"
    assert (
        _extract_dict_mode({}, key="savefig_kwargs", nested_key="transparent", auto_mode="off")
        == "off"
    )


def test_sync_mode_mapping_round_trips_manual_and_auto():
    assert _lock_to_sync_mode(True) == "Manual"
    assert _lock_to_sync_mode(False) == "Auto"
    assert _sync_mode_to_lock("Manual") is True
    assert _sync_mode_to_lock("manual") is True
    assert _sync_mode_to_lock("Auto") is False


def test_derive_warning_messages_reports_disabled_sections_and_partial_normalization():
    warnings = _derive_warning_messages(
        {
            "legend": False,
            "legend_title": "Runs",
            "grid": False,
            "grid_alpha": 0.3,
            "ticks": False,
            "tick_params_kwargs": {"direction": "out"},
            "series_normalization_modes": ["none", "max"],
        }
    )

    assert any("Legend is off" in message for message in warnings)
    assert all("Grid is off" not in message for message in warnings)
    assert any("Ticks are off" in message for message in warnings)
    assert any("Only part of the plotted series is normalized" in message for message in warnings)


def test_derive_warning_messages_ignores_advanced_json_overlap():
    warnings = _derive_warning_messages({"legend": True})

    assert all("Advanced JSON overlaps with standard controls" not in message for message in warnings)


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
