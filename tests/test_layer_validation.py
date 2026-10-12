import numpy as np
import pytest

from linak.cli import _gui_series_descriptors_from_settings, _merge_gui_series_descriptors
from linak.plot.plot_gui import _coerce_series_descriptors
from linak.plot.plotting import plot_multi_line_series
from linak.plot.settings_validation import validate_current_plot_settings


@pytest.mark.parametrize("kind", ["group", "unknown"])
def test_unsupported_layer_types_rejected_at_each_entry_point(tmp_path, kind):
    descriptor = {"series_id": "invalid", "source_kind": kind}
    settings = {"series_descriptors": [descriptor]}
    calls = [
        lambda: _coerce_series_descriptors([descriptor]),
        lambda: _merge_gui_series_descriptors([], [descriptor]),
        lambda: _merge_gui_series_descriptors([descriptor], None),
        lambda: _gui_series_descriptors_from_settings(settings, []),
        lambda: validate_current_plot_settings(settings),
        lambda: validate_current_plot_settings({"position_view_states": {"projection": settings}}),
        lambda: plot_multi_line_series(
            [np.array([0.0, 1.0])],
            [np.array([1.0, 2.0])],
            ["A"],
            title="Invalid layer",
            x_label="x",
            y_label="y",
            output=tmp_path / "invalid.png",
            show=False,
            render_series_descriptors=[descriptor],
            series_overrides_by_id={"invalid": {"enabled": False}},
        ),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="Unsupported layer type"):
            call()
    assert not (tmp_path / "invalid.png").exists()


def test_source_and_copy_descriptors_have_no_aggregation_fields():
    descriptors = _coerce_series_descriptors(
        [
            {"series_id": "a"},
            {"series_id": "copy", "source_series_id": "a", "is_generated": True},
        ]
    )
    assert all("member_series_ids" not in d and "group_reducer" not in d for d in descriptors)
    assert descriptors[1]["source_series_id"] == "a"
    validate_current_plot_settings({"series_descriptors": descriptors})
