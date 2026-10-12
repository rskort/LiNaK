from __future__ import annotations

import csv

import numpy as np
import pytest

from linak.cli import _write_plotted_data_export
from linak.plot.plotting import plot_heatmap_series, plot_multi_line_series


@pytest.mark.parametrize(
    ("suffix", "delimiter"), [("csv", ","), ("tsv", "\t"), ("txt", "\t"), ("dat", " ")]
)
@pytest.mark.parametrize("log_scale", [False, True])
def test_heatmap_render_to_export_preserves_grid_and_values(tmp_path, suffix, delimiter, log_scale):
    state = {"plotted_xy_series": [{"x": [999], "y": [999]}]}
    values = np.array([[0.0, 2.5, np.nan], [-3.0, 5.0, 8.0]])
    plot_heatmap_series(
        np.array([0.0, 2.0, 6.0]),
        np.array([10.0, 12.0, 16.0, 22.0]),
        values,
        title="Density",
        x_label="X (A)",
        y_label="Distance (A)",
        heatmap_colorbar_label="Density",
        heatmap_log_scale=log_scale,
        heatmap_vmin=1 if log_scale else None,
        heatmap_vmax=4,
        show=False,
        capture_state=state,
    )
    assert "plotted_xy_series" not in state
    output = _write_plotted_data_export(state, tmp_path / f"grid.{suffix}")
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    assert len(rows) == 6
    assert list(rows[0]) == ["x", "y", "value", "x_min", "x_max", "y_min", "y_max"]
    assert [float(row["x"]) for row in rows] == [1, 1, 1, 4, 4, 4]
    assert [float(row["y"]) for row in rows] == [11, 14, 19, 11, 14, 19]
    np.testing.assert_equal(np.array([float(row["value"]) for row in rows]).reshape(2, 3), values)
    assert rows[-1] == {
        "x": "4",
        "y": "19",
        "value": "8",
        "x_min": "2",
        "x_max": "6",
        "y_min": "16",
        "y_max": "22",
    }


def test_heatmap_export_metadata_and_delimiter_override(tmp_path):
    state = {
        "plotted_heatmap": {
            "x_edges": [0, 2],
            "y_edges": [0, 4],
            "values": [[7]],
            "x_label": "X (A)",
            "y_label": "Y (A)",
            "value_label": "Number density",
        }
    }
    output = _write_plotted_data_export(
        state, tmp_path / "grid.csv", delimiter="tab", include_metadata=True
    )
    text = output.read_text(encoding="utf-8")
    assert '# x_label="X (A)"' in text
    assert '# value_label="Number density"' in text
    assert "# grid_shape=1,1" in text
    assert text.splitlines()[-1] == "1\t2\t7\t0\t2\t0\t4"


def test_heatmap_export_validates_before_overwriting(tmp_path):
    output = tmp_path / "grid.csv"
    output.write_text("keep me")
    state = {"plotted_heatmap": {"x_edges": [0, 1], "y_edges": [0, 1], "values": [[1, 2]]}}
    with pytest.raises(ValueError, match="matching 2D"):
        _write_plotted_data_export(state, output)
    assert output.read_text() == "keep me"


def test_switch_to_line_clears_heatmap_export(tmp_path):
    state = {"plotted_heatmap": {"values": [[99]]}}
    plot_multi_line_series(
        [np.array([1.0, 2.0])],
        [np.array([3.0, 4.0])],
        ["line"],
        title="Line",
        x_label="X",
        y_label="Y",
        show=False,
        capture_state=state,
    )
    assert "plotted_heatmap" not in state
    output = _write_plotted_data_export(state, tmp_path / "line.csv")
    assert output.read_text().splitlines() == ["x,y", "1,3", "2,4"]
