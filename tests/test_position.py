import json

import h5py
import numpy as np
import pytest
from ase import Atoms

from linak.analysis.position import (
    _build_xy_segments,
    PositionProfile,
    compute_position_profile,
    compute_position_profiles,
    load_position_profile,
    plot_position_profile,
    plot_position_profiles,
    save_position_profile,
)


def _surface_test_frames() -> list[Atoms]:
    frame0 = Atoms(
        "PtPtOO",
        positions=[
            [0.0, 0.0, 0.20],
            [1.0, 0.0, 0.20],
            [0.0, 0.0, 1.00],
            [1.0, 0.0, 1.50],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "PtPtOO",
        positions=[
            [0.0, 0.0, 0.30],
            [1.0, 0.0, 0.30],
            [0.0, 0.0, 1.20],
            [1.0, 0.0, 1.70],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return [frame0, frame1]


def test_plot_position_profiles_keeps_descriptor_render_path_with_single_loaded_profile(
    monkeypatch, tmp_path
):
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
    )

    calls: list[str] = []

    def _fake_plot_position_profile(*_args, **_kwargs):
        calls.append("single")
        return tmp_path / "single.png"

    def _fake_plot_multi_line_series(*_args, **_kwargs):
        calls.append("multi")
        return tmp_path / "multi.png"

    monkeypatch.setattr(
        "linak.analysis.position.plot_position_profile", _fake_plot_position_profile
    )
    monkeypatch.setattr(
        "linak.analysis.position.plot_multi_line_series", _fake_plot_multi_line_series
    )

    result = plot_position_profiles(
        [profile],
        output=tmp_path / "position.png",
        show=False,
        render_series_descriptors=[
            {
                "series_id": "series:o",
                "source_kind": "source",
                "source_series_id": "series:o",
                "default_label": "O",
            }
        ],
        series_overrides_by_id={"series:o": {"enabled": False}},
    )

    assert result == tmp_path / "multi.png"
    assert calls == ["multi"]


def test_compute_position_profile_tracks_atom_resolved_coordinates_and_surface_distance():
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )

    assert profile.species == "O"
    np.testing.assert_array_equal(profile.atom_indices, np.array([2, 3]))
    assert profile.x.shape == (2, 2)
    assert profile.y.shape == (2, 2)
    assert profile.z.shape == (2, 2)
    assert profile.distance_to_surface.shape == (2, 2)
    np.testing.assert_allclose(profile.time_fs, np.array([0.0, 2.0]))
    np.testing.assert_allclose(profile.time_ps, np.array([0.0, 0.002]))
    np.testing.assert_allclose(
        profile.distance_to_surface,
        np.array(
            [
                [0.80, 1.30],
                [0.90, 1.40],
            ]
        ),
        atol=1e-12,
    )
    assert profile.coordinate_mode == "distance"
    assert profile.surface_position is not None
    assert profile.surface_position_per_frame is not None


def test_compute_position_profiles_all_is_element_resolved():
    frame0 = Atoms("HO", positions=[[0.0, 0.0, 0.2], [1.0, 0.0, 0.8]])
    frame1 = Atoms("HO", positions=[[0.0, 0.0, 0.3], [1.0, 0.0, 0.9]])

    profiles = compute_position_profiles(
        [frame0, frame1],
        species="all",
        axis="z",
        timestep_fs=1.0,
        surface_mode="auto",
    )

    assert [profile.species for profile in profiles] == ["H", "O"]
    assert [profile.n_atoms for profile in profiles] == [1, 1]


def test_save_and_load_position_profile(tmp_path):
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    output = tmp_path / "position.h5"

    save_position_profile(profile, output)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["analysis"] == "position"
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["species"] == "O"
        assert metadata["axis"] == "z"
        assert metadata["coordinate_mode"] == "distance"
        assert "distance_to_surface_A" in handle

    loaded = load_position_profile(output, species="O", axis="z")
    np.testing.assert_allclose(loaded.x, profile.x)
    np.testing.assert_allclose(loaded.y, profile.y)
    np.testing.assert_allclose(loaded.z, profile.z)
    np.testing.assert_allclose(loaded.distance_to_surface, profile.distance_to_surface)
    np.testing.assert_allclose(loaded.time_fs, profile.time_fs)
    np.testing.assert_allclose(loaded.step, profile.step)


def test_load_position_profile_rejects_csv_input(tmp_path):
    csv = tmp_path / "old_position.csv"
    csv.write_text("time_ps,distance\n0.0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Use .h5/.hdf5"):
        load_position_profile(csv)


def test_plot_position_profile_defaults_to_distance_vs_time(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_plot_multi_line_series(x_series, y_series, _labels, **kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr(
        "linak.analysis.position.plot_multi_line_series", _fake_plot_multi_line_series
    )

    profile = PositionProfile(
        species="O",
        axis="z",
        atom_indices=np.array([0, 1]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 1.0]),
        time_ps=np.array([0.0, 0.001]),
        x=np.array([[0.0, 1.0], [0.1, 1.1]]),
        y=np.array([[0.0, 0.0], [0.0, 0.0]]),
        z=np.array([[1.0, 2.0], [1.2, 2.2]]),
        distance_to_surface=np.array([[0.8, 1.8], [0.9, 1.9]]),
        n_frames=2,
        n_atoms=2,
        coordinate_mode="distance",
        surface_position=0.2,
        surface_position_std=0.0,
        surface_position_per_frame=np.array([0.2, 0.3]),
    )

    plot_position_profile(profile, show=False)
    assert captured["x_label"] == "Time (ps)"
    assert captured["y_label"] == "Distance to the surface ($\\mathrm{\\AA}$)"
    np.testing.assert_allclose(captured["y_series"][0], np.array([0.8, 0.9]))


def test_plot_position_profiles_single_profile_preserves_series_labels(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_plot_multi_line_series(_x_series, _y_series, labels, **kwargs):
        captured["labels"] = labels
        captured["legend"] = kwargs["legend"]
        return None

    monkeypatch.setattr(
        "linak.analysis.position.plot_multi_line_series", _fake_plot_multi_line_series
    )

    profile = PositionProfile(
        species="O",
        axis="z",
        atom_indices=np.array([0, 1]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 1.0]),
        time_ps=np.array([0.0, 0.001]),
        x=np.array([[0.0, 1.0], [0.1, 1.1]]),
        y=np.array([[0.0, 0.0], [0.0, 0.0]]),
        z=np.array([[1.0, 2.0], [1.2, 2.2]]),
        distance_to_surface=np.array([[0.8, 1.8], [0.9, 1.9]]),
        n_frames=2,
        n_atoms=2,
        coordinate_mode="distance",
        surface_position=0.2,
        surface_position_std=0.0,
        surface_position_per_frame=np.array([0.2, 0.3]),
    )

    plot_position_profiles(
        [profile],
        show=False,
        series_labels=["O-first", "O-second"],
    )

    assert captured["labels"] == ["O-first", "O-second"]
    assert captured["legend"] is True


def test_plot_position_profile_xy_z_projection_writes_output(tmp_path):
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    output = tmp_path / "position_xy_z.png"

    saved = plot_position_profile(
        profile,
        component="xy-z",
        show=False,
        output=output,
    )

    assert saved == output.resolve()
    assert output.exists()


def test_plot_position_profile_xy_z_defaults_limits_to_cell_dimensions():
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    plot_position_profile(
        profile,
        component="xy-z",
        show=False,
        capture_state=captured,
    )

    assert captured["x_lim"] == pytest.approx([0.0, 10.0])
    assert captured["y_lim"] == pytest.approx([0.0, 10.0])


def test_plot_position_profile_xy_z_applies_axis_label_padding(monkeypatch):
    import matplotlib.axes

    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def _capture_set_xlabel(self, xlabel, *args, **kwargs):
        captured["x_label"] = xlabel
        captured["x_label_pad"] = kwargs.get("labelpad")
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def _capture_set_ylabel(self, ylabel, *args, **kwargs):
        captured["y_label"] = ylabel
        captured["y_label_pad"] = kwargs.get("labelpad")
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", _capture_set_xlabel)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", _capture_set_ylabel)

    plot_position_profile(
        profile,
        component="xy-z",
        show=False,
        x_label_pad=11.0,
        y_label_pad=13.0,
    )

    assert captured["x_label"] == "X (A)"
    assert captured["y_label"] == "Y (A)"
    assert captured["x_label_pad"] == pytest.approx(11.0)
    assert captured["y_label_pad"] == pytest.approx(13.0)


def test_plot_position_profile_xy_z_preserves_explicit_blank_axis_labels(monkeypatch):
    import matplotlib.axes

    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def _capture_set_xlabel(self, xlabel, *args, **kwargs):
        captured["x_label"] = xlabel
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def _capture_set_ylabel(self, ylabel, *args, **kwargs):
        captured["y_label"] = ylabel
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", _capture_set_xlabel)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", _capture_set_ylabel)

    plot_position_profile(
        profile,
        component="xy-z",
        show=False,
        x_label="",
        y_label="",
    )

    assert captured["x_label"] == ""
    assert captured["y_label"] == ""


def test_build_xy_segments_breaks_periodic_jump_connectors():
    x_values = np.array([0.95, 0.05, 0.15], dtype=float)
    y_values = np.array([0.20, 0.20, 0.20], dtype=float)
    color_values = np.array([1.0, 1.1, 1.2], dtype=float)

    segments, segment_colors = _build_xy_segments(
        x_values,
        y_values,
        color_values,
        cell_lengths_xy=(1.0, 1.0),
    )

    assert segments.shape == (1, 2, 2)
    np.testing.assert_allclose(segments[0], np.array([[0.05, 0.20], [0.15, 0.20]]))
    np.testing.assert_allclose(segment_colors, np.array([1.15]))


def test_plot_position_profile_xy_z_projection_rejects_invalid_map_color():
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )

    with pytest.raises(ValueError, match="map_color"):
        plot_position_profile(
            profile,
            component="xy-z",
            map_color="not-a-mode",
            show=False,
        )


def test_plot_position_profile_xy_z_respects_private_per_axis_tick_visibility():
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    result = plot_position_profile(
        profile,
        component="xy-z",
        show=False,
        capture_state=captured,
        tick_params_kwargs={"_x_ticks_visible": False, "_y_ticks_visible": True},
    )

    assert result is None
    assert "x_lim" in captured
    assert "y_lim" in captured


def test_plot_position_profile_xy_z_logs_fixed_line_color_ignore_at_debug_only(caplog):
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )

    with caplog.at_level("DEBUG", logger="linak.analysis.position"):
        plot_position_profile(
            profile,
            component="xy-z",
            show=False,
            line_colors=["#ff0000", "#00ff00"],
        )

    assert any("ignores per-series fixed line colors" in message for message in caplog.messages)
    assert not any(record.levelname == "WARNING" for record in caplog.records)


def test_plot_position_profile_xy_z_distance_cutoff_masks_points_without_bridging_gaps(
    monkeypatch,
):
    profile = PositionProfile(
        species="H",
        axis="z",
        atom_indices=np.array([0]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 1.0, 2.0]),
        time_fs=np.array([0.0, 1.0, 2.0]),
        time_ps=np.array([0.0, 0.001, 0.002]),
        x=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([[0.0], [1.0], [2.0]]),
        z=np.array([[0.1], [0.2], [0.3]]),
        distance_to_surface=np.array([[1.0], [3.0], [1.5]]),
        n_frames=3,
        n_atoms=1,
        cell_lengths_angstrom=(10.0, 10.0, 10.0),
    )

    original_builder = _build_xy_segments
    call_lengths: list[int] = []

    def _capture_runs(x_values, y_values, color_values, *, cell_lengths_xy):
        call_lengths.append(int(len(x_values)))
        return original_builder(
            x_values,
            y_values,
            color_values,
            cell_lengths_xy=cell_lengths_xy,
        )

    monkeypatch.setattr("linak.analysis.position._build_xy_segments", _capture_runs)

    plot_position_profile(
        profile,
        component="xy-z",
        show=False,
        xy_z_distance_max=2.0,
    )

    assert call_lengths == [1, 1]


def test_plot_position_profile_xy_z_distance_cutoff_raises_when_all_filtered():
    profile = PositionProfile(
        species="H",
        axis="z",
        atom_indices=np.array([0]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 1.0]),
        time_ps=np.array([0.0, 0.001]),
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        z=np.array([[0.1], [0.2]]),
        distance_to_surface=np.array([[3.0], [4.0]]),
        n_frames=2,
        n_atoms=1,
        cell_lengths_angstrom=(10.0, 10.0, 10.0),
    )

    with pytest.raises(
        ValueError,
        match="No atom trajectories remain after applying the xy-z distance cutoff.",
    ):
        plot_position_profile(
            profile,
            component="xy-z",
            show=False,
            xy_z_distance_max=2.0,
        )


def test_plot_position_profile_2d_projection_line_colors_uses_per_series_colors():
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    result = plot_position_profile(
        profile,
        component="2d-projection",
        projection_x="x",
        projection_y="distance",
        projection_value="y",
        projection_render_mode="line-colors",
        line_colors=["#ff0000", "#00ff00"],
        show=False,
        capture_state=captured,
    )

    assert result is None
    ax = captured["axes"]
    assert ax.get_legend() is not None
    assert [line.get_color() for line in ax.lines] == ["#ff0000", "#00ff00"]


def test_plot_position_profile_2d_projection_filter_uses_selected_value_quantity():
    profile = PositionProfile(
        species="H",
        axis="z",
        atom_indices=np.array([0]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 1.0, 2.0]),
        time_fs=np.array([0.0, 1.0, 2.0]),
        time_ps=np.array([0.0, 0.001, 0.002]),
        x=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([[4.0], [5.0], [7.0]]),
        z=np.array([[0.1], [0.2], [0.3]]),
        distance_to_surface=np.array([[1.0], [1.5], [2.0]]),
        n_frames=3,
        n_atoms=1,
        cell_lengths_angstrom=(10.0, 10.0, 10.0),
    )
    captured: dict[str, object] = {}

    plot_position_profile(
        profile,
        component="2d-projection",
        projection_x="x",
        projection_y="distance",
        projection_value="y",
        projection_filter_min=4.5,
        projection_filter_max=5.5,
        projection_render_mode="line-colors",
        show=False,
        capture_state=captured,
    )

    ax = captured["axes"]
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 1
    assert offsets[0, 0] == pytest.approx(1.0)
    assert offsets[0, 1] == pytest.approx(1.5)


def test_plot_position_profiles_2d_projection_line_colors_honors_descriptor_overrides():
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    captured: dict[str, object] = {}

    plot_position_profiles(
        [profile],
        component="2d-projection",
        projection_x="x",
        projection_y="distance",
        projection_value="y",
        projection_render_mode="line-colors",
        show=False,
        capture_state=captured,
        render_series_descriptors=[
            {
                "series_id": "atom:0",
                "source_kind": "source",
                "source_series_id": "atom:0",
                "default_label": "O[2]",
            },
            {
                "series_id": "atom:1",
                "source_kind": "source",
                "source_series_id": "atom:1",
                "default_label": "O[3]",
            },
        ],
        series_overrides_by_id={
            "atom:0": {"enabled": True, "color": "#ff0000", "label_override": "Visible atom"},
            "atom:1": {"enabled": False},
        },
    )

    ax = captured["axes"]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == "#ff0000"
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Visible atom"]


def test_plot_position_profile_2d_projection_color_scale_ignores_mismatched_line_colors(caplog):
    profile = compute_position_profile(
        _surface_test_frames(),
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )

    with caplog.at_level("DEBUG", logger="linak.analysis.position"):
        result = plot_position_profile(
            profile,
            component="2d-projection",
            projection_x="x",
            projection_y="y",
            projection_value="distance",
            projection_render_mode="color-scale",
            line_colors=["#ff0000"],
            show=False,
        )

    assert result is None
    assert any(
        "ignores per-series fixed line colors in color-scale mode" in message
        for message in caplog.messages
    )
