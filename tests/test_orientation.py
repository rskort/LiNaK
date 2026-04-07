"""Tests for ``linak.analysis.orientation``."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms

import linak.analysis.density as density_mod
import linak.analysis.orientation as orientation_mod
from linak.analysis.orientation import (
    OrientationProfile,
    compute_orientation_profile,
    load_orientation_profile,
    load_orientation_profiles,
    save_orientation_profile,
)


def _water_frame(
    positions: list[list[float]] | None = None,
    cell: list[float] | None = None,
) -> Atoms:
    """Return a single-water-molecule frame."""
    if positions is None:
        positions = [
            [5.0, 5.0, 5.0],
            [5.8, 5.0, 5.4],
            [4.2, 5.0, 5.4],
        ]
    if cell is None:
        cell = [10.0, 10.0, 10.0]
    return Atoms("OHH", positions=positions, cell=cell, pbc=True)


def _multi_frame_trajectory(n_frames: int = 5) -> list[Atoms]:
    return [_water_frame() for _ in range(n_frames)]


def _occupied_bins(profile: OrientationProfile) -> np.ndarray:
    return np.flatnonzero(profile.count_total > 0)


def _single_occupied_bin(profile: OrientationProfile) -> int:
    occupied = _occupied_bins(profile)
    assert occupied.size == 1
    return int(occupied[0])


def _rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.asarray(
        [
            [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q3 * q4), 2.0 * (q1 * q3 + q2 * q4)],
            [2.0 * (q1 * q2 + q3 * q4), 1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q1 * q4)],
            [2.0 * (q1 * q3 - q2 * q4), 2.0 * (q2 * q3 + q1 * q4), 1.0 - 2.0 * (q1 * q1 + q2 * q2)],
        ],
        dtype=float,
    )


def _random_water_frames(n_frames: int, *, seed: int = 0) -> list[Atoms]:
    rng = np.random.default_rng(seed)
    base_vectors = np.asarray(
        [
            [0.8, 0.0, 0.4],
            [-0.8, 0.0, 0.4],
        ],
        dtype=float,
    )
    origin = np.asarray([15.0, 15.0, 15.0], dtype=float)
    frames: list[Atoms] = []
    for _ in range(n_frames):
        rotation = _rotation_matrix(rng)
        rotated = base_vectors @ rotation.T
        positions = np.vstack([origin, origin + rotated[0], origin + rotated[1]])
        frames.append(Atoms("OHH", positions=positions, cell=[30.0, 30.0, 30.0], pbc=True))
    return frames


def _mock_surface_estimate(per_frame: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        position=float(np.median(per_frame)),
        std=float(np.std(per_frame, ddof=0)),
        per_frame=np.asarray(per_frame, dtype=float),
    )


def test_compute_orientation_basic():
    frames = _multi_frame_trajectory(3)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)

    assert isinstance(profile, OrientationProfile)
    assert profile.axis == "z"
    assert profile.reference_axis == "z"
    assert profile.n_frames == 3
    assert profile.n_molecules_per_frame == 1
    assert len(profile.bin_centers) == len(profile.cos_polar_mean)
    assert len(profile.count_total) == len(profile.bin_centers)
    assert len(profile.count_polar_valid) == len(profile.bin_centers)
    assert len(profile.count_azimuthal_valid) == len(profile.bin_centers)
    assert profile.heatmap_polar.shape[0] == len(profile.bin_centers)
    assert profile.heatmap_azimuthal.shape[0] == len(profile.bin_centers)


def test_compute_orientation_empty_raises():
    with pytest.raises(ValueError, match="At least one"):
        compute_orientation_profile(frames=[], axis="z")


def test_compute_orientation_no_water_returns_nan_means():
    frame = Atoms("Au4", positions=[[0, 0, i] for i in range(4)], cell=[10, 10, 10], pbc=True)
    profile = compute_orientation_profile(frames=[frame, frame], axis="z", bin_width=1.0)

    assert profile.n_molecules_per_frame == 0
    np.testing.assert_array_equal(profile.count_total, 0)
    np.testing.assert_array_equal(profile.count_polar_valid, 0)
    np.testing.assert_array_equal(profile.count_azimuthal_valid, 0)
    assert np.all(np.isnan(profile.cos_polar_mean))
    assert np.all(np.isnan(profile.cos_azimuthal_mean))
    np.testing.assert_array_equal(profile.heatmap_polar, 0.0)
    np.testing.assert_array_equal(profile.heatmap_azimuthal, 0.0)


def test_compute_orientation_bisector_up():
    positions = [
        [5.0, 5.0, 3.0],
        [5.6, 5.0, 3.7],
        [4.4, 5.0, 3.7],
    ]
    frames = [_water_frame(positions) for _ in range(5)]
    profile = compute_orientation_profile(
        frames=frames, axis="z", reference_axis="z", bin_width=2.0
    )

    occupied = _occupied_bins(profile)
    assert occupied.size == 1
    assert profile.cos_polar_mean[occupied[0]] > 0.0


def test_compute_orientation_bisector_down():
    positions = [
        [5.0, 5.0, 5.0],
        [5.6, 5.0, 4.3],
        [4.4, 5.0, 4.3],
    ]
    frames = [_water_frame(positions) for _ in range(5)]
    profile = compute_orientation_profile(
        frames=frames, axis="z", reference_axis="z", bin_width=2.0
    )

    occupied = _occupied_bins(profile)
    assert occupied.size == 1
    assert profile.cos_polar_mean[occupied[0]] < 0.0


def test_compute_orientation_exact_polar_and_azimuthal():
    positions = [
        [5.0, 5.0, 5.0],
        [5.0, 5.7, 5.7],
        [5.0, 4.3, 5.7],
    ]
    profile = compute_orientation_profile(
        frames=[_water_frame(positions)],
        axis="z",
        reference_axis="z",
        bin_width=2.0,
    )

    occupied = _single_occupied_bin(profile)
    assert profile.count_total[occupied] == 1
    assert profile.count_polar_valid[occupied] == 1
    assert profile.count_azimuthal_valid[occupied] == 1
    assert profile.cos_polar_mean[occupied] == pytest.approx(1.0)
    assert profile.cos_azimuthal_mean[occupied] == pytest.approx(1.0)


def test_compute_orientation_projected_normal_degeneracy_skips_only_azimuthal():
    positions = [
        [5.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [5.0, 6.0, 5.0],
    ]
    profile = compute_orientation_profile(
        frames=[_water_frame(positions)],
        axis="z",
        reference_axis="z",
        bin_width=2.0,
    )

    occupied = _single_occupied_bin(profile)
    assert profile.count_total[occupied] == 1
    assert profile.count_polar_valid[occupied] == 1
    assert profile.count_azimuthal_valid[occupied] == 0
    assert np.isfinite(profile.cos_polar_mean[occupied])
    assert np.isnan(profile.cos_azimuthal_mean[occupied])
    assert np.sum(profile.heatmap_polar[occupied]) == pytest.approx(1.0)
    assert np.sum(profile.heatmap_azimuthal[occupied]) == pytest.approx(0.0)


def test_compute_orientation_bisector_degeneracy_marks_sample_invalid():
    positions = [
        [5.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [4.0, 5.0, 5.0],
    ]
    profile = compute_orientation_profile(
        frames=[_water_frame(positions)],
        axis="z",
        reference_axis="z",
        bin_width=2.0,
    )

    occupied = _single_occupied_bin(profile)
    assert profile.count_total[occupied] == 1
    assert profile.count_polar_valid[occupied] == 0
    assert profile.count_azimuthal_valid[occupied] == 0
    assert np.isnan(profile.cos_polar_mean[occupied])
    assert np.isnan(profile.cos_azimuthal_mean[occupied])


def test_compute_orientation_tiny_oh_length_is_invalid():
    positions = [
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.96],
    ]
    profile = compute_orientation_profile(
        frames=[_water_frame(positions)],
        axis="z",
        reference_axis="z",
        bin_width=2.0,
    )

    occupied = _single_occupied_bin(profile)
    assert profile.count_total[occupied] == 1
    assert profile.count_polar_valid[occupied] == 0
    assert profile.count_azimuthal_valid[occupied] == 0


def test_compute_orientation_different_axes():
    frames = _multi_frame_trajectory(2)
    for axis in ("x", "y"):
        profile = compute_orientation_profile(frames=frames, axis=axis, bin_width=2.0)
        assert profile.axis == axis
        assert len(profile.bin_centers) > 0


def test_compute_orientation_angle_bins():
    frames = _multi_frame_trajectory(2)
    for n_bins in (10, 30):
        profile = compute_orientation_profile(
            frames=frames, axis="z", bin_width=2.0, angle_bin_count=n_bins
        )
        assert profile.heatmap_polar.shape[1] == n_bins
        assert profile.heatmap_azimuthal.shape[1] == n_bins
        assert len(profile.heatmap_angle_bin_centers) == n_bins


def test_heatmap_boundary_values_land_in_first_and_last_bins():
    up_positions = [
        [5.0, 5.0, 5.0],
        [5.7, 5.0, 5.7],
        [4.3, 5.0, 5.7],
    ]
    down_positions = [
        [5.0, 5.0, 5.0],
        [5.7, 5.0, 4.3],
        [4.3, 5.0, 4.3],
    ]
    frames = [_water_frame(up_positions), _water_frame(down_positions)]
    profile = compute_orientation_profile(
        frames=frames,
        axis="z",
        reference_axis="z",
        bin_width=2.0,
        angle_bin_count=4,
    )

    occupied = _single_occupied_bin(profile)
    row = profile.heatmap_polar[occupied]
    assert row[0] == pytest.approx(1.0)
    assert row[-1] == pytest.approx(1.0)
    assert np.sum(row) == pytest.approx(2.0)


def test_variable_cell_density_uses_framewise_slab_volume():
    frames = [
        _water_frame(cell=[10.0, 10.0, 10.0]),
        _water_frame(cell=[20.0, 20.0, 10.0]),
    ]
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)

    occupied = _single_occupied_bin(profile)
    expected_density = 0.5 * ((1.0 / 100.0) + (1.0 / 400.0))
    assert profile.density[occupied] == pytest.approx(expected_density)


def test_surface_shifted_distance_mode_uses_framewise_offsets_and_cell_bounds(monkeypatch):
    positions = [
        [5.0, 5.0, 6.0],
        [5.6, 5.0, 6.7],
        [4.4, 5.0, 6.7],
    ]
    frames = [_water_frame(positions), _water_frame(positions)]
    fake_estimate = _mock_surface_estimate(np.array([2.0, 8.0], dtype=float))

    monkeypatch.setattr(
        density_mod, "_select_surface_estimate", lambda *args, **kwargs: (fake_estimate, "mock")
    )
    monkeypatch.setattr(
        orientation_mod, "_surface_estimate_supports_distance_mode", lambda *args, **kwargs: True
    )

    profile = compute_orientation_profile(
        frames=frames, axis="z", reference_axis="z", bin_width=2.0, binning="cell"
    )

    assert profile.coordinate_mode == "distance"
    assert profile.bin_edges[0] <= -8.0
    assert profile.bin_edges[-1] >= 8.0
    assert np.sum(profile.count_total) == 2
    occupied_centers = profile.bin_centers[_occupied_bins(profile)]
    assert np.any(occupied_centers < 0.0)
    assert np.any(occupied_centers > 0.0)


def test_isotropic_random_ensemble_has_small_mean_bias():
    frames = _random_water_frames(200, seed=123)
    profile = compute_orientation_profile(
        frames=frames, axis="z", reference_axis="z", bin_width=2.0
    )

    occupied = _single_occupied_bin(profile)
    assert abs(profile.cos_polar_mean[occupied]) < 0.15
    assert abs(profile.cos_azimuthal_mean[occupied]) < 0.15


def test_save_load_round_trip(tmp_path):
    frames = _multi_frame_trajectory(3)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)

    out = tmp_path / "orientation.h5"
    saved_path = save_orientation_profile(profile, out)
    assert saved_path == out.resolve()

    with h5py.File(out, "r") as handle:
        assert handle.attrs["analysis"] == "orientation"

    loaded = load_orientation_profile(out)
    assert isinstance(loaded, OrientationProfile)
    assert loaded.axis == profile.axis
    assert loaded.reference_axis == profile.reference_axis
    assert loaded.n_frames == profile.n_frames
    assert loaded.n_molecules_per_frame == profile.n_molecules_per_frame
    np.testing.assert_allclose(loaded.bin_centers, profile.bin_centers)
    np.testing.assert_allclose(loaded.cos_polar_mean, profile.cos_polar_mean, equal_nan=True)
    np.testing.assert_allclose(
        loaded.cos_azimuthal_mean, profile.cos_azimuthal_mean, equal_nan=True
    )
    np.testing.assert_array_equal(loaded.count_total, profile.count_total)
    assert loaded.series_statistics is not None
    assert "cos_polar_mean" in loaded.series_statistics
    assert "density" in loaded.series_statistics
    np.testing.assert_array_equal(
        loaded.series_statistics["density"].point_count,
        loaded.count_total,
    )
    np.testing.assert_array_equal(loaded.count_polar_valid, profile.count_polar_valid)
    np.testing.assert_array_equal(loaded.count_azimuthal_valid, profile.count_azimuthal_valid)
    np.testing.assert_allclose(loaded.cos_polar_density, profile.cos_polar_density, equal_nan=True)
    np.testing.assert_allclose(
        loaded.cos_azimuthal_density, profile.cos_azimuthal_density, equal_nan=True
    )
    np.testing.assert_allclose(loaded.density, profile.density)
    np.testing.assert_allclose(loaded.heatmap_polar, profile.heatmap_polar)
    np.testing.assert_allclose(loaded.heatmap_azimuthal, profile.heatmap_azimuthal)


def test_load_orientation_profiles_list(tmp_path):
    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)
    out = tmp_path / "orientation.h5"
    save_orientation_profile(profile, out)

    profiles = load_orientation_profiles(out)
    assert isinstance(profiles, list)
    assert len(profiles) == 1
    assert isinstance(profiles[0], OrientationProfile)


def test_load_orientation_profile_rejects_missing_count_datasets(tmp_path):
    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=1.0
    )
    out = tmp_path / "orientation_missing_counts.h5"
    save_orientation_profile(profile, out)

    with h5py.File(out, "a") as handle:
        dataset_paths: list[str] = []
        handle.visititems(
            lambda name, obj: dataset_paths.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        for suffix in ("count_total", "count_polar_valid", "count_azimuthal_valid"):
            target = next(path for path in dataset_paths if path.endswith(suffix))
            del handle[target]

    with pytest.raises(ValueError, match="Missing dataset 'count_total'"):
        load_orientation_profile(out)


def test_save_with_additional_metadata(tmp_path):
    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)
    out = tmp_path / "orientation_meta.h5"
    save_orientation_profile(profile, out, additional_metadata={"source_path": "/test/traj.xyz"})
    with h5py.File(out, "r") as handle:
        assert handle.attrs["analysis"] == "orientation"


def test_plot_orientation_profile_average(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    out = tmp_path / "orient_avg.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="average",
        angle="polar",
    )
    assert result is not None
    assert out.exists()


def test_plot_orientation_profile_density_weighted(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    out = tmp_path / "orient_dens.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="density-weighted",
        angle="polar",
    )
    assert result is not None


def test_plot_orientation_profile_density_weighted_uses_unicode_auto_label(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}
    result = plot_orientation_profile(
        profile,
        output=str(tmp_path / "orient_dens_label.png"),
        show=False,
        component="density-weighted",
        angle="polar",
        capture_state=capture_state,
    )
    assert result is not None
    assert capture_state["y_label"] == "H2O density-weighted ⟨cos(θ)⟩"


def test_plot_orientation_profile_preserves_explicit_labels(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}

    result = plot_orientation_profile(
        profile,
        output=str(tmp_path / "orient_explicit_labels.png"),
        show=False,
        component="average",
        angle="polar",
        title="H2O orientation (polar)",
        x_label="Distance to surface along Z (A)",
        y_label="<cos(theta)>",
        line_label="<cos(theta)>",
        capture_state=capture_state,
    )

    assert result is not None
    assert capture_state["title"] == "H2O orientation (polar)"
    assert capture_state["x_label"] == "Distance to surface along Z (A)"
    assert capture_state["y_label"] == "<cos(theta)>"
    assert capture_state["series_labels"] == ["<cos(theta)>"]


def test_plot_orientation_profile_heatmap(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    out = tmp_path / "orient_heat.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="heatmap",
        angle="polar",
    )
    assert result is not None
    assert out.exists()


def test_plot_orientation_profile_heatmap_normalize_uses_global_probability(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}
    out = tmp_path / "orient_heat_prob.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="heatmap",
        angle="polar",
        heatmap_normalize=True,
        capture_state=capture_state,
    )
    assert result is not None
    assert out.exists()
    ax = capture_state["axes"]
    mesh = ax.collections[0]
    values = np.asarray(mesh.get_array(), dtype=float)
    finite_values = values[np.isfinite(values)]
    assert finite_values.size > 0
    np.testing.assert_allclose(np.sum(finite_values), 1.0)


def test_plot_orientation_profile_heatmap_rejects_unknown_normalization_mode(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}
    out = tmp_path / "orient_heat_shrunk.png"
    with pytest.raises(ValueError, match="heatmap_normalization_mode must be one of"):
        plot_orientation_profile(
            profile,
            output=str(out),
            show=False,
            component="heatmap",
            angle="polar",
            heatmap_normalization_mode="shrunk_row_probability",
            capture_state=capture_state,
        )


def test_plot_orientation_profile_heatmap_bulk_water_reference_normalizes_bulk_mean_to_one(
    tmp_path,
):
    from linak.analysis.orientation import plot_orientation_profile

    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=1,
        n_molecules_per_frame=1,
        bin_edges=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
        bin_centers=np.asarray([0.5, 1.5, 2.5, 3.5, 4.5], dtype=float),
        cos_polar_mean=np.zeros(5, dtype=float),
        cos_azimuthal_mean=np.zeros(5, dtype=float),
        count_total=np.zeros(5, dtype=int),
        count_polar_valid=np.zeros(5, dtype=int),
        count_azimuthal_valid=np.zeros(5, dtype=int),
        cos_polar_density=np.zeros(5, dtype=float),
        cos_azimuthal_density=np.zeros(5, dtype=float),
        density=np.asarray([8.0, 8.0, 1.0, 8.0, 8.0], dtype=float),
        heatmap_polar=np.asarray(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ],
            dtype=float,
        ),
        heatmap_azimuthal=np.zeros((5, 2), dtype=float),
        heatmap_angle_bin_edges=np.asarray([-1.0, 0.0, 1.0], dtype=float),
        heatmap_angle_bin_centers=np.asarray([-0.5, 0.5], dtype=float),
        coordinate_mode="distance",
    )
    capture_state: dict[str, object] = {}
    out = tmp_path / "orient_heat_bulk.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="heatmap",
        angle="polar",
        heatmap_normalization_mode="bulk_water_reference",
        capture_state=capture_state,
    )
    assert result is not None
    assert out.exists()
    mesh = capture_state["axes"].collections[0]
    values = np.asarray(mesh.get_array(), dtype=float)
    np.testing.assert_allclose(np.mean(values[:, 3:5]), 1.0)
    np.testing.assert_allclose(np.mean(values[:, 0:2]), 0.5)


def test_plot_orientation_profile_heatmap_bulk_water_reference_requires_distance_mode(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=1,
        n_molecules_per_frame=1,
        bin_edges=np.asarray([0.0, 1.0, 2.0], dtype=float),
        bin_centers=np.asarray([0.5, 1.5], dtype=float),
        cos_polar_mean=np.zeros(2, dtype=float),
        cos_azimuthal_mean=np.zeros(2, dtype=float),
        count_total=np.zeros(2, dtype=int),
        count_polar_valid=np.zeros(2, dtype=int),
        count_azimuthal_valid=np.zeros(2, dtype=int),
        cos_polar_density=np.zeros(2, dtype=float),
        cos_azimuthal_density=np.zeros(2, dtype=float),
        density=np.asarray([1.0, 1.0], dtype=float),
        heatmap_polar=np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=float),
        heatmap_azimuthal=np.zeros((2, 2), dtype=float),
        heatmap_angle_bin_edges=np.asarray([-1.0, 0.0, 1.0], dtype=float),
        heatmap_angle_bin_centers=np.asarray([-0.5, 0.5], dtype=float),
        coordinate_mode="axis",
    )

    with pytest.raises(ValueError, match="distance-aligned profile"):
        plot_orientation_profile(
            profile,
            output=str(tmp_path / "orient_heat_bulk_axis.png"),
            show=False,
            component="heatmap",
            angle="polar",
            heatmap_normalization_mode="bulk_water_reference",
        )


def test_plot_orientation_profile_heatmap_bulk_water_reference_requires_bulk_plateau(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = OrientationProfile(
        axis="z",
        reference_axis="z",
        n_frames=1,
        n_molecules_per_frame=1,
        bin_edges=np.asarray([0.0, 1.0, 2.0], dtype=float),
        bin_centers=np.asarray([-0.5, -1.5], dtype=float),
        cos_polar_mean=np.zeros(2, dtype=float),
        cos_azimuthal_mean=np.zeros(2, dtype=float),
        count_total=np.zeros(2, dtype=int),
        count_polar_valid=np.zeros(2, dtype=int),
        count_azimuthal_valid=np.zeros(2, dtype=int),
        cos_polar_density=np.zeros(2, dtype=float),
        cos_azimuthal_density=np.zeros(2, dtype=float),
        density=np.asarray([0.0, 0.0], dtype=float),
        heatmap_polar=np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=float),
        heatmap_azimuthal=np.zeros((2, 2), dtype=float),
        heatmap_angle_bin_edges=np.asarray([-1.0, 0.0, 1.0], dtype=float),
        heatmap_angle_bin_centers=np.asarray([-0.5, 0.5], dtype=float),
        coordinate_mode="distance",
    )

    with pytest.raises(ValueError, match="water-bulk density plateau"):
        plot_orientation_profile(
            profile,
            output=str(tmp_path / "orient_heat_bulk_missing.png"),
            show=False,
            component="heatmap",
            angle="polar",
            heatmap_normalization_mode="bulk_water_reference",
        )


def test_plot_orientation_profile_heatmap_log_scale_masks_zero_cells(tmp_path):
    from matplotlib.colors import LogNorm

    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}
    out = tmp_path / "orient_heat_log.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="heatmap",
        angle="polar",
        heatmap_log_scale=True,
        capture_state=capture_state,
    )
    assert result is not None
    assert out.exists()
    ax = capture_state["axes"]
    assert isinstance(ax.collections[0].norm, LogNorm)


def test_plot_orientation_profile_heatmap_log_scale_rejects_nonpositive_vmin(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )

    with pytest.raises(ValueError, match="positive heatmap_vmin"):
        plot_orientation_profile(
            profile,
            output=str(tmp_path / "orient_heat_log_invalid.png"),
            show=False,
            component="heatmap",
            angle="polar",
            heatmap_log_scale=True,
            heatmap_vmin=0.0,
        )


def test_plot_orientation_profile_heatmap_honors_shared_plot_settings(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile
    from linak.plot.plotting import with_style_overrides

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}
    style = with_style_overrides(font_family="serif")

    result = plot_orientation_profile(
        profile,
        output=str(tmp_path / "orient_heat_style.png"),
        show=False,
        component="heatmap",
        angle="polar",
        style=style,
        x_label_pad=13.0,
        y_label_pad=17.0,
        axes_kwargs={"xmargin": 0.125},
        tight_layout_kwargs={"pad": 0.3},
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    cbar = capture_state["heatmap_colorbar"]
    assert ax.xaxis.label.get_fontfamily()[0].lower() == "serif"
    assert ax.yaxis.label.get_fontfamily()[0].lower() == "serif"
    assert cbar.ax.yaxis.label.get_fontfamily()[0].lower() == "serif"
    assert ax.xaxis.labelpad == pytest.approx(13.0)
    assert ax.yaxis.labelpad == pytest.approx(17.0)
    assert ax.get_xmargin() == pytest.approx(0.125)
    assert capture_state["font_family"] == "serif"
    assert capture_state["x_label_pad"] == pytest.approx(13.0)
    assert capture_state["y_label_pad"] == pytest.approx(17.0)
    assert capture_state["heatmap_normalization_mode"] == "counts"
    assert capture_state["heatmap_colorbar_enabled"] is True


def test_plot_orientation_profile_heatmap_matplotlib_rc_is_scoped(tmp_path):
    import matplotlib
    import matplotlib.colors as mcolors

    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    capture_state: dict[str, object] = {}
    previous_axes_facecolor = matplotlib.rcParams["axes.facecolor"]

    result = plot_orientation_profile(
        profile,
        output=str(tmp_path / "orient_heat_rc.png"),
        show=False,
        component="heatmap",
        angle="polar",
        matplotlib_rc={"axes.facecolor": "#abcdef"},
        capture_state=capture_state,
    )

    assert result is not None
    ax = capture_state["axes"]
    assert mcolors.to_hex(ax.get_facecolor(), keep_alpha=False) == "#abcdef"
    assert matplotlib.rcParams["axes.facecolor"] == previous_axes_facecolor


def test_plot_orientation_profile_heatmap_uses_shared_backend_configuration(
    tmp_path,
    monkeypatch,
):
    import linak.plot.plotting as plotting_mod

    from linak.analysis.orientation import plot_orientation_profile

    profile = compute_orientation_profile(
        frames=_multi_frame_trajectory(2), axis="z", bin_width=2.0
    )
    backend_calls: list[tuple[bool, str | None]] = []

    def _fake_configure_backend(*, interactive, preferred_backend=None):
        backend_calls.append((bool(interactive), preferred_backend))
        return "Agg"

    monkeypatch.setattr(plotting_mod, "configure_matplotlib_backend", _fake_configure_backend)

    result = plot_orientation_profile(
        profile,
        output=str(tmp_path / "orient_heat_backend.png"),
        show=False,
        component="heatmap",
        angle="polar",
        preferred_backend="Agg",
    )

    assert result is not None
    assert backend_calls == [(False, "Agg")]
