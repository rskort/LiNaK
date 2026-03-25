"""Tests for ``linak.analysis.orientation``."""

import numpy as np
import pytest
from ase import Atoms
import h5py

from linak.analysis.orientation import (
    OrientationProfile,
    compute_orientation_profile,
    load_orientation_profile,
    load_orientation_profiles,
    save_orientation_profile,
)


# ──────────────────────── helper builders ──────────────────────────────────

def _water_frame(
    positions: list[list[float]] | None = None,
    cell: list[float] | None = None,
) -> Atoms:
    """Return a single-water-molecule Atoms frame."""
    if positions is None:
        # O at origin, two H's in +z hemisphere ⇒ bisector points +z
        positions = [
            [5.0, 5.0, 5.0],  # O
            [5.8, 5.0, 5.4],  # H1
            [4.2, 5.0, 5.4],  # H2
        ]
    if cell is None:
        cell = [10.0, 10.0, 10.0]
    frame = Atoms("OHH", positions=positions, cell=cell, pbc=True)
    return frame


def _multi_frame_trajectory(n_frames: int = 5) -> list[Atoms]:
    """Return a short trajectory of identical water frames."""
    return [_water_frame() for _ in range(n_frames)]


# ──────────────────────── basic compute tests ─────────────────────────────

def test_compute_orientation_basic():
    frames = _multi_frame_trajectory(3)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)

    assert isinstance(profile, OrientationProfile)
    assert profile.axis == "z"
    assert profile.reference_axis == "z"
    assert profile.n_frames == 3
    assert profile.n_molecules_per_frame == 1
    assert len(profile.bin_centers) == len(profile.cos_polar_mean)
    assert len(profile.cos_polar_mean) == len(profile.cos_azimuthal_mean)
    assert len(profile.density) == len(profile.cos_polar_mean)
    assert profile.heatmap_polar.shape[0] == len(profile.bin_centers)
    assert profile.heatmap_azimuthal.shape[0] == len(profile.bin_centers)


def test_compute_orientation_empty_raises():
    with pytest.raises(ValueError, match="At least one"):
        compute_orientation_profile(frames=[], axis="z")


def test_compute_orientation_no_water():
    """Frame with no H₂O should produce flat-zero profiles."""
    frame = Atoms("Au4", positions=[[0, 0, i] for i in range(4)], cell=[10, 10, 10], pbc=True)
    profile = compute_orientation_profile(frames=[frame, frame], axis="z", bin_width=1.0)
    assert profile.n_molecules_per_frame == 0
    np.testing.assert_array_equal(profile.cos_polar_mean, 0.0)
    np.testing.assert_array_equal(profile.cos_azimuthal_mean, 0.0)


def test_compute_orientation_bisector_up():
    """H atoms above O ⇒ bisector pointing +z ⇒ cos(polar) > 0."""
    positions = [
        [5.0, 5.0, 3.0],  # O
        [5.6, 5.0, 3.7],  # H1
        [4.4, 5.0, 3.7],  # H2
    ]
    frames = [_water_frame(positions) for _ in range(5)]
    profile = compute_orientation_profile(
        frames=frames, axis="z", reference_axis="z", bin_width=2.0,
    )
    # Distance bin containing ~3.0 Å should show positive cos(polar)
    occupied = profile.cos_polar_mean != 0
    assert np.any(occupied)
    assert np.all(profile.cos_polar_mean[occupied] > 0)


def test_compute_orientation_bisector_down():
    """H atoms below O ⇒ bisector pointing −z ⇒ cos(polar) < 0."""
    positions = [
        [5.0, 5.0, 5.0],  # O
        [5.6, 5.0, 4.3],  # H1
        [4.4, 5.0, 4.3],  # H2
    ]
    frames = [_water_frame(positions) for _ in range(5)]
    profile = compute_orientation_profile(
        frames=frames, axis="z", reference_axis="z", bin_width=2.0,
    )
    occupied = profile.cos_polar_mean != 0
    assert np.any(occupied)
    assert np.all(profile.cos_polar_mean[occupied] < 0)


def test_compute_orientation_different_axes():
    """Compute should work for x and y axes too."""
    frames = _multi_frame_trajectory(2)
    for axis in ("x", "y"):
        profile = compute_orientation_profile(frames=frames, axis=axis, bin_width=2.0)
        assert profile.axis == axis
        assert len(profile.bin_centers) > 0


def test_compute_orientation_angle_bins():
    """The angle_bin_count controls heatmap columns."""
    frames = _multi_frame_trajectory(2)
    for n_bins in (10, 30):
        profile = compute_orientation_profile(
            frames=frames, axis="z", bin_width=2.0, angle_bin_count=n_bins,
        )
        assert profile.heatmap_polar.shape[1] == n_bins
        assert profile.heatmap_azimuthal.shape[1] == n_bins
        assert len(profile.heatmap_angle_bin_centers) == n_bins


# ──────────────────────── save / load round-trip ──────────────────────────

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
    np.testing.assert_allclose(loaded.cos_polar_mean, profile.cos_polar_mean)
    np.testing.assert_allclose(loaded.cos_azimuthal_mean, profile.cos_azimuthal_mean)
    np.testing.assert_allclose(loaded.cos_polar_density, profile.cos_polar_density)
    np.testing.assert_allclose(loaded.cos_azimuthal_density, profile.cos_azimuthal_density)
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


def test_save_with_additional_metadata(tmp_path):
    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=1.0)
    out = tmp_path / "orientation_meta.h5"
    save_orientation_profile(
        profile, out, additional_metadata={"source_path": "/test/traj.xyz"}
    )
    with h5py.File(out, "r") as handle:
        assert handle.attrs["analysis"] == "orientation"


# ──────────────────────── plot smoke tests ────────────────────────────────

def test_plot_orientation_profile_average(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=2.0)
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

    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=2.0)
    out = tmp_path / "orient_dens.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="density-weighted",
        angle="polar",
    )
    assert result is not None


def test_plot_orientation_profile_heatmap(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=2.0)
    out = tmp_path / "orient_heat.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="heatmap",
        angle="polar",
    )
    assert result is not None


def test_plot_heatmap_with_vmin_vmax_cmap(tmp_path):
    from linak.analysis.orientation import plot_orientation_profile

    frames = _multi_frame_trajectory(2)
    profile = compute_orientation_profile(frames=frames, axis="z", bin_width=2.0)
    out = tmp_path / "orient_heat_custom.png"
    result = plot_orientation_profile(
        profile,
        output=str(out),
        show=False,
        component="heatmap",
        angle="polar",
        heatmap_vmin=0.0,
        heatmap_vmax=0.5,
        heatmap_cmap="plasma",
    )
    assert result is not None
    assert out.exists()


def test_plot_orientation_profiles_multi(tmp_path):
    from linak.analysis.orientation import plot_orientation_profiles

    frames = _multi_frame_trajectory(2)
    p1 = compute_orientation_profile(frames=frames, axis="z", bin_width=2.0)
    p2 = compute_orientation_profile(frames=frames, axis="z", bin_width=2.0)
    out = tmp_path / "orient_multi.png"
    result = plot_orientation_profiles(
        [p1, p2],
        output=str(out),
        show=False,
        component="average",
        angle="polar",
    )
    assert result is not None
