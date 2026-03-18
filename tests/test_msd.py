import numpy as np
import pytest
import h5py
from ase import Atoms

from linak.analysis.msd import compute_msd, load_msd_profile, save_msd_profile


def test_compute_msd_returns_expected_profile():
    frame0 = Atoms("OO", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    frame1 = Atoms("OO", positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frame2 = Atoms("OO", positions=[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    profile = compute_msd([frame0, frame1, frame2], species="O", timestep_fs=2.0)

    np.testing.assert_allclose(profile.time_fs, np.array([0.0, 2.0, 4.0]))
    np.testing.assert_allclose(profile.time_ps, np.array([0.0, 0.002, 0.004]))
    np.testing.assert_allclose(profile.msd, np.array([0.0, 1.0, 4.0]))
    assert profile.species == "O"
    assert profile.n_frames == 3


def test_compute_msd_raises_for_missing_species():
    frame = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="No atoms found for species"):
        compute_msd([frame], species="O")


def test_save_and_load_msd_profile(tmp_path):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    profile = compute_msd([frame], species="O")
    out = tmp_path / "msd.h5"

    save_msd_profile(profile, out)
    with h5py.File(out, "r") as handle:
        assert "created_utc" in handle.attrs
        assert "linak_version" in handle.attrs
    loaded = load_msd_profile(out, species="O")

    np.testing.assert_allclose(loaded.time_fs, profile.time_fs)
    np.testing.assert_allclose(loaded.time_ps, profile.time_ps)
    np.testing.assert_allclose(loaded.msd, profile.msd)
    assert loaded.species == "O"


def test_load_msd_profile_rejects_csv_input(tmp_path):
    csv = tmp_path / "legacy_msd.csv"
    csv.write_text(
        "time_fs,time_ps,msd_A2\n0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Use .h5/.hdf5"):
        load_msd_profile(csv)


def test_compute_msd_uses_periodic_minimum_image_when_cell_is_present():
    frame0 = Atoms("O", positions=[[0.9, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)
    frame1 = Atoms("O", positions=[[0.1, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)

    profile = compute_msd([frame0, frame1], species="O", timestep_fs=1.0)

    np.testing.assert_allclose(profile.msd, np.array([0.0, 0.04]), rtol=0.0, atol=1e-12)


def test_plot_msd_profiles_uses_multi_line_plot_for_multiple_profiles(monkeypatch):
    from linak.analysis.msd import MSDProfile, plot_msd_profiles

    captured = {}

    def _fake_plot_multi_line_series(x_series, y_series, labels, **_kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["labels"] = labels
        return None

    monkeypatch.setattr("linak.analysis.msd.plot_multi_line_series", _fake_plot_multi_line_series)

    profile_a = MSDProfile(
        species="A",
        time_fs=np.array([0.0, 1.0]),
        time_ps=np.array([0.0, 0.001]),
        msd=np.array([0.0, 1.0]),
        n_frames=2,
    )
    profile_b = MSDProfile(
        species="B",
        time_fs=np.array([0.0, 1.0]),
        time_ps=np.array([0.0, 0.001]),
        msd=np.array([0.0, 2.0]),
        n_frames=2,
    )

    plot_msd_profiles([profile_a, profile_b], show=False)
    assert captured["labels"] == ["A", "B"]
