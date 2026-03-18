import json

import h5py
import numpy as np
import pytest
from ase import Atoms

import linak.analysis.coordination as coordination_module
from linak.analysis.coordination import (
    CoordinationCutoffResolution,
    CoordinationProfile,
    compute_coordination_profile,
    load_coordination_profile,
    plot_coordination_profile,
    resolve_coordination_cutoff,
    save_coordination_profile,
)
from linak.analysis.rdf import RDFProfile, save_rdf_profile


def _coordination_test_frames() -> list[Atoms]:
    frame0 = Atoms(
        "PtPtOH",
        positions=[
            [0.0, 0.0, 0.20],
            [1.0, 0.0, 0.20],
            [0.0, 0.0, 1.00],
            [0.70, 0.0, 1.00],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "PtPtOH",
        positions=[
            [0.0, 0.0, 0.30],
            [1.0, 0.0, 0.30],
            [0.0, 0.0, 1.20],
            [1.10, 0.0, 1.20],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return [frame0, frame1]


def test_compute_coordination_profile_tracks_continuous_cn_and_surface_distance():
    profile = compute_coordination_profile(
        _coordination_test_frames(),
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.4,
            mode="direct",
        ),
    )

    assert profile.species_a == "O"
    assert profile.species_b == "H"
    np.testing.assert_array_equal(profile.atom_indices, np.array([2]))
    np.testing.assert_allclose(profile.time_fs, np.array([0.0, 2.0]))
    np.testing.assert_allclose(profile.distance_to_surface[:, 0], np.array([0.80, 0.90]))
    assert profile.coordination_number.shape == (2, 1)
    assert profile.coordination_number[0, 0] == pytest.approx(1.0)
    assert 0.0 < profile.coordination_number[1, 0] < 1.0


def test_resolve_coordination_cutoff_from_rdf_file_saves_diagnostic_plot(tmp_path):
    rdf_path = tmp_path / "reference_rdf.h5"
    diagnostic = tmp_path / "reference_rdf_cutoff.png"
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float),
        bin_centers=np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75], dtype=float),
        g_r=np.array([0.2, 1.8, 1.2, 0.35, 0.55, 0.9], dtype=float),
        n_frames=10,
    )
    save_rdf_profile(profile, rdf_path)

    resolution = resolve_coordination_cutoff(
        frames=[],
        species_a="O",
        species_b="H",
        cutoff_A=None,
        cutoff_rdf_path=rdf_path,
        cutoff_from_rdf=False,
        diagnostic_plot_output=diagnostic,
    )

    assert resolution.mode == "rdf_file"
    assert resolution.rdf_g_r is not None
    assert resolution.rdf_g_r_smoothed is not None
    assert resolution.rdf_peak_A is not None
    assert resolution.rdf_minimum_A is not None
    assert diagnostic.exists()
    assert resolution.cutoff_A == pytest.approx(resolution.rdf_minimum_A)


def test_resolve_coordination_cutoff_defaults_to_sampled_rdf(monkeypatch):
    def _fake_compute_reference_rdf(*args, **kwargs):
        return (
            np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75], dtype=float),
            np.array([0.2, 1.8, 1.2, 0.35, 0.55, 0.9], dtype=float),
        )

    monkeypatch.setattr(coordination_module, "_compute_reference_rdf", _fake_compute_reference_rdf)

    resolution = resolve_coordination_cutoff(
        frames=_coordination_test_frames(),
        species_a="O",
        species_b="H",
        cutoff_A=None,
        cutoff_rdf_path=None,
        cutoff_from_rdf=False,
    )

    assert resolution.mode == "sampled_rdf"
    assert resolution.rdf_sampled_frame_index is not None
    assert resolution.cutoff_A == pytest.approx(resolution.rdf_minimum_A)


def test_resolve_coordination_cutoff_prefers_direct_cutoff_over_other_sources(tmp_path):
    rdf_path = tmp_path / "reference_rdf.h5"
    save_rdf_profile(
        RDFProfile(
            species_a="O",
            species_b="H",
            bin_edges=np.array([0.0, 0.5, 1.0, 1.5], dtype=float),
            bin_centers=np.array([0.25, 0.75, 1.25], dtype=float),
            g_r=np.array([0.2, 1.8, 0.4], dtype=float),
            n_frames=4,
        ),
        rdf_path,
    )

    resolution = resolve_coordination_cutoff(
        frames=[],
        species_a="O",
        species_b="H",
        cutoff_A=2.2,
        cutoff_rdf_path=rdf_path,
        cutoff_from_rdf=True,
    )

    assert resolution.mode == "direct"
    assert resolution.cutoff_A == pytest.approx(2.2)


def test_save_and_load_coordination_profile(tmp_path):
    profile = compute_coordination_profile(
        _coordination_test_frames(),
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.4,
            mode="direct",
        ),
    )
    output = tmp_path / "coordination.h5"

    save_coordination_profile(profile, output)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["analysis"] == "coordination"
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["species_a"] == "O"
        assert metadata["species_b"] == "H"
        assert metadata["cutoff_A"] == pytest.approx(1.0)
        assert "coordination_number" in handle

    loaded = load_coordination_profile(output, species_a="O", species_b="H", axis="z")
    np.testing.assert_allclose(loaded.coordination_number, profile.coordination_number)
    np.testing.assert_allclose(loaded.distance_to_surface, profile.distance_to_surface)
    np.testing.assert_allclose(loaded.time_fs, profile.time_fs)


def test_plot_coordination_profile_defaults_to_distance(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_plot_line_series(x, y, **kwargs):
        captured["x"] = x
        captured["y"] = y
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.analysis.coordination.plot_line_series", _fake_plot_line_series)

    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        distance_to_surface=np.array([[0.8], [1.0]], dtype=float),
        coordination_number=np.array([[1.0], [0.5]], dtype=float),
        n_frames=2,
        n_atoms=1,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    plot_coordination_profile(profile, show=False)
    assert captured["x_label"] == "Distance to surface (A)"
    assert captured["y_label"] == "Coordination number"
    assert len(captured["x"]) > 0


def test_plot_coordination_profile_time_uses_atom_series(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_plot_multi_line_series(x_series, y_series, labels, **kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["labels"] = labels
        captured["x_label"] = kwargs["x_label"]
        return None

    monkeypatch.setattr(
        "linak.analysis.coordination.plot_multi_line_series",
        _fake_plot_multi_line_series,
    )

    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2, 3]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        distance_to_surface=np.array([[0.8, 1.2], [0.9, 1.3]], dtype=float),
        coordination_number=np.array([[1.0, 0.5], [0.8, 0.4]], dtype=float),
        n_frames=2,
        n_atoms=2,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    plot_coordination_profile(profile, component="time", show=False)
    assert captured["x_label"] == "Time (ps)"
    assert captured["labels"] == ["O[2]", "O[3]"]
    assert len(captured["y_series"]) == 2
