import json
import logging
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
import h5py

from linak.density import (
    available_element_species,
    compute_density_profiles,
    compute_density_profile,
    estimate_surface_position,
    load_density_profile,
    normalize_backend_name,
    save_density_profile,
)


def test_density_profile_linear_density_without_cell():
    frame1 = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]])
    frame2 = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    oxygen_mass_g = float(frame1.get_masses()[0]) * 1.66053906660e-24

    profile = compute_density_profile(
        frames=[frame1, frame2],
        species="O",
        axis="z",
        bin_width=0.1,
    )

    np.testing.assert_allclose(
        np.sort(profile.counts_per_frame),
        np.sort(np.array([1.5 * oxygen_mass_g, 0.5 * oxygen_mass_g])),
    )
    np.testing.assert_allclose(
        np.sort(profile.density),
        np.sort(np.array([15.0 * oxygen_mass_g, 5.0 * oxygen_mass_g])),
    )
    assert profile.coordinate_mode == "distance"
    assert profile.units == "g/Angstrom"
    assert profile.axis == "z"


def test_density_profile_volumetric_density_with_cell():
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    oxygen_mass_g = float(frame.get_masses()[0]) * 1.66053906660e-24

    profile = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=1.0,
    )

    np.testing.assert_allclose(profile.counts_per_frame, np.array([oxygen_mass_g, oxygen_mass_g]))
    np.testing.assert_allclose(
        profile.density, np.array([0.01 * oxygen_mass_g, 0.01 * oxygen_mass_g])
    )
    assert profile.units == "g/Angstrom^3"


def test_density_profile_estimates_surface_from_low_mobility_atoms():
    frame1 = Atoms(
        "PtPtPtHH",
        positions=[
            [0.0, 0.0, 0.05],
            [1.0, 0.0, 0.20],
            [2.0, 0.0, 0.35],
            [0.5, 0.0, 3.10],
            [1.5, 0.0, 3.40],
        ],
        cell=[10.0, 10.0, 12.0],
        pbc=True,
    )
    frame2 = Atoms(
        "PtPtPtHH",
        positions=[
            [0.0, 0.0, 0.06],
            [1.0, 0.0, 0.19],
            [2.0, 0.0, 0.36],
            [0.5, 0.0, 4.10],
            [1.5, 0.0, 4.40],
        ],
        cell=[10.0, 10.0, 12.0],
        pbc=True,
    )

    profile = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="auto",
    )

    assert profile.surface_position is not None
    assert 0.15 < profile.surface_position < 0.25


def test_density_profile_surface_elements_override_changes_layer_reference():
    frame = Atoms(
        "AuAuAuAuOOOO",
        positions=[
            [0.0, 0.0, 0.10],
            [1.0, 0.0, 0.10],
            [0.0, 1.0, 1.60],
            [1.0, 1.0, 1.60],
            [0.0, 0.0, 2.50],
            [1.0, 0.0, 2.55],
            [0.0, 1.0, 2.50],
            [1.0, 1.0, 2.55],
        ],
        cell=[10.0, 10.0, 12.0],
        pbc=True,
    )

    auto_profile = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=0.2,
        surface_mode="layered",
    )
    au_only_profile = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=0.2,
        surface_mode="layered",
        surface_elements=["Au"],
    )

    assert auto_profile.surface_position is not None
    assert au_only_profile.surface_position is not None
    assert auto_profile.surface_position == pytest.approx(1.60)
    assert au_only_profile.surface_position == pytest.approx(1.60)


def test_density_profile_rough_surface_uses_framewise_mean_of_reference_atoms():
    frame1 = Atoms(
        "PtPtPtPtPtPtHH",
        positions=[
            [0.0, 0.0, 1.00],
            [1.0, 0.0, 2.00],
            [2.0, 0.0, 3.00],
            [3.0, 0.0, 5.00],
            [4.0, 0.0, 6.00],
            [5.0, 0.0, 7.00],
            [0.5, 0.0, 8.00],
            [1.5, 0.0, 8.40],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )
    frame2 = Atoms(
        "PtPtPtPtPtPtHH",
        positions=[
            [0.0, 0.0, 1.10],
            [1.0, 0.0, 2.10],
            [2.0, 0.0, 3.10],
            [3.0, 0.0, 6.80],
            [4.0, 0.0, 8.00],
            [5.0, 0.0, 9.20],
            [0.5, 0.0, 8.20],
            [1.5, 0.0, 8.50],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )

    profile = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
    )

    assert profile.surface_position is not None
    assert profile.surface_position == pytest.approx(2.05, abs=1e-8)


def test_density_profile_rough_surface_excludes_fixed_atoms_by_default():
    frame1 = Atoms(
        "PtPtPtPtPtPtHH",
        positions=[
            [0.0, 0.0, 0.10],
            [1.0, 0.0, 0.20],
            [2.0, 0.0, 0.30],
            [0.0, 1.0, 2.10],
            [1.0, 1.0, 2.20],
            [2.0, 1.0, 2.30],
            [0.5, 0.5, 3.80],
            [1.5, 0.5, 4.10],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )
    frame2 = Atoms(
        "PtPtPtPtPtPtHH",
        positions=[
            [0.0, 0.0, 0.10],
            [1.0, 0.0, 0.20],
            [2.0, 0.0, 0.30],
            [0.0, 1.0, 2.25],
            [1.0, 1.0, 2.35],
            [2.0, 1.0, 2.45],
            [0.5, 0.5, 3.90],
            [1.5, 0.5, 4.30],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )
    fixed = FixAtoms(indices=[0, 1, 2])
    frame1.set_constraint(fixed)
    frame2.set_constraint(fixed)

    without_fixed = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    with_fixed = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
        include_fixed_surface_atoms=True,
    )

    assert without_fixed.surface_position is not None
    assert with_fixed.surface_position is not None
    assert without_fixed.surface_position > 2.2
    assert with_fixed.surface_position < 0.5


def test_density_layered_mode_auto_surface_elements_ignore_mobile_water_oxygen():
    frame1 = Atoms(
        "PtPtPtPtOOOO",
        positions=[
            [0.0, 0.0, 0.10],
            [1.0, 0.0, 0.10],
            [0.0, 1.0, 1.60],
            [1.0, 1.0, 1.60],
            [0.0, 0.0, 2.90],
            [1.0, 0.0, 3.10],
            [0.0, 1.0, 3.30],
            [1.0, 1.0, 3.50],
        ],
        cell=[10.0, 10.0, 12.0],
        pbc=True,
    )
    frame2 = Atoms(
        "PtPtPtPtOOOO",
        positions=[
            [0.0, 0.0, 0.10],
            [1.0, 0.0, 0.10],
            [0.0, 1.0, 1.62],
            [1.0, 1.0, 1.62],
            [0.0, 0.0, 4.10],
            [1.0, 0.0, 4.25],
            [0.0, 1.0, 4.35],
            [1.0, 1.0, 4.55],
        ],
        cell=[10.0, 10.0, 12.0],
        pbc=True,
    )

    position, _std, method = estimate_surface_position(
        [frame1, frame2],
        axis="z",
        mode="layered",
    )

    assert position is not None
    assert position == pytest.approx(1.61, abs=0.05)
    assert method.startswith("layered_top_layer_mean")


def test_density_profile_raises_for_missing_species():
    frame = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="No entities found for selection"):
        compute_density_profile(frames=[frame], species="O", axis="z", bin_width=0.1)


def test_density_profile_defaults_to_all_atoms():
    frame1 = Atoms("OH", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]])
    frame2 = Atoms("OH", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    mass_o = float(frame1.get_masses()[0]) * 1.66053906660e-24
    mass_h = float(frame1.get_masses()[1]) * 1.66053906660e-24

    profile = compute_density_profile(
        frames=[frame1, frame2],
        species="all",
        axis="z",
        bin_width=0.1,
    )

    np.testing.assert_allclose(
        profile.counts_per_frame,
        np.array([(2.0 * mass_o + mass_h) / 2.0, mass_h / 2.0]),
    )
    np.testing.assert_allclose(
        profile.density,
        np.array([((2.0 * mass_o + mass_h) / 2.0) / 0.1, (mass_h / 2.0) / 0.1]),
    )
    assert profile.species == "ALL"
    assert profile.units == "g/Angstrom"


def test_density_profile_h2o_counts_water_molecules():
    frame = Atoms(
        "OHHOHH",
        positions=[
            [0.0, 0.0, 0.10],  # O
            [0.95, 0.0, 0.10],  # H
            [-0.30, 0.90, 0.10],  # H
            [0.0, 0.0, 1.10],  # O
            [0.95, 0.0, 1.10],  # H
            [-0.30, 0.90, 1.10],  # H
        ],
    )

    h2o_mass_g = float(Atoms("H2O").get_masses().sum()) * 1.66053906660e-24

    profile = compute_density_profile(
        frames=[frame],
        species="H2O",
        axis="z",
        bin_width=1.0,
    )

    np.testing.assert_allclose(profile.counts_per_frame, np.array([h2o_mass_g, h2o_mass_g]))
    np.testing.assert_allclose(profile.density, np.array([h2o_mass_g, h2o_mass_g]))
    assert profile.species == "H2O"
    assert profile.units == "g/Angstrom"


def test_compute_density_profiles_all_is_element_resolved():
    frame = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.10],
            [0.8, 0.0, 0.10],
            [-0.4, 0.7, 0.10],
        ],
    )
    profiles = compute_density_profiles(
        frames=[frame],
        species="all",
        axis="z",
        bin_width=1.0,
    )
    assert [profile.species for profile in profiles] == ["H", "O"]


def test_available_element_species_is_sorted_unique():
    frame1 = Atoms("OH")
    frame2 = Atoms("AuH")
    assert available_element_species([frame1, frame2]) == ["Au", "H", "O"]


def test_save_density_profile_writes_hdf5(tmp_path):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=1.0,
    )

    out = tmp_path / "density.h5"
    saved_path = save_density_profile(profile, out)
    assert saved_path == out.resolve()
    with h5py.File(out, "r") as handle:
        assert handle.attrs["analysis"] == "density"
        assert "metadata_json" in handle.attrs
        assert "created_utc" in handle.attrs
        assert "linak_version" in handle.attrs
        assert "bin_edges_A" not in handle
        assert "bin_centers_A" in handle
        assert "density" in handle
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["bin_width_A"] == pytest.approx(1.0)
        assert metadata["units_map"]["bin_width_A"] == "Angstrom"


def test_save_and_load_density_profile(tmp_path):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=1.0,
    )
    out = tmp_path / "density.h5"

    save_density_profile(profile, out)
    loaded = load_density_profile(out, axis="z", species="O")

    np.testing.assert_allclose(loaded.bin_centers, profile.bin_centers)
    np.testing.assert_allclose(loaded.density, profile.density)
    np.testing.assert_allclose(loaded.number_density, profile.number_density)
    assert loaded.units == profile.units
    assert loaded.coordinate_mode == profile.coordinate_mode
    if profile.surface_position is None:
        assert loaded.surface_position is None
    else:
        assert loaded.surface_position == pytest.approx(profile.surface_position)
    assert loaded.species == "O"
    assert loaded.axis == "z"


def test_load_density_profile_supports_legacy_bin_edges_dataset(tmp_path):
    out = tmp_path / "legacy_density.h5"
    with h5py.File(out, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["analysis"] = "density"
        handle.attrs["metadata_json"] = json.dumps(
            {
                "axis": "z",
                "species": "O",
                "units": "g/Angstrom^3",
                "n_frames": 1,
                "coordinate_mode": "axis",
            }
        )
        handle.create_dataset("bin_edges_A", data=np.array([0.0, 1.0, 2.0], dtype=float))
        handle.create_dataset("bin_centers_A", data=np.array([0.5, 1.5], dtype=float))
        handle.create_dataset("counts_per_frame", data=np.array([1.0, 2.0], dtype=float))
        handle.create_dataset("density", data=np.array([0.1, 0.2], dtype=float))

    loaded = load_density_profile(out, axis="z", species="O")
    np.testing.assert_allclose(loaded.bin_edges, np.array([0.0, 1.0, 2.0]))


def test_density_profile_uses_framewise_surface_distance_for_binning():
    frame1 = Atoms(
        "PtO",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    frame2 = Atoms(
        "PtO",
        positions=[
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
    )

    profile = compute_density_profile(
        frames=[frame1, frame2],
        species="O",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
    )

    assert profile.coordinate_mode == "distance"
    assert profile.bin_centers.size == 2
    assert profile.counts_per_frame[1] == pytest.approx(0.0)


def test_density_profile_auto_mode_fills_missing_framewise_surface_with_quantile(caplog):
    frame1 = Atoms(
        "PtPtPtPtO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [2.0, 0.0, 1.2],
            [3.0, 0.0, 1.3],
            [0.5, 0.0, 2.0],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )
    frame2 = Atoms(
        "PtPtPtPtO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [2.0, 0.0, 0.2],
            [3.0, 0.0, 0.3],
            [0.5, 0.0, 2.1],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )
    frame3 = Atoms(
        "PtPtPtPtO",
        positions=[
            [0.0, 0.0, 0.05],
            [1.0, 0.0, 0.15],
            [2.0, 0.0, 1.25],
            [3.0, 0.0, 1.35],
            [0.5, 0.0, 2.05],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )
    frame4 = Atoms(
        "PtPtPtPtO",
        positions=[
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.2],
            [2.0, 0.0, 1.3],
            [3.0, 0.0, 1.4],
            [0.5, 0.0, 2.2],
        ],
        cell=[12.0, 12.0, 14.0],
        pbc=True,
    )

    with caplog.at_level(logging.WARNING):
        profile = compute_density_profile(
            frames=[frame1, frame2, frame3, frame4],
            species="O",
            axis="z",
            bin_width=0.2,
            surface_mode="auto",
            surface_elements=["Pt"],
        )

    assert profile.coordinate_mode == "distance"
    assert profile.surface_position is not None
    assert "frame-wise surface alignment was unavailable" not in caplog.text


def test_load_density_profile_rejects_csv_input(tmp_path):
    csv = tmp_path / "legacy_density.csv"
    csv.write_text(
        "bin_left_A,bin_right_A,bin_center_A,counts_per_frame,density_atoms_per_Angstrom\n"
        "0.0,1.0,0.5,2.0,2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Use .h5/.hdf5"):
        load_density_profile(csv, axis="z", species="O")


def test_normalize_backend_name_accepts_aliases():
    assert normalize_backend_name("tkagg") == "TkAgg"
    assert normalize_backend_name("QtAgg") == "QtAgg"


def test_normalize_backend_name_suggests_on_typo():
    with pytest.raises(ValueError, match="Did you mean 'TkAgg'"):
        normalize_backend_name("TgAgg")


def test_plot_density_profile_defaults_to_distance_axis(monkeypatch):
    from linak.density import DensityProfile, plot_density_profile

    captured = {}

    def _fake_plot_line_series(x, y, **kwargs):
        captured["x"] = x
        captured["y"] = y
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.density.plot_line_series", _fake_plot_line_series)

    profile = DensityProfile(
        axis="z",
        species="O",
        bin_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([0.5]),
        counts_per_frame=np.array([2.0e-6]),
        density=np.array([2.0e-6]),
        units="g/Angstrom",
        n_frames=1,
        surface_position=0.25,
    )

    plot_density_profile(profile, show=False)
    np.testing.assert_allclose(captured["x"], np.array([0.25]))
    np.testing.assert_allclose(captured["y"], np.array([2.0e-6]))
    assert captured["x_label"] == "Distance to surface (A)"
    assert captured["y_label"] == "Mass density (g/A)"


def test_plot_density_profiles_use_g_per_cm3_without_si_scaling(monkeypatch):
    from linak.density import DensityProfile, plot_density_profiles

    captured = {}

    def _fake_plot_multi_line_series(x_series, y_series, _labels, **kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.density.plot_multi_line_series", _fake_plot_multi_line_series)

    profile_a = DensityProfile(
        axis="z",
        species="A",
        bin_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([0.5]),
        counts_per_frame=np.array([1.0e-9]),
        density=np.array([1.0e-9]),
        units="g/Angstrom^3",
        n_frames=1,
        surface_position=0.25,
    )
    profile_b = DensityProfile(
        axis="z",
        species="B",
        bin_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([0.5]),
        counts_per_frame=np.array([2.0e-9]),
        density=np.array([2.0e-9]),
        units="g/Angstrom^3",
        n_frames=1,
        surface_position=0.25,
    )

    plot_density_profiles([profile_a, profile_b], show=False)
    np.testing.assert_allclose(captured["x_series"][0], np.array([0.25]))
    np.testing.assert_allclose(captured["y_series"][0], np.array([1.0e15]))
    np.testing.assert_allclose(captured["y_series"][1], np.array([2.0e15]))
    assert captured["x_label"] == "Distance to surface (A)"
    assert captured["y_label"] == "Mass density (g/cm^3)"


def test_plot_density_profile_supports_number_density(monkeypatch):
    from linak.density import DensityProfile, plot_density_profile

    captured = {}

    def _fake_plot_line_series(_x, y, **kwargs):
        captured["y"] = y
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.density.plot_line_series", _fake_plot_line_series)
    profile = DensityProfile(
        axis="z",
        species="O",
        bin_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([0.5]),
        counts_per_frame=np.array([1.0]),
        density=np.array([1.0]),
        units="g/cm^3",
        n_frames=1,
        number_density=np.array([0.25]),
        number_density_units="atoms/Angstrom^3",
        surface_position=0.0,
    )

    plot_density_profile(profile, show=False, quantity="number")
    np.testing.assert_allclose(captured["y"], np.array([0.25]))
    assert captured["y_label"] == "Number density (atoms/A^3)"


def test_plot_line_series_keeps_explicit_y_limits_when_ticks_are_outside_range():
    from linak.plotting import plot_line_series

    captured = {}
    plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([0.2, 0.6, 1.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        y_ticks=[-1.0, 0.0, 1.0, 2.0],
        y_lim=[0.0, None],
        capture_state=captured,
    )

    assert captured["y_lim"][0] == pytest.approx(0.0)
