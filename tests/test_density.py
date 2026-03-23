import json
import logging
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
import h5py

import linak.analysis.density as density_module
from linak.analysis.density import (
    available_element_species,
    compute_density_profiles,
    compute_density_profile,
    estimate_surface_position,
    load_density_profile,
    load_density_profiles,
    normalize_backend_name,
    save_density_profile,
    save_density_profiles,
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
        profile.density,
        np.array(
            [
                (0.01 * oxygen_mass_g) * 1.0e24,
                (0.01 * oxygen_mass_g) * 1.0e24,
            ]
        ),
    )
    assert profile.units == "g/cm^3"


def test_density_profile_cell_binning_extends_to_empty_bins():
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    observed = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=1.0,
        binning="observed",
    )
    cell_binned = compute_density_profile(
        frames=[frame],
        species="O",
        axis="z",
        bin_width=1.0,
        binning="cell",
    )

    assert cell_binned.density.size > observed.density.size
    assert np.count_nonzero(cell_binned.density == 0.0) > 0
    np.testing.assert_allclose(
        np.sum(cell_binned.counts_per_frame),
        np.sum(observed.counts_per_frame),
    )


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
        bin_width=0.5,
    )

    np.testing.assert_allclose(profile.counts_per_frame, np.array([h2o_mass_g, h2o_mass_g]))
    np.testing.assert_allclose(profile.density, np.array([h2o_mass_g / 0.5, h2o_mass_g / 0.5]))
    assert profile.species == "H2O"
    assert profile.units == "g/Angstrom"


def test_select_water_axis_values_with_masses_uses_com_and_pbc():
    frame = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 9.90],
            [0.0, 0.0, 0.15],
            [0.0, 0.0, 9.95],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    axis_values, masses = density_module._select_water_axis_values_with_masses(frame, axis_index=2)

    atomic_masses_amu = np.asarray(frame.get_masses(), dtype=float)
    expected_com_z = (
        atomic_masses_amu[0] * 9.90 + atomic_masses_amu[1] * 10.15 + atomic_masses_amu[2] * 9.95
    ) / np.sum(atomic_masses_amu)
    expected_mass_g = float(np.sum(atomic_masses_amu) * 1.66053906660e-24)

    np.testing.assert_allclose(axis_values, np.array([expected_com_z]))
    np.testing.assert_allclose(masses, np.array([expected_mass_g]))


def test_water_molecule_triplets_exclude_oxygen_with_third_hydrogen():
    frame = Atoms(
        "OHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.30, 0.90, 0.0],
            [0.0, -0.95, 0.0],
        ],
    )

    water_triplets = density_module._water_molecule_triplets(frame)

    assert water_triplets.shape == (0, 3)


def test_compute_density_profiles_all_includes_elements_and_h2o():
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
    assert [profile.species for profile in profiles] == ["H", "O", "H2O"]


def test_compute_density_profiles_all_reuses_cached_h2o_topology(monkeypatch):
    frame = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.10],
            [0.8, 0.0, 0.10],
            [-0.4, 0.7, 0.10],
        ],
    )
    frames = [frame.copy() for _ in range(density_module.H2O_VALIDATION_STRIDE + 1)]
    original = density_module._water_molecule_triplets
    call_count = 0

    def counting_water_molecule_triplets(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        density_module,
        "_water_molecule_triplets",
        counting_water_molecule_triplets,
    )

    profiles = compute_density_profiles(
        frames=frames,
        species="all",
        axis="z",
        bin_width=1.0,
    )

    assert [profile.species for profile in profiles] == ["H", "O", "H2O"]
    assert call_count == 2


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
        assert "counts_per_frame" not in handle
        assert "density" in handle
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["bin_width_A"] == pytest.approx(1.0)
        assert metadata["units_map"]["bin_width_A"] == "Angstrom"
        assert metadata["units_map"]["density"] == "g/cm^3"


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


def test_save_density_profiles_writes_hdf5_collection(tmp_path):
    frame = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.10],
            [0.8, 0.0, 0.10],
            [-0.4, 0.7, 0.10],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profiles = compute_density_profiles(
        frames=[frame],
        species="all",
        axis="z",
        bin_width=1.0,
    )
    out = tmp_path / "density_collection.h5"

    saved_path = save_density_profiles(profiles, out)

    assert saved_path == out.resolve()
    loaded = load_density_profiles(out)
    assert [profile.species for profile in loaded] == ["H", "O", "H2O"]


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
    assert loaded.units == "g/cm^3"


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


def test_density_surface_logging_explains_reference_estimator_and_fill(caplog):
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

    caplog.set_level(logging.INFO, logger="linak.analysis.density")
    compute_density_profile(
        frames=[frame1, frame2, frame3, frame4],
        species="O",
        axis="z",
        bin_width=0.2,
        surface_mode="auto",
        surface_elements=["Pt"],
    )

    assert "Surface reference along Z: user-selected Pt reference atoms." in caplog.text
    assert (
        "Surface estimator along Z: layered top-layer mean on Z using Pt reference atoms"
        in caplog.text
    )
    assert (
        "Surface estimator gaps along Z: filled 1 missing frame values with tracked "
        "top-layer mean from nearest valid layered frames."
    ) in caplog.text
    assert "per-frame Z q90 of Pt reference atoms" not in caplog.text
    assert "Frame-wise Z surface alignment active:" in caplog.text
    assert "summary of per-frame surface estimates:" in caplog.text


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
    from linak.analysis.density import DensityProfile, plot_density_profile

    captured = {}

    def _fake_plot_line_series(x, y, **kwargs):
        captured["x"] = x
        captured["y"] = y
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.analysis.density.plot_line_series", _fake_plot_line_series)

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
    assert captured["x_label"] == "Distance to the surface ($\\mathrm{\\AA}$)"
    assert captured["y_label"] == "Density (g/A)"


def test_plot_density_profile_preserves_explicit_blank_axis_labels(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profile

    captured = {}

    def _fake_plot_line_series(_x, _y, **kwargs):
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.analysis.density.plot_line_series", _fake_plot_line_series)

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

    plot_density_profile(profile, show=False, x_label="", y_label="")

    assert captured["x_label"] == ""
    assert captured["y_label"] == ""


def test_plot_density_profiles_use_g_per_cm3_without_si_scaling(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured = {}

    def _fake_plot_multi_line_series(x_series, y_series, _labels, **kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr(
        "linak.analysis.density.plot_multi_line_series", _fake_plot_multi_line_series
    )

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
    assert captured["x_label"] == "Distance to the surface ($\\mathrm{\\AA}$)"
    assert captured["y_label"] == "Density (g/cm^3)"


def test_plot_density_profiles_auto_limits_ignore_all_zero_tails(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured = {}

    def _fake_plot_multi_line_series(_x_series, _y_series, _labels, **kwargs):
        captured["x_lim"] = kwargs["x_lim"]
        captured["y_lim"] = kwargs["y_lim"]
        return None

    monkeypatch.setattr(
        "linak.analysis.density.plot_multi_line_series", _fake_plot_multi_line_series
    )

    bin_edges = np.arange(0.0, 41.0, 1.0, dtype=float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    values_a = np.concatenate((np.ones(20, dtype=float), np.zeros(20, dtype=float)))
    values_b = values_a * 2.0
    profile_a = DensityProfile(
        axis="z",
        species="A",
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_per_frame=values_a,
        density=values_a,
        units="g/Angstrom",
        n_frames=1,
        surface_position=0.0,
    )
    profile_b = DensityProfile(
        axis="z",
        species="B",
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts_per_frame=values_b,
        density=values_b,
        units="g/Angstrom",
        n_frames=1,
        surface_position=0.0,
    )

    plot_density_profiles([profile_a, profile_b], show=False, x_mode="axis")

    assert captured["x_lim"] is not None
    assert captured["x_lim"][1] > 20.0
    assert captured["x_lim"][1] < 25.0
    assert captured["y_lim"] is not None
    assert captured["y_lim"][0] == pytest.approx(0.0)
    assert captured["y_lim"][1] > 2.0


def test_plot_density_profile_supports_number_density(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profile

    captured = {}

    def _fake_plot_line_series(_x, y, **kwargs):
        captured["y"] = y
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.analysis.density.plot_line_series", _fake_plot_line_series)
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
    np.testing.assert_allclose(captured["y"], np.array([250.0]))
    assert captured["y_label"] == "Number density (atom/nm^3)"


def test_plot_line_series_keeps_explicit_y_limits_when_ticks_are_outside_range():
    from linak.plot.plotting import plot_line_series

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


def test_plot_line_series_can_hide_ticks_for_one_axis_only():
    from linak.plot.plotting import plot_line_series

    x_hidden_capture: dict[str, object] = {}
    plot_line_series(
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, 2.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        ticks_visible=False,
        tick_params_kwargs={"_ticks_axis": "x"},
        capture_state=x_hidden_capture,
    )

    both_hidden_capture: dict[str, object] = {}
    plot_line_series(
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, 2.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        ticks_visible=False,
        capture_state=both_hidden_capture,
    )

    assert x_hidden_capture["ticks"] is True
    assert both_hidden_capture["ticks"] is False


def test_plot_line_series_grid_off_stays_off_even_with_grid_kwargs(monkeypatch):
    import matplotlib.pyplot as plt

    from linak.plot import plotting as plotting_mod

    monkeypatch.setattr(plotting_mod, "configure_matplotlib_backend", lambda **_kwargs: "Agg")
    monkeypatch.setattr(plotting_mod, "_import_pyplot", lambda: plt)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plotting_mod.plot_line_series(
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, 2.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        style=plotting_mod.with_style_overrides(grid=False),
        grid_kwargs={"axis": "both", "which": "major", "color": "#ff0000"},
    )

    figure = plt.gcf()
    try:
        axes = figure.axes[0]
        assert not any(line.get_visible() for line in axes.get_xgridlines())
        assert not any(line.get_visible() for line in axes.get_ygridlines())
    finally:
        plt.close(figure)


def test_plot_line_series_non_blocking_show_keeps_figure_open(monkeypatch):
    import matplotlib.pyplot as plt

    from linak.plot import plotting as plotting_mod

    show_blocks: list[bool | None] = []
    pause_calls: list[float] = []
    close_calls: list[object] = []

    monkeypatch.setattr(plotting_mod, "configure_matplotlib_backend", lambda **_kwargs: "QtAgg")
    monkeypatch.setattr(plotting_mod, "_import_pyplot", lambda: plt)
    monkeypatch.setattr(
        plt, "show", lambda *args, **kwargs: show_blocks.append(kwargs.get("block"))
    )
    monkeypatch.setattr(plt, "pause", lambda value: pause_calls.append(float(value)))
    monkeypatch.setattr(
        plt, "close", lambda *args, **_kwargs: close_calls.append(args[0] if args else None)
    )

    plotting_mod.plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([0.2, 0.6, 1.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        show=True,
        show_blocking=False,
    )

    assert show_blocks == [False]
    assert pause_calls == [pytest.approx(0.001)]
    assert close_calls == []


def test_plot_line_series_blocking_show_closes_figure(monkeypatch):
    import matplotlib.pyplot as plt

    from linak.plot import plotting as plotting_mod

    show_blocks: list[bool | None] = []
    close_calls: list[object] = []

    monkeypatch.setattr(plotting_mod, "configure_matplotlib_backend", lambda **_kwargs: "QtAgg")
    monkeypatch.setattr(plotting_mod, "_import_pyplot", lambda: plt)
    monkeypatch.setattr(
        plt, "show", lambda *args, **kwargs: show_blocks.append(kwargs.get("block"))
    )
    monkeypatch.setattr(
        plt, "close", lambda *args, **_kwargs: close_calls.append(args[0] if args else None)
    )

    plotting_mod.plot_line_series(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([0.2, 0.6, 1.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        show=True,
        show_blocking=True,
    )

    assert show_blocks == [True]
    assert len(close_calls) == 1


def test_plot_multi_line_series_hides_disabled_series_in_capture_state():
    from linak.plot.plotting import plot_multi_line_series

    captured = {}
    plot_multi_line_series(
        [np.array([0.0, 1.0], dtype=float), np.array([0.0, 1.0], dtype=float)],
        [np.array([1.0, 2.0], dtype=float), np.array([2.0, 3.0], dtype=float)],
        ["run-a", "run-b"],
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        series_enabled=[False, True],
        capture_state=captured,
    )

    assert captured["series_labels"] == ["run-b"]
    assert len(captured["line_colors"]) == 1


def test_plot_multi_line_series_accepts_advanced_line_kwargs():
    from linak.plot.plotting import plot_multi_line_series

    captured = {}
    plot_multi_line_series(
        [np.array([0.0, 1.0], dtype=float), np.array([0.0, 1.0], dtype=float)],
        [np.array([1.0, 2.0], dtype=float), np.array([2.0, 3.0], dtype=float)],
        ["run-a", "run-b"],
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        line_kwargs={"linestyle": "--"},
        series_line_kwargs=[{"alpha": 0.4}, {"alpha": 0.8}],
        capture_state=captured,
    )

    assert captured["series_labels"] == ["run-a", "run-b"]


def test_plot_multi_line_series_preserves_explicit_labels_and_legend_font_size():
    from linak.plot.plotting import plot_multi_line_series, with_style_overrides

    captured = {}
    plot_multi_line_series(
        [np.array([0.0, 1.0], dtype=float), np.array([0.0, 1.0], dtype=float)],
        [np.array([1.0, 2.0], dtype=float), np.array([2.0, 3.0], dtype=float)],
        ["custom-a", "custom-b"],
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        style=with_style_overrides(legend_font_size=17),
        line_kwargs={"label": "stale-label"},
        capture_state=captured,
    )

    assert captured["series_labels"] == ["custom-a", "custom-b"]
    assert captured["legend_font_size"] == 17


def test_plot_multi_line_series_applies_axis_label_padding():
    from linak.plot.plotting import plot_multi_line_series

    captured = {}
    plot_multi_line_series(
        [np.array([0.0, 1.0], dtype=float)],
        [np.array([1.0, 2.0], dtype=float)],
        ["run-a"],
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        x_label_pad=14.0,
        y_label_pad=18.0,
        capture_state=captured,
    )

    assert captured["x_label_pad"] == pytest.approx(14.0)
    assert captured["y_label_pad"] == pytest.approx(18.0)


def test_plot_density_profiles_accepts_axis_label_padding(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured_kwargs = {}

    def _fake_plot_multi_line_series(_x_series, _y_series, _labels, **kwargs):
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        density_module,
        "plot_multi_line_series",
        _fake_plot_multi_line_series,
    )

    profile_a = DensityProfile(
        axis="z",
        species="Au",
        bin_edges=np.array([-0.5, 0.5, 1.5], dtype=float),
        bin_centers=np.array([0.0, 1.0], dtype=float),
        counts_per_frame=np.array([1.0, 2.0], dtype=float),
        density=np.array([1.0, 2.0], dtype=float),
        units="g/cm^3",
        n_frames=2,
        coordinate_mode="distance",
        surface_position=0.0,
        surface_position_std=0.0,
    )
    profile_b = DensityProfile(
        axis="z",
        species="H2O",
        bin_edges=np.array([-0.5, 0.5, 1.5], dtype=float),
        bin_centers=np.array([0.0, 1.0], dtype=float),
        counts_per_frame=np.array([2.0, 3.0], dtype=float),
        density=np.array([2.0, 3.0], dtype=float),
        units="g/cm^3",
        n_frames=2,
        coordinate_mode="distance",
        surface_position=0.0,
        surface_position_std=0.0,
    )

    plot_density_profiles(
        [profile_a, profile_b],
        show=False,
        x_label_pad=11.0,
        y_label_pad=13.0,
    )

    assert captured_kwargs["x_label_pad"] == pytest.approx(11.0)
    assert captured_kwargs["y_label_pad"] == pytest.approx(13.0)


def test_plot_density_profiles_auto_limits_follow_normalized_data(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured_kwargs = {}

    def _fake_plot_multi_line_series(_x_series, _y_series, _labels, **kwargs):
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(
        density_module,
        "plot_multi_line_series",
        _fake_plot_multi_line_series,
    )

    profile_a = DensityProfile(
        axis="z",
        species="Au",
        bin_edges=np.array([-0.5, 0.5, 1.5], dtype=float),
        bin_centers=np.array([0.0, 1.0], dtype=float),
        counts_per_frame=np.array([1.0, 2.0], dtype=float),
        density=np.array([10.0, 20.0], dtype=float),
        units="g/cm^3",
        n_frames=2,
        coordinate_mode="distance",
        surface_position=0.0,
        surface_position_std=0.0,
    )
    profile_b = DensityProfile(
        axis="z",
        species="H2O",
        bin_edges=np.array([-0.5, 0.5, 1.5], dtype=float),
        bin_centers=np.array([0.0, 1.0], dtype=float),
        counts_per_frame=np.array([2.0, 3.0], dtype=float),
        density=np.array([100.0, 200.0], dtype=float),
        units="g/cm^3",
        n_frames=2,
        coordinate_mode="distance",
        surface_position=0.0,
        surface_position_std=0.0,
    )

    plot_density_profiles(
        [profile_a, profile_b],
        show=False,
        series_normalization_modes=["max", "max"],
        series_normalization_values=[1.0, 1.0],
        series_normalization_x_refs=[None, None],
    )

    assert captured_kwargs["x_lim"] == pytest.approx([-0.05, 1.05])
    assert captured_kwargs["y_lim"] == pytest.approx([0.0, 1.05])


def test_plot_multi_line_series_rejects_invalid_series_line_kwargs_length():
    from linak.plot.plotting import plot_multi_line_series

    with pytest.raises(ValueError, match="series_line_kwargs count must match"):
        plot_multi_line_series(
            [np.array([0.0, 1.0], dtype=float), np.array([0.0, 1.0], dtype=float)],
            [np.array([1.0, 2.0], dtype=float), np.array([2.0, 3.0], dtype=float)],
            ["run-a", "run-b"],
            title="demo",
            x_label="x",
            y_label="y",
            show=False,
            series_line_kwargs=[{"alpha": 0.5}],
        )


def test_plot_series_data_transform_supports_rebinning_and_max_normalization():
    from linak.plot import plotting as plotting_mod

    x_series, y_series, normalized_count = plotting_mod._prepare_plot_series_data(
        x_series=[np.array([0.0, 0.2, 0.8, 1.0], dtype=float)],
        y_series=[np.array([1.0, 3.0, 2.0, 6.0], dtype=float)],
        labels=["run-a"],
        x_bin_width=0.5,
        x_bin_reducer="mean",
        series_normalization_modes=["max"],
        series_normalization_values=[2.0],
        series_normalization_x_refs=[None],
    )

    assert normalized_count == 1
    assert x_series[0].size == 3
    assert np.max(y_series[0]) == pytest.approx(2.0)


def test_plot_multi_line_series_warns_for_mixed_normalization(caplog):
    from linak.plot.plotting import plot_multi_line_series

    caplog.set_level(logging.WARNING, logger="linak.plot.plotting")
    plot_multi_line_series(
        [np.array([0.0, 1.0], dtype=float), np.array([0.0, 1.0], dtype=float)],
        [np.array([1.0, 2.0], dtype=float), np.array([2.0, 3.0], dtype=float)],
        ["run-a", "run-b"],
        title="demo",
        x_label="x",
        y_label="y",
        show=False,
        series_normalization_modes=["max", "none"],
        series_normalization_values=[1.0, None],
        series_normalization_x_refs=[None, None],
    )

    assert "Only 1/2 plotted series are normalized" in caplog.text


def test_plot_line_series_can_suppress_save_log(tmp_path, caplog):
    from linak.plot.plotting import plot_line_series

    caplog.set_level(logging.INFO, logger="linak.plot.plotting")
    output = tmp_path / "quiet_plot.png"
    plot_line_series(
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, 2.0], dtype=float),
        title="demo",
        x_label="x",
        y_label="y",
        output=output,
        show=False,
        suppress_output_log=True,
    )

    assert output.exists()
    assert "Saved plot to" not in caplog.text
