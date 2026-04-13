import json
import logging
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
import h5py

from linak import __version__ as LINAK_VERSION
import linak.analysis.density as density_module
import linak.analysis.water as water_module
from linak.analysis.density import (
    SurfaceEstimatorOptions,
    available_element_species,
    compute_all_density_profiles,
    compute_density_profiles,
    compute_density_profile,
    estimate_surface_reference,
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

    np.testing.assert_allclose(profile.counts_per_frame, np.array([oxygen_mass_g, oxygen_mass_g]))
    np.testing.assert_allclose(
        profile.density,
        np.array([10.0 * oxygen_mass_g, 10.0 * oxygen_mass_g]),
    )
    assert profile.coordinate_mode == "distance"
    assert profile.surface_estimate is not None
    assert profile.surface_estimate.summary.valid_fraction == pytest.approx(1.0)
    assert profile.units == "g/Angstrom"
    assert profile.axis == "z"


def test_plot_density_profiles_keeps_descriptor_render_path_with_single_loaded_profile(
    monkeypatch, tmp_path
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)

    calls: list[str] = []

    def _fake_plot_density_profile(*_args, **_kwargs):
        calls.append("single")
        return tmp_path / "single.png"

    def _fake_plot_multi_line_series(*_args, **_kwargs):
        calls.append("multi")
        return tmp_path / "multi.png"

    monkeypatch.setattr(density_module, "plot_density_profile", _fake_plot_density_profile)
    monkeypatch.setattr(density_module, "plot_multi_line_series", _fake_plot_multi_line_series)

    result = density_module.plot_density_profiles(
        [profile],
        output=tmp_path / "density.png",
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


def test_density_profile_variable_cell_averages_framewise_normalized_density():
    frame1 = Atoms(
        "O",
        positions=[[0.0, 0.0, 0.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame2 = Atoms(
        "O",
        positions=[[0.0, 0.0, 0.10]],
        cell=[20.0, 10.0, 10.0],
        pbc=True,
    )
    oxygen_mass_g = float(frame1.get_masses()[0]) * 1.66053906660e-24

    profile = density_module._compute_density_profile_from_selected(
        frames=[frame1, frame2],
        selected_per_frame=[np.array([0.10]), np.array([0.10])],
        selected_masses_per_frame=[
            np.array([oxygen_mass_g], dtype=float),
            np.array([oxygen_mass_g], dtype=float),
        ],
        axis="z",
        axis_index=2,
        species_label="O",
        count_label="atoms",
        bin_width=1.0,
    )

    expected_density = (0.5 * ((oxygen_mass_g / 100.0) + (oxygen_mass_g / 200.0))) * 1.0e24
    expected_number_density = 0.5 * ((1.0 / 100.0) + (1.0 / 200.0)) * 1.0e3

    np.testing.assert_allclose(profile.density, np.array([expected_density, 0.0]))
    np.testing.assert_allclose(profile.number_density, np.array([expected_number_density, 0.0]))
    assert profile.number_density_units == "atom/nm^3"


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


def test_density_histogram_edges_follow_numpy_convention():
    frame = Atoms("OOO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    oxygen_mass_g = float(frame.get_masses()[0]) * 1.66053906660e-24

    profile = density_module._compute_density_profile_from_selected(
        frames=[frame],
        selected_per_frame=[np.array([0.0, 1.0, 2.0], dtype=float)],
        selected_masses_per_frame=[
            np.array([oxygen_mass_g, oxygen_mass_g, oxygen_mass_g], dtype=float)
        ],
        axis="z",
        axis_index=2,
        species_label="O",
        count_label="atoms",
        bin_width=1.0,
    )

    np.testing.assert_allclose(
        profile.counts_per_frame,
        np.array([oxygen_mass_g, oxygen_mass_g, oxygen_mass_g]),
    )
    np.testing.assert_allclose(profile.entities_per_frame, np.array([1.0, 1.0, 1.0]))

    histogram, _edges = np.histogram(np.array([0.0, 1.0, 2.0]), bins=np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(histogram, np.array([1, 2]))


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


def test_density_profile_rough_surface_respects_surface_side_selection():
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

    top_profile = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    bottom_profile = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
        surface_options=SurfaceEstimatorOptions(
            mode="rough",
            side="bottom",
            surface_elements=("Pt",),
        ),
    )

    assert top_profile.surface_position is not None
    assert bottom_profile.surface_position is not None
    assert top_profile.coordinate_mode == "axis"
    assert bottom_profile.coordinate_mode == "distance"
    assert top_profile.surface_position > bottom_profile.surface_position
    assert bottom_profile.surface_position == pytest.approx(2.05, abs=1e-8)


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
        surface_options=SurfaceEstimatorOptions(
            mode="rough",
            side="bottom",
            surface_elements=("Pt",),
        ),
    )
    with_fixed = compute_density_profile(
        frames=[frame1, frame2],
        species="H",
        axis="z",
        bin_width=0.2,
        surface_mode="rough",
        surface_elements=["Pt"],
        include_fixed_surface_atoms=True,
        surface_options=SurfaceEstimatorOptions(
            mode="rough",
            side="bottom",
            surface_elements=("Pt",),
            include_fixed_surface_atoms=True,
        ),
    )

    assert without_fixed.surface_position is not None
    assert with_fixed.surface_position is not None
    assert without_fixed.surface_position > 2.2
    assert with_fixed.surface_position < 0.5
    assert without_fixed.coordinate_mode == "distance"
    assert with_fixed.coordinate_mode == "distance"


def test_density_profile_rough_surface_prefers_outer_relaxed_layers_over_buried_fixed_like_layer():
    layer_z = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    xy_positions = [(0.0, 0.0), (1.5, 0.0), (0.75, 1.2)]
    symbols = "Au" * (len(layer_z) * len(xy_positions)) + "H"
    frame1_positions: list[list[float]] = []
    frame2_positions: list[list[float]] = []
    for layer_index, z_value in enumerate(layer_z):
        for atom_index, (x_value, y_value) in enumerate(xy_positions):
            frame1_positions.append([x_value, y_value, z_value])
            frame2_positions.append([x_value, y_value, z_value])
            if layer_index >= 4:
                displacement = 0.18 if atom_index % 2 == 0 else -0.16
                frame2_positions[-1][2] += displacement * (layer_index - 3)
    frame1_positions.append([0.5, 0.5, 12.0])
    frame2_positions.append([0.5, 0.5, 12.1])

    frames = [
        Atoms(
            symbols,
            positions=frame1_positions,
            cell=[14.0, 14.0, 24.0],
            pbc=True,
        ),
        Atoms(
            symbols,
            positions=frame2_positions,
            cell=[14.0, 14.0, 24.0],
            pbc=True,
        ),
    ]

    estimate = estimate_surface_reference(
        frames,
        axis="z",
        mode="rough",
        surface_elements=["Au"],
    )

    assert estimate is not None
    assert estimate.position is not None
    assert estimate.position > 7.5
    assert estimate.method.startswith("rough_low_mobility")


def test_density_profile_auto_surface_avoids_buried_fixed_like_layer():
    layer_z = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    xy_positions = [(0.0, 0.0), (1.5, 0.0), (0.75, 1.2)]
    symbols = "Au" * (len(layer_z) * len(xy_positions)) + "H"
    frames: list[Atoms] = []
    for frame_index in range(3):
        positions: list[list[float]] = []
        for layer_index, z_value in enumerate(layer_z):
            for atom_index, (x_value, y_value) in enumerate(xy_positions):
                offset = 0.0
                if layer_index >= 4:
                    offset = (0.10 + 0.05 * frame_index) * (1 if atom_index != 1 else -1)
                    offset *= layer_index - 3
                positions.append([x_value, y_value, z_value + offset])
        positions.append([0.5, 0.5, 12.0 + 0.05 * frame_index])
        frames.append(
            Atoms(
                symbols,
                positions=positions,
                cell=[14.0, 14.0, 24.0],
                pbc=True,
            )
        )

    estimate = estimate_surface_reference(
        frames,
        axis="z",
        mode="auto",
        surface_elements=["Au"],
    )

    assert estimate is not None
    assert estimate.position is not None
    assert estimate.position > 7.5


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
    assert method.startswith("layered_top_layer_median")


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
    np.testing.assert_allclose(profile.number_density, np.array([2.0, 2.0]))
    assert profile.species == "H2O"
    assert profile.units == "g/Angstrom"
    assert profile.number_density_units == "molecule/Angstrom"


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
    original = water_module.water_molecule_triplets
    call_count = 0

    def counting_water_molecule_triplets(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        water_module,
        "water_molecule_triplets",
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
    assert loaded.series_statistics is not None
    density_stats = loaded.series_statistics["density"]
    number_stats = loaded.series_statistics["number_density"]
    np.testing.assert_array_equal(density_stats.point_count, np.array([1, 1]))
    np.testing.assert_array_equal(number_stats.point_count, np.array([1, 1]))
    if profile.surface_position is None:
        assert loaded.surface_position is None
    else:
        assert loaded.surface_position == pytest.approx(profile.surface_position)
    if profile.surface_estimate is None:
        assert loaded.surface_estimate is None
    else:
        assert loaded.surface_estimate is not None
        np.testing.assert_allclose(
            loaded.surface_estimate.frame_values,
            profile.surface_estimate.frame_values,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            loaded.surface_estimate.confidence,
            profile.surface_estimate.confidence,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            loaded.surface_estimate.provenance.astype(str),
            profile.surface_estimate.provenance.astype(str),
        )
    assert loaded.species == "O"
    assert loaded.axis == "z"


def test_save_density_profile_writes_nested_surface_metadata(tmp_path):
    frame1 = Atoms(
        "PtPtPtH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 1.5],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    frame2 = Atoms(
        "PtPtPtH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 1.6],
        ],
        cell=[8.0, 8.0, 8.0],
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

    out = tmp_path / "density_surface_nested.h5"
    save_density_profile(profile, out)

    with h5py.File(out, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
    assert "surface" in metadata
    assert "surface_position" not in metadata
    assert "surface_position_std" not in metadata
    assert metadata["surface"]["position"] == pytest.approx(profile.surface_position)
    assert metadata["surface"]["mode"] == profile.surface_estimate.mode

    loaded = load_density_profile(out, axis="z", species="H")
    assert loaded.surface_position == pytest.approx(profile.surface_position)
    assert loaded.surface_estimate is not None
    assert loaded.surface_estimate.mode == profile.surface_estimate.mode


def test_load_density_profile_rejects_incompatible_flat_surface_metadata(tmp_path):
    out = tmp_path / "old_density_surface.h5"
    with h5py.File(out, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["analysis"] = "density"
        handle.attrs["metadata_json"] = json.dumps(
            {
                "axis": "z",
                "species": "O",
                "units": "g/cm^3",
                "n_frames": 2,
                "coordinate_mode": "distance",
                "surface_position": 1.25,
                "surface_position_std": 0.05,
                "surface_mode": "rough",
                "surface_side": "top",
                "surface_method_label": "rough_low_mobility_median",
                "surface_valid_fraction": 1.0,
                "surface_median_confidence": 0.8,
                "surface_composite_score": 0.75,
            }
        )
        handle.create_dataset("bin_centers_A", data=np.array([0.5, 1.5], dtype=float))
        handle.create_dataset("density", data=np.array([0.1, 0.2], dtype=float))
        handle.create_dataset(
            "surface_position_per_frame_A", data=np.array([1.2, 1.3], dtype=float)
        )
        handle.create_dataset("surface_valid_mask", data=np.array([True, True], dtype=bool))
        handle.create_dataset("surface_confidence", data=np.array([0.8, 0.8], dtype=float))

    with pytest.raises(ValueError, match="corrupted or originates from the wrong LiNaK version"):
        load_density_profile(out, axis="z", species="O")


def test_save_and_load_density_profile_preserves_molecular_number_density_units(tmp_path):
    frame = Atoms(
        "OHHOHH",
        positions=[
            [0.0, 0.0, 0.10],
            [0.95, 0.0, 0.10],
            [-0.30, 0.90, 0.10],
            [0.0, 0.0, 1.10],
            [0.95, 0.0, 1.10],
            [-0.30, 0.90, 1.10],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile(
        frames=[frame],
        species="H2O",
        axis="z",
        bin_width=1.0,
    )
    out = tmp_path / "density_h2o.h5"

    save_density_profile(profile, out)
    loaded = load_density_profile(out, axis="z", species="H2O")

    np.testing.assert_allclose(loaded.number_density, profile.number_density)
    assert loaded.number_density_units == "molecule/nm^3"


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


def test_load_density_profile_rejects_missing_v1_bin_width_metadata(tmp_path):
    out = tmp_path / "old_density.h5"
    with h5py.File(out, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["linak_format_version"] = 1
        handle.attrs["linak_version"] = LINAK_VERSION
        handle.attrs["analysis"] = "density"
        handle.attrs["metadata_json"] = json.dumps(
            {
                "analysis": "density",
                "analysis_schema_version": 1,
                "profile_uid": "density-without-bin-width",
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

    with pytest.raises(ValueError, match="missing required v1 metadata bin_width_A"):
        load_density_profile(out, axis="z", species="O")


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

    assert profile.coordinate_mode == "axis"
    assert profile.surface_position is not None
    assert profile.surface_estimate is not None
    assert profile.surface_estimate.summary.valid_fraction < 1.0
    assert "quantile_fill_inconsistent" in profile.surface_estimate.diagnostics.rejection_reason


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

    caplog.set_level(logging.DEBUG, logger="linak.analysis.density")
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
        "Surface estimator along Z: layered top-layer median on Z using Pt reference atoms"
        in caplog.text
    )
    assert "frame-wise surface alignment was unavailable" in caplog.text


def test_estimate_surface_reference_exposes_provenance_and_confidence():
    frames = [
        Atoms(
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
        ),
        Atoms(
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
        ),
        Atoms(
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
        ),
    ]
    estimate = estimate_surface_reference(
        frames,
        axis="z",
        mode="auto",
        surface_elements=["Pt"],
    )

    assert estimate is not None
    assert estimate.confidence.shape == (3,)
    assert estimate.provenance.shape == (3,)
    assert np.all((estimate.confidence >= 0.0) & (estimate.confidence <= 1.0))
    assert estimate.summary.method_label in {
        "layered_top_layer_median(median_layers=2)",
        "rough_low_mobility_median",
    }


def test_load_density_profile_rejects_csv_input(tmp_path):
    csv = tmp_path / "old_density.csv"
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

    def _fake_plot_line_series(x, y, **kwargs):
        captured["x"] = x
        captured["x_label"] = kwargs["x_label"]
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
    assert captured["y_label"] == "Entity density (atom/nm^3)"


def test_plot_density_profile_supports_explicit_cartesian_x_modes(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profile

    captured = {}

    def _fake_plot_line_series(x, _y, **kwargs):
        captured["x"] = x
        captured["x_label"] = kwargs["x_label"]
        return None

    monkeypatch.setattr("linak.analysis.density.plot_line_series", _fake_plot_line_series)
    profile = DensityProfile(
        axis="x",
        species="Li",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        counts_per_frame=np.array([1.0, 2.0]),
        density=np.array([0.1, 0.2]),
        units="g/cm^3",
        n_frames=1,
        surface_position=0.0,
    )

    plot_density_profile(profile, show=False, x_mode="x")

    np.testing.assert_allclose(captured["x"], np.array([0.5, 1.5]))
    assert captured["x_label"] == "X (A)"


def test_plot_density_profiles_explicit_cartesian_x_mode_filters_to_matching_axis(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured = {}

    def _fake_plot_multi_line_series(x_series, y_series, labels, **kwargs):
        captured["series_count"] = len(x_series)
        captured["labels"] = labels
        return None

    monkeypatch.setattr("linak.analysis.density.plot_multi_line_series", _fake_plot_multi_line_series)

    profiles = [
        DensityProfile(
            axis="x",
            species="O",
            bin_edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            counts_per_frame=np.array([1.0]),
            density=np.array([0.1]),
            units="g/cm^3",
            n_frames=1,
        ),
        DensityProfile(
            axis="y",
            species="O",
            bin_edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            counts_per_frame=np.array([1.0]),
            density=np.array([0.2]),
            units="g/cm^3",
            n_frames=1,
        ),
    ]

    plot_density_profiles(profiles, show=False, x_mode="y")

    assert captured["series_count"] == 1
    assert captured["labels"] == ["O"]


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

    assert captured_kwargs["x_lim"] is None
    assert captured_kwargs["y_lim"] is None


def test_plot_density_profiles_gui_overrides_skip_wrapper_auto_limits(monkeypatch):
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
        render_series_descriptors=[
            {
                "series_id": "series:0",
                "default_label": "Au",
                "source_kind": "source",
                "source_series_id": "series:0",
            },
            {
                "series_id": "series:1",
                "default_label": "H2O",
                "source_kind": "source",
                "source_series_id": "series:1",
            },
        ],
        series_overrides_by_id={
            "series:0": {"normalization_mode": "max", "normalization_value": 1.0}
        },
    )

    assert captured_kwargs["x_lim"] is None
    assert captured_kwargs["y_lim"] is None


def test_plot_density_profiles_autoscale_to_visible_normalized_data(tmp_path):
    from linak.analysis.density import DensityProfile, plot_density_profiles

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

    capture_state: dict[str, object] = {}
    plot_density_profiles(
        [profile_a, profile_b],
        output=tmp_path / "normalized_density.png",
        show=False,
        series_normalization_modes=["max", "max"],
        series_normalization_values=[1.0, 1.0],
        series_normalization_x_refs=[None, None],
        capture_state=capture_state,
    )

    ax = capture_state["axes"]
    bottom, top = ax.get_ylim()
    assert bottom >= -0.2
    assert top <= 1.2


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


def test_compute_density_profile_logs_single_run_summary_at_info_and_details_at_debug(caplog):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    with caplog.at_level(logging.INFO, logger="linak.analysis.density"):
        compute_density_profile(
            frames=[frame],
            species="O",
            axis="z",
            bin_width=1.0,
        )

    info_messages = [
        record.getMessage() for record in caplog.records if record.levelno == logging.INFO
    ]
    assert any("Density compute summary:" in message for message in info_messages)
    assert not any("Selected " in message for message in info_messages)
    assert not any("Density mode:" in message for message in info_messages)
    assert not any("Density normalization path" in message for message in info_messages)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="linak.analysis.density"):
        compute_density_profile(
            frames=[frame],
            species="O",
            axis="z",
            bin_width=1.0,
        )

    debug_messages = [record.getMessage() for record in caplog.records]
    assert any("Selected " in message for message in debug_messages)
    assert any("Density mode:" in message for message in debug_messages)
    assert any("Density normalization path" in message for message in debug_messages)


def test_compute_density_profiles_logs_one_compact_info_summary(caplog):
    frame = Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.50],
            [0.8, 0.0, 0.55],
            [-0.2, 0.75, 0.45],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    with caplog.at_level(logging.INFO, logger="linak.analysis.density"):
        profiles = compute_density_profiles(
            frames=[frame],
            species="all",
            axis="z",
            bin_width=1.0,
        )

    assert [profile.species for profile in profiles] == ["H", "O", "H2O"]
    info_messages = [
        record.getMessage() for record in caplog.records if record.levelno == logging.INFO
    ]
    summary_messages = [
        message for message in info_messages if "Density compute summary:" in message
    ]
    assert len(summary_messages) == 1
    assert "3 profile(s): H, O, H2O;" in summary_messages[0]
    assert any("Binning 3 density profiles." in message for message in info_messages)
    assert not any("Selected " in message for message in info_messages)
    assert not any("Density mode:" in message for message in info_messages)
    assert not any("Density normalization path" in message for message in info_messages)


def test_compute_all_density_profiles_logs_single_pass_and_aggregate_binning(caplog):
    frame = Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.50],
            [0.8, 0.0, 0.55],
            [-0.2, 0.75, 0.45],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    with caplog.at_level(logging.INFO, logger="linak.analysis.density"):
        profiles = compute_all_density_profiles(
            frames=[frame],
            species="all",
            surface_axis="z",
            bin_width=1.0,
        )

    assert len(profiles) == 12
    info_messages = [
        record.getMessage() for record in caplog.records if record.levelno == logging.INFO
    ]
    assert any("Single-pass selection complete:" in message for message in info_messages)
    assert any("Binning 12 density profiles." in message for message in info_messages)


def test_unify_number_density_units_merges_atom_and_molecule():
    from linak.analysis.density import _unify_number_density_units

    result = _unify_number_density_units(["atom/nm^3", "molecule/nm^3"])
    assert result == "entities/nm^3"


def test_unify_number_density_units_returns_none_for_incompatible():
    from linak.analysis.density import _unify_number_density_units

    assert _unify_number_density_units(["atom/nm^3", "g/cm^3"]) is None


def test_unify_number_density_units_returns_none_for_different_denominators():
    from linak.analysis.density import _unify_number_density_units

    assert _unify_number_density_units(["atom/nm^3", "molecule/Angstrom^3"]) is None


def test_unify_number_density_units_single_unit():
    from linak.analysis.density import _unify_number_density_units

    assert _unify_number_density_units(["atom/nm^3"]) == "entities/nm^3"


def test_plot_density_profiles_filters_by_axis_in_x_mode(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured = {}

    def _fake_plot_multi(x_series, y_series, labels, **kwargs):
        captured["labels"] = labels
        captured["x_series"] = x_series
        return None

    monkeypatch.setattr("linak.analysis.density.plot_multi_line_series", _fake_plot_multi)

    profiles = [
        DensityProfile(
            axis="x",
            species="O",
            bin_edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            counts_per_frame=np.array([1.0]),
            density=np.array([1.0]),
            units="g/cm^3",
            n_frames=1,
        ),
        DensityProfile(
            axis="y",
            species="O",
            bin_edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            counts_per_frame=np.array([1.0]),
            density=np.array([1.2]),
            units="g/cm^3",
            n_frames=1,
        ),
        DensityProfile(
            axis="z",
            species="O",
            bin_edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            counts_per_frame=np.array([1.0]),
            density=np.array([0.8]),
            units="g/cm^3",
            n_frames=1,
        ),
    ]

    plot_density_profiles(profiles, show=False, x_mode="z", quantity="mass")
    assert len(captured["labels"]) == 1
    assert captured["labels"][0] == "O"


def test_plot_density_profiles_unifies_number_density_units_across_atom_and_molecule(monkeypatch):
    from linak.analysis.density import DensityProfile, plot_density_profiles

    captured = {}

    def _fake_plot_multi(x_series, y_series, labels, **kwargs):
        captured["y_label"] = kwargs["y_label"]
        captured["y_series"] = y_series
        return None

    monkeypatch.setattr("linak.analysis.density.plot_multi_line_series", _fake_plot_multi)

    profiles = [
        DensityProfile(
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
        ),
        DensityProfile(
            axis="z",
            species="H2O",
            bin_edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            counts_per_frame=np.array([1.0]),
            density=np.array([0.5]),
            units="g/cm^3",
            n_frames=1,
            number_density=np.array([0.10]),
            number_density_units="molecules/Angstrom^3",
            surface_position=0.0,
        ),
    ]

    plot_density_profiles(profiles, show=False, x_mode="distance", quantity="number")
    assert "entities" in captured["y_label"]
