"""Scientific and persistence checks for dynamic molecular pair selections."""

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses

from linak.analysis.coordination import (
    CoordinationCutoffResolution,
    compute_coordination_profile,
    compute_coordination_profiles,
    load_coordination_profile,
    save_coordination_profile,
    resolve_coordination_cutoff,
)
from linak.analysis.rdf import (
    compute_rdf,
    compute_rdf_profiles,
    load_rdf_profile,
    save_rdf_profile,
)


def changing_frames():
    frame = Atoms(
        "NaOHO", positions=[[4, 1, 1], [1, 1, 1], [2, 1, 1], [8, 1, 1]], cell=[12] * 3, pbc=True
    )
    empty = frame.copy()
    empty.positions[2] = [5, 5, 5]
    switched = frame.copy()
    switched.positions[2] = [7, 1, 1]
    return [frame, empty, switched, frame.copy()]


def test_dynamic_rdf_normalization_and_empty_frame_statistics():
    frames = changing_frames()
    result = compute_rdf(frames, "Na", "OH", r_max=5, bin_width=0.25)
    mass_fraction = atomic_masses[1] / (atomic_masses[8] + atomic_masses[1])
    distances = [3 - mass_fraction, 4 - mass_fraction, 3 - mass_fraction]
    counts, _ = np.histogram(distances, result.bin_edges)
    shells = 4 * np.pi / 3 * np.diff(result.bin_edges**3)
    np.testing.assert_allclose(result.g_r, counts / (3 * shells / 12**3))
    np.testing.assert_array_equal(result.series_statistics["g_r"].sample_n, 3)
    assert result.species_b == "mol:OH"


@pytest.mark.parametrize("cell", [[12, 12, 12], [[12, 0, 0], [2, 12, 0], [1, 2, 12]]])
def test_boundary_crossing_com_and_member_exclusion(cell):
    frame = Atoms("NaOH", positions=[[2, 1, 1], [11.8, 1, 1], [0.8, 1, 1]], cell=cell, pbc=True)
    result = compute_rdf([frame], "Na", "mol:OH", r_max=5, bin_width=0.1)
    fraction = atomic_masses[1] / (atomic_masses[8] + atomic_masses[1])
    expected_bin = np.searchsorted(result.bin_edges, 2.2 - fraction, side="right") - 1
    assert np.flatnonzero(result.g_r).tolist() == [expected_bin]
    for pair in [("O", "mol:OH"), ("mol:OH", "mol:OH"), ("atoms[1]", "mol:OH")]:
        with pytest.raises(ValueError, match="no eligible pairs"):
            compute_rdf_profiles([frame], pairs=[pair], r_max=5)


def test_molecular_coordination_anchor_tracking_and_empty_neighbors():
    frames = changing_frames()
    cutoff = CoordinationCutoffResolution(5, 0, "direct")
    pairs = [("mol:OH", "Na"), ("Na", "mol:OH"), ("mol:OH", "O")]
    profiles = compute_coordination_profiles(
        frames,
        ordered_pairs=pairs,
        cutoff_resolutions=dict.fromkeys(pairs, cutoff),
        surface_mode="none",
    )
    molecular, atomic, internal = profiles
    np.testing.assert_array_equal(molecular.atom_indices, [1, 3])
    np.testing.assert_allclose(
        molecular.coordination_number,
        [[1, np.nan], [np.nan, np.nan], [np.nan, 1], [1, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(atomic.coordination_number[:, 0], [1, 0, 1, 1])
    # The own oxygen is excluded. The other oxygen is 7 A away, beyond cutoff.
    np.testing.assert_allclose(
        internal.coordination_number, molecular.coordination_number * 0, equal_nan=True
    )
    assert np.isnan(molecular.axis_coordinate[1]).all()
    assert np.isfinite(molecular.axis_coordinate[[0, 2, 3]]).sum() == 3


def test_molecular_roundtrip_and_reversed_rdf(tmp_path):
    frames = changing_frames()
    profile = compute_rdf(frames, "Na", "mol:OH", r_max=5)
    output = save_rdf_profile(profile, tmp_path / "molecules.rdf.h5")
    reverse = load_rdf_profile(output, species_a="OH", species_b="Na")
    assert reverse.selection_kind_a == "molecule"
    assert reverse.molecular_metadata["entity_kind_a"] == "molecule"
    np.testing.assert_allclose(reverse.g_r, profile.g_r)
    cn = compute_coordination_profile(
        frames,
        species_a="mol:OH",
        species_b="Na",
        surface_mode="none",
        cutoff_resolution=CoordinationCutoffResolution(5, 0, "direct"),
    )
    path = save_coordination_profile(cn, tmp_path / "molecules.coordination.h5")
    restored = load_coordination_profile(path)
    np.testing.assert_array_equal(restored.atom_indices, [1, 3])
    np.testing.assert_allclose(restored.coordination_number, cn.coordination_number, equal_nan=True)
    assert restored.molecular_metadata == cn.molecular_metadata
    with pytest.raises(ValueError, match="incompatible molecular selection metadata"):
        resolve_coordination_cutoff(
            frames=frames,
            species_a="Na",
            species_b="mol:OH",
            cutoff_A=None,
            cutoff_rdf_path=output,
            cutoff_from_rdf=False,
            oh_cutoff=1.1,
        )


@pytest.mark.parametrize("selector", ["mol:H", "mol:O", "mol:OH", "mol:H2O", "mol:H3O"])
def test_all_molecular_selectors(selector):
    hydrogen_count = {"mol:H": 1, "mol:O": 0, "mol:OH": 1, "mol:H2O": 2, "mol:H3O": 3}[selector]
    positions = [[4, 1, 1], [1, 1, 1]]
    positions += [[1.9, 1, 1], [1, 1.9, 1], [1, 1, 1.9]][:hydrogen_count]
    if selector == "mol:H":
        positions[2] = [6, 1, 1]
    frame = Atoms("NaO" + "H" * hydrogen_count, positions=positions, cell=[12] * 3, pbc=True)
    rdf = compute_rdf([frame], "Na", selector, r_max=5)
    assert rdf.selection_kind_b == "molecule"
    assert np.count_nonzero(rdf.g_r) == 1
    cn = compute_coordination_profile(
        [frame],
        species_a=selector,
        species_b="Na",
        surface_mode="none",
        cutoff_resolution=CoordinationCutoffResolution(5, 0, "direct"),
    )
    np.testing.assert_allclose(cn.coordination_number, [[1]])
    assert cn.atom_indices.tolist() == [2 if selector == "mol:H" else 1]


def test_molecular_cutoff_from_computed_and_saved_rdf(tmp_path):
    frames = changing_frames()
    profile = compute_rdf(frames, "Na", "mol:OH")
    path = save_rdf_profile(profile, tmp_path / "reference.rdf.h5")
    resolutions = [
        resolve_coordination_cutoff(
            frames=frames,
            species_a="Na",
            species_b="mol:OH",
            cutoff_A=None,
            cutoff_rdf_path=source,
            cutoff_from_rdf=source is None,
        )
        for source in (None, path)
    ]
    assert resolutions[0].cutoff_A == pytest.approx(resolutions[1].cutoff_A, abs=1e-6)
    assert resolutions[0].rdf_peak_A < resolutions[0].cutoff_A


def test_molecule_same_selection_ordered_pairs():
    frame = Atoms(
        "OHOH", positions=[[1, 1, 1], [2, 1, 1], [4.1, 1, 1], [5.1, 1, 1]], cell=[12] * 3, pbc=True
    )
    profile = compute_rdf([frame], "mol:OH", "OH", r_max=5, bin_width=0.25)
    shells = 4 * np.pi / 3 * np.diff(profile.bin_edges**3)
    expected, _ = np.histogram([3.1, 3.1], profile.bin_edges)
    np.testing.assert_allclose(profile.g_r, expected / (2 * shells / 12**3))
    coordination = compute_coordination_profile(
        [frame],
        species_a="mol:OH",
        species_b="OH",
        surface_mode="none",
        cutoff_resolution=CoordinationCutoffResolution(3.1, 1, "direct"),
    )
    np.testing.assert_allclose(coordination.coordination_number, [[0.5, 0.5]])


def test_mixed_indices_metadata_and_raw_labels():
    frames = changing_frames()
    for frame in frames:
        frame.new_array("linak_raw_species", np.array(["cation", "OH", "H", "O"]))
    profile = compute_rdf(frames, species_b="mol:OH", atom_indices_a=[0], r_max=5)
    assert profile.selection_kind_a == "atoms"
    np.testing.assert_array_equal(profile.atom_indices_a, [0])
    dynamic = compute_rdf(frames, "Na", "mol:OH", r_max=5)
    raw = compute_rdf(frames, "Na", "species:OH", r_max=5)
    assert raw.series_statistics["g_r"].sample_n[0] == 4
    assert dynamic.series_statistics["g_r"].sample_n[0] == 3


def test_collection_dedup_and_shared_frame_resolution(monkeypatch):
    from linak.analysis.entities import EntityResolver

    calls = []
    original = EntityResolver.select

    def select(self, frame, labels, frame_index=0):
        calls.append(frame_index)
        return original(self, frame, labels, frame_index)

    monkeypatch.setattr(EntityResolver, "select", select)
    frames = changing_frames()
    profiles = compute_rdf_profiles(
        frames, pairs=[("Na", "OH"), ("mol:OH", "Na"), ("Na", "mol:H")], r_max=5
    )
    assert len(profiles) == 2
    assert calls == list(range(4))
    calls.clear()
    pairs = [("Na", "mol:OH"), ("mol:OH", "Na")]
    compute_coordination_profiles(
        frames,
        ordered_pairs=pairs,
        surface_mode="none",
        cutoff_resolutions=dict.fromkeys(pairs, CoordinationCutoffResolution(5, 0, "direct")),
    )
    assert calls == list(range(4))


def test_molecular_surface_alignment_and_plots(tmp_path):
    from linak.analysis.rdf import plot_rdf_profile
    from linak.analysis.coordination import plot_coordination_profile
    from linak.plot.contracts.coordination_contract import (
        coordination_profile_to_plot_data_contract,
    )

    frames = changing_frames()
    for frame in frames:
        frame += Atoms("Pt4", positions=[[0, 0, 0], [6, 0, 0], [0, 6, 0], [6, 6, 0]])
        frame.set_cell([12] * 3)
        frame.pbc = True
    cn = compute_coordination_profile(
        frames,
        species_a="mol:OH",
        species_b="Na",
        surface_elements=["Pt"],
        surface_layout="single-top",
        cutoff_resolution=CoordinationCutoffResolution(5, 0, "direct"),
    )
    assert cn.surface_reference is not None
    np.testing.assert_array_equal(
        np.isfinite(cn.distance_to_surface), np.isfinite(cn.coordination_number)
    )
    contract = coordination_profile_to_plot_data_contract(cn)
    assert next(d for d in contract.dimensions if d.id == "atom").label == "Molecule"
    cn_path = tmp_path / "molecular_cn.png"
    rdf_path = tmp_path / "molecular_rdf.png"
    plot_coordination_profile(cn, output=cn_path, show=False, preferred_backend="Agg")
    plot_rdf_profile(
        compute_rdf(frames, "Na", "mol:OH", r_max=5),
        output=rdf_path,
        show=False,
        preferred_backend="Agg",
    )
    assert cn_path.exists() and rdf_path.exists()


def test_molecular_collection_metadata_compatibility(tmp_path):
    from linak.analysis.rdf import save_rdf_profiles

    path = tmp_path / "merged.rdf.h5"
    first = compute_rdf(changing_frames(), "Na", "mol:OH", r_max=5)
    save_rdf_profiles([first], path, force_collection=True, merge_existing=True)
    changed = compute_rdf(changing_frames(), "Na", "mol:OH", r_max=5, oh_cutoff=1.1)
    written = save_rdf_profiles([changed], path, force_collection=True, merge_existing=True)
    assert written != path
    assert load_rdf_profile(path).molecular_metadata["oh_cutoff_A"] == 1.27
    assert load_rdf_profile(written).molecular_metadata["oh_cutoff_A"] == 1.1


@pytest.mark.parametrize("backend", ["auto", "ckdtree", "ase"])
def test_classification_options_and_no_molecules(backend):
    frames = changing_frames()
    profile = compute_rdf(frames, "Na", "mol:OH", oh_topology_backend=backend, r_max=5)
    np.testing.assert_allclose(profile.g_r, compute_rdf(frames, "Na", "mol:OH", r_max=5).g_r)
    with pytest.raises(ValueError, match="no eligible pairs"):
        compute_rdf(frames, "Na", "mol:OH", r_max=5, oh_cutoff=0.8, oh_topology_backend=backend)
    with pytest.raises(ValueError, match="No selected molecular centers"):
        compute_coordination_profile(
            frames,
            species_a="mol:OH",
            species_b="Na",
            oh_cutoff=0.8,
            oh_topology_backend=backend,
            surface_mode="none",
            cutoff_resolution=CoordinationCutoffResolution(5, 0, "direct"),
        )


def test_mixed_collection_preserves_atom_results():
    frames = changing_frames()
    profiles = compute_rdf_profiles(frames, pairs=[("Na", "OH"), ("Na", "O")], r_max=5)
    atomic = next(profile for profile in profiles if profile.species_b == "O")
    np.testing.assert_allclose(atomic.g_r, compute_rdf(frames, "Na", "O", r_max=5).g_r)
