from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from ase import Atoms

import linak.analysis.rdf as rdf_mod
from linak import __version__ as LINAK_VERSION
from linak.analysis.rdf import (
    RDFProfile,
    compute_rdf,
    compute_rdf_profiles,
    load_rdf_profile,
    load_rdf_profiles,
    plot_rdf_profile,
    save_rdf_profile,
    save_rdf_profiles,
)
from linak.storage.hdf5_utils import read_linak_hdf5_profiles, write_linak_hdf5_profile_collection


def _histogram_index(profile, value: float) -> int:
    return int(np.searchsorted(profile.bin_edges, value, side="right") - 1)


def _frame_contribution(
    frame: Atoms,
    *,
    species_a: str,
    species_b: str,
    r_max: float,
    bin_width: float,
    strategy: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_edges, _effective_bin_width = rdf_mod._build_uniform_rdf_bins(
        r_max=r_max,
        target_bin_width=bin_width,
    )
    shell_volumes = rdf_mod._shell_volumes_from_edges(bin_edges)
    counts, expected = rdf_mod._compute_rdf_frame_contribution(
        0,
        frame,
        label_a=species_a,
        label_b=species_b,
        same_selection=species_a == species_b,
        r_max=r_max,
        bin_edges=bin_edges,
        shell_volumes=shell_volumes,
        max_sphere_volume=(4.0 / 3.0) * np.pi * (r_max**3),
        strategy_override=strategy,
    )
    return counts, expected, bin_edges


def test_compute_rdf_returns_expected_shape_and_values():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_rdf(
        [frame],
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=1.0,
    )

    np.testing.assert_allclose(profile.bin_edges, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(profile.bin_centers, np.array([0.5, 1.5]))
    assert np.isclose(profile.g_r[0], 0.0)
    assert profile.g_r[1] > 0.0
    assert profile.species_a == "O"
    assert profile.species_b == "H"


def test_compute_rdf_requires_fully_periodic_nonzero_cell():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, True, False],
    )
    with pytest.raises(ValueError, match="fully periodic boundary conditions"):
        compute_rdf([frame], species_a="O", species_b="H", r_max=2.0, bin_width=1.0)

    zero_volume = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=True,
    )
    with pytest.raises(ValueError, match="non-zero cell volume"):
        compute_rdf([zero_volume], species_a="O", species_b="H", r_max=2.0, bin_width=1.0)


def test_shell_volume_formula_matches_analytical_values():
    edges = np.array([0.0, 1.0, 2.0], dtype=float)
    shell_volumes = rdf_mod._shell_volumes_from_edges(edges)
    expected = (4.0 / 3.0) * np.pi * np.array([1.0**3 - 0.0**3, 2.0**3 - 1.0**3], dtype=float)
    np.testing.assert_allclose(shell_volumes, expected)


def test_compute_rdf_uses_uniform_bins_when_r_max_is_not_multiple_of_bin_width(tmp_path):
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_rdf(
        [frame],
        species_a="O",
        species_b="H",
        r_max=2.3,
        bin_width=1.0,
    )

    widths = np.diff(profile.bin_edges)
    assert np.allclose(widths, widths[0])
    assert profile.bin_edges[-1] == pytest.approx(2.3)

    out = tmp_path / "rdf_non_multiple.h5"
    save_rdf_profile(profile, out)
    loaded = load_rdf_profile(out, species_a="O", species_b="H")
    np.testing.assert_allclose(loaded.bin_edges, profile.bin_edges)


def test_histogram_edge_handling_is_deterministic():
    frame = Atoms(
        "OHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    counts, _expected, bin_edges = _frame_contribution(
        frame,
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=1.0,
    )

    np.testing.assert_array_equal(bin_edges, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(counts, np.array([1.0, 2.0]))


def test_same_species_self_pairs_are_excluded_but_ordered_pairs_remain():
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    counts, expected, _bin_edges = _frame_contribution(
        frame,
        species_a="O",
        species_b="O",
        r_max=2.0,
        bin_width=1.0,
    )

    np.testing.assert_array_equal(counts, np.array([0.0, 2.0]))
    expected_second = 2.0 * ((2.0 - 1.0) / 1000.0) * ((4.0 / 3.0) * np.pi * (2.0**3 - 1.0**3))
    assert expected[1] == pytest.approx(expected_second)


def test_single_atom_same_species_yields_nan_when_expected_count_is_zero():
    frame = Atoms(
        "O",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_rdf([frame], species_a="O", species_b="O", r_max=2.0, bin_width=1.0)
    assert np.all(np.isnan(profile.g_r))


def test_observed_pair_count_matches_direct_enumeration_for_small_cross_species_system():
    frame = Atoms(
        "OHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.9, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    counts, _expected, _bin_edges = _frame_contribution(
        frame,
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=1.0,
    )
    np.testing.assert_array_equal(counts, np.array([1.0, 2.0]))
    assert np.sum(counts) == pytest.approx(3.0)


def test_strategy_override_produces_equivalent_rdf_results():
    frames = [
        Atoms(
            "OOHH",
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [2.8, 0.0, 0.0],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        ),
        Atoms(
            "OOHH",
            positions=[
                [0.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [2.9, 0.0, 0.0],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        ),
    ]
    profiles = {
        strategy: compute_rdf(
            frames,
            species_a="O",
            species_b="H",
            r_max=4.0,
            bin_width=0.5,
            threads=1,
            _strategy_override=strategy,
        )
        for strategy in ("neighbor_list", "selected_matrix", "full_matrix")
    }

    reference = profiles["neighbor_list"]
    for strategy in ("selected_matrix", "full_matrix"):
        np.testing.assert_allclose(profiles[strategy].bin_edges, reference.bin_edges)
        np.testing.assert_allclose(profiles[strategy].g_r, reference.g_r, equal_nan=True)


def test_dense_orthorhombic_backend_matches_generic_fallback():
    frames = [
        Atoms(
            "OOHH",
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.2, 0.0],
                [0.8, 0.0, 0.1],
                [2.8, 0.1, 0.1],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        ),
        Atoms(
            "OOHH",
            positions=[
                [0.1, 0.0, 0.0],
                [2.2, 0.2, 0.0],
                [0.9, 0.0, 0.1],
                [2.9, 0.1, 0.1],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        ),
    ]

    generic = compute_rdf(
        frames,
        species_a="O",
        species_b="H",
        r_max=4.0,
        bin_width=0.5,
        threads=1,
        _strategy_override="framewise_generic_fallback",
    )
    dense = compute_rdf(
        frames,
        species_a="O",
        species_b="H",
        r_max=4.0,
        bin_width=0.5,
        threads=1,
        _strategy_override="chunked_dense_matrix",
    )

    np.testing.assert_allclose(dense.bin_edges, generic.bin_edges)
    np.testing.assert_allclose(dense.g_r, generic.g_r, equal_nan=True)


def test_sparse_orthorhombic_backend_matches_generic_fallback():
    frames = [
        Atoms(
            "OOOHHH",
            positions=[
                [0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [16.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [8.8, 0.0, 0.0],
                [16.7, 0.0, 0.0],
            ],
            cell=[40.0, 40.0, 40.0],
            pbc=True,
        ),
        Atoms(
            "OOOHHH",
            positions=[
                [0.1, 0.0, 0.0],
                [8.1, 0.0, 0.0],
                [16.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [8.9, 0.0, 0.0],
                [16.8, 0.0, 0.0],
            ],
            cell=[40.0, 40.0, 40.0],
            pbc=True,
        ),
    ]

    generic = compute_rdf(
        frames,
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=0.25,
        threads=1,
        _strategy_override="framewise_generic_fallback",
    )
    sparse = compute_rdf(
        frames,
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=0.25,
        threads=1,
        _strategy_override="chunked_sparse_cutoff",
    )

    np.testing.assert_allclose(sparse.bin_edges, generic.bin_edges)
    np.testing.assert_allclose(sparse.g_r, generic.g_r, equal_nan=True)


def test_variable_volume_accumulates_expected_counts_framewise():
    frames = [
        Atoms(
            "OH", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True
        ),
        Atoms(
            "OH", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True
        ),
    ]
    profile = compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=1)

    occupied = _histogram_index(profile, 1.5)
    shell = (4.0 / 3.0) * np.pi * (2.0**3 - 1.0**3)
    expected = shell * ((1.0 / 1000.0) + (1.0 / 8000.0))
    assert profile.g_r[occupied] == pytest.approx(2.0 / expected)


def test_variable_selection_counts_are_resolved_per_frame():
    frames = [
        Atoms(
            "OH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True
        ),
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    profile = compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=1)

    occupied = _histogram_index(profile, 1.5)
    shell = (4.0 / 3.0) * np.pi * (2.0**3 - 1.0**3)
    expected = shell * ((1.0 / 1000.0) + (2.0 / 1000.0))
    assert profile.g_r[occupied] == pytest.approx(3.0 / expected)


def test_auto_r_max_matches_half_min_length_for_orthorhombic_cells():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 20.0, 30.0],
        pbc=True,
    )
    profile = compute_rdf([frame], species_a="O", species_b="H", r_max=None, bin_width=1.0)
    assert profile.bin_edges[-1] == pytest.approx(5.0)


def test_auto_r_max_rounds_down_to_requested_bin_width():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 20.0, 30.0],
        pbc=True,
    )
    profile = compute_rdf([frame], species_a="O", species_b="H", r_max=None, bin_width=0.6)

    widths = np.diff(profile.bin_edges)
    assert np.allclose(widths, 0.6)
    assert profile.bin_edges[-1] == pytest.approx(4.8)


def test_auto_r_max_uses_half_min_perpendicular_cell_height_for_skewed_cells():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[[10.0, 0.0, 0.0], [2.0, 8.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=True,
    )
    profile = compute_rdf([frame], species_a="O", species_b="H", r_max=None, bin_width=1.0)
    expected_auto = rdf_mod._auto_r_max_from_frames([frame])
    expected_rounded = rdf_mod._resolve_auto_r_max_for_bin_width(
        auto_r_max=expected_auto,
        target_bin_width=1.0,
    )
    assert profile.bin_edges[-1] == pytest.approx(expected_rounded)


def test_large_r_limit_is_near_unity_for_random_ideal_gas_like_distribution():
    rng = np.random.default_rng(123)
    frames = []
    for _ in range(30):
        positions = rng.uniform(0.0, 20.0, size=(64, 3))
        frames.append(Atoms("Ar64", positions=positions, cell=[20.0, 20.0, 20.0], pbc=True))
    profile = compute_rdf(
        frames, species_a="Ar", species_b="Ar", r_max=5.0, bin_width=0.5, threads=1
    )

    reliable = profile.g_r[3:-2]
    reliable = reliable[np.isfinite(reliable)]
    assert reliable.size > 0
    assert float(np.mean(np.abs(reliable - 1.0))) < 0.35


def test_save_and_load_rdf_profile(tmp_path):
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_rdf(
        [frame],
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=1.0,
    )
    out = tmp_path / "rdf.h5"

    save_rdf_profile(profile, out)
    with h5py.File(out, "r") as handle:
        assert "created_utc" in handle.attrs
        assert "linak_version" in handle.attrs
        assert "bin_edges_A" not in handle
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["bin_width_A"] == pytest.approx(1.0)
    loaded = load_rdf_profile(out, species_a="O", species_b="H")

    np.testing.assert_allclose(loaded.bin_edges, profile.bin_edges)
    np.testing.assert_allclose(loaded.bin_centers, profile.bin_centers)
    np.testing.assert_allclose(loaded.g_r, profile.g_r, equal_nan=True)
    assert loaded.species_a == "O"
    assert loaded.species_b == "H"
    assert loaded.series_statistics is not None
    stats = loaded.series_statistics["g_r"]
    np.testing.assert_array_equal(stats.point_count, np.array([0, 1]))
    np.testing.assert_array_equal(stats.sample_n, np.array([1, 1]))


def test_compute_rdf_supports_explicit_atom_indices():
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )

    explicit = compute_rdf(
        [frame],
        atom_indices_a=[0],
        atom_indices_b=[1, 2],
        r_max=3.0,
        bin_width=1.0,
        threads=1,
    )
    species = compute_rdf(
        [frame],
        species_a="O",
        species_b="H",
        r_max=3.0,
        bin_width=1.0,
        threads=1,
    )

    np.testing.assert_allclose(explicit.bin_edges, species.bin_edges)
    np.testing.assert_allclose(explicit.g_r, species.g_r, equal_nan=True)
    assert explicit.species_a == "atoms[0]"
    assert explicit.species_b == "atoms[1..2]"
    np.testing.assert_array_equal(explicit.atom_indices_a, np.array([0]))
    np.testing.assert_array_equal(explicit.atom_indices_b, np.array([1, 2]))
    assert explicit.selection_kind_a == "atoms"
    assert explicit.selection_kind_b == "atoms"


def test_compute_rdf_supports_mixed_species_and_atom_selectors():
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )

    mixed = compute_rdf(
        [frame],
        species_a="O",
        atom_indices_b=[1, 2],
        r_max=3.0,
        bin_width=1.0,
        threads=1,
    )
    species = compute_rdf(
        [frame],
        species_a="O",
        species_b="H",
        r_max=3.0,
        bin_width=1.0,
        threads=1,
    )

    np.testing.assert_allclose(mixed.g_r, species.g_r, equal_nan=True)
    assert mixed.species_a == "O"
    assert mixed.species_b == "atoms[1..2]"
    assert mixed.selection_kind_a == "species"
    assert mixed.selection_kind_b == "atoms"


def test_compute_rdf_same_explicit_atom_selection_matches_same_species_behavior():
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    explicit = compute_rdf(
        [frame],
        atom_indices_a=[0, 1],
        atom_indices_b=[0, 1],
        r_max=2.0,
        bin_width=1.0,
        threads=1,
    )
    species = compute_rdf(
        [frame],
        species_a="O",
        species_b="O",
        r_max=2.0,
        bin_width=1.0,
        threads=1,
    )

    np.testing.assert_allclose(explicit.g_r, species.g_r, equal_nan=True)


def test_compute_rdf_expected_counts_handle_overlapping_explicit_selections():
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    cache = rdf_mod._resolve_rdf_selection_cache(
        [frame],
        label_a="atoms[0..1]",
        label_b="atoms[1..2]",
        atom_indices_a=np.array([0, 1], dtype=int),
        atom_indices_b=np.array([1, 2], dtype=int),
    )
    assert cache is not None

    shell_volumes = rdf_mod._shell_volumes_from_edges(np.array([0.0, 1.0, 2.0, 3.0], dtype=float))
    counts, expected = rdf_mod._compute_rdf_frame_contribution(
        0,
        frame,
        label_a="atoms[0..1]",
        label_b="atoms[1..2]",
        same_selection=False,
        r_max=3.0,
        bin_edges=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        shell_volumes=shell_volumes,
        max_sphere_volume=(4.0 / 3.0) * np.pi * (3.0**3),
        selection_cache=cache,
    )

    np.testing.assert_array_equal(counts, np.array([0.0, 2.0, 1.0]))
    np.testing.assert_allclose(expected, (3.0 / frame.get_volume()) * shell_volumes)


def test_compute_rdf_explicit_atom_indices_require_stable_layout():
    frame_a = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame_b = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    with pytest.raises(ValueError, match="stable atom identities/order"):
        compute_rdf(
            [frame_a, frame_b],
            atom_indices_a=[0],
            atom_indices_b=[1],
            r_max=2.0,
            bin_width=1.0,
            threads=1,
        )


def test_save_and_load_rdf_profile_preserves_atom_selection_metadata(tmp_path):
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    profile = compute_rdf(
        [frame],
        atom_indices_a=[0],
        atom_indices_b=[1, 2],
        r_max=3.0,
        bin_width=1.0,
        threads=1,
    )
    out = tmp_path / "rdf_selected.h5"

    save_rdf_profile(profile, out)
    with h5py.File(out, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["selection_kind_a"] == "atoms"
        assert metadata["selection_kind_b"] == "atoms"
        np.testing.assert_array_equal(handle["atom_indices_a"][...], np.array([0]))
        np.testing.assert_array_equal(handle["atom_indices_b"][...], np.array([1, 2]))

    loaded = load_rdf_profile(out)

    assert loaded.selection_kind_a == "atoms"
    assert loaded.selection_kind_b == "atoms"
    np.testing.assert_array_equal(loaded.atom_indices_a, np.array([0]))
    np.testing.assert_array_equal(loaded.atom_indices_b, np.array([1, 2]))
    assert loaded.species_a == "atoms[0]"
    assert loaded.species_b == "atoms[1..2]"


def test_atom_selected_rdf_hdf5_still_plots(tmp_path):
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    profile = compute_rdf(
        [frame],
        atom_indices_a=[0],
        atom_indices_b=[1, 2],
        r_max=3.0,
        bin_width=1.0,
        threads=1,
    )
    source = tmp_path / "rdf_selected.h5"
    output = tmp_path / "rdf_selected.png"
    save_rdf_profile(profile, source)

    loaded = load_rdf_profile(source)
    plot_rdf_profile(loaded, output=output, show=False)

    assert output.exists()


def test_compute_rdf_profiles_returns_unique_unordered_species_pairs():
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OHH",
            positions=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    profiles = compute_rdf_profiles(frames, r_max=3.0, bin_width=0.5, threads=1)

    assert [(profile.species_a, profile.species_b) for profile in profiles] == [
        ("H", "H"),
        ("H", "O"),
        ("O", "O"),
    ]


def test_compute_rdf_profiles_matches_single_pair_results():
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OHH",
            positions=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    pairwise_profiles = {
        (profile.species_a, profile.species_b): profile
        for profile in compute_rdf_profiles(frames, r_max=3.0, bin_width=0.5, threads=1)
    }
    single_profile = compute_rdf(
        frames,
        species_a="H",
        species_b="O",
        r_max=3.0,
        bin_width=0.5,
        threads=1,
    )

    pairwise_profile = pairwise_profiles[("H", "O")]
    np.testing.assert_allclose(pairwise_profile.bin_edges, single_profile.bin_edges)
    np.testing.assert_allclose(pairwise_profile.g_r, single_profile.g_r, equal_nan=True)


def test_compute_rdf_does_not_use_postpass_statistics_helper(monkeypatch):
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OHH",
            positions=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    def _raising_statistics_helper(*_args, **_kwargs):
        raise AssertionError("post-pass RDF statistics helper should not be called")

    monkeypatch.setattr(rdf_mod, "_compute_rdf_statistics_profile", _raising_statistics_helper)

    profile = compute_rdf(frames, species_a="H", species_b="O", r_max=3.0, bin_width=0.5, threads=1)

    assert profile.series_statistics is not None
    assert "g_r" in profile.series_statistics


def test_compute_rdf_profiles_do_not_use_postpass_statistics_helper(monkeypatch):
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OHH",
            positions=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.9, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    def _raising_statistics_helper(*_args, **_kwargs):
        raise AssertionError("post-pass RDF statistics helper should not be called")

    monkeypatch.setattr(rdf_mod, "_compute_rdf_statistics_profile", _raising_statistics_helper)

    profiles = compute_rdf_profiles(frames, r_max=3.0, bin_width=0.5, threads=1)

    assert profiles
    assert all(profile.series_statistics is not None for profile in profiles)


def test_compute_rdf_inline_statistics_match_reference_helper():
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9 + 0.005 * index, 0.0, 0.0], [1.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for index in range(100)
    ]

    profile = compute_rdf(frames, species_a="H", species_b="O", r_max=3.0, bin_width=0.5, threads=1)
    expected = rdf_mod._compute_rdf_statistics_profile(
        frames,
        label_a="H",
        label_b="O",
        r_max=3.0,
        bin_edges=profile.bin_edges,
        shell_volumes=rdf_mod._shell_volumes_from_edges(profile.bin_edges),
        selection_cache=rdf_mod._resolve_rdf_selection_cache(frames, label_a="H", label_b="O"),
        strategy_override=None,
    )

    stats = profile.series_statistics["g_r"]
    np.testing.assert_array_equal(stats.point_count, expected.point_count)
    np.testing.assert_array_equal(stats.sample_n, expected.sample_n)
    np.testing.assert_allclose(stats.sample_std, expected.sample_std, equal_nan=True, atol=1.0e-12)
    np.testing.assert_allclose(stats.sample_sem, expected.sample_sem, equal_nan=True, atol=1.0e-12)
    np.testing.assert_array_equal(stats.block_n, expected.block_n)
    np.testing.assert_allclose(stats.block_std, expected.block_std, equal_nan=True, atol=1.0e-12)
    np.testing.assert_allclose(stats.block_sem, expected.block_sem, equal_nan=True, atol=1.0e-12)


def test_compute_rdf_profiles_inline_statistics_match_single_pair_results():
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9 + 0.005 * index, 0.0, 0.0], [1.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for index in range(100)
    ]

    collection_profile = {
        (profile.species_a, profile.species_b): profile
        for profile in compute_rdf_profiles(frames, r_max=3.0, bin_width=0.5, threads=1)
    }[("H", "O")]
    single_profile = compute_rdf(
        frames,
        species_a="H",
        species_b="O",
        r_max=3.0,
        bin_width=0.5,
        threads=1,
    )

    collection_stats = collection_profile.series_statistics["g_r"]
    single_stats = single_profile.series_statistics["g_r"]
    np.testing.assert_array_equal(collection_stats.point_count, single_stats.point_count)
    np.testing.assert_array_equal(collection_stats.sample_n, single_stats.sample_n)
    np.testing.assert_allclose(collection_stats.sample_std, single_stats.sample_std, equal_nan=True)
    np.testing.assert_allclose(collection_stats.sample_sem, single_stats.sample_sem, equal_nan=True)
    np.testing.assert_array_equal(collection_stats.block_n, single_stats.block_n)
    np.testing.assert_allclose(collection_stats.block_std, single_stats.block_std, equal_nan=True)
    np.testing.assert_allclose(collection_stats.block_sem, single_stats.block_sem, equal_nan=True)


def test_save_rdf_profiles_writes_profile_collection(tmp_path):
    profiles = [
        RDFProfile(
            species_a="H",
            species_b="H",
            bin_edges=np.array([0.0, 1.0, 2.0], dtype=float),
            bin_centers=np.array([0.5, 1.5], dtype=float),
            g_r=np.array([0.3, 0.4], dtype=float),
            n_frames=2,
        ),
        RDFProfile(
            species_a="H",
            species_b="O",
            bin_edges=np.array([0.0, 1.0, 2.0], dtype=float),
            bin_centers=np.array([0.5, 1.5], dtype=float),
            g_r=np.array([0.1, 0.2], dtype=float),
            n_frames=2,
        ),
    ]

    out = tmp_path / "rdf_collection.h5"
    save_rdf_profiles(profiles, out)

    payloads = read_linak_hdf5_profiles(out, expected_analysis="rdf")
    assert len(payloads) == 2


def test_load_rdf_profile_rejects_missing_v1_bin_width_metadata(tmp_path):
    out = tmp_path / "old_rdf.h5"
    with h5py.File(out, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["linak_format_version"] = 1
        handle.attrs["linak_version"] = LINAK_VERSION
        handle.attrs["analysis"] = "rdf"
        handle.attrs["metadata_json"] = json.dumps(
            {
                "analysis": "rdf",
                "analysis_schema_version": 1,
                "profile_uid": "rdf-without-bin-width",
                "species_a": "O",
                "species_b": "H",
                "n_frames": 1,
            }
        )
        handle.create_dataset("bin_edges_A", data=np.array([0.0, 1.0, 2.0], dtype=float))
        handle.create_dataset("bin_centers_A", data=np.array([0.5, 1.5], dtype=float))
        handle.create_dataset("g_r", data=np.array([0.0, 1.0], dtype=float))

    with pytest.raises(ValueError, match="missing required v1 metadata bin_width_A"):
        load_rdf_profile(out, species_a="O", species_b="H")


def test_load_rdf_profile_rejects_csv_input(tmp_path):
    csv = tmp_path / "old_rdf.csv"
    csv.write_text(
        "bin_left_A,bin_right_A,r_A,g_r\n0.0,1.0,0.5,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Use .h5/.hdf5"):
        load_rdf_profile(csv)


def test_plot_rdf_profiles_uses_multi_line_plot_for_multiple_profiles(monkeypatch):
    from linak.analysis.rdf import RDFProfile, plot_rdf_profiles

    captured = {}

    def _fake_plot_multi_line_series(x_series, y_series, labels, **_kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["labels"] = labels
        return None

    monkeypatch.setattr("linak.analysis.rdf.plot_multi_line_series", _fake_plot_multi_line_series)

    profile_a = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.1, 0.2]),
        n_frames=1,
    )
    profile_b = RDFProfile(
        species_a="H",
        species_b="H",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.3, 0.4]),
        n_frames=1,
    )

    plot_rdf_profiles([profile_a, profile_b], show=False)
    assert captured["labels"] == ["O-H", "H-H"]


def test_load_rdf_profiles_filters_by_stored_species_metadata(tmp_path):
    out = tmp_path / "multi_rdf.h5"
    write_linak_hdf5_profile_collection(
        out,
        analysis="rdf",
        profiles=[
            {
                "metadata": {
                    "species_a": "O",
                    "species_b": "H",
                    "n_frames": 1,
                    "bin_width_A": 1.0,
                },
                "datasets": {
                    "bin_centers_A": np.array([0.5, 1.5], dtype=float),
                    "g_r": np.array([0.1, 0.2], dtype=float),
                },
            },
            {
                "metadata": {
                    "species_a": "H",
                    "species_b": "H",
                    "n_frames": 1,
                    "bin_width_A": 1.0,
                },
                "datasets": {
                    "bin_centers_A": np.array([0.5, 1.5], dtype=float),
                    "g_r": np.array([0.3, 0.4], dtype=float),
                },
            },
        ],
    )

    loaded = load_rdf_profiles(out, species_a="O", species_b="H")

    assert len(loaded) == 1
    assert loaded[0].species_a == "O"
    assert loaded[0].species_b == "H"


def test_load_rdf_profiles_supports_symmetric_cross_pair_lookup(tmp_path):
    out = tmp_path / "multi_rdf.h5"
    write_linak_hdf5_profile_collection(
        out,
        analysis="rdf",
        profiles=[
            {
                "metadata": {
                    "species_a": "O",
                    "species_b": "H",
                    "n_frames": 1,
                    "bin_width_A": 1.0,
                },
                "datasets": {
                    "bin_centers_A": np.array([0.5, 1.5], dtype=float),
                    "g_r": np.array([0.1, 0.2], dtype=float),
                },
            }
        ],
    )

    loaded = load_rdf_profiles(out, species_a="H", species_b="O")

    assert len(loaded) == 1
    assert loaded[0].species_a == "H"
    assert loaded[0].species_b == "O"


def test_compute_rdf_reuses_species_selection_when_atom_identities_are_stable(monkeypatch):
    frames = [
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    calls = {"count": 0}
    original_select_mask = rdf_mod._select_mask

    def _counting_select_mask(numbers, species):
        calls["count"] += 1
        return original_select_mask(numbers, species)

    monkeypatch.setattr("linak.analysis.rdf._select_mask", _counting_select_mask)

    compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=1)

    assert calls["count"] == 2


def test_compute_rdf_warns_and_falls_back_when_atom_identities_change(monkeypatch, caplog):
    frames = [
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "HO",
            positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    calls = {"count": 0}
    original_select_mask = rdf_mod._select_mask

    def _counting_select_mask(numbers, species):
        calls["count"] += 1
        return original_select_mask(numbers, species)

    monkeypatch.setattr("linak.analysis.rdf._select_mask", _counting_select_mask)

    with caplog.at_level("WARNING", logger="linak.analysis.rdf"):
        compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=1)

    assert "RDF atom identities/order changed" in caplog.text
    assert calls["count"] == 4


def test_compute_rdf_logs_dense_orthorhombic_backend(caplog):
    frames = [
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [1.0 + 0.01 * index, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for index in range(4)
    ]

    with caplog.at_level("DEBUG", logger="linak.analysis.rdf"):
        compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=1)

    assert "Using RDF backend: dense orthorhombic chunked" in caplog.text


def test_compute_rdf_logs_sparse_orthorhombic_backend(monkeypatch, caplog):
    frames = [
        Atoms(
            "OOHH",
            positions=[
                [0.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [15.8, 0.0, 0.0],
            ],
            cell=[80.0, 80.0, 80.0],
            pbc=True,
        )
        for _ in range(3)
    ]
    monkeypatch.setattr(rdf_mod, "_RDF_DENSE_PAIR_THRESHOLD", 1)

    with caplog.at_level("DEBUG", logger="linak.analysis.rdf"):
        compute_rdf(frames, species_a="O", species_b="H", r_max=1.5, bin_width=0.25, threads=1)

    assert "Using RDF backend: sparse orthorhombic cutoff" in caplog.text


def test_compute_rdf_logs_generic_fallback_for_skewed_cells(caplog):
    frames = [
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[[10.0, 0.0, 0.0], [2.0, 8.0, 0.0], [0.0, 0.0, 10.0]],
            pbc=True,
        )
    ]

    with caplog.at_level("DEBUG", logger="linak.analysis.rdf"):
        compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=1)

    assert "Using RDF backend: generic framewise fallback." in caplog.text


def test_compute_rdf_parallel_path_uses_thread_executor(monkeypatch):
    frames = [
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [1.0 + 0.01 * index, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for index in range(20)
    ]
    captured = {"chunks": 0}

    class _FakeThreadPoolExecutor:
        def __init__(self, *, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def map(self, func, chunks, configs):
            for chunk, config in zip(chunks, configs):
                captured["chunks"] += 1
                yield func(chunk, config)

    monkeypatch.setattr("linak.analysis.rdf.ThreadPoolExecutor", _FakeThreadPoolExecutor)

    profile = compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=2)

    assert captured["max_workers"] == 2
    assert captured["chunks"] >= 2
    assert profile.n_frames == len(frames)
