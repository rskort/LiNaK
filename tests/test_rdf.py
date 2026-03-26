import json

import h5py
import numpy as np
import pytest
from ase import Atoms

import linak.analysis.rdf as rdf_mod
from linak.analysis.rdf import compute_rdf, load_rdf_profile, load_rdf_profiles, save_rdf_profile
from linak.storage.hdf5_utils import write_linak_hdf5_profile_collection


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


def test_auto_r_max_uses_half_min_perpendicular_cell_height_for_skewed_cells():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[[10.0, 0.0, 0.0], [2.0, 8.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=True,
    )
    profile = compute_rdf([frame], species_a="O", species_b="H", r_max=None, bin_width=1.0)
    assert profile.bin_edges[-1] == pytest.approx(4.0)


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


def test_load_rdf_profile_supports_legacy_bin_edges_dataset(tmp_path):
    out = tmp_path / "legacy_rdf.h5"
    with h5py.File(out, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["analysis"] = "rdf"
        handle.attrs["metadata_json"] = json.dumps(
            {
                "species_a": "O",
                "species_b": "H",
                "n_frames": 1,
            }
        )
        handle.create_dataset("bin_edges_A", data=np.array([0.0, 1.0, 2.0], dtype=float))
        handle.create_dataset("bin_centers_A", data=np.array([0.5, 1.5], dtype=float))
        handle.create_dataset("g_r", data=np.array([0.0, 1.0], dtype=float))

    loaded = load_rdf_profile(out, species_a="O", species_b="H")
    np.testing.assert_allclose(loaded.bin_edges, np.array([0.0, 1.0, 2.0]))


def test_load_rdf_profile_rejects_csv_input(tmp_path):
    csv = tmp_path / "legacy_rdf.csv"
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
                "metadata": {"species_a": "O", "species_b": "H", "n_frames": 1},
                "datasets": {
                    "bin_centers_A": np.array([0.5, 1.5], dtype=float),
                    "g_r": np.array([0.1, 0.2], dtype=float),
                },
            },
            {
                "metadata": {"species_a": "H", "species_b": "H", "n_frames": 1},
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


def test_compute_rdf_parallel_path_uses_chunk_executor(monkeypatch):
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

    class _FakeProcessPoolExecutor:
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

    monkeypatch.setattr("linak.analysis.rdf.ProcessPoolExecutor", _FakeProcessPoolExecutor)

    profile = compute_rdf(frames, species_a="O", species_b="H", r_max=2.0, bin_width=1.0, threads=2)

    assert captured["max_workers"] == 2
    assert captured["chunks"] >= 2
    assert profile.n_frames == len(frames)
