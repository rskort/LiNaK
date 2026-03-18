import json
import numpy as np
import pytest
import h5py
from ase import Atoms

import linak.analysis.rdf as rdf_mod
from linak.analysis.rdf import compute_rdf, load_rdf_profile, save_rdf_profile


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


def test_compute_rdf_requires_nonzero_cell_volume():
    frame = Atoms("OH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-zero cell volume"):
        compute_rdf([frame], species_a="O", species_b="H", r_max=2.0, bin_width=1.0)


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

    out = tmp_path / "rdf_non_multiple.h5"
    save_rdf_profile(profile, out)
    loaded = load_rdf_profile(out, species_a="O", species_b="H")
    np.testing.assert_allclose(loaded.bin_edges, profile.bin_edges)


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
    np.testing.assert_allclose(loaded.g_r, profile.g_r)
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
