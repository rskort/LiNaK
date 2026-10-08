from __future__ import annotations

import argparse

from ase import Atoms
import numpy as np
import pytest

from linak import cli
from linak.analysis.coordination import load_coordination_profiles
from linak.analysis.rdf import compute_rdf, load_rdf_profiles
from linak.plot.profile_persistence import build_plot_profile_payload, flatten_plot_profile_payload
from linak.trajectory import io


@pytest.fixture
def pair_trajectory(tmp_path):
    frames = [
        Atoms(
            "PtOOHH",
            positions=[[1, 1, 1], [2, 1, 1], [5, 5, 5], [2, 2, 1], [6, 5, 5]],
            cell=[12, 12, 12],
            pbc=True,
        )
        for _ in range(2)
    ]
    source = tmp_path / "pairs.traj.h5"
    io.write_trajectory(frames, source)
    return source, frames


def test_rdf_requested_pairs_only_reuse_one_trajectory_load(pair_trajectory, tmp_path, monkeypatch):
    source, frames = pair_trajectory
    output = tmp_path / "selected.rdf.h5"
    reads = []
    reader = io.read_trajectory

    def counted_reader(path, **kwargs):
        reads.append(path)
        return reader(path, **kwargs)

    monkeypatch.setattr(io, "read_trajectory", counted_reader)
    assert (
        cli.main(
            [
                "compute",
                "rdf",
                str(source),
                "--species",
                "O",
                "H",
                "--species",
                "Pt",
                "O",
                "--species",
                "H",
                "element:O",
                "--r-max",
                "4",
                "--bin-width",
                "0.5",
                "--threads",
                "1",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    profiles = load_rdf_profiles(output)
    assert {(p.species_a, p.species_b) for p in profiles} == {("H", "O"), ("O", "Pt")}
    assert len(reads) == 1
    for profile in profiles:
        expected = compute_rdf(
            frames,
            species_a=profile.species_a,
            species_b=profile.species_b,
            r_max=4,
            bin_width=0.5,
            threads=1,
        )
        np.testing.assert_allclose(profile.g_r, expected.g_r, rtol=1e-6)


def test_cn_repeat_pairs_preserves_direction(pair_trajectory, tmp_path):
    source, _ = pair_trajectory
    output = tmp_path / "selected.coordination.h5"
    assert (
        cli.main(
            [
                "compute",
                "cn",
                str(source),
                "--species",
                "O",
                "H",
                "--species",
                "H",
                "O",
                "--species",
                "O",
                "H",
                "--cutoff",
                "2",
                "--timestep-fs",
                "1",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    profiles = load_coordination_profiles(output)
    assert [(p.species_a, p.species_b) for p in profiles] == [("O", "H"), ("H", "O")]


@pytest.mark.parametrize("command", ["rdf", "cn"])
@pytest.mark.parametrize(
    "extra",
    [
        ["--species", "O"],
        ["--species", "O", "H", "Pt"],
        ["--species-a", "O", "--species-b", "H"],
    ],
)
def test_pair_parser_rejects_incomplete_and_removed_flags(command, extra):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["compute", command, "input.traj.h5", *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("pair", [("mol:H", "Pt"), ("mol:OH", "element:O")])
def test_dynamic_molecule_rdf_has_clear_error_before_read(tmp_path, monkeypatch, pair, capsys):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Unsupported selector should fail before reading the trajectory")

    monkeypatch.setattr(io, "read_trajectory", unexpected_read)
    assert cli.main(["compute", "rdf", str(tmp_path / "input.traj.h5"), "--species", *pair]) == 1
    assert "Dynamic molecule selector" in capsys.readouterr().err


def test_rdf_raw_label_pair_and_atom_pair(pair_trajectory, tmp_path):
    source, frames = pair_trajectory
    for frame in frames:
        frame.new_array("linak_raw_species", np.asarray(["Pt", "OH", "O", "H", "freeH"]))
    io.write_trajectory(frames, source)
    output = tmp_path / "raw.rdf.h5"
    assert (
        cli.main(
            [
                "compute",
                "rdf",
                str(source),
                "--species",
                "species:freeH",
                "Pt",
                "--species",
                "species:OH",
                "element:O",
                "--species",
                "atoms[4]",
                "Pt",
                "--r-max",
                "4",
                "--bin-width",
                "0.5",
                "--threads",
                "1",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    profiles = load_rdf_profiles(output)
    assert len(profiles) == 3
    indexed = next(profile for profile in profiles if profile.selection_kind_a == "atoms")
    assert indexed.atom_indices_a is not None
    assert indexed.atom_indices_a.tolist() == [4]


def test_plot_repeated_pairs_filter_eager_and_lazy_contexts(pair_trajectory, tmp_path):
    source, _ = pair_trajectory
    output = tmp_path / "all.rdf.h5"
    assert (
        cli.main(
            [
                "compute",
                "rdf",
                str(source),
                "--bin-width",
                "0.5",
                "--threads",
                "1",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    args = argparse.Namespace(species_pairs=[["O", "H"], ["Pt", "O"]], view_mapping=None)
    context = cli._build_rdf_gui_context(args, sources=[str(output)])
    assert {(p.species_a, p.species_b) for p in context.profile} == {("H", "O"), ("O", "Pt")}
    catalog = cli._build_rdf_gui_lazy_catalog(args, sources=[str(output)])
    assert len(catalog.build_initial_context().series_descriptors) == 2


def test_generic_plot_parser_accepts_repeated_pairs():
    args = cli.build_parser().parse_args(
        [
            "plot",
            "input.rdf.h5",
            "--species",
            "O",
            "H",
            "--species",
            "Pt",
            "O",
        ]
    )
    assert args.species_pairs == [["O", "H"], ["Pt", "O"]]
    assert args.species is None


@pytest.mark.parametrize("command", ["rdf", "cn"])
def test_repeated_pairs_dry_run_does_not_load_trajectory(command, monkeypatch, capsys):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Dry run must not load trajectory frames")

    monkeypatch.setattr(io, "read_trajectory", unexpected_read)
    assert (
        cli.main(
            [
                "--log-level",
                "INFO",
                "compute",
                command,
                "input.traj.h5",
                "--species",
                "O",
                "H",
                "--species",
                "Pt",
                "O",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "pairs=" in capsys.readouterr().err


def test_cn_plot_pairs_filter_eager_and_lazy_contexts(pair_trajectory, tmp_path):
    source, _ = pair_trajectory
    output = tmp_path / "all.coordination.h5"
    assert (
        cli.main(
            [
                "compute",
                "cn",
                str(source),
                "--species",
                "O",
                "H",
                "--species",
                "H",
                "O",
                "--species",
                "Pt",
                "O",
                "--cutoff",
                "2",
                "--timestep-fs",
                "1",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    args = cli.build_parser().parse_args(
        [
            "plot",
            str(output),
            "--species",
            "O",
            "H",
            "--species",
            "Pt",
            "O",
        ]
    )
    context = cli._build_coordination_gui_context(args, sources=[str(output)])
    assert {(p.species_a, p.species_b) for p in context.profile} == {("O", "H"), ("Pt", "O")}
    catalog = cli._build_coordination_gui_lazy_catalog(args, sources=[str(output)])
    initial = catalog.build_initial_context()
    descriptors = initial.series_descriptors
    assert {(d["rendered_species_a"], d["rendered_species_b"]) for d in descriptors} == {
        ("O", "H"),
        ("Pt", "O"),
    }


@pytest.mark.parametrize("analysis", ["rdf", "coordination"])
def test_pair_plot_selection_roundtrips(analysis):
    pairs = [["O", "H"], ["Pt", "O"]]
    profile_key = f"plot:{analysis}"
    payload = build_plot_profile_payload(profile_key, {"species_pairs": pairs})
    assert payload["source_selection"]["species_pairs"] == pairs
    assert flatten_plot_profile_payload(profile_key, payload)["species_pairs"] == pairs


@pytest.mark.parametrize("analysis", [None, "rdf", "coordination"])
def test_plot_parser_rejects_removed_pair_flags(analysis):
    with pytest.raises(SystemExit) as error:
        cli.build_plot_parser(analysis=analysis).parse_args(
            [
                "input.h5",
                "--species-a",
                "O",
                "--species-b",
                "H",
            ]
        )
    assert error.value.code == 2
