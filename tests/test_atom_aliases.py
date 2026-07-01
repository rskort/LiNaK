from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms

from linak.analysis.common import (
    RAW_SPECIES_ARRAY,
    grouped_raw_species_for_split_elements,
    normalize_species_query,
    raw_species_labels,
    select_species_indices,
)
from linak.analysis.water import oh_molecule_topology
from linak.trajectory.io import read_trajectory, write_trajectory


def test_xyz_nonstandard_labels_resolve_and_preserve_raw_species(tmp_path):
    path = tmp_path / "labels.xyz"
    path.write_text(
        "\n".join(
            [
                "4",
                "raw labels",
                "Pt_top 0 0 0",
                "Pt 1 0 0",
                "Ow 0 1 0",
                "H 0 1 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    frames = read_trajectory(path)

    assert frames[0].get_chemical_symbols() == ["Pt", "Pt", "O", "H"]
    assert raw_species_labels(frames[0]).tolist() == ["Pt_top", "Pt", "Ow", "H"]
    assert select_species_indices(frames[0], "Pt").tolist() == [0, 1]
    assert select_species_indices(frames[0], "element:Pt").tolist() == [0, 1]
    assert select_species_indices(frames[0], "species:Pt_top").tolist() == [0]
    assert select_species_indices(frames[0], "species:Pt").tolist() == [1]
    assert select_species_indices(frames[0], "Ow").tolist() == [2]
    assert select_species_indices(frames[0], "O").tolist() == [2]
    assert normalize_species_query("Ow") == ("species", "species:Ow")


def test_xyz_unknown_label_suggests_atom_alias(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("1\nbad\nQq_top 0 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--atom-alias Qq_top=<element>"):
        read_trajectory(path)

    frames = read_trajectory(path, atom_aliases=["Qq_top=O"])
    assert frames[0].get_chemical_symbols() == ["O"]
    assert raw_species_labels(frames[0]).tolist() == ["Qq_top"]


def test_raw_species_labels_roundtrip_through_linak_trajectory_hdf5(tmp_path):
    frame = Atoms(symbols=["Pt", "Pt", "O"], positions=np.zeros((3, 3)))
    frame.new_array(RAW_SPECIES_ARRAY, np.asarray(["Pt", "Pt_top", "Ow"]))
    output = tmp_path / "roundtrip.traj.h5"

    write_trajectory([frame], output)
    loaded = read_trajectory(output)

    assert loaded[0].get_chemical_symbols() == ["Pt", "Pt", "O"]
    assert raw_species_labels(loaded[0]).tolist() == ["Pt", "Pt_top", "Ow"]
    assert select_species_indices(loaded[0], "element:Pt").tolist() == [0, 1]
    assert select_species_indices(loaded[0], "species:Pt_top").tolist() == [1]


def test_oh_topology_uses_resolved_elements_not_raw_species_labels():
    frame = Atoms(
        symbols=["O", "H", "H"],
        positions=np.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.95, 0.0]]),
    )
    frame.new_array(RAW_SPECIES_ARRAY, np.asarray(["Ow", "Hw", "Hw"]))

    topology = oh_molecule_topology(frame, oh_cutoff=1.27)

    assert topology.indices_for("mol:H2O").tolist() == [[0, 1, 2]]


def test_grouped_raw_species_only_exposes_split_elements():
    frame = Atoms(
        symbols=["Pt", "Pt", "O", "H", "Na"],
        positions=np.zeros((5, 3)),
    )
    frame.new_array(RAW_SPECIES_ARRAY, np.asarray(["Pt", "Pt_top", "O", "H", "Na"]))

    assert grouped_raw_species_for_split_elements([frame]) == ["Pt", "Pt_top"]
    assert select_species_indices(frame, "species:O").tolist() == [2]
