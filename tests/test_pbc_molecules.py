"""Whole-molecule PBC wrapping uses reconstructed mass-weighted centers."""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from linak import cli
from linak.analysis.water import oh_molecule_topology
from linak.cell_geometry import wrapped_frame_positions
from linak.pbc import apply_pbc_to_frames


@pytest.mark.parametrize("anchor_x", [9.8, 9.99, -0.2, 29.99])
@pytest.mark.parametrize("hydrogens", [1, 2, 3])
def test_wrap_by_com_reconstructs_molecules_and_keeps_free_atoms_independent(anchor_x, hydrogens):
    anchor = np.array([anchor_x, 4, 4])
    offsets = np.array([[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]])[: hydrogens + 1]
    whole = anchor + offsets
    frame = Atoms(
        "O" + "H" * hydrogens + "K",
        positions=np.vstack((whole, [12.3, -1, 5])),
        cell=[10, 10, 10],
        pbc=True,
    )
    frame.set_constraint(FixAtoms(indices=[0]))
    # Deliberately start with a molecule split across the boundary.
    frame.positions[1, 0] %= 10
    original = frame.positions.copy()
    result = apply_pbc_to_frames([frame], None, keep_molecules=True)[0]
    weights = frame.get_masses()[:-1]
    center = np.average(whole, axis=0, weights=weights)
    expected = whole + (wrapped_frame_positions(frame, center) - center)
    np.testing.assert_allclose(result.positions[:-1], expected, atol=1e-12)
    np.testing.assert_allclose(result.positions[-1], [2.3, 9, 5], atol=1e-12)
    np.testing.assert_allclose(
        result.positions[1:-1] - result.positions[0], offsets[1:], atol=1e-12
    )
    np.testing.assert_array_equal(frame.positions, original)
    np.testing.assert_array_equal(result.constraints[0].get_indices(), [0])


def test_triclinic_origin_changing_cells_and_masses():
    frames = []
    expected = []
    for scale in (1, 1.3):
        cell = np.array([[10, 0, 0], [2, 10, 0], [1, 1, 10]]) * scale
        origin = np.array([3, -2, 5])
        anchor = np.array([0.99, 0.4, 0.4]) @ cell + origin
        whole = anchor + [[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]
        frame = Atoms("OHH", positions=whole, cell=cell, pbc=True)
        frame.set_celldisp(origin)
        frame.set_masses([16, 2, 1])
        center = np.average(whole, axis=0, weights=frame.get_masses())
        expected.append(whole + wrapped_frame_positions(frame, center) - center)
        frame.positions[:] = wrapped_frame_positions(frame)
        frames.append(frame)
    results = apply_pbc_to_frames(frames, None, keep_molecules=True)
    for result, source, positions in zip(results, frames, expected):
        np.testing.assert_allclose(result.positions, positions, atol=1e-12)
        np.testing.assert_array_equal(result.cell, source.cell)
        np.testing.assert_array_equal(result.get_masses(), source.get_masses())


def test_compatible_topology_is_reused(monkeypatch):
    from linak.analysis import water

    frame = Atoms("OH", positions=[[9.8, 4, 4], [0.7, 4, 4]], cell=[10] * 3, pbc=True)
    oh_molecule_topology(frame)
    monkeypatch.setattr(
        water,
        "_oh_molecule_topology_with_backend",
        lambda *a, **k: pytest.fail("Compatible topology should be reused"),
    )
    result = apply_pbc_to_frames([frame], None, keep_molecules=True)[0]
    assert result.positions[1, 0] - result.positions[0, 0] == pytest.approx(0.9)


def test_membership_is_resolved_again_after_dissociation():
    bonded = Atoms("OH", positions=[[9.8, 4, 4], [0.7, 4, 4]], cell=[10] * 3, pbc=True)
    oh_molecule_topology(bonded)
    dissociated = bonded.copy()
    dissociated.positions[1, 0] = 2
    results = apply_pbc_to_frames([bonded, dissociated], None, keep_molecules=True)
    assert results[0].positions[1, 0] == pytest.approx(10.7)
    assert results[1].positions[1, 0] == pytest.approx(2)


@pytest.mark.parametrize("suffix", ["xyz", "traj.h5"])
def test_cli_keep_molecules_round_trip(tmp_path, monkeypatch, suffix):
    from linak.trajectory.io import read_trajectory

    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)
    source = tmp_path / "source.xyz"
    target = tmp_path / f"wrapped.{suffix}"
    frame = Atoms("OHH", positions=[[9.8, 4, 4], [0.7, 4, 4], [9.8, 4.9, 4]])
    write(source, frame, format="extxyz")
    assert (
        cli.main(
            [
                "apply",
                "pbc",
                str(source),
                "--cell",
                "10",
                "10",
                "10",
                "--keep-molecules",
                "--output",
                str(target),
            ]
        )
        == 0
    )
    result = read_trajectory(target)[0]
    np.testing.assert_allclose(
        result.positions, [[9.8, 4, 4], [10.7, 4, 4], [9.8, 4.9, 4]], atol=1e-6
    )
    np.testing.assert_allclose(read(source).positions, frame.positions)
