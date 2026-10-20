"""Bulk-water references must follow slab geometry across periodic cuts."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
from ase import Atoms

from linak.analysis.potential import (
    BOHR_TO_ANG,
    HARTREE_TO_EV,
    CubeHeader,
    PotentialConfig,
    _two_sided_water_bulk_potentials,
    _validate_symmetric_water_windows,
    _water_bulk_potential_ev,
)
from linak.cli import main
from linak.cube_io import CubeDataset, write_cube_file


@pytest.fixture(autouse=True)
def _capture_potential_warnings(caplog):
    # CLI tests in this process may have set the parent logger to ERROR.
    caplog.set_level("WARNING", logger="linak.analysis.potential")


def _water_cell(*, shift=0.0, reverse=False, missing_bottom=False, scale=1.0,
                cut_water_z=(0.2, 39.8)):
    symbols, positions = [], []
    for z in (14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0):
        for x, y in ((2, 2), (7, 2), (2, 7), (7, 7)):
            symbols.append("Pt")
            positions.append([x, y, z * scale])
    # The first and last molecules straddle the solvent cut. Individual atom
    # min/max bounds used to turn either side into an almost cell-wide window.
    for z in (cut_water_z[0], 3, 6, 10, 12, 28, 30, 34, 37, cut_water_z[1]):
        if missing_bottom and z < 14:
            continue
        direction = -1 if z < 14 else 1
        x = 2 if z < 1 else 8 if z > 39 else 5
        symbols.extend(("O", "H", "H"))
        positions.extend([
            [x, 5, z * scale],
            [x + 0.5, 5, z * scale + direction * 0.65],
            [x - 0.5, 5, z * scale + direction * 0.65],
        ])
    atoms = Atoms(symbols, positions=positions, cell=[10, 10, 40 * scale], pbc=True)
    atoms.positions[:, 2] += shift
    atoms.wrap()
    if reverse:
        atoms = atoms[::-1]
    counts = (10, 10, 400)
    header = CubeHeader(
        natoms=len(atoms), origin_bohr=np.zeros(3),
        nx=counts[0], ny=counts[1], nz=counts[2],
        vx_bohr=atoms.cell[0] / counts[0] / BOHR_TO_ANG,
        vy_bohr=atoms.cell[1] / counts[1] / BOHR_TO_ANG,
        vz_bohr=atoms.cell[2] / counts[2] / BOHR_TO_ANG,
        atom_numbers=atoms.numbers,
        atom_z_bohr=atoms.positions[:, 2] / BOHR_TO_ANG,
        atom_positions_bohr=atoms.positions / BOHR_TO_ANG,
    )
    grid = np.arange(400) * 0.1 * scale
    q = (grid - shift) % (40 * scale)
    values = np.where(q < 14 * scale, 2.0, np.where(q > 26 * scale, 8.0, 100.0))
    return header, grid, values


@pytest.mark.parametrize("shift", [0.0, 0.3, 7.9, 20.0, 39.8])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.parametrize("cut_water_z", [(0.2, 39.8), (0.6, 39.8), (0.2, 39.4), (0.6, 39.4)])
def test_periodic_windows_follow_slab_sides_without_duplicates_or_swaps(
    shift, reverse, scale, cut_water_z
):
    header, grid, values = _water_cell(
        shift=shift, reverse=reverse, scale=scale, cut_water_z=cut_water_z
    )
    result = _two_sided_water_bulk_potentials(
        header=header, z_ang=grid, v_xyavg_ev=values,
        config=PotentialConfig(cshe_offset_ev=-0.81),
    )
    reference = result["reference"]
    assert reference.resolved_layout == "symmetric"
    assert result["water_molecule_count"] == 10
    top_value, top_lo, top_hi = result["top"]
    bottom_value, bottom_lo, bottom_hi = result["bottom"]
    assert top_value == pytest.approx(8.0)
    assert bottom_value == pytest.approx(2.0)
    assert bottom_hi <= reference.bottom.frame_values[0]
    assert top_lo >= reference.top.frame_values[0]
    assert bottom_hi < top_lo
    assert top_hi - top_lo >= 5 * scale
    assert bottom_hi - bottom_lo >= 5 * scale


def test_narrow_padded_window_relaxes_padding_and_records_reason():
    notes = []
    value, lo, hi = _water_bulk_potential_ev(
        z_ang=np.arange(0, 12.1, 0.1), v_xyavg_ev=np.ones(121),
        water_z_min_ang=0, water_z_max_ang=12, padding_ang=5,
        minimum_width_ang=6, diagnostics=notes,
    )
    assert value == 1
    assert hi - lo >= 6
    assert any("padding relaxed" in note for note in notes)


def test_fragmented_water_with_only_narrow_clean_windows_is_unavailable(caplog):
    grid = np.arange(0, 12.1, 0.1)
    contaminated = np.ones(grid.size, dtype=bool)
    contaminated[50:63] = False
    result = _water_bulk_potential_ev(
        z_ang=grid, v_xyavg_ev=np.ones(grid.size),
        water_z_min_ang=0, water_z_max_ang=12, padding_ang=5,
        contaminated_z_mask=contaminated, minimum_width_ang=6,
    )
    assert result == (None, None, None)
    assert "sufficiently wide" in caplog.text


@pytest.mark.parametrize("top,bottom,problem", [
    ((8, 2, 10), (2, 30, 38), "outside its slab side"),  # swapped
    ((8, 30, 38), (2, 30, 38), "overlapping"),  # duplicate
    ((8, 10, 31), (2, 2, 13), "overlapping"),
    ((8, 30, 31), (2, 2, 10), "unexpectedly narrow"),
    ((None, None, None), (2, 2, 10), "missing"),
])
def test_invalid_symmetric_windows_are_rejected_with_diagnostics(top, bottom, problem, caplog):
    windows = {"top": top, "bottom": bottom}
    notes = _validate_symmetric_water_windows(
        windows, side_limits={"top": (26, 40), "bottom": (0, 14)},
        minimum_widths={"top": 5, "bottom": 5},
    )
    assert windows["top"] == (None, None, None)
    assert problem in caplog.text
    assert any(problem in note for note in notes)
    if problem == "overlapping" or top[1] == 2:
        assert windows["bottom"] == (None, None, None)
    else:
        assert windows["bottom"] == bottom


def test_missing_water_side_has_no_combined_potential(caplog):
    header, grid, values = _water_cell(missing_bottom=True)
    result = _two_sided_water_bulk_potentials(
        header=header, z_ang=grid, v_xyavg_ev=values,
        config=PotentialConfig(surface_layout="symmetric", water_padding_ang=0),
    )
    assert result["bottom"] == (None, None, None)
    assert result["combined"] is None
    assert "missing" in result["reason"]
    assert "Missing bottom" in caplog.text


def test_nonzero_cube_origin_preserves_window_assignment():
    header, grid, values = _water_cell()
    origin = np.asarray([0.0, 0.0, 6.2])
    header = replace(
        header, origin_bohr=origin / BOHR_TO_ANG,
        atom_positions_bohr=header.atom_positions_bohr + origin / BOHR_TO_ANG,
        atom_z_bohr=header.atom_z_bohr + origin[2] / BOHR_TO_ANG,
    )
    result = _two_sided_water_bulk_potentials(
        header=header, z_ang=grid + origin[2], v_xyavg_ev=values,
        config=PotentialConfig(),
    )
    assert result["top"][0] == pytest.approx(8)
    assert result["bottom"][0] == pytest.approx(2)


@pytest.mark.parametrize("missing_bottom", [False, True])
def test_cli_persists_validated_windows_and_incomplete_status(tmp_path, missing_bottom, caplog):
    header, _, values = _water_cell(missing_bottom=missing_bottom)
    dataset = CubeDataset(
        comment_1="Potential window regression", comment_2="",
        natoms_signed=header.natoms, origin_bohr=header.origin_bohr,
        grid_counts_signed=np.array([1, 1, header.nz]),
        grid_vectors_bohr=np.array([header.vx_bohr * header.nx,
                                    header.vy_bohr * header.ny, header.vz_bohr]),
        atom_numbers=header.atom_numbers, atom_charges=np.zeros(header.natoms),
        atom_positions_bohr=header.atom_positions_bohr,
        values=values.reshape(1, 1, -1) / HARTREE_TO_EV,
    )
    cube = write_cube_file(dataset, tmp_path / "water-v_hartree.cube")
    (tmp_path / "output.out").write_text("Fermi energy [eV]: 1.0\n")
    output = tmp_path / "water.potential.h5"
    caplog.set_level("INFO", logger="linak.analysis.surface")
    rc = main([
        "compute", "potential", str(cube), "--output", str(output),
        "--surface-layout", "symmetric", "--strict", "--cshe-offset-ev", "-0.81",
    ])
    assert rc == (1 if missing_bottom else 0)
    assert "Surface layout:" not in caplog.text
    with h5py.File(output) as f:
        r = f["records"]
        assert r["surface_layout_resolved"].asstr()[0] == "symmetric"
        assert r["water_bulk_potential_top_ev"][0] == pytest.approx(8)
        if missing_bottom:
            assert r["status"].asstr()[0] == "incomplete"
            for field in ("water_bulk_potential_ev", "electrode_cshe_ev",
                          "water_bulk_window_bottom_min_A", "water_bulk_window_bottom_max_A"):
                assert np.isnan(r[field][0])
            assert "missing" in r["water_bulk_provenance"].asstr()[0]
        else:
            assert r["status"].asstr()[0] == "ok"
            assert r["water_bulk_potential_bottom_ev"][0] == pytest.approx(2)
            assert r["electrode_cshe_ev"][0] == pytest.approx(5 - 1 + 0.81)
            assert r["water_bulk_window_bottom_max_A"][0] < r["water_bulk_window_top_min_A"][0]
            assert "padding relaxed" in r["water_bulk_provenance"].asstr()[0]
