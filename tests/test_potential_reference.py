from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms

from linak.analysis.potential import (
    BOHR_TO_ANG,
    HARTREE_TO_EV,
    PotentialConfig,
    PotentialHdf5Appender,
    compute_potential_record,
    compute_potential_records,
    cube_xyavg_vs_dataset,
    load_potential_records,
    write_potential_records,
)
from linak.analysis.potential_diagnostics import plot_potential_diagnostics
from linak.cube_io import CubeDataset, write_cube_file
from linak.cli import main


@pytest.fixture(autouse=True)
def restore_log_level():
    import logging

    logger = logging.getLogger("linak")
    previous = logger.level
    yield
    logger.setLevel(previous)


def water_cube(
    path,
    *,
    shift=0,
    top=26,
    bottom=14,
    single=False,
    split=False,
    profile=None,
    origin=0,
    step=False,
):
    symbols, positions = [], []
    for z in np.linspace(bottom, top, 7):
        for x, y in ((2, 2), (7, 2), (2, 7), (7, 7)):
            symbols.append("Pt")
            positions.append([x, y, z])
    if step:
        symbols.append("Pt")
        positions.append([4, 4, top + 1])
        top += 1
    gap = 40 - (top - bottom)
    for plane, distance in enumerate(np.arange(1.5, gap - 1, 1)):
        if single and distance > 11:
            continue
        if split and 9 < distance < gap - 9:
            continue
        xy_shift = 2.5 * (plane % 2)
        for x, y in ((2, 2), (7, 2), (2, 7), (7, 7)):
            x, y = (x + xy_shift) % 10, (y + xy_shift) % 10
            z = (top + distance) % 40
            symbols.extend(("O", "H", "H"))
            positions.extend(([x, y, z], [x + 0.6, y, z + 0.5], [x - 0.6, y, z + 0.5]))
    atoms = Atoms(symbols, positions=positions, cell=[10, 10, 40], pbc=True)
    atoms.positions[:, 2] += shift
    atoms.wrap()
    atoms.positions[:, 2] += origin
    z = origin + np.arange(400) * 0.1
    s = (z - origin - shift - top) % 40
    values = np.full(400, 4.0) if profile is None else profile(s, gap)
    return CubeDataset(
        comment_1="CP2K",
        comment_2="HARTREE POTENTIAL",
        natoms_signed=len(atoms),
        origin_bohr=np.array([0, 0, origin]) / BOHR_TO_ANG,
        grid_counts_signed=np.array([2, 3, 400]),
        grid_vectors_bohr=np.diag([5, 10 / 3, 0.1]) / BOHR_TO_ANG,
        atom_numbers=atoms.numbers,
        atom_charges=np.zeros(len(atoms)),
        atom_positions_bohr=atoms.positions / BOHR_TO_ANG,
        values=np.broadcast_to(values, (2, 3, 400)).copy() / HARTREE_TO_EV,
        source_path=str(Path(path).resolve()),
    )


def config_for(cubes, **kwargs):
    return PotentialConfig(fermi_map={cube.source_path: 1.0 for cube in cubes}, **kwargs)


@pytest.mark.parametrize("shift", [0, 7.3, 20, 39.8])
@pytest.mark.parametrize("step", [False, True])
def test_periodic_connected_water_reference_is_translation_invariant(tmp_path, shift, step):
    cube = water_cube(tmp_path / "hartree.cube", shift=shift, step=step)
    record = compute_potential_record(cube, config=config_for([cube]))
    assert record.reference_valid, record.reference_rejection_reason
    assert record.surface_layout_resolved == "symmetric"
    assert record.water_bulk_potential_ev == pytest.approx(4)
    assert record.electrode_cshe_ev == pytest.approx(4 - 1 - 0.65)
    assert record.requested_padding_ang == 5
    assert record.window_width_A >= 4
    assert record.window_grid_points >= 5


def test_asymmetric_single_interface_uses_water_distribution(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube", single=True)
    record = compute_potential_record(cube, config=config_for([cube], water_padding_ang=1))
    assert record.surface_layout_resolved == "single-top"
    assert record.reference_valid, record.reference_rejection_reason


def test_padding_is_never_halved_or_reduced(tmp_path):
    cube = water_cube(
        tmp_path / "hartree.cube", profile=lambda s, gap: 4 + 0.003 * (s - gap / 2) ** 2
    )
    records = [
        compute_potential_record(cube, config=config_for([cube], water_padding_ang=p))
        for p in (4, 8, 5, 10, 20)
    ]
    for padding, record in zip((4, 8, 5, 10, 20), records):
        assert record.requested_padding_ang == padding
        assert json.loads(record.reference_plan_json)["parameters"]["water_padding_ang"] == padding
    assert records[0].window_min_A != records[1].window_min_A
    assert records[2].window_min_A != records[3].window_min_A
    assert records[0].window_mean_ev != pytest.approx(records[1].window_mean_ev)
    assert not records[-1].reference_valid
    assert records[-1].electrode_cshe_ev is None
    assert "padding was not reduced" in records[-1].reference_rejection_reason


@pytest.mark.parametrize("gap_scale", [0.8, 1.1])
def test_reused_plan_preserves_angstrom_padding_when_gap_changes(tmp_path, gap_scale):
    from linak.analysis.potential_reference import ReferenceGeometry, evaluate_reference

    cube = water_cube(tmp_path / "hartree.cube")
    config = config_for([cube])
    initial = compute_potential_record(cube, config=config)
    plan = json.loads(initial.reference_plan_json)
    length = 28 * gap_scale
    period = 12 + length
    geometry = ReferenceGeometry(
        origin=0,
        period=period,
        area=100,
        start=26 % period,
        length=length,
        water_s=initial.profile["water_s_A"] * gap_scale,
        slab_bottom=14,
        slab_top=26 % period,
        layout="symmetric",
        boundary="periodic",
    )
    z = np.arange(0, period, 0.1)
    result, _ = evaluate_reference(z, np.full(len(z), 4.0), geometry, plan, config)
    support_lo, support_hi = np.asarray(plan["water_support_bounds_u"]) * length
    assert result["valid"], result["reasons"]
    assert result["min_A"] - support_lo == pytest.approx(5)
    assert support_hi - result["max_A"] == pytest.approx(5)


def test_manual_fixed_window_crosses_periodic_boundary_without_reselection(tmp_path):
    cubes = [water_cube(tmp_path / f"{i}.cube") for i in range(2)]
    config = config_for(cubes, bulk_window=(35, 5), water_padding_ang=0)
    records, failures = compute_potential_records(cubes, config=config)
    assert not failures
    assert all(r.reference_valid for r in records)
    assert records[0].window_min_A == records[1].window_min_A
    assert records[0].window_max_A > 40
    assert records[0].window_mean_ev == pytest.approx(4)


def test_one_batch_plan_and_saved_plan_are_reproducible(tmp_path):
    cubes = [water_cube(tmp_path / f"{i}.cube", shift=i * 0.7) for i in range(3)]
    config = config_for(cubes)
    records, failures = compute_potential_records(cubes, config=config)
    assert not failures
    assert len({r.reference_plan_json for r in records}) == 1
    plan = json.loads(records[0].reference_plan_json)
    rerun = compute_potential_record(cubes[1], config=replace(config, window_plan=plan))
    assert rerun.window_min_A == records[1].window_min_A
    assert rerun.window_max_A == records[1].window_max_A
    assert rerun.window_mean_ev == records[1].window_mean_ev
    with pytest.raises(ValueError, match="parameters differ"):
        compute_potential_record(
            cubes[1], config=replace(config, window_plan=plan, water_padding_ang=8)
        )


def test_disconnected_reservoirs_are_not_ranked_or_switched(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube", split=True)
    record = compute_potential_record(cube, config=config_for([cube], water_padding_ang=0))
    assert not record.reference_valid
    assert "connected water reservoir" in record.reference_rejection_reason
    assert record.window_mean_ev is None


@pytest.mark.parametrize("window", [(35, 36), (35.001, 35.002), (15, 22), (0, 0)])
def test_invalid_fixed_windows_are_rejected_not_repaired(tmp_path, window):
    cube = water_cube(tmp_path / "hartree.cube")
    record = compute_potential_record(
        cube, config=config_for([cube], water_padding_ang=0, bulk_window=window)
    )
    assert not record.reference_valid
    assert record.reference_rejection_reason
    assert record.water_bulk_potential_ev is None


@pytest.mark.parametrize(
    "field,profile",
    [
        ("gradient", lambda s, gap: 4 + 0.1 * s),
        ("std_ev", lambda s, gap: 4 + np.sin(2 * s)),
        ("Nonfinite", lambda s, gap: np.full_like(s, np.nan)),
    ],
)
def test_geometric_windows_do_not_guarantee_electrostatic_validity(tmp_path, field, profile):
    cube = water_cube(tmp_path / "hartree.cube", profile=profile)
    record = compute_potential_record(cube, config=config_for([cube]))
    assert not record.reference_valid
    assert field in record.reference_rejection_reason
    assert record.window_grid_points > 0
    assert record.water_bulk_potential_ev is None
    assert record.electrode_cshe_ev is None


def test_gauge_shift_leaves_electrode_potential_and_quality_unchanged(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube")
    config = config_for([cube])
    a = compute_potential_record(cube, config=config)
    shifted = replace(cube, values=cube.values + 7 / HARTREE_TO_EV)
    b = compute_potential_record(shifted, config=replace(config, fermi_map={cube.source_path: 8}))
    assert b.electrode_cshe_ev == pytest.approx(a.electrode_cshe_ev)
    assert b.window_std_ev == pytest.approx(a.window_std_ev, abs=1e-12)
    assert b.window_gradient_ev_A == pytest.approx(a.window_gradient_ev_A, abs=1e-12)


def test_cube_ordering_and_units_are_explicit(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube")
    values = (
        np.arange(2)[:, None, None] * 100
        + np.arange(3)[None, :, None] * 10
        + np.arange(400)[None, None, :]
    )
    cube = replace(cube, values=values)
    z, profile, _ = cube_xyavg_vs_dataset(cube)
    np.testing.assert_allclose(profile, 60 + np.arange(400))
    angstrom = replace(
        cube,
        grid_counts_signed=-cube.grid_counts_signed,
        grid_vectors_bohr=cube.grid_vectors_bohr * BOHR_TO_ANG,
        atom_positions_bohr=cube.atom_positions_bohr * BOHR_TO_ANG,
        origin_bohr=cube.origin_bohr * BOHR_TO_ANG,
    )
    z2, profile2, _ = cube_xyavg_vs_dataset(angstrom)
    np.testing.assert_allclose(z, z2)
    np.testing.assert_allclose(profile, profile2)
    with pytest.raises(ValueError, match="unit convention"):
        cube_xyavg_vs_dataset(replace(cube, grid_counts_signed=np.array([-2, 3, 400])))


def test_origin_does_not_change_auto_reference(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube", origin=6.2)
    record = compute_potential_record(cube, config=config_for([cube]))
    assert record.reference_valid, record.reference_rejection_reason
    assert record.water_bulk_potential_ev == pytest.approx(4)


def test_source_specific_fermi_association_is_required(tmp_path):
    cubes = [water_cube(tmp_path / f"{i}.cube") for i in range(2)]
    (tmp_path / "output.out").write_text("Fermi energy [eV]: 1\n")
    records, _ = compute_potential_records(cubes, config=PotentialConfig())
    assert all(r.reference_valid and r.electrode_cshe_ev is None for r in records)
    assert all("Multiple snapshots share" in r.fermi_provenance for r in records)
    assert all(r.status == "incomplete" for r in records)
    (tmp_path / "output.out").write_text(
        "MD| Step number 1\nFermi energy [eV]: 1\nMD| Step number 2\nFermi energy [eV]: 2\n"
    )
    record = compute_potential_record(cubes[0], config=PotentialConfig())
    assert "Multi-step" in record.fermi_provenance


def test_roundtrip_diagnostics_and_plan_mismatch_append_is_atomic(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube")
    record = compute_potential_record(cube, config=config_for([cube]))
    path = tmp_path / "potential.h5"
    write_potential_records([record], path)
    loaded = load_potential_records(path, include_profiles=True)[0]
    assert loaded.reference_plan_json == record.reference_plan_json
    assert loaded.window_mean_ev == record.window_mean_ev
    assert loaded.reference_valid
    np.testing.assert_array_equal(loaded.profile["selected"], record.profile["selected"])
    plot = plot_potential_diagnostics(path, tmp_path / "diagnostics.png")
    assert plot.stat().st_size > 1000
    other = compute_potential_record(cube, config=config_for([cube], water_padding_ang=8))
    with PotentialHdf5Appender(output=path) as writer:
        with pytest.raises(ValueError, match="different reference plan"):
            writer.append_record(other)
    assert len(load_potential_records(path)) == 1


def test_cli_fixed_window_diagnostics_and_rejection(tmp_path):
    cube = water_cube(tmp_path / "v_hartree.cube")
    write_cube_file(cube, cube.source_path)
    (tmp_path / "output.out").write_text("Fermi energy [eV]: 1\n")
    output = tmp_path / "result.potential.h5"
    plan = tmp_path / "reference.json"
    assert (
        main(
            [
                "compute",
                "potential",
                cube.source_path,
                "--bulk-window",
                "35",
                "5",
                "--output",
                str(output),
                "--save-window-plan",
                str(plan),
                "--strict",
            ]
        )
        == 0
    )
    assert output.with_suffix(".diagnostics.png").exists()
    assert json.loads(plan.read_text())["bounds"] == [35, 5]
    with h5py.File(output) as f:
        assert f.attrs["analysis_schema_version"] == 3
    assert (
        main(
            [
                "compute",
                "potential",
                cube.source_path,
                "--water-padding-ang",
                "20",
                "--output",
                str(tmp_path / "invalid.h5"),
                "--strict",
                "--no-diagnostics",
            ]
        )
        == 1
    )


@pytest.mark.parametrize("single,boundary", [(True, "open"), (True, "periodic")])
def test_single_bottom_and_open_boundaries(tmp_path, single, boundary):
    cube = water_cube(tmp_path / "hartree.cube", single=single)
    positions = cube.atom_positions_bohr * BOHR_TO_ANG
    positions[:, 2] = (-positions[:, 2]) % 40
    cube = replace(cube, atom_positions_bohr=positions / BOHR_TO_ANG)
    record = compute_potential_record(
        cube,
        config=config_for([cube], water_padding_ang=1, boundary=boundary, surface_elements=("Pt",)),
    )
    assert record.surface_layout_resolved == "single-bottom"
    assert record.reference_valid, record.reference_rejection_reason


def test_open_two_reservoirs_require_explicit_selection(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube")
    config = config_for([cube], boundary="open", surface_elements=("Pt",))
    auto = compute_potential_record(cube, config=config)
    assert not auto.reference_valid
    assert "disconnected open-boundary" in auto.reference_rejection_reason
    fixed = compute_potential_record(
        cube, config=replace(config, bulk_window=(2, 10), water_padding_ang=0)
    )
    assert fixed.reference_valid, fixed.reference_rejection_reason


def test_lost_water_invalidates_a_fixed_plan_instead_of_moving_it(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube")
    config = config_for([cube])
    initial = compute_potential_record(cube, config=config)
    plan = json.loads(initial.reference_plan_json)
    lo, hi = np.asarray(plan["bounds"]) * 28
    numbers, positions = cube.atom_numbers, cube.atom_positions_bohr * BOHR_TO_ANG
    keep = np.ones(len(numbers), dtype=bool)
    for index in np.flatnonzero(numbers == 8):
        s = (positions[index, 2] - 26) % 40
        if lo <= s <= hi:
            keep[index : index + 3] = False
    changed = replace(
        cube,
        natoms_signed=int(keep.sum()),
        atom_numbers=numbers[keep],
        atom_positions_bohr=cube.atom_positions_bohr[keep],
        atom_charges=cube.atom_charges[keep],
    )
    record = compute_potential_record(changed, config=replace(config, window_plan=plan))
    assert not record.reference_valid
    assert "water" in record.reference_rejection_reason.lower()
    assert record.window_min_A == initial.window_min_A
    assert record.window_max_A == initial.window_max_A


def test_sensitivity_is_reported_without_using_it_to_select_window(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube", profile=lambda s, gap: 4 + 0.01 * s)
    record = compute_potential_record(cube, config=config_for([cube]))
    sensitivity = json.loads(record.sensitivity_json)
    assert sensitivity["shift_plus"]["mean_ev"] - record.window_mean_ev == pytest.approx(
        0.01, abs=0.002
    )
    assert sensitivity["shift_minus"]["mean_ev"] - record.window_mean_ev == pytest.approx(
        -0.01, abs=0.002
    )
    assert record.reference_valid


def test_subgrid_window_motion_does_not_add_or_drop_a_whole_slice(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube", profile=lambda s, gap: 4 + 0.01 * s)
    a = compute_potential_record(
        cube, config=config_for([cube], bulk_window=(35, 5), water_padding_ang=0)
    )
    b = compute_potential_record(
        cube, config=config_for([cube], bulk_window=(35.001, 5.001), water_padding_ang=0)
    )
    assert a.reference_valid and b.reference_valid
    assert b.window_mean_ev - a.window_mean_ev == pytest.approx(0.00001, abs=1e-10)
    assert a.profile["averaging_weights"].sum() == pytest.approx(1)
    assert a.window_mean_ev == pytest.approx(
        a.profile["averaging_weights"] @ a.profile["hartree_ev"]
    )


def test_batch_callbacks_preserve_order_and_time_across_append(tmp_path):
    cubes = [water_cube(tmp_path / f"{i}.cube", shift=i * 0.3) for i in range(2)]
    seen = []
    records, failures = compute_potential_records(
        cubes,
        config=config_for(cubes, snapshot_interval_fs=2),
        threads=2,
        start_index=10,
        on_record=lambda r: seen.append(r.source),
    )
    assert not failures
    assert seen == [cube.source_path for cube in cubes]
    assert [record.time_fs for record in records] == [20, 22]


def test_saved_plan_can_be_reused_from_cli_without_repeating_its_selection(tmp_path):
    cube = water_cube(tmp_path / "v_hartree.cube")
    write_cube_file(cube, cube.source_path)
    (tmp_path / "output.out").write_text("Fermi energy [eV]: 1\n")
    plan = tmp_path / "plan.json"
    a, b = tmp_path / "a.potential.h5", tmp_path / "b.potential.h5"
    assert (
        main(
            [
                "compute",
                "potential",
                cube.source_path,
                "--bulk-window",
                "35",
                "5",
                "--output",
                str(a),
                "--save-window-plan",
                str(plan),
                "--no-diagnostics",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "compute",
                "potential",
                cube.source_path,
                "--window-plan",
                str(plan),
                "--output",
                str(b),
                "--no-diagnostics",
            ]
        )
        == 0
    )
    first, second = load_potential_records(a)[0], load_potential_records(b)[0]
    assert first.reference_plan_json == second.reference_plan_json
    assert first.electrode_cshe_ev == second.electrode_cshe_ev
    plot = tmp_path / "plot.png"
    assert main(["plot", str(b), "--reference-diagnostics", "--output", str(plot)]) == 0
    assert plot.exists()


def test_append_cannot_silently_skip_changed_padding_or_offset(tmp_path):
    cube = water_cube(tmp_path / "v_hartree.cube")
    write_cube_file(cube, cube.source_path)
    (tmp_path / "output.out").write_text("Fermi energy [eV]: 1\n")
    output = tmp_path / "result.potential.h5"
    base = ["compute", "potential", cube.source_path, "--output", str(output), "--no-diagnostics"]
    assert main(base) == 0
    original = output.read_bytes()
    assert main([*base, "--water-padding-ang", "8"]) == 1
    assert output.read_bytes() == original
    assert main([*base, "--cshe-offset-ev", "0.81"]) == 1
    assert output.read_bytes() == original


def test_plan_checksum_and_validation_changes_are_not_silent(tmp_path):
    cube = water_cube(tmp_path / "hartree.cube")
    config = config_for([cube])
    record = compute_potential_record(cube, config=config)
    plan = json.loads(record.reference_plan_json)
    plan["bounds"][0] += 0.01
    with pytest.raises(ValueError, match="checksum"):
        compute_potential_record(cube, config=replace(config, window_plan=plan))
    output = tmp_path / "reference.h5"
    write_potential_records([record], output)
    changed = compute_potential_record(cube, config=replace(config, max_std_ev=0.5))
    with PotentialHdf5Appender(output=output) as writer:
        with pytest.raises(ValueError, match="different validation"):
            writer.append_record(changed)
    assert len(load_potential_records(output)) == 1


@pytest.mark.parametrize(
    "entry",
    [
        "Fermi energy [eV]: 1",
        "Fermi energy [a.u.]: 0.036749322175655",
        "Fermi energy: 0.036749322175655",
    ],
)
def test_fermi_units_are_consistent_with_hartree_cube_units(tmp_path, entry):
    cube = water_cube(tmp_path / "hartree.cube")
    (tmp_path / "output.out").write_text("SCF run converged\n" + entry + "\n")
    record = compute_potential_record(cube, config=PotentialConfig())
    assert record.reference_valid
    assert record.efermi_ev == pytest.approx(1, abs=1e-7)
    assert record.electrode_cshe_ev == pytest.approx(2.35, abs=1e-7)


@pytest.mark.parametrize(
    "entry", ["SCF run NOT converged\nFermi energy [eV]: 1", "Fermi energy [eV]: 1e999"]
)
def test_unusable_fermi_energies_do_not_produce_electrode_potentials(tmp_path, entry):
    cube = water_cube(tmp_path / "hartree.cube")
    (tmp_path / "output.out").write_text(entry + "\n")
    record = compute_potential_record(cube, config=PotentialConfig())
    assert record.reference_valid
    assert record.efermi_ev is None
    assert record.electrode_cshe_ev is None
    assert record.status == "incomplete"
