from __future__ import annotations

from pathlib import Path

import h5py
from ase import Atoms
from ase.io import write

from linak.cli import main
from linak.cube_io import read_cube_sources
from linak.gui.actions import ActionContext, ActionRegistry
from linak.gui.detection import detect_project_item
from linak.out_h5 import (
    export_out_h5_component,
    inspect_out_h5,
    is_linak_out_hdf5,
    pack_simulation_directory,
    read_out_h5_cube_datasets,
    read_out_h5_trajectory,
)
from linak.trajectory.io import read_trajectory


def _write_xyz(path: Path) -> None:
    frames = [
        Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]], cell=[4.0, 4.0, 4.0], pbc=True),
        Atoms("OH", positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]], cell=[4.0, 4.0, 4.0], pbc=True),
    ]
    write(path, frames, format="extxyz")


def _write_cube(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CP2K CUBE FILE\n")
        handle.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        handle.write("2 0.0 0.0 0.0\n")
        handle.write("1 1.0 0.0 0.0\n")
        handle.write("1 0.0 1.0 0.0\n")
        handle.write("2 0.0 0.0 1.0\n")
        handle.write("8 0.0 0.0 0.0 0.0\n")
        handle.write("1 0.0 0.0 0.0 1.0\n")
        handle.write(" 1.00000000E-01 2.00000000E-01\n")


def test_pack_simulation_directory_writes_schema_and_metadata(tmp_path):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_xyz(sim / "traj.xyz")
    _write_cube(sim / "density.cube")
    (sim / "output.out").write_text(
        " CP2K| version string\n GLOBAL| Run type ENERGY\n PROGRAM ENDED AT 2026-01-01 00:00:00\n",
        encoding="utf-8",
    )

    result = pack_simulation_directory(sim, tmp_path / "project" / "run.out.h5")

    assert is_linak_out_hdf5(result.output_path)
    summary = inspect_out_h5(result.output_path)
    assert summary.schema_version == 1
    assert summary.trajectory_present is True
    assert summary.frame_count == 2
    assert summary.cube_count == 1
    assert summary.cp2k_output_count == 1
    with h5py.File(result.output_path, "r") as handle:
        assert handle.attrs["linak_format"] == "linak-out-hdf5"
        assert "provenance" in handle
        assert "system" in handle


def test_out_h5_accessors_and_exports_roundtrip(tmp_path):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_xyz(sim / "traj.xyz")
    _write_cube(sim / "density.cube")
    container = pack_simulation_directory(sim, tmp_path / "run.out.h5").output_path

    frames = read_out_h5_trajectory(container)
    cubes = read_out_h5_cube_datasets(container)

    assert len(frames) == 2
    assert frames[0].get_chemical_symbols() == ["O", "H"]
    assert len(cubes) == 1
    assert cubes[0].values.shape == (1, 1, 2)

    exported_traj = export_out_h5_component(container, "trajectory", tmp_path / "export.traj.h5")
    exported_cube = export_out_h5_component(container, "cube", tmp_path / "export.cube.h5")

    assert len(read_trajectory(exported_traj)) == 2
    assert len(read_cube_sources(exported_cube)) == 1


def test_cli_apply_pack_dry_run_and_real_pack(tmp_path, capsys):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_xyz(sim / "traj.xyz")

    dry_rc = main(["--log-level", "INFO", "apply", "pack", str(sim), "--dry-run"])
    assert dry_rc == 0
    assert "trajectory candidates: 1" in capsys.readouterr().err

    output = tmp_path / "workspace" / "run.out.h5"
    rc = main(["--log-level", "ERROR", "apply", "pack", str(sim), "--output", str(output)])

    assert rc == 0
    assert output.exists()
    assert is_linak_out_hdf5(output)
    assert str(output) in capsys.readouterr().out


def test_compute_density_accepts_out_h5_input(tmp_path):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_xyz(sim / "traj.xyz")
    container = pack_simulation_directory(sim, tmp_path / "run.out.h5").output_path
    output = tmp_path / "density.h5"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(container),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.5",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_compute_potential_accepts_out_h5_input(tmp_path):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_cube(sim / "density.cube")
    (sim / "output.out").write_text("Fermi energy: -0.100000\n", encoding="utf-8")
    container = pack_simulation_directory(sim, tmp_path / "run.out.h5").output_path
    output = tmp_path / "potential.h5"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(container),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_gui_detection_and_actions_treat_out_h5_as_primary_input(tmp_path):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_xyz(sim / "traj.xyz")
    container = pack_simulation_directory(sim, tmp_path / "run.out.h5").output_path

    item = detect_project_item(container, origin="generated")
    action_ids = {action.action_id for action in ActionRegistry().available_for(item)}

    assert item.item_type == "out_hdf5"
    assert item.metadata["frame_count"] == 2
    assert {"density", "msd", "rdf", "export_out_trajectory"} <= action_ids


def test_gui_pack_action_accepts_simulation_directory(tmp_path):
    sim = tmp_path / "run"
    sim.mkdir()
    _write_xyz(sim / "traj.xyz")

    item = detect_project_item(sim, origin="external")
    action_ids = {action.action_id for action in ActionRegistry().available_for(item)}

    assert item.item_type == "simulation_directory"
    assert "pack_out_h5" in action_ids


def test_gui_pack_action_reports_structured_progress(tmp_path):
    sim = tmp_path / "run"
    project = tmp_path / "workspace"
    sim.mkdir()
    project.mkdir()
    _write_xyz(sim / "traj.xyz")
    item = detect_project_item(sim, origin="external")
    action = ActionRegistry().by_id("pack_out_h5")
    progress_events: list[tuple[str, int | None, int | None]] = []

    result = action.backend(
        ActionContext(
            project_dir=project,
            item=item,
            settings={
                "output_name": "run.out.h5",
                "overwrite": False,
                "include": "",
                "exclude": "",
                "drop": "",
            },
            log=lambda _level, _message: None,
            progress=lambda label, current, total: progress_events.append(
                (label, current, total)
            ),
        )
    )

    assert result.output_paths == (project / "run.out.h5",)
    assert progress_events[0] == ("Discovering files", 0, 6)
    assert progress_events[-1] == ("Finished", 1, 1)
    assert ("Finished", 6, 6) in progress_events
