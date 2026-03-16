from pathlib import Path

from ase import Atoms
import pytest

from linak.resolution import (
    resolve_analysis_cell,
    resolve_analysis_timestep_fs,
)


def test_resolve_analysis_cell_uses_auto_inp(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text("ABC [angstrom] 10.0 11.0 12.0\n", encoding="utf-8")

    resolved = resolve_analysis_cell(trajectory)

    assert resolved.cell_angstrom == pytest.approx((10.0, 11.0, 12.0))
    assert resolved.source.startswith("auto-detected")


def test_resolve_analysis_cell_uses_auto_lmp(tmp_path):
    trajectory = tmp_path / "traj.dump"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.lmp").write_text("read_data system.data\n", encoding="utf-8")
    (tmp_path / "system.data").write_text(
        "LAMMPS data file\n\n"
        "2 atoms\n"
        "1 atom types\n\n"
        "0.0 8.0 xlo xhi\n"
        "0.0 9.0 ylo yhi\n"
        "0.0 10.0 zlo zhi\n",
        encoding="utf-8",
    )

    resolved = resolve_analysis_cell(trajectory)

    assert resolved.cell_angstrom == pytest.approx((8.0, 9.0, 10.0))
    assert resolved.source.startswith("auto-detected")


def test_resolve_analysis_cell_prefers_explicit_cell(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text("ABC [angstrom] 2.0 2.0 2.0\n", encoding="utf-8")
    explicit_input = tmp_path / "other.inp"
    explicit_input.write_text("ABC [angstrom] 4.0 4.0 4.0\n", encoding="utf-8")

    resolved = resolve_analysis_cell(
        trajectory,
        cell=(3.0, 3.0, 3.0),
        input_path=explicit_input,
    )

    assert resolved.cell_angstrom == pytest.approx((3.0, 3.0, 3.0))
    assert resolved.source == "explicit --cell"


def test_resolve_analysis_cell_prefers_explicit_input_over_auto(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text("ABC [angstrom] 2.0 2.0 2.0\n", encoding="utf-8")
    explicit_input = tmp_path / "external" / "other.inp"
    explicit_input.parent.mkdir(parents=True, exist_ok=True)
    explicit_input.write_text("ABC [angstrom] 4.0 4.0 4.0\n", encoding="utf-8")

    resolved = resolve_analysis_cell(trajectory, input_path=explicit_input)

    assert resolved.cell_angstrom == pytest.approx((4.0, 4.0, 4.0))
    assert resolved.source.startswith("explicit --input")


def test_resolve_analysis_cell_warns_when_sources_disagree(tmp_path, caplog):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text("ABC [angstrom] 2.0 2.0 2.0\n", encoding="utf-8")

    with caplog.at_level("WARNING", logger="linak.resolution"):
        resolved = resolve_analysis_cell(trajectory, cell=(1.0, 1.0, 1.0))

    assert resolved.cell_angstrom == pytest.approx((1.0, 1.0, 1.0))
    assert "Cell sources disagree" in caplog.text


def test_resolve_analysis_cell_raises_when_no_source_is_available(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not resolve cell dimensions"):
        resolve_analysis_cell(trajectory)


def test_resolve_analysis_timestep_uses_auto_inp(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text(
        "TIMESTEP [fs] 0.5\n&TRAJECTORY\n  &EACH\n    MD 5\n  &END EACH\n&END TRAJECTORY\n",
        encoding="utf-8",
    )

    resolved = resolve_analysis_timestep_fs(trajectory)

    assert resolved.frame_timestep_fs == pytest.approx(2.5)
    assert resolved.source.startswith("auto-detected")


def test_resolve_analysis_timestep_uses_auto_lmp(tmp_path):
    trajectory = tmp_path / "traj.dump"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.lmp").write_text(
        "units metal\ntimestep 0.001\ndump d all custom 10 lammps.dump id type element xu yu zu\n",
        encoding="utf-8",
    )

    resolved = resolve_analysis_timestep_fs(trajectory)

    assert resolved.frame_timestep_fs == pytest.approx(10.0)
    assert resolved.source.startswith("auto-detected")


def test_resolve_analysis_timestep_prefers_explicit_over_auto(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text(
        "TIMESTEP [fs] 0.5\n&TRAJECTORY\n  &EACH\n    MD 5\n  &END EACH\n&END TRAJECTORY\n",
        encoding="utf-8",
    )
    explicit_input = tmp_path / "other.inp"
    explicit_input.write_text("TIMESTEP [fs] 0.2\n", encoding="utf-8")

    resolved = resolve_analysis_timestep_fs(
        trajectory,
        timestep_fs=2.0,
        input_path=explicit_input,
    )

    assert resolved.frame_timestep_fs == pytest.approx(2.0)
    assert resolved.source == "explicit --timestep-fs"


def test_resolve_analysis_timestep_warns_when_sources_disagree(tmp_path, caplog):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text(
        "TIMESTEP [fs] 0.5\n&TRAJECTORY\n  &EACH\n    MD 5\n  &END EACH\n&END TRAJECTORY\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="linak.resolution"):
        resolved = resolve_analysis_timestep_fs(trajectory, timestep_fs=1.0)

    assert resolved.frame_timestep_fs == pytest.approx(1.0)
    assert "Timestep sources disagree" in caplog.text


def test_resolve_analysis_timestep_prefers_metadata_over_input(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    (tmp_path / "input.inp").write_text(
        "TIMESTEP [fs] 0.5\n&TRAJECTORY\n  &EACH\n    MD 5\n  &END EACH\n&END TRAJECTORY\n",
        encoding="utf-8",
    )

    frame0 = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    frame1 = Atoms("H", positions=[[0.0, 0.0, 0.1]])
    frame0.info["time_fs"] = 0.0
    frame1.info["time_fs"] = 1.25

    resolved = resolve_analysis_timestep_fs(trajectory, frames=[frame0, frame1])

    assert resolved.frame_timestep_fs == pytest.approx(1.25)
    assert resolved.source == "trajectory metadata"


def test_resolve_analysis_timestep_raises_when_no_source_is_available(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not resolve timestep"):
        resolve_analysis_timestep_fs(trajectory)
