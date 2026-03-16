from pathlib import Path

from linak.trajectory.io import read_trajectory


def _write_lammps_dump(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "0",
                "ITEM: NUMBER OF ATOMS",
                "1",
                "ITEM: BOX BOUNDS pp pp pp",
                "0.0 5.0",
                "0.0 5.0",
                "0.0 5.0",
                "ITEM: ATOMS id type element xu yu zu",
                "1 1 O 1.0 1.0 1.0",
                "ITEM: TIMESTEP",
                "10",
                "ITEM: NUMBER OF ATOMS",
                "1",
                "ITEM: BOX BOUNDS pp pp pp",
                "0.0 5.0",
                "0.0 5.0",
                "0.0 5.0",
                "ITEM: ATOMS id type element xu yu zu",
                "1 1 O 2.0 2.0 2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_read_trajectory_dump_does_not_depend_on_ase_iread(tmp_path, monkeypatch):
    dump_path = tmp_path / "traj.dump"
    _write_lammps_dump(dump_path)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("iread should not be called for .dump files")

    monkeypatch.setattr("linak.trajectory.io.iread", _raise_if_called)

    frames = read_trajectory(dump_path)

    assert len(frames) == 2
    assert frames[0].info.get("timestep") == 0
    assert frames[1].info.get("timestep") == 10

