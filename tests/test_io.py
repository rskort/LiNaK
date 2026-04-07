from pathlib import Path

import h5py
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

import linak.trajectory.io as io_mod
from linak.trajectory.io import (
    TrajectoryStoredMetadata,
    default_trajectory_hdf5_output_path,
    is_linak_trajectory_hdf5,
    read_trajectory_hdf5_metadata,
    read_trajectory_hdf5_surface_cache,
    read_trajectory,
    read_trajectory_chunks,
    write_trajectory,
)


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


def _write_lammps_input(path: Path, *, dump_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "units metal",
                "timestep 0.001",
                f"dump d1 all custom 10 {dump_name} id type element xu yu zu",
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


def test_read_trajectory_dump_progress_uses_exact_total(tmp_path, monkeypatch):
    dump_path = tmp_path / "traj.dump"
    _write_lammps_dump(dump_path)
    captured: dict[str, int | None] = {}

    class _DummyProgressBar:
        def __init__(self, *, desc, total=None, unit="it", **_kwargs):
            captured["desc"] = desc
            captured["total"] = total
            captured["unit"] = unit

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            captured["last_update"] = n

    monkeypatch.setattr(io_mod, "ProgressBar", _DummyProgressBar)

    frames = read_trajectory(dump_path)

    assert len(frames) == 2
    assert captured["desc"] == "Reading trajectory"
    assert captured["total"] == 2
    assert captured["unit"] == "frame"


def test_read_trajectory_lammps_input_progress_uses_resolved_dump_total(tmp_path, monkeypatch):
    dump_path = tmp_path / "traj.dump"
    input_path = tmp_path / "in.lmp"
    _write_lammps_dump(dump_path)
    _write_lammps_input(input_path, dump_name=dump_path.name)
    captured: dict[str, int | None] = {}

    class _DummyProgressBar:
        def __init__(self, *, total=None, **_kwargs):
            captured["total"] = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            captured["last_update"] = n

    monkeypatch.setattr(io_mod, "ProgressBar", _DummyProgressBar)

    frames = read_trajectory(input_path)

    assert len(frames) == 2
    assert captured["total"] == 2


def test_read_trajectory_extxyz_progress_uses_exact_total(tmp_path, monkeypatch):
    path = tmp_path / "traj.xyz"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[1.0, 0.0, 0.0]]),
        Atoms("O", positions=[[2.0, 0.0, 0.0]]),
    ]
    write(path, frames, format="extxyz")
    captured: dict[str, int | None] = {}

    class _DummyProgressBar:
        def __init__(self, *, total=None, **_kwargs):
            captured["total"] = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            captured["last_update"] = n

    monkeypatch.setattr(io_mod, "ProgressBar", _DummyProgressBar)

    loaded = read_trajectory(path)

    assert len(loaded) == 3
    assert captured["total"] is None


def test_read_trajectory_generic_ase_uses_single_pass_with_unknown_total(tmp_path, monkeypatch):
    path = tmp_path / "traj.traj"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[1.0, 0.0, 0.0]]),
    ]
    write(path, frames)
    captured = {"total": None, "calls": 0}
    original_iread = io_mod.iread

    def _counting_iread(*args, **kwargs):
        captured["calls"] += 1
        yield from original_iread(*args, **kwargs)

    class _DummyProgressBar:
        def __init__(self, *, total=None, **_kwargs):
            captured["total"] = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            captured["last_update"] = n

    monkeypatch.setattr(io_mod, "iread", _counting_iread)
    monkeypatch.setattr(io_mod, "ProgressBar", _DummyProgressBar)

    loaded = read_trajectory(path)

    assert len(loaded) == 2
    assert captured["total"] is None
    assert captured["calls"] == 1


def test_read_trajectory_xyz_rejects_malformed_input(tmp_path):
    path = tmp_path / "broken.xyz"
    path.write_text("1\ncomment only\n", encoding="utf-8")

    try:
        read_trajectory(path)
    except Exception:
        pass
    else:
        raise AssertionError("Malformed XYZ input should raise an error")


def test_read_trajectory_chunks_yields_expected_chunk_sizes(tmp_path):
    path = tmp_path / "traj.traj"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[1.0, 0.0, 0.0]]),
        Atoms("O", positions=[[2.0, 0.0, 0.0]]),
        Atoms("O", positions=[[3.0, 0.0, 0.0]]),
        Atoms("O", positions=[[4.0, 0.0, 0.0]]),
    ]
    write(path, frames)

    chunks = list(read_trajectory_chunks(path, chunk_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert chunks[0][0].positions[0, 0] == 0.0
    assert chunks[-1][0].positions[0, 0] == 4.0


def test_read_trajectory_chunks_rejects_non_positive_chunk_size(tmp_path):
    path = tmp_path / "traj.traj"
    write(path, [Atoms("O", positions=[[0.0, 0.0, 0.0]])])

    try:
        list(read_trajectory_chunks(path, chunk_size=0))
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("Non-positive chunk_size should raise ValueError")


def test_write_and_read_linak_trajectory_hdf5_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "traj.traj.h5"
    frames = [
        Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], cell=[5.0, 6.0, 7.0], pbc=True),
        Atoms("OH", positions=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 6.0, 7.0], pbc=True),
    ]
    frames[0].info["timestep"] = 0
    frames[1].info["timestep"] = 10
    captured: dict[str, int | None] = {}

    class _DummyProgressBar:
        def __init__(self, *, total=None, **_kwargs):
            captured["total"] = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            captured["last_update"] = n

    write_trajectory(frames, path, source_path=tmp_path / "traj.xyz", source_format="xyz")
    assert is_linak_trajectory_hdf5(path) is True

    monkeypatch.setattr(io_mod, "ProgressBar", _DummyProgressBar)
    loaded = read_trajectory(path)

    assert len(loaded) == 2
    assert captured["total"] == 2
    assert loaded[0].get_chemical_symbols() == ["O", "H"]
    assert loaded[1].positions[1, 0] == 1.0
    assert loaded[1].info["timestep"] == 10
    assert tuple(bool(value) for value in loaded[0].get_pbc()) == (True, True, True)


def test_write_and_read_linak_trajectory_hdf5_roundtrip_stored_metadata_and_constraints(tmp_path):
    path = tmp_path / "traj.traj.h5"
    frames = [
        Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        Atoms("OH", positions=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ]
    for frame in frames:
        frame.set_constraint(FixAtoms(indices=[1]))

    write_trajectory(
        frames,
        path,
        metadata=TrajectoryStoredMetadata(
            input_path=tmp_path / "input.inp",
            input_format="inp",
            cell_angstrom=(5.0, 6.0, 7.0),
            cell_source="simulation input",
            frame_timestep_fs=2.5,
            md_timestep_fs=0.5,
            trajectory_stride_md=5,
            timestep_source="simulation input",
            fixed_atom_indices=(1,),
            fixed_atoms_source="simulation input",
            pbc_applied=True,
            pbc_cell_angstrom=(5.0, 6.0, 7.0),
            pbc_source="simulation input",
            coordinate_basis="pbc-wrapped",
        ),
    )

    metadata = read_trajectory_hdf5_metadata(path)
    loaded = read_trajectory(path)

    assert metadata is not None
    assert metadata.input_path == (tmp_path / "input.inp").resolve()
    assert metadata.cell_angstrom == (5.0, 6.0, 7.0)
    assert metadata.frame_timestep_fs == 2.5
    assert metadata.trajectory_stride_md == 5
    assert metadata.fixed_atom_indices == (1,)
    assert metadata.pbc_applied is True
    assert metadata.pbc_cell_angstrom == (5.0, 6.0, 7.0)
    assert metadata.coordinate_basis == "pbc-wrapped"
    assert loaded[0].constraints
    assert tuple(int(index) for index in loaded[0].constraints[0].get_indices()) == (1,)


def test_read_trajectory_hdf5_surface_cache_rejects_malformed_available_cache(tmp_path):
    path = tmp_path / "traj.traj.h5"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[0.0, 0.0, 1.0]]),
    ]
    write_trajectory(
        frames,
        path,
        metadata=TrajectoryStoredMetadata(
            surface_cache_status="unavailable",
            surface_cache_axis="z",
            surface_cache_mode="auto",
        ),
    )
    with h5py.File(path, "a") as handle:
        surface_cache = handle["metadata/surface_cache"]
        surface_cache.attrs["status"] = "available"
        surface_cache.attrs["axis"] = "z"
        surface_cache.attrs["surface_mode"] = "auto"

    with pytest.raises(ValueError, match="surface cache is malformed"):
        read_trajectory_hdf5_surface_cache(
            path,
            axis="z",
            surface_mode="auto",
            surface_elements=None,
            include_fixed_surface_atoms=False,
            rough_surface_envelope_A=None,
            frame_count=2,
        )


def test_read_trajectory_hdf5_chunks_use_exact_total(tmp_path, monkeypatch):
    path = tmp_path / "traj.traj.h5"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[1.0, 0.0, 0.0]]),
        Atoms("O", positions=[[2.0, 0.0, 0.0]]),
    ]
    write_trajectory(frames, path)
    captured: dict[str, int | None] = {}

    class _DummyProgressBar:
        def __init__(self, *, total=None, **_kwargs):
            captured["total"] = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, n=1):
            captured["last_update"] = n

    monkeypatch.setattr(io_mod, "ProgressBar", _DummyProgressBar)
    chunks = list(read_trajectory_chunks(path, chunk_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert captured["total"] == 3


def test_write_trajectory_hdf5_rejects_variable_topology(tmp_path):
    path = tmp_path / "traj.traj.h5"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
    ]

    try:
        write_trajectory(frames, path)
    except ValueError as exc:
        assert "fixed topology only" in str(exc)
    else:
        raise AssertionError("Variable topology conversion should raise ValueError")


def test_default_trajectory_hdf5_output_path_uses_traj_suffix(tmp_path):
    path = tmp_path / "traj.xyz"

    resolved = default_trajectory_hdf5_output_path(path)

    assert resolved == tmp_path.resolve() / "traj.traj.h5"
