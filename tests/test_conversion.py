from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

import numpy as np
import pytest

from linak.conversion import (
    CONVERSION_REGISTRY,
    CubeConversionOptions,
    TrajectoryConversionOptions,
)
from linak.storage.hdf5_utils import read_linak_hdf5_profiles
from linak.trajectory.io import is_linak_trajectory_hdf5, read_trajectory, write_trajectory


def _write_xyz(path: Path) -> None:
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[1.0, 0.0, 0.0]]),
    ]
    write(path, frames, format="extxyz")


def _write_cube(path: Path) -> np.ndarray:
    values = np.arange(6, dtype=float).reshape((1, 2, 3), order="C")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CPMD CUBE FILE\n")
        handle.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        handle.write("2 0.0 0.0 0.0\n")
        handle.write("1 1.0 0.0 0.0\n")
        handle.write("2 0.0 1.0 0.0\n")
        handle.write("3 0.0 0.0 1.0\n")
        handle.write("8 0.0 0.0 0.0 0.0\n")
        handle.write("1 0.0 0.0 0.0 1.0\n")
        flat = values.reshape(-1, order="C")
        for index, value in enumerate(flat, start=1):
            handle.write(f" {float(value): .8E}")
            if index % 6 == 0:
                handle.write("\n")
        if flat.size % 6 != 0:
            handle.write("\n")
    return values


def test_conversion_registry_detects_supported_file_families(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    traj_h5_path = tmp_path / "traj.traj.h5"
    cube_path = tmp_path / "field.cube"
    cube_h5_path = tmp_path / "field.cube.h5"
    _write_xyz(xyz_path)
    write_trajectory(read_trajectory(xyz_path), traj_h5_path)
    cube_path.write_text(
        "CPMD CUBE FILE\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n", encoding="utf-8"
    )

    assert CONVERSION_REGISTRY.detect_file_family(xyz_path) == "trajectory"
    assert CONVERSION_REGISTRY.detect_file_family(traj_h5_path) == "trajectory"
    assert CONVERSION_REGISTRY.detect_file_family(cube_path) == "cube"
    assert CONVERSION_REGISTRY.detect_file_family(cube_h5_path) == "cube"


def test_conversion_registry_exposes_allowed_targets_by_family(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    cube_path = tmp_path / "field.cube"
    _write_xyz(xyz_path)
    cube_path.write_text(
        "CPMD CUBE FILE\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n", encoding="utf-8"
    )

    trajectory_targets = {
        file_type.id for file_type in CONVERSION_REGISTRY.allowed_target_file_types(xyz_path)
    }
    cube_targets = {
        file_type.id for file_type in CONVERSION_REGISTRY.allowed_target_file_types(cube_path)
    }

    assert trajectory_targets == {"trajectory_hdf5", "trajectory_xyz"}
    assert cube_targets == {"cube_file", "cube_hdf5"}


def test_conversion_registry_default_request_uses_traj_hdf5_and_uniquifies(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    _write_xyz(xyz_path)
    output_dir = tmp_path / "LiNaK_outputs"
    output_dir.mkdir()
    existing_output = output_dir / "traj.traj.h5"
    existing_output.write_text("placeholder", encoding="utf-8")

    request = CONVERSION_REGISTRY.build_default_request(
        xyz_path,
        target_selector="hdf5",
        uniquify_default_output=True,
    )

    assert request.family == "trajectory"
    assert request.source_file_type == "trajectory_xyz"
    assert request.target_file_type == "trajectory_hdf5"
    assert request.target_path == output_dir / "traj_1.traj.h5"


def test_conversion_registry_default_request_uses_cube_hdf5(tmp_path):
    cube_path = tmp_path / "field.cube"
    _write_cube(cube_path)

    request = CONVERSION_REGISTRY.build_default_request(cube_path)

    assert request.family == "cube"
    assert request.source_file_type == "cube_file"
    assert request.target_file_type == "cube_hdf5"
    assert request.target_path == tmp_path / "field.cube.h5"


def test_conversion_registry_executes_current_trajectory_conversion(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    input_path = tmp_path / "input.inp"
    _write_xyz(xyz_path)
    input_path.write_text(
        "&SUBSYS\n  &CELL\n    ABC 10.0 11.0 12.0\n  &END CELL\n&END SUBSYS\n",
        encoding="utf-8",
    )
    request = CONVERSION_REGISTRY.build_default_request(xyz_path, target_selector="hdf5")

    result = CONVERSION_REGISTRY.execute(
        request,
        options=TrajectoryConversionOptions(input_path=input_path),
    )

    assert result.output_path == tmp_path / "LiNaK_outputs" / "traj.traj.h5"
    assert is_linak_trajectory_hdf5(result.output_path)


def test_conversion_registry_executes_trajectory_hdf5_to_xyz_roundtrip(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    _write_xyz(xyz_path)
    traj_h5 = tmp_path / "traj.traj.h5"
    write_trajectory(read_trajectory(xyz_path), traj_h5)

    request = CONVERSION_REGISTRY.build_request(
        traj_h5,
        target_selector="xyz",
    )

    result = CONVERSION_REGISTRY.execute(request, options=TrajectoryConversionOptions())

    assert result.output_path == tmp_path / "traj.xyz"
    loaded = read_trajectory(result.output_path)
    assert len(loaded) == 2
    assert loaded[1].positions[0, 0] == pytest.approx(1.0)


def test_conversion_registry_executes_cube_roundtrip(tmp_path):
    cube_path = tmp_path / "field.cube"
    expected_values = _write_cube(cube_path)

    to_hdf5 = CONVERSION_REGISTRY.build_request(cube_path, target_selector="cube.h5")
    hdf5_result = CONVERSION_REGISTRY.execute(to_hdf5, options=CubeConversionOptions())
    assert hdf5_result.output_path == tmp_path / "field.cube.h5"

    payloads = read_linak_hdf5_profiles(hdf5_result.output_path, expected_analysis="cube")
    assert len(payloads) == 1
    datasets, metadata = payloads[0]
    assert metadata["comment_1"] == "CPMD CUBE FILE"
    assert np.array_equal(datasets["values"], expected_values)

    to_cube = CONVERSION_REGISTRY.build_request(
        hdf5_result.output_path,
        output_path=tmp_path / "field_roundtrip.cube",
        target_selector="cube",
    )
    cube_result = CONVERSION_REGISTRY.execute(to_cube, options=CubeConversionOptions())
    assert cube_result.output_path == tmp_path / "field_roundtrip.cube"
    roundtrip_payload = cube_result.output_path.read_text(encoding="utf-8")
    assert "CPMD CUBE FILE" in roundtrip_payload
    assert "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z" in roundtrip_payload


def test_conversion_registry_rejects_unsupported_target_selector(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    _write_xyz(xyz_path)

    with pytest.raises(ValueError, match="Unsupported target file type 'lammps-output'"):
        CONVERSION_REGISTRY.build_request(xyz_path, target_selector="lammps-output")


def test_conversion_registry_builds_default_trajectory_combine_request(tmp_path, monkeypatch):
    xyz_a = tmp_path / "a.xyz"
    xyz_b = tmp_path / "b.xyz"
    _write_xyz(xyz_a)
    _write_xyz(xyz_b)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    request = CONVERSION_REGISTRY.build_combine_request([xyz_a, xyz_b])

    assert request.family == "trajectory"
    assert request.target_file_type == "trajectory_hdf5"
    assert request.target_path == workdir / "a_combined.traj.h5"
    assert request.conversion_applied is True


def test_conversion_registry_rejects_mixed_family_combine(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    cube_path = tmp_path / "field.cube"
    _write_xyz(xyz_path)
    _write_cube(cube_path)

    with pytest.raises(ValueError, match="mixed file families"):
        CONVERSION_REGISTRY.build_combine_request([xyz_path, cube_path])


def test_conversion_registry_combines_multiple_cubes_into_cube_hdf5(tmp_path, monkeypatch):
    cube_a = tmp_path / "a.cube"
    cube_b = tmp_path / "b.cube"
    values_a = _write_cube(cube_a)
    _write_cube(cube_b)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    request = CONVERSION_REGISTRY.build_combine_request([cube_a, cube_b])
    result = CONVERSION_REGISTRY.execute_combine(request, options=CubeConversionOptions())

    assert result.output_path == workdir / "a_combined.cube.h5"
    payloads = read_linak_hdf5_profiles(result.output_path, expected_analysis="cube")
    assert len(payloads) == 2
    assert np.array_equal(payloads[0][0]["values"], values_a)


def test_conversion_registry_describe_plan_reports_family_and_target(tmp_path):
    xyz_path = tmp_path / "traj.xyz"
    _write_xyz(xyz_path)
    request = CONVERSION_REGISTRY.build_default_request(xyz_path, target_selector="hdf5")

    plan = CONVERSION_REGISTRY.describe_plan(request)

    assert any("source family: trajectory" in line for line in plan)
    assert any("target file type: trajectory_hdf5" in line for line in plan)


@pytest.mark.parametrize("select, expected_frames", [(None, 2), ("first:0.001ps", 1)])
def test_convert_cli_explicit_timestep_without_input_is_stored_and_used_by_position(
    tmp_path, select, expected_frames
):
    from linak.cli import main
    from linak.analysis.position import load_position_profile
    from linak.trajectory.io import read_trajectory_hdf5_metadata

    source = tmp_path / "trajectory.xyz"
    target = tmp_path / "trajectory.traj.h5"
    _write_xyz(source)
    arguments = [
        "--log-level",
        "ERROR",
        "apply",
        "convert",
        str(source),
        "--timestep",
        "2.5",
        "--output",
        str(target),
    ]
    if select is not None:
        arguments.extend(["--select", select])
    assert main(arguments) == 0
    metadata = read_trajectory_hdf5_metadata(target)
    assert metadata is not None
    assert metadata.frame_timestep_fs == 2.5
    assert metadata.timestep_source == "explicit --timestep"
    assert metadata.md_timestep_fs is None and metadata.trajectory_stride_md is None
    frames = read_trajectory(target)
    assert len(frames) == expected_frames
    assert [frame.info["time_fs"] for frame in frames] == [2.5 * i for i in range(expected_frames)]
    if select is None:
        output = tmp_path / "result.position.h5"
        assert (
            main(
                [
                    "--log-level",
                    "ERROR",
                    "compute",
                    "position",
                    str(target),
                    "--species",
                    "O",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        np.testing.assert_array_equal(load_position_profile(output).time_fs, [0.0, 2.5])


def test_convert_explicit_timestep_overrides_stored_timing_without_being_skipped(tmp_path):
    from linak.cli import main
    from linak.trajectory.io import TrajectoryStoredMetadata, read_trajectory_hdf5_metadata

    source = tmp_path / "trajectory.traj.h5"
    frames = [
        Atoms(
            "O",
            positions=[[0, 0, 0]],
            info={
                "time_fs": 10 + index,
                "frame_timestep_fs": 1,
                "md_timestep_fs": 0.5,
                "trajectory_stride_md": 2,
            },
        )
        for index in range(2)
    ]
    write_trajectory(
        frames,
        source,
        metadata=TrajectoryStoredMetadata(
            frame_timestep_fs=1, md_timestep_fs=0.5, trajectory_stride_md=2
        ),
    )
    assert main(["--log-level", "ERROR", "apply", "convert", str(source), "--timestep", "2.5"]) == 0
    metadata = read_trajectory_hdf5_metadata(source)
    assert metadata is not None and metadata.frame_timestep_fs == 2.5
    assert metadata.md_timestep_fs is None and metadata.trajectory_stride_md is None
    result = read_trajectory(source)
    assert [frame.info["time_fs"] for frame in result] == [10, 12.5]
    assert all(
        "md_timestep_fs" not in frame.info and "trajectory_stride_md" not in frame.info
        for frame in result
    )


def test_convert_explicit_timestep_dry_run_does_not_read_payload(tmp_path, monkeypatch, capsys):
    from linak.cli import main

    source = tmp_path / "trajectory.xyz"
    target = tmp_path / "result.traj.h5"
    _write_xyz(source)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("dry run must not read trajectory payload")

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", unexpected_read)
    assert (
        main(
            [
                "apply",
                "convert",
                str(source),
                "--timestep",
                "2.5",
                "--output",
                str(target),
                "--dry-run",
            ]
        )
        == 0
    )
    out, err = capsys.readouterr()
    assert not out
    assert "explicit --timestep (frame=2.5 fs)" in err
    assert not target.exists()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_convert_cli_rejects_invalid_timestep(value):
    from linak.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["apply", "convert", "trajectory.xyz", "--timestep", value])
    with pytest.raises(ValueError, match="timestep must be finite"):
        TrajectoryConversionOptions(timestep_fs=float(value))


def test_convert_rejects_timestep_for_cube_input(tmp_path, capsys):
    from linak.cli import main

    source = tmp_path / "volume.cube"
    _write_cube(source)
    assert main(["apply", "convert", str(source), "--timestep", "2.5"]) != 0
    assert "--timestep is supported for trajectory conversion only" in capsys.readouterr().err
