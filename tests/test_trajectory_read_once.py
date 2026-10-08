"""Multi-pass CLI analyses should load the trajectory only once by default."""

from pathlib import Path

from ase import Atoms
import pytest

from linak.cli import main
from linak.analysis import temperature
from linak.trajectory import execution, io


@pytest.mark.parametrize(
    ("analysis", "options"),
    [
        ("density", ["--bin-width", "1"]),
        ("position", ["--species", "Ar", "--timestep-fs", "2"]),
        ("coordination", ["--species", "Ar", "Ar", "--cutoff", "3", "--timestep-fs", "2"]),
    ],
)
def test_default_compute_reads_large_trajectory_once(tmp_path, monkeypatch, analysis, options):
    source = tmp_path / "source.traj.h5"
    output = tmp_path / f"result.{analysis}.h5"
    frames = [
        Atoms(
            "ArAr",
            positions=[[1 + index * 0.1, 1, 1], [3, 3, 3]],
            cell=[8, 8, 8],
            pbc=True,
        )
        for index in range(6)
    ]
    io.write_trajectory(frames, source)
    # Reproduce the automatic streaming trigger without allocating a large fixture.
    monkeypatch.setattr(
        execution, "estimate_materialized_trajectory_bytes", lambda _: 300 * 1024**2
    )
    reader = io._read_trajectory_hdf5_chunks
    reads = []

    def counted_reader(path, *args, **kwargs):
        reads.append(Path(path))
        yield from reader(path, *args, **kwargs)

    monkeypatch.setattr(io, "_read_trajectory_hdf5_chunks", counted_reader)
    assert main(["compute", analysis, str(source), "--output", str(output), *options]) == 0
    assert output.is_file()
    assert reads == [source]


@pytest.mark.parametrize("mode", ["auto", "stream"])
def test_explicit_streaming_remains_available(tmp_path, mode):
    source = tmp_path / "source.traj.h5"
    io.write_trajectory([Atoms("Ar", positions=[[1, 1, 1]], cell=[8, 8, 8], pbc=True)], source)
    with execution.prepare_cli_trajectory(
        source,
        options=execution.TrajectoryExecutionOptions(mode=mode, stream_threshold_mib=0.0001),
    ) as (frames, diagnostics):
        assert diagnostics.resolved_mode == "stream"
        assert len(list(frames)) == 1


@pytest.mark.parametrize(
    "entry_point",
    [
        temperature.compute_temperature_profiles,
        temperature.compute_temperature_from_velocity_xyz,
    ],
)
def test_velocity_temperature_reads_frames_once(tmp_path, monkeypatch, entry_point):
    source = tmp_path / "run-vel-1.xyz"
    source.write_text("1\ni = 0, time = 0\nHw 0.001 0 0\n" * 3, encoding="utf-8")
    reader = io._iter_xyz_like_frames
    reads = []

    def counted_reader(path, **kwargs):
        for frame in reader(path, **kwargs):
            reads.append(Path(path))
            yield frame

    monkeypatch.setattr(io, "_iter_xyz_like_frames", counted_reader)
    profiles = entry_point(source, atom_aliases=["Hw=H"])
    assert profiles[0].n_frames == 3
    assert profiles[0].element == "H"
    assert reads == [source] * 3


def test_temperature_metadata_reads_only_first_frame(tmp_path, monkeypatch):
    source = tmp_path / "run-vel-1.xyz"
    source.write_text("1\ni = 0, time = 0\nHw 0.001 0 0\n" * 3, encoding="utf-8")
    reader = io._iter_xyz_like_frames
    reads = []
    closed = []

    def counted_reader(path, **kwargs):
        try:
            for frame in reader(path, **kwargs):
                reads.append(Path(path))
                yield frame
        finally:
            closed.append(True)

    monkeypatch.setattr(io, "_iter_xyz_like_frames", counted_reader)
    temperature.discover_temperature_metadata(source, atom_aliases=["Hw=H"])
    assert reads == [source]
    assert closed == [True]
