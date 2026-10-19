from pathlib import Path
import io
import logging

import h5py
import numpy as np
import pytest
from ase import Atoms

from linak import cli, conversion
from linak.analysis.common import RAW_SPECIES_ARRAY
from linak.progress import ProgressBar, progress_stage
from linak.trajectory import io as trajectory_io


def test_xyz_payload_once_and_label_cache(tmp_path, monkeypatch):
    source = tmp_path / "frames.xyz"
    source.write_text("2\nfirst\nOw 1 2 3 extra\nH 4 5 6\n2\nsecond\nO 7 8 9\nOw 1_0 0 0\n")
    original_open = Path.open
    opens = []

    def track_open(path, *args, **kwargs):
        if path == source:
            opens.append(args[0] if args else kwargs.get("mode", "r"))
        return original_open(path, *args, **kwargs)

    original_resolve = trajectory_io._resolve_raw_atom_label
    labels = []

    def resolve(label, aliases):
        labels.append(label)
        return original_resolve(label, aliases)

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(trajectory_io, "_resolve_raw_atom_label", resolve)
    frames = list(trajectory_io._iter_xyz_like_frames(source, atom_aliases={"Ow": "O"}))
    assert opens == ["rb"]
    assert labels == ["Ow", "H", "O"]
    np.testing.assert_array_equal(frames[1].positions, [[7, 8, 9], [10, 0, 0]])
    assert frames[0].arrays[RAW_SPECIES_ARRAY].tolist() == ["Ow", "H"]
    assert frames[1].arrays[RAW_SPECIES_ARRAY].tolist() == ["O", "Ow"]


@pytest.mark.parametrize(
    "payload", ["-1\ncomment\n", "1\n", "1\nc\nH 0 bad 0\n", "1\nc\nH 0 0 1junk\n"]
)
def test_xyz_malformed_input(tmp_path, payload):
    source = tmp_path / "bad.xyz"
    source.write_text(payload)
    with pytest.raises(ValueError):
        list(trajectory_io._iter_xyz_like_frames(source))


def test_cli_summary_never_rereads_output(tmp_path, monkeypatch, capsys):
    source = tmp_path / "variable.xyz"
    source.write_text("2\na\nH 0 0 0\nO 5 0 1\n2\nb\nH 0 0 0\nO 0 0 1\n")
    output = tmp_path / "variable.traj.h5"
    reads = []
    original_read = trajectory_io.read_trajectory

    def read(path, **kwargs):
        assert Path(path) == source
        reads.append(path)
        return original_read(path, **kwargs)

    original_metadata = conversion._collect_trajectory_conversion_metadata
    metadata_calls = []

    def metadata(*args, **kwargs):
        metadata_calls.append(args)
        return original_metadata(*args, **kwargs)

    monkeypatch.setattr(trajectory_io, "read_trajectory", read)
    monkeypatch.setattr(conversion, "_collect_trajectory_conversion_metadata", metadata)
    assert (
        cli.main(["apply", "convert", str(source), "--output", str(output), "--x-range", "0:1"])
        == 0
    )
    captured = capsys.readouterr()
    assert Path(captured.out.strip()).exists()
    assert len(reads) == len(metadata_calls) == 1
    assert "atoms/frame=1..2" in captured.err
    assert "\r" not in captured.err
    for stage in (
        "Read trajectory",
        "Prepare cell/selection",
        "Build molecular topology",
        "Build surface cache",
        "Write HDF5",
    ):
        assert f"{stage} completed in" in captured.err


@pytest.mark.parametrize("compression", ["fast", "balanced", "none"])
def test_chunked_writer_variable_topology_and_progress(tmp_path, monkeypatch, compression):
    frames = [
        Atoms(
            "HO" if i % 2 else "H",
            positions=[[i, 0, 0], [0, 1, 0]] if i % 2 else [[i, 0, 0]],
            cell=[[8, 0, 0], [1, 9, 0], [0, 1, 10]],
            pbc=[True, True, False],
        )
        for i in range(300)
    ]
    original_zeros = np.zeros

    def bounded_zeros(shape, *args, **kwargs):
        assert shape != (len(frames), 2, 3)
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", bounded_zeros)
    for index, frame in enumerate(frames):
        frame.new_array(
            RAW_SPECIES_ARRAY,
            np.asarray(["Ha" if index % 3 else "Hb"] + (["O"] if len(frame) == 2 else [])),
        )
    written = []
    updates: list[int] = []
    original_set = h5py.Dataset.__setitem__

    def record(dataset, selection, value):
        if dataset.name == "/positions":
            written.append(len(value))
        return original_set(dataset, selection, value)

    original_update = ProgressBar.update

    def update(bar, n=1):
        if bar.desc == "Writing trajectory":
            assert sum(written) == sum(updates) + n
            updates.append(n)
        return original_update(bar, n)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", record)
    monkeypatch.setattr(ProgressBar, "update", update)
    output = tmp_path / "chunks.traj.h5"
    trajectory_io.write_trajectory(
        frames,
        output,
        compression_profile=compression,
        metadata=trajectory_io.TrajectoryStoredMetadata(spatial_filter_metadata={}),
    )
    assert len(updates) > 1
    assert sum(updates) == 300
    loaded = trajectory_io.read_trajectory(output)
    for expected, actual in zip(frames, loaded):
        np.testing.assert_array_equal(actual.positions, expected.positions)
        np.testing.assert_array_equal(actual.numbers, expected.numbers)
        np.testing.assert_array_equal(actual.pbc, expected.pbc)
        np.testing.assert_array_equal(actual.cell, expected.cell)
        np.testing.assert_array_equal(
            actual.arrays[RAW_SPECIES_ARRAY], expected.arrays[RAW_SPECIES_ARRAY]
        )


def test_interrupted_write_preserves_target(tmp_path, monkeypatch):
    output = tmp_path / "existing.traj.h5"
    output.write_bytes(b"previous output")
    original = h5py.Dataset.__setitem__

    def interrupted(dataset, key, value):
        if dataset.name == "/positions":
            raise KeyboardInterrupt()
        return original(dataset, key, value)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        trajectory_io.write_trajectory([Atoms("H")], output)
    assert output.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("level", [logging.INFO, logging.ERROR])
@pytest.mark.parametrize("tty", [True, False])
def test_stage_failure_cleans_progress_and_respects_level(level, tty):
    class Stream(io.StringIO):
        def isatty(self):
            return tty

    stream = Stream()
    logger = logging.Logger("stage-test", level=level)
    logger.addHandler(logging.StreamHandler(stream))
    with pytest.raises(RuntimeError):
        with progress_stage(logger, "Read trajectory"):
            with ProgressBar(desc="Reading XYZ", total=1024, unit="B", stream=stream) as bar:
                bar.detail = "1 frames"
                bar.update(512)
                raise RuntimeError("broken")
    text = stream.getvalue()
    assert "failed after" in text
    assert ("\r" in text) == (tty and level == logging.INFO)
    assert stream.getvalue().endswith("\n")
    assert id(stream) not in ProgressBar._ACTIVE_BY_STREAM


def test_byte_progress_shows_frames_and_closes_cleanly():
    class Stream(io.StringIO):
        def isatty(self):
            return True

    stream = Stream()
    with ProgressBar(
        desc="Read XYZ", total=2 * 1024**2, unit="B", stream=stream, min_interval=0
    ) as bar:
        bar.detail = "12 frames"
        bar.update(1024**2)
    text = stream.getvalue()
    assert "1.0/2.0 MiB" in text
    assert "12 frames" in text
    assert text.endswith("\n")
