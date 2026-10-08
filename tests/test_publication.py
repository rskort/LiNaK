"""Release-blocking safety, settings, and bounded-I/O regressions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

from ase import Atoms
import numpy as np
import h5py
import pytest

from linak import temporary_resources as resources
from linak.gui.actions import ActionRegistry
from linak.gui.model import ProjectItem
from linak.trajectory.execution import ChunkedTrajectorySequence
from linak.trajectory import io
from linak.plot.settings_validation import validate_current_plot_settings
from linak.plot.profile_persistence import build_plot_profile_payload, select_plot_profile_settings


@pytest.mark.parametrize("removed", ["heatmap_normalize", "heatmap_normalization_mode"])
@pytest.mark.parametrize("section", ["root", "style", "view_state"])
def test_removed_orientation_settings_rejected_without_mutation(removed, section):
    payload: dict[str, object] = {removed: True}
    if section == "style":
        payload = {"style": payload}
    elif section == "view_state":
        payload = {"orientation_view_states": {"plot_2d_heatmap": payload}}
    original = json.dumps(payload)
    for validate in (
        validate_current_plot_settings,
        lambda value: build_plot_profile_payload("plot:orientation", value),
        lambda value: select_plot_profile_settings("plot:orientation", value, keys=None),
    ):
        with pytest.raises(ValueError, match="Recreate the plot profile using heatmap_value_mode"):
            validate(payload)
        assert json.dumps(payload) == original


def test_orientation_python_plot_api_has_only_current_normalization_options():
    from inspect import signature
    from linak.analysis.orientation import plot_orientation_profile, plot_orientation_profiles

    for plot in (plot_orientation_profile, plot_orientation_profiles):
        parameters = signature(plot).parameters
        assert "heatmap_value_mode" in parameters
        assert "heatmap_normalize" not in parameters
        assert "heatmap_normalization_mode" not in parameters


def test_density_output_selection_rejects_removed_line_alias():
    import argparse
    from linak.analysis.density import _normalize_density_outputs
    from linak.cli import _resolve_density_outputs_from_args

    for value in (None, "1d", "3d", "all"):
        assert _resolve_density_outputs_from_args(argparse.Namespace(outputs=value)) == (
            _normalize_density_outputs(value)
        )
    with pytest.raises(ValueError, match="1d, 3d, all"):
        _resolve_density_outputs_from_args(argparse.Namespace(outputs="line"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "NaN", "Infinity", "1e999"])
@pytest.mark.parametrize("validator_name", ["_positive_float", "_non_negative_float"])
def test_numeric_cli_controls_require_finite_values(value, validator_name):
    import argparse
    from linak import cli

    validator = getattr(cli, validator_name)
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        validator(value)
    assert validator("0.125") == 0.125


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_positive_scientific_parameters_reject_nonfinite_and_nonpositive_values(value):
    from linak.utils import ensure_positive

    with pytest.raises(ValueError, match="finite and > 0"):
        ensure_positive("bin_width", value)
    ensure_positive("bin_width", 0.125)


def test_plot_header_cache_isolates_nested_metadata_and_invalidates_file_state(
    tmp_path, monkeypatch
):
    from linak.plot import profile_loading
    from linak.storage import hdf5_utils

    source = tmp_path / "headers.h5"
    source.write_bytes(b"first state")
    calls = []

    def headers_only(path, *, expected_analysis):
        calls.append(path)
        return [{"surface": {"layout_resolved": "symmetric"}, "dataset_inventory": [{"name": "x"}]}]

    monkeypatch.setattr(hdf5_utils, "read_linak_hdf5_profile_headers", headers_only)
    profile_loading.clear_profile_header_cache()
    first = profile_loading.read_analysis_profile_headers_by_source(
        sources=[str(source)], analysis="density"
    )
    first[0][1][0]["surface"]["layout_resolved"] = "mutated"
    first[0][1][0]["dataset_inventory"][0]["name"] = "mutated"
    second = profile_loading.read_analysis_profile_headers_by_source(
        sources=[str(source)], analysis="density"
    )
    assert second[0][1][0]["surface"]["layout_resolved"] == "symmetric"
    assert second[0][1][0]["dataset_inventory"][0]["name"] == "x"
    assert len(calls) == 1
    source.write_bytes(b"changed file state and size")
    profile_loading.read_analysis_profile_headers_by_source(
        sources=[str(source)], analysis="density"
    )
    assert len(calls) == 2
    profile_loading.clear_profile_header_cache()
    profile_loading.read_analysis_profile_headers_by_source(
        sources=[str(source)], analysis="density"
    )
    assert len(calls) == 3
    profile_loading.clear_profile_header_cache()


def test_plot_header_cache_evicts_least_recently_used_metadata(tmp_path, monkeypatch):
    from linak.plot import profile_loading
    from linak.storage import hdf5_utils

    calls = []

    def headers_only(path, *, expected_analysis):
        calls.append(path)
        return [{"species": "O"}]

    monkeypatch.setattr(hdf5_utils, "read_linak_hdf5_profile_headers", headers_only)
    profile_loading.clear_profile_header_cache()

    def read(index):
        profile_loading.read_analysis_profile_headers_by_source(
            sources=[str(tmp_path / f"source-{index}.h5")], analysis="density"
        )

    for index in range(128):
        read(index)
    read(0)
    read(128)
    read(0)
    assert len(calls) == 129
    read(1)  # least recently used entry, not the recently reused first entry
    assert len(calls) == 130
    assert profile_loading._cached_profile_headers.cache_info().currsize == 128
    profile_loading.clear_profile_header_cache()


@pytest.mark.parametrize("layout", [None, "auto", "single-top", "single-bottom", "symmetric"])
def test_obsolete_surface_cache_is_ignored_without_payload_reads(tmp_path, monkeypatch, layout):
    path = tmp_path / "obsolete.traj.h5"
    io.write_trajectory(
        [Atoms("Pt", positions=[[0, 0, 1]], cell=[5, 5, 10], pbc=True)],
        path,
        metadata=io.TrajectoryStoredMetadata(
            surface_cache_status="unavailable", surface_cache_axis="z", surface_cache_mode="auto"
        ),
    )
    with h5py.File(path, "a") as handle:
        group = handle["metadata/surface_cache"]
        group.attrs["status"] = "available"
        group.create_dataset("surface_position_per_frame_A", data=[1.0])
    original = path.read_bytes()

    def unexpected_read(*args, **kwargs):
        pytest.fail("Obsolete cache payload must not be read")

    monkeypatch.setattr(io, "_read_surface_cache_datasets", unexpected_read)
    assert (
        io.read_trajectory_hdf5_surface_cache(
            path,
            axis="z",
            surface_mode="auto",
            surface_layout=layout,
            surface_elements=None,
            include_fixed_surface_atoms=False,
            rough_surface_envelope_A=None,
            frame_count=1,
        )
        is None
    )
    assert path.read_bytes() == original


def test_process_probe_keeps_current_process_alive():
    assert resources.process_alive(os.getpid())
    assert resources.process_alive(0)  # invalid markers are never safe to delete


def test_uncertain_process_probe_protects_directory(monkeypatch):
    def denied(*args):
        raise PermissionError("not permitted")

    if os.name == "nt":
        monkeypatch.setattr(resources, "_windows_process_alive", denied)
    else:
        monkeypatch.setattr(os, "kill", denied)
    assert resources.process_alive(12345)


@pytest.mark.parametrize(
    "condition", ["dead", "alive", "foreign", "malformed", "future", "nan", "invalid_pid"]
)
def test_stale_cleanup_requires_all_safety_conditions(tmp_path, monkeypatch, condition):
    directory = tmp_path / "linak-check-one"
    directory.mkdir()
    payload = {"pid": 12345, "owner": resources.current_owner(), "started": time.time() - 200}
    if condition == "foreign":
        payload["owner"] = "not-current-owner"
    elif condition == "future":
        payload["started"] = time.time() + 200
    elif condition == "nan":
        payload["started"] = float("nan")
    elif condition == "invalid_pid":
        payload["pid"] = -1
    (directory / "marker.json").write_text(
        "[]" if condition == "malformed" else json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(resources, "process_alive", lambda pid: condition == "alive")
    removed = resources.cleanup_stale_directories(
        tmp_path,
        prefix="linak-check-",
        marker_name="marker.json",
        timestamp_key="started",
        stale_seconds=100,
    )
    assert removed == (1 if condition == "dead" else 0)
    assert directory.exists() == (condition != "dead")


def test_cleanup_rejects_links_and_outside_paths(tmp_path, monkeypatch):
    directory = tmp_path / "linak-check-one"
    directory.mkdir()
    (directory / "marker.json").write_text(json.dumps({"owner": resources.current_owner()}))
    assert (
        resources.owned_directory_payload(
            directory,
            parent=tmp_path / "elsewhere",
            prefix="linak-check-",
            marker_name="marker.json",
        )
        is None
    )
    monkeypatch.setattr(resources, "_is_link", lambda path: path == directory)
    assert not resources.remove_owned_directory(
        directory, parent=tmp_path, prefix="linak-check-", marker_name="marker.json"
    )
    assert directory.exists()


def test_indexed_trajectory_reads_only_requested_chunks_and_preserves_slice_order(
    tmp_path, monkeypatch
):
    paths = [tmp_path / "first.traj.h5", tmp_path / "second.traj.h5"]
    for part, path in enumerate(paths):
        io.write_trajectory(
            [
                Atoms("H", positions=[[part * 5 + i, 0, 0]], cell=[20, 20, 20], pbc=True)
                for i in range(5)
            ],
            path,
        )
    reads = []
    original = io._read_trajectory_hdf5_chunks

    def counted(path, **kwargs):
        reads.append((path, kwargs["start_frame"], kwargs["stop_frame"]))
        yield from original(path, **kwargs)

    monkeypatch.setattr(io, "_read_trajectory_hdf5_chunks", counted)
    sequence = ChunkedTrajectorySequence(paths, [5, 5], chunk_frames=2)
    assert sequence[-2].positions[0, 0] == 8
    assert sequence[7].positions[0, 0] == 7
    assert reads == [(paths[1], 2, 4)]
    sequence[7].positions[:] = 100
    assert sequence[7].positions[0, 0] == 7
    np.testing.assert_array_equal(
        [frame.positions[0, 0] for frame in sequence[::-2]], [9, 7, 5, 3, 1]
    )
    with pytest.raises(IndexError):
        sequence[10]


def test_all_gui_setting_fields_have_valid_structure(tmp_path):
    for action in ActionRegistry().actions:
        for item_type in action.supported_input_types:
            item = ProjectItem(
                path=Path(tmp_path / "input.xyz"), item_type=item_type, origin="external"
            )
            for field in action.settings_schema(item):
                assert isinstance(field.choices, tuple), (action.action_id, field.key)
                assert isinstance(field.required, bool), (action.action_id, field.key)
                assert isinstance(field.group, str), (action.action_id, field.key)


def test_default_palette_accepts_rgb_cycle():
    import matplotlib
    from cycler import cycler
    from linak.plot.plotting import default_series_colors

    with matplotlib.rc_context({"axes.prop_cycle": cycler(color=[(1.0, 0.0, 0.0), "navy"])}):
        assert default_series_colors(3) == ["#ff0000", "navy", "#ff0000"]


@pytest.mark.parametrize(
    "analysis,version",
    [
        ("density", 2),
        ("position", 2),
        ("coordination", 2),
        ("orientation", 2),
        ("rdf", 1),
        ("temperature", 1),
    ],
)
def test_schema_validation_precedes_payload_loading(tmp_path, monkeypatch, analysis, version):
    from linak.storage.hdf5_utils import write_linak_hdf5, read_linak_hdf5

    path = write_linak_hdf5(
        tmp_path / "profile.h5", analysis=analysis, datasets={"values": np.ones(4)}, metadata={}
    )
    _, metadata = read_linak_hdf5(path, expected_analysis=analysis)
    assert metadata["analysis_schema_version"] == version
    with h5py.File(path, "r+") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        metadata["analysis_schema_version"] = 99
        handle.attrs["metadata_json"] = json.dumps(metadata)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Schema rejection must precede numerical payload reads")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", unexpected_read)
    with pytest.raises(ValueError, match="Recompute"):
        read_linak_hdf5(path, expected_analysis=analysis)


def test_density_info_volume_does_not_grow_with_frames_or_species(caplog):
    from linak.analysis.density import compute_all_density_profiles

    frame = Atoms("ArNe", positions=[[1, 1, 1], [2, 2, 2]], cell=[5, 5, 5], pbc=True)
    counts = []
    for frame_count, species in [(1, "Ar"), (6, "all")]:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="linak.analysis.density"):
            compute_all_density_profiles(
                [frame] * frame_count, species=species, surface_mode="none", bin_width=1.0
            )
        messages = [
            r.getMessage()
            for r in caplog.records
            if r.name == "linak.analysis.density" and r.levelno == logging.INFO
        ]
        assert not any("sparse grid for" in message for message in messages)
        counts.append(len(messages))
    assert counts[0] == counts[1]


def test_coordination_info_volume_does_not_grow_with_frames_or_pairs(caplog):
    from linak.analysis.coordination import (
        compute_coordination_profiles,
        CoordinationCutoffResolution,
    )

    frame = Atoms("LiOO", positions=[[1, 1, 1], [2, 1, 1], [1, 2, 1]], cell=[6, 6, 6], pbc=True)
    all_pairs = [("Li", "O"), ("O", "Li"), ("O", "O")]
    resolution = CoordinationCutoffResolution(cutoff_A=2.0, smoothing_width_A=0.2, mode="manual")
    counts = []
    for frame_count, pairs in [(1, all_pairs[:1]), (4, all_pairs)]:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="linak.analysis.coordination"):
            compute_coordination_profiles(
                [frame] * frame_count,
                ordered_pairs=pairs,
                surface_mode="none",
                cutoff_resolutions={pair: resolution for pair in pairs},
            )
        records = [
            record for record in caplog.records if record.name == "linak.analysis.coordination"
        ]
        details = [
            record for record in records if "Computed coordination for" in record.getMessage()
        ]
        assert len(details) == len(pairs)
        assert all(record.levelno == logging.DEBUG for record in details)
        counts.append(sum(record.levelno == logging.INFO for record in records))
    assert counts[0] == counts[1] == 1
