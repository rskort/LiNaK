"""Removed analyses must not remain registered or damage existing user data."""

from __future__ import annotations


import argparse
from importlib.util import find_spec
import logging

import pytest

from linak import cli
from linak.analysis.schema import get_analysis_schema
from linak.gui.actions import ActionRegistry
from linak.gui.model import ProjectStore, Task
from linak.plot.plot_settings import read_plot_profile, write_plot_profile
from linak.storage.hdf5_utils import write_linak_hdf5


@pytest.fixture(autouse=True)
def restore_logging_level():
    logger = logging.getLogger("linak")
    original_level = logger.level
    yield
    logger.setLevel(original_level)


@pytest.mark.parametrize("command", ["msd", "MSD"])
def test_removed_compute_commands_are_invalid(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["compute", command, "traj.xyz"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_removed_analysis_is_not_advertised_or_importable():
    parser = cli.build_parser()

    def check_help(current):
        assert "msd" not in current.format_help().lower()
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    check_help(child)

    check_help(parser)
    assert "msd" not in {action.action_id for action in ActionRegistry().actions}
    assert "msd" not in cli._ANALYSIS_TO_PROFILE_KEY
    schema = get_analysis_schema("msd")
    assert schema.default_units_map == {}
    assert schema.default_plot_labels is None
    for module in (
        "linak.analysis.msd",
        "linak.plot.contracts.msd_contract",
        "linak.plot.mappings.msd_mapping",
    ):
        assert find_spec(module) is None


@pytest.fixture
def legacy_analysis_file(tmp_path):
    source = tmp_path / "legacy.msd.h5"
    write_linak_hdf5(
        source,
        analysis="msd",
        datasets={"time_ps": [0.0, 1.0], "msd_A2": [0.0, 0.5]},
        metadata={"species": "O"},
    )
    write_plot_profile(
        source,
        "plot:msd",
        {
            "__linak_plot_profile__": 2,
            "source_selection": {"species": "O"},
            "view_mapping": {"view_type_id": "plot_1d_line", "x": "time_ps", "y": "msd"},
            "style": {"title": "Old result"},
        },
    )
    return source


def test_legacy_analysis_is_not_detected_or_modified(legacy_analysis_file):
    source = legacy_analysis_file
    before = source.read_bytes()
    assert cli._detect_plot_analysis_from_hdf5_source(source) is None
    args = cli.build_parser().parse_args(["plot", str(source), "--no-show"])
    with pytest.raises(ValueError, match="Could not detect.*Use `linak hdf5 plot"):
        cli._handle_plot(args)
    assert source.read_bytes() == before
    saved = read_plot_profile(source, "plot:msd")
    assert saved is not None
    assert saved["style"]["title"] == "Old result"


def test_legacy_analysis_uses_generic_cli_fallback(legacy_analysis_file, monkeypatch):
    source = legacy_analysis_file
    before = source.read_bytes()
    calls = []

    def generic_plot(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_handle_csv_plot", generic_plot)
    assert cli.main(["--log-level", "ERROR", "plot", str(source), "--no-show"]) == 0
    assert len(calls) == 1
    assert source.read_bytes() == before


def test_legacy_analysis_still_supports_generic_hdf5_tools(legacy_analysis_file, tmp_path):
    source = legacy_analysis_file
    before = source.read_bytes()
    assert cli.main(["--log-level", "ERROR", "hdf5", "info", str(source)]) == 0
    output = tmp_path / "generic.png"
    assert (
        cli.main(
            [
                "--log-level",
                "ERROR",
                "hdf5",
                "plot",
                str(source),
                "--kind",
                "line",
                "--x",
                "time_ps",
                "--y",
                "msd_A2",
                "--no-show",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists()
    assert source.read_bytes() == before


def test_legacy_analysis_cannot_be_combined(legacy_analysis_file, tmp_path, capsys):
    source = legacy_analysis_file
    second = tmp_path / "second.msd.h5"
    before = source.read_bytes()
    second.write_bytes(before)
    output = tmp_path / "combined.h5"
    assert (
        cli.main(
            [
                "--log-level",
                "ERROR",
                "hdf5",
                "combine",
                "-f",
                str(source),
                str(second),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "supports LiNaK" in capsys.readouterr().err
    assert not output.exists()
    assert source.read_bytes() == second.read_bytes() == before


def test_legacy_task_history_loads_without_registering_removed_action(tmp_path):
    store = ProjectStore(tmp_path)
    store.tasks = [
        Task(
            action_id="msd",
            action_name="MSD",
            input_item_id="old-source",
            status="finished",
            settings_snapshot={"species": "O", "timestep_fs": 0.5},
        )
    ]
    store.save()
    before = store.manifest_path.read_bytes()
    reopened = ProjectStore(tmp_path)
    reopened.load()
    assert reopened.tasks[0].settings_snapshot == store.tasks[0].settings_snapshot
    assert store.manifest_path.read_bytes() == before
    with pytest.raises(KeyError):
        ActionRegistry().by_id(reopened.tasks[0].action_id)
