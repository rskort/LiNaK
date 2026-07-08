from __future__ import annotations

from pathlib import Path
import threading
import time

import h5py
from ase import Atoms
from ase.io import write

from linak.gui.actions import (
    Action,
    ActionContext,
    ActionExecutionResult,
    ActionRegistry,
    SettingField,
    validate_action_settings,
)
from linak.gui.detection import (
    detect_project_item,
    discover_generated_items,
    discover_generated_items_cached,
)
from linak.gui.model import ProjectStore, Task, WorkspaceIndex
from linak.gui.components import action_row_display, grouped_item_rows, task_detail_display
from linak.gui.defaults import (
    default_settings_for_action,
    out_h5_gui_summary_for_item,
    readiness_for_action,
)
from linak.gui.services import descriptor_for_action
from linak.gui.settings import build_gui_action_settings, settings_hash
from linak.gui.tasks import TaskManager
from linak.gui.viewmodels import (
    display_for_item,
    display_for_task,
    filter_items,
    suggested_actions_for_item,
    task_progress_label,
)
from linak.gui.styles import badge_style
from linak.gui.theme import plot_like_theme_tokens, workspace_stylesheet
from linak.storage.hdf5_utils import write_linak_hdf5


def _write_xyz(path: Path) -> None:
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        Atoms("O", positions=[[0.0, 0.0, 0.1]]),
    ]
    write(path, frames, format="extxyz")


def _write_minimal_out_h5(path: Path, *, trajectory: bool = True, cubes: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["linak_format"] = "linak-out-hdf5"
        handle.attrs["linak_out_schema_version"] = 1
        handle.attrs["source_directory"] = str(path.parent / "sim")
        trajectory_group = handle.create_group("trajectory")
        trajectory_group.attrs["present"] = trajectory
        trajectory_group.attrs["frame_count"] = 12 if trajectory else 0
        trajectory_group.attrs["atom_count"] = 2 if trajectory else 0
        trajectory_group.attrs["source_path"] = str(path.parent / "traj.xyz")
        trajectory_group.attrs["source_format"] = "xyz"
        trajectory_group.create_dataset("pbc", data=[[True, True, False]])
        frame_info = trajectory_group.create_group("frame_info")
        frame_info.create_dataset("frame_timestep_fs", data=[0.5])
        cubes_group = handle.create_group("cubes")
        cubes_group.attrs["count"] = cubes
        for index in range(cubes):
            cube = cubes_group.create_group(f"{index:04d}_density")
            cube.attrs["source_name"] = f"density_{index}.cube"
            cube.attrs["cube_kind"] = "density"
        cp2k = handle.create_group("singlepoint").create_group("cp2k")
        cp2k.attrs["output_count"] = 1
        cp2k.attrs["sections_json"] = '["md_steps"]'
        cp2k_output = cp2k.create_group("0000_output")
        md_steps = cp2k_output.create_group("md_steps")
        md_steps.create_dataset("step", data=[1, 2, 3])
        system = handle.create_group("system")
        system.create_dataset("species", data=[b"Li", b"O"])
        system.create_dataset("cell_angstrom", data=[[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]])
        provenance = handle.create_group("provenance")
        provenance.attrs["discovery_summary_json"] = '{"trajectories": ["traj.xyz"], "cubes": ["density.cube"], "cp2k_outputs": ["output.out"], "skipped": []}'
        provenance.create_dataset("messages", data=[b"Skipped optional cube"])


def test_gui_command_is_explicit_subcommand():
    from linak.cli import build_parser

    args = build_parser().parse_args(["project", "workspace"])

    assert args.command == "project"
    assert args.project_dir == "workspace"


def test_project_item_detection_distinguishes_external_and_generated_hdf5(tmp_path):
    source = tmp_path / "traj.xyz"
    output = tmp_path / "project" / "traj.density.h5"
    _write_xyz(source)
    write_linak_hdf5(
        output,
        analysis="density",
        datasets={"x": [0.0, 1.0], "density": [1.0, 2.0]},
        metadata={"source_path": str(source.resolve()), "species": "O"},
    )

    external = detect_project_item(source, origin="external")
    generated = detect_project_item(output, origin="generated")

    assert external.item_type == "raw_trajectory"
    assert external.origin == "external"
    assert generated.item_type == "analysis_hdf5"
    assert generated.metadata["analysis"] == "density"
    assert generated.validation.state == "valid"


def test_project_store_persists_external_references_without_copying(tmp_path):
    project = tmp_path / "project"
    source = tmp_path / "external" / "traj.xyz"
    source.parent.mkdir()
    _write_xyz(source)

    store = ProjectStore(project)
    assert store.initialize()
    item = detect_project_item(source, origin="external")
    store.upsert_item(item)
    store.save()

    loaded = ProjectStore(project)
    loaded.load()

    assert loaded.items[0].path == source.resolve()
    assert not (project / source.name).exists()


def test_generated_project_scan_detects_linak_outputs(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    write_linak_hdf5(
        project / "result.h5",
        analysis="msd",
        datasets={"time_fs": [0.0], "msd": [0.0]},
        metadata={"species": "O"},
    )
    with h5py.File(project / "ordinary.h5", "w") as handle:
        handle.create_dataset("x", data=[1])

    discovered = discover_generated_items(project)

    assert [item.path.name for item in discovered] == ["result.h5"]
    assert discovered[0].item_type == "analysis_hdf5"


def test_workspace_index_reuses_unchanged_detection_and_revalidates_changed_file(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    output = project / "result.h5"
    write_linak_hdf5(
        output,
        analysis="msd",
        datasets={"time_fs": [0.0], "msd": [0.0]},
        metadata={"species": "O"},
    )
    index = WorkspaceIndex()

    first = discover_generated_items_cached(project, index=index)
    cached = index.cached_item(output)
    second = discover_generated_items_cached(project, index=index)

    assert first[0].metadata["analysis"] == "msd"
    assert cached is not None
    assert second[0].metadata["analysis"] == "msd"

    write_linak_hdf5(
        output,
        analysis="density",
        datasets={"x": [0.0], "density": [1.0]},
        metadata={"species": "O"},
    )
    refreshed = discover_generated_items_cached(project, index=index)

    assert refreshed[0].metadata["analysis"] == "density"


def test_action_registry_filters_by_project_item_type(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")

    action_ids = {action.action_id for action in ActionRegistry().available_for(item)}

    assert {"convert", "density", "msd", "rdf", "position", "coordination", "orientation", "pbc"} <= action_ids
    assert "open_plot" not in action_ids


def test_action_settings_validation_rejects_invalid_values(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")
    action = ActionRegistry().by_id("density")

    validate_action_settings(
        action,
        item,
        {"species": "O", "axis": "z", "bin_width": 0.1, "outputs": "1d"},
    )

    try:
        validate_action_settings(
            action,
            item,
            {"species": "O", "axis": "q", "bin_width": -1, "outputs": "1d"},
        )
    except ValueError as exc:
        assert "Axis" in str(exc) or "Bin width" in str(exc)
    else:
        raise AssertionError("invalid density settings were accepted")


def test_expected_output_naming_versions_collisions(tmp_path):
    project = tmp_path / "workspace"
    project.mkdir()
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")
    action = ActionRegistry().by_id("density")
    existing = project / "traj.density.h5"
    existing.write_text("occupied", encoding="utf-8")

    outputs = action.expected_outputs(
        project_dir=project,
        item=item,
        settings={"species": "O", "axis": "z", "bin_width": 0.1, "outputs": "1d"},
    )

    assert outputs == (project / "traj_1.density.h5",)


def test_gui_density_defaults_match_current_density_engine(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")
    action = ActionRegistry().by_id("density")
    defaults = default_settings_for_action(action, item, tmp_path)

    assert defaults["outputs"] == "all"
    assert defaults["oh_cutoff"] == 1.27
    assert defaults["min_molecule_frames"] == 5
    fields = {field.key: field for field in action.settings_schema(item)}
    assert fields["outputs"].choices == ("1d", "3d", "all")
    assert "heatmap_planes" not in fields
    assert "atom_alias" in fields


def test_viewmodels_build_guided_display_rows(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")

    display = display_for_item(item)
    suggestions = suggested_actions_for_item(item, ActionRegistry())
    filtered = filter_items([item], query="traj", origin="external", item_type="raw_trajectory")

    assert display.icon == "[T]"
    assert [badge.text for badge in display.badges] == ["external", "valid"]
    assert suggestions[0].action_id == "convert"
    assert filtered == [item]


def test_task_display_counts_log_severity_and_outputs(tmp_path):
    task = Task(action_id="density", action_name="Density", input_item_id="input")
    task.status = "failed"
    task.error = "bad settings"
    task.output_paths = [tmp_path / "density.h5"]
    task.add_log("INFO", "started")
    task.add_log("WARNING", "careful")
    task.add_log("ERROR", "failed")

    display = display_for_task(task)

    assert display.status_badge.text == "failed"
    assert display.log_counts["INFO"] == 1
    assert display.log_counts["WARNING"] == 1
    assert display.log_counts["ERROR"] == 1
    assert display.output_labels == ("density.h5",)


def test_task_progress_fraction_and_display_label():
    task = Task(action_id="pack_out_h5", action_name="Pack", input_item_id="input")

    assert task.progress_fraction is None
    task.set_progress("Packing cubes", 2, 4)

    assert task.progress_fraction == 0.5
    assert task_progress_label(task) == "50% - Packing cubes"

    task.set_progress("Unknown work", None, None)

    assert task.progress_fraction is None
    assert task_progress_label(task) == "Unknown work"

    task.set_progress("Bad total", 1, 0)

    assert task.progress_fraction is None


def test_running_badge_style_changes_with_progress():
    empty = badge_style("running", progress_fraction=0.0)
    half = badge_style("running", progress_fraction=0.5)
    done = badge_style("running", progress_fraction=1.0)

    assert "progress 0%" in empty
    assert "progress 50%" in half
    assert "progress 100%" in done
    assert empty != half
    assert half != done


def test_project_workspace_uses_plot_gui_theme_switch_pattern():
    source = Path("src/linak/gui/workspace.py").read_text(encoding="utf-8")

    assert 'self._theme_switch = QCheckBox("Dark mode")' in source
    assert 'self._theme_switch.setObjectName("themeSwitch")' in source
    assert 'self._theme_mode = "system"' in source
    assert '"dark" if checked else "light"' in source
    assert "_gui_theme_mode" not in source


def test_project_theme_tokens_and_stylesheet_match_plotting_gui_contract():
    tokens = plot_like_theme_tokens(False)
    required = {
        "window_bg",
        "header_bg",
        "panel_bg",
        "card_bg",
        "panel_elevated",
        "border",
        "border_soft",
        "text",
        "heading",
        "muted_text",
        "accent",
        "accent_soft",
        "warning_bg",
        "badge_bg",
    }

    assert required <= set(tokens)
    stylesheet = workspace_stylesheet(tokens)
    assert "QCheckBox#themeSwitch" in stylesheet
    assert "QMessageBox" in stylesheet


def test_running_badge_style_uses_theme_tokens():
    dark = plot_like_theme_tokens(True)
    style = badge_style("running", progress_fraction=0.5, colors=dark)

    assert dark["accent"] in style
    assert dark["accent_hover"] in style
    assert "progress 50%" in style


def test_running_task_display_includes_progress():
    task = Task(action_id="density", action_name="Density", input_item_id="input")
    task.status = "running"
    task.set_progress("Reading trajectory", 3, 10)

    display = display_for_task(task)

    assert display.status_badge.text == "running"
    assert display.progress_fraction == 0.3
    assert display.progress_label == "30% - Reading trajectory"
    assert "running 30% - Reading trajectory" in display.subtitle


def test_out_h5_gui_summary_and_defaults_are_metadata_driven(tmp_path):
    output = tmp_path / "run.out.h5"
    _write_minimal_out_h5(output, trajectory=True, cubes=2)
    item = detect_project_item(output, origin="generated")
    action = ActionRegistry().by_id("coordination")

    summary = out_h5_gui_summary_for_item(item)
    defaults = default_settings_for_action(action, item, tmp_path)

    assert summary is not None
    assert summary.species == ("Li", "O")
    assert summary.cell_angstrom == (10.0, 11.0, 12.0)
    assert summary.cell_matrix_angstrom == (
        (10.0, 0.0, 0.0),
        (0.0, 11.0, 0.0),
        (0.0, 0.0, 12.0),
    )
    assert summary.pbc == (True, True, False)
    assert summary.timestep_fs == 0.5
    assert summary.cube_count == 2
    assert summary.trajectory_source_path.endswith("traj.xyz")
    assert summary.frame_range == (0, 11)
    assert summary.cube_kinds == ("density",)
    assert summary.cp2k_table_counts["md_steps"] == 3
    assert summary.discovery_summary["trajectories"] == ["traj.xyz"]
    assert defaults["species_a"] == "Li"
    assert defaults["species_b"] == "O"
    assert defaults["cell"] == "10 11 12"
    assert defaults["timestep_fs"] == 0.5


def test_gui_action_settings_snapshot_and_hash_are_stable(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    project = tmp_path / "project"
    project.mkdir()
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")
    action = ActionRegistry().by_id("density")
    settings = {"species": "O", "axis": "z", "bin_width": 0.1, "outputs": "1d"}

    gui_settings = build_gui_action_settings(action, item, settings, project_dir=project)

    assert gui_settings.to_backend_dict() == settings
    assert gui_settings.settings_hash == settings_hash("density", settings)
    assert gui_settings.collision_policy == "auto-version"
    assert gui_settings.output_paths == (project / "traj.density.h5",)


def test_position_gui_backend_forwards_oh_molecule_options(tmp_path, monkeypatch):
    import linak.gui.actions as actions_mod

    project = tmp_path / "project"
    project.mkdir()
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")
    action = ActionRegistry().by_id("position")
    captured: dict[str, object] = {}

    def _fake_run(ctx, argv, expected_outputs):
        captured["argv"] = tuple(argv)
        captured["expected_outputs"] = tuple(expected_outputs)
        return ActionExecutionResult(output_paths=tuple(expected_outputs))

    monkeypatch.setattr(actions_mod, "_run_cli_with_expected_outputs", _fake_run)

    result = action.backend(
        ActionContext(
            project_dir=project,
            item=item,
            settings={
                "species": "molecules",
                "axis": "z",
                "oh_cutoff": 1.15,
                "min_molecule_frames": 2,
                "oh_topology_stride": 3,
            },
            log=lambda _level, _message: None,
            progress=lambda _label, _current, _total: None,
        )
    )

    assert result.output_paths == (project / "traj.position.h5",)
    assert captured["argv"] == (
        "compute",
        "position",
        str(trajectory.resolve()),
        "--output",
        str(project / "traj.position.h5"),
        "--species",
        "molecules",
        "--axis",
        "z",
        "--oh-cutoff",
        "1.15",
        "--min-molecule-frames",
        "2",
        "--oh-topology-stride",
        "3",
    )


def test_density_gui_backend_forwards_atom_aliases(tmp_path, monkeypatch):
    import linak.gui.actions as actions_mod

    project = tmp_path / "project"
    project.mkdir()
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    item = detect_project_item(trajectory, origin="external")
    action = ActionRegistry().by_id("density")
    captured: dict[str, object] = {}

    def _fake_run(ctx, argv, expected_outputs):
        captured["argv"] = tuple(argv)
        return ActionExecutionResult(output_paths=tuple(expected_outputs))

    monkeypatch.setattr(actions_mod, "_run_cli_with_expected_outputs", _fake_run)

    action.backend(
        ActionContext(
            project_dir=project,
            item=item,
            settings={
                "species": "all",
                "axis": "z",
                "bin_width": 0.1,
                "outputs": "all",
                "atom_alias": "Ow=O Pt_top=Pt",
            },
            log=lambda _level, _message: None,
            progress=lambda _label, _current, _total: None,
        )
    )

    argv = captured["argv"]
    assert "--atom-alias" in argv
    assert argv.count("--atom-alias") == 2
    assert "Ow=O" in argv
    assert "Pt_top=Pt" in argv


def test_component_viewmodels_group_items_and_flag_outputs(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    container = project / "run.out.h5"
    output = project / "run.density.h5"
    _write_minimal_out_h5(container, trajectory=True, cubes=0)
    write_linak_hdf5(
        output,
        analysis="density",
        datasets={"x": [0.0], "density": [1.0]},
        metadata={"species": "O"},
    )
    store = ProjectStore(project)
    out_item = detect_project_item(container, origin="generated")
    analysis_item = detect_project_item(output, origin="generated")
    store.upsert_item(out_item)
    store.upsert_item(analysis_item)
    action = ActionRegistry().by_id("density")

    rows = grouped_item_rows(store, mode="Workflow")
    display = action_row_display(
        action=action,
        item=out_item,
        store=store,
        latest_task=None,
        cancel_capability=descriptor_for_action(action).cancel_capability,
    )

    assert rows[0].group_label == "2. Output containers"
    assert display.output_preview == ("run_1.density.h5",)
    assert display.can_run_defaults
    assert display.cancel_capability == "limited"


def test_task_manager_pause_reorder_and_task_settings_snapshot(tmp_path):
    store = ProjectStore(tmp_path / "project")
    store.initialize()
    input_path = tmp_path / "input.xyz"
    _write_xyz(input_path)
    item = detect_project_item(input_path, origin="external")
    store.upsert_item(item)
    starts: list[str] = []

    def backend(ctx: ActionContext) -> ActionExecutionResult:
        starts.append(str(ctx.settings.get("marker")))
        return ActionExecutionResult()

    action = Action(
        "marked",
        "Compute",
        "Marked",
        "Marked action",
        frozenset({"raw_trajectory"}),
        None,
        lambda _item: [SettingField("marker", "Marker", "text")],
        backend,
    )
    manager = TaskManager(store)
    manager.pause_queue()
    first = manager.start(action=action, item=item, settings={"marker": "first"}, priority=0)
    second = manager.start(action=action, item=item, settings={"marker": "second"}, priority=10)

    assert first.status == "queued"
    assert second.status == "queued"
    assert second.settings_snapshot == {"marker": "second"}
    assert second.settings_hash
    assert manager.reorder_queued(first.task_id, -1)

    manager.resume_queue()
    deadline = time.time() + 3
    while len(starts) < 2 and time.time() < deadline:
        time.sleep(0.01)

    assert starts == ["first", "second"]


def test_task_detail_display_includes_reproducibility_fields():
    task = Task(
        action_id="density",
        action_name="Density",
        input_item_id="input",
        settings_snapshot={"species": "O"},
        settings_hash="abc123",
        cancel_capability="limited",
    )

    detail = task_detail_display(task)

    assert detail.settings_hash == "abc123"
    assert detail.settings_lines == ("species: O",)
    assert detail.cancel_capability == "limited"


def test_action_readiness_reports_missing_out_h5_components(tmp_path):
    output = tmp_path / "run.out.h5"
    _write_minimal_out_h5(output, trajectory=False, cubes=0)
    item = detect_project_item(output, origin="generated")
    registry = ActionRegistry()

    density = readiness_for_action(registry.by_id("density"), item)
    potential = readiness_for_action(registry.by_id("potential"), item)

    assert not density.available
    assert "trajectory" in density.reason
    assert not potential.available
    assert "cube" in potential.reason


def test_project_store_removes_items_prunes_relationships_and_restricts_delete(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = ProjectStore(project)
    external_path = tmp_path / "input.xyz"
    generated_path = project / "density.h5"
    external_path.write_text("x", encoding="utf-8")
    generated_path.write_text("generated", encoding="utf-8")

    external = detect_project_item(external_path, origin="external")
    generated = detect_project_item(generated_path, origin="generated")
    external.relationships["outputs"] = [generated.item_id]
    generated.relationships["inputs"] = [external.item_id]
    store.upsert_item(external)
    store.upsert_item(generated)
    store.workspace_index.remember(generated)

    removed = store.remove_item(generated.item_id)

    assert removed is not None
    assert external.relationships == {}
    assert store.workspace_index.cached_item(generated_path) is None
    assert generated_path.exists()

    store.upsert_item(generated)
    deleted = store.delete_generated_item_file(generated.item_id)

    assert deleted == generated_path.resolve()
    assert not generated_path.exists()


def test_task_manager_queues_and_cancels_queued_task(tmp_path):
    store = ProjectStore(tmp_path / "project")
    store.initialize()
    input_path = tmp_path / "input.xyz"
    _write_xyz(input_path)
    item = detect_project_item(input_path, origin="external")
    store.upsert_item(item)
    started = threading.Event()
    release = threading.Event()

    def backend(ctx: ActionContext) -> ActionExecutionResult:
        started.set()
        release.wait(timeout=3)
        return ActionExecutionResult()

    action = Action(
        "slow",
        "Compute",
        "Slow",
        "Slow action",
        frozenset({"raw_trajectory"}),
        None,
        lambda _item: [],
        backend,
    )
    manager = TaskManager(store)
    first = manager.start(action=action, item=item, settings={})
    assert started.wait(timeout=3)
    second = manager.start(action=action, item=item, settings={})

    assert second.status == "queued"
    assert manager.cancel(second.task_id)
    assert second.status == "canceled"

    release.set()
    deadline = time.time() + 3
    while first.status not in {"finished", "failed"} and time.time() < deadline:
        time.sleep(0.01)
    assert first.status == "finished"


def test_task_manager_cooperative_running_cancel(tmp_path):
    store = ProjectStore(tmp_path / "project")
    store.initialize()
    input_path = tmp_path / "input.xyz"
    _write_xyz(input_path)
    item = detect_project_item(input_path, origin="external")
    store.upsert_item(item)
    started = threading.Event()

    def backend(ctx: ActionContext) -> ActionExecutionResult:
        started.set()
        deadline = time.time() + 3
        while time.time() < deadline:
            if ctx.cancel_requested():
                raise RuntimeError("noticed cancel")
            time.sleep(0.01)
        return ActionExecutionResult()

    action = Action(
        "cancelable",
        "Compute",
        "Cancelable",
        "Cancelable action",
        frozenset({"raw_trajectory"}),
        None,
        lambda _item: [],
        backend,
    )
    manager = TaskManager(store)
    task = manager.start(action=action, item=item, settings={})
    assert started.wait(timeout=3)

    assert manager.cancel(task.task_id)
    deadline = time.time() + 3
    while task.status not in {"canceled", "failed"} and time.time() < deadline:
        time.sleep(0.01)

    assert task.status == "canceled"


def test_conversion_action_writes_output_inside_project_dir(tmp_path):
    source = tmp_path / "input" / "traj.xyz"
    project = tmp_path / "workspace"
    source.parent.mkdir()
    project.mkdir()
    _write_xyz(source)
    item = detect_project_item(source, origin="external")
    action = ActionRegistry().by_id("convert")
    logs: list[tuple[str, str]] = []

    result = action.backend(
        ActionContext(
            project_dir=project,
            item=item,
            settings={"target_file_type": "traj.h5", "select": None, "input": None, "cell": None},
            log=lambda level, message: logs.append((level, message)),
            progress=lambda _label, _current, _total: None,
        )
    )

    assert len(result.output_paths) == 1
    assert result.output_paths[0].parent == project
    assert result.output_paths[0].name == "traj.traj.h5"
    assert result.output_paths[0].exists()
