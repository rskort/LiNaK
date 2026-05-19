import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

import linak.cli as cli_mod
from linak.cli import (
    _build_coordination_profile_filter_options,
    _build_density_gui_context,
    _build_gui_series_descriptors,
    _build_rdf_gui_context,
    _force_source_ids_enabled_for_gui_loading,
    _gui_series_descriptors_from_settings,
    _merge_gui_series_descriptors,
    _required_source_ids_for_gui_render,
    _build_rdf_profile_filter_options,
    _default_rdf_collection_hdf5_output_path,
    _combine_analysis_hdf5_sources,
    _load_rdf_plot_profiles,
    _resolve_density_plotter_kwargs,
    _resolve_coordination_plotter_kwargs,
    _resolve_msd_plotter_kwargs,
    _resolve_orientation_plotter_kwargs,
    _resolve_potential_plotter_kwargs,
    _resolve_rdf_plotter_kwargs,
    _without_preview_series_state,
    _rewrite_implicit_csv_interactive,
    _rewrite_implicit_plot_csv,
    build_parser,
    main,
)
from linak.analysis.density import (
    compute_all_density_profiles,
    compute_density_profile,
    load_density_heatmap_profiles,
    load_density_profile,
    load_density_profiles,
    load_density_profiles_by_index,
    save_density_profile,
    save_density_profiles,
)
from linak.analysis.position import (
    PositionProfile,
    compute_position_profile,
    load_position_profile,
    load_position_profiles_by_index,
    save_position_profile,
)
from linak.analysis.coordination import (
    CoordinationCutoffResolution,
    compute_coordination_profile,
    load_coordination_profiles,
    save_coordination_profile,
)
from linak.storage.hdf5_table import read_hdf5_frame
from linak.storage.hdf5_utils import (
    read_linak_hdf5,
    read_linak_hdf5_profiles,
    write_linak_hdf5_profile_collection,
)
from linak.analysis.msd import compute_msd, load_msd_profile, save_msd_profile
from linak.plot.plot_settings import (
    read_active_plot_profile_name,
    read_plot_profile,
    read_plot_profile_names,
    write_plot_profile,
)
from linak.plot.profile_persistence import (
    build_plot_profile_payload,
    flatten_plot_profile_payload,
)
from linak.plot.data_contract import PlotViewMapping
from linak.analysis.rdf import (
    RDFProfile,
    compute_rdf,
    load_rdf_profile,
    load_rdf_profiles,
    plot_rdf_profile,
    save_rdf_profile,
)
from linak.trajectory.io import (
    TrajectoryStoredMetadata,
    read_trajectory,
    read_trajectory_hdf5_metadata,
    read_trajectory_hdf5_surface_cache,
    write_trajectory,
)


def _write_xyz(path: Path) -> None:
    frame0 = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]])
    frame1 = Atoms("OO", positions=[[0.0, 0.0, 0.12], [0.0, 0.0, 0.18]])
    write(path, [frame0, frame1], format="extxyz")


def _write_xyz_frames(path: Path, *, z_values: list[float]) -> None:
    frames = [Atoms("O", positions=[[0.0, 0.0, float(z)]]) for z in z_values]
    write(path, frames, format="extxyz")


def _write_xyz_custom_frames(path: Path, frames: list[Atoms]) -> None:
    write(path, frames, format="extxyz")


def _write_traj_h5_with_time_metadata(
    path: Path,
    *,
    z_values: list[float],
    frame_timestep_fs: float | None = None,
    timestep_stride: int | None = None,
) -> None:
    frames = []
    for index, z_value in enumerate(z_values):
        frame = Atoms("O", positions=[[0.0, 0.0, float(z_value)]])
        if frame_timestep_fs is not None:
            frame.info["time_fs"] = float(index) * float(frame_timestep_fs)
            frame.info["frame_timestep_fs"] = float(frame_timestep_fs)
        if timestep_stride is not None:
            frame.info["timestep"] = int(index) * int(timestep_stride)
            frame.info["trajectory_stride_md"] = int(timestep_stride)
        frames.append(frame)
    write_trajectory(frames, path)


def _write_surface_xyz(path: Path) -> None:
    symbols = "Au8"
    base_positions = np.array(
        [
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [1.0, 1.0, 0.1],
            [0.0, 0.0, 2.1],
            [1.0, 0.0, 2.1],
            [0.0, 1.0, 2.1],
            [1.0, 1.0, 2.1],
        ],
        dtype=float,
    )
    frames = [
        Atoms(symbols, positions=base_positions.copy()),
        Atoms(symbols, positions=base_positions + np.array([0.0, 0.0, 0.05])),
    ]
    write(path, frames, format="extxyz")


def _saved_plot_profile(profile_key: str, settings: dict[str, object]) -> dict[str, object]:
    return build_plot_profile_payload(profile_key, dict(settings))


def _read_flat_plot_profile(path: Path, profile_key: str, *, profile_name: str | None = None):
    payload = read_plot_profile(path, profile_key, profile_name=profile_name)
    if payload is None:
        return None
    return flatten_plot_profile_payload(profile_key, payload)


def _write_lammps_dump(path: Path, *, positions: list[tuple[float, float, float]]) -> None:
    steps = [index * 10 for index in range(len(positions))]
    blocks: list[str] = []
    for step, (x, y, z) in zip(steps, positions):
        blocks.append("ITEM: TIMESTEP")
        blocks.append(str(step))
        blocks.append("ITEM: NUMBER OF ATOMS")
        blocks.append("1")
        blocks.append("ITEM: BOX BOUNDS pp pp pp")
        blocks.append("0.0 1.0")
        blocks.append("0.0 1.0")
        blocks.append("0.0 1.0")
        blocks.append("ITEM: ATOMS id type element xu yu zu vx vy vz")
        blocks.append(f"1 1 O {x} {y} {z} 0.0 0.0 0.0")
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _write_lammps_input(
    path: Path, *, dump_name: str = "lammps.dump", dump_every: int = 10
) -> None:
    path.write_text(
        "units metal\n"
        "boundary p p p\n"
        "atom_style atomic\n"
        "timestep 0.001\n"
        f"dump d1 all custom {dump_every} {dump_name} id type element xu yu zu vx vy vz\n",
        encoding="utf-8",
    )


def _write_cp2k_input(path: Path, *, timestep_fs: float = 0.5, stride_md: int = 5) -> None:
    path.write_text(
        "&SUBSYS\n"
        "  &CELL\n"
        "    ABC 17.887 15.491 59.671\n"
        "  &END CELL\n"
        "&END SUBSYS\n"
        "&MOTION\n"
        "  &MD\n"
        f"    TIMESTEP [fs] {timestep_fs}\n"
        "  &END MD\n"
        "  &PRINT\n"
        "    &TRAJECTORY\n"
        "      &EACH\n"
        f"        MD {stride_md}\n"
        "      &END EACH\n"
        "    &END TRAJECTORY\n"
        "  &END PRINT\n"
        "&END MOTION\n",
        encoding="utf-8",
    )


def _write_minimal_cp2k_output(path: Path) -> None:
    path.write_text(
        " CP2K| version string\n GLOBAL| Run type ENERGY\n PROGRAM ENDED AT 2026-01-01 00:00:00\n",
        encoding="utf-8",
    )


def _write_simple_hdf5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["linak_format_version"] = 1
        handle.attrs["analysis"] = "table"
        handle.attrs["created_utc"] = "2026-03-12T00:00:00+00:00"
        handle.attrs["linak_version"] = "0.5.0"
        handle.attrs["metadata_json"] = '{"source":"unit-test","note":"simple"}'
        records = handle.create_group("records")
        records.create_dataset("step", data=np.asarray([2, 0, 1], dtype=int))
        records.create_dataset("value", data=np.asarray([3.0, 1.0, 2.0], dtype=float))
        records.create_dataset(
            "label",
            data=np.asarray(["beta", "alpha", "alpha"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _linak_output_dir(path: Path) -> Path:
    return path / "LiNaK_outputs"


def test_read_project_author_falls_back_to_installed_package_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_project_pyproject_path", lambda: tmp_path / "missing.toml")
    monkeypatch.setattr(cli_mod, "package_metadata", lambda _name: {"Author": "R.S. Kort"})

    assert cli_mod._read_project_author(default="Unknown") == "R.S. Kort"


def _write_density_hdf5(path: Path, *, axis: str = "z") -> None:
    frame0 = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.12], [0.0, 0.0, 0.18]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame0, frame1], species="O", axis=axis, bin_width=0.1)
    save_density_profile(profile, path)


def _write_density_collection_hdf5(path: Path, *, species: str = "all", surface_axis: str = "z") -> None:
    frame0 = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.02],
            [0.8, 0.0, 0.02],
            [-0.4, 0.7, 0.02],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.18],
            [0.8, 0.0, 0.18],
            [-0.4, 0.7, 0.18],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profiles = compute_all_density_profiles(
        [frame0, frame1],
        species=species,
        surface_axis=surface_axis,
        bin_width=0.1,
        outputs="all",
    )
    save_density_profiles(profiles, path)


def _write_position_hdf5(path: Path) -> None:
    frame0 = Atoms(
        "PtPtOO",
        positions=[
            [0.0, 0.0, 0.20],
            [1.0, 0.0, 0.20],
            [0.0, 0.0, 1.00],
            [1.0, 0.0, 1.50],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "PtPtOO",
        positions=[
            [0.0, 0.0, 0.30],
            [1.0, 0.0, 0.30],
            [0.0, 0.0, 1.20],
            [1.0, 0.0, 1.70],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_position_profile(
        [frame0, frame1],
        species="O",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
    )
    save_position_profile(profile, path)


def _write_large_position_hdf5(path: Path, *, n_atoms: int, n_frames: int = 3) -> None:
    frame_index = np.arange(n_frames, dtype=int)
    step = np.arange(n_frames, dtype=float)
    time_fs = np.arange(n_frames, dtype=float) * 1000.0
    time_ps = time_fs / 1000.0
    atom_indices = np.arange(n_atoms, dtype=int)
    x = np.tile(np.linspace(0.0, 1.0, n_frames, dtype=float).reshape(-1, 1), (1, n_atoms))
    y = np.tile(np.linspace(0.5, 1.5, n_frames, dtype=float).reshape(-1, 1), (1, n_atoms))
    z = np.tile(np.linspace(1.0, 2.0, n_frames, dtype=float).reshape(-1, 1), (1, n_atoms))
    distance = np.tile(np.linspace(2.0, 3.0, n_frames, dtype=float).reshape(-1, 1), (1, n_atoms))
    profile = PositionProfile(
        species="H",
        axis="z",
        atom_indices=atom_indices,
        frame_index=frame_index,
        step=step,
        time_fs=time_fs,
        time_ps=time_ps,
        x=x,
        y=y,
        z=z,
        distance_to_surface=distance,
        n_frames=n_frames,
        n_atoms=n_atoms,
        coordinate_mode="distance",
        surface_position=0.0,
        surface_position_std=0.0,
        surface_position_per_frame=np.zeros(n_frames, dtype=float),
        surface_estimate=None,
        cell_lengths_angstrom=(10.0, 10.0, 10.0),
    )
    save_position_profile(profile, path)


def _write_coordination_hdf5(path: Path) -> None:
    frame0 = Atoms(
        "PtPtOH",
        positions=[
            [0.0, 0.0, 0.20],
            [1.0, 0.0, 0.20],
            [0.0, 0.0, 1.00],
            [0.70, 0.0, 1.00],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "PtPtOH",
        positions=[
            [0.0, 0.0, 0.30],
            [1.0, 0.0, 0.30],
            [0.0, 0.0, 1.20],
            [1.10, 0.0, 1.20],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_coordination_profile(
        [frame0, frame1],
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.4,
            mode="direct",
        ),
    )
    save_coordination_profile(profile, path)


def _write_potential_hdf5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["analysis"] = "potential"
        records = handle.create_group("records")
        records.create_dataset("id", data=np.asarray([3, 1, 2], dtype=np.int64))
        records.create_dataset(
            "source",
            data=np.asarray(["run3", "run1", "run2"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "source_dir",
            data=np.asarray(["dir3", "dir1", "dir2"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "status",
            data=np.asarray(["ok", "ok", "missing_fermi"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "efermi_ev",
            data=np.asarray([1.3, 1.1, np.nan], dtype=float),
        )
        records.create_dataset(
            "water_bulk_potential_ev",
            data=np.asarray([2.3, 2.1, 2.2], dtype=float),
        )
        records.create_dataset(
            "electrode_cshe_ev",
            data=np.asarray([0.2, 0.1, np.nan], dtype=float),
        )


def _read_table_frame(path: Path):
    frame, _info = read_hdf5_frame(path)
    return frame


def test_root_help_mentions_plot_and_compute(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "plot" in out
    assert "compute" in out
    assert "apply" in out
    assert "hdf5" in out


def test_filter_plotter_kwargs_drops_unsupported_entries():
    def _plotter(*, title: str | None = None) -> None:
        del title

    filtered = cli_mod._filter_plotter_kwargs(
        _plotter,
        {
            "title": "Example",
            "annotations": [{"kind": "text", "text": "x", "x": 0.1, "y": 0.2}],
        },
    )

    assert filtered == {"title": "Example"}


def test_filter_plotter_kwargs_keeps_kwargs_for_var_keyword_plotter():
    def _plotter(**kwargs: object) -> None:
        del kwargs

    payload = {
        "title": "Example",
        "annotations": [{"kind": "text", "text": "x", "x": 0.1, "y": 0.2}],
    }
    filtered = cli_mod._filter_plotter_kwargs(_plotter, payload)

    assert filtered == payload


def test_filter_plotter_kwargs_keeps_integration_config_for_profile_plotters():
    config = {"enabled": True, "source": "plotted", "target": "selected"}

    filtered = cli_mod._filter_plotter_kwargs(
        plot_rdf_profile,
        {"integration_config": config, "unsupported": "drop-me"},
    )

    assert filtered == {"integration_config": config}


def test_root_command_without_args_shows_overview(capsys):
    rc = main(["--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LiNaK Command Center" in out
    assert "Core workflow" in out


def test_hdf5_command_without_subcommand_shows_overview(capsys):
    rc = main(["--log-level", "ERROR", "hdf5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LiNaK HDF5 Table Usage" in out
    assert "interactive, info, preview, get, sort, filter, dedupe, combine, plot" in out


def test_csv_command_alias_is_removed():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["csv"])


def test_plot_command_without_subcommand_shows_overview(capsys):
    rc = main(["--log-level", "ERROR", "plot"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LiNaK Plot Usage (HDF5-only)" in out
    assert "linak plot /path/to/traj_density.h5" in out


def test_compute_command_without_subcommand_shows_overview(capsys):
    rc = main(["--log-level", "ERROR", "compute"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LiNaK Compute Usage" in out
    assert "linak compute density" in out


def test_apply_command_without_subcommand_shows_overview(capsys):
    rc = main(["--log-level", "ERROR", "apply"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LiNaK Apply Usage" in out
    assert "linak apply convert" in out
    assert "linak apply pbc" in out
    assert "linak apply compress" in out


def test_apply_convert_help_lists_hdf5_output(tmp_path, capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["apply", "convert", "--help"])

    out = capsys.readouterr().out
    assert "preferred HDF5" in out
    assert "--target-file-type" in out


def test_apply_convert_dry_run_reports_default_output(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "INFO",
            "apply",
            "convert",
            str(trajectory),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "traj.traj.h5" in capsys.readouterr().err


def test_apply_convert_dry_run_reports_explicit_target_file_type(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "INFO",
            "apply",
            "convert",
            str(trajectory),
            "--target-file-type",
            "xyz",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "target file type: trajectory_xyz" in capsys.readouterr().err


def test_apply_convert_writes_traj_hdf5(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
        ]
    )

    assert rc == 0
    assert (tmp_path / "traj.traj.h5").exists()


def test_apply_convert_traj_hdf5_to_xyz_roundtrip(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(trajectory),
            ]
        )
        == 0
    )

    converted_h5 = tmp_path / "traj.traj.h5"
    out_xyz = tmp_path / "traj_roundtrip.xyz"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(converted_h5),
            "--target-file-type",
            "xyz",
            "--output",
            str(out_xyz),
        ]
    )

    assert rc == 0
    loaded = read_trajectory(out_xyz)
    assert len(loaded) == 2
    assert loaded[1].positions[0, 2] == pytest.approx(0.12)


def test_apply_convert_preferred_hdf5_input_without_target_is_noop(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(trajectory),
            ]
        )
        == 0
    )

    converted_h5 = tmp_path / "traj.traj.h5"
    rc = main(
        [
            "--log-level",
            "INFO",
            "apply",
            "convert",
            str(converted_h5),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert str(converted_h5) in captured.out
    assert "already LiNaK's preferred trajectory working format" in captured.err


def test_apply_convert_rejects_unsupported_target_file_type(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--target-file-type",
            "lammps-output",
        ]
    )

    assert rc == 1


def test_apply_convert_select_first_frames_adds_suffix_and_metadata(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_frames(trajectory, z_values=[0.1, 0.2, 0.3, 0.4, 0.5])

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "first:3f",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_first3f.traj.h5"
    assert converted.exists()
    loaded = read_trajectory(converted)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.1, 0.2, 0.3])
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.selection_user == "first:3f"
    assert metadata.selection_kind == "first"
    assert metadata.selection_unit == "f"
    assert metadata.selection_start_frame == 0
    assert metadata.selection_stop_frame_exclusive == 3
    assert metadata.selection_selected_frame_count == 3


def test_apply_convert_select_last_percent_resolves_to_actual_frame_range(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_frames(trajectory, z_values=[0.1, 0.2, 0.3, 0.4])

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "last:50%",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_last50pct.traj.h5"
    loaded = read_trajectory(converted)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.3, 0.4])
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.selection_start_frame == 2
    assert metadata.selection_stop_frame_exclusive == 4
    assert metadata.selection_selected_frame_count == 2


def test_apply_convert_select_first_ps_uses_time_metadata(tmp_path):
    trajectory = tmp_path / "traj.traj.h5"
    _write_traj_h5_with_time_metadata(
        trajectory,
        z_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        frame_timestep_fs=1000.0,
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "first:2ps",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_first2ps.traj.h5"
    loaded = read_trajectory(converted)
    assert len(loaded) == 3
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.selection_user == "first:2ps"
    assert metadata.selection_resolved_start_time_fs == pytest.approx(0.0)
    assert metadata.selection_resolved_end_time_fs == pytest.approx(2000.0)


def test_apply_convert_select_last_ps_uses_hdf5_file_level_frame_timestep_metadata(tmp_path):
    trajectory = tmp_path / "traj.traj.h5"
    frames = [Atoms("O", positions=[[0.0, 0.0, float(z)]]) for z in [0.1, 0.2, 0.3, 0.4, 0.5]]
    write_trajectory(
        frames,
        trajectory,
        metadata=TrajectoryStoredMetadata(
            frame_timestep_fs=1000.0,
            md_timestep_fs=100.0,
            trajectory_stride_md=10,
            timestep_source="test",
        ),
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "last:2ps",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_last2ps.traj.h5"
    loaded = read_trajectory(converted)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.3, 0.4, 0.5])
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.selection_resolved_start_time_fs == pytest.approx(2000.0)
    assert metadata.selection_resolved_end_time_fs == pytest.approx(4000.0)


def test_apply_convert_select_last_step_uses_step_metadata(tmp_path):
    trajectory = tmp_path / "traj.traj.h5"
    _write_traj_h5_with_time_metadata(
        trajectory,
        z_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        timestep_stride=100,
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "last:200step",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_last200step.traj.h5"
    loaded = read_trajectory(converted)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.3, 0.4, 0.5])
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.selection_resolved_start_step == 200
    assert metadata.selection_resolved_end_step == 400


def test_apply_convert_select_last_ps_falls_back_to_md_timestep_times_stride(tmp_path):
    trajectory = tmp_path / "traj.traj.h5"
    frames = [Atoms("O", positions=[[0.0, 0.0, float(z)]]) for z in [0.1, 0.2, 0.3, 0.4, 0.5]]
    write_trajectory(
        frames,
        trajectory,
        metadata=TrajectoryStoredMetadata(
            md_timestep_fs=500.0,
            trajectory_stride_md=2,
            timestep_source="test",
        ),
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "last:2ps",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_last2ps.traj.h5"
    loaded = read_trajectory(converted)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.3, 0.4, 0.5])
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.selection_resolved_start_time_fs == pytest.approx(2000.0)
    assert metadata.selection_resolved_end_time_fs == pytest.approx(4000.0)


def test_apply_convert_select_range_frames_uses_deterministic_suffix(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_frames(trajectory, z_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "range:1f:4f",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_range1f_4f.traj.h5"
    loaded = read_trajectory(converted)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.2, 0.3, 0.4])


def test_apply_convert_select_time_fails_without_time_metadata(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_frames(trajectory, z_values=[0.1, 0.2, 0.3])

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "first:5ps",
        ]
    )

    assert rc == 1
    assert "time-based trajectory selection requires stored time metadata" in capsys.readouterr().err.lower()


def test_apply_convert_select_step_fails_without_step_metadata(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_frames(trajectory, z_values=[0.1, 0.2, 0.3])

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--select",
            "last:500step",
        ]
    )

    assert rc == 1
    assert "step-based trajectory selection requires stored step metadata" in capsys.readouterr().err.lower()


def test_apply_convert_spatial_filter_x_range_writes_variable_topology_hdf5(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_custom_frames(
        trajectory,
        [
            Atoms(
                "OOO",
                positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                cell=[12.0, 12.0, 12.0],
                pbc=True,
            ),
            Atoms(
                "OOO",
                positions=[[0.5, 0.0, 0.0], [6.5, 0.0, 0.0], [10.0, 0.0, 0.0]],
                cell=[12.0, 12.0, 12.0],
                pbc=True,
            ),
        ],
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--x-range",
            "min:5.0",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_x_0.0_5.0.traj.h5"
    loaded = read_trajectory(converted)
    assert [len(frame) for frame in loaded] == [2, 1]
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.spatial_filter_metadata is not None
    assert metadata.spatial_filter_metadata["used"] is True
    assert metadata.spatial_filter_metadata["bounds"]["x"]["resolved_lower"] == pytest.approx(0.0)
    assert metadata.spatial_filter_metadata["bounds"]["x"]["resolved_upper"] == pytest.approx(5.0)


def test_apply_convert_spatial_filter_keep_molecules_intact_keeps_full_water(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_custom_frames(
        trajectory,
        [
            Atoms(
                "OHH",
                positions=[[0.0, 0.0, 1.10], [0.8, 0.0, 0.60], [-0.8, 0.0, 1.70]],
                cell=[12.0, 12.0, 12.0],
                pbc=True,
            )
        ],
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--z-range",
            "1.0:1.2",
            "--keep-molecules-intact",
        ]
    )

    assert rc == 0
    converted = tmp_path / "traj_z_1.0_1.2.traj.h5"
    loaded = read_trajectory(converted)
    assert len(loaded) == 1
    assert len(loaded[0]) == 3
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.spatial_filter_metadata is not None
    assert metadata.spatial_filter_metadata["keep_molecules_intact"] is True
    assert metadata.spatial_filter_metadata["molecule_selection_mode"] == "water_com_plus_singletons"
    assert metadata.spatial_filter_metadata["retained_molecule_count_total"] == 1


def test_compute_density_spatial_filter_z_range_sets_suffix_and_metadata(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_custom_frames(
        trajectory,
        [
            Atoms(
                "OO",
                positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            ),
            Atoms(
                "OO",
                positions=[[0.0, 0.0, 1.2], [0.0, 0.0, 3.2]],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            ),
        ],
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--z-range",
            "0.5:1.5",
            "--bin-width",
            "0.5",
            "--cell",
            "10",
            "10",
            "10",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density_o_z_0.5_1.5.h5"
    assert output.exists()
    with h5py.File(output, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
    assert metadata["spatial_filter"]["used"] is True
    assert metadata["spatial_filter"]["bounds"]["z"]["resolved_lower"] == pytest.approx(0.5)
    assert metadata["spatial_filter"]["bounds"]["z"]["resolved_upper"] == pytest.approx(1.5)
    assert metadata["spatial_filter"]["retained_atom_count_total"] == 2


def test_compute_density_spatial_filter_distance_range_uses_surface_metadata(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz_custom_frames(
        trajectory,
        [
            Atoms(
                "PtPtOO",
                positions=[
                    [0.0, 0.0, 0.1],
                    [1.0, 0.0, 0.1],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 3.0],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=True,
            ),
            Atoms(
                "PtPtOO",
                positions=[
                    [0.0, 0.0, 0.2],
                    [1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.1],
                    [0.0, 0.0, 3.1],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=True,
            ),
        ],
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--distance-range",
            "0.5:1.5",
            "--bin-width",
            "0.5",
            "--surface-elements",
            "Pt",
            "--cell",
            "12",
            "12",
            "12",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density_o_dist_0.5_1.5.h5"
    assert output.exists()
    with h5py.File(output, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
    assert metadata["spatial_filter"]["used"] is True
    assert metadata["spatial_filter"]["distance"]["surface_axis"] == "z"
    assert metadata["spatial_filter"]["bounds"]["distance"]["resolved_lower"] == pytest.approx(0.5)
    assert metadata["spatial_filter"]["bounds"]["distance"]["resolved_upper"] == pytest.approx(1.5)


def test_apply_combine_xyz_default_writes_traj_hdf5_with_ordered_metadata(tmp_path, monkeypatch):
    source_a = tmp_path / "a.xyz"
    source_b = tmp_path / "b.xyz"
    write(source_a, [Atoms("O", positions=[[0.0, 0.0, 0.1]]), Atoms("O", positions=[[0.0, 0.0, 0.2]])], format="extxyz")
    write(source_b, [Atoms("O", positions=[[0.0, 0.0, 1.1]]), Atoms("O", positions=[[0.0, 0.0, 1.2]])], format="extxyz")
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
        ]
    )

    assert rc == 0
    combined = workdir / "a_combined.traj.h5"
    assert combined.exists()
    loaded = read_trajectory(combined)
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.1, 0.2, 1.1, 1.2])

    metadata = read_trajectory_hdf5_metadata(combined)
    assert metadata is not None
    assert metadata.combine_source_paths == (str(source_a.resolve()), str(source_b.resolve()))
    assert metadata.combine_source_file_types == ("trajectory_xyz", "trajectory_xyz")
    assert metadata.combine_total_frames == 4
    assert metadata.combine_conversion_applied is True
    assert metadata.combine_linak_version
    assert metadata.combine_timestamp_utc

    with h5py.File(combined, "r") as handle:
        metadata_group = handle["metadata"]
        stored_sources = [str(value) for value in metadata_group["combine_source_paths"].asstr()[:]]
        assert stored_sources == [str(source_a.resolve()), str(source_b.resolve())]
        assert int(metadata_group.attrs["combine_total_frames"]) == 4


def test_apply_combine_xyz_no_convert_writes_xyz_and_preserves_order(tmp_path, monkeypatch):
    source_a = tmp_path / "a.xyz"
    source_b = tmp_path / "b.xyz"
    write(source_a, [Atoms("O", positions=[[0.0, 0.0, 0.3]])], format="extxyz")
    write(source_b, [Atoms("O", positions=[[0.0, 0.0, 1.3]])], format="extxyz")
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
            "--no-convert",
        ]
    )

    assert rc == 0
    combined = workdir / "a_combined.xyz"
    assert combined.exists()
    loaded = read_trajectory(combined)
    assert len(loaded) == 2
    assert [frame.positions[0, 2] for frame in loaded] == pytest.approx([0.3, 1.3])


def test_apply_combine_rejects_mixed_families(tmp_path, capsys):
    source_a = tmp_path / "a.xyz"
    source_b = tmp_path / "field.cube"
    write(source_a, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")
    source_b.write_text("CPMD CUBE FILE\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
        ]
    )

    assert rc == 1
    assert "mixed file families" in capsys.readouterr().err


def test_apply_combine_traj_hdf5_auto_detects_consistent_per_source_input_metadata(
    tmp_path, monkeypatch
):
    source_dir_a = tmp_path / "run_a"
    source_dir_b = tmp_path / "run_b"
    source_dir_a.mkdir()
    source_dir_b.mkdir()
    source_a = source_dir_a / "a.xyz"
    source_b = source_dir_b / "b.xyz"
    write(source_a, [Atoms("O", positions=[[2.1, 0.0, 4.3]])], format="extxyz")
    write(source_b, [Atoms("O", positions=[[2.2, 0.0, 4.4]])], format="extxyz")
    _write_cp2k_input(source_dir_a / "input.inp", timestep_fs=0.5, stride_md=5)
    _write_cp2k_input(source_dir_b / "input.inp", timestep_fs=0.5, stride_md=5)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
        ]
    )

    assert rc == 0
    combined = workdir / "a_combined.traj.h5"
    metadata = read_trajectory_hdf5_metadata(combined)
    assert metadata is not None
    assert metadata.cell_angstrom == pytest.approx((17.887, 15.491, 59.671))
    assert metadata.frame_timestep_fs == pytest.approx(2.5)
    assert metadata.md_timestep_fs == pytest.approx(0.5)
    assert metadata.trajectory_stride_md == 5
    assert metadata.pbc_applied is True
    assert metadata.pbc_cell_angstrom == pytest.approx((17.887, 15.491, 59.671))


def test_apply_combine_traj_hdf5_rejects_inconsistent_per_source_input_metadata(
    tmp_path, monkeypatch, capsys
):
    source_dir_a = tmp_path / "run_a"
    source_dir_b = tmp_path / "run_b"
    source_dir_a.mkdir()
    source_dir_b.mkdir()
    source_a = source_dir_a / "a.xyz"
    source_b = source_dir_b / "b.xyz"
    write(source_a, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")
    write(source_b, [Atoms("O", positions=[[0.0, 0.0, 1.1]])], format="extxyz")
    _write_cp2k_input(source_dir_a / "input.inp", timestep_fs=0.5, stride_md=5)
    (source_dir_b / "input.inp").write_text(
        "&SUBSYS\n"
        "  &CELL\n"
        "    ABC 20.0 15.491 59.671\n"
        "  &END CELL\n"
        "&END SUBSYS\n"
        "&MOTION\n"
        "  &MD\n"
        "    TIMESTEP [fs] 0.5\n"
        "  &END MD\n"
        "  &PRINT\n"
        "    &TRAJECTORY\n"
        "      &EACH\n"
        "        MD 5\n"
        "      &END EACH\n"
        "    &END TRAJECTORY\n"
        "  &END PRINT\n"
        "&END MOTION\n",
        encoding="utf-8",
    )
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "inconsistent" in err
    assert "cell metadata" in err


def test_apply_combine_traj_hdf5_explicit_input_overrides_source_specific_metadata(
    tmp_path, monkeypatch
):
    source_dir_a = tmp_path / "run_a"
    source_dir_b = tmp_path / "run_b"
    source_dir_a.mkdir()
    source_dir_b.mkdir()
    source_a = source_dir_a / "a.xyz"
    source_b = source_dir_b / "b.xyz"
    write(source_a, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")
    write(source_b, [Atoms("O", positions=[[0.0, 0.0, 1.1]])], format="extxyz")
    _write_cp2k_input(source_dir_a / "input.inp", timestep_fs=0.5, stride_md=5)
    _write_cp2k_input(source_dir_b / "input.inp", timestep_fs=1.0, stride_md=10)
    override_input = tmp_path / "override.inp"
    override_input.write_text(
        "&SUBSYS\n"
        "  &CELL\n"
        "    ABC 12.0 13.0 14.0\n"
        "  &END CELL\n"
        "&END SUBSYS\n"
        "&MOTION\n"
        "  &MD\n"
        "    TIMESTEP [fs] 0.2\n"
        "  &END MD\n"
        "  &PRINT\n"
        "    &TRAJECTORY\n"
        "      &EACH\n"
        "        MD 4\n"
        "      &END EACH\n"
        "    &END TRAJECTORY\n"
        "  &END PRINT\n"
        "&END MOTION\n",
        encoding="utf-8",
    )
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
            "--input",
            str(override_input),
        ]
    )

    assert rc == 0
    metadata = read_trajectory_hdf5_metadata(workdir / "a_combined.traj.h5")
    assert metadata is not None
    assert metadata.input_path == override_input.resolve()
    assert metadata.cell_angstrom == pytest.approx((12.0, 13.0, 14.0))
    assert metadata.frame_timestep_fs == pytest.approx(0.8)
    assert metadata.md_timestep_fs == pytest.approx(0.2)
    assert metadata.trajectory_stride_md == 4


def test_apply_combine_traj_hdf5_explicit_cell_overrides_auto_detected_cell_only(
    tmp_path, monkeypatch, capsys
):
    source_dir_a = tmp_path / "run_a"
    source_dir_b = tmp_path / "run_b"
    source_dir_a.mkdir()
    source_dir_b.mkdir()
    source_a = source_dir_a / "a.xyz"
    source_b = source_dir_b / "b.xyz"
    write(source_a, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")
    write(source_b, [Atoms("O", positions=[[0.0, 0.0, 1.1]])], format="extxyz")
    _write_cp2k_input(source_dir_a / "input.inp", timestep_fs=0.5, stride_md=5)
    _write_cp2k_input(source_dir_b / "input.inp", timestep_fs=1.0, stride_md=10)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
            "--cell",
            "8.0",
            "9.0",
            "10.0",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "inconsistent" in err
    assert "frame timestep metadata" in err


def test_apply_convert_embeds_input_metadata_into_traj_hdf5(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    input_path = tmp_path / "input.inp"
    _write_xyz(trajectory)
    input_path.write_text(
        "&SUBSYS\n"
        "  &CELL\n"
        "    ABC 17.887 15.491 59.671\n"
        "  &END CELL\n"
        "&END SUBSYS\n"
        "&MOTION\n"
        "  &CONSTRAINT\n"
        "    &FIXED_ATOMS\n"
        "      LIST 1\n"
        "    &END FIXED_ATOMS\n"
        "  &END CONSTRAINT\n"
        "  &MD\n"
        "    TIMESTEP [fs] 0.5\n"
        "  &END MD\n"
        "  &PRINT\n"
        "    &TRAJECTORY\n"
        "      &EACH\n"
        "        MD 5\n"
        "      &END EACH\n"
        "    &END TRAJECTORY\n"
        "  &END PRINT\n"
        "&END MOTION\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--input",
            str(input_path),
        ]
    )

    assert rc == 0
    metadata = read_trajectory_hdf5_metadata(tmp_path / "traj.traj.h5")
    assert metadata is not None
    assert metadata.input_path == input_path.resolve()
    assert metadata.cell_angstrom == pytest.approx((17.887, 15.491, 59.671))
    assert metadata.frame_timestep_fs == pytest.approx(2.5)
    assert metadata.md_timestep_fs == pytest.approx(0.5)
    assert metadata.trajectory_stride_md == 5
    assert metadata.fixed_atom_indices == (0,)


def test_apply_convert_wraps_pbc_and_caches_default_surface(tmp_path):
    trajectory = tmp_path / "surface.xyz"
    input_path = tmp_path / "input.inp"
    _write_surface_xyz(trajectory)
    input_path.write_text(
        "&SUBSYS\n  &CELL\n    ABC 2.0 2.0 4.0\n  &END CELL\n&END SUBSYS\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--input",
            str(input_path),
        ]
    )

    assert rc == 0
    converted = tmp_path / "surface.traj.h5"
    metadata = read_trajectory_hdf5_metadata(converted)
    assert metadata is not None
    assert metadata.pbc_applied is True
    assert metadata.pbc_cell_angstrom == pytest.approx((2.0, 2.0, 4.0))
    assert metadata.coordinate_basis == "pbc-wrapped"
    assert metadata.surface_cache_status == "available"
    assert metadata.surface_cache_axis == "z"
    assert metadata.surface_cache_mode == "auto"

    cached = read_trajectory_hdf5_surface_cache(
        converted,
        axis="z",
        surface_mode="auto",
        surface_elements=None,
        include_fixed_surface_atoms=False,
        rough_surface_envelope_A=None,
        frame_count=2,
    )
    assert cached is not None
    assert cached.frame_values.shape == (2,)
    assert np.all(np.isfinite(cached.frame_values))

    loaded = read_trajectory(converted)
    assert np.all(np.asarray(loaded[0].positions) >= 0.0)
    assert np.all(np.asarray(loaded[0].positions) < np.array([2.0, 2.0, 4.0]))


def test_compute_position_reuses_conversion_cached_pbc_and_surface(tmp_path, monkeypatch):
    import linak.analysis.density as density_module
    import linak.pbc as pbc_module

    trajectory = tmp_path / "surface.xyz"
    input_path = tmp_path / "input.inp"
    _write_surface_xyz(trajectory)
    input_path.write_text(
        "&SUBSYS\n"
        "  &CELL\n"
        "    ABC 2.0 2.0 4.0\n"
        "  &END CELL\n"
        "&END SUBSYS\n"
        "&MOTION\n"
        "  &MD\n"
        "    TIMESTEP [fs] 1.0\n"
        "  &END MD\n"
        "  &PRINT\n"
        "    &TRAJECTORY\n"
        "      &EACH\n"
        "        MD 1\n"
        "      &END EACH\n"
        "    &END TRAJECTORY\n"
        "  &END PRINT\n"
        "&END MOTION\n",
        encoding="utf-8",
    )
    converted = tmp_path / "surface.traj.h5"
    rc_convert = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--input",
            str(input_path),
            "--output",
            str(converted),
        ]
    )
    assert rc_convert == 0

    def _fail_pbc(*_args, **_kwargs):
        raise AssertionError("PBC should not be reapplied for converted trajectory HDF5")

    def _fail_surface(*_args, **_kwargs):
        raise AssertionError("Surface should not be re-estimated for matching cache")

    monkeypatch.setattr(pbc_module, "apply_pbc_to_frames", _fail_pbc)
    monkeypatch.setattr(density_module, "_select_surface_estimate", _fail_surface)

    output = tmp_path / "position.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "position",
            str(converted),
            "--species",
            "Au",
            "--axis",
            "z",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    profile = load_position_profile(output)
    assert profile.coordinate_mode == "distance"
    assert profile.surface_position_per_frame is not None


def test_conversion_surface_cache_mismatch_is_not_used(tmp_path):
    trajectory = tmp_path / "surface.xyz"
    input_path = tmp_path / "input.inp"
    _write_surface_xyz(trajectory)
    input_path.write_text(
        "&SUBSYS\n  &CELL\n    ABC 2.0 2.0 4.0\n  &END CELL\n&END SUBSYS\n",
        encoding="utf-8",
    )
    converted = tmp_path / "surface.traj.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--input",
            str(input_path),
            "--output",
            str(converted),
        ]
    )
    assert rc == 0

    cached = read_trajectory_hdf5_surface_cache(
        converted,
        axis="x",
        surface_mode="auto",
        surface_elements=None,
        include_fixed_surface_atoms=False,
        rough_surface_envelope_A=None,
        frame_count=2,
    )
    assert cached is None


def test_compute_msd_uses_converted_trajectory_hdf5_timestep_metadata_without_input(tmp_path):
    source_dir = tmp_path / "source"
    converted_dir = tmp_path / "converted"
    source_dir.mkdir()
    converted_dir.mkdir()
    trajectory = source_dir / "traj.xyz"
    input_path = source_dir / "input.inp"
    _write_xyz(trajectory)
    _write_cp2k_input(input_path, timestep_fs=0.5, stride_md=5)
    converted_path = converted_dir / "traj.traj.h5"

    rc_convert = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--output",
            str(converted_path),
            "--input",
            str(input_path),
        ]
    )
    assert rc_convert == 0

    output = converted_dir / "traj_msd_o.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "msd",
            str(converted_path),
            "--species",
            "O",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    profile = load_msd_profile(output)
    assert profile.time_fs[1] == pytest.approx(2.5)


def test_compute_rdf_uses_converted_trajectory_hdf5_cell_metadata_without_input(tmp_path):
    source_dir = tmp_path / "source"
    converted_dir = tmp_path / "converted"
    source_dir.mkdir()
    converted_dir.mkdir()
    trajectory = source_dir / "traj.xyz"
    input_path = source_dir / "input.inp"
    _write_xyz(trajectory)
    _write_cp2k_input(input_path)
    converted_path = converted_dir / "traj.traj.h5"

    rc_convert = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "convert",
            str(trajectory),
            "--output",
            str(converted_path),
            "--input",
            str(input_path),
        ]
    )
    assert rc_convert == 0

    output = converted_dir / "rdf.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(converted_path),
            "--species-a",
            "O",
            "--species-b",
            "O",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    profile = load_rdf_profile(output)
    assert profile.species_a == "O"
    assert profile.species_b == "O"


def test_compute_density_accepts_converted_trajectory_hdf5(tmp_path):
    trajectory_h5 = tmp_path / "traj.traj.h5"
    frames = [
        Atoms(
            "OO",
            positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OO",
            positions=[[0.0, 0.0, 0.12], [0.0, 0.0, 0.18]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    write_trajectory(frames, trajectory_h5)
    output = tmp_path / "density.h5"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory_h5),
            "--species",
            "O",
            "--axis",
            "z",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    profile = load_density_profile(output)
    assert profile.species == "O"


def test_compute_density_logs_convert_hint_for_raw_text_trajectory(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    output = tmp_path / "density_raw.h5"

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert "linak apply convert" in capsys.readouterr().err


def test_compute_density_skips_convert_hint_for_converted_trajectory(tmp_path, caplog):
    trajectory_h5 = tmp_path / "traj.traj.h5"
    frames = [
        Atoms(
            "OO",
            positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OO",
            positions=[[0.0, 0.0, 0.12], [0.0, 0.0, 0.18]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    write_trajectory(frames, trajectory_h5)
    output = tmp_path / "density_converted.h5"

    with caplog.at_level("INFO"):
        rc = main(
            [
                "--log-level",
                "INFO",
                "compute",
                "density",
                str(trajectory_h5),
                "--species",
                "O",
                "--axis",
                "z",
                "--output",
                str(output),
            ]
        )

    assert rc == 0
    assert "linak apply convert" not in caplog.text


def test_csv_get_prints_numeric_metrics(tmp_path, capsys):
    source = tmp_path / "table.h5"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "get",
            str(source),
            "--column",
            "value",
            "--metric",
            "mean",
            "std",
            "min",
            "max",
            "--round",
            "5",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Column: value (numeric)" in out
    assert "mean" in out
    assert "std" in out
    assert "min" in out
    assert "max" in out


def test_csv_sort_writes_sorted_output(tmp_path):
    source = tmp_path / "table.h5"
    output = tmp_path / "sorted.h5"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "sort",
            str(source),
            "--by",
            "step",
            "--mode",
            "numeric",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    frame = _read_table_frame(output)
    assert frame["step"].tolist() == [0, 1, 2]
    assert frame["value"].tolist() == [1.0, 2.0, 3.0]
    assert frame["label"].tolist() == ["alpha", "alpha", "beta"]


def test_csv_filter_writes_subset_output(tmp_path):
    source = tmp_path / "table.h5"
    output = tmp_path / "filtered.h5"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "filter",
            str(source),
            "--column",
            "label",
            "--op",
            "eq",
            "--value",
            "alpha",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    frame = _read_table_frame(output)
    assert len(frame) == 2
    assert frame["label"].tolist() == ["alpha", "alpha"]


def test_csv_preview_accepts_single_source_via_files_option(tmp_path, capsys):
    source = tmp_path / "table.h5"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "preview",
            "-f",
            str(source),
            "--rows",
            "2",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Preview:" in out
    assert "head 2" in out


def test_hdf5_info_prints_metadata_overview(tmp_path, capsys):
    source = tmp_path / "table.h5"
    _write_simple_hdf5(source)

    rc = main(["--log-level", "ERROR", "hdf5", "info", str(source)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Metadata overview" in out
    assert "analysis       : table" in out
    assert "created_utc    : 2026-03-12T00:00:00+00:00" in out
    assert "linak_version  : 0.5.0" in out


def test_hdf5_preview_prints_metadata_overview(tmp_path, capsys):
    source = tmp_path / "table.h5"
    _write_simple_hdf5(source)

    rc = main(["--log-level", "ERROR", "hdf5", "preview", str(source), "--rows", "1"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Metadata overview" in out
    assert "selected group : /records" in out


def test_hdf5_info_reads_profile_collection_with_gui_settings_group(tmp_path, capsys):
    source = tmp_path / "combined_density.h5"
    write_linak_hdf5_profile_collection(
        source,
        analysis="density",
        profiles=[
            {
                "datasets": {
                    "bin_centers_A": np.asarray([0.5, 1.5], dtype=float),
                    "mass_density_g_per_angstrom": np.asarray([0.1, 0.2], dtype=float),
                },
                "metadata": {"species": "O"},
            },
            {
                "datasets": {
                    "bin_centers_A": np.asarray([0.5, 1.5], dtype=float),
                    "mass_density_g_per_angstrom": np.asarray([0.3, 0.4], dtype=float),
                },
                "metadata": {"species": "H"},
            },
        ],
        metadata={"source": "unit-test"},
    )
    write_plot_profile(
        source,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Saved title"}),
    )

    rc = main(["--log-level", "ERROR", "hdf5", "info", str(source)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Metadata overview" in out
    assert "selected group : /profiles" in out
    assert "profile_index" in out
    assert "Rows: 4" in out


def test_read_hdf5_frame_flattens_profile_collection_rows(tmp_path):
    source = tmp_path / "combined_msd.h5"
    write_linak_hdf5_profile_collection(
        source,
        analysis="msd",
        profiles=[
            {
                "datasets": {
                    "time_ps": np.asarray([0.0, 1.0], dtype=float),
                    "msd_angstrom2": np.asarray([0.0, 0.5], dtype=float),
                },
                "metadata": {"species": "O"},
            },
            {
                "datasets": {
                    "time_ps": np.asarray([0.0, 1.0], dtype=float),
                    "msd_angstrom2": np.asarray([0.0, 0.8], dtype=float),
                },
                "metadata": {"species": "Li"},
            },
        ],
    )

    frame, info = read_hdf5_frame(source)

    assert info.container == "/profiles"
    assert len(frame) == 4
    assert frame["profile_index"].tolist() == [0, 0, 1, 1]
    assert frame["time_ps"].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_csv_plot_writes_output_image(tmp_path):
    source = tmp_path / "table.h5"
    output = tmp_path / "plot.png"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            str(source),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_plot_shorthand_delegates_to_hdf5_plot(tmp_path):
    source = tmp_path / "table.h5"
    output = tmp_path / "plot_alias.png"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_csv_plot_multiple_files_overlay_with_files_option(tmp_path):
    source_a = tmp_path / "table_a.h5"
    source_b = tmp_path / "table_b.h5"
    output = tmp_path / "multi_plot.png"
    _write_simple_hdf5(source_a)
    with h5py.File(source_b, "w") as handle:
        handle.attrs["analysis"] = "table"
        records = handle.create_group("records")
        records.create_dataset("step", data=np.asarray([2, 0, 1], dtype=int))
        records.create_dataset("value", data=np.asarray([6.0, 2.0, 4.0], dtype=float))
        records.create_dataset(
            "label",
            data=np.asarray(["beta", "alpha", "alpha"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_hdf5_combine_density_outputs_plot_ready_hdf5(tmp_path):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    combined = tmp_path / "combined_density.h5"
    output = tmp_path / "combined_density.png"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)

    rc_combine = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
            "--output",
            str(combined),
        ]
    )
    assert rc_combine == 0
    assert combined.exists()
    loaded_profiles = load_density_profiles(combined)
    assert len(loaded_profiles) == 2

    rc_plot = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined),
            "--no-show",
            "--output",
            str(output),
        ]
    )
    assert rc_plot == 0
    assert output.exists()


def test_hdf5_combine_position_outputs_plot_ready_hdf5(tmp_path):
    source_a = tmp_path / "source_a_position.h5"
    source_b = tmp_path / "source_b_position.h5"
    combined = tmp_path / "combined_position.h5"
    output = tmp_path / "combined_position.png"
    _write_position_hdf5(source_a)
    _write_position_hdf5(source_b)

    rc_combine = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "combine",
            "-f",
            str(source_a),
            str(source_b),
            "--output",
            str(combined),
        ]
    )
    assert rc_combine == 0
    assert combined.exists()
    payloads = read_linak_hdf5_profiles(combined, expected_analysis="position")
    assert len(payloads) == 2

    rc_plot = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined),
            "--no-show",
            "--output",
            str(output),
        ]
    )
    assert rc_plot == 0
    assert output.exists()


def test_csv_plot_rejects_multiple_positional_sources_without_files_flag(tmp_path, capsys):
    source_a = tmp_path / "table_a.h5"
    source_b = tmp_path / "table_b.h5"
    _write_simple_hdf5(source_a)
    _write_simple_hdf5(source_b)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            str(source_a),
            str(source_b),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--no-show",
        ]
    )

    assert rc == 1
    assert "Use -f/--files when passing multiple input files." in capsys.readouterr().err


def test_csv_plot_series_labels_count_must_match_rendered_series(tmp_path, capsys):
    source_a = tmp_path / "table_a.h5"
    source_b = tmp_path / "table_b.h5"
    output = tmp_path / "bad_labels.png"
    _write_simple_hdf5(source_a)
    _write_simple_hdf5(source_b)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--labels",
            "only_one_label",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 1
    assert (
        "--labels/--series-labels count must match rendered series count" in capsys.readouterr().err
    )


def test_csv_plot_accepts_custom_axis_and_legend_options(tmp_path):
    source = tmp_path / "table.h5"
    output = tmp_path / "styled_plot.png"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            str(source),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--title",
            "Custom Title",
            "--x-label",
            "Time step",
            "--y-label",
            "Signal",
            "--legend",
            "--labels",
            "run-a",
            "--legend-title",
            "Series",
            "--legend-loc",
            "upper left",
            "--x-min",
            "0",
            "--x-max",
            "2",
            "--y-min",
            "0",
            "--y-max",
            "4",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_csv_plot_shows_preview_before_interactive_column_prompts(tmp_path, monkeypatch, capsys):
    source = tmp_path / "table.h5"
    output = tmp_path / "interactive_plot.png"
    _write_simple_hdf5(source)

    selections = iter([["step"], ["value"]])

    def _fake_prompt_for_columns(*, columns, prompt, allow_multiple):
        return next(selections)

    monkeypatch.setattr("linak.cli._interactive_prompts_available", lambda: True)
    monkeypatch.setattr("linak.cli._prompt_for_columns", _fake_prompt_for_columns)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            str(source),
            "--kind",
            "line",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    out = capsys.readouterr().out
    assert "Preview before interactive plot selection:" in out


def test_csv_get_requires_explicit_column_in_non_interactive_mode(tmp_path, capsys):
    source = tmp_path / "table.h5"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "get",
            str(source),
        ]
    )

    assert rc == 1
    assert "Interactive prompt unavailable in non-interactive mode" in capsys.readouterr().err


def test_rewrite_implicit_csv_interactive():
    assert _rewrite_implicit_csv_interactive(["hdf5", "data.h5"]) == [
        "hdf5",
        "interactive",
        "data.h5",
    ]
    assert _rewrite_implicit_csv_interactive(["hdf5", "info", "data.h5"]) == [
        "hdf5",
        "info",
        "data.h5",
    ]
    assert _rewrite_implicit_csv_interactive(["--log-level", "ERROR", "hdf5", "data.h5"]) == [
        "--log-level",
        "ERROR",
        "hdf5",
        "interactive",
        "data.h5",
    ]
    assert _rewrite_implicit_csv_interactive(["hdf5", "-f", "data.h5"]) == [
        "hdf5",
        "interactive",
        "-f",
        "data.h5",
    ]
    assert _rewrite_implicit_csv_interactive(["hd", "data.h5"]) == [
        "hd",
        "interactive",
        "data.h5",
    ]


def test_rewrite_implicit_plot_csv(tmp_path):
    source = tmp_path / "table.h5"
    _write_simple_hdf5(source)

    assert _rewrite_implicit_plot_csv(["plot", str(source)]) == [
        "hdf5",
        "plot",
        str(source),
    ]
    assert _rewrite_implicit_plot_csv(["plot", "density", "table.h5"]) == [
        "plot",
        "density",
        "table.h5",
    ]
    assert _rewrite_implicit_plot_csv(["plot", "missing_table.h5"]) == [
        "plot",
        "missing_table.h5",
    ]
    assert _rewrite_implicit_plot_csv(["plot", "--help"]) == [
        "plot",
        "--help",
    ]


def test_rewrite_implicit_plot_csv_auto_detects_density_analysis(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    rewritten = _rewrite_implicit_plot_csv(["plot", str(source)])
    assert rewritten == ["plot", str(source)]


def test_rewrite_implicit_plot_csv_auto_detects_position_analysis(tmp_path):
    source = tmp_path / "position.h5"
    _write_position_hdf5(source)

    rewritten = _rewrite_implicit_plot_csv(["plot", str(source)])
    assert rewritten == ["plot", str(source)]


def test_rewrite_implicit_plot_csv_auto_detects_coordination_analysis(tmp_path):
    source = tmp_path / "coordination.h5"
    _write_coordination_hdf5(source)

    rewritten = _rewrite_implicit_plot_csv(["plot", str(source)])
    assert rewritten == ["plot", str(source)]


def test_rewrite_implicit_plot_csv_auto_detects_potential_analysis(tmp_path):
    source = tmp_path / "potential.h5"
    _write_potential_hdf5(source)

    rewritten = _rewrite_implicit_plot_csv(["plot", str(source)])
    assert rewritten == ["plot", str(source)]


def test_hdf5_plot_settings_accept_potential_profile():
    args = build_parser().parse_args(
        ["hdf5", "plot-settings", "dummy.h5", "--profile", "potential"]
    )

    assert args.profile == "potential"


def test_plot_potential_multi_source_renders_successfully(tmp_path):
    source_a = tmp_path / "potential_a.h5"
    source_b = tmp_path / "potential_b.h5"
    _write_potential_hdf5(source_a)
    _write_potential_hdf5(source_b)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--no-gui",
            "--no-show",
        ]
    )

    assert rc == 0


def test_rewrite_implicit_plot_csv_detects_density_with_files_option(tmp_path):
    source_a = tmp_path / "density_a.h5"
    source_b = tmp_path / "density_b.h5"
    _write_density_hdf5(source_a)
    _write_density_hdf5(source_b)

    rewritten = _rewrite_implicit_plot_csv(
        ["plot", "-f", str(source_a), str(source_b), "--no-show"]
    )
    assert rewritten == [
        "plot",
        "-f",
        str(source_a),
        str(source_b),
        "--no-show",
    ]


def test_plot_density_non_gui_does_not_persist_plot_settings(tmp_path):
    source = tmp_path / "density.h5"
    output = tmp_path / "density.png"
    _write_density_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--title",
            "Stored Density Title",
            "--x-min",
            "0",
            "--x-max",
            "3",
            "--no-show",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()

    stored = _read_flat_plot_profile(source, "plot:density")
    assert stored is None


def test_plot_density_toggle_controls_non_gui_do_not_persist(tmp_path):
    source = tmp_path / "density.h5"
    output = tmp_path / "density_toggles.png"
    _write_density_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--title-mode",
            "off",
            "--legend",
            "off",
            "--grid",
            "off",
            "--ticks",
            "off",
            "--markers",
            "on",
            "--no-show",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()

    stored = _read_flat_plot_profile(source, "plot:density")
    assert stored is None


def test_plot_density_keeps_existing_settings_when_changed_non_interactively(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)
    write_plot_profile(
        source,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Original Title"}),
    )

    rc_second = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--title",
            "New Title",
            "--no-show",
            "--output",
            str(tmp_path / "second.png"),
        ]
    )
    assert rc_second == 0

    stored = _read_flat_plot_profile(source, "plot:density")
    assert stored is not None
    assert stored["title"] == "Original Title"


def test_hdf5_plot_settings_can_apply_profile_to_other_files(tmp_path):
    source = tmp_path / "source.h5"
    target = tmp_path / "target.h5"
    _write_simple_hdf5(source)
    _write_simple_hdf5(target)

    rc_set = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot-settings",
            str(source),
            "--profile",
            "table",
            "--set",
            'title="Shared Table Title"',
            "x_lim=[0,2]",
        ]
    )
    assert rc_set == 0

    rc_apply = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot-settings",
            str(source),
            "--profile",
            "table",
            "--export-to",
            str(target),
        ]
    )
    assert rc_apply == 0

    target_profile = _read_flat_plot_profile(target, "plot:table")
    assert target_profile is not None
    assert target_profile["title"] == "Shared Table Title"
    assert target_profile["x_lim"] == pytest.approx([0.0, 2.0])


def test_hdf5_plot_settings_accepts_position_profile_choice():
    args = build_parser().parse_args(
        ["hdf5", "plot-settings", "source.h5", "--profile", "position"]
    )
    assert args.profile == "position"


def test_csv_plot_accepts_one_sided_x_limits(tmp_path):
    source = tmp_path / "table.h5"
    output = tmp_path / "one_sided_xlim.png"
    _write_simple_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot",
            str(source),
            "--kind",
            "line",
            "--x",
            "step",
            "--y",
            "value",
            "--x-min",
            "0.5",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()

    stored = _read_flat_plot_profile(source, "plot:table")
    assert stored is None


def test_plot_help_lists_analysis_and_style_options(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--help"])
    out = capsys.readouterr().out
    assert "--x-mode" in out
    assert "--quantity" in out
    assert "--species-a" in out
    assert "--component" in out
    assert "--map-color" in out
    assert "--time-axis" in out
    assert "--time-section-width" in out
    assert "--font-family" in out
    assert "--title-font-size" in out


def test_plot_density_source_help_is_analysis_specific(tmp_path, capsys):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    rc = main(["--log-level", "ERROR", "plot", str(source), "--help"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Detected analysis from input: density" in out
    assert "--x-mode" in out
    assert "--quantity" in out
    assert "--component" not in out
    assert "--projection-x" not in out


def test_plot_position_source_help_is_analysis_specific(tmp_path, capsys):
    source = tmp_path / "position.h5"
    _write_position_hdf5(source)

    rc = main(["--log-level", "ERROR", "plot", str(source), "--help"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Detected analysis from input: position" in out
    assert "--component" in out
    assert "--projection-x" in out
    assert "--time-axis" in out
    assert "--x-mode" not in out
    assert "--quantity" not in out


def test_plot_help_with_files_uses_uniform_detected_analysis(tmp_path, capsys):
    source_a = tmp_path / "density_a.h5"
    source_b = tmp_path / "density_b.h5"
    _write_density_hdf5(source_a)
    _write_density_hdf5(source_b)

    rc = main(["--log-level", "ERROR", "plot", "-f", str(source_a), str(source_b), "--help"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Detected analysis from input: density" in out
    assert "--x-mode" in out
    assert "--component" not in out


def test_plot_help_with_mixed_sources_falls_back_to_generic_help(tmp_path, capsys):
    density_source = tmp_path / "density.h5"
    position_source = tmp_path / "position.h5"
    _write_density_hdf5(density_source)
    _write_position_hdf5(position_source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(density_source),
            str(position_source),
            "--help",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Detected analysis from input:" not in out
    assert "--x-mode" in out
    assert "--component" in out


def test_plot_help_for_generic_hdf5_falls_back_to_generic_help(tmp_path, capsys):
    source = tmp_path / "generic.h5"
    _write_simple_hdf5(source)

    rc = main(["--log-level", "ERROR", "plot", str(source), "--help"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Detected analysis from input:" not in out
    assert "--x-mode" in out
    assert "--component" in out


def test_plot_help_for_missing_hdf5_falls_back_to_generic_help(tmp_path, capsys):
    source = tmp_path / "missing.h5"

    rc = main(["--log-level", "ERROR", "plot", str(source), "--help"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Detected analysis from input:" not in out
    assert "--x-mode" in out
    assert "--component" in out


@pytest.mark.parametrize("token", ["density", "position"])
def test_plot_analysis_name_tokens_are_not_subcommands(tmp_path, capsys, token):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    rc = main(["--log-level", "ERROR", "plot", token, str(source), "--no-show"])

    assert rc == 1
    assert "Use -f/--files" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["plot", "input.h5", "--dry-run"],
        ["hdf5", "combine", "-f", "a.h5", "b.h5", "--dry-run"],
        ["compute", "density", "traj.xyz", "--dry-run"],
        ["compute", "msd", "traj.xyz", "--dry-run"],
        ["compute", "position", "traj.xyz", "--dry-run"],
        ["compute", "rdf", "traj.xyz", "--dry-run"],
        ["compute", "potential", "run_dir", "--dry-run"],
        ["apply", "pbc", "traj.xyz", "--dry-run"],
        ["apply", "compress", "output.out", "--dry-run"],
    ],
)
def test_all_leaf_commands_accept_dry_run_flag(argv):
    args = build_parser().parse_args(argv)
    assert args.dry_run is True


def test_all_leaf_commands_accept_dry_run_short_flag():
    args = build_parser().parse_args(["compute", "potential", "file.cube", "-n"])
    assert args.dry_run is True


def test_plot_density_defaults_to_surface_distance_mass():
    args = build_parser().parse_args(["plot", "input.h5"])
    assert args.x_mode == "distance"
    assert args.quantity == "mass"


def test_plot_position_accepts_xy_z_component():
    args = build_parser().parse_args(["plot", "input.h5", "--component", "xy-z"])
    assert args.component == "xy-z"
    assert args.map_color == "distance"


def test_plot_position_accepts_public_2d_projection_arguments():
    args = build_parser().parse_args(
        [
            "plot",
            "input.h5",
            "--component",
            "2d-projection",
            "--projection-x",
            "x",
            "--projection-y",
            "distance",
            "--projection-value",
            "y",
            "--projection-render-mode",
            "line-colors",
            "--projection-filter-min",
            "4.0",
            "--projection-filter-max",
            "6.0",
        ]
    )

    assert args.component == "2d-projection"
    assert args.projection_x == "x"
    assert args.projection_y == "distance"
    assert args.projection_value == "y"
    assert args.projection_render_mode == "line-colors"
    assert args.projection_filter_min == pytest.approx(4.0)
    assert args.projection_filter_max == pytest.approx(6.0)


def test_plot_position_accepts_xy_z_map_color_override():
    args = build_parser().parse_args(
        ["plot", "input.h5", "--component", "xy-z", "--map-color", "z"]
    )
    assert args.component == "xy-z"
    assert args.map_color == "z"


def test_plot_density_defaults_to_gui_when_show_is_enabled():
    args = build_parser().parse_args(["plot", "input.h5"])
    cli_mod._resolve_gui_mode(args)
    assert args.gui is True


def test_plot_density_defaults_to_non_gui_when_show_is_disabled():
    args = build_parser().parse_args(["plot", "input.h5", "--no-show"])
    cli_mod._resolve_gui_mode(args)
    assert args.gui is False


def test_plot_density_no_gui_flag_disables_gui_with_show_enabled():
    args = build_parser().parse_args(["plot", "input.h5", "--no-gui"])
    cli_mod._resolve_gui_mode(args)
    assert args.gui is False


def test_compute_density_defaults_surface_detection_options():
    args = build_parser().parse_args(["compute", "density", "traj.xyz"])
    assert args.axis == "z"
    assert args.surface_mode == "auto"
    assert args.surface_elements is None
    assert args.include_fixed_surface_atoms is False
    assert args.rough_surface_envelope is None
    assert args.outputs is None
    assert cli_mod._resolve_density_outputs_from_args(args) == "line"
    assert args.heatmap_planes is None


def test_compute_density_default_axis_produces_all_three_axes(tmp_path, monkeypatch, capsys):
    """Default --axis z produces raw x/y/z profiles and a distance profile."""
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--cell",
            "10",
            "10",
            "10",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    assert (
        "Density bin preparation uses cell bounds; skipped observed coordinate scan."
        in capsys.readouterr().err
    )
    output = _linak_output_dir(tmp_path) / "traj_density_o.h5"
    assert output.exists()
    profiles = load_density_profiles(output)
    raw_profiles = [p for p in profiles if p.coordinate_mode != "distance"]
    distance_profiles = [p for p in profiles if p.coordinate_mode == "distance"]
    assert {p.axis for p in raw_profiles} == {"x", "y", "z"}
    assert len(distance_profiles) >= 1
    assert all(p.axis == "z" for p in distance_profiles)
    assert load_density_heatmap_profiles(output) == []


def test_compute_density_outputs_all_writes_line_profiles_and_heatmaps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--bin-width",
            "0.1",
            "--outputs",
            "all",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density_o.h5"
    line_profiles = load_density_profiles(output)
    heatmap_profiles = load_density_heatmap_profiles(output)
    assert {profile.axis for profile in line_profiles if profile.coordinate_mode != "distance"} == {
        "x",
        "y",
        "z",
    }
    assert {profile.plane for profile in heatmap_profiles} == {"xy", "xz", "yz"}


def test_compute_density_outputs_heatmap_respects_selected_planes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--bin-width",
            "0.1",
            "--outputs",
            "heatmap",
            "--heatmap-planes",
            "xy",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density_o.h5"
    assert load_density_profiles(output) == []
    assert [profile.plane for profile in load_density_heatmap_profiles(output)] == ["xy"]


def test_compute_density_heatmap_planes_imply_heatmap_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--bin-width",
            "0.1",
            "--heatmap-planes",
            "xy",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density_o.h5"
    assert load_density_profiles(output) == []
    assert [profile.plane for profile in load_density_heatmap_profiles(output)] == ["xy"]


def test_compute_density_rejects_heatmap_planes_with_line_output(capsys):
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            "traj.xyz",
            "--outputs",
            "line",
            "--heatmap-planes",
            "xy",
        ]
    )

    assert rc == 1
    assert "--heatmap-planes requires --outputs heatmap or --outputs all" in capsys.readouterr().err


def test_density_profile_filter_options_only_offer_heatmap_when_sources_exist():
    line_only_options = cli_mod._build_density_profile_filter_options(
        [
            (
                "density.h5",
                [
                    {
                        "metadata": {
                            "analysis": "density",
                            "species": "H",
                            "axis": "x",
                            "coordinate_mode": "axis",
                            "profile_kind": "line_1d",
                        }
                    }
                ],
            )
        ],
        axis=None,
        species="H",
    )
    heatmap_options = cli_mod._build_density_profile_filter_options(
        [
            (
                "density.h5",
                [
                    {
                        "metadata": {
                            "analysis": "density",
                            "species": "H",
                            "plane": "xy",
                            "profile_kind": "heatmap_2d",
                        }
                    }
                ],
            )
        ],
        axis=None,
        species="H",
    )

    assert line_only_options["density_view_types"] == ["line_1d"]
    assert heatmap_options["density_view_types"] == ["heatmap_2d"]


def test_compute_density_axis_y_stores_all_axes_with_y_as_surface(tmp_path, monkeypatch):
    """--axis y produces raw x/y/z profiles and a distance profile using y as surface axis."""
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "y",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density_o.h5"
    assert output.exists()
    profiles = load_density_profiles(output)
    raw_profiles = [p for p in profiles if p.coordinate_mode != "distance"]
    distance_profiles = [p for p in profiles if p.coordinate_mode == "distance"]
    assert {p.axis for p in raw_profiles} == {"x", "y", "z"}
    assert len(distance_profiles) >= 1
    assert all(p.axis == "y" for p in distance_profiles)
    _datasets, metadata = read_linak_hdf5(output, expected_analysis="density")
    assert metadata["surface_axis"] == "y"


def test_compute_position_defaults_surface_detection_options():
    args = build_parser().parse_args(["compute", "position", "traj.xyz"])
    assert args.species is None
    assert args.axis == "z"
    assert args.surface_mode == "auto"
    assert args.surface_elements is None
    assert args.include_fixed_surface_atoms is False
    assert args.rough_surface_envelope is None


def test_compute_msd_accepts_uppercase_alias():
    args = build_parser().parse_args(["compute", "MSD", "traj.xyz"])
    assert args.compute_command == "msd"


def test_compute_rdf_accepts_uppercase_alias():
    args = build_parser().parse_args(["compute", "RDF", "traj.xyz"])
    assert args.compute_command == "rdf"


def test_plot_density_dry_run_skips_rendering(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--dry-run",
            "--no-show",
        ]
    )

    assert rc == 0
    assert source.exists()


def test_compute_density_dry_run_skips_trajectory_read_and_csv_write(tmp_path):
    missing_trajectory = tmp_path / "missing_traj.xyz"
    expected_default_output = _linak_output_dir(tmp_path) / "missing_traj_density_o.h5"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(missing_trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert not expected_default_output.exists()


def test_compute_msd_dry_run_resolves_input_metadata_without_reading_trajectory(
    tmp_path, monkeypatch, capsys
):
    missing_trajectory = tmp_path / "missing_traj.xyz"
    input_file = tmp_path / "input.inp"
    _write_cp2k_input(input_file, timestep_fs=0.5, stride_md=5)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("read_trajectory should not be called during dry-run")

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _raise_if_called)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "msd",
            str(missing_trajectory),
            "--species",
            "O",
            "--input",
            str(input_file),
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "cell resolution: resolved 17.887 15.491 59.671 Angstrom" in err
    assert "timestep resolution: 2.5 fs" in err
    assert "explicit --input" in err


def test_compute_density_dry_run_resolves_input_cell_without_reading_trajectory(
    tmp_path, monkeypatch, capsys
):
    missing_trajectory = tmp_path / "missing_traj.xyz"
    input_file = tmp_path / "input.inp"
    _write_cp2k_input(input_file, timestep_fs=0.5, stride_md=5)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("read_trajectory should not be called during dry-run")

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _raise_if_called)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "density",
            str(missing_trajectory),
            "--species",
            "K",
            "--input",
            str(input_file),
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "cell resolution: resolved 17.887 15.491 59.671 Angstrom" in err
    assert "explicit --input" in err


def test_compute_rdf_dry_run_resolves_input_cell_without_reading_trajectory(
    tmp_path, monkeypatch, capsys
):
    missing_trajectory = tmp_path / "missing_traj.xyz"
    input_file = tmp_path / "input.inp"
    _write_cp2k_input(input_file, timestep_fs=0.5, stride_md=5)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("read_trajectory should not be called during dry-run")

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _raise_if_called)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "rdf",
            str(missing_trajectory),
            "--species-a",
            "K",
            "--species-b",
            "K",
            "--input",
            str(input_file),
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "cell resolution: resolved 17.887 15.491 59.671 Angstrom" in err
    assert "r_max resolution: 7.7 (auto rounded down from 7.7455 to match bin_width=0.05)" in err


def test_compute_rdf_dry_run_uses_trajectory_hdf5_cell_without_adjacent_input_lookup(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.traj.h5"
    write_trajectory(
        [Atoms("OO", positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])],
        trajectory,
        metadata=TrajectoryStoredMetadata(
            input_path=tmp_path / "input.inp",
            input_format="inp",
            cell_angstrom=(10.0, 11.0, 12.0),
            cell_source="simulation input",
            frame_timestep_fs=2.5,
            md_timestep_fs=0.5,
            trajectory_stride_md=5,
            timestep_source="simulation input",
        ),
    )

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("_auto_detect_cell should not be called for trajectory HDF5 dry-run")

    monkeypatch.setattr("linak.resolution._auto_detect_cell", _raise_if_called)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "rdf",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "O",
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "trajectory HDF5 metadata" in err
    assert "auto-detected" not in err


def test_compute_msd_dry_run_uses_trajectory_hdf5_timestep_without_adjacent_input_lookup(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.traj.h5"
    frames = [
        Atoms("O", positions=[[0.0, 0.0, 0.1]]),
        Atoms("O", positions=[[0.0, 0.0, 0.2]]),
    ]
    write_trajectory(
        frames,
        trajectory,
        metadata=TrajectoryStoredMetadata(
            input_path=tmp_path / "input.inp",
            input_format="inp",
            cell_angstrom=(10.0, 11.0, 12.0),
            cell_source="simulation input",
            frame_timestep_fs=2.5,
            md_timestep_fs=0.5,
            trajectory_stride_md=5,
            timestep_source="simulation input",
        ),
    )

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError(
            "_auto_detect_frame_timestep_fs should not be called for trajectory HDF5 dry-run"
        )

    monkeypatch.setattr("linak.resolution._auto_detect_frame_timestep_fs", _raise_if_called)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "msd",
            str(trajectory),
            "--species",
            "O",
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "trajectory HDF5 metadata" in err
    assert "auto-detected" not in err


def test_compute_density_uses_trajectory_hdf5_cell_without_adjacent_input_lookup(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.traj.h5"
    write_trajectory(
        [Atoms("O", positions=[[0.0, 0.0, 0.1]])],
        trajectory,
        metadata=TrajectoryStoredMetadata(
            input_path=tmp_path / "input.inp",
            input_format="inp",
            cell_angstrom=(10.0, 11.0, 12.0),
            cell_source="simulation input",
        ),
    )

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("_auto_detect_cell should not be called for trajectory HDF5 compute")

    monkeypatch.setattr("linak.resolution._auto_detect_cell", _raise_if_called)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
        ]
    )

    assert rc == 0
    _datasets, metadata = read_linak_hdf5(
        _linak_output_dir(tmp_path) / "traj.traj_density_o.h5",
        expected_analysis="density",
    )
    assert metadata["cell_source"] == "trajectory HDF5 metadata"


def test_compute_potential_dry_run_validates_missing_hartree_file(tmp_path):
    missing_cube = tmp_path / "missing-v_hartree-1_0.cube"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(missing_cube),
            "--dry-run",
        ]
    )

    assert rc == 1


def test_apply_pbc_dry_run_skips_trajectory_read_and_write(tmp_path):
    missing_trajectory = tmp_path / "missing.xyz"
    default_output = tmp_path / "missing_pbc.xyz"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "pbc",
            str(missing_trajectory),
            "--dry-run",
            "--cell",
            "10.0",
            "10.0",
            "10.0",
        ]
    )

    assert rc == 0
    assert not default_output.exists()


def test_apply_compress_help_describes_outputs(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["apply", "compress", "--help"])
    out = capsys.readouterr().out
    assert "What `linak apply compress` creates" in out
    assert "manifest.json" in out
    assert ".meta.json" in out
    assert "--backup-dir" in out
    assert "--drop" in out
    assert ".linak_backups" in out
    assert "/scratch-shared" not in out


def test_apply_compress_dry_run_skips_file_operations(tmp_path):
    missing_output = tmp_path / "missing.out"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "compress",
            str(missing_output),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert not (tmp_path / "missing").exists()
    assert not (tmp_path / ".linak_backups").exists()


def test_apply_compress_moves_original_and_writes_outputs(tmp_path):
    source = tmp_path / "output.out"
    _write_minimal_cp2k_output(source)
    backup_dir = tmp_path / "private_backups"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "compress",
            str(source),
            "--backup-dir",
            str(backup_dir),
        ]
    )

    assert rc == 0
    output_dir = tmp_path / "output"
    assert output_dir.exists()
    assert (output_dir / "README.txt").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "backup_info.txt").exists()
    assert not source.exists()
    assert len(list(backup_dir.glob("compress_output__*.out"))) == 1
    assert len(list(backup_dir.glob("compress_output__*.out.meta.json"))) == 1


def test_plot_density_csv_does_not_write_csv_unless_requested(tmp_path, monkeypatch):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_csv = tmp_path / "source_density.h5"
    save_density_profile(profile, source_csv)
    output_plot = tmp_path / "density.png"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_csv),
            "--no-show",
            "--output",
            str(output_plot),
        ]
    )

    assert rc == 0
    assert output_plot.exists()
    assert list(tmp_path.glob("*.h5")) == [source_csv]


def test_plot_density_multiple_files_overlays_with_source_labels(tmp_path, monkeypatch):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_csv_1 = tmp_path / "source1_density.h5"
    source_csv_2 = tmp_path / "source2_density.h5"
    save_density_profile(profile, source_csv_1)
    save_density_profile(profile, source_csv_2)

    captured_labels: list[str] = []

    def _fake_plot_density_profiles(profiles, **_kwargs):
        captured_labels.extend([item.species for item in profiles])
        return None

    monkeypatch.setattr("linak.analysis.density.plot_density_profiles", _fake_plot_density_profiles)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_csv_1),
            str(source_csv_2),
            "--species",
            "O",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured_labels == [f"{source_csv_1.name}:O", f"{source_csv_2.name}:O"]


def test_plot_implicit_multi_density_with_files_option(tmp_path):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    output = tmp_path / "implicit_multi_density.png"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_plot_density_multi_ignores_saved_settings_source_and_starts_from_defaults(
    tmp_path, monkeypatch, caplog
):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)
    write_plot_profile(
        source_b,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "From second file"}),
    )

    captured: dict[str, object] = {}

    def _fake_render_profile_plot(**kwargs):
        captured["title"] = kwargs["args"].title
        return None, {}

    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)
    caplog.set_level("INFO")

    rc = main(
        [
            "--log-level",
            "INFO",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--settings-source",
            "2",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured["title"] is None
    assert "plot-settings source" not in caplog.text


def test_build_gui_series_descriptors_include_directory_metadata():
    descriptors = _build_gui_series_descriptors(
        sources=["/tmp/runs/run_04/density.h5"],
        fallback_labels_by_source=[["Au", "H2O"]],
    )

    assert [item["series_id"] for item in descriptors] == ["series:0:0", "series:0:1"]
    assert descriptors[0]["default_label"] == "Au"
    assert descriptors[1]["default_label"] == "H2O"
    assert descriptors[0]["source_name"] == "density.h5"
    assert descriptors[0]["source_directory"].endswith("run_04")
    assert descriptors[0]["source_kind"] == "source"
    assert descriptors[0]["source_series_id"] == "series:0:0"
    assert descriptors[0]["is_generated"] is False


def test_merge_gui_series_descriptors_preserves_generated_layers_only_when_sources_exist():
    current = [
        {
            "series_id": "source:a",
            "source_kind": "source",
            "source_series_id": "source:a",
            "is_generated": False,
            "default_label": "A",
        }
    ]
    saved = [
        *current,
        {
            "series_id": "copy:a",
            "source_kind": "source",
            "source_series_id": "source:a",
            "is_generated": True,
            "default_label": "A Copy",
        },
        {
            "series_id": "copy:missing",
            "source_kind": "source",
            "source_series_id": "missing",
            "is_generated": True,
            "default_label": "Missing Copy",
        },
        {
            "series_id": "group:a",
            "source_kind": "group",
            "is_generated": True,
            "member_series_ids": ["source:a", "copy:a", "missing"],
            "default_label": "Group",
        },
    ]

    merged = _merge_gui_series_descriptors(current, saved)

    assert [item["series_id"] for item in merged] == ["source:a", "copy:a", "group:a"]
    assert merged[1]["source_series_id"] == "source:a"
    assert merged[1]["is_generated"] is True
    assert merged[2]["member_series_ids"] == ["source:a", "copy:a"]


def test_gui_render_requires_source_data_for_enabled_copies_and_groups():
    settings = {
        "series_descriptors": [
            {"series_id": "source:a", "source_kind": "source", "source_series_id": "source:a"},
            {
                "series_id": "copy:a",
                "source_kind": "source",
                "source_series_id": "source:a",
                "is_generated": True,
            },
            {
                "series_id": "group:a",
                "source_kind": "group",
                "member_series_ids": ["copy:a"],
                "is_generated": True,
            },
        ],
        "series_overrides": {"source:a": {"enabled": False}},
    }
    args = build_parser().parse_args(["plot", "dummy.h5"])

    required = _required_source_ids_for_gui_render(settings)
    _force_source_ids_enabled_for_gui_loading(args, required)

    assert _gui_series_descriptors_from_settings(settings, [])[1]["series_id"] == "copy:a"
    assert required == {"source:a"}
    assert args.series_overrides["source:a"]["enabled"] is True


def test_gui_render_does_not_require_hidden_group_members():
    settings = {
        "series_descriptors": [
            {"series_id": "source:a", "source_kind": "source", "source_series_id": "source:a"},
            {"series_id": "source:b", "source_kind": "source", "source_series_id": "source:b"},
            {
                "series_id": "group:a",
                "source_kind": "group",
                "member_series_ids": ["source:a", "source:b"],
                "is_generated": True,
            },
        ],
        "series_overrides": {
            "source:a": {"enabled": False},
        },
    }

    required = _required_source_ids_for_gui_render(settings)

    assert required == {"source:b"}


def test_default_series_family_colors_preserve_hidden_source_family_slots():
    descriptors = [
        {"series_id": "source:a", "source_kind": "source", "source_series_id": "source:a"},
        {"series_id": "source:b", "source_kind": "source", "source_series_id": "source:b"},
        {"series_id": "source:c", "source_kind": "source", "source_series_id": "source:c"},
    ]
    active_descriptors = [descriptors[1], descriptors[2]]

    colors = cli_mod._default_series_family_colors(
        descriptors,
        2,
        target_descriptors=active_descriptors,
    )

    assert colors == cli_mod._default_multi_series_colors(3)[1:3]


def test_apply_gui_settings_to_args_forwards_annotations_without_declared_cli_attr():
    args = argparse.Namespace(title="Example")
    settings = {
        "annotations": [
            {
                "type": "text",
                "coord_system": "axes",
                "x": 0.5,
                "y": 0.9,
                "text": "Label",
            }
        ]
    }

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert getattr(args, "annotations", None) == settings["annotations"]


def test_apply_gui_settings_to_args_forwards_legend_kwargs_without_declared_cli_attr():
    args = argparse.Namespace(title="Example")
    settings = {"legend_kwargs": {"frameon": False, "ncols": 2}}

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert getattr(args, "legend_kwargs", None) == settings["legend_kwargs"]


def test_apply_gui_settings_to_args_forwards_heatmap_and_padding_without_declared_cli_attr():
    args = argparse.Namespace(title="Example")
    settings = {
        "x_label_pad": 7.5,
        "heatmap_vmin": 0.1,
        "heatmap_vmax": 2.0,
        "heatmap_cmap": "viridis",
        "heatmap_colorbar_enabled": False,
    }

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert getattr(args, "x_label_pad", None) == 7.5
    assert getattr(args, "heatmap_vmin", None) == 0.1
    assert getattr(args, "heatmap_vmax", None) == 2.0
    assert getattr(args, "heatmap_cmap", None) == "viridis"
    assert getattr(args, "heatmap_colorbar_enabled", None) is False


def test_apply_gui_settings_to_args_preserves_gui_manual_axis_limits():
    args = argparse.Namespace(
        x_min=1.0,
        x_max=2.0,
        y_min=3.0,
        y_max=4.0,
        x_lim=[1.0, 2.0],
        y_lim=[3.0, 4.0],
    )
    settings = {"x_lim": [5.0, 6.0], "y_lim": [7.0, 8.0]}

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert args.x_min is None
    assert args.x_max is None
    assert args.y_min is None
    assert args.y_max is None
    assert args.x_lim == [5.0, 6.0]
    assert args.y_lim == [7.0, 8.0]


def test_apply_gui_settings_to_args_clears_stale_saved_axis_limits_for_auto_mode():
    args = argparse.Namespace(
        x_min=1.0,
        x_max=2.0,
        y_min=3.0,
        y_max=4.0,
        x_lim=[1.0, 2.0],
        y_lim=[3.0, 4.0],
    )
    settings = {
        "title": "Auto axes",
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
    }

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert args.x_min is None
    assert args.x_max is None
    assert args.y_min is None
    assert args.y_max is None
    assert args.x_lim is None
    assert args.y_lim is None


def test_collect_plot_settings_for_persistence_drops_stale_auto_axis_limits():
    args = argparse.Namespace(
        x_min=1.0,
        x_max=2.0,
        y_min=3.0,
        y_max=4.0,
        x_lim=[1.0, 2.0],
        y_lim=[3.0, 4.0],
    )

    cli_mod._apply_gui_settings_to_args(
        args,
        {
            "title": "Auto axes",
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        },
    )
    persisted = cli_mod._collect_plot_settings_for_persistence(
        args,
        keys=("x_lim", "y_lim"),
    )

    assert persisted["x_lim"] is None
    assert persisted["y_lim"] is None


def test_collect_plot_settings_for_persistence_materializes_density_view_mapping():
    args = argparse.Namespace(
        species="H2O",
        axis="y",
        x_mode="axis",
        quantity="number",
        x_lim=None,
        y_lim=None,
    )

    persisted = cli_mod._collect_plot_settings_for_persistence(
        args,
        keys=cli_mod._PLOT_SETTINGS_DENSITY_KEYS,
    )

    assert persisted["species"] == "H2O"
    assert persisted["axis"] == "y"
    assert "x_mode" not in persisted
    assert "quantity" not in persisted
    assert persisted["view_mapping"]["x"] == "axis_coordinate"
    assert persisted["view_mapping"]["y"] == "number_density"
    assert persisted["view_mapping"]["fixed_values"]["x_mode"] == "axis"


def test_apply_saved_plot_settings_restores_view_mapping_without_argparse_field(monkeypatch):
    args = argparse.Namespace(_runtime_argv=())
    saved = {
        "source_selection": {},
        "view_mapping": {
            "view_type_id": "line_1d",
            "x": "axis_coordinate",
            "y": "number_density",
            "color": None,
            "split_by": None,
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"x_mode": "axis", "quantity": "number"},
        },
        "style": {},
    }
    monkeypatch.setattr(
        "linak.plot.plot_settings.read_plot_profile",
        lambda *args, **kwargs: saved,
    )

    cli_mod._apply_saved_plot_settings(
        args=args,
        source_path=Path("dummy.h5"),
        profile_key="plot:density",
        keys=cli_mod._PLOT_SETTINGS_DENSITY_KEYS,
        profile_name=None,
    )

    assert isinstance(args.view_mapping, dict)
    assert args.view_mapping["x"] == "axis_coordinate"
    assert args.view_mapping["fixed_values"]["quantity"] == "number"


def test_apply_saved_plot_settings_does_not_override_explicit_density_mapping_flags(monkeypatch):
    args = argparse.Namespace(_runtime_argv=("--x-mode", "y"))
    saved = {
        "view_mapping": {
            "view_type_id": "line_1d",
            "x": "axis_coordinate",
            "y": "number_density",
            "color": None,
            "split_by": None,
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"x_mode": "axis", "quantity": "number"},
        }
    }
    monkeypatch.setattr(cli_mod, "_read_flat_plot_profile", lambda *args, **kwargs: saved)

    cli_mod._apply_saved_plot_settings(
        args=args,
        source_path=Path("dummy.h5"),
        profile_key="plot:density",
        keys=cli_mod._PLOT_SETTINGS_DENSITY_KEYS,
        profile_name=None,
    )

    assert not hasattr(args, "view_mapping")


def test_apply_saved_plot_settings_reads_mapping_native_density_payload_without_flattening(
    monkeypatch,
):
    args = argparse.Namespace(_runtime_argv=())
    saved_payload = {
        "source_selection": {"species": "H2O", "axis": "y"},
        "view_mapping": {
            "view_type_id": "line_1d",
            "x": "axis_coordinate",
            "y": "number_density",
            "color": None,
            "split_by": None,
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"x_mode": "axis", "quantity": "number"},
        },
        "style": {"title": "Saved density"},
    }

    monkeypatch.setattr(
        "linak.plot.plot_settings.read_plot_profile",
        lambda *args, **kwargs: saved_payload,
    )
    monkeypatch.setattr(
        "linak.plot.profile_persistence.select_plot_profile_settings",
        lambda profile_key, payload, *, keys: {
            "species": payload["source_selection"]["species"],
            "axis": payload["source_selection"]["axis"],
            "view_mapping": payload["view_mapping"],
            "title": payload["style"]["title"],
        },
    )

    saved = cli_mod._apply_saved_plot_settings(
        args=args,
        source_path=Path("dummy.h5"),
        profile_key="plot:density",
        keys=cli_mod._PLOT_SETTINGS_DENSITY_KEYS,
        profile_name=None,
    )

    assert saved == {
        "species": "H2O",
        "axis": "y",
        "view_mapping": saved_payload["view_mapping"],
        "title": "Saved density",
    }
    assert args.species == "H2O"
    assert args.axis == "y"
    assert args.title == "Saved density"
    assert args.view_mapping["fixed_values"]["quantity"] == "number"


def test_apply_saved_plot_settings_still_uses_flatten_for_position_legacy_restore(monkeypatch):
    args = argparse.Namespace(_runtime_argv=())
    saved_payload = {
        "source_selection": {"species": "O", "axis": None},
        "view_mapping": {
            "view_type_id": "trajectory_2d",
            "x": "x",
            "y": "z",
            "color": "distance_to_surface",
            "split_by": "atom",
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"projection_render_mode": "line-colors"},
        },
        "style": {},
    }
    select_calls: list[str] = []

    monkeypatch.setattr(
        "linak.plot.plot_settings.read_plot_profile",
        lambda *args, **kwargs: saved_payload,
    )
    monkeypatch.setattr(
        "linak.plot.profile_persistence.select_plot_profile_settings",
        lambda profile_key, payload, *, keys: (
            select_calls.append("called")
            or {
                "species": "O",
                "axis": None,
                "view_mapping": saved_payload["view_mapping"],
                "component": "2d-projection",
                "projection_render_mode": "line-colors",
            }
        ),
    )

    saved = cli_mod._apply_saved_plot_settings(
        args=args,
        source_path=Path("dummy.h5"),
        profile_key="plot:position",
        keys=cli_mod._PLOT_SETTINGS_POSITION_KEYS,
        profile_name=None,
    )

    assert select_calls == ["called"]
    assert saved["component"] == "2d-projection"
    assert args.component == "2d-projection"
    assert args.projection_render_mode == "line-colors"


def test_read_plot_profile_for_apply_reads_density_payload_without_flattening(monkeypatch):
    saved_payload = {
        "source_selection": {"species": "H2O", "axis": "y"},
        "view_mapping": {
            "view_type_id": "line_1d",
            "x": "axis_coordinate",
            "y": "number_density",
            "color": None,
            "split_by": None,
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"x_mode": "axis", "quantity": "number"},
        },
        "style": {"title": "Saved density"},
    }
    monkeypatch.setattr(
        "linak.plot.plot_settings.read_plot_profile",
        lambda *args, **kwargs: saved_payload,
    )
    monkeypatch.setattr(
        "linak.plot.profile_persistence.select_plot_profile_settings",
        lambda profile_key, payload, *, keys: {
            "species": payload["source_selection"]["species"],
            "axis": payload["source_selection"]["axis"],
            "view_mapping": payload["view_mapping"],
            "title": payload["style"]["title"],
        },
    )

    loaded = cli_mod._read_plot_profile_for_apply(
        Path("dummy.h5"),
        profile_key="plot:density",
        keys=cli_mod._PLOT_SETTINGS_DENSITY_KEYS,
        profile_name=None,
    )

    assert loaded == {
        "species": "H2O",
        "axis": "y",
        "view_mapping": saved_payload["view_mapping"],
        "title": "Saved density",
    }


def test_read_plot_profile_for_apply_still_flattens_position_payload(monkeypatch):
    saved_payload = {
        "source_selection": {"species": "O", "axis": None},
        "view_mapping": {
            "view_type_id": "trajectory_2d",
            "x": "x",
            "y": "z",
            "color": "distance_to_surface",
            "split_by": "atom",
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"projection_render_mode": "line-colors"},
        },
        "style": {},
    }
    monkeypatch.setattr(
        "linak.plot.plot_settings.read_plot_profile",
        lambda *args, **kwargs: saved_payload,
    )
    monkeypatch.setattr(
        "linak.plot.profile_persistence.select_plot_profile_settings",
        lambda profile_key, payload, *, keys: {
            "species": "O",
            "axis": None,
            "view_mapping": saved_payload["view_mapping"],
            "component": "2d-projection",
        },
    )

    loaded = cli_mod._read_plot_profile_for_apply(
        Path("dummy.h5"),
        profile_key="plot:position",
        keys=cli_mod._PLOT_SETTINGS_POSITION_KEYS,
        profile_name=None,
    )

    assert loaded["component"] == "2d-projection"


def test_cli_no_longer_carries_dead_flatten_saved_profile_wrapper():
    source = Path("src/linak/cli.py").read_text(encoding="utf-8")

    assert "def _flatten_saved_plot_profile_payload(" not in source


def test_apply_gui_settings_to_args_auto_axis_fields_clear_stale_limits_with_gui_shape():
    args = argparse.Namespace(
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        x_lim=[10.0, 20.0],
        y_lim=[30.0, 40.0],
    )

    cli_mod._apply_gui_settings_to_args(
        args,
        {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "_gui_sync_modes": {"x_lim": "auto", "y_lim": "auto"},
        },
    )

    assert args.x_lim is None
    assert args.y_lim is None


def test_apply_gui_settings_to_args_forwards_position_xy_z_distance_max():
    args = argparse.Namespace(title="Example")
    settings = {"xy_z_distance_max": 2.5}

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert getattr(args, "xy_z_distance_max", None) == 2.5


def test_apply_gui_settings_to_args_forwards_position_projection_settings():
    args = argparse.Namespace(title="Example")
    settings = {
        "projection_x": "x",
        "projection_y": "distance",
        "projection_value": "y",
        "projection_render_mode": "line-colors",
        "projection_filter_min": 4.0,
        "projection_filter_max": 6.0,
    }

    cli_mod._apply_gui_settings_to_args(args, settings)

    assert getattr(args, "projection_x", None) == "x"
    assert getattr(args, "projection_y", None) == "distance"
    assert getattr(args, "projection_value", None) == "y"
    assert getattr(args, "projection_render_mode", None) == "line-colors"
    assert getattr(args, "projection_filter_min", None) == pytest.approx(4.0)
    assert getattr(args, "projection_filter_max", None) == pytest.approx(6.0)


def test_resolve_position_plotter_kwargs_uses_generic_view_mapping():
    args = argparse.Namespace(
        component="2d-projection",
        map_color="distance",
        projection_x="x",
        projection_y="z",
        projection_value="distance",
        projection_render_mode="line-colors",
        projection_filter_min=None,
        projection_filter_max=3.0,
        xy_z_distance_max=None,
        time_axis="ps",
    )

    kwargs = cli_mod._resolve_position_plotter_kwargs(args)

    assert "component" not in kwargs
    assert "time_axis" not in kwargs
    assert "view_mapping" in kwargs
    assert kwargs["view_mapping"].view_type_id == "trajectory_2d"
    assert kwargs["view_mapping"].x == "x"
    assert kwargs["view_mapping"].y == "z"


def test_resolve_coordination_plotter_kwargs_uses_generic_view_mapping():
    args = argparse.Namespace(
        component="time-distance",
        time_axis="fs",
    )

    kwargs = _resolve_coordination_plotter_kwargs(args)

    assert "component" not in kwargs
    assert "time_axis" not in kwargs
    assert "view_mapping" in kwargs
    assert kwargs["view_mapping"].view_type_id == "trajectory_2d"
    assert kwargs["view_mapping"].x == "time_fs"
    assert kwargs["view_mapping"].y == "distance_to_surface"
    assert kwargs["view_mapping"].color == "coordination_number"


def test_resolve_density_plotter_kwargs_uses_generic_view_mapping():
    args = argparse.Namespace(
        x_mode="z",
        quantity="number",
    )

    kwargs = _resolve_density_plotter_kwargs(args)

    assert "x_mode" not in kwargs
    assert "quantity" not in kwargs
    assert kwargs["view_mapping"].view_type_id == "line_1d"
    assert kwargs["view_mapping"].x == "axis_coordinate"
    assert kwargs["view_mapping"].y == "number_density"


def test_resolve_msd_plotter_kwargs_uses_generic_view_mapping():
    args = argparse.Namespace(time_axis="fs")

    kwargs = _resolve_msd_plotter_kwargs(args)

    assert "time_axis" not in kwargs
    assert kwargs["view_mapping"].view_type_id == "line_1d"
    assert kwargs["view_mapping"].x == "time_fs"
    assert kwargs["view_mapping"].y == "msd"


def test_resolve_rdf_plotter_kwargs_uses_generic_view_mapping():
    kwargs = _resolve_rdf_plotter_kwargs(argparse.Namespace())

    assert kwargs["view_mapping"].view_type_id == "line_1d"
    assert kwargs["view_mapping"].x == "radius"
    assert kwargs["view_mapping"].y == "g_r"


def test_resolve_potential_plotter_kwargs_uses_generic_summary_mapping():
    args = argparse.Namespace(y_quantity=None, table_view=False)

    kwargs = _resolve_potential_plotter_kwargs(args)

    assert kwargs["view_mapping"].view_type_id == "line_1d"
    assert kwargs["view_mapping"].x == "record_id"
    assert kwargs["view_mapping"].fixed_values["standard_plot"] == "summary"


def test_resolve_orientation_plotter_kwargs_uses_generic_heatmap_mapping():
    args = argparse.Namespace(component="heatmap", angle="azimuthal")

    kwargs = _resolve_orientation_plotter_kwargs(args)

    assert "component" not in kwargs
    assert "angle" not in kwargs
    assert kwargs["view_mapping"].view_type_id == "heatmap_2d"
    assert kwargs["view_mapping"].x == "bin_centers_A"
    assert kwargs["view_mapping"].resolved_role_assignments()["z"] == "heatmap_azimuthal"


def test_position_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(
        component="2d-projection",
        map_color="distance",
        projection_x="x",
        projection_y="z",
        projection_value="distance",
        projection_render_mode="line-colors",
        projection_filter_min=None,
        projection_filter_max=3.0,
        xy_z_distance_max=None,
        time_axis="ps",
        view_mapping=None,
    )

    summary = cli_mod._position_mapping_summary_for_dry_run(args)

    assert "view_mapping=trajectory_2d" in summary
    assert "x=x" in summary
    assert "y=z" in summary
    assert "value=distance_to_surface" in summary
    assert "render_mode=line-colors" in summary
    assert "filter=distance_to_surface[, 3.0]" in summary


def test_density_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(x_mode="z", quantity="number", view_mapping=None)

    summary = cli_mod._density_mapping_summary_for_dry_run(args)

    assert "view_mapping=line_1d" in summary
    assert "x=axis_coordinate" in summary
    assert "y=number_density" in summary
    assert "x_mode=z" in summary


def test_msd_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(time_axis="fs", view_mapping=None)

    summary = cli_mod._msd_mapping_summary_for_dry_run(args)

    assert "view_mapping=line_1d" in summary
    assert "x=time_fs" in summary
    assert "y=msd" in summary


def test_rdf_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(view_mapping=None)

    summary = cli_mod._rdf_mapping_summary_for_dry_run(args)

    assert "view_mapping=line_1d" in summary
    assert "x=radius" in summary
    assert "y=g_r" in summary


def test_coordination_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(component="time-distance", time_axis="fs", view_mapping=None)

    summary = cli_mod._coordination_mapping_summary_for_dry_run(args)

    assert "view_mapping=trajectory_2d" in summary
    assert "x=time_fs" in summary
    assert "y=distance_to_surface" in summary
    assert "color=coordination_number" in summary


def test_orientation_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(component="heatmap", angle="azimuthal", view_mapping=None)

    summary = cli_mod._orientation_mapping_summary_for_dry_run(args)

    assert "view_mapping=heatmap_2d" in summary
    assert "x=bin_centers_A" in summary
    assert "y=heatmap_angle_bin_centers" in summary
    assert "z=heatmap_azimuthal" in summary


def test_potential_mapping_summary_for_dry_run_uses_view_mapping_terms():
    args = argparse.Namespace(y_quantity=None, table_view=False, view_mapping=None)

    summary = cli_mod._potential_mapping_summary_for_dry_run(args)

    assert "view_mapping=line_1d" in summary
    assert "x=record_id" in summary
    assert "standard_plot=summary" in summary


def test_merge_preview_defaults_into_gui_settings_preserves_manual_synced_fields():
    settings = {
        "x_lim": [10.0, 20.0],
        "x_label_pad": 6.0,
        "_gui_sync_modes": {"x_lim": "manual", "x_label_pad": "manual"},
    }
    preview_state = {
        "x_lim": [0.0, 2.0],
        "x_label_pad": 14.0,
        "y_lim": [1.0, 3.0],
    }

    merged = cli_mod._merge_preview_defaults_into_gui_settings(settings, preview_state)

    assert merged["x_lim"] == [10.0, 20.0]
    assert merged["x_label_pad"] == 6.0
    assert merged["y_lim"] == [1.0, 3.0]


def test_merge_preview_defaults_into_gui_settings_does_not_touch_series_overrides():
    settings = {
        "series_overrides": {
            "series:0": {
                "normalization_mode": "max",
                "normalization_value": 1.0,
            }
        }
    }
    preview_state = {
        "series_overrides": {
            "series:0": {
                "normalization_mode": "none",
            }
        },
        "x_lim": [0.0, 2.0],
    }

    merged = cli_mod._merge_preview_defaults_into_gui_settings(settings, preview_state)

    assert merged["series_overrides"]["series:0"]["normalization_mode"] == "max"
    assert merged["series_overrides"]["series:0"]["normalization_value"] == 1.0
    assert merged["x_lim"] == [0.0, 2.0]


def test_materialize_gui_series_overrides_promotes_legacy_normalization_lists():
    settings = {
        "series_descriptors": [
            {"series_id": "series:0", "default_label": "A"},
            {"series_id": "series:1", "default_label": "B"},
        ],
        "series_overrides": {
            "series:1": {
                "normalization_mode": "factor",
                "normalization_value": 2.0,
            }
        },
        "series_normalization_modes": ["max", None],
        "series_normalization_values": [1.0, None],
        "series_normalization_x_refs": [None, None],
    }

    materialized = cli_mod._materialize_gui_series_overrides(settings)

    assert "series_normalization_modes" not in materialized
    assert "series_normalization_values" not in materialized
    assert "series_normalization_x_refs" not in materialized
    assert materialized["series_overrides"]["series:0"]["normalization_mode"] == "max"
    assert materialized["series_overrides"]["series:0"]["normalization_value"] == 1.0
    assert materialized["series_overrides"]["series:1"]["normalization_mode"] == "factor"
    assert materialized["series_overrides"]["series:1"]["normalization_value"] == 2.0


def test_build_gui_series_descriptors_use_origin_paths_for_metadata_and_grouping():
    descriptors = _build_gui_series_descriptors(
        sources=["/tmp/combined_density.h5"],
        fallback_labels_by_source=[["Au", "K", "O"]],
        series_id_segments_by_source=[["a", "b", "c"]],
        origin_path_segments_by_source=[
            [
                "/tmp/runs/run_01/density.h5",
                "/tmp/runs/run_01/density.h5",
                "/tmp/runs/run_02/density.h5",
            ]
        ],
    )

    assert [item["source_name"] for item in descriptors] == [
        "density.h5",
        "density.h5",
        "density.h5",
    ]
    assert descriptors[0]["source_directory"].endswith("run_01")
    assert descriptors[1]["source_index"] == descriptors[0]["source_index"]
    assert descriptors[2]["source_index"] != descriptors[0]["source_index"]
    assert descriptors[2]["source_directory"].endswith("run_02")


def test_build_rdf_profile_filter_options_uses_common_pairs_across_sources():
    options = _build_rdf_profile_filter_options(
        [
            (
                "a.h5",
                [
                    {"metadata": {"species_a": "O", "species_b": "H"}},
                    {"metadata": {"species_a": "H", "species_b": "H"}},
                ],
            ),
            (
                "b.h5",
                [
                    {"metadata": {"species_a": "H", "species_b": "O"}},
                    {"metadata": {"species_a": "O", "species_b": "O"}},
                ],
            ),
        ]
    )

    assert options["species_a"] == ["O", "H"]
    assert options["species_b_by_species_a"][""] == ["H", "O"]
    assert options["species_b_by_species_a"]["O"] == ["H"]
    assert options["species_b_by_species_a"]["H"] == ["O"]


def test_load_rdf_plot_profiles_supports_reversed_cross_pair_selection(tmp_path):
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        bin_centers=np.array([0.5, 1.5], dtype=float),
        g_r=np.array([0.1, 0.2], dtype=float),
        n_frames=2,
    )
    source = tmp_path / "rdf.h5"
    save_rdf_profile(profile, source)

    profiles, fallback_labels, _series_ids, _origins = _load_rdf_plot_profiles(
        sources=[str(source)],
        species_a="H",
        species_b="O",
    )

    assert len(profiles) == 1
    assert profiles[0].species_a == "H"
    assert profiles[0].species_b == "O"
    assert fallback_labels == [["H-O"]]


def test_build_rdf_gui_context_loads_all_pairs_as_layers_even_with_species_args(tmp_path):
    source = tmp_path / "rdf_collection.h5"
    write_linak_hdf5_profile_collection(
        source,
        analysis="rdf",
        profiles=[
            {
                "datasets": {
                    "bin_centers_A": np.array([0.5, 1.5], dtype=float),
                    "g_r": np.array([0.1, 0.2], dtype=float),
                },
                "metadata": {
                    "analysis": "rdf",
                    "analysis_schema_version": 1,
                    "profile_uid": "rdf-oh",
                    "species_a": "O",
                    "species_b": "H",
                    "n_frames": 2,
                    "bin_width_A": 1.0,
                },
            },
            {
                "datasets": {
                    "bin_centers_A": np.array([0.5, 1.5], dtype=float),
                    "g_r": np.array([0.3, 0.4], dtype=float),
                },
                "metadata": {
                    "analysis": "rdf",
                    "analysis_schema_version": 1,
                    "profile_uid": "rdf-oo",
                    "species_a": "O",
                    "species_b": "O",
                    "n_frames": 2,
                    "bin_width_A": 1.0,
                },
            },
        ],
    )

    context = _build_rdf_gui_context(
        argparse.Namespace(species_a="O", species_b="H", view_mapping=None),
        sources=[str(source)],
    )

    assert [item["default_label"] for item in context.series_descriptors] == ["O-H", "O-O"]
    assert [(profile.species_a, profile.species_b) for profile in context.profile] == [
        ("O", "H"),
        ("O", "O"),
    ]


def test_build_coordination_profile_filter_options_tracks_axes_by_pair():
    options = _build_coordination_profile_filter_options(
        [
            (
                "a.h5",
                [
                    {"metadata": {"species_a": "K", "species_b": "O", "axis": "z"}},
                    {"metadata": {"species_a": "K", "species_b": "O", "axis": "x"}},
                ],
            ),
            (
                "b.h5",
                [
                    {"metadata": {"species_a": "K", "species_b": "O", "axis": "z"}},
                    {"metadata": {"species_a": "Li", "species_b": "O", "axis": "z"}},
                ],
            ),
        ]
    )

    assert options["species_a"] == ["K"]
    assert options["species_b_by_species_a"]["K"] == ["O"]
    assert options["axes"] == ["z"]
    assert options["axes_by_species_pair"]["K"]["O"] == ["z"]


def test_without_preview_series_state_drops_rendered_series_arrays():
    filtered = _without_preview_series_state(
        {
            "title": "Preview",
            "series_labels": ["Only rendered line"],
            "line_color": "#1f77b4",
            "line_colors": ["#1f77b4"],
            "line_kwargs": {"alpha": 0.5},
            "markers": True,
            "series_enabled": [False, True],
        }
    )

    assert filtered == {"title": "Preview"}


def test_density_gui_series_ids_stay_stable_between_multi_source_and_reopened_combined_hdf5(
    tmp_path,
):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)

    args = cli_mod.argparse.Namespace(
        species=None,
        axis="z",
        x_mode="distance",
        quantity="mass",
        series_labels=None,
        line_colors=None,
        series_overrides=None,
        _runtime_argv=(),
    )
    multi_context = _build_density_gui_context(
        args,
        sources=[str(source_a), str(source_b)],
    )
    combined_path = _combine_analysis_hdf5_sources(
        sources=[str(source_a), str(source_b)],
        analysis="density",
        output=tmp_path / "combined_density.h5",
    )
    reopened_context = _build_density_gui_context(
        args,
        sources=[str(combined_path)],
    )

    assert [item["series_id"] for item in multi_context.series_descriptors] == [
        item["series_id"] for item in reopened_context.series_descriptors
    ]


def test_density_gui_reopened_combined_hdf5_preserves_original_source_metadata(tmp_path):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)

    combined_path = _combine_analysis_hdf5_sources(
        sources=[str(source_a), str(source_b)],
        analysis="density",
        output=tmp_path / "combined_density.h5",
    )
    args = cli_mod.argparse.Namespace(
        species=None,
        axis="z",
        x_mode="distance",
        quantity="mass",
        series_labels=None,
        line_colors=None,
        series_overrides=None,
        _runtime_argv=(),
    )
    reopened_context = _build_density_gui_context(args, sources=[str(combined_path)])

    assert [item["source_name"] for item in reopened_context.series_descriptors] == [
        source_a.name,
        source_b.name,
    ]
    assert [
        Path(item["source_path"]).resolve() for item in reopened_context.series_descriptors
    ] == [
        source_a.resolve(),
        source_b.resolve(),
    ]


def test_plot_density_multi_non_gui_does_not_write_combined_settings_hdf5(tmp_path, monkeypatch):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)
    write_plot_profile(
        source_a,
        "plot:density",
        _saved_plot_profile("plot:density", {
            "series_labels": ["H2O"],
            "line_colors": ["#1f77b4"],
        }),
    )

    captured: dict[str, object] = {}

    def _fake_render_profile_plot(**kwargs):
        captured["source"] = kwargs["source"]
        return None, {}

    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--no-show",
        ]
    )

    assert rc == 0
    original_settings = _read_flat_plot_profile(source_a, "plot:density")
    assert original_settings is not None
    assert original_settings["series_labels"] == ["H2O"]
    assert original_settings["line_colors"] == ["#1f77b4"]

    combined_files = sorted(tmp_path.glob("*density_combined*.h5"))
    assert len(combined_files) == 0
    assert captured["source"] == "multi_source_density"


def test_plot_density_multi_auto_merges_series_labels_and_colors_from_sources(
    tmp_path, monkeypatch
):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source_a = tmp_path / "source_a_density.h5"
    source_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_a)
    save_density_profile(profile, source_b)
    write_plot_profile(
        source_a,
        "plot:density",
        _saved_plot_profile(
            "plot:density",
            {"series_labels": ["run-A"], "line_colors": ["#ff0000"]},
        ),
    )
    write_plot_profile(
        source_b,
        "plot:density",
        _saved_plot_profile(
            "plot:density",
            {"series_labels": ["run-B"], "line_colors": ["#00ff00"]},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_render_profile_plot(**kwargs):
        captured["series_labels"] = kwargs["args"].series_labels
        captured["line_colors"] = kwargs["args"].line_colors
        return None, {}

    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured["series_labels"] == ["run-A", "run-B"]
    assert captured["line_colors"] == ["#ff0000", "#00ff00"]


def test_plot_density_ignores_stale_saved_series_settings_when_counts_do_not_match(
    tmp_path, monkeypatch
):
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    source = tmp_path / "source_density.h5"
    save_density_profile(profile, source)
    write_plot_profile(
        source,
        "plot:density",
        _saved_plot_profile("plot:density", {
            "series_labels": ["first", "second"],
            "line_colors": ["#ff0000", "#00ff00"],
        }),
    )

    captured: dict[str, object] = {}

    def _fake_render_profile_plot(**kwargs):
        captured["series_labels"] = kwargs["args"].series_labels
        captured["line_colors"] = kwargs["args"].line_colors
        return None, {}

    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured["series_labels"] is None
    assert captured["line_colors"] is None


def test_plot_density_passes_x_mode_and_quantity_to_plotter(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_csv = tmp_path / "source_density.h5"
    save_density_profile(profile, source_csv)

    captured_kwargs: dict[str, object] = {}

    def _fake_plot_density_profiles(_profiles, **kwargs):
        captured_kwargs["view_mapping"] = kwargs["view_mapping"]
        return None

    monkeypatch.setattr("linak.analysis.density.plot_density_profiles", _fake_plot_density_profiles)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_csv),
            "--x-mode",
            "axis",
            "--quantity",
            "number",
            "--no-show",
        ]
    )

    assert rc == 0
    view_mapping = captured_kwargs["view_mapping"]
    assert view_mapping.x == "axis_coordinate"
    assert view_mapping.y == "number_density"
    assert view_mapping.fixed_values["x_mode"] == "axis"
    assert view_mapping.fixed_values["quantity"] == "number"


def test_build_density_gui_context_selects_heatmap_contract(tmp_path):
    frame = Atoms(
        "OO",
        positions=[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
        cell=[10.0, 11.0, 12.0],
        pbc=True,
    )
    output = tmp_path / "density.h5"
    save_density_profiles(
        compute_all_density_profiles(
            [frame],
            species="O",
            bin_width=1.0,
            surface_mode="none",
            outputs="all",
        ),
        output,
    )
    args = argparse.Namespace(
        species="O",
        axis=None,
        plane="xy",
        x_mode="distance",
        quantity="mass",
        view_mapping=PlotViewMapping(
            view_type_id="heatmap_2d",
            x="x_bin_center",
            y="y_bin_center",
            role_assignments={"z": "mass_density_2d"},
        ),
        heatmap_vmin=None,
        heatmap_vmax=None,
        heatmap_cmap=None,
        heatmap_log_scale=False,
        heatmap_colorbar_enabled=True,
        heatmap_colorbar_label=None,
        heatmap_colorbar_label_size=None,
        heatmap_colorbar_tick_size=None,
        heatmap_colorbar_position="right",
        heatmap_colorbar_pad=None,
        heatmap_colorbar_shrink=None,
        heatmap_colorbar_aspect=None,
    )

    context = cli_mod._build_density_gui_context(args, sources=[str(output)])

    assert context.plotter_kwargs["view_mapping"].view_type_id == "heatmap_2d"
    assert context.profile_filter_options["density_heatmap_plot_contract"]["default_view_type_id"] == "heatmap_2d"


def test_resolve_density_plot_axis_and_x_mode_prefers_explicit_cartesian_values():
    assert cli_mod._resolve_density_plot_axis_and_x_mode(axis=None, x_mode="x") == (None, "x")
    assert cli_mod._resolve_density_plot_axis_and_x_mode(axis="z", x_mode="y") == ("z", "y")
    assert cli_mod._resolve_density_plot_axis_and_x_mode(axis="y", x_mode="axis") == (
        "y",
        "axis",
    )


def test_plot_density_gui_mode_uses_gui_launcher(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "source_density.h5"
    save_density_profile(profile, source_h5)

    captured: dict[str, object] = {}

    def _fake_launch_profile_plot_gui(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._launch_profile_plot_gui", _fake_launch_profile_plot_gui)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["profile_key"] == "plot:density"
    assert captured["analysis_name"] == "density"


def test_plot_parser_no_border_flag_reaches_shared_plot_style():
    args = build_parser().parse_args(["plot", "dummy.h5", "--no-border"])

    style = cli_mod._build_plot_style(args)

    assert args.border is False
    assert style.axes_border is False


def test_plot_density_gui_initial_settings_include_analysis_controls(tmp_path, monkeypatch):
    source_h5 = tmp_path / "source_density.h5"
    _write_density_collection_hdf5(source_h5)

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--species",
            "H2O",
            "--axis",
            "y",
            "--x-mode",
            "axis",
            "--quantity",
            "number",
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["analysis_name"] == "density"
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["species"] == "H2O"
    assert initial["axis"] == "y"
    assert initial["view_mapping"]["view_type_id"] == "line_1d"
    assert initial["view_mapping"]["x"] == "axis_coordinate"
    assert initial["view_mapping"]["y"] == "number_density"
    assert initial["view_mapping"]["fixed_values"]["x_mode"] == "axis"
    assert initial["view_mapping"]["fixed_values"]["quantity"] == "number"
    resolver = captured["on_resolve_series_defaults"]
    resolved = resolver(initial)
    assert resolved["series_count"] == 1
    assert resolved["series_labels"] == ["H2O"]
    assert captured["title"] == "LiNaK Plot Controls: Density"
    assert captured["initial_profile_name"] == "Default"


def test_plot_density_gui_explicit_cartesian_x_mode_sets_matching_axis(tmp_path, monkeypatch):
    source_h5 = tmp_path / "source_density_y.h5"
    _write_density_collection_hdf5(source_h5, surface_axis="y")

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--x-mode",
            "y",
            "--quantity",
            "number",
            "--gui",
        ]
    )

    assert rc == 0
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["view_mapping"]["fixed_values"]["x_mode"] == "y"
    assert initial["view_mapping"]["fixed_values"]["quantity"] == "number"


def test_plot_density_gui_accepts_combined_all_axis_density_hdf5(tmp_path, monkeypatch):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["series_count"] == 3
    assert len(initial["series_descriptors"]) == initial["series_count"]
    assert [item["default_label"] for item in initial["series_descriptors"]] == ["H", "O", "H2O"]


def test_density_gui_combined_all_axis_uses_species_based_logical_descriptors(tmp_path):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    args = cli_mod.argparse.Namespace(
        species=None,
        axis=None,
        x_mode="distance",
        quantity="mass",
        series_labels=None,
        line_colors=None,
        series_overrides=None,
        _runtime_argv=(),
    )

    context = cli_mod._build_density_gui_logical_context(args, sources=[str(source_h5)])

    assert [item["default_label"] for item in context.series_descriptors] == ["H", "O", "H2O"]
    assert all(
        set(item["density_backing_profiles_by_mode"]) == {"distance", "x", "y", "z"}
        for item in context.series_descriptors
    )


def test_density_gui_mode_switch_keeps_series_ids_and_labels_stable(tmp_path):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    base_kwargs = {
        "species": None,
        "axis": None,
        "quantity": "mass",
        "series_labels": None,
        "line_colors": None,
        "series_overrides": None,
        "_runtime_argv": (),
    }
    distance_args = cli_mod.argparse.Namespace(**base_kwargs, x_mode="distance")
    x_args = cli_mod.argparse.Namespace(**base_kwargs, x_mode="x")

    logical_distance = cli_mod._build_density_gui_logical_context(
        distance_args,
        sources=[str(source_h5)],
    )
    logical_x = cli_mod._build_density_gui_logical_context(
        x_args,
        sources=[str(source_h5)],
    )

    assert [item["series_id"] for item in logical_distance.series_descriptors] == [
        item["series_id"] for item in logical_x.series_descriptors
    ]
    assert [item["default_label"] for item in logical_distance.series_descriptors] == [
        item["default_label"] for item in logical_x.series_descriptors
    ]


def test_density_gui_render_context_switches_active_backing_mode_without_changing_layers(tmp_path):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    base_kwargs = {
        "species": None,
        "axis": None,
        "quantity": "mass",
        "series_labels": None,
        "line_colors": None,
        "series_overrides": None,
        "_runtime_argv": (),
    }
    distance_args = cli_mod.argparse.Namespace(**base_kwargs, x_mode="distance")
    x_args = cli_mod.argparse.Namespace(**base_kwargs, x_mode="x")

    logical_context = cli_mod._build_density_gui_logical_context(
        distance_args,
        sources=[str(source_h5)],
    )
    distance_render = cli_mod._build_density_gui_context(
        distance_args,
        sources=[str(source_h5)],
    )
    x_render = cli_mod._build_density_gui_context(
        x_args,
        sources=[str(source_h5)],
    )

    logical_ids = {item["series_id"] for item in logical_context.series_descriptors}
    assert {item["series_id"] for item in distance_render.series_descriptors} == logical_ids
    assert {item["series_id"] for item in x_render.series_descriptors} == logical_ids
    assert {profile.coordinate_mode for profile in distance_render.profile} == {"distance"}
    assert {profile.coordinate_mode for profile in x_render.profile} == {"axis"}
    assert {profile.axis for profile in x_render.profile} == {"x"}


def test_density_gui_ignores_axis_prefilter_when_switching_x_mode(tmp_path):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    args = cli_mod.argparse.Namespace(
        species=None,
        axis="z",
        x_mode="x",
        quantity="mass",
        series_labels=None,
        line_colors=None,
        series_overrides=None,
        _runtime_argv=(),
    )

    logical_context = cli_mod._build_density_gui_logical_context(args, sources=[str(source_h5)])
    render_context = cli_mod._build_density_gui_context(args, sources=[str(source_h5)])

    assert len(logical_context.series_descriptors) == 3
    assert len(render_context.series_descriptors) == 3
    assert {profile.coordinate_mode for profile in render_context.profile} == {"axis"}
    assert {profile.axis for profile in render_context.profile} == {"x"}


def test_plot_density_gui_preview_switch_to_x_mode_does_not_drop_all_profiles(tmp_path, monkeypatch):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    preview_calls: list[dict[str, object]] = []

    def _fake_render_profile_plot(**kwargs):
        preview_calls.append(kwargs)
        return None, {}

    def _fake_gui_launcher(**kwargs):
        initial_settings = deepcopy(kwargs["initial_settings"])
        preview_settings = deepcopy(initial_settings)
        preview_settings["x_mode"] = "x"
        preview_settings["axis"] = "x"
        kwargs["on_preview"](preview_settings)

    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert preview_calls
    x_mode_calls = [call for call in preview_calls if call["args"].x_mode == "x"]
    assert x_mode_calls
    rendered_profiles = x_mode_calls[-1]["profile"]
    assert len(rendered_profiles) == 3
    assert {profile.coordinate_mode for profile in rendered_profiles} == {"axis"}
    assert {profile.axis for profile in rendered_profiles} == {"x"}


def test_plot_density_combined_all_axis_hdf5_supports_number_quantity(tmp_path):
    source_h5 = tmp_path / "traj_density.h5"
    _write_density_collection_hdf5(source_h5)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--quantity",
            "number",
            "--no-show",
        ]
    )

    assert rc == 0


def test_plot_density_gui_multi_sources_create_combined_hdf5(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    captured: dict[str, object] = {}

    def _fake_launch_profile_plot_gui(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._launch_profile_plot_gui", _fake_launch_profile_plot_gui)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_h5_a),
            str(source_h5_b),
            "--gui",
        ]
    )

    assert rc == 0
    combined_source = captured["source_path"]
    assert isinstance(combined_source, Path)
    assert combined_source.exists()
    loaded = load_density_profiles(combined_source)
    assert len(loaded) == 2


def test_plot_density_gui_multi_sources_do_not_copy_saved_plot_settings(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    write_plot_profile(
        source_h5_b,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "From second"}),
    )

    captured: dict[str, object] = {}

    def _fake_launch_profile_plot_gui(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._launch_profile_plot_gui", _fake_launch_profile_plot_gui)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_h5_a),
            str(source_h5_b),
            "--settings-source",
            "2",
            "--gui",
        ]
    )

    assert rc == 0
    combined_source = captured["source_path"]
    assert isinstance(combined_source, Path)
    copied_settings = _read_flat_plot_profile(combined_source, "plot:density")
    assert copied_settings is None


def test_combine_analysis_hdf5_sources_write_data_without_plot_settings(tmp_path):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    write_plot_profile(
        source_h5_b,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "From second"}),
    )

    output_path = cli_mod._combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
        settings_source_path=source_h5_b,
    )

    assert output_path == combined_h5.resolve()
    imported_settings = _read_flat_plot_profile(output_path, "plot:density")
    assert imported_settings is None
    profiles = read_linak_hdf5_profiles(output_path, expected_analysis="density")
    assert profiles
    metadata_by_path = {
        Path(metadata["origin_hdf5_path"]).resolve() for _datasets, metadata in profiles
    }
    assert "settings_source" not in profiles[0][1]
    assert metadata_by_path == {source_h5_a.resolve(), source_h5_b.resolve()}


def test_plot_density_gui_multi_sources_use_default_series_labels_without_saved_overrides(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    write_plot_profile(
        source_h5_a,
        "plot:density",
        _saved_plot_profile(
            "plot:density",
            {"series_labels": ["run-A"], "line_colors": ["#ff0000"]},
        ),
    )
    write_plot_profile(
        source_h5_b,
        "plot:density",
        _saved_plot_profile(
            "plot:density",
            {"series_labels": ["run-B"], "line_colors": ["#00ff00"]},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_launch_profile_plot_gui(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._launch_profile_plot_gui", _fake_launch_profile_plot_gui)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_h5_a),
            str(source_h5_b),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["args"].series_labels is None
    assert captured["args"].line_colors is None
    assert captured["initial_context"].default_series_labels == [
        f"{source_h5_a.name}:O",
        f"{source_h5_b.name}:O",
    ]
    combined_source = captured["source_path"]
    assert isinstance(combined_source, Path)
    merged_settings = _read_flat_plot_profile(combined_source, "plot:density")
    assert merged_settings is None


def test_plot_density_gui_multi_source_first_open_matches_reopened_combined_hdf5_defaults(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)

    first_launch: dict[str, object] = {}

    def _fake_launch_profile_plot_gui(**kwargs):
        first_launch.update(kwargs)

    monkeypatch.setattr("linak.cli._launch_profile_plot_gui", _fake_launch_profile_plot_gui)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_h5_a),
            str(source_h5_b),
            "--gui",
        ]
    )

    assert rc == 0
    combined_source = first_launch["source_path"]
    assert isinstance(combined_source, Path)
    assert first_launch["initial_context"].default_series_labels == [
        f"{source_h5_a.name}:O",
        f"{source_h5_b.name}:O",
    ]

    reopened_args = cli_mod.build_parser().parse_args(["plot", str(combined_source)])
    reopened_args._runtime_argv = ("plot", str(combined_source))
    reopened_context = cli_mod._build_density_gui_logical_context(
        reopened_args,
        sources=[str(combined_source)],
    )

    assert reopened_context.default_series_labels == [
        f"{source_h5_a.name}:O",
        f"{source_h5_b.name}:O",
    ]
    assert reopened_context.series_descriptors == first_launch["initial_context"].series_descriptors


def test_plot_density_gui_preview_renders_for_each_request(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "source_density.h5"
    save_density_profile(profile, source_h5)

    preview_calls: list[dict[str, object]] = []

    def _fake_render_profile_plot(**kwargs):
        preview_calls.append(kwargs)
        return None, {}

    def _fake_gui_launcher(**kwargs):
        kwargs["on_preview"]({"x_min": 0.0})
        kwargs["on_preview"]({"x_min": 1.0})

    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(preview_calls) == 3
    assert preview_calls[0]["args"].show is False
    assert preview_calls[1]["args"].x_min == 0.0
    assert preview_calls[2]["args"].x_min == 1.0


def test_plot_density_gui_seed_render_uses_noninteractive_backend_without_warning(
    tmp_path, monkeypatch, caplog
):
    source_h5 = tmp_path / "source_density.h5"
    _write_density_hdf5(source_h5)

    backend_calls: list[bool] = []

    def _fake_configure_backend(*, interactive, preferred_backend=None):
        del preferred_backend
        backend_calls.append(bool(interactive))
        return "QtAgg" if interactive else "Agg"

    monkeypatch.setattr(
        "linak.plot.plotting.configure_matplotlib_backend",
        _fake_configure_backend,
    )
    monkeypatch.setattr(
        "linak.analysis.density.plot_density_profiles",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", lambda **_kwargs: None)

    caplog.set_level("INFO")
    rc = main(
        [
            "--log-level",
            "INFO",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert backend_calls
    assert all(call is False for call in backend_calls)
    assert (
        "No interactive display or output path requested. Nothing was rendered." not in caplog.text
    )


def test_plot_density_gui_preview_switches_from_seed_agg_to_interactive_backend(
    tmp_path, monkeypatch
):
    source_h5 = tmp_path / "source_density.h5"
    _write_density_hdf5(source_h5)

    backend_calls: list[bool] = []

    def _fake_configure_backend(*, interactive, preferred_backend=None):
        del preferred_backend
        backend_calls.append(bool(interactive))
        return "QtAgg" if interactive else "Agg"

    monkeypatch.setattr(
        "linak.plot.plotting.configure_matplotlib_backend",
        _fake_configure_backend,
    )
    monkeypatch.setattr(
        "linak.analysis.density.plot_density_profiles",
        lambda *_args, **_kwargs: None,
    )

    def _fake_gui_launcher(**kwargs):
        kwargs["on_preview"]({})

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert backend_calls
    assert backend_calls[0] is False
    assert True in backend_calls


def test_plot_density_gui_provides_hdf5_import_callback(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "source_density.h5"
    save_density_profile(profile, source_h5)
    write_plot_profile(
        source_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Imported title"}),
    )

    imported_payload: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        list_callback = kwargs.get("on_list_import_hdf5_profiles")
        assert callable(list_callback)
        listing = list_callback(str(source_h5))
        assert listing["available_names"] == ["Default"]
        callback = kwargs.get("on_import_hdf5")
        assert callable(callback)
        imported_payload.update(callback(str(source_h5), "Default"))

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert imported_payload["title"] == "Imported title"


def test_plot_density_gui_uses_requested_named_settings_profile(tmp_path, monkeypatch):
    source_h5 = tmp_path / "source_density.h5"
    _write_density_hdf5(source_h5)
    write_plot_profile(
        source_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Default title"}),
    )
    write_plot_profile(
        source_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Paper title", "x_mode": "axis"}),
        profile_name="Paper",
    )
    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--settings-profile",
            "Paper",
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["initial_profile_name"] == "Paper"
    assert captured["available_profile_names"] == ["Default", "Paper"]
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["title"] == "Paper title"
    assert initial["view_mapping"]["x"] == "axis_coordinate"
    assert initial["view_mapping"]["fixed_values"]["x_mode"] == "axis"


def test_plot_density_gui_combined_hdf5_enables_named_profile_management(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["allow_named_profiles"] is True
    assert captured["available_profile_names"] == ["Default"]
    assert captured["initial_profile_name"] == "Default"
    assert callable(captured["on_load_profile"])
    assert callable(captured["on_delete_profile"])
    assert callable(captured["on_set_active_profile"])


def test_plot_density_gui_combined_hdf5_named_profiles_round_trip(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    on_save = captured["on_save"]
    on_load_profile = captured["on_load_profile"]
    on_set_active_profile = captured["on_set_active_profile"]
    assert callable(on_save)
    assert callable(on_load_profile)
    assert callable(on_set_active_profile)

    on_save("Publication", {"title": "Combined paper"})
    on_set_active_profile("Publication")

    assert read_plot_profile_names(combined_h5, "plot:density") == ["Publication"]
    saved = _read_flat_plot_profile(
        combined_h5,
        "plot:density",
        profile_name="Publication",
    )
    assert isinstance(saved, dict)
    assert saved["title"] == "Combined paper"
    assert read_active_plot_profile_name(combined_h5, "plot:density") == "Publication"
    assert on_load_profile("Publication")["title"] == "Combined paper"


def test_plot_density_gui_duplicate_callback_copies_saved_profile_exactly(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    on_save = captured["on_save"]
    on_duplicate_profile = captured["on_duplicate_profile"]
    assert callable(on_save)
    assert callable(on_duplicate_profile)

    settings = {
        "title": "Exact profile",
        "x_lim": [0.0, 2.0],
        "y_lim": [None, 5.0],
        "_gui_sync_modes": {"x_lim": "manual", "y_lim": "manual"},
        "view_mapping": {
            "view_type_id": "line_1d",
            "x": "distance_to_surface",
            "y": "mass_density",
            "color": None,
            "split_by": None,
            "filter_by": None,
            "filter_min": None,
            "filter_max": None,
            "role_assignments": {},
            "fixed_values": {"quantity": "mass"},
        },
        "series_overrides": {"series:0": {"enabled": False}},
    }
    on_save("Default", settings)
    message = on_duplicate_profile("Default", "Default Copy")

    assert message == "Duplicated profile 'Default' as 'Default Copy' in 'combined_density.h5'."
    assert read_active_plot_profile_name(combined_h5, "plot:density") == "Default Copy"
    assert read_plot_profile(
        combined_h5,
        "plot:density",
        profile_name="Default Copy",
    ) == read_plot_profile(combined_h5, "plot:density", profile_name="Default")
    copied = _read_flat_plot_profile(
        combined_h5,
        "plot:density",
        profile_name="Default Copy",
    )
    assert copied["x_lim"] == [0.0, 2.0]
    assert copied["y_lim"] == [None, 5.0]
    assert copied["_gui_sync_modes"] == {"x_lim": "manual", "y_lim": "manual"}
    assert copied["series_overrides"] == {"series:0": {"enabled": False}}


def test_plot_density_gui_combined_hdf5_renames_default_profile_via_callback(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )
    write_plot_profile(
        combined_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Default title"}),
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    on_rename_profile = captured["on_rename_profile"]
    on_load_profile = captured["on_load_profile"]
    assert callable(on_rename_profile)
    assert callable(on_load_profile)

    rename_message = on_rename_profile("Default", "My Profile")

    assert rename_message == "Renamed profile 'Default' to 'My Profile' in 'combined_density.h5'."
    assert read_plot_profile_names(combined_h5, "plot:density") == ["My Profile"]
    assert read_active_plot_profile_name(combined_h5, "plot:density") == "My Profile"
    assert on_load_profile("My Profile")["title"] == "Default title"


def test_plot_density_gui_hdf5_import_callback_respects_explicit_selected_profile(
    tmp_path, monkeypatch
):
    source_h5 = tmp_path / "source_density.h5"
    _write_density_hdf5(source_h5)
    write_plot_profile(
        source_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Default title"}),
    )
    write_plot_profile(
        source_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Paper title"}),
        profile_name="Paper",
    )

    captured_listing: dict[str, object] = {}
    imported_payload: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        list_callback = kwargs.get("on_list_import_hdf5_profiles")
        import_callback = kwargs.get("on_import_hdf5")
        assert callable(list_callback)
        assert callable(import_callback)
        listing = list_callback(str(source_h5))
        captured_listing.update(listing)
        imported_payload.update(import_callback(str(source_h5), "Paper"))

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured_listing["available_names"] == ["Default", "Paper"]
    assert captured_listing["active_name"] == "Paper"
    assert imported_payload["title"] == "Paper title"


def test_plot_density_gui_combined_hdf5_does_not_pass_reordered_series_lists_to_gui(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    args = cli_mod.build_parser().parse_args(["plot", str(combined_h5)])
    args._runtime_argv = ("plot", str(combined_h5))
    cli_mod._apply_saved_plot_settings(
        args=args,
        source_path=combined_h5,
        profile_key="plot:density",
        keys=cli_mod._PLOT_SETTINGS_DENSITY_KEYS,
    )
    context = _build_density_gui_context(args, sources=[str(combined_h5)])
    cli_mod._apply_effective_series_settings(
        args=args,
        sources=[str(combined_h5)],
        profile_key="plot:density",
        fallback_labels_by_source=context.fallback_labels_by_source,
        series_descriptors=context.series_descriptors,
        allow_saved_multi_source_merge=True,
    )
    write_plot_profile(
        combined_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {
            "series_order": [
                context.series_descriptors[1]["series_id"],
                context.series_descriptors[0]["series_id"],
            ],
            "series_overrides": {
                context.series_descriptors[0]["series_id"]: {"enabled": False},
                context.series_descriptors[1]["series_id"]: {"label_override": "kept"},
            },
        }),
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["series_overrides"] is not None
    assert "series_enabled" not in initial
    assert "series_labels" not in initial
    assert "line_colors" not in initial


def test_plot_density_gui_first_open_materializes_id_keyed_series_state(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    descriptors = initial["series_descriptors"]
    assert isinstance(descriptors, list)
    overrides = initial.get("series_overrides")
    assert isinstance(overrides, dict)
    assert set(overrides) == {item["series_id"] for item in descriptors}
    assert "series_enabled" not in initial
    assert "series_labels" not in initial
    assert "line_colors" not in initial


def test_plot_density_gui_first_open_preview_disables_series_immediately(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    render_calls: list[dict[str, object]] = []
    disabled_series_id_holder: dict[str, str] = {}

    def _fake_load_density_profiles_by_index(path, indices, *, axis=None, species=None):
        return load_density_profiles_by_index(path, indices, axis=axis, species=species)

    def _fake_render_profile_plot(**kwargs):
        render_calls.append(
            {
                "series_enabled": deepcopy(kwargs["args"].series_enabled),
                "line_colors": deepcopy(kwargs["args"].line_colors),
                "series_overrides": deepcopy(getattr(kwargs["args"], "series_overrides", None)),
                "render_series_ids": [
                    str(item.get("series_id") or "")
                    for item in kwargs.get("render_series_descriptors") or []
                ],
            }
        )
        return None, {}

    def _fake_gui_launcher(**kwargs):
        initial_settings = deepcopy(kwargs["initial_settings"])
        descriptors = initial_settings["series_descriptors"]
        disabled_series_id = descriptors[0]["series_id"]
        disabled_series_id_holder["value"] = disabled_series_id
        disabled_settings = deepcopy(initial_settings)
        disabled_settings["series_overrides"] = deepcopy(
            disabled_settings.get("series_overrides") or {}
        )
        disabled_settings["series_overrides"].setdefault(disabled_series_id, {})
        disabled_settings["series_overrides"][disabled_series_id]["enabled"] = False
        kwargs["on_preview"](disabled_settings)

    monkeypatch.setattr(
        "linak.analysis.density.load_density_profiles_by_index",
        _fake_load_density_profiles_by_index,
    )
    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(render_calls) == 2
    assert render_calls[0]["series_enabled"] is None
    assert render_calls[0]["line_colors"] is None
    assert render_calls[1]["series_enabled"] is None
    assert render_calls[1]["line_colors"] is None
    assert (
        render_calls[1]["series_overrides"][disabled_series_id_holder["value"]]["enabled"] is False
    )


def test_plot_density_gui_reopen_keeps_per_series_alpha_out_of_global_line_kwargs(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    args = cli_mod.build_parser().parse_args(["plot", str(combined_h5)])
    args._runtime_argv = ("plot", str(combined_h5))
    context = _build_density_gui_context(args, sources=[str(combined_h5)])
    write_plot_profile(
        combined_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {
            "series_overrides": {
                context.series_descriptors[0]["series_id"]: {"alpha": 0.5},
            },
        }),
    )

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["series_overrides"][context.series_descriptors[0]["series_id"]]["alpha"] == 0.5
    assert initial.get("line_kwargs") is None or initial.get("line_kwargs") == {}


def test_plot_density_gui_embedded_preview_round_trips_after_save_for_combined_hdf5(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    plot_calls: list[dict[str, object]] = []
    preview_call_indices: list[int] = []
    launch_count = {"value": 0}
    expected: dict[str, object] = {}

    def _fake_plot_density_profiles(_profiles, **kwargs):
        output = kwargs.get("output")
        if output is not None:
            Path(output).write_text("preview", encoding="utf-8")
        plot_calls.append(deepcopy(kwargs))
        return Path(output) if output is not None else None

    def _normalized_plot_call(payload: dict[str, object]) -> dict[str, object]:
        normalized = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in {"output", "capture_state"}
        }
        overrides = normalized.get("series_overrides_by_id")
        if isinstance(overrides, dict):
            cleaned_overrides: dict[str, dict[str, object]] = {}
            for series_id, entry in overrides.items():
                if not isinstance(entry, dict):
                    continue
                cleaned_entry = deepcopy(entry)
                if cleaned_entry.get("enabled") is True:
                    cleaned_entry.pop("enabled", None)
                if cleaned_entry.get("show_in_legend") is True:
                    cleaned_entry.pop("show_in_legend", None)
                if cleaned_entry.get("line_kwargs") == {"alpha": 0.5} and "alpha" in cleaned_entry:
                    cleaned_entry.pop("line_kwargs", None)
                cleaned_overrides[str(series_id)] = cleaned_entry
            normalized["series_overrides_by_id"] = cleaned_overrides
        return normalized

    def _fake_gui_launcher(**kwargs):
        preview_call_indices.append(len(plot_calls))
        initial_settings = deepcopy(kwargs["initial_settings"])
        descriptors = initial_settings["series_descriptors"]
        reordered_ids = [descriptors[1]["series_id"], descriptors[0]["series_id"]]
        if launch_count["value"] == 0:
            expected["series_order"] = reordered_ids
            expected["first_series_id"] = descriptors[0]["series_id"]
            expected["second_series_id"] = descriptors[1]["series_id"]
            initial_settings["series_order"] = reordered_ids
            initial_settings["series_overrides"] = {
                descriptors[0]["series_id"]: {
                    "enabled": False,
                    "alpha": 0.5,
                    "color": "#ff0000",
                },
                descriptors[1]["series_id"]: {
                    "label_override": "reordered",
                    "line_width": 3.0,
                    "marker": "o",
                },
            }
            kwargs["on_save_figure"](initial_settings, str(tmp_path / "preview_before.png"))
            kwargs["on_save"]("Default", initial_settings)
        else:
            kwargs["on_save_figure"](initial_settings, str(tmp_path / "preview_after.png"))
        launch_count["value"] += 1

    monkeypatch.setattr("linak.analysis.density.plot_density_profiles", _fake_plot_density_profiles)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc_first = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )
    rc_second = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc_first == 0
    assert rc_second == 0
    assert len(preview_call_indices) == 2
    before_preview = _normalized_plot_call(plot_calls[preview_call_indices[0]])
    after_preview = _normalized_plot_call(plot_calls[preview_call_indices[1]])
    assert after_preview == before_preview
    saved = _read_flat_plot_profile(combined_h5, "plot:density")
    assert saved is not None
    assert saved["series_order"] == expected["series_order"]
    assert saved["series_overrides"][expected["first_series_id"]]["alpha"] == 0.5
    assert saved["series_overrides"][expected["first_series_id"]]["enabled"] is False
    assert saved["series_overrides"][expected["second_series_id"]]["label_override"] == "reordered"
    assert saved.get("line_kwargs") is None or saved.get("line_kwargs") == {}


def test_plot_density_gui_reopen_preserves_enabled_fit_settings(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "density.h5"
    save_density_profile(profile, source_h5)

    launches: list[dict[str, object]] = []
    preview_calls = {"count": 0}

    def _fake_gui_launcher(**kwargs):
        launches.append(deepcopy(kwargs["initial_settings"]))
        if len(launches) == 1:
            initial_settings = deepcopy(kwargs["initial_settings"])
            first_series_id = initial_settings["series_descriptors"][0]["series_id"]
            initial_settings["series_overrides"] = {
                first_series_id: {
                    "fit": {
                        "fit_enabled": True,
                        "fit_type": "linear",
                        "fit_range_mode": "visible",
                        "fit_x_min": None,
                        "fit_x_max": None,
                        "fit_initial_guess": None,
                        "fit_bounds": None,
                        "fit_label_override": None,
                        "fit_show_in_legend": True,
                    }
                }
            }
            kwargs["on_save"]("Default", initial_settings)
        else:
            kwargs["on_save_figure"](
                deepcopy(kwargs["initial_settings"]),
                str(tmp_path / "fit_preview_after_reopen.png"),
            )
            preview_calls["count"] += 1

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc_first = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )
    rc_second = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc_first == 0
    assert rc_second == 0
    assert len(launches) == 2
    first_series_id = launches[0]["series_descriptors"][0]["series_id"]
    saved = _read_flat_plot_profile(source_h5, "plot:density")
    assert saved is not None
    assert saved["series_overrides"][first_series_id]["fit"]["fit_enabled"] is True
    assert launches[1]["series_overrides"][first_series_id]["fit"]["fit_enabled"] is True
    assert preview_calls["count"] == 1


def test_plot_density_gui_reopen_preserves_normalization_settings(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "density.h5"
    save_density_profile(profile, source_h5)

    launches: list[dict[str, object]] = []

    def _fake_gui_launcher(**kwargs):
        launches.append(deepcopy(kwargs["initial_settings"]))
        if len(launches) == 1:
            initial_settings = deepcopy(kwargs["initial_settings"])
            first_series_id = initial_settings["series_descriptors"][0]["series_id"]
            initial_settings["series_overrides"] = {
                first_series_id: {
                    "normalization_mode": "max",
                    "normalization_value": 1.0,
                }
            }
            kwargs["on_save"]("Default", initial_settings)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc_first = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )
    rc_second = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc_first == 0
    assert rc_second == 0
    assert len(launches) == 2
    first_series_id = launches[0]["series_descriptors"][0]["series_id"]
    saved = _read_flat_plot_profile(source_h5, "plot:density")
    assert saved is not None
    assert saved["series_overrides"][first_series_id]["normalization_mode"] == "max"
    assert saved["series_overrides"][first_series_id]["normalization_value"] == 1.0
    assert launches[1]["series_overrides"][first_series_id]["normalization_mode"] == "max"
    assert launches[1]["series_overrides"][first_series_id]["normalization_value"] == 1.0


def test_plot_density_gui_reopen_preserves_normalization_for_all_series(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    launches: list[dict[str, object]] = []

    def _fake_gui_launcher(**kwargs):
        launches.append(deepcopy(kwargs["initial_settings"]))
        if len(launches) == 1:
            initial_settings = deepcopy(kwargs["initial_settings"])
            overrides: dict[str, dict[str, object]] = {}
            for descriptor in initial_settings["series_descriptors"]:
                series_id = descriptor["series_id"]
                overrides[series_id] = {
                    "normalization_mode": "max",
                    "normalization_value": 1.0,
                }
            initial_settings["series_overrides"] = overrides
            kwargs["on_save"]("Default", initial_settings)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc_first = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )
    rc_second = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc_first == 0
    assert rc_second == 0
    assert len(launches) == 2

    saved = _read_flat_plot_profile(combined_h5, "plot:density")
    assert saved is not None
    reopened_overrides = launches[1]["series_overrides"]
    assert isinstance(reopened_overrides, dict)
    for descriptor in launches[0]["series_descriptors"]:
        series_id = descriptor["series_id"]
        assert saved["series_overrides"][series_id]["normalization_mode"] == "max"
        assert saved["series_overrides"][series_id]["normalization_value"] == 1.0
        assert reopened_overrides[series_id]["normalization_mode"] == "max"
        assert reopened_overrides[series_id]["normalization_value"] == 1.0


def test_plot_density_gui_lazy_loading_only_reads_enabled_series_and_evicts_cache(
    tmp_path, monkeypatch
):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5_a = tmp_path / "source_a_density.h5"
    source_h5_b = tmp_path / "source_b_density.h5"
    combined_h5 = tmp_path / "combined_density.h5"
    save_density_profile(profile, source_h5_a)
    save_density_profile(profile, source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
    )

    args = cli_mod.build_parser().parse_args(["plot", str(combined_h5)])
    args._runtime_argv = ("plot", str(combined_h5))
    context = _build_density_gui_context(args, sources=[str(combined_h5)])
    disabled_series_id = context.series_descriptors[0]["series_id"]
    enabled_series_id = context.series_descriptors[1]["series_id"]
    write_plot_profile(
        combined_h5,
        "plot:density",
        _saved_plot_profile("plot:density", {
            "series_overrides": {
                disabled_series_id: {"enabled": False},
            },
        }),
    )

    load_calls: list[list[int]] = []

    def _fake_load_density_profiles_by_index(path, indices, *, axis=None, species=None):
        load_calls.append([int(index) for index in indices])
        return load_density_profiles_by_index(path, indices, axis=axis, species=species)

    render_args_seen: list[argparse.Namespace] = []

    def _fake_render_profile_plot(**_kwargs):
        render_args_seen.append(deepcopy(_kwargs["args"]))
        return None, {}

    def _fake_gui_launcher(**kwargs):
        initial_settings = deepcopy(kwargs["initial_settings"])
        resolved = kwargs["on_resolve_series_defaults"](initial_settings)
        assert resolved["series_count"] == 2
        assert [item["series_id"] for item in resolved["series_descriptors"]] == [
            disabled_series_id,
            enabled_series_id,
        ]

        kwargs["on_preview"](deepcopy(initial_settings))

        no_series = deepcopy(initial_settings)
        no_series["series_overrides"] = {
            disabled_series_id: {"enabled": False},
            enabled_series_id: {"enabled": False},
        }
        with pytest.raises(ValueError, match="No series are enabled"):
            kwargs["on_preview"](no_series)
        with pytest.raises(ValueError, match="before exporting"):
            kwargs["on_save_figure"](no_series, str(tmp_path / "disabled.png"))

        kwargs["on_preview"](deepcopy(initial_settings))

    monkeypatch.setattr(
        "linak.analysis.density.load_density_profiles_by_index",
        _fake_load_density_profiles_by_index,
    )
    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert load_calls == [[1], [1]]
    assert len(render_args_seen) >= 2
    for render_args in render_args_seen:
        assert render_args.series_enabled is None
        assert render_args.line_colors is None
        assert render_args.series_show_in_legend is None


def test_plot_position_gui_lazy_loading_only_reads_requested_parent_profile(tmp_path, monkeypatch):
    source_h5_a = tmp_path / "source_a_position.h5"
    source_h5_b = tmp_path / "source_b_position.h5"
    combined_h5 = tmp_path / "combined_position.h5"
    _write_position_hdf5(source_h5_a)
    _write_position_hdf5(source_h5_b)
    _combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="position",
        output=combined_h5,
    )

    args = cli_mod.build_parser().parse_args(["plot", str(combined_h5)])
    args._runtime_argv = ("plot", str(combined_h5))
    context = cli_mod._build_position_gui_context(args, sources=[str(combined_h5)])
    selected_series_id = context.series_descriptors[-1]["series_id"]
    write_plot_profile(
        combined_h5,
        "plot:position",
        _saved_plot_profile("plot:position", {
            "series_overrides": {
                descriptor["series_id"]: {"enabled": descriptor["series_id"] == selected_series_id}
                for descriptor in context.series_descriptors
            },
        }),
    )

    load_calls: list[list[int]] = []

    def _fake_load_position_profiles_by_index(path, indices, *, species=None, axis=None):
        load_calls.append([int(index) for index in indices])
        return load_position_profiles_by_index(path, indices, species=species, axis=axis)

    def _fake_render_profile_plot(**kwargs):
        assert len(kwargs["profile"]) == 1
        return None, {}

    monkeypatch.setattr(
        "linak.analysis.position.load_position_profiles_by_index",
        _fake_load_position_profiles_by_index,
    )
    monkeypatch.setattr("linak.cli._render_profile_plot", _fake_render_profile_plot)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", lambda **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(combined_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert load_calls == [[1]]


def test_hdf5_plot_settings_named_profile_copy_and_activate(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)
    write_plot_profile(
        source,
        "plot:density",
        _saved_plot_profile("plot:density", {"title": "Default title"}),
    )

    rc_copy = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot-settings",
            str(source),
            "--profile",
            "density",
            "--copy-name",
            "Publication",
        ]
    )
    assert rc_copy == 0
    assert read_plot_profile_names(source, "plot:density") == ["Default", "Publication"]

    rc_set = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot-settings",
            str(source),
            "--profile",
            "density",
            "--name",
            "Publication",
            "--set",
            'title="Publication title"',
        ]
    )
    assert rc_set == 0
    assert _read_flat_plot_profile(source, "plot:density", profile_name="Publication") == {
        "title": "Publication title"
    }

    rc_active = main(
        [
            "--log-level",
            "ERROR",
            "hdf5",
            "plot-settings",
            str(source),
            "--profile",
            "density",
            "--set-active",
            "Publication",
        ]
    )
    assert rc_active == 0
    assert read_active_plot_profile_name(source, "plot:density") == "Publication"


def test_plot_density_gui_preview_uses_non_blocking_show(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "source_density.h5"
    save_density_profile(profile, source_h5)

    plot_calls: list[dict[str, object]] = []

    def _fake_plot_density_profiles(_profiles, **kwargs):
        plot_calls.append(kwargs)
        return None

    def _fake_gui_launcher(**kwargs):
        kwargs["on_preview"]({})

    monkeypatch.setattr("linak.analysis.density.plot_density_profiles", _fake_plot_density_profiles)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(plot_calls) == 2
    assert plot_calls[0]["show"] is False
    assert plot_calls[1]["show"] is True
    assert plot_calls[1]["show_blocking"] is False


def test_plot_density_gui_single_series_label_maps_to_line_label(tmp_path, monkeypatch):
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=1.0)
    source_h5 = tmp_path / "source_density.h5"
    save_density_profile(profile, source_h5)

    plot_calls: list[dict[str, object]] = []

    def _fake_plot_density(_profile, **kwargs):
        plot_calls.append(kwargs)
        return None

    def _fake_gui_launcher(**kwargs):
        kwargs["on_preview"]({"series_labels": ["custom-series"]})

    monkeypatch.setattr("linak.analysis.density.plot_density_profile", _fake_plot_density)
    monkeypatch.setattr("linak.analysis.density.plot_density_profiles", _fake_plot_density)
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(plot_calls) == 2
    assert plot_calls[0]["show"] is False
    assert plot_calls[1].get("line_label") == "custom-series" or plot_calls[1].get(
        "series_labels"
    ) == ["custom-series"]


def test_compute_density_passes_surface_options_to_density_engine(tmp_path, monkeypatch):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.10], [0.0, 0.0, 1.10]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    captured: dict[str, object] = {}

    def _fake_read_trajectory(_source):
        return [frame]

    def _fake_compute_all_density_profiles(**kwargs):
        captured["surface_mode"] = kwargs["surface_mode"]
        captured["surface_elements"] = kwargs["surface_elements"]
        captured["include_fixed_surface_atoms"] = kwargs["include_fixed_surface_atoms"]
        captured["surface_options"] = kwargs["surface_options"]
        profile = compute_density_profile(
            [frame],
            species=kwargs["species"],
            axis=kwargs["surface_axis"],
            bin_width=kwargs["bin_width"],
        )
        return [profile]

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr(
        "linak.analysis.density.compute_all_density_profiles", _fake_compute_all_density_profiles
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--surface-mode",
            "layered",
            "--surface-elements",
            "Au",
            "Pt",
            "--include-fixed-surface-atoms",
            "--rough-surface-envelope",
            "2.5",
        ]
    )

    assert rc == 0
    assert captured["surface_mode"] == "layered"
    assert captured["surface_elements"] == ["Au", "Pt"]
    assert captured["include_fixed_surface_atoms"] is True
    assert captured["surface_options"] is not None
    assert captured["surface_options"].rough_surface_envelope_A == pytest.approx(2.5)


def test_plot_density_rejects_trajectory_input(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    write(trajectory, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(trajectory),
            "--no-show",
        ]
    )

    assert rc == 1
    assert "only accepts HDF5 input" in capsys.readouterr().err


def test_plot_density_rejects_multiple_positional_sources_without_files_flag(tmp_path, capsys):
    source_csv_1 = tmp_path / "source1_density.h5"
    source_csv_2 = tmp_path / "source2_density.h5"
    frame = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.18]])
    profile = compute_density_profile([frame], species="O", axis="z", bin_width=0.1)
    save_density_profile(profile, source_csv_1)
    save_density_profile(profile, source_csv_2)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_csv_1),
            str(source_csv_2),
            "--no-show",
        ]
    )

    assert rc == 1
    assert "Use -f/--files when passing multiple input files." in capsys.readouterr().err


def test_plot_msd_multiple_files_overlays_with_source_labels(tmp_path, monkeypatch):
    profile = compute_msd(
        [
            Atoms("O", positions=[[0.0, 0.0, 0.0]]),
            Atoms("O", positions=[[0.1, 0.0, 0.0]]),
        ],
        species="O",
        timestep_fs=1.0,
    )
    source_csv_1 = tmp_path / "source1_msd.h5"
    source_csv_2 = tmp_path / "source2_msd.h5"
    save_msd_profile(profile, source_csv_1)
    save_msd_profile(profile, source_csv_2)

    captured_labels: list[str] = []

    def _fake_plot_msd_profiles(profiles, **_kwargs):
        captured_labels.extend([item.species for item in profiles])
        return None

    monkeypatch.setattr("linak.analysis.msd.plot_msd_profiles", _fake_plot_msd_profiles)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_csv_1),
            str(source_csv_2),
            "--species",
            "O",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured_labels == [f"{source_csv_1.name}:O", f"{source_csv_2.name}:O"]


def test_plot_msd_rejects_trajectory_input(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    write(trajectory, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(trajectory),
            "--no-show",
        ]
    )

    assert rc == 1
    assert "only accepts HDF5 input" in capsys.readouterr().err


def test_plot_position_multiple_files_overlays_with_source_labels(tmp_path, monkeypatch):
    source_h5_1 = tmp_path / "source1_position.h5"
    source_h5_2 = tmp_path / "source2_position.h5"
    _write_position_hdf5(source_h5_1)
    _write_position_hdf5(source_h5_2)

    captured: dict[str, object] = {}

    def _fake_plot_position_profiles(profiles, **kwargs):
        captured["species"] = [item.species for item in profiles]
        captured["view_mapping"] = kwargs.get("view_mapping")
        captured["x_bin_width"] = kwargs.get("x_bin_width")
        return None

    monkeypatch.setattr(
        "linak.analysis.position.plot_position_profiles", _fake_plot_position_profiles
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_h5_1),
            str(source_h5_2),
            "--species",
            "O",
            "--component",
            "xy-z",
            "--time-section-width",
            "0.5",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured["species"] == [
        f"{source_h5_1.name}:O",
        f"{source_h5_2.name}:O",
    ]
    assert captured["view_mapping"].view_type_id == "trajectory_2d"
    assert captured["view_mapping"].x == "x"
    assert captured["view_mapping"].y == "y"
    assert captured["view_mapping"].color == "distance_to_surface"
    assert captured["x_bin_width"] == pytest.approx(0.5)


def test_plot_position_rejects_trajectory_input(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    write(trajectory, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(trajectory),
            "--species",
            "O",
            "--no-show",
        ]
    )

    assert rc == 1
    assert "only accepts HDF5 input" in capsys.readouterr().err


def test_plot_rdf_handler_rejects_section_width_larger_than_available_range(tmp_path):
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    source_h5 = tmp_path / "source_rdf.h5"
    save_rdf_profile(
        compute_rdf([frame], species_a="O", species_b="H", r_max=2.0, bin_width=1.0),
        source_h5,
    )

    argv = [
        "plot",
        str(source_h5),
        "--species-a",
        "O",
        "--species-b",
        "H",
        "--no-show",
        "--output",
        str(tmp_path / "rdf.png"),
    ]
    args = build_parser().parse_args(argv)
    args._runtime_argv = tuple(argv)
    args.x_bin_width = 5.0
    args.x_bin_reducer = None
    args.min_bin_points = None

    with pytest.raises(ValueError, match="Section width 5 is larger than the available x-range"):
        args.handler(args)


def test_plot_position_gui_uses_atom_level_series_in_initial_settings(tmp_path, monkeypatch):
    source_h5 = tmp_path / "source_position.h5"
    _write_position_hdf5(source_h5)

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["analysis_name"] == "position"
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["series_count"] == 2
    assert [item["default_label"] for item in initial["series_descriptors"]] == ["O[2]", "O[3]"]
    assert initial["axis"] is None
    assert initial["component"] == "distance"
    assert initial["map_color"] == "distance"
    assert initial["time_axis"] == "ps"
    resolver = captured["on_resolve_series_defaults"]
    resolved = resolver(initial)
    assert resolved["series_count"] == 2
    assert resolved["series_labels"] == ["O[2]", "O[3]"]


def test_plot_position_gui_refuses_excessive_atom_series(tmp_path):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=200, n_frames=3)

    argv = ["plot", str(source_h5), "--gui"]
    args = build_parser().parse_args(argv)
    args._runtime_argv = tuple(argv)
    default_args = deepcopy(args)
    catalog = cli_mod._build_position_gui_lazy_catalog(args, sources=[str(source_h5)])
    catalog.default_series_labels = cli_mod._resolve_gui_default_series_labels(
        args=args,
        sources=[str(source_h5)],
        profile_key="plot:position",
        fallback_labels_by_source=catalog.fallback_labels_by_source,
    )
    initial_context = catalog.build_initial_context()

    with pytest.raises(ValueError, match="too large for interactive GUI controls"):
        cli_mod._launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=source_h5,
            profile_key="plot:position",
            setting_keys=cli_mod._PLOT_SETTINGS_POSITION_KEYS,
            gui_title="LiNaK Plot Controls: Position",
            analysis_name="position",
            plotter=lambda *_args, **_kwargs: None,
            initial_context=initial_context,
            build_context=lambda current_args: catalog.build_render_context(current_args),
            build_full_context=lambda current_args: catalog.build_initial_context(),
        )


def test_plot_position_gui_combined_sources_refuses_after_descriptor_expansion(tmp_path):
    source_h5_a = tmp_path / "large_a_position.h5"
    source_h5_b = tmp_path / "large_b_position.h5"
    _write_large_position_hdf5(source_h5_a, n_atoms=100, n_frames=3)
    _write_large_position_hdf5(source_h5_b, n_atoms=100, n_frames=3)

    argv = ["plot", "-f", str(source_h5_a), str(source_h5_b), "--gui"]
    args = build_parser().parse_args(argv)
    args._runtime_argv = tuple(argv)
    default_args = deepcopy(args)
    combined_source = cli_mod._combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="position",
        output=None,
    )
    catalog = cli_mod._build_position_gui_lazy_catalog(args, sources=[str(combined_source)])
    catalog.default_series_labels = cli_mod._resolve_gui_default_series_labels(
        args=args,
        sources=[str(combined_source)],
        profile_key="plot:position",
        fallback_labels_by_source=catalog.fallback_labels_by_source,
    )
    initial_context = catalog.build_initial_context()

    with pytest.raises(ValueError, match="200 series"):
        cli_mod._launch_profile_plot_gui(
            args=args,
            default_args=default_args,
            source_path=combined_source,
            profile_key="plot:position",
            setting_keys=cli_mod._PLOT_SETTINGS_POSITION_KEYS,
            gui_title="LiNaK Plot Controls: Position",
            analysis_name="position",
            plotter=lambda *_args, **_kwargs: None,
            initial_context=initial_context,
            build_context=lambda current_args: catalog.build_render_context(current_args),
            build_full_context=lambda current_args: catalog.build_initial_context(),
        )


def test_plot_position_gui_uses_saved_projection_filter_before_initial_guard(tmp_path, monkeypatch):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=200, n_frames=3)

    write_plot_profile(
        source_h5,
        "plot:position",
        _saved_plot_profile("plot:position", {
            "component": "2d-projection",
            "projection_render_mode": "color-scale",
            "projection_value": "distance",
            "projection_filter_max": 2.1,
        }),
    )

    args = cli_mod.build_parser().parse_args(["plot", str(source_h5), "--gui"])
    args._runtime_argv = ("plot", str(source_h5), "--gui")
    cli_mod._apply_saved_plot_settings(
        args=args,
        source_path=source_h5,
        profile_key="plot:position",
        keys=cli_mod._PLOT_SETTINGS_POSITION_KEYS,
        profile_name=None,
    )
    default_args = deepcopy(args)
    catalog = cli_mod._build_position_gui_lazy_catalog(args, sources=[str(source_h5)])
    catalog.default_series_labels = cli_mod._resolve_gui_default_series_labels(
        args=args,
        sources=[str(source_h5)],
        profile_key="plot:position",
        fallback_labels_by_source=catalog.fallback_labels_by_source,
    )
    initial_context = catalog.build_initial_context()

    monkeypatch.setattr("linak.cli._render_profile_plot", lambda **_kwargs: (None, {}))
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", lambda **_kwargs: None)

    cli_mod._launch_profile_plot_gui(
        args=args,
        default_args=default_args,
        source_path=source_h5,
        profile_key="plot:position",
        setting_keys=cli_mod._PLOT_SETTINGS_POSITION_KEYS,
        gui_title="LiNaK Plot Controls: Position",
        analysis_name="position",
        plotter=lambda *_args, **_kwargs: None,
        initial_context=initial_context,
        build_context=lambda current_args: catalog.build_render_context(current_args),
        build_full_context=lambda current_args: cli_mod._build_position_gui_lazy_catalog(
            current_args, sources=[str(source_h5)]
        ).build_initial_context(),
    )


def test_position_projection_lazy_catalog_uses_filtered_point_count_in_line_colors_mode(
    tmp_path, caplog
):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=80, n_frames=3)

    args = cli_mod.build_parser().parse_args(
        [
            "plot",
            str(source_h5),
            "--component",
            "2d-projection",
            "--projection-render-mode",
            "line-colors",
            "--projection-value",
            "distance",
            "--projection-filter-max",
            "2.1",
        ]
    )
    args._runtime_argv = (
        "plot",
        str(source_h5),
        "--component",
        "2d-projection",
        "--projection-render-mode",
        "line-colors",
        "--projection-value",
        "distance",
        "--projection-filter-max",
        "2.1",
    )

    caplog.set_level(logging.DEBUG, logger="linak.cli")
    catalog = cli_mod._build_position_gui_lazy_catalog(args, sources=[str(source_h5)])
    context = catalog.build_initial_context()

    assert context.series_count == 80
    assert context.estimated_total_points == 80
    assert "position GUI complexity at lazy_catalog" in caplog.text
    assert "raw_points=240" in caplog.text
    assert "final_points=80" in caplog.text


def test_position_projection_lazy_catalog_uses_filtered_point_count_in_color_scale_render_context(
    tmp_path, caplog
):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=120, n_frames=10_000)

    args = cli_mod.build_parser().parse_args(
        [
            "plot",
            str(source_h5),
            "--component",
            "2d-projection",
            "--projection-x",
            "x",
            "--projection-y",
            "y",
            "--projection-value",
            "distance",
            "--projection-render-mode",
            "color-scale",
            "--projection-filter-max",
            "2.0",
        ]
    )
    args._runtime_argv = (
        "plot",
        str(source_h5),
        "--component",
        "2d-projection",
        "--projection-x",
        "x",
        "--projection-y",
        "y",
        "--projection-value",
        "distance",
        "--projection-render-mode",
        "color-scale",
        "--projection-filter-max",
        "2.0",
    )

    caplog.set_level(logging.DEBUG, logger="linak.cli")
    catalog = cli_mod._build_position_gui_lazy_catalog(args, sources=[str(source_h5)])
    initial_context = catalog.build_initial_context()
    render_context = catalog.build_render_context(args)

    assert initial_context.series_count == 1
    assert initial_context.estimated_total_points == 120
    assert render_context.series_count == 1
    assert render_context.estimated_total_points == 120
    assert "position projection guard at lazy_catalog" in caplog.text
    assert "position projection guard at render_context" in caplog.text
    assert "filter_max=2.0" in caplog.text
    assert "raw_candidate_points=1200000" in caplog.text
    assert "final_visible_points=120" in caplog.text


def test_plot_position_gui_uses_cli_projection_filter_before_initial_preview_guard(
    tmp_path, monkeypatch
):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=120, n_frames=10_000)

    monkeypatch.setattr("linak.cli._render_profile_plot", lambda **_kwargs: (None, {}))
    monkeypatch.setattr("linak.cli._open_plot_settings_gui", lambda **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
            "--component",
            "2d-projection",
            "--projection-x",
            "x",
            "--projection-y",
            "y",
            "--projection-value",
            "distance",
            "--projection-render-mode",
            "color-scale",
            "--projection-filter-max",
            "2.0",
        ]
    )

    assert rc == 0


def test_plot_position_gui_force_gui_bypasses_complexity_guard(tmp_path, monkeypatch):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=200, n_frames=3)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", lambda **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
            "--force-gui",
        ]
    )

    assert rc == 0


def test_plot_position_non_gui_warns_for_excessive_complexity(tmp_path, monkeypatch, caplog):
    source_h5 = tmp_path / "large_position.h5"
    _write_large_position_hdf5(source_h5, n_atoms=200, n_frames=3)

    monkeypatch.setattr("linak.cli._render_profile_plot", lambda **_kwargs: (None, {}))

    argv = ["plot", str(source_h5), "--no-show"]
    args = build_parser().parse_args(argv)
    args._runtime_argv = tuple(argv)

    caplog.set_level(logging.WARNING, logger="linak.cli")
    rc = args.handler(args)

    assert rc == 0
    assert "Position plot is too large for non-GUI plotting" in caplog.text
    assert "Proceeding anyway" in caplog.text


def test_plot_coordination_multiple_files_overlays_with_source_labels(tmp_path, monkeypatch):
    source_h5_1 = tmp_path / "source1_coordination.h5"
    source_h5_2 = tmp_path / "source2_coordination.h5"
    _write_coordination_hdf5(source_h5_1)
    _write_coordination_hdf5(source_h5_2)

    captured: dict[str, object] = {}

    def _fake_plot_coordination_profiles(profiles, **kwargs):
        captured["species_a"] = [item.species_a for item in profiles]
        captured["component"] = kwargs.get("component")
        return None

    monkeypatch.setattr(
        "linak.analysis.coordination.plot_coordination_profiles",
        _fake_plot_coordination_profiles,
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_h5_1),
            str(source_h5_2),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured["species_a"] == [f"{source_h5_1.name}:O", f"{source_h5_2.name}:O"]
    assert captured["component"] == "distance"


def test_plot_coordination_gui_defaults_to_distance_and_resolves_time_series(tmp_path, monkeypatch):
    source_h5 = tmp_path / "source_coordination.h5"
    _write_coordination_hdf5(source_h5)

    captured: dict[str, object] = {}

    def _fake_gui_launcher(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("linak.cli._open_plot_settings_gui", _fake_gui_launcher)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["analysis_name"] == "coordination"
    initial = captured["initial_settings"]
    assert isinstance(initial, dict)
    assert initial["view_mapping"]["view_type_id"] == "line_1d"
    assert initial["view_mapping"]["x"] == "distance_to_surface"
    assert initial["view_mapping"]["y"] == "coordination_number"
    assert initial["series_count"] == 1
    assert [item["default_label"] for item in initial["series_descriptors"]] == ["O-H"]
    resolver = captured["on_resolve_series_defaults"]
    resolved = resolver(
        {
            **initial,
            "view_mapping": {
                "view_type_id": "line_1d",
                "x": "time_ps",
                "y": "coordination_number",
                "color": None,
                "split_by": "atom",
                "filter_by": None,
                "filter_min": None,
                "filter_max": None,
                "role_assignments": {},
                "fixed_values": {"legacy_component": "time"},
            },
        }
    )
    assert resolved["series_count"] == 1
    assert resolved["series_labels"] == ["O[2]"]
    resolved_time_distance = resolver(
        {
            **initial,
            "view_mapping": {
                "view_type_id": "trajectory_2d",
                "x": "time_ps",
                "y": "distance_to_surface",
                "color": "coordination_number",
                "split_by": "atom",
                "filter_by": None,
                "filter_min": None,
                "filter_max": None,
                "role_assignments": {},
                "fixed_values": {"legacy_component": "time-distance"},
            },
        }
    )
    assert resolved_time_distance["series_count"] == 1
    assert resolved_time_distance["series_labels"] == ["O[2]"]


def test_plot_coordination_time_distance_writes_output(tmp_path):
    source_h5 = tmp_path / "source_coordination.h5"
    output = tmp_path / "coordination_time_distance.png"
    _write_coordination_hdf5(source_h5)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source_h5),
            "--component",
            "time-distance",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_plot_rdf_multiple_files_overlays_with_source_labels(tmp_path, monkeypatch):
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_rdf(
        [frame],
        species_a="O",
        species_b="H",
        r_max=2.0,
        bin_width=1.0,
    )
    source_csv_1 = tmp_path / "source1_rdf.h5"
    source_csv_2 = tmp_path / "source2_rdf.h5"
    save_rdf_profile(profile, source_csv_1)
    save_rdf_profile(profile, source_csv_2)

    captured_labels: list[str] = []

    def _fake_plot_rdf_profiles(profiles, **_kwargs):
        captured_labels.extend([f"{item.species_a}-{item.species_b}" for item in profiles])
        return None

    monkeypatch.setattr("linak.analysis.rdf.plot_rdf_profiles", _fake_plot_rdf_profiles)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_csv_1),
            str(source_csv_2),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured_labels == [f"{source_csv_1.name}:O-H", f"{source_csv_2.name}:O-H"]


def test_plot_rdf_rejects_trajectory_input(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    write(trajectory, [Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--no-show",
        ]
    )

    assert rc == 1
    assert "only accepts HDF5 input" in capsys.readouterr().err


def test_compute_rdf_threads_option_is_forwarded(tmp_path, monkeypatch):
    from linak.analysis.rdf import RDFProfile

    captured: dict[str, int | None] = {}
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 1.0]),
        n_frames=1,
    )

    def _fake_read_trajectory(_path):
        return [frame]

    def _fake_compute_rdf(*, threads, **_kwargs):
        captured["threads"] = threads
        return profile

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.rdf.compute_rdf", _fake_compute_rdf)
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profiles", lambda *_args, **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(tmp_path / "traj.xyz"),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--threads",
            "2",
            "--output",
            str(tmp_path / "rdf.h5"),
        ]
    )

    assert rc == 0
    assert captured["threads"] == 2


def test_compute_rdf_atom_selectors_are_forwarded(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    frame = Atoms(
        "OHHHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    profile = RDFProfile(
        species_a="atoms[0,2..3,5..6]",
        species_b="atoms[1]",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 1.0]),
        n_frames=1,
        atom_indices_a=np.array([0, 2, 3, 5, 6], dtype=int),
        atom_indices_b=np.array([1], dtype=int),
        selection_kind_a="atoms",
        selection_kind_b="atoms",
    )

    def _fake_read_trajectory(_path):
        return [frame]

    def _fake_compute_rdf(**kwargs):
        captured.update(kwargs)
        return profile

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.rdf.compute_rdf", _fake_compute_rdf)
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profiles", lambda *_args, **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(tmp_path / "traj.xyz"),
            "--atoms-a",
            "0",
            "2..3",
            "{5,6}",
            "--atoms-b",
            "1",
            "--output",
            str(tmp_path / "rdf_selected.h5"),
        ]
    )

    assert rc == 0
    assert captured["species_a"] is None
    assert captured["species_b"] is None
    assert captured["atom_indices_a"] == (0, 2, 3, 5, 6)
    assert captured["atom_indices_b"] == (1,)


def test_compute_rdf_mixed_species_and_atom_selectors_are_forwarded(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    frame = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    profile = RDFProfile(
        species_a="O",
        species_b="atoms[1..2]",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 1.0]),
        n_frames=1,
        atom_indices_b=np.array([1, 2], dtype=int),
        selection_kind_a="species",
        selection_kind_b="atoms",
    )

    def _fake_read_trajectory(_path):
        return [frame]

    def _fake_compute_rdf(**kwargs):
        captured.update(kwargs)
        return profile

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.rdf.compute_rdf", _fake_compute_rdf)
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profiles", lambda *_args, **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(tmp_path / "traj.xyz"),
            "--species-a",
            "O",
            "--atoms-b",
            "1",
            "2",
            "--output",
            str(tmp_path / "rdf_selected.h5"),
        ]
    )

    assert rc == 0
    assert captured["species_a"] == "O"
    assert captured["species_b"] is None
    assert captured["atom_indices_a"] is None
    assert captured["atom_indices_b"] == (1, 2)


def test_compute_rdf_without_explicit_species_writes_pairwise_collection(tmp_path, monkeypatch):
    trajectory_dir = tmp_path / "traj_dir"
    work_dir = tmp_path / "work_dir"
    trajectory_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    trajectory = trajectory_dir / "traj.xyz"
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    write(trajectory, [frame], format="extxyz")
    (trajectory_dir / "input.inp").write_text("ABC [angstrom] 10.0 10.0 10.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
        ]
    )

    assert rc == 0
    assert _default_rdf_collection_hdf5_output_path(trajectory).exists()


def test_compute_rdf_explicit_atom_selection_uses_canonical_default_output(tmp_path, monkeypatch):
    captured_output: dict[str, Path] = {}
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    profile = RDFProfile(
        species_a="atoms[0]",
        species_b="atoms[1]",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 1.0]),
        n_frames=1,
        atom_indices_a=np.array([0], dtype=int),
        atom_indices_b=np.array([1], dtype=int),
        selection_kind_a="atoms",
        selection_kind_b="atoms",
    )

    def _fake_read_trajectory(_path):
        return [frame]

    def _fake_save_rdf_profiles(_profiles, output, **_kwargs):
        captured_output["path"] = Path(output)
        return Path(output)

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.rdf.compute_rdf", lambda **_kwargs: profile)
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profiles", _fake_save_rdf_profiles)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
            "--atoms-a",
            "0",
            "--atoms-b",
            "1",
        ]
    )

    assert rc == 0
    assert captured_output["path"] == _default_rdf_collection_hdf5_output_path(trajectory)


def test_compute_rdf_explicit_all_keeps_single_profile_path(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = RDFProfile(
        species_a="ALL",
        species_b="ALL",
        bin_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array([0.5, 1.5]),
        g_r=np.array([0.0, 1.0]),
        n_frames=1,
    )

    def _fake_read_trajectory(_path):
        return [frame]

    def _fake_compute_rdf(**kwargs):
        captured["species_a"] = kwargs["species_a"]
        captured["species_b"] = kwargs["species_b"]
        return profile

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.rdf.compute_rdf", _fake_compute_rdf)
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profiles", lambda *_args, **_kwargs: None)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(tmp_path / "traj.xyz"),
            "--species-a",
            "all",
            "--species-b",
            "all",
            "--output",
            str(tmp_path / "rdf_all.h5"),
        ]
    )

    assert rc == 0
    assert captured == {"species_a": "all", "species_b": "all"}


def test_compute_rdf_rejects_selector_b_without_explicit_selector_a(tmp_path, capsys):
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(tmp_path / "traj.xyz"),
            "--atoms-b",
            "1",
        ]
    )

    assert rc == 1
    assert "selector A" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["compute", "rdf", "traj.xyz", "--atoms-a", "-1", "--dry-run"],
            "Atom indices must be >= 0",
        ),
        (
            ["compute", "rdf", "traj.xyz", "--atoms-a", "2..1", "--dry-run"],
            "range end must be >=",
        ),
        (
            ["compute", "rdf", "traj.xyz", "--atoms-a", "foo", "--dry-run"],
            "Malformed atom index",
        ),
    ],
)
def test_compute_rdf_rejects_invalid_atom_selector_tokens(argv, message, capsys):
    rc = main(["--log-level", "ERROR", *argv])

    assert rc == 1
    assert message in capsys.readouterr().err


def test_compute_rdf_rejects_out_of_range_atom_indices(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    write(
        trajectory,
        [
            Atoms(
                "OH",
                positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
        ],
        format="extxyz",
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
            "--atoms-a",
            "5",
        ]
    )

    assert rc == 1
    assert "out-of-range indices" in capsys.readouterr().err


def test_compute_rdf_dry_run_reports_atom_selectors(capsys):
    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "rdf",
            "traj.xyz",
            "--atoms-a",
            "0",
            "2..3",
            "--atoms-b",
            "5",
            "--dry-run",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "selector_a=atoms[0,2..3]" in err
    assert "selector_b=atoms[5]" in err


def test_compute_rdf_with_species_b_only_writes_filtered_collection_hdf5(tmp_path, monkeypatch):
    trajectory = tmp_path / "traj.xyz"
    frames = [
        Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
    ]
    trajectory.write_text("", encoding="utf-8")
    monkeypatch.setattr("linak.trajectory.io.read_trajectory", lambda _path: frames)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
            "--species-b",
            "H",
            "--cell",
            "10",
            "10",
            "10",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_rdf.h5"
    assert output.exists()
    profiles = load_rdf_profiles(output)
    assert {(profile.species_a, profile.species_b) for profile in profiles} == {
        ("H", "H"),
        ("H", "O"),
    }


def test_compute_rdf_preflights_output_before_loading_trajectory(tmp_path, monkeypatch):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    blocked_parent = tmp_path / "blocked_parent"
    blocked_parent.write_text("file", encoding="utf-8")

    def _unexpected_read(_path):
        raise AssertionError("trajectory should not be loaded when output preflight fails")

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _unexpected_read)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--output",
            str(blocked_parent / "rdf.h5"),
        ]
    )

    assert rc == 1


def test_compute_coordination_with_cutoff_rdf_writes_hdf5_and_diagnostic_png(tmp_path, monkeypatch):
    from linak.analysis.rdf import RDFProfile

    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    rdf_path = tmp_path / "cutoff_rdf.h5"
    save_rdf_profile(
        RDFProfile(
            species_a="O",
            species_b="H",
            bin_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float),
            bin_centers=np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75], dtype=float),
            g_r=np.array([0.2, 1.8, 1.2, 0.35, 0.55, 0.9], dtype=float),
            n_frames=10,
        ),
        rdf_path,
    )

    frames = [
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.20],
                [1.0, 0.0, 0.20],
                [0.0, 0.0, 1.00],
                [0.70, 0.0, 1.00],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.30],
                [1.0, 0.0, 0.30],
                [0.0, 0.0, 1.20],
                [1.10, 0.0, 1.20],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", lambda _path: frames)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "coordination",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--axis",
            "z",
            "--surface-mode",
            "rough",
            "--surface-elements",
            "Pt",
            "--timestep-fs",
            "2.0",
            "--cutoff-rdf",
            str(rdf_path),
            "--cell",
            "10",
            "10",
            "10",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(tmp_path) / "traj_coordination_o_h.h5").exists()
    assert (_linak_output_dir(tmp_path) / "traj_coordination_o_h_cutoff_rdf.png").exists()


def test_compute_coordination_defaults_to_cutoff_from_rdf_when_unspecified(tmp_path, monkeypatch):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")

    frames = [
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.20],
                [1.0, 0.0, 0.20],
                [0.0, 0.0, 1.00],
                [0.70, 0.0, 1.00],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.30],
                [1.0, 0.0, 0.30],
                [0.0, 0.0, 1.20],
                [1.10, 0.0, 1.20],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", lambda _path: frames)

    def _fake_resolve_coordination_cutoff(**kwargs):
        captured["cutoff_A"] = kwargs["cutoff_A"]
        captured["cutoff_rdf_path"] = kwargs["cutoff_rdf_path"]
        captured["cutoff_from_rdf"] = kwargs["cutoff_from_rdf"]
        diagnostic_path = kwargs.get("diagnostic_plot_output")
        if diagnostic_path is not None:
            Path(diagnostic_path).parent.mkdir(parents=True, exist_ok=True)
            Path(diagnostic_path).write_text("fake png", encoding="utf-8")
        return CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.4,
            mode="full_rdf",
        )

    monkeypatch.setattr(
        "linak.analysis.coordination.resolve_coordination_cutoff",
        _fake_resolve_coordination_cutoff,
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "coordination",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--axis",
            "z",
            "--surface-mode",
            "rough",
            "--surface-elements",
            "Pt",
            "--timestep-fs",
            "2.0",
            "--cell",
            "10",
            "10",
            "10",
        ]
    )

    assert rc == 0
    assert captured == {
        "cutoff_A": None,
        "cutoff_rdf_path": None,
        "cutoff_from_rdf": True,
    }
    assert (_linak_output_dir(tmp_path) / "traj_coordination_o_h.h5").exists()
    assert (_linak_output_dir(tmp_path) / "traj_coordination_o_h_cutoff_rdf.png").exists()


def test_compute_coordination_without_explicit_species_writes_collection_hdf5(
    tmp_path, monkeypatch
):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")

    frames = [
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.20],
                [1.0, 0.0, 0.20],
                [0.0, 0.0, 1.00],
                [0.70, 0.0, 1.00],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.30],
                [1.0, 0.0, 0.30],
                [0.0, 0.0, 1.20],
                [1.10, 0.0, 1.20],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", lambda _path: frames)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "coordination",
            str(trajectory),
            "--axis",
            "z",
            "--surface-mode",
            "rough",
            "--surface-elements",
            "Pt",
            "--timestep-fs",
            "2.0",
            "--cell",
            "10",
            "10",
            "10",
            "--cutoff",
            "1.0",
        ]
    )

    assert rc == 2


def test_build_parser_omits_coordination_rdf_tuning_flags():
    parser = build_parser()

    args = parser.parse_args(
        ["compute", "coordination", "traj.xyz", "--species-a", "O", "--species-b", "H"]
    )

    assert not hasattr(args, "rdf_sample_fraction")
    assert not hasattr(args, "rdf_bin_width")
    assert not hasattr(args, "rdf_r_max")
    assert not hasattr(args, "rdf_smoothing_sigma")


def test_compute_coordination_requires_at_least_one_species_selector(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(["--log-level", "ERROR", "compute", "coordination", str(trajectory)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Provide at least one coordination selector" in err


def test_compute_coordination_with_species_a_only_writes_collection_hdf5(tmp_path, monkeypatch):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    frames = [
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.20],
                [1.0, 0.0, 0.20],
                [0.0, 0.0, 1.00],
                [0.70, 0.0, 1.00],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
    ]
    monkeypatch.setattr("linak.trajectory.io.read_trajectory", lambda _path: frames)

    captured_pairs: dict[str, object] = {}

    def _fake_resolve_coordination_cutoffs(**kwargs):
        ordered_pairs = list(kwargs["ordered_pairs"])
        captured_pairs["ordered_pairs"] = ordered_pairs
        return {
            pair: CoordinationCutoffResolution(
                cutoff_A=1.0,
                smoothing_width_A=0.4,
                mode="direct",
            )
            for pair in ordered_pairs
        }

    monkeypatch.setattr(
        "linak.analysis.coordination.resolve_coordination_cutoffs",
        _fake_resolve_coordination_cutoffs,
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "coordination",
            str(trajectory),
            "--species-a",
            "O",
            "--axis",
            "z",
            "--surface-mode",
            "rough",
            "--surface-elements",
            "Pt",
            "--timestep-fs",
            "2.0",
            "--cell",
            "10",
            "10",
            "10",
            "--cutoff",
            "1.0",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_coordination.h5"
    assert output.exists()
    assert captured_pairs["ordered_pairs"] == [("O", "H"), ("O", "O"), ("O", "Pt")]
    profiles = load_coordination_profiles(output)
    assert {(profile.species_a, profile.species_b) for profile in profiles} == {
        ("O", "H"),
        ("O", "O"),
        ("O", "Pt"),
    }


def test_compute_coordination_preflights_missing_cutoff_rdf_before_loading_trajectory(
    tmp_path, monkeypatch
):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")

    def _unexpected_read(_path):
        raise AssertionError("trajectory should not be loaded when cutoff RDF preflight fails")

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _unexpected_read)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "coordination",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--cutoff-rdf",
            str(tmp_path / "missing_cutoff.h5"),
        ]
    )

    assert rc == 1


def test_compute_density_writes_default_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(tmp_path) / "traj_density_o.h5").exists()


def test_compute_density_accepts_single_source_via_files_option(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            "-f",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(tmp_path) / "traj_density_o.h5").exists()


def test_compute_density_default_hdf5_uses_linak_output_dir_in_source_folder(tmp_path, monkeypatch):
    trajectory_dir = tmp_path / "traj_dir"
    work_dir = tmp_path / "work_dir"
    trajectory_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    trajectory = trajectory_dir / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(trajectory_dir) / "traj_density_o.h5").exists()
    assert not (work_dir / "traj_density_o.h5").exists()


def test_compute_density_output_trailing_slash_uses_directory_with_default_filename(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    output_dir = tmp_path / "custom_output"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
            "--output",
            f"{output_dir.as_posix()}/",
        ]
    )

    assert rc == 0
    assert (output_dir / "traj_density_o.h5").exists()
    assert not (tmp_path / "custom_output.h5").exists()


def test_compute_density_output_without_suffix_stays_file_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    output_base = tmp_path / "custom_output"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
            "--output",
            str(output_base),
        ]
    )

    assert rc == 0
    assert (tmp_path / "custom_output.h5").exists()
    assert not (output_base / "traj_density_o.h5").exists()


def test_compute_msd_default_hdf5_uses_linak_output_dir_in_source_folder(tmp_path, monkeypatch):
    trajectory_dir = tmp_path / "traj_dir"
    work_dir = tmp_path / "work_dir"
    trajectory_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    trajectory = trajectory_dir / "traj.xyz"
    frame0 = Atoms("O", positions=[[0.9, 0.0, 0.0]])
    frame1 = Atoms("O", positions=[[0.1, 0.0, 0.0]])
    write(trajectory, [frame0, frame1], format="extxyz")
    (trajectory_dir / "input.inp").write_text("ABC [angstrom] 1.0 1.0 1.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "msd",
            str(trajectory),
            "--species",
            "O",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(trajectory_dir) / "traj_msd_o.h5").exists()
    assert not (work_dir / "traj_msd_o.h5").exists()


def test_compute_position_default_hdf5_uses_linak_output_dir_in_source_folder(
    tmp_path, monkeypatch
):
    trajectory_dir = tmp_path / "traj_dir"
    work_dir = tmp_path / "work_dir"
    trajectory_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    trajectory = trajectory_dir / "traj.xyz"
    frame0 = Atoms("O", positions=[[0.9, 0.0, 0.0]])
    frame1 = Atoms("O", positions=[[1.1, 0.0, 0.0]])
    write(trajectory, [frame0, frame1], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "position",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(trajectory_dir) / "traj_position_o_z.h5").exists()
    assert not (work_dir / "traj_position_o_z.h5").exists()


def test_compute_position_pbc_corrects_hdf5_positions_without_modifying_source_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj_pbc.xyz"
    frame0 = Atoms(
        "O",
        positions=[[1.2, -0.1, 0.5]],
        cell=[1.0, 1.0, 1.0],
        pbc=True,
    )
    frame1 = Atoms(
        "O",
        positions=[[1.3, -0.2, 0.6]],
        cell=[1.0, 1.0, 1.0],
        pbc=True,
    )
    write(trajectory, [frame0, frame1], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "position",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_pbc_position_o_z.h5"
    profile = load_position_profile(output, species="O", axis="z")
    np.testing.assert_allclose(profile.x[:, 0], np.array([0.2, 0.3]), atol=1e-12)
    np.testing.assert_allclose(profile.y[:, 0], np.array([0.9, 0.8]), atol=1e-12)
    np.testing.assert_allclose(profile.z[:, 0], np.array([0.5, 0.6]), atol=1e-12)

    _datasets, metadata = read_linak_hdf5(output, expected_analysis="position")
    assert metadata["positions_pbc_corrected"] is True
    assert metadata["pbc_cell_angstrom"] == pytest.approx([1.0, 1.0, 1.0])

    original = read(trajectory, index=":")
    assert len(original) == 2
    assert original[0].positions[0, 0] == pytest.approx(1.2)
    assert original[0].positions[0, 1] == pytest.approx(-0.1)
    assert original[1].positions[0, 0] == pytest.approx(1.3)
    assert original[1].positions[0, 1] == pytest.approx(-0.2)


def test_compute_position_without_species_warns_and_writes_per_species_files(tmp_path, capsys):
    trajectory = tmp_path / "mixed.xyz"
    frame0 = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.2], [1.0, 0.0, 0.8]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.3], [1.0, 0.0, 0.9]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    write(trajectory, [frame0, frame1], format="extxyz")

    rc = main(
        [
            "--log-level",
            "WARNING",
            "compute",
            "position",
            str(trajectory),
            "--axis",
            "z",
        ]
    )

    assert rc == 0
    expected_h = _linak_output_dir(tmp_path) / "mixed_position_h_z.h5"
    expected_o = _linak_output_dir(tmp_path) / "mixed_position_o_z.h5"
    assert expected_h.exists()
    assert expected_o.exists()
    for output_path in (expected_h, expected_o):
        datasets, metadata = read_linak_hdf5(output_path, expected_analysis="position")
        assert metadata["species"] in {"H", "O"}
        assert datasets["x_A"].shape == (2, 1)
        assert datasets["y_A"].shape == (2, 1)
        assert datasets["z_A"].shape == (2, 1)
        assert datasets["distance_to_surface_A"].shape == (2, 1)
    assert "No --species provided for position analysis" in capsys.readouterr().err


def test_compute_rdf_default_hdf5_uses_linak_output_dir_in_source_folder(tmp_path, monkeypatch):
    trajectory_dir = tmp_path / "traj_dir"
    work_dir = tmp_path / "work_dir"
    trajectory_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    trajectory = trajectory_dir / "traj.xyz"
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
    )
    write(trajectory, [frame], format="extxyz")
    (trajectory_dir / "input.inp").write_text("ABC [angstrom] 10.0 10.0 10.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(trajectory_dir) / "traj_rdf.h5").exists()
    assert not (work_dir / "traj_rdf.h5").exists()


def test_compute_density_default_output_avoids_overwriting_existing_hdf5(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    first_rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )
    second_rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )

    assert first_rc == 0
    assert second_rc == 0
    assert (_linak_output_dir(tmp_path) / "traj_density_o.h5").exists()
    assert (_linak_output_dir(tmp_path) / "traj_density_o_1.h5").exists()


def test_default_combined_analysis_hdf5_path_uses_pwd_linak_output_dir_for_multi_source(
    tmp_path, monkeypatch
):
    source_a = tmp_path / "run_a_density.h5"
    source_b = tmp_path / "run_b_density.h5"
    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    output = cli_mod._default_combined_analysis_hdf5_path(
        [str(source_a), str(source_b)],
        analysis="density",
    )

    assert output == _linak_output_dir(working_dir) / "linak_density_combined.h5"


def test_default_csv_output_path_uses_shared_linak_output_dir_without_nesting(tmp_path):
    source = _linak_output_dir(tmp_path) / "table_input.h5"

    output = cli_mod._default_csv_output_path(source, "sorted")

    assert output == _linak_output_dir(tmp_path) / "table_input_sorted.h5"


def test_compute_density_auto_detects_cell_for_volumetric_units(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    (tmp_path / "input.inp").write_text("ABC [angstrom] 10.0 10.0 10.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    profile = load_density_profile(_linak_output_dir(tmp_path) / "traj_density_o.h5")
    assert profile.units == "g/cm^3"


def test_compute_density_writes_resolution_metadata_to_hdf5(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    simulation_input = tmp_path / "input.inp"
    simulation_input.write_text("ABC [angstrom] 10.0 10.0 10.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
        ]
    )

    assert rc == 0
    _datasets, metadata = read_linak_hdf5(
        _linak_output_dir(tmp_path) / "traj_density_o.h5",
        expected_analysis="density",
    )
    assert metadata["source_path"] == str(trajectory.resolve())
    assert metadata["cell_source"].startswith("auto-detected")
    assert metadata["input_path"] == str(simulation_input.resolve())
    assert metadata["resolved_cell_angstrom"] == pytest.approx([10.0, 10.0, 10.0])


def test_compute_density_supports_save_data_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    custom_output = tmp_path / "custom_density.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--save-data",
            str(custom_output),
        ]
    )

    assert rc == 0
    assert custom_output.exists()


def test_compute_density_accepts_cp2k_input_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)
    cp2k_input = tmp_path / "cell.inp"
    cp2k_input.write_text("ABC [angstrom] 10.0 10.0 10.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "O",
            "--cp2k-input",
            str(cp2k_input),
        ]
    )

    assert rc == 0
    profile = load_density_profile(_linak_output_dir(tmp_path) / "traj_density_o.h5")
    assert profile.units == "g/cm^3"


def test_compute_density_all_writes_one_hdf5_with_all_species_series(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    frame = Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.10],
            [0.8, 0.0, 0.10],
            [-0.4, 0.7, 0.10],
        ],
    )
    write(trajectory, [frame], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "all",
            "--axis",
            "z",
            "--bin-width",
            "0.1",
        ]
    )

    assert rc == 0
    output = _linak_output_dir(tmp_path) / "traj_density.h5"
    assert output.exists()
    profiles = load_density_profiles(output)
    species_set = {profile.species for profile in profiles}
    raw_axes_set = {profile.axis for profile in profiles if profile.coordinate_mode != "distance"}
    assert species_set == {"H", "O", "H2O"}
    assert raw_axes_set == {"x", "y", "z"}
    distance_profiles = [p for p in profiles if p.coordinate_mode == "distance"]
    assert len(distance_profiles) >= 1


def test_compute_density_h2o_stays_single_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "water.xyz"
    frame = Atoms(
        "OHHOHH",
        positions=[
            [0.0, 0.0, 0.10],
            [0.95, 0.0, 0.10],
            [-0.30, 0.90, 0.10],
            [0.0, 0.0, 1.10],
            [0.95, 0.0, 1.10],
            [-0.30, 0.90, 1.10],
        ],
    )
    write(trajectory, [frame], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "density",
            str(trajectory),
            "--species",
            "H2O",
            "--axis",
            "z",
            "--bin-width",
            "0.5",
        ]
    )

    assert rc == 0
    assert (_linak_output_dir(tmp_path) / "water_density_h2o.h5").exists()
    assert not (_linak_output_dir(tmp_path) / "water_density_h.h5").exists()
    assert not (_linak_output_dir(tmp_path) / "water_density_o.h5").exists()


def test_apply_pbc_with_explicit_cell(tmp_path):
    trajectory = tmp_path / "in.xyz"
    out = tmp_path / "in_pbc.xyz"
    frame = Atoms("H", positions=[[1.2, -0.1, 0.5]])
    write(trajectory, [frame], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "pbc",
            str(trajectory),
            "--cell",
            "1.0",
            "1.0",
            "1.0",
        ]
    )

    assert rc == 0
    wrapped = read(out)
    assert wrapped.positions[0, 0] == pytest.approx(0.2)
    assert wrapped.positions[0, 1] == pytest.approx(0.9)
    assert wrapped.positions[0, 2] == pytest.approx(0.5)


def test_apply_pbc_accepts_single_source_via_files_option(tmp_path):
    trajectory = tmp_path / "in.xyz"
    out = tmp_path / "in_pbc.xyz"
    frame = Atoms("H", positions=[[1.2, -0.1, 0.5]])
    write(trajectory, [frame], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "pbc",
            "-f",
            str(trajectory),
            "--cell",
            "1.0",
            "1.0",
            "1.0",
        ]
    )

    assert rc == 0
    wrapped = read(out)
    assert wrapped.positions[0, 0] == pytest.approx(0.2)
    assert wrapped.positions[0, 1] == pytest.approx(0.9)
    assert wrapped.positions[0, 2] == pytest.approx(0.5)


def test_apply_pbc_auto_detects_cp2k_input_in_output_dir(tmp_path):
    trajectory = tmp_path / "in.xyz"
    out = tmp_path / "results" / "out.xyz"
    cp2k_input = out.parent / "input.inp"

    write(trajectory, [Atoms("H", positions=[[1.2, -0.1, 0.5]])], format="extxyz")
    out.parent.mkdir(parents=True, exist_ok=True)
    cp2k_input.write_text(
        "&CELL\n  ABC [angstrom] 2.0 3.0 4.0\n&END CELL\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "pbc",
            str(trajectory),
            "--output",
            str(out),
        ]
    )

    assert rc == 0
    wrapped = read(out)
    assert wrapped.cell.lengths()[0] == pytest.approx(2.0)
    assert wrapped.cell.lengths()[1] == pytest.approx(3.0)
    assert wrapped.cell.lengths()[2] == pytest.approx(4.0)


def test_apply_pbc_overwrite_replaces_input_file(tmp_path):
    trajectory = tmp_path / "in.xyz"
    write(trajectory, [Atoms("H", positions=[[1.2, -0.1, 0.5]])], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "pbc",
            str(trajectory),
            "--overwrite",
            "--cell",
            "1.0",
            "1.0",
            "1.0",
        ]
    )

    assert rc == 0
    wrapped = read(trajectory)
    assert wrapped.positions[0, 0] == pytest.approx(0.2)
    assert wrapped.positions[0, 1] == pytest.approx(0.9)
    assert wrapped.positions[0, 2] == pytest.approx(0.5)


def test_compute_msd_writes_resolution_metadata_to_hdf5(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    frame0 = Atoms("O", positions=[[0.9, 0.0, 0.0]])
    frame1 = Atoms("O", positions=[[0.1, 0.0, 0.0]])
    write(trajectory, [frame0, frame1], format="extxyz")
    (tmp_path / "input.inp").write_text(
        "ABC [angstrom] 1.0 1.0 1.0\n"
        "TIMESTEP [fs] 0.5\n"
        "&TRAJECTORY\n"
        "  &EACH\n"
        "    MD 5\n"
        "  &END EACH\n"
        "&END TRAJECTORY\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "msd",
            str(trajectory),
            "--species",
            "O",
        ]
    )
    assert rc == 0
    profile = load_msd_profile(_linak_output_dir(tmp_path) / "traj_msd_o.h5")
    assert profile.time_fs[1] == pytest.approx(2.5)

    _datasets, metadata = read_linak_hdf5(
        _linak_output_dir(tmp_path) / "traj_msd_o.h5",
        expected_analysis="msd",
    )
    assert metadata["source_path"] == str(trajectory.resolve())
    assert metadata["timestep_source"].startswith("auto-detected")
    assert metadata["frame_timestep_fs"] == pytest.approx(2.5)
    assert metadata["resolved_cell_angstrom"] == pytest.approx([1.0, 1.0, 1.0])


def test_compute_msd_resolves_sidecars_before_loading_trajectory(tmp_path, monkeypatch):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    events: list[str] = []

    frames = [
        Atoms(
            "O",
            positions=[[0.9, 0.0, 0.0]],
            cell=[1.0, 1.0, 1.0],
            pbc=True,
            info={"timestep_fs": 2.0},
        ),
        Atoms(
            "O",
            positions=[[0.1, 0.0, 0.0]],
            cell=[1.0, 1.0, 1.0],
            pbc=True,
            info={"timestep_fs": 2.0},
        ),
    ]

    def _fake_find_unique_simulation_input(_search_dir):
        events.append("input_lookup")
        raise FileNotFoundError("no simulation input")

    def _fake_read_trajectory(_path):
        events.append("read_trajectory")
        return frames

    monkeypatch.setattr(
        "linak.resolution.find_unique_simulation_input", _fake_find_unique_simulation_input
    )
    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "msd",
            str(trajectory),
            "--species",
            "O",
        ]
    )

    assert rc == 0
    assert events[:3] == ["input_lookup", "input_lookup", "read_trajectory"]


def test_compute_msd_from_lammps_dump_with_lmp_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "lammps.dump"
    lammps_input = tmp_path / "input.lmp"
    _write_lammps_dump(
        trajectory,
        positions=[
            (0.9, 0.0, 0.0),
            (0.1, 0.0, 0.0),
        ],
    )
    _write_lammps_input(lammps_input, dump_name=trajectory.name, dump_every=10)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "msd",
            str(trajectory),
            "--species",
            "O",
            "--input",
            str(lammps_input),
            "--output",
            str(tmp_path / "msd_from_dump.h5"),
        ]
    )

    assert rc == 0
    data = load_msd_profile(tmp_path / "msd_from_dump.h5")
    assert data.time_fs[1] == pytest.approx(10.0)


def test_compute_msd_from_lammps_input_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "lammps.dump"
    lammps_input = tmp_path / "input.lmp"
    _write_lammps_dump(
        trajectory,
        positions=[
            (0.9, 0.0, 0.0),
            (0.1, 0.0, 0.0),
        ],
    )
    _write_lammps_input(lammps_input, dump_name=trajectory.name, dump_every=10)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "msd",
            str(lammps_input),
            "--species",
            "O",
            "--output",
            str(tmp_path / "msd_from_lmp.h5"),
        ]
    )

    assert rc == 0
    data = load_msd_profile(tmp_path / "msd_from_lmp.h5")
    assert data.time_fs[1] == pytest.approx(10.0)


def test_compute_rdf_writes_resolution_metadata_to_hdf5(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectory = tmp_path / "traj.xyz"
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
    )
    write(trajectory, [frame], format="extxyz")
    simulation_input = tmp_path / "input.inp"
    simulation_input.write_text("ABC [angstrom] 10.0 10.0 10.0\n", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "rdf",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--bin-width",
            "0.2",
            "--output",
            str(tmp_path / "rdf_metadata.h5"),
        ]
    )
    assert rc == 0

    _datasets, metadata = read_linak_hdf5(tmp_path / "rdf_metadata.h5", expected_analysis="rdf")
    assert metadata["source_path"] == str(trajectory.resolve())
    assert metadata["cell_source"].startswith("auto-detected")
    assert metadata["input_path"] == str(simulation_input.resolve())
    assert metadata["resolved_cell_angstrom"] == pytest.approx([10.0, 10.0, 10.0])


def test_compute_coordination_logs_coordination_timestep_without_backend_noise(
    tmp_path, monkeypatch, capsys
):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("", encoding="utf-8")
    rdf_path = tmp_path / "reference_rdf.h5"
    save_rdf_profile(
        RDFProfile(
            species_a="O",
            species_b="H",
            bin_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float),
            bin_centers=np.array([0.25, 0.75, 1.25, 1.75], dtype=float),
            g_r=np.array([0.2, 1.8, 0.35, 0.7], dtype=float),
            n_frames=4,
        ),
        rdf_path,
    )

    frames = [
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.20],
                [1.0, 0.0, 0.20],
                [0.0, 0.0, 1.00],
                [0.70, 0.0, 1.00],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "PtPtOH",
            positions=[
                [0.0, 0.0, 0.30],
                [1.0, 0.0, 0.30],
                [0.0, 0.0, 1.20],
                [1.10, 0.0, 1.20],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    monkeypatch.setattr("linak.trajectory.io.read_trajectory", lambda _path: frames)

    rc = main(
        [
            "--log-level",
            "INFO",
            "compute",
            "coordination",
            str(trajectory),
            "--species-a",
            "O",
            "--species-b",
            "H",
            "--axis",
            "z",
            "--surface-mode",
            "rough",
            "--surface-elements",
            "Pt",
            "--cell",
            "10",
            "10",
            "10",
            "--timestep-fs",
            "2.0",
            "--cutoff-rdf",
            str(rdf_path),
        ]
    )

    assert rc == 0
    stderr = capsys.readouterr().err
    assert "Using timestep for coordination analysis: 2" in stderr
    assert "MSD analysis" not in stderr
    assert "Configured Matplotlib backend" not in stderr


def test_apply_pbc_dump_default_output_uses_xyz_suffix(tmp_path):
    trajectory = tmp_path / "lammps.dump"
    _write_lammps_dump(trajectory, positions=[(1.2, -0.1, 0.5)])

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "pbc",
            str(trajectory),
        ]
    )

    assert rc == 0
    wrapped = read(tmp_path / "lammps_pbc.xyz")
    assert wrapped.positions[0, 0] == pytest.approx(0.2)
