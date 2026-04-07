from copy import deepcopy
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
    _build_rdf_profile_filter_options,
    _default_rdf_collection_hdf5_output_path,
    _default_rdf_selected_hdf5_output_path,
    _combine_analysis_hdf5_sources,
    _load_rdf_plot_profiles,
    _without_preview_series_state,
    _rewrite_implicit_csv_interactive,
    _rewrite_implicit_plot_csv,
    build_parser,
    main,
)
from linak.analysis.density import (
    compute_density_profile,
    load_density_profile,
    load_density_profiles,
    load_density_profiles_by_index,
    save_density_profile,
)
from linak.analysis.position import (
    compute_position_profile,
    load_position_profile,
    load_position_profiles_by_index,
    save_position_profile,
)
from linak.analysis.coordination import (
    CoordinationCutoffResolution,
    compute_coordination_profile,
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
from linak.analysis.rdf import RDFProfile, compute_rdf, load_rdf_profile, save_rdf_profile
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


def _write_density_hdf5(path: Path) -> None:
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
    profile = compute_density_profile([frame0, frame1], species="O", axis="z", bin_width=0.1)
    save_density_profile(profile, path)


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
    assert "linak plot /path/to/traj_density_o_z.h5" in out


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
    assert ".traj.h5" in out
    assert "--format" in out


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
    write_plot_profile(source, "plot:density", {"title": "Saved title"})

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

    stored = read_plot_profile(source, "plot:density")
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

    stored = read_plot_profile(source, "plot:density")
    assert stored is None


def test_plot_density_keeps_existing_settings_when_changed_non_interactively(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)
    write_plot_profile(source, "plot:density", {"title": "Original Title"})

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

    stored = read_plot_profile(source, "plot:density")
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

    target_profile = read_plot_profile(target, "plot:table")
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

    stored = read_plot_profile(source, "plot:table")
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
    assert args.surface_mode == "auto"
    assert args.surface_elements is None
    assert args.include_fixed_surface_atoms is False
    assert args.rough_surface_envelope is None


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
    expected_default_output = _linak_output_dir(tmp_path) / "missing_traj_density_o_z.h5"

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
        _linak_output_dir(tmp_path) / "traj.traj_density_o_z.h5",
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
    write_plot_profile(source_b, "plot:density", {"title": "From second file"})

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
        {
            "series_labels": ["H2O"],
            "line_colors": ["#1f77b4"],
        },
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
    original_settings = read_plot_profile(source_a, "plot:density")
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
        {"series_labels": ["run-A"], "line_colors": ["#ff0000"]},
    )
    write_plot_profile(
        source_b,
        "plot:density",
        {"series_labels": ["run-B"], "line_colors": ["#00ff00"]},
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
        {
            "series_labels": ["first", "second"],
            "line_colors": ["#ff0000", "#00ff00"],
        },
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

    captured_kwargs: dict[str, str] = {}

    def _fake_plot_density_profiles(_profiles, **kwargs):
        captured_kwargs["x_mode"] = kwargs["x_mode"]
        captured_kwargs["quantity"] = kwargs["quantity"]
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
    assert captured_kwargs == {"x_mode": "axis", "quantity": "number"}


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
    _write_density_hdf5(source_h5)

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
    assert initial["x_mode"] == "axis"
    assert initial["quantity"] == "number"
    resolver = captured["on_resolve_series_defaults"]
    resolved = resolver(initial)
    assert resolved["series_count"] == 1
    assert resolved["series_labels"] == ["H2O"]
    assert captured["title"] == "LiNaK Plot Controls: Density"
    assert captured["initial_profile_name"] == "Default"


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
    write_plot_profile(source_h5_b, "plot:density", {"title": "From second"})

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
    copied_settings = read_plot_profile(combined_source, "plot:density")
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
    write_plot_profile(source_h5_b, "plot:density", {"title": "From second"})

    output_path = cli_mod._combine_analysis_hdf5_sources(
        sources=[str(source_h5_a), str(source_h5_b)],
        analysis="density",
        output=combined_h5,
        settings_source_path=source_h5_b,
    )

    assert output_path == combined_h5.resolve()
    imported_settings = read_plot_profile(output_path, "plot:density")
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
        {"series_labels": ["run-A"], "line_colors": ["#ff0000"]},
    )
    write_plot_profile(
        source_h5_b,
        "plot:density",
        {"series_labels": ["run-B"], "line_colors": ["#00ff00"]},
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
    merged_settings = read_plot_profile(combined_source, "plot:density")
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
    reopened_catalog = cli_mod._build_density_gui_lazy_catalog(
        reopened_args,
        sources=[str(combined_source)],
    )
    reopened_catalog.default_series_labels = cli_mod._resolve_gui_default_series_labels(
        args=reopened_args,
        sources=[str(combined_source)],
        profile_key=cli_mod._PLOT_PROFILE_DENSITY,
        fallback_labels_by_source=reopened_catalog.fallback_labels_by_source,
    )
    reopened_context = reopened_catalog.build_initial_context()

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
    write_plot_profile(source_h5, "plot:density", {"title": "Imported title"})

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
    write_plot_profile(source_h5, "plot:density", {"title": "Default title"})
    write_plot_profile(
        source_h5,
        "plot:density",
        {"title": "Paper title", "x_mode": "axis"},
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
    assert initial["x_mode"] == "axis"


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
    saved = read_plot_profile(combined_h5, "plot:density", profile_name="Publication")
    assert isinstance(saved, dict)
    assert saved["title"] == "Combined paper"
    assert read_active_plot_profile_name(combined_h5, "plot:density") == "Publication"
    assert on_load_profile("Publication")["title"] == "Combined paper"


def test_plot_density_gui_hdf5_import_callback_respects_explicit_selected_profile(
    tmp_path, monkeypatch
):
    source_h5 = tmp_path / "source_density.h5"
    _write_density_hdf5(source_h5)
    write_plot_profile(source_h5, "plot:density", {"title": "Default title"})
    write_plot_profile(
        source_h5,
        "plot:density",
        {"title": "Paper title"},
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
        {
            "series_descriptors": context.series_descriptors,
            "series_order": [
                context.series_descriptors[1]["series_id"],
                context.series_descriptors[0]["series_id"],
            ],
            "series_overrides": {
                context.series_descriptors[0]["series_id"]: {"enabled": False},
                context.series_descriptors[1]["series_id"]: {"label_override": "kept"},
            },
        },
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
        {
            "series_descriptors": context.series_descriptors,
            "series_overrides": {
                context.series_descriptors[0]["series_id"]: {"alpha": 0.5},
            },
        },
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
        return {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in {"output", "capture_state"}
        }

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
    saved = read_plot_profile(combined_h5, "plot:density")
    assert saved is not None
    assert saved["series_order"] == expected["series_order"]
    assert saved["series_overrides"][expected["first_series_id"]]["alpha"] == 0.5
    assert saved["series_overrides"][expected["first_series_id"]]["enabled"] is False
    assert saved["series_overrides"][expected["second_series_id"]]["label_override"] == "reordered"
    assert saved.get("line_kwargs") is None or saved.get("line_kwargs") == {}


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
        {
            "series_descriptors": context.series_descriptors,
            "series_overrides": {
                disabled_series_id: {"enabled": False},
            },
        },
    )

    load_calls: list[list[int]] = []

    def _fake_load_density_profiles_by_index(path, indices, *, axis=None, species=None):
        load_calls.append([int(index) for index in indices])
        return load_density_profiles_by_index(path, indices, axis=axis, species=species)

    def _fake_render_profile_plot(**_kwargs):
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
        {
            "series_descriptors": context.series_descriptors,
            "series_overrides": {
                descriptor["series_id"]: {"enabled": descriptor["series_id"] == selected_series_id}
                for descriptor in context.series_descriptors
            },
        },
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
    write_plot_profile(source, "plot:density", {"title": "Default title"})

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
    assert read_plot_profile(source, "plot:density", profile_name="Publication") == {
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

    def _fake_plot_density_profile(_profile, **kwargs):
        plot_calls.append(kwargs)
        return None

    def _fake_gui_launcher(**kwargs):
        kwargs["on_preview"]({"series_labels": ["custom-series"]})

    monkeypatch.setattr("linak.analysis.density.plot_density_profile", _fake_plot_density_profile)
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
    assert plot_calls[1]["line_label"] == "custom-series"


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

    def _fake_compute_density_profiles(**kwargs):
        captured["surface_mode"] = kwargs["surface_mode"]
        captured["surface_elements"] = kwargs["surface_elements"]
        captured["include_fixed_surface_atoms"] = kwargs["include_fixed_surface_atoms"]
        captured["surface_options"] = kwargs["surface_options"]
        profile = compute_density_profile(
            [frame],
            species=kwargs["species"],
            axis=kwargs["axis"],
            bin_width=kwargs["bin_width"],
        )
        return [profile]

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr(
        "linak.analysis.density.compute_density_profiles", _fake_compute_density_profiles
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
        captured["component"] = kwargs.get("component")
        captured["map_color"] = kwargs.get("map_color")
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
        f"{source_h5_1.name}:O",
        f"{source_h5_2.name}:O",
        f"{source_h5_2.name}:O",
    ]
    assert captured["component"] == "xy-z"
    assert captured["map_color"] == "distance"
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
    assert initial["series_labels"] == ["O[2]", "O[3]"]
    assert initial["axis"] is None
    assert initial["component"] == "distance"
    assert initial["map_color"] == "distance"
    assert initial["time_axis"] == "ps"
    resolver = captured["on_resolve_series_defaults"]
    resolved = resolver(initial)
    assert resolved["series_count"] == 2
    assert resolved["series_labels"] == ["O[2]", "O[3]"]


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
    assert initial["component"] == "distance"
    assert initial["series_count"] == 1
    assert initial["series_labels"] == ["O-H"]
    resolver = captured["on_resolve_series_defaults"]
    resolved = resolver({**initial, "component": "time"})
    assert resolved["series_count"] == 1
    assert resolved["series_labels"] == ["O[2]"]


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
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profile", lambda *_args, **_kwargs: None)

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
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profile", lambda *_args, **_kwargs: None)

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
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profile", lambda *_args, **_kwargs: None)

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


def test_compute_rdf_explicit_atom_selection_uses_selected_default_output(tmp_path, monkeypatch):
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

    def _fake_save_rdf_profile(_profile, output, **_kwargs):
        captured_output["path"] = Path(output)
        return Path(output)

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.rdf.compute_rdf", lambda **_kwargs: profile)
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profile", _fake_save_rdf_profile)

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
    assert captured_output["path"] == _default_rdf_selected_hdf5_output_path(trajectory)


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
    monkeypatch.setattr("linak.analysis.rdf.save_rdf_profile", lambda *_args, **_kwargs: None)

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


def test_compute_coordination_requires_explicit_species_pair(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    _write_xyz(trajectory)

    rc = main(["--log-level", "ERROR", "compute", "coordination", str(trajectory)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--species-a" in err
    assert "--species-b" in err


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
    assert (_linak_output_dir(tmp_path) / "traj_density_o_z.h5").exists()


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
    assert (_linak_output_dir(tmp_path) / "traj_density_o_z.h5").exists()


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
    assert (_linak_output_dir(trajectory_dir) / "traj_density_o_z.h5").exists()
    assert not (work_dir / "traj_density_o_z.h5").exists()


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
    assert (output_dir / "traj_density_o_z.h5").exists()
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
    assert not (output_base / "traj_density_o_z.h5").exists()


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
    assert (_linak_output_dir(trajectory_dir) / "traj_rdf_o_h.h5").exists()
    assert not (work_dir / "traj_rdf_o_h.h5").exists()


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
    assert (_linak_output_dir(tmp_path) / "traj_density_o_z.h5").exists()
    assert (_linak_output_dir(tmp_path) / "traj_density_o_z_1.h5").exists()


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
    profile = load_density_profile(_linak_output_dir(tmp_path) / "traj_density_o_z.h5")
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
        _linak_output_dir(tmp_path) / "traj_density_o_z.h5",
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
    profile = load_density_profile(_linak_output_dir(tmp_path) / "traj_density_o_z.h5")
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
    output = _linak_output_dir(tmp_path) / "traj_density_z.h5"
    assert output.exists()
    profiles = load_density_profiles(output)
    assert [profile.species for profile in profiles] == ["H", "O", "H2O"]


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
    assert (_linak_output_dir(tmp_path) / "water_density_h2o_z.h5").exists()
    assert not (_linak_output_dir(tmp_path) / "water_density_h_z.h5").exists()
    assert not (_linak_output_dir(tmp_path) / "water_density_o_z.h5").exists()


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
