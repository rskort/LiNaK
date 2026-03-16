from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from linak.cli import _rewrite_implicit_csv_interactive, _rewrite_implicit_plot_csv, build_parser, main
from linak.analysis.density import (
    compute_density_profile,
    load_density_profile,
    load_density_profiles,
    save_density_profile,
)
from linak.storage.hdf5_table import read_hdf5_frame
from linak.storage.hdf5_utils import read_linak_hdf5
from linak.analysis.msd import compute_msd, load_msd_profile, save_msd_profile
from linak.plot.plot_settings import read_plot_profile, write_plot_profile
from linak.analysis.rdf import compute_rdf, save_rdf_profile


def _write_xyz(path: Path) -> None:
    frame0 = Atoms("OO", positions=[[0.0, 0.0, 0.02], [0.0, 0.0, 0.08]])
    frame1 = Atoms("OO", positions=[[0.0, 0.0, 0.12], [0.0, 0.0, 0.18]])
    write(path, [frame0, frame1], format="extxyz")


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
        " CP2K| version string\n"
        " GLOBAL| Run type ENERGY\n"
        " PROGRAM ENDED AT 2026-01-01 00:00:00\n",
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
    assert "linak plot density" in out


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
    assert "linak apply pbc" in out
    assert "linak apply compress" in out


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
    assert "--labels/--series-labels count must match rendered series count" in capsys.readouterr().err


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


def test_rewrite_implicit_plot_csv():
    assert _rewrite_implicit_plot_csv(["plot", "table.h5"]) == [
        "hdf5",
        "plot",
        "table.h5",
    ]
    assert _rewrite_implicit_plot_csv(["plot", "density", "table.h5"]) == [
        "plot",
        "density",
        "table.h5",
    ]
    assert _rewrite_implicit_plot_csv(["plot", "--help"]) == [
        "plot",
        "--help",
    ]


def test_rewrite_implicit_plot_csv_auto_detects_density_analysis(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    rewritten = _rewrite_implicit_plot_csv(["plot", str(source)])
    assert rewritten == ["plot", "density", str(source)]


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
        "density",
        "-f",
        str(source_a),
        str(source_b),
        "--no-show",
    ]


def test_plot_density_persists_plot_settings_in_hdf5(tmp_path):
    source = tmp_path / "density.h5"
    output = tmp_path / "density.png"
    _write_density_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
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
    assert stored is not None
    assert stored["title"] == "Stored Density Title"
    assert stored["x_lim"] == pytest.approx([0.0, 3.0])


def test_plot_density_toggle_controls_persist_in_hdf5(tmp_path):
    source = tmp_path / "density.h5"
    output = tmp_path / "density_toggles.png"
    _write_density_hdf5(source)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
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
    assert stored is not None
    assert stored["title_visible"] is False
    assert stored["legend"] is False
    assert stored["grid"] is False
    assert stored["ticks"] is False
    assert stored["markers"] is True


def test_plot_density_keeps_existing_settings_when_changed_non_interactively(tmp_path):
    source = tmp_path / "density.h5"
    _write_density_hdf5(source)

    rc_first = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
            str(source),
            "--title",
            "Original Title",
            "--no-show",
            "--output",
            str(tmp_path / "first.png"),
        ]
    )
    assert rc_first == 0

    rc_second = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
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
    assert stored is not None
    assert stored["x_lim"][0] == pytest.approx(0.5)


def test_plot_help_lists_subcommands_and_style_options(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--help"])
    out = capsys.readouterr().out
    assert "density" in out
    assert "msd" in out
    assert "rdf" in out
    assert "--font-family" in out
    assert "--title-font-size" in out


def test_plot_subcommands_reject_save_data_flag():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "rdf", "input.h5", "--save-data", "copy.h5"])


@pytest.mark.parametrize(
    "argv",
    [
        ["plot", "density", "input.h5", "--dry-run"],
        ["plot", "msd", "input.h5", "--dry-run"],
        ["plot", "rdf", "input.h5", "--dry-run"],
        ["hdf5", "combine", "-f", "a.h5", "b.h5", "--dry-run"],
        ["compute", "density", "traj.xyz", "--dry-run"],
        ["compute", "msd", "traj.xyz", "--dry-run"],
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
    args = build_parser().parse_args(["plot", "density", "input.h5"])
    assert args.x_mode == "distance"
    assert args.quantity == "mass"


def test_compute_density_defaults_surface_detection_options():
    args = build_parser().parse_args(["compute", "density", "traj.xyz"])
    assert args.surface_mode == "auto"
    assert args.surface_elements is None
    assert args.include_fixed_surface_atoms is False


def test_plot_density_dry_run_skips_input_file_processing(tmp_path):
    missing_source = tmp_path / "missing_density.h5"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
            str(missing_source),
            "--dry-run",
            "--no-show",
        ]
    )

    assert rc == 0
    assert not missing_source.exists()


def test_compute_density_dry_run_skips_trajectory_read_and_csv_write(tmp_path):
    missing_trajectory = tmp_path / "missing_traj.xyz"
    expected_default_output = tmp_path / "missing_traj_density_o_z.h5"

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
    assert "r_max resolution: 7.7455" in err


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
            "density",
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
            "density",
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


def test_plot_density_multi_uses_requested_settings_source(tmp_path, monkeypatch):
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

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
            "-f",
            str(source_a),
            str(source_b),
            "--settings-source",
            "2",
            "--no-show",
        ]
    )

    assert rc == 0
    assert captured["title"] == "From second file"


def test_plot_density_multi_writes_new_combined_hdf5_for_settings(tmp_path, monkeypatch):
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
            "density",
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
    assert len(combined_files) == 1
    combined_source = combined_files[0]
    assert captured["source"] == str(combined_source)

    combined_settings = read_plot_profile(combined_source, "plot:density")
    assert combined_settings is not None
    assert combined_settings["series_labels"] == ["H2O", f"{source_b.name}:O"]
    assert len(combined_settings["line_colors"]) == 2


def test_plot_density_multi_auto_merges_series_labels_and_colors_from_sources(tmp_path, monkeypatch):
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
            "density",
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
            "density",
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
            "density",
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
            "density",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["profile_key"] == "plot:density"
    assert captured["analysis_name"] == "density"
    assert captured["gui_title"] == "LiNaK Plot Controls: Density"


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
            "density",
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


def test_plot_density_gui_multi_sources_copy_settings_from_requested_source(tmp_path, monkeypatch):
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
            "density",
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
    assert copied_settings is not None
    assert copied_settings["title"] == "From second"


def test_plot_density_gui_multi_sources_combine_series_labels_and_colors(tmp_path, monkeypatch):
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
            "density",
            "-f",
            str(source_h5_a),
            str(source_h5_b),
            "--gui",
        ]
    )

    assert rc == 0
    assert captured["args"].series_labels == ["run-A", "run-B"]
    assert captured["args"].line_colors == ["#ff0000", "#00ff00"]
    combined_source = captured["source_path"]
    assert isinstance(combined_source, Path)
    merged_settings = read_plot_profile(combined_source, "plot:density")
    assert merged_settings is not None
    assert merged_settings["series_labels"] == ["run-A", "run-B"]
    assert merged_settings["line_colors"] == ["#ff0000", "#00ff00"]


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
            "density",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(preview_calls) == 2


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
            "density",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(plot_calls) == 1
    assert plot_calls[0]["show"] is True
    assert plot_calls[0]["show_blocking"] is False


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
            "density",
            str(source_h5),
            "--gui",
        ]
    )

    assert rc == 0
    assert len(plot_calls) == 1
    assert plot_calls[0]["line_label"] == "custom-series"


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
        profile = compute_density_profile(
            [frame],
            species=kwargs["species"],
            axis=kwargs["axis"],
            bin_width=kwargs["bin_width"],
        )
        return [profile]

    monkeypatch.setattr("linak.trajectory.io.read_trajectory", _fake_read_trajectory)
    monkeypatch.setattr("linak.analysis.density.compute_density_profiles", _fake_compute_density_profiles)

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
        ]
    )

    assert rc == 0
    assert captured["surface_mode"] == "layered"
    assert captured["surface_elements"] == ["Au", "Pt"]
    assert captured["include_fixed_surface_atoms"] is True


def test_plot_density_rejects_trajectory_input(tmp_path, capsys):
    trajectory = tmp_path / "traj.xyz"
    write(trajectory, [Atoms("O", positions=[[0.0, 0.0, 0.1]])], format="extxyz")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "density",
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
            "density",
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
            "msd",
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
            "msd",
            str(trajectory),
            "--no-show",
        ]
    )

    assert rc == 1
    assert "only accepts HDF5 input" in capsys.readouterr().err


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
            "rdf",
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
            "rdf",
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
    assert (tmp_path / "traj_density_o_z.h5").exists()


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
    assert (tmp_path / "traj_density_o_z.h5").exists()


def test_compute_density_default_csv_is_next_to_input_not_cwd(tmp_path, monkeypatch):
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
    assert (trajectory_dir / "traj_density_o_z.h5").exists()
    assert not (work_dir / "traj_density_o_z.h5").exists()


def test_compute_msd_default_csv_is_next_to_input_not_cwd(tmp_path, monkeypatch):
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
    assert (trajectory_dir / "traj_msd_o.h5").exists()
    assert not (work_dir / "traj_msd_o.h5").exists()


def test_compute_rdf_default_csv_is_next_to_input_not_cwd(tmp_path, monkeypatch):
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
    assert (trajectory_dir / "traj_rdf_o_h.h5").exists()
    assert not (work_dir / "traj_rdf_o_h.h5").exists()


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
    profile = load_density_profile(tmp_path / "traj_density_o_z.h5")
    assert profile.units == "g/Angstrom^3"


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
        tmp_path / "traj_density_o_z.h5",
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
    profile = load_density_profile(tmp_path / "traj_density_o_z.h5")
    assert profile.units == "g/Angstrom^3"


def test_compute_density_all_writes_one_csv_per_element(tmp_path, monkeypatch):
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
    assert (tmp_path / "traj_density_h_z.h5").exists()
    assert (tmp_path / "traj_density_o_z.h5").exists()


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
    assert (tmp_path / "water_density_h2o_z.h5").exists()
    assert not (tmp_path / "water_density_h_z.h5").exists()
    assert not (tmp_path / "water_density_o_z.h5").exists()


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
    profile = load_msd_profile(tmp_path / "traj_msd_o.h5")
    assert profile.time_fs[1] == pytest.approx(2.5)

    _datasets, metadata = read_linak_hdf5(tmp_path / "traj_msd_o.h5", expected_analysis="msd")
    assert metadata["source_path"] == str(trajectory.resolve())
    assert metadata["timestep_source"].startswith("auto-detected")
    assert metadata["frame_timestep_fs"] == pytest.approx(2.5)
    assert metadata["resolved_cell_angstrom"] == pytest.approx([1.0, 1.0, 1.0])


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


