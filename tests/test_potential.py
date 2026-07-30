from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from linak.cli import main
from linak.analysis.potential import (
    BOHR_TO_ANG,
    CubeHeader,
    HARTREE_TO_EV,
    PotentialComputationFailure,
    PotentialConfig,
    PotentialRecord,
    combine_potential_hdf5_sources,
    compute_potential_records,
    load_potential_plot_profiles,
    plot_potential_profiles,
    _non_water_contaminated_z_slices,
    _resolve_worker_count,
    _water_bulk_potential_ev,
)
from linak.storage.hdf5_utils import read_linak_hdf5_profiles
from linak.plot.data_contract import PlotViewMapping
from linak.plot.mappings.potential_mapping import potential_plot_options_to_view_mapping


def _write_cube(
    path: Path,
    *,
    values_by_z: np.ndarray,
    atom_specs: list[tuple[int, float]],
) -> None:
    nx = 1
    ny = 1
    nz = int(values_by_z.size)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("CP2K CUBE FILE\n")
        handle.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        handle.write(f"{len(atom_specs)} 0.0 0.0 0.0\n")
        handle.write(f"{nx} 1.0 0.0 0.0\n")
        handle.write(f"{ny} 0.0 1.0 0.0\n")
        handle.write(f"{nz} 0.0 0.0 1.0\n")

        for atomic_number, z_bohr in atom_specs:
            handle.write(f"{atomic_number} 0.0 0.0 0.0 {z_bohr:.8f}\n")

        flat_values: list[float] = []
        for value in values_by_z:
            for _ in range(nx * ny):
                flat_values.append(float(value))

        for index, value in enumerate(flat_values, start=1):
            handle.write(f" {value: .8E}")
            if index % 6 == 0:
                handle.write("\n")
        if flat_values and len(flat_values) % 6 != 0:
            handle.write("\n")


def _write_potential_case(run_dir: Path, *, fermi_au: float | None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    potential_ev = np.array(
        [2.0] * 10 + [8.0] * 6 + [1.0] * 4,
        dtype=float,
    )
    atom_specs = [
        (8, 2.0),
        (8, 8.0),
        (1, 7.0),
        (79, 12.0),
    ]

    hartree_cube = run_dir / "sample-v_hartree-1_0.cube"
    _write_cube(
        hartree_cube,
        values_by_z=potential_ev / HARTREE_TO_EV,
        atom_specs=atom_specs,
    )

    if fermi_au is not None:
        (run_dir / "output.out").write_text(
            f"Some CP2K output\nFermi energy: {fermi_au:.12f}\n",
            encoding="utf-8",
        )
    return hartree_cube


def _read_hdf5_rows(path: Path) -> list[dict[str, str]]:
    with h5py.File(path, "r") as handle:
        records = handle["records"]
        columns = list(records.keys())
        row_count = int(records[columns[0]].shape[0]) if columns else 0
        rows: list[dict[str, str]] = []
        for index in range(row_count):
            row: dict[str, str] = {}
            for column in columns:
                value = records[column][index]
                if isinstance(value, bytes):
                    row[column] = value.decode("utf-8")
                elif isinstance(value, (np.floating, float)):
                    row[column] = "" if np.isnan(float(value)) else str(float(value))
                else:
                    row[column] = str(value)
            rows.append(row)
    return rows


def _write_potential_summary_hdf5(
    path: Path,
    *,
    ids: list[int],
    efermi: list[float | None],
    water_bulk: list[float | None],
    cshe: list[float | None],
    status: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["analysis"] = "potential"
        records = handle.create_group("records")
        records.create_dataset("id", data=np.asarray(ids, dtype=np.int64))
        records.create_dataset(
            "source",
            data=np.asarray([f"source_{index}" for index in range(len(ids))], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "source_dir",
            data=np.asarray([f"dir_{index}" for index in range(len(ids))], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "output_out",
            data=np.asarray(["output.out" for _index in range(len(ids))], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "status",
            data=np.asarray(status, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "error",
            data=np.asarray(["" for _index in range(len(ids))], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        records.create_dataset(
            "efermi_ev",
            data=np.asarray([np.nan if value is None else value for value in efermi], dtype=float),
        )
        records.create_dataset(
            "water_bulk_potential_ev",
            data=np.asarray(
                [np.nan if value is None else value for value in water_bulk],
                dtype=float,
            ),
        )
        records.create_dataset(
            "electrode_cshe_ev",
            data=np.asarray([np.nan if value is None else value for value in cshe], dtype=float),
        )


def _test_cube_header(atom_specs_ang: list[tuple[int, float]], *, nz: int) -> CubeHeader:
    return CubeHeader(
        natoms=len(atom_specs_ang),
        origin_bohr=np.zeros(3, dtype=float),
        nx=1,
        ny=1,
        nz=nz,
        vx_bohr=np.asarray([1.0, 0.0, 0.0], dtype=float),
        vy_bohr=np.asarray([0.0, 1.0, 0.0], dtype=float),
        vz_bohr=np.asarray([0.0, 0.0, 1.0 / BOHR_TO_ANG], dtype=float),
        atom_numbers=np.asarray([atomic_number for atomic_number, _ in atom_specs_ang], dtype=int),
        atom_z_bohr=np.asarray(
            [z_ang / BOHR_TO_ANG for _, z_ang in atom_specs_ang],
            dtype=float,
        ),
    )


def test_water_bulk_potential_excludes_non_water_z_slice_from_widest_span():
    z_ang = np.arange(9, dtype=float)
    v_xyavg_ev = np.asarray([4.0, 4.0, 99.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    header = _test_cube_header([(8, 0.0), (1, 8.0), (11, 2.1)], nz=z_ang.size)

    contaminated = _non_water_contaminated_z_slices(z_ang, header)
    value, z_min, z_max = _water_bulk_potential_ev(
        z_ang=z_ang,
        v_xyavg_ev=v_xyavg_ev,
        water_z_min_ang=0.0,
        water_z_max_ang=8.0,
        padding_ang=0.0,
        contaminated_z_mask=contaminated,
    )

    assert contaminated.tolist() == [False, False, True, False, False, False, False, False, False]
    assert value == pytest.approx(2.0)
    assert z_min == pytest.approx(3.0)
    assert z_max == pytest.approx(8.0)


def test_water_bulk_potential_padding_fallback_keeps_contaminated_slice_out():
    z_ang = np.arange(7, dtype=float)
    v_xyavg_ev = np.asarray([1.0, 1.0, 7.0, 99.0, 7.0, 1.0, 1.0])
    header = _test_cube_header([(8, 0.0), (1, 6.0), (11, 3.0)], nz=z_ang.size)

    value, _z_min, _z_max = _water_bulk_potential_ev(
        z_ang=z_ang,
        v_xyavg_ev=v_xyavg_ev,
        water_z_min_ang=0.0,
        water_z_max_ang=6.0,
        padding_ang=2.9,
        contaminated_z_mask=_non_water_contaminated_z_slices(z_ang, header),
    )

    assert value == pytest.approx(7.0)


def test_water_bulk_potential_returns_none_when_all_candidate_slices_are_contaminated():
    z_ang = np.arange(3, dtype=float)
    header = _test_cube_header(
        [(8, 0.0), (1, 2.0), (11, 0.0), (19, 1.0), (3, 2.0)],
        nz=z_ang.size,
    )

    value, z_min, z_max = _water_bulk_potential_ev(
        z_ang=z_ang,
        v_xyavg_ev=np.asarray([10.0, 20.0, 30.0]),
        water_z_min_ang=0.0,
        water_z_max_ang=2.0,
        padding_ang=0.0,
        contaminated_z_mask=_non_water_contaminated_z_slices(z_ang, header),
    )

    assert value is None
    assert z_min is None
    assert z_max is None


def test_load_potential_plot_profiles_sorts_rows_and_summarizes(tmp_path):
    source = tmp_path / "potential.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[3, 1, 2],
        efermi=[1.3, 1.1, 1.2],
        water_bulk=[2.3, 2.1, 2.2],
        cshe=[0.3, 0.1, 0.2],
        status=["ok", "ok", "missing_fermi"],
    )

    profiles, summary = load_potential_plot_profiles(source)

    assert [profile.series_id for profile in profiles] == [
        "water_bulk_potential_ev",
        "efermi_ev",
        "electrode_cshe_ev",
    ]
    assert [profile.default_label for profile in profiles] == ["Water bulk", "Fermi", "cSHE"]
    assert profiles[0].x_values.tolist() == [1.0, 2.0, 3.0]
    assert profiles[0].y_values.tolist() == [2.1, 2.2, 2.3]
    assert profiles[1].y_values.tolist() == [1.1, 1.2, 1.3]
    assert summary == {
        "x_axis_label": "Record ID",
        "total_rows": 3,
        "complete_rows": 2,
        "incomplete_rows": 1,
    }


def test_load_potential_plot_profiles_falls_back_to_row_order_for_invalid_ids(tmp_path):
    source = tmp_path / "potential_invalid_ids.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[2, 2, -1],
        efermi=[1.0, 2.0, 3.0],
        water_bulk=[4.0, 5.0, 6.0],
        cshe=[0.1, 0.2, 0.3],
        status=["ok", "ok", "ok"],
    )

    profiles, _summary = load_potential_plot_profiles(source)

    assert profiles[0].x_values.tolist() == [1.0, 2.0, 3.0]
    assert profiles[1].y_values.tolist() == [1.0, 2.0, 3.0]


def test_combine_potential_hdf5_sources_records_original_row_spans(tmp_path):
    source_a = tmp_path / "potential_a.h5"
    source_b = tmp_path / "potential_b.h5"
    combined = tmp_path / "combined.h5"
    _write_potential_summary_hdf5(
        source_a,
        ids=[1, 2],
        efermi=[1.1, 1.2],
        water_bulk=[2.1, 2.2],
        cshe=[0.1, 0.2],
        status=["ok", "ok"],
    )
    _write_potential_summary_hdf5(
        source_b,
        ids=[1, 2, 3],
        efermi=[3.1, 3.2, 3.3],
        water_bulk=[4.1, 4.2, 4.3],
        cshe=[0.3, 0.4, 0.5],
        status=["ok", "missing_fermi", "ok"],
    )

    output = combine_potential_hdf5_sources([source_a, source_b], output=combined)

    assert output == combined.resolve()
    with h5py.File(output, "r") as handle:
        group = handle["combined_sources"]
        assert [str(value) for value in group["source_path"].asstr()[...]] == [
            str(source_a.resolve()),
            str(source_b.resolve()),
        ]
        assert group["row_start"][:].tolist() == [0, 2]
        assert group["row_count"][:].tolist() == [2, 3]


def test_load_potential_plot_profiles_reopens_combined_sources_as_separate_layers(tmp_path):
    source_a = tmp_path / "potential_a.h5"
    source_b = tmp_path / "potential_b.h5"
    combined = tmp_path / "combined.h5"
    _write_potential_summary_hdf5(
        source_a,
        ids=[2, 1],
        efermi=[1.2, 1.1],
        water_bulk=[2.2, 2.1],
        cshe=[0.2, 0.1],
        status=["ok", "ok"],
    )
    _write_potential_summary_hdf5(
        source_b,
        ids=[2, 1],
        efermi=[3.2, 3.1],
        water_bulk=[4.2, 4.1],
        cshe=[0.4, 0.3],
        status=["missing_fermi", "ok"],
    )
    combine_potential_hdf5_sources([source_a, source_b], output=combined)

    profiles, summary = load_potential_plot_profiles(combined)

    assert len(profiles) == 6
    assert [profile.series_id for profile in profiles] == [
        f"{source_a.resolve()}::water_bulk_potential_ev",
        f"{source_a.resolve()}::efermi_ev",
        f"{source_a.resolve()}::electrode_cshe_ev",
        f"{source_b.resolve()}::water_bulk_potential_ev",
        f"{source_b.resolve()}::efermi_ev",
        f"{source_b.resolve()}::electrode_cshe_ev",
    ]
    assert [profile.default_label for profile in profiles] == [
        "potential_a: Water bulk",
        "potential_a: Fermi",
        "potential_a: cSHE",
        "potential_b: Water bulk",
        "potential_b: Fermi",
        "potential_b: cSHE",
    ]
    assert profiles[1].x_values.tolist() == [1.0, 2.0]
    assert profiles[1].y_values.tolist() == [1.1, 1.2]
    assert profiles[4].y_values.tolist() == [3.1, 3.2]
    assert summary["total_rows"] == 4
    assert summary["complete_rows"] == 3
    assert summary["incomplete_rows"] == 1


def test_plot_potential_profiles_capture_defaults_and_fit_summary(tmp_path):
    source = tmp_path / "potential_plot.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[1, 2, 3, 4],
        efermi=[1.0, 1.2, 1.4, 1.6],
        water_bulk=[2.0, 2.1, 2.2, 2.3],
        cshe=[0.2, None, 0.4, 0.5],
        status=["ok", "ok", "ok", "ok"],
    )
    profiles, _summary = load_potential_plot_profiles(source)
    output = tmp_path / "potential.png"
    capture_state: dict[str, object] = {}

    result = plot_potential_profiles(
        profiles,
        show=False,
        output=output,
        capture_state=capture_state,
        series_fit_configs=[
            {"fit_enabled": True},
            {"fit_enabled": False},
            {"fit_enabled": True},
        ],
    )

    assert result == output.resolve()
    assert output.exists()
    assert capture_state["title"] == "Hartree potential summary"
    assert capture_state["x_label"] == "Record ID"
    assert capture_state["y_label"] == "Potential (eV)"
    fit_summaries = capture_state["series_fit_summaries"]
    assert isinstance(fit_summaries, dict)
    assert fit_summaries["water_bulk_potential_ev"]["status"] == "ok"
    assert fit_summaries["water_bulk_potential_ev"]["point_count"] == 4
    assert fit_summaries["efermi_ev"]["status"] == "off"
    assert fit_summaries["electrode_cshe_ev"]["status"] == "ok"


def test_plot_potential_profiles_rescales_record_id_axis(tmp_path):
    source = tmp_path / "potential_scaled_x.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[1, 2, 3],
        efermi=[1.0, 1.2, 1.4],
        water_bulk=[2.0, 2.1, 2.2],
        cshe=[0.2, 0.3, 0.4],
        status=["ok", "ok", "ok"],
    )
    profiles, _summary = load_potential_plot_profiles(source)
    capture_state: dict[str, object] = {}

    result = plot_potential_profiles(
        profiles,
        show=False,
        output=tmp_path / "potential_scaled_x.png",
        x_axis_scale=0.2,
        x_axis_offset=1.0,
        capture_state=capture_state,
    )

    assert result is not None
    line = capture_state["axes"].lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.array([1.2, 1.4, 1.6]))
    assert capture_state["x_axis_scale"] == pytest.approx(0.2)
    assert capture_state["x_axis_offset"] == pytest.approx(1.0)


def test_plot_potential_profiles_accepts_generic_single_series_mapping(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_plot_multi_line_series(x_series, y_series, labels, **kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["labels"] = labels
        captured["series_ids"] = kwargs["series_ids"]
        return tmp_path / "noop.png"

    monkeypatch.setattr(
        "linak.analysis.potential.plot_multi_line_series",
        _fake_plot_multi_line_series,
    )

    source = tmp_path / "potential_single_series.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[1, 2],
        efermi=[1.0, 1.1],
        water_bulk=[2.0, 2.1],
        cshe=[0.2, 0.3],
        status=["ok", "ok"],
    )
    profiles, _summary = load_potential_plot_profiles(source)

    plot_potential_profiles(
        profiles,
        show=False,
        view_mapping=potential_plot_options_to_view_mapping(y_quantity="efermi"),
    )

    assert captured["labels"] == ["Fermi"]
    assert captured["series_ids"] == ["efermi_ev"]
    np.testing.assert_allclose(captured["x_series"][0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(captured["y_series"][0], np.array([1.0, 1.1]))


def test_plot_potential_profiles_legacy_table_view_renders_summary_line(tmp_path):
    source = tmp_path / "potential_legacy_table_view.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[1, 2],
        efermi=[1.0, 1.1],
        water_bulk=[2.0, 2.1],
        cshe=[0.2, 0.3],
        status=["ok", "ok"],
    )
    profiles, _summary = load_potential_plot_profiles(source)
    capture_state: dict[str, object] = {}
    output = tmp_path / "potential_legacy_table_view.png"

    result = plot_potential_profiles(
        profiles,
        show=False,
        output=output,
        view_mapping=PlotViewMapping(view_type_id="table_records"),
        capture_state=capture_state,
    )

    assert result == output.resolve()
    assert output.exists()
    assert "table_rows" not in capture_state
    assert capture_state["x_label"] == "Record ID"
    assert capture_state["y_label"] == "Potential (eV)"


def test_plot_potential_hdf5_non_gui_renders_png(tmp_path):
    run_dir = tmp_path / "run"
    cube = _write_potential_case(run_dir, fermi_au=0.0367493036)
    source = tmp_path / "potential_summary.potential.h5"
    output = tmp_path / "potential_plot.png"

    compute_rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube),
            "--output",
            str(source),
            "--water-padding-ang",
            "0.2",
        ]
    )
    assert compute_rc == 0

    plot_rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            str(source),
            "--no-gui",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert plot_rc == 0
    assert output.exists()


def test_compute_potential_water_bulk_uses_clean_span_when_ion_is_in_water_region(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cube = run_dir / "ion-in-water-v_hartree-1_0.cube"
    output = tmp_path / "potential_summary.potential.h5"
    potential_ev = np.asarray([4.0, 4.0, 99.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    _write_cube(
        cube,
        values_by_z=potential_ev / HARTREE_TO_EV,
        atom_specs=[
            (8, 0.0),
            (1, 8.0),
            (11, 2.0),
        ],
    )
    (run_dir / "output.out").write_text(
        "Some CP2K output\nFermi energy: 0.0367493036\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube),
            "--output",
            str(output),
            "--water-padding-ang",
            "0.0",
        ]
    )

    assert rc == 0
    rows = _read_hdf5_rows(output)
    assert len(rows) == 1
    assert float(rows[0]["water_bulk_potential_ev"]) == pytest.approx(2.0, abs=1e-6)
    assert float(rows[0]["electrode_cshe_ev"]) == pytest.approx(0.19, abs=1e-3)


def test_apply_convert_cube_to_cube_hdf5_and_back(tmp_path):
    run_dir = tmp_path / "run"
    cube = _write_potential_case(run_dir, fermi_au=0.0367493036)
    cube_h5 = tmp_path / "field.cube.h5"
    roundtrip_cube = tmp_path / "field_roundtrip.cube"

    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(cube),
                "--output",
                str(cube_h5),
            ]
        )
        == 0
    )
    assert cube_h5.exists()
    payloads = read_linak_hdf5_profiles(cube_h5, expected_analysis="cube")
    assert len(payloads) == 1
    assert payloads[0][1]["source_path"] == str(cube.resolve())

    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(cube_h5),
                "--target-file-type",
                "cube",
                "--output",
                str(roundtrip_cube),
            ]
        )
        == 0
    )
    assert roundtrip_cube.exists()


def test_apply_combine_cubes_writes_cube_hdf5_and_preserves_metadata(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    cube_a = _write_potential_case(run_a, fermi_au=0.0367493036)
    cube_b = _write_potential_case(run_b, fermi_au=0.0734986072)
    combined = tmp_path / "combined.cube.h5"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(cube_a),
            str(cube_b),
            "--output",
            str(combined),
        ]
    )

    assert rc == 0
    payloads = read_linak_hdf5_profiles(combined, expected_analysis="cube")
    assert len(payloads) == 2
    assert payloads[0][1]["source_path"] == str(cube_a.resolve())
    assert payloads[1][1]["source_path"] == str(cube_b.resolve())
    assert payloads[0][1]["combine_source_paths"] == [str(cube_a.resolve()), str(cube_b.resolve())]
    assert payloads[0][1]["combine_source_file_types"] == ["cube_file", "cube_file"]
    assert payloads[0][1]["combine_conversion_applied"] is True
    assert payloads[0][1]["combine_total_fields"] == 2
    assert payloads[0][1]["combine_timestamp_utc"]
    assert payloads[0][1]["combine_linak_version"]


def test_apply_combine_cubes_no_convert_fails_clearly(tmp_path, capsys):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    cube_a = _write_potential_case(run_a, fermi_au=0.0367493036)
    cube_b = _write_potential_case(run_b, fermi_au=0.0734986072)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "apply",
            "combine",
            "-f",
            str(cube_a),
            str(cube_b),
            "--no-convert",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "no-convert" in err or ".cube.h5" in err


def test_compute_potential_accepts_cube_hdf5_input(tmp_path):
    run_dir = tmp_path / "run"
    cube = _write_potential_case(run_dir, fermi_au=0.0367493036)
    cube_h5 = tmp_path / "field.cube.h5"
    output = tmp_path / "potential_summary.potential.h5"

    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "convert",
                str(cube),
                "--output",
                str(cube_h5),
            ]
        )
        == 0
    )
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube_h5),
            "--output",
            str(output),
            "--water-padding-ang",
            "0.2",
        ]
    )

    assert rc == 0
    rows = _read_hdf5_rows(output)
    assert len(rows) == 1
    assert rows[0]["status"] in {"ok", "incomplete"}


def test_compute_potential_accepts_combined_cube_hdf5_input(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    cube_a = _write_potential_case(run_a, fermi_au=0.0367493036)
    cube_b = _write_potential_case(run_b, fermi_au=0.0734986072)
    combined = tmp_path / "combined.cube.h5"
    output = tmp_path / "potential_summary.potential.h5"

    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "apply",
                "combine",
                "-f",
                str(cube_a),
                str(cube_b),
                "--output",
                str(combined),
            ]
        )
        == 0
    )
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(combined),
            "--output",
            str(output),
            "--water-padding-ang",
            "0.2",
        ]
    )

    assert rc == 0
    rows = _read_hdf5_rows(output)
    assert len(rows) == 2


def test_plot_potential_profiles_preserve_explicit_blank_axis_labels(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_plot_multi_line_series(_x_series, _y_series, _labels, **kwargs):
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return tmp_path / "noop.png"

    monkeypatch.setattr(
        "linak.analysis.potential.plot_multi_line_series",
        _fake_plot_multi_line_series,
    )

    source = tmp_path / "potential_summary.h5"
    _write_potential_summary_hdf5(
        source,
        ids=[1, 2],
        efermi=[1.0, 1.1],
        water_bulk=[2.0, 2.1],
        cshe=[0.2, 0.3],
        status=["ok", "ok"],
    )
    profiles, _summary = load_potential_plot_profiles(source)

    plot_potential_profiles(profiles, show=False, x_label="", y_label="")

    assert captured["x_label"] == ""
    assert captured["y_label"] == ""


def test_plot_potential_hdf5_multi_source_non_gui_renders_png(tmp_path):
    source_a = tmp_path / "potential_a.h5"
    source_b = tmp_path / "potential_b.h5"
    output = tmp_path / "potential_overlay.png"
    _write_potential_summary_hdf5(
        source_a,
        ids=[1, 2, 3],
        efermi=[1.0, 1.1, 1.2],
        water_bulk=[2.0, 2.1, 2.2],
        cshe=[0.2, 0.3, 0.4],
        status=["ok", "ok", "ok"],
    )
    _write_potential_summary_hdf5(
        source_b,
        ids=[1, 2, 3],
        efermi=[1.5, 1.6, 1.7],
        water_bulk=[2.5, 2.6, 2.7],
        cshe=[0.6, 0.7, 0.8],
        status=["ok", "ok", "ok"],
    )

    plot_rc = main(
        [
            "--log-level",
            "ERROR",
            "plot",
            "-f",
            str(source_a),
            str(source_b),
            "--no-gui",
            "--no-show",
            "--output",
            str(output),
        ]
    )

    assert plot_rc == 0
    assert output.exists()


def test_compute_potential_writes_and_appends_hdf5(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    cube1 = _write_potential_case(run1, fermi_au=0.0367493036)  # about 1.0 eV
    cube2 = _write_potential_case(run2, fermi_au=0.0734986072)  # about 2.0 eV

    output_h5 = tmp_path / "potentials.potential.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            "-f",
            str(cube1),
            str(cube2),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
        ]
    )

    assert rc == 0
    assert output_h5.exists()
    with h5py.File(output_h5, "r") as handle:
        assert "created_utc" in handle.attrs
        assert "linak_version" in handle.attrs

    rows = _read_hdf5_rows(output_h5)
    assert len(rows) == 2
    assert {"efermi_ev", "water_bulk_potential_ev", "electrode_cshe_ev"}.issubset(rows[0].keys())

    by_source = {row["source"]: row for row in rows}
    row_1 = by_source[str(cube1)]
    row_2 = by_source[str(cube2)]

    assert sorted(int(row["id"]) for row in rows) == [1, 2]
    assert row_1["source_dir"] == "run1"
    assert row_2["source_dir"] == "run2"
    assert row_1["output_out"] == "output.out"
    assert row_2["output_out"] == "output.out"

    assert float(row_1["efermi_ev"]) == pytest.approx(1.0, abs=1e-3)
    assert float(row_1["water_bulk_potential_ev"]) == pytest.approx(2.0, abs=1e-6)
    assert float(row_1["electrode_cshe_ev"]) == pytest.approx(0.19, abs=1e-3)
    assert float(row_2["efermi_ev"]) == pytest.approx(2.0, abs=1e-3)

    rc_append = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube1),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
        ]
    )
    assert rc_append == 0
    rows_after = _read_hdf5_rows(output_h5)
    assert len(rows_after) == 2


def test_compute_potential_shows_loading_progress_for_raw_inputs(tmp_path, monkeypatch):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    cube1 = _write_potential_case(run1, fermi_au=0.0367493036)
    cube2 = _write_potential_case(run2, fermi_au=0.0734986072)
    output_h5 = tmp_path / "potentials.potential.h5"

    class RecordingProgressBar:
        instances: list["RecordingProgressBar"] = []

        def __init__(self, *, desc, total=None, unit="it", **_kwargs):
            self.desc = desc
            self.total = total
            self.unit = unit
            self.updates = 0
            self.entered = False
            self.closed = False
            RecordingProgressBar.instances.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            self.closed = True

        def update(self, n=1):
            self.updates += n

        @classmethod
        def prepare_for_external_write(cls, _stream):
            return None

    monkeypatch.setattr("linak.progress.ProgressBar", RecordingProgressBar)

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            "-f",
            str(cube1),
            str(cube2),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
        ]
    )

    assert rc == 0
    loading_bars = [
        bar for bar in RecordingProgressBar.instances if bar.desc == "Loading potential inputs"
    ]
    assert len(loading_bars) == 1
    loading_bar = loading_bars[0]
    assert loading_bar.total == 2
    assert loading_bar.unit == "file"
    assert loading_bar.entered
    assert loading_bar.closed
    assert loading_bar.updates == 2


def test_compute_potential_incompatible_hdf5_schema_uses_fallback_file(tmp_path):
    run_dir = tmp_path / "run"
    cube = _write_potential_case(run_dir, fermi_au=0.0367493036)

    output_h5 = tmp_path / "potentials.potential.h5"
    with h5py.File(output_h5, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["analysis"] = "unexpected"

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
        ]
    )

    assert rc == 0
    assert output_h5.exists()
    with h5py.File(output_h5, "r") as handle:
        assert handle.attrs["analysis"] == "unexpected"

    fallback_path = tmp_path / "potentials.linak.potential.h5"
    assert fallback_path.exists()
    assert not (tmp_path / "potentials.linak.potential_1.h5").exists()
    fallback_rows = _read_hdf5_rows(fallback_path)
    assert len(fallback_rows) == 1


def test_compute_potential_incompatible_hdf5_schema_versions_fallback_before_analysis_suffix(
    tmp_path,
):
    run_dir = tmp_path / "run"
    cube = _write_potential_case(run_dir, fermi_au=0.0367493036)

    output_h5 = tmp_path / "potentials.potential.h5"
    fallback_path = tmp_path / "potentials.linak.potential.h5"
    with h5py.File(output_h5, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["analysis"] = "unexpected"
    fallback_path.write_text("occupied", encoding="utf-8")

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
        ]
    )

    versioned_fallback = tmp_path / "potentials.linak_1.potential.h5"
    assert rc == 0
    assert versioned_fallback.exists()
    assert not (tmp_path / "potentials.linak.potential_1.h5").exists()
    fallback_rows = _read_hdf5_rows(versioned_fallback)
    assert len(fallback_rows) == 1


def test_compute_potential_strict_mode_returns_error_but_still_writes_hdf5(tmp_path):
    run_dir = tmp_path / "run_missing_fermi"
    cube = _write_potential_case(run_dir, fermi_au=None)

    output_h5 = tmp_path / "strict.potential.h5"
    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            str(cube),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
            "--strict",
        ]
    )

    assert rc == 1
    assert output_h5.exists()
    rows = _read_hdf5_rows(output_h5)
    assert len(rows) == 1
    assert rows[0]["status"] == "incomplete"


def test_compute_potential_persists_rows_before_post_compute_crash(tmp_path, monkeypatch):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    cube1 = _write_potential_case(run1, fermi_au=0.0367493036)
    cube2 = _write_potential_case(run2, fermi_au=0.0734986072)

    output_h5 = tmp_path / "partial.potential.h5"

    def _raise_after_compute(_records):
        raise RuntimeError("synthetic post-compute crash")

    monkeypatch.setattr(
        "linak.analysis.potential.summarize_potential_statistics", _raise_after_compute
    )

    rc = main(
        [
            "--log-level",
            "ERROR",
            "compute",
            "potential",
            "-f",
            str(cube1),
            str(cube2),
            "--output",
            str(output_h5),
            "--water-padding-ang",
            "0.2",
        ]
    )

    assert rc == 1
    assert output_h5.exists()
    rows = _read_hdf5_rows(output_h5)
    assert len(rows) == 2


def test_compute_potential_records_calls_callbacks_in_source_order(monkeypatch):
    def _fake_compute(source_item, *, config):
        assert isinstance(config, PotentialConfig)
        source = str(source_item)
        if source.endswith("bad.cube"):
            raise RuntimeError("synthetic failure")
        return PotentialRecord(
            id=None,
            source=source,
            source_dir="/tmp",
            output_out=None,
            efermi_ev=1.0,
            water_bulk_potential_ev=2.0,
            electrode_cshe_ev=0.19,
            status="ok",
            error=None,
        )

    monkeypatch.setattr("linak.analysis.potential.compute_potential_record", _fake_compute)

    seen_successes: list[str] = []
    seen_failures: list[str] = []

    records, failures = compute_potential_records(
        ["a.cube", "bad.cube", "c.cube"],
        config=PotentialConfig(),
        threads=1,
        on_record=lambda record: seen_successes.append(record.source),
        on_failure=lambda failure: seen_failures.append(failure.source),
    )

    assert [record.source for record in records] == ["a.cube", "c.cube"]
    assert seen_successes == ["a.cube", "c.cube"]
    assert [failure.source for failure in failures] == ["bad.cube"]
    assert seen_failures == ["bad.cube"]
    assert isinstance(failures[0], PotentialComputationFailure)


def test_resolve_worker_count_defaults_auto_to_single_worker():
    assert _resolve_worker_count(None, 1) == 1
    assert _resolve_worker_count(None, 5) == 1
