from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from linak.cli import main
from linak.analysis.potential import (
    HARTREE_TO_EV,
    PotentialComputationFailure,
    PotentialConfig,
    PotentialRecord,
    compute_potential_records,
)


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


def test_compute_potential_writes_and_appends_hdf5(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    cube1 = _write_potential_case(run1, fermi_au=0.0367493036)  # about 1.0 eV
    cube2 = _write_potential_case(run2, fermi_au=0.0734986072)  # about 2.0 eV

    output_h5 = tmp_path / "potentials.h5"
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


def test_compute_potential_incompatible_hdf5_schema_uses_fallback_file(tmp_path):
    run_dir = tmp_path / "run"
    cube = _write_potential_case(run_dir, fermi_au=0.0367493036)

    output_h5 = tmp_path / "potentials.h5"
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

    fallback_matches = sorted(tmp_path.glob("potentials_linak_potential*.h5"))
    assert fallback_matches
    fallback_rows = _read_hdf5_rows(fallback_matches[0])
    assert len(fallback_rows) == 1


def test_compute_potential_strict_mode_returns_error_but_still_writes_hdf5(tmp_path):
    run_dir = tmp_path / "run_missing_fermi"
    cube = _write_potential_case(run_dir, fermi_au=None)

    output_h5 = tmp_path / "strict.h5"
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

    output_h5 = tmp_path / "partial.h5"

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
