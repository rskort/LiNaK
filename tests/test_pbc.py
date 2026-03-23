import pytest
from ase import Atoms
import numpy as np

from linak.pbc import (
    apply_pbc_to_frames,
    extract_cell_from_lammps_input,
    extract_cell_from_simulation_input,
    extract_cell_from_cp2k_input,
    extract_frame_timestep_fs_from_lammps_input,
    extract_frame_timestep_fs_from_simulation_input,
    extract_frame_timestep_fs_from_cp2k_input,
    extract_trajectory_stride_md_from_cp2k_input,
    extract_timestep_fs_from_cp2k_input,
    find_unique_simulation_input,
    find_unique_cp2k_input,
    resolve_cell_dimensions,
)


def test_extract_cell_from_cp2k_input_parses_abc_line(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text(
        "&CELL\n  ABC [angstrom] 17.887 15.491 59.671\n&END CELL\n",
        encoding="utf-8",
    )

    cell = extract_cell_from_cp2k_input(cp2k_input)
    assert cell == pytest.approx((17.887, 15.491, 59.671))


def test_extract_cell_from_cp2k_input_accepts_orthorhombic_alpha_beta_gamma(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text(
        "&CELL\n  ABC [angstrom] 14.25 14.81 54.48\n  ALPHA_BETA_GAMMA [deg] 90 90 90\n&END CELL\n",
        encoding="utf-8",
    )

    cell = extract_cell_from_cp2k_input(cp2k_input)
    assert cell == pytest.approx((14.25, 14.81, 54.48))


def test_extract_cell_from_cp2k_input_rejects_non_orthorhombic_alpha_beta_gamma(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text(
        "&CELL\n  ABC [angstrom] 14.25 14.81 54.48\n  ALPHA_BETA_GAMMA [deg] 90 91 90\n&END CELL\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="orthorhombic cells only"):
        extract_cell_from_cp2k_input(cp2k_input)


def test_extract_cell_from_cp2k_input_raises_without_abc(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text("&CELL\n  ALPHA_BETA_GAMMA 90 90 90\n&END CELL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid 'ABC"):
        extract_cell_from_cp2k_input(cp2k_input)


def test_extract_timestep_fs_from_cp2k_input_parses_timestep_line(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text("TIMESTEP [fs] 0.5\n", encoding="utf-8")

    timestep_fs = extract_timestep_fs_from_cp2k_input(cp2k_input)
    assert timestep_fs == pytest.approx(0.5)


def test_extract_timestep_fs_from_cp2k_input_rejects_non_fs_unit(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text("TIMESTEP [ps] 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported TIMESTEP unit"):
        extract_timestep_fs_from_cp2k_input(cp2k_input)


def test_extract_trajectory_stride_md_from_cp2k_input_parses_each_md(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text(
        "&MOTION\n"
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

    stride_md = extract_trajectory_stride_md_from_cp2k_input(cp2k_input)
    assert stride_md == 5


def test_extract_trajectory_stride_md_from_cp2k_input_defaults_to_one(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text("TIMESTEP [fs] 0.5\n", encoding="utf-8")

    stride_md = extract_trajectory_stride_md_from_cp2k_input(cp2k_input)
    assert stride_md == 1


def test_extract_frame_timestep_fs_from_cp2k_input_multiplies_timestep_and_stride(tmp_path):
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text(
        "TIMESTEP [fs] 0.5\n&TRAJECTORY\n  &EACH\n    MD 5\n  &END EACH\n&END TRAJECTORY\n",
        encoding="utf-8",
    )

    frame_timestep_fs, md_timestep_fs, stride_md = extract_frame_timestep_fs_from_cp2k_input(
        cp2k_input
    )
    assert md_timestep_fs == pytest.approx(0.5)
    assert stride_md == 5
    assert frame_timestep_fs == pytest.approx(2.5)


def test_find_unique_cp2k_input_requires_exactly_one_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No CP2K .inp file"):
        find_unique_cp2k_input(tmp_path)

    (tmp_path / "a.inp").write_text("ABC 1 1 1\n", encoding="utf-8")
    assert find_unique_cp2k_input(tmp_path).name == "a.inp"

    (tmp_path / "b.inp").write_text("ABC 1 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple CP2K .inp files"):
        find_unique_cp2k_input(tmp_path)


def test_find_unique_simulation_input_supports_inp_and_lmp(tmp_path):
    with pytest.raises(FileNotFoundError, match="No simulation input file"):
        find_unique_simulation_input(tmp_path)

    (tmp_path / "a.lmp").write_text("units metal\n", encoding="utf-8")
    assert find_unique_simulation_input(tmp_path).name == "a.lmp"

    (tmp_path / "b.inp").write_text("ABC 1 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple simulation input files"):
        find_unique_simulation_input(tmp_path)


def test_extract_cell_from_lammps_input_uses_read_data_bounds(tmp_path):
    lammps_input = tmp_path / "input.lmp"
    data_file = tmp_path / "system.data"
    lammps_input.write_text("read_data system.data\n", encoding="utf-8")
    data_file.write_text(
        "LAMMPS data file\n\n"
        "2 atoms\n"
        "1 atom types\n\n"
        "0.0 17.887 xlo xhi\n"
        "0.0 15.491 ylo yhi\n"
        "0.0 59.671 zlo zhi\n",
        encoding="utf-8",
    )

    cell = extract_cell_from_lammps_input(lammps_input)
    assert cell == pytest.approx((17.887, 15.491, 59.671))


def test_extract_frame_timestep_fs_from_lammps_input_uses_units_and_dump_stride(tmp_path):
    lammps_input = tmp_path / "input.lmp"
    lammps_input.write_text(
        "units metal\n"
        "timestep 0.0005\n"
        "dump d1 all custom 10 lammps.dump id type element xu yu zu\n",
        encoding="utf-8",
    )

    frame_timestep_fs, md_timestep_fs, stride_md = extract_frame_timestep_fs_from_lammps_input(
        lammps_input
    )
    assert md_timestep_fs == pytest.approx(0.5)
    assert stride_md == 10
    assert frame_timestep_fs == pytest.approx(5.0)


def test_extract_cell_and_timestep_from_simulation_input_dispatch_lammps(tmp_path):
    lammps_input = tmp_path / "input.lmp"
    data_file = tmp_path / "system.data"
    lammps_input.write_text(
        "units metal\n"
        "timestep 0.001\n"
        "read_data system.data\n"
        "dump d1 all custom 20 lammps.dump id type element xu yu zu\n",
        encoding="utf-8",
    )
    data_file.write_text(
        "LAMMPS data file\n\n"
        "2 atoms\n"
        "1 atom types\n\n"
        "0.0 10.0 xlo xhi\n"
        "0.0 11.0 ylo yhi\n"
        "0.0 12.0 zlo zhi\n",
        encoding="utf-8",
    )

    cell = extract_cell_from_simulation_input(lammps_input)
    frame_timestep_fs, md_timestep_fs, stride_md = extract_frame_timestep_fs_from_simulation_input(
        lammps_input
    )

    assert cell == pytest.approx((10.0, 11.0, 12.0))
    assert md_timestep_fs == pytest.approx(1.0)
    assert stride_md == 20
    assert frame_timestep_fs == pytest.approx(20.0)


def test_resolve_cell_dimensions_prefers_explicit_cell(tmp_path):
    out = tmp_path / "out.xyz"
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "input.inp").write_text("ABC [angstrom] 2.0 2.0 2.0\n", encoding="utf-8")

    cell = resolve_cell_dimensions(output_path=out, cell=(3.0, 4.0, 5.0))
    assert cell == pytest.approx((3.0, 4.0, 5.0))


def test_resolve_cell_dimensions_rejects_input_and_cell_together(tmp_path):
    out = tmp_path / "out.xyz"
    cp2k_input = tmp_path / "input.inp"
    cp2k_input.write_text("ABC [angstrom] 2.0 2.0 2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Use either --input or --cell"):
        resolve_cell_dimensions(
            output_path=out,
            input_path=cp2k_input,
            cell=(3.0, 4.0, 5.0),
        )


def test_apply_pbc_to_frames_wraps_positions():
    frame = Atoms("H", positions=[[1.2, -0.1, 0.5]])
    wrapped = apply_pbc_to_frames([frame], (1.0, 1.0, 1.0))[0]

    assert wrapped.positions[0, 0] == pytest.approx(0.2)
    assert wrapped.positions[0, 1] == pytest.approx(0.9)
    assert wrapped.positions[0, 2] == pytest.approx(0.5)


def test_apply_pbc_to_frames_wraps_positions_for_non_cubic_orthorhombic_cell():
    frame = Atoms("H2", positions=[[3.7, -0.1, 5.2], [-1.1, 4.6, -0.2]])
    wrapped = apply_pbc_to_frames([frame], (2.0, 3.0, 4.0))[0]

    np.testing.assert_allclose(
        wrapped.positions,
        np.array([[1.7, 2.9, 1.2], [0.9, 1.6, 3.8]]),
        atol=1e-12,
    )
