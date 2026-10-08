from __future__ import annotations

from itertools import product
import logging

from ase import Atoms
from ase.geometry import find_mic
import h5py
import numpy as np
import pytest

from linak.cell_geometry import (
    bin_volumes,
    cartesian_bounds,
    cell_angles,
    cell_matrix,
    intersection_volume,
    log_cell_geometry_summary,
    wrap_points,
    unwrap_positions,
)
from linak.pbc import apply_pbc_to_frames, extract_cell_from_cp2k_input
from linak.trajectory.io import (
    TrajectoryStoredMetadata,
    read_trajectory,
    read_trajectory_hdf5_metadata,
    write_trajectory,
)

PT553 = [13.859251, 10.7354, 40.7792, 90, 90, 97.4176]


@pytest.fixture
def cell():
    return cell_matrix(PT553)


def test_pt553_parse_and_geometry(tmp_path, cell):
    path = tmp_path / "input.inp"
    path.write_text(
        "&CELL\n ABC [angstrom] 13.859251 10.7354 40.7792\n"
        " ALPHA_BETA_GAMMA [deg] 90 90 97.4176\n&END CELL\n"
    )
    np.testing.assert_allclose(extract_cell_from_cp2k_input(path), cell)
    assert cell[1, 0] == pytest.approx(-1.385942395)
    assert cell[1, 1] == pytest.approx(10.645561368)
    assert np.linalg.det(cell) == pytest.approx(6016.543065260448)
    assert cartesian_bounds(cell)[0, 0] < -1.38
    np.testing.assert_allclose(cell_angles(cell), PT553[3:])


@pytest.mark.parametrize(
    ("matrix", "classification", "warning_count"),
    [
        (np.diag([10.0, 11.0, 20.0]), "axis-aligned orthorhombic", 0),
        (
            np.asarray([[10.0, 0.0, 0.0], [2.0, 11.0, 0.0], [0.0, 0.0, 20.0]]),
            "tilted prismatic",
            0,
        ),
        (
            np.asarray([[10.0, 0.0, 2.0], [0.0, 11.0, 0.0], [0.0, 0.0, 20.0]]),
            "general triclinic/rotated",
            1,
        ),
    ],
)
def test_geometry_summary_is_concise_and_reports_axis_support(
    caplog, matrix, classification, warning_count
):
    frames = [Atoms("Pt", cell=matrix, pbc=True) for _ in range(3)]
    caplog.set_level(logging.INFO)

    summary = log_cell_geometry_summary(
        frames,
        logger=logging.getLogger("linak.test.geometry"),
        context="Test",
        surface_axis="z",
        source="trajectory",
    )

    assert summary.classification == classification
    info_records = [record for record in caplog.records if record.levelno == logging.INFO]
    warning_records = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(info_records) == 1
    assert len(warning_records) == warning_count
    assert "constant over 3 frame(s)" in info_records[0].getMessage()
    assert "Z surface axis" in info_records[0].getMessage()


def test_cp2k_vectors_units_and_conflicts(tmp_path):
    path = tmp_path / "input.inp"
    path.write_text(
        "&FORCE_EVAL\n&SUBSYS\n&CELL\n"
        " A [bohr] 2 0 0\n B [bohr] 0 4 0\n C [bohr] 1 0 6\n"
        "&END CELL\n&END SUBSYS\n&END FORCE_EVAL\n"
    )
    from ase.units import Bohr

    np.testing.assert_allclose(
        extract_cell_from_cp2k_input(path),
        np.asarray([[2, 0, 0], [0, 4, 0], [1, 0, 6]]) * Bohr,
    )
    path.write_text("&CELL\n ABC 2 3 4\n A 2 0 0\n B 0 3 0\n C 0 0 4\n&END CELL\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        extract_cell_from_cp2k_input(path)


@pytest.mark.parametrize(
    "parameters",
    [
        [2, 3, 4, 70, 80, 100],
        [2, 3, 4, 90, 90, 45],
        [2, 3, 4, 100, 110, 120],
        [2, 3, 4],
    ],
)
def test_fractional_wrapping_and_minimum_images(parameters):
    matrix = cell_matrix(parameters)
    rng = np.random.default_rng(413)
    fractional = rng.uniform(0.01, 0.99, (30, 3))
    points = fractional @ matrix
    translated = points + rng.integers(-4, 5, (30, 3)) @ matrix
    np.testing.assert_allclose(wrap_points(translated, matrix), points, atol=1e-12)
    np.testing.assert_allclose(
        wrap_points(wrap_points(translated, matrix), matrix), points, atol=1e-12
    )
    deltas = points - points[0]
    mic, distances = find_mic(deltas, matrix, pbc=True)
    images = np.asarray(list(product(range(-3, 4), repeat=3))) @ matrix
    reference = np.linalg.norm(deltas[:, None] + images, axis=-1).min(axis=1)
    np.testing.assert_allclose(distances, reference, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(mic, axis=1), reference, atol=1e-12)


@pytest.mark.parametrize(
    "value",
    [
        [0, 1, 1],
        [1, 2, np.nan],
        [1, 2, 3, 0, 90, 90],
        [1, 2, 3, 180, 90, 90],
        [1, 2, 3, 10, 10, 170],
        [[1, 0, 0], [2, 0, 0], [0, 0, 1]],
    ],
)
def test_invalid_cells(value):
    with pytest.raises(ValueError):
        cell_matrix(value)


def test_clipped_volume_conservation_and_analytic_slabs(cell):
    for axis in range(3):
        edges = np.linspace(*cartesian_bounds(cell)[axis], 13)
        volumes = bin_volumes(cell, edges, axis)
        assert np.all(volumes > 0)
        assert volumes.sum() == pytest.approx(abs(np.linalg.det(cell)), rel=1e-10)
    edges = np.linspace(0, cell[2, 2], 9)
    np.testing.assert_allclose(bin_volumes(cell, edges, 2), np.linalg.det(cell) / 8)
    # A column wholly inside the parallelogram has volume dx * dy * height.
    bounds = [[2, 3], [2, 4], [-np.inf, np.inf]]
    assert intersection_volume(cell, bounds) == pytest.approx(2 * cell[2, 2])
    assert intersection_volume(cell, [[-20, -10], [0, 1], [0, 1]]) == 0


def test_rotated_cell_voxel_volume_conservation():
    matrix = cell_matrix([3, 4, 5, 72, 81, 104])
    angle = 0.42
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    matrix = matrix @ rotation
    bounds = cartesian_bounds(matrix)
    edges = [np.linspace(*pair, 4) for pair in bounds]
    total = sum(
        intersection_volume(matrix, [edge[i : i + 2] for edge, i in zip(edges, index)])
        for index in product(range(3), repeat=3)
    )
    assert total == pytest.approx(abs(np.linalg.det(matrix)), rel=1e-10)


@pytest.mark.parametrize("suffix", [".xyz", ".traj.h5"])
def test_geometry_roundtrip_with_origins_and_variable_angles(tmp_path, suffix, cell):
    frames = []
    for gamma in [97.4176, 105.0]:
        frame = Atoms(
            "Ar",
            scaled_positions=[[0.02, 0.98, 0.2]],
            cell=cell_matrix([*PT553[:3], 90, 90, gamma]),
            pbc=[True, True, False],
        )
        frame.set_celldisp([1, 2, 3])
        frame.positions += [1, 2, 3]
        frames.append(frame)
    path = tmp_path / ("trajectory" + suffix)
    metadata = TrajectoryStoredMetadata(cell_angstrom=cell, pbc_cell_angstrom=cell)
    write_trajectory(frames, path, metadata=metadata, raw_species_as_symbols=True)
    loaded = read_trajectory(path)
    for before, after in zip(frames, loaded):
        np.testing.assert_allclose(after.cell, before.cell)
        np.testing.assert_allclose(after.positions, before.positions)
        np.testing.assert_allclose(after.get_celldisp(), before.get_celldisp())
        np.testing.assert_array_equal(after.pbc, before.pbc)
    if suffix.endswith("h5"):
        with h5py.File(path) as handle:
            assert handle["metadata/cell_angstrom"].shape == (3,)
            assert handle["metadata/cell_matrix_angstrom"].shape == (3, 3)
        stored_metadata = read_trajectory_hdf5_metadata(path)
        assert stored_metadata is not None
        np.testing.assert_allclose(stored_metadata.cell_geometry, cell)


def test_conversion_preserves_varying_cells_and_pbc_masks(tmp_path):
    from linak.conversion import CONVERSION_REGISTRY, TrajectoryConversionOptions

    source = tmp_path / "varying.xyz"
    cells = [cell_matrix([4, 5, 6, 90, 90, 75]), cell_matrix([4, 5, 6, 90, 90, 110])]
    frames = [
        Atoms("Ar", scaled_positions=[[0.2, 0.3, 0.4]], cell=matrix, pbc=[True, True, False])
        for matrix in cells
    ]
    write_trajectory(frames, source)
    request = CONVERSION_REGISTRY.build_request(source, target_selector="hdf5")
    result = CONVERSION_REGISTRY.execute(request, options=TrajectoryConversionOptions())
    loaded = read_trajectory(result.output_path)
    for expected_cell, frame in zip(cells, loaded):
        np.testing.assert_allclose(frame.cell, expected_cell)
        np.testing.assert_array_equal(frame.pbc, [True, True, False])


def test_crossing_surface_unwrap(cell):
    fractional = np.array([[0.2, 0.98, 0.4], [0.2, 1.02, 0.4], [0.2, 1.06, 0.4]])
    positions = fractional @ cell
    frames = [Atoms("Ar", positions=wrap_points([p], cell), cell=cell, pbc=True) for p in positions]
    unwrapped = unwrap_positions(
        np.asarray([f.positions for f in frames]), [f.cell for f in frames]
    )
    np.testing.assert_allclose(
        unwrapped[:, 0] - unwrapped[0, 0], positions - positions[0], atol=1e-12
    )


def test_rdf_and_coordination_use_triclinic_minimum_image(cell):
    from linak.analysis import coordination as coordination_module
    from linak.analysis.rdf import compute_rdf

    frame = Atoms(
        "OH",
        scaled_positions=[[0.5, 0.98, 0.5], [0.5, 0.02, 0.5]],
        cell=cell,
        pbc=True,
    )
    expected_distance = 0.04 * np.linalg.norm(cell[1])
    profile = compute_rdf(
        [frame], species_a="O", species_b="H", r_max=1.0, bin_width=0.2, threads=1
    )
    assert profile.g_r[int(expected_distance // 0.2)] > 0
    values = coordination_module._compute_coordination_frame_values(
        frame,
        center_indices=np.array([0]),
        neighbor_indices=np.array([1]),
        same_selection=False,
        cutoff_A=0.6,
        smoothing_width_A=0.0,
    )
    np.testing.assert_allclose(values, [1.0])


def test_apply_pbc_is_copy_and_supports_angles(cell):
    frame = Atoms("Pt", positions=[[-4, 14, 45]])
    old = frame.positions.copy()
    result = apply_pbc_to_frames([frame], PT553)[0]
    np.testing.assert_allclose(result.cell, cell)
    np.testing.assert_allclose(result.positions, wrap_points(old, cell))
    np.testing.assert_array_equal(frame.positions, old)


def test_triclinic_density_direct_grid_and_reload(tmp_path, cell):
    from linak.analysis.density import (
        DensityGridProfile,
        compute_density_profile,
        compute_all_density_profiles,
        density_grid_to_line_profile,
        density_grid_to_heatmap_profile,
        save_density_profiles,
        load_density_grid_profiles,
    )

    fractional = np.asarray(list(product(np.linspace(0.025, 0.975, 10), repeat=3)))
    frame = Atoms("Ar" * len(fractional), scaled_positions=fractional, cell=cell, pbc=True)
    direct = compute_density_profile(
        [frame], species="Ar", axis="x", bin_width=2.0, binning="cell", surface_mode="none"
    )
    volumes = bin_volumes(cell, direct.bin_edges, 0)
    assert direct.entities_per_frame is not None
    assert direct.entities_per_frame.sum() == len(frame)
    assert np.nansum(direct.number_density * volumes / 1000) == pytest.approx(len(frame))
    assert direct.bin_edges[0] < 0
    profiles = compute_all_density_profiles(
        [frame], species="Ar", outputs="3d", grid_bin_width=2.0, surface_mode="none"
    )
    grid = next(p for p in profiles if isinstance(p, DensityGridProfile))
    assert grid.geometry_groups is not None
    line = density_grid_to_line_profile(grid, x_mode="x")
    np.testing.assert_allclose(
        line.number_density, direct.number_density, rtol=1e-6, equal_nan=True
    )
    path = tmp_path / "density.density.h5"
    save_density_profiles(profiles, path)
    reloaded = load_density_grid_profiles(path)[0]
    assert reloaded.geometry_groups is not None
    before = density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="y")
    after = density_grid_to_heatmap_profile(reloaded, x_axis="x", y_axis="y")
    np.testing.assert_allclose(
        after.number_density, before.number_density, rtol=1e-6, equal_nan=True
    )


def test_cli_angles_and_per_frame_pbc(tmp_path, cell):
    from linak.cli import main

    source = tmp_path / "source.xyz"
    output = tmp_path / "wrapped.xyz"
    write_trajectory([Atoms("Pt", positions=[[-1, 10, 4]])], source)
    assert (
        main(
            [
                "apply",
                "pbc",
                str(source),
                "--output",
                str(output),
                "--cell",
                *map(str, PT553[:3]),
                "--cell-angles",
                *map(str, PT553[3:]),
            ]
        )
        == 0
    )
    np.testing.assert_allclose(read_trajectory(output)[0].cell, cell)


def test_lammps_restricted_and_general_data(tmp_path):
    from linak.trajectory.lammps import extract_cell_from_lammps_data_file, parse_lammps_input

    path = tmp_path / "data"
    path.write_text("LAMMPS\n\n0 10 xlo xhi\n0 8 ylo yhi\n0 7 zlo zhi\n-2 1 3 xy xz yz\n")
    expected = [[10, 0, 0], [-2, 8, 0], [1, 3, 7]]
    np.testing.assert_allclose(extract_cell_from_lammps_data_file(path), expected)
    path.write_text("LAMMPS\n\n10 0 0 avec\n-2 8 0 bvec\n1 3 7 cvec\n0 0 0 abc origin\n")
    np.testing.assert_allclose(extract_cell_from_lammps_data_file(path), expected)
    input_path = tmp_path / "input.lmp"
    input_path.write_text("units metal\nboundary p f p\nread_data data\n")
    assert parse_lammps_input(input_path).boundary == (True, False, True)

    from linak.conversion import CONVERSION_REGISTRY, TrajectoryConversionOptions

    source = tmp_path / "source.xyz"
    write_trajectory([Atoms("Ar", positions=[[1, 2, 3]])], source)
    request = CONVERSION_REGISTRY.build_request(source, target_selector="hdf5")
    result = CONVERSION_REGISTRY.execute(
        request, options=TrajectoryConversionOptions(input_path=input_path)
    )
    converted = read_trajectory(result.output_path)[0]
    np.testing.assert_allclose(converted.cell, expected)
    np.testing.assert_array_equal(converted.pbc, [True, False, True])


def test_lammps_general_dump_preserves_vectors_origin_and_boundary_flags(tmp_path):
    path = tmp_path / "trajectory.dump"
    path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS abc origin pp ff pp\n"
        "10 0 0 1\n-2 8 0 2\n1 3 7 3\n"
        "ITEM: ATOMS id element xs ys zs\n1 Ar 0.25 0.5 0.75\n"
    )
    frame = read_trajectory(path)[0]
    expected_cell = np.asarray([[10, 0, 0], [-2, 8, 0], [1, 3, 7]], dtype=float)
    np.testing.assert_allclose(frame.cell, expected_cell)
    np.testing.assert_allclose(frame.get_celldisp(), [1, 2, 3])
    np.testing.assert_allclose(
        frame.positions, [np.array([0.25, 0.5, 0.75]) @ expected_cell + [1, 2, 3]]
    )
    np.testing.assert_array_equal(frame.pbc, [True, False, True])
