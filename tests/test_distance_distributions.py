"""Physical densities and nearest-distance distributions must remain distinct."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_masses, atomic_numbers
from linak.analysis import density
from linak.analysis.surface import SurfaceEstimatorOptions, NearestSubstrateSearch
from linak import cli


@pytest.fixture(autouse=True)
def preserve_logging_configuration(monkeypatch):
    # CLI rendering tests must not replace pytest handlers or leak ERROR level.
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)


def frames():
    first = Atoms(
        "Pt2K2OHH",
        positions=[[1, 1, 1], [5, 5, 1], [1, 1, 3], [5, 5, 4], [4, 4, 6], [4.9, 4, 6], [4, 4.9, 6]],
        cell=[10, 10, 12],
        pbc=True,
    )
    first.set_constraint(FixAtoms(indices=[0]))
    second = first[[0, 1, 2, 4, 5, 6]]
    second.positions[2, 2] += 1
    return [first, second]


def profiles(source=None, **options):
    return density.compute_all_density_profiles(
        frames() if source is None else source,
        species="all",
        bin_width=0.5,
        grid_bin_width=0.5,
        min_molecule_frames=1,
        surface_layout="single-top",
        surface_elements=["Pt"],
        include_fixed_surface_atoms=True,
        surface_options=SurfaceEstimatorOptions(
            distance_mode="nearest-atom", surface_elements=("Pt",), include_fixed_surface_atoms=True
        ),
        **options,
    )


def test_number_and_mass_conservation_molecules_and_variable_frames():
    values = profiles()
    lines = {p.species: p for p in values if isinstance(p, density.DensityProfile)}
    assert all(p.coordinate_mode == "distance" for p in lines.values())
    for label, count, mass in [
        ("K", 1.5, atomic_masses[atomic_numbers["K"]]),
        ("mol:H2O", 1, 2 * atomic_masses[1] + atomic_masses[8]),
        ("Pt", 2, atomic_masses[78]),
    ]:
        line = lines[label]
        assert line.units == "amu/Angstrom"
        assert np.sum(line.number_density * np.diff(line.bin_edges)) == pytest.approx(count)
        assert np.sum(line.density * np.diff(line.bin_edges)) == pytest.approx(count * mass)
        assert line.number_density_units == (
            "molecules/Angstrom" if label.startswith("mol:") else "atoms/Angstrom"
        )
        # Counts remain grams internally; the observable and its errors use amu.
        np.testing.assert_allclose(
            line.density * np.diff(line.bin_edges) * density.AMU_TO_G, line.counts_per_frame
        )


def test_sparse_1d_2d_filters_conservation_no_volume_geometry(monkeypatch):
    monkeypatch.setattr(
        density,
        "_prepare_density_grid_geometry",
        lambda *a, **k: pytest.fail("unnecessary physical volume geometry"),
    )
    grids = [p for p in profiles() if isinstance(p, density.DensityGridProfile)]
    for grid in grids:
        assert grid.geometry_groups is None
        one = density.density_grid_to_line_profile(grid)
        two = density.density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="distance")
        assert two.units == "amu/Angstrom^2"
        np.testing.assert_allclose(
            two.number_density.sum(axis=0) * 0.5, one.number_density, atol=1e-12
        )
        np.testing.assert_allclose(two.density.sum(axis=0) * 0.5, one.density, atol=1e-12)
        filtered = density.density_grid_to_heatmap_profile(
            grid, x_axis="distance", y_axis="y", filters={"x": (0, 3)}
        )
        assert np.sum(filtered.number_density) * 0.25 <= np.sum(two.number_density) * 0.25 + 1e-12


def test_multisource_distributions_keep_amu_scale():
    grid = next(
        p for p in profiles() if isinstance(p, density.DensityGridProfile) and p.species == "K"
    )
    one = density.density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="distance")
    averaged = density.density_grids_to_averaged_heatmap_profile(
        [grid, grid], x_axis="x", y_axis="distance"
    )
    np.testing.assert_allclose(averaged.density, one.density)
    np.testing.assert_allclose(averaged.number_density, one.number_density)
    assert averaged.units == "amu/Angstrom^2"


@pytest.mark.parametrize(
    "axes", [("x", "y"), ("x", "z"), ("z", "distance"), ("distance", "distance")]
)
def test_invalid_axes_backend_and_averaging(axes):
    grid = next(p for p in profiles() if isinstance(p, density.DensityGridProfile))
    for call in (
        lambda: density.density_grid_to_heatmap_profile(grid, x_axis=axes[0], y_axis=axes[1]),
        lambda: density.density_grids_to_averaged_heatmap_profile(
            [grid], x_axis=axes[0], y_axis=axes[1]
        ),
    ):
        with pytest.raises(ValueError, match="axes|Distance|distance"):
            call()
    with pytest.raises(ValueError, match="Distance"):
        density.density_grid_to_line_profile(grid, x_mode="x")


def test_persistence_semantics_and_legacy_rejection(tmp_path):
    values = profiles()
    output = density.save_density_profiles(values, tmp_path / "test.density.h5")
    stored = density.load_density_profiles(output)
    original = [p for p in values if isinstance(p, density.DensityProfile)]
    for actual, expected in zip(stored, original):
        assert actual.units == expected.units
        np.testing.assert_allclose(actual.density, expected.density, rtol=1e-6)
    payload = density._density_profile_hdf5_payload(original[0])
    assert payload["metadata"]["observable_kind"] == "distance_distribution"
    assert payload["metadata"]["mass_unit"] == "amu"
    payload["metadata"].pop("distribution_version")
    with pytest.raises(ValueError, match="Legacy nearest-atom.*Recompute"):
        density._load_density_profiles_from_payloads(
            output, [(payload["datasets"], payload["metadata"])]
        )
    grid = next(p for p in values if isinstance(p, density.DensityGridProfile))
    heat = density.density_grid_to_heatmap_profile(
        grid, x_axis="x", y_axis="distance", filters={"y": (0, 8)}
    )
    file = density.save_density_profiles([heat], tmp_path / "heat.density.h5")
    actual = density.load_density_heatmap_profiles(file)[0]
    np.testing.assert_allclose(actual.density, heat.density, rtol=1e-6)
    assert actual.units == "amu/Angstrom^2"
    assert actual.spatial_filters == {"y": [0.0, 8.0]}
    assert actual.surface_metadata == heat.surface_metadata


def test_plane_density_unchanged():
    source = frames()
    plane = density.compute_density_profile(source, species="K", surface_mode="none", bin_width=0.5)
    assert plane.units == "g/cm^3"
    expected = 1.5 * atomic_masses[19] * density.AMU_TO_G / (100 * 0.5) * 1e24
    assert plane.density.sum() == pytest.approx(expected)


def test_mass_and_number_computed_with_shared_nearest_queries(monkeypatch):
    original = NearestSubstrateSearch.query
    counts = []

    def query(self, points):
        counts.append(len(points))
        return original(self, points)

    monkeypatch.setattr(NearestSubstrateSearch, "query", query)
    profiles()
    # One batch per frame in the range pass and one in accumulation, regardless
    # of species, molecule/element overlap, or requested weighting channels.
    assert len(counts) == 4
    assert max(counts) <= 8


def test_distribution_plot_labels_contract_and_export(tmp_path):
    from matplotlib import pyplot as plt
    from linak.plot.contracts.density_contract import density_profile_to_plot_data_contract

    values = profiles()
    one = next(p for p in values if isinstance(p, density.DensityProfile) and p.species == "K")
    contract = density_profile_to_plot_data_contract(one)
    assert (
        next(q for q in contract.quantities if q.id == "mass_density").label == "Mass distribution"
    )
    assert next(q for q in contract.quantities if q.id == "mass_density").unit == "amu/Angstrom"
    state: dict = {}
    density.plot_density_profile(one, quantity="mass", capture_state=state, show=False)
    try:
        assert "Mass distribution" in state["axes"].get_ylabel()
        assert "amu" in state["axes"].get_ylabel()
        assert "Nearest surface-atom distance" in state["axes"].get_xlabel()
    finally:
        export = cli._write_plotted_data_export(
            state, tmp_path / "curve.csv", include_metadata=True
        )
        assert "Mass distribution" in export.read_text(encoding="utf-8")
        assert "amu" in export.read_text(encoding="utf-8")
        plt.close(state["figure"])


def test_studio_prevents_invalid_nearest_axes_and_reopens_settings(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from linak.plot.data_contract import PLOT_VIEW_2D_HEATMAP

    app = QApplication.instance() or QApplication([])
    source = density.save_density_profiles(profiles(), tmp_path / "gui.density.h5")
    checked = []

    def inspect(application):
        application.processEvents()
        window = next(
            w
            for w in application.topLevelWidgets()
            if hasattr(w, "density_quantity") and w.isVisible()
        )
        try:
            window._preview_timer.stop()
            monkeypatch.setattr(window, "_schedule_preview_update", lambda *a, **k: None)
            assert window.density_x_mode.currentText() == "Nearest surface-atom distance"
            assert window.density_quantity.currentText() == "Mass distribution"
            assert (
                not window.density_x_mode.model()
                .item(window.density_x_mode.findData("x"))
                .isEnabled()
                if window.density_x_mode.findData("x") >= 0
                else True
            )
            window._set_combo_value(window.density_view_type, "2D Heatmap")
            application.processEvents()
            settings = window._collect_settings()
            assert {settings["density_2d_x_axis"], settings["density_2d_y_axis"]} in (
                {"x", "distance"},
                {"y", "distance"},
            )
            assert (
                not window.density_2d_x_axis.model()
                .item(window.density_2d_x_axis.findData("z"))
                .isEnabled()
            )
            window._set_combo_value(window.density_2d_y_axis, "y")
            window._refresh_widget_states()
            settings = window._collect_settings()
            assert "distance" in {settings["density_2d_x_axis"], settings["density_2d_y_axis"]}
            window._set_combo_value(window.density_quantity, "number")
            assert window._selected_density_quantity() == "number"
            assert window._current_density_mapping().view_type_id == PLOT_VIEW_2D_HEATMAP
            assert "Number distribution" in window.density_quantity.currentText()
            window._nav_list.setCurrentRow(1)
            window.resize(1100, 760)
            application.processEvents()
            assert window.grab().save(str(tmp_path / "studio.png"))
            checked.append(True)
        finally:
            window._saved_signature = window._signature(window._collect_settings())
            window.close()
            application.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect)
    assert cli.main(["--log-level", "ERROR", "plot", str(source), "--gui"]) == 0
    assert checked
    app.processEvents()


def test_cli_rejects_invalid_nearest_map_and_exports_correct_labels(tmp_path):

    values = profiles()
    source = density.save_density_profiles(values, tmp_path / "cli.density.h5")
    assert (
        cli.main(
            [
                "--log-level",
                "ERROR",
                "plot",
                str(source),
                "--no-gui",
                "--no-show",
                "--view-type",
                "2d",
                "--2d-x-axis",
                "x",
                "--2d-y-axis",
                "y",
            ]
        )
        != 0
    )

    assert (
        cli.main(
            [
                "--log-level",
                "ERROR",
                "plot",
                str(source),
                "--no-gui",
                "--no-show",
                "--view-type",
                "2d",
                "--2d-x-axis",
                "x",
                "--2d-y-axis",
                "distance",
                "--quantity",
                "number",
                "--species",
                "K",
                "--output",
                str(tmp_path / "valid.png"),
            ]
        )
        == 0
    )
    assert (tmp_path / "valid.png").is_file()


def test_symmetric_heatmap_preserves_sides_and_pooling_on_reopen(tmp_path):
    source = frames()
    for frame in source:
        frame.positions[1, 2] = 2
    values = density.compute_all_density_profiles(
        source,
        species="K",
        bin_width=0.5,
        grid_bin_width=0.5,
        surface_layout="symmetric",
        surface_elements=["Pt"],
        include_fixed_surface_atoms=True,
        surface_options=SurfaceEstimatorOptions(
            distance_mode="nearest-atom", surface_elements=("Pt",), include_fixed_surface_atoms=True
        ),
    )
    grid = next(p for p in values if isinstance(p, density.DensityGridProfile))
    heat = density.density_grid_to_heatmap_profile(grid, x_axis="distance", y_axis="x")
    np.testing.assert_allclose(heat.density, (heat.top_density + heat.bottom_density) / 2)
    payload = density._density_heatmap_profile_hdf5_payload(heat)
    assert payload["metadata"]["interface_pooling"] == "equal_interface_mean"
    output = density.save_density_profiles([heat], tmp_path / "sides.density.h5")
    restored = density.load_density_heatmap_profiles(output)[0]
    assert restored.interface_view == "folded"
    for name in (
        "density",
        "top_density",
        "bottom_density",
        "top_number_density",
        "bottom_number_density",
    ):
        np.testing.assert_allclose(getattr(restored, name), getattr(heat, name), rtol=1e-6)
    payload = density._density_grid_profile_hdf5_payload(grid)
    payload["metadata"].pop("distribution_version")
    with pytest.raises(ValueError, match="Legacy nearest-atom"):
        density._load_density_grid_profiles_from_payloads(
            output, [(payload["datasets"], payload["metadata"])]
        )
