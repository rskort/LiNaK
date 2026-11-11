from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import h5py
import numpy as np
import pytest

from linak.analysis.interface_alignment import (
    InterfaceAlignment,
    InterfaceReference,
    InterfaceSnapshot,
    align_position_profile,
    apply_interface_alignment,
    evaluate_interface_alignment,
    lattice_symmetries,
    suggest_interface_alignment,
    validate_alignment,
)
from linak.analysis.interface_density import align_density_grid, footprint_weights
from linak.analysis.density import (
    DensityGridProfile,
    compute_all_density_profiles,
    density_grid_to_heatmap_profile,
    load_density_grid_profiles,
    save_density_profiles,
)
from linak.analysis.position import (
    PositionProfile,
    compute_position_profile,
    load_position_profile,
    save_position_profile,
)
from linak.plot.profile_persistence import build_plot_profile_payload, flatten_plot_profile_payload
from tests.test_symmetric_surface import _symmetric_frame


def stepped_reference(
    matrix=((1, 0), (0, 1)), shift=(0.0, 0.0), *, tilted=False, origin=(0.0, 0.0, 0.0)
):
    cell = np.asarray(
        [[10.0, 0, 0], [3.0 if tilted else 0.0, 8.0 if tilted else 10.0, 0], [0, 0, 30.0]]
    )
    origin = np.asarray(origin)
    uv = np.asarray(
        [[0.12, 0.17], [0.33, 0.18], [0.69, 0.25], [0.83, 0.72], [0.21, 0.81], [0.49, 0.59]]
    )
    height = np.asarray([0.0, 0.0, -2.0, -2.0, -2.0, 0.0])
    bottom_uv = ((uv - shift) @ np.linalg.inv(matrix)) % 1
    top = np.column_stack([uv @ cell[:2, :2], 14 + height]) + origin
    bottom = np.column_stack([bottom_uv @ cell[:2, :2], 6 - height]) + origin
    snapshot = InterfaceSnapshot(
        0,
        np.arange(12),
        np.asarray(["Pt"] * 12),
        np.concatenate([top, bottom]),
        cell,
        origin,
        np.ones(3, bool),
        14.0,
        6.0,
        0.0,
        float(origin[2]),
        30.0,
    )
    return InterfaceReference("a" * 64, "z", cell[None], (snapshot,))


@pytest.mark.parametrize(
    "matrix,shift",
    [
        (((1, 0), (0, 1)), (0.173, 0.267)),
        (((-1, 0), (0, 1)), (0.31, 0.27)),
        (((0, 1), (-1, 0)), (0.2, 0.4)),
    ],
)
def test_suggestions_recover_known_stepped_registration(matrix, shift):
    reference = stepped_reference(matrix, shift, origin=(3.0, -4.0, 2.0))
    wrong = evaluate_interface_alignment(reference, InterfaceAlignment(reference.reference_id))
    assert not wrong.good_match
    suggestions = suggest_interface_alignment(reference)
    assert suggestions
    best = suggestions[0]
    assert best.diagnostics["good_match"]
    np.testing.assert_array_equal(best.matrix, matrix)
    np.testing.assert_allclose(
        np.mod(np.asarray(best.translation) - shift + 0.5, 1) - 0.5, 0, atol=1e-5
    )
    assert max(best.diagnostics["rmsd_A"]) < 1e-5


def test_terrace_heights_and_species_prevent_false_matches():
    reference = stepped_reference()
    snapshot = reference.snapshots[0]
    positions = snapshot.positions.copy()
    positions[6:9, 2] += 1
    elements = snapshot.elements.copy()
    elements[9] = "Au"
    changed = replace(
        reference, snapshots=(replace(snapshot, positions=positions, elements=elements),)
    )
    diagnostics = evaluate_interface_alignment(changed, InterfaceAlignment(reference.reference_id))
    assert not diagnostics.good_match
    assert diagnostics.top_coverage == pytest.approx((2 / 6,))
    assert len(set(diagnostics.matched_indices[0][1])) == 2


def test_unequal_populations_and_missing_side_remain_mismatches():
    reference = stepped_reference()
    snapshot = reference.snapshots[0]
    reference = replace(
        reference,
        snapshots=(
            replace(
                snapshot,
                positions=snapshot.positions[:-1],
                elements=snapshot.elements[:-1],
                atom_indices=snapshot.atom_indices[:-1],
            ),
        ),
    )
    diagnostics = evaluate_interface_alignment(
        reference, InterfaceAlignment(reference.reference_id)
    )
    assert diagnostics.top_coverage == pytest.approx((5 / 6,))
    assert diagnostics.bottom_coverage == (1.0,)
    assert not diagnostics.good_match


def test_fixed_registration_must_match_every_reference_snapshot():
    reference = stepped_reference()
    initial = reference.snapshots[0]
    moved = initial.positions.copy()
    moved[6:, 0] += 1.0
    reference = replace(
        reference, snapshots=(initial, replace(initial, frame_index=4, positions=moved))
    )
    diagnostics = evaluate_interface_alignment(
        reference, InterfaceAlignment(reference.reference_id)
    )
    assert diagnostics.top_coverage == (1.0, 0.0)
    assert diagnostics.bottom_coverage == (1.0, 0.0)
    assert not diagnostics.good_match


def test_tilted_metric_and_every_cell_are_validated():
    reference = stepped_reference(tilted=True)
    matrices = lattice_symmetries(reference)
    assert len(matrices) == 2
    validate_alignment(reference, InterfaceAlignment(reference.reference_id, ((-1, 0), (0, -1))))
    with pytest.raises(ValueError, match="incompatible"):
        validate_alignment(reference, InterfaceAlignment(reference.reference_id, ((-1, 0), (0, 1))))
    square = stepped_reference()
    assert len(lattice_symmetries(square)) == 8
    variable = replace(square, cells=np.stack([square.cells[0], reference.cells[0]]))
    assert len(lattice_symmetries(variable)) == 2
    with pytest.raises(ValueError, match="different interface reference"):
        validate_alignment(square, InterfaceAlignment("b" * 64))


def test_non_orthogonal_hexagonal_lattice_has_twelve_symmetries():
    reference = stepped_reference()
    cell = reference.cells[0].copy()
    cell[1] = (5.0, np.sqrt(75), 0.0)
    assert len(lattice_symmetries(replace(reference, cells=cell[None]))) == 12


@pytest.mark.parametrize("tilted,matrix", [(False, ((-1, 0), (0, 1))), (True, ((-1, 0), (0, -1)))])
def test_conservative_stencils_preserve_uniform_cell_coverage(tilted, matrix):
    reference = stepped_reference(tilted=tilted)
    basis = reference.cells[0, :2, :2]
    edges = (np.arange(0, 14.0 if tilted else 11.0), np.arange(0, 9.0 if tilted else 11.0))
    alignment = InterfaceAlignment(reference.reference_id, matrix, (0.173, 0.247))
    from linak.analysis.interface_density import _area, _rectangle

    polygon = np.asarray([[0, 0], basis[0], basis[0] + basis[1], basis[1]])
    before, after = (
        np.zeros((len(edges[0]) - 1, len(edges[1]) - 1)),
        np.zeros((len(edges[0]) - 1, len(edges[1]) - 1)),
    )
    for ix, iy in np.ndindex(before.shape):
        area = _area(
            _rectangle(
                polygon,
                np.asarray([edges[0][ix], edges[1][iy]]),
                np.asarray([edges[0][ix + 1], edges[1][iy + 1]]),
            )
        )
        before[ix, iy] = area
        if area <= 1e-12:
            continue
        stencil = footprint_weights((ix, iy), edges, edges, basis, np.zeros(2), alignment)
        assert sum(w for _, _, w in stencil) == pytest.approx(1.0)
        for x, y, weight in stencil:
            after[x, y] += area * weight
    np.testing.assert_allclose(after, before, atol=1e-9)


def _density_grid(frames):
    profiles = compute_all_density_profiles(
        frames,
        species="O",
        surface_layout="symmetric",
        surface_elements=["Pt"],
        outputs="all",
        bin_width=1.0,
        grid_bin_width=1.0,
        binning="cell",
    )
    return profiles, next(p for p in profiles if isinstance(p, DensityGridProfile))


def test_new_files_share_lazy_reference_and_legacy_has_no_fallback(tmp_path, monkeypatch):
    frames = [_symmetric_frame() for _ in range(7)]
    profiles, grid = _density_grid(frames)
    position = compute_position_profile(
        frames, species="O", surface_layout="symmetric", surface_elements=["Pt"]
    )
    assert position.interface_reference.reference_id == grid.interface_reference.reference_id
    assert len(grid.interface_reference.snapshots) == 5
    assert all(p.interface_reference is grid.interface_reference for p in profiles)
    path = tmp_path / "test.density.h5"
    save_density_profiles(profiles, path)
    with h5py.File(path) as handle:
        assert len(handle["interface_references"]) == 1
    # Ordinary loading must not read substrate atom arrays.
    original = h5py.Dataset.__array__

    def guarded_array(dataset, *args, **kwargs):
        assert not dataset.name.startswith("/interface_references/")
        return original(dataset, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset, "__array__", guarded_array)
        loaded = load_density_grid_profiles(path)[0]
        assert loaded.interface_reference.snapshots == ()
    assert len(loaded.interface_reference.materialize().snapshots) == 5
    ppath = tmp_path / "test.position.h5"
    save_position_profile(position, ppath)
    assert len(load_position_profile(ppath).interface_reference.materialize().snapshots) == 5
    legacy = replace(grid, interface_reference=None)
    legacy_path = tmp_path / "legacy.density.h5"
    save_density_profiles([legacy], legacy_path)
    assert load_density_grid_profiles(legacy_path)[0].interface_reference is None
    with pytest.raises(ValueError, match="Recompute"):
        align_density_grid(legacy, InterfaceAlignment(grid.interface_reference.reference_id))


def test_density_registration_conserves_counts_and_matches_direct_binning():
    frame = _symmetric_frame()
    frame.positions[np.asarray(frame.get_chemical_symbols()) == "O", :2] += 0.25
    _, grid = _density_grid([frame])
    reference = grid.interface_reference
    alignment = InterfaceAlignment(
        reference.reference_id, ((-1, 0), (0, 1)), (0.2, 0.1), allow_mismatches=True
    )
    registered = align_density_grid(grid, alignment)
    assert registered.mass_sum_g.sum() == pytest.approx(grid.mass_sum_g.sum(), rel=1e-7)
    assert registered.entity_sum.sum() == pytest.approx(grid.entity_sum.sum())
    assert registered.top_entity_sum.sum() == pytest.approx(grid.top_entity_sum.sum())
    assert registered.bottom_entity_sum.sum() == pytest.approx(grid.bottom_entity_sum.sum())
    altered = frame.copy()
    # The fixture's oxygen at z=1 is on the negative side.
    indices = np.flatnonzero(
        (np.asarray(frame.get_chemical_symbols()) == "O") & (frame.positions[:, 2] < 5)
    )
    altered.positions[indices] = apply_interface_alignment(
        frame.positions[indices], np.asarray(frame.cell), np.zeros(3), "z", alignment
    )
    _, direct = _density_grid([altered])
    actual = density_grid_to_heatmap_profile(
        grid,
        x_axis="x",
        y_axis="distance",
        interface_alignments={reference.reference_id: alignment.to_dict()},
    )
    expected = density_grid_to_heatmap_profile(direct, x_axis="x", y_axis="distance")
    np.testing.assert_allclose(actual.density, expected.density, rtol=1e-7, atol=1e-12)
    # No mutation of the reusable original accumulators.
    np.testing.assert_array_equal(grid.entity_sum, np.asarray([1.0, 1.0]))


def test_position_alignment_preserves_identity_and_breaks_seams_and_side_changes():
    reference = stepped_reference()
    profile = PositionProfile(
        "O",
        "z",
        np.asarray([42]),
        np.arange(4),
        np.arange(4.0),
        np.arange(4.0),
        np.arange(4.0) / 1000,
        np.asarray([[6.0], [8.5], [2.0], [3.0]]),
        np.ones((4, 1)),
        np.ones((4, 1)),
        np.ones((4, 1)),
        4,
        1,
        interface_side=np.asarray([[-1], [-1], [1], [0]]),
        interface_reference=reference,
        cell_matrices_angstrom=np.repeat(reference.cells, 4, axis=0),
        cell_origins_angstrom=np.zeros((4, 3)),
    )
    result = align_position_profile(
        profile, InterfaceAlignment(reference.reference_id, translation=(0.3, 0))
    )
    np.testing.assert_allclose(result.x[:3, 0], [9.0, 1.5, 2.0])
    assert np.isnan(result.x[3, 0])
    np.testing.assert_array_equal(result.atom_indices, [42])
    np.testing.assert_array_equal(result.periodic_crossings[:, 0], [False, True, True, True])
    np.testing.assert_array_equal(profile.x[:, 0], [6.0, 8.5, 2.0, 3.0])
    assert result.interface_alignment_metadata["excluded_ambiguous_samples"] == 1


def test_alignment_settings_round_trip_and_reject_invalid_records():
    reference = stepped_reference()
    alignment = InterfaceAlignment(reference.reference_id, translation=(0.1, 0.2))
    settings = {"interface_alignments": {reference.reference_id: alignment.to_dict()}}
    payload = build_plot_profile_payload("plot:position", settings)
    assert (
        flatten_plot_profile_payload("plot:position", payload)["interface_alignments"]
        == settings["interface_alignments"]
    )
    with pytest.raises(ValueError, match="Invalid"):
        InterfaceAlignment.from_dict({**alignment.to_dict(), "matrix": [[1, 0.1], [0, 1]]})


def test_dialog_requires_mismatch_acknowledgement_and_cancel_discards_changes(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from linak.plot.interface_alignment_dialog import InterfaceAlignmentDialog

    application = QApplication.instance() or QApplication([])
    reference = stepped_reference(shift=(0.33, 0.22))
    dialog = InterfaceAlignmentDialog([{"label": "Stepped slab", "reference": reference}], {})
    dialog.enabled_check.setChecked(True)
    deadline = time.monotonic() + 10
    while dialog._diagnostics is None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert dialog._diagnostics is not None
    assert not dialog._diagnostics.good_match
    assert not dialog.apply_button.isEnabled()
    dialog.mismatch_check.setChecked(True)
    assert dialog.apply_button.isEnabled()
    dialog.reject()
    assert dialog.result_alignments == {}
    assert dialog._cancel.is_set()
    dialog.deleteLater()
    application.processEvents()


def test_position_projection_export_uses_transformed_coordinates_and_values(tmp_path):
    import csv
    from linak.analysis.position import plot_position_profile
    from linak.cli import _write_plotted_data_export

    frames = [_symmetric_frame() for _ in range(3)]
    profile = compute_position_profile(
        frames, species="O", surface_layout="symmetric", surface_elements=["Pt"]
    )
    reference = profile.interface_reference
    alignment = InterfaceAlignment(
        reference.reference_id, ((-1, 0), (0, 1)), (0.2, 0.0), allow_mismatches=True
    )
    state: dict[str, Any] = {}
    plot_position_profile(
        profile,
        component="2d-projection",
        projection_x="x",
        projection_y="distance",
        projection_value="x",
        projection_filter_min=6.0,
        show=False,
        capture_state=state,
        interface_alignments={reference.reference_id: alignment.to_dict()},
    )
    assert len(state["plotted_projection"]) == 1
    samples = state["plotted_projection"][0]["samples"]
    np.testing.assert_allclose(samples[:, 2], 7.0)
    np.testing.assert_allclose(samples[:, 4], 7.0)
    assert state["interface_alignments"][reference.reference_id]["matrix"] == [[-1, 0], [0, 1]]
    path = _write_plotted_data_export(state, tmp_path / "position.csv", include_metadata=True)
    contents = path.read_text(encoding="utf-8")
    assert "# interface_alignments=" in contents
    rows = list(csv.DictReader(line for line in contents.splitlines() if not line.startswith("#")))
    assert len(rows) == 3
    assert all(float(row["x"]) == pytest.approx(7.0) for row in rows)


def test_density_averaging_and_export_preserve_transform_provenance(tmp_path):
    from linak.analysis.density import (
        density_grids_to_averaged_heatmap_profile,
        plot_density_profile,
    )
    from linak.cli import _write_plotted_data_export

    _, grid = _density_grid([_symmetric_frame()])
    reference = grid.interface_reference
    alignment = InterfaceAlignment(
        reference.reference_id, translation=(0.15, 0.05), allow_mismatches=True
    )
    settings = {reference.reference_id: alignment.to_dict()}
    single = density_grid_to_heatmap_profile(
        grid, x_axis="x", y_axis="distance", interface_alignments=settings
    )
    average = density_grids_to_averaged_heatmap_profile(
        [grid, grid], x_axis="x", y_axis="distance", interface_alignments=settings
    )
    np.testing.assert_allclose(average.density, single.density)
    state: dict[str, Any] = {}
    from linak.plot.mappings.density_mapping import density_plot_options_to_view_mapping

    plot_density_profile(
        average,
        show=False,
        capture_state=state,
        view_mapping=density_plot_options_to_view_mapping(view_type="heatmap_2d"),
    )
    assert state["interface_alignments"][reference.reference_id]["source_grid_resolution_A"] == [
        1.0,
        1.0,
        1.0,
    ]
    output = _write_plotted_data_export(state, tmp_path / "density.csv", include_metadata=True)
    assert "conservative periodic footprint overlap" in output.read_text(encoding="utf-8")


def test_identity_alignment_keeps_combined_in_plane_density_magnitude():
    _, grid = _density_grid([_symmetric_frame()])
    plain = density_grid_to_heatmap_profile(grid, x_axis="x", y_axis="y")
    alignment = InterfaceAlignment(grid.interface_reference.reference_id)
    registered = density_grid_to_heatmap_profile(
        grid,
        x_axis="x",
        y_axis="y",
        interface_alignments={alignment.reference_id: alignment.to_dict()},
    )
    np.testing.assert_allclose(registered.density, plain.density)


def test_lazy_density_catalog_and_slice_cache_include_alignment(tmp_path):
    from linak.cli import (
        build_plot_parser,
        _build_density_gui_lazy_catalog,
        _build_density_gui_logical_context,
    )
    from linak.plot.mappings.density_mapping import density_plot_options_to_view_mapping

    profiles, grid = _density_grid([_symmetric_frame()])
    path = tmp_path / "slab.density.h5"
    save_density_profiles(profiles, path)
    args = build_plot_parser(analysis="density").parse_args([str(path)])
    args.species = "O"
    args.view_mapping = density_plot_options_to_view_mapping(view_type="heatmap_2d")
    args.density_2d_x_axis = "x"
    args.density_2d_y_axis = "distance"
    slices: dict[str, list[Any]] = {}
    sources: dict[str, Any] = {}
    results = []
    for shift in (0.1, 0.2):
        alignment = InterfaceAlignment(
            grid.interface_reference.reference_id, translation=(shift, 0), allow_mismatches=True
        )
        args.interface_alignments = {alignment.reference_id: alignment.to_dict()}
        catalog = _build_density_gui_lazy_catalog(
            args,
            sources=[str(path)],
            density_grid_profile_cache=sources,
            density_grid_slice_cache=slices,
        )
        assert catalog.build_initial_context().profile_filter_options["interface_references"][0][
            "can_align"
        ]
        context = catalog.build_render_context(args)
        results.append(context.profile[0].density.copy())
    assert len(slices) == 2
    assert not np.array_equal(results[0], results[1])
    logical = _build_density_gui_logical_context(args, sources=[str(path)])
    assert logical.profile_filter_options["interface_references"][0]["can_align"]


def test_atom_series_extraction_preserves_matching_side_column():
    from linak.cli import (
        _extract_position_profile_atom_series,
        _split_position_profile_into_atom_series,
    )

    profile = compute_position_profile(
        [_symmetric_frame()], species="O", surface_layout="symmetric", surface_elements=["Pt"]
    )
    child = _extract_position_profile_atom_series(profile, int(profile.atom_indices[1]))
    assert child.interface_side.shape == (1, 1)
    assert child.interface_side[0, 0] == 1
    assert child.distance_to_top_surface.shape == (1, 1)
    assert all(
        p.interface_side.shape == (1, 1) for p in _split_position_profile_into_atom_series(profile)
    )


def test_lazy_position_catalog_loads_geometry_for_registered_projection(tmp_path):
    from linak.cli import build_plot_parser, _build_position_gui_lazy_catalog
    from linak.analysis.position import plot_position_profiles
    from linak.plot.mappings.position_mapping import position_plot_options_to_view_mapping

    frames = [_symmetric_frame() for _ in range(2)]
    profile = compute_position_profile(
        frames, species="O", surface_layout="symmetric", surface_elements=["Pt"]
    )
    path = tmp_path / "slab.position.h5"
    save_position_profile(profile, path)
    args = build_plot_parser(analysis="position").parse_args([str(path)])
    args.species = "O"
    args.view_mapping = position_plot_options_to_view_mapping(
        component="2d-projection",
        projection_x="x",
        projection_y="distance",
        projection_render_mode="line-colors",
    )
    alignment = InterfaceAlignment(
        profile.interface_reference.reference_id, translation=(0.2, 0), allow_mismatches=True
    )
    args.interface_alignments = {alignment.reference_id: alignment.to_dict()}
    context = _build_position_gui_lazy_catalog(args, sources=[str(path)]).build_render_context(args)
    state: dict[str, Any] = {}
    plot_position_profiles(
        context.profile, show=False, capture_state=state, **context.plotter_kwargs
    )
    assert len(state["plotted_projection"]) == 2
    assert sorted(float(item["samples"][0, 2]) for item in state["plotted_projection"]) == [
        5.0,
        7.0,
    ]
    assert (
        state["interface_alignments"][alignment.reference_id]["reference_id"]
        == alignment.reference_id
    )


def test_dialog_discards_stale_worker_results(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from linak.plot.interface_alignment_dialog import InterfaceAlignmentDialog

    application = QApplication.instance() or QApplication([])
    reference = stepped_reference()
    dialog = InterfaceAlignmentDialog([{"label": "Slab", "reference": reference}], {})
    dialog._results.put((dialog._generation - 1, False, reference, None, "stale failure"))
    dialog._poll_results()
    assert "stale failure" not in dialog.diagnostics_label.text()
    dialog.reject()
    dialog.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("legacy", [False, True])
def test_studio_panel_exposes_alignment_capability_without_loading_snapshots(monkeypatch, legacy):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMainWindow
    from linak.plot.plot_gui import launch_plot_settings_panel

    application = QApplication.instance() or QApplication([])
    reference = stepped_reference()
    checked = []

    def inspect(window):
        checked.append(window.interface_alignment_button.isEnabled())
        assert window.interface_alignment_status.text() == "Off"
        if legacy:
            assert "Recompute this density/position" in window.interface_alignment_note.text()

    monkeypatch.setattr(QMainWindow, "showMaximized", inspect)
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    references = (
        []
        if legacy
        else [
            {
                "metadata": {"interface_reference": reference.metadata()},
                "source_path": "new.position.h5",
                "label": "Slab",
                "can_align": True,
            }
        ]
    )
    launch_plot_settings_panel(
        title="Alignment test",
        initial_settings={
            "_profile_filter_options": {
                "surface_layout_resolved": "symmetric",
                "interface_references": references,
            }
        },
        analysis_name="position",
        on_preview=lambda settings: {},
        on_save=lambda name, settings: "Saved",
    )
    assert checked == [not legacy]
    application.processEvents()
