"""Exact fast-path oracles, bounded work, and GUI source/registration reuse."""

from dataclasses import replace
from itertools import product
import os

import numpy as np
import pytest

from linak.cache import PREVIEW_CACHE
from linak.cancellation import ComputationCancelled, cancellation_scope
from linak.analysis.interface_alignment import InterfaceAlignment
from linak.analysis.interface_density import footprint_weights
from linak.analysis.interface_remap import transfer_operator
from linak.analysis.orientation_spatial import spatial_orientation_map, aligned_orientation_profile
from linak.analysis.orientation import save_orientation_profile
from linak.analysis.orientation_cache import load_gui_orientation_profiles
from tests.test_orientation_alignment import _frame, _profile, _setting
from tests.support.orientation_performance import orientation_case


@pytest.fixture(autouse=True)
def cache_isolation():
    budget = PREVIEW_CACHE.budget_bytes
    PREVIEW_CACHE.clear()
    yield
    PREVIEW_CACHE.clear()
    PREVIEW_CACHE.set_budget_mib(budget / (1024 * 1024))


@pytest.mark.parametrize("matrix", [((1, 0), (0, 1)), ((-1, 0), (0, 1)), ((0, 1), (-1, 0))])
@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.125, 0.25), (0.173, 0.927)])
def test_analytical_stencils_match_scalar_polygon_oracle(matrix, shift):
    basis = np.diag([10.0, 10.0])
    origin = np.array([3.7, -4.2])
    sources = (np.arange(3.3, 15.0, 1.3), np.arange(-4.6, 7.0, 0.7))
    targets = (np.arange(3.0, 15.0, 0.8), np.arange(-5.0, 7.0, 1.1))
    alignment = InterfaceAlignment("a" * 64, matrix=matrix, translation=shift)
    operator = transfer_operator(sources, targets, basis, origin, alignment)
    expected = {}
    bins = []
    for ix, iy in product(range(len(sources[0]) - 1), range(len(sources[1]) - 1)):
        if min(sources[0][ix + 1], origin[0] + 10) <= max(sources[0][ix], origin[0]):
            continue
        if min(sources[1][iy + 1], origin[1] + 10) <= max(sources[1][iy], origin[1]):
            continue
        bins.append(ix * (len(sources[1]) - 1) + iy)
        for x, y, weight in footprint_weights((ix, iy), sources, targets, basis, origin, alignment):
            expected[len(bins) - 1, x * (len(targets[1]) - 1) + y] = weight
    actual = {}
    for owners, destinations, weights in operator.expand(np.asarray(bins), max_entries=3):
        assert len(owners) <= 3
        for row, target, weight in zip(owners, destinations, weights):
            actual[int(row), int(target)] = float(weight)
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(
        list(actual[key] for key in expected), list(expected.values()), atol=1e-12, rtol=1e-10
    )


def test_coincident_stencil_is_exact_permutation_and_fractional_wrap_is_conservative():
    edges = (np.linspace(0.0, 12.0, 5),) * 2
    alignment = InterfaceAlignment("a" * 64, matrix=((0, 1), (-1, 0)), translation=(0.25, 0.5))
    operator = transfer_operator(edges, edges, np.diag([12.0, 12.0]), np.zeros(2), alignment)
    batches = list(operator.expand(np.arange(16), max_entries=7))
    np.testing.assert_array_equal(np.concatenate([b[2] for b in batches]), np.ones(16))
    assert len(np.unique(np.concatenate([b[1] for b in batches]))) == 16
    shifted = replace(alignment, matrix=((1, 0), (0, 1)), translation=(0.125, 0.125))
    operator = transfer_operator(edges, edges, np.diag([12.0, 12.0]), np.zeros(2), shifted)
    batches = list(operator.expand(np.array([15]), max_entries=1))
    np.testing.assert_allclose(np.concatenate([b[2] for b in batches]), [0.25] * 4)
    assert set(np.concatenate([b[1] for b in batches])) == {0, 3, 12, 15}


def test_tilted_stencil_uses_cached_exact_polygon_fallback(monkeypatch):
    import linak.analysis.interface_density as density

    basis = np.array([[10.0, 0.0], [3.0, 8.0]])
    origin = np.array([2.0, -3.0])
    edges = (np.arange(2.0, 16.0, 1.0), np.arange(-3.0, 6.0, 1.0))
    alignment = InterfaceAlignment("a" * 64, matrix=((-1, 0), (0, -1)), translation=(0.31, 0.27))
    bins, expected = [], {}
    for ix, iy in product(range(13), range(8)):
        try:
            stencil = footprint_weights((ix, iy), edges, edges, basis, origin, alignment)
        except ValueError as error:
            assert "no physical" in str(error)
            continue
        bins.append(ix * 8 + iy)
        for x, y, weight in stencil:
            expected[len(bins) - 1, x * 8 + y] = weight
    operator = transfer_operator(
        edges, edges, basis, origin, alignment, source_bins=np.asarray(bins)
    )
    actual = {}
    for owners, targets, weights in operator.expand(np.asarray(bins), 11):
        actual.update({(int(r), int(t)): w for r, t, w in zip(owners, targets, weights)})
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(
        [actual[key] for key in expected], list(expected.values()), atol=1e-12
    )
    monkeypatch.setattr(
        density, "footprint_weights", lambda *a, **kw: pytest.fail("cached fallback rebuilt")
    )
    assert (
        transfer_operator(edges, edges, basis, origin, alignment, source_bins=np.asarray(bins))
        is operator
    )


def test_orthorhombic_maps_do_not_call_polygon_clipping(monkeypatch):
    import linak.analysis.interface_density as density
    import linak.analysis.density as normalization

    profile = orientation_case(24, 6)
    monkeypatch.setattr(
        density, "footprint_weights", lambda *a, **kw: pytest.fail("scalar registration")
    )
    monkeypatch.setattr(
        normalization,
        "_polygon_rectangle_intersection_area",
        lambda *a, **kw: pytest.fail("scalar volume"),
    )
    settings = _setting(profile, translation=(0.173, 0.267))
    mean = spatial_orientation_map([profile], interface_alignments=settings)
    rho = spatial_orientation_map([profile], quantity="density", interface_alignments=settings)
    weighted = spatial_orientation_map(
        [profile], quantity="density-weighted", interface_alignments=settings
    )
    np.testing.assert_allclose(weighted.values, mean.values * rho.values, equal_nan=True)


def test_source_and_aligned_cache_reuse_are_independent_of_display_settings(tmp_path, monkeypatch):
    import linak.analysis.orientation as orientation
    import linak.analysis.interface_density as density

    profile = _profile([_frame()])
    path = tmp_path / "source.orientation.h5"
    save_orientation_profile(profile, path)
    real_load = orientation.load_orientation_profiles_by_index
    calls = []

    def load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(orientation, "load_orientation_profiles_by_index", load)
    import h5py

    with h5py.File(path) as handle:
        names = tuple(name for name, value in handle.items() if isinstance(value, h5py.Dataset))
    loaded = load_gui_orientation_profiles(path, [0], names)[0]
    assert not loaded.sparse_grid.entity_sum.flags.writeable
    settings = _setting(profile)
    reference = spatial_orientation_map([loaded], interface_alignments=settings)
    assert load_gui_orientation_profiles(path, [0], names)[0] is loaded
    assert len(calls) == 1
    real_remap = density.align_density_grid
    monkeypatch.setattr(
        density, "align_density_grid", lambda *a, **kw: pytest.fail("cached registration rebuilt")
    )
    for quantity in ("average", "density", "density-weighted"):
        result = spatial_orientation_map(
            [loaded],
            quantity=quantity,
            angle="azimuthal",
            filters={"x": (0.0, 6.0)},
            interface_alignments=settings,
        )
        assert result.alignments == reference.alignments
    # Ordinary Python profiles intentionally remain mutable and uncached.
    with pytest.raises(pytest.fail.Exception, match="cached registration rebuilt"):
        spatial_orientation_map([profile], interface_alignments=settings)
    monkeypatch.setattr(density, "align_density_grid", real_remap)
    changed = spatial_orientation_map(
        [loaded], interface_alignments=_setting(profile, translation=(0.13, 0.2))
    )
    assert changed.alignments != reference.alignments
    assert len(calls) == 1
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert load_gui_orientation_profiles(path, [0], names)[0] is not loaded
    assert len(calls) == 2


def test_fit_cache_validates_lazy_files_but_never_trusts_saved_diagnostics(tmp_path, monkeypatch):
    import linak.analysis.interface_alignment as alignment_module
    from linak.analysis.orientation import load_orientation_profile

    profile = _profile([_frame()])
    path = tmp_path / "fit.orientation.h5"
    save_orientation_profile(profile, path)
    lazy = load_orientation_profile(path)
    transform = InterfaceAlignment(profile.interface_reference.reference_id, depth_A=1.5)
    settings = {transform.reference_id: transform.to_dict()}
    real = alignment_module.evaluate_interface_alignment
    calls = []

    def evaluate(*args):
        calls.append(True)
        return real(*args)

    monkeypatch.setattr(alignment_module, "evaluate_interface_alignment", evaluate)
    for _ in range(2):
        alignment_module.alignment_for_reference(lazy.interface_reference, settings)
    assert len(calls) == 1
    transform = replace(transform, translation=(0.173, 0.267), diagnostics={"good_match": True})
    with pytest.raises(ValueError, match="acknowledge"):
        alignment_module.alignment_for_reference(
            lazy.interface_reference, {transform.reference_id: transform.to_dict()}
        )
    assert len(calls) == 2


def test_cancelled_transfer_and_alignment_do_not_publish_partial_results(tmp_path, monkeypatch):
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) > 3

    edges = (np.linspace(0.0, 12.0, 65),) * 2
    with pytest.raises(ComputationCancelled), cancellation_scope(cancelled):
        transfer_operator(
            edges,
            edges,
            np.diag([12.0, 12.0]),
            np.zeros(2),
            InterfaceAlignment("a" * 64, translation=(0.17, 0.23)),
        )
    assert PREVIEW_CACHE.stats()["entry_count"] == 0
    import h5py
    import linak.analysis.interface_density as density

    profile = _profile([_frame()])
    path = tmp_path / "cancel.orientation.h5"
    save_orientation_profile(profile, path)
    with h5py.File(path) as handle:
        names = tuple(name for name, value in handle.items() if isinstance(value, h5py.Dataset))
    loaded = load_gui_orientation_profiles(path, [0], names)[0]
    real = density.align_density_grid
    stop = []

    def cancel_after_remap(*args, **kwargs):
        result = real(*args, **kwargs)
        stop.append(True)
        return result

    monkeypatch.setattr(density, "align_density_grid", cancel_after_remap)
    with pytest.raises(ComputationCancelled), cancellation_scope(lambda: bool(stop)):
        aligned_orientation_profile(loaded, ("x", "y"), interface_alignments=_setting(profile))
    assert not any(key[0] == "orientation.aligned" for key in PREVIEW_CACHE._entries)


def test_vector_expansion_batches_respect_memory_policy(tmp_path, monkeypatch):
    import tempfile
    import linak.analysis.interface_density as density
    from linak.analysis.interface_remap import expansion_budget

    monkeypatch.setenv("LINAK_MEMORY_BUDGET_MIB", ".125")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    sizes = []
    original = density.SparseReducer.add

    def add(self, rows):
        sizes.append(rows.nbytes)
        return original(self, rows)

    monkeypatch.setattr(density.SparseReducer, "add", add)
    profile = orientation_case(12, 3)
    aligned = aligned_orientation_profile(
        profile, ("x", "y"), interface_alignments=_setting(profile, translation=(0.173, 0.267))
    )
    assert max(sizes) <= expansion_budget()
    assert aligned.sparse_grid.bottom_entity_sum.sum() == pytest.approx(
        profile.sparse_grid.bottom_entity_sum.sum()
    )


def test_lru_eviction_does_not_change_transfer_results():
    PREVIEW_CACHE.set_budget_mib(0.001)
    edges = (np.linspace(0.0, 12.0, 65),) * 2
    alignment = InterfaceAlignment("a" * 64, translation=(0.173, 0.267))
    first = transfer_operator(edges, edges, np.diag([12.0, 12.0]), np.zeros(2), alignment)
    second = transfer_operator(edges, edges, np.diag([12.0, 12.0]), np.zeros(2), alignment)
    assert first is not second
    assert PREVIEW_CACHE.current_bytes <= PREVIEW_CACHE.budget_bytes
    a = list(first.expand(np.array([0, 63, 4095]), 4))
    b = list(second.expand(np.array([0, 63, 4095]), 4))
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            np.testing.assert_array_equal(x, y)


def test_gui_edits_reuse_data_and_registration_and_notice_changed_secondary_source(
    tmp_path, monkeypatch
):
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import linak.cli as cli
    import linak.analysis.orientation as orientation
    import linak.analysis.interface_density as density
    import linak.analysis.interface_alignment as matching
    from linak.plot.mappings.orientation_mapping import orientation_plot_options_to_view_mapping
    from linak.plot.profile_persistence import serialize_plot_view_mapping

    first, second = _profile([_frame()]), _profile([_frame()])
    paths = [tmp_path / "first.orientation.h5", tmp_path / "second.orientation.h5"]
    for profile, path, uid in zip((first, second), paths, ("first", "second")):
        save_orientation_profile(profile, path, additional_metadata={"profile_uid": uid})
    real_load = orientation.load_orientation_profiles_by_index
    real_remap = density.align_density_grid
    real_fit = matching.evaluate_interface_alignment
    loads, remaps, fits = [], [], []

    def load(*args, **kwargs):
        loads.append(str(args[0]))
        return real_load(*args, **kwargs)

    def remap(*args, **kwargs):
        remaps.append(True)
        return real_remap(*args, **kwargs)

    def fit(*args, **kwargs):
        fits.append(True)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(orientation, "load_orientation_profiles_by_index", load)
    monkeypatch.setattr(density, "align_density_grid", remap)
    monkeypatch.setattr(matching, "evaluate_interface_alignment", fit)
    real_combine = cli._combine_analysis_hdf5_sources

    def combine(*, sources, analysis, output):
        return real_combine(
            sources=sources, analysis=analysis, output=tmp_path / "combined.orientation.h5"
        )

    monkeypatch.setattr(cli, "_combine_analysis_hdf5_sources", combine)
    checked = []

    def gui(**kwargs):
        settings = deepcopy(kwargs["initial_settings"])
        transform = InterfaceAlignment(first.interface_reference.reference_id, depth_A=1.5)
        settings["interface_alignments"] = {transform.reference_id: transform.to_dict()}

        def preview(angle="polar", quantity="average", **changes):
            settings.update(
                orientation_heatmap_kind="spatial",
                orientation_heatmap_quantity=quantity,
                angle=angle,
                view_mapping=serialize_plot_view_mapping(
                    orientation_plot_options_to_view_mapping(
                        orientation_heatmap_kind="spatial",
                        orientation_heatmap_quantity=quantity,
                        angle=angle,
                    )
                ),
                **changes,
            )
            state = kwargs["on_preview_figure"](settings)
            values = state["plotted_heatmap"]["values"].copy()
            plt.close(state["figure"])
            return values

        original = preview()
        counts = len(loads), len(remaps), len(fits)
        assert counts[1:] == (2, 2)
        import h5py

        # Guard actual file opens, including metadata and lazy substrate snapshots.
        with monkeypatch.context() as warm:
            warm.setattr(h5py, "File", lambda *a, **kw: pytest.fail("warm render opened HDF5"))
            preview("azimuthal", "density-weighted", heatmap_cmap="viridis")
            density_values = preview(
                "polar", "density", orientation_filter_x_min=0.0, orientation_filter_x_max=6.0
            )
            _, exported = kwargs["on_save_data"](settings, str(tmp_path / "warm.csv"))
            np.testing.assert_allclose(
                exported["plotted_heatmap"]["values"], density_values, equal_nan=True
            )
            plt.close(exported["figure"])
        assert (len(loads), len(remaps), len(fits)) == counts
        settings.update(orientation_filter_x_min=None, orientation_filter_x_max=None)
        changed_frame = _frame()
        for start in (15, 18, 21):
            changed_frame.positions[start + 1 : start + 3, 2] = (
                changed_frame.positions[start, 2] - 0.65
            )
        changed = _profile([changed_frame])
        save_orientation_profile(changed, paths[1], additional_metadata={"profile_uid": "second"})
        new = preview()
        assert not np.allclose(new, original, equal_nan=True)
        assert len(loads) == counts[0] + 1
        assert loads[-1] == str(paths[1])
        assert len(remaps) == counts[1] + 1
        assert len(fits) == counts[2] + 1
        checked.append(True)

    monkeypatch.setattr(cli, "_open_plot_settings_gui", gui)
    assert (
        cli.main(
            ["--log-level", "ERROR", "plot", "--files", *(str(path) for path in paths), "--gui"]
        )
        == 0
    )
    assert checked == [True]


@pytest.mark.parametrize("signs", [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_signed_cell_axes_match_polygon_oracle(signs):
    basis = np.diag(np.array(signs) * 12.0)
    origin = np.array([2.0, -3.0])
    lower = origin + np.minimum(np.diag(basis), 0.0)
    edges = tuple(np.linspace(x, x + 12.0, 7) for x in lower)
    alignment = InterfaceAlignment("a" * 64, matrix=((0, 1), (-1, 0)), translation=(0.173, 0.267))
    operator = transfer_operator(edges, edges, basis, origin, alignment)
    actual = {}
    for owners, targets, weights in operator.expand(np.arange(36), 5):
        actual.update({(int(r), int(t)): w for r, t, w in zip(owners, targets, weights)})
    expected = {}
    for ix, iy in product(range(6), repeat=2):
        for x, y, w in footprint_weights((ix, iy), edges, edges, basis, origin, alignment):
            expected[ix * 6 + iy, x * 6 + y] = w
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(
        [actual[key] for key in expected], list(expected.values()), atol=1e-12
    )


@pytest.mark.parametrize("axes", [("x", "y"), ("y", "x"), ("distance", "x")])
def test_rectangular_coverage_cache_respects_edges_and_filters(axes):
    from linak.analysis import density
    from linak.analysis.orientation_spatial import _density_adapter

    profile = _density_adapter(orientation_case(8, 2))
    group = dict(profile.geometry_groups[0], origin=np.array([1.3, -2.1, 0.0]))
    local_cache: dict[tuple, np.ndarray] = {}
    for offset, filters in ((0.0, {}), (0.4, {}), (0.4, {"x": (3.1, 7.8), "y": (-0.8, 5.2)})):
        edges = tuple(np.linspace(-4.0 + offset, 16.0 + offset, 13) for _ in axes)
        area, _ = density._prismatic_geometry_volume_factors(
            profile, group, axes, edges, filters, area_cache=local_cache
        )
        plane = np.asarray(group["cell"])[np.ix_([0, 1], [0, 1])]
        origin = group["origin"][:2]
        polygon = np.array(
            [origin, origin + plane[0], origin + plane[0] + plane[1], origin + plane[1]]
        )
        expected = np.zeros(area.shape)
        for index in np.ndindex(area.shape):
            bounds = np.array([[-np.inf, np.inf]] * 2)
            for coordinate, limits in filters.items():
                dimension = ("x", "y").index(coordinate)
                bounds[dimension] = density._merge_bound(*bounds[dimension], limits)
            for dimension, axis in enumerate(axes):
                if axis == "distance":
                    continue
                coordinate_index = ("x", "y").index(axis)
                bounds[coordinate_index] = density._merge_bound(
                    *bounds[coordinate_index],
                    (edges[dimension][index[dimension]], edges[dimension][index[dimension] + 1]),
                )
            expected[index] = density._polygon_rectangle_intersection_area(polygon, bounds)
        np.testing.assert_allclose(area, expected, atol=1e-12)
        # A fresh local cache must hit the shared entry with identical results.
        shared, _ = density._prismatic_geometry_volume_factors(
            profile, group, axes, edges, filters, area_cache={}
        )
        assert shared is area


# Callback tests receive a prepared session; startup itself is tested separately.
pytestmark = pytest.mark.usefixtures("prepared_studio")
