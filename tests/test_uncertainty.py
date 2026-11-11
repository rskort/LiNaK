"""Statistical contracts shared by calculation, plots, Studio and exports."""

import csv

from ase import Atoms
import numpy as np
import pytest

from linak.analysis.statistics import (
    build_series_statistics,
    statistics_payload_from_series_map,
    statistics_series_map_from_datasets,
)
from linak.analysis.uncertainty import (
    bootstrap_interval,
    frame_blocks,
    grouped_frame_statistics,
    estimate,
)
from linak.plot.plotting import _prepare_line_render_series, plot_line_series


def samples():
    return np.repeat(np.arange(20, dtype=float), 25)[:, None] + np.array([[1.0, 2.0]])


def test_block_resampling_recomputes_ratio_and_is_reproducible():
    counts = samples()
    weights = np.linspace(1, 20, len(counts))[:, None] * np.ones((1, 2))
    data = frame_blocks(counts, weights)
    np.testing.assert_allclose(
        estimate(data.numerator.sum(0), data.denominator.sum(0)), counts.sum(0) / weights.sum(0)
    )
    assert not np.allclose(counts.sum(0) / weights.sum(0), (counts / weights).mean(0))
    a = bootstrap_interval(data, seed=17, repeats=200)
    b = bootstrap_interval(data, seed=17, repeats=200)
    np.testing.assert_array_equal(a.lower, b.lower)
    rng = np.random.default_rng(17)
    draws = rng.multinomial(20, np.full(20, 0.05), size=200)
    direct = (draws @ data.numerator[..., 0]) / (draws @ data.denominator[..., 0])
    np.testing.assert_allclose(a.lower, np.quantile(direct, 0.025, axis=0))
    np.testing.assert_allclose(a.upper, np.quantile(direct, 0.975, axis=0))


def test_duplicating_entities_does_not_reduce_frame_uncertainty():
    y = samples()[:, 0]
    a = grouped_frame_statistics(np.zeros(len(y)), y, frame_indices=np.arange(len(y)))
    b = grouped_frame_statistics(
        np.zeros(len(y) * 10), np.repeat(y, 10), frame_indices=np.repeat(np.arange(len(y)), 10)
    )
    np.testing.assert_allclose(a[1], b[1])
    np.testing.assert_allclose(a[2].sample_n, b[2].sample_n)
    ci_a, ci_b = [bootstrap_interval(item[2].resampling, repeats=200) for item in (a, b)]
    np.testing.assert_allclose(ci_a.lower, ci_b.lower)
    np.testing.assert_allclose(ci_a.upper, ci_b.upper)
    assert b[2].point_count[0] == 10 * a[2].point_count[0]
    assert (ci_a.upper - ci_a.lower)[0] > 1.96 * a[2].sample_sem[0] * 2


def test_block_data_roundtrip_and_incomplete_data_rejected():
    stats = build_series_statistics(point_count=np.array([500, 500]), sample_values=samples())
    arrays = statistics_payload_from_series_map({"y": stats})
    restored = statistics_series_map_from_datasets(arrays, dataset_names=["y"])["y"]
    np.testing.assert_array_equal(restored.resampling.numerator, stats.resampling.numerator)
    arrays.pop("y_resampling_denominator")
    with pytest.raises(ValueError, match="Incomplete uncertainty"):
        statistics_series_map_from_datasets(arrays, dataset_names=["y"])


def test_legacy_statistics_do_not_silently_fall_back():
    stats = build_series_statistics(
        point_count=np.array([2]), sample_values=np.array([[1.0], [2.0]])
    )
    prepared, _ = _prepare_line_render_series(
        x_series=[np.array([0.0])],
        y_series=[np.array([1.5])],
        labels=["legacy"],
        series_statistics_data=[stats],
        series_error_configs=[{"enabled": True}],
    )
    assert prepared[0].error_status == "unavailable"
    assert "No estimator was substituted" in prepared[0].error_reason
    assert prepared[0].interval_lower is None


def test_insufficient_raw_data_do_not_create_zero_error():
    prepared, _ = _prepare_line_render_series(
        x_series=[np.arange(5.0)],
        y_series=[np.arange(5.0)],
        labels=["trajectory"],
        series_raw_statistics=[True],
        analysis_name="position",
        series_error_configs=[{"enabled": True, "stat": "sample_std"}],
    )
    assert prepared[0].error_status == "insufficient_samples"
    assert np.isnan(prepared[0].statistics.sample_std).all()


def test_normalization_resamples_shared_reference_and_rebin_preserves_blocks():
    values = samples()[:, :1] * np.array([[1.0, 2.0, 3.0, 4.0]])
    stats = build_series_statistics(point_count=np.array([500] * 4), sample_values=values)
    args = dict(
        x_series=[np.arange(4.0)],
        y_series=[values.mean(0)],
        labels=["density"],
        series_statistics_data=[stats],
        series_error_configs=[{"enabled": True, "resamples": 200}],
        series_normalization_modes=["area"],
        series_normalization_values=[1.0],
    )
    prepared, _ = _prepare_line_render_series(**args)
    np.testing.assert_allclose(prepared[0].interval_lower, prepared[0].y, atol=1e-12)
    np.testing.assert_allclose(prepared[0].interval_upper, prepared[0].y, atol=1e-12)
    args.pop("series_normalization_modes")
    args.pop("series_normalization_values")
    prepared, _ = _prepare_line_render_series(**args, x_bin_width=2.0)
    assert prepared[0].error_status == "ok"
    assert prepared[0].interval_lower.shape == (2,)


def test_bars_and_band_export_identical_intervals(tmp_path):
    from linak.cli import _write_plotted_data_export

    stats = build_series_statistics(point_count=np.array([500, 500]), sample_values=samples())
    states = []
    for style in ("band", "whiskers"):
        state = {}
        plot_line_series(
            np.array([0.0, 1.0]),
            samples().mean(0),
            title="Test",
            x_label="x",
            y_label="y",
            show=False,
            output=tmp_path / f"{style}.png",
            capture_state=state,
            series_statistics=stats,
            error_config={"enabled": True, "style": style, "resamples": 200},
        )
        states.append(state)
    for key in ("lower", "upper"):
        np.testing.assert_array_equal(
            states[0]["series_error_summaries"]["series"][key],
            states[1]["series_error_summaries"]["series"][key],
        )
    output = _write_plotted_data_export(states[0], tmp_path / "data.csv")
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert "error_lower" in rows[0]
    assert rows[0]["error_method"] == "temporal block bootstrap"
    assert float(rows[0]["confidence"]) == 0.95


def test_export_interval_bounds_follow_finite_points(tmp_path):
    from linak.cli import _write_plotted_data_export

    values = np.column_stack([samples()[:, 0], np.full(500, np.nan), samples()[:, 1]])
    values[100:, 2] = np.nan
    stats = build_series_statistics(point_count=np.array([500, 0, 100]), sample_values=values)
    state = {}
    plot_line_series(
        np.arange(3.0),
        np.array([values[:, 0].mean(), np.nan, np.nanmean(values[:, 2])]),
        title="Missing bin",
        x_label="x",
        y_label="y",
        show=False,
        output=tmp_path / "missing.png",
        series_statistics=stats,
        error_config={"enabled": True, "resamples": 200},
        capture_state=state,
    )
    output = _write_plotted_data_export(state, tmp_path / "missing.csv")
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    summary = state["series_error_summaries"]["series"]
    assert float(rows[1]["error_lower"]) == pytest.approx(summary["lower"][2])
    assert int(rows[1]["sample_n"]) == 100
    assert rows[1]["sampling_unit"] == "frames"


def test_temperature_time_windows_have_same_shared_uncertainty(tmp_path):
    from linak.analysis.temperature import plot_temperature_profile

    # Use a real compute result to avoid coupling this test to profile constructor details.
    from linak.analysis.temperature import compute_temperature_from_velocity_xyz

    source = tmp_path / "test-vel-1.xyz"
    source.write_text(
        "".join(f"1\ni = {i}, time = {i}\nH {1 + i * 0.001} 0 0\n" for i in range(100))
    )
    profile = compute_temperature_from_velocity_xyz(source, velocity_unit="angstrom/fs")[0]
    state = {}
    plot_temperature_profile(
        profile,
        show=False,
        output=tmp_path / "temperature.png",
        x_bin_width=100,
        error_config={"enabled": True, "resamples": 200},
        capture_state=state,
    )
    assert next(iter(state["series_error_summaries"].values()))["status"] == "ok"


def test_rdf_saved_confidence_interval_uses_pair_counts(tmp_path):
    from linak.analysis.rdf import compute_rdf, save_rdf_profile, load_rdf_profile

    frames = [
        Atoms("NaO", positions=[[0, 0, 0], [2, 0, 0]], cell=[10 + i * 0.01] * 3, pbc=True)
        for i in range(100)
    ]
    profile = compute_rdf(frames, "Na", "O", r_max=4)
    data = profile.series_statistics["g_r"].resampling
    np.testing.assert_allclose(
        estimate(data.numerator.sum(0), data.denominator.sum(0)), profile.g_r
    )
    path = save_rdf_profile(profile, tmp_path / "saved.rdf.h5")
    restored = load_rdf_profile(path)
    np.testing.assert_allclose(
        restored.series_statistics["g_r"].resampling.numerator, data.numerator
    )


def test_resampled_median_is_not_mean_sem():
    values = np.arange(100.0) ** 2
    grouped = grouped_frame_statistics(np.arange(100.0), values, bin_width=100.0, reducer="median")
    data = grouped[2].resampling
    result = bootstrap_interval(data, repeats=200, seed=11)
    draws = np.random.default_rng(11).multinomial(4, [0.25] * 4, size=200)
    medians = [
        np.median(
            np.concatenate(
                [np.tile(values[b * 25 : (b + 1) * 25], weight) for b, weight in enumerate(row)]
            )
        )
        for row in draws
    ]
    np.testing.assert_allclose(result.lower, [np.quantile(medians, 0.025)])
    np.testing.assert_allclose(result.upper, [np.quantile(medians, 0.975)])
    restored = statistics_series_map_from_datasets(
        statistics_payload_from_series_map({"y": grouped[2]}), dataset_names=["y"]
    )["y"]
    np.testing.assert_allclose(
        bootstrap_interval(restored.resampling, repeats=200, seed=11).lower, result.lower
    )
    prepared, _ = _prepare_line_render_series(
        x_series=[np.arange(100.0)],
        y_series=[values],
        labels=["median"],
        x_bin_width=100.0,
        x_bin_reducer="median",
        series_raw_statistics=[True],
        analysis_name="position",
        series_error_configs=[{"enabled": True, "stat": "sample_sem"}],
    )
    assert prepared[0].error_status == "unavailable"
    assert "cannot describe a median" in prepared[0].error_reason


def test_potential_requires_explicit_sampling_assumption():
    args = dict(
        x_series=[np.arange(100.0)],
        y_series=[np.arange(100.0)],
        labels=["potential"],
        x_bin_width=100.0,
        series_raw_statistics=[True],
        analysis_name="potential",
    )
    default, _ = _prepare_line_render_series(**args, series_error_configs=[{"enabled": True}])
    assert default[0].error_status == "unavailable"
    assert "Record IDs do not establish" in default[0].error_reason
    for assumption, unit in (
        ("temporal", "ordered records"),
        ("independent", "independent replicates"),
    ):
        result, _ = _prepare_line_render_series(
            **args,
            series_error_configs=[
                {
                    "enabled": True,
                    "sampling": assumption,
                    "resamples": 200,
                }
            ],
        )
        assert result[0].error_status == "ok"
        assert result[0].uncertainty_details["sampling_unit"] == unit


def test_orientation_uses_population_weighted_estimator():
    from linak.analysis.orientation import (
        _OrientationFrameData,
        _orientation_uncertainty_statistics,
    )

    frames = []
    for index in range(100):
        count, cosine = (1, 0.0) if index < 50 else (3, 1.0)
        frames.append(
            _OrientationFrameData(
                np.ones((count, 3)),
                np.full(count, 0.5),
                np.full(count, cosine),
                np.full(count, cosine),
                np.ones(count, bool),
                np.zeros(count),
                np.ones(count, bool),
                np.ones(count, dtype=np.int8),
            )
        )
    density = np.r_[np.full(50, 10.0), np.ones(50)][:, None]
    stats = _orientation_uncertainty_statistics(frames, np.array([0.0, 1.0]), density)
    for key, expected in (("cos_polar_mean", 0.75), ("cos_polar_density", 0.75 * 5.5)):
        data = stats[key].resampling
        np.testing.assert_allclose(
            estimate(data.numerator.sum(0), data.denominator.sum(0), data.estimator), [expected]
        )


def test_symmetric_density_retains_side_uncertainty(tmp_path):
    from tests.test_symmetric_surface import _symmetric_frame
    from linak.analysis.density import (
        compute_density_profile,
        save_density_profiles,
        load_density_profiles,
    )

    frames = [_symmetric_frame() for _ in range(100)]
    profile = compute_density_profile(
        frames, species="O", surface_layout="symmetric", surface_elements=["Pt"], bin_width=1.0
    )
    data = profile.series_statistics["density"].resampling
    assert data.estimator == "side_mean"
    np.testing.assert_allclose(
        estimate(data.numerator.sum(0), data.denominator.sum(0), data.estimator), profile.density
    )
    output = save_density_profiles([profile], tmp_path / "density.density.h5")
    restored = load_density_profiles(output)[0]
    assert restored.series_statistics["top_density"].resampling is not None
    assert restored.series_statistics["bottom_density"].resampling is not None


@pytest.mark.parametrize("mode", ["estimate", "mean"])
def test_cli_and_studio_share_bootstrap_configuration(tmp_path, monkeypatch, mode):
    from linak import cli
    import linak.analysis.rdf as rdf
    from linak.plot.plot_gui import _coerce_series_error_config, _resolve_error_stat_for_available

    profile = rdf.RDFProfile(
        "O",
        "H",
        np.array([0.0, 1.0, 2.0]),
        np.array([0.5, 1.5]),
        samples().mean(0),
        500,
        series_statistics={
            "g_r": build_series_statistics(
                point_count=np.array([500, 500]), sample_values=samples()
            )
        },
    )
    path = rdf.save_rdf_profile(profile, tmp_path / "test.rdf.h5")
    state = {}
    plotter = rdf.plot_rdf_profiles
    from functools import wraps

    @wraps(plotter)
    def capture(*args, **kwargs):
        kwargs["capture_state"] = state
        return plotter(*args, **kwargs)

    monkeypatch.setattr(rdf, "plot_rdf_profiles", capture)
    assert (
        cli.main(
            [
                "plot",
                str(path),
                "--no-gui",
                "--no-show",
                "--output",
                str(tmp_path / "cli.png"),
                "--uncertainty",
                "mean",
                "--uncertainty-resamples",
                "200",
                "--uncertainty-seed",
                "17",
            ]
        )
        == 0
    )
    summary = next(iter(state["series_error_summaries"].values()))
    assert summary["status"] == "ok"
    assert summary["details"]["seed"] == 17
    assert summary["details"]["resamples"] == 200
    config = _coerce_series_error_config(
        {"enabled": True, "stat": "block_ci", "seed": 17, "resamples": 200}
    )
    assert config["seed"] == 17 and config["resamples"] == 200
    assert _resolve_error_stat_for_available("block_ci", ["sample_std"]) == "block_ci"


def test_unequal_time_windows_keep_their_own_blocks():
    x = np.r_[np.arange(500.0), np.arange(500.0, 600.0)]
    y = np.arange(600.0) ** 2
    grouped = grouped_frame_statistics(x, y, bin_width=500.0)
    data = grouped[2].resampling
    assert data.independent_bins
    assert np.count_nonzero(data.lengths_per_bin[:, 0]) == 20
    assert np.count_nonzero(data.lengths_per_bin[:, 1]) == 4
    interval = bootstrap_interval(data, repeats=200, seed=7)
    rng = np.random.default_rng(7)
    rng.multinomial(20, [0.05] * 20, size=200)  # First window consumes its own resamples.
    draws = rng.multinomial(4, [0.25] * 4, size=200)
    n = data.numerator[:4, 1, 0]
    d = data.denominator[:4, 1, 0]
    direct = (draws @ n) / (draws @ d)
    np.testing.assert_allclose(interval.lower[1], np.quantile(direct, 0.025))
    restored = statistics_series_map_from_datasets(
        statistics_payload_from_series_map({"y": grouped[2]}), dataset_names=["y"]
    )["y"]
    np.testing.assert_allclose(
        bootstrap_interval(restored.resampling, repeats=200, seed=7).upper, interval.upper
    )


def test_studio_uncertainty_controls_preserve_meaning_and_options(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from linak.plot.plot_gui import launch_plot_settings_panel, _ERROR_STAT_DISPLAY

    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No)
    checked = []

    def inspect(app):
        for window in app.topLevelWidgets():
            if not window.isVisible() or not hasattr(window, "_uncertainty_inputs"):
                continue
            window._series_active_index = 0
            window._load_series_into_editor(0)
            window._series_error_mode.setCurrentText("on")
            window._series_error_stat.setCurrentText(_ERROR_STAT_DISPLAY["block_ci"])
            window._uncertainty_inputs["seed"].setText("17")
            window._uncertainty_inputs["resamples"].setText("200")
            settings = window._collect_settings()
            error = settings["series_overrides"]["test"]["error"]
            assert error["enabled"] and error["stat"] == "block_ci"
            assert error["seed"] == 17 and error["resamples"] == 200
            # Missing preview statistics must not change the chosen meaning.
            window._refresh_error_stat_choices(0)
            assert window._series_error_stat.currentText() == _ERROR_STAT_DISPLAY["block_ci"]
            checked.append(True)
            window._closing = True
            window._cleanup_preview_artifacts()
            window.hide()
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect)
    launch_plot_settings_panel(
        title="Uncertainty regression",
        analysis_name="rdf",
        initial_settings={
            "series_labels": ["test"],
            "series_descriptors": [
                {"series_id": "test", "default_label": "test", "source_kind": "source"}
            ],
        },
        on_preview=lambda settings: None,
        on_save=lambda name, settings: name,
    )
    assert checked
