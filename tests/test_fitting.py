import numpy as np

from linak.plot.fitting import execute_series_fit, resolve_series_fit_configs


def test_execute_series_fit_linear_returns_expected_parameters():
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y = 2.0 * x + 1.0

    summary = execute_series_fit(
        x,
        y,
        fit_config={"fit_enabled": True, "fit_type": "linear"},
    )

    assert summary["status"] == "ok"
    assert summary["parameters"]["slope"] == 2.0
    assert summary["parameters"]["intercept"] == 1.0
    assert summary["fit_point_count"] == 4
    assert summary["display_point_count"] == 4


def test_execute_series_fit_polynomial_uses_degree_and_manual_range():
    x = np.linspace(-2.0, 2.0, 50)
    y = 3.0 * x**2 - 2.0 * x + 4.0

    summary = execute_series_fit(
        x,
        y,
        fit_config={
            "fit_enabled": True,
            "fit_type": "polynomial",
            "fit_degree": 2,
            "fit_range_mode": "manual",
            "fit_x_min": -1.0,
            "fit_x_max": 1.0,
        },
    )

    assert summary["status"] == "ok"
    assert summary["parameter_order"] == ["a2", "a1", "a0"]
    assert summary["fit_point_count"] < summary["display_point_count"]
    assert summary["parameters"]["a2"] == 3.0
    assert summary["parameters"]["a1"] == -2.0
    assert summary["parameters"]["a0"] == 4.0


def test_execute_series_fit_gaussian_converges_on_synthetic_data():
    x = np.linspace(-5.0, 5.0, 100)
    y = 4.0 * np.exp(-0.5 * ((x - 1.5) / 0.8) ** 2) + 0.2

    summary = execute_series_fit(
        x,
        y,
        fit_config={"fit_enabled": True, "fit_type": "gaussian"},
    )

    assert summary["status"] == "ok"
    assert summary["parameters"]["amplitude"] == np.testing.assert_approx_equal(4.0, significant=4)


def test_execute_series_fit_logarithmic_requires_positive_x():
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
    y = np.array([0.0, 0.0, 1.0, 1.7, 2.1], dtype=float)

    summary = execute_series_fit(
        x,
        y,
        fit_config={"fit_enabled": True, "fit_type": "logarithmic"},
    )

    assert summary["status"] in {"ok", "invalid"}
    assert summary["display_point_count"] == 5


def test_resolve_series_fit_configs_upgrades_legacy_linear_fit_flags():
    configs = resolve_series_fit_configs(
        series_count=1,
        series_fit_enabled=[True],
        series_fit_labels=["demo fit"],
        series_fit_show_in_legend=[False],
    )

    assert configs == [
        {
            "fit_enabled": True,
            "fit_type": "linear",
            "fit_degree": None,
            "fit_range_mode": "visible",
            "fit_x_min": None,
            "fit_x_max": None,
            "fit_initial_guess": None,
            "fit_bounds": None,
            "fit_label_override": "demo fit",
            "fit_show_in_legend": False,
        }
    ]
