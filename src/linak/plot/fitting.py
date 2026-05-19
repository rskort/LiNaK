"""Shared curve-fitting helpers for Plot Studio and CLI rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import curve_fit

FitConfigDict = dict[str, Any]
FitSummaryDict = dict[str, Any]

_DEFAULT_FIT_CONFIG: FitConfigDict = {
    "fit_enabled": False,
    "fit_type": "linear",
    "fit_degree": None,
    "fit_range_mode": "visible",
    "fit_x_min": None,
    "fit_x_max": None,
    "fit_initial_guess": None,
    "fit_bounds": None,
    "fit_label_override": None,
    "fit_show_in_legend": True,
    "fit_color": None,
    "fit_alpha": None,
    "fit_line_width": None,
    "fit_line_style": None,
}

_OPTIONAL_FIT_STYLE_KEYS = (
    "fit_color",
    "fit_alpha",
    "fit_line_width",
    "fit_line_style",
)

_SUPPORTED_FIT_TYPES = frozenset(
    {
        "linear",
        "polynomial",
        "exponential",
        "logarithmic",
        "power_law",
        "gaussian",
        "lorentzian",
    }
)


@dataclass(frozen=True)
class FitModel:
    """Static definition for one fit family."""

    fit_type: str
    parameter_order: tuple[str, ...]
    equation_template: str
    model: Callable[..., np.ndarray] | None = None
    requires_positive_x: bool = False


def _as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_degree(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_fit_bounds(
    value: Any, parameter_order: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, dict):
        lower_bounds = np.full(len(parameter_order), -np.inf, dtype=float)
        upper_bounds = np.full(len(parameter_order), np.inf, dtype=float)
        return lower_bounds, upper_bounds

    lower_limits: list[float] = []
    upper_limits: list[float] = []
    for name in parameter_order:
        raw = value.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            lower_limits.append(-np.inf)
            upper_limits.append(np.inf)
            continue
        lower_limits.append(-np.inf if raw[0] is None else float(raw[0]))
        upper_limits.append(np.inf if raw[1] is None else float(raw[1]))
    return np.asarray(lower_limits, dtype=float), np.asarray(upper_limits, dtype=float)


def _coerce_fit_initial_guess(
    value: Any,
    parameter_order: tuple[str, ...],
) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    guess: list[float] = []
    for name in parameter_order:
        numeric = _as_float(value.get(name))
        if numeric is None:
            return None
        guess.append(numeric)
    return guess


def _linear_model(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * x + intercept


def _exponential_model(x: np.ndarray, amplitude: float, rate: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(rate * x) + offset


def _logarithmic_model(x: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return scale * np.log(x) + offset


def _power_law_model(x: np.ndarray, scale: float, exponent: float) -> np.ndarray:
    return scale * np.power(x, exponent)


def _gaussian_model(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    offset: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * np.square((x - center) / sigma)) + offset


def _lorentzian_model(
    x: np.ndarray,
    amplitude: float,
    center: float,
    gamma: float,
    offset: float,
) -> np.ndarray:
    return amplitude * (gamma**2 / (np.square(x - center) + gamma**2)) + offset


_FIT_MODELS: dict[str, FitModel] = {
    "linear": FitModel(
        fit_type="linear",
        parameter_order=("slope", "intercept"),
        equation_template="y = slope·x + intercept",
        model=_linear_model,
    ),
    "polynomial": FitModel(
        fit_type="polynomial",
        parameter_order=(),
        equation_template="y = Σ a_n·x^n",
        model=None,
    ),
    "exponential": FitModel(
        fit_type="exponential",
        parameter_order=("amplitude", "rate", "offset"),
        equation_template="y = amplitude·exp(rate·x) + offset",
        model=_exponential_model,
    ),
    "logarithmic": FitModel(
        fit_type="logarithmic",
        parameter_order=("scale", "offset"),
        equation_template="y = scale·ln(x) + offset",
        model=_logarithmic_model,
        requires_positive_x=True,
    ),
    "power_law": FitModel(
        fit_type="power_law",
        parameter_order=("scale", "exponent"),
        equation_template="y = scale·x^exponent",
        model=_power_law_model,
        requires_positive_x=True,
    ),
    "gaussian": FitModel(
        fit_type="gaussian",
        parameter_order=("amplitude", "center", "sigma", "offset"),
        equation_template="y = amplitude·exp(-0.5·((x-center)/sigma)^2) + offset",
        model=_gaussian_model,
    ),
    "lorentzian": FitModel(
        fit_type="lorentzian",
        parameter_order=("amplitude", "center", "gamma", "offset"),
        equation_template="y = amplitude·gamma^2 / ((x-center)^2 + gamma^2) + offset",
        model=_lorentzian_model,
    ),
}


def supported_fit_types() -> tuple[str, ...]:
    """Return the fit families supported by the plotting layer."""
    return tuple(_FIT_MODELS)


def default_fit_config() -> FitConfigDict:
    """Return a fresh copy of the default fit configuration."""
    return dict(_DEFAULT_FIT_CONFIG)


def coerce_fit_config(
    raw: Any,
) -> FitConfigDict:
    """Normalize a raw fit configuration."""
    config = default_fit_config()

    if isinstance(raw, dict):
        if "fit_enabled" in raw:
            config["fit_enabled"] = bool(raw.get("fit_enabled"))
        fit_type = str(raw.get("fit_type") or config["fit_type"]).strip().lower()
        config["fit_type"] = fit_type if fit_type in _SUPPORTED_FIT_TYPES else "linear"
        fit_degree = _coerce_degree(raw.get("fit_degree"))
        config["fit_degree"] = fit_degree if fit_degree is None or fit_degree >= 1 else 1
        range_mode = str(raw.get("fit_range_mode") or "visible").strip().lower()
        config["fit_range_mode"] = "manual" if range_mode == "manual" else "visible"
        config["fit_x_min"] = _as_float(raw.get("fit_x_min"))
        config["fit_x_max"] = _as_float(raw.get("fit_x_max"))
        guess = raw.get("fit_initial_guess")
        config["fit_initial_guess"] = dict(guess) if isinstance(guess, dict) else None
        bounds = raw.get("fit_bounds")
        config["fit_bounds"] = dict(bounds) if isinstance(bounds, dict) else None
        fit_label = raw.get("fit_label_override")
        config["fit_label_override"] = None if fit_label in {None, ""} else str(fit_label).strip()
        if "fit_show_in_legend" in raw:
            config["fit_show_in_legend"] = bool(raw.get("fit_show_in_legend"))
        fit_color = raw.get("fit_color")
        config["fit_color"] = None if fit_color in {None, ""} else str(fit_color).strip()
        config["fit_alpha"] = _as_float(raw.get("fit_alpha"))
        config["fit_line_width"] = _as_float(raw.get("fit_line_width"))
        fit_line_style = raw.get("fit_line_style")
        config["fit_line_style"] = (
            None if fit_line_style in {None, ""} else str(fit_line_style).strip()
        )

    if config["fit_type"] == "polynomial":
        config["fit_degree"] = max(1, int(config.get("fit_degree") or 2))
    else:
        config["fit_degree"] = None

    if config["fit_x_min"] is not None and config["fit_x_max"] is not None:
        if float(config["fit_x_min"]) > float(config["fit_x_max"]):
            config["fit_x_min"], config["fit_x_max"] = config["fit_x_max"], config["fit_x_min"]
    return config


def resolve_series_fit_configs(
    *,
    series_count: int,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
) -> list[FitConfigDict]:
    """Resolve per-series fit configurations."""
    configs: list[FitConfigDict] = []
    for index in range(series_count):
        raw = (
            None
            if series_fit_configs is None or index >= len(series_fit_configs)
            else series_fit_configs[index]
        )
        config = coerce_fit_config(raw)
        if isinstance(raw, dict):
            for key in _OPTIONAL_FIT_STYLE_KEYS:
                if key not in raw and config.get(key) is None:
                    config.pop(key, None)
        configs.append(config)
    return configs


def _initial_guess_for_model(
    model: FitModel, x: np.ndarray, y: np.ndarray, *, degree: int | None
) -> list[float] | None:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = max(x_max - x_min, 1.0e-12)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = y_max - y_min

    if model.fit_type == "exponential":
        amplitude = y_span if abs(y_span) > 1.0e-12 else (y_max or 1.0)
        return [amplitude, 1.0 / x_span, y_min]
    if model.fit_type == "logarithmic":
        return [y_span if abs(y_span) > 1.0e-12 else 1.0, float(np.mean(y))]
    if model.fit_type == "power_law":
        safe_scale = y[0] if abs(float(y[0])) > 1.0e-12 else 1.0
        return [float(safe_scale), 1.0]
    if model.fit_type == "gaussian":
        center = float(x[np.argmax(y)])
        sigma = max(x_span / 6.0, 1.0e-6)
        return [y_span if abs(y_span) > 1.0e-12 else 1.0, center, sigma, y_min]
    if model.fit_type == "lorentzian":
        center = float(x[np.argmax(y)])
        gamma = max(x_span / 10.0, 1.0e-6)
        return [y_span if abs(y_span) > 1.0e-12 else 1.0, center, gamma, y_min]
    return None


def _polynomial_parameter_names(degree: int) -> tuple[str, ...]:
    return tuple(f"a{power}" for power in range(degree, -1, -1))


def _polynomial_equation(degree: int) -> str:
    terms: list[str] = []
    for power in range(degree, -1, -1):
        name = f"a{power}"
        if power == 0:
            terms.append(name)
        elif power == 1:
            terms.append(f"{name}·x")
        else:
            terms.append(f"{name}·x^{power}")
    return "y = " + " + ".join(terms)


def _format_fit_number(value: float) -> str:
    numeric = float(value)
    if math.isclose(numeric, 0.0, abs_tol=1.0e-15):
        numeric = 0.0
    return f"{numeric:.6g}"


def _linear_equation_from_parameters(*, slope: float, intercept: float) -> str:
    return f"y = {_format_fit_number(slope)}*x + {_format_fit_number(intercept)}"


def _polynomial_equation_from_coefficients(coefficients: np.ndarray) -> str:
    resolved = np.asarray(coefficients, dtype=float)
    degree = int(resolved.size - 1)
    terms: list[str] = []
    for index, coefficient in enumerate(resolved):
        power = degree - index
        numeric = _format_fit_number(float(coefficient))
        if power == 0:
            terms.append(numeric)
        elif power == 1:
            terms.append(f"{numeric}*x")
        else:
            terms.append(f"{numeric}*x^{power}")
    return "y = " + " + ".join(terms)


def _build_fit_summary(
    *,
    status: str,
    fit_type: str,
    equation: str,
    parameters: dict[str, float],
    parameter_order: list[str],
    fit_point_count: int,
    display_point_count: int,
    x_fit: np.ndarray | None = None,
    y_fit: np.ndarray | None = None,
    r_squared: float | None = None,
    rmse: float | None = None,
    reason: str = "",
    characteristic_point: dict[str, float | str] | None = None,
) -> FitSummaryDict:
    return {
        "status": str(status),
        "fit_type": str(fit_type),
        "equation": str(equation),
        "parameters": dict(parameters),
        "parameter_order": list(parameter_order),
        "r_squared": None if r_squared is None else float(r_squared),
        "rmse": None if rmse is None else float(rmse),
        "fit_point_count": int(fit_point_count),
        "point_count": int(fit_point_count),
        "display_point_count": int(display_point_count),
        "x_fit": [] if x_fit is None else np.asarray(x_fit, dtype=float).tolist(),
        "y_fit": [] if y_fit is None else np.asarray(y_fit, dtype=float).tolist(),
        "reason": str(reason),
        "characteristic_point": (
            None if characteristic_point is None else dict(characteristic_point)
        ),
    }


def _build_fit_x_grid(fit_x: np.ndarray, *, requires_positive_x: bool) -> np.ndarray:
    min_x = float(np.min(fit_x))
    max_x = float(np.max(fit_x))
    if math.isclose(min_x, max_x):
        return np.asarray([min_x, max_x], dtype=float)
    if requires_positive_x:
        min_x = max(min_x, float(np.min(fit_x[fit_x > 0])))
    return np.linspace(min_x, max_x, 256, dtype=float)


def execute_series_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    fit_config: dict[str, Any] | None,
    visible_x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
) -> FitSummaryDict:
    """Fit displayed series data using the requested fit family."""
    config = coerce_fit_config(fit_config)
    if not config["fit_enabled"]:
        return _build_fit_summary(
            status="off",
            fit_type=str(config["fit_type"]),
            equation="",
            parameters={},
            parameter_order=[],
            fit_point_count=0,
            display_point_count=0,
        )

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if visible_x_lim is not None:
        left = None if visible_x_lim[0] is None else float(visible_x_lim[0])
        right = None if visible_x_lim[1] is None else float(visible_x_lim[1])
        if left is not None:
            mask = x >= left
            x = x[mask]
            y = y[mask]
        if right is not None:
            mask = x <= right
            x = x[mask]
            y = y[mask]

    display_point_count = int(x.size)
    if display_point_count < 2:
        return _build_fit_summary(
            status="invalid",
            fit_type=str(config["fit_type"]),
            equation="",
            parameters={},
            parameter_order=[],
            fit_point_count=int(x.size),
            display_point_count=display_point_count,
            reason="Fit requires at least two finite displayed points.",
        )

    fit_x = x
    fit_y = y
    if str(config["fit_range_mode"]).lower() == "manual":
        x_min = _as_float(config.get("fit_x_min"))
        x_max = _as_float(config.get("fit_x_max"))
        if x_min is not None:
            mask = fit_x >= x_min
            fit_x = fit_x[mask]
            fit_y = fit_y[mask]
        if x_max is not None:
            mask = fit_x <= x_max
            fit_x = fit_x[mask]
            fit_y = fit_y[mask]

    fit_type = str(config["fit_type"]).strip().lower()
    model = _FIT_MODELS.get(fit_type, _FIT_MODELS["linear"])
    if model.requires_positive_x:
        positive_mask = fit_x > 0
        fit_x = fit_x[positive_mask]
        fit_y = fit_y[positive_mask]

    fit_point_count = int(fit_x.size)
    if fit_point_count < 2 or np.unique(fit_x).size < 2:
        return _build_fit_summary(
            status="invalid",
            fit_type=fit_type,
            equation=model.equation_template,
            parameters={},
            parameter_order=list(model.parameter_order),
            fit_point_count=fit_point_count,
            display_point_count=display_point_count,
            reason="Fit requires at least two valid points and two distinct x values.",
        )

    try:
        characteristic_point: dict[str, float | str] | None = None
        if fit_type == "linear":
            coefficients = np.polyfit(fit_x, fit_y, deg=1)
            parameter_order = list(model.parameter_order)
            parameters = {
                "slope": float(coefficients[0]),
                "intercept": float(coefficients[1]),
            }
            predictor = lambda values: _linear_model(  # noqa: E731
                np.asarray(values, dtype=float),
                parameters["slope"],
                parameters["intercept"],
            )
            equation = _linear_equation_from_parameters(
                slope=parameters["slope"],
                intercept=parameters["intercept"],
            )
        elif fit_type == "polynomial":
            degree = max(1, int(config.get("fit_degree") or 2))
            coefficients = np.polyfit(fit_x, fit_y, deg=degree)
            parameter_order = list(_polynomial_parameter_names(degree))
            parameters = {
                parameter_order[index]: float(coefficients[index])
                for index in range(len(parameter_order))
            }
            predictor = lambda values: np.polyval(  # noqa: E731
                np.asarray(coefficients, dtype=float),
                np.asarray(values, dtype=float),
            )
            equation = _polynomial_equation_from_coefficients(
                np.asarray(coefficients, dtype=float)
            )
            if degree == 2:
                a, b, _c = np.asarray(coefficients, dtype=float)
                if np.isfinite(a) and abs(a) > 1.0e-15:
                    vertex_x = float(-b / (2.0 * a))
                    characteristic_point = {
                        "label": "Vertex",
                        "x": vertex_x,
                        "y": float(
                            np.polyval(np.asarray(coefficients, dtype=float), np.asarray([vertex_x]))[0]
                        ),
                    }
        else:
            assert model.model is not None
            nonlinear_model = model.model
            parameter_order = list(model.parameter_order)
            initial_guess = _coerce_fit_initial_guess(
                config.get("fit_initial_guess"), model.parameter_order
            )
            if initial_guess is None:
                initial_guess = _initial_guess_for_model(
                    model,
                    fit_x,
                    fit_y,
                    degree=config.get("fit_degree"),
                )
            lower, upper = _coerce_fit_bounds(config.get("fit_bounds"), model.parameter_order)
            if fit_type == "gaussian":
                sigma_index = parameter_order.index("sigma")
                lower[sigma_index] = max(lower[sigma_index], 1.0e-12)
            if fit_type == "lorentzian":
                gamma_index = parameter_order.index("gamma")
                lower[gamma_index] = max(lower[gamma_index], 1.0e-12)
            coefficients, _cov = curve_fit(
                nonlinear_model,
                fit_x,
                fit_y,
                p0=initial_guess,
                bounds=(lower, upper),
                maxfev=20000,
            )
            parameters = {
                parameter_order[index]: float(coefficients[index])
                for index in range(len(parameter_order))
            }
            predictor = lambda values: nonlinear_model(  # noqa: E731
                np.asarray(values, dtype=float),
                *np.asarray(coefficients, dtype=float),
            )
            equation = model.equation_template

        predicted_y = np.asarray(predictor(fit_x), dtype=float)
        residual = fit_y - predicted_y
        ss_res = float(np.sum(residual**2))
        centered = fit_y - float(np.mean(fit_y))
        ss_tot = float(np.sum(centered**2))
        r_squared = 1.0 if ss_tot <= 1.0e-15 else 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(ss_res / max(fit_point_count, 1)))
        x_fit = _build_fit_x_grid(fit_x, requires_positive_x=model.requires_positive_x)
        y_fit = np.asarray(predictor(x_fit), dtype=float)
        if characteristic_point is not None:
            point_x = float(characteristic_point["x"])
            characteristic_point = {
                "label": str(characteristic_point["label"]),
                "x": point_x,
                "y": float(np.asarray(predictor(np.asarray([point_x], dtype=float)), dtype=float)[0]),
            }
        return _build_fit_summary(
            status="ok",
            fit_type=fit_type,
            equation=equation,
            parameters=parameters,
            parameter_order=parameter_order,
            fit_point_count=fit_point_count,
            display_point_count=display_point_count,
            x_fit=x_fit,
            y_fit=y_fit,
            r_squared=float(r_squared),
            rmse=float(rmse),
            characteristic_point=characteristic_point,
        )
    except Exception as exc:
        return _build_fit_summary(
            status="error",
            fit_type=fit_type,
            equation=model.equation_template,
            parameters={},
            parameter_order=list(model.parameter_order),
            fit_point_count=fit_point_count,
            display_point_count=display_point_count,
            reason=str(exc),
        )
