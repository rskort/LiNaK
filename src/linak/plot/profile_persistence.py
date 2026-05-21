"""Helpers for LiNaK's structured saved plot-profile format.

Saved plot profiles now persist three explicit sections:

- ``source_selection``: which loaded profile subset is selected
- ``view_mapping``: how available quantities map onto plot roles
- ``style``: styling, per-series state, annotations, and other presentation data
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .data_contract import PlotViewMapping

_PLOT_PROFILE_DENSITY = "plot:density"
_PLOT_PROFILE_MSD = "plot:msd"
_PLOT_PROFILE_RDF = "plot:rdf"
_PLOT_PROFILE_POSITION = "plot:position"
_PLOT_PROFILE_COORDINATION = "plot:coordination"
_PLOT_PROFILE_POTENTIAL = "plot:potential"
_PLOT_PROFILE_ORIENTATION = "plot:orientation"
_PLOT_PROFILE_TEMPERATURE = "plot:temperature"
_PLOT_PROFILE_TABLE = "plot:table"
_PLOT_PROFILE_SENTINEL = "__linak_plot_profile__"
_PLOT_PROFILE_VERSION = 2

_SOURCE_SELECTION_FIELDS_BY_PROFILE_KEY: dict[str, tuple[str, ...]] = {
    _PLOT_PROFILE_DENSITY: ("species", "axis"),
    _PLOT_PROFILE_MSD: ("species",),
    _PLOT_PROFILE_RDF: ("species_a", "species_b"),
    _PLOT_PROFILE_POSITION: ("species", "axis"),
    _PLOT_PROFILE_COORDINATION: ("species_a", "species_b", "axis"),
    _PLOT_PROFILE_POTENTIAL: (),
    _PLOT_PROFILE_ORIENTATION: (),
    _PLOT_PROFILE_TEMPERATURE: (),
    _PLOT_PROFILE_TABLE: ("group",),
}

_LEGACY_MAPPING_FIELDS_BY_PROFILE_KEY: dict[str, tuple[str, ...]] = {
    _PLOT_PROFILE_DENSITY: ("x_mode", "quantity"),
    _PLOT_PROFILE_MSD: ("time_axis",),
    _PLOT_PROFILE_RDF: (),
    _PLOT_PROFILE_POSITION: (
        "component",
        "map_color",
        "projection_x",
        "projection_y",
        "projection_value",
        "projection_render_mode",
        "projection_filter_min",
        "projection_filter_max",
        "xy_z_distance_max",
        "time_axis",
    ),
    _PLOT_PROFILE_COORDINATION: ("component", "time_axis"),
    _PLOT_PROFILE_POTENTIAL: ("y_quantity", "table_view", "view_type"),
    _PLOT_PROFILE_ORIENTATION: ("component", "angle"),
    _PLOT_PROFILE_TEMPERATURE: ("time_axis",),
    _PLOT_PROFILE_TABLE: ("kind", "x", "y", "bins"),
}

_NON_PERSISTED_STYLE_KEYS = frozenset(
    {
        "series_count",
        "series_descriptors",
        "_profile_filter_options",
        "data_contract",
        "view_mapping",
        "source_selection",
        "style",
    }
)


def serialize_plot_view_mapping(mapping: PlotViewMapping) -> dict[str, Any]:
    """Convert one mapping object into a JSON-ready payload."""

    return {
        "view_type_id": str(mapping.view_type_id),
        "x": None if mapping.x is None else str(mapping.x),
        "y": None if mapping.y is None else str(mapping.y),
        "color": None if mapping.color is None else str(mapping.color),
        "split_by": None if mapping.split_by is None else str(mapping.split_by),
        "filter_by": None if mapping.filter_by is None else str(mapping.filter_by),
        "filter_min": mapping.filter_min,
        "filter_max": mapping.filter_max,
        "role_assignments": {
            str(key): str(value)
            for key, value in dict(mapping.role_assignments).items()
            if str(key).strip() and value is not None
        },
        "fixed_values": {
            str(key): str(value)
            for key, value in dict(mapping.fixed_values).items()
            if str(key).strip() and value is not None
        },
    }


def deserialize_plot_view_mapping(payload: dict[str, Any]) -> PlotViewMapping:
    """Convert one JSON payload back into a mapping object."""

    if not isinstance(payload, dict):
        raise ValueError("Saved plot profile view_mapping must be an object.")
    return PlotViewMapping(
        view_type_id=str(payload.get("view_type_id") or "").strip(),
        x=None if payload.get("x") is None else str(payload.get("x")),
        y=None if payload.get("y") is None else str(payload.get("y")),
        color=None if payload.get("color") is None else str(payload.get("color")),
        split_by=None if payload.get("split_by") is None else str(payload.get("split_by")),
        filter_by=None if payload.get("filter_by") is None else str(payload.get("filter_by")),
        filter_min=_optional_float(payload.get("filter_min")),
        filter_max=_optional_float(payload.get("filter_max")),
        role_assignments=_string_dict(payload.get("role_assignments")),
        fixed_values=_string_dict(payload.get("fixed_values")),
    )


def _resolve_explicit_view_mapping(settings: dict[str, Any]) -> PlotViewMapping | None:
    raw_mapping = settings.get("view_mapping")
    if raw_mapping is None:
        return None
    if isinstance(raw_mapping, PlotViewMapping):
        return raw_mapping
    if isinstance(raw_mapping, dict):
        return deserialize_plot_view_mapping(raw_mapping)
    raise ValueError("Plot profile setting 'view_mapping' must be a mapping payload object.")


def build_plot_profile_payload(profile_key: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Split one flat in-memory settings dict into the persisted profile sections."""

    if not isinstance(settings, dict):
        raise ValueError("Plot profile settings must be an object.")
    source_fields = _SOURCE_SELECTION_FIELDS_BY_PROFILE_KEY.get(str(profile_key), ())
    mapping_fields = _LEGACY_MAPPING_FIELDS_BY_PROFILE_KEY.get(str(profile_key), ())
    source_selection = {key: deepcopy(settings[key]) for key in source_fields if key in settings}
    mapping = _resolve_explicit_view_mapping(settings)
    if mapping is None:
        mapping = _build_view_mapping(str(profile_key), settings)
    style = {
        str(key): deepcopy(value)
        for key, value in settings.items()
        if key not in _NON_PERSISTED_STYLE_KEYS
        and key not in source_fields
        and key not in mapping_fields
    }
    return {
        _PLOT_PROFILE_SENTINEL: _PLOT_PROFILE_VERSION,
        "source_selection": source_selection,
        "view_mapping": serialize_plot_view_mapping(mapping),
        "style": style,
    }


def flatten_plot_profile_payload(profile_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one compatibility-oriented flat settings view from a persisted payload."""

    if not isinstance(payload, dict):
        raise ValueError("Saved plot profile payload must be an object.")
    source_selection = payload.get("source_selection")
    view_mapping_payload = payload.get("view_mapping")
    style = payload.get("style")
    if not isinstance(source_selection, dict):
        raise ValueError("Saved plot profile source_selection must be an object.")
    if not isinstance(style, dict):
        raise ValueError("Saved plot profile style must be an object.")
    mapping = deserialize_plot_view_mapping(view_mapping_payload)
    flattened = {str(key): deepcopy(value) for key, value in source_selection.items()}
    mapping_settings = _flatten_view_mapping(str(profile_key), mapping)
    default_mapping_settings = _default_mapping_settings(str(profile_key))
    for key, value in mapping_settings.items():
        if key not in default_mapping_settings or default_mapping_settings[key] != value:
            flattened[key] = deepcopy(value)
    for key, value in style.items():
        flattened[str(key)] = deepcopy(value)
    return flattened


def plot_profile_requires_legacy_mapping_flatten(
    *,
    profile_key: str,
    keys: tuple[str, ...] | None,
) -> bool:
    """Return whether one requested settings subset still needs compatibility flat mapping fields."""

    if keys is None:
        return True
    legacy_keys = _LEGACY_MAPPING_FIELDS_BY_PROFILE_KEY.get(str(profile_key), ())
    return any(str(key) in legacy_keys for key in keys)


def select_plot_profile_settings(
    profile_key: str,
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...] | None,
) -> dict[str, Any]:
    """Return either a structured settings subset or one legacy-flattened compatibility view."""

    if plot_profile_requires_legacy_mapping_flatten(profile_key=profile_key, keys=keys):
        return flatten_plot_profile_payload(profile_key, payload)

    if not isinstance(payload, dict):
        raise ValueError("Saved plot profile payload must be an object.")
    source_selection = payload.get("source_selection")
    style = payload.get("style")
    view_mapping_payload = payload.get("view_mapping")
    if not isinstance(source_selection, dict):
        raise ValueError("Saved plot profile source_selection must be an object.")
    if not isinstance(style, dict):
        raise ValueError("Saved plot profile style must be an object.")

    selected: dict[str, Any] = {
        str(key): deepcopy(value) for key, value in source_selection.items() if str(key) in keys
    }
    if isinstance(view_mapping_payload, dict) and "view_mapping" in keys:
        selected["view_mapping"] = deepcopy(view_mapping_payload)
    for key, value in style.items():
        if str(key) in keys:
            selected[str(key)] = deepcopy(value)
    return selected


def _build_view_mapping(profile_key: str, settings: dict[str, Any]) -> PlotViewMapping:
    if profile_key == _PLOT_PROFILE_DENSITY:
        from .mappings.density_mapping import density_plot_options_to_view_mapping

        return density_plot_options_to_view_mapping(
            x_mode=str(settings.get("x_mode") or "distance"),
            quantity=str(settings.get("quantity") or "mass"),
        )
    if profile_key == _PLOT_PROFILE_MSD:
        from .mappings.msd_mapping import msd_plot_options_to_view_mapping

        return msd_plot_options_to_view_mapping(
            time_axis=str(settings.get("time_axis") or "ps"),
        )
    if profile_key == _PLOT_PROFILE_RDF:
        from .mappings.rdf_mapping import rdf_plot_options_to_view_mapping

        return rdf_plot_options_to_view_mapping()
    if profile_key == _PLOT_PROFILE_POSITION:
        from .mappings.position_mapping import position_plot_options_to_view_mapping

        return position_plot_options_to_view_mapping(
            component=str(settings.get("component") or "distance"),
            time_axis=str(settings.get("time_axis") or "ps"),
            map_color=str(settings.get("map_color") or "distance"),
            projection_x=_optional_str(settings.get("projection_x")),
            projection_y=_optional_str(settings.get("projection_y")),
            projection_value=_optional_str(settings.get("projection_value")),
            projection_render_mode=_optional_str(settings.get("projection_render_mode")),
            projection_filter_min=_optional_float(settings.get("projection_filter_min")),
            projection_filter_max=_optional_float(settings.get("projection_filter_max")),
            xy_z_distance_max=_optional_float(settings.get("xy_z_distance_max")),
        )
    if profile_key == _PLOT_PROFILE_COORDINATION:
        from .mappings.coordination_mapping import coordination_plot_options_to_view_mapping

        return coordination_plot_options_to_view_mapping(
            component=str(settings.get("component") or "distance"),
            time_axis=str(settings.get("time_axis") or "ps"),
        )
    if profile_key == _PLOT_PROFILE_POTENTIAL:
        from .mappings.potential_mapping import potential_plot_options_to_view_mapping

        table_view = bool(settings.get("table_view")) or (
            str(settings.get("view_type") or "").strip().lower() == "table_records"
        )
        return potential_plot_options_to_view_mapping(
            y_quantity=_optional_str(settings.get("y_quantity")),
            table_view=table_view,
        )
    if profile_key == _PLOT_PROFILE_ORIENTATION:
        from .mappings.orientation_mapping import orientation_plot_options_to_view_mapping

        return orientation_plot_options_to_view_mapping(
            component=str(settings.get("component") or "average"),
            angle=str(settings.get("angle") or "polar"),
        )
    if profile_key == _PLOT_PROFILE_TEMPERATURE:
        from .mappings.temperature_mapping import temperature_plot_options_to_view_mapping

        return temperature_plot_options_to_view_mapping(
            time_axis=str(settings.get("time_axis") or "ps"),
        )
    if profile_key == _PLOT_PROFILE_TABLE:
        kind = str(settings.get("kind") or "line").strip().lower() or "line"
        fixed_values: dict[str, str] = {"kind": kind}
        if settings.get("bins") is not None:
            fixed_values["bins"] = str(settings["bins"])
        return PlotViewMapping(
            view_type_id=f"table_{kind}",
            x=_optional_str(settings.get("x")),
            y=_optional_str(settings.get("y")),
            fixed_values=fixed_values,
        )
    raise ValueError(f"Unsupported plot profile key '{profile_key}'.")


def _flatten_view_mapping(profile_key: str, mapping: PlotViewMapping) -> dict[str, Any]:
    if profile_key == _PLOT_PROFILE_DENSITY:
        from .mappings.density_mapping import density_view_mapping_to_plot_options

        return density_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_MSD:
        from .mappings.msd_mapping import msd_view_mapping_to_plot_options

        return msd_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_RDF:
        from .mappings.rdf_mapping import rdf_view_mapping_to_plot_options

        return rdf_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_POSITION:
        from .mappings.position_mapping import position_view_mapping_to_plot_options

        return position_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_COORDINATION:
        from .mappings.coordination_mapping import coordination_view_mapping_to_plot_options

        return coordination_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_POTENTIAL:
        from .mappings.potential_mapping import potential_view_mapping_to_plot_options

        options = potential_view_mapping_to_plot_options(mapping)
        flattened: dict[str, Any] = {}
        view_type = str(options.get("view_type") or "").strip().lower()
        if view_type == "table_records":
            flattened["table_view"] = True
            flattened["view_type"] = "table_records"
        else:
            flattened["table_view"] = False
            if options.get("y_quantity") is not None:
                flattened["y_quantity"] = options.get("y_quantity")
        return flattened
    if profile_key == _PLOT_PROFILE_ORIENTATION:
        from .mappings.orientation_mapping import orientation_view_mapping_to_plot_options

        return orientation_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_TEMPERATURE:
        from .mappings.temperature_mapping import temperature_view_mapping_to_plot_options

        return temperature_view_mapping_to_plot_options(mapping)
    if profile_key == _PLOT_PROFILE_TABLE:
        kind = (
            str(
                mapping.fixed_values.get("kind")
                or str(mapping.view_type_id).removeprefix("table_")
                or "line"
            )
            .strip()
            .lower()
            or "line"
        )
        flattened = {"kind": kind}
        if mapping.x is not None:
            flattened["x"] = str(mapping.x)
        if mapping.y is not None:
            flattened["y"] = str(mapping.y)
        bins_token = str(mapping.fixed_values.get("bins") or "").strip()
        if bins_token:
            flattened["bins"] = _parse_numeric_token(bins_token)
        return flattened
    raise ValueError(f"Unsupported plot profile key '{profile_key}'.")


def _default_mapping_settings(profile_key: str) -> dict[str, Any]:
    return _flatten_view_mapping(profile_key, _build_view_mapping(profile_key, {}))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item) for key, item in value.items() if str(key).strip() and item is not None
    }


def _parse_numeric_token(value: str) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric
