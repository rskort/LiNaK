"""Minimal validation helpers for generic plot-data mappings.

This module provides small reusable rules for checking whether quantities from a
``PlotDataContract`` can participate in a generic view mapping. It is
intentionally standalone and does not alter current LiNaK plotting behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from .data_contract import (
    PLOT_VIEW_1D_LINE,
    PLOT_VIEW_2D_HEATMAP,
    PlotDataContract,
    PlotQuantity,
    PlotViewMapping,
    canonical_plot_view_id,
)

MappingStatus = Literal["valid_preferred", "valid_nonpreferred", "invalid"]

_SUPPORTED_GENERIC_VIEW_TYPES = frozenset(
    {
        "line_1d",
        "scatter_2d",
        "trajectory_2d",
        "heatmap_2d",
        PLOT_VIEW_1D_LINE,
        PLOT_VIEW_2D_HEATMAP,
    }
)

_REQUIRED_ROLES_BY_VIEW_TYPE: dict[str, tuple[str, ...]] = {
    "line_1d": ("x", "y"),
    PLOT_VIEW_1D_LINE: ("x", "y"),
    "scatter_2d": ("x", "y"),
    "trajectory_2d": ("x", "y"),
    "heatmap_2d": ("x", "y"),
    PLOT_VIEW_2D_HEATMAP: ("x", "y"),
}

_OPTIONAL_ROLES_BY_VIEW_TYPE: dict[str, tuple[str, ...]] = {
    "line_1d": (),
    PLOT_VIEW_1D_LINE: (),
    "scatter_2d": ("color", "split_by", "filter_by"),
    "trajectory_2d": ("color", "split_by", "filter_by"),
    "heatmap_2d": ("z", "color", "split_by", "filter_by"),
    PLOT_VIEW_2D_HEATMAP: ("z", "color", "split_by", "filter_by"),
}

_NON_QUANTITATIVE_KINDS = frozenset(
    {
        "label",
        "labels",
        "text",
        "string",
        "strings",
        "metadata",
        "annotation",
    }
)

_NONPREFERRED_QUANTITATIVE_KINDS = frozenset(
    {
        "category",
        "categorical",
        "enum",
        "identifier",
        "id",
    }
)


def exact_shape_compatibility(
    contract: PlotDataContract,
    quantity_ids: Iterable[str],
) -> bool:
    """Return whether all referenced quantities have exactly the same shape."""

    quantities = _resolve_quantities(contract, quantity_ids)
    if len(quantities) <= 1:
        return True
    first_dimensions = quantities[0].dimensions
    for quantity in quantities[1:]:
        if quantity.dimensions != first_dimensions:
            return False
    return _shared_dimension_lengths_are_compatible(contract, quantities)


def broadcast_compatibility(
    contract: PlotDataContract,
    quantity_ids: Iterable[str],
) -> bool:
    """Return whether the referenced quantities are broadcast-compatible by named dimensions."""

    quantities = _resolve_quantities(contract, quantity_ids)
    if len(quantities) <= 1:
        return True
    if not _shared_dimension_lengths_are_compatible(contract, quantities):
        return False

    merged_dimension_order: list[str] = []
    for quantity in quantities:
        for dimension_id in quantity.dimensions:
            if dimension_id not in merged_dimension_order:
                merged_dimension_order.append(dimension_id)
    merged_tuple = tuple(merged_dimension_order)
    return all(_is_subsequence(quantity.dimensions, merged_tuple) for quantity in quantities)


def visual_role_compatibility(
    contract: PlotDataContract,
    *,
    view_type_id: str,
    role: str,
    quantity_id: str,
) -> MappingStatus:
    """Classify whether one quantity is suitable for one visual role."""

    normalized_view_type = _validation_view_type_id(contract, view_type_id)
    normalized_role = str(role).strip()
    if normalized_view_type not in _SUPPORTED_GENERIC_VIEW_TYPES:
        return "invalid"
    allowed_roles = set(_REQUIRED_ROLES_BY_VIEW_TYPE[normalized_view_type]) | set(
        _OPTIONAL_ROLES_BY_VIEW_TYPE[normalized_view_type]
    )
    if normalized_role not in allowed_roles:
        return "invalid"
    if normalized_role == "split_by":
        return "invalid"

    quantity = _quantity_by_id(contract, quantity_id)
    kind = str(quantity.kind).strip().lower()
    if kind in _NON_QUANTITATIVE_KINDS:
        return "invalid"
    if kind in _NONPREFERRED_QUANTITATIVE_KINDS:
        return "valid_nonpreferred"
    return "valid_preferred"


def generic_view_type_compatibility(
    contract: PlotDataContract,
    mapping: PlotViewMapping,
) -> MappingStatus:
    """Classify whether one generic view mapping is structurally compatible."""

    view_type_id = _validation_view_type_id(contract, mapping.view_type_id)
    if view_type_id not in _SUPPORTED_GENERIC_VIEW_TYPES:
        return "invalid"
    if not _contract_supports_view_type(contract, view_type_id):
        return "invalid"

    required_roles = _REQUIRED_ROLES_BY_VIEW_TYPE[view_type_id]
    role_assignments = mapping.resolved_role_assignments()
    if any(role not in role_assignments for role in required_roles):
        return "invalid"

    role_statuses = [
        visual_role_compatibility(
            contract,
            view_type_id=view_type_id,
            role=role,
            quantity_id=role_assignments[role],
        )
        for role in role_assignments
        if role != "split_by"
    ]
    if any(status == "invalid" for status in role_statuses):
        return "invalid"
    split_by = role_assignments.get("split_by")
    if split_by is not None and not _dimension_exists(contract, split_by):
        return "invalid"

    if view_type_id in {"line_1d", PLOT_VIEW_1D_LINE}:
        x_quantity = _quantity_by_id(contract, role_assignments["x"])
        y_quantity = _quantity_by_id(contract, role_assignments["y"])
        if exact_shape_compatibility(contract, (x_quantity.id, y_quantity.id)):
            if len(y_quantity.dimensions) == 1:
                return _merge_statuses(role_statuses)
            return _merge_statuses(("valid_nonpreferred", *role_statuses))
        if broadcast_compatibility(contract, (x_quantity.id, y_quantity.id)):
            return _merge_statuses(("valid_nonpreferred", *role_statuses))
        return "invalid"

    if view_type_id in {"scatter_2d", "trajectory_2d"}:
        base_ids = [role_assignments["x"], role_assignments["y"]]
        if "color" in role_assignments:
            base_ids.append(role_assignments["color"])
        if exact_shape_compatibility(contract, base_ids):
            return _merge_statuses(role_statuses)
        if broadcast_compatibility(contract, base_ids):
            return _merge_statuses(("valid_nonpreferred", *role_statuses))
        return "invalid"

    if view_type_id in {"heatmap_2d", PLOT_VIEW_2D_HEATMAP}:
        x_quantity = _quantity_by_id(contract, role_assignments["x"])
        y_quantity = _quantity_by_id(contract, role_assignments["y"])
        z_quantity_id = role_assignments.get("z")
        color_quantity_id = role_assignments.get("color")
        categorical_split = (
            str(mapping.fixed_values.get("projection_render_mode") or "")
            .strip()
            .lower()
            == "line-colors"
            and split_by is not None
        )
        if categorical_split and z_quantity_id is None and color_quantity_id is None:
            if exact_shape_compatibility(contract, (x_quantity.id, y_quantity.id)):
                return _merge_statuses(role_statuses)
            if broadcast_compatibility(contract, (x_quantity.id, y_quantity.id)):
                return _merge_statuses(("valid_nonpreferred", *role_statuses))
            return "invalid"
        if z_quantity_id is not None:
            z_quantity = _quantity_by_id(contract, z_quantity_id)
            if len(x_quantity.dimensions) != 1 or len(y_quantity.dimensions) != 1:
                return "invalid"
            if len(z_quantity.dimensions) != 2:
                return "invalid"
            x_dimension = x_quantity.dimensions[0]
            y_dimension = y_quantity.dimensions[0]
            if z_quantity.dimensions == (x_dimension, y_dimension):
                return _merge_statuses(role_statuses)
            if set(z_quantity.dimensions) == {x_dimension, y_dimension}:
                return _merge_statuses(("valid_nonpreferred", *role_statuses))
            return "invalid"
        if color_quantity_id is not None:
            base_ids = [x_quantity.id, y_quantity.id, color_quantity_id]
            if exact_shape_compatibility(contract, base_ids):
                return _merge_statuses(role_statuses)
            if broadcast_compatibility(contract, base_ids):
                return _merge_statuses(("valid_nonpreferred", *role_statuses))
            return "invalid"
        return "invalid"

    return "invalid"


def _contract_supports_view_type(contract: PlotDataContract, view_type_id: str) -> bool:
    available = {str(view_type.id).strip() for view_type in contract.view_types}
    if not available or view_type_id in available:
        return True
    canonical_target = canonical_plot_view_id(view_type_id)
    return any(canonical_plot_view_id(available_id) == canonical_target for available_id in available)


def _validation_view_type_id(contract: PlotDataContract, view_type_id: str | None) -> str:
    token = str(view_type_id or "").strip()
    if token == PLOT_VIEW_1D_LINE:
        available = {str(view_type.id).strip() for view_type in contract.view_types}
        if PLOT_VIEW_1D_LINE in available:
            return PLOT_VIEW_1D_LINE
        return "line_1d"
    if token == PLOT_VIEW_2D_HEATMAP:
        available = {str(view_type.id).strip() for view_type in contract.view_types}
        if PLOT_VIEW_2D_HEATMAP in available:
            return PLOT_VIEW_2D_HEATMAP
        if "heatmap_2d" in available or not available:
            return "heatmap_2d"
        if "trajectory_2d" in available:
            return "trajectory_2d"
        if "scatter_2d" in available:
            return "scatter_2d"
    return token


def _dimension_length_map(contract: PlotDataContract) -> dict[str, int | None]:
    return {str(dimension.id): dimension.length for dimension in contract.dimensions}


def _dimension_exists(contract: PlotDataContract, dimension_id: str) -> bool:
    target_id = str(dimension_id)
    return any(str(dimension.id) == target_id for dimension in contract.dimensions)


def _quantity_by_id(contract: PlotDataContract, quantity_id: str) -> PlotQuantity:
    target_id = str(quantity_id)
    for quantity in contract.quantities:
        if quantity.id == target_id:
            return quantity
    raise KeyError(f"Unknown plot quantity id '{target_id}'.")


def _resolve_quantities(
    contract: PlotDataContract,
    quantity_ids: Iterable[str],
) -> list[PlotQuantity]:
    return [_quantity_by_id(contract, quantity_id) for quantity_id in quantity_ids]


def _shared_dimension_lengths_are_compatible(
    contract: PlotDataContract,
    quantities: Iterable[PlotQuantity],
) -> bool:
    lengths_by_dimension = _dimension_length_map(contract)
    seen_lengths: dict[str, int] = {}
    for quantity in quantities:
        for dimension_id in quantity.dimensions:
            length = lengths_by_dimension.get(dimension_id)
            if length is None:
                continue
            previous = seen_lengths.get(dimension_id)
            if previous is not None and previous != length:
                return False
            seen_lengths[dimension_id] = length
    return True


def _is_subsequence(candidate: tuple[str, ...], reference: tuple[str, ...]) -> bool:
    if not candidate:
        return True
    index = 0
    for token in reference:
        if token == candidate[index]:
            index += 1
            if index == len(candidate):
                return True
    return False


def _merge_statuses(statuses: Iterable[MappingStatus]) -> MappingStatus:
    normalized = tuple(statuses)
    if any(status == "invalid" for status in normalized):
        return "invalid"
    if any(status == "valid_nonpreferred" for status in normalized):
        return "valid_nonpreferred"
    return "valid_preferred"
