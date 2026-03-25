"""Shared analysis schema and units helpers.

This module centralizes per-analysis metadata conventions so read/write/plot
code paths stay consistent and new analyses can be added with minimal edits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class AnalysisSchema:
    """Canonical metadata shape for one analysis family."""

    analysis: str
    version: int
    default_units_map: Mapping[str, str]
    default_plot_labels: tuple[str, str] | None = None


_ANALYSIS_SCHEMAS: dict[str, AnalysisSchema] = {
    "density": AnalysisSchema(
        analysis="density",
        version=1,
        default_units_map={
            "bin_width_A": "Angstrom",
            "bin_centers_A": "Angstrom",
            "density": "g/cm^3",
            "number_density": "atom/nm^3",
        },
    ),
    "msd": AnalysisSchema(
        analysis="msd",
        version=1,
        default_units_map={
            "time_fs": "fs",
            "time_ps": "ps",
            "msd_A2": "Angstrom^2",
        },
        default_plot_labels=("Time (ps)", "MSD (Angstrom^2)"),
    ),
    "rdf": AnalysisSchema(
        analysis="rdf",
        version=1,
        default_units_map={
            "bin_width_A": "Angstrom",
            "bin_centers_A": "Angstrom",
            "g_r": "dimensionless",
        },
        default_plot_labels=("r (Angstrom)", "g(r)"),
    ),
    "position": AnalysisSchema(
        analysis="position",
        version=1,
        default_units_map={
            "frame_index": "index",
            "step": "step",
            "time_fs": "fs",
            "time_ps": "ps",
            "x_A": "Angstrom",
            "y_A": "Angstrom",
            "z_A": "Angstrom",
            "distance_to_surface_A": "Angstrom",
            "surface_position_per_frame_A": "Angstrom",
        },
        default_plot_labels=("Time (ps)", "Distance to surface (Angstrom)"),
    ),
    "coordination": AnalysisSchema(
        analysis="coordination",
        version=1,
        default_units_map={
            "frame_index": "index",
            "step": "step",
            "time_fs": "fs",
            "time_ps": "ps",
            "distance_to_surface_A": "Angstrom",
            "coordination_number": "dimensionless",
            "surface_position_per_frame_A": "Angstrom",
            "cutoff_A": "Angstrom",
            "cutoff_smoothing_width_A": "Angstrom",
            "cutoff_rdf_bin_centers_A": "Angstrom",
            "cutoff_rdf_g_r": "dimensionless",
            "cutoff_rdf_g_r_smoothed": "dimensionless",
        },
        default_plot_labels=("Distance to surface (Angstrom)", "Coordination number"),
    ),
    "orientation": AnalysisSchema(
        analysis="orientation",
        version=1,
        default_units_map={
            "bin_centers_A": "Angstrom",
            "bin_edges_A": "Angstrom",
            "cos_polar_mean": "dimensionless",
            "cos_azimuthal_mean": "dimensionless",
            "cos_polar_density": "1/Angstrom^3",
            "cos_azimuthal_density": "1/Angstrom^3",
            "density": "1/Angstrom^3",
            "heatmap_polar": "counts",
            "heatmap_azimuthal": "counts",
            "heatmap_angle_bin_centers": "dimensionless",
            "heatmap_angle_bin_edges": "dimensionless",
        },
        default_plot_labels=("Distance to surface (Angstrom)", "cos(theta)"),
    ),
}


def register_analysis_schema(schema: AnalysisSchema) -> None:
    """Register/override one schema entry."""
    _ANALYSIS_SCHEMAS[schema.analysis] = schema


def get_analysis_schema(analysis: str) -> AnalysisSchema:
    """Return schema for an analysis, or a minimal fallback schema."""
    normalized = str(analysis).strip().lower()
    schema = _ANALYSIS_SCHEMAS.get(normalized)
    if schema is not None:
        return schema
    return AnalysisSchema(
        analysis=normalized,
        version=1,
        default_units_map={},
        default_plot_labels=None,
    )


def build_profile_metadata(
    *,
    analysis: str,
    metadata: Mapping[str, Any] | None = None,
    units_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create metadata payload with shared schema markers and merged units."""
    schema = get_analysis_schema(analysis)
    payload = dict(metadata or {})

    resolved_units = dict(schema.default_units_map)
    if units_map:
        for key, value in units_map.items():
            resolved_units[str(key)] = str(value)
    if resolved_units:
        payload["units_map"] = resolved_units

    payload.setdefault("analysis", schema.analysis)
    payload["analysis_schema_version"] = schema.version
    payload.setdefault("profile_uid", uuid4().hex)
    return payload


def resolve_units_map(
    *,
    analysis: str,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve units_map with schema defaults and metadata overrides."""
    schema = get_analysis_schema(analysis)
    resolved = dict(schema.default_units_map)
    raw_units = metadata.get("units_map")
    if isinstance(raw_units, Mapping):
        for key, value in raw_units.items():
            resolved[str(key)] = str(value)
    return resolved


def default_plot_labels(analysis: str) -> tuple[str, str] | None:
    """Return default x/y plot labels declared by schema, if any."""
    return get_analysis_schema(analysis).default_plot_labels


_MASS_UNIT_TO_G_PER_CM3 = {
    "g/cm^3": 1.0,
    "g/Angstrom^3": 1.0e24,
}
_NUMBER_UNIT_TO_ATOM_PER_NM3 = {
    "atom/nm^3": 1.0,
    "atoms/nm^3": 1.0,
    "atom/Angstrom^3": 1.0e3,
    "atoms/Angstrom^3": 1.0e3,
}


def canonicalize_density_units(
    *,
    density: np.ndarray,
    density_units: str,
    number_density: np.ndarray | None,
    number_density_units: str | None,
) -> tuple[np.ndarray, str, np.ndarray | None, str | None]:
    """Convert known legacy volumetric density units to canonical LiNaK units."""
    canonical_density = np.asarray(density, dtype=float)
    canonical_density_units = str(density_units)
    factor = _MASS_UNIT_TO_G_PER_CM3.get(canonical_density_units)
    if factor is not None:
        canonical_density = canonical_density * factor
        canonical_density_units = "g/cm^3"

    canonical_number_density = (
        None if number_density is None else np.asarray(number_density, dtype=float)
    )
    canonical_number_density_units = (
        None if number_density_units is None else str(number_density_units)
    )
    if canonical_number_density is not None and canonical_number_density_units is not None:
        number_factor = _NUMBER_UNIT_TO_ATOM_PER_NM3.get(canonical_number_density_units)
        if number_factor is not None:
            canonical_number_density = canonical_number_density * number_factor
            canonical_number_density_units = "atom/nm^3"

    return (
        canonical_density,
        canonical_density_units,
        canonical_number_density,
        canonical_number_density_units,
    )
