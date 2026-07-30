"""Shared spatial filtering for trajectory-backed workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..analysis.surface import _surface_estimate_metadata, estimate_surface_reference
from ..analysis.water import water_molecule_triplets, water_triplet_geometry
from ..utils import axis_to_index

_FILTER_TOL = 1.0e-9


@dataclass(frozen=True)
class SpatialRangeSpec:
    axis_id: str
    lower_token: str
    upper_token: str


@dataclass(frozen=True)
class ResolvedSpatialRange:
    axis_id: str
    requested: str
    resolved_lower: float
    resolved_upper: float
    full_lower: float
    full_upper: float

    @property
    def is_full_range(self) -> bool:
        return bool(
            np.isclose(self.resolved_lower, self.full_lower, atol=_FILTER_TOL, rtol=0.0)
            and np.isclose(self.resolved_upper, self.full_upper, atol=_FILTER_TOL, rtol=0.0)
        )


@dataclass(frozen=True)
class SpatialFilterOptions:
    x_range: str | None = None
    y_range: str | None = None
    z_range: str | None = None
    distance_range: str | None = None
    keep_molecules_intact: bool = False
    surface_axis: str = "z"
    surface_mode: str = "auto"
    surface_elements: tuple[str, ...] | None = None
    include_fixed_surface_atoms: bool = False
    rough_surface_envelope_A: float | None = None

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (self.x_range, self.y_range, self.z_range, self.distance_range)
        )


@dataclass(frozen=True)
class SpatialFilterResult:
    frames: list[Atoms]
    metadata: dict[str, Any]
    filename_suffix: str
    resolved_ranges: dict[str, ResolvedSpatialRange]
    surface_estimate: Any | None = None


def parse_spatial_range(axis_id: str, raw: str | None) -> SpatialRangeSpec | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid {axis_id}-range '{raw}'. Use the form <min:max>, for example 2.0:6.0."
        )
    lower_token = parts[0].strip().lower()
    upper_token = parts[1].strip().lower()
    if not lower_token or not upper_token:
        raise ValueError(
            f"Invalid {axis_id}-range '{raw}'. Both lower and upper bounds are required."
        )
    for token in (lower_token, upper_token):
        if token in {"min", "max"}:
            continue
        try:
            float(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {axis_id}-range bound '{token}'. Use a number or the symbolic bounds min/max."
            ) from exc
    return SpatialRangeSpec(axis_id=axis_id, lower_token=lower_token, upper_token=upper_token)


def append_output_name_suffix(path: Path, suffix: str) -> Path:
    from ..analysis.output_naming import append_hdf5_name_suffix

    return append_hdf5_name_suffix(path, suffix)


def spatial_filter_options_from_mapping(
    values: Mapping[str, Any],
    *,
    surface_axis: str = "z",
    surface_mode: str = "auto",
    surface_elements: Sequence[str] | None = None,
    include_fixed_surface_atoms: bool = False,
    rough_surface_envelope_A: float | None = None,
) -> SpatialFilterOptions:
    return SpatialFilterOptions(
        x_range=None if values.get("x_range") is None else str(values.get("x_range")),
        y_range=None if values.get("y_range") is None else str(values.get("y_range")),
        z_range=None if values.get("z_range") is None else str(values.get("z_range")),
        distance_range=(
            None if values.get("distance_range") is None else str(values.get("distance_range"))
        ),
        keep_molecules_intact=bool(values.get("keep_molecules_intact", False)),
        surface_axis=str(surface_axis).strip().lower() or "z",
        surface_mode=str(surface_mode).strip().lower() or "auto",
        surface_elements=(
            None
            if surface_elements is None
            else tuple(str(value).strip() for value in surface_elements if str(value).strip())
        ),
        include_fixed_surface_atoms=bool(include_fixed_surface_atoms),
        rough_surface_envelope_A=(
            None if rough_surface_envelope_A is None else float(rough_surface_envelope_A)
        ),
    )


def _format_bound_for_filename(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    if "." not in text:
        text += ".0"
    return text


def _wrapped_positions_for_frame(frame: Atoms) -> np.ndarray:
    if len(frame) == 0:
        return np.empty((0, 3), dtype=float)
    try:
        wrapped = frame.get_positions(wrap=True)
    except Exception:
        wrapped = frame.get_positions()
    return np.asarray(wrapped, dtype=float)


def _wrap_points(points: np.ndarray, frame: Atoms) -> np.ndarray:
    data = np.asarray(points, dtype=float)
    if data.size == 0:
        return np.empty((0, 3), dtype=float)
    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return data.copy()
    try:
        fractional = np.linalg.solve(cell.T, data.T).T
    except np.linalg.LinAlgError:
        return data.copy()
    pbc = np.asarray(frame.get_pbc(), dtype=bool)
    for axis_index in range(3):
        if pbc[axis_index]:
            fractional[:, axis_index] = np.mod(fractional[:, axis_index], 1.0)
    return fractional @ cell


def _resolve_bound(token: str, *, full_lower: float, full_upper: float) -> float:
    if token == "min":
        return float(full_lower)
    if token == "max":
        return float(full_upper)
    return float(token)


def _resolve_range(spec: SpatialRangeSpec, values: list[np.ndarray]) -> ResolvedSpatialRange:
    finite_segments = [
        np.asarray(segment, dtype=float)[np.isfinite(np.asarray(segment, dtype=float))]
        for segment in values
        if np.asarray(segment, dtype=float).size > 0
    ]
    if not finite_segments:
        raise ValueError(
            f"Cannot resolve {spec.axis_id}-range because no finite coordinates are available."
        )
    all_values = np.concatenate(finite_segments)
    full_lower = float(np.min(all_values))
    full_upper = float(np.max(all_values))
    lower = _resolve_bound(spec.lower_token, full_lower=full_lower, full_upper=full_upper)
    upper = _resolve_bound(spec.upper_token, full_lower=full_lower, full_upper=full_upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError(f"Resolved {spec.axis_id}-range is invalid: lower={lower}, upper={upper}.")
    return ResolvedSpatialRange(
        axis_id=spec.axis_id,
        requested=f"{spec.lower_token}:{spec.upper_token}",
        resolved_lower=float(lower),
        resolved_upper=float(upper),
        full_lower=full_lower,
        full_upper=full_upper,
    )


def _filter_suffix_from_ranges(resolved_ranges: Sequence[ResolvedSpatialRange]) -> str:
    parts: list[str] = []
    for resolved in resolved_ranges:
        if resolved.is_full_range:
            continue
        axis_label = "dist" if resolved.axis_id == "distance" else resolved.axis_id
        parts.extend(
            [
                axis_label,
                _format_bound_for_filename(resolved.resolved_lower),
                _format_bound_for_filename(resolved.resolved_upper),
            ]
        )
    return "" if not parts else "_" + "_".join(parts)


def _empty_frame_like(frame: Atoms) -> Atoms:
    empty = frame[[]]
    empty.info = dict(frame.info)
    return empty


def _surface_distance_values_for_atoms(
    frames: list[Atoms],
    *,
    options: SpatialFilterOptions,
    precomputed_surface_estimate: Any | None,
) -> tuple[list[np.ndarray], Any, str]:
    axis = str(options.surface_axis).strip().lower() or "z"
    axis_index = axis_to_index(axis)
    estimate = precomputed_surface_estimate
    provenance = "precomputed"
    if estimate is None:
        estimate = estimate_surface_reference(
            frames,
            axis=axis,
            mode=options.surface_mode,
            surface_elements=None
            if options.surface_elements is None
            else list(options.surface_elements),
            include_fixed_surface_atoms=bool(options.include_fixed_surface_atoms),
            surface_options=None,
        )
        provenance = "computed"
    if estimate is None or estimate.per_frame is None:
        raise ValueError(
            "Distance-based spatial filtering requires a usable distance-to-surface reference."
        )
    if estimate.per_frame.shape[0] != len(frames):
        raise ValueError(
            "Distance-based spatial filtering failed because the cached surface reference does not match the trajectory."
        )
    values = [
        np.asarray(frame.positions[:, axis_index], dtype=float)
        - float(estimate.per_frame[frame_index])
        for frame_index, frame in enumerate(frames)
    ]
    return values, estimate, provenance


def _group_centers_and_indices(
    frame: Atoms,
) -> tuple[list[np.ndarray], np.ndarray, str]:
    triplets = water_molecule_triplets(frame)
    used_indices: set[int] = set()
    group_indices: list[np.ndarray] = []
    group_centers: list[np.ndarray] = []
    if triplets.size > 0:
        geometry = water_triplet_geometry(frame, triplets)
        wrapped_com = _wrap_points(np.asarray(geometry.com_positions, dtype=float), frame)
        for triplet_index, triplet in enumerate(np.asarray(triplets, dtype=int)):
            indices = np.asarray(triplet, dtype=int)
            group_indices.append(indices)
            group_centers.append(np.asarray(wrapped_com[triplet_index], dtype=float))
            used_indices.update(int(value) for value in indices)
    wrapped_positions = _wrapped_positions_for_frame(frame)
    for atom_index in range(len(frame)):
        if atom_index in used_indices:
            continue
        group_indices.append(np.asarray([atom_index], dtype=int))
        group_centers.append(np.asarray(wrapped_positions[atom_index], dtype=float))
    centers = np.vstack(group_centers) if group_centers else np.empty((0, 3), dtype=float)
    return group_indices, centers, "water_com_plus_singletons"


def apply_spatial_filter(
    frames: list[Atoms],
    *,
    options: SpatialFilterOptions,
    precomputed_surface_estimate: Any | None = None,
) -> SpatialFilterResult:
    if not options.active:
        return SpatialFilterResult(
            frames=list(frames),
            metadata={
                "used": False,
                "keep_molecules_intact": bool(options.keep_molecules_intact),
            },
            filename_suffix="",
            resolved_ranges={},
            surface_estimate=None,
        )
    if not frames:
        raise ValueError("Spatial filtering requires at least one trajectory frame.")

    range_specs = {
        axis_id: parse_spatial_range(axis_id, raw)
        for axis_id, raw in (
            ("x", options.x_range),
            ("y", options.y_range),
            ("z", options.z_range),
            ("distance", options.distance_range),
        )
    }
    active_specs = {axis_id: spec for axis_id, spec in range_specs.items() if spec is not None}
    if not active_specs:
        return SpatialFilterResult(
            frames=list(frames),
            metadata={
                "used": False,
                "keep_molecules_intact": bool(options.keep_molecules_intact),
            },
            filename_suffix="",
            resolved_ranges={},
            surface_estimate=None,
        )

    wrapped_positions_by_frame = [_wrapped_positions_for_frame(frame) for frame in frames]
    distance_values_by_atoms: list[np.ndarray] | None = None
    surface_estimate = None
    distance_provenance = None
    if "distance" in active_specs:
        distance_values_by_atoms, surface_estimate, distance_provenance = (
            _surface_distance_values_for_atoms(
                frames,
                options=options,
                precomputed_surface_estimate=precomputed_surface_estimate,
            )
        )

    resolved_ranges: dict[str, ResolvedSpatialRange] = {}
    filtered_frames: list[Atoms] = []
    original_atom_counts: list[int] = []
    retained_atom_counts: list[int] = []
    original_group_counts: list[int] = []
    retained_group_counts: list[int] = []

    if options.keep_molecules_intact:
        axis_values_by_groups: dict[str, list[np.ndarray]] = {
            axis_id: [] for axis_id in active_specs
        }
        group_indices_by_frame: list[list[np.ndarray]] = []
        group_centers_by_frame: list[np.ndarray] = []
        distance_values_by_groups: list[np.ndarray] = []
        for frame_index, frame in enumerate(frames):
            group_indices, group_centers, group_mode = _group_centers_and_indices(frame)
            group_indices_by_frame.append(group_indices)
            group_centers_by_frame.append(group_centers)
            original_atom_counts.append(len(frame))
            original_group_counts.append(len(group_indices))
            for axis_id, spec in active_specs.items():
                del spec
                if axis_id == "distance":
                    group_distances = np.asarray(
                        group_centers[:, axis_to_index(options.surface_axis)]
                        - (
                            0.0
                            if surface_estimate is None
                            else float(surface_estimate.per_frame[frame_index])
                        ),
                        dtype=float,
                    )
                    distance_values_by_groups.append(group_distances)
                    axis_values_by_groups[axis_id].append(group_distances)
                else:
                    axis_index = axis_to_index(axis_id)
                    axis_values_by_groups[axis_id].append(
                        np.asarray(group_centers[:, axis_index], dtype=float)
                    )
        for axis_id, spec in active_specs.items():
            resolved_ranges[axis_id] = _resolve_range(spec, axis_values_by_groups[axis_id])

        for frame_index, frame in enumerate(frames):
            group_indices = group_indices_by_frame[frame_index]
            group_centers = group_centers_by_frame[frame_index]
            keep_mask = np.ones(len(group_indices), dtype=bool)
            for axis_id, resolved in resolved_ranges.items():
                values = (
                    distance_values_by_groups[frame_index]
                    if axis_id == "distance"
                    else np.asarray(group_centers[:, axis_to_index(axis_id)], dtype=float)
                )
                keep_mask &= values >= (resolved.resolved_lower - _FILTER_TOL)
                keep_mask &= values <= (resolved.resolved_upper + _FILTER_TOL)
            retained_indices = (
                np.concatenate(
                    [np.asarray(group_indices[idx], dtype=int) for idx in np.where(keep_mask)[0]]
                )
                if np.any(keep_mask)
                else np.empty((0,), dtype=int)
            )
            retained_indices = np.asarray(np.unique(retained_indices), dtype=int)
            filtered_frames.append(
                frame[retained_indices.tolist()]
                if retained_indices.size > 0
                else _empty_frame_like(frame)
            )
            retained_atom_counts.append(int(retained_indices.size))
            retained_group_counts.append(int(np.sum(keep_mask)))
        molecule_selection_mode = group_mode
    else:
        axis_values_by_atoms: dict[str, list[np.ndarray]] = {
            axis_id: [] for axis_id in active_specs
        }
        for frame_index, frame in enumerate(frames):
            wrapped_positions = wrapped_positions_by_frame[frame_index]
            original_atom_counts.append(len(frame))
            retained_group_counts.append(0)
            original_group_counts.append(0)
            for axis_id in active_specs:
                if axis_id == "distance":
                    assert distance_values_by_atoms is not None
                    axis_values_by_atoms[axis_id].append(
                        np.asarray(distance_values_by_atoms[frame_index], dtype=float)
                    )
                else:
                    axis_values_by_atoms[axis_id].append(
                        np.asarray(wrapped_positions[:, axis_to_index(axis_id)], dtype=float)
                    )
        for axis_id, spec in active_specs.items():
            resolved_ranges[axis_id] = _resolve_range(spec, axis_values_by_atoms[axis_id])

        for frame_index, frame in enumerate(frames):
            keep_mask = np.ones(len(frame), dtype=bool)
            wrapped_positions = wrapped_positions_by_frame[frame_index]
            for axis_id, resolved in resolved_ranges.items():
                values = (
                    np.asarray(distance_values_by_atoms[frame_index], dtype=float)
                    if axis_id == "distance"
                    else np.asarray(wrapped_positions[:, axis_to_index(axis_id)], dtype=float)
                )
                keep_mask &= values >= (resolved.resolved_lower - _FILTER_TOL)
                keep_mask &= values <= (resolved.resolved_upper + _FILTER_TOL)
            retained_indices = np.where(keep_mask)[0].astype(int)
            filtered_frames.append(
                frame[retained_indices.tolist()]
                if retained_indices.size > 0
                else _empty_frame_like(frame)
            )
            retained_atom_counts.append(int(retained_indices.size))
        molecule_selection_mode = "atoms"

    if sum(retained_atom_counts) <= 0:
        raise ValueError("Spatial filtering removed all atoms from the trajectory.")

    filename_suffix = _filter_suffix_from_ranges(list(resolved_ranges.values()))
    distance_metadata: dict[str, Any] | None = None
    if "distance" in resolved_ranges:
        distance_metadata = {
            "provenance": distance_provenance,
            "surface_axis": str(options.surface_axis).strip().lower() or "z",
            "surface_mode": str(options.surface_mode).strip().lower() or "auto",
            **_surface_estimate_metadata(surface_estimate),
        }
    metadata: dict[str, Any] = {
        "used": True,
        "keep_molecules_intact": bool(options.keep_molecules_intact),
        "molecule_selection_mode": molecule_selection_mode,
        "pbc_selection_notes": (
            "Wrapped Cartesian coordinates were used for x/y/z bounds when a periodic cell was available."
        ),
        "bounds": {
            axis_id: {
                "requested": resolved.requested,
                "resolved_lower": float(resolved.resolved_lower),
                "resolved_upper": float(resolved.resolved_upper),
                "full_lower": float(resolved.full_lower),
                "full_upper": float(resolved.full_upper),
                "is_full_range": bool(resolved.is_full_range),
            }
            for axis_id, resolved in resolved_ranges.items()
        },
        "original_atom_counts": [int(value) for value in original_atom_counts],
        "retained_atom_counts": [int(value) for value in retained_atom_counts],
        "original_atom_count_total": int(sum(original_atom_counts)),
        "retained_atom_count_total": int(sum(retained_atom_counts)),
        "filename_suffix": filename_suffix,
    }
    if options.keep_molecules_intact:
        metadata["original_molecule_counts"] = [int(value) for value in original_group_counts]
        metadata["retained_molecule_counts"] = [int(value) for value in retained_group_counts]
        metadata["original_molecule_count_total"] = int(sum(original_group_counts))
        metadata["retained_molecule_count_total"] = int(sum(retained_group_counts))
    if distance_metadata is not None:
        metadata["distance"] = distance_metadata
    return SpatialFilterResult(
        frames=filtered_frames,
        metadata=metadata,
        filename_suffix=filename_suffix,
        resolved_ranges=resolved_ranges,
        surface_estimate=surface_estimate,
    )
