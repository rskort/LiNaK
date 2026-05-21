"""Trajectory I/O helpers built on top of ASE."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import iread, write
from ase.io.formats import UnknownFileTypeError
from ase.io import lammpsrun as ase_lammpsrun

from .lammps import (
    extract_cell_from_lammps_input,
    extract_frame_timestep_fs_from_lammps_input,
    resolve_dump_path_from_lammps_input,
)

from ..progress import ProgressBar

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..analysis.density import SurfaceEstimate

_ASE_LAMMPS_DATA_TO_ASE_ATOMS = getattr(ase_lammpsrun, "lammps_data_to_ase_atoms", None)

_XYZ_LIKE_SUFFIXES = {".xyz", ".extxyz"}
LINAK_TRAJECTORY_HDF5_FORMAT = "linak-trajectory-hdf5"
LINAK_TRAJECTORY_HDF5_VERSION = 1
_TRAJECTORY_INFO_INT_KEYS = ("timestep",)
_TRAJECTORY_INFO_FLOAT_KEYS = (
    "frame_timestep_fs",
    "md_timestep_fs",
    "trajectory_stride_md",
    "time_fs",
)


@dataclass(frozen=True)
class TrajectoryStoredMetadata:
    """Optional simulation-context metadata stored alongside trajectory arrays."""

    input_path: Path | None = None
    input_format: str | None = None
    cell_angstrom: tuple[float, float, float] | None = None
    cell_source: str | None = None
    frame_timestep_fs: float | None = None
    md_timestep_fs: float | None = None
    trajectory_stride_md: int | None = None
    timestep_source: str | None = None
    fixed_atom_indices: tuple[int, ...] = ()
    fixed_atoms_source: str | None = None
    pbc_applied: bool = False
    pbc_cell_angstrom: tuple[float, float, float] | None = None
    pbc_source: str | None = None
    coordinate_basis: str | None = None
    surface_cache_status: str | None = None
    surface_cache_axis: str | None = None
    surface_cache_mode: str | None = None
    surface_cache_elements: tuple[str, ...] | None = None
    surface_cache_include_fixed_surface_atoms: bool = False
    surface_cache_rough_surface_envelope_A: float | None = None
    surface_cache_source: str | None = None
    surface_cache_unavailable_reason: str | None = None
    surface_cache_estimate: SurfaceEstimate | None = None
    combine_source_paths: tuple[str, ...] = ()
    combine_source_file_types: tuple[str, ...] = ()
    combine_timestamp_utc: str | None = None
    combine_total_frames: int | None = None
    combine_conversion_applied: bool | None = None
    combine_linak_version: str | None = None
    selection_user: str | None = None
    selection_kind: str | None = None
    selection_unit: str | None = None
    selection_start_frame: int | None = None
    selection_stop_frame_exclusive: int | None = None
    selection_selected_frame_count: int | None = None
    selection_resolved_start_time_fs: float | None = None
    selection_resolved_end_time_fs: float | None = None
    selection_resolved_start_step: int | None = None
    selection_resolved_end_step: int | None = None
    spatial_filter_metadata: dict[str, Any] | None = None


def default_trajectory_hdf5_output_path(source: str | Path) -> Path:
    """Return the default output path for a converted LiNaK trajectory HDF5."""
    from ..analysis.output_naming import analysis_source_base

    source_path = Path(source).expanduser().resolve()
    output_dir = source_path.parent / "LiNaK_outputs"
    stem = analysis_source_base(source_path, default="trajectory")
    return output_dir / f"{stem}.traj.h5"


def is_linak_trajectory_hdf5(path: str | Path) -> bool:
    """Return ``True`` when ``path`` is a LiNaK trajectory HDF5 container."""
    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() not in {".h5", ".hdf5"}:
        return False
    try:
        with h5py.File(source_path, "r") as handle:
            return (
                str(handle.attrs.get("linak_format", "")).strip() == LINAK_TRAJECTORY_HDF5_FORMAT
                and str(handle.attrs.get("kind", "")).strip() == "trajectory"
            )
    except OSError:
        return False


def _normalize_stored_metadata(
    metadata: TrajectoryStoredMetadata | None,
) -> TrajectoryStoredMetadata | None:
    if metadata is None:
        return None

    input_path = (
        metadata.input_path.expanduser().resolve() if metadata.input_path is not None else None
    )
    input_format = (
        str(metadata.input_format).strip().lower() or None if metadata.input_format else None
    )
    cell_angstrom = (
        cast(
            tuple[float, float, float],
            tuple(float(value) for value in metadata.cell_angstrom),
        )
        if metadata.cell_angstrom is not None
        else None
    )
    cell_source = str(metadata.cell_source).strip() or None if metadata.cell_source else None
    frame_timestep_fs = (
        float(metadata.frame_timestep_fs) if metadata.frame_timestep_fs is not None else None
    )
    md_timestep_fs = float(metadata.md_timestep_fs) if metadata.md_timestep_fs is not None else None
    trajectory_stride_md = (
        int(metadata.trajectory_stride_md) if metadata.trajectory_stride_md is not None else None
    )
    timestep_source = (
        str(metadata.timestep_source).strip() or None if metadata.timestep_source else None
    )
    fixed_atom_indices = tuple(
        sorted({int(index) for index in metadata.fixed_atom_indices if int(index) >= 0})
    )
    fixed_atoms_source = (
        str(metadata.fixed_atoms_source).strip() or None if metadata.fixed_atoms_source else None
    )
    pbc_cell_angstrom = (
        cast(
            tuple[float, float, float],
            tuple(float(value) for value in metadata.pbc_cell_angstrom),
        )
        if metadata.pbc_cell_angstrom is not None
        else None
    )
    pbc_source = str(metadata.pbc_source).strip() or None if metadata.pbc_source else None
    coordinate_basis = (
        str(metadata.coordinate_basis).strip() or None if metadata.coordinate_basis else None
    )
    surface_cache_status = (
        str(metadata.surface_cache_status).strip().lower() or None
        if metadata.surface_cache_status
        else None
    )
    surface_cache_axis = (
        str(metadata.surface_cache_axis).strip().lower() or None
        if metadata.surface_cache_axis
        else None
    )
    surface_cache_mode = (
        str(metadata.surface_cache_mode).strip().lower() or None
        if metadata.surface_cache_mode
        else None
    )
    surface_cache_elements = (
        tuple(str(value).strip() for value in metadata.surface_cache_elements if str(value).strip())
        if metadata.surface_cache_elements is not None
        else None
    )
    surface_cache_source = (
        str(metadata.surface_cache_source).strip() or None
        if metadata.surface_cache_source
        else None
    )
    surface_cache_unavailable_reason = (
        str(metadata.surface_cache_unavailable_reason).strip() or None
        if metadata.surface_cache_unavailable_reason
        else None
    )
    surface_cache_rough_surface_envelope_A = (
        float(metadata.surface_cache_rough_surface_envelope_A)
        if metadata.surface_cache_rough_surface_envelope_A is not None
        else None
    )
    combine_source_paths = tuple(
        str(value).strip() for value in metadata.combine_source_paths if str(value).strip()
    )
    combine_source_file_types = tuple(
        str(value).strip() for value in metadata.combine_source_file_types if str(value).strip()
    )
    combine_timestamp_utc = (
        str(metadata.combine_timestamp_utc).strip() or None
        if metadata.combine_timestamp_utc
        else None
    )
    combine_total_frames = (
        int(metadata.combine_total_frames) if metadata.combine_total_frames is not None else None
    )
    combine_conversion_applied = (
        None
        if metadata.combine_conversion_applied is None
        else bool(metadata.combine_conversion_applied)
    )
    combine_linak_version = (
        str(metadata.combine_linak_version).strip() or None
        if metadata.combine_linak_version
        else None
    )
    selection_user = (
        str(metadata.selection_user).strip() or None if metadata.selection_user else None
    )
    selection_kind = (
        str(metadata.selection_kind).strip() or None if metadata.selection_kind else None
    )
    selection_unit = (
        str(metadata.selection_unit).strip() or None if metadata.selection_unit else None
    )
    selection_start_frame = (
        int(metadata.selection_start_frame) if metadata.selection_start_frame is not None else None
    )
    selection_stop_frame_exclusive = (
        int(metadata.selection_stop_frame_exclusive)
        if metadata.selection_stop_frame_exclusive is not None
        else None
    )
    selection_selected_frame_count = (
        int(metadata.selection_selected_frame_count)
        if metadata.selection_selected_frame_count is not None
        else None
    )
    selection_resolved_start_time_fs = (
        float(metadata.selection_resolved_start_time_fs)
        if metadata.selection_resolved_start_time_fs is not None
        else None
    )
    selection_resolved_end_time_fs = (
        float(metadata.selection_resolved_end_time_fs)
        if metadata.selection_resolved_end_time_fs is not None
        else None
    )
    selection_resolved_start_step = (
        int(metadata.selection_resolved_start_step)
        if metadata.selection_resolved_start_step is not None
        else None
    )
    selection_resolved_end_step = (
        int(metadata.selection_resolved_end_step)
        if metadata.selection_resolved_end_step is not None
        else None
    )
    spatial_filter_metadata = (
        None if metadata.spatial_filter_metadata is None else dict(metadata.spatial_filter_metadata)
    )
    return TrajectoryStoredMetadata(
        input_path=input_path,
        input_format=input_format,
        cell_angstrom=cell_angstrom,
        cell_source=cell_source,
        frame_timestep_fs=frame_timestep_fs,
        md_timestep_fs=md_timestep_fs,
        trajectory_stride_md=trajectory_stride_md,
        timestep_source=timestep_source,
        fixed_atom_indices=fixed_atom_indices,
        fixed_atoms_source=fixed_atoms_source,
        pbc_applied=bool(metadata.pbc_applied),
        pbc_cell_angstrom=pbc_cell_angstrom,
        pbc_source=pbc_source,
        coordinate_basis=coordinate_basis,
        surface_cache_status=surface_cache_status,
        surface_cache_axis=surface_cache_axis,
        surface_cache_mode=surface_cache_mode,
        surface_cache_elements=surface_cache_elements,
        surface_cache_include_fixed_surface_atoms=bool(
            metadata.surface_cache_include_fixed_surface_atoms
        ),
        surface_cache_rough_surface_envelope_A=surface_cache_rough_surface_envelope_A,
        surface_cache_source=surface_cache_source,
        surface_cache_unavailable_reason=surface_cache_unavailable_reason,
        surface_cache_estimate=metadata.surface_cache_estimate,
        combine_source_paths=combine_source_paths,
        combine_source_file_types=combine_source_file_types,
        combine_timestamp_utc=combine_timestamp_utc,
        combine_total_frames=combine_total_frames,
        combine_conversion_applied=combine_conversion_applied,
        combine_linak_version=combine_linak_version,
        selection_user=selection_user,
        selection_kind=selection_kind,
        selection_unit=selection_unit,
        selection_start_frame=selection_start_frame,
        selection_stop_frame_exclusive=selection_stop_frame_exclusive,
        selection_selected_frame_count=selection_selected_frame_count,
        selection_resolved_start_time_fs=selection_resolved_start_time_fs,
        selection_resolved_end_time_fs=selection_resolved_end_time_fs,
        selection_resolved_start_step=selection_resolved_start_step,
        selection_resolved_end_step=selection_resolved_end_step,
        spatial_filter_metadata=spatial_filter_metadata,
    )


def _fixed_atom_indices_from_constraints(frames: list[Atoms]) -> tuple[int, ...]:
    if not frames:
        return ()
    reference_indices: tuple[int, ...] | None = None
    for frame in frames:
        indices: set[int] = set()
        for constraint in getattr(frame, "constraints", ()) or ():
            get_indices = getattr(constraint, "get_indices", None)
            if get_indices is None:
                continue
            try:
                indices.update(int(index) for index in np.asarray(get_indices(), dtype=int).ravel())
            except Exception:  # pragma: no cover - defensive against third-party constraints.
                continue
        current = tuple(sorted(index for index in indices if index >= 0))
        if reference_indices is None:
            reference_indices = current
            continue
        if current != reference_indices:
            return ()
    return reference_indices or ()


def _resolve_write_metadata(
    frames: list[Atoms],
    metadata: TrajectoryStoredMetadata | None,
) -> TrajectoryStoredMetadata | None:
    resolved = _normalize_stored_metadata(metadata)
    if resolved is None:
        fixed_atom_indices = _fixed_atom_indices_from_constraints(frames)
        if not fixed_atom_indices:
            return None
        return TrajectoryStoredMetadata(
            fixed_atom_indices=fixed_atom_indices,
            fixed_atoms_source="trajectory constraints",
        )

    fixed_atom_indices = resolved.fixed_atom_indices or _fixed_atom_indices_from_constraints(frames)
    fixed_atoms_source = resolved.fixed_atoms_source
    if fixed_atom_indices and fixed_atoms_source is None:
        fixed_atoms_source = "trajectory constraints"
    return TrajectoryStoredMetadata(
        input_path=resolved.input_path,
        input_format=resolved.input_format,
        cell_angstrom=resolved.cell_angstrom,
        cell_source=resolved.cell_source,
        frame_timestep_fs=resolved.frame_timestep_fs,
        md_timestep_fs=resolved.md_timestep_fs,
        trajectory_stride_md=resolved.trajectory_stride_md,
        timestep_source=resolved.timestep_source,
        fixed_atom_indices=fixed_atom_indices,
        fixed_atoms_source=fixed_atoms_source,
        pbc_applied=resolved.pbc_applied,
        pbc_cell_angstrom=resolved.pbc_cell_angstrom,
        pbc_source=resolved.pbc_source,
        coordinate_basis=resolved.coordinate_basis,
        surface_cache_status=resolved.surface_cache_status,
        surface_cache_axis=resolved.surface_cache_axis,
        surface_cache_mode=resolved.surface_cache_mode,
        surface_cache_elements=resolved.surface_cache_elements,
        surface_cache_include_fixed_surface_atoms=resolved.surface_cache_include_fixed_surface_atoms,
        surface_cache_rough_surface_envelope_A=resolved.surface_cache_rough_surface_envelope_A,
        surface_cache_source=resolved.surface_cache_source,
        surface_cache_unavailable_reason=resolved.surface_cache_unavailable_reason,
        surface_cache_estimate=resolved.surface_cache_estimate,
        combine_source_paths=resolved.combine_source_paths,
        combine_source_file_types=resolved.combine_source_file_types,
        combine_timestamp_utc=resolved.combine_timestamp_utc,
        combine_total_frames=resolved.combine_total_frames,
        combine_conversion_applied=resolved.combine_conversion_applied,
        combine_linak_version=resolved.combine_linak_version,
        selection_user=resolved.selection_user,
        selection_kind=resolved.selection_kind,
        selection_unit=resolved.selection_unit,
        selection_start_frame=resolved.selection_start_frame,
        selection_stop_frame_exclusive=resolved.selection_stop_frame_exclusive,
        selection_selected_frame_count=resolved.selection_selected_frame_count,
        selection_resolved_start_time_fs=resolved.selection_resolved_start_time_fs,
        selection_resolved_end_time_fs=resolved.selection_resolved_end_time_fs,
        selection_resolved_start_step=resolved.selection_resolved_start_step,
        selection_resolved_end_step=resolved.selection_resolved_end_step,
        spatial_filter_metadata=resolved.spatial_filter_metadata,
    )


def _optional_attr_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _decode_hdf5_string_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8", errors="replace")
    return array.astype(str)


def read_trajectory_hdf5_metadata(path: str | Path) -> TrajectoryStoredMetadata | None:
    """Read stored simulation-context metadata from a LiNaK trajectory HDF5."""
    trajectory_path = Path(path).expanduser().resolve()
    if not is_linak_trajectory_hdf5(trajectory_path):
        return None

    with h5py.File(trajectory_path, "r") as handle:
        metadata_group = handle.get("metadata")
        if not isinstance(metadata_group, h5py.Group):
            return None

        input_path_raw = metadata_group.attrs.get("input_path")
        input_path = (
            Path(str(input_path_raw)).expanduser().resolve()
            if input_path_raw not in (None, "")
            else None
        )
        input_format_raw = metadata_group.attrs.get("input_format")
        input_format = (
            str(input_format_raw).strip().lower() if input_format_raw not in (None, "") else None
        )
        cell_source_raw = metadata_group.attrs.get("cell_source")
        cell_source = str(cell_source_raw).strip() if cell_source_raw not in (None, "") else None
        timestep_source_raw = metadata_group.attrs.get("timestep_source")
        timestep_source = (
            str(timestep_source_raw).strip() if timestep_source_raw not in (None, "") else None
        )
        fixed_atoms_source_raw = metadata_group.attrs.get("fixed_atoms_source")
        fixed_atoms_source = (
            str(fixed_atoms_source_raw).strip()
            if fixed_atoms_source_raw not in (None, "")
            else None
        )

        cell_dataset = metadata_group.get("cell_angstrom")
        cell_angstrom = (
            cast(
                tuple[float, float, float],
                tuple(
                    float(value) for value in np.asarray(cell_dataset, dtype=np.float64).tolist()
                ),
            )
            if isinstance(cell_dataset, h5py.Dataset)
            else None
        )
        fixed_atom_indices_dataset = metadata_group.get("fixed_atom_indices")
        fixed_atom_indices = (
            tuple(
                int(value)
                for value in np.asarray(fixed_atom_indices_dataset, dtype=np.int64).tolist()
            )
            if isinstance(fixed_atom_indices_dataset, h5py.Dataset)
            else ()
        )
        frame_timestep_fs_raw = metadata_group.attrs.get("frame_timestep_fs")
        md_timestep_fs_raw = metadata_group.attrs.get("md_timestep_fs")
        trajectory_stride_md_raw = metadata_group.attrs.get("trajectory_stride_md")
        pbc_applied_raw = metadata_group.attrs.get("pbc_applied")
        pbc_cell_dataset = metadata_group.get("pbc_cell_angstrom")
        pbc_cell_angstrom = (
            cast(
                tuple[float, float, float],
                tuple(
                    float(value)
                    for value in np.asarray(pbc_cell_dataset, dtype=np.float64).tolist()
                ),
            )
            if isinstance(pbc_cell_dataset, h5py.Dataset)
            else None
        )
        pbc_source_raw = metadata_group.attrs.get("pbc_source")
        coordinate_basis_raw = metadata_group.attrs.get("coordinate_basis")

        surface_group = metadata_group.get("surface_cache")
        surface_cache_status: str | None = None
        surface_cache_axis: str | None = None
        surface_cache_mode: str | None = None
        surface_cache_elements: tuple[str, ...] | None = None
        surface_cache_include_fixed_surface_atoms = False
        surface_cache_rough_surface_envelope_A: float | None = None
        surface_cache_source: str | None = None
        surface_cache_unavailable_reason: str | None = None
        combine_source_paths: tuple[str, ...] = ()
        combine_source_file_types: tuple[str, ...] = ()
        combine_timestamp_utc: str | None = None
        combine_linak_version: str | None = None
        selection_user: str | None = None
        selection_kind: str | None = None
        selection_unit: str | None = None
        spatial_filter_metadata: dict[str, Any] | None = None
        if isinstance(surface_group, h5py.Group):
            surface_cache_status = _optional_attr_str(surface_group.attrs.get("status"))
            surface_cache_axis = _optional_attr_str(surface_group.attrs.get("axis"))
            surface_cache_mode = _optional_attr_str(surface_group.attrs.get("surface_mode"))
            surface_elements_dataset = surface_group.get("surface_elements")
            if isinstance(surface_elements_dataset, h5py.Dataset):
                surface_cache_elements = tuple(
                    str(value)
                    for value in _decode_hdf5_string_array(np.asarray(surface_elements_dataset))
                    if str(value)
                )
            surface_cache_include_fixed_surface_atoms = bool(
                surface_group.attrs.get("include_fixed_surface_atoms", False)
            )
            rough_envelope_raw = surface_group.attrs.get("rough_surface_envelope_A")
            surface_cache_rough_surface_envelope_A = (
                float(rough_envelope_raw) if rough_envelope_raw is not None else None
            )
            surface_cache_source = _optional_attr_str(surface_group.attrs.get("source"))
            surface_cache_unavailable_reason = _optional_attr_str(
                surface_group.attrs.get("unavailable_reason")
            )
        combine_source_paths_dataset = metadata_group.get("combine_source_paths")
        if isinstance(combine_source_paths_dataset, h5py.Dataset):
            combine_source_paths = tuple(
                str(value)
                for value in _decode_hdf5_string_array(np.asarray(combine_source_paths_dataset))
                if str(value)
            )
        combine_source_file_types_dataset = metadata_group.get("combine_source_file_types")
        if isinstance(combine_source_file_types_dataset, h5py.Dataset):
            combine_source_file_types = tuple(
                str(value)
                for value in _decode_hdf5_string_array(
                    np.asarray(combine_source_file_types_dataset)
                )
                if str(value)
            )
        combine_timestamp_utc = _optional_attr_str(
            metadata_group.attrs.get("combine_timestamp_utc")
        )
        combine_total_frames_raw = metadata_group.attrs.get("combine_total_frames")
        combine_conversion_applied_raw = metadata_group.attrs.get("combine_conversion_applied")
        combine_linak_version = _optional_attr_str(
            metadata_group.attrs.get("combine_linak_version")
        )
        selection_user = _optional_attr_str(metadata_group.attrs.get("selection_user"))
        selection_kind = _optional_attr_str(metadata_group.attrs.get("selection_kind"))
        selection_unit = _optional_attr_str(metadata_group.attrs.get("selection_unit"))
        selection_start_frame_raw = metadata_group.attrs.get("selection_start_frame")
        selection_stop_frame_exclusive_raw = metadata_group.attrs.get(
            "selection_stop_frame_exclusive"
        )
        selection_selected_frame_count_raw = metadata_group.attrs.get(
            "selection_selected_frame_count"
        )
        selection_resolved_start_time_fs_raw = metadata_group.attrs.get(
            "selection_resolved_start_time_fs"
        )
        selection_resolved_end_time_fs_raw = metadata_group.attrs.get(
            "selection_resolved_end_time_fs"
        )
        selection_resolved_start_step_raw = metadata_group.attrs.get(
            "selection_resolved_start_step"
        )
        selection_resolved_end_step_raw = metadata_group.attrs.get("selection_resolved_end_step")
        spatial_filter_json = _optional_attr_str(metadata_group.attrs.get("spatial_filter_json"))
        spatial_filter_metadata = None
        if spatial_filter_json is not None:
            try:
                loaded_spatial_filter = json.loads(spatial_filter_json)
            except json.JSONDecodeError:
                loaded_spatial_filter = None
            if isinstance(loaded_spatial_filter, dict):
                spatial_filter_metadata = loaded_spatial_filter

        return _normalize_stored_metadata(
            TrajectoryStoredMetadata(
                input_path=input_path,
                input_format=input_format,
                cell_angstrom=cell_angstrom,
                cell_source=cell_source,
                frame_timestep_fs=(
                    float(frame_timestep_fs_raw) if frame_timestep_fs_raw is not None else None
                ),
                md_timestep_fs=float(md_timestep_fs_raw)
                if md_timestep_fs_raw is not None
                else None,
                trajectory_stride_md=(
                    int(trajectory_stride_md_raw) if trajectory_stride_md_raw is not None else None
                ),
                timestep_source=timestep_source,
                fixed_atom_indices=fixed_atom_indices,
                fixed_atoms_source=fixed_atoms_source,
                pbc_applied=bool(pbc_applied_raw),
                pbc_cell_angstrom=pbc_cell_angstrom,
                pbc_source=_optional_attr_str(pbc_source_raw),
                coordinate_basis=_optional_attr_str(coordinate_basis_raw),
                surface_cache_status=surface_cache_status,
                surface_cache_axis=surface_cache_axis,
                surface_cache_mode=surface_cache_mode,
                surface_cache_elements=surface_cache_elements,
                surface_cache_include_fixed_surface_atoms=surface_cache_include_fixed_surface_atoms,
                surface_cache_rough_surface_envelope_A=surface_cache_rough_surface_envelope_A,
                surface_cache_source=surface_cache_source,
                surface_cache_unavailable_reason=surface_cache_unavailable_reason,
                combine_source_paths=combine_source_paths,
                combine_source_file_types=combine_source_file_types,
                combine_timestamp_utc=combine_timestamp_utc,
                combine_total_frames=(
                    int(combine_total_frames_raw) if combine_total_frames_raw is not None else None
                ),
                combine_conversion_applied=(
                    None
                    if combine_conversion_applied_raw is None
                    else bool(combine_conversion_applied_raw)
                ),
                combine_linak_version=combine_linak_version,
                selection_user=selection_user,
                selection_kind=selection_kind,
                selection_unit=selection_unit,
                selection_start_frame=(
                    int(selection_start_frame_raw)
                    if selection_start_frame_raw is not None
                    else None
                ),
                selection_stop_frame_exclusive=(
                    int(selection_stop_frame_exclusive_raw)
                    if selection_stop_frame_exclusive_raw is not None
                    else None
                ),
                selection_selected_frame_count=(
                    int(selection_selected_frame_count_raw)
                    if selection_selected_frame_count_raw is not None
                    else None
                ),
                selection_resolved_start_time_fs=(
                    float(selection_resolved_start_time_fs_raw)
                    if selection_resolved_start_time_fs_raw is not None
                    else None
                ),
                selection_resolved_end_time_fs=(
                    float(selection_resolved_end_time_fs_raw)
                    if selection_resolved_end_time_fs_raw is not None
                    else None
                ),
                selection_resolved_start_step=(
                    int(selection_resolved_start_step_raw)
                    if selection_resolved_start_step_raw is not None
                    else None
                ),
                selection_resolved_end_step=(
                    int(selection_resolved_end_step_raw)
                    if selection_resolved_end_step_raw is not None
                    else None
                ),
                spatial_filter_metadata=spatial_filter_metadata,
            )
        )


def _normalize_surface_cache_elements(
    surface_elements: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if surface_elements is None:
        return None
    from ..analysis.surface import _normalize_surface_elements_argument

    normalized = _normalize_surface_elements_argument(surface_elements)
    return None if normalized is None else tuple(normalized)


def _surface_cache_settings_match(
    group: h5py.Group,
    *,
    axis: str,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    rough_surface_envelope_A: float | None,
) -> bool:
    cached_axis = _optional_attr_str(group.attrs.get("axis"))
    cached_mode = _optional_attr_str(group.attrs.get("surface_mode"))
    if cached_axis is None or cached_axis.lower() != axis.lower():
        return False
    if cached_mode is None or cached_mode.lower() != surface_mode.lower():
        return False
    if bool(group.attrs.get("include_fixed_surface_atoms", False)) != bool(
        include_fixed_surface_atoms
    ):
        return False
    cached_rough_raw = group.attrs.get("rough_surface_envelope_A")
    cached_rough = float(cached_rough_raw) if cached_rough_raw is not None else None
    if cached_rough != rough_surface_envelope_A:
        return False
    cached_elements_dataset = group.get("surface_elements")
    cached_elements = (
        tuple(
            str(value)
            for value in _decode_hdf5_string_array(np.asarray(cached_elements_dataset))
            if str(value)
        )
        if isinstance(cached_elements_dataset, h5py.Dataset)
        else None
    )
    return cached_elements == _normalize_surface_cache_elements(surface_elements)


def _surface_cache_metadata_from_group(group: h5py.Group) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    attr_to_metadata = {
        "surface_position": "surface_position",
        "surface_position_std": "surface_position_std",
        "surface_estimate_mode": "surface_mode",
        "surface_side": "surface_side",
        "surface_method_label": "surface_method_label",
        "surface_valid_fraction": "surface_valid_fraction",
        "surface_median_confidence": "surface_median_confidence",
        "surface_composite_score": "surface_composite_score",
        "surface_low_confidence_threshold": "surface_low_confidence_threshold",
    }
    for attr_name, metadata_name in attr_to_metadata.items():
        if attr_name in group.attrs:
            metadata[metadata_name] = group.attrs[attr_name]
    selected_elements = group.get("surface_selected_elements")
    if isinstance(selected_elements, h5py.Dataset):
        metadata["surface_selected_elements"] = tuple(
            str(value)
            for value in _decode_hdf5_string_array(np.asarray(selected_elements))
            if str(value)
        )
    candidate_indices = group.get("surface_candidate_indices")
    if isinstance(candidate_indices, h5py.Dataset):
        metadata["surface_candidate_indices"] = np.asarray(candidate_indices, dtype=np.int64)
    return metadata


def _read_surface_cache_datasets(group: h5py.Group) -> dict[str, np.ndarray]:
    required = (
        "surface_position_per_frame_A",
        "surface_valid_mask",
        "surface_confidence",
        "surface_provenance",
    )
    missing = [name for name in required if not isinstance(group.get(name), h5py.Dataset)]
    if missing:
        raise ValueError(
            f"Trajectory HDF5 surface cache is malformed: missing dataset(s) {', '.join(missing)}."
        )
    datasets: dict[str, np.ndarray] = {}
    for name, item in group.items():
        if isinstance(item, h5py.Dataset) and name.startswith("surface_"):
            datasets[name] = np.asarray(item)
    return datasets


def read_trajectory_hdf5_surface_cache(
    path: str | Path,
    *,
    axis: str,
    surface_mode: str,
    surface_elements: list[str] | tuple[str, ...] | None,
    include_fixed_surface_atoms: bool,
    rough_surface_envelope_A: float | None,
    frame_count: int,
) -> SurfaceEstimate | None:
    """Return a matching conversion-time surface cache from a trajectory HDF5."""
    trajectory_path = Path(path).expanduser().resolve()
    if not is_linak_trajectory_hdf5(trajectory_path):
        return None

    with h5py.File(trajectory_path, "r") as handle:
        metadata_group = handle.get("metadata")
        if not isinstance(metadata_group, h5py.Group):
            return None
        surface_group = metadata_group.get("surface_cache")
        if not isinstance(surface_group, h5py.Group):
            return None
        status = _optional_attr_str(surface_group.attrs.get("status"))
        if status != "available":
            return None
        if not _surface_cache_settings_match(
            surface_group,
            axis=axis,
            surface_mode=surface_mode,
            surface_elements=surface_elements,
            include_fixed_surface_atoms=include_fixed_surface_atoms,
            rough_surface_envelope_A=rough_surface_envelope_A,
        ):
            LOGGER.debug(
                "Trajectory HDF5 surface cache in '%s' does not match requested settings.",
                trajectory_path,
            )
            return None
        datasets = _read_surface_cache_datasets(surface_group)
        metadata = _surface_cache_metadata_from_group(surface_group)

    from ..analysis.density import _surface_estimate_from_payload

    estimate = _surface_estimate_from_payload(datasets=datasets, metadata=metadata)
    if estimate is None:
        raise ValueError("Trajectory HDF5 surface cache is malformed: no surface estimate data.")
    if estimate.frame_values.shape != (int(frame_count),):
        raise ValueError(
            "Trajectory HDF5 surface cache is malformed: "
            f"surface_position_per_frame_A has shape {estimate.frame_values.shape}, "
            f"expected ({int(frame_count)},)."
        )
    for label, values in (
        ("surface_valid_mask", estimate.valid_mask),
        ("surface_confidence", estimate.confidence),
        ("surface_provenance", estimate.provenance),
    ):
        if np.asarray(values).shape != estimate.frame_values.shape:
            raise ValueError(
                "Trajectory HDF5 surface cache is malformed: "
                f"{label} shape {np.asarray(values).shape} does not match "
                f"surface_position_per_frame_A shape {estimate.frame_values.shape}."
            )
    return estimate


def _collect_frame_info_values(
    frames: list[Atoms],
    *,
    key: str,
    dtype: type[np.floating[Any]] | type[np.integer[Any]] | str,
) -> np.ndarray | None:
    values: list[float | int] = []
    for frame in frames:
        raw_value = frame.info.get(key)
        if raw_value is None or isinstance(raw_value, bool):
            return None
        if not isinstance(raw_value, (int, float, np.integer, np.floating)):
            return None
        values.append(float(raw_value) if dtype == np.float64 else int(raw_value))
    if not values:
        return None
    return np.asarray(values, dtype=dtype)


def _topology_is_fixed(frames: list[Atoms]) -> bool:
    reference = np.asarray(frames[0].get_atomic_numbers(), dtype=np.int64)
    for frame in frames[1:]:
        current = np.asarray(frame.get_atomic_numbers(), dtype=np.int64)
        if len(current) != len(reference) or not np.array_equal(current, reference):
            return False
    return True


def _write_string_dataset(group: h5py.Group, name: str, values: tuple[str, ...]) -> None:
    maxlen = max(1, *(len(value.encode("utf-8")) for value in values))
    group.create_dataset(name, data=np.asarray(list(values), dtype=f"S{maxlen}"))


def _write_trajectory_surface_cache(
    metadata_group: h5py.Group,
    metadata: TrajectoryStoredMetadata,
) -> None:
    if metadata.surface_cache_status is None:
        return

    surface_group = metadata_group.create_group("surface_cache")
    surface_group.attrs["status"] = metadata.surface_cache_status
    if metadata.surface_cache_axis:
        surface_group.attrs["axis"] = metadata.surface_cache_axis
    if metadata.surface_cache_mode:
        surface_group.attrs["surface_mode"] = metadata.surface_cache_mode
    surface_group.attrs["include_fixed_surface_atoms"] = bool(
        metadata.surface_cache_include_fixed_surface_atoms
    )
    if metadata.surface_cache_rough_surface_envelope_A is not None:
        surface_group.attrs["rough_surface_envelope_A"] = (
            metadata.surface_cache_rough_surface_envelope_A
        )
    if metadata.surface_cache_source:
        surface_group.attrs["source"] = metadata.surface_cache_source
    if metadata.surface_cache_unavailable_reason:
        surface_group.attrs["unavailable_reason"] = metadata.surface_cache_unavailable_reason
    if metadata.surface_cache_elements is not None:
        _write_string_dataset(surface_group, "surface_elements", metadata.surface_cache_elements)

    estimate = metadata.surface_cache_estimate
    if metadata.surface_cache_status != "available" or estimate is None:
        return

    from ..analysis.density import _surface_estimate_datasets, _surface_metadata_payload

    nested_metadata = _surface_metadata_payload(
        surface_position=estimate.position,
        surface_position_std=estimate.std,
        estimate=estimate,
    ).get("surface", {})
    if "position" in nested_metadata:
        surface_group.attrs["surface_position"] = float(nested_metadata["position"])
    if "position_std" in nested_metadata:
        surface_group.attrs["surface_position_std"] = float(nested_metadata["position_std"])
    if "mode" in nested_metadata:
        surface_group.attrs["surface_estimate_mode"] = str(nested_metadata["mode"])
    if "side" in nested_metadata:
        surface_group.attrs["surface_side"] = str(nested_metadata["side"])
    if "method_label" in nested_metadata:
        surface_group.attrs["surface_method_label"] = str(nested_metadata["method_label"])
    for key in (
        "valid_fraction",
        "median_confidence",
        "composite_score",
        "low_confidence_threshold",
    ):
        if key in nested_metadata and nested_metadata[key] is not None:
            surface_group.attrs[f"surface_{key}"] = float(nested_metadata[key])
    selected_elements = tuple(str(value) for value in nested_metadata.get("selected_elements", ()))
    if selected_elements:
        _write_string_dataset(surface_group, "surface_selected_elements", selected_elements)
    candidate_indices = nested_metadata.get("candidate_indices")
    if candidate_indices is not None:
        surface_group.create_dataset(
            "surface_candidate_indices",
            data=np.asarray(candidate_indices, dtype=np.int64),
        )

    for name, values in _surface_estimate_datasets(estimate).items():
        if values is None:
            continue
        array = np.asarray(values)
        compression = "lzf" if array.ndim > 0 and array.size > 0 else None
        surface_group.create_dataset(name, data=array, compression=compression)


def _write_trajectory_hdf5(
    frames: list[Atoms],
    output_path: Path,
    *,
    source_path: str | Path | None = None,
    source_format: str | None = None,
    metadata: TrajectoryStoredMetadata | None = None,
) -> Path:
    frame_count = len(frames)
    atom_counts = np.asarray([len(frame) for frame in frames], dtype=np.int64)
    if atom_counts.size == 0 or int(np.max(atom_counts)) <= 0:
        raise ValueError("Converted trajectory HDF5 requires at least one atom in the trajectory.")
    topology_is_fixed = _topology_is_fixed(frames)
    stored_metadata = _resolve_write_metadata(frames, metadata)
    if not topology_is_fixed and (
        stored_metadata is None or stored_metadata.spatial_filter_metadata is None
    ):
        raise ValueError(
            "Converted trajectory HDF5 supports fixed topology only unless a spatial filter "
            "produced variable per-frame atom counts."
        )
    max_atom_count = int(np.max(atom_counts))
    chunk_frames = max(1, min(frame_count, 64))

    positions = np.zeros((frame_count, max_atom_count, 3), dtype=np.float64)
    cells = np.empty((frame_count, 3, 3), dtype=np.float64)
    pbc = np.empty((frame_count, 3), dtype=bool)
    if topology_is_fixed:
        atomic_numbers: np.ndarray = np.asarray(frames[0].get_atomic_numbers(), dtype=np.int64)
    else:
        atomic_numbers = np.zeros((frame_count, max_atom_count), dtype=np.int64)
    for index, frame in enumerate(frames):
        atom_count = len(frame)
        if atom_count > 0:
            positions[index, :atom_count] = np.asarray(frame.get_positions(), dtype=np.float64)
        if not topology_is_fixed and atom_count > 0:
            atomic_numbers[index, :atom_count] = np.asarray(
                frame.get_atomic_numbers(),
                dtype=np.int64,
            )
        cells[index] = np.asarray(frame.cell.array, dtype=np.float64)
        pbc[index] = np.asarray(frame.get_pbc(), dtype=bool)

    info_arrays: dict[str, np.ndarray] = {}
    for key in _TRAJECTORY_INFO_INT_KEYS:
        values = _collect_frame_info_values(frames, key=key, dtype=np.int64)
        if values is not None:
            info_arrays[key] = values
    for key in _TRAJECTORY_INFO_FLOAT_KEYS:
        values = _collect_frame_info_values(frames, key=key, dtype=np.float64)
        if values is not None:
            info_arrays[key] = values
    with ProgressBar(desc="Writing trajectory", total=frame_count, unit="frame") as progress:
        with h5py.File(output_path, "w") as handle:
            handle.attrs["linak_format"] = LINAK_TRAJECTORY_HDF5_FORMAT
            handle.attrs["linak_trajectory_version"] = LINAK_TRAJECTORY_HDF5_VERSION
            handle.attrs["kind"] = "trajectory"
            handle.attrs["frame_count"] = frame_count
            handle.attrs["atom_count"] = max_atom_count
            handle.attrs["topology_mode"] = "fixed" if topology_is_fixed else "variable"
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            if source_path is not None:
                handle.attrs["source_path"] = str(Path(source_path).expanduser().resolve())
            if source_format:
                handle.attrs["source_format"] = str(source_format)

            handle.create_dataset(
                "positions",
                data=positions,
                chunks=(chunk_frames, max_atom_count, 3),
                compression="lzf",
                shuffle=True,
            )
            handle.create_dataset(
                "cell",
                data=cells,
                chunks=(chunk_frames, 3, 3),
                compression="lzf",
                shuffle=True,
            )
            handle.create_dataset(
                "pbc",
                data=pbc,
                chunks=(chunk_frames, 3),
                compression="lzf",
            )
            handle.create_dataset(
                "atomic_numbers",
                data=atomic_numbers.astype(np.int64),
                compression="lzf" if not topology_is_fixed else None,
                shuffle=not topology_is_fixed,
            )
            if not topology_is_fixed:
                handle.create_dataset(
                    "atom_counts",
                    data=atom_counts,
                    chunks=(chunk_frames,),
                    compression="lzf",
                    shuffle=True,
                )
            if info_arrays:
                info_group = handle.create_group("frame_info")
                for key, values in info_arrays.items():
                    info_group.create_dataset(
                        key,
                        data=values,
                        chunks=(chunk_frames,),
                        compression="lzf",
                        shuffle=values.dtype.kind in {"i", "u", "f"},
                    )
            if stored_metadata is not None:
                metadata_group = handle.create_group("metadata")
                if stored_metadata.input_path is not None:
                    metadata_group.attrs["input_path"] = str(stored_metadata.input_path)
                if stored_metadata.input_format:
                    metadata_group.attrs["input_format"] = stored_metadata.input_format
                if stored_metadata.cell_source:
                    metadata_group.attrs["cell_source"] = stored_metadata.cell_source
                if stored_metadata.timestep_source:
                    metadata_group.attrs["timestep_source"] = stored_metadata.timestep_source
                if stored_metadata.fixed_atoms_source:
                    metadata_group.attrs["fixed_atoms_source"] = stored_metadata.fixed_atoms_source
                if stored_metadata.pbc_applied:
                    metadata_group.attrs["pbc_applied"] = True
                if stored_metadata.pbc_source:
                    metadata_group.attrs["pbc_source"] = stored_metadata.pbc_source
                if stored_metadata.coordinate_basis:
                    metadata_group.attrs["coordinate_basis"] = stored_metadata.coordinate_basis
                if stored_metadata.frame_timestep_fs is not None:
                    metadata_group.attrs["frame_timestep_fs"] = stored_metadata.frame_timestep_fs
                if stored_metadata.md_timestep_fs is not None:
                    metadata_group.attrs["md_timestep_fs"] = stored_metadata.md_timestep_fs
                if stored_metadata.trajectory_stride_md is not None:
                    metadata_group.attrs["trajectory_stride_md"] = (
                        stored_metadata.trajectory_stride_md
                    )
                if stored_metadata.cell_angstrom is not None:
                    metadata_group.create_dataset(
                        "cell_angstrom",
                        data=np.asarray(stored_metadata.cell_angstrom, dtype=np.float64),
                    )
                if stored_metadata.pbc_cell_angstrom is not None:
                    metadata_group.create_dataset(
                        "pbc_cell_angstrom",
                        data=np.asarray(stored_metadata.pbc_cell_angstrom, dtype=np.float64),
                    )
                if stored_metadata.fixed_atom_indices:
                    metadata_group.create_dataset(
                        "fixed_atom_indices",
                        data=np.asarray(stored_metadata.fixed_atom_indices, dtype=np.int64),
                    )
                if stored_metadata.combine_source_paths:
                    _write_string_dataset(
                        metadata_group,
                        "combine_source_paths",
                        stored_metadata.combine_source_paths,
                    )
                if stored_metadata.combine_source_file_types:
                    _write_string_dataset(
                        metadata_group,
                        "combine_source_file_types",
                        stored_metadata.combine_source_file_types,
                    )
                if stored_metadata.combine_timestamp_utc:
                    metadata_group.attrs["combine_timestamp_utc"] = (
                        stored_metadata.combine_timestamp_utc
                    )
                if stored_metadata.combine_total_frames is not None:
                    metadata_group.attrs["combine_total_frames"] = (
                        stored_metadata.combine_total_frames
                    )
                if stored_metadata.combine_conversion_applied is not None:
                    metadata_group.attrs["combine_conversion_applied"] = bool(
                        stored_metadata.combine_conversion_applied
                    )
                if stored_metadata.combine_linak_version:
                    metadata_group.attrs["combine_linak_version"] = (
                        stored_metadata.combine_linak_version
                    )
                if stored_metadata.selection_user:
                    metadata_group.attrs["selection_user"] = stored_metadata.selection_user
                if stored_metadata.selection_kind:
                    metadata_group.attrs["selection_kind"] = stored_metadata.selection_kind
                if stored_metadata.selection_unit:
                    metadata_group.attrs["selection_unit"] = stored_metadata.selection_unit
                if stored_metadata.selection_start_frame is not None:
                    metadata_group.attrs["selection_start_frame"] = (
                        stored_metadata.selection_start_frame
                    )
                if stored_metadata.selection_stop_frame_exclusive is not None:
                    metadata_group.attrs["selection_stop_frame_exclusive"] = (
                        stored_metadata.selection_stop_frame_exclusive
                    )
                if stored_metadata.selection_selected_frame_count is not None:
                    metadata_group.attrs["selection_selected_frame_count"] = (
                        stored_metadata.selection_selected_frame_count
                    )
                if stored_metadata.selection_resolved_start_time_fs is not None:
                    metadata_group.attrs["selection_resolved_start_time_fs"] = (
                        stored_metadata.selection_resolved_start_time_fs
                    )
                if stored_metadata.selection_resolved_end_time_fs is not None:
                    metadata_group.attrs["selection_resolved_end_time_fs"] = (
                        stored_metadata.selection_resolved_end_time_fs
                    )
                if stored_metadata.selection_resolved_start_step is not None:
                    metadata_group.attrs["selection_resolved_start_step"] = (
                        stored_metadata.selection_resolved_start_step
                    )
                if stored_metadata.selection_resolved_end_step is not None:
                    metadata_group.attrs["selection_resolved_end_step"] = (
                        stored_metadata.selection_resolved_end_step
                    )
                if stored_metadata.spatial_filter_metadata is not None:
                    metadata_group.attrs["spatial_filter_json"] = json.dumps(
                        stored_metadata.spatial_filter_metadata,
                        sort_keys=True,
                    )
                _write_trajectory_surface_cache(metadata_group, stored_metadata)
            progress.update(frame_count)

    return output_path


def _build_atoms_from_hdf5_chunk(
    *,
    atomic_numbers: np.ndarray,
    positions_chunk: np.ndarray,
    cell_chunk: np.ndarray,
    pbc_chunk: np.ndarray,
    frame_info_chunks: dict[str, np.ndarray],
    fixed_atom_indices: tuple[int, ...],
    atom_counts_chunk: np.ndarray | None = None,
) -> list[Atoms]:
    chunk: list[Atoms] = []
    for offset in range(len(positions_chunk)):
        if atom_counts_chunk is None:
            frame_atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
            frame_positions = np.asarray(positions_chunk[offset], dtype=np.float64)
        else:
            atom_count = int(atom_counts_chunk[offset])
            frame_atomic_numbers = np.asarray(atomic_numbers[offset, :atom_count], dtype=np.int64)
            frame_positions = np.asarray(positions_chunk[offset, :atom_count], dtype=np.float64)
        frame = Atoms(
            numbers=frame_atomic_numbers,
            positions=frame_positions,
            cell=cell_chunk[offset],
            pbc=tuple(bool(value) for value in pbc_chunk[offset]),
        )
        for key, values in frame_info_chunks.items():
            scalar = values[offset]
            if np.issubdtype(values.dtype, np.integer):
                frame.info[key] = int(scalar)
            elif np.issubdtype(values.dtype, np.floating):
                frame.info[key] = float(scalar)
            else:  # pragma: no cover - guarded by writer dtype choices.
                frame.info[key] = scalar.item()
        if fixed_atom_indices:
            frame.set_constraint(FixAtoms(indices=list(fixed_atom_indices)))
        chunk.append(frame)
    return chunk


def _read_trajectory_hdf5_chunks(path: Path, *, chunk_size: int) -> Iterator[list[Atoms]]:
    with h5py.File(path, "r") as handle:
        if (
            str(handle.attrs.get("linak_format", "")).strip() != LINAK_TRAJECTORY_HDF5_FORMAT
            or str(handle.attrs.get("kind", "")).strip() != "trajectory"
        ):
            raise ValueError(f"Unsupported trajectory HDF5 format in '{path}'.")

        positions = handle["positions"]
        cells = handle["cell"]
        pbc = handle["pbc"]
        atomic_numbers = np.asarray(handle["atomic_numbers"], dtype=np.int64)
        atom_counts_dataset = handle.get("atom_counts")
        frame_count = int(handle.attrs.get("frame_count", positions.shape[0]))
        info_group = handle.get("frame_info")
        info_names = list(info_group.keys()) if isinstance(info_group, h5py.Group) else []
        stored_metadata = read_trajectory_hdf5_metadata(path)
        fixed_atom_indices = (
            stored_metadata.fixed_atom_indices if stored_metadata is not None else ()
        )
        if isinstance(atom_counts_dataset, h5py.Dataset):
            fixed_atom_indices = ()

        with ProgressBar(desc="Reading trajectory", total=frame_count, unit="frame") as progress:
            for start in range(0, frame_count, chunk_size):
                stop = min(start + chunk_size, frame_count)
                positions_chunk = np.asarray(positions[start:stop], dtype=np.float64)
                cell_chunk = np.asarray(cells[start:stop], dtype=np.float64)
                pbc_chunk = np.asarray(pbc[start:stop], dtype=bool)
                atom_counts_chunk = (
                    None
                    if not isinstance(atom_counts_dataset, h5py.Dataset)
                    else np.asarray(atom_counts_dataset[start:stop], dtype=np.int64)
                )
                frame_info_chunks = (
                    {key: np.asarray(info_group[key][start:stop]) for key in info_names}
                    if isinstance(info_group, h5py.Group)
                    else {}
                )
                chunk = _build_atoms_from_hdf5_chunk(
                    atomic_numbers=(
                        atomic_numbers
                        if atom_counts_chunk is None
                        else np.asarray(atomic_numbers[start:stop], dtype=np.int64)
                    ),
                    positions_chunk=positions_chunk,
                    cell_chunk=cell_chunk,
                    pbc_chunk=pbc_chunk,
                    frame_info_chunks=frame_info_chunks,
                    fixed_atom_indices=fixed_atom_indices,
                    atom_counts_chunk=atom_counts_chunk,
                )
                progress.update(len(chunk))
                yield chunk


def _read_trajectory_hdf5(path: Path) -> list[Atoms]:
    frames: list[Atoms] = []
    for chunk in _read_trajectory_hdf5_chunks(path, chunk_size=256):
        frames.extend(chunk)
    return frames


def _lammps_data_to_ase_atoms(
    data: np.ndarray,
    colnames: list[str],
    cell: np.ndarray,
    celldisp: np.ndarray,
    *,
    pbc: tuple[bool, bool, bool] = (False, False, False),
    atomsobj: type[Atoms] = Atoms,
    order: bool = True,
    specorder: list[str] | None = None,
    units: str = "metal",
) -> Atoms:
    """Compatibility wrapper for ASE's removed ``lammps_data_to_ase_atoms`` helper."""
    if _ASE_LAMMPS_DATA_TO_ASE_ATOMS is not None:
        return _ASE_LAMMPS_DATA_TO_ASE_ATOMS(
            data=data,
            colnames=colnames,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
            atomsobj=atomsobj,
            order=order,
            specorder=specorder,
            units=units,
        )

    if len(data.shape) == 1:
        data = data[np.newaxis, :]

    if "id" in colnames and order:
        ids = data[:, colnames.index("id")].astype(int)
        data = data[np.argsort(ids), :]

    if "element" in colnames:
        elements = data[:, colnames.index("element")]
    elif "mass" in colnames:
        mass_to_element = getattr(ase_lammpsrun, "_mass2element", None)
        if mass_to_element is None:
            raise ValueError("ASE does not expose mass-to-element conversion for LAMMPS dumps.")
        elements = [mass_to_element(m) for m in data[:, colnames.index("mass")].astype(float)]
    elif "type" in colnames:
        elements = data[:, colnames.index("type")].astype(int)
        if specorder is not None:
            elements = [specorder[int(value) - 1] for value in elements]
    else:
        raise ValueError("Cannot determine atom types from LAMMPS dump file.")

    convert = getattr(ase_lammpsrun, "convert", None)

    def get_quantity(labels: list[str], quantity: str | None = None) -> np.ndarray | None:
        try:
            cols = [colnames.index(label) for label in labels]
        except ValueError:
            return None

        values = data[:, cols].astype(float)
        if quantity is not None and convert is not None:
            return convert(values, quantity, units, "ASE")
        return values

    positions = None
    scaled_positions = None
    if "x" in colnames:
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output.")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")

    if convert is not None:
        cell = convert(cell, "distance", units, "ASE")
        celldisp = convert(celldisp, "distance", units, "ASE")

    if positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )
    elif scaled_positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )
    else:  # pragma: no cover - guarded by position checks above.
        raise ValueError("No usable coordinates found in LAMMPS dump.")

    if velocities is not None:
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges([float(charge[0]) for charge in charges])
    if forces is not None:
        out_atoms.calc = SinglePointCalculator(out_atoms, energy=0.0, forces=forces)

    if "type" in colnames:
        out_atoms.new_array("type", data[:, colnames.index("type")], dtype="int")

    return out_atoms


def _parse_box_bound(
    line: str, box_rows: list[str]
) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    """Parse a LAMMPS ``ITEM: BOX BOUNDS`` block."""
    tilt_items = line.split()[3:]
    celldata = np.loadtxt(box_rows, dtype=float, ndmin=2)
    diagdisp = celldata[:, :2].reshape(6, 1).flatten()

    if celldata.shape[1] > 2:
        offdiag = celldata[:, 2].astype(float)
        if len(tilt_items) >= 3:
            sort_index = [tilt_items.index(item) for item in ("xy", "xz", "yz")]
            offdiag = offdiag[sort_index]
        xy, xz, yz = (float(value) for value in offdiag)
    else:
        xy, xz, yz = 0.0, 0.0, 0.0

    xlo, xhi, ylo, yhi, zlo, zhi = (float(value) for value in diagdisp)
    xlo_bound = xlo - min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi - max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo - min(0.0, yz)
    yhi_bound = yhi - max(0.0, yz)
    zlo_bound = zlo
    zhi_bound = zhi

    cell = np.array(
        [
            [xhi_bound - xlo_bound, 0.0, 0.0],
            [xy, yhi_bound - ylo_bound, 0.0],
            [xz, yz, zhi_bound - zlo_bound],
        ],
        dtype=float,
    )
    celldisp = np.array([xlo_bound, ylo_bound, zlo_bound], dtype=float)

    if len(tilt_items) == 3:
        pbc_items = tilt_items
    elif len(tilt_items) > 3:
        pbc_items = tilt_items[3:6]
    else:
        pbc_items = ["f", "f", "f"]
    pbc = cast(tuple[bool, bool, bool], tuple("p" in item.lower() for item in pbc_items))
    return cell, celldisp, pbc


def _read_frames(
    path: Path,
    *,
    format: str | None = None,
    total_frames: int | None = None,
) -> list[Atoms]:
    frames: list[Atoms] = []
    with ProgressBar(desc="Reading trajectory", total=total_frames, unit="frame") as progress:
        for frame in iread(str(path), index=":", format=format):
            frames.append(frame)
            progress.update()
    return frames


def _count_lammps_dump_frames(path: Path) -> int:
    """Count frames in a LAMMPS text dump by scanning timestep markers."""
    frame_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: TIMESTEP"):
                frame_count += 1
    return frame_count


def _count_xyz_like_frames(path: Path) -> int:
    """Count frames in XYZ-like text trajectories without parsing Atoms objects."""
    frame_count = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            stripped = natoms_line.strip()
            if not stripped:
                continue
            try:
                atom_count = int(stripped)
            except ValueError:
                raise ValueError(
                    f"Could not resolve XYZ-style frame count from '{path}': "
                    f"expected atom count line, got {stripped!r}."
                ) from None
            comment_line = handle.readline()
            if comment_line == "":
                raise ValueError(f"Incomplete XYZ trajectory '{path}': missing comment line.")
            for _ in range(atom_count):
                atom_line = handle.readline()
                if atom_line == "":
                    raise ValueError(f"Incomplete XYZ trajectory '{path}': truncated atom block.")
            frame_count += 1
    return frame_count


def _read_lammps_dump_frames(path: Path) -> list[Atoms]:
    """Read a LAMMPS text dump frame-by-frame to keep progress responsive."""
    frames: list[Atoms] = []
    total_frames = _count_lammps_dump_frames(path)
    n_atoms = 0
    cell = None
    celldisp = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    info: dict[str, int] = {}

    with (
        path.open("r", encoding="utf-8") as handle,
        ProgressBar(
            desc="Reading trajectory",
            total=total_frames,
            unit="frame",
        ) as progress,
    ):
        while True:
            line = handle.readline()
            if not line:
                break

            if line.startswith("ITEM: TIMESTEP"):
                timestep_line = handle.readline()
                if not timestep_line:
                    raise ValueError(f"Incomplete LAMMPS dump '{path}': missing timestep value.")
                info["timestep"] = int(timestep_line.split()[0])
                continue

            if line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms_line = handle.readline()
                if not natoms_line:
                    raise ValueError(f"Incomplete LAMMPS dump '{path}': missing atom count value.")
                n_atoms = int(natoms_line.split()[0])
                continue

            if line.startswith("ITEM: BOX BOUNDS"):
                cell_lines = [handle.readline() for _ in range(3)]
                if any(not entry for entry in cell_lines):
                    raise ValueError(f"Incomplete LAMMPS dump '{path}': missing box bounds rows.")
                cell, celldisp, pbc = _parse_box_bound(line, cell_lines)
                continue

            if line.startswith("ITEM: ATOMS"):
                if n_atoms <= 0:
                    raise ValueError(
                        f"Incomplete LAMMPS dump '{path}': ITEM: NUMBER OF ATOMS must "
                        "precede ITEM: ATOMS."
                    )
                colnames = line.split()[2:]
                datarows = [handle.readline() for _ in range(n_atoms)]
                if any(not row for row in datarows):
                    raise ValueError(f"Incomplete LAMMPS dump '{path}': truncated atom table.")
                data = np.loadtxt(datarows, dtype=str, ndmin=2)
                frame = _lammps_data_to_ase_atoms(
                    data=data,
                    colnames=colnames,
                    cell=cell,
                    celldisp=celldisp,
                    atomsobj=Atoms,
                    pbc=pbc,
                )
                frame.info.update(info)
                frames.append(frame)
                progress.update()

    return frames


def read_trajectory_chunks(path: str | Path, *, chunk_size: int) -> Iterator[list[Atoms]]:
    """Yield trajectory frames in fixed-size chunks.

    This allows analyses to stream large trajectories without materializing all frames.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    trajectory_path = Path(path).expanduser().resolve()
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    from ..out_h5 import is_linak_out_hdf5, read_out_h5_trajectory_chunks

    if is_linak_out_hdf5(trajectory_path):
        yield from read_out_h5_trajectory_chunks(trajectory_path, chunk_size=chunk_size)
        return

    if is_linak_trajectory_hdf5(trajectory_path):
        yield from _read_trajectory_hdf5_chunks(trajectory_path, chunk_size=chunk_size)
        return

    suffix = trajectory_path.suffix.lower()
    if suffix == ".lmp":
        frames = _read_lammps_input_trajectory(trajectory_path)
        for start in range(0, len(frames), chunk_size):
            yield frames[start : start + chunk_size]
        return

    if suffix == ".dump":
        frames = _read_lammps_dump_frames(trajectory_path)
        for start in range(0, len(frames), chunk_size):
            yield frames[start : start + chunk_size]
        return

    total_frames = _count_xyz_like_frames(trajectory_path) if suffix in _XYZ_LIKE_SUFFIXES else None
    with ProgressBar(desc="Reading trajectory", total=total_frames, unit="frame") as progress:
        chunk: list[Atoms] = []
        for frame in iread(str(trajectory_path), index=":"):
            chunk.append(frame)
            progress.update()
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _frame_has_usable_cell(frame: Atoms) -> bool:
    if not all(bool(value) for value in frame.get_pbc()):
        return False
    try:
        volume = abs(float(frame.get_volume()))
    except Exception:
        return False
    if volume <= 0.0:
        return False
    return all(length > 0.0 for length in frame.cell.lengths())


def _set_lammps_timestep_metadata(frames: list[Atoms], *, input_path: Path) -> None:
    try:
        frame_timestep_fs, md_timestep_fs, stride_md = extract_frame_timestep_fs_from_lammps_input(
            input_path
        )
    except Exception as exc:
        LOGGER.debug(
            "Could not extract timestep metadata from LAMMPS input '%s': %s", input_path, exc
        )
        return

    for frame in frames:
        frame.info.setdefault("frame_timestep_fs", frame_timestep_fs)
        frame.info.setdefault("md_timestep_fs", md_timestep_fs)
        frame.info.setdefault("trajectory_stride_md", stride_md)
        raw_timestep = frame.info.get("timestep")
        if isinstance(raw_timestep, (int, float)):
            frame.info.setdefault("time_fs", float(raw_timestep) * md_timestep_fs)


def _set_lammps_cell_from_input_if_missing(frames: list[Atoms], *, input_path: Path) -> None:
    if not frames:
        return
    if all(_frame_has_usable_cell(frame) for frame in frames):
        return

    try:
        cell = extract_cell_from_lammps_input(input_path)
    except Exception as exc:
        LOGGER.debug("Could not extract cell from LAMMPS input '%s': %s", input_path, exc)
        return

    for frame in frames:
        frame.set_cell(cell)
        frame.set_pbc((True, True, True))
    LOGGER.info(
        "Applied orthorhombic cell from LAMMPS input '%s': A=%.6g, B=%.6g, C=%.6g Angstrom.",
        input_path,
        cell[0],
        cell[1],
        cell[2],
    )


def _read_lammps_input_trajectory(input_path: Path) -> list[Atoms]:
    dump_path, _ = resolve_dump_path_from_lammps_input(input_path)
    LOGGER.info("Resolved LAMMPS dump '%s' from input '%s'.", dump_path, input_path)
    frames = _read_lammps_dump_frames(dump_path)
    _set_lammps_timestep_metadata(frames, input_path=input_path)
    _set_lammps_cell_from_input_if_missing(frames, input_path=input_path)
    return frames


def read_trajectory(path: str | Path) -> list[Atoms]:
    """Read all frames from a trajectory file.

    Parameters
    ----------
    path
        Path to a trajectory file.
        Supported values include ASE-supported trajectory files (e.g. `.xyz`, `.dump`),
        LiNaK trajectory HDF5 files (`.traj.h5`), and LAMMPS input files (`.lmp`) that
        reference a dump file.

    Returns
    -------
    list[ase.Atoms]
        Frames in the trajectory.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no frames can be read.
    """
    trajectory_path = Path(path).expanduser().resolve()
    try:
        _display = os.path.relpath(trajectory_path)
    except ValueError:
        _display = str(trajectory_path)
    LOGGER.info("Loading trajectory from '%s'.", _display)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    from ..out_h5 import is_linak_out_hdf5, read_out_h5_trajectory

    if is_linak_out_hdf5(trajectory_path):
        frames = read_out_h5_trajectory(trajectory_path)
    elif is_linak_trajectory_hdf5(trajectory_path):
        frames = _read_trajectory_hdf5(trajectory_path)
    else:
        suffix = trajectory_path.suffix.lower()
        if suffix == ".lmp":
            frames = _read_lammps_input_trajectory(trajectory_path)
        elif suffix == ".dump":
            frames = _read_lammps_dump_frames(trajectory_path)
        else:
            frames = _read_frames(trajectory_path)

    if not frames:
        raise ValueError(f"No frames were read from: {trajectory_path}")

    LOGGER.info("Loaded %d frame(s) from '%s'.", len(frames), _display)
    if frames:
        LOGGER.debug("Atoms per frame (frame 0): %d", len(frames[0]))

    return frames


def write_trajectory(
    frames: list[Atoms],
    path: str | Path,
    *,
    source_path: str | Path | None = None,
    source_format: str | None = None,
    metadata: TrajectoryStoredMetadata | None = None,
) -> Path:
    """Write trajectory frames to disk and return the written path."""
    if not frames:
        raise ValueError("At least one trajectory frame is required for writing.")

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".h5", ".hdf5"}:
        written = _write_trajectory_hdf5(
            frames,
            output_path,
            source_path=source_path,
            source_format=source_format,
            metadata=metadata,
        )
        LOGGER.info("Wrote %d frame(s) to '%s'.", len(frames), output_path)
        return written

    with ProgressBar(desc="Writing trajectory", total=1, unit="step") as progress:
        try:
            write(str(output_path), frames)
        except UnknownFileTypeError as exc:
            raise ValueError(
                f"Unsupported output trajectory format for '{output_path}'. "
                "Use a writable extension such as .xyz or .traj.h5."
            ) from exc
        progress.update()
    LOGGER.info("Wrote %d frame(s) to '%s'.", len(frames), output_path)
    return output_path
