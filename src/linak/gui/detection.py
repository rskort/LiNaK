"""Lightweight ProjectItem detection and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .model import ProjectItem, ProjectItemOrigin, ValidationResult, WorkspaceIndex


def _size_label(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0


def _stat_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "size_label": _size_label(int(stat.st_size)),
        "modified_time": float(stat.st_mtime),
        "mtime": float(stat.st_mtime),
    }


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _strip_known_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".out.hdf5", ".out.h5", ".traj.hdf5", ".traj.h5", ".cube.hdf5", ".cube.h5", ".hdf5", ".h5"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem or name


def _hdf5_metadata(path: Path) -> tuple[str, dict[str, Any], ValidationResult] | None:
    try:
        import h5py
    except ModuleNotFoundError:
        return None

    try:
        with h5py.File(path, "r") as handle:
            attrs = {str(key): _decode_attr(value) for key, value in handle.attrs.items()}
            metadata: dict[str, Any] = {
                "hdf5_attrs": {
                    key: value
                    for key, value in attrs.items()
                    if key
                    in {
                        "linak_format",
                        "linak_format_version",
                        "kind",
                        "analysis",
                        "linak_version",
                        "created_utc",
                    }
                },
            }
            metadata_json = attrs.get("metadata_json")
            if metadata_json:
                try:
                    decoded = json.loads(str(metadata_json))
                    if isinstance(decoded, dict):
                        metadata["profile_metadata"] = decoded
                except json.JSONDecodeError:
                    metadata["metadata_warning"] = "metadata_json could not be decoded"

            linak_format = str(attrs.get("linak_format", "")).strip()
            kind = str(attrs.get("kind", "")).strip()
            analysis = str(attrs.get("analysis", "")).strip()
            if linak_format == "linak-out-hdf5":
                try:
                    from ..out_h5 import inspect_out_h5

                    summary = inspect_out_h5(path)
                    metadata.update(
                        {
                            "schema_version": summary.schema_version,
                            "source_directory": summary.source_directory,
                            "frame_count": summary.frame_count,
                            "atom_count": summary.atom_count,
                            "cube_count": summary.cube_count,
                            "cp2k_output_count": summary.cp2k_output_count,
                            "singlepoint_sections": list(summary.singlepoint_sections),
                            "species": list(summary.species),
                            "warnings": list(summary.warnings),
                            "trajectory_present": summary.trajectory_present,
                            "cell_angstrom": list(summary.cell_angstrom)
                            if summary.cell_angstrom is not None
                            else None,
                            "timestep_fs": summary.timestep_fs,
                            "trajectory_source_path": summary.trajectory_source_path,
                            "trajectory_source_format": summary.trajectory_source_format,
                            "cell_matrix_angstrom": [
                                list(row) for row in summary.cell_matrix_angstrom
                            ],
                            "pbc": list(summary.pbc)
                            if summary.pbc is not None
                            else None,
                            "timestep_candidates_fs": list(summary.timestep_candidates_fs),
                            "frame_range": list(summary.frame_range)
                            if summary.frame_range is not None
                            else None,
                            "cube_kinds": list(summary.cube_kinds),
                            "cube_source_names": list(summary.cube_source_names),
                            "cp2k_table_counts": dict(summary.cp2k_table_counts),
                            "provenance_messages": list(summary.provenance_messages),
                            "discovery_summary": dict(summary.discovery_summary),
                            "parser_coverage": dict(summary.parser_coverage),
                        }
                    )
                except Exception as exc:
                    metadata["metadata_warning"] = str(exc)
                return "out_hdf5", metadata, ValidationResult("valid", "LiNaK output container")

            if linak_format == "linak-trajectory-hdf5" and kind == "trajectory":
                frames = handle.get("frames")
                if frames is not None and hasattr(frames, "shape"):
                    metadata["frames_shape"] = tuple(int(v) for v in frames.shape)
                    if len(frames.shape) >= 1:
                        metadata["frame_count"] = int(frames.shape[0])
                return "trajectory_hdf5", metadata, ValidationResult("valid", "LiNaK trajectory HDF5")

            if linak_format == "linak-hdf5" and analysis:
                profile_count = 1
                profiles = handle.get("profiles")
                if profiles is not None and hasattr(profiles, "items"):
                    profile_count = len([node for node in profiles.values() if hasattr(node, "items")])
                metadata["analysis"] = analysis
                metadata["profile_count"] = int(profile_count)
                item_type = "cube_hdf5" if analysis == "cube" else "analysis_hdf5"
                return item_type, metadata, ValidationResult("valid", f"LiNaK {analysis} HDF5")

            if analysis:
                metadata["analysis"] = analysis
                return "analysis_hdf5", metadata, ValidationResult(
                    "warning",
                    "HDF5 declares an analysis but lacks the current LiNaK format marker",
                )

            return "table_hdf5", metadata, ValidationResult(
                "warning",
                "HDF5 file is readable but is not a recognized LiNaK analysis output",
            )
    except OSError as exc:
        return "unsupported", {}, ValidationResult("invalid", f"Unreadable HDF5 file: {exc}")


def detect_project_item(path: str | Path, *, origin: ProjectItemOrigin) -> ProjectItem:
    """Create a ProjectItem using lightweight, format-aware validation."""

    resolved = Path(path).expanduser().resolve()
    metadata: dict[str, Any] = {}
    if not resolved.exists():
        return ProjectItem(
            path=resolved,
            item_type="missing",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("invalid", "File does not exist"),
        )
    if resolved.is_dir():
        metadata.update(_stat_metadata(resolved))
        return ProjectItem(
            path=resolved,
            item_type="simulation_directory",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("valid", "Simulation output directory"),
        )
    if not resolved.is_file():
        return ProjectItem(
            path=resolved,
            item_type="unsupported",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("invalid", "Path is not a file"),
        )
    if not os.access(resolved, os.R_OK):
        return ProjectItem(
            path=resolved,
            item_type="unsupported",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("invalid", "File is not readable"),
        )

    metadata.update(_stat_metadata(resolved))
    lower_name = resolved.name.lower()
    if lower_name.endswith((".h5", ".hdf5")):
        detected = _hdf5_metadata(resolved)
        if detected is not None:
            item_type, hdf5_metadata, validation = detected
            metadata.update(hdf5_metadata)
            return ProjectItem(
                path=resolved,
                item_type=item_type,
                origin=origin,
                metadata=metadata,
                validation=validation,
            )

    if lower_name.endswith((".cube",)):
        return ProjectItem(
            path=resolved,
            item_type="cube_file",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("valid", "Cube file"),
        )

    if lower_name.endswith((".temp", ".tregion")):
        metadata["stem"] = _strip_known_suffix(resolved.name)
        metadata["temperature_source_type"] = "tregion" if lower_name.endswith(".tregion") else "temp"
        return ProjectItem(
            path=resolved,
            item_type="temperature_file",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("valid", "Temperature source"),
        )

    if lower_name.endswith((".xyz", ".extxyz", ".dump", ".lmp")):
        metadata["stem"] = _strip_known_suffix(resolved.name)
        if lower_name.endswith((".xyz", ".extxyz")) and "-vel-" in lower_name:
            metadata["trajectory_role"] = "velocity"
        return ProjectItem(
            path=resolved,
            item_type="raw_trajectory",
            origin=origin,
            metadata=metadata,
            validation=ValidationResult("valid", "Trajectory source"),
        )

    return ProjectItem(
        path=resolved,
        item_type="unsupported",
        origin=origin,
        metadata=metadata,
        validation=ValidationResult("invalid", "Unsupported LiNaK workspace file type"),
    )


def discover_generated_items(project_dir: str | Path) -> list[ProjectItem]:
    """Scan a project directory for generated LiNaK-readable outputs."""

    root = Path(project_dir).expanduser().resolve()
    if not root.exists():
        return []
    candidates: list[ProjectItem] = []
    for path in root.rglob("*"):
        if path.name == ".linak_project.json" or not path.is_file():
            continue
        lower_name = path.name.lower()
        if not lower_name.endswith((".h5", ".hdf5", ".out.h5", ".out.hdf5", ".traj.h5", ".traj.hdf5", ".cube.h5", ".cube.hdf5", ".xyz", ".cube", ".temp", ".tregion")):
            continue
        item = detect_project_item(path, origin="generated")
        if (
            item.validation.state != "invalid"
            and item.item_type not in {"unsupported", "table_hdf5"}
        ):
            candidates.append(item)
    return candidates


def discover_generated_items_cached(
    project_dir: str | Path,
    *,
    index: WorkspaceIndex,
) -> list[ProjectItem]:
    """Scan project outputs using cached detection for unchanged files."""

    root = Path(project_dir).expanduser().resolve()
    if not root.exists():
        return []
    candidates: list[ProjectItem] = []
    for path in root.rglob("*"):
        if path.name == ".linak_project.json" or not path.is_file():
            continue
        lower_name = path.name.lower()
        if not lower_name.endswith((
            ".h5",
            ".hdf5",
            ".out.h5",
            ".out.hdf5",
            ".traj.h5",
            ".traj.hdf5",
            ".cube.h5",
            ".cube.hdf5",
            ".xyz",
            ".cube",
        )):
            continue
        item = index.detect_or_reuse(path, origin="generated", detector=detect_project_item)
        if (
            item.validation.state != "invalid"
            and item.item_type not in {"unsupported", "table_hdf5"}
        ):
            candidates.append(item)
    return candidates
