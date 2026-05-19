"""GUI-facing defaults, summaries, and action readiness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..out_h5 import OutH5Summary, inspect_out_h5
from .actions import Action, validate_action_settings
from .model import ProjectItem


@dataclass(frozen=True)
class OutH5GuiSummary:
    """Small GUI-friendly projection of `.out.h5` metadata."""

    source_directory: str = ""
    species: tuple[str, ...] = ()
    cell_angstrom: tuple[float, float, float] | None = None
    timestep_fs: float | None = None
    frame_count: int | None = None
    atom_count: int | None = None
    cube_count: int = 0
    cp2k_output_count: int = 0
    singlepoint_sections: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    trajectory_present: bool = False
    trajectory_source_path: str = ""
    trajectory_source_format: str = ""
    cell_matrix_angstrom: tuple[tuple[float, float, float], ...] = ()
    pbc: tuple[bool, bool, bool] | None = None
    timestep_candidates_fs: tuple[float, ...] = ()
    frame_range: tuple[int, int] | None = None
    cube_kinds: tuple[str, ...] = ()
    cube_source_names: tuple[str, ...] = ()
    cp2k_table_counts: dict[str, int] = field(default_factory=dict)
    provenance_messages: tuple[str, ...] = ()
    discovery_summary: dict[str, Any] = field(default_factory=dict)
    parser_coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def has_cubes(self) -> bool:
        return self.cube_count > 0

    @property
    def has_singlepoint(self) -> bool:
        return self.cp2k_output_count > 0 or bool(self.singlepoint_sections)

    def detail_lines(self) -> tuple[str, ...]:
        lines: list[str] = []
        if self.source_directory:
            lines.append(f"Source: {self.source_directory}")
        if self.trajectory_source_path:
            lines.append(f"Trajectory: {Path(self.trajectory_source_path).name}")
        if self.frame_count is not None:
            lines.append(f"Frames: {self.frame_count}")
        if self.frame_range is not None:
            lines.append(f"Frame range: {self.frame_range[0]}-{self.frame_range[1]}")
        if self.atom_count is not None:
            lines.append(f"Atoms: {self.atom_count}")
        if self.species:
            lines.append("Species: " + ", ".join(self.species))
        if self.cell_angstrom is not None:
            lines.append(
                "Cell A B C: "
                + " ".join(f"{value:.6g}" for value in self.cell_angstrom)
                + " A"
            )
        if self.cell_matrix_angstrom:
            matrix = "; ".join(
                " ".join(f"{value:.6g}" for value in row)
                for row in self.cell_matrix_angstrom
            )
            lines.append(f"Cell matrix: {matrix} A")
        if self.pbc is not None:
            lines.append("PBC: " + " ".join("on" if value else "off" for value in self.pbc))
        if self.timestep_fs is not None:
            lines.append(f"Timestep: {self.timestep_fs:g} fs")
        if len(self.timestep_candidates_fs) > 1:
            lines.append(
                "Timestep candidates: "
                + ", ".join(f"{value:g} fs" for value in self.timestep_candidates_fs[:4])
            )
        lines.append(f"Cubes: {self.cube_count}")
        if self.cube_source_names:
            lines.append("Cube sources: " + ", ".join(self.cube_source_names[:4]))
        if self.cube_kinds:
            lines.append("Cube kinds: " + ", ".join(self.cube_kinds))
        if self.has_singlepoint:
            sections = ", ".join(self.singlepoint_sections) or "metadata"
            lines.append(f"CP2K outputs: {self.cp2k_output_count} ({sections})")
        if self.cp2k_table_counts:
            lines.append(
                "CP2K rows: "
                + ", ".join(f"{key}={value}" for key, value in sorted(self.cp2k_table_counts.items())[:4])
            )
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return tuple(lines)


@dataclass(frozen=True)
class ActionReadiness:
    """Whether a GUI action can run and why."""

    available: bool
    reason: str = "Ready"


def out_h5_gui_summary_from_summary(summary: OutH5Summary) -> OutH5GuiSummary:
    return OutH5GuiSummary(
        source_directory=summary.source_directory,
        species=tuple(summary.species),
        cell_angstrom=summary.cell_angstrom,
        timestep_fs=summary.timestep_fs,
        frame_count=summary.frame_count,
        atom_count=summary.atom_count,
        cube_count=summary.cube_count,
        cp2k_output_count=summary.cp2k_output_count,
        singlepoint_sections=tuple(summary.singlepoint_sections),
        warnings=tuple(summary.warnings),
        trajectory_present=summary.trajectory_present,
        trajectory_source_path=summary.trajectory_source_path,
        trajectory_source_format=summary.trajectory_source_format,
        cell_matrix_angstrom=summary.cell_matrix_angstrom,
        pbc=summary.pbc,
        timestep_candidates_fs=summary.timestep_candidates_fs,
        frame_range=summary.frame_range,
        cube_kinds=summary.cube_kinds,
        cube_source_names=summary.cube_source_names,
        cp2k_table_counts=dict(summary.cp2k_table_counts),
        provenance_messages=summary.provenance_messages,
        discovery_summary=dict(summary.discovery_summary),
        parser_coverage=dict(summary.parser_coverage),
    )


def out_h5_gui_summary_for_item(item: ProjectItem) -> OutH5GuiSummary | None:
    if item.item_type != "out_hdf5":
        return None
    metadata = item.metadata
    if metadata:
        return OutH5GuiSummary(
            source_directory=str(metadata.get("source_directory") or ""),
            species=tuple(str(value) for value in metadata.get("species", ()) or ()),
            cell_angstrom=_coerce_cell(metadata.get("cell_angstrom")),
            timestep_fs=_coerce_float(metadata.get("timestep_fs")),
            frame_count=_coerce_int(metadata.get("frame_count")),
            atom_count=_coerce_int(metadata.get("atom_count")),
            cube_count=_coerce_int(metadata.get("cube_count")) or 0,
            cp2k_output_count=_coerce_int(metadata.get("cp2k_output_count")) or 0,
            singlepoint_sections=tuple(
                str(value) for value in metadata.get("singlepoint_sections", ()) or ()
            ),
            warnings=tuple(str(value) for value in metadata.get("warnings", ()) or ()),
            trajectory_present=bool(metadata.get("trajectory_present", False)),
            trajectory_source_path=str(metadata.get("trajectory_source_path") or ""),
            trajectory_source_format=str(metadata.get("trajectory_source_format") or ""),
            cell_matrix_angstrom=_coerce_cell_matrix(metadata.get("cell_matrix_angstrom")),
            pbc=_coerce_pbc(metadata.get("pbc")),
            timestep_candidates_fs=tuple(
                float(value) for value in metadata.get("timestep_candidates_fs", ()) or ()
            ),
            frame_range=_coerce_frame_range(metadata.get("frame_range")),
            cube_kinds=tuple(str(value) for value in metadata.get("cube_kinds", ()) or ()),
            cube_source_names=tuple(
                str(value) for value in metadata.get("cube_source_names", ()) or ()
            ),
            cp2k_table_counts={
                str(key): int(value)
                for key, value in dict(metadata.get("cp2k_table_counts") or {}).items()
            },
            provenance_messages=tuple(
                str(value) for value in metadata.get("provenance_messages", ()) or ()
            ),
            discovery_summary=dict(metadata.get("discovery_summary") or {}),
            parser_coverage=dict(metadata.get("parser_coverage") or {}),
        )
    try:
        return out_h5_gui_summary_from_summary(inspect_out_h5(item.path))
    except Exception:
        return None


def default_settings_for_action(
    action: Action,
    item: ProjectItem,
    project_dir: str | Path,
) -> dict[str, Any]:
    """Build editable GUI defaults from action schema plus item metadata."""

    settings = {field.key: field.default for field in action.settings_schema(item)}
    summary = out_h5_gui_summary_for_item(item)
    species = tuple(summary.species if summary is not None else item.metadata.get("species", ()) or ())
    if item.item_type == "simulation_directory" and action.action_id == "pack_out_h5":
        settings["output_name"] = _unique_name(Path(project_dir) / f"{item.path.name}.out.h5").name
    if species:
        if "species" in settings:
            settings["species"] = "all"
        if "species_a" in settings:
            settings["species_a"] = species[0]
        if "species_b" in settings:
            settings["species_b"] = species[1] if len(species) > 1 else species[0]
        if "surface_elements" in settings and not settings.get("surface_elements"):
            settings["surface_elements"] = " ".join(species)
    if summary is not None:
        if summary.cell_angstrom is not None and "cell" in settings:
            settings["cell"] = " ".join(f"{value:.8g}" for value in summary.cell_angstrom)
        timestep = summary.timestep_fs
        if timestep is None and summary.timestep_candidates_fs:
            timestep = summary.timestep_candidates_fs[0]
        if timestep is not None and "timestep_fs" in settings:
            settings["timestep_fs"] = timestep
        if summary.cube_kinds and "cube_kind" in settings:
            settings["cube_kind"] = summary.cube_kinds[0]
    return settings


def readiness_for_action(action: Action, item: ProjectItem) -> ActionReadiness:
    if not action.supports(item):
        if item.validation.state == "invalid":
            return ActionReadiness(False, item.validation.message or "Item is invalid")
        return ActionReadiness(False, "Action does not support this item type")
    summary = out_h5_gui_summary_for_item(item)
    if item.item_type == "out_hdf5" and summary is not None:
        trajectory_actions = {
            "density",
            "msd",
            "rdf",
            "position",
            "coordination",
            "orientation",
            "pbc",
            "export_out_trajectory",
        }
        if action.action_id in trajectory_actions and not summary.trajectory_present:
            return ActionReadiness(False, "Missing trajectory data in .out.h5")
        if action.action_id in {"potential", "export_out_cube"} and not summary.has_cubes:
            return ActionReadiness(False, "Missing cube data in .out.h5")
    return ActionReadiness(True)


def defaults_validate(action: Action, item: ProjectItem, project_dir: str | Path) -> bool:
    try:
        validate_action_settings(action, item, default_settings_for_action(action, item, project_dir))
    except Exception:
        return False
    return True


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_cell(value: Any) -> tuple[float, float, float] | None:
    if value in (None, ""):
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3:
        return None
    return values  # type: ignore[return-value]


def _coerce_cell_matrix(value: Any) -> tuple[tuple[float, float, float], ...]:
    if not value:
        return ()
    rows: list[tuple[float, float, float]] = []
    try:
        for row in value:
            converted = tuple(float(item) for item in row)
            if len(converted) == 3:
                rows.append(converted)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ()
    return tuple(rows)


def _coerce_frame_range(value: Any) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        start, finish = value
        return int(start), int(finish)
    except (TypeError, ValueError):
        return None


def _coerce_pbc(value: Any) -> tuple[bool, bool, bool] | None:
    if value in (None, ""):
        return None
    try:
        converted = tuple(bool(item) for item in value)
    except TypeError:
        return None
    if len(converted) != 3:
        return None
    return converted  # type: ignore[return-value]


def _unique_name(path: Path) -> Path:
    if not path.exists():
        return path
    lower = path.name.lower()
    for suffix in (".out.hdf5", ".out.h5", ".traj.hdf5", ".traj.h5", ".cube.hdf5", ".cube.h5"):
        if lower.endswith(suffix):
            base = path.name[: -len(suffix)]
            counter = 1
            while True:
                candidate = path.with_name(f"{base}_{counter}{path.name[-len(suffix):]}")
                if not candidate.exists():
                    return candidate
                counter += 1
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
