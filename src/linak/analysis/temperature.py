"""Temperature analysis from CP2K temperature tables and velocity trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from ..plot.data_contract import PlotDataContract, PlotViewMapping
from ..plot.mappings.temperature_mapping import resolve_temperature_plot_mapping
from ..plot.plotting import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    plot_line_series,
    plot_multi_line_series,
    resolve_explicit_plot_text,
    resolve_series_labels,
    resolve_single_series_options,
)
from .common import read_profile_payloads, read_profile_payloads_by_index, use_multi_series_plot, write_profile_collection
from .schema import build_profile_metadata, default_plot_labels

LOGGER = logging.getLogger(__name__)

ATOMIC_VELOCITY_TO_M_PER_S = 2.18769126364e6
ANGSTROM_PER_FS_TO_M_PER_S = 1.0e5
AMU_TO_KG = 1.66053906660e-27
BOLTZMANN_J_PER_K = 1.380649e-23

_INT_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:\.\.\s*(\d+))?\s*$")
_FRAME_COMMENT_RE = re.compile(
    r"\bi\s*=\s*([+-]?\d+).*?\btime\s*=\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemperatureRegion:
    """Resolved CP2K thermal-region selection."""

    region_index: int
    region_name: str
    atom_indices: tuple[int, ...]
    cp2k_list_expression: str
    target_temperature_K: float | None = None
    region_elements: tuple[str, ...] = ()
    region_formula: str | None = None
    label_resolution_status: str = "resolved"


@dataclass(frozen=True)
class TemperatureMetadata:
    """Best-effort labels discovered from CP2K input or sibling trajectories."""

    elements: tuple[str, ...] = ()
    regions: tuple[TemperatureRegion, ...] = ()
    atom_symbols: tuple[str, ...] = ()
    input_path: Path | None = None
    label_resolution_status: str = "unresolved"


@dataclass(frozen=True)
class TemperatureProfile:
    """One temperature time series for an element or region."""

    default_label: str
    selection_kind: str
    frame_index: np.ndarray
    step: np.ndarray
    time_fs: np.ndarray
    time_ps: np.ndarray
    temperature_K: np.ndarray
    n_frames: int
    source_type: str
    element: str | None = None
    region_index: int | None = None
    region_name: str | None = None
    cp2k_list_expression: str | None = None
    region_elements: tuple[str, ...] = ()
    region_formula: str | None = None
    atom_count: int | None = None
    atom_indices: np.ndarray | None = None
    atom_index_base: int | None = None
    target_temperature_K: float | None = None
    velocity_unit: str | None = None
    dof_mode: str | None = None
    label_resolution_status: str = "resolved"


def _strip_inline_comment(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _parse_cp2k_list_expression(expression: str) -> tuple[int, ...]:
    indices: list[int] = []
    for token in re.split(r"[\s,]+", str(expression).strip()):
        if not token:
            continue
        match = _INT_RANGE_RE.match(token)
        if match is None:
            LOGGER.warning("Ignoring unsupported CP2K atom-list token '%s'.", token)
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start <= 0 or end <= 0:
            continue
        if end < start:
            start, end = end, start
        indices.extend(range(start - 1, end))
    return tuple(sorted(set(indices)))


def _format_cp2k_range(indices: Sequence[int]) -> str:
    ordered = sorted({int(index) + 1 for index in indices if int(index) >= 0})
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}..{prev}")
    return " ".join(ranges)


def _extract_block(lines: list[str], start_index: int) -> tuple[list[str], int]:
    start_line = _strip_inline_comment(lines[start_index])
    match = re.match(r"&([A-Za-z0-9_]+)", start_line)
    if match is None:
        return [], start_index
    block_name = match.group(1).upper()
    depth = 1
    block_lines: list[str] = []
    index = start_index + 1
    while index < len(lines):
        stripped = _strip_inline_comment(lines[index])
        upper = stripped.upper()
        if re.match(r"&END(?:\s+" + re.escape(block_name) + r")?\b", upper):
            depth -= 1
            if depth == 0:
                return block_lines, index
        elif re.match(r"&" + re.escape(block_name) + r"\b", upper):
            depth += 1
        block_lines.append(lines[index])
        index += 1
    return block_lines, len(lines) - 1


def _parse_cp2k_input(path: str | Path) -> tuple[list[str], list[TemperatureRegion], tuple[int, ...]]:
    input_path = Path(path).expanduser().resolve()
    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    kinds: list[str] = []
    fixed_indices: set[int] = set()
    define_regions: list[TemperatureRegion] = []

    index = 0
    while index < len(lines):
        stripped = _strip_inline_comment(lines[index])
        upper = stripped.upper()
        kind_match = re.match(r"&KIND\s+(\S+)", stripped, re.IGNORECASE)
        if kind_match:
            token = kind_match.group(1).strip()
            if token and token not in kinds:
                kinds.append(token)
        if re.match(r"&FIXED_ATOMS\b", upper):
            block, end_index = _extract_block(lines, index)
            for block_line in block:
                clean = _strip_inline_comment(block_line)
                if clean.upper().startswith("LIST "):
                    fixed_indices.update(_parse_cp2k_list_expression(clean[5:].strip()))
            index = end_index
        elif re.match(r"&DEFINE_REGION\b", upper):
            block, end_index = _extract_block(lines, index)
            list_expression = ""
            target_temperature: float | None = None
            for block_line in block:
                clean = _strip_inline_comment(block_line)
                upper_clean = clean.upper()
                if upper_clean.startswith("LIST "):
                    list_expression = clean[5:].strip()
                elif upper_clean.startswith("TEMPERATURE "):
                    try:
                        target_temperature = float(clean.split(None, 1)[1])
                    except (IndexError, ValueError):
                        target_temperature = None
            atom_indices = _parse_cp2k_list_expression(list_expression)
            region_index = len(define_regions) + 1
            define_regions.append(
                TemperatureRegion(
                    region_index=region_index,
                    region_name=f"Region {region_index}",
                    atom_indices=atom_indices,
                    cp2k_list_expression=list_expression,
                    target_temperature_K=target_temperature,
                    label_resolution_status="resolved" if atom_indices else "generic",
                )
            )
            index = end_index
        index += 1

    return kinds, define_regions, tuple(sorted(fixed_indices))


def _first_frame_symbols_from_xyz(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        try:
            atom_count = int(first.strip())
        except ValueError:
            return []
        handle.readline()
        symbols: list[str] = []
        for _ in range(atom_count):
            parts = handle.readline().split()
            if parts:
                symbols.append(parts[0])
        return symbols


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return tuple(ordered)


def _region_composition(
    atom_indices: Sequence[int],
    atom_symbols: Sequence[str],
) -> tuple[tuple[str, ...], str | None]:
    if not atom_indices or not atom_symbols:
        return (), None
    counts: dict[str, int] = {}
    order: list[str] = []
    symbol_count = len(atom_symbols)
    for raw_index in atom_indices:
        index = int(raw_index)
        if index < 0 or index >= symbol_count:
            continue
        symbol = str(atom_symbols[index]).strip()
        if not symbol:
            continue
        if symbol not in counts:
            counts[symbol] = 0
            order.append(symbol)
        counts[symbol] += 1
    if not order:
        return (), None
    formula = " ".join(f"{symbol}{counts[symbol]}" for symbol in order)
    return tuple(order), formula


def _region_label(region: TemperatureRegion) -> str:
    if region.region_formula:
        return f"{region.region_name} [{region.region_formula}]"
    return region.region_name


def _sibling_input_path(source: Path) -> Path | None:
    candidate = source.with_name("input.inp")
    return candidate if candidate.exists() else None


def _sibling_xyz_atom_symbols(source: Path) -> tuple[str, ...]:
    candidates = list(source.parent.glob("*-vel-*.xyz")) + list(source.parent.glob("*-pos-*.xyz"))
    for candidate in candidates:
        try:
            symbols = _first_frame_symbols_from_xyz(candidate)
        except OSError:
            continue
        if symbols:
            return tuple(symbols)
    return ()


def _with_region_composition(
    regions: Sequence[TemperatureRegion],
    atom_symbols: Sequence[str],
) -> tuple[TemperatureRegion, ...]:
    resolved: list[TemperatureRegion] = []
    for region in regions:
        region_elements, region_formula = _region_composition(region.atom_indices, atom_symbols)
        resolved.append(
            TemperatureRegion(
                region_index=region.region_index,
                region_name=region.region_name,
                atom_indices=region.atom_indices,
                cp2k_list_expression=region.cp2k_list_expression,
                target_temperature_K=region.target_temperature_K,
                region_elements=region_elements,
                region_formula=region_formula,
                label_resolution_status=region.label_resolution_status,
            )
        )
    return tuple(resolved)


def discover_temperature_metadata(
    source: str | Path,
    *,
    input_path: str | Path | None = None,
) -> TemperatureMetadata:
    """Discover element and region labels for a temperature source."""

    source_path = Path(source).expanduser().resolve()
    resolved_input = Path(input_path).expanduser().resolve() if input_path else _sibling_input_path(source_path)
    input_elements: tuple[str, ...] = ()
    regions: tuple[TemperatureRegion, ...] = ()
    fixed_indices: tuple[int, ...] = ()
    status = "generic"
    if resolved_input is not None and resolved_input.exists():
        try:
            kinds, parsed_regions, fixed_indices = _parse_cp2k_input(resolved_input)
            input_elements = tuple(kinds)
            regions = tuple(parsed_regions)
            status = "resolved"
        except OSError as exc:
            LOGGER.warning("Could not read CP2K input '%s': %s", resolved_input, exc)
            resolved_input = None

    atom_symbols = _sibling_xyz_atom_symbols(source_path)
    if source_path.suffix.lower() == ".xyz":
        try:
            atom_symbols = tuple(_first_frame_symbols_from_xyz(source_path))
        except OSError:
            atom_symbols = ()
    elements = _ordered_unique(atom_symbols) or input_elements

    if fixed_indices:
        defined = {index for region in regions for index in region.atom_indices}
        default_indices = tuple(index for index in fixed_indices if index not in defined)
        if default_indices:
            default_region = TemperatureRegion(
                region_index=0,
                region_name="Unassigned",
                atom_indices=default_indices,
                cp2k_list_expression=_format_cp2k_range(default_indices),
                target_temperature_K=0.0,
                label_resolution_status="resolved",
            )
            regions = (default_region, *regions)
    if atom_symbols and regions:
        regions = _with_region_composition(regions, atom_symbols)

    if not elements and not regions:
        status = "generic"
    return TemperatureMetadata(
        elements=elements,
        regions=regions,
        atom_symbols=atom_symbols,
        input_path=resolved_input,
        label_resolution_status=status,
    )


def _read_numeric_table(path: str | Path, *, skip_comments: bool) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or (skip_comments and stripped.startswith("#")):
                continue
            try:
                rows.append([float(part) for part in stripped.split()])
            except ValueError:
                LOGGER.warning("Skipping non-numeric temperature table row: %s", stripped)
    if not rows:
        raise ValueError(f"Temperature table '{path}' contains no numeric rows.")
    width = len(rows[0])
    if width < 3:
        raise ValueError(f"Temperature table '{path}' must have at least 3 columns.")
    if any(len(row) != width for row in rows):
        raise ValueError(f"Temperature table '{path}' has inconsistent column counts.")
    return np.asarray(rows, dtype=float)


def _table_time_arrays(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_index = np.arange(table.shape[0], dtype=int)
    step = np.asarray(table[:, 0], dtype=np.int64)
    time_fs = np.asarray(table[:, 1], dtype=float)
    return frame_index, step, time_fs, time_fs / 1000.0


def _profile_from_table_column(
    *,
    table: np.ndarray,
    column_index: int,
    label: str,
    selection_kind: str,
    source_type: str,
    element: str | None = None,
    region: TemperatureRegion | None = None,
    status: str,
) -> TemperatureProfile:
    frame_index, step, time_fs, time_ps = _table_time_arrays(table)
    atom_indices = (
        None if region is None or not region.atom_indices else np.asarray(region.atom_indices, dtype=int)
    )
    return TemperatureProfile(
        default_label=label,
        selection_kind=selection_kind,
        frame_index=frame_index,
        step=step,
        time_fs=time_fs,
        time_ps=time_ps,
        temperature_K=np.asarray(table[:, column_index], dtype=float),
        n_frames=table.shape[0],
        source_type=source_type,
        element=element,
        region_index=None if region is None else region.region_index,
        region_name=None if region is None else region.region_name,
        cp2k_list_expression=None if region is None else region.cp2k_list_expression,
        region_elements=() if region is None else region.region_elements,
        region_formula=None if region is None else region.region_formula,
        atom_count=None if region is None else len(region.atom_indices),
        atom_indices=atom_indices,
        atom_index_base=0 if atom_indices is not None else None,
        target_temperature_K=None if region is None else region.target_temperature_K,
        label_resolution_status=status,
    )


def compute_temperature_from_temp(
    source: str | Path,
    *,
    metadata: TemperatureMetadata | None = None,
) -> list[TemperatureProfile]:
    """Load CP2K ``.temp`` data as element/kind temperature profiles."""

    source_path = Path(source).expanduser().resolve()
    table = _read_numeric_table(source_path, skip_comments=True)
    metadata = metadata or discover_temperature_metadata(source_path)
    value_count = table.shape[1] - 2
    labels = list(metadata.elements)
    if len(labels) < value_count:
        LOGGER.warning("Could not resolve all .temp labels; using generic kind labels.")
        labels.extend(f"kind {index + 1}" for index in range(len(labels), value_count))
    return [
        _profile_from_table_column(
            table=table,
            column_index=2 + index,
            label=labels[index],
            selection_kind="element",
            source_type="temp",
            element=labels[index] if not labels[index].startswith("kind ") else None,
            status=metadata.label_resolution_status if index < len(metadata.elements) else "generic",
        )
        for index in range(value_count)
    ]


def compute_temperature_from_tregion(
    source: str | Path,
    *,
    metadata: TemperatureMetadata | None = None,
) -> list[TemperatureProfile]:
    """Load CP2K ``.tregion`` data as thermal-region temperature profiles."""

    source_path = Path(source).expanduser().resolve()
    table = _read_numeric_table(source_path, skip_comments=True)
    metadata = metadata or discover_temperature_metadata(source_path)
    value_count = table.shape[1] - 2
    regions = list(metadata.regions)
    if len(regions) < value_count:
        LOGGER.warning("Could not resolve all .tregion labels; using generic region labels.")
        for index in range(len(regions), value_count):
            regions.append(
                TemperatureRegion(
                    region_index=index,
                    region_name=f"Region {index}",
                    atom_indices=(),
                    cp2k_list_expression="",
                    label_resolution_status="generic",
                )
            )
    return [
        _profile_from_table_column(
            table=table,
            column_index=2 + index,
            label=_region_label(regions[index]),
            selection_kind="region",
            source_type="tregion",
            region=regions[index],
            status=regions[index].label_resolution_status,
        )
        for index in range(value_count)
    ]


def _parse_velocity_frame_comment(comment: str, frame_index: int) -> tuple[int, float]:
    match = _FRAME_COMMENT_RE.search(comment)
    if match is None:
        return frame_index, float(frame_index)
    return int(match.group(1)), float(match.group(2))


def _iter_velocity_xyz(path: Path) -> list[tuple[int, float, list[str], np.ndarray]]:
    frames: list[tuple[int, float, list[str], np.ndarray]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            first = handle.readline()
            if not first:
                break
            if not first.strip():
                continue
            try:
                atom_count = int(first.strip())
            except ValueError as exc:
                raise ValueError(f"Velocity XYZ '{path}' has an invalid atom-count line.") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"Velocity XYZ '{path}' ended before frame comment.")
            step, time_fs = _parse_velocity_frame_comment(comment, frame_index)
            symbols: list[str] = []
            velocities = np.zeros((atom_count, 3), dtype=float)
            for atom_index in range(atom_count):
                line = handle.readline()
                if not line:
                    raise ValueError(f"Velocity XYZ '{path}' ended inside frame {frame_index}.")
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Velocity XYZ '{path}' has an invalid atom row: {line!r}")
                symbols.append(parts[0])
                velocities[atom_index, :] = [float(parts[1]), float(parts[2]), float(parts[3])]
            frames.append((step, time_fs, symbols, velocities))
            frame_index += 1
    if not frames:
        raise ValueError(f"Velocity XYZ '{path}' contains no frames.")
    return frames


def _velocity_factor(unit: str) -> tuple[str, float]:
    normalized = str(unit or "auto").strip().lower()
    if normalized in {"auto", "atomic", "au", "a.u."}:
        return "atomic", ATOMIC_VELOCITY_TO_M_PER_S
    if normalized in {"angstrom/fs", "a/fs", "ang/fs"}:
        return "angstrom/fs", ANGSTROM_PER_FS_TO_M_PER_S
    raise ValueError("velocity_unit must be one of auto, atomic, or angstrom/fs.")


def _temperature_for_selection(
    *,
    symbols: Sequence[str],
    velocities: np.ndarray,
    indices: Sequence[int],
    velocity_factor: float,
    remove_com: bool,
) -> float:
    selected = np.asarray(indices, dtype=int)
    if selected.size == 0:
        return np.nan
    selected_velocities = np.asarray(velocities[selected], dtype=float) * velocity_factor
    masses_amu: list[float] = []
    for index in selected:
        symbol = str(symbols[int(index)])
        atomic_number = atomic_numbers.get(symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol '{symbol}' in velocity trajectory.")
        masses_amu.append(float(atomic_masses[atomic_number]))
    masses = np.asarray(masses_amu, dtype=float) * AMU_TO_KG
    dof = 3 * selected.size
    if remove_com:
        if selected.size <= 1:
            return np.nan
        v_com = np.sum(selected_velocities * masses[:, None], axis=0) / float(np.sum(masses))
        selected_velocities = selected_velocities - v_com
        dof -= 3
    if dof <= 0:
        return np.nan
    kinetic_j = 0.5 * float(np.sum(masses[:, None] * selected_velocities * selected_velocities))
    return 2.0 * kinetic_j / (float(dof) * BOLTZMANN_J_PER_K)


def compute_temperature_from_velocity_xyz(
    source: str | Path,
    *,
    metadata: TemperatureMetadata | None = None,
    group_by: str = "auto",
    velocity_unit: str = "auto",
    remove_com: bool = False,
) -> list[TemperatureProfile]:
    """Compute element and/or region temperatures from a CP2K velocity XYZ file."""

    source_path = Path(source).expanduser().resolve()
    frames = _iter_velocity_xyz(source_path)
    metadata = metadata or discover_temperature_metadata(source_path)
    normalized_group = str(group_by or "auto").strip().lower()
    if normalized_group == "auto":
        normalized_group = "both" if metadata.regions else "elements"
    if normalized_group not in {"elements", "regions", "both"}:
        raise ValueError("group_by must be one of auto, elements, regions, or both.")
    resolved_unit, factor = _velocity_factor(velocity_unit)

    first_symbols = frames[0][2]
    elements = list(metadata.elements or _ordered_unique(first_symbols))
    selections: list[tuple[str, str, str | None, TemperatureRegion | None, np.ndarray]] = []
    if normalized_group in {"elements", "both"}:
        for element in elements:
            indices = np.asarray(
                [index for index, symbol in enumerate(first_symbols) if symbol == element],
                dtype=int,
            )
            selections.append(("element", element, element, None, indices))
    if normalized_group in {"regions", "both"}:
        regions = list(metadata.regions)
        if not regions:
            LOGGER.warning("No region metadata resolved for velocity input; using elements only.")
        elif not any(region.region_formula for region in regions):
            regions = list(_with_region_composition(regions, first_symbols))
        for region in regions:
            selections.append(
                (
                    "region",
                    _region_label(region),
                    None,
                    region,
                    np.asarray(region.atom_indices, dtype=int),
                )
            )

    frame_index = np.arange(len(frames), dtype=int)
    step = np.asarray([frame[0] for frame in frames], dtype=np.int64)
    time_fs = np.asarray([frame[1] for frame in frames], dtype=float)
    profiles: list[TemperatureProfile] = []
    for selection_kind, label, element, region, indices in selections:
        values = np.zeros(len(frames), dtype=float)
        for index, (_step, _time_fs, symbols, velocities) in enumerate(frames):
            if len(symbols) != len(first_symbols):
                raise ValueError("Velocity temperature requires stable atom count across frames.")
            values[index] = _temperature_for_selection(
                symbols=symbols,
                velocities=velocities,
                indices=indices,
                velocity_factor=factor,
                remove_com=remove_com,
            )
        profiles.append(
            TemperatureProfile(
                default_label=label,
                selection_kind=selection_kind,
                frame_index=frame_index.copy(),
                step=step.copy(),
                time_fs=time_fs.copy(),
                time_ps=time_fs / 1000.0,
                temperature_K=values,
                n_frames=len(frames),
                source_type="velocity_xyz",
                element=element,
                region_index=None if region is None else region.region_index,
                region_name=None if region is None else region.region_name,
                cp2k_list_expression=None if region is None else region.cp2k_list_expression,
                region_elements=() if region is None else region.region_elements,
                region_formula=None if region is None else region.region_formula,
                atom_count=int(indices.size),
                atom_indices=indices.copy(),
                atom_index_base=0,
                target_temperature_K=None if region is None else region.target_temperature_K,
                velocity_unit=resolved_unit,
                dof_mode="3N-3COM" if remove_com else "3N",
                label_resolution_status=metadata.label_resolution_status
                if region is None
                else region.label_resolution_status,
            )
        )
    if not profiles:
        raise ValueError("No temperature profiles could be computed from velocity input.")
    return profiles


def compute_temperature_profiles(
    source: str | Path,
    *,
    input_path: str | Path | None = None,
    group_by: str = "auto",
    velocity_unit: str = "auto",
    remove_com: bool = False,
) -> list[TemperatureProfile]:
    """Compute or load temperature profiles from a supported source file."""

    source_path = Path(source).expanduser().resolve()
    metadata = discover_temperature_metadata(source_path, input_path=input_path)
    suffix = source_path.suffix.lower()
    name = source_path.name.lower()
    if suffix == ".temp":
        return compute_temperature_from_temp(source_path, metadata=metadata)
    if suffix == ".tregion":
        return compute_temperature_from_tregion(source_path, metadata=metadata)
    if suffix == ".xyz" and "-vel-" in name:
        return compute_temperature_from_velocity_xyz(
            source_path,
            metadata=metadata,
            group_by=group_by,
            velocity_unit=velocity_unit,
            remove_com=remove_com,
        )
    raise ValueError(
        f"Unsupported temperature source '{source_path.name}'. Use .temp, .tregion, or *-vel-*.xyz."
    )


def _temperature_profile_hdf5_payload(profile: TemperatureProfile) -> dict[str, Any]:
    metadata = build_profile_metadata(
        analysis="temperature",
        metadata={
            "source_type": profile.source_type,
            "selection_kind": profile.selection_kind,
            "default_label": profile.default_label,
            "element": profile.element,
            "region_index": profile.region_index,
            "region_name": profile.region_name,
            "cp2k_list_expression": profile.cp2k_list_expression,
            "region_elements": list(profile.region_elements),
            "region_formula": profile.region_formula,
            "atom_count": profile.atom_count,
            "atom_index_base": profile.atom_index_base,
            "target_temperature_K": profile.target_temperature_K,
            "velocity_unit": profile.velocity_unit,
            "dof_mode": profile.dof_mode,
            "label_resolution_status": profile.label_resolution_status,
            "n_frames": int(profile.n_frames),
        },
    )
    return {
        "datasets": {
            "frame_index": profile.frame_index,
            "step": profile.step,
            "time_fs": profile.time_fs,
            "time_ps": profile.time_ps,
            "temperature_K": profile.temperature_K,
            "atom_indices": profile.atom_indices,
        },
        "metadata": metadata,
    }


def save_temperature_profiles(
    profiles: list[TemperatureProfile],
    output: str | Path,
    *,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one or more temperature profiles to LiNaK HDF5 and return the path."""

    if not profiles:
        raise ValueError("At least one temperature profile is required.")
    output_path = write_profile_collection(
        output,
        analysis="temperature",
        profiles=[_temperature_profile_hdf5_payload(profile) for profile in profiles],
        metadata=dict(additional_metadata or {}),
    )
    LOGGER.info("Saved %d temperature profiles to '%s'.", len(profiles), output_path)
    return output_path


def _load_temperature_profiles_from_payloads(
    source_path: Path,
    payloads: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
) -> list[TemperatureProfile]:
    profiles: list[TemperatureProfile] = []
    for datasets, metadata in payloads:
        required = ("frame_index", "step", "time_fs", "time_ps", "temperature_K")
        missing = [name for name in required if name not in datasets]
        if missing:
            raise ValueError(
                f"Temperature HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )
        default_label = str(metadata.get("default_label") or "").strip()
        if not default_label:
            default_label = str(metadata.get("element") or metadata.get("region_name") or "temperature")
        atom_indices = (
            np.asarray(datasets["atom_indices"], dtype=int)
            if "atom_indices" in datasets
            else None
        )
        raw_region_elements = metadata.get("region_elements")
        if isinstance(raw_region_elements, (list, tuple)):
            region_elements = tuple(str(value) for value in raw_region_elements if str(value))
        else:
            region_elements = ()
        time_fs = np.asarray(datasets["time_fs"], dtype=float)
        profiles.append(
            TemperatureProfile(
                default_label=default_label,
                selection_kind=str(metadata.get("selection_kind") or "temperature"),
                frame_index=np.asarray(datasets["frame_index"], dtype=int),
                step=np.asarray(datasets["step"], dtype=np.int64),
                time_fs=time_fs,
                time_ps=np.asarray(datasets["time_ps"], dtype=float),
                temperature_K=np.asarray(datasets["temperature_K"], dtype=float),
                n_frames=int(metadata.get("n_frames", time_fs.size)),
                source_type=str(metadata.get("source_type") or ""),
                element=str(metadata.get("element") or "") or None,
                region_index=(
                    None
                    if metadata.get("region_index") is None
                    else int(metadata.get("region_index"))
                ),
                region_name=str(metadata.get("region_name") or "") or None,
                cp2k_list_expression=str(metadata.get("cp2k_list_expression") or "") or None,
                region_elements=region_elements,
                region_formula=str(metadata.get("region_formula") or "") or None,
                atom_count=(
                    None if metadata.get("atom_count") is None else int(metadata.get("atom_count"))
                ),
                atom_indices=atom_indices,
                atom_index_base=(
                    None
                    if metadata.get("atom_index_base") is None
                    else int(metadata.get("atom_index_base"))
                ),
                target_temperature_K=(
                    None
                    if metadata.get("target_temperature_K") is None
                    else float(metadata.get("target_temperature_K"))
                ),
                velocity_unit=str(metadata.get("velocity_unit") or "") or None,
                dof_mode=str(metadata.get("dof_mode") or "") or None,
                label_resolution_status=str(metadata.get("label_resolution_status") or "resolved"),
            )
        )
    return profiles


def load_temperature_profiles(path: str | Path) -> list[TemperatureProfile]:
    """Load one or more temperature profiles from LiNaK HDF5."""

    source_path, payloads = read_profile_payloads(
        path,
        analysis="temperature",
        label="Temperature",
    )
    return _load_temperature_profiles_from_payloads(source_path, payloads)


def load_temperature_profiles_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
) -> list[TemperatureProfile]:
    """Load selected temperature profiles by profile index from LiNaK HDF5."""

    source_path, payloads = read_profile_payloads_by_index(
        path,
        profile_indices,
        analysis="temperature",
        label="Temperature",
    )
    return _load_temperature_profiles_from_payloads(source_path, payloads)


def plot_temperature_profile(
    profile: TemperatureProfile,
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    series_id: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    time_axis: str = "ps",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_font_size: int | None = None,
    y_label_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_pad: float | None = None,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    line_label: str | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    cumulative_config: dict[str, Any] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> Path | None:
    resolved_mapping = resolve_temperature_plot_mapping(
        contract=data_contract,
        profile=profile,
        mapping=view_mapping,
        time_axis=time_axis,
    )
    runtime_time_axis = str(resolved_mapping.renderer_options.get("time_axis") or "ps")
    x_values = profile.time_fs if runtime_time_axis == "fs" else profile.time_ps
    schema_labels = default_plot_labels("temperature")
    default_x = "Time (fs)" if runtime_time_axis == "fs" else "Time (ps)"
    default_y = "Temperature (K)" if schema_labels is None else schema_labels[1]
    if schema_labels is not None and runtime_time_axis != "fs":
        default_x = schema_labels[0]
    single_series = resolve_single_series_options(
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
    )
    return plot_line_series(
        x_values,
        profile.temperature_K,
        title=title or f"{profile.default_label} temperature",
        x_label=resolve_explicit_plot_text(x_label, default_x),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_id=series_id,
        line_label=line_label if line_label is not None else profile.default_label if legend else None,
        line_color=single_series.line_color,
        line_width_override=single_series.line_width_override,
        line_marker=single_series.line_marker,
        line_visible=single_series.line_visible,
        show_in_legend=True if not series_show_in_legend else bool(series_show_in_legend[0]),
        fit_config=None if not series_fit_configs else series_fit_configs[0],
        cumulative_config=cumulative_config,
        normalization_mode=single_series.normalization_mode,
        normalization_value=single_series.normalization_value,
        normalization_x_ref=single_series.normalization_x_ref,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="temperature",
        annotations=annotations,
        integration_config=integration_config,
        style=style,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        x_label_font_size=x_label_font_size,
        y_label_font_size=y_label_font_size,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
        title_pad=title_pad,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )


def plot_temperature_profiles(
    profiles: list[TemperatureProfile],
    output: str | Path | None = None,
    show: bool = True,
    show_blocking: bool = True,
    preferred_backend: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    data_contract: PlotDataContract | None = None,
    view_mapping: PlotViewMapping | None = None,
    time_axis: str = "ps",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    y_lim: tuple[float | None, float | None] | list[float | None] | None = None,
    x_ticks: list[float] | tuple[float, ...] | None = None,
    y_ticks: list[float] | tuple[float, ...] | None = None,
    x_tick_rotation: float | None = None,
    y_tick_rotation: float | None = None,
    x_label_font_size: int | None = None,
    y_label_font_size: int | None = None,
    x_label_pad: float | None = None,
    y_label_pad: float | None = None,
    title_pad: float | None = None,
    x_axis_scale: float | None = None,
    x_axis_offset: float | None = None,
    title_visible: bool | None = None,
    ticks_visible: bool | None = None,
    markers: bool | None = None,
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    series_ids: list[str] | None = None,
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    series_enabled: list[bool] | None = None,
    series_show_in_legend: list[bool] | None = None,
    series_line_widths: list[float | None] | None = None,
    series_markers: list[str | None] | None = None,
    series_fit_configs: list[dict[str, Any] | None] | None = None,
    series_cumulative_configs: list[dict[str, Any] | None] | None = None,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
    series_normalization_modes: list[str | None] | None = None,
    series_normalization_values: list[float | None] | None = None,
    series_normalization_x_refs: list[float | None] | None = None,
    x_bin_width: float | None = None,
    x_bin_reducer: str | None = None,
    min_bin_points: int | None = None,
    annotations: list[dict[str, Any]] | None = None,
    integration_config: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None = None,
    suppress_output_log: bool = False,
    matplotlib_rc: dict[str, Any] | None = None,
    figure_kwargs: dict[str, Any] | None = None,
    axes_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
    series_line_kwargs: list[dict[str, Any] | None] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    tick_params_kwargs: dict[str, Any] | None = None,
    tight_layout_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> Path | None:
    if not profiles:
        raise ValueError("At least one temperature profile is required.")
    resolved_mapping = resolve_temperature_plot_mapping(
        contract=data_contract,
        profile=profiles[0],
        mapping=view_mapping,
        time_axis=time_axis,
    )
    runtime_time_axis = str(resolved_mapping.renderer_options.get("time_axis") or "ps")
    schema_labels = default_plot_labels("temperature")
    default_x = "Time (fs)" if runtime_time_axis == "fs" else "Time (ps)"
    if schema_labels is not None and runtime_time_axis != "fs":
        default_x = schema_labels[0]
    default_y = "Temperature (K)" if schema_labels is None else schema_labels[1]
    labels = resolve_series_labels(
        [profile.default_label for profile in profiles],
        series_labels,
        series_kind="temperature",
    )
    if not use_multi_series_plot(
        profile_count=len(profiles),
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
    ):
        return plot_temperature_profile(
            profiles[0],
            output=output,
            show=show,
            show_blocking=show_blocking,
            preferred_backend=preferred_backend,
            style=style,
            data_contract=resolved_mapping.contract,
            view_mapping=resolved_mapping.mapping,
            time_axis=runtime_time_axis,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_scale=x_scale,
            y_scale=y_scale,
            x_lim=x_lim,
            y_lim=y_lim,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            x_label_font_size=x_label_font_size,
            y_label_font_size=y_label_font_size,
            x_label_pad=x_label_pad,
            y_label_pad=y_label_pad,
            title_pad=title_pad,
            x_axis_scale=x_axis_scale,
            x_axis_offset=x_axis_offset,
            title_visible=title_visible,
            ticks_visible=ticks_visible,
            markers=markers,
            legend=legend,
            legend_title=legend_title,
            legend_loc=legend_loc,
            line_label=labels[0] if labels else None,
            line_colors=line_colors,
            series_enabled=series_enabled,
            series_show_in_legend=series_show_in_legend,
            series_line_widths=series_line_widths,
            series_markers=series_markers,
            series_fit_configs=series_fit_configs,
            cumulative_config=None if not series_cumulative_configs else series_cumulative_configs[0],
            series_normalization_modes=series_normalization_modes,
            series_normalization_values=series_normalization_values,
            series_normalization_x_refs=series_normalization_x_refs,
            x_bin_width=x_bin_width,
            x_bin_reducer=x_bin_reducer,
            min_bin_points=min_bin_points,
            annotations=annotations,
            integration_config=integration_config,
            capture_state=capture_state,
            suppress_output_log=suppress_output_log,
            matplotlib_rc=matplotlib_rc,
            figure_kwargs=figure_kwargs,
            axes_kwargs=axes_kwargs,
            line_kwargs=line_kwargs,
            grid_kwargs=grid_kwargs,
            legend_kwargs=legend_kwargs,
            tick_params_kwargs=tick_params_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
            savefig_kwargs=savefig_kwargs,
        )
    return plot_multi_line_series(
        [
            profile.time_fs if runtime_time_axis == "fs" else profile.time_ps
            for profile in profiles
        ],
        [profile.temperature_K for profile in profiles],
        labels,
        title=title or "Temperature",
        x_label=resolve_explicit_plot_text(x_label, default_x),
        y_label=resolve_explicit_plot_text(y_label, default_y),
        output=output,
        show=show,
        show_blocking=show_blocking,
        preferred_backend=preferred_backend,
        series_ids=series_ids,
        style=style,
        line_colors=line_colors,
        series_enabled=series_enabled,
        series_show_in_legend=series_show_in_legend,
        series_line_widths=series_line_widths,
        series_markers=series_markers,
        series_fit_configs=series_fit_configs,
        series_cumulative_configs=series_cumulative_configs,
        series_normalization_modes=series_normalization_modes,
        series_normalization_values=series_normalization_values,
        series_normalization_x_refs=series_normalization_x_refs,
        render_series_descriptors=render_series_descriptors,
        series_overrides_by_id=series_overrides_by_id,
        x_bin_width=x_bin_width,
        x_bin_reducer=x_bin_reducer,
        min_bin_points=min_bin_points,
        analysis_name="temperature",
        annotations=annotations,
        integration_config=integration_config,
        x_scale=x_scale,
        y_scale=y_scale,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_rotation=x_tick_rotation,
        y_tick_rotation=y_tick_rotation,
        x_label_font_size=x_label_font_size,
        y_label_font_size=y_label_font_size,
        x_label_pad=x_label_pad,
        y_label_pad=y_label_pad,
        title_pad=title_pad,
        x_axis_scale=x_axis_scale,
        x_axis_offset=x_axis_offset,
        title_visible=title_visible,
        ticks_visible=ticks_visible,
        markers=markers,
        legend=legend,
        legend_title=legend_title,
        legend_loc=legend_loc,
        capture_state=capture_state,
        matplotlib_rc=matplotlib_rc,
        figure_kwargs=figure_kwargs,
        axes_kwargs=axes_kwargs,
        line_kwargs=line_kwargs,
        series_line_kwargs=series_line_kwargs,
        grid_kwargs=grid_kwargs,
        legend_kwargs=legend_kwargs,
        tick_params_kwargs=tick_params_kwargs,
        tight_layout_kwargs=tight_layout_kwargs,
        savefig_kwargs=savefig_kwargs,
        suppress_output_log=suppress_output_log,
    )
