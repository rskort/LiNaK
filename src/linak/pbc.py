"""PBC application helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import re

from ase import Atoms
import numpy as np

from .trajectory.lammps import (
    extract_cell_from_lammps_input,
    extract_frame_timestep_fs_from_lammps_input,
    extract_fixed_atom_indices_from_lammps_input,
)
from .progress import ProgressBar
from .utils import ensure_positive
from .cell_geometry import cell_matrix, compact_cell, cell_description, wrapped_frame_positions

LOGGER = logging.getLogger(__name__)
SUPPORTED_SIM_INPUT_SUFFIXES = (".inp", ".lmp")

_ABC_PATTERN = re.compile(
    r"^\s*ABC(?:\s+\[[^\]]+\])?\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)
_ALPHA_BETA_GAMMA_PATTERN = re.compile(
    r"^\s*ALPHA_BETA_GAMMA(?:\s+\[[^\]]+\])?\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)
_TIMESTEP_PATTERN = re.compile(
    r"^\s*TIMESTEP(?:\s+\[([^\]]+)\])?\s+"
    r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)
_SECTION_START_PATTERN = re.compile(r"^\s*&([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
_SECTION_END_PATTERN = re.compile(r"^\s*&END(?:\s+([A-Za-z_][A-Za-z0-9_]*))?\b", re.IGNORECASE)
_MD_EACH_PATTERN = re.compile(r"^\s*MD\s+([+-]?\d+)\s*$", re.IGNORECASE)
_FIXED_ATOM_RANGE_PATTERN = re.compile(r"^([+-]?\d+)\.\.([+-]?\d+)$")


def _normalize_input_path(path: str | Path) -> Path:
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Simulation input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Simulation input path is not a file: {input_path}")
    return input_path


def _strip_cp2k_comment(line: str) -> str:
    stripped = line
    for marker in ("!", "#"):
        stripped = stripped.split(marker, 1)[0]
    return stripped.strip()


def find_unique_simulation_input(search_dir: str | Path) -> Path:
    """Find exactly one supported simulation input file in a directory."""
    directory = Path(search_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found for simulation input search: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Expected a directory for simulation input search, got: {directory}")

    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SIM_INPUT_SUFFIXES
    )
    if not candidates:
        supported = ", ".join(SUPPORTED_SIM_INPUT_SUFFIXES)
        raise FileNotFoundError(
            f"No simulation input file ({supported}) found in '{directory}'. "
            "Provide --input or --cell A B C."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple simulation input files found in '{directory}': {names}. "
            "Provide --input to choose one file, or --cell A B C."
        )
    return candidates[0]


def find_unique_cp2k_input(search_dir: str | Path) -> Path:
    """Find exactly one CP2K input file in a directory."""
    directory = Path(search_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found for CP2K input search: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Expected a directory for CP2K input search, got: {directory}")

    candidates = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".inp"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No CP2K .inp file found in '{directory}'. "
            "Provide --input /path/to/input.inp or --cell A B C."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple CP2K .inp files found in '{directory}': {names}. "
            "Provide --input to choose one file, or --cell A B C."
        )
    return candidates[0]


def extract_cell_from_simulation_input(path: str | Path) -> tuple[float, float, float]:
    """Extract full cell geometry from a CP2K or LAMMPS input file."""
    input_path = _normalize_input_path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".lmp":
        return extract_cell_from_lammps_input(input_path)
    if suffix == ".inp":
        return extract_cell_from_cp2k_input(input_path)
    supported = ", ".join(SUPPORTED_SIM_INPUT_SUFFIXES)
    raise ValueError(
        f"Unsupported simulation input format '{input_path.suffix}' for '{input_path}'. "
        f"Supported formats: {supported}."
    )


def extract_cell_from_cp2k_input(path: str | Path):
    """Read a CP2K cell without discarding angles or vector orientation."""
    from ase.units import Bohr

    input_path = _normalize_input_path(path)
    sections: list[str] = []
    candidates: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    value_pattern = re.compile(
        r"^(ABC|ALPHA_BETA_GAMMA|ANGLES|A|B|C)(?:\s+\[([^\]]+)\])?\s+(.+)$", re.I
    )
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        text = _strip_cp2k_comment(line)
        if not text:
            continue
        if _SECTION_END_PATTERN.match(text):
            if sections and sections[-1] == "CELL" and active is not None:
                candidates.append(active)
                active = None
            if sections:
                sections.pop()
            continue
        start = _SECTION_START_PATTERN.match(text)
        if start:
            sections.append(start.group(1).upper())
            if sections[-1] == "CELL":
                active = {"scope": tuple(sections), "values": {}}
            continue
        match = value_pattern.match(text)
        if match is None:
            continue
        # Keep compatibility with historical bare ABC snippets.
        if active is None:
            if sections:
                continue
            active = {"scope": (), "values": {}}
        keyword, unit, raw = match.groups()
        keyword = keyword.upper()
        unit = (
            unit or ("deg" if keyword in {"ANGLES", "ALPHA_BETA_GAMMA"} else "angstrom")
        ).lower()
        is_angle = keyword in {"ANGLES", "ALPHA_BETA_GAMMA"}
        factors = (
            {"deg": 1.0, "degree": 1.0, "rad": 180.0 / np.pi}
            if is_angle
            else {"angstrom": 1.0, "bohr": Bohr, "au": Bohr}
        )
        if unit not in factors:
            raise ValueError(
                f"Unsupported cell unit [{unit}] in '{input_path}' line {line_number}."
            )
        try:
            values = tuple(
                float(x.replace("D", "E").replace("d", "e")) * factors[unit] for x in raw.split()
            )
        except ValueError as exc:
            raise ValueError(f"Invalid cell value in '{input_path}' line {line_number}.") from exc
        if len(values) != 3:
            raise ValueError(f"Cell {keyword} requires three numbers on line {line_number}.")
        storage_key = "ALPHA_BETA_GAMMA" if is_angle else keyword
        previous = active["values"].get(storage_key)
        if previous is not None and not np.allclose(previous, values, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Conflicting {storage_key} definitions in '{input_path}' line {line_number}."
            )
        active["values"][storage_key] = values
    if active is not None:
        candidates.append(active)
    preferred = [c for c in candidates if c["scope"][-3:] == ("FORCE_EVAL", "SUBSYS", "CELL")]
    matrices = []
    for candidate in preferred or candidates:
        cell_values = candidate["values"]
        if "ABC" in cell_values and any(key in cell_values for key in ("A", "B", "C")):
            raise ValueError(f"Ambiguous ABC and A/B/C definitions in '{input_path}'.")
        if "ABC" in cell_values:
            matrix = cell_matrix(
                cell_values["ABC"], cell_values.get("ALPHA_BETA_GAMMA", (90.0, 90.0, 90.0))
            )
        elif all(key in cell_values for key in ("A", "B", "C")):
            matrix = cell_matrix([cell_values[key] for key in ("A", "B", "C")])
        else:
            continue
        matrices.append(matrix)
    if not matrices:
        raise ValueError(
            f"No valid 'ABC ...' line or A/B/C vectors found in CP2K input file '{input_path}'."
        )
    if any(not np.allclose(matrix, matrices[0], rtol=0.0, atol=1e-8) for matrix in matrices[1:]):
        raise ValueError(f"Ambiguous conflicting CELL sections in '{input_path}'.")
    LOGGER.debug("Parsed CP2K cell from '%s': %s", input_path, cell_description(matrices[0]))
    return compact_cell(matrices[0])


def extract_timestep_fs_from_cp2k_input(path: str | Path) -> float:
    """Extract a CP2K `TIMESTEP` value in fs from an input file."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"CP2K input path is not a file: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            no_comment = _strip_cp2k_comment(line)
            if not no_comment:
                continue
            match = _TIMESTEP_PATTERN.match(no_comment)
            if not match:
                continue

            unit = (match.group(1) or "").strip().lower()
            if unit and unit != "fs":
                raise ValueError(
                    f"Unsupported TIMESTEP unit '{unit}' in '{input_path}' line {line_number}. "
                    "Only [fs] is supported."
                )

            timestep_fs = float(match.group(2))
            ensure_positive("timestep_fs", timestep_fs)
            LOGGER.debug(
                "Parsed CP2K TIMESTEP from '%s' line %d: %.6g fs.",
                input_path,
                line_number,
                timestep_fs,
            )
            return timestep_fs

    raise ValueError(
        f"No valid 'TIMESTEP ...' line found in CP2K input file '{input_path}'. "
        "Provide --timestep-fs explicitly."
    )


def extract_trajectory_stride_md_from_cp2k_input(path: str | Path) -> int:
    """Extract `&TRAJECTORY / &EACH / MD N` stride from a CP2K input file.

    Returns 1 when no explicit MD stride is configured.
    """
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"CP2K input path is not a file: {input_path}")

    sections: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            no_comment = _strip_cp2k_comment(line)
            if not no_comment:
                continue

            section_end = _SECTION_END_PATTERN.match(no_comment)
            if section_end:
                name = section_end.group(1)
                if name:
                    expected = name.upper()
                    for idx in range(len(sections) - 1, -1, -1):
                        if sections[idx] == expected:
                            sections = sections[:idx]
                            break
                elif sections:
                    sections.pop()
                continue

            section_start = _SECTION_START_PATTERN.match(no_comment)
            if section_start:
                sections.append(section_start.group(1).upper())
                continue

            if "TRAJECTORY" not in sections or "EACH" not in sections:
                continue

            match = _MD_EACH_PATTERN.match(no_comment)
            if not match:
                continue

            stride_md = int(match.group(1))
            if stride_md <= 0:
                raise ValueError(
                    f"Invalid TRAJECTORY/EACH MD value '{stride_md}' in '{input_path}' line {line_number}."
                )
            LOGGER.debug(
                "Parsed CP2K TRAJECTORY stride from '%s' line %d: every %d MD step(s).",
                input_path,
                line_number,
                stride_md,
            )
            return stride_md

    LOGGER.debug(
        "No explicit TRAJECTORY/EACH MD stride found in '%s'; using default MD 1.",
        input_path,
    )
    return 1


def extract_frame_timestep_fs_from_cp2k_input(path: str | Path) -> tuple[float, float, int]:
    """Extract per-frame timestep from CP2K input as `TIMESTEP [fs] * EACH MD`."""
    input_path = Path(path).expanduser().resolve()
    md_timestep_fs = extract_timestep_fs_from_cp2k_input(input_path)
    stride_md = extract_trajectory_stride_md_from_cp2k_input(input_path)
    frame_timestep_fs = md_timestep_fs * float(stride_md)
    ensure_positive("frame_timestep_fs", frame_timestep_fs)
    LOGGER.debug(
        "Resolved frame timestep from '%s': %.6g fs (TIMESTEP %.6g fs x MD %d).",
        input_path,
        frame_timestep_fs,
        md_timestep_fs,
        stride_md,
    )
    return frame_timestep_fs, md_timestep_fs, stride_md


def extract_frame_timestep_fs_from_simulation_input(path: str | Path) -> tuple[float, float, int]:
    """Extract per-frame timestep from a CP2K or LAMMPS input file."""
    input_path = _normalize_input_path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".lmp":
        return extract_frame_timestep_fs_from_lammps_input(input_path)
    if suffix == ".inp":
        return extract_frame_timestep_fs_from_cp2k_input(input_path)
    supported = ", ".join(SUPPORTED_SIM_INPUT_SUFFIXES)
    raise ValueError(
        f"Unsupported simulation input format '{input_path.suffix}' for '{input_path}'. "
        f"Supported formats: {supported}."
    )


def _expand_cp2k_fixed_atom_list(raw: str, *, input_path: Path, line_number: int) -> list[int]:
    tokens = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]
    indices: list[int] = []
    for token in tokens:
        range_match = _FIXED_ATOM_RANGE_PATTERN.match(token)
        if range_match:
            start = int(range_match.group(1))
            stop = int(range_match.group(2))
            step = 1 if stop >= start else -1
            indices.extend(range(start, stop + step, step))
            continue
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported FIXED_ATOMS LIST token '{token}' in '{input_path}' line {line_number}."
            ) from exc
    return indices


def extract_fixed_atom_indices_from_cp2k_input(path: str | Path) -> tuple[int, ...] | None:
    """Extract zero-based fixed atom indices from CP2K ``&FIXED_ATOMS / LIST`` entries."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"CP2K input path is not a file: {input_path}")

    sections: list[str] = []
    fixed_atom_indices: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            no_comment = _strip_cp2k_comment(line)
            if not no_comment:
                continue

            section_end = _SECTION_END_PATTERN.match(no_comment)
            if section_end:
                name = section_end.group(1)
                if name:
                    expected = name.upper()
                    for idx in range(len(sections) - 1, -1, -1):
                        if sections[idx] == expected:
                            sections = sections[:idx]
                            break
                elif sections:
                    sections.pop()
                continue

            section_start = _SECTION_START_PATTERN.match(no_comment)
            if section_start:
                sections.append(section_start.group(1).upper())
                continue

            if "CONSTRAINT" not in sections or "FIXED_ATOMS" not in sections:
                continue
            if not no_comment.upper().startswith("LIST"):
                continue

            list_payload = no_comment[4:].strip()
            indices = _expand_cp2k_fixed_atom_list(
                list_payload,
                input_path=input_path,
                line_number=line_number,
            )
            fixed_atom_indices.update(index - 1 for index in indices if index > 0)

    if not fixed_atom_indices:
        return None
    return tuple(sorted(fixed_atom_indices))


def extract_fixed_atom_indices_from_simulation_input(path: str | Path) -> tuple[int, ...] | None:
    """Extract zero-based fixed atom indices from a CP2K or LAMMPS input file."""
    input_path = _normalize_input_path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".lmp":
        return extract_fixed_atom_indices_from_lammps_input(input_path)
    if suffix == ".inp":
        return extract_fixed_atom_indices_from_cp2k_input(input_path)
    supported = ", ".join(SUPPORTED_SIM_INPUT_SUFFIXES)
    raise ValueError(
        f"Unsupported simulation input format '{input_path.suffix}' for '{input_path}'. "
        f"Supported formats: {supported}."
    )


def resolve_cell_dimensions(
    *,
    output_path: str | Path,
    input_path: str | Path | None = None,
    cell: Any = None,
):
    """Resolve full cell geometry from explicit arguments or input discovery."""
    if input_path is not None and cell is not None:
        raise ValueError("Use either --input or --cell, not both.")

    if cell is not None:
        return compact_cell(cell)

    if input_path is not None:
        return extract_cell_from_simulation_input(input_path)

    out_path = Path(output_path).expanduser().resolve()
    auto_input = find_unique_simulation_input(out_path.parent)
    return extract_cell_from_simulation_input(auto_input)


def apply_pbc_to_frames(
    frames: list[Atoms],
    cell: Any | None,
) -> list[Atoms]:
    """Apply PBC and wrap positions into one cell or each frame's own cell.

    Passing ``cell=None`` preserves variable-cell trajectories without forcing
    callers to invoke this batch operation once per frame.
    """
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    cell_array = None if cell is None else cell_matrix(cell)
    wrapped_frames: list[Atoms] = []
    with ProgressBar(desc="Applying PBC", total=len(frames), unit="frame") as progress:
        for frame in frames:
            wrapped = frame.copy()
            wrapped.set_cell(cell_matrix(frame.cell) if cell_array is None else cell_array)
            wrapped.set_pbc((True, True, True))
            wrapped.positions[:] = wrapped_frame_positions(wrapped)
            wrapped_frames.append(wrapped)
            progress.update()

    return wrapped_frames
