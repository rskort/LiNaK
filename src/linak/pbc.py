"""PBC application helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
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
    """Extract orthorhombic cell dimensions from a CP2K or LAMMPS input file."""
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


def extract_cell_from_cp2k_input(path: str | Path) -> tuple[float, float, float]:
    """Extract orthorhombic `ABC` cell dimensions from a CP2K input file."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"CP2K input path is not a file: {input_path}")

    cell: tuple[float, float, float] | None = None
    angles: tuple[float, float, float] | None = None
    angle_line_number: int | None = None
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            no_comment = _strip_cp2k_comment(line)
            if not no_comment:
                continue
            abc_match = _ABC_PATTERN.match(no_comment)
            if abc_match:
                cell = (
                    float(abc_match.group(1)),
                    float(abc_match.group(2)),
                    float(abc_match.group(3)),
                )
                continue

            angle_match = _ALPHA_BETA_GAMMA_PATTERN.match(no_comment)
            if angle_match:
                angles = (
                    float(angle_match.group(1)),
                    float(angle_match.group(2)),
                    float(angle_match.group(3)),
                )
                angle_line_number = line_number

    if cell is None:
        raise ValueError(
            f"No valid 'ABC ...' line found in CP2K input file '{input_path}'. "
            "Provide --input with a valid file or --cell A B C."
        )

    ensure_positive("cell_a", cell[0])
    ensure_positive("cell_b", cell[1])
    ensure_positive("cell_c", cell[2])

    if angles is not None:
        if not all(abs(angle - 90.0) <= 1e-6 for angle in angles):
            raise ValueError(
                f"CP2K input '{input_path}' defines ALPHA_BETA_GAMMA "
                f"{angles[0]:.6g} {angles[1]:.6g} {angles[2]:.6g} on line "
                f"{angle_line_number}, but LiNaK PBC handling currently supports "
                "orthorhombic cells only (90 90 90)."
            )

    try:
        display = os.path.relpath(input_path)
    except ValueError:
        display = str(input_path)
    LOGGER.debug(
        "Parsed CP2K cell from '%s': %.6g \u00d7 %.6g \u00d7 %.6g \u00c5 (90\u00b0 90\u00b0 90\u00b0).",
        display,
        cell[0],
        cell[1],
        cell[2],
    )
    return cell


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
    cell: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Resolve cell dimensions from explicit arguments or simulation input discovery."""
    if input_path is not None and cell is not None:
        raise ValueError("Use either --input or --cell, not both.")

    if cell is not None:
        ensure_positive("cell_a", cell[0])
        ensure_positive("cell_b", cell[1])
        ensure_positive("cell_c", cell[2])
        return cell

    if input_path is not None:
        return extract_cell_from_simulation_input(input_path)

    out_path = Path(output_path).expanduser().resolve()
    auto_input = find_unique_simulation_input(out_path.parent)
    return extract_cell_from_simulation_input(auto_input)


def apply_pbc_to_frames(
    frames: list[Atoms],
    cell: tuple[float, float, float],
) -> list[Atoms]:
    """Apply orthorhombic PBC and wrap atom positions into the unit cell."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    ensure_positive("cell_a", cell[0])
    ensure_positive("cell_b", cell[1])
    ensure_positive("cell_c", cell[2])

    cell_array = np.asarray(cell, dtype=float)
    wrapped_frames: list[Atoms] = []
    with ProgressBar(desc="Applying PBC", total=len(frames), unit="frame") as progress:
        for frame in frames:
            wrapped = frame.copy()
            wrapped.set_cell(cell)
            wrapped.set_pbc((True, True, True))
            wrapped.positions[:] = np.mod(np.asarray(wrapped.positions, dtype=float), cell_array)
            wrapped_frames.append(wrapped)
            progress.update()

    return wrapped_frames
