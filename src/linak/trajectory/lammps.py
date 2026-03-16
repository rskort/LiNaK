"""LAMMPS parsing helpers for trajectory discovery and metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
import re
import shlex

from ..utils import ensure_positive

_FLOAT_PATTERN = r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?"
_LO_HI_PATTERN = re.compile(
    rf"^\s*({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+([xyz])lo\s+\3hi\s*$",
    re.IGNORECASE,
)
_TILT_PATTERN = re.compile(
    rf"^\s*({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+xy\s+xz\s+yz\s*$",
    re.IGNORECASE,
)

# LAMMPS base time units converted to femtoseconds.
_LAMMPS_TIME_UNIT_TO_FS = {
    "real": 1.0,
    "metal": 1000.0,
    "si": 1.0e15,
    "cgs": 1.0e15,
    "electron": 1.0,
    "micro": 1.0e9,
    "nano": 1.0e6,
}


@dataclass(frozen=True)
class LammpsDumpCommand:
    """Parsed ``dump`` command metadata."""

    line_number: int
    every: int | None
    path: Path


@dataclass(frozen=True)
class LammpsInputMetadata:
    """Parsed metadata from a LAMMPS input script."""

    input_path: Path
    units: str | None
    timestep: float | None
    read_data_path: Path | None
    dumps: tuple[LammpsDumpCommand, ...]


def _strip_lammps_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _tokenize_lammps_line(line: str) -> list[str]:
    stripped = _strip_lammps_comment(line)
    if not stripped:
        return []
    try:
        return shlex.split(stripped, posix=True)
    except ValueError:
        return stripped.split()


def _resolve_path(input_path: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (input_path.parent / raw).resolve()


def parse_lammps_input(path: str | Path) -> LammpsInputMetadata:
    """Parse key metadata from a LAMMPS ``.lmp`` input script."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"LAMMPS input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"LAMMPS input path is not a file: {input_path}")

    units: str | None = None
    timestep: float | None = None
    read_data_path: Path | None = None
    dumps: list[LammpsDumpCommand] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            tokens = _tokenize_lammps_line(line)
            if not tokens:
                continue

            command = tokens[0].lower()
            if command == "units" and len(tokens) >= 2:
                units = tokens[1].strip().lower()
                continue

            if command == "timestep" and len(tokens) >= 2:
                try:
                    parsed_timestep = float(tokens[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid LAMMPS timestep value '{tokens[1]}' in '{input_path}' line {line_number}."
                    ) from exc
                ensure_positive("lammps_timestep", parsed_timestep)
                timestep = parsed_timestep
                continue

            if command == "read_data" and len(tokens) >= 2:
                read_data_path = _resolve_path(input_path, tokens[1])
                continue

            if command == "dump" and len(tokens) >= 6:
                every: int | None
                try:
                    every = int(tokens[4])
                    if every <= 0:
                        raise ValueError
                except ValueError:
                    every = None
                dump_path = _resolve_path(input_path, tokens[5])
                dumps.append(
                    LammpsDumpCommand(
                        line_number=line_number,
                        every=every,
                        path=dump_path,
                    )
                )

    return LammpsInputMetadata(
        input_path=input_path,
        units=units,
        timestep=timestep,
        read_data_path=read_data_path,
        dumps=tuple(dumps),
    )


def _expand_dump_candidates(command: LammpsDumpCommand) -> list[Path]:
    pattern = str(command.path)
    if any(char in pattern for char in ["*", "?", "["]):
        return sorted(Path(match).expanduser().resolve() for match in glob.glob(pattern))
    if command.path.exists():
        return [command.path]
    return []


def resolve_dump_path_from_lammps_input(path: str | Path) -> tuple[Path, int | None]:
    """Resolve exactly one existing dump file referenced by a LAMMPS input file."""
    metadata = parse_lammps_input(path)
    if not metadata.dumps:
        raise ValueError(
            f"No 'dump ...' command found in LAMMPS input file '{metadata.input_path}'. "
            "Provide a trajectory .dump path directly."
        )

    candidates: dict[Path, int | None] = {}
    for command in metadata.dumps:
        matches = _expand_dump_candidates(command)
        if not matches:
            continue
        if len(matches) > 1:
            names = ", ".join(match.name for match in matches)
            raise ValueError(
                f"LAMMPS dump pattern from '{metadata.input_path}' line {command.line_number} matched "
                f"multiple files ({names}). Provide one explicit .dump file path."
            )
        match = matches[0]
        if match not in candidates:
            candidates[match] = command.every

    if not candidates:
        if len(metadata.dumps) == 1:
            missing = metadata.dumps[0].path
            raise FileNotFoundError(
                f"LAMMPS dump file referenced by '{metadata.input_path}' was not found: {missing}"
            )
        raise FileNotFoundError(
            f"No existing dump file referenced by LAMMPS input '{metadata.input_path}' was found."
        )

    if len(candidates) > 1:
        names = ", ".join(path.name for path in sorted(candidates))
        raise ValueError(
            f"Multiple dump files are referenced by '{metadata.input_path}': {names}. "
            "Provide one explicit .dump file path."
        )

    resolved_path = next(iter(candidates))
    every = candidates[resolved_path]
    return resolved_path, every


def extract_cell_from_lammps_data_file(path: str | Path) -> tuple[float, float, float]:
    """Extract orthorhombic box lengths from a LAMMPS ``read_data`` file."""
    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {data_path}")
    if not data_path.is_file():
        raise ValueError(f"LAMMPS data path is not a file: {data_path}")

    lengths: dict[str, float] = {}
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            no_comment = line.split("#", 1)[0].strip()
            if not no_comment:
                continue

            lo_hi_match = _LO_HI_PATTERN.match(no_comment)
            if lo_hi_match:
                lo = float(lo_hi_match.group(1))
                hi = float(lo_hi_match.group(2))
                axis = lo_hi_match.group(3).lower()
                length = hi - lo
                ensure_positive(f"cell_{axis}", length)
                lengths[axis] = length
                continue

            tilt_match = _TILT_PATTERN.match(no_comment)
            if tilt_match:
                xy = float(tilt_match.group(1))
                xz = float(tilt_match.group(2))
                yz = float(tilt_match.group(3))
                if any(abs(value) > 1e-12 for value in (xy, xz, yz)):
                    raise ValueError(
                        f"LAMMPS triclinic tilt factors are not supported in '{data_path}' line {line_number}. "
                        "Provide --cell A B C explicitly."
                    )

    missing_axes = [axis for axis in ("x", "y", "z") if axis not in lengths]
    if missing_axes:
        missing = ", ".join(f"{axis}lo/{axis}hi" for axis in missing_axes)
        raise ValueError(
            f"Missing LAMMPS cell bounds ({missing}) in '{data_path}'. "
            "Provide --cell A B C explicitly."
        )

    return lengths["x"], lengths["y"], lengths["z"]


def extract_cell_from_lammps_input(path: str | Path) -> tuple[float, float, float]:
    """Extract cell lengths from a LAMMPS input by following ``read_data``."""
    metadata = parse_lammps_input(path)
    if metadata.read_data_path is None:
        raise ValueError(
            f"No 'read_data ...' command found in LAMMPS input file '{metadata.input_path}'. "
            "Provide --cell A B C explicitly."
        )
    return extract_cell_from_lammps_data_file(metadata.read_data_path)


def _extract_md_timestep_fs(metadata: LammpsInputMetadata) -> float:
    if metadata.units is None:
        raise ValueError(
            f"No 'units ...' command found in LAMMPS input file '{metadata.input_path}'. "
            "Provide --timestep-fs explicitly."
        )
    if metadata.timestep is None:
        raise ValueError(
            f"No 'timestep ...' command found in LAMMPS input file '{metadata.input_path}'. "
            "Provide --timestep-fs explicitly."
        )

    factor = _LAMMPS_TIME_UNIT_TO_FS.get(metadata.units)
    if factor is None:
        raise ValueError(
            f"Unsupported LAMMPS time unit '{metadata.units}' in '{metadata.input_path}'. "
            "Provide --timestep-fs explicitly."
        )

    md_timestep_fs = metadata.timestep * factor
    ensure_positive("md_timestep_fs", md_timestep_fs)
    return md_timestep_fs


def extract_frame_timestep_fs_from_lammps_input(path: str | Path) -> tuple[float, float, int]:
    """Extract per-frame timestep from LAMMPS input as ``timestep * dump_every``."""
    metadata = parse_lammps_input(path)
    md_timestep_fs = _extract_md_timestep_fs(metadata)

    stride_md: int
    if not metadata.dumps:
        stride_md = 1
    else:
        parsed_stride = next(
            (dump.every for dump in metadata.dumps if dump.every is not None),
            None,
        )
        if parsed_stride is None:
            raise ValueError(
                f"Could not parse a numeric dump stride from '{metadata.input_path}'. "
                "Provide --timestep-fs explicitly."
            )
        stride_md = parsed_stride

    frame_timestep_fs = md_timestep_fs * float(stride_md)
    ensure_positive("frame_timestep_fs", frame_timestep_fs)
    return frame_timestep_fs, md_timestep_fs, stride_md
