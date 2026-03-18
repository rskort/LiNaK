"""Trajectory I/O helpers built on top of ASE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
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

_ASE_LAMMPS_DATA_TO_ASE_ATOMS = getattr(ase_lammpsrun, "lammps_data_to_ase_atoms", None)


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


def _read_frames(path: Path, *, format: str | None = None) -> list[Atoms]:
    frames: list[Atoms] = []
    with ProgressBar(desc="Reading trajectory", unit="frame") as progress:
        for frame in iread(str(path), index=":", format=format):
            frames.append(frame)
            progress.update()
    return frames


def _read_lammps_dump_frames(path: Path) -> list[Atoms]:
    """Read a LAMMPS text dump frame-by-frame to keep progress responsive."""
    frames: list[Atoms] = []
    n_atoms = 0
    cell = None
    celldisp = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    info: dict[str, int] = {}

    with (
        path.open("r", encoding="utf-8") as handle,
        ProgressBar(
            desc="Reading trajectory",
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
        Supported values include ASE-supported trajectory files (e.g. `.xyz`, `.dump`)
        and LAMMPS input files (`.lmp`) that reference a dump file.

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
    LOGGER.info("Loading trajectory from '%s'.", trajectory_path)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    suffix = trajectory_path.suffix.lower()
    if suffix == ".lmp":
        frames = _read_lammps_input_trajectory(trajectory_path)
    elif suffix == ".dump":
        frames = _read_lammps_dump_frames(trajectory_path)
    else:
        frames = _read_frames(trajectory_path)

    if not frames:
        raise ValueError(f"No frames were read from: {trajectory_path}")

    LOGGER.info("Loaded %d frame(s) from '%s'.", len(frames), trajectory_path)
    if frames:
        LOGGER.debug("Atoms per frame (frame 0): %d", len(frames[0]))

    return frames


def write_trajectory(frames: list[Atoms], path: str | Path) -> Path:
    """Write trajectory frames to disk and return the written path."""
    if not frames:
        raise ValueError("At least one trajectory frame is required for writing.")

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ProgressBar(desc="Writing trajectory", total=1, unit="step") as progress:
        try:
            write(str(output_path), frames)
        except UnknownFileTypeError as exc:
            raise ValueError(
                f"Unsupported output trajectory format for '{output_path}'. "
                "Use a writable extension such as .xyz."
            ) from exc
        progress.update()
    LOGGER.info("Wrote %d frame(s) to '%s'.", len(frames), output_path)
    return output_path
