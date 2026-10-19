"""Deterministic evolving surfaces for geometry regression and timing comparisons."""

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

CASES = (
    "fixed",
    "mobile",
    "partial",
    "triclinic",
    "variable-triclinic",
    "symmetric",
    "asymmetric",
    "crossing",
    "reconstruction",
    "reactive",
    "variable-count",
    "reordered",
    "unsupported",
    "missing",
    "axis-x",
    "axis-y",
)


def surface_frames(case, frame_count=9, grid=4):
    """Return frames/options; only fixture generation imposes slab structure."""
    cell = np.diag([12.0, 12.0, 24.0])
    if case in {"triclinic", "variable-triclinic"}:
        cell = np.array([[12.0, 0, 0], [8.0, 10.0, 0], [0, 0, 24.0]])
    elif case == "unsupported":
        cell = np.array([[12.0, 0, 2.0], [6.0, 10.0, 1.0], [0, 0, 24.0]])
    xy = np.linspace(1, 10, grid)
    slab = np.array([[x, y, z] for z in (6.0, 8.0, 10.0) for x in xy for y in xy])
    # Different fluid environments on opposite sides; no imposed slab symmetry.
    water = np.array(
        [
            [3.0, 3.0, 14.0],
            [3.7, 3.0, 14.1],
            [2.3, 3.0, 14.1],
            [9.0, 9.0, 2.0],
            [9.7, 9.0, 1.9],
            [8.3, 9.0, 1.9],
        ]
    )
    if case == "asymmetric":
        water[3:, 2] += 14.0
    symbols = ["Pt"] * len(slab) + ["O", "H", "H", "O", "H", "H"]
    positions = np.concatenate((slab, water))
    if case == "reactive":
        positions = np.concatenate((positions, [[5.4, 3.0, 14.0], [5.8, 3.0, 14.8]]))
        symbols += ["O", "H"]
    frames = []
    for index in range(frame_count):
        current_cell = cell.copy()
        if case == "variable-triclinic":
            current_cell[1, 0] += 0.07 * np.sin(index * 0.1)
            current_cell *= 1 + 0.002 * np.sin(index * 0.1)
        current = positions.copy()
        if case not in {"fixed", "missing"}:
            current[: len(slab), 2] += 0.06 * np.sin(index * 0.21 + np.arange(len(slab)) * 0.12)
            current[: len(slab), :2] += 0.03 * np.sin(index * 0.17)
        water_shift = 0.7 * np.sin(index * 0.31)
        current[len(slab) :] += water_shift
        if case == "crossing":
            current[:, 2] += 12.0 + 0.8 * np.sin(index * 0.14)
        if case == "reconstruction" and index % 9 == 4:
            current[1:4, 2] = 10.3
        if case == "reactive" and index % 9 in {3, 4}:
            current[len(slab) + 1] = (
                np.array([4.2 if index % 9 == 3 else 4.5, 3.0, 14.0]) + water_shift
            )
        frame = Atoms(symbols, positions=current, cell=current_cell, pbc=True)
        frame.wrap()
        if case == "fixed":
            frame.set_constraint(FixAtoms(indices=np.arange(len(slab))))
        elif case == "partial":
            frame.set_constraint(FixAtoms(indices=np.arange(grid * grid)))
        if case == "variable-count" and index % 2:
            frame = frame[:-1]
        if case == "reordered" and index % 2:
            frame = frame[np.arange(len(frame))[::-1]]
        if case == "missing":
            frame.set_cell(np.zeros((3, 3)))
            frame.set_pbc(False)
        if case in {"axis-x", "axis-y"}:
            perm = [2, 0, 1] if case == "axis-x" else [0, 2, 1]
            frame.positions = frame.positions[:, perm]
            frame.set_cell(current_cell[np.ix_(perm, perm)])
        frames.append(frame)
    options: dict = {"axis": "x" if case == "axis-x" else "y" if case == "axis-y" else "z"}
    if case == "fixed":
        options["include_fixed_surface_atoms"] = True
    if case == "crossing":
        options["surface_elements"] = ["Pt"]
    return frames, options
