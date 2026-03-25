"""Shared water-molecule detection, COM, and geometry utilities.

Extracted from density.py so multiple analysis modules can reuse
water-molecule identification and PBC-aware geometry without duplicating logic.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list

from ..progress import ProgressBar

LOGGER = logging.getLogger(__name__)

H2O_OH_CUTOFF_A: float = 1.25
"""Default O-H cutoff (Angstrom) used to identify water molecules."""

H2O_VALIDATION_STRIDE: int = 100
"""Re-validate cached water topology every *N* frames."""

AMU_TO_G: float = 1.66053906660e-24
"""Conversion factor from atomic mass units to grams."""


# ---------------------------------------------------------------------------
# Water molecule detection
# ---------------------------------------------------------------------------

def water_molecule_triplets(
    frame: Atoms,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
) -> np.ndarray:
    """Return unique ``(O, H1, H2)`` index triplets for genuine water molecules.

    A valid water molecule has exactly one O bonded to exactly two H atoms
    within *oh_cutoff*, where each H is bonded to exactly one O.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_molecules, 3)``.
    """
    oxygen_indices, hydrogen_indices = neighbor_list(
        "ij", frame, {("O", "H"): oh_cutoff}
    )
    if oxygen_indices.size == 0:
        return np.empty((0, 3), dtype=int)

    oxygen_to_hydrogen: dict[int, set[int]] = {}
    hydrogen_to_oxygen: dict[int, set[int]] = {}
    for oxygen_index, hydrogen_index in zip(
        oxygen_indices.astype(int, copy=False),
        hydrogen_indices.astype(int, copy=False),
    ):
        oxygen_key = int(oxygen_index)
        hydrogen_key = int(hydrogen_index)
        oxygen_to_hydrogen.setdefault(oxygen_key, set()).add(hydrogen_key)
        hydrogen_to_oxygen.setdefault(hydrogen_key, set()).add(oxygen_key)

    water_triplets: list[tuple[int, int, int]] = []
    for oxygen_index in sorted(oxygen_to_hydrogen):
        bonded_hydrogen_indices = sorted(oxygen_to_hydrogen[oxygen_index])
        if len(bonded_hydrogen_indices) != 2:
            continue
        if any(
            len(hydrogen_to_oxygen[hydrogen_index]) != 1
            for hydrogen_index in bonded_hydrogen_indices
        ):
            continue
        water_triplets.append(
            (oxygen_index, bonded_hydrogen_indices[0], bonded_hydrogen_indices[1])
        )

    if not water_triplets:
        return np.empty((0, 3), dtype=int)
    return np.asarray(water_triplets, dtype=int)


# ---------------------------------------------------------------------------
# PBC-aware geometry helpers
# ---------------------------------------------------------------------------

class WaterGeometry(NamedTuple):
    """PBC-corrected water molecule geometry for a single frame.

    All position arrays have shape ``(n_molecules, 3)``.
    """

    oxygen_positions: np.ndarray
    hydrogen1_positions: np.ndarray
    hydrogen2_positions: np.ndarray
    com_positions: np.ndarray
    molecular_masses_amu: np.ndarray


def water_triplet_geometry(
    frame: Atoms,
    water_triplets: np.ndarray,
) -> WaterGeometry:
    """Compute PBC-corrected O/H1/H2 positions and mass-weighted COM.

    Parameters
    ----------
    frame
        Single ASE Atoms snapshot.
    water_triplets
        ``(n_molecules, 3)`` integer index array from
        :func:`water_molecule_triplets`.

    Returns
    -------
    WaterGeometry
        Named tuple with PBC-unwrapped positions and COM.
    """
    if water_triplets.size == 0:
        empty = np.empty((0, 3), dtype=float)
        return WaterGeometry(
            oxygen_positions=empty,
            hydrogen1_positions=empty,
            hydrogen2_positions=empty,
            com_positions=empty,
            molecular_masses_amu=np.empty((0,), dtype=float),
        )

    oxygen_indices = water_triplets[:, 0]
    hydrogen1_indices = water_triplets[:, 1]
    hydrogen2_indices = water_triplets[:, 2]

    positions = np.asarray(frame.positions, dtype=float)
    oxygen_positions = positions[oxygen_indices]

    hydrogen1_vectors, _ = find_mic(
        positions[hydrogen1_indices] - oxygen_positions,
        frame.cell,
        pbc=frame.pbc,
    )
    hydrogen2_vectors, _ = find_mic(
        positions[hydrogen2_indices] - oxygen_positions,
        frame.cell,
        pbc=frame.pbc,
    )

    hydrogen1_positions = oxygen_positions + hydrogen1_vectors
    hydrogen2_positions = oxygen_positions + hydrogen2_vectors

    atomic_masses_amu = np.asarray(frame.get_masses(), dtype=float)
    oxygen_masses = atomic_masses_amu[oxygen_indices]
    hydrogen1_masses = atomic_masses_amu[hydrogen1_indices]
    hydrogen2_masses = atomic_masses_amu[hydrogen2_indices]
    molecular_masses_amu = oxygen_masses + hydrogen1_masses + hydrogen2_masses

    com_positions = (
        oxygen_positions * oxygen_masses[:, None]
        + hydrogen1_positions * hydrogen1_masses[:, None]
        + hydrogen2_positions * hydrogen2_masses[:, None]
    ) / molecular_masses_amu[:, None]

    return WaterGeometry(
        oxygen_positions=oxygen_positions,
        hydrogen1_positions=hydrogen1_positions,
        hydrogen2_positions=hydrogen2_positions,
        com_positions=com_positions,
        molecular_masses_amu=molecular_masses_amu,
    )


def water_triplet_axis_values_with_masses(
    frame: Atoms,
    water_triplets: np.ndarray,
    axis_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PBC-aware water COM axis positions and molecular masses in grams.

    This is the original helper used by the density analysis.
    """
    geom = water_triplet_geometry(frame, water_triplets)
    if geom.com_positions.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    axis_values = np.asarray(geom.com_positions[:, axis_index], dtype=float)
    molecular_masses = np.asarray(geom.molecular_masses_amu * AMU_TO_G, dtype=float)
    return axis_values, molecular_masses


def water_axis_values_per_frame(
    frames: list[Atoms],
    axis_index: int,
    *,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    progress_desc: str = "Selecting H2O",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select water-molecule COM axis values with cached-topology validation.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(axis_values_per_frame, masses_per_frame)`` - one array per frame.
    """
    selected_per_frame: list[np.ndarray] = []
    selected_masses_per_frame: list[np.ndarray] = []
    cached_water_triplets: np.ndarray | None = None

    with ProgressBar(desc=progress_desc, total=len(frames), unit="frame") as progress:
        for frame_index, frame in enumerate(frames):
            if cached_water_triplets is None:
                cached_water_triplets = water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
            elif frame_index % H2O_VALIDATION_STRIDE == 0:
                validated = water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
                if not np.array_equal(validated, cached_water_triplets):
                    LOGGER.warning(
                        "Detected H2O topology change at frame %d; refreshing cached water triplets.",
                        frame_index,
                    )
                    cached_water_triplets = validated

            axis_values, masses = water_triplet_axis_values_with_masses(
                frame, cached_water_triplets, axis_index,
            )
            selected_per_frame.append(axis_values)
            selected_masses_per_frame.append(masses)
            progress.update()

    return selected_per_frame, selected_masses_per_frame


def water_oxygen_indices(
    frame: Atoms,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
) -> np.ndarray:
    """Return oxygen indices classified as water oxygens."""
    triplets = water_molecule_triplets(frame, oh_cutoff=oh_cutoff)
    if triplets.size == 0:
        return np.array([], dtype=int)
    return triplets[:, 0].astype(int, copy=False)
