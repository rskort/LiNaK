"""Shared water-molecule detection, COM, and geometry utilities.

Extracted from density.py so multiple analysis modules can reuse
water-molecule identification and PBC-aware geometry without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dataclasses import dataclass
import logging
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

from ..progress import ProgressBar
from .common import MOLECULE_SPECIES_LABELS, normalize_molecule_label

LOGGER = logging.getLogger(__name__)

H2O_OH_CUTOFF_A: float = 1.27
"""Default O-H cutoff (Angstrom) used to identify water molecules."""

H2O_VALIDATION_STRIDE: int = 100
"""Compatibility default; topology is now resolved exactly for every frame."""

AMU_TO_G: float = 1.66053906660e-24
"""Conversion factor from atomic mass units to grams."""

_MOLECULE_ATOM_COUNTS = {
    "mol:H": 1,
    "mol:O": 1,
    "mol:OH": 2,
    "mol:H2O": 3,
    "mol:H3O": 4,
}

OH_TOPOLOGY_BACKENDS = ("auto", "ckdtree", "ase")
_MAX_CKDTREE_IMAGE_POINTS = 2_000_000
_MAX_CKDTREE_CELL_CONDITION = 1.0e10


@dataclass(frozen=True)
class OHTopology:
    """Classified O/H molecular entities for one frame."""

    groups: Mapping[str, np.ndarray]
    ambiguous_hydrogen_indices: np.ndarray
    ambiguous_oxygen_indices: np.ndarray
    overcoordinated_oxygen_indices: np.ndarray

    def indices_for(self, molecule_label: str) -> np.ndarray:
        label = normalize_molecule_label(molecule_label)
        if label is None:
            raise ValueError(f"Unsupported O/H molecule label '{molecule_label}'.")
        count = _MOLECULE_ATOM_COUNTS[label]
        return np.asarray(
            self.groups.get(label, np.empty((0, count), dtype=int)),
            dtype=int,
        )

    def signature(self) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
        parts: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        for label in MOLECULE_SPECIES_LABELS:
            values = np.asarray(self.indices_for(label), dtype=int)
            rows = tuple(tuple(int(item) for item in row) for row in values.tolist())
            parts.append((label, rows))
        return tuple(parts)


class OHTopologyCache:
    """Analysis adapter to the shared, exact per-frame topology resolver."""

    def __init__(
        self,
        *,
        oh_cutoff: float = H2O_OH_CUTOFF_A,
        validation_stride: int = H2O_VALIDATION_STRIDE,
        logger: logging.Logger | None = None,
        context: str = "O/H molecule topology",
        backend: str = "auto",
    ) -> None:
        self.oh_cutoff = float(oh_cutoff)
        self.validation_stride = max(1, int(validation_stride))
        self.logger = LOGGER if logger is None else logger
        self.context = str(context)
        self.backend = _normalize_topology_backend(backend)
        self.topology: OHTopology | None = None
        self.per_frame = True
        self._logged_backend: str | None = None

    def _detect(self, frame: Atoms) -> OHTopology:
        if self._logged_backend is None:
            self.logger.debug(
                "%s: exact shared resolver (requested backend: %s).", self.context, self.backend
            )
            self._logged_backend = self.backend
        return oh_molecule_topology(frame, oh_cutoff=self.oh_cutoff, backend=self.backend)

    def select(self, frame: Atoms, *, frame_index: int) -> OHTopology:
        self.topology = self._detect(frame)
        return self.topology


def _empty_groups() -> dict[str, np.ndarray]:
    return {
        label: np.empty((0, _MOLECULE_ATOM_COUNTS[label]), dtype=int)
        for label in MOLECULE_SPECIES_LABELS
    }


def _rows_to_array(rows: list[tuple[int, ...]], *, width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, int(width)), dtype=int)
    return np.asarray(rows, dtype=int)


def _normalize_topology_backend(value: str | None) -> str:
    backend = "auto" if value is None else str(value).strip().lower()
    if backend not in OH_TOPOLOGY_BACKENDS:
        raise ValueError(
            f"Unknown O/H topology backend '{value}'. Choose from: {', '.join(OH_TOPOLOGY_BACKENDS)}."
        )
    return backend


def _ase_oh_pairs(
    frame: Atoms,
    *,
    oh_cutoff: float,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del oxygen_indices, hydrogen_indices
    oxygen, hydrogen = neighbor_list("ij", frame, {("O", "H"): oh_cutoff})
    return oxygen.astype(int, copy=False), hydrogen.astype(int, copy=False)


def _axis_aligned_orthorhombic(cell: np.ndarray) -> bool:
    diagonal = np.diag(np.diag(cell))
    return bool(np.allclose(cell, diagonal, rtol=0.0, atol=1.0e-12))


def _ckdtree_orthorhombic_pairs(
    frame: Atoms,
    *,
    oh_cutoff: float,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cell = np.asarray(frame.cell.array, dtype=np.float64)
    lengths = np.abs(np.diag(cell))
    if not np.all(np.asarray(frame.pbc, dtype=bool)) or not _axis_aligned_orthorhombic(cell):
        raise ValueError("fully periodic axis-aligned orthorhombic cell required")
    if np.any(lengths <= 0.0):
        raise ValueError("positive orthorhombic cell lengths required")
    positions = np.asarray(frame.positions, dtype=np.float64)
    oxygen_positions = np.mod(positions[oxygen_indices], lengths)
    hydrogen_positions = np.mod(positions[hydrogen_indices], lengths)
    tree = cKDTree(hydrogen_positions, boxsize=lengths)
    neighbors = tree.query_ball_point(oxygen_positions, r=float(oh_cutoff))
    oxygen_rows: list[int] = []
    hydrogen_rows: list[int] = []
    cutoff_sq = float(oh_cutoff) ** 2
    for local_oxygen, local_hydrogens in enumerate(neighbors):
        if not local_hydrogens:
            continue
        delta = (
            hydrogen_positions[np.asarray(local_hydrogens, dtype=int)]
            - oxygen_positions[local_oxygen]
        )
        delta -= np.rint(delta / lengths) * lengths
        for local_hydrogen, vector in zip(local_hydrogens, delta):
            if float(np.dot(vector, vector)) <= cutoff_sq:
                oxygen_rows.append(int(oxygen_indices[local_oxygen]))
                hydrogen_rows.append(int(hydrogen_indices[int(local_hydrogen)]))
    return np.asarray(oxygen_rows, dtype=int), np.asarray(hydrogen_rows, dtype=int)


def _ckdtree_image_pairs(
    frame: Atoms,
    *,
    oh_cutoff: float,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cell = np.asarray(frame.cell.array, dtype=np.float64)
    determinant = float(np.linalg.det(cell))
    condition = float(np.linalg.cond(cell)) if np.isfinite(cell).all() else np.inf
    if abs(determinant) <= 1.0e-14 or condition > _MAX_CKDTREE_CELL_CONDITION:
        raise ValueError("invalid or ill-conditioned periodic cell")
    pbc = np.asarray(frame.pbc, dtype=bool)
    inverse = np.linalg.inv(cell)
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    image_radius = np.ceil(float(oh_cutoff) * reciprocal_norms).astype(int) + 1
    shift_axes = [
        range(-int(image_radius[index]), int(image_radius[index]) + 1)
        if pbc[index]
        else range(0, 1)
        for index in range(3)
    ]
    shifts = np.asarray(
        [(i, j, k) for i in shift_axes[0] for j in shift_axes[1] for k in shift_axes[2]],
        dtype=np.float64,
    )
    image_count = int(hydrogen_indices.size) * int(shifts.shape[0])
    if image_count > _MAX_CKDTREE_IMAGE_POINTS:
        raise ValueError("replicated O/H image set exceeds safety limit")
    positions = np.asarray(frame.positions, dtype=np.float64)
    image_offsets = shifts @ cell
    hydrogen_images = (positions[hydrogen_indices][None, :, :] + image_offsets[:, None, :]).reshape(
        -1, 3
    )
    original_local = np.tile(np.arange(hydrogen_indices.size, dtype=int), shifts.shape[0])
    tree = cKDTree(hydrogen_images)
    candidates = tree.query_ball_point(positions[oxygen_indices], r=float(oh_cutoff))
    cutoff_sq = float(oh_cutoff) ** 2
    pairs: set[tuple[int, int]] = set()
    for local_oxygen, image_candidates in enumerate(candidates):
        oxygen_position = positions[oxygen_indices[local_oxygen]]
        for image_index in image_candidates:
            vector = hydrogen_images[int(image_index)] - oxygen_position
            if float(np.dot(vector, vector)) <= cutoff_sq:
                pairs.add(
                    (
                        int(oxygen_indices[local_oxygen]),
                        int(hydrogen_indices[original_local[int(image_index)]]),
                    )
                )
    ordered = sorted(pairs)
    return (
        np.asarray([pair[0] for pair in ordered], dtype=int),
        np.asarray([pair[1] for pair in ordered], dtype=int),
    )


def _find_oh_pairs(
    frame: Atoms,
    *,
    oh_cutoff: float,
    backend: str,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    normalized = _normalize_topology_backend(backend)
    if normalized != "ase":
        try:
            cell = np.asarray(frame.cell.array, dtype=np.float64)
            if np.all(np.asarray(frame.pbc, dtype=bool)) and _axis_aligned_orthorhombic(cell):
                oxygen, hydrogen = _ckdtree_orthorhombic_pairs(
                    frame,
                    oh_cutoff=oh_cutoff,
                    oxygen_indices=oxygen_indices,
                    hydrogen_indices=hydrogen_indices,
                )
                return oxygen, hydrogen, "ckdtree-orthorhombic"
            oxygen, hydrogen = _ckdtree_image_pairs(
                frame,
                oh_cutoff=oh_cutoff,
                oxygen_indices=oxygen_indices,
                hydrogen_indices=hydrogen_indices,
            )
            return oxygen, hydrogen, "ckdtree-images"
        except Exception as exc:
            LOGGER.debug("cKDTree O/H topology backend unavailable (%s); using ASE.", exc)
    oxygen, hydrogen = _ase_oh_pairs(
        frame,
        oh_cutoff=oh_cutoff,
        oxygen_indices=oxygen_indices,
        hydrogen_indices=hydrogen_indices,
    )
    return oxygen, hydrogen, "ase"


# ---------------------------------------------------------------------------
# O/H molecule detection
# ---------------------------------------------------------------------------


def _oh_molecule_topology_with_backend(
    frame: Atoms,
    *,
    oh_cutoff: float,
    backend: str,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
) -> tuple[OHTopology, str]:
    """Classify O/H entities into free H/O, OH, H2O, and H3O molecule groups."""

    symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
    oxygen_all = (
        np.where(symbols == "O")[0].astype(int, copy=False)
        if oxygen_indices is None
        else np.asarray(oxygen_indices, dtype=int)
    )
    hydrogen_all = (
        np.where(symbols == "H")[0].astype(int, copy=False)
        if hydrogen_indices is None
        else np.asarray(hydrogen_indices, dtype=int)
    )
    rows: dict[str, list[tuple[int, ...]]] = {label: [] for label in MOLECULE_SPECIES_LABELS}
    ambiguous_hydrogens: set[int] = set()
    ambiguous_oxygens: set[int] = set()
    overcoordinated_oxygens: set[int] = set()

    if oxygen_all.size == 0 and hydrogen_all.size == 0:
        return (
            OHTopology(
                groups=_empty_groups(),
                ambiguous_hydrogen_indices=np.empty((0,), dtype=int),
                ambiguous_oxygen_indices=np.empty((0,), dtype=int),
                overcoordinated_oxygen_indices=np.empty((0,), dtype=int),
            ),
            "empty",
        )

    pair_oxygen_indices, pair_hydrogen_indices, resolved_backend = _find_oh_pairs(
        frame,
        oh_cutoff=oh_cutoff,
        backend=backend,
        oxygen_indices=oxygen_all,
        hydrogen_indices=hydrogen_all,
    )
    oxygen_to_hydrogen: dict[int, set[int]] = {int(index): set() for index in oxygen_all}
    hydrogen_to_oxygen: dict[int, set[int]] = {int(index): set() for index in hydrogen_all}
    for oxygen_index, hydrogen_index in zip(
        pair_oxygen_indices.astype(int, copy=False),
        pair_hydrogen_indices.astype(int, copy=False),
    ):
        oxygen_key = int(oxygen_index)
        hydrogen_key = int(hydrogen_index)
        oxygen_to_hydrogen.setdefault(oxygen_key, set()).add(hydrogen_key)
        hydrogen_to_oxygen.setdefault(hydrogen_key, set()).add(oxygen_key)

    for hydrogen_index in sorted(int(index) for index in hydrogen_all):
        bonded_oxygens = hydrogen_to_oxygen.get(hydrogen_index, set())
        if len(bonded_oxygens) == 0:
            rows["mol:H"].append((hydrogen_index,))
        elif len(bonded_oxygens) > 1:
            ambiguous_hydrogens.add(hydrogen_index)
            ambiguous_oxygens.update(int(index) for index in bonded_oxygens)

    label_by_h_count = {
        0: "mol:O",
        1: "mol:OH",
        2: "mol:H2O",
        3: "mol:H3O",
    }
    for oxygen_index in sorted(int(index) for index in oxygen_all):
        bonded_hydrogens = sorted(oxygen_to_hydrogen.get(oxygen_index, set()))
        ambiguous_neighbors = [
            hydrogen_index
            for hydrogen_index in bonded_hydrogens
            if len(hydrogen_to_oxygen.get(hydrogen_index, set())) != 1
        ]
        if ambiguous_neighbors:
            ambiguous_oxygens.add(oxygen_index)
            ambiguous_hydrogens.update(int(index) for index in ambiguous_neighbors)
            continue
        if len(bonded_hydrogens) > 3:
            overcoordinated_oxygens.add(oxygen_index)
            continue
        label = label_by_h_count[len(bonded_hydrogens)]
        rows[label].append((oxygen_index, *bonded_hydrogens))

    groups = {
        label: _rows_to_array(rows[label], width=_MOLECULE_ATOM_COUNTS[label])
        for label in MOLECULE_SPECIES_LABELS
    }
    topology = OHTopology(
        groups=groups,
        ambiguous_hydrogen_indices=np.asarray(sorted(ambiguous_hydrogens), dtype=int),
        ambiguous_oxygen_indices=np.asarray(sorted(ambiguous_oxygens), dtype=int),
        overcoordinated_oxygen_indices=np.asarray(sorted(overcoordinated_oxygens), dtype=int),
    )
    if (
        topology.ambiguous_hydrogen_indices.size
        or topology.ambiguous_oxygen_indices.size
        or topology.overcoordinated_oxygen_indices.size
    ):
        LOGGER.debug(
            "O/H molecule topology excluded ambiguous/overcoordinated atoms: H=%s O=%s O_over=%s.",
            topology.ambiguous_hydrogen_indices.tolist(),
            topology.ambiguous_oxygen_indices.tolist(),
            topology.overcoordinated_oxygen_indices.tolist(),
        )
    return topology, resolved_backend


def oh_molecule_topology(
    frame: Atoms,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    *,
    backend: str = "auto",
) -> OHTopology:
    """Resolve exact O/H membership, detecting geometry only when no valid cache exists."""

    from .topology import resolve_topology

    return resolve_topology(frame, cutoff=float(oh_cutoff), backend=backend)


def oh_molecule_indices(
    frame: Atoms,
    molecule_label: str,
    *,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    backend: str = "auto",
) -> np.ndarray:
    """Return atom-index rows for one supported O/H molecule selector."""

    return oh_molecule_topology(frame, oh_cutoff=oh_cutoff, backend=backend).indices_for(
        molecule_label
    )


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
    return oh_molecule_indices(frame, "mol:H2O", oh_cutoff=oh_cutoff)


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


class MoleculeGeometry(NamedTuple):
    """PBC-corrected COM positions and masses for one molecule group."""

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


def molecule_indices_geometry(
    frame: Atoms,
    molecule_indices: np.ndarray,
) -> MoleculeGeometry:
    """Compute PBC-aware COM positions and masses for molecule index rows."""

    indices = np.asarray(molecule_indices, dtype=int)
    if indices.size == 0:
        return MoleculeGeometry(
            com_positions=np.empty((0, 3), dtype=float),
            molecular_masses_amu=np.empty((0,), dtype=float),
        )
    if indices.ndim != 2:
        raise ValueError("Molecule index rows must be a 2D integer array.")

    positions = np.asarray(frame.positions, dtype=float)
    masses = np.asarray(frame.get_masses(), dtype=float)
    anchor_positions = positions[indices[:, 0]]
    unwrapped_positions = np.empty((indices.shape[0], indices.shape[1], 3), dtype=float)
    unwrapped_positions[:, 0, :] = anchor_positions
    for column in range(1, indices.shape[1]):
        vectors, _ = find_mic(
            positions[indices[:, column]] - anchor_positions,
            frame.cell,
            pbc=frame.pbc,
        )
        unwrapped_positions[:, column, :] = anchor_positions + vectors

    row_masses = masses[indices]
    molecular_masses_amu = np.sum(row_masses, axis=1)
    com_positions = np.sum(unwrapped_positions * row_masses[:, :, None], axis=1)
    com_positions = com_positions / molecular_masses_amu[:, None]
    return MoleculeGeometry(
        com_positions=np.asarray(com_positions, dtype=float),
        molecular_masses_amu=np.asarray(molecular_masses_amu, dtype=float),
    )


def molecule_positions_with_masses(
    frame: Atoms,
    molecule_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return molecule COM positions and molecular masses in grams."""

    geometry = molecule_indices_geometry(frame, molecule_indices)
    return (
        np.asarray(geometry.com_positions, dtype=float),
        np.asarray(geometry.molecular_masses_amu * AMU_TO_G, dtype=float),
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
    frames: Sequence[Atoms],
    axis_index: int,
    *,
    oh_cutoff: float = H2O_OH_CUTOFF_A,
    progress_desc: str = "Selecting H2O",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select water-molecule COM axis values using exact shared frame topology.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(axis_values_per_frame, masses_per_frame)`` - one array per frame.
    """
    selected_per_frame: list[np.ndarray] = []
    selected_masses_per_frame: list[np.ndarray] = []
    topology_cache = OHTopologyCache(
        oh_cutoff=oh_cutoff,
        context="H2O topology",
    )

    with ProgressBar(desc=progress_desc, total=len(frames), unit="frame") as progress:
        for frame_index, frame in enumerate(frames):
            cached_water_triplets = topology_cache.select(
                frame,
                frame_index=frame_index,
            ).indices_for("mol:H2O")
            axis_values, masses = water_triplet_axis_values_with_masses(
                frame,
                cached_water_triplets,
                axis_index,
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
