"""Shared frame-local atom and O/H molecule selections for pair analyses."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

from .common import normalize_species_query, select_species_indices
from .water import OHTopologyCache, molecule_positions_with_masses
from .topology import CLASSIFIER_VERSION


def pair_label(value: str | None) -> str:
    """Canonicalize pair selectors without interpreting raw labels as molecules."""
    if value is not None and str(value).startswith("atoms["):
        return str(value)
    mode, label = normalize_species_query(value, allow_molecules=True)
    if mode in {"elements", "molecules"}:
        raise ValueError(
            "Pair selectors require an individual element, molecule, raw label, or all."
        )
    if value is not None and str(value).lower().startswith("mol:") and mode != "molecule":
        raise ValueError(f"Unsupported molecular selector '{value}'.")
    return label


def is_molecular(value: str | None) -> bool:
    return pair_label(value).startswith("mol:")


@dataclass(frozen=True)
class EntitySelection:
    positions: np.ndarray
    members: np.ndarray

    @property
    def anchors(self) -> np.ndarray:
        return self.members[:, 0]


class EntityResolver:
    """Resolve every requested selector once per frame; never cache changing geometry."""

    def __init__(self, *, oh_cutoff: float = 1.27, oh_topology_backend: str = "auto"):
        if not np.isfinite(oh_cutoff) or oh_cutoff <= 0:
            raise ValueError("oh_cutoff must be positive and finite.")
        self.topology = OHTopologyCache(oh_cutoff=oh_cutoff, backend=oh_topology_backend)

    def select(
        self, frame: Atoms, labels: Iterable[str], frame_index: int = 0
    ) -> dict[str, EntitySelection]:
        labels = list(dict.fromkeys(pair_label(label) for label in labels))
        topology = (
            self.topology.select(frame, frame_index=frame_index)
            if any(label.startswith("mol:") for label in labels)
            else None
        )
        result = {}
        for label in labels:
            if label.startswith("mol:"):
                members = topology.indices_for(label)
                positions, _ = molecule_positions_with_masses(frame, members)
            else:
                if label.startswith("atoms["):
                    indices = atom_selector_indices(label)
                    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(frame)):
                        raise ValueError("RDF atom indices are outside the frame.")
                else:
                    indices = select_species_indices(frame, label)
                members = np.asarray(indices, dtype=int).reshape(-1, 1)
                positions = frame.positions[indices]
            result[label] = EntitySelection(
                np.asarray(positions, dtype=float).reshape(-1, 3), members
            )
        return result


def atom_selector_indices(label: str) -> np.ndarray:
    """Expand an RDF atom selector into its unique atom indices."""
    indices = []
    for part in label[6:-1].split(","):
        if ".." in part:
            start, end = map(int, part.split(".."))
            if start > end:
                raise ValueError("Atom-index range end must be >= start.")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return np.asarray(list(dict.fromkeys(indices)), dtype=int)


def entity_distances(
    frame: Atoms, a: EntitySelection, b: EntitySelection
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum-image distances and eligibility excluding shared members."""
    shape = (len(a.members), len(b.members))
    eligible = np.ones(shape, dtype=bool)
    for column_a in a.members.T:
        for column_b in b.members.T:
            eligible &= column_a[:, None] != column_b[None, :]
    if not all(shape):
        return np.empty(shape), eligible
    _, distances = get_distances(a.positions, b.positions, cell=frame.cell, pbc=frame.pbc)
    return distances, eligible


def molecule_metadata(label_a, label_b, oh_cutoff, backend):
    molecular = is_molecular(label_a) or is_molecular(label_b)
    return dict(
        selection_kind_a=selection_kind(label_a),
        selection_kind_b=selection_kind(label_b),
        entity_kind_a="molecule" if is_molecular(label_a) else "atom",
        entity_kind_b="molecule" if is_molecular(label_b) else "atom",
        molecular_coordinate_mode="center_of_mass" if molecular else None,
        oh_cutoff_A=float(oh_cutoff) if molecular else None,
        oh_topology_backend=backend if molecular else None,
        molecular_classifier=CLASSIFIER_VERSION if molecular else None,
        pair_exclusion="shared_members" if molecular else None,
    )


def selection_kind(label: str) -> str:
    return (
        "molecule" if is_molecular(label) else "atoms" if label.startswith("atoms[") else "species"
    )


MOLECULAR_METADATA_KEYS = (
    "selection_kind_a",
    "selection_kind_b",
    "entity_kind_a",
    "entity_kind_b",
    "molecular_coordinate_mode",
    "oh_cutoff_A",
    "oh_topology_backend",
    "molecular_classifier",
    "pair_exclusion",
    "center_tracking",
    "atom_indices_role",
    "n_entities",
)


def read_molecule_metadata(metadata):
    return (
        {key: metadata[key] for key in MOLECULAR_METADATA_KEYS if key in metadata}
        if metadata.get("molecular_coordinate_mode")
        else None
    )
