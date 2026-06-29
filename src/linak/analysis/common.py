"""Reusable helpers shared across analysis modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..storage.hdf5_utils import (
    is_hdf5_path,
    read_linak_hdf5_profiles,
    read_linak_hdf5_profiles_by_index,
    write_linak_hdf5_profile_collection,
)


MOLECULE_SPECIES_LABELS: tuple[str, ...] = (
    "mol:H",
    "mol:O",
    "mol:OH",
    "mol:H2O",
    "mol:H3O",
)

_MOLECULE_FORMULA_LABELS = {label[4:].upper(): label for label in MOLECULE_SPECIES_LABELS}
_MOLECULE_FORMULA_LABELS["HO"] = "mol:OH"
_MOLECULE_FORMULA_LABELS["OH"] = "mol:OH"

_MOLECULE_DISPLAY_LABELS = {
    "mol:H": "free H",
    "mol:O": "free O",
    "mol:OH": "OH",
    "mol:H2O": "H2O",
    "mol:H3O": "H3O",
}


def normalize_molecule_label(species: str | None) -> str | None:
    """Return the canonical molecule label for supported O/H molecule selectors."""

    if species is None:
        return None
    token = str(species).strip()
    if not token:
        return None
    if token.lower().startswith("mol:"):
        formula = token[4:].strip().upper()
        return _MOLECULE_FORMULA_LABELS.get(formula)
    formula = token.upper()
    if formula in {"H", "O"}:
        return None
    return _MOLECULE_FORMULA_LABELS.get(formula)


def is_molecule_species_label(species: str | None) -> bool:
    """Return whether *species* is a canonical supported molecule label."""

    return normalize_molecule_label(species) in MOLECULE_SPECIES_LABELS


def molecule_display_label(species: str | None) -> str:
    """Return a short user-facing label for a supported molecule selector."""

    label = normalize_molecule_label(species)
    if label is None:
        return "" if species is None else str(species)
    return _MOLECULE_DISPLAY_LABELS[label]


def normalize_species_label(species: str | None) -> str:
    """Normalize user-facing species selectors used by atom-resolved analyses."""
    if species is None:
        return "ALL"

    token = str(species).strip()
    if not token or token.lower() == "all" or token == "*":
        return "ALL"
    if token.lower() in {"elements", "molecules"}:
        return token.upper()
    molecule_label = normalize_molecule_label(token)
    if token.lower().startswith("mol:") and molecule_label is not None:
        return molecule_label
    if token.upper() == "H2O":
        return "H2O"
    return token[0].upper() + token[1:].lower()


def normalize_species_query(
    species: str | None,
    *,
    allow_h2o: bool = False,
    allow_molecules: bool = False,
) -> tuple[str, str]:
    """Resolve density-style species selectors to a mode token plus canonical label."""
    if species is None:
        return "all", "ALL"

    token = str(species).strip()
    if not token or token.lower() == "all" or token == "*":
        return "all", "ALL"
    if token.lower() == "elements":
        return "elements", "ELEMENTS"
    if token.lower() == "molecules":
        return "molecules", "MOLECULES"
    molecule_label = normalize_molecule_label(token)
    if molecule_label is not None and (allow_molecules or (allow_h2o and molecule_label == "mol:H2O")):
        return "molecule", molecule_label
    return "element", normalize_species_label(token)


def available_element_species(frames: list[Atoms]) -> list[str]:
    """Return sorted unique element symbols found across all frames."""
    species_set: set[str] = set()
    for frame in frames:
        species_set.update(str(symbol) for symbol in frame.get_chemical_symbols())
    return sorted(species_set)


def select_species_indices(frame: Atoms, species: str) -> np.ndarray:
    """Return atom indices for one normalized species selection."""
    if species == "ALL":
        return np.arange(len(frame), dtype=int)
    symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
    return np.where(symbols == species)[0].astype(int, copy=False)


def frame_has_usable_cell(
    frame: Atoms,
    *,
    axis_index: int | None = None,
    require_all_pbc: bool = False,
) -> bool:
    """Return whether a frame has a finite non-zero cell compatible with analysis use."""
    if require_all_pbc and not bool(np.all(frame.get_pbc())):
        return False

    cell = np.asarray(frame.cell.array, dtype=float)
    if cell.shape != (3, 3):
        return False

    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(volume) or volume <= 0.0:
        return False

    if axis_index is not None:
        axis_length = float(np.linalg.norm(cell[int(axis_index)]))
        if not np.isfinite(axis_length) or axis_length <= 0.0:
            return False

    if require_all_pbc:
        lengths = np.asarray(frame.cell.lengths(), dtype=float)
        if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
            return False

    return True


def optional_finite_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for missing/invalid values."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def optional_cell_lengths(value: Any) -> tuple[float, float, float] | None:
    """Parse three positive finite cell lengths from common metadata payloads."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None
    if len(items) < 3:
        return None

    parsed: list[float] = []
    for raw in items[:3]:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        parsed.append(numeric)
    return (parsed[0], parsed[1], parsed[2])


def validate_stable_atom_layout(
    frames: list[Atoms],
    *,
    description: str,
) -> np.ndarray:
    """Validate constant atom count and symbol layout across a frame sequence."""
    if not frames:
        raise ValueError("At least one trajectory frame is required.")

    reference_symbols = np.asarray(frames[0].get_chemical_symbols(), dtype=object)
    for frame_index, frame in enumerate(frames[1:], start=1):
        symbols = np.asarray(frame.get_chemical_symbols(), dtype=object)
        if symbols.size != reference_symbols.size:
            raise ValueError(
                f"{description} requires all frames to have the same atom count "
                f"(frame 0: {reference_symbols.size}, frame {frame_index}: {symbols.size})."
            )
        if not np.array_equal(symbols, reference_symbols):
            raise ValueError(
                f"{description} requires a stable atom ordering/symbol layout "
                f"across frames (mismatch at frame {frame_index})."
            )
    return reference_symbols


def resolve_profile_source_path(path: str | Path, *, label: str) -> Path:
    """Resolve and validate an HDF5 analysis-profile path."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"{label} profile not found: {source_path}")
    if not is_hdf5_path(source_path):
        raise ValueError(
            f"Unsupported {label.lower()} profile format for '{source_path}'. Use .h5/.hdf5."
        )
    return source_path


def read_profile_payloads(
    path: str | Path,
    *,
    analysis: str,
    label: str,
) -> tuple[Path, list[tuple[dict[str, np.ndarray], dict[str, Any]]]]:
    """Load all LiNaK HDF5 profile payloads for one analysis type."""
    source_path = resolve_profile_source_path(path, label=label)
    payloads = read_linak_hdf5_profiles(source_path, expected_analysis=analysis)
    return source_path, payloads


def read_profile_payloads_by_index(
    path: str | Path,
    profile_indices: list[int] | tuple[int, ...],
    *,
    analysis: str,
    label: str,
) -> tuple[Path, list[tuple[dict[str, np.ndarray], dict[str, Any]]]]:
    """Load selected LiNaK HDF5 profile payloads for one analysis type."""
    source_path = resolve_profile_source_path(path, label=label)
    payloads = read_linak_hdf5_profiles_by_index(
        source_path,
        profile_indices,
        expected_analysis=analysis,
    )
    return source_path, payloads


def write_profile_collection(
    output: str | Path,
    *,
    analysis: str,
    profiles: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a LiNaK multi-profile HDF5 collection."""
    return write_linak_hdf5_profile_collection(
        output,
        analysis=analysis,
        profiles=profiles,
        metadata=dict(metadata or {}),
    )


def use_multi_series_plot(
    *,
    profile_count: int,
    render_series_descriptors: list[dict[str, Any]] | None = None,
    series_overrides_by_id: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Return whether the shared multi-series plot path should be used."""
    return profile_count != 1 or bool(render_series_descriptors) or bool(series_overrides_by_id)
