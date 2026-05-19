"""Shared cube-file/container IO for LiNaK."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis.common import write_profile_collection
from .storage.hdf5_utils import read_linak_hdf5_profiles


@dataclass(frozen=True)
class CubeDataset:
    """Logical cube field that can come from raw `.cube` or LiNaK `.cube.h5`."""

    comment_1: str
    comment_2: str
    natoms_signed: int
    origin_bohr: np.ndarray
    grid_counts_signed: np.ndarray
    grid_vectors_bohr: np.ndarray
    atom_numbers: np.ndarray
    atom_charges: np.ndarray
    atom_positions_bohr: np.ndarray
    values: np.ndarray
    source_path: str | None = None
    source_name: str | None = None
    source_file_type: str | None = None
    source_profile_index: int | None = None


def is_linak_cube_hdf5(path: str | Path) -> bool:
    candidate = str(Path(path).expanduser().resolve()).lower()
    return candidate.endswith(".cube.h5") or candidate.endswith(".cube.hdf5")


def validate_cube_source(path: str | Path) -> Path:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Hartree cube file not found: {source_path}")
    if source_path.is_dir():
        raise ValueError(f"Hartree cube source is a directory, not a file: {source_path}")
    suffix = source_path.suffix.lower()
    from .out_h5 import is_linak_out_hdf5

    if suffix == ".cube" or is_linak_cube_hdf5(source_path) or is_linak_out_hdf5(source_path):
        return source_path
    raise ValueError(f"Hartree cube source must be a .cube, .cube.h5, or .out.h5 file: {source_path}")


def _cube_profile_payload(dataset: CubeDataset) -> dict[str, Any]:
    return {
        "datasets": {
            "origin_bohr": np.asarray(dataset.origin_bohr, dtype=float),
            "grid_counts_signed": np.asarray(dataset.grid_counts_signed, dtype=int),
            "grid_vectors_bohr": np.asarray(dataset.grid_vectors_bohr, dtype=float),
            "atom_numbers": np.asarray(dataset.atom_numbers, dtype=int),
            "atom_charges": np.asarray(dataset.atom_charges, dtype=float),
            "atom_positions_bohr": np.asarray(dataset.atom_positions_bohr, dtype=float),
            "values": np.asarray(dataset.values, dtype=float),
        },
        "metadata": {
            "analysis": "cube",
            "comment_1": dataset.comment_1,
            "comment_2": dataset.comment_2,
            "source_path": dataset.source_path,
            "source_name": dataset.source_name,
            "source_file_type": dataset.source_file_type,
            "source_profile_index": dataset.source_profile_index,
            "natoms_signed": int(dataset.natoms_signed),
        },
    }


def parse_cube_file(path: str | Path) -> CubeDataset:
    cube_path = Path(path).expanduser().resolve()
    with cube_path.open("r", encoding="utf-8", errors="replace") as handle:
        comment_1 = handle.readline()
        comment_2 = handle.readline()
        if not comment_1 or not comment_2:
            raise ValueError(f"Cube file ended unexpectedly while reading comments: {cube_path}")
        origin_line = handle.readline()
        if not origin_line:
            raise ValueError(f"Cube file ended unexpectedly: {cube_path}")
        origin_tokens = origin_line.split()
        if len(origin_tokens) < 4:
            raise ValueError(f"Invalid cube header line (natoms/origin) in '{cube_path}'.")
        natoms_signed = int(origin_tokens[0])
        origin_bohr = np.asarray(
            [float(origin_tokens[1]), float(origin_tokens[2]), float(origin_tokens[3])],
            dtype=float,
        )

        grid_counts_signed = np.empty(3, dtype=int)
        grid_vectors_bohr = np.empty((3, 3), dtype=float)
        for axis_index in range(3):
            line = handle.readline()
            if not line:
                raise ValueError(f"Cube file ended unexpectedly while reading grid: {cube_path}")
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError(f"Invalid cube grid line in '{cube_path}'.")
            grid_counts_signed[axis_index] = int(tokens[0])
            grid_vectors_bohr[axis_index] = np.asarray(
                [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                dtype=float,
            )

        natoms = abs(int(natoms_signed))
        atom_numbers = np.empty(natoms, dtype=int)
        atom_charges = np.empty(natoms, dtype=float)
        atom_positions_bohr = np.empty((natoms, 3), dtype=float)
        for atom_index in range(natoms):
            line = handle.readline()
            if not line:
                raise ValueError(
                    f"Cube file ended unexpectedly while reading atom list: {cube_path}"
                )
            tokens = line.split()
            if len(tokens) < 5:
                raise ValueError(f"Invalid atom line in cube file '{cube_path}'.")
            atom_numbers[atom_index] = int(round(float(tokens[0])))
            atom_charges[atom_index] = float(tokens[1])
            atom_positions_bohr[atom_index] = np.asarray(
                [float(tokens[2]), float(tokens[3]), float(tokens[4])],
                dtype=float,
            )
        payload = handle.read()

    if "D" in payload or "d" in payload:
        payload = payload.replace("D", "E").replace("d", "E")
    grid_shape = tuple(int(abs(count)) for count in grid_counts_signed.tolist())
    n_expected = int(np.prod(grid_shape, dtype=int))
    values = np.fromstring(payload, sep=" ", dtype=float, count=n_expected)
    if values.size != n_expected:
        raise ValueError(
            f"Cube data length mismatch in '{cube_path}'. Expected {n_expected}, got {values.size}."
        )
    return CubeDataset(
        comment_1=comment_1.rstrip("\n"),
        comment_2=comment_2.rstrip("\n"),
        natoms_signed=int(natoms_signed),
        origin_bohr=origin_bohr,
        grid_counts_signed=grid_counts_signed,
        grid_vectors_bohr=grid_vectors_bohr,
        atom_numbers=atom_numbers,
        atom_charges=atom_charges,
        atom_positions_bohr=atom_positions_bohr,
        values=values.reshape(grid_shape, order="C"),
        source_path=str(cube_path),
        source_name=cube_path.name,
        source_file_type="cube_file",
        source_profile_index=0,
    )


def write_cube_file(dataset: CubeDataset, output: str | Path) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{dataset.comment_1}\n")
        handle.write(f"{dataset.comment_2}\n")
        handle.write(
            f"{int(dataset.natoms_signed)} "
            f"{float(dataset.origin_bohr[0]):.8f} {float(dataset.origin_bohr[1]):.8f} {float(dataset.origin_bohr[2]):.8f}\n"
        )
        for count, vector in zip(dataset.grid_counts_signed.tolist(), dataset.grid_vectors_bohr):
            handle.write(
                f"{int(count)} "
                f"{float(vector[0]):.8f} {float(vector[1]):.8f} {float(vector[2]):.8f}\n"
            )
        for atomic_number, charge, position in zip(
            dataset.atom_numbers,
            dataset.atom_charges,
            dataset.atom_positions_bohr,
        ):
            handle.write(
                f"{int(atomic_number)} {float(charge):.8f} "
                f"{float(position[0]):.8f} {float(position[1]):.8f} {float(position[2]):.8f}\n"
            )
        flat_values = np.asarray(dataset.values, dtype=float).reshape(-1, order="C")
        for index, value in enumerate(flat_values, start=1):
            handle.write(f" {float(value): .8E}")
            if index % 6 == 0:
                handle.write("\n")
        if flat_values.size and flat_values.size % 6 != 0:
            handle.write("\n")
    return output_path


def save_cube_datasets(
    datasets: Sequence[CubeDataset],
    output: str | Path,
    *,
    additional_metadata: dict[str, Any] | None = None,
) -> Path:
    if not datasets:
        raise ValueError("At least one cube dataset is required.")
    return write_profile_collection(
        output,
        analysis="cube",
        profiles=[_cube_profile_payload(dataset) for dataset in datasets],
        metadata=dict(additional_metadata or {}),
    )


def load_cube_datasets(path: str | Path) -> list[CubeDataset]:
    source_path = Path(path).expanduser().resolve()
    payloads = read_linak_hdf5_profiles(source_path, expected_analysis="cube")
    datasets: list[CubeDataset] = []
    for datasets_payload, metadata in payloads:
        required = (
            "origin_bohr",
            "grid_counts_signed",
            "grid_vectors_bohr",
            "atom_numbers",
            "atom_charges",
            "atom_positions_bohr",
            "values",
        )
        missing = [name for name in required if name not in datasets_payload]
        if missing:
            raise ValueError(
                f"Cube HDF5 '{source_path}' is missing required dataset(s): {', '.join(missing)}."
            )
        datasets.append(
            CubeDataset(
                comment_1=str(metadata.get("comment_1", "LiNaK cube HDF5")),
                comment_2=str(metadata.get("comment_2", "Generated by LiNaK conversion")),
                natoms_signed=int(metadata.get("natoms_signed", np.asarray(datasets_payload["atom_numbers"]).shape[0])),
                origin_bohr=np.asarray(datasets_payload["origin_bohr"], dtype=float),
                grid_counts_signed=np.asarray(datasets_payload["grid_counts_signed"], dtype=int),
                grid_vectors_bohr=np.asarray(datasets_payload["grid_vectors_bohr"], dtype=float),
                atom_numbers=np.asarray(datasets_payload["atom_numbers"], dtype=int),
                atom_charges=np.asarray(datasets_payload["atom_charges"], dtype=float),
                atom_positions_bohr=np.asarray(datasets_payload["atom_positions_bohr"], dtype=float),
                values=np.asarray(datasets_payload["values"], dtype=float),
                source_path=(
                    str(metadata.get("source_path")).strip()
                    if metadata.get("source_path") not in (None, "")
                    else None
                ),
                source_name=(
                    str(metadata.get("source_name")).strip()
                    if metadata.get("source_name") not in (None, "")
                    else None
                ),
                source_file_type=(
                    str(metadata.get("source_file_type")).strip()
                    if metadata.get("source_file_type") not in (None, "")
                    else "cube_hdf5"
                ),
                source_profile_index=(
                    int(metadata.get("source_profile_index"))
                    if metadata.get("source_profile_index") is not None
                    else int(metadata.get("profile_index", 0))
                ),
            )
        )
    return datasets


def read_cube_sources(path: str | Path) -> list[CubeDataset]:
    source_path = validate_cube_source(path)
    from .out_h5 import is_linak_out_hdf5, read_out_h5_cube_datasets

    if is_linak_out_hdf5(source_path):
        return read_out_h5_cube_datasets(source_path)
    if is_linak_cube_hdf5(source_path):
        return load_cube_datasets(source_path)
    return [parse_cube_file(source_path)]
