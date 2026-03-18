"""Shared HDF5 helpers for LiNaK analysis files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__

try:  # pragma: no cover - exercised in environments missing optional dependency.
    _H5PY_IMPORT_ERROR: ModuleNotFoundError | None = None
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc

LINAK_HDF5_FORMAT = "linak-hdf5"
LINAK_HDF5_VERSION = 1
HDF5_SUFFIXES = (".h5", ".hdf5")
_COLLECTION_GROUP = "profiles"


def require_h5py() -> None:
    """Raise a user-facing error if h5py is unavailable."""
    if h5py is None:  # pragma: no cover - environment dependent.
        raise ValueError(
            "HDF5 support requires 'h5py'. Install it and rerun (for example: pip install h5py)."
        ) from _H5PY_IMPORT_ERROR


def is_hdf5_path(path: str | Path) -> bool:
    """Return whether the path uses an HDF5 extension."""
    return Path(path).suffix.lower() in HDF5_SUFFIXES


def resolve_hdf5_output_path(path: str | Path) -> Path:
    """Resolve output path and enforce .h5/.hdf5 suffixes."""
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        return resolved
    if not suffix:
        return resolved.with_suffix(".h5")
    raise ValueError(
        f"HDF5 output path must use one of {', '.join(HDF5_SUFFIXES)} (got '{resolved.name}')."
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _decode_metadata_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items()}
    return {"value": parsed}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_linak_hdf5(
    output: str | Path,
    *,
    analysis: str,
    datasets: Mapping[str, np.ndarray | list[float] | tuple[float, ...] | None],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a LiNaK HDF5 analysis file."""
    require_h5py()
    output_path = resolve_hdf5_output_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(_json_ready(dict(metadata or {})), sort_keys=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["linak_format"] = LINAK_HDF5_FORMAT
        handle.attrs["linak_format_version"] = LINAK_HDF5_VERSION
        handle.attrs["analysis"] = str(analysis)
        handle.attrs["created_utc"] = _now_utc_iso()
        handle.attrs["linak_version"] = __version__
        handle.attrs["metadata_json"] = metadata_json

        for name, values in datasets.items():
            if values is None:
                continue
            array = np.asarray(values)
            use_compression = array.ndim > 0 and array.size > 0
            if use_compression:
                handle.create_dataset(name, data=array, compression="gzip", shuffle=True)
            else:
                handle.create_dataset(name, data=array)
    return output_path


def read_linak_hdf5_profiles(
    path: str | Path,
    *,
    expected_analysis: str | None = None,
) -> list[tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Read one or more LiNaK HDF5 analysis profiles.

    Files written by :func:`write_linak_hdf5` return exactly one profile.
    Files written by :func:`write_linak_hdf5_profile_collection` can return multiple profiles.
    """
    require_h5py()
    hdf5_path = Path(path).expanduser().resolve()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as handle:
        file_format = str(handle.attrs.get("linak_format", ""))
        if file_format != LINAK_HDF5_FORMAT:
            raise ValueError(f"File is not a LiNaK HDF5 analysis file: {hdf5_path}")

        analysis = str(handle.attrs.get("analysis", ""))
        if expected_analysis is not None and analysis != expected_analysis:
            raise ValueError(
                f"HDF5 analysis mismatch for '{hdf5_path}': expected '{expected_analysis}', got '{analysis}'."
            )

        root_metadata = _decode_metadata_json(handle.attrs.get("metadata_json", "{}"))
        root_metadata.setdefault("analysis", analysis)

        collection = handle.get(_COLLECTION_GROUP)
        if collection is not None and hasattr(collection, "items"):
            member_items = [
                (str(name), node) for name, node in collection.items() if hasattr(node, "items")
            ]
            if member_items:

                def _member_sort_key(item: tuple[str, Any]) -> tuple[int, int | str]:
                    name = item[0]
                    return (0, int(name)) if name.isdigit() else (1, name)

                profiles: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
                for index, (_name, group) in enumerate(sorted(member_items, key=_member_sort_key)):
                    datasets = {
                        dataset_name: np.asarray(dataset)
                        for dataset_name, dataset in group.items()
                        if hasattr(dataset, "shape")
                    }
                    profile_metadata = dict(root_metadata)
                    profile_metadata.update(
                        _decode_metadata_json(group.attrs.get("metadata_json", "{}"))
                    )
                    profile_metadata.setdefault("analysis", analysis)
                    profile_metadata.setdefault("profile_index", index)
                    profiles.append((datasets, profile_metadata))
                return profiles

        datasets = {
            name: np.asarray(dataset)
            for name, dataset in handle.items()
            if hasattr(dataset, "shape")
        }
        return [(datasets, root_metadata)]


def read_linak_hdf5(
    path: str | Path,
    *,
    expected_analysis: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a LiNaK HDF5 file and return datasets with decoded metadata."""
    profiles = read_linak_hdf5_profiles(path, expected_analysis=expected_analysis)
    if not profiles:
        hdf5_path = Path(path).expanduser().resolve()
        raise ValueError(f"HDF5 file has no readable datasets: {hdf5_path}")
    return profiles[0]


def write_linak_hdf5_profile_collection(
    output: str | Path,
    *,
    analysis: str,
    profiles: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a LiNaK HDF5 file containing multiple analysis profiles."""
    require_h5py()
    if not profiles:
        raise ValueError("At least one profile is required when writing a profile collection.")

    output_path = resolve_hdf5_output_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root_metadata = dict(metadata or {})
    root_metadata["combined"] = True
    root_metadata["profile_count"] = len(profiles)

    with h5py.File(output_path, "w") as handle:
        handle.attrs["linak_format"] = LINAK_HDF5_FORMAT
        handle.attrs["linak_format_version"] = LINAK_HDF5_VERSION
        handle.attrs["analysis"] = str(analysis)
        handle.attrs["created_utc"] = _now_utc_iso()
        handle.attrs["linak_version"] = __version__
        handle.attrs["metadata_json"] = json.dumps(_json_ready(root_metadata), sort_keys=True)

        profiles_group = handle.require_group(_COLLECTION_GROUP)
        for index, payload in enumerate(profiles):
            datasets = payload.get("datasets")
            if not isinstance(datasets, Mapping):
                raise ValueError("Each profile payload must provide a 'datasets' mapping.")
            profile_metadata = payload.get("metadata")
            if profile_metadata is None:
                profile_metadata_map: dict[str, Any] = {}
            elif isinstance(profile_metadata, Mapping):
                profile_metadata_map = dict(profile_metadata)
            else:
                raise ValueError("Each profile payload metadata must be a mapping when provided.")
            profile_group = profiles_group.require_group(f"{index:04d}")
            profile_group.attrs["metadata_json"] = json.dumps(
                _json_ready(profile_metadata_map),
                sort_keys=True,
            )
            for name, values in datasets.items():
                if values is None:
                    continue
                array = np.asarray(values)
                use_compression = array.ndim > 0 and array.size > 0
                if use_compression:
                    profile_group.create_dataset(name, data=array, compression="gzip", shuffle=True)
                else:
                    profile_group.create_dataset(name, data=array)
    return output_path


def hdf5_string_dtype() -> Any:
    """Return UTF-8 string dtype for HDF5 datasets."""
    require_h5py()
    return h5py.string_dtype(encoding="utf-8")


def decode_hdf5_string(value: Any) -> str:
    """Decode HDF5 string scalars robustly."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
