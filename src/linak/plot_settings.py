"""Persistent plot settings stored inside LiNaK HDF5 files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hdf5_utils import decode_hdf5_string, hdf5_string_dtype, require_h5py

_PRIVATE_GROUP = "_linak"
_SETTINGS_GROUP = "plot_settings"
_SCHEMA_VERSION = 1
_SCHEMA_ATTR = "schema_version"
_UPDATED_ATTR = "updated_utc"
_PROFILES_DATASET = "profiles_json"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _read_profiles_map(group: Any) -> dict[str, Any]:
    if _PROFILES_DATASET not in group:
        return {}
    dataset = group[_PROFILES_DATASET]
    raw = dataset[()]
    decoded = decode_hdf5_string(raw)
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): value for key, value in parsed.items()}


def _write_profiles_map(group: Any, profiles: dict[str, Any]) -> None:
    payload = json.dumps(_json_ready(profiles), sort_keys=True)
    if _PROFILES_DATASET in group:
        del group[_PROFILES_DATASET]
    group.create_dataset(
        _PROFILES_DATASET,
        data=np.asarray(payload, dtype=object),
        dtype=hdf5_string_dtype(),
    )
    group.attrs[_SCHEMA_ATTR] = _SCHEMA_VERSION
    group.attrs[_UPDATED_ATTR] = _now_utc_iso()


def _open_settings_group(handle: Any, *, create: bool) -> Any | None:
    private_group = handle.get(_PRIVATE_GROUP)
    if private_group is None:
        if not create:
            return None
        private_group = handle.require_group(_PRIVATE_GROUP)

    settings_group = private_group.get(_SETTINGS_GROUP)
    if settings_group is None:
        if not create:
            return None
        settings_group = private_group.require_group(_SETTINGS_GROUP)
    return settings_group


def read_hdf5_analysis(path: str | Path) -> str | None:
    """Return the normalized ``analysis`` attribute from an HDF5 file, if present."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        return None
    try:
        with h5py.File(source_path, "r") as handle:
            raw = handle.attrs.get("analysis")
            if raw is None:
                return None
            value = decode_hdf5_string(raw).strip().lower()
            return value or None
    except OSError:
        return None


def read_plot_profiles(path: str | Path) -> dict[str, Any]:
    """Return all persisted plot-setting profiles from an HDF5 file."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")

    with h5py.File(source_path, "r") as handle:
        group = _open_settings_group(handle, create=False)
        if group is None:
            return {}
        return _read_profiles_map(group)


def read_plot_profile(path: str | Path, profile_key: str) -> dict[str, Any] | None:
    """Return one persisted plot-setting profile by key, if present."""
    profiles = read_plot_profiles(path)
    value = profiles.get(profile_key)
    if not isinstance(value, dict):
        return None
    return value


def write_plot_profile(path: str | Path, profile_key: str, settings: dict[str, Any]) -> None:
    """Write or replace one plot-setting profile in an HDF5 file."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")

    with h5py.File(source_path, "r+") as handle:
        group = _open_settings_group(handle, create=True)
        assert group is not None
        profiles = _read_profiles_map(group)
        profiles[str(profile_key)] = _json_ready(settings)
        _write_profiles_map(group, profiles)


def delete_plot_profile(path: str | Path, profile_key: str) -> bool:
    """Delete one plot-setting profile. Returns ``True`` if it existed."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")

    with h5py.File(source_path, "r+") as handle:
        group = _open_settings_group(handle, create=False)
        if group is None:
            return False
        profiles = _read_profiles_map(group)
        if str(profile_key) not in profiles:
            return False
        del profiles[str(profile_key)]
        _write_profiles_map(group, profiles)
        return True


def copy_plot_profile(
    source: str | Path,
    target: str | Path,
    *,
    source_key: str,
    target_key: str | None = None,
) -> None:
    """Copy one profile from a source HDF5 file into a target HDF5 file."""
    profile = read_plot_profile(source, source_key)
    if profile is None:
        source_path = Path(source).expanduser().resolve()
        raise ValueError(
            f"No plot-setting profile '{source_key}' found in '{source_path}'."
        )
    write_plot_profile(target, target_key or source_key, profile)

