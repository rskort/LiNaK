"""Persistent plot settings stored inside LiNaK HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..storage.hdf5_utils import decode_hdf5_string, hdf5_string_dtype, require_h5py

_PRIVATE_GROUP = "_linak"
_SETTINGS_GROUP = "plot_settings"
_SCHEMA_VERSION = 1
_SCHEMA_ATTR = "schema_version"
_UPDATED_ATTR = "updated_utc"
_PROFILES_DATASET = "profiles_json"
_NAMED_PROFILE_STORE_SENTINEL = "__linak_named_plot_profiles__"
_NAMED_PROFILE_STORE_VERSION = 1
_NAMED_PROFILE_STORE_ACTIVE = "active_profile"
_NAMED_PROFILE_STORE_PROFILES = "profiles"
DEFAULT_PLOT_PROFILE_NAME = "Default"


@dataclass(frozen=True)
class PlotProfileStore:
    """Named settings profiles for one analysis key."""

    active_profile: str | None
    profiles: dict[str, dict[str, Any]]


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


def _normalize_profile_name(name: str | None) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Plot profile name cannot be empty.")
    return normalized


def _is_named_profile_store(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get(_NAMED_PROFILE_STORE_SENTINEL) == _NAMED_PROFILE_STORE_VERSION
    )


def _coerce_settings_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): _json_ready(item) for key, item in value.items()}


def _first_profile_name(profiles: dict[str, dict[str, Any]]) -> str | None:
    if not profiles:
        return None
    if DEFAULT_PLOT_PROFILE_NAME in profiles:
        return DEFAULT_PLOT_PROFILE_NAME
    return next(iter(profiles))


def _coerce_profile_store(value: Any) -> PlotProfileStore | None:
    settings = _coerce_settings_dict(value)
    if settings is None:
        return None

    if not _is_named_profile_store(settings):
        return PlotProfileStore(
            active_profile=DEFAULT_PLOT_PROFILE_NAME,
            profiles={DEFAULT_PLOT_PROFILE_NAME: settings},
        )

    raw_profiles = settings.get(_NAMED_PROFILE_STORE_PROFILES)
    if not isinstance(raw_profiles, dict):
        return None

    profiles: dict[str, dict[str, Any]] = {}
    for raw_name, raw_settings in raw_profiles.items():
        name = _normalize_profile_name(str(raw_name))
        settings_dict = _coerce_settings_dict(raw_settings)
        if settings_dict is None:
            continue
        profiles[name] = settings_dict

    if not profiles:
        return None

    raw_active = settings.get(_NAMED_PROFILE_STORE_ACTIVE)
    active_profile = None
    if raw_active is not None:
        candidate = _normalize_profile_name(str(raw_active))
        if candidate in profiles:
            active_profile = candidate
    if active_profile is None:
        active_profile = _first_profile_name(profiles)
    return PlotProfileStore(active_profile=active_profile, profiles=profiles)


def _serialize_profile_store(
    store: PlotProfileStore,
    *,
    prefer_legacy_single: bool,
) -> dict[str, Any]:
    if prefer_legacy_single:
        legacy = store.profiles.get(DEFAULT_PLOT_PROFILE_NAME)
        if (
            legacy is not None
            and store.active_profile == DEFAULT_PLOT_PROFILE_NAME
            and len(store.profiles) == 1
        ):
            return _json_ready(legacy)
    return {
        _NAMED_PROFILE_STORE_SENTINEL: _NAMED_PROFILE_STORE_VERSION,
        _NAMED_PROFILE_STORE_ACTIVE: store.active_profile,
        _NAMED_PROFILE_STORE_PROFILES: _json_ready(store.profiles),
    }


def _read_profiles_map_with_store(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, PlotProfileStore]]:
    raw_profiles = read_plot_profiles_raw(path)
    stores: dict[str, PlotProfileStore] = {}
    for key, value in raw_profiles.items():
        store = _coerce_profile_store(value)
        if store is not None:
            stores[str(key)] = store
    return raw_profiles, stores


def _coerce_single_profile_store(value: Any) -> PlotProfileStore | None:
    store = _coerce_profile_store(value)
    if store is None or store.active_profile is None:
        return None
    active_settings = store.profiles.get(store.active_profile)
    if not isinstance(active_settings, dict):
        return None
    return PlotProfileStore(
        active_profile=DEFAULT_PLOT_PROFILE_NAME,
        profiles={DEFAULT_PLOT_PROFILE_NAME: dict(active_settings)},
    )


def default_plot_profile_name() -> str:
    """Return the conventional default saved-profile name."""
    return DEFAULT_PLOT_PROFILE_NAME


def is_combined_plot_settings_source(path: str | Path) -> bool:
    """Return whether an HDF5 source should behave like one combined plot document."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        return False
    try:
        with h5py.File(source_path, "r") as handle:
            raw_metadata = handle.attrs.get("metadata_json", "{}")
    except OSError:
        return False

    decoded = decode_hdf5_string(raw_metadata).strip()
    if not decoded:
        return False
    try:
        metadata = json.loads(decoded)
    except json.JSONDecodeError:
        return False
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("combined"))


def supports_named_plot_profiles(path: str | Path) -> bool:
    """Return whether an HDF5 source supports multiple named plot-setting profiles."""
    return True


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
    profiles: dict[str, Any] = {}
    for key, store in read_plot_profile_stores(path).items():
        if store.active_profile is None:
            continue
        profiles[key] = dict(store.profiles[store.active_profile])
    return profiles


def read_plot_profiles_raw(path: str | Path) -> dict[str, Any]:
    """Return the raw persisted plot-settings payload map from an HDF5 file."""
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


def read_plot_profile_stores(path: str | Path) -> dict[str, PlotProfileStore]:
    """Return every saved plot-settings store keyed by analysis profile key."""
    _raw, stores = _read_profiles_map_with_store(path)
    return stores


def read_plot_profile_store(path: str | Path, profile_key: str) -> PlotProfileStore | None:
    """Return the named-profile store for one analysis key, if present."""
    stores = read_plot_profile_stores(path)
    return stores.get(str(profile_key))


def read_plot_profile_names(path: str | Path, profile_key: str) -> list[str]:
    """Return saved profile names for one analysis key."""
    store = read_plot_profile_store(path, profile_key)
    if store is None:
        return []
    return list(store.profiles.keys())


def read_active_plot_profile_name(path: str | Path, profile_key: str) -> str | None:
    """Return the active saved profile name for one analysis key."""
    store = read_plot_profile_store(path, profile_key)
    if store is None:
        return None
    return store.active_profile


def read_plot_profile(
    path: str | Path,
    profile_key: str,
    *,
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    """Return one persisted plot-setting profile by key and optional name."""
    store = read_plot_profile_store(path, profile_key)
    if store is None:
        return None
    selected_name = (
        _normalize_profile_name(profile_name) if profile_name is not None else store.active_profile
    )
    if selected_name is None:
        return None
    value = store.profiles.get(selected_name)
    if value is None:
        return None
    return dict(value)


def write_plot_profile(
    path: str | Path,
    profile_key: str,
    settings: dict[str, Any],
    *,
    profile_name: str | None = None,
    set_active: bool = True,
) -> None:
    """Write or replace one plot-setting profile in an HDF5 file."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")
    settings_dict = _coerce_settings_dict(settings)
    if settings_dict is None:
        raise ValueError("Plot settings payload must be a JSON-like object.")
    with h5py.File(source_path, "r+") as handle:
        group = _open_settings_group(handle, create=True)
        assert group is not None
        profiles = _read_profiles_map(group)
        key = str(profile_key)
        existing_raw = profiles.get(key)
        existing_store = _coerce_profile_store(existing_raw)

        if profile_name is None and existing_raw is None:
            profiles[key] = settings_dict
            _write_profiles_map(group, profiles)
            return
        if (
            profile_name is None
            and existing_raw is not None
            and not _is_named_profile_store(existing_raw)
        ):
            profiles[key] = settings_dict
            _write_profiles_map(group, profiles)
            return

        target_name = (
            _normalize_profile_name(profile_name)
            if profile_name is not None
            else (
                existing_store.active_profile
                if existing_store is not None and existing_store.active_profile is not None
                else DEFAULT_PLOT_PROFILE_NAME
            )
        )

        if existing_store is None:
            new_store = PlotProfileStore(
                active_profile=target_name,
                profiles={target_name: settings_dict},
            )
        else:
            updated_profiles = dict(existing_store.profiles)
            updated_profiles[target_name] = settings_dict
            active_profile = (
                target_name if set_active else (existing_store.active_profile or target_name)
            )
            new_store = PlotProfileStore(active_profile=active_profile, profiles=updated_profiles)

        profiles[key] = _serialize_profile_store(
            new_store,
            prefer_legacy_single=(
                profile_name is None
                and existing_raw is not None
                and not _is_named_profile_store(existing_raw)
            ),
        )
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


def delete_named_plot_profile(
    path: str | Path,
    profile_key: str,
    profile_name: str,
) -> tuple[bool, str | None]:
    """Delete one named plot profile and return ``(removed, active_profile)``."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")
    target_name = _normalize_profile_name(profile_name)

    with h5py.File(source_path, "r+") as handle:
        group = _open_settings_group(handle, create=False)
        if group is None:
            return False, None
        profiles = _read_profiles_map(group)
        key = str(profile_key)
        store = _coerce_profile_store(profiles.get(key))
        if store is None or target_name not in store.profiles:
            return False, store.active_profile if store is not None else None

        updated_profiles = dict(store.profiles)
        del updated_profiles[target_name]
        if not updated_profiles:
            del profiles[key]
            _write_profiles_map(group, profiles)
            return True, None

        active_profile = store.active_profile
        if active_profile == target_name:
            active_profile = _first_profile_name(updated_profiles)
        assert active_profile is not None
        new_store = PlotProfileStore(active_profile=active_profile, profiles=updated_profiles)
        profiles[key] = _serialize_profile_store(
            new_store,
            prefer_legacy_single=True,
        )
        _write_profiles_map(group, profiles)
        return True, active_profile


def set_active_plot_profile(path: str | Path, profile_key: str, profile_name: str) -> None:
    """Set the active named plot profile for one analysis key."""
    require_h5py()
    import h5py

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")
    target_name = _normalize_profile_name(profile_name)

    with h5py.File(source_path, "r+") as handle:
        group = _open_settings_group(handle, create=False)
        if group is None:
            raise ValueError(
                f"No plot-setting profile store '{profile_key}' found in '{source_path}'."
            )
        profiles = _read_profiles_map(group)
        key = str(profile_key)
        raw = profiles.get(key)
        store = _coerce_profile_store(raw)
        if store is None or target_name not in store.profiles:
            raise ValueError(
                f"No named plot profile '{target_name}' found for '{key}' in '{source_path}'."
            )
        updated_store = PlotProfileStore(
            active_profile=target_name,
            profiles=dict(store.profiles),
        )
        profiles[key] = _serialize_profile_store(
            updated_store,
            prefer_legacy_single=(raw is not None and not _is_named_profile_store(raw)),
        )
        _write_profiles_map(group, profiles)


def copy_plot_profile(
    source: str | Path,
    target: str | Path,
    *,
    source_key: str,
    target_key: str | None = None,
    source_name: str | None = None,
    target_name: str | None = None,
) -> None:
    """Copy one profile from a source HDF5 file into a target HDF5 file."""
    source_path = Path(source).expanduser().resolve()
    resolved_target_key = target_key or source_key
    source_supports_named = supports_named_plot_profiles(source_path)
    target_supports_named = supports_named_plot_profiles(target)

    if (
        source_name is None
        and target_name is None
        and source_supports_named
        and target_supports_named
    ):
        store = read_plot_profile_store(source, source_key)
        if store is None:
            raise ValueError(f"No plot-setting profile '{source_key}' found in '{source_path}'.")
        for name, settings in store.profiles.items():
            write_plot_profile(
                target,
                resolved_target_key,
                settings,
                profile_name=name,
                set_active=(name == store.active_profile),
            )
        return

    profile = read_plot_profile(source, source_key, profile_name=source_name)
    if profile is None:
        raise ValueError(
            f"No named plot-setting profile '{source_name or source_key}' found in '{source_path}'."
        )
    write_plot_profile(
        target,
        resolved_target_key,
        profile,
        profile_name=(
            None
            if not target_supports_named
            else (target_name or source_name or DEFAULT_PLOT_PROFILE_NAME)
        ),
    )
