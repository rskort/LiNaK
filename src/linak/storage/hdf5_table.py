"""Tabular HDF5 helpers for CLI-style inspection and transformations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from .hdf5_utils import (
    LINAK_HDF5_FORMAT,
    LINAK_HDF5_VERSION,
    hdf5_string_dtype,
    require_h5py,
    resolve_hdf5_output_path,
)


@dataclass(frozen=True)
class HDF5TableInfo:
    """Metadata returned when reading a tabular HDF5 view."""

    source_path: Path
    analysis: str
    container: str
    row_count: int
    included_columns: tuple[str, ...]
    skipped_datasets: tuple[str, ...]
    linak_format: str
    linak_format_version: int | None
    created_utc: str | None
    linak_version: str | None
    metadata: dict[str, Any]


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_attr_value(value.item())
        return [_decode_attr_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_decode_attr_value(item) for item in value]
    return _decode_scalar(value)


def _decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="replace")
    if values.dtype.kind == "U":
        return values.astype(str)
    if values.dtype.kind != "O":
        return values

    decoded = [_decode_scalar(item) for item in values.tolist()]
    return np.asarray(decoded, dtype=object)


def _iter_first_level_datasets(group: Any) -> list[tuple[str, Any]]:
    return [
        (str(name), node)
        for name, node in group.items()
        if getattr(node, "shape", None) is not None
    ]


def _resolve_container(handle: Any, *, group: str | None) -> tuple[Any, str]:
    if group:
        normalized = group.strip().strip("/")
        if not normalized:
            return handle, "/"
        if normalized not in handle:
            raise ValueError(f"HDF5 group '{group}' was not found in '{handle.filename}'.")
        selected = handle[normalized]
        if not hasattr(selected, "items"):
            raise ValueError(f"HDF5 path '{group}' is not a group.")
        return selected, f"/{normalized}"

    if "records" in handle and hasattr(handle["records"], "items"):
        datasets = _iter_first_level_datasets(handle["records"])
        if datasets:
            return handle["records"], "/records"

    root_datasets = _iter_first_level_datasets(handle)
    if root_datasets:
        return handle, "/"

    for name, node in handle.items():
        if hasattr(node, "items"):
            datasets = _iter_first_level_datasets(node)
            if datasets:
                return node, f"/{name}"

    raise ValueError(
        f"No tabular datasets found in '{handle.filename}'. "
        "Provide --group to target a specific HDF5 group."
    )


def _row_count_from_datasets(datasets: list[tuple[str, np.ndarray]]) -> int:
    counter: Counter[int] = Counter()
    for _name, values in datasets:
        if values.ndim == 0:
            continue
        counter[int(values.shape[0])] += 1
    if not counter:
        raise ValueError("Selected HDF5 group has no row-like datasets.")
    return max(counter.items(), key=lambda item: (item[1], item[0]))[0]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_from_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    metadata_raw = attrs.get("metadata_json")
    if metadata_raw is None:
        return {}
    if not isinstance(metadata_raw, str):
        metadata_raw = str(metadata_raw)
    try:
        decoded = json.loads(metadata_raw)
    except json.JSONDecodeError:
        return {"_parse_error": "metadata_json is not valid JSON"}
    if isinstance(decoded, dict):
        return {str(key): value for key, value in decoded.items()}
    return {"value": decoded}


def _format_metadata_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        keys = [str(key) for key in value]
        preview = ", ".join(keys[:6])
        if len(keys) > 6:
            preview += ", ..."
        return f"object({len(keys)} keys: {preview})"
    if isinstance(value, list):
        if not value:
            return "[]"
        if len(value) <= 4 and all(isinstance(item, (str, int, float, bool)) for item in value):
            return "[" + ", ".join(str(item) for item in value) + "]"
        return f"list(len={len(value)})"
    return str(value)


def format_hdf5_metadata_overview(info: HDF5TableInfo) -> str:
    """Render a compact metadata summary suitable for CLI output."""
    format_label = info.linak_format or "unknown"
    if info.linak_format_version is not None:
        format_label = f"{format_label} v{info.linak_format_version}"

    lines = [
        "Metadata overview",
        f"  format         : {format_label}",
        f"  analysis       : {info.analysis or 'unknown'}",
        f"  created_utc    : {info.created_utc or 'unknown'}",
        f"  linak_version  : {info.linak_version or 'unknown'}",
        f"  selected group : {info.container}",
        f"  tabular rows   : {info.row_count}",
    ]

    if info.metadata:
        lines.append("  metadata_json")
        for key in sorted(info.metadata):
            lines.append(f"    {key}: {_format_metadata_value(info.metadata[key])}")

    if info.skipped_datasets:
        preview = ", ".join(info.skipped_datasets[:4])
        if len(info.skipped_datasets) > 4:
            preview += ", ..."
        lines.append(f"  skipped datasets: {preview}")

    return "\n".join(lines)


def read_hdf5_frame(
    source: str | Path,
    *,
    group: str | None = None,
) -> tuple[pd.DataFrame, HDF5TableInfo]:
    """Read HDF5 datasets into a tabular DataFrame view."""
    require_h5py()
    import h5py

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"HDF5 source is not a file: {source_path}")

    try:
        with h5py.File(source_path, "r") as handle:
            container, container_label = _resolve_container(handle, group=group)
            raw_items = _iter_first_level_datasets(container)
            if not raw_items:
                raise ValueError(f"No datasets found in group '{container_label}' for '{source_path}'.")

            attrs = {
                str(key): _decode_attr_value(value)
                for key, value in handle.attrs.items()
            }
            metadata = _metadata_from_attrs(attrs)

            datasets = [(name, np.asarray(dataset)) for name, dataset in raw_items]
            row_count = _row_count_from_datasets(datasets)

            frame_data: dict[str, np.ndarray] = {}
            skipped: list[str] = []
            for name, values in datasets:
                if values.ndim == 0:
                    skipped.append(f"{name} (scalar)")
                    continue
                if int(values.shape[0]) != row_count:
                    skipped.append(f"{name} (rows={values.shape[0]})")
                    continue

                if values.ndim == 1:
                    frame_data[name] = _decode_array(values)
                    continue

                flattened = values.reshape(row_count, -1)
                for index in range(flattened.shape[1]):
                    column_name = f"{name}[{index}]"
                    frame_data[column_name] = _decode_array(flattened[:, index])

            if not frame_data:
                raise ValueError(
                    f"Could not build a tabular view from '{source_path}'. "
                    "All datasets were non-row-aligned scalars or incompatible shapes."
                )

            frame = pd.DataFrame(frame_data)
            info = HDF5TableInfo(
                source_path=source_path,
                analysis=str(attrs.get("analysis", "")),
                container=container_label,
                row_count=row_count,
                included_columns=tuple(str(column) for column in frame.columns),
                skipped_datasets=tuple(skipped),
                linak_format=str(attrs.get("linak_format", "")),
                linak_format_version=_safe_int(attrs.get("linak_format_version")),
                created_utc=str(attrs.get("created_utc")) if attrs.get("created_utc") is not None else None,
                linak_version=str(attrs.get("linak_version")) if attrs.get("linak_version") is not None else None,
                metadata=metadata,
            )
    except OSError as exc:
        raise ValueError(f"Could not read HDF5 file '{source_path}': {exc}") from exc
    return frame, info


def _series_to_dataset_array(series: pd.Series) -> tuple[np.ndarray, Any | None]:
    if pd.api.types.is_bool_dtype(series) and not series.isna().any():
        return series.to_numpy(dtype=bool), None

    if pd.api.types.is_integer_dtype(series) and not series.isna().any():
        return series.to_numpy(dtype=np.int64), None

    if pd.api.types.is_float_dtype(series):
        return series.to_numpy(dtype=float), None

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().sum() / len(series)) if len(series) else 0.0
    if numeric_ratio >= 0.95:
        return numeric.to_numpy(dtype=float), None

    values = series.astype("string").fillna("").to_numpy(dtype=object)
    return values, hdf5_string_dtype()


def write_hdf5_frame(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    source_info: HDF5TableInfo | None = None,
) -> Path:
    """Persist a DataFrame as a LiNaK-compatible tabular HDF5 file."""
    require_h5py()
    import h5py

    output_path = resolve_hdf5_output_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "source": str(source_info.source_path) if source_info is not None else None,
        "source_analysis": source_info.analysis if source_info is not None else None,
        "source_container": source_info.container if source_info is not None else None,
    }

    with h5py.File(output_path, "w") as handle:
        handle.attrs["linak_format"] = LINAK_HDF5_FORMAT
        handle.attrs["linak_format_version"] = LINAK_HDF5_VERSION
        handle.attrs["analysis"] = "table"
        handle.attrs["created_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        handle.attrs["linak_version"] = __version__
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.attrs["columns_json"] = json.dumps(metadata["columns"])

        records = handle.require_group("records")
        for column in frame.columns:
            values, dtype = _series_to_dataset_array(frame[column])
            use_compression = values.ndim > 0 and values.size > 0
            kwargs: dict[str, Any] = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if use_compression:
                kwargs["compression"] = "gzip"
                kwargs["shuffle"] = True
            records.create_dataset(str(column), data=values, **kwargs)

    return output_path

