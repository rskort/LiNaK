"""Utilities for robust tabular CSV handling used by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_DEFAULT_NA_TOKENS = (
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
)

_COMPARISON_OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le"}
_TEXT_OPERATORS = {"contains", "startswith", "endswith", "regex"}
_SET_OPERATORS = {"in", "not-in"}


@dataclass(frozen=True)
class CSVLoadConfig:
    """Read-time options for CSV loading."""

    delimiter: str | None = None
    encoding: str = "utf-8"


@dataclass(frozen=True)
class ColumnProfile:
    """Metadata summary for one DataFrame column."""

    name: str
    dtype: str
    rows: int
    non_null: int
    missing: int
    distinct: int
    numeric_ratio: float
    numeric_valid: int


def normalize_delimiter(value: str | None) -> str | None:
    """Normalize delimiter aliases used by the CLI."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized in {"tab", r"\t"}:
        return "\t"
    if normalized == "comma":
        return ","
    if normalized == "semicolon":
        return ";"
    if normalized in {"pipe", "bar"}:
        return "|"
    if len(value) != 1:
        raise ValueError(
            f"Unsupported delimiter '{value}'. Use one character or one of: auto, tab, comma, semicolon, pipe."
        )
    return value


def _normalize_headers(columns: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: dict[str, int] = {}
    for index, raw_name in enumerate(columns, start=1):
        name = str(raw_name).strip() if raw_name is not None else ""
        if not name:
            name = f"column_{index}"

        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        normalized.append(name)
    return normalized


def read_csv_frame(source: str | Path, *, config: CSVLoadConfig) -> tuple[pd.DataFrame, Path]:
    """Read CSV input and return a normalized DataFrame."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"CSV source is not a file: {source_path}")

    separator = normalize_delimiter(config.delimiter)
    read_kwargs = {
        "encoding": config.encoding,
        "na_values": list(_DEFAULT_NA_TOKENS),
        "keep_default_na": True,
    }
    if separator is None:
        read_kwargs["sep"] = None
        read_kwargs["engine"] = "python"
    else:
        read_kwargs["sep"] = separator

    try:
        frame = pd.read_csv(source_path, **read_kwargs)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode '{source_path}' with encoding '{config.encoding}'. "
            "Try --encoding utf-8-sig or latin-1."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file '{source_path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        hint = (
            f"Failed to parse '{source_path}'. "
            "If the delimiter is not comma-like, pass --delimiter explicitly."
        )
        raise ValueError(hint) from exc

    if frame.columns.size == 0:
        raise ValueError(f"CSV file '{source_path}' has no header columns.")

    normalized_columns = _normalize_headers(list(frame.columns))
    if normalized_columns != list(frame.columns):
        frame = frame.copy()
        frame.columns = normalized_columns

    return frame, source_path


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype("string"), errors="coerce")


def _numeric_valid_count(series: pd.Series) -> int:
    coerced = _coerce_numeric(series)
    return int(coerced.notna().sum())


def profile_columns(frame: pd.DataFrame) -> list[ColumnProfile]:
    """Build a rich summary for every column in the DataFrame."""
    profiles: list[ColumnProfile] = []
    rows = int(len(frame))
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        missing = rows - non_null
        distinct = int(series.nunique(dropna=True))
        numeric_valid = _numeric_valid_count(series)
        numeric_ratio = float(numeric_valid / non_null) if non_null else 0.0
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                rows=rows,
                non_null=non_null,
                missing=missing,
                distinct=distinct,
                numeric_ratio=numeric_ratio,
                numeric_valid=numeric_valid,
            )
        )
    return profiles


def infer_numeric_columns(frame: pd.DataFrame, *, min_ratio: float = 0.95) -> list[str]:
    """Return columns that are fully or almost fully numeric-like."""
    numeric: list[str] = []
    for profile in profile_columns(frame):
        if profile.non_null == 0:
            continue
        if profile.numeric_ratio >= min_ratio:
            numeric.append(profile.name)
    return numeric


def infer_sort_mode(series: pd.Series, *, mode: str) -> str:
    """Resolve sorting mode from explicit/auto configuration."""
    if mode not in {"auto", "numeric", "string"}:
        raise ValueError(f"Unsupported sort mode '{mode}'.")
    if mode != "auto":
        return mode
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = int(series.notna().sum())
    if non_null == 0:
        return "string"
    numeric_ratio = _numeric_valid_count(series) / non_null
    return "numeric" if numeric_ratio >= 0.95 else "string"


def format_profiles_table(profiles: list[ColumnProfile]) -> str:
    """Render column profile rows as a compact fixed-width table."""
    if not profiles:
        return "No columns available."

    rows = [
        (
            profile.name,
            profile.dtype,
            str(profile.non_null),
            str(profile.missing),
            str(profile.distinct),
            f"{profile.numeric_ratio:.2f}",
        )
        for profile in profiles
    ]
    headers = ("column", "dtype", "non-null", "missing", "distinct", "numeric-ratio")
    widths = [
        max(len(headers[index]), max(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def compute_column_statistics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    """Compute rich numeric or categorical statistics for one column."""
    if column not in frame.columns:
        raise ValueError(f"Unknown column '{column}'.")

    series = frame[column]
    non_null = int(series.notna().sum())
    missing = int(series.isna().sum())
    distinct = int(series.nunique(dropna=True))
    base: dict[str, Any] = {
        "column": column,
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "non_null": non_null,
        "missing": missing,
        "distinct": distinct,
    }

    numeric = _coerce_numeric(series)
    numeric_valid = int(numeric.notna().sum())
    numeric_ratio = float(numeric_valid / non_null) if non_null else 0.0
    base["numeric_ratio"] = numeric_ratio

    if numeric_valid > 0 and numeric_ratio >= 0.95:
        clean = numeric.dropna()
        q05, q25, q50, q75, q95 = np.percentile(clean.to_numpy(), [5, 25, 50, 75, 95])
        std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
        base.update(
            {
                "kind": "numeric",
                "count": int(len(clean)),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "mean": float(clean.mean()),
                "median": float(q50),
                "std": std,
                "var": float(clean.var(ddof=1)) if len(clean) > 1 else 0.0,
                "sum": float(clean.sum()),
                "q05": float(q05),
                "q25": float(q25),
                "q75": float(q75),
                "q95": float(q95),
                "iqr": float(q75 - q25),
            }
        )
        return base

    text = series.dropna().astype("string")
    value_counts = text.value_counts(dropna=True)
    mode_value = None
    mode_count = 0
    if not value_counts.empty:
        mode_value = str(value_counts.index[0])
        mode_count = int(value_counts.iloc[0])

    base.update(
        {
            "kind": "categorical",
            "count": int(non_null),
            "mode": mode_value,
            "mode_count": mode_count,
            "top_values": [
                (str(index), int(count)) for index, count in value_counts.head(5).items()
            ],
        }
    )
    return base


def sort_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    descending: bool,
    na_position: str,
    mode: str,
) -> pd.DataFrame:
    """Sort by one or more columns with numeric/string auto inference."""
    if na_position not in {"first", "last"}:
        raise ValueError("na_position must be either 'first' or 'last'.")
    if not columns:
        raise ValueError("At least one sort column is required.")

    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"Unknown column '{column}'.")

    temp = frame.copy()
    sort_columns: list[str] = []
    for index, column in enumerate(columns):
        resolved_mode = infer_sort_mode(temp[column], mode=mode)
        shadow_name = f"__linak_sort_{index}"
        if resolved_mode == "numeric":
            temp[shadow_name] = _coerce_numeric(temp[column])
        else:
            temp[shadow_name] = temp[column].astype("string").str.casefold()
        sort_columns.append(shadow_name)

    sorted_frame = temp.sort_values(
        by=sort_columns,
        ascending=not descending,
        na_position=na_position,
        kind="mergesort",
    )
    return sorted_frame[frame.columns]


def _parse_set_values(raw_value: str) -> set[str]:
    return {token.strip() for token in raw_value.split(",") if token.strip()}


def filter_frame(
    frame: pd.DataFrame,
    *,
    column: str,
    operator: str,
    value: str,
    case_sensitive: bool,
    invert: bool,
) -> pd.DataFrame:
    """Filter DataFrame rows using numeric/text operators."""
    if column not in frame.columns:
        raise ValueError(f"Unknown column '{column}'.")

    normalized_op = operator.strip().lower()
    if normalized_op not in (_COMPARISON_OPERATORS | _TEXT_OPERATORS | _SET_OPERATORS):
        raise ValueError(
            f"Unsupported operator '{operator}'. "
            "Use: eq, ne, gt, ge, lt, le, contains, startswith, endswith, regex, in, not-in."
        )

    source = frame[column]
    if normalized_op in {"eq", "ne"}:
        numeric = _coerce_numeric(source)
        non_null = int(source.notna().sum())
        numeric_ratio = float(numeric.notna().sum() / non_null) if non_null else 0.0
        numeric_candidate = pd.api.types.is_numeric_dtype(source) or numeric_ratio >= 0.95

        target_numeric: float | None = None
        try:
            target_numeric = float(value)
        except ValueError:
            target_numeric = None

        if numeric_candidate and target_numeric is not None:
            if normalized_op == "eq":
                mask = numeric.eq(target_numeric).fillna(False)
            else:
                mask = numeric.ne(target_numeric).fillna(False)
        else:
            text = source.astype("string")
            observed = text if case_sensitive else text.str.casefold()
            target = value if case_sensitive else value.casefold()
            if normalized_op == "eq":
                mask = observed.eq(target).fillna(False)
            else:
                mask = observed.ne(target).fillna(False)
    elif normalized_op in {"gt", "ge", "lt", "le"}:
        numeric = _coerce_numeric(source)
        try:
            target_numeric = float(value)
        except ValueError as exc:
            raise ValueError(f"Operator '{normalized_op}' requires a numeric --value.") from exc

        comparator_map = {
            "gt": numeric.gt(target_numeric),
            "ge": numeric.ge(target_numeric),
            "lt": numeric.lt(target_numeric),
            "le": numeric.le(target_numeric),
        }
        mask = comparator_map[normalized_op]
        mask = mask.fillna(False)
    elif normalized_op in {"in", "not-in"}:
        values = _parse_set_values(value)
        if not values:
            raise ValueError("Operator 'in'/'not-in' requires a comma-separated --value list.")
        text = source.astype("string")
        transformed = text if case_sensitive else text.str.casefold()
        lookup = values if case_sensitive else {token.casefold() for token in values}
        mask = transformed.isin(lookup).fillna(False)
        if normalized_op == "not-in":
            mask = ~mask
    elif normalized_op == "regex":
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(value, flags=flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern '{value}': {exc}") from exc
        text = source.astype("string")
        mask = text.str.contains(pattern, na=False)
    else:
        text = source.astype("string")
        target = value if case_sensitive else value.casefold()
        observed = text if case_sensitive else text.str.casefold()
        if normalized_op == "contains":
            mask = observed.str.contains(re.escape(target), regex=True, na=False)
        elif normalized_op == "startswith":
            mask = observed.str.startswith(target, na=False)
        else:
            mask = observed.str.endswith(target, na=False)

    if invert:
        mask = ~mask
    return frame.loc[mask].copy()


def deduplicate_frame(
    frame: pd.DataFrame,
    *,
    subset: list[str] | None,
    keep: str,
) -> pd.DataFrame:
    """Return DataFrame with duplicates removed based on subset columns."""
    if keep not in {"first", "last", "none"}:
        raise ValueError("keep must be one of: first, last, none.")
    if subset is not None:
        unknown = [column for column in subset if column not in frame.columns]
        if unknown:
            raise ValueError(f"Unknown dedupe column(s): {', '.join(unknown)}")

    keep_arg: str | bool = keep
    if keep == "none":
        keep_arg = False
    return frame.drop_duplicates(subset=subset, keep=keep_arg)


def format_frame_preview(
    frame: pd.DataFrame,
    *,
    rows: int,
    tail: bool,
    show_index: bool,
) -> str:
    """Return a printable head/tail preview for the DataFrame."""
    if rows <= 0:
        raise ValueError("rows must be > 0.")
    sliced = frame.tail(rows) if tail else frame.head(rows)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        160,
        "display.max_colwidth",
        40,
    ):
        return sliced.to_string(index=show_index)


def write_frame_csv(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    delimiter: str | None,
    encoding: str,
) -> Path:
    """Write DataFrame to CSV and return resolved output path."""
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = normalize_delimiter(delimiter) or ","
    frame.to_csv(output_path, index=False, sep=sep, encoding=encoding)
    return output_path
