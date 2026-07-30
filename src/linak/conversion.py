"""Reusable file-family-aware conversion routing for LiNaK.

This module keeps source/target format detection and conversion dispatch out of
the CLI layer. The registry resolves one source family, one concrete target
file type, and the family-specific reader/writer path needed for execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import __version__
from .cube_io import (
    CubeDataset,
    is_linak_cube_hdf5,
    load_cube_datasets,
    parse_cube_file,
    save_cube_datasets,
    write_cube_file,
)
from .progress import ProgressBar

LOGGER = logging.getLogger(__name__)

FileFamily = Literal["trajectory", "cube"]


@dataclass(frozen=True)
class ConversionFileType:
    """Describe one concrete file type handled by the conversion layer."""

    id: str
    family: FileFamily
    label: str
    suffixes: tuple[str, ...]
    readable: bool = True
    writable: bool = True
    detector: Callable[[Path], bool] | None = None
    default_output_path_factory: Callable[[Path], Path] | None = None
    target_aliases: tuple[str, ...] = ()

    def matches(self, path: Path) -> bool:
        """Return whether this file type claims the given path."""

        if self.detector is not None:
            return bool(self.detector(path))
        path_text = str(path).lower()
        return any(path_text.endswith(suffix.lower()) for suffix in self.suffixes)

    def matches_output_path(self, path: Path) -> bool:
        """Return whether this file type matches one output path by suffix alone."""

        path_text = str(path).lower()
        return any(path_text.endswith(suffix.lower()) for suffix in self.suffixes)


@dataclass(frozen=True)
class ConversionRequest:
    """Resolved conversion request with detected source and target routing."""

    source_path: Path
    target_path: Path
    source_file_type: str
    target_file_type: str
    family: FileFamily


@dataclass(frozen=True)
class ConversionResult:
    """Result returned by one executed conversion."""

    output_path: Path
    metadata_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CombineRequest:
    """Resolved request for combining multiple same-family sources."""

    source_paths: tuple[Path, ...]
    source_file_types: tuple[str, ...]
    target_path: Path
    target_file_type: str
    family: FileFamily
    conversion_applied: bool


@dataclass(frozen=True)
class CombineResult:
    """Result returned by one executed combine request."""

    output_path: Path


@dataclass(frozen=True)
class TrajectoryConversionOptions:
    """Trajectory-specific conversion options currently used by `apply convert`."""

    input_path: str | Path | None = None
    cell: tuple[float, float, float] | None = None
    select: str | None = None
    x_range: str | None = None
    y_range: str | None = None
    z_range: str | None = None
    distance_range: str | None = None
    keep_molecules_intact: bool = False
    output_was_default: bool = False
    atom_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectorySelectionRequest:
    """Parsed compact partial-trajectory selection request."""

    kind: str
    start_token: str
    end_token: str | None
    unit: str
    user_selector: str
    suffix: str


@dataclass(frozen=True)
class TrajectorySelectionResolution:
    """Resolved contiguous frame slice and metadata for one selection request."""

    request: TrajectorySelectionRequest
    start_frame: int
    stop_frame_exclusive: int
    selected_frame_count: int
    resolved_start_time_fs: float | None = None
    resolved_end_time_fs: float | None = None
    resolved_start_step: int | None = None
    resolved_end_step: int | None = None


@dataclass(frozen=True)
class CubeConversionOptions:
    """Placeholder for future cube-specific conversion controls."""

    pass


@dataclass(frozen=True)
class _FamilyConversionHandler:
    """Bundle the per-family conversion behavior."""

    family: FileFamily
    default_target_file_type: str
    convert: Callable[[ConversionRequest, Any | None], ConversionResult]
    describe_plan: Callable[[ConversionRequest, Any | None], list[str]]


@dataclass(frozen=True)
class _FamilyCombineHandler:
    """Bundle per-family combine behavior."""

    family: FileFamily
    default_target_file_type: str
    raw_target_file_type: str | None
    combine: Callable[[CombineRequest, Any | None], CombineResult]
    describe_plan: Callable[[CombineRequest, Any | None], list[str]]


class ConversionRegistry:
    """Central registry for file-family detection and conversion dispatch."""

    def __init__(
        self,
        *,
        file_types: Sequence[ConversionFileType],
        family_handlers: Mapping[str, _FamilyConversionHandler],
        combine_handlers: Mapping[str, _FamilyCombineHandler] | None = None,
    ) -> None:
        self._file_types = tuple(file_types)
        self._file_types_by_id = {file_type.id: file_type for file_type in self._file_types}
        self._family_handlers = dict(family_handlers)
        self._combine_handlers = {} if combine_handlers is None else dict(combine_handlers)

    def detect_source_file_type(self, path: str | Path) -> ConversionFileType:
        """Detect and return the registered source file type for one path."""

        resolved = Path(path).expanduser().resolve()
        for file_type in self._file_types:
            if file_type.matches(resolved):
                return file_type
        raise ValueError(f"Unsupported conversion source format: {resolved}")

    def detect_file_family(self, path: str | Path) -> FileFamily:
        """Return the detected file family for one path."""

        return self.detect_source_file_type(path).family

    def allowed_target_file_types(self, path_or_file_type: str | Path) -> list[ConversionFileType]:
        """Return writable target file types available for one source family."""

        if isinstance(path_or_file_type, (str, Path)) and Path(path_or_file_type).suffix:
            source_type = self.detect_source_file_type(path_or_file_type)
        else:
            source_type = self._file_types_by_id[str(path_or_file_type)]
        return [
            file_type
            for file_type in self._file_types
            if file_type.family == source_type.family and file_type.writable
        ]

    def default_output_path(
        self,
        source: str | Path,
        *,
        target_file_type: str | None = None,
    ) -> Path:
        """Return the default output path for one resolved conversion target."""

        source_path = Path(source).expanduser().resolve()
        source_type = self.detect_source_file_type(source_path)
        target_id = (
            str(target_file_type)
            if target_file_type is not None
            else self._family_handlers[source_type.family].default_target_file_type
        )
        target_type = self._file_types_by_id[target_id]
        if target_type.default_output_path_factory is None:
            raise ValueError(f"No default output path rule is defined for '{target_type.id}'.")
        return target_type.default_output_path_factory(source_path)

    def preferred_target_file_type(
        self,
        source: str | Path | ConversionFileType,
    ) -> ConversionFileType:
        """Return LiNaK's preferred working-format target for one source."""

        source_type = (
            source
            if isinstance(source, ConversionFileType)
            else self.detect_source_file_type(source)
        )
        target_id = self._family_handlers[source_type.family].default_target_file_type
        return self._file_types_by_id[target_id]

    def _resolve_target_by_selector(
        self,
        *,
        source_type: ConversionFileType,
        target_selector: str,
    ) -> ConversionFileType:
        family_targets = self.allowed_target_file_types(source_type.id)
        selector = str(target_selector).strip().lower()
        for file_type in family_targets:
            aliases = {file_type.id.lower(), *(alias.lower() for alias in file_type.target_aliases)}
            if selector in aliases:
                return file_type
        allowed = ", ".join(
            sorted(
                {
                    alias
                    for file_type in family_targets
                    for alias in (file_type.id, *file_type.target_aliases)
                }
            )
        )
        raise ValueError(
            f"Unsupported target file type '{target_selector}' for {source_type.family} input. "
            f"Supported target file types: {allowed}."
        )

    def resolve_target_file_type(
        self,
        source: str | Path,
        *,
        output_path: str | Path | None = None,
        target_selector: str | None = None,
    ) -> ConversionFileType:
        """Resolve the requested target file type from output path or selector."""

        source_type = self.detect_source_file_type(source)
        family_targets = self.allowed_target_file_types(source_type.id)
        selected_by_selector: ConversionFileType | None = None
        if target_selector is not None:
            selected_by_selector = self._resolve_target_by_selector(
                source_type=source_type,
                target_selector=target_selector,
            )
        selected_by_output: ConversionFileType | None = None
        if output_path is not None:
            resolved_output = Path(output_path).expanduser().resolve()
            for file_type in family_targets:
                if file_type.matches_output_path(resolved_output):
                    selected_by_output = file_type
                    break
            if selected_by_output is None:
                supported_suffixes = sorted(
                    {
                        suffix
                        for file_type in family_targets
                        for suffix in file_type.suffixes
                        if suffix
                    }
                )
                raise ValueError(
                    f"Could not infer a supported target format from output path '{resolved_output}'. "
                    f"Use --target-file-type or one of: {', '.join(supported_suffixes)}."
                )
        if selected_by_selector is not None and selected_by_output is not None:
            if selected_by_selector.id != selected_by_output.id:
                raise ValueError(
                    "Requested target file type and output path extension do not agree: "
                    f"--target-file-type={selected_by_selector.id}, output='{output_path}'."
                )
            return selected_by_selector
        if selected_by_selector is not None:
            return selected_by_selector
        if selected_by_output is not None:
            return selected_by_output
        return self.preferred_target_file_type(source_type)

    def build_request(
        self,
        source: str | Path,
        *,
        output_path: str | Path | None = None,
        target_selector: str | None = None,
    ) -> ConversionRequest:
        """Resolve one conversion request from source and target selectors."""

        source_path = Path(source).expanduser().resolve()
        source_type = self.detect_source_file_type(source_path)
        target_type = self.resolve_target_file_type(
            source_path,
            output_path=output_path,
            target_selector=target_selector,
        )
        if target_type.family != source_type.family:
            raise ValueError(
                f"Cannot convert {source_type.family} sources into {target_type.family} targets."
            )
        resolved_output = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else self.default_output_path(source_path, target_file_type=target_type.id)
        )
        return ConversionRequest(
            source_path=source_path,
            target_path=resolved_output,
            source_file_type=source_type.id,
            target_file_type=target_type.id,
            family=source_type.family,
        )

    def build_default_request(
        self,
        source: str | Path,
        *,
        output_path: str | Path | None = None,
        target_selector: str | None = None,
        uniquify_default_output: bool = False,
        output_name_suffix: str | None = None,
    ) -> ConversionRequest:
        """Build one request and optionally avoid clobbering the default target path."""

        request = self.build_request(
            source,
            output_path=output_path,
            target_selector=target_selector,
        )
        if (
            output_path is None
            and target_selector is None
            and output_name_suffix is None
            and request.source_file_type == request.target_file_type
        ):
            return ConversionRequest(
                source_path=request.source_path,
                target_path=request.source_path,
                source_file_type=request.source_file_type,
                target_file_type=request.target_file_type,
                family=request.family,
            )
        if output_path is None and output_name_suffix:
            target_path = request.target_path
            if request.source_file_type == request.target_file_type:
                target_path = request.source_path
            request = ConversionRequest(
                source_path=request.source_path,
                target_path=_append_output_name_suffix(target_path, str(output_name_suffix)),
                source_file_type=request.source_file_type,
                target_file_type=request.target_file_type,
                family=request.family,
            )
        if (
            uniquify_default_output
            and output_path is None
            and request.target_path.exists()
        ):
            return ConversionRequest(
                source_path=request.source_path,
                target_path=_unique_path_with_numeric_suffix(request.target_path),
                source_file_type=request.source_file_type,
                target_file_type=request.target_file_type,
                family=request.family,
            )
        return request

    def describe_plan(
        self,
        request: ConversionRequest,
        *,
        options: Any | None = None,
    ) -> list[str]:
        """Return a human-readable conversion plan."""

        return self._family_handlers[request.family].describe_plan(request, options)

    def execute(
        self,
        request: ConversionRequest,
        *,
        options: Any | None = None,
    ) -> ConversionResult:
        """Execute one conversion request through the family handler."""

        return self._family_handlers[request.family].convert(request, options)

    def build_combine_request(
        self,
        sources: Sequence[str | Path],
        *,
        output_path: str | Path | None = None,
        no_convert: bool = False,
        uniquify_default_output: bool = False,
    ) -> CombineRequest:
        """Resolve one combine request across multiple sources."""

        resolved_sources = tuple(Path(source).expanduser().resolve() for source in sources)
        if len(resolved_sources) < 2:
            raise ValueError("Combine requires at least two input files.")
        source_types = tuple(self.detect_source_file_type(source) for source in resolved_sources)
        families = {source_type.family for source_type in source_types}
        if len(families) != 1:
            raise ValueError("Cannot combine mixed file families in one command.")
        family = source_types[0].family
        if family not in self._combine_handlers:
            raise ValueError(f"Combine is not implemented for the '{family}' file family.")
        handler = self._combine_handlers[family]
        if output_path is not None:
            target_type = self.resolve_target_file_type(
                resolved_sources[0],
                output_path=output_path,
                target_selector=None,
            )
        else:
            target_id = handler.raw_target_file_type if no_convert else handler.default_target_file_type
            if target_id is None:
                raise ValueError(
                    f"--no-convert is not supported for the '{family}' file family because "
                    "LiNaK has no clean lossless non-HDF5 combined representation."
                )
            target_type = self._file_types_by_id[target_id]
        if target_type.family != family:
            raise ValueError(
                f"Cannot combine {family} sources into a {target_type.family} target."
            )
        target_path = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else self.default_combined_output_path(
                resolved_sources,
                target_file_type=target_type.id,
            )
        )
        if uniquify_default_output and output_path is None and target_path.exists():
            target_path = _unique_path_with_numeric_suffix(target_path)
        return CombineRequest(
            source_paths=resolved_sources,
            source_file_types=tuple(source_type.id for source_type in source_types),
            target_path=target_path,
            target_file_type=target_type.id,
            family=family,
            conversion_applied=(target_type.id == handler.default_target_file_type),
        )

    def default_combined_output_path(
        self,
        sources: Sequence[str | Path],
        *,
        target_file_type: str,
    ) -> Path:
        """Return one default combined output path for the resolved target type."""

        resolved_sources = [Path(source).expanduser().resolve() for source in sources]
        if not resolved_sources:
            raise ValueError("At least one source is required.")
        target_type = self._file_types_by_id[str(target_file_type)]
        first_source = resolved_sources[0]
        output_dir = Path.cwd().resolve()
        base_stem = _combine_base_stem(first_source)
        if target_type.id == "trajectory_hdf5":
            return output_dir / f"{base_stem}_combined.traj.h5"
        if target_type.id == "trajectory_xyz":
            return output_dir / f"{base_stem}_combined.xyz"
        if target_type.id == "cube_hdf5":
            return output_dir / f"{base_stem}_combined.cube.h5"
        if target_type.id == "cube_file":
            return output_dir / f"{base_stem}_combined.cube"
        suffix = target_type.suffixes[0] if target_type.suffixes else ""
        return output_dir / f"{base_stem}_combined{suffix}"

    def describe_combine_plan(
        self,
        request: CombineRequest,
        *,
        options: Any | None = None,
    ) -> list[str]:
        """Return a human-readable combine plan."""

        return self._combine_handlers[request.family].describe_plan(request, options)

    def execute_combine(
        self,
        request: CombineRequest,
        *,
        options: Any | None = None,
    ) -> CombineResult:
        """Execute one combine request through the family handler."""

        return self._combine_handlers[request.family].combine(request, options)
def _default_cube_hdf5_output_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    source_text = str(source_path).lower()
    if source_text.endswith(".cube.h5"):
        return source_path
    if source_text.endswith(".cube.hdf5"):
        return source_path.with_suffix(".h5")
    if source_path.suffix.lower() == ".cube":
        return source_path.with_suffix(".cube.h5")
    return source_path.with_suffix(".cube.h5")


def _default_cube_output_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    source_text = str(source_path).lower()
    if source_text.endswith(".cube.h5"):
        return Path(str(source_path)[:-3])
    if source_text.endswith(".cube.hdf5"):
        return Path(str(source_path)[:-5])
    if source_path.suffix.lower() == ".cube":
        return source_path
    return source_path.with_suffix(".cube")


def _default_xyz_output_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    source_text = str(source_path).lower()
    if source_text.endswith(".traj.h5"):
        return Path(str(source_path)[:-8] + ".xyz")
    if source_text.endswith(".traj.hdf5"):
        return Path(str(source_path)[:-10] + ".xyz")
    return source_path.with_suffix(".xyz")


def _combine_base_stem(source: str | Path) -> str:
    source_path = Path(source).expanduser().resolve()
    source_text = str(source_path).lower()
    name = source_path.name
    if source_text.endswith(".traj.h5"):
        return name[:-8]
    if source_text.endswith(".traj.hdf5"):
        return name[:-10]
    if source_text.endswith(".cube.h5"):
        return name[:-8]
    if source_text.endswith(".cube.hdf5"):
        return name[:-10]
    return source_path.stem or name or "combined"


def _append_output_name_suffix(path: Path, suffix: str) -> Path:
    from .analysis.output_naming import append_hdf5_name_suffix

    return append_hdf5_name_suffix(path, suffix)


def _format_selector_number(value_text: str, *, for_filename: bool) -> str:
    text = str(value_text).strip().lower()
    if not text:
        raise ValueError("Trajectory selection value cannot be empty.")
    if for_filename:
        return text.replace(".", "p")
    return text


def _selector_suffix_token(unit: str) -> str:
    return {"f": "f", "ps": "ps", "%": "pct", "step": "step"}[unit]


def _parse_selection_token(token: str) -> tuple[str, str]:
    value_text = str(token).strip().lower()
    for suffix in ("step", "ps", "f", "%"):
        if value_text.endswith(suffix):
            numeric_text = value_text[: -len(suffix)].strip()
            if not numeric_text:
                break
            return numeric_text, suffix
    raise ValueError(
        f"Unsupported selection token '{token}'. Use forms like 1000f, 5ps, 50%, or 500step."
    )


def parse_trajectory_selection(selector: str) -> TrajectorySelectionRequest:
    raw = str(selector).strip()
    if not raw:
        raise ValueError("Trajectory selection cannot be empty.")
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            "Unsupported trajectory selection syntax. Use first:<value>, last:<value>, or range:<start>:<end>."
        )
    kind = parts[0].strip().lower()
    if kind not in {"first", "last", "range"}:
        raise ValueError(f"Unsupported trajectory selection kind '{parts[0]}'.")
    if kind == "range" and len(parts) != 3:
        raise ValueError("Range selection must use 'range:<start>:<end>'.")
    if kind != "range" and len(parts) != 2:
        raise ValueError("First/last selection must use '<kind>:<value>'.")
    start_token = parts[1].strip()
    end_token = parts[2].strip() if kind == "range" else None
    start_value_text, unit = _parse_selection_token(start_token)
    if kind == "range":
        assert end_token is not None
        end_value_text, end_unit = _parse_selection_token(end_token)
        if end_unit != unit:
            raise ValueError("Range selection start/end units must match.")
        suffix = (
            f"_range{_format_selector_number(start_value_text, for_filename=True)}{_selector_suffix_token(unit)}"
            f"_{_format_selector_number(end_value_text, for_filename=True)}{_selector_suffix_token(unit)}"
        )
    else:
        suffix = (
            f"_{kind}"
            f"{_format_selector_number(start_value_text, for_filename=True)}"
            f"{_selector_suffix_token(unit)}"
        )
    return TrajectorySelectionRequest(
        kind=kind,
        start_token=start_token,
        end_token=end_token,
        unit=unit,
        user_selector=raw,
        suffix=suffix,
    )


def _trajectory_time_values_fs(frames: list[Any], *, frame_timestep_fs: float | None) -> np.ndarray:
    values: list[float] = []
    has_per_frame_time = True
    for frame in frames:
        raw = getattr(frame, "info", {}).get("time_fs")
        if raw is None:
            has_per_frame_time = False
            break
        values.append(float(raw))
    if has_per_frame_time and values:
        return np.asarray(values, dtype=float)
    if frame_timestep_fs is None:
        raise ValueError(
            "Time-based trajectory selection requires stored time metadata (time_fs or frame_timestep_fs)."
        )
    return np.arange(len(frames), dtype=float) * float(frame_timestep_fs)


def _trajectory_step_values(frames: list[Any], *, stride_md: int | None) -> np.ndarray:
    values: list[int] = []
    has_per_frame_steps = True
    for frame in frames:
        raw = getattr(frame, "info", {}).get("timestep")
        if raw is None:
            has_per_frame_steps = False
            break
        values.append(int(raw))
    if has_per_frame_steps and values:
        return np.asarray(values, dtype=int)
    if stride_md is None:
        raise ValueError(
            "Step-based trajectory selection requires stored step metadata (timestep or trajectory_stride_md)."
        )
    return np.arange(len(frames), dtype=int) * int(stride_md)


def _resolve_selection_time_and_step_metadata(
    frames: list[Any],
    *,
    source_path: Path,
) -> tuple[float | None, int | None]:
    """Resolve trajectory selection time/step metadata from frames and stored source metadata."""

    frame_timestep_fs: float | None = None
    trajectory_stride_md: int | None = None
    if frames:
        raw_frame_timestep = frames[0].info.get("frame_timestep_fs")
        if raw_frame_timestep is not None:
            frame_timestep_fs = float(raw_frame_timestep)
        raw_stride = frames[0].info.get("trajectory_stride_md")
        if raw_stride is not None:
            trajectory_stride_md = int(raw_stride)

    if frame_timestep_fs is not None and trajectory_stride_md is not None:
        return frame_timestep_fs, trajectory_stride_md

    from .trajectory.io import is_linak_trajectory_hdf5, read_trajectory_hdf5_metadata

    if not is_linak_trajectory_hdf5(source_path):
        return frame_timestep_fs, trajectory_stride_md

    stored_metadata = read_trajectory_hdf5_metadata(source_path)
    if stored_metadata is None:
        return frame_timestep_fs, trajectory_stride_md

    if frame_timestep_fs is None:
        if stored_metadata.frame_timestep_fs is not None:
            frame_timestep_fs = float(stored_metadata.frame_timestep_fs)
        elif (
            stored_metadata.md_timestep_fs is not None
            and stored_metadata.trajectory_stride_md is not None
        ):
            frame_timestep_fs = (
                float(stored_metadata.md_timestep_fs)
                * float(stored_metadata.trajectory_stride_md)
            )

    if trajectory_stride_md is None and stored_metadata.trajectory_stride_md is not None:
        trajectory_stride_md = int(stored_metadata.trajectory_stride_md)

    return frame_timestep_fs, trajectory_stride_md


def _resolve_selector_count(total: int, value_text: str) -> int:
    try:
        value = int(value_text)
    except ValueError as exc:
        raise ValueError(f"Frame/step selector '{value_text}' must be an integer.") from exc
    if value <= 0:
        raise ValueError("Trajectory selection counts must be positive.")
    return min(int(total), value)


def _resolve_selector_fraction(total: int, value_text: str) -> int:
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"Percentage selector '{value_text}' must be numeric.") from exc
    if value <= 0.0 or value > 100.0:
        raise ValueError("Percentage trajectory selection must be within (0, 100].")
    return min(int(total), max(1, int(np.floor(total * (value / 100.0)))))


def resolve_trajectory_selection(
    frames: list[Any],
    request: TrajectorySelectionRequest,
    *,
    frame_timestep_fs: float | None = None,
    trajectory_stride_md: int | None = None,
) -> TrajectorySelectionResolution:
    total = len(frames)
    if total <= 0:
        raise ValueError("Trajectory selection requires at least one frame.")

    time_values_fs = None
    step_values = None
    unit = request.unit
    if unit == "ps":
        time_values_fs = _trajectory_time_values_fs(frames, frame_timestep_fs=frame_timestep_fs)
    elif unit == "step":
        step_values = _trajectory_step_values(frames, stride_md=trajectory_stride_md)

    def _value_to_count(value_token: str) -> int:
        value_text, parsed_unit = _parse_selection_token(value_token)
        if parsed_unit != unit:
            raise ValueError("Selection units do not match parsed request.")
        if unit == "f":
            return _resolve_selector_count(total, value_text)
        if unit == "%":
            return _resolve_selector_fraction(total, value_text)
        if unit == "ps":
            assert time_values_fs is not None
            target_time_fs = float(value_text) * 1000.0
            if target_time_fs <= 0.0:
                raise ValueError("Time-based trajectory selection must be positive.")
            return min(total, max(1, int(np.count_nonzero(time_values_fs <= target_time_fs))))
        assert step_values is not None
        target_step = int(value_text)
        if target_step <= 0:
            raise ValueError("Step-based trajectory selection must be positive.")
        return min(total, max(1, int(np.count_nonzero(step_values <= target_step))))

    if request.kind in {"first", "last"}:
        count = _value_to_count(request.start_token)
        start_frame = 0 if request.kind == "first" else total - count
        stop_frame_exclusive = start_frame + count
    else:
        assert request.end_token is not None
        if unit == "f":
            start_value = int(_parse_selection_token(request.start_token)[0])
            end_value = int(_parse_selection_token(request.end_token)[0])
            if start_value < 0 or end_value < 0 or end_value <= start_value:
                raise ValueError("Frame range selection must satisfy 0 <= start < end.")
            start_frame = min(total, start_value)
            stop_frame_exclusive = min(total, end_value)
        elif unit == "%":
            start_pct = float(_parse_selection_token(request.start_token)[0])
            end_pct = float(_parse_selection_token(request.end_token)[0])
            if start_pct < 0.0 or end_pct <= start_pct or end_pct > 100.0:
                raise ValueError("Percentage range selection must satisfy 0 <= start < end <= 100.")
            start_frame = min(total - 1, max(0, int(np.floor(total * (start_pct / 100.0)))))
            stop_frame_exclusive = min(total, max(start_frame + 1, int(np.floor(total * (end_pct / 100.0)))))
        elif unit == "ps":
            assert time_values_fs is not None
            start_time_fs = float(_parse_selection_token(request.start_token)[0]) * 1000.0
            end_time_fs = float(_parse_selection_token(request.end_token)[0]) * 1000.0
            if end_time_fs <= start_time_fs or start_time_fs < 0.0:
                raise ValueError("Time range selection must satisfy 0 <= start < end.")
            start_frame = int(np.searchsorted(time_values_fs, start_time_fs, side="left"))
            stop_frame_exclusive = int(np.searchsorted(time_values_fs, end_time_fs, side="right"))
        else:
            assert step_values is not None
            start_step = int(_parse_selection_token(request.start_token)[0])
            end_step = int(_parse_selection_token(request.end_token)[0])
            if end_step <= start_step or start_step < 0:
                raise ValueError("Step range selection must satisfy 0 <= start < end.")
            start_frame = int(np.searchsorted(step_values, start_step, side="left"))
            stop_frame_exclusive = int(np.searchsorted(step_values, end_step, side="right"))

    start_frame = max(0, min(total - 1, start_frame))
    stop_frame_exclusive = max(start_frame + 1, min(total, stop_frame_exclusive))
    selected_count = stop_frame_exclusive - start_frame
    resolved_start_time_fs = None
    resolved_end_time_fs = None
    resolved_start_step = None
    resolved_end_step = None
    if time_values_fs is not None:
        resolved_start_time_fs = float(time_values_fs[start_frame])
        resolved_end_time_fs = float(time_values_fs[stop_frame_exclusive - 1])
    if step_values is not None:
        resolved_start_step = int(step_values[start_frame])
        resolved_end_step = int(step_values[stop_frame_exclusive - 1])
    return TrajectorySelectionResolution(
        request=request,
        start_frame=start_frame,
        stop_frame_exclusive=stop_frame_exclusive,
        selected_frame_count=selected_count,
        resolved_start_time_fs=resolved_start_time_fs,
        resolved_end_time_fs=resolved_end_time_fs,
        resolved_start_step=resolved_start_step,
        resolved_end_step=resolved_end_step,
    )


def _unique_path_with_numeric_suffix(path: Path) -> Path:
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    candidate = path
    while candidate.exists():
        candidate = parent / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _raw_trajectory_detector(path: Path) -> bool:
    if is_linak_cube_hdf5(path):
        return False
    path_text = str(path).lower()
    if path_text.endswith(".cube"):
        return False
    from .trajectory.io import is_linak_trajectory_hdf5

    if is_linak_trajectory_hdf5(path):
        return False
    return path.suffix.lower() not in {".h5", ".hdf5"}


def _xyz_trajectory_detector(path: Path) -> bool:
    path_text = str(path).lower()
    return path_text.endswith(".xyz") or path_text.endswith(".extxyz")


def _collect_trajectory_conversion_metadata(
    trajectory: str | Path,
    *,
    input_path: str | None,
) -> tuple[Any | None, list[str]]:
    from .pbc import (
        extract_cell_from_simulation_input,
        extract_fixed_atom_indices_from_simulation_input,
        extract_frame_timestep_fs_from_simulation_input,
        find_unique_simulation_input,
    )
    from .trajectory.io import TrajectoryStoredMetadata

    trajectory_path = Path(trajectory).expanduser().resolve()
    resolved_input: Path | None = None
    metadata_notes: list[str] = []
    if input_path is not None:
        resolved_input = Path(input_path).expanduser().resolve()
        metadata_notes.append(f"input metadata source: explicit --input ({resolved_input})")
    else:
        try:
            resolved_input = find_unique_simulation_input(trajectory_path.parent)
            metadata_notes.append(f"input metadata source: auto-detected ({resolved_input})")
        except (FileNotFoundError, ValueError) as exc:
            metadata_notes.append(f"input metadata source: none ({exc})")

    if resolved_input is None:
        metadata_notes.extend(
            [
                "cell metadata: not found",
                "timestep metadata: not found",
                "fixed atoms metadata: not found",
            ]
        )
        return None, metadata_notes

    input_format = resolved_input.suffix.lower().lstrip(".") or None
    cell_angstrom: tuple[float, float, float] | None = None
    frame_timestep_fs: float | None = None
    md_timestep_fs: float | None = None
    trajectory_stride_md: int | None = None
    fixed_atom_indices: tuple[int, ...] = ()

    try:
        cell_angstrom = extract_cell_from_simulation_input(resolved_input)
        metadata_notes.append(
            "cell metadata: found "
            f"({cell_angstrom[0]:.6g} {cell_angstrom[1]:.6g} {cell_angstrom[2]:.6g} Angstrom)"
        )
    except Exception as exc:
        metadata_notes.append(f"cell metadata: not found ({exc})")

    try:
        frame_timestep_fs, md_timestep_fs, trajectory_stride_md = (
            extract_frame_timestep_fs_from_simulation_input(resolved_input)
        )
        metadata_notes.append(
            "timestep metadata: found "
            f"(frame={frame_timestep_fs:.6g} fs, md={md_timestep_fs:.6g} fs, stride={trajectory_stride_md})"
        )
    except Exception as exc:
        metadata_notes.append(f"timestep metadata: not found ({exc})")

    try:
        parsed_fixed = extract_fixed_atom_indices_from_simulation_input(resolved_input)
        if parsed_fixed:
            fixed_atom_indices = parsed_fixed
            metadata_notes.append(
                f"fixed atoms metadata: found ({len(fixed_atom_indices)} atom(s))"
            )
        else:
            metadata_notes.append("fixed atoms metadata: not found")
    except Exception as exc:
        metadata_notes.append(f"fixed atoms metadata: not found ({exc})")

    metadata = TrajectoryStoredMetadata(
        input_path=resolved_input,
        input_format=input_format,
        cell_angstrom=cell_angstrom,
        cell_source="simulation input",
        frame_timestep_fs=frame_timestep_fs,
        md_timestep_fs=md_timestep_fs,
        trajectory_stride_md=trajectory_stride_md,
        timestep_source="simulation input",
        fixed_atom_indices=fixed_atom_indices,
        fixed_atoms_source="simulation input" if fixed_atom_indices else None,
    )
    if (
        metadata.cell_angstrom is None
        and metadata.frame_timestep_fs is None
        and not metadata.fixed_atom_indices
    ):
        return None, metadata_notes
    return metadata, metadata_notes


def _frames_have_usable_periodic_cell(frames: list[Any]) -> bool:
    if not frames:
        return False
    lengths = np.asarray(frames[0].cell.lengths(), dtype=float)
    return lengths.shape == (3,) and np.all(np.isfinite(lengths)) and np.all(lengths > 0.0)


def _cell_lengths_from_frame(frame: Any) -> tuple[float, float, float]:
    lengths = np.asarray(frame.cell.lengths(), dtype=float)
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]))


def _conversion_metadata_with_frame_cell_fallback(
    metadata: Any | None,
    frames: list[Any],
    metadata_notes: list[str],
) -> Any | None:
    if metadata is not None and metadata.cell_angstrom is not None:
        return metadata
    if not _frames_have_usable_periodic_cell(frames):
        return metadata

    from .trajectory.io import TrajectoryStoredMetadata
    from dataclasses import replace

    frame_cell = _cell_lengths_from_frame(frames[0])
    metadata_notes.append(
        "cell metadata: using periodic cell embedded in trajectory "
        f"({frame_cell[0]:.6g} {frame_cell[1]:.6g} {frame_cell[2]:.6g} Angstrom)"
    )
    base = metadata if metadata is not None else TrajectoryStoredMetadata()
    return replace(
        base,
        cell_angstrom=frame_cell,
        cell_source=base.cell_source or "trajectory frame cell",
    )


def _apply_fixed_constraints_from_conversion_metadata(
    frames: list[Any],
    metadata: Any | None,
) -> None:
    if metadata is None or not metadata.fixed_atom_indices:
        return
    from ase.constraints import FixAtoms

    indices = list(metadata.fixed_atom_indices)
    for frame in frames:
        frame.set_constraint(FixAtoms(indices=indices))


def _conversion_metadata_with_pbc_cache(
    metadata: Any | None,
    *,
    cell: tuple[float, float, float],
) -> Any:
    from dataclasses import replace
    from .trajectory.io import TrajectoryStoredMetadata

    base = metadata if metadata is not None else TrajectoryStoredMetadata()
    return replace(
        base,
        pbc_applied=True,
        pbc_cell_angstrom=cell,
        pbc_source=base.cell_source or "conversion cell",
        coordinate_basis="pbc-wrapped",
    )


def _conversion_metadata_with_surface_cache(
    metadata: Any | None,
    frames: list[Any],
) -> Any:
    from dataclasses import replace
    from .trajectory.io import TrajectoryStoredMetadata

    base = metadata if metadata is not None else TrajectoryStoredMetadata()
    try:
        from .analysis.density import estimate_surface_reference

        estimate = estimate_surface_reference(
            frames,
            axis="z",
            mode="auto",
            surface_elements=None,
            include_fixed_surface_atoms=False,
            surface_options=None,
        )
    except Exception as exc:
        LOGGER.warning("Conversion surface cache unavailable: %s", exc)
        return replace(
            base,
            surface_cache_status="unavailable",
            surface_cache_axis="z",
            surface_cache_mode="auto",
            surface_cache_elements=None,
            surface_cache_include_fixed_surface_atoms=False,
            surface_cache_rough_surface_envelope_A=None,
            surface_cache_source="conversion",
            surface_cache_unavailable_reason=str(exc),
            surface_cache_estimate=None,
        )

    if estimate is None:
        LOGGER.warning("Conversion surface cache unavailable: no surface reference found.")
        return replace(
            base,
            surface_cache_status="unavailable",
            surface_cache_axis="z",
            surface_cache_mode="auto",
            surface_cache_elements=None,
            surface_cache_include_fixed_surface_atoms=False,
            surface_cache_rough_surface_envelope_A=None,
            surface_cache_source="conversion",
            surface_cache_unavailable_reason="no surface reference found",
            surface_cache_estimate=None,
        )

    LOGGER.info("Cached default per-frame surface positions during conversion (axis=Z, mode=auto).")
    return replace(
        base,
        surface_cache_status="available",
        surface_cache_axis="z",
        surface_cache_mode="auto",
        surface_cache_elements=None,
        surface_cache_include_fixed_surface_atoms=False,
        surface_cache_rough_surface_envelope_A=None,
        surface_cache_source="conversion",
        surface_cache_unavailable_reason=None,
        surface_cache_estimate=estimate,
    )


def _describe_trajectory_conversion_plan(
    request: ConversionRequest,
    options: TrajectoryConversionOptions | None,
) -> list[str]:
    selection_request = None if options is None or options.select is None else parse_trajectory_selection(options.select)
    plan = [
        f"input trajectory: {request.source_path}",
        f"source family: {request.family}",
        f"source file type: {request.source_file_type}",
        f"target file type: {request.target_file_type}",
        f"output path: {request.target_path}",
    ]
    if selection_request is not None:
        plan.append(f"selection: {selection_request.user_selector}")
    if options is not None:
        filter_parts = [
            f"x={options.x_range}" if options.x_range else None,
            f"y={options.y_range}" if options.y_range else None,
            f"z={options.z_range}" if options.z_range else None,
            f"distance={options.distance_range}" if options.distance_range else None,
        ]
        filter_parts = [part for part in filter_parts if part is not None]
        if filter_parts:
            plan.append(
                "spatial filter: "
                + ", ".join(filter_parts)
                + f", keep_molecules_intact={bool(options.keep_molecules_intact)}"
            )
    if request.target_file_type == "trajectory_hdf5":
        stored_metadata, metadata_notes = _collect_trajectory_conversion_metadata(
            request.source_path,
            input_path=None if options is None else (
                None if options.input_path is None else str(options.input_path)
            ),
        )
        plan.extend(
            [
                "format marker: linak-trajectory-hdf5",
                "storage: chunked HDF5 datasets with lightweight compression",
                "topology policy: fixed or variable atom counts across frames are preserved in HDF5",
                "workflow: converted trajectory remains a direct input to `linak compute ...`",
            ]
        )
        if stored_metadata is not None:
            plan.append(
                "embedded metadata precedence: explicit CLI > trajectory HDF5 > input discovery"
            )
        plan.extend(metadata_notes)
        return plan
    plan.extend(
        [
            "format marker: standard multi-frame trajectory file",
            "workflow: LiNaK reads the source trajectory and rewrites all frames to the requested format",
        ]
    )
    return plan


def _convert_trajectory_request(
    request: ConversionRequest,
    options: TrajectoryConversionOptions | None,
) -> ConversionResult:
    from dataclasses import replace
    from .pbc import apply_pbc_to_frames
    from .trajectory.spatial_filter import (
        append_output_name_suffix,
        apply_spatial_filter,
        spatial_filter_options_from_mapping,
    )
    from .trajectory.io import TrajectoryStoredMetadata, read_trajectory, write_trajectory

    frames = read_trajectory(
        request.source_path,
        atom_aliases=options.atom_aliases if options is not None else None,
    )
    selection_resolution: TrajectorySelectionResolution | None = None
    if options is not None and options.select:
        selection_request = parse_trajectory_selection(options.select)
        frame_timestep_fs, trajectory_stride_md = _resolve_selection_time_and_step_metadata(
            frames,
            source_path=request.source_path,
        )
        selection_resolution = resolve_trajectory_selection(
            frames,
            selection_request,
            frame_timestep_fs=frame_timestep_fs,
            trajectory_stride_md=trajectory_stride_md,
        )
        frames = frames[
            selection_resolution.start_frame : selection_resolution.stop_frame_exclusive
        ]
    spatial_filter_result = None
    if options is not None:
        spatial_options = spatial_filter_options_from_mapping(
            {
                "x_range": options.x_range,
                "y_range": options.y_range,
                "z_range": options.z_range,
                "distance_range": options.distance_range,
                "keep_molecules_intact": options.keep_molecules_intact,
            },
            surface_axis="z",
            surface_mode="auto",
            surface_elements=None,
            include_fixed_surface_atoms=False,
            rough_surface_envelope_A=None,
        )
        if spatial_options.active:
            spatial_filter_result = apply_spatial_filter(
                frames,
                options=spatial_options,
                precomputed_surface_estimate=None,
            )
            frames = spatial_filter_result.frames
            if options.output_was_default and spatial_filter_result.filename_suffix:
                request = ConversionRequest(
                    source_path=request.source_path,
                    target_path=append_output_name_suffix(
                        request.target_path,
                        spatial_filter_result.filename_suffix,
                    ),
                    source_file_type=request.source_file_type,
                    target_file_type=request.target_file_type,
                    family=request.family,
                )
    if request.target_file_type == "trajectory_hdf5":
        stored_metadata, metadata_notes = _collect_trajectory_conversion_metadata(
            request.source_path,
            input_path=None if options is None else (
                None if options.input_path is None else str(options.input_path)
            ),
        )
        stored_metadata = _conversion_metadata_with_frame_cell_fallback(
            stored_metadata,
            frames,
            metadata_notes,
        )
        _apply_fixed_constraints_from_conversion_metadata(frames, stored_metadata)
        if stored_metadata is not None and stored_metadata.cell_angstrom is not None:
            conversion_cell = stored_metadata.cell_angstrom
            frames = apply_pbc_to_frames(frames, conversion_cell)
            stored_metadata = _conversion_metadata_with_pbc_cache(
                stored_metadata,
                cell=conversion_cell,
            )
            LOGGER.info(
                "Applied PBC during conversion using cell %.6g %.6g %.6g Angstrom.",
                conversion_cell[0],
                conversion_cell[1],
                conversion_cell[2],
            )
        else:
            LOGGER.info("PBC conversion cache unavailable: no valid cell metadata found.")
        stored_metadata = _conversion_metadata_with_surface_cache(stored_metadata, frames)
        if selection_resolution is not None:
            base_metadata = (
                stored_metadata
                if stored_metadata is not None
                else TrajectoryStoredMetadata()
            )
            stored_metadata = replace(
                base_metadata,
                selection_user=selection_resolution.request.user_selector,
                selection_kind=selection_resolution.request.kind,
                selection_unit=selection_resolution.request.unit,
                selection_start_frame=selection_resolution.start_frame,
                selection_stop_frame_exclusive=selection_resolution.stop_frame_exclusive,
                selection_selected_frame_count=selection_resolution.selected_frame_count,
                selection_resolved_start_time_fs=selection_resolution.resolved_start_time_fs,
                selection_resolved_end_time_fs=selection_resolution.resolved_end_time_fs,
                selection_resolved_start_step=selection_resolution.resolved_start_step,
                selection_resolved_end_step=selection_resolution.resolved_end_step,
            )
        if spatial_filter_result is not None:
            base_metadata = (
                stored_metadata
                if stored_metadata is not None
                else TrajectoryStoredMetadata()
            )
            stored_metadata = replace(
                base_metadata,
                fixed_atom_indices=(),
                fixed_atoms_source=None,
                spatial_filter_metadata=spatial_filter_result.metadata,
            )
        for note in metadata_notes:
            LOGGER.info("%s", note)
        converted_path = write_trajectory(
            frames,
            request.target_path,
            source_path=request.source_path,
            source_format=request.source_path.suffix.lower().lstrip("."),
            metadata=stored_metadata,
        )
        return ConversionResult(output_path=converted_path, metadata_notes=tuple(metadata_notes))
    converted_path = write_trajectory(
        frames,
        request.target_path,
        source_path=request.source_path,
        source_format=request.source_path.suffix.lower().lstrip("."),
        metadata=(
            None
            if spatial_filter_result is None
            else TrajectoryStoredMetadata(
                spatial_filter_metadata=spatial_filter_result.metadata,
            )
        ),
    )
    return ConversionResult(output_path=converted_path, metadata_notes=())


def _trajectory_combine_metadata(request: CombineRequest) -> Any:
    from .trajectory.io import TrajectoryStoredMetadata

    total_frames = 0
    combine_timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return TrajectoryStoredMetadata(
        combine_source_paths=tuple(str(path) for path in request.source_paths),
        combine_source_file_types=request.source_file_types,
        combine_timestamp_utc=combine_timestamp_utc,
        combine_total_frames=total_frames,
        combine_conversion_applied=bool(request.conversion_applied),
        combine_linak_version=__version__,
    )


def _metadata_consistent_tuple(
    values: list[tuple[float, float, float] | None],
    *,
    label: str,
) -> tuple[float, float, float] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    reference = np.asarray(present[0], dtype=float)
    for candidate in present[1:]:
        if not np.allclose(np.asarray(candidate, dtype=float), reference, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"Cannot combine trajectories with inconsistent {label}: "
                f"{present[0]} vs {candidate}."
            )
    if len(present) != len(values):
        raise ValueError(
            f"Cannot combine trajectories when some sources define {label} and others do not."
        )
    return tuple(float(value) for value in reference)


def _metadata_consistent_scalar(
    values: list[float | int | None],
    *,
    label: str,
    tolerance: float | None = None,
) -> float | int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    reference = present[0]
    for candidate in present[1:]:
        if tolerance is None:
            matches = candidate == reference
        else:
            matches = bool(
                np.isclose(float(candidate), float(reference), rtol=0.0, atol=float(tolerance))
            )
        if not matches:
            raise ValueError(
                f"Cannot combine trajectories with inconsistent {label}: "
                f"{reference} vs {candidate}."
            )
    if len(present) != len(values):
        raise ValueError(
            f"Cannot combine trajectories when some sources define {label} and others do not."
        )
    return reference


def _metadata_consistent_indices(
    values: list[tuple[int, ...]],
    *,
    label: str,
) -> tuple[int, ...]:
    if not values:
        return ()
    reference = tuple(int(value) for value in values[0])
    for candidate in values[1:]:
        normalized = tuple(int(value) for value in candidate)
        if normalized != reference:
            raise ValueError(
                f"Cannot combine trajectories with inconsistent {label}: "
                f"{reference} vs {normalized}."
            )
    return reference


def _resolve_combined_trajectory_metadata(
    request: CombineRequest,
    options: TrajectoryConversionOptions | None,
    *,
    frames: list[Any],
) -> tuple[Any, list[str]]:
    from dataclasses import replace
    from .trajectory.io import TrajectoryStoredMetadata

    metadata_notes: list[str] = []
    explicit_input = None if options is None or options.input_path is None else str(options.input_path)
    explicit_cell = None if options is None else options.cell
    if explicit_input is not None:
        stored_metadata, source_notes = _collect_trajectory_conversion_metadata(
            request.source_paths[0],
            input_path=explicit_input,
        )
        metadata_notes.extend(source_notes)
        base = stored_metadata if stored_metadata is not None else TrajectoryStoredMetadata()
        if explicit_cell is not None:
            base = replace(
                base,
                cell_angstrom=tuple(float(value) for value in explicit_cell),
                cell_source="explicit --cell",
            )
            metadata_notes.append(
                "cell metadata: explicit "
                f"({explicit_cell[0]:.6g} {explicit_cell[1]:.6g} {explicit_cell[2]:.6g} Angstrom)"
            )
        resolved = _conversion_metadata_with_frame_cell_fallback(base, frames, metadata_notes)
        return resolved, metadata_notes

    source_metadatas: list[Any | None] = []
    for source_path in request.source_paths:
        metadata, source_notes = _collect_trajectory_conversion_metadata(source_path, input_path=None)
        source_metadatas.append(metadata)
        for note in source_notes:
            metadata_notes.append(f"{source_path.name}: {note}")

    cell_angstrom = (
        tuple(float(value) for value in explicit_cell)
        if explicit_cell is not None
        else _metadata_consistent_tuple(
            [None if metadata is None else metadata.cell_angstrom for metadata in source_metadatas],
            label="cell metadata",
        )
    )
    cell_source = (
        "explicit --cell"
        if explicit_cell is not None
        else ("auto-detected per-source simulation input" if cell_angstrom is not None else None)
    )
    frame_timestep_fs = _metadata_consistent_scalar(
        [None if metadata is None else metadata.frame_timestep_fs for metadata in source_metadatas],
        label="frame timestep metadata",
        tolerance=1.0e-12,
    )
    md_timestep_fs = _metadata_consistent_scalar(
        [None if metadata is None else metadata.md_timestep_fs for metadata in source_metadatas],
        label="MD timestep metadata",
        tolerance=1.0e-12,
    )
    trajectory_stride_md = _metadata_consistent_scalar(
        [None if metadata is None else metadata.trajectory_stride_md for metadata in source_metadatas],
        label="trajectory stride metadata",
    )
    fixed_atom_indices = _metadata_consistent_indices(
        [() if metadata is None else metadata.fixed_atom_indices for metadata in source_metadatas],
        label="fixed-atom metadata",
    )
    fixed_atoms_source = (
        "auto-detected per-source simulation input" if fixed_atom_indices else None
    )
    timestep_source = (
        None if frame_timestep_fs is None else "auto-detected per-source simulation input"
    )

    resolved = TrajectoryStoredMetadata(
        input_path=None,
        input_format=None,
        cell_angstrom=cell_angstrom,
        cell_source=cell_source,
        frame_timestep_fs=(
            None if frame_timestep_fs is None else float(frame_timestep_fs)
        ),
        md_timestep_fs=None if md_timestep_fs is None else float(md_timestep_fs),
        trajectory_stride_md=(
            None if trajectory_stride_md is None else int(trajectory_stride_md)
        ),
        timestep_source=timestep_source,
        fixed_atom_indices=fixed_atom_indices,
        fixed_atoms_source=fixed_atoms_source,
    )
    resolved = _conversion_metadata_with_frame_cell_fallback(resolved, frames, metadata_notes)
    return resolved, metadata_notes


def _annotate_raw_combined_frames(
    frames: list[Any],
    *,
    request: CombineRequest,
    total_frames: int,
) -> None:
    source_paths_json = json.dumps([str(path) for path in request.source_paths])
    source_types_json = json.dumps(list(request.source_file_types))
    combine_timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for frame in frames:
        frame.info["linak_combine_source_paths_json"] = source_paths_json
        frame.info["linak_combine_source_file_types_json"] = source_types_json
        frame.info["linak_combine_timestamp_utc"] = combine_timestamp_utc
        frame.info["linak_combine_total_frames"] = int(total_frames)
        frame.info["linak_combine_conversion_applied"] = bool(request.conversion_applied)
        frame.info["linak_combine_linak_version"] = __version__


def _describe_trajectory_combine_plan(
    request: CombineRequest,
    options: TrajectoryConversionOptions | None,
) -> list[str]:
    plan = [
        f"input family: {request.family}",
        f"sources ({len(request.source_paths)}): {', '.join(str(path) for path in request.source_paths)}",
        f"source file types: {', '.join(request.source_file_types)}",
        f"target file type: {request.target_file_type}",
        f"output path: {request.target_path}",
        f"conversion applied: {'yes' if request.conversion_applied else 'no'}",
        "ordering policy: input file order is preserved exactly",
    ]
    if request.target_file_type == "trajectory_hdf5":
        if options is not None and options.input_path is not None:
            plan.append(f"metadata override: explicit --input ({Path(options.input_path).expanduser().resolve()})")
        elif options is not None and options.cell is not None:
            plan.append(
                "metadata override: explicit --cell "
                f"({options.cell[0]:.6g} {options.cell[1]:.6g} {options.cell[2]:.6g} Angstrom)"
            )
        else:
            plan.append("metadata policy: auto-detect one simulation input per source directory and require consistent resolved settings")
    return plan


def _combine_trajectory_request(
    request: CombineRequest,
    options: TrajectoryConversionOptions | None,
) -> CombineResult:
    from dataclasses import replace
    from .pbc import apply_pbc_to_frames
    from .trajectory.io import read_trajectory, write_trajectory

    combined_frames: list[Any] = []
    with ProgressBar(
        desc="Combining trajectories",
        total=len(request.source_paths),
        unit="file",
    ) as progress:
        for source_path in request.source_paths:
            combined_frames.extend(read_trajectory(source_path))
            progress.update()
    if not combined_frames:
        raise ValueError("No frames were read from the requested trajectory sources.")

    total_frames = len(combined_frames)
    if request.target_file_type == "trajectory_hdf5":
        resolved_metadata, metadata_notes = _resolve_combined_trajectory_metadata(
            request,
            options,
            frames=combined_frames,
        )
        _apply_fixed_constraints_from_conversion_metadata(combined_frames, resolved_metadata)
        if resolved_metadata is not None and resolved_metadata.cell_angstrom is not None:
            combine_cell = resolved_metadata.cell_angstrom
            combined_frames = apply_pbc_to_frames(combined_frames, combine_cell)
            resolved_metadata = _conversion_metadata_with_pbc_cache(
                resolved_metadata,
                cell=combine_cell,
            )
            LOGGER.info(
                "Applied PBC during combine using cell %.6g %.6g %.6g Angstrom.",
                combine_cell[0],
                combine_cell[1],
                combine_cell[2],
            )
        else:
            LOGGER.info("PBC combine cache unavailable: no valid cell metadata found.")
        resolved_metadata = _conversion_metadata_with_surface_cache(
            resolved_metadata,
            combined_frames,
        )
        for note in metadata_notes:
            LOGGER.info("%s", note)
        metadata = replace(
            _trajectory_combine_metadata(request),
            input_path=None if resolved_metadata is None else resolved_metadata.input_path,
            input_format=None if resolved_metadata is None else resolved_metadata.input_format,
            cell_angstrom=None if resolved_metadata is None else resolved_metadata.cell_angstrom,
            cell_source=None if resolved_metadata is None else resolved_metadata.cell_source,
            frame_timestep_fs=(
                None if resolved_metadata is None else resolved_metadata.frame_timestep_fs
            ),
            md_timestep_fs=None if resolved_metadata is None else resolved_metadata.md_timestep_fs,
            trajectory_stride_md=(
                None if resolved_metadata is None else resolved_metadata.trajectory_stride_md
            ),
            timestep_source=None if resolved_metadata is None else resolved_metadata.timestep_source,
            fixed_atom_indices=(
                () if resolved_metadata is None else resolved_metadata.fixed_atom_indices
            ),
            fixed_atoms_source=(
                None if resolved_metadata is None else resolved_metadata.fixed_atoms_source
            ),
            pbc_applied=False if resolved_metadata is None else resolved_metadata.pbc_applied,
            pbc_cell_angstrom=(
                None if resolved_metadata is None else resolved_metadata.pbc_cell_angstrom
            ),
            pbc_source=None if resolved_metadata is None else resolved_metadata.pbc_source,
            coordinate_basis=(
                None if resolved_metadata is None else resolved_metadata.coordinate_basis
            ),
            surface_cache_status=(
                None if resolved_metadata is None else resolved_metadata.surface_cache_status
            ),
            surface_cache_axis=(
                None if resolved_metadata is None else resolved_metadata.surface_cache_axis
            ),
            surface_cache_mode=(
                None if resolved_metadata is None else resolved_metadata.surface_cache_mode
            ),
            surface_cache_elements=(
                None if resolved_metadata is None else resolved_metadata.surface_cache_elements
            ),
            surface_cache_include_fixed_surface_atoms=(
                False
                if resolved_metadata is None
                else resolved_metadata.surface_cache_include_fixed_surface_atoms
            ),
            surface_cache_rough_surface_envelope_A=(
                None
                if resolved_metadata is None
                else resolved_metadata.surface_cache_rough_surface_envelope_A
            ),
            surface_cache_source=(
                None if resolved_metadata is None else resolved_metadata.surface_cache_source
            ),
            surface_cache_unavailable_reason=(
                None
                if resolved_metadata is None
                else resolved_metadata.surface_cache_unavailable_reason
            ),
            surface_cache_estimate=(
                None if resolved_metadata is None else resolved_metadata.surface_cache_estimate
            ),
            combine_total_frames=total_frames,
        )
        output_path = write_trajectory(
            combined_frames,
            request.target_path,
            source_path=request.source_paths[0],
            source_format=request.target_file_type,
            metadata=metadata,
        )
    elif request.target_file_type == "trajectory_xyz":
        _annotate_raw_combined_frames(
            combined_frames,
            request=request,
            total_frames=total_frames,
        )
        output_path = write_trajectory(
            combined_frames,
            request.target_path,
            source_path=request.source_paths[0],
            source_format="xyz",
        )
    else:
        raise ValueError(
            f"Unsupported trajectory combine target file type '{request.target_file_type}'."
        )
    return CombineResult(output_path=output_path)


def _describe_cube_combine_plan(
    request: CombineRequest,
    options: CubeConversionOptions | None,
) -> list[str]:
    del options
    return [
        f"input family: {request.family}",
        f"sources ({len(request.source_paths)}): {', '.join(str(path) for path in request.source_paths)}",
        f"source file types: {', '.join(request.source_file_types)}",
        f"target file type: {request.target_file_type}",
        f"output path: {request.target_path}",
        "ordering policy: input file order is preserved exactly",
        (
            "raw combined cube output is unavailable; LiNaK requires `.cube.h5` for multi-cube containers."
            if not request.conversion_applied
            else "workflow: all cube fields are stored losslessly in one `.cube.h5` container"
        ),
    ]


def _combine_cube_request(
    request: CombineRequest,
    options: CubeConversionOptions | None,
) -> CombineResult:
    del options
    if request.target_file_type != "cube_hdf5":
        raise ValueError(
            "LiNaK does not support a clean lossless non-HDF5 combined cube format. "
            "Use the default `.cube.h5` output and omit `--no-convert`."
        )
    datasets: list[CubeDataset] = []
    with ProgressBar(
        desc="Combining cubes",
        total=len(request.source_paths),
        unit="file",
    ) as progress:
        for source_path in request.source_paths:
            if is_linak_cube_hdf5(source_path):
                datasets.extend(load_cube_datasets(source_path))
            else:
                datasets.append(parse_cube_file(source_path))
            progress.update()
    output_path = save_cube_datasets(
        datasets,
        request.target_path,
        additional_metadata={
            "combine_source_paths": [str(path) for path in request.source_paths],
            "combine_source_file_types": list(request.source_file_types),
            "combine_conversion_applied": True,
            "combine_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "combine_linak_version": __version__,
            "combine_total_fields": len(datasets),
        },
    )
    return CombineResult(output_path=output_path)


def _describe_cube_conversion_plan(
    request: ConversionRequest,
    options: CubeConversionOptions | None,
) -> list[str]:
    del options
    return [
        f"input cube: {request.source_path}",
        f"source family: {request.family}",
        f"source file type: {request.source_file_type}",
        f"target file type: {request.target_file_type}",
        f"output path: {request.target_path}",
        (
            "workflow: parse raw cube field and store it in LiNaK cube HDF5"
            if request.target_file_type == "cube_hdf5"
            else "workflow: load LiNaK cube HDF5 and reconstruct a raw cube file"
        ),
    ]


def _convert_cube_request(
    request: ConversionRequest,
    options: CubeConversionOptions | None,
) -> ConversionResult:
    del options
    datasets = (
        [parse_cube_file(request.source_path)]
        if request.source_file_type == "cube_file"
        else load_cube_datasets(request.source_path)
    )
    if request.target_file_type == "cube_hdf5":
        output_path = save_cube_datasets(
            datasets,
            request.target_path,
            additional_metadata={
                "source_path": str(request.source_path),
                "source_file_type": request.source_file_type,
            },
        )
    elif request.target_file_type == "cube_file":
        if len(datasets) != 1:
            raise ValueError(
                "Cannot reconstruct multiple cube fields into one raw `.cube` file. "
                "Select one profile or keep the `.cube.h5` container."
            )
        output_path = write_cube_file(datasets[0], request.target_path)
    else:
        raise ValueError(f"Unsupported cube target file type '{request.target_file_type}'.")
    return ConversionResult(output_path=output_path)


def _build_default_registry() -> ConversionRegistry:
    from .trajectory.io import default_trajectory_hdf5_output_path, is_linak_trajectory_hdf5

    file_types = (
        ConversionFileType(
            id="trajectory_hdf5",
            family="trajectory",
            label="LiNaK trajectory HDF5",
            suffixes=(".traj.h5", ".traj.hdf5"),
            detector=is_linak_trajectory_hdf5,
            default_output_path_factory=default_trajectory_hdf5_output_path,
            target_aliases=("hdf5", "traj.h5", "trajectory-hdf5"),
        ),
        ConversionFileType(
            id="trajectory_xyz",
            family="trajectory",
            label="XYZ trajectory",
            suffixes=(".xyz", ".extxyz"),
            detector=_xyz_trajectory_detector,
            default_output_path_factory=_default_xyz_output_path,
            target_aliases=("xyz", "extxyz"),
        ),
        ConversionFileType(
            id="cube_hdf5",
            family="cube",
            label="LiNaK cube HDF5",
            suffixes=(".cube.h5", ".cube.hdf5"),
            detector=is_linak_cube_hdf5,
            default_output_path_factory=_default_cube_hdf5_output_path,
            target_aliases=("hdf5", "cube.h5", "cube-hdf5"),
        ),
        ConversionFileType(
            id="cube_file",
            family="cube",
            label="Cube file",
            suffixes=(".cube",),
            default_output_path_factory=_default_cube_output_path,
            target_aliases=("cube",),
        ),
        ConversionFileType(
            id="trajectory_file",
            family="trajectory",
            label="Generic trajectory file",
            suffixes=(),
            detector=_raw_trajectory_detector,
            writable=False,
            default_output_path_factory=default_trajectory_hdf5_output_path,
            target_aliases=("raw", "file"),
        ),
    )
    family_handlers = {
        "trajectory": _FamilyConversionHandler(
            family="trajectory",
            default_target_file_type="trajectory_hdf5",
            convert=_convert_trajectory_request,
            describe_plan=_describe_trajectory_conversion_plan,
        ),
        "cube": _FamilyConversionHandler(
            family="cube",
            default_target_file_type="cube_hdf5",
            convert=_convert_cube_request,
            describe_plan=_describe_cube_conversion_plan,
        ),
    }
    combine_handlers = {
        "trajectory": _FamilyCombineHandler(
            family="trajectory",
            default_target_file_type="trajectory_hdf5",
            raw_target_file_type="trajectory_xyz",
            combine=_combine_trajectory_request,
            describe_plan=_describe_trajectory_combine_plan,
        ),
        "cube": _FamilyCombineHandler(
            family="cube",
            default_target_file_type="cube_hdf5",
            raw_target_file_type=None,
            combine=_combine_cube_request,
            describe_plan=_describe_cube_combine_plan,
        ),
    }
    return ConversionRegistry(
        file_types=file_types,
        family_handlers=family_handlers,
        combine_handlers=combine_handlers,
    )


CONVERSION_REGISTRY = _build_default_registry()
