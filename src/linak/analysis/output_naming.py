"""Default output naming for LiNaK analysis files."""

from __future__ import annotations

from pathlib import Path
import re

_CP2K_TRAJECTORY_RE = re.compile(r"-(?:pos|vel)-\d+(?:\.traj)?\.xyz$", re.IGNORECASE)
_CP2K_TEMPERATURE_RE = re.compile(r"-\d+\.(?:temp|tregion)$", re.IGNORECASE)

_ANALYSIS_SOURCE_SUFFIXES = (
    ".out.hdf5",
    ".out.h5",
    ".traj.hdf5",
    ".traj.h5",
    ".cube.hdf5",
    ".cube.h5",
    ".hdf5",
    ".h5",
)

_HDF5_SUFFIXES = (".hdf5", ".h5")


def analysis_source_base(source: str | Path, *, default: str = "source") -> str:
    """Return the source stem used for default analysis HDF5 filenames."""

    name = Path(source).name
    lower = name.lower()
    for suffix in _ANALYSIS_SOURCE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)] or default
    if lower.endswith("-v_hartree-1_0.cube"):
        return name[: -len("-v_hartree-1_0.cube")] or default

    stripped = _CP2K_TRAJECTORY_RE.sub("", name)
    if stripped != name:
        return stripped or default
    stripped = _CP2K_TEMPERATURE_RE.sub("", name)
    if stripped != name:
        return stripped or default
    return Path(name).stem or default


def analysis_hdf5_filename(source: str | Path, analysis: str, *, default: str = "source") -> str:
    """Return the default HDF5 filename for one analysis and source."""

    base = analysis_source_base(source, default=default)
    analysis_token = str(analysis).strip().lower() or "analysis"
    return f"{base}.{analysis_token}.h5"


def combined_analysis_hdf5_filename(analysis: str, *, base: str = "linak_combined") -> str:
    """Return the default HDF5 filename for combined profiles of one analysis."""

    analysis_token = str(analysis).strip().lower() or "analysis"
    base_token = str(base).strip() or "linak_combined"
    return f"{base_token}.{analysis_token}.h5"


def ensure_analysis_hdf5_path(path: str | Path, analysis: str) -> Path:
    """Return *path* with an invariant ``.<analysis>.h5/.hdf5`` ending.

    The analysis token is part of the file type, not part of the user-editable
    base name.  For example, ``sample-all.h5`` becomes
    ``sample-all.position.h5`` for the position analysis.
    """

    path_obj = Path(path)
    analysis_token = str(analysis).strip().lower() or "analysis"
    lower_name = path_obj.name.lower()
    hdf5_suffix = next(
        (item for item in _HDF5_SUFFIXES if lower_name.endswith(item)),
        ".h5",
    )
    expected_ending = f".{analysis_token}{hdf5_suffix}"
    if lower_name.endswith(expected_ending):
        return path_obj
    if lower_name.endswith(hdf5_suffix):
        base_name = path_obj.name[: -len(hdf5_suffix)]
    else:
        base_name = path_obj.name
    return path_obj.with_name(f"{base_name}.{analysis_token}{hdf5_suffix}")


def split_analysis_hdf5_name(name: str | Path) -> tuple[str, str, str] | None:
    """Split '<base>.<analysis>.h5/.hdf5' into base, analysis, suffix."""

    path_name = Path(name).name
    lower_name = path_name.lower()
    suffix = next((item for item in _HDF5_SUFFIXES if lower_name.endswith(item)), None)
    if suffix is None:
        return None
    without_suffix = path_name[: -len(suffix)]
    if "." not in without_suffix:
        return None
    base, analysis = without_suffix.rsplit(".", 1)
    if not base or not analysis:
        return None
    return base, analysis, path_name[-len(suffix) :]


def numbered_hdf5_path(path: str | Path, index: int) -> Path:
    """Return an auto-versioned HDF5 path, keeping analysis suffixes at the end."""

    path_obj = Path(path)
    split = split_analysis_hdf5_name(path_obj.name)
    if split is not None:
        base, analysis, suffix = split
        return path_obj.with_name(f"{base}_{index}.{analysis}{suffix}")
    return path_obj.with_name(f"{path_obj.stem}_{index}{path_obj.suffix}")


def append_hdf5_name_suffix(path: str | Path, suffix: str) -> Path:
    """Append a base-name suffix without changing a compound HDF5 file type."""

    path_obj = Path(path)
    split = split_analysis_hdf5_name(path_obj.name)
    if split is not None:
        base, analysis, hdf5_suffix = split
        return path_obj.with_name(f"{base}{suffix}.{analysis}{hdf5_suffix}")
    lower_name = path_obj.name.lower()
    hdf5_suffix = next(
        (item for item in _HDF5_SUFFIXES if lower_name.endswith(item)),
        None,
    )
    if hdf5_suffix is not None:
        base = path_obj.name[: -len(hdf5_suffix)]
        return path_obj.with_name(f"{base}{suffix}{path_obj.name[-len(hdf5_suffix):]}")
    return path_obj.with_name(f"{path_obj.stem}{suffix}{path_obj.suffix}")
