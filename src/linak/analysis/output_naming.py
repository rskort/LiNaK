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
