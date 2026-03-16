"""LiNaK: Lightweight molecular dynamics trajectory analysis toolkit."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

_VERSION_LINE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')


def _read_source_tree_version() -> str | None:
    """Best-effort version lookup from local ``pyproject.toml``."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        lines = pyproject_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    in_project_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project_section = stripped == "[project]"
            continue
        if not in_project_section:
            continue
        match = _VERSION_LINE.match(stripped)
        if match:
            return match.group(1)
    return None


_SOURCE_TREE_VERSION = _read_source_tree_version()
if _SOURCE_TREE_VERSION is not None:
    __version__ = _SOURCE_TREE_VERSION
else:
    try:
        __version__ = version("LiNaK")
    except PackageNotFoundError:  # pragma: no cover - used when running from source tree
        __version__ = "0.0.0"

__all__ = ("__version__",)
