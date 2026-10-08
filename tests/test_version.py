from __future__ import annotations

from pathlib import Path
import re

import linak
from linak.analysis.schema import get_analysis_schema


def _pyproject_version() -> str:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = project_file.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', content)
    if match is None:
        raise AssertionError("Could not find project version in pyproject.toml")
    return match.group(1)


def test_package_version_matches_pyproject() -> None:
    assert linak.__version__ == _pyproject_version()


def test_registered_analysis_schema_versions() -> None:
    for analysis in ("density", "position", "coordination", "orientation"):
        assert get_analysis_schema(analysis).version == 2
    for analysis in ("rdf", "temperature"):
        assert get_analysis_schema(analysis).version == 1
