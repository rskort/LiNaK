"""Verify built archives and a --target wheel installation outside the checkout.

Usage: python benchmarks/verify_distribution.py DIST_DIR WHEEL_INSTALL_DIR
Install the wheel with pip --target or uv pip --target first, without dependencies;
run this checker in a fresh environment that already has LiNaK's dependencies.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile


def _project_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"Could not read the project version from '{pyproject}'.")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("install_dir", type=Path)
    args = parser.parse_args()
    wheel = next(args.dist_dir.glob("*.whl"))
    sdist = next(args.dist_dir.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata["License-Expression"] == "Apache-2.0"
        assert metadata["Version"] == _project_version()
        assert metadata["Requires-Python"] == ">=3.9"
        assert any(n.endswith("/licenses/LICENSE") for n in names)
        assert "linak/temporary_resources.py" in names
        assert not any("msd" in n.lower() for n in names)
        for asset in ("logo_simple.svg", "linak_gui_banner.svg", "screenshot_gui.png"):
            assert any(n.endswith("/assets/" + asset) for n in names)
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(n.endswith("/LICENSE") for n in names)
        assert not any("msd" in n.lower() for n in names)

    install_dir = args.install_dir.resolve(strict=True)
    env = dict(
        os.environ, PYTHONPATH=str(install_dir), QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg"
    )
    code = """
import json, sys
from pathlib import Path
import linak
from linak.cli import main
from linak.plot.plot_gui import _resolve_asset_path
from PySide6.QtWidgets import QApplication
from PySide6.QtSvg import QSvgRenderer
expected = Path(sys.argv[1]).resolve()
assert expected in Path(linak.__file__).resolve().parents
app = QApplication([])
for name in ('logo_simple.svg', 'linak_gui_banner.svg'):
    path = _resolve_asset_path(name)
    assert path.is_file(), path
    assert expected in path.resolve().parents, path
    assert QSvgRenderer(str(path)).isValid(), path
assert main(['--version']) == 0
print(json.dumps({'installed_module': linak.__file__, 'version': linak.__version__, 'assets': 'passed'}))
"""
    with TemporaryDirectory(prefix="linak-wheel-smoke-") as directory:
        subprocess.run(
            [sys.executable, "-c", code, str(install_dir)], cwd=directory, env=env, check=True
        )
    print(json.dumps({"wheel": str(wheel), "sdist": str(sdist), "archive_checks": "passed"}))


if __name__ == "__main__":
    main()
