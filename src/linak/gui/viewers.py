"""Viewer launchers for workspace items."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def open_project_item(path: str | Path) -> None:
    """Open a generated output with the best currently available LiNaK viewer."""

    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() in {".h5", ".hdf5"}:
        subprocess.Popen(
            [sys.executable, "-m", "linak.cli", "plot", str(source_path), "--gui"],
            close_fds=True,
        )
        return

    subprocess.Popen([str(source_path)], shell=True)
