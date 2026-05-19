"""Project-workspace GUI support for LiNaK."""

from __future__ import annotations

__all__ = ("launch_project_workspace",)


def launch_project_workspace(project_dir: str) -> None:
    """Open the LiNaK project workspace GUI."""

    from .workspace import launch_project_workspace as _launch_project_workspace

    _launch_project_workspace(project_dir)
