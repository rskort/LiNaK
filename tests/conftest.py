from __future__ import annotations

from pathlib import Path

import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def prepared_studio(monkeypatch):
    """Exercise existing CLI callback tests at their prepared-session boundary.

    Full asynchronous startup and Qt event ordering have dedicated integration
    tests. These tests replace the GUI presenter and inspect its callbacks.
    """
    import linak.cli as cli
    from copy import deepcopy

    def launch(args):
        return args.handler(deepcopy(args))

    monkeypatch.setattr(cli, "_launch_studio", launch)
