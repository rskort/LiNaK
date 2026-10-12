from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from linak.cancellation import ComputationCancelled, cancellation_scope, check_cancelled
from linak.progress import ProgressBar, progress_stage
from linak.storage.hdf5_utils import read_linak_hdf5, write_linak_hdf5


def test_cancellation_is_thread_local_and_scope_is_restored():
    cancel = threading.Event()
    other_thread = []
    with cancellation_scope(cancel.is_set):
        check_cancelled()
        cancel.set()

        def run_other_thread():
            check_cancelled()
            other_thread.append(True)

        thread = threading.Thread(target=run_other_thread)
        thread.start()
        thread.join(timeout=2)
        assert other_thread == [True]
        with pytest.raises(ComputationCancelled):
            check_cancelled()
    check_cancelled()


def test_cancelled_progress_closes_without_reporting_an_error(caplog):
    cancel = threading.Event()
    logger = logging.getLogger("linak.preview.cancellation")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with pytest.raises(ComputationCancelled), cancellation_scope(cancel.is_set):
            with (
                progress_stage(logger, "Preview data"),
                ProgressBar(desc="Preview data", total=10, enabled=False) as bar,
            ):
                cancel.set()
                bar.update()
    assert bar._closed
    assert "cancelled after" in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_cancelled_hdf5_read_closes_handles_and_preserves_source(tmp_path):
    path = write_linak_hdf5(
        tmp_path / "input.density.h5", analysis="density", datasets={"x": np.arange(20.0)}
    )
    original = path.read_bytes()
    with pytest.raises(ComputationCancelled), cancellation_scope(lambda: True):
        read_linak_hdf5(path, expected_analysis="density")
    assert path.read_bytes() == original
    # This rename would fail on Windows if the cancelled reader kept its handle.
    path.rename(tmp_path / "moved.density.h5")


def test_density_geometry_projection_cancels_inside_large_grid():
    from linak.analysis.density import _generic_geometry_volumes
    from types import SimpleNamespace

    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(ComputationCancelled), cancellation_scope(cancelled):
        _generic_geometry_volumes(
            SimpleNamespace(surface_axis="z"),
            {"cell": np.eye(3) * 10, "origin": np.zeros(3), "shift": 0.0},
            ("x", "y"),
            (np.linspace(0, 10, 100), np.linspace(0, 10, 100)),
            {},
        )
    assert checks == 4
