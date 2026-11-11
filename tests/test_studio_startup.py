"""Real event-loop startup, distinct from prepared CLI callback tests."""

from __future__ import annotations

import threading
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pytest

import linak.cli as cli
from linak.cache import PREVIEW_CACHE
from linak.cancellation import cancellation_scope, ComputationCancelled
from linak.plot.gui_session import (
    PlotGuiSession,
    PlotProgress,
    prepare_session,
    preparing_session,
    own_preparation_output,
    report_progress,
)
from linak.plot.setting_keys import copy_settings


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    application = widgets.QApplication.instance() or widgets.QApplication([])
    application.setQuitOnLastWindowClosed(True)
    PREVIEW_CACHE.clear()
    return application


def _source(path, analysis):
    from tests.test_cli import (
        _write_density_hdf5,
        _write_position_hdf5,
        _write_coordination_hdf5,
        _write_potential_hdf5,
    )

    writers = dict(
        density=_write_density_hdf5,
        position=_write_position_hdf5,
        coordination=_write_coordination_hdf5,
        potential=_write_potential_hdf5,
    )
    if analysis in writers:
        writers[analysis](path)
    elif analysis == "orientation":
        from tests.test_orientation_alignment import _frame, _profile
        from linak.analysis.orientation import save_orientation_profile

        save_orientation_profile(_profile([_frame()]), path)
    elif analysis == "rdf":
        from linak.analysis.rdf import RDFProfile, save_rdf_profile

        save_rdf_profile(
            RDFProfile(
                "O", "H", np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5]), np.array([1.0, 2.0]), 2
            ),
            path,
        )
    else:
        from linak.analysis.temperature import TemperatureProfile, save_temperature_profiles

        values = np.arange(3.0)
        save_temperature_profiles(
            [
                TemperatureProfile(
                    "O",
                    "element",
                    values.astype(int),
                    values.astype(int),
                    values,
                    values / 1000,
                    values + 300.0,
                    3,
                    "test",
                    element="O",
                )
            ],
            path,
        )


def _poll_until_closed(app, inspect, *, timeout=20):
    """Return failures to pytest, rather than losing assertions in Qt callbacks."""
    from PySide6.QtCore import QTimer

    errors = []
    started = perf_counter()
    ticks = []

    def poll():
        windows = [
            w for w in app.topLevelWidgets() if hasattr(w, "_session_ready") and w.isVisible()
        ]
        if windows:
            window = windows[0]
            ticks.append(1)
            try:
                if perf_counter() - started > timeout:
                    raise AssertionError("Studio startup timed out")
                if inspect(window):
                    if window._session_ready:
                        window._saved_signature = window._signature(window._collect_settings())
                    window.close()
                    return
            except BaseException as error:
                errors.append(error)
                if window._session_ready:
                    window._saved_signature = window._signature(window._collect_settings())
                window.close()
                return
        QTimer.singleShot(10, poll)

    QTimer.singleShot(0, poll)
    return errors, ticks


@pytest.mark.parametrize(
    "analysis",
    ["density", "orientation", "position", "rdf", "coordination", "potential", "temperature"],
)
@pytest.mark.parametrize("count", [1, 2])
def test_paints_before_io_and_combining_then_renders_once(
    app, tmp_path, monkeypatch, analysis, count
):
    from PySide6.QtWidgets import QMainWindow

    paths = [tmp_path / f"source{i}.{analysis}.h5" for i in range(count)]
    for path in paths:
        _source(path, analysis)
    painted = threading.Event()
    reads, renders = [], []
    original_paint, original_read = QMainWindow.paintEvent, h5py.Dataset.__getitem__
    original_direct = h5py.Dataset.read_direct
    original_render = cli._render_profile_plot

    def paint(window, event):
        original_paint(window, event)
        if hasattr(window, "_session_ready"):
            painted.set()

    def check_read(dataset):
        assert painted.is_set(), "Source data was read before the main window painted"
        # Combined-source metadata is intentionally allowed in the header catalog.
        if (
            count == 1
            and not dataset.name.startswith("/combined_sources/")
            and not dataset.name.endswith("/atom_indices")
        ):
            assert not preparing_session(), f"Catalog loaded numerical data: {dataset.name}"
            assert threading.current_thread() is not threading.main_thread()
        reads.append(dataset.name)

    def read(dataset, *args, **kwargs):
        check_read(dataset)
        return original_read(dataset, *args, **kwargs)

    def direct(dataset, *args, **kwargs):
        check_read(dataset)
        return original_direct(dataset, *args, **kwargs)

    def render(**kwargs):
        assert painted.is_set()
        assert not preparing_session()
        renders.append(True)
        return original_render(**kwargs)

    monkeypatch.setattr(QMainWindow, "paintEvent", paint)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    monkeypatch.setattr(h5py.Dataset, "read_direct", direct)
    monkeypatch.setattr(cli, "_render_profile_plot", render)
    original_combine = cli._combine_analysis_hdf5_sources

    def combine(**kwargs):
        assert painted.is_set()
        return original_combine(**kwargs)

    monkeypatch.setattr(cli, "_combine_analysis_hdf5_sources", combine)

    def inspect(window):
        assert not window._startup_failed, window._startup_label.text()
        if window._session_ready and getattr(window, "_preview_error", None):
            raise AssertionError(window._preview_error)
        return window._session_ready and window._preview_figure is not None

    errors, ticks = _poll_until_closed(app, inspect)
    result = cli.main(["--log-level", "ERROR", "plot", "--files", *map(str, paths), "--gui"])
    if errors:
        raise errors[0]
    assert result == 0
    assert len(ticks) > 1
    assert reads
    assert len(renders) == 1


class _NoCopy:
    def __deepcopy__(self, memo):
        raise AssertionError("A runtime object entered settings copying")


def test_render_objects_are_removed_before_copying_nested_settings():
    values: dict[str, Any] = dict(
        title="Manual title",
        figure=_NoCopy(),
        axes=_NoCopy(),
        line_artists=[_NoCopy()],
        plotted_heatmap=_NoCopy(),
        orientation_view_states={
            "plot_1d_line": {
                "title": "Saved",
                "figure": _NoCopy(),
                "series_artists": _NoCopy(),
            }
        },
    )
    copied = copy_settings(values, gui=True)
    assert copied == {
        "title": "Manual title",
        "orientation_view_states": {"plot_1d_line": {"title": "Saved"}},
    }
    merged = cli._merge_preview_defaults_into_gui_settings(
        {"title": "Manual title", "_gui_sync_modes": {"title": "manual"}}, values
    )
    assert "figure" not in merged and "line_artists" not in merged
    assert merged["title"] == "Manual title"
    from linak.plot.profile_persistence import (
        build_plot_profile_payload,
        flatten_plot_profile_payload,
    )

    values["orientation_view_states"]["plot_1d_line"]["_orientation_view_state_initialized"] = True
    saved = build_plot_profile_payload("plot:orientation", values)
    restored = flatten_plot_profile_payload("plot:orientation", saved)
    assert restored["orientation_view_states"]["plot_1d_line"][
        "_orientation_view_state_initialized"
    ]
    assert "figure" not in saved["style"]


def test_saved_spatial_view_alignment_and_manual_title_survive_deferred_startup(app, tmp_path):
    from tests.test_orientation_alignment import _frame, _profile, _setting
    from linak.analysis.orientation import save_orientation_profile
    from linak.analysis.orientation_spatial import spatial_orientation_map
    from linak.plot.plot_settings import write_plot_profile
    from linak.plot.profile_persistence import (
        build_plot_profile_payload,
        serialize_plot_view_mapping,
    )
    from linak.plot.mappings.orientation_mapping import orientation_plot_options_to_view_mapping

    profile = _profile([_frame()])
    source = tmp_path / "aligned.orientation.h5"
    save_orientation_profile(profile, source)
    alignment = _setting(profile, translation=(0.173, 0.267))
    settings = {
        "title": "Saved manual title",
        "_gui_sync_modes": {"title": "manual"},
        "interface_alignments": alignment,
        "orientation_filter_distance_min": 3.0,
        "orientation_filter_distance_max": 6.0,
        "orientation_heatmap_kind": "spatial",
        "orientation_heatmap_quantity": "density-weighted",
        "view_mapping": serialize_plot_view_mapping(
            orientation_plot_options_to_view_mapping(
                orientation_heatmap_kind="spatial", orientation_heatmap_quantity="density-weighted"
            )
        ),
    }
    write_plot_profile(
        source, "plot:orientation", build_plot_profile_payload("plot:orientation", settings)
    )
    expected = spatial_orientation_map(
        [profile],
        quantity="density-weighted",
        interface_alignments=alignment,
        filters={"distance": (3.0, 6.0)},
    )

    def inspect(window):
        assert not window._startup_failed, window._startup_label.text()
        if window._session_ready and window._preview_error:
            raise AssertionError(window._preview_error)
        if window._session_ready and window._preview_figure is not None:
            current = window._collect_settings()
            assert window.title_text.text() == "Saved manual title"
            assert current["interface_alignments"] == alignment
            assert current["orientation_heatmap_kind"] == "spatial"
            assert current["orientation_heatmap_quantity"] == "density-weighted"
            np.testing.assert_allclose(
                window._last_preview_state["plotted_heatmap"]["values"],
                expected.values,
                equal_nan=True,
            )
            assert window._signature(current) == window._saved_signature
            return True
        return False

    errors, _ = _poll_until_closed(app, inspect)
    assert cli.main(["--log-level", "ERROR", "plot", str(source), "--gui"]) == 0
    if errors:
        raise errors[0]


def test_cancelled_preparation_only_removes_its_owned_output(tmp_path):
    source, owned = tmp_path / "source.h5", tmp_path / "combined.h5"
    source.write_bytes(b"source")
    cancelled: list[bool] = []

    def build():
        owned.write_bytes(b"owned")
        own_preparation_output(owned)
        cancelled.append(True)
        report_progress("Preparing combined results")

    with pytest.raises(ComputationCancelled), cancellation_scope(lambda: bool(cancelled)):
        prepare_session(build)
    assert source.read_bytes() == b"source"
    assert not owned.exists()


def test_retry_and_stale_startup_messages(app):
    from linak.plot.plot_gui import launch_plot_settings_panel

    attempts = []

    def load():
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError("Broken input")
        return PlotGuiSession(
            "Ready",
            {"series_count": 0},
            {"on_preview": lambda settings: {}, "on_save": lambda *a: ""},
            {},
        )

    def inspect(window):
        if window._startup_failed:
            assert "Broken input" in window._startup_label.text()
            window._start_preparation()
            generation = window._startup_generation
            window._startup_progress(generation - 1, PlotProgress("Stale progress"))
            assert window._startup_label.text() != "Stale progress"
        return window._session_ready

    errors, _ = _poll_until_closed(app, inspect)
    result = launch_plot_settings_panel(
        title="Loading",
        initial_settings={},
        on_preview=lambda settings: {},
        on_save=lambda *a: "",
        session_loader=load,
    )
    if errors:
        raise errors[0]
    assert result == 0 and len(attempts) == 2


def test_closing_during_preparation_discards_late_session(app, tmp_path):
    from linak.plot.plot_gui import launch_plot_settings_panel

    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    owned = tmp_path / "late.h5"

    def load():
        started.set()
        release.wait(5)
        owned.write_bytes(b"owned")
        session = PlotGuiSession("Late", {}, {}, {}, (lambda: owned.unlink(missing_ok=True),))
        finished.set()
        return session

    def inspect(window):
        return started.is_set()

    errors, _ = _poll_until_closed(app, inspect)
    result = launch_plot_settings_panel(
        title="Loading",
        initial_settings={},
        on_preview=lambda settings: {},
        on_save=lambda *a: "",
        session_loader=load,
    )
    release.set()
    assert finished.wait(5)
    # Worker cleanup follows the late loader return.
    deadline = perf_counter() + 5
    while owned.exists() and perf_counter() < deadline:
        threading.Event().wait(0.01)
    assert not errors and result == 0 and not owned.exists()


def test_source_width_validation_accepts_storage_roundoff_only():
    from linak.plot.plotting import _validate_section_width_request

    values = np.array([0.05, 0.15, 0.25], dtype=np.float32).astype(float)
    _validate_section_width_request(values, bin_width=0.1)
    with pytest.raises(ValueError, match="smaller than the data bin width"):
        _validate_section_width_request(values, bin_width=0.09)


def test_cancel_before_atomic_publish_preserves_existing_file(tmp_path):
    from linak.storage.hdf5_utils import _atomic_hdf5_target

    target = tmp_path / "original.h5"
    target.write_bytes(b"original")
    cancelled: list[bool] = []
    with pytest.raises(ComputationCancelled), cancellation_scope(lambda: bool(cancelled)):
        with _atomic_hdf5_target(target) as temporary:
            temporary.write_bytes(b"new")
            cancelled.append(True)
    assert target.read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["original.h5"]
