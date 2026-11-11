from __future__ import annotations

from linak.runtime_threads import configure_native_thread_env


def test_configure_native_thread_env_applies_default_single_thread_caps():
    env: dict[str, str] = {}

    result = configure_native_thread_env(env)

    assert result.skipped_reason is None
    assert result.requested_threads == 1
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["BLIS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"


def test_configure_native_thread_env_respects_preconfigured_backend_env():
    env = {"OPENBLAS_NUM_THREADS": "8"}

    result = configure_native_thread_env(env)

    assert result.skipped_reason == "preconfigured"
    assert env["OPENBLAS_NUM_THREADS"] == "8"
    assert "OPENBLAS_NUM_THREADS" not in result.applied
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"


def test_configure_native_thread_env_respects_explicit_disable():
    env = {"LINAK_DISABLE_THREAD_CAP": "true"}

    result = configure_native_thread_env(env)

    assert result.skipped_reason == "disabled"
    assert result.applied == {}
    assert env == {"LINAK_DISABLE_THREAD_CAP": "true"}


def test_configure_native_thread_env_uses_linak_num_threads_when_valid():
    env = {"LINAK_NUM_THREADS": "3"}

    result = configure_native_thread_env(env)

    assert result.skipped_reason is None
    assert result.requested_threads == 3
    assert all(
        env[key] == "3"
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    )


def test_configure_native_thread_env_ignores_invalid_linak_num_threads():
    env = {"LINAK_NUM_THREADS": "abc"}

    result = configure_native_thread_env(env)

    assert result.skipped_reason == "invalid_linak_num_threads"
    assert result.invalid_value == "abc"
    assert env == {"LINAK_NUM_THREADS": "abc"}


def test_cli_configures_native_thread_env_before_numpy_import():
    import subprocess
    import sys

    script = """
import sys
import linak.runtime_threads as runtime
original = runtime.configure_native_thread_env
def checked(*args, **kwargs):
    assert 'numpy' not in sys.modules, 'NumPy imported before native bootstrap'
    assert 'ase' not in sys.modules, 'ASE imported before native bootstrap'
    return original(*args, **kwargs)
runtime.configure_native_thread_env = checked
import linak.cli
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
