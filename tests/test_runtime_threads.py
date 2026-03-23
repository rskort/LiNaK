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
    assert result.applied == {}
    assert env == {"OPENBLAS_NUM_THREADS": "8"}


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
