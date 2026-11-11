"""Run native environment setup before importing any numerical CLI dependencies."""

from .runtime_threads import configure_native_thread_env

NATIVE_THREAD_ENV_CONFIGURATION = configure_native_thread_env()
