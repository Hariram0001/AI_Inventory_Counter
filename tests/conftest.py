"""Pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make shared test helpers (e.g. secret_scan) importable from every test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: tests that call the live Roboflow API (opt-in only)",
    )


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point DATA_DIR (and therefore the database) at a throwaway directory.

    The repo `.env` is loaded with override=True, so it must be disabled first
    or it would stamp the real DATA_DIR back into the environment.
    """
    yield from _isolate(tmp_path, monkeypatch)


def _isolate(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BOOTSTRAP_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("BOOTSTRAP_ADMIN_EMAIL", raising=False)
    config.reload_settings()
    try:
        yield tmp_path
    finally:
        config.reload_settings()
