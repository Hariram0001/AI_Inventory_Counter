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
