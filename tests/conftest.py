# Apache-2.0
"""The lab's modules live beside the tests, not on the install path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def studio(tmp_path, monkeypatch):
    """A lab rooted in a temp folder, with the worker left asleep."""
    monkeypatch.setenv("EASYDETECT_LAB_HOME", str(tmp_path / "home"))
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import app as module

    module = importlib.reload(module)
    for folder in (module.DATASETS, module.RUNS):
        folder.mkdir(parents=True, exist_ok=True)
    # no context manager: startup never runs, so no worker picks jobs up
    return TestClient(module.app), module
