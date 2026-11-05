"""Keep optional Songsterr lookups offline throughout the test suite."""

import pytest


@pytest.fixture(autouse=True)
def disable_web_hints(monkeypatch):
    monkeypatch.setenv("GROOVEMAP_WEB_HINTS", "0")
