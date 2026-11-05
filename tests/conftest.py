"""Keep optional Songsterr lookups offline, and let TestClient's host name through the server's host check."""

import pytest

from stickmap import server


@pytest.fixture(autouse=True)
def disable_web_hints(monkeypatch):
    monkeypatch.setenv("STICKMAP_WEB_HINTS", "0")


@pytest.fixture(autouse=True)
def allow_test_client_host(monkeypatch):
    monkeypatch.setattr(server, "ALLOWED_HOSTS", server.LOOPBACK_HOSTS | {"testserver"})
