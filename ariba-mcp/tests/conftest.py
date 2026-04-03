"""Shared test fixtures for ariba-mcp tests."""

import pytest


@pytest.fixture
def ariba_env(monkeypatch):
    """Set required Ariba env vars for testing."""
    monkeypatch.setenv("ARIBA_REALM", "test-realm")
    monkeypatch.setenv("ARIBA_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("ARIBA_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("ARIBA_API_KEY", "test-api-key")
    monkeypatch.setenv("ARIBA_NETWORK_ID", "test-network-id")
