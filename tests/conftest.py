"""
Pytest Configuration and Shared Fixtures

This file contains fixtures and configuration shared across all tests.
"""

import os
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the config singleton before each test."""
    from spock_rag.config import reset_settings
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set a mock OpenAI API key for testing config."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

