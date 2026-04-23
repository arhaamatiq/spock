"""
Tests for API settings parsing.

These tests verify Railway-friendly environment parsing for list settings.
"""

from spock_api.core.settings import Settings


class TestApiSettings:
    """Tests for pydantic-settings env parsing behavior."""

    def test_cors_origins_accepts_comma_separated_env(self):
        """Railway-style CSV env values should parse into a list."""
        settings = Settings(
            _env_file=None,
            CORS_ORIGINS="https://app.example.com,https://www.example.com",
        )

        assert settings.CORS_ORIGINS == [
            "https://app.example.com",
            "https://www.example.com",
        ]

    def test_cors_origins_accepts_json_array_env(self):
        """JSON array strings should still work for compatibility."""
        settings = Settings(
            _env_file=None,
            CORS_ORIGINS='["https://app.example.com", "https://www.example.com"]',
        )

        assert settings.CORS_ORIGINS == [
            "https://app.example.com",
            "https://www.example.com",
        ]

    def test_cors_allow_methods_accepts_comma_separated_env(self):
        """Other list settings should support the same parsing behavior."""
        settings = Settings(
            _env_file=None,
            CORS_ALLOW_METHODS="GET,POST,OPTIONS",
        )

        assert settings.CORS_ALLOW_METHODS == ["GET", "POST", "OPTIONS"]
