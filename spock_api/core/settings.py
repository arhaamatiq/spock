"""
API Settings Module

Configuration via environment variables using pydantic-settings.
All settings have sensible defaults for development.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    API configuration loaded from environment variables.
    
    Environment variables:
        API_KEY: Required API key for authentication
        API_HOST: Host to bind the server to (default: 0.0.0.0)
        API_PORT: Port to bind the server to (default: 8000)
        CORS_ORIGINS: Comma-separated list of allowed CORS origins
        LOG_LEVEL: Logging level (default: INFO)
        DEBUG: Enable debug mode (default: False)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # ==========================================================================
    # API Server Configuration
    # ==========================================================================
    
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server to",
    )
    
    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind the API server to",
    )
    
    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    
    API_KEY: str = Field(
        default="",
        description="API key for authenticating requests. Required in production.",
    )
    
    API_KEY_HEADER: str = Field(
        default="x-api-key",
        description="Header name for the API key",
    )
    
    # ==========================================================================
    # CORS Configuration
    # ==========================================================================
    
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins (comma-separated in env)",
    )
    
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )
    
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["GET", "POST", "OPTIONS"],
        description="Allowed HTTP methods for CORS",
    )
    
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed headers for CORS",
    )
    
    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | [%(request_id)s] | %(message)s",
        description="Log format string",
    )
    
    # ==========================================================================
    # Application Configuration
    # ==========================================================================
    
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (more verbose errors)",
    )
    
    APP_NAME: str = Field(
        default="Spock API",
        description="Application name for OpenAPI docs",
    )
    
    APP_VERSION: str = Field(
        default="0.1.0",
        description="Application version",
    )
    
    # ==========================================================================
    # RAG Configuration Overrides (optional)
    # ==========================================================================
    
    RAG_RETRIEVAL_K: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override number of documents to retrieve (uses RAG default if None)",
    )
    
    RAG_MAX_HISTORY: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Override max chat history turns (uses RAG default if None)",
    )
    
    # ==========================================================================
    # Streaming Configuration
    # ==========================================================================
    
    STREAM_CHUNK_DELAY_MS: int = Field(
        default=0,
        ge=0,
        description="Artificial delay between stream chunks in ms (for testing)",
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse comma-separated CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @field_validator("LOG_LEVEL", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}, got '{v}'")
        return upper_v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode (API key required)."""
        return bool(self.API_KEY)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance loaded from environment.
    
    Example:
        settings = get_settings()
        print(f"Running on port {settings.API_PORT}")
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()

