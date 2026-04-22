"""
Configuration Module for Spock AI RAG System

This module handles all configuration loading from environment variables.
It provides a type-safe Settings dataclass with sensible defaults.

Usage:
    from spock_rag.config import get_settings
    settings = get_settings()
    print(settings.OPENAI_MODEL)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from .env file (if it exists)
load_dotenv()


@dataclass
class Settings:
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults except OPENAI_API_KEY which is required.
    """
    
    # ==========================================================================
    # OpenAI Configuration
    # ==========================================================================
    
    # Required: Your OpenAI API key
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # The chat model to use for generating responses
    # gpt-4o-mini is cost-effective and fast with good quality
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # The embedding model for vectorizing documents
    # text-embedding-3-small is efficient and cost-effective
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    
    # ==========================================================================
    # Vector Store Configuration
    # ==========================================================================
    
    # Directory where ChromaDB persists its data
    # This allows the knowledge base to survive restarts
    PERSIST_DIR: Path = field(
        default_factory=lambda: Path(os.getenv("PERSIST_DIR", "./chroma_store"))
    )
    
    # ==========================================================================
    # Text Splitting Configuration
    # ==========================================================================
    
    # Maximum size of each text chunk (in characters)
    # Larger chunks = more context but less precise retrieval
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    
    # Overlap between consecutive chunks (in characters)
    # Overlap helps preserve context across chunk boundaries
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    
    # ==========================================================================
    # Retrieval Configuration
    # ==========================================================================
    
    # Number of relevant documents to retrieve for each query
    # Higher = more context but potentially more noise
    RETRIEVAL_K: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4"))
    )
    
    # Minimum relevance score (0.0 to 1.0) for retrieved documents
    # Documents below this threshold are filtered out
    MIN_RELEVANCE_SCORE: float = field(
        default_factory=lambda: float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))
    )
    
    # ==========================================================================
    # Session Configuration
    # ==========================================================================
    
    # Maximum number of conversation turns to remember
    # Each turn = 1 user message + 1 assistant response
    MAX_HISTORY: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY", "10"))
    )
    
    def validate(self) -> None:
        """
        Validate that all required settings are present and valid.
        
        Raises:
            ValueError: If required settings are missing or invalid.
        """
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Set it in your .env file or as an environment variable."
            )
        
        if self.CHUNK_SIZE <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {self.CHUNK_SIZE}")
        
        if self.CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP must be non-negative, got {self.CHUNK_OVERLAP}")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        
        if self.RETRIEVAL_K <= 0:
            raise ValueError(f"RETRIEVAL_K must be positive, got {self.RETRIEVAL_K}")
        
        if self.MAX_HISTORY <= 0:
            raise ValueError(f"MAX_HISTORY must be positive, got {self.MAX_HISTORY}")
        
        if not 0.0 <= self.MIN_RELEVANCE_SCORE <= 1.0:
            raise ValueError(
                f"MIN_RELEVANCE_SCORE must be between 0.0 and 1.0, "
                f"got {self.MIN_RELEVANCE_SCORE}"
            )


# Module-level cached settings instance
_settings: Optional[Settings] = None


def get_settings(validate: bool = True) -> Settings:
    """
    Get the application settings (cached singleton).
    
    Args:
        validate: If True, validate settings on first load. Default is True.
    
    Returns:
        The Settings instance with all configuration values.
    
    Raises:
        ValueError: If validate=True and settings are invalid.
    
    Example:
        settings = get_settings()
        print(f"Using model: {settings.OPENAI_MODEL}")
    """
    global _settings
    
    if _settings is None:
        _settings = Settings()
        if validate:
            _settings.validate()
    
    return _settings


def reset_settings() -> None:
    """
    Reset the cached settings instance.
    
    Useful for testing or when environment variables change.
    """
    global _settings
    _settings = None

