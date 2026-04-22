"""
Logging Configuration for Spock AI RAG System

This module sets up structured logging with configurable levels.
Logs include timestamp, level, module name, and message.

Usage:
    from spock_rag.logging_config import setup_logging, get_logger
    
    setup_logging(level="DEBUG")  # Call once at startup
    logger = get_logger(__name__)
    logger.info("Starting application...")
"""

import logging
import sys
from typing import Optional


# Default log format with timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the entire application.
    
    This should be called once at application startup, before any logging occurs.
    
    Args:
        level: Logging level as string. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
               Default is INFO.
        format_string: Custom log format string. Uses Python logging format codes.
                      Default shows timestamp, level, module name, and message.
        date_format: Custom date format for timestamps. Default is ISO-like format.
    
    Example:
        # Basic setup
        setup_logging()
        
        # Debug mode with more verbose output
        setup_logging(level="DEBUG")
        
        # Custom format
        setup_logging(format_string="%(levelname)s: %(message)s")
    """
    # Use defaults if not specified
    format_string = format_string or DEFAULT_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler (output to stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set specific levels for noisy third-party libraries
    # These libraries can be very verbose at DEBUG level
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    This is a simple wrapper around logging.getLogger that ensures
    consistent logger naming across the application.
    
    Args:
        name: The name for the logger, typically __name__ of the calling module.
    
    Returns:
        A configured Logger instance.
    
    Example:
        # In any module:
        from spock_rag.logging_config import get_logger
        logger = get_logger(__name__)
        
        logger.debug("Detailed debugging info")
        logger.info("Normal operation info")
        logger.warning("Something unexpected happened")
        logger.error("An error occurred")
    """
    return logging.getLogger(name)


# =============================================================================
# Convenience functions for quick logging level checks
# =============================================================================


def set_debug_mode() -> None:
    """Enable debug logging for all spock_rag modules."""
    logging.getLogger("spock_rag").setLevel(logging.DEBUG)


def set_quiet_mode() -> None:
    """Only show warnings and errors."""
    logging.getLogger("spock_rag").setLevel(logging.WARNING)

