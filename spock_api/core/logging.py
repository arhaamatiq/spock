"""
Logging Module

Structured logging with request ID tracking for the Spock API.
"""

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from spock_api.core.settings import get_settings


# Context variable for request ID - thread/async safe
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """
    Get the current request ID from context.
    
    Returns:
        The request ID string, or "-" if not in a request context.
    """
    return request_id_ctx.get()


def set_request_id(request_id: str) -> None:
    """
    Set the request ID in the current context.
    
    Args:
        request_id: The request ID to set.
    """
    request_id_ctx.set(request_id)


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        A short unique identifier (first 8 chars of UUID4).
    """
    return str(uuid.uuid4())[:8]


class RequestIdFilter(logging.Filter):
    """
    Logging filter that adds request_id to log records.
    
    This allows the request ID to be included in log format strings
    via %(request_id)s.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record."""
        record.request_id = get_request_id()
        return True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Assigns a unique request ID to each request
    2. Logs request start/end with latency
    3. Adds request ID to response headers
    """
    
    def __init__(self, app: Callable, logger: Optional[logging.Logger] = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger("spock_api.request")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with logging and request ID tracking."""
        # Check for existing request ID in header (for tracing)
        request_id = request.headers.get("x-request-id") or generate_request_id()
        
        # Set in context for this request
        set_request_id(request_id)
        
        # Store in request state for access in routes
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.perf_counter()
        self.logger.info(
            f"Request started: {request.method} {request.url.path}"
        )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Log request end
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} latency={latency_ms:.2f}ms"
            )
            
            # Add request ID to response headers
            response.headers["x-request-id"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Log error
            self.logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={type(e).__name__} latency={latency_ms:.2f}ms"
            )
            raise
        
        finally:
            # Reset request ID after request completes
            set_request_id("-")


def setup_logging() -> None:
    """
    Configure logging for the API application.
    
    Sets up:
    - Root logger with appropriate level
    - Console handler with formatted output
    - Request ID filter for all loggers
    - Suppresses noisy third-party loggers
    """
    settings = get_settings()
    
    # Create formatter
    formatter = logging.Formatter(
        settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)
    
    # Configure spock_api logger
    api_logger = logging.getLogger("spock_api")
    api_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Suppress noisy third-party loggers
    for logger_name in [
        "httpx",
        "httpcore", 
        "chromadb",
        "openai",
        "uvicorn.access",
        "uvicorn.error",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with request ID support.
    
    Args:
        name: Logger name, typically __name__.
    
    Returns:
        Configured logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing request")  # Includes request_id automatically
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding structured context to logs.
    
    Example:
        with LogContext(user_id="123", action="query"):
            logger.info("Processing")  # Will include user_id and action
    """
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        context = self.context
        
        def factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
        return False

