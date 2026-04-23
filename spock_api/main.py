"""
Spock API - FastAPI Application

Main application entry point with route wiring and middleware configuration.

Usage:
    # Development
    uvicorn spock_api.main:app --reload --port 8000
    
    # Production
    uvicorn spock_api.main:app --host 0.0.0.0 --port 8000 --workers 4
    
    # Or run directly
    python -m spock_api
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from spock_api.core.logging import (
    RequestLoggingMiddleware,
    get_logger,
    get_request_id,
    setup_logging,
)
from spock_api.core.settings import get_settings
from spock_api.routes import chat
from spock_api.schemas import ErrorDetail, HealthResponse


# Setup logging before anything else
setup_logging()

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    
    Runs startup and shutdown logic for the application.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"CORS origins: {settings.CORS_ORIGINS}")
    
    if not settings.API_KEY:
        logger.warning(
            "API_KEY not set - authentication is disabled. "
            "Set API_KEY environment variable for production."
        )
    
    # Pre-initialize the RAG service to catch configuration errors early
    try:
        from spock_api.services.rag_service import get_rag_service
        get_rag_service()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # Don't fail startup - let individual requests fail
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="REST API for Spock AI RAG System. "
                "Provides chat endpoints with optional streaming support.",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================


# CORS middleware - must be added before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["x-request-id"],
)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic validation errors with consistent error schema.
    """
    request_id = get_request_id()
    
    # Format validation errors
    errors = exc.errors()
    if errors:
        # Get first error message
        first_error = errors[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        message = f"{field}: {first_error.get('msg', 'Validation error')}"
    else:
        message = "Validation error"
    
    logger.warning(f"Validation error: {message}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": ErrorDetail(
                error="validation_error",
                message=message,
                request_id=request_id,
            ).model_dump(mode="json"),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle uncaught exceptions with consistent error schema.
    """
    request_id = get_request_id()
    
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    
    # In debug mode, include the actual error message
    if settings.DEBUG:
        message = f"{type(exc).__name__}: {str(exc)}"
    else:
        message = "An internal server error occurred."
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": ErrorDetail(
                error="server_error",
                message=message,
                request_id=request_id,
            ).model_dump(mode="json"),
        },
    )


# =============================================================================
# Routes
# =============================================================================


# Include chat routes
app.include_router(chat.router)


# Health check endpoint (no auth required)
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the API is running and healthy.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status, version, and current time.
    Does not require authentication.
    """
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
    )


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API root",
    include_in_schema=False,
)
async def root() -> dict:
    """Root endpoint with basic API info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs" if settings.DEBUG else "Disabled in production",
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def run_server() -> None:
    """Run the API server using uvicorn."""
    import uvicorn

    port = int(os.getenv("PORT", str(settings.API_PORT)))

    uvicorn.run(
        "spock_api.main:app",
        host=settings.API_HOST,
        port=port,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run_server()

