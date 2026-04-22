"""
Security Module

API key authentication for the Spock API.
"""

from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from spock_api.core.logging import get_request_id
from spock_api.core.settings import get_settings
from spock_api.schemas import ErrorDetail


# API key header scheme
api_key_header = APIKeyHeader(
    name="x-api-key",
    auto_error=False,
    description="API key for authentication. Pass via x-api-key header.",
)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Verify the API key from request header.
    
    If API_KEY is not set in settings, authentication is skipped (development mode).
    If API_KEY is set, requests must include a valid x-api-key header.
    
    Args:
        request: The FastAPI request object.
        api_key: API key from the x-api-key header.
    
    Returns:
        The validated API key if authentication succeeded, None if auth is disabled.
    
    Raises:
        HTTPException: 401 if API key is missing or invalid.
    """
    settings = get_settings()
    
    # If no API key is configured, skip authentication (development mode)
    if not settings.API_KEY:
        return None
    
    # API key is required but not provided
    if not api_key:
        request_id = get_request_id()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                error="unauthorized",
                message="API key is required. Provide it via the x-api-key header.",
                request_id=request_id,
            ).model_dump(),
        )
    
    # Validate the API key using constant-time comparison
    if not _constant_time_compare(api_key, settings.API_KEY):
        request_id = get_request_id()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                error="unauthorized",
                message="Invalid API key.",
                request_id=request_id,
            ).model_dump(),
        )
    
    return api_key


def _constant_time_compare(val1: str, val2: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        val1: First string to compare.
        val2: Second string to compare.
    
    Returns:
        True if strings are equal, False otherwise.
    """
    import hmac
    return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))


class APIKeyAuth:
    """
    Callable dependency for API key authentication.
    
    Can be used as a dependency in route definitions or as a router-level dependency.
    
    Example:
        @router.post("/chat", dependencies=[Depends(APIKeyAuth())])
        async def chat(...):
            ...
    """
    
    async def __call__(
        self,
        request: Request,
        api_key: Optional[str] = Security(api_key_header),
    ) -> Optional[str]:
        """Verify API key."""
        return await verify_api_key(request, api_key)

