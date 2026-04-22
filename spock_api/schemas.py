"""
Pydantic Schemas

Request and response models for the Spock API.
All models are designed for JSON compatibility with Next.js fetch.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Chat Message Models
# =============================================================================


class ChatMessage(BaseModel):
    """
    A single chat message in conversation history.
    
    Attributes:
        role: Who sent the message ("user" or "assistant").
        content: The message text.
    """
    role: Literal["user", "assistant"] = Field(
        description="Message sender role"
    )
    content: str = Field(
        description="Message content text"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language..."},
            ]
        }
    }


# =============================================================================
# Request Models
# =============================================================================


class ChatRequest(BaseModel):
    """
    Request body for chat endpoints.
    
    Attributes:
        message: The user's current message/question.
        chat_history: Optional list of previous messages for context.
    """
    message: str = Field(
        min_length=1,
        max_length=10000,
        description="The user's question or message",
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
        description="Previous conversation messages for context",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is Python?",
                    "chat_history": []
                },
                {
                    "message": "What are its main features?",
                    "chat_history": [
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a high-level programming language known for its readability and versatility."}
                    ]
                }
            ]
        }
    }


# =============================================================================
# Response Models
# =============================================================================


class ChatResponse(BaseModel):
    """
    Response body for the non-streaming chat endpoint.
    
    Attributes:
        answer: The generated answer text.
        request_id: Unique identifier for this request (for debugging/logging).
    """
    answer: str = Field(
        description="The generated answer to the user's question"
    )
    request_id: str = Field(
        description="Unique request identifier for tracing"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Python is a high-level, interpreted programming language known for its clear syntax and readability.",
                    "request_id": "abc12345"
                }
            ]
        }
    }


class StreamEvent(BaseModel):
    """
    A single Server-Sent Event (SSE) payload for streaming responses.
    
    Events are sent as: `data: <json>\n\n`
    
    Attributes:
        event: Event type ("token", "done", or "error").
        token: The text token (for "token" events).
        done: Whether streaming is complete.
        answer: The complete answer (only sent with "done" event).
        error: Error message (only sent with "error" event).
        request_id: Request identifier.
    """
    event: Literal["token", "done", "error"] = Field(
        description="Event type"
    )
    token: str = Field(
        default="",
        description="Token text for incremental streaming"
    )
    done: bool = Field(
        default=False,
        description="Whether streaming is complete"
    )
    answer: Optional[str] = Field(
        default=None,
        description="Complete answer (sent with 'done' event)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (sent with 'error' event)"
    )
    request_id: str = Field(
        description="Request identifier"
    )


# =============================================================================
# Error Models
# =============================================================================


class ErrorDetail(BaseModel):
    """
    Structured error detail.
    
    Attributes:
        error: Error code/type (e.g., "validation_error", "server_error").
        message: Human-readable error description.
        request_id: Request identifier for debugging.
        timestamp: When the error occurred.
    """
    error: str = Field(
        description="Error code or type"
    )
    message: str = Field(
        description="Human-readable error description"
    )
    request_id: str = Field(
        description="Request identifier for tracing"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred (UTC)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "validation_error",
                    "message": "Message cannot be empty",
                    "request_id": "abc12345",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Standard error response wrapper.
    
    Attributes:
        detail: The error details.
    """
    detail: ErrorDetail


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """
    Health check response.
    
    Attributes:
        status: Service status ("healthy" or "unhealthy").
        version: API version.
        timestamp: Current server time.
    """
    status: Literal["healthy", "unhealthy"] = Field(
        description="Service health status"
    )
    version: str = Field(
        description="API version"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server time (UTC)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "0.1.0",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }

