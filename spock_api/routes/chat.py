"""
Chat Routes

Endpoints for chat interactions with the RAG system.
Provides both non-streaming and streaming (SSE) endpoints.
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from spock_api.core.logging import get_logger, get_request_id
from spock_api.core.security import verify_api_key
from spock_api.core.settings import get_settings
from spock_api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    StreamEvent,
)
from spock_api.services.rag_service import get_rag_service


logger = get_logger(__name__)

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Depends(verify_api_key)],
)


# =============================================================================
# Non-Streaming Chat Endpoint
# =============================================================================


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Spock AI",
    description="Send a message and receive a complete response. "
                "Include chat_history for context-aware conversations.",
    responses={
        200: {
            "description": "Successful response",
            "model": ChatResponse,
        },
        401: {
            "description": "Unauthorized - Invalid or missing API key",
        },
        422: {
            "description": "Validation error - Invalid request body",
        },
        500: {
            "description": "Internal server error",
        },
    },
)
async def chat(
    request: Request,
    body: ChatRequest,
) -> ChatResponse:
    """
    Generate a complete chat response.
    
    This endpoint:
    1. Accepts the user's message and optional chat history
    2. Retrieves relevant documents from the knowledge base
    3. Generates a response using the RAG pipeline
    4. Returns the complete answer
    
    Args:
        request: FastAPI request object.
        body: Chat request containing message and optional history.
    
    Returns:
        ChatResponse with the answer and request ID.
    """
    request_id = get_request_id()
    
    logger.info(
        f"Chat request: message_len={len(body.message)}, "
        f"history_len={len(body.chat_history)}"
    )
    
    try:
        service = get_rag_service()
        
        answer = await service.generate_answer(
            question=body.message,
            chat_history=body.chat_history,
        )
        
        logger.info(f"Chat response generated: answer_len={len(answer)}")
        
        return ChatResponse(
            answer=answer,
            request_id=request_id,
        )
        
    except Exception as e:
        logger.error(f"Chat error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                error="server_error",
                message="An error occurred while generating the response.",
                request_id=request_id,
            ).model_dump(),
        )


# =============================================================================
# Streaming Chat Endpoint (SSE)
# =============================================================================


async def generate_sse_stream(
    message: str,
    chat_history: list,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for streaming chat.
    
    SSE Format:
        data: {"event": "token", "token": "...", "done": false, "request_id": "..."}
        
        data: {"event": "done", "token": "", "done": true, "answer": "...", "request_id": "..."}
    
    Args:
        message: The user's message.
        chat_history: Previous conversation messages.
        request_id: Request identifier.
    
    Yields:
        SSE-formatted strings.
    """
    settings = get_settings()
    service = get_rag_service()
    
    full_answer_chunks = []
    
    try:
        async for token in service.stream_answer(message, chat_history):
            full_answer_chunks.append(token)
            
            # Create token event
            event = StreamEvent(
                event="token",
                token=token,
                done=False,
                request_id=request_id,
            )
            
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Optional artificial delay for testing
            if settings.STREAM_CHUNK_DELAY_MS > 0:
                await asyncio.sleep(settings.STREAM_CHUNK_DELAY_MS / 1000)
        
        # Send completion event with full answer
        full_answer = "".join(full_answer_chunks)
        done_event = StreamEvent(
            event="done",
            token="",
            done=True,
            answer=full_answer,
            request_id=request_id,
        )
        
        yield f"data: {done_event.model_dump_json()}\n\n"
        
        logger.info(f"Stream completed: answer_len={len(full_answer)}")
        
    except Exception as e:
        logger.error(f"Stream error: {type(e).__name__}: {e}")
        
        # Send error event
        error_event = StreamEvent(
            event="error",
            token="",
            done=True,
            error="An error occurred while generating the response.",
            request_id=request_id,
        )
        
        yield f"data: {error_event.model_dump_json()}\n\n"


@router.post(
    "/chat/stream",
    summary="Stream chat with Spock AI (SSE)",
    description="Send a message and receive a streaming response via Server-Sent Events. "
                "Each event contains a token for incremental display.",
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {
                "text/event-stream": {
                    "example": 'data: {"event":"token","token":"Hello","done":false,"request_id":"abc123"}\n\n'
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing API key",
        },
        422: {
            "description": "Validation error - Invalid request body",
        },
    },
)
async def chat_stream(
    request: Request,
    body: ChatRequest,
) -> StreamingResponse:
    """
    Stream a chat response using Server-Sent Events (SSE).
    
    This endpoint:
    1. Accepts the user's message and optional chat history
    2. Streams tokens as they're generated
    3. Sends a final "done" event with the complete answer
    
    Event Types:
        - "token": Incremental token (stream in progress)
        - "done": Stream complete (includes full answer)
        - "error": An error occurred
    
    Args:
        request: FastAPI request object.
        body: Chat request containing message and optional history.
    
    Returns:
        StreamingResponse with SSE content.
    """
    request_id = get_request_id()
    
    logger.info(
        f"Stream request: message_len={len(body.message)}, "
        f"history_len={len(body.chat_history)}"
    )
    
    return StreamingResponse(
        generate_sse_stream(
            message=body.message,
            chat_history=body.chat_history,
            request_id=request_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-ID": request_id,
        },
    )

