# Spock API

Production-quality FastAPI backend for the Spock RAG system.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root (or export variables):

```bash
# Required for RAG (inherited from spock_rag)
OPENAI_API_KEY=sk-your-key-here

# Optional API configuration
API_KEY=your-api-key-here        # If set, enables API key authentication
API_PORT=8000                     # Default: 8000
API_HOST=0.0.0.0                  # Default: 0.0.0.0
DEBUG=true                        # Enable debug mode (shows /docs)

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Logging
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 3. Run the Server

```bash
# Development (with auto-reload)
uvicorn spock_api.main:app --reload --port 8000

# Development with API key authentication
API_KEY=your-secret-key uvicorn spock_api.main:app --reload --port 8000

# Or run as module
python -m spock_api

# Production (multiple workers)
uvicorn spock_api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

On Railway, use `python -m spock_api` (or the repo `Procfile`). The app will bind to
Railway's injected `PORT` automatically.

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service status. No authentication required.

### Chat (Non-Streaming)

```bash
POST /v1/chat
Content-Type: application/json
x-api-key: your-api-key  # If API_KEY is configured

{
    "message": "What is Python?",
    "chat_history": []
}
```

**Response:**
```json
{
    "answer": "Python is a high-level programming language...",
    "request_id": "abc12345"
}
```

### Chat (Streaming)

```bash
POST /v1/chat/stream
Content-Type: application/json
x-api-key: your-api-key  # If API_KEY is configured

{
    "message": "What is Python?",
    "chat_history": []
}
```

**Response:** Server-Sent Events (SSE)

```
data: {"event":"token","token":"Python","done":false,"request_id":"abc123"}

data: {"event":"token","token":" is","done":false,"request_id":"abc123"}

data: {"event":"done","token":"","done":true,"answer":"Python is...","request_id":"abc123"}
```

## Using with Chat History

Include previous messages for context-aware conversations:

```json
{
    "message": "What are its main features?",
    "chat_history": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language..."}
    ]
}
```

## Next.js Integration

### Non-Streaming

```typescript
const response = await fetch('http://localhost:8000/v1/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.SPOCK_API_KEY,
    },
    body: JSON.stringify({
        message: userMessage,
        chat_history: messages,
    }),
});

const data = await response.json();
console.log(data.answer);
```

### Streaming (SSE)

```typescript
const response = await fetch('http://localhost:8000/v1/chat/stream', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.SPOCK_API_KEY,
    },
    body: JSON.stringify({
        message: userMessage,
        chat_history: messages,
    }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const event = JSON.parse(line.slice(6));
            if (event.event === 'token') {
                // Append token to UI
                console.log(event.token);
            } else if (event.event === 'done') {
                // Streaming complete
                console.log('Full answer:', event.answer);
            }
        }
    }
}
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (none) | API key for authentication. If not set, auth is disabled. |
| `API_HOST` | `0.0.0.0` | Host to bind the server to |
| `API_PORT` | `8000` | Port to bind the server to |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG` | `false` | Enable debug mode (exposes /docs) |
| `RAG_RETRIEVAL_K` | (from spock_rag) | Override number of documents to retrieve |

## Project Structure

```
spock_api/
├── __init__.py
├── __main__.py          # Entry point for `python -m spock_api`
├── main.py              # FastAPI app + route wiring
├── schemas.py           # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── settings.py      # Configuration via pydantic-settings
│   ├── security.py      # API key authentication
│   └── logging.py       # Structured logging + request ID middleware
├── routes/
│   ├── __init__.py
│   └── chat.py          # Chat endpoints (non-streaming + SSE)
├── services/
│   ├── __init__.py
│   └── rag_service.py   # Thin adapter that calls spock_rag
└── README.md
```

## Error Handling

All errors return a consistent schema:

```json
{
    "detail": {
        "error": "validation_error",
        "message": "body.message: String should have at least 1 character",
        "request_id": "abc12345",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Security Notes

- **API Key**: In production, always set `API_KEY` to enable authentication
- **CORS**: Configure `CORS_ORIGINS` to only allow your frontend domain
- **Debug**: Set `DEBUG=false` in production to hide `/docs` endpoint
- **Secrets**: API keys are never logged
- **Vector store**: If you are deploying with a committed `chroma_store/`, rebuild it
  locally after updating `data/docs/` and commit the regenerated files before deploy

## Logging

Structured logs include request ID for tracing:

```
2024-01-15 10:30:00 | INFO     | spock_api.request | [abc12345] | Request started: POST /v1/chat
2024-01-15 10:30:00 | INFO     | spock_api.services.rag_service | [abc12345] | Retrieved doc: source=docs/python.txt, chunk=0, score=0.8521
2024-01-15 10:30:01 | INFO     | spock_api.request | [abc12345] | Request completed: POST /v1/chat status=200 latency=1234.56ms
```

