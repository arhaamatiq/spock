# Spock AI RAG System

A robust Python RAG (Retrieval-Augmented Generation) engine using LangChain and ChromaDB. Designed to be simple, testable, and easy to integrate with a future web API.

## Features

- **Persistent Knowledge Base**: ChromaDB stores vectors locally, surviving restarts
- **Smart Document Processing**: Recursive text splitting with configurable chunk size/overlap
- **History-Aware Conversations**: In-memory session management with automatic history windowing
- **Streaming Output**: Token-by-token response streaming for real-time UI updates
- **Robust Error Handling**: Graceful fallbacks, structured logging, and clear error messages
- **Clean Architecture**: Modular design separating ingestion, retrieval, prompting, and generation

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```
OPENAI_API_KEY=your-actual-api-key-here
```

### 3. Add Documents

Place your PDF, TXT, or Markdown files in the `data/docs/` directory:

```bash
mkdir -p data/docs
# Copy your documents here
cp /path/to/your/documents/* data/docs/
```

### 4. Ingest Documents

Process and store your documents in the vector database:

```bash
python -m spock_rag.ingest --docs ./data/docs
```

Optional flags:
- `--force`: Re-ingest all documents (ignore existing data)
- `--chunk-size 1000`: Override default chunk size
- `--chunk-overlap 200`: Override default chunk overlap
- `--debug`: Enable debug logging

If you plan to deploy with a committed local Chroma store, use a full rebuild so the
persisted directory matches the current contents of `data/docs/` exactly:

```bash
rm -rf chroma_store
python -m spock_rag.ingest --docs ./data/docs --force
```

### 5. Start Chatting

Launch the interactive chat interface:

```bash
python -m spock_rag.cli chat
```

Chat commands:
- `/new` - Start a new session (clear history)
- `/history` - Show conversation history
- `/quit` - Exit the chat
- `/help` - Show available commands

## Project Structure

```
spock/
├── spock_rag/
│   ├── __init__.py          # Package initialization
│   ├── __main__.py          # Module entry point
│   ├── config.py            # Configuration from environment variables
│   ├── logging_config.py    # Structured logging setup
│   ├── ingest.py            # Document loading and ingestion
│   ├── retrieval.py         # Vector store and document retrieval
│   ├── rag_engine.py        # Main RAG orchestration engine
│   ├── session.py           # In-memory chat session management
│   ├── prompts.py           # Customizable prompt templates
│   ├── cli.py               # Command-line interface
│   └── utils.py             # Helper utilities
├── tests/
│   ├── conftest.py          # Pytest configuration
│   ├── test_ingest.py       # Ingestion tests
│   ├── test_retrieval.py    # Retrieval tests
│   └── test_streaming.py    # Streaming and session tests
├── data/
│   └── docs/                # Your documents go here
├── env.example              # Example environment file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration Options

All configuration is done via environment variables. See `env.example` for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for document embeddings |
| `PERSIST_DIR` | `./chroma_store` | ChromaDB storage directory |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Number of documents to retrieve |
| `MAX_HISTORY` | `10` | Conversation turns to remember |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  Documents → Loaders → Text Splitter → Embeddings → ChromaDB    │
│  (PDF/TXT/MD)  (LangChain)  (Recursive)   (OpenAI)   (Persist)  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        RUNTIME PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  Question → Session → Reformulate → Retrieve → Prompt → Stream  │
│  (User)    (History)  (Standalone)  (ChromaDB) (Template) (LLM) │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **LangChain Core**: Uses LangChain for document processing, embeddings, and LLM interaction, providing a well-tested foundation.

2. **ChromaDB Persistence**: Local vector storage that survives restarts without external dependencies.

3. **In-Memory Sessions**: Chat history lives only in memory, matching the "page refresh clears history" behavior expected in web UIs.

4. **File Hash Tracking**: Documents are hashed to detect changes, avoiding unnecessary re-ingestion.

5. **Generator-Based Streaming**: Uses Python generators for streaming, making it easy to wrap in FastAPI `StreamingResponse` later.

6. **History-Aware Retrieval**: Follow-up questions are reformulated to be standalone before retrieval, improving context relevance.

## Programmatic Usage

```python
from spock_rag.rag_engine import RAGEngine

# Initialize the engine
engine = RAGEngine()

# Get a complete answer
answer = engine.answer("What is Python?", session_id="user123")
print(answer)

# Stream an answer
for chunk in engine.stream_answer("Tell me more", session_id="user123"):
    print(chunk, end="", flush=True)
print()

# Manage sessions
engine.clear_session("user123")
history = engine.get_session_history("user123")
```

## Running Tests

```bash
# Run all tests (requires OPENAI_API_KEY)
pytest tests/ -v

# Run specific test file
pytest tests/test_ingest.py -v

# Run with debug output
pytest tests/ -v -s
```

**Note**: Tests require a valid `OPENAI_API_KEY` in your environment. Tests that make API calls are skipped if the key is not set.

## Error Handling

The system handles errors gracefully:

- **Empty question**: Returns a friendly message asking for input
- **Missing docs directory**: Raises `FileNotFoundError` with clear message
- **No relevant documents**: Returns "I don't have information about that"
- **Bad PDF files**: Logs warning, skips file, continues processing
- **API errors**: Logs error, returns fallback message

## Future Integration

The engine is designed for easy FastAPI integration:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from spock_rag.rag_engine import RAGEngine

app = FastAPI()
engine = RAGEngine()

@app.post("/chat")
async def chat(question: str, session_id: str):
    return {"answer": engine.answer(question, session_id)}

@app.post("/chat/stream")
async def chat_stream(question: str, session_id: str):
    return StreamingResponse(
        engine.stream_answer(question, session_id),
        media_type="text/plain"
    )
```

## Troubleshooting

### "OPENAI_API_KEY is required"
Make sure your `.env` file exists and contains a valid API key.

### "No documents found to ingest"
Check that your documents directory contains supported files (.pdf, .txt, .md).

### "Vector store is empty"
Run the ingestion command before chatting: `python -m spock_rag.ingest --docs ./data/docs`

## Railway Deployment

This repo is set up for a simple Railway deploy that serves the API from a committed
`chroma_store/` directory.

1. Rebuild the store locally after doc changes:
   `python -m spock_rag.ingest --docs ./data/docs --force`
2. Commit both `data/docs/` changes and the regenerated `chroma_store/`.
3. Deploy the repo to Railway. The included `Procfile` runs `python -m spock_api`.
4. In Railway variables, set at least:
   - `OPENAI_API_KEY`
   - `API_KEY`
   - `DEBUG=false`
   - `CORS_ORIGINS=https://your-frontend-domain`

Railway injects `PORT` automatically, and the API now respects it out of the box.

### Tests are skipped
Tests require `OPENAI_API_KEY` in your environment. Export it or add to `.env`.

## License

MIT License - Feel free to use and modify as needed.

