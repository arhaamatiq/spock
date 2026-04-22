"""
Smoke Tests for Streaming Functionality

These tests verify:
- Streaming generator yields chunks
- Session management works correctly
- History-aware conversations
"""

import os
import tempfile
from pathlib import Path

import pytest

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def rag_engine_with_store(monkeypatch):
    """
    Create a RAG engine with a populated vector store.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories
        docs_dir = Path(tmpdir) / "docs"
        persist_dir = Path(tmpdir) / "chroma_store"
        docs_dir.mkdir()
        
        # Create a test document
        (docs_dir / "test.txt").write_text(
            "The Spock AI system is a RAG-based question answering system. "
            "It retrieves relevant documents from a knowledge base and uses "
            "them to generate accurate answers. The system supports streaming "
            "output for real-time responses."
        )
        
        # Set environment and reset config
        from spock_rag.config import reset_settings
        monkeypatch.setenv("PERSIST_DIR", str(persist_dir))
        reset_settings()
        
        # Ingest documents
        from spock_rag.ingest import ingest_documents
        ingest_documents(docs_dir, force=True)
        
        # Create the engine
        from spock_rag.rag_engine import RAGEngine
        engine = RAGEngine()
        
        yield engine


class TestStreaming:
    """Tests for streaming functionality."""
    
    def test_stream_answer_yields_chunks(self, rag_engine_with_store):
        """Test that stream_answer yields multiple chunks."""
        engine = rag_engine_with_store
        
        chunks = list(engine.stream_answer(
            "What is the Spock AI system?",
            session_id="test-session"
        ))
        
        # Should yield multiple chunks
        assert len(chunks) > 0
        
        # Combined chunks should form a coherent response
        full_response = "".join(chunks)
        assert len(full_response) > 10
    
    def test_stream_answer_handles_empty_question(self, rag_engine_with_store):
        """Test that stream_answer handles empty questions."""
        engine = rag_engine_with_store
        
        chunks = list(engine.stream_answer("   ", session_id="test-session"))
        
        # Should yield an error message
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "empty" in full_response.lower() or "question" in full_response.lower()


class TestSessionManagement:
    """Tests for session management."""
    
    def test_session_history_preserved(self, rag_engine_with_store):
        """Test that conversation history is preserved within a session."""
        engine = rag_engine_with_store
        session_id = "test-history-session"
        
        # First question
        _ = engine.answer("What is Spock AI?", session_id)
        
        # Check history
        history = engine.get_session_history(session_id)
        
        assert len(history) == 2  # User + Assistant messages
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_different_sessions_independent(self, rag_engine_with_store):
        """Test that different sessions have independent histories."""
        engine = rag_engine_with_store
        
        # First session
        _ = engine.answer("What is Spock?", session_id="session-1")
        
        # Second session
        _ = engine.answer("What is RAG?", session_id="session-2")
        
        history_1 = engine.get_session_history("session-1")
        history_2 = engine.get_session_history("session-2")
        
        # Each session should have its own history
        assert "Spock" in history_1[0]["content"]
        assert "RAG" in history_2[0]["content"]
    
    def test_clear_session(self, rag_engine_with_store):
        """Test clearing a session."""
        engine = rag_engine_with_store
        session_id = "test-clear-session"
        
        # Add some history
        _ = engine.answer("Test question", session_id)
        
        # Clear session
        engine.clear_session(session_id)
        
        # History should be empty
        history = engine.get_session_history(session_id)
        assert len(history) == 0


class TestAnswerGeneration:
    """Tests for answer generation (non-streaming)."""
    
    def test_answer_returns_string(self, rag_engine_with_store):
        """Test that answer() returns a string response."""
        engine = rag_engine_with_store
        
        response = engine.answer(
            "What does the Spock AI system do?",
            session_id="test-answer"
        )
        
        assert isinstance(response, str)
        assert len(response) > 10
    
    def test_answer_updates_history(self, rag_engine_with_store):
        """Test that answer() updates session history."""
        engine = rag_engine_with_store
        session_id = "test-answer-history"
        
        question = "What is Spock AI?"
        _ = engine.answer(question, session_id)
        
        history = engine.get_session_history(session_id)
        
        # Should have user question and assistant response
        assert len(history) >= 2
        assert question in history[0]["content"]


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_module_level_answer(self, rag_engine_with_store, monkeypatch):
        """Test the module-level answer function."""
        # Import after fixture sets up the store
        from spock_rag import rag_engine
        
        # Reset the module-level engine
        rag_engine._engine = rag_engine_with_store
        
        response = rag_engine.answer("What is Spock?", "test-module")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_module_level_stream_answer(self, rag_engine_with_store, monkeypatch):
        """Test the module-level stream_answer function."""
        from spock_rag import rag_engine
        
        # Reset the module-level engine
        rag_engine._engine = rag_engine_with_store
        
        chunks = list(rag_engine.stream_answer("What is Spock?", "test-module-stream"))
        
        assert len(chunks) > 0

