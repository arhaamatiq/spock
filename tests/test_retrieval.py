"""
Smoke Tests for Document Retrieval

These tests verify:
- Vector store loading
- Document retrieval
- Relevance scoring
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
def populated_store(monkeypatch):
    """
    Create a temporary store populated with test documents.
    
    This fixture:
    1. Creates temp directories for docs and persistence
    2. Creates test documents
    3. Ingests them into ChromaDB
    4. Yields the persist directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories
        docs_dir = Path(tmpdir) / "docs"
        persist_dir = Path(tmpdir) / "chroma_store"
        docs_dir.mkdir()
        
        # Create test documents with specific content
        (docs_dir / "python.txt").write_text(
            "Python is a high-level programming language known for its simplicity. "
            "It supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming. Python uses indentation "
            "for code blocks rather than braces."
        )
        
        (docs_dir / "javascript.txt").write_text(
            "JavaScript is a programming language primarily used for web development. "
            "It runs in web browsers and can manipulate the DOM. Node.js allows "
            "JavaScript to run on servers. It supports asynchronous programming."
        )
        
        (docs_dir / "rust.txt").write_text(
            "Rust is a systems programming language focused on safety and performance. "
            "It prevents memory errors without garbage collection. Rust has a concept "
            "called ownership that manages memory automatically at compile time."
        )
        
        # Set environment and reset config
        from spock_rag.config import reset_settings
        monkeypatch.setenv("PERSIST_DIR", str(persist_dir))
        reset_settings()
        
        # Ingest documents
        from spock_rag.ingest import ingest_documents
        ingest_documents(docs_dir, force=True)
        
        yield persist_dir


class TestVectorStore:
    """Tests for vector store functionality."""
    
    def test_get_vector_store(self, populated_store):
        """Test loading the vector store."""
        from spock_rag.retrieval import get_vector_store
        
        store = get_vector_store(populated_store)
        
        # Should have documents
        count = store._collection.count()
        assert count > 0
    
    def test_check_store_exists(self, populated_store, monkeypatch):
        """Test checking if store exists."""
        from spock_rag.config import reset_settings
        from spock_rag.retrieval import check_store_exists
        
        monkeypatch.setenv("PERSIST_DIR", str(populated_store))
        reset_settings()
        
        assert check_store_exists() is True


class TestRetrieval:
    """Tests for document retrieval."""
    
    def test_retrieve_documents(self, populated_store, monkeypatch):
        """Test retrieving documents for a query."""
        from spock_rag.config import reset_settings
        from spock_rag.retrieval import retrieve_documents
        
        monkeypatch.setenv("PERSIST_DIR", str(populated_store))
        reset_settings()
        
        # Query about Python
        docs = retrieve_documents("What is Python?", k=2)
        
        assert len(docs) > 0
        # The most relevant doc should mention Python
        assert "Python" in docs[0].page_content or "python" in docs[0].page_content.lower()
    
    def test_retrieve_with_scores(self, populated_store, monkeypatch):
        """Test retrieving documents with relevance scores."""
        from spock_rag.config import reset_settings
        from spock_rag.retrieval import retrieve_with_scores
        
        monkeypatch.setenv("PERSIST_DIR", str(populated_store))
        reset_settings()
        
        results = retrieve_with_scores("What is Rust?", k=3)
        
        assert len(results) > 0
        
        # Results should be (document, score) tuples
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, float)
    
    def test_retrieve_filters_by_min_score(self, populated_store, monkeypatch):
        """Test that min_score filtering works."""
        from spock_rag.config import reset_settings
        from spock_rag.retrieval import retrieve_with_scores
        
        monkeypatch.setenv("PERSIST_DIR", str(populated_store))
        reset_settings()
        
        # Get results with high minimum score
        # This might filter out some results
        results = retrieve_with_scores("programming language", k=5, min_score=0.5)
        
        # All returned scores should be above threshold
        for _, score in results:
            assert score >= 0.5


class TestContextFormatting:
    """Tests for context formatting."""
    
    def test_format_context(self, populated_store, monkeypatch):
        """Test formatting documents into context string."""
        from spock_rag.config import reset_settings
        from spock_rag.retrieval import retrieve_documents, format_context
        
        monkeypatch.setenv("PERSIST_DIR", str(populated_store))
        reset_settings()
        
        docs = retrieve_documents("What is JavaScript?", k=2)
        context = format_context(docs)
        
        assert len(context) > 0
        assert "Source" in context  # Should include source info
    
    def test_format_empty_context(self):
        """Test formatting empty document list."""
        from spock_rag.retrieval import format_context
        
        context = format_context([])
        
        assert "No relevant context" in context

