"""
Smoke Tests for Document Ingestion

These tests verify:
- Document loading from various file types
- Text splitting with metadata preservation
- ChromaDB persistence
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
def temp_docs_dir():
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()
        
        # Create a simple text file
        txt_file = docs_dir / "test.txt"
        txt_file.write_text(
            "Python is a high-level programming language.\n"
            "It was created by Guido van Rossum in 1991.\n"
            "Python is known for its simple syntax and readability."
        )
        
        # Create a markdown file
        md_file = docs_dir / "readme.md"
        md_file.write_text(
            "# Test Document\n\n"
            "This is a test markdown document.\n\n"
            "## Features\n\n"
            "- Easy to use\n"
            "- Well documented\n"
        )
        
        yield docs_dir


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "chroma_store"


class TestDocumentLoading:
    """Tests for document loading functionality."""
    
    def test_load_text_file(self, temp_docs_dir):
        """Test loading a text file."""
        from spock_rag.ingest import load_single_document
        
        txt_file = temp_docs_dir / "test.txt"
        docs = load_single_document(txt_file)
        
        assert len(docs) >= 1
        assert "Python" in docs[0].page_content
    
    def test_load_markdown_file(self, temp_docs_dir):
        """Test loading a markdown file."""
        from spock_rag.ingest import load_single_document
        
        md_file = temp_docs_dir / "readme.md"
        docs = load_single_document(md_file)
        
        assert len(docs) >= 1
        assert "Test Document" in docs[0].page_content
    
    def test_load_documents_from_directory(self, temp_docs_dir):
        """Test loading all documents from a directory."""
        from spock_rag.ingest import load_documents
        
        docs = load_documents(temp_docs_dir)
        
        # Should load both txt and md files
        assert len(docs) >= 2
    
    def test_load_nonexistent_directory_raises(self):
        """Test that loading from non-existent directory raises error."""
        from spock_rag.ingest import load_documents
        
        with pytest.raises(FileNotFoundError):
            load_documents(Path("/nonexistent/path"))


class TestTextSplitting:
    """Tests for text splitting functionality."""
    
    def test_split_documents(self, temp_docs_dir):
        """Test document splitting."""
        from spock_rag.ingest import load_documents, split_documents
        
        docs = load_documents(temp_docs_dir)
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        
        # Should produce at least as many chunks as documents
        assert len(chunks) >= len(docs)
    
    def test_split_preserves_metadata(self, temp_docs_dir):
        """Test that splitting preserves document metadata."""
        from spock_rag.ingest import load_documents, split_documents
        
        docs = load_documents(temp_docs_dir)
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        
        for chunk in chunks:
            # Each chunk should have source metadata
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "doc_id" in chunk.metadata


class TestIngestion:
    """Tests for the full ingestion pipeline."""
    
    def test_ingest_creates_store(self, temp_docs_dir, temp_persist_dir, monkeypatch):
        """Test that ingestion creates a persistent store."""
        from spock_rag.config import reset_settings
        from spock_rag.ingest import ingest_documents
        
        # Set the persist directory to our temp directory
        monkeypatch.setenv("PERSIST_DIR", str(temp_persist_dir))
        reset_settings()
        
        # Run ingestion
        num_chunks = ingest_documents(temp_docs_dir, force=True)
        
        assert num_chunks > 0
        assert temp_persist_dir.exists()
    
    def test_ingest_skips_unchanged(self, temp_docs_dir, temp_persist_dir, monkeypatch):
        """Test that re-ingestion skips unchanged documents."""
        from spock_rag.config import reset_settings
        from spock_rag.ingest import ingest_documents
        
        monkeypatch.setenv("PERSIST_DIR", str(temp_persist_dir))
        reset_settings()
        
        # First ingestion
        num_chunks_1 = ingest_documents(temp_docs_dir, force=True)
        
        # Second ingestion (should skip unchanged files)
        num_chunks_2 = ingest_documents(temp_docs_dir, force=False)
        
        assert num_chunks_1 > 0
        assert num_chunks_2 == 0  # Nothing new to ingest

    def test_ingest_removes_deleted_documents(self, temp_docs_dir, temp_persist_dir, monkeypatch):
        """Test that deleted source files are removed from the persistent store."""
        from spock_rag.config import reset_settings
        from spock_rag.ingest import ingest_documents
        from spock_rag.retrieval import get_vector_store

        monkeypatch.setenv("PERSIST_DIR", str(temp_persist_dir))
        reset_settings()

        ingest_documents(temp_docs_dir, force=True)

        deleted_file = temp_docs_dir / "readme.md"
        deleted_file.unlink()

        ingest_documents(temp_docs_dir, force=False)

        store = get_vector_store(temp_persist_dir)
        records = store.get(include=["metadatas"])
        sources = {metadata["source"] for metadata in records["metadatas"]}

        assert str(deleted_file) not in sources
        assert str(temp_docs_dir / "test.txt") in sources

    def test_ingest_replaces_old_chunks_for_changed_document(
        self,
        temp_docs_dir,
        temp_persist_dir,
        monkeypatch,
    ):
        """Test that stale chunks are removed when a file shrinks after re-ingest."""
        from spock_rag.config import reset_settings
        from spock_rag.ingest import ingest_documents
        from spock_rag.retrieval import get_vector_store

        monkeypatch.setenv("PERSIST_DIR", str(temp_persist_dir))
        reset_settings()

        target_file = temp_docs_dir / "test.txt"
        target_file.write_text("A " * 300)

        first_count = ingest_documents(
            temp_docs_dir,
            force=True,
            chunk_size=80,
            chunk_overlap=0,
        )
        assert first_count > 1

        target_file.write_text("Short replacement document.")

        ingest_documents(
            temp_docs_dir,
            force=False,
            chunk_size=80,
            chunk_overlap=0,
        )

        store = get_vector_store(temp_persist_dir)
        records = store.get(include=["metadatas"])
        target_chunks = [
            metadata
            for metadata in records["metadatas"]
            if metadata["source"] == str(target_file)
        ]

        assert len(target_chunks) == 1

