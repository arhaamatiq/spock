"""
Document Ingestion Module for Spock AI RAG System

This module handles loading, splitting, and storing documents in ChromaDB.

Key features:
- Loads PDF, TXT, and Markdown files
- Splits documents into chunks with configurable size/overlap
- Preserves metadata (source file, document ID, chunk index)
- Avoids re-ingesting unchanged documents

Usage:
    from spock_rag.ingest import ingest_documents
    
    # Ingest all documents from a directory
    ingest_documents("./data/docs")
    
    # Force re-ingestion (ignore existing data)
    ingest_documents("./data/docs", force=True)
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from spock_rag.config import get_settings
from spock_rag.logging_config import get_logger
from spock_rag.utils import (
    hash_file,
    ensure_directory,
    get_file_extension,
    is_supported_file,
)


logger = get_logger(__name__)


# =============================================================================
# Document Loading
# =============================================================================


def load_single_document(file_path: Path) -> List[Document]:
    """
    Load a single document file based on its extension.
    
    Supports: PDF, TXT, MD/Markdown
    
    Args:
        file_path: Path to the document file.
    
    Returns:
        List of Document objects (PDFs may have multiple pages).
        Returns empty list if loading fails.
    """
    ext = get_file_extension(file_path)
    
    try:
        if ext == "pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == "txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif ext in ("md", "markdown"):
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} document(s) from {file_path}")
        return documents
        
    except Exception as e:
        # Log error but don't crash - skip this file and continue
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def load_documents(docs_dir: Path) -> List[Document]:
    """
    Load all supported documents from a directory (recursively).
    
    Args:
        docs_dir: Path to the directory containing documents.
    
    Returns:
        List of Document objects from all successfully loaded files.
    
    Raises:
        FileNotFoundError: If the directory doesn't exist.
    """
    docs_dir = Path(docs_dir)
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {docs_dir}")
    
    all_documents: List[Document] = []
    files_processed = 0
    files_failed = 0
    
    # Find all files recursively
    for file_path in docs_dir.rglob("*"):
        if file_path.is_file() and is_supported_file(file_path):
            docs = load_single_document(file_path)
            if docs:
                # Add source metadata to each document
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["file_hash"] = hash_file(file_path)
                all_documents.extend(docs)
                files_processed += 1
            else:
                files_failed += 1
    
    logger.info(
        f"Loaded {len(all_documents)} documents from {files_processed} files "
        f"({files_failed} files failed)"
    )
    
    return all_documents


# =============================================================================
# Document Splitting
# =============================================================================


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Uses RecursiveCharacterTextSplitter which intelligently splits on
    paragraph boundaries, sentences, and words (in that order of preference).
    
    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum size of each chunk in characters.
                   Uses config default if not specified.
        chunk_overlap: Overlap between consecutive chunks.
                      Uses config default if not specified.
    
    Returns:
        List of Document objects (chunks) with preserved metadata
        plus additional chunk_index metadata.
    """
    settings = get_settings()
    
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    logger.debug(f"Splitting documents with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Create the text splitter
    # RecursiveCharacterTextSplitter tries to split on:
    # 1. Paragraphs ("\n\n")
    # 2. Lines ("\n")
    # 3. Sentences (". ")
    # 4. Words (" ")
    # 5. Characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    # Split all documents
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata for traceability
    # Group chunks by source file to assign indices
    source_chunk_counts: Dict[str, int] = {}
    
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        
        # Get and increment the chunk index for this source
        chunk_index = source_chunk_counts.get(source, 0)
        chunk.metadata["chunk_index"] = chunk_index
        source_chunk_counts[source] = chunk_index + 1
        
        # Create a unique document ID
        chunk.metadata["doc_id"] = f"{source}::chunk_{chunk_index}"
    
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks


# =============================================================================
# Vector Store Management
# =============================================================================


def get_ingestion_metadata_path(persist_dir: Path) -> Path:
    """Get the path to the ingestion metadata file."""
    return persist_dir / "ingestion_metadata.json"


def load_ingestion_metadata(persist_dir: Path) -> Dict[str, Any]:
    """Load the ingestion metadata from disk."""
    metadata_path = get_ingestion_metadata_path(persist_dir)
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ingestion metadata: {e}")
    
    return {"file_hashes": {}}


def save_ingestion_metadata(persist_dir: Path, metadata: Dict[str, Any]) -> None:
    """Save the ingestion metadata to disk."""
    metadata_path = get_ingestion_metadata_path(persist_dir)
    
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save ingestion metadata: {e}")


def get_vector_store(
    persist_dir: Optional[Path] = None,
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> Chroma:
    """
    Get or create the ChromaDB vector store.
    
    Args:
        persist_dir: Directory for persistence. Uses config default if not specified.
        embeddings: OpenAI embeddings instance. Creates one if not specified.
    
    Returns:
        Chroma vector store instance.
    """
    settings = get_settings()
    
    persist_dir = persist_dir or settings.PERSIST_DIR
    ensure_directory(persist_dir)
    
    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    
    # Create/load ChromaDB with persistence
    # Use cosine similarity for consistent distance-to-similarity conversion
    vector_store = Chroma(
        collection_name="spock_rag",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    
    return vector_store


# =============================================================================
# Main Ingestion Function
# =============================================================================


def ingest_documents(
    docs_dir: Path,
    force: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> int:
    """
    Ingest documents from a directory into the vector store.
    
    This is the main entry point for document ingestion. It:
    1. Loads all supported documents from the directory
    2. Splits them into chunks with metadata
    3. Stores them in ChromaDB with persistence
    4. Tracks file hashes to avoid re-ingesting unchanged files
    
    Args:
        docs_dir: Path to the directory containing documents.
        force: If True, re-ingest all documents even if unchanged.
        chunk_size: Override the default chunk size.
        chunk_overlap: Override the default chunk overlap.
    
    Returns:
        Number of chunks ingested.
    
    Raises:
        FileNotFoundError: If the documents directory doesn't exist.
    """
    settings = get_settings()
    docs_dir = Path(docs_dir)
    
    logger.info(f"Starting document ingestion from {docs_dir}")
    
    # Load existing metadata to check for changes
    metadata = load_ingestion_metadata(settings.PERSIST_DIR)
    existing_hashes = metadata.get("file_hashes", {})
    
    # Load all documents
    all_documents = load_documents(docs_dir)
    
    if not all_documents:
        logger.warning("No documents found to ingest")
        return 0
    
    # Filter out unchanged documents (unless force=True)
    if not force:
        documents_to_process = []
        new_hashes = {}
        
        for doc in all_documents:
            source = doc.metadata.get("source", "")
            file_hash = doc.metadata.get("file_hash", "")
            
            # Check if this file has changed
            if source not in existing_hashes or existing_hashes[source] != file_hash:
                documents_to_process.append(doc)
            
            new_hashes[source] = file_hash
        
        if not documents_to_process:
            logger.info("All documents are unchanged, skipping ingestion")
            return 0
        
        logger.info(
            f"Found {len(documents_to_process)} new/changed documents "
            f"(skipping {len(all_documents) - len(documents_to_process)} unchanged)"
        )
        all_documents = documents_to_process
    else:
        # Collect all file hashes for metadata
        new_hashes = {
            doc.metadata.get("source", ""): doc.metadata.get("file_hash", "")
            for doc in all_documents
        }
    
    # Split documents into chunks
    chunks = split_documents(all_documents, chunk_size, chunk_overlap)
    
    if not chunks:
        logger.warning("No chunks produced after splitting")
        return 0
    
    # Get the vector store
    vector_store = get_vector_store()
    
    # Add documents to the vector store
    # ChromaDB handles persistence automatically
    try:
        # Extract texts and metadata for batch addition
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.metadata["doc_id"] for chunk in chunks]
        
        # Add to vector store
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        logger.info(f"Successfully ingested {len(chunks)} chunks into vector store")
        
    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {e}")
        raise
    
    # Update metadata with new file hashes
    metadata["file_hashes"] = new_hashes
    save_ingestion_metadata(settings.PERSIST_DIR, metadata)
    
    return len(chunks)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for document ingestion."""
    import argparse
    
    from spock_rag.logging_config import setup_logging
    
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Spock AI RAG knowledge base",
        epilog=(
            "Note: If vector store configuration changes (e.g., similarity metric), "
            "delete the PERSIST_DIR directory and re-ingest all documents."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docs",
        type=str,
        required=True,
        help="Path to the directory containing documents to ingest",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all documents (ignore existing data)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override the default chunk size",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override the default chunk overlap",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    
    try:
        num_chunks = ingest_documents(
            docs_dir=Path(args.docs),
            force=args.force,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"\nIngestion complete! {num_chunks} chunks added to knowledge base.")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nError: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\nError: Ingestion failed - {e}")
        exit(1)


if __name__ == "__main__":
    main()

