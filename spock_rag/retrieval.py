"""
Retrieval Module for Spock AI RAG System

This module handles document retrieval from the ChromaDB vector store.

Key features:
- Load/create persistent ChromaDB store
- Configurable number of retrieved documents (k)
- Optional minimum relevance score filtering

Usage:
    from spock_rag.retrieval import get_retriever, retrieve_with_scores
    
    # Get a retriever for use in chains
    retriever = get_retriever(k=4)
    
    # Or retrieve with relevance scores
    docs = retrieve_with_scores("What is Python?", k=4, min_score=0.5)
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from spock_rag.config import get_settings
from spock_rag.logging_config import get_logger
from spock_rag.utils import ensure_directory


logger = get_logger(__name__)


# These local source files are small, high-signal profile documents. They give
# the chatbot a grounded baseline for broad recruiter questions where embeddings
# may not retrieve the right chunk from vague wording like "where are you from?"
PROFILE_SOURCE_FILES = (
    "identity_snapshot.txt",
    "origin_story.txt",
    "personal_positioning.txt",
    "projects.txt",
)

PROFILE_QUERY_EXPANSIONS = (
    "Arhaam Atiq personal background origin Bangalore India international student San Jose State University",
    "Arhaam Atiq education Applied Math CS SJSU Physics UL Solutions AI",
    "Arhaam Atiq projects skills AgentGate HireSignal DiscoverAI Applied Engineering leadership",
    "Arhaam Atiq recruiter interview strengths production AI engineering evaluation metrics",
)

PROFILE_TERMS = (
    "arhaam",
    "atiq",
    "you",
    "your",
    "yourself",
    "candidate",
    "student",
    "background",
    "origin",
    "from",
    "built",
    "project",
    "skill",
    "strength",
    "hire",
    "recruiter",
    "experience",
    "education",
    "university",
    "sjsu",
    "agentgate",
    "hiresignal",
    "discoverai",
    "ae",
    "chatbot",
    "ul",
)


# =============================================================================
# Vector Store Access
# =============================================================================


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get the OpenAI embeddings instance.
    
    Returns:
        Configured OpenAIEmbeddings instance.
    """
    settings = get_settings()
    
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def get_vector_store(persist_dir: Optional[Path] = None) -> Chroma:
    """
    Get the ChromaDB vector store (loads existing or creates new).
    
    Args:
        persist_dir: Directory for persistence. Uses config default if not specified.
    
    Returns:
        Chroma vector store instance.
    
    Raises:
        RuntimeError: If the vector store cannot be loaded or created.
    """
    settings = get_settings()
    persist_dir = persist_dir or settings.PERSIST_DIR
    
    try:
        ensure_directory(persist_dir)
        
        embeddings = get_embeddings()
        
        vector_store = Chroma(
            collection_name="spock_rag",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        
        # Log collection info
        collection_count = vector_store._collection.count()
        logger.debug(f"Loaded vector store with {collection_count} documents")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise RuntimeError(f"Failed to load vector store: {e}") from e


def get_retriever(
    k: Optional[int] = None,
    persist_dir: Optional[Path] = None,
) -> VectorStoreRetriever:
    """
    Get a retriever from the vector store.
    
    The retriever can be used directly in LangChain chains.
    
    Args:
        k: Number of documents to retrieve. Uses config default if not specified.
        persist_dir: Directory for persistence. Uses config default if not specified.
    
    Returns:
        VectorStoreRetriever instance.
    
    Example:
        retriever = get_retriever(k=5)
        docs = retriever.invoke("What is machine learning?")
    """
    settings = get_settings()
    k = k or settings.RETRIEVAL_K
    
    vector_store = get_vector_store(persist_dir)
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    
    logger.debug(f"Created retriever with k={k}")
    return retriever


# =============================================================================
# Profile-Aware Retrieval Helpers
# =============================================================================


def is_candidate_profile_question(query: str) -> bool:
    """
    Return True when a query is likely about Arhaam or this portfolio chatbot.

    Recruiters often ask short, human questions such as "where are you from?"
    Those are semantically about Arhaam even though they do not name him.
    """
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    return any(term in tokens for term in PROFILE_TERMS)


def build_retrieval_queries(query: str) -> List[str]:
    """
    Build one or more search queries for retrieval.

    The original wording is always preserved. For candidate/profile questions we
    add high-signal Arhaam-specific expansions to avoid missing obvious facts
    behind vague pronouns or conversational phrasing.
    """
    queries = [query.strip()]

    if is_candidate_profile_question(query):
        queries.extend(PROFILE_QUERY_EXPANSIONS)

    deduped: List[str] = []
    seen = set()
    for item in queries:
        normalized = " ".join(item.split()).lower()
        if item and normalized not in seen:
            deduped.append(item)
            seen.add(normalized)

    return deduped


@lru_cache(maxsize=8)
def _load_profile_documents_from_dir(docs_dir: str) -> Tuple[Document, ...]:
    """Load curated profile documents directly from disk."""
    directory = Path(docs_dir)
    documents: List[Document] = []

    for filename in PROFILE_SOURCE_FILES:
        path = directory / filename
        if not path.exists() or not path.is_file():
            continue

        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to load profile fallback document {path}: {e}")
            continue

        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "chunk_index": 0,
                        "doc_id": f"profile::{filename}",
                    },
                )
            )

    return tuple(documents)


def load_profile_context_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    """
    Load curated profile documents for grounded fallback context.

    This intentionally does not require the vector store or embeddings to be
    available, so personal questions can still be answered from committed docs.
    """
    directory = docs_dir or Path(os.getenv("DOCS_DIR", "./data/docs"))
    return list(_load_profile_documents_from_dir(str(directory)))


def has_profile_fallback(query: str) -> bool:
    """Return whether a profile-like query can be answered from local docs."""
    return is_candidate_profile_question(query) and bool(load_profile_context_documents())


# =============================================================================
# Advanced Retrieval with Scores
# =============================================================================


def retrieve_with_scores(
    query: str,
    k: Optional[int] = None,
    min_score: Optional[float] = None,
    persist_dir: Optional[Path] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with their relevance scores.
    
    This is useful when you need to filter by relevance or display
    confidence scores to users.
    
    Args:
        query: The search query.
        k: Number of documents to retrieve. Uses config default if not specified.
        min_score: Minimum relevance score (0.0 to 1.0). Documents below
                  this threshold are filtered out. Uses config default if not specified.
        persist_dir: Directory for persistence. Uses config default if not specified.
    
    Returns:
        List of (Document, score) tuples, sorted by relevance (highest first).
        
    Note:
        ChromaDB returns distance scores where lower = more similar.
        We convert these to similarity scores where higher = more similar.
    """
    settings = get_settings()
    k = k or settings.RETRIEVAL_K
    min_score = min_score if min_score is not None else settings.MIN_RELEVANCE_SCORE
    
    vector_store = get_vector_store(persist_dir)
    
    try:
        # ChromaDB returns (document, distance) pairs using cosine distance
        # Distance ranges from 0 (identical) to 2 (opposite) for cosine
        results_with_distances = vector_store.similarity_search_with_score(query, k=k)
        
        logger.debug(f"Retrieved {len(results_with_distances)} documents for query: '{query[:50]}...'")
        
        # Convert cosine distance to similarity score [0, 1]
        # Cosine distance: 0 = identical, 2 = opposite
        # Similarity: 1 = identical, 0 = opposite
        # Formula: sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        results = []
        for doc, distance in results_with_distances:
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            results.append((doc, similarity))
        
        # Filter by minimum similarity score
        if min_score > 0:
            filtered_results = [
                (doc, sim) for doc, sim in results if sim >= min_score
            ]
            
            if len(filtered_results) < len(results):
                logger.debug(
                    f"Filtered {len(results) - len(filtered_results)} documents "
                    f"below min_score={min_score}"
                )
            
            results = filtered_results
        
        return results
        
    except Exception as e:
        logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
        raise


def retrieve_documents(
    query: str,
    k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """
    Retrieve documents without scores (convenience function).
    
    Args:
        query: The search query.
        k: Number of documents to retrieve.
        min_score: Minimum relevance score filter.
    
    Returns:
        List of Document objects.
    """
    results = retrieve_profile_aware_documents(query, k, min_score)
    return [doc for doc, _ in results]


def retrieve_profile_aware_documents(
    query: str,
    k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with candidate-aware query expansion and profile fallback.

    For broad personal/recruiter questions, this combines vector retrieval with
    curated profile files from `data/docs` so the LLM has enough grounded context
    to infer useful answers instead of refusing.
    """
    if k is None:
        k = get_settings().RETRIEVAL_K
    results_by_key: Dict[str, Tuple[Document, float]] = {}

    for search_query in build_retrieval_queries(query):
        try:
            for doc, score in retrieve_with_scores(search_query, k, min_score):
                key = (
                    doc.metadata.get("doc_id")
                    or f"{doc.metadata.get('source', '')}:{doc.metadata.get('chunk_index', '')}"
                    or doc.page_content[:200]
                )
                existing = results_by_key.get(key)
                if existing is None or score > existing[1]:
                    results_by_key[key] = (doc, score)
        except Exception as e:
            logger.warning(
                f"Profile-aware retrieval skipped query '{search_query[:50]}...': {e}"
            )

    if is_candidate_profile_question(query):
        for doc in load_profile_context_documents():
            key = doc.metadata.get("doc_id", doc.page_content[:200])
            results_by_key.setdefault(key, (doc, 1.0))

    results = list(results_by_key.values())
    results.sort(key=lambda item: item[1], reverse=True)
    return results[: max(k, len(PROFILE_SOURCE_FILES))]


def format_context(documents: List[Document], include_source: bool = True) -> str:
    """
    Format retrieved documents into a context string for the prompt.
    
    Args:
        documents: List of Document objects.
        include_source: Whether to include source file info.
    
    Returns:
        Formatted context string.
    """
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        chunk_index = doc.metadata.get("chunk_index", 0)
        
        if include_source:
            header = f"[Source {i}: {source} (chunk {chunk_index})]"
        else:
            header = f"[Context {i}]"
        
        context_parts.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)


def check_store_exists() -> bool:
    """
    Check if the vector store exists and has documents.
    
    Returns:
        True if the store exists and contains documents, False otherwise.
    """
    settings = get_settings()
    
    try:
        vector_store = get_vector_store(settings.PERSIST_DIR)
        count = vector_store._collection.count()
        return count > 0
    except Exception:
        return False

