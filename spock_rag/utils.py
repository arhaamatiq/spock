"""
Utility Functions for Spock AI RAG System

This module contains helper functions used across the application:
- File hashing for change detection
- Input validation
- Directory management
"""

import hashlib
from pathlib import Path
from typing import Union

from spock_rag.logging_config import get_logger


logger = get_logger(__name__)


def hash_file(file_path: Union[str, Path]) -> str:
    """
    Compute MD5 hash of a file's contents.
    
    Used to detect when documents have changed and need re-ingestion.
    MD5 is fast and sufficient for change detection (not security).
    
    Args:
        file_path: Path to the file to hash.
    
    Returns:
        Hexadecimal string of the MD5 hash.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        PermissionError: If the file can't be read.
    
    Example:
        hash1 = hash_file("document.pdf")
        # ... document is modified ...
        hash2 = hash_file("document.pdf")
        if hash1 != hash2:
            print("Document changed, re-ingestion needed")
    """
    file_path = Path(file_path)
    
    # Read file in chunks to handle large files efficiently
    hasher = hashlib.md5()
    
    with open(file_path, "rb") as f:
        # Read in 64KB chunks
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def hash_content(content: str) -> str:
    """
    Compute MD5 hash of a string's contents.
    
    Args:
        content: The string to hash.
    
    Returns:
        Hexadecimal string of the MD5 hash.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def validate_question(question: str) -> str:
    """
    Validate and clean a user question.
    
    Ensures the question is not empty and strips whitespace.
    
    Args:
        question: The user's question string.
    
    Returns:
        The cleaned question string.
    
    Raises:
        ValueError: If the question is empty or whitespace-only.
    
    Example:
        clean_q = validate_question("  What is Python?  ")
        # Returns: "What is Python?"
        
        validate_question("")  # Raises ValueError
    """
    if question is None:
        raise ValueError("Question cannot be None")
    
    # Strip whitespace
    cleaned = question.strip()
    
    if not cleaned:
        raise ValueError("Question cannot be empty or whitespace-only")
    
    return cleaned


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory.
    
    Returns:
        The Path object for the directory.
    
    Example:
        docs_dir = ensure_directory("./data/docs")
        # Directory now exists
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the lowercase file extension without the dot.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        Lowercase extension string (e.g., "pdf", "txt", "md").
    
    Example:
        ext = get_file_extension("document.PDF")
        # Returns: "pdf"
    """
    return Path(file_path).suffix.lower().lstrip(".")


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file type is supported for ingestion.
    
    Supported types: PDF, TXT, MD (Markdown)
    
    Args:
        file_path: Path to the file.
    
    Returns:
        True if the file type is supported, False otherwise.
    """
    supported_extensions = {"pdf", "txt", "md", "markdown"}
    ext = get_file_extension(file_path)
    return ext in supported_extensions

