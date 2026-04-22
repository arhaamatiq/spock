"""
Session Management Module for Spock AI RAG System

This module provides in-memory chat history management.
History is stored per session_id and cleared on process restart.

Key features:
- In-memory storage (no persistence - clears on restart)
- Configurable maximum history window
- Simple message list structure compatible with LangChain

Usage:
    from spock_rag.session import SessionManager
    
    manager = SessionManager(max_history=10)
    
    # Add messages
    manager.add_message("session123", "user", "What is Python?")
    manager.add_message("session123", "assistant", "Python is a programming language...")
    
    # Get history
    history = manager.get_history("session123")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import uuid

from spock_rag.config import get_settings
from spock_rag.logging_config import get_logger


logger = get_logger(__name__)


# Type alias for message roles
MessageRole = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """
    A single chat message.
    
    Attributes:
        role: Who sent the message ("user", "assistant", or "system").
        content: The message text.
    """
    role: MessageRole
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """
    A chat session containing message history.
    
    Attributes:
        session_id: Unique identifier for this session.
        messages: List of messages in chronological order.
        max_messages: Maximum number of messages to retain.
    """
    session_id: str
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 20  # 10 turns * 2 messages per turn
    
    def add_message(self, role: MessageRole, content: str) -> None:
        """
        Add a message to the session history.
        
        If adding this message would exceed max_messages, the oldest
        messages are removed (keeping the most recent ones).
        
        Args:
            role: Who sent the message.
            content: The message text.
        """
        self.messages.append(Message(role=role, content=content))
        
        # Trim to max_messages if needed
        if len(self.messages) > self.max_messages:
            # Remove oldest messages (keep most recent)
            excess = len(self.messages) - self.max_messages
            self.messages = self.messages[excess:]
            logger.debug(f"Trimmed {excess} old messages from session {self.session_id}")
    
    def clear(self) -> None:
        """Clear all messages from this session."""
        self.messages = []
        logger.debug(f"Cleared session {self.session_id}")
    
    def get_messages_as_dicts(self) -> List[Dict[str, str]]:
        """Get messages as list of dictionaries."""
        return [msg.to_dict() for msg in self.messages]


class SessionManager:
    """
    Manages chat sessions and their message histories.
    
    Sessions are stored in-memory and will be lost when the process restarts.
    This matches the expected behavior where refreshing the page clears history.
    
    Attributes:
        max_history: Maximum number of conversation turns to remember per session.
                    Each turn = 1 user message + 1 assistant response.
    
    Example:
        manager = SessionManager(max_history=10)
        
        # Start a new session
        session_id = manager.create_session()
        
        # Add messages
        manager.add_message(session_id, "user", "Hello!")
        manager.add_message(session_id, "assistant", "Hi! How can I help?")
        
        # Get history for prompting
        history = manager.get_history(session_id)
    """
    
    def __init__(self, max_history: Optional[int] = None):
        """
        Initialize the session manager.
        
        Args:
            max_history: Maximum conversation turns per session.
                        Uses config default if not specified.
        """
        settings = get_settings()
        self.max_history = max_history or settings.MAX_HISTORY
        
        # In-memory storage: session_id -> Session
        self._sessions: Dict[str, Session] = {}
        
        logger.debug(f"SessionManager initialized with max_history={self.max_history}")
    
    def create_session(self) -> str:
        """
        Create a new session with a unique ID.
        
        Returns:
            The new session's ID.
        """
        session_id = str(uuid.uuid4())
        
        # max_messages = turns * 2 (user + assistant per turn)
        max_messages = self.max_history * 2
        
        self._sessions[session_id] = Session(
            session_id=session_id,
            max_messages=max_messages,
        )
        
        logger.debug(f"Created new session: {session_id}")
        return session_id
    
    def get_or_create_session(self, session_id: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: The session ID to look up.
        
        Returns:
            The Session object.
        """
        if session_id not in self._sessions:
            max_messages = self.max_history * 2
            self._sessions[session_id] = Session(
                session_id=session_id,
                max_messages=max_messages,
            )
            logger.debug(f"Created session on-demand: {session_id}")
        
        return self._sessions[session_id]
    
    def add_message(self, session_id: str, role: MessageRole, content: str) -> None:
        """
        Add a message to a session's history.
        
        If the session doesn't exist, it will be created.
        
        Args:
            session_id: The session to add the message to.
            role: Who sent the message ("user", "assistant", or "system").
            content: The message text.
        """
        session = self.get_or_create_session(session_id)
        session.add_message(role, content)
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the message history for a session.
        
        Args:
            session_id: The session to get history for.
        
        Returns:
            List of message dictionaries with "role" and "content" keys.
            Returns empty list if session doesn't exist.
        """
        if session_id not in self._sessions:
            return []
        
        return self._sessions[session_id].get_messages_as_dicts()
    
    def get_history_as_string(self, session_id: str) -> str:
        """
        Get the message history formatted as a string.
        
        Useful for including in prompts.
        
        Args:
            session_id: The session to get history for.
        
        Returns:
            Formatted string with chat history.
        """
        history = self.get_history(session_id)
        
        if not history:
            return ""
        
        formatted_lines = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_lines.append(f"{role}: {content}")
        
        return "\n".join(formatted_lines)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages from a session.
        
        The session will still exist but with empty history.
        
        Args:
            session_id: The session to clear.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.
        
        Args:
            session_id: The session to delete.
        
        Returns:
            True if the session was deleted, False if it didn't exist.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.
        
        Returns:
            List of session ID strings.
        """
        return list(self._sessions.keys())
    
    def get_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of sessions.
        """
        return len(self._sessions)

