"""
RAG Engine Module for Spock AI RAG System

This is the main orchestration module that ties together all components:
- Document retrieval from ChromaDB
- Session management for chat history
- Prompt construction
- LLM generation with streaming support

Usage:
    from spock_rag.rag_engine import RAGEngine
    
    engine = RAGEngine()
    
    # Full response
    answer = engine.answer("What is Python?", session_id="user123")
    
    # Streaming response
    for chunk in engine.stream_answer("What is Python?", session_id="user123"):
        print(chunk, end="", flush=True)
"""

from typing import Iterator, List, Optional, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from spock_rag.config import get_settings
from spock_rag.logging_config import get_logger
from spock_rag.prompts import (
    get_rag_prompt,
    get_standalone_question_prompt,
    NO_CONTEXT_MESSAGE,
    ERROR_MESSAGE,
    EMPTY_QUESTION_MESSAGE,
)
from spock_rag.retrieval import (
    get_retriever,
    retrieve_documents,
    format_context,
    check_store_exists,
    has_profile_fallback,
)
from spock_rag.session import SessionManager
from spock_rag.utils import validate_question


logger = get_logger(__name__)


class RAGEngine:
    """
    Main RAG engine that orchestrates retrieval and generation.
    
    This class provides both blocking and streaming interfaces for
    answering questions using the RAG approach.
    
    Attributes:
        session_manager: Manages chat history per session.
        retrieval_k: Number of documents to retrieve.
    
    Example:
        engine = RAGEngine()
        
        # Ask a question (new session created automatically)
        session_id = "user123"
        answer = engine.answer("What is Python?", session_id)
        print(answer)
        
        # Follow-up question (history-aware)
        answer = engine.answer("What are its main features?", session_id)
        print(answer)
        
        # Streaming
        for token in engine.stream_answer("Tell me more", session_id):
            print(token, end="", flush=True)
    """
    
    def __init__(
        self,
        retrieval_k: Optional[int] = None,
        max_history: Optional[int] = None,
    ):
        """
        Initialize the RAG engine.
        
        Args:
            retrieval_k: Number of documents to retrieve per query.
                        Uses config default if not specified.
            max_history: Maximum conversation turns to remember.
                        Uses config default if not specified.
        """
        settings = get_settings()
        
        self.retrieval_k = retrieval_k or settings.RETRIEVAL_K
        self.session_manager = SessionManager(max_history=max_history)
        
        # Initialize the LLM
        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.8,
            streaming=True,  # Enable streaming by default
        )
        
        # Initialize prompts
        self._rag_prompt = get_rag_prompt()
        self._standalone_prompt = get_standalone_question_prompt()
        
        logger.info(
            f"RAGEngine initialized with model={settings.OPENAI_MODEL}, "
            f"retrieval_k={self.retrieval_k}"
        )
    
    def _convert_history_to_messages(
        self, 
        history: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """
        Convert session history to LangChain message objects.
        
        Args:
            history: List of message dicts with "role" and "content".
        
        Returns:
            List of LangChain BaseMessage objects.
        """
        messages = []
        
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        
        return messages
    
    def _reformulate_question(
        self,
        question: str,
        chat_history: List[BaseMessage],
    ) -> str:
        """
        Reformulate a follow-up question to be standalone.
        
        This helps with retrieval when the question references
        previous messages (e.g., "What about that?").
        
        Args:
            question: The user's question.
            chat_history: Previous messages in the conversation.
        
        Returns:
            The reformulated standalone question.
        """
        # If no history, the question is already standalone
        if not chat_history:
            return question
        
        try:
            # Use the LLM to reformulate the question
            prompt = self._standalone_prompt.format_messages(
                chat_history=chat_history,
                question=question,
            )
            
            response = self._llm.invoke(prompt)
            reformulated = response.content.strip()
            
            logger.debug(f"Reformulated question: '{question}' -> '{reformulated}'")
            return reformulated
            
        except Exception as e:
            # If reformulation fails, use the original question
            logger.warning(f"Question reformulation failed: {e}")
            return question
    
    def _retrieve_context(self, question: str) -> str:
        """
        Retrieve relevant documents and format as context.
        
        Args:
            question: The question to search for.
        
        Returns:
            Formatted context string.
        """
        try:
            documents = retrieve_documents(question, k=self.retrieval_k)
            
            if not documents:
                logger.warning(f"No documents retrieved for: '{question[:50]}...'")
                return ""
            
            context = format_context(documents)
            logger.debug(f"Retrieved {len(documents)} documents for context")
            return context
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ""
    
    def answer(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        Generate a complete answer to a question.
        
        This method:
        1. Validates the question
        2. Gets chat history for the session
        3. Reformulates the question if needed (history-aware)
        4. Retrieves relevant documents
        5. Generates the answer
        6. Updates session history
        
        Args:
            question: The user's question.
            session_id: Unique identifier for the chat session.
        
        Returns:
            The complete answer string.
        
        Raises:
            ValueError: If the question is empty.
        """
        # Validate input
        try:
            question = validate_question(question)
        except ValueError:
            return EMPTY_QUESTION_MESSAGE
        
        # Check if vector store exists
        if not check_store_exists() and not has_profile_fallback(question):
            logger.warning("Vector store is empty or doesn't exist")
            return NO_CONTEXT_MESSAGE
        
        try:
            # Get chat history
            history = self.session_manager.get_history(session_id)
            chat_history = self._convert_history_to_messages(history)
            
            # Reformulate question for better retrieval
            search_question = self._reformulate_question(question, chat_history)
            
            # Retrieve context
            context = self._retrieve_context(search_question)
            
            if not context:
                # No context found - provide fallback
                self.session_manager.add_message(session_id, "user", question)
                self.session_manager.add_message(session_id, "assistant", NO_CONTEXT_MESSAGE)
                return NO_CONTEXT_MESSAGE
            
            # Build the prompt
            prompt_messages = self._rag_prompt.format_messages(
                context=context,
                chat_history=chat_history,
                question=question,
            )
            
            # Generate response
            logger.debug(f"Generating answer for: '{question[:50]}...'")
            response = self._llm.invoke(prompt_messages)
            answer = response.content
            
            # Update session history
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            
            logger.info(f"Generated answer for session {session_id[:8]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ERROR_MESSAGE
    
    def stream_answer(
        self,
        question: str,
        session_id: str,
    ) -> Iterator[str]:
        """
        Stream an answer to a question token by token.
        
        This generator yields partial tokens as they're generated,
        allowing for real-time display of the response.
        
        Args:
            question: The user's question.
            session_id: Unique identifier for the chat session.
        
        Yields:
            String chunks of the response as they're generated.
        
        Example:
            for chunk in engine.stream_answer("What is Python?", "user123"):
                print(chunk, end="", flush=True)
            print()  # Newline after streaming completes
        """
        # Validate input
        try:
            question = validate_question(question)
        except ValueError:
            yield EMPTY_QUESTION_MESSAGE
            return
        
        # Check if vector store exists
        if not check_store_exists() and not has_profile_fallback(question):
            logger.warning("Vector store is empty or doesn't exist")
            yield NO_CONTEXT_MESSAGE
            return
        
        try:
            # Get chat history
            history = self.session_manager.get_history(session_id)
            chat_history = self._convert_history_to_messages(history)
            
            # Reformulate question for better retrieval
            search_question = self._reformulate_question(question, chat_history)
            
            # Retrieve context
            context = self._retrieve_context(search_question)
            
            if not context:
                # No context found
                self.session_manager.add_message(session_id, "user", question)
                self.session_manager.add_message(session_id, "assistant", NO_CONTEXT_MESSAGE)
                yield NO_CONTEXT_MESSAGE
                return
            
            # Build the prompt
            prompt_messages = self._rag_prompt.format_messages(
                context=context,
                chat_history=chat_history,
                question=question,
            )
            
            # Stream the response
            logger.debug(f"Streaming answer for: '{question[:50]}...'")
            
            full_response = []
            
            for chunk in self._llm.stream(prompt_messages):
                # Extract content from the chunk
                if hasattr(chunk, 'content') and chunk.content:
                    full_response.append(chunk.content)
                    yield chunk.content
            
            # Combine the full response for history
            complete_answer = "".join(full_response)
            
            # Update session history
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", complete_answer)
            
            logger.info(f"Streamed answer for session {session_id[:8]}...")
            
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield ERROR_MESSAGE
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the chat history for a session.
        
        Args:
            session_id: The session to get history for.
        
        Returns:
            List of message dictionaries.
        """
        return self.session_manager.get_history(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear the chat history for a session.
        
        Args:
            session_id: The session to clear.
        """
        self.session_manager.clear_session(session_id)
        logger.info(f"Cleared session {session_id[:8]}...")
    
    def create_session(self) -> str:
        """
        Create a new chat session.
        
        Returns:
            The new session ID.
        """
        session_id = self.session_manager.create_session()
        logger.debug(f"Created new session: {session_id[:8]}...")
        return session_id


# =============================================================================
# Convenience Functions
# =============================================================================


# Module-level engine instance (lazy initialization)
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    """
    Get the global RAG engine instance (singleton pattern).
    
    Returns:
        The RAGEngine instance.
    """
    global _engine
    
    if _engine is None:
        _engine = RAGEngine()
    
    return _engine


def answer(question: str, session_id: str) -> str:
    """
    Convenience function to get a complete answer.
    
    Args:
        question: The user's question.
        session_id: The session ID.
    
    Returns:
        The complete answer string.
    """
    return get_engine().answer(question, session_id)


def stream_answer(question: str, session_id: str) -> Iterator[str]:
    """
    Convenience function to stream an answer.
    
    Args:
        question: The user's question.
        session_id: The session ID.
    
    Yields:
        String chunks of the response.
    """
    yield from get_engine().stream_answer(question, session_id)

