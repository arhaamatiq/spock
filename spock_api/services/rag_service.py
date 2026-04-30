"""
RAG Service Adapter

Thin adapter layer that wraps spock_rag for use in the API.
Handles chat history from client and provides both blocking and streaming interfaces.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Iterator, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from spock_api.core.logging import get_logger
from spock_api.core.settings import get_settings
from spock_api.schemas import ChatMessage

# Import from existing RAG package
from spock_rag.config import get_settings as get_rag_settings
from spock_rag.prompts import (
    ERROR_MESSAGE,
    NO_CONTEXT_MESSAGE,
    get_rag_prompt,
    get_standalone_question_prompt,
)
from spock_rag.retrieval import (
    check_store_exists,
    format_context,
    has_profile_fallback,
    retrieve_profile_aware_documents,
)
from spock_rag.utils import validate_question


logger = get_logger(__name__)

# Thread pool for running sync RAG operations
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="rag_worker")


class RAGService:
    """
    Service class for RAG operations.
    
    This is a stateless service that:
    - Accepts chat history from the client (no server-side session storage)
    - Provides both sync and async interfaces
    - Is safe for concurrent requests
    
    Example:
        service = RAGService()
        
        # Blocking call
        answer = await service.generate_answer("What is Python?", chat_history=[])
        
        # Streaming call
        async for token in service.stream_answer("What is Python?", chat_history=[]):
            print(token, end="")
    """
    
    def __init__(
        self,
        retrieval_k: Optional[int] = None,
    ):
        """
        Initialize the RAG service.
        
        Args:
            retrieval_k: Number of documents to retrieve per query.
                        Uses config default if not specified.
        """
        api_settings = get_settings()
        rag_settings = get_rag_settings()
        
        self.retrieval_k = (
            retrieval_k 
            or api_settings.RAG_RETRIEVAL_K 
            or rag_settings.RETRIEVAL_K
        )
        
        # Initialize the LLM (thread-safe, can be shared)
        self._llm = ChatOpenAI(
            model=rag_settings.OPENAI_MODEL,
            api_key=rag_settings.OPENAI_API_KEY,
            temperature=0.8,
            streaming=True,
        )
        
        # Initialize prompts
        self._rag_prompt = get_rag_prompt()
        self._standalone_prompt = get_standalone_question_prompt()
        
        logger.info(
            f"RAGService initialized with model={rag_settings.OPENAI_MODEL}, "
            f"retrieval_k={self.retrieval_k}"
        )
    
    def _convert_history_to_messages(
        self,
        chat_history: List[ChatMessage],
    ) -> List[BaseMessage]:
        """
        Convert API chat history to LangChain message objects.
        
        Args:
            chat_history: List of ChatMessage from the request.
        
        Returns:
            List of LangChain BaseMessage objects.
        """
        messages: List[BaseMessage] = []
        
        for msg in chat_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        return messages
    
    def _reformulate_question(
        self,
        question: str,
        chat_history: List[BaseMessage],
    ) -> str:
        """
        Reformulate a follow-up question to be standalone.
        
        Args:
            question: The user's question.
            chat_history: Previous messages.
        
        Returns:
            The reformulated standalone question.
        """
        if not chat_history:
            return question
        
        try:
            prompt = self._standalone_prompt.format_messages(
                chat_history=chat_history,
                question=question,
            )
            
            response = self._llm.invoke(prompt)
            reformulated = response.content.strip()
            
            logger.debug(f"Reformulated: '{question[:50]}...' -> '{reformulated[:50]}...'")
            return reformulated
            
        except Exception as e:
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
            # Get documents with scores for logging
            results = retrieve_profile_aware_documents(question, k=self.retrieval_k)
            
            if not results:
                logger.warning(f"No documents retrieved for: '{question[:50]}...'")
                return ""
            
            # Log retrieved doc info (IDs and scores)
            for doc, score in results:
                source = doc.metadata.get("source", "unknown")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                logger.info(
                    f"Retrieved doc: source={source}, chunk={chunk_idx}, score={score:.4f}"
                )
            
            # Extract just the documents for formatting
            documents = [doc for doc, _ in results]
            context = format_context(documents)
            
            logger.debug(f"Retrieved {len(documents)} documents for context")
            return context
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ""
    
    def generate_answer_sync(
        self,
        question: str,
        chat_history: List[ChatMessage],
    ) -> str:
        """
        Generate a complete answer synchronously.
        
        Args:
            question: The user's question.
            chat_history: Previous conversation messages.
        
        Returns:
            The complete answer string.
        """
        # Validate input
        try:
            question = validate_question(question)
        except ValueError:
            return "It looks like you didn't ask a question. Please type your question."
        
        # Check if vector store exists
        if not check_store_exists() and not has_profile_fallback(question):
            logger.warning("Vector store is empty or doesn't exist")
            return NO_CONTEXT_MESSAGE
        
        try:
            # Convert chat history
            langchain_history = self._convert_history_to_messages(chat_history)
            
            # Reformulate question for better retrieval
            search_question = self._reformulate_question(question, langchain_history)
            
            # Retrieve context
            context = self._retrieve_context(search_question)
            
            if not context:
                return NO_CONTEXT_MESSAGE
            
            # Build the prompt
            prompt_messages = self._rag_prompt.format_messages(
                context=context,
                chat_history=langchain_history,
                question=question,
            )
            
            # Generate response
            logger.debug(f"Generating answer for: '{question[:50]}...'")
            response = self._llm.invoke(prompt_messages)
            answer = response.content
            
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ERROR_MESSAGE
    
    async def generate_answer(
        self,
        question: str,
        chat_history: List[ChatMessage],
    ) -> str:
        """
        Generate a complete answer asynchronously.
        
        Runs the sync generation in a thread pool to avoid blocking.
        
        Args:
            question: The user's question.
            chat_history: Previous conversation messages.
        
        Returns:
            The complete answer string.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self.generate_answer_sync,
            question,
            chat_history,
        )
    
    def stream_answer_sync(
        self,
        question: str,
        chat_history: List[ChatMessage],
    ) -> Iterator[str]:
        """
        Stream answer tokens synchronously.
        
        This is real token streaming from the LLM.
        
        Args:
            question: The user's question.
            chat_history: Previous conversation messages.
        
        Yields:
            String chunks of the response as they're generated.
        """
        # Validate input
        try:
            question = validate_question(question)
        except ValueError:
            yield "It looks like you didn't ask a question. Please type your question."
            return
        
        # Check if vector store exists
        if not check_store_exists() and not has_profile_fallback(question):
            logger.warning("Vector store is empty or doesn't exist")
            yield NO_CONTEXT_MESSAGE
            return
        
        try:
            # Convert chat history
            langchain_history = self._convert_history_to_messages(chat_history)
            
            # Reformulate question for better retrieval
            search_question = self._reformulate_question(question, langchain_history)
            
            # Retrieve context
            context = self._retrieve_context(search_question)
            
            if not context:
                yield NO_CONTEXT_MESSAGE
                return
            
            # Build the prompt
            prompt_messages = self._rag_prompt.format_messages(
                context=context,
                chat_history=langchain_history,
                question=question,
            )
            
            # Stream the response - real token streaming!
            logger.debug(f"Streaming answer for: '{question[:50]}...'")
            
            for chunk in self._llm.stream(prompt_messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
            
            logger.info("Streaming completed")
            
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield ERROR_MESSAGE
    
    async def stream_answer(
        self,
        question: str,
        chat_history: List[ChatMessage],
    ) -> AsyncIterator[str]:
        """
        Stream answer tokens asynchronously.
        
        Uses a thread to run the sync streaming and yields tokens as they arrive.
        
        Args:
            question: The user's question.
            chat_history: Previous conversation messages.
        
        Yields:
            String chunks of the response as they're generated.
        """
        import queue
        import threading
        
        # Queue for passing tokens from sync thread to async generator
        token_queue: queue.Queue = queue.Queue()
        
        def run_sync_stream():
            """Run the sync stream and put tokens in the queue."""
            try:
                for token in self.stream_answer_sync(question, chat_history):
                    token_queue.put(("token", token))
            except Exception as e:
                token_queue.put(("error", str(e)))
            finally:
                token_queue.put(("done", None))
        
        # Start the sync stream in a thread
        thread = threading.Thread(target=run_sync_stream, daemon=True)
        thread.start()
        
        # Yield tokens as they arrive
        loop = asyncio.get_event_loop()
        
        while True:
            # Non-blocking check with small sleep
            try:
                event_type, value = await loop.run_in_executor(
                    None,
                    lambda: token_queue.get(timeout=0.1),
                )
            except queue.Empty:
                # Check if thread is still alive
                if not thread.is_alive():
                    break
                continue
            
            if event_type == "done":
                break
            elif event_type == "error":
                logger.error(f"Stream error: {value}")
                yield ERROR_MESSAGE
                break
            elif event_type == "token":
                yield value


# =============================================================================
# Service Singleton
# =============================================================================


_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get the RAG service singleton.
    
    Returns:
        The RAGService instance.
    """
    global _service
    
    if _service is None:
        _service = RAGService()
    
    return _service

