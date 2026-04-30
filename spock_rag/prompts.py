"""
Prompt Templates for Spock AI RAG System

This module contains the prompt templates used for RAG generation.
Templates are designed to be easily editable and customizable.

Key features:
- Clear instructions to use provided context
- "I don't know" fallback for insufficient context
- Support for chat history
- Concise answer preference

Usage:
    from spock_rag.prompts import get_rag_prompt, RAG_SYSTEM_PROMPT
    
    prompt = get_rag_prompt()
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# =============================================================================
# System Prompt Template
# =============================================================================

RAG_SYSTEM_PROMPT = """You are Spock AI, Arhaam Atiq's personal portfolio chatbot.

Your job is to answer recruiter, interviewer, and collaborator questions about Arhaam with grounded confidence. Follow these rules:

1. **Use the context as your source of truth**: Base claims on the provided context and chat history. Do not invent facts, dates, employers, metrics, degrees, or locations that are not supported.

2. **Infer when the context supports it**: Do not require exact wording. If the context says Arhaam is an international student from Bangalore, India, you can answer "Where are you from?" directly. If the user says "you" or "your," interpret that as Arhaam unless the conversation clearly says otherwise.

3. **Sell Arhaam thoughtfully**: Lead with the direct answer, then add one or two relevant proof points that make Arhaam sound compelling to a recruiter. Prefer concrete evidence: shipped projects, metrics, leadership, production thinking, evaluation discipline, and user impact.

4. **Be honest about gaps**: If a specific fact is not in the context, say what is known and what is not. Avoid generic refusals when partial context can answer the spirit of the question.

5. **Stay focused and conversational**: Keep answers natural, specific, and concise unless the user asks for depth.

CONTEXT:
{context}

---

If the context above is empty or says "No relevant context found", respond with:
"I don't have enough grounded context to answer that yet."
"""


# =============================================================================
# User Message Template
# =============================================================================

RAG_USER_TEMPLATE = """{question}"""


# =============================================================================
# Prompt Construction Functions
# =============================================================================


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get the RAG prompt template for chat completion.
    
    This prompt includes:
    - System message with RAG instructions
    - Placeholder for chat history
    - User question
    
    The {context} variable should be filled with retrieved documents.
    The {question} variable should be filled with the user's question.
    Chat history is handled via the messages placeholder.
    
    Returns:
        ChatPromptTemplate ready for use with LangChain.
    
    Example:
        prompt = get_rag_prompt()
        
        # Format with values
        messages = prompt.format_messages(
            context="Python is a programming language...",
            question="What is Python?",
        )
    """
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", RAG_USER_TEMPLATE),
    ])


def get_standalone_question_prompt() -> ChatPromptTemplate:
    """
    Get a prompt for converting a follow-up question to a standalone question.
    
    This is useful for making questions history-aware in RAG pipelines.
    It reformulates questions that reference previous messages into
    self-contained questions that can be used for retrieval.
    
    Returns:
        ChatPromptTemplate for question reformulation.
    
    Example:
        # Chat history: "What is Python?" -> "Python is a programming language"
        # Follow-up: "What are its main features?"
        # Reformulated: "What are the main features of Python?"
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that reformulates questions.

Given a chat history and a follow-up question, reformulate the follow-up question 
to be a standalone question that can be understood without the chat history.

If the question is already standalone or there's no relevant history, 
return the question as-is.

Do NOT answer the question. Only reformulate it."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Reformulate this question to be standalone: {question}

Standalone question:"""),
    ])


# =============================================================================
# Prompt Customization Helpers
# =============================================================================


def create_custom_rag_prompt(
    system_instructions: str,
    include_history: bool = True,
) -> ChatPromptTemplate:
    """
    Create a custom RAG prompt with specified instructions.
    
    Args:
        system_instructions: Custom system prompt text.
                           Must include {context} placeholder.
        include_history: Whether to include chat history placeholder.
    
    Returns:
        ChatPromptTemplate with custom instructions.
    
    Example:
        custom_prompt = create_custom_rag_prompt(
            system_instructions=\"\"\"You are a technical documentation assistant.
            Answer questions based on:
            {context}
            
            Be precise and technical.\"\"\"
        )
    """
    messages = [("system", system_instructions)]
    
    if include_history:
        messages.append(MessagesPlaceholder(variable_name="chat_history", optional=True))
    
    messages.append(("human", "{question}"))
    
    return ChatPromptTemplate.from_messages(messages)


# =============================================================================
# Fallback Messages
# =============================================================================

# Message when no context is available
NO_CONTEXT_MESSAGE = (
    "I don't have enough grounded context to answer that yet. "
    "Try asking about Arhaam's background, projects, skills, or experience."
)

# Message when an error occurs
ERROR_MESSAGE = (
    "I'm sorry, I encountered an error while processing your question. "
    "Please try again or contact Arhaam directly"
)

# Message when the question is empty
EMPTY_QUESTION_MESSAGE = (
    "It looks like you didn't ask a question. Please type your question "
    "and I'll do my best to help."
)

