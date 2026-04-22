"""
Command Line Interface for Spock AI RAG System

This module provides interactive CLI commands for:
- Chatting with the RAG system (with streaming output)
- Managing sessions
- Testing the system

Usage:
    # Interactive chat mode
    python -m spock_rag.cli chat
    
    # Chat with debug logging
    python -m spock_rag.cli chat --debug

Commands in chat mode:
    /new     - Start a new session (clear history)
    /history - Show conversation history
    /quit    - Exit the chat
    /help    - Show available commands
"""

import argparse
import sys
import uuid

from spock_rag.config import get_settings
from spock_rag.logging_config import setup_logging, get_logger
from spock_rag.rag_engine import RAGEngine
from spock_rag.retrieval import check_store_exists


logger = get_logger(__name__)


# =============================================================================
# Chat Interface
# =============================================================================


def print_welcome() -> None:
    """Print the welcome message."""
    print("\n" + "=" * 60)
    print("  🖖 SPOCK AI - RAG Chat Interface")
    print("=" * 60)
    print("\nCommands:")
    print("  /new     - Start a new session (clear history)")
    print("  /history - Show conversation history")
    print("  /quit    - Exit the chat")
    print("  /help    - Show this help message")
    print("\n" + "-" * 60)


def print_history(engine: RAGEngine, session_id: str) -> None:
    """Print the conversation history for a session."""
    history = engine.get_session_history(session_id)
    
    if not history:
        print("\n[No conversation history yet]")
        return
    
    print("\n" + "-" * 40)
    print("Conversation History:")
    print("-" * 40)
    
    for msg in history:
        role = msg["role"].upper()
        content = msg["content"]
        
        if role == "USER":
            print(f"\n👤 You: {content}")
        else:
            print(f"\n🖖 Spock: {content}")
    
    print("\n" + "-" * 40)


def run_chat(debug: bool = False) -> None:
    """
    Run the interactive chat interface.
    
    Args:
        debug: If True, enable debug logging.
    """
    # Set up logging
    setup_logging(level="DEBUG" if debug else "INFO")
    
    # Check if vector store exists
    if not check_store_exists():
        print("\n⚠️  Warning: No documents have been ingested yet!")
        print("Run the following command to ingest documents:")
        print("  python -m spock_rag.ingest --docs ./data/docs")
        print("\nYou can still chat, but I won't have any knowledge base to draw from.")
        print("\nNote: If vector store configuration changes, delete PERSIST_DIR and re-ingest.\n")
    
    # Initialize the RAG engine
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG engine: {e}")
        print("Make sure your .env file is configured with a valid OPENAI_API_KEY")
        sys.exit(1)
    
    # Create a new session
    session_id = str(uuid.uuid4())
    
    print_welcome()
    print(f"Session ID: {session_id[:8]}...")
    print("-" * 60)
    print("\nAsk me anything! (or type /help for commands)\n")
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command == "/quit" or command == "/exit":
                    print("\n🖖 Live long and prosper!")
                    break
                
                elif command == "/new":
                    session_id = str(uuid.uuid4())
                    print(f"\n✨ New session started: {session_id[:8]}...")
                    print()
                    continue
                
                elif command == "/history":
                    print_history(engine, session_id)
                    print()
                    continue
                
                elif command == "/help":
                    print("\nCommands:")
                    print("  /new     - Start a new session (clear history)")
                    print("  /history - Show conversation history")
                    print("  /quit    - Exit the chat")
                    print("  /help    - Show this help message")
                    print()
                    continue
                
                else:
                    print(f"\n❓ Unknown command: {command}")
                    print("Type /help for available commands\n")
                    continue
            
            # Stream the response
            print("\n🖖 Spock: ", end="", flush=True)
            
            for chunk in engine.stream_answer(user_input, session_id):
                print(chunk, end="", flush=True)
            
            print("\n")  # Newline after response
            
        except KeyboardInterrupt:
            print("\n\n🖖 Live long and prosper!")
            break
        
        except EOFError:
            print("\n\n🖖 Live long and prosper!")
            break
        
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Spock AI RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m spock_rag.cli chat          Start interactive chat
  python -m spock_rag.cli chat --debug  Chat with debug logging
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.command == "chat":
        run_chat(debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

