"""
Main entry point for running spock_rag as a module.

This allows the package to be run with:
    python -m spock_rag

Which defaults to the CLI interface.
"""

from spock_rag.cli import main

if __name__ == "__main__":
    main()

