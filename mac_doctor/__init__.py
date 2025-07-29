"""
Mac Doctor - An agentic AI assistant for macOS system diagnostics.

A Python-based diagnostic tool that analyzes macOS system performance and health,
diagnoses issues, and provides actionable recommendations using local diagnostic
tools and configurable LLM providers.
"""

__version__ = "0.1.0"
__author__ = "Mac Doctor Team"

from .interfaces import BaseMCP, BaseLLM
from .tool_registry import ToolRegistry
from .logging_config import get_logger, setup_logging, trace_execution, debug_context, log_performance

__all__ = [
    "BaseMCP",
    "BaseLLM", 
    "ToolRegistry",
    "get_logger",
    "setup_logging",
    "trace_execution",
    "debug_context",
    "log_performance",
]