"""
Core Components - Core functionality and utilities for Mac Doctor.

This package contains core components like the action engine, report generator,
and other shared utilities.
"""

from .action_engine import ActionEngine
from .report_generator import ReportGenerator

__all__ = ['ActionEngine', 'ReportGenerator']