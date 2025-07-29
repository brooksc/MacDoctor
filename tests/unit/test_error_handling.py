"""
Unit tests for comprehensive error handling system.

This module tests the error handling, logging, and fallback mechanisms
implemented in the Mac Doctor error handling system.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from rich.console import Console

from mac_doctor.error_handling import (
    ErrorHandler, ErrorInfo, ErrorSeverity, ErrorCategory,
    MacDoctorError, SystemCompatibilityError, DependencyError,
    MacDoctorPermissionError, LLMError, MCPError, ConfigurationError,
    TimeoutError, LoggingManager, safe_execute,
    create_safe_mcp_result, create_safe_diagnostic_result
)
from mac_doctor.interfaces import MCPResult, DiagnosticResult


class TestErrorInfo:
    """Test ErrorInfo data class."""
    
    def test_error_info_creation(self):
        """Test ErrorInfo creation with all fields."""
        error_info = ErrorInfo(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            message="Test error message",
            details="Additional details",
            suggestions=["Suggestion 1", "Suggestion 2"],
            technical_details="Technical details",
            error_code="ERR001"
        )
        
        assert error_info.category == ErrorCategory.SYSTEM
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.message == "Test error message"
        assert error_info.details == "Additional details"
        assert error_info.suggestions == ["Suggestion 1", "Suggestion 2"]
        assert error_info.technical_details == "Technical details"
        assert error_info.error_code == "ERR001"
        assert isinstance(error_info.timestamp, datetime)
    
    def test_error_info_defaults(self):
        """Test ErrorInfo creation with default values."""
        error_info = ErrorInfo(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="Network error"
        )
        
        assert error_info.suggestions == []
        assert error_info.details is None
        assert error_info.technical_details is None
        assert error_info.error_code is None
        assert isinstance(error_info.timestamp, datetime)


class TestMacDoctorExceptions:
    """Test custom Mac Doctor exception classes."""
    
    def test_mac_doctor_error(self):
        """Test base MacDoctorError exception."""
        error = MacDoctorError(
            message="Base error",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.LOW,
            suggestions=["Fix config"],
            technical_details="Config issue",
            error_code="CFG001"
        )
        
        assert str(error) == "Base error"
        assert error.error_info.category == ErrorCategory.CONFIGURATION
        assert error.error_info.severity == ErrorSeverity.LOW
        assert error.error_info.suggestions == ["Fix config"]
        assert error.error_info.technical_details == "Config issue"
        assert error.error_info.error_code == "CFG001"
    
    def test_system_compatibility_error(self):
        """Test SystemCompatibilityError exception."""
        error = SystemCompatibilityError(
            message="Incompatible system",
            suggestions=["Upgrade macOS"]
        )
        
        assert str(error) == "Incompatible system"
        assert error.error_info.category == ErrorCategory.SYSTEM
        assert error.error_info.severity == ErrorSeverity.HIGH
        assert "Upgrade macOS" in error.error_info.suggestions
    
    def test_dependency_error(self):
        """Test DependencyError exception."""
        error = DependencyError(
            dependency="psutil",
            message="not installed",
            suggestions=["pip install psutil"]
        )
        
        assert "psutil" in str(error)
        assert "not installed" in str(error)
        assert error.error_info.category == ErrorCategory.DEPENDENCY
        assert error.error_info.severity == ErrorSeverity.HIGH
        assert "pip install psutil" in error.error_info.suggestions
    
    def test_llm_error(self):
        """Test LLMError exception."""
        error = LLMError(
            provider="gemini",
            message="API key invalid",
            suggestions=["Check API key"]
        )
        
        assert "gemini" in str(error)
        assert "API key invalid" in str(error)
        assert error.error_info.category == ErrorCategory.LLM
        assert error.error_info.severity == ErrorSeverity.HIGH
        assert "Check API key" in error.error_info.suggestions
    
    def test_mcp_error(self):
        """Test MCPError exception."""
        error = MCPError(
            tool_name="process",
            message="tool unavailable",
            suggestions=["Install dependencies"]
        )
        
        assert "process" in str(error)
        assert "tool unavailable" in str(error)
        assert error.error_info.category == ErrorCategory.MCP
        assert error.error_info.severity == ErrorSeverity.MEDIUM
        assert "Install dependencies" in error.error_info.suggestions
    
    def test_permission_error(self):
        """Test MacDoctorPermissionError exception."""
        error = MacDoctorPermissionError(
            operation="file_access",
            suggestions=["Use sudo"]
        )
        
        assert "file_access" in str(error)
        assert error.error_info.category == ErrorCategory.PERMISSION
        assert error.error_info.severity == ErrorSeverity.MEDIUM
        assert "Use sudo" in error.error_info.suggestions
    
    def test_timeout_error(self):
        """Test TimeoutError exception."""
        error = TimeoutError(
            operation="system_scan",
            timeout_seconds=30,
            suggestions=["Increase timeout"]
        )
        
        assert "system_scan" in str(error)
        assert "30 seconds" in str(error)
        assert error.error_info.category == ErrorCategory.TIMEOUT
        assert error.error_info.severity == ErrorSeverity.MEDIUM
        assert "Increase timeout" in error.error_info.suggestions


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_console = Mock(spec=Console)
        self.mock_logger = Mock(spec=logging.Logger)
        self.error_handler = ErrorHandler(
            console=self.mock_console,
            logger=self.mock_logger
        )
    
    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        assert self.error_handler.console == self.mock_console
        assert self.error_handler.logger == self.mock_logger
        assert self.error_handler._error_history == []
        assert len(self.error_handler._fallback_handlers) > 0
    
    def test_handle_mac_doctor_error(self):
        """Test handling MacDoctorError."""
        error = LLMError(
            provider="test",
            message="test error",
            suggestions=["test suggestion"]
        )
        
        result = self.error_handler.handle_error(
            error,
            context={"test": "context"},
            show_user_message=True,
            attempt_recovery=False
        )
        
        # Check error was logged
        self.mock_logger.error.assert_called_once()
        
        # Check error was added to history
        assert len(self.error_handler._error_history) == 1
        assert self.error_handler._error_history[0].category == ErrorCategory.LLM
        
        # Check user message was shown
        self.mock_console.print.assert_called()
    
    def test_handle_generic_exception(self):
        """Test handling generic Python exceptions."""
        error = ValueError("Invalid input")
        
        result = self.error_handler.handle_error(
            error,
            show_user_message=False,
            attempt_recovery=False
        )
        
        # Check error was converted to ErrorInfo
        assert len(self.error_handler._error_history) == 1
        error_info = self.error_handler._error_history[0]
        assert error_info.category == ErrorCategory.VALIDATION
        assert "Invalid input" in error_info.message
    
    def test_fallback_handling(self):
        """Test fallback mechanism."""
        error = DependencyError(
            dependency="psutil",
            message="not available"
        )
        
        result = self.error_handler.handle_error(
            error,
            attempt_recovery=True
        )
        
        # Check fallback was attempted
        assert result is not None
        assert result.get("fallback") == "basic_process_info"
    
    def test_register_custom_fallback(self):
        """Test registering custom fallback handler."""
        def custom_handler(error_info, context):
            return {"custom": "fallback"}
        
        self.error_handler.register_fallback_handler(
            ErrorCategory.NETWORK,
            custom_handler
        )
        
        error = MacDoctorError(
            message="Network error",
            category=ErrorCategory.NETWORK
        )
        
        result = self.error_handler.handle_error(
            error,
            attempt_recovery=True
        )
        
        assert result == {"custom": "fallback"}
    
    def test_error_history_management(self):
        """Test error history management."""
        # Add some errors
        for i in range(3):
            error = MacDoctorError(f"Error {i}")
            self.error_handler.handle_error(error, show_user_message=False)
        
        # Check history
        history = self.error_handler.get_error_history()
        assert len(history) == 3
        assert all(isinstance(e, ErrorInfo) for e in history)
        
        # Test summary
        summary = self.error_handler.get_error_summary()
        assert summary["total_errors"] == 3
        assert "system" in summary["categories"]
        assert "medium" in summary["severities"]
        
        # Clear history
        self.error_handler.clear_error_history()
        assert len(self.error_handler.get_error_history()) == 0


class TestLoggingManager:
    """Test LoggingManager class."""
    
    def test_setup_logging_console_only(self):
        """Test logging setup with console output only."""
        logger = LoggingManager.setup_logging(
            level=logging.INFO,
            console_output=True,
            debug_mode=False
        )
        
        assert logger.name == "mac_doctor"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            logger = LoggingManager.setup_logging(
                level=logging.DEBUG,
                log_file=log_file,
                console_output=False,
                debug_mode=True
            )
            
            assert logger.level == logging.DEBUG
            assert log_file.exists()
    
    def test_setup_logging_debug_mode(self):
        """Test logging setup in debug mode."""
        logger = LoggingManager.setup_logging(
            level=logging.INFO,
            debug_mode=True
        )
        
        assert logger.level == logging.DEBUG


class TestSafeExecute:
    """Test safe_execute function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_error_handler = Mock(spec=ErrorHandler)
        self.mock_error_handler.handle_error.return_value = "recovery_result"
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        def success_func():
            return "success"
        
        result = safe_execute(
            success_func,
            error_handler=self.mock_error_handler,
            fallback_result="fallback"
        )
        
        assert result == "success"
        self.mock_error_handler.handle_error.assert_not_called()
    
    def test_safe_execute_with_exception(self):
        """Test safe_execute with function that raises exception."""
        def failing_func():
            raise ValueError("Test error")
        
        result = safe_execute(
            failing_func,
            error_handler=self.mock_error_handler,
            context={"test": "context"},
            fallback_result="fallback"
        )
        
        # Should return recovery result from error handler
        assert result == "recovery_result"
        self.mock_error_handler.handle_error.assert_called_once()
    
    def test_safe_execute_no_recovery(self):
        """Test safe_execute when recovery fails."""
        def failing_func():
            raise ValueError("Test error")
        
        self.mock_error_handler.handle_error.return_value = None
        
        result = safe_execute(
            failing_func,
            error_handler=self.mock_error_handler,
            fallback_result="fallback"
        )
        
        # Should return fallback result
        assert result == "fallback"


class TestSafeResultCreation:
    """Test safe result creation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_error_handler = Mock(spec=ErrorHandler)
    
    def test_create_safe_mcp_result(self):
        """Test create_safe_mcp_result function."""
        error = ValueError("Test error")
        
        result = create_safe_mcp_result(
            tool_name="test_tool",
            error=error,
            execution_time=1.5,
            error_handler=self.mock_error_handler
        )
        
        assert isinstance(result, MCPResult)
        assert result.tool_name == "test_tool"
        assert result.success is False
        assert result.error == "Test error"
        assert result.execution_time == 1.5
        assert result.metadata["error_type"] == "ValueError"
        
        # Check error handler was called
        self.mock_error_handler.handle_error.assert_called_once()
    
    def test_create_safe_diagnostic_result(self):
        """Test create_safe_diagnostic_result function."""
        error = RuntimeError("Analysis failed")
        
        result = create_safe_diagnostic_result(
            query="test query",
            error=error,
            execution_time=2.0,
            error_handler=self.mock_error_handler
        )
        
        assert isinstance(result, DiagnosticResult)
        assert result.query == "test query"
        assert "Analysis failed" in result.analysis
        assert result.execution_time == 2.0
        assert len(result.issues_detected) == 0
        assert len(result.recommendations) == 0
        assert len(result.tool_results) == 0
        
        # Check error handler was called
        self.mock_error_handler.handle_error.assert_called_once()


class TestErrorHandlerIntegration:
    """Integration tests for error handling system."""
    
    def test_end_to_end_error_handling(self):
        """Test complete error handling workflow."""
        console = Mock(spec=Console)
        logger = Mock(spec=logging.Logger)
        error_handler = ErrorHandler(console=console, logger=logger)
        
        # Simulate a complex operation that fails
        def complex_operation():
            # First operation succeeds
            step1_result = safe_execute(
                lambda: "step1_success",
                error_handler=error_handler
            )
            
            # Second operation fails
            def failing_step():
                raise LLMError(
                    provider="test",
                    message="LLM unavailable",
                    suggestions=["Check connection"]
                )
            
            step2_result = safe_execute(
                failing_step,
                error_handler=error_handler,
                fallback_result="fallback_analysis"
            )
            
            return step1_result, step2_result
        
        result1, result2 = complex_operation()
        
        # Check results
        assert result1 == "step1_success"
        # Should use recovery result from error handler (LLM fallback)
        assert result2 == {"fallback": "rule_based_analysis", "available": True}
        
        # Check error history
        history = error_handler.get_error_history()
        assert len(history) == 1
        assert history[0].category == ErrorCategory.LLM
        
        # Check logging was called
        logger.error.assert_called()
        
        # Check user message was shown
        console.print.assert_called()
    
    def test_multiple_error_categories(self):
        """Test handling multiple different error categories."""
        error_handler = ErrorHandler()
        
        errors = [
            SystemCompatibilityError("System incompatible"),
            DependencyError("psutil", "not found"),
            LLMError("gemini", "API error"),
            MCPError("process", "unavailable"),
            TimeoutError("scan", 30)
        ]
        
        for error in errors:
            error_handler.handle_error(error, show_user_message=False)
        
        # Check all errors were recorded
        history = error_handler.get_error_history()
        assert len(history) == 5
        
        # Check different categories are represented
        categories = {e.category for e in history}
        expected_categories = {
            ErrorCategory.SYSTEM,
            ErrorCategory.DEPENDENCY,
            ErrorCategory.LLM,
            ErrorCategory.MCP,
            ErrorCategory.TIMEOUT
        }
        assert categories == expected_categories
        
        # Check summary
        summary = error_handler.get_error_summary()
        assert summary["total_errors"] == 5
        assert len(summary["categories"]) == 5