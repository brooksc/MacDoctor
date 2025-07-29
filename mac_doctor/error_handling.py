"""
Comprehensive error handling and logging system for Mac Doctor.

This module provides centralized error handling, logging configuration,
fallback mechanisms, and user-friendly error messages with actionable guidance.
"""

import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .interfaces import MCPResult, ActionResult, DiagnosticResult


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    SYSTEM = "system"
    NETWORK = "network"
    PERMISSION = "permission"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    LLM = "llm"
    MCP = "mcp"
    CLI = "cli"


@dataclass
class ErrorInfo:
    """Structured error information."""
    
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    suggestions: List[str] = None
    technical_details: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MacDoctorError(Exception):
    """Base exception class for Mac Doctor errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggestions: List[str] = None,
        technical_details: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=message,
            suggestions=suggestions or [],
            technical_details=technical_details,
            error_code=error_code
        )


class SystemCompatibilityError(MacDoctorError):
    """Error for system compatibility issues."""
    
    def __init__(self, message: str, suggestions: List[str] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions or [
                "Check macOS version compatibility (requires macOS 12+)",
                "Verify system architecture (Intel/Apple Silicon)",
                "Install missing system tools"
            ]
        )


class DependencyError(MacDoctorError):
    """Error for missing or incompatible dependencies."""
    
    def __init__(self, dependency: str, message: str, suggestions: List[str] = None):
        super().__init__(
            message=f"Dependency '{dependency}': {message}",
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions or [
                f"Install {dependency} using: pip install {dependency}",
                "Check requirements.txt for version compatibility",
                "Verify Python version (requires 3.9+)"
            ]
        )


class MacDoctorPermissionError(MacDoctorError):
    """Error for permission-related issues."""
    
    def __init__(self, operation: str, suggestions: List[str] = None):
        super().__init__(
            message=f"Permission denied for operation: {operation}",
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.MEDIUM,
            suggestions=suggestions or [
                "Run with appropriate permissions (sudo if required)",
                "Check file/directory permissions",
                "Verify user has access to system resources"
            ]
        )


class LLMError(MacDoctorError):
    """Error for LLM provider issues."""
    
    def __init__(self, provider: str, message: str, suggestions: List[str] = None):
        super().__init__(
            message=f"LLM provider '{provider}': {message}",
            category=ErrorCategory.LLM,
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions or [
                "Check LLM provider configuration",
                "Verify API keys and credentials",
                "Try fallback provider if available",
                "Check network connectivity for remote providers"
            ]
        )


class MCPError(MacDoctorError):
    """Error for MCP tool issues."""
    
    def __init__(self, tool_name: str, message: str, suggestions: List[str] = None):
        super().__init__(
            message=f"MCP tool '{tool_name}': {message}",
            category=ErrorCategory.MCP,
            severity=ErrorSeverity.MEDIUM,
            suggestions=suggestions or [
                "Check if required system tools are installed",
                "Verify tool permissions and availability",
                "Try alternative diagnostic tools"
            ]
        )


class ConfigurationError(MacDoctorError):
    """Error for configuration issues."""
    
    def __init__(self, config_item: str, message: str, suggestions: List[str] = None):
        super().__init__(
            message=f"Configuration '{config_item}': {message}",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.MEDIUM,
            suggestions=suggestions or [
                "Check configuration file syntax",
                "Verify configuration values",
                "Reset to default configuration if needed"
            ]
        )


class TimeoutError(MacDoctorError):
    """Error for timeout issues."""
    
    def __init__(self, operation: str, timeout_seconds: int, suggestions: List[str] = None):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            suggestions=suggestions or [
                "Increase timeout value in configuration",
                "Check system performance and load",
                "Try operation with fewer resources"
            ]
        )


class ErrorHandler:
    """Centralized error handler with logging and user-friendly messages."""
    
    def __init__(self, console: Optional[Console] = None, logger: Optional[logging.Logger] = None):
        """Initialize error handler.
        
        Args:
            console: Rich console for user output
            logger: Logger instance for error logging
        """
        self.console = console or Console()
        self.logger = logger or logging.getLogger(__name__)
        self._error_history: List[ErrorInfo] = []
        self._fallback_handlers: Dict[ErrorCategory, callable] = {}
        
        # Register default fallback handlers
        self._register_default_fallbacks()
    
    def _register_default_fallbacks(self) -> None:
        """Register default fallback handlers for different error categories."""
        self._fallback_handlers[ErrorCategory.DEPENDENCY] = self._handle_dependency_fallback
        self._fallback_handlers[ErrorCategory.LLM] = self._handle_llm_fallback
        self._fallback_handlers[ErrorCategory.MCP] = self._handle_mcp_fallback
        self._fallback_handlers[ErrorCategory.PERMISSION] = self._handle_permission_fallback
        self._fallback_handlers[ErrorCategory.TIMEOUT] = self._handle_timeout_fallback
    
    def handle_error(
        self,
        error: Union[Exception, ErrorInfo],
        context: Optional[Dict[str, Any]] = None,
        show_user_message: bool = True,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle an error with logging, user messaging, and optional recovery.
        
        Args:
            error: Exception or ErrorInfo to handle
            context: Additional context information
            show_user_message: Whether to show user-friendly error message
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Convert exception to ErrorInfo if needed
        if isinstance(error, Exception):
            error_info = self._exception_to_error_info(error)
        else:
            error_info = error
        
        # Add context details
        if context:
            error_info.details = f"{error_info.details or ''}\nContext: {context}"
        
        # Log the error
        self._log_error(error_info, context)
        
        # Add to error history
        self._error_history.append(error_info)
        
        # Show user-friendly message
        if show_user_message:
            self._show_user_error_message(error_info)
        
        # Attempt recovery if requested
        if attempt_recovery:
            return self._attempt_recovery(error_info, context)
        
        return None
    
    def _exception_to_error_info(self, exception: Exception) -> ErrorInfo:
        """Convert a generic exception to structured ErrorInfo."""
        # Handle known Mac Doctor exceptions
        if isinstance(exception, MacDoctorError):
            return exception.error_info
        
        # Handle common Python exceptions
        error_message = str(exception)
        technical_details = traceback.format_exc()
        
        if isinstance(exception, FileNotFoundError):
            return ErrorInfo(
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                message=f"File not found: {error_message}",
                technical_details=technical_details,
                suggestions=[
                    "Check if the file path is correct",
                    "Verify file permissions",
                    "Ensure the file exists"
                ]
            )
        
        elif isinstance(exception, PermissionError):
            return ErrorInfo(
                category=ErrorCategory.PERMISSION,
                severity=ErrorSeverity.MEDIUM,
                message=f"Permission denied: {error_message}",
                technical_details=technical_details,
                suggestions=[
                    "Run with appropriate permissions",
                    "Check file/directory permissions",
                    "Use sudo if required"
                ]
            )
        
        elif isinstance(exception, ImportError):
            return ErrorInfo(
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.HIGH,
                message=f"Import error: {error_message}",
                technical_details=technical_details,
                suggestions=[
                    "Install missing dependencies",
                    "Check Python path configuration",
                    "Verify package installation"
                ]
            )
        
        elif isinstance(exception, TimeoutError):
            return ErrorInfo(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                message=f"Operation timed out: {error_message}",
                technical_details=technical_details,
                suggestions=[
                    "Increase timeout value",
                    "Check system performance",
                    "Retry the operation"
                ]
            )
        
        elif isinstance(exception, ValueError):
            return ErrorInfo(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                message=f"Invalid value: {error_message}",
                technical_details=technical_details,
                suggestions=[
                    "Check input parameters",
                    "Verify data format",
                    "Review configuration values"
                ]
            )
        
        else:
            # Generic exception handling
            return ErrorInfo(
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                message=f"Unexpected error: {error_message}",
                technical_details=technical_details,
                suggestions=[
                    "Check system logs for more details",
                    "Retry the operation",
                    "Report this issue if it persists"
                ]
            )
    
    def _log_error(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error information with appropriate level."""
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        # Choose log level based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log technical details at debug level
        if error_info.technical_details:
            self.logger.debug(f"Technical details: {error_info.technical_details}")
    
    def _show_user_error_message(self, error_info: ErrorInfo) -> None:
        """Show user-friendly error message with suggestions."""
        # Choose color based on severity
        severity_colors = {
            ErrorSeverity.LOW: "blue",
            ErrorSeverity.MEDIUM: "yellow",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bright_red"
        }
        
        color = severity_colors.get(error_info.severity, "yellow")
        
        # Create error message panel
        error_text = Text()
        error_text.append("❌ Error: ", style="bold red")
        error_text.append(error_info.message, style=color)
        
        if error_info.details:
            error_text.append(f"\n\nDetails: {error_info.details}", style="dim")
        
        # Add suggestions if available
        if error_info.suggestions:
            error_text.append("\n\n💡 Suggestions:", style="bold blue")
            for i, suggestion in enumerate(error_info.suggestions, 1):
                error_text.append(f"\n  {i}. {suggestion}", style="blue")
        
        # Show the panel
        panel = Panel(
            error_text,
            title=f"[bold]{error_info.category.value.title()} Error[/bold]",
            border_style=color,
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def _attempt_recovery(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Attempt automatic recovery based on error category."""
        fallback_handler = self._fallback_handlers.get(error_info.category)
        
        if fallback_handler:
            try:
                self.logger.info(f"Attempting recovery for {error_info.category.value} error")
                return fallback_handler(error_info, context)
            except Exception as e:
                self.logger.warning(f"Recovery attempt failed: {e}")
        
        return None
    
    def _handle_dependency_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle dependency-related errors with fallbacks."""
        # Try to suggest alternative implementations or graceful degradation
        if "psutil" in error_info.message:
            self.console.print("[yellow]⚠️  psutil not available, some process monitoring features will be limited[/yellow]")
            return {"fallback": "basic_process_info", "available": False}
        
        return None
    
    def _handle_llm_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle LLM provider errors with fallbacks."""
        # Try to switch to fallback provider or rule-based analysis
        self.console.print("[yellow]⚠️  LLM provider unavailable, switching to rule-based analysis[/yellow]")
        return {"fallback": "rule_based_analysis", "available": True}
    
    def _handle_mcp_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle MCP tool errors with fallbacks."""
        # Try alternative tools or limited functionality
        tool_name = context.get("tool_name") if context else "unknown"
        self.console.print(f"[yellow]⚠️  Tool '{tool_name}' unavailable, using alternative methods[/yellow]")
        return {"fallback": "alternative_tool", "available": False}
    
    def _handle_permission_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle permission errors with fallbacks."""
        # Suggest running with appropriate permissions or alternative approaches
        self.console.print("[yellow]⚠️  Permission denied, some features may be limited[/yellow]")
        return {"fallback": "limited_access", "available": True}
    
    def _handle_timeout_fallback(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle timeout errors with fallbacks."""
        # Try with reduced scope or shorter timeout
        self.console.print("[yellow]⚠️  Operation timed out, trying with reduced scope[/yellow]")
        return {"fallback": "reduced_scope", "available": True}
    
    def register_fallback_handler(self, category: ErrorCategory, handler: callable) -> None:
        """Register a custom fallback handler for an error category.
        
        Args:
            category: Error category to handle
            handler: Callable that takes (error_info, context) and returns recovery result
        """
        self._fallback_handlers[category] = handler
        self.logger.info(f"Registered fallback handler for {category.value} errors")
    
    def get_error_history(self) -> List[ErrorInfo]:
        """Get the history of handled errors."""
        return self._error_history.copy()
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self._error_history.clear()
        self.logger.info("Error history cleared")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of error patterns and frequencies."""
        if not self._error_history:
            return {"total_errors": 0, "categories": {}, "severities": {}}
        
        categories = {}
        severities = {}
        
        for error in self._error_history:
            # Count by category
            cat_name = error.category.value
            categories[cat_name] = categories.get(cat_name, 0) + 1
            
            # Count by severity
            sev_name = error.severity.value
            severities[sev_name] = severities.get(sev_name, 0) + 1
        
        return {
            "total_errors": len(self._error_history),
            "categories": categories,
            "severities": severities,
            "most_recent": self._error_history[-1].timestamp.isoformat() if self._error_history else None
        }


class LoggingManager:
    """Manages logging configuration and setup for Mac Doctor."""
    
    @staticmethod
    def setup_logging(
        level: Union[int, str] = logging.INFO,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        debug_mode: bool = False
    ) -> logging.Logger:
        """Set up comprehensive logging for Mac Doctor.
        
        Args:
            level: Logging level
            log_file: Optional log file path
            console_output: Whether to output to console
            debug_mode: Enable debug mode with verbose logging
            
        Returns:
            Configured root logger
        """
        # Create root logger
        root_logger = logging.getLogger("mac_doctor")
        root_logger.setLevel(logging.DEBUG if debug_mode else level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        if debug_mode:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG if debug_mode else level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Reduce noise from external libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        return root_logger


def safe_execute(
    func: callable,
    error_handler: Optional[ErrorHandler] = None,
    context: Optional[Dict[str, Any]] = None,
    fallback_result: Any = None,
    show_errors: bool = True
) -> Any:
    """Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        error_handler: Error handler instance
        context: Additional context for error handling
        fallback_result: Result to return if function fails
        show_errors: Whether to show error messages to user
        
    Returns:
        Function result or fallback_result if function fails
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    try:
        return func()
    except Exception as e:
        recovery_result = error_handler.handle_error(
            e,
            context=context,
            show_user_message=show_errors,
            attempt_recovery=True
        )
        
        # Return recovery result if available, otherwise fallback
        return recovery_result if recovery_result is not None else fallback_result


def create_safe_mcp_result(
    tool_name: str,
    error: Exception,
    execution_time: float = 0.0,
    error_handler: Optional[ErrorHandler] = None
) -> MCPResult:
    """Create a safe MCPResult from an error.
    
    Args:
        tool_name: Name of the MCP tool
        error: Exception that occurred
        execution_time: Time taken before error
        error_handler: Optional error handler for logging
        
    Returns:
        MCPResult with error information
    """
    if error_handler:
        error_handler.handle_error(error, context={"tool_name": tool_name}, show_user_message=False)
    
    return MCPResult(
        tool_name=tool_name,
        success=False,
        data={},
        error=str(error),
        execution_time=execution_time,
        metadata={"error_type": type(error).__name__}
    )


def create_safe_diagnostic_result(
    query: str,
    error: Exception,
    execution_time: float = 0.0,
    error_handler: Optional[ErrorHandler] = None
) -> DiagnosticResult:
    """Create a safe DiagnosticResult from an error.
    
    Args:
        query: Original query
        error: Exception that occurred
        execution_time: Time taken before error
        error_handler: Optional error handler for logging
        
    Returns:
        DiagnosticResult with error information
    """
    if error_handler:
        error_handler.handle_error(error, context={"query": query}, show_user_message=False)
    
    from datetime import datetime
    
    return DiagnosticResult(
        query=query,
        analysis=f"Analysis failed due to error: {str(error)}",
        issues_detected=[],
        tool_results={},
        recommendations=[],
        execution_time=execution_time,
        timestamp=datetime.now()
    )