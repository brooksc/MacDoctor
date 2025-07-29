"""
Comprehensive logging and debugging system for Mac Doctor.

This module provides structured logging, debug mode with detailed execution traces,
log rotation and cleanup mechanisms, and performance monitoring.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


@dataclass
class LogEntry:
    """Structured log entry for debugging and analysis."""
    
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
            "extra_data": self.extra_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            logger_name=data["logger_name"],
            message=data["message"],
            module=data["module"],
            function=data["function"],
            line_number=data["line_number"],
            thread_id=data["thread_id"],
            process_id=data["process_id"],
            extra_data=data.get("extra_data", {})
        )


@dataclass
class ExecutionTrace:
    """Detailed execution trace for debugging."""
    
    trace_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    function_name: str = ""
    module_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    exception: Optional[str] = None
    duration_ms: Optional[float] = None
    child_traces: List["ExecutionTrace"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, return_value: Any = None, exception: Optional[Exception] = None) -> None:
        """Mark trace as finished."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if exception:
            self.exception = str(exception)
        else:
            self.return_value = return_value
    
    def add_child(self, child_trace: "ExecutionTrace") -> None:
        """Add a child trace."""
        self.child_traces.append(child_trace)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "function_name": self.function_name,
            "module_name": self.module_name,
            "arguments": self.arguments,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "exception": self.exception,
            "duration_ms": self.duration_ms,
            "child_traces": [child.to_dict() for child in self.child_traces],
            "metadata": self.metadata
        }


class StructuredLogHandler(logging.Handler):
    """Custom log handler that captures structured log entries."""
    
    def __init__(self, max_entries: int = 10000):
        """Initialize handler.
        
        Args:
            max_entries: Maximum number of log entries to keep in memory
        """
        super().__init__()
        self.max_entries = max_entries
        self.log_entries: List[LogEntry] = []
        self._lock = Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            # Create structured log entry
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                thread_id=record.thread,
                process_id=record.process,
                extra_data=getattr(record, 'extra_data', {})
            )
            
            with self._lock:
                self.log_entries.append(entry)
                
                # Maintain max entries limit
                if len(self.log_entries) > self.max_entries:
                    self.log_entries = self.log_entries[-self.max_entries:]
        
        except Exception:
            # Don't let logging errors break the application
            pass
    
    def get_entries(
        self,
        level: Optional[str] = None,
        logger_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[LogEntry]:
        """Get log entries with optional filtering.
        
        Args:
            level: Filter by log level
            logger_name: Filter by logger name
            since: Filter entries since this timestamp
            limit: Maximum number of entries to return
            
        Returns:
            List of matching log entries
        """
        with self._lock:
            entries = self.log_entries.copy()
        
        # Apply filters
        if level:
            entries = [e for e in entries if e.level == level]
        
        if logger_name:
            entries = [e for e in entries if logger_name in e.logger_name]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            entries = entries[:limit]
        
        return entries
    
    def clear_entries(self) -> None:
        """Clear all log entries."""
        with self._lock:
            self.log_entries.clear()
    
    def export_entries(self, file_path: Path, format: str = "json") -> None:
        """Export log entries to file.
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        with self._lock:
            entries = self.log_entries.copy()
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump([entry.to_dict() for entry in entries], f, indent=2)
        elif format == "csv":
            import csv
            with open(file_path, 'w', newline='') as f:
                if entries:
                    writer = csv.DictWriter(f, fieldnames=entries[0].to_dict().keys())
                    writer.writeheader()
                    for entry in entries:
                        writer.writerow(entry.to_dict())


class ExecutionTracer:
    """Execution tracer for detailed debugging."""
    
    def __init__(self):
        """Initialize tracer."""
        self.traces: Dict[str, ExecutionTrace] = {}
        self.active_traces: List[str] = []
        self._lock = Lock()
        self._trace_counter = 0
    
    def start_trace(
        self,
        function_name: str,
        module_name: str,
        arguments: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Start a new execution trace.
        
        Args:
            function_name: Name of the function being traced
            module_name: Name of the module
            arguments: Function arguments
            metadata: Additional metadata
            
        Returns:
            Trace ID
        """
        with self._lock:
            self._trace_counter += 1
            trace_id = f"trace_{self._trace_counter}_{int(time.time() * 1000)}"
        
        trace = ExecutionTrace(
            trace_id=trace_id,
            start_time=datetime.now(),
            function_name=function_name,
            module_name=module_name,
            arguments=arguments or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self.traces[trace_id] = trace
            self.active_traces.append(trace_id)
        
        return trace_id
    
    def finish_trace(
        self,
        trace_id: str,
        return_value: Any = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Finish an execution trace.
        
        Args:
            trace_id: Trace ID to finish
            return_value: Function return value
            exception: Exception if one occurred
        """
        with self._lock:
            if trace_id in self.traces:
                trace = self.traces[trace_id]
                trace.finish(return_value, exception)
                
                if trace_id in self.active_traces:
                    self.active_traces.remove(trace_id)
    
    def add_trace_metadata(self, trace_id: str, key: str, value: Any) -> None:
        """Add metadata to an existing trace.
        
        Args:
            trace_id: Trace ID
            key: Metadata key
            value: Metadata value
        """
        with self._lock:
            if trace_id in self.traces:
                self.traces[trace_id].metadata[key] = value
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Get a specific trace.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            ExecutionTrace if found, None otherwise
        """
        with self._lock:
            return self.traces.get(trace_id)
    
    def get_recent_traces(self, limit: int = 100) -> List[ExecutionTrace]:
        """Get recent execution traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent traces
        """
        with self._lock:
            traces = list(self.traces.values())
        
        # Sort by start time (most recent first)
        traces.sort(key=lambda t: t.start_time, reverse=True)
        
        return traces[:limit]
    
    def clear_traces(self) -> None:
        """Clear all traces."""
        with self._lock:
            self.traces.clear()
            self.active_traces.clear()
    
    def export_traces(self, file_path: Path) -> None:
        """Export traces to JSON file.
        
        Args:
            file_path: Path to export file
        """
        with self._lock:
            traces = list(self.traces.values())
        
        with open(file_path, 'w') as f:
            json.dump([trace.to_dict() for trace in traces], f, indent=2)


class LogRotationManager:
    """Manages log file rotation and cleanup."""
    
    def __init__(self, log_directory: Path, max_file_size: int = 10 * 1024 * 1024, max_files: int = 10):
        """Initialize rotation manager.
        
        Args:
            log_directory: Directory containing log files
            max_file_size: Maximum size per log file in bytes
            max_files: Maximum number of log files to keep
        """
        self.log_directory = log_directory
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.log_directory.mkdir(parents=True, exist_ok=True)
    
    def rotate_logs(self) -> None:
        """Rotate log files if needed."""
        log_files = list(self.log_directory.glob("*.log"))
        
        for log_file in log_files:
            if log_file.stat().st_size > self.max_file_size:
                self._rotate_file(log_file)
    
    def _rotate_file(self, log_file: Path) -> None:
        """Rotate a specific log file.
        
        Args:
            log_file: Path to log file to rotate
        """
        base_name = log_file.stem
        extension = log_file.suffix
        
        # Find existing rotated files
        rotated_files = list(self.log_directory.glob(f"{base_name}.*.{extension.lstrip('.')}"))
        rotated_files.sort(key=lambda f: int(f.stem.split('.')[-1]) if f.stem.split('.')[-1].isdigit() else 0)
        
        # Remove oldest files if we exceed max_files
        while len(rotated_files) >= self.max_files:
            oldest_file = rotated_files.pop(0)
            oldest_file.unlink()
        
        # Rotate existing files
        for i in range(len(rotated_files) - 1, -1, -1):
            old_file = rotated_files[i]
            old_number = int(old_file.stem.split('.')[-1])
            new_file = self.log_directory / f"{base_name}.{old_number + 1}{extension}"
            old_file.rename(new_file)
        
        # Rotate current file
        rotated_file = self.log_directory / f"{base_name}.1{extension}"
        log_file.rename(rotated_file)
        
        # Create new empty log file
        log_file.touch()
    
    def cleanup_old_logs(self, max_age_days: int = 30) -> None:
        """Clean up old log files.
        
        Args:
            max_age_days: Maximum age of log files to keep
        """
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in self.log_directory.glob("*.log*"):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_time:
                log_file.unlink()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about log files.
        
        Returns:
            Dictionary with log file statistics
        """
        log_files = list(self.log_directory.glob("*.log*"))
        
        total_size = sum(f.stat().st_size for f in log_files)
        file_count = len(log_files)
        
        oldest_file = min(log_files, key=lambda f: f.stat().st_mtime) if log_files else None
        newest_file = max(log_files, key=lambda f: f.stat().st_mtime) if log_files else None
        
        return {
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file": str(oldest_file) if oldest_file else None,
            "newest_file": str(newest_file) if newest_file else None,
            "oldest_date": datetime.fromtimestamp(oldest_file.stat().st_mtime).isoformat() if oldest_file else None,
            "newest_date": datetime.fromtimestamp(newest_file.stat().st_mtime).isoformat() if newest_file else None
        }


class MacDoctorLogger:
    """Main logging manager for Mac Doctor."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config or {}
        self.console = Console()
        
        # Initialize components
        self.structured_handler = StructuredLogHandler(
            max_entries=self.config.get("max_log_entries", 10000)
        )
        self.execution_tracer = ExecutionTracer()
        
        # Setup log directory
        log_dir = Path(self.config.get("log_directory", Path.home() / ".mac_doctor" / "logs"))
        self.rotation_manager = LogRotationManager(
            log_directory=log_dir,
            max_file_size=self.config.get("max_file_size", 10 * 1024 * 1024),
            max_files=self.config.get("max_files", 10)
        )
        
        # Setup logging
        self._setup_logging()
        
        # Get main logger
        self.logger = logging.getLogger("mac_doctor")
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration."""
        debug_mode = self.config.get("debug_mode", False)
        log_level = logging.DEBUG if debug_mode else self.config.get("log_level", logging.INFO)
        
        # Create root logger
        root_logger = logging.getLogger("mac_doctor")
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with Rich formatting
        if self.config.get("console_output", True):
            console_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=debug_mode,
                rich_tracebacks=True,
                tracebacks_show_locals=debug_mode
            )
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.rotation_manager.log_directory / "mac_doctor.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.rotation_manager.max_file_size,
            backupCount=self.rotation_manager.max_files
        )
        
        if debug_mode:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
            )
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG if debug_mode else log_level)
        root_logger.addHandler(file_handler)
        
        # Add structured handler
        root_logger.addHandler(self.structured_handler)
        
        # Reduce noise from external libraries
        external_loggers = ["httpx", "langchain", "urllib3", "requests", "openai"]
        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information.
        
        Returns:
            Dictionary with debug information
        """
        recent_logs = self.structured_handler.get_entries(limit=50)
        recent_traces = self.execution_tracer.get_recent_traces(limit=20)
        log_stats = self.rotation_manager.get_log_stats()
        
        return {
            "logging": {
                "recent_entries": len(recent_logs),
                "total_entries": len(self.structured_handler.log_entries),
                "log_levels": {
                    level: len([e for e in recent_logs if e.level == level])
                    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                }
            },
            "tracing": {
                "total_traces": len(self.execution_tracer.traces),
                "active_traces": len(self.execution_tracer.active_traces),
                "recent_traces": len(recent_traces)
            },
            "files": log_stats,
            "config": self.config
        }
    
    def show_debug_panel(self) -> None:
        """Show debug information panel."""
        debug_info = self.get_debug_info()
        
        # Create debug panel
        debug_text = Text()
        debug_text.append("🔍 Mac Doctor Debug Information\n\n", style="bold blue")
        
        # Logging info
        debug_text.append("📝 Logging:\n", style="bold yellow")
        debug_text.append(f"  Recent entries: {debug_info['logging']['recent_entries']}\n")
        debug_text.append(f"  Total entries: {debug_info['logging']['total_entries']}\n")
        
        for level, count in debug_info['logging']['log_levels'].items():
            if count > 0:
                debug_text.append(f"  {level}: {count}\n")
        
        # Tracing info
        debug_text.append("\n🔬 Execution Tracing:\n", style="bold yellow")
        debug_text.append(f"  Total traces: {debug_info['tracing']['total_traces']}\n")
        debug_text.append(f"  Active traces: {debug_info['tracing']['active_traces']}\n")
        debug_text.append(f"  Recent traces: {debug_info['tracing']['recent_traces']}\n")
        
        # File info
        debug_text.append("\n📁 Log Files:\n", style="bold yellow")
        debug_text.append(f"  Total files: {debug_info['files']['total_files']}\n")
        debug_text.append(f"  Total size: {debug_info['files']['total_size_mb']} MB\n")
        
        panel = Panel(
            debug_text,
            title="[bold]Debug Information[/bold]",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def export_debug_data(self, export_path: Path) -> None:
        """Export debug data to files.
        
        Args:
            export_path: Directory to export debug data
        """
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export log entries
        self.structured_handler.export_entries(
            export_path / "log_entries.json",
            format="json"
        )
        
        # Export execution traces
        self.execution_tracer.export_traces(
            export_path / "execution_traces.json"
        )
        
        # Export debug info
        with open(export_path / "debug_info.json", 'w') as f:
            json.dump(self.get_debug_info(), f, indent=2)
        
        self.logger.info(f"Debug data exported to {export_path}")
    
    def cleanup_logs(self, max_age_days: int = 30) -> None:
        """Clean up old log files.
        
        Args:
            max_age_days: Maximum age of log files to keep
        """
        self.rotation_manager.cleanup_old_logs(max_age_days)
        self.logger.info(f"Cleaned up log files older than {max_age_days} days")


# Global logger instance
_global_logger: Optional[MacDoctorLogger] = None


def get_logger() -> MacDoctorLogger:
    """Get the global Mac Doctor logger instance.
    
    Returns:
        MacDoctorLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = MacDoctorLogger()
    return _global_logger


def setup_logging(config: Optional[Dict[str, Any]] = None) -> MacDoctorLogger:
    """Setup Mac Doctor logging with configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        MacDoctorLogger instance
    """
    global _global_logger
    _global_logger = MacDoctorLogger(config)
    return _global_logger


def trace_execution(func: Optional[Callable] = None, *, include_args: bool = True, include_return: bool = True):
    """Decorator to trace function execution.
    
    Args:
        func: Function to trace (when used as @trace_execution)
        include_args: Whether to include function arguments in trace
        include_return: Whether to include return value in trace
        
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # Prepare arguments for tracing
            trace_args = {}
            if include_args:
                try:
                    # Convert args to dict with parameter names
                    import inspect
                    sig = inspect.signature(f)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    trace_args = dict(bound_args.arguments)
                    
                    # Sanitize large objects
                    for key, value in trace_args.items():
                        if hasattr(value, '__len__') and len(str(value)) > 1000:
                            trace_args[key] = f"<{type(value).__name__} with {len(value)} items>"
                        elif len(str(value)) > 500:
                            trace_args[key] = f"{str(value)[:500]}..."
                except Exception:
                    trace_args = {"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
            
            # Start trace
            trace_id = logger.execution_tracer.start_trace(
                function_name=f.__name__,
                module_name=f.__module__,
                arguments=trace_args
            )
            
            try:
                result = f(*args, **kwargs)
                
                # Finish trace with result
                trace_result = result if include_return else "<return value hidden>"
                logger.execution_tracer.finish_trace(trace_id, return_value=trace_result)
                
                return result
            
            except Exception as e:
                # Finish trace with exception
                logger.execution_tracer.finish_trace(trace_id, exception=e)
                raise
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def debug_context(context_name: str, metadata: Dict[str, Any] = None):
    """Context manager for debugging code blocks.
    
    Args:
        context_name: Name of the debug context
        metadata: Additional metadata
    """
    logger = get_logger()
    
    # Start trace
    trace_id = logger.execution_tracer.start_trace(
        function_name=context_name,
        module_name="debug_context",
        metadata=metadata or {}
    )
    
    start_time = time.time()
    
    try:
        yield trace_id
    except Exception as e:
        logger.execution_tracer.finish_trace(trace_id, exception=e)
        raise
    else:
        duration = time.time() - start_time
        logger.execution_tracer.finish_trace(trace_id, return_value=f"completed in {duration:.3f}s")


def log_performance(operation: str, duration: float, metadata: Dict[str, Any] = None) -> None:
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        metadata: Additional metadata
    """
    logger = get_logger()
    
    extra_data = {
        "operation": operation,
        "duration_seconds": duration,
        "duration_ms": duration * 1000,
        **(metadata or {})
    }
    
    # Log with appropriate level based on duration
    if duration > 10:
        level = logging.WARNING
        message = f"Slow operation: {operation} took {duration:.3f}s"
    elif duration > 5:
        level = logging.INFO
        message = f"Operation: {operation} took {duration:.3f}s"
    else:
        level = logging.DEBUG
        message = f"Operation: {operation} took {duration:.3f}s"
    
    logger.logger.log(level, message, extra={"extra_data": extra_data})