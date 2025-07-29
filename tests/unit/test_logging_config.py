"""
Tests for Mac Doctor logging and debugging functionality.
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mac_doctor.logging_config import (
    LogEntry,
    ExecutionTrace,
    StructuredLogHandler,
    ExecutionTracer,
    LogRotationManager,
    MacDoctorLogger,
    get_logger,
    setup_logging,
    trace_execution,
    debug_context,
    log_performance
)


class TestLogEntry:
    """Test LogEntry functionality."""
    
    def test_log_entry_creation(self):
        """Test creating a log entry."""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            logger_name="test_logger",
            message="Test message",
            module="test_module",
            function="test_function",
            line_number=42,
            thread_id=12345,
            process_id=67890,
            extra_data={"key": "value"}
        )
        
        assert entry.timestamp == timestamp
        assert entry.level == "INFO"
        assert entry.logger_name == "test_logger"
        assert entry.message == "Test message"
        assert entry.extra_data == {"key": "value"}
    
    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary."""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level="ERROR",
            logger_name="test",
            message="Error message",
            module="test_module",
            function="test_func",
            line_number=10,
            thread_id=1,
            process_id=2
        )
        
        data = entry.to_dict()
        
        assert data["timestamp"] == timestamp.isoformat()
        assert data["level"] == "ERROR"
        assert data["message"] == "Error message"
        assert data["line_number"] == 10
    
    def test_log_entry_from_dict(self):
        """Test creating log entry from dictionary."""
        timestamp = datetime.now()
        data = {
            "timestamp": timestamp.isoformat(),
            "level": "WARNING",
            "logger_name": "test",
            "message": "Warning message",
            "module": "test_module",
            "function": "test_func",
            "line_number": 20,
            "thread_id": 1,
            "process_id": 2,
            "extra_data": {"test": True}
        }
        
        entry = LogEntry.from_dict(data)
        
        assert entry.timestamp == timestamp
        assert entry.level == "WARNING"
        assert entry.message == "Warning message"
        assert entry.extra_data == {"test": True}


class TestExecutionTrace:
    """Test ExecutionTrace functionality."""
    
    def test_execution_trace_creation(self):
        """Test creating an execution trace."""
        start_time = datetime.now()
        trace = ExecutionTrace(
            trace_id="test_trace_1",
            start_time=start_time,
            function_name="test_function",
            module_name="test_module",
            arguments={"arg1": "value1"},
            metadata={"test": True}
        )
        
        assert trace.trace_id == "test_trace_1"
        assert trace.start_time == start_time
        assert trace.function_name == "test_function"
        assert trace.arguments == {"arg1": "value1"}
        assert trace.end_time is None
        assert trace.duration_ms is None
    
    def test_execution_trace_finish(self):
        """Test finishing an execution trace."""
        trace = ExecutionTrace(
            trace_id="test_trace_2",
            start_time=datetime.now(),
            function_name="test_function",
            module_name="test_module"
        )
        
        # Simulate some execution time
        time.sleep(0.01)
        
        trace.finish(return_value="test_result")
        
        assert trace.end_time is not None
        assert trace.duration_ms is not None
        assert trace.duration_ms > 0
        assert trace.return_value == "test_result"
        assert trace.exception is None
    
    def test_execution_trace_finish_with_exception(self):
        """Test finishing an execution trace with exception."""
        trace = ExecutionTrace(
            trace_id="test_trace_3",
            start_time=datetime.now(),
            function_name="test_function",
            module_name="test_module"
        )
        
        exception = ValueError("Test error")
        trace.finish(exception=exception)
        
        assert trace.end_time is not None
        assert trace.exception == "Test error"
        assert trace.return_value is None
    
    def test_execution_trace_add_child(self):
        """Test adding child traces."""
        parent_trace = ExecutionTrace(
            trace_id="parent_trace",
            start_time=datetime.now(),
            function_name="parent_function",
            module_name="test_module"
        )
        
        child_trace = ExecutionTrace(
            trace_id="child_trace",
            start_time=datetime.now(),
            function_name="child_function",
            module_name="test_module"
        )
        
        parent_trace.add_child(child_trace)
        
        assert len(parent_trace.child_traces) == 1
        assert parent_trace.child_traces[0] == child_trace
    
    def test_execution_trace_to_dict(self):
        """Test converting execution trace to dictionary."""
        start_time = datetime.now()
        trace = ExecutionTrace(
            trace_id="test_trace_4",
            start_time=start_time,
            function_name="test_function",
            module_name="test_module",
            arguments={"arg1": "value1"}
        )
        trace.finish(return_value="result")
        
        data = trace.to_dict()
        
        assert data["trace_id"] == "test_trace_4"
        assert data["start_time"] == start_time.isoformat()
        assert data["function_name"] == "test_function"
        assert data["arguments"] == {"arg1": "value1"}
        assert data["return_value"] == "result"
        assert data["duration_ms"] is not None


class TestStructuredLogHandler:
    """Test StructuredLogHandler functionality."""
    
    def test_handler_creation(self):
        """Test creating a structured log handler."""
        handler = StructuredLogHandler(max_entries=100)
        
        assert handler.max_entries == 100
        assert len(handler.log_entries) == 0
    
    def test_handler_emit(self):
        """Test emitting log records."""
        handler = StructuredLogHandler(max_entries=10)
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.process = 67890
        record.created = time.time()
        
        handler.emit(record)
        
        assert len(handler.log_entries) == 1
        entry = handler.log_entries[0]
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.module == "test_module"
        assert entry.function == "test_function"
    
    def test_handler_max_entries_limit(self):
        """Test that handler respects max entries limit."""
        handler = StructuredLogHandler(max_entries=3)
        
        # Emit more records than the limit
        for i in range(5):
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="/test/path.py",
                lineno=i,
                msg=f"Message {i}",
                args=(),
                exc_info=None
            )
            record.module = "test_module"
            record.funcName = "test_function"
            record.thread = 1
            record.process = 1
            record.created = time.time()
            
            handler.emit(record)
        
        # Should only keep the last 3 entries
        assert len(handler.log_entries) == 3
        assert handler.log_entries[0].message == "Message 2"
        assert handler.log_entries[-1].message == "Message 4"
    
    def test_get_entries_filtering(self):
        """Test filtering log entries."""
        handler = StructuredLogHandler()
        
        # Add test entries
        for i, level in enumerate(["INFO", "WARNING", "ERROR", "INFO"]):
            record = logging.LogRecord(
                name=f"logger_{i}",
                level=getattr(logging, level),
                pathname="/test/path.py",
                lineno=i,
                msg=f"Message {i}",
                args=(),
                exc_info=None
            )
            record.module = "test_module"
            record.funcName = "test_function"
            record.thread = 1
            record.process = 1
            record.created = time.time() + i
            
            handler.emit(record)
        
        # Test level filtering
        error_entries = handler.get_entries(level="ERROR")
        assert len(error_entries) == 1
        assert error_entries[0].message == "Message 2"
        
        # Test logger name filtering
        logger_0_entries = handler.get_entries(logger_name="logger_0")
        assert len(logger_0_entries) == 1
        assert logger_0_entries[0].message == "Message 0"
        
        # Test limit
        limited_entries = handler.get_entries(limit=2)
        assert len(limited_entries) == 2
    
    def test_clear_entries(self):
        """Test clearing log entries."""
        handler = StructuredLogHandler()
        
        # Add a test entry
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 1
        record.process = 1
        record.created = time.time()
        
        handler.emit(record)
        assert len(handler.log_entries) == 1
        
        handler.clear_entries()
        assert len(handler.log_entries) == 0
    
    def test_export_entries_json(self):
        """Test exporting entries to JSON."""
        handler = StructuredLogHandler()
        
        # Add test entry
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 1
        record.process = 1
        record.created = time.time()
        
        handler.emit(record)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            handler.export_entries(export_path, format="json")
            
            # Verify export
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["message"] == "Test message"
            assert data[0]["level"] == "INFO"
        
        finally:
            export_path.unlink()


class TestExecutionTracer:
    """Test ExecutionTracer functionality."""
    
    def test_tracer_creation(self):
        """Test creating an execution tracer."""
        tracer = ExecutionTracer()
        
        assert len(tracer.traces) == 0
        assert len(tracer.active_traces) == 0
    
    def test_start_trace(self):
        """Test starting a trace."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace(
            function_name="test_function",
            module_name="test_module",
            arguments={"arg1": "value1"},
            metadata={"test": True}
        )
        
        assert trace_id is not None
        assert trace_id in tracer.traces
        assert trace_id in tracer.active_traces
        
        trace = tracer.traces[trace_id]
        assert trace.function_name == "test_function"
        assert trace.module_name == "test_module"
        assert trace.arguments == {"arg1": "value1"}
        assert trace.metadata == {"test": True}
    
    def test_finish_trace(self):
        """Test finishing a trace."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("test_function", "test_module")
        assert trace_id in tracer.active_traces
        
        tracer.finish_trace(trace_id, return_value="result")
        
        assert trace_id not in tracer.active_traces
        assert trace_id in tracer.traces
        
        trace = tracer.traces[trace_id]
        assert trace.return_value == "result"
        assert trace.end_time is not None
    
    def test_add_trace_metadata(self):
        """Test adding metadata to a trace."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("test_function", "test_module")
        tracer.add_trace_metadata(trace_id, "key", "value")
        
        trace = tracer.traces[trace_id]
        assert trace.metadata["key"] == "value"
    
    def test_get_trace(self):
        """Test getting a specific trace."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("test_function", "test_module")
        
        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert trace.trace_id == trace_id
        
        # Test non-existent trace
        non_existent = tracer.get_trace("non_existent")
        assert non_existent is None
    
    def test_get_recent_traces(self):
        """Test getting recent traces."""
        tracer = ExecutionTracer()
        
        # Create multiple traces
        trace_ids = []
        for i in range(5):
            trace_id = tracer.start_trace(f"function_{i}", "test_module")
            tracer.finish_trace(trace_id, return_value=f"result_{i}")
            trace_ids.append(trace_id)
            time.sleep(0.001)  # Ensure different timestamps
        
        recent_traces = tracer.get_recent_traces(limit=3)
        
        assert len(recent_traces) == 3
        # Should be in reverse chronological order (most recent first)
        assert recent_traces[0].function_name == "function_4"
        assert recent_traces[1].function_name == "function_3"
        assert recent_traces[2].function_name == "function_2"
    
    def test_clear_traces(self):
        """Test clearing all traces."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("test_function", "test_module")
        assert len(tracer.traces) == 1
        assert len(tracer.active_traces) == 1
        
        tracer.clear_traces()
        
        assert len(tracer.traces) == 0
        assert len(tracer.active_traces) == 0
    
    def test_export_traces(self):
        """Test exporting traces to JSON."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("test_function", "test_module")
        tracer.finish_trace(trace_id, return_value="result")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            tracer.export_traces(export_path)
            
            # Verify export
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["function_name"] == "test_function"
            assert data[0]["return_value"] == "result"
        
        finally:
            export_path.unlink()


class TestLogRotationManager:
    """Test LogRotationManager functionality."""
    
    def test_rotation_manager_creation(self):
        """Test creating a log rotation manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogRotationManager(log_dir, max_file_size=1024, max_files=5)
            
            assert manager.log_directory == log_dir
            assert manager.max_file_size == 1024
            assert manager.max_files == 5
            assert log_dir.exists()
    
    def test_rotate_logs(self):
        """Test log rotation when file exceeds size limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogRotationManager(log_dir, max_file_size=100, max_files=3)
            
            # Create a log file that exceeds the size limit
            log_file = log_dir / "test.log"
            with open(log_file, 'w') as f:
                f.write("x" * 200)  # Exceeds 100 byte limit
            
            manager.rotate_logs()
            
            # Original file should be rotated
            assert (log_dir / "test.1.log").exists()
            assert log_file.exists()  # New empty file created
            assert log_file.stat().st_size == 0
    
    def test_cleanup_old_logs(self):
        """Test cleaning up old log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogRotationManager(log_dir)
            
            # Create old and new log files
            old_file = log_dir / "old.log"
            new_file = log_dir / "new.log"
            
            old_file.touch()
            new_file.touch()
            
            # Make old file appear old
            old_time = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
            os.utime(old_file, (old_time, old_time))
            
            manager.cleanup_old_logs(max_age_days=30)
            
            assert not old_file.exists()
            assert new_file.exists()
    
    def test_get_log_stats(self):
        """Test getting log file statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            manager = LogRotationManager(log_dir)
            
            # Create test log files
            (log_dir / "test1.log").write_text("content1")
            (log_dir / "test2.log").write_text("content2")
            
            stats = manager.get_log_stats()
            
            assert stats["total_files"] == 2
            assert stats["total_size_bytes"] > 0
            assert stats["total_size_mb"] >= 0  # Can be 0.0 for very small files
            assert stats["oldest_file"] is not None
            assert stats["newest_file"] is not None


class TestMacDoctorLogger:
    """Test MacDoctorLogger functionality."""
    
    def test_logger_creation(self):
        """Test creating a Mac Doctor logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "log_directory": temp_dir,
                "debug_mode": True,
                "console_output": False
            }
            
            logger = MacDoctorLogger(config)
            
            assert logger.config == config
            assert logger.structured_handler is not None
            assert logger.execution_tracer is not None
            assert logger.rotation_manager is not None
    
    def test_get_debug_info(self):
        """Test getting debug information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"log_directory": temp_dir}
            logger = MacDoctorLogger(config)
            
            # Generate some log entries
            logger.logger.info("Test message")
            
            debug_info = logger.get_debug_info()
            
            assert "logging" in debug_info
            assert "tracing" in debug_info
            assert "files" in debug_info
            assert "config" in debug_info
            
            assert debug_info["logging"]["recent_entries"] >= 0
            assert debug_info["tracing"]["total_traces"] >= 0
    
    def test_export_debug_data(self):
        """Test exporting debug data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            export_dir = Path(temp_dir) / "export"
            
            config = {"log_directory": str(log_dir)}
            logger = MacDoctorLogger(config)
            
            # Generate some data
            logger.logger.info("Test message")
            trace_id = logger.execution_tracer.start_trace("test_func", "test_module")
            logger.execution_tracer.finish_trace(trace_id, return_value="result")
            
            logger.export_debug_data(export_dir)
            
            assert (export_dir / "log_entries.json").exists()
            assert (export_dir / "execution_traces.json").exists()
            assert (export_dir / "debug_info.json").exists()
    
    def test_cleanup_logs(self):
        """Test log cleanup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"log_directory": temp_dir}
            logger = MacDoctorLogger(config)
            
            # This should not raise an exception
            logger.cleanup_logs(max_age_days=30)


class TestGlobalFunctions:
    """Test global logging functions."""
    
    def test_get_logger(self):
        """Test getting global logger instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return the same instance
        assert logger1 is logger2
    
    def test_setup_logging(self):
        """Test setting up logging with configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"log_directory": temp_dir, "debug_mode": True}
            
            logger = setup_logging(config)
            
            assert logger.config == config
            assert logger.config["debug_mode"] is True
    
    def test_trace_execution_decorator(self):
        """Test the trace_execution decorator."""
        @trace_execution
        def test_function(arg1, arg2="default"):
            return f"{arg1}_{arg2}"
        
        result = test_function("test", arg2="value")
        
        assert result == "test_value"
        
        # Check that trace was created
        logger = get_logger()
        traces = logger.execution_tracer.get_recent_traces(limit=1)
        assert len(traces) >= 1
        assert traces[0].function_name == "test_function"
    
    def test_trace_execution_with_exception(self):
        """Test trace_execution decorator with exception."""
        @trace_execution
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Check that trace was created with exception
        logger = get_logger()
        traces = logger.execution_tracer.get_recent_traces(limit=1)
        assert len(traces) >= 1
        assert traces[0].function_name == "failing_function"
        assert traces[0].exception == "Test error"
    
    def test_debug_context(self):
        """Test debug context manager."""
        with debug_context("test_context", {"key": "value"}) as trace_id:
            assert trace_id is not None
            time.sleep(0.01)  # Simulate some work
        
        # Check that trace was created
        logger = get_logger()
        trace = logger.execution_tracer.get_trace(trace_id)
        assert trace is not None
        assert trace.function_name == "test_context"
        assert trace.metadata == {"key": "value"}
        assert trace.end_time is not None
    
    def test_debug_context_with_exception(self):
        """Test debug context manager with exception."""
        with pytest.raises(ValueError):
            with debug_context("failing_context") as trace_id:
                raise ValueError("Test error")
        
        # Check that trace was created with exception
        logger = get_logger()
        trace = logger.execution_tracer.get_trace(trace_id)
        assert trace is not None
        assert trace.exception == "Test error"
    
    def test_log_performance(self):
        """Test performance logging."""
        log_performance("test_operation", 2.5, {"extra": "data"})
        
        # Check that performance was logged
        logger = get_logger()
        entries = logger.structured_handler.get_entries(limit=1)
        assert len(entries) >= 1
        
        # Find the performance log entry
        perf_entry = None
        for entry in entries:
            if "test_operation" in entry.message:
                perf_entry = entry
                break
        
        assert perf_entry is not None
        assert "2.500s" in perf_entry.message
        assert perf_entry.extra_data.get("operation") == "test_operation"
        assert perf_entry.extra_data.get("duration_seconds") == 2.5


class TestIntegration:
    """Integration tests for logging system."""
    
    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "log_directory": temp_dir,
                "debug_mode": True,
                "max_log_entries": 100
            }
            
            # Setup logging
            logger = setup_logging(config)
            
            # Generate various types of logs
            logger.logger.info("Starting test workflow")
            logger.logger.warning("This is a warning")
            logger.logger.error("This is an error")
            
            # Create execution traces
            @trace_execution
            def traced_function(value):
                with debug_context("inner_context"):
                    time.sleep(0.01)
                    return value * 2
            
            result = traced_function(5)
            assert result == 10
            
            # Log performance
            log_performance("test_workflow", 0.1)
            
            # Get debug info
            debug_info = logger.get_debug_info()
            assert debug_info["logging"]["recent_entries"] > 0
            assert debug_info["tracing"]["total_traces"] > 0
            
            # Export debug data
            export_dir = Path(temp_dir) / "export"
            logger.export_debug_data(export_dir)
            
            # Verify exports
            assert (export_dir / "log_entries.json").exists()
            assert (export_dir / "execution_traces.json").exists()
            assert (export_dir / "debug_info.json").exists()
            
            # Verify log file was created
            log_files = list(Path(temp_dir).glob("*.log"))
            assert len(log_files) > 0