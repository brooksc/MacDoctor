"""
Unit tests for DTrace MCP.

Tests the DTrace Mac Collector Plugin with mocked system commands
and various DTrace output scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import subprocess

from mac_doctor.mcps.dtrace_mcp import DTraceMCP
from mac_doctor.interfaces import MCPResult


class TestDTraceMCP:
    """Test cases for DTrace MCP functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dtrace_mcp = DTraceMCP()
    
    def test_name_property(self):
        """Test that name property returns correct value."""
        assert self.dtrace_mcp.name == "dtrace"
    
    def test_description_property(self):
        """Test that description property returns meaningful text."""
        description = self.dtrace_mcp.description
        assert isinstance(description, str)
        assert len(description) > 0
        assert "dtrace" in description.lower() or "trace" in description.lower()
    
    def test_get_schema(self):
        """Test that get_schema returns valid schema."""
        schema = self.dtrace_mcp.get_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        properties = schema["properties"]
        assert "script_type" in properties
        assert "duration" in properties
        assert "custom_script" in properties
        
        # Check script_type enum values
        script_types = properties["script_type"]["enum"]
        expected_types = ["syscall", "io", "proc", "network", "memory"]
        for expected_type in expected_types:
            assert expected_type in script_types
    
    @patch('subprocess.run')
    def test_is_available_dtrace_exists_and_works(self, mock_run):
        """Test is_available when dtrace exists and works."""
        # Mock successful 'which dtrace' command
        mock_run.side_effect = [
            Mock(returncode=0),  # which dtrace
            Mock(returncode=0)   # dtrace test command
        ]
        
        assert self.dtrace_mcp.is_available() is True
        
        # Verify the calls
        assert mock_run.call_count == 2
        mock_run.assert_any_call(
            ["which", "dtrace"],
            capture_output=True,
            text=True,
            timeout=5
        )
        mock_run.assert_any_call(
            ["dtrace", "-n", "BEGIN { exit(0); }"],
            capture_output=True,
            text=True,
            timeout=10
        )
    
    @patch('subprocess.run')
    def test_is_available_dtrace_not_found(self, mock_run):
        """Test is_available when dtrace command not found."""
        # Mock failed 'which dtrace' command
        mock_run.return_value = Mock(returncode=1)
        
        assert self.dtrace_mcp.is_available() is False
        
        # Should only call 'which dtrace'
        assert mock_run.call_count == 1
    
    @patch('subprocess.run')
    def test_is_available_dtrace_no_privileges(self, mock_run):
        """Test is_available when dtrace exists but no privileges."""
        # Mock successful 'which dtrace' but failed test command
        mock_run.side_effect = [
            Mock(returncode=0),  # which dtrace
            Mock(returncode=1)   # dtrace test command fails
        ]
        
        assert self.dtrace_mcp.is_available() is False
    
    @patch('subprocess.run')
    def test_is_available_timeout_exception(self, mock_run):
        """Test is_available when subprocess times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("dtrace", 5)
        
        assert self.dtrace_mcp.is_available() is False
    
    @patch('subprocess.run')
    def test_is_available_file_not_found(self, mock_run):
        """Test is_available when dtrace command not found."""
        mock_run.side_effect = FileNotFoundError()
        
        assert self.dtrace_mcp.is_available() is False
    
    @patch.object(DTraceMCP, 'is_available')
    def test_execute_dtrace_not_available(self, mock_is_available):
        """Test execute when DTrace is not available."""
        mock_is_available.return_value = False
        
        result = self.dtrace_mcp.execute()
        
        assert isinstance(result, MCPResult)
        assert result.tool_name == "dtrace"
        assert not result.success
        assert "not available" in result.error.lower()
        assert "privileges" in result.error.lower()
    
    def test_execute_invalid_duration(self):
        """Test execute with invalid duration parameter."""
        with patch.object(self.dtrace_mcp, 'is_available', return_value=True):
            # Test duration too short
            result = self.dtrace_mcp.execute(duration=3)
            assert not result.success
            assert "between 5 and 60" in result.error
            
            # Test duration too long
            result = self.dtrace_mcp.execute(duration=120)
            assert not result.success
            assert "between 5 and 60" in result.error
    
    def test_execute_invalid_script_type(self):
        """Test execute with invalid script_type parameter."""
        with patch.object(self.dtrace_mcp, 'is_available', return_value=True):
            result = self.dtrace_mcp.execute(script_type="invalid_type")
            assert not result.success
            assert "Unknown script_type" in result.error
    
    @patch.object(DTraceMCP, '_execute_dtrace_script')
    @patch.object(DTraceMCP, 'is_available')
    def test_execute_syscall_success(self, mock_is_available, mock_execute_script):
        """Test successful syscall script execution."""
        mock_is_available.return_value = True
        
        # Mock DTrace output for syscall script
        mock_dtrace_output = """
        dtrace: script '/dev/stdin' matched 1000 probes
        
        kernel_task    mach_msg_trap    150
        loginwindow    read             45
        Finder         write            32
        Safari         open             28
        """
        
        mock_execute_script.return_value = {
            "success": True,
            "output": mock_dtrace_output,
            "error": None
        }
        
        result = self.dtrace_mcp.execute(script_type="syscall", duration=10)
        
        assert isinstance(result, MCPResult)
        assert result.tool_name == "dtrace"
        assert result.success is True
        assert "summary" in result.data
        assert "top_syscalls" in result.data
        assert "top_processes" in result.data
        
        # Check metadata
        assert result.metadata["script_type"] == "syscall"
        assert result.metadata["duration"] == 10
    
    @patch.object(DTraceMCP, '_execute_dtrace_script')
    @patch.object(DTraceMCP, 'is_available')
    def test_execute_custom_script(self, mock_is_available, mock_execute_script):
        """Test execution with custom DTrace script."""
        mock_is_available.return_value = True
        
        custom_script = "BEGIN { printf(\"Hello DTrace\"); exit(0); }"
        mock_execute_script.return_value = {
            "success": True,
            "output": "Hello DTrace",
            "error": None
        }
        
        result = self.dtrace_mcp.execute(custom_script=custom_script)
        
        assert result.success is True
        assert result.metadata["script_type"] == "custom"
        
        # Verify the custom script was passed to _execute_dtrace_script
        mock_execute_script.assert_called_once()
        args, kwargs = mock_execute_script.call_args
        assert args[0] == custom_script
    
    @patch.object(DTraceMCP, '_execute_dtrace_script')
    @patch.object(DTraceMCP, 'is_available')
    def test_execute_script_execution_failure(self, mock_is_available, mock_execute_script):
        """Test handling of DTrace script execution failure."""
        mock_is_available.return_value = True
        mock_execute_script.return_value = {
            "success": False,
            "output": "",
            "error": "DTrace requires additional privileges"
        }
        
        result = self.dtrace_mcp.execute()
        
        assert not result.success
        assert "DTrace requires additional privileges" in result.error
    
    @patch('subprocess.run')
    def test_execute_dtrace_script_success(self, mock_run):
        """Test _execute_dtrace_script with successful execution."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "dtrace output"
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        script = "BEGIN { exit(0); }"
        result = self.dtrace_mcp._execute_dtrace_script(script, 10)
        
        assert result["success"] is True
        assert result["output"] == "dtrace output"
        assert result["error"] is None
        
        mock_run.assert_called_once_with(
            ["dtrace", "-n", script],
            capture_output=True,
            text=True,
            timeout=15  # 10 + 5 buffer
        )
    
    @patch('subprocess.run')
    def test_execute_dtrace_script_compilation_error(self, mock_run):
        """Test _execute_dtrace_script with compilation error."""
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "dtrace: failed to compile script"
        mock_run.return_value = mock_process
        
        script = "INVALID SCRIPT"
        result = self.dtrace_mcp._execute_dtrace_script(script, 10)
        
        assert result["success"] is False
        assert "script compilation failed" in result["error"]
    
    @patch('subprocess.run')
    def test_execute_dtrace_script_privilege_error(self, mock_run):
        """Test _execute_dtrace_script with privilege error."""
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "dtrace: failed to initialize dtrace: DTrace requires additional privileges"
        mock_run.return_value = mock_process
        
        script = "BEGIN { exit(0); }"
        result = self.dtrace_mcp._execute_dtrace_script(script, 10)
        
        assert result["success"] is False
        assert "elevated privileges" in result["error"]
        assert "System Preferences" in result["error"]
    
    @patch('subprocess.run')
    def test_execute_dtrace_script_timeout(self, mock_run):
        """Test _execute_dtrace_script with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("dtrace", 15)
        
        script = "BEGIN { exit(0); }"
        result = self.dtrace_mcp._execute_dtrace_script(script, 10)
        
        assert result["success"] is False
        assert "timed out" in result["error"]
    
    def test_parse_syscall_output(self):
        """Test parsing of syscall DTrace output."""
        lines = [
            "dtrace: script matched 1000 probes",
            "",
            "kernel_task    mach_msg_trap    150",
            "loginwindow    read             45",
            "Finder         write            32",
            "Safari         open             28",
            "kernel_task    vm_fault         25"
        ]
        
        result = self.dtrace_mcp._parse_syscall_output(lines)
        
        assert "summary" in result
        assert result["summary"]["total_syscalls"] == 280  # Sum of all counts
        assert result["summary"]["unique_syscalls"] == 5
        assert result["summary"]["active_processes"] == 4
        
        assert "top_syscalls" in result
        assert len(result["top_syscalls"]) > 0
        
        assert "top_processes" in result
        assert len(result["top_processes"]) > 0
        
        # Check that kernel_task has the highest total (150 + 25 = 175)
        top_process = result["top_processes"][0]
        assert top_process["process"] == "kernel_task"
        assert top_process["data"]["total_syscalls"] == 175
    
    def test_parse_io_output(self):
        """Test parsing of I/O DTrace output."""
        lines = [
            "dtrace: script matched 500 probes",
            "",
            "Finder         25",
            "Safari         18",
            "disk0s1        45",
            "/dev/disk1     12"
        ]
        
        result = self.dtrace_mcp._parse_io_output(lines)
        
        assert "summary" in result
        assert result["summary"]["total_io_operations"] == 100
        assert result["summary"]["active_processes"] == 2
        assert result["summary"]["active_devices"] == 2
        
        assert "io_by_process" in result
        assert "io_by_device" in result
    
    def test_parse_proc_output(self):
        """Test parsing of process DTrace output."""
        lines = [
            "dtrace: script matched 200 probes",
            "2024-01-15 10:30:15: launchd executed /usr/bin/login",
            "2024-01-15 10:30:16: Finder executed /Applications/Safari.app/Contents/MacOS/Safari",
            "",
            "launchd        5",
            "Finder         3"
        ]
        
        result = self.dtrace_mcp._parse_proc_output(lines)
        
        assert "summary" in result
        assert result["summary"]["events_captured"] == 2
        
        assert "recent_events" in result
        assert len(result["recent_events"]) == 2
    
    def test_parse_network_output(self):
        """Test parsing of network DTrace output."""
        lines = [
            "dtrace: script matched 300 probes",
            "",
            "tcp_input      Safari         25",
            "tcp_output     Safari         18",
            "udp_input      mDNSResponder  12",
            "udp_output     mDNSResponder  8"
        ]
        
        result = self.dtrace_mcp._parse_network_output(lines)
        
        assert "summary" in result
        assert result["summary"]["total_network_events"] == 63
        assert result["summary"]["tcp_events"] == 43
        assert result["summary"]["udp_events"] == 20
        
        assert "top_activity" in result
        assert len(result["top_activity"]) == 4
    
    def test_parse_memory_output(self):
        """Test parsing of memory/VM DTrace output."""
        lines = [
            "dtrace: script matched 100 probes",
            "",
            "pgin           1500",
            "pgout          800",
            "pgpgin         2000",
            "pgpgout        1200"
        ]
        
        result = self.dtrace_mcp._parse_memory_output(lines)
        
        assert "summary" in result
        assert result["summary"]["total_vm_events"] == 5500
        assert result["summary"]["stat_types"] == 4
        
        assert "vm_statistics" in result
        assert result["vm_statistics"]["pgin"] == 1500
        assert result["vm_statistics"]["pgout"] == 800
    
    def test_parse_dtrace_output_custom_script(self):
        """Test parsing of custom script output."""
        output = "Custom output line 1\nCustom output line 2\n"
        
        result = self.dtrace_mcp._parse_dtrace_output(output, "custom")
        
        assert result["script_type"] == "custom"
        assert result["raw_output"] == output
        assert "summary" in result
        assert result["summary"]["total_lines"] == 2
        assert "details" in result
        assert len(result["details"]) == 2
    
    @patch.object(DTraceMCP, '_execute_dtrace_script')
    @patch.object(DTraceMCP, 'is_available')
    def test_execute_all_script_types(self, mock_is_available, mock_execute_script):
        """Test execution of all predefined script types."""
        mock_is_available.return_value = True
        mock_execute_script.return_value = {
            "success": True,
            "output": "test output",
            "error": None
        }
        
        script_types = ["syscall", "io", "proc", "network", "memory"]
        
        for script_type in script_types:
            result = self.dtrace_mcp.execute(script_type=script_type)
            assert result.success is True
            assert result.metadata["script_type"] == script_type
    
    @patch.object(DTraceMCP, 'is_available')
    def test_execute_exception_handling(self, mock_is_available):
        """Test that exceptions during execution are properly handled."""
        mock_is_available.return_value = True
        
        # Mock _execute_dtrace_script to raise an exception
        with patch.object(self.dtrace_mcp, '_execute_dtrace_script', side_effect=Exception("Test error")):
            result = self.dtrace_mcp.execute()
            
            assert not result.success
            assert "Error executing DTrace" in result.error
            assert "Test error" in result.error
    
    def test_dtrace_scripts_exist(self):
        """Test that all expected DTrace scripts are defined."""
        expected_scripts = ["syscall", "io", "proc", "network", "memory"]
        
        for script_type in expected_scripts:
            assert script_type in self.dtrace_mcp.DTRACE_SCRIPTS
            script = self.dtrace_mcp.DTRACE_SCRIPTS[script_type]
            assert isinstance(script, str)
            assert len(script.strip()) > 0
            # Each script should have a tick timer for exit
            assert "tick-" in script
            assert "exit(0)" in script