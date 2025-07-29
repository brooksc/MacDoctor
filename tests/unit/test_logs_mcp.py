"""
Unit tests for LogsMCP - System log monitoring Mac Collector Plugin.

These tests verify the LogsMCP functionality with mocked system log data
to ensure consistent behavior across different environments.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import subprocess

from mac_doctor.mcps.logs_mcp import LogsMCP
from mac_doctor.interfaces import MCPResult


class TestLogsMCP(unittest.TestCase):
    """Test cases for LogsMCP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logs_mcp = LogsMCP()
        
        # Sample log entries for testing
        self.sample_log_lines = [
            "2024-01-15 10:30:45.123456-0800 MacBook-Pro kernel[0]: <Error> Memory pressure detected",
            "2024-01-15 10:30:46.234567-0800 MacBook-Pro loginwindow[123]: <Fault> Authentication failed for user",
            "2024-01-15 10:30:47.345678-0800 MacBook-Pro Safari[456]: <Default> Page loaded successfully",
            "2024-01-15 10:30:48.456789-0800 MacBook-Pro com.apple.xpc.launchd[1]: <Info> Service started",
            "2024-01-15 10:30:49.567890-0800 MacBook-Pro Finder[789]: <Error> File not found: /missing/file.txt"
        ]
        
        self.sample_log_output = "\n".join(self.sample_log_lines)
    
    def test_name_property(self):
        """Test that the name property returns correct value."""
        self.assertEqual(self.logs_mcp.name, "logs")
    
    def test_description_property(self):
        """Test that the description property returns correct value."""
        expected = "Monitors system logs using 'log show' command with filtering by time range and severity levels"
        self.assertEqual(self.logs_mcp.description, expected)
    
    def test_get_schema(self):
        """Test that get_schema returns proper schema definition."""
        schema = self.logs_mcp.get_schema()
        
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        
        properties = schema["properties"]
        expected_props = ["hours", "severity_levels", "max_entries", "subsystems", "processes"]
        for prop in expected_props:
            self.assertIn(prop, properties)
        
        # Check hours schema
        hours_schema = properties["hours"]
        self.assertEqual(hours_schema["type"], "integer")
        self.assertEqual(hours_schema["default"], 1)
        self.assertEqual(hours_schema["minimum"], 1)
        self.assertEqual(hours_schema["maximum"], 24)
        
        # Check severity_levels schema
        severity_schema = properties["severity_levels"]
        self.assertEqual(severity_schema["type"], "array")
        self.assertEqual(severity_schema["default"], ["error", "fault"])
        
        # Check max_entries schema
        max_entries_schema = properties["max_entries"]
        self.assertEqual(max_entries_schema["type"], "integer")
        self.assertEqual(max_entries_schema["default"], 1000)
        self.assertEqual(max_entries_schema["minimum"], 100)
        self.assertEqual(max_entries_schema["maximum"], 5000)
    
    @patch('subprocess.run')
    def test_is_available_when_log_command_present(self, mock_run):
        """Test is_available returns True when log command is available."""
        mock_run.return_value.returncode = 0
        self.assertTrue(self.logs_mcp.is_available())
        mock_run.assert_called_once_with(['which', 'log'], 
                                       capture_output=True, 
                                       text=True, 
                                       timeout=5)
    
    @patch('subprocess.run')
    def test_is_available_when_log_command_missing(self, mock_run):
        """Test is_available returns False when log command is not available."""
        mock_run.return_value.returncode = 1
        self.assertFalse(self.logs_mcp.is_available())
    
    @patch('subprocess.run')
    def test_is_available_with_timeout(self, mock_run):
        """Test is_available handles timeout gracefully."""
        mock_run.side_effect = subprocess.TimeoutExpired(['which', 'log'], 5)
        self.assertFalse(self.logs_mcp.is_available())
    
    @patch('subprocess.run')
    def test_is_available_with_file_not_found(self, mock_run):
        """Test is_available handles FileNotFoundError gracefully."""
        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(self.logs_mcp.is_available())
    
    def test_build_log_command_basic(self):
        """Test building basic log command."""
        cmd = self.logs_mcp._build_log_command(
            hours=2, 
            severity_levels=['error', 'fault'], 
            max_entries=500,
            subsystems=[],
            processes=[]
        )
        
        self.assertIn('log', cmd)
        self.assertIn('show', cmd)
        self.assertIn('--start', cmd)
        self.assertIn('--style', cmd)
        self.assertIn('syslog', cmd)
        self.assertIn('--level', cmd)
        self.assertIn('error,fault', cmd)
        self.assertIn('--max', cmd)
        self.assertIn('500', cmd)
    
    def test_build_log_command_with_filters(self):
        """Test building log command with subsystem and process filters."""
        cmd = self.logs_mcp._build_log_command(
            hours=1,
            severity_levels=['error'],
            max_entries=100,
            subsystems=['com.apple.kernel'],
            processes=['Safari', 'Finder']
        )
        
        self.assertIn('--subsystem', cmd)
        self.assertIn('com.apple.kernel', cmd)
        self.assertIn('--process', cmd)
        self.assertIn('Safari', cmd)
        self.assertIn('Finder', cmd)
    
    def test_build_log_command_invalid_severity_levels(self):
        """Test building log command with invalid severity levels."""
        cmd = self.logs_mcp._build_log_command(
            hours=1,
            severity_levels=['invalid', 'error', 'also_invalid'],
            max_entries=100,
            subsystems=[],
            processes=[]
        )
        
        # Should only include valid levels
        level_index = cmd.index('--level')
        level_value = cmd[level_index + 1]
        self.assertEqual(level_value, 'error')
    
    def test_parse_log_entry_with_microseconds(self):
        """Test parsing log entry with microseconds timestamp."""
        line = "2024-01-15 10:30:45.123456-0800 MacBook-Pro kernel[0]: <Error> Memory pressure detected"
        entry = self.logs_mcp._parse_log_entry(line)
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry['process'], 'kernel')
        self.assertEqual(entry['pid'], 0)
        self.assertEqual(entry['level'], 'error')
        self.assertEqual(entry['message'], 'Memory pressure detected')
        self.assertEqual(entry['hostname'], 'MacBook-Pro')
        self.assertIn('2024-01-15 10:30:45', entry['timestamp'])
    
    def test_parse_log_entry_without_microseconds(self):
        """Test parsing log entry without microseconds timestamp."""
        line = "2024-01-15 10:30:45-0800 MacBook-Pro Safari[456]: <Default> Page loaded"
        entry = self.logs_mcp._parse_log_entry(line)
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry['process'], 'Safari')
        self.assertEqual(entry['pid'], 456)
        self.assertEqual(entry['level'], 'default')
        self.assertEqual(entry['message'], 'Page loaded')
    
    def test_parse_log_entry_without_level(self):
        """Test parsing log entry without explicit level."""
        line = "2024-01-15 10:30:45-0800 MacBook-Pro Finder[789]: File operation completed"
        entry = self.logs_mcp._parse_log_entry(line)
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry['process'], 'Finder')
        self.assertEqual(entry['pid'], 789)
        self.assertEqual(entry['level'], 'default')
        self.assertEqual(entry['message'], 'File operation completed')
    
    def test_parse_log_entry_malformed(self):
        """Test parsing malformed log entry."""
        line = "This is not a valid log entry"
        entry = self.logs_mcp._parse_log_entry(line)
        
        self.assertIsNone(entry)
    
    def test_parse_log_entry_with_spaces_in_process_name(self):
        """Test parsing log entry with spaces in process name."""
        line = "2024-01-15 10:30:45-0800 MacBook-Pro My App Name[123]: <Info> Application started"
        entry = self.logs_mcp._parse_log_entry(line)
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry['process'], 'My App Name')
        self.assertEqual(entry['pid'], 123)
        self.assertEqual(entry['level'], 'info')
    
    def test_extract_error_patterns_basic(self):
        """Test basic error pattern extraction."""
        # Create sample log entries
        entries = [
            {
                'level': 'error',
                'process': 'kernel',
                'message': 'Memory allocation failed',
                'timestamp': '2024-01-15 10:30:45'
            },
            {
                'level': 'fault',
                'process': 'Safari',
                'message': 'Network timeout occurred',
                'timestamp': '2024-01-15 10:30:46'
            },
            {
                'level': 'info',
                'process': 'Finder',
                'message': 'File copied successfully',
                'timestamp': '2024-01-15 10:30:47'
            }
        ]
        
        analysis = self.logs_mcp._extract_error_patterns(entries)
        
        # Check severity counts
        self.assertEqual(analysis['severity_counts']['error'], 1)
        self.assertEqual(analysis['severity_counts']['fault'], 1)
        self.assertEqual(analysis['severity_counts']['info'], 1)
        
        # Check process errors
        self.assertEqual(analysis['process_errors']['kernel'], 1)
        self.assertEqual(analysis['process_errors']['Safari'], 1)
        
        # Check error patterns
        self.assertIn('fail', analysis['error_patterns'])
        self.assertIn('timeout', analysis['error_patterns'])
        
        # Check recent errors
        self.assertEqual(len(analysis['recent_errors']), 2)  # Only error and fault levels
    
    def test_extract_error_patterns_with_keywords(self):
        """Test error pattern extraction with various error keywords."""
        entries = [
            {
                'level': 'error',
                'process': 'app1',
                'message': 'Permission denied accessing file',
                'timestamp': '2024-01-15 10:30:45'
            },
            {
                'level': 'error',
                'process': 'app2',
                'message': 'Cannot connect to server',
                'timestamp': '2024-01-15 10:30:46'
            },
            {
                'level': 'fault',
                'process': 'app1',
                'message': 'Access denied for user',
                'timestamp': '2024-01-15 10:30:47'
            }
        ]
        
        analysis = self.logs_mcp._extract_error_patterns(entries)
        
        # Check that error patterns are detected
        self.assertIn('denied', analysis['error_patterns'])
        self.assertIn('cannot', analysis['error_patterns'])
        
        # Check pattern counts
        denied_pattern = analysis['error_patterns']['denied']
        self.assertEqual(denied_pattern['count'], 2)  # Both "permission denied" and "access denied"
        self.assertIn('app1', denied_pattern['processes'])
        self.assertEqual(len(denied_pattern['processes']), 1)  # Only app1 has "denied" messages
        
        cannot_pattern = analysis['error_patterns']['cannot']
        self.assertEqual(cannot_pattern['count'], 1)
        self.assertIn('app2', cannot_pattern['processes'])
        self.assertEqual(len(cannot_pattern['processes']), 1)  # Only app2 has "cannot" message
    
    def test_extract_error_patterns_empty_list(self):
        """Test error pattern extraction with empty log entries."""
        analysis = self.logs_mcp._extract_error_patterns([])
        
        self.assertEqual(analysis['error_patterns'], {})
        self.assertEqual(analysis['process_errors'], {})
        self.assertEqual(analysis['severity_counts'], {})
        self.assertEqual(analysis['recent_errors'], [])
        self.assertEqual(analysis['total_error_keywords_found'], 0)
        self.assertEqual(analysis['unique_error_patterns'], 0)
        self.assertEqual(analysis['processes_with_errors'], 0)
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    def test_execute_when_not_available(self, mock_is_available):
        """Test execute returns error when log command is not available."""
        mock_is_available.return_value = False
        
        result = self.logs_mcp.execute()
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "logs")
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("'log' command is not available", result.error)
        self.assertGreater(result.execution_time, 0)
    
    def test_execute_parameter_validation(self):
        """Test execute validates parameters correctly."""
        with patch.object(self.logs_mcp, 'is_available', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = ""
                
                # Test invalid hours
                result = self.logs_mcp.execute(hours=0)
                # Should use default value of 1
                self.assertEqual(result.metadata['hours'], 1)
                
                result = self.logs_mcp.execute(hours=25)
                # Should use default value of 1
                self.assertEqual(result.metadata['hours'], 1)
                
                # Test invalid max_entries
                result = self.logs_mcp.execute(max_entries=50)
                # Should use default value of 1000
                self.assertEqual(result.metadata['max_entries'], 1000)
                
                result = self.logs_mcp.execute(max_entries=6000)
                # Should use default value of 1000
                self.assertEqual(result.metadata['max_entries'], 1000)
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_successful(self, mock_run, mock_is_available):
        """Test successful execution with sample log data."""
        mock_is_available.return_value = True
        
        # Mock successful log command execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = self.sample_log_output
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.logs_mcp.execute(hours=1, severity_levels=['error', 'fault'])
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "logs")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertGreater(result.execution_time, 0)
        
        # Check that log entries were parsed
        self.assertIn('log_entries', result.data)
        self.assertIn('error_analysis', result.data)
        self.assertIn('summary', result.data)
        
        # Check summary data
        summary = result.data['summary']
        self.assertGreater(summary['total_entries'], 0)
        self.assertEqual(summary['time_range_hours'], 1)
        self.assertEqual(summary['severity_levels_requested'], ['error', 'fault'])
        
        # Check metadata
        self.assertEqual(result.metadata['hours'], 1)
        self.assertEqual(result.metadata['severity_levels'], ['error', 'fault'])
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_permission_denied(self, mock_run, mock_is_available):
        """Test execute handles permission denied error."""
        mock_is_available.return_value = True
        
        # Mock permission denied error
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "permission denied accessing system logs"
        mock_run.return_value = mock_result
        
        result = self.logs_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertIn("Permission denied accessing system logs", result.error)
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_command_failed(self, mock_run, mock_is_available):
        """Test execute handles general command failure."""
        mock_is_available.return_value = True
        
        # Mock command failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Invalid argument provided"
        mock_run.return_value = mock_result
        
        result = self.logs_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertIn("Log command failed: Invalid argument provided", result.error)
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_timeout(self, mock_run, mock_is_available):
        """Test execute handles command timeout."""
        mock_is_available.return_value = True
        
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(['log', 'show'], 60)
        
        result = self.logs_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertIn("Log command timed out", result.error)
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_with_filters(self, mock_run, mock_is_available):
        """Test execute with subsystem and process filters."""
        mock_is_available.return_value = True
        
        # Mock successful execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = self.sample_log_output
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.logs_mcp.execute(
            hours=2,
            severity_levels=['error', 'info'],
            subsystems=['com.apple.kernel'],
            processes=['Safari']
        )
        
        self.assertTrue(result.success)
        
        # Check that filters were applied in metadata
        self.assertEqual(result.metadata['subsystems'], ['com.apple.kernel'])
        self.assertEqual(result.metadata['processes'], ['Safari'])
        
        # Verify the command was built with filters
        command_used = result.metadata['command_used']
        self.assertIn('--subsystem', command_used)
        self.assertIn('com.apple.kernel', command_used)
        self.assertIn('--process', command_used)
        self.assertIn('Safari', command_used)
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_with_invalid_severity_levels(self, mock_run, mock_is_available):
        """Test execute with invalid severity levels."""
        mock_is_available.return_value = True
        
        # Mock successful execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.logs_mcp.execute(severity_levels=['invalid', 'also_invalid'])
        
        self.assertTrue(result.success)
        
        # Should fall back to default severity levels
        self.assertEqual(result.metadata['severity_levels'], ['error', 'fault'])
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_with_parsing_errors(self, mock_run, mock_is_available):
        """Test execute handles log parsing errors gracefully."""
        mock_is_available.return_value = True
        
        # Mock log output with some malformed lines
        malformed_output = "\n".join([
            "2024-01-15 10:30:45-0800 MacBook-Pro kernel[0]: <Error> Valid log entry",
            "This is not a valid log entry",
            "Another malformed line",
            "2024-01-15 10:30:46-0800 MacBook-Pro Safari[456]: <Info> Another valid entry"
        ])
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = malformed_output
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.logs_mcp.execute()
        
        self.assertTrue(result.success)
        
        # Should have parsed valid entries and recorded parse errors
        summary = result.data['summary']
        self.assertEqual(summary['total_entries'], 2)  # Only valid entries
        self.assertEqual(summary['parse_errors'], 2)   # Two malformed lines
    
    @patch('mac_doctor.mcps.logs_mcp.LogsMCP.is_available')
    @patch('subprocess.run')
    def test_execute_exception_handling(self, mock_run, mock_is_available):
        """Test execute handles unexpected exceptions."""
        mock_is_available.return_value = True
        
        # Mock unexpected exception
        mock_run.side_effect = Exception("Unexpected error occurred")
        
        result = self.logs_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("Error collecting system logs: Unexpected error occurred", result.error)
        self.assertGreater(result.execution_time, 0)


if __name__ == '__main__':
    unittest.main()