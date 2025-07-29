"""
Unit tests for ProcessMCP - Process monitoring Mac Collector Plugin.

These tests verify the ProcessMCP functionality with mocked psutil data
to ensure consistent behavior across different environments.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from mac_doctor.mcps.process_mcp import ProcessMCP
from mac_doctor.interfaces import MCPResult


class TestProcessMCP(unittest.TestCase):
    """Test cases for ProcessMCP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.process_mcp = ProcessMCP()
    
    def test_name_property(self):
        """Test that the name property returns correct value."""
        self.assertEqual(self.process_mcp.name, "process")
    
    def test_description_property(self):
        """Test that the description property returns correct value."""
        expected = "Monitors running processes and their CPU/memory usage using psutil"
        self.assertEqual(self.process_mcp.description, expected)
    
    def test_get_schema(self):
        """Test that get_schema returns proper schema definition."""
        schema = self.process_mcp.get_schema()
        
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        
        properties = schema["properties"]
        self.assertIn("top_n", properties)
        self.assertIn("sort_by", properties)
        
        # Check top_n schema
        top_n_schema = properties["top_n"]
        self.assertEqual(top_n_schema["type"], "integer")
        self.assertEqual(top_n_schema["default"], 10)
        self.assertEqual(top_n_schema["minimum"], 1)
        self.assertEqual(top_n_schema["maximum"], 100)
        
        # Check sort_by schema
        sort_by_schema = properties["sort_by"]
        self.assertEqual(sort_by_schema["type"], "string")
        self.assertEqual(sort_by_schema["default"], "cpu")
        self.assertIn("cpu", sort_by_schema["enum"])
        self.assertIn("memory", sort_by_schema["enum"])
        self.assertIn("pid", sort_by_schema["enum"])
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True)
    def test_is_available_when_psutil_present(self):
        """Test is_available returns True when psutil is available."""
        self.assertTrue(self.process_mcp.is_available())
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', False)
    def test_is_available_when_psutil_missing(self):
        """Test is_available returns False when psutil is not available."""
        self.assertFalse(self.process_mcp.is_available())
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', False)
    def test_execute_without_psutil(self):
        """Test execute returns error when psutil is not available."""
        result = self.process_mcp.execute()
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "process")
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("psutil is not available", result.error)
        self.assertGreater(result.execution_time, 0)
    
    def test_execute_with_invalid_top_n(self):
        """Test execute with invalid top_n parameter."""
        with patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True):
            # Test with negative number
            result = self.process_mcp.execute(top_n=-1)
            self.assertFalse(result.success)
            self.assertIn("Process monitoring failed with unknown error", result.error)
            
            # Test with too large number
            result = self.process_mcp.execute(top_n=101)
            self.assertFalse(result.success)
            self.assertIn("Process monitoring failed with unknown error", result.error)
    
    def test_execute_with_invalid_sort_by(self):
        """Test execute with invalid sort_by parameter."""
        with patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True):
            result = self.process_mcp.execute(sort_by="invalid")
            self.assertFalse(result.success)
            self.assertIn("Process monitoring failed with unknown error", result.error)
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True)
    @patch('mac_doctor.mcps.process_mcp.psutil')
    def test_execute_successful_cpu_sort(self, mock_psutil):
        """Test successful execution with CPU sorting."""
        # Mock system stats
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.cpu_percent.return_value = 25.5
        
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory.used = 8 * 1024 * 1024 * 1024    # 8GB
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock process data
        mock_proc1 = Mock()
        mock_proc1.info = {
            'pid': 1234,
            'name': 'test_process_1',
            'cpu_percent': 15.5,
            'memory_percent': 5.2,
            'memory_info': Mock(rss=100 * 1024 * 1024),  # 100MB
            'status': 'running',
            'create_time': 1640995200.0,
            'cmdline': ['test_process_1', '--arg1', '--arg2']
        }
        
        mock_proc2 = Mock()
        mock_proc2.info = {
            'pid': 5678,
            'name': 'test_process_2',
            'cpu_percent': 25.8,
            'memory_percent': 8.1,
            'memory_info': Mock(rss=200 * 1024 * 1024),  # 200MB
            'status': 'running',
            'create_time': 1640995300.0,
            'cmdline': ['test_process_2']
        }
        
        mock_psutil.process_iter.return_value = [mock_proc1, mock_proc2]
        
        result = self.process_mcp.execute(top_n=2, sort_by="cpu")
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "process")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertGreater(result.execution_time, 0)
        
        # Check system stats
        system_stats = result.data["system_overview"]
        self.assertEqual(system_stats["cpu_count"], 8)
        self.assertEqual(system_stats["cpu_usage_percent"], 25.5)
        self.assertEqual(system_stats["memory_total_gb"], 16.0)
        self.assertEqual(system_stats["memory_used_gb"], 8.0)
        self.assertEqual(system_stats["memory_usage_percent"], 50.0)
        self.assertEqual(system_stats["memory_available_gb"], 8.0)
        
        # Check process data
        self.assertEqual(result.data["process_summary"]["total_processes"], 2)
        top_processes = result.data["top_processes"]
        self.assertEqual(len(top_processes), 2)
        
        # Should be sorted by CPU (highest first)
        self.assertEqual(top_processes[0]["pid"], 5678)  # Higher CPU
        self.assertEqual(top_processes[0]["cpu_percent"], 25.8)
        self.assertEqual(top_processes[1]["pid"], 1234)  # Lower CPU
        self.assertEqual(top_processes[1]["cpu_percent"], 15.5)
        
        # Check metadata
        self.assertEqual(result.metadata["sort_by"], "cpu")
        self.assertEqual(result.metadata["top_n"], 2)
        self.assertEqual(result.metadata["total_processes"], 2)
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True)
    @patch('mac_doctor.mcps.process_mcp.psutil')
    def test_execute_successful_memory_sort(self, mock_psutil):
        """Test successful execution with memory sorting."""
        # Mock system stats
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_percent.return_value = 10.0
        
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024   # 8GB
        mock_memory.used = 4 * 1024 * 1024 * 1024    # 4GB
        mock_memory.percent = 50.0
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock process data with different memory usage
        mock_proc1 = Mock()
        mock_proc1.info = {
            'pid': 1111,
            'name': 'low_memory_proc',
            'cpu_percent': 20.0,
            'memory_percent': 2.5,
            'memory_info': Mock(rss=50 * 1024 * 1024),  # 50MB
            'status': 'running',
            'create_time': 1640995200.0,
            'cmdline': ['low_memory_proc']
        }
        
        mock_proc2 = Mock()
        mock_proc2.info = {
            'pid': 2222,
            'name': 'high_memory_proc',
            'cpu_percent': 5.0,
            'memory_percent': 15.8,
            'memory_info': Mock(rss=500 * 1024 * 1024),  # 500MB
            'status': 'running',
            'create_time': 1640995300.0,
            'cmdline': ['high_memory_proc', '--config', '/path/to/config']
        }
        
        mock_psutil.process_iter.return_value = [mock_proc1, mock_proc2]
        
        result = self.process_mcp.execute(top_n=2, sort_by="memory")
        
        self.assertTrue(result.success)
        top_processes = result.data["top_processes"]
        
        # Should be sorted by memory (highest first)
        self.assertEqual(top_processes[0]["pid"], 2222)  # Higher memory
        self.assertEqual(top_processes[0]["memory_percent"], 15.8)
        self.assertEqual(top_processes[1]["pid"], 1111)  # Lower memory
        self.assertEqual(top_processes[1]["memory_percent"], 2.5)
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True)
    @patch('mac_doctor.mcps.process_mcp.psutil')
    def test_execute_with_process_access_denied(self, mock_psutil):
        """Test execution when some processes can't be accessed."""
        # Create proper exception classes for mocking
        class MockAccessDenied(Exception):
            pass
        
        mock_psutil.AccessDenied = MockAccessDenied
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.ZombieProcess = Exception
        
        # Mock system stats
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_percent.return_value = 15.0
        
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024
        mock_memory.used = 2 * 1024 * 1024 * 1024
        mock_memory.percent = 25.0
        mock_memory.available = 6 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock one accessible process and one that raises AccessDenied
        mock_proc_good = Mock()
        mock_proc_good.info = {
            'pid': 1234,
            'name': 'accessible_proc',
            'cpu_percent': 10.0,
            'memory_percent': 5.0,
            'memory_info': Mock(rss=100 * 1024 * 1024),
            'status': 'running',
            'create_time': 1640995200.0,
            'cmdline': ['accessible_proc']
        }
        
        mock_proc_denied = Mock()
        # Configure the mock to raise AccessDenied when .info is accessed
        type(mock_proc_denied).info = Mock(side_effect=MockAccessDenied())
        
        mock_psutil.process_iter.return_value = [mock_proc_good, mock_proc_denied]
        
        result = self.process_mcp.execute(top_n=5)
        
        # Should succeed and only include the accessible process
        self.assertTrue(result.success)
        self.assertEqual(result.data["process_summary"]["total_processes"], 1)
        self.assertEqual(len(result.data["top_processes"]), 1)
        self.assertEqual(result.data["top_processes"][0]["pid"], 1234)
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True)
    @patch('mac_doctor.mcps.process_mcp.psutil')
    def test_execute_handles_none_values(self, mock_psutil):
        """Test execution handles None values in process data gracefully."""
        # Mock system stats
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_percent.return_value = 20.0
        
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024
        mock_memory.used = 3 * 1024 * 1024 * 1024
        mock_memory.percent = 37.5
        mock_memory.available = 5 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock process with None values
        mock_proc = Mock()
        mock_proc.info = {
            'pid': 9999,
            'name': None,
            'cpu_percent': None,
            'memory_percent': None,
            'memory_info': None,
            'status': 'sleeping',
            'create_time': 1640995200.0,
            'cmdline': None
        }
        
        mock_psutil.process_iter.return_value = [mock_proc]
        
        result = self.process_mcp.execute(top_n=1)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["process_summary"]["total_processes"], 1)
        
        proc_data = result.data["top_processes"][0]
        self.assertEqual(proc_data["cpu_percent"], 0.0)
        self.assertEqual(proc_data["memory_percent"], 0.0)
        self.assertEqual(proc_data["memory_mb"], 0.0)
        self.assertEqual(proc_data["command"], "Unknown")
    
    @patch('mac_doctor.mcps.process_mcp.PSUTIL_AVAILABLE', True)
    @patch('mac_doctor.mcps.process_mcp.psutil')
    def test_execute_with_exception(self, mock_psutil):
        """Test execution when psutil raises an unexpected exception."""
        mock_psutil.process_iter.side_effect = Exception("Unexpected error")
        
        result = self.process_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("Process monitoring failed with unknown error", result.error)
        self.assertGreater(result.execution_time, 0)


if __name__ == '__main__':
    unittest.main()