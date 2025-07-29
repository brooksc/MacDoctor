"""
Unit tests for VMStatMCP - VM Statistics monitoring Mac Collector Plugin.

These tests verify the VMStatMCP functionality with mocked command outputs
to ensure consistent behavior across different environments.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess

from mac_doctor.mcps.vmstat_mcp import VMStatMCP
from mac_doctor.interfaces import MCPResult


class TestVMStatMCP(unittest.TestCase):
    """Test cases for VMStatMCP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vmstat_mcp = VMStatMCP()
        
        # Sample vm_stat output for testing
        self.sample_vm_stat_output = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               123456.
Pages active:                             234567.
Pages inactive:                           345678.
Pages speculative:                         12345.
Pages throttled:                               0.
Pages wired down:                         456789.
Pages purgeable:                           23456.
"Translation faults":                    1234567.
Pages copy-on-write:                       34567.
Pages zero filled:                        456789.
Pages reactivated:                         56789.
Pages purged:                              67890.
File-backed pages:                        123456.
Anonymous pages:                          234567.
Pages stored in compressor:                78901.
Pages occupied by compressor:              12345.
Decompressions:                            89012.
Compressions:                              90123.
Pageins:                                   10111.
Pageouts:                                  11222.
Swapins:                                       0.
Swapouts:                                      0.
"""
    
    def test_name_property(self):
        """Test that the name property returns correct value."""
        self.assertEqual(self.vmstat_mcp.name, "vmstat")
    
    def test_description_property(self):
        """Test that the description property returns correct value."""
        expected = "Monitors virtual memory statistics and memory pressure using vm_stat and memory_pressure commands"
        self.assertEqual(self.vmstat_mcp.description, expected)
    
    def test_get_schema(self):
        """Test that get_schema returns proper schema definition."""
        schema = self.vmstat_mcp.get_schema()
        
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        
        properties = schema["properties"]
        self.assertIn("include_memory_pressure", properties)
        
        # Check include_memory_pressure schema
        pressure_schema = properties["include_memory_pressure"]
        self.assertEqual(pressure_schema["type"], "boolean")
        self.assertEqual(pressure_schema["default"], True)
    
    @patch('subprocess.run')
    def test_is_available_when_vm_stat_present(self, mock_run):
        """Test is_available returns True when vm_stat command is available."""
        mock_run.return_value = Mock(returncode=0)
        self.assertTrue(self.vmstat_mcp.is_available())
        mock_run.assert_called_once_with(['which', 'vm_stat'], 
                                       capture_output=True, 
                                       text=True, 
                                       timeout=5)
    
    @patch('subprocess.run')
    def test_is_available_when_vm_stat_missing(self, mock_run):
        """Test is_available returns False when vm_stat command is not available."""
        mock_run.return_value = Mock(returncode=1)
        self.assertFalse(self.vmstat_mcp.is_available())
    
    @patch('subprocess.run')
    def test_is_available_with_timeout(self, mock_run):
        """Test is_available handles timeout gracefully."""
        mock_run.side_effect = subprocess.TimeoutExpired('which', 5)
        self.assertFalse(self.vmstat_mcp.is_available())
    
    @patch('subprocess.run')
    def test_is_available_with_file_not_found(self, mock_run):
        """Test is_available handles FileNotFoundError gracefully."""
        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(self.vmstat_mcp.is_available())
    
    def test_parse_vm_stat_output(self):
        """Test parsing of vm_stat command output."""
        stats = self.vmstat_mcp._parse_vm_stat_output(self.sample_vm_stat_output)
        
        # Check basic parsing
        self.assertEqual(stats['page_size_bytes'], 4096)
        self.assertEqual(stats['pages_free'], 123456)
        self.assertEqual(stats['pages_active'], 234567)
        self.assertEqual(stats['pages_inactive'], 345678)
        self.assertEqual(stats['pages_wired_down'], 456789)
        self.assertEqual(stats['swapins'], 0)
        self.assertEqual(stats['swapouts'], 0)
        self.assertEqual(stats['pageins'], 10111)
        self.assertEqual(stats['pageouts'], 11222)
        
        # Check calculated values
        expected_total_pages = 123456 + 234567 + 345678 + 12345 + 456789
        self.assertEqual(stats['total_pages'], expected_total_pages)
        
        expected_total_memory_bytes = expected_total_pages * 4096
        self.assertEqual(stats['total_memory_bytes'], expected_total_memory_bytes)
        
        expected_used_pages = 234567 + 345678 + 456789
        self.assertEqual(stats['used_pages'], expected_used_pages)
        
        # Check memory pressure score calculation
        self.assertIn('memory_pressure_score', stats)
        self.assertIsInstance(stats['memory_pressure_score'], float)
        self.assertGreaterEqual(stats['memory_pressure_score'], 0.0)
        self.assertLessEqual(stats['memory_pressure_score'], 1.0)
    
    def test_parse_vm_stat_output_with_missing_values(self):
        """Test parsing handles missing values gracefully."""
        incomplete_output = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               100000.
Pages active:                             200000.
"""
        stats = self.vmstat_mcp._parse_vm_stat_output(incomplete_output)
        
        # Should have defaults for missing values
        self.assertEqual(stats['pages_free'], 100000)
        self.assertEqual(stats['pages_active'], 200000)
        self.assertEqual(stats['pages_inactive'], 0)  # Default for missing
        self.assertEqual(stats['swapins'], 0)  # Default for missing
    
    def test_calculate_memory_pressure_score_no_pressure(self):
        """Test memory pressure score calculation with no pressure indicators."""
        stats = {
            'swapins': 0,
            'swapouts': 0,
            'pageins': 1000,
            'pageouts': 500,
            'compressions': 100,
            'decompressions': 200,
            'pages_free': 200000,
            'total_pages': 1000000
        }
        
        score = self.vmstat_mcp._calculate_memory_pressure_score(stats)
        self.assertEqual(score, 0.0)
    
    def test_calculate_memory_pressure_score_high_pressure(self):
        """Test memory pressure score calculation with high pressure indicators."""
        stats = {
            'swapins': 1000,
            'swapouts': 2000,
            'pageins': 500,
            'pageouts': 2000,
            'compressions': 5000,
            'decompressions': 1000,
            'pages_free': 50000,
            'total_pages': 1000000
        }
        
        score = self.vmstat_mcp._calculate_memory_pressure_score(stats)
        self.assertGreater(score, 0.5)  # Should indicate high pressure
    
    @patch('subprocess.run')
    def test_get_memory_pressure_info_normal(self, mock_run):
        """Test memory pressure info parsing for normal status."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="System-wide memory pressure: Normal\nSome other info\n"
        )
        
        pressure_info = self.vmstat_mcp._get_memory_pressure_info()
        
        self.assertIsNotNone(pressure_info)
        self.assertEqual(pressure_info['status'], 'normal')
        self.assertIn('raw_output', pressure_info)
    
    @patch('subprocess.run')
    def test_get_memory_pressure_info_warn(self, mock_run):
        """Test memory pressure info parsing for warn status."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="System-wide memory pressure: Warn\nWarning details\n"
        )
        
        pressure_info = self.vmstat_mcp._get_memory_pressure_info()
        
        self.assertIsNotNone(pressure_info)
        self.assertEqual(pressure_info['status'], 'warn')
    
    @patch('subprocess.run')
    def test_get_memory_pressure_info_critical(self, mock_run):
        """Test memory pressure info parsing for critical status."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="System-wide memory pressure: Critical\nCritical details\n"
        )
        
        pressure_info = self.vmstat_mcp._get_memory_pressure_info()
        
        self.assertIsNotNone(pressure_info)
        self.assertEqual(pressure_info['status'], 'critical')
    
    @patch('subprocess.run')
    def test_get_memory_pressure_info_command_fails(self, mock_run):
        """Test memory pressure info when command fails."""
        mock_run.return_value = Mock(returncode=1)
        
        pressure_info = self.vmstat_mcp._get_memory_pressure_info()
        
        self.assertIsNone(pressure_info)
    
    @patch('subprocess.run')
    def test_get_memory_pressure_info_timeout(self, mock_run):
        """Test memory pressure info handles timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('memory_pressure', 10)
        
        pressure_info = self.vmstat_mcp._get_memory_pressure_info()
        
        self.assertIsNone(pressure_info)
    
    @patch.object(VMStatMCP, 'is_available')
    def test_execute_when_not_available(self, mock_is_available):
        """Test execute returns error when vm_stat is not available."""
        mock_is_available.return_value = False
        
        result = self.vmstat_mcp.execute()
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "vmstat")
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("vm_stat command is not available", result.error)
        self.assertGreater(result.execution_time, 0)
    
    @patch.object(VMStatMCP, 'is_available')
    @patch('subprocess.run')
    def test_execute_vm_stat_command_fails(self, mock_run, mock_is_available):
        """Test execute when vm_stat command fails."""
        mock_is_available.return_value = True
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Permission denied"
        )
        
        result = self.vmstat_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertIn("vm_stat command failed with return code 1", result.error)
    
    @patch.object(VMStatMCP, 'is_available')
    @patch('subprocess.run')
    def test_execute_vm_stat_timeout(self, mock_run, mock_is_available):
        """Test execute when vm_stat command times out."""
        mock_is_available.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired('vm_stat', 15)
        
        result = self.vmstat_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertIn("vm_stat command timed out after 15 seconds", result.error)
    
    @patch.object(VMStatMCP, 'is_available')
    @patch.object(VMStatMCP, '_get_memory_pressure_info')
    @patch('subprocess.run')
    def test_execute_successful_with_memory_pressure(self, mock_run, mock_pressure, mock_is_available):
        """Test successful execution with memory pressure info."""
        mock_is_available.return_value = True
        mock_run.return_value = Mock(
            returncode=0,
            stdout=self.sample_vm_stat_output
        )
        mock_pressure.return_value = {
            'status': 'normal',
            'raw_output': 'System-wide memory pressure: Normal'
        }
        
        result = self.vmstat_mcp.execute(include_memory_pressure=True)
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "vmstat")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertGreater(result.execution_time, 0)
        
        # Check data structure
        self.assertIn('vm_stat', result.data)
        self.assertIn('memory_pressure', result.data)
        self.assertIn('raw_vm_stat_output', result.data)
        
        # Check vm_stat data
        vm_stats = result.data['vm_stat']
        self.assertEqual(vm_stats['page_size_bytes'], 4096)
        self.assertEqual(vm_stats['pages_free'], 123456)
        
        # Check memory pressure data
        pressure_data = result.data['memory_pressure']
        self.assertEqual(pressure_data['status'], 'normal')
        
        # Check metadata
        self.assertEqual(result.metadata['include_memory_pressure'], True)
        self.assertIn('memory_pressure_score', result.metadata)
    
    @patch.object(VMStatMCP, 'is_available')
    @patch.object(VMStatMCP, '_get_memory_pressure_info')
    @patch('subprocess.run')
    def test_execute_successful_without_memory_pressure(self, mock_run, mock_pressure, mock_is_available):
        """Test successful execution without memory pressure info."""
        mock_is_available.return_value = True
        mock_run.return_value = Mock(
            returncode=0,
            stdout=self.sample_vm_stat_output
        )
        
        result = self.vmstat_mcp.execute(include_memory_pressure=False)
        
        self.assertTrue(result.success)
        self.assertIn('vm_stat', result.data)
        self.assertNotIn('memory_pressure', result.data)
        self.assertEqual(result.metadata['include_memory_pressure'], False)
        
        # Should not call memory pressure function
        mock_pressure.assert_not_called()
    
    @patch.object(VMStatMCP, 'is_available')
    @patch.object(VMStatMCP, '_get_memory_pressure_info')
    @patch('subprocess.run')
    def test_execute_memory_pressure_unavailable(self, mock_run, mock_pressure, mock_is_available):
        """Test execution when memory pressure command is unavailable."""
        mock_is_available.return_value = True
        mock_run.return_value = Mock(
            returncode=0,
            stdout=self.sample_vm_stat_output
        )
        mock_pressure.return_value = None  # Command unavailable
        
        result = self.vmstat_mcp.execute(include_memory_pressure=True)
        
        self.assertTrue(result.success)
        self.assertIn('memory_pressure', result.data)
        
        pressure_data = result.data['memory_pressure']
        self.assertEqual(pressure_data['status'], 'unavailable')
        self.assertIn('error', pressure_data)
    
    @patch.object(VMStatMCP, 'is_available')
    @patch('subprocess.run')
    def test_execute_with_exception(self, mock_run, mock_is_available):
        """Test execute when an unexpected exception occurs."""
        mock_is_available.return_value = True
        mock_run.side_effect = Exception("Unexpected error")
        
        result = self.vmstat_mcp.execute()
        
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("Error collecting VM statistics: Unexpected error", result.error)
        self.assertGreater(result.execution_time, 0)


if __name__ == '__main__':
    unittest.main()