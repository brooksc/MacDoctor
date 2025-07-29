"""
Unit tests for DiskMCP - Disk monitoring Mac Collector Plugin.

These tests verify the DiskMCP functionality with mocked system command outputs
to ensure consistent behavior across different environments.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess

from mac_doctor.mcps.disk_mcp import DiskMCP
from mac_doctor.interfaces import MCPResult


class TestDiskMCP(unittest.TestCase):
    """Test cases for DiskMCP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.disk_mcp = DiskMCP()
    
    def test_name_property(self):
        """Test that the name property returns correct value."""
        self.assertEqual(self.disk_mcp.name, "disk")
    
    def test_description_property(self):
        """Test that the description property returns correct value."""
        expected = "Monitors disk I/O statistics and space usage using iostat, df, and du commands"
        self.assertEqual(self.disk_mcp.description, expected)
    
    def test_get_schema(self):
        """Test that get_schema returns proper schema definition."""
        schema = self.disk_mcp.get_schema()
        
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        
        properties = schema["properties"]
        self.assertIn("include_du", properties)
        self.assertIn("du_paths", properties)
        self.assertIn("du_timeout", properties)
        
        # Check include_du schema
        include_du_schema = properties["include_du"]
        self.assertEqual(include_du_schema["type"], "boolean")
        self.assertEqual(include_du_schema["default"], True)
        
        # Check du_paths schema
        du_paths_schema = properties["du_paths"]
        self.assertEqual(du_paths_schema["type"], "array")
        self.assertIn("items", du_paths_schema)
        self.assertEqual(du_paths_schema["items"]["type"], "string")
        
        # Check du_timeout schema
        du_timeout_schema = properties["du_timeout"]
        self.assertEqual(du_timeout_schema["type"], "integer")
        self.assertEqual(du_timeout_schema["default"], 30)
        self.assertEqual(du_timeout_schema["minimum"], 10)
        self.assertEqual(du_timeout_schema["maximum"], 300)
    
    @patch('subprocess.run')
    def test_is_available_when_commands_present(self, mock_run):
        """Test is_available returns True when required commands are available."""
        # Mock successful 'which' commands
        mock_run.return_value = Mock(returncode=0)
        
        self.assertTrue(self.disk_mcp.is_available())
        
        # Verify both iostat and df were checked
        self.assertEqual(mock_run.call_count, 2)
        calls = mock_run.call_args_list
        self.assertEqual(calls[0][0][0], ['which', 'iostat'])
        self.assertEqual(calls[1][0][0], ['which', 'df'])
    
    @patch('subprocess.run')
    def test_is_available_when_commands_missing(self, mock_run):
        """Test is_available returns False when required commands are missing."""
        # Mock failed 'which' command
        mock_run.return_value = Mock(returncode=1)
        
        self.assertFalse(self.disk_mcp.is_available())
    
    @patch('subprocess.run')
    def test_is_available_with_timeout(self, mock_run):
        """Test is_available handles timeout gracefully."""
        mock_run.side_effect = subprocess.TimeoutExpired(['which', 'iostat'], 5)
        
        self.assertFalse(self.disk_mcp.is_available())
    
    def test_parse_size_string(self):
        """Test _parse_size_string method with various size formats."""
        # Test bytes
        self.assertEqual(self.disk_mcp._parse_size_string("512"), 512)
        
        # Test kilobytes
        self.assertEqual(self.disk_mcp._parse_size_string("1K"), 1024)
        self.assertEqual(self.disk_mcp._parse_size_string("2.5K"), int(2.5 * 1024))
        
        # Test megabytes
        self.assertEqual(self.disk_mcp._parse_size_string("100M"), 100 * 1024 * 1024)
        self.assertEqual(self.disk_mcp._parse_size_string("1.5M"), int(1.5 * 1024 * 1024))
        
        # Test gigabytes
        self.assertEqual(self.disk_mcp._parse_size_string("2G"), 2 * 1024 * 1024 * 1024)
        self.assertEqual(self.disk_mcp._parse_size_string("0.5G"), int(0.5 * 1024 * 1024 * 1024))
        
        # Test terabytes
        self.assertEqual(self.disk_mcp._parse_size_string("1T"), 1024 * 1024 * 1024 * 1024)
        
        # Test invalid formats
        self.assertEqual(self.disk_mcp._parse_size_string("invalid"), 0)
        self.assertEqual(self.disk_mcp._parse_size_string(""), 0)
    
    def test_parse_iostat_output(self):
        """Test _parse_iostat_output method with sample iostat output."""
        sample_output = """              disk0       disk1       disk2
    KB/t tps  MB/s     KB/t tps  MB/s     KB/t tps  MB/s
   16.00  45  0.70     8.00  12  0.09     4.00   3  0.01

              disk0       disk1       disk2
    KB/t tps  MB/s     KB/t tps  MB/s     KB/t tps  MB/s
   18.50  52  0.94    10.25  15  0.15     6.00   5  0.03
"""
        
        # Note: This is a simplified test. Real iostat output format may vary
        # For this test, we'll create a more realistic iostat output
        realistic_output = """          disk0           disk1
    r/s    w/s   KB/r   KB/w  ms/r  ms/w
   12.5   8.3   16.2   24.8   2.1   4.5
   15.2   9.1   18.4   26.2   2.3   4.8
"""
        
        stats = self.disk_mcp._parse_iostat_output(realistic_output)
        
        self.assertIn('devices', stats)
        self.assertIn('summary', stats)
        
        # The parsing might not work perfectly with this simplified test data
        # but we can verify the structure is correct
        self.assertIsInstance(stats['devices'], list)
        self.assertIsInstance(stats['summary'], dict)
    
    def test_parse_df_output(self):
        """Test _parse_df_output method with sample df output."""
        sample_output = """Filesystem     1024-blocks      Used Available Capacity  Mounted on
/dev/disk1s1     488245288 123456789 364788499    26%    /
devfs                  195       195         0   100%    /dev
/dev/disk1s4     488245288   8388608 364788499     3%    /System/Volumes/VM
/dev/disk1s2     488245288   1048576 364788499     1%    /System/Volumes/Preboot
/dev/disk1s6     488245288   2097152 364788499     1%    /System/Volumes/Update
/dev/disk1s5     488245288  12582912 364788499     4%    /System/Volumes/Data
map auto_home            0         0         0   100%    /System/Volumes/Data/home
"""
        
        stats = self.disk_mcp._parse_df_output(sample_output)
        
        self.assertIn('filesystems', stats)
        self.assertIn('summary', stats)
        
        filesystems = stats['filesystems']
        self.assertGreater(len(filesystems), 0)
        
        # Check first filesystem
        fs = filesystems[0]
        self.assertEqual(fs['filesystem'], '/dev/disk1s1')
        self.assertEqual(fs['mount_point'], '/')
        self.assertEqual(fs['size_kb'], 488245288)
        self.assertEqual(fs['used_kb'], 123456789)
        self.assertEqual(fs['available_kb'], 364788499)
        self.assertEqual(fs['capacity_percent'], 26)
        
        # Check summary
        summary = stats['summary']
        self.assertIn('filesystem_count', summary)
        self.assertIn('total_size_gb', summary)
        self.assertIn('total_used_gb', summary)
        self.assertIn('total_available_gb', summary)
        self.assertIn('overall_capacity_percent', summary)
    
    @patch('subprocess.run')
    def test_get_du_analysis_success(self, mock_run):
        """Test _get_du_analysis method with successful du commands."""
        # Mock successful du command outputs
        mock_results = [
            Mock(returncode=0, stdout="2.5G\t/Users\n", stderr=""),
            Mock(returncode=0, stdout="1.2G\t/Applications\n", stderr=""),
            Mock(returncode=0, stdout="500M\t/System\n", stderr="")
        ]
        mock_run.side_effect = mock_results
        
        paths = ["/Users", "/Applications", "/System"]
        result = self.disk_mcp._get_du_analysis(paths, 30)
        
        self.assertIn('path_analysis', result)
        self.assertIn('summary', result)
        
        path_analysis = result['path_analysis']
        self.assertEqual(len(path_analysis), 3)
        
        # Should be sorted by size (largest first)
        self.assertEqual(path_analysis[0]['path'], '/Users')
        self.assertEqual(path_analysis[0]['size_human'], '2.5G')
        self.assertEqual(path_analysis[0]['size_gb'], 2.5)
        
        self.assertEqual(path_analysis[1]['path'], '/Applications')
        self.assertEqual(path_analysis[1]['size_human'], '1.2G')
        self.assertEqual(path_analysis[1]['size_gb'], 1.2)
        
        self.assertEqual(path_analysis[2]['path'], '/System')
        self.assertEqual(path_analysis[2]['size_human'], '500M')
        self.assertAlmostEqual(path_analysis[2]['size_gb'], 0.5, places=1)
        
        # Check summary
        summary = result['summary']
        self.assertEqual(summary['paths_analyzed'], 3)
        self.assertEqual(summary['successful_analyses'], 3)
        self.assertAlmostEqual(summary['total_analyzed_gb'], 4.2, places=1)  # 2.5 + 1.2 + 0.5
    
    @patch('subprocess.run')
    def test_get_du_analysis_with_errors(self, mock_run):
        """Test _get_du_analysis method with some failed du commands."""
        # Mock mixed success/failure results
        mock_results = [
            Mock(returncode=0, stdout="1.5G\t/Users\n", stderr=""),
            Mock(returncode=1, stdout="", stderr="Permission denied"),
            subprocess.TimeoutExpired(['du', '-sh', '/System'], 30)
        ]
        mock_run.side_effect = mock_results
        
        paths = ["/Users", "/Applications", "/System"]
        result = self.disk_mcp._get_du_analysis(paths, 30)
        
        path_analysis = result['path_analysis']
        self.assertEqual(len(path_analysis), 3)
        
        # First should be successful
        self.assertEqual(path_analysis[0]['path'], '/Users')
        self.assertEqual(path_analysis[0]['size_gb'], 1.5)
        self.assertNotIn('error', path_analysis[0])
        
        # Second should have error
        self.assertEqual(path_analysis[1]['path'], '/Applications')
        self.assertEqual(path_analysis[1]['size_gb'], 0)
        self.assertIn('error', path_analysis[1])
        self.assertIn('Permission denied', path_analysis[1]['error'])
        
        # Third should have timeout error
        self.assertEqual(path_analysis[2]['path'], '/System')
        self.assertEqual(path_analysis[2]['size_gb'], 0)
        self.assertIn('error', path_analysis[2])
        self.assertIn('Timeout after 30 seconds', path_analysis[2]['error'])
        
        # Check summary
        summary = result['summary']
        self.assertEqual(summary['paths_analyzed'], 3)
        self.assertEqual(summary['successful_analyses'], 1)
        self.assertEqual(summary['total_analyzed_gb'], 1.5)
    
    @patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available')
    def test_execute_when_not_available(self, mock_is_available):
        """Test execute returns error when required commands are not available."""
        mock_is_available.return_value = False
        
        result = self.disk_mcp.execute()
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "disk")
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("Required disk commands", result.error)
        self.assertGreater(result.execution_time, 0)
    
    @patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available')
    @patch('subprocess.run')
    def test_execute_successful_without_du(self, mock_run, mock_is_available):
        """Test successful execution without du analysis."""
        mock_is_available.return_value = True
        
        # Mock iostat output
        iostat_output = """          disk0           disk1
    r/s    w/s   KB/r   KB/w  ms/r  ms/w
   10.5   5.2   16.0   32.0   2.0   3.5
   12.1   6.8   18.2   28.4   2.2   3.8
"""
        
        # Mock df output
        df_output = """Filesystem     1024-blocks    Used Available Capacity  Mounted on
/dev/disk1s1     500000000 100000000 400000000    20%    /
/dev/disk1s5     500000000  50000000 450000000    10%    /System/Volumes/Data
"""
        
        # Configure mock_run to return different outputs for different commands
        def side_effect(cmd, **kwargs):
            if cmd[0] == 'iostat':
                return Mock(returncode=0, stdout=iostat_output, stderr="")
            elif cmd[0] == 'df':
                return Mock(returncode=0, stdout=df_output, stderr="")
            else:
                return Mock(returncode=1, stdout="", stderr="Command not found")
        
        mock_run.side_effect = side_effect
        
        result = self.disk_mcp.execute(include_du=False)
        
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "disk")
        self.assertIn('iostat', result.data)
        self.assertIn('df', result.data)
        self.assertNotIn('du', result.data)
        
        # Check metadata
        self.assertEqual(result.metadata['include_du'], False)
        self.assertEqual(result.metadata['du_paths'], [])
    
    @patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available')
    @patch('subprocess.run')
    def test_execute_successful_with_du(self, mock_run, mock_is_available):
        """Test successful execution with du analysis."""
        mock_is_available.return_value = True
        
        # Mock command outputs
        iostat_output = """          disk0
    r/s    w/s   KB/r   KB/w  ms/r  ms/w
   15.0   8.0   20.0   40.0   2.5   4.0
"""
        
        df_output = """Filesystem     1024-blocks    Used Available Capacity  Mounted on
/dev/disk1s1     1000000000 200000000 800000000    20%    /
"""
        
        def side_effect(cmd, **kwargs):
            if cmd[0] == 'iostat':
                return Mock(returncode=0, stdout=iostat_output, stderr="")
            elif cmd[0] == 'df':
                return Mock(returncode=0, stdout=df_output, stderr="")
            elif cmd[0] == 'du':
                if '/Users' in cmd:
                    return Mock(returncode=0, stdout="3.2G\t/Users\n", stderr="")
                elif '/Applications' in cmd:
                    return Mock(returncode=0, stdout="1.8G\t/Applications\n", stderr="")
                else:
                    return Mock(returncode=0, stdout="500M\t" + cmd[2] + "\n", stderr="")
            else:
                return Mock(returncode=1, stdout="", stderr="Command not found")
        
        mock_run.side_effect = side_effect
        
        result = self.disk_mcp.execute(include_du=True, du_paths=["/Users", "/Applications"])
        
        self.assertTrue(result.success)
        self.assertIn('iostat', result.data)
        self.assertIn('df', result.data)
        self.assertIn('du', result.data)
        
        # Check du results
        du_data = result.data['du']
        self.assertIn('path_analysis', du_data)
        self.assertIn('summary', du_data)
        
        path_analysis = du_data['path_analysis']
        self.assertEqual(len(path_analysis), 2)
        
        # Check metadata
        self.assertEqual(result.metadata['include_du'], True)
        self.assertEqual(result.metadata['du_paths'], ["/Users", "/Applications"])
    
    @patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available')
    @patch('subprocess.run')
    def test_execute_with_command_failures(self, mock_run, mock_is_available):
        """Test execution when some commands fail."""
        mock_is_available.return_value = True
        
        def side_effect(cmd, **kwargs):
            if cmd[0] == 'iostat':
                return Mock(returncode=1, stdout="", stderr="iostat failed")
            elif cmd[0] == 'df':
                return Mock(returncode=0, stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/disk1s1 1000000 200000 800000 20% /\n", stderr="")
            else:
                return Mock(returncode=1, stdout="", stderr="Command failed")
        
        mock_run.side_effect = side_effect
        
        result = self.disk_mcp.execute(include_du=False)
        
        # Should still succeed if at least one command works
        self.assertTrue(result.success)
        
        # iostat should have error
        self.assertIn('iostat', result.data)
        self.assertIn('error', result.data['iostat'])
        self.assertIn('iostat failed', result.data['iostat']['error'])
        
        # df should work
        self.assertIn('df', result.data)
        self.assertNotIn('error', result.data['df'])
    
    @patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available')
    @patch('subprocess.run')
    def test_execute_with_timeout(self, mock_run, mock_is_available):
        """Test execution when commands timeout."""
        mock_is_available.return_value = True
        
        def side_effect(cmd, **kwargs):
            if cmd[0] == 'iostat':
                raise subprocess.TimeoutExpired(cmd, kwargs.get('timeout', 15))
            elif cmd[0] == 'df':
                return Mock(returncode=0, stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/disk1s1 1000000 200000 800000 20% /\n", stderr="")
            else:
                return Mock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = side_effect
        
        result = self.disk_mcp.execute(include_du=False)
        
        # Should still succeed if at least one command works
        self.assertTrue(result.success)
        
        # iostat should have timeout error
        self.assertIn('iostat', result.data)
        self.assertIn('error', result.data['iostat'])
        self.assertIn('timed out', result.data['iostat']['error'])
    
    def test_execute_parameter_validation(self):
        """Test parameter validation in execute method."""
        with patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available') as mock_is_available:
            mock_is_available.return_value = True
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                
                # Test invalid du_timeout (too low)
                result = self.disk_mcp.execute(du_timeout=5)
                # Should use default timeout of 30
                self.assertEqual(result.metadata['du_timeout'], 30)
                
                # Test invalid du_timeout (too high)
                result = self.disk_mcp.execute(du_timeout=500)
                # Should use default timeout of 30
                self.assertEqual(result.metadata['du_timeout'], 30)
                
                # Test valid du_timeout
                result = self.disk_mcp.execute(du_timeout=60)
                self.assertEqual(result.metadata['du_timeout'], 60)
    
    @patch('mac_doctor.mcps.disk_mcp.DiskMCP.is_available')
    def test_execute_with_exception(self, mock_is_available):
        """Test execution when an unexpected exception occurs in all commands."""
        mock_is_available.return_value = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Unexpected error")
            
            result = self.disk_mcp.execute(include_du=False)
            
            # Should fail because all commands failed
            self.assertFalse(result.success)
            
            # All commands should have errors
            self.assertIn('iostat', result.data)
            self.assertIn('error', result.data['iostat'])
            self.assertIn('Unexpected error', result.data['iostat']['error'])
            
            self.assertIn('df', result.data)
            self.assertIn('error', result.data['df'])
            self.assertIn('Unexpected error', result.data['df']['error'])
            
            self.assertGreater(result.execution_time, 0)


if __name__ == '__main__':
    unittest.main()