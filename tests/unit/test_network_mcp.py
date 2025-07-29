"""
Unit tests for NetworkMCP - Network monitoring Mac Collector Plugin.

These tests verify the NetworkMCP functionality with mocked network command outputs
to ensure consistent behavior across different environments.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess

from mac_doctor.mcps.network_mcp import NetworkMCP
from mac_doctor.interfaces import MCPResult


class TestNetworkMCP(unittest.TestCase):
    """Test cases for NetworkMCP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network_mcp = NetworkMCP()
    
    def test_name_property(self):
        """Test that the name property returns correct value."""
        self.assertEqual(self.network_mcp.name, "network")
    
    def test_description_property(self):
        """Test that the description property returns correct value."""
        expected = "Monitors network activity and connections using nettop and netstat commands"
        self.assertEqual(self.network_mcp.description, expected)
    
    def test_get_schema(self):
        """Test that get_schema returns proper schema definition."""
        schema = self.network_mcp.get_schema()
        
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        
        properties = schema["properties"]
        self.assertIn("nettop_duration", properties)
        self.assertIn("include_connections", properties)
        self.assertIn("connection_states", properties)
        
        # Check nettop_duration schema
        duration_schema = properties["nettop_duration"]
        self.assertEqual(duration_schema["type"], "integer")
        self.assertEqual(duration_schema["default"], 5)
        self.assertEqual(duration_schema["minimum"], 1)
        self.assertEqual(duration_schema["maximum"], 30)
        
        # Check include_connections schema
        connections_schema = properties["include_connections"]
        self.assertEqual(connections_schema["type"], "boolean")
        self.assertEqual(connections_schema["default"], True)
        
        # Check connection_states schema
        states_schema = properties["connection_states"]
        self.assertEqual(states_schema["type"], "array")
        self.assertIn("ESTABLISHED", states_schema["default"])
        self.assertIn("LISTEN", states_schema["default"])
        self.assertIn("TIME_WAIT", states_schema["default"])
    
    @patch('subprocess.run')
    def test_is_available_when_commands_present(self, mock_run):
        """Test is_available returns True when both nettop and netstat are available."""
        # Mock successful 'which' commands
        mock_run.return_value = Mock(returncode=0)
        
        self.assertTrue(self.network_mcp.is_available())
        
        # Verify both commands were checked
        self.assertEqual(mock_run.call_count, 2)
        calls = mock_run.call_args_list
        self.assertEqual(calls[0][0][0], ['which', 'nettop'])
        self.assertEqual(calls[1][0][0], ['which', 'netstat'])
    
    @patch('subprocess.run')
    def test_is_available_when_commands_missing(self, mock_run):
        """Test is_available returns False when commands are not available."""
        # Mock failed 'which' command
        mock_run.return_value = Mock(returncode=1)
        
        self.assertFalse(self.network_mcp.is_available())
    
    @patch('subprocess.run')
    def test_is_available_with_timeout(self, mock_run):
        """Test is_available handles timeout gracefully."""
        mock_run.side_effect = subprocess.TimeoutExpired(['which', 'nettop'], 5)
        
        self.assertFalse(self.network_mcp.is_available())
    
    def test_parse_address(self):
        """Test _parse_address method with various address formats."""
        # Test IPv4 with port
        host, port = self.network_mcp._parse_address("192.168.1.1.80")
        self.assertEqual(host, "192.168.1.1")
        self.assertEqual(port, "80")
        
        # Test wildcard
        host, port = self.network_mcp._parse_address("*.22")
        self.assertEqual(host, "*")
        self.assertEqual(port, "22")
        
        # Test IPv6 with brackets
        host, port = self.network_mcp._parse_address("[::1].8080")
        self.assertEqual(host, "::1")
        self.assertEqual(port, "8080")
        
        # Test hostname without port
        host, port = self.network_mcp._parse_address("localhost")
        self.assertEqual(host, "localhost")
        self.assertEqual(port, "")
        
        # Test empty address
        host, port = self.network_mcp._parse_address("*")
        self.assertEqual(host, "*")
        self.assertEqual(port, "")
    
    def test_parse_nettop_output(self):
        """Test _parse_nettop_output method with sample nettop data."""
        sample_output = """
nettop: sampling for 5 seconds...

                                     BYTES_IN   BYTES_OUT    PKTS_IN   PKTS_OUT
PID  COMMAND
1234 Safari                           1024000      512000        100        50
5678 Chrome                           2048000     1024000        200       100
9999 ssh                               10240        5120         10         5

"""
        
        result = self.network_mcp._parse_nettop_output(sample_output)
        
        self.assertIn('processes', result)
        self.assertIn('summary', result)
        
        processes = result['processes']
        self.assertEqual(len(processes), 3)
        
        # Check sorting (by total bytes, descending)
        self.assertEqual(processes[0]['command'], 'Chrome')  # Highest total
        self.assertEqual(processes[0]['bytes_in'], 2048000)
        self.assertEqual(processes[0]['bytes_out'], 1024000)
        self.assertEqual(processes[0]['total_bytes'], 3072000)
        
        # Check summary
        summary = result['summary']
        self.assertEqual(summary['total_processes'], 3)
        self.assertEqual(summary['total_bytes_in'], 3082240)  # Sum of all bytes_in
        self.assertEqual(summary['total_bytes_out'], 1541120)  # Sum of all bytes_out
        self.assertEqual(summary['total_bytes'], 4623360)  # Total
    
    def test_parse_netstat_output(self):
        """Test _parse_netstat_output method with sample netstat data."""
        sample_output = """
Active Internet connections (including servers)
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  192.168.1.100.22       192.168.1.200.54321    ESTABLISHED
tcp4       0      0  *.80                   *.*                    LISTEN
tcp4       0      0  127.0.0.1.3306         *.*                    LISTEN
tcp4       0      0  192.168.1.100.443      10.0.0.1.12345         TIME_WAIT
udp4       0      0  *.53                   *.*                    
"""
        
        filter_states = ["ESTABLISHED", "LISTEN", "TIME_WAIT"]
        result = self.network_mcp._parse_netstat_output(sample_output, filter_states)
        
        self.assertIn('connections', result)
        self.assertIn('summary', result)
        
        connections = result['connections']
        self.assertEqual(len(connections), 4)  # UDP line should be filtered out due to no state
        
        # Check first connection
        conn = connections[0]
        self.assertEqual(conn['protocol'], 'tcp4')
        self.assertEqual(conn['local_host'], '192.168.1.100')
        self.assertEqual(conn['local_port'], '22')
        self.assertEqual(conn['foreign_host'], '192.168.1.200')
        self.assertEqual(conn['foreign_port'], '54321')
        self.assertEqual(conn['state'], 'ESTABLISHED')
        
        # Check summary
        summary = result['summary']
        self.assertEqual(summary['total_connections'], 4)
        self.assertIn('ESTABLISHED', summary['state_counts'])
        self.assertIn('LISTEN', summary['state_counts'])
        self.assertIn('tcp4', summary['protocol_counts'])
    
    @patch('subprocess.run')
    def test_execute_without_commands_available(self, mock_run):
        """Test execute returns error when commands are not available."""
        # Mock 'which' command to return failure
        mock_run.return_value = Mock(returncode=1)
        
        result = self.network_mcp.execute()
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "network")
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn("Required network commands", result.error)
        self.assertGreater(result.execution_time, 0)
    
    def test_execute_with_invalid_parameters(self):
        """Test execute with invalid parameters uses defaults."""
        with patch('subprocess.run') as mock_run:
            # Mock 'which' commands to succeed
            mock_run.side_effect = [
                Mock(returncode=0),  # which nettop
                Mock(returncode=0),  # which netstat
                Mock(returncode=0, stdout="mock nettop output"),  # nettop
                Mock(returncode=0, stdout="mock netstat output")   # netstat
            ]
            
            # Test with invalid duration
            result = self.network_mcp.execute(nettop_duration=-1)
            
            # Should use default duration of 5
            nettop_call = None
            for call in mock_run.call_args_list:
                if call[0][0][0] == 'nettop':
                    nettop_call = call[0][0]
                    break
            
            self.assertIsNotNone(nettop_call)
            self.assertIn('5', nettop_call)  # Default duration
    
    @patch('subprocess.run')
    def test_execute_successful_with_both_commands(self, mock_run):
        """Test successful execution with both nettop and netstat."""
        # Mock command availability checks
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock nettop output
            Mock(returncode=0, stdout="""
nettop: sampling for 5 seconds...

                                     BYTES_IN   BYTES_OUT    PKTS_IN   PKTS_OUT
PID  COMMAND
1234 Safari                           1024000      512000        100        50
"""),
            # Mock netstat output
            Mock(returncode=0, stdout="""
Active Internet connections
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  192.168.1.100.22       192.168.1.200.54321    ESTABLISHED
""")
        ]
        
        result = self.network_mcp.execute(nettop_duration=5, include_connections=True)
        
        self.assertIsInstance(result, MCPResult)
        self.assertEqual(result.tool_name, "network")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertGreater(result.execution_time, 0)
        
        # Check that both nettop and netstat data are present
        self.assertIn('nettop', result.data)
        self.assertIn('netstat', result.data)
        
        # Check nettop data structure
        nettop_data = result.data['nettop']
        self.assertIn('processes', nettop_data)
        self.assertIn('summary', nettop_data)
        
        # Check netstat data structure
        netstat_data = result.data['netstat']
        self.assertIn('connections', netstat_data)
        self.assertIn('summary', netstat_data)
        
        # Check metadata
        self.assertEqual(result.metadata['nettop_duration'], 5)
        self.assertEqual(result.metadata['include_connections'], True)
    
    @patch('subprocess.run')
    def test_execute_nettop_permission_denied(self, mock_run):
        """Test execution when nettop requires elevated privileges."""
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock nettop permission error
            Mock(returncode=1, stderr="nettop requires elevated privileges"),
            # Mock successful netstat
            Mock(returncode=0, stdout="Active Internet connections\n")
        ]
        
        result = self.network_mcp.execute()
        
        self.assertTrue(result.success)  # Should succeed because netstat worked
        self.assertIn('nettop', result.data)
        self.assertIn('error', result.data['nettop'])
        self.assertIn('elevated privileges', result.data['nettop']['error'])
    
    @patch('subprocess.run')
    def test_execute_without_connections(self, mock_run):
        """Test execution with include_connections=False."""
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock nettop output
            Mock(returncode=0, stdout="nettop output")
        ]
        
        result = self.network_mcp.execute(include_connections=False)
        
        self.assertTrue(result.success)
        self.assertIn('nettop', result.data)
        self.assertNotIn('netstat', result.data)
        self.assertEqual(result.metadata['include_connections'], False)
    
    @patch('subprocess.run')
    def test_execute_nettop_timeout(self, mock_run):
        """Test execution when nettop times out."""
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock nettop timeout
            subprocess.TimeoutExpired(['nettop'], 15),
            # Mock successful netstat
            Mock(returncode=0, stdout="Active Internet connections\n")
        ]
        
        result = self.network_mcp.execute(nettop_duration=5)
        
        self.assertTrue(result.success)  # Should succeed because netstat worked
        self.assertIn('nettop', result.data)
        self.assertIn('error', result.data['nettop'])
        self.assertIn('timed out', result.data['nettop']['error'])
    
    @patch('subprocess.run')
    def test_execute_netstat_failure(self, mock_run):
        """Test execution when netstat fails."""
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock successful nettop
            Mock(returncode=0, stdout="nettop output"),
            # Mock netstat failure
            Mock(returncode=1, stderr="netstat error")
        ]
        
        result = self.network_mcp.execute()
        
        self.assertTrue(result.success)  # Should succeed because nettop worked
        self.assertIn('netstat', result.data)
        self.assertIn('error', result.data['netstat'])
        self.assertIn('netstat failed', result.data['netstat']['error'])
    
    @patch('subprocess.run')
    def test_execute_both_commands_fail(self, mock_run):
        """Test execution when both commands fail."""
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock nettop failure
            Mock(returncode=1, stderr="nettop error"),
            # Mock netstat failure
            Mock(returncode=1, stderr="netstat error")
        ]
        
        result = self.network_mcp.execute()
        
        self.assertFalse(result.success)  # Should fail because both commands failed
        self.assertIn('nettop', result.data)
        self.assertIn('netstat', result.data)
        self.assertIn('error', result.data['nettop'])
        self.assertIn('error', result.data['netstat'])
    
    @patch('subprocess.run')
    def test_execute_with_custom_connection_states(self, mock_run):
        """Test execution with custom connection states filter."""
        mock_run.side_effect = [
            Mock(returncode=0),  # which nettop
            Mock(returncode=0),  # which netstat
            # Mock nettop output
            Mock(returncode=0, stdout="nettop output"),
            # Mock netstat output
            Mock(returncode=0, stdout="""
Active Internet connections
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  192.168.1.100.22       192.168.1.200.54321    ESTABLISHED
tcp4       0      0  *.80                   *.*                    LISTEN
tcp4       0      0  192.168.1.100.443      10.0.0.1.12345         TIME_WAIT
""")
        ]
        
        custom_states = ["ESTABLISHED", "LISTEN"]
        result = self.network_mcp.execute(connection_states=custom_states)
        
        self.assertTrue(result.success)
        self.assertEqual(result.metadata['connection_states'], custom_states)
        
        # Check that TIME_WAIT connections are filtered out
        netstat_data = result.data['netstat']
        connections = netstat_data['connections']
        states = [conn['state'] for conn in connections]
        self.assertIn('ESTABLISHED', states)
        self.assertIn('LISTEN', states)
        self.assertNotIn('TIME_WAIT', states)
    
    @patch('subprocess.run')
    def test_execute_with_exception(self, mock_run):
        """Test execution when an unexpected exception occurs."""
        # Mock command availability checks to succeed, then exception in main execution
        def side_effect_func(*args, **kwargs):
            if args[0] == ['which', 'nettop'] or args[0] == ['which', 'netstat']:
                return Mock(returncode=0)
            else:
                raise Exception("Unexpected error")
        
        mock_run.side_effect = side_effect_func
        
        result = self.network_mcp.execute()
        
        # Both commands should have errors, so overall success should be False
        self.assertFalse(result.success)
        self.assertIn('nettop', result.data)
        self.assertIn('netstat', result.data)
        self.assertIn('error', result.data['nettop'])
        self.assertIn('error', result.data['netstat'])
        self.assertIn('Unexpected error', result.data['nettop']['error'])
        self.assertIn('Unexpected error', result.data['netstat']['error'])
        self.assertGreater(result.execution_time, 0)


if __name__ == '__main__':
    unittest.main()