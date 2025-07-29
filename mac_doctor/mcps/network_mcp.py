"""
Network MCP - Mac Collector Plugin for network monitoring.

This module provides network monitoring capabilities using nettop and netstat commands
to gather network activity and connection information.
"""

import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..interfaces import BaseMCP, MCPResult


class NetworkMCP(BaseMCP):
    """Mac Collector Plugin for network monitoring using nettop and netstat commands."""
    
    @property
    def name(self) -> str:
        """Return the name of this MCP tool."""
        return "network"
    
    @property
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        return "Monitors network activity and connections using nettop and netstat commands"
    
    def is_available(self) -> bool:
        """Check if required network commands are available on the current system."""
        required_commands = ['nettop', 'netstat']
        for cmd in required_commands:
            try:
                result = subprocess.run(['which', cmd], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode != 0:
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the input schema for this tool."""
        return {
            "type": "object",
            "properties": {
                "nettop_duration": {
                    "type": "integer",
                    "description": "Duration in seconds to run nettop for sampling",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 30
                },
                "include_connections": {
                    "type": "boolean",
                    "description": "Whether to include detailed connection information from netstat",
                    "default": True
                },
                "connection_states": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Connection states to filter for in netstat output",
                    "default": ["ESTABLISHED", "LISTEN", "TIME_WAIT"]
                }
            }
        }
    
    def _parse_nettop_output(self, output: str) -> Dict[str, Any]:
        """
        Parse nettop command output into structured data.
        
        Args:
            output: Raw output from nettop command
            
        Returns:
            Dictionary with parsed network activity data
        """
        stats = {
            'processes': [],
            'summary': {}
        }
        
        lines = output.strip().split('\n')
        
        # Find the header line to understand column positions
        header_line = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'PID' in line and 'COMMAND' in line:
                header_line = line
                data_start_idx = i + 1
                break
        
        if not header_line:
            return stats
        
        # Parse data lines
        total_bytes_in = 0
        total_bytes_out = 0
        total_packets_in = 0
        total_packets_out = 0
        
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-'):
                continue
            
            # Split by whitespace but handle command names with spaces
            parts = line.split()
            if len(parts) < 6:
                continue
            
            try:
                # Basic parsing - nettop format can vary
                pid = parts[0]
                command = parts[1]
                
                # Try to extract numeric values (positions may vary)
                numeric_parts = []
                for part in parts[2:]:
                    # Remove commas and convert to numbers
                    clean_part = part.replace(',', '')
                    if clean_part.replace('.', '').replace('-', '').isdigit():
                        numeric_parts.append(float(clean_part) if '.' in clean_part else int(clean_part))
                
                if len(numeric_parts) >= 4:
                    bytes_in = numeric_parts[0] if numeric_parts[0] >= 0 else 0
                    bytes_out = numeric_parts[1] if numeric_parts[1] >= 0 else 0
                    packets_in = numeric_parts[2] if len(numeric_parts) > 2 and numeric_parts[2] >= 0 else 0
                    packets_out = numeric_parts[3] if len(numeric_parts) > 3 and numeric_parts[3] >= 0 else 0
                    
                    process_stats = {
                        'pid': pid,
                        'command': command,
                        'bytes_in': bytes_in,
                        'bytes_out': bytes_out,
                        'packets_in': packets_in,
                        'packets_out': packets_out,
                        'total_bytes': bytes_in + bytes_out,
                        'total_packets': packets_in + packets_out
                    }
                    
                    stats['processes'].append(process_stats)
                    
                    # Accumulate totals
                    total_bytes_in += bytes_in
                    total_bytes_out += bytes_out
                    total_packets_in += packets_in
                    total_packets_out += packets_out
                    
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
        
        # Sort by total network activity
        stats['processes'].sort(key=lambda x: x['total_bytes'], reverse=True)
        
        # Calculate summary statistics
        stats['summary'] = {
            'total_processes': len(stats['processes']),
            'total_bytes_in': total_bytes_in,
            'total_bytes_out': total_bytes_out,
            'total_bytes': total_bytes_in + total_bytes_out,
            'total_packets_in': total_packets_in,
            'total_packets_out': total_packets_out,
            'total_packets': total_packets_in + total_packets_out,
            'bytes_in_mb': round(total_bytes_in / 1024 / 1024, 2),
            'bytes_out_mb': round(total_bytes_out / 1024 / 1024, 2),
            'total_mb': round((total_bytes_in + total_bytes_out) / 1024 / 1024, 2)
        }
        
        return stats
    
    def _parse_netstat_output(self, output: str, filter_states: List[str]) -> Dict[str, Any]:
        """
        Parse netstat command output into structured data.
        
        Args:
            output: Raw output from netstat command
            filter_states: List of connection states to include
            
        Returns:
            Dictionary with parsed connection data
        """
        connections = []
        state_counts = {}
        protocol_counts = {}
        
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Active') or line.startswith('Proto'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            try:
                protocol = parts[0].lower()
                local_address = parts[3] if len(parts) > 3 else ''
                foreign_address = parts[4] if len(parts) > 4 else ''
                state = parts[5] if len(parts) > 5 else 'UNKNOWN'
                
                # Filter by state if specified
                if filter_states and state not in filter_states:
                    continue
                
                # Parse addresses and ports
                local_host, local_port = self._parse_address(local_address)
                foreign_host, foreign_port = self._parse_address(foreign_address)
                
                connection = {
                    'protocol': protocol,
                    'local_host': local_host,
                    'local_port': local_port,
                    'foreign_host': foreign_host,
                    'foreign_port': foreign_port,
                    'state': state
                }
                
                connections.append(connection)
                
                # Count states and protocols
                state_counts[state] = state_counts.get(state, 0) + 1
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
                
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
        
        return {
            'connections': connections,
            'summary': {
                'total_connections': len(connections),
                'state_counts': state_counts,
                'protocol_counts': protocol_counts,
                'unique_local_ports': len(set(conn['local_port'] for conn in connections if conn['local_port'])),
                'unique_foreign_hosts': len(set(conn['foreign_host'] for conn in connections if conn['foreign_host'] and conn['foreign_host'] != '*'))
            }
        }
    
    def _parse_address(self, address: str) -> tuple:
        """
        Parse a network address into host and port components.
        
        Args:
            address: Network address string (e.g., "192.168.1.1.80", "*.22")
            
        Returns:
            Tuple of (host, port)
        """
        if not address or address == '*':
            return ('*', '')
        
        # Handle IPv6 addresses (enclosed in brackets)
        if address.startswith('['):
            bracket_end = address.find(']')
            if bracket_end != -1:
                host = address[1:bracket_end]
                port_part = address[bracket_end + 1:]
                port = port_part.lstrip('.') if port_part.startswith('.') else ''
                return (host, port)
        
        # Handle IPv4 addresses and hostnames
        parts = address.split('.')
        if len(parts) >= 2:
            # Last part is likely the port
            port = parts[-1]
            host = '.'.join(parts[:-1])
            
            # Validate port is numeric
            if not port.isdigit():
                # Might be a hostname with no port
                return (address, '')
            
            return (host, port)
        
        return (address, '')
    
    def execute(self, nettop_duration: int = 5, include_connections: bool = True, 
                connection_states: List[str] = None, **kwargs) -> MCPResult:
        """
        Execute network monitoring and return activity and connection data.
        
        Args:
            nettop_duration: Duration in seconds to run nettop for sampling
            include_connections: Whether to include detailed connection information
            connection_states: Connection states to filter for in netstat output
            
        Returns:
            MCPResult with network statistics or error information
        """
        start_time = time.time()
        
        if not self.is_available():
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error="Required network commands (nettop, netstat) are not available on this system",
                execution_time=time.time() - start_time
            )
        
        # Set default connection states if not provided
        if connection_states is None:
            connection_states = ["ESTABLISHED", "LISTEN", "TIME_WAIT"]
        
        # Validate parameters
        if not isinstance(nettop_duration, int) or nettop_duration < 1 or nettop_duration > 30:
            nettop_duration = 5
        
        try:
            result_data = {}
            
            # Get network activity using nettop
            try:
                # Run nettop with specified duration
                # -P flag for per-process stats, -l 1 for one sample after duration
                nettop_result = subprocess.run(['nettop', '-P', '-l', '1', '-t', str(nettop_duration)], 
                                             capture_output=True, 
                                             text=True, 
                                             timeout=nettop_duration + 10)
                
                if nettop_result.returncode == 0:
                    nettop_stats = self._parse_nettop_output(nettop_result.stdout)
                    result_data['nettop'] = nettop_stats
                    result_data['raw_nettop_output'] = nettop_result.stdout.strip()
                else:
                    error_msg = nettop_result.stderr.strip()
                    if "requires elevated privileges" in error_msg.lower() or "permission denied" in error_msg.lower():
                        result_data['nettop'] = {'error': 'nettop requires elevated privileges (try with sudo)'}
                    else:
                        result_data['nettop'] = {'error': f"nettop failed: {error_msg}"}
                    
            except subprocess.TimeoutExpired:
                result_data['nettop'] = {'error': f'nettop command timed out after {nettop_duration + 10} seconds'}
            except Exception as e:
                result_data['nettop'] = {'error': f'nettop error: {str(e)}'}
            
            # Get connection information using netstat if requested
            if include_connections:
                try:
                    # Use netstat to get connection information
                    # -an for all connections with numeric addresses
                    netstat_result = subprocess.run(['netstat', '-an'], 
                                                  capture_output=True, 
                                                  text=True, 
                                                  timeout=15)
                    
                    if netstat_result.returncode == 0:
                        netstat_stats = self._parse_netstat_output(netstat_result.stdout, connection_states)
                        result_data['netstat'] = netstat_stats
                        result_data['raw_netstat_output'] = netstat_result.stdout.strip()
                    else:
                        result_data['netstat'] = {'error': f"netstat failed: {netstat_result.stderr.strip()}"}
                        
                except subprocess.TimeoutExpired:
                    result_data['netstat'] = {'error': 'netstat command timed out'}
                except Exception as e:
                    result_data['netstat'] = {'error': f'netstat error: {str(e)}'}
            
            execution_time = time.time() - start_time
            
            # Determine overall success
            success = any(key in result_data and 'error' not in result_data[key] 
                         for key in ['nettop', 'netstat'])
            
            return MCPResult(
                tool_name=self.name,
                success=success,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    'nettop_duration': nettop_duration,
                    'include_connections': include_connections,
                    'connection_states': connection_states
                }
            )
            
        except Exception as e:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Error collecting network statistics: {str(e)}",
                execution_time=time.time() - start_time
            )