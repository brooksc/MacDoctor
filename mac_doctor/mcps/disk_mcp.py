"""
Disk MCP - Mac Collector Plugin for disk I/O and space monitoring.

This module provides disk monitoring capabilities using iostat, df, and du commands
to gather disk I/O statistics and space usage information.
"""

import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..interfaces import BaseMCP, MCPResult


class DiskMCP(BaseMCP):
    """Mac Collector Plugin for disk I/O and space monitoring using iostat, df, and du commands."""
    
    @property
    def name(self) -> str:
        """Return the name of this MCP tool."""
        return "disk"
    
    @property
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        return "Monitors disk I/O statistics and space usage using iostat, df, and du commands"
    
    def is_available(self) -> bool:
        """Check if required disk commands are available on the current system."""
        required_commands = ['iostat', 'df']
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
                "include_du": {
                    "type": "boolean",
                    "description": "Whether to include disk usage analysis with du command",
                    "default": True
                },
                "du_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to analyze with du command",
                    "default": ["/", "/Users", "/Applications", "/System", "/var"]
                },
                "du_timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for du operations",
                    "default": 30,
                    "minimum": 10,
                    "maximum": 300
                }
            }
        }
    
    def _parse_iostat_output(self, output: str) -> Dict[str, Any]:
        """
        Parse iostat command output into structured data.
        
        Args:
            output: Raw output from iostat command
            
        Returns:
            Dictionary with parsed I/O statistics
        """
        stats = {
            'devices': [],
            'summary': {}
        }
        
        lines = output.strip().split('\n')
        
        # Skip header lines and find device data
        device_section_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for device header line
            if 'device' in line.lower() and ('r/s' in line or 'KB/t' in line):
                device_section_started = True
                continue
            
            # Parse device lines
            if device_section_started and line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        device_stats = {
                            'device': parts[0],
                            'reads_per_sec': float(parts[1]) if parts[1] != '-' else 0.0,
                            'writes_per_sec': float(parts[2]) if parts[2] != '-' else 0.0,
                            'kb_per_read': float(parts[3]) if parts[3] != '-' else 0.0,
                            'kb_per_write': float(parts[4]) if parts[4] != '-' else 0.0,
                            'ms_per_read': float(parts[5]) if parts[5] != '-' else 0.0,
                            'ms_per_write': float(parts[6]) if len(parts) > 6 and parts[6] != '-' else 0.0
                        }
                        
                        # Calculate derived metrics
                        device_stats['total_ops_per_sec'] = device_stats['reads_per_sec'] + device_stats['writes_per_sec']
                        device_stats['read_throughput_kb_per_sec'] = device_stats['reads_per_sec'] * device_stats['kb_per_read']
                        device_stats['write_throughput_kb_per_sec'] = device_stats['writes_per_sec'] * device_stats['kb_per_write']
                        device_stats['total_throughput_kb_per_sec'] = device_stats['read_throughput_kb_per_sec'] + device_stats['write_throughput_kb_per_sec']
                        
                        stats['devices'].append(device_stats)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        
        # Calculate summary statistics
        if stats['devices']:
            total_reads = sum(d['reads_per_sec'] for d in stats['devices'])
            total_writes = sum(d['writes_per_sec'] for d in stats['devices'])
            total_throughput = sum(d['total_throughput_kb_per_sec'] for d in stats['devices'])
            
            stats['summary'] = {
                'total_reads_per_sec': round(total_reads, 2),
                'total_writes_per_sec': round(total_writes, 2),
                'total_ops_per_sec': round(total_reads + total_writes, 2),
                'total_throughput_kb_per_sec': round(total_throughput, 2),
                'total_throughput_mb_per_sec': round(total_throughput / 1024, 2),
                'device_count': len(stats['devices'])
            }
        
        return stats
    
    def _parse_df_output(self, output: str) -> Dict[str, Any]:
        """
        Parse df command output into structured data.
        
        Args:
            output: Raw output from df command
            
        Returns:
            Dictionary with parsed filesystem usage data
        """
        filesystems = []
        lines = output.strip().split('\n')
        
        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # Handle cases where filesystem name contains spaces
                    if len(parts) > 6:
                        # Filesystem name might be split across multiple parts
                        filesystem = ' '.join(parts[:-5])
                        size_kb = int(parts[-5])
                        used_kb = int(parts[-4])
                        avail_kb = int(parts[-3])
                        capacity_str = parts[-2]
                        mount_point = parts[-1]
                    else:
                        filesystem = parts[0]
                        size_kb = int(parts[1])
                        used_kb = int(parts[2])
                        avail_kb = int(parts[3])
                        capacity_str = parts[4]
                        mount_point = parts[5]
                    
                    # Parse capacity percentage
                    capacity_pct = 0
                    if capacity_str.endswith('%'):
                        capacity_pct = int(capacity_str[:-1])
                    
                    fs_info = {
                        'filesystem': filesystem,
                        'mount_point': mount_point,
                        'size_kb': size_kb,
                        'used_kb': used_kb,
                        'available_kb': avail_kb,
                        'capacity_percent': capacity_pct,
                        'size_gb': round(size_kb / 1024 / 1024, 2),
                        'used_gb': round(used_kb / 1024 / 1024, 2),
                        'available_gb': round(avail_kb / 1024 / 1024, 2)
                    }
                    
                    filesystems.append(fs_info)
                    
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue
        
        # Calculate summary
        total_size = sum(fs['size_kb'] for fs in filesystems)
        total_used = sum(fs['used_kb'] for fs in filesystems)
        total_available = sum(fs['available_kb'] for fs in filesystems)
        
        summary = {
            'filesystem_count': len(filesystems),
            'total_size_gb': round(total_size / 1024 / 1024, 2),
            'total_used_gb': round(total_used / 1024 / 1024, 2),
            'total_available_gb': round(total_available / 1024 / 1024, 2),
            'overall_capacity_percent': round((total_used / total_size * 100) if total_size > 0 else 0, 1)
        }
        
        return {
            'filesystems': filesystems,
            'summary': summary
        }
    
    def _get_du_analysis(self, paths: List[str], timeout: int) -> Dict[str, Any]:
        """
        Get disk usage analysis for specified paths using du command.
        
        Args:
            paths: List of paths to analyze
            timeout: Timeout in seconds for du operations
            
        Returns:
            Dictionary with disk usage analysis
        """
        du_results = []
        
        for path in paths:
            try:
                # Use du with human-readable output and max depth of 1
                result = subprocess.run(['du', '-sh', path], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=timeout)
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        parts = output.split('\t', 1)
                        if len(parts) == 2:
                            size_str = parts[0].strip()
                            path_name = parts[1].strip()
                            
                            # Convert size to bytes for comparison
                            size_bytes = self._parse_size_string(size_str)
                            
                            du_results.append({
                                'path': path_name,
                                'size_human': size_str,
                                'size_bytes': size_bytes,
                                'size_gb': round(size_bytes / 1024 / 1024 / 1024, 2) if size_bytes else 0
                            })
                else:
                    du_results.append({
                        'path': path,
                        'size_human': 'N/A',
                        'size_bytes': 0,
                        'size_gb': 0,
                        'error': f"Access denied or path not found: {result.stderr.strip()}"
                    })
                    
            except subprocess.TimeoutExpired:
                du_results.append({
                    'path': path,
                    'size_human': 'N/A',
                    'size_bytes': 0,
                    'size_gb': 0,
                    'error': f"Timeout after {timeout} seconds"
                })
            except Exception as e:
                du_results.append({
                    'path': path,
                    'size_human': 'N/A',
                    'size_bytes': 0,
                    'size_gb': 0,
                    'error': str(e)
                })
        
        # Sort by size (largest first)
        du_results.sort(key=lambda x: x['size_bytes'], reverse=True)
        
        # Calculate summary
        total_analyzed_bytes = sum(r['size_bytes'] for r in du_results if 'error' not in r)
        successful_analyses = len([r for r in du_results if 'error' not in r])
        
        return {
            'path_analysis': du_results,
            'summary': {
                'paths_analyzed': len(paths),
                'successful_analyses': successful_analyses,
                'total_analyzed_gb': round(total_analyzed_bytes / 1024 / 1024 / 1024, 2)
            }
        }
    
    def _parse_size_string(self, size_str: str) -> int:
        """
        Parse a human-readable size string (e.g., '1.5G', '500M') to bytes.
        
        Args:
            size_str: Size string from du command
            
        Returns:
            Size in bytes
        """
        size_str = size_str.strip().upper()
        
        # Extract number and unit
        match = re.match(r'([0-9.]+)([KMGT]?)', size_str)
        if not match:
            return 0
        
        number = float(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            '': 1,
            'K': 1024,
            'M': 1024 ** 2,
            'G': 1024 ** 3,
            'T': 1024 ** 4
        }
        
        return int(number * multipliers.get(unit, 1))
    
    def execute(self, include_du: bool = True, du_paths: List[str] = None, du_timeout: int = 30, **kwargs) -> MCPResult:
        """
        Execute disk monitoring and return I/O statistics and space usage.
        
        Args:
            include_du: Whether to include disk usage analysis with du command
            du_paths: Paths to analyze with du command
            du_timeout: Timeout in seconds for du operations
            
        Returns:
            MCPResult with disk statistics or error information
        """
        start_time = time.time()
        
        if not self.is_available():
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error="Required disk commands (iostat, df) are not available on this system",
                execution_time=time.time() - start_time
            )
        
        # Set default du_paths if not provided
        if du_paths is None:
            du_paths = ["/", "/Users", "/Applications", "/System", "/var"]
        
        # Validate parameters
        if not isinstance(du_timeout, int) or du_timeout < 10 or du_timeout > 300:
            du_timeout = 30
        
        try:
            result_data = {}
            
            # Get I/O statistics using iostat
            try:
                iostat_result = subprocess.run(['iostat', '-d', '1', '2'], 
                                             capture_output=True, 
                                             text=True, 
                                             timeout=15)
                
                if iostat_result.returncode == 0:
                    iostat_stats = self._parse_iostat_output(iostat_result.stdout)
                    result_data['iostat'] = iostat_stats
                    result_data['raw_iostat_output'] = iostat_result.stdout.strip()
                else:
                    result_data['iostat'] = {'error': f"iostat failed: {iostat_result.stderr.strip()}"}
                    
            except subprocess.TimeoutExpired:
                result_data['iostat'] = {'error': 'iostat command timed out'}
            except Exception as e:
                result_data['iostat'] = {'error': f'iostat error: {str(e)}'}
            
            # Get filesystem usage using df
            try:
                df_result = subprocess.run(['df', '-k'], 
                                         capture_output=True, 
                                         text=True, 
                                         timeout=10)
                
                if df_result.returncode == 0:
                    df_stats = self._parse_df_output(df_result.stdout)
                    result_data['df'] = df_stats
                    result_data['raw_df_output'] = df_result.stdout.strip()
                else:
                    result_data['df'] = {'error': f"df failed: {df_result.stderr.strip()}"}
                    
            except subprocess.TimeoutExpired:
                result_data['df'] = {'error': 'df command timed out'}
            except Exception as e:
                result_data['df'] = {'error': f'df error: {str(e)}'}
            
            # Get disk usage analysis using du if requested
            if include_du:
                try:
                    du_stats = self._get_du_analysis(du_paths, du_timeout)
                    result_data['du'] = du_stats
                except Exception as e:
                    result_data['du'] = {'error': f'du analysis error: {str(e)}'}
            
            execution_time = time.time() - start_time
            
            # Determine overall success
            success = any(key in result_data and 'error' not in result_data[key] 
                         for key in ['iostat', 'df', 'du'])
            
            return MCPResult(
                tool_name=self.name,
                success=success,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    'include_du': include_du,
                    'du_paths': du_paths if include_du else [],
                    'du_timeout': du_timeout
                }
            )
            
        except Exception as e:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Error collecting disk statistics: {str(e)}",
                execution_time=time.time() - start_time
            )