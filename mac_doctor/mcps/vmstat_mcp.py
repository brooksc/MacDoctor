"""
VM Stats MCP - Mac Collector Plugin for virtual memory statistics.

This module provides virtual memory monitoring capabilities using vm_stat and
memory_pressure commands to gather memory metrics and paging behavior.
"""

import re
import subprocess
import time
from typing import Any, Dict, Optional

from ..interfaces import BaseMCP, MCPResult


class VMStatMCP(BaseMCP):
    """Mac Collector Plugin for virtual memory statistics using vm_stat and memory_pressure."""
    
    @property
    def name(self) -> str:
        """Return the name of this MCP tool."""
        return "vmstat"
    
    @property
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        return "Monitors virtual memory statistics and memory pressure using vm_stat and memory_pressure commands"
    
    def is_available(self) -> bool:
        """Check if vm_stat command is available on the current system."""
        try:
            result = subprocess.run(['which', 'vm_stat'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the input schema for this tool."""
        return {
            "type": "object",
            "properties": {
                "include_memory_pressure": {
                    "type": "boolean",
                    "description": "Whether to include memory_pressure command output",
                    "default": True
                }
            }
        }
    
    def _parse_vm_stat_output(self, output: str) -> Dict[str, Any]:
        """
        Parse vm_stat command output into structured data.
        
        Args:
            output: Raw output from vm_stat command
            
        Returns:
            Dictionary with parsed memory statistics
        """
        stats = {}
        
        # Extract page size from the first line
        page_size_match = re.search(r'page size of (\d+) bytes', output)
        page_size = int(page_size_match.group(1)) if page_size_match else 4096
        stats['page_size_bytes'] = page_size
        
        # Parse memory statistics
        patterns = {
            'pages_free': r'Pages free:\s+(\d+)',
            'pages_active': r'Pages active:\s+(\d+)',
            'pages_inactive': r'Pages inactive:\s+(\d+)',
            'pages_speculative': r'Pages speculative:\s+(\d+)',
            'pages_throttled': r'Pages throttled:\s+(\d+)',
            'pages_wired_down': r'Pages wired down:\s+(\d+)',
            'pages_purgeable': r'Pages purgeable:\s+(\d+)',
            'translation_faults': r'Translation faults:\s+(\d+)',
            'pages_copy_on_write': r'Pages copy-on-write:\s+(\d+)',
            'pages_zero_filled': r'Pages zero filled:\s+(\d+)',
            'pages_reactivated': r'Pages reactivated:\s+(\d+)',
            'pages_purged': r'Pages purged:\s+(\d+)',
            'file_backed_pages': r'File-backed pages:\s+(\d+)',
            'anonymous_pages': r'Anonymous pages:\s+(\d+)',
            'pages_stored_in_compressor': r'Pages stored in compressor:\s+(\d+)',
            'pages_occupied_by_compressor': r'Pages occupied by compressor:\s+(\d+)',
            'decompressions': r'Decompressions:\s+(\d+)',
            'compressions': r'Compressions:\s+(\d+)',
            'pageins': r'Pageins:\s+(\d+)',
            'pageouts': r'Pageouts:\s+(\d+)',
            'swapins': r'Swapins:\s+(\d+)',
            'swapouts': r'Swapouts:\s+(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                stats[key] = int(match.group(1))
            else:
                stats[key] = 0
        
        # Calculate memory usage in bytes and MB
        total_pages = (stats['pages_free'] + stats['pages_active'] + 
                      stats['pages_inactive'] + stats['pages_speculative'] + 
                      stats['pages_wired_down'])
        
        stats['total_pages'] = total_pages
        stats['total_memory_bytes'] = total_pages * page_size
        stats['total_memory_mb'] = round(stats['total_memory_bytes'] / 1024 / 1024, 2)
        
        # Calculate used memory
        used_pages = stats['pages_active'] + stats['pages_inactive'] + stats['pages_wired_down']
        stats['used_pages'] = used_pages
        stats['used_memory_bytes'] = used_pages * page_size
        stats['used_memory_mb'] = round(stats['used_memory_bytes'] / 1024 / 1024, 2)
        
        # Calculate free memory
        stats['free_memory_bytes'] = stats['pages_free'] * page_size
        stats['free_memory_mb'] = round(stats['free_memory_bytes'] / 1024 / 1024, 2)
        
        # Calculate memory pressure indicators
        stats['memory_pressure_score'] = self._calculate_memory_pressure_score(stats)
        
        return stats
    
    def _calculate_memory_pressure_score(self, stats: Dict[str, Any]) -> float:
        """
        Calculate a memory pressure score based on various metrics.
        
        Args:
            stats: Parsed vm_stat statistics
            
        Returns:
            Memory pressure score (0.0 = no pressure, 1.0 = high pressure)
        """
        score = 0.0
        
        # Factor in swap activity (high weight)
        if stats['swapins'] > 0 or stats['swapouts'] > 0:
            score += 0.4
        
        # Factor in page activity
        if stats['pageouts'] > stats['pageins'] * 2:
            score += 0.3
        
        # Factor in compression activity
        if stats['compressions'] > stats['decompressions'] * 2:
            score += 0.2
        
        # Factor in free memory percentage
        if stats['total_pages'] > 0:
            free_percentage = stats['pages_free'] / stats['total_pages']
            if free_percentage < 0.1:  # Less than 10% free
                score += 0.3
            elif free_percentage < 0.2:  # Less than 20% free
                score += 0.1
        
        return min(score, 1.0)
    
    def _get_memory_pressure_info(self) -> Optional[Dict[str, Any]]:
        """
        Get memory pressure information using memory_pressure command.
        
        Returns:
            Dictionary with memory pressure info or None if command fails
        """
        try:
            result = subprocess.run(['memory_pressure'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode != 0:
                return None
            
            output = result.stdout.strip()
            pressure_info = {
                'raw_output': output,
                'status': 'unknown'
            }
            
            # Parse memory pressure status
            if 'System-wide memory pressure: Normal' in output:
                pressure_info['status'] = 'normal'
            elif 'System-wide memory pressure: Warn' in output:
                pressure_info['status'] = 'warn'
            elif 'System-wide memory pressure: Critical' in output:
                pressure_info['status'] = 'critical'
            
            return pressure_info
            
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return None
    
    def execute(self, include_memory_pressure: bool = True, **kwargs) -> MCPResult:
        """
        Execute VM statistics collection and return memory metrics.
        
        Args:
            include_memory_pressure: Whether to include memory_pressure command output
            
        Returns:
            MCPResult with memory statistics or error information
        """
        start_time = time.time()
        
        if not self.is_available():
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error="vm_stat command is not available on this system",
                execution_time=time.time() - start_time
            )
        
        try:
            # Execute vm_stat command
            result = subprocess.run(['vm_stat'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=15)
            
            if result.returncode != 0:
                return MCPResult(
                    tool_name=self.name,
                    success=False,
                    data={},
                    error=f"vm_stat command failed with return code {result.returncode}: {result.stderr}",
                    execution_time=time.time() - start_time
                )
            
            # Parse vm_stat output
            vm_stats = self._parse_vm_stat_output(result.stdout)
            
            # Prepare result data
            result_data = {
                'vm_stat': vm_stats,
                'raw_vm_stat_output': result.stdout.strip()
            }
            
            # Get memory pressure info if requested
            if include_memory_pressure:
                pressure_info = self._get_memory_pressure_info()
                if pressure_info:
                    result_data['memory_pressure'] = pressure_info
                else:
                    result_data['memory_pressure'] = {
                        'status': 'unavailable',
                        'error': 'memory_pressure command not available or failed'
                    }
            
            execution_time = time.time() - start_time
            
            return MCPResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    'include_memory_pressure': include_memory_pressure,
                    'memory_pressure_score': vm_stats.get('memory_pressure_score', 0.0)
                }
            )
            
        except subprocess.TimeoutExpired:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error="vm_stat command timed out after 15 seconds",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Error collecting VM statistics: {str(e)}",
                execution_time=time.time() - start_time
            )