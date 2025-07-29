"""
Logs MCP - Mac Collector Plugin for system log monitoring.

This module provides system log monitoring capabilities using the 'log show' command
to read system logs, filter by time range and severity levels, and extract relevant error patterns.
"""

import re
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..interfaces import BaseMCP, MCPResult


class LogsMCP(BaseMCP):
    """Mac Collector Plugin for system log monitoring using 'log show' command."""
    
    @property
    def name(self) -> str:
        """Return the name of this MCP tool."""
        return "logs"
    
    @property
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        return "Monitors system logs using 'log show' command with filtering by time range and severity levels"
    
    def is_available(self) -> bool:
        """Check if the log command is available on the current system."""
        try:
            result = subprocess.run(['which', 'log'], 
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
                "hours": {
                    "type": "integer",
                    "description": "Number of hours back to retrieve logs",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 24
                },
                "severity_levels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Log severity levels to include (debug, info, default, error, fault)",
                    "default": ["error", "fault"]
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum number of log entries to retrieve",
                    "default": 1000,
                    "minimum": 100,
                    "maximum": 5000
                },
                "subsystems": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific subsystems to filter logs for (optional)",
                    "default": []
                },
                "processes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific processes to filter logs for (optional)",
                    "default": []
                }
            }
        }
    
    def _build_log_command(self, hours: int, severity_levels: List[str], max_entries: int,
                          subsystems: List[str], processes: List[str]) -> List[str]:
        """
        Build the log show command with appropriate filters.
        
        Args:
            hours: Number of hours back to retrieve logs
            severity_levels: Log severity levels to include
            max_entries: Maximum number of log entries
            subsystems: Specific subsystems to filter for
            processes: Specific processes to filter for
            
        Returns:
            List of command arguments for subprocess
        """
        # Calculate start time
        start_time = datetime.now() - timedelta(hours=hours)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Base command
        cmd = ['log', 'show', '--start', start_time_str, '--style', 'syslog']
        
        # Add severity level filter
        if severity_levels:
            # Map severity levels to log command format
            level_map = {
                'debug': 'debug',
                'info': 'info', 
                'default': 'default',
                'error': 'error',
                'fault': 'fault'
            }
            
            valid_levels = [level_map[level] for level in severity_levels if level in level_map]
            if valid_levels:
                cmd.extend(['--level', ','.join(valid_levels)])
        
        # Add subsystem filters
        for subsystem in subsystems:
            cmd.extend(['--subsystem', subsystem])
        
        # Add process filters
        for process in processes:
            cmd.extend(['--process', process])
        
        # Limit number of entries
        cmd.extend(['--max', str(max_entries)])
        
        return cmd
    
    def _parse_log_entry(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single log entry line into structured data.
        
        Args:
            line: Raw log line from 'log show' command
            
        Returns:
            Dictionary with parsed log entry data or None if parsing fails
        """
        # Log format: timestamp hostname process[pid]: <level> message
        # Example: 2024-01-15 10:30:45.123456-0800 MacBook-Pro kernel[0]: <Error> Something went wrong
        
        # Regex pattern to match log entries
        pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+[+-]\d{4})\s+(\S+)\s+([^[]+)\[(\d+)\]:\s*(?:<(\w+)>)?\s*(.*)$'
        
        match = re.match(pattern, line.strip())
        if not match:
            # Try simpler pattern without microseconds
            pattern2 = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s+(\S+)\s+([^[]+)\[(\d+)\]:\s*(?:<(\w+)>)?\s*(.*)$'
            match = re.match(pattern2, line.strip())
        
        if not match:
            return None
        
        timestamp_str = match.group(1)
        hostname = match.group(2)
        process = match.group(3).strip()
        pid = match.group(4)
        level = match.group(5) or 'default'
        message = match.group(6).strip()
        
        try:
            # Parse timestamp (handle different formats)
            if '.' in timestamp_str:
                # With microseconds
                dt = datetime.strptime(timestamp_str.split('.')[0] + timestamp_str[-5:], '%Y-%m-%d %H:%M:%S%z')
            else:
                # Without microseconds
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            # Fallback to current time if parsing fails
            dt = datetime.now()
        
        return {
            'timestamp': timestamp_str,
            'datetime': dt,
            'hostname': hostname,
            'process': process,
            'pid': int(pid),
            'level': level.lower(),
            'message': message,
            'raw_line': line.strip()
        }
    
    def _extract_error_patterns(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and analyze error patterns from log entries.
        
        Args:
            log_entries: List of parsed log entries
            
        Returns:
            Dictionary with error pattern analysis
        """
        error_patterns = {}
        process_errors = {}
        severity_counts = {}
        recent_errors = []
        
        # Common error keywords to look for
        error_keywords = [
            'error', 'fail', 'crash', 'panic', 'abort', 'exception', 'timeout',
            'denied', 'refused', 'invalid', 'corrupt', 'missing', 'not found',
            'unable', 'cannot', 'could not', 'permission', 'access'
        ]
        
        for entry in log_entries:
            level = entry['level']
            message = entry['message'].lower()
            process = entry['process']
            
            # Count severity levels
            severity_counts[level] = severity_counts.get(level, 0) + 1
            
            # Count errors by process
            if level in ['error', 'fault']:
                process_errors[process] = process_errors.get(process, 0) + 1
            
            # Look for error patterns in messages
            for keyword in error_keywords:
                if keyword in message:
                    if keyword not in error_patterns:
                        error_patterns[keyword] = {
                            'count': 0,
                            'processes': set(),
                            'recent_messages': []
                        }
                    
                    error_patterns[keyword]['count'] += 1
                    error_patterns[keyword]['processes'].add(process)
                    
                    # Keep recent messages (up to 5 per pattern)
                    if len(error_patterns[keyword]['recent_messages']) < 5:
                        error_patterns[keyword]['recent_messages'].append({
                            'timestamp': entry['timestamp'],
                            'process': process,
                            'message': entry['message'][:200]  # Truncate long messages
                        })
            
            # Collect recent high-severity errors
            if level in ['error', 'fault'] and len(recent_errors) < 20:
                recent_errors.append({
                    'timestamp': entry['timestamp'],
                    'process': process,
                    'level': level,
                    'message': entry['message'][:200]
                })
        
        # Convert sets to lists for JSON serialization
        for pattern in error_patterns.values():
            pattern['processes'] = list(pattern['processes'])
        
        # Sort patterns by frequency
        sorted_patterns = dict(sorted(error_patterns.items(), 
                                    key=lambda x: x[1]['count'], 
                                    reverse=True))
        
        # Sort process errors by count
        sorted_process_errors = dict(sorted(process_errors.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
        
        return {
            'error_patterns': sorted_patterns,
            'process_errors': sorted_process_errors,
            'severity_counts': severity_counts,
            'recent_errors': recent_errors,
            'total_error_keywords_found': sum(p['count'] for p in error_patterns.values()),
            'unique_error_patterns': len(error_patterns),
            'processes_with_errors': len(process_errors)
        }
    
    def execute(self, hours: int = 1, severity_levels: List[str] = None, max_entries: int = 1000,
                subsystems: List[str] = None, processes: List[str] = None, **kwargs) -> MCPResult:
        """
        Execute log monitoring and return system log analysis.
        
        Args:
            hours: Number of hours back to retrieve logs
            severity_levels: Log severity levels to include
            max_entries: Maximum number of log entries to retrieve
            subsystems: Specific subsystems to filter for
            processes: Specific processes to filter for
            
        Returns:
            MCPResult with log analysis or error information
        """
        start_time = time.time()
        
        if not self.is_available():
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error="The 'log' command is not available on this system",
                execution_time=time.time() - start_time
            )
        
        # Set defaults and validate parameters
        if severity_levels is None:
            severity_levels = ["error", "fault"]
        
        if subsystems is None:
            subsystems = []
        
        if processes is None:
            processes = []
        
        # Validate parameters
        if not isinstance(hours, int) or hours < 1 or hours > 24:
            hours = 1
        
        if not isinstance(max_entries, int) or max_entries < 100 or max_entries > 5000:
            max_entries = 1000
        
        # Validate severity levels
        valid_levels = ['debug', 'info', 'default', 'error', 'fault']
        severity_levels = [level for level in severity_levels if level in valid_levels]
        if not severity_levels:
            severity_levels = ["error", "fault"]
        
        try:
            # Build and execute log command
            log_cmd = self._build_log_command(hours, severity_levels, max_entries, subsystems, processes)
            
            # Execute log command with timeout
            timeout_seconds = min(60, hours * 10)  # Scale timeout with hours requested
            
            log_result = subprocess.run(log_cmd,
                                      capture_output=True,
                                      text=True,
                                      timeout=timeout_seconds)
            
            if log_result.returncode != 0:
                error_msg = log_result.stderr.strip()
                if "permission denied" in error_msg.lower():
                    return MCPResult(
                        tool_name=self.name,
                        success=False,
                        data={},
                        error="Permission denied accessing system logs. Try running with elevated privileges.",
                        execution_time=time.time() - start_time
                    )
                else:
                    return MCPResult(
                        tool_name=self.name,
                        success=False,
                        data={},
                        error=f"Log command failed: {error_msg}",
                        execution_time=time.time() - start_time
                    )
            
            # Parse log output
            log_lines = log_result.stdout.strip().split('\n')
            parsed_entries = []
            parse_errors = 0
            
            for line in log_lines:
                if line.strip():
                    entry = self._parse_log_entry(line)
                    if entry:
                        parsed_entries.append(entry)
                    else:
                        parse_errors += 1
            
            # Sort entries by timestamp (most recent first)
            parsed_entries.sort(key=lambda x: x['datetime'], reverse=True)
            
            # Extract error patterns and analysis
            error_analysis = self._extract_error_patterns(parsed_entries)
            
            # Build result data
            result_data = {
                'log_entries': parsed_entries,
                'error_analysis': error_analysis,
                'summary': {
                    'total_entries': len(parsed_entries),
                    'parse_errors': parse_errors,
                    'time_range_hours': hours,
                    'severity_levels_requested': severity_levels,
                    'subsystems_filtered': subsystems,
                    'processes_filtered': processes,
                    'oldest_entry': parsed_entries[-1]['timestamp'] if parsed_entries else None,
                    'newest_entry': parsed_entries[0]['timestamp'] if parsed_entries else None
                },
                'raw_log_output': log_result.stdout.strip()
            }
            
            execution_time = time.time() - start_time
            
            return MCPResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    'hours': hours,
                    'severity_levels': severity_levels,
                    'max_entries': max_entries,
                    'subsystems': subsystems,
                    'processes': processes,
                    'command_used': ' '.join(log_cmd)
                }
            )
            
        except subprocess.TimeoutExpired:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Log command timed out after {timeout_seconds} seconds. Try reducing the time range or max entries.",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Error collecting system logs: {str(e)}",
                execution_time=time.time() - start_time
            )