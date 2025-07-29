"""
DTrace MCP - Mac Collector Plugin for system call and performance tracing.

This module provides DTrace capabilities for system call analysis and performance
tracing on macOS systems. Requires elevated privileges and handles graceful
degradation when unavailable.
"""

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..interfaces import BaseMCP, MCPResult


class DTraceMCP(BaseMCP):
    """Mac Collector Plugin for DTrace system call and performance tracing."""
    
    # Common DTrace scripts for diagnostic scenarios
    DTRACE_SCRIPTS = {
        "syscall": """
            syscall:::entry
            /execname != "dtrace"/
            {
                @syscalls[execname, probefunc] = count();
            }
            
            tick-10s
            {
                exit(0);
            }
        """,
        
        "io": """
            io:::start
            {
                @io_by_process[execname] = count();
                @io_by_device[args[1]->dev_statname] = count();
            }
            
            tick-10s
            {
                exit(0);
            }
        """,
        
        "proc": """
            proc:::exec-success
            {
                @execs[execname] = count();
                printf("%Y: %s executed %s\\n", walltimestamp, execname, stringof(args[0]->pr_fname));
            }
            
            proc:::exit
            {
                @exits[execname] = count();
            }
            
            tick-10s
            {
                exit(0);
            }
        """,
        
        "network": """
            fbt::tcp_input:entry,
            fbt::tcp_output:entry,
            fbt::udp_input:entry,
            fbt::udp_output:entry
            {
                @network[probefunc, execname] = count();
            }
            
            tick-10s
            {
                exit(0);
            }
        """,
        
        "memory": """
            vminfo:::pgin,
            vminfo:::pgout,
            vminfo:::pgpgin,
            vminfo:::pgpgout
            {
                @vm_activity[probename] = sum(arg0);
            }
            
            tick-10s
            {
                exit(0);
            }
        """
    }
    
    @property
    def name(self) -> str:
        """Return the name of this MCP tool."""
        return "dtrace"
    
    @property
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        return "Performs system call and performance tracing using DTrace"
    
    def is_available(self) -> bool:
        """Check if DTrace is available and we have sufficient privileges."""
        try:
            # Check if dtrace command exists
            result = subprocess.run(
                ["which", "dtrace"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False
            
            # Check if we can run a simple dtrace command
            # This will fail if we don't have proper privileges
            test_result = subprocess.run(
                ["dtrace", "-n", "BEGIN { exit(0); }"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return test_result.returncode == 0
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the input schema for this tool."""
        return {
            "type": "object",
            "properties": {
                "script_type": {
                    "type": "string",
                    "description": "Type of DTrace script to run",
                    "enum": list(self.DTRACE_SCRIPTS.keys()),
                    "default": "syscall"
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration to run DTrace in seconds",
                    "default": 10,
                    "minimum": 5,
                    "maximum": 60
                },
                "custom_script": {
                    "type": "string",
                    "description": "Custom DTrace script to execute (optional)"
                }
            }
        }
    
    def execute(self, script_type: str = "syscall", duration: int = 10, 
                custom_script: Optional[str] = None, **kwargs) -> MCPResult:
        """
        Execute DTrace script for system call and performance analysis.
        
        Args:
            script_type: Type of predefined script to run ('syscall', 'io', 'proc', 'network', 'memory')
            duration: Duration to run DTrace in seconds (5-60)
            custom_script: Optional custom DTrace script to execute
            
        Returns:
            MCPResult with DTrace analysis data or error information
        """
        start_time = time.time()
        
        if not self.is_available():
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error="DTrace is not available or insufficient privileges. "
                      "Try running with sudo or enable DTrace in System Preferences > Security & Privacy",
                execution_time=time.time() - start_time
            )
        
        try:
            # Validate parameters
            if duration < 5 or duration > 60:
                raise ValueError("Duration must be between 5 and 60 seconds")
            
            # Determine which script to use
            if custom_script:
                dtrace_script = custom_script
                script_name = "custom"
            elif script_type in self.DTRACE_SCRIPTS:
                dtrace_script = self.DTRACE_SCRIPTS[script_type]
                # Modify the tick interval based on duration
                dtrace_script = dtrace_script.replace("tick-10s", f"tick-{duration}s")
                script_name = script_type
            else:
                raise ValueError(f"Unknown script_type: {script_type}. "
                               f"Available types: {list(self.DTRACE_SCRIPTS.keys())}")
            
            # Execute DTrace script
            result = self._execute_dtrace_script(dtrace_script, duration)
            
            if not result["success"]:
                return MCPResult(
                    tool_name=self.name,
                    success=False,
                    data={},
                    error=result["error"],
                    execution_time=time.time() - start_time
                )
            
            # Parse DTrace output based on script type
            parsed_data = self._parse_dtrace_output(result["output"], script_name)
            
            execution_time = time.time() - start_time
            
            return MCPResult(
                tool_name=self.name,
                success=True,
                data=parsed_data,
                execution_time=execution_time,
                metadata={
                    "script_type": script_name,
                    "duration": duration,
                    "raw_output_lines": len(result["output"].split('\n'))
                }
            )
            
        except Exception as e:
            return MCPResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Error executing DTrace: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _execute_dtrace_script(self, script: str, timeout: int) -> Dict[str, Any]:
        """
        Execute a DTrace script with proper error handling.
        
        Args:
            script: DTrace script content
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with success status, output, and error information
        """
        try:
            # Run DTrace with the script
            process = subprocess.run(
                ["dtrace", "-n", script],
                capture_output=True,
                text=True,
                timeout=timeout + 5  # Add buffer for DTrace startup/shutdown
            )
            
            if process.returncode != 0:
                error_msg = process.stderr.strip() if process.stderr else "Unknown DTrace error"
                
                # Provide helpful error messages for common issues
                if "dtrace: failed to initialize dtrace: DTrace requires additional privileges" in error_msg:
                    error_msg = ("DTrace requires elevated privileges. "
                               "Run with sudo or enable DTrace in System Preferences > Security & Privacy")
                elif "dtrace: failed to compile script" in error_msg:
                    error_msg = f"DTrace script compilation failed: {error_msg}"
                
                return {
                    "success": False,
                    "output": "",
                    "error": error_msg
                }
            
            return {
                "success": True,
                "output": process.stdout,
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"DTrace execution timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to execute DTrace: {str(e)}"
            }
    
    def _parse_dtrace_output(self, output: str, script_type: str) -> Dict[str, Any]:
        """
        Parse DTrace output based on the script type.
        
        Args:
            output: Raw DTrace output
            script_type: Type of script that was executed
            
        Returns:
            Structured data extracted from DTrace output
        """
        lines = output.strip().split('\n')
        parsed_data = {
            "script_type": script_type,
            "raw_output": output,
            "summary": {},
            "details": []
        }
        
        if script_type == "syscall":
            parsed_data.update(self._parse_syscall_output(lines))
        elif script_type == "io":
            parsed_data.update(self._parse_io_output(lines))
        elif script_type == "proc":
            parsed_data.update(self._parse_proc_output(lines))
        elif script_type == "network":
            parsed_data.update(self._parse_network_output(lines))
        elif script_type == "memory":
            parsed_data.update(self._parse_memory_output(lines))
        else:
            # For custom scripts, provide basic parsing
            parsed_data["summary"]["total_lines"] = len(lines)
            parsed_data["details"] = [{"line": i+1, "content": line} 
                                    for i, line in enumerate(lines) if line.strip()]
        
        return parsed_data
    
    def _parse_syscall_output(self, lines: List[str]) -> Dict[str, Any]:
        """Parse syscall DTrace output."""
        syscalls = {}
        processes = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('dtrace:'):
                continue
            
            # Look for aggregation output format: process syscall count
            parts = line.split()
            if len(parts) >= 3:
                try:
                    process = parts[0]
                    syscall = parts[1]
                    count = int(parts[2])
                    
                    if process not in processes:
                        processes[process] = {"total_syscalls": 0, "syscalls": {}}
                    
                    processes[process]["syscalls"][syscall] = count
                    processes[process]["total_syscalls"] += count
                    
                    if syscall not in syscalls:
                        syscalls[syscall] = 0
                    syscalls[syscall] += count
                    
                except (ValueError, IndexError):
                    continue
        
        # Sort by frequency
        top_syscalls = sorted(syscalls.items(), key=lambda x: x[1], reverse=True)[:10]
        top_processes = sorted(processes.items(), key=lambda x: x[1]["total_syscalls"], reverse=True)[:10]
        
        return {
            "summary": {
                "total_syscalls": sum(syscalls.values()),
                "unique_syscalls": len(syscalls),
                "active_processes": len(processes)
            },
            "top_syscalls": [{"syscall": name, "count": count} for name, count in top_syscalls],
            "top_processes": [{"process": name, "data": data} for name, data in top_processes]
        }
    
    def _parse_io_output(self, lines: List[str]) -> Dict[str, Any]:
        """Parse I/O DTrace output."""
        io_by_process = {}
        io_by_device = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('dtrace:'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    name = parts[0]
                    count = int(parts[1])
                    
                    # Heuristic: if it looks like a device name, it's a device
                    if name.startswith('disk') or '/' in name:
                        io_by_device[name] = count
                    else:
                        io_by_process[name] = count
                        
                except (ValueError, IndexError):
                    continue
        
        return {
            "summary": {
                "total_io_operations": sum(io_by_process.values()) + sum(io_by_device.values()),
                "active_processes": len(io_by_process),
                "active_devices": len(io_by_device)
            },
            "io_by_process": sorted(io_by_process.items(), key=lambda x: x[1], reverse=True)[:10],
            "io_by_device": sorted(io_by_device.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _parse_proc_output(self, lines: List[str]) -> Dict[str, Any]:
        """Parse process DTrace output."""
        execs = {}
        exits = {}
        events = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('dtrace:'):
                continue
            
            # Look for timestamped events
            if ':' in line and 'executed' in line:
                events.append({"type": "exec", "event": line})
            
            # Look for aggregation data
            parts = line.split()
            if len(parts) >= 2:
                try:
                    process = parts[0]
                    count = int(parts[1])
                    
                    if 'exec' in line.lower():
                        execs[process] = count
                    elif 'exit' in line.lower():
                        exits[process] = count
                        
                except (ValueError, IndexError):
                    continue
        
        return {
            "summary": {
                "total_execs": sum(execs.values()),
                "total_exits": sum(exits.values()),
                "events_captured": len(events)
            },
            "top_execs": sorted(execs.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_exits": sorted(exits.items(), key=lambda x: x[1], reverse=True)[:10],
            "recent_events": events[-10:]  # Last 10 events
        }
    
    def _parse_network_output(self, lines: List[str]) -> Dict[str, Any]:
        """Parse network DTrace output."""
        network_activity = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('dtrace:'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    function = parts[0]
                    process = parts[1]
                    count = int(parts[2])
                    
                    key = f"{function}:{process}"
                    network_activity[key] = count
                    
                except (ValueError, IndexError):
                    continue
        
        # Aggregate by function type
        tcp_activity = sum(count for key, count in network_activity.items() if 'tcp' in key.lower())
        udp_activity = sum(count for key, count in network_activity.items() if 'udp' in key.lower())
        
        return {
            "summary": {
                "total_network_events": sum(network_activity.values()),
                "tcp_events": tcp_activity,
                "udp_events": udp_activity,
                "active_connections": len(network_activity)
            },
            "top_activity": sorted(network_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _parse_memory_output(self, lines: List[str]) -> Dict[str, Any]:
        """Parse memory/VM DTrace output."""
        vm_stats = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('dtrace:'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stat_name = parts[0]
                    value = int(parts[1])
                    vm_stats[stat_name] = value
                    
                except (ValueError, IndexError):
                    continue
        
        return {
            "summary": {
                "total_vm_events": sum(vm_stats.values()),
                "stat_types": len(vm_stats)
            },
            "vm_statistics": vm_stats
        }