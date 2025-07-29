"""
Process MCP - Mac Collector Plugin for process monitoring.

This module provides process monitoring capabilities using psutil to gather
CPU and memory usage data for running processes.
"""

import time
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..interfaces import BaseMCP, MCPResult
from ..error_handling import (
    ErrorHandler, DependencyError, MacDoctorPermissionError, TimeoutError,
    safe_execute, create_safe_mcp_result
)
from ..logging_config import trace_execution, debug_context, log_performance


class ProcessMCP(BaseMCP):
    """Mac Collector Plugin for process monitoring using psutil."""
    
    @property
    def name(self) -> str:
        """Return the name of this MCP tool."""
        return "process"
    
    @property
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        return "Monitors running processes and their CPU/memory usage using psutil"
    
    def is_available(self) -> bool:
        """Check if psutil is available on the current system."""
        return PSUTIL_AVAILABLE
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the input schema for this tool."""
        return {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top processes to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort processes by this metric",
                    "enum": ["cpu", "memory", "pid"],
                    "default": "cpu"
                }
            }
        }
    
    @trace_execution(include_args=True, include_return=False)
    def execute(self, top_n: int = 10, sort_by: str = "cpu", **kwargs) -> MCPResult:
        """
        Execute process monitoring and return top processes by CPU/memory usage.
        
        Args:
            top_n: Number of top processes to return (default: 10)
            sort_by: Sort processes by 'cpu', 'memory', or 'pid' (default: 'cpu')
            
        Returns:
            MCPResult with process data or error information
        """
        start_time = time.time()
        error_handler = ErrorHandler()
        
        if not self.is_available():
            error = DependencyError(
                dependency="psutil",
                message="psutil is not available",
                suggestions=[
                    "Install psutil with: pip install psutil",
                    "Verify Python environment is correct",
                    "Check if psutil is in requirements.txt"
                ]
            )
            return create_safe_mcp_result(self.name, error, time.time() - start_time, error_handler)
        
        def _execute_logic():
            with debug_context("process_monitoring", {"top_n": top_n, "sort_by": sort_by}):
                # Validate parameters
                if not isinstance(top_n, int) or top_n < 1 or top_n > 100:
                    raise ValueError("top_n must be an integer between 1 and 100")
                
                if sort_by not in ["cpu", "memory", "pid"]:
                    raise ValueError("sort_by must be one of: cpu, memory, pid")
                
                # Get all running processes with error handling
                processes = []
                access_denied_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 
                                           'memory_info', 'status', 'create_time', 'cmdline']):
                try:
                    # Get process info
                    pinfo = proc.info
                    
                    # Skip processes with no CPU or memory data
                    if pinfo['cpu_percent'] is None:
                        pinfo['cpu_percent'] = 0.0
                    if pinfo['memory_percent'] is None:
                        pinfo['memory_percent'] = 0.0
                    
                    # Add memory info in MB
                    if pinfo['memory_info']:
                        pinfo['memory_mb'] = round(pinfo['memory_info'].rss / 1024 / 1024, 2)
                    else:
                        pinfo['memory_mb'] = 0.0
                    
                    # Clean up cmdline for readability
                    if pinfo['cmdline']:
                        pinfo['command'] = ' '.join(pinfo['cmdline'][:3])  # First 3 args
                    else:
                        pinfo['command'] = pinfo['name'] or 'Unknown'
                    
                    processes.append(pinfo)
                    
                except psutil.AccessDenied:
                    access_denied_count += 1
                    continue
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    # Skip processes that we can't access or that have disappeared
                    continue
                except Exception as e:
                    # Log unexpected errors but continue
                    error_handler.handle_error(
                        e,
                        context={"operation": "process_iteration", "pid": getattr(proc, 'pid', 'unknown')},
                        show_user_message=False
                    )
                    continue
            
            # Sort processes based on the specified criteria
            if sort_by == "cpu":
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            elif sort_by == "memory":
                processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            elif sort_by == "pid":
                processes.sort(key=lambda x: x['pid'])
            
            # Get top N processes
            top_processes = processes[:top_n]
            
            # Get system-wide statistics with error handling
            try:
                cpu_count = psutil.cpu_count()
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
            except Exception as e:
                raise MacDoctorPermissionError(
                    operation="system statistics collection",
                    suggestions=[
                        "Check system permissions",
                        "Verify psutil has access to system resources",
                        "Try running with elevated privileges if needed"
                    ]
                )
            
            # Prepare structured output
            result_data = {
                "system_overview": {
                    "cpu_count": cpu_count,
                    "cpu_usage_percent": cpu_percent,
                    "memory_total_gb": round(memory.total / 1024 / 1024 / 1024, 2),
                    "memory_used_gb": round(memory.used / 1024 / 1024 / 1024, 2),
                    "memory_usage_percent": memory.percent,
                    "memory_available_gb": round(memory.available / 1024 / 1024 / 1024, 2)
                },
                "process_summary": {
                    "total_processes": len(processes),
                    "access_denied_count": access_denied_count,
                    "top_n_requested": top_n,
                    "sort_criteria": sort_by
                },
                "top_processes": []
            }
            
            # Format top processes for output
            for proc in top_processes:
                result_data["top_processes"].append({
                    "pid": proc['pid'],
                    "name": proc['name'],
                    "command": proc['command'],
                    "cpu_percent": proc['cpu_percent'],
                    "memory_percent": round(proc['memory_percent'], 2),
                    "memory_mb": proc['memory_mb'],
                    "status": proc['status'],
                    "create_time": proc['create_time']
                })
            
            execution_time = time.time() - start_time
            
            return MCPResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    "sort_by": sort_by,
                    "top_n": top_n,
                    "total_processes": len(processes),
                    "access_denied_count": access_denied_count
                }
            )
        
        # Use safe execution with comprehensive error handling
        result = safe_execute(
            _execute_logic,
            error_handler=error_handler,
            context={"tool": self.name, "top_n": top_n, "sort_by": sort_by},
            fallback_result=None,
            show_errors=True
        )
        
        if result is not None:
            return result
        else:
            # Create fallback result
            return create_safe_mcp_result(
                self.name,
                Exception("Process monitoring failed with unknown error"),
                time.time() - start_time,
                error_handler
            )