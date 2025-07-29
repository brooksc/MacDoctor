"""
LangChain Tools Integration - Convert MCP tools to LangChain Tool format.

This module provides LangChain-compatible tool wrappers for Mac Collector Plugins,
enabling them to be used with LangChain agents for dynamic tool selection and execution.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .interfaces import BaseMCP, MCPResult
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ProcessToolInput(BaseModel):
    """Input schema for Process MCP tool."""
    
    top_n: int = Field(
        default=10,
        description="Number of top processes to return",
        ge=1,
        le=100
    )
    sort_by: str = Field(
        default="cpu",
        description="Sort processes by this metric",
        pattern="^(cpu|memory|pid)$"
    )


class DiskToolInput(BaseModel):
    """Input schema for Disk MCP tool."""
    
    include_du: bool = Field(
        default=False,
        description="Whether to include disk usage analysis with du command"
    )
    du_paths: List[str] = Field(
        default=["/Applications"],
        description="Paths to analyze with du command"
    )
    du_timeout: int = Field(
        default=10,
        description="Timeout in seconds for du operations",
        ge=5,
        le=60
    )


class NetworkToolInput(BaseModel):
    """Input schema for Network MCP tool."""
    
    duration: int = Field(
        default=5,
        description="Duration in seconds to monitor network activity",
        ge=1,
        le=60
    )
    include_connections: bool = Field(
        default=True,
        description="Whether to include active network connections"
    )


class LogsToolInput(BaseModel):
    """Input schema for Logs MCP tool."""
    
    hours: int = Field(
        default=1,
        description="Number of hours of logs to analyze",
        ge=1,
        le=24
    )
    severity: str = Field(
        default="error",
        description="Minimum log severity level to include",
        pattern="^(debug|info|notice|warning|error|critical|alert|emergency)$"
    )


class VMStatToolInput(BaseModel):
    """Input schema for VM Stats MCP tool."""
    
    interval: int = Field(
        default=1,
        description="Interval in seconds between measurements",
        ge=1,
        le=10
    )
    count: int = Field(
        default=5,
        description="Number of measurements to take",
        ge=1,
        le=20
    )


class DTraceToolInput(BaseModel):
    """Input schema for DTrace MCP tool."""
    
    script_type: str = Field(
        default="syscall",
        description="Type of DTrace script to run",
        pattern="^(syscall|io|network|process)$"
    )
    duration: int = Field(
        default=10,
        description="Duration in seconds to run DTrace",
        ge=5,
        le=60
    )


class FlexibleToolInput(BaseModel):
    """Flexible input schema that accepts any parameters."""
    class Config:
        extra = "allow"  # Allow any additional fields


class LangChainMCPTool(BaseTool):
    """Base class for LangChain-compatible MCP tools."""
    
    name: str
    description: str
    args_schema: Type[BaseModel] = FlexibleToolInput
    mcp_tool: BaseMCP = Field(exclude=True)  # Exclude from serialization
    original_schema: Type[BaseModel] = Field(exclude=True)
    
    def __init__(self, mcp_tool: BaseMCP, args_schema: Type[BaseModel], **kwargs):
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description,
            args_schema=FlexibleToolInput,  # Use flexible schema for LangChain
            mcp_tool=mcp_tool,
            original_schema=args_schema,  # Store original for validation
            **kwargs
        )
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool and return formatted results."""
        # Debug logging to see what parameters we're getting
        logger.debug(f"Tool {self.name} received parameters: {kwargs}")
        
        # Handle different input formats from LangChain agents
        input_args = {}
        
        # Check if any parameter value is a JSON string that needs parsing
        for key, value in kwargs.items():
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                try:
                    # Try to parse as JSON - if successful, use the parsed data
                    parsed_json = json.loads(value)
                    if isinstance(parsed_json, dict):
                        input_args.update(parsed_json)
                        logger.debug(f"Successfully parsed JSON from {key}: {parsed_json}")
                        continue
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from {key}: {e}")
                    pass
            # If not JSON or parsing failed, use the parameter as-is
            input_args[key] = value

        # If no parameters were provided, use empty dict to get defaults
        if not input_args:
            input_args = {}
            
        logger.debug(f"Final input_args for {self.name}: {input_args}")

        # Validate input using the original schema
        try:
            validated_input = self.original_schema(**input_args)
            input_dict = validated_input.model_dump()
        except Exception as e:
            # If validation fails, try with just defaults
            try:
                validated_input = self.original_schema()
                input_dict = validated_input.model_dump()
                logger.warning(f"Tool {self.name} validation failed: {e}, using defaults")
            except Exception as e2:
                return f"Error validating input: {str(e)}"
        
        # Check if tool is available
        if not self.mcp_tool.is_available():
            return f"Tool '{self.mcp_tool.name}' is not available on this system"
        
        # Execute the MCP tool
        result = self.mcp_tool.execute(**input_dict)
        
        # Format the result for LangChain consumption
        return self._format_result(result)
    
    def _format_result(self, result: MCPResult) -> str:
        """Format MCPResult for LangChain agent consumption."""
        if not result.success:
            return f"Tool execution failed: {result.error}"
        
        # Create a structured summary for the agent
        formatted_result = {
            "tool_name": result.tool_name,
            "success": result.success,
            "execution_time": f"{result.execution_time:.2f}s",
            "summary": self._create_summary(result),
            "data": result.data
        }
        
        return json.dumps(formatted_result, indent=2, default=str)
    
    def _create_summary(self, result: MCPResult) -> str:
        """Create a human-readable summary of the tool results."""
        tool_name = result.tool_name
        
        if tool_name == "process":
            return self._summarize_process_data(result.data)
        elif tool_name == "disk":
            return self._summarize_disk_data(result.data)
        elif tool_name == "network":
            return self._summarize_network_data(result.data)
        elif tool_name == "logs":
            return self._summarize_logs_data(result.data)
        elif tool_name == "vmstat":
            return self._summarize_vmstat_data(result.data)
        elif tool_name == "dtrace":
            return self._summarize_dtrace_data(result.data)
        else:
            return f"Data collected from {tool_name} tool"
    
    def _summarize_process_data(self, data: Dict[str, Any]) -> str:
        """Create summary for process data."""
        if "system_stats" in data and "top_processes" in data:
            sys_stats = data["system_stats"]
            top_procs = data["top_processes"]
            
            summary = f"System: {sys_stats.get('cpu_percent', 0):.1f}% CPU, "
            summary += f"{sys_stats.get('memory_percent', 0):.1f}% memory used. "
            
            if top_procs:
                top_proc = top_procs[0]
                summary += f"Top process: {top_proc.get('name', 'Unknown')} "
                summary += f"({top_proc.get('cpu_percent', 0):.1f}% CPU, "
                summary += f"{top_proc.get('memory_mb', 0):.1f}MB RAM)"
            
            return summary
        return "Process monitoring data collected"
    
    def _summarize_disk_data(self, data: Dict[str, Any]) -> str:
        """Create summary for disk data."""
        summary_parts = []
        
        if "df" in data and "summary" in data["df"]:
            df_summary = data["df"]["summary"]
            summary_parts.append(
                f"Storage: {df_summary.get('total_used_gb', 0):.1f}GB used "
                f"of {df_summary.get('total_size_gb', 0):.1f}GB total "
                f"({df_summary.get('overall_capacity_percent', 0):.1f}% full)"
            )
        
        if "iostat" in data and "summary" in data["iostat"]:
            io_summary = data["iostat"]["summary"]
            summary_parts.append(
                f"I/O: {io_summary.get('total_ops_per_sec', 0):.1f} ops/sec, "
                f"{io_summary.get('total_throughput_mb_per_sec', 0):.1f} MB/sec"
            )
        
        return ". ".join(summary_parts) if summary_parts else "Disk monitoring data collected"
    
    def _summarize_network_data(self, data: Dict[str, Any]) -> str:
        """Create summary for network data."""
        # This will be implemented when NetworkMCP is available
        return "Network monitoring data collected"
    
    def _summarize_logs_data(self, data: Dict[str, Any]) -> str:
        """Create summary for logs data."""
        # This will be implemented when LogsMCP is available
        return "System logs data collected"
    
    def _summarize_vmstat_data(self, data: Dict[str, Any]) -> str:
        """Create summary for vmstat data."""
        # This will be implemented when VMStatMCP is available
        return "Virtual memory statistics collected"
    
    def _summarize_dtrace_data(self, data: Dict[str, Any]) -> str:
        """Create summary for dtrace data."""
        # This will be implemented when DTraceMCP is available
        return "DTrace system analysis data collected"


class MCPToolFactory:
    """Factory for creating LangChain-compatible MCP tools."""
    
    # Mapping of MCP tool names to their input schemas
    TOOL_SCHEMAS = {
        "process": ProcessToolInput,
        "disk": DiskToolInput,
        "network": NetworkToolInput,
        "logs": LogsToolInput,
        "vmstat": VMStatToolInput,
        "dtrace": DTraceToolInput,
    }
    
    @classmethod
    def create_langchain_tool(cls, mcp_tool: BaseMCP) -> LangChainMCPTool:
        """Create a LangChain-compatible tool from an MCP tool."""
        schema = cls.TOOL_SCHEMAS.get(mcp_tool.name)
        if not schema:
            # Create a generic schema for unknown tools
            schema = BaseModel
        
        return LangChainMCPTool(mcp_tool=mcp_tool, args_schema=schema)
    
    @classmethod
    def create_tools_from_registry(cls, registry: ToolRegistry) -> List[LangChainMCPTool]:
        """Create LangChain tools for all available MCP tools in the registry."""
        langchain_tools = []
        
        for mcp_tool in registry.get_available_tools():
            langchain_tool = cls.create_langchain_tool(mcp_tool)
            langchain_tools.append(langchain_tool)
        
        return langchain_tools


# Convenience functions for creating LangChain tools
def create_process_analysis_tool() -> LangChainMCPTool:
    """Create a LangChain tool for process analysis."""
    from .mcps.process_mcp import ProcessMCP
    return LangChainMCPTool(mcp_tool=ProcessMCP(), args_schema=ProcessToolInput)


def create_disk_analysis_tool() -> LangChainMCPTool:
    """Create a LangChain tool for disk analysis."""
    from .mcps.disk_mcp import DiskMCP
    return LangChainMCPTool(mcp_tool=DiskMCP(), args_schema=DiskToolInput)


# Standalone functions using the @tool decorator for direct use
@tool
def analyze_processes(
    top_n: int = 10,
    sort_by: str = "cpu"
) -> str:
    """
    Analyze running processes and their resource usage.
    
    Args:
        top_n: Number of top processes to return (1-100)
        sort_by: Sort processes by 'cpu', 'memory', or 'pid'
    
    Returns:
        JSON string with process analysis results
    """
    from .mcps.process_mcp import ProcessMCP
    
    tool = ProcessMCP()
    if not tool.is_available():
        return "Process monitoring tool is not available"
    
    result = tool.execute(top_n=top_n, sort_by=sort_by)
    
    if not result.success:
        return f"Process analysis failed: {result.error}"
    
    # Create formatted summary
    data = result.data
    if "system_stats" in data and "top_processes" in data:
        sys_stats = data["system_stats"]
        top_procs = data["top_processes"]
        
        summary = {
            "system_overview": {
                "cpu_usage_percent": sys_stats.get("cpu_percent", 0),
                "memory_usage_percent": sys_stats.get("memory_percent", 0),
                "memory_used_gb": sys_stats.get("memory_used_gb", 0),
                "memory_total_gb": sys_stats.get("memory_total_gb", 0),
                "total_processes": data.get("process_count", 0)
            },
            "top_processes": [
                {
                    "name": proc.get("name", "Unknown"),
                    "pid": proc.get("pid", 0),
                    "cpu_percent": proc.get("cpu_percent", 0),
                    "memory_percent": proc.get("memory_percent", 0),
                    "memory_mb": proc.get("memory_mb", 0),
                    "command": proc.get("command", "")
                }
                for proc in top_procs[:5]  # Limit to top 5 for summary
            ]
        }
        
        return json.dumps(summary, indent=2)
    
    return json.dumps({"error": "Unexpected data format"})


@tool
def analyze_disk_usage(
    include_du: bool = True,
    du_paths: List[str] = None,
    du_timeout: int = 30
) -> str:
    """
    Analyze disk I/O statistics and space usage.
    
    Args:
        include_du: Whether to include disk usage analysis
        du_paths: Paths to analyze (defaults to common system paths)
        du_timeout: Timeout for disk usage operations (10-300 seconds)
    
    Returns:
        JSON string with disk analysis results
    """
    from .mcps.disk_mcp import DiskMCP
    
    if du_paths is None:
        du_paths = ["/", "/Users", "/Applications", "/System", "/var"]
    
    tool = DiskMCP()
    if not tool.is_available():
        return "Disk monitoring tool is not available"
    
    result = tool.execute(
        include_du=include_du,
        du_paths=du_paths,
        du_timeout=du_timeout
    )
    
    if not result.success:
        return f"Disk analysis failed: {result.error}"
    
    # Create formatted summary
    data = result.data
    summary = {"disk_analysis": {}}
    
    # Add filesystem summary
    if "df" in data and "summary" in data["df"]:
        df_summary = data["df"]["summary"]
        summary["disk_analysis"]["storage"] = {
            "total_size_gb": df_summary.get("total_size_gb", 0),
            "used_gb": df_summary.get("total_used_gb", 0),
            "available_gb": df_summary.get("total_available_gb", 0),
            "usage_percent": df_summary.get("overall_capacity_percent", 0),
            "filesystem_count": df_summary.get("filesystem_count", 0)
        }
    
    # Add I/O summary
    if "iostat" in data and "summary" in data["iostat"]:
        io_summary = data["iostat"]["summary"]
        summary["disk_analysis"]["io_performance"] = {
            "total_ops_per_sec": io_summary.get("total_ops_per_sec", 0),
            "reads_per_sec": io_summary.get("total_reads_per_sec", 0),
            "writes_per_sec": io_summary.get("total_writes_per_sec", 0),
            "throughput_mb_per_sec": io_summary.get("total_throughput_mb_per_sec", 0)
        }
    
    # Add disk usage summary
    if include_du and "du" in data and "summary" in data["du"]:
        du_summary = data["du"]["summary"]
        summary["disk_analysis"]["usage_analysis"] = {
            "paths_analyzed": du_summary.get("paths_analyzed", 0),
            "successful_analyses": du_summary.get("successful_analyses", 0),
            "total_analyzed_gb": du_summary.get("total_analyzed_gb", 0)
        }
        
        # Add top space consumers
        if "path_analysis" in data["du"]:
            top_consumers = [
                {
                    "path": item.get("path", ""),
                    "size_gb": item.get("size_gb", 0),
                    "size_human": item.get("size_human", "")
                }
                for item in data["du"]["path_analysis"][:5]
                if "error" not in item
            ]
            summary["disk_analysis"]["top_space_consumers"] = top_consumers
    
    return json.dumps(summary, indent=2)