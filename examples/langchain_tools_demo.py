#!/usr/bin/env python3
"""
Demo script showing how to use Mac Doctor's LangChain tools integration.

This script demonstrates:
1. Creating LangChain-compatible tools from MCP tools
2. Using the tool factory to create tools from a registry
3. Using the convenience functions for direct analysis
"""

import json
from mac_doctor.langchain_tools import (
    MCPToolFactory,
    create_process_analysis_tool,
    create_disk_analysis_tool,
    analyze_processes,
    analyze_disk_usage,
)
from mac_doctor.tool_registry import ToolRegistry
from mac_doctor.mcps.process_mcp import ProcessMCP
from mac_doctor.mcps.disk_mcp import DiskMCP


def demo_langchain_tool_wrapper():
    """Demonstrate LangChain tool wrapper functionality."""
    print("=== LangChain Tool Wrapper Demo ===")
    
    # Create a LangChain tool from a ProcessMCP
    process_tool = create_process_analysis_tool()
    
    print(f"Tool name: {process_tool.name}")
    print(f"Tool description: {process_tool.description}")
    print(f"Tool available: {process_tool.mcp_tool.is_available()}")
    
    if process_tool.mcp_tool.is_available():
        # Use the tool with LangChain's invoke method
        result = process_tool.invoke({"top_n": 5, "sort_by": "cpu"})
        print("Tool result (first 200 chars):")
        print(result[:200] + "..." if len(result) > 200 else result)
    else:
        print("Process tool not available (psutil not installed)")
    
    print()


def demo_tool_factory():
    """Demonstrate the MCPToolFactory functionality."""
    print("=== Tool Factory Demo ===")
    
    # Create a tool registry and register some tools
    registry = ToolRegistry()
    registry.register_tool(ProcessMCP())
    registry.register_tool(DiskMCP())
    
    # Create LangChain tools from the registry
    langchain_tools = MCPToolFactory.create_tools_from_registry(registry)
    
    print(f"Created {len(langchain_tools)} LangChain tools:")
    for tool in langchain_tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Available: {tool.mcp_tool.is_available()}")
    
    print()


def demo_convenience_functions():
    """Demonstrate the convenience functions using @tool decorator."""
    print("=== Convenience Functions Demo ===")
    
    # These are LangChain tools created with @tool decorator
    print(f"analyze_processes tool: {analyze_processes.name}")
    print(f"analyze_disk_usage tool: {analyze_disk_usage.name}")
    
    # Use the tools (they return structured JSON)
    if ProcessMCP().is_available():
        print("\nRunning process analysis...")
        result = analyze_processes.invoke({"top_n": 3, "sort_by": "memory"})
        
        # Parse and display the result
        try:
            data = json.loads(result)
            if "system_overview" in data:
                overview = data["system_overview"]
                print(f"System CPU: {overview['cpu_usage_percent']:.1f}%")
                print(f"System Memory: {overview['memory_usage_percent']:.1f}%")
                print(f"Total Processes: {overview['total_processes']}")
                
                if "top_processes" in data:
                    print("\nTop processes:")
                    for proc in data["top_processes"]:
                        print(f"  {proc['name']} (PID {proc['pid']}): "
                              f"{proc['cpu_percent']:.1f}% CPU, {proc['memory_mb']:.1f}MB RAM")
        except json.JSONDecodeError:
            print("Result:", result)
    else:
        print("Process analysis not available (psutil not installed)")
    
    print()


def demo_tool_schemas():
    """Demonstrate tool input schemas."""
    print("=== Tool Schemas Demo ===")
    
    # Create tools and show their schemas
    process_tool = create_process_analysis_tool()
    disk_tool = create_disk_analysis_tool()
    
    print("Process tool schema:")
    print(json.dumps(process_tool.args_schema.model_json_schema(), indent=2))
    
    print("\nDisk tool schema:")
    print(json.dumps(disk_tool.args_schema.model_json_schema(), indent=2))


if __name__ == "__main__":
    print("Mac Doctor LangChain Tools Integration Demo")
    print("=" * 50)
    print()
    
    demo_langchain_tool_wrapper()
    demo_tool_factory()
    demo_convenience_functions()
    demo_tool_schemas()
    
    print("Demo completed!")