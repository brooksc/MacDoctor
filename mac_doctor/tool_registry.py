"""
Tool Registry - Dynamic registry for Mac Collector Plugins.

This module provides a registry system for dynamically registering and
discovering MCP tools based on user queries and system capabilities.
"""

from typing import Dict, List, Optional

from .interfaces import BaseMCP


class ToolRegistry:
    """Registry for managing Mac Collector Plugins."""
    
    def __init__(self):
        self._tools: Dict[str, BaseMCP] = {}
    
    def register_tool(self, tool: BaseMCP) -> None:
        """Register a new MCP tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseMCP]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_available_tools(self) -> List[BaseMCP]:
        """Get all tools that are available on the current system."""
        return [tool for tool in self._tools.values() if tool.is_available()]
    
    def get_tools_for_query(self, query: str) -> List[BaseMCP]:
        """Get tools that might be relevant for a specific query."""
        # For now, return all available tools
        # This will be enhanced with more sophisticated matching later
        return self.get_available_tools()
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
    
    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools