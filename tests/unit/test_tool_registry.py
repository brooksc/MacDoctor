"""Unit tests for the ToolRegistry class."""

import pytest
from unittest.mock import Mock

from mac_doctor.interfaces import BaseMCP, MCPResult
from mac_doctor.tool_registry import ToolRegistry


class MockMCPTool(BaseMCP):
    """Mock MCP tool for testing."""
    
    def __init__(self, name: str, description: str, available: bool = True):
        self._name = name
        self._description = description
        self._available = available
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def execute(self, **kwargs) -> MCPResult:
        return MCPResult(
            tool_name=self.name,
            success=True,
            data={"test": "data"},
            execution_time=0.1
        )
    
    def is_available(self) -> bool:
        return self._available


class TestToolRegistry:
    """Test cases for ToolRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.mock_tool1 = MockMCPTool("test_tool_1", "Test tool 1", available=True)
        self.mock_tool2 = MockMCPTool("test_tool_2", "Test tool 2", available=True)
        self.mock_tool_unavailable = MockMCPTool("unavailable_tool", "Unavailable tool", available=False)
    
    def test_init(self):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.list_tools() == []
    
    def test_register_tool(self):
        """Test registering a single tool."""
        self.registry.register_tool(self.mock_tool1)
        
        assert len(self.registry) == 1
        assert "test_tool_1" in self.registry
        assert self.registry.list_tools() == ["test_tool_1"]
    
    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool2)
        
        assert len(self.registry) == 2
        assert "test_tool_1" in self.registry
        assert "test_tool_2" in self.registry
        assert set(self.registry.list_tools()) == {"test_tool_1", "test_tool_2"}
    
    def test_register_tool_overwrites_existing(self):
        """Test that registering a tool with the same name overwrites the existing one."""
        self.registry.register_tool(self.mock_tool1)
        
        # Create a new tool with the same name
        new_tool = MockMCPTool("test_tool_1", "Updated test tool 1")
        self.registry.register_tool(new_tool)
        
        assert len(self.registry) == 1
        retrieved_tool = self.registry.get_tool("test_tool_1")
        assert retrieved_tool.description == "Updated test tool 1"
    
    def test_get_tool_existing(self):
        """Test getting an existing tool."""
        self.registry.register_tool(self.mock_tool1)
        
        retrieved_tool = self.registry.get_tool("test_tool_1")
        assert retrieved_tool is self.mock_tool1
        assert retrieved_tool.name == "test_tool_1"
    
    def test_get_tool_nonexistent(self):
        """Test getting a non-existent tool returns None."""
        retrieved_tool = self.registry.get_tool("nonexistent_tool")
        assert retrieved_tool is None
    
    def test_list_tools_empty(self):
        """Test listing tools when registry is empty."""
        assert self.registry.list_tools() == []
    
    def test_list_tools_with_tools(self):
        """Test listing tools when registry has tools."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool2)
        
        tools = self.registry.list_tools()
        assert len(tools) == 2
        assert set(tools) == {"test_tool_1", "test_tool_2"}
    
    def test_get_available_tools_all_available(self):
        """Test getting available tools when all are available."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool2)
        
        available_tools = self.registry.get_available_tools()
        assert len(available_tools) == 2
        assert self.mock_tool1 in available_tools
        assert self.mock_tool2 in available_tools
    
    def test_get_available_tools_some_unavailable(self):
        """Test getting available tools when some are unavailable."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool_unavailable)
        
        available_tools = self.registry.get_available_tools()
        assert len(available_tools) == 1
        assert self.mock_tool1 in available_tools
        assert self.mock_tool_unavailable not in available_tools
    
    def test_get_available_tools_none_available(self):
        """Test getting available tools when none are available."""
        self.registry.register_tool(self.mock_tool_unavailable)
        
        available_tools = self.registry.get_available_tools()
        assert len(available_tools) == 0
    
    def test_get_tools_for_query(self):
        """Test getting tools for a specific query."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool2)
        self.registry.register_tool(self.mock_tool_unavailable)
        
        # Currently returns all available tools
        query_tools = self.registry.get_tools_for_query("test query")
        assert len(query_tools) == 2
        assert self.mock_tool1 in query_tools
        assert self.mock_tool2 in query_tools
        assert self.mock_tool_unavailable not in query_tools
    
    def test_unregister_tool_existing(self):
        """Test unregistering an existing tool."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool2)
        
        result = self.registry.unregister_tool("test_tool_1")
        assert result is True
        assert len(self.registry) == 1
        assert "test_tool_1" not in self.registry
        assert "test_tool_2" in self.registry
    
    def test_unregister_tool_nonexistent(self):
        """Test unregistering a non-existent tool."""
        self.registry.register_tool(self.mock_tool1)
        
        result = self.registry.unregister_tool("nonexistent_tool")
        assert result is False
        assert len(self.registry) == 1
        assert "test_tool_1" in self.registry
    
    def test_clear(self):
        """Test clearing all tools from the registry."""
        self.registry.register_tool(self.mock_tool1)
        self.registry.register_tool(self.mock_tool2)
        
        assert len(self.registry) == 2
        
        self.registry.clear()
        
        assert len(self.registry) == 0
        assert self.registry.list_tools() == []
        assert "test_tool_1" not in self.registry
        assert "test_tool_2" not in self.registry
    
    def test_len(self):
        """Test the __len__ method."""
        assert len(self.registry) == 0
        
        self.registry.register_tool(self.mock_tool1)
        assert len(self.registry) == 1
        
        self.registry.register_tool(self.mock_tool2)
        assert len(self.registry) == 2
        
        self.registry.unregister_tool("test_tool_1")
        assert len(self.registry) == 1
    
    def test_contains(self):
        """Test the __contains__ method."""
        assert "test_tool_1" not in self.registry
        
        self.registry.register_tool(self.mock_tool1)
        assert "test_tool_1" in self.registry
        assert "test_tool_2" not in self.registry
        
        self.registry.register_tool(self.mock_tool2)
        assert "test_tool_1" in self.registry
        assert "test_tool_2" in self.registry
    
    def test_registry_with_real_mcp_interface(self):
        """Test that the registry works with the actual BaseMCP interface."""
        # Create a mock that properly implements BaseMCP
        mock_tool = Mock(spec=BaseMCP)
        mock_tool.name = "mock_tool"
        mock_tool.description = "Mock tool for testing"
        mock_tool.is_available.return_value = True
        mock_tool.execute.return_value = MCPResult(
            tool_name="mock_tool",
            success=True,
            data={"result": "success"}
        )
        
        self.registry.register_tool(mock_tool)
        
        assert len(self.registry) == 1
        assert "mock_tool" in self.registry
        
        retrieved_tool = self.registry.get_tool("mock_tool")
        assert retrieved_tool is mock_tool
        
        available_tools = self.registry.get_available_tools()
        assert len(available_tools) == 1
        assert mock_tool in available_tools
        
        # Test that the tool can be executed
        result = retrieved_tool.execute()
        assert result.tool_name == "mock_tool"
        assert result.success is True
        assert result.data == {"result": "success"}


class TestBaseMCPInterface:
    """Test cases for BaseMCP interface compliance."""
    
    def test_mock_tool_implements_interface(self):
        """Test that MockMCPTool properly implements BaseMCP interface."""
        tool = MockMCPTool("test", "Test tool")
        
        # Test required properties
        assert tool.name == "test"
        assert tool.description == "Test tool"
        
        # Test required methods
        assert tool.is_available() is True
        
        result = tool.execute()
        assert isinstance(result, MCPResult)
        assert result.tool_name == "test"
        assert result.success is True
    
    def test_mock_tool_unavailable(self):
        """Test MockMCPTool when unavailable."""
        tool = MockMCPTool("test", "Test tool", available=False)
        assert tool.is_available() is False
    
    def test_mcp_result_creation(self):
        """Test MCPResult creation and default values."""
        result = MCPResult(
            tool_name="test_tool",
            success=True,
            data={"key": "value"}
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.execution_time == 0.0
        assert result.metadata == {}
    
    def test_mcp_result_with_all_fields(self):
        """Test MCPResult with all fields specified."""
        metadata = {"source": "test"}
        result = MCPResult(
            tool_name="test_tool",
            success=False,
            data={"key": "value"},
            error="Test error",
            execution_time=1.5,
            metadata=metadata
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is False
        assert result.data == {"key": "value"}
        assert result.error == "Test error"
        assert result.execution_time == 1.5
        assert result.metadata == metadata