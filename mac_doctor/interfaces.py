"""
Base interfaces for Mac Doctor components.

This module defines the core interfaces that all MCP tools and LLM providers
must implement to ensure consistent behavior across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MCPResult:
    """Result from executing a Mac Collector Plugin."""
    
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Issue:
    """Represents a detected system issue."""
    
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'cpu', 'memory', 'disk', 'network', 'thermal'
    title: str
    description: str
    affected_processes: List[str] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.affected_processes is None:
            self.affected_processes = []
        if self.metrics is None:
            self.metrics = {}


@dataclass
class Recommendation:
    """Represents an actionable recommendation."""
    
    title: str
    description: str
    action_type: str  # 'info', 'command', 'sudo_command'
    command: Optional[str] = None
    risk_level: str = 'low'  # 'low', 'medium', 'high'
    confirmation_required: bool = True


@dataclass
class DiagnosticResult:
    """Complete result from a diagnostic analysis."""
    
    query: str
    analysis: str
    issues_detected: List[Issue]
    tool_results: Dict[str, MCPResult]
    recommendations: List[Recommendation]
    execution_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseMCP(ABC):
    """Base class for all Mac Collector Plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this MCP tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this MCP tool does."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> MCPResult:
        """Execute the diagnostic tool and return results."""
        pass
    
    def is_available(self) -> bool:
        """Check if this tool is available on the current system."""
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the input schema for this tool."""
        return {}


class BaseLLM(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    def analyze_system_data(self, data: Dict[str, Any], query: str) -> str:
        """Analyze system diagnostic data and return analysis."""
        pass
    
    @abstractmethod
    def generate_recommendations(self, analysis: str) -> List[Recommendation]:
        """Generate actionable recommendations based on analysis."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM provider is available."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass


@dataclass
class SystemMetrics:
    """System-wide metrics collected from various tools."""
    
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_activity: Dict[str, float]
    thermal_state: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ActionResult:
    """Result from executing a system action."""
    
    success: bool
    output: str
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CLIConfig:
    """Configuration for CLI operations."""
    
    mode: str  # 'diagnose', 'ask', 'list-tools', 'trace'
    query: Optional[str] = None
    output_format: str = 'markdown'  # 'markdown', 'json'
    debug: bool = False
    export_path: Optional[str] = None
    llm_provider: str = 'gemini'  # 'ollama', 'gemini'
    llm_model: Optional[str] = None
    privacy_mode: bool = False  # True for local-only processing