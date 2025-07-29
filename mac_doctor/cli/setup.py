"""
Setup utilities for CLI components.

This module provides functions to initialize and configure the diagnostic
components needed by the CLI interface.
"""

import logging
from typing import Tuple

from rich.console import Console

from ..agent.diagnostic_agent import DiagnosticAgent
from ..core.report_generator import ReportGenerator
from ..interfaces import CLIConfig
from ..llm.factory import LLMFactory
from ..config import ConfigManager
from ..tool_registry import ToolRegistry
from ..system_validator import SystemValidator
from ..mcps.disk_mcp import DiskMCP
from ..mcps.dtrace_mcp import DTraceMCP
from ..mcps.logs_mcp import LogsMCP
from ..mcps.network_mcp import NetworkMCP
from ..mcps.process_mcp import ProcessMCP
from ..mcps.vmstat_mcp import VMStatMCP

logger = logging.getLogger(__name__)
console = Console()


def setup_tools_and_agent(config: CLIConfig) -> Tuple[DiagnosticAgent, ReportGenerator]:
    """Setup and initialize all diagnostic components.
    
    Args:
        config: CLI configuration
        
    Returns:
        Tuple of (DiagnosticAgent, ReportGenerator)
        
    Raises:
        RuntimeError: If setup fails
    """
    try:
        # Validate system compatibility first
        validator = SystemValidator()
        system_info = validator.validate_system()
        
        if not system_info.is_compatible:
            console.print("[red]❌ System compatibility check failed[/red]")
            validator.display_system_status(show_details=True)
            
            guidance = validator.get_installation_guidance()
            if guidance:
                console.print("\n[yellow]Installation Guidance:[/yellow]")
                for line in guidance:
                    console.print(line)
            
            raise RuntimeError("System does not meet compatibility requirements")
        
        # Show warnings if any
        if system_info.warnings:
            console.print("[yellow]⚠️  System warnings detected:[/yellow]")
            for warning in system_info.warnings:
                console.print(f"  • {warning}")
            console.print()
        
        # Setup LLM
        llm = setup_llm(config)
        
        # Setup tool registry with system validation
        tool_registry = setup_tool_registry(validator)
        
        # Create diagnostic agent
        agent = DiagnosticAgent(llm, tool_registry)
        
        if not agent.is_available():
            raise RuntimeError("Diagnostic agent is not available")
        
        # Create report generator
        report_generator = ReportGenerator()
        
        logger.info("Successfully initialized all diagnostic components")
        return agent, report_generator
        
    except Exception as e:
        logger.error(f"Failed to setup diagnostic components: {e}")
        raise RuntimeError(f"Component setup failed: {e}")


def setup_llm(config: CLIConfig):
    """Setup and configure the LLM provider.
    
    Args:
        config: CLI configuration
        
    Returns:
        Configured LLM instance
        
    Raises:
        RuntimeError: If LLM setup fails
    """
    try:
        # Load configuration
        config_manager = ConfigManager()
        mac_doctor_config = config_manager.load_config()
        
        # Override with CLI parameters
        if config.llm_provider:
            mac_doctor_config.default_llm_provider = config.llm_provider
        
        if config.llm_model:
            provider_config = mac_doctor_config.get_provider_config(mac_doctor_config.default_llm_provider)
            if provider_config:
                provider_config.model = config.llm_model
        
        if config.privacy_mode:
            mac_doctor_config.privacy_mode = True
            # Force local provider in privacy mode
            if mac_doctor_config.default_llm_provider != "ollama":
                console.print("[yellow]⚠️  Privacy mode enabled, switching to Ollama[/yellow]")
                mac_doctor_config.default_llm_provider = "ollama"
        
        # Create LLM factory and instance
        factory = LLMFactory(mac_doctor_config)
        
        # Check provider availability
        availability = factory.get_available_providers()
        if not availability.get(mac_doctor_config.default_llm_provider, False):
            console.print(f"[yellow]⚠️  Provider '{mac_doctor_config.default_llm_provider}' not available, trying fallback...[/yellow]")
            
            # Try to auto-configure
            if not factory.auto_configure():
                raise RuntimeError("No LLM providers are available")
        
        # Create LLM instance
        llm = factory.create_llm()
        
        if not llm.is_available():
            raise RuntimeError(f"LLM provider '{llm.provider_name}' is not available")
        
        console.print(f"[green]✅ Using LLM: {llm.provider_name} ({llm.model_name})[/green]")
        return llm
        
    except Exception as e:
        logger.error(f"LLM setup failed: {e}")
        raise RuntimeError(f"LLM setup failed: {e}")


def setup_tool_registry(validator: SystemValidator = None) -> ToolRegistry:
    """Setup and populate the tool registry with available MCPs.
    
    Args:
        validator: Optional SystemValidator instance for tool availability checking
    
    Returns:
        Configured ToolRegistry instance
    """
    registry = ToolRegistry()
    
    # Register all available MCP tools
    tools = [
        ProcessMCP(),
        VMStatMCP(),
        DiskMCP(),
        NetworkMCP(),
        LogsMCP(),
        DTraceMCP(),
    ]
    
    available_count = 0
    for tool in tools:
        try:
            registry.register_tool(tool)
            if tool.is_available():
                available_count += 1
                logger.debug(f"Registered available tool: {tool.name}")
            else:
                logger.debug(f"Registered unavailable tool: {tool.name}")
                
                # If we have validator info, show more details about why tool is unavailable
                if validator and hasattr(validator, 'tool_availability'):
                    tool_info = validator.tool_availability.get(tool.name.lower())
                    if tool_info and tool_info.error:
                        logger.debug(f"Tool {tool.name} unavailable: {tool_info.error}")
                        
        except Exception as e:
            logger.warning(f"Failed to register tool {tool.__class__.__name__}: {e}")
    
    logger.info(f"Registered {len(registry)} tools, {available_count} available")
    
    if available_count == 0:
        logger.warning("No diagnostic tools are available")
        if validator:
            missing_tools = validator.get_missing_tools()
            if missing_tools:
                logger.error(f"Missing required system tools: {', '.join(missing_tools)}")
    
    return registry


def validate_system_requirements() -> bool:
    """Validate that the system meets requirements for Mac Doctor.
    
    Returns:
        True if system requirements are met, False otherwise
    """
    validator = SystemValidator()
    system_info = validator.validate_system()
    
    if not system_info.is_compatible:
        validator.display_system_status(show_details=True)
        
        guidance = validator.get_installation_guidance()
        if guidance:
            console.print("\n[yellow]Installation Guidance:[/yellow]")
            for line in guidance:
                console.print(line)
        
        return False
    
    # Show warnings if any
    if system_info.warnings:
        for warning in system_info.warnings:
            console.print(f"[yellow]⚠️  {warning}[/yellow]")
    
    return True


def check_dependencies() -> bool:
    """Check if required dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    validator = SystemValidator()
    
    # Just check Python dependencies without full system validation
    missing_deps = []
    for package in validator.REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        console.print("[red]❌ Missing required dependencies:[/red]")
        for dep in missing_deps:
            console.print(f"  - {dep}")
        console.print("\n[yellow]Install with: pip install -e .[/yellow]")
        return False
    
    return True


def show_system_info() -> None:
    """Display system information for debugging."""
    validator = SystemValidator()
    system_info = validator.validate_system()
    validator.display_system_status(show_details=True)