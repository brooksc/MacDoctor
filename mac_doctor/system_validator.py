"""
System compatibility validation for Mac Doctor.

This module provides comprehensive system requirements checking, including
macOS version validation, architecture detection, system tool availability,
and dependency verification.
"""

import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class SystemInfo:
    """System information and compatibility status."""
    
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    macos_version_tuple: Optional[Tuple[int, int, int]] = None
    is_apple_silicon: bool = False
    is_compatible: bool = True
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class ToolAvailability:
    """Availability status of system diagnostic tools."""
    
    tool_name: str
    is_available: bool
    path: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    requires_sudo: bool = False
    alternatives: List[str] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class SystemValidator:
    """Validates system compatibility and tool availability."""
    
    # Minimum supported macOS version (12.0 - Monterey)
    MIN_MACOS_VERSION = (12, 0, 0)
    
    # Required system tools and their alternatives
    SYSTEM_TOOLS = {
        'ps': {
            'description': 'Process monitoring',
            'alternatives': ['top'],
            'required': True
        },
        'vm_stat': {
            'description': 'Virtual memory statistics',
            'alternatives': [],
            'required': True
        },
        'iostat': {
            'description': 'I/O statistics',
            'alternatives': ['sar'],
            'required': False
        },
        'df': {
            'description': 'Disk space usage',
            'alternatives': [],
            'required': True
        },
        'du': {
            'description': 'Directory usage',
            'alternatives': [],
            'required': True
        },
        'netstat': {
            'description': 'Network statistics',
            'alternatives': ['lsof'],
            'required': False
        },
        'nettop': {
            'description': 'Network activity monitoring',
            'alternatives': ['netstat'],
            'required': False
        },
        'log': {
            'description': 'System log access',
            'alternatives': ['syslog'],
            'required': True
        },
        'dtrace': {
            'description': 'Dynamic tracing',
            'alternatives': [],
            'required': False,
            'requires_sudo': True
        },
        'memory_pressure': {
            'description': 'Memory pressure monitoring',
            'alternatives': ['vm_stat'],
            'required': False
        },
        'sw_vers': {
            'description': 'System version information',
            'alternatives': [],
            'required': True
        }
    }
    
    # Required Python packages
    REQUIRED_PACKAGES = [
        'typer',
        'rich', 
        'psutil',
        'langchain',
        'langchain_ollama',
        'langchain_google_genai',
        'langgraph',
        'pydantic'
    ]
    
    def __init__(self):
        self.system_info: Optional[SystemInfo] = None
        self.tool_availability: Dict[str, ToolAvailability] = {}
        
    def validate_system(self) -> SystemInfo:
        """Perform comprehensive system validation.
        
        Returns:
            SystemInfo object with validation results
        """
        logger.info("Starting system compatibility validation")
        
        # Gather basic system information
        self.system_info = self._gather_system_info()
        
        # Validate operating system
        self._validate_operating_system()
        
        # Validate macOS version
        self._validate_macos_version()
        
        # Validate Python version
        self._validate_python_version()
        
        # Check architecture-specific considerations
        self._check_architecture()
        
        # Validate system tools
        self._validate_system_tools()
        
        # Check Python dependencies
        self._check_python_dependencies()
        
        logger.info(f"System validation complete. Compatible: {self.system_info.is_compatible}")
        return self.system_info
    
    def _gather_system_info(self) -> SystemInfo:
        """Gather basic system information."""
        try:
            os_name = platform.system()
            os_version = platform.release()
            architecture = platform.machine()
            python_version = f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
            
            # Detect Apple Silicon
            is_apple_silicon = architecture in ['arm64', 'aarch64']
            
            # Get macOS version tuple
            macos_version_tuple = None
            if os_name == "Darwin":
                try:
                    result = subprocess.run(
                        ["sw_vers", "-productVersion"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5
                    )
                    version_str = result.stdout.strip()
                    version_parts = version_str.split('.')
                    macos_version_tuple = tuple(
                        int(part) for part in version_parts[:3]
                    ) + (0,) * (3 - len(version_parts))
                except Exception as e:
                    logger.warning(f"Could not determine macOS version: {e}")
            
            return SystemInfo(
                os_name=os_name,
                os_version=os_version,
                architecture=architecture,
                python_version=python_version,
                macos_version_tuple=macos_version_tuple,
                is_apple_silicon=is_apple_silicon
            )
            
        except Exception as e:
            logger.error(f"Failed to gather system info: {e}")
            return SystemInfo(
                os_name="Unknown",
                os_version="Unknown", 
                architecture="Unknown",
                python_version="Unknown",
                is_compatible=False,
                errors=[f"Failed to gather system information: {e}"]
            )
    
    def _validate_operating_system(self):
        """Validate that we're running on macOS."""
        if self.system_info.os_name != "Darwin":
            self.system_info.is_compatible = False
            self.system_info.errors.append(
                f"Mac Doctor requires macOS, but detected {self.system_info.os_name}"
            )
            logger.error(f"Incompatible OS: {self.system_info.os_name}")
    
    def _validate_macos_version(self):
        """Validate macOS version compatibility."""
        if not self.system_info.macos_version_tuple:
            self.system_info.warnings.append(
                "Could not determine macOS version - some features may not work"
            )
            return
        
        current_version = self.system_info.macos_version_tuple
        min_version = self.MIN_MACOS_VERSION
        
        if current_version < min_version:
            self.system_info.is_compatible = False
            current_str = '.'.join(map(str, current_version))
            min_str = '.'.join(map(str, min_version))
            self.system_info.errors.append(
                f"macOS {current_str} is not supported. Minimum required: {min_str}"
            )
            logger.error(f"Incompatible macOS version: {current_str}")
        else:
            current_str = '.'.join(map(str, current_version))
            logger.info(f"macOS version {current_str} is compatible")
    
    def _validate_python_version(self):
        """Validate Python version compatibility."""
        if sys.version_info < (3, 9):
            self.system_info.is_compatible = False
            current_version = f"{sys.version_info[0]}.{sys.version_info[1]}"
            self.system_info.errors.append(
                f"Python {current_version} is not supported. Minimum required: 3.9"
            )
            logger.error(f"Incompatible Python version: {current_version}")
        else:
            logger.info(f"Python version {self.system_info.python_version} is compatible")
    
    def _check_architecture(self):
        """Check architecture-specific considerations."""
        if self.system_info.is_apple_silicon:
            logger.info("Apple Silicon detected - using ARM64 optimizations")
            # Add any Apple Silicon specific warnings or optimizations
            self.system_info.warnings.append(
                "Running on Apple Silicon - some diagnostic tools may behave differently"
            )
        else:
            logger.info("Intel architecture detected")
    
    def _validate_system_tools(self):
        """Validate availability of required system tools."""
        logger.info("Checking system tool availability")
        
        for tool_name, tool_info in self.SYSTEM_TOOLS.items():
            availability = self._check_tool_availability(tool_name, tool_info)
            self.tool_availability[tool_name] = availability
            
            if tool_info['required'] and not availability.is_available:
                if not availability.alternatives:
                    self.system_info.is_compatible = False
                    self.system_info.errors.append(
                        f"Required tool '{tool_name}' is not available: {availability.error}"
                    )
                else:
                    self.system_info.warnings.append(
                        f"Tool '{tool_name}' not available, will use alternatives: {', '.join(availability.alternatives)}"
                    )
            elif not availability.is_available:
                self.system_info.warnings.append(
                    f"Optional tool '{tool_name}' not available: {availability.error}"
                )
    
    def _check_tool_availability(self, tool_name: str, tool_info: Dict) -> ToolAvailability:
        """Check if a specific system tool is available."""
        try:
            # Try to find the tool
            result = subprocess.run(
                ["which", tool_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                tool_path = result.stdout.strip()
                
                # Try to get version if possible
                version = self._get_tool_version(tool_name)
                
                return ToolAvailability(
                    tool_name=tool_name,
                    is_available=True,
                    path=tool_path,
                    version=version,
                    requires_sudo=tool_info.get('requires_sudo', False),
                    alternatives=tool_info.get('alternatives', [])
                )
            else:
                return ToolAvailability(
                    tool_name=tool_name,
                    is_available=False,
                    error="Tool not found in PATH",
                    requires_sudo=tool_info.get('requires_sudo', False),
                    alternatives=tool_info.get('alternatives', [])
                )
                
        except subprocess.TimeoutExpired:
            return ToolAvailability(
                tool_name=tool_name,
                is_available=False,
                error="Tool check timed out",
                requires_sudo=tool_info.get('requires_sudo', False),
                alternatives=tool_info.get('alternatives', [])
            )
        except Exception as e:
            return ToolAvailability(
                tool_name=tool_name,
                is_available=False,
                error=f"Error checking tool: {e}",
                requires_sudo=tool_info.get('requires_sudo', False),
                alternatives=tool_info.get('alternatives', [])
            )
    
    def _get_tool_version(self, tool_name: str) -> Optional[str]:
        """Try to get version information for a tool."""
        version_flags = ['--version', '-V', '-v']
        
        for flag in version_flags:
            try:
                result = subprocess.run(
                    [tool_name, flag],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Return first line of version output
                    return result.stdout.strip().split('\n')[0]
            except Exception:
                continue
        
        return None
    
    def _check_python_dependencies(self):
        """Check if required Python packages are available."""
        logger.info("Checking Python dependencies")
        
        missing_packages = []
        for package in self.REQUIRED_PACKAGES:
            try:
                __import__(package)
                logger.debug(f"Package {package} is available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"Package {package} is missing")
        
        if missing_packages:
            self.system_info.is_compatible = False
            self.system_info.errors.append(
                f"Missing required Python packages: {', '.join(missing_packages)}"
            )
    
    def get_installation_guidance(self) -> List[str]:
        """Get installation guidance for missing dependencies."""
        guidance = []
        
        if not self.system_info:
            return ["Run system validation first"]
        
        # Check for missing Python packages
        missing_packages = []
        for package in self.REQUIRED_PACKAGES:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            guidance.append("Install missing Python packages:")
            guidance.append("  pip install -e .")
            guidance.append("  # or")
            guidance.append("  pip install -e .[dev]  # for development")
        
        # Check for missing system tools
        missing_tools = [
            name for name, availability in self.tool_availability.items()
            if self.SYSTEM_TOOLS[name]['required'] and not availability.is_available
        ]
        
        if missing_tools:
            guidance.append("\nMissing required system tools:")
            for tool in missing_tools:
                guidance.append(f"  - {tool}: {self.SYSTEM_TOOLS[tool]['description']}")
            guidance.append("\nThese tools should be available on macOS by default.")
            guidance.append("If missing, try updating macOS or installing Xcode Command Line Tools:")
            guidance.append("  xcode-select --install")
        
        # macOS version guidance
        if (self.system_info.macos_version_tuple and 
            self.system_info.macos_version_tuple < self.MIN_MACOS_VERSION):
            guidance.append(f"\nUpgrade macOS to version {'.'.join(map(str, self.MIN_MACOS_VERSION))} or later")
        
        # Python version guidance
        if sys.version_info < (3, 9):
            guidance.append("\nUpgrade Python to version 3.9 or later")
            guidance.append("Consider using pyenv or homebrew to manage Python versions")
        
        return guidance
    
    def display_system_status(self, show_details: bool = False):
        """Display system compatibility status to console."""
        if not self.system_info:
            console.print("[red]❌ System validation not performed[/red]")
            return
        
        console.print("[bold blue]🖥️  System Compatibility Status[/bold blue]")
        console.print()
        
        # Basic system info
        console.print(f"OS: {self.system_info.os_name} {self.system_info.os_version}")
        console.print(f"Architecture: {self.system_info.architecture}")
        if self.system_info.is_apple_silicon:
            console.print("  [dim]Apple Silicon detected[/dim]")
        console.print(f"Python: {self.system_info.python_version}")
        
        if self.system_info.macos_version_tuple:
            macos_version = '.'.join(map(str, self.system_info.macos_version_tuple))
            console.print(f"macOS: {macos_version}")
        
        console.print()
        
        # Compatibility status
        if self.system_info.is_compatible:
            console.print("[green]✅ System is compatible with Mac Doctor[/green]")
        else:
            console.print("[red]❌ System compatibility issues detected[/red]")
        
        # Show errors
        if self.system_info.errors:
            console.print("\n[red]Errors:[/red]")
            for error in self.system_info.errors:
                console.print(f"  • {error}")
        
        # Show warnings
        if self.system_info.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in self.system_info.warnings:
                console.print(f"  • {warning}")
        
        # Show tool availability if requested
        if show_details and self.tool_availability:
            console.print("\n[bold]System Tool Availability:[/bold]")
            for tool_name, availability in self.tool_availability.items():
                tool_info = self.SYSTEM_TOOLS[tool_name]
                status_icon = "✅" if availability.is_available else "❌"
                required_text = " (required)" if tool_info['required'] else " (optional)"
                
                console.print(f"  {status_icon} {tool_name}{required_text}")
                if availability.is_available and availability.path:
                    console.print(f"    [dim]Path: {availability.path}[/dim]")
                    if availability.version:
                        console.print(f"    [dim]Version: {availability.version}[/dim]")
                elif not availability.is_available:
                    console.print(f"    [dim]Error: {availability.error}[/dim]")
                    if availability.alternatives:
                        console.print(f"    [dim]Alternatives: {', '.join(availability.alternatives)}[/dim]")
        
        console.print()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available diagnostic tools."""
        return [
            name for name, availability in self.tool_availability.items()
            if availability.is_available
        ]
    
    def get_missing_tools(self) -> List[str]:
        """Get list of missing required tools."""
        return [
            name for name, availability in self.tool_availability.items()
            if self.SYSTEM_TOOLS[name]['required'] and not availability.is_available
        ]