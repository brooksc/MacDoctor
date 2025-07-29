"""
Unit tests for system compatibility validation.

Tests the SystemValidator class functionality including macOS version checking,
architecture detection, system tool availability, and dependency verification.
"""

import platform
import subprocess
import sys
from unittest.mock import Mock, patch, MagicMock

import pytest

from mac_doctor.system_validator import SystemValidator, SystemInfo, ToolAvailability


class TestSystemValidator:
    """Test cases for SystemValidator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = SystemValidator()
    
    def test_init(self):
        """Test SystemValidator initialization."""
        assert self.validator.system_info is None
        assert self.validator.tool_availability == {}
        assert self.validator.MIN_MACOS_VERSION == (12, 0, 0)
        assert len(self.validator.SYSTEM_TOOLS) > 0
        assert len(self.validator.REQUIRED_PACKAGES) > 0
    
    @patch('platform.system')
    @patch('platform.release')
    @patch('platform.machine')
    @patch('subprocess.run')
    def test_gather_system_info_macos_intel(self, mock_run, mock_machine, mock_release, mock_system):
        """Test gathering system info on Intel macOS."""
        # Mock system calls
        mock_system.return_value = "Darwin"
        mock_release.return_value = "21.6.0"
        mock_machine.return_value = "x86_64"
        
        # Mock sw_vers output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="13.5.2\n"
        )
        
        with patch('sys.version_info', (3, 11, 5)):
            system_info = self.validator._gather_system_info()
        
        assert system_info.os_name == "Darwin"
        assert system_info.os_version == "21.6.0"
        assert system_info.architecture == "x86_64"
        assert system_info.python_version == "3.11.5"
        assert system_info.macos_version_tuple == (13, 5, 2)
        assert not system_info.is_apple_silicon
        assert system_info.is_compatible  # Default value
    
    @patch('platform.system')
    @patch('platform.release')
    @patch('platform.machine')
    @patch('subprocess.run')
    def test_gather_system_info_macos_apple_silicon(self, mock_run, mock_machine, mock_release, mock_system):
        """Test gathering system info on Apple Silicon macOS."""
        # Mock system calls
        mock_system.return_value = "Darwin"
        mock_release.return_value = "22.1.0"
        mock_machine.return_value = "arm64"
        
        # Mock sw_vers output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="13.0.1\n"
        )
        
        with patch('sys.version_info', (3, 10, 8)):
            system_info = self.validator._gather_system_info()
        
        assert system_info.os_name == "Darwin"
        assert system_info.architecture == "arm64"
        assert system_info.is_apple_silicon
        assert system_info.macos_version_tuple == (13, 0, 1)
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_gather_system_info_sw_vers_failure(self, mock_run, mock_system):
        """Test handling sw_vers command failure."""
        mock_system.return_value = "Darwin"
        mock_run.side_effect = subprocess.CalledProcessError(1, "sw_vers")
        
        system_info = self.validator._gather_system_info()
        
        assert system_info.macos_version_tuple is None
    
    def test_validate_operating_system_macos(self):
        """Test OS validation on macOS."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        self.validator._validate_operating_system()
        
        assert self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 0
    
    def test_validate_operating_system_non_macos(self):
        """Test OS validation on non-macOS system."""
        self.validator.system_info = SystemInfo(
            os_name="Linux",
            os_version="5.4.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        self.validator._validate_operating_system()
        
        assert not self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 1
        assert "requires macOS" in self.validator.system_info.errors[0]
    
    def test_validate_macos_version_compatible(self):
        """Test macOS version validation with compatible version."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            macos_version_tuple=(13, 5, 2)
        )
        
        self.validator._validate_macos_version()
        
        assert self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 0
    
    def test_validate_macos_version_incompatible(self):
        """Test macOS version validation with incompatible version."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="20.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            macos_version_tuple=(11, 7, 10)
        )
        
        self.validator._validate_macos_version()
        
        assert not self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 1
        assert "11.7.10 is not supported" in self.validator.system_info.errors[0]
    
    def test_validate_macos_version_unknown(self):
        """Test macOS version validation when version is unknown."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            macos_version_tuple=None
        )
        
        self.validator._validate_macos_version()
        
        assert self.validator.system_info.is_compatible
        assert len(self.validator.system_info.warnings) == 1
        assert "Could not determine macOS version" in self.validator.system_info.warnings[0]
    
    def test_validate_python_version_compatible(self):
        """Test Python version validation with compatible version."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        with patch('sys.version_info', (3, 11, 5)):
            self.validator._validate_python_version()
        
        assert self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 0
    
    def test_validate_python_version_incompatible(self):
        """Test Python version validation with incompatible version."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.8.10"
        )
        
        with patch('sys.version_info', (3, 8, 10)):
            self.validator._validate_python_version()
        
        assert not self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 1
        assert "Python 3.8 is not supported" in self.validator.system_info.errors[0]
    
    def test_check_architecture_apple_silicon(self):
        """Test architecture checking for Apple Silicon."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="22.1.0",
            architecture="arm64",
            python_version="3.11.5",
            is_apple_silicon=True
        )
        
        self.validator._check_architecture()
        
        assert len(self.validator.system_info.warnings) == 1
        assert "Apple Silicon" in self.validator.system_info.warnings[0]
    
    def test_check_architecture_intel(self):
        """Test architecture checking for Intel."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            is_apple_silicon=False
        )
        
        self.validator._check_architecture()
        
        # Intel should not add warnings
        assert len(self.validator.system_info.warnings) == 0
    
    @patch('subprocess.run')
    def test_check_tool_availability_available(self, mock_run):
        """Test checking availability of an available tool."""
        # Mock which command success
        mock_run.side_effect = [
            Mock(returncode=0, stdout="/usr/bin/ps\n"),  # which ps
            Mock(returncode=0, stdout="ps version 1.0\n")  # ps --version
        ]
        
        tool_info = {'required': True, 'alternatives': [], 'requires_sudo': False}
        availability = self.validator._check_tool_availability('ps', tool_info)
        
        assert availability.tool_name == 'ps'
        assert availability.is_available
        assert availability.path == '/usr/bin/ps'
        assert availability.version == 'ps version 1.0'
        assert not availability.requires_sudo
    
    @patch('subprocess.run')
    def test_check_tool_availability_not_found(self, mock_run):
        """Test checking availability of a missing tool."""
        # Mock which command failure
        mock_run.return_value = Mock(returncode=1, stdout="")
        
        tool_info = {'required': True, 'alternatives': ['top'], 'requires_sudo': False}
        availability = self.validator._check_tool_availability('nonexistent', tool_info)
        
        assert availability.tool_name == 'nonexistent'
        assert not availability.is_available
        assert availability.error == "Tool not found in PATH"
        assert availability.alternatives == ['top']
    
    @patch('subprocess.run')
    def test_check_tool_availability_timeout(self, mock_run):
        """Test checking availability when command times out."""
        mock_run.side_effect = subprocess.TimeoutExpired('which', 5)
        
        tool_info = {'required': False, 'alternatives': [], 'requires_sudo': True}
        availability = self.validator._check_tool_availability('dtrace', tool_info)
        
        assert availability.tool_name == 'dtrace'
        assert not availability.is_available
        assert availability.error == "Tool check timed out"
        assert availability.requires_sudo
    
    @patch('subprocess.run')
    def test_get_tool_version_success(self, mock_run):
        """Test getting tool version successfully."""
        mock_run.return_value = Mock(returncode=0, stdout="tool version 2.1.0\nother info\n")
        
        version = self.validator._get_tool_version('tool')
        
        assert version == "tool version 2.1.0"
    
    @patch('subprocess.run')
    def test_get_tool_version_failure(self, mock_run):
        """Test getting tool version when it fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "tool")
        
        version = self.validator._get_tool_version('tool')
        
        assert version is None
    
    def test_check_python_dependencies_all_available(self):
        """Test checking Python dependencies when all are available."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        # Mock all imports to succeed
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            self.validator._check_python_dependencies()
        
        assert self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 0
    
    def test_check_python_dependencies_missing(self):
        """Test checking Python dependencies when some are missing."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        # Mock some imports to fail
        def mock_import(name, *args, **kwargs):
            if name in ['typer', 'rich']:
                return Mock()
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            self.validator._check_python_dependencies()
        
        assert not self.validator.system_info.is_compatible
        assert len(self.validator.system_info.errors) == 1
        assert "Missing required Python packages" in self.validator.system_info.errors[0]
    
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('platform.release')
    @patch('platform.machine')
    def test_validate_system_full_success(self, mock_machine, mock_release, mock_system, mock_run):
        """Test full system validation with successful result."""
        # Mock system info
        mock_system.return_value = "Darwin"
        mock_release.return_value = "22.1.0"
        mock_machine.return_value = "arm64"
        
        # Mock all subprocess calls to return success
        def mock_run_side_effect(cmd, **kwargs):
            if cmd == ["sw_vers", "-productVersion"]:
                return Mock(returncode=0, stdout="13.0.1\n")
            elif cmd[0] == "which":
                # All tools are available
                tool_name = cmd[1]
                return Mock(returncode=0, stdout=f"/usr/bin/{tool_name}\n")
            else:
                # Version checks
                return Mock(returncode=0, stdout=f"{cmd[0]} version 1.0\n")
        
        mock_run.side_effect = mock_run_side_effect
        
        # Mock Python dependencies
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            with patch('sys.version_info', (3, 11, 5)):
                system_info = self.validator.validate_system()
        
        assert system_info.is_compatible
        assert system_info.os_name == "Darwin"
        assert system_info.is_apple_silicon
        assert len(self.validator.tool_availability) > 0
    
    def test_get_installation_guidance_missing_packages(self):
        """Test getting installation guidance for missing packages."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            is_compatible=False
        )
        
        # Mock missing packages
        def mock_import(name, *args, **kwargs):
            if name in ['typer', 'rich']:
                return Mock()
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            guidance = self.validator.get_installation_guidance()
        
        assert len(guidance) > 0
        assert any("pip install" in line for line in guidance)
    
    def test_get_installation_guidance_old_macos(self):
        """Test getting installation guidance for old macOS."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="20.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            macos_version_tuple=(11, 7, 10),
            is_compatible=False
        )
        
        guidance = self.validator.get_installation_guidance()
        
        assert len(guidance) > 0
        assert any("Upgrade macOS" in line for line in guidance)
    
    def test_get_installation_guidance_old_python(self):
        """Test getting installation guidance for old Python."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.8.10",
            is_compatible=False
        )
        
        with patch('sys.version_info', (3, 8, 10)):
            guidance = self.validator.get_installation_guidance()
        
        assert len(guidance) > 0
        assert any("Upgrade Python" in line for line in guidance)
    
    def test_get_installation_guidance_missing_tools(self):
        """Test getting installation guidance for missing system tools."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            is_compatible=False
        )
        
        # Mock missing required tools
        self.validator.tool_availability = {
            'ps': ToolAvailability('ps', False, error="Not found"),
            'vm_stat': ToolAvailability('vm_stat', False, error="Not found"),
            'df': ToolAvailability('df', True, path="/bin/df")
        }
        
        guidance = self.validator.get_installation_guidance()
        
        assert len(guidance) > 0
        assert any("Missing required system tools" in line for line in guidance)
        assert any("xcode-select --install" in line for line in guidance)
    
    def test_get_installation_guidance_comprehensive(self):
        """Test comprehensive installation guidance with multiple issues."""
        self.validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="20.6.0",
            architecture="x86_64",
            python_version="3.8.10",
            macos_version_tuple=(11, 7, 10),
            is_compatible=False
        )
        
        # Mock missing tools and packages
        self.validator.tool_availability = {
            'ps': ToolAvailability('ps', False, error="Not found")
        }
        
        def mock_import(name, *args, **kwargs):
            if name in ['typer', 'rich']:
                return Mock()
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            with patch('sys.version_info', (3, 8, 10)):
                guidance = self.validator.get_installation_guidance()
        
        assert len(guidance) > 0
        # Should include guidance for all issues
        guidance_text = '\n'.join(guidance)
        assert "pip install" in guidance_text
        assert "Upgrade macOS" in guidance_text
        assert "Upgrade Python" in guidance_text
        assert "xcode-select --install" in guidance_text
    
    def test_get_available_tools(self):
        """Test getting list of available tools."""
        self.validator.tool_availability = {
            'ps': ToolAvailability('ps', True),
            'vm_stat': ToolAvailability('vm_stat', True),
            'dtrace': ToolAvailability('dtrace', False, error="Not found")
        }
        
        available = self.validator.get_available_tools()
        
        assert len(available) == 2
        assert 'ps' in available
        assert 'vm_stat' in available
        assert 'dtrace' not in available
    
    def test_get_missing_tools(self):
        """Test getting list of missing required tools."""
        self.validator.tool_availability = {
            'ps': ToolAvailability('ps', True),
            'vm_stat': ToolAvailability('vm_stat', False, error="Not found"),
            'dtrace': ToolAvailability('dtrace', False, error="Not found")
        }
        
        missing = self.validator.get_missing_tools()
        
        # vm_stat is required, dtrace is not
        assert 'vm_stat' in missing
        assert 'dtrace' not in missing
        assert 'ps' not in missing


class TestSystemInfo:
    """Test cases for SystemInfo dataclass."""
    
    def test_system_info_defaults(self):
        """Test SystemInfo default values."""
        info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        assert info.macos_version_tuple is None
        assert not info.is_apple_silicon
        assert info.is_compatible
        assert info.warnings == []
        assert info.errors == []
    
    def test_system_info_post_init(self):
        """Test SystemInfo __post_init__ method."""
        info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            warnings=None,
            errors=None
        )
        
        assert info.warnings == []
        assert info.errors == []


class TestToolAvailability:
    """Test cases for ToolAvailability dataclass."""
    
    def test_tool_availability_defaults(self):
        """Test ToolAvailability default values."""
        availability = ToolAvailability(
            tool_name="ps",
            is_available=True
        )
        
        assert availability.path is None
        assert availability.version is None
        assert availability.error is None
        assert not availability.requires_sudo
        assert availability.alternatives == []
    
    def test_tool_availability_post_init(self):
        """Test ToolAvailability __post_init__ method."""
        availability = ToolAvailability(
            tool_name="ps",
            is_available=True,
            alternatives=None
        )
        
        assert availability.alternatives == []


# Integration test scenarios for different macOS configurations
class TestSystemValidatorIntegration:
    """Integration tests for different system configurations."""
    
    @pytest.mark.parametrize("macos_version,expected_compatible", [
        ((15, 0, 0), True),   # macOS Sequoia
        ((14, 6, 1), True),   # macOS Sonoma
        ((13, 5, 2), True),   # macOS Ventura
        ((12, 7, 6), True),   # macOS Monterey
        ((12, 0, 0), True),   # Minimum supported
        ((11, 7, 10), False), # macOS Big Sur (too old)
        ((10, 15, 7), False), # macOS Catalina (very old)
        ((10, 14, 6), False), # macOS Mojave (very old)
    ])
    def test_macos_version_compatibility(self, macos_version, expected_compatible):
        """Test compatibility across different macOS versions."""
        validator = SystemValidator()
        validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5",
            macos_version_tuple=macos_version
        )
        
        validator._validate_macos_version()
        
        assert validator.system_info.is_compatible == expected_compatible
    
    @pytest.mark.parametrize("python_version,expected_compatible", [
        ((3, 12, 0), True),  # Latest
        ((3, 11, 5), True),  # Supported
        ((3, 10, 8), True),  # Supported
        ((3, 9, 16), True),  # Minimum
        ((3, 8, 18), False), # Too old
        ((3, 7, 17), False), # Very old
    ])
    def test_python_version_compatibility(self, python_version, expected_compatible):
        """Test compatibility across different Python versions."""
        validator = SystemValidator()
        validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version=f"{python_version[0]}.{python_version[1]}.{python_version[2]}"
        )
        
        with patch('sys.version_info', python_version):
            validator._validate_python_version()
        
        assert validator.system_info.is_compatible == expected_compatible
    
    @pytest.mark.parametrize("architecture,is_apple_silicon", [
        ("arm64", True),      # Apple Silicon M1/M2/M3
        ("aarch64", True),    # Alternative ARM64 identifier
        ("x86_64", False),    # Intel 64-bit
        ("i386", False),      # Intel 32-bit (legacy)
    ])
    def test_architecture_detection(self, architecture, is_apple_silicon):
        """Test architecture detection for different Mac types."""
        validator = SystemValidator()
        validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture=architecture,
            python_version="3.11.5",
            is_apple_silicon=is_apple_silicon
        )
        
        validator._check_architecture()
        
        if is_apple_silicon:
            assert len(validator.system_info.warnings) == 1
            assert "Apple Silicon" in validator.system_info.warnings[0]
        else:
            assert len(validator.system_info.warnings) == 0
    
    @pytest.mark.parametrize("os_name,macos_version,architecture,expected_compatible", [
        # Valid macOS configurations
        ("Darwin", (14, 0, 0), "arm64", True),     # M1/M2 Mac with Sonoma
        ("Darwin", (13, 0, 0), "x86_64", True),    # Intel Mac with Ventura
        ("Darwin", (12, 0, 0), "arm64", True),     # M1 Mac with Monterey
        ("Darwin", (12, 0, 0), "x86_64", True),    # Intel Mac with Monterey
        
        # Invalid configurations
        ("Linux", (14, 0, 0), "x86_64", False),    # Wrong OS
        ("Windows", (14, 0, 0), "x86_64", False),  # Wrong OS
        ("Darwin", (11, 0, 0), "x86_64", False),   # Too old macOS
        ("Darwin", (10, 15, 0), "x86_64", False),  # Very old macOS
    ])
    def test_hardware_configuration_compatibility(self, os_name, macos_version, architecture, expected_compatible):
        """Test compatibility across different hardware configurations."""
        validator = SystemValidator()
        validator.system_info = SystemInfo(
            os_name=os_name,
            os_version="21.6.0",
            architecture=architecture,
            python_version="3.11.5",
            macos_version_tuple=macos_version,
            is_apple_silicon=(architecture in ["arm64", "aarch64"])
        )
        
        # Run all validation steps
        validator._validate_operating_system()
        validator._validate_macos_version()
        validator._check_architecture()
        
        assert validator.system_info.is_compatible == expected_compatible
    
    @patch('subprocess.run')
    def test_system_tool_availability_scenarios(self, mock_run):
        """Test different system tool availability scenarios."""
        validator = SystemValidator()
        validator.system_info = SystemInfo(
            os_name="Darwin",
            os_version="21.6.0",
            architecture="x86_64",
            python_version="3.11.5"
        )
        
        # Scenario 1: All required tools available
        def mock_all_available(cmd, **kwargs):
            if cmd[0] == "which":
                return Mock(returncode=0, stdout=f"/usr/bin/{cmd[1]}\n")
            return Mock(returncode=0, stdout="version 1.0\n")
        
        mock_run.side_effect = mock_all_available
        validator._validate_system_tools()
        
        # Should be compatible with all tools available
        assert validator.system_info.is_compatible
        
        # Reset for next scenario
        validator.system_info.is_compatible = True
        validator.system_info.errors = []
        validator.system_info.warnings = []
        validator.tool_availability = {}
        
        # Scenario 2: Some optional tools missing
        def mock_optional_missing(cmd, **kwargs):
            if cmd[0] == "which":
                tool_name = cmd[1]
                if tool_name in ['dtrace', 'iostat', 'nettop']:  # Optional tools
                    return Mock(returncode=1, stdout="")
                return Mock(returncode=0, stdout=f"/usr/bin/{tool_name}\n")
            return Mock(returncode=0, stdout="version 1.0\n")
        
        mock_run.side_effect = mock_optional_missing
        validator._validate_system_tools()
        
        # Should still be compatible with only optional tools missing
        assert validator.system_info.is_compatible
        assert len(validator.system_info.warnings) > 0
        
        # Reset for next scenario
        validator.system_info.is_compatible = True
        validator.system_info.errors = []
        validator.system_info.warnings = []
        validator.tool_availability = {}
        
        # Scenario 3: Required tool missing with no alternatives
        def mock_required_missing(cmd, **kwargs):
            if cmd[0] == "which":
                tool_name = cmd[1]
                if tool_name == 'vm_stat':  # Required tool with no alternatives
                    return Mock(returncode=1, stdout="")
                return Mock(returncode=0, stdout=f"/usr/bin/{tool_name}\n")
            return Mock(returncode=0, stdout="version 1.0\n")
        
        mock_run.side_effect = mock_required_missing
        validator._validate_system_tools()
        
        # Should be incompatible with required tool missing
        assert not validator.system_info.is_compatible
        assert len(validator.system_info.errors) > 0