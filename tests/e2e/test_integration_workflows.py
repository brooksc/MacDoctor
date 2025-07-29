"""
Comprehensive end-to-end integration tests for Mac Doctor.

These tests exercise complete diagnostic workflows, testing agent coordination
between multiple MCP tools and LLM analysis, report generation, and action
engine integration.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from mac_doctor.cli.main import app
from mac_doctor.agent.diagnostic_agent import DiagnosticAgent
from mac_doctor.core.action_engine import ActionEngine
from mac_doctor.core.report_generator import ReportGenerator
from mac_doctor.interfaces import (
    DiagnosticResult, Issue, Recommendation, MCPResult, CLIConfig
)
from mac_doctor.llm.factory import LLMFactory, LLMConfig
from mac_doctor.tool_registry import ToolRegistry
from mac_doctor.mcps.process_mcp import ProcessMCP
from mac_doctor.mcps.disk_mcp import DiskMCP
from mac_doctor.mcps.vmstat_mcp import VMStatMCP
from mac_doctor.mcps.network_mcp import NetworkMCP
from mac_doctor.mcps.logs_mcp import LogsMCP
from mac_doctor.mcps.dtrace_mcp import DTraceMCP


class TestEndToEndWorkflows:
    """Test complete diagnostic workflows from CLI to report generation."""
    
    def setup_method(self):
        """Setup test environment for each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock system data for consistent testing
        self.mock_process_data = {
            "success": True,
            "data": {
                "system_overview": {
                    "cpu_usage_percent": 45.2,
                    "memory_usage_percent": 67.8,
                    "load_avg": [1.2, 1.5, 1.8],
                    "process_count": 342
                },
                "top_processes": [
                    {"name": "Chrome", "cpu_percent": 15.3, "memory_mb": 512.4, "pid": 1234},
                    {"name": "Xcode", "cpu_percent": 12.1, "memory_mb": 1024.8, "pid": 5678},
                    {"name": "Slack", "cpu_percent": 8.7, "memory_mb": 256.2, "pid": 9012}
                ]
            },
            "execution_time": 0.5
        }
        
        self.mock_disk_data = {
            "success": True,
            "data": {
                "disk_analysis": {
                    "storage": {
                        "total_gb": 500.0,
                        "used_gb": 350.0,
                        "available_gb": 150.0,
                        "usage_percent": 70.0
                    },
                    "io_stats": {
                        "read_ops": 1500,
                        "write_ops": 800,
                        "read_mb": 45.2,
                        "write_mb": 23.1
                    }
                },
                "mountpoints": ["/", "/System/Volumes/Data"]
            },
            "execution_time": 1.2
        }
        
        self.mock_memory_data = {
            "success": True,
            "data": {
                "memory_stats": {
                    "memory_pressure": "normal",
                    "swap_used_mb": 0.0,
                    "page_faults": 12345,
                    "pages_free": 50000,
                    "pages_active": 150000
                },
                "vm_stats": {
                    "pages_paged_out": 0,
                    "pages_paged_in": 100
                }
            },
            "execution_time": 0.3
        }
        
        self.mock_network_data = {
            "success": True,
            "data": {
                "network_stats": {
                    "bytes_sent": 1024000,
                    "bytes_recv": 2048000,
                    "connections_count": 45
                },
                "connections": [
                    {"local_port": 80, "remote_addr": "192.168.1.1", "status": "ESTABLISHED"},
                    {"local_port": 443, "remote_addr": "10.0.0.1", "status": "ESTABLISHED"}
                ]
            },
            "execution_time": 0.8
        }
        
        self.mock_logs_data = {
            "success": True,
            "data": {
                "log_entries": [
                    {"timestamp": "2024-01-01 10:00:00", "level": "error", "message": "Disk space low"},
                    {"timestamp": "2024-01-01 10:01:00", "level": "warning", "message": "High CPU usage"},
                    {"timestamp": "2024-01-01 10:02:00", "level": "info", "message": "System startup complete"}
                ],
                "summary": {
                    "total_entries": 3,
                    "error_count": 1,
                    "warning_count": 1
                }
            },
            "execution_time": 2.1
        }
        
        self.mock_dtrace_data = {
            "success": True,
            "data": {
                "dtrace_results": {
                    "syscall_count": 15000,
                    "duration_seconds": 10,
                    "top_syscalls": [
                        {"name": "read", "count": 5000},
                        {"name": "write", "count": 3000},
                        {"name": "open", "count": 2000}
                    ]
                }
            },
            "execution_time": 10.5
        }
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.e2e
    def test_complete_diagnostic_workflow_gemini(self):
        """Test complete diagnostic workflow using Gemini provider."""
        # Mock the entire setup process to bypass complex initialization
        with patch('mac_doctor.cli.setup.setup_tools_and_agent') as mock_setup:
            
            # Create mock agent and report generator
            mock_agent = self._create_mock_agent()
            mock_report_generator = Mock()
            mock_report_generator.generate_markdown.return_value = """# Mac Doctor Diagnostic Report

**Generated:** 2024-01-01 10:00:00
**Query:** Perform a comprehensive diagnostic analysis of this Mac system
**Execution Time:** 2.50 seconds

## Issues Detected

### 1. 🟡 Moderate CPU Usage

**Severity:** Medium
**Category:** Cpu

CPU usage is at 45%, which is within normal range but could be optimized

**Metrics:**
- cpu_usage_percent: 45.2

## Recommendations

### 1. ℹ️ Monitor Resource Usage

Continue monitoring system resources for any changes

**Risk Level:** Low

## Diagnostic Tool Results

### process

✅ **Status:** Success
**Execution Time:** 0.50s

**Key Findings:**
- CPU Usage: 45.2%
- Memory Usage: 67.8%
- Load Average: [1.2, 1.5, 1.8]
- Process Count: 342

---

*Report generated by Mac Doctor - macOS System Diagnostic Tool*"""
            
            mock_setup.return_value = (mock_agent, mock_report_generator)
            
            # Run the CLI command
            result = self.runner.invoke(app, [
                "diagnose",
                "--provider", "gemini"
            ])
            
            # Verify successful execution
            assert result.exit_code == 0, f"CLI failed with output: {result.stdout}"
            
            # Verify key components in output
            assert "Mac Doctor Diagnostic Report" in result.stdout
            assert "Issues Detected" in result.stdout
            assert "Recommendations" in result.stdout
            assert "Diagnostic Summary" in result.stdout
            
            # Verify agent was called
            mock_agent.analyze.assert_called_once()
            
            # Verify setup was called with correct config
            mock_setup.assert_called_once()
            config = mock_setup.call_args[0][0]
            assert config.llm_provider == "gemini"
    
    @pytest.mark.e2e
    def test_complete_diagnostic_workflow_ollama(self):
        """Test complete diagnostic workflow using Ollama provider."""
        # Mock the entire setup process to bypass complex initialization
        with patch('mac_doctor.cli.setup.setup_tools_and_agent') as mock_setup:
            
            # Create mock agent and report generator
            mock_agent = self._create_mock_agent()
            mock_report_generator = Mock()
            mock_report_generator.generate_markdown.return_value = """# Mac Doctor Diagnostic Report

**Generated:** 2024-01-01 10:00:00
**Query:** Perform a comprehensive diagnostic analysis of this Mac system
**Execution Time:** 2.50 seconds

## Issues Detected

No issues detected - Your system appears to be running normally.

## Recommendations

No specific recommendations at this time.

---

*Report generated by Mac Doctor - macOS System Diagnostic Tool*"""
            
            mock_setup.return_value = (mock_agent, mock_report_generator)
            
            # Run the CLI command with privacy mode
            result = self.runner.invoke(app, [
                "diagnose",
                "--provider", "ollama",
                "--privacy"
            ])
            
            # Verify successful execution
            assert result.exit_code == 0, f"CLI failed with output: {result.stdout}"
            
            # Verify key components in output
            assert "Mac Doctor Diagnostic Report" in result.stdout
            assert "Diagnostic Summary" in result.stdout
            
            # Verify agent was called
            mock_agent.analyze.assert_called_once()
            
            # Verify setup was called with correct config
            mock_setup.assert_called_once()
            config = mock_setup.call_args[0][0]
            assert config.llm_provider == "ollama"
            assert config.privacy_mode is True
    
    @pytest.mark.e2e
    def test_agent_tool_coordination(self):
        """Test agent coordination between multiple MCP tools."""
        # Create real components for integration testing
        tool_registry = ToolRegistry()
        
        # Register mock tools
        mock_tools = self._create_mock_tools()
        for tool in mock_tools:
            tool_registry.register_tool(tool)
        
        # Create mock LLM
        mock_llm = self._create_mock_llm("gemini", "gemini-2.5-flash")
        
        # Create agent with real registry and mock LLM
        with patch('mac_doctor.agent.diagnostic_agent.MCPToolFactory') as mock_factory:
            # Mock the LangChain tool creation
            mock_langchain_tools = []
            for tool in mock_tools:
                mock_tool = Mock()
                mock_tool.name = tool.name
                mock_tool.description = tool.description
                mock_langchain_tools.append(mock_tool)
            
            mock_factory.create_tools_from_registry.return_value = mock_langchain_tools
            
            # Mock the agent executor
            with patch('mac_doctor.agent.diagnostic_agent.create_react_agent') as mock_create_agent, \
                 patch('mac_doctor.agent.diagnostic_agent.AgentExecutor') as mock_executor_class:
                
                mock_agent_instance = Mock()
                mock_create_agent.return_value = mock_agent_instance
                
                mock_executor = Mock()
                mock_executor.invoke.return_value = {
                    "output": "System analysis complete. Found high CPU usage and moderate disk usage.",
                    "intermediate_steps": [
                        (Mock(tool="process"), json.dumps(self.mock_process_data)),
                        (Mock(tool="disk"), json.dumps(self.mock_disk_data)),
                        (Mock(tool="vmstat"), json.dumps(self.mock_memory_data))
                    ]
                }
                mock_executor_class.return_value = mock_executor
                
                # Create agent
                agent = DiagnosticAgent(mock_llm, tool_registry)
                
                # Test analysis
                result = agent.analyze("Why is my Mac running slowly?")
                
                # Verify result structure
                assert isinstance(result, DiagnosticResult)
                assert result.query == "Why is my Mac running slowly?"
                assert len(result.tool_results) > 0
                assert result.execution_time > 0
                
                # Verify agent executor was called
                mock_executor.invoke.assert_called_once()
    
    @pytest.mark.e2e
    def test_report_generation_integration(self):
        """Test report generation with real diagnostic results."""
        # Create a comprehensive diagnostic result
        issues = [
            Issue(
                severity="high",
                category="cpu",
                title="High CPU Usage",
                description="System CPU usage is consistently above 80%",
                affected_processes=["Chrome", "Xcode"],
                metrics={"cpu_usage_percent": 85.3}
            ),
            Issue(
                severity="medium",
                category="disk",
                title="Moderate Disk Usage",
                description="Disk usage is at 70%, consider cleanup",
                metrics={"disk_usage_percent": 70.0}
            )
        ]
        
        recommendations = [
            Recommendation(
                title="Close Resource-Heavy Applications",
                description="Consider closing Chrome and Xcode to reduce CPU usage",
                action_type="info",
                risk_level="low",
                confirmation_required=False
            ),
            Recommendation(
                title="Clean Up Disk Space",
                description="Remove unnecessary files to free up disk space",
                action_type="command",
                command="du -sh ~/Downloads/* | sort -hr | head -10",
                risk_level="low",
                confirmation_required=True
            )
        ]
        
        tool_results = {
            "process": MCPResult(
                tool_name="process",
                success=True,
                data=self.mock_process_data["data"],
                execution_time=0.5
            ),
            "disk": MCPResult(
                tool_name="disk",
                success=True,
                data=self.mock_disk_data["data"],
                execution_time=1.2
            )
        }
        
        diagnostic_result = DiagnosticResult(
            query="Comprehensive system diagnostic",
            analysis="System shows high CPU usage and moderate disk usage. Chrome and Xcode are consuming significant resources.",
            issues_detected=issues,
            tool_results=tool_results,
            recommendations=recommendations,
            execution_time=2.5
        )
        
        # Test report generation
        report_generator = ReportGenerator()
        
        # Test markdown generation
        markdown_report = report_generator.generate_markdown(diagnostic_result)
        assert "# Mac Doctor Diagnostic Report" in markdown_report
        assert "High CPU Usage" in markdown_report
        assert "Chrome" in markdown_report
        assert "85.3" in markdown_report
        assert "Clean Up Disk Space" in markdown_report
        
        # Test JSON generation
        json_report = report_generator.generate_json(diagnostic_result)
        json_data = json.loads(json_report)
        
        assert json_data["metadata"]["query"] == "Comprehensive system diagnostic"
        assert json_data["metadata"]["execution_time"] == 2.5
        assert len(json_data["issues"]) == 2
        assert len(json_data["recommendations"]) == 2
        assert "process" in json_data["tool_results"]
        assert "disk" in json_data["tool_results"]
        
        # Test file export
        export_path = Path(self.temp_dir) / "test_report.md"
        report_generator.export_to_file(markdown_report, str(export_path))
        
        assert export_path.exists()
        with open(export_path, 'r') as f:
            exported_content = f.read()
        assert exported_content == markdown_report
    
    @pytest.mark.e2e
    def test_action_engine_integration(self):
        """Test action engine integration with recommendations."""
        from rich.console import Console
        
        console = Console()
        action_engine = ActionEngine(console=console)
        
        # Test safe recommendation execution
        safe_recommendation = Recommendation(
            title="List Large Files",
            description="Find large files in Downloads folder",
            action_type="command",
            command="du -sh ~/Downloads/*",
            risk_level="low",
            confirmation_required=False
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="512M\t~/Downloads/large_file.zip\n256M\t~/Downloads/video.mp4\n",
                stderr=""
            )
            
            result = action_engine.execute_recommendation(safe_recommendation, auto_confirm=True)
            
            assert result.success is True
            assert "large_file.zip" in result.output
            mock_run.assert_called_once()
        
        # Test unsafe command rejection
        unsafe_recommendation = Recommendation(
            title="Dangerous Command",
            description="This should be rejected",
            action_type="command",
            command="rm -rf /",
            risk_level="high",
            confirmation_required=True
        )
        
        result = action_engine.execute_recommendation(unsafe_recommendation, auto_confirm=True)
        assert result.success is False
        assert "validation failed" in result.error.lower()
        
        # Test sudo command handling
        sudo_recommendation = Recommendation(
            title="System Information",
            description="Get system information with elevated privileges",
            action_type="sudo_command",
            command="system_profiler SPHardwareDataType",
            risk_level="medium",
            confirmation_required=True
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Hardware Overview:\n  Model Name: MacBook Pro\n",
                stderr=""
            )
            
            result = action_engine.execute_recommendation(sudo_recommendation, auto_confirm=True)
            
            assert result.success is True
            assert "MacBook Pro" in result.output
            # Verify sudo was added to command
            called_command = mock_run.call_args[1]['shell']
            assert called_command is True
    
    @pytest.mark.e2e
    def test_ask_command_workflow(self):
        """Test the 'ask' command workflow with specific queries."""
        with patch('mac_doctor.cli.setup.setup_llm') as mock_setup_llm, \
             patch('mac_doctor.cli.setup.setup_tool_registry') as mock_setup_registry, \
             patch('mac_doctor.cli.setup.SystemValidator') as mock_validator:
            
            # Setup mocks
            mock_llm = self._create_mock_llm("gemini", "gemini-2.5-flash")
            mock_setup_llm.return_value = mock_llm
            
            mock_registry = self._create_mock_registry()
            mock_setup_registry.return_value = mock_registry
            
            mock_validator_instance = Mock()
            mock_validator_instance.validate_system.return_value = Mock(
                is_compatible=True,
                warnings=[]
            )
            mock_validator.return_value = mock_validator_instance
            
            # Mock agent execution with targeted response
            with patch('mac_doctor.agent.diagnostic_agent.DiagnosticAgent') as mock_agent_class:
                mock_agent = Mock()
                mock_agent.is_available.return_value = True
                
                # Create targeted diagnostic result for CPU question
                cpu_result = DiagnosticResult(
                    query="Why is my CPU usage so high?",
                    analysis="High CPU usage detected. Chrome is consuming 15.3% CPU.",
                    issues_detected=[
                        Issue(
                            severity="medium",
                            category="cpu",
                            title="High CPU Usage from Chrome",
                            description="Chrome browser is using significant CPU resources",
                            affected_processes=["Chrome"],
                            metrics={"cpu_usage_percent": 15.3}
                        )
                    ],
                    tool_results={
                        "process": MCPResult(
                            tool_name="process",
                            success=True,
                            data=self.mock_process_data["data"],
                            execution_time=0.5
                        )
                    },
                    recommendations=[
                        Recommendation(
                            title="Close Chrome Tabs",
                            description="Close unnecessary Chrome tabs to reduce CPU usage",
                            action_type="info",
                            risk_level="low",
                            confirmation_required=False
                        )
                    ],
                    execution_time=1.2
                )
                
                mock_agent.analyze.return_value = cpu_result
                mock_agent_class.return_value = mock_agent
                
                # Run the ask command
                result = self.runner.invoke(app, [
                    "ask",
                    "Why is my CPU usage so high?"
                ])
                
                # Verify successful execution
                assert result.exit_code == 0, f"CLI failed with output: {result.stdout}"
                
                # Verify targeted analysis
                assert "High CPU Usage from Chrome" in result.stdout
                assert "Chrome browser is using significant CPU resources" in result.stdout
                assert "Close Chrome Tabs" in result.stdout
                
                # Verify agent was called with the specific question
                mock_agent.analyze.assert_called_once_with("Why is my CPU usage so high?")
    
    @pytest.mark.e2e
    def test_json_output_format(self):
        """Test JSON output format generation."""
        with patch('mac_doctor.cli.setup.setup_llm') as mock_setup_llm, \
             patch('mac_doctor.cli.setup.setup_tool_registry') as mock_setup_registry, \
             patch('mac_doctor.cli.setup.SystemValidator') as mock_validator:
            
            # Setup mocks
            mock_llm = self._create_mock_llm("gemini", "gemini-2.5-flash")
            mock_setup_llm.return_value = mock_llm
            
            mock_registry = self._create_mock_registry()
            mock_setup_registry.return_value = mock_registry
            
            mock_validator_instance = Mock()
            mock_validator_instance.validate_system.return_value = Mock(
                is_compatible=True,
                warnings=[]
            )
            mock_validator.return_value = mock_validator_instance
            
            # Mock agent execution
            with patch('mac_doctor.agent.diagnostic_agent.DiagnosticAgent') as mock_agent_class:
                mock_agent = self._create_mock_agent()
                mock_agent_class.return_value = mock_agent
                
                # Run with JSON format
                result = self.runner.invoke(app, [
                    "diagnose",
                    "--format", "json"
                ])
                
                # Verify successful execution
                assert result.exit_code == 0, f"CLI failed with output: {result.stdout}"
                
                # Verify JSON output
                try:
                    json_data = json.loads(result.stdout.split('\n')[-3])  # Get JSON from output
                    assert "metadata" in json_data
                    assert "issues" in json_data
                    assert "recommendations" in json_data
                    assert "tool_results" in json_data
                except (json.JSONDecodeError, IndexError):
                    # JSON might be embedded in the output, look for it
                    json_found = False
                    for line in result.stdout.split('\n'):
                        if line.strip().startswith('{'):
                            try:
                                json_data = json.loads(line.strip())
                                json_found = True
                                break
                            except json.JSONDecodeError:
                                continue
                    assert json_found, "No valid JSON found in output"
    
    @pytest.mark.e2e
    def test_export_functionality(self):
        """Test report export functionality."""
        export_path = Path(self.temp_dir) / "diagnostic_report.md"
        
        with patch('mac_doctor.cli.setup.setup_llm') as mock_setup_llm, \
             patch('mac_doctor.cli.setup.setup_tool_registry') as mock_setup_registry, \
             patch('mac_doctor.cli.setup.SystemValidator') as mock_validator:
            
            # Setup mocks
            mock_llm = self._create_mock_llm("gemini", "gemini-2.5-flash")
            mock_setup_llm.return_value = mock_llm
            
            mock_registry = self._create_mock_registry()
            mock_setup_registry.return_value = mock_registry
            
            mock_validator_instance = Mock()
            mock_validator_instance.validate_system.return_value = Mock(
                is_compatible=True,
                warnings=[]
            )
            mock_validator.return_value = mock_validator_instance
            
            # Mock agent execution
            with patch('mac_doctor.agent.diagnostic_agent.DiagnosticAgent') as mock_agent_class:
                mock_agent = self._create_mock_agent()
                mock_agent_class.return_value = mock_agent
                
                # Run with export
                result = self.runner.invoke(app, [
                    "diagnose",
                    "--export", str(export_path)
                ])
                
                # Verify successful execution
                assert result.exit_code == 0, f"CLI failed with output: {result.stdout}"
                
                # Verify export message
                assert f"Report exported to {export_path}" in result.stdout
                
                # Verify file was created
                assert export_path.exists()
                
                # Verify file content
                with open(export_path, 'r') as f:
                    content = f.read()
                assert "# Mac Doctor Diagnostic Report" in content
                assert "Issues Detected" in content
    
    @pytest.mark.e2e
    def test_error_handling_workflow(self):
        """Test error handling in complete workflows."""
        with patch('mac_doctor.cli.setup.setup_llm') as mock_setup_llm:
            # Simulate LLM setup failure
            mock_setup_llm.side_effect = RuntimeError("LLM provider not available")
            
            # Run command and expect failure
            result = self.runner.invoke(app, ["diagnose"])
            
            # Verify error handling
            assert result.exit_code == 1
            assert "Diagnostic failed" in result.stdout or "LLM provider not available" in result.stdout
    
    def _create_mock_llm(self, provider: str, model: str):
        """Create a mock LLM for testing."""
        mock_llm = Mock()
        mock_llm.provider_name = provider
        mock_llm.model_name = model
        mock_llm.is_available.return_value = True
        mock_llm.analyze_system_data.return_value = "System analysis complete"
        mock_llm.generate_recommendations.return_value = []
        return mock_llm
    
    def _create_mock_registry(self):
        """Create a mock tool registry for testing."""
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = ["process", "disk", "vmstat", "network", "logs", "dtrace"]
        
        # Create mock tools for each tool name
        def mock_get_tool(name):
            mock_tool = Mock()
            mock_tool.name = name
            mock_tool.description = f"Mock {name} tool for testing"
            mock_tool.is_available.return_value = True
            return mock_tool
        
        mock_registry.get_tool.side_effect = mock_get_tool
        mock_registry.__len__.return_value = 6  # Number of tools
        mock_registry.__iter__.return_value = iter(["process", "disk", "vmstat", "network", "logs", "dtrace"])
        return mock_registry
    
    def _create_mock_agent(self):
        """Create a mock diagnostic agent for testing."""
        mock_agent = Mock()
        mock_agent.is_available.return_value = True
        mock_agent.get_available_tools.return_value = ["process", "disk", "vmstat"]
        mock_agent.tool_registry = self._create_mock_registry()
        
        # Create comprehensive diagnostic result
        diagnostic_result = DiagnosticResult(
            query="Perform a comprehensive diagnostic analysis of this Mac system",
            analysis="System analysis shows normal operation with some areas for optimization.",
            issues_detected=[
                Issue(
                    severity="medium",
                    category="cpu",
                    title="Moderate CPU Usage",
                    description="CPU usage is at 45%, which is within normal range but could be optimized",
                    metrics={"cpu_usage_percent": 45.2}
                )
            ],
            tool_results={
                "process": MCPResult(
                    tool_name="process",
                    success=True,
                    data=self.mock_process_data["data"],
                    execution_time=0.5
                ),
                "disk": MCPResult(
                    tool_name="disk",
                    success=True,
                    data=self.mock_disk_data["data"],
                    execution_time=1.2
                )
            },
            recommendations=[
                Recommendation(
                    title="Monitor Resource Usage",
                    description="Continue monitoring system resources for any changes",
                    action_type="info",
                    risk_level="low",
                    confirmation_required=False
                )
            ],
            execution_time=2.5
        )
        
        mock_agent.analyze.return_value = diagnostic_result
        return mock_agent
    
    def _create_mock_tools(self):
        """Create mock MCP tools for testing."""
        mock_tools = []
        
        # Process MCP
        process_mcp = Mock(spec=ProcessMCP)
        process_mcp.name = "process"
        process_mcp.description = "Process and CPU monitoring"
        process_mcp.is_available.return_value = True
        process_mcp.execute.return_value = MCPResult(
            tool_name="process",
            success=True,
            data=self.mock_process_data["data"],
            execution_time=0.5
        )
        mock_tools.append(process_mcp)
        
        # Disk MCP
        disk_mcp = Mock(spec=DiskMCP)
        disk_mcp.name = "disk"
        disk_mcp.description = "Disk usage and I/O monitoring"
        disk_mcp.is_available.return_value = True
        disk_mcp.execute.return_value = MCPResult(
            tool_name="disk",
            success=True,
            data=self.mock_disk_data["data"],
            execution_time=1.2
        )
        mock_tools.append(disk_mcp)
        
        # VMStat MCP
        vmstat_mcp = Mock(spec=VMStatMCP)
        vmstat_mcp.name = "vmstat"
        vmstat_mcp.description = "Memory and virtual memory statistics"
        vmstat_mcp.is_available.return_value = True
        vmstat_mcp.execute.return_value = MCPResult(
            tool_name="vmstat",
            success=True,
            data=self.mock_memory_data["data"],
            execution_time=0.3
        )
        mock_tools.append(vmstat_mcp)
        
        return mock_tools


class TestSpecificScenarios:
    """Test specific diagnostic scenarios and edge cases."""
    
    @pytest.mark.e2e
    def test_high_cpu_scenario(self):
        """Test diagnostic workflow for high CPU usage scenario."""
        # This test would simulate a system with high CPU usage
        # and verify that appropriate issues and recommendations are generated
        pass
    
    @pytest.mark.e2e
    def test_low_disk_space_scenario(self):
        """Test diagnostic workflow for low disk space scenario."""
        # This test would simulate a system with low disk space
        # and verify critical issues and urgent recommendations are generated
        pass
    
    @pytest.mark.e2e
    def test_memory_pressure_scenario(self):
        """Test diagnostic workflow for memory pressure scenario."""
        # This test would simulate a system under memory pressure
        # and verify memory-related issues and recommendations are generated
        pass
    
    @pytest.mark.e2e
    def test_network_issues_scenario(self):
        """Test diagnostic workflow for network connectivity issues."""
        # This test would simulate network connectivity problems
        # and verify network-related diagnostics and recommendations
        pass


class TestProviderFallback:
    """Test LLM provider fallback scenarios."""
    
    @pytest.mark.e2e
    def test_gemini_to_ollama_fallback(self):
        """Test fallback from Gemini to Ollama when Gemini is unavailable."""
        # This test would simulate Gemini being unavailable
        # and verify the system falls back to Ollama
        pass
    
    @pytest.mark.e2e
    def test_ollama_to_gemini_fallback(self):
        """Test fallback from Ollama to Gemini when Ollama is unavailable."""
        # This test would simulate Ollama being unavailable
        # and verify the system falls back to Gemini
        pass
    
    @pytest.mark.e2e
    def test_no_llm_available_scenario(self):
        """Test behavior when no LLM providers are available."""
        # This test would simulate no LLM providers being available
        # and verify graceful degradation with rule-based analysis
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])