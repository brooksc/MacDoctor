"""
Unit tests for the ReportGenerator class.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from mac_doctor.core.report_generator import ReportGenerator
from mac_doctor.interfaces import (
    DiagnosticResult,
    Issue,
    Recommendation,
    MCPResult
)


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()
        
        # Create sample data
        self.sample_issues = [
            Issue(
                severity='high',
                category='cpu',
                title='High CPU Usage',
                description='CPU usage is consistently above 80%',
                affected_processes=['chrome', 'python'],
                metrics={'cpu_percent': 85.5, 'load_avg': 3.2}
            ),
            Issue(
                severity='medium',
                category='memory',
                title='Memory Pressure',
                description='System is experiencing memory pressure',
                affected_processes=['safari'],
                metrics={'memory_percent': 78.0}
            )
        ]
        
        self.sample_recommendations = [
            Recommendation(
                title='Restart High CPU Process',
                description='Consider restarting Chrome to reduce CPU usage',
                action_type='info',
                risk_level='low',
                confirmation_required=False
            ),
            Recommendation(
                title='Clear System Cache',
                description='Clear system caches to free up memory',
                action_type='sudo_command',
                command='purge',
                risk_level='medium',
                confirmation_required=True
            )
        ]
        
        self.sample_tool_results = {
            'process_mcp': MCPResult(
                tool_name='process_mcp',
                success=True,
                data={'top_processes': [{'name': 'chrome', 'cpu': 45.2}]},
                execution_time=0.5
            ),
            'memory_mcp': MCPResult(
                tool_name='memory_mcp',
                success=False,
                error='Permission denied',
                data={},
                execution_time=0.1
            )
        }
        
        self.sample_result = DiagnosticResult(
            query='Why is my Mac slow?',
            analysis="Test analysis",
            issues_detected=self.sample_issues,
            tool_results=self.sample_tool_results,
            recommendations=self.sample_recommendations,
            execution_time=2.5,
            timestamp=datetime(2024, 1, 15, 10, 30, 0
        )
        )
    
    def test_generate_markdown_with_issues(self):
        """Test markdown generation with issues and recommendations."""
        result = self.generator.generate_markdown(self.sample_result)
        
        # Check header
        assert '# Mac Doctor Diagnostic Report' in result
        assert '**Generated:** 2024-01-15 10:30:00' in result
        assert '**Query:** Why is my Mac slow?' in result
        assert '**Execution Time:** 2.50 seconds' in result
        
        # Check executive summary
        assert '## Executive Summary' in result
        assert 'Found **2** issues:' in result
        assert '🟠 **1** High' in result
        assert '🟡 **1** Medium' in result
        
        # Check issues section
        assert '## Issues Detected' in result
        assert '🟠 High CPU Usage' in result
        assert 'CPU usage is consistently above 80%' in result
        assert '**Affected Processes:**' in result
        assert '- chrome' in result
        assert '- python' in result
        assert '**Metrics:**' in result
        assert '- cpu_percent: 85.5' in result
        
        # Check recommendations section
        assert '## Recommendations' in result
        assert 'ℹ️ Restart High CPU Process' in result
        assert '⚡ Clear System Cache' in result
        assert '```bash' in result
        assert 'sudo purge' in result
        assert '**⚠️ Confirmation Required:**' in result
        
        # Check tool results section
        assert '## Diagnostic Tool Results' in result
        assert '### process_mcp' in result
        assert '✅ **Status:** Success' in result
        assert '### memory_mcp' in result
        assert '❌ **Status:** Failed' in result
        assert '**Error:** Permission denied' in result
    
    def test_generate_markdown_no_issues(self):
        """Test markdown generation with no issues."""
        result = DiagnosticResult(
            query='System check',
            analysis="Test analysis",
            issues_detected=[],
            tool_results={},
            recommendations=[],
            execution_time=1.0,
            timestamp=datetime(2024, 1, 15, 10, 30, 0
        )
        )
        
        markdown = self.generator.generate_markdown(result)
        
        assert '✅ **No issues detected**' in markdown
        assert 'Your system appears to be running normally' in markdown
        assert '## Issues Detected' not in markdown
        assert '## Recommendations' not in markdown
    
    def test_generate_json_format(self):
        """Test JSON generation with proper structure."""
        json_result = self.generator.generate_json(self.sample_result)
        data = json.loads(json_result)
        
        # Check metadata
        assert data['metadata']['query'] == 'Why is my Mac slow?'
        assert data['metadata']['execution_time'] == 2.5
        assert data['metadata']['issue_count'] == 2
        assert data['metadata']['recommendation_count'] == 2
        assert data['metadata']['tool_count'] == 2
        
        # Check summary
        assert data['summary']['issues_by_severity']['high'] == 1
        assert data['summary']['issues_by_severity']['medium'] == 1
        assert data['summary']['issues_by_category']['cpu'] == 1
        assert data['summary']['issues_by_category']['memory'] == 1
        assert data['summary']['recommendations_by_risk']['low'] == 1
        assert data['summary']['recommendations_by_risk']['medium'] == 1
        
        # Check issues
        assert len(data['issues']) == 2
        assert data['issues'][0]['severity'] == 'high'
        assert data['issues'][0]['title'] == 'High CPU Usage'
        assert data['issues'][0]['affected_processes'] == ['chrome', 'python']
        
        # Check recommendations
        assert len(data['recommendations']) == 2
        assert data['recommendations'][0]['title'] == 'Restart High CPU Process'
        assert data['recommendations'][1]['command'] == 'purge'
        
        # Check tool results
        assert 'process_mcp' in data['tool_results']
        assert data['tool_results']['process_mcp']['success'] is True
        assert 'memory_mcp' in data['tool_results']
        assert data['tool_results']['memory_mcp']['success'] is False
    
    def test_export_to_file_success(self):
        """Test successful file export."""
        content = "Test report content"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_report.md"
            
            self.generator.export_to_file(content, str(file_path))
            
            assert file_path.exists()
            assert file_path.read_text(encoding='utf-8') == content
    
    def test_export_to_file_creates_directories(self):
        """Test that export creates necessary directories."""
        content = "Test report content"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "subdir" / "nested" / "test_report.md"
            
            self.generator.export_to_file(content, str(file_path))
            
            assert file_path.exists()
            assert file_path.read_text(encoding='utf-8') == content
    
    def test_export_to_file_failure(self):
        """Test file export failure handling."""
        content = "Test report content"
        invalid_path = "/invalid/path/that/does/not/exist/report.md"
        
        with pytest.raises(IOError) as exc_info:
            self.generator.export_to_file(content, invalid_path)
        
        assert "Failed to export report" in str(exc_info.value)
    
    def test_format_recommendations_with_data(self):
        """Test formatting recommendations list."""
        result = self.generator.format_recommendations(self.sample_recommendations)
        
        assert '1. ℹ️ Restart High CPU Process' in result
        assert 'Consider restarting Chrome to reduce CPU usage' in result
        assert '2. ⚡ Clear System Cache' in result
        assert 'Action: sudo purge' in result
        assert '⚠️  Requires confirmation' in result
    
    def test_format_recommendations_empty(self):
        """Test formatting empty recommendations list."""
        result = self.generator.format_recommendations([])
        
        assert result == "No recommendations available."
    
    def test_severity_icons(self):
        """Test severity icon mapping."""
        assert self.generator._get_severity_icon('critical') == '🔴'
        assert self.generator._get_severity_icon('high') == '🟠'
        assert self.generator._get_severity_icon('medium') == '🟡'
        assert self.generator._get_severity_icon('low') == '🟢'
        assert self.generator._get_severity_icon('unknown') == '⚪'
    
    def test_risk_icons(self):
        """Test risk level icon mapping."""
        assert self.generator._get_risk_icon('high') == '⚠️'
        assert self.generator._get_risk_icon('medium') == '⚡'
        assert self.generator._get_risk_icon('low') == 'ℹ️'
        assert self.generator._get_risk_icon('unknown') == 'ℹ️'
    
    def test_count_issues_by_severity(self):
        """Test issue counting by severity."""
        counts = self.generator._count_issues_by_severity(self.sample_issues)
        
        assert counts['high'] == 1
        assert counts['medium'] == 1
        assert counts['low'] == 0
        assert counts['critical'] == 0
    
    def test_count_issues_by_category(self):
        """Test issue counting by category."""
        counts = self.generator._count_issues_by_category(self.sample_issues)
        
        assert counts['cpu'] == 1
        assert counts['memory'] == 1
    
    def test_count_recommendations_by_risk(self):
        """Test recommendation counting by risk level."""
        counts = self.generator._count_recommendations_by_risk(self.sample_recommendations)
        
        assert counts['low'] == 1
        assert counts['medium'] == 1
        assert counts['high'] == 0
    
    def test_issue_to_dict(self):
        """Test converting Issue to dictionary."""
        issue = self.sample_issues[0]
        result = self.generator._issue_to_dict(issue)
        
        assert result['severity'] == 'high'
        assert result['category'] == 'cpu'
        assert result['title'] == 'High CPU Usage'
        assert result['affected_processes'] == ['chrome', 'python']
        assert result['metrics']['cpu_percent'] == 85.5
    
    def test_recommendation_to_dict(self):
        """Test converting Recommendation to dictionary."""
        rec = self.sample_recommendations[1]
        result = self.generator._recommendation_to_dict(rec)
        
        assert result['title'] == 'Clear System Cache'
        assert result['action_type'] == 'sudo_command'
        assert result['command'] == 'purge'
        assert result['risk_level'] == 'medium'
        assert result['confirmation_required'] is True
    
    def test_mcp_result_to_dict(self):
        """Test converting MCPResult to dictionary."""
        mcp_result = self.sample_tool_results['process_mcp']
        result = self.generator._mcp_result_to_dict(mcp_result)
        
        assert result['tool_name'] == 'process_mcp'
        assert result['success'] is True
        assert result['data']['top_processes'][0]['name'] == 'chrome'
        assert result['execution_time'] == 0.5
    
    def test_markdown_with_all_severity_levels(self):
        """Test markdown generation with all severity levels."""
        issues = [
            Issue(severity='critical', category='system', title='Critical Issue', description='Critical'),
            Issue(severity='high', category='cpu', title='High Issue', description='High'),
            Issue(severity='medium', category='memory', title='Medium Issue', description='Medium'),
            Issue(severity='low', category='disk', title='Low Issue', description='Low')
        ]
        
        result = DiagnosticResult(
            query='Test query',
            analysis="Test analysis",
            issues_detected=issues,
            tool_results={},
            recommendations=[],
            execution_time=1.0
        
        )
        
        markdown = self.generator.generate_markdown(result)
        
        assert '🔴 **1** Critical' in markdown
        assert '🟠 **1** High' in markdown
        assert '🟡 **1** Medium' in markdown
        assert '🟢 **1** Low' in markdown
    
    def test_json_serialization_with_datetime(self):
        """Test JSON serialization handles datetime objects."""
        json_result = self.generator.generate_json(self.sample_result)
        data = json.loads(json_result)
        
        # Should not raise an exception and should contain timestamp
        assert 'generated_at' in data['metadata']
        assert '2024-01-15T10:30:00' in data['metadata']['generated_at']