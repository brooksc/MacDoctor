"""
Unit tests for structured output parsing functionality.

Tests the Pydantic models, validation, and LangChain integration for
consistent LLM response parsing.
"""

import json
import pytest
from datetime import datetime
from pydantic import ValidationError

from mac_doctor.output_parsing import (
    StructuredIssue,
    StructuredRecommendation,
    DiagnosticAnalysis,
    StructuredOutputParser
)
from mac_doctor.interfaces import Issue, Recommendation


class TestStructuredIssue:
    """Test the StructuredIssue Pydantic model."""
    
    def test_valid_issue_creation(self):
        """Test creating a valid structured issue."""
        issue = StructuredIssue(
            severity="high",
            category="cpu",
            title="High CPU Usage",
            description="CPU usage is consistently above 90% due to runaway processes.",
            affected_processes=["chrome", "python"],
            metrics={"cpu_percent": 95.5, "load_average": 8.2},
            confidence=0.9
        )
        
        assert issue.severity == "high"
        assert issue.category == "cpu"
        assert issue.title == "High CPU Usage"
        assert len(issue.affected_processes) == 2
        assert issue.metrics["cpu_percent"] == 95.5
        assert issue.confidence == 0.9
    
    def test_invalid_severity(self):
        """Test validation of severity field."""
        with pytest.raises(ValidationError) as exc_info:
            StructuredIssue(
                severity="extreme",  # Invalid severity
                category="cpu",
                title="Test Issue",
                description="Test description for validation."
            )
        
        assert "severity" in str(exc_info.value)
    
    def test_invalid_category(self):
        """Test validation of category field."""
        with pytest.raises(ValidationError) as exc_info:
            StructuredIssue(
                severity="high",
                category="invalid_category",  # Invalid category
                title="Test Issue",
                description="Test description for validation."
            )
        
        assert "category" in str(exc_info.value)
    
    def test_title_length_validation(self):
        """Test title length validation."""
        # Too short title
        with pytest.raises(ValidationError):
            StructuredIssue(
                severity="high",
                category="cpu",
                title="",  # Empty title
                description="Test description for validation."
            )
        
        # Too long title
        with pytest.raises(ValidationError):
            StructuredIssue(
                severity="high",
                category="cpu",
                title="x" * 101,  # Too long title
                description="Test description for validation."
            )
    
    def test_description_length_validation(self):
        """Test description length validation."""
        # Too short description
        with pytest.raises(ValidationError):
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="Short"  # Too short
            )
        
        # Too long description
        with pytest.raises(ValidationError):
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="x" * 501  # Too long
            )
    
    def test_confidence_range_validation(self):
        """Test confidence value range validation."""
        # Valid confidence
        issue = StructuredIssue(
            severity="high",
            category="cpu",
            title="Test Issue",
            description="Test description for validation.",
            confidence=0.5
        )
        assert issue.confidence == 0.5
        
        # Invalid confidence (too low)
        with pytest.raises(ValidationError):
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="Test description for validation.",
                confidence=-0.1
            )
        
        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="Test description for validation.",
                confidence=1.1
            )
    
    def test_too_many_processes(self):
        """Test validation of affected processes count."""
        with pytest.raises(ValidationError) as exc_info:
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="Test description for validation.",
                affected_processes=["process"] * 21  # Too many processes
            )
        
        assert "Too many affected processes" in str(exc_info.value)
    
    def test_invalid_metrics(self):
        """Test validation of metrics values."""
        # Non-numeric metric
        with pytest.raises(ValidationError) as exc_info:
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="Test description for validation.",
                metrics={"cpu_percent": "high"}  # Non-numeric value
            )
        
        assert "valid number" in str(exc_info.value)
        
        # Negative metric
        with pytest.raises(ValidationError) as exc_info:
            StructuredIssue(
                severity="high",
                category="cpu",
                title="Test Issue",
                description="Test description for validation.",
                metrics={"cpu_percent": -10.0}  # Negative value
            )
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_to_interface_issue(self):
        """Test conversion to interface Issue class."""
        structured_issue = StructuredIssue(
            severity="high",
            category="cpu",
            title="High CPU Usage",
            description="CPU usage is consistently above 90%.",
            affected_processes=["chrome"],
            metrics={"cpu_percent": 95.5}
        )
        
        interface_issue = structured_issue.to_interface_issue()
        
        assert isinstance(interface_issue, Issue)
        assert interface_issue.severity == "high"
        assert interface_issue.category == "cpu"
        assert interface_issue.title == "High CPU Usage"
        assert interface_issue.affected_processes == ["chrome"]
        assert interface_issue.metrics == {"cpu_percent": 95.5}


class TestStructuredRecommendation:
    """Test the StructuredRecommendation Pydantic model."""
    
    def test_valid_recommendation_creation(self):
        """Test creating a valid structured recommendation."""
        rec = StructuredRecommendation(
            title="Restart Chrome",
            description="Chrome is consuming excessive CPU. Restarting it may resolve the issue.",
            action_type="command",
            command="pkill -f chrome",
            risk_level="medium",
            confirmation_required=True,
            priority=3,
            estimated_impact="high"
        )
        
        assert rec.title == "Restart Chrome"
        assert rec.action_type == "command"
        assert rec.command == "pkill -f chrome"
        assert rec.risk_level == "medium"
        assert rec.priority == 3
    
    def test_invalid_action_type(self):
        """Test validation of action_type field."""
        with pytest.raises(ValidationError) as exc_info:
            StructuredRecommendation(
                title="Test Recommendation",
                description="Test description for validation.",
                action_type="invalid_action"  # Invalid action type
            )
        
        assert "action_type" in str(exc_info.value)
    
    def test_command_validation_for_command_action(self):
        """Test that command is required for command action types."""
        # Command action without command should fail
        with pytest.raises(ValidationError) as exc_info:
            StructuredRecommendation(
                title="Test Command",
                description="Test description for validation.",
                action_type="command",
                command=None  # Missing command
            )
        
        assert "Command is required" in str(exc_info.value)
        
        # Sudo command without command should fail
        with pytest.raises(ValidationError) as exc_info:
            StructuredRecommendation(
                title="Test Sudo Command",
                description="Test description for validation.",
                action_type="sudo_command",
                command=""  # Empty command
            )
        
        assert "Command is required" in str(exc_info.value)
    
    def test_info_action_with_command(self):
        """Test that info actions can have commands (with warning)."""
        # This should work but log a warning
        rec = StructuredRecommendation(
            title="Info Action",
            description="Test description for validation.",
            action_type="info",
            command="some command"  # Command for info action
        )
        
        assert rec.action_type == "info"
        assert rec.command == "some command"
    
    def test_priority_range_validation(self):
        """Test priority value range validation."""
        # Valid priority
        rec = StructuredRecommendation(
            title="Test Recommendation",
            description="Test description for validation.",
            action_type="info",
            priority=5
        )
        assert rec.priority == 5
        
        # Invalid priority (too low)
        with pytest.raises(ValidationError):
            StructuredRecommendation(
                title="Test Recommendation",
                description="Test description for validation.",
                action_type="info",
                priority=0
            )
        
        # Invalid priority (too high)
        with pytest.raises(ValidationError):
            StructuredRecommendation(
                title="Test Recommendation",
                description="Test description for validation.",
                action_type="info",
                priority=11
            )
    
    def test_to_interface_recommendation(self):
        """Test conversion to interface Recommendation class."""
        structured_rec = StructuredRecommendation(
            title="Restart Service",
            description="Restart the problematic service to resolve the issue.",
            action_type="sudo_command",
            command="sudo systemctl restart service",
            risk_level="high",
            confirmation_required=True
        )
        
        interface_rec = structured_rec.to_interface_recommendation()
        
        assert isinstance(interface_rec, Recommendation)
        assert interface_rec.title == "Restart Service"
        assert interface_rec.action_type == "sudo_command"
        assert interface_rec.command == "sudo systemctl restart service"
        assert interface_rec.risk_level == "high"
        assert interface_rec.confirmation_required is True


class TestDiagnosticAnalysis:
    """Test the DiagnosticAnalysis Pydantic model."""
    
    def test_valid_analysis_creation(self):
        """Test creating a valid diagnostic analysis."""
        issue = StructuredIssue(
            severity="high",
            category="cpu",
            title="High CPU Usage",
            description="CPU usage is consistently above 90%."
        )
        
        rec = StructuredRecommendation(
            title="Restart Chrome",
            description="Chrome is consuming excessive CPU.",
            action_type="command",
            command="pkill -f chrome"
        )
        
        analysis = DiagnosticAnalysis(
            summary="System is experiencing high CPU usage due to Chrome browser.",
            issues=[issue],
            recommendations=[rec],
            system_health_score=0.6,
            analysis_confidence=0.8
        )
        
        assert len(analysis.issues) == 1
        assert len(analysis.recommendations) == 1
        assert analysis.system_health_score == 0.6
        assert analysis.analysis_confidence == 0.8
        assert isinstance(analysis.timestamp, datetime)
    
    def test_too_many_issues(self):
        """Test validation of issues count."""
        issues = [
            StructuredIssue(
                severity="low",
                category="cpu",
                title=f"Issue {i}",
                description="Test description for validation."
            )
            for i in range(51)  # Too many issues
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            DiagnosticAnalysis(
                summary="Test analysis with too many issues.",
                issues=issues
            )
        
        assert "Too many issues detected" in str(exc_info.value)
    
    def test_too_many_recommendations(self):
        """Test validation of recommendations count."""
        recommendations = [
            StructuredRecommendation(
                title=f"Recommendation {i}",
                description="Test description for validation.",
                action_type="info"
            )
            for i in range(21)  # Too many recommendations
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            DiagnosticAnalysis(
                summary="Test analysis with too many recommendations.",
                recommendations=recommendations
            )
        
        assert "Too many recommendations provided" in str(exc_info.value)


class TestStructuredOutputParser:
    """Test the StructuredOutputParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = StructuredOutputParser()
    
    def test_clean_json_output(self):
        """Test JSON output cleaning functionality."""
        # Test markdown code block removal
        markdown_json = """```json
        {"title": "Test", "description": "Test description"}
        ```"""
        
        cleaned = self.parser._clean_json_output(markdown_json)
        assert cleaned == '{"title": "Test", "description": "Test description"}'
        
        # Test generic code block removal
        generic_json = """```
        {"title": "Test", "description": "Test description"}
        ```"""
        
        cleaned = self.parser._clean_json_output(generic_json)
        assert cleaned == '{"title": "Test", "description": "Test description"}'
        
        # Test JSON extraction from mixed content
        mixed_content = """Here is the analysis:
        {"title": "Test", "description": "Test description"}
        That's the result."""
        
        cleaned = self.parser._clean_json_output(mixed_content)
        assert cleaned == '{"title": "Test", "description": "Test description"}'
    
    def test_parse_diagnostic_analysis_success(self):
        """Test successful parsing of diagnostic analysis."""
        valid_json = json.dumps({
            "summary": "System is running well with minor issues detected.",
            "issues": [
                {
                    "severity": "medium",
                    "category": "cpu",
                    "title": "Moderate CPU Usage",
                    "description": "CPU usage is at 70% which is acceptable but could be optimized.",
                    "confidence": 0.8
                }
            ],
            "recommendations": [
                {
                    "title": "Monitor CPU Usage",
                    "description": "Keep an eye on CPU usage patterns to identify potential issues.",
                    "action_type": "info",
                    "risk_level": "low",
                    "priority": 5
                }
            ],
            "system_health_score": 0.7,
            "analysis_confidence": 0.8
        })
        
        analysis = self.parser.parse_diagnostic_analysis(valid_json)
        
        assert isinstance(analysis, DiagnosticAnalysis)
        assert analysis.summary == "System is running well with minor issues detected."
        assert len(analysis.issues) == 1
        assert len(analysis.recommendations) == 1
        assert analysis.system_health_score == 0.7
    
    def test_parse_diagnostic_analysis_fallback(self):
        """Test fallback behavior for invalid diagnostic analysis."""
        invalid_json = "This is not valid JSON"
        
        analysis = self.parser.parse_diagnostic_analysis(invalid_json)
        
        assert isinstance(analysis, DiagnosticAnalysis)
        assert "parsing failed" in analysis.summary.lower()
        assert len(analysis.issues) == 1
        assert analysis.issues[0].title == "Output Parsing Error"
        assert len(analysis.recommendations) == 1
        assert analysis.analysis_confidence == 0.3
    
    def test_parse_recommendations_success(self):
        """Test successful parsing of recommendations."""
        valid_json = json.dumps([
            {
                "title": "Restart Service",
                "description": "Restart the problematic service to resolve the issue.",
                "action_type": "command",
                "command": "sudo systemctl restart service",
                "risk_level": "medium",
                "priority": 3
            },
            {
                "title": "Check Logs",
                "description": "Review system logs for additional information.",
                "action_type": "info",
                "risk_level": "low",
                "priority": 7
            }
        ])
        
        recommendations = self.parser.parse_recommendations(valid_json)
        
        assert len(recommendations) == 2
        assert all(isinstance(rec, StructuredRecommendation) for rec in recommendations)
        assert recommendations[0].title == "Restart Service"
        assert recommendations[1].action_type == "info"
    
    def test_parse_recommendations_single(self):
        """Test parsing of single recommendation."""
        valid_json = json.dumps({
            "title": "Single Recommendation",
            "description": "This is a single recommendation for testing.",
            "action_type": "info",
            "risk_level": "low",
            "priority": 5
        })
        
        recommendations = self.parser.parse_recommendations(valid_json)
        
        assert len(recommendations) == 1
        assert isinstance(recommendations[0], StructuredRecommendation)
        assert recommendations[0].title == "Single Recommendation"
    
    def test_parse_recommendations_fallback(self):
        """Test fallback behavior for invalid recommendations."""
        invalid_json = "This is not valid JSON"
        
        recommendations = self.parser.parse_recommendations(invalid_json)
        
        assert len(recommendations) == 1
        assert isinstance(recommendations[0], StructuredRecommendation)
        assert recommendations[0].title == "Review System Analysis"
        assert recommendations[0].action_type == "info"
    
    def test_parse_recommendations_skip_invalid(self):
        """Test skipping invalid recommendations in array."""
        mixed_json = json.dumps([
            {
                "title": "Valid Recommendation",
                "description": "This is a valid recommendation.",
                "action_type": "info",
                "risk_level": "low",
                "priority": 5
            },
            {
                "title": "",  # Invalid: empty title
                "description": "This recommendation has an empty title.",
                "action_type": "info"
            },
            {
                "title": "Another Valid Recommendation",
                "description": "This is another valid recommendation.",
                "action_type": "command",
                "command": "echo test",
                "risk_level": "low",
                "priority": 6
            }
        ])
        
        recommendations = self.parser.parse_recommendations(mixed_json)
        
        # Should have 2 valid recommendations (invalid one skipped)
        assert len(recommendations) == 2
        assert recommendations[0].title == "Valid Recommendation"
        assert recommendations[1].title == "Another Valid Recommendation"
    
    def test_validate_and_sanitize_output(self):
        """Test output validation and sanitization."""
        dirty_data = {
            "title": "Test\x00Title",  # Contains null byte
            "description": "  Test description  ",  # Extra whitespace
            "items": ["item1\x00", "  item2  "],  # List with null bytes and whitespace
            "nested": {
                "key": "value\x00"  # Nested null byte
            },
            "number": 42
        }
        
        sanitized = self.parser.validate_and_sanitize_output(dirty_data)
        
        assert sanitized["title"] == "TestTitle"
        assert sanitized["description"] == "Test description"
        assert sanitized["items"] == ["item1", "item2"]
        assert sanitized["nested"]["key"] == "value"
        assert sanitized["number"] == 42
    
    def test_create_analysis_prompt(self):
        """Test creation of analysis prompt template."""
        prompt = self.parser.create_analysis_prompt()
        
        assert "system_data" in prompt.input_variables
        assert "query" in prompt.input_variables
        assert "format_instructions" in prompt.partial_variables
        assert "JSON format" in prompt.template
    
    def test_create_recommendations_prompt(self):
        """Test creation of recommendations prompt template."""
        prompt = self.parser.create_recommendations_prompt()
        
        assert "analysis" in prompt.input_variables
        assert "format_instructions" in prompt.partial_variables
        assert "JSON format" in prompt.template


class TestIntegration:
    """Integration tests for output parsing with LLM providers."""
    
    def test_issue_to_interface_conversion(self):
        """Test conversion from structured to interface objects."""
        structured_issue = StructuredIssue(
            severity="high",
            category="memory",
            title="High Memory Usage",
            description="Memory usage is at 95% capacity.",
            affected_processes=["chrome", "firefox"],
            metrics={"memory_percent": 95.0},
            confidence=0.9
        )
        
        interface_issue = structured_issue.to_interface_issue()
        
        # Verify all fields are correctly transferred
        assert interface_issue.severity == structured_issue.severity
        assert interface_issue.category == structured_issue.category
        assert interface_issue.title == structured_issue.title
        assert interface_issue.description == structured_issue.description
        assert interface_issue.affected_processes == structured_issue.affected_processes
        assert interface_issue.metrics == structured_issue.metrics
    
    def test_recommendation_to_interface_conversion(self):
        """Test conversion from structured to interface objects."""
        structured_rec = StructuredRecommendation(
            title="Clear Browser Cache",
            description="Clear browser cache to free up memory.",
            action_type="manual",
            risk_level="low",
            confirmation_required=False,
            priority=4,
            estimated_impact="medium"
        )
        
        interface_rec = structured_rec.to_interface_recommendation()
        
        # Verify all fields are correctly transferred
        assert interface_rec.title == structured_rec.title
        assert interface_rec.description == structured_rec.description
        assert interface_rec.action_type == structured_rec.action_type
        assert interface_rec.risk_level == structured_rec.risk_level
        assert interface_rec.confirmation_required == structured_rec.confirmation_required
        # Note: priority and estimated_impact are not in the interface class