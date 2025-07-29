"""
Structured output parsing for Mac Doctor using Pydantic models and LangChain.

This module provides Pydantic models for diagnostic results and recommendations,
along with LangChain-based structured output parsing for consistent LLM responses.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .interfaces import Issue, Recommendation

logger = logging.getLogger(__name__)


class StructuredIssue(BaseModel):
    """Pydantic model for system issues with validation."""
    
    severity: str = Field(
        ...,
        description="Issue severity level",
        pattern="^(low|medium|high|critical)$"
    )
    category: str = Field(
        ...,
        description="Issue category",
        pattern="^(cpu|memory|disk|network|thermal|system|security)$"
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Brief issue title"
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Detailed issue description"
    )
    affected_processes: List[str] = Field(
        default_factory=list,
        description="List of processes affected by this issue"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevant metrics for this issue"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level in issue detection (0.0-1.0)"
    )
    
    @field_validator('affected_processes')
    @classmethod
    def validate_processes(cls, v):
        """Validate process names."""
        if len(v) > 20:  # Reasonable limit
            raise ValueError("Too many affected processes")
        return v
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v):
        """Validate metrics values."""
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {key} must be numeric")
            if value < 0:
                raise ValueError(f"Metric {key} cannot be negative")
        return v
    
    def to_interface_issue(self) -> Issue:
        """Convert to the interface Issue class."""
        return Issue(
            severity=self.severity,
            category=self.category,
            title=self.title,
            description=self.description,
            affected_processes=self.affected_processes,
            metrics=self.metrics
        )


class StructuredRecommendation(BaseModel):
    """Pydantic model for recommendations with validation."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Brief recommendation title"
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Detailed recommendation description"
    )
    action_type: str = Field(
        ...,
        description="Type of action required",
        pattern="^(info|command|sudo_command|manual)$"
    )
    command: Optional[str] = Field(
        None,
        max_length=500,
        description="Command to execute (if applicable)"
    )
    risk_level: str = Field(
        default="low",
        description="Risk level of executing this recommendation",
        pattern="^(low|medium|high)$"
    )
    confirmation_required: bool = Field(
        default=True,
        description="Whether user confirmation is required"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority level (1=highest, 10=lowest)"
    )
    estimated_impact: str = Field(
        default="medium",
        description="Expected impact of this recommendation",
        pattern="^(low|medium|high)$"
    )
    
    @model_validator(mode='after')
    def validate_command_for_action_type(self):
        """Validate command based on action type."""
        if self.action_type in ['command', 'sudo_command'] and not self.command:
            raise ValueError(f"Command is required for action_type '{self.action_type}'")
        if self.action_type == 'info' and self.command:
            logger.warning("Command provided for info action_type, will be ignored")
        return self
    
    def to_interface_recommendation(self) -> Recommendation:
        """Convert to the interface Recommendation class."""
        return Recommendation(
            title=self.title,
            description=self.description,
            action_type=self.action_type,
            command=self.command,
            risk_level=self.risk_level,
            confirmation_required=self.confirmation_required
        )


class DiagnosticAnalysis(BaseModel):
    """Pydantic model for complete diagnostic analysis."""
    
    summary: str = Field(
        ...,
        min_length=50,
        max_length=2000,
        description="Overall system health summary"
    )
    issues: List[StructuredIssue] = Field(
        default_factory=list,
        description="List of detected issues"
    )
    recommendations: List[StructuredRecommendation] = Field(
        default_factory=list,
        description="List of actionable recommendations"
    )
    system_health_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall system health score (0.0-1.0)"
    )
    analysis_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis (0.0-1.0)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Analysis timestamp"
    )
    
    @field_validator('issues')
    @classmethod
    def validate_issues_count(cls, v):
        """Validate reasonable number of issues."""
        if len(v) > 50:  # Reasonable limit
            raise ValueError("Too many issues detected")
        return v
    
    @field_validator('recommendations')
    @classmethod
    def validate_recommendations_count(cls, v):
        """Validate reasonable number of recommendations."""
        if len(v) > 20:  # Reasonable limit
            raise ValueError("Too many recommendations provided")
        return v


class StructuredOutputParser:
    """Parser for structured LLM outputs using Pydantic models."""
    
    def __init__(self):
        """Initialize the structured output parser."""
        self.issue_parser = PydanticOutputParser(pydantic_object=StructuredIssue)
        self.recommendation_parser = PydanticOutputParser(pydantic_object=StructuredRecommendation)
        self.analysis_parser = PydanticOutputParser(pydantic_object=DiagnosticAnalysis)
        self.json_parser = JsonOutputParser()
    
    def create_analysis_prompt(self) -> PromptTemplate:
        """Create a prompt template for structured diagnostic analysis."""
        template = """You are a macOS system diagnostic expert. Analyze the provided system data and generate a structured diagnostic analysis.

System Data:
{system_data}

User Query: {query}

Please provide a comprehensive analysis in the following JSON format:
{format_instructions}

Guidelines:
- Be thorough but concise in your analysis
- Assign appropriate severity levels to issues
- Provide actionable recommendations with clear risk assessments
- Include confidence scores for your assessments
- Focus on the most impactful issues and solutions

Analysis:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["system_data", "query"],
            partial_variables={
                "format_instructions": self.analysis_parser.get_format_instructions()
            }
        )
    
    def create_recommendations_prompt(self) -> PromptTemplate:
        """Create a prompt template for structured recommendations."""
        template = """Based on the following diagnostic analysis, generate structured recommendations to address the identified issues.

Analysis: {analysis}

Please provide recommendations in the following JSON format:
{format_instructions}

Guidelines:
- Prioritize recommendations by impact and safety
- Provide clear, actionable steps
- Include appropriate risk levels and confirmation requirements
- Ensure commands are safe and well-tested
- Focus on the most effective solutions

Recommendations:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["analysis"],
            partial_variables={
                "format_instructions": self.recommendation_parser.get_format_instructions()
            }
        )
    
    def parse_diagnostic_analysis(self, llm_output: str) -> DiagnosticAnalysis:
        """Parse LLM output into a structured DiagnosticAnalysis.
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            DiagnosticAnalysis object
            
        Raises:
            ValidationError: If parsing or validation fails
        """
        try:
            # Clean the output
            cleaned_output = self._clean_json_output(llm_output)
            
            # Parse using Pydantic parser
            analysis = self.analysis_parser.parse(cleaned_output)
            
            logger.info(f"Successfully parsed diagnostic analysis with {len(analysis.issues)} issues and {len(analysis.recommendations)} recommendations")
            return analysis
            
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse diagnostic analysis: {e}")
            # Return a fallback analysis
            return self._create_fallback_analysis(llm_output, str(e))
    
    def parse_recommendations(self, llm_output: str) -> List[StructuredRecommendation]:
        """Parse LLM output into structured recommendations.
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            List of StructuredRecommendation objects
            
        Raises:
            ValidationError: If parsing or validation fails
        """
        try:
            # Clean the output
            cleaned_output = self._clean_json_output(llm_output)
            
            # Try to parse as a list of recommendations
            if cleaned_output.strip().startswith('['):
                # Parse as JSON array
                recommendations_data = json.loads(cleaned_output)
                recommendations = []
                
                for rec_data in recommendations_data:
                    try:
                        rec = StructuredRecommendation(**rec_data)
                        recommendations.append(rec)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid recommendation: {e}")
                        continue
                
                logger.info(f"Successfully parsed {len(recommendations)} recommendations")
                return recommendations
            else:
                # Try to parse as single recommendation
                rec = self.recommendation_parser.parse(cleaned_output)
                return [rec]
                
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse recommendations: {e}")
            # Return a fallback recommendation
            return [self._create_fallback_recommendation(str(e))]
    
    def parse_issues(self, llm_output: str) -> List[StructuredIssue]:
        """Parse LLM output into structured issues.
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            List of StructuredIssue objects
        """
        try:
            # Clean the output
            cleaned_output = self._clean_json_output(llm_output)
            
            # Try to parse as a list of issues
            if cleaned_output.strip().startswith('['):
                # Parse as JSON array
                issues_data = json.loads(cleaned_output)
                issues = []
                
                for issue_data in issues_data:
                    try:
                        issue = StructuredIssue(**issue_data)
                        issues.append(issue)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid issue: {e}")
                        continue
                
                logger.info(f"Successfully parsed {len(issues)} issues")
                return issues
            else:
                # Try to parse as single issue
                issue = self.issue_parser.parse(cleaned_output)
                return [issue]
                
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse issues: {e}")
            return []
    
    def _clean_json_output(self, output: str) -> str:
        """Clean LLM output to extract valid JSON."""
        # Remove markdown code blocks
        output = output.strip()
        if output.startswith('```json'):
            output = output[7:]
        elif output.startswith('```'):
            output = output[3:]
        
        if output.endswith('```'):
            output = output[:-3]
        
        # Find JSON content between braces or brackets
        output = output.strip()
        
        # Try to find the start and end of JSON
        start_chars = ['{', '[']
        end_chars = ['}', ']']
        
        start_idx = -1
        start_char = None
        for char in start_chars:
            idx = output.find(char)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx
                start_char = char
        
        if start_idx == -1:
            return output
        
        # Find matching end character
        end_char = '}' if start_char == '{' else ']'
        end_idx = output.rfind(end_char)
        
        if end_idx != -1 and end_idx > start_idx:
            output = output[start_idx:end_idx + 1]
        
        return output.strip()
    
    def _create_fallback_analysis(self, original_output: str, error_msg: str) -> DiagnosticAnalysis:
        """Create a fallback analysis when parsing fails."""
        return DiagnosticAnalysis(
            summary=f"Analysis completed but output parsing failed: {error_msg}. Raw output available for manual review.",
            issues=[
                StructuredIssue(
                    severity="medium",
                    category="system",
                    title="Output Parsing Error",
                    description=f"The diagnostic analysis completed but the output could not be parsed into structured format: {error_msg}",
                    confidence=0.9
                )
            ],
            recommendations=[
                StructuredRecommendation(
                    title="Review Raw Analysis",
                    description=f"Please review the raw analysis output manually: {original_output[:200]}...",
                    action_type="manual",
                    risk_level="low",
                    confirmation_required=False,
                    priority=8
                )
            ],
            system_health_score=0.5,
            analysis_confidence=0.3
        )
    
    def _create_fallback_recommendation(self, error_msg: str) -> StructuredRecommendation:
        """Create a fallback recommendation when parsing fails."""
        return StructuredRecommendation(
            title="Review System Analysis",
            description=f"The system analysis completed but recommendations could not be parsed: {error_msg}. Please review the analysis manually.",
            action_type="info",
            risk_level="low",
            confirmation_required=False,
            priority=9,
            estimated_impact="low"
        )
    
    def validate_and_sanitize_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize parsed output data.
        
        Args:
            data: Parsed data dictionary
            
        Returns:
            Sanitized and validated data dictionary
        """
        sanitized = {}
        
        # Sanitize strings
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized[key] = value.replace('\x00', '').strip()
            elif isinstance(value, list):
                sanitized[key] = [
                    item.replace('\x00', '').strip() if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                sanitized[key] = self.validate_and_sanitize_output(value)
            else:
                sanitized[key] = value
        
        return sanitized