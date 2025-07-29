"""
LLM provider implementations using LangChain.

This module provides LangChain-compatible LLM wrappers for both Ollama and Gemini
providers, implementing the BaseLLM interface for consistent usage across the system.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from ..interfaces import BaseLLM, Recommendation
from ..output_parsing import StructuredOutputParser, DiagnosticAnalysis

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """LangChain-based Ollama LLM provider for local model inference."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        host: str = "localhost:11434",
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        """Initialize Ollama LLM provider.
        
        Args:
            model_name: Name of the Ollama model to use
            host: Ollama server host and port
            temperature: Sampling temperature for generation
            timeout: Request timeout in seconds
        """
        self._model_name = model_name
        self._host = host
        self._temperature = temperature
        self._timeout = timeout
        self._llm = None
        self._output_parser = StructuredOutputParser()
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the LangChain ChatOllama instance."""
        try:
            self._llm = ChatOllama(
                model=self._model_name,
                base_url=f"http://{self._host}",
                temperature=self._temperature,
                timeout=self._timeout,
            )
            logger.info(f"Initialized Ollama LLM with model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            self._llm = None
    
    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return "ollama"
    
    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model_name
    
    def is_available(self) -> bool:
        """Check if Ollama is available and the model is accessible."""
        if self._llm is None:
            return False
        
        try:
            # Test with a simple message
            test_message = [HumanMessage(content="Hello")]
            response = self._llm.invoke(test_message)
            return bool(response.content)
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def analyze_system_data(self, data: Dict[str, Any], query: str) -> str:
        """Analyze system diagnostic data and return analysis.
        
        Args:
            data: Dictionary containing diagnostic data from various MCPs
            query: User's diagnostic query or question
            
        Returns:
            Analysis string describing findings and issues
        """
        if not self.is_available():
            raise RuntimeError("Ollama LLM is not available")
        
        try:
            # Use structured output parsing for analysis
            analysis = self.analyze_system_data_structured(data, query)
            return analysis.summary
        except Exception as e:
            logger.warning(f"Structured analysis failed, falling back to text analysis: {e}")
            # Fallback to original text-based analysis
            return self._analyze_system_data_text(data, query)
    
    def analyze_system_data_structured(self, data: Dict[str, Any], query: str) -> DiagnosticAnalysis:
        """Analyze system diagnostic data and return structured analysis.
        
        Args:
            data: Dictionary containing diagnostic data from various MCPs
            query: User's diagnostic query or question
            
        Returns:
            DiagnosticAnalysis object with structured results
        """
        if not self.is_available():
            raise RuntimeError("Ollama LLM is not available")
        
        # Create structured prompt
        prompt = self._output_parser.create_analysis_prompt()
        
        # Format system data for analysis
        system_data_str = json.dumps(data, indent=2, default=str)
        
        # Generate the prompt
        formatted_prompt = prompt.format(
            system_data=system_data_str,
            query=query
        )
        
        messages = [HumanMessage(content=formatted_prompt)]
        
        try:
            response = self._llm.invoke(messages)
            return self._output_parser.parse_diagnostic_analysis(response.content)
        except Exception as e:
            logger.error(f"Failed to analyze system data with structured parsing: {e}")
            raise RuntimeError(f"Structured analysis failed: {e}")
    
    def _analyze_system_data_text(self, data: Dict[str, Any], query: str) -> str:
        """Fallback text-based analysis method."""
        system_prompt = self._create_analysis_system_prompt()
        user_prompt = self._create_analysis_user_prompt(data, query)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to analyze system data with Ollama: {e}")
            raise RuntimeError(f"Analysis failed: {e}")
    
    def generate_recommendations(self, analysis: str) -> List[Recommendation]:
        """Generate actionable recommendations based on analysis.
        
        Args:
            analysis: Analysis text from analyze_system_data
            
        Returns:
            List of Recommendation objects
        """
        if not self.is_available():
            raise RuntimeError("Ollama LLM is not available")
        
        try:
            # Use structured output parsing for recommendations
            structured_recs = self.generate_recommendations_structured(analysis)
            return [rec.to_interface_recommendation() for rec in structured_recs]
        except Exception as e:
            logger.warning(f"Structured recommendations failed, falling back to text parsing: {e}")
            # Fallback to original text-based parsing
            return self._generate_recommendations_text(analysis)
    
    def generate_recommendations_structured(self, analysis: str) -> List:
        """Generate structured recommendations based on analysis.
        
        Args:
            analysis: Analysis text from analyze_system_data
            
        Returns:
            List of StructuredRecommendation objects
        """
        if not self.is_available():
            raise RuntimeError("Ollama LLM is not available")
        
        # Create structured prompt
        prompt = self._output_parser.create_recommendations_prompt()
        
        # Generate the prompt
        formatted_prompt = prompt.format(analysis=analysis)
        
        messages = [HumanMessage(content=formatted_prompt)]
        
        try:
            response = self._llm.invoke(messages)
            return self._output_parser.parse_recommendations(response.content)
        except Exception as e:
            logger.error(f"Failed to generate structured recommendations: {e}")
            raise RuntimeError(f"Structured recommendation generation failed: {e}")
    
    def _generate_recommendations_text(self, analysis: str) -> List[Recommendation]:
        """Fallback text-based recommendation generation."""
        system_prompt = self._create_recommendations_system_prompt()
        user_prompt = self._create_recommendations_user_prompt(analysis)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self._llm.invoke(messages)
            return self._parse_recommendations(response.content)
        except Exception as e:
            logger.error(f"Failed to generate recommendations with Ollama: {e}")
            raise RuntimeError(f"Recommendation generation failed: {e}")
    
    def _create_analysis_system_prompt(self) -> str:
        """Create system prompt for diagnostic analysis."""
        return """You are a macOS system diagnostic expert. Analyze the provided system data and identify performance issues, bottlenecks, and potential problems.

Focus on:
- CPU usage patterns and high-consuming processes
- Memory pressure and swap usage
- Disk I/O bottlenecks and space issues
- Network connectivity problems
- System log errors and warnings
- Thermal throttling indicators

Provide a clear, technical analysis that explains what the data shows and what issues are present."""
    
    def _create_analysis_user_prompt(self, data: Dict[str, Any], query: str) -> str:
        """Create user prompt for analysis with diagnostic data."""
        data_summary = json.dumps(data, indent=2, default=str)
        return f"""User Query: {query}

System Diagnostic Data:
{data_summary}

Please analyze this diagnostic data and provide a comprehensive assessment of the system's health and performance. Identify any issues or concerns."""
    
    def _create_recommendations_system_prompt(self) -> str:
        """Create system prompt for generating recommendations."""
        return """You are a macOS system administrator providing actionable recommendations. Based on the diagnostic analysis, generate specific, safe recommendations that users can follow to resolve issues.

Each recommendation should be:
- Specific and actionable
- Safe to execute
- Clearly explained with rationale
- Categorized by risk level (low/medium/high)

Format your response as a JSON array of recommendations with this structure:
[
  {
    "title": "Brief title",
    "description": "Detailed explanation",
    "action_type": "info|command|sudo_command",
    "command": "actual command if applicable",
    "risk_level": "low|medium|high",
    "confirmation_required": true|false
  }
]"""
    
    def _create_recommendations_user_prompt(self, analysis: str) -> str:
        """Create user prompt for recommendations based on analysis."""
        return f"""Based on this diagnostic analysis, please provide actionable recommendations:

{analysis}

Generate specific recommendations that address the identified issues. Return only the JSON array of recommendations."""
    
    def _parse_recommendations(self, response_content: str) -> List[Recommendation]:
        """Parse LLM response into Recommendation objects."""
        try:
            # Try to extract JSON from the response
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            recommendations_data = json.loads(content)
            recommendations = []
            
            for rec_data in recommendations_data:
                try:
                    recommendation = Recommendation(
                        title=rec_data.get("title", ""),
                        description=rec_data.get("description", ""),
                        action_type=rec_data.get("action_type", "info"),
                        command=rec_data.get("command"),
                        risk_level=rec_data.get("risk_level", "low"),
                        confirmation_required=rec_data.get("confirmation_required", True)
                    )
                    recommendations.append(recommendation)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping malformed recommendation: {e}")
                    continue
            
            return recommendations
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse recommendations: {e}")
            # Return a fallback recommendation
            return [
                Recommendation(
                    title="Review System Analysis",
                    description="The system analysis completed but recommendations could not be parsed. Please review the analysis manually.",
                    action_type="info",
                    risk_level="low",
                    confirmation_required=False
                )
            ]


class GeminiLLM(BaseLLM):
    """LangChain-based Gemini LLM provider for cloud-based inference."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ):
        """Initialize Gemini LLM provider.
        
        Args:
            api_key: Google AI API key
            model: Gemini model name to use
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate (None for default)
        """
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llm = None
        self._output_parser = StructuredOutputParser()
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the LangChain ChatGoogleGenerativeAI instance."""
        try:
            # Validate model to ensure only Gemini 2.5+ models are used
            self._validate_model()
            
            kwargs = {
                "model": self._model,
                "google_api_key": self._api_key,
                "temperature": self._temperature,
            }
            if self._max_tokens is not None:
                kwargs["max_tokens"] = self._max_tokens
            
            self._llm = ChatGoogleGenerativeAI(**kwargs)
            logger.info(f"Initialized Gemini LLM with model: {self._model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            self._llm = None
    
    def _validate_model(self) -> None:
        """Validate that the model is a supported Gemini 2.5+ version."""
        supported_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-exp",
            "gemini-2.5-pro-exp"
        ]
        
        if self._model not in supported_models:
            # Check if it's an older version
            if self._model.startswith(("gemini-1.", "gemini-2.0", "gemini-pro", "gemini-flash")):
                raise ValueError(
                    f"Model '{self._model}' is not supported. Mac Doctor only supports "
                    f"Gemini 2.5+ models: {', '.join(supported_models)}"
                )
            elif not self._model.startswith("gemini-"):
                raise ValueError(f"Invalid Gemini model: {self._model}")
            else:
                # Allow newer models that might be released
                logger.warning(f"Using unrecognized Gemini model: {self._model}")
                logger.warning("This model may not be tested. Supported models: " + ", ".join(supported_models))
    
    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return "gemini"
    
    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model
    
    def is_available(self) -> bool:
        """Check if Gemini is available and API key is valid."""
        if self._llm is None:
            return False
        
        try:
            # Test with a simple message
            test_message = [HumanMessage(content="Hello")]
            response = self._llm.invoke(test_message)
            return bool(response.content)
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            return False
    
    def analyze_system_data(self, data: Dict[str, Any], query: str) -> str:
        """Analyze system diagnostic data and return analysis.
        
        Args:
            data: Dictionary containing diagnostic data from various MCPs
            query: User's diagnostic query or question
            
        Returns:
            Analysis string describing findings and issues
        """
        if not self.is_available():
            raise RuntimeError("Gemini LLM is not available")
        
        try:
            # Use structured output parsing for analysis
            analysis = self.analyze_system_data_structured(data, query)
            return analysis.summary
        except Exception as e:
            logger.warning(f"Structured analysis failed, falling back to text analysis: {e}")
            # Fallback to original text-based analysis
            return self._analyze_system_data_text(data, query)
    
    def analyze_system_data_structured(self, data: Dict[str, Any], query: str) -> DiagnosticAnalysis:
        """Analyze system diagnostic data and return structured analysis.
        
        Args:
            data: Dictionary containing diagnostic data from various MCPs
            query: User's diagnostic query or question
            
        Returns:
            DiagnosticAnalysis object with structured results
        """
        if not self.is_available():
            raise RuntimeError("Gemini LLM is not available")
        
        # Create structured prompt
        prompt = self._output_parser.create_analysis_prompt()
        
        # Format system data for analysis (privacy-safe for Gemini)
        system_data_str = self._create_privacy_safe_summary(data)
        
        # Generate the prompt
        formatted_prompt = prompt.format(
            system_data=system_data_str,
            query=query
        )
        
        messages = [HumanMessage(content=formatted_prompt)]
        
        try:
            response = self._llm.invoke(messages)
            return self._output_parser.parse_diagnostic_analysis(response.content)
        except Exception as e:
            logger.error(f"Failed to analyze system data with structured parsing: {e}")
            raise RuntimeError(f"Structured analysis failed: {e}")
    
    def _analyze_system_data_text(self, data: Dict[str, Any], query: str) -> str:
        """Fallback text-based analysis method."""
        system_prompt = self._create_analysis_system_prompt()
        user_prompt = self._create_analysis_user_prompt(data, query)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to analyze system data with Gemini: {e}")
            raise RuntimeError(f"Analysis failed: {e}")
    
    def generate_recommendations(self, analysis: str) -> List[Recommendation]:
        """Generate actionable recommendations based on analysis.
        
        Args:
            analysis: Analysis text from analyze_system_data
            
        Returns:
            List of Recommendation objects
        """
        if not self.is_available():
            raise RuntimeError("Gemini LLM is not available")
        
        try:
            # Use structured output parsing for recommendations
            structured_recs = self.generate_recommendations_structured(analysis)
            return [rec.to_interface_recommendation() for rec in structured_recs]
        except Exception as e:
            logger.warning(f"Structured recommendations failed, falling back to text parsing: {e}")
            # Fallback to original text-based parsing
            return self._generate_recommendations_text(analysis)
    
    def generate_recommendations_structured(self, analysis: str) -> List:
        """Generate structured recommendations based on analysis.
        
        Args:
            analysis: Analysis text from analyze_system_data
            
        Returns:
            List of StructuredRecommendation objects
        """
        if not self.is_available():
            raise RuntimeError("Gemini LLM is not available")
        
        # Create structured prompt
        prompt = self._output_parser.create_recommendations_prompt()
        
        # Generate the prompt
        formatted_prompt = prompt.format(analysis=analysis)
        
        messages = [HumanMessage(content=formatted_prompt)]
        
        try:
            response = self._llm.invoke(messages)
            return self._output_parser.parse_recommendations(response.content)
        except Exception as e:
            logger.error(f"Failed to generate structured recommendations: {e}")
            raise RuntimeError(f"Structured recommendation generation failed: {e}")
    
    def _generate_recommendations_text(self, analysis: str) -> List[Recommendation]:
        """Fallback text-based recommendation generation."""
        system_prompt = self._create_recommendations_system_prompt()
        user_prompt = self._create_recommendations_user_prompt(analysis)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self._llm.invoke(messages)
            return self._parse_recommendations(response.content)
        except Exception as e:
            logger.error(f"Failed to generate recommendations with Gemini: {e}")
            raise RuntimeError(f"Recommendation generation failed: {e}")
    
    def _create_analysis_system_prompt(self) -> str:
        """Create system prompt for diagnostic analysis."""
        return """You are a macOS system diagnostic expert. Analyze the provided system data and identify performance issues, bottlenecks, and potential problems.

Focus on:
- CPU usage patterns and high-consuming processes
- Memory pressure and swap usage
- Disk I/O bottlenecks and space issues
- Network connectivity problems
- System log errors and warnings
- Thermal throttling indicators

Provide a clear, technical analysis that explains what the data shows and what issues are present."""
    
    def _create_analysis_user_prompt(self, data: Dict[str, Any], query: str) -> str:
        """Create user prompt for analysis with diagnostic data."""
        # For privacy, we'll send aggregated summaries rather than raw data
        data_summary = self._create_privacy_safe_summary(data)
        return f"""User Query: {query}

System Diagnostic Summary:
{data_summary}

Please analyze this diagnostic summary and provide a comprehensive assessment of the system's health and performance. Identify any issues or concerns."""
    
    def _create_privacy_safe_summary(self, data: Dict[str, Any]) -> str:
        """Create a privacy-safe summary of diagnostic data for remote processing."""
        summary = {}
        
        for tool_name, result in data.items():
            if not isinstance(result, dict) or not result.get("success", False):
                continue
            
            tool_data = result.get("data", {})
            
            # Create aggregated summaries without sensitive details
            if tool_name == "process":
                summary[tool_name] = {
                    "high_cpu_processes": len([p for p in tool_data.get("processes", []) if p.get("cpu_percent", 0) > 10]),
                    "high_memory_processes": len([p for p in tool_data.get("processes", []) if p.get("memory_percent", 0) > 5]),
                    "total_processes": len(tool_data.get("processes", [])),
                }
            elif tool_name == "vmstat":
                summary[tool_name] = {
                    "memory_pressure": tool_data.get("memory_pressure", "normal"),
                    "swap_usage": tool_data.get("swap_usage", {}),
                }
            elif tool_name == "disk":
                summary[tool_name] = {
                    "disk_usage_high": any(d.get("usage_percent", 0) > 80 for d in tool_data.get("disk_usage", [])),
                    "io_wait_high": tool_data.get("io_stats", {}).get("io_wait", 0) > 10,
                }
            elif tool_name == "network":
                summary[tool_name] = {
                    "active_connections": len(tool_data.get("connections", [])),
                    "high_traffic": tool_data.get("traffic_summary", {}).get("total_bytes", 0) > 1000000,
                }
            elif tool_name == "logs":
                summary[tool_name] = {
                    "error_count": len(tool_data.get("errors", [])),
                    "warning_count": len(tool_data.get("warnings", [])),
                }
        
        return json.dumps(summary, indent=2)
    
    def _create_recommendations_system_prompt(self) -> str:
        """Create system prompt for generating recommendations."""
        return """You are a macOS system administrator providing actionable recommendations. Based on the diagnostic analysis, generate specific, safe recommendations that users can follow to resolve issues.

Each recommendation should be:
- Specific and actionable
- Safe to execute
- Clearly explained with rationale
- Categorized by risk level (low/medium/high)

Format your response as a JSON array of recommendations with this structure:
[
  {
    "title": "Brief title",
    "description": "Detailed explanation",
    "action_type": "info|command|sudo_command",
    "command": "actual command if applicable",
    "risk_level": "low|medium|high",
    "confirmation_required": true|false
  }
]"""
    
    def _create_recommendations_user_prompt(self, analysis: str) -> str:
        """Create user prompt for recommendations based on analysis."""
        return f"""Based on this diagnostic analysis, please provide actionable recommendations:

{analysis}

Generate specific recommendations that address the identified issues. Return only the JSON array of recommendations."""
    
    def _parse_recommendations(self, response_content: str) -> List[Recommendation]:
        """Parse LLM response into Recommendation objects."""
        try:
            # Try to extract JSON from the response
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            recommendations_data = json.loads(content)
            recommendations = []
            
            for rec_data in recommendations_data:
                try:
                    recommendation = Recommendation(
                        title=rec_data.get("title", ""),
                        description=rec_data.get("description", ""),
                        action_type=rec_data.get("action_type", "info"),
                        command=rec_data.get("command"),
                        risk_level=rec_data.get("risk_level", "low"),
                        confirmation_required=rec_data.get("confirmation_required", True)
                    )
                    recommendations.append(recommendation)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping malformed recommendation: {e}")
                    continue
            
            return recommendations
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse recommendations: {e}")
            # Return a fallback recommendation
            return [
                Recommendation(
                    title="Review System Analysis",
                    description="The system analysis completed but recommendations could not be parsed. Please review the analysis manually.",
                    action_type="info",
                    risk_level="low",
                    confirmation_required=False
                )
            ]