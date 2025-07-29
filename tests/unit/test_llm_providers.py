"""
Unit tests for LLM providers.

Tests the LangChain-based LLM implementations with mocked responses
to ensure proper functionality without requiring actual LLM services.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from langchain_core.messages import AIMessage
from mac_doctor.interfaces import Recommendation
from mac_doctor.llm.providers import OllamaLLM, GeminiLLM


class TestOllamaLLM:
    """Test cases for OllamaLLM provider."""
    
    def test_initialization(self):
        """Test OllamaLLM initialization with default parameters."""
        llm = OllamaLLM()
        
        assert llm.provider_name == "ollama"
        assert llm.model_name == "llama3.2"
        assert llm._host == "localhost:11434"
        assert llm._temperature == 0.1
        assert llm._timeout == 60
    
    def test_initialization_with_custom_params(self):
        """Test OllamaLLM initialization with custom parameters."""
        llm = OllamaLLM(
            model_name="llama3.1",
            host="custom-host:8080",
            temperature=0.5,
            timeout=120
        )
        
        assert llm.model_name == "llama3.1"
        assert llm._host == "custom-host:8080"
        assert llm._temperature == 0.5
        assert llm._timeout == 120
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_initialization_success(self, mock_chat_ollama):
        """Test successful LLM initialization."""
        mock_instance = Mock()
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        mock_chat_ollama.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.1,
            timeout=60
        )
        assert llm._llm == mock_instance
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_initialization_failure(self, mock_chat_ollama):
        """Test LLM initialization failure."""
        mock_chat_ollama.side_effect = Exception("Connection failed")
        
        llm = OllamaLLM()
        
        assert llm._llm is None
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_is_available_success(self, mock_chat_ollama):
        """Test availability check when LLM is available."""
        mock_instance = Mock()
        mock_response = AIMessage(content="Hello response")
        mock_instance.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        assert llm.is_available() is True
        mock_instance.invoke.assert_called_once()
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_is_available_failure(self, mock_chat_ollama):
        """Test availability check when LLM is not available."""
        mock_instance = Mock()
        mock_instance.invoke.side_effect = Exception("Service unavailable")
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        assert llm.is_available() is False
    
    def test_is_available_no_llm(self):
        """Test availability check when LLM is not initialized."""
        with patch('mac_doctor.llm.providers.ChatOllama') as mock_chat_ollama:
            mock_chat_ollama.side_effect = Exception("Init failed")
            llm = OllamaLLM()
            
            assert llm.is_available() is False
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_analyze_system_data_success(self, mock_chat_ollama):
        """Test successful system data analysis."""
        mock_instance = Mock()
        
        # Mock structured response first (which will fail), then text response
        structured_response = AIMessage(content="Invalid JSON response")
        text_response = AIMessage(content="System analysis: High CPU usage detected")
        
        # Need to provide enough responses for all the calls (availability check + actual calls)
        mock_instance.invoke.side_effect = [
            AIMessage(content="Hello"),  # Availability check
            structured_response,  # Structured parsing attempt
            text_response  # Text parsing fallback
        ]
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        test_data = {
            "process": {"success": True, "data": {"processes": [{"name": "test", "cpu_percent": 50}]}}
        }
        query = "Why is my system slow?"
        
        result = llm.analyze_system_data(test_data, query)
        
        # The result should be the fallback analysis since structured parsing failed
        assert "Analysis completed but output parsing failed" in result
        assert "System analysis: High CPU usage detected" in result
        assert mock_instance.invoke.call_count == 3  # Called three times (availability + structured + text)
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_analyze_system_data_unavailable(self, mock_chat_ollama):
        """Test system data analysis when LLM is unavailable."""
        mock_chat_ollama.side_effect = Exception("Init failed")
        llm = OllamaLLM()
        
        with pytest.raises(RuntimeError, match="Ollama LLM is not available"):
            llm.analyze_system_data({}, "test query")
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_analyze_system_data_invoke_failure(self, mock_chat_ollama):
        """Test system data analysis when invoke fails."""
        mock_instance = Mock()
        mock_instance.invoke.side_effect = Exception("API error")
        mock_chat_ollama.return_value = mock_instance
        
        # Mock is_available to return True
        with patch.object(OllamaLLM, 'is_available', return_value=True):
            llm = OllamaLLM()
            
            with pytest.raises(RuntimeError, match="Analysis failed"):
                llm.analyze_system_data({}, "test query")
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_generate_recommendations_success(self, mock_chat_ollama):
        """Test successful recommendation generation."""
        mock_instance = Mock()
        recommendations_json = json.dumps([
            {
                "title": "Restart high CPU process",
                "description": "Process X is using too much CPU",
                "action_type": "command",
                "command": "killall ProcessX",
                "risk_level": "medium",
                "confirmation_required": True
            }
        ])
        mock_response = AIMessage(content=f"```json\n{recommendations_json}\n```")
        mock_instance.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        analysis = "High CPU usage detected from ProcessX"
        recommendations = llm.generate_recommendations(analysis)
        
        assert len(recommendations) == 1
        rec = recommendations[0]
        assert isinstance(rec, Recommendation)
        assert rec.title == "Restart high CPU process"
        assert rec.description == "Process X is using too much CPU"
        assert rec.action_type == "command"
        assert rec.command == "killall ProcessX"
        assert rec.risk_level == "medium"
        assert rec.confirmation_required is True
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_generate_recommendations_malformed_json(self, mock_chat_ollama):
        """Test recommendation generation with malformed JSON response."""
        mock_instance = Mock()
        mock_response = AIMessage(content="Invalid JSON response")
        mock_instance.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        with patch.object(OllamaLLM, 'is_available', return_value=True):
            recommendations = llm.generate_recommendations("test analysis")
            
            # Should return fallback recommendation
            assert len(recommendations) == 1
            assert recommendations[0].title == "Review System Analysis"
            assert recommendations[0].action_type == "info"
    
    @patch('mac_doctor.llm.providers.ChatOllama')
    def test_generate_recommendations_partial_malformed(self, mock_chat_ollama):
        """Test recommendation generation with partially malformed data."""
        mock_instance = Mock()
        
        # First call for structured parsing (with invalid data), then fallback to text parsing
        structured_response = AIMessage(content=json.dumps([
            {
                "title": "Valid recommendation",
                "description": "This is valid",
                "action_type": "info"
            },
            {
                "title": "Partial recommendation",
                "description": "Missing some fields"
                # Missing action_type - will be skipped by structured parser
            }
        ]))
        
        # Fallback text parsing response
        text_response = AIMessage(content=json.dumps([
            {
                "title": "Valid recommendation",
                "description": "This is valid",
                "action_type": "info"
            }
        ]))
        
        mock_instance.invoke.side_effect = [structured_response, text_response]
        mock_chat_ollama.return_value = mock_instance
        
        llm = OllamaLLM()
        
        with patch.object(OllamaLLM, 'is_available', return_value=True):
            recommendations = llm.generate_recommendations("test analysis")
            
            # Should return only the valid recommendation (invalid one skipped)
            assert len(recommendations) == 1
            assert recommendations[0].title == "Valid recommendation"
            assert recommendations[0].action_type == "info"


class TestGeminiLLM:
    """Test cases for GeminiLLM provider."""
    
    def test_initialization(self):
        """Test GeminiLLM initialization with default parameters."""
        llm = GeminiLLM(api_key="test-key")
        
        assert llm.provider_name == "gemini"
        assert llm.model_name == "gemini-2.5-flash"
        assert llm._api_key == "test-key"
        assert llm._temperature == 0.1
        assert llm._max_tokens is None
    
    def test_initialization_with_custom_params(self):
        """Test GeminiLLM initialization with custom parameters."""
        llm = GeminiLLM(
            api_key="custom-key",
            model="gemini-pro",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert llm.model_name == "gemini-pro"
        assert llm._api_key == "custom-key"
        assert llm._temperature == 0.5
        assert llm._max_tokens == 1000
    
    @patch('mac_doctor.llm.providers.ChatGoogleGenerativeAI')
    def test_initialization_success(self, mock_chat_gemini):
        """Test successful LLM initialization."""
        mock_instance = Mock()
        mock_chat_gemini.return_value = mock_instance
        
        llm = GeminiLLM(api_key="test-key")
        
        mock_chat_gemini.assert_called_once_with(
            model="gemini-2.5-flash",
            google_api_key="test-key",
            temperature=0.1
        )
        assert llm._llm == mock_instance
    
    @patch('mac_doctor.llm.providers.ChatGoogleGenerativeAI')
    def test_initialization_with_max_tokens(self, mock_chat_gemini):
        """Test LLM initialization with max_tokens parameter."""
        mock_instance = Mock()
        mock_chat_gemini.return_value = mock_instance
        
        llm = GeminiLLM(api_key="test-key", max_tokens=500)
        
        mock_chat_gemini.assert_called_once_with(
            model="gemini-2.5-flash",
            google_api_key="test-key",
            temperature=0.1,
            max_tokens=500
        )
    
    @patch('mac_doctor.llm.providers.ChatGoogleGenerativeAI')
    def test_is_available_success(self, mock_chat_gemini):
        """Test availability check when LLM is available."""
        mock_instance = Mock()
        mock_response = AIMessage(content="Hello response")
        mock_instance.invoke.return_value = mock_response
        mock_chat_gemini.return_value = mock_instance
        
        llm = GeminiLLM(api_key="test-key")
        
        assert llm.is_available() is True
        mock_instance.invoke.assert_called_once()
    
    @patch('mac_doctor.llm.providers.ChatGoogleGenerativeAI')
    def test_analyze_system_data_privacy_safe(self, mock_chat_gemini):
        """Test that system data analysis uses privacy-safe summaries."""
        mock_instance = Mock()
        
        # Mock structured response first (which will fail), then text response
        structured_response = AIMessage(content="Invalid JSON response")
        text_response = AIMessage(content="Privacy-safe analysis")
        
        # Need to provide enough responses for all the calls (availability check + actual calls)
        mock_instance.invoke.side_effect = [
            AIMessage(content="Hello"),  # Availability check
            structured_response,  # Structured parsing attempt
            text_response  # Text parsing fallback
        ]
        mock_chat_gemini.return_value = mock_instance
        
        llm = GeminiLLM(api_key="test-key")
        
        # Test data with sensitive information
        test_data = {
            "process": {
                "success": True,
                "data": {
                    "processes": [
                        {"name": "sensitive_app", "cpu_percent": 15, "memory_percent": 10},
                        {"name": "normal_app", "cpu_percent": 5, "memory_percent": 2}
                    ]
                }
            },
            "vmstat": {
                "success": True,
                "data": {
                    "memory_pressure": "warn",
                    "swap_usage": {"used": 1000, "total": 2000}
                }
            }
        }
        
        result = llm.analyze_system_data(test_data, "System performance check")
        
        # The result should be the fallback analysis since structured parsing failed
        assert "Analysis completed but output parsing failed" in result
        assert "Privacy-safe analysis" in result
        assert mock_instance.invoke.call_count == 3  # Called three times (availability + structured + text)
    
    def test_create_privacy_safe_summary(self):
        """Test privacy-safe summary creation."""
        llm = GeminiLLM(api_key="test-key")
        
        test_data = {
            "process": {
                "success": True,
                "data": {
                    "processes": [
                        {"name": "app1", "cpu_percent": 15, "memory_percent": 8},
                        {"name": "app2", "cpu_percent": 5, "memory_percent": 3},
                        {"name": "app3", "cpu_percent": 2, "memory_percent": 1}
                    ]
                }
            },
            "disk": {
                "success": True,
                "data": {
                    "disk_usage": [{"usage_percent": 85}],
                    "io_stats": {"io_wait": 15}
                }
            },
            "failed_tool": {
                "success": False,
                "error": "Tool failed"
            }
        }
        
        summary = llm._create_privacy_safe_summary(test_data)
        summary_data = json.loads(summary)
        
        # Check process summary
        assert "process" in summary_data
        assert summary_data["process"]["high_cpu_processes"] == 1  # cpu > 10%
        assert summary_data["process"]["high_memory_processes"] == 1  # memory > 5%
        assert summary_data["process"]["total_processes"] == 3
        
        # Check disk summary
        assert "disk" in summary_data
        assert summary_data["disk"]["disk_usage_high"] is True  # usage > 80%
        assert summary_data["disk"]["io_wait_high"] is True  # io_wait > 10
        
        # Failed tools should not appear in summary
        assert "failed_tool" not in summary_data
        
        # Sensitive data should not be present
        assert "app1" not in summary
        assert "app2" not in summary
        assert "app3" not in summary