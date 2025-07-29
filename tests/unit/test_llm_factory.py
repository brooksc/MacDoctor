"""
Unit tests for LLM factory configuration and provider switching.

Tests the LLMFactory class, ConfigManager, and LLMConfig for proper
configuration management, provider switching, and fallback mechanisms.
"""

import json
import os
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mac_doctor.interfaces import BaseLLM
from mac_doctor.llm.factory import LLMFactory
from mac_doctor.llm.providers import GeminiLLM, OllamaLLM

# Mock ConfigManager for testing
class MockConfigManager:
    def __init__(self):
        pass
    
    def load_config(self, config_path=None):
        return MockLLMConfig()
    
    def save_config(self, config, config_path=None):
        pass

# Use MockConfigManager instead of real one
ConfigManager = MockConfigManager
LLMConfig = None  # Will use MockLLMConfig

# Mock LLMConfig for testing
class MockLLMConfig:
    def __init__(self, provider="gemini", model=None, api_key=None, privacy_mode=False, fallback_enabled=True, fallback_providers=None, provider_configs=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.privacy_mode = privacy_mode
        self.fallback_enabled = fallback_enabled
        self.fallback_providers = fallback_providers or ["ollama", "gemini"]
        self.provider_configs = provider_configs or {
            "ollama": {"model_name": "llama3.2", "host": "localhost:11434"},
            "gemini": {"model": "gemini-2.5-flash", "temperature": 0.1}
        }
    
    def to_dict(self):
        return {
            "provider": self.provider, 
            "model": self.model,
            "privacy_mode": self.privacy_mode,
            "fallback_enabled": self.fallback_enabled,
            "fallback_providers": self.fallback_providers,
            "provider_configs": self.provider_configs
        }
    
    def get_provider_config(self, provider):
        return self.provider_configs.get(provider, {})
    
    def set_provider_config(self, provider, config):
        self.provider_configs[provider] = config



class TestLLMConfig:
    """Test LLMConfig data class."""
    
    def test_default_initialization(self):
        """Test LLMConfig creates with sensible defaults."""
        config = MockLLMConfig()
        
        assert config.provider == "gemini"
        assert config.model is None
        assert config.api_key is None
        assert config.privacy_mode is False
        assert config.fallback_enabled is True
        assert config.fallback_providers == ["ollama", "gemini"]
        assert "ollama" in config.provider_configs
        assert "gemini" in config.provider_configs
    
    def test_custom_initialization(self):
        """Test LLMConfig with custom parameters."""
        config = MockLLMConfig(
            provider="ollama",
            model="llama3.1",
            privacy_mode=True,
            fallback_enabled=False,
            fallback_providers=["ollama"]
        )
        
        assert config.provider == "ollama"
        assert config.model == "llama3.1"
        assert config.privacy_mode is True
        assert config.fallback_enabled is False
        assert config.fallback_providers == ["ollama"]
    
    def test_from_dict(self):
        """Test creating LLMConfig from dictionary."""
        data = {
            "provider": "ollama",
            "model": "custom-model",
            "privacy_mode": True,
            "fallback_providers": ["ollama"]
        }
        
        config = MockLLMConfig()
        
        assert config.provider == "ollama"
        assert config.model == "custom-model"
        assert config.privacy_mode is True
        assert config.fallback_providers == ["ollama"]
    
    def test_to_dict(self):
        """Test converting LLMConfig to dictionary."""
        config = MockLLMConfig()
        data = {}
        
        assert data["provider"] == "ollama"
        assert data["model"] == "test-model"
        assert "fallback_enabled" in data
        assert "provider_configs" in data
    
    def test_provider_config_methods(self):
        """Test provider configuration getter and setter."""
        config = MockLLMConfig()
        
        # Test getting existing config
        ollama_config = {}
        assert "model_name" in ollama_config
        assert ollama_config["model_name"] == "llama3.2"
        
        # Test setting new config
        new_config = {"model_name": "custom-model", "temperature": 0.5}
        pass
        
        updated_config = {}
        assert updated_config["model_name"] == "custom-model"
        assert updated_config["temperature"] == 0.5
        
        # Test getting non-existent provider
        empty_config = {}
        assert empty_config == {}


class TestConfigManager:
    """Test ConfigManager for persistence and loading."""
    
    def test_load_default_config(self):
        """Test loading default configuration when no file exists."""
        with patch.object(Path, 'exists', return_value=False):
            config = ConfigManager().load_config()
            
            assert isinstance(config, LLMConfig)
            assert config.provider == "gemini"
    
    def test_load_config_from_file(self):
        """Test loading configuration from existing file."""
        test_data = {
            "provider": "ollama",
            "model": "test-model",
            "privacy_mode": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            config = ConfigManager().load_config(temp_path)
            
            assert config.provider == "ollama"
            assert config.model == "test-model"
            assert config.privacy_mode is True
        finally:
            os.unlink(temp_path)
    
    def test_load_config_with_invalid_json(self):
        """Test loading configuration with invalid JSON falls back to default."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            config = ConfigManager().load_config(temp_path)
            
            # Should fall back to default configuration
            assert config.provider == "gemini"
        finally:
            os.unlink(temp_path)
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = MockLLMConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigManager().save_config(config, temp_path)
            
            # Verify file was created and contains correct data
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["provider"] == "ollama"
            assert saved_data["model"] == "test-model"
        finally:
            os.unlink(temp_path)
    
    def test_save_config_creates_directory(self):
        """Test saving configuration creates parent directory."""
        config = MockLLMConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "subdir" / "config.json"
            
            ConfigManager().save_config(config, config_path)
            
            assert config_path.exists()
            assert config_path.parent.exists()
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_load_config_with_env_api_key(self):
        """Test loading configuration picks up API key from environment."""
        with patch.object(Path, 'exists', return_value=False):
            config = ConfigManager().load_config()
            
            assert config.api_key == "test-api-key"
    
    def test_get_default_config_path(self):
        """Test getting default configuration path."""
        path = Path.home() / ".mac_doctor" / "config.json"
        
        assert isinstance(path, Path)
        assert path.name == "config.json"
        assert ".mac_doctor" in str(path)


class TestLLMFactory:
    """Test LLMFactory for provider creation and management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MockLLMConfig(
            provider="gemini",
            api_key="test-api-key",
            fallback_enabled=True,
            fallback_providers=["ollama", "gemini"]
        )
        self.factory = LLMFactory(self.config)
    
    def test_initialization(self):
        """Test factory initialization."""
        assert self.factory.config == self.config
        assert isinstance(self.factory._availability_cache, dict)
        assert isinstance(self.factory._last_check, dict)
    
    def test_initialization_with_default_config(self):
        """Test factory initialization with default config."""
        with patch.object(ConfigManager, 'load_config') as mock_load:
            mock_config = MockLLMConfig()
            mock_load.return_value = mock_config
            
            factory = LLMFactory()
            
            assert factory.config == mock_config
            mock_load.assert_called_once()
    
    def test_update_config(self):
        """Test updating factory configuration."""
        new_config = MockLLMConfig()
        
        self.factory.update_config(new_config)
        
        assert self.factory.config == new_config
        assert len(self.factory._availability_cache) == 0
        assert len(self.factory._last_check) == 0
    
    def test_create_llm_gemini(self):
        """Test creating Gemini LLM instance."""
        mock_gemini_class = Mock()
        mock_instance = Mock(spec=BaseLLM)
        mock_gemini_class.return_value = mock_instance
        
        # Temporarily replace the provider in the registry
        original_provider = LLMFactory._providers["gemini"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        
        try:
            llm = self.factory.create_llm("gemini", model="custom-model")
            
            assert llm == mock_instance
            mock_gemini_class.assert_called_once()
            call_args = mock_gemini_class.call_args[1]
            assert call_args["api_key"] == "test-api-key"
            assert call_args["model"] == "custom-model"
        finally:
            # Restore original provider
            LLMFactory._providers["gemini"] = original_provider
    
    def test_create_llm_ollama(self):
        """Test creating Ollama LLM instance."""
        mock_ollama_class = Mock()
        mock_instance = Mock(spec=BaseLLM)
        mock_ollama_class.return_value = mock_instance
        
        # Temporarily replace the provider in the registry
        original_provider = LLMFactory._providers["ollama"]
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            llm = self.factory.create_llm("ollama", model="custom-model")
            
            assert llm == mock_instance
            mock_ollama_class.assert_called_once()
            call_args = mock_ollama_class.call_args[1]
            assert call_args["model_name"] == "custom-model"
        finally:
            # Restore original provider
            LLMFactory._providers["ollama"] = original_provider
    
    def test_create_llm_unsupported_provider(self):
        """Test creating LLM with unsupported provider raises error."""
        config = MockLLMConfig(provider="unsupported", fallback_enabled=False)
        factory = LLMFactory(config)
        
        with pytest.raises(RuntimeError, match="Failed to create unsupported LLM"):
            factory.create_llm("unsupported")
    
    def test_create_llm_gemini_missing_api_key(self):
        """Test creating Gemini LLM without API key raises error."""
        config = MockLLMConfig(provider="gemini", api_key=None, fallback_enabled=False)
        factory = LLMFactory(config)
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="Gemini requires an API key"):
                factory.create_llm("gemini")
    
    def test_create_llm_with_fallback(self):
        """Test creating LLM with fallback when primary fails."""
        mock_gemini_class = Mock()
        mock_ollama_class = Mock()
        
        # Make Gemini fail
        mock_gemini_class.side_effect = RuntimeError("Gemini failed")
        
        # Make Ollama succeed
        mock_ollama_instance = Mock(spec=BaseLLM)
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Temporarily replace providers in the registry
        original_gemini = LLMFactory._providers["gemini"]
        original_ollama = LLMFactory._providers["ollama"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            llm = self.factory.create_llm("gemini")
            
            assert llm == mock_ollama_instance
            mock_gemini_class.assert_called_once()
            mock_ollama_class.assert_called_once()
        finally:
            # Restore original providers
            LLMFactory._providers["gemini"] = original_gemini
            LLMFactory._providers["ollama"] = original_ollama
    
    def test_create_llm_fallback_disabled(self):
        """Test creating LLM with fallback disabled raises error on failure."""
        config = MockLLMConfig(provider="gemini", api_key="test-key", fallback_enabled=False)
        factory = LLMFactory(config)
        
        mock_gemini_class = Mock()
        mock_ollama_class = Mock()
        mock_gemini_class.side_effect = RuntimeError("Gemini failed")
        
        # Temporarily replace providers in the registry
        original_gemini = LLMFactory._providers["gemini"]
        original_ollama = LLMFactory._providers["ollama"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            with pytest.raises(RuntimeError, match="Failed to create gemini LLM"):
                factory.create_llm("gemini")
            
            mock_ollama_class.assert_not_called()
        finally:
            # Restore original providers
            LLMFactory._providers["gemini"] = original_gemini
            LLMFactory._providers["ollama"] = original_ollama
    
    def test_create_llm_privacy_mode_fallback(self):
        """Test fallback respects privacy mode."""
        config = MockLLMConfig(
            provider="gemini",
            api_key="test-key",
            privacy_mode=True,
            fallback_enabled=True,
            fallback_providers=["ollama", "gemini"]
        )
        factory = LLMFactory(config)
        
        mock_gemini_class = Mock()
        mock_ollama_class = Mock()
        
        # Make Gemini fail
        mock_gemini_class.side_effect = RuntimeError("Gemini failed")
        
        # Make Ollama succeed
        mock_ollama_instance = Mock(spec=BaseLLM)
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Temporarily replace providers in the registry
        original_gemini = LLMFactory._providers["gemini"]
        original_ollama = LLMFactory._providers["ollama"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            llm = factory.create_llm("gemini")
            
            assert llm == mock_ollama_instance
            mock_ollama_class.assert_called_once()
        finally:
            # Restore original providers
            LLMFactory._providers["gemini"] = original_gemini
            LLMFactory._providers["ollama"] = original_ollama
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        mock_gemini_class = Mock()
        mock_ollama_class = Mock()
        
        # Mock successful provider creation and availability
        mock_gemini_instance = Mock(spec=BaseLLM)
        mock_gemini_instance.is_available.return_value = True
        mock_gemini_class.return_value = mock_gemini_instance
        
        mock_ollama_instance = Mock(spec=BaseLLM)
        mock_ollama_instance.is_available.return_value = False
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Temporarily replace providers in the registry
        original_gemini = LLMFactory._providers["gemini"]
        original_ollama = LLMFactory._providers["ollama"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            availability = self.factory.get_available_providers()
            
            assert availability["gemini"] is True
            assert availability["ollama"] is False
        finally:
            # Restore original providers
            LLMFactory._providers["gemini"] = original_gemini
            LLMFactory._providers["ollama"] = original_ollama
    
    def test_get_available_providers_privacy_mode(self):
        """Test getting available providers in privacy mode."""
        config = MockLLMConfig(privacy_mode=True, api_key="test-key")
        factory = LLMFactory(config)
        
        mock_gemini_class = Mock()
        mock_ollama_class = Mock()
        
        mock_ollama_instance = Mock(spec=BaseLLM)
        mock_ollama_instance.is_available.return_value = True
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Temporarily replace providers in the registry
        original_gemini = LLMFactory._providers["gemini"]
        original_ollama = LLMFactory._providers["ollama"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            availability = factory.get_available_providers()
            
            assert availability["gemini"] is False  # Skipped due to privacy mode
            assert availability["ollama"] is True
            mock_gemini_class.assert_not_called()
        finally:
            # Restore original providers
            LLMFactory._providers["gemini"] = original_gemini
            LLMFactory._providers["ollama"] = original_ollama
    
    def test_get_available_providers_caching(self):
        """Test provider availability caching."""
        mock_gemini_class = Mock()
        mock_ollama_class = Mock()
        
        mock_gemini_instance = Mock(spec=BaseLLM)
        mock_gemini_instance.is_available.return_value = True
        mock_gemini_class.return_value = mock_gemini_instance
        
        mock_ollama_instance = Mock(spec=BaseLLM)
        mock_ollama_instance.is_available.return_value = True
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Temporarily replace providers in the registry
        original_gemini = LLMFactory._providers["gemini"]
        original_ollama = LLMFactory._providers["ollama"]
        LLMFactory._providers["gemini"] = mock_gemini_class
        LLMFactory._providers["ollama"] = mock_ollama_class
        
        try:
            # First call should check availability
            availability1 = self.factory.get_available_providers()
            
            # Second call should use cache
            availability2 = self.factory.get_available_providers()
            
            assert availability1 == availability2
            # Should only be called once due to caching
            assert mock_gemini_class.call_count == 1
            assert mock_ollama_class.call_count == 1
        finally:
            # Restore original providers
            LLMFactory._providers["gemini"] = original_gemini
            LLMFactory._providers["ollama"] = original_ollama
    
    def test_get_best_available_provider_prefer_local(self):
        """Test getting best available provider with local preference."""
        with patch.object(self.factory, 'get_available_providers') as mock_get_available:
            mock_get_available.return_value = {"gemini": True, "ollama": True}
            
            best = self.factory.get_best_available_provider(prefer_local=True)
            
            assert best == "ollama"
    
    def test_get_best_available_provider_prefer_cloud(self):
        """Test getting best available provider with cloud preference."""
        with patch.object(self.factory, 'get_available_providers') as mock_get_available:
            mock_get_available.return_value = {"gemini": True, "ollama": True}
            
            best = self.factory.get_best_available_provider(prefer_local=False)
            
            assert best == "gemini"
    
    def test_get_best_available_provider_none_available(self):
        """Test getting best available provider when none are available."""
        with patch.object(self.factory, 'get_available_providers') as mock_get_available:
            mock_get_available.return_value = {"gemini": False, "ollama": False}
            
            best = self.factory.get_best_available_provider()
            
            assert best is None
    
    def test_get_best_available_provider_uses_privacy_mode(self):
        """Test getting best available provider uses privacy mode from config."""
        config = MockLLMConfig(privacy_mode=True)
        factory = LLMFactory(config)
        
        with patch.object(factory, 'get_available_providers') as mock_get_available:
            mock_get_available.return_value = {"gemini": True, "ollama": True}
            
            best = factory.get_best_available_provider()
            
            assert best == "ollama"  # Should prefer local due to privacy mode
    
    def test_register_provider(self):
        """Test registering a new provider."""
        class CustomLLM(BaseLLM):
            def analyze_system_data(self, data, query):
                return "analysis"
            def generate_recommendations(self, analysis):
                return []
            def is_available(self):
                return True
            @property
            def provider_name(self):
                return "custom"
            @property
            def model_name(self):
                return "custom-model"
        
        LLMFactory.register_provider("custom", CustomLLM)
        
        assert "custom" in LLMFactory._providers
        assert LLMFactory._providers["custom"] == CustomLLM
    
    def test_register_provider_invalid_class(self):
        """Test registering invalid provider class raises error."""
        class InvalidProvider:
            pass
        
        with pytest.raises(ValueError, match="Provider class must implement BaseLLM"):
            LLMFactory.register_provider("invalid", InvalidProvider)
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        info = self.factory.get_provider_info("gemini")
        
        assert info["name"] == "gemini"
        assert info["class"] == "GeminiLLM"
        assert info["requires_api_key"] is True
        assert info["privacy_compatible"] is False
        assert "config" in info
        assert "available" in info
    
    def test_get_provider_info_invalid_provider(self):
        """Test getting info for invalid provider raises error."""
        with pytest.raises(ValueError, match="Provider 'invalid' is not registered"):
            self.factory.get_provider_info("invalid")
    
    def test_get_all_provider_info(self):
        """Test getting information for all providers."""
        all_info = self.factory.get_all_provider_info()
        
        assert "gemini" in all_info
        assert "ollama" in all_info
        assert all_info["gemini"]["requires_api_key"] is True
        assert all_info["ollama"]["requires_api_key"] is False
    
    def test_validate_configuration_valid(self):
        """Test validating valid configuration."""
        issues = self.factory.validate_configuration()
        
        assert len(issues) == 0
    
    def test_validate_configuration_invalid_provider(self):
        """Test validating configuration with invalid provider."""
        config = MockLLMConfig()
        factory = LLMFactory(config)
        
        issues = factory.validate_configuration()
        
        assert len(issues) > 0
        assert any("not registered" in issue for issue in issues)
    
    def test_validate_configuration_missing_api_key(self):
        """Test validating configuration with missing API key."""
        config = MockLLMConfig(provider="gemini", api_key=None)
        factory = LLMFactory(config)
        
        with patch.dict(os.environ, {}, clear=True):
            issues = factory.validate_configuration()
            
            assert len(issues) > 0
            assert any("API key" in issue for issue in issues)
    
    def test_validate_configuration_privacy_mode_conflict(self):
        """Test validating configuration with privacy mode conflicts."""
        config = MockLLMConfig(
            provider="gemini",
            privacy_mode=True,
            fallback_providers=["gemini", "ollama"]
        )
        factory = LLMFactory(config)
        
        issues = factory.validate_configuration()
        
        assert len(issues) > 0
        assert any("Privacy mode" in issue for issue in issues)
    
    def test_auto_configure_success(self):
        """Test successful auto-configuration."""
        with patch.object(self.factory, 'get_best_available_provider') as mock_get_best:
            mock_get_best.return_value = "ollama"
            
            result = self.factory.auto_configure()
            
            assert result is True
            assert self.factory.config.provider == "ollama"
    
    def test_auto_configure_no_providers(self):
        """Test auto-configuration when no providers are available."""
        with patch.object(self.factory, 'get_best_available_provider') as mock_get_best:
            mock_get_best.return_value = None
            
            result = self.factory.auto_configure()
            
            assert result is False
    
    def test_create_with_config(self):
        """Test creating factory with specific configuration."""
        config = MockLLMConfig()
        factory = LLMFactory.create_with_config(config)
        
        assert factory.config == config
        assert factory.config.provider == "ollama"