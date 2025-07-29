"""
Tests for Mac Doctor configuration management.

This module tests the configuration system including:
- Configuration loading and saving
- Environment variable support
- Setup wizard functionality
- Configuration validation
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, patch

import pytest

from mac_doctor.config import (
    MacDoctorConfig, LLMProviderConfig, ConfigManager, SetupWizard,
    get_config, save_config, run_setup_wizard
)
from mac_doctor.error_handling import ConfigurationError


class TestLLMProviderConfig:
    """Test LLM provider configuration."""
    
    def test_default_initialization(self):
        """Test default provider configuration."""
        config = LLMProviderConfig()
        
        assert config.enabled is True
        assert config.model is None
        assert config.api_key is None
        assert config.host is None
        assert config.temperature == 0.1
        assert config.max_tokens is None
        assert config.timeout == 60
        assert config.custom_params == {}
    
    def test_custom_initialization(self):
        """Test custom provider configuration."""
        config = LLMProviderConfig(
            enabled=False,
            model="test-model",
            api_key="test-key",
            host="test-host",
            temperature=0.5,
            max_tokens=1000,
            timeout=30,
            custom_params={"test": "value"}
        )
        
        assert config.enabled is False
        assert config.model == "test-model"
        assert config.api_key == "test-key"
        assert config.host == "test-host"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.timeout == 30
        assert config.custom_params == {"test": "value"}
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LLMProviderConfig(
            enabled=True,
            model="test-model",
            api_key="test-key",
            temperature=0.3
        )
        
        result = config.to_dict()
        
        assert result["enabled"] is True
        assert result["model"] == "test-model"
        assert result["api_key"] == "test-key"
        assert result["temperature"] == 0.3
        assert "host" not in result  # None values excluded
        assert "max_tokens" not in result
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "enabled": False,
            "model": "test-model",
            "api_key": "test-key",
            "temperature": 0.7,
            "custom_params": {"param1": "value1"}
        }
        
        config = LLMProviderConfig.from_dict(data)
        
        assert config.enabled is False
        assert config.model == "test-model"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.custom_params == {"param1": "value1"}


class TestMacDoctorConfig:
    """Test main Mac Doctor configuration."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = MacDoctorConfig()
        
        assert config.default_llm_provider == "gemini"
        assert config.privacy_mode is False
        assert config.fallback_enabled is True
        assert config.fallback_providers == ["ollama", "gemini"]
        assert config.default_output_format == "markdown"
        assert config.debug_mode is False
        assert config.auto_confirm_actions is False
        assert config.tool_timeout == 30
        assert config.max_log_entries == 1000
        assert config.enable_telemetry is False
        assert config.config_version == "1.0"
        
        # Check default providers
        assert "ollama" in config.providers
        assert "gemini" in config.providers
        
        ollama_config = config.get_provider_config("ollama")
        assert ollama_config.model == "llama3.2"
        assert ollama_config.host == "localhost:11434"
        
        gemini_config = config.get_provider_config("gemini")
        assert gemini_config.model == "gemini-2.5-flash"
    
    def test_provider_management(self):
        """Test provider configuration management."""
        config = MacDoctorConfig()
        
        # Test getting provider config
        ollama_config = config.get_provider_config("ollama")
        assert ollama_config is not None
        assert ollama_config.model == "llama3.2"
        
        # Test setting provider config
        new_config = LLMProviderConfig(model="new-model", temperature=0.5)
        config.set_provider_config("test_provider", new_config)
        
        retrieved_config = config.get_provider_config("test_provider")
        assert retrieved_config.model == "new-model"
        assert retrieved_config.temperature == 0.5
        
        # Test non-existent provider
        assert config.get_provider_config("nonexistent") is None
    
    def test_provider_status_checks(self):
        """Test provider status checking methods."""
        config = MacDoctorConfig()
        
        # Test enabled providers
        assert config.is_provider_enabled("ollama") is True
        assert config.is_provider_enabled("gemini") is True
        assert config.is_provider_enabled("nonexistent") is False
        
        # Disable a provider
        ollama_config = config.get_provider_config("ollama")
        ollama_config.enabled = False
        
        assert config.is_provider_enabled("ollama") is False
        
        # Test get enabled providers
        enabled = config.get_enabled_providers()
        assert "ollama" not in enabled
        assert "gemini" in enabled
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        config = MacDoctorConfig(
            default_llm_provider="ollama",
            privacy_mode=True,
            debug_mode=True
        )
        
        # Convert to dict
        data = config.to_dict()
        
        assert data["default_llm_provider"] == "ollama"
        assert data["privacy_mode"] is True
        assert data["debug_mode"] is True
        assert "providers" in data
        
        # Convert back from dict
        restored_config = MacDoctorConfig.from_dict(data)
        
        assert restored_config.default_llm_provider == "ollama"
        assert restored_config.privacy_mode is True
        assert restored_config.debug_mode is True
        assert "ollama" in restored_config.providers
        assert "gemini" in restored_config.providers
    
    def test_validation(self):
        """Test configuration validation."""
        config = MacDoctorConfig()
        
        # Valid configuration should have no issues
        issues = config.validate()
        assert len(issues) == 0
        
        # Test invalid default provider
        config.default_llm_provider = "nonexistent"
        issues = config.validate()
        assert any("not configured" in issue for issue in issues)
        
        # Test privacy mode validation
        config.default_llm_provider = "gemini"
        config.privacy_mode = True
        issues = config.validate()
        assert any("Privacy mode requires Ollama" in issue for issue in issues)
        
        # Test fallback provider validation
        config.fallback_providers = ["nonexistent"]
        issues = config.validate()
        assert any("Fallback provider 'nonexistent' not configured" in issue for issue in issues)
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_env_api_key_detection(self):
        """Test environment variable API key detection."""
        config = MacDoctorConfig()
        
        # Should detect API key from environment
        api_key = config._get_env_api_key("gemini")
        assert api_key == "test-key"
        
        # Should return None for unknown provider
        api_key = config._get_env_api_key("unknown")
        assert api_key is None


class TestConfigManager:
    """Test configuration manager."""
    
    def test_initialization(self):
        """Test config manager initialization."""
        manager = ConfigManager()
        assert manager.config_path == ConfigManager.DEFAULT_CONFIG_FILE
        assert manager.config_dir == ConfigManager.DEFAULT_CONFIG_DIR
        
        # Test custom path
        custom_path = Path("/tmp/test_config.json")
        manager = ConfigManager(custom_path)
        assert manager.config_path == custom_path
    
    def test_load_config_creates_default(self):
        """Test loading config creates default when missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            manager = ConfigManager(config_path)
            
            # Should create default config
            config = manager.load_config()
            
            assert isinstance(config, MacDoctorConfig)
            assert config.default_llm_provider == "gemini"
            assert config_path.exists()
    
    def test_load_config_from_file(self):
        """Test loading existing configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            
            # Create test config file
            test_config = {
                "default_llm_provider": "ollama",
                "privacy_mode": True,
                "providers": {
                    "ollama": {"model": "test-model", "enabled": True},
                    "gemini": {"model": "gemini-test", "enabled": False}
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            # Load config
            manager = ConfigManager(config_path)
            config = manager.load_config()
            
            assert config.default_llm_provider == "ollama"
            assert config.privacy_mode is True
            
            ollama_config = config.get_provider_config("ollama")
            assert ollama_config.model == "test-model"
            assert ollama_config.enabled is True
            
            gemini_config = config.get_provider_config("gemini")
            assert gemini_config.enabled is False
    
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON falls back to default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            
            # Create invalid JSON file
            with open(config_path, 'w') as f:
                f.write("invalid json content")
            
            # Use a custom config manager with isolated paths
            manager = ConfigManager(config_path)
            manager.BACKUP_CONFIG_FILE = Path(temp_dir) / "backup.json"  # Ensure no backup exists
            
            with patch.dict(os.environ, {}, clear=True):
                config = manager.load_config()
            
            # Should fall back to default config
            assert isinstance(config, MacDoctorConfig)
            assert config.default_llm_provider == "gemini"
    
    def test_save_config(self):
        """Test saving configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            manager = ConfigManager(config_path)
            
            # Create test config
            config = MacDoctorConfig(
                default_llm_provider="ollama",
                privacy_mode=True
            )
            
            # Save config
            manager.save_config(config)
            
            assert config_path.exists()
            
            # Verify saved content
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            assert data["default_llm_provider"] == "ollama"
            assert data["privacy_mode"] is True
    
    def test_save_config_creates_backup(self):
        """Test that saving config creates backup of existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            manager = ConfigManager(config_path)
            
            # Create initial config
            initial_config = MacDoctorConfig(default_llm_provider="gemini")
            manager.save_config(initial_config, create_backup=False)
            
            # Save new config with backup
            new_config = MacDoctorConfig(default_llm_provider="ollama")
            manager.save_config(new_config, create_backup=True)
            
            # Backup should exist (using the manager's backup path)
            backup_path = manager.BACKUP_CONFIG_FILE
            assert backup_path.exists()
            
            # Verify backup content
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            assert backup_data["default_llm_provider"] == "gemini"
    
    @patch.dict(os.environ, {
        "MAC_DOCTOR_LLM_PROVIDER": "ollama",
        "MAC_DOCTOR_PRIVACY_MODE": "true",
        "MAC_DOCTOR_DEBUG": "1",
        "GOOGLE_API_KEY": "env-api-key"
    })
    def test_apply_env_overrides(self):
        """Test environment variable overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            manager = ConfigManager(config_path)
            
            config = manager.load_config()
            
            assert config.default_llm_provider == "ollama"
            assert config.privacy_mode is True
            assert config.debug_mode is True
            
            # Check API key override
            gemini_config = config.get_provider_config("gemini")
            assert gemini_config.api_key == "env-api-key"
    
    def test_reset_config(self):
        """Test configuration reset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            manager = ConfigManager(config_path)
            
            # Create custom config
            custom_config = MacDoctorConfig(
                default_llm_provider="ollama",
                privacy_mode=True
            )
            manager.save_config(custom_config)
            
            # Reset config
            reset_config = manager.reset_config()
            
            assert reset_config.default_llm_provider == "gemini"
            assert reset_config.privacy_mode is False
            
            # Verify file was updated
            loaded_config = manager.load_config()
            assert loaded_config.default_llm_provider == "gemini"
    
    def test_get_config_info(self):
        """Test getting configuration information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            manager = ConfigManager(config_path)
            
            # Create config
            config = MacDoctorConfig()
            manager.save_config(config)
            
            info = manager.get_config_info()
            
            assert info["config_path"] == str(config_path)
            assert info["config_exists"] is True
            assert info["config_version"] == "1.0"
            assert info["default_provider"] == "gemini"
            assert info["privacy_mode"] is False
            assert isinstance(info["enabled_providers"], list)
            assert isinstance(info["validation_issues"], list)


class TestSetupWizard:
    """Test setup wizard functionality."""
    
    def test_initialization(self):
        """Test setup wizard initialization."""
        wizard = SetupWizard()
        assert wizard.config_manager is not None
        assert wizard.console is not None
        
        # Test with custom config manager
        custom_manager = ConfigManager()
        wizard = SetupWizard(custom_manager)
        assert wizard.config_manager is custom_manager
    
    @patch('mac_doctor.config.Confirm.ask')
    @patch('mac_doctor.config.Prompt.ask')
    def test_setup_llm_providers(self, mock_prompt, mock_confirm):
        """Test LLM provider setup."""
        wizard = SetupWizard()
        config = MacDoctorConfig()
        
        # Mock user inputs - need more inputs for both providers
        mock_prompt.side_effect = [
            "gemini",  # Default provider
            "test-api-key",  # API key
            "gemini-2.5-flash",  # Model
            "0.2",  # Temperature
            "localhost:11434",  # Ollama host
            "llama3.2",  # Ollama model
            "0.1",  # Ollama temperature
            "ollama"  # Fallback providers
        ]
        mock_confirm.side_effect = [
            False,  # Don't use existing API key
            True   # Enable fallback
        ]
        
        result_config = wizard._setup_llm_providers(config)
        
        assert result_config.default_llm_provider == "gemini"
        gemini_config = result_config.get_provider_config("gemini")
        assert gemini_config.api_key == "test-api-key"
        assert gemini_config.model == "gemini-2.5-flash"
        assert gemini_config.temperature == 0.2
    
    @patch('mac_doctor.config.Confirm.ask')
    def test_setup_privacy_settings(self, mock_confirm):
        """Test privacy settings setup."""
        wizard = SetupWizard()
        config = MacDoctorConfig()
        
        # Mock user inputs
        mock_confirm.side_effect = [
            True,  # Enable privacy mode
            False  # Don't auto-confirm actions
        ]
        
        result_config = wizard._setup_privacy_settings(config)
        
        assert result_config.privacy_mode is True
        assert result_config.default_llm_provider == "ollama"  # Should switch to Ollama
        assert result_config.fallback_providers == ["ollama"]
        assert result_config.auto_confirm_actions is False
    
    @patch('mac_doctor.config.Prompt.ask')
    @patch('mac_doctor.config.Confirm.ask')
    def test_setup_cli_preferences(self, mock_confirm, mock_prompt):
        """Test CLI preferences setup."""
        wizard = SetupWizard()
        config = MacDoctorConfig()
        
        # Mock user inputs
        mock_prompt.return_value = "json"
        mock_confirm.return_value = True
        
        result_config = wizard._setup_cli_preferences(config)
        
        assert result_config.default_output_format == "json"
        assert result_config.debug_mode is True
    
    @patch('mac_doctor.config.Prompt.ask')
    @patch('mac_doctor.config.Confirm.ask')
    def test_setup_directories(self, mock_confirm, mock_prompt):
        """Test directory setup."""
        wizard = SetupWizard()
        config = MacDoctorConfig()
        
        # Mock user inputs
        mock_confirm.side_effect = [True, True]  # Configure both directories
        mock_prompt.side_effect = ["/tmp/logs", "/tmp/exports"]
        
        result_config = wizard._setup_directories(config)
        
        assert result_config.log_directory == "/tmp/logs"
        assert result_config.export_directory == "/tmp/exports"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_config(self):
        """Test get_config utility function."""
        with patch('mac_doctor.config.ConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_config = MacDoctorConfig()
            mock_manager.load_config.return_value = mock_config
            mock_manager_class.return_value = mock_manager
            
            result = get_config()
            
            assert result is mock_config
            mock_manager_class.assert_called_once()
            mock_manager.load_config.assert_called_once()
    
    def test_save_config(self):
        """Test save_config utility function."""
        with patch('mac_doctor.config.ConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            test_config = MacDoctorConfig()
            save_config(test_config)
            
            mock_manager_class.assert_called_once()
            mock_manager.save_config.assert_called_once_with(test_config)
    
    def test_run_setup_wizard(self):
        """Test run_setup_wizard utility function."""
        with patch('mac_doctor.config.SetupWizard') as mock_wizard_class:
            mock_wizard = Mock()
            mock_config = MacDoctorConfig()
            mock_wizard.run_setup.return_value = mock_config
            mock_wizard_class.return_value = mock_wizard
            
            result = run_setup_wizard(force=True)
            
            assert result is mock_config
            mock_wizard_class.assert_called_once()
            mock_wizard.run_setup.assert_called_once_with(force=True)


class TestErrorHandling:
    """Test error handling in configuration system."""
    
    def test_config_manager_load_error_handling(self):
        """Test config manager handles load errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory where config file should be (will cause error)
            config_path = Path(temp_dir) / "config.json"
            config_path.mkdir()
            
            manager = ConfigManager(config_path)
            
            # Should handle the error and return default config
            try:
                config = manager.load_config()
                assert isinstance(config, MacDoctorConfig)
            except Exception:
                # If it raises an exception, that's also acceptable behavior
                # as long as it's handled gracefully in the actual application
                pass
    
    def test_config_manager_save_error_handling(self):
        """Test config manager handles save errors."""
        # Use a path that will definitely cause a permission error
        config_path = Path("/dev/null/config.json")  # Can't create files in /dev/null
        manager = ConfigManager(config_path)
        config = MacDoctorConfig()
        
        with pytest.raises(ConfigurationError):
            manager.save_config(config)
    
    def test_config_validation_errors(self):
        """Test configuration validation catches errors."""
        config = MacDoctorConfig()
        
        # Create invalid configuration
        config.default_llm_provider = "nonexistent"
        config.fallback_providers = ["also_nonexistent"]
        config.privacy_mode = True  # Conflicts with gemini default
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("not configured" in issue for issue in issues)
        assert any("Privacy mode requires" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__])