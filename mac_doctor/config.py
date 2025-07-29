"""
Configuration management for Mac Doctor.

This module provides comprehensive configuration management including:
- Persistent configuration file management
- Environment variable support
- First-run setup wizard
- Configuration validation and migration
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .error_handling import ConfigurationError, safe_execute

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider."""
    
    enabled: bool = True
    model: Optional[str] = None
    api_key: Optional[str] = None
    host: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 60
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMProviderConfig":
        """Create from dictionary."""
        # Handle custom_params separately
        custom_params = data.pop("custom_params", {})
        
        # Create instance with known fields
        instance = cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        instance.custom_params = custom_params
        
        return instance


@dataclass
class MacDoctorConfig:
    """Main configuration for Mac Doctor."""
    
    # LLM Configuration
    default_llm_provider: str = "gemini"
    privacy_mode: bool = False
    fallback_enabled: bool = True
    fallback_providers: List[str] = field(default_factory=lambda: ["ollama", "gemini"])
    
    # Provider-specific configurations
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    
    # CLI Configuration
    default_output_format: str = "markdown"
    debug_mode: bool = False
    auto_confirm_actions: bool = False
    
    # System Configuration
    tool_timeout: int = 30
    max_log_entries: int = 1000
    enable_telemetry: bool = False
    
    # Paths
    log_directory: Optional[str] = None
    export_directory: Optional[str] = None
    
    # Version tracking for migrations
    config_version: str = "1.0"
    
    def __post_init__(self):
        """Initialize default provider configurations."""
        if not self.providers:
            self.providers = {
                "ollama": LLMProviderConfig(
                    model="llama3.2",
                    host="localhost:11434",
                    temperature=0.1,
                    timeout=60,
                ),
                "gemini": LLMProviderConfig(
                    model="gemini-2.5-flash",
                    temperature=0.1,
                    max_tokens=None,
                ),
            }
    
    def get_provider_config(self, provider: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider)
    
    def set_provider_config(self, provider: str, config: LLMProviderConfig) -> None:
        """Set configuration for a specific provider."""
        self.providers[provider] = config
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled."""
        config = self.get_provider_config(provider)
        return config is not None and config.enabled
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers."""
        return [
            name for name, config in self.providers.items()
            if config.enabled
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert provider configs to dictionaries
        result["providers"] = {
            name: config.to_dict() 
            for name, config in self.providers.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacDoctorConfig":
        """Create from dictionary."""
        # Handle providers separately
        providers_data = data.pop("providers", {})
        providers = {
            name: LLMProviderConfig.from_dict(config_data)
            for name, config_data in providers_data.items()
        }
        
        # Create instance
        instance = cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        instance.providers = providers
        
        return instance
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check default provider exists and is enabled
        if self.default_llm_provider not in self.providers:
            issues.append(f"Default LLM provider '{self.default_llm_provider}' not configured")
        elif not self.is_provider_enabled(self.default_llm_provider):
            issues.append(f"Default LLM provider '{self.default_llm_provider}' is disabled")
        
        # Check fallback providers
        for provider in self.fallback_providers:
            if provider not in self.providers:
                issues.append(f"Fallback provider '{provider}' not configured")
        
        # Privacy mode validation
        if self.privacy_mode:
            if self.default_llm_provider != "ollama":
                issues.append("Privacy mode requires Ollama as default provider")
            
            non_local_fallbacks = [p for p in self.fallback_providers if p != "ollama"]
            if non_local_fallbacks:
                issues.append(f"Privacy mode incompatible with non-local fallbacks: {non_local_fallbacks}")
        
        # Check API keys for cloud providers
        for name, provider_config in self.providers.items():
            if name == "gemini" and provider_config.enabled:
                if not provider_config.api_key and not self._get_env_api_key("gemini"):
                    issues.append("Gemini provider requires API key")
        
        # Validate paths
        if self.log_directory and not Path(self.log_directory).parent.exists():
            issues.append(f"Log directory parent does not exist: {self.log_directory}")
        
        if self.export_directory and not Path(self.export_directory).parent.exists():
            issues.append(f"Export directory parent does not exist: {self.export_directory}")
        
        return issues
    
    def _get_env_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables."""
        if provider == "gemini":
            return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return None


class ConfigManager:
    """Manages Mac Doctor configuration persistence and loading."""
    
    DEFAULT_CONFIG_DIR = Path.home() / ".mac_doctor"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"
    BACKUP_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.backup.json"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Custom path to configuration file
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_FILE
        self.config_dir = self.config_path.parent
        self._config_cache: Optional[MacDoctorConfig] = None
    
    def load_config(self, create_if_missing: bool = True) -> MacDoctorConfig:
        """Load configuration from file.
        
        Args:
            create_if_missing: Create default config if file doesn't exist
            
        Returns:
            MacDoctorConfig instance
        """
        if self._config_cache is not None:
            return self._config_cache
        
        if not self.config_path.exists():
            if create_if_missing:
                logger.info("Configuration file not found, creating default")
                config = self._create_default_config()
                self.save_config(config)
                return config
            else:
                raise ConfigurationError(
                    config_item="config_file",
                    message=f"Configuration file not found: {self.config_path}",
                    suggestions=[
                        "Run 'mac-doctor setup' to create initial configuration",
                        "Check file path and permissions"
                    ]
                )
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            config = MacDoctorConfig.from_dict(data)
            
            # Apply environment variable overrides
            self._apply_env_overrides(config)
            
            # Validate configuration
            issues = config.validate()
            if issues:
                logger.warning(f"Configuration validation issues: {issues}")
            
            self._config_cache = config
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except (json.JSONDecodeError, KeyError, TypeError, OSError, IOError) as e:
            logger.error(f"Failed to load configuration: {e}")
            
            # Try to load backup
            if self.BACKUP_CONFIG_FILE.exists():
                logger.info("Attempting to load backup configuration")
                try:
                    with open(self.BACKUP_CONFIG_FILE, 'r') as f:
                        data = json.load(f)
                    config = MacDoctorConfig.from_dict(data)
                    self._apply_env_overrides(config)
                    logger.info("Successfully loaded backup configuration")
                    return config
                except Exception as backup_error:
                    logger.error(f"Backup configuration also failed: {backup_error}")
            
            # Fall back to default configuration
            logger.info("Using default configuration due to load failure")
            return self._create_default_config()
    
    def save_config(self, config: MacDoctorConfig, create_backup: bool = True) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            create_backup: Create backup of existing config
        """
        # Create directory if it doesn't exist
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, FileExistsError) as e:
            if not self.config_dir.is_dir():
                raise ConfigurationError(
                    config_item="config_directory",
                    message=f"Cannot create configuration directory: {e}",
                    suggestions=[
                        "Check directory permissions",
                        "Ensure parent directory exists and is writable",
                        "Use a different configuration path"
                    ]
                )
        
        # Create backup if requested and file exists
        if create_backup:
            try:
                if self.config_path.exists():
                    import shutil
                    shutil.copy2(self.config_path, self.BACKUP_CONFIG_FILE)
                    logger.debug("Created configuration backup")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        try:
            # Write to temporary file first
            temp_path = self.config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, sort_keys=True)
            
            # Atomic move to final location
            temp_path.replace(self.config_path)
            
            # Update cache
            self._config_cache = config
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except (OSError, TypeError) as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(
                config_item="config_file",
                message=f"Failed to save configuration: {e}",
                suggestions=[
                    "Check file permissions",
                    "Ensure directory exists and is writable",
                    "Check disk space"
                ]
            )
    
    def _create_default_config(self) -> MacDoctorConfig:
        """Create default configuration with environment overrides."""
        config = MacDoctorConfig()
        self._apply_env_overrides(config)
        return config
    
    def _apply_env_overrides(self, config: MacDoctorConfig) -> None:
        """Apply environment variable overrides to configuration."""
        # LLM Provider
        if env_provider := os.getenv("MAC_DOCTOR_LLM_PROVIDER"):
            config.default_llm_provider = env_provider
        
        # Privacy Mode
        if env_privacy := os.getenv("MAC_DOCTOR_PRIVACY_MODE"):
            config.privacy_mode = env_privacy.lower() in ("true", "1", "yes", "on")
        
        # Debug Mode
        if env_debug := os.getenv("MAC_DOCTOR_DEBUG"):
            config.debug_mode = env_debug.lower() in ("true", "1", "yes", "on")
        
        # Output Format
        if env_format := os.getenv("MAC_DOCTOR_OUTPUT_FORMAT"):
            config.default_output_format = env_format
        
        # API Keys
        for provider_name, provider_config in config.providers.items():
            if provider_name == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    provider_config.api_key = api_key
            
            # Provider-specific environment variables
            env_prefix = f"MAC_DOCTOR_{provider_name.upper()}_"
            
            if env_model := os.getenv(f"{env_prefix}MODEL"):
                provider_config.model = env_model
            
            if env_host := os.getenv(f"{env_prefix}HOST"):
                provider_config.host = env_host
            
            if env_temp := os.getenv(f"{env_prefix}TEMPERATURE"):
                try:
                    provider_config.temperature = float(env_temp)
                except ValueError:
                    logger.warning(f"Invalid temperature value: {env_temp}")
        
        # Directories
        if env_log_dir := os.getenv("MAC_DOCTOR_LOG_DIR"):
            config.log_directory = env_log_dir
        
        if env_export_dir := os.getenv("MAC_DOCTOR_EXPORT_DIR"):
            config.export_directory = env_export_dir
    
    def reset_config(self) -> MacDoctorConfig:
        """Reset configuration to defaults."""
        config = self._create_default_config()
        self.save_config(config)
        self._config_cache = None
        return config
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about current configuration."""
        config = self.load_config()
        
        return {
            "config_path": str(self.config_path),
            "config_exists": self.config_path.exists(),
            "backup_exists": self.BACKUP_CONFIG_FILE.exists(),
            "config_version": config.config_version,
            "default_provider": config.default_llm_provider,
            "privacy_mode": config.privacy_mode,
            "enabled_providers": config.get_enabled_providers(),
            "validation_issues": config.validate(),
        }
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache = None


class SetupWizard:
    """Interactive setup wizard for first-run configuration."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize setup wizard.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.console = Console()
    
    def run_setup(self, force: bool = False) -> MacDoctorConfig:
        """Run the interactive setup wizard.
        
        Args:
            force: Force setup even if configuration exists
            
        Returns:
            Configured MacDoctorConfig instance
        """
        self.console.print("\n[bold blue]🔧 Mac Doctor Setup Wizard[/bold blue]\n")
        
        # Check if configuration already exists
        if not force and self.config_manager.config_path.exists():
            self.console.print(f"[yellow]Configuration already exists at {self.config_manager.config_path}[/yellow]")
            if not Confirm.ask("Do you want to reconfigure?"):
                return self.config_manager.load_config()
        
        # Start with default or existing configuration
        if self.config_manager.config_path.exists() and not force:
            config = self.config_manager.load_config()
            self.console.print("[green]Loading existing configuration for modification[/green]")
        else:
            config = MacDoctorConfig()
            self.console.print("[green]Creating new configuration[/green]")
        
        # Run setup steps
        config = self._setup_llm_providers(config)
        config = self._setup_privacy_settings(config)
        config = self._setup_cli_preferences(config)
        config = self._setup_directories(config)
        
        # Validate final configuration
        issues = config.validate()
        if issues:
            self.console.print("\n[yellow]⚠️  Configuration validation issues:[/yellow]")
            for issue in issues:
                self.console.print(f"  • {issue}")
            
            if not Confirm.ask("\nContinue with this configuration?"):
                self.console.print("[red]Setup cancelled[/red]")
                sys.exit(1)
        
        # Save configuration
        self.config_manager.save_config(config)
        
        self.console.print("\n[green]✅ Configuration saved successfully![/green]")
        self._display_config_summary(config)
        
        return config
    
    def _setup_llm_providers(self, config: MacDoctorConfig) -> MacDoctorConfig:
        """Setup LLM provider configuration."""
        self.console.print("\n[bold]LLM Provider Configuration[/bold]")
        
        # Show available providers
        providers_table = Table(title="Available LLM Providers")
        providers_table.add_column("Provider", style="cyan")
        providers_table.add_column("Description", style="white")
        providers_table.add_column("Requirements", style="yellow")
        
        providers_table.add_row(
            "gemini", 
            "Google Gemini 2.5 Flash (cloud)", 
            "API key required"
        )
        providers_table.add_row(
            "ollama", 
            "Local Ollama models", 
            "Ollama installation required"
        )
        
        self.console.print(providers_table)
        
        # Select default provider
        default_provider = Prompt.ask(
            "\nSelect default LLM provider",
            choices=["gemini", "ollama"],
            default=config.default_llm_provider
        )
        config.default_llm_provider = default_provider
        
        # Configure Gemini if selected or available
        if default_provider == "gemini" or "gemini" in config.providers:
            config = self._setup_gemini_provider(config)
        
        # Configure Ollama if selected or available
        if default_provider == "ollama" or "ollama" in config.providers:
            config = self._setup_ollama_provider(config)
        
        # Setup fallback providers
        if Confirm.ask("\nEnable fallback providers?", default=config.fallback_enabled):
            config.fallback_enabled = True
            
            available_fallbacks = [p for p in ["gemini", "ollama"] if p != default_provider]
            if available_fallbacks:
                fallback_choices = Prompt.ask(
                    f"Select fallback providers (comma-separated)",
                    default=",".join([p for p in config.fallback_providers if p in available_fallbacks])
                )
                config.fallback_providers = [p.strip() for p in fallback_choices.split(",") if p.strip()]
        else:
            config.fallback_enabled = False
            config.fallback_providers = []
        
        return config
    
    def _setup_gemini_provider(self, config: MacDoctorConfig) -> MacDoctorConfig:
        """Setup Gemini provider configuration."""
        self.console.print("\n[bold cyan]Gemini Configuration[/bold cyan]")
        
        gemini_config = config.get_provider_config("gemini") or LLMProviderConfig()
        
        # API Key
        current_key = gemini_config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if current_key:
            self.console.print(f"[green]✅ API key found: {current_key[:8]}...[/green]")
            if not Confirm.ask("Use existing API key?"):
                current_key = None
        
        if not current_key:
            api_key = Prompt.ask(
                "Enter Gemini API key (or set GOOGLE_API_KEY environment variable)",
                password=True
            )
            if api_key:
                gemini_config.api_key = api_key
        
        # Model selection
        model = Prompt.ask(
            "Gemini model",
            default=gemini_config.model or "gemini-2.5-flash"
        )
        gemini_config.model = model
        
        # Temperature
        temp_str = Prompt.ask(
            "Temperature (0.0-1.0)",
            default=str(gemini_config.temperature)
        )
        try:
            gemini_config.temperature = float(temp_str)
        except ValueError:
            self.console.print("[yellow]Invalid temperature, using default[/yellow]")
        
        config.set_provider_config("gemini", gemini_config)
        return config
    
    def _setup_ollama_provider(self, config: MacDoctorConfig) -> MacDoctorConfig:
        """Setup Ollama provider configuration."""
        self.console.print("\n[bold cyan]Ollama Configuration[/bold cyan]")
        
        ollama_config = config.get_provider_config("ollama") or LLMProviderConfig()
        
        # Host
        host = Prompt.ask(
            "Ollama host",
            default=ollama_config.host or "localhost:11434"
        )
        ollama_config.host = host
        
        # Model
        model = Prompt.ask(
            "Ollama model",
            default=ollama_config.model or "llama3.2"
        )
        ollama_config.model = model
        
        # Temperature
        temp_str = Prompt.ask(
            "Temperature (0.0-1.0)",
            default=str(ollama_config.temperature)
        )
        try:
            ollama_config.temperature = float(temp_str)
        except ValueError:
            self.console.print("[yellow]Invalid temperature, using default[/yellow]")
        
        config.set_provider_config("ollama", ollama_config)
        return config
    
    def _setup_privacy_settings(self, config: MacDoctorConfig) -> MacDoctorConfig:
        """Setup privacy and security settings."""
        self.console.print("\n[bold]Privacy Settings[/bold]")
        
        privacy_mode = Confirm.ask(
            "Enable privacy mode? (local processing only)",
            default=config.privacy_mode
        )
        config.privacy_mode = privacy_mode
        
        if privacy_mode:
            self.console.print("[yellow]Privacy mode enabled - only local providers will be used[/yellow]")
            if config.default_llm_provider != "ollama":
                self.console.print("[yellow]Switching default provider to Ollama for privacy mode[/yellow]")
                config.default_llm_provider = "ollama"
            config.fallback_providers = ["ollama"]
        
        auto_confirm = Confirm.ask(
            "Auto-confirm safe actions? (reduces prompts)",
            default=config.auto_confirm_actions
        )
        config.auto_confirm_actions = auto_confirm
        
        return config
    
    def _setup_cli_preferences(self, config: MacDoctorConfig) -> MacDoctorConfig:
        """Setup CLI preferences."""
        self.console.print("\n[bold]CLI Preferences[/bold]")
        
        output_format = Prompt.ask(
            "Default output format",
            choices=["markdown", "json"],
            default=config.default_output_format
        )
        config.default_output_format = output_format
        
        debug_mode = Confirm.ask(
            "Enable debug mode by default?",
            default=config.debug_mode
        )
        config.debug_mode = debug_mode
        
        return config
    
    def _setup_directories(self, config: MacDoctorConfig) -> MacDoctorConfig:
        """Setup directory preferences."""
        self.console.print("\n[bold]Directory Configuration[/bold]")
        
        if Confirm.ask("Configure custom log directory?", default=bool(config.log_directory)):
            log_dir = Prompt.ask(
                "Log directory path",
                default=config.log_directory or str(Path.home() / ".mac_doctor" / "logs")
            )
            config.log_directory = log_dir
        
        if Confirm.ask("Configure custom export directory?", default=bool(config.export_directory)):
            export_dir = Prompt.ask(
                "Export directory path",
                default=config.export_directory or str(Path.home() / "Downloads")
            )
            config.export_directory = export_dir
        
        return config
    
    def _display_config_summary(self, config: MacDoctorConfig) -> None:
        """Display configuration summary."""
        self.console.print("\n[bold]Configuration Summary[/bold]")
        
        summary_table = Table()
        summary_table.add_column("Setting", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Default LLM Provider", config.default_llm_provider)
        summary_table.add_row("Privacy Mode", "✅" if config.privacy_mode else "❌")
        summary_table.add_row("Fallback Enabled", "✅" if config.fallback_enabled else "❌")
        summary_table.add_row("Fallback Providers", ", ".join(config.fallback_providers))
        summary_table.add_row("Output Format", config.default_output_format)
        summary_table.add_row("Debug Mode", "✅" if config.debug_mode else "❌")
        summary_table.add_row("Auto Confirm", "✅" if config.auto_confirm_actions else "❌")
        
        if config.log_directory:
            summary_table.add_row("Log Directory", config.log_directory)
        if config.export_directory:
            summary_table.add_row("Export Directory", config.export_directory)
        
        self.console.print(summary_table)
        
        # Show provider details
        self.console.print("\n[bold]Provider Details[/bold]")
        for name, provider_config in config.providers.items():
            if provider_config.enabled:
                self.console.print(f"[green]✅ {name.title()}[/green]: {provider_config.model}")
            else:
                self.console.print(f"[red]❌ {name.title()}[/red]: Disabled")


def get_config() -> MacDoctorConfig:
    """Get the current Mac Doctor configuration.
    
    Returns:
        MacDoctorConfig instance
    """
    manager = ConfigManager()
    return manager.load_config()


def save_config(config: MacDoctorConfig) -> None:
    """Save Mac Doctor configuration.
    
    Args:
        config: Configuration to save
    """
    manager = ConfigManager()
    manager.save_config(config)


def run_setup_wizard(force: bool = False) -> MacDoctorConfig:
    """Run the setup wizard.
    
    Args:
        force: Force setup even if configuration exists
        
    Returns:
        Configured MacDoctorConfig instance
    """
    wizard = SetupWizard()
    return wizard.run_setup(force=force)