"""
LLM factory for unified configuration and model selection.

This module provides a factory pattern for creating LLM instances with
unified configuration management across different providers.
"""

import logging
import os
from typing import Dict, List, Optional, Type

from ..interfaces import BaseLLM
from .providers import GeminiLLM, OllamaLLM
from ..error_handling import (
    ErrorHandler, ConfigurationError, LLMError, DependencyError,
    safe_execute
)
from ..config import MacDoctorConfig, ConfigManager

logger = logging.getLogger(__name__)


# Legacy LLMConfig class for backward compatibility
class LLMConfig:
    """Legacy LLM configuration class for backward compatibility."""
    
    def __init__(self, mac_doctor_config: MacDoctorConfig):
        """Initialize from MacDoctorConfig."""
        self._config = mac_doctor_config
    
    @property
    def provider(self) -> str:
        return self._config.default_llm_provider
    
    @property
    def model(self) -> Optional[str]:
        provider_config = self._config.get_provider_config(self.provider)
        return provider_config.model if provider_config else None
    
    @property
    def api_key(self) -> Optional[str]:
        provider_config = self._config.get_provider_config(self.provider)
        return provider_config.api_key if provider_config else None
    
    @property
    def privacy_mode(self) -> bool:
        return self._config.privacy_mode
    
    @property
    def fallback_enabled(self) -> bool:
        return self._config.fallback_enabled
    
    @property
    def fallback_providers(self) -> List[str]:
        return self._config.fallback_providers
    
    def get_provider_config(self, provider: str) -> Dict[str, any]:
        """Get configuration for a specific provider."""
        provider_config = self._config.get_provider_config(provider)
        if not provider_config:
            return {}
        
        # Convert to legacy format
        config_dict = provider_config.to_dict()
        
        # Handle provider-specific mappings
        if provider == "ollama":
            if "model" in config_dict:
                config_dict["model_name"] = config_dict.pop("model")
        
        return config_dict


# Legacy ConfigManager for backward compatibility
class LegacyConfigManager:
    """Legacy configuration manager for backward compatibility."""
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> LLMConfig:
        """Load configuration and return legacy LLMConfig."""
        manager = ConfigManager(config_path)
        mac_doctor_config = manager.load_config()
        return LLMConfig(mac_doctor_config)


class LLMFactory:
    """Factory for creating LLM instances with unified configuration."""
    
    # Registry of available LLM providers
    _providers: Dict[str, Type[BaseLLM]] = {
        "ollama": OllamaLLM,
        "gemini": GeminiLLM,
    }
    
    def __init__(self, config: Optional[MacDoctorConfig] = None, error_handler: Optional[ErrorHandler] = None):
        """Initialize LLM factory with configuration.
        
        Args:
            config: Mac Doctor configuration (loads default if None)
            error_handler: Error handler for comprehensive error management
        """
        if config is None:
            manager = ConfigManager()
            self._config = manager.load_config()
        else:
            self._config = config
        
        self.error_handler = error_handler or ErrorHandler()
        self._availability_cache: Dict[str, bool] = {}
        self._cache_timeout = 300  # 5 minutes
        self._last_check: Dict[str, float] = {}
    
    @property
    def config(self) -> MacDoctorConfig:
        """Get current configuration."""
        return self._config
    
    def update_config(self, config: MacDoctorConfig) -> None:
        """Update factory configuration.
        
        Args:
            config: New Mac Doctor configuration
        """
        self._config = config
        # Clear availability cache when config changes
        self._availability_cache.clear()
        self._last_check.clear()
    
    def create_llm(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM instance with fallback support.
        
        Args:
            provider: LLM provider name (uses config default if None)
            model: Model name to use (uses config default if None)
            api_key: API key for cloud providers (uses config default if None)
            **kwargs: Additional provider-specific configuration
            
        Returns:
            Configured LLM instance
            
        Raises:
            RuntimeError: If no providers are available
        """
        # Use config defaults if not specified
        target_provider = provider or self._config.default_llm_provider
        
        provider_config = self._config.get_provider_config(target_provider)
        target_model = model or (provider_config.model if provider_config else None)
        target_api_key = api_key or (provider_config.api_key if provider_config else None)
        
        # Try primary provider first with comprehensive error handling
        def _create_primary():
            return self._create_provider_instance(
                target_provider, target_model, target_api_key, **kwargs
            )
        
        result = safe_execute(
            _create_primary,
            error_handler=self.error_handler,
            context={
                "operation": "llm_creation",
                "provider": target_provider,
                "model": target_model
            },
            fallback_result=None,
            show_errors=False  # We'll handle fallback first
        )
        
        if result is not None:
            return result
        
        # Try fallback providers if enabled
        if self._config.fallback_enabled:
            return self._create_with_fallback(
                target_provider, target_model, target_api_key, **kwargs
            )
        else:
            raise LLMError(
                provider=target_provider,
                message="Failed to create LLM instance and fallback is disabled",
                suggestions=[
                    "Check provider configuration",
                    "Enable fallback providers",
                    "Verify API keys and credentials"
                ]
            )
    
    def _create_provider_instance(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an instance of a specific provider.
        
        Args:
            provider: Provider name
            model: Model name
            api_key: API key
            **kwargs: Additional configuration
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If provider configuration is invalid
        """
        if provider not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ConfigurationError(
                config_item="llm_provider",
                message=f"Unsupported provider '{provider}'. Available: {available}",
                suggestions=[
                    f"Use one of the supported providers: {available}",
                    "Check provider name spelling",
                    "Register custom provider if needed"
                ]
            )
        
        provider_class = self._providers[provider]
        
        # Get provider configuration
        provider_config = self._config.get_provider_config(provider)
        if provider_config:
            config = provider_config.to_dict()
            # Remove configuration parameters that aren't needed by the provider classes
            config.pop('enabled', None)
            config.pop('custom_params', None)
        else:
            config = {}
        
        config.update(kwargs)
        
        # Handle provider-specific configuration with error handling
        if provider == "ollama":
            # Filter to only Ollama-specific parameters
            ollama_config = {}
            if model:
                ollama_config["model_name"] = model
            elif "model" in config:
                ollama_config["model_name"] = config["model"]
            
            if "host" in config:
                ollama_config["host"] = config["host"]
            if "temperature" in config:
                ollama_config["temperature"] = config["temperature"]
            if "timeout" in config:
                ollama_config["timeout"] = config["timeout"]
            
            def _create_ollama():
                return provider_class(**ollama_config)
            
            return safe_execute(
                _create_ollama,
                error_handler=self.error_handler,
                context={"provider": "ollama", "config": ollama_config},
                fallback_result=None,
                show_errors=True
            ) or self._handle_provider_creation_failure(provider, ollama_config)
        
        elif provider == "gemini":
            # API key is required for Gemini
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    config_item="gemini_api_key",
                    message="Gemini requires an API key",
                    suggestions=[
                        "Set GOOGLE_API_KEY environment variable",
                        "Set GEMINI_API_KEY environment variable",
                        "Pass api_key parameter to create_llm()",
                        "Configure API key in LLM configuration file"
                    ]
                )
            
            # Filter to only Gemini-specific parameters
            gemini_config = {"api_key": api_key}
            if model:
                gemini_config["model"] = model
            elif "model" in config:
                gemini_config["model"] = config["model"]
            
            if "temperature" in config:
                gemini_config["temperature"] = config["temperature"]
            if "max_tokens" in config:
                gemini_config["max_tokens"] = config["max_tokens"]
            
            def _create_gemini():
                return provider_class(**gemini_config)
            
            return safe_execute(
                _create_gemini,
                error_handler=self.error_handler,
                context={"provider": "gemini", "config": {k: v for k, v in gemini_config.items() if k != "api_key"}},
                fallback_result=None,
                show_errors=True
            ) or self._handle_provider_creation_failure(provider, gemini_config)
        
        else:
            # Handle custom providers - just pass all config as-is
            if model:
                config["model"] = model
            if api_key:
                config["api_key"] = api_key
            
            def _create_custom():
                return provider_class(**config)
            
            return safe_execute(
                _create_custom,
                error_handler=self.error_handler,
                context={"provider": provider, "config": config},
                fallback_result=None,
                show_errors=True
            ) or self._handle_provider_creation_failure(provider, config)
    
    def _handle_provider_creation_failure(self, provider: str, config: Dict[str, any]) -> None:
        """Handle provider creation failure with specific error."""
        raise LLMError(
            provider=provider,
            message="Failed to create provider instance",
            suggestions=[
                f"Check {provider} provider configuration",
                "Verify all required parameters are provided",
                "Check network connectivity for remote providers",
                "Ensure provider dependencies are installed"
            ]
        )
    
    def _create_with_fallback(
        self,
        failed_provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create LLM instance with fallback providers.
        
        Args:
            failed_provider: Provider that failed
            model: Model name
            api_key: API key
            **kwargs: Additional configuration
            
        Returns:
            LLM instance from fallback provider
            
        Raises:
            RuntimeError: If all fallback providers fail
        """
        fallback_providers = [
            p for p in self._config.fallback_providers 
            if p != failed_provider and p in self._providers
        ]
        
        if not fallback_providers:
            raise RuntimeError(f"No fallback providers available after {failed_provider} failed")
        
        last_error = None
        for fallback_provider in fallback_providers:
            try:
                logger.info(f"Trying fallback provider: {fallback_provider}")
                
                # For privacy mode, only use local providers
                if self._config.privacy_mode and fallback_provider != "ollama":
                    logger.info(f"Skipping {fallback_provider} due to privacy mode")
                    continue
                
                return self._create_provider_instance(
                    fallback_provider, model, api_key, **kwargs
                )
            except Exception as e:
                logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError(f"All fallback providers failed. Last error: {last_error}")
    
    @classmethod
    def create_with_config(cls, config: MacDoctorConfig) -> "LLMFactory":
        """Create factory instance with specific configuration.
        
        Args:
            config: Mac Doctor configuration
            
        Returns:
            LLMFactory instance
        """
        return cls(config)
    
    def get_available_providers(self, force_check: bool = False) -> Dict[str, bool]:
        """Get availability status of all registered providers.
        
        Args:
            force_check: Force availability check even if cached
            
        Returns:
            Dictionary mapping provider names to availability status
        """
        import time
        
        current_time = time.time()
        availability = {}
        
        for provider_name in self._providers:
            # Check cache first
            if (not force_check and 
                provider_name in self._availability_cache and
                current_time - self._last_check.get(provider_name, 0) < self._cache_timeout):
                availability[provider_name] = self._availability_cache[provider_name]
                continue
            
            # Perform availability check
            try:
                # For privacy mode, only check local providers
                if self._config.privacy_mode and provider_name != "ollama":
                    availability[provider_name] = False
                    self._availability_cache[provider_name] = False
                    self._last_check[provider_name] = current_time
                    continue
                
                # Try to create and test the provider
                if provider_name == "ollama":
                    llm = self._create_provider_instance(provider_name)
                elif provider_name == "gemini":
                    # Skip Gemini if no API key is available
                    provider_config = self._config.get_provider_config("gemini")
                    api_key = (provider_config.api_key if provider_config else None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        availability[provider_name] = False
                        self._availability_cache[provider_name] = False
                        self._last_check[provider_name] = current_time
                        continue
                    llm = self._create_provider_instance(provider_name, api_key=api_key)
                else:
                    llm = self._create_provider_instance(provider_name)
                
                is_available = llm.is_available()
                availability[provider_name] = is_available
                self._availability_cache[provider_name] = is_available
                self._last_check[provider_name] = current_time
                
            except Exception as e:
                logger.debug(f"Provider {provider_name} not available: {e}")
                availability[provider_name] = False
                self._availability_cache[provider_name] = False
                self._last_check[provider_name] = current_time
        
        return availability
    
    def get_best_available_provider(self, prefer_local: Optional[bool] = None) -> Optional[str]:
        """Get the best available provider based on preferences.
        
        Args:
            prefer_local: If True, prefer local providers over cloud providers.
                         If None, uses config privacy_mode setting.
            
        Returns:
            Name of the best available provider, or None if none available
        """
        if prefer_local is None:
            prefer_local = self._config.privacy_mode
        
        availability = self.get_available_providers()
        available_providers = [name for name, available in availability.items() if available]
        
        if not available_providers:
            return None
        
        # Priority order based on preference
        if prefer_local:
            priority_order = ["ollama", "gemini"]
        else:
            priority_order = ["gemini", "ollama"]
        
        # Return the first available provider in priority order
        for provider in priority_order:
            if provider in available_providers:
                return provider
        
        # Fallback to any available provider
        return available_providers[0]
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """Register a new LLM provider.
        
        Args:
            name: Provider name
            provider_class: Provider class implementing BaseLLM
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError("Provider class must implement BaseLLM interface")
        
        cls._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    def get_provider_info(self, provider: str) -> Dict[str, any]:
        """Get information about a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with provider information
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider not in self._providers:
            raise ValueError(f"Provider '{provider}' is not registered")
        
        provider_class = self._providers[provider]
        provider_config = self._config.get_provider_config(provider)
        availability = self.get_available_providers()
        
        return {
            "name": provider,
            "class": provider_class.__name__,
            "module": provider_class.__module__,
            "config": provider_config.to_dict() if provider_config else {},
            "requires_api_key": provider == "gemini",
            "available": availability.get(provider, False),
            "privacy_compatible": provider == "ollama",
        }
    
    def get_all_provider_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about all registered providers.
        
        Returns:
            Dictionary mapping provider names to their information
        """
        return {
            provider: self.get_provider_info(provider)
            for provider in self._providers
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check if primary provider is registered
        if self._config.default_llm_provider not in self._providers:
            issues.append(f"Primary provider '{self._config.default_llm_provider}' is not registered")
        
        # Check fallback providers
        for provider in self._config.fallback_providers:
            if provider not in self._providers:
                issues.append(f"Fallback provider '{provider}' is not registered")
        
        # Check API key for Gemini if it's the primary or fallback provider
        providers_to_check = [self._config.default_llm_provider] + self._config.fallback_providers
        if "gemini" in providers_to_check:
            gemini_config = self._config.get_provider_config("gemini")
            api_key = (gemini_config.api_key if gemini_config else None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                issues.append("Gemini provider requires an API key (GOOGLE_API_KEY or GEMINI_API_KEY environment variable or config)")
        
        # Check privacy mode compatibility
        if self._config.privacy_mode:
            if self._config.default_llm_provider != "ollama":
                issues.append(f"Privacy mode requires local provider, but primary is '{self._config.default_llm_provider}'")
            
            non_local_fallbacks = [p for p in self._config.fallback_providers if p != "ollama"]
            if non_local_fallbacks:
                issues.append(f"Privacy mode incompatible with non-local fallback providers: {non_local_fallbacks}")
        
        return issues
    
    def auto_configure(self) -> bool:
        """Automatically configure the factory based on available providers.
        
        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            best_provider = self.get_best_available_provider()
            if not best_provider:
                logger.error("No LLM providers are available")
                return False
            
            # Update configuration with best available provider
            self._config.default_llm_provider = best_provider
            logger.info(f"Auto-configured to use provider: {best_provider}")
            return True
            
        except Exception as e:
            logger.error(f"Auto-configuration failed: {e}")
            return False
# Backward compatibility aliases
ConfigManager = LegacyConfigManager