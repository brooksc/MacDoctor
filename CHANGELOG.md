# Changelog

All notable changes to Mac Doctor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-27

### Added
- Initial release of Mac Doctor
- AI-powered macOS system diagnostics using local LLM providers
- Support for Ollama and Google Gemini LLM providers
- Comprehensive system monitoring with Mac Collector Plugins (MCPs):
  - Process monitoring with CPU and memory usage analysis
  - Memory analysis including pressure monitoring and swap activity
  - Disk health monitoring with I/O performance and usage statistics
  - Network activity monitoring with connection analysis
  - System log analysis with error pattern detection
  - VM statistics monitoring for memory management insights
- Rich CLI interface with multiple output formats (terminal, Markdown, JSON)
- Privacy-first design with local processing capabilities
- Interactive setup wizard for LLM provider configuration
- Comprehensive error handling and structured logging
- Modular architecture supporting plugin extensibility
- Action engine for safe execution of system recommendations
- Extensive test coverage (82%) with unit and integration tests
- Production-ready configuration management
- System validation for compatibility checking

### Features
- **Intelligent Analysis**: Uses AI to correlate system metrics and identify root causes
- **Privacy Mode**: Option to use only local LLM providers (Ollama)
- **Interactive Recommendations**: Safe execution of suggested system improvements
- **Structured Output**: Machine-readable JSON format for automation
- **Debug Mode**: Comprehensive logging for troubleshooting
- **Cross-Architecture**: Support for both Intel and Apple Silicon Macs
- **Graceful Degradation**: Continues operation when some tools are unavailable

### Technical Details
- Python 3.9+ support with modern async/await patterns
- LangChain integration for AI agent orchestration
- Rich terminal UI with progress indicators and colored output
- Pydantic models for robust data validation
- Comprehensive error handling with context preservation
- Modular plugin system for easy extensibility
- Type hints throughout codebase for better maintainability

### Documentation
- Comprehensive README with installation and usage instructions
- Contributing guidelines for open source development
- Architecture documentation explaining the plugin system
- API documentation for developers
- Example configurations and use cases

### Security
- No sensitive data collection or transmission
- API keys handled securely through environment variables
- Local processing ensures data privacy
- Secure defaults for all configuration options