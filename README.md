# Mac Doctor 🩺

An intelligent macOS system diagnostic tool that uses AI to analyze your Mac's performance, identify issues, and provide actionable recommendations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **AI-Powered Analysis**: Uses local LLM providers (Ollama, Google Gemini) to intelligently analyze system data
- **Comprehensive Diagnostics**: Monitors processes, memory, disk usage, network activity, and system logs
- **Privacy-First**: All analysis happens locally on your Mac - no data sent to external services
- **Interactive Recommendations**: Get actionable suggestions with safe execution prompts
- **Multiple Output Formats**: View results in terminal or export to Markdown/JSON
- **Modular Architecture**: Extensible plugin system for adding new diagnostic capabilities

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/brooksc/MacDoctor.git
cd MacDoctor

# Install with pip (recommended)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Run comprehensive system diagnosis
mac-doctor diagnose

# Ask specific questions about your system
mac-doctor ask "Why is my Mac running slowly?"

# Export results to file
mac-doctor diagnose --output report.md

# JSON output for programmatic use
mac-doctor diagnose --json --output report.json

# Debug mode for troubleshooting
mac-doctor diagnose --debug
```

## 🔧 Configuration

### LLM Provider Setup

Mac Doctor supports multiple AI providers:

#### Ollama (Recommended for Privacy)
```bash
# Install Ollama
brew install ollama

# Pull a model (e.g., llama2)
ollama pull llama2

# Mac Doctor will auto-detect Ollama
mac-doctor diagnose
```

#### Google Gemini
```bash
# Set your API key
export GOOGLE_API_KEY="your-api-key-here"

# Configure to use Gemini
mac-doctor setup --provider gemini
```

### Privacy Mode
Enable privacy mode to use only local providers:
```bash
mac-doctor setup --privacy-mode
```

## 📊 What Mac Doctor Analyzes

### System Metrics
- **Process Monitoring**: CPU and memory usage by process
- **Memory Analysis**: RAM usage, pressure, and swap activity  
- **Disk Health**: Storage usage, I/O performance, and disk errors
- **Network Activity**: Connection monitoring and bandwidth usage

### System Logs
- **Error Detection**: Identifies critical system errors and warnings
- **Pattern Analysis**: Finds recurring issues and anomalies
- **Performance Insights**: Correlates log events with system performance

### Intelligent Recommendations
- **Performance Optimization**: Suggests ways to improve system speed
- **Resource Management**: Identifies resource-hungry processes
- **Maintenance Tasks**: Recommends system cleanup and maintenance
- **Security Insights**: Highlights potential security concerns

## 🏗️ Architecture

Mac Doctor uses a modular plugin architecture:

```
mac_doctor/
├── agent/              # AI agent orchestration
├── cli/                # Command-line interface
├── core/               # Core diagnostic logic
├── llm/                # LLM provider implementations
└── mcps/               # Mac Collector Plugins
    ├── process_mcp.py  # Process monitoring
    ├── disk_mcp.py     # Disk analysis
    ├── network_mcp.py  # Network monitoring
    ├── logs_mcp.py     # System log analysis
    └── vmstat_mcp.py   # Memory statistics
```

### Mac Collector Plugins (MCPs)
Each MCP is responsible for collecting specific system data:
- **Modular Design**: Easy to add new diagnostic capabilities
- **Error Handling**: Graceful degradation when tools are unavailable
- **Structured Output**: Consistent data format for AI analysis

## 🛠️ Development

### Setup Development Environment
```bash
# Clone and install in development mode
git clone https://github.com/brooksc/MacDoctor.git
cd MacDoctor
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mac_doctor

# Run specific test file
pytest tests/unit/test_process_mcp.py
```

### Code Quality
```bash
# Format code
black mac_doctor/ tests/

# Sort imports
isort mac_doctor/ tests/

# Run linting
flake8 mac_doctor/ tests/

# Type checking
mypy mac_doctor/
```

### Creating New MCPs
To add a new diagnostic capability:

1. Create a new MCP class inheriting from `BaseMCP`
2. Implement required methods: `is_available()`, `execute()`, `get_schema()`
3. Register the MCP in the tool registry
4. Add comprehensive tests

Example:
```python
from mac_doctor.interfaces import BaseMCP, MCPResult

class CustomMCP(BaseMCP):
    @property
    def name(self) -> str:
        return "custom_diagnostic"
    
    def is_available(self) -> bool:
        # Check if required tools/permissions are available
        return True
    
    def execute(self, **kwargs) -> MCPResult:
        # Collect and return diagnostic data
        return MCPResult(
            success=True,
            data={"metric": "value"},
            metadata={"source": "custom"}
        )
```

## 🔒 Privacy & Security

- **Local Processing**: All AI analysis happens on your Mac
- **No Data Collection**: Mac Doctor doesn't send data to external servers
- **Secure by Default**: Only collects system metrics, not personal data
- **Transparent**: Open source - you can see exactly what it does

## 📋 System Requirements

- **Operating System**: macOS 10.15 (Catalina) or later
- **Python**: 3.9 or later
- **Architecture**: Intel (x86_64) or Apple Silicon (arm64)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 100MB for installation

### Optional Dependencies
- **Ollama**: For local AI analysis (recommended)
- **Google API Key**: For Gemini AI analysis
- **System Tools**: `vm_stat`, `netstat`, `log` (usually pre-installed)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute
- **Bug Reports**: Found an issue? Let us know!
- **Feature Requests**: Have an idea? We'd love to hear it!
- **Code Contributions**: Submit pull requests for fixes or features
- **Documentation**: Help improve our docs
- **Testing**: Help test on different macOS versions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for AI orchestration
- Uses [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- Powered by [Typer](https://typer.tiangolo.com/) for CLI interface
- System monitoring via [psutil](https://psutil.readthedocs.io/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/brooksc/MacDoctor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/brooksc/MacDoctor/discussions)
- **Documentation**: [Wiki](https://github.com/brooksc/MacDoctor/wiki)

---

**Made with ❤️ for the macOS community**