# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mac Doctor is an agentic AI assistant for macOS system diagnostics. It uses LangChain-based agents to dynamically select and execute diagnostic tools (MCPs - Mac Collector Plugins) to analyze system performance and health issues. The project leverages local LLM providers (Gemini/Ollama) for analysis and provides actionable recommendations while maintaining privacy through local processing.

## Development Commands

### Core Development
```bash
# Install dependencies (development mode)
uv pip install -e ".[dev]"

# Run the main CLI application
mac-doctor diagnose                    # Full system diagnostic
mac-doctor ask "question"              # Ask specific questions
mac-doctor list-tools                  # List available diagnostic tools
mac-doctor trace                       # Run with detailed execution traces
mac-doctor system-check                # Check system compatibility

# Test commands
pytest                                 # Run all tests
pytest tests/unit/                     # Run unit tests only
pytest tests/e2e/                      # Run e2e tests only
pytest --cov=mac_doctor                # Run with coverage

# Code quality
black .                                # Format code
isort .                                # Sort imports  
flake8 .                               # Lint code
mypy mac_doctor                        # Type checking
```

## Architecture

### Core Components

1. **Agent Framework** (`mac_doctor/agent/`)
   - `DiagnosticAgent`: LangChain-based ReAct agent that orchestrates tool execution
   - Uses structured prompts to plan diagnostic workflows
   - Converts MCP tools to LangChain tools for seamless integration

2. **MCP System** (`mac_doctor/mcps/`)
   - **Process MCP**: CPU, memory, and process analysis using psutil and ps
   - **Disk MCP**: Storage analysis using iostat, df, and du
   - **Network MCP**: Network activity analysis with nettop and netstat
   - **VMstat MCP**: Memory pressure and paging behavior
   - **Logs MCP**: System log analysis
   - **DTrace MCP**: System call tracing and performance debugging

3. **LLM Integration** (`mac_doctor/llm/`)
   - `LLMFactory`: Provider abstraction supporting Ollama and Gemini
   - `ConfigManager`: Handles LLM configuration and fallback providers
   - Privacy-first design with local processing emphasis

4. **Core Services** (`mac_doctor/core/`)
   - `ActionEngine`: Safe execution of recommended actions with confirmations
   - `ReportGenerator`: Markdown and JSON report generation with export capabilities

5. **CLI Interface** (`mac_doctor/cli/`)
   - Typer-based CLI with rich terminal output
   - Multiple modes: diagnose, ask, trace, system-check, config management

### Tool Registry Pattern

The `ToolRegistry` dynamically manages MCP tools:
- Tools self-register their capabilities and availability
- Agent queries the registry to plan execution based on user queries
- Each tool implements the `BaseMCP` interface with standardized execution and output

### Agent Workflow

1. User query → Agent planning using ReAct pattern
2. Tool selection based on query keywords and available tools
3. Sequential tool execution with intermediate result analysis
4. LLM analysis of aggregated diagnostic data
5. Issue detection and recommendation generation
6. Structured output with actionable insights

## Key Files

- `mac_doctor/cli/main.py:54` - Main CLI entry point with command definitions
- `mac_doctor/agent/diagnostic_agent.py:138` - Core agent analysis method
- `mac_doctor/tool_registry.py:35` - Tool selection logic
- `mac_doctor/interfaces.py` - Core data structures and base classes
- `pyproject.toml:53` - CLI script configuration (`mac-doctor` command)

## Testing Strategy

- **Unit tests**: Individual component testing in `tests/unit/`
- **E2E tests**: Full CLI workflow testing in `tests/e2e/`
- **Coverage**: Configured for comprehensive coverage reporting with HTML output
- **Markers**: Tests categorized by type (unit, integration, system, e2e)

## Development Notes

- All MCPs implement error handling and system availability checks
- LLM providers support fallback mechanisms for reliability
- Agent execution includes timeout protection and iteration limits
- Privacy mode restricts to local-only processing
- Rich console output with progress indicators and structured tables