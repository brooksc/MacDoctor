Mac Doctor: Requirements Specification

Overview

“Mac Doctor” is a Python-based agentic AI assistant designed to analyze the current performance and health of a macOS system, diagnose issues, and suggest actions. It leverages local system diagnostic tools (e.g., ps, top, dtrace, vm_stat, iostat, and others) and feeds collected metrics into a locally running Gemini-compatible LLM (Flash 2.5 model) to provide user-friendly recommendations. The agent dynamically selects which tools to invoke based on the problem it’s asked to solve.

⸻

Goals
	1.	Consumer-/Hobbyist-Friendly tool.
	2.	Provide an actionable diagnostic report about Mac performance issues.
	3.	Offer interactive recommendations with confirmations for suggested actions (e.g., kill a runaway process).
	4.	Be agent-driven: capable of answering arbitrary user queries by calling a set of modular toolchains.
	5.	Maintain privacy (all data and model runs locally).

⸻

System Requirements
	•	macOS 12 or later (Apple Silicon M1/M2/M3)
	•	Python 3.10+
	•	Gemini Flash 2.5-compatible inference runtime (e.g., gemini-python + local model)
	•	Dependencies:
	•	langchain or langgraph (for agent orchestration)
	•	psutil
	•	subprocess
	•	rich (for output formatting)
	•	typer or argparse (for CLI)
	•	gemini-python SDK (local model)
	•	(optional) matplotlib or plotext for graphs

⸻

Architecture

Agent-Centric Workflow

The agent should be implemented using LangChain or another agentic framework that enables:
	•	Tool-based planning: given a question or directive (e.g., “Why is my Mac slow?”), the agent plans and executes a series of steps using tools.
	•	Pluggable tools (MCPs): Each diagnostic component (e.g., ps, dtrace, vm_stat) is exposed as a modular callable tool.
	•	Memory (optional): cache recent diagnostics to avoid repeating steps.
	•	Autonomous or conversational mode.

1. Tools (MCPs)

Each diagnostic interface is implemented as a modular Mac Collector Plugin (MCP):
	•	ps_mcp: Lists top CPU/memory processes using psutil and ps
	•	dtrace_mcp: Runs and summarizes a specific DTrace script (e.g., syscall count, file access, latency)
	•	xrg_mcp: Integrates with or mimics Stats/XRG output via system commands
	•	vmstat_mcp: Reports memory pressure and paging behavior
	•	disk_mcp: Uses iostat, df, and du to analyze disk I/O and free space
	•	logs_mcp: Reads and summarizes recent system logs
	•	net_mcp: Summarizes network activity (nettop, netstat)
	•	Each MCP is self-contained, outputs structured JSON, and can be registered/unregistered

2. Diagnostic Agent (LLM Integration)
	•	Implemented with LangChain’s agent interface
	•	Input: user question or default prompt (e.g., “Diagnose Mac slowness”)
	•	Agent dynamically calls MCPs and aggregates results
	•	Output from MCPs is embedded into prompts to Gemini Flash 2.5 model via gemini-python
	•	LLM responds with:
	•	Issue analysis
	•	Root cause(s)
	•	Actionable fixes

3. Recommendation Interpreter
	•	Parses LLM output
	•	Proposes safe actions (e.g., kill PID, sudo purge, disable Spotlight)
	•	Uses confirmation prompt before execution
	•	Logs actions taken

4. CLI Interface
	•	Entry point for invoking the agent

$ python mac_doctor.py diagnose

	•	Modes:
	•	diagnose: general analysis
	•	ask "<question>": arbitrary user query (e.g., “Why is fan loud?”)
	•	list-tools: show registered MCPs
	•	trace: run a trace-only debug

⸻

Output Format
	•	Markdown-formatted report (print to terminal and optionally export)
	•	JSON output available with --json
	•	Includes:
	•	Detected Issues
	•	Tool results
	•	Suggested Fixes (with command options)

⸻

Future Enhancements (Post-MVP)
	•	GUI menu-bar app
	•	Natural language remediation commands
	•	GPT-4o or Claude 3 as backends
	•	Integration with system_profiler
	•	Autonomous monitoring agent

⸻

Privacy & Security Considerations
	•	No remote data transfer
	•	Local model use only
	•	Clear indication before any sudo command or destructive action

⸻

Example Prompt and Flow
	1.	User: Why is my Mac slow today?
	2.	Agent loads plan:
	•	Run ps_mcp, vmstat_mcp, disk_mcp, dtrace_mcp
	3.	Data fed into Gemini Flash 2.5
	4.	LLM output:

🛠 Problem: Kernel_task using excessive CPU (likely thermal management)
💡 Recommendation: Check ambient temperature, disconnect extra monitors, clean vents
❓ Suggestion: Kill runaway helper process (PID 2314)? [Y/n]

Would you like a scaffolded implementation of the LangChain agent and MCP structure next?