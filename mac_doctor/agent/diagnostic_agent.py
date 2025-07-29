"""
LangChain Diagnostic Agent - Orchestrates Mac diagnostic analysis.

This module provides a LangChain-based agent that dynamically selects and executes
diagnostic tools based on user queries, analyzes the results using LLM providers,
and generates actionable recommendations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from ..interfaces import BaseLLM, DiagnosticResult, Issue, MCPResult, Recommendation
from ..langchain_tools import MCPToolFactory
from ..tool_registry import ToolRegistry
from ..output_parsing import StructuredOutputParser, DiagnosticAnalysis
from ..error_handling import (
    ErrorHandler, LLMError, ConfigurationError, TimeoutError,
    safe_execute, create_safe_diagnostic_result
)
from ..logging_config import trace_execution, debug_context, log_performance

logger = logging.getLogger(__name__)


class DiagnosticAgent:
    """LangChain-based diagnostic agent for Mac system analysis."""
    
    def __init__(self, llm: BaseLLM, tool_registry: ToolRegistry, error_handler: Optional[ErrorHandler] = None):
        """Initialize the diagnostic agent.
        
        Args:
            llm: LLM provider for analysis and planning
            tool_registry: Registry containing available MCP tools
            error_handler: Error handler for comprehensive error management
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.error_handler = error_handler or ErrorHandler()
        self.langchain_tools = []
        self.agent = None
        self.agent_executor = None
        self.output_parser = StructuredOutputParser()
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent with tools and prompts."""
        def _init_logic():
            # Convert MCP tools to LangChain tools
            self.langchain_tools = MCPToolFactory.create_tools_from_registry(
                self.tool_registry
            )
            
            if not self.langchain_tools:
                raise ConfigurationError(
                    "tool_registry",
                    "No tools available for agent initialization",
                    suggestions=[
                        "Check if MCP tools are properly registered",
                        "Verify tool availability on the system",
                        "Ensure required dependencies are installed"
                    ]
                )
            
            # Create the ReAct agent with custom prompt
            prompt = self._create_agent_prompt()
            
            # Create the underlying LangChain LLM from our BaseLLM
            langchain_llm = self._get_langchain_llm()
            
            # Create the ReAct agent
            self.agent = create_react_agent(
                llm=langchain_llm,
                tools=self.langchain_tools,
                prompt=prompt
            )
            
            # Create the agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.langchain_tools,
                verbose=True,
                max_iterations=6,
                max_execution_time=120,  # 2 minute timeout
                handle_parsing_errors=True
            )
            
            logger.info(f"Initialized diagnostic agent with {len(self.langchain_tools)} tools")
            return True
        
        # Use safe execution with error handling
        result = safe_execute(
            _init_logic,
            error_handler=self.error_handler,
            context={"operation": "agent_initialization"},
            fallback_result=False,
            show_errors=True
        )
        
        if not result:
            self.agent = None
            self.agent_executor = None
    
    def _get_langchain_llm(self):
        """Get the underlying LangChain LLM from our BaseLLM wrapper."""
        # Access the internal LangChain LLM from our providers
        if hasattr(self.llm, '_llm') and self.llm._llm is not None:
            return self.llm._llm
        else:
            raise LLMError(
                provider=self.llm.provider_name,
                message="LLM provider does not have a LangChain LLM instance",
                suggestions=[
                    "Check LLM provider initialization",
                    "Verify provider configuration",
                    "Try reinitializing the LLM provider"
                ]
            )
    
    def _create_agent_prompt(self) -> PromptTemplate:
        """Create the system prompt template for the diagnostic agent."""
        template = """You are a macOS system diagnostic expert agent. Your job is to analyze Mac system performance and health issues by intelligently selecting and using diagnostic tools.

Here are the tools available to you:
{tools}

When a user asks a question or requests a diagnosis:
1. Think about what diagnostic information would be needed to answer their question
2. Select the most relevant tool by its `Tool Name` to gather that information
3. Execute the tool to collect diagnostic data
4. Analyze the results to identify issues and patterns
5. Provide a comprehensive analysis with actionable recommendations

Guidelines:
- Start with broader tools (like process analysis) before diving into specific areas
- If the user asks about a specific issue (CPU, memory, disk, network), focus on relevant tools
- Always explain your reasoning for tool selection
- Provide clear, actionable insights based on the diagnostic data
- If you encounter errors, try alternative approaches or tools

IMPORTANT: When using tools, provide proper input parameters:
- For 'process' tool: Use {{"top_n": 10, "sort_by": "cpu"}}
- For 'disk' tool: Use {{"include_du": true}}
- For 'vmstat' tool: Use {{"count": 5}}
- For 'network' tool: Use {{"duration": 5}}
- For 'logs' tool: Use {{"hours": 1}}
- For 'dtrace' tool: Use {{"script_type": "syscall", "duration": 10}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": self._format_tools_description(),
                "tool_names": ", ".join([f"`{tool.name}`" for tool in self.langchain_tools])
            }
        )
    
    def _format_tools_description(self) -> str:
        """Format tool descriptions for the prompt."""
        if not self.langchain_tools:
            return "No tools available"
        
        descriptions = []
        for tool in self.langchain_tools:
            descriptions.append(f"Tool Name: `{tool.name}`\nDescription: {tool.description}")
        
        return "\n\n".join(descriptions)
    
    @trace_execution(include_args=True, include_return=False)
    def analyze(self, query: str) -> DiagnosticResult:
        """Analyze a user query using the diagnostic agent.
        
        Args:
            query: User's diagnostic query or question
            
        Returns:
            DiagnosticResult with analysis, issues, and recommendations
        """
        if not self.agent_executor:
            error = ConfigurationError(
                "agent_executor",
                "Diagnostic agent is not properly initialized",
                suggestions=[
                    "Check agent initialization logs",
                    "Verify LLM provider availability",
                    "Ensure MCP tools are registered"
                ]
            )
            return create_safe_diagnostic_result(query, error, 0.0, self.error_handler)
        
        start_time = time.time()
        
        def _analyze_logic():
            with debug_context("diagnostic_analysis", {"query": query[:100]}):
                # Run the agent executor with timeout handling
                try:
                    with debug_context("agent_executor_invoke"):
                        result = self.agent_executor.invoke({"input": query})
                except Exception as e:
                    if "timeout" in str(e).lower():
                        raise TimeoutError(
                            operation="agent_analysis",
                            timeout_seconds=120,
                            suggestions=[
                                "Try a simpler query",
                                "Check system performance",
                                "Increase agent timeout in configuration"
                            ]
                        )
                    else:
                        raise LLMError(
                            provider=self.llm.provider_name,
                            message=f"Agent execution failed: {str(e)}",
                            suggestions=[
                                "Check LLM provider connectivity",
                                "Verify agent configuration",
                                "Try with a different LLM provider"
                            ]
                        )
                
                # Extract the agent's analysis from the output
                agent_analysis = result.get("output", "No analysis provided")
                
                # Extract tool results from intermediate steps
                intermediate_steps = result.get("intermediate_steps", [])
                with debug_context("extract_tool_results"):
                    tool_results = self._extract_tool_results(intermediate_steps)
                
                # Convert to MCP results format
                with debug_context("convert_to_mcp_results"):
                    mcp_results = self._convert_to_mcp_results(tool_results)
                
                # Detect issues from analysis and tool results
                with debug_context("detect_issues"):
                    issues = self._detect_issues_from_analysis(agent_analysis, tool_results)
                
                # Generate recommendations based on detected issues
                with debug_context("generate_recommendations"):
                    recommendations = self._generate_recommendations(issues, tool_results)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log performance
                log_performance("diagnostic_analysis", execution_time, {
                    "query_length": len(query),
                    "tools_executed": len(tool_results),
                    "issues_detected": len(issues),
                    "recommendations_generated": len(recommendations)
                })
                
                # Create and return DiagnosticResult
                from datetime import datetime
                return DiagnosticResult(
                    query=query,
                    analysis=agent_analysis,
                    tool_results=mcp_results,
                    issues_detected=issues,
                    recommendations=recommendations,
                    execution_time=execution_time,
                    timestamp=datetime.fromtimestamp(start_time)
                )
        
        # Use safe execution with comprehensive error handling
        result = safe_execute(
            _analyze_logic,
            error_handler=self.error_handler,
            context={"query": query, "operation": "diagnostic_analysis"},
            fallback_result=None,
            show_errors=True
        )
        
        if result is not None:
            return result
        else:
            # Create fallback diagnostic result
            execution_time = time.time() - start_time
            return create_safe_diagnostic_result(
                query, 
                Exception("Analysis failed with unknown error"), 
                execution_time, 
                self.error_handler
            )
    
    def _extract_tool_results(self, intermediate_steps: List) -> Dict[str, Any]:
        """Extract tool execution results from agent intermediate steps."""
        tool_results = {}
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action, observation = step[0], step[1]
                tool_name = getattr(action, 'tool', None)
                
                if tool_name and observation:
                    try:
                        # Try to parse JSON observation
                        import json
                        if isinstance(observation, str):
                            parsed_result = json.loads(observation)
                            tool_results[tool_name] = parsed_result
                        else:
                            tool_results[tool_name] = {"raw_output": str(observation)}
                    except (json.JSONDecodeError, TypeError):
                        tool_results[tool_name] = {"raw_output": str(observation)}
        
        return tool_results
    
    def _convert_to_mcp_results(self, tool_results: Dict[str, Any]) -> Dict[str, MCPResult]:
        """Convert tool results to MCPResult format."""
        mcp_results = {}
        
        for tool_name, result_data in tool_results.items():
            if isinstance(result_data, dict) and "success" in result_data:
                mcp_results[tool_name] = MCPResult(
                    tool_name=tool_name,
                    success=result_data.get("success", True),
                    data=result_data.get("data", result_data),
                    error=result_data.get("error"),
                    execution_time=result_data.get("execution_time", 0.0),
                    metadata=result_data.get("metadata", {})
                )
            else:
                mcp_results[tool_name] = MCPResult(
                    tool_name=tool_name,
                    success=True,
                    data=result_data,
                    execution_time=0.0
                )
        
        return mcp_results
    
    def _detect_issues_from_analysis(
        self, 
        analysis: str, 
        tool_results: Dict[str, Any]
    ) -> List[Issue]:
        """Detect system issues from analysis and tool results."""
        issues = []
        
        # Analyze process data for high resource usage
        if "process" in tool_results:
            process_data = tool_results["process"]
            if isinstance(process_data, dict) and "data" in process_data:
                data = process_data["data"]
                
                # Check for high CPU usage
                if "system_overview" in data:
                    cpu_usage = data["system_overview"].get("cpu_usage_percent", 0)
                    if cpu_usage > 80:
                        issues.append(Issue(
                            severity="high",
                            category="cpu",
                            title="High CPU Usage",
                            description=f"System CPU usage is at {cpu_usage:.1f}%, which may cause performance issues.",
                            metrics={"cpu_usage_percent": cpu_usage}
                        ))
                    
                    memory_usage = data["system_overview"].get("memory_usage_percent", 0)
                    if memory_usage > 85:
                        issues.append(Issue(
                            severity="high",
                            category="memory",
                            title="High Memory Usage",
                            description=f"System memory usage is at {memory_usage:.1f}%, which may cause swapping and performance degradation.",
                            metrics={"memory_usage_percent": memory_usage}
                        ))
        
        # Analyze disk data for space and I/O issues
        if "disk" in tool_results:
            disk_data = tool_results["disk"]
            if isinstance(disk_data, dict) and "data" in disk_data:
                data = disk_data["data"]
                
                if "disk_analysis" in data and "storage" in data["disk_analysis"]:
                    storage = data["disk_analysis"]["storage"]
                    usage_percent = storage.get("usage_percent", 0)
                    
                    if usage_percent > 90:
                        issues.append(Issue(
                            severity="critical",
                            category="disk",
                            title="Critical Disk Space",
                            description=f"Disk usage is at {usage_percent:.1f}%, which may cause system instability.",
                            metrics={"disk_usage_percent": usage_percent}
                        ))
                    elif usage_percent > 80:
                        issues.append(Issue(
                            severity="medium",
                            category="disk",
                            title="High Disk Usage",
                            description=f"Disk usage is at {usage_percent:.1f}%, consider cleaning up files.",
                            metrics={"disk_usage_percent": usage_percent}
                        ))
        
        return issues
    
    def _generate_recommendations(
        self, 
        issues: List[Issue], 
        tool_results: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate actionable recommendations based on detected issues."""
        recommendations = []
        
        for issue in issues:
            if issue.category == "cpu" and issue.severity in ["high", "critical"]:
                recommendations.append(Recommendation(
                    priority="high",
                    category="performance",
                    title="Reduce CPU Usage",
                    description="Consider identifying and stopping high CPU processes",
                    action_type="suggestion",
                    safe_to_execute=False,
                    estimated_impact="May improve system responsiveness"
                ))
            
            elif issue.category == "memory" and issue.severity in ["high", "critical"]:
                recommendations.append(Recommendation(
                    priority="high",
                    category="memory",
                    title="Free Up Memory",
                    description="Consider closing unused applications or restarting the system",
                    action_type="suggestion",
                    safe_to_execute=False,
                    estimated_impact="May prevent swapping and improve performance"
                ))
            
            elif issue.category == "disk" and issue.severity == "critical":
                recommendations.append(Recommendation(
                    priority="critical",
                    category="storage",
                    title="Free Disk Space",
                    description="Urgently clean up disk space to prevent system instability",
                    action_type="suggestion",
                    safe_to_execute=False,
                    estimated_impact="Prevents system crashes and data loss"
                ))
        
        return recommendations
    
    def plan_execution(self, query: str) -> List[str]:
        """Plan which tools should be executed for a given query.
        
        Args:
            query: User's diagnostic query
            
        Returns:
            List of tool names that should be executed
        """
        # Simple heuristic-based planning
        # This could be enhanced with LLM-based planning in the future
        
        query_lower = query.lower()
        planned_tools = []
        
        # Always include process analysis for general diagnostics
        if "process" in [tool.name for tool in self.langchain_tools]:
            planned_tools.append("process")
        
        # Add specific tools based on query content
        if any(keyword in query_lower for keyword in ["cpu", "slow", "performance", "lag"]):
            if "process" not in planned_tools:
                planned_tools.append("process")
        
        if any(keyword in query_lower for keyword in ["memory", "ram", "swap"]):
            planned_tools.extend(["vmstat"])
        
        if any(keyword in query_lower for keyword in ["disk", "storage", "space", "io"]):
            planned_tools.extend(["disk"])
        
        if any(keyword in query_lower for keyword in ["network", "internet", "connection"]):
            planned_tools.extend(["network"])
        
        if any(keyword in query_lower for keyword in ["log", "error", "crash"]):
            planned_tools.extend(["logs"])
        
        if any(keyword in query_lower for keyword in ["trace", "system call", "debug"]):
            planned_tools.extend(["dtrace"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in planned_tools:
            if tool not in seen and tool in [t.name for t in self.langchain_tools]:
                seen.add(tool)
                unique_tools.append(tool)
        
        # If no specific tools were selected, use all available tools
        if not unique_tools:
            unique_tools = [tool.name for tool in self.langchain_tools]
        
        logger.info(f"Planned tools for query '{query}': {unique_tools}")
        return unique_tools
    
    def is_available(self) -> bool:
        """Check if the diagnostic agent is available and ready to use."""
        return (
            self.agent_executor is not None and
            self.llm.is_available() and
            len(self.langchain_tools) > 0
        )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.langchain_tools]