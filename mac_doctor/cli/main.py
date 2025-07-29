"""
Main CLI interface for Mac Doctor.

This module provides the command-line interface using Typer for user interaction
with the Mac Doctor diagnostic system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ..agent.diagnostic_agent import DiagnosticAgent
from ..core.action_engine import ActionEngine
from ..core.report_generator import ReportGenerator
from ..interfaces import CLIConfig
from ..llm.factory import LLMFactory
from ..config import ConfigManager, run_setup_wizard
from ..system_validator import SystemValidator
from ..tool_registry import ToolRegistry
from ..error_handling import (
    ErrorHandler, LoggingManager, SystemCompatibilityError,
    ConfigurationError, safe_execute
)
from ..logging_config import setup_logging as setup_comprehensive_logging, get_logger, debug_context
from .setup import setup_tools_and_agent

# Create the main Typer app
app = typer.Typer(
    name="mac-doctor",
    help="Mac Doctor - AI-powered macOS system diagnostics",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global console for rich output
console = Console()


def setup_logging(debug: bool = False) -> None:
    """Setup comprehensive logging configuration."""
    # Use the new comprehensive logging system
    config = {
        "debug_mode": debug,
        "console_output": True,
        "log_level": logging.DEBUG if debug else logging.INFO,
        "log_directory": str(Path.home() / ".mac_doctor" / "logs"),
        "max_log_entries": 10000,
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "max_files": 10
    }
    
    setup_comprehensive_logging(config)


@app.command()
def diagnose(
    output_format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format (markdown, json)"
    ),
    export_path: Optional[str] = typer.Option(
        None,
        "--export", "-e",
        help="Export report to file"
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider (ollama, gemini)"
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use (gemini-2.5-flash, gemini-2.5-pro, llama3.2)"
    ),
    privacy_mode: bool = typer.Option(
        False,
        "--privacy",
        help="Enable privacy mode (local processing only)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output"
    )
) -> None:
    """Run a comprehensive system diagnostic analysis."""
    setup_logging(debug)
    
    config = CLIConfig(
        mode="diagnose",
        output_format=output_format,
        export_path=export_path,
        llm_provider=llm_provider or "gemini",
        llm_model=llm_model,
        privacy_mode=privacy_mode,
        debug=debug
    )
    
    # Initialize error handler
    error_handler = ErrorHandler(console=console)
    
    def _diagnose_logic():
        # Setup components
        agent, report_generator = setup_tools_and_agent(config)
        
        console.print("[bold blue]🔍 Starting comprehensive system diagnostic...[/bold blue]")
        
        # Run diagnostic analysis
        query = "Perform a comprehensive diagnostic analysis of this Mac system"
        result = agent.analyze(query)
        
        # Generate report
        if config.output_format == "json":
            report_content = report_generator.generate_json(result)
        else:
            report_content = report_generator.generate_markdown(result)
        
        # Output report
        if config.export_path:
            report_generator.export_to_file(report_content, config.export_path)
            console.print(f"[green]✅ Report exported to {config.export_path}[/green]")
        else:
            console.print(report_content)
        
        # Show summary
        _show_diagnostic_summary(result)
        return True
    
    # Use safe execution with comprehensive error handling
    success = safe_execute(
        _diagnose_logic,
        error_handler=error_handler,
        context={"operation": "diagnose", "config": config.__dict__},
        fallback_result=False,
        show_errors=True
    )
    
    if not success:
        console.print("[red]❌ Diagnostic failed. Check error messages above for details.[/red]")
        if debug:
            console.print("\n[dim]Error history:[/dim]")
            for error in error_handler.get_error_history():
                console.print(f"[dim]- {error.category.value}: {error.message}[/dim]")
        sys.exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about your system"),
    output_format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format (markdown, json)"
    ),
    export_path: Optional[str] = typer.Option(
        None,
        "--export", "-e",
        help="Export report to file"
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider (ollama, gemini)"
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use (gemini-2.5-flash, gemini-2.5-pro, llama3.2)"
    ),
    privacy_mode: bool = typer.Option(
        False,
        "--privacy",
        help="Enable privacy mode (local processing only)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output"
    )
) -> None:
    """Ask a specific question about your system."""
    setup_logging(debug)
    
    config = CLIConfig(
        mode="ask",
        query=question,
        output_format=output_format,
        export_path=export_path,
        llm_provider=llm_provider or "gemini",
        llm_model=llm_model,
        privacy_mode=privacy_mode,
        debug=debug
    )
    
    try:
        # Setup components
        agent, report_generator = setup_tools_and_agent(config)
        
        console.print(f"[bold blue]🤔 Analyzing: {question}[/bold blue]")
        
        # Run targeted analysis
        result = agent.analyze(question)
        
        # Generate report
        if config.output_format == "json":
            report_content = report_generator.generate_json(result)
        else:
            report_content = report_generator.generate_markdown(result)
        
        # Output report
        if config.export_path:
            report_generator.export_to_file(report_content, config.export_path)
            console.print(f"[green]✅ Report exported to {config.export_path}[/green]")
        else:
            console.print(report_content)
        
        # Show summary
        _show_diagnostic_summary(result)
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command("list-tools")
def list_tools(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output"
    )
) -> None:
    """List available diagnostic tools."""
    setup_logging(debug)
    
    config = CLIConfig(mode="list-tools", debug=debug)
    
    try:
        # Setup components
        agent, _ = setup_tools_and_agent(config)
        
        console.print("[bold blue]🔧 Available Diagnostic Tools[/bold blue]\n")
        
        # Create table for tools
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        
        # Get tool information
        tool_registry = agent.tool_registry
        available_tools = agent.get_available_tools()
        
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if tool:
                status = "✅ Available" if tool_name in available_tools else "❌ Unavailable"
                table.add_row(tool.name, tool.description, status)
        
        console.print(table)
        console.print(f"\n[dim]Total tools: {len(tool_registry.list_tools())}[/dim]")
        console.print(f"[dim]Available tools: {len(available_tools)}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list tools: {str(e)}[/red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def trace(
    query: Optional[str] = typer.Argument(None, help="Optional query to trace"),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider (ollama, gemini)"
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use (gemini-2.5-flash, gemini-2.5-pro, llama3.2)"
    ),
    privacy_mode: bool = typer.Option(
        False,
        "--privacy",
        help="Enable privacy mode (local processing only)"
    ),
    debug: bool = typer.Option(
        True,
        "--debug/--no-debug",
        help="Enable debug output (default: enabled for trace mode)"
    )
) -> None:
    """Run diagnostic analysis with detailed execution traces."""
    setup_logging(True)  # Always enable debug logging for trace mode
    
    # Use default query if none provided
    if not query:
        query = "Perform a basic system diagnostic with detailed tracing"
    
    config = CLIConfig(
        mode="trace",
        query=query,
        output_format="markdown",
        llm_provider=llm_provider or "gemini",
        llm_model=llm_model,
        privacy_mode=privacy_mode,
        debug=True
    )
    
    try:
        # Setup components
        agent, report_generator = setup_tools_and_agent(config)
        
        console.print("[bold blue]🔍 Starting diagnostic analysis with detailed tracing...[/bold blue]")
        console.print(f"[dim]Query: {query}[/dim]\n")
        
        # Run analysis with tracing
        result = agent.analyze(query)
        
        # Generate and display report
        report_content = report_generator.generate_markdown(result)
        console.print(report_content)
        
        # Show detailed execution information
        console.print("\n[bold blue]📊 Execution Details[/bold blue]")
        console.print(f"Total execution time: {result.execution_time:.2f} seconds")
        console.print(f"Tools executed: {len(result.tool_results)}")
        console.print(f"Issues detected: {len(result.issues_detected)}")
        console.print(f"Recommendations generated: {len(result.recommendations)}")
        
    except Exception as e:
        console.print(f"[red]❌ Trace analysis failed: {str(e)}[/red]")
        console.print_exception()
        sys.exit(1)


@app.command("system-check")
def system_check(
    show_details: bool = typer.Option(
        False,
        "--details",
        help="Show detailed system tool availability"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output"
    )
) -> None:
    """Check system compatibility and requirements."""
    setup_logging(debug)
    
    try:
        console.print("[bold blue]🔍 Checking system compatibility...[/bold blue]\n")
        
        # Run system validation
        validator = SystemValidator()
        system_info = validator.validate_system()
        
        # Display results
        validator.display_system_status(show_details=show_details)
        
        # Show installation guidance if needed
        if not system_info.is_compatible:
            guidance = validator.get_installation_guidance()
            if guidance:
                console.print("\n[yellow]📋 Installation Guidance:[/yellow]")
                for line in guidance:
                    console.print(line)
            
            sys.exit(1)
        else:
            console.print("[green]🎉 System is ready for Mac Doctor![/green]")
    
    except Exception as e:
        console.print(f"[red]❌ System check failed: {str(e)}[/red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def setup(
    force: bool = typer.Option(
        False,
        "--force",
        help="Force setup even if configuration exists"
    )
) -> None:
    """Run the interactive setup wizard."""
    try:
        console.print("[bold blue]🔧 Mac Doctor Setup[/bold blue]\n")
        
        # Run setup wizard
        config = run_setup_wizard(force=force)
        
        console.print("\n[green]🎉 Setup completed successfully![/green]")
        console.print("You can now use Mac Doctor with your configured settings.")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Setup failed: {str(e)}[/red]")
        sys.exit(1)


@app.command()
def debug(
    show_logs: bool = typer.Option(
        False,
        "--logs",
        help="Show recent log entries"
    ),
    show_traces: bool = typer.Option(
        False,
        "--traces",
        help="Show execution traces"
    ),
    export_debug: Optional[str] = typer.Option(
        None,
        "--export",
        help="Export debug data to directory"
    ),
    clear_logs: bool = typer.Option(
        False,
        "--clear-logs",
        help="Clear log entries"
    ),
    clear_traces: bool = typer.Option(
        False,
        "--clear-traces",
        help="Clear execution traces"
    ),
    cleanup_old: bool = typer.Option(
        False,
        "--cleanup",
        help="Clean up old log files"
    )
) -> None:
    """Show debug information and manage logging."""
    setup_logging(True)  # Always enable debug mode for debug command
    
    try:
        logger = get_logger()
        
        if clear_logs:
            logger.structured_handler.clear_entries()
            console.print("[green]✅ Log entries cleared[/green]")
            return
        
        if clear_traces:
            logger.execution_tracer.clear_traces()
            console.print("[green]✅ Execution traces cleared[/green]")
            return
        
        if cleanup_old:
            logger.cleanup_logs(max_age_days=30)
            console.print("[green]✅ Old log files cleaned up[/green]")
            return
        
        if export_debug:
            export_path = Path(export_debug)
            logger.export_debug_data(export_path)
            console.print(f"[green]✅ Debug data exported to {export_path}[/green]")
            return
        
        # Show debug panel by default
        logger.show_debug_panel()
        
        if show_logs:
            console.print("\n[bold blue]📝 Recent Log Entries[/bold blue]")
            recent_logs = logger.structured_handler.get_entries(limit=20)
            
            if recent_logs:
                log_table = Table(show_header=True, header_style="bold magenta")
                log_table.add_column("Time", style="dim")
                log_table.add_column("Level", style="cyan")
                log_table.add_column("Logger", style="yellow")
                log_table.add_column("Message", style="white")
                
                for entry in recent_logs:
                    time_str = entry.timestamp.strftime("%H:%M:%S")
                    log_table.add_row(
                        time_str,
                        entry.level,
                        entry.logger_name.split('.')[-1],  # Show only last part
                        entry.message[:80] + "..." if len(entry.message) > 80 else entry.message
                    )
                
                console.print(log_table)
            else:
                console.print("[dim]No recent log entries found[/dim]")
        
        if show_traces:
            console.print("\n[bold blue]🔬 Recent Execution Traces[/bold blue]")
            recent_traces = logger.execution_tracer.get_recent_traces(limit=10)
            
            if recent_traces:
                trace_table = Table(show_header=True, header_style="bold magenta")
                trace_table.add_column("Function", style="cyan")
                trace_table.add_column("Module", style="yellow")
                trace_table.add_column("Duration", style="green")
                trace_table.add_column("Status", style="white")
                
                for trace in recent_traces:
                    duration = f"{trace.duration_ms:.1f}ms" if trace.duration_ms else "Running"
                    status = "Error" if trace.exception else "Success"
                    status_style = "red" if trace.exception else "green"
                    
                    trace_table.add_row(
                        trace.function_name,
                        trace.module_name.split('.')[-1] if trace.module_name else "",
                        duration,
                        f"[{status_style}]{status}[/{status_style}]"
                    )
                
                console.print(trace_table)
            else:
                console.print("[dim]No recent execution traces found[/dim]")
    
    except Exception as e:
        console.print(f"[red]❌ Debug command failed: {str(e)}[/red]")
        console.print_exception()
        sys.exit(1)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Set default LLM provider"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Set default LLM model"
    ),
    privacy_mode: Optional[bool] = typer.Option(
        None,
        "--privacy/--no-privacy",
        help="Enable/disable privacy mode"
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Reset configuration to defaults"
    )
) -> None:
    """Manage Mac Doctor configuration."""
    try:
        config_manager = ConfigManager()
        
        if reset:
            # Reset to default configuration
            config_manager.reset_config()
            console.print("[green]✅ Configuration reset to defaults[/green]")
            return
        
        # Load current configuration
        current_config = config_manager.load_config()
        
        if show:
            # Show current configuration
            console.print("[bold blue]📋 Current Configuration[/bold blue]\n")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("LLM Provider", current_config.default_llm_provider)
            
            provider_config = current_config.get_provider_config(current_config.default_llm_provider)
            model_name = provider_config.model if provider_config else "Default"
            table.add_row("Model", model_name)
            
            table.add_row("Privacy Mode", "Enabled" if current_config.privacy_mode else "Disabled")
            table.add_row("Fallback Enabled", "Yes" if current_config.fallback_enabled else "No")
            table.add_row("Fallback Providers", ", ".join(current_config.fallback_providers))
            table.add_row("Output Format", current_config.default_output_format)
            table.add_row("Debug Mode", "Enabled" if current_config.debug_mode else "Disabled")
            
            console.print(table)
            
            # Show provider availability
            console.print("\n[bold blue]🔌 Provider Availability[/bold blue]\n")
            factory = LLMFactory(current_config)
            availability = factory.get_available_providers()
            
            provider_table = Table(show_header=True, header_style="bold magenta")
            provider_table.add_column("Provider", style="cyan")
            provider_table.add_column("Status", style="white")
            
            for provider_name, is_available in availability.items():
                status = "✅ Available" if is_available else "❌ Unavailable"
                provider_table.add_row(provider_name, status)
            
            console.print(provider_table)
            
            # Show configuration info
            config_info = config_manager.get_config_info()
            console.print(f"\n[dim]Configuration file: {config_info['config_path']}[/dim]")
            
            if config_info['validation_issues']:
                console.print("\n[yellow]⚠️  Configuration Issues:[/yellow]")
                for issue in config_info['validation_issues']:
                    console.print(f"  • {issue}")
            
            return
        
        # Update configuration
        updated = False
        
        if provider:
            current_config.default_llm_provider = provider
            updated = True
            console.print(f"[green]✅ Set provider to: {provider}[/green]")
        
        if model:
            provider_config = current_config.get_provider_config(current_config.default_llm_provider)
            if provider_config:
                provider_config.model = model
                updated = True
                console.print(f"[green]✅ Set model to: {model}[/green]")
            else:
                console.print(f"[red]❌ Provider '{current_config.default_llm_provider}' not configured[/red]")
        
        if privacy_mode is not None:
            current_config.privacy_mode = privacy_mode
            updated = True
            status = "enabled" if privacy_mode else "disabled"
            console.print(f"[green]✅ Privacy mode {status}[/green]")
        
        if updated:
            config_manager.save_config(current_config)
            console.print("[green]✅ Configuration saved[/green]")
        else:
            console.print("[yellow]No configuration changes specified. Use --show to view current config.[/yellow]")
    
    except Exception as e:
        console.print(f"[red]❌ Configuration failed: {str(e)}[/red]")
        sys.exit(1)


def _show_diagnostic_summary(result) -> None:
    """Show a summary of diagnostic results."""
    console.print("\n[bold blue]📊 Diagnostic Summary[/bold blue]")
    
    # Issues summary
    if result.issues_detected:
        critical = sum(1 for i in result.issues_detected if i.severity == 'critical')
        high = sum(1 for i in result.issues_detected if i.severity == 'high')
        medium = sum(1 for i in result.issues_detected if i.severity == 'medium')
        low = sum(1 for i in result.issues_detected if i.severity == 'low')
        
        console.print(f"Issues found: {len(result.issues_detected)}")
        if critical > 0:
            console.print(f"  🔴 Critical: {critical}")
        if high > 0:
            console.print(f"  🟠 High: {high}")
        if medium > 0:
            console.print(f"  🟡 Medium: {medium}")
        if low > 0:
            console.print(f"  🟢 Low: {low}")
    else:
        console.print("✅ No issues detected")
    
    # Recommendations summary
    if result.recommendations:
        console.print(f"Recommendations: {len(result.recommendations)}")
        
        # Ask if user wants to execute recommendations
        if result.recommendations:
            console.print("\n[bold yellow]💡 Would you like to execute any recommendations?[/bold yellow]")
            console.print("Use the action engine to safely execute suggested fixes.")
    
    console.print(f"Analysis completed in {result.execution_time:.2f} seconds")


if __name__ == "__main__":
    app()