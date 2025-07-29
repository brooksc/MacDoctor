"""
Action engine for safe execution of system commands.

This module provides the ActionEngine class that safely executes system commands
with user confirmation, validation, and comprehensive logging.
"""

import logging
import subprocess
import shlex
from datetime import datetime
from typing import List, Set, Optional
from rich.console import Console
from rich.prompt import Confirm

from ..interfaces import Recommendation, ActionResult
from ..error_handling import (
    ErrorHandler, MacDoctorPermissionError, TimeoutError, ConfigurationError,
    safe_execute
)


class ActionEngine:
    """Engine for safe execution of system commands with validation and logging."""
    
    # Whitelist of safe commands that can be executed
    SAFE_COMMANDS = {
        # System information commands (read-only)
        'ps', 'top', 'htop', 'df', 'du', 'free', 'uptime', 'whoami', 'id',
        'uname', 'sw_vers', 'system_profiler', 'sysctl', 'vm_stat', 'iostat',
        'netstat', 'lsof', 'dscl', 'pmset', 'log',
        
        # Safe system maintenance commands
        'killall', 'kill', 'launchctl', 'sudo',
        
        # File operations (with restrictions)
        'ls', 'cat', 'head', 'tail', 'find', 'grep', 'wc', 'sort', 'uniq',
        'rm', 'mv', 'cp',  # Added for controlled file operations
        
        # Package management (read-only operations)
        'brew', 'pip', 'npm', 'yarn',
        
        # Network diagnostics
        'ping', 'traceroute', 'nslookup', 'dig', 'curl', 'wget',
        
        # Shell utilities
        'echo', 'printf',
    }
    
    # Commands that require elevated privileges
    SUDO_COMMANDS = {
        'launchctl', 'pmset', 'dscl', 'systemsetup', 'nvram', 'diskutil',
        'fsck', 'mount', 'umount', 'chown', 'chmod', 'rm', 'mv', 'cp'
    }
    
    # High-risk commands that require extra confirmation
    HIGH_RISK_COMMANDS = {
        'rm', 'mv', 'dd', 'diskutil', 'fsck', 'format', 'erase', 'delete',
        'shutdown', 'reboot', 'halt', 'init', 'systemctl', 'service'
    }
    
    def __init__(self, console: Optional[Console] = None, logger: Optional[logging.Logger] = None, error_handler: Optional[ErrorHandler] = None):
        """Initialize the ActionEngine.
        
        Args:
            console: Rich console for user interaction (optional)
            logger: Logger for action logging (optional)
            error_handler: Error handler for comprehensive error management (optional)
        """
        self.console = console or Console()
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = error_handler or ErrorHandler(console=self.console, logger=self.logger)
        self._action_history: List[dict] = []
    
    def execute_recommendation(self, recommendation: Recommendation, auto_confirm: bool = False) -> ActionResult:
        """Execute a recommendation with appropriate safety checks.
        
        Args:
            recommendation: The recommendation to execute
            auto_confirm: If True, skip user confirmation (for testing)
            
        Returns:
            ActionResult with execution details
        """
        if recommendation.action_type == 'info':
            # Info recommendations don't require execution
            return ActionResult(
                success=True,
                output=f"Info: {recommendation.description}",
                timestamp=datetime.now()
            )
        
        if not recommendation.command:
            return ActionResult(
                success=False,
                output="",
                error="No command specified in recommendation",
                timestamp=datetime.now()
            )
        
        # Validate command safety
        validation_result = self.validate_command_safety(recommendation.command)
        if not validation_result['is_safe']:
            return ActionResult(
                success=False,
                output="",
                error=f"Command validation failed: {validation_result['reason']}",
                timestamp=datetime.now()
            )
        
        # Get user confirmation if required
        if recommendation.confirmation_required and not auto_confirm:
            if not self.confirm_action(recommendation):
                return ActionResult(
                    success=False,
                    output="",
                    error="Action cancelled by user",
                    timestamp=datetime.now()
                )
        
        # Execute the command with comprehensive error handling
        try:
            result = self._execute_command(recommendation.command, recommendation.action_type == 'sudo_command')
            self.log_action(recommendation, result)
            return result
        except Exception as e:
            # Handle errors with the error handler
            self.error_handler.handle_error(
                e,
                context={
                    "operation": "command_execution",
                    "command": recommendation.command,
                    "action_type": recommendation.action_type
                },
                show_user_message=True
            )
            
            error_result = ActionResult(
                success=False,
                output="",
                error=f"Command execution failed: {str(e)}",
                timestamp=datetime.now()
            )
            self.log_action(recommendation, error_result)
            return error_result
    
    def confirm_action(self, recommendation: Recommendation) -> bool:
        """Prompt user for confirmation before executing an action.
        
        Args:
            recommendation: The recommendation to confirm
            
        Returns:
            True if user confirms, False otherwise
        """
        self.console.print(f"\n[bold]Recommendation:[/bold] {recommendation.title}")
        self.console.print(f"[dim]{recommendation.description}[/dim]")
        
        if recommendation.command:
            self.console.print(f"[bold]Command:[/bold] {recommendation.command}")
        
        # Show risk level
        risk_color = {
            'low': 'green',
            'medium': 'yellow',
            'high': 'red'
        }.get(recommendation.risk_level, 'yellow')
        
        self.console.print(f"[bold]Risk Level:[/bold] [{risk_color}]{recommendation.risk_level.upper()}[/{risk_color}]")
        
        if recommendation.action_type == 'sudo_command':
            self.console.print("[bold red]⚠️  This command requires elevated privileges (sudo)[/bold red]")
        
        if recommendation.risk_level == 'high':
            self.console.print("[bold red]⚠️  HIGH RISK: This action could affect system stability[/bold red]")
            return Confirm.ask("Are you absolutely sure you want to proceed?", default=False)
        
        return Confirm.ask("Do you want to execute this action?", default=True)
    
    def validate_command_safety(self, command: str) -> dict:
        """Validate if a command is safe to execute.
        
        Args:
            command: The command to validate
            
        Returns:
            Dict with 'is_safe' boolean and 'reason' string
        """
        if not command or not command.strip():
            return {'is_safe': False, 'reason': 'Empty command'}
        
        try:
            # Check for dangerous patterns first
            dangerous_patterns = [
                '&&', '||', ';', '|', '>', '>>', '<', '`', '$(',
                'rm -rf /', 'rm -rf *', 'dd if=', 'format', 'mkfs',
                'fdisk', 'parted', 'gpt', 'diskutil erase'
            ]
            
            command_lower = command.lower()
            for pattern in dangerous_patterns:
                if pattern in command_lower:
                    return {'is_safe': False, 'reason': f'Dangerous pattern detected: {pattern}'}
            
            # Parse the command to get the base command
            parts = shlex.split(command.strip())
            if not parts:
                return {'is_safe': False, 'reason': 'Invalid command format'}
            
            base_command = parts[0]
            
            # Handle sudo commands
            if base_command == 'sudo' and len(parts) > 1:
                base_command = parts[1]
            
            # Check if command is in whitelist
            if base_command not in self.SAFE_COMMANDS:
                return {'is_safe': False, 'reason': f'Command "{base_command}" not in whitelist'}
            
            # Additional checks for high-risk commands
            if base_command in self.HIGH_RISK_COMMANDS:
                # Extra validation for rm commands
                if base_command == 'rm':
                    if '-rf' in command or '-fr' in command:
                        if '/' in command or '*' in command:
                            return {'is_safe': False, 'reason': 'Dangerous rm command with wildcards or root paths'}
            
            return {'is_safe': True, 'reason': 'Command passed validation'}
            
        except Exception as e:
            return {'is_safe': False, 'reason': f'Command parsing failed: {str(e)}'}
    
    def _execute_command(self, command: str, use_sudo: bool = False) -> ActionResult:
        """Execute a system command safely.
        
        Args:
            command: The command to execute
            use_sudo: Whether to use sudo for execution
            
        Returns:
            ActionResult with execution details
        """
        start_time = datetime.now()
        
        try:
            # Prepare the command
            if use_sudo and not command.startswith('sudo'):
                command = f"sudo {command}"
            
            # Execute with timeout
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout
                    check=False
                )
                
                if result.returncode == 0:
                    return ActionResult(
                        success=True,
                        output=result.stdout,
                        timestamp=start_time
                    )
                else:
                    # Handle specific error conditions
                    error_msg = result.stderr or f"Command failed with exit code {result.returncode}"
                    
                    if result.returncode == 126:
                        raise MacDoctorPermissionError(
                            operation=command,
                            suggestions=[
                                "Check if the command has execute permissions",
                                "Verify the command path is correct",
                                "Try running with sudo if appropriate"
                            ]
                        )
                    elif result.returncode == 127:
                        raise ConfigurationError(
                            config_item="command_path",
                            message=f"Command not found: {command}",
                            suggestions=[
                                "Check if the command is installed",
                                "Verify the command is in PATH",
                                "Install missing dependencies"
                            ]
                        )
                    else:
                        return ActionResult(
                            success=False,
                            output=result.stdout,
                            error=error_msg,
                            timestamp=start_time
                        )
                        
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    operation=f"command execution: {command}",
                    timeout_seconds=30,
                    suggestions=[
                        "Try a simpler command",
                        "Check system performance",
                        "Increase command timeout"
                    ]
                )
        
        except Exception as e:
            return ActionResult(
                success=False,
                output="",
                error=f"Command execution failed: {str(e)}",
                timestamp=start_time
            )
    
    def log_action(self, recommendation: Recommendation, result: ActionResult) -> None:
        """Log an executed action for audit purposes.
        
        Args:
            recommendation: The recommendation that was executed
            result: The result of the execution
        """
        log_entry = {
            'timestamp': result.timestamp.isoformat(),
            'recommendation_title': recommendation.title,
            'command': recommendation.command,
            'action_type': recommendation.action_type,
            'risk_level': recommendation.risk_level,
            'success': result.success,
            'output': result.output[:500] if result.output else '',  # Truncate long output
            'error': result.error
        }
        
        self._action_history.append(log_entry)
        
        # Log to configured logger
        if result.success:
            self.logger.info(f"Action executed successfully: {recommendation.title}")
        else:
            self.logger.error(f"Action failed: {recommendation.title} - {result.error}")
        
        # Detailed debug logging
        self.logger.debug(f"Action details: {log_entry}")
    
    def get_action_history(self) -> List[dict]:
        """Get the history of executed actions.
        
        Returns:
            List of action log entries
        """
        return self._action_history.copy()
    
    def clear_action_history(self) -> None:
        """Clear the action history."""
        self._action_history.clear()
        self.logger.info("Action history cleared")