"""
Unit tests for the ActionEngine class.

Tests cover command validation, user confirmation, safe execution,
and comprehensive logging functionality.
"""

import pytest
import subprocess
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from mac_doctor.core.action_engine import ActionEngine
from mac_doctor.interfaces import Recommendation, ActionResult


class TestActionEngine:
    """Test cases for ActionEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_console = Mock()
        self.mock_logger = Mock()
        self.engine = ActionEngine(console=self.mock_console, logger=self.mock_logger)
    
    def test_init_default_console_and_logger(self):
        """Test ActionEngine initialization with default console and logger."""
        engine = ActionEngine()
        assert engine.console is not None
        assert engine.logger is not None
        assert engine._action_history == []
    
    def test_init_with_custom_console_and_logger(self):
        """Test ActionEngine initialization with custom console and logger."""
        assert self.engine.console == self.mock_console
        assert self.engine.logger == self.mock_logger
        assert self.engine._action_history == []
    
    def test_execute_info_recommendation(self):
        """Test executing an info-type recommendation."""
        recommendation = Recommendation(
            title="System Info",
            description="Display system information",
            action_type="info"
        )
        
        result = self.engine.execute_recommendation(recommendation)
        
        assert result.success is True
        assert "Info: Display system information" in result.output
        assert result.error is None
        assert isinstance(result.timestamp, datetime)
    
    def test_execute_recommendation_no_command(self):
        """Test executing a recommendation without a command."""
        recommendation = Recommendation(
            title="Test Action",
            description="Test description",
            action_type="command",
            command=None
        )
        
        result = self.engine.execute_recommendation(recommendation)
        
        assert result.success is False
        assert result.error == "No command specified in recommendation"
    
    def test_execute_recommendation_unsafe_command(self):
        """Test executing a recommendation with an unsafe command."""
        recommendation = Recommendation(
            title="Dangerous Action",
            description="Dangerous command",
            action_type="command",
            command="dangerous_command --delete-everything"
        )
        
        result = self.engine.execute_recommendation(recommendation)
        
        assert result.success is False
        assert "Command validation failed" in result.error
    
    @patch('mac_doctor.core.action_engine.subprocess.run')
    def test_execute_safe_command_success(self, mock_run):
        """Test executing a safe command successfully."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Command output",
            stderr=""
        )
        
        recommendation = Recommendation(
            title="List Processes",
            description="List running processes",
            action_type="command",
            command="ps aux",
            confirmation_required=False
        )
        
        result = self.engine.execute_recommendation(recommendation, auto_confirm=True)
        
        assert result.success is True
        assert result.output == "Command output"
        assert result.error is None
        mock_run.assert_called_once()
    
    @patch('mac_doctor.core.action_engine.subprocess.run')
    def test_execute_command_failure(self, mock_run):
        """Test executing a command that fails."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Command failed"
        )
        
        recommendation = Recommendation(
            title="Failing Command",
            description="A command that fails",
            action_type="command",
            command="ps aux",
            confirmation_required=False
        )
        
        result = self.engine.execute_recommendation(recommendation, auto_confirm=True)
        
        assert result.success is False
        assert result.error == "Command failed"
    
    @patch('mac_doctor.core.action_engine.subprocess.run')
    def test_execute_command_timeout(self, mock_run):
        """Test executing a command that times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("ps aux", 30)
        
        recommendation = Recommendation(
            title="Slow Command",
            description="A command that times out",
            action_type="command",
            command="ps aux",
            confirmation_required=False
        )
        
        result = self.engine.execute_recommendation(recommendation, auto_confirm=True)
        
        assert result.success is False
        assert "timed out after 30 seconds" in result.error
    
    @patch('mac_doctor.core.action_engine.subprocess.run')
    def test_execute_sudo_command(self, mock_run):
        """Test executing a sudo command."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Sudo command output",
            stderr=""
        )
        
        recommendation = Recommendation(
            title="Sudo Command",
            description="A command requiring sudo",
            action_type="sudo_command",
            command="launchctl list",
            confirmation_required=False
        )
        
        result = self.engine.execute_recommendation(recommendation, auto_confirm=True)
        
        assert result.success is True
        assert result.output == "Sudo command output"
        # Verify sudo was prepended
        args, kwargs = mock_run.call_args
        assert "sudo launchctl list" in kwargs.get('args', [''])[0] if 'args' in kwargs else True
    
    def test_confirm_action_low_risk(self):
        """Test user confirmation for low-risk action."""
        recommendation = Recommendation(
            title="Safe Action",
            description="A safe action",
            action_type="command",
            command="ps aux",
            risk_level="low"
        )
        
        with patch('mac_doctor.core.action_engine.Confirm.ask', return_value=True) as mock_confirm:
            result = self.engine.confirm_action(recommendation)
            
            assert result is True
            mock_confirm.assert_called_once_with("Do you want to execute this action?", default=True)
    
    def test_confirm_action_high_risk(self):
        """Test user confirmation for high-risk action."""
        recommendation = Recommendation(
            title="Dangerous Action",
            description="A dangerous action",
            action_type="sudo_command",
            command="rm /tmp/test",
            risk_level="high"
        )
        
        with patch('mac_doctor.core.action_engine.Confirm.ask', return_value=False) as mock_confirm:
            result = self.engine.confirm_action(recommendation)
            
            assert result is False
            mock_confirm.assert_called_once_with("Are you absolutely sure you want to proceed?", default=False)
    
    def test_validate_command_safety_empty_command(self):
        """Test validation of empty command."""
        result = self.engine.validate_command_safety("")
        
        assert result['is_safe'] is False
        assert "Empty command" in result['reason']
    
    def test_validate_command_safety_safe_command(self):
        """Test validation of safe command."""
        result = self.engine.validate_command_safety("ps aux")
        
        assert result['is_safe'] is True
        assert result['reason'] == "Command passed validation"
    
    def test_validate_command_safety_unsafe_command(self):
        """Test validation of unsafe command."""
        result = self.engine.validate_command_safety("dangerous_command")
        
        assert result['is_safe'] is False
        assert "not in whitelist" in result['reason']
    
    def test_validate_command_safety_dangerous_pattern(self):
        """Test validation of command with dangerous patterns."""
        dangerous_commands = [
            "ps aux && rm -rf /",
            "ls | rm -rf *",
            "cat file > /dev/null",
            "echo `rm file`",
            "ps $(rm file)"
        ]
        
        for cmd in dangerous_commands:
            result = self.engine.validate_command_safety(cmd)
            assert result['is_safe'] is False
            assert "Dangerous pattern detected" in result['reason']
    
    def test_validate_command_safety_sudo_command(self):
        """Test validation of sudo command."""
        result = self.engine.validate_command_safety("sudo ps aux")
        
        assert result['is_safe'] is True
        assert result['reason'] == "Command passed validation"
    
    def test_validate_command_safety_dangerous_rm(self):
        """Test validation of dangerous rm commands."""
        dangerous_rm_commands = [
            "rm -rf /",
            "rm -rf *",
            "rm -fr /tmp/*"
        ]
        
        for cmd in dangerous_rm_commands:
            result = self.engine.validate_command_safety(cmd)
            assert result['is_safe'] is False
            # These should be caught by dangerous pattern detection
            assert ("Dangerous pattern detected" in result['reason'] or 
                    "Dangerous rm command" in result['reason'])
    
    def test_validate_command_safety_safe_rm(self):
        """Test validation of safe rm commands."""
        result = self.engine.validate_command_safety("rm /tmp/specific_file.txt")
        
        assert result['is_safe'] is True
        assert result['reason'] == "Command passed validation"
    
    def test_validate_command_safety_parsing_error(self):
        """Test validation with command parsing error."""
        # Create a command that will cause shlex.split to fail
        result = self.engine.validate_command_safety("ps 'unclosed quote")
        
        assert result['is_safe'] is False
        assert "Command parsing failed" in result['reason']
    
    def test_log_action_success(self):
        """Test logging of successful action."""
        recommendation = Recommendation(
            title="Test Action",
            description="Test description",
            action_type="command",
            command="ps aux",
            risk_level="low"
        )
        
        result = ActionResult(
            success=True,
            output="Command output",
            timestamp=datetime.now()
        )
        
        self.engine.log_action(recommendation, result)
        
        # Check action history
        assert len(self.engine._action_history) == 1
        log_entry = self.engine._action_history[0]
        assert log_entry['recommendation_title'] == "Test Action"
        assert log_entry['command'] == "ps aux"
        assert log_entry['success'] is True
        
        # Check logger calls
        self.mock_logger.info.assert_called_once()
        self.mock_logger.debug.assert_called_once()
    
    def test_log_action_failure(self):
        """Test logging of failed action."""
        recommendation = Recommendation(
            title="Failed Action",
            description="Test description",
            action_type="command",
            command="ps aux",
            risk_level="medium"
        )
        
        result = ActionResult(
            success=False,
            output="",
            error="Command failed",
            timestamp=datetime.now()
        )
        
        self.engine.log_action(recommendation, result)
        
        # Check action history
        assert len(self.engine._action_history) == 1
        log_entry = self.engine._action_history[0]
        assert log_entry['success'] is False
        assert log_entry['error'] == "Command failed"
        
        # Check logger calls
        self.mock_logger.error.assert_called_once()
        self.mock_logger.debug.assert_called_once()
    
    def test_log_action_truncates_long_output(self):
        """Test that long output is truncated in logs."""
        recommendation = Recommendation(
            title="Long Output Action",
            description="Test description",
            action_type="command",
            command="ps aux"
        )
        
        long_output = "x" * 1000  # 1000 character output
        result = ActionResult(
            success=True,
            output=long_output,
            timestamp=datetime.now()
        )
        
        self.engine.log_action(recommendation, result)
        
        log_entry = self.engine._action_history[0]
        assert len(log_entry['output']) == 500  # Truncated to 500 chars
    
    def test_get_action_history(self):
        """Test getting action history."""
        # Add some actions to history
        self.engine._action_history = [
            {'action': 'test1'},
            {'action': 'test2'}
        ]
        
        history = self.engine.get_action_history()
        
        assert len(history) == 2
        assert history == [{'action': 'test1'}, {'action': 'test2'}]
        
        # Verify it returns a copy (modifying returned list doesn't affect original)
        history.append({'action': 'test3'})
        assert len(self.engine._action_history) == 2
    
    def test_clear_action_history(self):
        """Test clearing action history."""
        # Add some actions to history
        self.engine._action_history = [
            {'action': 'test1'},
            {'action': 'test2'}
        ]
        
        self.engine.clear_action_history()
        
        assert len(self.engine._action_history) == 0
        self.mock_logger.info.assert_called_with("Action history cleared")
    
    def test_safe_commands_whitelist(self):
        """Test that the safe commands whitelist contains expected commands."""
        expected_commands = {
            'ps', 'top', 'df', 'du', 'uptime', 'whoami', 'uname', 'sw_vers',
            'killall', 'kill', 'launchctl', 'sudo', 'ls', 'cat', 'brew',
            'ping', 'traceroute', 'nslookup'
        }
        
        for cmd in expected_commands:
            assert cmd in ActionEngine.SAFE_COMMANDS
    
    def test_sudo_commands_set(self):
        """Test that sudo commands are properly identified."""
        expected_sudo_commands = {
            'launchctl', 'pmset', 'dscl', 'systemsetup', 'nvram', 'diskutil'
        }
        
        for cmd in expected_sudo_commands:
            assert cmd in ActionEngine.SUDO_COMMANDS
    
    def test_high_risk_commands_set(self):
        """Test that high-risk commands are properly identified."""
        expected_high_risk = {
            'rm', 'mv', 'dd', 'diskutil', 'fsck', 'format', 'erase',
            'shutdown', 'reboot'
        }
        
        for cmd in expected_high_risk:
            assert cmd in ActionEngine.HIGH_RISK_COMMANDS
    
    @patch('mac_doctor.core.action_engine.subprocess.run')
    def test_execute_recommendation_user_cancellation(self, mock_run):
        """Test that user can cancel action execution."""
        recommendation = Recommendation(
            title="Cancellable Action",
            description="User can cancel this",
            action_type="command",
            command="ps aux",
            confirmation_required=True
        )
        
        with patch.object(self.engine, 'confirm_action', return_value=False):
            result = self.engine.execute_recommendation(recommendation)
            
            assert result.success is False
            assert result.error == "Action cancelled by user"
            mock_run.assert_not_called()
    
    @patch('mac_doctor.core.action_engine.subprocess.run')
    def test_execute_recommendation_exception_handling(self, mock_run):
        """Test exception handling during command execution."""
        mock_run.side_effect = Exception("Unexpected error")
        
        recommendation = Recommendation(
            title="Exception Action",
            description="This will cause an exception",
            action_type="command",
            command="ps aux",
            confirmation_required=False
        )
        
        result = self.engine.execute_recommendation(recommendation, auto_confirm=True)
        
        assert result.success is False
        assert "Unexpected error" in result.error
        
        # Verify the action was logged even though it failed
        assert len(self.engine._action_history) == 1
        assert self.engine._action_history[0]['success'] is False