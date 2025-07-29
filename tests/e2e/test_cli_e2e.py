"""
End-to-end tests for the Mac Doctor CLI.

These tests run the full application against a live (local) LLM
to verify that the main commands work as expected.
"""

import pytest
from typer.testing import CliRunner

from mac_doctor.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
def test_diagnose_command_e2e():
    """
    Run a full end-to-end test of the 'diagnose' command.

    This test requires Ollama to be running with a model available.
    It invokes the command and checks for a successful execution and
    the presence of key sections in the final report.
    """
    # Force the use of a local provider for consistency in tests
    args = [
        "--provider", "ollama",
        "diagnose",
        "Perform a brief, general system health check."
    ]

    result = runner.invoke(app, args)

    # Check for successful exit code
    assert result.exit_code == 0, f"CLI command failed with output: {result.stdout}"

    # Check for key elements in the output to ensure it ran completely
    assert "Mac Doctor Diagnostic Report" in result.stdout
    assert "Issues Detected" in result.stdout
    assert "Recommendations" in result.stdout
    assert "Diagnostic Summary" in result.stdout

