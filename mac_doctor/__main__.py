"""
Mac Doctor CLI entry point.

This module allows Mac Doctor to be run as a module:
    python -m mac_doctor
"""

from mac_doctor.cli.main import app

if __name__ == "__main__":
    app()