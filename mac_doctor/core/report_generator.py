"""
Report generator for Mac Doctor diagnostic results.

This module provides functionality to format diagnostic results into various
output formats (markdown, JSON) and export them to files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..interfaces import DiagnosticResult, Issue, Recommendation, MCPResult


class ReportGenerator:
    """Generates formatted reports from diagnostic results."""
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    def generate_markdown(self, result: DiagnosticResult) -> str:
        """
        Generate a markdown-formatted report from diagnostic results.
        
        Args:
            result: The diagnostic result to format
            
        Returns:
            Formatted markdown report as a string
        """
        lines = []
        
        # Header
        lines.append("# Mac Doctor Diagnostic Report")
        lines.append("")
        lines.append(f"**Generated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Query:** {result.query}")
        lines.append(f"**Execution Time:** {result.execution_time:.2f} seconds")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        if result.issues_detected:
            critical_count = sum(1 for issue in result.issues_detected if issue.severity == 'critical')
            high_count = sum(1 for issue in result.issues_detected if issue.severity == 'high')
            medium_count = sum(1 for issue in result.issues_detected if issue.severity == 'medium')
            low_count = sum(1 for issue in result.issues_detected if issue.severity == 'low')
            
            lines.append(f"Found **{len(result.issues_detected)}** issues:")
            if critical_count > 0:
                lines.append(f"- 🔴 **{critical_count}** Critical")
            if high_count > 0:
                lines.append(f"- 🟠 **{high_count}** High")
            if medium_count > 0:
                lines.append(f"- 🟡 **{medium_count}** Medium")
            if low_count > 0:
                lines.append(f"- 🟢 **{low_count}** Low")
        else:
            lines.append("✅ **No issues detected** - Your system appears to be running normally.")
        lines.append("")
        
        # Issues Detected
        if result.issues_detected:
            lines.append("## Issues Detected")
            lines.append("")
            
            for i, issue in enumerate(result.issues_detected, 1):
                severity_icon = self._get_severity_icon(issue.severity)
                lines.append(f"### {i}. {severity_icon} {issue.title}")
                lines.append("")
                lines.append(f"**Severity:** {issue.severity.title()}")
                lines.append(f"**Category:** {issue.category.title()}")
                lines.append("")
                lines.append(issue.description)
                lines.append("")
                
                if issue.affected_processes:
                    lines.append("**Affected Processes:**")
                    for process in issue.affected_processes:
                        lines.append(f"- {process}")
                    lines.append("")
                
                if issue.metrics:
                    lines.append("**Metrics:**")
                    for metric, value in issue.metrics.items():
                        lines.append(f"- {metric}: {value}")
                    lines.append("")
        
        # Recommendations
        if result.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            
            for i, rec in enumerate(result.recommendations, 1):
                risk_icon = self._get_risk_icon(rec.risk_level)
                lines.append(f"### {i}. {risk_icon} {rec.title}")
                lines.append("")
                lines.append(rec.description)
                lines.append("")
                
                if rec.command:
                    lines.append("**Action:**")
                    if rec.action_type == 'sudo_command':
                        lines.append("```bash")
                        lines.append(f"sudo {rec.command}")
                        lines.append("```")
                    else:
                        lines.append("```bash")
                        lines.append(rec.command)
                        lines.append("```")
                    lines.append("")
                
                lines.append(f"**Risk Level:** {rec.risk_level.title()}")
                if rec.confirmation_required:
                    lines.append("**⚠️ Confirmation Required:** This action requires user approval before execution.")
                lines.append("")
        
        # Tool Results
        if result.tool_results:
            lines.append("## Diagnostic Tool Results")
            lines.append("")
            
            for tool_name, tool_result in result.tool_results.items():
                lines.append(f"### {tool_name}")
                lines.append("")
                
                if tool_result.success:
                    lines.append("✅ **Status:** Success")
                    lines.append(f"**Execution Time:** {tool_result.execution_time:.2f}s")
                    lines.append("")
                    
                    # Add key data points if available
                    if tool_result.data:
                        lines.append("**Key Findings:**")
                        self._add_tool_data_summary(lines, tool_result.data)
                        lines.append("")
                else:
                    lines.append("❌ **Status:** Failed")
                    if tool_result.error:
                        lines.append(f"**Error:** {tool_result.error}")
                    lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Mac Doctor - macOS System Diagnostic Tool*")
        
        return "\n".join(lines)
    
    def generate_json(self, result: DiagnosticResult) -> str:
        """
        Generate a JSON-formatted report from diagnostic results.
        
        Args:
            result: The diagnostic result to format
            
        Returns:
            Formatted JSON report as a string
        """
        report_data = {
            "metadata": {
                "generated_at": result.timestamp.isoformat(),
                "query": result.query,
                "execution_time": result.execution_time,
                "tool_count": len(result.tool_results),
                "issue_count": len(result.issues_detected),
                "recommendation_count": len(result.recommendations)
            },
            "summary": {
                "issues_by_severity": self._count_issues_by_severity(result.issues_detected),
                "issues_by_category": self._count_issues_by_category(result.issues_detected),
                "recommendations_by_risk": self._count_recommendations_by_risk(result.recommendations)
            },
            "issues": [self._issue_to_dict(issue) for issue in result.issues_detected],
            "recommendations": [self._recommendation_to_dict(rec) for rec in result.recommendations],
            "tool_results": {
                name: self._mcp_result_to_dict(result) 
                for name, result in result.tool_results.items()
            }
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def export_to_file(self, content: str, path: str) -> None:
        """
        Export report content to a file.
        
        Args:
            content: The report content to write
            path: The file path to write to
            
        Raises:
            IOError: If the file cannot be written
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            raise IOError(f"Failed to export report to {path}: {str(e)}")
    
    def format_recommendations(self, recommendations: List[Recommendation]) -> str:
        """
        Format a list of recommendations for display.
        
        Args:
            recommendations: List of recommendations to format
            
        Returns:
            Formatted recommendations as a string
        """
        if not recommendations:
            return "No recommendations available."
        
        lines = []
        for i, rec in enumerate(recommendations, 1):
            risk_icon = self._get_risk_icon(rec.risk_level)
            lines.append(f"{i}. {risk_icon} {rec.title}")
            lines.append(f"   {rec.description}")
            
            if rec.command:
                action_prefix = "sudo " if rec.action_type == 'sudo_command' else ""
                lines.append(f"   Action: {action_prefix}{rec.command}")
            
            if rec.confirmation_required:
                lines.append("   ⚠️  Requires confirmation")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_severity_icon(self, severity: str) -> str:
        """Get an icon for the given severity level."""
        icons = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        return icons.get(severity.lower(), '⚪')
    
    def _get_risk_icon(self, risk_level: str) -> str:
        """Get an icon for the given risk level."""
        icons = {
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        return icons.get(risk_level.lower(), 'ℹ️')
    
    def _add_tool_data_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add a detailed summary of tool data to the lines list."""
        if not isinstance(data, dict):
            lines.append(f"- Raw data: {str(data)[:100]}...")
            return
        
        # Handle different tool data formats
        if "system_stats" in data or "top_processes" in data:
            self._add_process_summary(lines, data)
        elif "disk_analysis" in data or "mountpoints" in data:
            self._add_disk_summary(lines, data)
        elif "memory_stats" in data or "vm_stats" in data:
            self._add_memory_summary(lines, data)
        elif "network_stats" in data or "connections" in data:
            self._add_network_summary(lines, data)
        elif "logs" in data or "log_entries" in data:
            self._add_logs_summary(lines, data)
        elif "dtrace_results" in data or "syscalls" in data:
            self._add_dtrace_summary(lines, data)
        else:
            # Generic fallback for unknown data formats
            self._add_generic_summary(lines, data)
    
    def _add_process_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add process tool data summary."""
        if "system_stats" in data:
            stats = data["system_stats"]
            lines.append(f"- CPU Usage: {stats.get('cpu_percent', 'N/A'):.1f}%")
            lines.append(f"- Memory Usage: {stats.get('memory_percent', 'N/A'):.1f}%")
            lines.append(f"- Load Average: {stats.get('load_avg', 'N/A')}")
            lines.append(f"- Process Count: {stats.get('process_count', 'N/A')}")
        
        if "top_processes" in data and data["top_processes"]:
            lines.append("- **Top CPU Consumers:**")
            for i, proc in enumerate(data["top_processes"][:3], 1):
                cpu = proc.get('cpu_percent', 0)
                mem = proc.get('memory_mb', 0)
                name = proc.get('name', 'Unknown')
                lines.append(f"  {i}. {name}: {cpu:.1f}% CPU, {mem:.1f}MB RAM")
    
    def _add_disk_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add disk tool data summary."""
        if "disk_analysis" in data:
            analysis = data["disk_analysis"]
            if "storage" in analysis:
                storage = analysis["storage"]
                lines.append(f"- Disk Usage: {storage.get('usage_percent', 'N/A'):.1f}%")
                lines.append(f"- Available Space: {storage.get('available_gb', 'N/A'):.1f}GB")
                lines.append(f"- Total Space: {storage.get('total_gb', 'N/A'):.1f}GB")
            
            if "io_stats" in analysis:
                io = analysis["io_stats"]
                lines.append(f"- Read Operations: {io.get('read_ops', 'N/A')}")
                lines.append(f"- Write Operations: {io.get('write_ops', 'N/A')}")
        
        if "mountpoints" in data:
            lines.append(f"- Mounted Filesystems: {len(data['mountpoints'])}")
    
    def _add_memory_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add memory tool data summary."""
        if "memory_stats" in data:
            stats = data["memory_stats"]
            lines.append(f"- Memory Pressure: {stats.get('memory_pressure', 'N/A')}")
            lines.append(f"- Swap Usage: {stats.get('swap_used_mb', 'N/A'):.1f}MB")
            lines.append(f"- Page Faults: {stats.get('page_faults', 'N/A')}")
    
    def _add_network_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add network tool data summary."""
        if "network_stats" in data:
            stats = data["network_stats"]
            lines.append(f"- Bytes Sent: {stats.get('bytes_sent', 'N/A')}")
            lines.append(f"- Bytes Received: {stats.get('bytes_recv', 'N/A')}")
            lines.append(f"- Active Connections: {stats.get('connections_count', 'N/A')}")
    
    def _add_logs_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add logs tool data summary."""
        if "log_entries" in data:
            entries = data["log_entries"]
            lines.append(f"- Log Entries Analyzed: {len(entries)}")
            if entries:
                error_count = sum(1 for entry in entries if 'error' in entry.get('level', '').lower())
                warning_count = sum(1 for entry in entries if 'warning' in entry.get('level', '').lower())
                lines.append(f"- Errors Found: {error_count}")
                lines.append(f"- Warnings Found: {warning_count}")
    
    def _add_dtrace_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add DTrace tool data summary."""
        if "dtrace_results" in data:
            results = data["dtrace_results"]
            lines.append(f"- System Calls Traced: {results.get('syscall_count', 'N/A')}")
            lines.append(f"- Trace Duration: {results.get('duration_seconds', 'N/A')}s")
    
    def _add_generic_summary(self, lines: List[str], data: Dict[str, any]) -> None:
        """Add generic summary for unknown data formats."""
        # Show top-level keys and some sample values
        for key, value in list(data.items())[:5]:  # Limit to first 5 items
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value}")
            elif isinstance(value, str) and len(value) < 50:
                lines.append(f"- {key}: {value}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {len(value)} items")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {len(value)} fields")
            else:
                lines.append(f"- {key}: {type(value).__name__}")
    
    def _count_issues_by_severity(self, issues: List[Issue]) -> Dict[str, int]:
        """Count issues by severity level."""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        return counts
    
    def _count_issues_by_category(self, issues: List[Issue]) -> Dict[str, int]:
        """Count issues by category."""
        counts = {}
        for issue in issues:
            counts[issue.category] = counts.get(issue.category, 0) + 1
        return counts
    
    def _count_recommendations_by_risk(self, recommendations: List[Recommendation]) -> Dict[str, int]:
        """Count recommendations by risk level."""
        counts = {'high': 0, 'medium': 0, 'low': 0}
        for rec in recommendations:
            if rec.risk_level in counts:
                counts[rec.risk_level] += 1
        return counts
    
    def _issue_to_dict(self, issue: Issue) -> Dict[str, any]:
        """Convert an Issue to a dictionary."""
        return {
            'severity': issue.severity,
            'category': issue.category,
            'title': issue.title,
            'description': issue.description,
            'affected_processes': issue.affected_processes,
            'metrics': issue.metrics
        }
    
    def _recommendation_to_dict(self, rec: Recommendation) -> Dict[str, any]:
        """Convert a Recommendation to a dictionary."""
        return {
            'title': rec.title,
            'description': rec.description,
            'action_type': rec.action_type,
            'command': rec.command,
            'risk_level': rec.risk_level,
            'confirmation_required': rec.confirmation_required
        }
    
    def _mcp_result_to_dict(self, result: MCPResult) -> Dict[str, any]:
        """Convert an MCPResult to a dictionary."""
        return {
            'tool_name': result.tool_name,
            'success': result.success,
            'data': result.data,
            'error': result.error,
            'execution_time': result.execution_time,
            'metadata': result.metadata
        }