#!/usr/bin/env python3
"""
Basic usage example for Mac Doctor.

This script demonstrates how to use Mac Doctor programmatically
to perform system diagnostics and get AI-powered recommendations.
"""

import asyncio
import json
from pathlib import Path

from mac_doctor.agent.diagnostic_agent import DiagnosticAgent
from mac_doctor.config import ConfigManager
from mac_doctor.core.report_generator import ReportGenerator
from mac_doctor.llm.factory import LLMFactory
from mac_doctor.tool_registry import ToolRegistry


async def main():
    """Run a basic Mac Doctor diagnostic."""
    print("🩺 Mac Doctor - Basic Usage Example")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Initialize LLM factory
        llm_factory = LLMFactory(config)
        
        # Check available providers
        providers = llm_factory.get_available_providers()
        print(f"Available LLM providers: {list(providers.keys())}")
        
        if not any(providers.values()):
            print("❌ No LLM providers available. Please set up Ollama or Gemini.")
            print("   Run: mac-doctor setup")
            return
        
        # Initialize tool registry and agent
        registry = ToolRegistry()
        agent = DiagnosticAgent(llm_factory, registry)
        
        # Run diagnostic
        print("\n🔍 Running system diagnostic...")
        result = await agent.diagnose_system()
        
        if result.success:
            print(f"✅ Diagnostic completed successfully!")
            print(f"   Found {len(result.issues)} issues")
            print(f"   Generated {len(result.recommendations)} recommendations")
            
            # Generate report
            report_generator = ReportGenerator()
            
            # Console output
            print("\n📊 System Analysis Summary:")
            print("-" * 30)
            for issue in result.issues[:3]:  # Show first 3 issues
                print(f"• {issue.title} ({issue.severity})")
            
            if len(result.issues) > 3:
                print(f"  ... and {len(result.issues) - 3} more issues")
            
            print("\n💡 Top Recommendations:")
            print("-" * 25)
            for rec in result.recommendations[:3]:  # Show first 3 recommendations
                print(f"• {rec.title} (Risk: {rec.risk})")
            
            if len(result.recommendations) > 3:
                print(f"  ... and {len(result.recommendations) - 3} more recommendations")
            
            # Save detailed report
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            
            # Markdown report
            md_report = report_generator.generate_markdown(result)
            md_path = output_dir / "diagnostic_report.md"
            with open(md_path, 'w') as f:
                f.write(md_report)
            print(f"\n📄 Detailed report saved to: {md_path}")
            
            # JSON report for programmatic use
            json_report = report_generator.generate_json(result)
            json_path = output_dir / "diagnostic_report.json"
            with open(json_path, 'w') as f:
                f.write(json_report)
            print(f"📄 JSON report saved to: {json_path}")
            
        else:
            print(f"❌ Diagnostic failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Error running diagnostic: {e}")
        print("   Try running with debug mode: mac-doctor diagnose --debug")


if __name__ == "__main__":
    asyncio.run(main())