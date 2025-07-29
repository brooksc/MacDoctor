# Contributing to Mac Doctor

Thank you for your interest in contributing to Mac Doctor! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/MacDoctor.git
   cd MacDoctor
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   
   # Install in development mode
   pip install -e ".[dev]"
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   pytest
   
   # Try the CLI
   mac-doctor --help
   ```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mac_doctor --cov-report=html

# Run specific test file
pytest tests/unit/test_process_mcp.py

# Run with verbose output
pytest -v
```

### Writing Tests
- All new code should include comprehensive tests
- Tests should cover success cases, error conditions, and edge cases
- Use descriptive test names that explain what is being tested
- Mock external dependencies (system commands, network calls, etc.)

Example test structure:
```python
class TestNewFeature:
    def test_feature_success_case(self):
        """Test that feature works correctly under normal conditions."""
        # Arrange
        # Act
        # Assert
        
    def test_feature_error_handling(self):
        """Test that feature handles errors gracefully."""
        # Test error scenarios
```

## 🎨 Code Style

### Formatting
We use automated code formatting tools:

```bash
# Format code
black mac_doctor/ tests/

# Sort imports
isort mac_doctor/ tests/

# Run linting
flake8 mac_doctor/ tests/

# Type checking
mypy mac_doctor/
```

### Style Guidelines
- **Line Length**: 88 characters (Black default)
- **Imports**: Use isort with black-compatible profile
- **Type Hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings
- **Variable Names**: Use descriptive names, avoid abbreviations

### Example Code Style
```python
from typing import Dict, List, Optional

def analyze_system_data(
    data: Dict[str, Any], 
    threshold: float = 0.8
) -> Optional[List[str]]:
    """Analyze system data and return recommendations.
    
    Args:
        data: System metrics dictionary
        threshold: Performance threshold (0.0 to 1.0)
        
    Returns:
        List of recommendations or None if no issues found
        
    Raises:
        ValueError: If threshold is out of range
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    # Implementation here
    return recommendations
```

## 🏗️ Architecture Guidelines

### Mac Collector Plugins (MCPs)
When creating new MCPs:

1. **Inherit from BaseMCP**
   ```python
   from mac_doctor.interfaces import BaseMCP, MCPResult
   
   class NewMCP(BaseMCP):
       @property
       def name(self) -> str:
           return "descriptive_name"
   ```

2. **Implement Required Methods**
   - `is_available()`: Check if MCP can run
   - `execute()`: Collect and return data
   - `get_schema()`: Return parameter schema

3. **Error Handling**
   - Always return `MCPResult` objects
   - Handle missing tools gracefully
   - Provide meaningful error messages

4. **Testing**
   - Mock system commands and external dependencies
   - Test availability checking
   - Test successful execution and error cases

### LLM Providers
When adding new LLM providers:

1. **Inherit from BaseLLM**
2. **Implement required methods**
3. **Handle API keys securely**
4. **Support privacy mode**
5. **Add comprehensive tests**

## 📝 Documentation

### Code Documentation
- Use clear, descriptive docstrings
- Document all public APIs
- Include examples for complex functions
- Keep documentation up to date with code changes

### README Updates
- Update feature lists when adding new capabilities
- Add new configuration options
- Update installation instructions if needed

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues to avoid duplicates
2. Test with the latest version
3. Gather system information

### Bug Report Template
```markdown
**Describe the Bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected Behavior**
What you expected to happen.

**System Information**
- macOS Version: [e.g. 14.0]
- Python Version: [e.g. 3.11.5]
- Mac Doctor Version: [e.g. 1.0.0]
- Architecture: [Intel/Apple Silicon]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Submitting
1. Check if the feature already exists
2. Consider if it fits Mac Doctor's scope
3. Think about implementation complexity

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why would this feature be useful?

**Proposed Implementation**
Any ideas on how this could be implemented?

**Additional Context**
Any other context or screenshots.
```

## 🔄 Pull Request Process

### Before Submitting
1. **Create an Issue**: Discuss the change first
2. **Fork the Repository**: Work on your own fork
3. **Create a Branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/new-mcp-plugin
   git checkout -b fix/memory-leak-issue
   ```

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No API keys or sensitive data included
- [ ] Commit messages are clear and descriptive

### Pull Request Template
```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🏷️ Commit Messages

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add network monitoring MCP plugin"
git commit -m "Fix memory leak in process monitoring"
git commit -m "Update README with new installation instructions"

# Avoid
git commit -m "Fix bug"
git commit -m "Update code"
git commit -m "Changes"
```

## 🤝 Code Review

### For Reviewers
- Be constructive and helpful
- Focus on code quality and maintainability
- Check for security issues
- Verify tests are comprehensive
- Ensure documentation is updated

### For Contributors
- Respond to feedback promptly
- Ask questions if feedback is unclear
- Make requested changes in separate commits
- Update the PR description if scope changes

## 📋 Release Process

### Version Numbering
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release tag
5. Update documentation

## 🆘 Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/brooksc/MacDoctor/discussions)
- **Issues**: Use [GitHub Issues](https://github.com/brooksc/MacDoctor/issues)
- **Chat**: Join our community discussions

## 📜 Code of Conduct

### Our Pledge
We are committed to making participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Enforcement
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

---

Thank you for contributing to Mac Doctor! 🎉