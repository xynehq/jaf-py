# Contributing to JAF (Juspay Agent Framework)

Thank you for your interest in contributing to JAF! We welcome contributions from the community and are pleased to have you join us.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🛠️ How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that the problem has already been reported. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include Python version, JAF version, and OS details**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Explain why this enhancement would be useful**

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issue labels:

- `good-first-issue` - Issues which should only require a few lines of code
- `help-wanted` - Issues which should be a bit more involved than beginner issues

### Pull Requests

We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## 🚀 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- pip (Python package installer)

### Setup Instructions

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/jaf-python.git
   cd jaf-python
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,memory,visualization,server]"
   ```

4. **Verify Installation**
   ```bash
   python -c "import jaf; print('JAF installed successfully!')"
   ```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test suite
python -m pytest tests/test_engine.py

# Run with coverage
python -m pytest --cov=jaf --cov-report=html

# Run fast tests only
python run_all_tests.py --suite fast
```

### Code Quality Checks

We use several tools to maintain code quality:

```bash
# Type checking
mypy jaf/

# Linting
ruff check jaf/

# Code formatting
black jaf/

# Import sorting
isort jaf/

# Run all quality checks
make lint  # or run individually
```

## 📝 Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Your Changes**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Follow the coding standards below

3. **Test Your Changes**
   ```bash
   python -m pytest
   mypy jaf/
   ruff check jaf/
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request through GitHub.

5. **PR Requirements**
   - Clear description of changes
   - Tests pass (CI will verify)
   - Code follows style guidelines
   - Documentation updated if needed
   - Reviewers assigned

## 📊 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (enforced by Black)
- **Import organization**: Use isort for consistent imports
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public functions and classes

### Code Organization

- **Functional Programming**: JAF follows functional programming principles
- **Immutability**: Core data structures should be immutable
- **Pure Functions**: Business logic should be pure functions
- **Type Safety**: Use Pydantic models for data validation

### Example Code Style

```python
from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass(frozen=True)  # Immutable
class AgentState:
    """Represents the current state of an agent execution."""
    
    agent_name: str
    turn_count: int
    messages: List[Message]
    
    def with_new_message(self, message: Message) -> 'AgentState':
        """Return new state with added message."""
        return AgentState(
            agent_name=self.agent_name,
            turn_count=self.turn_count + 1,
            messages=[*self.messages, message]
        )

class ToolArgs(BaseModel):
    """Arguments for example tool."""
    
    query: str = Field(description="The query to process")
    max_results: Optional[int] = Field(default=10, description="Maximum results")

async def process_query(args: ToolArgs, context: Context) -> str:
    """Process a query and return results.
    
    Args:
        args: Validated tool arguments
        context: Execution context
        
    Returns:
        Formatted results string
    """
    # Implementation here
    pass
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### Test Organization

```
tests/
├── test_engine.py          # Core engine tests
├── test_validation.py      # Validation policy tests
├── visualization/          # Visualization tests
│   ├── test_visualization.py
│   └── test_demo.py
└── conftest.py            # Shared test fixtures
```

### Writing Tests

```python
import pytest
from jaf import Agent, Tool
from jaf.core.types import RunState, Message

class TestAgent:
    """Test suite for Agent functionality."""
    
    def test_agent_creation(self):
        """Test basic agent creation."""
        def instructions(state: RunState) -> str:
            return "Test agent instructions"
        
        agent = Agent(
            name="TestAgent",
            instructions=instructions,
            tools=[]
        )
        
        assert agent.name == "TestAgent"
        assert callable(agent.instructions)

    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """Test agent execution with mock provider."""
        # Test implementation
        pass
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_functionality():
    pass

@pytest.mark.integration
async def test_integration_flow():
    pass

@pytest.mark.slow
def test_performance_benchmark():
    pass
```

## 📚 Documentation

### Documentation Standards

- **API Documentation**: Use clear docstrings with type information
- **Examples**: Include practical examples in docstrings
- **README Updates**: Update README.md for user-facing changes
- **Changelog**: Add entries to CHANGELOG.md for significant changes

### Documentation Structure

```
docs/
├── README.md               # Documentation hub
├── getting-started.md      # Installation and quickstart
├── core-concepts.md        # Architecture and philosophy
├── api-reference.md        # Complete API documentation
├── tools.md               # Tool creation guide
├── memory-system.md       # Memory providers
├── model-providers.md     # LLM integration
├── server-api.md          # FastAPI server reference
├── examples.md            # Example walkthroughs
├── deployment.md          # Production deployment
└── troubleshooting.md     # FAQ and common issues
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
./docs.sh build

# Serve locally
./docs.sh serve

# Deploy to GitHub Pages
./docs.sh deploy
```

## 🏗️ Architecture Guidelines

### Core Principles

1. **Immutability**: All core data structures are deeply immutable
2. **Pure Functions**: Core logic expressed as pure, predictable functions
3. **Effects at the Edge**: Side effects isolated in Provider modules
4. **Composition**: Build complex behavior by composing simple functions
5. **Type Safety**: Leverage Python's type system with runtime validation

### Module Organization

```
jaf/
├── core/              # Core framework (immutable, pure)
│   ├── types.py      # Core data types
│   ├── engine.py     # Main execution engine
│   └── errors.py     # Error handling
├── providers/        # External integrations (side effects)
│   ├── model.py      # LLM providers
│   └── mcp.py        # MCP integration
├── policies/         # Composable policies
│   ├── validation.py # Input/output validation
│   └── handoff.py    # Agent handoff rules
└── memory/           # Persistence layer
    └── providers/    # Memory implementations
```

## 🔧 Performance Considerations

- **Async/Await**: Use async patterns for I/O operations
- **Memory Efficiency**: Be mindful of object creation in hot paths
- **Caching**: Cache expensive computations appropriately
- **Profiling**: Profile performance-critical code paths

## 🐛 Debugging

### Common Issues

1. **Import Errors**: Check virtual environment activation
2. **Test Failures**: Ensure all dependencies are installed
3. **Type Errors**: Run mypy to catch type issues early
4. **Performance**: Use profiling tools for bottlenecks

### Debug Tools

```bash
# Verbose test output
python -m pytest -v -s

# Debug specific test
python -m pytest tests/test_engine.py::test_specific_function -v -s

# Python debugger
python -m pdb your_script.py
```

## 🌟 Recognition

Contributors who make significant contributions will be:

- Added to the CONTRIBUTORS.md file
- Mentioned in release notes
- Recognized in project documentation

## 📞 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community support
- **Documentation**: Check the comprehensive docs first

### Communication Guidelines

- Be respectful and constructive
- Provide clear, actionable feedback
- Help others learn and grow
- Follow the Code of Conduct

## 🎯 Project Roadmap

Check our [GitHub Projects](https://github.com/juspay/jaf-python/projects) for current priorities and upcoming features.

---

Thank you for contributing to JAF! Your contributions help make this project better for everyone. 🚀