# Juspay Agentic Framework (JAF-PY)

A production-ready, purely functional framework for building robust and type-safe AI agents in Python.

<div style="margin: 2rem 0;">
  <a href="getting-started/" class="md-button md-button--primary" style="margin-right: 1rem;">Get Started</a>
  <a href="https://github.com/xynehq/jaf-py" class="md-button">GitHub</a>
</div>

## Overview

JAF follows functional programming principles for predictable, testable AI systems:

- **Immutable State**: All core data structures are deeply immutable
- **Pure Functions**: Core logic is side-effect free and predictable  
- **Type Safety**: Leverages Python's type system with Pydantic
- **Effects at Edge**: Side effects isolated in Provider modules

## Quick Example

```python
from jaf import Agent, function_tool

@function_tool
async def calculate(expression: str, context) -> str:
    """Perform safe arithmetic calculations.
    
    Args:
        expression: Math expression to evaluate (e.g., '2 + 3', '10 * 5')
    """
    allowed_chars = set('0123456789+-*/(). ')
    if not all(c in allowed_chars for c in expression):
        return 'Error: Invalid characters'
    try:
        result = eval(expression)
        return f'{expression} = {result}'
    except Exception:
        return 'Error: Invalid expression'

agent = Agent(
    name='MathAgent',
    instructions=lambda state: 'You are a helpful math assistant.',
    tools=[calculate]
)
```

## Installation

```bash
pip install git+https://github.com/xynehq/jaf-py.git
```

## Documentation

### Core Framework
- [Getting Started](getting-started.md) - Build your first agent
- [Core Concepts](core-concepts.md) - Understand the architecture  
- [API Reference](api-reference.md) - Complete documentation

### Advanced Features
- [Agent-as-Tool](agent-as-tool.md) - Hierarchical agent orchestration
- [Tracing & Observability](tracing.md) - Monitor and debug agents
- [Memory System](memory-system.md) - Conversation persistence
- [Server API](server-api.md) - HTTP API deployment

### Guides & Examples
- [Examples](examples.md) - Working code samples
- [Tools Guide](tools.md) - Building custom tools
- [Deployment](deployment.md) - Production deployment