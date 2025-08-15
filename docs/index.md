# Juspay Agentic Framework (JAF-PY)

A production-ready, purely functional framework for building robust and type-safe AI agents in Python.

<div style="margin: 2rem 0;">
  <a href="getting-started.md" class="md-button md-button--primary" style="margin-right: 1rem;">Get Started</a>
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
from jaf import Agent, create_function_tool
from pydantic import BaseModel

class CalculateArgs(BaseModel):
    expression: str

async def calculate(args: CalculateArgs, context) -> str:
    allowed_chars = set('0123456789+-*/(). ')
    if not all(c in allowed_chars for c in args.expression):
        return 'Error: Invalid characters'
    try:
        result = eval(args.expression)
        return f'{args.expression} = {result}'
    except Exception:
        return 'Error: Invalid expression'

agent = Agent(
    name='MathAgent',
    instructions=lambda state: 'You are a helpful math assistant.',
    tools=[create_function_tool({
        'name': 'calculate',
        'description': 'Perform calculations',
        'execute': calculate,
        'parameters': CalculateArgs
    })]
)
```

## Installation

```bash
pip install jaf-py
```

## Documentation

- [Getting Started](getting-started.md) - Build your first agent
- [Core Concepts](core-concepts.md) - Understand the architecture  
- [API Reference](api-reference.md) - Complete documentation
- [Examples](examples.md) - Working code samples