# JAF Python Documentation

<div class="hero">
  <h1>🚀 JAF Python Framework</h1>
  <p>Functional Agent Framework for building production-ready AI agent systems</p>
</div>

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://pypi.org/project/jaf-python/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://juspay.github.io/jaf-python/)

A purely functional agent framework with immutable state and composable tools, professionally converted from TypeScript to Python. JAF enables building production-ready AI agent systems with built-in security, observability, and error handling.

!!! tip "Quick Start"
    New to JAF? Start with our [Getting Started Guide](getting-started.md) to build your first agent in minutes!

## ✨ Key Features

<div class="feature-grid">
  <div class="feature-card">
    <h3>🏗️ Complete TypeScript Conversion</h3>
    <ul>
      <li>✅ Full Feature Parity</li>
      <li>✅ Type Safety with Pydantic</li>
      <li>✅ Immutable State Management</li>
      <li>✅ Tool Integration System</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🚀 Production Ready Server</h3>
    <ul>
      <li>✅ FastAPI Server</li>
      <li>✅ Auto Documentation</li>
      <li>✅ Health Monitoring</li>
      <li>✅ CORS Support</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🔌 Model Context Protocol</h3>
    <ul>
      <li>✅ Full MCP Support</li>
      <li>✅ WebSocket & Stdio</li>
      <li>✅ Tool Integration</li>
      <li>✅ Auto Discovery</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🛡️ Enterprise Security</h3>
    <ul>
      <li>✅ Input Guardrails</li>
      <li>✅ Output Guardrails</li>
      <li>✅ Permission System</li>
      <li>✅ Audit Logging</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>📊 Observability & Monitoring</h3>
    <ul>
      <li>✅ Real-time Tracing</li>
      <li>✅ Structured Logging</li>
      <li>✅ Error Handling</li>
      <li>✅ Performance Metrics</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🔧 Developer Experience</h3>
    <ul>
      <li>✅ CLI Tools</li>
      <li>✅ Hot Reload</li>
      <li>✅ Type Hints</li>
      <li>✅ Rich Examples</li>
    </ul>
  </div>
</div>

## 🎯 Core Philosophy

JAF follows functional programming principles for predictable, testable AI systems:

- **Immutability**: All core data structures are deeply immutable
- **Pure Functions**: Core logic is side-effect free and predictable  
- **Effects at the Edge**: Side effects isolated in Provider modules
- **Composition over Configuration**: Build complex behavior by composing simple functions
- **Type-Safe by Design**: Leverages Python's type system with Pydantic

## 🚀 Quick Installation

=== "PyPI (Recommended)"

    ```bash
    # Install from PyPI
    pip install jaf-python
    
    # Or with all optional dependencies
    pip install "jaf-python[all]"
    ```

=== "Development"

    ```bash
    git clone https://github.com/juspay/jaf-python
    cd jaf-python
    pip install -e ".[dev]"
    ```

=== "Docker"

    ```bash
    docker pull juspay/jaf-python:latest
    docker run -p 8000:8000 juspay/jaf-python
    ```

## 📖 Documentation Structure

### 🏗️ Foundation
- **[Getting Started](getting-started.md)** - Installation and first agent tutorial
- **[Core Concepts](core-concepts.md)** - JAF's functional architecture principles  
- **[API Reference](api-reference.md)** - Complete Python API documentation

### 🔧 Building with JAF
- **[Tools Guide](tools.md)** - Creating and using tools
- **[Memory System](memory-system.md)** - Persistence and memory providers
- **[Model Providers](model-providers.md)** - LiteLLM integration

### 🚀 Production
- **[Server API](server-api.md)** - FastAPI endpoints reference
- **[Deployment](deployment.md)** - Production deployment guide

### 📚 Learning
- **[Examples](examples.md)** - Detailed example walkthroughs

## 🎮 Quick Example

Here's a simple agent that can perform calculations:

```python
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, Field
from jaf import Agent, run, make_litellm_provider
from jaf.core.types import RunState, RunConfig, Message

# Define context type
@dataclass  
class MyContext:
    user_id: str
    permissions: list[str]

# Define tool schema
class CalculateArgs(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

# Create a tool
class CalculatorTool:
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'calculate',
            'description': 'Perform mathematical calculations',
            'parameters': CalculateArgs
        })()
    
    async def execute(self, args: CalculateArgs, context: MyContext) -> str:
        # Safe evaluation (don't use eval in production!)
        result = eval(args.expression)
        return f"{args.expression} = {result}"

# Create an agent
def create_math_agent():
    def instructions(state):
        return 'You are a helpful math tutor. Use the calculator tool for math problems.'
    
    return Agent(
        name='MathTutor',
        instructions=instructions,
        tools=[CalculatorTool()]
    )

# Run the framework
async def main():
    model_provider = make_litellm_provider('http://localhost:4000')
    math_agent = create_math_agent()
    
    config = RunConfig(
        agent_registry={'MathTutor': math_agent},
        model_provider=model_provider,
        max_turns=10
    )
    
    initial_state = RunState(
        run_id='demo_run',
        trace_id='demo_trace', 
        messages=[Message(role='user', content='What is 15 * 7?')],
        current_agent_name='MathTutor',
        context=MyContext(user_id='demo_user', permissions=['user']),
        turn_count=0
    )
    
    result = await run(initial_state, config)
    print(result.outcome.output if result.outcome.status == 'completed' else result.outcome.error)

# Run the example
# asyncio.run(main())
```

## 🎨 Architecture Visualization

JAF includes powerful visualization capabilities to document your agent systems:

```python
from jaf import generate_agent_graph, GraphOptions

# Generate system diagram
result = await generate_agent_graph(
    [math_agent],
    GraphOptions(
        title="Math Tutoring System",
        output_path="./system.png",
        color_scheme="modern"
    )
)
```

## 🌟 What's Next?

Choose your path based on your goals:

!!! tip "I want to..."
    
    **Build my first agent** → [Getting Started](getting-started.md)
    
    **Understand the architecture** → [Core Concepts](core-concepts.md)
    
    **Create custom tools** → [Tools Guide](tools.md)
    
    **Add persistence** → [Memory System](memory-system.md)
    
    **Deploy to production** → [Server API](server-api.md) + [Deployment](deployment.md)
    
    **See working examples** → [Examples](examples.md)

## 🤝 Community & Support

- **[GitHub Repository](https://github.com/juspay/jaf-python)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/jaf-python/)** - Official releases  
- **[Examples](examples.md)** - Working code samples
- **[API Reference](api-reference.md)** - Complete documentation

---

<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: var(--md-default-fg-color--lightest); border-radius: 8px;">
  <strong>JAF Python v2.0</strong> - Building the future of functional AI agent systems 🚀
</div>