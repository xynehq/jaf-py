# JAF Python Documentation

<div class="hero">
  <h1>🚀 JAF Python Framework</h1>
  <p>Functional Agent Framework for building production-ready AI agent systems</p>
</div>

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://pypi.org/project/jaf-python/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://juspay.github.io/jaf-python/)

A **production-ready** functional agent framework with immutable state and composable tools, featuring enterprise-grade security, real database integration, and comprehensive LLM provider support. JAF transforms from prototype to production with robust error handling, input sanitization, and functional programming principles.

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
    <h3>🚀 Production Ready Infrastructure</h3>
    <ul>
      <li>✅ FastAPI Server with A2A Protocol</li>
      <li>✅ Redis & PostgreSQL Session Providers</li>
      <li>✅ Multi-LLM Provider Support</li>
      <li>✅ Real Streaming Implementation</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🔌 ADK Production Framework</h3>
    <ul>
      <li>✅ Agent Development Kit (ADK)</li>
      <li>✅ Safe Math Evaluation (AST-based)</li>
      <li>✅ Circuit Breakers & Retries</li>
      <li>✅ Configuration Management</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🛡️ Enterprise Security Framework</h3>
    <ul>
      <li>✅ Multi-Level Input Sanitization</li>
      <li>✅ Authentication & Authorization</li>
      <li>✅ Injection Attack Protection</li>
      <li>✅ Safe Code Execution</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>📊 Functional Programming Core</h3>
    <ul>
      <li>✅ Immutable Data Structures</li>
      <li>✅ Pure Functions & Composability</li>
      <li>✅ Thread-Safe Operations</li>
      <li>✅ Side-Effect Isolation</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🎣 Advanced Callback System</h3>
    <ul>
      <li>✅ 14+ Instrumentation Hooks</li>
      <li>✅ ReAct Pattern Support</li>
      <li>✅ Iterative Agent Behaviors</li>
      <li>✅ LLM Call Interception</li>
    </ul>
  </div>

  <div class="feature-card">
    <h3>🔧 Production Quality Assurance</h3>
    <ul>
      <li>✅ Comprehensive Validation Suite</li>
      <li>✅ Real API Integration Tests</li>
      <li>✅ Security Vulnerability Scanning</li>
      <li>✅ Performance Optimization</li>
    </ul>
  </div>
</div>

## 🔥 Production Transformation Highlights

!!! success "From Prototype to Production"
    JAF has undergone comprehensive transformation from a sophisticated mock-up to a **production-ready enterprise framework**:

<div class="transformation-grid">
  <div class="before-after">
    <h4>🔒 Security Overhaul</h4>
    <div class="comparison">
      <div class="before">
        <strong>Before:</strong> 3/10
        <ul>
          <li>❌ Dangerous eval() usage</li>
          <li>❌ No input sanitization</li>
          <li>❌ Missing authentication</li>
        </ul>
      </div>
      <div class="after">
        <strong>After:</strong> 9/10
        <ul>
          <li>✅ AST-based safe evaluation</li>
          <li>✅ Multi-level input sanitization</li>
          <li>✅ Enterprise auth framework</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="before-after">
    <h4>🧠 Functional Programming</h4>
    <div class="comparison">
      <div class="before">
        <strong>Before:</strong> 4/10
        <ul>
          <li>❌ Mutable state everywhere</li>
          <li>❌ Side effects mixed with logic</li>
          <li>❌ Thread safety concerns</li>
        </ul>
      </div>
      <div class="after">
        <strong>After:</strong> 8/10
        <ul>
          <li>✅ Immutable data structures</li>
          <li>✅ Pure functions isolated</li>
          <li>✅ Thread-safe by design</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="before-after">
    <h4>🏭 Infrastructure</h4>
    <div class="comparison">
      <div class="before">
        <strong>Before:</strong> 6/10
        <ul>
          <li>❌ Mock providers only</li>
          <li>❌ No real database support</li>
          <li>❌ Limited error handling</li>
        </ul>
      </div>
      <div class="after">
        <strong>After:</strong> 8/10
        <ul>
          <li>✅ Redis & PostgreSQL support</li>
          <li>✅ Multi-LLM providers</li>
          <li>✅ Circuit breakers & retries</li>
        </ul>
      </div>
    </div>
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
- **[Function Composition](function-composition.md)** - Advanced functional patterns
- **[Memory System](memory-system.md)** - Persistence and memory providers
- **[Model Providers](model-providers.md)** - LiteLLM integration

### 🛡️ ADK Production Framework
- **[ADK Overview](adk-overview.md)** - Agent Development Kit introduction
- **[Callback System](callback-system.md)** - Advanced agent instrumentation and control
- **[Security Framework](security-framework.md)** - Input sanitization and safe execution
- **[Session Management](session-management.md)** - Immutable sessions and functional patterns
- **[Error Handling](error-handling.md)** - Circuit breakers, retries, and recovery

### 🚀 Production Deployment
- **[Server API](server-api.md)** - FastAPI and A2A protocol endpoints
- **[Infrastructure](infrastructure.md)** - Database providers and configuration
- **[Deployment](deployment.md)** - Production deployment guide
- **[Validation Suite](validation-suite.md)** - Comprehensive testing and validation

### 📚 Learning
- **[Examples](examples.md)** - Detailed example walkthroughs
- **[Flight Booking System](flight-booking-example.md)** - Multi-agent production example

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
        # Production-safe evaluation using AST-based parser
        from adk.utils.safe_evaluator import safe_calculate
        result = safe_calculate(args.expression)
        if result["status"] == "success":
            return f"{args.expression} = {result['result']}"
        else:
            return f"Error: {result['error']}"

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