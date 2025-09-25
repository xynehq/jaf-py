# JAF - Juspay Agent Framework (Python)

JAF is a purely functional framework for building AI agents with immutable state and composable tools. It provides enterprise-grade features for building production-ready AI applications.

<div style="margin: 2rem 0;">
  <a href="getting-started/" class="md-button md-button--primary" style="margin-right: 1rem;">Get Started</a>
  <a href="https://github.com/xynehq/jaf-py" class="md-button">GitHub</a>
</div>

## Key Features

### Core Framework
- **Functional Architecture** - Pure functions, immutable state, predictable behavior  
- **Type Safety** - Full TypeScript-style typing with runtime validation
- **Tool Composition** - Build complex behaviors from simple, reusable components
- **Memory System** - Persistent conversation storage with multiple providers
- **Model Agnostic** - Works with OpenAI, Google Gemini, Claude, and more via LiteLLM

### Advanced Features  
- **Streaming Responses** - Real-time content delivery with Server-Sent Events
- **Performance Monitoring** - Built-in metrics and observability
- **Error Handling** - Circuit breakers, retries, and graceful degradation
- **Security Framework** - Input validation, guardrails, and secure execution
- **Plugin System** - Extensible architecture for custom functionality

### Enterprise Ready
- **A2A Protocol** - Agent-to-agent communication for multi-agent systems
- **Workflow Orchestration** - Complex automation with conditional logic  
- **Analytics System** - Real-time insights and conversation quality metrics
- **MCP Integration** - Model Context Protocol for external tool integration
- **Production Deployment** - FastAPI server with auto-documentation and monitoring

## Quick Start

### Installation

```bash
pip install jaf-py
```

### Your First Agent

```python
from jaf import Agent, run, create_function_tool, FunctionToolConfig
from pydantic import BaseModel

# Define a tool
class CalculatorArgs(BaseModel):
    expression: str

def calculate(args: CalculatorArgs, context) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Simple calculator - in production, use a safer evaluator
        result = eval(args.expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

calculator_tool = create_function_tool(FunctionToolConfig(
    name="calculate",
    description="Evaluate mathematical expressions",
    execute=calculate,
    parameters=CalculatorArgs
))

# Create agent
agent = Agent(
    name="math_agent",
    instructions="You are a helpful math assistant. Use the calculator tool for computations.",
    tools=[calculator_tool]
)

# Run agent
result = run(
    agent=agent,
    messages=[{"role": "user", "content": "What is 15 * 23 + 7?"}]
)

print(result.messages[-1]["content"])
## Architecture Overview

JAF follows functional programming principles with immutable data structures:

```python
# Everything is immutable and composable
agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant",
    model="gpt-4",
    tools=[tool1, tool2],
    memory=memory_provider
)

# State flows through pure functions
result = run(
    agent=agent,
    messages=messages,
    config=RunConfig(max_turns=5)
)

# Results are immutable data structures
print(f"Status: {result.status}")
print(f"Messages: {len(result.messages)}")
```

## Core Concepts

### 1. Agents
Agents are immutable configurations that define behavior:

```python
from jaf import Agent

agent = Agent(
    name="customer_service",
    instructions="Help customers with their questions",
    model="gpt-4",
    tools=[search_tool, email_tool],
    memory=memory_provider
)
```

### 2. Tools  
Tools are composable functions that agents can call:

```python
from jaf import create_function_tool
from pydantic import BaseModel

class EmailArgs(BaseModel):
    to: str
    subject: str
    body: str

email_tool = create_function_tool({
    "name": "send_email",
    "description": "Send an email",
    "execute": send_email_impl,
    "parameters": EmailArgs
})
```

### 3. Memory
Persistent conversation storage across sessions:

```python
from jaf.memory import create_memory_provider_from_env

# Supports Redis, PostgreSQL, and in-memory
memory = create_memory_provider_from_env()

agent = Agent(
    name="persistent_agent", 
    memory=memory,
    # ... other config
)
```

## Advanced Examples

### Multi-Agent System

```python
from jaf import Agent, run

# Specialized agents
researcher = Agent(
    name="researcher",
    instructions="Research topics thoroughly",
    tools=[search_tool, web_scraper_tool]
)

writer = Agent(
    name="writer", 
    instructions="Write clear, engaging content",
    tools=[grammar_tool, style_tool]
)

# Orchestrator agent using others as tools
orchestrator = Agent(
    name="content_creator",
    instructions="Create high-quality content using research and writing specialists",
    tools=[researcher.as_tool(), writer.as_tool()]
)
```

### Streaming Server

```python
from jaf import run_streaming
from jaf.server import run_server

# Stream responses in real-time
async def handle_chat(message: str):
    async for event in run_streaming(agent=agent, messages=messages):
        if event.type == "assistant_message":
            yield event.data["content"]

# Production server
if __name__ == "__main__":
    run_server(
        agents={"assistant": agent},
        host="0.0.0.0",
        port=8000
    )
```

## Next Steps

### Learning Path

1. **[Getting Started](getting-started.md)** - Complete installation and first agent tutorial
2. **[Core Concepts](core-concepts.md)** - Understand JAF's functional architecture  
3. **[Tools Guide](tools.md)** - Learn to create and compose tools
4. **[Examples](examples.md)** - Explore real-world applications

### Key Documentation

- **[API Reference](api-reference.md)** - Complete function and class reference
- **[Memory System](memory-system.md)** - Persistent storage and providers
- **[MCP Integration](mcp.md)** - External tool integration 
- **[Deployment](deployment.md)** - Production deployment guide
- **[Performance](performance-monitoring.md)** - Monitoring and optimization

### Community

- **GitHub**: [xynehq/jaf-py](https://github.com/xynehq/jaf-py)
- **Issues**: [Bug reports and feature requests](https://github.com/xynehq/jaf-py/issues)
- **Discussions**: [Community support and questions](https://github.com/xynehq/jaf-py/discussions)

JAF empowers developers to build sophisticated AI agents with the reliability and maintainability of functional programming. Get started today!