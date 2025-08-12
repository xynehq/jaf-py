# JAF (Juspay Agent Framework) - Python Implementation

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://pypi.org/project/jaf-python/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

A purely functional agent framework with immutable state and composable tools, professionally converted from TypeScript to Python. JAF enables building production-ready AI agent systems with built-in security, observability, and error handling.

**🎯 Production Ready**: Complete feature parity with TypeScript version, comprehensive test suite, and production deployment support.

## ✨ Key Features

### 🏗️ **Complete TypeScript Conversion**
- ✅ **Full Feature Parity**: All TypeScript functionality converted to Python
- ✅ **Type Safety**: Pydantic models with runtime validation
- ✅ **Immutable State**: Functional programming principles preserved
- ✅ **Tool Integration**: Complete tool calling and execution system

### 🚀 **Production Ready Server**
- ✅ **FastAPI Server**: High-performance async HTTP API
- ✅ **Auto Documentation**: Interactive API docs at `/docs`
- ✅ **Health Monitoring**: Built-in health checks and metrics
- ✅ **CORS Support**: Ready for browser integration

### 🔌 **Model Context Protocol (MCP)**
- ✅ **MCP Client**: Full MCP specification support
- ✅ **WebSocket & Stdio**: Multiple transport protocols
- ✅ **Tool Integration**: Seamless MCP tool integration
- ✅ **Auto Discovery**: Dynamic tool loading from MCP servers

### 🛡️ **Enterprise Security**
- ✅ **Input Guardrails**: Content filtering and validation
- ✅ **Output Guardrails**: Response sanitization
- ✅ **Permission System**: Role-based access control
- ✅ **Audit Logging**: Complete interaction tracing

### 📊 **Observability & Monitoring**
- ✅ **Real-time Tracing**: Event-driven observability
- ✅ **Structured Logging**: JSON-formatted logs
- ✅ **Error Handling**: Comprehensive error types and recovery
- ✅ **Performance Metrics**: Built-in timing and counters

### 🔧 **Developer Experience**
- ✅ **CLI Tools**: Project initialization and management
- ✅ **Hot Reload**: Development server with auto-reload
- ✅ **Type Hints**: Full mypy compatibility
- ✅ **Rich Examples**: RAG, multi-agent, and server demos

## 🎯 Core Philosophy

- **Immutability**: All core data structures are deeply immutable
- **Pure Functions**: Core logic expressed as pure, predictable functions
- **Effects at the Edge**: Side effects isolated in Provider modules
- **Composition over Configuration**: Build complex behavior by composing simple functions
- **Type-Safe by Design**: Leverages Python's type system with Pydantic for runtime safety

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended for production)
pip install jaf-python

# Or install with all optional dependencies
pip install "jaf-python[all]"

# Install specific feature sets
pip install "jaf-python[server]"     # FastAPI server support
pip install "jaf-python[memory]"     # Redis/PostgreSQL memory providers
pip install "jaf-python[dev]"        # Development tools
```

### CLI Usage

JAF includes a powerful CLI for project management:

```bash
# Initialize a new JAF project
jaf init my-agent-project

# Run the development server
jaf server --host 0.0.0.0 --port 8000

# Show version and help
jaf version
jaf --help
```

### Development Setup

```bash
git clone https://github.com/juspay/jaf-python
cd jaf-python
pip install -e ".[dev]"

# Run tests
pytest

# Type checking and linting
mypy jaf/
ruff check jaf/
black jaf/
```

## 📁 Project Structure

```
jaf/
├── core/             # Core framework types and engine
│   ├── engine.py     # Main execution engine
│   ├── types.py      # Type definitions and schemas
│   ├── errors.py     # Error handling and types
│   ├── tool_results.py # Tool result handling
│   └── tracing.py    # Observability and event tracing
├── providers/        # External integrations (LLM, MCP)
│   ├── model.py      # LiteLLM and OpenAI providers  
│   └── mcp.py        # Model Context Protocol client
├── policies/         # Validation and security policies
│   ├── validation.py # Input/output guardrails
│   └── handoff.py    # Agent handoff policies
├── server/           # Production-ready FastAPI server
│   ├── server.py     # Main server implementation
│   ├── types.py      # Server-specific types
│   └── main.py       # Server entry point
├── cli.py            # Command-line interface
└── __init__.py       # Main exports
examples/
├── server_example.py   # Complete server demo with multiple agents
└── rag_example.py      # RAG implementation with Google AI
tests/
├── test_engine.py      # Core engine tests
└── test_validation.py  # Validation policy tests
```

## 🏗️ Key Components

### Core Types

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from jaf import Agent, Tool, RunState, run

# Define your context type
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
        result = eval(args.expression)  # Don't do this in production!
        return f"{args.expression} = {result}"

# Define an agent
def create_math_agent():
    def instructions(state):
        return 'You are a helpful math tutor'
    
    return Agent(
        name='MathTutor',
        instructions=instructions,
        tools=[CalculatorTool()]
    )
```

### Running the Framework

```python
import asyncio
from jaf import run, make_litellm_provider, generate_run_id, generate_trace_id
from jaf.core.types import RunState, RunConfig, Message

async def main():
    model_provider = make_litellm_provider('http://localhost:4000')
    math_agent = create_math_agent()
    
    config = RunConfig(
        agent_registry={'MathTutor': math_agent},
        model_provider=model_provider,
        max_turns=10,
        on_event=lambda event: print(event),  # Real-time tracing
    )
    
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role='user', content='What is 2 + 2?')],
        current_agent_name='MathTutor',
        context=MyContext(user_id='user123', permissions=['user']),
        turn_count=0,
    )
    
    result = await run(initial_state, config)
    print(result.outcome.output if result.outcome.status == 'completed' else result.outcome.error)

asyncio.run(main())
```

## 🛡️ Security & Validation

### Composable Validation Policies

```python
from jaf.policies.validation import create_path_validator, create_permission_validator, compose_validations

# Create individual validators
path_validator = create_path_validator(['/shared', '/public'])
permission_validator = create_permission_validator('admin', lambda ctx: ctx)

# Compose them
combined_validator = compose_validations(path_validator, permission_validator)

# Apply to tools
secure_file_tool = with_validation(base_file_tool, combined_validator)
```

### Guardrails

```python
from jaf.policies.validation import create_content_filter, create_rate_limiter

config = RunConfig(
    # ... other config
    initial_input_guardrails=[
        create_content_filter(),
        create_rate_limiter(10, 60000, lambda input: 'global')
    ],
    final_output_guardrails=[
        create_content_filter()
    ],
)
```

## 🔗 Agent Handoffs

```python
from jaf.policies.handoff import handoff_tool
from pydantic import BaseModel
from enum import Enum

class AgentName(str, Enum):
    MATH_TUTOR = 'MathTutor'
    FILE_MANAGER = 'FileManager'

class HandoffOutput(BaseModel):
    agent_name: AgentName

def create_triage_agent():
    def instructions(state):
        return 'Route requests to specialized agents'
    
    return Agent(
        name='TriageAgent',
        instructions=instructions,
        tools=[handoff_tool],
        handoffs=['MathTutor', 'FileManager'],  # Allowed handoff targets
        output_schema=HandoffOutput,
    )
```

## 📊 Observability

### Real-time Tracing

```python
from jaf.core.tracing import ConsoleTraceCollector, FileTraceCollector, create_composite_trace_collector

# Console logging
console_tracer = ConsoleTraceCollector()

# File logging
file_tracer = FileTraceCollector('./traces.log')

# Composite tracing
tracer = create_composite_trace_collector(console_tracer, file_tracer)

config = RunConfig(
    # ... other config
    on_event=tracer.collect,
)
```

### Error Handling

```python
from jaf.core.errors import JAFErrorHandler

if result.outcome.status == 'error':
    formatted_error = JAFErrorHandler.format(result.outcome.error)
    is_retryable = JAFErrorHandler.is_retryable(result.outcome.error)
    severity = JAFErrorHandler.get_severity(result.outcome.error)
    
    print(f"[{severity}] {formatted_error} (retryable: {is_retryable})")
```

## 🔌 Provider Integrations

### LiteLLM Provider

```python
from jaf.providers.model import make_litellm_provider

# Connect to LiteLLM proxy for 100+ model support
model_provider = make_litellm_provider(
    'http://localhost:4000',  # LiteLLM proxy URL
    'your-api-key'           # Optional API key
)
```

### MCP (Model Context Protocol) Integration

JAF includes full Model Context Protocol support for seamless tool integration:

```python
from jaf.providers.mcp import create_mcp_stdio_client, create_mcp_tools_from_client

# Connect to MCP server via stdio
mcp_client = create_mcp_stdio_client(['python', '-m', 'my_mcp_server'])

# Initialize and get all available tools
await mcp_client.initialize()
mcp_tools = await create_mcp_tools_from_client(mcp_client)

# Use MCP tools in your agent
def create_mcp_agent():
    def instructions(state):
        return "You have access to powerful MCP tools for various tasks."
    
    return Agent(
        name='MCPAgent',
        instructions=instructions,
        tools=mcp_tools  # Automatically converted JAF tools
    )

# WebSocket MCP client is also supported
from jaf.providers.mcp import create_mcp_websocket_client
ws_client = create_mcp_websocket_client('ws://localhost:8080/mcp')
```

## 🚀 Development Server

JAF includes a built-in development server for testing agents locally via HTTP endpoints:

```python
from jaf.server import run_server
from jaf.providers.model import make_litellm_provider

def create_my_agent():
    def instructions(state):
        return 'You are a helpful assistant'
    
    return Agent(
        name='MyAgent',
        instructions=instructions,
        tools=[calculator_tool, greeting_tool]
    )

model_provider = make_litellm_provider('http://localhost:4000')

# Start server on port 3000
await run_server(
    [create_my_agent()], 
    {'model_provider': model_provider},
    {'port': 3000}
)
```

Server provides RESTful endpoints:
- `GET /health` - Health check
- `GET /agents` - List available agents  
- `POST /chat` - General chat endpoint
- `POST /agents/{name}/chat` - Agent-specific endpoint

## 🎮 Example Applications

Explore the example applications to see the framework in action:

### 1. Multi-Agent Server Demo

```bash
cd examples
python server_example.py
```

**Features demonstrated:**
- ✅ Multiple specialized agents (math, weather, general)
- ✅ Tool integration (calculator, weather API)
- ✅ Agent handoffs and routing
- ✅ RESTful API with auto-documentation
- ✅ Real-time tracing and error handling
- ✅ Production-ready server configuration

**Available endpoints:**
- `GET /health` - Server health check
- `GET /agents` - List all available agents
- `POST /chat` - Chat with any agent
- `GET /docs` - Interactive API documentation

### 2. RAG (Retrieval-Augmented Generation) Demo

```bash
cd examples  
python rag_example.py
```

**Features demonstrated:**
- ✅ Knowledge base integration with semantic search
- ✅ Google Generative AI integration (optional)
- ✅ Document retrieval and context preparation
- ✅ Source citation and attribution
- ✅ Interactive and automated demo modes
- ✅ Fallback strategies for offline operation

**Interactive mode:** Ask questions about programming, ML, web development, and AI frameworks.

## 🧪 Testing

```bash
pytest          # Run tests
ruff check .    # Lint code
mypy .          # Type checking
black .         # Format code
```

## 🏛️ Architecture Principles

### Immutable State Machine
- All state transformations create new state objects
- No mutation of existing data structures
- Predictable, testable state transitions

### Type Safety
- Runtime validation with Pydantic schemas
- Compile-time safety with Python type hints
- NewType for type-safe identifiers

### Pure Functions
- Core logic is side-effect free
- Easy to test and reason about
- Deterministic behavior

### Effect Isolation
- Side effects only in Provider modules
- Clear boundaries between pure and impure code
- Easier mocking and testing

## 📜 License

MIT

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

---

**JAF (Juspay Agentic Framework) v2.0** - Building the future of functional AI agent systems 🚀
