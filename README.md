# JAF (Juspay Agent Framework) - Python Implementation

<!-- ![JAF Banner](docs/cover.png) -->

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://pypi.org/project/jaf-py/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/Docs-Live-brightgreen)](https://xynehq.github.io/jaf-py/)

A purely functional agent framework with immutable state and composable tools, professionally converted from TypeScript to Python. JAF enables building production-ready AI agent systems with built-in security, observability, and error handling.

**🎯 Production Ready**: Complete feature parity with TypeScript version, comprehensive test suite, and production deployment support.

## 📚 **[Read the Full Documentation](https://xynehq.github.io/jaf-py/)**

**[🚀 Get Started →](https://xynehq.github.io/jaf-py/getting-started/)** | **[📖 API Reference →](https://xynehq.github.io/jaf-py/api-reference/)** | **[🎮 Examples →](https://xynehq.github.io/jaf-py/examples/)**

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
- ✅ **WebSocket, Stdio, SSE & HTTP**: Multiple transport protocols
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
- ✅ **Visual Architecture**: Graphviz-powered agent and tool diagrams

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
pip install jaf-py

# Or install with all optional dependencies
pip install "jaf-py[all]"

# Install specific feature sets
pip install "jaf-py[server]"        # FastAPI server support
pip install "jaf-py[memory]"        # Redis/PostgreSQL memory providers
pip install "jaf-py[visualization]" # Graphviz visualization tools
pip install "jaf-py[dev]"           # Development tools
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
git clone https://github.com/xynehq/jaf-py
cd jaf-py
pip install -e ".[dev]"

# Run tests
pytest

# Type checking and linting
mypy jaf/
ruff check jaf/
black jaf/

# Documentation
pip install -r requirements-docs.txt
./docs.sh serve  # Start documentation server
./docs.sh deploy # Deploy to GitHub Pages
```

## 📖 Documentation

### 🌐 **[Official Documentation Website](https://xynehq.github.io/jaf-py/)**

The complete, searchable documentation is available at **[xynehq.github.io/jaf-py](https://xynehq.github.io/jaf-py/)** with:

- ✅ **Interactive navigation** with search and filtering
- ✅ **Dark/light mode** with automatic system preference detection  
- ✅ **Mobile-responsive design** for documentation on any device
- ✅ **Live code examples** with syntax highlighting
- ✅ **API reference** with auto-generated documentation
- ✅ **Always up-to-date** with automatic deployments

### 📁 **Local Documentation**

For offline access, documentation is also available in the [`docs/`](docs/) directory:

- **[📚 Documentation Hub](docs/README.md)** - Your starting point for all documentation
- **[🚀 Getting Started](docs/getting-started.md)** - Installation and first agent tutorial
- **[🏗️ Core Concepts](docs/core-concepts.md)** - JAF's functional architecture principles
- **[📋 API Reference](docs/api-reference.md)** - Complete Python API documentation
- **[🔧 Tools Guide](docs/tools.md)** - Creating and using tools
- **[💾 Memory System](docs/memory-system.md)** - Persistence and memory providers
- **[🤖 Model Providers](docs/model-providers.md)** - LiteLLM integration
- **[🌐 Server API](docs/server-api.md)** - FastAPI endpoints reference
- **[📦 Deployment](docs/deployment.md)** - Production deployment guide
- **[🎮 Examples](docs/examples.md)** - Detailed example walkthroughs
- **[🔧 Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 📁 Project Structure

```
jaf-py/
├── docs/                    # 📚 Complete documentation suite
│   ├── README.md           # Documentation hub and navigation
│   ├── cover.png          # Project banner/logo
│   ├── getting-started.md  # Installation and first steps
│   ├── core-concepts.md    # Architecture and philosophy
│   ├── api-reference.md    # Complete API documentation
│   ├── tools.md            # Tool creation and usage
│   ├── memory-system.md    # Memory providers and persistence
│   ├── model-providers.md  # LLM integration guide
│   ├── server-api.md       # FastAPI server reference
│   ├── examples.md         # Example walkthroughs
│   ├── deployment.md       # Production deployment
│   └── troubleshooting.md  # FAQ and common issues
├── jaf/                    # 🏗️ Core framework package
│   ├── core/              # Core framework types and engine
│   │   ├── __init__.py    # Core exports
│   │   ├── engine.py      # Main execution engine (run_agent)
│   │   ├── types.py       # Type definitions and Pydantic schemas
│   │   ├── errors.py      # Error handling and custom exceptions
│   │   ├── tool_results.py # Tool result handling utilities
│   │   └── tracing.py     # Observability and event tracing
│   ├── memory/            # 💾 Conversation persistence system
│   │   ├── __init__.py    # Memory exports
│   │   ├── factory.py     # Memory provider factory
│   │   ├── types.py       # Memory provider protocol
│   │   └── providers/     # Memory provider implementations
│   │       ├── __init__.py
│   │       ├── in_memory.py   # In-memory provider
│   │       ├── redis.py       # Redis provider
│   │       └── postgres.py    # PostgreSQL provider
│   ├── providers/         # 🔌 External integrations
│   │   ├── __init__.py    # Provider exports
│   │   ├── model.py       # LiteLLM and OpenAI model providers
│   │   └── mcp.py         # Model Context Protocol client
│   ├── policies/          # 🛡️ Validation and security policies
│   │   ├── __init__.py    # Policy exports
│   │   ├── validation.py  # Input/output guardrails
│   │   └── handoff.py     # Agent handoff policies
│   ├── server/            # 🌐 Production-ready FastAPI server
│   │   ├── __init__.py    # Server exports
│   │   ├── server.py      # Main server implementation
│   │   ├── types.py       # Server-specific request/response types
│   │   └── main.py        # Server entry point and configuration
│   ├── cli.py             # 💻 Command-line interface
│   └── __init__.py        # Main package exports
├── examples/              # 🎮 Example applications and demos
│   ├── server_demo.py     # Multi-agent server demo
│   └── rag_demo.py        # RAG implementation with embeddings
├── tests/                 # 🧪 Test suite
│   ├── test_engine.py     # Core engine tests
│   ├── test_validation.py # Validation policy tests
│   └── ...                # Additional test files
├── pyproject.toml         # 📦 Package configuration and dependencies
├── README.md              # 📄 This file - project overview
└── .gitignore             # Git ignore patterns
```

## 🎨 Architectural Visualization

JAF includes powerful visualization capabilities to help you understand and document your agent systems.

### Prerequisites

First, install the system Graphviz dependency:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian  
sudo apt-get install graphviz

# Windows (via Chocolatey)
choco install graphviz
```

Then install JAF with visualization support:

```bash
pip install "jaf-py[visualization]"
```

### Quick Start

```python
import asyncio
from jaf import Agent, Tool, generate_agent_graph, GraphOptions

# Create your agents
agent = Agent(
    name='MyAgent',
    instructions=lambda state: "I am a helpful assistant.",
    tools=[my_tool],
    handoffs=['OtherAgent']
)

# Generate visualization
async def main():
    result = await generate_agent_graph(
        [agent],
        GraphOptions(
            title="My Agent System",
            output_path="./my-agents.png",
            color_scheme="modern",
            show_tool_details=True
        )
    )
    
    if result.success:
        print(f"✅ Visualization saved to: {result.output_path}")
    else:
        print(f"❌ Error: {result.error}")

asyncio.run(main())
```

### Features

- **🎨 Multiple Color Schemes**: Choose from `default`, `modern`, or `minimal` themes
- **📊 Agent Architecture**: Visualize agents, tools, and handoff relationships  
- **🔧 Tool Ecosystems**: Generate dedicated tool interaction diagrams
- **🏃 Runner Architecture**: Show complete system architecture with session layers
- **📄 Multiple Formats**: Export as PNG, SVG, or PDF
- **⚙️ Customizable Layouts**: Support for various Graphviz layouts (`dot`, `circo`, `neato`, etc.)

### Example Output

The visualization system generates clear, professional diagrams showing:

- **Agent Nodes**: Rounded rectangles with agent names and model information
- **Tool Nodes**: Ellipses showing tool names and descriptions  
- **Handoff Edges**: Dashed lines indicating agent handoff relationships
- **Tool Connections**: Colored edges connecting agents to their tools
- **Cluster Organization**: Grouped components in runner architecture views

### Advanced Usage

```python
from jaf.visualization import run_visualization_examples

# Run comprehensive examples
await run_visualization_examples()

# This generates multiple example files:
# - ./examples/agent-graph.png (agent system overview)
# - ./examples/tool-graph.png (tool ecosystem)  
# - ./examples/runner-architecture.png (complete system)
# - ./examples/agent-modern.png (modern color scheme)
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

### A2A (Agent-to-Agent) Communication

JAF provides a robust A2A communication layer that allows agents to interact with each other. This is useful for building multi-agent systems where different agents have specialized skills.

```python
from jaf.a2a import create_a2a_agent, create_a2a_client, create_a2a_server

# Create agents
echo_agent = create_a2a_agent("EchoBot", "Echoes messages", "You are an echo bot.", [])
ping_agent = create_a2a_agent("PingBot", "Responds to pings", "You are a ping bot.", [])

# Create a server
server_config = {
    "agents": {"EchoBot": echo_agent, "PingBot": ping_agent},
    "agentCard": {"name": "Test Server"},
    "port": 8080,
}
server = create_a2a_server(server_config)

# Create a client
client = create_a2a_client("http://localhost:8080")
```

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

# SSE and HTTP clients are also supported
from jaf.providers.mcp import create_mcp_sse_client, create_mcp_http_client
sse_client = create_mcp_sse_client('http://localhost:8080/sse')
http_client = create_mcp_http_client('http://localhost:8080/mcp')
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

## 🔧 Function Composition

JAF supports powerful functional composition patterns that allow you to build complex behaviors from simple, reusable functions. The new object-based API makes composition even more elegant and type-safe.

### Tool Composition

Higher-order functions that enhance tool behavior:

```python
from jaf import create_function_tool, ToolSource

# Base tool
async def search_execute(args, context):
    return await perform_search(args.query)

base_tool = create_function_tool({
    'name': 'search',
    'description': 'Search for information',
    'execute': search_execute,
    'parameters': SearchArgs,
    'source': ToolSource.NATIVE
})

# Composing Tools with Higher-Order Functions
def with_cache(tool_func):
    cache = {}
    async def cached_execute(args, context):
        cache_key = str(args)
        if cache_key in cache:
            return cache[cache_key]
        result = await tool_func(args, context)
        if result.status == "success":
            cache[cache_key] = result
        return result
    return cached_execute

def with_retry(tool_func, max_retries=3):
    async def retry_execute(args, context):
        for attempt in range(max_retries):
            try:
                result = await tool_func(args, context)
                if result.status == "success":
                    return result
            except Exception:
                if attempt == max_retries - 1:
                    raise
        return result
    return retry_execute

# Compose enhancements
enhanced_search = create_function_tool({
    'name': 'enhanced_search',
    'description': 'Search with caching and retry',
    'execute': with_cache(with_retry(search_execute)),
    'parameters': SearchArgs,
    'metadata': {'enhanced': True},
    'source': ToolSource.NATIVE
})
```

### Validator Composition

Build complex validation from simple functions:

```python
def compose_validators(*validators):
    """Compose multiple validation functions into one."""
    def composed_validator(data):
        for validator in validators:
            result = validator(data)
            if not result.is_valid:
                return result
        return ValidationResult(is_valid=True)
    return composed_validator

def validate_required_fields(data):
    required = ['name', 'email']
    for field in required:
        if not hasattr(data, field) or not getattr(data, field):
            return ValidationResult(is_valid=False, error_message=f"Missing {field}")
    return ValidationResult(is_valid=True)

def validate_email_format(data):
    if not hasattr(data, 'email'):
        return ValidationResult(is_valid=True)  # Skip if no email field
    email = data.email
    if '@' not in email or '.' not in email:
        return ValidationResult(is_valid=False, error_message="Invalid email format")
    return ValidationResult(is_valid=True)

# Compose validators
user_validator = compose_validators(
    validate_required_fields,
    validate_email_format
)

# Use in tool
async def create_user_execute(args, context):
    validation = user_validator(args)
    if not validation.is_valid:
        return ToolResponse.validation_error(validation.error_message)
    # Proceed with user creation...
```

### Agent Behavior Composition

Layer agent functionality using middleware-style functions:

```python
def with_logging(agent_func):
    """Add logging to agent behavior."""
    def logged_agent(state):
        print(f"Agent {state.current_agent_name} processing message")
        return agent_func(state)
    return logged_agent

def with_rate_limiting(agent_func, max_requests=10):
    """Add rate limiting to agent behavior."""
    request_counts = {}
    def rate_limited_agent(state):
        user_id = state.context.get('user_id', 'anonymous')
        count = request_counts.get(user_id, 0)
        if count >= max_requests:
            return "Rate limit exceeded. Please try again later."
        request_counts[user_id] = count + 1
        return agent_func(state)
    return rate_limited_agent

# Compose agent behaviors
def enhanced_instructions(state):
    base_instructions = "You are a helpful assistant."
    return with_logging(
        with_rate_limiting(lambda s: base_instructions)
    )(state)
```

### Memory Provider Composition

Create layered caching strategies:

```python
def create_layered_memory(l1_provider, l2_provider):
    """Create a two-tier memory system."""
    class LayeredMemoryProvider:
        async def get_conversation(self, conversation_id):
            # Try L1 cache first
            result = await l1_provider.get_conversation(conversation_id)
            if result.data:
                return result
            
            # Fall back to L2
            result = await l2_provider.get_conversation(conversation_id)
            if result.data:
                # Populate L1 cache
                await l1_provider.store_messages(
                    conversation_id, 
                    result.data.messages
                )
            return result
        
        async def store_messages(self, conversation_id, messages, metadata=None):
            # Store in both layers
            await l1_provider.store_messages(conversation_id, messages, metadata)
            await l2_provider.store_messages(conversation_id, messages, metadata)
    
    return LayeredMemoryProvider()

# Use composed memory
from jaf.memory import create_in_memory_provider, create_redis_provider

fast_cache = create_in_memory_provider(InMemoryConfig(max_conversations=100))
persistent_store = create_redis_provider(RedisConfig(host="localhost"))
layered_memory = create_layered_memory(fast_cache, persistent_store)
```

### Pipeline Composition

Build processing pipelines:

```python
def create_pipeline(*steps):
    """Create a processing pipeline from multiple steps."""
    async def pipeline_execute(args, context):
        data = args
        for step in steps:
            result = await step(data, context)
            if hasattr(result, 'status') and result.status != 'success':
                return result  # Short-circuit on error
            data = result.data if hasattr(result, 'data') else result
        return ToolResponse.success(data)
    return pipeline_execute

# Pipeline steps
async def extract_entities(text, context):
    # Extract named entities from text
    entities = await nlp_service.extract_entities(text)
    return ToolResponse.success(entities)

async def classify_intent(entities, context):
    # Classify user intent based on entities
    intent = await classifier.predict(entities)
    return ToolResponse.success(intent)

async def generate_response(intent, context):
    # Generate appropriate response
    response = await response_generator.generate(intent)
    return ToolResponse.success(response)

# Create NLP pipeline
nlp_pipeline = create_function_tool({
    'name': 'nlp_pipeline',
    'description': 'Process text through NLP pipeline',
    'execute': create_pipeline(extract_entities, classify_intent, generate_response),
    'parameters': TextArgs,
    'source': ToolSource.NATIVE
})
```

### Benefits of Functional Composition

1. **Reusability**: Write once, compose everywhere
2. **Testability**: Each function can be tested in isolation  
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Mix and match behaviors as needed
5. **Type Safety**: Compose with full type checking
6. **Performance**: Optimize individual pieces independently

The object-based tool API makes these patterns even more powerful by providing explicit configuration points for metadata, source tracking, and behavior composition.

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
