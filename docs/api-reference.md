# API Reference

This comprehensive reference documents all public APIs in the JAF (Juspay Agent Framework) Python implementation. All examples assume you have imported JAF as shown:

```python
import jaf
from jaf import RunState, Agent, Message, RunConfig
```

## Enums

JAF provides comprehensive enums for type safety and to eliminate magic strings throughout your code.

### ContentRole

Defines the roles for messages in conversations.

```python
from jaf import ContentRole

class ContentRole(str, Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL = 'tool'
    SYSTEM = 'system'
```

**Usage:**
```python
message = Message(role=ContentRole.USER, content="Hello!")
```

### ToolSource

Specifies the source of tool definitions.

```python
from jaf import ToolSource

class ToolSource(str, Enum):
    NATIVE = 'native'
    MCP = 'mcp'
    PLUGIN = 'plugin'
    EXTERNAL = 'external'
```

### Model

Supported model identifiers.

```python
from jaf import Model

class Model(str, Enum):
    GEMINI_2_0_FLASH = 'gemini-2.0-flash'
    GEMINI_2_5_PRO = 'gemini-2.5-pro'
    GEMINI_PRO = 'gemini-pro'
    GPT_4 = 'gpt-4'
    GPT_4_TURBO = 'gpt-4-turbo'
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    CLAUDE_3_SONNET = 'claude-3-sonnet'
    CLAUDE_3_HAIKU = 'claude-3-haiku'
    CLAUDE_3_OPUS = 'claude-3-opus'
```

### ToolParameterType

Types for tool parameter definitions.

```python
from jaf import ToolParameterType

class ToolParameterType(str, Enum):
    STRING = 'string'
    NUMBER = 'number'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    ARRAY = 'array'
    OBJECT = 'object'
    NULL = 'null'
```

### PartType

Message part types for multimodal content.

```python
from jaf import PartType

class PartType(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio'
    VIDEO = 'video'
    FILE = 'file'
```

## Tool Creation Functions

JAF provides both modern object-based and legacy positional APIs for creating tools.

### create_function_tool (Recommended)

Create a function-based tool using object configuration for better type safety and extensibility.

```python
from jaf import create_function_tool, ToolSource
from pydantic import BaseModel, Field

class GreetArgs(BaseModel):
    name: str = Field(description="Name to greet")

async def greet_execute(args: GreetArgs, context) -> str:
    return f"Hello, {args.name}!"

tool = create_function_tool({
    'name': 'greet',
    'description': 'Greets a user by name',
    'execute': greet_execute,
    'parameters': GreetArgs,
    'metadata': {'category': 'social'},
    'source': ToolSource.NATIVE
})
```

**Parameters:**
- `config: FunctionToolConfig` - Object containing:
  - `name: str` - Tool name
  - `description: str` - Tool description
  - `execute: ToolExecuteFunction` - Function to execute
  - `parameters: Any` - Pydantic model for parameter validation
  - `metadata: Optional[Dict[str, Any]]` - Optional metadata
  - `source: Optional[ToolSource]` - Tool source (defaults to NATIVE)

**Returns:**
- `Tool` - Tool implementation ready for use with agents

### create_function_tool_legacy (Deprecated)

Legacy positional argument API for backward compatibility.

```python
# Deprecated - use object-based API instead
tool = create_function_tool_legacy(
    'greet',
    'Greets a user by name', 
    greet_execute,
    GreetArgs,
    {'category': 'social'},
    ToolSource.NATIVE
)
```

!!! warning "Deprecated"
    This function is deprecated. Use `create_function_tool` with object configuration for better type safety and extensibility.

### create_async_function_tool

Convenience function identical to `create_function_tool` but with a name that emphasizes async execution.

```python
tool = create_async_function_tool({
    'name': 'async_operation',
    'description': 'Performs an async operation',
    'execute': async_execute_func,
    'parameters': AsyncArgs,
    'source': ToolSource.NATIVE
})
```

### FunctionToolConfig

TypedDict defining the configuration structure for object-based tool creation.

```python
from jaf.core.types import FunctionToolConfig

class FunctionToolConfig(TypedDict):
    name: str
    description: str
    execute: ToolExecuteFunction
    parameters: Any
    metadata: Optional[Dict[str, Any]]
    source: Optional[ToolSource]
```

## Core Functions

### Main Execution

#### `run(initial_state: RunState[Ctx], config: RunConfig[Ctx]) -> RunResult[Out]`

Main execution function for running agents with functional purity and immutable state.

**Parameters:**
- `initial_state: RunState[Ctx]` - Initial state containing messages, context, and agent info
- `config: RunConfig[Ctx]` - Configuration including agents, model provider, and guardrails

**Returns:**
- `RunResult[Out]` - Contains final state and outcome (completed or error)

**Example:**
```python
result = await jaf.run(initial_state, config)
if result.outcome.status == 'completed':
    print(f"Success: {result.outcome.output}")
else:
    print(f"Error: {result.outcome.error}")
```

### ID Generation

#### `generate_run_id() -> RunId`

Generate a new unique run ID using UUID4.

**Returns:**
- `RunId` - Branded string type for run identification

#### `generate_trace_id() -> TraceId`

Generate a new unique trace ID using UUID4.

**Returns:**
- `TraceId` - Branded string type for trace identification

#### `create_run_id(id_str: str) -> RunId`

Create a RunId from an existing string.

#### `create_trace_id(id_str: str) -> TraceId`

Create a TraceId from an existing string.

## Core Types

### RunState

#### `RunState[Ctx]`

Immutable state of a run containing all execution context.

**Type Parameters:**
- `Ctx` - Type of the context object

**Fields:**
- `run_id: RunId` - Unique identifier for this run
- `trace_id: TraceId` - Unique identifier for tracing
- `messages: List[Message]` - Conversation history
- `current_agent_name: str` - Name of the currently active agent
- `context: Ctx` - User-defined context object
- `turn_count: int` - Number of execution turns
- `final_response: Optional[str]` - Final agent response (if completed)

**Example:**
```python
@dataclass
class MyContext:
    user_id: str
    permissions: List[str]

state = RunState(
    run_id=jaf.generate_run_id(),
    trace_id=jaf.generate_trace_id(),
    messages=[Message(role='user', content='Hello!')],
    current_agent_name='assistant',
    context=MyContext(user_id='123', permissions=['read']),
    turn_count=0
)
```

### Agent

#### `Agent[Ctx, Out]`

An agent definition with instructions, tools, and configuration.

**Type Parameters:**
- `Ctx` - Type of the context object
- `Out` - Type of the expected output

**Fields:**
- `name: str` - Unique name for the agent
- `instructions: Callable[[RunState[Ctx]], str]` - Function that generates system prompt
- `tools: Optional[List[Tool[Any, Ctx]]]` - Available tools for the agent
- `output_codec: Optional[Any]` - Pydantic model for output validation
- `handoffs: Optional[List[str]]` - List of agents this agent can handoff to
- `model_config: Optional[ModelConfig]` - Model-specific configuration

**Example:**
```python
def create_assistant(context_type):
    def instructions(state: RunState[context_type]) -> str:
        return f"You are a helpful assistant. User: {state.context.user_id}"
    
    return Agent(
        name='Assistant',
        instructions=instructions,
        tools=[calculator_tool, weather_tool],
        handoffs=['SpecialistAgent']
    )
```

### Tool Protocol

#### `Tool[Args, Ctx]`

Protocol for tool implementations.

**Type Parameters:**
- `Args` - Type of the tool arguments (Pydantic model)
- `Ctx` - Type of the context object

**Required Attributes:**
- `schema: ToolSchema[Args]` - Tool schema with name, description, and parameters

**Required Methods:**
- `execute(args: Args, context: Ctx) -> Union[str, ToolResult]` - Execute the tool

**Example:**
```python
class CalculatorArgs(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool:
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'calculate',
            'description': 'Perform mathematical calculations',
            'parameters': CalculatorArgs
        })()
    
    async def execute(self, args: CalculatorArgs, context: MyContext) -> str:
        result = eval(args.expression)  # Use safe evaluator in production
        return f"Result: {result}"
```

### Message

#### `Message`

A message in the conversation.

**Fields:**
- `role: Literal['user', 'assistant', 'tool']` - Message sender role
- `content: str` - Message content
- `tool_call_id: Optional[str]` - ID for tool response messages
- `tool_calls: Optional[List[ToolCall]]` - Tool calls from assistant

**Example:**
```python
user_message = Message(role='user', content='What is 2+2?')
assistant_message = Message(role='assistant', content='Let me calculate that for you.')
```

### RunConfig

#### `RunConfig[Ctx]`

Configuration for running agents.

**Fields:**
- `agent_registry: Dict[str, Agent[Ctx, Any]]` - Available agents by name
- `model_provider: ModelProvider[Ctx]` - LLM provider implementation
- `max_turns: Optional[int]` - Maximum execution turns (default: 50)
- `model_override: Optional[str]` - Override model name for all agents
- `initial_input_guardrails: Optional[List[Guardrail]]` - Input validation
- `final_output_guardrails: Optional[List[Guardrail]]` - Output validation
- `on_event: Optional[Callable[[TraceEvent], None]]` - Event handler for tracing
- `memory: Optional[MemoryConfig]` - Memory provider configuration
- `conversation_id: Optional[str]` - Conversation identifier for memory

**Example:**
```python
config = RunConfig(
    agent_registry={'assistant': my_agent},
    model_provider=jaf.make_litellm_provider('http://localhost:4000'),
    max_turns=20,
    on_event=lambda event: print(f"Event: {event.type}"),
    initial_input_guardrails=[content_filter],
    final_output_guardrails=[output_validator]
)
```

### RunResult

#### `RunResult[Out]`

Result of a run execution.

**Fields:**
- `final_state: RunState[Any]` - Final state after execution
- `outcome: RunOutcome[Out]` - Success with output or error details

**Outcome Types:**
- `CompletedOutcome[Out]` - Success with `output: Out`
- `ErrorOutcome` - Error with `error: JAFError`

### ValidationResult

#### `ValidationResult`

Union type for validation results.

**Types:**
- `ValidValidationResult` - `is_valid: True`
- `InvalidValidationResult` - `is_valid: False, error_message: str`

### ModelConfig

#### `ModelConfig`

Configuration for model behavior.

**Fields:**
- `name: Optional[str]` - Model name (e.g., "gpt-4o")
- `temperature: Optional[float]` - Sampling temperature (0.0-1.0)
- `max_tokens: Optional[int]` - Maximum tokens to generate

## Model Provider Functions

### LiteLLM Provider

#### `make_litellm_provider(base_url: str, api_key: str = "anything") -> ModelProvider[Ctx]`

Create a LiteLLM-compatible model provider for OpenAI-compatible APIs.

**Parameters:**
- `base_url: str` - Base URL for the LiteLLM server
- `api_key: str` - API key (defaults to "anything" for local servers)

**Returns:**
- `ModelProvider[Ctx]` - Provider instance implementing the ModelProvider protocol

**Examples:**
```python
# Local LiteLLM server
provider = jaf.make_litellm_provider("http://localhost:4000")

# OpenAI API
provider = jaf.make_litellm_provider(
    "https://api.openai.com/v1", 
    api_key="your-openai-api-key"
)

# Custom LiteLLM deployment
provider = jaf.make_litellm_provider(
    "https://your-litellm-server.com/v1",
    api_key="your-api-key"
)
```

### ModelProvider Protocol

#### `ModelProvider[Ctx]`

Protocol defining the interface for model providers.

**Methods:**
- `get_completion(state: RunState[Ctx], agent: Agent[Ctx, Any], config: RunConfig[Ctx]) -> ModelCompletionResponse` - Get completion from the model

## Memory Provider System

### Memory Provider Factory

#### `create_memory_provider_from_env(external_clients: Optional[Dict[str, Any]] = None) -> MemoryProvider`

Create a memory provider based on environment variables.

**Environment Variables:**
- `JAF_MEMORY_TYPE`: "memory", "redis", or "postgres" (default: "memory")
- **Redis**: `JAF_REDIS_URL`, `JAF_REDIS_HOST`, `JAF_REDIS_PORT`, `JAF_REDIS_DB`
- **PostgreSQL**: `JAF_POSTGRES_CONNECTION_STRING`, `JAF_POSTGRES_HOST`, `JAF_POSTGRES_PORT`, etc.
- **In-Memory**: `JAF_MEMORY_MAX_CONVERSATIONS`, `JAF_MEMORY_MAX_MESSAGES`

**Parameters:**
- `external_clients: Optional[Dict[str, Any]]` - Pre-initialized client connections

**Returns:**
- `MemoryProvider` - Configured memory provider

**Example:**
```python
# Set environment variable
import os
os.environ['JAF_MEMORY_TYPE'] = 'redis'
os.environ['JAF_REDIS_URL'] = 'redis://localhost:6379'

# Create provider
memory_provider = jaf.create_memory_provider_from_env()

# Use in run config
config = RunConfig(
    # ... other config
    memory=MemoryConfig(provider=memory_provider)
)
```

### Specific Provider Creators

#### `create_in_memory_provider(config: InMemoryConfig) -> MemoryProvider`

Create an in-memory provider for development and testing.

#### `create_redis_provider(config: RedisConfig, client: Optional[Any] = None) -> MemoryProvider`

Create a Redis memory provider for distributed scenarios.

#### `create_postgres_provider(config: PostgresConfig, client: Optional[Any] = None) -> MemoryProvider`

Create a PostgreSQL memory provider for production persistence.

### Memory Provider Protocol

#### `MemoryProvider`

Protocol defining the interface for memory providers.

**Key Methods:**
- `store_messages(conversation_id: str, messages: List[Message], metadata: Optional[Dict[str, Any]] = None) -> Result` - Store messages
- `get_conversation(conversation_id: str) -> Union[ConversationMemory, None]` - Retrieve conversation
- `append_messages(conversation_id: str, messages: List[Message], metadata: Optional[Dict[str, Any]] = None) -> Result` - Append messages
- `get_recent_messages(conversation_id: str, limit: int = 50) -> List[Message]` - Get recent messages
- `delete_conversation(conversation_id: str) -> bool` - Delete conversation
- `health_check() -> Dict[str, Any]` - Check provider health

### Memory Configuration Types

#### `MemoryConfig`

Configuration for memory integration.

**Fields:**
- `provider: MemoryProvider` - Memory provider instance
- `auto_store: bool` - Automatically store conversations (default: True)
- `max_messages: Optional[int]` - Message limit for storage
- `ttl: Optional[int]` - Time-to-live in seconds
- `compression_threshold: Optional[int]` - Compression threshold

#### `ConversationMemory`

Immutable conversation memory object.

**Fields:**
- `conversation_id: str` - Unique conversation identifier
- `user_id: Optional[str]` - User identifier
- `messages: List[Message]` - Conversation messages
- `metadata: Optional[Dict[str, Any]]` - Additional metadata

## Validation and Policy Functions

### Input/Output Guardrails

#### `create_length_guardrail(max_length: int, min_length: int = 0) -> Guardrail`

Create a guardrail that validates text length.

**Example:**
```python
length_guard = jaf.create_length_guardrail(max_length=1000, min_length=10)

config = RunConfig(
    # ... other config
    initial_input_guardrails=[length_guard]
)
```

#### `create_content_filter_guardrail(blocked_patterns: List[str], case_sensitive: bool = False) -> Guardrail`

Create a guardrail that filters content based on regex patterns.

**Example:**
```python
content_filter = jaf.create_content_filter_guardrail([
    r'\b(password|secret|api_key)\b',
    r'\d{16}',  # Credit card patterns
])

config = RunConfig(
    # ... other config
    initial_input_guardrails=[content_filter]
)
```

#### `create_json_validation_guardrail(schema_class: type[BaseModel]) -> Guardrail`

Create a guardrail that validates JSON against a Pydantic schema.

**Example:**
```python
class OrderOutput(BaseModel):
    order_id: str
    total: float
    items: List[str]

json_validator = jaf.create_json_validation_guardrail(OrderOutput)

config = RunConfig(
    # ... other config
    final_output_guardrails=[json_validator]
)
```

#### `combine_guardrails(guardrails: List[Guardrail], require_all: bool = True) -> Guardrail`

Combine multiple guardrails into a single guardrail.

**Example:**
```python
combined_guard = jaf.combine_guardrails([
    length_guard,
    content_filter,
    json_validator
], require_all=True)
```

### Handoff Policies

#### `create_handoff_guardrail(policy: HandoffPolicy, current_agent: str) -> Guardrail`

Create a guardrail that validates agent handoffs.

#### `create_role_based_handoff_policy(agent_roles: Dict[str, str], role_permissions: Dict[str, List[str]]) -> HandoffPolicy`

Create a handoff policy based on agent roles.

**Example:**
```python
# Define roles
agent_roles = {
    "TriageAgent": "triage",
    "TechnicalAgent": "technical", 
    "BillingAgent": "billing"
}

# Define permissions (which roles can handoff to which)
role_permissions = {
    "triage": ["technical", "billing"],
    "technical": ["triage"],
    "billing": ["triage"]
}

handoff_policy = jaf.create_role_based_handoff_policy(agent_roles, role_permissions)
```

## Server Functions

### Server Creation

#### `run_server(config: ServerConfig) -> None`

Start a JAF server with the given configuration.

**Example:**
```python
from jaf.server import ServerConfig

server_config = ServerConfig(
    agent_registry={'assistant': my_agent},
    run_config=run_config,
    host='0.0.0.0',
    port=3000,
    cors=True
)

await jaf.run_server(server_config)
```

#### `create_simple_server(agents: List[Agent], model_provider: ModelProvider, host: str = 'localhost', port: int = 3000) -> JAFServer`

Create a simple JAF server with minimal configuration.

**Example:**
```python
server = jaf.create_simple_server(
    agents=[assistant_agent, specialist_agent],
    model_provider=jaf.make_litellm_provider('http://localhost:4000'),
    host='0.0.0.0',
    port=8000
)

await server.start()
```

## Tool Result System

### ToolResult Type

#### `ToolResult[T]`

Standardized tool result with status, data, and metadata.

**Fields:**
- `status: ToolResultStatus` - Status ('success', 'error', 'validation_error', etc.)
- `data: Optional[T]` - Result data for successful operations
- `error: Optional[ToolErrorInfo]` - Error information for failures
- `metadata: Optional[ToolMetadata]` - Execution metadata

### ToolResponse Helper Class

#### `ToolResponse`

Helper functions for creating standardized tool results.

**Static Methods:**
- `success(data: T, metadata: Optional[Dict] = None) -> ToolResult[T]` - Create success result
- `error(code: str, message: str, details: Optional[Any] = None) -> ToolResult[None]` - Create error result
- `validation_error(message: str, details: Optional[Any] = None) -> ToolResult[None]` - Create validation error
- `permission_denied(message: str, required_permissions: Optional[List[str]] = None) -> ToolResult[None]` - Create permission denied error
- `not_found(resource: str, identifier: Optional[str] = None) -> ToolResult[None]` - Create not found error

**Example:**
```python
class DatabaseTool:
    async def execute(self, args: QueryArgs, context: Context) -> ToolResult[Dict]:
        try:
            if not context.has_permission('database_read'):
                return ToolResponse.permission_denied(
                    "Database access requires read permission",
                    required_permissions=['database_read']
                )
            
            result = await self.db.query(args.sql)
            return ToolResponse.success(
                data={'rows': result.rows, 'count': len(result.rows)},
                metadata={'execution_time_ms': result.duration}
            )
            
        except DatabaseError as e:
            return ToolResponse.error(
                code='database_error',
                message=str(e),
                details={'error_code': e.code}
            )
```

### Utility Functions

#### `with_error_handling(tool_name: str, executor: Callable) -> Callable`

Tool execution wrapper that provides standardized error handling.

#### `tool_result_to_string(result: ToolResult[Any]) -> str`

Convert ToolResult to string for backward compatibility.

## Tracing System

### TraceCollector Protocol

#### `TraceCollector`

Protocol for trace collectors.

**Methods:**
- `collect(event: TraceEvent) -> None` - Collect a trace event
- `get_trace(trace_id: TraceId) -> List[TraceEvent]` - Get events for a specific trace
- `clear(trace_id: Optional[TraceId] = None) -> None` - Clear traces

### Built-in Collectors

#### `ConsoleTraceCollector`

Console trace collector with detailed logging.

**Example:**
```python
tracer = jaf.ConsoleTraceCollector()

config = RunConfig(
    # ... other config
    on_event=tracer.collect
)
```

#### `FileTraceCollector(file_path: str)`

File trace collector that writes events to a file.

**Example:**
```python
file_tracer = jaf.FileTraceCollector('./traces.jsonl')

config = RunConfig(
    # ... other config
    on_event=file_tracer.collect
)
```

## Complete Example

Here's a complete example showing how to use the main APIs together:

```python
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, Field
import jaf

@dataclass
class UserContext:
    user_id: str
    permissions: List[str]

class CalculateArgs(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool:
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'calculate',
            'description': 'Perform safe mathematical calculations',
            'parameters': CalculateArgs
        })()
    
    async def execute(self, args: CalculateArgs, context: UserContext) -> str:
        if 'calculator' not in context.permissions:
            return jaf.ToolResponse.permission_denied(
                "Calculator access denied",
                required_permissions=['calculator']
            ).format()
        
        try:
            # Use safe evaluation in production
            result = eval(args.expression)
            return jaf.ToolResponse.success(
                f"Result: {args.expression} = {result}"
            ).format()
        except Exception as e:
            return jaf.ToolResponse.error(
                'calculation_error', 
                str(e)
            ).format()

def create_math_agent():
    def instructions(state: jaf.RunState[UserContext]) -> str:
        return f"""You are a helpful math assistant for user {state.context.user_id}.
        You can perform calculations using the calculate tool.
        Always explain your reasoning clearly."""
    
    return jaf.Agent(
        name='MathAssistant',
        instructions=instructions,
        tools=[CalculatorTool()]
    )

async def main():
    # Set up tracing
    tracer = jaf.ConsoleTraceCollector()
    
    # Create model provider
    model_provider = jaf.make_litellm_provider('http://localhost:4000')
    
    # Create memory provider
    memory_provider = jaf.create_memory_provider_from_env()
    
    # Create agent
    math_agent = create_math_agent()
    
    # Set up configuration
    config = jaf.RunConfig(
        agent_registry={'MathAssistant': math_agent},
        model_provider=model_provider,
        max_turns=10,
        on_event=tracer.collect,
        memory=jaf.MemoryConfig(provider=memory_provider),
        conversation_id='user_123_session',
        initial_input_guardrails=[
            jaf.create_length_guardrail(max_length=500)
        ]
    )
    
    # Create initial state
    initial_state = jaf.RunState(
        run_id=jaf.generate_run_id(),
        trace_id=jaf.generate_trace_id(),
        messages=[jaf.Message(role='user', content='What is 15 * 8 + 32?')],
        current_agent_name='MathAssistant',
        context=UserContext(user_id='user_123', permissions=['calculator']),
        turn_count=0
    )
    
    # Run the agent
    result = await jaf.run(initial_state, config)
    
    # Handle result
    if result.outcome.status == 'completed':
        print(f"✅ Success: {result.outcome.output}")
    else:
        print(f"❌ Error: {result.outcome.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

This API reference provides comprehensive documentation for building sophisticated AI agent systems with JAF's functional architecture, type safety, and production-ready features.