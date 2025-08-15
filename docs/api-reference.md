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

## ADK Callback System

### RunnerCallbacks Protocol

#### `RunnerCallbacks`

Protocol defining hooks for advanced agent instrumentation and control.

**Available Hooks:**

```python
from adk.runners import RunnerCallbacks, RunnerConfig, execute_agent

class MyCallbacks:
    """Custom callback implementation for advanced agent behaviors."""
    
    # === Lifecycle Hooks ===
    async def on_start(self, context: RunContext, message: Message, session_state: Dict[str, Any]) -> None:
        """Called when agent execution starts."""
        pass
    
    async def on_complete(self, response: AgentResponse) -> None:
        """Called when execution completes successfully."""
        pass
    
    async def on_error(self, error: Exception, context: RunContext) -> None:
        """Called when execution encounters an error."""
        pass
    
    # === LLM Interaction Hooks ===
    async def on_before_llm_call(self, agent: Agent, message: Message, session_state: Dict[str, Any]) -> Optional[LLMControlResult]:
        """Modify or skip LLM calls."""
        return None
    
    async def on_after_llm_call(self, response: Message, session_state: Dict[str, Any]) -> Optional[Message]:
        """Modify LLM responses."""
        return None
    
    # === Iteration Control Hooks ===
    async def on_iteration_start(self, iteration: int) -> Optional[IterationControlResult]:
        """Control iteration flow."""
        return None
    
    async def on_iteration_complete(self, iteration: int, has_tool_calls: bool) -> Optional[IterationControlResult]:
        """Decide whether to continue iterating."""
        return None
    
    # === Tool Execution Hooks ===
    async def on_before_tool_selection(self, tools: List[Tool], context_data: List[Any]) -> Optional[ToolSelectionControlResult]:
        """Filter or modify available tools."""
        return None
    
    async def on_tool_selected(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Track tool usage."""
        pass
    
    async def on_before_tool_execution(self, tool: Tool, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Modify parameters or skip execution."""
        return None
    
    async def on_after_tool_execution(self, tool: Tool, result: Any, error: Optional[Exception] = None) -> Optional[Any]:
        """Process tool results."""
        return None
    
    # === Context and Synthesis Hooks ===
    async def on_check_synthesis(self, session_state: Dict[str, Any], context_data: List[Any]) -> Optional[SynthesisCheckResult]:
        """Determine if synthesis is complete."""
        return None
    
    async def on_query_rewrite(self, original_query: str, context_data: List[Any]) -> Optional[str]:
        """Refine queries based on accumulated context."""
        return None
    
    async def on_context_update(self, current_context: List[Any], new_items: List[Any]) -> Optional[List[Any]]:
        """Manage context accumulation."""
        return None
    
    # === Loop Detection ===
    async def on_loop_detection(self, tool_history: List[Dict[str, Any]], current_tool: str) -> bool:
        """Detect and prevent loops."""
        return False
```

### Callback Configuration

#### `RunnerConfig`

Enhanced configuration for callback-enabled agent execution.

**Fields:**
- `agent: Agent` - JAF agent to execute
- `session_provider: Optional[Any]` - Session provider for persistence
- `callbacks: Optional[RunnerCallbacks]` - Callback implementation
- `max_llm_calls: int` - Maximum LLM calls per execution (default: 10)
- `enable_context_accumulation: bool` - Enable context management (default: False)
- `enable_loop_detection: bool` - Enable loop prevention (default: False)
- `max_context_items: int` - Maximum context items to retain (default: 50)
- `max_repeated_tools: int` - Maximum repeated tool calls before loop detection (default: 3)

**Example:**
```python
from adk.runners import RunnerConfig, execute_agent

config = RunnerConfig(
    agent=my_agent,
    session_provider=session_provider,
    callbacks=MyCallbacks(),
    max_llm_calls=15,
    enable_context_accumulation=True,
    enable_loop_detection=True,
    max_context_items=100
)

result = await execute_agent(config, session_state, message, context, model_provider)
```

### Callback Return Types

#### `LLMControlResult`

TypedDict for controlling LLM interactions.

**Fields:**
- `skip: Optional[bool]` - Skip LLM call if True
- `message: Optional[Message]` - Modified message for LLM
- `response: Optional[Message]` - Direct response (when skipping)

#### `ToolSelectionControlResult`

TypedDict for controlling tool selection.

**Fields:**
- `tools: Optional[List[Tool]]` - Filtered tool list
- `custom_selection: Optional[Dict[str, Any]]` - Custom tool selection logic

#### `IterationControlResult`

TypedDict for controlling iteration flow.

**Fields:**
- `continue_iteration: Optional[bool]` - Whether to continue current iteration
- `should_stop: Optional[bool]` - Whether to stop execution
- `should_continue: Optional[bool]` - Whether to continue to next iteration

#### `SynthesisCheckResult`

TypedDict for synthesis completion results.

**Fields:**
- `complete: bool` - Whether synthesis is complete
- `answer: Optional[str]` - Final synthesized answer
- `confidence: Optional[float]` - Confidence score (0.0-1.0)

### Advanced Agent Execution

#### `execute_agent(config: RunnerConfig, session_state: Dict[str, Any], message: Message, context: RunContext, model_provider: ModelProvider) -> AgentResponse`

Execute an agent with full callback instrumentation.

**Parameters:**
- `config: RunnerConfig` - Callback-enabled configuration
- `session_state: Dict[str, Any]` - Mutable session state
- `message: Message` - Input message to process
- `context: RunContext` - Execution context
- `model_provider: ModelProvider` - LLM provider

**Returns:**
- `AgentResponse` - Enhanced response with execution metadata

**Example:**
```python
import asyncio
from adk.runners import RunnerConfig, execute_agent
from jaf.core.types import Agent, Message

# Create callback implementation
class ReActCallbacks:
    def __init__(self):
        self.iteration_count = 0
        self.context_accumulator = []
    
    async def on_iteration_start(self, iteration):
        self.iteration_count = iteration
        print(f"🔄 Iteration {iteration}")
        return None
    
    async def on_check_synthesis(self, session_state, context_data):
        if len(context_data) >= 3:
            return {
                'complete': True,
                'answer': self.synthesize_information(context_data),
                'confidence': 0.85
            }
        return None
    
    async def on_query_rewrite(self, original_query, context_data):
        gaps = self.identify_gaps(context_data)
        if gaps:
            return f"{original_query} focusing on {', '.join(gaps)}"
        return None

# Configure and execute
config = RunnerConfig(
    agent=research_agent,
    callbacks=ReActCallbacks(),
    enable_context_accumulation=True,
    max_llm_calls=10
)

result = await execute_agent(
    config, 
    session_state={}, 
    message=Message(role='user', content='Research machine learning applications'),
    context={'user_id': 'researcher_123'},
    model_provider=litellm_provider
)

print(f"Result: {result.content}")
print(f"Iterations: {result.metadata.get('iterations', 0)}")
print(f"Synthesis confidence: {result.metadata.get('synthesis_confidence', 0)}")
```

### Common Callback Patterns

#### ReAct (Reasoning + Acting) Pattern

```python
class ReActAgent:
    async def on_iteration_start(self, iteration):
        thought = f"Iteration {iteration}: I need to gather more information"
        print(f"🤔 Thought: {thought}")
        return None
    
    async def on_before_tool_execution(self, tool, params):
        action = f"Using {tool.schema.name} with {params}"
        print(f" Action: {action}")
        return None
    
    async def on_after_tool_execution(self, tool, result, error=None):
        if error:
            observation = f"Action failed: {error}"
        else:
            observation = f"Observed: {result}"
        print(f"👁️ Observation: {observation}")
        return None
```

#### Intelligent Caching Pattern

```python
class CachingCallbacks:
    def __init__(self):
        self.cache = {}
    
    async def on_before_llm_call(self, agent, message, session_state):
        cache_key = hash(message.content)
        if cache_key in self.cache:
            return {'skip': True, 'response': self.cache[cache_key]}
        return None
    
    async def on_after_llm_call(self, response, session_state):
        cache_key = hash(response.content)
        self.cache[cache_key] = response
        return None
```

#### Context Accumulation Pattern

```python
class ContextAccumulator:
    def __init__(self):
        self.context_items = []
    
    async def on_context_update(self, current_context, new_items):
        # Deduplicate and filter
        filtered_items = self.filter_duplicates(new_items)
        
        # Merge and sort by relevance
        merged = current_context + filtered_items
        sorted_context = sorted(merged, key=lambda x: x.get('relevance', 0), reverse=True)
        
        # Keep top items
        return sorted_context[:50]
    
    async def on_check_synthesis(self, session_state, context_data):
        if len(context_data) >= 5:
            confidence = self.calculate_confidence(context_data)
            if confidence >= 0.8:
                return {
                    'complete': True,
                    'answer': self.synthesize(context_data),
                    'confidence': confidence
                }
        return None
```

#### Loop Detection Pattern

```python
class LoopDetector:
    def __init__(self, similarity_threshold=0.7):
        self.threshold = similarity_threshold
        self.tool_history = []
    
    async def on_loop_detection(self, tool_history, current_tool):
        if len(tool_history) < 3:
            return False
        
        # Check for repeated tool calls
        recent_tools = [item['tool'] for item in tool_history[-3:]]
        if recent_tools.count(current_tool) > 2:
            return True
        
        # Check parameter similarity
        for item in tool_history[-3:]:
            similarity = self.calculate_similarity(item.get('params', {}), current_tool)
            if similarity > self.threshold:
                return True
        
        return False
```

## Complete Example

Here's a complete example showing how to use the main APIs together with advanced callback functionality:

```python
import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import jaf
from adk.runners import RunnerConfig, execute_agent

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
            from adk.utils.safe_evaluator import safe_calculate
            result = safe_calculate(args.expression)
            if result["status"] == "success":
                return jaf.ToolResponse.success(
                    f"Result: {args.expression} = {result['result']}"
                ).format()
            else:
                return jaf.ToolResponse.error(
                    'calculation_error', 
                    result['error']
                ).format()
        except Exception as e:
            return jaf.ToolResponse.error(
                'calculation_error', 
                str(e)
            ).format()

# Advanced callback implementation for production use
class ProductionMathCallbacks:
    """Production-ready callbacks with caching and monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.calculations_cache = {}
        self.performance_metrics = {
            'llm_calls': 0,
            'tool_calls': 0,
            'cache_hits': 0
        }
    
    async def on_start(self, context, message, session_state):
        """Initialize execution with user context."""
        self.start_time = time.time()
        user_id = context.get('user_id', 'unknown')
        print(f"🧮 Math Assistant started for user: {user_id}")
        print(f" Query: {message.content}")
    
    async def on_before_llm_call(self, agent, message, session_state):
        """Implement intelligent caching and context enhancement."""
        self.performance_metrics['llm_calls'] += 1
        
        # Check for cached mathematical explanations
        cache_key = hash(f"math:{message.content}")
        if cache_key in self.calculations_cache:
            self.performance_metrics['cache_hits'] += 1
            print(f"💾 Using cached explanation")
            return {
                'skip': True, 
                'response': self.calculations_cache[cache_key]
            }
        
        # Enhance message with mathematical context
        enhanced_content = f"""Mathematical Problem: {message.content}
        
Please provide step-by-step explanations and use the calculator tool for all arithmetic operations.
        """
        
        return {
            'message': jaf.Message(role='user', content=enhanced_content)
        }
    
    async def on_after_llm_call(self, response, session_state):
        """Cache educational responses."""
        if 'step' in response.content.lower() or 'calculate' in response.content.lower():
            cache_key = hash(f"explanation:{response.content[:100]}")
            self.calculations_cache[cache_key] = response
        return None
    
    async def on_tool_selected(self, tool_name, params):
        """Track tool usage and validate calculations."""
        self.performance_metrics['tool_calls'] += 1
        if tool_name == 'calculate':
            expression = params.get('expression', '')
            print(f"🔢 Calculating: {expression}")
    
    async def on_after_tool_execution(self, tool, result, error=None):
        """Validate and enhance calculation results."""
        if error:
            print(f" Calculation error: {error}")
            return None
        
        if tool.schema.name == 'calculate' and 'Result:' in str(result):
            # Extract and validate the calculation
            print(f" Calculation completed: {result}")
        
        return None
    
    async def on_complete(self, response):
        """Log comprehensive execution metrics."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n Execution Summary:")
        print(f"   Duration: {duration*1000:.0f}ms")
        print(f"   LLM Calls: {self.performance_metrics['llm_calls']}")
        print(f"   Tool Calls: {self.performance_metrics['tool_calls']}")
        print(f"   Cache Hits: {self.performance_metrics['cache_hits']}")
        print(f"   Cache Size: {len(self.calculations_cache)} items")
    
    async def on_error(self, error, context):
        """Handle mathematical errors gracefully."""
        print(f" Math Assistant Error: {str(error)}")
        # In production, log to monitoring system

def create_math_agent():
    def instructions(state: jaf.RunState[UserContext]) -> str:
        return f"""You are an advanced math tutor for user {state.context.user_id}.
        
Your capabilities:
- Perform calculations using the calculate tool
- Provide step-by-step explanations
- Show alternative solving methods
- Explain mathematical concepts clearly

Always:
1. Break down complex problems into steps
2. Use the calculator tool for all arithmetic
3. Explain your reasoning
4. Verify your answers"""
    
    return jaf.Agent(
        name='AdvancedMathAssistant',
        instructions=instructions,
        tools=[CalculatorTool()]
    )

async def demonstrate_traditional_jaf():
    """Demonstrate traditional JAF Core approach."""
    print("=== Traditional JAF Core Approach ===")
    
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
        agent_registry={'AdvancedMathAssistant': math_agent},
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
        current_agent_name='AdvancedMathAssistant',
        context=UserContext(user_id='user_123', permissions=['calculator']),
        turn_count=0
    )
    
    # Run the agent
    result = await jaf.run(initial_state, config)
    
    # Handle result
    if result.outcome.status == 'completed':
        print(f" JAF Core Result: {result.outcome.output}")
    else:
        print(f" JAF Core Error: {result.outcome.error}")

async def demonstrate_callback_approach():
    """Demonstrate ADK Callback approach with advanced features."""
    print("\n=== ADK Callback Approach with Advanced Features ===")
    
    # Create model provider
    model_provider = jaf.make_litellm_provider('http://localhost:4000')
    
    # Create agent
    math_agent = create_math_agent()
    
    # Set up callback configuration
    callback_config = RunnerConfig(
        agent=math_agent,
        callbacks=ProductionMathCallbacks(),
        max_llm_calls=8,
        enable_context_accumulation=True,
        enable_loop_detection=True
    )
    
    # Execute with full instrumentation
    result = await execute_agent(
        callback_config,
        session_state={'learning_level': 'intermediate'},
        message=jaf.Message(role='user', content='Solve step by step: (25 + 17) * 3 - 15'),
        context=UserContext(user_id='callback_user', permissions=['calculator']),
        model_provider=model_provider
    )
    
    print(f" Callback Result: {result.content}")
    print(f" Metadata: {result.metadata}")

async def main():
    """Complete demonstration of JAF APIs with both approaches."""
    print("🧮 JAF Python Framework - Complete API Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate traditional JAF approach
        await demonstrate_traditional_jaf()
        
        # Demonstrate advanced callback approach
        await demonstrate_callback_approach()
        
        print("\n Both approaches completed successfully!")
        print("\nKey Differences:")
        print("• JAF Core: Functional, immutable, production-ready")
        print("• ADK Callbacks: Enhanced with instrumentation, caching, monitoring")
        print("• Both: Type-safe, composable, enterprise-grade")
        
    except Exception as e:
        print(f" Demo Error: {e}")
        # In production, comprehensive error handling would be here

if __name__ == "__main__":
    asyncio.run(main())
```

This API reference provides comprehensive documentation for building sophisticated AI agent systems with JAF's functional architecture, type safety, and production-ready features.