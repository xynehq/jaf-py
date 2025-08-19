# Core Concepts

JAF (Juspay Agent Framework) is built on functional programming principles, emphasizing immutability, composability, and type safety. This guide explains the fundamental concepts that make JAF powerful and predictable.

## Philosophy

### Functional at the Core

JAF treats agent execution as a pure function: given an initial state and configuration, it produces a deterministic result. This approach brings several benefits:

- **Predictability**: Same inputs always produce the same outputs
- **Testability**: Easy to test individual components in isolation
- **Debuggability**: State transitions are explicit and traceable
- **Scalability**: Stateless design enables horizontal scaling

### Immutability First

All core data structures in JAF are immutable. When state changes, new objects are created rather than modifying existing ones:

```python
# Mutable approach (not JAF)
state.messages.append(new_message)  # Modifies existing state

# Immutable approach (JAF way)
new_state = replace(state, messages=[*state.messages, new_message])
```

This ensures:
- **Thread Safety**: Multiple agents can safely share state
- **Time Travel**: Previous states remain accessible for debugging
- **Reproducibility**: Exact state at any point can be recreated

## Core Types

### 1. RunState - The Heart of JAF

`RunState` represents the complete state of an agent execution at any point in time:

```python
@dataclass(frozen=True)
class RunState(Generic[Ctx]):
    """Immutable state of an agent run."""
    run_id: RunId                    # Unique identifier for this run
    trace_id: TraceId               # Trace identifier for observability
    messages: List[Message]         # Conversation history
    current_agent_name: str         # Currently active agent
    context: Ctx                    # User-defined context data
    turn_count: int                 # Number of turns taken
    final_response: Optional[str] = None    # Final agent response
```

**Key Properties:**
- **Frozen**: Cannot be modified after creation
- **Generic**: Type-safe context with `Ctx` type parameter  
- **Complete**: Contains all information needed to reproduce the run

**State Transitions:**
```python
# Every operation creates a new state
from dataclasses import replace

async def add_message(state: RunState[Ctx], message: Message) -> RunState[Ctx]:
    return replace(state, 
        messages=[*state.messages, message],
        turn_count=state.turn_count + 1
    )
```

### 2. Agent - Behavior Definition

Agents define how to respond to messages and what tools are available:

```python
@dataclass(frozen=True)
class Agent(Generic[Ctx]):
    """Agent definition with instructions and capabilities."""
    name: str
    instructions: Callable[[RunState[Ctx]], str]  # Dynamic instructions
    tools: List[Tool[Ctx]] = field(default_factory=list)
    handoffs: Optional[List[str]] = None         # Allowed handoff targets
    output_codec: Optional[type] = None         # Expected output codec
```

**Dynamic Instructions:**
Instructions are functions that receive the current state, enabling context-aware behavior:

```python
def math_tutor_instructions(state: RunState[StudentContext]) -> str:
    problem_count = len([m for m in state.messages if 'calculate' in m.content])
    
    base = "You are a patient math tutor."
    
    if problem_count > 3:
        return base + " The student has solved several problems. Offer encouragement!"
    elif state.context.difficulty_level == "beginner":
        return base + " Use simple explanations and encourage step-by-step thinking."
    else:
        return base + " Challenge the student with follow-up questions."
```

### 3. Tool - Executable Capabilities

Tools encapsulate external capabilities that agents can use:

```python
from jaf import function_tool

@function_tool
async def get_weather(location: str, units: str = "metric", context=None) -> str:
    """Get current weather for a location.
    
    Args:
        location: The location to get weather for
        units: Temperature units (metric/imperial)
    """
    # Implementation here
    weather_data = await fetch_weather_api(location, units)
    return f"Weather in {location}: {weather_data['temperature']}°"
```

**Tool Properties:**
- **Schema-Driven**: Pydantic models define arguments
- **Context-Aware**: Access to run context for authorization/customization
- **Async**: Built for modern Python async/await patterns
- **Type-Safe**: Full typing support with generics

### 4. RunConfig - Execution Parameters

Configuration object that controls how agents execute:

```python
@dataclass
class RunConfig(Generic[Ctx]):
    """Configuration for agent execution."""
    agent_registry: Dict[str, Agent[Ctx]]        # Available agents
    model_provider: ModelProvider                # LLM integration
    memory_provider: Optional[MemoryProvider] = None  # Conversation storage
    max_turns: int = 100                        # Safety limit
    on_event: Optional[Callable[[TraceEvent], None]] = None  # Observability
    initial_input_guardrails: List[Guardrail] = field(default_factory=list)
    final_output_guardrails: List[Guardrail] = field(default_factory=list)
```

## The Execution Flow

### Pure Function at the Core

The main `run` function is a pure function that transforms state:

```python
async def run(
    initial_state: RunState[Ctx], 
    config: RunConfig[Ctx]
) -> RunResult[Out]:
    """
    Pure function: RunState + RunConfig → RunResult
    
    No side effects in core logic - all effects happen in providers.
    """
```

### Step-by-Step Execution

1. **Initialization**: Validate state and configuration
2. **Guard Rails**: Apply input validation policies
3. **Agent Selection**: Get current agent from registry  
4. **Instruction Generation**: Call agent's instruction function with current state
5. **LLM Call**: Send messages and instructions to model provider
6. **Response Processing**: Parse LLM response for tool calls or final answer
7. **Tool Execution**: If tool calls present, execute them with context
8. **State Update**: Create new state with response and tool results
9. **Loop Check**: If not complete and under turn limit, continue
10. **Final Guards**: Apply output validation policies
11. **Memory Storage**: Persist conversation if memory provider configured

### Error Handling

JAF uses a Result-style approach for error handling:

```python
@dataclass(frozen=True)
class RunResult(Generic[Out]):
    """Result of an agent run."""
    final_state: RunState
    outcome: Union[CompletedOutcome[Out], ErrorOutcome]

# Usage
result = await run(state, config)
if result.outcome.status == 'completed':
    print(f"Success: {result.outcome.output}")
else:
    print(f"Error: {result.outcome.error}")
```

## Type Safety

### Generic Context

JAF uses Python generics to maintain type safety across the entire execution:

```python
# Define your domain types
@dataclass
class ECommerceContext:
    user_id: str
    cart_items: List[str]
    is_premium: bool

# Agents are typed to your context
shopping_agent: Agent[ECommerceContext] = Agent(
    name="ShoppingAssistant",
    instructions=lambda state: f"Help user {state.context.user_id} with shopping",
    tools=[add_to_cart_tool, checkout_tool]
)

# State maintains type safety
state: RunState[ECommerceContext] = RunState(
    # ... other fields
    context=ECommerceContext(user_id="user123", cart_items=[], is_premium=True)
)

# Tool implementations are context-aware
async def execute(self, args: AddToCartArgs, context: ECommerceContext) -> str:
    # context.is_premium is properly typed as bool
    discount = 0.1 if context.is_premium else 0.0
```

### Runtime Validation

While maintaining compile-time type safety, JAF also provides runtime validation with Pydantic:

```python
class CreateOrderArgs(BaseModel):
    """Validated arguments for order creation."""
    items: List[str] = Field(min_items=1, description="Items to order")
    shipping_address: str = Field(min_length=10, description="Delivery address")
    priority: Literal["standard", "express"] = Field(default="standard")

# Automatic validation when LLM calls the tool
# Invalid calls result in clear error messages
```

## Composition Patterns

### Tool Composition

Tools can be composed to create more complex behaviors:

```python
from jaf import function_tool

@function_tool
async def read_file(filepath: str, context=None) -> str:
    """Read contents of a file."""
    with open(filepath, 'r') as f:
        return f.read()

@function_tool
async def write_file(filepath: str, content: str, context=None) -> str:
    """Write content to a file."""
    with open(filepath, 'w') as f:
        f.write(content)
    return f"File written: {filepath}"

@function_tool
async def list_directory(path: str = ".", context=None) -> str:
    """List files in a directory."""
    import os
    files = os.listdir(path)
    return f"Files in {path}: {', '.join(files)}"

@function_tool
async def search_files(pattern: str, directory: str = ".", context=None) -> str:
    """Search for files matching a pattern."""
    import os
    import fnmatch
    matches = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            matches.append(os.path.join(root, filename))
    return f"Found files: {', '.join(matches)}"

def create_file_manager_agent() -> Agent[FileContext]:
    return Agent(
        name="FileManager",
        instructions=file_manager_instructions,
        tools=[read_file, write_file, list_directory, search_files]
    )
```

### Agent Handoffs

Agents can transfer control to other specialized agents:

```python
from jaf import function_tool

@function_tool
async def route_customer(query: str, context=None) -> str:
    """Route customer to appropriate specialist based on query analysis."""
    if "technical" in query.lower() or "bug" in query.lower():
        return handoff_to_agent("TechnicalSupport", context=context)
    elif "billing" in query.lower() or "payment" in query.lower():
        return handoff_to_agent("Billing", context=context)
    elif "sales" in query.lower() or "purchase" in query.lower():
        return handoff_to_agent("Sales", context=context)
    else:
        return "I'll help you with your general inquiry."

def create_triage_agent() -> Agent[CustomerContext]:
    return Agent(
        name="TriageAgent",
        instructions=lambda state: "Route customers to appropriate specialists",
        tools=[route_customer],  # Modern handoff capability
        handoffs=["TechnicalSupport", "Billing", "Sales"]  # Allowed targets
    )
```

### Validation Composition

Multiple validation policies can be composed:

```python
from jaf.policies.validation import compose_validations

# Individual validators
content_filter = create_content_filter(['spam', 'inappropriate'])
length_guardrail = create_length_guardrail(max_length=1000, min_length=1)
permission_check = create_permission_validator("file_access", lambda ctx: ctx.permissions)

# Compose them
combined_validator = compose_validations(
    content_filter,
    length_guardrail, 
    permission_check
)

config = RunConfig(
    # ...
    initial_input_guardrails=[combined_validator]
)
```

## Memory and Persistence

JAF separates the pure execution logic from persistence concerns using the Provider pattern:

```python
# Core execution remains pure
result = await run(initial_state, config)

# Memory provider handles persistence as a side effect
if config.memory_provider:
    await config.memory_provider.store_conversation(
        conversation_id="user_123_session", 
        messages=result.final_state.messages
    )
```

This separation enables:
- **Testing**: Easy to test without databases
- **Flexibility**: Swap memory providers without changing core logic
- **Scalability**: Different storage strategies for different needs

## Observability

JAF provides comprehensive observability through event tracing:

```python
def trace_handler(event: TraceEvent) -> None:
    """Handle trace events for monitoring."""
    if event.type == "llm_call_start":
        print(f"LLM call: {event.data['model']}")
    elif event.type == "tool_call_start":
        print(f"Tool call: {event.data['tool_name']}")
    elif event.type == "error":
        print(f"Error: {event.data['error_type']}")

config = RunConfig(
    # ...
    on_event=trace_handler
)
```

Events provide insights into:
- Agent execution flow
- Tool usage patterns  
- Performance metrics
- Error conditions
- State transitions

## Best Practices

### 1. Keep Instructions Pure

Instructions should be pure functions of state:

```python
# Good: Pure function
def instructions(state: RunState[Ctx]) -> str:
    return f"Help user with {len(state.messages)} previous messages"

# Avoid: Side effects or external dependencies
def instructions(state: RunState[Ctx]) -> str:
    current_time = datetime.now()  # External dependency
    log.info("Generating instructions")  # Side effect
    return f"Current time is {current_time}"
```

### 2. Design Immutable Context

Context should contain all domain data needed for the conversation:

```python
@dataclass(frozen=True)  # Frozen ensures immutability
class OrderContext:
    customer_id: str
    order_items: Tuple[str, ...]  # Immutable collection
    shipping_preference: str
    
    # Methods can compute derived data
    @property
    def total_items(self) -> int:
        return len(self.order_items)
```

### 3. Handle Errors Gracefully

Use JAF's error types for clear error handling:

```python
async def execute(self, args: OrderArgs, context: OrderContext) -> str:
    try:
        result = await external_api.create_order(args.items)
        return ToolSuccess(f"Order created: {result.order_id}").format()
    except APIError as e:
        return ToolError(f"Failed to create order: {e}").format()
    except ValidationError as e:
        return ToolError(f"Invalid order data: {e}").format()
```

### 4. Leverage Type Safety

Use generics and type hints throughout:

```python
# Type-safe agent factory
def create_agent[T](
    name: str,
    instructions: Callable[[RunState[T]], str],
    tools: List[Tool[T]]
) -> Agent[T]:
    return Agent(name=name, instructions=instructions, tools=tools)

# Usage maintains type safety
math_agent: Agent[StudentContext] = create_agent(
    "MathTutor",
    math_instructions,
    [calculator, graph_plotter]
)
```

This functional approach makes JAF agents predictable, testable, and maintainable while providing the flexibility to build complex AI systems.

## Next Steps

- **[API Reference](api-reference.md)** - Detailed API documentation
- **[Tools Guide](tools.md)** - Building custom tools
- **[Memory System](memory-system.md)** - Adding persistence
- **[Examples](examples.md)** - See these concepts in action
