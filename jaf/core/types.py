"""
Core types for the JAF framework.

This module defines all the fundamental data structures and types used throughout
the framework, maintaining immutability and type safety.
"""

from collections.abc import Awaitable

# ReadOnly is only available in Python 3.13+, so we'll use a simpler approach
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    TypedDict,
    Union,
    runtime_checkable,
    TYPE_CHECKING,
)
from enum import Enum

if TYPE_CHECKING:
    from .tool_results import ToolResult


# Comprehensive enums for type safety and improved developer experience
class Model(str, Enum):
    """Supported model identifiers."""
    GEMINI_2_0_FLASH = 'gemini-2.0-flash'
    GEMINI_2_5_PRO = 'gemini-2.5-pro'
    GEMINI_PRO = 'gemini-pro'
    GPT_4 = 'gpt-4'
    GPT_4_TURBO = 'gpt-4-turbo'
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    CLAUDE_3_SONNET = 'claude-3-sonnet'
    CLAUDE_3_HAIKU = 'claude-3-haiku'
    CLAUDE_3_OPUS = 'claude-3-opus'


class ToolParameterType(str, Enum):
    """Tool parameter types."""
    STRING = 'string'
    NUMBER = 'number'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    ARRAY = 'array'
    OBJECT = 'object'
    NULL = 'null'


class ToolSource(str, Enum):
    """Source of tool definitions."""
    NATIVE = 'native'
    MCP = 'mcp'
    PLUGIN = 'plugin'
    EXTERNAL = 'external'


class ContentRole(str, Enum):
    """Message content roles."""
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL = 'tool'
    SYSTEM = 'system'


class PartType(str, Enum):
    """Message part types."""
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio'
    VIDEO = 'video'
    FILE = 'file'


# Branded types for type safety - using class-based approach for better type safety
class TraceId(str):
    """Branded string type for trace IDs."""
    def __new__(cls, value: str) -> 'TraceId':
        return str.__new__(cls, value)

class RunId(str):
    """Branded string type for run IDs."""
    def __new__(cls, value: str) -> 'RunId':
        return str.__new__(cls, value)

def create_trace_id(id_str: str) -> TraceId:
    """Create a TraceId from a string."""
    return TraceId(id_str)

def create_run_id(id_str: str) -> RunId:
    """Create a RunId from a string."""
    return RunId(id_str)

def generate_run_id() -> RunId:
    """Generate a new unique run ID."""
    import time
    import uuid
    return RunId(f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}")

def generate_trace_id() -> TraceId:
    """Generate a new unique trace ID."""
    import time
    import uuid
    return TraceId(f"trace_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}")

# Type variables for generic contexts and outputs
Ctx = TypeVar('Ctx')
Out = TypeVar('Out')
Args = TypeVar('Args')

# Discriminated union for ValidationResult to match TypeScript version
@dataclass(frozen=True)
class ValidValidationResult:
    """Valid validation result."""
    is_valid: Literal[True] = True

@dataclass(frozen=True)
class InvalidValidationResult:
    """Invalid validation result with error message."""
    is_valid: Literal[False] = False
    error_message: str = ""

ValidationResult = Union[ValidValidationResult, InvalidValidationResult]

@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    type: Literal['function']
    function: 'ToolCallFunction'

@dataclass(frozen=True)
class ToolCallFunction:
    """Function information within a tool call."""
    name: str
    arguments: str

@dataclass(frozen=True)
class Message:
    """A message in the conversation."""
    role: ContentRole
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model behavior."""
    name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

@dataclass(frozen=True)
class ToolSchema(Generic[Args]):
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Any  # Should be a type that can validate Args (like Pydantic model or Zod equivalent)
    timeout: Optional[float] = None  # Optional timeout in seconds for tool execution

@runtime_checkable
class Tool(Protocol[Args, Ctx]):
    """Protocol for tool implementations."""

    @property
    def schema(self) -> ToolSchema[Args]:
        """Tool schema including name, description, and parameter validation."""
        ...

    async def execute(self, args: Args, context: Ctx) -> Union[str, 'ToolResult[Any]']:
        """Execute the tool with given arguments and context."""
        ...


# Function tool configuration for improved DX
class FunctionToolConfig(TypedDict):
    """Configuration for creating function-based tools with object-based API."""
    name: str
    description: str
    execute: Callable[[Any, Any], Union[str, 'ToolResult[Any]', Awaitable[Union[str, 'ToolResult[Any]']]]]
    parameters: Any  # Pydantic model or similar for parameter validation
    metadata: Optional[Dict[str, Any]]  # Optional metadata
    source: Optional[ToolSource]  # Optional source tracking
    timeout: Optional[float]  # Optional timeout in seconds for tool execution


# Type alias for tool execution functions
ToolExecuteFunction = Callable[[Any, Any], Union[str, 'ToolResult[Any]', Awaitable[Union[str, 'ToolResult[Any]']]]]


@dataclass(frozen=True)
class Agent(Generic[Ctx, Out]):
    """An agent definition with instructions, tools, and configuration."""
    name: str
    instructions: Callable[['RunState[Ctx]'], str]
    tools: Optional[List[Tool[Any, Ctx]]] = None
    output_codec: Optional[Any] = None  # Type that can validate Out (like Pydantic model or Zod equivalent)
    handoffs: Optional[List[str]] = None
    model_config: Optional[ModelConfig] = None

# Guardrail type
Guardrail = Callable[[Any], Union[ValidationResult, Awaitable[ValidationResult]]]

@dataclass(frozen=True)
class RunState(Generic[Ctx]):
    """Immutable state of a run."""
    run_id: RunId
    trace_id: TraceId
    messages: List[Message]
    current_agent_name: str
    context: Ctx
    turn_count: int

# Error types using dataclasses for immutability
@dataclass(frozen=True)
class MaxTurnsExceeded:
    _tag: Literal["MaxTurnsExceeded"] = "MaxTurnsExceeded"
    turns: int = 0

@dataclass(frozen=True)
class ModelBehaviorError:
    _tag: Literal["ModelBehaviorError"] = "ModelBehaviorError"
    detail: str = ""

@dataclass(frozen=True)
class DecodeError:
    _tag: Literal["DecodeError"] = "DecodeError"
    errors: List[Dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class InputGuardrailTripwire:
    _tag: Literal["InputGuardrailTripwire"] = "InputGuardrailTripwire"
    reason: str = ""

@dataclass(frozen=True)
class OutputGuardrailTripwire:
    _tag: Literal["OutputGuardrailTripwire"] = "OutputGuardrailTripwire"
    reason: str = ""

@dataclass(frozen=True)
class ToolCallError:
    _tag: Literal["ToolCallError"] = "ToolCallError"
    tool: str = ""
    detail: str = ""

@dataclass(frozen=True)
class HandoffError:
    _tag: Literal["HandoffError"] = "HandoffError"
    detail: str = ""

@dataclass(frozen=True)
class AgentNotFound:
    _tag: Literal["AgentNotFound"] = "AgentNotFound"
    agent_name: str = ""

@dataclass(frozen=True)
class RecoverableError:
    """Error that can be recovered from with retry strategies."""
    _tag: Literal["RecoverableError"] = "RecoverableError"
    detail: str = ""
    retry_strategy: Optional[str] = None
    max_retries: int = 3
    current_attempt: int = 1
    backoff_seconds: float = 1.0

@dataclass(frozen=True)
class RateLimitError:
    """Error indicating rate limit has been exceeded."""
    _tag: Literal["RateLimitError"] = "RateLimitError"
    detail: str = ""
    retry_after_seconds: Optional[float] = None
    limit_type: str = "requests"  # "requests", "tokens", "concurrent"

@dataclass(frozen=True)
class ValidationError:
    """Enhanced validation error with detailed context."""
    _tag: Literal["ValidationError"] = "ValidationError"
    detail: str = ""
    field_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_code: str = "validation_failed"

@dataclass(frozen=True)
class NetworkError:
    """Network-related errors with retry information."""
    _tag: Literal["NetworkError"] = "NetworkError"
    detail: str = ""
    status_code: Optional[int] = None
    is_retryable: bool = True
    endpoint: Optional[str] = None

# Union type for all possible errors
JAFError = Union[
    MaxTurnsExceeded,
    ModelBehaviorError,
    DecodeError,
    InputGuardrailTripwire,
    OutputGuardrailTripwire,
    ToolCallError,
    HandoffError,
    AgentNotFound,
    RecoverableError,
    RateLimitError,
    ValidationError,
    NetworkError
]

@dataclass(frozen=True)
class CompletedOutcome(Generic[Out]):
    """Successful completion outcome."""
    status: Literal['completed'] = 'completed'
    output: Out = field(default=None)

@dataclass(frozen=True)
class ErrorOutcome:
    """Error outcome."""
    status: Literal['error'] = 'error'
    error: JAFError = field(default=None)

# Union type for outcomes
RunOutcome = Union[CompletedOutcome[Out], ErrorOutcome]

@dataclass(frozen=True)
class RunResult(Generic[Out]):
    """Result of a run execution."""
    final_state: RunState[Any]
    outcome: RunOutcome[Out]

# Trace event types with specific data structures to match TypeScript
@dataclass(frozen=True)
class RunStartEventData:
    """Data for run start events."""
    run_id: RunId
    trace_id: TraceId

@dataclass(frozen=True)
class RunStartEvent:
    type: Literal['run_start'] = 'run_start'
    data: RunStartEventData = field(default_factory=lambda: RunStartEventData(RunId(""), TraceId("")))

@dataclass(frozen=True)
class LLMCallStartEventData:
    """Data for LLM call start events."""
    agent_name: str
    model: str

@dataclass(frozen=True)
class LLMCallStartEvent:
    type: Literal['llm_call_start'] = 'llm_call_start'
    data: LLMCallStartEventData = field(default_factory=lambda: LLMCallStartEventData("", ""))

@dataclass(frozen=True)
class LLMCallEndEventData:
    """Data for LLM call end events."""
    choice: Any

@dataclass(frozen=True)
class LLMCallEndEvent:
    type: Literal['llm_call_end'] = 'llm_call_end'
    data: LLMCallEndEventData = field(default_factory=lambda: LLMCallEndEventData(None))

@dataclass(frozen=True)
class ToolCallStartEventData:
    """Data for tool call start events."""
    tool_name: str
    args: Any

@dataclass(frozen=True)
class ToolCallStartEvent:
    type: Literal['tool_call_start'] = 'tool_call_start'
    data: ToolCallStartEventData = field(default_factory=lambda: ToolCallStartEventData("", None))

@dataclass(frozen=True)
class ToolCallEndEventData:
    """Data for tool call end events."""
    tool_name: str
    result: str
    tool_result: Optional[Any] = None
    status: Optional[str] = None

@dataclass(frozen=True)
class ToolCallEndEvent:
    type: Literal['tool_call_end'] = 'tool_call_end'
    data: ToolCallEndEventData = field(default_factory=lambda: ToolCallEndEventData("", ""))

@dataclass(frozen=True)
class HandoffEventData:
    """Data for handoff events."""
    from_: str = field(metadata={'alias': 'from'})  # Using from_ since 'from' is a Python keyword
    to: str

@dataclass(frozen=True)
class HandoffEvent:
    type: Literal['handoff'] = 'handoff'
    data: HandoffEventData = field(default_factory=lambda: HandoffEventData("", ""))

@dataclass(frozen=True)
class RunEndEventData:
    """Data for run end events."""
    outcome: 'RunOutcome[Any]'

@dataclass(frozen=True)
class RunEndEvent:
    type: Literal['run_end'] = 'run_end'
    data: RunEndEventData = field(default_factory=lambda: RunEndEventData(None))

# Union type for all trace events
TraceEvent = Union[
    RunStartEvent,
    LLMCallStartEvent,
    LLMCallEndEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    HandoffEvent,
    RunEndEvent
]

@dataclass(frozen=True)
class ModelCompletionMessage:
    """Message structure returned by model completion."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

@dataclass(frozen=True)
class ModelCompletionResponse:
    """Response structure from model completion."""
    message: Optional[ModelCompletionMessage] = None

@runtime_checkable
class ModelProvider(Protocol[Ctx]):
    """Protocol for model providers."""

    async def get_completion(
        self,
        state: RunState[Ctx],
        agent: Agent[Ctx, Any],
        config: 'RunConfig[Ctx]'
    ) -> ModelCompletionResponse:
        """Get completion from the model."""
        ...

@dataclass(frozen=True)
class RunConfig(Generic[Ctx]):
    """Configuration for running agents."""
    agent_registry: Dict[str, Agent[Ctx, Any]]
    model_provider: ModelProvider[Ctx]
    max_turns: Optional[int] = 50
    model_override: Optional[str] = None
    initial_input_guardrails: Optional[List[Guardrail]] = None
    final_output_guardrails: Optional[List[Guardrail]] = None
    on_event: Optional[Callable[[TraceEvent], None]] = None
    memory: Optional['MemoryConfig'] = None
    conversation_id: Optional[str] = None
    default_tool_timeout: Optional[float] = 30.0  # Default timeout for tool execution in seconds
