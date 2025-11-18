"""
Core types for the JAF framework.

This module defines all the fundamental data structures and types used throughout
the framework, maintaining immutability and type safety.
"""

from collections.abc import Awaitable, AsyncIterator

# ReadOnly is only available in Python 3.13+, so we'll use a simpler approach
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
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
    from ..memory.approval_storage import ApprovalStorage
    from ..memory.types import MemoryConfig


# Comprehensive enums for type safety and improved developer experience
class Model(str, Enum):
    """Supported model identifiers."""

    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_PRO = "gemini-pro"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_3_OPUS = "claude-3-opus"


class ToolParameterType(str, Enum):
    """Tool parameter types."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class ToolSource(str, Enum):
    """Source of tool definitions."""

    NATIVE = "native"
    MCP = "mcp"
    PLUGIN = "plugin"
    EXTERNAL = "external"


class ContentRole(str, Enum):
    """Message content roles."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class PartType(str, Enum):
    """Message part types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


# Branded types for type safety - using class-based approach for better type safety
class TraceId(str):
    """Branded string type for trace IDs."""

    def __new__(cls, value: str) -> "TraceId":
        return str.__new__(cls, value)


class RunId(str):
    """Branded string type for run IDs."""

    def __new__(cls, value: str) -> "RunId":
        return str.__new__(cls, value)


class MessageId(str):
    """Branded string type for message IDs."""

    def __new__(cls, value: str) -> "MessageId":
        return str.__new__(cls, value)


def create_trace_id(id_str: str) -> TraceId:
    """Create a TraceId from a string."""
    return TraceId(id_str)


def create_run_id(id_str: str) -> RunId:
    """Create a RunId from a string."""
    return RunId(id_str)


def create_message_id(id_str: Union[str, MessageId]) -> MessageId:
    """
    Create a MessageId from a string or return existing MessageId.

    Args:
        id_str: Either a string to convert to MessageId or an existing MessageId

    Returns:
        MessageId: A validated MessageId instance

    Raises:
        ValueError: If the input is invalid or empty
    """
    # Handle None input
    if id_str is None:
        raise ValueError("Message ID cannot be None")

    # If already a MessageId, return as-is
    if isinstance(id_str, MessageId):
        return id_str

    # Convert string to MessageId with validation
    if isinstance(id_str, str):
        if not id_str.strip():
            raise ValueError("Message ID cannot be empty or whitespace")
        return MessageId(id_str.strip())

    # Handle any other type
    raise ValueError(f"Message ID must be a string or MessageId, got {type(id_str)}")


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


def generate_message_id() -> MessageId:
    """Generate a new unique message ID."""
    import time
    import uuid

    return MessageId(f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}")


# Type variables for generic contexts and outputs
Ctx = TypeVar("Ctx")
Out = TypeVar("Out")
Args = TypeVar("Args")


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
    type: Literal["function"]
    function: "ToolCallFunction"


@dataclass(frozen=True)
class ToolCallFunction:
    """Function information within a tool call."""

    name: str
    arguments: str


@dataclass(frozen=True)
class Attachment:
    """Represents an attachment with various content types."""

    kind: Literal["image", "document", "file"]
    mime_type: Optional[str] = None  # e.g. image/png, application/pdf
    name: Optional[str] = None  # Optional filename
    url: Optional[str] = None  # Remote URL or data URL
    data: Optional[str] = None  # Base64 without data: prefix
    format: Optional[str] = None  # Optional short format like 'pdf', 'txt'
    use_litellm_format: Optional[bool] = None  # Use LiteLLM native file format

    def __post_init__(self):
        """Validate that at least one of url or data is provided."""
        if self.url is None and self.data is None:
            raise ValueError("At least one of 'url' or 'data' must be provided for an Attachment.")


@dataclass(frozen=True)
class MessageContentPart:
    """Part of multi-part message content."""

    type: Literal["text", "image_url", "file"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None  # Contains url and optional detail
    file: Optional[Dict[str, Any]] = None  # Contains file_id and optional format


@dataclass(frozen=True)
class Message:
    """
    A message in the conversation.

    BACKWARDS COMPATIBILITY:
    - Messages created with string content remain fully backwards compatible
    - Direct access to .content returns the original string when created with string
    - Use .text_content property for guaranteed string access in all cases
    - Use get_text_content() function to extract text from any content type
    - message_id is optional for backward compatibility

    Examples:
        # Original usage - still works exactly the same
        msg = Message(role='user', content='Hello')
        text = msg.content  # Returns 'Hello' as string

        # New usage with message ID
        msg = Message(role='user', content='Hello', message_id='msg_123')

        # Guaranteed string access (recommended for new code)
        text = msg.text_content  # Always returns string

        # Universal text extraction
        text = get_text_content(msg.content)  # Works with any content type
    """

    role: ContentRole
    content: Union[str, List[MessageContentPart]]
    attachments: Optional[List[Attachment]] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    message_id: Optional[MessageId] = None  # Optional for backward compatibility

    def __post_init__(self):
        """
        Auto-generate message ID if not provided.

        This implementation uses object.__setattr__ to bypass frozen dataclass restrictions,
        which is a recommended pattern for one-time initialization of computed fields in
        frozen dataclasses. This ensures:

        1. Backward compatibility - existing code with message_id=None continues to work
        2. Immutability - the dataclass remains frozen after initialization
        3. Guaranteed unique IDs - every message gets a unique identifier
        4. Clean API - users don't need to manually generate IDs in most cases

        This pattern is preferred over using field(default_factory=...) because it
        maintains the Optional[MessageId] type hint for backward compatibility while
        ensuring the field is never actually None after object creation.
        """
        if self.message_id is None:
            object.__setattr__(self, "message_id", generate_message_id())

    @property
    def text_content(self) -> str:
        """Get text content as string for backwards compatibility."""
        return get_text_content(self.content)

    @classmethod
    def create(
        cls,
        role: ContentRole,
        content: str,
        attachments: Optional[List[Attachment]] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        message_id: Optional[MessageId] = None,
    ) -> "Message":
        """Create a message with string content and optional attachments."""
        return cls(
            role=role,
            content=content,
            attachments=attachments,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            message_id=message_id,
        )


def get_text_content(content: Union[str, List[MessageContentPart]]) -> str:
    """Extract text content from message content."""
    if isinstance(content, str):
        return content

    text_parts = [part.text for part in content if part.type == "text" and part.text]
    return " ".join(text_parts)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model behavior."""

    name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    inline_tool_schemas: Optional[bool] = (
        None  # If True, resolve $refs and inline $defs in tool schemas
    )


@dataclass(frozen=True)
class ToolSchema(Generic[Args]):
    """Schema definition for a tool."""

    name: str
    description: str
    parameters: (
        Any  # Should be a type that can validate Args (like Pydantic model or Zod equivalent)
    )
    timeout: Optional[float] = None  # Optional timeout in seconds for tool execution


@runtime_checkable
class Tool(Protocol[Args, Ctx]):
    """Protocol for tool implementations."""

    @property
    def schema(self) -> ToolSchema[Args]:
        """Tool schema including name, description, and parameter validation."""
        ...

    async def execute(self, args: Args, context: Ctx) -> Union[str, "ToolResult[Any]"]:
        """Execute the tool with given arguments and context."""
        ...

    @property
    def needs_approval(self) -> Union[bool, Callable[[Ctx, Args], Union[bool, Awaitable[bool]]]]:
        """Whether this tool requires approval before execution."""
        return False


# Function tool configuration for improved DX
class FunctionToolConfig(TypedDict):
    """Configuration for creating function-based tools with object-based API."""

    name: str
    description: str
    execute: Callable[
        [Any, Any], Union[str, "ToolResult[Any]", Awaitable[Union[str, "ToolResult[Any]"]]]
    ]
    parameters: Any  # Pydantic model or similar for parameter validation
    metadata: Optional[Dict[str, Any]]  # Optional metadata
    source: Optional[ToolSource]  # Optional source tracking
    timeout: Optional[float]  # Optional timeout in seconds for tool execution


# Type alias for tool execution functions
ToolExecuteFunction = Callable[
    [Any, Any], Union[str, "ToolResult[Any]", Awaitable[Union[str, "ToolResult[Any]"]]]
]


@dataclass(frozen=True)
class Agent(Generic[Ctx, Out]):
    """An agent definition with instructions, tools, and configuration."""

    name: str
    instructions: Callable[["RunState[Ctx]"], str]
    tools: Optional[List[Tool[Any, Ctx]]] = None
    output_codec: Optional[Any] = (
        None  # Type that can validate Out (like Pydantic model or Zod equivalent)
    )
    handoffs: Optional[List[str]] = None
    model_config: Optional[ModelConfig] = None
    advanced_config: Optional["AdvancedConfig"] = None

    def as_tool(
        self,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        max_turns: Optional[int] = None,
        custom_output_extractor: Optional[
            Callable[["RunResult[Out]"], Union[str, Awaitable[str]]]
        ] = None,
        is_enabled: Union[
            bool,
            Callable[[Any, "Agent[Ctx, Out]"], bool],
            Callable[[Any, "Agent[Ctx, Out]"], Awaitable[bool]],
        ] = True,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        preserve_session: bool = False,
    ) -> Tool[Any, Ctx]:
        """
        Convert this agent into a tool that can be used by other agents.

        Args:
            tool_name: Optional custom name for the tool (defaults to agent name)
            tool_description: Optional custom description (defaults to generic description)
            max_turns: Maximum turns for the agent execution (defaults to RunConfig max_turns)
            custom_output_extractor: Optional function to extract specific output from RunResult
            is_enabled: Whether the tool is enabled (bool, sync function, or async function)
            metadata: Optional metadata for the tool
            timeout: Optional timeout for the tool execution

        Returns:
            A Tool that wraps this agent's execution
        """
        from .agent_tool import create_agent_tool

        return create_agent_tool(
            agent=self,
            tool_name=tool_name,
            tool_description=tool_description,
            max_turns=max_turns,
            custom_output_extractor=custom_output_extractor,
            is_enabled=is_enabled,
            metadata=metadata,
            timeout=timeout,
            preserve_session=preserve_session,
        )


# Guardrail type
Guardrail = Callable[[Any], Union[ValidationResult, Awaitable[ValidationResult]]]


@dataclass(frozen=True)
class AdvancedGuardrailsConfig:
    """Configuration for advanced guardrails with LLM-based validation."""

    input_prompt: Optional[str] = None
    output_prompt: Optional[str] = None
    require_citations: bool = False
    fast_model: Optional[str] = None
    fail_safe: Literal["allow", "block"] = "allow"
    execution_mode: Literal["parallel", "sequential"] = "parallel"
    timeout_ms: int = 30000

    def __post_init__(self):
        """Validate configuration."""
        if self.timeout_ms < 1000:
            object.__setattr__(self, "timeout_ms", 1000)


@dataclass(frozen=True)
class AdvancedConfig:
    """Advanced agent configuration including guardrails."""

    guardrails: Optional[AdvancedGuardrailsConfig] = None


def validate_guardrails_config(
    config: Optional[AdvancedGuardrailsConfig],
) -> AdvancedGuardrailsConfig:
    """Validate and provide defaults for guardrails configuration."""
    if config is None:
        return AdvancedGuardrailsConfig()

    return AdvancedGuardrailsConfig(
        input_prompt=config.input_prompt.strip()
        if isinstance(config.input_prompt, str) and config.input_prompt
        else None,
        output_prompt=config.output_prompt.strip()
        if isinstance(config.output_prompt, str) and config.output_prompt
        else None,
        require_citations=config.require_citations,
        fast_model=config.fast_model.strip()
        if isinstance(config.fast_model, str) and config.fast_model
        else None,
        fail_safe=config.fail_safe,
        execution_mode=config.execution_mode,
        timeout_ms=max(1000, config.timeout_ms),
    )


def json_parse_llm_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM output, handling common formatting issues."""
    import json
    import re

    if not text:
        return None

    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the first JSON object in the text
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


@dataclass(frozen=True)
class ApprovalValue:
    """Represents an approval decision with context."""

    status: str  # 'pending', 'approved', 'rejected'
    approved: bool
    additional_context: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RunState(Generic[Ctx]):
    """Immutable state of a run."""

    run_id: RunId
    trace_id: TraceId
    messages: List[Message]
    current_agent_name: str
    context: Ctx
    turn_count: int
    approvals: Dict[str, ApprovalValue] = field(default_factory=dict)


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


# Interruption types for HITL
@dataclass(frozen=True)
class ToolApprovalInterruption(Generic[Ctx]):
    """Interruption for tool approval."""

    type: Literal["tool_approval"] = "tool_approval"
    tool_call: ToolCall = field(
        default_factory=lambda: ToolCall("", "function", ToolCallFunction("", ""))
    )
    agent: "Agent[Ctx, Any]" = None
    session_id: Optional[str] = None


# Union type for all interruptions
Interruption = Union[ToolApprovalInterruption[Any]]

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
    NetworkError,
]


@dataclass(frozen=True)
class CompletedOutcome(Generic[Out]):
    """Successful completion outcome."""

    status: Literal["completed"] = "completed"
    output: Out = field(default=None)


@dataclass(frozen=True)
class ErrorOutcome:
    """Error outcome."""

    status: Literal["error"] = "error"
    error: JAFError = field(default=None)


@dataclass(frozen=True)
class InterruptedOutcome:
    """Interrupted outcome for HITL."""

    status: Literal["interrupted"] = "interrupted"
    interruptions: List[Interruption] = field(default_factory=list)


# Union type for outcomes
RunOutcome = Union[CompletedOutcome[Out], ErrorOutcome, InterruptedOutcome]


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
    session_id: Optional[str] = None
    context: Optional[Any] = None
    messages: Optional[List[Message]] = None
    agent_name: Optional[str] = None


@dataclass(frozen=True)
class RunStartEvent:
    type: Literal["run_start"] = "run_start"
    data: RunStartEventData = field(
        default_factory=lambda: RunStartEventData(RunId(""), TraceId(""))
    )


@dataclass(frozen=True)
class LLMCallStartEventData:
    """Data for LLM call start events."""

    agent_name: str
    model: str
    trace_id: TraceId
    run_id: RunId
    context: Optional[Any] = None
    messages: Optional[List[Message]] = None


@dataclass(frozen=True)
class LLMCallStartEvent:
    type: Literal["llm_call_start"] = "llm_call_start"
    data: LLMCallStartEventData = field(
        default_factory=lambda: LLMCallStartEventData("", "", TraceId(""), RunId(""))
    )


@dataclass(frozen=True)
class LLMCallEndEventData:
    """Data for LLM call end events."""

    choice: Any
    trace_id: TraceId
    run_id: RunId
    usage: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class LLMCallEndEvent:
    type: Literal["llm_call_end"] = "llm_call_end"
    data: LLMCallEndEventData = field(
        default_factory=lambda: LLMCallEndEventData(None, TraceId(""), RunId(""))
    )


@dataclass(frozen=True)
class AssistantMessageEventData:
    """Data for assistant message events (partial or complete)."""

    message: Message


@dataclass(frozen=True)
class AssistantMessageEvent:
    type: Literal["assistant_message"] = "assistant_message"
    data: AssistantMessageEventData = field(
        default_factory=lambda: AssistantMessageEventData(
            Message(role=ContentRole.ASSISTANT, content="")
        )
    )


@dataclass(frozen=True)
class ToolCallStartEventData:
    """Data for tool call start events."""

    tool_name: str
    args: Any
    trace_id: TraceId
    run_id: RunId
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ToolCallStartEvent:
    type: Literal["tool_call_start"] = "tool_call_start"
    data: ToolCallStartEventData = field(
        default_factory=lambda: ToolCallStartEventData("", None, TraceId(""), RunId(""), None)
    )


@dataclass(frozen=True)
class ToolCallEndEventData:
    """
    Data for tool call end events.

    IMPORTANT: There are two different status concepts:
    1. execution_status (this field): Indicates whether the tool execution itself succeeded or failed
       - 'success': Tool executed without errors
       - 'error': Tool execution failed due to validation, not found, or runtime errors
       - 'timeout': Tool execution timed out

    2. hitl_status (in result JSON): Indicates HITL workflow status
       - 'executed': Tool ran normally (no approval needed)
       - 'approved_and_executed': Tool required approval, was approved, and executed
       - 'pending_approval': Tool requires approval and is waiting
       - 'rejected': Tool was rejected by user
       - 'execution_error', 'validation_error', etc.: Various error states
    """

    tool_name: str
    result: str
    trace_id: TraceId
    run_id: RunId
    tool_result: Optional[Any] = None
    execution_status: Optional[str] = (
        None  # success/error/timeout - indicates if tool executed successfully
    )
    status: Optional[str] = (
        None  # DEPRECATED: maintained for backward-compatible initialization/serialization
    )
    call_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Handle backward compatibility with explicit conflict detection
        if (
            self.execution_status is not None
            and self.status is not None
            and self.execution_status != self.status
        ):
            raise ValueError(
                f"Conflicting values for execution_status ('{self.execution_status}') and status ('{self.status}'). "
                f"Please use only execution_status for new code."
            )

        # Prefer execution_status (new field) over status (deprecated field)
        canonical = self.execution_status if self.execution_status is not None else self.status
        object.__setattr__(self, "execution_status", canonical)
        object.__setattr__(self, "status", canonical)


@dataclass(frozen=True)
class ToolCallEndEvent:
    type: Literal["tool_call_end"] = "tool_call_end"
    data: ToolCallEndEventData = field(
        default_factory=lambda: ToolCallEndEventData("", "", TraceId(""), RunId(""), None, None)
    )


@dataclass(frozen=True)
class HandoffEventData:
    """Data for handoff events."""

    from_: str = field(metadata={"alias": "from"})  # Using from_ since 'from' is a Python keyword
    to: str


@dataclass(frozen=True)
class HandoffEvent:
    type: Literal["handoff"] = "handoff"
    data: HandoffEventData = field(default_factory=lambda: HandoffEventData("", ""))


@dataclass(frozen=True)
class RunEndEventData:
    """Data for run end events."""

    outcome: "RunOutcome[Any]"
    trace_id: TraceId
    run_id: RunId


@dataclass(frozen=True)
class RunEndEvent:
    type: Literal["run_end"] = "run_end"
    data: RunEndEventData = field(
        default_factory=lambda: RunEndEventData(None, TraceId(""), RunId(""))
    )


@dataclass(frozen=True)
class GuardrailEventData:
    """Data for guardrail check events."""

    guardrail_name: str
    content: Any
    is_valid: Optional[bool] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class GuardrailEvent:
    type: Literal["guardrail_check"] = "guardrail_check"
    data: GuardrailEventData = field(default_factory=lambda: GuardrailEventData(""))


@dataclass(frozen=True)
class GuardrailViolationEventData:
    """Data for guardrail violation events."""

    stage: Literal["input", "output"]
    reason: str


@dataclass(frozen=True)
class GuardrailViolationEvent:
    type: Literal["guardrail_violation"] = "guardrail_violation"
    data: GuardrailViolationEventData = field(
        default_factory=lambda: GuardrailViolationEventData("input", "")
    )


@dataclass(frozen=True)
class MemoryEventData:
    """Data for memory operation events."""

    operation: Literal["load", "store"]
    conversation_id: str
    status: Literal["start", "end", "fail"]
    error: Optional[str] = None
    message_count: Optional[int] = None


@dataclass(frozen=True)
class MemoryEvent:
    type: Literal["memory_operation"] = "memory_operation"
    data: MemoryEventData = field(default_factory=lambda: MemoryEventData("load", "", "start"))


@dataclass(frozen=True)
class OutputParseEventData:
    """Data for output parsing events."""

    content: str
    status: Literal["start", "end", "fail"]
    parsed_output: Optional[Any] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class OutputParseEvent:
    type: Literal["output_parse"] = "output_parse"
    data: OutputParseEventData = field(default_factory=lambda: OutputParseEventData("", "start"))


@dataclass(frozen=True)
class RetryEventData:
    """Data for retry events."""

    attempt: int  # Current retry attempt (1-indexed)
    max_retries: int  # Maximum number of retries configured
    reason: str  # Reason for retry (e.g., "HTTP 429 - Rate Limit", "HTTP 500 - Server Error")
    operation: Literal["llm_call", "tool_call", "workflow_step"]  # What operation is being retried
    trace_id: TraceId
    run_id: RunId
    delay: Optional[float] = None  # Backoff delay in seconds before next retry
    error_details: Optional[Dict[str, Any]] = None  # Additional error context


@dataclass(frozen=True)
class RetryEvent:
    """Event emitted when a retry occurs."""

    type: Literal["retry"] = "retry"
    data: RetryEventData = field(
        default_factory=lambda: RetryEventData(
            attempt=1,
            max_retries=3,
            reason="",
            operation="llm_call",
            trace_id=TraceId(""),
            run_id=RunId(""),
        )
    )


# Union type for all trace events
TraceEvent = Union[
    RunStartEvent,
    GuardrailEvent,
    GuardrailViolationEvent,
    MemoryEvent,
    OutputParseEvent,
    LLMCallStartEvent,
    LLMCallEndEvent,
    AssistantMessageEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    HandoffEvent,
    RunEndEvent,
    RetryEvent,
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


# Streaming chunk structures for provider-level streaming support
@dataclass(frozen=True)
class ToolCallFunctionDelta:
    """Function fields that may stream as deltas."""

    name: Optional[str] = None
    arguments_delta: Optional[str] = None


@dataclass(frozen=True)
class ToolCallDelta:
    """Represents a partial tool call delta in a streamed response."""

    index: int
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: Optional[ToolCallFunctionDelta] = None


@dataclass(frozen=True)
class CompletionStreamChunk:
    """A streamed chunk from the model provider."""

    delta: Optional[str] = None
    tool_call_delta: Optional[ToolCallDelta] = None
    is_done: Optional[bool] = False
    finish_reason: Optional[str] = None
    raw: Optional[Any] = None


@runtime_checkable
class ModelProvider(Protocol[Ctx]):
    """Protocol for model providers."""

    async def get_completion(
        self, state: RunState[Ctx], agent: Agent[Ctx, Any], config: "RunConfig[Ctx]"
    ) -> ModelCompletionResponse:
        """Get completion from the model."""
        ...

    async def get_completion_stream(
        self, state: RunState[Ctx], agent: Agent[Ctx, Any], config: "RunConfig[Ctx]"
    ) -> AsyncIterator[CompletionStreamChunk]:
        """Optional streaming API: yields incremental deltas while generating."""
        ...


@dataclass(frozen=True)
class RunConfig(Generic[Ctx]):
    """Configuration for running agents."""

    agent_registry: Dict[str, Agent[Ctx, Any]]
    model_provider: ModelProvider[Ctx]
    max_turns: Optional[int] = 50
    max_tokens: Optional[int] = None  # Default max_tokens for all agents (can be overridden per agent)
    model_override: Optional[str] = None
    initial_input_guardrails: Optional[List[Guardrail]] = None
    final_output_guardrails: Optional[List[Guardrail]] = None
    on_event: Optional[Callable[[TraceEvent], None]] = None
    memory: Optional[Any] = None  # MemoryConfig - avoiding circular import
    conversation_id: Optional[str] = None
    default_fast_model: Optional[str] = None  # Default model for fast operations like guardrails
    default_tool_timeout: Optional[float] = 300.0  # Default timeout for tool execution in seconds
    approval_storage: Optional["ApprovalStorage"] = None  # Storage for approval decisions
    before_llm_call: Optional[
        Callable[[RunState[Ctx], Agent[Ctx, Any]], Union[RunState[Ctx], Awaitable[RunState[Ctx]]]]
    ] = None  # Callback before LLM call - can modify context/messages
    after_llm_call: Optional[
        Callable[
            [RunState[Ctx], ModelCompletionResponse],
            Union[ModelCompletionResponse, Awaitable[ModelCompletionResponse]],
        ]
    ] = None  # Callback after LLM call - can process response
    max_empty_response_retries: int = 3  # Maximum retries when LLM returns empty response
    empty_response_retry_delay: float = (
        1.0  # Initial delay in seconds before retrying empty response (uses exponential backoff)
    )
    log_empty_responses: bool = True  # Whether to log diagnostic info for empty responses
    prefer_streaming: Optional[bool] = (
        None  # Whether to prefer streaming responses. None (default) = use streaming if available, True = prefer streaming, False = disable streaming
    )


# Regeneration types for conversation management
@dataclass(frozen=True)
class RegenerationRequest:
    """Request to regenerate a conversation from a specific message."""

    conversation_id: str
    message_id: MessageId  # ID of the message to regenerate from
    context: Optional[Dict[str, Any]] = None  # Optional context override


@dataclass(frozen=True)
class RegenerationContext:
    """Context information for a regeneration operation."""

    original_message_count: int
    truncated_at_index: int
    regenerated_message_id: MessageId
    regeneration_id: str  # Unique ID for this regeneration operation
    timestamp: int  # Unix timestamp in milliseconds


# Checkpoint types for conversation management
@dataclass(frozen=True)
class CheckpointRequest:
    """Request to checkpoint a conversation after a specific message."""

    conversation_id: str
    message_id: MessageId  # ID of the message to checkpoint after (this message is kept)
    context: Optional[Dict[str, Any]] = None  # Optional context for the checkpoint


@dataclass(frozen=True)
class CheckpointContext:
    """Context information for a checkpoint operation."""

    original_message_count: int
    checkpointed_at_index: int
    checkpointed_message_id: MessageId
    checkpoint_id: str  # Unique ID for this checkpoint operation
    timestamp: int  # Unix timestamp in milliseconds


# Message utility functions
def find_message_index(messages: List[Message], message_id: MessageId) -> Optional[int]:
    """Find the index of a message by its ID."""
    for i, msg in enumerate(messages):
        if msg.message_id == message_id:
            return i
    return None


def truncate_messages_after(messages: List[Message], message_id: MessageId) -> List[Message]:
    """Truncate messages after (and including) the specified message ID."""
    index = find_message_index(messages, message_id)
    if index is None:
        return messages  # Message not found, return unchanged
    return messages[:index]


def get_message_by_id(messages: List[Message], message_id: MessageId) -> Optional[Message]:
    """Get a message by its ID."""
    for msg in messages:
        if msg.message_id == message_id:
            return msg
    return None
