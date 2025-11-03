"""
ADK Core Types - Production-Ready Type System

This module defines the core types for the ADK layer that bridge between
high-level ADK concepts and the underlying JAF Core implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    Generic,
    Callable,
    AsyncIterator,
    Tuple,
    Mapping,
)
from pydantic import BaseModel

# Re-export core JAF types for compatibility
from jaf.core.types import Message as CoreMessage, Agent as CoreAgent, Tool as CoreTool

# Generic type variables
T = TypeVar("T")
E = TypeVar("E")

# ========== Model and Provider Types ==========


class AdkModelType(str, Enum):
    """ADK model types with production-ready support."""

    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class AdkProviderType(str, Enum):
    """ADK provider types for LLM services."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LITELLM = "litellm"


# ========== Result Types ==========


@dataclass
class AdkSuccess(Generic[T]):
    """Represents a successful ADK operation."""

    data: T
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AdkFailure(Generic[E]):
    """Represents a failed ADK operation."""

    error: E
    metadata: Optional[Dict[str, Any]] = None


# Union type for ADK results
AdkResult = Union[AdkSuccess[T], AdkFailure[E]]

# ========== Message Types ==========


@dataclass
class AdkMessage:
    """ADK message type with production features."""

    role: str  # 'user', 'assistant', 'tool', 'system'
    content: str
    timestamp: Optional[datetime] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_core_message(self) -> CoreMessage:
        """Convert to JAF Core Message format."""
        return CoreMessage(
            role=self.role,
            content=self.content,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
        )

    @classmethod
    def from_core_message(cls, msg: CoreMessage) -> "AdkMessage":
        """Create from JAF Core Message."""
        return cls(
            role=msg.role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            timestamp=datetime.now(),
        )


# ========== Tool Types ==========


class AdkTool(ABC):
    """ADK tool interface with production features."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema."""
        pass

    @abstractmethod
    async def execute(
        self, args: Dict[str, Any], context: "AdkContext"
    ) -> AdkResult[Any, Exception]:
        """Execute the tool with given arguments."""
        pass


# ========== Agent Types ==========


@dataclass
class AdkAgent:
    """ADK agent with production-ready features."""

    name: str
    instructions: Union[str, Callable[["AdkContext"], str]]
    model: AdkModelType = AdkModelType.GPT_4O
    tools: List[AdkTool] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []

    def to_core_agent(self, context: "AdkContext") -> CoreAgent:
        """Convert to JAF Core Agent format."""
        from jaf.core.types import ModelConfig

        # Handle instructions
        if callable(self.instructions):
            instruction_text = self.instructions(context)
        else:
            instruction_text = self.instructions

        # Create model config
        model_config = ModelConfig(
            name=self.model.value, temperature=self.temperature, max_tokens=self.max_tokens
        )

        # Convert tools (simplified for now)
        core_tools = []  # Will be populated by tool conversion logic

        return CoreAgent(
            name=self.name,
            instructions=lambda state: instruction_text,
            tools=core_tools,
            model_config=model_config,
        )


# ========== Session Types ==========


@dataclass
class AdkSession:
    """ADK session with production-grade features."""

    session_id: str
    user_id: str
    app_name: str
    messages: List[AdkMessage]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    def add_message(self, message: AdkMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[AdkMessage]:
        """Get recent messages from the session."""
        return self.messages[-limit:] if limit > 0 else self.messages


# ========== Context Types ==========


@dataclass
class AdkContext:
    """ADK execution context with production features."""

    user_id: str
    session_id: Optional[str] = None
    app_name: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the context metadata."""
        if self.metadata:
            return self.metadata.get(key, default)
        return default

    def set_context_value(self, key: str, value: Any) -> None:
        """Set a value in the context metadata."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value


# ========== Stream Types ==========


@dataclass
class AdkStreamChunk:
    """ADK streaming chunk with production features."""

    delta: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    is_done: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ========== Configuration Types ==========


@dataclass
class AdkConfiguration:
    """ADK global configuration."""

    default_model: AdkModelType = AdkModelType.GPT_4O
    default_provider: AdkProviderType = AdkProviderType.OPENAI
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_streaming: bool = True
    enable_function_calling: bool = True
    debug_mode: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ========== Utility Functions ==========


def create_adk_message(role: str, content: str, **kwargs) -> AdkMessage:
    """Create an ADK message with timestamp."""
    return AdkMessage(role=role, content=content, timestamp=datetime.now(), **kwargs)


def create_user_message(content: str, **kwargs) -> AdkMessage:
    """Create a user message."""
    return create_adk_message("user", content, **kwargs)


def create_assistant_message(content: str, **kwargs) -> AdkMessage:
    """Create an assistant message."""
    return create_adk_message("assistant", content, **kwargs)


def create_system_message(content: str, **kwargs) -> AdkMessage:
    """Create a system message."""
    return create_adk_message("system", content, **kwargs)


def create_adk_context(user_id: str, **kwargs) -> AdkContext:
    """Create an ADK context."""
    return AdkContext(user_id=user_id, **kwargs)


# ========== Immutable Session Types ==========


@dataclass(frozen=True)
class ImmutableAdkSession:
    """
    Immutable ADK session following functional programming principles.

    This session type eliminates all mutable state and provides functional
    methods for creating new sessions with modifications, ensuring predictable
    behavior and eliminating race conditions.
    """

    session_id: str
    user_id: str
    app_name: str
    messages: Tuple[AdkMessage, ...]  # Immutable tuple instead of list
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Mapping[str, Any]] = None  # Immutable mapping

    def with_message(self, message: AdkMessage) -> "ImmutableAdkSession":
        """
        Create a new session with an additional message.

        This is a pure function that returns a new session instance without
        modifying the original session, following functional programming principles.

        Args:
            message: Message to add to the session

        Returns:
            New ImmutableAdkSession with the added message
        """
        return ImmutableAdkSession(
            session_id=self.session_id,
            user_id=self.user_id,
            app_name=self.app_name,
            messages=self.messages + (message,),  # Create new tuple
            created_at=self.created_at,
            updated_at=datetime.now(),
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: Any) -> "ImmutableAdkSession":
        """
        Create a new session with updated metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            New ImmutableAdkSession with updated metadata
        """
        current_metadata = dict(self.metadata) if self.metadata else {}
        current_metadata[key] = value

        return ImmutableAdkSession(
            session_id=self.session_id,
            user_id=self.user_id,
            app_name=self.app_name,
            messages=self.messages,
            created_at=self.created_at,
            updated_at=datetime.now(),
            metadata=current_metadata,
        )

    def get_recent_messages(self, limit: int = 10) -> Tuple[AdkMessage, ...]:
        """
        Get recent messages from the session.

        This is a pure function that doesn't modify state.

        Args:
            limit: Maximum number of messages to return

        Returns:
            Tuple of recent messages
        """
        if limit <= 0:
            return self.messages
        return self.messages[-limit:]


# Factory function for creating immutable sessions
def create_immutable_session(
    session_id: str,
    user_id: str,
    app_name: str,
    initial_messages: Optional[List[AdkMessage]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImmutableAdkSession:
    """
    Create a new immutable session.

    Args:
        session_id: Unique session identifier
        user_id: User identifier
        app_name: Application name
        initial_messages: Optional initial messages
        metadata: Optional metadata

    Returns:
        New ImmutableAdkSession instance
    """
    now = datetime.now()
    messages = tuple(initial_messages) if initial_messages else ()

    return ImmutableAdkSession(
        session_id=session_id,
        user_id=user_id,
        app_name=app_name,
        messages=messages,
        created_at=now,
        updated_at=now,
        metadata=metadata,
    )


# Pure function for session operations
def add_message_to_session(
    session: ImmutableAdkSession, message: AdkMessage
) -> ImmutableAdkSession:
    """
    Pure function to add a message to a session.

    Args:
        session: Original session
        message: Message to add

    Returns:
        New session with added message
    """
    return session.with_message(message)
