"""
Memory system types for the JAF framework.

This module provides the core data structures and interfaces for persistent conversation storage,
including provider abstractions, configuration models, and error handling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, Union

from pydantic import BaseModel, Field

from ..core.types import Message, TraceId

# Generic Result type for functional error handling
T = TypeVar('T')
E = TypeVar('E', bound='MemoryError')

@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful operation result."""
    data: T

@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed operation result."""
    error: E

Result = Union[Success[T], Failure[E]]

@dataclass(frozen=True)
class ConversationMemory:
    """
    Immutable conversation memory object containing conversation history and metadata.
    
    This represents a complete conversation stored in memory, including all messages
    and associated metadata like creation time, user information, and trace data.
    """
    conversation_id: str
    user_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure messages list is frozen (immutable)."""
        if self.messages is not None:
            object.__setattr__(self, 'messages', tuple(self.messages))

@dataclass(frozen=True)
class MemoryQuery:
    """Query parameters for searching conversations in memory providers."""
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[TraceId] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None

class MemoryProvider(Protocol):
    """
    Protocol defining the interface that all memory providers must implement.
    
    This protocol ensures consistent behavior across different storage backends
    (in-memory, Redis, PostgreSQL) while maintaining type safety.
    """

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, 'MemoryStorageError']:
        """Store messages for a conversation."""
        ...

    async def get_conversation(
        self,
        conversation_id: str
    ) -> Result[Optional[ConversationMemory], 'MemoryStorageError']:
        """Retrieve conversation history."""
        ...

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, Union['MemoryNotFoundError', 'MemoryStorageError']]:
        """Append new messages to existing conversation."""
        ...

    async def find_conversations(
        self,
        query: MemoryQuery
    ) -> Result[List[ConversationMemory], 'MemoryStorageError']:
        """Search conversations by query parameters."""
        ...

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> Result[List[Message], Union['MemoryNotFoundError', 'MemoryStorageError']]:
        """Get recent messages from a conversation."""
        ...

    async def delete_conversation(
        self,
        conversation_id: str
    ) -> Result[bool, 'MemoryStorageError']:
        """Delete conversation and return True if it existed."""
        ...

    async def clear_user_conversations(
        self,
        user_id: str
    ) -> Result[int, 'MemoryStorageError']:
        """Clear all conversations for a user and return count deleted."""
        ...

    async def get_stats(
        self,
        user_id: Optional[str] = None
    ) -> Result[Dict[str, Any], 'MemoryStorageError']:
        """Get conversation statistics."""
        ...

    async def health_check(self) -> Result[Dict[str, Any], 'MemoryConnectionError']:
        """Check provider health and return status information."""
        ...

    async def close(self) -> Result[None, 'MemoryConnectionError']:
        """Close/cleanup the provider."""
        ...

# Configuration models using Pydantic for validation

class InMemoryConfig(BaseModel):
    """Configuration for in-memory provider."""
    type: str = Field(default="memory", json_schema_extra={"literal": True})
    max_conversations: int = Field(default=1000, ge=1)
    max_messages_per_conversation: int = Field(default=1000, ge=1)

class RedisConfig(BaseModel):
    """Configuration for Redis provider."""
    type: str = Field(default="redis", json_schema_extra={"literal": True})
    url: Optional[str] = None
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = None
    db: int = Field(default=0, ge=0)
    key_prefix: str = Field(default="jaf:memory:")
    ttl: Optional[int] = Field(default=None, ge=1)  # seconds

class PostgresConfig(BaseModel):
    """Configuration for PostgreSQL provider."""
    type: str = Field(default="postgres", json_schema_extra={"literal": True})
    connection_string: Optional[str] = None
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="jaf_memory")
    username: str = Field(default="postgres")
    password: Optional[str] = None
    ssl: bool = Field(default=False)
    table_name: str = Field(default="conversations")
    max_connections: int = Field(default=10, ge=1)

# Union type for all provider configurations
MemoryProviderConfig = Union[InMemoryConfig, RedisConfig, PostgresConfig]

# Functional error types for memory providers

class MemoryError:
    """Base class for memory-related errors."""
    def __init__(self, message: str, provider: str, cause: Optional[Exception] = None):
        self.message = message
        self.provider = provider
        self.cause = cause

    def __eq__(self, other):
        if not isinstance(other, MemoryError):
            return False
        return (self.message == other.message and 
                self.provider == other.provider and 
                self.cause == other.cause)

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, provider={self.provider!r}, cause={self.cause!r})"

class MemoryConnectionError(MemoryError):
    """Error for connection failures."""
    pass

class MemoryNotFoundError(MemoryError):
    """Error when a conversation is not found."""
    def __init__(self, message: str, provider: str, conversation_id: str, cause: Optional[Exception] = None):
        super().__init__(message, provider, cause)
        self.conversation_id = conversation_id

    def __eq__(self, other):
        if not isinstance(other, MemoryNotFoundError):
            return False
        return super().__eq__(other) and self.conversation_id == other.conversation_id

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, provider={self.provider!r}, conversation_id={self.conversation_id!r}, cause={self.cause!r})"

class MemoryStorageError(MemoryError):
    """Error for storage operation failures."""
    def __init__(self, message: str, provider: str, operation: str, cause: Optional[Exception] = None):
        super().__init__(message, provider, cause)
        self.operation = operation

    def __eq__(self, other):
        if not isinstance(other, MemoryStorageError):
            return False
        return super().__eq__(other) and self.operation == other.operation

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, provider={self.provider!r}, operation={self.operation!r}, cause={self.cause!r})"

# Union of all possible memory errors
MemoryErrorUnion = Union[MemoryConnectionError, MemoryNotFoundError, MemoryStorageError]

# Memory configuration for the engine - using dataclass instead of Pydantic
@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory integration in the engine."""
    provider: MemoryProvider
    auto_store: bool = True
    max_messages: Optional[int] = None
    ttl: Optional[int] = None
    compression_threshold: Optional[int] = None
