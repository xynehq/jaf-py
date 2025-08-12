"""
Memory system types for the JAF framework.

This module provides the core data structures and interfaces for persistent conversation storage,
including provider abstractions, configuration models, and error handling.
"""

from typing import Protocol, Union, Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
from ..core.types import Message, TraceId

# Result type for functional error handling
@dataclass(frozen=True)
class Success:
    """Represents a successful operation result."""
    pass

@dataclass(frozen=True)
class Failure:
    """Represents a failed operation result."""
    error: str

Result = Union[Success, Failure]

@dataclass(frozen=True)
class ConversationMemory:
    """
    Immutable conversation memory object containing conversation history and metadata.
    
    This represents a complete conversation stored in memory, including all messages
    and associated metadata like creation time, user information, and trace data.
    """
    conversation_id: str
    user_id: Optional[str]
    messages: List[Message]
    metadata: Optional[Dict[str, Any]] = None
    
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
    ) -> Result:
        """Store messages for a conversation."""
        ...
    
    async def get_conversation(self, conversation_id: str) -> Union[ConversationMemory, None]:
        """Retrieve conversation history."""
        ...
    
    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Append new messages to existing conversation."""
        ...
    
    async def find_conversations(self, query: MemoryQuery) -> List[ConversationMemory]:
        """Search conversations by query parameters."""
        ...
    
    async def get_recent_messages(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> List[Message]:
        """Get recent messages from a conversation."""
        ...
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and return True if it existed."""
        ...
    
    async def clear_user_conversations(self, user_id: str) -> int:
        """Clear all conversations for a user and return count deleted."""
        ...
    
    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and return status information."""
        ...
    
    async def close(self) -> None:
        """Close/cleanup the provider."""
        ...

# Configuration models using Pydantic for validation

class InMemoryConfig(BaseModel):
    """Configuration for in-memory provider."""
    type: str = Field(default="memory", literal=True)
    max_conversations: int = Field(default=1000, ge=1)
    max_messages: int = Field(default=1000, ge=1)

class RedisConfig(BaseModel):
    """Configuration for Redis provider."""
    type: str = Field(default="redis", literal=True)
    url: Optional[str] = None
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = None
    db: int = Field(default=0, ge=0)
    key_prefix: str = Field(default="jaf:memory:")
    ttl: Optional[int] = Field(default=None, ge=1)  # seconds

class PostgresConfig(BaseModel):
    """Configuration for PostgreSQL provider."""
    type: str = Field(default="postgres", literal=True)
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

# Memory error classes

class MemoryError(Exception):
    """Base exception for memory-related errors."""
    def __init__(self, message: str, provider: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.cause = cause

class MemoryConnectionError(MemoryError):
    """Raised when failing to connect to memory provider."""
    pass

class MemoryNotFoundError(MemoryError):
    """Raised when conversation is not found."""
    def __init__(self, conversation_id: str, provider: str):
        super().__init__(f"Conversation {conversation_id} not found", provider)
        self.conversation_id = conversation_id

class MemoryStorageError(MemoryError):
    """Raised when storage operation fails."""
    def __init__(self, operation: str, provider: str, cause: Optional[Exception] = None):
        super().__init__(f"Failed to {operation} in {provider}", provider, cause)
        self.operation = operation

# Memory configuration for the engine - using dataclass instead of Pydantic
@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory integration in the engine."""
    provider: MemoryProvider
    auto_store: bool = True
    max_messages: Optional[int] = None
    ttl: Optional[int] = None
    compression_threshold: Optional[int] = None