"""
Memory system for the JAF framework.

This module provides persistent conversation storage across different backends
including in-memory, Redis, and PostgreSQL providers.
"""

from .factory import create_memory_provider_from_env
from .providers.in_memory import create_in_memory_provider
from .providers.postgres import create_postgres_provider
from .providers.redis import create_redis_provider
from .types import (
    # Core types
    ConversationMemory,
    Failure,
    # Configuration types
    InMemoryConfig,
    MemoryConfig,
    MemoryConnectionError,
    # Error types
    MemoryError,
    MemoryNotFoundError,
    MemoryProvider,
    MemoryProviderConfig,
    MemoryQuery,
    MemoryStorageError,
    PostgresConfig,
    RedisConfig,
    # Result types
    Result,
    Success,
)

__all__ = [
    # Core types
    "ConversationMemory",
    "MemoryProvider",
    "MemoryQuery",
    "MemoryConfig",

    # Result types
    "Result",
    "Success",
    "Failure",

    # Configuration types
    "InMemoryConfig",
    "RedisConfig",
    "PostgresConfig",
    "MemoryProviderConfig",

    # Error types
    "MemoryError",
    "MemoryConnectionError",
    "MemoryNotFoundError",
    "MemoryStorageError",

    # Factory functions
    "create_memory_provider_from_env",

    # Provider factories
    "create_in_memory_provider",
    "create_redis_provider",
    "create_postgres_provider"
]
