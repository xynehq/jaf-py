"""
Memory provider factory for the JAF framework.

This module provides factory functions to create memory providers based on environment
configuration, enabling flexible deployment across different storage backends.
"""

import os
from typing import Any, Dict, Optional

from .providers.in_memory import create_in_memory_provider
from .providers.postgres import create_postgres_provider
from .providers.redis import create_redis_provider
from .types import (
    Failure,
    InMemoryConfig,
    MemoryConnectionError,
    MemoryProvider,
    PostgresConfig,
    RedisConfig,
    Result,
    Success,
)


async def create_memory_provider_from_env(
    external_clients: Optional[Dict[str, Any]] = None
) -> Result[MemoryProvider, MemoryConnectionError]:
    """
    Create a memory provider based on environment variables.
    """
    memory_type = os.getenv("JAF_MEMORY_TYPE", "memory").lower()
    external_clients = external_clients or {}

    if memory_type == "memory":
        config = InMemoryConfig(
            max_conversations=int(os.getenv("JAF_MEMORY_MAX_CONVERSATIONS", "1000")),
            max_messages_per_conversation=int(os.getenv("JAF_MEMORY_MAX_MESSAGES", "1000"))
        )
        return Success(create_in_memory_provider(config))

    elif memory_type == "redis":
        redis_password = os.getenv("JAF_REDIS_PASSWORD")
        config_data = {
            "url": os.getenv("JAF_REDIS_URL"),
            "host": os.getenv("JAF_REDIS_HOST", "localhost"),
            "port": int(os.getenv("JAF_REDIS_PORT", "6379")),
            "db": int(os.getenv("JAF_REDIS_DB", "0")),
            "key_prefix": os.getenv("JAF_REDIS_PREFIX", "jaf:memory:"),
            "ttl": int(os.getenv("JAF_REDIS_TTL")) if os.getenv("JAF_REDIS_TTL") else None
        }
        if redis_password:
            config_data["password"] = redis_password

        config = RedisConfig(**config_data)
        return await create_redis_provider(config)

    elif memory_type == "postgres":
        connection_string = os.getenv("JAF_POSTGRES_CONNECTION_STRING")
        config_data = {
            "host": os.getenv("JAF_POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("JAF_POSTGRES_PORT", "5432")),
            "database": os.getenv("JAF_POSTGRES_DB", "jaf_test"),
            "username": os.getenv("JAF_POSTGRES_USER", "postgres"),
            "password": os.getenv("JAF_POSTGRES_PASSWORD"),
            "ssl": os.getenv("JAF_POSTGRES_SSL", "false").lower() == "true",
            "table_name": os.getenv("JAF_POSTGRES_TABLE", "conversations"),
            "max_connections": int(os.getenv("JAF_POSTGRES_MAX_CONNECTIONS", "10"))
        }
        if connection_string:
            config_data["connection_string"] = connection_string

        config = PostgresConfig(**config_data)
        return await create_postgres_provider(config)

    else:
        return Failure(MemoryConnectionError(f"Unsupported memory type: {memory_type}", "Factory"))
