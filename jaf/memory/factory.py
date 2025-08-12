"""
Memory provider factory for the JAF framework.

This module provides factory functions to create memory providers based on environment
configuration, enabling flexible deployment across different storage backends.
"""

import os
from typing import Optional, Dict, Any, Union

from .types import (
    MemoryProvider, InMemoryConfig, RedisConfig, PostgresConfig,
    MemoryConnectionError
)
from .providers.in_memory import create_in_memory_provider
from .providers.redis import create_redis_provider
from .providers.postgres import create_postgres_provider

async def create_memory_provider_from_env(
    external_clients: Optional[Dict[str, Any]] = None
) -> MemoryProvider:
    """
    Create a memory provider based on environment variables.
    
    Environment variables:
    - JAF_MEMORY_TYPE: "memory", "redis", or "postgres" (default: "memory")
    
    For Redis:
    - JAF_REDIS_URL: Full Redis URL (redis://host:port/db)
    - JAF_REDIS_HOST: Redis host (default: "localhost")
    - JAF_REDIS_PORT: Redis port (default: 6379)
    - JAF_REDIS_PASSWORD: Redis password
    - JAF_REDIS_DB: Redis database number (default: 0)
    - JAF_REDIS_KEY_PREFIX: Key prefix (default: "jaf:memory:")
    - JAF_REDIS_TTL: TTL in seconds for conversation keys
    
    For PostgreSQL:
    - JAF_POSTGRES_CONNECTION_STRING: Full connection string
    - JAF_POSTGRES_HOST: PostgreSQL host (default: "localhost")
    - JAF_POSTGRES_PORT: PostgreSQL port (default: 5432)
    - JAF_POSTGRES_DATABASE: Database name (default: "jaf_memory")
    - JAF_POSTGRES_USERNAME: Username (default: "postgres")
    - JAF_POSTGRES_PASSWORD: Password
    - JAF_POSTGRES_SSL: Enable SSL (default: false)
    - JAF_POSTGRES_TABLE_NAME: Table name (default: "conversations")
    - JAF_POSTGRES_MAX_CONNECTIONS: Max connections (default: 10)
    
    For In-Memory:
    - JAF_MEMORY_MAX_CONVERSATIONS: Max conversations (default: 1000)
    - JAF_MEMORY_MAX_MESSAGES: Max messages per conversation (default: 1000)
    
    Args:
        external_clients: Dictionary of pre-initialized client connections.
                         Keys: "redis" or "postgres" for respective clients.
                         
    Returns:
        Configured memory provider instance.
        
    Raises:
        MemoryConnectionError: If provider creation fails.
        ValueError: If unsupported memory type is specified.
    """
    memory_type = os.getenv("JAF_MEMORY_TYPE", "memory").lower()
    external_clients = external_clients or {}
    
    print(f"[MEMORY:Factory] Creating {memory_type} provider from environment")
    
    if memory_type == "memory":
        return await _create_in_memory_from_env()
    elif memory_type == "redis":
        return await _create_redis_from_env(external_clients)
    elif memory_type == "postgres":
        return await _create_postgres_from_env(external_clients)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}. Must be 'memory', 'redis', or 'postgres'")

async def _create_in_memory_from_env() -> MemoryProvider:
    """Create in-memory provider from environment variables."""
    config = InMemoryConfig(
        max_conversations=int(os.getenv("JAF_MEMORY_MAX_CONVERSATIONS", "1000")),
        max_messages=int(os.getenv("JAF_MEMORY_MAX_MESSAGES", "1000"))
    )
    
    return create_in_memory_provider(config)

async def _create_redis_from_env(external_clients: Dict[str, Any]) -> MemoryProvider:
    """Create Redis provider from environment variables."""
    # Use external client if provided
    redis_client = external_clients.get("redis")
    
    if redis_client is None:
        # Create new Redis client
        try:
            import redis.asyncio as redis
        except ImportError:
            raise MemoryConnectionError(
                "Redis not installed. Run: pip install redis",
                "Redis"
            )
        
        # Configure Redis connection
        redis_url = os.getenv("JAF_REDIS_URL")
        if redis_url:
            # Use full URL if provided
            redis_client = redis.from_url(redis_url)
        else:
            # Build connection from individual parameters
            redis_client = redis.Redis(
                host=os.getenv("JAF_REDIS_HOST", "localhost"),
                port=int(os.getenv("JAF_REDIS_PORT", "6379")),
                password=os.getenv("JAF_REDIS_PASSWORD"),
                db=int(os.getenv("JAF_REDIS_DB", "0")),
                decode_responses=False  # We handle encoding ourselves
            )
    
    # Create configuration
    config = RedisConfig(
        url=os.getenv("JAF_REDIS_URL"),
        host=os.getenv("JAF_REDIS_HOST", "localhost"),
        port=int(os.getenv("JAF_REDIS_PORT", "6379")),
        password=os.getenv("JAF_REDIS_PASSWORD"),
        db=int(os.getenv("JAF_REDIS_DB", "0")),
        key_prefix=os.getenv("JAF_REDIS_KEY_PREFIX", "jaf:memory:"),
        ttl=int(os.getenv("JAF_REDIS_TTL")) if os.getenv("JAF_REDIS_TTL") else None
    )
    
    return await create_redis_provider(config, redis_client)

async def _create_postgres_from_env(external_clients: Dict[str, Any]) -> MemoryProvider:
    """Create PostgreSQL provider from environment variables."""
    # Use external client if provided
    postgres_client = external_clients.get("postgres")
    
    if postgres_client is None:
        # Create new PostgreSQL client
        try:
            import asyncpg
        except ImportError:
            raise MemoryConnectionError(
                "asyncpg not installed. Run: pip install asyncpg",
                "PostgreSQL"
            )
        
        # Configure PostgreSQL connection
        connection_string = os.getenv("JAF_POSTGRES_CONNECTION_STRING")
        if connection_string:
            # Use full connection string if provided
            postgres_client = await asyncpg.connect(connection_string)
        else:
            # Build connection from individual parameters
            postgres_client = await asyncpg.connect(
                host=os.getenv("JAF_POSTGRES_HOST", "localhost"),
                port=int(os.getenv("JAF_POSTGRES_PORT", "5432")),
                database=os.getenv("JAF_POSTGRES_DATABASE", "jaf_memory"),
                user=os.getenv("JAF_POSTGRES_USERNAME", "postgres"),
                password=os.getenv("JAF_POSTGRES_PASSWORD"),
                ssl=os.getenv("JAF_POSTGRES_SSL", "false").lower() == "true"
            )
    
    # Create configuration
    config = PostgresConfig(
        connection_string=os.getenv("JAF_POSTGRES_CONNECTION_STRING"),
        host=os.getenv("JAF_POSTGRES_HOST", "localhost"),
        port=int(os.getenv("JAF_POSTGRES_PORT", "5432")),
        database=os.getenv("JAF_POSTGRES_DATABASE", "jaf_memory"),
        username=os.getenv("JAF_POSTGRES_USERNAME", "postgres"),
        password=os.getenv("JAF_POSTGRES_PASSWORD"),
        ssl=os.getenv("JAF_POSTGRES_SSL", "false").lower() == "true",
        table_name=os.getenv("JAF_POSTGRES_TABLE_NAME", "conversations"),
        max_connections=int(os.getenv("JAF_POSTGRES_MAX_CONNECTIONS", "10"))
    )
    
    return await create_postgres_provider(config, postgres_client)

def get_memory_provider_info() -> Dict[str, Any]:
    """
    Get information about the configured memory provider without creating it.
    
    Returns:
        Dictionary with provider type and configuration summary.
    """
    memory_type = os.getenv("JAF_MEMORY_TYPE", "memory").lower()
    
    info = {
        "type": memory_type,
        "environment_variables": []
    }
    
    if memory_type == "memory":
        info.update({
            "max_conversations": int(os.getenv("JAF_MEMORY_MAX_CONVERSATIONS", "1000")),
            "max_messages": int(os.getenv("JAF_MEMORY_MAX_MESSAGES", "1000")),
            "persistence": False
        })
        info["environment_variables"] = [
            "JAF_MEMORY_MAX_CONVERSATIONS",
            "JAF_MEMORY_MAX_MESSAGES"
        ]
    
    elif memory_type == "redis":
        info.update({
            "host": os.getenv("JAF_REDIS_HOST", "localhost"),
            "port": int(os.getenv("JAF_REDIS_PORT", "6379")),
            "db": int(os.getenv("JAF_REDIS_DB", "0")),
            "key_prefix": os.getenv("JAF_REDIS_KEY_PREFIX", "jaf:memory:"),
            "ttl": int(os.getenv("JAF_REDIS_TTL")) if os.getenv("JAF_REDIS_TTL") else None,
            "persistence": True
        })
        info["environment_variables"] = [
            "JAF_REDIS_URL",
            "JAF_REDIS_HOST", 
            "JAF_REDIS_PORT",
            "JAF_REDIS_PASSWORD",
            "JAF_REDIS_DB",
            "JAF_REDIS_KEY_PREFIX",
            "JAF_REDIS_TTL"
        ]
    
    elif memory_type == "postgres":
        info.update({
            "host": os.getenv("JAF_POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("JAF_POSTGRES_PORT", "5432")),
            "database": os.getenv("JAF_POSTGRES_DATABASE", "jaf_memory"),
            "username": os.getenv("JAF_POSTGRES_USERNAME", "postgres"),
            "table_name": os.getenv("JAF_POSTGRES_TABLE_NAME", "conversations"),
            "ssl": os.getenv("JAF_POSTGRES_SSL", "false").lower() == "true",
            "persistence": True
        })
        info["environment_variables"] = [
            "JAF_POSTGRES_CONNECTION_STRING",
            "JAF_POSTGRES_HOST",
            "JAF_POSTGRES_PORT", 
            "JAF_POSTGRES_DATABASE",
            "JAF_POSTGRES_USERNAME",
            "JAF_POSTGRES_PASSWORD",
            "JAF_POSTGRES_SSL",
            "JAF_POSTGRES_TABLE_NAME",
            "JAF_POSTGRES_MAX_CONNECTIONS"
        ]
    
    return info

async def test_memory_provider_connection(provider_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Test memory provider connection without creating a full provider.
    
    Args:
        provider_type: Type of provider to test. If None, uses JAF_MEMORY_TYPE.
        
    Returns:
        Dictionary with connection test results.
    """
    test_type = provider_type or os.getenv("JAF_MEMORY_TYPE", "memory").lower()
    
    try:
        if test_type == "memory":
            return {
                "type": "memory",
                "healthy": True,
                "message": "In-memory provider always available"
            }
        
        elif test_type == "redis":
            try:
                import redis.asyncio as redis
            except ImportError:
                return {
                    "type": "redis",
                    "healthy": False,
                    "error": "Redis library not installed. Run: pip install redis"
                }
            
            # Test Redis connection
            redis_url = os.getenv("JAF_REDIS_URL")
            if redis_url:
                client = redis.from_url(redis_url)
            else:
                client = redis.Redis(
                    host=os.getenv("JAF_REDIS_HOST", "localhost"),
                    port=int(os.getenv("JAF_REDIS_PORT", "6379")),
                    password=os.getenv("JAF_REDIS_PASSWORD"),
                    db=int(os.getenv("JAF_REDIS_DB", "0"))
                )
            
            await client.ping()
            await client.close()
            
            return {
                "type": "redis",
                "healthy": True,
                "message": "Redis connection successful"
            }
        
        elif test_type == "postgres":
            try:
                import asyncpg
            except ImportError:
                return {
                    "type": "postgres",
                    "healthy": False,
                    "error": "asyncpg library not installed. Run: pip install asyncpg"
                }
            
            # Test PostgreSQL connection
            connection_string = os.getenv("JAF_POSTGRES_CONNECTION_STRING")
            if connection_string:
                conn = await asyncpg.connect(connection_string)
            else:
                conn = await asyncpg.connect(
                    host=os.getenv("JAF_POSTGRES_HOST", "localhost"),
                    port=int(os.getenv("JAF_POSTGRES_PORT", "5432")),
                    database=os.getenv("JAF_POSTGRES_DATABASE", "jaf_memory"),
                    user=os.getenv("JAF_POSTGRES_USERNAME", "postgres"),
                    password=os.getenv("JAF_POSTGRES_PASSWORD"),
                    ssl=os.getenv("JAF_POSTGRES_SSL", "false").lower() == "true"
                )
            
            await conn.execute("SELECT 1")
            await conn.close()
            
            return {
                "type": "postgres",
                "healthy": True,
                "message": "PostgreSQL connection successful"
            }
        
        else:
            return {
                "type": test_type,
                "healthy": False,
                "error": f"Unsupported provider type: {test_type}"
            }
    
    except Exception as e:
        return {
            "type": test_type,
            "healthy": False,
            "error": str(e)
        }