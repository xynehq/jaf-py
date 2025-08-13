"""
A2A Task Provider Factory for JAF

This module provides factory functions for creating A2A task providers based on configuration.
It leverages the existing JAF memory infrastructure while providing A2A-specific task storage
capabilities with optimized operations for task lifecycle management.
"""

import os
from typing import Any, Dict, Optional

# Import A2A task provider implementations
from .providers.in_memory import create_a2a_in_memory_task_provider
from .providers.postgres import create_a2a_postgres_task_provider
from .providers.redis import create_a2a_redis_task_provider
from .types import (
    A2AInMemoryTaskConfig,
    A2APostgresTaskConfig,
    A2ARedisTaskConfig,
    A2AResult,
    A2ATaskProvider,
    A2ATaskProviderConfig,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_storage_error,
)


async def create_a2a_task_provider(
    config: A2ATaskProviderConfig,
    external_clients: Optional[Dict[str, Any]] = None
) -> A2AResult[A2ATaskProvider]:
    """
    Create an A2A task provider from configuration
    
    Args:
        config: Provider configuration
        external_clients: Optional external clients (redis, postgres)
        
    Returns:
        A2AResult containing the task provider or an error
    """
    try:
        external_clients = external_clients or {}

        if config.type == "memory":
            return create_a2a_success(
                create_a2a_in_memory_task_provider(config)
            )

        elif config.type == "redis":
            redis_client = external_clients.get('redis')
            if not redis_client:
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'create-provider',
                        'redis',
                        None,
                        Exception('Redis client instance required. Please provide a Redis client in external_clients.redis')
                    )
                )
            return await create_a2a_redis_task_provider(config, redis_client)

        elif config.type == "postgres":
            postgres_client = external_clients.get('postgres')
            if not postgres_client:
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'create-provider',
                        'postgres',
                        None,
                        Exception('PostgreSQL client instance required. Please provide a PostgreSQL client in external_clients.postgres')
                    )
                )
            return await create_a2a_postgres_task_provider(config, postgres_client)

        else:
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'create-provider',
                    'unknown',
                    None,
                    Exception(f'Unknown A2A task provider type: {config.type}')
                )
            )

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('create-provider', 'factory', None, error)
        )

async def create_a2a_task_provider_from_env(
    external_clients: Optional[Dict[str, Any]] = None
) -> A2AResult[A2ATaskProvider]:
    """
    Create A2A task provider from environment variables
    
    Args:
        external_clients: Optional external clients (redis, postgres)
        
    Returns:
        A2AResult containing the task provider or an error
    """
    try:
        # Fall back to regular JAF memory type if A2A-specific one isn't set
        task_memory_type = os.getenv('JAF_A2A_MEMORY_TYPE', os.getenv('JAF_MEMORY_TYPE', 'memory')).lower()
        external_clients = external_clients or {}

        if task_memory_type == 'memory':
            config = A2AInMemoryTaskConfig(
                type='memory',
                key_prefix=os.getenv('JAF_A2A_KEY_PREFIX', 'jaf:a2a:tasks:'),
                default_ttl=int(os.getenv('JAF_A2A_DEFAULT_TTL')) if os.getenv('JAF_A2A_DEFAULT_TTL') else None,
                cleanup_interval=int(os.getenv('JAF_A2A_CLEANUP_INTERVAL', '3600')),
                max_tasks=int(os.getenv('JAF_A2A_MAX_TASKS', '10000')),
                max_tasks_per_context=int(os.getenv('JAF_A2A_MAX_TASKS_PER_CONTEXT', '1000')),
                enable_history=os.getenv('JAF_A2A_ENABLE_HISTORY', 'true').lower() != 'false',
                enable_artifacts=os.getenv('JAF_A2A_ENABLE_ARTIFACTS', 'true').lower() != 'false'
            )
            return create_a2a_success(create_a2a_in_memory_task_provider(config))

        elif task_memory_type == 'redis':
            if not external_clients.get('redis'):
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'create-provider-from-env',
                        'redis',
                        None,
                        Exception('Redis client required for Redis A2A task provider')
                    )
                )

            config = A2ARedisTaskConfig(
                type='redis',
                key_prefix=os.getenv('JAF_A2A_KEY_PREFIX', 'jaf:a2a:tasks:'),
                default_ttl=int(os.getenv('JAF_A2A_DEFAULT_TTL', os.getenv('JAF_REDIS_TTL'))) if os.getenv('JAF_A2A_DEFAULT_TTL', os.getenv('JAF_REDIS_TTL')) else None,
                cleanup_interval=int(os.getenv('JAF_A2A_CLEANUP_INTERVAL', '3600')),
                max_tasks=int(os.getenv('JAF_A2A_MAX_TASKS', '10000')),
                enable_history=os.getenv('JAF_A2A_ENABLE_HISTORY', 'true').lower() != 'false',
                enable_artifacts=os.getenv('JAF_A2A_ENABLE_ARTIFACTS', 'true').lower() != 'false',
                host=os.getenv('JAF_A2A_REDIS_HOST', os.getenv('JAF_REDIS_HOST', 'localhost')),
                port=int(os.getenv('JAF_A2A_REDIS_PORT', os.getenv('JAF_REDIS_PORT', '6379'))),
                password=os.getenv('JAF_A2A_REDIS_PASSWORD', os.getenv('JAF_REDIS_PASSWORD')),
                db=int(os.getenv('JAF_A2A_REDIS_DB', os.getenv('JAF_REDIS_DB', '0')))
            )
            return await create_a2a_redis_task_provider(config, external_clients['redis'])

        elif task_memory_type == 'postgres':
            if not external_clients.get('postgres'):
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'create-provider-from-env',
                        'postgres',
                        None,
                        Exception('PostgreSQL client required for PostgreSQL A2A task provider')
                    )
                )

            config = A2APostgresTaskConfig(
                type='postgres',
                key_prefix=os.getenv('JAF_A2A_KEY_PREFIX', 'jaf:a2a:tasks:'),
                default_ttl=int(os.getenv('JAF_A2A_DEFAULT_TTL')) if os.getenv('JAF_A2A_DEFAULT_TTL') else None,
                cleanup_interval=int(os.getenv('JAF_A2A_CLEANUP_INTERVAL', '3600')),
                max_tasks=int(os.getenv('JAF_A2A_MAX_TASKS', '10000')),
                enable_history=os.getenv('JAF_A2A_ENABLE_HISTORY', 'true').lower() != 'false',
                enable_artifacts=os.getenv('JAF_A2A_ENABLE_ARTIFACTS', 'true').lower() != 'false',
                host=os.getenv('JAF_A2A_POSTGRES_HOST', os.getenv('JAF_POSTGRES_HOST', 'localhost')),
                port=int(os.getenv('JAF_A2A_POSTGRES_PORT', os.getenv('JAF_POSTGRES_PORT', '5432'))),
                database=os.getenv('JAF_A2A_POSTGRES_DB', os.getenv('JAF_POSTGRES_DB', 'jaf_a2a')),
                username=os.getenv('JAF_A2A_POSTGRES_USER', os.getenv('JAF_POSTGRES_USER', 'postgres')),
                password=os.getenv('JAF_A2A_POSTGRES_PASSWORD', os.getenv('JAF_POSTGRES_PASSWORD')),
                ssl=os.getenv('JAF_A2A_POSTGRES_SSL', os.getenv('JAF_POSTGRES_SSL', 'false')).lower() == 'true',
                table_name=os.getenv('JAF_A2A_POSTGRES_TABLE', 'a2a_tasks'),
                max_connections=int(os.getenv('JAF_A2A_POSTGRES_MAX_CONNECTIONS', os.getenv('JAF_POSTGRES_MAX_CONNECTIONS', '10')))
            )
            return await create_a2a_postgres_task_provider(config, external_clients['postgres'])

        else:
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'create-provider-from-env',
                    'unknown',
                    None,
                    Exception(f'Unknown A2A task provider type: {task_memory_type}')
                )
            )

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('create-provider-from-env', 'factory', None, error)
        )

async def create_simple_a2a_task_provider(
    provider_type: str,
    client: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> A2AResult[A2ATaskProvider]:
    """
    Helper function to create A2A task provider with sensible defaults
    
    Args:
        provider_type: Type of provider ('memory', 'redis', 'postgres')
        client: External client for redis/postgres providers
        config: Optional configuration overrides
        
    Returns:
        A2AResult containing the task provider or an error
    """
    try:
        config = config or {}

        if provider_type == 'memory':
            provider_config = A2AInMemoryTaskConfig(
                type='memory',
                key_prefix='jaf:a2a:tasks:',
                default_ttl=None,
                cleanup_interval=3600,
                max_tasks=10000,
                max_tasks_per_context=1000,
                enable_history=True,
                enable_artifacts=True,
                **config
            )
            return create_a2a_success(create_a2a_in_memory_task_provider(provider_config))

        elif provider_type == 'redis':
            if not client:
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'create-simple-provider',
                        'redis',
                        None,
                        Exception('Redis client required for Redis A2A task provider')
                    )
                )

            provider_config = A2ARedisTaskConfig(
                type='redis',
                key_prefix='jaf:a2a:tasks:',
                default_ttl=None,
                cleanup_interval=3600,
                max_tasks=10000,
                enable_history=True,
                enable_artifacts=True,
                host='localhost',
                port=6379,
                password=None,
                db=0,
                **config
            )
            return await create_a2a_redis_task_provider(provider_config, client)

        elif provider_type == 'postgres':
            if not client:
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'create-simple-provider',
                        'postgres',
                        None,
                        Exception('PostgreSQL client required for PostgreSQL A2A task provider')
                    )
                )

            provider_config = A2APostgresTaskConfig(
                type='postgres',
                key_prefix='jaf:a2a:tasks:',
                default_ttl=None,
                cleanup_interval=3600,
                max_tasks=10000,
                enable_history=True,
                enable_artifacts=True,
                host='localhost',
                port=5432,
                database='jaf_a2a',
                username='postgres',
                password=None,
                ssl=False,
                table_name='a2a_tasks',
                max_connections=10,
                **config
            )
            return await create_a2a_postgres_task_provider(provider_config, client)

        else:
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'create-simple-provider',
                    'unknown',
                    None,
                    Exception(f'Unknown A2A task provider type: {provider_type}')
                )
            )

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('create-simple-provider', 'factory', None, error)
        )

def create_composite_a2a_task_provider(
    primary: A2ATaskProvider,
    fallback: Optional[A2ATaskProvider] = None
) -> A2ATaskProvider:
    """
    Create a composite A2A task provider that can use multiple backends
    Useful for implementing failover or read/write splitting
    
    Args:
        primary: Primary task provider
        fallback: Optional fallback provider
        
    Returns:
        Composite A2ATaskProvider
    """
    from .providers.composite import create_composite_a2a_task_provider as create_composite
    return create_composite(primary, fallback)

def validate_a2a_task_provider_config(config: A2ATaskProviderConfig) -> Dict[str, Any]:
    """
    Pure function to validate A2A task provider configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        Dictionary with 'valid' boolean and 'errors' list
    """
    errors = []

    if not config.type:
        errors.append('Provider type is required')
    elif config.type not in ['memory', 'redis', 'postgres']:
        errors.append(f'Invalid provider type: {config.type}')

    if config.max_tasks and config.max_tasks <= 0:
        errors.append('max_tasks must be greater than 0')

    if config.cleanup_interval and config.cleanup_interval <= 0:
        errors.append('cleanup_interval must be greater than 0')

    if config.default_ttl and config.default_ttl <= 0:
        errors.append('default_ttl must be greater than 0')

    # Type-specific validation
    if config.type == 'memory':
        if hasattr(config, 'max_tasks_per_context') and config.max_tasks_per_context <= 0:
            errors.append('max_tasks_per_context must be greater than 0')

    elif config.type == 'redis':
        if config.port and (config.port < 1 or config.port > 65535):
            errors.append('Redis port must be between 1 and 65535')
        if config.db and config.db < 0:
            errors.append('Redis database index must be non-negative')

    elif config.type == 'postgres':
        if config.port and (config.port < 1 or config.port > 65535):
            errors.append('PostgreSQL port must be between 1 and 65535')
        if hasattr(config, 'max_connections') and config.max_connections <= 0:
            errors.append('max_connections must be greater than 0')

    return {
        'valid': len(errors) == 0,
        'errors': errors
    }
