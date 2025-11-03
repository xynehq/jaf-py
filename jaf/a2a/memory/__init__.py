"""
A2A Memory Module for JAF Python

This module provides A2A task persistence functionality that extends the core JAF memory system.
It includes task providers, serialization utilities, and factory functions for creating
A2A-specific storage backends while leveraging the existing memory infrastructure.
"""

from .cleanup import (
    A2ATaskCleanupConfig,
    CleanupResult,
    create_cleanup_config_from_env,
    create_task_cleanup_scheduler,
    default_cleanup_config,
    perform_task_cleanup,
    validate_cleanup_config,
)
from .factory import (
    create_a2a_task_provider,
    create_a2a_task_provider_from_env,
    create_composite_a2a_task_provider,
    create_simple_a2a_task_provider,
    validate_a2a_task_provider_config,
)
from .serialization import (
    A2ATaskSerialized,
    clone_task,
    create_task_index,
    deserialize_a2a_task,
    extract_task_search_text,
    sanitize_task,
    serialize_a2a_task,
    validate_task_integrity,
)
from .types import (
    A2AInMemoryTaskConfig,
    A2APostgresTaskConfig,
    A2ARedisTaskConfig,
    A2AResult,
    # Error types
    A2ATaskError,
    A2ATaskErrorUnion,
    A2ATaskMemoryConfig,
    A2ATaskNotFoundError,
    A2ATaskProvider,
    A2ATaskProviderConfig,
    # Core types
    A2ATaskQuery,
    A2ATaskStorage,
    A2ATaskStorageError,
    create_a2a_failure,
    create_a2a_success,
    # Factory functions
    create_a2a_task_error,
    create_a2a_task_not_found_error,
    create_a2a_task_storage_error,
    # Type checking functions
    is_a2a_task_error,
    is_a2a_task_not_found_error,
    is_a2a_task_storage_error,
)

__all__ = [
    # Types
    "A2ATaskQuery",
    "A2ATaskStorage",
    "A2ATaskProvider",
    "A2ATaskMemoryConfig",
    "A2AInMemoryTaskConfig",
    "A2ARedisTaskConfig",
    "A2APostgresTaskConfig",
    "A2ATaskProviderConfig",
    "A2ATaskError",
    "A2ATaskNotFoundError",
    "A2ATaskStorageError",
    "A2ATaskErrorUnion",
    "A2AResult",
    "A2ATaskSerialized",
    "A2ATaskCleanupConfig",
    "CleanupResult",
    # Factory functions
    "create_a2a_task_error",
    "create_a2a_task_not_found_error",
    "create_a2a_task_storage_error",
    "create_a2a_success",
    "create_a2a_failure",
    "create_a2a_task_provider",
    "create_a2a_task_provider_from_env",
    "create_simple_a2a_task_provider",
    "create_composite_a2a_task_provider",
    # Serialization functions
    "serialize_a2a_task",
    "deserialize_a2a_task",
    "create_task_index",
    "extract_task_search_text",
    "validate_task_integrity",
    "clone_task",
    "sanitize_task",
    # Cleanup functions
    "default_cleanup_config",
    "perform_task_cleanup",
    "create_task_cleanup_scheduler",
    "validate_cleanup_config",
    "create_cleanup_config_from_env",
    # Validation functions
    "validate_a2a_task_provider_config",
    "is_a2a_task_error",
    "is_a2a_task_not_found_error",
    "is_a2a_task_storage_error",
]
