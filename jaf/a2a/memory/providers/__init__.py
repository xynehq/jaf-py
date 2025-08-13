"""
A2A Task Memory Providers for JAF

This module contains concrete implementations of A2A task storage providers
that leverage the existing JAF memory infrastructure while providing
A2A-specific optimizations and features.
"""

from .composite import create_composite_a2a_task_provider
from .in_memory import create_a2a_in_memory_task_provider
from .postgres import create_a2a_postgres_task_provider
from .redis import create_a2a_redis_task_provider

__all__ = [
    "create_a2a_in_memory_task_provider",
    "create_a2a_postgres_task_provider",
    "create_a2a_redis_task_provider",
    "create_composite_a2a_task_provider",
]
