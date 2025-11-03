"""
A2A Composite Task Provider for JAF

This module provides a composite A2A task provider that can use multiple backends.
It's useful for implementing failover, read/write splitting, and high availability
patterns while maintaining the same interface.
"""

from typing import Any, Dict, List, Optional

from ....memory.types import Failure
from ...types import A2ATask, TaskState
from ..types import (
    A2AResult,
    A2ATaskProvider,
    A2ATaskQuery,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_storage_error,
)


def create_composite_a2a_task_provider(
    primary: A2ATaskProvider, fallback: Optional[A2ATaskProvider] = None
) -> A2ATaskProvider:
    """
    Create a composite A2A task provider that can use multiple backends

    Args:
        primary: Primary task provider
        fallback: Optional fallback provider for failover

    Returns:
        Composite A2ATaskProvider
    """

    class CompositeA2ATaskProvider:
        """Composite implementation of A2ATaskProvider"""

        async def store_task(
            self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None
        ) -> A2AResult[None]:
            """Store task in primary, fallback to secondary if primary fails"""
            result = await primary.store_task(task, metadata)
            if isinstance(result, Failure) and fallback:
                return await fallback.store_task(task, metadata)
            return result

        async def get_task(self, task_id: str) -> A2AResult[Optional[A2ATask]]:
            """Get task from primary, fallback to secondary if primary fails"""
            result = await primary.get_task(task_id)
            if isinstance(result, Failure) and fallback:
                return await fallback.get_task(task_id)
            return result

        async def update_task(
            self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None
        ) -> A2AResult[None]:
            """Update task in primary, fallback to secondary if primary fails"""
            result = await primary.update_task(task, metadata)
            if isinstance(result, Failure) and fallback:
                return await fallback.update_task(task, metadata)
            return result

        async def update_task_status(
            self,
            task_id: str,
            state: TaskState,
            status_message: Optional[Any] = None,
            timestamp: Optional[str] = None,
        ) -> A2AResult[None]:
            """Update task status in primary, fallback to secondary if primary fails"""
            result = await primary.update_task_status(task_id, state, status_message, timestamp)
            if isinstance(result, Failure) and fallback:
                return await fallback.update_task_status(task_id, state, status_message, timestamp)
            return result

        async def find_tasks(self, query: A2ATaskQuery) -> A2AResult[List[A2ATask]]:
            """Find tasks in primary, fallback to secondary if primary fails"""
            result = await primary.find_tasks(query)
            if isinstance(result, Failure) and fallback:
                return await fallback.find_tasks(query)
            return result

        async def get_tasks_by_context(
            self, context_id: str, limit: Optional[int] = None
        ) -> A2AResult[List[A2ATask]]:
            """Get tasks by context from primary, fallback to secondary if primary fails"""
            result = await primary.get_tasks_by_context(context_id, limit)
            if isinstance(result, Failure) and fallback:
                return await fallback.get_tasks_by_context(context_id, limit)
            return result

        async def delete_task(self, task_id: str) -> A2AResult[bool]:
            """Delete task from both providers (for consistency)"""
            primary_result = await primary.delete_task(task_id)

            # For delete operations, try both providers regardless of success
            if fallback:
                await fallback.delete_task(task_id)

            return primary_result

        async def delete_tasks_by_context(self, context_id: str) -> A2AResult[int]:
            """Delete tasks by context from both providers"""
            primary_result = await primary.delete_tasks_by_context(context_id)
            fallback_count = 0

            # For delete operations, try both providers regardless of success
            if fallback:
                fallback_result = await fallback.delete_tasks_by_context(context_id)
                if isinstance(fallback_result.data, int):
                    fallback_count = fallback_result.data

            # Return combined count if primary succeeded
            if isinstance(primary_result.data, int):
                total_count = primary_result.data + fallback_count
                return create_a2a_success(total_count)

            return primary_result

        async def cleanup_expired_tasks(self) -> A2AResult[int]:
            """Cleanup expired tasks from both providers"""
            primary_result = await primary.cleanup_expired_tasks()
            fallback_count = 0

            if fallback:
                fallback_result = await fallback.cleanup_expired_tasks()
                if isinstance(fallback_result.data, int):
                    fallback_count = fallback_result.data

            # Return combined count
            if isinstance(primary_result.data, int):
                total_count = primary_result.data + fallback_count
                return create_a2a_success(total_count)

            return primary_result

        async def get_task_stats(
            self, context_id: Optional[str] = None
        ) -> A2AResult[Dict[str, Any]]:
            """Get task stats from primary, fallback to secondary if primary fails"""
            result = await primary.get_task_stats(context_id)
            if isinstance(result, Failure) and fallback:
                return await fallback.get_task_stats(context_id)
            return result

        async def health_check(self) -> A2AResult[Dict[str, Any]]:
            """Check health of both providers"""
            primary_health = await primary.health_check()
            fallback_health = None

            if fallback:
                fallback_health = await fallback.health_check()

            # Determine overall health
            primary_healthy = isinstance(primary_health.data, dict) and primary_health.data.get(
                "healthy", False
            )
            fallback_healthy = fallback_health is None or (
                isinstance(fallback_health.data, dict)
                and fallback_health.data.get("healthy", False)
            )

            overall_healthy = primary_healthy or fallback_healthy

            health_data = {
                "healthy": overall_healthy,
                "primary": primary_health.data
                if isinstance(primary_health.data, dict)
                else {"healthy": False},
                "fallback": fallback_health.data
                if fallback_health and isinstance(fallback_health.data, dict)
                else None,
            }

            return create_a2a_success(health_data)

        async def close(self) -> A2AResult[None]:
            """Close both providers"""
            try:
                # Close primary
                primary_result = await primary.close()

                # Close fallback
                if fallback:
                    await fallback.close()

                return primary_result

            except Exception as error:
                return create_a2a_failure(
                    create_a2a_task_storage_error("close", "composite", None, error)
                )

    return CompositeA2ATaskProvider()
