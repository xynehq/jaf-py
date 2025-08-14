"""
A2A In-Memory Task Provider for JAF

This module provides a pure functional in-memory storage implementation for A2A tasks.
It leverages the existing JAF memory patterns while providing A2A-specific optimizations
for task lifecycle management and efficient querying.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ....memory.types import Failure, Success
from ...types import A2ATask, TaskState
from ..serialization import (
    A2ATaskSerialized,
    deserialize_a2a_task,
    sanitize_task,
    serialize_a2a_task,
)
from ..types import (
    A2AInMemoryTaskConfig,
    A2AResult,
    A2ATaskProvider,
    A2ATaskQuery,
    A2ATaskStorage,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_not_found_error,
    create_a2a_task_storage_error,
)


class InMemoryTaskState:
    """Immutable-style state management for in-memory A2A task storage"""

    def __init__(self, config: A2AInMemoryTaskConfig):
        self.config = config
        self.tasks: Dict[str, A2ATaskStorage] = {}
        self.context_index: Dict[str, Set[str]] = defaultdict(set)  # contextId -> Set[taskId]
        self.state_index: Dict[str, Set[str]] = defaultdict(set)    # state -> Set[taskId]
        self.stats = {
            'total_tasks': 0,
            'created_at': datetime.now()
        }
        self.lock = asyncio.Lock()

def create_a2a_in_memory_task_provider(
    config: A2AInMemoryTaskConfig
) -> A2ATaskProvider:
    """
    Create an in-memory A2A task provider
    
    Args:
        config: Configuration for the in-memory provider
        
    Returns:
        A2ATaskProvider implementation
    """
    # Initialize state
    state = InMemoryTaskState(config)

    def _convert_storage_to_serialized(stored: A2ATaskStorage) -> A2ATaskSerialized:
        """Convert storage format to serialized format"""
        import json
        metadata_str = None
        if stored.metadata:
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            metadata_str = json.dumps(stored.metadata, separators=(',', ':'), default=datetime_converter)

        return A2ATaskSerialized(
            task_id=stored.task_id,
            context_id=stored.context_id,
            state=stored.state.value,
            task_data=stored.task_data,
            status_message=stored.status_message,
            created_at=stored.created_at.isoformat(),
            updated_at=stored.updated_at.isoformat(),
            metadata=metadata_str
        )

    def _add_to_indices(task_id: str, context_id: str, task_state: TaskState) -> None:
        """Add task to search indices"""
        state.context_index[context_id].add(task_id)
        state.state_index[task_state.value].add(task_id)

    def _remove_from_indices(task_id: str, context_id: str, task_state: TaskState) -> None:
        """Remove task from search indices"""
        state.context_index[context_id].discard(task_id)
        if not state.context_index[context_id]:
            del state.context_index[context_id]

        state.state_index[task_state.value].discard(task_id)
        if not state.state_index[task_state.value]:
            del state.state_index[task_state.value]

    def _check_storage_limits() -> A2AResult[None]:
        """Check if storage limits would be exceeded"""
        if len(state.tasks) >= config.max_tasks:
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'store',
                    'in-memory',
                    None,
                    Exception(f'Storage limit exceeded: maximum {config.max_tasks} tasks allowed')
                )
            )
        return create_a2a_success(None)

    async def store_task(
        task: A2ATask,
        metadata: Optional[Dict[str, Any]] = None
    ) -> A2AResult[None]:
        """Store a new A2A task"""
        try:
            # Validate and sanitize task
            sanitize_result = sanitize_task(task)
            if isinstance(sanitize_result, Failure):
                return sanitize_result

            # Check storage limits
            limits_result = _check_storage_limits()
            if isinstance(limits_result, Failure):
                return limits_result

            # Serialize task
            serialize_result = serialize_a2a_task(sanitize_result.data, metadata)
            if isinstance(serialize_result, Failure):
                return serialize_result

            serialized_task = serialize_result.data

            # Create storage object
            task_storage = A2ATaskStorage(
                task_id=serialized_task.task_id,
                context_id=serialized_task.context_id,
                state=TaskState(serialized_task.state),
                task_data=serialized_task.task_data,
                status_message=serialized_task.status_message,
                created_at=datetime.fromisoformat(serialized_task.created_at),
                updated_at=datetime.fromisoformat(serialized_task.updated_at),
                expires_at=metadata.get('expires_at') if metadata else None,
                metadata=metadata
            )

            async with state.lock:
                # Store task and update indices
                state.tasks[task.id] = task_storage
                _add_to_indices(task.id, task.context_id, task.status.state)
                state.stats['total_tasks'] = len(state.tasks)

            return create_a2a_success(None)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('store', 'in-memory', task.id, error)
            )

    async def get_task(task_id: str) -> A2AResult[Optional[A2ATask]]:
        """Retrieve a task by ID"""
        try:
            stored = state.tasks.get(task_id)
            if not stored:
                return create_a2a_success(None)

            # Check expiration
            if stored.expires_at and stored.expires_at < datetime.now():
                return create_a2a_success(None)

            # Deserialize task
            serialized = _convert_storage_to_serialized(stored)
            deserialize_result = deserialize_a2a_task(serialized)

            if isinstance(deserialize_result, Success):
                return create_a2a_success(deserialize_result.data)
            else:
                return deserialize_result

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('get', 'in-memory', task_id, error)
            )

    async def update_task(
        task: A2ATask,
        metadata: Optional[Dict[str, Any]] = None
    ) -> A2AResult[None]:
        """Update an existing task"""
        try:
            existing = state.tasks.get(task.id)
            if not existing:
                return create_a2a_failure(
                    create_a2a_task_not_found_error(task.id, 'in-memory')
                )

            # Validate and sanitize task
            sanitize_result = sanitize_task(task)
            if isinstance(sanitize_result, Failure):
                return sanitize_result

            # Merge metadata
            merged_metadata = {**(existing.metadata or {}), **(metadata or {})}

            # Serialize updated task
            serialize_result = serialize_a2a_task(sanitize_result.data, merged_metadata)
            if isinstance(serialize_result, Failure):
                return serialize_result

            serialized_task = serialize_result.data

            # Create updated storage object
            updated_storage = A2ATaskStorage(
                task_id=existing.task_id,
                context_id=existing.context_id,
                state=TaskState(serialized_task.state),
                task_data=serialized_task.task_data,
                status_message=serialized_task.status_message,
                created_at=existing.created_at,
                updated_at=datetime.fromisoformat(serialized_task.updated_at),
                expires_at=existing.expires_at,
                metadata=merged_metadata
            )

            # Update indices if state changed
            if existing.state != task.status.state:
                _remove_from_indices(task.id, task.context_id, existing.state)
                _add_to_indices(task.id, task.context_id, task.status.state)

            # Update storage
            state.tasks[task.id] = updated_storage

            return create_a2a_success(None)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('update', 'in-memory', task.id, error)
            )

    async def update_task_status(
        task_id: str,
        new_state: TaskState,
        status_message: Optional[Any] = None,
        timestamp: Optional[str] = None
    ) -> A2AResult[None]:
        """Update task status only (optimized for frequent status changes)"""
        try:
            existing = state.tasks.get(task_id)
            if not existing:
                return create_a2a_failure(
                    create_a2a_task_not_found_error(task_id, 'in-memory')
                )

            # Deserialize existing task
            serialized = _convert_storage_to_serialized(existing)
            deserialize_result = deserialize_a2a_task(serialized)

            if isinstance(deserialize_result, Failure):
                return deserialize_result

            task = deserialize_result.data

            # Build updated history - add current status message to history
            updated_history = list(task.history or [])
            if task.status.message:
                updated_history.append(task.status.message)

            # Update task status
            from ...types import A2ATaskStatus
            updated_status = A2ATaskStatus(
                state=new_state,
                message=status_message,
                timestamp=timestamp or datetime.now().isoformat()
            )

            updated_task = task.model_copy(update={
                'status': updated_status,
                'history': updated_history
            })

            # Use update_task for the actual update
            return await update_task(updated_task)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('update-status', 'in-memory', task_id, error)
            )

    async def find_tasks(query: A2ATaskQuery) -> A2AResult[List[A2ATask]]:
        """Search tasks by query parameters"""
        try:
            results: List[A2ATask] = []
            for task_id, stored in state.tasks.items():
                # Filter by context
                if query.context_id and stored.context_id != query.context_id:
                    continue

                # Filter by state
                if query.state and stored.state != query.state:
                    continue

                # Filter by task_id
                if query.task_id and task_id != query.task_id:
                    continue
                
                # Check expiration
                if stored.expires_at and stored.expires_at < datetime.now():
                    continue

                # Date filtering - use metadata created_at if available, otherwise use stored created_at
                task_timestamp = stored.created_at
                if stored.metadata and stored.metadata.get('created_at'):
                    try:
                        task_timestamp = datetime.fromisoformat(stored.metadata['created_at'].replace('Z', '+00:00'))
                    except:
                        # Fall back to stored timestamp if parsing fails
                        pass
                
                if query.since and task_timestamp < query.since:
                    continue
                if query.until and task_timestamp > query.until:
                    continue

                # Deserialize task
                serialized = _convert_storage_to_serialized(stored)
                deserialize_result = deserialize_a2a_task(serialized)

                if isinstance(deserialize_result, Success):
                    results.append(deserialize_result.data)
            
            # Sort by timestamp (newest first)
            results.sort(
                key=lambda t: t.status.timestamp or "1970-01-01T00:00:00Z",
                reverse=True
            )

            # Apply pagination
            offset = query.offset or 0
            limit = query.limit or len(results)
            paginated_results = results[offset:offset + limit]

            return create_a2a_success(paginated_results)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('find', 'in-memory', None, error)
            )

    async def get_tasks_by_context(
        context_id: str,
        limit: Optional[int] = None
    ) -> A2AResult[List[A2ATask]]:
        """Get tasks by context ID"""
        return await find_tasks(A2ATaskQuery(context_id=context_id, limit=limit))

    async def delete_task(task_id: str) -> A2AResult[bool]:
        """Delete a task and return True if it existed"""
        try:
            existing = state.tasks.get(task_id)
            if not existing:
                return create_a2a_success(False)

            # Remove from storage and indices
            del state.tasks[task_id]
            _remove_from_indices(task_id, existing.context_id, existing.state)
            state.stats['total_tasks'] = len(state.tasks)

            return create_a2a_success(True)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('delete', 'in-memory', task_id, error)
            )

    async def delete_tasks_by_context(context_id: str) -> A2AResult[int]:
        """Delete tasks by context ID and return count deleted"""
        try:
            async with state.lock:
                context_tasks = state.context_index.get(context_id, set()).copy()
                deleted_count = 0

                for task_id in context_tasks:
                    if task_id in state.tasks:
                        stored_task = state.tasks[task_id]
                        # Remove from storage and indices
                        del state.tasks[task_id]
                        _remove_from_indices(task_id, stored_task.context_id, stored_task.state)
                        deleted_count += 1

                state.stats['total_tasks'] = len(state.tasks)

            return create_a2a_success(deleted_count)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('delete-by-context', 'in-memory', None, error)
            )

    async def cleanup_expired_tasks() -> A2AResult[int]:
        """Clean up expired tasks and return count deleted"""
        try:
            now = datetime.now()
            cleaned_count = 0

            # Find expired tasks
            expired_task_ids = []
            for task_id, stored in state.tasks.items():
                if stored.expires_at and stored.expires_at < now:
                    expired_task_ids.append(task_id)

            # Delete expired tasks
            for task_id in expired_task_ids:
                delete_result = await delete_task(task_id)
                if isinstance(delete_result.data, bool) and delete_result.data:
                    cleaned_count += 1

            return create_a2a_success(cleaned_count)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('cleanup', 'in-memory', None, error)
            )

    async def cleanup_expired_tasks_by_context(context_id: str) -> A2AResult[int]:
        """Clean up expired tasks for a specific context"""
        try:
            now = datetime.now()
            cleaned_count = 0
            
            task_ids = state.context_index.get(context_id, set()).copy()
            expired_task_ids = []
            for task_id in task_ids:
                stored = state.tasks.get(task_id)
                if stored and stored.expires_at and stored.expires_at < now:
                    expired_task_ids.append(task_id)

            for task_id in expired_task_ids:
                delete_result = await delete_task(task_id)
                if isinstance(delete_result.data, bool) and delete_result.data:
                    cleaned_count += 1
            
            return create_a2a_success(cleaned_count)
        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('cleanup-by-context', 'in-memory', None, error)
            )

    async def get_task_stats(context_id: Optional[str] = None) -> A2AResult[Dict[str, Any]]:
        """Get task statistics"""
        try:
            tasks_by_state = defaultdict(int)
            total_tasks = 0
            oldest_task: Optional[datetime] = None
            newest_task: Optional[datetime] = None

            # Determine which tasks to count
            tasks_to_count = set()
            if context_id:
                tasks_to_count = state.context_index.get(context_id, set())
            else:
                tasks_to_count = set(state.tasks.keys())

            for task_id in tasks_to_count:
                stored = state.tasks.get(task_id)
                if not stored:
                    continue

                # Skip expired tasks
                if stored.expires_at and stored.expires_at < datetime.now():
                    continue

                total_tasks += 1
                tasks_by_state[stored.state.value] += 1

                if not oldest_task or stored.created_at < oldest_task:
                    oldest_task = stored.created_at
                if not newest_task or stored.created_at > newest_task:
                    newest_task = stored.created_at
            
            stats = {
                'total_tasks': total_tasks,
                'tasks_by_state': dict(tasks_by_state),
                'oldest_task': oldest_task,
                'newest_task': newest_task
            }
            # Also add individual state counts for backwards compatibility
            for task_state in TaskState:
                stats[task_state.value] = tasks_by_state[task_state.value]

            return create_a2a_success(stats)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('stats', 'in-memory', None, error)
            )

    async def health_check() -> A2AResult[Dict[str, Any]]:
        """Check provider health and return status information"""
        try:
            start_time = datetime.now()

            # Simple health check - verify we can access storage
            task_count = len(state.tasks)
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return create_a2a_success({
                'healthy': True,
                'provider': 'in_memory',
                'latency_ms': latency_ms,
                'task_count': task_count
            })

        except Exception as error:
            return create_a2a_success({
                'healthy': False,
                'error': str(error)
            })

    async def close() -> A2AResult[None]:
        """Close/cleanup the provider"""
        try:
            # Clear all data for cleanup
            state.tasks.clear()
            state.context_index.clear()
            state.state_index.clear()
            state.stats = {'total_tasks': 0, 'created_at': datetime.now()}

            return create_a2a_success(None)

        except Exception as error:
            return create_a2a_failure(
                create_a2a_task_storage_error('close', 'in-memory', None, error)
            )

    # Create a simple class that implements the protocol
    class InMemoryA2ATaskProvider:
        """In-memory implementation of A2ATaskProvider"""

        async def store_task(self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None) -> A2AResult[None]:
            return await store_task(task, metadata)

        async def get_task(self, task_id: str) -> A2AResult[Optional[A2ATask]]:
            return await get_task(task_id)

        async def update_task(self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None) -> A2AResult[None]:
            return await update_task(task, metadata)

        async def update_task_status(self, task_id: str, state: TaskState, status_message: Optional[Any] = None, timestamp: Optional[str] = None) -> A2AResult[None]:
            return await update_task_status(task_id, state, status_message, timestamp)

        async def find_tasks(self, query: A2ATaskQuery) -> A2AResult[List[A2ATask]]:
            return await find_tasks(query)

        async def get_tasks_by_context(self, context_id: str, limit: Optional[int] = None) -> A2AResult[List[A2ATask]]:
            return await get_tasks_by_context(context_id, limit)

        async def delete_task(self, task_id: str) -> A2AResult[bool]:
            return await delete_task(task_id)

        async def delete_tasks_by_context(self, context_id: str) -> A2AResult[int]:
            return await delete_tasks_by_context(context_id)

        async def cleanup_expired_tasks(self) -> A2AResult[int]:
            return await cleanup_expired_tasks()

        async def cleanup_expired_tasks_by_context(self, context_id: str) -> A2AResult[int]:
            return await cleanup_expired_tasks_by_context(context_id)

        async def get_task_stats(self, context_id: Optional[str] = None) -> A2AResult[Dict[str, Any]]:
            return await get_task_stats(context_id)

        async def health_check(self) -> A2AResult[Dict[str, Any]]:
            return await health_check()

        async def close(self) -> A2AResult[None]:
            return await close()

    return InMemoryA2ATaskProvider()
