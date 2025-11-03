"""
A2A Redis Task Provider for JAF

This module provides a Redis-based storage implementation for A2A tasks.
It leverages Redis for high-performance, distributed task storage with features
like automatic expiration, atomic operations, and efficient indexing.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...types import A2ATask, TaskState
from ..serialization import (
    A2ATaskSerialized,
    deserialize_a2a_task,
    sanitize_task,
    serialize_a2a_task,
)
from ..types import (
    A2ARedisTaskConfig,
    A2AResult,
    A2ATaskProvider,
    A2ATaskQuery,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_not_found_error,
    create_a2a_task_storage_error,
)


async def create_a2a_redis_task_provider(
    config: A2ARedisTaskConfig, redis_client: Any
) -> A2AResult[A2ATaskProvider]:
    """
    Create a Redis-based A2A task provider

    Args:
        config: Configuration for the Redis provider
        redis_client: Redis client instance

    Returns:
        A2AResult containing the task provider or an error
    """
    try:
        key_prefix = config.key_prefix or "jaf:a2a:tasks:"

        # Pure functions for key generation
        def get_task_key(task_id: str) -> str:
            return f"{key_prefix}task:{task_id}"

        def get_context_index_key(context_id: str) -> str:
            return f"{key_prefix}context:{context_id}"

        def get_state_index_key(state: str) -> str:
            return f"{key_prefix}state:{state}"

        def get_stats_key() -> str:
            return f"{key_prefix}stats"

        def serialized_task_to_hash(serialized: A2ATaskSerialized) -> Dict[str, str]:
            """Convert serialized task to Redis hash"""
            hash_data = {
                "taskId": serialized.task_id,
                "contextId": serialized.context_id,
                "state": serialized.state,
                "taskData": serialized.task_data,
                "createdAt": serialized.created_at,
                "updatedAt": serialized.updated_at,
            }
            if serialized.status_message:
                hash_data["statusMessage"] = serialized.status_message
            if serialized.metadata:
                hash_data["metadata"] = serialized.metadata
            return hash_data

        def hash_to_serialized_task(hash_data: Dict[str, str]) -> A2ATaskSerialized:
            """Convert Redis hash to serialized task"""
            return A2ATaskSerialized(
                task_id=hash_data["taskId"],
                context_id=hash_data["contextId"],
                state=hash_data["state"],
                task_data=hash_data["taskData"],
                status_message=hash_data.get("statusMessage"),
                created_at=hash_data["createdAt"],
                updated_at=hash_data["updatedAt"],
                metadata=hash_data.get("metadata"),
            )

        class RedisA2ATaskProvider:
            """Redis implementation of A2ATaskProvider"""

            async def store_task(
                self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None
            ) -> A2AResult[None]:
                """Store a new A2A task in Redis"""
                try:
                    # Validate and sanitize task
                    sanitize_result = sanitize_task(task)
                    if not isinstance(sanitize_result.data, A2ATask):
                        return sanitize_result

                    # Serialize task
                    serialize_result = serialize_a2a_task(sanitize_result.data, metadata)
                    if not isinstance(serialize_result.data, A2ATaskSerialized):
                        return serialize_result

                    serialized_task = serialize_result.data
                    task_key = get_task_key(task.id)
                    context_index_key = get_context_index_key(task.context_id)
                    state_index_key = get_state_index_key(task.status.state.value)

                    # Use Redis pipeline for atomicity
                    async with redis_client.pipeline() as pipe:
                        # Store task data as hash
                        task_hash = serialized_task_to_hash(serialized_task)
                        await pipe.hset(task_key, mapping=task_hash)

                        # Set TTL if specified
                        if metadata and metadata.get("expires_at"):
                            expires_at = metadata["expires_at"]
                            if isinstance(expires_at, datetime):
                                ttl_seconds = int((expires_at - datetime.now()).total_seconds())
                                if ttl_seconds > 0:
                                    await pipe.expire(task_key, ttl_seconds)
                        elif config.default_ttl:
                            await pipe.expire(task_key, config.default_ttl)

                        # Add to indices
                        await pipe.sadd(context_index_key, task.id)
                        await pipe.sadd(state_index_key, task.id)

                        # Update stats
                        await pipe.hincrby(get_stats_key(), "totalTasks", 1)
                        await pipe.hincrby(get_stats_key(), f"state:{task.status.state.value}", 1)

                        await pipe.execute()

                    return create_a2a_success(None)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("store", "redis", task.id, error)
                    )

            async def get_task(self, task_id: str) -> A2AResult[Optional[A2ATask]]:
                """Retrieve a task by ID from Redis"""
                try:
                    task_key = get_task_key(task_id)
                    exists = await redis_client.exists(task_key)

                    if not exists:
                        return create_a2a_success(None)

                    hash_data = await redis_client.hgetall(task_key)
                    if not hash_data or "taskData" not in hash_data:
                        return create_a2a_success(None)

                    # Convert bytes to strings if needed (depends on Redis client)
                    if isinstance(hash_data, dict):
                        hash_data = {
                            k.decode() if isinstance(k, bytes) else k: v.decode()
                            if isinstance(v, bytes)
                            else v
                            for k, v in hash_data.items()
                        }

                    serialized = hash_to_serialized_task(hash_data)
                    deserialize_result = deserialize_a2a_task(serialized)

                    if isinstance(deserialize_result.data, A2ATask):
                        return create_a2a_success(deserialize_result.data)
                    else:
                        return deserialize_result

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("get", "redis", task_id, error)
                    )

            async def update_task(
                self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None
            ) -> A2AResult[None]:
                """Update an existing task in Redis"""
                try:
                    task_key = get_task_key(task.id)
                    exists = await redis_client.exists(task_key)

                    if not exists:
                        return create_a2a_failure(create_a2a_task_not_found_error(task.id, "redis"))

                    # Get existing task to check for state changes
                    existing_hash = await redis_client.hgetall(task_key)
                    if isinstance(existing_hash, dict):
                        existing_hash = {
                            k.decode() if isinstance(k, bytes) else k: v.decode()
                            if isinstance(v, bytes)
                            else v
                            for k, v in existing_hash.items()
                        }

                    old_state = existing_hash.get("state")

                    # Validate and sanitize task
                    sanitize_result = sanitize_task(task)
                    if not isinstance(sanitize_result.data, A2ATask):
                        return sanitize_result

                    # Merge metadata
                    existing_metadata = {}
                    if existing_hash.get("metadata"):
                        try:
                            existing_metadata = json.loads(existing_hash["metadata"])
                        except:
                            pass
                    merged_metadata = {**existing_metadata, **(metadata or {})}

                    # Serialize updated task
                    serialize_result = serialize_a2a_task(sanitize_result.data, merged_metadata)
                    if not isinstance(serialize_result.data, A2ATaskSerialized):
                        return serialize_result

                    serialized_task = serialize_result.data

                    async with redis_client.pipeline() as pipe:
                        # Update task data
                        task_hash = serialized_task_to_hash(serialized_task)
                        await pipe.hset(task_key, mapping=task_hash)

                        # Update indices if state changed
                        if old_state and old_state != task.status.state.value:
                            old_state_index_key = get_state_index_key(old_state)
                            new_state_index_key = get_state_index_key(task.status.state.value)

                            await pipe.srem(old_state_index_key, task.id)
                            await pipe.sadd(new_state_index_key, task.id)

                            # Update stats
                            await pipe.hincrby(get_stats_key(), f"state:{old_state}", -1)
                            await pipe.hincrby(
                                get_stats_key(), f"state:{task.status.state.value}", 1
                            )

                        await pipe.execute()

                    return create_a2a_success(None)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("update", "redis", task.id, error)
                    )

            async def update_task_status(
                self,
                task_id: str,
                state: TaskState,
                status_message: Optional[Any] = None,
                timestamp: Optional[str] = None,
            ) -> A2AResult[None]:
                """Update task status only"""
                try:
                    # Get existing task first
                    get_result = await self.get_task(task_id)
                    if not isinstance(get_result.data, A2ATask):
                        return get_result

                    if not get_result.data:
                        return create_a2a_failure(create_a2a_task_not_found_error(task_id, "redis"))

                    task = get_result.data

                    # Build updated history - add current status message to history before updating
                    updated_history = list(task.history or [])
                    if task.status.message:
                        updated_history.append(task.status.message)

                    # Update task status
                    from ...types import A2ATaskStatus

                    updated_status = A2ATaskStatus(
                        state=state,
                        message=status_message or task.status.message,
                        timestamp=timestamp or datetime.now().isoformat(),
                    )

                    updated_task = task.model_copy(
                        update={"status": updated_status, "history": updated_history}
                    )

                    # Use update_task for the actual update
                    return await self.update_task(updated_task)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("update-status", "redis", task_id, error)
                    )

            async def find_tasks(self, query: A2ATaskQuery) -> A2AResult[List[A2ATask]]:
                """Search tasks by query parameters"""
                try:
                    task_ids: List[str] = []

                    # Determine which sets to use for filtering
                    keys_to_intersect = []
                    if query.context_id:
                        keys_to_intersect.append(get_context_index_key(query.context_id))
                    if query.state:
                        keys_to_intersect.append(get_state_index_key(query.state.value))

                    if keys_to_intersect:
                        # If we have sets to intersect, use SINTER for efficiency
                        if len(keys_to_intersect) > 1:
                            task_ids = await redis_client.sinter(keys_to_intersect)
                        else:
                            task_ids = await redis_client.smembers(keys_to_intersect[0])
                    else:
                        # Get all task keys if no context or state is provided
                        pattern = f"{key_prefix}task:*"
                        keys = await redis_client.keys(pattern)
                        task_ids = [key.replace(f"{key_prefix}task:", "") for key in keys]

                    # Convert bytes to strings if needed
                    task_ids = [tid.decode() if isinstance(tid, bytes) else tid for tid in task_ids]

                    # Filter by specific task ID if provided
                    if query.task_id:
                        task_ids = [tid for tid in task_ids if tid == query.task_id]

                    # Fetch tasks and apply additional filters
                    results: List[A2ATask] = []

                    for task_id in task_ids:
                        task_result = await self.get_task(task_id)
                        if not isinstance(task_result.data, A2ATask):
                            continue

                        task = task_result.data

                        # Apply date filters
                        if query.since or query.until:
                            # Get task data to check timestamp
                            task_key = get_task_key(task_id)
                            hash_data = await redis_client.hgetall(task_key)
                            if hash_data:
                                # Convert bytes to strings if needed
                                if isinstance(hash_data, dict):
                                    hash_data = {
                                        k.decode() if isinstance(k, bytes) else k: v.decode()
                                        if isinstance(v, bytes)
                                        else v
                                        for k, v in hash_data.items()
                                    }

                                # Try to get timestamp from metadata first, then created_at
                                task_timestamp = None
                                if hash_data.get("metadata"):
                                    try:
                                        metadata = json.loads(hash_data["metadata"])
                                        if metadata.get("created_at"):
                                            task_timestamp = datetime.fromisoformat(
                                                metadata["created_at"].replace("Z", "+00:00")
                                            )
                                    except:
                                        pass

                                # Fall back to createdAt field
                                if not task_timestamp and hash_data.get("createdAt"):
                                    try:
                                        task_timestamp = datetime.fromisoformat(
                                            hash_data["createdAt"].replace("Z", "+00:00")
                                        )
                                    except:
                                        pass

                                # Apply time filters
                                if task_timestamp:
                                    if query.since and task_timestamp < query.since:
                                        continue
                                    if query.until and task_timestamp > query.until:
                                        continue

                        results.append(task)

                    # Sort by timestamp (newest first)
                    results.sort(
                        key=lambda t: t.status.timestamp or "1970-01-01T00:00:00Z", reverse=True
                    )

                    # Apply pagination
                    offset = query.offset or 0
                    limit = query.limit or len(results)
                    paginated_results = results[offset : offset + limit]

                    return create_a2a_success(paginated_results)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("find", "redis", None, error)
                    )

            async def get_tasks_by_context(
                self, context_id: str, limit: Optional[int] = None
            ) -> A2AResult[List[A2ATask]]:
                """Get tasks by context ID"""
                return await self.find_tasks(A2ATaskQuery(context_id=context_id, limit=limit))

            async def delete_task(self, task_id: str) -> A2AResult[bool]:
                """Delete a task and return True if it existed"""
                try:
                    task_key = get_task_key(task_id)

                    # Get task data for index cleanup
                    hash_data = await redis_client.hgetall(task_key)
                    if not hash_data:
                        return create_a2a_success(False)

                    # Convert bytes to strings if needed
                    if isinstance(hash_data, dict):
                        hash_data = {
                            k.decode() if isinstance(k, bytes) else k: v.decode()
                            if isinstance(v, bytes)
                            else v
                            for k, v in hash_data.items()
                        }

                    context_id = hash_data.get("contextId")
                    state = hash_data.get("state")

                    async with redis_client.pipeline() as pipe:
                        # Delete task
                        await pipe.delete(task_key)

                        # Remove from indices
                        if context_id:
                            context_index_key = get_context_index_key(context_id)
                            await pipe.srem(context_index_key, task_id)

                        if state:
                            state_index_key = get_state_index_key(state)
                            await pipe.srem(state_index_key, task_id)

                        # Update stats
                        await pipe.hincrby(get_stats_key(), "totalTasks", -1)
                        if state:
                            await pipe.hincrby(get_stats_key(), f"state:{state}", -1)

                        await pipe.execute()

                    return create_a2a_success(True)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("delete", "redis", task_id, error)
                    )

            async def delete_tasks_by_context(self, context_id: str) -> A2AResult[int]:
                """Delete tasks by context ID and return count deleted"""
                try:
                    context_index_key = get_context_index_key(context_id)
                    task_ids = await redis_client.smembers(context_index_key)

                    if not task_ids:
                        return create_a2a_success(0)

                    # Convert bytes to strings if needed
                    if task_ids and isinstance(list(task_ids)[0], bytes):
                        task_ids = [tid.decode() for tid in task_ids]

                    deleted_count = 0
                    for task_id in task_ids:
                        delete_result = await self.delete_task(task_id)
                        if (
                            hasattr(delete_result, "data")
                            and isinstance(delete_result.data, bool)
                            and delete_result.data
                        ):
                            deleted_count += 1
                        elif hasattr(delete_result, "error"):
                            # Log error but continue with other deletions
                            continue

                    return create_a2a_success(deleted_count)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("delete-by-context", "redis", None, error)
                    )

            async def cleanup_expired_tasks(self) -> A2AResult[int]:
                """Clean up expired tasks and return count deleted"""
                try:
                    # Redis automatically handles TTL expiration
                    # This is a placeholder for consistency
                    return create_a2a_success(0)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("cleanup", "redis", None, error)
                    )

            async def get_task_stats(
                self, context_id: Optional[str] = None
            ) -> A2AResult[Dict[str, Any]]:
                """Get task statistics"""
                try:
                    if context_id:
                        # Get stats for specific context (simplified implementation)
                        tasks_result = await self.get_tasks_by_context(context_id)
                        if not isinstance(tasks_result.data, list):
                            return tasks_result

                        tasks = tasks_result.data
                        tasks_by_state = {state.value: 0 for state in TaskState}

                        for task in tasks:
                            tasks_by_state[task.status.state.value] += 1

                        stats = {
                            "total_tasks": len(tasks),
                            "tasks_by_state": tasks_by_state,
                            "oldest_task": None,
                            "newest_task": None,
                        }
                        # Also add individual state counts for backwards compatibility
                        for state in TaskState:
                            stats[state.value] = tasks_by_state[state.value]

                        return create_a2a_success(stats)
                    else:
                        # Get global stats from Redis hash
                        stats_key = get_stats_key()
                        stats = await redis_client.hgetall(stats_key)

                        if isinstance(stats, dict):
                            stats = {
                                k.decode() if isinstance(k, bytes) else k: v.decode()
                                if isinstance(v, bytes)
                                else v
                                for k, v in stats.items()
                            }

                        total_tasks = int(stats.get("totalTasks", 0))

                        # Build tasks_by_state dict
                        tasks_by_state = {}
                        for state in TaskState:
                            tasks_by_state[state.value] = int(stats.get(f"state:{state.value}", 0))

                        # Build stats dict with individual state counts
                        result_stats = {
                            "total_tasks": total_tasks,
                            "tasks_by_state": tasks_by_state,
                            "oldest_task": None,
                            "newest_task": None,
                        }

                        # Also add individual state counts for backwards compatibility
                        for state in TaskState:
                            result_stats[state.value] = tasks_by_state[state.value]

                        return create_a2a_success(result_stats)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("stats", "redis", None, error)
                    )

            async def health_check(self) -> A2AResult[Dict[str, Any]]:
                """Check provider health and return status information"""
                try:
                    start_time = datetime.now()

                    # Simple ping to Redis
                    await redis_client.ping()

                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return create_a2a_success(
                        {"healthy": True, "provider": "redis", "latency_ms": latency_ms}
                    )

                except Exception as error:
                    return create_a2a_success({"healthy": False, "error": str(error)})

            async def close(self) -> A2AResult[None]:
                """Close/cleanup the provider"""
                try:
                    # Redis client cleanup is typically handled externally
                    # We don't close the client here as it might be shared
                    return create_a2a_success(None)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("close", "redis", None, error)
                    )

        return create_a2a_success(RedisA2ATaskProvider())

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error("create-redis-provider", "redis", None, error)
        )
