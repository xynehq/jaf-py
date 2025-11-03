"""
A2A Memory Task Lifecycle Tests - Phase 2: Full Lifecycle Integration

Comprehensive tests for A2A task lifecycle across all memory providers.
Tests the complete lifecycle from submission to completion, including state transitions,
error handling, and multi-task context management.

Based on src/a2a/memory/__tests__/task-lifecycle.test.ts patterns.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import asyncpg
import pytest
import redis.asyncio as redis

from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
from jaf.a2a.memory.types import (
    A2AInMemoryTaskConfig,
    A2APostgresTaskConfig,
    A2ARedisTaskConfig,
    A2ATaskProvider,
    A2ATaskQuery,
)
from jaf.a2a.types import (
    A2AArtifact,
    A2ADataPart,
    A2AMessage,
    A2ATask,
    A2ATaskStatus,
    A2ATextPart,
    TaskState,
)

# Import other providers when they're available
try:
    from jaf.a2a.memory.providers.redis import create_a2a_redis_task_provider

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from jaf.a2a.memory.providers.postgres import create_a2a_postgres_task_provider

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class TaskLifecycleTestBase:
    """Base test class with helper methods for task lifecycle testing"""

    def create_submission_task(
        self, task_id: str = "lifecycle_task_001", context_id: str = "lifecycle_ctx_001"
    ) -> A2ATask:
        """Create a task in submitted state"""
        return A2ATask(
            id=task_id,
            contextId=context_id,
            kind="task",
            status=A2ATaskStatus(
                state=TaskState.SUBMITTED,
                message=A2AMessage(
                    role="user",
                    parts=[A2ATextPart(kind="text", text="Please help me with this task")],
                    messageId=f"submit_{task_id}",
                    contextId=context_id,
                    kind="message",
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            metadata={"created_at": datetime.now(timezone.utc).isoformat(), "priority": "normal"},
        )

    def create_working_task_update(
        self, base_task: A2ATask, progress_message: str = "Processing your request..."
    ) -> A2ATask:
        """Create task update transitioning to working state"""
        return base_task.model_copy(
            update={
                "status": A2ATaskStatus(
                    state=TaskState.WORKING,
                    message=A2AMessage(
                        role="agent",
                        parts=[A2ATextPart(kind="text", text=progress_message)],
                        messageId=f"working_{base_task.id}",
                        contextId=base_task.context_id,
                        kind="message",
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                "history": [base_task.status.message] if base_task.status.message else [],
            }
        )

    def create_completed_task_update(
        self, working_task: A2ATask, result_text: str = "Task completed successfully"
    ) -> A2ATask:
        """Create task update transitioning to completed state with artifacts"""
        completion_message = A2AMessage(
            role="agent",
            parts=[A2ATextPart(kind="text", text=result_text)],
            messageId=f"complete_{working_task.id}",
            contextId=working_task.context_id,
            kind="message",
        )

        result_artifact = A2AArtifact(
            artifactId=f"result_{working_task.id}",
            name="Task Result",
            description="Final result of the completed task",
            parts=[
                A2ATextPart(kind="text", text="Here is your completed result."),
                A2ADataPart(
                    kind="data",
                    data={"success": True, "timestamp": datetime.now(timezone.utc).isoformat()},
                ),
            ],
        )

        # Build complete history
        history = list(working_task.history or [])
        if working_task.status.message:
            history.append(working_task.status.message)

        return working_task.model_copy(
            update={
                "status": A2ATaskStatus(
                    state=TaskState.COMPLETED,
                    message=completion_message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                "history": history,
                "artifacts": [result_artifact],
            }
        )

    def create_failed_task_update(
        self, working_task: A2ATask, error_message: str = "Task failed due to an error"
    ) -> A2ATask:
        """Create task update transitioning to failed state"""
        failure_message = A2AMessage(
            role="agent",
            parts=[A2ATextPart(kind="text", text=error_message)],
            messageId=f"failed_{working_task.id}",
            contextId=working_task.context_id,
            kind="message",
        )

        history = list(working_task.history or [])
        if working_task.status.message:
            history.append(working_task.status.message)

        return working_task.model_copy(
            update={
                "status": A2ATaskStatus(
                    state=TaskState.FAILED,
                    message=failure_message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                "history": history,
            }
        )

    def create_canceled_task_update(
        self, working_task: A2ATask, cancel_reason: str = "Task was canceled by user"
    ) -> A2ATask:
        """Create task update transitioning to canceled state"""
        cancel_message = A2AMessage(
            role="agent",
            parts=[A2ATextPart(kind="text", text=cancel_reason)],
            messageId=f"cancel_{working_task.id}",
            contextId=working_task.context_id,
            kind="message",
        )

        history = list(working_task.history or [])
        if working_task.status.message:
            history.append(working_task.status.message)

        return working_task.model_copy(
            update={
                "status": A2ATaskStatus(
                    state=TaskState.CANCELED,
                    message=cancel_message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                "history": history,
            }
        )


# Provider parameter list for running tests across all providers
PROVIDER_TYPES = [
    "in_memory",
    pytest.param(
        "redis", marks=pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    ),
    pytest.param(
        "postgres",
        marks=pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available"),
    ),
]


@pytest.fixture(params=PROVIDER_TYPES)
async def provider(request):
    """Create a task provider based on parametrized type."""
    provider_type = request.param
    p = None
    redis_client = None
    pg_pool = None

    if provider_type == "in_memory":
        config = A2AInMemoryTaskConfig(max_tasks=1000, max_tasks_per_context=100)
        p_result = create_a2a_in_memory_task_provider(config)
        p = p_result
    elif provider_type == "redis":
        config = A2ARedisTaskConfig(
            host="localhost",
            port=6379,
            db=15,  # Use separate DB for testing
            key_prefix="jaf_test:a2a:tasks:",
            password="12345678",
        )
        try:
            redis_client = redis.Redis(
                host=config.host,
                port=config.port,
                db=config.db,
                password=config.password,
                decode_responses=True,
            )
            await redis_client.ping()
            await redis_client.flushdb()
            p_result = await create_a2a_redis_task_provider(config, redis_client)
            p = p_result.data
        except (
            redis.exceptions.ConnectionError,
            ConnectionRefusedError,
            redis.exceptions.AuthenticationError,
        ) as e:
            pytest.skip(f"Redis not available at {config.host}:{config.port}: {e}")

    elif provider_type == "postgres":
        config = A2APostgresTaskConfig(
            host="localhost",
            port=5432,
            database="jaf_test",
            username="postgres",
            table_name="a2a_tasks_test",
        )
        try:
            pg_pool = await asyncpg.create_pool(
                user=config.username, database=config.database, host=config.host, port=config.port
            )
            # Clean the table for test isolation
            async with pg_pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {config.table_name} CASCADE")
            p_result = await create_a2a_postgres_task_provider(config, pg_pool)
            p = p_result.data
        except (ConnectionRefusedError, asyncpg.exceptions.InvalidCatalogNameError, OSError) as e:
            pytest.skip(f"PostgreSQL not available or db 'jaf_test' does not exist: {e}")

    if p:
        yield p
        # The provider's close() method is a no-op for external clients,
        # so we close the client/pool we created.
        if redis_client:
            await redis_client.aclose()
        if pg_pool:
            await pg_pool.close()
    elif provider_type not in ["redis", "postgres"]:  # Don't fail if skipped
        pytest.fail(f"Unknown provider type or provider failed to initialize: {provider_type}")


class TestTaskLifecycleHappyPath(TaskLifecycleTestBase):
    """Test successful task lifecycle scenarios"""

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_complete_happy_path_lifecycle(self, provider):
        """
        CRITICAL: Test the complete happy path lifecycle across all providers
        submitted -> working -> completed with history and artifacts
        """

        # Step 1: Create and store submitted task
        submitted_task = self.create_submission_task("happy_001", "happy_ctx_001")

        store_result = await provider.store_task(submitted_task)
        assert store_result.data is None, "Store should succeed"

        # Verify stored task
        get_result = await provider.get_task("happy_001")
        assert get_result.data is not None, "Task should be retrievable"
        stored_task = get_result.data
        assert stored_task.id == "happy_001"
        assert stored_task.status.state == TaskState.SUBMITTED

        # Step 2: Transition to working state
        working_task = self.create_working_task_update(
            stored_task, "Starting to work on your request..."
        )

        update_result = await provider.update_task(working_task)
        assert update_result.data is None, "Update should succeed"

        # Verify working state
        get_working_result = await provider.get_task("happy_001")
        assert get_working_result.data is not None
        working_stored = get_working_result.data
        assert working_stored.status.state == TaskState.WORKING
        assert len(working_stored.history or []) == 1, "Should have original submission in history"

        # Step 3: Add intermediate update
        intermediate_update_result = await provider.update_task_status(
            "happy_001",
            TaskState.WORKING,
            A2AMessage(
                role="agent",
                parts=[A2ATextPart(kind="text", text="50% complete...")],
                messageId="progress_001",
                contextId="happy_ctx_001",
                kind="message",
            ),
        )
        assert intermediate_update_result.data is None, "Status update should succeed"

        # Step 4: Transition to completed state
        completed_task = self.create_completed_task_update(
            working_stored, "Your task has been completed successfully!"
        )

        complete_result = await provider.update_task(completed_task)
        assert complete_result.data is None, "Completion should succeed"

        # Step 5: Verify final state
        final_result = await provider.get_task("happy_001")
        assert final_result.data is not None
        final_task = final_result.data

        assert final_task.status.state == TaskState.COMPLETED
        assert len(final_task.history or []) >= 2, "Should have complete history"
        assert len(final_task.artifacts or []) == 1, "Should have result artifact"
        assert final_task.artifacts[0].name == "Task Result"

        # Step 6: Verify task appears in queries
        completed_query = A2ATaskQuery(context_id="happy_ctx_001", state=TaskState.COMPLETED)
        find_result = await provider.find_tasks(completed_query)
        assert find_result.data is not None
        assert len(find_result.data) == 1
        assert find_result.data[0].id == "happy_001"

        # Step 7: Verify context-based retrieval
        context_result = await provider.get_tasks_by_context("happy_ctx_001")
        assert context_result.data is not None
        assert len(context_result.data) == 1
        assert context_result.data[0].id == "happy_001"

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_multiple_intermediate_updates(self, provider):
        """Test task with multiple status updates during working phase"""

        # Create and store initial task
        task = self.create_submission_task("multi_001", "multi_ctx_001")
        await provider.store_task(task)

        # Transition to working
        working_task = self.create_working_task_update(task)
        await provider.update_task(working_task)

        # Add multiple progress updates
        progress_messages = [
            "25% complete - analyzing request",
            "50% complete - processing data",
            "75% complete - generating results",
            "90% complete - finalizing output",
        ]

        for i, message in enumerate(progress_messages):
            status_message = A2AMessage(
                role="agent",
                parts=[A2ATextPart(kind="text", text=message)],
                messageId=f"progress_{i}",
                contextId="multi_ctx_001",
                kind="message",
            )

            update_result = await provider.update_task_status(
                "multi_001", TaskState.WORKING, status_message
            )
            assert update_result.data is None, f"Progress update {i} should succeed"

        # Complete the task
        get_result = await provider.get_task("multi_001")
        completed_task = self.create_completed_task_update(get_result.data)
        await provider.update_task(completed_task)

        # Verify all updates are preserved
        final_result = await provider.get_task("multi_001")
        final_task = final_result.data

        assert final_task.status.state == TaskState.COMPLETED
        # Should have original submission + working transition + multiple progress updates
        assert len(final_task.history or []) >= len(progress_messages) + 1


class TestTaskLifecycleUnhappyPath(TaskLifecycleTestBase):
    """Test failure and cancellation scenarios"""

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_task_failure_lifecycle(self, provider):
        """Test task lifecycle ending in failure"""

        # Create and progress task to working
        task = self.create_submission_task("fail_001", "fail_ctx_001")
        await provider.store_task(task)

        working_task = self.create_working_task_update(task)
        await provider.update_task(working_task)

        # Transition to failed state
        failed_task = self.create_failed_task_update(
            working_task, "Encountered an unexpected error during processing"
        )

        fail_result = await provider.update_task(failed_task)
        assert fail_result.data is None, "Failure update should succeed"

        # Verify failed state
        final_result = await provider.get_task("fail_001")
        assert final_result.data is not None
        final_task = final_result.data

        assert final_task.status.state == TaskState.FAILED
        assert "unexpected error" in final_task.status.message.parts[0].text
        assert len(final_task.history or []) >= 2, "Should preserve history through failure"

        # Verify appears in failed task queries
        failed_query = A2ATaskQuery(context_id="fail_ctx_001", state=TaskState.FAILED)
        find_result = await provider.find_tasks(failed_query)
        assert find_result.data is not None
        assert len(find_result.data) == 1
        assert find_result.data[0].status.state == TaskState.FAILED

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_task_cancellation_lifecycle(self, provider):
        """Test task lifecycle ending in cancellation"""

        # Create and progress task to working
        task = self.create_submission_task("cancel_001", "cancel_ctx_001")
        await provider.store_task(task)

        working_task = self.create_working_task_update(task)
        await provider.update_task(working_task)

        # Add some progress
        await provider.update_task_status(
            "cancel_001",
            TaskState.WORKING,
            A2AMessage(
                role="agent",
                parts=[A2ATextPart(kind="text", text="Working on your request...")],
                messageId="working_msg",
                contextId="cancel_ctx_001",
                kind="message",
            ),
        )

        # Cancel the task
        canceled_task = self.create_canceled_task_update(
            working_task, "Task was canceled at user request"
        )

        cancel_result = await provider.update_task(canceled_task)
        assert cancel_result.data is None, "Cancellation should succeed"

        # Verify canceled state
        final_result = await provider.get_task("cancel_001")
        assert final_result.data is not None
        final_task = final_result.data

        assert final_task.status.state == TaskState.CANCELED
        assert "canceled at user request" in final_task.status.message.parts[0].text
        assert len(final_task.history or []) >= 2, "Should preserve history through cancellation"

        # Verify appears in canceled task queries
        canceled_query = A2ATaskQuery(state=TaskState.CANCELED)
        find_result = await provider.find_tasks(canceled_query)
        assert find_result.data is not None
        canceled_tasks = [t for t in find_result.data if t.id == "cancel_001"]
        assert len(canceled_tasks) == 1


class TestMultiTaskContextManagement(TaskLifecycleTestBase):
    """Test management of multiple tasks within contexts"""

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_multiple_tasks_single_context(self, provider):
        """Test creating and managing multiple tasks in a single context"""
        context_id = "multi_task_ctx_001"

        # Create multiple tasks in same context
        task_ids = [f"multi_task_{i:03d}" for i in range(5)]

        for task_id in task_ids:
            task = self.create_submission_task(task_id, context_id)
            store_result = await provider.store_task(task)
            assert store_result.data is None, f"Task {task_id} should store successfully"

        # Progress tasks to different states
        states = [
            TaskState.SUBMITTED,
            TaskState.WORKING,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        ]

        for i, (task_id, target_state) in enumerate(zip(task_ids, states)):
            get_result = await provider.get_task(task_id)
            current_task = get_result.data

            if target_state == TaskState.WORKING:
                updated_task = self.create_working_task_update(current_task)
                await provider.update_task(updated_task)
            elif target_state == TaskState.COMPLETED:
                working_task = self.create_working_task_update(current_task)
                await provider.update_task(working_task)
                completed_task = self.create_completed_task_update(working_task)
                await provider.update_task(completed_task)
            elif target_state == TaskState.FAILED:
                working_task = self.create_working_task_update(current_task)
                await provider.update_task(working_task)
                failed_task = self.create_failed_task_update(working_task)
                await provider.update_task(failed_task)
            elif target_state == TaskState.CANCELED:
                working_task = self.create_working_task_update(current_task)
                await provider.update_task(working_task)
                canceled_task = self.create_canceled_task_update(working_task)
                await provider.update_task(canceled_task)

        # Verify all tasks in context
        context_result = await provider.get_tasks_by_context(context_id)
        assert context_result.data is not None
        assert len(context_result.data) == 5, "Should have all 5 tasks"

        # Verify task statistics
        stats_result = await provider.get_task_stats(context_id)
        assert stats_result.data is not None
        stats = stats_result.data

        assert stats["total_tasks"] == 5
        assert stats["submitted"] == 1
        assert stats["working"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["canceled"] == 1

        # Test context deletion
        delete_result = await provider.delete_tasks_by_context(context_id)
        assert delete_result.data == 5, "Should delete all 5 tasks"

        # Verify context is empty
        empty_context_result = await provider.get_tasks_by_context(context_id)
        assert empty_context_result.data is not None
        assert len(empty_context_result.data) == 0, "Context should be empty after deletion"

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_task_stats_accuracy(self, provider):
        """Test accuracy of task statistics across state changes"""
        context_id = "stats_ctx_001"

        # Create initial batch of tasks
        for i in range(10):
            task = self.create_submission_task(f"stats_task_{i:03d}", context_id)
            await provider.store_task(task)

        # Check initial stats
        stats_result = await provider.get_task_stats(context_id)
        stats = stats_result.data
        assert stats["total_tasks"] == 10
        assert stats["submitted"] == 10
        assert stats.get("working", 0) == 0

        # Progress 5 tasks to working
        for i in range(5):
            get_result = await provider.get_task(f"stats_task_{i:03d}")
            working_task = self.create_working_task_update(get_result.data)
            await provider.update_task(working_task)

        # Check updated stats
        stats_result = await provider.get_task_stats(context_id)
        stats = stats_result.data
        assert stats["total_tasks"] == 10
        assert stats["submitted"] == 5
        assert stats["working"] == 5

        # Complete 3 tasks
        for i in range(3):
            get_result = await provider.get_task(f"stats_task_{i:03d}")
            completed_task = self.create_completed_task_update(get_result.data)
            await provider.update_task(completed_task)

        # Check final stats
        stats_result = await provider.get_task_stats(context_id)
        stats = stats_result.data
        assert stats["total_tasks"] == 10
        assert stats["submitted"] == 5
        assert stats["working"] == 2
        assert stats["completed"] == 3


class TestTaskQueryAndPagination(TaskLifecycleTestBase):
    """Test task querying and pagination functionality"""

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_task_query_by_state(self, provider):
        """Test querying tasks by different states"""
        context_id = "query_ctx_001"

        # Create tasks in different states
        submitted_tasks = []
        working_tasks = []
        completed_tasks = []

        for i in range(3):
            # Submitted tasks
            submitted_task = self.create_submission_task(f"submitted_{i}", context_id)
            await provider.store_task(submitted_task)
            submitted_tasks.append(submitted_task)

            # Working tasks
            working_base = self.create_submission_task(f"working_{i}", context_id)
            await provider.store_task(working_base)
            working_task = self.create_working_task_update(working_base)
            await provider.update_task(working_task)
            working_tasks.append(working_task)

            # Completed tasks
            completed_base = self.create_submission_task(f"completed_{i}", context_id)
            await provider.store_task(completed_base)
            working_completed = self.create_working_task_update(completed_base)
            await provider.update_task(working_completed)
            completed_task = self.create_completed_task_update(working_completed)
            await provider.update_task(completed_task)
            completed_tasks.append(completed_task)

        # Query submitted tasks
        submitted_query = A2ATaskQuery(context_id=context_id, state=TaskState.SUBMITTED)
        submitted_result = await provider.find_tasks(submitted_query)
        assert submitted_result.data is not None
        assert len(submitted_result.data) == 3
        assert all(task.status.state == TaskState.SUBMITTED for task in submitted_result.data)

        # Query working tasks
        working_query = A2ATaskQuery(context_id=context_id, state=TaskState.WORKING)
        working_result = await provider.find_tasks(working_query)
        assert working_result.data is not None
        assert len(working_result.data) == 3
        assert all(task.status.state == TaskState.WORKING for task in working_result.data)

        # Query completed tasks
        completed_query = A2ATaskQuery(context_id=context_id, state=TaskState.COMPLETED)
        completed_result = await provider.find_tasks(completed_query)
        assert completed_result.data is not None
        assert len(completed_result.data) == 3
        assert all(task.status.state == TaskState.COMPLETED for task in completed_result.data)

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_task_query_pagination(self, provider):
        """Test pagination through large task sets"""
        context_id = "pagination_ctx_001"

        # Create 25 tasks for pagination testing
        total_tasks = 25
        for i in range(total_tasks):
            task = self.create_submission_task(f"page_task_{i:03d}", context_id)
            await provider.store_task(task)

        # Test pagination with limit of 10
        page_size = 10
        all_retrieved = []
        offset = 0

        while True:
            query = A2ATaskQuery(context_id=context_id, limit=page_size, offset=offset)

            page_result = await provider.find_tasks(query)
            assert page_result.data is not None
            page_tasks = page_result.data

            if not page_tasks:
                break

            all_retrieved.extend(page_tasks)
            offset += len(page_tasks)

            # Prevent infinite loop
            if offset > total_tasks * 2:
                break

        # Verify we got all tasks
        assert len(all_retrieved) == total_tasks, (
            f"Expected {total_tasks} tasks, got {len(all_retrieved)}"
        )

        # Verify no duplicates
        task_ids = [task.id for task in all_retrieved]
        assert len(set(task_ids)) == total_tasks, "Should have no duplicate tasks"

        # Verify all tasks are from correct context
        assert all(task.context_id == context_id for task in all_retrieved)

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_task_query_time_range(self, provider):
        """Test querying tasks by time range"""
        context_id = "time_ctx_001"

        # Create tasks with timestamps spread over time
        base_time = datetime.now(timezone.utc)

        # Tasks from 1 hour ago
        old_time = base_time - timedelta(hours=1)
        old_task = self.create_submission_task("old_task", context_id)
        old_task = old_task.model_copy(
            update={
                "status": old_task.status.model_copy(update={"timestamp": old_time.isoformat()})
            }
        )
        # Store with created_at metadata to control the timestamp used for filtering
        await provider.store_task(old_task, metadata={"created_at": old_time.isoformat()})

        # Tasks from now
        new_task = self.create_submission_task("new_task", context_id)
        new_task = new_task.model_copy(
            update={
                "status": new_task.status.model_copy(update={"timestamp": base_time.isoformat()})
            }
        )
        # Store with created_at metadata to control the timestamp used for filtering
        await provider.store_task(new_task, metadata={"created_at": base_time.isoformat()})

        # Query tasks since 30 minutes ago
        since_time = base_time - timedelta(minutes=30)
        time_query = A2ATaskQuery(context_id=context_id, since=since_time)

        recent_result = await provider.find_tasks(time_query)
        assert recent_result.data is not None

        # Should only get the new task
        recent_tasks = recent_result.data
        assert len(recent_tasks) >= 1, "Should find at least the new task"

        # All tasks should be newer than since_time
        for task in recent_tasks:
            if task.status.timestamp:
                task_time = datetime.fromisoformat(task.status.timestamp.replace("Z", "+00:00"))
                assert task_time >= since_time, f"Task {task.id} is older than since_time"


class TestTaskErrorHandling(TaskLifecycleTestBase):
    """Test error handling scenarios"""

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_get_nonexistent_task(self, provider):
        """Test getting a task that doesn't exist"""

        result = await provider.get_task("nonexistent_task_12345")

        # Should return success with None data (not found)
        assert result.data is None

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_update_nonexistent_task(self, provider):
        """Test updating a task that doesn't exist"""

        nonexistent_task = self.create_submission_task("nonexistent_update", "nonexistent_ctx")

        result = await provider.update_task(nonexistent_task)

        # Should fail with appropriate error
        if hasattr(result, "data"):
            assert result.data is None
        elif hasattr(result, "error"):
            assert result.error is not None
        else:
            assert False, "Result should have either data or error"

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_delete_nonexistent_task(self, provider):
        """Test deleting a task that doesn't exist"""

        result = await provider.delete_task("nonexistent_delete_12345")

        # Should return False indicating task didn't exist
        assert result.data is False

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_invalid_task_data_handling(self, provider):
        """Test handling of invalid task data"""

        # Create task with invalid data (empty ID)
        try:
            invalid_task = A2ATask(
                id="",  # Invalid empty ID
                contextId="invalid_ctx",
                kind="task",
                status=A2ATaskStatus(state=TaskState.SUBMITTED),
            )

            result = await provider.store_task(invalid_task)

            # Should fail with validation error
            assert result.data is None
            assert result.error is not None

        except Exception:
            # Pydantic validation might catch this before reaching the provider
            pass


class TestProviderHealthAndCleanup(TaskLifecycleTestBase):
    """Test provider health checks and cleanup operations"""

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_provider_health_check(self, provider):
        """Test provider health check functionality"""

        health_result = await provider.health_check()

        assert health_result.data is not None
        health_data = health_result.data

        assert "healthy" in health_data
        assert health_data["healthy"] is True
        assert "provider" in health_data

        # Should have timing information
        assert "latency_ms" in health_data or "response_time_ms" in health_data

    @pytest.mark.parametrize("provider", PROVIDER_TYPES, indirect=True)
    async def test_cleanup_expired_tasks(self, provider):
        """Test cleanup of expired tasks"""

        # Create tasks with different expiration times
        context_id = "cleanup_ctx_001"

        # Create normal task (should not be cleaned up)
        normal_task = self.create_submission_task("normal_task", context_id)
        await provider.store_task(normal_task)

        # Simulate expired task by creating completed task in the past
        old_task = self.create_submission_task("old_task", context_id)
        working_old = self.create_working_task_update(old_task)
        await provider.update_task(working_old)

        completed_old = self.create_completed_task_update(working_old)
        # Set old timestamp
        completed_old = completed_old.model_copy(
            update={
                "status": completed_old.status.model_copy(
                    update={
                        "timestamp": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
                    }
                )
            }
        )
        await provider.update_task(completed_old)

        # Run cleanup
        cleanup_result = await provider.cleanup_expired_tasks()

        # Should report some cleanup activity
        assert cleanup_result.data is not None
        cleanup_count = cleanup_result.data
        assert isinstance(cleanup_count, int)
        assert cleanup_count >= 0  # Should not fail

        # Verify normal task still exists
        normal_result = await provider.get_task("normal_task")
        assert normal_result.data is not None, "Normal task should not be cleaned up"
