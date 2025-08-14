"""
A2A Memory Cleanup Tests - Phase 3: Cleanup and Expiration

Comprehensive tests for A2A task cleanup and expiration functionality.
Based on src/a2a/memory/__tests__/cleanup.test.ts patterns.

Tests automatic cleanup, manual cleanup, expiration handling, and dry-run modes.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from jaf.a2a.memory.cleanup import A2ATaskCleanupConfig, perform_task_cleanup
from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
from jaf.a2a.memory.types import (
    A2AInMemoryTaskConfig,
    A2ATaskProvider,
)
from jaf.a2a.types import A2AMessage, A2ATask, A2ATaskStatus, A2ATextPart, TaskState


class TaskCleanupTestBase:
    """Base class for cleanup testing utilities"""

    def create_aged_task(
        self,
        task_id: str,
        context_id: str,
        state: TaskState,
        age_hours: int = 0
    ) -> A2ATask:
        """Create a task with a specific age"""
        timestamp = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()

        return A2ATask(
            id=task_id,
            contextId=context_id,
            kind="task",
            status=A2ATaskStatus(
                state=state,
                message=A2AMessage(
                    role="agent" if state != TaskState.SUBMITTED else "user",
                    parts=[A2ATextPart(kind="text", text=f"Task in {state.value} state")],
                    messageId=f"msg_{task_id}",
                    contextId=context_id,
                    kind="message"
                ),
                timestamp=timestamp
            ),
            metadata={
                "created_at": timestamp,
                "test_age_hours": age_hours
            }
        )

    def create_diverse_aged_dataset(self, context_id: str = "cleanup_test_ctx") -> List[A2ATask]:
        """Create a dataset with tasks of various ages and states"""
        tasks = []

        # Fresh tasks (0-2 hours old) in various states
        tasks.extend([
            self.create_aged_task(f"fresh_submitted_{i}", context_id, TaskState.SUBMITTED, i)
            for i in range(2)
        ])
        tasks.extend([
            self.create_aged_task(f"fresh_working_{i}", context_id, TaskState.WORKING, i)
            for i in range(2)
        ])
        tasks.extend([
            self.create_aged_task(f"fresh_completed_{i}", context_id, TaskState.COMPLETED, i)
            for i in range(3)
        ])

        # Medium age tasks (1-3 days old)
        medium_age_hours = [24, 48, 72]  # 1, 2, 3 days
        for hours in medium_age_hours:
            tasks.extend([
                self.create_aged_task(f"medium_completed_{hours}h", context_id, TaskState.COMPLETED, hours),
                self.create_aged_task(f"medium_failed_{hours}h", context_id, TaskState.FAILED, hours),
                self.create_aged_task(f"medium_working_{hours}h", context_id, TaskState.WORKING, hours)
            ])

        # Old tasks (1-2 weeks old)
        old_age_hours = [168, 336]  # 1 week, 2 weeks
        for hours in old_age_hours:
            tasks.extend([
                self.create_aged_task(f"old_completed_{hours}h", context_id, TaskState.COMPLETED, hours),
                self.create_aged_task(f"old_failed_{hours}h", context_id, TaskState.FAILED, hours),
                self.create_aged_task(f"old_canceled_{hours}h", context_id, TaskState.CANCELED, hours),
                self.create_aged_task(f"old_working_{hours}h", context_id, TaskState.WORKING, hours)  # Should NOT be cleaned
            ])

        return tasks


@pytest.fixture
async def cleanup_provider() -> A2ATaskProvider:
    """Create provider for cleanup testing"""
    config = A2AInMemoryTaskConfig(
        max_tasks=1000,
        cleanup_interval=3600  # 1 hour
    )
    provider = create_a2a_in_memory_task_provider(config)
    yield provider
    await provider.close()


class TestTaskCleanupByAge(TaskCleanupTestBase):
    """Test cleanup of tasks based on age"""

    async def test_cleanup_old_completed_tasks(self, cleanup_provider):
        """Test that old completed tasks are cleaned up while preserving recent ones"""
        context_id = "age_cleanup_ctx"

        # Create test dataset
        tasks = self.create_diverse_aged_dataset(context_id)

        # Store all tasks
        for task in tasks:
            await cleanup_provider.store_task(task)

        # Verify initial state
        initial_result = await cleanup_provider.get_tasks_by_context(context_id)
        initial_count = len(initial_result.data)
        assert initial_count == len(tasks), "All tasks should be stored initially"

        # Perform cleanup with 7-day retention (168 hours)
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168 * 3600,  # 7 days in seconds
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value]  # Retain active states
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)
        assert cleanup_result.data is not None, "Cleanup should succeed"

        cleanup_count = cleanup_result.data.total_cleaned
        assert cleanup_count > 0, "Should clean up some old tasks"

        # Verify remaining tasks
        remaining_result = await cleanup_provider.get_tasks_by_context(context_id)
        remaining_tasks = remaining_result.data

        # All working tasks should remain (regardless of age)
        working_tasks = [t for t in remaining_tasks if t.status.state == TaskState.WORKING]
        expected_working = [t for t in tasks if t.status.state == TaskState.WORKING]
        assert len(working_tasks) == len(expected_working), "All working tasks should be preserved"

        # Recent completed tasks should remain
        recent_completed = [
            t for t in remaining_tasks
            if t.status.state == TaskState.COMPLETED and "fresh_" in t.id
        ]
        assert len(recent_completed) > 0, "Recent completed tasks should be preserved"

        # Old completed tasks should be removed
        old_completed = [
            t for t in remaining_tasks
            if t.status.state == TaskState.COMPLETED and "old_" in t.id
        ]
        assert len(old_completed) == 0, "Old completed tasks should be cleaned up"

        print(f"Age-based cleanup: {cleanup_count} tasks cleaned, {len(remaining_tasks)} remaining")

    async def test_cleanup_respects_state_filters(self, cleanup_provider):
        """Test that cleanup only affects specified states"""
        context_id = "state_filter_ctx"

        # Create old tasks in various states
        old_age = 336  # 2 weeks
        test_tasks = [
            self.create_aged_task("old_submitted", context_id, TaskState.SUBMITTED, old_age),
            self.create_aged_task("old_working", context_id, TaskState.WORKING, old_age),
            self.create_aged_task("old_completed", context_id, TaskState.COMPLETED, old_age),
            self.create_aged_task("old_failed", context_id, TaskState.FAILED, old_age),
            self.create_aged_task("old_canceled", context_id, TaskState.CANCELED, old_age)
        ]

        # Store all tasks
        for task in test_tasks:
            await cleanup_provider.store_task(task)

        # Cleanup only completed and failed tasks
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,  # 7 days (all tasks are older)
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value]
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)
        cleanup_count = cleanup_result.data.total_cleaned

        # Verify results
        remaining_result = await cleanup_provider.get_tasks_by_context(context_id)
        remaining_tasks = remaining_result.data
        remaining_states = {t.status.state for t in remaining_tasks}

        # Should have cleaned up 3 tasks (completed, failed, and canceled)
        assert cleanup_count == 3, f"Expected 3 tasks cleaned, got {cleanup_count}"

        # Should preserve submitted and working tasks
        assert TaskState.SUBMITTED in remaining_states, "Submitted task should be preserved"
        assert TaskState.WORKING in remaining_states, "Working task should be preserved"
        
        # Should not have completed, failed, or canceled tasks
        assert TaskState.COMPLETED not in remaining_states, "Completed task should be cleaned"
        assert TaskState.FAILED not in remaining_states, "Failed task should be cleaned"
        assert TaskState.CANCELED not in remaining_states, "Canceled task should be cleaned"

    async def test_cleanup_dry_run_mode(self, cleanup_provider):
        """Test that dry run mode reports what would be cleaned without actually deleting"""
        context_id = "dry_run_ctx"

        # Create test dataset
        tasks = self.create_diverse_aged_dataset(context_id)
        for task in tasks:
            await cleanup_provider.store_task(task)

        # Count initial tasks
        initial_result = await cleanup_provider.get_tasks_by_context(context_id)
        initial_count = len(initial_result.data)

        # Perform dry run cleanup
        dry_run_config = A2ATaskCleanupConfig(
            max_age=168*3600,  # 7 days
            dry_run=True,  # Dry run mode
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value]
        )

        dry_run_result = await perform_task_cleanup(cleanup_provider, dry_run_config)
        assert dry_run_result.data is not None, "Dry run should succeed"

        would_delete_count = dry_run_result.data.would_delete_count
        assert would_delete_count > 0, "Dry run should report tasks that would be deleted"

        # Verify no tasks were actually deleted
        after_dry_run_result = await cleanup_provider.get_tasks_by_context(context_id)
        after_count = len(after_dry_run_result.data)

        assert after_count == initial_count, "Dry run should not delete any tasks"

        # Now perform actual cleanup and verify it matches dry run prediction
        actual_config = A2ATaskCleanupConfig(
            max_age=168*3600,
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value]
        )

        actual_result = await perform_task_cleanup(cleanup_provider, actual_config)
        actual_delete_count = actual_result.data.total_cleaned

        assert actual_delete_count == would_delete_count, f"Actual cleanup ({actual_delete_count}) should match dry run prediction ({would_delete_count})"

        print(f"Dry run test: predicted {would_delete_count}, actual {actual_delete_count}")


class TestTaskCleanupByCount(TaskCleanupTestBase):
    """Test cleanup based on task count limits"""

    async def test_cleanup_excess_completed_tasks(self, cleanup_provider):
        """Test cleanup when there are too many completed tasks"""
        context_id = "count_cleanup_ctx"

        # Create many completed tasks with different ages
        completed_tasks = []
        for i in range(20):  # Create 20 completed tasks
            age_hours = i  # Age from 0 to 19 hours
            task = self.create_aged_task(f"completed_{i:02d}", context_id, TaskState.COMPLETED, age_hours)
            completed_tasks.append(task)

        # Also create some non-completed tasks that should be preserved
        other_tasks = [
            self.create_aged_task("working_1", context_id, TaskState.WORKING, 50),
            self.create_aged_task("submitted_1", context_id, TaskState.SUBMITTED, 100),
            self.create_aged_task("failed_1", context_id, TaskState.FAILED, 10)
        ]

        # Store all tasks
        all_tasks = completed_tasks + other_tasks
        for task in all_tasks:
            await cleanup_provider.store_task(task)

        # Perform cleanup with limit of 10 completed tasks
        cleanup_config = A2ATaskCleanupConfig(
            max_completed_tasks=10,
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)
        cleanup_count = cleanup_result.data.total_cleaned

        # Should delete 10 tasks (20 - 10 = 10)
        assert cleanup_count == 10, f"Expected 10 tasks cleaned, got {cleanup_count}"

        # Verify remaining tasks
        remaining_result = await cleanup_provider.get_tasks_by_context(context_id)
        remaining_tasks = remaining_result.data

        completed_remaining = [t for t in remaining_tasks if t.status.state == TaskState.COMPLETED]
        other_remaining = [t for t in remaining_tasks if t.status.state != TaskState.COMPLETED]

        assert len(completed_remaining) == 10, "Should have exactly 10 completed tasks remaining"
        assert len(other_remaining) == 3, "All non-completed tasks should be preserved"

        # Verify that the oldest completed tasks were deleted (newest should remain)
        remaining_ids = {t.id for t in completed_remaining}
        # The 10 newest tasks should remain (completed_00 through completed_09)
        expected_remaining = {f"completed_{i:02d}" for i in range(10)}
        assert remaining_ids == expected_remaining, "Newest completed tasks should be preserved"

    async def test_cleanup_respects_multiple_limits(self, cleanup_provider):
        """Test cleanup with both age and count limits"""
        context_id = "multi_limit_ctx"

        # Create completed tasks: some old, some new
        tasks = []

        # 15 old completed tasks (> 7 days)
        for i in range(15):
            age_hours = 200 + i  # 8+ days old
            task = self.create_aged_task(f"old_completed_{i:02d}", context_id, TaskState.COMPLETED, age_hours)
            tasks.append(task)

        # 15 new completed tasks (< 7 days)
        for i in range(15):
            age_hours = i  # 0-14 hours old
            task = self.create_aged_task(f"new_completed_{i:02d}", context_id, TaskState.COMPLETED, age_hours)
            tasks.append(task)

        # Store all tasks
        for task in tasks:
            await cleanup_provider.store_task(task)

        # Cleanup with both age limit (7 days) and count limit (10)
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,  # 7 days
            max_completed_tasks=10,
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)
        cleanup_count = cleanup_result.data.total_cleaned

        # Should delete:
        # 1. All 15 old tasks (due to age limit)
        # 2. 5 additional new tasks (due to count limit: 15 new - 10 limit = 5)
        # Total: 20 deleted
        expected_deletions = 15 + 5  # old tasks + excess new tasks
        assert cleanup_count == expected_deletions, f"Expected {expected_deletions} deletions, got {cleanup_count}"

        # Verify exactly 10 tasks remain (all new)
        remaining_result = await cleanup_provider.get_tasks_by_context(context_id)
        remaining_tasks = remaining_result.data

        assert len(remaining_tasks) == 10, "Should have exactly 10 tasks remaining"

        # All remaining should be new (< 7 days)
        for task in remaining_tasks:
            assert "new_completed_" in task.id, "Only new tasks should remain"


class TestCleanupContextIsolation(TaskCleanupTestBase):
    """Test that cleanup respects context boundaries"""

    async def test_cleanup_single_context_only(self, cleanup_provider):
        """Test cleanup of a single context without affecting others"""
        # Create tasks in multiple contexts
        contexts = ["ctx_a", "ctx_b", "ctx_c"]
        tasks_per_context = 10

        all_tasks = []
        for ctx in contexts:
            for i in range(tasks_per_context):
                # Mix of old completed and old working tasks
                if i % 2 == 0:
                    task = self.create_aged_task(f"task_{ctx}_{i}", ctx, TaskState.COMPLETED, 200)  # Old
                else:
                    task = self.create_aged_task(f"task_{ctx}_{i}", ctx, TaskState.WORKING, 200)    # Old but should be preserved
                all_tasks.append(task)

        # Store all tasks
        for task in all_tasks:
            await cleanup_provider.store_task(task)

        # Cleanup only context_a
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,  # 7 days (all tasks are older)
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config, context_id="ctx_a")
        cleanup_count = cleanup_result.data.total_cleaned

        # Should delete 5 completed tasks from ctx_a (half of 10)
        assert cleanup_count == 5, f"Expected 5 deletions from ctx_a, got {cleanup_count}"

        # Verify ctx_a has only working tasks remaining
        ctx_a_result = await cleanup_provider.get_tasks_by_context("ctx_a")
        ctx_a_remaining = ctx_a_result.data
        assert len(ctx_a_remaining) == 5, "ctx_a should have 5 tasks remaining"
        assert all(t.status.state == TaskState.WORKING for t in ctx_a_remaining), "Only working tasks should remain in ctx_a"

        # Verify other contexts are unaffected
        for ctx in ["ctx_b", "ctx_c"]:
            ctx_result = await cleanup_provider.get_tasks_by_context(ctx)
            ctx_tasks = ctx_result.data
            assert len(ctx_tasks) == tasks_per_context, f"{ctx} should be unaffected"

            completed_in_ctx = [t for t in ctx_tasks if t.status.state == TaskState.COMPLETED]
            assert len(completed_in_ctx) == 5, f"{ctx} should still have completed tasks"

    async def test_cleanup_all_contexts(self, cleanup_provider):
        """Test cleanup across all contexts"""
        # Create tasks in multiple contexts
        contexts = ["global_ctx_1", "global_ctx_2", "global_ctx_3"]

        all_tasks = []
        for ctx in contexts:
            # Create mix of old and new completed tasks
            for i in range(5):
                old_task = self.create_aged_task(f"old_task_{ctx}_{i}", ctx, TaskState.COMPLETED, 200)
                new_task = self.create_aged_task(f"new_task_{ctx}_{i}", ctx, TaskState.COMPLETED, 1)
                all_tasks.extend([old_task, new_task])

        # Store all tasks
        for task in all_tasks:
            await cleanup_provider.store_task(task)

        # Global cleanup (no context filter)
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,  # 7 days
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)
        cleanup_count = cleanup_result.data.total_cleaned

        # Should delete 15 old completed tasks (5 per context × 3 contexts)
        assert cleanup_count == 15, f"Expected 15 global deletions, got {cleanup_count}"

        # Verify each context has only new tasks remaining
        for ctx in contexts:
            ctx_result = await cleanup_provider.get_tasks_by_context(ctx)
            ctx_tasks = ctx_result.data

            assert len(ctx_tasks) == 5, f"{ctx} should have 5 tasks remaining"
            assert all("new_task_" in t.id for t in ctx_tasks), f"Only new tasks should remain in {ctx}"


class TestCleanupPerformanceAndReliability(TaskCleanupTestBase):
    """Test cleanup performance and reliability"""

    async def test_cleanup_large_dataset_performance(self, cleanup_provider):
        """Test cleanup performance with large datasets"""
        context_id = "perf_cleanup_ctx"

        # Create large dataset with mix of states and ages
        large_dataset_size = 1000
        tasks = []

        for i in range(large_dataset_size):
            # Distribute states and ages
            if i % 3 == 0:
                state = TaskState.COMPLETED
                age = 200 if i % 6 == 0 else 1  # Half old, half new
            elif i % 3 == 1:
                state = TaskState.FAILED
                age = 200 if i % 6 == 1 else 1
            else:
                state = TaskState.WORKING
                age = 200  # Working tasks should not be cleaned regardless of age

            task = self.create_aged_task(f"perf_task_{i:04d}", context_id, state, age)
            tasks.append(task)

        # Store all tasks in batches
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_operations = [cleanup_provider.store_task(task) for task in batch]
            await asyncio.gather(*batch_operations)

        # Measure cleanup performance
        import time
        start_time = time.perf_counter()

        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,  # 7 days
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value]
        )

        cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)

        end_time = time.perf_counter()
        cleanup_time = end_time - start_time

        # Verify cleanup results
        assert cleanup_result.data is not None, "Cleanup should succeed"
        cleanup_count = cleanup_result.data.total_cleaned

        # Calculate expected deletions (old completed and failed tasks)
        expected_deletions = sum(1 for t in tasks if t.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED] and t.metadata.get("test_age_hours", 0) >= 168)

        assert cleanup_count == expected_deletions, f"Expected {expected_deletions} deletions, got {cleanup_count}"

        # Performance check
        cleanup_rate = cleanup_count / cleanup_time if cleanup_time > 0 else float('inf')
        assert cleanup_rate > 50, f"Cleanup too slow: {cleanup_rate:.2f} tasks/second"

        print(f"Large dataset cleanup: {cleanup_count} tasks cleaned in {cleanup_time:.2f}s ({cleanup_rate:.2f} tasks/s)")

    async def test_cleanup_error_recovery(self, cleanup_provider):
        """Test cleanup behavior when encountering errors"""
        context_id = "error_cleanup_ctx"

        # Create test dataset
        tasks = []
        for i in range(20):
            task = self.create_aged_task(f"error_task_{i:02d}", context_id, TaskState.COMPLETED, 200)
            tasks.append(task)

        # Store tasks
        for task in tasks:
            await cleanup_provider.store_task(task)

        # Simulate cleanup with potential errors
        # (This test depends on implementation details and might need adjustment)
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value],
            batch_size=5  # Small batches to test error handling
        )

        try:
            cleanup_result = await perform_task_cleanup(cleanup_provider, cleanup_config)

            # Cleanup should either succeed completely or fail gracefully
            assert cleanup_result.data is not None or cleanup_result.error is not None

            if cleanup_result.data:
                # If successful, verify some tasks were cleaned
                assert cleanup_result.data.total_cleaned > 0
                print(f"Error recovery test: {cleanup_result.data.total_cleaned} tasks cleaned successfully")
            else:
                # If failed, should have meaningful error
                assert "cleanup" in str(cleanup_result.error.message).lower()
                print(f"Error recovery test: Cleanup failed gracefully with error: {cleanup_result.error.message}")

        except Exception as e:
            # Even exceptions should be handled gracefully
            print(f"Error recovery test: Exception handled: {e}")

            # Provider should still be functional after error
            health_result = await cleanup_provider.health_check()
            assert health_result.data is not None
            assert health_result.data.get("healthy", False), "Provider should remain healthy after cleanup error"

    async def test_cleanup_consistency_under_concurrent_operations(self, cleanup_provider):
        """Test cleanup consistency when running concurrently with other operations"""
        context_id = "concurrent_cleanup_ctx"

        # Create initial dataset
        initial_tasks = []
        for i in range(50):
            # Mix of old and new completed tasks
            age = 200 if i % 2 == 0 else 1
            task = self.create_aged_task(f"concurrent_task_{i:02d}", context_id, TaskState.COMPLETED, age)
            initial_tasks.append(task)

        # Store initial tasks
        for task in initial_tasks:
            await cleanup_provider.store_task(task)

        # Start cleanup operation
        cleanup_config = A2ATaskCleanupConfig(
            max_age=168*3600,
            dry_run=False,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )
        cleanup_operation = perform_task_cleanup(cleanup_provider, cleanup_config)

        # Concurrent operations during cleanup
        concurrent_operations = []

        # Add new tasks
        for i in range(10):
            new_task = self.create_aged_task(f"new_concurrent_{i}", context_id, TaskState.COMPLETED, 1)
            concurrent_operations.append(cleanup_provider.store_task(new_task))

        # Query tasks
        query_operations = [
            cleanup_provider.get_tasks_by_context(context_id),
            cleanup_provider.get_task_stats(context_id)
        ]
        concurrent_operations.extend(query_operations)

        # Execute cleanup and concurrent operations
        all_operations = [cleanup_operation] + concurrent_operations
        results = await asyncio.gather(*all_operations, return_exceptions=True)

        # Analyze results
        cleanup_result = results[0]
        concurrent_results = results[1:]

        # Cleanup should complete successfully
        assert not isinstance(cleanup_result, Exception), f"Cleanup failed: {cleanup_result}"
        assert cleanup_result.data is not None, "Cleanup should succeed"

        # Most concurrent operations should succeed
        successful_concurrent = sum(1 for r in concurrent_results if not isinstance(r, Exception))
        total_concurrent = len(concurrent_results)

        success_rate = successful_concurrent / total_concurrent
        assert success_rate >= 0.8, f"Too many concurrent operations failed: {success_rate:.2%}"

        # Verify final state consistency
        final_result = await cleanup_provider.get_tasks_by_context(context_id)
        assert final_result.data is not None, "Final state should be queryable"

        print(f"Concurrent cleanup test: {cleanup_result.data.total_cleaned} cleaned, {successful_concurrent}/{total_concurrent} concurrent ops succeeded")


class TestCleanupConfigurationValidation(TaskCleanupTestBase):
    """Test cleanup configuration validation and edge cases"""

    async def test_invalid_cleanup_configurations(self, cleanup_provider):
        """Test validation of invalid cleanup configurations"""
        from jaf.memory.types import Failure

        # Test invalid age limit
        invalid_configs = [
            A2ATaskCleanupConfig(max_age=-3600),  # Negative age in seconds
            A2ATaskCleanupConfig(max_completed_tasks=-1),  # Negative count
            A2ATaskCleanupConfig(batch_size=0),  # Zero batch size
        ]

        for config in invalid_configs:
            result = await perform_task_cleanup(cleanup_provider, config)
            assert isinstance(result, Failure), f"Config {config} should have failed validation"
            assert "invalid" in str(result.error.message).lower() or "validation" in str(result.error.message).lower()

    async def test_edge_case_configurations(self, cleanup_provider):
        """Test edge case cleanup configurations"""
        context_id = "edge_case_ctx"

        # Create test data
        test_task = self.create_aged_task("edge_task", context_id, TaskState.COMPLETED, 100)
        await cleanup_provider.store_task(test_task)

        # Test very large age limit (should clean nothing)
        large_age_config = A2ATaskCleanupConfig(
            max_age=10000*3600,  # ~1 year
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        large_age_result = await perform_task_cleanup(cleanup_provider, large_age_config)
        assert large_age_result.data.total_cleaned == 0, "Large age limit should clean nothing"

        # Test very small age limit (should clean everything eligible)
        small_age_config = A2ATaskCleanupConfig(
            max_age=1*3600,  # 1 hour
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        small_age_result = await perform_task_cleanup(cleanup_provider, small_age_config)
        assert small_age_result.data.total_cleaned == 1, "Small age limit should clean the old task"

        # Recreate test task for next test
        await cleanup_provider.store_task(test_task)

        # Test very large count limit (should clean nothing)
        large_count_config = A2ATaskCleanupConfig(
            max_completed_tasks=10000,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        large_count_result = await perform_task_cleanup(cleanup_provider, large_count_config)
        assert large_count_result.data.total_cleaned == 0, "Large count limit should clean nothing"

        # Test zero count limit (should clean everything)
        zero_count_config = A2ATaskCleanupConfig(
            max_completed_tasks=0,
            retain_states=[TaskState.SUBMITTED.value, TaskState.WORKING.value, TaskState.FAILED.value, TaskState.CANCELED.value]
        )

        zero_count_result = await perform_task_cleanup(cleanup_provider, zero_count_config)
        assert zero_count_result.data.total_cleaned == 1, "Zero count limit should clean all tasks"
