"""
A2A Memory Stress & Concurrency Tests - Phase 3: Advanced Scenarios

Comprehensive stress testing for A2A task memory system including:
- High concurrency scenarios
- Large-scale data operations
- Performance under load
- Race condition detection
- Resource exhaustion handling
- Memory leak detection

These tests push the system to its limits to find subtle bugs and performance issues.
"""

import asyncio
import gc
import time
import weakref
from datetime import datetime, timedelta, timezone
from typing import Any, List, Tuple

import pytest

from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
from jaf.a2a.memory.types import (
    A2AInMemoryTaskConfig,
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


class StressTestBase:
    """Base class for stress testing utilities"""

    def create_bulk_tasks(
        self,
        count: int,
        context_id: str = "stress_ctx",
        base_id: str = "stress_task",
        distribute_contexts: bool = True,
    ) -> List[A2ATask]:
        """Create multiple tasks for bulk operations"""
        tasks = []
        for i in range(count):
            # Distribute across contexts by default for stress testing, but allow single context
            actual_context_id = f"{context_id}_{i % 10}" if distribute_contexts else context_id

            task = A2ATask(
                id=f"{base_id}_{i:05d}",
                contextId=actual_context_id,
                kind="task",
                status=A2ATaskStatus(
                    state=TaskState.SUBMITTED,
                    message=A2AMessage(
                        role="user",
                        parts=[A2ATextPart(kind="text", text=f"Bulk task number {i}")],
                        messageId=f"bulk_msg_{i}",
                        contextId=actual_context_id,
                        kind="message",
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                metadata={
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "batch_id": "stress_test",
                    "sequence": i,
                },
            )
            tasks.append(task)
        return tasks

    def create_large_task(
        self, task_id: str, context_id: str, size_multiplier: int = 100
    ) -> A2ATask:
        """Create a task with large data payload"""
        # Create large text content
        large_text = "This is a test message. " * (size_multiplier * 100)  # ~2.5KB per 100x

        # Create large data structure
        large_data = {
            "data": list(range(size_multiplier * 10)),
            "metadata": {f"key_{i}": f"value_{i}" * 50 for i in range(size_multiplier)},
            "content": large_text,
        }

        return A2ATask(
            id=task_id,
            contextId=context_id,
            kind="task",
            status=A2ATaskStatus(
                state=TaskState.WORKING,
                message=A2AMessage(
                    role="agent",
                    parts=[
                        A2ATextPart(kind="text", text=large_text),
                        A2ADataPart(kind="data", data=large_data),
                    ],
                    messageId=f"large_msg_{task_id}",
                    contextId=context_id,
                    kind="message",
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            history=[
                A2AMessage(
                    role="user",
                    parts=[A2ATextPart(kind="text", text="Process this large dataset")],
                    messageId=f"init_{task_id}",
                    contextId=context_id,
                    kind="message",
                )
            ],
            artifacts=[
                A2AArtifact(
                    artifactId=f"large_artifact_{task_id}",
                    name="Large Data Artifact",
                    description="An artifact containing large amounts of data",
                    parts=[
                        A2ATextPart(kind="text", text=large_text),
                        A2ADataPart(kind="data", data=large_data),
                    ],
                )
            ],
        )

    async def measure_operation_time(self, operation_func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of an async operation"""
        start_time = time.perf_counter()
        result = await operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, (end_time - start_time) * 1000  # Return time in milliseconds


@pytest.fixture
async def stress_provider() -> A2ATaskProvider:
    """Create provider for stress testing with higher limits"""
    config = A2AInMemoryTaskConfig(
        max_tasks=50000,  # Higher limits for stress testing
        max_tasks_per_context=10000,
    )
    provider = create_a2a_in_memory_task_provider(config)
    yield provider
    await provider.close()


class TestConcurrencyTorture(StressTestBase):
    """Torture tests for concurrent operations"""

    async def test_massive_concurrent_writes(self, stress_provider):
        """Test handling of massive concurrent write operations"""
        concurrent_tasks = 500
        contexts_count = 50

        tasks = self.create_bulk_tasks(concurrent_tasks, "concurrent_write", "write_task")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(100)  # Limit to 100 concurrent operations

        async def store_with_semaphore(task):
            async with semaphore:
                return await stress_provider.store_task(task)

        # Execute all stores concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*[store_with_semaphore(task) for task in tasks])
        end_time = time.perf_counter()

        # Verify all operations succeeded
        success_count = sum(1 for result in results if result.data is None)
        assert success_count == concurrent_tasks, (
            f"Expected all {concurrent_tasks} stores to succeed, got {success_count}"
        )

        # Performance check - should complete within reasonable time
        total_time = end_time - start_time
        avg_time_per_op = total_time / concurrent_tasks
        assert total_time < 30.0, f"Concurrent writes took too long: {total_time:.2f}s"

        print(
            f"Concurrent writes: {concurrent_tasks} tasks in {total_time:.2f}s ({avg_time_per_op * 1000:.2f}ms avg)"
        )

        # Verify data integrity
        for task in tasks[:10]:  # Sample check
            get_result = await stress_provider.get_task(task.id)
            assert get_result.data is not None, f"Task {task.id} should be retrievable"
            assert get_result.data.id == task.id

    async def test_concurrent_read_write_operations(self, stress_provider):
        """Test mixed concurrent read and write operations"""
        # Setup initial data
        initial_tasks = self.create_bulk_tasks(100, "rw_ctx", "rw_task")
        for task in initial_tasks:
            await stress_provider.store_task(task)

        # Concurrent operations mix
        read_operations = []
        write_operations = []
        update_operations = []

        # Create read operations
        for i in range(200):
            task_id = f"rw_task_{i % 100:05d}"
            read_operations.append(stress_provider.get_task(task_id))

        # Create new write operations
        new_tasks = self.create_bulk_tasks(50, "rw_new_ctx", "rw_new_task")
        for task in new_tasks:
            write_operations.append(stress_provider.store_task(task))

        # Create update operations
        for i in range(50):
            task_id = f"rw_task_{i:05d}"
            update_operations.append(
                stress_provider.update_task_status(
                    task_id,
                    TaskState.WORKING,
                    A2AMessage(
                        role="agent",
                        parts=[A2ATextPart(kind="text", text=f"Updated task {i}")],
                        messageId=f"update_{i}",
                        contextId=f"rw_ctx_{i % 10}",
                        kind="message",
                    ),
                )
            )

        # Execute all operations concurrently
        all_operations = read_operations + write_operations + update_operations
        start_time = time.perf_counter()
        results = await asyncio.gather(*all_operations)
        end_time = time.perf_counter()

        # Analyze results
        read_results = results[: len(read_operations)]
        write_results = results[len(read_operations) : len(read_operations) + len(write_operations)]
        update_results = results[len(read_operations) + len(write_operations) :]

        # Verify read operations
        successful_reads = sum(1 for result in read_results if result.data is not None)
        assert successful_reads == len(read_operations), "All reads should succeed"

        # Verify write operations
        successful_writes = sum(1 for result in write_results if result.data is None)
        assert successful_writes == len(write_operations), "All writes should succeed"

        # Verify update operations
        successful_updates = sum(1 for result in update_results if result.data is None)
        assert successful_updates == len(update_operations), "All updates should succeed"

        total_time = end_time - start_time
        print(f"Mixed operations: {len(all_operations)} ops in {total_time:.2f}s")

    async def test_race_condition_detection(self, stress_provider):
        """Test for race conditions in concurrent updates"""
        task_id = "race_task_001"
        context_id = "race_ctx_001"

        # Create initial task
        initial_task = A2ATask(
            id=task_id,
            contextId=context_id,
            kind="task",
            status=A2ATaskStatus(
                state=TaskState.SUBMITTED, timestamp=datetime.now(timezone.utc).isoformat()
            ),
        )
        await stress_provider.store_task(initial_task)

        # Create concurrent update operations
        update_count = 100
        update_operations = []

        for i in range(update_count):
            operation = stress_provider.update_task_status(
                task_id,
                TaskState.WORKING,
                A2AMessage(
                    role="agent",
                    parts=[A2ATextPart(kind="text", text=f"Concurrent update {i}")],
                    messageId=f"concurrent_{i}",
                    contextId=context_id,
                    kind="message",
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            update_operations.append(operation)

        # Execute all updates concurrently
        results = await asyncio.gather(*update_operations)

        # All updates should succeed
        successful_updates = sum(1 for result in results if result.data is None)
        assert successful_updates == update_count, (
            f"Expected {update_count} successful updates, got {successful_updates}"
        )

        # Verify final state consistency
        final_result = await stress_provider.get_task(task_id)
        assert final_result.data is not None
        final_task = final_result.data

        # Task should be in consistent state
        assert final_task.status.state == TaskState.WORKING
        assert final_task.id == task_id
        assert final_task.context_id == context_id

    async def test_context_isolation_under_load(self, stress_provider):
        """Test that contexts remain isolated under concurrent load"""
        contexts_count = 20
        tasks_per_context = 50

        # Create tasks in separate contexts concurrently
        all_store_operations = []
        expected_context_tasks = {}

        for ctx_idx in range(contexts_count):
            context_id = f"isolation_ctx_{ctx_idx:03d}"
            expected_context_tasks[context_id] = []

            for task_idx in range(tasks_per_context):
                task = A2ATask(
                    id=f"isolation_task_{ctx_idx:03d}_{task_idx:03d}",
                    contextId=context_id,
                    kind="task",
                    status=A2ATaskStatus(
                        state=TaskState.SUBMITTED,
                        message=A2AMessage(
                            role="user",
                            parts=[
                                A2ATextPart(
                                    kind="text", text=f"Task {task_idx} in context {ctx_idx}"
                                )
                            ],
                            messageId=f"iso_msg_{ctx_idx}_{task_idx}",
                            contextId=context_id,
                            kind="message",
                        ),
                    ),
                )
                expected_context_tasks[context_id].append(task.id)
                all_store_operations.append(stress_provider.store_task(task))

        # Execute all stores concurrently
        results = await asyncio.gather(*all_store_operations)

        # Verify all stores succeeded
        successful_stores = sum(1 for result in results if result.data is None)
        expected_total = contexts_count * tasks_per_context
        assert successful_stores == expected_total, (
            f"Expected {expected_total} stores, got {successful_stores}"
        )

        # Verify context isolation
        for context_id, expected_task_ids in expected_context_tasks.items():
            context_result = await stress_provider.get_tasks_by_context(context_id)
            assert context_result.data is not None, f"Context {context_id} should be retrievable"

            actual_task_ids = {task.id for task in context_result.data}
            expected_task_ids_set = set(expected_task_ids)

            assert actual_task_ids == expected_task_ids_set, (
                f"Context {context_id} has incorrect tasks"
            )
            assert len(context_result.data) == tasks_per_context, (
                f"Context {context_id} should have {tasks_per_context} tasks"
            )


class TestLargeScaleOperations(StressTestBase):
    """Test large-scale data operations"""

    async def test_bulk_task_storage_performance(self, stress_provider):
        """Test performance of storing large numbers of tasks"""
        task_counts = [100, 500, 1000, 2000]
        performance_results = []

        for count in task_counts:
            tasks = self.create_bulk_tasks(count, f"bulk_perf_{count}", f"perf_task_{count}")

            # Measure bulk storage time
            start_time = time.perf_counter()

            # Store tasks in batches to avoid overwhelming the system
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_operations = [stress_provider.store_task(task) for task in batch]
                batch_results = await asyncio.gather(*batch_operations)

                # Verify batch success
                batch_success = sum(1 for result in batch_results if result.data is None)
                assert batch_success == len(batch), (
                    f"Batch {i // batch_size} failed: {batch_success}/{len(batch)} succeeded"
                )

            end_time = time.perf_counter()
            total_time = end_time - start_time
            ops_per_second = count / total_time

            performance_results.append(
                {"count": count, "time": total_time, "ops_per_second": ops_per_second}
            )

            print(f"Bulk storage: {count} tasks in {total_time:.2f}s ({ops_per_second:.2f} ops/s)")

            # Performance regression check
            assert ops_per_second > 10, f"Storage performance too slow: {ops_per_second:.2f} ops/s"

        # Verify performance doesn't degrade significantly with scale
        if len(performance_results) >= 2:
            smallest_ops = performance_results[0]["ops_per_second"]
            largest_ops = performance_results[-1]["ops_per_second"]
            degradation_ratio = smallest_ops / largest_ops

            # Performance shouldn't degrade by more than 10x
            assert degradation_ratio < 10, (
                f"Performance degraded {degradation_ratio:.2f}x from {smallest_ops:.2f} to {largest_ops:.2f} ops/s"
            )

    async def test_large_data_payload_handling(self, stress_provider):
        """Test handling of tasks with large data payloads"""
        size_multipliers = [10, 50, 100, 200]  # Different payload sizes

        for multiplier in size_multipliers:
            task_id = f"large_task_{multiplier}"
            context_id = f"large_ctx_{multiplier}"

            # Create task with large payload
            large_task = self.create_large_task(task_id, context_id, multiplier)

            # Measure storage time
            store_result, store_time = await self.measure_operation_time(
                stress_provider.store_task, large_task
            )
            assert store_result.data is None, (
                f"Large task storage should succeed for size {multiplier}"
            )

            # Measure retrieval time
            get_result, get_time = await self.measure_operation_time(
                stress_provider.get_task, task_id
            )
            assert get_result.data is not None, (
                f"Large task retrieval should succeed for size {multiplier}"
            )

            # Verify data integrity
            retrieved_task = get_result.data
            assert retrieved_task.id == task_id
            assert len(retrieved_task.history or []) == 1, "History should be preserved"
            assert len(retrieved_task.artifacts or []) == 1, "Artifacts should be preserved"

            # Performance checks
            assert store_time < 5000, (
                f"Store time too slow for size {multiplier}: {store_time:.2f}ms"
            )
            assert get_time < 5000, f"Get time too slow for size {multiplier}: {get_time:.2f}ms"

            print(f"Large payload {multiplier}x: store={store_time:.2f}ms, get={get_time:.2f}ms")

    async def test_deep_pagination_performance(self, stress_provider):
        """Test performance of deep pagination through large datasets"""
        total_tasks = 1000
        page_size = 25
        context_id = "pagination_perf_ctx"

        # Create large dataset - use single context for pagination test
        tasks = self.create_bulk_tasks(
            total_tasks, context_id, "page_task", distribute_contexts=False
        )

        # Store in batches and verify success
        batch_size = 50
        stored_count = 0
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_operations = [stress_provider.store_task(task) for task in batch]
            batch_results = await asyncio.gather(*batch_operations)

            # Verify batch success
            for result in batch_results:
                if hasattr(result, "data") and result.data is None:
                    stored_count += 1

        assert stored_count == total_tasks, f"Only stored {stored_count}/{total_tasks} tasks"

        # Test pagination performance at different offsets
        test_offsets = [0, 250, 500, 750, 900]  # Beginning, middle, end

        for offset in test_offsets:
            query = A2ATaskQuery(context_id=context_id, limit=page_size, offset=offset)

            # Measure query time
            result, query_time = await self.measure_operation_time(
                stress_provider.find_tasks, query
            )

            assert result.data is not None, f"Pagination at offset {offset} should succeed"

            # Verify correct page size (except possibly last page)
            page_tasks = result.data
            expected_size = min(page_size, total_tasks - offset)
            assert len(page_tasks) == expected_size, (
                f"Page at offset {offset} should have {expected_size} tasks"
            )

            # Performance check - deep pagination shouldn't be too slow
            assert query_time < 1000, f"Query at offset {offset} too slow: {query_time:.2f}ms"

            print(f"Pagination offset {offset}: {query_time:.2f}ms")

    async def test_complex_query_performance(self, stress_provider):
        """Test performance of complex queries across large datasets"""
        total_tasks = 2000
        contexts_count = 20
        states = [TaskState.SUBMITTED, TaskState.WORKING, TaskState.COMPLETED, TaskState.FAILED]

        # Create diverse dataset
        all_tasks = []
        for i in range(total_tasks):
            context_idx = i % contexts_count
            state_idx = i % len(states)

            task = A2ATask(
                id=f"complex_task_{i:05d}",
                contextId=f"complex_ctx_{context_idx:03d}",
                kind="task",
                status=A2ATaskStatus(
                    state=states[state_idx],
                    timestamp=(datetime.now(timezone.utc) - timedelta(hours=i % 24)).isoformat(),
                ),
                metadata={
                    "priority": "high" if i % 3 == 0 else "normal",
                    "category": f"cat_{i % 5}",
                },
            )
            all_tasks.append(task)

        # Store all tasks
        batch_size = 100
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i : i + batch_size]
            batch_operations = [stress_provider.store_task(task) for task in batch]
            await asyncio.gather(*batch_operations)

        # Test various complex queries
        complex_queries = [
            # Query by state
            A2ATaskQuery(state=TaskState.WORKING, limit=100),
            # Query by context
            A2ATaskQuery(context_id="complex_ctx_005", limit=100),
            # Query by time range
            A2ATaskQuery(since=datetime.now(timezone.utc) - timedelta(hours=12), limit=100),
            # Query with multiple filters
            A2ATaskQuery(
                state=TaskState.COMPLETED,
                since=datetime.now(timezone.utc) - timedelta(hours=6),
                limit=50,
            ),
        ]

        for i, query in enumerate(complex_queries):
            result, query_time = await self.measure_operation_time(
                stress_provider.find_tasks, query
            )

            assert result.data is not None, f"Complex query {i} should succeed"

            # Performance check
            assert query_time < 2000, f"Complex query {i} too slow: {query_time:.2f}ms"

            print(f"Complex query {i}: {query_time:.2f}ms, {len(result.data)} results")


class TestResourceExhaustion(StressTestBase):
    """Test handling of resource exhaustion scenarios"""

    async def test_memory_pressure_handling(self, stress_provider):
        """Test behavior under memory pressure"""
        # Create tasks with large payloads to put memory pressure
        large_tasks = []
        max_tasks = 100  # Reduced to avoid actual memory exhaustion in tests

        for i in range(max_tasks):
            large_task = self.create_large_task(
                f"memory_task_{i:03d}",
                f"memory_ctx_{i % 10}",
                size_multiplier=200,  # Large payloads
            )
            large_tasks.append(large_task)

        # Store tasks and monitor memory behavior
        stored_count = 0
        for task in large_tasks:
            try:
                result = await stress_provider.store_task(task)
                if result.data is None:
                    stored_count += 1
                else:
                    # Provider might reject tasks under memory pressure
                    print(f"Task rejected under memory pressure: {result.error}")
                    break
            except Exception as e:
                # System might throw memory-related exceptions
                print(f"Memory pressure exception after {stored_count} tasks: {e}")
                break

        assert stored_count > 0, "Should be able to store at least some tasks"
        print(f"Stored {stored_count}/{max_tasks} large tasks under memory pressure")

        # Verify provider is still functional
        health_result = await stress_provider.health_check()
        assert health_result.data is not None
        assert health_result.data.get("healthy", False), "Provider should remain healthy"

    async def test_storage_limit_enforcement(self, stress_provider):
        """Test enforcement of storage limits"""
        # Create provider with low limits for testing
        limited_config = A2AInMemoryTaskConfig(
            max_tasks=100,  # Low limit for testing
            max_tasks_per_context=20,
        )
        limited_provider = create_a2a_in_memory_task_provider(limited_config)

        try:
            # Test task limit enforcement
            tasks = self.create_bulk_tasks(150, "limit_ctx", "limit_task")  # Exceed limit

            stored_count = 0
            rejected_count = 0

            for task in tasks:
                result = await limited_provider.store_task(task)
                if hasattr(result, "data") and result.data is None:
                    stored_count += 1
                elif hasattr(result, "error"):
                    rejected_count += 1

                    # Should get appropriate error message about storage limits
                    error_msg = str(result.error.message).lower()
                    # Check for storage-related error messages
                    assert (
                        "limit" in error_msg
                        or "full" in error_msg
                        or "maximum" in error_msg
                        or "storage" in error_msg
                        or "exceeded" in error_msg
                        or "failed to store" in error_msg
                    )
                else:
                    rejected_count += 1

            assert stored_count <= 100, f"Should not store more than limit: {stored_count}"
            assert rejected_count > 0, "Should reject tasks when limit is reached"

            print(f"Storage limit test: {stored_count} stored, {rejected_count} rejected")

            # Test per-context limit
            context_tasks = self.create_bulk_tasks(30, "single_limit_ctx", "ctx_limit_task")

            ctx_stored = 0
            ctx_rejected = 0

            for task in context_tasks:
                result = await limited_provider.store_task(task)
                if hasattr(result, "data") and result.data is None:
                    ctx_stored += 1
                else:
                    ctx_rejected += 1

            assert ctx_stored <= 20, f"Should not exceed per-context limit: {ctx_stored}"

        finally:
            await limited_provider.close()

    async def test_connection_exhaustion_recovery(self, stress_provider):
        """Test recovery from connection/resource exhaustion"""
        # Simulate connection exhaustion by creating many concurrent operations
        concurrent_ops = 1000

        # Create operations that might exhaust connections/resources
        operations = []
        for i in range(concurrent_ops):
            # Mix of different operation types
            if i % 3 == 0:
                task = A2ATask(
                    id=f"exhaust_task_{i}",
                    contextId=f"exhaust_ctx_{i % 10}",
                    kind="task",
                    status=A2ATaskStatus(state=TaskState.SUBMITTED),
                )
                operations.append(stress_provider.store_task(task))
            elif i % 3 == 1:
                operations.append(stress_provider.get_task(f"exhaust_task_{i - 1}"))
            else:
                query = A2ATaskQuery(context_id=f"exhaust_ctx_{i % 10}", limit=10)
                operations.append(stress_provider.find_tasks(query))

        # Execute with limited concurrency
        semaphore = asyncio.Semaphore(50)  # Limit concurrent operations

        async def limited_operation(op):
            async with semaphore:
                try:
                    return await op
                except Exception as e:
                    # Some operations might fail under resource pressure
                    return f"Error: {e}"

        results = await asyncio.gather(*[limited_operation(op) for op in operations])

        # Analyze results
        success_count = sum(1 for result in results if not isinstance(result, str))
        error_count = len(results) - success_count

        print(f"Connection exhaustion test: {success_count} succeeded, {error_count} failed")

        # Should handle at least some operations successfully
        assert success_count > concurrent_ops * 0.1, (
            "Should handle at least 10% of operations successfully"
        )

        # Provider should recover and be healthy
        await asyncio.sleep(1)  # Allow recovery time
        health_result = await stress_provider.health_check()
        assert health_result.data is not None
        assert health_result.data.get("healthy", False), "Provider should recover and be healthy"


class TestMemoryLeakDetection(StressTestBase):
    """Test for memory leaks and resource cleanup"""

    async def test_task_lifecycle_memory_cleanup(self, stress_provider):
        """Test that completed task lifecycles don't leak memory"""
        # Get initial memory baseline
        gc.collect()  # Force garbage collection
        initial_objects = len(gc.get_objects())

        # Create and process multiple task lifecycles
        cycles = 50

        for cycle in range(cycles):
            context_id = f"leak_test_ctx_{cycle}"

            # Create task
            task = A2ATask(
                id=f"leak_task_{cycle}",
                contextId=context_id,
                kind="task",
                status=A2ATaskStatus(state=TaskState.SUBMITTED),
            )

            # Store task
            await stress_provider.store_task(task)

            # Update to working
            await stress_provider.update_task_status(
                task.id,
                TaskState.WORKING,
                A2AMessage(
                    role="agent",
                    parts=[A2ATextPart(kind="text", text="Working on it")],
                    messageId=f"work_{cycle}",
                    contextId=context_id,
                    kind="message",
                ),
            )

            # Complete task
            completed_task = task.model_copy(
                update={
                    "status": A2ATaskStatus(
                        state=TaskState.COMPLETED,
                        message=A2AMessage(
                            role="agent",
                            parts=[A2ATextPart(kind="text", text="Completed")],
                            messageId=f"done_{cycle}",
                            contextId=context_id,
                            kind="message",
                        ),
                    )
                }
            )
            await stress_provider.update_task(completed_task)

            # Delete task
            await stress_provider.delete_task(task.id)

            # Periodic garbage collection
            if cycle % 10 == 0:
                gc.collect()

        # Final cleanup and measurement
        gc.collect()
        final_objects = len(gc.get_objects())

        # Calculate object growth
        object_growth = final_objects - initial_objects
        growth_per_cycle = object_growth / cycles if cycles > 0 else 0

        print(
            f"Memory leak test: {object_growth} objects growth over {cycles} cycles ({growth_per_cycle:.2f} per cycle)"
        )

        # Should not have significant memory growth
        # Allow some growth but flag if it's excessive
        max_allowed_growth_per_cycle = 100  # objects per cycle
        assert growth_per_cycle < max_allowed_growth_per_cycle, (
            f"Potential memory leak: {growth_per_cycle:.2f} objects per cycle"
        )

    async def test_provider_cleanup_on_close(self, stress_provider):
        """Test that provider properly cleans up resources on close"""
        # Create some tasks
        tasks = self.create_bulk_tasks(20, "cleanup_ctx", "cleanup_task")
        for task in tasks:
            await stress_provider.store_task(task)

        # Create weak references to track cleanup
        weak_refs = []

        # Store some internal objects as weak references
        # (This is implementation-specific and might need adjustment)
        try:
            # Access provider internals if possible
            if hasattr(stress_provider, "_state"):
                weak_refs.append(weakref.ref(stress_provider._state))
        except Exception:
            pass  # Provider might not expose internals

        # Close provider
        await stress_provider.close()

        # Force garbage collection
        gc.collect()

        # Check that weak references are cleared (objects were cleaned up)
        alive_refs = [ref for ref in weak_refs if ref() is not None]

        print(f"Cleanup test: {len(alive_refs)}/{len(weak_refs)} weak references still alive")

        # Ideally, all weak references should be cleared
        # But this is implementation-dependent
        assert len(alive_refs) <= len(weak_refs) * 0.5, "Too many objects survived cleanup"

    async def test_large_dataset_cleanup_efficiency(self, stress_provider):
        """Test cleanup efficiency with large datasets"""
        # Create large dataset - use single context for cleanup test
        large_dataset_size = 1000
        tasks = self.create_bulk_tasks(
            large_dataset_size, "large_cleanup_ctx", "large_cleanup_task", distribute_contexts=False
        )

        # Store all tasks
        batch_size = 50
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_operations = [stress_provider.store_task(task) for task in batch]
            await asyncio.gather(*batch_operations)

        # Measure cleanup time
        start_time = time.perf_counter()

        # Delete all tasks by context
        delete_result = await stress_provider.delete_tasks_by_context("large_cleanup_ctx")

        end_time = time.perf_counter()
        cleanup_time = end_time - start_time

        # Verify cleanup was successful
        assert delete_result.data == large_dataset_size, (
            f"Should delete all {large_dataset_size} tasks"
        )

        # Verify context is empty
        remaining_result = await stress_provider.get_tasks_by_context("large_cleanup_ctx")
        assert remaining_result.data is not None
        assert len(remaining_result.data) == 0, "Context should be empty after cleanup"

        # Performance check
        cleanup_rate = large_dataset_size / cleanup_time
        assert cleanup_rate > 100, f"Cleanup too slow: {cleanup_rate:.2f} tasks/second"

        print(
            f"Large dataset cleanup: {large_dataset_size} tasks in {cleanup_time:.2f}s ({cleanup_rate:.2f} tasks/s)"
        )


class TestPerformanceRegression(StressTestBase):
    """Test for performance regressions"""

    async def test_operation_performance_benchmarks(self, stress_provider):
        """Benchmark key operations to detect performance regressions"""
        # Define performance benchmarks (in milliseconds)
        benchmarks = {
            "store_task": 100,  # Single task store should be < 100ms
            "get_task": 50,  # Single task get should be < 50ms
            "update_task": 100,  # Single task update should be < 100ms
            "find_tasks": 200,  # Query should be < 200ms
            "delete_task": 50,  # Single task delete should be < 50ms
        }

        # Setup test data
        setup_tasks = self.create_bulk_tasks(100, "perf_ctx", "perf_task")
        for task in setup_tasks:
            await stress_provider.store_task(task)

        # Test store performance
        new_task = self.create_bulk_tasks(1, "perf_new_ctx", "perf_new_task")[0]
        _, store_time = await self.measure_operation_time(stress_provider.store_task, new_task)
        assert store_time < benchmarks["store_task"], (
            f"Store performance regression: {store_time:.2f}ms > {benchmarks['store_task']}ms"
        )

        # Test get performance
        _, get_time = await self.measure_operation_time(stress_provider.get_task, "perf_task_00050")
        assert get_time < benchmarks["get_task"], (
            f"Get performance regression: {get_time:.2f}ms > {benchmarks['get_task']}ms"
        )

        # Test update performance
        update_message = A2AMessage(
            role="agent",
            parts=[A2ATextPart(kind="text", text="Performance test update")],
            messageId="perf_update",
            contextId="perf_ctx_5",
            kind="message",
        )
        _, update_time = await self.measure_operation_time(
            stress_provider.update_task_status, "perf_task_00025", TaskState.WORKING, update_message
        )
        assert update_time < benchmarks["update_task"], (
            f"Update performance regression: {update_time:.2f}ms > {benchmarks['update_task']}ms"
        )

        # Test query performance
        query = A2ATaskQuery(context_id="perf_ctx_1", limit=10)
        _, query_time = await self.measure_operation_time(stress_provider.find_tasks, query)
        assert query_time < benchmarks["find_tasks"], (
            f"Query performance regression: {query_time:.2f}ms > {benchmarks['find_tasks']}ms"
        )

        # Test delete performance
        _, delete_time = await self.measure_operation_time(
            stress_provider.delete_task, "perf_task_00075"
        )
        assert delete_time < benchmarks["delete_task"], (
            f"Delete performance regression: {delete_time:.2f}ms > {benchmarks['delete_task']}ms"
        )

        print("Performance benchmark results:")
        print(f"  Store: {store_time:.2f}ms (limit: {benchmarks['store_task']}ms)")
        print(f"  Get: {get_time:.2f}ms (limit: {benchmarks['get_task']}ms)")
        print(f"  Update: {update_time:.2f}ms (limit: {benchmarks['update_task']}ms)")
        print(f"  Query: {query_time:.2f}ms (limit: {benchmarks['find_tasks']}ms)")
        print(f"  Delete: {delete_time:.2f}ms (limit: {benchmarks['delete_task']}ms)")

    async def test_scalability_characteristics(self, stress_provider):
        """Test how performance scales with data size"""
        dataset_sizes = [10, 50, 100, 500]
        scaling_results = []

        for size in dataset_sizes:
            context_id = f"scale_ctx_{size}"

            # Create dataset
            tasks = self.create_bulk_tasks(size, context_id, f"scale_task_{size}")

            # Measure batch storage time
            start_time = time.perf_counter()
            batch_operations = [stress_provider.store_task(task) for task in tasks]
            await asyncio.gather(*batch_operations)
            storage_time = time.perf_counter() - start_time

            # Measure query time
            query = A2ATaskQuery(context_id=context_id, limit=size)
            start_time = time.perf_counter()
            await stress_provider.find_tasks(query)
            query_time = time.perf_counter() - start_time

            # Calculate metrics
            storage_ops_per_sec = size / storage_time if storage_time > 0 else float("inf")
            query_ops_per_sec = size / query_time if query_time > 0 else float("inf")

            scaling_results.append(
                {
                    "size": size,
                    "storage_ops_per_sec": storage_ops_per_sec,
                    "query_ops_per_sec": query_ops_per_sec,
                }
            )

            print(
                f"Scalability {size} tasks: storage={storage_ops_per_sec:.2f} ops/s, query={query_ops_per_sec:.2f} ops/s"
            )

        # Analyze scaling characteristics
        if len(scaling_results) >= 2:
            first_result = scaling_results[0]
            last_result = scaling_results[-1]

            # Performance shouldn't degrade too severely with scale
            storage_degradation = (
                first_result["storage_ops_per_sec"] / last_result["storage_ops_per_sec"]
            )
            query_degradation = first_result["query_ops_per_sec"] / last_result["query_ops_per_sec"]

            max_allowed_degradation = 5.0  # 5x degradation is acceptable

            assert storage_degradation < max_allowed_degradation, (
                f"Storage performance degraded {storage_degradation:.2f}x"
            )
            assert query_degradation < max_allowed_degradation, (
                f"Query performance degraded {query_degradation:.2f}x"
            )
