#!/usr/bin/env python3
"""
Quick validation test for A2A memory system
"""

import asyncio
import sys

sys.path.append('.')

from datetime import datetime, timezone

from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
from jaf.a2a.memory.serialization import deserialize_a2a_task, serialize_a2a_task
from jaf.a2a.memory.types import A2AInMemoryTaskConfig
from jaf.a2a.types import A2AMessage, A2ATask, A2ATaskStatus, A2ATextPart, TaskState


async def test_serialization():
    """Test basic serialization functionality"""
    print("🧪 Testing Serialization...")

    task = A2ATask(
        id='test_task_001',
        contextId='test_ctx_001',
        kind='task',
        status=A2ATaskStatus(
            state=TaskState.SUBMITTED,
            message=A2AMessage(
                role='user',
                parts=[A2ATextPart(kind='text', text='Test message')],
                messageId='test_msg_001',
                contextId='test_ctx_001',
                kind='message'
            ),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )

    # Test serialization
    serialize_result = serialize_a2a_task(task)
    assert serialize_result.data is not None, "Serialization should succeed"

    # Test deserialization
    deserialize_result = deserialize_a2a_task(serialize_result.data)
    assert deserialize_result.data is not None, "Deserialization should succeed"

    # Verify round-trip integrity
    original = task
    round_trip = deserialize_result.data

    assert original.id == round_trip.id, "ID should be preserved"
    assert original.context_id == round_trip.context_id, "Context ID should be preserved"
    assert original.status.state == round_trip.status.state, "State should be preserved"

    print("✅ Serialization test PASSED")
    return True

async def test_provider_lifecycle():
    """Test provider lifecycle functionality"""
    print("🧪 Testing Provider Lifecycle...")

    # Create provider
    config = A2AInMemoryTaskConfig(max_tasks=100, max_tasks_per_context=50)
    provider = create_a2a_in_memory_task_provider(config)

    try:
        # Create test task
        task = A2ATask(
            id='lifecycle_test_001',
            contextId='lifecycle_ctx_001',
            kind='task',
            status=A2ATaskStatus(
                state=TaskState.SUBMITTED,
                message=A2AMessage(
                    role='user',
                    parts=[A2ATextPart(kind='text', text='Please help me with this task')],
                    messageId='submit_msg_001',
                    contextId='lifecycle_ctx_001',
                    kind='message'
                ),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )

        # Store task
        store_result = await provider.store_task(task)
        assert store_result.data is None, "Store should succeed"  # Success returns None data

        # Retrieve task
        get_result = await provider.get_task('lifecycle_test_001')
        assert get_result.data is not None, "Get should return task"

        retrieved_task = get_result.data
        assert retrieved_task.id == 'lifecycle_test_001', "Task ID should match"
        assert retrieved_task.status.state == TaskState.SUBMITTED, "Task state should match"

        # Update task status
        update_result = await provider.update_task_status(
            'lifecycle_test_001',
            TaskState.WORKING,
            A2AMessage(
                role='agent',
                parts=[A2ATextPart(kind='text', text='Starting to work on your request')],
                messageId='working_msg_001',
                contextId='lifecycle_ctx_001',
                kind='message'
            )
        )

        assert update_result.data is None, "Update should succeed"

        # Verify update
        updated_result = await provider.get_task('lifecycle_test_001')
        updated_task = updated_result.data
        assert updated_task.status.state == TaskState.WORKING, "Task state should be updated"

        # Test context queries
        context_result = await provider.get_tasks_by_context('lifecycle_ctx_001')
        context_tasks = context_result.data or []
        assert len(context_tasks) == 1, f"Expected 1 task in context, got {len(context_tasks)}"

        # Test health check
        health_result = await provider.health_check()
        assert health_result.data is not None, "Health check should return data"
        assert health_result.data.get('healthy', False), "Provider should be healthy"

        print("✅ Provider lifecycle test PASSED")
        return True

    finally:
        await provider.close()

async def test_concurrency():
    """Test basic concurrency scenarios"""
    print("🧪 Testing Basic Concurrency...")

    config = A2AInMemoryTaskConfig(max_tasks=1000)
    provider = create_a2a_in_memory_task_provider(config)

    try:
        # Create multiple tasks concurrently
        tasks = []
        for i in range(10):
            task = A2ATask(
                id=f'concurrent_task_{i:03d}',
                contextId=f'concurrent_ctx_{i % 3}',  # Distribute across 3 contexts
                kind='task',
                status=A2ATaskStatus(
                    state=TaskState.SUBMITTED,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            )
            tasks.append(task)

        # Store all tasks concurrently
        store_operations = [provider.store_task(task) for task in tasks]
        results = await asyncio.gather(*store_operations)

        # All stores should succeed
        for i, result in enumerate(results):
            assert result.data is None, f"Store operation {i} should succeed"

        # Retrieve all tasks concurrently
        get_operations = [provider.get_task(f'concurrent_task_{i:03d}') for i in range(10)]
        get_results = await asyncio.gather(*get_operations)

        # All gets should succeed
        for i, result in enumerate(get_results):
            assert result.data is not None, f"Get operation {i} should return task"
            assert result.data.id == f'concurrent_task_{i:03d}', f"Task {i} ID should match"

        print("✅ Basic concurrency test PASSED")
        return True

    finally:
        await provider.close()

async def main():
    """Run all validation tests"""
    print("🚀 A2A Memory System Validation")
    print("=" * 50)

    try:
        # Run all tests
        await test_serialization()
        await test_provider_lifecycle()
        await test_concurrency()

        print("")
        print("🎉 ALL VALIDATION TESTS PASSED")
        print("✅ A2A Memory System is working correctly")
        return True

    except Exception as e:
        print("")
        print(f"❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
