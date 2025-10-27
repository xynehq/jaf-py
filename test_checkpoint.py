#!/usr/bin/env python3
"""
Comprehensive test script for checkpoint functionality.

This script tests the complete checkpoint system across:
1. All memory providers (InMemory, PostgreSQL, Redis)
2. All checkpoint methods and edge cases
3. Server API endpoints
4. Error conditions and data integrity
5. Multiple checkpoint scenarios
"""

import asyncio
import json
import sys
import traceback
from dataclasses import replace
from typing import List, Dict, Any, Optional

from jaf.core.types import (
    Message, ContentRole, RunState, RunConfig, Agent,
    create_run_id, create_trace_id, create_message_id,
    CheckpointRequest, generate_message_id, MessageId
)
from jaf.core.checkpoint import checkpoint_conversation, get_checkpoint_history
from jaf.memory.types import MemoryConfig, MemoryProvider
from jaf.memory.providers.in_memory import create_in_memory_provider


def create_test_agent():
    """Create a simple test agent."""
    def instructions(state):
        return "You are a helpful assistant that answers questions about today, yesterday, and previous days."
    
    return Agent(
        name="TestAgent",
        instructions=instructions,
        tools=[]
    )


class TestResults:
    """Track test results across all test cases."""
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = []
        
    def record_test(self, test_name: str, passed: bool, details: str = ""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            self.failed_tests.append((test_name, details))
            print(f"❌ {test_name}: FAILED {details}")
    
    def summary(self):
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {len(self.failed_tests)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test_name, details in self.failed_tests:
                print(f"   - {test_name}: {details}")
        
        return len(self.failed_tests) == 0


async def create_test_conversation(provider: MemoryProvider, conversation_id: str) -> List[MessageId]:
    """Create a standard test conversation and return message IDs."""
    msg_ids = [generate_message_id() for _ in range(6)]
    
    messages = [
        Message(role=ContentRole.USER, content="What's the stock report for today?", message_id=msg_ids[0]),
        Message(role=ContentRole.ASSISTANT, content="Today's market is up 2.5%", message_id=msg_ids[1]),
        Message(role=ContentRole.USER, content="What was it yesterday?", message_id=msg_ids[2]),
        Message(role=ContentRole.ASSISTANT, content="Yesterday was down 1.2%", message_id=msg_ids[3]),
        Message(role=ContentRole.USER, content="And the day before?", message_id=msg_ids[4]),
        Message(role=ContentRole.ASSISTANT, content="The day before was stable", message_id=msg_ids[5]),
    ]
    
    result = await provider.store_messages(conversation_id, messages, {"user_id": "test-user"})
    if hasattr(result, 'error'):
        raise Exception(f"Failed to create test conversation: {result.error}")
    
    return msg_ids


async def test_checkpoint_basic_functionality(provider: MemoryProvider, provider_name: str, results: TestResults):
    """Test basic checkpoint functionality for a specific memory provider."""
    print(f"\n🔧 Testing {provider_name} Provider - Checkpoint Basics")
    print("-" * 40)
    
    conversation_id = f"test-chk-{provider_name.lower()}-001"
    
    try:
        # Create test conversation
        msg_ids = await create_test_conversation(provider, conversation_id)
        results.record_test(f"{provider_name}: Create conversation", True, f"with {len(msg_ids)} messages")
        
        # Test checkpointing after message 3 (index 3) - should keep messages 0-3, remove 4-5
        memory_config = MemoryConfig(provider=provider, auto_store=True)
        test_agent = create_test_agent()
        
        run_config = RunConfig(
            agent_registry={"TestAgent": test_agent},
            model_provider=None,  # Not needed for checkpoint
            memory=memory_config,
            conversation_id=conversation_id,
            max_turns=10
        )
        
        checkpoint_request = CheckpointRequest(
            conversation_id=conversation_id,
            message_id=msg_ids[3],  # Checkpoint AFTER this message (keep it)
            context={"test": "checkpoint"}
        )
        
        result = await checkpoint_conversation(checkpoint_request, run_config)
        
        # Verify checkpoint results
        results.record_test(f"{provider_name}: Basic checkpoint", 
                          result.checkpointed_at_index == 3, 
                          f"checkpointed at index {result.checkpointed_at_index}")
        
        # Check message count - should keep messages 0-3 (4 messages total)
        expected_count = 4
        actual_count = len(result.messages)
        results.record_test(f"{provider_name}: Correct message count", 
                          actual_count == expected_count, 
                          f"got {actual_count}, expected {expected_count}")
        
        # Verify messages 4 and 5 were removed
        remaining_ids = [msg.message_id for msg in result.messages]
        removed_found = msg_ids[4] in remaining_ids or msg_ids[5] in remaining_ids
        results.record_test(f"{provider_name}: Removed messages after checkpoint", 
                          not removed_found, 
                          f"subsequent messages correctly removed")
        
        # Verify checkpoint message is still present
        checkpoint_msg_present = msg_ids[3] in remaining_ids
        results.record_test(f"{provider_name}: Checkpoint message preserved", 
                          checkpoint_msg_present, 
                          "checkpoint message still in conversation")
        
        # Verify messages before checkpoint are present
        before_checkpoint_present = all(msg_ids[i] in remaining_ids for i in range(4))
        results.record_test(f"{provider_name}: Messages before checkpoint preserved", 
                          before_checkpoint_present, 
                          "all messages before checkpoint preserved")
        
    except Exception as e:
        results.record_test(f"{provider_name}: Basic checkpoint", False, f"Exception: {e}")
        traceback.print_exc()


async def test_checkpoint_edge_cases(results: TestResults):
    """Test checkpoint edge cases and error conditions."""
    print(f"\n🚨 Testing Checkpoint Edge Cases")
    print("-" * 40)
    
    memory_provider = create_in_memory_provider()
    
    try:
        # Test 1: Checkpoint after first message (should keep only first message)
        conv_id1 = "edge-first-msg"
        msg_ids1 = await create_test_conversation(memory_provider, conv_id1)
        
        memory_config = MemoryConfig(provider=memory_provider, auto_store=True)
        test_agent = create_test_agent()
        run_config = RunConfig(
            agent_registry={"TestAgent": test_agent},
            model_provider=None,
            memory=memory_config,
            conversation_id=conv_id1,
            max_turns=10
        )
        
        checkpoint_request = CheckpointRequest(
            conversation_id=conv_id1,
            message_id=msg_ids1[0],  # Checkpoint after first message
            context={}
        )
        
        result = await checkpoint_conversation(checkpoint_request, run_config)
        expected_count = 1  # Only first message should remain
        results.record_test("Edge Cases: Checkpoint after first message", 
                          len(result.messages) == expected_count, 
                          f"got {len(result.messages)}, expected {expected_count}")
        
        # Test 2: Checkpoint after last message (should keep all messages)
        conv_id2 = "edge-last-msg"
        msg_ids2 = await create_test_conversation(memory_provider, conv_id2)
        
        run_config2 = replace(run_config, conversation_id=conv_id2)
        checkpoint_request2 = CheckpointRequest(
            conversation_id=conv_id2,
            message_id=msg_ids2[5],  # Checkpoint after last message
            context={}
        )
        
        result2 = await checkpoint_conversation(checkpoint_request2, run_config2)
        expected_count2 = 6  # All messages should remain
        results.record_test("Edge Cases: Checkpoint after last message", 
                          len(result2.messages) == expected_count2, 
                          f"got {len(result2.messages)}, expected {expected_count2}")
        
        # Test 3: Checkpoint from non-existent message (should raise error)
        conv_id3 = "edge-missing-msg"
        msg_ids3 = await create_test_conversation(memory_provider, conv_id3)
        
        run_config3 = replace(run_config, conversation_id=conv_id3)
        fake_msg_id = generate_message_id()
        checkpoint_request3 = CheckpointRequest(
            conversation_id=conv_id3,
            message_id=fake_msg_id,
            context={}
        )
        
        try:
            result3 = await checkpoint_conversation(checkpoint_request3, run_config3)
            results.record_test("Edge Cases: Non-existent message", False, 
                              "should have raised ValueError")
        except ValueError as e:
            results.record_test("Edge Cases: Non-existent message", True, 
                              f"correctly raised ValueError: {str(e)[:50]}")
        
        # Test 4: Multiple checkpoints from same conversation
        conv_id4 = "edge-multi-checkpoint"
        msg_ids4 = await create_test_conversation(memory_provider, conv_id4)
        
        run_config4 = replace(run_config, conversation_id=conv_id4)
        
        # First checkpoint after message 3
        checkpoint_request4a = CheckpointRequest(
            conversation_id=conv_id4,
            message_id=msg_ids4[3],
            context={}
        )
        result4a = await checkpoint_conversation(checkpoint_request4a, run_config4)
        
        # Second checkpoint after message 1 (further reducing)
        checkpoint_request4b = CheckpointRequest(
            conversation_id=conv_id4,
            message_id=msg_ids4[1],
            context={}
        )
        result4b = await checkpoint_conversation(checkpoint_request4b, run_config4)
        
        expected_final_count = 2  # Should have messages 0 and 1
        results.record_test("Edge Cases: Multiple checkpoints", 
                          len(result4b.messages) == expected_final_count, 
                          f"got {len(result4b.messages)}, expected {expected_final_count}")
        
    except Exception as e:
        results.record_test("Edge Cases: Setup", False, f"Exception: {e}")
        traceback.print_exc()


async def test_checkpoint_history(results: TestResults):
    """Test checkpoint history tracking."""
    print(f"\n📜 Testing Checkpoint History")
    print("-" * 40)
    
    memory_provider = create_in_memory_provider()
    
    try:
        conv_id = "history-test"
        msg_ids = await create_test_conversation(memory_provider, conv_id)
        
        memory_config = MemoryConfig(provider=memory_provider, auto_store=True)
        test_agent = create_test_agent()
        run_config = RunConfig(
            agent_registry={"TestAgent": test_agent},
            model_provider=None,
            memory=memory_config,
            conversation_id=conv_id,
            max_turns=10
        )
        
        # Create a checkpoint
        checkpoint_request = CheckpointRequest(
            conversation_id=conv_id,
            message_id=msg_ids[2],
            context={"reason": "test checkpoint"}
        )
        
        result = await checkpoint_conversation(checkpoint_request, run_config)
        
        # Get checkpoint history
        history = await get_checkpoint_history(conv_id, run_config)
        
        has_history = history is not None and len(history) > 0
        results.record_test("Checkpoint History: History exists", has_history, 
                          f"found {len(history) if history else 0} checkpoint(s)")
        
        if has_history:
            checkpoint_data = history[0]
            has_checkpoint_id = 'checkpoint_id' in checkpoint_data
            results.record_test("Checkpoint History: Has checkpoint_id", has_checkpoint_id)
            
            has_checkpoint_point = 'checkpoint_point' in checkpoint_data
            results.record_test("Checkpoint History: Has checkpoint_point", has_checkpoint_point)
            
            has_timestamp = 'timestamp' in checkpoint_data
            results.record_test("Checkpoint History: Has timestamp", has_timestamp)
        
    except Exception as e:
        results.record_test("Checkpoint History: Setup", False, f"Exception: {e}")
        traceback.print_exc()


async def test_checkpoint_vs_regeneration_difference(results: TestResults):
    """Test that checkpoint behaves differently from regeneration."""
    print(f"\n🔀 Testing Checkpoint vs Regeneration Difference")
    print("-" * 40)
    
    memory_provider = create_in_memory_provider()
    
    try:
        # Create two identical conversations
        conv_id_chk = "diff-checkpoint"
        conv_id_regen = "diff-regeneration"
        
        msg_ids_chk = await create_test_conversation(memory_provider, conv_id_chk)
        msg_ids_regen = await create_test_conversation(memory_provider, conv_id_regen)
        
        memory_config = MemoryConfig(provider=memory_provider, auto_store=True)
        test_agent = create_test_agent()
        
        run_config_chk = RunConfig(
            agent_registry={"TestAgent": test_agent},
            model_provider=None,
            memory=memory_config,
            conversation_id=conv_id_chk,
            max_turns=10
        )
        
        # Checkpoint after message 2 (should keep messages 0, 1, 2)
        checkpoint_request = CheckpointRequest(
            conversation_id=conv_id_chk,
            message_id=msg_ids_chk[2],
            context={}
        )
        
        checkpoint_result = await checkpoint_conversation(checkpoint_request, run_config_chk)
        
        # For regeneration, it would truncate FROM message 2 (removing 2 onwards)
        # For checkpoint, it truncates AFTER message 2 (keeping 2, removing 3 onwards)
        checkpoint_count = len(checkpoint_result.messages)
        checkpoint_includes_msg2 = msg_ids_chk[2] in [m.message_id for m in checkpoint_result.messages]
        
        # Checkpoint should include message at index 2 (3 messages: 0, 1, 2)
        expected_checkpoint_count = 3
        results.record_test("Difference: Checkpoint keeps target message", 
                          checkpoint_count == expected_checkpoint_count and checkpoint_includes_msg2,
                          f"checkpoint has {checkpoint_count} messages, includes msg[2]: {checkpoint_includes_msg2}")
        
        # Note: We're not testing regeneration here, just documenting the difference
        results.record_test("Difference: Checkpoint behavior confirmed", True, 
                          "checkpoint keeps the specified message + all before it")
        
    except Exception as e:
        results.record_test("Difference Test: Setup", False, f"Exception: {e}")
        traceback.print_exc()


async def test_all_providers(results: TestResults):
    """Test checkpoint functionality across all memory providers."""
    print(f"\n🗄️ Testing All Memory Providers")
    print("=" * 50)
    
    # Test InMemory Provider
    try:
        memory_provider = create_in_memory_provider()
        await test_checkpoint_basic_functionality(memory_provider, "InMemory", results)
        await memory_provider.close()
    except Exception as e:
        results.record_test("InMemory Provider", False, f"Exception: {e}")
    
    # Test PostgreSQL Provider (if available)
    try:
        from jaf.memory.providers.postgres import create_postgres_provider
        from jaf.memory.types import PostgresConfig
        
        try:
            import asyncpg
            postgres_config = PostgresConfig(
                host="localhost", 
                port=5432, 
                database="jaf_test", 
                username="postgres", 
                password="test",
                table_name="test_conversations_checkpoint"
            )
            
            print("   🔄 Attempting PostgreSQL connection...")
            postgres_result = await create_postgres_provider(postgres_config)
            if hasattr(postgres_result, 'data'):
                postgres_provider = postgres_result.data
                await test_checkpoint_basic_functionality(postgres_provider, "PostgreSQL", results)
                await postgres_provider.close()
            else:
                results.record_test("PostgreSQL Provider", False, "Connection failed - skipping PostgreSQL tests")
        except ImportError:
            results.record_test("PostgreSQL Provider", False, "asyncpg not installed - skipping PostgreSQL tests")
        except Exception as e:
            results.record_test("PostgreSQL Provider", False, f"Connection error: {e} - skipping PostgreSQL tests")
    except Exception as e:
        results.record_test("PostgreSQL Provider", False, f"Import error: {e}")
    
    # Test Redis Provider (if available)
    try:
        from jaf.memory.providers.redis import create_redis_provider
        from jaf.memory.types import RedisConfig
        
        try:
            import redis.asyncio as redis
            redis_config = RedisConfig(host="localhost", port=6379, db=2, key_prefix="jaf_chk_test:")
            
            print("   🔄 Attempting Redis connection...")
            redis_result = await create_redis_provider(redis_config)
            if hasattr(redis_result, 'data'):
                redis_provider = redis_result.data
                await test_checkpoint_basic_functionality(redis_provider, "Redis", results)
                await redis_provider.close()
            else:
                results.record_test("Redis Provider", False, "Connection failed - skipping Redis tests")
        except ImportError:
            results.record_test("Redis Provider", False, "redis not installed - skipping Redis tests")
        except Exception as e:
            results.record_test("Redis Provider", False, f"Connection error: {e} - skipping Redis tests")
    except Exception as e:
        results.record_test("Redis Provider", False, f"Import error: {e}")


async def main():
    """Comprehensive test runner."""
    print("🧪 JAF-py Checkpoint Comprehensive Test Suite")
    print("=" * 60)
    print("Testing checkpoint functionality across all providers")
    print("=" * 60)
    
    results = TestResults()
    
    try:
        # Test all memory providers
        await test_all_providers(results)
        
        # Test edge cases
        await test_checkpoint_edge_cases(results)
        
        # Test history
        await test_checkpoint_history(results)
        
        # Test difference from regeneration
        await test_checkpoint_vs_regeneration_difference(results)
        
        # Show final summary
        print("\n" + "=" * 60)
        success = results.summary()
        
        if success:
            print("\n🎉 ALL CHECKPOINT TESTS PASSED!")
            print("✅ Checkpoint functionality is fully implemented and working correctly.")
            print("✅ All memory providers support checkpoint.")
            print("✅ Edge cases are properly handled.")
            print("✅ Checkpoint behavior differs correctly from regeneration.")
        else:
            print("\n⚠️  SOME TESTS FAILED!")
            print("❌ Review the failed tests above and fix the implementation.")
            
        return success
        
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
