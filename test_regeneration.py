#!/usr/bin/env python3
"""
Comprehensive test script for regeneration functionality.

This script tests the complete regeneration system across:
1. All memory providers (InMemory, PostgreSQL, Redis)
2. All regeneration methods and edge cases
3. Server API endpoints
4. Error conditions and data integrity
5. Multiple regeneration scenarios
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
    RegenerationRequest, generate_message_id, MessageId
)
from jaf.core.regeneration import regenerate_conversation, get_regeneration_points
from jaf.memory.types import MemoryConfig, MemoryProvider
from jaf.memory.providers.in_memory import create_in_memory_provider


class MockModelProvider:
    """Mock model provider for testing."""
    
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
    
    async def get_completion(self, state, agent, config):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return {"message": {"content": response}}
        return {"message": {"content": "Default response"}}


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


async def test_memory_provider_methods(provider: MemoryProvider, provider_name: str, results: TestResults):
    """Test all regeneration methods for a specific memory provider."""
    print(f"\n🔧 Testing {provider_name} Provider Methods")
    print("-" * 40)
    
    conversation_id = f"test-{provider_name.lower()}-001"
    
    try:
        # Create test conversation and store message IDs
        msg_ids = await create_test_conversation(provider, conversation_id)
        results.record_test(f"{provider_name}: Create conversation", True, f"with {len(msg_ids)} messages")
        
        # Test 1: truncate_conversation_after (should remove messages from index 4 onwards)
        truncate_result = await provider.truncate_conversation_after(conversation_id, msg_ids[4])
        if hasattr(truncate_result, 'data'):
            removed_count = truncate_result.data
            results.record_test(f"{provider_name}: truncate_conversation_after", 
                              removed_count == 2, f"removed {removed_count} messages (expected 2)")
        else:
            results.record_test(f"{provider_name}: truncate_conversation_after", False, "method failed")
        
        # Test 2: get_conversation_until_message (use fresh conversation for this test)
        conversation_id2 = f"test-{provider_name.lower()}-002"
        msg_ids2 = await create_test_conversation(provider, conversation_id2)  # Use fresh conversation
        until_result = await provider.get_conversation_until_message(conversation_id2, msg_ids2[2])
        if hasattr(until_result, 'data') and until_result.data:
            until_messages = until_result.data.messages
            results.record_test(f"{provider_name}: get_conversation_until_message", 
                              len(until_messages) == 2, f"got {len(until_messages)} messages (expected 2)")
        else:
            results.record_test(f"{provider_name}: get_conversation_until_message", False, "method failed")
        
        # Test 3: mark_regeneration_point (use the fresh conversation)
        mark_result = await provider.mark_regeneration_point(
            conversation_id2, msg_ids2[2], {"test": "regeneration_point"}
        )
        success = hasattr(mark_result, 'data') or not hasattr(mark_result, 'error')
        results.record_test(f"{provider_name}: mark_regeneration_point", success)
        
        # Test 4: Edge case - non-existent message
        fake_id = generate_message_id()
        truncate_result = await provider.truncate_conversation_after(conversation_id, fake_id)
        if hasattr(truncate_result, 'data'):
            removed_count = truncate_result.data
            results.record_test(f"{provider_name}: truncate non-existent message", 
                              removed_count == 0, "correctly handled missing message")
        else:
            results.record_test(f"{provider_name}: truncate non-existent message", False, "failed to handle missing message")
        
        # Test 5: Edge case - non-existent conversation
        fake_conv_id = "non-existent-conversation"
        truncate_result = await provider.truncate_conversation_after(fake_conv_id, msg_ids2[0])
        has_error = hasattr(truncate_result, 'error')
        results.record_test(f"{provider_name}: truncate non-existent conversation", 
                          has_error, "correctly returned error for missing conversation")
        
    except Exception as e:
        results.record_test(f"{provider_name}: Provider test setup", False, f"Exception: {e}")


async def test_regeneration_engine(results: TestResults):
    """Test the core regeneration engine functionality."""
    print(f"\n⚙️ Testing Regeneration Engine")
    print("-" * 40)
    
    # Create memory provider
    memory_provider = create_in_memory_provider()
    memory_config = MemoryConfig(provider=memory_provider, auto_store=True)
    
    # Create test setup
    model_provider = MockModelProvider(["Regenerated response for testing"])
    test_agent = create_test_agent()
    
    run_config = RunConfig(
        agent_registry={"TestAgent": test_agent},
        model_provider=model_provider,
        memory=memory_config,
        conversation_id="test-engine-001",
        max_turns=10
    )
    
    try:
        # Create initial conversation
        msg_ids = await create_test_conversation(memory_provider, "test-engine-001")
        
        # Test regeneration from 3rd message (should remove 4th, 5th, 6th)
        regeneration_request = RegenerationRequest(
            conversation_id="test-engine-001",
            message_id=msg_ids[2],  # "What was it yesterday?"
            context={"test": "regeneration"}
        )
        
        result = await regenerate_conversation(
            regeneration_request, run_config, {"user_id": "test-user"}, "TestAgent"
        )
        
        # Verify regeneration results
        success = result.outcome.status == 'completed'
        results.record_test("Regeneration Engine: Basic regeneration", success, f"status: {result.outcome.status}")
        
        # Check message count (should be: 2 original messages + 1 new assistant response = 3 total)
        final_count = len(result.final_state.messages)
        expected_count = 3  # msg[0], msg[1], new_response (regeneration removes msg[2] and generates new response)
        results.record_test("Regeneration Engine: Correct message count", 
                          final_count == expected_count, f"got {final_count}, expected {expected_count}")
        
        # Check that subsequent messages were removed
        subsequent_found = any(msg.message_id in [msg_ids[3], msg_ids[4], msg_ids[5]] 
                             for msg in result.final_state.messages)
        results.record_test("Regeneration Engine: Subsequent messages removed", 
                          not subsequent_found, f"subsequent messages found: {subsequent_found}")
        
        # Test regeneration history
        regen_points = await get_regeneration_points("test-engine-001", run_config)
        has_history = regen_points is not None and len(regen_points) > 0
        results.record_test("Regeneration Engine: History tracking", has_history, 
                          f"found {len(regen_points) if regen_points else 0} regeneration points")
        
    except Exception as e:
        results.record_test("Regeneration Engine: Setup", False, f"Exception: {e}")


async def test_edge_cases(results: TestResults):
    """Test edge cases and error conditions."""
    print(f"\n🚨 Testing Edge Cases and Error Conditions")
    print("-" * 40)
    
    memory_provider = create_in_memory_provider()
    
    try:
        # Test 1: Empty conversation
        empty_conv_id = "empty-conversation"
        result = await memory_provider.store_messages(empty_conv_id, [], {})
        results.record_test("Edge Cases: Store empty conversation", 
                          not hasattr(result, 'error'), "stored empty conversation")
        
        # Test 2: Regenerate from non-existent conversation
        fake_conv_id = "non-existent-conv"
        fake_msg_id = generate_message_id()
        truncate_result = await memory_provider.truncate_conversation_after(fake_conv_id, fake_msg_id)
        has_error = hasattr(truncate_result, 'error')
        results.record_test("Edge Cases: Non-existent conversation", has_error, 
                          "correctly handled missing conversation")
        
        # Test 3: Multiple regenerations from same point
        conv_id = "multi-regen-test"
        msg_ids = await create_test_conversation(memory_provider, conv_id)
        
        # First regeneration
        first_result = await memory_provider.truncate_conversation_after(conv_id, msg_ids[2])
        # Second regeneration from same point
        second_result = await memory_provider.truncate_conversation_after(conv_id, msg_ids[2])
        
        both_successful = (not hasattr(first_result, 'error') and 
                         not hasattr(second_result, 'error'))
        results.record_test("Edge Cases: Multiple regenerations", both_successful, 
                          "handled multiple regenerations from same point")
        
        # Test 4: Regenerate from first message (should remove everything from and after first message)
        first_conv_id = "first-msg-test"
        first_msg_ids = await create_test_conversation(memory_provider, first_conv_id)
        first_msg_result = await memory_provider.truncate_conversation_after(first_conv_id, first_msg_ids[0])
        if hasattr(first_msg_result, 'data'):
            removed = first_msg_result.data
            results.record_test("Edge Cases: Regenerate from first message", 
                              removed == 6, f"removed {removed} messages (expected 6)")
        else:
            results.record_test("Edge Cases: Regenerate from first message", False, "failed")
        
        # Test 5: Regenerate from last message (should remove only the last message)
        last_conv_id = "last-msg-test"
        last_msg_ids = await create_test_conversation(memory_provider, last_conv_id)
        last_msg_result = await memory_provider.truncate_conversation_after(last_conv_id, last_msg_ids[5])
        if hasattr(last_msg_result, 'data'):
            removed = last_msg_result.data
            results.record_test("Edge Cases: Regenerate from last message", 
                              removed == 1, f"removed {removed} messages (expected 1)")
        else:
            results.record_test("Edge Cases: Regenerate from last message", False, "failed")
            
    except Exception as e:
        results.record_test("Edge Cases: Setup", False, f"Exception: {e}")


async def test_all_providers(results: TestResults):
    """Test regeneration functionality across all memory providers."""
    print(f"\n🗄️ Testing All Memory Providers")
    print("=" * 50)
    
    # Test InMemory Provider
    try:
        memory_provider = create_in_memory_provider()
        await test_memory_provider_methods(memory_provider, "InMemory", results)
        await memory_provider.close()
    except Exception as e:
        results.record_test("InMemory Provider", False, f"Exception: {e}")
    
    # Test PostgreSQL Provider (if available)
    try:
        from jaf.memory.providers.postgres import create_postgres_provider
        from jaf.memory.types import PostgresConfig
        
        # Only test if asyncpg is available
        try:
            import asyncpg
            postgres_config = PostgresConfig(
                host="localhost", 
                port=5432, 
                database="jaf_test", 
                username="postgres", 
                password="test",
                table_name="test_conversations"
            )
            
            print("   🔄 Attempting PostgreSQL connection...")
            postgres_result = await create_postgres_provider(postgres_config)
            if hasattr(postgres_result, 'data'):
                postgres_provider = postgres_result.data
                await test_memory_provider_methods(postgres_provider, "PostgreSQL", results)
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
            redis_config = RedisConfig(host="localhost", port=6379, db=1, key_prefix="jaf_test:")
            
            print("   🔄 Attempting Redis connection...")
            redis_result = await create_redis_provider(redis_config)
            if hasattr(redis_result, 'data'):
                redis_provider = redis_result.data
                await test_memory_provider_methods(redis_provider, "Redis", results)
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
    print("🧪 JAF-py Regeneration Comprehensive Test Suite")
    print("=" * 60)
    print("Testing across all providers, edge cases, and functionality")
    print("=" * 60)
    
    results = TestResults()
    
    try:
        # Test all memory providers
        await test_all_providers(results)
        
        # Test regeneration engine
        await test_regeneration_engine(results)
        
        # Test edge cases
        await test_edge_cases(results)
        
        # Show final summary
        print("\n" + "=" * 60)
        success = results.summary()
        
        if success:
            print("\n🎉 ALL REGENERATION TESTS PASSED!")
            print("✅ Regeneration functionality is fully implemented and working correctly.")
            print("✅ All memory providers support regeneration.")
            print("✅ Edge cases are properly handled.")
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
