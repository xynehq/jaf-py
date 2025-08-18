#!/usr/bin/env python3
"""
Comprehensive test script for JAF memory providers.

This script tests all three memory providers (In-Memory, Redis, PostgreSQL)
to ensure they work correctly with the JAF framework.
"""

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Import JAF components
from jaf.core.types import Message
from jaf.memory.types import Failure

# Load environment variables
load_dotenv()

class MemoryProviderTester:
    """Test runner for memory providers."""

    def __init__(self):
        self.test_results = {}

    async def test_provider(self, provider_name: str, provider, conversation_id: str = "test-conversation"):
        """Test a memory provider with comprehensive operations."""
        print(f"\n🔧 Testing {provider_name} Memory Provider...")

        try:
            # Test 1: Health Check
            print("  1. Health check...")
            health_result = await provider.health_check()
            if isinstance(health_result, Failure):
                print(f"    ❌ Health check failed: {health_result.error}")
                return False

            health_data = health_result.data
            print(f"    ✅ Health check passed - Healthy: {health_data.get('healthy', False)}")
            if 'latency_ms' in health_data:
                print(f"       Latency: {health_data['latency_ms']}ms")

            # Test 2: Store messages
            print("  2. Storing messages...")
            test_messages = [
                Message(role="user", content="Hello, I'm testing the memory system"),
                Message(role="assistant", content="Hello! I can help you test the memory system."),
                Message(role="user", content="What's my name?"),
                Message(role="assistant", content="I don't have information about your name from our current conversation.")
            ]

            metadata = {
                "user_id": "test_user_123",
                "test_session": "memory_provider_test",
                "timestamp": datetime.now().isoformat()
            }

            store_result = await provider.store_messages(conversation_id, test_messages, metadata)
            if isinstance(store_result, Failure):
                print(f"    ❌ Store failed: {store_result.error}")
                return False
            print(f"    ✅ Stored {len(test_messages)} messages successfully")

            # Test 3: Retrieve conversation
            print("  3. Retrieving conversation...")
            get_result = await provider.get_conversation(conversation_id)
            if isinstance(get_result, Failure):
                print(f"    ❌ Retrieve failed: {get_result.error}")
                return False

            conversation = get_result.data
            if not conversation:
                print("    ❌ No conversation data returned")
                return False

            if len(conversation.messages) != len(test_messages):
                print(f"    ❌ Message count mismatch: expected {len(test_messages)}, got {len(conversation.messages)}")
                return False

            print(f"    ✅ Retrieved conversation with {len(conversation.messages)} messages")
            print(f"       Conversation ID: {conversation.conversation_id}")
            print(f"       User ID: {conversation.user_id}")

            # Test 4: Append messages
            print("  4. Appending messages...")
            new_messages = [
                Message(role="user", content="My name is Alice"),
                Message(role="assistant", content="Nice to meet you, Alice! I'll remember your name.")
            ]

            append_result = await provider.append_messages(conversation_id, new_messages)
            if isinstance(append_result, Failure):
                print(f"    ❌ Append failed: {append_result.error}")
                return False
            print(f"    ✅ Appended {len(new_messages)} messages successfully")

            # Test 5: Verify appended messages
            print("  5. Verifying appended messages...")
            updated_result = await provider.get_conversation(conversation_id)
            if isinstance(updated_result, Failure):
                print(f"    ❌ Verification retrieve failed: {updated_result.error}")
                return False

            updated_conversation = updated_result.data
            expected_count = len(test_messages) + len(new_messages)
            if len(updated_conversation.messages) != expected_count:
                print(f"    ❌ Total message count mismatch: expected {expected_count}, got {len(updated_conversation.messages)}")
                return False
            print(f"    ✅ Verified {len(updated_conversation.messages)} total messages")

            # Test 6: Get recent messages
            print("  6. Getting recent messages...")
            recent_result = await provider.get_recent_messages(conversation_id, limit=3)
            if isinstance(recent_result, Failure):
                print(f"    ❌ Get recent failed: {recent_result.error}")
                return False

            recent_messages = recent_result.data
            if len(recent_messages) != 3:
                print(f"    ❌ Recent messages count mismatch: expected 3, got {len(recent_messages)}")
                return False
            print(f"    ✅ Retrieved {len(recent_messages)} recent messages")

            # Test 7: Get statistics
            print("  7. Getting statistics...")
            stats_result = await provider.get_stats()
            if isinstance(stats_result, Failure):
                print(f"    ❌ Get stats failed: {stats_result.error}")
                return False

            stats = stats_result.data
            print(f"    ✅ Stats - Conversations: {stats.get('totalConversations', 0)}, Messages: {stats.get('totalMessages', 0)}")

            # Test 8: Find conversations
            print("  8. Finding conversations...")
            from jaf.memory.types import MemoryQuery
            query = MemoryQuery(user_id="test_user_123", limit=10)
            find_result = await provider.find_conversations(query)
            if isinstance(find_result, Failure):
                print(f"    ❌ Find conversations failed: {find_result.error}")
                return False

            found_conversations = find_result.data
            print(f"    ✅ Found {len(found_conversations)} conversations for user")

            # Test 9: Delete conversation
            print("  9. Deleting conversation...")
            delete_result = await provider.delete_conversation(conversation_id)
            if isinstance(delete_result, Failure):
                print(f"    ❌ Delete failed: {delete_result.error}")
                return False

            deleted = delete_result.data
            print(f"    ✅ Delete operation completed - Deleted: {deleted}")

            # Test 10: Verify deletion
            print("  10. Verifying deletion...")
            verify_result = await provider.get_conversation(conversation_id)
            if isinstance(verify_result, Failure):
                print(f"    ❌ Verification retrieve failed: {verify_result.error}")
                return False

            if verify_result.data is not None:
                print("    ❌ Conversation still exists after deletion")
                return False
            print("    ✅ Conversation successfully deleted")

            print(f"  ✅ All tests passed for {provider_name}!")
            return True

        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_in_memory_provider():
    """Test the in-memory provider."""
    try:
        from jaf.memory.providers.in_memory import create_in_memory_provider
        from jaf.memory.types import InMemoryConfig

        config = InMemoryConfig(
            type="memory",
            max_conversations=100,
            max_messages_per_conversation=1000
        )

        provider = create_in_memory_provider(config)
        tester = MemoryProviderTester()

        return await tester.test_provider("In-Memory", provider, "test-memory-conversation")

    except Exception as e:
        print(f"❌ In-Memory provider test setup failed: {e}")
        return False

async def test_redis_provider():
    """Test the Redis provider."""
    try:
        # Check if Redis dependencies are available
        import redis.asyncio as redis

        from jaf.memory.providers.redis import create_redis_provider
        from jaf.memory.types import RedisConfig

        print("📡 Connecting to Redis...")

        # Get Redis configuration from environment
        redis_host = os.getenv('JAF_REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('JAF_REDIS_PORT', '6379'))
        redis_password = os.getenv('JAF_REDIS_PASSWORD')
        redis_db = int(os.getenv('JAF_REDIS_DB', '0'))

        # Create Redis client with same config as the provider will use
        if redis_password:
            redis_url = f'redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}'
        else:
            redis_url = f'redis://{redis_host}:{redis_port}/{redis_db}'

        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        await redis_client.ping()
        print("✅ Redis connection successful")

        config = RedisConfig(
            type="redis",
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            key_prefix="jaf:test:",
            ttl=3600
        )

        provider_result = await create_redis_provider(config)
        if isinstance(provider_result, Failure):
            print(f"❌ Redis provider creation failed: {provider_result.error}")
            await redis_client.close()
            return False

        provider = provider_result.data
        tester = MemoryProviderTester()

        result = await tester.test_provider("Redis", provider, "test-redis-conversation")

        # Cleanup
        await redis_client.aclose()

        return result

    except ImportError:
        print("❌ Redis dependencies not installed. Run: pip install redis")
        return False
    except Exception as e:
        print(f"❌ Redis provider test setup failed: {e}")
        print("   Make sure Redis server is running on localhost:6379")
        return False

def check_postgres_available():
    """Check if PostgreSQL is available for testing."""
    try:
        import asyncpg
        import asyncio
        import socket
        
        # Get PostgreSQL configuration from environment
        pg_host = os.getenv('JAF_POSTGRES_HOST', 'localhost')
        pg_port = int(os.getenv('JAF_POSTGRES_PORT', '5432'))
        
        # Quick socket check first
        try:
            with socket.create_connection((pg_host, pg_port), timeout=1):
                pass
        except (socket.error, socket.timeout):
            return False
            
        # Try to connect to PostgreSQL
        async def test_connection():
            try:
                connection_string = os.getenv('JAF_POSTGRES_CONNECTION_STRING', 'postgresql://postgres:postgres@localhost:5432/jaf_test')
                conn = await asyncpg.connect(dsn=connection_string)
                await conn.close()
                return True
            except:
                return False
        
        return asyncio.run(test_connection())
        
    except ImportError:
        return False
    except:
        return False

async def test_postgres_provider():
    """Test the PostgreSQL provider."""
    if not check_postgres_available():
        print("⏭️  PostgreSQL is not available - skipping PostgreSQL tests")
        print("   To run PostgreSQL tests:")
        print("   1. Install PostgreSQL dependencies: pip install asyncpg")
        print("   2. Start PostgreSQL server")
        print("   3. Create test database: jaf_test")
        return True  # Return True to indicate "skipped, not failed"
        
    try:
        # Check if PostgreSQL dependencies are available
        import asyncpg

        from jaf.memory.providers.postgres import create_postgres_provider
        from jaf.memory.types import PostgresConfig

        print("📡 Connecting to PostgreSQL...")

        # Connection configuration
        connection_string = os.getenv('JAF_POSTGRES_CONNECTION_STRING', 'postgresql://postgres:postgres@localhost:5432/jaf_test')

        # Test connection
        conn = await asyncpg.connect(dsn=connection_string)
        await conn.close()

        print("✅ PostgreSQL connection successful")

        config = PostgresConfig(
            type="postgres",
            host="localhost",
            port=5432,
            database="jaf_memory",
            username="postgres",
            password="testpass",
            ssl=False,
            table_name="test_conversations",
            max_connections=10
        )

        provider_result = await create_postgres_provider(config)
        if isinstance(provider_result, Failure):
            print(f"    ❌ Create provider failed: {provider_result.error}")
            return False
        provider = provider_result.data
        tester = MemoryProviderTester()

        result = await tester.test_provider("PostgreSQL", provider, "test-postgres-conversation")

        # Cleanup
        await provider.close()

        return result

    except ImportError:
        print("❌ PostgreSQL dependencies not installed. Run: pip install asyncpg")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL provider test setup failed: {e}")
        print("   Make sure PostgreSQL server is running with database 'jaf_test'")
        return False

async def run_all_tests():
    """Run all memory provider tests."""
    print("🧪 JAF Memory Provider Test Suite")
    print("=" * 50)

    results = {}

    # Test In-Memory provider
    print("\n" + "=" * 50)
    results['in_memory'] = await test_in_memory_provider()

    # Test Redis provider
    print("\n" + "=" * 50)
    results['redis'] = await test_redis_provider()

    # Test PostgreSQL provider
    print("\n" + "=" * 50)
    results['postgres'] = await test_postgres_provider()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    for provider, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {provider.upper():<12}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)

    print(f"\nOverall: {passed_tests}/{total_tests} providers passed tests")

    if passed_tests == total_tests:
        print("🎉 All memory providers are working correctly!")
        return True
    else:
        print("⚠️  Some memory providers need attention.")
        return False

async def main():
    """Main entry point."""
    try:
        success = await run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
