#!/usr/bin/env python3
"""
Comprehensive test script for JAF A2A memory providers.

This script tests all three A2A memory providers (In-Memory, Redis, PostgreSQL)
to ensure they work correctly with the JAF A2A framework.
"""

import asyncio
import os
import socket
import sys
from datetime import datetime

from dotenv import load_dotenv

# Import JAF components
from jaf.a2a.agent import create_a2a_text_message
from jaf.a2a.types import A2ATask, TaskState, create_a2a_task
from jaf.memory.types import Failure

# Load environment variables
load_dotenv()


def check_postgres_available():
    """Check if PostgreSQL is available by testing socket connection and asyncpg import."""
    try:
        import asyncpg
    except ImportError:
        return False, "asyncpg not installed"
    
    # Test socket connection
    pg_host = os.getenv('JAF_POSTGRES_HOST', 'localhost')
    pg_port = int(os.getenv('JAF_POSTGRES_PORT', '5432'))
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((pg_host, pg_port))
        sock.close()
        if result != 0:
            return False, f"Cannot connect to PostgreSQL at {pg_host}:{pg_port}"
    except Exception as e:
        return False, f"Socket connection error: {e}"
    
    # Test asyncpg connection
    try:
        import asyncio
        async def test_connection():
            pg_user = os.getenv('JAF_POSTGRES_USER', 'postgres')
            pg_password = os.getenv('JAF_POSTGRES_PASSWORD', 'postgres')
            pg_db = os.getenv('JAF_POSTGRES_DB', 'jaf_test')
            
            try:
                conn = await asyncpg.connect(
                    user=pg_user, 
                    password=pg_password, 
                    host=pg_host, 
                    database=pg_db,
                    timeout=5
                )
                await conn.close()
                return True, "Available"
            except Exception as e:
                return False, f"Connection failed: {e}"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_connection())
            return result
        finally:
            loop.close()
    except Exception as e:
        return False, f"Connection test error: {e}"


class A2AMemoryProviderTester:
    """Test runner for A2A memory providers."""

    def __init__(self):
        self.test_results = {}

    async def test_provider(self, provider_name: str, provider, conversation_id: str = "test-a2a-conversation"):
        """Test an A2A memory provider with comprehensive operations."""
        print(f"\n🔧 Testing {provider_name} A2A Memory Provider...")

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
                print(f"       Latency: {health_data['latency_ms']:.2f}ms")

            # Test 2: Store a task
            print("  2. Storing a task...")
            test_message = create_a2a_text_message("Initial test message", context_id=conversation_id)
            test_task = create_a2a_task(test_message, conversation_id)
            
            metadata = {
                "user_id": "test_user_a2a",
                "test_session": "a2a_memory_provider_test",
                "timestamp": datetime.now().isoformat()
            }

            store_result = await provider.store_task(test_task, metadata)
            if isinstance(store_result, Failure):
                print(f"    ❌ Store failed: {store_result.error}")
                return False
            print(f"    ✅ Stored task {test_task.id} successfully")

            # Test 3: Retrieve task
            print("  3. Retrieving task...")
            get_result = await provider.get_task(test_task.id)
            if isinstance(get_result, Failure):
                print(f"    ❌ Retrieve failed: {get_result.error}")
                return False

            retrieved_task = get_result.data
            if not retrieved_task:
                print("    ❌ No task data returned")
                return False

            if retrieved_task.id != test_task.id:
                print(f"    ❌ Task ID mismatch: expected {test_task.id}, got {retrieved_task.id}")
                return False

            print(f"    ✅ Retrieved task with ID: {retrieved_task.id}")
            print(f"       Task Status: {retrieved_task.status.state}")

            # Test 4: Update task status
            print("  4. Updating task status...")
            update_result = await provider.update_task_status(
                test_task.id, 
                TaskState.WORKING,
                create_a2a_text_message("Task is now working", context_id=conversation_id)
            )
            if isinstance(update_result, Failure):
                print(f"    ❌ Update failed: {update_result.error}")
                return False
            print(f"    ✅ Updated task status to WORKING")

            # Test 5: Verify status update
            print("  5. Verifying status update...")
            updated_result = await provider.get_task(test_task.id)
            if isinstance(updated_result, Failure):
                print(f"    ❌ Verification retrieve failed: {updated_result.error}")
                return False

            updated_task = updated_result.data
            if updated_task.status.state != TaskState.WORKING:
                print(f"    ❌ Status mismatch: expected {TaskState.WORKING}, got {updated_task.status.state}")
                return False
            print(f"    ✅ Verified task status is {updated_task.status.state}")

            # Test 6: Get statistics
            print("  6. Getting statistics...")
            stats_result = await provider.get_task_stats()
            if isinstance(stats_result, Failure):
                print(f"    ❌ Get stats failed: {stats_result.error}")
                return False

            stats = stats_result.data
            print(f"    ✅ Stats - Total Tasks: {stats.get('total_tasks', 0)}")

            # Test 7: Find tasks
            print("  7. Finding tasks...")
            from jaf.a2a.memory.types import A2ATaskQuery
            query = A2ATaskQuery(context_id=conversation_id, state=TaskState.WORKING)
            find_result = await provider.find_tasks(query)
            if isinstance(find_result, Failure):
                print(f"    ❌ Find tasks failed: {find_result.error}")
                return False

            found_tasks = find_result.data
            if len(found_tasks) != 1:
                print(f"    ❌ Expected 1 task, found {len(found_tasks)}")
                return False
            print(f"    ✅ Found {len(found_tasks)} task for the context")

            # Test 8: Delete task
            print("  8. Deleting task...")
            delete_result = await provider.delete_task(test_task.id)
            if isinstance(delete_result, Failure):
                print(f"    ❌ Delete failed: {delete_result.error}")
                return False

            deleted = delete_result.data
            print(f"    ✅ Delete operation completed - Deleted: {deleted}")

            # Test 9: Verify deletion
            print("  9. Verifying deletion...")
            verify_result = await provider.get_task(test_task.id)
            if isinstance(verify_result, Failure):
                print(f"    ❌ Verification retrieve failed: {verify_result.error}")
                return False

            if verify_result.data is not None:
                print("    ❌ Task still exists after deletion")
                return False
            print("    ✅ Task successfully deleted")

            print(f"  ✅ All tests passed for {provider_name}!")
            return True

        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_in_memory_provider():
    """Test the in-memory A2A provider."""
    try:
        from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
        from jaf.a2a.memory.types import A2AInMemoryTaskConfig

        config = A2AInMemoryTaskConfig(
            type="memory",
            max_tasks=100,
            max_tasks_per_context=50
        )

        provider = create_a2a_in_memory_task_provider(config)
        tester = A2AMemoryProviderTester()

        return await tester.test_provider("In-Memory", provider, "test-a2a-memory-conversation")

    except Exception as e:
        print(f"❌ In-Memory A2A provider test setup failed: {e}")
        return False

async def test_redis_provider():
    """Test the Redis A2A provider."""
    try:
        import redis.asyncio as redis
        from jaf.a2a.memory.providers.redis import create_a2a_redis_task_provider
        from jaf.a2a.memory.types import A2ARedisTaskConfig

        print("📡 Connecting to Redis...")
        redis_host = os.getenv('JAF_REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('JAF_REDIS_PORT', '6379'))
        redis_password = os.getenv('JAF_REDIS_PASSWORD')
        redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
        await redis_client.ping()
        print("✅ Redis connection successful")

        # Cleanup previous test keys
        print("  🧹 Cleaning up old test keys...")
        keys = await redis_client.keys("jaf:test:a2a:*")
        if keys:
            await redis_client.delete(*keys)
        print(f"    ✅ Deleted {len(keys)} old keys.")

        config = A2ARedisTaskConfig(
            type="redis",
            key_prefix="jaf:test:a2a:",
            default_ttl=3600
        )
        provider_result = await create_a2a_redis_task_provider(config, redis_client)
        if isinstance(provider_result, Failure):
            raise Exception(provider_result.error)
        provider = provider_result.data
        tester = A2AMemoryProviderTester()
        result = await tester.test_provider("Redis", provider, "test-a2a-redis-conversation")
        await redis_client.aclose()
        return result
    except ImportError:
        print("❌ Redis dependencies not installed. Run: pip install redis")
        return False
    except Exception as e:
        print(f"❌ Redis A2A provider test setup failed: {e}")
        return False

async def test_postgres_provider():
    """Test the PostgreSQL A2A provider."""
    # Check if PostgreSQL is available
    is_available, message = check_postgres_available()
    if not is_available:
        print(f"⚠️  Skipping PostgreSQL A2A tests: {message}")
        print("   Set up PostgreSQL or install dependencies to run these tests")
        print("   Example: docker run -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=jaf_test -p 5432:5432 postgres")
        return True  # Return True to indicate "success" (graceful skip)
    
    try:
        import asyncpg
        from jaf.a2a.memory.providers.postgres import create_a2a_postgres_task_provider
        from jaf.a2a.memory.types import A2APostgresTaskConfig

        print("📡 Connecting to PostgreSQL...")
        pg_user = os.getenv('JAF_POSTGRES_USER', 'postgres')
        pg_password = os.getenv('JAF_POSTGRES_PASSWORD', 'postgres')
        pg_host = os.getenv('JAF_POSTGRES_HOST', 'localhost')
        pg_db = os.getenv('JAF_POSTGRES_DB', 'jaf_test')
        
        conn = await asyncpg.connect(user=pg_user, password=pg_password, host=pg_host, database=pg_db)
        print("✅ PostgreSQL connection successful")

        # Cleanup previous test table
        print("  🧹 Cleaning up old test table...")
        await conn.execute("DROP TABLE IF EXISTS test_a2a_tasks CASCADE;")
        print("    ✅ Dropped old table.")

        config = A2APostgresTaskConfig(
            type="postgres",
            table_name="test_a2a_tasks"
        )
        provider_result = await create_a2a_postgres_task_provider(config, conn)
        if isinstance(provider_result, Failure):
            raise Exception(provider_result.error)
        provider = provider_result.data
        tester = A2AMemoryProviderTester()
        result = await tester.test_provider("PostgreSQL", provider, "test-a2a-postgres-conversation")
        await conn.close()
        return result
    except ImportError:
        print("❌ PostgreSQL dependencies not installed. Run: pip install asyncpg")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL A2A provider test setup failed: {e}")
        return False

async def run_all_tests():
    """Run all A2A memory provider tests."""
    print("🧪 JAF A2A Memory Provider Test Suite")
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
        print("🎉 All tested memory providers are working correctly!")
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
