import asyncio
import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from jaf.core.types import Message
from jaf.memory.providers.redis import create_redis_provider
from jaf.memory.types import RedisConfig, Failure

load_dotenv()


def check_redis_available():
    """Check if Redis is available for testing."""
    import socket

    redis_host = os.getenv("JAF_REDIS_HOST", "localhost")
    redis_port = int(os.getenv("JAF_REDIS_PORT", "6379"))

    try:
        with socket.create_connection((redis_host, redis_port), timeout=1):
            return True
    except (socket.error, socket.timeout):
        return False


def skip_if_no_redis():
    """Decorator to skip tests if Redis is not available."""
    return pytest.mark.skipif(
        not check_redis_available(),
        reason="Redis is not available - install and start Redis to run these tests",
    )


@pytest.fixture(scope="function")
async def redis_provider():
    """A fixture to create and tear down the Redis provider."""
    if not check_redis_available():
        pytest.skip("Redis is not available")

    redis_host = os.getenv("JAF_REDIS_HOST", "localhost")
    redis_port = int(os.getenv("JAF_REDIS_PORT", "6379"))
    redis_password = os.getenv("JAF_REDIS_PASSWORD")
    redis_db = int(os.getenv("JAF_REDIS_DB", "0"))

    config = RedisConfig(
        type="redis",
        host=redis_host,
        port=redis_port,
        password=redis_password,
        db=redis_db,
        key_prefix="jaf:test:deep:",
        ttl=3600,
    )

    provider_result = await create_redis_provider(config)
    if isinstance(provider_result, Failure):
        pytest.skip(f"Failed to create Redis provider: {provider_result.error}")

    provider = provider_result.data
    yield provider
    await provider.close()


@skip_if_no_redis()
@pytest.mark.asyncio
async def test_redis_large_number_of_messages(redis_provider):
    """Test the Redis provider with a large number of messages."""
    conversation_id = "test-large-messages"
    messages = [Message(role="user", content=f"Message {i}") for i in range(1000)]

    store_result = await redis_provider.store_messages(conversation_id, messages)
    assert not isinstance(store_result, Failure)

    get_result = await redis_provider.get_conversation(conversation_id)
    assert not isinstance(get_result, Failure)
    assert len(get_result.data.messages) == 1000

    await redis_provider.delete_conversation(conversation_id)


@skip_if_no_redis()
@pytest.mark.asyncio
async def test_redis_multiple_conversations(redis_provider):
    """Test the Redis provider with multiple conversations."""
    conversation_ids = [f"test-multi-conv-{i}" for i in range(10)]

    for conv_id in conversation_ids:
        messages = [Message(role="user", content=f"Hello from {conv_id}")]
        store_result = await redis_provider.store_messages(conv_id, messages)
        assert not isinstance(store_result, Failure)

    for conv_id in conversation_ids:
        get_result = await redis_provider.get_conversation(conv_id)
        assert not isinstance(get_result, Failure)
        assert len(get_result.data.messages) == 1
        assert get_result.data.messages[0].content == f"Hello from {conv_id}"

    for conv_id in conversation_ids:
        await redis_provider.delete_conversation(conv_id)


@skip_if_no_redis()
@pytest.mark.asyncio
async def test_redis_large_number_of_conversations(redis_provider):
    """Test the Redis provider with a large number of conversations."""
    conversation_ids = [f"test-large-conv-{i}" for i in range(100)]

    for conv_id in conversation_ids:
        messages = [Message(role="user", content=f"Hello from {conv_id}")]
        store_result = await redis_provider.store_messages(conv_id, messages)
        assert not isinstance(store_result, Failure)

    for conv_id in conversation_ids:
        get_result = await redis_provider.get_conversation(conv_id)
        assert not isinstance(get_result, Failure)
        assert len(get_result.data.messages) == 1
        assert get_result.data.messages[0].content == f"Hello from {conv_id}"

    for conv_id in conversation_ids:
        await redis_provider.delete_conversation(conv_id)


@skip_if_no_redis()
@pytest.mark.asyncio
async def test_redis_large_messages_and_conversations(redis_provider):
    """Test the Redis provider with a large number of messages and conversations."""
    conversation_ids = [f"test-large-both-{i}" for i in range(10)]

    for conv_id in conversation_ids:
        messages = [Message(role="user", content=f"Message {i} from {conv_id}") for i in range(100)]
        store_result = await redis_provider.store_messages(conv_id, messages)
        assert not isinstance(store_result, Failure)

    for conv_id in conversation_ids:
        get_result = await redis_provider.get_conversation(conv_id)
        assert not isinstance(get_result, Failure)
        assert len(get_result.data.messages) == 100

    for conv_id in conversation_ids:
        await redis_provider.delete_conversation(conv_id)
