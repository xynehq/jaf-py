"""
Tests for the ADK module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pytest
import os
from adk.config.llm_config import create_adk_llm_config_from_environment, AdkProviderType
from adk.sessions.redis import create_redis_session_provider, AdkRedisSessionConfig
from adk.sessions.in_memory import create_in_memory_session_provider
from adk.providers.llm_service import create_adk_llm_service, AdkLLMServiceConfig
from adk.types import AdkAgent, AdkMessage, AdkContext, AdkModelType, create_user_message, create_adk_context
from adk.errors import AdkError, create_circuit_breaker
from adk.sessions.base import AdkSessionProvider, AdkSuccess, AdkFailure

class TestAdkConfiguration:
    """Tests for the ADK configuration system."""

    def test_create_adk_llm_config_from_environment(self):
        """Test creating LLM config from environment variables."""
        if not os.environ.get('LITELLM_URL'):
            pytest.skip("LITELLM_URL environment variable not set.")
        config = create_adk_llm_config_from_environment()
        assert config.provider == AdkProviderType.LITELLM
        assert config.base_url == os.environ.get('LITELLM_URL')
        assert config.api_key == os.environ.get('LITELLM_API_KEY')
        assert config.default_model is not None

class TestAdkSessionProviders:
    """Tests for the ADK session providers."""

    @pytest.mark.asyncio
    async def test_in_memory_session_provider(self):
        """Test the in-memory session provider."""
        provider = create_in_memory_session_provider()
        session_result = await provider.create_session("test_user", "test_app")
        assert isinstance(session_result, AdkSuccess)
        session = session_result.data
        assert session.user_id == "test_user"

        message = create_user_message("Hello")
        add_result = await provider.add_message(session.session_id, message)
        assert isinstance(add_result, AdkSuccess)

        messages_result = await provider.get_messages(session.session_id)
        assert isinstance(messages_result, AdkSuccess)
        messages = messages_result.data
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_redis_session_provider(self):
        """Test the Redis session provider."""
        if not os.environ.get('JAF_REDIS_HOST'):
            pytest.skip("JAF_REDIS_HOST environment variable not set.")
        redis_config = AdkRedisSessionConfig(
            host=os.environ['JAF_REDIS_HOST'],
            port=int(os.environ['JAF_REDIS_PORT']),
            password=os.environ.get('JAF_REDIS_PASSWORD'),
            db=int(os.environ.get('JAF_REDIS_DB', 0)),
            key_prefix=os.environ.get('JAF_REDIS_KEY_PREFIX', 'jaf:memory:'),
            ttl_seconds=int(os.environ.get('JAF_REDIS_TTL', 3600))
        )
        provider_result = await create_redis_session_provider(redis_config)

        if isinstance(provider_result, AdkFailure):
            pytest.skip(f"Redis not available: {provider_result.error}")

        provider = provider_result.data
        health_check = await provider.health_check()
        if isinstance(health_check, AdkFailure):
            pytest.skip(f"Redis health check failed: {health_check.error}")

        session_result = await provider.create_session("test_user", "test_app")
        assert isinstance(session_result, AdkSuccess)
        session = session_result.data
        assert session.user_id == "test_user"

        message = create_user_message("Hello Redis")
        add_result = await provider.add_message(session.session_id, message)
        assert isinstance(add_result, AdkSuccess)

        messages_result = await provider.get_messages(session.session_id)
        assert isinstance(messages_result, AdkSuccess)
        messages = messages_result.data
        assert len(messages) == 1
        assert messages[0].content == "Hello Redis"
        await provider.close()

class TestAdkLLMService:
    """Tests for the ADK LLM service."""

    @pytest.mark.asyncio
    async def test_llm_service_creation(self):
        """Test creating the LLM service."""
        if not os.environ.get('LITELLM_URL'):
            pytest.skip("LITELLM_URL environment variable not set.")
        config = create_adk_llm_config_from_environment()
        service_config = AdkLLMServiceConfig(
            provider=config.provider,
            base_url=config.base_url,
            api_key=config.api_key,
            default_model=config.default_model,
            timeout=10.0,
            enable_streaming=True,
            enable_circuit_breaker=True
        )
        service = create_adk_llm_service(service_config)
        assert service.config.provider == AdkProviderType.LITELLM

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test generating a response from the LLM service."""
        if not os.environ.get('LITELLM_URL'):
            pytest.skip("LITELLM_URL environment variable not set.")
        config = create_adk_llm_config_from_environment()
        service_config = AdkLLMServiceConfig(
            provider=config.provider,
            base_url=config.base_url,
            api_key=config.api_key,
            default_model=config.default_model
        )
        service = create_adk_llm_service(service_config)
        agent = AdkAgent(
            name="TestAgent",
            instructions="You are a helpful assistant.",
            model=os.environ.get('LITELLM_MODEL')
        )
        provider = create_in_memory_session_provider()
        session_result = await provider.create_session("test_user", "test_app")
        session = session_result.data
        context = create_adk_context("test_user", session_id=session.session_id)
        message = create_user_message("What is 2+2?")
        response = await service.generate_response(agent, session, context, message)
        assert isinstance(response, AdkSuccess)
        assert "4" in response.data.content

class TestAdkErrorHandling:
    """Tests for ADK error handling."""

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test the circuit breaker."""
        circuit_breaker = create_circuit_breaker("test_breaker", failure_threshold=2, recovery_timeout=10)

        async def failing_function():
            raise ValueError("Simulated failure")

        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_function)

        with pytest.raises(AdkError):
            await circuit_breaker.call(failing_function)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
