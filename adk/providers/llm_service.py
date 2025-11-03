"""
ADK LLM Service - Production-Ready LLM Integration

This module provides a production-grade LLM service with real API integration,
streaming support, multiple providers, and comprehensive error handling.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from datetime import datetime

from ..config import AdkLLMConfig, create_adk_llm_config_from_environment
from ..types import (
    AdkAgent,
    AdkMessage,
    AdkSession,
    AdkContext,
    AdkResult,
    AdkSuccess,
    AdkFailure,
    AdkModelType,
    AdkProviderType,
    AdkStreamChunk,
)
from ..errors import (
    AdkLLMError,
    AdkErrorType,
    AdkErrorSeverity,
    create_adk_timeout_error,
    create_adk_rate_limit_error,
    with_adk_retry,
    with_adk_timeout,
    CircuitBreaker,
)
from .converters import AdkTypeConverter

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


@dataclass
class AdkLLMServiceConfig:
    """LLM service configuration with provider-specific settings."""

    provider: AdkProviderType
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: Optional[Union[AdkModelType, str]] = None
    timeout: float = 30.0
    max_retries: int = 3
    enable_streaming: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


@dataclass
class AdkLLMStreamChunk:
    """LLM streaming chunk with production features."""

    delta: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    is_done: bool = False
    metadata: Optional[Dict[str, Any]] = None


class AdkLLMService:
    """
    Production-ready LLM service with real API integration.

    Features:
    - Multi-provider support (OpenAI, Anthropic, Google, LiteLLM)
    - Real streaming using provider SDKs
    - Circuit breaker pattern for resilience
    - Comprehensive error handling with retries
    - Type conversion between ADK and provider formats
    """

    def __init__(self, config: AdkLLMServiceConfig):
        self.config = config
        self.client = None
        self.circuit_breaker = None

        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout,
                name=f"llm_{config.provider.value}",
            )

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.config.provider == AdkProviderType.OPENAI:
            if not OPENAI_AVAILABLE:
                raise AdkLLMError(
                    "OpenAI client not available. Install with: pip install openai",
                    provider=self.config.provider.value,
                    error_type=AdkErrorType.VALIDATION,
                    severity=AdkErrorSeverity.CRITICAL,
                )

            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

        elif self.config.provider == AdkProviderType.LITELLM:
            if not OPENAI_AVAILABLE:
                raise AdkLLMError(
                    "OpenAI client not available for LiteLLM. Install with: pip install openai",
                    provider=self.config.provider.value,
                    error_type=AdkErrorType.VALIDATION,
                    severity=AdkErrorSeverity.CRITICAL,
                )

            # LiteLLM uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=self.config.api_key or "anything",
                base_url=self.config.base_url or "http://localhost:4000",
                timeout=self.config.timeout,
            )

        elif self.config.provider == AdkProviderType.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise AdkLLMError(
                    "Anthropic client not available. Install with: pip install anthropic",
                    provider=self.config.provider.value,
                    error_type=AdkErrorType.VALIDATION,
                    severity=AdkErrorSeverity.CRITICAL,
                )

            # For production: use Anthropic SDK directly or via LiteLLM
            # For now, use OpenAI-compatible via LiteLLM
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "http://localhost:4000",
                timeout=self.config.timeout,
            )

        elif self.config.provider == AdkProviderType.GOOGLE:
            # For production: use Google SDK directly or via LiteLLM
            # For now, use OpenAI-compatible via LiteLLM
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "http://localhost:4000",
                timeout=self.config.timeout,
            )

        else:
            raise AdkLLMError(
                f"Unsupported provider: {self.config.provider}",
                provider=self.config.provider.value,
                error_type=AdkErrorType.VALIDATION,
                severity=AdkErrorSeverity.HIGH,
            )

    def _prepare_messages(
        self, agent: AdkAgent, session: AdkSession, context: AdkContext
    ) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        messages = []

        # Add system message with agent instructions
        if callable(agent.instructions):
            instruction_text = agent.instructions(context)
        else:
            instruction_text = agent.instructions

        messages.append(AdkTypeConverter.create_system_message_openai(instruction_text))

        # Add session messages
        for msg in session.messages:
            openai_msg = AdkTypeConverter.adk_to_openai_message(msg)
            messages.append(openai_msg)

        return messages

    def _prepare_tools(self, agent: AdkAgent) -> Optional[List[Dict[str, Any]]]:
        """Prepare tools for LLM API call."""
        if not agent.tools:
            return None

        return AdkTypeConverter.adk_tools_to_openai_tools(agent.tools)

    async def _make_llm_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make LLM API request with error handling."""
        request_params = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if tools:
            request_params["tools"] = tools
            # Determine tool choice based on message history
            last_message = messages[-1] if messages else None
            if last_message and last_message.get("role") == "tool":
                request_params["tool_choice"] = "auto"

        try:
            if stream:
                return await self.client.chat.completions.create(**request_params)
            else:
                response = await self.client.chat.completions.create(**request_params)
                return self._convert_openai_response(response)

        except Exception as e:
            # Convert provider-specific errors to ADK errors
            raise self._handle_provider_error(e)

    def _convert_openai_response(self, response) -> Dict[str, Any]:
        """Convert OpenAI response to ADK format."""
        choice = response.choices[0]

        # Convert tool calls to dict format if present
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

        return {
            "message": {"content": choice.message.content, "tool_calls": tool_calls},
            "usage": response.usage.model_dump()
            if hasattr(response, "usage") and response.usage
            else None,
            "model": response.model,
        }

    def _handle_provider_error(self, error: Exception) -> AdkLLMError:
        """Convert provider-specific errors to ADK errors."""
        error_message = str(error).lower()

        # Rate limiting
        if "rate limit" in error_message or "too many requests" in error_message:
            return create_adk_rate_limit_error(provider=self.config.provider.value, cause=error)

        # Timeout
        if "timeout" in error_message or "timed out" in error_message:
            return create_adk_timeout_error(
                self.config.timeout, "LLM request", provider=self.config.provider.value, cause=error
            )

        # Authentication
        if "authentication" in error_message or "unauthorized" in error_message:
            return AdkLLMError(
                "Authentication failed - check API key",
                provider=self.config.provider.value,
                error_type=AdkErrorType.AUTHENTICATION,
                severity=AdkErrorSeverity.HIGH,
                cause=error,
            )

        # Content filter
        if "content" in error_message and "filter" in error_message:
            return AdkLLMError(
                "Content filtered by provider",
                provider=self.config.provider.value,
                error_type=AdkErrorType.CONTENT_FILTER,
                severity=AdkErrorSeverity.MEDIUM,
                retryable=False,
                cause=error,
            )

        # Generic error
        return AdkLLMError(
            f"LLM request failed: {error}",
            provider=self.config.provider.value,
            error_type=AdkErrorType.INTERNAL,
            cause=error,
        )

    async def generate_response(
        self, agent: AdkAgent, session: AdkSession, context: AdkContext, new_message: AdkMessage
    ) -> AdkResult[AdkMessage, AdkLLMError]:
        """
        Generate a response from the LLM.

        Args:
            agent: ADK agent configuration
            session: Current session with message history
            context: Execution context
            new_message: New message to respond to

        Returns:
            Result containing the response message or error
        """
        try:
            # Add new message to session temporarily for context
            temp_session = AdkSession(
                session_id=session.session_id,
                user_id=session.user_id,
                app_name=session.app_name,
                messages=session.messages + [new_message],
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=session.metadata,
            )

            # Prepare request
            messages = self._prepare_messages(agent, temp_session, context)
            tools = self._prepare_tools(agent)
            model = AdkTypeConverter.model_type_to_string(
                agent.model or self.config.default_model or AdkModelType.GPT_4O
            )

            # Prepare additional parameters
            request_kwargs = {}
            if agent.temperature is not None:
                request_kwargs["temperature"] = agent.temperature
            if agent.max_tokens is not None:
                request_kwargs["max_tokens"] = agent.max_tokens

            # Create request function for retry/circuit breaker
            async def make_request():
                return await self._make_llm_request(
                    messages=messages, model=model, tools=tools, stream=False, **request_kwargs
                )

            # Apply timeout
            async def request_with_timeout():
                return await with_adk_timeout(
                    make_request(), self.config.timeout, "LLM response generation"
                )

            # Apply retry logic
            retry_func = await with_adk_retry(
                request_with_timeout,
                max_retries=self.config.max_retries,
                base_delay=1.0,
                max_delay=10.0,
            )

            # Apply circuit breaker if enabled
            if self.circuit_breaker:
                response = await self.circuit_breaker.call(retry_func)
            else:
                response = await retry_func()

            # Convert response to ADK message
            response_content = response.get("message", {}).get("content", "")
            tool_calls = response.get("message", {}).get("tool_calls")

            response_message = AdkMessage(
                role="assistant",
                content=response_content,
                tool_calls=tool_calls,
                timestamp=datetime.now(),
                metadata={
                    "model": response.get("model"),
                    "usage": response.get("usage"),
                    "provider": self.config.provider.value,
                },
            )

            return AdkSuccess(response_message)

        except AdkLLMError as e:
            return AdkFailure(e)
        except Exception as e:
            error = self._handle_provider_error(e)
            return AdkFailure(error)

    async def generate_streaming_response(
        self, agent: AdkAgent, session: AdkSession, context: AdkContext, new_message: AdkMessage
    ) -> AsyncIterator[AdkLLMStreamChunk]:
        """
        Generate a streaming response from the LLM.

        Args:
            agent: ADK agent configuration
            session: Current session with message history
            context: Execution context
            new_message: New message to respond to

        Yields:
            Stream chunks with delta content and function calls
        """
        if not self.config.enable_streaming:
            # Fall back to non-streaming
            result = await self.generate_response(agent, session, context, new_message)
            if isinstance(result, AdkSuccess):
                yield AdkLLMStreamChunk(
                    delta=result.data.content, is_done=True, metadata=result.data.metadata
                )
            else:
                yield AdkLLMStreamChunk(
                    delta="", is_done=True, metadata={"error": str(result.error)}
                )
            return

        try:
            # Add new message to session temporarily for context
            temp_session = AdkSession(
                session_id=session.session_id,
                user_id=session.user_id,
                app_name=session.app_name,
                messages=session.messages + [new_message],
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=session.metadata,
            )

            # Prepare request
            messages = self._prepare_messages(agent, temp_session, context)
            tools = self._prepare_tools(agent)
            model = AdkTypeConverter.model_type_to_string(
                agent.model or self.config.default_model or AdkModelType.GPT_4O
            )

            # Prepare additional parameters
            request_kwargs = {}
            if agent.temperature is not None:
                request_kwargs["temperature"] = agent.temperature
            if agent.max_tokens is not None:
                request_kwargs["max_tokens"] = agent.max_tokens

            # Make streaming request
            stream = await self._make_llm_request(
                messages=messages, model=model, tools=tools, stream=True, **request_kwargs
            )

            # Process streaming response
            current_function_call = None
            function_call_buffer = ""

            async for chunk in stream:
                try:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue

                    delta = choice.delta

                    # Handle content delta
                    if hasattr(delta, "content") and delta.content:
                        yield AdkLLMStreamChunk(
                            delta=delta.content, is_done=False, metadata={"model": chunk.model}
                        )

                    # Handle function call delta
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    # Start of new function call
                                    current_function_call = {
                                        "id": tool_call_delta.id,
                                        "name": tool_call_delta.function.name,
                                        "arguments": "",
                                    }
                                    function_call_buffer = ""

                                if tool_call_delta.function.arguments:
                                    # Accumulate function arguments
                                    function_call_buffer += tool_call_delta.function.arguments

                                    # Try to parse complete JSON
                                    try:
                                        parsed_args = json.loads(function_call_buffer)
                                        if current_function_call:
                                            current_function_call["arguments"] = parsed_args
                                            yield AdkLLMStreamChunk(
                                                delta=None,
                                                function_call=current_function_call,
                                                is_done=False,
                                                metadata={"model": chunk.model},
                                            )
                                            current_function_call = None
                                            function_call_buffer = ""
                                    except json.JSONDecodeError:
                                        # Not complete JSON yet, continue accumulating
                                        pass

                    # Check if stream is done
                    if choice.finish_reason:
                        yield AdkLLMStreamChunk(
                            delta=None,
                            is_done=True,
                            metadata={"model": chunk.model, "finish_reason": choice.finish_reason},
                        )
                        break

                except Exception as chunk_error:
                    # Handle individual chunk errors
                    yield AdkLLMStreamChunk(
                        delta="", is_done=True, metadata={"error": str(chunk_error)}
                    )
                    break

        except Exception as e:
            # Handle streaming errors
            error = self._handle_provider_error(e)
            yield AdkLLMStreamChunk(delta="", is_done=True, metadata={"error": str(error)})


def create_adk_llm_service(config: AdkLLMServiceConfig) -> AdkLLMService:
    """
    Create an ADK LLM service with the given configuration.

    Args:
        config: LLM service configuration

    Returns:
        AdkLLMService instance
    """
    return AdkLLMService(config)


def create_default_adk_llm_service() -> AdkLLMService:
    """
    Create a default ADK LLM service from environment configuration.

    Returns:
        AdkLLMService instance with environment-based configuration
    """
    # Get LLM config from environment
    llm_config = create_adk_llm_config_from_environment()

    # Convert to service config
    service_config = AdkLLMServiceConfig(
        provider=llm_config.provider,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        default_model=llm_config.default_model,
        timeout=llm_config.timeout / 1000,  # Convert from ms to seconds
        max_retries=llm_config.retries,
        enable_streaming=llm_config.streaming,
    )

    return create_adk_llm_service(service_config)
