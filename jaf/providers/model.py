"""
Model provider implementations for the JAF framework.

This module provides model providers that integrate with various LLM services,
starting with LiteLLM for multi-provider support.
"""

from typing import Any, Dict, Optional, TypeVar, AsyncIterator, List
import asyncio
import httpx

from openai import OpenAI
from pydantic import BaseModel

from ..core.types import Agent, ContentRole, Message, ModelProvider, RunConfig, RunState, CompletionStreamChunk, ToolCallDelta, ToolCallFunctionDelta
from ..core.proxy import ProxyConfig

Ctx = TypeVar('Ctx')

def make_litellm_provider(
    base_url: str,
    api_key: str = "anything",
    default_timeout: Optional[float] = None,
    proxy_config: Optional[ProxyConfig] = None
) -> ModelProvider[Ctx]:
    """
    Create a LiteLLM-compatible model provider.
    
    Args:
        base_url: Base URL for the LiteLLM server
        api_key: API key (defaults to "anything" for local servers)
        default_timeout: Default timeout for model API calls in seconds
        proxy_config: Optional proxy configuration
        
    Returns:
        ModelProvider instance
    """

    class LiteLLMProvider:
        def __init__(self):
            # Default to "anything" if api_key is not provided, for local servers
            effective_api_key = api_key if api_key is not None else "anything"
            
            # Configure HTTP client with proxy support
            client_kwargs = {
                "base_url": base_url,
                "api_key": effective_api_key,
            }
            
            if proxy_config:
                proxies = proxy_config.to_httpx_proxies()
                if proxies:
                    # Create httpx client with proxy configuration
                    try:
                        # Use the https proxy if available, otherwise http proxy
                        proxy_url = proxies.get('https://') or proxies.get('http://')
                        if proxy_url:
                            http_client = httpx.Client(proxy=proxy_url)
                            client_kwargs["http_client"] = http_client
                    except Exception as e:
                        print(f"Warning: Could not configure proxy: {e}")
                        # Fall back to environment variables for proxy
            
            self.client = OpenAI(**client_kwargs)
            self.default_timeout = default_timeout

        async def get_completion(
            self,
            state: RunState[Ctx],
            agent: Agent[Ctx, Any],
            config: RunConfig[Ctx]
        ) -> Dict[str, Any]:
            """Get completion from the model."""

            # Determine model to use
            model = (config.model_override or
                    (agent.model_config.name if agent.model_config else "gpt-4o"))

            # Create system message
            system_message = {
                "role": "system",
                "content": agent.instructions(state)
            }

            # Convert messages to OpenAI format
            messages = [system_message] + [
                _convert_message(msg) for msg in state.messages
            ]

            # Convert tools to OpenAI format
            tools = None
            if agent.tools:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.schema.name,
                            "description": tool.schema.description,
                            "parameters": _pydantic_to_json_schema(tool.schema.parameters),
                        }
                    }
                    for tool in agent.tools
                ]

            # Determine tool choice behavior
            last_message = state.messages[-1] if state.messages else None
            is_after_tool_call = last_message and (last_message.role == ContentRole.TOOL or last_message.role == 'tool')

            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,
            }

            # Add optional parameters
            if agent.model_config:
                if agent.model_config.temperature is not None:
                    request_params["temperature"] = agent.model_config.temperature
                if agent.model_config.max_tokens is not None:
                    request_params["max_tokens"] = agent.model_config.max_tokens

            if tools:
                request_params["tools"] = tools
                # Always set tool_choice to auto when tools are available
                request_params["tool_choice"] = "auto"

            if agent.output_codec:
                request_params["response_format"] = {"type": "json_object"}

            # Make the API call
            response = self.client.chat.completions.create(**request_params)

            # Return in the expected format that the engine expects
            choice = response.choices[0]

            # Convert tool_calls to dict format if present
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        'id': tc.id,
                        'type': tc.type,
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]

            # Extract usage data
            usage_data = None
            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return {
                'id': response.id,
                'created': response.created,
                'model': response.model,
                'system_fingerprint': response.system_fingerprint,
                'message': {
                    'content': choice.message.content,
                    'tool_calls': tool_calls
                },
                'usage': usage_data,
                'prompt': messages
            }

        async def get_completion_stream(
            self,
            state: RunState[Ctx],
            agent: Agent[Ctx, Any],
            config: RunConfig[Ctx]
        ) -> AsyncIterator[CompletionStreamChunk]:
            """
            Stream completion chunks from the model provider, yielding text deltas and tool-call deltas.
            Uses OpenAI-compatible streaming via LiteLLM endpoint.
            """
            # Determine model to use
            model = (config.model_override or
                     (agent.model_config.name if agent.model_config else "gpt-4o"))

            # Create system message
            system_message = {
                "role": "system",
                "content": agent.instructions(state)
            }

            # Convert messages to OpenAI format
            messages = [system_message] + [
                _convert_message(msg) for msg in state.messages
            ]

            # Convert tools to OpenAI format
            tools = None
            if agent.tools:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.schema.name,
                            "description": tool.schema.description,
                            "parameters": _pydantic_to_json_schema(tool.schema.parameters),
                        }
                    }
                    for tool in agent.tools
                ]

            # Determine tool choice behavior
            last_message = state.messages[-1] if state.messages else None
            is_after_tool_call = last_message and (last_message.role == ContentRole.TOOL or last_message.role == 'tool')

            # Prepare request parameters
            request_params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }

            # Add optional parameters
            if agent.model_config:
                if agent.model_config.temperature is not None:
                    request_params["temperature"] = agent.model_config.temperature
                if agent.model_config.max_tokens is not None:
                    request_params["max_tokens"] = agent.model_config.max_tokens

            if tools:
                request_params["tools"] = tools
                # Set tool_choice to auto when tools are available
                request_params["tool_choice"] = "auto"

            if agent.output_codec:
                request_params["response_format"] = {"type": "json_object"}

            # Enable streaming
            request_params["stream"] = True

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue(maxsize=256)
            SENTINEL = object()

            def _put(item: CompletionStreamChunk):
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                except RuntimeError:
                    # Event loop closed; drop silently
                    pass

            def _producer():
                try:
                    stream = self.client.chat.completions.create(**request_params)
                    for chunk in stream:
                        try:
                            # Best-effort extraction of raw for debugging
                            try:
                                raw_obj = chunk.model_dump()  # pydantic BaseModel
                            except Exception:
                                raw_obj = None

                            choice = None
                            if getattr(chunk, "choices", None):
                                choice = chunk.choices[0]

                            if choice is None:
                                continue

                            delta = getattr(choice, "delta", None)
                            finish_reason = getattr(choice, "finish_reason", None)

                            # Text content delta
                            if delta is not None:
                                content_delta = getattr(delta, "content", None)
                                if content_delta:
                                    _put(CompletionStreamChunk(delta=content_delta, raw=raw_obj))

                                # Tool call deltas
                                tool_calls = getattr(delta, "tool_calls", None)
                                if isinstance(tool_calls, list):
                                    for tc in tool_calls:
                                        # Each tc is likely a pydantic model with .index/.id/.function
                                        try:
                                            idx = getattr(tc, "index", 0) or 0
                                            tc_id = getattr(tc, "id", None)
                                            fn = getattr(tc, "function", None)
                                            fn_name = getattr(fn, "name", None) if fn is not None else None
                                            # OpenAI streams "arguments" as incremental deltas
                                            args_delta = getattr(fn, "arguments", None) if fn is not None else None

                                            _put(CompletionStreamChunk(
                                                tool_call_delta=ToolCallDelta(
                                                    index=idx,
                                                    id=tc_id,
                                                    type='function',
                                                    function=ToolCallFunctionDelta(
                                                        name=fn_name,
                                                        arguments_delta=args_delta
                                                    )
                                                ),
                                                raw=raw_obj
                                            ))
                                        except Exception:
                                            # Skip malformed tool-call deltas
                                            continue

                            # Completion ended
                            if finish_reason:
                                _put(CompletionStreamChunk(is_done=True, finish_reason=finish_reason, raw=raw_obj))
                        except Exception:
                            # Skip individual chunk errors, keep streaming
                            continue
                except Exception:
                    # On top-level stream error, signal done
                    pass
                finally:
                    try:
                        asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)
                    except RuntimeError:
                        pass

            # Start producer in background
            loop.run_in_executor(None, _producer)

            # Consume queue and yield
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                # Guarantee type for consumers
                if isinstance(item, CompletionStreamChunk):
                    yield item

    return LiteLLMProvider()

def _convert_message(msg: Message) -> Dict[str, Any]:
    """Convert JAF Message to OpenAI message format."""
    if msg.role == 'user':
        return {
            "role": "user",
            "content": msg.content
        }
    elif msg.role == 'assistant':
        result = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        return result
    elif msg.role == ContentRole.TOOL:
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id
        }
    else:
        raise ValueError(f"Unknown message role: {msg.role}")

def _pydantic_to_json_schema(model_class: type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model to JSON schema for OpenAI tools.
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        JSON schema dictionary
    """
    if hasattr(model_class, 'model_json_schema'):
        # Pydantic v2
        return model_class.model_json_schema()
    else:
        # Pydantic v1 fallback
        return model_class.schema()
