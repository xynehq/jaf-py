"""
Model provider implementations for the JAF framework.

This module provides model providers that integrate with various LLM services,
starting with LiteLLM for multi-provider support.
"""

from typing import Any, Dict, List, Optional, TypeVar, AsyncIterator
import httpx
import time
import os
import base64
import asyncio
import json
import logging
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI
from pydantic import BaseModel
import litellm
import websockets
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from ..core.types import (
    Agent,
    ContentRole,
    Message,
    ModelProvider,
    RunConfig,
    RunState,
    CompletionStreamChunk,
    ToolCallDelta,
    ToolCallFunctionDelta,
    MessageContentPart,
    get_text_content,
    RetryEvent,
    RetryEventData,
    FallbackEvent,
    FallbackEventData,
)
from ..core.proxy import ProxyConfig
from ..utils.document_processor import (
    extract_document_content,
    is_document_supported,
    get_document_description,
    DocumentProcessingError,
)

Ctx = TypeVar("Ctx")
logger = logging.getLogger(__name__)

# Vision model caching
VISION_MODEL_CACHE_TTL = 5 * 60  # 5 minutes
VISION_API_TIMEOUT = 3.0  # 3 seconds
_vision_model_cache: Dict[str, Dict[str, Any]] = {}
MAX_IMAGE_BYTES = int(os.environ.get("JAF_MAX_IMAGE_BYTES", 8 * 1024 * 1024))


async def _is_vision_model(model: str, base_url: str) -> bool:
    """
    Check if a model supports vision capabilities.

    Args:
        model: Model name to check
        base_url: Base URL of the LiteLLM server

    Returns:
        True if model supports vision, False otherwise
    """
    cache_key = f"{base_url}:{model}"
    cached = _vision_model_cache.get(cache_key)

    if cached and time.time() - cached["timestamp"] < VISION_MODEL_CACHE_TTL:
        return cached["supports"]

    try:
        async with httpx.AsyncClient(timeout=VISION_API_TIMEOUT) as client:
            response = await client.get(
                f"{base_url}/model_group/info", headers={"accept": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                model_info = None

                if "data" in data and isinstance(data["data"], list):
                    for m in data["data"]:
                        if m.get("model_group") == model or model in str(m.get("model_group", "")):
                            model_info = m
                            break

                if model_info and "supports_vision" in model_info:
                    result = model_info["supports_vision"]
                    _vision_model_cache[cache_key] = {"supports": result, "timestamp": time.time()}
                    return result
            else:
                print(
                    f"Warning: Vision API returned status {response.status_code} for model {model}"
                )

    except Exception as e:
        print(f"Warning: Vision API error for model {model}: {e}")

    # Fallback to known vision models
    known_vision_models = [
        "gpt-4-vision-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4",
        "claude-sonnet-4-20250514",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    is_known_vision_model = any(
        vision_model.lower() in model.lower() for vision_model in known_vision_models
    )

    _vision_model_cache[cache_key] = {"supports": is_known_vision_model, "timestamp": time.time()}

    return is_known_vision_model


def _classify_error_for_fallback(e: Exception) -> tuple[str, str]:
    """
    Classify an error to determine the fallback type and reason.

    Args:
        e: Exception from model call

    Returns:
        Tuple of (fallback_type, reason)
    """
    error_message = str(e).lower()
    error_type = type(e).__name__

    # Check for content policy violations
    if (
        "content" in error_message
        and ("policy" in error_message or "filter" in error_message)
        or "contentpolicyviolation" in error_type.lower()
        or "content_filter" in error_message
        or "safety" in error_message
    ):
        return ("content_policy", "Content Policy Violation")

    # Check for context window exceeded
    if (
        "context" in error_message
        and "window" in error_message
        or "too long" in error_message
        or "maximum context" in error_message
        or "contextwindowexceeded" in error_type.lower()
        or "prompt is too long" in error_message
        or "tokens" in error_message
        and "limit" in error_message
    ):
        return ("context_window", "Context Window Exceeded")

    # Default to general fallback
    if hasattr(e, "status_code"):
        status_code = e.status_code
        if status_code == 429:
            return ("general", f"HTTP {status_code} - Rate Limit")
        elif 500 <= status_code < 600:
            return ("general", f"HTTP {status_code} - Server Error")
        else:
            return ("general", f"HTTP {status_code}")

    return ("general", error_type)


async def _retry_with_events(
    operation_func,
    state: RunState,
    config: RunConfig,
    operation_name: str = "llm_call",
    max_retries: int = 3,
    backoff_factor: float = 1.0,
):
    """
    Wrapper that retries an async operation and emits retry events.

    Args:
        operation_func: Async function to execute (should accept no arguments)
        state: Current run state
        config: Run configuration with event handler
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier

    Returns:
        Result from operation_func

    Raises:
        Last exception if all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await operation_func()
        except Exception as e:
            last_exception = e

            # Check if this is a retryable HTTP error
            is_retryable = False
            reason = str(e)
            error_details = {"error_type": type(e).__name__, "error_message": str(e)}

            # Check for HTTP errors (common in OpenAI/LiteLLM)
            if hasattr(e, "status_code"):
                status_code = e.status_code
                error_details["status_code"] = status_code

                # Retry on rate limits (429) and server errors (5xx)
                if status_code == 429:
                    is_retryable = True
                    reason = f"HTTP {status_code} - Rate Limit"
                elif 500 <= status_code < 600:
                    is_retryable = True
                    reason = f"HTTP {status_code} - Server Error"
                else:
                    reason = f"HTTP {status_code}"

            # Check for common exception names
            elif "RateLimitError" in type(e).__name__:
                is_retryable = True
                reason = "Rate Limit Error"
            elif "ServiceUnavailableError" in type(e).__name__ or "APIError" in type(e).__name__:
                is_retryable = True
                reason = "API Error"
            elif "Timeout" in type(e).__name__:
                is_retryable = True
                reason = "Timeout"

            # If not last attempt and is retryable, retry with backoff
            if attempt < max_retries and is_retryable:
                delay = backoff_factor * (2**attempt)  # Exponential backoff

                # Emit retry event
                if config.on_event:
                    retry_event = RetryEvent(
                        data=RetryEventData(
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            reason=reason,
                            operation=operation_name,
                            trace_id=state.trace_id,
                            run_id=state.run_id,
                            delay=delay,
                            error_details=error_details,
                        )
                    )
                    config.on_event(retry_event)

                print(
                    f"[JAF:RETRY] Attempt {attempt + 1}/{max_retries} failed: {reason}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                # Not retryable or last attempt, re-raise
                raise

    # Should never reach here, but just in case
    raise last_exception


def make_litellm_provider(
    base_url: str,
    api_key: str = "anything",
    default_timeout: Optional[float] = None,
    proxy_config: Optional[ProxyConfig] = None,
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
                        proxy_url = proxies.get("https://") or proxies.get("http://")
                        if proxy_url:
                            http_client = httpx.AsyncClient(proxy=proxy_url)
                            client_kwargs["http_client"] = http_client
                    except Exception as e:
                        print(f"Warning: Could not configure proxy: {e}")
                        # Fall back to environment variables for proxy

            self.client = AsyncOpenAI(**client_kwargs)
            self.default_timeout = default_timeout

        async def get_completion(
            self, state: RunState[Ctx], agent: Agent[Ctx, Any], config: RunConfig[Ctx]
        ) -> Dict[str, Any]:
            """Get completion from the model with fallback support."""

            # Determine initial model to use
            primary_model = config.model_override or (
                agent.model_config.name if agent.model_config else "gpt-4o"
            )

            # Check if any message contains image content or image attachments
            has_image_content = any(
                (
                    isinstance(msg.content, list)
                    and any(part.type == "image_url" for part in msg.content)
                )
                or (msg.attachments and any(att.kind == "image" for att in msg.attachments))
                for msg in state.messages
            )

            if has_image_content:
                supports_vision = await _is_vision_model(primary_model, base_url)
                if not supports_vision:
                    raise ValueError(
                        f"Model {primary_model} does not support vision capabilities. "
                        f"Please use a vision-capable model like gpt-4o, claude-3-5-sonnet, or gemini-1.5-pro."
                    )

            # Create system message
            system_message = {"role": "system", "content": agent.instructions(state)}

            # Convert messages to OpenAI format
            converted_messages = []
            for msg in state.messages:
                converted_msg = await _convert_message(msg)
                converted_messages.append(converted_msg)

            messages = [system_message] + converted_messages

            # Convert tools to OpenAI format
            tools = None
            if agent.tools:
                # Check if we should inline schema refs
                inline_refs = (
                    agent.model_config.inline_tool_schemas if agent.model_config else False
                )
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.schema.name,
                            "description": tool.schema.description,
                            "parameters": _pydantic_to_json_schema(
                                tool.schema.parameters, inline_refs=inline_refs or False
                            ),
                        },
                    }
                    for tool in agent.tools
                ]

            # Determine tool choice behavior
            last_message = state.messages[-1] if state.messages else None
            is_after_tool_call = last_message and (
                last_message.role == ContentRole.TOOL or last_message.role == "tool"
            )

            # Helper function to make API call with a specific model
            async def _make_completion_call(model_name: str) -> Dict[str, Any]:
                # Prepare request parameters
                request_params = {"model": model_name, "messages": messages, "stream": False}

                # Add session_id from conversation_id for LiteLLM tracking
                if config.conversation_id:
                    request_params["extra_body"] = {"litellm_session_id": config.conversation_id}

                # Add optional parameters
                if agent.model_config:
                    if agent.model_config.temperature is not None:
                        request_params["temperature"] = agent.model_config.temperature
                    # Use agent's max_tokens if set, otherwise fall back to config's max_tokens
                    max_tokens = agent.model_config.max_tokens
                    if max_tokens is None:
                        max_tokens = config.max_tokens
                    if max_tokens is not None:
                        request_params["max_tokens"] = max_tokens
                elif config.max_tokens is not None:
                    # No model_config but config has max_tokens
                    request_params["max_tokens"] = config.max_tokens

                if tools:
                    request_params["tools"] = tools
                    # Always set tool_choice to auto when tools are available
                    request_params["tool_choice"] = "auto"

                if agent.output_codec:
                    request_params["response_format"] = {"type": "json_object"}

                # Make the API call with retry handling
                async def _api_call():
                    return await self.client.chat.completions.create(**request_params)

                # Use retry wrapper to track retries in Langfuse
                return await _retry_with_events(
                    _api_call,
                    state,
                    config,
                    operation_name="llm_call",
                    max_retries=3,
                    backoff_factor=1.0,
                )

            # Try primary model first
            last_exception = None
            current_model = primary_model

            try:
                response = await _make_completion_call(current_model)
            except Exception as e:
                last_exception = e

                # Classify the error to determine which fallback list to use
                fallback_type, reason = _classify_error_for_fallback(e)

                # Determine which fallback list to use
                fallback_models = []
                if fallback_type == "content_policy" and config.content_policy_fallbacks:
                    fallback_models = config.content_policy_fallbacks
                elif fallback_type == "context_window" and config.context_window_fallbacks:
                    fallback_models = config.context_window_fallbacks
                elif config.fallbacks:
                    fallback_models = config.fallbacks

                # Try fallback models
                if fallback_models:
                    print(
                        f"[JAF:FALLBACK] Primary model '{current_model}' failed with {reason}. "
                        f"Trying {len(fallback_models)} fallback model(s)..."
                    )

                    for i, fallback_model in enumerate(fallback_models, 1):
                        try:
                            # Emit fallback event
                            if config.on_event:
                                fallback_event = FallbackEvent(
                                    data=FallbackEventData(
                                        from_model=current_model,
                                        to_model=fallback_model,
                                        reason=reason,
                                        fallback_type=fallback_type,
                                        attempt=i,
                                        trace_id=state.trace_id,
                                        run_id=state.run_id,
                                        error_details={
                                            "error_type": type(last_exception).__name__,
                                            "error_message": str(last_exception),
                                        },
                                    )
                                )
                                config.on_event(fallback_event)

                            print(
                                f"[JAF:FALLBACK] Attempting fallback {i}/{len(fallback_models)}: {fallback_model}"
                            )

                            # Try the fallback model
                            response = await _make_completion_call(fallback_model)
                            current_model = fallback_model
                            print(
                                f"[JAF:FALLBACK] Successfully used fallback model: {fallback_model}"
                            )
                            break  # Success - exit the fallback loop

                        except Exception as fallback_error:
                            last_exception = fallback_error
                            print(
                                f"[JAF:FALLBACK] Fallback model '{fallback_model}' also failed: {fallback_error}"
                            )

                            # If this was the last fallback, re-raise
                            if i == len(fallback_models):
                                print(
                                    f"[JAF:FALLBACK] All fallback models exhausted. Raising last exception."
                                )
                                raise
                else:
                    # No fallbacks configured, re-raise original exception
                    raise

            # Return in the expected format that the engine expects
            choice = response.choices[0]

            # Convert tool_calls to dict format if present
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

            # Extract usage data with detailed cache information
            usage_data = None
            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                # Extract cache-related fields if available (for prompt caching support)
                if hasattr(response.usage, "cache_creation_input_tokens"):
                    usage_data["cache_creation_input_tokens"] = (
                        response.usage.cache_creation_input_tokens
                    )
                if hasattr(response.usage, "cache_read_input_tokens"):
                    usage_data["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

                # Extract detailed token breakdowns
                if (
                    hasattr(response.usage, "prompt_tokens_details")
                    and response.usage.prompt_tokens_details
                ):
                    details = {}
                    if hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
                        details["cached_tokens"] = (
                            response.usage.prompt_tokens_details.cached_tokens
                        )
                    if hasattr(response.usage.prompt_tokens_details, "audio_tokens"):
                        details["audio_tokens"] = response.usage.prompt_tokens_details.audio_tokens
                    if details:
                        usage_data["prompt_tokens_details"] = details

                if (
                    hasattr(response.usage, "completion_tokens_details")
                    and response.usage.completion_tokens_details
                ):
                    details = {}
                    if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                        details["reasoning_tokens"] = (
                            response.usage.completion_tokens_details.reasoning_tokens
                        )
                    if hasattr(response.usage.completion_tokens_details, "audio_tokens"):
                        details["audio_tokens"] = (
                            response.usage.completion_tokens_details.audio_tokens
                        )
                    if details:
                        usage_data["completion_tokens_details"] = details

            return {
                "id": response.id,
                "created": response.created,
                "model": response.model,
                "system_fingerprint": response.system_fingerprint,
                "message": {"content": choice.message.content, "tool_calls": tool_calls},
                "usage": usage_data,
                "prompt": messages,
            }

        async def get_completion_stream(
            self, state: RunState[Ctx], agent: Agent[Ctx, Any], config: RunConfig[Ctx]
        ) -> AsyncIterator[CompletionStreamChunk]:
            """
            Stream completion chunks from the model provider, yielding text deltas and tool-call deltas.
            Uses OpenAI-compatible streaming via LiteLLM endpoint.
            """
            # Determine model to use
            model = config.model_override or (
                agent.model_config.name if agent.model_config else "gpt-4o"
            )

            # Create system message
            system_message = {"role": "system", "content": agent.instructions(state)}

            # Convert messages to OpenAI format
            converted_messages = []
            for msg in state.messages:
                converted_msg = await _convert_message(msg)
                converted_messages.append(converted_msg)

            messages = [system_message] + converted_messages

            # Convert tools to OpenAI format
            tools = None
            if agent.tools:
                # Check if we should inline schema refs
                inline_refs = (
                    agent.model_config.inline_tool_schemas if agent.model_config else False
                )
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.schema.name,
                            "description": tool.schema.description,
                            "parameters": _pydantic_to_json_schema(
                                tool.schema.parameters, inline_refs=inline_refs or False
                            ),
                        },
                    }
                    for tool in agent.tools
                ]

            # Determine tool choice behavior
            last_message = state.messages[-1] if state.messages else None
            is_after_tool_call = last_message and (
                last_message.role == ContentRole.TOOL or last_message.role == "tool"
            )

            # Prepare request parameters
            request_params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }

            # Add session_id from conversation_id for LiteLLM tracking
            if config.conversation_id:
                request_params["extra_body"] = {"litellm_session_id": config.conversation_id}

            # Add optional parameters
            if agent.model_config:
                if agent.model_config.temperature is not None:
                    request_params["temperature"] = agent.model_config.temperature
                # Use agent's max_tokens if set, otherwise fall back to config's max_tokens
                max_tokens = agent.model_config.max_tokens
                if max_tokens is None:
                    max_tokens = config.max_tokens
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
            elif config.max_tokens is not None:
                # No model_config but config has max_tokens
                request_params["max_tokens"] = config.max_tokens

            if tools:
                request_params["tools"] = tools
                # Set tool_choice to auto when tools are available
                request_params["tool_choice"] = "auto"

            if agent.output_codec:
                request_params["response_format"] = {"type": "json_object"}

            # Enable streaming
            request_params["stream"] = True

            # Use async streaming directly with AsyncOpenAI
            stream = await self.client.chat.completions.create(**request_params)

            async for chunk in stream:
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
                            yield CompletionStreamChunk(delta=content_delta, raw=raw_obj)

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
                                    args_delta = (
                                        getattr(fn, "arguments", None) if fn is not None else None
                                    )

                                    yield CompletionStreamChunk(
                                        tool_call_delta=ToolCallDelta(
                                            index=idx,
                                            id=tc_id,
                                            type="function",
                                            function=ToolCallFunctionDelta(
                                                name=fn_name, arguments_delta=args_delta
                                            ),
                                        ),
                                        raw=raw_obj,
                                    )
                                except Exception:
                                    # Skip malformed tool-call deltas
                                    continue

                    # Completion ended
                    if finish_reason:
                        yield CompletionStreamChunk(
                            is_done=True, finish_reason=finish_reason, raw=raw_obj
                        )
                except Exception:
                    # Skip individual chunk errors, keep streaming
                    continue

    return LiteLLMProvider()

_RESPONSES_API_REQUIRED_MARKERS = (
    "use /v1/responses instead",
    "please use /v1/responses",
)


def _requires_responses_api(error: Exception) -> bool:
    """Detect the OpenAI/Azure error signaling a model only supports
    reasoning + tools via the Responses API, not Chat Completions."""
    message = str(error).lower()
    return any(marker in message for marker in _RESPONSES_API_REQUIRED_MARKERS)


def _is_previous_response_not_found(error: Exception) -> bool:
    """Detect Azure/OpenAI's error for a previous_response_id that's expired
    (30-day retention) or was never stored (store=false, evicted WS cache)."""
    return "previous_response_not_found" in str(error).lower()


def _chat_content_part_to_responses_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one Chat-Completions content part (`text` / `image_url` /
    `file`) into the Responses API's `input_text` / `input_image` /
    `input_file` shape. Parts already in Responses shape (or of an unknown
    type) pass through unchanged so callers don't need to know which shape
    upstream code produced."""
    part_type = part.get("type")

    if part_type == "text":
        return {"type": "input_text", "text": part.get("text", "")}

    if part_type == "image_url":
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        detail = image_url.get("detail") if isinstance(image_url, dict) else None
        responses_part: Dict[str, Any] = {"type": "input_image", "image_url": url}
        responses_part["detail"] = detail or "auto"
        return responses_part

    if part_type == "file":
        file_obj = part.get("file") or {}
        responses_part = {"type": "input_file"}
        if file_obj.get("file_data"):
            responses_part["file_data"] = file_obj["file_data"]
        if file_obj.get("file_id"):
            responses_part["file_id"] = file_obj["file_id"]
        if file_obj.get("file_url"):
            responses_part["file_url"] = file_obj["file_url"]
        if file_obj.get("filename"):
            responses_part["filename"] = file_obj["filename"]
        return responses_part

    # Already Responses-shaped (input_text/input_image/input_file/...) or
    # unrecognized - pass through rather than dropping content silently.
    return part


def _chat_content_to_responses_content(content: Any) -> Any:
    """Chat Completions content is either a plain string or a list of
    content parts; only the list form needs type translation for Responses."""
    if not isinstance(content, list):
        return content
    return [_chat_content_part_to_responses_part(part) for part in content]


def _chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a Chat-Completions-style `messages` list into the Responses API's
    `input` item list.

    Chat Completions folds tool calls into the assistant message and matches
    tool results by role="tool" + tool_call_id; Responses represents both as
    standalone items keyed by call_id, so those need explicit expansion.
    Multi-part content (attachments) also uses different part `type` values
    between the two APIs (`text`/`image_url`/`file` vs `input_text`/
    `input_image`/`input_file`), so it's translated here too.
    """
    input_items: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": msg.get("content") or "",
                }
            )
            continue

        if role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            if content:
                input_items.append(
                    {"role": "assistant", "content": _chat_content_to_responses_content(content)}
                )

            for tc in tool_calls:
                function = tc.get("function") or {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments", ""),
                    }
                )
            continue

        # system / user messages: plain string content maps through
        # unchanged; structured content-part lists need type translation.
        input_items.append(
            {"role": role, "content": _chat_content_to_responses_content(msg.get("content"))}
        )

    return input_items


def _chat_tools_to_responses_tools(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Flatten Chat Completions' `{"type":"function","function":{...}}` tool
    specs into the Responses API's flat `{"type":"function","name":...}` shape."""
    if not tools:
        return None

    responses_tools = []
    for tool in tools:
        fn = tool.get("function") or {}
        responses_tools.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    return responses_tools


def _responses_output_to_message(response: Any) -> Dict[str, Any]:
    """Parse a Responses API result into the {"content", "tool_calls"} shape
    the rest of JAF already expects from Chat Completions."""
    content_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if text:
                    content_parts.append(text)

        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", "") or "",
                    },
                }
            )

    return {
        "content": "\n".join(content_parts) if content_parts else None,
        "tool_calls": tool_calls or None,
    }


def _responses_usage_to_chat_usage(response: Any) -> Dict[str, Any]:
    """Normalize Responses API usage into the prompt/completion_tokens shape
    used elsewhere in JAF for cost tracking."""
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    usage_data = {
        "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }

    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None) if output_details else None
    if reasoning_tokens is not None:
        usage_data["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}

    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", None) if input_details else None
    if cached_tokens is not None:
        usage_data["prompt_tokens_details"] = {"cached_tokens": cached_tokens}

    return usage_data


# Azure Responses API WebSocket transport. litellm has no WS support for this,
# so we talk to wss://{resource}.openai.azure.com/openai/v1/responses directly.

# Azure caps connections at 60 min; reconnect a bit early to avoid racing it.
_WS_CONNECTION_LIFETIME_SECONDS = 55 * 60

# Auth/routing kwargs that don't belong in the response.create JSON body.
_WS_PAYLOAD_EXCLUDED_KEYS = {
    "api_key",
    "api_base",
    "api_version",
    "azure_deployment",
    "custom_llm_provider",
    "timeout",
    "litellm_session_id",
    "stream",
}


def _azure_responses_ws_url(api_base: str) -> str:
    """api_base -> Responses API WS URL. Accepts an existing ws(s):// URL too;
    path is always normalized to /openai/v1/responses."""
    parsed = urlsplit(api_base)
    if not parsed.netloc:
        raise ValueError(f"websocket=True needs a full api_base URL, got: {api_base!r}")
    scheme = "wss" if parsed.scheme in ("https", "wss", "") else "ws"
    return urlunsplit((scheme, parsed.netloc, "/openai/v1/responses", "", ""))


def _responses_params_to_ws_payload(responses_params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop auth/routing kwargs; `model` becomes the deployment name."""
    payload = {k: v for k, v in responses_params.items() if k not in _WS_PAYLOAD_EXCLUDED_KEYS}

    deployment = responses_params.get("azure_deployment")
    if deployment:
        payload["model"] = deployment
    elif isinstance(payload.get("model"), str) and "/" in payload["model"]:
        payload["model"] = payload["model"].split("/")[-1]

    return payload


def _trim_messages_for_continuation(
    messages: List[Dict[str, Any]], response_id_index: Optional[int], server_history: bool
) -> List[Dict[str, Any]]:
    """messages[0] is the system message; messages[1:] maps 1:1 to state.messages.
    When server_history is on and a boundary is known, keep the system message
    plus only items new since it -- Azure reconstructs the rest via
    previous_response_id. Falls back to the full list otherwise."""
    if not server_history or response_id_index is None:
        return messages
    return [messages[0]] + messages[1 + response_id_index :]


def _dict_to_namespace(value: Any) -> Any:
    """Recursively turn dicts/lists into attribute-accessible objects. Used when
    strict pydantic validation of a raw WS payload fails -- getattr-based access
    (what the rest of this module expects) doesn't work on a plain dict."""
    if isinstance(value, dict):
        import types

        ns = types.SimpleNamespace()
        for k, v in value.items():
            setattr(ns, k, _dict_to_namespace(v))
        return ns
    if isinstance(value, list):
        return [_dict_to_namespace(v) for v in value]
    return value


def _patch_azure_response_for_validation(value: Dict[str, Any]) -> Dict[str, Any]:
    """Azure's Responses API omits a couple of fields the OpenAI SDK's Response
    model requires (schema drift, not an error on Azure's part) -- fill in
    harmless defaults so validation succeeds instead of always falling back."""
    patched = dict(value)
    usage = patched.get("usage")
    if isinstance(usage, dict):
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict) and "cache_write_tokens" not in input_details:
            usage = dict(usage)
            usage["input_tokens_details"] = {**input_details, "cache_write_tokens": 0}
            patched["usage"] = usage
    return patched


class _WSEventProxy(dict):
    """Dict -> attribute access, so raw WS events reuse the existing (litellm
    object-shaped) event translation code unchanged."""

    def __getattr__(self, name: str) -> Any:
        value = self.get(name)
        if name == "response" and isinstance(value, dict):
            from openai.types.responses import Response as _OpenAIResponse

            try:
                return _OpenAIResponse.model_validate(_patch_azure_response_for_validation(value))
            except Exception:
                return _dict_to_namespace(value)
        return value


class _AzureResponsesWebSocketConnection:
    """One persistent WS connection, owned by a provider instance. Caller must
    reuse the provider across a session's turns for this to help. Azure allows
    one response.create in flight per connection, so calls are serialized."""

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str],
        default_timeout: Optional[float],
        ping_interval: Optional[float] = 20,
        ping_timeout: Optional[float] = 60,
    ):
        self._url = _azure_responses_ws_url(api_base)
        self._api_key = api_key
        self._default_timeout = default_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._ws: Optional[Any] = None
        self._connected_at: Optional[float] = None
        self._dirty = False
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> None:
        stale = (
            self._ws is None
            or self._ws.state != State.OPEN
            or self._connected_at is None
            or self._dirty
            or (time.monotonic() - self._connected_at) > _WS_CONNECTION_LIFETIME_SECONDS
        )
        if not stale:
            return

        await self._close()
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._ws = await websockets.connect(
            self._url,
            additional_headers=headers,
            open_timeout=self._default_timeout,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        self._connected_at = time.monotonic()
        self._dirty = False

    async def _close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._connected_at = None

    async def aclose(self) -> None:
        """Close the socket. For owners of a pooled connection."""
        await self._close()

    async def create_response(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Send one response.create, yield raw events until completed/failed.
        Retry once on disconnect before the first event. Replay is best-effort:
        Azure may have processed the request already, duplicating work and usage.
        """
        body = {"type": "response.create", **payload}

        async with self._lock:
            for attempt in (0, 1):
                yielded = False
                try:
                    await self._ensure_connected()
                    self._dirty = True
                    await self._ws.send(json.dumps(body))

                    while True:
                        raw = await self._ws.recv()
                        event = json.loads(raw)
                        event_type = event.get("type")
                        self._dirty = event_type != "response.completed"
                        yielded = True
                        yield event

                        if event_type == "response.completed":
                            return
                        if event_type in ("response.failed", "error"):
                            error = event.get("error") or (event.get("response") or {}).get("error")
                            code = (error or {}).get("code")
                            if code == "websocket_connection_limit_reached":
                                await self._close()
                            raise RuntimeError(f"Azure Responses WebSocket error: {error}")
                except ConnectionClosed as exc:
                    await self._close()
                    if yielded or attempt:
                        raise
                    logger.warning(
                        "Azure Responses WebSocket closed before first event; "
                        "retrying once (received_close_code=%s, sent_close_code=%s)",
                        exc.rcvd.code if exc.rcvd else None,
                        exc.sent.code if exc.sent else None,
                    )


AzureResponsesWebSocketConnection = _AzureResponsesWebSocketConnection


def _extract_reasoning_for_responses(request_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pop the Chat-Completions-style bare `reasoning_effort` kwarg out of a
    request dict and translate it to the Responses API's nested `reasoning`
    shape. Falls back to an already-nested `reasoning` kwarg if present."""
    effort = request_params.pop("reasoning_effort", None)
    if effort is not None:
        return {"effort": effort}
    return request_params.pop("reasoning", None)


def _chat_request_params_to_responses_params(
    request_params: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build the kwargs for `litellm.aresponses()` from the Chat-Completions
    style request dict JAF already assembled (model/api_key/api_base/etc. are
    shared verbatim; messages/tools/reasoning need reshaping)."""
    responses_params = {
        k: v
        for k, v in request_params.items()
        if k not in ("messages", "tools", "tool_choice", "stream", "stream_options")
    }

    reasoning = _extract_reasoning_for_responses(responses_params)
    if reasoning:
        responses_params["reasoning"] = reasoning

    if "max_tokens" in responses_params:
        responses_params["max_output_tokens"] = responses_params.pop("max_tokens")

    response_format = responses_params.pop("response_format", None)
    if response_format and response_format.get("type"):
        responses_params["text"] = {"format": {"type": response_format["type"]}}

    responses_tools = _chat_tools_to_responses_tools(tools)
    if responses_tools:
        responses_params["tools"] = responses_tools
        responses_params["tool_choice"] = "auto"

    responses_params["input"] = _chat_messages_to_responses_input(messages)

    return responses_params


def make_litellm_sdk_provider(
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    base_url: Optional[str] = None,
    default_timeout: Optional[float] = None,
    api_type: str = "auto",
    websocket: bool = False,
    ws_connection: Optional[Any] = None,
    server_history: bool = True,
    **litellm_kwargs: Any,
) -> ModelProvider[Ctx]:
    """
    Create a LiteLLM SDK-based model provider with universal provider support.

    LiteLLM automatically detects the provider from the model name and handles
    API key management through environment variables or direct parameters.

    Args:
        api_key: API key for the provider (optional, can use env vars)
        model: Model name (e.g., "gpt-4", "claude-3-sonnet", "gemini-pro", "llama2", etc.)
        base_url: Optional base URL for custom endpoints
        default_timeout: Default timeout for model API calls in seconds
        api_type: Which OpenAI-style API to call. One of:
                  - "auto" (default): use Chat Completions as today. If the
                    provider rejects a request because the model requires the
                    Responses API (newer OpenAI/Azure reasoning models only
                    support `reasoning_effort` + tools on `/v1/responses`, not
                    `/v1/chat/completions`), automatically retry that call via
                    the Responses API and remember the choice for that model
                    for the lifetime of this provider instance, so later calls
                    skip straight past the failing round trip.
                  - "responses": always call the Responses API
                    (`litellm.aresponses`). Use this when the caller already
                    knows the model requires it.
                  - "chat_completions": always call Chat Completions
                    (`litellm.acompletion`) with no Responses API fallback -
                    the exact pre-existing behavior.
                  Existing callers are unaffected by this parameter: "auto"
                  only engages the Responses API path for models that actively
                  reject Chat Completions, and is a no-op for every model that
                  already works today.
        websocket: Use a persistent WebSocket to Azure's Responses API instead
                  of per-call HTTPS. Requires api_type="responses" (or "auto"
                  once it falls through). One connection per provider instance,
                  reused across calls (or supply ws_connection to share one
                  from a pool) -- reuse the same provider across a
                  session's turns to get the benefit. Default False, no effect
                  on existing callers.
        server_history: Only meaningful with the Responses API. True (default):
                  once a response_id is available, only send messages new since
                  it plus previous_response_id -- Azure reconstructs the rest
                  server-side. False: always send the full conversation as
                  `input`, never reference a previous response, even if one is
                  available -- for callers that don't want Azure holding any
                  server-side state. Chat Completions is unaffected either way.
        **litellm_kwargs: Additional arguments passed to litellm.completion()
                         Common examples:
                         - vertex_project: "your-project" (for Google models)
                         - vertex_location: "us-central1" (for Google models)
                         - azure_deployment: "your-deployment" (for Azure OpenAI)
                         - api_base: "https://your-endpoint.com" (custom endpoint)
                         - custom_llm_provider: "custom_provider_name"

    Returns:
        ModelProvider instance

    Examples:
        # OpenAI
        make_litellm_sdk_provider(api_key="sk-...", model="gpt-4")

        # Anthropic Claude
        make_litellm_sdk_provider(api_key="sk-ant-...", model="claude-3-sonnet-20240229")

        # Google Gemini
        make_litellm_sdk_provider(model="gemini-pro", vertex_project="my-project")

        # Ollama (local)
        make_litellm_sdk_provider(model="ollama/llama2", base_url="http://localhost:11434")

        # Azure OpenAI
        make_litellm_sdk_provider(
            model="azure/gpt-4",
            api_key="your-azure-key",
            azure_deployment="gpt-4-deployment",
            api_base="https://your-resource.openai.azure.com"
        )

        # Hugging Face
        make_litellm_sdk_provider(
            model="huggingface/microsoft/DialoGPT-medium",
            api_key="hf_..."
        )

        # Any custom provider
        make_litellm_sdk_provider(
            model="custom_provider/model-name",
            api_key="your-key",
            custom_llm_provider="your_provider"
        )
    """

    if websocket and api_type == "chat_completions":
        raise ValueError("websocket=True requires api_type='responses' or 'auto'")

    class LiteLLMSDKProvider:
        def __init__(self):
            self.api_key = api_key
            self.model = model
            self.base_url = base_url
            self.default_timeout = default_timeout
            self.litellm_kwargs = litellm_kwargs
            self.api_type = api_type
            self.websocket = websocket
            self.server_history = server_history
            self._responses_only_models: set = set()
            self._ws_connection: Optional[_AzureResponsesWebSocketConnection] = ws_connection
            self._owns_ws_connection = ws_connection is None

        def _wants_responses_api(self, model_name: str) -> bool:
            return self.api_type == "responses" or (
                self.api_type == "auto" and model_name in self._responses_only_models
            )

        async def aclose(self) -> None:
            if self._ws_connection is not None and self._owns_ws_connection:
                await self._ws_connection._close()
            self._ws_connection = None

        def _get_ws_connection(self) -> _AzureResponsesWebSocketConnection:
            if self._ws_connection is None:
                api_base = self.litellm_kwargs.get("api_base") or self.base_url
                self._ws_connection = _AzureResponsesWebSocketConnection(
                    api_base=api_base, api_key=self.api_key, default_timeout=self.default_timeout
                )
            return self._ws_connection

        async def _call_responses_api_via_ws(self, responses_params: Dict[str, Any]) -> Any:
            """Drive one turn over the WS connection, return the same typed
            Response object litellm.aresponses() would."""
            from openai.types.responses import Response as _OpenAIResponse

            payload = _responses_params_to_ws_payload(responses_params)
            final_response = None
            async for event in self._get_ws_connection().create_response(payload):
                if event.get("type") == "response.completed":
                    final_response = event.get("response")
                    break

            if final_response is None:
                raise RuntimeError("Responses WebSocket stream ended without response.completed")
            try:
                return _OpenAIResponse.model_validate(
                    _patch_azure_response_for_validation(final_response)
                )
            except Exception:
                return _dict_to_namespace(final_response)

        async def _call_responses_api(
            self,
            model_name: str,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]],
            request_params: Dict[str, Any],
            state: RunState[Ctx],
            config: RunConfig[Ctx],
        ) -> Dict[str, Any]:
            """Non-streaming call via the Responses API, adapted back to the
            Chat-Completions-shaped dict the rest of JAF expects."""
            responses_params = _chat_request_params_to_responses_params(
                request_params, messages, tools
            )

            async def _api_call():
                if self.websocket:
                    return await self._call_responses_api_via_ws(responses_params)
                return await litellm.aresponses(**responses_params)

            response = await _retry_with_events(
                _api_call,
                state,
                config,
                operation_name="llm_call",
                max_retries=3,
                backoff_factor=1.0,
            )

            message_dict = _responses_output_to_message(response)
            usage_data = _responses_usage_to_chat_usage(response)
            actual_model = getattr(response, "model", model_name)

            # CRITICAL: Embed usage and model here so trace collector can find them
            message_dict["_usage"] = usage_data
            message_dict["_model"] = actual_model

            return {
                "id": getattr(response, "id", None),
                "created": getattr(response, "created_at", None),
                "model": actual_model,
                "system_fingerprint": None,
                "message": message_dict,
                "usage": usage_data,
                "prompt": messages,
            }

        async def _call_responses_api_checked(
            self,
            model_name: str,
            full_messages: List[Dict[str, Any]],
            responses_messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]],
            request_params: Dict[str, Any],
            state: RunState[Ctx],
            config: RunConfig[Ctx],
        ) -> Dict[str, Any]:
            """Call the Responses API; if previous_response_id has expired or
            was never stored, retry once with full history and no reference."""
            try:
                return await self._call_responses_api(
                    model_name, responses_messages, tools, dict(request_params), state, config
                )
            except Exception as e:
                if "previous_response_id" in request_params and _is_previous_response_not_found(e):
                    fallback_params = {
                        k: v for k, v in request_params.items() if k != "previous_response_id"
                    }
                    return await self._call_responses_api(
                        model_name, full_messages, tools, fallback_params, state, config
                    )
                raise

        async def get_completion(
            self, state: RunState[Ctx], agent: Agent[Ctx, Any], config: RunConfig[Ctx]
        ) -> Dict[str, Any]:
            """Get completion from the model using LiteLLM SDK."""

            # Determine model to use
            model_name = config.model_override or self.model

            # Create system message
            system_message = {"role": "system", "content": agent.instructions(state)}

            # Convert messages to OpenAI format
            messages = [system_message]
            for msg in state.messages:
                converted_msg = await _convert_message(msg)
                messages.append(converted_msg)

            # Convert tools to OpenAI format
            tools = None
            if agent.tools:
                # Check if we should inline schema refs
                inline_refs = (
                    agent.model_config.inline_tool_schemas if agent.model_config else False
                )
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.schema.name,
                            "description": tool.schema.description,
                            "parameters": _pydantic_to_json_schema(
                                tool.schema.parameters, inline_refs=inline_refs or False
                            ),
                        },
                    }
                    for tool in agent.tools
                ]

            # Prepare request parameters for LiteLLM
            request_params = {"model": model_name, "messages": messages, **self.litellm_kwargs}

            # Add session_id from conversation_id for LiteLLM tracking
            if config.conversation_id:
                request_params["litellm_session_id"] = config.conversation_id

            # Add API key if provided
            if self.api_key:
                request_params["api_key"] = self.api_key

            # Add optional parameters
            if agent.model_config:
                if agent.model_config.temperature is not None:
                    request_params["temperature"] = agent.model_config.temperature
                # Use agent's max_tokens if set, otherwise fall back to config's max_tokens
                max_tokens = agent.model_config.max_tokens
                if max_tokens is None:
                    max_tokens = config.max_tokens
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
            elif config.max_tokens is not None:
                # No model_config but config has max_tokens
                request_params["max_tokens"] = config.max_tokens

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if agent.output_codec:
                request_params["response_format"] = {"type": "json_object"}

            # LiteLLM will use api_base from kwargs or base_url parameter
            if self.base_url:
                request_params["api_base"] = self.base_url

            if self._wants_responses_api(model_name):
                responses_messages = messages
                if self.server_history and state.response_id:
                    request_params["previous_response_id"] = state.response_id
                    responses_messages = _trim_messages_for_continuation(
                        messages, state.response_id_index, self.server_history
                    )
                return await self._call_responses_api_checked(
                    model_name, messages, responses_messages, tools, dict(request_params), state, config
                )

            # Make the API call using litellm with retry handling
            async def _api_call():
                return await litellm.acompletion(**request_params)

            # Use retry wrapper to track retries in Langfuse
            try:
                response = await _retry_with_events(
                    _api_call,
                    state,
                    config,
                    operation_name="llm_call",
                    max_retries=3,
                    backoff_factor=1.0,
                )
            except Exception as e:
                if self.api_type == "auto" and _requires_responses_api(e):
                    self._responses_only_models.add(model_name)
                    responses_messages = messages
                    if self.server_history and state.response_id:
                        request_params["previous_response_id"] = state.response_id
                        responses_messages = _trim_messages_for_continuation(
                            messages, state.response_id_index, self.server_history
                        )
                    return await self._call_responses_api_checked(
                        model_name, messages, responses_messages, tools, dict(request_params), state, config
                    )
                raise

            # Return in the expected format that the engine expects
            choice = response.choices[0]

            # Convert tool_calls to dict format if present
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

            # Extract usage data with detailed cache information - ALWAYS return a dict with defaults for Langfuse cost tracking
            # Initialize with zeros as defensive default (matches AzureDirectProvider pattern)
            usage_data = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            actual_model = getattr(response, "model", model_name)

            if response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                # Extract cache-related fields if available (for prompt caching support)
                if hasattr(response.usage, "cache_creation_input_tokens"):
                    usage_data["cache_creation_input_tokens"] = (
                        response.usage.cache_creation_input_tokens
                    )
                if hasattr(response.usage, "cache_read_input_tokens"):
                    usage_data["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

                # Extract detailed token breakdowns
                if (
                    hasattr(response.usage, "prompt_tokens_details")
                    and response.usage.prompt_tokens_details
                ):
                    details = {}
                    if hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
                        details["cached_tokens"] = (
                            response.usage.prompt_tokens_details.cached_tokens
                        )
                    if hasattr(response.usage.prompt_tokens_details, "audio_tokens"):
                        details["audio_tokens"] = response.usage.prompt_tokens_details.audio_tokens
                    if details:
                        usage_data["prompt_tokens_details"] = details

                if (
                    hasattr(response.usage, "completion_tokens_details")
                    and response.usage.completion_tokens_details
                ):
                    details = {}
                    if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                        details["reasoning_tokens"] = (
                            response.usage.completion_tokens_details.reasoning_tokens
                        )
                    if hasattr(response.usage.completion_tokens_details, "audio_tokens"):
                        details["audio_tokens"] = (
                            response.usage.completion_tokens_details.audio_tokens
                        )
                    if details:
                        usage_data["completion_tokens_details"] = details

            message_content = {
                "content": choice.message.content,
                "tool_calls": tool_calls,
                # CRITICAL: Embed usage and model here so trace collector can find them
                "_usage": usage_data,
                "_model": actual_model,
            }

            return {
                "id": response.id,
                "created": response.created,
                "model": actual_model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "message": message_content,
                "usage": usage_data,
                "prompt": messages,
            }

        async def _stream_via_responses_api(
            self,
            model_name: str,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]],
            request_params: Dict[str, Any],
        ) -> AsyncIterator[CompletionStreamChunk]:
            """Stream completion chunks via the Responses API, translated into
            the same CompletionStreamChunk deltas Chat Completions streaming
            produces so the engine's consumption logic doesn't need to change."""
            responses_params = _chat_request_params_to_responses_params(
                request_params, messages, tools
            )
            responses_params.pop("stream_options", None)
            responses_params["stream"] = True

            if self.websocket:
                payload = _responses_params_to_ws_payload(responses_params)
                connection = self._get_ws_connection()

                async def _ws_events():
                    async for event in connection.create_response(payload):
                        yield _WSEventProxy(event)

                stream = _ws_events()
            else:
                stream = await litellm.aresponses(**responses_params)

            # Responses events key tool calls by output_index; JAF's
            # ToolCallDelta expects a stable, densely-packed `index` per call.
            tool_call_indices: Dict[int, int] = {}
            next_tool_index = 0

            async for event in stream:
                try:
                    event_type = getattr(event, "type", None)
                    event_name = getattr(event_type, "value", event_type)

                    if event_name == "response.output_text.delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            yield CompletionStreamChunk(delta=delta)

                    elif event_name == "response.output_item.added":
                        item = getattr(event, "item", None) or {}
                        item_type = (
                            item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                        )
                        if item_type == "function_call":
                            output_index = getattr(event, "output_index", 0) or 0
                            if output_index not in tool_call_indices:
                                tool_call_indices[output_index] = next_tool_index
                                next_tool_index += 1
                            call_id = (
                                item.get("call_id")
                                if isinstance(item, dict)
                                else getattr(item, "call_id", None)
                            )
                            name = (
                                item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
                            )
                            yield CompletionStreamChunk(
                                tool_call_delta=ToolCallDelta(
                                    index=tool_call_indices[output_index],
                                    id=call_id,
                                    type="function",
                                    function=ToolCallFunctionDelta(name=name, arguments_delta=None),
                                )
                            )

                    elif event_name == "response.function_call_arguments.delta":
                        output_index = getattr(event, "output_index", 0) or 0
                        if output_index not in tool_call_indices:
                            tool_call_indices[output_index] = next_tool_index
                            next_tool_index += 1
                        delta = getattr(event, "delta", None)
                        yield CompletionStreamChunk(
                            tool_call_delta=ToolCallDelta(
                                index=tool_call_indices[output_index],
                                id=None,
                                type="function",
                                function=ToolCallFunctionDelta(name=None, arguments_delta=delta),
                            )
                        )

                    elif event_name == "response.completed":
                        response = getattr(event, "response", None)
                        finish_reason = "tool_calls" if tool_call_indices else "stop"
                        raw_obj = None
                        if response is not None:
                            usage_data = _responses_usage_to_chat_usage(response)
                            raw_obj = {
                                "usage": usage_data,
                                "model": getattr(response, "model", model_name),
                                "id": getattr(response, "id", None),
                            }
                        yield CompletionStreamChunk(
                            is_done=True, finish_reason=finish_reason, raw=raw_obj
                        )

                    elif event_name in ("response.failed", "error"):
                        error_obj = getattr(event, "response", None) or getattr(event, "error", None)
                        raise RuntimeError(f"Responses API stream failed: {error_obj}")
                except RuntimeError:
                    raise
                except Exception:
                    continue

        async def _stream_via_responses_api_checked(
            self,
            model_name: str,
            full_messages: List[Dict[str, Any]],
            responses_messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]],
            request_params: Dict[str, Any],
        ) -> AsyncIterator[CompletionStreamChunk]:
            """Stream via the Responses API; if previous_response_id has expired
            or was never stored, retry once with full history and no reference.
            Only safe to retry before any chunk has been yielded -- in practice
            this error is a request-level rejection returned before any output,
            so that's the case this handles."""
            try:
                async for chunk in self._stream_via_responses_api(
                    model_name, responses_messages, tools, dict(request_params)
                ):
                    yield chunk
            except Exception as e:
                if "previous_response_id" in request_params and _is_previous_response_not_found(e):
                    fallback_params = {
                        k: v for k, v in request_params.items() if k != "previous_response_id"
                    }
                    async for chunk in self._stream_via_responses_api(
                        model_name, full_messages, tools, fallback_params
                    ):
                        yield chunk
                    return
                raise

        async def get_completion_stream(
            self, state: RunState[Ctx], agent: Agent[Ctx, Any], config: RunConfig[Ctx]
        ) -> AsyncIterator[CompletionStreamChunk]:
            """
            Stream completion chunks from the model provider using LiteLLM SDK.
            """
            # Determine model to use
            model_name = config.model_override or self.model

            # Create system message
            system_message = {"role": "system", "content": agent.instructions(state)}

            # Convert messages to OpenAI format
            messages = [system_message]
            for msg in state.messages:
                converted_msg = await _convert_message(msg)
                messages.append(converted_msg)

            # Convert tools to OpenAI format
            tools = None
            if agent.tools:
                # Check if we should inline schema refs
                inline_refs = (
                    agent.model_config.inline_tool_schemas if agent.model_config else False
                )
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.schema.name,
                            "description": tool.schema.description,
                            "parameters": _pydantic_to_json_schema(
                                tool.schema.parameters, inline_refs=inline_refs or False
                            ),
                        },
                    }
                    for tool in agent.tools
                ]

            # Prepare request parameters for LiteLLM streaming
            request_params: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},  # Request usage data in streaming
                **self.litellm_kwargs,
            }

            # Add session_id from conversation_id for LiteLLM tracking
            if config.conversation_id:
                request_params["litellm_session_id"] = config.conversation_id

            # Add API key if provided
            if self.api_key:
                request_params["api_key"] = self.api_key

            # Add optional parameters
            if agent.model_config:
                if agent.model_config.temperature is not None:
                    request_params["temperature"] = agent.model_config.temperature
                # Use agent's max_tokens if set, otherwise fall back to config's max_tokens
                max_tokens = agent.model_config.max_tokens
                if max_tokens is None:
                    max_tokens = config.max_tokens
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
            elif config.max_tokens is not None:
                # No model_config but config has max_tokens
                request_params["max_tokens"] = config.max_tokens

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if agent.output_codec:
                request_params["response_format"] = {"type": "json_object"}

            # LiteLLM will use api_base from kwargs or base_url parameter
            if self.base_url:
                request_params["api_base"] = self.base_url

            if self._wants_responses_api(model_name):
                responses_messages = messages
                if self.server_history and state.response_id:
                    request_params["previous_response_id"] = state.response_id
                    responses_messages = _trim_messages_for_continuation(
                        messages, state.response_id_index, self.server_history
                    )
                async for chunk in self._stream_via_responses_api_checked(
                    model_name, messages, responses_messages, tools, dict(request_params)
                ):
                    yield chunk
                return

            # Stream using litellm
            try:
                stream = await litellm.acompletion(**request_params)
            except Exception as e:
                if self.api_type == "auto" and _requires_responses_api(e):
                    self._responses_only_models.add(model_name)
                    responses_messages = messages
                    if self.server_history and state.response_id:
                        request_params["previous_response_id"] = state.response_id
                        responses_messages = _trim_messages_for_continuation(
                            messages, state.response_id_index, self.server_history
                        )
                    async for chunk in self._stream_via_responses_api_checked(
                        model_name, messages, responses_messages, tools, dict(request_params)
                    ):
                        yield chunk
                    return
                raise

            accumulated_usage: Optional[Dict[str, int]] = None
            response_model: Optional[str] = None

            async for chunk in stream:
                try:
                    # Best-effort extraction of raw for debugging
                    try:
                        raw_obj = chunk.model_dump() if hasattr(chunk, "model_dump") else None

                        # Capture usage from chunk if present
                        if raw_obj and "usage" in raw_obj and raw_obj["usage"]:
                            accumulated_usage = raw_obj["usage"]

                        # Capture model from chunk if present
                        if raw_obj and "model" in raw_obj and raw_obj["model"]:
                            response_model = raw_obj["model"]

                    except Exception as e:
                        raw_obj = None

                    if raw_obj and "usage" in raw_obj and raw_obj["usage"]:
                        # Yield this chunk so engine.py can capture usage from raw
                        yield CompletionStreamChunk(delta="", raw=raw_obj)

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
                            # Include accumulated usage and model in raw_obj for engine
                            if raw_obj and (accumulated_usage or response_model):
                                if accumulated_usage:
                                    raw_obj["usage"] = accumulated_usage
                                if response_model:
                                    raw_obj["model"] = response_model
                            yield CompletionStreamChunk(delta=content_delta, raw=raw_obj)

                        # Tool call deltas
                        tool_calls = getattr(delta, "tool_calls", None)
                        if isinstance(tool_calls, list):
                            for tc in tool_calls:
                                try:
                                    idx = getattr(tc, "index", 0) or 0
                                    tc_id = getattr(tc, "id", None)
                                    fn = getattr(tc, "function", None)
                                    fn_name = getattr(fn, "name", None) if fn is not None else None
                                    args_delta = (
                                        getattr(fn, "arguments", None) if fn is not None else None
                                    )

                                    # Include accumulated usage and model in raw_obj
                                    if raw_obj and (accumulated_usage or response_model):
                                        if accumulated_usage:
                                            raw_obj["usage"] = accumulated_usage
                                        if response_model:
                                            raw_obj["model"] = response_model

                                    yield CompletionStreamChunk(
                                        tool_call_delta=ToolCallDelta(
                                            index=idx,
                                            id=tc_id,
                                            type="function",
                                            function=ToolCallFunctionDelta(
                                                name=fn_name, arguments_delta=args_delta
                                            ),
                                        ),
                                        raw=raw_obj,
                                    )
                                except Exception:
                                    continue

                    # Completion ended
                    if finish_reason:
                        # Include accumulated usage and model in final chunk
                        if raw_obj and (accumulated_usage or response_model):
                            if accumulated_usage:
                                raw_obj["usage"] = accumulated_usage
                            if response_model:
                                raw_obj["model"] = response_model
                        yield CompletionStreamChunk(
                            is_done=True, finish_reason=finish_reason, raw=raw_obj
                        )
                except Exception:
                    continue

    return LiteLLMSDKProvider()


async def _convert_message(msg: Message) -> Dict[str, Any]:
    """
    Handles all possible role types (string and enum) and content formats.
    """
    # Normalize role to handle both string and enum values
    role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role).lower()

    # Handle user messages
    if role_value in ("user", ContentRole.USER.value if hasattr(ContentRole, "USER") else "user"):
        if isinstance(msg.content, list):
            # Multi-part content
            return {
                "role": "user",
                "content": [_convert_content_part(part) for part in msg.content],
            }
        else:
            # Build message with attachments if available
            return await _build_chat_message_with_attachments("user", msg)

    # Handle assistant messages
    elif role_value in (
        "assistant",
        ContentRole.ASSISTANT.value if hasattr(ContentRole, "ASSISTANT") else "assistant",
    ):
        result = {
            "role": "assistant",
            "content": get_text_content(msg.content) or "",  # Ensure content is never None
        }

        # Add tool calls if present
        if msg.tool_calls and len(msg.tool_calls) > 0:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
                if tc.id and tc.function and tc.function.name  # Validate tool call structure
            ]

        return result

    # Handle system messages
    elif role_value in (
        "system",
        ContentRole.SYSTEM.value if hasattr(ContentRole, "SYSTEM") else "system",
    ):
        return {"role": "system", "content": get_text_content(msg.content) or ""}

    # Handle tool messages
    elif role_value in ("tool", ContentRole.TOOL.value if hasattr(ContentRole, "TOOL") else "tool"):
        if not msg.tool_call_id:
            raise ValueError(f"Tool message must have tool_call_id. Message: {msg}")

        return {
            "role": "tool",
            "content": get_text_content(msg.content) or "",
            "tool_call_id": msg.tool_call_id,
        }

    # Handle function messages (legacy support)
    elif role_value == "function":
        if not msg.tool_call_id:
            raise ValueError(f"Function message must have tool_call_id. Message: {msg}")

        return {
            "role": "function",
            "content": get_text_content(msg.content) or "",
            "name": getattr(msg, "name", "unknown_function"),
        }

    # Unknown role - provide helpful error message
    else:
        available_roles = ["user", "assistant", "system", "tool", "function"]
        raise ValueError(
            f"Unknown message role: {msg.role} (type: {type(msg.role)}). "
            f"Supported roles: {available_roles}. "
            f"Message content: {get_text_content(msg.content)[:100] if msg.content else 'None'}"
        )


def _convert_content_part(part: MessageContentPart) -> Dict[str, Any]:
    """Convert MessageContentPart to OpenAI format."""
    if part.type == "text":
        return {"type": "text", "text": part.text}
    elif part.type == "image_url":
        return {"type": "image_url", "image_url": part.image_url}
    elif part.type == "file":
        return {"type": "file", "file": part.file}
    else:
        raise ValueError(f"Unknown content part type: {part.type}")


async def _build_chat_message_with_attachments(role: str, msg: Message) -> Dict[str, Any]:
    """
    Build multi-part content for Chat Completions if attachments exist.
    Supports images via image_url and documents via content extraction.
    """
    has_attachments = msg.attachments and len(msg.attachments) > 0
    if not has_attachments:
        if role == "assistant":
            base_msg = {"role": "assistant", "content": get_text_content(msg.content)}
            if msg.tool_calls:
                base_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            return base_msg
        return {"role": "user", "content": get_text_content(msg.content)}

    parts = []
    text_content = get_text_content(msg.content)
    if text_content and text_content.strip():
        parts.append({"type": "text", "text": text_content})

    for att in msg.attachments:
        if att.kind == "image":
            # Prefer explicit URL; otherwise construct a data URL from base64
            url = att.url
            if not url and att.data and att.mime_type:
                # Validate base64 data size before creating data URL
                try:
                    # Estimate decoded size (base64 is ~4/3 of decoded size)
                    estimated_size = len(att.data) * 3 // 4

                    if estimated_size > MAX_IMAGE_BYTES:
                        print(
                            f"Warning: Skipping oversized image ({estimated_size} bytes > {MAX_IMAGE_BYTES}). "
                            f"Set JAF_MAX_IMAGE_BYTES env var to adjust limit."
                        )
                        parts.append(
                            {
                                "type": "text",
                                "text": f"[IMAGE SKIPPED: Size exceeds limit of {MAX_IMAGE_BYTES // 1024 // 1024}MB. "
                                f"Image name: {att.name or 'unnamed'}]",
                            }
                        )
                        continue

                    # Create data URL for valid-sized images
                    url = f"data:{att.mime_type};base64,{att.data}"
                except Exception as e:
                    print(f"Error processing image data: {e}")
                    parts.append(
                        {
                            "type": "text",
                            "text": f"[IMAGE ERROR: Failed to process image data. Image name: {att.name or 'unnamed'}]",
                        }
                    )
                    continue

            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})

        elif att.kind in ["document", "file"]:
            # Check if attachment has use_litellm_format flag or is a large document
            use_litellm_format = att.use_litellm_format is True

            if use_litellm_format and (att.url or att.data):
                # For now, fall back to content extraction since most providers don't support native file format
                # TODO: Add provider-specific file format support
                print(
                    f"Info: LiteLLM format requested for {att.name}, falling back to content extraction"
                )
                use_litellm_format = False

            if not use_litellm_format:
                # Extract document content if supported and we have data or URL
                if is_document_supported(att.mime_type) and (att.data or att.url):
                    try:
                        processed = await extract_document_content(att)
                        file_name = att.name or "document"
                        description = get_document_description(att.mime_type)

                        parts.append(
                            {
                                "type": "text",
                                "text": f"DOCUMENT: {file_name} ({description}):\n\n{processed.content}",
                            }
                        )
                    except DocumentProcessingError as e:
                        # Fallback to filename if extraction fails
                        label = att.name or att.format or att.mime_type or "attachment"
                        parts.append(
                            {
                                "type": "text",
                                "text": f"ERROR: Failed to process {att.kind}: {label} ({e})",
                            }
                        )
                else:
                    # Unsupported document type - show placeholder
                    label = att.name or att.format or att.mime_type or "attachment"
                    url_info = f" ({att.url})" if att.url else ""
                    parts.append(
                        {"type": "text", "text": f"ATTACHMENT: {att.kind}: {label}{url_info}"}
                    )

    base_msg = {"role": role, "content": parts}
    if role == "assistant" and msg.tool_calls:
        base_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]

    return base_msg


def _resolve_schema_refs(
    schema: Dict[str, Any], defs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Recursively resolve $ref references in a JSON schema by inlining definitions.

    Args:
        schema: The schema object to process (may contain $ref)
        defs: The $defs dictionary containing reusable definitions

    Returns:
        Schema with all references resolved inline
    """
    if defs is None:
        # Extract $defs from root schema if present
        defs = schema.get("$defs", {})

    # If this is a reference, resolve it
    if isinstance(schema, dict) and "$ref" in schema:
        ref_path = schema["$ref"]

        # Handle #/$defs/DefinitionName format
        if ref_path.startswith("#/$defs/"):
            def_name = ref_path.split("/")[-1]
            if def_name in defs:
                # Recursively resolve the definition (it might have refs too)
                resolved_def = _resolve_schema_refs(defs[def_name], defs)
                return resolved_def
            else:
                # If definition not found, return the original ref
                return schema
        else:
            # Other ref formats - return as is
            return schema

    # If this is a dict, recursively process all values
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            # Skip $defs as we're inlining them
            if key == "$defs":
                continue
            result[key] = _resolve_schema_refs(value, defs)
        return result

    # If this is a list, recursively process all items
    if isinstance(schema, list):
        return [_resolve_schema_refs(item, defs) for item in schema]

    # For primitive types, return as is
    return schema


def _pydantic_to_json_schema(
    model_class: type[BaseModel], inline_refs: bool = False
) -> Dict[str, Any]:
    """
    Convert a Pydantic model to JSON schema for OpenAI tools.

    Args:
        model_class: Pydantic model class
        inline_refs: If True, resolve $refs and inline $defs in the schema

    Returns:
        JSON schema dictionary
    """
    if hasattr(model_class, "model_json_schema"):
        # Pydantic v2
        schema = model_class.model_json_schema()
    else:
        # Pydantic v1 fallback
        schema = model_class.schema()

    # If inline_refs is True, resolve all references
    if inline_refs:
        schema = _resolve_schema_refs(schema)

    return schema
