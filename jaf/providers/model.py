"""
Model provider implementations for the JAF framework.

This module provides model providers that integrate with various LLM services,
starting with LiteLLM for multi-provider support.
"""

from typing import Any, Dict, Optional, TypeVar, AsyncIterator
import asyncio
import httpx
import time
import os
import base64

from openai import OpenAI
from pydantic import BaseModel

from ..core.types import (
    Agent, ContentRole, Message, ModelProvider, RunConfig, RunState, 
    CompletionStreamChunk, ToolCallDelta, ToolCallFunctionDelta,
    MessageContentPart, get_text_content
)
from ..core.proxy import ProxyConfig
from ..utils.document_processor import (
    extract_document_content, is_document_supported, 
    get_document_description, DocumentProcessingError
)

Ctx = TypeVar('Ctx')

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
    
    if cached and time.time() - cached['timestamp'] < VISION_MODEL_CACHE_TTL:
        return cached['supports']
    
    try:
        async with httpx.AsyncClient(timeout=VISION_API_TIMEOUT) as client:
            response = await client.get(
                f"{base_url}/model_group/info",
                headers={'accept': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                model_info = None
                
                if 'data' in data and isinstance(data['data'], list):
                    for m in data['data']:
                        if (m.get('model_group') == model or 
                            model in str(m.get('model_group', ''))):
                            model_info = m
                            break
                
                if model_info and 'supports_vision' in model_info:
                    result = model_info['supports_vision']
                    _vision_model_cache[cache_key] = {
                        'supports': result,
                        'timestamp': time.time()
                    }
                    return result
            else:
                print(f"Warning: Vision API returned status {response.status_code} for model {model}")
                
    except Exception as e:
        print(f"Warning: Vision API error for model {model}: {e}")
    
    # Fallback to known vision models
    known_vision_models = [
        'gpt-4-vision-preview',
        'gpt-4o',
        'gpt-4o-mini',
        'claude-sonnet-4',
        'claude-sonnet-4-20250514',
        'gemini-2.5-flash',
        'gemini-2.5-pro'
    ]
    
    is_known_vision_model = any(
        vision_model.lower() in model.lower()
        for vision_model in known_vision_models
    )
    
    _vision_model_cache[cache_key] = {
        'supports': is_known_vision_model,
        'timestamp': time.time()
    }
    
    return is_known_vision_model

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

            # Check if any message contains image content or image attachments
            has_image_content = any(
                (isinstance(msg.content, list) and
                 any(part.type == 'image_url' for part in msg.content)) or
                (msg.attachments and
                 any(att.kind == 'image' for att in msg.attachments))
                for msg in state.messages
            )

            if has_image_content:
                supports_vision = await _is_vision_model(model, base_url)
                if not supports_vision:
                    raise ValueError(
                        f"Model {model} does not support vision capabilities. "
                        f"Please use a vision-capable model like gpt-4o, claude-3-5-sonnet, or gemini-1.5-pro."
                    )

            # Create system message
            system_message = {
                "role": "system",
                "content": agent.instructions(state)
            }

            # Convert messages to OpenAI format
            converted_messages = []
            for msg in state.messages:
                converted_msg = await _convert_message(msg)
                converted_messages.append(converted_msg)
            
            messages = [system_message] + converted_messages

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
            converted_messages = []
            for msg in state.messages:
                converted_msg = await _convert_message(msg)
                converted_messages.append(converted_msg)
            
            messages = [system_message] + converted_messages

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

async def _convert_message(msg: Message) -> Dict[str, Any]:
    """Convert JAF Message to OpenAI message format with attachment support."""
    if msg.role == 'user':
        if isinstance(msg.content, list):
            # Multi-part content
            return {
                "role": "user",
                "content": [_convert_content_part(part) for part in msg.content]
            }
        else:
            # Build message with attachments if available
            return await _build_chat_message_with_attachments('user', msg)
    elif msg.role == 'assistant':
        result = {
            "role": "assistant",
            "content": get_text_content(msg.content),
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
            "content": get_text_content(msg.content),
            "tool_call_id": msg.tool_call_id
        }
    else:
        raise ValueError(f"Unknown message role: {msg.role}")


def _convert_content_part(part: MessageContentPart) -> Dict[str, Any]:
    """Convert MessageContentPart to OpenAI format."""
    if part.type == 'text':
        return {
            "type": "text",
            "text": part.text
        }
    elif part.type == 'image_url':
        return {
            "type": "image_url",
            "image_url": part.image_url
        }
    elif part.type == 'file':
        return {
            "type": "file",
            "file": part.file
        }
    else:
        raise ValueError(f"Unknown content part type: {part.type}")


async def _build_chat_message_with_attachments(
    role: str,
    msg: Message
) -> Dict[str, Any]:
    """
    Build multi-part content for Chat Completions if attachments exist.
    Supports images via image_url and documents via content extraction.
    """
    has_attachments = msg.attachments and len(msg.attachments) > 0
    if not has_attachments:
        if role == 'assistant':
            base_msg = {"role": "assistant", "content": get_text_content(msg.content)}
            if msg.tool_calls:
                base_msg["tool_calls"] = [
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
            return base_msg
        return {"role": "user", "content": get_text_content(msg.content)}

    parts = []
    text_content = get_text_content(msg.content)
    if text_content and text_content.strip():
        parts.append({"type": "text", "text": text_content})

    for att in msg.attachments:
        if att.kind == 'image':
            # Prefer explicit URL; otherwise construct a data URL from base64
            url = att.url
            if not url and att.data and att.mime_type:
                # Validate base64 data size before creating data URL
                try:
                    # Estimate decoded size (base64 is ~4/3 of decoded size)
                    estimated_size = len(att.data) * 3 // 4
                    
                    if estimated_size > MAX_IMAGE_BYTES:
                        print(f"Warning: Skipping oversized image ({estimated_size} bytes > {MAX_IMAGE_BYTES}). "
                              f"Set JAF_MAX_IMAGE_BYTES env var to adjust limit.")
                        parts.append({
                            "type": "text",
                            "text": f"[IMAGE SKIPPED: Size exceeds limit of {MAX_IMAGE_BYTES//1024//1024}MB. "
                                   f"Image name: {att.name or 'unnamed'}]"
                        })
                        continue
                    
                    # Create data URL for valid-sized images
                    url = f"data:{att.mime_type};base64,{att.data}"
                except Exception as e:
                    print(f"Error processing image data: {e}")
                    parts.append({
                        "type": "text",
                        "text": f"[IMAGE ERROR: Failed to process image data. Image name: {att.name or 'unnamed'}]"
                    })
                    continue
            
            if url:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
        
        elif att.kind in ['document', 'file']:
            # Check if attachment has use_litellm_format flag or is a large document
            use_litellm_format = att.use_litellm_format is True
            
            if use_litellm_format and (att.url or att.data):
                # For now, fall back to content extraction since most providers don't support native file format
                # TODO: Add provider-specific file format support
                print(f"Info: LiteLLM format requested for {att.name}, falling back to content extraction")
                use_litellm_format = False
            
            if not use_litellm_format:
                # Extract document content if supported and we have data or URL
                if is_document_supported(att.mime_type) and (att.data or att.url):
                    try:
                        processed = await extract_document_content(att)
                        file_name = att.name or 'document'
                        description = get_document_description(att.mime_type)
                        
                        parts.append({
                            "type": "text",
                            "text": f"DOCUMENT: {file_name} ({description}):\n\n{processed.content}"
                        })
                    except DocumentProcessingError as e:
                        # Fallback to filename if extraction fails
                        label = att.name or att.format or att.mime_type or 'attachment'
                        parts.append({
                            "type": "text",
                            "text": f"ERROR: Failed to process {att.kind}: {label} ({e})"
                        })
                else:
                    # Unsupported document type - show placeholder
                    label = att.name or att.format or att.mime_type or 'attachment'
                    url_info = f" ({att.url})" if att.url else ""
                    parts.append({
                        "type": "text",
                        "text": f"ATTACHMENT: {att.kind}: {label}{url_info}"
                    })

    base_msg = {"role": role, "content": parts}
    if role == 'assistant' and msg.tool_calls:
        base_msg["tool_calls"] = [
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
    
    return base_msg

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
