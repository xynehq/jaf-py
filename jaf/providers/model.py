"""
Model provider implementations for the JAF framework.

This module provides model providers that integrate with various LLM services,
starting with LiteLLM for multi-provider support.
"""

from typing import Any, Dict, Optional, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from ..core.types import Agent, ContentRole, Message, ModelProvider, RunConfig, RunState

Ctx = TypeVar('Ctx')

def make_litellm_provider(
    base_url: str,
    api_key: str = "anything",
    default_timeout: Optional[float] = None
) -> ModelProvider[Ctx]:
    """
    Create a LiteLLM-compatible model provider.
    
    Args:
        base_url: Base URL for the LiteLLM server
        api_key: API key (defaults to "anything" for local servers)
        default_timeout: Default timeout for model API calls in seconds
        
    Returns:
        ModelProvider instance
    """

    class LiteLLMProvider:
        def __init__(self):
            # Default to "anything" if api_key is not provided, for local servers
            effective_api_key = api_key if api_key is not None else "anything"
            self.client = OpenAI(
                base_url=base_url,
                api_key=effective_api_key,
                # Note: dangerouslyAllowBrowser is JavaScript-specific
            )
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

            return {
                'message': {
                    'content': choice.message.content,
                    'tool_calls': tool_calls
                }
            }

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
