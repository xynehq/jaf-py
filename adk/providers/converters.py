"""
ADK Type Converters - Bridge Between ADK and Core/OpenAI Formats

This module provides type conversion utilities to bridge between ADK types
and various LLM provider formats (JAF Core, OpenAI, etc.).
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..types import AdkMessage, AdkAgent, AdkTool, AdkContext, AdkModelType
from jaf.core.types import Message as CoreMessage, Agent as CoreAgent, Tool as CoreTool, ModelConfig

class AdkTypeConverter:
    """
    Centralized type conversion utilities for ADK integration.
    
    Provides functional conversion methods between ADK types and various
    LLM provider formats while maintaining type safety.
    """
    
    @staticmethod
    def adk_to_core_message(adk_message: AdkMessage) -> CoreMessage:
        """Convert ADK message to JAF Core message."""
        return CoreMessage(
            role=adk_message.role,
            content=adk_message.content,
            tool_calls=adk_message.tool_calls,
            tool_call_id=adk_message.tool_call_id
        )
    
    @staticmethod
    def core_to_adk_message(core_message: CoreMessage) -> AdkMessage:
        """Convert JAF Core message to ADK message."""
        return AdkMessage(
            role=core_message.role,
            content=core_message.content,
            tool_calls=core_message.tool_calls,
            tool_call_id=core_message.tool_call_id,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def adk_to_openai_message(adk_message: AdkMessage) -> Dict[str, Any]:
        """Convert ADK message to OpenAI message format."""
        openai_message = {
            "role": adk_message.role,
            "content": adk_message.content or ""
        }
        
        # Add tool calls if present
        if adk_message.tool_calls:
            openai_message["tool_calls"] = adk_message.tool_calls
        
        # Add tool call ID for tool messages
        if adk_message.tool_call_id:
            openai_message["tool_call_id"] = adk_message.tool_call_id
        
        return openai_message
    
    @staticmethod
    def openai_to_adk_message(openai_message: Dict[str, Any]) -> AdkMessage:
        """Convert OpenAI message format to ADK message."""
        return AdkMessage(
            role=openai_message.get("role", "assistant"),
            content=openai_message.get("content", ""),
            tool_calls=openai_message.get("tool_calls"),
            tool_call_id=openai_message.get("tool_call_id"),
            timestamp=datetime.now()
        )
    
    @staticmethod
    def adk_agent_to_core_agent(adk_agent: AdkAgent, context: AdkContext) -> CoreAgent:
        """Convert ADK agent to JAF Core agent."""
        # Handle instructions
        if callable(adk_agent.instructions):
            instruction_text = adk_agent.instructions(context)
        else:
            instruction_text = adk_agent.instructions
        
        # Create model config
        model_config = ModelConfig(
            name=adk_agent.model.value,
            temperature=adk_agent.temperature,
            max_tokens=adk_agent.max_tokens
        )
        
        # Convert tools (simplified - would need proper tool conversion)
        core_tools = []  # Will be populated by tool conversion logic
        
        return CoreAgent(
            name=adk_agent.name,
            instructions=lambda state: instruction_text,
            tools=core_tools,
            model_config=model_config
        )
    
    @staticmethod
    def adk_tools_to_openai_tools(adk_tools: List[AdkTool]) -> List[Dict[str, Any]]:
        """Convert ADK tools to OpenAI tools format."""
        openai_tools = []
        
        for tool in adk_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    @staticmethod
    def messages_list_to_openai(messages: List[AdkMessage]) -> List[Dict[str, Any]]:
        """Convert list of ADK messages to OpenAI format."""
        return [
            AdkTypeConverter.adk_to_openai_message(msg) 
            for msg in messages
        ]
    
    @staticmethod
    def openai_messages_to_adk(openai_messages: List[Dict[str, Any]]) -> List[AdkMessage]:
        """Convert list of OpenAI messages to ADK format."""
        return [
            AdkTypeConverter.openai_to_adk_message(msg) 
            for msg in openai_messages
        ]
    
    @staticmethod
    def create_system_message_openai(instructions: str) -> Dict[str, Any]:
        """Create OpenAI system message from instructions."""
        return {
            "role": "system",
            "content": instructions
        }
    
    @staticmethod
    def extract_tool_calls_from_openai_response(response: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from OpenAI response."""
        message = response.get("message", {})
        return message.get("tool_calls")
    
    @staticmethod
    def model_type_to_string(model: Union[AdkModelType, str]) -> str:
        """Convert model type to string."""
        if isinstance(model, AdkModelType):
            return model.value
        return str(model)

# Convenience functions for direct access
def convert_adk_to_core_message(adk_message: AdkMessage) -> CoreMessage:
    """Convert ADK message to JAF Core message."""
    return AdkTypeConverter.adk_to_core_message(adk_message)

def convert_core_to_adk_message(core_message: CoreMessage) -> AdkMessage:
    """Convert JAF Core message to ADK message."""
    return AdkTypeConverter.core_to_adk_message(core_message)

def convert_adk_to_openai_message(adk_message: AdkMessage) -> Dict[str, Any]:
    """Convert ADK message to OpenAI message format."""
    return AdkTypeConverter.adk_to_openai_message(adk_message)

def convert_openai_to_adk_message(openai_message: Dict[str, Any]) -> AdkMessage:
    """Convert OpenAI message format to ADK message."""
    return AdkTypeConverter.openai_to_adk_message(openai_message)

def convert_adk_agent_to_core_agent(adk_agent: AdkAgent, context: AdkContext) -> CoreAgent:
    """Convert ADK agent to JAF Core agent."""
    return AdkTypeConverter.adk_agent_to_core_agent(adk_agent, context)

def convert_adk_tools_to_openai_tools(adk_tools: List[AdkTool]) -> List[Dict[str, Any]]:
    """Convert ADK tools to OpenAI tools format."""
    return AdkTypeConverter.adk_tools_to_openai_tools(adk_tools)