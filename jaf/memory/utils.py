"""
Shared utilities for memory providers to eliminate code duplication.

This module provides common functions for serialization, validation, and other
operations used across different memory provider implementations.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.types import Message, ToolCall, ToolCallFunction
from .types import ConversationMemory


def serialize_message(msg: Message) -> dict:
    """
    Convert Message dataclass to dict for storage.
    
    This provides a consistent serialization format across all memory providers.
    """
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_call_id": msg.tool_call_id,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in msg.tool_calls
        ] if msg.tool_calls else None
    }


def deserialize_message(msg_data: dict) -> Message:
    """
    Convert dict back to Message dataclass from storage.
    
    This provides a consistent deserialization format across all memory providers.
    """
    tool_calls = None
    if msg_data.get("tool_calls"):
        tool_calls = [
            ToolCall(
                id=tc["id"],
                type=tc["type"],
                function=ToolCallFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"]
                )
            ) for tc in msg_data["tool_calls"]
        ]

    return Message(
        role=msg_data["role"],
        content=msg_data["content"],
        tool_call_id=msg_data.get("tool_call_id"),
        tool_calls=tool_calls
    )


def serialize_conversation_for_json(conversation: ConversationMemory) -> str:
    """
    Serialize conversation to JSON string for storage (Redis, file systems, etc.).
    
    Args:
        conversation: The conversation memory to serialize
        
    Returns:
        JSON string representation
    """
    data = {
        "conversation_id": conversation.conversation_id,
        "user_id": conversation.user_id,
        "messages": [serialize_message(msg) for msg in conversation.messages],
        "metadata": {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in conversation.metadata.items()
        }
    }
    return json.dumps(data)


def deserialize_conversation_from_json(data: str) -> ConversationMemory:
    """
    Deserialize conversation from JSON string.
    
    Args:
        data: JSON string representation
        
    Returns:
        ConversationMemory instance
    """
    parsed = json.loads(data)

    # Parse metadata dates back to datetime objects
    metadata = {}
    for k, v in parsed.get("metadata", {}).items():
        if isinstance(v, str) and ("_at" in k or "activity" in k):
            try:
                metadata[k] = datetime.fromisoformat(v)
            except ValueError:
                metadata[k] = v
        else:
            metadata[k] = v

    return ConversationMemory(
        conversation_id=parsed["conversation_id"],
        user_id=parsed.get("user_id"),
        messages=[deserialize_message(msg) for msg in parsed.get("messages", [])],
        metadata=metadata
    )


def prepare_message_list_for_db(messages: List[Message]) -> str:
    """
    Prepare a list of messages for database storage.
    
    Args:
        messages: List of Message objects
        
    Returns:
        JSON string suitable for database storage
    """
    return json.dumps([serialize_message(msg) for msg in messages])


def extract_messages_from_db_row(messages_json: str) -> List[Message]:
    """
    Extract messages from database row JSON.
    
    Args:
        messages_json: JSON string from database
        
    Returns:
        List of Message objects
    """
    messages_data = json.loads(messages_json)
    return [deserialize_message(msg_data) for msg_data in messages_data]


def sanitize_conversation_id(conversation_id: str) -> str:
    """
    Sanitize conversation ID to ensure it's safe for storage.
    
    Args:
        conversation_id: Raw conversation ID
        
    Returns:
        Sanitized conversation ID
    """
    # Remove potentially problematic characters and limit length
    sanitized = "".join(c for c in conversation_id if c.isalnum() or c in "-_.")
    return sanitized[:100]  # Limit length for database compatibility


def create_default_metadata(user_id: Optional[str] = None, message_count: int = 0) -> Dict[str, Any]:
    """
    Create default metadata for a new conversation.
    
    Args:
        user_id: Optional user ID
        message_count: Initial message count
        
    Returns:
        Default metadata dictionary
    """
    now = datetime.now()
    return {
        "created_at": now,
        "updated_at": now,
        "last_activity": now,
        "total_messages": message_count,
        "user_id": user_id
    }


def update_conversation_metadata(
    existing_metadata: Dict[str, Any],
    new_message_count: int,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update conversation metadata with new activity.
    
    Args:
        existing_metadata: Current metadata
        new_message_count: Updated message count
        additional_metadata: Additional metadata to merge
        
    Returns:
        Updated metadata dictionary
    """
    now = datetime.now()
    updated = existing_metadata.copy()

    # Update standard fields
    updated["updated_at"] = now
    updated["last_activity"] = now
    updated["total_messages"] = new_message_count

    # Merge additional metadata if provided
    if additional_metadata:
        updated.update(additional_metadata)

    return updated


def validate_conversation_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean conversation metadata.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Validated and cleaned metadata
    """
    cleaned = {}

    for key, value in metadata.items():
        # Ensure keys are strings
        str_key = str(key)

        # Skip None values
        if value is None:
            continue

        # Handle datetime objects
        if isinstance(value, datetime):
            cleaned[str_key] = value
        # Handle datetime strings
        elif isinstance(value, str) and ("_at" in str_key or "activity" in str_key):
            try:
                cleaned[str_key] = datetime.fromisoformat(value)
            except ValueError:
                cleaned[str_key] = value
        # Handle other types
        else:
            cleaned[str_key] = value

    return cleaned
