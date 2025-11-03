"""
Checkpoint functionality for the JAF framework.

This module implements conversation checkpointing where a conversation can be
saved at a specific point, removing all subsequent messages without regeneration.
"""

import time
from dataclasses import dataclass
from typing import Any, TypeVar, Optional, List

from .types import (
    RunState,
    RunConfig,
    CheckpointRequest,
    CheckpointContext,
    MessageId,
    Message,
    find_message_index,
    get_message_by_id,
    generate_run_id,
    generate_trace_id,
)
from ..memory.types import Success, Failure

Ctx = TypeVar("Ctx")
Out = TypeVar("Out")


@dataclass(frozen=True)
class CheckpointResult:
    """Result of a checkpoint operation."""

    checkpoint_id: str
    conversation_id: str
    original_message_count: int
    checkpointed_at_index: int
    checkpointed_message_id: MessageId
    messages: List[Message]
    execution_time_ms: int


async def checkpoint_conversation(
    checkpoint_request: CheckpointRequest, config: RunConfig[Ctx]
) -> CheckpointResult:
    """
    Checkpoint a conversation after a specific message ID.

    This function:
    1. Loads the full conversation from memory
    2. Finds the message to checkpoint after
    3. Truncates the conversation AFTER that point (keeps the checkpoint message)
    4. Stores the checkpointed conversation

    Args:
        checkpoint_request: The checkpoint request containing conversation_id and message_id
        config: The run configuration

    Returns:
        CheckpointResult with the checkpointed conversation data

    Raises:
        ValueError: If memory provider is not configured or conversation not found
    """
    start_time = time.time()

    if not config.memory or not config.memory.provider or not config.conversation_id:
        raise ValueError("Checkpoint requires memory provider and conversation_id to be configured")

    # Load the conversation from memory
    conversation_result = await config.memory.provider.get_conversation(
        checkpoint_request.conversation_id
    )
    if isinstance(conversation_result, Failure):
        raise ValueError(f"Failed to load conversation: {conversation_result.error}")

    conversation_memory = conversation_result.data
    if not conversation_memory:
        raise ValueError(f"Conversation {checkpoint_request.conversation_id} not found")

    # Convert tuple back to list for processing
    original_messages = list(conversation_memory.messages)

    # Find the message to checkpoint after
    checkpoint_message = get_message_by_id(original_messages, checkpoint_request.message_id)
    if not checkpoint_message:
        raise ValueError(f"Message {checkpoint_request.message_id} not found in conversation")

    # Get the index of the checkpoint message
    checkpoint_index = find_message_index(original_messages, checkpoint_request.message_id)
    if checkpoint_index is None:
        raise ValueError(f"Failed to find index for message {checkpoint_request.message_id}")

    # Truncate messages AFTER the checkpoint message (keep the checkpoint message itself)
    # This is the KEY difference from regeneration which truncates FROM the message
    checkpointed_messages = original_messages[
        : checkpoint_index + 1
    ]  # +1 to include the checkpoint message

    print(f"[JAF:CHECKPOINT] Checkpointed conversation to {len(checkpointed_messages)} messages")
    print(
        f"[JAF:CHECKPOINT] Original: {len(original_messages)}, Removed: {len(original_messages) - len(checkpointed_messages)}"
    )

    # Create checkpoint context
    checkpoint_context = CheckpointContext(
        original_message_count=len(original_messages),
        checkpointed_at_index=checkpoint_index,
        checkpointed_message_id=checkpoint_request.message_id,
        checkpoint_id=f"chk_{int(time.time() * 1000)}_{checkpoint_request.message_id}",
        timestamp=int(time.time() * 1000),
    )

    # Prepare metadata (simpler than regeneration - no audit trail needed)
    def serialize_metadata(metadata):
        import json
        import datetime

        def json_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, datetime.date):
                return obj.isoformat()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        try:
            json_str = json.dumps(metadata, default=json_serializer)
            return json.loads(json_str)
        except Exception as e:
            print(f"[JAF:CHECKPOINT] Warning: Metadata serialization failed: {e}")
            return {
                "checkpoint_truncated": True,
                "checkpoint_point": str(checkpoint_request.message_id),
                "original_message_count": len(original_messages),
                "checkpointed_at_index": checkpoint_index,
                "checkpointed_messages": len(checkpointed_messages),
            }

    metadata = serialize_metadata(
        {
            **conversation_memory.metadata,
            "checkpoint_truncated": True,
            "checkpoint_point": str(checkpoint_request.message_id),
            "original_message_count": len(original_messages),
            "checkpointed_at_index": checkpoint_index,
            "checkpointed_messages": len(checkpointed_messages),
            "checkpoint_id": checkpoint_context.checkpoint_id,
            "checkpoint_timestamp": checkpoint_context.timestamp,
        }
    )

    # Store the checkpointed conversation
    store_result = await config.memory.provider.store_messages(
        checkpoint_request.conversation_id, checkpointed_messages, metadata
    )

    if isinstance(store_result, Failure):
        raise ValueError(f"Failed to store checkpointed conversation: {store_result.error}")

    print(
        f"[JAF:CHECKPOINT] Successfully checkpointed conversation {checkpoint_request.conversation_id}"
    )

    execution_time_ms = int((time.time() - start_time) * 1000)

    return CheckpointResult(
        checkpoint_id=checkpoint_context.checkpoint_id,
        conversation_id=checkpoint_request.conversation_id,
        original_message_count=len(original_messages),
        checkpointed_at_index=checkpoint_index,
        checkpointed_message_id=checkpoint_request.message_id,
        messages=checkpointed_messages,
        execution_time_ms=execution_time_ms,
    )


async def get_checkpoint_history(
    conversation_id: str, config: RunConfig[Ctx]
) -> Optional[List[dict]]:
    """
    Get checkpoint history for a conversation.

    Note: Unlike regeneration, we don't maintain a checkpoint_points list in metadata.
    This function returns the last checkpoint information if available.

    Args:
        conversation_id: The conversation ID
        config: The run configuration

    Returns:
        List of checkpoint metadata or empty list if none found
    """
    if not config.memory or not config.memory.provider:
        return None

    try:
        conversation_result = await config.memory.provider.get_conversation(conversation_id)
        if hasattr(conversation_result, "data") and conversation_result.data:
            metadata = conversation_result.data.metadata

            # Check if there's checkpoint metadata
            if metadata.get("checkpoint_truncated"):
                checkpoint_data = {
                    "checkpoint_id": metadata.get("checkpoint_id", ""),
                    "checkpoint_point": metadata.get("checkpoint_point", ""),
                    "timestamp": metadata.get("checkpoint_timestamp", 0),
                    "original_message_count": metadata.get("original_message_count", 0),
                    "checkpointed_at_index": metadata.get("checkpointed_at_index", 0),
                    "checkpointed_messages": metadata.get("checkpointed_messages", 0),
                }
                print(f"[JAF:CHECKPOINT] Retrieved checkpoint data for {conversation_id}")
                return [checkpoint_data]
            else:
                print(f"[JAF:CHECKPOINT] No checkpoint data found for {conversation_id}")
                return []
        else:
            print(f"[JAF:CHECKPOINT] No conversation data found for {conversation_id}")
            return []
    except Exception as e:
        print(f"[JAF:CHECKPOINT] Failed to get checkpoint history: {e}")
        return []
