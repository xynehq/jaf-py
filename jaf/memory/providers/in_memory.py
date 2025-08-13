"""
In-memory memory provider implementation.

This provider stores conversations in Python dictionaries with LRU eviction.
Best for development, testing, or single-instance deployments where persistence
across restarts is not required.
"""

import asyncio
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...core.types import Message
from ..types import (
    ConversationMemory,
    Failure,
    InMemoryConfig,
    MemoryConnectionError,
    MemoryNotFoundError,
    MemoryProvider,
    MemoryQuery,
    MemoryStorageError,
    Result,
    Success,
)


class InMemoryProvider(MemoryProvider):
    """
    In-memory implementation of MemoryProvider.
    
    Uses OrderedDict for LRU behavior and enforces memory limits through
    automatic eviction of oldest conversations.
    """

    def __init__(self, config: InMemoryConfig):
        self.config = config
        self._conversations: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._lock = asyncio.Lock()

        print(f"[MEMORY:InMemory] Initialized with max {config.max_conversations} conversations, {config.max_messages_per_conversation} messages each")

    async def _enforce_memory_limits(self) -> None:
        """Enforce conversation and message limits through LRU eviction."""
        while len(self._conversations) > self.config.max_conversations:
            oldest_id, _ = self._conversations.popitem(last=False)
            print(f"[MEMORY:InMemory] Evicted oldest conversation {oldest_id}")

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, MemoryStorageError]:
        """Store messages for a conversation."""
        async with self._lock:
            try:
                now = datetime.now()

                limited_messages = messages[-self.config.max_messages_per_conversation:]

                conversation_metadata = {
                    "created_at": now,
                    "updated_at": now,
                    "total_messages": len(limited_messages),
                    "last_activity": now,
                    **(metadata or {})
                }

                conversation = ConversationMemory(
                    conversation_id=conversation_id,
                    user_id=metadata.get("user_id") if metadata else None,
                    messages=limited_messages,
                    metadata=conversation_metadata
                )

                self._conversations[conversation_id] = conversation
                self._conversations.move_to_end(conversation_id)
                await self._enforce_memory_limits()

                print(f"[MEMORY:InMemory] Stored {len(limited_messages)} messages for conversation {conversation_id}")
                return Success(None)

            except Exception as e:
                return Failure(MemoryStorageError(
                    operation="store_messages",
                    provider="InMemory",
                    message=f"Failed to store messages: {e}",
                    cause=e
                ))

    async def get_conversation(
        self,
        conversation_id: str
    ) -> Result[Optional[ConversationMemory], MemoryStorageError]:
        """Retrieve conversation history."""
        async with self._lock:
            try:
                conversation = self._conversations.get(conversation_id)
                if conversation is None:
                    return Success(None)

                self._conversations.move_to_end(conversation_id)
                conversation.metadata["last_activity"] = datetime.now()

                print(f"[MEMORY:InMemory] Retrieved conversation {conversation_id}")
                return Success(conversation)

            except Exception as e:
                return Failure(MemoryStorageError(
                    message=f"Error retrieving conversation: {e}",
                    provider="InMemory",
                    operation="get_conversation",
                    cause=e
                ))

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        """Append new messages to existing conversation."""
        async with self._lock:
            try:
                existing = self._conversations.get(conversation_id)
                if existing is None:
                    return Failure(MemoryNotFoundError(
                        message=f"Conversation {conversation_id} not found",
                        provider="InMemory",
                        conversation_id=conversation_id
                    ))

                combined_messages = list(existing.messages) + messages
                limited_messages = combined_messages[-self.config.max_messages_per_conversation:]

                now = datetime.now()
                existing.metadata.update({
                    "updated_at": now,
                    "last_activity": now,
                    "total_messages": len(limited_messages),
                    **(metadata or {})
                })

                updated_conversation = ConversationMemory(
                    conversation_id=conversation_id,
                    user_id=existing.user_id,
                    messages=limited_messages,
                    metadata=existing.metadata
                )

                self._conversations[conversation_id] = updated_conversation
                self._conversations.move_to_end(conversation_id)

                print(f"[MEMORY:InMemory] Appended {len(messages)} messages to conversation {conversation_id}")
                return Success(None)

            except Exception as e:
                return Failure(MemoryStorageError(
                    message=f"Failed to append messages: {e}",
                    provider="InMemory",
                    operation="append_messages",
                    cause=e
                ))

    async def find_conversations(
        self,
        query: MemoryQuery
    ) -> Result[List[ConversationMemory], MemoryStorageError]:
        """Search conversations by query parameters."""
        async with self._lock:
            try:
                results = []
                for conv in self._conversations.values():
                    if query.conversation_id and conv.conversation_id != query.conversation_id:
                        continue
                    if query.user_id and conv.user_id != query.user_id:
                        continue
                    if query.trace_id and conv.metadata.get("trace_id") != query.trace_id:
                        continue
                    if query.since and conv.metadata.get("created_at") < query.since:
                        continue
                    if query.until and conv.metadata.get("created_at") > query.until:
                        continue
                    results.append(conv)

                results.sort(key=lambda c: c.metadata.get("last_activity", datetime.min), reverse=True)

                offset = query.offset or 0
                limit = query.limit or len(results)
                paginated_results = results[offset:offset + limit]

                return Success(paginated_results)

            except Exception as e:
                return Failure(MemoryStorageError(
                    message=f"Error finding conversations: {e}",
                    provider="InMemory",
                    operation="find_conversations",
                    cause=e
                ))

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> Result[List[Message], Union[MemoryNotFoundError, MemoryStorageError]]:
        """Get recent messages from a conversation."""
        result = await self.get_conversation(conversation_id)
        if isinstance(result, Failure):
            return result

        conversation = result.data
        if not conversation:
            return Failure(MemoryNotFoundError(
                message=f"Conversation {conversation_id} not found",
                provider="InMemory",
                conversation_id=conversation_id
            ))

        return Success(list(conversation.messages[-limit:]))

    async def delete_conversation(
        self,
        conversation_id: str
    ) -> Result[bool, MemoryStorageError]:
        """Delete conversation and return True if it existed."""
        async with self._lock:
            try:
                existed = conversation_id in self._conversations
                if existed:
                    del self._conversations[conversation_id]
                return Success(existed)
            except Exception as e:
                return Failure(MemoryStorageError(
                    message=f"Error deleting conversation: {e}",
                    provider="InMemory",
                    operation="delete_conversation",
                    cause=e
                ))

    async def clear_user_conversations(
        self,
        user_id: str
    ) -> Result[int, MemoryStorageError]:
        """Clear all conversations for a user and return count deleted."""
        async with self._lock:
            try:
                to_delete = [
                    conv_id for conv_id, conv in self._conversations.items()
                    if conv.user_id == user_id
                ]
                for conv_id in to_delete:
                    del self._conversations[conv_id]
                return Success(len(to_delete))
            except Exception as e:
                return Failure(MemoryStorageError(
                    message=f"Error clearing conversations: {e}",
                    provider="InMemory",
                    operation="clear_user_conversations",
                    cause=e
                ))

    async def get_stats(
        self,
        user_id: Optional[str] = None
    ) -> Result[Dict[str, Any], MemoryStorageError]:
        """Get conversation statistics."""
        async with self._lock:
            try:
                convs = [c for c in self._conversations.values() if not user_id or c.user_id == user_id]
                total_messages = sum(len(c.messages) for c in convs)
                created_dates = [c.metadata["created_at"] for c in convs if "created_at" in c.metadata]

                return Success({
                    "total_conversations": len(convs),
                    "total_messages": total_messages,
                    "oldest_conversation": min(created_dates) if created_dates else None,
                    "newest_conversation": max(created_dates) if created_dates else None
                })
            except Exception as e:
                return Failure(MemoryStorageError(
                    message=f"Error getting stats: {e}",
                    provider="InMemory",
                    operation="get_stats",
                    cause=e
                ))

    async def health_check(self) -> Result[Dict[str, Any], MemoryConnectionError]:
        """Check provider health and return status information."""
        start_time = datetime.now()
        try:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return Success({
                "healthy": True,
                "latency_ms": latency_ms,
                "provider": "InMemory",
                "details": {
                    "conversations_count": len(self._conversations)
                }
            })
        except Exception as e:
            return Failure(MemoryConnectionError(
                message=f"Health check failed: {e}",
                provider="InMemory",
                cause=e
            ))

    async def close(self) -> Result[None, MemoryConnectionError]:
        """Close/cleanup the provider."""
        async with self._lock:
            self._conversations.clear()
            print("[MEMORY:InMemory] Closed provider, cleared all conversations")
            return Success(None)

def create_in_memory_provider(config: Optional[InMemoryConfig] = None) -> InMemoryProvider:
    """
    Factory function to create an in-memory provider instance.
    
    Args:
        config: Configuration for the provider. If None, uses defaults.
        
    Returns:
        Configured InMemoryProvider instance.
    """
    return InMemoryProvider(config or InMemoryConfig())
