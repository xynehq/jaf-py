"""
Redis memory provider implementation.

This provider uses Redis for persistent conversation storage with JSON serialization.
Best for production environments with shared state and persistence across restarts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...core.types import Message, MessageId, find_message_index
from ..types import (
    ConversationMemory,
    Failure,
    MemoryConnectionError,
    MemoryNotFoundError,
    MemoryProvider,
    MemoryQuery,
    MemoryStorageError,
    RedisConfig,
    Result,
    Success,
)

try:
    import redis.asyncio as redis

    RedisClient = redis.Redis
except ImportError:
    RedisClient = Any


class RedisProvider(MemoryProvider):
    """
    Redis implementation of MemoryProvider.
    """

    def __init__(self, config: RedisConfig, redis_client: RedisClient):
        self.config = config
        self.redis_client = redis_client

        print(f"[MEMORY:Redis] Initialized with prefix '{config.key_prefix}'")

    def _get_key(self, conversation_id: str) -> str:
        return f"{self.config.key_prefix}{conversation_id}"

    def _serialize(self, conversation: ConversationMemory) -> str:
        """Serialize conversation using shared utilities."""
        from ..utils import serialize_conversation_for_json

        return serialize_conversation_for_json(conversation)

    def _deserialize(self, data: str) -> ConversationMemory:
        """Deserialize conversation using shared utilities."""
        from ..utils import deserialize_conversation_from_json

        return deserialize_conversation_from_json(data)

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result[None, MemoryStorageError]:
        try:
            now = datetime.now()
            conversation = ConversationMemory(
                conversation_id=conversation_id,
                user_id=metadata.get("user_id") if metadata else None,
                messages=messages,
                metadata={
                    "created_at": now,
                    "updated_at": now,
                    "total_messages": len(messages),
                    "last_activity": now,
                    **(metadata or {}),
                },
            )
            key = self._get_key(conversation_id)
            await self.redis_client.set(key, self._serialize(conversation), ex=self.config.ttl)
            return Success(None)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="store_messages", provider="Redis", message=str(e), cause=e
                )
            )

    async def get_conversation(
        self, conversation_id: str
    ) -> Result[Optional[ConversationMemory], MemoryStorageError]:
        try:
            key = self._get_key(conversation_id)
            value = await self.redis_client.get(key)
            if not value:
                return Success(None)

            conversation = self._deserialize(value)
            conversation.metadata["last_activity"] = datetime.now()

            await self.redis_client.set(key, self._serialize(conversation), ex=self.config.ttl)
            return Success(conversation)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="get_conversation", provider="Redis", message=str(e), cause=e
                )
            )

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        result = await self.get_conversation(conversation_id)
        if isinstance(result, Failure):
            return result

        existing = result.data
        if not existing:
            return Failure(
                MemoryNotFoundError(
                    conversation_id=conversation_id,
                    provider="Redis",
                    message=f"Conversation {conversation_id} not found",
                )
            )

        # Convert tuple back to list, append new messages, then store
        all_messages = list(existing.messages) + messages
        updated_metadata = {
            **existing.metadata,
            "updated_at": datetime.now(),
            "last_activity": datetime.now(),
            "total_messages": len(all_messages),
            **(metadata or {}),
        }

        return await self.store_messages(conversation_id, all_messages, updated_metadata)

    async def find_conversations(
        self, query: MemoryQuery
    ) -> Result[List[ConversationMemory], MemoryStorageError]:
        try:
            keys = await self.redis_client.keys(f"{self.config.key_prefix}*")
            conversations = []
            for key in keys:
                # Ensure we are only processing conversation keys
                if not key.decode().startswith(f"{self.config.key_prefix}conversation:"):
                    continue
                value = await self.redis_client.get(key)
                if value:
                    conv = self._deserialize(value)
                    # Filtering logic here
                    if query.user_id and conv.user_id != query.user_id:
                        continue
                    if query.conversation_id and conv.conversation_id != query.conversation_id:
                        continue
                    conversations.append(conv)
            return Success(conversations)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="find_conversations", provider="Redis", message=str(e), cause=e
                )
            )

    async def get_recent_messages(
        self, conversation_id: str, limit: int = 50
    ) -> Result[List[Message], Union[MemoryNotFoundError, MemoryStorageError]]:
        result = await self.get_conversation(conversation_id)
        if isinstance(result, Failure):
            return result

        conversation = result.data
        if not conversation:
            return Failure(
                MemoryNotFoundError(
                    conversation_id=conversation_id,
                    provider="Redis",
                    message=f"Conversation {conversation_id} not found",
                )
            )

        return Success(conversation.messages[-limit:])

    async def delete_conversation(self, conversation_id: str) -> Result[bool, MemoryStorageError]:
        try:
            deleted = await self.redis_client.delete(self._get_key(conversation_id))
            return Success(deleted > 0)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="delete_conversation", provider="Redis", message=str(e), cause=e
                )
            )

    async def clear_user_conversations(self, user_id: str) -> Result[int, MemoryStorageError]:
        # This is inefficient in Redis, consider a different approach for production
        return Failure(
            MemoryStorageError(
                operation="clear_user_conversations",
                provider="Redis",
                message="clear_user_conversations not efficiently supported",
            )
        )

    async def get_stats(
        self, user_id: Optional[str] = None
    ) -> Result[Dict[str, Any], MemoryStorageError]:
        try:
            keys = await self.redis_client.keys(f"{self.config.key_prefix}*")
            return Success({"total_conversations": len(keys)})
        except Exception as e:
            return Failure(
                MemoryStorageError(operation="get_stats", provider="Redis", message=str(e), cause=e)
            )

    async def health_check(self) -> Result[Dict[str, Any], MemoryConnectionError]:
        start_time = datetime.now()
        try:
            await self.redis_client.ping()
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            db_size = await self.redis_client.dbsize()
            return Success(
                {
                    "healthy": True,
                    "latency_ms": latency_ms,
                    "provider": "Redis",
                    "details": {"db_size": db_size},
                }
            )
        except Exception as e:
            return Failure(
                MemoryConnectionError(
                    provider="Redis", message="Redis health check failed", cause=e
                )
            )

    async def truncate_conversation_after(
        self, conversation_id: str, message_id: MessageId
    ) -> Result[int, Union[MemoryNotFoundError, MemoryStorageError]]:
        """
        Truncate conversation after (and including) the specified message ID.
        Returns the number of messages removed.
        """
        try:
            # Get the conversation
            conv_result = await self.get_conversation(conversation_id)
            if isinstance(conv_result, Failure):
                return conv_result

            if not conv_result.data:
                return Failure(
                    MemoryNotFoundError(
                        message=f"Conversation {conversation_id} not found",
                        provider="Redis",
                        conversation_id=conversation_id,
                    )
                )

            conversation = conv_result.data
            messages = list(conversation.messages)
            truncate_index = find_message_index(messages, message_id)

            if truncate_index is None:
                # Message not found, nothing to truncate
                return Success(0)

            # Truncate messages from the found index onwards
            original_count = len(messages)
            truncated_messages = messages[:truncate_index]
            removed_count = original_count - len(truncated_messages)

            # Update conversation with truncated messages
            now = datetime.now()
            updated_metadata = {
                **conversation.metadata,
                "updated_at": now,
                "last_activity": now,
                "total_messages": len(truncated_messages),
                "regeneration_truncated": True,
                "truncated_at": now.isoformat(),
                "messages_removed": removed_count,
            }

            # Store updated conversation
            updated_conversation = ConversationMemory(
                conversation_id=conversation_id,
                user_id=conversation.user_id,
                messages=truncated_messages,
                metadata=updated_metadata,
            )

            key = self._get_key(conversation_id)
            await self.redis_client.set(
                key, self._serialize(updated_conversation), ex=self.config.ttl
            )

            print(
                f"[MEMORY:Redis] Truncated conversation {conversation_id}: removed {removed_count} messages after message {message_id}"
            )
            return Success(removed_count)

        except Exception as e:
            return Failure(
                MemoryStorageError(
                    message=f"Failed to truncate conversation: {e}",
                    provider="Redis",
                    operation="truncate_conversation_after",
                    cause=e,
                )
            )

    async def get_conversation_until_message(
        self, conversation_id: str, message_id: MessageId
    ) -> Result[Optional[ConversationMemory], Union[MemoryNotFoundError, MemoryStorageError]]:
        """
        Get conversation history up to (but not including) the specified message ID.
        Useful for regeneration scenarios.
        """
        try:
            # Get the conversation
            conv_result = await self.get_conversation(conversation_id)
            if isinstance(conv_result, Failure):
                return conv_result

            if not conv_result.data:
                return Success(None)

            conversation = conv_result.data
            messages = list(conversation.messages)
            until_index = find_message_index(messages, message_id)

            if until_index is None:
                # Message not found, return None as lightweight indicator
                print(
                    f"[MEMORY:Redis] Message {message_id} not found in conversation {conversation_id}"
                )
                return Success(None)

            # Return conversation up to (but not including) the specified message
            truncated_messages = messages[:until_index]

            # Create a copy of the conversation with truncated messages
            truncated_conversation = ConversationMemory(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                messages=truncated_messages,
                metadata={
                    **conversation.metadata,
                    "truncated_for_regeneration": True,
                    "truncated_until_message": str(message_id),
                    "original_message_count": len(messages),
                    "truncated_message_count": len(truncated_messages),
                },
            )

            print(
                f"[MEMORY:Redis] Retrieved conversation {conversation_id} until message {message_id}: {len(truncated_messages)} messages"
            )
            return Success(truncated_conversation)

        except Exception as e:
            return Failure(
                MemoryStorageError(
                    message=f"Failed to get conversation until message: {e}",
                    provider="Redis",
                    operation="get_conversation_until_message",
                    cause=e,
                )
            )

    async def mark_regeneration_point(
        self, conversation_id: str, message_id: MessageId, regeneration_metadata: Dict[str, Any]
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        """
        Mark a regeneration point in the conversation for audit purposes.
        """
        try:
            # Get the conversation
            conv_result = await self.get_conversation(conversation_id)
            if isinstance(conv_result, Failure):
                return conv_result

            if not conv_result.data:
                return Failure(
                    MemoryNotFoundError(
                        message=f"Conversation {conversation_id} not found",
                        provider="Redis",
                        conversation_id=conversation_id,
                    )
                )

            conversation = conv_result.data

            # Add regeneration point to metadata
            regeneration_points = conversation.metadata.get("regeneration_points", [])
            regeneration_point = {
                "message_id": str(message_id),
                "timestamp": datetime.now().isoformat(),
                **regeneration_metadata,
            }
            regeneration_points.append(regeneration_point)

            # Update conversation metadata
            updated_metadata = {
                **conversation.metadata,
                "regeneration_points": regeneration_points,
                "last_regeneration": regeneration_point,
                "updated_at": datetime.now(),
                "regeneration_count": len(regeneration_points),
            }

            # Store updated conversation
            updated_conversation = ConversationMemory(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                messages=conversation.messages,
                metadata=updated_metadata,
            )

            key = self._get_key(conversation_id)
            await self.redis_client.set(
                key, self._serialize(updated_conversation), ex=self.config.ttl
            )

            print(
                f"[MEMORY:Redis] Marked regeneration point for conversation {conversation_id} at message {message_id}"
            )
            return Success(None)

        except Exception as e:
            return Failure(
                MemoryStorageError(
                    message=f"Failed to mark regeneration point: {e}",
                    provider="Redis",
                    operation="mark_regeneration_point",
                    cause=e,
                )
            )

    async def close(self) -> Result[None, MemoryConnectionError]:
        try:
            await self.redis_client.aclose()
            return Success(None)
        except Exception as e:
            return Failure(
                MemoryConnectionError(
                    provider="Redis", message="Failed to close Redis connection", cause=e
                )
            )


async def create_redis_provider(
    config: RedisConfig,
) -> Result[RedisProvider, MemoryConnectionError]:
    try:
        # These will be passed to the Redis client constructor
        # and will override any values parsed from the URL.
        conn_kwargs = {
            "host": config.host,
            "port": config.port,
            "db": config.db,
            "password": config.password,
        }
        # Filter out None values so we don't override URL parts with None
        conn_kwargs = {k: v for k, v in conn_kwargs.items() if v is not None}

        if config.url:
            redis_client = redis.from_url(config.url, **conn_kwargs)
        else:
            redis_client = redis.Redis(**conn_kwargs)

        await redis_client.ping()
        return Success(RedisProvider(config, redis_client))
    except Exception as e:
        return Failure(
            MemoryConnectionError(provider="Redis", message="Failed to connect to Redis", cause=e)
        )
