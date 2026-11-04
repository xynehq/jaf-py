"""
Redis memory provider implementation.

This provider uses Redis for persistent conversation storage with JSON serialization.
Best for production environments with shared state and persistence across restarts.
"""

import asyncio
import logging
import random
import weakref
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

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis

    from redis.exceptions import WatchError

    RedisClient = redis.Redis
except ImportError:
    RedisClient = Any

    class WatchError(Exception):  # type: ignore[no-redef]
        """Placeholder so the retry loop below works without redis installed."""


_MAX_TRANSACTION_RETRIES = 10


class _ConversationMissing(Exception):
    """Raised by an update callback when the conversation does not exist."""


class RedisProvider(MemoryProvider):
    """
    Redis implementation of MemoryProvider.
    """

    def __init__(self, config: RedisConfig, redis_client: RedisClient):
        self.config = config
        self.redis_client = redis_client
        # Used only when the client has no pipeline support (e.g. minimal test doubles)
        self._fallback_lock = asyncio.Lock()
        # Per-conversation locks serialize writers in this process; WATCH covers other processes
        self._key_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = (
            weakref.WeakValueDictionary()
        )

        logger.debug(f"[MEMORY:Redis] Initialized with prefix '{config.key_prefix}'")

    def _get_key(self, conversation_id: str) -> str:
        return f"{self.config.key_prefix}{conversation_id}"

    async def _update(self, conversation_id: str, mutate):
        """
        Atomic read-modify-write for one conversation.

        ``mutate(existing)`` receives the current ConversationMemory (or None) and returns
        ``(new_conversation_or_None, result)``; None skips the write. It may raise
        _ConversationMissing. Uses WATCH/MULTI/EXEC and retries if another client changes the
        key between the read and the write, so concurrent updates are never lost.
        """
        key = self._get_key(conversation_id)

        if not hasattr(self.redis_client, "pipeline"):
            async with self._fallback_lock:
                raw = await self.redis_client.get(key)
                new_conversation, result = mutate(self._deserialize(raw) if raw else None)
                if new_conversation is not None:
                    await self.redis_client.set(
                        key, self._serialize(new_conversation), ex=self.config.ttl
                    )
                return result

        lock = self._key_locks.get(key)
        if lock is None:
            lock = self._key_locks.setdefault(key, asyncio.Lock())

        async with lock:
            for attempt in range(_MAX_TRANSACTION_RETRIES):
                async with self.redis_client.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(key)
                        raw = await pipe.get(key)
                        new_conversation, result = mutate(
                            self._deserialize(raw) if raw else None
                        )
                        if new_conversation is None:
                            await pipe.unwatch()
                            return result
                        pipe.multi()
                        pipe.set(key, self._serialize(new_conversation), ex=self.config.ttl)
                        await pipe.execute()
                        return result
                    except WatchError:
                        pass
                # Another process changed the key; back off briefly before re-reading
                await asyncio.sleep(random.uniform(0, 0.01 * (attempt + 1)))
        raise RuntimeError(
            f"Could not update conversation {conversation_id}: too much write contention"
        )

    @staticmethod
    def _failure(exc: Exception, operation: str, conversation_id: str):
        if isinstance(exc, _ConversationMissing):
            return Failure(
                MemoryNotFoundError(
                    message=f"Conversation {conversation_id} not found",
                    provider="Redis",
                    conversation_id=conversation_id,
                )
            )
        return Failure(
            MemoryStorageError(
                operation=operation, provider="Redis", message=str(exc), cause=exc
            )
        )

    def _serialize(self, conversation: ConversationMemory) -> str:
        """Serialize conversation using shared utilities."""
        from ..utils import serialize_conversation_for_json

        return serialize_conversation_for_json(conversation)

    def _deserialize(self, data: str) -> ConversationMemory:
        """Deserialize conversation using shared utilities."""
        from ..utils import deserialize_conversation_from_json

        return deserialize_conversation_from_json(data)

    @staticmethod
    def _merged(
        existing: Optional[ConversationMemory],
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]],
    ) -> ConversationMemory:
        now = datetime.now()
        return ConversationMemory(
            conversation_id=conversation_id,
            # The first owner sticks: later writes cannot reassign the conversation
            user_id=(existing.user_id if existing else None)
            or (metadata.get("user_id") if metadata else None),
            messages=messages,
            metadata={
                **(existing.metadata if existing else {}),
                **(metadata or {}),
                # Bookkeeping fields come last so caller-supplied metadata cannot overwrite them
                "created_at": (existing.metadata.get("created_at") if existing else None) or now,
                "updated_at": now,
                "total_messages": len(messages),
                "last_activity": now,
            },
        )

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result[None, MemoryStorageError]:
        try:
            await self._update(
                conversation_id,
                lambda existing: (self._merged(existing, conversation_id, messages, metadata), None),
            )
            return Success(None)
        except Exception as e:
            return self._failure(e, "store_messages", conversation_id)

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

            # Refresh the TTL only; rewriting the value here could overwrite a concurrent update
            if self.config.ttl:
                await self.redis_client.expire(key, self.config.ttl)
            return Success(conversation)
        except Exception as e:
            return self._failure(e, "get_conversation", conversation_id)

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        def mutate(existing: Optional[ConversationMemory]):
            if not existing:
                raise _ConversationMissing()
            all_messages = list(existing.messages) + messages
            now = datetime.now()
            updated_metadata = {
                **existing.metadata,
                "updated_at": now,
                "last_activity": now,
                "total_messages": len(all_messages),
                **(metadata or {}),
            }
            return self._merged(existing, conversation_id, all_messages, updated_metadata), None

        try:
            await self._update(conversation_id, mutate)
            return Success(None)
        except Exception as e:
            return self._failure(e, "append_messages", conversation_id)

    async def find_conversations(
        self, query: MemoryQuery
    ) -> Result[List[ConversationMemory], MemoryStorageError]:
        try:
            keys = [
                key
                async for key in self.redis_client.scan_iter(
                    match=f"{self.config.key_prefix}*", count=500
                )
            ]
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
                    if query.trace_id and (conv.metadata or {}).get("trace_id") != query.trace_id:
                        continue
                    activity = (conv.metadata or {}).get("updated_at") or (conv.metadata or {}).get(
                        "created_at"
                    )
                    if isinstance(activity, str):
                        try:
                            activity = datetime.fromisoformat(activity)
                        except ValueError:
                            activity = None
                    if activity is not None and (query.since or query.until):
                        try:
                            if query.since and activity < query.since:
                                continue
                            if query.until and activity > query.until:
                                continue
                        except TypeError:
                            pass  # naive/aware mismatch: do not guess, keep the conversation
                    conversations.append(conv)
            offset = query.offset or 0
            if query.offset is not None or query.limit is not None:
                conversations = conversations[offset : offset + query.limit if query.limit is not None else None]
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
            keys = [
                key
                async for key in self.redis_client.scan_iter(
                    match=f"{self.config.key_prefix}*", count=500
                )
            ]
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

        def mutate(conversation: Optional[ConversationMemory]):
            if not conversation:
                raise _ConversationMissing()

            messages = list(conversation.messages)
            truncate_index = find_message_index(messages, message_id)
            if truncate_index is None:
                # Message not found, nothing to truncate
                return None, 0

            truncated_messages = messages[:truncate_index]
            removed_count = len(messages) - len(truncated_messages)
            now = datetime.now()
            updated = ConversationMemory(
                conversation_id=conversation_id,
                user_id=conversation.user_id,
                messages=truncated_messages,
                metadata={
                    **conversation.metadata,
                    "updated_at": now,
                    "last_activity": now,
                    "total_messages": len(truncated_messages),
                    "regeneration_truncated": True,
                    "truncated_at": now.isoformat(),
                    "messages_removed": removed_count,
                },
            )
            return updated, removed_count

        try:
            removed_count = await self._update(conversation_id, mutate)
            logger.debug(
                f"[MEMORY:Redis] Truncated conversation {conversation_id}: removed {removed_count} messages after message {message_id}"
            )
            return Success(removed_count)
        except Exception as e:
            return self._failure(e, "truncate_conversation_after", conversation_id)

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
                logger.debug(f"[MEMORY:Redis] Message {message_id} not found in conversation {conversation_id}")
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

            logger.debug(f"[MEMORY:Redis] Retrieved conversation {conversation_id} until message {message_id}: {len(truncated_messages)} messages")
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

        def mutate(conversation: Optional[ConversationMemory]):
            if not conversation:
                raise _ConversationMissing()

            regeneration_points = list(conversation.metadata.get("regeneration_points", []))
            regeneration_point = {
                "message_id": str(message_id),
                "timestamp": datetime.now().isoformat(),
                **regeneration_metadata,
            }
            regeneration_points.append(regeneration_point)

            updated = ConversationMemory(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                messages=conversation.messages,
                metadata={
                    **conversation.metadata,
                    "regeneration_points": regeneration_points,
                    "last_regeneration": regeneration_point,
                    "updated_at": datetime.now(),
                    "regeneration_count": len(regeneration_points),
                },
            )
            return updated, None

        try:
            await self._update(conversation_id, mutate)
            logger.debug(
                f"[MEMORY:Redis] Marked regeneration point for conversation {conversation_id} at message {message_id}"
            )
            return Success(None)
        except Exception as e:
            return self._failure(e, "mark_regeneration_point", conversation_id)

    async def get_previous_response_id(
        self, conversation_id: str
    ) -> Result[Optional[str], MemoryStorageError]:
        conv_result = await self.get_conversation(conversation_id)
        if isinstance(conv_result, Failure):
            return conv_result
        conversation = conv_result.data
        return Success(conversation.metadata.get("previous_response_id") if conversation else None)

    async def get_previous_response_message_id(
        self, conversation_id: str
    ) -> Result[Optional[str], MemoryStorageError]:
        conv_result = await self.get_conversation(conversation_id)
        if isinstance(conv_result, Failure):
            return conv_result
        conversation = conv_result.data
        return Success(
            conversation.metadata.get("previous_response_message_id") if conversation else None
        )

    async def get_prior_response_id(
        self, conversation_id: str
    ) -> Result[Optional[str], MemoryStorageError]:
        conv_result = await self.get_conversation(conversation_id)
        if isinstance(conv_result, Failure):
            return conv_result
        conversation = conv_result.data
        return Success(conversation.metadata.get("prior_response_id") if conversation else None)

    async def get_previous_response_message_index(
        self, conversation_id: str
    ) -> Result[Optional[int], MemoryStorageError]:
        conv_result = await self.get_conversation(conversation_id)
        if isinstance(conv_result, Failure):
            return conv_result
        conversation = conv_result.data
        return Success(
            conversation.metadata.get("previous_response_message_index") if conversation else None
        )

    async def set_previous_response_id(
        self,
        conversation_id: str,
        response_id: str,
        message_id: Optional[str] = None,
        message_index: Optional[int] = None,
        user_id: Optional[str] = None,
        shift: bool = True,
    ) -> Result[None, MemoryStorageError]:
        def mutate(conversation: Optional[ConversationMemory]):
            prior_response_id = (
                (conversation.metadata if conversation else {}).get("previous_response_id")
                if shift
                else None
            )
            updated = ConversationMemory(
                conversation_id=conversation_id,
                user_id=user_id if conversation is None else conversation.user_id,
                messages=conversation.messages if conversation else [],
                metadata={
                    **(conversation.metadata if conversation else {}),
                    "prior_response_id": prior_response_id,
                    "previous_response_id": response_id,
                    "previous_response_message_id": message_id,
                    "previous_response_message_index": message_index,
                },
            )
            return updated, None

        try:
            await self._update(conversation_id, mutate)
            return Success(None)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    message=f"Failed to set previous_response_id: {e}",
                    provider="Redis",
                    operation="set_previous_response_id",
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
        if config.ssl or config.ssl_ca_certs:
            conn_kwargs["ssl"] = True
            if config.ssl_ca_certs:
                conn_kwargs["ssl_ca_certs"] = config.ssl_ca_certs

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
