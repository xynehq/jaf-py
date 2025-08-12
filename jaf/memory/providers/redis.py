"""
Redis memory provider implementation.

This provider uses Redis for persistent conversation storage with JSON serialization.
Best for production environments with shared state and persistence across restarts.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..types import (
    MemoryProvider, ConversationMemory, MemoryQuery, RedisConfig,
    Result, Success, Failure, MemoryConnectionError, MemoryNotFoundError, MemoryStorageError
)
from ...core.types import Message

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
        data = {
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "messages": [msg.dict() for msg in conversation.messages],
            "metadata": {k: v.isoformat() if isinstance(v, datetime) else v for k, v in conversation.metadata.items()}
        }
        return json.dumps(data)
    
    def _deserialize(self, data: str) -> ConversationMemory:
        parsed = json.loads(data)
        metadata = parsed.get("metadata", {})
        for key in ["created_at", "updated_at", "last_activity"]:
            if key in metadata and isinstance(metadata[key], str):
                metadata[key] = datetime.fromisoformat(metadata[key])
        
        return ConversationMemory(
            conversation_id=parsed["conversation_id"],
            user_id=parsed.get("user_id"),
            messages=[Message(**msg) for msg in parsed.get("messages", [])],
            metadata=metadata
        )
    
    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
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
                    **(metadata or {})
                }
            )
            key = self._get_key(conversation_id)
            await self.redis_client.set(key, self._serialize(conversation), ex=self.config.ttl)
            return Success(None)
        except Exception as e:
            return Failure(MemoryStorageError(operation="store_messages", provider="Redis", message=str(e), cause=e))
    
    async def get_conversation(
        self, 
        conversation_id: str
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
            return Failure(MemoryStorageError(operation="get_conversation", provider="Redis", message=str(e), cause=e))
    
    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        result = await self.get_conversation(conversation_id)
        if isinstance(result, Failure):
            return result
        
        existing = result.data
        if not existing:
            return Failure(MemoryNotFoundError(conversation_id=conversation_id, provider="Redis", message=f"Conversation {conversation_id} not found"))
        
        existing.messages.extend(messages)
        existing.metadata.update({
            "updated_at": datetime.now(),
            "last_activity": datetime.now(),
            "total_messages": len(existing.messages),
            **(metadata or {})
        })
        
        return await self.store_messages(conversation_id, existing.messages, existing.metadata)
    
    async def find_conversations(
        self, 
        query: MemoryQuery
    ) -> Result[List[ConversationMemory], MemoryStorageError]:
        try:
            keys = await self.redis_client.keys(f"{self.config.key_prefix}*")
            conversations = []
            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    conv = self._deserialize(value)
                    # Filtering logic here
                    conversations.append(conv)
            return Success(conversations)
        except Exception as e:
            return Failure(MemoryStorageError(operation="find_conversations", provider="Redis", message=str(e), cause=e))
    
    async def get_recent_messages(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> Result[List[Message], Union[MemoryNotFoundError, MemoryStorageError]]:
        result = await self.get_conversation(conversation_id)
        if isinstance(result, Failure):
            return result
        
        conversation = result.data
        if not conversation:
            return Failure(MemoryNotFoundError(conversation_id=conversation_id, provider="Redis", message=f"Conversation {conversation_id} not found"))
        
        return Success(conversation.messages[-limit:])
    
    async def delete_conversation(
        self, 
        conversation_id: str
    ) -> Result[bool, MemoryStorageError]:
        try:
            deleted = await self.redis_client.delete(self._get_key(conversation_id))
            return Success(deleted > 0)
        except Exception as e:
            return Failure(MemoryStorageError(operation="delete_conversation", provider="Redis", message=str(e), cause=e))
    
    async def clear_user_conversations(
        self, 
        user_id: str
    ) -> Result[int, MemoryStorageError]:
        # This is inefficient in Redis, consider a different approach for production
        return Failure(MemoryStorageError(operation="clear_user_conversations", provider="Redis", message="clear_user_conversations not efficiently supported"))
    
    async def get_stats(
        self, 
        user_id: Optional[str] = None
    ) -> Result[Dict[str, Any], MemoryStorageError]:
        try:
            keys = await self.redis_client.keys(f"{self.config.key_prefix}*")
            return Success({"total_conversations": len(keys)})
        except Exception as e:
            return Failure(MemoryStorageError(operation="get_stats", provider="Redis", message=str(e), cause=e))
    
    async def health_check(self) -> Result[Dict[str, Any], MemoryConnectionError]:
        try:
            await self.redis_client.ping()
            return Success({"healthy": True})
        except Exception as e:
            return Failure(MemoryConnectionError(provider="Redis", message="Redis health check failed", cause=e))
    
    async def close(self) -> Result[None, MemoryConnectionError]:
        try:
            await self.redis_client.close()
            return Success(None)
        except Exception as e:
            return Failure(MemoryConnectionError(provider="Redis", message="Failed to close Redis connection", cause=e))

async def create_redis_provider(config: RedisConfig) -> Result[RedisProvider, MemoryConnectionError]:
    try:
        redis_client = redis.from_url(config.url) if config.url else redis.Redis(
            host=config.host, port=config.port, db=config.db, password=config.password
        )
        await redis_client.ping()
        return Success(RedisProvider(config, redis_client))
    except Exception as e:
        return Failure(MemoryConnectionError(provider="Redis", message="Failed to connect to Redis", cause=e))
