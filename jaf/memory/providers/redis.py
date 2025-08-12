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

# Type hint for Redis client - compatible with redis-py async
try:
    import redis.asyncio as redis
    RedisClient = redis.Redis
except ImportError:
    # Fallback for type hints when redis is not installed
    RedisClient = Any

class RedisProvider:
    """
    Redis implementation of MemoryProvider.
    
    Uses JSON serialization to store conversation data in Redis with
    configurable TTL and key prefixing for namespace isolation.
    """
    
    def __init__(self, config: RedisConfig, redis_client: Any):
        self.config = config
        self.redis_client = redis_client
        self._lock = asyncio.Lock()
        
        print(f"[MEMORY:Redis] Initialized with prefix '{config.key_prefix}' on {config.host}:{config.port}")
    
    def _get_key(self, conversation_id: str) -> str:
        """Generate Redis key for conversation."""
        return f"{self.config.key_prefix}{conversation_id}"
    
    def _get_user_pattern(self, user_id: str) -> str:
        """Generate Redis key pattern for user conversations."""
        return f"{self.config.key_prefix}user:{user_id}:*"
    
    def _serialize_conversation(self, conversation: ConversationMemory) -> str:
        """Serialize conversation to JSON string."""
        # Convert datetime objects to ISO strings for JSON compatibility
        metadata = conversation.metadata or {}
        serializable_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, datetime):
                serializable_metadata[key] = value.isoformat()
            else:
                serializable_metadata[key] = value
        
        data = {
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "messages": [msg.model_dump() if hasattr(msg, 'model_dump') else msg.__dict__ for msg in conversation.messages],
            "metadata": serializable_metadata
        }
        
        return json.dumps(data, separators=(',', ':'))  # Compact JSON
    
    def _deserialize_conversation(self, data: str) -> ConversationMemory:
        """Deserialize conversation from JSON string."""
        parsed = json.loads(data)
        
        # Convert ISO strings back to datetime objects
        metadata = parsed.get("metadata", {})
        for key, value in metadata.items():
            if isinstance(value, str) and key in ["created_at", "updated_at", "last_activity"]:
                try:
                    metadata[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Keep as string if not valid ISO format
        
        # Reconstruct messages
        messages = []
        for msg_data in parsed.get("messages", []):
            if isinstance(msg_data, dict):
                # Create Message object from dict
                from ...core.types import Message
                messages.append(Message(**msg_data))
            else:
                messages.append(msg_data)
        
        return ConversationMemory(
            conversation_id=parsed["conversation_id"],
            user_id=parsed.get("user_id"),
            messages=messages,
            metadata=metadata
        )
    
    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Store messages for a conversation."""
        async with self._lock:
            try:
                now = datetime.now()
                
                conversation_metadata = {
                    "created_at": now,
                    "updated_at": now,
                    "total_messages": len(messages),
                    "last_activity": now,
                    **(metadata or {})
                }
                
                conversation = ConversationMemory(
                    conversation_id=conversation_id,
                    user_id=metadata.get("user_id") if metadata else None,
                    messages=messages,
                    metadata=conversation_metadata
                )
                
                key = self._get_key(conversation_id)
                value = self._serialize_conversation(conversation)
                
                await self.redis_client.set(key, value)
                
                # Set TTL if configured
                if self.config.ttl:
                    await self.redis_client.expire(key, self.config.ttl)
                
                print(f"[MEMORY:Redis] Stored {len(messages)} messages for conversation {conversation_id}")
                return Success()
                
            except Exception as e:
                error_msg = f"Failed to store messages: {str(e)}"
                print(f"[MEMORY:Redis] {error_msg}")
                return Failure(error_msg)
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Retrieve conversation history."""
        try:
            key = self._get_key(conversation_id)
            value = await self.redis_client.get(key)
            
            if value is None:
                print(f"[MEMORY:Redis] No conversation found for {conversation_id}")
                return None
            
            conversation = self._deserialize_conversation(value.decode('utf-8') if isinstance(value, bytes) else value)
            
            # Update last activity
            now = datetime.now()
            updated_metadata = dict(conversation.metadata or {})
            updated_metadata["last_activity"] = now
            
            updated_conversation = ConversationMemory(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                messages=conversation.messages,
                metadata=updated_metadata
            )
            
            # Store updated conversation (fire and forget)
            updated_value = self._serialize_conversation(updated_conversation)
            asyncio.create_task(self.redis_client.set(key, updated_value))
            
            print(f"[MEMORY:Redis] Retrieved conversation {conversation_id} with {len(conversation.messages)} messages")
            return updated_conversation
            
        except Exception as e:
            print(f"[MEMORY:Redis] Error retrieving conversation {conversation_id}: {str(e)}")
            return None
    
    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Append new messages to existing conversation."""
        async with self._lock:
            try:
                existing = await self.get_conversation(conversation_id)
                if existing is None:
                    error_msg = f"Conversation {conversation_id} not found"
                    print(f"[MEMORY:Redis] {error_msg}")
                    return Failure(error_msg)
                
                # Combine existing and new messages
                combined_messages = list(existing.messages) + messages
                
                now = datetime.now()
                updated_metadata = dict(existing.metadata or {})
                updated_metadata.update({
                    "updated_at": now,
                    "last_activity": now,
                    "total_messages": len(combined_messages),
                    **(metadata or {})
                })
                
                updated_conversation = ConversationMemory(
                    conversation_id=conversation_id,
                    user_id=existing.user_id,
                    messages=combined_messages,
                    metadata=updated_metadata
                )
                
                key = self._get_key(conversation_id)
                value = self._serialize_conversation(updated_conversation)
                
                await self.redis_client.set(key, value)
                
                # Refresh TTL if configured
                if self.config.ttl:
                    await self.redis_client.expire(key, self.config.ttl)
                
                print(f"[MEMORY:Redis] Appended {len(messages)} messages to conversation {conversation_id} (total: {len(combined_messages)})")
                return Success()
                
            except Exception as e:
                error_msg = f"Failed to append messages: {str(e)}"
                print(f"[MEMORY:Redis] {error_msg}")
                return Failure(error_msg)
    
    async def find_conversations(self, query: MemoryQuery) -> List[ConversationMemory]:
        """Search conversations by query parameters."""
        try:
            # Get conversation keys
            if query.user_id:
                pattern = self._get_user_pattern(query.user_id)
            else:
                pattern = f"{self.config.key_prefix}*"
            
            keys = await self.redis_client.keys(pattern)
            results = []
            
            # Fetch conversations in batches to avoid overwhelming Redis
            batch_size = 50
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                # Get values for this batch
                if batch_keys:
                    values = await self.redis_client.mget(batch_keys)
                    
                    for value in values:
                        if value is None:
                            continue
                        
                        try:
                            conversation = self._deserialize_conversation(
                                value.decode('utf-8') if isinstance(value, bytes) else value
                            )
                            
                            # Apply filters
                            if query.conversation_id and conversation.conversation_id != query.conversation_id:
                                continue
                            if query.trace_id and conversation.metadata and conversation.metadata.get("trace_id") != query.trace_id:
                                continue
                            if query.since and conversation.metadata:
                                created_at = conversation.metadata.get("created_at")
                                if created_at and isinstance(created_at, datetime) and created_at < query.since:
                                    continue
                            if query.until and conversation.metadata:
                                created_at = conversation.metadata.get("created_at")
                                if created_at and isinstance(created_at, datetime) and created_at > query.until:
                                    continue
                            
                            results.append(conversation)
                            
                        except Exception as e:
                            print(f"[MEMORY:Redis] Error deserializing conversation: {str(e)}")
                            continue
            
            # Sort by last activity (most recent first)
            results.sort(
                key=lambda c: c.metadata.get("last_activity", datetime.min) if c.metadata else datetime.min,
                reverse=True
            )
            
            # Apply pagination
            offset = query.offset or 0
            limit = query.limit or len(results)
            paginated_results = results[offset:offset + limit]
            
            print(f"[MEMORY:Redis] Found {len(paginated_results)} conversations matching query")
            return paginated_results
            
        except Exception as e:
            print(f"[MEMORY:Redis] Error finding conversations: {str(e)}")
            return []
    
    async def get_recent_messages(self, conversation_id: str, limit: int = 50) -> List[Message]:
        """Get recent messages from a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages[-limit:] if len(conversation.messages) > limit else conversation.messages
        print(f"[MEMORY:Redis] Retrieved {len(messages)} recent messages for conversation {conversation_id}")
        return list(messages)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and return True if it existed."""
        try:
            key = self._get_key(conversation_id)
            deleted = await self.redis_client.delete(key)
            
            existed = deleted > 0
            if existed:
                print(f"[MEMORY:Redis] Deleted conversation {conversation_id}")
            else:
                print(f"[MEMORY:Redis] Attempted to delete non-existent conversation {conversation_id}")
            
            return existed
            
        except Exception as e:
            print(f"[MEMORY:Redis] Error deleting conversation {conversation_id}: {str(e)}")
            return False
    
    async def clear_user_conversations(self, user_id: str) -> int:
        """Clear all conversations for a user and return count deleted."""
        try:
            pattern = self._get_user_pattern(user_id)
            keys = await self.redis_client.keys(pattern)
            
            if not keys:
                return 0
            
            # Delete in batches
            deleted_count = 0
            batch_size = 100
            
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                if batch:
                    deleted_count += await self.redis_client.delete(*batch)
            
            print(f"[MEMORY:Redis] Cleared {deleted_count} conversations for user {user_id}")
            return deleted_count
            
        except Exception as e:
            print(f"[MEMORY:Redis] Error clearing conversations for user {user_id}: {str(e)}")
            return 0
    
    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics."""
        try:
            pattern = self._get_user_pattern(user_id) if user_id else f"{self.config.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            total_conversations = len(keys)
            total_messages = 0
            oldest_conversation = None
            newest_conversation = None
            
            # Sample conversations to get statistics (to avoid loading all data)
            sample_size = min(100, total_conversations)
            if sample_size > 0:
                sample_keys = keys[:sample_size]
                values = await self.redis_client.mget(sample_keys)
                
                created_dates = []
                for value in values:
                    if value is None:
                        continue
                    
                    try:
                        conversation = self._deserialize_conversation(
                            value.decode('utf-8') if isinstance(value, bytes) else value
                        )
                        total_messages += len(conversation.messages)
                        
                        if conversation.metadata and "created_at" in conversation.metadata:
                            created_at = conversation.metadata["created_at"]
                            if isinstance(created_at, datetime):
                                created_dates.append(created_at)
                                
                    except Exception:
                        continue
                
                if created_dates:
                    oldest_conversation = min(created_dates)
                    newest_conversation = max(created_dates)
                
                # Extrapolate message count for all conversations
                if sample_size < total_conversations:
                    avg_messages = total_messages / sample_size if sample_size > 0 else 0
                    total_messages = int(avg_messages * total_conversations)
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "oldest_conversation": oldest_conversation,
                "newest_conversation": newest_conversation
            }
            
        except Exception as e:
            print(f"[MEMORY:Redis] Error getting stats: {str(e)}")
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "oldest_conversation": None,
                "newest_conversation": None
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and return status information."""
        start_time = datetime.now()
        
        try:
            # Test Redis connectivity
            await self.redis_client.ping()
            
            # Test basic operations
            test_id = f"health-check-{int(start_time.timestamp())}"
            test_key = self._get_key(test_id)
            
            # Test set/get/delete cycle
            await self.redis_client.set(test_key, json.dumps({"test": True}))
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if value is None:
                return {
                    "healthy": False,
                    "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "error": "Failed to retrieve test value"
                }
            
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "healthy": True,
                "latency_ms": latency_ms,
                "provider": "Redis",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients")
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "error": str(e)
            }
    
    async def close(self) -> None:
        """Close/cleanup the provider."""
        try:
            if hasattr(self.redis_client, 'close'):
                await self.redis_client.close()
            print("[MEMORY:Redis] Closed Redis connection")
        except Exception as e:
            print(f"[MEMORY:Redis] Error closing Redis connection: {str(e)}")

async def create_redis_provider(config: RedisConfig, redis_client: Any) -> RedisProvider:
    """
    Factory function to create a Redis provider instance.
    
    Args:
        config: Configuration for the Redis provider.
        redis_client: Connected Redis client instance.
        
    Returns:
        Configured RedisProvider instance.
        
    Raises:
        MemoryConnectionError: If Redis connection fails.
    """
    try:
        # Test connection
        await redis_client.ping()
        print(f"[MEMORY:Redis] Successfully connected to Redis at {config.host}:{config.port}")
        
        return RedisProvider(config, redis_client)
        
    except Exception as e:
        raise MemoryConnectionError(f"Failed to connect to Redis: {str(e)}", "Redis", e)