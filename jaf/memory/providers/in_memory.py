"""
In-memory memory provider implementation.

This provider stores conversations in Python dictionaries with LRU eviction.
Best for development, testing, or single-instance deployments where persistence
across restarts is not required.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import OrderedDict

from ..types import (
    MemoryProvider, ConversationMemory, MemoryQuery, InMemoryConfig,
    Result, Success, Failure, MemoryNotFoundError, MemoryStorageError
)
from ...core.types import Message

class InMemoryProvider:
    """
    In-memory implementation of MemoryProvider.
    
    Uses OrderedDict for LRU behavior and enforces memory limits through
    automatic eviction of oldest conversations.
    """
    
    def __init__(self, config: InMemoryConfig):
        self.config = config
        # Use OrderedDict for LRU behavior
        self._conversations: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._lock = asyncio.Lock()
        
        print(f"[MEMORY:InMemory] Initialized with max {config.max_conversations} conversations, {config.max_messages} messages each")
    
    async def _enforce_memory_limits(self) -> None:
        """Enforce conversation and message limits through LRU eviction."""
        # Remove oldest conversations if over limit
        while len(self._conversations) > self.config.max_conversations:
            oldest_id = next(iter(self._conversations))
            removed = self._conversations.pop(oldest_id)
            print(f"[MEMORY:InMemory] Evicted oldest conversation {oldest_id} with {len(removed.messages)} messages")
    
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
                
                # Enforce per-conversation message limits
                limited_messages = messages[-self.config.max_messages:] if len(messages) > self.config.max_messages else messages
                
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
                
                # Move to end (most recently used)
                if conversation_id in self._conversations:
                    del self._conversations[conversation_id]
                
                self._conversations[conversation_id] = conversation
                await self._enforce_memory_limits()
                
                print(f"[MEMORY:InMemory] Stored {len(limited_messages)} messages for conversation {conversation_id}")
                return Success()
                
            except Exception as e:
                error_msg = f"Failed to store messages: {str(e)}"
                print(f"[MEMORY:InMemory] {error_msg}")
                return Failure(error_msg)
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Retrieve conversation history."""
        async with self._lock:
            try:
                conversation = self._conversations.get(conversation_id)
                if conversation is None:
                    print(f"[MEMORY:InMemory] No conversation found for {conversation_id}")
                    return None
                
                # Move to end (most recently used) and update last activity
                del self._conversations[conversation_id]
                updated_metadata = dict(conversation.metadata or {})
                updated_metadata["last_activity"] = datetime.now()
                
                updated_conversation = ConversationMemory(
                    conversation_id=conversation.conversation_id,
                    user_id=conversation.user_id,
                    messages=conversation.messages,
                    metadata=updated_metadata
                )
                
                self._conversations[conversation_id] = updated_conversation
                
                print(f"[MEMORY:InMemory] Retrieved conversation {conversation_id} with {len(conversation.messages)} messages")
                return updated_conversation
                
            except Exception as e:
                print(f"[MEMORY:InMemory] Error retrieving conversation {conversation_id}: {str(e)}")
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
                existing = self._conversations.get(conversation_id)
                if existing is None:
                    error_msg = f"Conversation {conversation_id} not found"
                    print(f"[MEMORY:InMemory] {error_msg}")
                    return Failure(error_msg)
                
                # Combine existing and new messages
                combined_messages = list(existing.messages) + messages
                
                # Enforce message limits
                limited_messages = combined_messages[-self.config.max_messages:] if len(combined_messages) > self.config.max_messages else combined_messages
                
                now = datetime.now()
                updated_metadata = dict(existing.metadata or {})
                updated_metadata.update({
                    "updated_at": now,
                    "last_activity": now,
                    "total_messages": len(limited_messages),
                    **(metadata or {})
                })
                
                updated_conversation = ConversationMemory(
                    conversation_id=conversation_id,
                    user_id=existing.user_id,
                    messages=limited_messages,
                    metadata=updated_metadata
                )
                
                # Move to end (most recently used)
                del self._conversations[conversation_id]
                self._conversations[conversation_id] = updated_conversation
                
                print(f"[MEMORY:InMemory] Appended {len(messages)} messages to conversation {conversation_id} (total: {len(limited_messages)})")
                return Success()
                
            except Exception as e:
                error_msg = f"Failed to append messages: {str(e)}"
                print(f"[MEMORY:InMemory] {error_msg}")
                return Failure(error_msg)
    
    async def find_conversations(self, query: MemoryQuery) -> List[ConversationMemory]:
        """Search conversations by query parameters."""
        async with self._lock:
            try:
                results = []
                
                for conversation in self._conversations.values():
                    # Apply filters
                    if query.conversation_id and conversation.conversation_id != query.conversation_id:
                        continue
                    if query.user_id and conversation.user_id != query.user_id:
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
                
                # Sort by last activity (most recent first)
                results.sort(key=lambda c: c.metadata.get("last_activity", datetime.min) if c.metadata else datetime.min, reverse=True)
                
                # Apply pagination
                offset = query.offset or 0
                limit = query.limit or len(results)
                paginated_results = results[offset:offset + limit]
                
                print(f"[MEMORY:InMemory] Found {len(paginated_results)} conversations matching query")
                return paginated_results
                
            except Exception as e:
                print(f"[MEMORY:InMemory] Error finding conversations: {str(e)}")
                return []
    
    async def get_recent_messages(self, conversation_id: str, limit: int = 50) -> List[Message]:
        """Get recent messages from a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages[-limit:] if len(conversation.messages) > limit else conversation.messages
        print(f"[MEMORY:InMemory] Retrieved {len(messages)} recent messages for conversation {conversation_id}")
        return list(messages)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and return True if it existed."""
        async with self._lock:
            try:
                existed = conversation_id in self._conversations
                if existed:
                    del self._conversations[conversation_id]
                    print(f"[MEMORY:InMemory] Deleted conversation {conversation_id}")
                else:
                    print(f"[MEMORY:InMemory] Attempted to delete non-existent conversation {conversation_id}")
                return existed
                
            except Exception as e:
                print(f"[MEMORY:InMemory] Error deleting conversation {conversation_id}: {str(e)}")
                return False
    
    async def clear_user_conversations(self, user_id: str) -> int:
        """Clear all conversations for a user and return count deleted."""
        async with self._lock:
            try:
                to_delete = [
                    conv_id for conv_id, conversation in self._conversations.items()
                    if conversation.user_id == user_id
                ]
                
                for conv_id in to_delete:
                    del self._conversations[conv_id]
                
                print(f"[MEMORY:InMemory] Cleared {len(to_delete)} conversations for user {user_id}")
                return len(to_delete)
                
            except Exception as e:
                print(f"[MEMORY:InMemory] Error clearing conversations for user {user_id}: {str(e)}")
                return 0
    
    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics."""
        async with self._lock:
            try:
                conversations = [
                    conv for conv in self._conversations.values()
                    if not user_id or conv.user_id == user_id
                ]
                
                total_conversations = len(conversations)
                total_messages = sum(len(conv.messages) for conv in conversations)
                
                oldest_conversation = None
                newest_conversation = None
                
                if conversations and all(conv.metadata for conv in conversations):
                    created_dates = [
                        conv.metadata["created_at"] for conv in conversations
                        if conv.metadata and "created_at" in conv.metadata and isinstance(conv.metadata["created_at"], datetime)
                    ]
                    if created_dates:
                        oldest_conversation = min(created_dates)
                        newest_conversation = max(created_dates)
                
                return {
                    "total_conversations": total_conversations,
                    "total_messages": total_messages,
                    "oldest_conversation": oldest_conversation,
                    "newest_conversation": newest_conversation
                }
                
            except Exception as e:
                print(f"[MEMORY:InMemory] Error getting stats: {str(e)}")
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
            # Test basic operations
            test_id = f"health-check-{int(start_time.timestamp())}"
            test_messages = [{"role": "user", "content": "health check"}]
            
            # Test store
            store_result = await self.store_messages(test_id, test_messages)
            if isinstance(store_result, Failure):
                return {
                    "healthy": False,
                    "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "error": store_result.error
                }
            
            # Test retrieve
            retrieved = await self.get_conversation(test_id)
            if not retrieved:
                return {
                    "healthy": False,
                    "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "error": "Failed to retrieve test conversation"
                }
            
            # Test delete
            deleted = await self.delete_conversation(test_id)
            if not deleted:
                return {
                    "healthy": False,
                    "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "error": "Failed to delete test conversation"
                }
            
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "healthy": True,
                "latency_ms": latency_ms,
                "provider": "InMemory",
                "conversations_count": len(self._conversations)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "error": str(e)
            }
    
    async def close(self) -> None:
        """Close/cleanup the provider."""
        async with self._lock:
            conversation_count = len(self._conversations)
            self._conversations.clear()
            print(f"[MEMORY:InMemory] Closed provider, cleared {conversation_count} conversations")

def create_in_memory_provider(config: Optional[InMemoryConfig] = None) -> InMemoryProvider:
    """
    Factory function to create an in-memory provider instance.
    
    Args:
        config: Configuration for the provider. If None, uses defaults.
        
    Returns:
        Configured InMemoryProvider instance.
    """
    if config is None:
        config = InMemoryConfig()
    
    return InMemoryProvider(config)