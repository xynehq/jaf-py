"""
PostgreSQL memory provider implementation.

This provider uses PostgreSQL for fully persistent conversation storage with JSONB columns.
Best for production environments requiring complex queries and full persistence.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..types import (
    MemoryProvider, ConversationMemory, MemoryQuery, PostgresConfig,
    Result, Success, Failure, MemoryConnectionError, MemoryNotFoundError, MemoryStorageError
)
from ...core.types import Message

# Type hint for PostgreSQL client - compatible with asyncpg
try:
    import asyncpg
    PostgresClient = Union[asyncpg.Connection, asyncpg.Pool]
except ImportError:
    # Fallback for type hints when asyncpg is not installed
    PostgresClient = Any

class PostgresProvider:
    """
    PostgreSQL implementation of MemoryProvider.
    
    Uses JSONB columns for efficient storage and querying of conversation data
    with automatic schema initialization and indexing.
    """
    
    def __init__(self, config: PostgresConfig, postgres_client: Any):
        self.config = config
        self.postgres_client = postgres_client
        self._lock = asyncio.Lock()
        self._schema_initialized = False
        
        print(f"[MEMORY:Postgres] Initialized for database {config.database} on {config.host}:{config.port}")
    
    async def _ensure_schema(self):
        """Initialize database schema if not already done."""
        if self._schema_initialized:
            return
            
        async with self._lock:
            if self._schema_initialized:
                return
                
            try:
                # Create table with JSONB columns for efficient storage and querying
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    conversation_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255),
                    messages JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_user_id 
                    ON {self.config.table_name} (user_id);
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_created_at 
                    ON {self.config.table_name} (created_at);
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_last_activity 
                    ON {self.config.table_name} (last_activity);
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_metadata_gin 
                    ON {self.config.table_name} USING GIN (metadata);
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_trace_id 
                    ON {self.config.table_name} ((metadata->>'trace_id'));
                """
                
                if hasattr(self.postgres_client, 'execute'):
                    # Direct connection
                    await self.postgres_client.execute(create_table_sql)
                else:
                    # Connection pool
                    async with self.postgres_client.acquire() as conn:
                        await conn.execute(create_table_sql)
                
                self._schema_initialized = True
                print(f"[MEMORY:Postgres] Schema initialized for table {self.config.table_name}")
                
            except Exception as e:
                raise MemoryConnectionError(f"Failed to initialize PostgreSQL schema: {str(e)}", "PostgreSQL", e)
    
    async def _execute_query(self, query: str, *args):
        """Execute a query with proper connection handling."""
        await self._ensure_schema()
        
        if hasattr(self.postgres_client, 'execute'):
            # Direct connection
            return await self.postgres_client.execute(query, *args)
        else:
            # Connection pool
            async with self.postgres_client.acquire() as conn:
                return await conn.execute(query, *args)
    
    async def _fetch_query(self, query: str, *args):
        """Fetch query results with proper connection handling."""
        await self._ensure_schema()
        
        if hasattr(self.postgres_client, 'fetch'):
            # Direct connection
            return await self.postgres_client.fetch(query, *args)
        else:
            # Connection pool
            async with self.postgres_client.acquire() as conn:
                return await conn.fetch(query, *args)
    
    async def _fetchrow_query(self, query: str, *args):
        """Fetch single row with proper connection handling."""
        await self._ensure_schema()
        
        if hasattr(self.postgres_client, 'fetchrow'):
            # Direct connection
            return await self.postgres_client.fetchrow(query, *args)
        else:
            # Connection pool
            async with self.postgres_client.acquire() as conn:
                return await conn.fetchrow(query, *args)
    
    def _serialize_messages(self, messages: List[Message]) -> str:
        """Serialize messages to JSON string."""
        return json.dumps([
            msg.model_dump() if hasattr(msg, 'model_dump') else msg.__dict__ 
            for msg in messages
        ], separators=(',', ':'))
    
    def _deserialize_messages(self, data: Union[str, list]) -> List[Message]:
        """Deserialize messages from JSON data."""
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data
        
        messages = []
        for msg_data in parsed:
            if isinstance(msg_data, dict):
                from ...core.types import Message
                messages.append(Message(**msg_data))
            else:
                messages.append(msg_data)
        
        return messages
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize metadata to JSON string."""
        # Convert datetime objects to ISO strings
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                serializable_metadata[key] = value.isoformat()
            else:
                serializable_metadata[key] = value
        
        return json.dumps(serializable_metadata, separators=(',', ':'))
    
    def _deserialize_metadata(self, data: Union[str, dict]) -> Dict[str, Any]:
        """Deserialize metadata from JSON data."""
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data or {}
        
        # Convert ISO strings back to datetime objects
        for key, value in parsed.items():
            if isinstance(value, str) and key in ["created_at", "updated_at", "last_activity"]:
                try:
                    parsed[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Keep as string if not valid ISO format
        
        return parsed
    
    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Store messages for a conversation."""
        try:
            now = datetime.now()
            
            conversation_metadata = {
                "total_messages": len(messages),
                **(metadata or {})
            }
            
            messages_json = self._serialize_messages(messages)
            metadata_json = self._serialize_metadata(conversation_metadata)
            
            # Use INSERT ... ON CONFLICT for upsert behavior
            query = f"""
            INSERT INTO {self.config.table_name} 
            (conversation_id, user_id, messages, metadata, created_at, updated_at, last_activity)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (conversation_id) 
            DO UPDATE SET 
                messages = $3,
                metadata = $4,
                updated_at = $6,
                last_activity = $7
            """
            
            await self._execute_query(
                query,
                conversation_id,
                metadata.get("user_id") if metadata else None,
                messages_json,
                metadata_json,
                now,
                now,
                now
            )
            
            print(f"[MEMORY:Postgres] Stored {len(messages)} messages for conversation {conversation_id}")
            return Success()
            
        except Exception as e:
            error_msg = f"Failed to store messages: {str(e)}"
            print(f"[MEMORY:Postgres] {error_msg}")
            return Failure(error_msg)
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Retrieve conversation history."""
        try:
            query = f"""
            SELECT conversation_id, user_id, messages, metadata, created_at, updated_at, last_activity
            FROM {self.config.table_name}
            WHERE conversation_id = $1
            """
            
            row = await self._fetchrow_query(query, conversation_id)
            
            if not row:
                print(f"[MEMORY:Postgres] No conversation found for {conversation_id}")
                return None
            
            # Update last activity
            now = datetime.now()
            update_query = f"""
            UPDATE {self.config.table_name} 
            SET last_activity = $1 
            WHERE conversation_id = $2
            """
            asyncio.create_task(self._execute_query(update_query, now, conversation_id))
            
            # Deserialize data
            messages = self._deserialize_messages(row['messages'])
            metadata = self._deserialize_metadata(row['metadata'])
            
            # Add database timestamps to metadata
            metadata.update({
                "created_at": row['created_at'],
                "updated_at": row['updated_at'],
                "last_activity": now,  # Use current time since we just updated it
                "total_messages": len(messages)
            })
            
            conversation = ConversationMemory(
                conversation_id=row['conversation_id'],
                user_id=row['user_id'],
                messages=messages,
                metadata=metadata
            )
            
            print(f"[MEMORY:Postgres] Retrieved conversation {conversation_id} with {len(messages)} messages")
            return conversation
            
        except Exception as e:
            print(f"[MEMORY:Postgres] Error retrieving conversation {conversation_id}: {str(e)}")
            return None
    
    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Append new messages to existing conversation."""
        try:
            existing = await self.get_conversation(conversation_id)
            if existing is None:
                error_msg = f"Conversation {conversation_id} not found"
                print(f"[MEMORY:Postgres] {error_msg}")
                return Failure(error_msg)
            
            # Combine existing and new messages
            combined_messages = list(existing.messages) + messages
            
            now = datetime.now()
            updated_metadata = dict(existing.metadata or {})
            updated_metadata.update({
                "total_messages": len(combined_messages),
                **(metadata or {})
            })
            
            messages_json = self._serialize_messages(combined_messages)
            metadata_json = self._serialize_metadata(updated_metadata)
            
            query = f"""
            UPDATE {self.config.table_name}
            SET messages = $1, metadata = $2, updated_at = $3, last_activity = $3
            WHERE conversation_id = $4
            """
            
            await self._execute_query(query, messages_json, metadata_json, now, conversation_id)
            
            print(f"[MEMORY:Postgres] Appended {len(messages)} messages to conversation {conversation_id} (total: {len(combined_messages)})")
            return Success()
            
        except Exception as e:
            error_msg = f"Failed to append messages: {str(e)}"
            print(f"[MEMORY:Postgres] {error_msg}")
            return Failure(error_msg)
    
    async def find_conversations(self, query: MemoryQuery) -> List[ConversationMemory]:
        """Search conversations by query parameters."""
        try:
            sql = f"""
            SELECT conversation_id, user_id, messages, metadata, created_at, updated_at, last_activity
            FROM {self.config.table_name}
            WHERE 1=1
            """
            params = []
            param_index = 1
            
            if query.conversation_id:
                sql += f" AND conversation_id = ${param_index}"
                params.append(query.conversation_id)
                param_index += 1
            
            if query.user_id:
                sql += f" AND user_id = ${param_index}"
                params.append(query.user_id)
                param_index += 1
            
            if query.trace_id:
                sql += f" AND metadata->>'trace_id' = ${param_index}"
                params.append(str(query.trace_id))
                param_index += 1
            
            if query.since:
                sql += f" AND created_at >= ${param_index}"
                params.append(query.since)
                param_index += 1
            
            if query.until:
                sql += f" AND created_at <= ${param_index}"
                params.append(query.until)
                param_index += 1
            
            # Sort by last activity (most recent first)
            sql += " ORDER BY last_activity DESC"
            
            # Add pagination
            if query.limit:
                sql += f" LIMIT ${param_index}"
                params.append(query.limit)
                param_index += 1
            
            if query.offset:
                sql += f" OFFSET ${param_index}"
                params.append(query.offset)
                param_index += 1
            
            rows = await self._fetch_query(sql, *params)
            
            conversations = []
            for row in rows:
                messages = self._deserialize_messages(row['messages'])
                metadata = self._deserialize_metadata(row['metadata'])
                
                # Add database timestamps to metadata
                metadata.update({
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at'],
                    "last_activity": row['last_activity'],
                    "total_messages": len(messages)
                })
                
                conversation = ConversationMemory(
                    conversation_id=row['conversation_id'],
                    user_id=row['user_id'],
                    messages=messages,
                    metadata=metadata
                )
                conversations.append(conversation)
            
            print(f"[MEMORY:Postgres] Found {len(conversations)} conversations matching query")
            return conversations
            
        except Exception as e:
            print(f"[MEMORY:Postgres] Error finding conversations: {str(e)}")
            return []
    
    async def get_recent_messages(self, conversation_id: str, limit: int = 50) -> List[Message]:
        """Get recent messages from a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages[-limit:] if len(conversation.messages) > limit else conversation.messages
        print(f"[MEMORY:Postgres] Retrieved {len(messages)} recent messages for conversation {conversation_id}")
        return list(messages)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and return True if it existed."""
        try:
            query = f"DELETE FROM {self.config.table_name} WHERE conversation_id = $1"
            result = await self._execute_query(query, conversation_id)
            
            # Extract row count from result (asyncpg returns command tag like "DELETE 1")
            deleted = int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else 0
            
            if deleted > 0:
                print(f"[MEMORY:Postgres] Deleted conversation {conversation_id}")
            else:
                print(f"[MEMORY:Postgres] Attempted to delete non-existent conversation {conversation_id}")
            
            return deleted > 0
            
        except Exception as e:
            print(f"[MEMORY:Postgres] Error deleting conversation {conversation_id}: {str(e)}")
            return False
    
    async def clear_user_conversations(self, user_id: str) -> int:
        """Clear all conversations for a user and return count deleted."""
        try:
            query = f"DELETE FROM {self.config.table_name} WHERE user_id = $1"
            result = await self._execute_query(query, user_id)
            
            # Extract row count from result
            deleted_count = int(result.split()[-1]) if isinstance(result, str) and result.startswith("DELETE") else 0
            
            print(f"[MEMORY:Postgres] Cleared {deleted_count} conversations for user {user_id}")
            return deleted_count
            
        except Exception as e:
            print(f"[MEMORY:Postgres] Error clearing conversations for user {user_id}: {str(e)}")
            return 0
    
    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics."""
        try:
            base_query = f"""
            SELECT 
                COUNT(*) as total_conversations,
                MIN(created_at) as oldest_conversation,
                MAX(created_at) as newest_conversation,
                SUM(jsonb_array_length(messages)) as total_messages
            FROM {self.config.table_name}
            """
            
            params = []
            if user_id:
                base_query += " WHERE user_id = $1"
                params.append(user_id)
            
            row = await self._fetchrow_query(base_query, *params)
            
            return {
                "total_conversations": row['total_conversations'] or 0,
                "total_messages": row['total_messages'] or 0,
                "oldest_conversation": row['oldest_conversation'],
                "newest_conversation": row['newest_conversation']
            }
            
        except Exception as e:
            print(f"[MEMORY:Postgres] Error getting stats: {str(e)}")
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
            await self._ensure_schema()
            
            # Test basic connectivity
            test_query = "SELECT 1"
            await self._fetch_query(test_query)
            
            # Test table operations
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
                "provider": "PostgreSQL",
                "database": self.config.database,
                "table": self.config.table_name
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
            if hasattr(self.postgres_client, 'close'):
                await self.postgres_client.close()
            print("[MEMORY:Postgres] Closed PostgreSQL connection")
        except Exception as e:
            print(f"[MEMORY:Postgres] Error closing PostgreSQL connection: {str(e)}")

async def create_postgres_provider(config: PostgresConfig, postgres_client: Any) -> PostgresProvider:
    """
    Factory function to create a PostgreSQL provider instance.
    
    Args:
        config: Configuration for the PostgreSQL provider.
        postgres_client: Connected PostgreSQL client instance (connection or pool).
        
    Returns:
        Configured PostgresProvider instance.
        
    Raises:
        MemoryConnectionError: If PostgreSQL connection fails.
    """
    try:
        # Test connection
        if hasattr(postgres_client, 'execute'):
            # Direct connection
            await postgres_client.execute("SELECT 1")
        else:
            # Connection pool
            async with postgres_client.acquire() as conn:
                await conn.execute("SELECT 1")
        
        print(f"[MEMORY:Postgres] Successfully connected to PostgreSQL at {config.host}:{config.port}")
        
        provider = PostgresProvider(config, postgres_client)
        await provider._ensure_schema()  # Initialize schema immediately
        
        return provider
        
    except Exception as e:
        raise MemoryConnectionError(f"Failed to connect to PostgreSQL: {str(e)}", "PostgreSQL", e)