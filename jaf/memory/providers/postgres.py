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

try:
    import asyncpg
    PostgresClient = Union[asyncpg.Connection, asyncpg.Pool]
except ImportError:
    PostgresClient = Any

class PostgresProvider(MemoryProvider):
    """
    PostgreSQL implementation of MemoryProvider.
    """
    
    def __init__(self, config: PostgresConfig, client: PostgresClient):
        self.config = config
        self.client = client

    async def _db_fetch(self, query: str, *args):
        if hasattr(self.client, 'fetch'): # Pool
            return await self.client.fetch(query, *args)
        else: # Connection
            return await self.client.fetch(query, *args)

    async def _db_fetchrow(self, query: str, *args):
        if hasattr(self.client, 'fetchrow'):
            return await self.client.fetchrow(query, *args)
        else:
            return await self.client.fetchrow(query, *args)

    async def _db_execute(self, query: str, *args) -> str:
        if hasattr(self.client, 'execute'):
            return await self.client.execute(query, *args)
        else:
            return await self.client.execute(query, *args)

    def _row_to_conversation(self, row) -> ConversationMemory:
        return ConversationMemory(
            conversation_id=row['conversation_id'],
            user_id=row['user_id'],
            messages=[Message(**msg) for msg in json.loads(row['messages'])],
            metadata=json.loads(row['metadata'])
        )

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, MemoryStorageError]:
        try:
            now = datetime.now()
            metadata = metadata or {}
            metadata.update({"created_at": now.isoformat(), "updated_at": now.isoformat(), "last_activity": now.isoformat()})
            
            query = f"""
            INSERT INTO {self.config.table_name} (conversation_id, user_id, messages, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (conversation_id) DO UPDATE SET
            messages = EXCLUDED.messages, metadata = EXCLUDED.metadata;
            """
            await self._db_execute(
                query,
                conversation_id,
                metadata.get("user_id"),
                json.dumps([msg.dict() for msg in messages]),
                json.dumps(metadata)
            )
            return Success(None)
        except Exception as e:
            return Failure(MemoryStorageError(operation="store_messages", provider="Postgres", message=str(e), cause=e))

    async def get_conversation(
        self, 
        conversation_id: str
    ) -> Result[Optional[ConversationMemory], MemoryStorageError]:
        try:
            row = await self._db_fetchrow(f"SELECT * FROM {self.config.table_name} WHERE conversation_id = $1", conversation_id)
            if not row:
                return Success(None)
            
            # Update last activity
            await self._db_execute(f"UPDATE {self.config.table_name} SET metadata = metadata || '{{\"last_activity\": \"{datetime.now().isoformat()}\"}}' WHERE conversation_id = $1", conversation_id)
            
            return Success(self._row_to_conversation(row))
        except Exception as e:
            return Failure(MemoryStorageError(operation="get_conversation", provider="Postgres", message=str(e), cause=e))

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        # This is complex with JSONB, often easier to read-modify-write
        return Failure(MemoryStorageError(operation="append_messages", provider="Postgres", message="append_messages not efficiently supported, use get and store"))

    async def find_conversations(
        self, 
        query: MemoryQuery
    ) -> Result[List[ConversationMemory], MemoryStorageError]:
        try:
            rows = await self._db_fetch(f"SELECT * FROM {self.config.table_name}")
            return Success([self._row_to_conversation(row) for row in rows])
        except Exception as e:
            return Failure(MemoryStorageError(operation="find_conversations", provider="Postgres", message=str(e), cause=e))

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
            return Failure(MemoryNotFoundError(conversation_id=conversation_id, provider="Postgres", message=f"Conversation {conversation_id} not found"))
        
        return Success(conversation.messages[-limit:])

    async def delete_conversation(
        self, 
        conversation_id: str
    ) -> Result[bool, MemoryStorageError]:
        try:
            result = await self._db_execute(f"DELETE FROM {self.config.table_name} WHERE conversation_id = $1", conversation_id)
            return Success('DELETE 1' in result)
        except Exception as e:
            return Failure(MemoryStorageError(operation="delete_conversation", provider="Postgres", message=str(e), cause=e))

    async def clear_user_conversations(
        self, 
        user_id: str
    ) -> Result[int, MemoryStorageError]:
        try:
            result = await self._db_execute(f"DELETE FROM {self.config.table_name} WHERE user_id = $1", user_id)
            return Success(int(result.split(' ')[1]))
        except Exception as e:
            return Failure(MemoryStorageError(operation="clear_user_conversations", provider="Postgres", message=str(e), cause=e))

    async def get_stats(
        self, 
        user_id: Optional[str] = None
    ) -> Result[Dict[str, Any], MemoryStorageError]:
        try:
            row = await self._db_fetchrow(f"SELECT COUNT(*) as count FROM {self.config.table_name}")
            return Success({"total_conversations": row['count']})
        except Exception as e:
            return Failure(MemoryStorageError(operation="get_stats", provider="Postgres", message=str(e), cause=e))

    async def health_check(self) -> Result[Dict[str, Any], MemoryConnectionError]:
        try:
            await self._db_fetch("SELECT 1")
            return Success({"healthy": True})
        except Exception as e:
            return Failure(MemoryConnectionError(provider="Postgres", message="Postgres health check failed", cause=e))

    async def close(self) -> Result[None, MemoryConnectionError]:
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
            return Success(None)
        except Exception as e:
            return Failure(MemoryConnectionError(provider="Postgres", message="Failed to close Postgres connection", cause=e))

async def create_postgres_provider(config: PostgresConfig) -> Result[PostgresProvider, MemoryConnectionError]:
    try:
        client = await asyncpg.connect(
            dsn=config.connection_string,
            host=config.host,
            port=config.port,
            user=config.username,
            password=config.password,
            database=config.database
        )
        # Initialize schema
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {config.table_name} (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT,
            messages JSONB,
            metadata JSONB
        );
        """
        await client.execute(create_table_sql)
        return Success(PostgresProvider(config, client))
    except Exception as e:
        return Failure(MemoryConnectionError(provider="Postgres", message="Failed to connect to PostgreSQL", cause=e))
