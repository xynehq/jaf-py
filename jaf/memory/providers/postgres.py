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
from ...core.types import Message, ToolCall, ToolCallFunction
from dataclasses import asdict

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
        messages_data = json.loads(row['messages'])
        messages = []
        for msg_data in messages_data:
            tool_calls = None
            if msg_data.get('tool_calls'):
                tool_calls = [
                    ToolCall(
                        id=tc['id'],
                        type=tc['type'],
                        function=ToolCallFunction(
                            name=tc['function']['name'],
                            arguments=tc['function']['arguments']
                        )
                    ) for tc in msg_data['tool_calls']
                ]
            messages.append(Message(
                role=msg_data['role'],
                content=msg_data['content'],
                tool_call_id=msg_data.get('tool_call_id'),
                tool_calls=tool_calls
            ))

        return ConversationMemory(
            conversation_id=row['conversation_id'],
            user_id=row['user_id'],
            messages=messages,
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
            current_metadata = metadata or {}

            # Prepare metadata for insertion/update
            update_metadata = current_metadata.copy()
            update_metadata["updated_at"] = now.isoformat()
            update_metadata["last_activity"] = now.isoformat()

            # For new rows, we also need created_at
            insert_metadata = update_metadata.copy()
            if not "created_at" in insert_metadata:
                insert_metadata["created_at"] = now.isoformat()

            # Using INSERT ON CONFLICT for a single, atomic operation
            query = f"""
            INSERT INTO {self.config.table_name} (conversation_id, user_id, messages, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (conversation_id) DO UPDATE SET
                messages = EXCLUDED.messages,
                metadata = {self.config.table_name}.metadata || $5::jsonb;
            """
            
            await self._db_execute(
                query,
                conversation_id,
                current_metadata.get("user_id"),
                json.dumps([asdict(msg) for msg in messages]),
                json.dumps(insert_metadata),
                json.dumps(update_metadata)
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
        try:
            # First, check if the conversation exists to provide a proper MemoryNotFoundError
            # A more optimized way might be to just run the UPDATE and check the result,
            # but this is clearer and aligns with the other providers.
            check_query = f"SELECT 1 FROM {self.config.table_name} WHERE conversation_id = $1"
            exists = await self._db_fetchrow(check_query, conversation_id)
            if not exists:
                return Failure(MemoryNotFoundError(conversation_id=conversation_id, provider="Postgres"))

            now = datetime.now()
            update_metadata = metadata or {}
            update_metadata["updated_at"] = now.isoformat()
            update_metadata["last_activity"] = now.isoformat()

            # This query appends new messages to the existing JSONB array `messages`
            # and merges new metadata into the existing `metadata` JSONB object.
            query = f"""
            UPDATE {self.config.table_name}
            SET
                messages = messages || $1::jsonb,
                metadata = metadata || $2::jsonb
            WHERE conversation_id = $3;
            """

            new_messages_json = json.dumps([asdict(msg) for msg in messages])
            
            await self._db_execute(
                query,
                new_messages_json,
                json.dumps(update_metadata),
                conversation_id
            )
            return Success(None)
        except Exception as e:
            return Failure(MemoryStorageError(operation="append_messages", provider="Postgres", message=str(e), cause=e))

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
        start_time = datetime.now()
        try:
            await self._db_fetch("SELECT 1")
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return Success({
                "healthy": True,
                "provider": "Postgres",
                "latency_ms": latency_ms
            })
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
        # Connect to the default 'postgres' database to check if the target database exists
        try:
            conn = await asyncpg.connect(user=config.username, password=config.password, host=config.host, port=config.port, database='postgres')
            db_exists = await conn.fetchval(f"SELECT 1 FROM pg_database WHERE datname = '{config.database}'")
            if not db_exists:
                await conn.execute(f'CREATE DATABASE "{config.database}"')
            await conn.close()
        except Exception as e:
            # This might fail if the user doesn't have permission to create databases,
            # but we can continue and hope the database already exists.
            print(f"Could not ensure database exists: {e}")

        # Now connect to the target database
        if config.connection_string:
            client = await asyncpg.connect(dsn=config.connection_string)
        else:
            client = await asyncpg.connect(
                host=config.host,
                port=config.port,
                user=config.username,
                password=config.password,
                database=config.database
            )
        
        table_name = config.table_name or "conversations"
        
        # Initialize schema
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT,
            messages JSONB,
            metadata JSONB
        );
        """
        await client.execute(create_table_sql)
        
        # We need to update the config with the table name we are using
        # to ensure the provider uses it.
        provider_config = config
        if not provider_config.table_name:
            from dataclasses import replace
            provider_config = replace(config, table_name=table_name)

        return Success(PostgresProvider(provider_config, client))
    except Exception as e:
        return Failure(MemoryConnectionError(provider="Postgres", message="Failed to connect to PostgreSQL", cause=e))
