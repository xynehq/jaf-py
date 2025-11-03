"""
PostgreSQL memory provider implementation.

This provider uses PostgreSQL for fully persistent conversation storage with JSONB columns.
Best for production environments requiring complex queries and full persistence.
"""

import json
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
    PostgresConfig,
    Result,
    Success,
)
from ..utils import prepare_message_list_for_db

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
        if hasattr(self.client, "fetch"):  # Pool
            return await self.client.fetch(query, *args)
        else:  # Connection
            return await self.client.fetch(query, *args)

    async def _db_fetchrow(self, query: str, *args):
        if hasattr(self.client, "fetchrow"):
            return await self.client.fetchrow(query, *args)
        else:
            return await self.client.fetchrow(query, *args)

    async def _db_execute(self, query: str, *args) -> str:
        if hasattr(self.client, "execute"):
            return await self.client.execute(query, *args)
        else:
            return await self.client.execute(query, *args)

    def _row_to_conversation(self, row) -> ConversationMemory:
        """Convert database row to ConversationMemory using shared utilities."""
        from ..utils import extract_messages_from_db_row, validate_conversation_metadata

        messages = extract_messages_from_db_row(row["messages"])
        metadata = validate_conversation_metadata(json.loads(row["metadata"]))

        return ConversationMemory(
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            messages=messages,
            metadata=metadata,
        )

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
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
            if "created_at" not in insert_metadata:
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
                prepare_message_list_for_db(messages),
                json.dumps(insert_metadata),
                json.dumps(update_metadata),
            )
            return Success(None)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="store_messages", provider="Postgres", message=str(e), cause=e
                )
            )

    async def get_conversation(
        self, conversation_id: str
    ) -> Result[Optional[ConversationMemory], MemoryStorageError]:
        try:
            row = await self._db_fetchrow(
                f"SELECT * FROM {self.config.table_name} WHERE conversation_id = $1",
                conversation_id,
            )
            if not row:
                return Success(None)

            # Update last activity
            timestamp = datetime.now().isoformat()
            await self._db_execute(
                f"UPDATE {self.config.table_name} SET metadata = metadata || $2 WHERE conversation_id = $1",
                conversation_id,
                f'{{"last_activity": "{timestamp}"}}',
            )

            return Success(self._row_to_conversation(row))
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="get_conversation", provider="Postgres", message=str(e), cause=e
                )
            )

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result[None, Union[MemoryNotFoundError, MemoryStorageError]]:
        try:
            # First, check if the conversation exists to provide a proper MemoryNotFoundError
            # A more optimized way might be to just run the UPDATE and check the result,
            # but this is clearer and aligns with the other providers.
            check_query = f"SELECT 1 FROM {self.config.table_name} WHERE conversation_id = $1"
            exists = await self._db_fetchrow(check_query, conversation_id)
            if not exists:
                return Failure(
                    MemoryNotFoundError(conversation_id=conversation_id, provider="Postgres")
                )

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

            new_messages_json = prepare_message_list_for_db(messages)

            await self._db_execute(
                query, new_messages_json, json.dumps(update_metadata), conversation_id
            )
            return Success(None)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="append_messages", provider="Postgres", message=str(e), cause=e
                )
            )

    async def find_conversations(
        self, query: MemoryQuery
    ) -> Result[List[ConversationMemory], MemoryStorageError]:
        try:
            rows = await self._db_fetch(f"SELECT * FROM {self.config.table_name}")
            return Success([self._row_to_conversation(row) for row in rows])
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="find_conversations", provider="Postgres", message=str(e), cause=e
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
                    provider="Postgres",
                    message=f"Conversation {conversation_id} not found",
                )
            )

        return Success(conversation.messages[-limit:])

    async def delete_conversation(self, conversation_id: str) -> Result[bool, MemoryStorageError]:
        try:
            result = await self._db_execute(
                f"DELETE FROM {self.config.table_name} WHERE conversation_id = $1", conversation_id
            )
            return Success("DELETE 1" in result)
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="delete_conversation", provider="Postgres", message=str(e), cause=e
                )
            )

    async def clear_user_conversations(self, user_id: str) -> Result[int, MemoryStorageError]:
        try:
            result = await self._db_execute(
                f"DELETE FROM {self.config.table_name} WHERE user_id = $1", user_id
            )
            return Success(int(result.split(" ")[1]))
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="clear_user_conversations",
                    provider="Postgres",
                    message=str(e),
                    cause=e,
                )
            )

    async def get_stats(
        self, user_id: Optional[str] = None
    ) -> Result[Dict[str, Any], MemoryStorageError]:
        try:
            row = await self._db_fetchrow(f"SELECT COUNT(*) as count FROM {self.config.table_name}")
            return Success({"total_conversations": row["count"]})
        except Exception as e:
            return Failure(
                MemoryStorageError(
                    operation="get_stats", provider="Postgres", message=str(e), cause=e
                )
            )

    async def health_check(self) -> Result[Dict[str, Any], MemoryConnectionError]:
        start_time = datetime.now()
        try:
            await self._db_fetch("SELECT 1")
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return Success({"healthy": True, "provider": "Postgres", "latency_ms": latency_ms})
        except Exception as e:
            return Failure(
                MemoryConnectionError(
                    provider="Postgres", message="Postgres health check failed", cause=e
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
                        provider="Postgres",
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

            # Convert any datetime objects in existing metadata to ISO strings
            serializable_metadata = {}
            for key, value in conversation.metadata.items():
                if isinstance(value, datetime):
                    serializable_metadata[key] = value.isoformat()
                else:
                    serializable_metadata[key] = value

            updated_metadata = {
                **serializable_metadata,
                "updated_at": now.isoformat(),
                "last_activity": now.isoformat(),
                "total_messages": len(truncated_messages),
                "regeneration_truncated": True,
                "truncated_at": now.isoformat(),
                "messages_removed": removed_count,
            }

            # Update in database
            query = f"""
            UPDATE {self.config.table_name}
            SET messages = $1::jsonb, metadata = $2::jsonb
            WHERE conversation_id = $3
            """

            await self._db_execute(
                query,
                prepare_message_list_for_db(truncated_messages),
                json.dumps(updated_metadata),
                conversation_id,
            )

            print(
                f"[MEMORY:Postgres] Truncated conversation {conversation_id}: removed {removed_count} messages after message {message_id}"
            )
            return Success(removed_count)

        except Exception as e:
            print(f"[MEMORY:Postgres] DEBUG: Exception in truncate_conversation_after: {e}")
            import traceback

            traceback.print_exc()
            return Failure(
                MemoryStorageError(
                    message=f"Failed to truncate conversation: {e}",
                    provider="Postgres",
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
                    f"[MEMORY:Postgres] Message {message_id} not found in conversation {conversation_id}"
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
                f"[MEMORY:Postgres] Retrieved conversation {conversation_id} until message {message_id}: {len(truncated_messages)} messages"
            )
            return Success(truncated_conversation)

        except Exception as e:
            return Failure(
                MemoryStorageError(
                    message=f"Failed to get conversation until message: {e}",
                    provider="Postgres",
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
                        provider="Postgres",
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
                "updated_at": datetime.now().isoformat(),
                "regeneration_count": len(regeneration_points),
            }

            # Update in database using JSONB merge
            query = f"""
            UPDATE {self.config.table_name}
            SET metadata = metadata || $1::jsonb
            WHERE conversation_id = $2
            """

            await self._db_execute(
                query,
                json.dumps(
                    {
                        "regeneration_points": regeneration_points,
                        "last_regeneration": regeneration_point,
                        "updated_at": updated_metadata["updated_at"],
                        "regeneration_count": len(regeneration_points),
                    }
                ),
                conversation_id,
            )

            print(
                f"[MEMORY:Postgres] Marked regeneration point for conversation {conversation_id} at message {message_id}"
            )
            return Success(None)

        except Exception as e:
            return Failure(
                MemoryStorageError(
                    message=f"Failed to mark regeneration point: {e}",
                    provider="Postgres",
                    operation="mark_regeneration_point",
                    cause=e,
                )
            )

    async def close(self) -> Result[None, MemoryConnectionError]:
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
            return Success(None)
        except Exception as e:
            return Failure(
                MemoryConnectionError(
                    provider="Postgres", message="Failed to close Postgres connection", cause=e
                )
            )


async def create_postgres_provider(
    config: PostgresConfig,
) -> Result[PostgresProvider, MemoryConnectionError]:
    try:
        # Connect to the default 'postgres' database to check if the target database exists
        try:
            conn = await asyncpg.connect(
                user=config.username,
                password=config.password,
                host=config.host,
                port=config.port,
                database="postgres",
            )
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", config.database
            )
            if not db_exists:
                await conn.execute(f'CREATE DATABASE "{config.database}"')
            await conn.close()
        except Exception as e:
            # This might fail if the user doesn't have permission to create databases,
            # but we can continue and hope the database already exists.
            print(f"Could not ensure database exists: {e}")

        # Now connect to the target database using connection pool with max_connections
        if config.connection_string:
            client = await asyncpg.create_pool(
                dsn=config.connection_string, min_size=1, max_size=config.max_connections
            )
        else:
            client = await asyncpg.create_pool(
                host=config.host,
                port=config.port,
                user=config.username,
                password=config.password,
                database=config.database,
                min_size=1,
                max_size=config.max_connections,
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
        return Failure(
            MemoryConnectionError(
                provider="Postgres", message="Failed to connect to PostgreSQL", cause=e
            )
        )
