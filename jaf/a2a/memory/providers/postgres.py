"""
A2A PostgreSQL Task Provider for JAF

This module provides a PostgreSQL-based storage implementation for A2A tasks.
It leverages PostgreSQL for reliable, ACID-compliant task storage with advanced
querying capabilities, full-text search, and robust data integrity.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...types import A2ATask, TaskState
from ..serialization import (
    A2ATaskSerialized,
    deserialize_a2a_task,
    sanitize_task,
    serialize_a2a_task,
)
from ..types import (
    A2APostgresTaskConfig,
    A2AResult,
    A2ATaskProvider,
    A2ATaskQuery,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_not_found_error,
    create_a2a_task_storage_error,
)

# SQL queries for A2A task operations
SQL_QUERIES = {
    "CREATE_TABLE": """
        CREATE TABLE IF NOT EXISTS {table_name} (
            task_id VARCHAR(255) PRIMARY KEY,
            context_id VARCHAR(255) NOT NULL,
            state VARCHAR(50) NOT NULL,
            task_data JSONB NOT NULL,
            status_message JSONB,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE,
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_context_id ON {table_name} (context_id);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_state ON {table_name} (state);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at ON {table_name} (created_at);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_expires_at ON {table_name} (expires_at) WHERE expires_at IS NOT NULL;
    """,
    "INSERT_TASK": """
        INSERT INTO {table_name} (
            task_id, context_id, state, task_data, status_message, 
            created_at, updated_at, expires_at, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """,
    "SELECT_TASK": """
        SELECT task_id, context_id, state, task_data, status_message, 
               created_at, updated_at, expires_at, metadata
        FROM {table_name} 
        WHERE task_id = $1 
          AND (expires_at IS NULL OR expires_at > NOW())
    """,
    "UPDATE_TASK": """
        UPDATE {table_name} 
        SET state = $2, task_data = $3, status_message = $4, 
            updated_at = $5, metadata = $6
        WHERE task_id = $1
          AND (expires_at IS NULL OR expires_at > NOW())
    """,
    "DELETE_TASK": """
        DELETE FROM {table_name} 
        WHERE task_id = $1
    """,
    "DELETE_TASKS_BY_CONTEXT": """
        DELETE FROM {table_name} 
        WHERE context_id = $1
    """,
    "CLEANUP_EXPIRED": """
        DELETE FROM {table_name} 
        WHERE expires_at IS NOT NULL AND expires_at <= NOW()
    """,
    "COUNT_TASKS": """
        SELECT COUNT(*) as total 
        FROM {table_name} 
        WHERE (expires_at IS NULL OR expires_at > NOW())
    """,
    "STATS_BY_STATE": """
        SELECT state, COUNT(*) as count 
        FROM {table_name} 
        WHERE (expires_at IS NULL OR expires_at > NOW())
          AND ($1::text IS NULL OR context_id = $1)
        GROUP BY state
    """,
    "DATE_RANGE": """
        SELECT MIN(created_at) as oldest, MAX(created_at) as newest
        FROM {table_name} 
        WHERE (expires_at IS NULL OR expires_at > NOW())
          AND ($1::text IS NULL OR context_id = $1)
    """,
}


async def create_a2a_postgres_task_provider(
    config: A2APostgresTaskConfig, postgres_client: Any
) -> A2AResult[A2ATaskProvider]:
    """
    Create a PostgreSQL-based A2A task provider

    Args:
        config: Configuration for the PostgreSQL provider
        postgres_client: PostgreSQL client instance (asyncpg connection/pool)

    Returns:
        A2AResult containing the task provider or an error
    """
    try:
        table_name = config.table_name or "a2a_tasks"

        # Initialize database schema
        create_table_sql = SQL_QUERIES["CREATE_TABLE"].format(table_name=table_name)
        await postgres_client.execute(create_table_sql)

        def row_to_serialized_task(row: Any) -> A2ATaskSerialized:
            """Convert database row to serialized task"""
            return A2ATaskSerialized(
                task_id=row["task_id"],
                context_id=row["context_id"],
                state=row["state"],
                task_data=row["task_data"]
                if isinstance(row["task_data"], str)
                else json.dumps(row["task_data"]),
                status_message=row["status_message"]
                if isinstance(row["status_message"], str)
                else json.dumps(row["status_message"])
                if row["status_message"]
                else None,
                created_at=row["created_at"].isoformat(),
                updated_at=row["updated_at"].isoformat(),
                metadata=row["metadata"]
                if isinstance(row["metadata"], str)
                else json.dumps(row["metadata"])
                if row["metadata"]
                else None,
            )

        def build_where_clause(query: A2ATaskQuery) -> tuple[str, list]:
            """Build WHERE clause for queries"""
            conditions = ["(expires_at IS NULL OR expires_at > NOW())"]
            params = []
            param_index = 1

            if query.task_id:
                conditions.append(f"task_id = ${param_index}")
                params.append(query.task_id)
                param_index += 1

            if query.context_id:
                conditions.append(f"context_id = ${param_index}")
                params.append(query.context_id)
                param_index += 1

            if query.state:
                conditions.append(f"state = ${param_index}")
                params.append(query.state.value)
                param_index += 1

            if query.since:
                # Use metadata created_at if available, otherwise use database created_at
                conditions.append(f"""(
                    CASE 
                        WHEN metadata ? 'created_at' THEN 
                            (metadata->>'created_at')::timestamp with time zone >= ${param_index}
                        ELSE 
                            created_at >= ${param_index}
                    END
                )""")
                params.append(query.since)
                param_index += 1

            if query.until:
                # Use metadata created_at if available, otherwise use database created_at
                conditions.append(f"""(
                    CASE 
                        WHEN metadata ? 'created_at' THEN 
                            (metadata->>'created_at')::timestamp with time zone <= ${param_index}
                        ELSE 
                            created_at <= ${param_index}
                    END
                )""")
                params.append(query.until)
                param_index += 1

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            return where_clause, params

        class PostgresA2ATaskProvider:
            """PostgreSQL implementation of A2ATaskProvider"""

            async def store_task(
                self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None
            ) -> A2AResult[None]:
                """Store a new A2A task in PostgreSQL"""
                try:
                    # Validate and sanitize task
                    sanitize_result = sanitize_task(task)
                    if not isinstance(sanitize_result.data, A2ATask):
                        return sanitize_result

                    # Serialize task
                    serialize_result = serialize_a2a_task(sanitize_result.data, metadata)
                    if not isinstance(serialize_result.data, A2ATaskSerialized):
                        return serialize_result

                    serialized = serialize_result.data
                    query = SQL_QUERIES["INSERT_TASK"].format(table_name=table_name)

                    await postgres_client.execute(
                        query,
                        serialized.task_id,
                        serialized.context_id,
                        serialized.state,
                        serialized.task_data,
                        serialized.status_message,
                        datetime.fromisoformat(serialized.created_at),
                        datetime.fromisoformat(serialized.updated_at),
                        metadata.get("expires_at") if metadata else None,
                        json.dumps(metadata) if metadata else None,
                    )

                    return create_a2a_success(None)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("store", "postgres", task.id, error)
                    )

            async def get_task(self, task_id: str) -> A2AResult[Optional[A2ATask]]:
                """Retrieve a task by ID from PostgreSQL"""
                try:
                    query = SQL_QUERIES["SELECT_TASK"].format(table_name=table_name)
                    row = await postgres_client.fetchrow(query, task_id)

                    if not row:
                        return create_a2a_success(None)

                    serialized = row_to_serialized_task(row)
                    deserialize_result = deserialize_a2a_task(serialized)

                    if isinstance(deserialize_result.data, A2ATask):
                        return create_a2a_success(deserialize_result.data)
                    else:
                        return deserialize_result

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("get", "postgres", task_id, error)
                    )

            async def update_task(
                self, task: A2ATask, metadata: Optional[Dict[str, Any]] = None
            ) -> A2AResult[None]:
                """Update an existing task in PostgreSQL"""
                try:
                    # Check if task exists
                    existing_result = await self.get_task(task.id)
                    if not isinstance(existing_result.data, A2ATask):
                        return existing_result

                    if not existing_result.data:
                        return create_a2a_failure(
                            create_a2a_task_not_found_error(task.id, "postgres")
                        )

                    # Validate and sanitize task
                    sanitize_result = sanitize_task(task)
                    if not isinstance(sanitize_result.data, A2ATask):
                        return sanitize_result

                    # Get existing metadata
                    existing_query = SQL_QUERIES["SELECT_TASK"].format(table_name=table_name)
                    existing_data = await postgres_client.fetchrow(existing_query, task.id)
                    existing_metadata = (
                        existing_data["metadata"]
                        if existing_data and existing_data["metadata"]
                        else {}
                    )
                    if isinstance(existing_metadata, str):
                        existing_metadata = json.loads(existing_metadata)

                    merged_metadata = {**existing_metadata, **(metadata or {})}

                    # Serialize updated task
                    serialize_result = serialize_a2a_task(sanitize_result.data, merged_metadata)
                    if not isinstance(serialize_result.data, A2ATaskSerialized):
                        return serialize_result

                    serialized = serialize_result.data
                    query = SQL_QUERIES["UPDATE_TASK"].format(table_name=table_name)

                    result = await postgres_client.execute(
                        query,
                        task.id,
                        serialized.state,
                        serialized.task_data,
                        serialized.status_message,
                        datetime.fromisoformat(serialized.updated_at),
                        json.dumps(merged_metadata),
                    )

                    # Check if any rows were affected
                    if result == "UPDATE 0":
                        return create_a2a_failure(
                            create_a2a_task_not_found_error(task.id, "postgres")
                        )

                    return create_a2a_success(None)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("update", "postgres", task.id, error)
                    )

            async def update_task_status(
                self,
                task_id: str,
                state: TaskState,
                status_message: Optional[Any] = None,
                timestamp: Optional[str] = None,
            ) -> A2AResult[None]:
                """Update task status only"""
                try:
                    # Get existing task
                    existing_result = await self.get_task(task_id)
                    if not isinstance(existing_result.data, A2ATask):
                        return existing_result

                    if not existing_result.data:
                        return create_a2a_failure(
                            create_a2a_task_not_found_error(task_id, "postgres")
                        )

                    task = existing_result.data

                    # Build updated history - add current status message to history before updating
                    updated_history = list(task.history or [])
                    if task.status.message:
                        updated_history.append(task.status.message)

                    # Update task status
                    from ...types import A2ATaskStatus

                    updated_status = A2ATaskStatus(
                        state=state,
                        message=status_message or task.status.message,
                        timestamp=timestamp or datetime.now().isoformat(),
                    )

                    updated_task = task.model_copy(
                        update={"status": updated_status, "history": updated_history}
                    )

                    return await self.update_task(updated_task)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("update-status", "postgres", task_id, error)
                    )

            async def find_tasks(self, query: A2ATaskQuery) -> A2AResult[List[A2ATask]]:
                """Search tasks by query parameters"""
                try:
                    where_clause, params = build_where_clause(query)

                    sql = f"""
                        SELECT task_id, context_id, state, task_data, status_message, 
                               created_at, updated_at, expires_at, metadata
                        FROM {table_name} 
                        {where_clause}
                        ORDER BY created_at DESC
                    """

                    # Add pagination
                    if query.limit:
                        sql += f" LIMIT {query.limit}"
                    if query.offset:
                        sql += f" OFFSET {query.offset}"

                    rows = await postgres_client.fetch(sql, *params)
                    tasks: List[A2ATask] = []

                    for row in rows:
                        serialized = row_to_serialized_task(row)
                        deserialize_result = deserialize_a2a_task(serialized)

                        if isinstance(deserialize_result.data, A2ATask):
                            tasks.append(deserialize_result.data)

                    return create_a2a_success(tasks)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("find", "postgres", None, error)
                    )

            async def get_tasks_by_context(
                self, context_id: str, limit: Optional[int] = None
            ) -> A2AResult[List[A2ATask]]:
                """Get tasks by context ID"""
                return await self.find_tasks(A2ATaskQuery(context_id=context_id, limit=limit))

            async def delete_task(self, task_id: str) -> A2AResult[bool]:
                """Delete a task and return True if it existed"""
                try:
                    query = SQL_QUERIES["DELETE_TASK"].format(table_name=table_name)
                    result = await postgres_client.execute(query, task_id)

                    # Check if any rows were affected
                    rows_affected = int(result.split()[-1]) if "DELETE" in result else 0
                    return create_a2a_success(rows_affected > 0)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("delete", "postgres", task_id, error)
                    )

            async def delete_tasks_by_context(self, context_id: str) -> A2AResult[int]:
                """Delete tasks by context ID and return count deleted"""
                try:
                    query = SQL_QUERIES["DELETE_TASKS_BY_CONTEXT"].format(table_name=table_name)
                    result = await postgres_client.execute(query, context_id)

                    # Extract count from result
                    rows_affected = int(result.split()[-1]) if "DELETE" in result else 0
                    return create_a2a_success(rows_affected)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("delete-by-context", "postgres", None, error)
                    )

            async def cleanup_expired_tasks(self) -> A2AResult[int]:
                """Clean up expired tasks and return count deleted"""
                try:
                    query = SQL_QUERIES["CLEANUP_EXPIRED"].format(table_name=table_name)
                    result = await postgres_client.execute(query)

                    # Extract count from result
                    rows_affected = int(result.split()[-1]) if "DELETE" in result else 0
                    return create_a2a_success(rows_affected)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("cleanup", "postgres", None, error)
                    )

            async def get_task_stats(
                self, context_id: Optional[str] = None
            ) -> A2AResult[Dict[str, Any]]:
                """Get task statistics"""
                try:
                    tasks_by_state = {state.value: 0 for state in TaskState}

                    # Get state counts
                    state_query = SQL_QUERIES["STATS_BY_STATE"].format(table_name=table_name)
                    state_rows = await postgres_client.fetch(state_query, context_id)

                    total_tasks = 0
                    for row in state_rows:
                        state = row["state"]
                        count = int(row["count"])
                        if state in tasks_by_state:
                            tasks_by_state[state] = count
                        total_tasks += count

                    # Get date range
                    date_query = SQL_QUERIES["DATE_RANGE"].format(table_name=table_name)
                    date_row = await postgres_client.fetchrow(date_query, context_id)

                    oldest_task = None
                    newest_task = None

                    if date_row and date_row["oldest"]:
                        oldest_task = date_row["oldest"]
                        newest_task = date_row["newest"]

                    # Build stats dict with individual state counts as top-level keys
                    stats = {
                        "total_tasks": total_tasks,
                        "tasks_by_state": tasks_by_state,
                        "oldest_task": oldest_task,
                        "newest_task": newest_task,
                    }

                    # Also add individual state counts for backwards compatibility
                    for state in TaskState:
                        stats[state.value] = tasks_by_state[state.value]

                    return create_a2a_success(stats)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("stats", "postgres", None, error)
                    )

            async def health_check(self) -> A2AResult[Dict[str, Any]]:
                """Check provider health and return status information"""
                try:
                    start_time = datetime.now()

                    # Simple query to check database connectivity
                    await postgres_client.fetchval("SELECT 1")

                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return create_a2a_success(
                        {"healthy": True, "provider": "postgres", "latency_ms": latency_ms}
                    )

                except Exception as error:
                    return create_a2a_success({"healthy": False, "error": str(error)})

            async def close(self) -> A2AResult[None]:
                """Close/cleanup the provider"""
                try:
                    # PostgreSQL client cleanup is typically handled externally
                    # We don't close the client here as it might be a pool or shared connection
                    return create_a2a_success(None)

                except Exception as error:
                    return create_a2a_failure(
                        create_a2a_task_storage_error("close", "postgres", None, error)
                    )

        return create_a2a_success(PostgresA2ATaskProvider())

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error("create-postgres-provider", "postgres", None, error)
        )
