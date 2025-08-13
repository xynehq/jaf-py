"""
ADK PostgreSQL Session Provider - Production-Ready PostgreSQL Session Storage

This module provides a production-grade PostgreSQL session provider with
automatic schema creation, transactions, connection pooling, and JSONB storage.
"""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import AdkSessionProvider, AdkSessionConfig
from ..types import AdkSession, AdkMessage, AdkResult, AdkSuccess, AdkFailure
from ..errors import AdkSessionError, AdkErrorType, AdkErrorSeverity

try:
    import asyncpg
    from asyncpg import Pool as AsyncPGPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    # Create dummy types for type annotations when asyncpg is not available
    class _DummyAsyncPG:
        class Connection:
            pass
        class Pool:
            pass
    
    asyncpg = _DummyAsyncPG()
    AsyncPGPool = _DummyAsyncPG.Pool

@dataclass
class AdkPostgresSessionConfig(AdkSessionConfig):
    """PostgreSQL-specific session configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "adk_sessions"
    user: str = "postgres"
    password: Optional[str] = None
    url: Optional[str] = None
    pool_min_size: int = 2
    pool_max_size: int = 10
    table_name: str = "adk_sessions"
    auto_create_schema: bool = True
    command_timeout: float = 60.0

class AdkPostgresSessionProvider(AdkSessionProvider):
    """
    Production-ready PostgreSQL session provider.
    
    Features:
    - Connection pooling for high performance
    - JSONB storage for efficient querying
    - Automatic schema creation
    - Transaction support for atomicity
    - TTL support with background cleanup
    """
    
    def __init__(self, config: AdkPostgresSessionConfig, connection_pool: Optional[AsyncPGPool] = None):
        super().__init__(config)
        self.config: AdkPostgresSessionConfig = config
        self.connection_pool = connection_pool
        
        if not POSTGRES_AVAILABLE:
            raise AdkSessionError(
                "PostgreSQL not available. Install with: pip install asyncpg",
                error_type=AdkErrorType.VALIDATION,
                severity=AdkErrorSeverity.CRITICAL
            )
    
    async def _ensure_schema(self, connection: asyncpg.Connection):
        """Ensure the database schema exists."""
        if not self.config.auto_create_schema:
            return
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.table_name} (
            session_id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            app_name VARCHAR(255) NOT NULL,
            session_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            metadata JSONB DEFAULT '{{}}'::JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_user_id 
        ON {self.config.table_name} (user_id);
        
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_app_name 
        ON {self.config.table_name} (app_name);
        
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_expires_at 
        ON {self.config.table_name} (expires_at);
        
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_updated_at 
        ON {self.config.table_name} (updated_at);
        """
        
        await connection.execute(create_table_sql)
    
    def _serialize_messages(self, messages: List[AdkMessage]) -> List[Dict[str, Any]]:
        """Serialize messages to JSON-compatible format."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "tool_calls": msg.tool_calls,
                "tool_call_id": msg.tool_call_id,
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    def _deserialize_messages(self, messages_data: List[Dict[str, Any]]) -> List[AdkMessage]:
        """Deserialize messages from JSON format."""
        messages = []
        for msg_data in messages_data:
            timestamp = None
            if msg_data.get("timestamp"):
                timestamp = datetime.fromisoformat(msg_data["timestamp"])
            
            messages.append(AdkMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=timestamp,
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
                metadata=msg_data.get("metadata")
            ))
        
        return messages
    
    def _calculate_expiry(self) -> datetime:
        """Calculate expiry timestamp."""
        from datetime import timedelta
        return datetime.now(timezone.utc) + timedelta(seconds=self.config.ttl_seconds)
    
    async def create_session(
        self,
        user_id: str,
        app_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """Create a new session with transaction support."""
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            now = datetime.now(timezone.utc)
            expires_at = self._calculate_expiry()
            
            session = AdkSession(
                session_id=session_id,
                user_id=user_id,
                app_name=app_name,
                messages=[],
                created_at=now,
                updated_at=now,
                metadata=metadata
            )
            
            session_data = {
                "messages": self._serialize_messages(session.messages)
            }
            
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                # Use transaction for atomicity
                async with connection.transaction():
                    # Enforce per-user session limit
                    user_session_count = await connection.fetchval(
                        f"SELECT COUNT(*) FROM {self.config.table_name} WHERE user_id = $1",
                        user_id
                    )
                    
                    if user_session_count >= self.config.max_sessions_per_user:
                        # Remove oldest session for user
                        await connection.execute(f"""
                            DELETE FROM {self.config.table_name} 
                            WHERE session_id = (
                                SELECT session_id FROM {self.config.table_name} 
                                WHERE user_id = $1 
                                ORDER BY updated_at ASC 
                                LIMIT 1
                            )
                        """, user_id)
                    
                    # Insert new session
                    await connection.execute(f"""
                        INSERT INTO {self.config.table_name} 
                        (session_id, user_id, app_name, session_data, created_at, updated_at, expires_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, session_id, user_id, app_name, json.dumps(session_data), 
                    now, now, expires_at, json.dumps(metadata or {}))
            
            return AdkSuccess(session)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to create session: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def get_session(
        self,
        session_id: str
    ) -> AdkResult[Optional[AdkSession], AdkSessionError]:
        """Get session from PostgreSQL."""
        try:
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                # Get session and update expiry
                async with connection.transaction():
                    row = await connection.fetchrow(f"""
                        SELECT session_id, user_id, app_name, session_data, 
                               created_at, updated_at, metadata
                        FROM {self.config.table_name}
                        WHERE session_id = $1 AND expires_at > NOW()
                    """, session_id)
                    
                    if not row:
                        return AdkSuccess(None)
                    
                    # Update expiry on access
                    new_expiry = self._calculate_expiry()
                    await connection.execute(f"""
                        UPDATE {self.config.table_name} 
                        SET expires_at = $1 
                        WHERE session_id = $2
                    """, new_expiry, session_id)
                
                # Deserialize session
                session_data = row['session_data']
                messages = self._deserialize_messages(session_data.get('messages', []))
                
                session = AdkSession(
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    app_name=row['app_name'],
                    messages=messages,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=row['metadata']
                )
                
                return AdkSuccess(session)
                
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to get session: {e}",
                session_id=session_id,
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def update_session(
        self,
        session: AdkSession
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """Update session in PostgreSQL."""
        try:
            session.updated_at = datetime.now(timezone.utc)
            expires_at = self._calculate_expiry()
            
            session_data = {
                "messages": self._serialize_messages(session.messages)
            }
            
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                result = await connection.execute(f"""
                    UPDATE {self.config.table_name} 
                    SET session_data = $1, updated_at = $2, expires_at = $3, metadata = $4
                    WHERE session_id = $5
                """, json.dumps(session_data), session.updated_at, expires_at, 
                json.dumps(session.metadata or {}), session.session_id)
                
                if result == "UPDATE 0":
                    return AdkFailure(AdkSessionError(
                        f"Session not found: {session.session_id}",
                        session_id=session.session_id,
                        error_type=AdkErrorType.VALIDATION
                    ))
            
            return AdkSuccess(session)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to update session: {e}",
                session_id=session.session_id,
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def delete_session(
        self,
        session_id: str
    ) -> AdkResult[bool, AdkSessionError]:
        """Delete session from PostgreSQL."""
        try:
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                result = await connection.execute(f"""
                    DELETE FROM {self.config.table_name} 
                    WHERE session_id = $1
                """, session_id)
                
                return AdkSuccess("DELETE 1" in result)
                
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to delete session: {e}",
                session_id=session_id,
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def add_message(
        self,
        session_id: str,
        message: AdkMessage
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """Add message to session."""
        try:
            # Get current session
            session_result = await self.get_session(session_id)
            if isinstance(session_result, AdkFailure):
                return session_result
            
            session = session_result.data
            if not session:
                return AdkFailure(AdkSessionError(
                    f"Session not found: {session_id}",
                    session_id=session_id,
                    error_type=AdkErrorType.VALIDATION
                ))
            
            # Add message and enforce limit
            session.add_message(message)
            
            if len(session.messages) > self.config.max_messages_per_session:
                session.messages = session.messages[-self.config.max_messages_per_session:]
            
            return await self.update_session(session)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to add message: {e}",
                session_id=session_id,
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> AdkResult[List[AdkMessage], AdkSessionError]:
        """Get messages from session."""
        try:
            session_result = await self.get_session(session_id)
            if isinstance(session_result, AdkFailure):
                return session_result
            
            session = session_result.data
            if not session:
                return AdkSuccess([])
            
            messages = session.messages[offset:]
            if limit is not None:
                messages = messages[:limit]
            
            return AdkSuccess(messages)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to get messages: {e}",
                session_id=session_id,
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def get_user_sessions(
        self,
        user_id: str,
        app_name: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> AdkResult[List[AdkSession], AdkSessionError]:
        """Get sessions for a user."""
        try:
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                # Build query with optional app filter
                where_clause = "WHERE user_id = $1 AND expires_at > NOW()"
                params = [user_id]
                
                if app_name:
                    where_clause += " AND app_name = $2"
                    params.append(app_name)
                    limit_offset_params = [limit, offset]
                else:
                    limit_offset_params = [limit, offset]
                
                query = f"""
                    SELECT session_id, user_id, app_name, session_data, 
                           created_at, updated_at, metadata
                    FROM {self.config.table_name}
                    {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """
                
                rows = await connection.fetch(query, *params, *limit_offset_params)
                
                sessions = []
                for row in rows:
                    session_data = row['session_data']
                    messages = self._deserialize_messages(session_data.get('messages', []))
                    
                    session = AdkSession(
                        session_id=row['session_id'],
                        user_id=row['user_id'],
                        app_name=row['app_name'],
                        messages=messages,
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=row['metadata']
                    )
                    sessions.append(session)
                
                return AdkSuccess(sessions)
                
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to get user sessions: {e}",
                user_id=user_id,
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def cleanup_expired_sessions(self) -> AdkResult[int, AdkSessionError]:
        """Clean up expired sessions."""
        try:
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                result = await connection.execute(f"""
                    DELETE FROM {self.config.table_name} 
                    WHERE expires_at <= NOW()
                """)
                
                # Extract count from result string like "DELETE 5"
                count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                
                return AdkSuccess(count)
                
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to cleanup expired sessions: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def get_stats(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """Get provider statistics."""
        try:
            async with self.connection_pool.acquire() as connection:
                await self._ensure_schema(connection)
                
                # Get various statistics
                total_sessions = await connection.fetchval(f"""
                    SELECT COUNT(*) FROM {self.config.table_name}
                    WHERE expires_at > NOW()
                """)
                
                total_users = await connection.fetchval(f"""
                    SELECT COUNT(DISTINCT user_id) FROM {self.config.table_name}
                    WHERE expires_at > NOW()
                """)
                
                avg_messages = await connection.fetchval(f"""
                    SELECT AVG(jsonb_array_length(session_data->'messages')) 
                    FROM {self.config.table_name}
                    WHERE expires_at > NOW()
                """) or 0
                
                stats = {
                    "provider": "postgresql",
                    "total_sessions": total_sessions,
                    "total_users": total_users,
                    "average_messages_per_session": float(avg_messages),
                    "table_name": self.config.table_name
                }
                
                return AdkSuccess(stats)
                
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to get stats: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def health_check(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """Perform health check."""
        try:
            start_time = datetime.now()
            
            async with self.connection_pool.acquire() as connection:
                # Test basic connectivity
                await connection.fetchval("SELECT 1")
                
                # Test table access
                await connection.fetchval(f"""
                    SELECT COUNT(*) FROM {self.config.table_name}
                    WHERE expires_at > NOW()
                """)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            health = {
                "healthy": True,
                "provider": "postgresql",
                "latency_ms": latency,
                "pool_size": self.connection_pool.get_size(),
                "pool_min_size": self.config.pool_min_size,
                "pool_max_size": self.config.pool_max_size,
                "database": self.config.database,
                "table_name": self.config.table_name
            }
            
            return AdkSuccess(health)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Health check failed: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def close(self) -> AdkResult[None, AdkSessionError]:
        """Close PostgreSQL connection pool."""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
            return AdkSuccess(None)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to close connection pool: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))

async def create_postgres_session_provider(
    config: AdkPostgresSessionConfig
) -> AdkResult[AdkPostgresSessionProvider, AdkSessionError]:
    """
    Create a PostgreSQL session provider with connection pooling.
    
    Args:
        config: PostgreSQL session configuration
        
    Returns:
        Result containing the provider or error
    """
    try:
        if not POSTGRES_AVAILABLE:
            return AdkFailure(AdkSessionError(
                "PostgreSQL not available. Install with: pip install asyncpg",
                error_type=AdkErrorType.VALIDATION,
                severity=AdkErrorSeverity.CRITICAL
            ))
        
        # Create connection pool
        if config.url:
            connection_pool = await asyncpg.create_pool(
                config.url,
                min_size=config.pool_min_size,
                max_size=config.pool_max_size,
                command_timeout=config.command_timeout
            )
        else:
            connection_pool = await asyncpg.create_pool(
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.user,
                password=config.password,
                min_size=config.pool_min_size,
                max_size=config.pool_max_size,
                command_timeout=config.command_timeout
            )
        
        # Test connection
        async with connection_pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
        
        provider = AdkPostgresSessionProvider(config, connection_pool)
        return AdkSuccess(provider)
        
    except Exception as e:
        return AdkFailure(AdkSessionError(
            f"Failed to create PostgreSQL session provider: {e}",
            error_type=AdkErrorType.INTERNAL,
            cause=e
        ))