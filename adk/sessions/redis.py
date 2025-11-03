"""
ADK Redis Session Provider - Production-Ready Redis Session Storage

This module provides a production-grade Redis session provider with
connection pooling, atomic operations, TTL support, and comprehensive error handling.
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
    import redis.asyncio as redis
    from redis.asyncio import Redis as RedisClient

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisClient = None


@dataclass
class AdkRedisSessionConfig(AdkSessionConfig):
    """Redis-specific session configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    url: Optional[str] = None
    connection_pool_size: int = 10
    key_prefix: str = "adk:session:"
    user_sessions_key_prefix: str = "adk:user_sessions:"
    enable_compression: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0


class AdkRedisSessionProvider(AdkSessionProvider):
    """
    Production-ready Redis session provider.

    Features:
    - Connection pooling for high performance
    - Atomic operations using Redis transactions
    - TTL support for automatic cleanup
    - Compression for large sessions
    - Comprehensive error handling
    """

    def __init__(self, config: AdkRedisSessionConfig, redis_client: Optional[RedisClient] = None):
        super().__init__(config)
        self.config: AdkRedisSessionConfig = config
        self.redis_client = redis_client

        if not REDIS_AVAILABLE:
            raise AdkSessionError(
                "Redis not available. Install with: pip install redis",
                error_type=AdkErrorType.VALIDATION,
                severity=AdkErrorSeverity.CRITICAL,
            )

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"{self.config.key_prefix}{session_id}"

    def _get_user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user sessions set."""
        return f"{self.config.user_sessions_key_prefix}{user_id}"

    def _serialize_session(self, session: AdkSession) -> bytes:
        """Serialize session to bytes."""
        data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "app_name": session.app_name,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "metadata": msg.metadata,
                }
                for msg in session.messages
            ],
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
        }

        json_str = json.dumps(data, separators=(",", ":"))

        if self.config.enable_compression:
            import gzip

            return gzip.compress(json_str.encode())

        return json_str.encode()

    def _deserialize_session(self, data: bytes) -> AdkSession:
        """Deserialize session from bytes."""
        try:
            if self.config.enable_compression:
                import gzip

                data = gzip.decompress(data)

            session_data = json.loads(data.decode())

            messages = []
            for msg_data in session_data.get("messages", []):
                timestamp = None
                if msg_data.get("timestamp"):
                    timestamp = datetime.fromisoformat(msg_data["timestamp"])

                messages.append(
                    AdkMessage(
                        role=msg_data["role"],
                        content=msg_data["content"],
                        timestamp=timestamp,
                        tool_calls=msg_data.get("tool_calls"),
                        tool_call_id=msg_data.get("tool_call_id"),
                        metadata=msg_data.get("metadata"),
                    )
                )

            return AdkSession(
                session_id=session_data["session_id"],
                user_id=session_data["user_id"],
                app_name=session_data["app_name"],
                messages=messages,
                created_at=datetime.fromisoformat(session_data["created_at"]),
                updated_at=datetime.fromisoformat(session_data["updated_at"]),
                metadata=session_data.get("metadata"),
            )
        except Exception as e:
            raise AdkSessionError(
                f"Failed to deserialize session: {e}", error_type=AdkErrorType.INTERNAL, cause=e
            )

    async def create_session(
        self,
        user_id: str,
        app_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """Create a new session with atomic operations."""
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())

            now = datetime.now(timezone.utc)
            session = AdkSession(
                session_id=session_id,
                user_id=user_id,
                app_name=app_name,
                messages=[],
                created_at=now,
                updated_at=now,
                metadata=metadata,
            )

            # Use Redis transaction for atomic operation
            async with self.redis_client.pipeline(transaction=True) as pipe:
                session_key = self._get_session_key(session_id)
                user_sessions_key = self._get_user_sessions_key(user_id)

                # Set session data with TTL
                pipe.setex(session_key, self.config.ttl_seconds, self._serialize_session(session))

                # Add session ID to user's session set with TTL
                pipe.sadd(user_sessions_key, session_id)
                pipe.expire(user_sessions_key, self.config.ttl_seconds)

                # Execute transaction
                await pipe.execute()

            return AdkSuccess(session)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to create session: {e}", error_type=AdkErrorType.INTERNAL, cause=e
                )
            )

    async def get_session(
        self, session_id: str
    ) -> AdkResult[Optional[AdkSession], AdkSessionError]:
        """Get session from Redis."""
        try:
            session_key = self._get_session_key(session_id)
            data = await self.redis_client.get(session_key)

            if not data:
                return AdkSuccess(None)

            session = self._deserialize_session(data)

            # Update TTL on access
            await self.redis_client.expire(session_key, self.config.ttl_seconds)

            return AdkSuccess(session)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to get session: {e}",
                    session_id=session_id,
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def update_session(self, session: AdkSession) -> AdkResult[AdkSession, AdkSessionError]:
        """Update session in Redis."""
        try:
            session.updated_at = datetime.now(timezone.utc)

            session_key = self._get_session_key(session.session_id)
            await self.redis_client.setex(
                session_key, self.config.ttl_seconds, self._serialize_session(session)
            )

            return AdkSuccess(session)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to update session: {e}",
                    session_id=session.session_id,
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def delete_session(self, session_id: str) -> AdkResult[bool, AdkSessionError]:
        """Delete session from Redis."""
        try:
            # Get session to find user_id for cleanup
            session_result = await self.get_session(session_id)
            if isinstance(session_result, AdkFailure):
                return session_result

            session = session_result.data
            if not session:
                return AdkSuccess(False)

            # Use transaction for atomic cleanup
            async with self.redis_client.pipeline(transaction=True) as pipe:
                session_key = self._get_session_key(session_id)
                user_sessions_key = self._get_user_sessions_key(session.user_id)

                # Delete session
                pipe.delete(session_key)

                # Remove from user's session set
                pipe.srem(user_sessions_key, session_id)

                result = await pipe.execute()

            return AdkSuccess(result[0] > 0)  # First command result

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to delete session: {e}",
                    session_id=session_id,
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def add_message(
        self, session_id: str, message: AdkMessage
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """Add message to session."""
        try:
            # Get current session
            session_result = await self.get_session(session_id)
            if isinstance(session_result, AdkFailure):
                return session_result

            session = session_result.data
            if not session:
                return AdkFailure(
                    AdkSessionError(
                        f"Session not found: {session_id}",
                        session_id=session_id,
                        error_type=AdkErrorType.VALIDATION,
                    )
                )

            # Add message and update session
            session.add_message(message)

            # Enforce message limit
            if len(session.messages) > self.config.max_messages_per_session:
                # Keep only the most recent messages
                session.messages = session.messages[-self.config.max_messages_per_session :]

            return await self.update_session(session)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to add message: {e}",
                    session_id=session_id,
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def get_messages(
        self, session_id: str, limit: Optional[int] = None, offset: int = 0
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
            return AdkFailure(
                AdkSessionError(
                    f"Failed to get messages: {e}",
                    session_id=session_id,
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def get_user_sessions(
        self, user_id: str, app_name: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> AdkResult[List[AdkSession], AdkSessionError]:
        """Get sessions for a user."""
        try:
            user_sessions_key = self._get_user_sessions_key(user_id)
            session_ids = await self.redis_client.smembers(user_sessions_key)

            sessions = []
            for session_id in session_ids:
                session_result = await self.get_session(session_id.decode())
                if isinstance(session_result, AdkSuccess) and session_result.data:
                    session = session_result.data
                    if app_name is None or session.app_name == app_name:
                        sessions.append(session)

            # Sort by updated_at descending
            sessions.sort(key=lambda s: s.updated_at, reverse=True)

            # Apply pagination
            sessions = sessions[offset : offset + limit]

            return AdkSuccess(sessions)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to get user sessions: {e}",
                    user_id=user_id,
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def cleanup_expired_sessions(self) -> AdkResult[int, AdkSessionError]:
        """Cleanup expired sessions (Redis handles this automatically via TTL)."""
        try:
            # Redis automatically handles TTL cleanup
            # We can return 0 as cleanup count since it's automatic
            return AdkSuccess(0)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to cleanup expired sessions: {e}",
                    error_type=AdkErrorType.INTERNAL,
                    cause=e,
                )
            )

    async def get_stats(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """Get provider statistics."""
        try:
            info = await self.redis_client.info("memory")
            keyspace = await self.redis_client.info("keyspace")

            # Count sessions by scanning keys
            session_count = 0
            async for key in self.redis_client.scan_iter(match=f"{self.config.key_prefix}*"):
                session_count += 1

            stats = {
                "provider": "redis",
                "total_sessions": session_count,
                "memory_used": info.get("used_memory", 0),
                "memory_used_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_info": keyspace,
            }

            return AdkSuccess(stats)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to get stats: {e}", error_type=AdkErrorType.INTERNAL, cause=e
                )
            )

    async def health_check(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """Perform health check."""
        try:
            start_time = datetime.now()
            await self.redis_client.ping()
            latency = (datetime.now() - start_time).total_seconds() * 1000

            info = await self.redis_client.info()

            health = {
                "healthy": True,
                "provider": "redis",
                "latency_ms": latency,
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
            }

            return AdkSuccess(health)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Health check failed: {e}", error_type=AdkErrorType.INTERNAL, cause=e
                )
            )

    async def close(self) -> AdkResult[None, AdkSessionError]:
        """Close Redis connection."""
        try:
            if self.redis_client:
                await self.redis_client.aclose()
            return AdkSuccess(None)

        except Exception as e:
            return AdkFailure(
                AdkSessionError(
                    f"Failed to close connection: {e}", error_type=AdkErrorType.INTERNAL, cause=e
                )
            )


async def create_redis_session_provider(
    config: AdkRedisSessionConfig,
) -> AdkResult[AdkRedisSessionProvider, AdkSessionError]:
    """
    Create a Redis session provider with connection pooling.

    Args:
        config: Redis session configuration

    Returns:
        Result containing the provider or error
    """
    try:
        if not REDIS_AVAILABLE:
            return AdkFailure(
                AdkSessionError(
                    "Redis not available. Install with: pip install redis",
                    error_type=AdkErrorType.VALIDATION,
                    severity=AdkErrorSeverity.CRITICAL,
                )
            )

        # Create connection pool
        connection_kwargs = {
            "socket_timeout": config.socket_timeout,
            "socket_connect_timeout": config.socket_connect_timeout,
            "max_connections": config.connection_pool_size,
            "decode_responses": False,  # We handle encoding manually
        }

        if config.password:
            connection_kwargs["password"] = config.password

        # Create Redis client
        if config.url:
            redis_client = redis.from_url(config.url, **connection_kwargs)
        else:
            redis_client = redis.Redis(
                host=config.host, port=config.port, db=config.db, **connection_kwargs
            )

        # Test connection
        await redis_client.ping()

        provider = AdkRedisSessionProvider(config, redis_client)
        return AdkSuccess(provider)

    except Exception as e:
        return AdkFailure(
            AdkSessionError(
                f"Failed to create Redis session provider: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e,
            )
        )
