"""
ADK Session Provider Base - Abstract Session Provider Interface

This module defines the abstract base class for all ADK session providers,
ensuring consistent interface across different storage backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..errors import AdkSessionError
from ..types import AdkFailure, AdkMessage, AdkResult, AdkSession, AdkSuccess


@dataclass
class AdkSessionConfig:
    """Base configuration for ADK session providers."""

    ttl_seconds: int = 3600  # 1 hour default
    max_sessions_per_user: int = 100
    max_messages_per_session: int = 1000
    enable_compression: bool = False
    metadata: Optional[Dict[str, Any]] = None


class AdkSessionProvider(ABC):
    """
    Abstract base class for ADK session providers.

    Provides production-ready session management with consistent interface
    across different storage backends (in-memory, Redis, PostgreSQL).
    """

    def __init__(self, config: AdkSessionConfig):
        self.config = config

    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        app_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """
        Create a new session.

        Args:
            user_id: User identifier
            app_name: Application name
            session_id: Optional session ID (generated if not provided)
            metadata: Optional session metadata

        Returns:
            Result containing the created session or error
        """
        pass

    @abstractmethod
    async def get_session(
        self, session_id: str
    ) -> AdkResult[Optional[AdkSession], AdkSessionError]:
        """
        Get an existing session.

        Args:
            session_id: Session identifier

        Returns:
            Result containing the session (None if not found) or error
        """
        pass

    @abstractmethod
    async def update_session(self, session: AdkSession) -> AdkResult[AdkSession, AdkSessionError]:
        """
        Update an existing session.

        Args:
            session: Session to update

        Returns:
            Result containing the updated session or error
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> AdkResult[bool, AdkSessionError]:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            Result containing True if deleted, False if not found, or error
        """
        pass

    @abstractmethod
    async def add_message(
        self, session_id: str, message: AdkMessage
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """
        Add a message to a session.

        Args:
            session_id: Session identifier
            message: Message to add

        Returns:
            Result containing the updated session or error
        """
        pass

    @abstractmethod
    async def get_messages(
        self, session_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> AdkResult[List[AdkMessage], AdkSessionError]:
        """
        Get messages from a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            Result containing the messages or error
        """
        pass

    @abstractmethod
    async def get_user_sessions(
        self, user_id: str, app_name: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> AdkResult[List[AdkSession], AdkSessionError]:
        """
        Get sessions for a user.

        Args:
            user_id: User identifier
            app_name: Optional app name filter
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            Result containing the sessions or error
        """
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self) -> AdkResult[int, AdkSessionError]:
        """
        Clean up expired sessions.

        Returns:
            Result containing the number of sessions cleaned up or error
        """
        pass

    @abstractmethod
    async def get_stats(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """
        Get provider statistics.

        Returns:
            Result containing statistics or error
        """
        pass

    @abstractmethod
    async def health_check(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """
        Perform a health check.

        Returns:
            Result containing health status or error
        """
        pass

    @abstractmethod
    async def close(self) -> AdkResult[None, AdkSessionError]:
        """
        Close the provider and clean up resources.

        Returns:
            Result indicating success or error
        """
        pass
