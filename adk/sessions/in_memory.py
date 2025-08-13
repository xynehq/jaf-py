"""
ADK In-Memory Session Provider - Development and Testing

This module provides a simple in-memory session provider for development
and testing purposes. Not recommended for production use.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict

from .base import AdkSessionProvider, AdkSessionConfig
from ..types import AdkSession, AdkMessage, AdkResult, AdkSuccess, AdkFailure
from ..errors import AdkSessionError, AdkErrorType

class AdkInMemorySessionProvider(AdkSessionProvider):
    """
    In-memory session provider for development and testing.
    
    Features:
    - Fast access with in-memory storage
    - TTL support with background cleanup
    - Thread-safe operations
    - Development-friendly logging
    
    Note: Data is lost when the process restarts.
    """
    
    def __init__(self, config: AdkSessionConfig):
        super().__init__(config)
        self.sessions: Dict[str, AdkSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.session_expiry: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop for expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue cleanup even if there's an error
                pass
    
    def _is_expired(self, session_id: str) -> bool:
        """Check if a session is expired."""
        expiry = self.session_expiry.get(session_id)
        if not expiry:
            return False
        return datetime.now(timezone.utc) > expiry
    
    def _update_expiry(self, session_id: str):
        """Update session expiry time."""
        self.session_expiry[session_id] = (
            datetime.now(timezone.utc) + timedelta(seconds=self.config.ttl_seconds)
        )
    
    async def create_session(
        self,
        user_id: str,
        app_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AdkResult[AdkSession, AdkSessionError]:
        """Create a new session."""
        try:
            async with self._lock:
                if session_id is None:
                    session_id = str(uuid.uuid4())
                
                # Check if session already exists
                if session_id in self.sessions:
                    return AdkFailure(AdkSessionError(
                        f"Session already exists: {session_id}",
                        session_id=session_id,
                        error_type=AdkErrorType.VALIDATION
                    ))
                
                # Enforce per-user session limit
                user_session_ids = self.user_sessions[user_id]
                if len(user_session_ids) >= self.config.max_sessions_per_user:
                    # Remove oldest session
                    oldest_session_id = user_session_ids.pop(0)
                    self.sessions.pop(oldest_session_id, None)
                    self.session_expiry.pop(oldest_session_id, None)
                
                now = datetime.now(timezone.utc)
                session = AdkSession(
                    session_id=session_id,
                    user_id=user_id,
                    app_name=app_name,
                    messages=[],
                    created_at=now,
                    updated_at=now,
                    metadata=metadata
                )
                
                self.sessions[session_id] = session
                self.user_sessions[user_id].append(session_id)
                self._update_expiry(session_id)
                
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
        """Get an existing session."""
        try:
            async with self._lock:
                if session_id not in self.sessions:
                    return AdkSuccess(None)
                
                if self._is_expired(session_id):
                    # Remove expired session
                    session = self.sessions.pop(session_id, None)
                    self.session_expiry.pop(session_id, None)
                    if session:
                        user_sessions = self.user_sessions[session.user_id]
                        if session_id in user_sessions:
                            user_sessions.remove(session_id)
                    return AdkSuccess(None)
                
                # Update expiry on access
                self._update_expiry(session_id)
                
                return AdkSuccess(self.sessions[session_id])
                
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
        """Update an existing session."""
        try:
            async with self._lock:
                if session.session_id not in self.sessions:
                    return AdkFailure(AdkSessionError(
                        f"Session not found: {session.session_id}",
                        session_id=session.session_id,
                        error_type=AdkErrorType.VALIDATION
                    ))
                
                session.updated_at = datetime.now(timezone.utc)
                self.sessions[session.session_id] = session
                self._update_expiry(session.session_id)
                
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
        """Delete a session."""
        try:
            async with self._lock:
                session = self.sessions.pop(session_id, None)
                self.session_expiry.pop(session_id, None)
                
                if session:
                    user_sessions = self.user_sessions[session.user_id]
                    if session_id in user_sessions:
                        user_sessions.remove(session_id)
                    return AdkSuccess(True)
                
                return AdkSuccess(False)
                
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
        """Add a message to a session."""
        try:
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
        """Get messages from a session."""
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
            async with self._lock:
                user_session_ids = self.user_sessions.get(user_id, [])
                sessions = []
                
                for session_id in user_session_ids:
                    if session_id in self.sessions and not self._is_expired(session_id):
                        session = self.sessions[session_id]
                        if app_name is None or session.app_name == app_name:
                            sessions.append(session)
                
                # Sort by updated_at descending
                sessions.sort(key=lambda s: s.updated_at, reverse=True)
                
                # Apply pagination
                sessions = sessions[offset:offset + limit]
                
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
            async with self._lock:
                expired_session_ids = []
                now = datetime.now(timezone.utc)
                
                for session_id, expiry in self.session_expiry.items():
                    if now > expiry:
                        expired_session_ids.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_session_ids:
                    session = self.sessions.pop(session_id, None)
                    self.session_expiry.pop(session_id, None)
                    
                    if session:
                        user_sessions = self.user_sessions[session.user_id]
                        if session_id in user_sessions:
                            user_sessions.remove(session_id)
                
                return AdkSuccess(len(expired_session_ids))
                
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to cleanup expired sessions: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def get_stats(self) -> AdkResult[Dict[str, Any], AdkSessionError]:
        """Get provider statistics."""
        try:
            async with self._lock:
                total_messages = sum(len(session.messages) for session in self.sessions.values())
                
                stats = {
                    "provider": "in_memory",
                    "total_sessions": len(self.sessions),
                    "total_messages": total_messages,
                    "total_users": len(self.user_sessions),
                    "average_messages_per_session": (
                        total_messages / len(self.sessions) if self.sessions else 0
                    )
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
            health = {
                "healthy": True,
                "provider": "in_memory",
                "total_sessions": len(self.sessions),
                "cleanup_task_running": (
                    self._cleanup_task is not None and not self._cleanup_task.done()
                )
            }
            
            return AdkSuccess(health)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Health check failed: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))
    
    async def close(self) -> AdkResult[None, AdkSessionError]:
        """Close the provider and clean up resources."""
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear all data
            async with self._lock:
                self.sessions.clear()
                self.user_sessions.clear()
                self.session_expiry.clear()
            
            return AdkSuccess(None)
            
        except Exception as e:
            return AdkFailure(AdkSessionError(
                f"Failed to close provider: {e}",
                error_type=AdkErrorType.INTERNAL,
                cause=e
            ))

def create_in_memory_session_provider(
    config: Optional[AdkSessionConfig] = None
) -> AdkInMemorySessionProvider:
    """
    Create an in-memory session provider.
    
    Args:
        config: Optional session configuration
        
    Returns:
        AdkInMemorySessionProvider instance
    """
    if config is None:
        config = AdkSessionConfig()
    
    return AdkInMemorySessionProvider(config)