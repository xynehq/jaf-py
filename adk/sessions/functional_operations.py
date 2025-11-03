"""
ADK Functional Session Operations

This module provides pure functions for session operations, separating business
logic from side effects to follow functional programming principles.
"""

from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime
from ..types import AdkMessage, AdkResult, AdkSuccess, AdkFailure
from ..types.immutable_session import ImmutableAdkSession, create_immutable_session
from ..errors import AdkSessionError


# Type aliases for better readability
SessionOperation = Callable[[ImmutableAdkSession], ImmutableAdkSession]
SessionPredicate = Callable[[ImmutableAdkSession], bool]
SessionMapper = Callable[[ImmutableAdkSession], Any]


# ========== Pure Session Creation Functions ==========


def create_new_session(
    session_id: str,
    user_id: str,
    app_name: str,
    initial_message: Optional[AdkMessage] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImmutableAdkSession:
    """
    Pure function to create a new session.

    Args:
        session_id: Unique session identifier
        user_id: User identifier
        app_name: Application name
        initial_message: Optional initial message
        metadata: Optional metadata

    Returns:
        New ImmutableAdkSession
    """
    initial_messages = [initial_message] if initial_message else []
    return create_immutable_session(
        session_id=session_id,
        user_id=user_id,
        app_name=app_name,
        initial_messages=initial_messages,
        metadata=metadata,
    )


def restore_session_from_data(session_data: Dict[str, Any]) -> ImmutableAdkSession:
    """
    Pure function to restore a session from serialized data.

    Args:
        session_data: Serialized session data

    Returns:
        Restored ImmutableAdkSession
    """
    return ImmutableAdkSession.from_dict(session_data)


# ========== Pure Session Transformation Functions ==========


def add_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> SessionOperation:
    """
    Create a session operation that adds a user message.

    Args:
        content: Message content
        metadata: Optional message metadata

    Returns:
        Function that adds the message to a session
    """

    def operation(session: ImmutableAdkSession) -> ImmutableAdkSession:
        from ..types import create_user_message

        message = create_user_message(content, metadata=metadata)
        return session.with_message(message)

    return operation


def add_assistant_message(
    content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> SessionOperation:
    """
    Create a session operation that adds an assistant message.

    Args:
        content: Message content
        tool_calls: Optional tool calls

    Returns:
        Function that adds the message to a session
    """

    def operation(session: ImmutableAdkSession) -> ImmutableAdkSession:
        from ..types import create_assistant_message

        message = create_assistant_message(content, tool_calls=tool_calls)
        return session.with_message(message)

    return operation


def update_metadata(key: str, value: Any) -> SessionOperation:
    """
    Create a session operation that updates metadata.

    Args:
        key: Metadata key
        value: Metadata value

    Returns:
        Function that updates the metadata
    """

    def operation(session: ImmutableAdkSession) -> ImmutableAdkSession:
        return session.with_metadata(key, value)

    return operation


def compose_operations(*operations: SessionOperation) -> SessionOperation:
    """
    Compose multiple session operations into a single operation.

    Args:
        operations: Session operations to compose

    Returns:
        Composed operation function
    """

    def composed(session: ImmutableAdkSession) -> ImmutableAdkSession:
        result = session
        for operation in operations:
            result = operation(result)
        return result

    return composed


# ========== Pure Session Query Functions ==========


def is_session_empty(session: ImmutableAdkSession) -> bool:
    """
    Pure predicate to check if a session has no messages.

    Args:
        session: Session to check

    Returns:
        True if session has no messages
    """
    return len(session.messages) == 0


def has_recent_activity(minutes: int) -> SessionPredicate:
    """
    Create a predicate that checks for recent activity.

    Args:
        minutes: Number of minutes to consider "recent"

    Returns:
        Predicate function
    """

    def predicate(session: ImmutableAdkSession) -> bool:
        time_diff = datetime.now() - session.updated_at
        return time_diff.total_seconds() < (minutes * 60)

    return predicate


def has_message_from_role(role: str) -> SessionPredicate:
    """
    Create a predicate that checks for messages from a specific role.

    Args:
        role: Role to check for

    Returns:
        Predicate function
    """

    def predicate(session: ImmutableAdkSession) -> bool:
        return any(msg.role == role for msg in session.messages)

    return predicate


def get_last_message(session: ImmutableAdkSession) -> Optional[AdkMessage]:
    """
    Pure function to get the last message from a session.

    Args:
        session: Session to query

    Returns:
        Last message or None if session is empty
    """
    if session.messages:
        return session.messages[-1]
    return None


def get_messages_by_role(role: str) -> SessionMapper:
    """
    Create a mapper that extracts messages by role.

    Args:
        role: Role to filter by

    Returns:
        Mapper function that returns messages
    """

    def mapper(session: ImmutableAdkSession) -> List[AdkMessage]:
        return list(msg for msg in session.messages if msg.role == role)

    return mapper


def count_messages_by_role(session: ImmutableAdkSession) -> Dict[str, int]:
    """
    Pure function to count messages by role.

    Args:
        session: Session to analyze

    Returns:
        Dictionary with role counts
    """
    counts = {}
    for message in session.messages:
        counts[message.role] = counts.get(message.role, 0) + 1
    return counts


# ========== Session Validation Functions ==========


def validate_session_structure(session: ImmutableAdkSession) -> List[str]:
    """
    Pure function to validate session structure.

    Args:
        session: Session to validate

    Returns:
        List of validation error messages
    """
    errors = []

    if not session.session_id:
        errors.append("Session ID is required")

    if not session.user_id:
        errors.append("User ID is required")

    if not session.app_name:
        errors.append("App name is required")

    if session.created_at > session.updated_at:
        errors.append("Created time cannot be after updated time")

    # Validate messages
    for i, message in enumerate(session.messages):
        if not message.role:
            errors.append(f"Message {i} missing role")

        if not message.content and not message.tool_calls:
            errors.append(f"Message {i} missing content and tool calls")

    return errors


def is_valid_session(session: ImmutableAdkSession) -> bool:
    """
    Pure predicate to check if a session is valid.

    Args:
        session: Session to validate

    Returns:
        True if session is valid
    """
    return len(validate_session_structure(session)) == 0


# ========== Session Serialization Functions ==========


def serialize_session(session: ImmutableAdkSession) -> Dict[str, Any]:
    """
    Pure function to serialize a session to a dictionary.

    Args:
        session: Session to serialize

    Returns:
        Serialized session data
    """
    return session.to_dict()


def serialize_session_summary(session: ImmutableAdkSession) -> Dict[str, Any]:
    """
    Pure function to create a lightweight session summary.

    Args:
        session: Session to summarize

    Returns:
        Session summary data
    """
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "app_name": session.app_name,
        "message_count": len(session.messages),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "last_message_role": session.messages[-1].role if session.messages else None,
    }


# ========== Session Comparison Functions ==========


def sessions_equal(session1: ImmutableAdkSession, session2: ImmutableAdkSession) -> bool:
    """
    Pure function to compare two sessions for equality.

    Args:
        session1: First session
        session2: Second session

    Returns:
        True if sessions are equal
    """
    return (
        session1.session_id == session2.session_id
        and session1.user_id == session2.user_id
        and session1.app_name == session2.app_name
        and session1.messages == session2.messages
        and session1.metadata == session2.metadata
    )


def get_session_diff(
    old_session: ImmutableAdkSession, new_session: ImmutableAdkSession
) -> Dict[str, Any]:
    """
    Pure function to get the difference between two sessions.

    Args:
        old_session: Original session
        new_session: Updated session

    Returns:
        Dictionary describing the differences
    """
    diff = {}

    if old_session.messages != new_session.messages:
        old_count = len(old_session.messages)
        new_count = len(new_session.messages)
        diff["message_count_change"] = new_count - old_count

        if new_count > old_count:
            diff["new_messages"] = new_session.messages[old_count:]

    if old_session.metadata != new_session.metadata:
        diff["metadata_changed"] = True

    if old_session.updated_at != new_session.updated_at:
        diff["last_updated"] = new_session.updated_at.isoformat()

    return diff


# ========== Higher-Order Session Functions ==========


def map_session(mapper: SessionMapper, session: ImmutableAdkSession) -> Any:
    """
    Apply a mapper function to a session.

    Args:
        mapper: Function to apply to the session
        session: Session to map

    Returns:
        Result of applying the mapper
    """
    return mapper(session)


def filter_sessions(
    predicate: SessionPredicate, sessions: List[ImmutableAdkSession]
) -> List[ImmutableAdkSession]:
    """
    Filter a list of sessions using a predicate.

    Args:
        predicate: Function to test each session
        sessions: List of sessions to filter

    Returns:
        Filtered list of sessions
    """
    return [session for session in sessions if predicate(session)]


def transform_session(
    operation: SessionOperation, session: ImmutableAdkSession
) -> ImmutableAdkSession:
    """
    Transform a session using an operation.

    Args:
        operation: Function to transform the session
        session: Session to transform

    Returns:
        Transformed session
    """
    return operation(session)


def reduce_sessions(
    reducer: Callable[[Any, ImmutableAdkSession], Any],
    initial: Any,
    sessions: List[ImmutableAdkSession],
) -> Any:
    """
    Reduce a list of sessions to a single value.

    Args:
        reducer: Function to combine sessions
        initial: Initial value for reduction
        sessions: List of sessions to reduce

    Returns:
        Reduced value
    """
    result = initial
    for session in sessions:
        result = reducer(result, session)
    return result
