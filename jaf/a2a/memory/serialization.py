"""
A2A Task Serialization Functions for JAF

This module provides pure functions for serializing and deserializing A2A tasks for storage.
It handles the conversion between A2A task objects and their storage representations,
ensuring data integrity and consistency across different storage backends.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..types import A2AMessage, A2ATask
from .types import A2AResult, create_a2a_failure, create_a2a_success, create_a2a_task_storage_error


@dataclass(frozen=True)
class A2ATaskSerialized:
    """Serialized representation of an A2A task for storage"""
    task_id: str
    context_id: str
    state: str
    task_data: str  # JSON string of the full task
    created_at: str  # ISO string
    updated_at: str  # ISO string
    status_message: Optional[str] = None  # Serialized status message for quick access
    metadata: Optional[str] = None  # JSON string of metadata

def serialize_a2a_task(
    task: A2ATask,
    metadata: Optional[Dict[str, Any]] = None
) -> A2AResult[A2ATaskSerialized]:
    """
    Pure function to serialize an A2A task for storage
    
    Args:
        task: The A2A task to serialize
        metadata: Optional metadata to store with the task
        
    Returns:
        A2AResult containing the serialized task or an error
    """
    try:
        now = datetime.now(timezone.utc).isoformat()

        # Determine the creation time from metadata or status, falling back to now
        created_at_iso = now
        if task.metadata and task.metadata.get("created_at"):
            created_at_iso = task.metadata["created_at"]
        elif task.status and task.status.timestamp:
            created_at_iso = task.status.timestamp

        # Extract status message for indexing if present
        status_message = None
        if task.status.message:
            try:
                status_message = task.status.message.model_dump_json(by_alias=True)
            except Exception:
                # If message serialization fails, continue without it
                pass

        # Serialize the full task, handling circular references
        try:
            # First check for circular references in metadata
            if task.metadata:
                for key, value in task.metadata.items():
                    if isinstance(value, A2ATask):
                        raise TypeError(f"Object of type A2ATask is not JSON serializable")
            
            task_data = task.model_dump_json(by_alias=True)
        except TypeError as e:
            # Re-raise TypeError (including circular reference errors) directly
            raise
        except Exception as e:
            # Check if this is due to circular references
            if "circular" in str(e).lower() or "recursion" in str(e).lower():
                raise TypeError(f"Object of type A2ATask is not JSON serializable")
            raise

        # Serialize metadata if provided
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        metadata_str = None
        if metadata:
            metadata_str = json.dumps(metadata, separators=(',', ':'), default=datetime_converter)

        serialized = A2ATaskSerialized(
            task_id=task.id,
            context_id=task.context_id,
            state=task.status.state.value,
            task_data=task_data,
            status_message=status_message,
            created_at=created_at_iso,
            updated_at=now,
            metadata=metadata_str,
        )

        return create_a2a_success(serialized)

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('serialize', 'memory', task.id, error)
        )

def deserialize_a2a_task(stored: A2ATaskSerialized) -> A2AResult[A2ATask]:
    """
    Pure function to deserialize an A2A task from storage
    
    Args:
        stored: The serialized task data
        
    Returns:
        A2AResult containing the deserialized task or an error
    """
    try:
        # Parse the task data
        task_dict = json.loads(stored.task_data)

        # Create the task object using Pydantic
        try:
            task = A2ATask.model_validate(task_dict)
        except Exception as validation_error:
            # Convert Pydantic validation errors to our format
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'deserialize',
                    'memory',
                    stored.task_id,
                    Exception('Invalid task structure')
                )
            )

        # Validate that the deserialized task has required fields
        if not task.id or not task.context_id or not task.status or task.kind != "task":
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'deserialize',
                    'memory',
                    stored.task_id,
                    Exception('Invalid task structure')
                )
            )

        return create_a2a_success(task)

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('deserialize', 'memory', stored.task_id, error)
        )

def create_task_index(task: A2ATask) -> A2AResult[Dict[str, Any]]:
    """
    Pure function to create a minimal task representation for indexing
    
    Args:
        task: The A2A task to index
        
    Returns:
        A2AResult containing the task index data or an error
    """
    try:
        index_data = {
            'task_id': task.id,
            'context_id': task.context_id,
            'state': task.status.state.value,
            'timestamp': task.status.timestamp or datetime.now().isoformat(),
            'has_history': bool(task.history and len(task.history) > 0),
            'has_artifacts': bool(task.artifacts and len(task.artifacts) > 0)
        }

        return create_a2a_success(index_data)

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('index', 'memory', task.id, error)
        )

def extract_task_search_text(task: A2ATask) -> A2AResult[str]:
    """
    Pure function to extract searchable text from a task for full-text search
    
    Args:
        task: The A2A task to extract text from
        
    Returns:
        A2AResult containing the extracted text or an error
    """
    try:
        text_parts: List[str] = []

        # Extract text from status message
        if task.status.message:
            _extract_text_from_message(task.status.message, text_parts)

        # Extract text from history
        if task.history:
            for message in task.history:
                _extract_text_from_message(message, text_parts)

        # Extract text from artifacts
        if task.artifacts:
            for artifact in task.artifacts:
                if artifact.name:
                    text_parts.append(artifact.name)
                if artifact.description:
                    text_parts.append(artifact.description)

                for part in artifact.parts:
                    if part.kind == "text":
                        text_parts.append(part.text)

        return create_a2a_success(' '.join(text_parts).strip())

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('extract-text', 'memory', task.id, error)
        )

def _extract_text_from_message(message: A2AMessage, text_parts: List[str]) -> None:
    """Helper function to extract text from A2A message parts"""
    for part in message.parts:
        if part.kind == "text":
            text_parts.append(part.text)
        elif part.kind == "data" and part.data:
            # Extract any string values from data
            for value in part.data.values():
                if isinstance(value, str):
                    text_parts.append(value)
        elif part.kind == "file" and part.file.name:
            text_parts.append(part.file.name)

def validate_task_integrity(task: A2ATask) -> A2AResult[bool]:
    """
    Pure function to validate task data integrity
    
    Args:
        task: The A2A task to validate
        
    Returns:
        A2AResult containing True if valid or an error
    """
    try:
        # Check required fields
        if not task.id or not isinstance(task.id, str):
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'validate',
                    'memory',
                    getattr(task, 'id', None),
                    Exception('Task ID is required and must be a string')
                )
            )

        if not task.context_id or not isinstance(task.context_id, str):
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'validate',
                    'memory',
                    task.id,
                    Exception('Context ID is required')
                )
            )

        if not task.status or not task.status.state:
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'validate',
                    'memory',
                    task.id,
                    Exception('Task status and state are required')
                )
            )

        if task.kind != "task":
            return create_a2a_failure(
                create_a2a_task_storage_error(
                    'validate',
                    'memory',
                    task.id,
                    Exception('Task kind must be "task"')
                )
            )

        # Validate history if present
        if task.history:
            if not isinstance(task.history, list):
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'validate',
                        'memory',
                        task.id,
                        Exception('Task history must be a list')
                    )
                )

            for i, message in enumerate(task.history):
                if not message.message_id or not message.parts or not isinstance(message.parts, list):
                    return create_a2a_failure(
                        create_a2a_task_storage_error(
                            'validate',
                            'memory',
                            task.id,
                            Exception(f'Invalid message at index {i} in task history')
                        )
                    )

        # Validate artifacts if present
        if task.artifacts:
            if not isinstance(task.artifacts, list):
                return create_a2a_failure(
                    create_a2a_task_storage_error(
                        'validate',
                        'memory',
                        task.id,
                        Exception('Task artifacts must be a list')
                    )
                )

            for i, artifact in enumerate(task.artifacts):
                if not artifact.artifact_id or not artifact.parts or not isinstance(artifact.parts, list):
                    return create_a2a_failure(
                        create_a2a_task_storage_error(
                            'validate',
                            'memory',
                            task.id,
                            Exception(f'Invalid artifact at index {i} in task')
                        )
                    )

        return create_a2a_success(True)

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('validate', 'memory', getattr(task, 'id', None), error)
        )

def clone_task(task: A2ATask) -> A2AResult[A2ATask]:
    """
    Pure function to create a deep copy of a task (for immutability)
    
    Args:
        task: The A2A task to clone
        
    Returns:
        A2AResult containing the cloned task or an error
    """
    try:
        # Use Pydantic's built-in copy mechanism for deep cloning
        cloned = task.model_copy(deep=True)
        return create_a2a_success(cloned)

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('clone', 'memory', task.id, error)
        )

def sanitize_task(task: A2ATask) -> A2AResult[A2ATask]:
    """
    Pure function to sanitize task data for storage
    Removes any potentially dangerous or invalid data
    
    Args:
        task: The A2A task to sanitize
        
    Returns:
        A2AResult containing the sanitized task or an error
    """
    try:
        # First validate the task
        validation_result = validate_task_integrity(task)
        if not validation_result.data:
            return validation_result

        # Clone the task to avoid mutation
        clone_result = clone_task(task)
        if not clone_result.data:
            return clone_result

        sanitized = clone_result.data

        # Ensure timestamps are valid ISO strings
        if sanitized.status.timestamp:
            try:
                # Parse and re-format timestamp to ensure it's valid
                dt = datetime.fromisoformat(sanitized.status.timestamp.replace('Z', '+00:00'))
                # Create a new task with the corrected timestamp
                sanitized = sanitized.model_copy(update={
                    'status': sanitized.status.model_copy(update={
                        'timestamp': dt.isoformat()
                    })
                })
            except Exception:
                # Remove invalid timestamp
                sanitized = sanitized.model_copy(update={
                    'status': sanitized.status.model_copy(update={
                        'timestamp': None
                    })
                })

        return create_a2a_success(sanitized)

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('sanitize', 'memory', getattr(task, 'id', None), error)
        )
