"""
A2A Task Memory Types for JAF

This module extends the core memory system to support A2A task queue persistence.
It provides types, interfaces, and error handling specifically for A2A task storage
while leveraging the existing JAF memory infrastructure.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union

from pydantic import BaseModel, Field

from ...memory.types import Failure, Success
from ..types import A2ATask, TaskState

# Generic types for A2A results
T = TypeVar('T')
E = TypeVar('E', bound='A2ATaskError')

# A2A Task storage and retrieval types

@dataclass(frozen=True)
class A2ATaskQuery:
    """Query parameters for searching A2A tasks in memory providers"""
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    state: Optional[TaskState] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    include_history: bool = True
    include_artifacts: bool = True

@dataclass(frozen=True)
class A2ATaskStorage:
    """Internal storage representation of an A2A task"""
    task_id: str
    context_id: str
    state: TaskState
    task_data: str  # Serialized A2ATask JSON
    status_message: Optional[str] = None  # Serialized status message for quick access
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.now())
        if self.updated_at is None:
            object.__setattr__(self, 'updated_at', datetime.now())

class A2ATaskProvider(Protocol):
    """
    Protocol defining the interface for A2A task storage providers.
    
    This extends the memory provider pattern for A2A-specific task persistence,
    providing optimized operations for task lifecycle management.
    """

    async def store_task(
        self,
        task: A2ATask,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'A2AResult[None]':
        """Store a new A2A task"""
        ...

    async def get_task(self, task_id: str) -> 'A2AResult[Optional[A2ATask]]':
        """Retrieve a task by ID"""
        ...

    async def update_task(
        self,
        task: A2ATask,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'A2AResult[None]':
        """Update an existing task"""
        ...

    async def update_task_status(
        self,
        task_id: str,
        state: TaskState,
        status_message: Optional[Any] = None,
        timestamp: Optional[str] = None
    ) -> 'A2AResult[None]':
        """Update task status only (optimized for frequent status changes)"""
        ...

    async def find_tasks(self, query: A2ATaskQuery) -> 'A2AResult[List[A2ATask]]':
        """Search tasks by query parameters"""
        ...

    async def get_tasks_by_context(
        self,
        context_id: str,
        limit: Optional[int] = None
    ) -> 'A2AResult[List[A2ATask]]':
        """Get tasks by context ID"""
        ...

    async def delete_task(self, task_id: str) -> 'A2AResult[bool]':
        """Delete a task and return True if it existed"""
        ...

    async def delete_tasks_by_context(self, context_id: str) -> 'A2AResult[int]':
        """Delete tasks by context ID and return count deleted"""
        ...

    async def cleanup_expired_tasks(self) -> 'A2AResult[int]':
        """Clean up expired tasks and return count deleted"""
        ...

    async def get_task_stats(self, context_id: Optional[str] = None) -> 'A2AResult[Dict[str, Any]]':
        """Get task statistics"""
        ...

    async def health_check(self) -> 'A2AResult[Dict[str, Any]]':
        """Check provider health and return status information"""
        ...

    async def close(self) -> 'A2AResult[None]':
        """Close/cleanup the provider"""
        ...

# Configuration models for A2A task storage

class A2ATaskMemoryConfig(BaseModel):
    """Base configuration for A2A task memory"""
    model_config = {"frozen": True}

    type: str
    key_prefix: str = Field(default="jaf:a2a:tasks:")
    default_ttl: Optional[int] = None  # Default TTL in seconds for tasks
    cleanup_interval: int = Field(default=3600)  # Cleanup interval in seconds
    max_tasks: int = Field(default=10000)  # Maximum tasks to store
    enable_history: bool = Field(default=True)  # Store task history
    enable_artifacts: bool = Field(default=True)  # Store task artifacts

class A2AInMemoryTaskConfig(A2ATaskMemoryConfig):
    """Configuration for A2A in-memory task provider"""
    model_config = {"frozen": True}

    type: str = Field(default="memory", frozen=True)
    max_tasks_per_context: int = Field(default=1000)

class A2ARedisTaskConfig(A2ATaskMemoryConfig):
    """Configuration for A2A Redis task provider"""
    model_config = {"frozen": True}

    type: str = Field(default="redis", frozen=True)
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = None
    db: int = Field(default=0, ge=0)

class A2APostgresTaskConfig(A2ATaskMemoryConfig):
    """Configuration for A2A PostgreSQL task provider"""
    model_config = {"frozen": True}

    type: str = Field(default="postgres", frozen=True)
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="jaf_a2a")
    username: str = Field(default="postgres")
    password: Optional[str] = None
    ssl: bool = Field(default=False)
    table_name: str = Field(default="a2a_tasks")
    max_connections: int = Field(default=10, ge=1)

# Union type for all A2A task provider configurations
A2ATaskProviderConfig = Union[A2AInMemoryTaskConfig, A2ARedisTaskConfig, A2APostgresTaskConfig]

# Error types specific to A2A task storage

class A2ATaskError:
    """Base class for A2A task-related errors"""
    def __init__(self, message: str, code: str, provider: str, task_id: Optional[str] = None, cause: Optional[Exception] = None):
        self.message = message
        self.code = code
        self.provider = provider
        self.task_id = task_id
        self.cause = cause

    def __eq__(self, other):
        if not isinstance(other, A2ATaskError):
            return False
        return (self.message == other.message and 
                self.code == other.code and
                self.provider == other.provider and
                self.task_id == other.task_id and
                self.cause == other.cause)

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r}, provider={self.provider!r}, task_id={self.task_id!r}, cause={self.cause!r})"

class A2ATaskNotFoundError(A2ATaskError):
    """Error when an A2A task is not found"""
    def __init__(self, message: str, code: str, provider: str, task_id: str, cause: Optional[Exception] = None):
        super().__init__(message, code, provider, task_id, cause)

class A2ATaskStorageError(A2ATaskError):
    """Error for A2A task storage operation failures"""
    def __init__(self, message: str, code: str, provider: str, operation: str, task_id: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message, code, provider, task_id, cause)
        self.operation = operation

    def __eq__(self, other):
        if not isinstance(other, A2ATaskStorageError):
            return False
        return super().__eq__(other) and self.operation == other.operation

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r}, provider={self.provider!r}, operation={self.operation!r}, task_id={self.task_id!r}, cause={self.cause!r})"

# Union of all possible A2A task errors
A2ATaskErrorUnion = Union[A2ATaskError, A2ATaskNotFoundError, A2ATaskStorageError]

# A2A-specific Result type for task operations
A2AResult = Union[Success[T], Failure[A2ATaskErrorUnion]]

# Error factory functions

def create_a2a_task_error(
    message: str,
    code: str,
    provider: str,
    task_id: Optional[str] = None,
    cause: Optional[Exception] = None
) -> A2ATaskError:
    """Create an A2A task error"""
    return A2ATaskError(
        message=message,
        code=code,
        provider=provider,
        task_id=task_id,
        cause=cause
    )

def create_a2a_task_not_found_error(
    task_id: str,
    provider: str
) -> A2ATaskNotFoundError:
    """Create an A2A task not found error"""
    return A2ATaskNotFoundError(
        message=f"A2A task {task_id} not found",
        code="TASK_NOT_FOUND",
        provider=provider,
        task_id=task_id
    )

def create_a2a_task_storage_error(
    operation: str,
    provider: str,
    task_id: Optional[str] = None,
    cause: Optional[Exception] = None
) -> A2ATaskStorageError:
    """Create an A2A task storage error"""
    message = f"Failed to {operation} A2A task"
    if task_id:
        message += f" {task_id}"
    message += f" in {provider}"

    return A2ATaskStorageError(
        message=message,
        code="STORAGE_ERROR",
        provider=provider,
        operation=operation,
        task_id=task_id,
        cause=cause
    )

# Error checking functions

def is_a2a_task_error(error: Any) -> bool:
    """Check if error is an A2A task error"""
    return isinstance(error, (A2ATaskError, A2ATaskNotFoundError, A2ATaskStorageError))

def is_a2a_task_not_found_error(error: Any) -> bool:
    """Check if error is an A2A task not found error"""
    return isinstance(error, A2ATaskNotFoundError)

def is_a2a_task_storage_error(error: Any) -> bool:
    """Check if error is an A2A task storage error"""
    return isinstance(error, A2ATaskStorageError)

# A2A-specific Result factory functions

def create_a2a_success(data: T) -> Success[T]:
    """Create an A2A success result"""
    return Success(data=data)

def create_a2a_failure(error: A2ATaskErrorUnion) -> Failure[A2ATaskErrorUnion]:
    """Create an A2A failure result"""
    return Failure(error=error)
