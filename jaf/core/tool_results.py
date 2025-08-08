"""
Standardized tool result types for consistent error handling.

This module provides a consistent interface for tool execution results,
error handling, and metadata tracking.
"""

from typing import Any, Dict, List, Optional, Callable, Awaitable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pydantic import BaseModel, Field

T = TypeVar('T')
TArgs = TypeVar('TArgs')
TResult = TypeVar('TResult')
TContext = TypeVar('TContext')

class ToolResultStatus(str, Enum):
    """Status values for tool execution results."""
    SUCCESS = 'success'
    ERROR = 'error'
    VALIDATION_ERROR = 'validation_error'
    PERMISSION_DENIED = 'permission_denied'
    NOT_FOUND = 'not_found'

@dataclass(frozen=True)
class ToolErrorInfo:
    """Error information for tool results."""
    code: str
    message: str
    details: Optional[Any] = None

@dataclass(frozen=True)
class ToolResult(Generic[T]):
    """Standardized tool result with status, data, and metadata."""
    status: ToolResultStatus
    data: Optional[T] = None
    error: Optional[ToolErrorInfo] = None
    metadata: Optional[Dict[str, Any]] = None

class ToolErrorCodes:
    """Common error codes for tool execution."""
    # Validation errors
    INVALID_INPUT = 'INVALID_INPUT'
    MISSING_REQUIRED_FIELD = 'MISSING_REQUIRED_FIELD'
    INVALID_FORMAT = 'INVALID_FORMAT'
    
    # Permission errors
    PERMISSION_DENIED = 'PERMISSION_DENIED'
    INSUFFICIENT_PERMISSIONS = 'INSUFFICIENT_PERMISSIONS'
    
    # Resource errors
    NOT_FOUND = 'NOT_FOUND'
    RESOURCE_UNAVAILABLE = 'RESOURCE_UNAVAILABLE'
    
    # Execution errors
    EXECUTION_FAILED = 'EXECUTION_FAILED'
    TIMEOUT = 'TIMEOUT'
    EXTERNAL_SERVICE_ERROR = 'EXTERNAL_SERVICE_ERROR'
    
    # Generic
    UNKNOWN_ERROR = 'UNKNOWN_ERROR'

class ToolResponse:
    """Helper functions for creating standardized tool results."""
    
    @staticmethod
    def success(data: T, metadata: Optional[Dict[str, Any]] = None) -> ToolResult[T]:
        """Create a successful tool result."""
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data=data,
            metadata=metadata
        )
    
    @staticmethod
    def error(
        code: str,
        message: str,
        details: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult[None]:
        """Create an error tool result."""
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=ToolErrorInfo(code=code, message=message, details=details),
            metadata=metadata
        )
    
    @staticmethod
    def validation_error(
        message: str,
        details: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult[None]:
        """Create a validation error tool result."""
        return ToolResult(
            status=ToolResultStatus.VALIDATION_ERROR,
            error=ToolErrorInfo(
                code=ToolErrorCodes.INVALID_INPUT,
                message=message,
                details=details
            ),
            metadata=metadata
        )
    
    @staticmethod
    def permission_denied(
        message: str,
        required_permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult[None]:
        """Create a permission denied tool result."""
        return ToolResult(
            status=ToolResultStatus.PERMISSION_DENIED,
            error=ToolErrorInfo(
                code=ToolErrorCodes.PERMISSION_DENIED,
                message=message,
                details={'required_permissions': required_permissions} if required_permissions else None
            ),
            metadata=metadata
        )
    
    @staticmethod
    def not_found(
        resource: str,
        identifier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult[None]:
        """Create a not found tool result."""
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        
        return ToolResult(
            status=ToolResultStatus.NOT_FOUND,
            error=ToolErrorInfo(
                code=ToolErrorCodes.NOT_FOUND,
                message=message,
                details={'resource': resource, 'identifier': identifier}
            ),
            metadata=metadata
        )

def with_error_handling(
    tool_name: str,
    executor: Callable[[TArgs, TContext], Union[TResult, Awaitable[TResult]]]
) -> Callable[[TArgs, TContext], Awaitable[ToolResult[TResult]]]:
    """
    Tool execution wrapper that provides standardized error handling.
    
    Args:
        tool_name: Name of the tool for logging and metadata
        executor: The actual tool execution function
        
    Returns:
        Wrapped function that returns a ToolResult
    """
    async def wrapper(args: TArgs, context: TContext) -> ToolResult[TResult]:
        start_time = time.time()
        
        try:
            print(f"[TOOL:{tool_name}] Starting execution with args: {args}")
            
            # Handle both sync and async executors
            if callable(executor):
                result = executor(args, context)
                if hasattr(result, '__await__'):  # Check if it's awaitable
                    result = await result
            else:
                raise ValueError(f"Invalid executor for tool {tool_name}")
            
            execution_time = int((time.time() - start_time) * 1000)
            print(f"[TOOL:{tool_name}] Completed successfully in {execution_time}ms")
            
            return ToolResponse.success(result, {
                'execution_time_ms': execution_time,
                'tool_name': tool_name
            })
            
        except Exception as error:
            execution_time = int((time.time() - start_time) * 1000)
            print(f"[TOOL:{tool_name}] Failed after {execution_time}ms: {error}")
            
            if isinstance(error, Exception):
                return ToolResponse.error(
                    ToolErrorCodes.EXECUTION_FAILED,
                    str(error),
                    {'stack': str(error.__traceback__) if error.__traceback__ else None},
                    {'execution_time_ms': execution_time, 'tool_name': tool_name}
                )
            
            return ToolResponse.error(
                ToolErrorCodes.UNKNOWN_ERROR,
                'Unknown error occurred',
                error,
                {'execution_time_ms': execution_time, 'tool_name': tool_name}
            )
    
    return wrapper

def require_permissions(
    required_permissions: List[str]
) -> Callable[[TContext], Optional[ToolResult[None]]]:
    """
    Permission checking helper.
    
    Args:
        required_permissions: List of permissions required
        
    Returns:
        Function that checks permissions and returns ToolResult if denied, None if allowed
    """
    def check_permissions(context: TContext) -> Optional[ToolResult[None]]:
        # Assume context has a 'permissions' attribute
        user_permissions = getattr(context, 'permissions', [])
        if not isinstance(user_permissions, list):
            user_permissions = []
        
        missing_permissions = [
            perm for perm in required_permissions 
            if perm not in user_permissions
        ]
        
        if missing_permissions:
            return ToolResponse.permission_denied(
                f"Missing required permissions: {', '.join(missing_permissions)}",
                required_permissions
            )
        
        return None  # No error
    
    return check_permissions

def tool_result_to_string(result: ToolResult[Any]) -> str:
    """
    Convert ToolResult to string for backward compatibility with existing tools.
    
    Args:
        result: The ToolResult to convert
        
    Returns:
        String representation of the result
    """
    if result.status == ToolResultStatus.SUCCESS:
        if isinstance(result.data, str):
            return result.data
        return json.dumps(result.data, default=str)
    
    # For errors, return a structured error message
    error_obj = {
        'error': result.status.value,
        'code': result.error.code if result.error else 'UNKNOWN',
        'message': result.error.message if result.error else 'Unknown error',
    }
    
    if result.error and result.error.details:
        error_obj['details'] = result.error.details
    
    if result.metadata:
        error_obj['metadata'] = result.metadata
    
    return json.dumps(error_obj, indent=2, default=str)