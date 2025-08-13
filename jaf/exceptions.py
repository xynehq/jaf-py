"""
Central exceptions module for the JAF framework.

This module provides all custom exception classes for consistent error handling
and better error categorization throughout the framework.
"""

from typing import Any, Dict, List, Optional


# Base JAF exception
class JAFException(Exception):
    """Base exception for all JAF-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Agent and execution errors
class AgentException(JAFException):
    """Base exception for agent-related errors."""
    pass


class AgentNotFoundError(AgentException):
    """Raised when a requested agent is not found."""

    def __init__(self, agent_name: str):
        super().__init__(f"Agent not found: {agent_name}", {"agent_name": agent_name})
        self.agent_name = agent_name


class HandoffError(AgentException):
    """Raised when agent handoff fails."""

    def __init__(self, message: str, source_agent: Optional[str] = None, target_agent: Optional[str] = None):
        details = {}
        if source_agent:
            details["source_agent"] = source_agent
        if target_agent:
            details["target_agent"] = target_agent
        super().__init__(message, details)


# Tool execution errors
class ToolException(JAFException):
    """Base exception for tool-related errors."""
    pass


class ToolExecutionError(ToolException):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str, cause: Optional[Exception] = None):
        super().__init__(f"Tool '{tool_name}' execution failed: {message}", {
            "tool_name": tool_name,
            "cause": str(cause) if cause else None
        })
        self.tool_name = tool_name
        self.cause = cause


class ToolValidationError(ToolException):
    """Raised when tool arguments fail validation."""

    def __init__(self, tool_name: str, validation_errors: List[str]):
        message = f"Tool '{tool_name}' validation failed: {'; '.join(validation_errors)}"
        super().__init__(message, {
            "tool_name": tool_name,
            "validation_errors": validation_errors
        })
        self.tool_name = tool_name
        self.validation_errors = validation_errors


# Model and provider errors
class ModelException(JAFException):
    """Base exception for model-related errors."""
    pass


class ModelProviderError(ModelException):
    """Raised when model provider encounters an error."""

    def __init__(self, provider: str, message: str, status_code: Optional[int] = None):
        super().__init__(f"Model provider '{provider}' error: {message}", {
            "provider": provider,
            "status_code": status_code
        })
        self.provider = provider
        self.status_code = status_code


class ModelResponseError(ModelException):
    """Raised when model response is invalid or cannot be parsed."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(f"Model response error: {message}", {
            "raw_response": raw_response
        })
        self.raw_response = raw_response


# Validation and guardrail errors
class ValidationException(JAFException):
    """Base exception for validation errors."""
    pass


class GuardrailViolationError(ValidationException):
    """Raised when input or output violates a guardrail."""

    def __init__(self, guardrail_type: str, message: str, violation_details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Guardrail violation ({guardrail_type}): {message}", {
            "guardrail_type": guardrail_type,
            "violation_details": violation_details or {}
        })
        self.guardrail_type = guardrail_type
        self.violation_details = violation_details or {}


class InputValidationError(ValidationException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field_errors: Optional[Dict[str, List[str]]] = None):
        super().__init__(f"Input validation error: {message}", {
            "field_errors": field_errors or {}
        })
        self.field_errors = field_errors or {}


# Memory system errors
class MemoryException(JAFException):
    """Base exception for memory system errors."""
    pass


class MemoryConnectionError(MemoryException):
    """Raised when memory provider connection fails."""

    def __init__(self, provider: str, message: str):
        super().__init__(f"Memory connection error ({provider}): {message}", {
            "provider": provider
        })
        self.provider = provider


class MemoryStorageError(MemoryException):
    """Raised when memory storage operation fails."""

    def __init__(self, operation: str, message: str, conversation_id: Optional[str] = None):
        details = {"operation": operation}
        if conversation_id:
            details["conversation_id"] = conversation_id
        super().__init__(f"Memory storage error ({operation}): {message}", details)
        self.operation = operation
        self.conversation_id = conversation_id


# Session and workflow errors
class SessionException(JAFException):
    """Base exception for session-related errors."""
    pass


class SessionStateError(SessionException):
    """Raised when session state is invalid."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        details = {}
        if session_id:
            details["session_id"] = session_id
        super().__init__(f"Session state error: {message}", details)
        self.session_id = session_id


class MaxTurnsExceededError(SessionException):
    """Raised when maximum number of turns is exceeded."""

    def __init__(self, max_turns: int, current_turns: int):
        super().__init__(f"Maximum turns exceeded: {current_turns}/{max_turns}", {
            "max_turns": max_turns,
            "current_turns": current_turns
        })
        self.max_turns = max_turns
        self.current_turns = current_turns


# A2A protocol errors
class A2AException(JAFException):
    """Base exception for A2A protocol errors."""
    pass


class A2AProtocolError(A2AException):
    """Raised when A2A protocol operation fails."""

    def __init__(self, message: str, method: Optional[str] = None, context_id: Optional[str] = None):
        details = {}
        if method:
            details["method"] = method
        if context_id:
            details["context_id"] = context_id
        super().__init__(f"A2A protocol error: {message}", details)
        self.method = method
        self.context_id = context_id


class A2ATaskError(A2AException):
    """Raised when A2A task operation fails."""

    def __init__(self, message: str, task_id: Optional[str] = None):
        details = {}
        if task_id:
            details["task_id"] = task_id
        super().__init__(f"A2A task error: {message}", details)
        self.task_id = task_id


# Configuration errors
class ConfigurationException(JAFException):
    """Base exception for configuration errors."""
    pass


class InvalidConfigurationError(ConfigurationException):
    """Raised when configuration is invalid."""

    def __init__(self, config_type: str, message: str, config_errors: Optional[List[str]] = None):
        super().__init__(f"Invalid {config_type} configuration: {message}", {
            "config_type": config_type,
            "config_errors": config_errors or []
        })
        self.config_type = config_type
        self.config_errors = config_errors or []


# Convenience functions for creating common exceptions
def create_agent_error(message: str, agent_name: Optional[str] = None) -> AgentException:
    """Create an agent-related error."""
    if agent_name and "not found" in message.lower():
        return AgentNotFoundError(agent_name)
    return AgentException(message)


def create_tool_error(tool_name: str, message: str, cause: Optional[Exception] = None) -> ToolException:
    """Create a tool-related error."""
    if "validation" in message.lower():
        return ToolValidationError(tool_name, [message])
    return ToolExecutionError(tool_name, message, cause)


def create_session_error(message: str, session_id: Optional[str] = None) -> SessionException:
    """Create a session-related error."""
    if "max turns" in message.lower() or "maximum turns" in message.lower():
        # Extract numbers if possible
        import re
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            return MaxTurnsExceededError(int(numbers[0]), int(numbers[1]))
        return MaxTurnsExceededError(10, 10)  # Default fallback
    return SessionStateError(message, session_id)


def create_memory_error(message: str, provider: Optional[str] = None) -> MemoryException:
    """Create a memory-related error."""
    if "connection" in message.lower():
        return MemoryConnectionError(provider or "unknown", message)
    return MemoryException(message)


# Backward compatibility - maintain old error classes for existing code

# Re-export for backward compatibility
JAFError = JAFException
