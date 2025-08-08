"""
Error types and handling for the JAF framework.

This module defines all error types used throughout the framework,
maintaining consistency and providing clear error information.
"""

from typing import List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

class ErrorType(str, Enum):
    """Enumeration of error types in the framework."""
    MAX_TURNS_EXCEEDED = "MaxTurnsExceeded"
    MODEL_BEHAVIOR_ERROR = "ModelBehaviorError"
    DECODE_ERROR = "DecodeError"
    INPUT_GUARDRAIL_TRIPWIRE = "InputGuardrailTripwire"
    OUTPUT_GUARDRAIL_TRIPWIRE = "OutputGuardrailTripwire"
    TOOL_CALL_ERROR = "ToolCallError"
    HANDOFF_ERROR = "HandoffError"
    AGENT_NOT_FOUND = "AgentNotFound"

@dataclass(frozen=True)
class FAFError:
    """Base error class for all JAF framework errors."""
    error_type: ErrorType
    message: str
    details: Dict[str, Any] = None
    
    def __str__(self) -> str:
        return f"{self.error_type.value}: {self.message}"

class JAFException(Exception):
    """Exception wrapper for FAF errors."""
    
    def __init__(self, error: FAFError):
        self.error = error
        super().__init__(str(error))

# Convenience functions for creating specific error types
def max_turns_exceeded_error(turns: int) -> FAFError:
    """Create a max turns exceeded error."""
    return FAFError(
        error_type=ErrorType.MAX_TURNS_EXCEEDED,
        message=f"Maximum number of turns exceeded: {turns}",
        details={"turns": turns}
    )

def model_behavior_error(detail: str) -> FAFError:
    """Create a model behavior error."""
    return FAFError(
        error_type=ErrorType.MODEL_BEHAVIOR_ERROR,
        message=f"Model behavior error: {detail}",
        details={"detail": detail}
    )

def decode_error(errors: List[Dict[str, Any]]) -> FAFError:
    """Create a decode error."""
    return FAFError(
        error_type=ErrorType.DECODE_ERROR,
        message="Failed to decode output",
        details={"errors": errors}
    )

def input_guardrail_error(reason: str) -> FAFError:
    """Create an input guardrail error."""
    return FAFError(
        error_type=ErrorType.INPUT_GUARDRAIL_TRIPWIRE,
        message=f"Input guardrail failed: {reason}",
        details={"reason": reason}
    )

def output_guardrail_error(reason: str) -> FAFError:
    """Create an output guardrail error."""
    return FAFError(
        error_type=ErrorType.OUTPUT_GUARDRAIL_TRIPWIRE,
        message=f"Output guardrail failed: {reason}",
        details={"reason": reason}
    )

def tool_call_error(tool: str, detail: str) -> FAFError:
    """Create a tool call error."""
    return FAFError(
        error_type=ErrorType.TOOL_CALL_ERROR,
        message=f"Tool call failed for {tool}: {detail}",
        details={"tool": tool, "detail": detail}
    )

def handoff_error(detail: str) -> FAFError:
    """Create a handoff error."""
    return FAFError(
        error_type=ErrorType.HANDOFF_ERROR,
        message=f"Handoff failed: {detail}",
        details={"detail": detail}
    )

def agent_not_found_error(agent_name: str) -> FAFError:
    """Create an agent not found error."""
    return FAFError(
        error_type=ErrorType.AGENT_NOT_FOUND,
        message=f"Agent not found: {agent_name}",
        details={"agent_name": agent_name}
    )