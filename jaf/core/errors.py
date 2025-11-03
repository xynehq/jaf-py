"""
Error handling utilities for the JAF framework.

This module provides error formatting, classification, and creation utilities
for consistent error handling throughout the framework.
"""

import json
from typing import Any, Literal

from .types import (
    AgentNotFound,
    DecodeError,
    HandoffError,
    InputGuardrailTripwire,
    JAFError,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwire,
    ToolCallError,
)


class JAFErrorHandler:
    """Handler for formatting and classifying JAF errors."""

    @staticmethod
    def format_error(error: JAFError) -> str:
        """
        Format an error into a human-readable string.

        Args:
            error: The JAF error to format

        Returns:
            Formatted error message
        """
        if isinstance(error, MaxTurnsExceeded):
            return f"Maximum turns exceeded: {error.turns} turns completed"

        elif isinstance(error, ModelBehaviorError):
            return f"Model behavior error: {error.detail}"

        elif isinstance(error, DecodeError):
            if error.errors:
                issues = []
                for e in error.errors:
                    if isinstance(e, dict):
                        message = e.get("message", "Unknown error")
                        if "path" in e:
                            path = (
                                ".".join(str(p) for p in e["path"])
                                if isinstance(e["path"], list)
                                else str(e["path"])
                            )
                            issues.append(f"{path}: {message}")
                        else:
                            issues.append(message)
                    else:
                        issues.append(str(e))
                return f"Decode error: {', '.join(issues)}"
            return "Decode error: Invalid output format"

        elif isinstance(error, InputGuardrailTripwire):
            return f"Input guardrail triggered: {error.reason}"

        elif isinstance(error, OutputGuardrailTripwire):
            return f"Output guardrail triggered: {error.reason}"

        elif isinstance(error, ToolCallError):
            return f"Tool call error in {error.tool}: {error.detail}"

        elif isinstance(error, HandoffError):
            return f"Handoff error: {error.detail}"

        elif isinstance(error, AgentNotFound):
            return f"Agent not found: {error.agent_name}"

        else:
            return f"Unknown error: {json.dumps(error.__dict__, default=str)}"

    @staticmethod
    def is_retryable(error: JAFError) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: The JAF error to check

        Returns:
            True if the error is retryable, False otherwise
        """
        if isinstance(error, (ModelBehaviorError, ToolCallError)):
            return True

        elif isinstance(
            error,
            (
                MaxTurnsExceeded,
                DecodeError,
                InputGuardrailTripwire,
                OutputGuardrailTripwire,
                HandoffError,
                AgentNotFound,
            ),
        ):
            return False

        else:
            return False

    @staticmethod
    def get_severity(error: JAFError) -> Literal["low", "medium", "high", "critical"]:
        """
        Get the severity level of an error.

        Args:
            error: The JAF error to classify

        Returns:
            Severity level: 'low', 'medium', 'high', or 'critical'
        """
        if isinstance(error, (ModelBehaviorError, ToolCallError)):
            return "medium"

        elif isinstance(error, DecodeError):
            return "high"

        elif isinstance(error, MaxTurnsExceeded):
            return "low"

        elif isinstance(error, (InputGuardrailTripwire, OutputGuardrailTripwire)):
            return "high"

        elif isinstance(error, (HandoffError, AgentNotFound)):
            return "critical"

        else:
            return "medium"


def create_jaf_error(tag: str, details: Any) -> JAFError:
    """
    Create a JAF error from a tag and details.

    Args:
        tag: The error tag/type
        details: Error details (can be dict or simple value)

    Returns:
        Appropriate JAF error instance

    Raises:
        ValueError: If the error tag is unknown
    """
    # Normalize details to dict if it's a simple value
    if not isinstance(details, dict):
        if isinstance(details, str):
            details = {"detail": details}
        else:
            details = {"value": details}

    if tag == "MaxTurnsExceeded":
        return MaxTurnsExceeded(turns=details.get("turns", 0))

    elif tag == "ModelBehaviorError":
        return ModelBehaviorError(detail=details.get("detail", str(details)))

    elif tag == "DecodeError":
        return DecodeError(errors=details.get("errors", []))

    elif tag == "InputGuardrailTripwire":
        return InputGuardrailTripwire(reason=details.get("reason", str(details)))

    elif tag == "OutputGuardrailTripwire":
        return OutputGuardrailTripwire(reason=details.get("reason", str(details)))

    elif tag == "ToolCallError":
        return ToolCallError(
            tool=details.get("tool", ""), detail=details.get("detail", str(details))
        )

    elif tag == "HandoffError":
        return HandoffError(detail=details.get("detail", str(details)))

    elif tag == "AgentNotFound":
        return AgentNotFound(
            agent_name=details.get("agentName", details.get("agent_name", str(details)))
        )

    else:
        raise ValueError(f"Unknown error tag: {tag}")
