"""
Handoff system for JAF framework.

This module provides a simple, elegant handoff mechanism that allows agents
to seamlessly transfer control to other agents with clean state management.
"""

import json
from typing import Any, Optional, TypeVar
from dataclasses import dataclass

from .types import Tool, ToolSchema, ToolSource

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None

Ctx = TypeVar("Ctx")


def _create_handoff_json(agent_name: str, message: str = "") -> str:
    """Create the JSON structure for handoff requests."""
    return json.dumps(
        {
            "handoff_to": agent_name,
            "message": message or f"Handing off to {agent_name}",
            "type": "handoff",
        }
    )


if BaseModel is not None and Field is not None:

    class _HandoffInput(BaseModel):
        """Input parameters for handoff tool (Pydantic model)."""

        agent_name: str = Field(description="Name of the agent to hand off to")
        message: str = Field(description="Message or context to pass to the target agent")
else:

    class _HandoffInput(object):
        """Plain-Python fallback for handoff input when Pydantic is unavailable.

        This class intentionally does not call Field() so it is safe to import
        when Pydantic is not installed.
        """

        agent_name: str
        message: str

        def __init__(self, agent_name: str, message: str = ""):
            self.agent_name = agent_name
            self.message = message


HandoffInput = _HandoffInput


@dataclass
class HandoffResult:
    """Result of a handoff operation."""

    target_agent: str
    message: str
    success: bool = True
    error: Optional[str] = None


class HandoffTool:
    """A tool that enables agents to hand off to other agents."""

    def __init__(self):
        # Create schema
        if BaseModel:
            parameters_model = HandoffInput
        else:
            # Fallback schema when Pydantic is not available
            parameters_model = {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent to hand off to",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message or context to pass to the target agent",
                    },
                },
                "required": ["agent_name", "message"],
            }

        self.schema = ToolSchema(
            name="handoff",
            description="Hand off the conversation to another agent",
            parameters=parameters_model,
        )
        self.source = ToolSource.NATIVE
        self.metadata = {"type": "handoff", "system": True}

    async def execute(self, args: HandoffInput, context: Any) -> str:
        """
        Execute the handoff.

        Parameters:
            args (HandoffInput): The handoff input arguments.
            context (Any): Context containing current agent and run state information.
        """
        # Extract arguments
        if hasattr(args, "agent_name"):
            agent_name = args.agent_name
            message = args.message
        elif isinstance(args, dict):
            agent_name = args.get("agent_name", "")
            message = args.get("message", "")
        else:
            return json.dumps(
                {
                    "error": "invalid_handoff_args",
                    "message": "Invalid handoff arguments provided",
                    "usage": "handoff(agent_name='target_agent', message='optional context')",
                }
            )

        if not agent_name:
            return json.dumps(
                {
                    "error": "missing_agent_name",
                    "message": "Agent name is required for handoff",
                    "usage": "handoff(agent_name='target_agent', message='optional context')",
                }
            )

        # Add agent validation if we have access to current agent info
        if context and hasattr(context, "current_agent"):
            current_agent = context.current_agent
            if current_agent.handoffs and agent_name not in current_agent.handoffs:
                return json.dumps(
                    {
                        "error": "handoff_not_allowed",
                        "message": f"Agent {current_agent.name} cannot handoff to {agent_name}",
                        "allowed_handoffs": current_agent.handoffs,
                    }
                )

        # Return the special handoff JSON that the engine recognizes
        return _create_handoff_json(agent_name, message)


def create_handoff_tool() -> Tool:
    """Create a handoff tool that can be added to any agent."""
    return HandoffTool()


handoff_tool = create_handoff_tool()


def handoff(agent_name: str, message: str = "") -> str:
    """
    Simple function to perform a handoff (for use in agent tools).

    Args:
        agent_name: Name of the agent to hand off to
        message: Optional message to pass to the target agent

    Returns:
        JSON string that triggers a handoff
    """
    return _create_handoff_json(agent_name, message)


def is_handoff_request(result: str) -> bool:
    """
    Check if a tool result is a handoff request.

    Args:
        result: Tool execution result

    Returns:
        True if the result is a handoff request
    """
    try:
        parsed = json.loads(result)
        return isinstance(parsed, dict) and "handoff_to" in parsed
    except (json.JSONDecodeError, TypeError):
        return False


def extract_handoff_target(result: str) -> Optional[str]:
    """
    Extract the target agent name from a handoff result.

    Args:
        result: Tool execution result

    Returns:
        Target agent name if it's a handoff, None otherwise
    """
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and "handoff_to" in parsed:
            return parsed["handoff_to"]
    except (json.JSONDecodeError, TypeError):
        pass
    return None
