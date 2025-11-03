"""
Agent-as-tool implementation for JAF framework.

This module provides functionality to convert agents into tools that can be used
by other agents, enabling hierarchical agent orchestration patterns.
"""

import asyncio
import json
import inspect
import inspect
import contextvars
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable, TypeVar, get_type_hints

try:
    from pydantic import BaseModel, create_model
except ImportError:
    BaseModel = None
    create_model = None

from .types import (
    Agent,
    Tool,
    ToolSchema,
    ToolSource,
    RunConfig,
    RunState,
    RunResult,
    Message,
    get_text_content,
    ContentRole,
    generate_run_id,
    generate_trace_id,
)

Ctx = TypeVar("Ctx")
Out = TypeVar("Out")

# Context variable to store the current RunConfig for agent tools
_current_run_config: contextvars.ContextVar[Optional[RunConfig]] = contextvars.ContextVar(
    "current_run_config", default=None
)


def set_current_run_config(config: RunConfig) -> None:
    """Set the current RunConfig in context for agent tools to use."""
    _current_run_config.set(config)


def get_current_run_config() -> Optional[RunConfig]:
    """Get the current RunConfig from context."""
    return _current_run_config.get()


class AgentToolInput(BaseModel if BaseModel else object):
    """Input parameters for agent tools."""

    input: str

    if not BaseModel:

        def __init__(self, input: str):
            self.input = input


def create_agent_tool(
    agent: Agent[Ctx, Out],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    max_turns: Optional[int] = None,
    custom_output_extractor: Optional[
        Callable[[RunResult[Out]], Union[str, Awaitable[str]]]
    ] = None,
    is_enabled: Union[
        bool,
        Callable[[Any, Agent[Ctx, Out]], bool],
        Callable[[Any, Agent[Ctx, Out]], Awaitable[bool]],
    ] = True,
    metadata: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    preserve_session: bool = False,
) -> Tool[AgentToolInput, Ctx]:
    """
    Create a tool from an agent.

    Args:
        agent: The agent to convert into a tool
        tool_name: Optional custom name for the tool
        tool_description: Optional custom description for the tool
        max_turns: Maximum turns for agent execution
        custom_output_extractor: Optional function to extract output from RunResult
        is_enabled: Whether the tool is enabled (bool, sync function, or async function)
        metadata: Optional metadata for the tool
        timeout: Optional timeout for tool execution

    Returns:
        A Tool that wraps the agent execution
    """

    # Default names and descriptions
    final_tool_name = tool_name or f"run_{agent.name.lower().replace(' ', '_')}"
    final_tool_description = (
        tool_description or f"Execute the {agent.name} agent with the given input"
    )

    # Create the tool schema
    if BaseModel and create_model:
        # Use Pydantic if available
        parameters_model = AgentToolInput
    else:
        # Fallback schema for when Pydantic is not available
        parameters_model = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input message to send to the agent"}
            },
            "required": ["input"],
        }

    tool_schema = ToolSchema(
        name=final_tool_name,
        description=final_tool_description,
        parameters=parameters_model,
        timeout=timeout,
    )

    async def _check_if_enabled(context: Ctx) -> bool:
        """Check if the tool is enabled based on the is_enabled parameter."""
        if isinstance(is_enabled, bool):
            return is_enabled
        elif callable(is_enabled):
            result = is_enabled(context, agent)
            if hasattr(result, "__await__"):
                return await result
            return result
        return True

    async def _execute_agent_tool(args: AgentToolInput, context: Ctx) -> str:
        """Execute the agent and return the result."""
        # Check if tool is enabled
        if not await _check_if_enabled(context):
            return json.dumps(
                {
                    "error": "tool_disabled",
                    "message": f"Tool {final_tool_name} is currently disabled",
                }
            )

        # Extract input from args
        if hasattr(args, "input"):
            user_input = args.input
        elif isinstance(args, dict):
            user_input = args.get("input", "")
        else:
            user_input = str(args)

        # Create initial state for the agent
        initial_messages = [Message(role=ContentRole.USER, content=user_input)]

        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=initial_messages,
            current_agent_name=agent.name,
            context=context,
            turn_count=0,
        )

        # Get the current RunConfig from context variable
        parent_config = _current_run_config.get()
        if parent_config is None:
            # If no parent config available, we can't execute the agent
            return json.dumps(
                {
                    "error": "no_parent_config",
                    "message": f"Agent tool {final_tool_name} requires a parent RunConfig to execute. Please ensure the agent tool is called from within a JAF run context.",
                }
            )

        # Create a sub-config that inherits from parent but uses this agent
        # Session inheritance is configurable via preserve_session.
        # - When True: inherit parent's conversation_id and memory (shared memory/session)
        # - When False: do not inherit (ephemeral, per-invocation sub-agent run)
        sub_config = RunConfig(
            agent_registry={agent.name: agent, **parent_config.agent_registry},
            model_provider=parent_config.model_provider,
            max_turns=max_turns or parent_config.max_turns,
            model_override=parent_config.model_override,
            initial_input_guardrails=parent_config.initial_input_guardrails,
            final_output_guardrails=parent_config.final_output_guardrails,
            on_event=parent_config.on_event,
            memory=parent_config.memory if preserve_session else None,
            conversation_id=parent_config.conversation_id if preserve_session else None,
            default_tool_timeout=parent_config.default_tool_timeout,
            prefer_streaming=parent_config.prefer_streaming,
        )

        token = _current_run_config.set(sub_config)
        try:
            # Import here to avoid circular imports
            from . import engine

            # Execute the agent
            result = await engine.run(initial_state, sub_config)
        finally:
            _current_run_config.reset(token)

        # Output extraction and error handling
        try:
            if custom_output_extractor:
                output = custom_output_extractor(result)
                if inspect.isawaitable(output):
                    output = await output
                return str(output)
            if result.outcome.status == "completed":
                if hasattr(result.outcome, "output") and result.outcome.output is not None:
                    return str(result.outcome.output)
                else:
                    # Fall back to the last assistant message
                    from .types import get_text_content

                    assistant_messages = [
                        msg
                        for msg in result.final_state.messages
                        if msg.role == ContentRole.ASSISTANT and get_text_content(msg.content)
                    ]
                    if assistant_messages:
                        return get_text_content(assistant_messages[-1].content)
                    return "Agent completed successfully but produced no output"
            else:
                # Error case
                error_detail = getattr(result.outcome.error, "detail", str(result.outcome.error))
                return json.dumps(
                    {
                        "error": "agent_execution_failed",
                        "message": f"Agent {agent.name} failed: {error_detail}",
                    }
                )
        except Exception as e:
            return json.dumps(
                {
                    "error": "agent_tool_error",
                    "message": f"Error executing agent {agent.name}: {str(e)}",
                }
            )

    # Create the tool wrapper
    class AgentTool:
        def __init__(self):
            self.schema = tool_schema
            self.metadata = metadata or {"source": "agent", "agent_name": agent.name}
            self.source = ToolSource.NATIVE

        async def execute(self, args: AgentToolInput, context: Ctx) -> str:
            """Execute the agent tool."""
            return await _execute_agent_tool(args, context)

    return AgentTool()


def create_default_output_extractor(extract_json: bool = False) -> Callable[[RunResult], str]:
    """
    Create a default output extractor function.

    Args:
        extract_json: If True, attempts to extract JSON from the output

    Returns:
        An output extractor function
    """

    def extractor(run_result: RunResult) -> str:
        if run_result.outcome.status == "completed":
            output = run_result.outcome.output
            if extract_json and isinstance(output, str):
                try:
                    # Try to parse as JSON and re-serialize for consistency
                    parsed = json.loads(output)
                    return json.dumps(parsed)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, return the original output
                    pass
            return str(output) if output is not None else ""
        else:
            # Return error information
            error_detail = getattr(
                run_result.outcome.error, "detail", str(run_result.outcome.error)
            )
            return json.dumps({"error": True, "message": error_detail})

    return extractor


def create_json_output_extractor() -> Callable[[RunResult], str]:
    """
    Create an output extractor that specifically looks for JSON in the agent's output.

    Returns:
        An output extractor that finds and returns JSON content
    """

    def json_extractor(run_result: RunResult) -> str:
        # Scan the agent's outputs in reverse order until we find a JSON-like message
        for message in reversed(run_result.final_state.messages):
            if message.role == ContentRole.ASSISTANT and get_text_content(message.content):
                content = get_text_content(message.content).strip()
                if content.startswith("{") or content.startswith("["):
                    try:
                        # Validate it's proper JSON
                        json.loads(content)
                        return content
                    except (json.JSONDecodeError, TypeError):
                        continue

        # Fallback to empty JSON object if nothing was found
        return "{}"

    return json_extractor


# Convenience function for conditional tool enabling
def create_conditional_enabler(
    condition_key: str, expected_value: Any = True
) -> Callable[[Any, Agent], bool]:
    """
    Create a conditional enabler function based on context attributes.

    Args:
        condition_key: The key to check in the context
        expected_value: The expected value for the condition

    Returns:
        A function that checks if the tool should be enabled
    """

    def enabler(context: Any, agent: Agent) -> bool:
        return getattr(context, condition_key, None) == expected_value

    return enabler
