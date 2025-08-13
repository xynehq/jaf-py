"""
Pure functional A2A agent creation utilities
No classes, only pure functions and immutable data
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.engine import run
from ..core.types import Agent, Message, RunConfig, RunState
from .types import (
    A2AAgent,
    A2AAgentTool,
    AgentState,
    StreamEvent,
    ToolContext,
)


def create_a2a_agent(
    name: str,
    description: str,
    instruction: str,
    tools: List[A2AAgentTool],
    supported_content_types: Optional[List[str]] = None
) -> A2AAgent:
    """Pure function to create A2A compatible agent"""
    return A2AAgent(
        name=name,
        description=description,
        supportedContentTypes=supported_content_types or ["text/plain", "application/json"],
        instruction=instruction,
        tools=tools
    )


def create_a2a_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    execute_func: Callable[[Any, Optional[ToolContext]], Any]
) -> A2AAgentTool:
    """Pure function to create A2A tool"""
    return A2AAgentTool(
        name=name,
        description=description,
        parameters=parameters,
        execute=execute_func
    )


def get_processing_message(agent_name: str) -> str:
    """Pure function to get processing message"""
    return f"{agent_name} is processing your request..."


def create_initial_agent_state(session_id: str) -> AgentState:
    """Pure function to create initial agent state"""
    return AgentState(
        sessionId=session_id,
        messages=[],
        context={},
        artifacts=[],
        timestamp=datetime.now().isoformat()
    )


def add_message_to_state(state: AgentState, message: Any) -> AgentState:
    """Pure function to add message to state"""
    return AgentState(
        sessionId=state.sessionId,
        messages=[*state.messages, message],
        context=state.context,
        artifacts=state.artifacts,
        timestamp=datetime.now().isoformat()
    )


def update_state_from_run_result(state: AgentState, outcome: Any) -> AgentState:
    """Pure function to update state from run result"""
    new_artifacts = state.artifacts
    if hasattr(outcome, 'artifacts') and outcome.artifacts:
        new_artifacts = [*state.artifacts, *outcome.artifacts]

    return AgentState(
        sessionId=state.sessionId,
        messages=state.messages,
        context=state.context,
        artifacts=new_artifacts,
        timestamp=datetime.now().isoformat()
    )


def create_user_message(text: str) -> Message:
    """Pure function to create user message"""
    return Message(role="user", content=text)


def transform_a2a_agent_to_jaf(a2a_agent: A2AAgent) -> Agent:
    """Pure function to transform A2A agent to JAF agent"""

    def instructions_func(state: Any) -> str:
        return a2a_agent.instruction

    # Transform tools using functional approach
    jaf_tools = [transform_a2a_tool_to_jaf(a2a_tool) for a2a_tool in a2a_agent.tools]

    return Agent(
        name=a2a_agent.name,
        instructions=instructions_func,
        tools=jaf_tools
    )


def transform_a2a_tool_to_jaf(a2a_tool: A2AAgentTool) -> 'JAFToolImplementation':
    """Pure function to transform A2A tool to JAF tool"""

    async def execute_wrapper(args: Any, context: Any) -> Any:
        tool_context = ToolContext(
            actions={
                "requiresInput": False,
                "skipSummarization": False,
                "escalate": False
            },
            metadata=context or {}
        )

        result = await a2a_tool.execute(args, tool_context)

        # Handle ToolResult format
        if isinstance(result, dict) and "result" in result:
            return result["result"]

        return result

    # Create ToolSchema object
    from ..core.types import ToolSchema
    tool_schema = ToolSchema(
        name=a2a_tool.name,
        description=a2a_tool.description,
        parameters=a2a_tool.parameters
    )

    return JAFToolImplementation(
        schema=tool_schema,
        execute=execute_wrapper
    )


class JAFToolImplementation:
    """Concrete implementation of the Tool protocol"""

    def __init__(self, schema, execute: Callable):
        self._schema = schema
        self._execute = execute
        self.name = schema.name  # Add name attribute for JAF engine
        self.description = schema.description
        self.parameters = schema.parameters

    @property
    def schema(self):
        return self._schema

    async def execute(self, args: Any, context: Any) -> Any:
        return await self._execute(args, context)


def create_run_config_for_a2a_agent(
    a2a_agent: A2AAgent,
    model_provider: Any
) -> RunConfig:
    """Pure function to create run configuration for A2A agent"""
    jaf_agent = transform_a2a_agent_to_jaf(a2a_agent)

    def event_handler(event: Any) -> None:
        # Handle different event types properly
        if hasattr(event, 'data'):
            event_data = event.data
            event_type = type(event).__name__
        elif isinstance(event, dict):
            event_data = event.get('data', '')
            event_type = event.get('type', 'unknown')
        else:
            event_data = str(event)
            event_type = type(event).__name__

        print(f"[A2A:{a2a_agent.name}] {event_type}: {event_data}")

    return RunConfig(
        agent_registry={a2a_agent.name: jaf_agent},
        model_provider=model_provider,
        max_turns=10,
        on_event=event_handler
    )


def transform_to_run_state(
    state: AgentState,
    agent_name: str,
    context: Optional[Dict[str, Any]] = None
) -> RunState:
    """Pure function to transform agent state to JAF run state"""
    from ..core.types import generate_run_id, generate_trace_id

    # Convert A2A messages to JAF Message format
    jaf_messages = [
        create_user_message(msg.content if hasattr(msg, 'content') else str(msg))
        for msg in state.messages
        if msg is not None
    ]

    return RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=jaf_messages,
        current_agent_name=agent_name,
        context=context or {},
        turn_count=0
    )


async def process_agent_query(
    agent: A2AAgent,
    query: str,
    state: AgentState,
    model_provider: Any
) -> AsyncGenerator[StreamEvent, None]:
    """Pure async generator function to process agent query with improved streaming"""
    start_time = time.time()

    # Transform query to JAF message format
    user_message = create_user_message(query)
    new_state = add_message_to_state(state, user_message)

    # Create JAF configuration
    run_config = create_run_config_for_a2a_agent(agent, model_provider)
    run_state = transform_to_run_state(new_state, agent.name)

    # Yield initial processing event
    yield StreamEvent(
        isTaskComplete=False,
        content="Starting query processing...",
        new_state=new_state.model_dump(),
        timestamp=datetime.now().isoformat(),
        updates="Initializing agent state",
        metrics={"start_time": start_time, "status": "starting"}
    )

    try:
        # Add small delay for metrics tracking
        await asyncio.sleep(0.001)

        # Yield progress update
        processing_time = time.time()
        yield StreamEvent(
            isTaskComplete=False,
            content="Processing with JAF engine...",
            new_state=new_state.model_dump(),
            timestamp=datetime.now().isoformat(),
            updates="Executing agent logic",
            metrics={
                "start_time": start_time,
                "processing_time": processing_time,
                "status": "processing"
            }
        )

        # Execute JAF engine (pure function)
        result = await run(run_state, run_config)
        completion_time = time.time()

        if hasattr(result.outcome, 'status') and result.outcome.status == 'completed':
            final_state = update_state_from_run_result(new_state, result.outcome)
            yield StreamEvent(
                isTaskComplete=True,
                content=getattr(result.outcome, 'output', 'Task completed'),
                new_state=final_state.model_dump(),
                timestamp=datetime.now().isoformat(),
                updates="Task completed successfully",
                metrics={
                    "start_time": start_time,
                    "completion_time": completion_time,
                    "total_duration": completion_time - start_time,
                    "status": "completed"
                }
            )
        else:
            final_state = update_state_from_run_result(new_state, result.outcome)
            error_content = getattr(result.outcome, 'error', 'Unknown error')
            yield StreamEvent(
                isTaskComplete=True,
                content=f"Error: {json.dumps(error_content) if isinstance(error_content, dict) else str(error_content)}",
                new_state=final_state.model_dump(),
                timestamp=datetime.now().isoformat(),
                updates="Task completed with error",
                metrics={
                    "start_time": start_time,
                    "completion_time": completion_time,
                    "total_duration": completion_time - start_time,
                    "status": "error"
                }
            )
    except Exception as error:
        error_time = time.time()

        # Log the actual error for debugging but don't expose internal details
        import logging
        logging.error(f"Agent execution error for {agent.name}: {error!s}", exc_info=True)

        # Determine error type and provide safe error message
        if "timeout" in str(error).lower():
            safe_error = "Request timeout - please try again"
        elif "connection" in str(error).lower():
            safe_error = "Connection error - please check connectivity"
        elif "validation" in str(error).lower():
            safe_error = f"Validation error: {error!s}"
        elif "permission" in str(error).lower() or "unauthorized" in str(error).lower():
            safe_error = "Insufficient permissions for this operation"
        else:
            safe_error = "An internal error occurred while processing your request"

        yield StreamEvent(
            isTaskComplete=True,
            content=f"Error: {safe_error}",
            new_state=new_state.model_dump(),
            timestamp=datetime.now().isoformat(),
            updates="Task failed with error",
            metrics={
                "start_time": start_time,
                "error_time": error_time,
                "total_duration": error_time - start_time,
                "status": "failed",
                "error_type": type(error).__name__
            }
        )


def extract_text_from_a2a_message(message: Dict[str, Any]) -> str:
    """Pure function to extract text from A2A message"""
    if not message or not message.get("parts"):
        return ""

    # Use functional approach instead of mutation
    text_parts = [
        part.get("text", "")
        for part in message["parts"]
        if part.get("kind") == "text"
    ]

    return "\n".join(text_parts)


def create_a2a_text_message(
    text: str,
    context_id: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """Pure function to create A2A text message"""
    return {
        "role": "agent",
        "parts": [{"kind": "text", "text": text}],
        "messageId": f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "contextId": context_id,
        "taskId": task_id,
        "kind": "message",
        "timestamp": datetime.now().isoformat()
    }


def create_a2a_data_message(
    data: Dict[str, Any],
    context_id: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """Pure function to create A2A data message"""
    return {
        "role": "agent",
        "parts": [{"kind": "data", "data": data}],
        "messageId": f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "contextId": context_id,
        "taskId": task_id,
        "kind": "message",
        "timestamp": datetime.now().isoformat()
    }


def create_a2a_task(
    message: Dict[str, Any],
    context_id: Optional[str] = None
) -> Dict[str, Any]:
    """Pure function to create A2A task"""
    task_id = f"task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    current_time = datetime.now().isoformat()

    return {
        "id": task_id,
        "contextId": context_id or f"ctx_{int(time.time() * 1000)}",
        "status": {
            "state": "submitted",
            "timestamp": current_time
        },
        "history": [message],
        "artifacts": [],
        "kind": "task"
    }


def update_a2a_task_status(
    task: Dict[str, Any],
    state: str,
    message: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pure function to update A2A task status"""
    updated_task = task.copy()
    updated_task["status"] = {
        "state": state,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    return updated_task


def add_artifact_to_a2a_task(
    task: Dict[str, Any],
    parts: List[Dict[str, Any]],
    name: str
) -> Dict[str, Any]:
    """Pure function to add artifact to A2A task"""
    artifact = {
        "artifactId": f"artifact_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "name": name,
        "parts": parts,
        "timestamp": datetime.now().isoformat()
    }

    updated_task = task.copy()
    updated_task["artifacts"] = [*task.get("artifacts", []), artifact]
    return updated_task


def complete_a2a_task(task: Dict[str, Any], result: Optional[Any] = None) -> Dict[str, Any]:
    """Pure function to complete A2A task"""
    updated_task = update_a2a_task_status(task, "completed")

    if result is not None:
        result_artifact = {
            "artifactId": f"result_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            "name": "final_result",
            "parts": [{"kind": "text", "text": str(result)}],
            "timestamp": datetime.now().isoformat()
        }

        updated_task["artifacts"] = [*task.get("artifacts", []), result_artifact]

    return updated_task


# Execution functions

async def execute_a2a_agent(
    context: Dict[str, Any],
    agent: A2AAgent,
    model_provider: Any
) -> Dict[str, Any]:
    """Execute A2A agent and return result"""
    message = context.get("message", {})
    query = extract_text_from_a2a_message(message)
    session_id = context.get("session_id", "default_session")

    # Create task if none exists
    current_task = context.get("current_task")
    if not current_task:
        current_task = create_a2a_task(message, session_id)

    events = []

    try:
        # Process through agent
        agent_state = create_initial_agent_state(session_id)

        processing_events = []
        async for event in process_agent_query(agent, query, agent_state, model_provider):
            processing_events = [*processing_events, event.model_dump()]

            if event.isTaskComplete:
                # Handle final result
                if isinstance(event.content, str) and event.content.startswith("Error: "):
                    # Error case
                    error_message = event.content[7:]  # Remove "Error: " prefix
                    updated_task = update_a2a_task_status(
                        current_task,
                        "failed",
                        create_a2a_text_message(error_message, session_id, current_task["id"])
                    )

                    return {
                        "events": [*events, *processing_events],
                        "final_task": updated_task,
                        "error": error_message
                    }
                else:
                    # Success case
                    content = event.content
                    updated_task = add_artifact_to_a2a_task(
                        current_task,
                        [{"kind": "text", "text": str(content)}],
                        "response"
                    )
                    updated_task = complete_a2a_task(updated_task, content)

                    return {
                        "events": [*events, *processing_events],
                        "final_task": updated_task
                    }

        # If we reach here without completion, mark as failed
        updated_task = update_a2a_task_status(current_task, "failed")
        return {
            "events": [*events, *processing_events],
            "final_task": updated_task,
            "error": "Agent processing did not complete"
        }

    except Exception as error:
        # Log the actual error for debugging but don't expose internal details
        import logging
        logging.error(f"Agent execution error for {agent.name}: {error!s}", exc_info=True)

        # Determine error type and provide safe error message
        if "timeout" in str(error).lower():
            safe_error = "Request timeout - please try again"
        elif "connection" in str(error).lower():
            safe_error = "Connection error - please check connectivity"
        elif "validation" in str(error).lower():
            safe_error = f"Validation error: {error!s}"
        elif "permission" in str(error).lower() or "unauthorized" in str(error).lower():
            safe_error = "Insufficient permissions for this operation"
        else:
            safe_error = "An internal error occurred while processing your request"

        failed_task = update_a2a_task_status(
            current_task,
            "failed",
            create_a2a_text_message(safe_error, session_id, current_task["id"])
        )

        return {
            "events": events,
            "final_task": failed_task,
            "error": safe_error
        }


async def execute_a2a_agent_with_streaming(
    context: Dict[str, Any],
    agent: A2AAgent,
    model_provider: Any
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute A2A agent with streaming"""
    message = context.get("message", {})
    query = extract_text_from_a2a_message(message)
    session_id = context.get("session_id", "default_session")

    # Create task if none exists
    current_task = context.get("current_task")
    if not current_task:
        current_task = create_a2a_task(message, session_id)

        yield {
            "kind": "status-update",
            "taskId": current_task["id"],
            "contextId": current_task["contextId"],
            "status": {"state": "submitted", "timestamp": datetime.now().isoformat()},
            "final": False
        }

    yield {
        "kind": "status-update",
        "taskId": current_task["id"],
        "contextId": current_task["contextId"],
        "status": {"state": "working", "timestamp": datetime.now().isoformat()},
        "final": False
    }

    try:
        agent_state = create_initial_agent_state(session_id)

        async for event in process_agent_query(agent, query, agent_state, model_provider):
            if not event.isTaskComplete:
                yield {
                    "kind": "status-update",
                    "taskId": current_task["id"],
                    "contextId": current_task["contextId"],
                    "status": {
                        "state": "working",
                        "message": create_a2a_text_message(
                            event.updates or "Processing...",
                            current_task["contextId"],
                            current_task["id"]
                        ),
                        "timestamp": event.timestamp
                    },
                    "final": False
                }
            else:
                # Handle final result
                if isinstance(event.content, str) and event.content.startswith("Error: "):
                    error_message = event.content[7:]
                    yield {
                        "kind": "status-update",
                        "taskId": current_task["id"],
                        "contextId": current_task["contextId"],
                        "status": {
                            "state": "failed",
                            "message": create_a2a_text_message(error_message, current_task["contextId"], current_task["id"]),
                            "timestamp": event.timestamp
                        },
                        "final": True
                    }
                else:
                    yield {
                        "kind": "artifact-update",
                        "taskId": current_task["id"],
                        "contextId": current_task["contextId"],
                        "artifact": {
                            "artifactId": f"result_{int(time.time() * 1000)}",
                            "name": "response",
                            "parts": [{"kind": "text", "text": str(event.content)}]
                        }
                    }

                    yield {
                        "kind": "status-update",
                        "taskId": current_task["id"],
                        "contextId": current_task["contextId"],
                        "status": {"state": "completed", "timestamp": event.timestamp},
                        "final": True
                    }
                break

    except Exception as error:
        error_message = str(error)
        yield {
            "kind": "status-update",
            "taskId": current_task["id"],
            "contextId": current_task["contextId"],
            "status": {
                "state": "failed",
                "message": create_a2a_text_message(error_message, current_task["contextId"], current_task["id"]),
                "timestamp": datetime.now().isoformat()
            },
            "final": True
        }
