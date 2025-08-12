"""
Pure functional A2A agent creation utilities
No classes, only pure functions and immutable data
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Union
from datetime import datetime

from ..core.types import Agent, Tool, Message, RunState, RunConfig
from ..core.engine import run
from .types import (
    A2AAgent, A2AAgentTool, ToolContext, A2AToolResult, 
    AgentState, StreamEvent, A2AMessage, A2ATask, A2AArtifact,
    TaskState, A2ATaskStatus, create_a2a_text_part, create_a2a_data_part
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
        session_id=session_id,
        messages=[],
        context={},
        artifacts=[],
        timestamp=datetime.now().isoformat()
    )


def add_message_to_state(state: AgentState, message: Any) -> AgentState:
    """Pure function to add message to state"""
    return AgentState(
        session_id=state.session_id,
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
        session_id=state.session_id,
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
    
    # Transform tools
    jaf_tools = []
    for a2a_tool in a2a_agent.tools:
        jaf_tool = transform_a2a_tool_to_jaf(a2a_tool)
        jaf_tools.append(jaf_tool)
    
    return Agent(
        name=a2a_agent.name,
        instructions=instructions_func,
        tools=jaf_tools
    )


def transform_a2a_tool_to_jaf(a2a_tool: A2AAgentTool) -> Tool:
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
    
    return Tool(
        schema={
            "name": a2a_tool.name,
            "description": a2a_tool.description,
            "parameters": a2a_tool.parameters
        },
        execute=execute_wrapper
    )


def create_run_config_for_a2a_agent(
    a2a_agent: A2AAgent,
    model_provider: Any
) -> RunConfig:
    """Pure function to create run configuration for A2A agent"""
    jaf_agent = transform_a2a_agent_to_jaf(a2a_agent)
    
    def event_handler(event: Any) -> None:
        print(f"[A2A:{a2a_agent.name}] {event.get('type', 'unknown')}: {event.get('data', '')}")
    
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
    
    return RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=state.messages,
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
    """Pure async generator function to process agent query"""
    # Transform query to JAF message format
    user_message = create_user_message(query)
    new_state = add_message_to_state(state, user_message)
    
    # Create JAF configuration
    run_config = create_run_config_for_a2a_agent(agent, model_provider)
    run_state = transform_to_run_state(new_state, agent.name)
    
    try:
        # Execute JAF engine (pure function)
        result = await run(run_state, run_config)
        
        if hasattr(result.outcome, 'status') and result.outcome.status == 'completed':
            final_state = update_state_from_run_result(new_state, result.outcome)
            yield StreamEvent(
                is_task_complete=True,
                content=getattr(result.outcome, 'output', 'Task completed'),
                new_state=final_state.model_dump(),
                timestamp=datetime.now().isoformat()
            )
        else:
            final_state = update_state_from_run_result(new_state, result.outcome)
            error_content = getattr(result.outcome, 'error', 'Unknown error')
            yield StreamEvent(
                is_task_complete=True,
                content=f"Error: {json.dumps(error_content) if isinstance(error_content, dict) else str(error_content)}",
                new_state=final_state.model_dump(),
                timestamp=datetime.now().isoformat()
            )
    except Exception as error:
        yield StreamEvent(
            is_task_complete=True,
            content=f"Error: {str(error)}",
            new_state=new_state.model_dump(),
            timestamp=datetime.now().isoformat()
        )


def extract_text_from_a2a_message(message: Dict[str, Any]) -> str:
    """Pure function to extract text from A2A message"""
    if not message or not message.get("parts"):
        return ""
    
    text_parts = []
    for part in message["parts"]:
        if part.get("kind") == "text":
            text_parts.append(part.get("text", ""))
    
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
            processing_events.append(event.model_dump())
            
            if event.is_task_complete:
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
        error_message = str(error)
        failed_task = update_a2a_task_status(
            current_task,
            "failed",
            create_a2a_text_message(error_message, session_id, current_task["id"])
        )
        
        return {
            "events": events,
            "final_task": failed_task,
            "error": error_message
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
            if not event.is_task_complete:
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