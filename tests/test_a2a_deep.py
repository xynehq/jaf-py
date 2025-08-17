import asyncio
import json
from typing import Dict, Any

import pytest
import respx
from httpx import Response

from jaf.a2a.client import send_message_to_agent
from jaf.a2a import (
    create_a2a_agent,
    create_a2a_client,
    create_a2a_server,
    create_a2a_tool,
    create_server_config,
)
from jaf.a2a.types import A2AAgent
from pydantic import BaseModel, Field

# Define tool schemas
class MathArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class HandoffArgs(BaseModel):
    agent_name: str = Field(description="The name of the agent to handoff to")
    query: str = Field(description="The query to pass to the next agent")

# Define tool functions
async def math_tool(args: MathArgs, context) -> Dict[str, Any]:
    """A simple math tool."""
    result = eval(args.expression)
    return {"result": result}

async def handoff_tool(args: HandoffArgs, context) -> Dict[str, Any]:
    """A tool to handoff to another agent."""
    return {"handoff_to": args.agent_name, "query": args.query}

@pytest.mark.asyncio
@respx.mock
async def test_a2a_deep_handoff():
    """A deep test of the A2A handoff functionality."""
    # 1. Create Agents
    math_tool_obj = create_a2a_tool("math", "Performs calculations", MathArgs.model_json_schema(), math_tool)
    math_agent = create_a2a_agent("MathAgent", "A mathematical agent", "You are a math agent.", [math_tool_obj])

    handoff_tool_obj = create_a2a_tool("handoff", "Handoffs to another agent", HandoffArgs.model_json_schema(), handoff_tool)
    triage_agent = create_a2a_agent("TriageAgent", "A triage agent", "You are a triage agent.", [handoff_tool_obj])

    # 2. Create Server
    server_config = create_server_config(
        agents={"MathAgent": math_agent, "TriageAgent": triage_agent},
        name="Test A2A Server",
        description="A server for deep A2A testing",
        port=8080,
    )
    server = create_a2a_server(server_config)

    # 3. Mock the server responses
    respx.post("http://localhost:8080/a2a/agents/TriageAgent").mock(return_value=Response(200, json={
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "handoff",
                    "arguments": json.dumps({"agent_name": "MathAgent", "query": "What is 2 + 2?"})
                }
            }]
        }
    }))
    respx.post("http://localhost:8080/a2a/agents/MathAgent").mock(side_effect=[
        Response(200, json={
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tool_calls": [{
                    "id": "call_2",
                    "function": {
                        "name": "math",
                        "arguments": json.dumps({"expression": "2 + 2"})
                    }
                }]
            }
        }),
        Response(200, json={
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"content": "The answer is 4."}
        }),
    ])

    # 4. Create Client and run the scenario
    client = create_a2a_client("http://localhost:8080")
    
    # This is a simplified simulation of the client-side logic
    # A real scenario would involve a more complex state machine
    
    # Initial request to TriageAgent
    print("\n--- Sending initial request to TriageAgent ---")
    triage_response = await send_message_to_agent(client, "TriageAgent", "Calculate 2 + 2")
    print(f"TriageAgent response: {triage_response}")
    assert "tool_calls" in triage_response
    
    # Handoff to MathAgent
    print("\n--- Handing off to MathAgent ---")
    math_response = await send_message_to_agent(client, "MathAgent", "What is 2 + 2?")
    print(f"MathAgent response: {math_response}")
    assert "tool_calls" in math_response
    
    # Get final response
    print("\n--- Getting final response from MathAgent ---")
    final_response = await send_message_to_agent(client, "MathAgent", "What is 2 + 2?")
    print(f"Final response: {final_response}")
    assert "The answer is 4." in final_response
