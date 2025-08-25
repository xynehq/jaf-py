"""
An example demonstrating streamable HTTP responses with MCP, agents, and tools.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


from jaf.core.types import (
    Agent,
    Message,
    RunConfig,
    RunState,
    Tool,
    ToolSchema,
    ContentRole,
    create_run_id,
    create_trace_id,
    ModelConfig
)
from jaf.core.streaming import run_streaming, create_sse_response

# --- Mock Classes (copied from tests/test_engine.py) ---

class MockModelProvider:
    """Mock model provider for testing."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = responses
        self.call_count = 0

    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        """Return a mock completion response."""
        if self.call_count >= len(self.responses):
            response = {
                'message': {
                    'content': 'Default response',
                    'tool_calls': None
                }
            }
        else:
            response = self.responses[self.call_count]

        self.call_count += 1
        return response


class SimpleToolArgs(BaseModel):
    """Simple tool arguments for testing."""
    message: str


class SimpleTool:
    """Simple tool for testing."""

    def __init__(self, name: str = "simple_tool", result: str = "tool result"):
        self.name = name
        self.result = result
        self._schema = ToolSchema(
            name=name,
            description="A simple test tool",
            parameters=SimpleToolArgs
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, args: SimpleToolArgs, context: Any) -> str:
        return f"{self.result}: {args.message}"


class WeatherToolArgs(BaseModel):
    """Arguments for the weather tool."""
    city: str

class WeatherTool(SimpleTool):
    """A tool to get the weather."""
    def __init__(self):
        super().__init__(name="weather_tool", result="The weather is sunny.")
        self._schema = ToolSchema(
            name="weather_tool",
            description="Get the weather for a city.",
            parameters=WeatherToolArgs
        )

    async def execute(self, args: WeatherToolArgs, context: Any) -> str:
        return f"The weather in {args.city} is sunny."

# --- FastAPI App ---
app = FastAPI()

# --- Agent and Tool Setup ---
math_tool = SimpleTool(name="math_tool", result="42")
weather_tool = WeatherTool()
agent = Agent(
    name="math_agent",
    instructions=lambda s: "You are a math and weather agent.",
    tools=[math_tool, weather_tool],
    model_config=ModelConfig(name="mock-model")
)

agent_registry = {"math_agent": agent}

# --- Mock Model Provider ---
model_provider = MockModelProvider([
    {
        'message': {
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_math_123',
                    'type': 'function',
                    'function': {
                        'name': 'math_tool',
                        'arguments': '{"message": "what is 2+2?"}'
                    }
                }
            ]
        }
    },
    {
        'message': {
            'content': 'The answer is 42.',
            'tool_calls': None
        }
    },
    {
        'message': {
            'content': '',
            'tool_calls': [
                {
                    'id': 'call_weather_123',
                    'type': 'function',
                    'function': {
                        'name': 'weather_tool',
                        'arguments': '{"city": "London"}'
                    }
                }
            ]
        }
    },
    {
        'message': {
            'content': 'The weather in London is sunny.',
            'tool_calls': None
        }
    }
])

# --- Run Configuration ---
run_config = RunConfig(
    agent_registry=agent_registry,
    model_provider=model_provider
)

class RunRequest(BaseModel):
    message: str
    agent_name: str = "math_agent"

@app.get("/tools/{agent_name}")
async def get_agent_tools(agent_name: str):
    """
    Get the list of tools for a given agent.
    """
    agent = agent_registry.get(agent_name)
    if not agent:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Agent not found")
    
    tools_list = []
    for tool in agent.tools:
        tools_list.append({
            "name": tool.schema.name,
            "description": tool.schema.description,
            "parameters": tool.schema.parameters.schema()
        })
    return {"tools": tools_list}

@app.post("/stream")
async def stream_agent_run(request: RunRequest):
    """
    Run an agent and stream the results back as an HTTP response.
    """
    initial_state = RunState(
        run_id=create_run_id("test-stream"),
        trace_id=create_trace_id("test-stream"),
        messages=[Message(role=ContentRole.USER, content=request.message)],
        current_agent_name=request.agent_name,
        context={},
        turn_count=0
    )

    stream = run_streaming(initial_state, run_config)

    async def event_generator():
        async for event in stream:
            yield create_sse_response(event)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
