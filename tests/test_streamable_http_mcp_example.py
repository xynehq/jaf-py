"""
Tests for the streamable HTTP MCP example.
"""

import sys
import os
import asyncio
import json
import pytest
import httpx

# Add project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.streamable_http_mcp_example import app
from jaf.core.streaming import StreamingEventType

@pytest.mark.asyncio
async def test_stream_agent_run():
    """
    Test the /stream endpoint to ensure it streams tool execution events correctly.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        request_body = {"message": "what is 2+2?"}
        
        async with client.stream("POST", "/stream", json=request_body) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            events = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    event_data = json.loads(line[len("data:"):])
                    events.append(event_data)

            # Check for the presence and order of key events
            event_types = [e["type"] for e in events]
            
            assert StreamingEventType.START.value in event_types
            assert StreamingEventType.TOOL_CALL.value in event_types
            assert StreamingEventType.TOOL_RESULT.value in event_types
            assert StreamingEventType.CHUNK.value in event_types
            assert StreamingEventType.COMPLETE.value in event_types

            # Verify tool call details
            tool_call_event = next((e for e in events if e["type"] == StreamingEventType.TOOL_CALL.value), None)
            assert tool_call_event is not None
            assert tool_call_event["data"]["tool_name"] == "math_tool"
            assert tool_call_event["data"]["arguments"] == {"message": "what is 2+2?"}

            # Verify tool result details
            tool_result_event = next((e for e in events if e["type"] == StreamingEventType.TOOL_RESULT.value), None)
            assert tool_result_event is not None
            assert tool_result_event["data"]["tool_name"] == "math_tool"
            assert "42" in tool_result_event["data"]["result"]

            # Verify final content
            final_chunk_event = next((e for e in events if e["type"] == StreamingEventType.COMPLETE.value), None)
            assert final_chunk_event is not None
            assert "The answer is 42." in final_chunk_event["data"]["content"]
